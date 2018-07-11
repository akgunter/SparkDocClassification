package net.ddns.akgunter.spark_learning.document_classifier

import java.nio.file.Paths

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, ChiSqSelector, IDF, VectorSlicer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.spark.parameterserver.training.{CustomTrainingMaster, SharedTrainingMaster}
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration
import net.ddns.akgunter.spark_learning.spark.CanSpark
import net.ddns.akgunter.spark_learning.sparkml_processing.{CommonElementFilter, WordCountToVec}
import net.ddns.akgunter.spark_learning.util.FileUtil._

object RunClassifier extends CanSpark {

  def loadData(trainingDir: String, validationDir: String)(implicit spark: SparkSession):
  (DataFrame, DataFrame, String, String, Integer, Integer) = {

    val commonElementFilter = new CommonElementFilter()
      .setDropFreq(0.1)
    val wordVectorizer = new WordCountToVec()
    val vectorSlicer = new VectorSlicer()
      .setInputCol("raw_word_vector")
      .setOutputCol("sliced_vector")
      .setIndices((0 until 10).toArray)
    val binarizer = new Binarizer()
      .setThreshold(0.0)
      //.setInputCol("raw_word_vector")
      .setInputCol("sliced_vector")
      .setOutputCol("binarized_word_vector")
    val idf = new IDF()
      .setInputCol("binarized_word_vector")
      .setOutputCol("tfidf_vector")
      .setMinDocFreq(2)
    val chiSel = new ChiSqSelector()
      .setFeaturesCol("tfidf_vector")
      .setLabelCol("label")
      .setOutputCol("chi_sel_features")
      .setSelectorType("fdr")
      .setFdr(0.005)
      //.setSelectorType("fpr")
      //.setFpr(0.00001)

    val preprocPipeline = new Pipeline()
      .setStages(Array(commonElementFilter, wordVectorizer, vectorSlicer))

    logger.info("Loading data...")
    val trainingData = dataFrameFromDirectory(trainingDir, isTraining = true)
    val validationData = dataFrameFromDirectory(validationDir, isTraining = true)

    logger.info("Fitting preprocessing pipeline...")
    val preprocModel = preprocPipeline.fit(trainingData)

    logger.info("Preprocessing data...")
    val trainingDataProcessed = preprocModel.transform(trainingData)
    val validationDataProcessed = preprocModel.transform(validationData)

    val lastStage = preprocPipeline.getStages.last
    val featuresColParam = lastStage.getParam("outputCol")
    val featuresCol = lastStage.getOrDefault(featuresColParam).asInstanceOf[String]

    val numFeatures = trainingDataProcessed.head.getAs[SparseVector](featuresCol).size
    val numClasses = getLabelDirectories(trainingDir).length

    (trainingDataProcessed, validationDataProcessed, featuresCol, "label", numFeatures, numClasses)
  }

  def runSparkML(trainingData: DataFrame,
                 validationData: DataFrame,
                 featuresCol: String,
                 labelCol: String,
                 numFeatures: Int,
                 numClasses: Int)(implicit spark: SparkSession): Unit = {

    logger.info(s"Configuring neural net with $numFeatures features and $numClasses classes...")
    val mlpc = new MultilayerPerceptronClassifier()
      .setLayers(Array(numFeatures, numClasses))
      .setMaxIter(100)
      //.setBlockSize(20)
      .setFeaturesCol(featuresCol)

    logger.info("Training neural network...")
    val mlpcModel = mlpc.fit(trainingData)

    logger.info("Calculating predictions...")
    val trainingPredictions = mlpcModel.transform(trainingData)
    val validationPredictions = mlpcModel.transform(validationData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    logger.info(s"Training accuracy: ${evaluator.evaluate(trainingPredictions)}")
    logger.info(s"Validation accuracy: ${evaluator.evaluate(validationPredictions)}")
  }

  def runDL4J(trainingData: DataFrame,
              validationData: DataFrame,
              featuresCol: String,
              labelCol: String,
              numFeatures: Int,
              numClasses: Int)(implicit spark: SparkSession): Unit = {

    val trainingRDD = trainingData.rdd.map {
      row =>
        val fvec = Nd4j.create(row.getAs[SparseVector](featuresCol).toArray)
        val lvec = Nd4j.create(Array(row.getAs[Int](labelCol).toFloat))
        new DataSet(fvec, lvec)
    }.toJavaRDD
    val validationRDD = validationData.rdd.map {
      row =>
        val fvec = Nd4j.create(row.getAs[SparseVector](featuresCol).toArray)
        val lvec = Nd4j.create(Array(row.getAs[Int](labelCol).toFloat))
        new DataSet(fvec, lvec)
    }.toJavaRDD

    logger.info(s"Configuring neural net with $numFeatures features and $numClasses classes...")
    val nnConf = new NeuralNetConfiguration.Builder()
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.02))
      .l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numFeatures).nOut(numClasses).build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                  .activation(Activation.SOFTMAX).nIn(numFeatures).nOut(numClasses).build)
      .pretrain(false)
      .backprop(true)
      .build

    val voidConfig = VoidConfiguration.builder()
      .unicastPort(40123)
      .networkMask("10.0.0.0/24")
      .controllerAddress("127.0.0.1")
      .build

    val tm = new CustomTrainingMaster.Builder(voidConfig, 1)
      .updatesThreshold(1e-3)
      .rddTrainingApproach(RDDTrainingApproach.Direct)
      .batchSizePerWorker(1)
      .workersPerNode(4)
      .build

    val sparkNet = new SparkDl4jMultiLayer(spark.sparkContext, nnConf, tm)

    logger.info("Training neural network...")
    (0 until 5).foreach {
      epoch =>
        logger.info(s"TM Config values BEFORE: ${tm.getBatchSizePerWorker}, ${tm.getNumObjectsEachWorker}, ${tm.getRDDDataSetNumExamples}")
        sparkNet.fit(trainingRDD)
        logger.info(s"TM Config values AFTER: ${tm.getBatchSizePerWorker}, ${tm.getNumObjectsEachWorker}, ${tm.getRDDDataSetNumExamples}")
        logger.info(s"Completed Epoch $epoch")
    }
  }

  def runML(dataDir: String, useDL4J: Boolean)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(dataDir, "Training").toString
    val validationDir = Paths.get(dataDir, "Validation").toString
    val (trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses) = loadData(trainingDir, validationDir)

    if (useDL4J) runDL4J(trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses)
    else runSparkML(trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses)
  }

  def main(args: Array[String]): Unit = {
    val dataDir = args(0)
    val useDL4J = args.length > 1 && args(1) == "dl4j"

    println(s"Running with dataDir=$dataDir and useDL4J=$useDL4J")
    withSpark() { spark => runML(dataDir, useDL4J)(spark) }
  }
}