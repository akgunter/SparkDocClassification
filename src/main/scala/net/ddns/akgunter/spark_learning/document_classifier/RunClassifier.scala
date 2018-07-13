package net.ddns.akgunter.spark_learning.document_classifier

import java.nio.file.Paths
import java.util.{ArrayList => JavaArrayList}

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, ChiSqSelector, IDF, VectorSlicer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._


import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import net.ddns.akgunter.spark_learning.spark.CanSpark
import net.ddns.akgunter.spark_learning.sparkml_processing.{CommonElementFilter, WordCountToVec, WordCountToVecModel}
import net.ddns.akgunter.spark_learning.util.FileUtil._
import net.ddns.akgunter.spark_learning.util.DataFrameUtil._


object RunMode extends Enumeration {
  val PREPROCESS, SPARKML, DL4J, DL4JSPARK = Value
}

object RunClassifier extends CanSpark {
  def runPreprocess(inputDataDir: String, outputDataDir: String)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(inputDataDir, TrainingDirName).toString
    val validationDir = Paths.get(inputDataDir, ValidationDirName).toString

    val commonElementFilter = new CommonElementFilter()
      .setDropFreq(0.1)
    val wordVectorizer = new WordCountToVec()

    val rawWordVectorColParam = wordVectorizer.getParam("vectorCol")
    val rawWordVectorCol = wordVectorizer.getOrDefault(rawWordVectorColParam).asInstanceOf[String]
    val vectorSlicer = new VectorSlicer()
      .setInputCol(rawWordVectorCol)
      .setOutputCol("sliced_vector")
      .setIndices((0 until 10).toArray)
    val binarizer = new Binarizer()
      .setThreshold(0.0)
      .setInputCol("raw_word_vector")
      .setOutputCol("binarized_word_vector")
    val idf = new IDF()
      .setInputCol("binarized_word_vector")
      .setOutputCol("tfidf_vector")
      .setMinDocFreq(2)
    val chiSel = new ChiSqSelector()
      .setFeaturesCol("tfidf_vector")
      .setLabelCol("label")
      .setOutputCol("chi_sel_features")
      .setNumTopFeatures(8000)

    val preprocStages = Array(commonElementFilter, wordVectorizer, binarizer, idf, chiSel)
    val preprocPipeline = new Pipeline().setStages(preprocStages)


    logger.info("Loading data...")
    val trainingData = dataFrameFromRawDirectory(trainingDir, isLabelled = true)
    val validationData = dataFrameFromRawDirectory(validationDir, isLabelled = true)

    logger.info("Fitting preprocessing pipeline...")
    val preprocModel = preprocPipeline.fit(trainingData)


    logger.info("Preprocessing data...")
    val trainingDataSparse = preprocModel.transform(trainingData)
    val validationDataSparse = preprocModel.transform(validationData)


    val lastStage = preprocPipeline.getStages.last
    val pipeFeaturesColParam = lastStage.getParam("outputCol")
    val pipeFeaturesCol = lastStage.getOrDefault(pipeFeaturesColParam).asInstanceOf[String]
    val pipeLabelColParam = wordVectorizer.getParam("labelCol")
    val pipeLabelCol = wordVectorizer.getOrDefault(pipeLabelColParam).asInstanceOf[String]


    val trainingDataFilePath = Paths.get(outputDataDir, TrainingDirName).toString
    val validationDataFilePath = Paths.get(outputDataDir, ValidationDirName).toString

    logger.info("Writing training data to CSV...")
    val trainingDataCSVReady = sparseDFToCSVReadyDF(trainingDataSparse, pipeFeaturesCol, pipeLabelCol)
    trainingDataCSVReady.write
      .mode("overwrite")
      .csv(trainingDataFilePath)

    logger.info("Writing validation data to CSV...")
    val validationDataCSVReady = sparseDFToCSVReadyDF(validationDataSparse, pipeFeaturesCol, pipeLabelCol)
    validationDataCSVReady.write
    .mode("overwrite")
    .csv(validationDataFilePath)
  }

  def runSparkML(inputDataDir: String)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(inputDataDir, TrainingDirName).toString
    val validationDir = Paths.get(inputDataDir, ValidationDirName).toString

    logger.info("Loading data files...")
    val trainingDataCSVSourced = dataFrameFromProcessedDirectory(trainingDir)
    val validationDataCSVSourced = dataFrameFromProcessedDirectory(validationDir)


    logger.info("Creating data sets...")
    val trainingData = sparseDFFromCSVReadyDF(trainingDataCSVSourced)
    val validationData = sparseDFFromCSVReadyDF(validationDataCSVSourced)


    val Array(csvNumFeaturesCol, _, _, csvLabelCol) = trainingDataCSVSourced.columns
    val numFeatures = trainingDataCSVSourced.head.getAs[Int](csvNumFeaturesCol)
    val numClasses = trainingData.select(csvLabelCol).distinct.count.toInt

    logger.info(s"Configuring neural net with $numFeatures features and $numClasses classes...")
    val Array(sparseFeaturesCol, sparseLabelsCol) = SchemaForCoreDataFrames.fieldNames
    val mlpc = new MultilayerPerceptronClassifier()
      .setLayers(Array(numFeatures, numClasses))
      .setMaxIter(100)
      //.setBlockSize(20)
      .setFeaturesCol(sparseFeaturesCol)
      .setLabelCol(sparseLabelsCol)


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

  def runDL4J(inputDataDir: String): Unit = {
    val trainingDir = Paths.get(inputDataDir, TrainingDirName).toString
    val validationDir = Paths.get(inputDataDir, ValidationDirName).toString


  }

  def runDL4JSpark(inputDataDir: String)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(inputDataDir, TrainingDirName).toString
    val validationDir = Paths.get(inputDataDir, ValidationDirName).toString

    logger.info("Loading data files...")
    val trainingDataCSVSourced = dataFrameFromProcessedDirectory(trainingDir)
    val validationDataCSVSourced = dataFrameFromProcessedDirectory(validationDir)


    logger.info("Creating data sets...")
    val trainingData = sparseDFFromCSVReadyDF(trainingDataCSVSourced)
    val validationData = sparseDFFromCSVReadyDF(validationDataCSVSourced)

    val Array(csvNumFeaturesCol, _, _, csvLabelCol) = trainingDataCSVSourced.columns
    val Array(sparseFeaturesCol, sparseLabelsCol) = SchemaForCoreDataFrames.fieldNames
    val numFeatures = trainingDataCSVSourced.head.getAs[Int](csvNumFeaturesCol)
    val numClasses = trainingData.select(csvLabelCol).distinct.count.toInt

    val trainingRDD = trainingData.rdd.map {
      row =>
        val sparseVector = row.getAs[SparseVector](sparseFeaturesCol).toArray
        val label = row.getAs[Int](sparseLabelsCol)
        val fvec = Nd4j.create(sparseVector)
        val lvec = Nd4j.zeros(numClasses)
        lvec.putScalar(label, 1)
        new DataSet(fvec, lvec)
    }.toJavaRDD
    val validationRDD = validationData.rdd.map {
      row =>
        val sparseVector = row.getAs[SparseVector](sparseFeaturesCol).toArray
        val label = row.getAs[Int](sparseLabelsCol)
        val fvec = Nd4j.create(sparseVector)
        val lvec = Nd4j.zeros(numClasses)
        lvec.putScalar(label, 1)
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
        .activation(Activation.SOFTMAX).nIn(numClasses).nOut(numClasses).build)
      .pretrain(false)
      .backprop(true)
      .build

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(1)
      .rddTrainingApproach(RDDTrainingApproach.Export)
      .exportDirectory("/tmp/alex-spark/SparkLearning")
      .build

    val sparkNet = new SparkDl4jMultiLayer(spark.sparkContext, nnConf, trainingMaster)

    logger.info("Training neural network...")
    val trainedNet = sparkNet.fit(trainingRDD)

    val trainingDataSet = DataSet.merge(new JavaArrayList(trainingRDD.collect))
    val validationDataSet = DataSet.merge(new JavaArrayList[DataSet](validationRDD.collect))

    logger.info("Evaluating training performance...")
    val eval = new Evaluation(numClasses)
    eval.eval(trainingDataSet.getLabels, trainingDataSet.getFeatureMatrix, trainedNet)
    logger.info(eval.stats())

    logger.info("Evaluating validation performance...")
    eval.eval(validationDataSet.getLabels, validationDataSet.getFeatureMatrix, trainedNet)
    logger.info(eval.stats())
  }

  def main(args: Array[String]): Unit = {
    val runMode = RunMode.withName(args(0).toUpperCase)
    val inputDataDir = args(1)
    val outputDataDir = runMode match {
      case RunMode.PREPROCESS => args(2)
      case _ => ""
    }

    runMode match {
      case RunMode.PREPROCESS =>
        println(s"Running with runMode=${runMode.toString}, inputDataDir=$inputDataDir, and outputDataDir=$outputDataDir")
      case _ =>
        println(s"Running with runMode=${runMode.toString} and inputDataDir=$inputDataDir")
    }

    runMode match {
      case RunMode.PREPROCESS =>
        withSpark() { spark => runPreprocess(inputDataDir, outputDataDir)(spark) }
      case RunMode.SPARKML =>
        withSpark() { spark => runSparkML(inputDataDir)(spark) }
      case RunMode.DL4J =>
        runDL4J(inputDataDir)
      case RunMode.DL4JSPARK =>
        withSpark() { spark => runDL4JSpark(inputDataDir)(spark) }
    }
  }
}