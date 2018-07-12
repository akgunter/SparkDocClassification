package net.ddns.akgunter.spark_learning.document_classifier

import java.nio.file.Paths
import java.util.{ArrayList => JavaArrayList}

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, ChiSqSelector, IDF, VectorSlicer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf, struct}

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
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration
import org.nd4j.parameterserver.distributed.enums.ExecutionMode

import net.ddns.akgunter.spark_learning.spark.CanSpark
import net.ddns.akgunter.spark_learning.sparkml_processing.{CommonElementFilter, WordCountToVec, WordCountToVecModel}
import net.ddns.akgunter.spark_learning.util.FileUtil._



object RunMode extends Enumeration {
  val PREPROCESS, SPARKML, DL4J, DL4JSPARK = Value
}

object RunClassifier extends CanSpark {
  def runPreprocess(inputDataDir: String, outputDataDir: String)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(inputDataDir, TRAINING_DIRNAME).toString
    val validationDir = Paths.get(inputDataDir, VALIDATION_DIRNAME).toString

    val commonElementFilter = new CommonElementFilter()
      .setDropFreq(0.1)
    val wordVectorizer = new WordCountToVec()
    val vectorSlicer = new VectorSlicer()
      .setInputCol("raw_word_vector")
      .setOutputCol("sliced_vector")
      .setIndices((0 until 10).toArray)
    val binarizer = new Binarizer()
      .setThreshold(0.0)
      .setInputCol("raw_word_vector")
      //.setInputCol("sliced_vector")
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
      //.setSelectorType("fdr")
      //.setFdr(0.005)
      //.setSelectorType("fpr")
      //.setFpr(0.00001)

    val preprocStages = Array(commonElementFilter, wordVectorizer, vectorSlicer)
    val preprocPipeline = new Pipeline().setStages(preprocStages)

    logger.info("Loading data...")
    val trainingData = dataFrameFromRawDirectory(trainingDir, isLabelled = true)
    val validationData = dataFrameFromRawDirectory(validationDir, isLabelled = true)

    logger.info("Fitting preprocessing pipeline...")
    val preprocModel = preprocPipeline.fit(trainingData)

    logger.info("Preprocessing data...")
    val trainingDataProcessed = preprocModel.transform(trainingData)
    val validationDataProcessed = preprocModel.transform(validationData)

    trainingDataProcessed.show(10, truncate = false)

    val lastStage = preprocPipeline.getStages.last
    val featuresColParam = lastStage.getParam("outputCol")
    val featuresCol = lastStage.getOrDefault(featuresColParam).asInstanceOf[String]
    val labelColParam = wordVectorizer.getParam("labelCol")
    val labelCol = wordVectorizer.getOrDefault(labelColParam).asInstanceOf[String]

    val dictionaryFilePath = Paths.get(outputDataDir, DICTIONARY_DIRNAME).toString
    val trainingDataFilePath = Paths.get(outputDataDir, TRAINING_DIRNAME).toString
    val validationDataFilePath = Paths.get(outputDataDir, VALIDATION_DIRNAME).toString
    val schemaFilePath = Paths.get(outputDataDir, SCHEMA_DIRNAME).toString

    logger.info("Writing dictionary to CSV...")
    val wordVectorizerModel = preprocModel.stages(preprocStages.indexOf(wordVectorizer)).asInstanceOf[WordCountToVecModel]
    val dictionary = wordVectorizerModel.getDictionary
    dictionary.write
      .mode("overwrite")
      .csv(dictionaryFilePath)

    val getSparseIndices = udf {
      v: SparseVector =>
        v.indices.mkString(",")
    }
    val getSparseValues = udf {
      v: SparseVector =>
        v.values.mkString(",")
    }

    val wordIndicesCol = "word_indices_string"
    val wordCountsCol = "word_counts_string"

    logger.info("Writing training data to CSV...")
    val trainingDataToWrite = trainingDataProcessed.select(
      getSparseIndices(col(featuresCol)) as wordIndicesCol,
      getSparseValues(col(featuresCol)) as wordCountsCol,
      col(labelCol)
    )
    trainingDataToWrite.write
      .mode("overwrite")
      .csv(trainingDataFilePath)

    logger.info("Writing validation data to CSV...")
    val validationDataToWrite = validationDataProcessed.select(
      getSparseIndices(col(featuresCol)) as wordIndicesCol,
      getSparseValues(col(featuresCol)) as wordCountsCol,
      col(labelCol)
    )
    validationDataToWrite.write
    .mode("overwrite")
    .csv(validationDataFilePath)

    logger.info("Writing schemas to CSV...")
    import spark.implicits._
    Seq(
      dictionaryFilePath -> dictionary.schema.json,
      trainingDataFilePath -> trainingDataToWrite.schema.json,
      validationDataFilePath -> validationDataToWrite.schema.json
    ).toDF(SCHEMA_DATAPATH_COLUMN, SCHEMA_DATASCHEMA_COLUMN)
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(schemaFilePath)
  }

  def runSparkML(inputDataDir: String)(implicit spark: SparkSession): Unit = {
    val schemaDir = Paths.get(inputDataDir, SCHEMA_DIRNAME).toString
    val dictionaryDir = Paths.get(inputDataDir, DICTIONARY_DIRNAME).toString
    val trainingDir = Paths.get(inputDataDir, TRAINING_DIRNAME).toString
    val validationDir = Paths.get(inputDataDir, VALIDATION_DIRNAME).toString

    logger.info("Loading data files...")
    val dictionaryData = dataFrameFromProcessedDirectory(dictionaryDir, schemaDir)
    val trainingDataProcessed = dataFrameFromProcessedDirectory(trainingDir, schemaDir)
    val validationDataProcessed = dataFrameFromProcessedDirectory(validationDir, schemaDir)

    val Array(wordIndicesCol, wordCountsCol, labelCol) = trainingDataProcessed.columns
    val featuresCol = "word_vector"
    val numFeatures = dictionaryData.count.toInt

    logger.info("Creating data sets...")
    val createSparseColumn = udf {
      (wordIndicesStr: String, wordCountsStr: String) =>
        val wordIndices = wordIndicesStr.split(",").map(_.toInt)
        val wordCounts = wordCountsStr.split(",").map(_.toDouble)
        new SparseVector(numFeatures, wordIndices, wordCounts)
    }
    val trainingData = trainingDataProcessed.select(
      createSparseColumn(col(wordIndicesCol), col(wordCountsCol)) as featuresCol,
      col(labelCol)
    )
    val validationData = validationDataProcessed.select(
      createSparseColumn(col(wordIndicesCol), col(wordCountsCol)) as featuresCol,
      col(labelCol)
    )

    val numClasses = trainingData.select(labelCol).distinct.count.toInt

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

  def runDL4J(): Unit = {

  }

  def runDL4JSpark()(implicit spark: SparkSession): Unit = {

  }

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
        .setInputCol("raw_word_vector")
        //.setInputCol("sliced_vector")
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
        .setStages(Array(commonElementFilter, wordVectorizer, binarizer, idf, chiSel))

      logger.info("Loading data...")
      val trainingData = dataFrameFromRawDirectory(trainingDir, isLabelled = true)
      val validationData = dataFrameFromRawDirectory(validationDir, isLabelled = true)

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

  def runDL4JOld(trainingData: DataFrame,
                 validationData: DataFrame,
                 featuresCol: String,
                 labelCol: String,
                 numFeatures: Int,
                 numClasses: Int)(implicit spark: SparkSession): Unit = {

    val trainingRDD = trainingData.rdd.map {
      row =>
        val sparseVector = row.getAs[SparseVector](featuresCol).toArray
        val label = row.getAs[Int](labelCol)
        val fvec = Nd4j.create(sparseVector)
        val lvec = Nd4j.zeros(numClasses)
        lvec.putScalar(label, 1)
        new DataSet(fvec, lvec)
    }.toJavaRDD
    val validationRDD = validationData.rdd.map {
      row =>
        val sparseVector = row.getAs[SparseVector](featuresCol).toArray
        val label = row.getAs[Int](labelCol)
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

    val voidConfig = VoidConfiguration.builder()
      .unicastPort(4050)
      .networkMask("10.0.0.0/24")
      .controllerAddress("127.0.0.1")
      .executionMode(ExecutionMode.AVERAGING)
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

  def runML(dataDir: String, useDL4J: Boolean)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(dataDir, "Training").toString
    val validationDir = Paths.get(dataDir, "Validation").toString
    val (trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses) = loadData(trainingDir, validationDir)

    if (useDL4J) runDL4JOld(trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses)
    else runSparkML(trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses)
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
      case RunMode.DL4J => return
      case RunMode.DL4JSPARK => return
    }
  }
}