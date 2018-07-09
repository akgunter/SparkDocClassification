package net.ddns.akgunter.spark_learning.document_classifier

import java.nio.file.Paths

import net.ddns.akgunter.spark_learning.util.FileUtil._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, ChiSqSelector, IDF, PCA}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

import org.nd4j.linalg.factory.Nd4j

import net.ddns.akgunter.spark_learning.sparkml_processing._
import net.ddns.akgunter.spark_learning.spark.CanSpark
import net.ddns.akgunter.spark_learning.sparkml_processing.{CommonElementFilter, WordCountToVec}
import net.ddns.akgunter.spark_learning.util.FileUtil._

object RunClassifier extends CanSpark {

  def loadData(trainingDir: String, validationDir: String)(implicit spark: SparkSession):
  (DataFrame, DataFrame, String, String, Integer, Integer) = {

    val commonElementFilter = new CommonElementFilter()
      .setDropFreq(0.1)
    val wordVectorizer = new WordCountToVec()
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
      .setSelectorType("fpr")
      .setFpr(0.0001)
    val pca = new PCA()
      .setInputCol("chi_sel_features")
      .setK(8000)
      .setOutputCol("pca_features")

    val preprocPipeline = new Pipeline()
      .setStages(Array(commonElementFilter, wordVectorizer, binarizer, idf, chiSel, pca))

    logger.info("Loading data...")
    val trainingData = dataFrameFromDirectory(trainingDir, isTraining = true)
    val validationData = dataFrameFromDirectory(validationDir, isTraining = true)

    logger.info("Fitting preprocessing pipeline...")
    val dataModel = preprocPipeline.fit(trainingData)

    logger.info("Preprocessing data...")
    val trainingDataProcessed = dataModel.transform(trainingData)
    val validationDataProcessed = dataModel.transform(validationData)

    val lastStage = preprocPipeline.getStages.last
    val featuresColParam = lastStage.getParam("outputCol")
    val featuresCol = lastStage.getOrDefault(featuresColParam).asInstanceOf[String]

    val numFeatures = trainingData.head.getAs[SparseVector](featuresCol).size
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

  def runDL4J(numFeatures: Int,
              numClasses: Int)(implicit spark: SparkSession): Unit = {
    val numTrainingSamples = 100

    val sparseFactory = Nd4j.sparseFactory()
    println(sparseFactory.zeros(numTrainingSamples, numClasses))
  }

  def runML(dataDir: String, useDL4J: Boolean)(implicit spark: SparkSession): Unit = {
    if (useDL4J) {

      runDL4J(20, 5)
    }
    else {
      val trainingDir = Paths.get(dataDir, "Training").toString
      val validationDir = Paths.get(dataDir, "Validation").toString
      val (trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses) = loadData(trainingDir, validationDir)

      runSparkML(trainingData, validationData, featuresCol, labelCol, numFeatures, numClasses)
    }
  }

  def main(args: Array[String]): Unit = {
    val dataDir = args(0)

    println(s"Running with dataDir=$dataDir")
    withSpark() { spark => runML(dataDir, useDL4J = true)(spark) }
  }
}