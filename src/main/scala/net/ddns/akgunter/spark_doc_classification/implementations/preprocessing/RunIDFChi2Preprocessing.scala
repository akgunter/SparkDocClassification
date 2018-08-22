package net.ddns.akgunter.spark_doc_classification.implementations.preprocessing
import java.nio.file.Paths

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Binarizer, ChiSqSelector, IDF}
import org.apache.spark.sql.SparkSession

import net.ddns.akgunter.spark_doc_classification.lib.pipeline_stages.{CommonElementFilter, WordCountToVec}
import net.ddns.akgunter.spark_doc_classification.util.DataFrameUtil.sparseDFToCSVReadyDF
import net.ddns.akgunter.spark_doc_classification.util.FileUtil.{TrainingDirName, ValidationDirName, dataFrameFromRawDirectory}


object RunIDFChi2Preprocessing extends PreprocessingImplementation {
  override def run(trainingDir: String, validationDir: String, outputDataDir: String)(implicit spark: SparkSession): Unit = {
    // Instantiate the operations
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
      .setNumTopFeatures(8000)

    // Construct the pipeline
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


    // Get important columns programmatically, in case of reconfiguration
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
}
