package net.ddns.akgunter.spark_doc_classification.implementations.classification

import net.ddns.akgunter.spark_doc_classification.RunClassifier.logger
import net.ddns.akgunter.spark_doc_classification.util.DataFrameUtil.{SchemaForSparseDataFrames, sparseDFFromCSVReadyDF}
import net.ddns.akgunter.spark_doc_classification.util.FileUtil.dataFrameFromProcessedDirectory
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

/*
Perform classification with a 2-layer Spark MLlib neural network.
*/
object RunSparkMLlib extends ClassificationImplementation {
  override def run(trainingDir: String, validationDir: String, numEpochs: Int)(implicit spark: SparkSession): Unit = {
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
    val Array(sparseFeaturesCol, sparseLabelsCol) = SchemaForSparseDataFrames.fieldNames
    val mlpc = new MultilayerPerceptronClassifier()
      .setLayers(Array(numFeatures, numClasses))
      .setMaxIter(numEpochs)
      //.setBlockSize(20)
      .setFeaturesCol(sparseFeaturesCol)
      .setLabelCol(sparseLabelsCol)


    logger.info("Training neural network...")
    val mlpcModel = mlpc.fit(trainingData)


    logger.info("Calculating predictions...")
    val trainingPredictions = mlpcModel.transform(trainingData)
    val validationPredictions = mlpcModel.transform(validationData)

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
    val recallEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedRecall")
    val f1Evaluator = new  MulticlassClassificationEvaluator()
      .setMetricName("f1")

    logger.info(s"Training accuracy: ${accuracyEvaluator.evaluate(trainingPredictions)}")
    logger.info(s"Training precision: ${precisionEvaluator.evaluate(trainingPredictions)}")
    logger.info(s"Training recall: ${recallEvaluator.evaluate(trainingPredictions)}")
    logger.info(s"Training F1: ${f1Evaluator.evaluate(trainingPredictions)}")

    logger.info(s"Validation accuracy: ${accuracyEvaluator.evaluate(validationPredictions)}")
    logger.info(s"Validation precision: ${precisionEvaluator.evaluate(validationPredictions)}")
    logger.info(s"Validation recall: ${recallEvaluator.evaluate(validationPredictions)}")
    logger.info(s"Validation F1: ${f1Evaluator.evaluate(validationPredictions)}")
  }
}
