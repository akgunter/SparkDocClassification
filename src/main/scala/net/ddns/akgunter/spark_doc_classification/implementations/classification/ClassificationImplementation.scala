package net.ddns.akgunter.spark_doc_classification.implementations.classification

import org.apache.spark.sql.SparkSession

import net.ddns.akgunter.spark_doc_classification.spark.CanLog

trait ClassificationImplementation extends CanLog {
  def run(trainingDir: String, validationDir: String, numEpochs: Int)(implicit spark: SparkSession): Unit
}
