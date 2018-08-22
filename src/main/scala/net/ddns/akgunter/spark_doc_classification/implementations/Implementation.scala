package net.ddns.akgunter.spark_doc_classification.implementations

import org.apache.spark.sql.SparkSession

trait Implementation {
  def run(trainingDir: String, validationDir: String, numEpochs: Int)(implicit spark: SparkSession): Unit
}