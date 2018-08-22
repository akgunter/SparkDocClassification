package net.ddns.akgunter.spark_doc_classification.implementations.preprocessing

import org.apache.spark.sql.SparkSession

import net.ddns.akgunter.spark_doc_classification.spark.CanLog


trait PreprocessingImplementation extends CanLog {
  def run(trainingDir: String, validationDir: String, outputDataDir: String)(implicit spark: SparkSession): Unit
}
