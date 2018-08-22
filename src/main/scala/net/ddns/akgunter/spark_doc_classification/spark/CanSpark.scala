package net.ddns.akgunter.spark_doc_classification.spark

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import net.ddns.akgunter.spark_doc_classification.util.LogHelper

trait CanSpark extends LogHelper {

  def withSpark[A]()(body: SparkSession => A): A = {

    val sparkConf = new SparkConf()
      .setAppName(this.getClass.getSimpleName.stripSuffix("$"))

    val spark = SparkSession
      .builder
      .config(sparkConf)
      .getOrCreate

    try body(spark)
    finally spark.stop
  }
}