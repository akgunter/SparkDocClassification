package net.ddns.akgunter.spark_doc_classification.spark

/*
CanSpark trait authored by Wil Adamec (https://github.com/wiladamec) and used with permission.
 */

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession


trait CanSpark extends CanLog {

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