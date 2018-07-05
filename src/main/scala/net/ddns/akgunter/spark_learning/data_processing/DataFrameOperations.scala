package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.sql.{DataFrame, Row}

object DataFrameOperations {

  def dropCommonWords(df: DataFrame, dropRatio: Double): DataFrame = {
    val numDocs = df.select("input_file").distinct.count

    val keptWords = df.groupBy("word")
      .count
      .filter {
        row => row.getAs[Long]("count") < (1 - dropRatio) * numDocs
      }
      .select("word")

    df.join(keptWords, "word")
  }
}