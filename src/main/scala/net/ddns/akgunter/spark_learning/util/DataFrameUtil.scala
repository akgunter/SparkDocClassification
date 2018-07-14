package net.ddns.akgunter.spark_learning.util

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}

object DataFrameUtil {
  val SchemaForRawDataFiles: StructType = new StructType()
    .add("word", StringType, nullable = false)
    .add("word_count", IntegerType, nullable = false)

  val SchemaForProcDataFiles: StructType = new StructType()
    .add("num_features", IntegerType, nullable = false)
    .add("word_indices_str", StringType, nullable = false)
    .add("word_counts_str", StringType, nullable = false)
    .add("label", IntegerType, nullable = true)

  val SchemaForSparseDataFrames: StructType = new StructType()
    .add("word_vector", VectorType, nullable = false)
    .add("label", IntegerType, nullable = true)


  def sparseDFToCSVReadyDF(sparseDF: DataFrame, sparseFeaturesCol: String, sparseLabelsCol: String): DataFrame = {
    val getSparseIndices = udf {
      v: SparseVector =>
        v.indices.mkString(",")
    }
    val getSparseValues = udf {
      v: SparseVector =>
        v.values.mkString(",")
    }

    val Array(csvNumFeaturesCol, csvWordIndicesStrCol, csvWordCountsStrCol, csvLabelCol) = SchemaForProcDataFiles.fieldNames
    val numFeatures = sparseDF.head.getAs[SparseVector](sparseFeaturesCol).size

    sparseDF.select(
      lit(numFeatures) as csvNumFeaturesCol,
      getSparseIndices(col(sparseFeaturesCol)) as csvWordIndicesStrCol,
      getSparseValues(col(sparseFeaturesCol)) as csvWordCountsStrCol,
      col(sparseLabelsCol) as csvLabelCol
    )
  }

  def sparseDFFromCSVReadyDF(csvReadyDF: DataFrame): DataFrame = {
    val Array(sparseFeaturesCol, sparseLabelsCol) = SchemaForSparseDataFrames.fieldNames
    val Array(numFeaturesCol, wordIndicesCol, wordCountsCol, labelCol) = csvReadyDF.columns

    val createSparseColumn = udf {
      (numFeatures: Int, wordIndicesStr: String, wordCountsStr: String) =>
        val wordIndices = Option(wordIndicesStr)
          .map(_.split(",").map(_.toInt))
          .getOrElse(Array.empty[Int])
        val wordCounts = Option(wordCountsStr)
          .map(_.split(",").map(_.toDouble))
          .getOrElse(Array.empty[Double])
        new SparseVector(numFeatures, wordIndices, wordCounts)
    }

    csvReadyDF.select(
      createSparseColumn(col(numFeaturesCol), col(wordIndicesCol), col(wordCountsCol)) as sparseFeaturesCol,
      col(labelCol) as sparseLabelsCol
    )
  }
}