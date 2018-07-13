package net.ddns.akgunter.spark_learning.util

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{Column, DataFrame}
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


  def pipelineDFToProcessedDF(pipelineDF: DataFrame, pipelineFeaturesCol: String, pipelineLabelsCol: String): DataFrame = {
    val getSparseIndices = udf {
      v: SparseVector =>
        v.indices.mkString(",")
    }
    val getSparseValues = udf {
      v: SparseVector =>
        v.values.mkString(",")
    }

    val Array(procNumFeaturesCol, procWordIndicesStrCol, procWordCountsStrCol, procLabelCol) = SchemaForProcDataFiles.fieldNames
    val numFeatures = pipelineDF.head.getAs[SparseVector](pipelineFeaturesCol).size

    pipelineDF.select(
      lit(numFeatures) as procNumFeaturesCol,
      getSparseIndices(col(pipelineFeaturesCol)) as procWordIndicesStrCol,
      getSparseValues(col(pipelineFeaturesCol)) as procWordCountsStrCol,
      col(pipelineLabelsCol) as procLabelCol
    )
  }
}