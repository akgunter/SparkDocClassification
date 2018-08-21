package net.ddns.akgunter.spark_doc_classification.sparkml_processing

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.types.{IntegerType, StringType, DataType}

trait WordVectorParams extends Params {
  final val fileCol = new Param[String](this, "fileCol", "The input file column")
  final val wordCol = new Param[String](this, "wordCol", "The input word column")
  final val wordCountCol = new Param[String](this, "wordCountCol", "The input word-count column")
  final val labelCol = new Param[String](this, "labelCol", "The optional input label column")
  final val indexCol = new Param[String](this, "indexCol", "The index output column")
  final val vectorCol = new Param[String](this, "vectorCol", "The vector output column")

  setDefault(fileCol, "input_file")
  setDefault(wordCol, "word")
  setDefault(wordCountCol, "word_count")
  setDefault(labelCol, "label")
  setDefault(indexCol, "word_index")
  setDefault(vectorCol, "raw_word_vector")
}

object WordVectorParams {
  final val COLUMN_TYPES = Map(
    "fileCol" -> StringType,
    "wordCol" -> StringType,
    "wordCountCol" -> IntegerType,
    "labelCol" -> IntegerType,
    "indexCol" -> IntegerType,
    "vectorCol" -> VectorType
  )

  protected def getColType(param: Param[String]): DataType = {
    COLUMN_TYPES(param.name)
  }
}