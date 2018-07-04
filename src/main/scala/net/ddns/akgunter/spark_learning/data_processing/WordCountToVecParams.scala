package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.ml.param.{Param, Params}

trait WordCountToVecParams extends Params {
  final val fileCol = new Param[String](this, "fileCol", "The input file column")
  final val wordCol = new Param[String](this, "wordCol", "The input word column")
  final val countCol = new Param[String](this, "countCol", "The input word-count column")
  final val labelCol = new Param[String](this, "labelCol", "The optional input label column")
  final val indexCol = new Param[String](this, "indexCol", "The index output column")
  final val vectorCol = new Param[String](this, "vectorCol", "The vector output column")

  setDefault(fileCol, "input_file")
  setDefault(wordCol, "word")
  setDefault(countCol, "count")
  setDefault(labelCol, "label")
  setDefault(indexCol, "index")
  setDefault(vectorCol, "raw_word_vector")
}