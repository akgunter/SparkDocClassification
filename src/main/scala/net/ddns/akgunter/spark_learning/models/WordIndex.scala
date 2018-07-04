package net.ddns.akgunter.spark_learning.models

case class WordIndex(wordCounts: DataPoint, wordOrdering: Array[String]) {

  def length: Int = wordOrdering.length
}

object WordIndex {
  def fromDataSet(dataSet: Array[DataPoint]): WordIndex = {
    val wordCounts = dataSet.reduce(_ ++ _)
    val wordOrdering = wordCounts.toMap.keySet.toArray

    WordIndex(wordCounts, wordOrdering)
  }
}