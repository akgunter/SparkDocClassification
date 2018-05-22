package net.ddns.akgunter.scala_classifier.models

case class WordIndex(wordCounts: DataPoint, wordOrdering: Array[String])

object WordIndex {
  def fromDataSet(dataSet: Array[DataPoint]): WordIndex = {
    val wordCounts = dataSet.reduce(_ ++ _)
    val wordOrdering = wordCounts.toMap.keySet.toArray

    WordIndex(wordCounts, wordOrdering)
  }
}