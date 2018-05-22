package net.ddns.akgunter.scala_classifier.util

object PreprocessingUtil {

  def buildWordIndex(dataSets: Array[Array[DataPoint]]): (DataPoint, Array[String]) = {
    val wordLookup = dataSets.flatten.reduce(_ ++ _)
    val wordOrdering = wordLookup.toMap.keySet.toArray

    wordLookup -> wordOrdering
  }

  def vectorize(point: DataPoint,
                wordOrdering: Array[String]): Array[Int] = {

    wordOrdering.map(point.toMap.getOrElse(_, 0))
  }

  def buildMatrix(dataSet: Array[DataPoint],
                  wordOrdering: Array[String]): Array[Array[Int]] = {

    dataSet.map(vectorize(_, wordOrdering))
  }
}