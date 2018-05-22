package net.ddns.akgunter.scala_classifier.util

import net.ddns.akgunter.scala_classifier.models.DataPoint

object PreprocessingUtil {

  def vectorize(point: DataPoint,
                wordOrdering: Array[String]): Array[Int] = {

    wordOrdering.map(point.toMap.getOrElse(_, 0))
  }

  def buildMatrix(dataSet: Array[DataPoint],
                  wordOrdering: Array[String]): Array[Array[Int]] = {

    dataSet.map(vectorize(_, wordOrdering))
  }

  def tfidf()
}