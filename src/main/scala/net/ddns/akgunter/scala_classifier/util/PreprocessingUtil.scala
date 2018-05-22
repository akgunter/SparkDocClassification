package net.ddns.akgunter.scala_classifier.util

import net.ddns.akgunter.scala_classifier.models.DataPoint
import net.ddns.akgunter.scala_classifier.models.WordIndex

object PreprocessingUtil {

  def vectorize(point: DataPoint,
                wordIndex: WordIndex): Array[Int] = {

    wordIndex.wordOrdering.map(point.toMap.getOrElse(_, 0))
  }

  def buildMatrix(dataSet: Array[DataPoint],
                  wordIndex: WordIndex): Array[Array[Int]] = {

    dataSet.map(vectorize(_, wordIndex))
  }

  def calcTF(dataRow: Array[Int]): Array[Double] = {
    val wordsInRow = dataRow.sum.toDouble
    dataRow.map(_ / wordsInRow)
  }

  def calcIDF(dataMatrix: Array[Array[Int]]): Array[Double] = {
    dataMatrix.transpose.map {
      col => Math.log10(dataMatrix.length / col.count(_ != 0))
    }
  }

  def tfidf(dataMatrix: Array[Array[Int]],
            idfVector: Array[Double]): Array[Array[Double]] = {

    val tfMatrix = dataMatrix.map(calcTF)
    println(Array(tfMatrix.length, tfMatrix.head.length, idfVector.length).mkString(","))

    Array(Array(0.0))
  }
}