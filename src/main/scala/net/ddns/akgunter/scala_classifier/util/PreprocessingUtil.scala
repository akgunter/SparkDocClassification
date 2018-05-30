package net.ddns.akgunter.scala_classifier.util

import net.ddns.akgunter.scala_classifier.lib._
import net.ddns.akgunter.scala_classifier.models.DataPoint
import net.ddns.akgunter.scala_classifier.models.WordIndex

object PreprocessingUtil {

  def vectorize(point: DataPoint,
                wordIndex: WordIndex): SparseVector[Int] = {

    val vector = wordIndex.wordOrdering.map(point.toMap.getOrElse(_, 0))
    SparseVector.fromVector(vector)
  }

  def buildSparseMatrix(dataSet: Array[DataPoint],
                        wordIndex: WordIndex): SparseMatrix[Int] = {

    val vectorList = dataSet.map(vectorize(_, wordIndex))
    SparseMatrix.fromSparseVectors(vectorList)
  }

  def calcTF(vector: SparseVector[Int]): SparseVector[Double] = {
    val wordsInRow = vector.sum.toDouble
    val newVector = vector.vector.map { case (k, v) => k -> v / wordsInRow }.toMap
    SparseVector(newVector, vector.length)
  }

  def calcIDF(dataMatrix: SparseMatrix[Int]): SparseVector[Double] = {
    val mtrx = dataMatrix.transpose.table.map {
      case (k, col) =>
        val numDocs = col.count(_ != 0).toDouble
          k -> -Math.log10(numDocs / (dataMatrix.length + numDocs))
    }.toMap

    SparseVector(mtrx, dataMatrix.width)
  }

  def calcTFIDF(dataMatrix: SparseMatrix[Int]): SparseMatrix[Double] = {

    val tfMatrix = dataMatrix.table.map { case (k, v) => k -> calcTF(v) }
    val idfVector = calcIDF(dataMatrix)
    val tfidfMatrix = tfMatrix.map { case (k, v) => k -> v * idfVector }.toMap

    SparseMatrix(tfidfMatrix, dataMatrix.shape)
  }
}