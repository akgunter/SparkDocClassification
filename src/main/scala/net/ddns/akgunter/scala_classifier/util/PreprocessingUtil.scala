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
    SparseMatrix.fromMatrix(vectorList)
  }

  def calcTF(dataRow: SparseVector[Int]): SparseVector[Double] = {
    val wordsInRow = dataRow.sum.toDouble
    SparseVector.fromVector(dataRow.map(_ / wordsInRow))
  }

  def calcIDF(dataMatrix: SparseMatrix[Int]): SparseVector[Double] = {
    val mtrx = dataMatrix.transpose.map {
      col =>
        val numDocs = col.count(_ != 0)
        -Math.log10(numDocs / (dataMatrix.length + numDocs))
    }

    SparseVector.fromVector(mtrx)
  }

  def calcTFIDF(dataMatrix: SparseMatrix[Int],
                idfVector: SparseVector[Double]): SparseMatrix[Double] = {

    val tfMatrix: SparseMatrix[Double] = SparseMatrix.fromMatrix(dataMatrix.map(calcTF))
    println(Array(tfMatrix.length, tfMatrix.width, idfVector.length).mkString(","))

    SparseMatrix.fromMatrix(Array(Array(0.0).toIterable))
  }
}