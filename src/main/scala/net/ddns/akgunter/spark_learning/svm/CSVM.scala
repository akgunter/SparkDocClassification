package net.ddns.akgunter.spark_learning.svm

import net.ddns.akgunter.spark_learning.lib.{SparseMatrix, SparseVector}

case class CSVM(labelMap: Map[String, Int], sampleWeights: SparseVector[Double]) {

  override def toString: String = {
    (labelMap -> sampleWeights).toString()
  }
}

object CSVM {

  def fromData(points: SparseMatrix[Double], labels: Array[String]): CSVM = {
    val labelSet = labels.toSet
    require(labelSet.size <= 2)

    val labelMap = labelSet.toList.zipWithIndex.map {
      case (label, 0) =>
        label -> -1
      case (label, idx) =>
        label -> idx
    }.toMap

    val alpha = Array.fill(points.length) {
      1.0 / points.length
    }

    val denseWeights = CSVM.gradient_descent(alpha, points, labels.map(labelMap(_)))
    val sampleWeights = SparseVector.fromVector(denseWeights)

    CSVM(labelMap, sampleWeights)
  }

  private def partialObjective(idx: Int,
                               h: Double,
                               sampleWeights: Array[Double],
                               points: SparseMatrix[Double],
                               labels: Array[Int]): Double = {

    val arr = sampleWeights.indices.map {
      case j if j == idx => (0 max (sampleWeights(j) + h).toInt) * labels(j) * (points(idx) * points(j))
      case j => sampleWeights(j) * labels(j) * (points(idx) * points(j)) / 2
    }

    (0 max (sampleWeights(idx) + h).toInt) - arr.sum
  }

  private def gradient_descent(sampleWeights: Array[Double],
                               points: SparseMatrix[Double],
                               labels: Array[Int]): Array[Double] = {

    val h = 0.1
    val f0Arr = sampleWeights.indices.map(this.partialObjective(_, 0, sampleWeights, points, labels))
    val fhArr = sampleWeights.indices.map(this.partialObjective(_, h, sampleWeights, points, labels))

    (f0Arr zip fhArr).map {
      case (f, fh) => (f - fh) / h
    }.toArray
  }
}