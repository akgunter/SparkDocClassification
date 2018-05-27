package net.ddns.akgunter.scala_classifier.lib

import scala.math.Numeric

import scala.collection.GenTraversableOnce

case class SparseVector[A: Numeric](vector: Map[Int, A],
                                    length: Int) {

  def apply(idx: Int): A = {
    this.vector.getOrElse(idx, implicitly[Numeric[A]].zero)
  }

  def keySet: Set[Int] = this.vector.keySet

  def isZero: Boolean = this.keySet.nonEmpty

  def count(p: A => Boolean): Int = this.vector.values.count(p)

  def size: Int = this.length

  def sum: A = this.vector.values.sum

  def map[B: Numeric](f: (Int, A) => (Int, B)): SparseVector[B] = {
    SparseVector(this.vector.map(f(_)), this.length)
  }

  def +(that: SparseVector[A]): SparseVector[A] = {
    assert(
      this.length == that.length,
      s"Vectors have different lengths ${this.length} and ${that.length}"
    )

    val sumVector = (this.keySet | that.keySet).map {
      k => k -> implicitly[Numeric[A]].plus(this(k), that(k))
    }.toMap

    SparseVector(sumVector, this.length max that.length)
  }

  def *(that: SparseVector[A]): SparseVector[A] = {
    assert(
      this.length == that.length,
      s"Vectors have different lengths ${this.length} and ${that.length}"
    )

    val prodVector = (this.keySet & that.keySet).map {
      k => k -> implicitly[Numeric[A]].times(this(k), that(k))
    }.toMap

    SparseVector(prodVector, this.length)
  }

  override def toString: String = this.vector.toString
}

object SparseVector {
  def fromVector[A: Numeric](vector: Array[A]): SparseVector[A] = {
    val vectorMap = vector.zipWithIndex
      .filter {
        case (0, _) => false
        case _ => true
      }.map {
      case(v, i) => i -> v
    }.toMap

    SparseVector(vectorMap, vector.size)
  }

  def empty[A: Numeric](length: Int): SparseVector[A] = SparseVector(Map.empty[Int, A], length)
}