package net.ddns.akgunter.scala_classifier.lib

import scala.math.Numeric

case class SparseVector[A: Numeric](vector: Map[Int, A],
                                    length: Int) extends Iterable[A] {

  def apply(idx: Int): A = {
    if (this.vector.contains(idx)) this.vector(idx)
    else implicitly[Numeric[A]].zero
  }

  def keySet: Set[Int] = this.vector.keySet

  override def iterator: Iterator[A] = {
    this.vector.valuesIterator
  }

  override def count(p: A => Boolean): Int = this.vector.values.count(p)

  def sum: A = this.vector.values.sum

  override def size: Int = this.length

  def +(that: SparseVector[A]): SparseVector[A] = {
    if (this.length != that.length)
      throw new IllegalArgumentException(s"Vectors have different lengths ${this.length} and ${that.length}")

    val sumVector = (this.keySet | that.keySet).map {
      k => k -> implicitly[Numeric[A]].plus(this(k), that(k))
    }.toMap

    SparseVector(sumVector, this.length max that.length)
  }

  def *(that: SparseVector[A]): SparseVector[A] = {
    if (this.length != that.length)
      throw new IllegalArgumentException(s"Vectors have different lengths ${this.length} and ${that.length}")

    val prodVector = (this.keySet & that.keySet).map {
      k => k -> implicitly[Numeric[A]].times(this(k), that(k))
    }.toMap

    SparseVector(prodVector, this.length)
  }

  override def toString(): String = this.vector.toString
}

object SparseVector {
  def fromVector[A: Numeric](vector: Iterable[A]): SparseVector[A] = {
    val vectorMap = vector.zipWithIndex
      .filter {
        case (0, _) => false
        case _ => true
      }.map {
      case(v, i) => i -> v
    }.toMap

    SparseVector(vectorMap, vector.size)
  }
}