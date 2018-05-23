package net.ddns.akgunter.scala_classifier.lib

import scalaz.\&/.That

import scala.collection.generic.CanBuildFrom

case class SparseMatrix[A: Numeric](table: Map[Int, SparseVector[A]],
                                    shape: (Int, Int)) extends Iterable[SparseVector[A]] {

  val rowDomain = this.table.keySet
  val colDomain = this.table.map(_._2.keySet).reduce(_ | _)


  def apply(idx: Int): SparseVector[A] = {
    if (this.table.contains(idx)) this.table(idx)
    else SparseVector(this.table(this.colDomain.head).vector.empty, this.width)
  }

  def transpose: SparseMatrix[A] = {
    val grid = this.colDomain.map {
      j =>
        val row = this.rowDomain.map {
          i => i -> this.table(i)(j)
        }.toMap
        j -> SparseVector(row, row.size)
    }.toMap
    SparseMatrix(grid, this.shape)
  }

  override def iterator: Iterator[SparseVector[A]] = {
    this.table.valuesIterator
  }

  override def size: Int = this.shape._1

  def length: Int = this.size

  def width: Int = this.shape._2

  def +(that: SparseMatrix[A]): SparseMatrix[A] = {
    if (this.shape != that.shape)
      throw new ArithmeticException(s"Shape ${this.shape} does not match shape ${that.shape}")

    val rows = (this.rowDomain | that.rowDomain).map {
      k: Int => this(k) + that(k)
    }

    SparseMatrix.fromMatrix(rows)
  }

  def ++(that: SparseMatrix[A]): SparseMatrix[A] = {
    if (this.width != that.width)
      throw new ArithmeticException(s"width ${this.width} does not match width ${that.width}")

    val newRows = that.table.map {
      case(k, v) => (k + this.length) -> v
    }

    val newTable = this.table ++ newRows
    SparseMatrix(newTable, (this.length + that.length, this.width))
  }
}

object SparseMatrix {

  def fromMatrix[A: Numeric](array: Iterable[Iterable[A]]): SparseMatrix[A] = {
    val table = array.zipWithIndex
      .map {
      case(row, i) =>
        val sparseVector = SparseVector.fromVector(row)
        i -> sparseVector
    }.filter {
      case(i, v) =>
        if (v.keySet.size > 0) true
        else false
    }.toMap

    SparseMatrix(table, array.size -> array.head.size)
  }
}