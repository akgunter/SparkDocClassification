package net.ddns.akgunter.scala_classifier.lib

import scalaz.\&/.That

case class SparseMatrix[A: Numeric](table: Map[Int, SparseVector[A]],
                                    shape: (Int, Int)) extends Iterable[SparseVector[A]] {

  val rowDomain = this.table.keySet.toArray.sorted
  val colDomain = this.table.map(_._2.keySet).reduce(_ | _).toArray.sorted


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
}

object SparseMatrix {

  def fromMatrix[A: Numeric](array: Iterable[Iterable[A]]): SparseMatrix[A] = {
    val table = array.zipWithIndex.map {
      case(row, i) =>
        val sparseVector = SparseVector.fromVector(row)
        i -> sparseVector
    }.toMap

    SparseMatrix(table, array.size -> array.head.size)
  }
}