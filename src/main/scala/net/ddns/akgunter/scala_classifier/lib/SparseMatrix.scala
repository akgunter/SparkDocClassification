package net.ddns.akgunter.scala_classifier.lib

case class SparseMatrix[A: Numeric](table: Map[Int, SparseVector[A]],
                                    shape: (Int, Int)) {

  private val rowDomain: Set[Int] = this.table.keySet
  private val colDomain: Set[Int] = this.table.map(_._2.keySet).reduce(_ | _)

  def apply(idx: Int): SparseVector[A] = {
    this.table.getOrElse(idx, SparseVector.empty[A](this.width))
  }

  def map[B: Numeric](f: SparseVector[A] => SparseVector[B]): SparseMatrix[B] = {
    SparseMatrix(this.table.mapValues(f), this.shape)
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

  def length: Int = this.shape._1

  def width: Int = this.shape._2

  def +(that: SparseMatrix[A]): SparseMatrix[A] = {
    if (this.shape != that.shape)
      throw new ArithmeticException(s"Shape ${this.shape} does not match shape ${that.shape}")

    val rows = (this.rowDomain | that.rowDomain).map {
      k => k -> (this(k) + that(k))
    }.toMap

    SparseMatrix(rows, this.shape)
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

  def sum: SparseVector[A] = this.table.values.reduce(_ + _)

  override def toString: String = {

    val mtrxStr = this.table.keySet.toList.sorted.map {
      k => "\t" + k.toString + " ->\t" + this(k).toString
    }.mkString("\n")

    "SparseMatrix(\n" + mtrxStr + "\n)"
  }
}

object SparseMatrix {

  def fromMatrix[A: Numeric](array: Array[Array[A]]): SparseMatrix[A] = {
    fromSparseVectors(array.map(SparseVector.fromVector[A]))
  }

  def fromSparseVectors[A: Numeric](array: Seq[SparseVector[A]]): SparseMatrix[A] = {
    val table = array
      .zipWithIndex
      .map(_.swap)
      .filter {
        case (_, v) => v.nonEmpty
      }.toMap

    SparseMatrix(table, array.length -> array.head.length)
  }
}