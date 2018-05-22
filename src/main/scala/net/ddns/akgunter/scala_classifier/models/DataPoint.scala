package net.ddns.akgunter.scala_classifier.models

import scalaz.Scalaz._

import scala.io.Source
import scala.util.{Failure, Success, Try}

case class DataPoint(private val data: Map[String, Int]) {

  def toMap: Map[String, Int] = this.data

  def ++(that: DataPoint): DataPoint = {
    DataPoint(this.data |+| that.data)
  }
}

object DataPoint {

  def fromFile(filename: String): DataPoint = {
    val data = Source.fromFile(filename, "iso-8859-1").mkString
      .split("\n")
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap {
        l =>
          Try {
            val Array(word, count) = l.split(" ", 2)
            word.trim -> count.toInt
          } match {
            case Success(y) => Some(y)
            case Failure(_) => println(s"Malformed line in $filename: $l"); None
          }
      }
      .toMap

    DataPoint(data)
  }
}