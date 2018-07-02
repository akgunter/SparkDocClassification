package net.ddns.akgunter.scala_classifier.util

import java.io.File
import java.nio.file.Paths
import scala.util.matching.Regex

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._

object FileUtil {
  val labelPattern: String = "class[A-Z]*"

  final val fileSchema: StructType = new StructType()
    .add("word", StringType)
    .add("count", IntegerType)

  def getDataFiles(baseDir: String): Seq[String] = {
    new File(baseDir)
      .listFiles
      .filter { f => f.isFile && f.getName.endsWith(".res") }
      .map(_.toString)
  }

  def getLabelDirectories(baseDir: String): Seq[String] = {
    new File(baseDir)
      .listFiles
      .filter {
        name =>
          name.isDirectory && labelPattern.r.findFirstIn(name.getName).isDefined
      }
      .map(_.toString)
  }

  def traverseLabeledDataFiles(baseDir: String): Seq[String] = {
    val subdirs = new File(baseDir)
      .listFiles
      .filter(_.isDirectory)
      .map(_.toString)

    subdirs.flatMap(getDataFiles)
  }

  def traverseUnlabeledDataFiles(baseDir: String): Seq[String] = {
    getDataFiles(baseDir)
  }

  def getLabelFromFilePath(filePath: String): String = {
    val foundPattern = labelPattern.r.findFirstIn(filePath)

    foundPattern match {
      case Some(v) => v
      case None => throw new IllegalArgumentException(s"File path $filePath is unlabelled")
    }
  }

  def loadChenCSV(filePath: String, training: Boolean)(implicit spark: SparkSession): DataFrame = {
    import org.apache.spark.sql.functions.{input_file_name, lit}

    println(s"Loading file $filePath")

    val df = spark.read
      .schema(fileSchema)
      .option("mode", "DROPMALFORMED")
      .option("delimiter", " ")
      .csv(filePath)
      .withColumn("input_file", input_file_name)

    if (training)
      df.withColumn("label", lit(getLabelFromFilePath(filePath)))
    else
      df
  }

  def dataFrameFromDirectory(baseDir: String, training: Boolean)(implicit spark: SparkSession): DataFrame = {
    import org.apache.spark.sql.functions.{col, input_file_name, udf}

    val dirPattern = {
      if (training)
        Paths.get(baseDir, labelPattern).toString
      else
        baseDir
    }

    val df = spark.read
      .schema(fileSchema)
      .option("mode", "DROPMALFORMED")
      .option("delimiter", " ")
      .csv(dirPattern)
      .withColumn("input_file", input_file_name)

    if (training) {
      val getLabel = udf((path: String) => getLabelFromFilePath(path))
      df.withColumn("label", getLabel(col("input_file")))
    }
    else df
  }
}