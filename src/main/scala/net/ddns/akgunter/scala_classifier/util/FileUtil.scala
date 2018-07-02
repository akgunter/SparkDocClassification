package net.ddns.akgunter.scala_classifier.util

import java.io.File
import scala.util.matching.Regex

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

object FileUtil {
  val labelPattern: Regex = "class[A-Z]*".r

  final val fileSchema: StructType = new StructType()
    .add("word", StringType)
    .add("count", IntegerType)

  def getDataFiles(baseDir: String): Seq[String] = {
    new File(baseDir)
      .listFiles
      .filter { f => f.isFile && f.getName.endsWith(".res") }
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
    val foundPattern = labelPattern.findFirstIn(filePath)

    foundPattern match {
      case Some(v) => v
      case None => throw new IllegalArgumentException(s"File path $filePath is unlabelled")
    }
  }

  def dataFrameFromFile(filePath: String, training: Boolean)(implicit spark: SparkSession): DataFrame = {
    import org.apache.spark.sql.functions.lit

    println(s"Loading file $filePath")

    val df = spark.read
      .schema(fileSchema)
      .option("mode", "DROPMALFORMED")
      .option("delimiter", " ")
      .csv(filePath)
      .withColumn("input_file", lit(filePath))

    if (training)
      df.withColumn("label", lit(getLabelFromFilePath(filePath)))
    else
      df
  }

  def dataFrameFromDir(baseDir: String, training: Boolean)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    val fList = {
      if (training) traverseLabeledDataFiles(baseDir)
      else traverseUnlabeledDataFiles(baseDir)
    }.map(Tuple1.apply).toDF("path")

    fList.map {
      pathRow =>
        dataFrameFromFile(pathRow.getAs[String]("path"), training)
    }.reduce(_ union _)
  }
}