package net.ddns.akgunter.spark_learning.util

import java.io.File
import java.nio.file.Paths

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

object FileUtil {
  val labelPrefix: String = "class"
  val labelPattern: String = s"$labelPrefix[A-Z]*"

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
    val subDirs = new File(baseDir)
      .listFiles
      .filter(_.isDirectory)
      .map(_.toString)

    subDirs.flatMap(getDataFiles)
  }

  def traverseUnlabeledDataFiles(baseDir: String): Seq[String] = {
    getDataFiles(baseDir)
  }

  def getLabelFromFilePath(filePath: String): String = {
    val foundPattern = labelPattern.r.findFirstIn(filePath)

    foundPattern match {
      case Some(v) => v.substring(labelPrefix.length)
      case None => throw new IllegalArgumentException(s"File path $filePath is unlabelled")
    }
  }

  def labelToInt(label: String): Int = {
    label.map(_.toInt - 'A'.toInt)
      .zipWithIndex
      .map {
        case (ord, idx) =>
          ord * math.pow(26, idx).toInt
      }
      .sum
  }

  def dataFrameFromDirectory(baseDir: String, training: Boolean)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    import org.apache.spark.sql.functions._

    val fileSchema: StructType = new StructType()
      .add("word", StringType)
      .add("word_count", IntegerType)

    val dirPattern = {
      if (training)
        Paths.get(baseDir, labelPattern + "/*.res").toString
      else
        baseDir
    }

    val df = spark.read
      .schema(fileSchema)
      .option("mode", "DROPMALFORMED")
      .option("delimiter", " ")
      .csv(dirPattern)
      .select(
        trim(lower('word)) as "word",
        'word_count
      )
      .withColumn("input_file", input_file_name)

    if (training) {
      val getLabelStr = udf((path: String) => getLabelFromFilePath(path))
      val getLabel = udf((labelStr: String) => labelToInt(labelStr))
      df.withColumn("label_str", getLabelStr(col("input_file")))
        .withColumn("label", getLabel(col("label_str")))
    }
    else df
  }
}