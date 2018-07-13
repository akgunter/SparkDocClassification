package net.ddns.akgunter.spark_learning.util

import java.io.File
import java.nio.file.Paths

import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object FileUtil {
  val LabelPrefix: String = "class"
  val LabelPattern: String = s"$LabelPrefix[A-Z]*"

  val DictionaryDirName: String = "Dictionary"
  val TrainingDirName: String = "Training"
  val ValidationDirName: String = "Validation"
  val SchemaDirName: String = "Schema"

  val SchemaForRawDataFiles: StructType = new StructType()
    .add("word", StringType, nullable = false)
    .add("word_count", IntegerType, nullable = false)

  val SchemaForProcDataFiles: StructType = new StructType()
    .add("dictionary_size", IntegerType, nullable = false)
    .add("word_indices_str", StringType, nullable = false)
    .add("word_counts_str", StringType, nullable = false)
    .add("label", IntegerType, nullable = true)

  val SchemaForDictionaryFiles: StructType = new StructType()
    .add("word", StringType, nullable = false)
    .add("word_index", IntegerType, nullable = false)


  def createCSVDirectoryPattern(dirPath: String): String = Paths.get(dirPath, "/*.csv").toString

  def getDataFiles(baseDirPath: String): Seq[String] = {
    new File(baseDirPath)
      .listFiles
      .filter { f => f.isFile && f.getName.endsWith(".res") }
      .map(_.toString)
  }

  def getLabelDirectories(baseDirPath: String): Seq[String] = {
    new File(baseDirPath)
      .listFiles
      .filter {
        name =>
          name.isDirectory && LabelPattern.r.findFirstIn(name.getName).isDefined
      }
      .map(_.toString)
  }

  def traverseLabeledDataFiles(baseDirPath: String): Seq[String] = {
    val subDirs = new File(baseDirPath)
      .listFiles
      .filter(_.isDirectory)
      .map(_.toString)

    subDirs.flatMap(getDataFiles)
  }

  def traverseUnlabeledDataFiles(baseDirPath: String): Seq[String] = {
    getDataFiles(baseDirPath)
  }

  def getLabelFromFilePath(col: Column): Column = {
    regexp_replace(
      regexp_extract(col, LabelPattern, 0),
      LabelPrefix,
      ""
    )
  }

  def getLabelFromFilePath(filePath: String): String = {
    val foundPattern = LabelPattern.r.findFirstIn(filePath)

    foundPattern match {
      case Some(v) => v.substring(LabelPrefix.length)
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

  def dataFrameFromRawDirectory(baseDirPath: String, isLabelled: Boolean)(implicit spark: SparkSession): DataFrame = {
    val dirPattern = {
      if (isLabelled)
        Paths.get(baseDirPath, LabelPattern + "/*.res").toString
      else
        Paths.get(baseDirPath, "/*.res").toString
    }

    val Array(rawWordCol, rawWordCountCol) = SchemaForRawDataFiles.fieldNames
    val inputFileCol = "input_file"
    val labelStrCol = "label_str"
    val labelCol = "label"

    val df = spark.read
      .schema(SchemaForRawDataFiles)
      .option("mode", "DROPMALFORMED")
      .option("delimiter", " ")
      .csv(dirPattern)
      .select(
        trim(lower(col(rawWordCol))) as rawWordCol,
        col(rawWordCountCol)
      )
      .withColumn(inputFileCol, input_file_name)

    val getLabel = udf {
      labelStr: String =>
        Option(labelStr).map(labelToInt)
    }

    df.withColumn(labelStrCol, getLabelFromFilePath(col(inputFileCol)))
      .withColumn(labelCol, getLabel(col(labelStrCol)))
  }

  def dataFrameFromProcessedDirectory(baseDirPath: String)(implicit spark: SparkSession): DataFrame = {
    val baseDirPattern = createCSVDirectoryPattern(baseDirPath)

    spark.read
      .option("header", "false")
      .schema(SchemaForProcDataFiles)
      .csv(baseDirPattern)
  }

  def dataFrameFromDictionaryDirectory(baseDirPath: String)(implicit spark: SparkSession): DataFrame = {
    val baseDirPattern = createCSVDirectoryPattern(baseDirPath)

    spark.read
      .option("header", "false")
      .schema(SchemaForDictionaryFiles)
      .csv(baseDirPattern)
  }
}