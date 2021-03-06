package net.ddns.akgunter.spark_doc_classification.util

import java.io.File
import java.nio.file.Paths

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

import net.ddns.akgunter.spark_doc_classification.util.DataFrameUtil._


object FileUtil {
  val LabelPrefix: String = "class"
  val LabelPattern: String = s"$LabelPrefix[A-Z]*"

  val TrainingDirName: String = "Training"
  val ValidationDirName: String = "Validation"
  val SchemaDirName: String = "Schema"


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

  /*
  Load the raw Bag-of-Words files, dropping garbage rows
  - Uses the SchemaForRawDataFiles schema
  - Adds a label column if this is loading training or validation data
   */
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

  /*
  Load preprocessed files into a dataframe
  - Uses the SchemaForProcDataFiles schema
   */
  def dataFrameFromProcessedDirectory(baseDirPath: String)(implicit spark: SparkSession): DataFrame = {
    val baseDirPattern = createCSVDirectoryPattern(baseDirPath)

    spark.read
      .option("header", "false")
      .schema(SchemaForProcDataFiles)
      .csv(baseDirPattern)
  }
}