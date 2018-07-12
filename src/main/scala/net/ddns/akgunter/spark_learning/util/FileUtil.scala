package net.ddns.akgunter.spark_learning.util

import java.io.File
import java.nio.file.Paths

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.transform.schema.Schema
import org.datavec.api.transform.TransformProcess
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction


object FileUtil {
  val DICTIONARY_DIRNAME: String = "Dictionary"
  val TRAINING_DIRNAME: String = "Training"
  val VALIDATION_DIRNAME: String = "Validation"
  val SCHEMA_DIRNAME: String = "Schema"

  val LABEL_PREFIX: String = "class"
  val LABEL_PATTERN: String = s"$LABEL_PREFIX[A-Z]*"

  val SCHEMA_DATAPATH_COLUMN: String = "data_path"
  val SCHEMA_DATASCHEMA_COLUMN: String = "data_schema"

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
          name.isDirectory && LABEL_PATTERN.r.findFirstIn(name.getName).isDefined
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

  def getLabelFromFilePath(filePath: String): String = {
    val foundPattern = LABEL_PATTERN.r.findFirstIn(filePath)

    foundPattern match {
      case Some(v) => v.substring(LABEL_PREFIX.length)
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
    import spark.implicits._
    import org.apache.spark.sql.functions._

    val fileSchema = new StructType()
      .add("word", StringType)
      .add("word_count", IntegerType)

    val dirPattern = {
      if (isLabelled)
        Paths.get(baseDirPath, LABEL_PATTERN + "/*.res").toString
      else
        Paths.get(baseDirPath, "/*.res").toString
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

    if (isLabelled) {
      val getLabelStr = udf((path: String) => getLabelFromFilePath(path))
      val getLabel = udf((labelStr: String) => labelToInt(labelStr))
      df.withColumn("label_str", getLabelStr(col("input_file")))
        .withColumn("label", getLabel(col("label_str")))
    }
    else df
  }

  protected def getDataFrameSchemas(schemaDirPath: String)(implicit spark: SparkSession): DataFrame = {
    val schemaForSchemaDF = new StructType()
      .add(SCHEMA_DATAPATH_COLUMN, StringType)
      .add(SCHEMA_DATASCHEMA_COLUMN, StringType)

    val schemaDirPattern = Paths.get(schemaDirPath, "/*.csv").toString
    spark.read
      .option("header", "true")
      .csv(schemaDirPattern)
  }

  def dataFrameFromProcessedDirectory(baseDirPath: String, schemaDirPath: String)(implicit spark: SparkSession): DataFrame = {
    val schemaDF = getDataFrameSchemas(schemaDirPath)

    val dataSchemaJSON = schemaDF.where(s"$SCHEMA_DATAPATH_COLUMN == '$baseDirPath'")
      .select(SCHEMA_DATAPATH_COLUMN)
      .head
      .getString(0)
    val dataSchema = DataType.fromJson(dataSchemaJSON).asInstanceOf[StructType]

    val baseDirPattern = Paths.get(baseDirPath, "/*.csv").toString

    spark.read
      .option("header", "false")
      .schema(dataSchema)
      .csv(baseDirPattern)
  }

  def writeProcessedDataFrame(dataFrame: DataFrame, dirPath: String, schemaDirPath: String): Unit = {
    val spark = dataFrame.sparkSession

    val schemaForSchemaDF = new StructType()
      .add(SCHEMA_DATAPATH_COLUMN, StringType)
      .add(SCHEMA_DATASCHEMA_COLUMN, StringType)

    val schemaDirPattern = Paths.get(schemaDirPath, "/*.csv").toString
    val schemaDF = spark.read
      .option("header", "true")
      .csv(schemaDirPattern)


  }
}