package net.ddns.akgunter.spark_learning.util

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.schema.Schema
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction

import org.nd4j.linalg.dataset.DataSet

import net.ddns.akgunter.spark_learning.util.DataFrameUtil._
import net.ddns.akgunter.spark_learning.util.FileUtil._

object DataSetUtil {

  def buildDataVecSchema(sparkSchema: StructType): Schema = {
    buildDataVecSchema(new Schema.Builder(), sparkSchema.fields.iterator)
  }

  protected def buildDataVecSchema(dataVecSchemaBuilder: Schema.Builder, schemaIterator: Iterator[StructField]): Schema = {
    if (!schemaIterator.hasNext) dataVecSchemaBuilder.build

    val nextField = schemaIterator.next
    val updatedSchema = nextField.dataType match {
      case StringType => dataVecSchemaBuilder.addColumnString(nextField.name)
      case IntegerType => dataVecSchemaBuilder.addColumnInteger(nextField.name)
      case DoubleType => dataVecSchemaBuilder.addColumnDouble(nextField.name)
    }

    buildDataVecSchema(updatedSchema, schemaIterator)
  }

  def dataSetFromDataFrame(baseDirPath: String)(implicit spark: SparkSession): DataSet = {
    val Array(sparseFeaturesCol, sparseLabelsCol) = SchemaForCoreDataFrames.fieldNames

    val dataFrameSourced = dataFrameFromProcessedDirectory(baseDirPath)
    val dataFrameSparse = sparseDFFromCSVReadyDF(dataFrameSourced)

    // TODO: Complete this using the dataset conversion code from runDL4JSpark()
    // TODO: Create functions to convert a DataFrame to a dense JavaRDD
    // TODO: Create functions to convert a dense JavaRDD to a DataSet

    null
  }
}