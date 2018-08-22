package net.ddns.akgunter.spark_doc_classification.util

import java.util.{ArrayList => JavaArrayList}

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.api.java.JavaRDD

import org.datavec.api.transform.schema.Schema
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import net.ddns.akgunter.spark_doc_classification.util.DataFrameUtil._


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

  def dl4jRDDFromSparseDataFrame(dataFrame: DataFrame, numClasses: Int): JavaRDD[DataSet] = {
    val Array(sparseFeaturesCol, sparseLabelsCol) = SchemaForSparseDataFrames.fieldNames

    dataFrame.rdd.map {
      row =>
        val sparseVector = row.getAs[SparseVector](sparseFeaturesCol).toArray
        val label = row.getAs[Int](sparseLabelsCol)
        val fvec = Nd4j.create(sparseVector)
        val lvec = Nd4j.zeros(numClasses)
        lvec.putScalar(label, 1)
        new DataSet(fvec, lvec)
    }.toJavaRDD
  }

  /*
  Merge an RDD of DL4J DataSets into a single DataSet
   */
  def dataSetFromDL4JRDD(dl4jRDD: JavaRDD[DataSet]): DataSet = {
    DataSet.merge(new JavaArrayList(dl4jRDD.collect))
  }
}