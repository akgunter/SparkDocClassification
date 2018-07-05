package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.types._

trait WordVectorPipelineStage extends PipelineStage with WordVectorParams {

  abstract val requiredColumns: Set[Param[String]]

  override def transformSchema(schema: StructType): StructType = {
    val requiredColumnStrings = requiredColumns.map(param => $(param))
    val inputColumns = schema.fieldNames.toSet
    require(
      requiredColumnStrings.forall(inputColumns),
      s"Dataset is missing required column(s): ${requiredColumnStrings.diff(inputColumns).mkString(", ")}"
    )

    val requiredColumnTypes = COLUMN_TYPES.filter {
      case (col, _) => requiredColumns(col)
    }
    val failedColTypes = requiredColumnTypes.map {
      case (col, reqColType) => col -> (reqColType, schema.fields(schema.fieldIndex($(col))).dataType)
    }.filterNot {
      case (_, (reqColType, realColType)) => reqColType == realColType
    }
    require(
      failedColTypes.isEmpty,
      s"Dataset has incorrect column type(s):\n${failedColTypes.map {
        case (col, (reqColType, realColType)) =>
          s"$col expected: $reqColType got: $realColType"
      }.mkString(", ")}"
    )

    val outSchema = new StructType()
      .add($(fileCol), StringType)
      .add($(vectorCol), VectorType)

    if (schema.fieldNames.contains($(labelCol)))
      outSchema.add($(labelCol), IntegerType)
    else
      outSchema
  }
}