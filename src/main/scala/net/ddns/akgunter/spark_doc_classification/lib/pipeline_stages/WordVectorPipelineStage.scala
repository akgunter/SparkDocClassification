package net.ddns.akgunter.spark_doc_classification.lib.pipeline_stages

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.types._


/*
A base trait that ensures the implementing class is a PipelineStage with the parameters defined in WordVectorParams
- Enforces common schema requirements
 */
trait WordVectorPipelineStage extends PipelineStage with WordVectorParams {

  protected val requiredInputColumns: Set[Param[String]]
  protected val requiredOutputColumns: Set[Param[String]]

  override def transformSchema(schema: StructType): StructType = {
    val requiredColumnStrings = requiredInputColumns.map(param => $(param))
    val inputColumns = schema.fieldNames.toSet
    require(
      requiredColumnStrings.forall(inputColumns),
      s"Dataset is missing required column(s): ${requiredColumnStrings.diff(inputColumns).mkString(", ")}"
    )

    val requiredIColumnTypes = requiredInputColumns.map {
      param => $(param) -> WordVectorParams.COLUMN_TYPES(param.name)
    }
    val inputColumnTypes = schema.fieldNames.map {
      colStr => colStr -> schema.fields(schema.fieldIndex(colStr)).dataType
    }.toMap
    val failedColTypes = requiredIColumnTypes.map {
      case (colStr, reqColType) => colStr -> (reqColType, inputColumnTypes(colStr))
    }.filterNot {
      case (_, (reqColType, realColType)) => reqColType == realColType
    }
    require(
      failedColTypes.isEmpty,
      s"Dataset has incorrect column type(s):\n${failedColTypes.map {
        case (colStr, (reqColType, realColType)) =>
          s"$colStr expected: $reqColType got: $realColType"
      }.mkString(", ")}"
    )

    val requiredOColumnTypes = requiredOutputColumns.map {
      param => $(param) -> WordVectorParams.COLUMN_TYPES(param.name)
    }

    val outputColumnTypes = inputColumnTypes ++ requiredOColumnTypes
    val outSchema = WordVectorPipelineStage.buildSchema(outputColumnTypes)

    if (schema.fieldNames.contains($(labelCol)) && !outSchema.fieldNames.contains($(labelCol)))
      outSchema.add($(labelCol), IntegerType)
    else
      outSchema
  }
}

object WordVectorPipelineStage {
  protected def buildSchema(columnTypes: Map[String, DataType]): StructType = {
    buildSchema(columnTypes.iterator, new StructType())
  }

  protected def buildSchema(colTypeIter: Iterator[(String, DataType)], schema: StructType): StructType = {
    if (!colTypeIter.hasNext)
      schema
    else {
      val (nextCol, nextType) = colTypeIter.next
      buildSchema(colTypeIter, schema.add(nextCol, nextType))
    }
  }
}