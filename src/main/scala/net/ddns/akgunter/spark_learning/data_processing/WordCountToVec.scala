package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{IntegerType, LongType, StringType, StructType}


class WordCountToVec(override val uid: String)
  extends Estimator[WordCountToVecModel]
  with WordCountToVecParams {

  def this() = this(Identifiable.randomUID("WordCountToVec"))

  def setFileCol(value: String): WordCountToVec = set(fileCol, value)

  def setWordCol(value: String): WordCountToVec = set(wordCol, value)

  def setCountCol(value: String): WordCountToVec = set(countCol, value)

  def setLabelCol(value: String): WordCountToVec = set(labelCol, value)

  def setIndexCol(value: String): WordCountToVec = set(indexCol, value)

  def setVectorCol(value: String): WordCountToVec = set(vectorCol, value)

  override def copy(extra: ParamMap): Estimator[WordCountToVecModel] = defaultCopy(extra)

  protected def getVocabOrdering(dataset: Dataset[_]): DataFrame = {
    val wordSet = dataset.select($(wordCol))

    dataset.sparkSession.createDataFrame(
      wordSet
        .distinct
        .rdd
        .zipWithIndex
        .map { case (row, idx) =>
          Row.fromSeq(row.toSeq :+ idx)
        },
      wordSet.schema.add($(indexCol), LongType)
    )
  }

  override def fit(dataset: Dataset[_]): WordCountToVecModel = {
    import org.apache.spark.sql.functions.max

    val ordering = getVocabOrdering(dataset)
    val maxIndex = ordering.agg(max($(indexCol)))
      .head()
      .getLong(0)

    new WordCountToVecModel(ordering, maxIndex + 1)
      .setParent(this)
      .setCountCol($(countCol))
      .setFileCol($(fileCol))
      .setIndexCol($(indexCol))
      .setLabelCol($(labelCol))
      .setVectorCol($(vectorCol))
      .setWordCol($(wordCol))
  }

  override def transformSchema(schema: StructType): StructType = {
    val requiredColumns = Map(
      $(fileCol) -> StringType,
      $(wordCol) -> StringType,
      $(countCol) -> IntegerType
    )
    val inputColumns = schema.fieldNames.toSet
    require(
      requiredColumns.keySet.forall(inputColumns),
      s"Dataset is missing required column(s): ${requiredColumns.keySet.diff(inputColumns).mkString(", ")}"
    )

    val failedCols = requiredColumns.map {
      case (col, reqColType) => col -> (reqColType, schema.fields(schema.fieldIndex(col)).dataType)
    }.filterNot {
      case (_, (reqColType, realColType)) => reqColType == realColType
    }
    require(
      failedCols.isEmpty,
      s"Dataset has incorrect column type(s):\n${failedCols.map {
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