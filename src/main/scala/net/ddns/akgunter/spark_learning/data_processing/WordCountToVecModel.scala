package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}


class WordCountToVecModel protected (
    protected val ordering: Dataset[_],
    protected val dictionarySize: Long,
    override val uid: String)
  extends Model[WordCountToVecModel]
    with WordCountToVecParams {

  protected[data_processing] def this(ordering: Dataset[_], maxIndex: Long) = {
    this(ordering, maxIndex, Identifiable.randomUID("WordCountToVecModel"))
  }

  def setFileCol(value: String): WordCountToVecModel = set(fileCol, value)

  def setWordCol(value: String): WordCountToVecModel = set(wordCol, value)

  def setCountCol(value: String): WordCountToVecModel = set(countCol, value)

  def setLabelCol(value: String): WordCountToVecModel = set(labelCol, value)

  def setIndexCol(value: String): WordCountToVecModel = set(indexCol, value)

  def setVectorCol(value: String): WordCountToVecModel = set(vectorCol, value)

  override def copy(extra: ParamMap): WordCountToVecModel = defaultCopy(extra)

  def getDictionarySize: Long = this.dictionarySize


  override def transform(dataset: Dataset[_]): DataFrame = {
    import org.apache.spark.sql.functions.col

    val groupByColumns = {
      if (dataset.columns.contains($(labelCol)))
        List($(fileCol), $(labelCol))
      else
        List($(fileCol))
    }.map(colParam => new Column(colParam))

    val fileRowVectorizer = new VectorizeFileRow(dictionarySize.toInt)
      .setCountCol($(countCol))
      .setIndexCol($(indexCol))

    dataset.join(ordering, $(wordCol))
      .groupBy(groupByColumns: _*)
      .agg(fileRowVectorizer(col($(indexCol)), col($(countCol))) as $(vectorCol))
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