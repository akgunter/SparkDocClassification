package net.ddns.akgunter.spark_learning.sparkml_processing

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Column, DataFrame, Dataset}


class WordCountToVecModel protected (
    protected val ordering: DataFrame,
    protected val dictionarySize: Long,
    override val uid: String)
  extends Model[WordCountToVecModel]
    with WordVectorPipelineStage {

  override protected val requiredInputColumns: Set[Param[String]] = Set(
    fileCol,
    wordCol,
    wordCountCol
  )
  override protected val requiredOutputColumns: Set[Param[String]] = Set (
    vectorCol
  )

  protected[sparkml_processing] def this(ordering: DataFrame, maxIndex: Long) = {
    this(ordering, maxIndex, Identifiable.randomUID("WordCountToVecModel"))
  }

  def setFileCol(value: String): WordCountToVecModel = set(fileCol, value)

  def setWordCol(value: String): WordCountToVecModel = set(wordCol, value)

  def setCountCol(value: String): WordCountToVecModel = set(wordCountCol, value)

  def setLabelCol(value: String): WordCountToVecModel = set(labelCol, value)

  def setIndexCol(value: String): WordCountToVecModel = set(indexCol, value)

  def setVectorCol(value: String): WordCountToVecModel = set(vectorCol, value)

  override def copy(extra: ParamMap): WordCountToVecModel = defaultCopy(extra)

  def getDictionarySize: Long = this.dictionarySize

  def getOrdering: DataFrame = this.ordering

  override def transform(dataset: Dataset[_]): DataFrame = {
    import org.apache.spark.sql.functions.col

    val groupByColumns = {
      if (dataset.columns.contains($(labelCol)))
        List($(fileCol), $(labelCol))
      else
        List($(fileCol))
    }.map(colParam => new Column(colParam))

    val fileRowVectorizer = new VectorizeFileRow(dictionarySize.toInt)
      .setCountCol($(wordCountCol))
      .setIndexCol($(indexCol))

    dataset.join(ordering, $(wordCol))
      .groupBy(groupByColumns: _*)
      .agg(fileRowVectorizer(col($(indexCol)), col($(wordCountCol))) as $(vectorCol))
  }
}