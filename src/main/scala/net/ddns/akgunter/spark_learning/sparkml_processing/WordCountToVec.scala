package net.ddns.akgunter.spark_learning.sparkml_processing

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.LongType

class WordCountToVec(override val uid: String)
  extends Estimator[WordCountToVecModel]
    with WordVectorPipelineStage {

  override protected val requiredInputColumns: Set[Param[String]] = Set(
    fileCol,
    wordCol,
    wordCountCol
  )
  override protected val requiredOutputColumns: Set[Param[String]] = Set (
    vectorCol
  )

  def this() = this(Identifiable.randomUID("WordCountToVec"))

  def setFileCol(value: String): WordCountToVec = set(fileCol, value)

  def setWordCol(value: String): WordCountToVec = set(wordCol, value)

  def setCountCol(value: String): WordCountToVec = set(wordCountCol, value)

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
      .setCountCol($(wordCountCol))
      .setFileCol($(fileCol))
      .setIndexCol($(indexCol))
      .setLabelCol($(labelCol))
      .setVectorCol($(vectorCol))
      .setWordCol($(wordCol))
  }
}