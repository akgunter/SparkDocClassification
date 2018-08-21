package net.ddns.akgunter.spark_doc_classification.sparkml_processing

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}

class CommonElementFilterModel protected (
   protected val wordsToKeep: DataFrame,
   override val uid: String)
  extends Model[CommonElementFilterModel]
    with WordVectorPipelineStage {

  override protected val requiredInputColumns: Set[Param[String]] = Set(
    fileCol,
    wordCol
  )
  override protected val requiredOutputColumns: Set[Param[String]] = Set()

  protected[sparkml_processing] def this(wordsToKeep: DataFrame) = {
    this(wordsToKeep, Identifiable.randomUID("CommonElementFilterModel"))
  }

  def setFileCol(value: String): CommonElementFilterModel = set(fileCol, value)

  def setWordCol(value: String): CommonElementFilterModel = set(wordCol, value)

  def getWordsToKeep: DataFrame = this.wordsToKeep

  override def copy(extra: ParamMap): CommonElementFilterModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = dataset.join(wordsToKeep, $(wordCol))
}
