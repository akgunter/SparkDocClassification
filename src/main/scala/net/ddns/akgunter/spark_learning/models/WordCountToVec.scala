package net.ddns.akgunter.spark_learning.models

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.{Param, Params, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}


trait WordCountToVecParams extends Params {
  final val fileCol = new Param[String](this, "fileCol", "The input file column")
  final val wordCol = new Param[String](this, "wordCol", "The input word column")
  final val countCol = new Param[String](this, "countCol", "The input word-count column")
  final val labelCol = new Param[String](this, "labelCol", "The optional input label column")
  final val indexCol = new Param[String](this, "indexCol", "The index output column")
  final val vectorCol = new Param[String](this, "vectorCol", "The vector output column")

  setDefault(fileCol, "input_file")
  setDefault(wordCol, "word")
  setDefault(countCol, "count")
  setDefault(labelCol, "label")
  setDefault(indexCol, "index")
  setDefault(vectorCol, "raw_word_vector")
}


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

class WordCountToVecModel protected (
    protected val ordering: Dataset[_],
    protected val dictionarySize: Long,
    override val uid: String)
  extends Model[WordCountToVecModel]
  with WordCountToVecParams {

  protected[models] def this(ordering: Dataset[_], maxIndex: Long) = {
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


trait VectorizeFileRowParams extends WordCountToVecParams {
  final val mapCol = new Param[String](this, "map", "The buffer's map column")

  setDefault(mapCol, "vfr_buffer_map")
}


class VectorizeFileRow protected (
    protected val dictionarySize: Int,
    override val uid: String)
  extends UserDefinedAggregateFunction
  with VectorizeFileRowParams {

  protected[models] def this(dictionarySize: Int) = {
    this(dictionarySize, Identifiable.randomUID("VectorizeFileRow"))
  }

  def setCountCol(value: String): VectorizeFileRow = set(countCol, value)

  def setIndexCol(value: String): VectorizeFileRow = set(indexCol, value)

  override def copy(extra: ParamMap): Params = defaultCopy(extra)

  override def inputSchema: StructType = {
    new StructType()
      .add($(indexCol), IntegerType)
      .add($(countCol), IntegerType)
  }

  override def bufferSchema: StructType = {
    new StructType()
      .add($(mapCol), MapType(IntegerType, IntegerType))
  }

  override def dataType: DataType = VectorType

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Map.empty[Int, Int]
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val index = input.getAs[Int](0)
    val count = input.getAs[Int](1)

    val idxMap = buffer.getAs[Map[Int, Int]](0)

    buffer(0) = idxMap + (index -> count)
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val map1 = buffer1.getAs[Map[Int, Int]](0)
    val map2 = buffer2.getAs[Map[Int, Int]](0)

    buffer1(0) = map1 ++ map2.map {
      case (k, v) => k -> (map1.getOrElse(k, 0) + v)
    }
  }

  override def evaluate(buffer: Row): Any = {
    val idxCountMap = buffer.getAs[Map[Int, Int]](0)

    val idxList = idxCountMap.keySet
      .toArray
      .sorted
      .map(_.toInt)
    val idxOrder = idxList.zipWithIndex.toMap

    val countList = idxCountMap.toArray
      .sortBy {
        case (idx, _) => idxOrder(idx)
      }
      .map {
        case (_, count) => count.toDouble
      }

    new SparseVector(dictionarySize, idxList, countList)
  }
}