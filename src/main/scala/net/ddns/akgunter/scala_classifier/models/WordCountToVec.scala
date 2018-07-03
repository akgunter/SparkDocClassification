package net.ddns.akgunter.scala_classifier.models

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

//import scala.collection.mutable.{Map => MMap}

class WordCountToVec(override val uid: String) extends Estimator[WordCountToVecModel] {

  def this() = this(Identifiable.randomUID("wctv"))

  protected def getVocabOrdering(dataset: Dataset[_]): DataFrame = {
    val wordSet = dataset.select("word")

    dataset.sparkSession.createDataFrame(
      wordSet
        .distinct
        .rdd
        .zipWithIndex
        .map { case (row, idx) =>
          Row.fromSeq(row.toSeq :+ idx)
        },
      wordSet.schema.add("index", LongType)
    )
  }

  override def fit(dataset: Dataset[_]): WordCountToVecModel = {
    import org.apache.spark.sql.functions.max

    val requiredColumns = Set("input_file", "word", "count")
    val inputColumns = dataset.columns.toSet
    require(
      requiredColumns.forall(inputColumns),
      s"Dataset is missing required column(s): ${requiredColumns.diff(inputColumns).mkString(", ")}"
    )

    val ordering = getVocabOrdering(dataset)
    val maxIndex = ordering.agg(max("index"))
      .head()
      .getLong(0)

    new WordCountToVecModel(ordering, maxIndex + 1)
      .setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[WordCountToVecModel] = ???

  override def transformSchema(schema: StructType): StructType = ???
}


class WordCountToVecModel protected (
  protected val ordering: Dataset[_],
  protected val dictionarySize: Long,
  override val uid: String) extends Model[WordCountToVecModel] {

  def this(ordering: Dataset[_], maxIndex: Long) = this(ordering, maxIndex, Identifiable.randomUID("wctvm"))

  override def copy(extra: ParamMap): WordCountToVecModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = {
    import org.apache.spark.sql.functions.{col, max}

    val requiredColumns = Set("input_file", "word", "count")
    val inputColumns = dataset.columns.toSet
    require(
      requiredColumns.forall(inputColumns),
      s"Dataset is missing required column(s): ${requiredColumns.diff(inputColumns).mkString(", ")}"
    )

    val fileRowVectorizer = new VectorizeFileRow(dictionarySize.toInt)
    dataset.join(ordering, "word")
      .select("input_file", "index", "count")
      .groupBy("input_file")
      .agg(fileRowVectorizer(col("index"), col("count")))
  }

  override def transformSchema(schema: StructType): StructType = ???
}


class VectorizeFileRow(dictionarySize: Int) extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = {
    new StructType()
      .add("index", IntegerType)
      .add("count", IntegerType)
  }

  override def bufferSchema: StructType = {
    new StructType()
      .add("map", MapType(IntegerType, IntegerType))
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