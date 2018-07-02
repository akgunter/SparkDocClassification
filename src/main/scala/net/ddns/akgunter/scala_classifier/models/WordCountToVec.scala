package net.ddns.akgunter.scala_classifier.models

import scala.collection.mutable.{Map => MMap}

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

class WordCountToVec(override val uid: String) extends Estimator[WordCountToVecModel] {

  def this() = this(Identifiable.randomUID("wctv"))

  protected def getVocabOrdering(dataset: Dataset[_]): Dataset[_] = {
    import org.apache.spark.sql.functions.monotonically_increasing_id

    dataset.select("word")
      .distinct
      .withColumn("index", monotonically_increasing_id)
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
    val maxIndex = ordering
      .select(max("index"))
      .collect()
      .head
      .getAs[Long]("index")

    new WordCountToVecModel(ordering, maxIndex)
      .setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[WordCountToVecModel] = ???

  override def transformSchema(schema: StructType): StructType = ???
}


class WordCountToVecModel protected (
  protected val ordering: Dataset[_],
  protected val maxIndex: Long,
  override val uid: String) extends Model[WordCountToVecModel] {

  def this(ordering: Dataset[_], maxIndex: Long) = this(ordering, maxIndex, Identifiable.randomUID("wctvm"))

  override def copy(extra: ParamMap): WordCountToVecModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = {
    import org.apache.spark.sql.functions.col

    val requiredColumns = Set("input_file", "word", "count")
    val inputColumns = dataset.columns.toSet
    require(
      requiredColumns.forall(inputColumns),
      s"Dataset is missing required column(s): ${requiredColumns.diff(inputColumns).mkString(", ")}"
    )

    val fileRowVectorizer = new VectorizeFileRow(maxIndex.toInt)
    dataset.join(ordering, "word")
      .select("input_file", "index", "count")
      .groupBy("input_file")
      .agg(fileRowVectorizer(col("index"), col("count")))
  }

  override def transformSchema(schema: StructType): StructType = ???
}


class VectorizeFileRow(maxIndex: Int) extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = {
    new StructType()
      .add("index", IntegerType)
      .add("count", IntegerType)
  }

  override def bufferSchema: StructType = {
    new StructType()
      .add("map", MapType(LongType, IntegerType))
  }

  override def dataType: DataType = {
    new StructType()
      .add("input_file", StringType)
      .add("vector", VectorType)
  }

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = MMap.empty[Int, Int]
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val index = input.getAs[Long]("index")
    val count = input.getAs[Int]("count")

    buffer.getAs[MMap[Int, Int]](0)(index.toInt) = count
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val map1 = buffer1.getAs[MMap[Int, Int]](0)
    val map2 = buffer2.getAs[MMap[Int, Int]](0)

    map2.foreach {
      case (idx, count) =>
        if (map1.contains(idx)) map1(idx) += count
        else map1(idx) = count
    }
  }

  override def evaluate(buffer: Row): Any = {
    val idxCountMap = buffer.getAs[MMap[Int, Int]](0)

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

    new SparseVector(maxIndex, idxList, countList)
  }
}