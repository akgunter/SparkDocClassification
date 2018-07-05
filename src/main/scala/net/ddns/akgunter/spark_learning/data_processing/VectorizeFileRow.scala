package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{DataType, IntegerType, MapType, StructType}

class VectorizeFileRow protected (
    protected val dictionarySize: Int,
    override val uid: String)
  extends UserDefinedAggregateFunction
    with WordVectorParams {

  final val mapCol = new Param[String](this, "map", "The buffer's map column")
  setDefault(mapCol, "vfr_buffer_map")

  protected[data_processing] def this(dictionarySize: Int) = {
    this(dictionarySize, Identifiable.randomUID("VectorizeFileRow"))
  }

  def setCountCol(value: String): VectorizeFileRow = set(wordCountCol, value)

  def setIndexCol(value: String): VectorizeFileRow = set(indexCol, value)

  override def copy(extra: ParamMap): Params = defaultCopy(extra)

  override def inputSchema: StructType = {
    new StructType()
      .add($(indexCol), IntegerType)
      .add($(wordCountCol), IntegerType)
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