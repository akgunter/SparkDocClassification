package net.ddns.akgunter.spark_learning.data_processing

import org.apache.spark.ml.param.Param

trait VectorizeFileRowParams extends WordCountToVecParams {
  final val mapCol = new Param[String](this, "map", "The buffer's map column")

  setDefault(mapCol, "vfr_buffer_map")
}