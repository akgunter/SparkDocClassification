package net.ddns.akgunter.spark_doc_classification.util

import org.slf4j.{Logger, LoggerFactory}

trait LogHelper {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
}