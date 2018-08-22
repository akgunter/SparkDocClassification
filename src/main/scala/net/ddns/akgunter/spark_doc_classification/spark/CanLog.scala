package net.ddns.akgunter.spark_doc_classification.spark

/*
CanLog trait authored by Wil Adamec (https://github.com/wiladamec) and used with permission.
 */

import org.slf4j.{Logger, LoggerFactory}


trait CanLog {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
}
