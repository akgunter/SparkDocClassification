package net.ddns.akgunter.scala_classifier.util

import java.io.File

object FileUtil {

  def listDataFiles(rootDir: String): Array[String] = {
    new File(rootDir)
      .listFiles
      .filter { f => f.isFile && f.getName.endsWith(".res") }
      .map(_.toString)
  }

  def traverseLabeledDataFiles(baseDir: String): Array[String] = {
    val subdirs = new File(baseDir)
      .listFiles
      .filter(_.isDirectory)
      .map(_.toString)

    subdirs.flatMap(listDataFiles)
  }

  def traverseUnlabeledDataFiles(baseDir: String): Array[String] = {
    listDataFiles(baseDir)
  }

  def getLabelFromFilePath(filePath: String): String = {
    val foundPattern = "class[A-Z]*".r.findFirstIn(filePath)

    foundPattern match {
      case Some(v) => v
      case None => throw new IllegalArgumentException(s"File path ${filePath} is unlabelled")
    }
  }
}