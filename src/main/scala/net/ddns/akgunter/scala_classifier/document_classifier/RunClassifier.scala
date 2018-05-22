package net.ddns.akgunter.scala_classifier.document_classifier

import scopt.OptionParser

import net.ddns.akgunter.scala_classifier.util.FileUtil._
import net.ddns.akgunter.scala_classifier.util.PreprocessingUtil._
import net.ddns.akgunter.scala_classifier.models.DataPoint
import net.ddns.akgunter.scala_classifier.models.WordIndex

object RunClassifier {

  case class Opts(trainingDir: String = "",
                  validationDir: String = "",
                  testingDir: String = "")

  val parser = new OptionParser[Opts](this.getClass.getSimpleName.stripSuffix("$")) {
    val defaults = Opts()

    note("Arguments:")

    arg[String]("trainingDir")
      .action { (td, opts) => opts.copy(trainingDir = td) }
      .text("Path to directory containing training data files.")

    arg[String]("validationDir")
      .action { (vd, opts) => opts.copy(validationDir = vd) }
      .text("Path to directory containing validation data files.")

    arg[String]("testingDir")
      .action { (td, opts) => opts.copy(testingDir = td) }
      .text("Path to directory containing testing data files.")
  }

  def main(args: Array[String]): Unit = {
    val opts = parser.parse(args, Opts()) match {
      case Some(parsedOpts) => parsedOpts
      case _ => sys.error("Failed to parse opts.")
    }

    val trainingFilenames = traverseLabeledDataFiles(opts.trainingDir)
    val validationFilenames = traverseLabeledDataFiles(opts.validationDir)
    val testingFileNames = traverseUnlabeledDataFiles(opts.testingDir)

    val trainingData = trainingFilenames.map(DataPoint.fromFile)
    val validationData = validationFilenames.map(DataPoint.fromFile)
    val testingData = testingFileNames.map(DataPoint.fromFile)

    val wordIndex = WordIndex.fromDataSet(Array(trainingData, validationData).flatten)

    val trainingMatrix = buildMatrix(trainingData, wordIndex.wordOrdering)
    val validationMatrix = buildMatrix(validationData, wordIndex.wordOrdering)
    val testingMatrix = buildMatrix(testingData, wordIndex.wordOrdering)

    println(trainingMatrix.length, trainingMatrix.head.length)
    println(validationMatrix.length, validationMatrix.head.length)
    println(testingMatrix.length, testingMatrix.head.length)
  }
}