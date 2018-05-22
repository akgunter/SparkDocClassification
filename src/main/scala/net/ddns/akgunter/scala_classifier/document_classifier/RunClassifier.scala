package net.ddns.akgunter.scala_classifier.document_classifier

import scopt._

import net.ddns.akgunter.scala_classifier.util.DataPoint
import net.ddns.akgunter.scala_classifier.util.FileUtil._
import net.ddns.akgunter.scala_classifier.util.PreprocessingUtil._

object RunClassifier {

  def main(args: Array[String]): Unit = {
    val trainingDir = "./data/Training"
    val validationDir = "./data/Validation"
    val testingDir = "./data/Testing"

    val trainingFilenames = traverseLabeledDataFiles(trainingDir)
    val validationFilenames = traverseLabeledDataFiles(validationDir)
    val testingFileNames = traverseUnlabeledDataFiles(testingDir)

    val trainingData = trainingFilenames.map(DataPoint.fromFile)
    val validationData = validationFilenames.map(DataPoint.fromFile)
    val testingData = testingFileNames.map(DataPoint.fromFile)

    val (wordLookup, wordOrdering) = buildWordIndex(Array(trainingData, validationData))

    val trainingMatrix = buildMatrix(trainingData, wordOrdering)
    val validationMatrix = buildMatrix(validationData, wordOrdering)
    val testingMatrix = buildMatrix(testingData, wordOrdering)

    println(trainingMatrix.length, trainingMatrix(0).length)
    println(validationMatrix.length, validationMatrix(0).length)
    println(testingMatrix.length, testingMatrix(0).length)
  }
}