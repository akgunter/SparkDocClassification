package net.ddns.akgunter.scala_classifier.document_classifier

import scopt._

import net.ddns.akgunter.scala_classifier.util.FileUtil._
import net.ddns.akgunter.scala_classifier.util.PreprocessingUtil._
import net.ddns.akgunter.scala_classifier.models.DataPoint
import net.ddns.akgunter.scala_classifier.models.WordIndex
import net.ddns.akgunter.scala_classifier.lib._

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

    val wordIndex = WordIndex.fromDataSet(Array(trainingData, validationData).flatten)

    val trainingMatrix = buildSparseMatrix(trainingData, wordIndex)
    val validationMatrix = buildSparseMatrix(validationData, wordIndex)
    val testingMatrix = buildSparseMatrix(testingData, wordIndex)

    println(trainingMatrix.length, trainingMatrix.head.length)
    println(validationMatrix.length, validationMatrix.head.length)
    println(testingMatrix.length, testingMatrix.head.length)

    val idfVector = calcIDF(trainingData)
  }
}