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

    println("Loading data...")
    val trainingFilenames = traverseLabeledDataFiles(trainingDir)
    val validationFilenames = traverseLabeledDataFiles(validationDir)
    val testingFileNames = traverseUnlabeledDataFiles(testingDir)

    val trainingData = trainingFilenames.map(DataPoint.fromFile)
    val validationData = validationFilenames.map(DataPoint.fromFile)
    val testingData = testingFileNames.map(DataPoint.fromFile)

    val trainingLabels = trainingFilenames.map(getLabelFromFilePath)
    val validationLabels = validationFilenames.map(getLabelFromFilePath)

    println("Building word index...")
    val wordIndex = WordIndex.fromDataSet(Array(trainingData, validationData).flatten)

    println(calcTF(vectorize(trainingData.head, wordIndex)))
    
    /*
    println("Building matrices...")
    val trainingMatrix = buildSparseMatrix(trainingData, wordIndex)
    val validationMatrix = buildSparseMatrix(validationData, wordIndex)
    val testingMatrix = buildSparseMatrix(testingData, wordIndex)

    println("Calculating TF-IDF values...")
    val idfVector = calcIDF(trainingMatrix ++ validationMatrix)
    val trainingProc = calcTFIDF(trainingMatrix, idfVector)
    val validationProc = calcTFIDF(validationMatrix, idfVector)
    val testingProc = calcTFIDF(testingMatrix, idfVector)
    */
  }
}