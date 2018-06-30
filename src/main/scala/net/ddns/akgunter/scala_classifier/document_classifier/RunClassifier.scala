package net.ddns.akgunter.scala_classifier.document_classifier


import scala.util.Random

import org.apache.spark.sql.SparkSession

import net.ddns.akgunter.scala_classifier.lib.{SparseMatrix, SparseVector}
import net.ddns.akgunter.scala_classifier.util.FileUtil._
import net.ddns.akgunter.scala_classifier.util.PreprocessingUtil._
import net.ddns.akgunter.scala_classifier.models.DataPoint
import net.ddns.akgunter.scala_classifier.models.WordIndex
import net.ddns.akgunter.scala_classifier.svm.CSVM
import net.ddns.akgunter.scala_classifier.spark.CanSpark

object RunClassifier extends CanSpark {

  def dataProcessing(): Unit = {
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

    println("Building matrices...")
    val trainingMatrix = buildSparseMatrix(trainingData, wordIndex)
    val validationMatrix = buildSparseMatrix(validationData, wordIndex)
    val testingMatrix = buildSparseMatrix(testingData, wordIndex)

    println("Calculating TF-IDF values...")
    val idfVector = calcIDF(trainingMatrix ++ validationMatrix)
    val trainingProc = calcTFIDF(trainingMatrix, idfVector)
    val validationProc = calcTFIDF(validationMatrix, idfVector)
    val testingProc = calcTFIDF(testingMatrix, idfVector)
  }

  def dataProcessingOld(): Unit = {
    val testFilename = "./data/test_file.res"
    val testData = DataPoint.fromFile(testFilename)
    val testIndex = WordIndex.fromDataSet(Array(testData))

    val testRow = vectorize(testData, testIndex)
    val testMatrix = buildSparseMatrix(Array(testData), testIndex)

    val tfRow = calcTF(testRow)
    val idfVector = calcIDF(testMatrix)
    val tfidfMatrix = calcTFIDF(testMatrix)

    println(testData)
    println(testIndex.wordOrdering.mkString(", "))
    println(testIndex.wordCounts)
    println(testRow)
    println(tfRow)
    println(idfVector)
    println(tfidfMatrix)
  }

  def CSVMTest(): Unit = {
    val maxVal = 10
    val d = 1
    val numPoints = 10

    val negPoints = Array.fill(numPoints / 2)(Array(Random.nextDouble(), Random.nextDouble() * (maxVal/2 - d)))
    val posPoints = Array.fill(numPoints / 2)(Array(Random.nextDouble(), Random.nextDouble() * (maxVal/2 - d) + maxVal/2 + d))

    val negLabels = negPoints.map(_ => "neg")
    val posLabels = posPoints.map(_ => "pos")

    val points = SparseMatrix.fromMatrix[Double](negPoints ++ posPoints)
    val labels = negLabels ++ posLabels

    val csvm = CSVM.fromData(points, labels)

    println(points)
    println(csvm.sampleWeights)
  }

  def dl4jTest(implicit spark: SparkSession): Unit = {

  }


  def main(args: Array[String]): Unit = {
    val testArg = args(0)

    if (testArg == "1") dataProcessing()
    else if (testArg == "2") dataProcessingOld()
    else if (testArg == "3") CSVMTest()
    else if (testArg == "4") withSpark() { spark => dl4jTest(spark) }
  }
}