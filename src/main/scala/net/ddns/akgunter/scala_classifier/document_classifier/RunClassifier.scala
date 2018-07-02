package net.ddns.akgunter.scala_classifier.document_classifier

import java.io.File
import java.nio.file.Paths
import scala.util.Random

import org.apache.spark.sql.SparkSession

import net.ddns.akgunter.scala_classifier.lib.SparseMatrix
import net.ddns.akgunter.scala_classifier.models.DataPoint
import net.ddns.akgunter.scala_classifier.models.WordIndex
import net.ddns.akgunter.scala_classifier.spark.CanSpark
import net.ddns.akgunter.scala_classifier.svm.CSVM
import net.ddns.akgunter.scala_classifier.util.FileUtil._
import net.ddns.akgunter.scala_classifier.util.PreprocessingUtil._

object RunClassifier extends CanSpark {

  def dataProcessing(dataDir: String): Unit = {
    val trainingDir = Paths.get(dataDir, "Training").toString
    val validationDir = Paths.get(dataDir, "Validation").toString
    val testingDir = Paths.get(dataDir, "Testing").toString

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

  def dataProcessingOld(dataDir: String): Unit = {
    val testFilename = Paths.get(dataDir, "test_file.res").toAbsolutePath.toString
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

  def CSVMTest(dataDir: String): Unit = {
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

  def dl4jTest(dataDir: String)(implicit spark: SparkSession): Unit = {
    val trainingDir = Paths.get(dataDir, "Training").toString
    val validationDir = Paths.get(dataDir, "Validation").toString
    val testingDir = Paths.get(dataDir, "Testing").toString

    logger.info("Loading data...")
    val trainingData = dataFrameFromDirectory(trainingDir, training = true)
    val validationData = dataFrameFromDirectory(validationDir, training = true)
    val testingData = dataFrameFromDirectory(testingDir, training = false)

    logger.info(
      s"""Loaded:\n
         |\t${trainingData.count} training records
         |\t${validationData.count} validation records
         |\t${testingData.count} testing records
       """.stripMargin)
  }


  def main(args: Array[String]): Unit = {
    val dataDir = args(0)
    val testArg = args(1)
    println(s"Running test $testArg in dataDir=$dataDir")

    if (testArg == "1") dataProcessing(dataDir)
    else if (testArg == "2") dataProcessingOld(dataDir)
    else if (testArg == "3") CSVMTest(dataDir)
    else if (testArg == "4") withSpark() { spark => dl4jTest(dataDir)(spark) }
    else println("Argument not recognized")
  }
}