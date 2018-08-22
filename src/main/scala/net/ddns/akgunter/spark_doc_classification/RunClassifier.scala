package net.ddns.akgunter.spark_doc_classification

import java.nio.file.Paths

import net.ddns.akgunter.spark_doc_classification.implementations._
import net.ddns.akgunter.spark_doc_classification.sparkml_processing.Preprocessor
import net.ddns.akgunter.spark_doc_classification.spark.CanSpark
import net.ddns.akgunter.spark_doc_classification.util.FileUtil._

object RunMode extends Enumeration {
  val NOOP, PREPROCESS, SPARKML, DL4J, DL4JDEEP, DL4JSPARK = Value
}

object RunClassifier extends CanSpark {

  case class Config(runMode: RunMode.Value = RunMode.NOOP,
                    inputDataDir: String = "",
                    outputDataDir: String = "",
                    numEpochs: Int = 0)

  def getOptionParser: scopt.OptionParser[Config] = {
    new scopt.OptionParser[Config]("DocClassifier") {

      val preprocessorArgs = Seq(
        arg[String]("<inputDataDir>")
          .action( (x, c) => c.copy(inputDataDir = x))
          .text("The file path to the input data"),
        arg[String]("<outputDataDir>")
          .action( (x, c) => c.copy(outputDataDir = x) )
          .text("The file path to write preprocessed data to")
      )

      val learningImplArgs = Seq(
        arg[String]("<inputDataDir>")
          .action( (x, c) => c.copy(inputDataDir = x))
          .text("The file path to the input data"),
        arg[Int]("<numEpochs>")
          .action( (x, c) => c.copy(numEpochs = x) )
          .text("The number of epochs to run")
      )

      cmd(RunMode.PREPROCESS.toString)
        .action( (_, c) => c.copy(runMode = RunMode.PREPROCESS) )
        .text("Run the program in PREPROCESS mode")
        .children(preprocessorArgs: _*)

      cmd(RunMode.SPARKML.toString)
        .action( (_, c) => c.copy(runMode = RunMode.SPARKML) )
        .text("Run the program in SPARKML mode")
        .children(learningImplArgs: _*)

      cmd(RunMode.DL4J.toString)
        .action( (_, c) => c.copy(runMode = RunMode.DL4J) )
        .text("Run the program in DL4J mode")
        .children(learningImplArgs: _*)

      cmd(RunMode.DL4JDEEP.toString)
        .action( (_, c) => c.copy(runMode = RunMode.DL4JDEEP) )
        .text("Run the program in DL4JDEEP mode")
        .children(learningImplArgs: _*)

      cmd(RunMode.DL4JSPARK.toString)
        .action( (_, c) => c.copy(runMode = RunMode.DL4JSPARK) )
        .text("Run the program in DL4JSPARK mode")
        .children(learningImplArgs: _*)
    }
  }

  def main(args: Array[String]): Unit = {
    val parser = getOptionParser
    parser.parse(args, Config()) match {
      case Some(config) =>
        val trainingDir = Paths.get(config.inputDataDir, TrainingDirName).toString
        val validationDir = Paths.get(config.inputDataDir, ValidationDirName).toString

        config.runMode match {
          case RunMode.PREPROCESS =>
            withSpark() { implicit spark => Preprocessor.preprocessData(trainingDir, validationDir, config.outputDataDir) }
          case RunMode.SPARKML =>
            withSpark() { implicit spark => SparkMLImplementation.run(trainingDir, validationDir, config.numEpochs) }
          case RunMode.DL4J =>
            withSpark() { implicit spark => DL4JImplementation.run(trainingDir, validationDir, config.numEpochs) }
          case RunMode.DL4JDEEP =>
            withSpark() { implicit spark => DL4JDeepImplementation.run(trainingDir, validationDir, config.numEpochs) }
          case RunMode.DL4JSPARK =>
            withSpark() { implicit spark => DL4JSparkImplementation.run(trainingDir, validationDir, config.numEpochs) }
          case RunMode.NOOP =>
            ()
        }
      case _ => // args are bad, error message will be displayed
    }
  }
}