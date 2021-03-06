package net.ddns.akgunter.spark_doc_classification

import java.nio.file.Paths

import net.ddns.akgunter.spark_doc_classification.implementations.classification._
import net.ddns.akgunter.spark_doc_classification.implementations.preprocessing._
import net.ddns.akgunter.spark_doc_classification.spark.CanSpark
import net.ddns.akgunter.spark_doc_classification.util.FileUtil


object RunClassifier extends CanSpark {

  object RunMode extends Enumeration {
    val NOOP, PREPROCESS, CLASSIFY = Value
  }

  object PreprocessMode extends Enumeration {
    val NOOP, IDFCHI2 = Value
  }

  object ClassificationMode extends Enumeration {
    val NOOP, SPARKML, DL4J, DL4JDEEP, DL4JSPARK = Value
  }

  case class Opts(
                   runMode: RunMode.Value = RunMode.NOOP,
                   preprocessMode: PreprocessMode.Value = PreprocessMode.NOOP,
                   classificationMode: ClassificationMode.Value = ClassificationMode.NOOP,
                   inputDataDir: String = "",
                   outputDataDir: String = "",
                   numEpochs: Int = 0
                   )

  def getOptionParser: scopt.OptionParser[Opts] = {
    new scopt.OptionParser[Opts]("DocClassifier") {
      /*
      Create a sequence of child arguments for use with a preprocessing command
       */
      def createPreprocessingChildArguments: Seq[scopt.OptionDef[_, RunClassifier.Opts]] = {
        Seq(
          arg[String]("<inputDataDir>")
            .action( (x, c) => c.copy(inputDataDir = x))
            .text("The file path to the input data"),
          arg[String]("<outputDataDir>")
            .action( (x, c) => c.copy(outputDataDir = x) )
            .text("The file path to write preprocessed data to")
        )
      }

      /*
      Create a sequence of child arguments for use with a classification command
       */
      def createClassificationChildArguments: Seq[scopt.OptionDef[_, RunClassifier.Opts]] = {
        Seq(
          arg[String]("<inputDataDir>")
            .action( (x, c) => c.copy(inputDataDir = x))
            .text("The file path to the input data"),
          arg[Int]("<numEpochs>")
            .action( (x, c) => c.copy(numEpochs = x) )
            .text("The number of epochs to run")
        )
      }

      // The allowed preprocessing commands
      val preproccessingArguments = Seq(
        cmd(PreprocessMode.IDFCHI2.toString)
          .action( (_, c) => c.copy(preprocessMode = PreprocessMode.IDFCHI2) )
          .text("Preprocess the data with the IDF and Chi2Sel transformations")
          .children(createPreprocessingChildArguments: _*)
      )

      // The allowed classification commands
      val classificationArguments = Seq(
        cmd(ClassificationMode.SPARKML.toString)
          .action( (_, c) => c.copy(classificationMode = ClassificationMode.SPARKML) )
          .text("Classify the data using a Spark MLlib perceptron")
          .children(createClassificationChildArguments: _*),
        cmd(ClassificationMode.DL4J.toString)
          .action( (_, c) => c.copy(classificationMode = ClassificationMode.DL4J) )
          .text("Classify the data using a DL4J perceptron")
          .children(createClassificationChildArguments: _*),
        cmd(ClassificationMode.DL4JDEEP.toString)
          .action( (_, c) => c.copy(classificationMode = ClassificationMode.DL4JDEEP) )
          .text("Classify the data using a DL4J deep neural network")
          .children(createClassificationChildArguments: _*),
        cmd(ClassificationMode.DL4JSPARK.toString)
          .action( (_, c) => c.copy(classificationMode = ClassificationMode.DL4JSPARK) )
          .text("Classify the data using a DL4J perceptron and Spark integration")
          .children(createClassificationChildArguments: _*)
      )

      // The base run-mode command to preprocess data
      cmd(RunMode.PREPROCESS.toString)
        .action( (_, c) => c.copy(runMode = RunMode.PREPROCESS) )
        .text("Preprocess the data using the specified pipeline")
        .children(preproccessingArguments: _*)

      // The base run-mode command to classify data that has been preprocessed
      cmd(RunMode.CLASSIFY.toString)
        .action( (_, c) => c.copy(runMode = RunMode.CLASSIFY) )
        .text("Classify the preprocessed data using the specified classifier")
        .children(classificationArguments: _*)
    }
  }

  def main(args: Array[String]): Unit = {
    val preprocessImplementations = Map(
      PreprocessMode.IDFCHI2 -> RunIDFChi2Preprocessing
    )

    val classificationImplementations = Map(
      ClassificationMode.SPARKML -> RunSparkMLlib,
      ClassificationMode.DL4J -> RunDL4J,
      ClassificationMode.DL4JDEEP -> RunDL4JDeep,
      ClassificationMode.DL4JSPARK -> RunDL4JSpark
    )

    val parser = getOptionParser

    parser.parse(args, Opts()) match {
      case Some(config) =>
        val trainingDir = Paths.get(config.inputDataDir, FileUtil.TrainingDirName).toString
        val validationDir = Paths.get(config.inputDataDir, FileUtil.ValidationDirName).toString

        config.runMode match {
          case RunMode.NOOP => parser.showUsage()
          case RunMode.PREPROCESS =>
            config.preprocessMode match {
              case PreprocessMode.NOOP => parser.showUsage()
              case _ =>
                withSpark() {
                  spark =>
                    val implementation = preprocessImplementations(config.preprocessMode)
                    implementation.run(trainingDir, validationDir, config.outputDataDir)(spark)
                }
            }
          case RunMode.CLASSIFY =>
            config.classificationMode match {
              case ClassificationMode.NOOP => parser.showUsage()
              case _ =>
                withSpark() {
                  spark =>
                    val implementation = classificationImplementations(config.classificationMode)
                    implementation.run(trainingDir, validationDir, config.numEpochs)(spark)
                }
            }
        }
      case _ =>
    }
  }
}