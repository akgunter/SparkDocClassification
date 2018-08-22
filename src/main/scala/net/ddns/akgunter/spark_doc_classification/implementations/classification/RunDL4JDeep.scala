package net.ddns.akgunter.spark_doc_classification.implementations.classification

import org.apache.spark.sql.SparkSession

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import net.ddns.akgunter.spark_doc_classification.util.DataFrameUtil.sparseDFFromCSVReadyDF
import net.ddns.akgunter.spark_doc_classification.util.DataSetUtil.{dataSetFromDL4JRDD, dl4jRDDFromSparseDataFrame}
import net.ddns.akgunter.spark_doc_classification.util.FileUtil.dataFrameFromProcessedDirectory


/*
Perform classification with a 3-layer DL4J deep neural network.
- Explicitly uses Spark to load the data sets but otherwise lets DL4J manage training
  and processing internally.
*/
object RunDL4JDeep extends ClassificationImplementation {
  override def run(trainingDir: String, validationDir: String, numEpochs: Int)(implicit spark: SparkSession): Unit = {
    // Use Spark to load the data sets
    logger.info("Loading data files...")
    val trainingDataCSVSourced = dataFrameFromProcessedDirectory(trainingDir)(spark)
    val validationDataCSVSourced = dataFrameFromProcessedDirectory(validationDir)(spark)

    logger.info("Creating data sets...")
    val trainingDataSparse = sparseDFFromCSVReadyDF(trainingDataCSVSourced)
    val validationDataSparse = sparseDFFromCSVReadyDF(validationDataCSVSourced)

    val Array(csvNumFeaturesCol, _, _, csvLabelCol) = trainingDataCSVSourced.columns
    val numFeatures = trainingDataCSVSourced.head.getAs[Int](csvNumFeaturesCol)
    val numClasses = trainingDataSparse.select(csvLabelCol).distinct.count.toInt

    val trainingRDD = dl4jRDDFromSparseDataFrame(trainingDataSparse, numClasses)
    val validationRDD = dl4jRDDFromSparseDataFrame(validationDataSparse, numClasses)

    val trainingDataSet = dataSetFromDL4JRDD(trainingRDD)
    val validationDataSet = dataSetFromDL4JRDD(validationRDD)


    logger.info(s"Configuring neural net with $numFeatures features and $numClasses classes...")
    val midLayerSize = numFeatures/2
    val nnConf = new NeuralNetConfiguration.Builder()
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.02))
      .l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numFeatures).nOut(midLayerSize).build)
      .layer(1, new DenseLayer.Builder().nIn(midLayerSize).nOut(numClasses).build)
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX).nIn(numClasses).nOut(numClasses).build)
      .pretrain(false)
      .backprop(true)
      .build

    val network = new MultiLayerNetwork(nnConf)
    network.init()
    network.setListeners(new ScoreIterationListener(10))

    logger.info("Training neural network...")
    0 until numEpochs foreach {
      epoch =>
        if (epoch % 5 == 0) logger.info(s"Running epoch $epoch...")
        network.fit(trainingDataSet)
    }


    logger.info("Evaluating performance...")
    val eval = new Evaluation()
    eval.eval(trainingDataSet.getLabels, trainingDataSet.getFeatureMatrix, network)
    logger.info(eval.stats)

    eval.eval(validationDataSet.getLabels, validationDataSet.getFeatureMatrix, network)
    logger.info(eval.stats)
  }
}
