package net.ddns.akgunter.spark_doc_classification.implementations.classification

import org.apache.spark.sql.SparkSession

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import net.ddns.akgunter.spark_doc_classification.util.DataFrameUtil.sparseDFFromCSVReadyDF
import net.ddns.akgunter.spark_doc_classification.util.DataSetUtil.dl4jRDDFromSparseDataFrame
import net.ddns.akgunter.spark_doc_classification.util.FileUtil.dataFrameFromProcessedDirectory


/*
Use DL4J's Spark integration to train a 2-layer neural network.
*/
object RunDL4JSpark extends ClassificationImplementation {
  override def run(trainingDir: String, validationDir: String, numEpochs: Int)(implicit spark: SparkSession): Unit = {
    logger.info("Loading data files...")
    val trainingDataCSVSourced = dataFrameFromProcessedDirectory(trainingDir)
    val validationDataCSVSourced = dataFrameFromProcessedDirectory(validationDir)


    logger.info("Creating data sets...")
    val trainingDataSparse = sparseDFFromCSVReadyDF(trainingDataCSVSourced)
    val validationDataSparse = sparseDFFromCSVReadyDF(validationDataCSVSourced)

    val Array(csvNumFeaturesCol, _, _, csvLabelCol) = trainingDataCSVSourced.columns
    val numFeatures = trainingDataCSVSourced.head.getAs[Int](csvNumFeaturesCol)
    val numClasses = trainingDataSparse.select(csvLabelCol).distinct.count.toInt

    val trainingRDD = dl4jRDDFromSparseDataFrame(trainingDataSparse, numClasses)
    val validationRDD = dl4jRDDFromSparseDataFrame(validationDataSparse, numClasses)


    logger.info(s"Configuring neural net with $numFeatures features and $numClasses classes...")
    val nnConf = new NeuralNetConfiguration.Builder()
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.02))
      .l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numFeatures).nOut(numClasses).build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX).nIn(numClasses).nOut(numClasses).build)
      .pretrain(false)
      .backprop(true)
      .build

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(1)
      .rddTrainingApproach(RDDTrainingApproach.Export)
      .exportDirectory("/tmp/alex-spark/SparkLearning")
      .build

    val sparkNet = new SparkDl4jMultiLayer(spark.sparkContext, nnConf, trainingMaster)


    logger.info("Training neural network...")
    0 until numEpochs foreach {
      epoch =>
        if (epoch % 5 == 0) logger.info(s"Running epoch $epoch...")
        sparkNet.fit(trainingRDD)
    }


    logger.info("Evaluating performance...")
    val trainingEval = sparkNet.doEvaluation(trainingRDD, new Evaluation(numClasses), 4)
    logger.info(trainingEval.stats)

    val validationEval = sparkNet.doEvaluation(validationRDD, new Evaluation(numClasses), 4)
    logger.info(validationEval.stats)
  }
}
