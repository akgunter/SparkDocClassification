package org.deeplearning4j.spark.parameterserver.training;

import lombok.NonNull;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.transport.Transport;

public class CustomTrainingMaster extends SharedTrainingMaster {
    public CustomTrainingMaster(@NonNull VoidConfiguration voidConfiguration, Integer numWorkers,
                                RDDTrainingApproach rddTrainingApproach, StorageLevel storageLevel, boolean collectTrainingStats,
                                RepartitionStrategy repartitionStrategy, Repartition repartition, double threshold,
                                double minThreshold, double thresholdStep, double stepTrigger, int stepDelay, int shakeFrequency,
                                int batchSizePerWorker, long debugLongerIterations, int numWorkersPerNode) {

        super(voidConfiguration, numWorkers,
              rddTrainingApproach, storageLevel, collectTrainingStats,
              repartitionStrategy, repartition, threshold,
              minThreshold, thresholdStep, stepTrigger, stepDelay, shakeFrequency,
              batchSizePerWorker, debugLongerIterations, numWorkersPerNode);
    }


    public int getBatchSizePerWorker() {
        return this.batchSizePerWorker;
    }

    public int getRDDDataSetNumExamples() {
        return this.rddDataSetNumExamples;
    }

    public int getNumObjectsEachWorker() {
        return this.numObjectsEachWorker(this.rddDataSetNumExamples);
    }

    public static class Builder {
        protected double threshold = 1e-3;
        protected double thresholdStep = 1e-5;
        protected double minThreshold = 1e-5;
        protected double stepTrigger = 0.05;
        protected int stepDelay = 50;
        protected int shakeFrequency = 0;
        protected Repartition repartition = Repartition.Always;
        protected RepartitionStrategy repartitionStrategy = RepartitionStrategy.Balanced;
        protected StorageLevel storageLevel = StorageLevel.MEMORY_ONLY_SER();
        protected StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
        protected VoidConfiguration voidConfiguration;
        protected RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
        protected long rngSeed;
        protected String exportDirectory = null;
        protected Integer numWorkers;
        protected boolean collectTrainingStats;
        protected Transport transport;
        protected int batchSize;
        protected long debugLongerIterations = 0L;
        protected int numWorkersPerNode = -1;


        public Builder(int rddDataSetNumExamples) {
            this(1e-3, rddDataSetNumExamples);
        }

        public Builder(@NonNull VoidConfiguration voidConfiguration, int rddDataSetNumExamples) {
            this(voidConfiguration, 1e-3, rddDataSetNumExamples);
        }

        public Builder(double threshold, int rddDataSetNumExamples) {
            this(VoidConfiguration.builder().executionMode(ExecutionMode.MANAGED).forcedRole(NodeRole.SHARD)

                            // we're setting controller to Spark Master, if it's null - that's ok for now.
                            .controllerAddress(System.getenv("SPARK_PUBLIC_DNS")).build(), null, threshold,
                    rddDataSetNumExamples);
        }

        public Builder(@NonNull VoidConfiguration voidConfiguration, double threshold, int rddDataSetNumExamples) {
            this(voidConfiguration, null, threshold, rddDataSetNumExamples);
        }

        /**
         *
         * @param voidConfiguration ParameterServer configuration POJO
         * @param numWorkers
         * @param threshold Update sharing threshold
         * @param rddDataSetNumExamples
         */
        public Builder(@NonNull VoidConfiguration voidConfiguration, Integer numWorkers, double threshold,
                       int rddDataSetNumExamples) {
            this.threshold = threshold;
            this.voidConfiguration = voidConfiguration;

            // we're enforcing managed mode in all cases here
            this.voidConfiguration.setExecutionMode(ExecutionMode.MANAGED);
        }

        /**
         * Enable/disable collection of training statistics
         * @param reallyConnect
         * @return
         */
        public CustomTrainingMaster.Builder collectTrainingStats(boolean reallyConnect) {
            this.collectTrainingStats = reallyConnect;
            return this;
        }

        /**
         * This parameter defines when repartition is applied (if applied)
         * @param repartition
         * @return
         */
        public CustomTrainingMaster.Builder repartitionData(Repartition repartition) {
            this.repartition = repartition;
            return this;
        }

        /**
         * Used in conjunction with {@link #repartitionData(Repartition)} (which defines <i>when</i> repartitioning should be
         * conducted), repartitionStrategy defines <i>how</i> the repartitioning should be done. See {@link RepartitionStrategy}
         * for details
         *
         * @param repartitionStrategy Repartitioning strategy to use
         */
        public CustomTrainingMaster.Builder repartitionStrategy(RepartitionStrategy repartitionStrategy) {
            this.repartitionStrategy = repartitionStrategy;
            return this;
        }

        /**
         * Set the storage level for {@code RDD<DataSet>}s.<br>
         * Default: StorageLevel.MEMORY_ONLY_SER() - i.e., store in memory, in serialized form<br>
         * To use no RDD persistence, use {@code null}<br>
         * <p>
         * <b>Note</b>: Spark's StorageLevel.MEMORY_ONLY() and StorageLevel.MEMORY_AND_DISK() can be problematic when
         * it comes to off-heap data (which DL4J/ND4J uses extensively). Spark does not account for off-heap memory
         * when deciding if/when to drop blocks to ensure enough free memory; consequently, for DataSet RDDs that are
         * larger than the total amount of (off-heap) memory, this can lead to OOM issues. Put another way: Spark counts
         * the on-heap size of DataSet and INDArray objects only (which is negligible) resulting in a significant
         * underestimate of the true DataSet object sizes. More DataSets are thus kept in memory than we can really afford.
         *
         * @param storageLevel Storage level to use for DataSet RDDs
         */
        public CustomTrainingMaster.Builder storageLevel(StorageLevel storageLevel) {
            this.storageLevel = storageLevel;
            return this;
        }

        /**
         * The approach to use when training on a {@code RDD<DataSet>} or {@code RDD<MultiDataSet>}.
         * Default: {@link RDDTrainingApproach#Export}, which exports data to a temporary directory first
         *
         * @param rddTrainingApproach Training approach to use when training from a {@code RDD<DataSet>} or {@code RDD<MultiDataSet>}
         */
        public CustomTrainingMaster.Builder rddTrainingApproach(RDDTrainingApproach rddTrainingApproach) {
            this.rddTrainingApproach = rddTrainingApproach;
            return this;
        }

        /**
         * When {@link #rddTrainingApproach(RDDTrainingApproach)} is set to {@link RDDTrainingApproach#Export} (as it is by default)
         * the data is exported to a temporary directory first.
         * <p>
         * Default: null. -> use {hadoop.tmp.dir}/dl4j/. In this case, data is exported to {hadoop.tmp.dir}/dl4j/SOME_UNIQUE_ID/<br>
         * If you specify a directory, the directory {exportDirectory}/SOME_UNIQUE_ID/ will be used instead.
         *
         * @param exportDirectory Base directory to export data
         */
        public CustomTrainingMaster.Builder exportDirectory(String exportDirectory) {
            this.exportDirectory = exportDirectory;
            return this;
        }

        /**
         * Random number generator seed, used mainly for enforcing repeatable splitting on RDDs
         * Default: no seed set (i.e., random seed)
         *
         * @param rngSeed RNG seed
         * @return
         */
        public CustomTrainingMaster.Builder rngSeed(long rngSeed) {
            this.rngSeed = rngSeed;
            return this;
        }

        /**
         * Threshold for updates encoding
         *
         * Default value: 1e-3
         * @param threshold
         * @return
         */
        public CustomTrainingMaster.Builder updatesThreshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * Once update with given threshold become too sparse, threshold will be decreased by thresholdStep, but not below minimum threshold
         *
         * Default value: 1e-5
         * @param threshold
         * @return
         */
        public CustomTrainingMaster.Builder minUpdatesThreshold(double threshold) {
            this.minThreshold = threshold;
            return this;
        }

        /**
         * Step size for threshold decay
         *
         * Default value: 1e-5
         * @param step
         * @return
         */
        public CustomTrainingMaster.Builder thresholdStep(double step) {
            if (step < 0.0)
                throw new DL4JInvalidConfigException("shakeFrequency should be non-negative value");

            this.thresholdStep = step;
            return this;
        }

        /**
         * Target sparsity/dense level, when threshold step will happen. i.e. 5 value = 5% of original updates size.
         *
         * Default value: 0.05
         * @param step
         * @return
         */
        public CustomTrainingMaster.Builder stepTrigger(double step) {
            if (step < 0.0 || step > 100.0)
                throw new DL4JInvalidConfigException("stepTrigger value should be in range of 0..100");

            return this;
        }

        /**
         * Wait at least X iterations between applying threshold decay
         *
         * Default value: 50
         * @param step
         * @return
         */
        public CustomTrainingMaster.Builder stepDelay(int step) {
            this.stepDelay = step;
            return this;
        }

        /**
         * During NN training, each X iterations, executors will send encoded dense updates with lower threshold.
         * Please note: If you'll set this value too low (i.e. 1) - it might lead to worse performance
         *
         * Default value: 0 (disabled)
         * @param frequency
         * @return
         */
        public CustomTrainingMaster.Builder shakeFrequency(int frequency) {
            if (frequency < 0)
                throw new DL4JInvalidConfigException("shakeFrequency should be non-negative value");

            if (frequency == 1)
                log.warn("shakeFrequency of 1 means that all updates will be sparse, and might lead to worse performance");

            this.shakeFrequency = frequency;
            return this;
        }

        /**
         * Batch size value,  used for repartition purposes
         *
         * @param batchSize
         * @return
         */
        public CustomTrainingMaster.Builder batchSizePerWorker(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * This method allows to configure number of trainer threads per cluster node.
         *
         *
         * Default value: -1, which defines automated number of workers selection, based on hardware present in system
         *
         * @param numWorkers
         * @return
         */
        public CustomTrainingMaster.Builder workersPerNode(int numWorkers) {
            if (numWorkers < 1)
                numWorkers = -1;

            this.numWorkersPerNode = numWorkers;
            return this;
        }

        /**
         * This method allows you to artificially extend iteration time using Thread.sleep() for a given time.
         *
         * PLEASE NOTE: Never use that option in production environment. It's suited for debugging purposes only.
         *
         * @param timeMs
         * @return
         */
        @Deprecated
        public CustomTrainingMaster.Builder debugLongerIterations(long timeMs) {
            if (timeMs < 0)
                timeMs = 0L;
            this.debugLongerIterations = timeMs;
            return this;
        }

        /**
         * Optional method: Transport implementation to be used as TransportType.CUSTOM for VoidParameterAveraging method
         *
         * @param transport
         * @return
         */
        public CustomTrainingMaster.Builder transport(Transport transport) {
            this.transport = transport;
            return this;
        }

        public CustomTrainingMaster build() {
            CustomTrainingMaster master = new CustomTrainingMaster(voidConfiguration, numWorkers, rddTrainingApproach,
                    storageLevel, collectTrainingStats, repartitionStrategy, repartition, threshold,
                    minThreshold, thresholdStep, stepTrigger, stepDelay, shakeFrequency, batchSize,
                    debugLongerIterations, numWorkersPerNode);
            if (transport != null)
                master.transport = this.transport;

            return master;
        }
    }
}
