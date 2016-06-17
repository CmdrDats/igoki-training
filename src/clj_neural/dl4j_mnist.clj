(ns clj-neural.dl4j-mnist
  (:import (org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator)
           (org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder Updater)
           (org.deeplearning4j.nn.api OptimizationAlgorithm)
           (org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.deeplearning4j.eval Evaluation)
           (org.nd4j.linalg.dataset.api DataSet)
           (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)))

(defn config [seed rows cols out]
  (->
    (NeuralNetConfiguration$Builder.)
    (.seed (int seed))
    (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
    (.iterations 1)
    (.learningRate 0.006)
    (.updater Updater/NESTEROVS)
    (.momentum 0.9)
    (.regularization true)
    (.l2 1e-4)
    (.list)
    (.layer 0
      (->
        (DenseLayer$Builder.)
        (.nIn (* rows cols))
        (.nOut 1000)
        (.activation "relu")
        (.weightInit WeightInit/XAVIER)
        (.build)))
    (.layer 1
      (->
        (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
        (.nIn 1000)
        (.nOut out)
        (.activation "softmax")
        (.weightInit WeightInit/XAVIER)
        (.build)))
    (.pretrain false)
    (.backprop true)
    (.build)))

(def net (atom nil))

(defn evaluate [model ^long out ^long batch ^long seed]
  (let [mnist-test (MnistDataSetIterator. batch false seed)]
    (println "Evaluate:")
    (let [eval (Evaluation. out)]
      (doseq [^DataSet test (iterator-seq mnist-test)]
        (.eval eval (.getLabels test) (.output model (.getFeatureMatrix test))))
      (println (.stats eval)))))

(defn train [model epoch batch seed]
  (let [mnist-train (MnistDataSetIterator. (int batch) true (int seed))]
    (println "Training:")
    (doseq
      [i (range epoch)]
      (println "Epoch: " i)
      (.fit model mnist-train))))

(defn mnist []
  (let [rows 28
        cols 28
        out 10
        batch 128
        seed 123
        epoch 15

        conf (config seed rows cols out)
        model (MultiLayerNetwork. conf)
        ]
    (.init model)
    (.setListeners model [(ScoreIterationListener. 1)])
    (train model epoch batch seed)

    (reset! net model)
    (evaluate model out batch seed)))

