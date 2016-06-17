(ns clj-neural.dl4j-images
  (:require [clojure.java.io :as io])
  (:import (org.canova.image.recordreader ImageRecordReader)
           (java.util LinkedList ArrayList Vector Iterator Random Date)
           (org.canova.api.split FileSplit)
           (java.io File)
           (org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder MultiLayerConfiguration Updater GradientNormalization)
           (org.deeplearning4j.nn.api OptimizationAlgorithm Layer)
           (org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder ConvolutionLayer ConvolutionLayer$Builder SubsamplingLayer SubsamplingLayer$Builder SubsamplingLayer$PoolingType DenseLayer RBM$Builder RBM$VisibleUnit RBM$HiddenUnit)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)
           (org.deeplearning4j.datasets.canova RecordReaderDataSetIterator)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.nd4j.linalg.dataset DataSet)
           (org.deeplearning4j.eval Evaluation)
           (org.deeplearning4j.nn.conf.layers.setup ConvolutionLayerSetup)
           (org.deeplearning4j.datasets.iterator DataSetPreProcessor)
           (org.deeplearning4j.util ModelSerializer)
           (org.nd4j.linalg.convolution Convolution$Type)
           (java.awt.image BufferedImage WritableRaster)
           (javax.imageio ImageIO)))


(defn load-net [nm]
  {:name nm
   :model
   (ModelSerializer/restoreMultiLayerNetwork (File. (str nm ".net")))})

(defn save-net [{^MultiLayerNetwork model :model nm :name}]
  (ModelSerializer/writeModel model (File. (str nm ".net")) true)
  (ModelSerializer/writeModel model (File. (str nm ".cnet")) false))

(def optimizations
  {:sgd OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
   :line OptimizationAlgorithm/LINE_GRADIENT_DESCENT
   :hessian OptimizationAlgorithm/HESSIAN_FREE
   :conjugate OptimizationAlgorithm/CONJUGATE_GRADIENT
   :lbfgs OptimizationAlgorithm/LBFGS})

(def updaters
  {:nesterovs Updater/NESTEROVS
   })

{:seed 123
 :optimization :sgd
 :iterations 1
 :learning 0.006
 :updater :nesterovs}



(defn single-layer-config [seed [sx sy d] out]
  {:name "single"
   :config
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
       (-> (DenseLayer$Builder.)
         (.nIn (* sx sy d))
         (.nOut 1000)
         (.activation "relu")
         (.weightInit WeightInit/XAVIER)
         (.build)))
     (.layer 1
       (-> (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
         (.nIn 1000)
         (.nOut out)
         (.activation "softmax")
         (.weightInit WeightInit/XAVIER)
         (.build)))
     (.pretrain false)
     (.backprop true))})

(defn two-layer-config [seed [sx sy d] out]
  {:name "twolayer"
   :config
   (->
     (NeuralNetConfiguration$Builder.)
     (.seed (int seed))
     (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
     (.activation "relu")
     (.weightInit WeightInit/XAVIER)
     (.iterations 1)
     (.learningRate 0.0015)
     (.updater Updater/NESTEROVS)
     (.momentum 0.98)
     (.regularization true)
     (.l2 (* 0.0015 0.005))
     (.list)
     (.layer 0
       (-> (DenseLayer$Builder.)
         (.nIn (* sx sy d))
         (.nOut 500)
         (.build)))
     (.layer 1
       (-> (DenseLayer$Builder.)
         (.nIn 500)
         (.nOut 100)
         (.build)))
     (.layer 2
       (-> (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
         (.nIn 100)
         (.nOut out)
         (.activation "softmax")
         (.build)))
     (.pretrain false)
     (.backprop true))})

(defn single-convnet-config [seed [sx sy d] out]
  (let [result
        (->
          (NeuralNetConfiguration$Builder.)
          (.seed (int seed))
          (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
          (.iterations 1)
          (.learningRate 0.0002)
          (.weightInit WeightInit/XAVIER)
          (.updater Updater/NESTEROVS)
          (.momentum 0.8)
          (.regularization true)
          (.l2 0.000015)
          (.list)
          (.layer 0
            (->
              (ConvolutionLayer$Builder. (int-array [5 5]) (int-array [1 1]))
              (.nIn 3)
              (.nOut 20)
              (.activation "identity")
              (.build)))
          (.layer 1
            (->
              (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
              (.kernelSize (int-array [2 2]))
              (.stride (int-array [1 1]))
              (.build)))
          (.layer 2
            (->
              (DenseLayer$Builder.)
              (.activation "relu")
              (.dropOut 0.5)
              (.nOut 300)
              (.build)))
          (.layer 3
            (->
              (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
              (.nOut out)
              (.activation "softmax")
              (.build)))
          (.pretrain false)
          (.backprop true)
          )]
    (ConvolutionLayerSetup. result sx sy d)
    {:name "single-conv"
     :config result}))

(defn dual-convnet-config [seed [sx sy d] out]
  (let [result
        (->
          (NeuralNetConfiguration$Builder.)
          (.seed (int seed))
          (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
          (.iterations 1)
          (.learningRate 0.000002)
          (.weightInit WeightInit/XAVIER)
          (.updater Updater/NESTEROVS)
          (.momentum 0.8)
          (.regularization true)
          (.l2 (* 0.000002 0.0015))
          (.list)
          (.layer 0
            (->
              (ConvolutionLayer$Builder. (int-array [5 5]))
              (.nIn 3)
              (.nOut 20)
              (.activation "identity")
              (.build)))
          (.layer 1
            (->
              (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
              (.kernelSize (int-array [2 2]))
              (.build)))
          (.layer 2
            (->
              (ConvolutionLayer$Builder. (int-array [5 5]))
              (.nIn 20)
              (.nOut 40)
              (.activation "identity")
              (.build)
              ))
          (.layer 3
            (->
              (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX )
              (.kernelSize (int-array [3 3]))
              (.build)))
          (.layer 4
            (->
              (DenseLayer$Builder.)
              (.activation "relu")
              (.nOut 100)
              (.dropOut 0.5)
              (.build)))
          (.layer 5
            (->
              (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
              (.nIn 100)
              (.nOut out)
              (.activation "softmax")
              (.build)))
          (.pretrain false)
          (.backprop true)
          )]
    (ConvolutionLayerSetup. result sx sy d)
    {:name "dual-conv-2"
     :config result}))

(def preprocessor
  (proxy [DataSetPreProcessor] []
    (preProcess [^DataSet dataset]
      (.divi (.getFeatureMatrix dataset) 255))))

(defonce net (atom nil))

(defn trainset* []
  (let [reader (ImageRecordReader. 36 36 3 true (ArrayList. ["b" "e" "w"]))
        fs (FileSplit. (File. "resources/training") (Random. 747347834))
        _ (.initialize reader fs)
        iter (RecordReaderDataSetIterator. reader 20 -1 3)]
    (.setPreProcessor iter preprocessor)
    iter))

(def trainset (memoize trainset*))

(defn testset* []
  (let [reader (ImageRecordReader. 36 36 3 true (ArrayList. ["b" "e" "w"]))
        fs (FileSplit. (File. "resources/testing") (Random. 947584738))
        _ (.initialize reader fs)
        iter (RecordReaderDataSetIterator. reader 20 -1 3)]
    (.setPreProcessor iter preprocessor)
    iter))

(def testset (memoize testset*))

(defn evaluate [modelconf]
  (let [test ^RecordReaderDataSetIterator (trainset)]
    (.reset test)
    (println "Evaluate:")
    (let [eval (Evaluation. 3)]
      (doseq [^DataSet test (iterator-seq test)]
        (.eval eval (.getLabels test) (.output (:model modelconf) (.getFeatureMatrix test))))

      (spit (str (:name modelconf) ".log")
        (pr-str
          {:time (Date.)
           :score (.score (:model modelconf))
           :accuracy (.accuracy eval)
           :precision (.precision eval)
           :recall (.recall eval)
           :f1 (.f1 eval)})
        :append true)

      (println (.stats eval)))))

(defn evaluate-training [modelconf]
  (let [test ^RecordReaderDataSetIterator (trainset)]
    (.reset test)
    (println "Evaluate on Training set:")
    (let [eval (Evaluation. 3)]
      (doseq [^DataSet test (iterator-seq test)]
        (.eval eval (.getLabels test) (.output (:model modelconf) (.getFeatureMatrix test))))

      (println (.stats eval)))))


(defn fit [modelconf epochs]
  (let [iter ^RecordReaderDataSetIterator (trainset)]
    (.reset iter)
    (doseq [e (range epochs)]
      (println "Epoch: " e)
      (.fit (:model modelconf) iter))
    (println "Done.")
    ))

(def stop-cycle (atom false))

(defn fit-eval-cycle [modelconf]
  (doto
    (Thread.
      #(do
        (while (not @stop-cycle)
          (time (fit modelconf 1))
          (time (evaluate modelconf))
          (save-net modelconf))
        (reset! stop-cycle false)))
    (.setDaemon true)
    (.start)))

(defn stop-fit []
  (reset! stop-cycle true))

(defn setup [conf-fn]
  (let [conf (conf-fn 123 [36 36 3] 3)
        ^MultiLayerNetwork model (MultiLayerNetwork. (.build (:config conf)))]
    (.init model)
    #_(.setListeners model [(HistogramIterationListener. 1)])
    (.setListeners model [(ScoreIterationListener. 1)])
    (spit (str (:name conf) ".log") "")
    (reset! net (assoc conf :model model))))
Layer
(defn img-layer-dump []
  (let [layers (.getInput (second (seq (.getLayers (:model @net)))))
        layer (.getRow layers 5)
        r-row (.getRow layer 0)
        g-row (.getRow layer 1)
        b-row (.getRow layer 2)
        [c w h] (seq (.shape layer))
        ^BufferedImage img (BufferedImage. w h 2)
        ^WritableRaster raster (.getRaster img)
        equiv (int-array (.length r-row))]
    (doseq [i (range (.length r-row))]
      (let [f #(int (max 0 (min 255 (* 255 (.getDouble % i)))))
            [r g b] [(f r-row) (f g-row) (f b-row)]]
        (println (bit-or (bit-shift-left r 16) (bit-shift-left g 8) b))
        (aset-int equiv i (bit-or (bit-shift-right r 16) (bit-shift-left g 8) b))))
    (.setDataElements raster 0 0 w h equiv)
    (ImageIO/write img "jpg" (File. "resources/layer.jpg"))))

(defn input-image-dump []
  (let [layers (.getInput (second (seq (.getLayers (:model @net)))))]))

#_(setup two-layer-config)

;; Goodzilla420
;; sundaay (instagram photo's)
;; crispybacon404
;; emdio
;; florinandrei
