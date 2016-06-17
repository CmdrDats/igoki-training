(ns clj-neural.core
  (:use 
    [clojure.core.matrix]
    [clojure.core.matrix.operators]
    )
  (:require
    [clj-neural.mnist :as mnist])
  (:import (java.util Random)))

(def generator (Random.))

(defn bias-shape [biases addmat]
  (let [e (second (shape addmat))]
    (mapv
      (fn [b]
        (mapv (fn [_] b) (range e)))
      biases)))

(defn network [sizes]
  (let [num-layers (count sizes)
        biases
        (mapv
          (fn [y]
            (matrix
              (vec (for [_ (range y)] [(rand)]))))
          (drop 1 sizes))
        weights
        (mapv
          (fn [x y]
            (matrix
              (vec (for [_ (range x)]
                     (vec (for [_ (range y)]
                            (.nextGaussian generator)))))))
          (butlast sizes) (drop 1 sizes))]
    {:num-layers (count sizes)
     :sizes sizes
     :biases biases
     :weights weights}))

(defn feedforward [{:keys [biases weights] :as network} a]
  (reduce
    (fn [ai [b w]]
      (let [d (dot w (bias-shape ai w))]
        (logistic (+ d b))))
    a (mapv vector biases weights)))

(defn cost-derivative [activations y]
  (- activations y))

(defn sigmoid-prime [z]
  (* (logistic z) (- 1 (logistic z))))

(def sneak (atom {}))

(defn backprop [{:keys [biases weights num-layers] :as net} x y]
  (let [nabla-b (mapv #(zero-array (shape %)) biases)
        nabla-w (mapv #(zero-array (shape %)) weights)
        {:keys [zs activations]}
        (reduce
          (fn [{:keys [zs activations] :as acc} [b w]]
            (reset! sneak {:w w :a (last activations)})
            (let [act (dot w (bias-shape (last activations) w))
                  z (+ b act)]
              (assoc acc
                :zs (conj zs z)
                :activations (conj activations (logistic z)))))
          {:zs []
           :activations [x]}
          (mapv vector biases weights))]
    (loop [[l & ls] (range 2 num-layers)
           delta (* (cost-derivative (last activations) y) (sigmoid-prime (last zs)))
           nb (assoc nabla-b (dec (count nabla-b)) delta)
           nw (assoc nabla-w (dec (count nabla-w)) (dot delta (get activations (dec (count activations)))))]
      (cond
        (nil? l)
        [nb nw]
        :else
        (let [z (get zs (- (count zs) l))
              sp (sigmoid-prime z)
              ws (transpose (get weights (inc (- (count weights) l))))
              d (* (dot ws (bias-shape delta ws)) sp)]
          (recur ls d
            (assoc nb (- (count nb) l) d)
            (assoc nw (- (count nw) l) (dot d (get activations (- (count activations) l)))))
          )))))

(defn update-mini-batch [eta total {:keys [biases weights] :as network} [n mini-batch]]
  (println n " / " total " mini batches")
  (let [nabla-apply
        (fn [w nw] (- w (* nw (/ eta (count mini-batch)))))
        [nabla-b nabla-w]
        (reduce
          (fn [[nb nw] [x y]]
            (let [[dnb dnw] (backprop network x y)]
              [(mapv + nb dnb) (mapv + nw dnw)]))
          [(mapv #(zero-array (shape %)) biases)
           (mapv #(zero-array (shape %)) weights)]
          mini-batch)]
    (assoc network
      :weights (mapv nabla-apply weights nabla-w)
      :biases (mapv nabla-apply biases nabla-b))))

(defn evaluate [network test-data]
  (let [ff (for [[x y] test-data] [(maximum (feedforward network x)) y])]
    (count (filter (fn [[a b]] (= a b)) ff))))

(defn SGD [{:as network} training-data epochs mini-batch-size eta & [test-data]]
  (reduce
    (fn [acc epoch]
      (let [mini-batches (partition-all mini-batch-size (shuffle training-data))
            updated-network
            (reduce (partial update-mini-batch eta (count mini-batches)) acc (map-indexed vector mini-batches))]
        (if test-data
          (println "Epoch" epoch ":" (evaluate updated-network test-data) "/" (count test-data))
          (println "Epoch" epoch "complete"))
        updated-network))
    network (range epochs)))