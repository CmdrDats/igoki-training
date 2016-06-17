(ns clj-neural.mnist
  (:require
    [clojure.java.io :as io]
    [clojure.core.matrix :as matrix]
    [clatrix.core :as clatrix])
  (:import java.io.DataInputStream
           java.io.BufferedInputStream
           java.nio.MappedByteBuffer
           java.util.zip.GZIPInputStream
           [org.jblas DoubleMatrix]))


(set! *unchecked-math* true)
(set! *warn-on-reflection* true)

;;MNIST params
(def training-n 60000)
(def test-n 10000)
(def image-size 28)

(defn mnist-data-stream ^DataInputStream [name]
  (let [mnist-path "resources"]
    (DataInputStream. (BufferedInputStream. (GZIPInputStream. (io/input-stream (str mnist-path "/" name)))))))

(defn ub-to-double ^Double [x]
  (double (/ x 255.0)))

(println "Loading MNIST data; may take a hot 10 seconds.")

(def training-images
  "A DoubleMatrix with one row per training image"
  (with-open [ds (mnist-data-stream "train-images-idx3-ubyte.gz")]
    (assert (= (.readInt ds) 2051)
      "Wrong magic number")
    (assert (= (.readInt ds) training-n)
      "Unexpected image count")
    (assert (= (.readInt ds) image-size)
      "Unexpected row count")
    (assert (= (.readInt ds) image-size)
      "Unexpected column count")

    (let [mat (DoubleMatrix. (* image-size image-size) training-n)]
      (dotimes [i (* training-n image-size image-size)]
        (.put mat i (ub-to-double (.readUnsignedByte ds))))
      ;;transpose so that resulting matrix has one image per row.
      (clatrix/matrix (.transpose mat)))))

(def test-images
  "A DoubleMatrix with one row per test image"
  (with-open [ds (mnist-data-stream "t10k-images-idx3-ubyte.gz")]
    (assert (= (.readInt ds) 2051)
      "Wrong magic number")
    (assert (= (.readInt ds) test-n)
      "Unexpected image count")
    (assert (= (.readInt ds) image-size)
      "Unexpected row count")
    (assert (= (.readInt ds) image-size)
      "Unexpected column count")

    (let [mat (DoubleMatrix. (* image-size image-size) test-n)]
      (dotimes [i (* test-n image-size image-size)]
        (.put mat i (ub-to-double (.readUnsignedByte ds))))
      (clatrix/matrix (.transpose mat)))))

(def training-labels
  "A DoubleMatrix vector of training image labels"
  (let [mat (DoubleMatrix. (int training-n))]
    (with-open [ds (mnist-data-stream "train-labels-idx1-ubyte.gz")]
      (assert (= (.readInt ds) 2049)
        "Wrong magic number")
      (assert (= (.readInt ds) training-n)
        "Unexpected label count")
      (dotimes [i training-n]
        (.put mat i (double (.readUnsignedByte ds))))
      (clatrix/matrix mat))))

(def test-labels
  "A DoubleMatrix vector of test image labels"
  (let [mat (DoubleMatrix. (int test-n))]
    (with-open [ds (mnist-data-stream "t10k-labels-idx1-ubyte.gz")]
      (assert (= (.readInt ds) 2049)
        "Wrong magic number")
      (assert (= (.readInt ds) test-n)
        "Unexpected label count")
      (dotimes [i test-n]
        (.put mat i (double (.readUnsignedByte ds))))
      (clatrix/matrix mat))))

(defn vectorized-result [j]
  (assoc (mapv (constantly [0.0]) (range 10)) (int j) [1.0]))

(def training-data
  (map vector
    (map #(matrix/reshape % [784 1]) training-images)
    (map vectorized-result training-labels)))
(def test-data
  (map vector
    (map #(matrix/reshape % [784 1]) test-images)
    test-labels))