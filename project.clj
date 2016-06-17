(defproject clj-neural "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies
  [[org.clojure/clojure "1.8.0"]
   [net.mikera/core.matrix "0.52.0"]
   [clatrix "0.5.0"]
   [com.google.guava/guava "18.0"]

   [org.nd4j/nd4j "0.4-rc3.10" :extension "pom"]

   [org.nd4j/nd4j-cuda-7.5 "0.4-rc3.10"]
   #_[org.nd4j/nd4j-cuda-7.5 "0.4-rc3.10"  :classifier "macosx-x86_64"]
   #_[org.nd4j/nd4j-native "0.4-rc3.10"]
   [org.deeplearning4j/deeplearning4j-core "0.4-rc3.10"]
   [org.nd4j/canova-api "0.0.0.16"]

   ]
  :jvm-opts
  ["-Xms3024M" "-Xmx6048M" "-XX:NewSize=1024M" "-XX:+UseParNewGC" "-XX:+UseConcMarkSweepGC"
   "-XX:+CMSParallelRemarkEnabled" "-server" "-XX:-OmitStackTraceInFastThrow"])

