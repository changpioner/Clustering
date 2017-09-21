package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by Namhwik on 2017/9/19.
  */
object Lda {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    // Loads data.
    val dataset: DataFrame = spark.read.format("libsvm")
      .load("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\sample_lda_libsvm_data.txt")

    // Trains a LDA model.
    val lda = new LDA().setK(4).setMaxIter(10)
    val model = lda.fit(dataset)

    val ll = model.logLikelihood(dataset)
    val lp = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")
    println(model.topicsMatrix)
    // Describe topics.
    val topics = model.describeTopics(5)
    println("The topics described by their top-weighted terms:")
    topics.show(false)

    // Shows the result.
    val transformed: DataFrame = model.transform(dataset)
    transformed.show(false)
  }
}
