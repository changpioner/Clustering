package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Created by Namhwik on 2017/9/19.
  */
object Gaussian {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val sc =spark.sparkContext
    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\gmm_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using GaussianMixture
    val gmm: GaussianMixtureModel = new GaussianMixture().setK(4).run(parsedData)

    // Save and load model
    //gmm.save(sc, "target/org/apache/spark/GaussianMixtureExample/GaussianMixtureModel")
    //val sameModel = GaussianMixtureModel.load(sc,
     // "target/org/apache/spark/GaussianMixtureExample/GaussianMixtureModel")

    // output parameters of max-likelihood model
    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }
  }
}
