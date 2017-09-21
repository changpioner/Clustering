package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.sql.SparkSession

/**
  * Created by Namhwik on 2017/9/19.
  */
object BisectingKMeans {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val sc =spark.sparkContext
    def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))
    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\kmeans_data.txt").map(parse).cache()

    // Clustering the data into 6 clusters by BisectingKMeans.
    val bkm = new BisectingKMeans().setK(10)
    val model = bkm.run(data)

    // Show the compute cost and the cluster centers
    println(s"Compute Cost: ${model.computeCost(data)}")
    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      println(s"Cluster Center ${idx}: ${center}")
    }
  }
}
