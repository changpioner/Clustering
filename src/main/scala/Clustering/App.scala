package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * @author ${user.name}
 */
object App {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args : Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    // Loads data.
    val data = spark.sparkContext.textFile("C:\\编程\\jars\\spark-1.4.0-bin-hadoop2.6\\data\\mllib\\kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    //clusters.save(spark.sparkContext, "target/org/apache/spark/KMeansExample/KMeansModel")
    //val sameModel = KMeansModel.load(spark.sparkContext, "target/org/apache/spark/KMeansExample/KMeansModel")

  }

}
