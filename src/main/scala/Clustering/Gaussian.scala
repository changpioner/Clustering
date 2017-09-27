package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
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
    //spark.read.te
    val data: RDD[String] = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\kmeans_data.txt")
    val parsedData: RDD[Vector] = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

   //创建一个高斯混合模型
    val gmm: GaussianMixtureModel = new GaussianMixture()
      //.setSeed(8192)
      .setConvergenceTol(0.0001) //设置模型迭代阈值
      //.setInitialModel()
      .setMaxIterations(200) //最大迭代次数
      .setK(2)
      .run(parsedData)

    //根据模型对数据进行预测
    val preResult0 = gmm
      .predict(Vectors.dense(Array(8.1,7.1,9.1,228.0)))
    val preResult1= gmm
      .predictSoft(Vectors.dense(Array(8.1,7.1,9.1,228.0))).mkString(",")
    val preResult2 =gmm
      .predictSoft(parsedData)
    println(preResult0)
    println()
    println(preResult1)
    println()
    preResult2.collect().foreach(x=>println(x.mkString(",")))

    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }
  }
}
