package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.mllib.clustering.PowerIterationClustering.Assignment
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * Created by Namhwik on 2017/9/19.
  */
object PowerIterationClustering {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if (os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val sc = spark.sparkContext
    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\pic_data.txt")
    val similarities: RDD[(Long, Long, Double)] = data.map { line =>
      val parts = line.split(' ')
      (parts(0).toLong, parts(1).toLong, parts(2).toDouble)
    }
    // 使用快速迭代算法将数据分为两类
    val pic = new PowerIterationClustering()
      .setK(2)
      .setMaxIterations(10)
    val model = pic.run(similarities)

    //打印出所有的簇
    model.assignments.foreach { (a: Assignment) =>
      println(s"${a.id} -> ${a.cluster}")
    }


  }
}
