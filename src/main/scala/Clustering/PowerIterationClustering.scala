package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.PowerIterationClustering
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
      SparkSession.builder().appName("test").master("local[4]").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val sc = spark.sparkContext


    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\pic_test_data.txt")
    val similarities: RDD[(Long, Long, Double)] = data.map { line =>
      val parts = line.split(' ')
      (parts(0).toLong, parts(1).toLong, parts(2).toDouble)
    }

    // 使用快速迭代算法将数据聚类
    val pic = new PowerIterationClustering()
      .setK(5)  //k : 期望聚类数
      .setInitializationMode("degree") //模型初始化，默认使用”random” ，即使用随机向量作为初始聚类的边界点，可以设置”degree”（就是图论中的度）
      //随机初始化后，特征值为随机值；度初始化后，特征为度的平均值。
      //度向量会给图中度大的节点分配更多的初始化权重，使其值可以更平均和快速的分布，从而更快的局部收敛。
      .setMaxIterations(20) //幂迭代最大次数
    val model = pic.run(similarities)


    //打印出所有的簇
    val res = model.assignments.collect().map(x=>x.id -> x.cluster).groupBy[Int](_._2).map(x=>x._1 -> x._2.map(x=>x._1).mkString(","))
    res.foreach(x=>println("cluster "+x._1+": " +x._2))
  }
}
