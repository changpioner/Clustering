package Clustering
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
/**
  * Created by Namhwik on 2017/9/21.
  */
object test {
  def main(args: Array[String]): Unit = {
    val map = new mutable.HashMap[Int,Double]()
    map(0)=1.11
    map.put(1,2.22)
    map.put(2,3.33)
    map(3)=4.44
    val seq = map.toSeq
    val vector = Vectors.sparse(map.size,seq)
    println(vector)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val para: RDD[(Int, Double)] = spark.sparkContext.parallelize(map.toSeq,3)
    para.foreachPartition(
      par=>{
        par.foreach(
          x=>println(x)
        )
      }
    )
  }
}
