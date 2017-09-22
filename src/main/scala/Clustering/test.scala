package Clustering
import org.apache.spark.mllib.linalg.Vectors

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
  }
}
