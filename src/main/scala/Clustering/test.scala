package Clustering
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkException
import org.apache.spark.graphx._
import org.apache.spark.mllib.clustering.KMeans
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
    val sortedSet=   mutable.SortedSet[Double]()

    sortedSet.add(5.56)
    sortedSet.add(0.09)
    sortedSet.add(7.62)
    sortedSet.add(0.45)
    val sortedArr= sortedSet.toArray
    sortedArr.foreach(println(_))


  }

  def normalize(graph: Graph[Double, Double]): Graph[Double, Double] = {
    val vD: VertexRDD[Double] = graph.aggregateMessages[Double](
      sendMsg = (ctx: EdgeContext[Double, Double, Double]) => {
        val i: VertexId = ctx.srcId
        val j: VertexId = ctx.dstId
        val s: Double = ctx.attr
        if (s < 0.0) {
          throw new SparkException(s"Similarity must be nonnegative but found s($i, $j) = $s.")
        }
        if (s > 0.0) {
          ctx.sendToSrc(s)
        }
      },
      mergeMsg = _ + _,
      TripletFields.EdgeOnly)
    Graph(vD, graph.edges)
      .mapTriplets(
        e => e.attr / math.max(e.srcAttr, 0.0001),
        new TripletFields(/* useSrc */ true,
          /* useDst */ false,
          /* useEdge */ true))
  }



  def kMeans(v: VertexRDD[Double], k: Int): VertexRDD[Int] = {
    val points = v.mapValues(x => Vectors.dense(x)).cache()
    val values = points.values
    val model = new KMeans()
      .setK(k)
      .setSeed(0L)
      .run(points.values)
    points.mapValues(p => model.predict(p)).cache()
  }
}
