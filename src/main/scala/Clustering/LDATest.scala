package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
/**
  * Created by Namhwik on 2017/9/20.
  */
object LDATest {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if (os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val sc = spark.sparkContext
    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\test.txt")
    data.foreach(println)
    val corpus: RDD[String] = data

    // Split each document into a sequence of terms (words)
    val tokenized: RDD[Seq[String]] =
      corpus.map(_.toLowerCase.split(" ")).map(_.filter(_.length > 3).filter(_.forall(java.lang.Character.isLetter)))

    // Choose the vocabulary.
    //   termCounts: Sorted list of (term, termCount) pairs
    val termCounts: Array[(String, Long)] =
    tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    //   vocabArray: Chosen vocab (removing common terms)
    val numStopwords = 20
    val vocabArray: Array[String] =
      termCounts
       //.takeRight(termCounts.length - numStopwords)
        .map(_._1)
    //   vocab: Map term -> term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
    vocab.foreach(println(_))
    println(vocabArray.length)
    tokenized.foreach(println(_))
   // vocab.foreach(println(_))
    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }

        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }
    // Set LDA parameters
    val numTopics = 1
    val lda = new LDA().setK(numTopics).setMaxIterations(90)

    val ldaModel = lda.run(documents)

   // val avgLogLikelihood = ldaModel.logLikelihood / documents.count()
   val avgLogLikelihood = ldaModel.asInstanceOf[DistributedLDAModel].logLikelihood / documents.count()
    // Print topics, showing top-weighted 10 terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 2)
    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabArray(term.toInt)}\t$weight")
      }
      println()
    }
  }
}
