package Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.mutable

/**
  * Created by Namhwik on 2017/9/19.
  */
object MLLDA {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val  sc = spark.sparkContext
    import spark.sqlContext.implicits._


    // Loads data.
    //val dataset: DataFrame = spark.read.format("libsvm")
    //  .load("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\sample_lda_libsvm_data.txt")
   // dataset.show(false)
  //  dataset.printSchema()


    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\test.txt")
    data.foreach(println)
    val corpus: RDD[String] = data

    // Split each document into a sequence of terms (words)
    val tokenized: RDD[Seq[String]] =
      corpus.map(_.toLowerCase.split(" ")).map(_.filter(_.length >= 3)
        //.filter(_.forall(java.lang.Character.isLetter))
        )
    val removeArr = Array(""
     /* "very","there","will","super","this","have","with","some",
      "them","because","their","then","going","they","after",
      "the","has","are","too","for","you","good","make","once","not","but","new"*/
    )

    // Choose the vocabulary.
    //   termCounts: Sorted list of (term, termCount) pairs
    val termCounts: Array[(String, Long)] =
    tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().filter(x => !removeArr.contains(x._1)).sortBy(-_._2)
    //   vocabArray: Chosen vocab (removing common terms)
    val numStopwords = 40
    val vocabArray: Array[String] =
      termCounts
        .takeRight(termCounts.length - numStopwords)
        .map(_._1)
    //   vocab: Map term -> term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
   // termCounts.foreach(println(_))
    //vocabArray.foreach(println(_))
    //println(vocabArray.length)
    tokenized.foreach(println(_))
    // vocab.foreach(println(_))

    // Convert documents into term count vectors
    val documents =
    tokenized.zipWithIndex.map {
      case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach {
          (term: String) =>
            if (vocab.contains(term)) {
              val idx = vocab(term)
              counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
            }
        }

        (id, ml.linalg.Vectors.sparse(vocab.size, counts.toSeq))
    }
    val dataset: DataFrame = documents.toDF("label","features")
    dataset.show()





    // Trains a LDA model.
    val lda = new LDA()
      .setK(6)
      .setMaxIter(50)
      .setCheckpointInterval(5)
      .setOptimizer("EM")
     // .setDocConcentration(5.00)
     // .setTopicConcentration(5.00)
     // .setSeed(1024L)
    val model = lda.fit(dataset)

    val ll = model.logLikelihood(dataset)
    val lp = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")
    //println(model.topicsMatrix)
    // Describe topics.
    val topics = model.describeTopics(6)
    println("The topics described by their top-weighted terms:")
    topics.printSchema()
    topics.show(false)
    val vocabArrayBr = spark.sparkContext.broadcast(vocabArray)
    //topics.foreach(x=>println(x.getAs[mutable.WrappedArray[Int]]("termIndices").mkString(",")))
    val getTermName = udf {
      (termIndices: mutable.WrappedArray[Int]) => termIndices.map(vocabArrayBr.value(_))
    }
    val res = topics.withColumn("termNames",getTermName(col("termIndices")))
    res.select("topic","termNames","termWeights").show(false)
    vocabArrayBr.destroy()
    // Shows the result.
    val transformed: DataFrame = model.transform(dataset)
    transformed.show(false)
  }
}
