package fr.thomasrobert.inforetrieval

object Search {

  val (tfs, idf) = Indexer.loadFromFile()
  val docsTfIdfs = Indexer.computeTfIdfs(tfs, idf)

  def computeSimilarities(queryTfIdf: Indexer.TFIDF, docsTfIdfs: Indexer.TFIDFs): Map[String, Double] = {
    docsTfIdfs mapValues {
      // ∀ doc, compute cos(q, d) = <q, d> = Σ qi * di
      dis ⇒ queryTfIdf map {
        case (qiTerm, qi) ⇒ qi * dis.getOrElse(qiTerm, 0.0d)
      } reduce (_ + _)
    }
  }

  def orderResults(similarities: Map[String, Double]): Seq[(String, Double)] = {
    similarities.toSeq.sortBy(- _._2)
  }

  def computeQuery(query: String): Seq[(String, Double)] = {
    val queryTf = Indexer.getPageFrequencies(query)
    val queryTfIdf = Indexer.computeTfIdf(queryTf, idf)

    orderResults(computeSimilarities(queryTfIdf, docsTfIdfs))
  }

  // main
  def main(args: Array[String]): Unit = {
    var continue = true
    while (continue) {
      val ln = scala.io.StdIn.readLine()
      if (ln != null)
        computeQuery(ln) take 25 map println
      else
        continue = false
    }
  }

}
