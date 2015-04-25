package fr.thomasrobert.inforetrieval

import org.jsoup.Jsoup
import org.json4s.native.Serialization.{read, write}
import org.json4s.DefaultFormats

object Indexer {

  // formats for JSON lib
  implicit val formats = DefaultFormats

  // types
  type IDF    = Map[String, Double]
  type TF     = Map[String, Double]
  type TFIDF  = Map[String, Double]
  type TFs    = Map[String, TF]
  type TFIDFs = Map[String, TFIDF]

  // file save
  def saveAsFile(tfs: TFs, idf: IDF) = {
    PageLoader.saveAsFile(write(tfs), "files/tfs.json")
    PageLoader.saveAsFile(write(idf), "files/idf.json")
  }

  def loadFromFile(): (TFs, IDF) = {
    val tfs = read[TFs](PageLoader.loadFile("files/tfs.json"))
    val idf = read[IDF](PageLoader.loadFile("files/idf.json"))
    (tfs, idf)
  }

  // process HTML & words
  def isHTML(page: String): Boolean = page.toLowerCase.replaceAll("[ \t]", "").contains("<body")
  def cleanHTML(page: String): String = Jsoup.parse(page).text

  def getWords(s: String): Iterable[String] = {
    val string = "[^a-zA-Z]".r replaceAllIn (s, " ")
    string split " " filter (x ⇒ x.length > 3) filter (x ⇒ x forall (l ⇒ l.isLetter))
  }

  def getStems(words: Iterable[String]): Iterable[String] = {
    words map Stemmer.stem
  }

  // process frequencies (tf & idf)
  def getNbOccs(wordsList: Iterable[String]): Map[String, Int] = {
    wordsList groupBy (x ⇒ x) mapValues (_.size)
  }

  def getFrequencies(wordsList: Iterable[String]): TF = {
    getNbOccs(wordsList) mapValues (tf ⇒ if (tf == 0) 0 else 1 + Math.log10(tf.toDouble) / wordsList.size)
  }

  def getPageFrequencies(page: String): TF = {
    getFrequencies(getStems(getWords(page)))
  }

  def computeTfAndIdf(urlDatabase: Iterable[String]): (TFs, IDF) = {

    // filter to only keep valid HTMLs
    val urlAndHTMLDatabaseCandidates = (urlDatabase map (url ⇒ url → PageLoader.getContent(url)) ).toMap
    val urlAndTextDatabase = urlAndHTMLDatabaseCandidates filter {case (_, html) ⇒ isHTML(html)} mapValues cleanHTML
    val nbDocs = urlAndTextDatabase.size

    // compute termsFrequencies Map[url, Map[term, tf]]
    val termsFrequencies = urlAndTextDatabase mapValues getPageFrequencies

    // compute documentsFrequencies Map[term, nbDocsWithTerm]
    val documentsFrequencies = getNbOccs(termsFrequencies.values flatMap (_.keys))

    // compute inverseDocumentsFrequencies Map[term, idf]
    val inverseDocumentsFrequencies = documentsFrequencies mapValues ( nbDocsWithTerm ⇒ Math.log10(nbDocs.toDouble / nbDocsWithTerm ))

    (termsFrequencies, inverseDocumentsFrequencies)
  }

  def computeTfIdf(tfs: TF, idf: IDF):  TFIDF = {
    val tfIdfs = tfs map {case (term, tf) ⇒ (term, tf * idf.getOrElse(term, 0.0d)) }
    val norm = Math.sqrt( tfIdfs.values.map(x ⇒ x * x).fold(0.0d)(_ + _) )
    tfIdfs mapValues (_ / norm)
  }

  def computeTfIdfs(tfs: TFs, idf: IDF):  TFIDFs = {
    tfs mapValues (tf ⇒ computeTfIdf(tf, idf))
  }

  // main
  def main(args: Array[String]) {

    val (termsFrequencies, inverseDocumentsFrequencies) = computeTfAndIdf(PageLoader.loadFileAsSet("files/database.txt"))
    saveAsFile(termsFrequencies, inverseDocumentsFrequencies)
    //  val (termsFrequencies, inverseDocumentsFrequencies) = loadFromFile()

    println(termsFrequencies.size+" docs")
    println(termsFrequencies.head)
    println(inverseDocumentsFrequencies)
  }
}