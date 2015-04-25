package fr.thomasrobert.inforetrieval

object Crawler {
  /**
   * Return the parts of an URL
   */
  def getInfos(url : String): (String, String, String) = {
    val protocol = ("^([^:]+)://([^/]+)/".r findFirstMatchIn url).get.group(1)
    val domain = ("^([^:]+)://([^/]+)/".r findFirstMatchIn url).get.group(2)
    val directory = ("^(.+/)[^/]*$".r findFirstMatchIn url).get.group(1)
    (protocol, domain, directory)
  }

  /**
   * Get an absolute URL from an HREF and information on the source URL
   */
  def getAbsoluteHref(protocol: String, domain: String, directory: String)(href : String): String = {
    if (matches("[^:]+://.+", href))
      href
    else if (matches("^//.+", href))
      protocol + ":" + href
    else if (matches("^/.+", href))
      protocol + "://" + domain + href
    else
      directory + href
  }

  /**
   * Remove the anchor from an URL
   */
  def removeAnchors(s: String): String = {
    "#.+$".r replaceAllIn(s, "")
  }

  /**
   * Utility function that says if "s" matches the regex string "r"
   */
  def matches(r : String, s : String): Boolean = r.r.pattern.matcher(s).matches()

  /**
   * Extract URLs in the page "parentUrl"
   */
  def getUrls(parentUrl: String): List[String] = {
    try {
      val (protocol, domain, directory) = getInfos(parentUrl)

      val HTML = PageLoader.getContent(parentUrl)
      val HTMLbody = ("<body[^>]+>(.+)</body".r findFirstMatchIn HTML).get.group(1)

      val regex = "href=['\"]([^'\"]+)['\"]".r

      val hrefs = (regex findAllIn HTMLbody) map (x ⇒ (regex findFirstMatchIn x).get.group(1))

      def getAbsoluteHrefLocal = getAbsoluteHref(protocol, domain, directory)(_)

      (hrefs map getAbsoluteHrefLocal map removeAnchors).toSet.toList
    }
    catch {
      case e: Exception ⇒ List()
    }
  }

  /**
   * Crawl to look for URLs from a source, stop the crawling when we found at least "limit" URLs.
   */
  def crawl(source: String, limit: Int): Set[String] = {

    def crawlRec(set: Set[String], nextHrefs: List[String]): Set[String] = {
      if (set.size > limit)
        set
      else {
        // crawl new links to add to set
        val toAdd = getUrls(nextHrefs.head) filter (x ⇒ !(set contains x))
        // recursive call with new links minus processed link
        crawlRec(set ++ toAdd, nextHrefs.tail ::: toAdd)
      }
    }

    val init = getUrls(source)
    crawlRec(init.toSet, init)
  }

  /**
   * Main
   */
  def main(args: Array[String]) {
    PageLoader.saveAsFile(crawl("https://duckduckgo.com/html/?q=information%20retrieval", 3000).mkString("\n"), "files/database.txt")
    val database = PageLoader.loadFileAsSet("files/database.txt")
    println(database.size)
    println(database)
  }
}
