package fr.thomasrobert.inforetrieval

import java.nio.file.{Paths, Files}
import java.io.{File, PrintWriter}
import java.net.URL

import scala.io.Source

object PageLoader {

  def getSha1(s: String) = {
    val md = java.security.MessageDigest.getInstance("SHA-1")
    md.digest(s.getBytes("UTF-8")).map("%02x".format(_)).mkString
  }

  def saveAsFile(HTML: String, filename: String) = {
    val writer = new PrintWriter(new File(filename))
    writer.write(HTML)
    writer.close()
  }

  def loadFile(filename: String): String = {
    try {
      Source.fromFile(filename).getLines().mkString("\n")
    }
    catch {
      case e:Exception ⇒ ""
    }
  }

  def loadFileAsSet(filename: String): Set[String] = {
    try {
      Source.fromFile(filename).getLines().toSet
    }
    catch {
      case e:Exception ⇒ Set()
    }
  }

  def getLines(url: String): Iterator[String] = {
    val connection = new URL(url).openConnection
    connection.setRequestProperty("User-Agent", "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.0)")

    Source.fromInputStream(connection.getInputStream).getLines()
  }

  def downloadContent(url: String): String = {
    try
      getLines(url).toList.mkString(" ")
    catch {
      case e:Exception ⇒ ""
    }
  }

  def getContent(url: String): String = {
    val filename = "files/"+getSha1(url)+".txt"
    if (Files.exists(Paths.get(filename)))
      loadFile(filename)
    else {
      val HTML = downloadContent(url)
      saveAsFile(HTML, filename)
      HTML
    }
  }

}
