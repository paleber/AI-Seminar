package network

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import breeze.linalg.DenseVector

import scala.collection.immutable.Seq


object MnistLoader {

  private val baseDirectory = "src/main/resources"

  lazy val testLabels: Seq[Int] = readLabels(s"$baseDirectory/t10k-labels-idx1-ubyte.gz")
  lazy val trainingLabels: Seq[Int] = readLabels(s"$baseDirectory/train-labels-idx1-ubyte.gz")

  lazy val testImages: Seq[DenseVector[Double]] = readImages(s"$baseDirectory/t10k-images-idx3-ubyte.gz")
  lazy val trainingImages: Seq[DenseVector[Double]] = readImages(s"$baseDirectory/train-images-idx3-ubyte.gz")

  lazy val trainingData: Seq[(DenseVector[Double], DenseVector[Double])] = trainingImages.zip(trainingLabels.map { testLabel =>
    val vector = DenseVector.zeros[Double](10)
    vector.update(testLabel, 1)
    vector
  })
  lazy val testData: Seq[(DenseVector[Double], Int)] = testImages.zip(testLabels)

  private def gzipInputStream(s: String) = new GZIPInputStream(new BufferedInputStream(new FileInputStream(s)))

  private def read32BitInt(i: GZIPInputStream) = i.read() * 16777216 + i.read() * 65536 + i.read() * 256 + i.read()

  /**
    *
    * @param filepath the full file path the labels file
    * @return
    */
  def readLabels(filepath: String): Seq[Int] = {
    val g = gzipInputStream(filepath)
    val _ = read32BitInt(g) // not used
    val numberOfLabels = read32BitInt(g)
    1.to(numberOfLabels).map(_ => g.read())
  }

  /**
    *
    * @param filepath the full file path of the images file
    * @return
    */
  def readImages(filepath: String): Seq[DenseVector[Double]] = {
    val g = gzipInputStream(filepath)
    val _ = read32BitInt(g) // not used
    val numberOfImages = read32BitInt(g)
    val imageSize = read32BitInt(g) * read32BitInt(g) //cols * rows

    1.to(numberOfImages).map(_ =>
      DenseVector(1.to(imageSize).map(_ => g.read().toDouble / 256).toArray)
    )
  }

}
