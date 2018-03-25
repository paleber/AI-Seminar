import scala.collection.mutable.ListBuffer
import scala.util.Random
import Network.sigmoid
import Network.sigmoidPrime
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian


class Network(sizes: List[Int]) {

  val numLayers: Int = sizes.size

  var biases: List[DenseVector[Double]] = sizes.drop(1).map(numRows =>
    DenseVector.rand(numRows, Gaussian(0, 1))
  )


  var weights: List[DenseMatrix[Double]] = sizes.drop(1).zip(sizes.dropRight(1)).map { case (numRows, numCols) =>
    DenseMatrix.rand(numRows, numCols, Gaussian(0, 1))
  }


  def feedForward(in: DenseVector[Double]): DenseVector[Double] = {
    biases.zip(weights).foldLeft(in) { case (a, (b, w)) =>
      (w * a + b).map(sigmoid)
    }
  }

  def SGD(trainingData: IndexedSeq[(DenseVector[Double], DenseVector[Double])],
          epochs: Int,
          miniBatchSize: Int,
          eta: Double,
          testData: IndexedSeq[(DenseVector[Double], Int)]): Unit = {

    0.until(epochs).foreach { j =>
      val miniBatches = Random.shuffle(trainingData).grouped(miniBatchSize)
      miniBatches.foreach(miniBatch => updateMiniBatch(miniBatch, eta))

      if (testData.nonEmpty) {
        println(s"Epoch $j: ${evaluate(testData)} / ${testData.size}")
      } else {
        println(s"Epoch $j complete")
      }
    }

  }

  def updateMiniBatch(miniBatch: IndexedSeq[(DenseVector[Double], DenseVector[Double])], eta: Double): Unit = {
    var nablaB = biases.map(bias => DenseVector.zeros[Double](bias.length))
    var nablaW = weights.map(weight => DenseMatrix.zeros[Double](weight.rows, weight.cols))
    miniBatch.foreach { case (x, y) =>
      val (deltaNablaB, deltaNablaW) = backProp(x, y)
      nablaB = nablaB.zip(deltaNablaB).map { case (nb, dnb) => nb + dnb }
      nablaW = nablaW.zip(deltaNablaW).map { case (nw, dnw) => nw + dnw }
    }
    weights = weights.zip(nablaW).map { case (w, nw) =>
      w - (eta / miniBatch.size) * nw
    }
    biases = biases.zip(nablaB).map { case (b, nb) =>
      b - (eta / miniBatch.size) * nb
    }

  }


  def backProp(x: DenseVector[Double], y: DenseVector[Double]): (IndexedSeq[DenseVector[Double]], IndexedSeq[DenseMatrix[Double]]) = {

    val nablaB = biases.map(vector => DenseVector.zeros[Double](vector.length)).toArray
    val nablaW = weights.map(matrix => DenseMatrix.zeros[Double](matrix.rows, matrix.cols)).toArray

    // Feed forward
    var activation = x
    val activations = ListBuffer(x) // list to store all the activations, layer by layer
    val zs = ListBuffer.empty[DenseVector[Double]] // list to store all the z vectors, layer by layer

    biases.zip(weights).foreach { case (b, w) =>
      val z = w * activation + b
      zs += z
      activation = z.map(sigmoid)
      activations += z.map(sigmoid)
    }

    var delta = costDerivative(activations.last, y) * zs.last.map(sigmoidPrime)
    nablaB(nablaB.length - 1) = delta
    nablaW(nablaW.length - 1) = delta * activations(activations.length - 2).t

    for (layer <- 2 until numLayers) {
      val z = zs(zs.length - layer)
      val sp = z.map(sigmoidPrime)
      delta = (weights(weights.length - layer + 1).t * delta) * sp // dot vs multi = both dot?
      nablaB(nablaB.length - layer) = delta
      nablaW(nablaW.length - layer) = delta * activations(activations.length - layer - 1).t
    }
    (nablaB, nablaW)

  }


  def evaluate(testData: IndexedSeq[(DenseVector[Double], Int)]): Int = {
    testData.map { case (x, y) =>
      val output = feedForward(x).data
      (output.indexOf(output.max), y)
    }.count {
      case (x, y) => x == y
    }
  }


  def costDerivative(outputActivations: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = outputActivations - y

}

object Network {

  def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))

  def sigmoidPrime(z: Double): Double = sigmoid(z) * (1 - sigmoid(z))

}


object Start extends App {

  val network: Network = new Network(List(784, 30, 10))

  network.SGD(
    trainingData = MnistLoader.trainingData,
    epochs = 30,
    miniBatchSize = 10,
    eta = 3,
    testData = MnistLoader.testData
  )

}
