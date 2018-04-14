import scala.collection.mutable.ListBuffer
import scala.util.Random
import Network.sigmoid
import Network.sigmoidPrime
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian


class Network(layers: List[Int]) {

  private var biases: List[DenseVector[Double]] = layers.drop(1).map(numRows =>
    DenseVector.rand(numRows, Gaussian(0, 1))
  )

  private var weights: List[DenseMatrix[Double]] = layers.drop(1).zip(layers.dropRight(1)).map { case (numRows, numCols) =>
    DenseMatrix.rand(numRows, numCols, Gaussian(0, 1))
  }

  def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    biases.zip(weights).foldLeft(input) { case (a, (bias, weight)) =>
      (weight * a + bias).map(sigmoid)
    }
  }

  // stochastic gradient descent (Stochastische Gradientenabstieg)
  def SGD(trainingData: IndexedSeq[(DenseVector[Double], DenseVector[Double])],
          epochs: Int,
          miniBatchSize: Int,
          eta: Double,
          testData: IndexedSeq[(DenseVector[Double], Int)]): Unit = {

    0.until(epochs).foreach { epoch =>
      val miniBatches = Random.shuffle(trainingData).grouped(miniBatchSize)
      miniBatches.foreach(miniBatch => updateMiniBatch(miniBatch, eta))

      // Evaluation
      print(s"Epoch $epoch: ")
      evaluate(testData)
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

    for (layer <- 2 until layers.size) {
      val z = zs(zs.length - layer)
      val sp = z.map(sigmoidPrime)
      delta = (weights(weights.length - layer + 1).t * delta) * sp // dot vs multi = both dot?
      nablaB(nablaB.length - layer) = delta
      nablaW(nablaW.length - layer) = delta * activations(activations.length - layer - 1).t
    }
    (nablaB, nablaW)

  }


  def evaluate(testData: IndexedSeq[(DenseVector[Double], Int)]): Unit = {
    val hits = testData.map { case (x, y) =>
      val output = feedForward(x).data
      (output.indexOf(output.max), y)
    }.count {
      case (x, y) => x == y
    }

    println(s"$hits / ${testData.size} (${100.0 * hits / testData.size}%)")
  }


  def costDerivative(outputActivations: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = outputActivations - y

}

object Network {

  def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))

  def sigmoidPrime(z: Double): Double = sigmoid(z) * (1 - sigmoid(z))

}


object Start extends App {

  println("Untrained net:")
  0.until(10).foreach { _ =>
    new Network(List(784, 30, 10)).evaluate(MnistLoader.testData)
  }

  println("\nTrain net:")

  val network: Network = new Network(List(784, 30, 10))
  network.SGD(
    trainingData = MnistLoader.trainingData,
    epochs = 30,
    miniBatchSize = 10,
    eta = 3,
    testData = MnistLoader.testData
  )

}
