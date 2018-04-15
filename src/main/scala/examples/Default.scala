package examples

import network.{MnistLoader, Network}

object Default extends App {

  val network: Network = new Network(List(784, 30, 10))
  network.SGD(
    trainingData = MnistLoader.trainingData,
    epochs = 30,
    miniBatchSize = 10,
    eta = 3,
    testData = MnistLoader.testData
  )

}
