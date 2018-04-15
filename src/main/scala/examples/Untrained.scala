package examples

import network.{MnistLoader, Network}

object Untrained extends App {

  (0 until 10).foreach { _ =>
    new Network(List(784, 30, 10)).evaluate(MnistLoader.testData)
  }

}
