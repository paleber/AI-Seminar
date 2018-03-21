import scala.util.Random

case class Matrix(matrix: Array[Array[Double]]) {

  def +(other: Matrix): Matrix = {
    Matrix(
      matrix.indices.map(m =>
        matrix.head.indices.map(n =>
          matrix(m)(n) + other.matrix(m)(n)
        ).toArray
      ).toArray
    )
  }

  def *(other: Matrix): Matrix = {
    Matrix(
      matrix.indices.map(m =>
        other.matrix.head.indices.map(n =>
          other.matrix.indices.map(i =>
            matrix(m)(i) * other.matrix(i)(n)
          ).sum
        ).toArray
      ).toArray
    )
  }

  def map(f: Double => Double): Matrix = {
    Matrix(matrix.map(_.map(f)))
  }

  override def toString: String = matrix.map(row =>
    s"(${row.mkString(",")})"
  ).mkString(sys.props("line.separator"))

}


/*
   def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
 */
class Network(sizes: List[Int]) {

  val numLayers: Int = sizes.size

  val biases: List[Matrix] = sizes.drop(1).map(y =>
    Matrix(Array.fill(y, 1)(Random.nextGaussian))
  )

  val weights = sizes.dropRight(1).zip(sizes.drop(1)).map(t =>
    Matrix(Array.fill(t._2, t._1)(Random.nextGaussian))
  )

  /*
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
   */
  def feedForward(matrix: Matrix): Matrix = {
    var a = matrix
    biases.zip(weights).foreach(t => // TODO use reduce
      a = (t._2 * a + t._1).map(sigmoid)
    )
    a
  }

  /*
  def SGD(self, training_data, epochs, mini_batch_size, eta,
           test_data=None):
       """Train the neural network using mini-batch stochastic
       gradient descent.  The ``training_data`` is a list of tuples
       ``(x, y)`` representing the training inputs and the desired
       outputs.  The other non-optional parameters are
       self-explanatory.  If ``test_data`` is provided then the
       network will be evaluated against the test data after each
       epoch, and partial progress printed out.  This is useful for
       tracking progress, but slows things down substantially."""
       if test_data: n_test = len(test_data)
       n = len(training_data)
       for j in xrange(epochs):
           random.shuffle(training_data)
           mini_batches = [
               training_data[k:k+mini_batch_size]
               for k in xrange(0, n, mini_batch_size)]
           for mini_batch in mini_batches:
               self.update_mini_batch(mini_batch, eta)
           if test_data:
               print "Epoch {0}: {1} / {2}".format(
                   j, self.evaluate(test_data), n_test)
           else:
               print "Epoch {0} complete".format(j)
  */
  def SGD(trainingData: List[(Matrix, Int)], epochs: Int, miniBatchSize: Int, eta: Any, testData: Option[Any]): Unit = {
    val n = trainingData.size
    (0 until epochs).foreach { j =>
      val shuffled = Random.shuffle(trainingData)
      val mini_batches = (0 to n by miniBatchSize).map(k =>
        trainingData.slice(k, k + miniBatchSize)
      )
      // val updatedMiniBatches =
      // TODO
    }

  }


  /*
 def update_mini_batch(self, mini_batch, eta):
      """Update the network's weights and biases by applying
      gradient descent using backpropagation to a single mini batch.
      The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
      is the learning rate."""
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      for x, y in mini_batch:
          delta_nabla_b, delta_nabla_w = self.backprop(x, y)
          nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
          nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      self.weights = [w-(eta/len(mini_batch))*nw
                      for w, nw in zip(self.weights, nabla_w)]
      self.biases = [b-(eta/len(mini_batch))*nb
                     for b, nb in zip(self.biases, nabla_b)]
 */
  def updateMiniBatch(miniBatch: List[(Double, Double)], eta: Any): Unit = {
    val nablaB = biases.map(_.map(_ => 0))
    val nablaW = weights.map(_.map(_ => 0))
    for (elem <- miniBatch) {
      // TODO
    }
  }


  /*
   def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
   */

  /*
     def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
*/
  def evaluate(testData: List[(Matrix, Int)]) = {
    testData.map { case (x, y) =>
      val r = feedForward(x).matrix.head
      (r.indexOf(r.max), y)
    }.count {
      case (x, y) => x == y
    }
  }

  /*
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
   */
  def cost_derivative(output_activations: Double, y: Double): Double = output_activations - y

}


/*
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

 */
def sigmoid(z: Double) = 1 / (1 + math.exp(-z))

/*
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
 */
def sigmoidPrime(z: Double) = sigmoid(z) * (1 - sigmoid(z))


object Start extends App {

  val a = Matrix(Array(Array(1.0, 2.0, 3.0), Array(2.0, 3, 4)))
  println(a)
  println()
  val b = Matrix(Array(Array(3.0, 1.0), Array(2.0, 2), Array(7.0, 5)))
  println(b)
  println()
  println(a * b)

}

