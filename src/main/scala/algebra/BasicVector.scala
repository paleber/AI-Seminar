package algebra

case class BasicVector(data: IndexedSeq[Double]) {

  def +(other: BasicVector): BasicVector = {
    copy(
      data.zip(other.data).map(n => n._1 + n._2)
    )
  }

  def *(factor: Double): BasicVector = {
    copy(data.map(_ * factor))
  }

}

object BasicVector {

  implicit class DoubleMethods(n: Double) {
    def *(v: BasicVector): BasicVector = v * n
  }

}


object MyApp extends App {

  val a = BasicVector(IndexedSeq(1, 2, 3))
  val b = BasicVector(IndexedSeq(4, 5, 6))

  println(a + b) // BasicVector(Vector(5.0, 7.0, 9.0))

  println(a * 3) // BasicVector(Vector(3.0, 6.0, 9.0))

  println(5 * a) // BasicVector(Vector(5.0, 10.0, 15.0))

}
