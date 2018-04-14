
case class BasicMatrix(data: IndexedSeq[IndexedSeq[Double]]) {

  def *(other: BasicMatrix): BasicMatrix = ???

  def +(other: BasicMatrix): BasicMatrix = ???

  def map(f: Double => Double): BasicMatrix = ???

}
