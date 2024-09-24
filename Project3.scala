// Databricks notebook source
// MAGIC %md
// MAGIC **Main Project (100 pts)** \
// MAGIC Implement closed-form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC Steps:
// MAGIC 1. Create an example RDD for matrix X of type RDD\[Int, Int, Double\] and vector y of type RDD\[Int, Double\]
// MAGIC 2. Compute \\[ \scriptsize \mathbf{(X^TX)}\\]
// MAGIC 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse
// MAGIC 4. Compute \\[ \scriptsize \mathbf{X^Ty}\\] and convert it to Breeze Vector
// MAGIC 5. Multiply \\[ \scriptsize \mathbf{(X^TX)}^{-1}\\] with \\[ \scriptsize \mathbf{X^Ty}\\]

// COMMAND ----------

//Main Project Code

// Creating a example RDD for matrix X
import org.apache.spark.rdd.RDD
val matrix_X = Array(Array(1.0, 2.0), Array(4.0, 5.0), Array(7.0, 8.0), Array(10.0, 11.0))
val matrix_X_RDD = sc.parallelize(matrix_X.zipWithIndex.flatMap{case (row, i) => row.zipWithIndex.map{case (value, j) => ((i,j), value)}})

println("Elements of the matrix in coordinate form (row, column, value):")
matrix_X_RDD.collect.foreach(println)

// Transpose of matrix X
println("Elements of the transposed matrix in coordinate form (row, column, value):")

val transpose_of_X = matrix_X_RDD.map{case ((i,j), value)=> ((j,i), value)}

transpose_of_X.collect.foreach(println)


// COMMAND ----------

// Function to find (X^TX)
def Multiply ( M: RDD[((Int,Int),Double)], N: RDD[((Int,Int),Double)] ): RDD[((Int,Int),Double)] = {
  M.map{case (index, value) => (index._2, (index._1, value))}
   .join(N.map{case (index,value) => (index._1, (index._2,value))})
   .map{case (key, ((i, value_1), (j, value_2))) => ((i, j), value_1 * value_2)}
   .reduceByKey(_+_)
}

val XTX = Multiply(transpose_of_X, matrix_X_RDD)
XTX.collect.foreach(println)

// COMMAND ----------

import breeze.linalg.{DenseMatrix, DenseVector, pinv}

//Function to convert the result matrix into a Breeze Dense Matrix and compute the pseudo-inverse of (X^TX)

def computePseudoinverse(XTX: RDD[((Int, Int), Double)]): DenseMatrix[Double] = {
 
val number_of_rows = XTX.map{case ((i,j),k) => i}.max + 1

val number_of_columns = XTX.map{case ((i,j),k) => j}.max + 1

val XTX_array = XTX.collect()

  
val breeze_dense_matrix = DenseMatrix.zeros[Double](number_of_rows, number_of_columns)
XTX_array.foreach{case ((i,j), value) => breeze_dense_matrix(i,j) = value}

pinv(breeze_dense_matrix)

}


// COMMAND ----------

//calling the computePseudoinverse function
val pseudoInverse = computePseudoinverse(XTX)

println(pseudoInverse)

// COMMAND ----------

// Computing X^Ty
import breeze.linalg.DenseVector

val y  = Array(Array(2.0), Array(3.0), Array(4.0), Array(5.0))

val y_RDD = sc.parallelize(y.zipWithIndex.flatMap{case (row, i) => row.map{case (value) => (i,value)}})

def MultiplyWithVector ( M: RDD[((Int,Int),Double)], N: RDD[(Int,Double)] ): RDD[(Int, Double)] = {
  M.map{case (index, value) => (index._2, (index._1, value))}
   .join(N.map{case (index,value) => (index, (value))})
   .map{case (key, ((i, value_1), (value_2))) => ((i), value_1 * value_2)}
   .reduceByKey(_+_)
}

val XTy = MultiplyWithVector(transpose_of_X, y_RDD)

XTy.collect.foreach(println)


// COMMAND ----------

import breeze.linalg.DenseVector

// Function to convert X^Ty into Breeze Vector
def createDenseVector(XTy: RDD[(Int, Double)]): DenseVector[Double] = {
  val vector_length = XTy.map{case (i,value) => i}.max + 1
  val XTy_array = XTy.collect()
  val breeze_vector = DenseVector.zeros[Double](vector_length)
  XTy_array.foreach{case (i,value) => breeze_vector(i) = value}
  breeze_vector
}


// COMMAND ----------

val breezeVector = createDenseVector(XTy)

println(breezeVector)

// COMMAND ----------

// Myltiply (XTX)^−1  with X^Ty

if (pseudoInverse.cols == breezeVector.length) {
  // Perform matrix-vector multiplication
  println("Multiplication result:")
  val final_result = pseudoInverse * breezeVector
  print(final_result)
}
else{
  print("Dimension Mismatch")
}

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 1(10 pts)** \
// MAGIC Implement \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\] using Spark DataFrame.  
// MAGIC
// MAGIC Note: Your queries should be in the following format:
// MAGIC \\[ \scriptsize \mathbf{spark.sql("select ... from ...")}\\]

// COMMAND ----------

// Bounus 1 Code

import org.apache.spark.sql.types._

// Matrix Schema

val matrix_schema = StructType(Seq(
  StructField("row", IntegerType, nullable = true),
  StructField("column", IntegerType, nullable = true),
  StructField("value", DoubleType, nullable = true)
))

//Vector_schema

val vector_schema =  StructType(Seq(
  StructField("row", IntegerType, nullable = true),
  StructField("value", DoubleType, nullable = true)
))


// COMMAND ----------

// Creating RDDs
val matrixRDD = sc.parallelize(Seq(
  (0, 0, 1.0), (0, 1, 2.0),
  (1, 0, 4.0), (1, 1, 5.0),
  (2, 0, 7.0), (2, 1, 8.0),
  (3, 0, 10.0), (3, 1, 11.0)
))

val vector_RDD = sc.parallelize(Seq(
  (0, 2.0), (1, 3.0),
  (2, 4.0), (3, 5.0)
))

val matrixRDD_schema = matrixRDD.map{case (i,j,v) => Row(i,j,v)}
val vector_RDD_schema = vector_RDD.map{case (i,v) => Row(i,v)}

//Creating DataFrames
val matrixDF = spark.createDataFrame(matrixRDD_schema, matrix_schema)
val vectorDF = spark.createDataFrame(vector_RDD_schema, vector_schema)

matrixDF.createOrReplaceTempView("matrixDF")

vectorDF.createOrReplaceTempView("vectorDF")

matrixDF.show()
vectorDF.show()



// COMMAND ----------

// function to Compute (X^TX) for Dataframes
import org.apache.spark.sql.DataFrame
def calculateXTX(matrixDF: DataFrame): DataFrame = {
  // Create the transposed matrix view
  matrixDF.createOrReplaceTempView("matrixDF")
  val transposed_matrix = spark.sql(s"SELECT column AS row, row AS column, value FROM matrixDF")
  transposed_matrix.createOrReplaceTempView("transposed_matrix")

  // Compute XTX
  val XTX = spark.sql("select r.row as row, t.column as column, SUM(r.value*t.value) as value from transposed_matrix r,matrixDF t where r.column = t.row Group By r.row, t.column")
  
  XTX.createOrReplaceTempView("XTX")

  XTX
}

// COMMAND ----------

val XTX = calculateXTX(matrixDF)
XTX.show()

// COMMAND ----------

// function to compute inverse of (X^TX) for Dataframes

import breeze.linalg.{DenseMatrix, DenseVector, pinv}
def calculatePseudoinverse(XTX: DataFrame): DenseMatrix[Double] = {
  XTX.createOrReplaceTempView("XTX")
  
  val max_rows = spark.sql("select MAX(row) as max from XTX")
  val maxRow = max_rows.collect()(0).getAs[Int]("max") + 1
  val max_columns = spark.sql("select MAX(column) as max from XTX")
  val maxColumns = max_columns.collect()(0).getAs[Int]("max") + 1
  val breezeDenseMatrix = DenseMatrix.zeros[Double](maxRow, maxColumns)
  val XTX_array = XTX.collect()
  XTX_array.foreach { row =>
   breezeDenseMatrix(row.getAs[Int]("row"), row.getAs[Int]("column")) = row.getAs[Double]("value")
  }
  pinv(breezeDenseMatrix)
}

// COMMAND ----------

val pseudoInverseMatrix = calculatePseudoinverse(XTX)
println(pseudoInverseMatrix)


// COMMAND ----------


//Function to compute X^Ty for dataframes and converting the result to Breeze Vector

def calculateXTy(matrixDF: DataFrame, vectorDF: DataFrame): DenseVector[Double] = {
  val transposed_matrix = spark.sql(s"SELECT column AS row, row AS column, value FROM matrixDF")
  transposed_matrix.createOrReplaceTempView("transpose_matrix")
  vectorDF.createOrReplaceTempView("vectorDF")
  val Xty = spark.sql("select x.row as row , SUM(x.value*v.value) as value from transpose_matrix x, vectorDF v where x.column = v.row Group By x.row")

  val XTy_array = Xty.collect()
  val breeze_XTy = DenseVector.zeros[Double](XTy_array.length)
  XTy_array.foreach { row =>
  breeze_XTy(row.getAs[Int]("row")) = row.getAs[Double]("value")
  }
  breeze_XTy
}

// COMMAND ----------

val breeze_XTy = calculateXTy(matrixDF, vectorDF)
println(breeze_XTy)

// COMMAND ----------

// Perform matrix-vector multiplication
if (pseudoInverse.cols == breeze_XTy.length) {
  println("Multiplication result:")
  val final_result = pseudoInverse * breeze_XTy
  println(final_result)
} else {
  println("Dimension mismatch")
}


// COMMAND ----------

// MAGIC %md
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 2(10 pts)** \
// MAGIC Run both of your implementations (main project using RDDs, bonus 1 using Dataframes) on Boston Housing Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices?resource=download. Which implementation performs better?

// COMMAND ----------

//Bonus 2 Code

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val schema = StructType(Array(
  StructField("Combined", StringType, true)
))

// Uploaded the CSV file and reading it
val df = spark.read
  .option("header", "false")
  .schema(schema)
  .csv("/FileStore/tables/housing.csv")

//Processing the CSV file
val dfSplit = df.withColumn("SplitColumns", split(trim(regexp_replace(col("Combined"), "\\s+", " ")), " "))
  .select((0 until 14).map(i => col("SplitColumns").getItem(i).as(s"col_$i")): _*)

val dfWithoutLastCol = dfSplit.drop(dfSplit.columns.last)
val lastColumnName = dfSplit.columns.last
val dfLastCol = dfSplit.select(lastColumnName)


// COMMAND ----------


//Testing the data with RDD implementation.

val startTime = System.currentTimeMillis()

// Creating RDDs
val indexedRdd = dfWithoutLastCol.rdd.zipWithIndex.flatMap { case (row, rowIndex) =>
  row.toSeq.zipWithIndex.map { case (value, colIndex) =>
    ((rowIndex.toInt, colIndex.toInt), value.toString.toDouble)
  }
}

val rddLastCol = dfLastCol.rdd.zipWithIndex.map { case (row, rowIndex) =>
  (rowIndex.toInt, row.getString(0).toDouble)
}

val transpose_of_indexdRdd = indexedRdd.map{case ((i,j), value)=> ((j,i), value)}

// Calling Multiply Function to find (X^TX)
val XT_multiply_X = Multiply(transpose_of_indexdRdd, indexedRdd)

// Calling computePseudoinverse function to compute inverse of (X^TX)
val pseudoinverse = computePseudoinverse(XT_multiply_X)

// Calling MultiplyWithVector function to compute (X^Ty)
val XTransposeY = MultiplyWithVector(transpose_of_indexdRdd, rddLastCol)

// Calling createDenseVector function to create a dense vector 
val BreezeVector = createDenseVector(XTransposeY)

println("Multiplication result:")
val final_result = pseudoinverse * BreezeVector
println(final_result)
println()
val endTime = System.currentTimeMillis()
val elapsedTime = endTime - startTime
println(s"Time taken for RDD implementation: $elapsedTime milliseconds")
println()
println()

// COMMAND ----------


//Testing the data with Dataframes implementation.
val startTime = System.currentTimeMillis()
val matrixSchema = indexedRdd.map{case ((i,j),v) => Row(i,j,v)}
val vectorSchema = rddLastCol.map{case (i,v) => Row(i,v)}

// Creating Dataframes
val matrix_DF = spark.createDataFrame(matrixSchema, matrix_schema)
val vector_DF = spark.createDataFrame(vectorSchema, vector_schema)

matrix_DF.createOrReplaceTempView("matrix_DF")

vector_DF.createOrReplaceTempView("vector_DF")

// Calling calculateXTX Function to find (X^TX)
val xtx = calculateXTX(matrix_DF)

// Calling calculatePseudoinverse function to compute inverse of (X^TX)
val inverse_Matrix = calculatePseudoinverse(xtx)

// Calling calculateXTy function to compute inverse of (X^Ty) and convert it into breeze vector
val breeze = calculateXTy(matrix_DF, vector_DF)

val result = inverse_Matrix * breeze
println("Multiplication result:")
println(result)
println()
val endTime = System.currentTimeMillis()
val elapsedTime = endTime - startTime
println(s"Time taken: $elapsedTime milliseconds")
println()
println()

// COMMAND ----------

// MAGIC %md
// MAGIC Result: The Boston Housing Dataset performs better with an implementation using RDDs compared to one using DataFrames.
