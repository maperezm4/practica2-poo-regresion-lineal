#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <vector>
#include <string>

/**
 * @file MatrixOps.h
 * @brief Metodos auxiliares de procesamiento matricial y calculo estadistico.
 *
 * El enunciado prohibe el uso de librerias externas, por lo que todas las
 * operaciones de algebra lineal y estadistica se implementan desde cero
 * usando unicamente la biblioteca estandar de C++.
 */

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace MatrixOps {

// ---------------------------------------------------------------------------
// Procesamiento matricial
// ---------------------------------------------------------------------------

/**
 * @brief Devuelve la transpuesta de una matriz A (m x n) -> (n x m).
 * @param A Matriz de entrada.
 * @return Matriz transpuesta.
 */
Matrix transpose(const Matrix& A);

/**
 * @brief Multiplicacion matricial clasica C = A * B.
 * @param A Matriz (m x n).
 * @param B Matriz (n x p).
 * @return Matriz (m x p).
 * @throws std::invalid_argument si las dimensiones no son compatibles.
 */
Matrix multiply(const Matrix& A, const Matrix& B);

/**
 * @brief Producto matriz-vector y = A * x.
 * @param A Matriz (m x n).
 * @param x Vector de tamano n.
 * @return Vector de tamano m.
 */
Vector multiply(const Matrix& A, const Vector& x);

/**
 * @brief Producto punto entre dos vectores del mismo tamano.
 */
double dot(const Vector& a, const Vector& b);

/**
 * @brief Resuelve el sistema lineal A*x = b por eliminacion de Gauss-Jordan
 *        con pivoteo parcial.
 *
 * Se usa para la solucion exacta de la regresion por ecuaciones normales,
 * evitando calcular explicitamente la inversa (mas estable numericamente).
 *
 * @param A Matriz cuadrada (n x n).
 * @param b Vector independiente de tamano n.
 * @return Vector solucion x de tamano n.
 * @throws std::runtime_error si la matriz es singular.
 */
Vector solve(Matrix A, Vector b);

/**
 * @brief Inversa de una matriz cuadrada por Gauss-Jordan (uso didactico).
 */
Matrix inverse(Matrix A);

/**
 * @brief Extrae la columna j de una matriz.
 */
Vector column(const Matrix& A, std::size_t j);

/**
 * @brief Imprime una matriz con un rotulo (util para evidencias de prueba).
 */
void print(const Matrix& A, const std::string& label, std::size_t maxRows = 10);

/**
 * @brief Imprime un vector con un rotulo.
 */
void print(const Vector& v, const std::string& label, std::size_t maxItems = 20);

// ---------------------------------------------------------------------------
// Calculo estadistico
// ---------------------------------------------------------------------------

/** @brief Media aritmetica de un vector. */
double mean(const Vector& v);

/** @brief Desviacion estandar poblacional de un vector. */
double stddev(const Vector& v);

/** @brief Valor minimo de un vector. */
double minimum(const Vector& v);

/** @brief Valor maximo de un vector. */
double maximum(const Vector& v);

/** @brief Varianza poblacional de un vector. */
double variance(const Vector& v);

} // namespace MatrixOps

#endif // MATRIXOPS_H
