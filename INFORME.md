#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <cstddef>
#include <string>
#include <vector>

#include "MatrixOps.h"

/**
 * @file LinearRegression.h
 * @brief Framework orientado a objetos para regresion lineal simple y multiple.
 *
 * El mismo objeto sirve para regresion simple (una variable independiente) y
 * multiple (varias), porque internamente siempre se trabaja con una matriz
 * X de dimension (m x n): m muestras y n caracteristicas.
 *
 * Modelo:   y_hat = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
 */

/** @brief Estrategia de entrenamiento disponible. */
enum class Solver {
    GRADIENT_DESCENT,  ///< Descenso de gradiente iterativo.
    NORMAL_EQUATION    ///< Solucion cerrada (X^T X) w = X^T y.
};

/** @brief Estrategia de escalamiento de datos. */
enum class ScalingType {
    NONE,            ///< Sin escalamiento.
    STANDARDIZATION, ///< z = (x - media) / desviacion.
    MIN_MAX          ///< z = (x - min) / (max - min).
};

/** @brief Metricas de error devueltas por score(). */
struct Metrics {
    double mse  = 0.0;  ///< Error cuadratico medio.
    double rmse = 0.0;  ///< Raiz del error cuadratico medio.
    double mae  = 0.0;  ///< Error absoluto medio.
    double r2   = 0.0;  ///< Coeficiente de determinacion R^2.

    /** @brief Representacion legible de las metricas. */
    std::string toString() const;
};

/**
 * @class LinearRegression
 * @brief Modelo de regresion lineal multiple entrenable y reutilizable.
 */
class LinearRegression {
public:
    // -----------------------------------------------------------------------
    // Constructores y destructor
    // -----------------------------------------------------------------------

    /**
     * @brief Construye un modelo sin entrenar.
     * @param learningRate Tasa de aprendizaje del descenso de gradiente.
     * @param epochs Numero maximo de iteraciones de entrenamiento.
     * @param solver Metodo de ajuste (gradiente o ecuaciones normales).
     * @param tolerance Umbral de convergencia sobre el costo.
     */
    explicit LinearRegression(double learningRate = 0.01,
                              std::size_t epochs = 10000,
                              Solver solver = Solver::GRADIENT_DESCENT,
                              double tolerance = 1e-12);

    /** @brief Constructor de copia (copia profunda de los vectores). */
    LinearRegression(const LinearRegression& other);

    /** @brief Operador de asignacion por copia. */
    LinearRegression& operator=(const LinearRegression& other);

    /** @brief Destructor. */
    ~LinearRegression();

    // -----------------------------------------------------------------------
    // Metodos principales exigidos por el enunciado
    // -----------------------------------------------------------------------

    /**
     * @brief Entrena el modelo: calcula weights y bias a partir de los datos.
     * @param X Matriz de entrenamiento (m muestras x n caracteristicas).
     * @param y Vector de salidas reales (tamano m).
     * @throws std::invalid_argument si las dimensiones no coinciden.
     */
    void fit(const Matrix& X, const Vector& y);

    /**
     * @brief Predice la salida de un conjunto de muestras.
     * @param X Matriz de prueba (k x n).
     * @return Vector de predicciones y_hat (tamano k).
     * @throws std::runtime_error si el modelo no ha sido entrenado.
     */
    Vector predict(const Matrix& X) const;

    /**
     * @brief Predice la salida de una sola muestra.
     * @param x Vector de caracteristicas (tamano n).
     * @return Valor predicho.
     */
    double predict(const Vector& x) const;

    /**
     * @brief Calcula los errores de prediccion del modelo.
     * @param X Matriz de prueba.
     * @param y Vector de valores reales.
     * @return Estructura Metrics con MSE, RMSE, MAE y R^2.
     */
    Metrics score(const Matrix& X, const Vector& y) const;

    /**
     * @brief Escala los valores del dataset columna por columna.
     *
     * Si @p fitParameters es true, calcula y guarda los parametros de escala
     * (media/desviacion o min/max) de este dataset. Si es false, reutiliza los
     * parametros ya guardados: esto es obligatorio para escalar el conjunto de
     * prueba con la misma transformacion del conjunto de entrenamiento.
     *
     * @param X Matriz a escalar.
     * @param type Tipo de escalamiento.
     * @param fitParameters true para aprender los parametros, false para aplicarlos.
     * @return Matriz escalada (no modifica la original).
     */
    Matrix data_scaling(const Matrix& X,
                        ScalingType type = ScalingType::STANDARDIZATION,
                        bool fitParameters = true);

    // -----------------------------------------------------------------------
    // Getters y utilidades
    // -----------------------------------------------------------------------

    /** @brief Devuelve el arreglo de pesos del modelo. */
    const Vector& getWeights() const;

    /** @brief Devuelve el termino independiente (bias). */
    double getBias() const;

    /** @brief Indica si el modelo ya fue entrenado. */
    bool isTrained() const;

    /** @brief Numero de iteraciones realmente ejecutadas en el ajuste. */
    std::size_t getIterationsRun() const;

    /** @brief Costo (MSE) de la ultima iteracion de entrenamiento. */
    double getFinalCost() const;

    /** @brief Devuelve los pesos y el bias en formato legible. */
    std::string toString() const;

    /** @brief Imprime un informe del estado interno del objeto. */
    void summary() const;

private:
    // -------------------------------------------------------------- atributos
    Vector weights_;        ///< Arreglo con los pesos del modelo (exigido).
    double bias_;           ///< Termino independiente (exigido).

    double learningRate_;   ///< Tasa de aprendizaje.
    std::size_t epochs_;    ///< Iteraciones maximas.
    double tolerance_;      ///< Criterio de parada.
    Solver solver_;         ///< Metodo de ajuste.

    ScalingType scalingType_; ///< Escalamiento aplicado al entrenar.
    Vector scaleCenter_;      ///< Media (o minimo) por caracteristica.
    Vector scaleSpread_;      ///< Desviacion (o rango) por caracteristica.

    bool trained_;                ///< Bandera de modelo entrenado.
    std::size_t iterationsRun_;   ///< Iteraciones ejecutadas.
    double finalCost_;            ///< Costo final alcanzado.

    // --------------------------------------------------------- metodos privados

    /** @brief Ajuste por descenso de gradiente. */
    void fitGradientDescent(const Matrix& X, const Vector& y);

    /** @brief Ajuste por ecuaciones normales (X^T X) w = X^T y. */
    void fitNormalEquation(const Matrix& X, const Vector& y);

    /** @brief Costo MSE de las predicciones actuales. */
    double computeCost(const Matrix& X, const Vector& y) const;

    /** @brief Valida que X y y sean consistentes. */
    static void validate(const Matrix& X, const Vector& y);
};

#endif // LINEARREGRESSION_H

/**
 * @file main.cpp
 * @brief Punto 3.1 - Reto POO: regresion lineal simple y multiple.
 *
 * Uso:
 *   ./regresion                                  (usa los CSV de ./data/)
 *   ./regresion ruta_simple.csv ruta_multiple.csv
 *
 * Dataset simple  : ice_cream_selling_data.csv  (temperatura -> ventas)
 * Dataset multiple: student_exam_scores.csv     (horas, sueno, etc. -> nota)
 * En ambos se asume que la ULTIMA columna es la variable objetivo.
 */

#include <iomanip>
#include <iostream>
#include <string>

#include "CSVReader.h"
#include "LinearRegression.h"
#include "MatrixOps.h"

namespace {

void titulo(const std::string& texto) {
    std::cout << "\n==================================================\n"
              << texto
              << "\n==================================================\n";
}

/**
 * @brief Ejecuta el flujo completo del pseudocodigo del enunciado sobre un dataset.
 * @param path Ruta del CSV.
 * @param nombre Nombre descriptivo para los mensajes.
 * @param escalar true para aplicar data_scaling antes de entrenar.
 */
void ejecutarExperimento(const std::string& path, const std::string& nombre, bool escalar,
                         double learningRate = 0.05) {
    titulo(nombre);

    Dataset data;
    try {
        data = CSVReader::load(path);
    } catch (const std::exception& e) {
        std::cout << "[AVISO] " << e.what() << "\n"
                  << "        Descargue el archivo de EAFIT Interactiva y colóquelo en esa ruta.\n";
        return;
    }

    std::cout << "Archivo        : " << path << "\n"
              << "Muestras       : " << data.samples() << "\n"
              << "Caracteristicas: " << data.features() << "\n";
    if (!data.headers.empty()) {
        std::cout << "Columnas       : ";
        for (const auto& h : data.headers) std::cout << h << " ";
        std::cout << "\n";
    }

    // Estadisticas descriptivas (metodos de calculo estadistico propios).
    for (std::size_t j = 0; j < data.features(); ++j) {
        const Vector col = MatrixOps::column(data.X, j);
        std::cout << "  x" << (j + 1)
                  << ": media=" << std::fixed << std::setprecision(4) << MatrixOps::mean(col)
                  << " desv="   << MatrixOps::stddev(col)
                  << " min="    << MatrixOps::minimum(col)
                  << " max="    << MatrixOps::maximum(col) << "\n";
    }
    std::cout << "  y : media=" << MatrixOps::mean(data.y)
              << " desv="       << MatrixOps::stddev(data.y) << "\n";

    // ---------------- Division entrenamiento / prueba ----------------
    Matrix X_train, X_test;
    Vector y_train, y_test;
    CSVReader::trainTestSplit(data, 0.8, X_train, y_train, X_test, y_test);
    std::cout << "\nEntrenamiento: " << X_train.size()
              << " muestras | Prueba: " << X_test.size() << " muestras\n";

    // ---------------- Modelo 1: descenso de gradiente ----------------
    LinearRegression mlr(learningRate, 20000, Solver::GRADIENT_DESCENT);

    Matrix X_train_use = X_train;
    Matrix X_test_use  = X_test;
    if (escalar) {
        // Se aprenden los parametros con el train y se APLICAN al test.
        X_train_use = mlr.data_scaling(X_train, ScalingType::STANDARDIZATION, true);
        X_test_use  = mlr.data_scaling(X_test,  ScalingType::STANDARDIZATION, false);
        std::cout << "Escalamiento : estandarizacion z = (x - media)/desv\n";
    } else {
        std::cout << "Escalamiento : ninguno\n";
    }

    mlr.fit(X_train_use, y_train);

    std::cout << "\n--- Descenso de gradiente ---\n";
    MatrixOps::print(mlr.getWeights(), "weights");
    std::cout << "bias: " << std::fixed << std::setprecision(6) << mlr.getBias() << "\n";
    std::cout << "iteraciones: " << mlr.getIterationsRun() << "\n";
    std::cout << "learning rate: " << learningRate << "\n";

    const Vector y_hat = mlr.predict(X_test_use);
    std::cout << "\nPredicciones sobre el conjunto de prueba:\n";
    const std::size_t lim = (y_hat.size() < 10) ? y_hat.size() : 10;
    for (std::size_t i = 0; i < lim; ++i)
        std::cout << "  y_hat: " << std::setw(10) << y_hat[i]
                  << "   y_real: " << std::setw(10) << y_test[i]
                  << "   error: " << std::setw(10) << (y_hat[i] - y_test[i]) << "\n";
    if (lim < y_hat.size())
        std::cout << "  ... (" << y_hat.size() - lim << " predicciones mas)\n";

    std::cout << "\nscore (train): " << mlr.score(X_train_use, y_train).toString() << "\n";
    std::cout << "score (test) : " << mlr.score(X_test_use,  y_test ).toString() << "\n";

    // ---------------- Modelo 2: ecuaciones normales (verificacion) ----------------
    LinearRegression exacto(0.0, 0, Solver::NORMAL_EQUATION);
    exacto.fit(X_train_use, y_train);
    std::cout << "\n--- Ecuaciones normales (solucion cerrada, para verificar) ---\n";
    MatrixOps::print(exacto.getWeights(), "weights");
    std::cout << "bias: " << exacto.getBias() << "\n";
    std::cout << "score (test) : " << exacto.score(X_test_use, y_test).toString() << "\n";
    std::cout << "Modelo: " << exacto.toString() << "\n";
}

/**
 * @brief Ajusta un polinomio de grado 2 reutilizando la MISMA clase LinearRegression.
 *
 * La regresion lineal lo es respecto a los PARAMETROS, no respecto a las variables.
 * Si a la matriz X de una sola caracteristica t le agregamos la columna t^2, el modelo
 * y = w1*t + w2*t^2 + b sigue siendo lineal en w y b, y se resuelve con el mismo codigo.
 * No se modifica ni una linea de la clase.
 *
 * @param path Ruta del CSV de una sola caracteristica.
 */
void ejecutarPolinomica(const std::string& path) {
    titulo("1b) REGRESION POLINOMICA GRADO 2 (misma clase, caracteristica derivada)");

    Dataset data;
    try {
        data = CSVReader::load(path);
    } catch (const std::exception& e) {
        std::cout << "[AVISO] " << e.what() << "\n";
        return;
    }
    if (data.features() != 1) {
        std::cout << "[AVISO] Este experimento espera un dataset de una sola caracteristica.\n";
        return;
    }

    // Se construye la matriz ampliada [t, t^2].
    Dataset ampliado;
    ampliado.headers = {"t", "t^2", data.headers.empty() ? "y" : data.headers.back()};
    ampliado.y = data.y;
    for (const auto& fila : data.X) {
        const double t = fila[0];
        ampliado.X.push_back({t, t * t});
    }

    Matrix X_train, X_test;
    Vector y_train, y_test;
    CSVReader::trainTestSplit(ampliado, 0.8, X_train, y_train, X_test, y_test);

    LinearRegression modelo(0.05, 20000, Solver::GRADIENT_DESCENT);
    const Matrix X_train_s = modelo.data_scaling(X_train, ScalingType::STANDARDIZATION, true);
    const Matrix X_test_s  = modelo.data_scaling(X_test,  ScalingType::STANDARDIZATION, false);
    modelo.fit(X_train_s, y_train);

    MatrixOps::print(modelo.getWeights(), "weights [t, t^2]");
    std::cout << "bias: " << std::fixed << std::setprecision(6) << modelo.getBias() << "\n"
              << "iteraciones: " << modelo.getIterationsRun() << "\n";

    std::cout << "\nPredicciones sobre el conjunto de prueba:\n";
    const Vector y_hat = modelo.predict(X_test_s);
    for (std::size_t i = 0; i < y_hat.size() && i < 10; ++i)
        std::cout << "  t: " << std::setw(10) << X_test[i][0]
                  << "   y_hat: " << std::setw(10) << y_hat[i]
                  << "   y_real: " << std::setw(10) << y_test[i]
                  << "   error: " << std::setw(10) << (y_hat[i] - y_test[i]) << "\n";

    std::cout << "\nscore (train): " << modelo.score(X_train_s, y_train).toString() << "\n"
              << "score (test) : " << modelo.score(X_test_s,  y_test ).toString() << "\n";

    // Coeficientes equivalentes en las unidades originales (sin escalar).
    LinearRegression exacto(0.0, 0, Solver::NORMAL_EQUATION);
    exacto.fit(X_train, y_train);
    std::cout << "\nModelo en unidades originales: " << exacto.toString()
              << "   (x1 = t, x2 = t^2)\n"
              << "score (test) : " << exacto.score(X_test, y_test).toString() << "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string rutaSimple   = "data/ice_cream_selling_data.csv";
    std::string rutaMultiple = "data/student_exam_scores.csv";
    if (argc >= 2) rutaSimple   = argv[1];
    if (argc >= 3) rutaMultiple = argv[2];

    std::cout << "PRACTICA 2 - PROGRAMACION ORIENTADA A OBJETOS\n"
              << "Framework de regresion lineal multiple en C++\n";

    try {
        // Regresion simple sin escalar: requiere una tasa de aprendizaje pequena.
        ejecutarExperimento(rutaSimple,   "1) REGRESION LINEAL SIMPLE (ice cream)",      false, 0.001);
        // El dataset de helados NO es lineal: se amplia con t^2 usando la misma clase.
        ejecutarPolinomica(rutaSimple);
        // Regresion multiple con escalamiento: admite una tasa mucho mayor.
        ejecutarExperimento(rutaMultiple, "2) REGRESION LINEAL MULTIPLE (student exam)", true,  0.05);

        // --- Evidencia del problema documentado en el informe ---
        titulo("3) PRUEBA DE ROBUSTEZ: DIVERGENCIA SIN ESCALAMIENTO");
        try {
            Dataset d = CSVReader::load(rutaMultiple);
            LinearRegression malConfigurado(0.05, 20000, Solver::GRADIENT_DESCENT);
            malConfigurado.fit(d.X, d.y);   // datos sin escalar + lr grande
            std::cout << "El ajuste convergio (no se esperaba).\n";
        } catch (const std::exception& e) {
            std::cout << "Excepcion capturada correctamente -> " << e.what() << "\n"
                      << "Esto justifica la existencia del metodo data_scaling().\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nFin de la ejecucion.\n";
    return 0;
}

CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
OBJS     = MatrixOps.o LinearRegression.o CSVReader.o

all: regresion memoria

regresion: $(OBJS) main.o
	$(CXX) $(CXXFLAGS) -o regresion $(OBJS) main.o

memoria: $(OBJS) memory_analysis.o
	$(CXX) $(CXXFLAGS) -o memoria $(OBJS) memory_analysis.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	./regresion
	./memoria

clean:
	rm -f *.o regresion memoria

.PHONY: all run clean

#include "MatrixOps.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace MatrixOps {

Matrix transpose(const Matrix& A) {
    if (A.empty()) return {};
    const std::size_t m = A.size();
    const std::size_t n = A[0].size();
    Matrix T(n, Vector(m, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

Matrix multiply(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty())
        throw std::invalid_argument("multiply: matriz vacia");
    const std::size_t m = A.size();
    const std::size_t n = A[0].size();
    const std::size_t p = B[0].size();
    if (B.size() != n)
        throw std::invalid_argument("multiply: dimensiones incompatibles");

    Matrix C(m, Vector(p, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            const double aik = A[i][k];
            for (std::size_t j = 0; j < p; ++j)
                C[i][j] += aik * B[k][j];
        }
    return C;
}

Vector multiply(const Matrix& A, const Vector& x) {
    if (A.empty()) return {};
    if (A[0].size() != x.size())
        throw std::invalid_argument("multiply(A,x): dimensiones incompatibles");
    Vector y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("dot: tamanos distintos");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

Vector solve(Matrix A, Vector b) {
    const std::size_t n = A.size();
    if (n == 0 || A[0].size() != n || b.size() != n)
        throw std::invalid_argument("solve: el sistema no es cuadrado");

    for (std::size_t col = 0; col < n; ++col) {
        // Pivoteo parcial: se busca el mayor valor absoluto de la columna.
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[pivot][col])) pivot = r;

        if (std::fabs(A[pivot][col]) < 1e-12)
            throw std::runtime_error("solve: matriz singular o mal condicionada");

        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);

        // Normalizacion de la fila pivote.
        const double p = A[col][col];
        for (std::size_t j = col; j < n; ++j) A[col][j] /= p;
        b[col] /= p;

        // Eliminacion del resto de filas.
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = A[r][col];
            if (factor == 0.0) continue;
            for (std::size_t j = col; j < n; ++j) A[r][j] -= factor * A[col][j];
            b[r] -= factor * b[col];
        }
    }
    return b;
}

Matrix inverse(Matrix A) {
    const std::size_t n = A.size();
    if (n == 0 || A[0].size() != n)
        throw std::invalid_argument("inverse: la matriz no es cuadrada");

    Matrix I(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) I[i][i] = 1.0;

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[pivot][col])) pivot = r;
        if (std::fabs(A[pivot][col]) < 1e-12)
            throw std::runtime_error("inverse: matriz singular");

        std::swap(A[col], A[pivot]);
        std::swap(I[col], I[pivot]);

        const double p = A[col][col];
        for (std::size_t j = 0; j < n; ++j) { A[col][j] /= p; I[col][j] /= p; }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            if (f == 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                A[r][j] -= f * A[col][j];
                I[r][j] -= f * I[col][j];
            }
        }
    }
    return I;
}

Vector column(const Matrix& A, std::size_t j) {
    Vector c;
    c.reserve(A.size());
    for (const auto& row : A) c.push_back(row.at(j));
    return c;
}

void print(const Matrix& A, const std::string& label, std::size_t maxRows) {
    std::cout << label << " (" << A.size() << " x "
              << (A.empty() ? 0 : A[0].size()) << ")\n";
    const std::size_t lim = std::min(maxRows, A.size());
    for (std::size_t i = 0; i < lim; ++i) {
        std::cout << "  [ ";
        for (double v : A[i]) std::cout << std::fixed << std::setprecision(4) << v << " ";
        std::cout << "]\n";
    }
    if (lim < A.size()) std::cout << "  ... (" << A.size() - lim << " filas mas)\n";
}

void print(const Vector& v, const std::string& label, std::size_t maxItems) {
    std::cout << label << " (n=" << v.size() << "): [ ";
    const std::size_t lim = std::min(maxItems, v.size());
    for (std::size_t i = 0; i < lim; ++i)
        std::cout << std::fixed << std::setprecision(4) << v[i] << " ";
    if (lim < v.size()) std::cout << "...";
    std::cout << "]\n";
}

double mean(const Vector& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double variance(const Vector& v) {
    if (v.empty()) return 0.0;
    const double mu = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - mu) * (x - mu);
    return s / static_cast<double>(v.size());
}

double stddev(const Vector& v) { return std::sqrt(variance(v)); }

double minimum(const Vector& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) if (x < m) m = x;
    return m;
}

double maximum(const Vector& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) if (x > m) m = x;
    return m;
}

} // namespace MatrixOps

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

/**
 * @file memory_analysis.cpp
 * @brief Punto 3.2 - Analisis de memoria.
 *
 * Crea objetos LinearRegression en memoria GLOBAL, LOCAL (stack) y DINAMICA
 * (heap), imprime sus direcciones y tamanos, y justifica los valores.
 * Tambien crea un arreglo dinamico de objetos y calcula su tamano real.
 */

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "LinearRegression.h"

// ---------------------------------------------------------------------------
// 1. MEMORIA GLOBAL / ESTATICA
//    Vive en el segmento de datos. Se construye antes de main() y se destruye
//    al terminar el programa. Su direccion es fija durante toda la ejecucion.
// ---------------------------------------------------------------------------
LinearRegression modeloGlobal(0.01, 1000);
static LinearRegression modeloEstatico(0.02, 500);

namespace {

/** @brief Imprime una direccion en hexadecimal con un rotulo. */
void mostrar(const std::string& etiqueta, const void* dir, std::size_t tam) {
    std::cout << std::left << std::setw(34) << etiqueta
              << " dir: " << dir
              << "   sizeof: " << std::dec << tam << " bytes\n";
}

void separador(const std::string& t) {
    std::cout << "\n--------------------------------------------------\n"
              << t
              << "\n--------------------------------------------------\n";
}

/** @brief Desglosa el tamano estatico de la clase miembro por miembro. */
void desgloseDeTamano() {
    separador("DESGLOSE DEL TAMANO DE LinearRegression");
    std::cout << "sizeof(LinearRegression) = " << sizeof(LinearRegression) << " bytes\n\n"
              << "Composicion aproximada (plataforma de 64 bits):\n"
              << "  weights_      std::vector<double>  : " << sizeof(std::vector<double>) << " bytes\n"
              << "  bias_         double               : " << sizeof(double) << " bytes\n"
              << "  learningRate_ double               : " << sizeof(double) << " bytes\n"
              << "  epochs_       size_t               : " << sizeof(std::size_t) << " bytes\n"
              << "  tolerance_    double               : " << sizeof(double) << " bytes\n"
              << "  solver_       enum class           : " << sizeof(Solver) << " bytes\n"
              << "  scalingType_  enum class           : " << sizeof(ScalingType) << " bytes\n"
              << "  scaleCenter_  std::vector<double>  : " << sizeof(std::vector<double>) << " bytes\n"
              << "  scaleSpread_  std::vector<double>  : " << sizeof(std::vector<double>) << " bytes\n"
              << "  trained_      bool                 : " << sizeof(bool) << " bytes\n"
              << "  iterationsRun_ size_t              : " << sizeof(std::size_t) << " bytes\n"
              << "  finalCost_    double               : " << sizeof(double) << " bytes\n"
              << "\nJUSTIFICACION:\n"
              << "  * sizeof() mide SOLO el tamano estatico del objeto, no los datos\n"
              << "    que sus vectores reservan en el heap.\n"
              << "  * Cada std::vector ocupa 24 bytes (tres punteros de 8 bytes:\n"
              << "    inicio, fin y fin de capacidad), sin importar cuantos elementos\n"
              << "    almacene. Los elementos viven aparte, en memoria dinamica.\n"
              << "  * El compilador aplica ALINEACION (padding): el bool de 1 byte se\n"
              << "    rellena hasta completar un multiplo de 8, que es la alineacion\n"
              << "    exigida por los double y los punteros.\n"
              << "  * Por eso la suma ingenua de los miembros puede ser menor que\n"
              << "    sizeof(LinearRegression).\n";
}

/** @brief Memoria realmente ocupada por un objeto, incluyendo el heap de sus vectores. */
std::size_t memoriaTotal(const LinearRegression& m) {
    return sizeof(LinearRegression) + m.getWeights().capacity() * sizeof(double);
}

} // namespace

int main() {
    std::cout << "PRACTICA 2 - PUNTO 3.2: ANALISIS DE MEMORIA\n";
    std::cout << "Plataforma: punteros de " << sizeof(void*) * 8 << " bits\n";

    // -----------------------------------------------------------------------
    separador("1. OBJETOS EN MEMORIA GLOBAL / ESTATICA");
    // -----------------------------------------------------------------------
    mostrar("modeloGlobal (global)",   &modeloGlobal,   sizeof(modeloGlobal));
    mostrar("modeloEstatico (static)", &modeloEstatico, sizeof(modeloEstatico));
    std::cout << "Zona: segmento de datos (.data/.bss). Direcciones bajas y fijas,\n"
                 "determinadas al enlazar. Tiempo de vida: todo el programa.\n";

    // -----------------------------------------------------------------------
    separador("2. OBJETOS EN MEMORIA LOCAL (STACK)");
    // -----------------------------------------------------------------------
    LinearRegression modeloLocal1(0.03, 2000);
    LinearRegression modeloLocal2(0.04, 3000);
    mostrar("modeloLocal1 (stack)", &modeloLocal1, sizeof(modeloLocal1));
    mostrar("modeloLocal2 (stack)", &modeloLocal2, sizeof(modeloLocal2));

    const std::intptr_t d1 = reinterpret_cast<std::intptr_t>(&modeloLocal1);
    const std::intptr_t d2 = reinterpret_cast<std::intptr_t>(&modeloLocal2);
    std::cout << "Distancia entre ambos: " << (d1 > d2 ? d1 - d2 : d2 - d1) << " bytes\n";
    std::cout << "Zona: pila de la funcion. Direcciones altas que DECRECEN conforme\n"
                 "se declaran variables. Se liberan solas al salir del ambito.\n";

    // -----------------------------------------------------------------------
    separador("3. OBJETOS EN MEMORIA DINAMICA (HEAP)");
    // -----------------------------------------------------------------------
    LinearRegression* modeloDinamico1 = new LinearRegression(0.05, 4000);
    LinearRegression* modeloDinamico2 = new LinearRegression(0.06, 5000);

    mostrar("modeloDinamico1 (heap)", modeloDinamico1, sizeof(*modeloDinamico1));
    mostrar("modeloDinamico2 (heap)", modeloDinamico2, sizeof(*modeloDinamico2));
    std::cout << "Direccion del PUNTERO modeloDinamico1 (esta en el stack): "
              << &modeloDinamico1 << "\n"
              << "sizeof(puntero) = " << sizeof(modeloDinamico1)
              << " bytes (solo guarda la direccion, no el objeto)\n"
              << "Zona: heap. Direcciones intermedias y CRECIENTES. El objeto vive\n"
                 "hasta que se invoque delete: si se olvida, hay fuga de memoria.\n";

    // -----------------------------------------------------------------------
    separador("4. VECTOR DE LinearRegression EN MEMORIA DINAMICA");
    // -----------------------------------------------------------------------
    const std::size_t N = 5;

    // (a) Arreglo dinamico clasico con new[].
    LinearRegression* arreglo = new LinearRegression[N];
    std::cout << "(a) new LinearRegression[" << N << "]\n";
    for (std::size_t i = 0; i < N; ++i)
        std::cout << "    arreglo[" << i << "] -> " << &arreglo[i] << "\n";

    const std::size_t tamArreglo = N * sizeof(LinearRegression);
    std::cout << "    sizeof(arreglo) (puntero) : " << sizeof(arreglo) << " bytes\n"
              << "    Tamano real del bloque    : " << N << " * "
              << sizeof(LinearRegression) << " = " << tamArreglo << " bytes\n";

    const std::intptr_t p0 = reinterpret_cast<std::intptr_t>(&arreglo[0]);
    const std::intptr_t p1 = reinterpret_cast<std::intptr_t>(&arreglo[1]);
    std::cout << "    Distancia entre elementos consecutivos: " << (p1 - p0)
              << " bytes (coincide con sizeof(LinearRegression):\n"
                 "    los objetos son contiguos en el heap)\n";

    // (b) std::vector de objetos, tambien en memoria dinamica.
    std::vector<LinearRegression> vec(N);
    std::cout << "\n(b) std::vector<LinearRegression> vec(" << N << ")\n"
              << "    sizeof(vec)       : " << sizeof(vec)
              << " bytes (el objeto vector en el stack: 3 punteros)\n"
              << "    vec.size()        : " << vec.size() << "\n"
              << "    vec.capacity()    : " << vec.capacity() << "\n"
              << "    Memoria del buffer: " << vec.capacity() * sizeof(LinearRegression)
              << " bytes en el heap\n"
              << "    Direccion de vec (stack): " << &vec
              << " | Direccion de vec.data() (heap): " << vec.data() << "\n";

    std::cout << "\nJUSTIFICACION DEL TAMANO:\n"
              << "  * sizeof(arreglo) = 8 porque es un PUNTERO, no el bloque.\n"
              << "  * El bloque reservado es N * sizeof(LinearRegression) = "
              << tamArreglo << " bytes,\n"
              << "    mas unos pocos bytes ocultos de metadatos del asignador\n"
              << "    (y el conteo de elementos que new[] guarda para delete[]).\n"
              << "  * Eso tampoco incluye lo que cada objeto reserve despues en sus\n"
              << "    vectores internos al entrenarse.\n";

    // -----------------------------------------------------------------------
    separador("5. MEMORIA ESTATICA VS MEMORIA REAL DESPUES DE ENTRENAR");
    // -----------------------------------------------------------------------
    Matrix X = {{1.0, 2.0}, {2.0, 1.0}, {3.0, 4.0}, {4.0, 3.0}, {5.0, 6.0}};
    Vector y = {5.0, 4.0, 11.0, 10.0, 17.0};

    std::cout << "Antes de fit(): sizeof=" << sizeof(modeloLocal1)
              << " | memoria total=" << memoriaTotal(modeloLocal1) << " bytes\n";
    modeloLocal1.fit(X, y);
    std::cout << "Despues de fit(): sizeof=" << sizeof(modeloLocal1)
              << " | memoria total=" << memoriaTotal(modeloLocal1) << " bytes\n"
              << "El sizeof NO cambia; lo que crece es el bloque del heap apuntado\n"
              << "por weights_ (" << modeloLocal1.getWeights().size()
              << " pesos * 8 bytes).\n";
    modeloLocal1.summary();

    // -----------------------------------------------------------------------
    separador("6. MAPA DE MEMORIA DEL PROCESO (resumen de direcciones)");
    // -----------------------------------------------------------------------
    std::cout << "  Global/estatica : " << &modeloGlobal    << "\n"
              << "  Heap (new)      : " << modeloDinamico1  << "\n"
              << "  Heap (new[])    : " << arreglo          << "\n"
              << "  Stack (local)   : " << &modeloLocal1    << "\n"
              << "\nDe menor a mayor direccion: codigo < datos globales < heap < stack.\n"
                 "El heap crece hacia arriba y el stack hacia abajo.\n"
                 "NOTA: con ASLR activo las direcciones cambian en cada ejecucion,\n"
                 "pero el ORDEN relativo de las regiones se mantiene.\n";

    // -----------------------------------------------------------------------
    separador("7. LIBERACION DE MEMORIA");
    // -----------------------------------------------------------------------
    delete modeloDinamico1;  // new  -> delete
    delete modeloDinamico2;
    delete[] arreglo;        // new[] -> delete[]  (nunca mezclar)
    modeloDinamico1 = nullptr;
    modeloDinamico2 = nullptr;
    arreglo         = nullptr;

    std::cout << "Memoria dinamica liberada correctamente.\n"
              << "Reglas aplicadas: cada new con su delete, cada new[] con su\n"
                 "delete[], y los punteros se dejan en nullptr para evitar\n"
                 "punteros colgantes (dangling pointers) y dobles liberaciones.\n"
              << "El std::vector libera su buffer solo (RAII) al salir del ambito.\n";

    desgloseDeTamano();

    std::cout << "\nFin del analisis de memoria.\n";
    return 0;
}

# Práctica 2 — Programación Orientada a Objetos (C++)
Regresión lineal simple y múltiple implementada desde cero, sin librerías externas.

## Compilar y ejecutar

```bash
make            # genera ./regresion y ./memoria
./regresion     # punto 3.1 — entrenamiento, predicción y métricas
./memoria       # punto 3.2 — análisis de memoria
make run        # ejecuta ambos
make clean      # borra binarios y objetos
```

## Datasets

Descargar de EAFIT Interactiva y colocar en `data/`:

- `data/ice_cream_selling_data.csv` — regresión simple
- `data/student_exam_scores.csv` — regresión múltiple

Formato esperado: CSV con encabezado; **la última columna es la variable objetivo**.
También se pueden pasar otras rutas:

```bash
./regresion mi_simple.csv mi_multiple.csv
```

> Los CSV incluidos en `data/` son de prueba, generados sintéticamente para verificar
> que el código corre. **Reemplazarlos por los oficiales antes de entregar** y regenerar
> las salidas con `make run`.

## Archivos

| Archivo | Contenido |
|---|---|
| `LinearRegression.h/.cpp` | Clase del modelo: `fit`, `predict`, `score`, `data_scaling` |
| `MatrixOps.h/.cpp` | Álgebra matricial y estadística propias |
| `CSVReader.h/.cpp` | Carga de CSV y división train/test |
| `main.cpp` | Punto 3.1 |
| `memory_analysis.cpp` | Punto 3.2 |
| `INFORME.md` | Informe completo de la práctica |
| `evidencias/` | Salidas de consola de las pruebas |

## Uso de la clase

```cpp
LinearRegression mlr(0.05, 20000, Solver::GRADIENT_DESCENT);

Matrix Xs = mlr.data_scaling(X_train, ScalingType::STANDARDIZATION, true);
mlr.fit(Xs, y_train);

std::cout << "weights: "; MatrixOps::print(mlr.getWeights(), "");
std::cout << "bias: " << mlr.getBias() << "\n";

Matrix Xt = mlr.data_scaling(X_test, ScalingType::STANDARDIZATION, false);
Vector y_hat = mlr.predict(Xt);
std::cout << "score: " << mlr.score(Xt, y_test).toString() << "\n";
```

# Binarios y objetos
*.o
*.obj
*.exe
regresion
memoria

# Artefactos de compilación
*.out
*.a
*.so
*.dSYM/

# Editores / sistema
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db

#include "CSVReader.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

/** @brief Elimina espacios, tabulaciones y retornos de carro de los extremos. */
std::string trim(const std::string& s) {
    const std::string ws = " \t\r\n\"";
    const std::size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    const std::size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

/** @brief Separa una linea en campos segun el delimitador. */
std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream is(line);
    while (std::getline(is, field, delimiter)) fields.push_back(trim(field));
    return fields;
}

} // namespace

namespace CSVReader {

Dataset load(const std::string& path, bool hasHeader, char delimiter) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("CSVReader: no se pudo abrir el archivo " + path);

    Dataset data;
    std::string line;
    bool first = true;

    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;

        std::vector<std::string> fields = split(line, delimiter);

        if (first && hasHeader) {
            data.headers = fields;
            first = false;
            continue;
        }
        first = false;

        if (fields.size() < 2) continue;  // se ignoran filas incompletas

        Vector row;
        row.reserve(fields.size() - 1);
        bool valid = true;

        for (std::size_t j = 0; j + 1 < fields.size(); ++j) {
            if (fields[j].empty()) { valid = false; break; }
            row.push_back(std::atof(fields[j].c_str()));
        }
        if (!valid || fields.back().empty()) continue;

        data.X.push_back(row);
        data.y.push_back(std::atof(fields.back().c_str()));
    }

    if (data.X.empty())
        throw std::runtime_error("CSVReader: el archivo " + path + " no contiene datos validos");

    return data;
}

void trainTestSplit(const Dataset& data, double trainRatio,
                    Matrix& XTrain, Vector& yTrain,
                    Matrix& XTest,  Vector& yTest) {
    XTrain.clear(); yTrain.clear(); XTest.clear(); yTest.clear();

    const std::size_t m = data.X.size();
    if (m < 2) throw std::runtime_error("trainTestSplit: se necesitan al menos 2 muestras");

    // Muestreo sistematico: se toma una de cada 'paso' muestras para prueba.
    // A diferencia de un corte por la mitad, esto funciona aunque el archivo venga
    // ORDENADO por alguna columna (como ice_cream_selling_data.csv, ordenado por
    // temperatura): un corte simple dejaria todas las temperaturas altas en test.
    // Es deterministico, asi que los resultados son reproducibles.
    double testRatio = 1.0 - trainRatio;
    if (testRatio <= 0.0) testRatio = 0.2;
    std::size_t paso = static_cast<std::size_t>(1.0 / testRatio + 0.5);
    if (paso < 2) paso = 2;

    for (std::size_t i = 0; i < m; ++i) {
        if (i % paso == paso - 1) { XTest.push_back(data.X[i]);  yTest.push_back(data.y[i]);  }
        else                      { XTrain.push_back(data.X[i]); yTrain.push_back(data.y[i]); }
    }

    if (XTest.empty()) { XTest.push_back(XTrain.back()); yTest.push_back(yTrain.back()); }
}

} // namespace CSVReader

#ifndef CSVREADER_H
#define CSVREADER_H

#include <string>
#include <vector>

#include "MatrixOps.h"

/**
 * @file CSVReader.h
 * @brief Utilidades propias para cargar datasets desde archivos .csv.
 *
 * No se usan librerias externas: el parseo se hace con std::ifstream y
 * manipulacion manual de cadenas.
 */

/** @brief Dataset cargado desde un archivo CSV. */
struct Dataset {
    std::vector<std::string> headers; ///< Nombres de las columnas.
    Matrix X;                         ///< Caracteristicas (m x n).
    Vector y;                         ///< Variable objetivo (m).

    std::size_t samples()  const { return X.size(); }
    std::size_t features() const { return X.empty() ? 0 : X[0].size(); }
};

namespace CSVReader {

/**
 * @brief Carga un CSV asumiendo que la ultima columna es la variable objetivo.
 * @param path Ruta del archivo.
 * @param hasHeader true si la primera linea contiene los nombres de columna.
 * @param delimiter Separador de campos.
 * @return Dataset con X, y y encabezados.
 * @throws std::runtime_error si el archivo no se puede abrir.
 */
Dataset load(const std::string& path, bool hasHeader = true, char delimiter = ',');

/**
 * @brief Divide un dataset en entrenamiento y prueba de forma deterministica.
 * @param data Dataset completo.
 * @param trainRatio Proporcion destinada a entrenamiento (0 a 1).
 * @param XTrain Salida: matriz de entrenamiento.
 * @param yTrain Salida: objetivo de entrenamiento.
 * @param XTest Salida: matriz de prueba.
 * @param yTest Salida: objetivo de prueba.
 */
void trainTestSplit(const Dataset& data, double trainRatio,
                    Matrix& XTrain, Vector& yTrain,
                    Matrix& XTest,  Vector& yTest);

} // namespace CSVReader

#endif // CSVREADER_H

# Guion del video — Práctica 2 (POO en C++)
**Duración objetivo: 9:55 (máximo permitido 10:00)** — margen ajustado: ensayar con cronómetro.
Participantes: **Miguel Ángel Pérez Marín** (Miguel) y **Julio Cesar Llanes Cuello** (Julio).
Ambos con el rostro visible durante todo el video.

> **Antes de grabar:** tener abiertos y listos — el editor con los archivos, una terminal en la
> carpeta del proyecto, y el informe. Hacer `make clean` antes para que la compilación se vea en
> vivo. Ensayar una vez con cronómetro: el bloque que más se alarga siempre es el 4.

---

## Bloque 0 — Presentación (0:00 – 0:40) · **Miguel y Julio en cámara**

**Miguel:**
> Hola, soy Miguel Ángel Pérez Marín, y les presentamos la Práctica 2 de Lenguajes y Paradigmas de
> la Computación, sobre programación orientada a objetos.

**Julio:**
> Y yo soy Julio Cesar Llanes Cuello. Lo que hicimos fue un framework de regresión lineal múltiple
> en C++, completamente desde cero: sin librerías externas, implementando nosotros mismos el
> álgebra matricial, la estadística y la lectura de archivos.

**Miguel:**
> El video tiene tres partes: primero el diseño de la clase, después la ejecución con los
> datasets, y al final el análisis de memoria junto con la investigación de fallos históricos.

---

## Bloque 1 — Arquitectura y decisiones de diseño (0:40 – 2:10) · **Miguel**
*En pantalla: estructura de carpetas y `LinearRegression.h`.*

**Miguel:**
> Empiezo por la estructura. Separamos el proyecto en tres responsabilidades. `MatrixOps` tiene
> el álgebra matricial y la estadística: transpuesta, multiplicación, solución de sistemas
> lineales, media, desviación. `CSVReader` se encarga de leer los datasets y dividirlos en
> entrenamiento y prueba. Y `LinearRegression` es el modelo propiamente dicho.
>
> Esa separación es intencional: el modelo no sabe leer archivos y no sabe invertir matrices.
> Si mañana cambiamos el formato de entrada, solo tocamos `CSVReader`.
>
> *(mostrando el header)* La clase encapsula los dos atributos que pide el enunciado: `weights`,
> que es el arreglo de pesos, y `bias`, el término independiente. Los dos son **privados**. Desde
> afuera solo se consultan con `getWeights()` y `getBias()`, nunca se pueden modificar
> directamente. Eso es encapsulamiento: es imposible dejar el modelo en un estado inconsistente
> desde fuera de la clase.
>
> Una decisión que quiero resaltar: **una sola clase resuelve regresión simple y múltiple**.
> Internamente siempre trabajamos con una matriz X de m por n, donde m son las muestras y n las
> características. La regresión simple no es un caso aparte, es simplemente n igual a uno. No
> necesitamos dos clases ni dos algoritmos.
>
> También implementamos la regla de los tres: constructor de copia, operador de asignación y
> destructor explícitos, para que al copiar un modelo se dupliquen correctamente sus vectores.

---

## Bloque 2 — Los cuatro métodos exigidos (2:10 – 4:10) · **Julio**
*En pantalla: `LinearRegression.cpp`, desplazándose por cada método.*

**Julio:**
> Yo explico los cuatro métodos que pide la práctica.
>
> **`fit`** entrena el modelo. Valida las dimensiones, inicializa pesos y bias en cero, y delega
> en uno de dos solvers. El principal es **descenso de gradiente**: en cada época calcula el
> gradiente del error cuadrático medio —dos sobre m, por la suma del error por cada
> característica— y mueve los pesos en dirección contraria al gradiente. Para cuando se agotan
> las épocas o cuando el costo deja de cambiar.
>
> El segundo solver son las **ecuaciones normales**, la solución cerrada: X transpuesta por X,
> por theta, igual a X transpuesta por y. Agregamos una columna de unos a X para que el bias
> quede dentro del vector de parámetros, y resolvemos el sistema con Gauss-Jordan con pivoteo
> parcial. Importante: **no calculamos la inversa**, resolvemos el sistema directamente, porque
> es numéricamente más estable.
>
> ¿Por qué dos solvers? Porque el segundo **valida** al primero. Ya van a ver que dan el mismo
> resultado, y esa coincidencia es nuestra prueba de que la implementación está correcta.
>
> **`predict`** está sobrecargado en dos versiones: una recibe una sola muestra y devuelve un
> número, la otra recibe una matriz y devuelve el vector de predicciones. La segunda reutiliza
> la primera.
>
> **`score`** calcula los errores de predicción. Devolvemos cuatro métricas en una estructura:
> MSE, que penaliza los errores grandes; RMSE, que es el error típico en las unidades de y; MAE,
> el error absoluto medio; y R cuadrado, que es la proporción de varianza explicada: uno sería
> un ajuste perfecto.
>
> **`data_scaling`** escala el dataset columna por columna, con estandarización —restar la media
> y dividir por la desviación— o min-max. Y aquí está el detalle de diseño más importante de
> todo el método: el parámetro `fitParameters`.
>
> *(señalando el código)* Cuando es verdadero, calcula **y guarda** la media y la desviación de
> ese dataset. Cuando es falso, **reutiliza** las que ya guardó. Con el conjunto de entrenamiento
> lo llamamos en verdadero, y con el de prueba en falso. Si escaláramos el test con sus propias
> estadísticas, train y test quedarían en espacios distintos, los pesos aprendidos ya no
> corresponderían, y estaríamos cometiendo *data leakage*. Es un error metodológico clásico y
> nos pasó durante el desarrollo.

---

## Bloque 3 — Ejecución en vivo, punto 3.1 (4:10 – 6:00) · **Miguel**
*En pantalla: terminal. Ejecutar `make` y luego `./regresion`.*

**Miguel:**
> Compilamos con `make`, que genera dos ejecutables: `regresion` para el punto 3.1 y `memoria`
> para el 3.2. Compila sin advertencias con `-Wall -Wextra`.
>
> *(ejecuta `./regresion`)*
>
> El primer experimento es **regresión simple** con el dataset de venta de helados: la
> temperatura prediciendo las ventas. Cuarenta y ocho muestras.
>
> Y aquí pasó lo más interesante de toda la práctica. Miren el R cuadrado: **cero punto cero
> uno cinco**. Prácticamente cero. El modelo no explica nada.
>
> Lo primero que pensamos fue que teníamos un error en el código. Pero fíjense abajo: la
> verificación por ecuaciones normales da **exactamente los mismos pesos**. Dos métodos
> matemáticos independientes llegando al mismo resultado. Entonces el código está bien.
>
> El problema es otro: **este dataset no tiene una relación lineal**. Si uno mira los datos
> ordenados por temperatura, las ventas son altas en los dos extremos —unas cuarenta y dos
> unidades a menos cuatro grados, y cuarenta y uno a casi cinco grados— y caen casi a cero
> alrededor de los cero grados. La nube de puntos tiene **forma de U**, es una parábola.
>
> Y la mejor recta posible sobre una parábola es una recta casi horizontal. Eso es literalmente
> lo que encontró el modelo: pendiente de menos cero punto setenta y siete, casi plana, y un
> bias de dieciséis punto cuatro, que es prácticamente la media de las ventas.
>
> O sea: un R cuadrado de cero no significa que el código esté mal. Significa que el modelo
> elegido no corresponde a estos datos. Distinguir esas dos cosas es la parte difícil.

---

## Bloque 3b — La extensión polinómica (6:00 – 6:40) · **Miguel**
*En pantalla: la sección 1b de la salida y la función `ejecutarPolinomica` en `main.cpp`.*

**Miguel:**
> Entonces lo resolvimos, y aquí es donde el diseño orientado a objetos se paga solo.
>
> La regresión lineal lo es respecto a los **parámetros**, no respecto a las variables. Si yo
> agrego a la matriz X una columna con t al cuadrado, el modelo sigue siendo lineal en los pesos
> y se resuelve con el mismo código.
>
> *(mostrando la función)* Fíjense que **no modificamos ni una línea de la clase**. Solo
> construimos la matriz ampliada con t y t al cuadrado, y se la pasamos a `fit`. Funciona porque
> la clase se diseñó desde el principio para n características, no para una.
>
> El resultado: el R cuadrado pasa de **cero punto cero uno cinco a cero punto noventa**. El
> coeficiente dominante es el de t al cuadrado, que confirma la forma parabólica.
>
> Si hubiéramos escrito una clase específica para regresión simple, esto habría exigido reescribir
> todo. Esa es la diferencia entre un script y un framework.

---

## Bloque 4 — Regresión múltiple y divergencia (6:40 – 7:40) · **Julio**
*En pantalla: secciones 2 y 3 de la salida de `./regresion`.*

**Julio:**
> El segundo experimento es **regresión múltiple** con las notas de estudiantes: doscientas
> muestras y cuatro características —horas de estudio, horas de sueño, porcentaje de asistencia
> y notas previas— prediciendo la nota del examen. Aquí sí aplicamos escalamiento.
>
> El R cuadrado en prueba es **cero punto ochenta y cuatro**, con un error típico de dos punto
> siete puntos de nota. Y un detalle importante: el R cuadrado de entrenamiento y el de prueba
> son casi idénticos, cero punto ochenta y tres contra cero punto ochenta y cuatro. Eso significa
> que **no hay sobreajuste**: el modelo generaliza igual de bien a datos que nunca vio.
>
> Ahora, los pesos: cuatro punto ocho nueve, uno punto cincuenta, uno punto cincuenta y tres, y
> dos punto noventa y dos. Como los datos están estandarizados, estos pesos son **directamente
> comparables entre sí**. Se leen como cuánto sube la nota por cada desviación estándar de
> aumento en esa variable.
>
> Las **horas de estudio** son por lejos la variable más influyente. Le siguen las notas previas,
> con alrededor del sesenta por ciento del peso. Y la asistencia y las horas de sueño quedan
> prácticamente empatadas al final. Esa comparación no se podría hacer sin escalar, porque cada
> variable estaría en unidades distintas.
>
> Y la última sección muestra el problema más instructivo que tuvimos. Cuando entrenamos este
> mismo dataset **sin escalar** y con tasa de aprendizaje de cero punto cero cinco, el modelo
> **diverge**: el costo se va a infinito. La razón es que las escalas son muy distintas: las
> horas de estudio van de uno a doce, pero la asistencia va de cincuenta a cien. Un paso
> razonable para una variable es gigantesco para la otra.
>
> Con estandarización, el mismo problema converge en **ciento sesenta y dos iteraciones**. Y en
> vez de devolver `NaN` silenciosamente, el modelo detecta que el costo dejó de ser finito y
> lanza una excepción que dice qué hacer. Un error que se detecta y se explica vale mucho más que
> un resultado que parece correcto y no lo es.

---

## Bloque 5 — Análisis de memoria, punto 3.2 (7:40 – 9:00) · **Miguel**
*En pantalla: ejecutar `./memoria`.*

**Miguel:**
> Pasamos al punto 3.2. Este programa crea objetos `LinearRegression` en las tres zonas de
> memoria e imprime sus direcciones reales.
>
> *(señalando la salida)* Arriba, los objetos **globales y estáticos**: direcciones que empiezan
> en cero equis cincuenta y cinco. Viven en el segmento de datos, se construyen antes de `main`
> y duran todo el programa.
>
> En el medio, el **heap**: los objetos creados con `new`. Direcciones intermedias. Noten que el
> puntero vive en el stack y ocupa ocho bytes, mientras que el objeto al que apunta vive en el
> heap. Son dos cosas distintas.
>
> Y abajo, el **stack**: direcciones que empiezan en cero equis siete efe efe, mucho más altas.
> Las variables locales. Fíjense que el segundo objeto local está por encima del primero: la pila
> crece hacia direcciones menores.
>
> El orden es siempre: datos globales, heap, stack. Con ASLR las direcciones cambian en cada
> ejecución, pero ese orden relativo se mantiene.
>
> **Ahora el tamaño.** `sizeof(LinearRegression)` da ciento treinta y seis bytes. ¿Por qué ese
> valor y no otro? Sumando los miembros uno por uno dan ciento veintinueve. Faltan siete.
>
> Tres razones. Primera: cada `std::vector` ocupa **veinticuatro bytes**, sin importar cuántos
> elementos tenga. Son tres punteros de ocho bytes: inicio de los datos, fin de los datos y fin
> de la capacidad. Los elementos no están dentro del objeto, viven aparte en el heap.
>
> Segunda: **alineación**. El tipo de mayor alineación es de ocho bytes, así que el tamaño total
> tiene que ser múltiplo de ocho. El `bool` de un byte arrastra siete bytes de relleno. Ciento
> veintinueve se redondea a ciento treinta y seis.
>
> Tercera, y la más importante: **`sizeof` es estático**. Se resuelve en tiempo de compilación.
> Mide el esqueleto del objeto, nunca lo que ese objeto reserve después.
>
> Lo comprobamos experimentalmente: antes de entrenar, el objeto consume ciento treinta y seis
> bytes. Después de entrenar con dos características consume ciento cincuenta y dos bytes
> reales. Pero `sizeof` sigue devolviendo ciento treinta y seis. Esa diferencia de dieciséis
> bytes son los dos pesos que `weights` reservó en el heap.

---

## Bloque 6 — Vector dinámico de objetos (9:00 – 9:30) · **Julio**
*En pantalla: sección 4 de la salida de `./memoria`.*

**Julio:**
> Aquí creamos el arreglo de cinco modelos en memoria dinámica, de las dos formas.
>
> Con `new LinearRegression[5]`: `sizeof` del arreglo da **ocho bytes**, y eso confunde al
> principio. Da ocho porque `arreglo` es un **puntero**, no el bloque. `sizeof` no tiene cómo
> saber cuántos elementos se reservaron.
>
> El tamaño real es cinco por ciento treinta y seis, igual a **seiscientos ochenta bytes**. Y lo
> verificamos: la distancia entre elementos consecutivos es exactamente ciento treinta y seis
> bytes, lo que confirma que están contiguos en memoria. El asignador además guarda unos bytes
> ocultos con el número de elementos, para que `delete[]` sepa cuántos destructores llamar.
>
> Con `std::vector` pasa lo mismo pero gestionado solo: el objeto vector está en el stack y ocupa
> veinticuatro bytes, su buffer está en el heap y ocupa los mismos seiscientos ochenta. Dos
> direcciones completamente distintas.
>
> Al final liberamos todo: cada `new` con su `delete`, cada `new[]` con su `delete[]` —mezclarlos
> es comportamiento indefinido— y dejamos los punteros en `nullptr` para evitar punteros
> colgantes y dobles liberaciones.

---

## Bloque 7 — Fallos históricos y cierre (9:30 – 9:55) · **Miguel y Julio**
*En pantalla: la sección 6 del informe.*

**Julio:**
> Investigamos tres fallos catastróficos por manejo de memoria dinámica.
>
> **Heartbleed**, OpenSSL dos mil catorce: el servidor reservaba un búfer del tamaño que
> declaraba el cliente y copiaba sin verificar que coincidiera con el real. Permitía leer sesenta
> y cuatro kilobytes de memoria ajena por petición, incluidas claves privadas, sin dejar rastro.
> Afectó a dos tercios de la web y costó unos quinientos millones de dólares. El parche fue
> literalmente añadir una comprobación de longitud.
>
> **Cloudbleed**, Cloudflare dos mil diecisiete: un analizador de HTML se pasaba del final de un
> búfer y devolvía memoria no inicializada dentro de las páginas web. Cookies y tokens de
> autenticación filtrados durante cinco meses, y además cacheados por los buscadores.

**Miguel:**
> Y el tercero es el que más nos gustó, porque es el caso más puro de memoria dinámica: el rover
> **Spirit** de la NASA, dos mil cuatro. En el sol diecisiete dejó de comunicarse y entró en un
> ciclo de reinicio infinito. No era hardware: el sistema de archivos flash tenía demasiados
> archivos acumulados del viaje, y al arrancar el software **no lograba reservar suficiente RAM**
> para la estructura de directorios, terminaba, y volvía a reiniciar. Una misión de ochocientos
> veinte millones de dólares casi perdida por una reserva de memoria que falla y no se maneja.
>
> Los tres comparten la misma raíz: **el programa confía en un tamaño que no controla**.
> Heartbleed confía en la longitud que declara el cliente; Cloudbleed confía en que el puntero no
> se pasará del búfer; Spirit confía en que la reserva de memoria siempre va a funcionar. En C y
> en C++ nada de eso se verifica solo. Hay que escribir la comprobación.

**Julio:**
> Esa fue la práctica. Gracias.

---

## Notas de grabación

- **Reparto:** Miguel lleva diseño, ejecución y memoria (≈ 5 min); Julio lleva métodos,
  divergencia, vector dinámico y fallos (≈ 4:30). Queda balanceado.
- Todas las cifras del guion corresponden a los datasets oficiales de Interactiva, ya verificadas
  contra la salida real del programa.
- No leer de corrido: tener el guion como apoyo, pero mirar a cámara al menos en la presentación
  y en el cierre.
- Grabar la terminal en vivo, no pegar capturas: el enunciado pide evidencia de la ejecución.
- Ambos rostros visibles durante todo el video — es requisito explícito de la rúbrica.
- Subir a YouTube como **no listado**.
- El guion va justo contra el límite de diez minutos. Si se pasan, recorten el bloque 1
  (arquitectura) a cuarenta segundos: no se pierde nada evaluable. **No recorten** el bloque 3
  (el hallazgo del dataset no lineal), el 3b ni el 5 (el porqué de los 136 bytes): esos tres son
  los que demuestran comprensión y no solo ejecución.
- El profesor puede preguntarle a cualquiera de los dos sobre cualquier parte, así que conviene
  que cada uno entienda también el bloque del otro.

# Práctica No. 2 — Programación Orientada a Objetos
### Programming Languages and Paradigms — Escuela de Ciencias Aplicadas e Ingeniería, EAFIT

**Integrantes**

| Nombre completo | ID | Correo institucional |
|---|---|---|
| Miguel Ángel Pérez Marín | 1018242716 | maperezm4@eafit.edu.co |
| Julio Cesar Llanes Cuello | 1118835249 | jcllanesc@eafit.edu.co |

**Repositorio:** (completar enlace)
**Video YouTube (no listado):** (completar enlace)
**Lenguaje:** C++17 · **Librerías externas:** ninguna (solo STL)

---

## 1. Estructura del proyecto

```
.
├── MatrixOps.h / MatrixOps.cpp          Métodos auxiliares: álgebra matricial y estadística
├── LinearRegression.h / .cpp            Clase del modelo (fit, predict, score, data_scaling)
├── CSVReader.h / CSVReader.cpp          Carga de datasets y división train/test
├── main.cpp                             Punto 3.1 — Reto POO (incluye extensión polinómica)
├── memory_analysis.cpp                  Punto 3.2 — Análisis de memoria
├── Makefile                             Compilación de los dos ejecutables
├── data/                                ice_cream_selling_data.csv, student_exam_scores.csv
└── evidencias/                          Salidas de consola de las pruebas
```

**Compilación y ejecución**

```bash
make            # genera ./regresion y ./memoria
./regresion     # punto 3.1
./memoria       # punto 3.2
```

> Los CSV se descargan de EAFIT Interactiva y se colocan en `data/`. El programa también acepta
> rutas por argumentos: `./regresion ruta_simple.csv ruta_multiple.csv`.

---

## 2. Diseño orientado a objetos

La clase `LinearRegression` encapsula el modelo

$$\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

La **misma clase** resuelve regresión simple y múltiple: internamente siempre trabaja con una
matriz `X` de dimensión *m × n* (m muestras, n características); la regresión simple es el caso
particular *n = 1*. No se necesitan dos clases.

**Principios de POO aplicados**

- **Encapsulamiento:** `weights_` y `bias_` son privados; el exterior solo los consulta por
  `getWeights()` y `getBias()`. Es imposible corromper el estado del modelo desde fuera.
- **Abstracción:** el usuario llama `fit`, `predict`, `score`; no necesita saber si por dentro se
  usa descenso de gradiente o ecuaciones normales.
- **Sobrecarga de métodos:** `predict()` existe en dos versiones (una muestra / un conjunto) y
  `MatrixOps::multiply()` en tres (matriz×matriz, matriz×vector, impresión).
- **Regla de los tres:** constructor de copia, `operator=` y destructor explícitos, para que copiar
  un modelo duplique correctamente sus vectores.
- **Separación de responsabilidades:** el modelo no sabe leer archivos (eso es `CSVReader`) ni
  cómo se invierte una matriz (eso es `MatrixOps`).

### Atributos de `LinearRegression`

| Atributo | Tipo | Descripción |
|---|---|---|
| `weights_` | `std::vectordouble>` | **Arreglo con los pesos del modelo** (exigido por el enunciado) |
| `bias_` | `double` | **Término independiente** (exigido por el enunciado) |
| `learningRate_` | `double` | Tasa de aprendizaje del descenso de gradiente |
| `epochs_` | `std::size_t` | Número máximo de iteraciones |
| `tolerance_` | `double` | Umbral de convergencia sobre el costo |
| `solver_` | `enum class Solver` | `GRADIENT_DESCENT` o `NORMAL_EQUATION` |
| `scalingType_` | `enum class ScalingType` | `NONE`, `STANDARDIZATION` o `MIN_MAX` |
| `scaleCenter_` | `std::vectordouble>` | Media (o mínimo) por característica |
| `scaleSpread_` | `std::vectordouble>` | Desviación (o rango) por característica |
| `trained_` | `bool` | Bandera de modelo entrenado |
| `iterationsRun_` | `std::size_t` | Iteraciones realmente ejecutadas |
| `finalCost_` | `double` | Costo (MSE) de la última iteración |

---

## 3. Documentación de todos los métodos desarrollados

### 3.1 Clase `LinearRegression`

#### `LinearRegression(double learningRate, size_t epochs, Solver solver, double tolerance)`
Constructor. Crea un modelo **sin entrenar** con los hiperparámetros indicados.
Valores por defecto: `learningRate = 0.01`, `epochs = 10000`,
`solver = GRADIENT_DESCENT`, `tolerance = 1e-12`.
Es `explicit` para impedir conversiones implícitas desde `double`.

#### `LinearRegression(const LinearRegression&)` y `operator=`
Constructor de copia y asignación. Copian todos los atributos, incluidos los `std::vector`
(copia profunda: cada objeto queda con su propio bloque en el heap). El `operator=` verifica
autoasignación (`this != &other`).

#### `~LinearRegression()`
Destructor. No libera nada manualmente porque la clase no usa punteros crudos: los
`std::vector` liberan su memoria por RAII. Se declara explícitamente por claridad y para
respetar la regla de los tres.

#### `void fit(const Matrix& X, const Vector& y)`
**Entrena el modelo.** Valida dimensiones, inicializa `weights_` en cero y `bias_` en cero, y
delega en `fitGradientDescent()` o `fitNormalEquation()` según el solver. Al final marca
`trained_ = true` y guarda el costo final.
- **Entrada:** `X` (m×n), `y` (m).
- **Salida:** ninguna; modifica el estado interno.
- **Excepciones:** `std::invalid_argument` si `X` está vacía, si `X.size() != y.size()` o si las
  filas tienen distinto número de columnas.

#### `void fitGradientDescent(const Matrix& X, const Vector& y)` *(privado)*
Minimiza el MSE iterativamente. En cada época calcula el gradiente

$$\frac{\partial J}{\partial w_j} = \frac{2}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)\,x_{ij}, \qquad
\frac{\partial J}{\partial b} = \frac{2}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

y actualiza `w ← w − α·∇w`, `b ← b − α·∇b`. Se detiene cuando se agotan las épocas o cuando la
variación del costo cae por debajo de `tolerance_`. Si el costo deja de ser finito (divergencia),
lanza `std::runtime_error` con un mensaje accionable.
**Complejidad:** O(épocas · m · n).

#### `void fitNormalEquation(const Matrix& X, const Vector& y)` *(privado)*
Solución cerrada. Agrega a `X` una columna de unos (para absorber el bias dentro del vector de
parámetros), construye el sistema $(X^TX)\theta = X^Ty$ y lo resuelve con `MatrixOps::solve`.
No usa inversa explícita: resolver el sistema es más estable numéricamente que invertir.
**Complejidad:** O(m·n² + n³). Sirve además como **verificación independiente** del gradiente.

#### `double predict(const Vector& x) const`
Predice la salida de **una** muestra: producto punto de `x` con `weights_` más `bias_`.
Lanza `std::runtime_error` si el modelo no está entrenado y `std::invalid_argument` si el número
de características no coincide.

#### `Vector predict(const Matrix& X) const`
Sobrecarga que predice un **conjunto** de muestras. Reserva memoria con `reserve()` para evitar
reasignaciones del vector de salida y reutiliza la versión de una muestra.

#### `Metrics score(const Matrix& X, const Vector& y) const`
**Calcula los errores de predicción.** Devuelve una estructura `Metrics` con cuatro indicadores:

| Métrica | Fórmula | Interpretación |
|---|---|---|
| MSE | $\frac{1}{m}\sum(\hat{y}_i-y_i)^2$ | Error cuadrático medio; penaliza errores grandes |
| RMSE | $\sqrt{\text{MSE}}$ | Error típico en las unidades de `y` |
| MAE | $\frac{1}{m}\sum|\hat{y}_i-y_i|$ | Error absoluto medio; robusto a atípicos |
| R² | $1-\frac{SSE}{SST}$ | Proporción de varianza explicada (1 = ajuste perfecto) |

#### `double computeCost(const Matrix& X, const Vector& y) const` *(privado)*
Función de costo MSE usada internamente como criterio de parada del gradiente.

#### `Matrix data_scaling(const Matrix& X, ScalingType type, bool fitParameters)`
**Escala los valores del dataset**, columna por columna, y devuelve una matriz nueva
(no modifica la original).

- `STANDARDIZATION`: $z = (x - \mu)/\sigma$
- `MIN\_MAX`: $z = (x - \min)/(\max - \min)$

El parámetro `fitParameters` es la parte importante del diseño:
- `true` → calcula y **guarda** media/desviación (o mín/máx) del dataset. Se usa con *train*.
- `false` → **reutiliza** los parámetros guardados. Se usa con *test*.

Escalar el test con sus propias estadísticas sería un error metodológico (*data leakage*) y
rompería la correspondencia con los pesos aprendidos. Si una columna es constante, el divisor se
fuerza a 1 para evitar división por cero.

#### Getters y utilidades
| Método | Devuelve |
|---|---|
| `getWeights()` | Referencia constante al arreglo de pesos |
| `getBias()` | Término independiente |
| `isTrained()` | Si el modelo ya fue entrenado |
| `getIterationsRun()` | Iteraciones realmente ejecutadas |
| `getFinalCost()` | Costo final alcanzado |
| `toString()` | La ecuación del modelo como texto |
| `summary()` | Imprime el estado interno completo del objeto |

#### `static void validate(const Matrix& X, const Vector& y)` *(privado)*
Valida consistencia de dimensiones. Es `static` porque no depende del estado del objeto.

#### `Metrics::toString()`
Formatea las cuatro métricas en una sola línea legible.

### 3.2 Namespace `MatrixOps` — procesamiento matricial

| Método | Descripción | Complejidad |
|---|---|---|
| `transpose(A)` | Transpuesta de A (m×n → n×m) | O(m·n) |
| `multiply(A,B)` | Producto matricial clásico de tres bucles | O(m·n·p) |
| `multiply(A,x)` | Producto matriz–vector | O(m·n) |
| `dot(a,b)` | Producto punto; lanza excepción si los tamaños difieren | O(n) |
| `solve(A,b)` | Resuelve A·x = b por **Gauss-Jordan con pivoteo parcial**. Lanza `runtime_error` si el pivote es menor que 1e-12 (matriz singular) | O(n³) |
| `inverse(A)` | Inversa por Gauss-Jordan sobre la matriz aumentada [A\|I] | O(n³) |
| `column(A,j)` | Extrae la columna j (lo usa `data_scaling`) | O(m) |
| `print(A,label)` / `print(v,label)` | Impresión formateada para evidencias; trunca salidas largas | O(m·n) |

**Pivoteo parcial:** antes de eliminar cada columna se intercambia la fila actual por la que tenga
el mayor valor absoluto en esa columna. Esto evita dividir por números muy pequeños y la
acumulación de error de redondeo en punto flotante.

### 3.3 Namespace `MatrixOps` — cálculo estadístico

| Método | Fórmula |
|---|---|
| `mean(v)` | $\mu = \frac{1}{n}\sum x_i$ |
| `variance(v)` | $\sigma^2 = \frac{1}{n}\sum (x_i-\mu)^2$ (poblacional) |
| `stddev(v)` | $\sigma = \sqrt{\sigma^2}$ |
| `minimum(v)` / `maximum(v)` | Recorrido lineal |

### 3.4 Namespace `CSVReader`

| Método | Descripción |
|---|---|
| `load(path, hasHeader, delimiter)` | Lee un CSV con `std::ifstream`; la **última columna** es la variable objetivo y las anteriores son las características. Guarda los encabezados, descarta líneas vacías o incompletas y lanza `runtime_error` si el archivo no abre o no tiene datos válidos |
| `trainTestSplit(data, ratio, ...)` | Divide el dataset en entrenamiento y prueba por **muestreo sistemático**: toma una de cada cinco muestras para prueba (80/20). Es determinístico, así que los resultados son reproducibles, y a diferencia de un corte por la mitad funciona aunque el archivo venga **ordenado** por alguna columna |
| `trim(s)` *(interno)* | Elimina espacios, tabulaciones, retornos de carro y comillas |
| `split(line, delim)` *(interno)* | Separa una línea en campos |

---

## 4. Evidencia de pruebas (punto 3.1)

Salida completa en `evidencias/salida_regresion.txt`. Datasets oficiales de EAFIT Interactiva.

### 4.1 Regresión lineal simple — `ice_cream_selling_data.csv`

**Entrada:** 48 muestras, 1 característica (temperatura en °C → ventas de helado). División
sistemática 40 entrenamiento / 9 prueba. Sin escalamiento, `learning_rate = 0.001`.

**Estadísticas del dataset**

```
x1 (temperatura): media=0.2718  desv=2.6700  min=-4.6623  max=4.8990
y  (ventas)     : media=15.9053 desv=12.1389
```

**Salida:**

```
weights (n=1): [ -0.7693 ]
bias: 16.437650
iteraciones: 6973

y_hat:  19.190561   y_real:  32.284531   error: -13.093971
y_hat:  18.478002   y_real:  20.279679   error:  -1.801677
y_hat:  17.714923   y_real:  12.615181   error:   5.099742
y_hat:  16.552761   y_real:   4.673643   error:  11.879119
y_hat:  15.907785   y_real:   2.576782   error:  13.331003
...

score (train): MSE = 148.874535 | RMSE = 12.201415 | MAE = 10.480477 | R2 = 0.028741
score (test) : MSE = 116.565289 | RMSE = 10.796541 | MAE =  9.793198 | R2 = 0.014854
```

**Verificación por ecuaciones normales:** `w = -0.7693`, `b = 16.4377`, R² test = 0.014853.
Los dos solvers coinciden hasta la cuarta cifra decimal.

**Interpretación — este es el resultado más informativo de la práctica.** El R² es
prácticamente **cero**: el modelo no explica nada. Pero los dos solvers coinciden, así que el
problema **no** es la implementación: es que **este dataset no tiene una relación lineal**.

Mirando los datos ordenados por temperatura se ve por qué. Las ventas son altas en los extremos
(≈ 42 unidades a −4.66 °C y ≈ 41 a 4.90 °C) y caen casi a cero alrededor de 0 °C. La nube de
puntos tiene forma de **U**, no de recta. La mejor recta posible sobre una parábola es
aproximadamente horizontal, y eso es exactamente lo que encontró el modelo: una pendiente de
−0.77, casi plana, y un bias de 16.44 que es prácticamente la media de `y` (15.91).

Un R² de cero **no significa que el código esté mal**; significa que el modelo elegido no es el
adecuado para estos datos. Distinguir esos dos casos es, en la práctica, la parte difícil.

### 4.1b Extensión polinómica — la misma clase, sin modificarla

La regresión lineal lo es respecto a los **parámetros**, no respecto a las variables. Si a la
matriz X le agregamos una columna con t², el modelo

$$\hat{y} = w_1 t + w_2 t^2 + b$$

sigue siendo lineal en `w` y `b`, y se resuelve con **exactamente el mismo código**. No se
modificó ni una línea de `LinearRegression`: solo se construyó la matriz ampliada `[t, t²]` y se
le pasó a `fit()`. Aquí se ve el beneficio de haber diseñado la clase para n características
desde el principio.

**Salida:**

```
weights [t, t^2] (n=2): [ -2.1420 11.7916 ]      (espacio estandarizado)
bias: 16.238460
iteraciones: 153

t:  -3.578554   y_hat:  28.763782   y_real:  32.284531   error:  -3.520749
t:  -1.660348   y_hat:   9.203025   y_real:  12.615181   error:  -3.412156
t:   0.688781   y_hat:   3.258312   y_real:   2.576782   error:   0.681530
t:   2.318591   y_hat:  10.784900   y_real:   7.412094   error:   3.372806
t:   3.610778   y_hat:  23.535975   y_real:  28.912188   error:  -5.376213

score (train): MSE =  9.835456 | RMSE = 3.136153 | MAE = 2.640746 | R2 = 0.935833
score (test) : MSE = 11.333225 | RMSE = 3.366486 | MAE = 3.104097 | R2 = 0.904218

Modelo en unidades originales: y_hat = -0.785058*t + 1.796630*t^2 + 2.946690
```

**El R² pasa de 0.015 a 0.904** solo por agregar una característica derivada. El coeficiente
dominante es el de t² (1.7966), que confirma la forma cuadrática, y el término lineal (−0.785)
desplaza ligeramente el vértice de la parábola.

| Modelo | R² (train) | R² (test) | RMSE (test) |
|---|---|---|---|
| Lineal: `y = w·t + b` | 0.0287 | 0.0149 | 10.797 |
| Cuadrático: `y = w₁·t + w₂·t² + b` | 0.9358 | 0.9042 | 3.366 |

### 4.2 Regresión lineal múltiple — `student_exam_scores.csv`

**Entrada:** 200 muestras, 4 características (horas de estudio, horas de sueño, porcentaje de
asistencia, notas previas → nota del examen). División 160 entrenamiento / 40 prueba. Con
estandarización, `learning_rate = 0.05`.

**Estadísticas del dataset**

```
x1 (horas de estudio): media= 6.3255  desv= 3.2192  min= 1.00  max= 12.00
x2 (horas de sueño)  : media= 6.6220  desv= 1.4934  min= 4.00  max=  9.00
x3 (% asistencia)    : media=74.8300  desv=14.2142  min=50.30  max=100.00
x4 (notas previas)   : media=66.8000  desv=15.6247  min=40.00  max= 95.00
y  (nota del examen) : media=33.9550  desv= 6.7726
```

**Salida:**

```
weights (n=4): [ 4.8858 1.4974 1.5269 2.9237 ]
bias: 34.387499
iteraciones: 162

y_hat:  40.493732   y_real:  40.300000   error:   0.193732
y_hat:  28.417783   y_real:  24.700000   error:   3.717783
y_hat:  41.692547   y_real:  41.100000   error:   0.592547
y_hat:  22.285016   y_real:  17.100000   error:   5.185016
y_hat:  41.499135   y_real:  39.200000   error:   2.299135
...

score (train): MSE = 7.239961 | RMSE = 2.690718 | MAE = 2.288725 | R2 = 0.837611
score (test) : MSE = 7.578094 | RMSE = 2.752834 | MAE = 2.255675 | R2 = 0.839651
```

**Verificación por ecuaciones normales:** pesos y bias idénticos hasta la cuarta cifra decimal;
R² test = 0.839650.

**Interpretación.** El modelo explica el **84 %** de la varianza, con un error típico de **2.75
puntos** de nota. Como los datos están estandarizados, los pesos son **directamente comparables
entre sí** y se leen como "cuánto sube la nota por cada desviación estándar de aumento en esa
variable":

| Variable | Peso | Lectura |
|---|---|---|
| Horas de estudio | **4.886** | La más influyente con diferencia |
| Notas previas | 2.924 | Segunda; pesa ≈ 60 % de lo que pesan las horas de estudio |
| % de asistencia | 1.527 | Influencia moderada |
| Horas de sueño | 1.497 | La menor, prácticamente empatada con la asistencia |

El bias (34.39) es la nota esperada para un estudiante que está en el promedio de las cuatro
variables, y coincide con la media de `y` (33.96) como debe ser cuando los datos están centrados.

Un detalle importante: el R² de entrenamiento (0.8376) y el de prueba (0.8397) son casi
idénticos. Eso indica que **no hay sobreajuste**: el modelo generaliza igual de bien a datos que
nunca vio.

> Nota: los pesos de arriba están en el espacio **escalado**. Para volver a las unidades
> originales: $w_j^{orig} = w_j/\sigma_j$ y $b^{orig} = b - \sum_j w_j\mu_j/\sigma_j$.

### 4.3 Prueba de robustez

```
Excepcion capturada correctamente -> fit: el costo divergio.
Reduzca learningRate o escale los datos.
```

Entrenando el dataset múltiple **sin escalar** con `learning_rate = 0.05`, el modelo detecta la
divergencia y lanza una excepción con un mensaje accionable, en vez de devolver `NaN`
silenciosamente. Con estandarización, el mismo problema converge en **162 iteraciones**.

---

## 5. Análisis de memoria (punto 3.2)

Salida completa en `evidencias/salida_memoria.txt`. Plataforma: Linux x86-64, g++ 17, punteros de
64 bits.

### 5.1 Direcciones de los objetos

| Objeto | Zona | Dirección (ejemplo de ejecución) | `sizeof` |
|---|---|---|---|
| `modeloGlobal` | Global (.data/.bss) | `0x559a25b27160` | 136 B |
| `modeloEstatico` | Estática | `0x559a25b27200` | 136 B |
| `modeloLocal1` | Stack | `0x7ffcc4bd9250` | 136 B |
| `modeloLocal2` | Stack | `0x7ffcc4bd92e0` | 136 B |
| `*modeloDinamico1` | Heap | `0x559a2df0c310` | 136 B |
| `*modeloDinamico2` | Heap | `0x559a2df0c3a0` | 136 B |

**Lectura del mapa de memoria:**

```
0x559a25b27160  ← datos globales   (direcciones bajas, fijas al enlazar)
0x559a2df0c310  ← heap             (crece hacia arriba)
0x7ffcc4bd9250  ← stack            (direcciones altas, crece hacia abajo)
```

- Las variables **globales/estáticas** quedan en el segmento de datos; se construyen antes de
  `main()` y viven hasta que termina el programa. Su dirección la fija el enlazador.
- Las variables **locales** viven en el marco de pila de la función. `modeloLocal2` está 144 bytes
  por encima de `modeloLocal1`: la pila crece hacia direcciones menores, y el compilador añade
  alineación entre marcos.
- Los objetos **dinámicos** están en el heap, en direcciones intermedias. El *puntero*
  `modeloDinamico1` (8 bytes) vive en el stack; el *objeto* al que apunta vive en el heap.
- Con **ASLR** activo las direcciones concretas cambian en cada ejecución, pero el orden relativo
  de las regiones (datos < heap < stack) se mantiene.

### 5.2 ¿Por qué `sizeof(LinearRegression)` = 136 bytes?

| Miembro | Tipo | Bytes |
|---|---|---|
| `weights_` | `std::vectordouble>` | 24 |
| `bias_` | `double` | 8 |
| `learningRate_` | `double` | 8 |
| `epochs_` | `size_t` | 8 |
| `tolerance_` | `double` | 8 |
| `solver_` | `enum class` | 4 |
| `scalingType_` | `enum class` | 4 |
| `scaleCenter_` | `std::vectordouble>` | 24 |
| `scaleSpread_` | `std::vectordouble>` | 24 |
| `trained_` | `bool` | 1 |
| `iterationsRun_` | `size_t` | 8 |
| `finalCost_` | `double` | 8 |
| **Suma de los miembros** | | **129** |
| **`sizeof` real** | | **136** |

Tres razones explican el valor:

1. **Cada `std::vector` ocupa 24 bytes**, no importa cuántos elementos tenga. Son tres punteros de
   8 bytes: inicio de los datos, fin de los datos y fin de la capacidad. Los elementos **no** están
   dentro del objeto: viven en un bloque aparte del heap.
2. **Alineación (padding).** El tipo de mayor alineación es de 8 bytes (`double`, `size_t`,
   punteros), así que el tamaño total debe ser múltiplo de 8. Los 129 bytes útiles se redondean a
   136; el `bool` de 1 byte arrastra 7 bytes de relleno.
3. **`sizeof` es estático.** Se resuelve en tiempo de compilación y mide solo el "esqueleto" del
   objeto, nunca la memoria que ese objeto reserve después.

**Comprobación experimental:** antes de `fit()` el objeto ocupaba 136 bytes; después de entrenar
con 2 características ocupa 136 + 2·8 = **152 bytes reales**, pero `sizeof` sigue devolviendo 136.
Esa diferencia es exactamente el bloque del heap que `weights_` reservó.

### 5.3 Vector de `LinearRegression` en memoria dinámica

**(a) Arreglo clásico `new LinearRegression[5]`**

```
arreglo[0] -> 0x559a2df0c478
arreglo[1] -> 0x559a2df0c500      ← 136 bytes de distancia
arreglo[2] -> 0x559a2df0c588
arreglo[3] -> 0x559a2df0c610
arreglo[4] -> 0x559a2df0c698
sizeof(arreglo)        : 8 bytes
Tamaño real del bloque : 5 × 136 = 680 bytes
```

- `sizeof(arreglo)` da **8** porque `arreglo` es un **puntero**, no el bloque. El operador `sizeof`
  no tiene forma de saber cuántos elementos se reservaron.
- El tamaño real es **N × sizeof(LinearRegression) = 680 bytes**. La distancia entre elementos
  consecutivos (136 bytes) confirma que están **contiguos** en memoria.
- El asignador reserva además unos bytes ocultos de metadatos, y `new[]` guarda el número de
  elementos para que `delete[]` sepa cuántos destructores llamar.

**(b) `std::vectorLinearRegression> vec(5)`**

```
sizeof(vec)        : 24 bytes      ← el objeto vector (3 punteros)
vec.capacity()     : 5
Memoria del buffer : 680 bytes en el heap
Dirección de vec      (stack): 0x7ffcc4bd91f0
Dirección de vec.data() (heap): 0x559a2df0c730
```

Dos direcciones distintas: el **objeto vector** está en el stack (24 bytes), su **buffer** en el
heap (680 bytes). Es el mismo almacenamiento que `new[]`, pero liberado automáticamente por RAII.

### 5.4 Liberación

```cpp
delete  modeloDinamico1;   // new   → delete
delete[] arreglo;          // new[] → delete[]
modeloDinamico1 = nullptr; // evita punteros colgantes
```

Reglas aplicadas: cada `new` con su `delete`, cada `new[]` con su `delete[]` (mezclarlos es
comportamiento indefinido), y los punteros se dejan en `nullptr` para evitar *dangling pointers* y
dobles liberaciones. El programa no tiene fugas: toda la memoria dinámica se libera antes de
`return 0`.

---

## 6. Fallos históricos catastróficos por manejo de memoria dinámica

### 6.1 Heartbleed — OpenSSL (2014) · CVE-2014-0160

**Personas y empresas implicadas**
- Robin Seggelmann, desarrollador que implementó la extensión Heartbeat en OpenSSL (2011–2012),
  revisada e integrada por el equipo del proyecto OpenSSL.
- Descubierto de forma independiente por investigadores de **Google Security** (Neel Mehta) y la
  empresa finlandesa **Codenomicon**.
- Afectados: prácticamente toda la web. Apache, que usa OpenSSL para HTTPS, era empleado por el 66 % de los sitios web según netcraft.com, y un estudio de Netcraft determinó que el 17,5 % de los sitios SSL podían ser vulnerables.

**Qué pasó y cuáles fueron las consecuencias**
El servidor reservaba con `malloc()` un búfer del tamaño que **declaraba el cliente** y luego
copiaba allí la carga útil con `memcpy()`, sin verificar que la longitud declarada coincidiera con
la real. Al enviar peticiones Heartbeat con un campo de longitud mayor que la carga real, el atacante recibía bytes adicionales pertenecientes a la memoria del proceso remoto. Es un *buffer over-read* clasificado como CWE-126.

Con cada petición se podían leer hasta 64 KB de memoria en crudo, que podían contener credenciales sin cifrar, cookies de sesión, números de tarjeta y, críticamente, las claves privadas de los certificados. El ataque no dejaba rastro en los registros. El fallo estuvo presente desde OpenSSL 1.0.1 (marzo de 2012) y el coste global de remediarlo se estimó inicialmente en 500 millones de dólares.

**Cómo pudo evitarse / corregirse**
- Validando la longitud del mensaje e ignorando las peticiones Heartbeat que pidieran más datos de los que su carga útil contenía; una revisión de seguridad del código también habría detectado el fallo. El RFC 6520 ya exigía descartar silenciosamente esos mensajes.
- El parche llegó en la versión 1.0.1g del 7 de abril de 2014 y consistió simplemente en añadir una comprobación de valores.
- A largo plazo: usar lenguajes con seguridad de memoria, inicializar la memoria reservada
  (`calloc` en vez de `malloc`), compilar con *sanitizers* y auditar las librerías críticas.

**Fuentes**
- OWASP — Heartbleed Bug: https://owasp.org/www-community/vulnerabilities/Heartbleed_Bug
- Invicti — The Heartbleed Bug: https://www.invicti.com/blog/web-security/the-heartbleed-bug
- Black Duck — Heartbleed Bug: https://www.blackduck.com/blog/heartbleed-bug.html

---

### 6.2 Cloudbleed — Cloudflare (2017)

**Personas y empresas implicadas**
- **Cloudflare**, CDN y proxy inverso usado por millones de sitios; su CTO John Graham-Cumming
  publicó el informe del incidente.
- **Tavis Ormandy**, investigador de **Google Project Zero**, que lo descubrió el 17 de febrero de
  2017 y lo bautizó por su parecido con Heartbleed.

**Qué pasó y cuáles fueron las consecuencias**
Cloudflare rastreó el fallo hasta un analizador de HTML presente en tres de sus funciones: Automatic Rewrites, Server-Side Excludes y Email Obfuscation. El error ya existía en el analizador Ragel, pero nunca se activaba por la forma en que se usaban los búferes internos de NGINX; el nuevo analizador cambió ese esquema de búferes y provocó la fuga.

Por un error en la lógica de análisis al procesar HTML mal formado, podía producirse un desbordamiento de búfer que volcaba el contenido de memoria no inicializada dentro de la página web devuelta. Los servidores de borde leían más allá del final de un búfer y devolvían memoria que contenía información privada como cookies HTTP, tokens de autenticación y cuerpos de peticiones POST.

Lo agravante: esas respuestas contaminadas fueron **cacheadas por los buscadores**, de modo que
datos privados quedaron indexados públicamente. Según Cloudflare, la fuga pudo comenzar el 22 de septiembre de 2016 y se prolongó hasta el 18 de febrero de 2017, unos cinco meses. Millones de sitios tuvieron que forzar cambios de contraseña.

**Cómo pudo evitarse / corregirse**
- Comprobar siempre el puntero contra el final del búfer (`p == end` frente a `p >= end`): un
  único operador mal elegido permitió pasarse del límite.
- Project Zero y Cloudflare actuaron con rapidez: el fallo se reportó el 17 de febrero de 2017, en una hora había una mitigación activa, y la notificación pública se dio el 23 de febrero. La mitigación inmediata fue desactivar las tres funciones afectadas.
- Inicializar la memoria antes de escribir en ella, usar *fuzzing* sobre el analizador y pruebas
  con ASan/MSan, y no reutilizar búferes entre peticiones de clientes distintos.

**Fuentes**
- Cloudflare — Incident report on memory leak caused by Cloudflare parser bug:
  https://blog.cloudflare.com/incident-report-on-memory-leak-caused-by-cloudflare-parser-bug/
- Google Project Zero, Issue 1139: https://bugs.chromium.org/p/project-zero/issues/detail?id=1139
- BitSight — Cloudbleed: Breakdown of Cloudflare's Memory Leak:
  https://bitsight.com/blog/cloudbleed-breakdown-of-cloudflares-memory-leak

---

### 6.3 Mars Rover *Spirit* — Anomalía del sol 17 (NASA/JPL, 2004)

**Personas y empresas implicadas**
- **NASA** y el **Jet Propulsion Laboratory (JPL)**; Pete Theisinger (director del proyecto MER),
  Jennifer Trosper (gerente de misión), Glenn Reeves (arquitecto del software de vuelo).
- **Wind River Systems**, proveedor del sistema operativo de tiempo real VxWorks 5.3.1 y de su
  extensión de sistema de archivos en memoria flash.
- Misión de 820 millones de dólares puesta en riesgo.

**Qué pasó y cuáles fueron las consecuencias**
El 21 de enero de 2004 (sol 17) Spirit dejó bruscamente de comunicarse con el control de la misión. Entró en un ciclo de reinicio continuo en el que se reiniciaba una y otra vez en lugar de dormir por la noche, con riesgo real de dañar permanentemente el sistema de alimentación.

La causa no fue un fallo de hardware. El sistema de archivos en flash tenía una capacidad limitada de archivos, limitación que no estaba solo en la flash sino también en la estructura de directorios, y los cálculos del equipo no habían previsto los directorios sobrantes de una carga anterior que seguían en el sistema. Al reiniciar, el software intentaba montar el sistema de archivos flash, el programa no lograba reservar suficiente memoria en RAM para la estructura de directorios, terminaba, y el ciclo se repetía. Es decir: **una reserva de memoria dinámica que falla y no se maneja**, en bucle infinito.

Spirit se bloqueó porque no tenía suficiente RAM para manejar la cantidad de archivos almacenados en la flash, incluidos los datos grabados durante los siete meses de crucero desde la Tierra y los 18 días de operaciones en la superficie. La misión estuvo diez días detenida.

**Cómo pudo evitarse / corregirse**
- El comando INIT_CRIPPLED, que permitía arrancar el computador sin usar el sistema de archivos flash, salvó la misión; Glenn Reeves lo había dejado en el sistema desde los primeros días del proyecto previendo la posibilidad de un sistema de archivos flash roto. Es decir, un **modo de arranque degradado** planeado de antemano.
- Tras identificar el problema los ingenieros borraron archivos y finalmente reformatearon todo el sistema de memoria flash; el 6 de febrero el rover quedó restaurado a su estado normal.
- Prevención: verificar **siempre** el valor de retorno de cada reserva de memoria y degradar con
  elegancia en vez de abortar; dimensionar la RAM para el **peor caso** de la estructura de
  directorios; limpiar los datos de crucero antes del aterrizaje; y no permitir que un fallo
  durante el arranque dispare otro reinicio (detección de ciclos de reinicio).

**Fuentes**
- The Planetary Society — Spirit is "In Recovery":
  https://www.planetary.org/articles/0126_spirit_is_in_recovery
- Spaceflight Now — Reconstructing Spirit's hopeful road to recovery:
  https://spaceflightnow.com/mars/mera/040126spirit.html
- The Register (20 aniversario): https://www.theregister.com/2024/01/05/it_is_20_years_since
- Wikipedia — Timeline of Spirit: https://en.wikipedia.org/wiki/Timeline_of_Spirit

---

### 6.4 Patrón común

Los tres casos comparten la misma raíz: **el programa confía en un tamaño que no controla**.
Heartbleed confía en la longitud declarada por el cliente; Cloudbleed confía en que el puntero no
pasará del final del búfer; Spirit confía en que la reserva de RAM siempre tendrá éxito. En C y
C++ nada de eso se verifica solo: hay que escribir la comprobación.

---

## 7. Problemas encontrados y soluciones

| # | Problema | Causa | Solución aplicada |
|---|---|---|---|
| 1 | El descenso de gradiente divergía (costo → infinito) con el dataset múltiple | Las características estaban en escalas muy distintas (horas: 0–10; asistencia: 50–100). Con una misma tasa de aprendizaje, el paso adecuado para una variable es gigantesco para otra | Se implementó `data_scaling()` con estandarización y se añadió una comprobación `std::isfinite` que lanza una excepción con mensaje accionable en vez de devolver `NaN` |
| 2 | Al escalar el conjunto de prueba con sus propias estadísticas, las métricas empeoraban sin razón aparente | *Data leakage*: train y test quedaban en espacios distintos y los pesos ya no correspondían | Se añadió el parámetro `fitParameters` a `data_scaling()`: `true` aprende los parámetros con el train, `false` los reutiliza con el test |
| 3 | No había forma de saber si los pesos obtenidos eran realmente los óptimos | El descenso de gradiente es aproximado; sin referencia no hay validación | Se implementó un segundo solver por **ecuaciones normales** y se compararon ambos. Coinciden hasta la cuarta cifra decimal |
| 4 | `MatrixOps::inverse()` fallaba o daba resultados imprecisos en algunas matrices | Eliminación gaussiana sin pivoteo: división por pivotes muy pequeños y acumulación de error de redondeo | Se añadió **pivoteo parcial** y un umbral de 1e-12 que lanza `runtime_error` si la matriz es singular. Además se prefirió `solve()` sobre `inverse()` en las ecuaciones normales |
| 5 | Los CSV tenían encabezados, comillas, espacios y líneas vacías que rompían la conversión numérica | El parseo ingenuo con `>>` no limpia los campos | Se escribieron `trim()` y `split()` propias, y `load()` descarta líneas vacías o incompletas |
| 6 | Confusión inicial al interpretar `sizeof` de un arreglo dinámico (devolvía 8) | `sizeof` sobre un puntero mide el puntero, no el bloque apuntado | Se documentó explícitamente en `memory_analysis.cpp` y se calculó el tamaño real como `N * sizeof(LinearRegression)`, verificado con la distancia entre elementos consecutivos |
| 7 | Una columna constante en el dataset producía división por cero al estandarizar | Desviación estándar igual a cero | `data_scaling()` fuerza el divisor a 1.0 cuando el rango o la desviación son menores que 1e-12 |
| 8 | Con el dataset oficial de helados, la división train/test dejaba **todas las temperaturas altas** en el conjunto de prueba | El archivo viene **ordenado por temperatura** y la división original era un corte por la mitad | Se cambió a **muestreo sistemático**: una de cada cinco muestras va a prueba. Determinístico, reproducible e inmune al orden del archivo |
| 9 | El modelo lineal daba R² ≈ 0.015 sobre el dataset de helados, y al principio creímos que la implementación estaba mal | El dataset **no es lineal**: la relación entre temperatura y ventas tiene forma de parábola | Primero se descartó el error de código (los dos solvers coincidían). Luego se agregó la característica derivada t², reutilizando la misma clase sin modificarla: el R² subió a **0.904** |

---

## 8. Conclusiones

1. **La POO convierte un algoritmo en una herramienta reutilizable.** Encapsular los pesos y el
   bias dentro de la clase permitió que la misma clase resuelva regresión simple y múltiple sin
   cambiar una línea: la regresión simple es el caso particular *n = 1*. Si los datos fueran
   variables sueltas en `main()`, cada nuevo dataset exigiría reescribir el código.

2. **El escalamiento de datos no es un adorno, es una condición de convergencia.** El experimento
   fue contundente: con los datos sin escalar y `learning_rate = 0.05` el modelo diverge hacia
   infinito; con estandarización converge en **176 iteraciones**. El mismo problema, resuelto cien
   veces más rápido, solo por cambiar la escala de la entrada.

3. **Tener dos implementaciones del mismo cálculo es la mejor forma de validarlo.** El descenso de
   gradiente y las ecuaciones normales coincidieron hasta la cuarta cifra decimal en los dos
   datasets. Esa coincidencia es la prueba de que la implementación es correcta: dos caminos
   independientes que llegan al mismo punto difícilmente están mal de la misma manera. El gradiente
   escala mejor con muchas características; la solución cerrada es exacta pero cuesta O(n³).

4. **`sizeof` mide el esqueleto del objeto, no lo que el objeto consume.** Un `LinearRegression`
   "pesa" 136 bytes según `sizeof`, pero después de entrenarse con 2 características consume 152
   bytes reales. Los 24 bytes de un `std::vector` son tres punteros, no los datos. Confundir ambas
   cifras es el origen de fallos de dimensionamiento de memoria como el de *Spirit*.

5. **Cada zona de memoria tiene un contrato distinto y hay que elegirla a conciencia.** Las
   variables globales viven todo el programa y se inicializan antes de `main()`; las locales se
   liberan solas al salir del ámbito; las dinámicas viven hasta que alguien escriba `delete`, y ese
   "alguien" es siempre el programador. Verlo en las direcciones reales (datos < heap < stack) hace
   tangible algo que hasta ahora era un diagrama en una diapositiva.

6. **La gestión manual de memoria es la grieta por la que entran los desastres.** Los tres casos
   investigados —Heartbleed, Cloudbleed y *Spirit*— comparten la misma raíz: el programa confía en
   un tamaño que no controló. Una sola comprobación de límites omitida le costó a la industria 500
   millones de dólares, expuso cinco meses de tráfico privado y estuvo a punto de perder una misión
   de 820 millones en Marte. En C++ moderno la defensa existe: RAII, `std::vector` en vez de
   punteros crudos, validar toda entrada externa y nunca mezclar `delete` con `delete[]`.

7. **Un R² bajo no siempre significa código defectuoso: a veces significa modelo equivocado.**
   Con el dataset de helados obtuvimos un R² de 0.015, prácticamente cero. La reacción
   instintiva fue buscar el error en la implementación, pero los dos solvers coincidían: el
   código estaba bien. El problema era que la relación entre temperatura y ventas tiene forma
   de parábola, y la mejor recta sobre una parábola es casi horizontal. Al agregar t² como
   característica derivada —**sin tocar una sola línea de la clase**, porque ya estaba diseñada
   para n características— el R² saltó a 0.904. Distinguir "mi código falla" de "mi modelo no
   corresponde a estos datos" resultó ser la lección más valiosa de la práctica, y solo fue
   posible porque teníamos una segunda implementación con la cual contrastar.

8. **Las excepciones con mensajes accionables ahorran horas de depuración.** El mensaje
   "el costo divergió, reduzca learningRate o escale los datos" convirtió un `NaN` inexplicable en
   un diagnóstico inmediato. Un error que se detecta y se explica vale mucho más que un resultado
   silenciosamente incorrecto.

---

## 9. Checklist de entrega

- [x] Nombres, ID y correos institucionales completados en la tabla inicial
- [ ] Repositorio público creado y enlace añadido
- [x] Datasets oficiales de EAFIT Interactiva colocados en `data/`
- [x] Salidas regeneradas con los datasets oficiales (`make run`)
- [x] Cifras de las secciones 4 y 5 actualizadas con esas salidas
- [ ] Video de máximo 10 minutos en YouTube (no listado), con ambos rostros visibles y
      participación de los dos integrantes
- [ ] Enlaces del repositorio y del video subidos a EAFIT Interactiva

#include "LinearRegression.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
std::string Metrics::toString() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6)
       << "MSE = "  << mse
       << " | RMSE = " << rmse
       << " | MAE = "  << mae
       << " | R2 = "   << r2;
    return os.str();
}

// ---------------------------------------------------------------------------
// Constructores / destructor
// ---------------------------------------------------------------------------
LinearRegression::LinearRegression(double learningRate,
                                   std::size_t epochs,
                                   Solver solver,
                                   double tolerance)
    : weights_(),
      bias_(0.0),
      learningRate_(learningRate),
      epochs_(epochs),
      tolerance_(tolerance),
      solver_(solver),
      scalingType_(ScalingType::NONE),
      scaleCenter_(),
      scaleSpread_(),
      trained_(false),
      iterationsRun_(0),
      finalCost_(0.0) {}

LinearRegression::LinearRegression(const LinearRegression& other)
    : weights_(other.weights_),
      bias_(other.bias_),
      learningRate_(other.learningRate_),
      epochs_(other.epochs_),
      tolerance_(other.tolerance_),
      solver_(other.solver_),
      scalingType_(other.scalingType_),
      scaleCenter_(other.scaleCenter_),
      scaleSpread_(other.scaleSpread_),
      trained_(other.trained_),
      iterationsRun_(other.iterationsRun_),
      finalCost_(other.finalCost_) {}

LinearRegression& LinearRegression::operator=(const LinearRegression& other) {
    if (this != &other) {
        weights_       = other.weights_;
        bias_          = other.bias_;
        learningRate_  = other.learningRate_;
        epochs_        = other.epochs_;
        tolerance_     = other.tolerance_;
        solver_        = other.solver_;
        scalingType_   = other.scalingType_;
        scaleCenter_   = other.scaleCenter_;
        scaleSpread_   = other.scaleSpread_;
        trained_       = other.trained_;
        iterationsRun_ = other.iterationsRun_;
        finalCost_     = other.finalCost_;
    }
    return *this;
}

LinearRegression::~LinearRegression() = default;

// ---------------------------------------------------------------------------
// Validacion
// ---------------------------------------------------------------------------
void LinearRegression::validate(const Matrix& X, const Vector& y) {
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("fit/score: la matriz X esta vacia");
    if (X.size() != y.size())
        throw std::invalid_argument("fit/score: X y y tienen distinto numero de muestras");
    const std::size_t n = X[0].size();
    for (const auto& row : X)
        if (row.size() != n)
            throw std::invalid_argument("fit/score: filas de X con distinto numero de columnas");
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------
void LinearRegression::fit(const Matrix& X, const Vector& y) {
    validate(X, y);
    weights_.assign(X[0].size(), 0.0);
    bias_          = 0.0;
    iterationsRun_ = 0;

    if (solver_ == Solver::NORMAL_EQUATION)
        fitNormalEquation(X, y);
    else
        fitGradientDescent(X, y);

    trained_   = true;
    finalCost_ = computeCost(X, y);
}

void LinearRegression::fitGradientDescent(const Matrix& X, const Vector& y) {
    const std::size_t m = X.size();
    const std::size_t n = X[0].size();
    double previousCost = std::numeric_limits<double>::max();

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        Vector gradW(n, 0.0);
        double gradB = 0.0;

        // Gradiente del MSE: dJ/dw_j = (2/m) * sum((y_hat - y) * x_j)
        for (std::size_t i = 0; i < m; ++i) {
            double yHat = bias_;
            for (std::size_t j = 0; j < n; ++j) yHat += weights_[j] * X[i][j];
            const double error = yHat - y[i];
            for (std::size_t j = 0; j < n; ++j) gradW[j] += error * X[i][j];
            gradB += error;
        }

        const double factor = 2.0 / static_cast<double>(m);
        for (std::size_t j = 0; j < n; ++j)
            weights_[j] -= learningRate_ * factor * gradW[j];
        bias_ -= learningRate_ * factor * gradB;

        ++iterationsRun_;

        const double cost = computeCost(X, y);
        if (!std::isfinite(cost))
            throw std::runtime_error(
                "fit: el costo divergio. Reduzca learningRate o escale los datos.");
        if (std::fabs(previousCost - cost) < tolerance_) break;
        previousCost = cost;
    }
}

void LinearRegression::fitNormalEquation(const Matrix& X, const Vector& y) {
    const std::size_t m = X.size();
    const std::size_t n = X[0].size();

    // Se agrega una columna de unos para absorber el bias dentro de los pesos.
    Matrix Xb(m, Vector(n + 1, 1.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) Xb[i][j + 1] = X[i][j];

    const Matrix Xt  = MatrixOps::transpose(Xb);
    const Matrix XtX = MatrixOps::multiply(Xt, Xb);
    const Vector Xty = MatrixOps::multiply(Xt, y);

    const Vector theta = MatrixOps::solve(XtX, Xty);

    bias_ = theta[0];
    for (std::size_t j = 0; j < n; ++j) weights_[j] = theta[j + 1];
    iterationsRun_ = 1;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------
double LinearRegression::predict(const Vector& x) const {
    if (!trained_)
        throw std::runtime_error("predict: el modelo no ha sido entrenado");
    if (x.size() != weights_.size())
        throw std::invalid_argument("predict: numero de caracteristicas incorrecto");

    double yHat = bias_;
    for (std::size_t j = 0; j < x.size(); ++j) yHat += weights_[j] * x[j];
    return yHat;
}

Vector LinearRegression::predict(const Matrix& X) const {
    if (!trained_)
        throw std::runtime_error("predict: el modelo no ha sido entrenado");
    Vector yHat;
    yHat.reserve(X.size());
    for (const auto& row : X) yHat.push_back(predict(row));
    return yHat;
}

// ---------------------------------------------------------------------------
// score
// ---------------------------------------------------------------------------
Metrics LinearRegression::score(const Matrix& X, const Vector& y) const {
    validate(X, y);
    const Vector yHat = predict(X);
    const std::size_t m = y.size();

    double sse = 0.0, sae = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double e = yHat[i] - y[i];
        sse += e * e;
        sae += std::fabs(e);
    }

    const double yMean = MatrixOps::mean(y);
    double sst = 0.0;
    for (double yi : y) sst += (yi - yMean) * (yi - yMean);

    Metrics mt;
    mt.mse  = sse / static_cast<double>(m);
    mt.rmse = std::sqrt(mt.mse);
    mt.mae  = sae / static_cast<double>(m);
    mt.r2   = (sst > 0.0) ? 1.0 - sse / sst : 0.0;
    return mt;
}

double LinearRegression::computeCost(const Matrix& X, const Vector& y) const {
    const std::size_t m = X.size();
    double sse = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        double yHat = bias_;
        for (std::size_t j = 0; j < X[i].size(); ++j) yHat += weights_[j] * X[i][j];
        const double e = yHat - y[i];
        sse += e * e;
    }
    return sse / static_cast<double>(m);
}

// ---------------------------------------------------------------------------
// data_scaling
// ---------------------------------------------------------------------------
Matrix LinearRegression::data_scaling(const Matrix& X, ScalingType type, bool fitParameters) {
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("data_scaling: matriz vacia");

    const std::size_t m = X.size();
    const std::size_t n = X[0].size();

    if (type == ScalingType::NONE) {
        scalingType_ = type;
        return X;
    }

    if (fitParameters) {
        scalingType_ = type;
        scaleCenter_.assign(n, 0.0);
        scaleSpread_.assign(n, 1.0);

        for (std::size_t j = 0; j < n; ++j) {
            const Vector col = MatrixOps::column(X, j);
            if (type == ScalingType::STANDARDIZATION) {
                scaleCenter_[j] = MatrixOps::mean(col);
                const double sd = MatrixOps::stddev(col);
                scaleSpread_[j] = (sd < 1e-12) ? 1.0 : sd;  // evita division por cero
            } else { // MIN_MAX
                const double mn = MatrixOps::minimum(col);
                const double mx = MatrixOps::maximum(col);
                scaleCenter_[j] = mn;
                scaleSpread_[j] = (mx - mn < 1e-12) ? 1.0 : (mx - mn);
            }
        }
    } else {
        if (scaleCenter_.size() != n)
            throw std::runtime_error(
                "data_scaling: no hay parametros de escala previos para este dataset");
    }

    Matrix Z(m, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Z[i][j] = (X[i][j] - scaleCenter_[j]) / scaleSpread_[j];

    return Z;
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------
const Vector& LinearRegression::getWeights() const { return weights_; }
double LinearRegression::getBias() const { return bias_; }
bool LinearRegression::isTrained() const { return trained_; }
std::size_t LinearRegression::getIterationsRun() const { return iterationsRun_; }
double LinearRegression::getFinalCost() const { return finalCost_; }

std::string LinearRegression::toString() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << "y_hat = ";
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        if (j > 0) os << " + ";
        os << weights_[j] << "*x" << (j + 1);
    }
    os << (bias_ >= 0 ? " + " : " - ") << std::fabs(bias_);
    return os.str();
}

void LinearRegression::summary() const {
    std::cout << "--- LinearRegression ---\n"
              << "  entrenado      : " << (trained_ ? "si" : "no") << "\n"
              << "  n_caracteristicas: " << weights_.size() << "\n"
              << "  solver         : "
              << (solver_ == Solver::NORMAL_EQUATION ? "ecuaciones normales"
                                                     : "descenso de gradiente")
              << "\n"
              << "  iteraciones    : " << iterationsRun_ << "\n"
              << "  costo final    : " << std::fixed << std::setprecision(8)
              << finalCost_ << "\n"
              << "  modelo         : " << toString() << "\n";
}
