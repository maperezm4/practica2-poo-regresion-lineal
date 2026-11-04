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
