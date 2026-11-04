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
