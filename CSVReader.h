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
