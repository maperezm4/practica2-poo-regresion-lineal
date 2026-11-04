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
