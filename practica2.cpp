// ============================================================
// Practica 2 - Programacion Orientada a Objetos
// Lenguajes y Paradigmas de la Computacion - EAFIT
//
// Miguel Angel Perez Marin  - 1018242716 - maperezm4@eafit.edu.co
// Julio Cesar Llanes Cuello - 1118835249 - jcllanesc@eafit.edu.co
//
// Compilar:  g++ practica2.cpp -o practica2
// Ejecutar:  ./practica2
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

// ============================================================
// FUNCIONES AUXILIARES DE ESTADISTICA
// (el enunciado pide que las desarrollemos nosotros)
// ============================================================

// Calcula el promedio de una lista de numeros
double media(vector<double> datos) {
    double suma = 0;
    for (int i = 0; i < (int)datos.size(); i++) {
        suma = suma + datos[i];
    }
    return suma / datos.size();
}

// Calcula la desviacion estandar de una lista de numeros
double desviacion(vector<double> datos) {
    double m = media(datos);
    double suma = 0;
    for (int i = 0; i < (int)datos.size(); i++) {
        suma = suma + (datos[i] - m) * (datos[i] - m);
    }
    return sqrt(suma / datos.size());
}

// Devuelve el valor mas pequeno de una lista
double minimo(vector<double> datos) {
    double m = datos[0];
    for (int i = 0; i < (int)datos.size(); i++) {
        if (datos[i] < m) m = datos[i];
    }
    return m;
}

// Devuelve el valor mas grande de una lista
double maximo(vector<double> datos) {
    double m = datos[0];
    for (int i = 0; i < (int)datos.size(); i++) {
        if (datos[i] > m) m = datos[i];
    }
    return m;
}

// Saca una columna completa de la matriz X
vector<double> sacarColumna(vector<vector<double> > X, int col) {
    vector<double> columna;
    for (int i = 0; i < (int)X.size(); i++) {
        columna.push_back(X[i][col]);
    }
    return columna;
}

// ============================================================
// CLASE REGRESION LINEAL
// Modelo:  y = w1*x1 + w2*x2 + ... + wn*xn + bias
// ============================================================

class RegresionLineal {
private:
    vector<double> pesos;          // un peso por cada caracteristica
    double bias;                   // termino independiente
    double tasaAprendizaje;        // que tan grande es cada paso
    int iteraciones;               // cuantas veces repetimos el ajuste

    vector<double> medias;         // para escalar los datos
    vector<double> desviaciones;

public:
    // Constructor
    // Los valores por defecto sirven para poder crear un arreglo de
    // objetos con new[], que los construye sin pasarle parametros.
    RegresionLineal(double tasa = 0.01, int iters = 1000) {
        bias = 0;
        tasaAprendizaje = tasa;
        iteraciones = iters;
    }

    // --------------------------------------------------------
    // fit: entrena el modelo con descenso de gradiente
    // --------------------------------------------------------
    void fit(vector<vector<double> > X, vector<double> y) {
        int m = X.size();          // numero de muestras
        int n = X[0].size();       // numero de caracteristicas

        // empezamos con todos los pesos en cero
        pesos.clear();
        for (int j = 0; j < n; j++) {
            pesos.push_back(0.0);
        }
        bias = 0;

        // repetimos el ajuste muchas veces
        for (int paso = 0; paso < iteraciones; paso++) {

            // acumuladores del gradiente
            vector<double> gradientePesos(n, 0.0);
            double gradienteBias = 0;

            // recorremos todas las muestras
            for (int i = 0; i < m; i++) {

                // calculamos lo que predice el modelo ahora mismo
                double prediccion = bias;
                for (int j = 0; j < n; j++) {
                    prediccion = prediccion + pesos[j] * X[i][j];
                }

                // que tan lejos estamos del valor real
                double error = prediccion - y[i];

                // acumulamos el error por cada caracteristica
                for (int j = 0; j < n; j++) {
                    gradientePesos[j] = gradientePesos[j] + error * X[i][j];
                }
                gradienteBias = gradienteBias + error;
            }

            // movemos los pesos en direccion contraria al error
            for (int j = 0; j < n; j++) {
                pesos[j] = pesos[j] - tasaAprendizaje * (2.0 / m) * gradientePesos[j];
            }
            bias = bias - tasaAprendizaje * (2.0 / m) * gradienteBias;
        }
    }

    // --------------------------------------------------------
    // predict: calcula las predicciones de un conjunto de datos
    // --------------------------------------------------------
    vector<double> predict(vector<vector<double> > X) {
        vector<double> predicciones;

        for (int i = 0; i < (int)X.size(); i++) {
            double valor = bias;
            for (int j = 0; j < (int)X[i].size(); j++) {
                valor = valor + pesos[j] * X[i][j];
            }
            predicciones.push_back(valor);
        }
        return predicciones;
    }

    // --------------------------------------------------------
    // score: calcula el error de las predicciones (MSE)
    // --------------------------------------------------------
    double score(vector<vector<double> > X, vector<double> y) {
        vector<double> predicciones = predict(X);
        double suma = 0;

        for (int i = 0; i < (int)y.size(); i++) {
            double error = predicciones[i] - y[i];
            suma = suma + error * error;
        }
        return suma / y.size();    // error cuadratico medio
    }

    // --------------------------------------------------------
    // r2: que tan bien explica el modelo los datos (0 malo, 1 perfecto)
    // --------------------------------------------------------
    double r2(vector<vector<double> > X, vector<double> y) {
        vector<double> predicciones = predict(X);
        double promedio = media(y);
        double errorModelo = 0;
        double errorPromedio = 0;

        for (int i = 0; i < (int)y.size(); i++) {
            errorModelo = errorModelo + (predicciones[i] - y[i]) * (predicciones[i] - y[i]);
            errorPromedio = errorPromedio + (y[i] - promedio) * (y[i] - promedio);
        }
        return 1 - (errorModelo / errorPromedio);
    }

    // --------------------------------------------------------
    // data_scaling: pone todas las columnas en la misma escala
    //
    // guardar = true  -> calcula y guarda media y desviacion (entrenamiento)
    // guardar = false -> usa las que ya guardo (prueba)
    // --------------------------------------------------------
    vector<vector<double> > data_scaling(vector<vector<double> > X, bool guardar) {
        int m = X.size();
        int n = X[0].size();

        if (guardar) {
            medias.clear();
            desviaciones.clear();

            for (int j = 0; j < n; j++) {
                vector<double> columna = sacarColumna(X, j);
                medias.push_back(media(columna));

                double d = desviacion(columna);
                if (d == 0) d = 1;      // evitamos dividir por cero
                desviaciones.push_back(d);
            }
        }

        // aplicamos la formula  z = (x - media) / desviacion
        vector<vector<double> > escalada;
        for (int i = 0; i < m; i++) {
            vector<double> fila;
            for (int j = 0; j < n; j++) {
                fila.push_back((X[i][j] - medias[j]) / desviaciones[j]);
            }
            escalada.push_back(fila);
        }
        return escalada;
    }

    // --------------------------------------------------------
    // Metodos para consultar el modelo desde afuera
    // --------------------------------------------------------
    vector<double> getPesos() {
        return pesos;
    }

    double getBias() {
        return bias;
    }
};

// ============================================================
// LECTURA DEL ARCHIVO CSV
// La ultima columna del archivo es el valor que queremos predecir
// ============================================================

bool leerCSV(string ruta, vector<vector<double> > &X, vector<double> &y) {
    ifstream archivo(ruta.c_str());

    if (!archivo.is_open()) {
        cout << "ERROR: no se pudo abrir el archivo " << ruta << endl;
        return false;
    }

    string linea;
    bool primeraLinea = true;

    while (getline(archivo, linea)) {

        // la primera linea son los nombres de las columnas, la saltamos
        if (primeraLinea) {
            primeraLinea = false;
            continue;
        }

        if (linea.size() < 2) continue;   // linea vacia

        // separamos la linea por comas
        vector<double> numeros;
        stringstream ss(linea);
        string campo;

        while (getline(ss, campo, ',')) {
            numeros.push_back(atof(campo.c_str()));
        }

        if (numeros.size() < 2) continue;

        // todas las columnas menos la ultima son las caracteristicas
        vector<double> fila;
        for (int j = 0; j < (int)numeros.size() - 1; j++) {
            fila.push_back(numeros[j]);
        }

        X.push_back(fila);
        y.push_back(numeros[numeros.size() - 1]);   // la ultima columna
    }

    archivo.close();
    return true;
}

// ============================================================
// PARTE 1 - RETO POO (punto 3.1 del enunciado)
// ============================================================

void probarModelo(string ruta, string nombre) {
    cout << endl;
    cout << "==================================================" << endl;
    cout << nombre << endl;
    cout << "==================================================" << endl;

    vector<vector<double> > X;
    vector<double> y;

    if (!leerCSV(ruta, X, y)) return;

    cout << "Archivo: " << ruta << endl;
    cout << "Muestras: " << X.size() << endl;
    cout << "Caracteristicas: " << X[0].size() << endl << endl;

    // estadisticas de cada columna
    for (int j = 0; j < (int)X[0].size(); j++) {
        vector<double> columna = sacarColumna(X, j);
        cout << "  x" << (j + 1)
             << ":  media = " << media(columna)
             << "   desviacion = " << desviacion(columna)
             << "   min = " << minimo(columna)
             << "   max = " << maximo(columna) << endl;
    }
    cout << "  y :  media = " << media(y)
         << "   desviacion = " << desviacion(y) << endl << endl;

    // --------------------------------------------------------
    // Separamos los datos: 80% entrenamiento, 20% prueba
    // Tomamos una de cada cinco muestras para prueba. Lo hacemos asi
    // y no cortando por la mitad porque el archivo de helados viene
    // ordenado por temperatura, y un corte simple dejaria todas las
    // temperaturas altas en el conjunto de prueba.
    // --------------------------------------------------------
    vector<vector<double> > X_train, X_test;
    vector<double> y_train, y_test;

    for (int i = 0; i < (int)X.size(); i++) {
        if (i % 5 == 4) {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        } else {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        }
    }

    cout << "Entrenamiento: " << X_train.size() << " muestras" << endl;
    cout << "Prueba: " << X_test.size() << " muestras" << endl << endl;

    // --------------------------------------------------------
    // Creamos el modelo y lo entrenamos
    // --------------------------------------------------------
    RegresionLineal modelo(0.05, 5000);

    // escalamos: con true aprende media y desviacion, con false las reusa
    vector<vector<double> > X_train_escalado = modelo.data_scaling(X_train, true);
    vector<vector<double> > X_test_escalado  = modelo.data_scaling(X_test, false);

    modelo.fit(X_train_escalado, y_train);

    // --------------------------------------------------------
    // Mostramos los resultados
    // --------------------------------------------------------
    vector<double> pesos = modelo.getPesos();

    cout << "weights: ";
    for (int j = 0; j < (int)pesos.size(); j++) {
        cout << pesos[j] << "  ";
    }
    cout << endl;
    cout << "bias: " << modelo.getBias() << endl << endl;

    // predicciones sobre los datos de prueba
    vector<double> y_hat = modelo.predict(X_test_escalado);

    cout << "Predicciones sobre el conjunto de prueba:" << endl;
    for (int i = 0; i < (int)y_hat.size() && i < 10; i++) {
        cout << "  y_hat: " << y_hat[i]
             << "    y_real: " << y_test[i]
             << "    error: " << (y_hat[i] - y_test[i]) << endl;
    }
    if ((int)y_hat.size() > 10) {
        cout << "  ... (" << y_hat.size() - 10 << " predicciones mas)" << endl;
    }
    cout << endl;

    cout << "score (error) entrenamiento: " << modelo.score(X_train_escalado, y_train) << endl;
    cout << "score (error) prueba:        " << modelo.score(X_test_escalado, y_test) << endl;
    cout << "R2 entrenamiento: " << modelo.r2(X_train_escalado, y_train) << endl;
    cout << "R2 prueba:        " << modelo.r2(X_test_escalado, y_test) << endl;
}

// ============================================================
// PARTE 2 - ANALISIS DE MEMORIA (punto 3.2 del enunciado)
// ============================================================

// Este objeto es GLOBAL: se crea antes del main y vive todo el programa
RegresionLineal modeloGlobal(0.01, 1000);

void analisisDeMemoria() {
    cout << endl;
    cout << "==================================================" << endl;
    cout << "PARTE 2 - ANALISIS DE MEMORIA" << endl;
    cout << "==================================================" << endl << endl;

    // --------------------------------------------------------
    // 1. Memoria global
    // --------------------------------------------------------
    cout << "1. MEMORIA GLOBAL" << endl;
    cout << "   modeloGlobal   -> direccion: " << &modeloGlobal
         << "   tamano: " << sizeof(modeloGlobal) << " bytes" << endl;
    cout << "   Se crea antes del main y vive hasta que termina el programa." << endl << endl;

    // --------------------------------------------------------
    // 2. Memoria local (stack)
    // --------------------------------------------------------
    RegresionLineal modeloLocal1(0.02, 2000);
    RegresionLineal modeloLocal2(0.03, 3000);

    cout << "2. MEMORIA LOCAL (stack)" << endl;
    cout << "   modeloLocal1   -> direccion: " << &modeloLocal1
         << "   tamano: " << sizeof(modeloLocal1) << " bytes" << endl;
    cout << "   modeloLocal2   -> direccion: " << &modeloLocal2
         << "   tamano: " << sizeof(modeloLocal2) << " bytes" << endl;
    cout << "   Se borran solos cuando termina la funcion." << endl << endl;

    // --------------------------------------------------------
    // 3. Memoria dinamica (heap)
    // --------------------------------------------------------
    RegresionLineal *modeloDinamico = new RegresionLineal(0.04, 4000);

    cout << "3. MEMORIA DINAMICA (heap)" << endl;
    cout << "   modeloDinamico -> direccion: " << modeloDinamico
         << "   tamano: " << sizeof(*modeloDinamico) << " bytes" << endl;
    cout << "   El puntero esta en el stack, en: " << &modeloDinamico << endl;
    cout << "   El puntero ocupa " << sizeof(modeloDinamico)
         << " bytes (solo guarda la direccion)." << endl;
    cout << "   Hay que borrarlo con delete o queda ocupando memoria." << endl << endl;

    // --------------------------------------------------------
    // 4. Vector de objetos en memoria dinamica
    // --------------------------------------------------------
    int N = 5;
    RegresionLineal *arreglo = new RegresionLineal[N];

    cout << "4. VECTOR DE OBJETOS EN MEMORIA DINAMICA" << endl;
    for (int i = 0; i < N; i++) {
        cout << "   arreglo[" << i << "] -> " << &arreglo[i] << endl;
    }
    cout << endl;
    cout << "   sizeof(arreglo) = " << sizeof(arreglo)
         << " bytes  <- es el tamano del PUNTERO, no del arreglo" << endl;
    cout << "   Tamano real = " << N << " * " << sizeof(RegresionLineal)
         << " = " << N * sizeof(RegresionLineal) << " bytes" << endl;
    cout << "   Las direcciones van de " << sizeof(RegresionLineal)
         << " en " << sizeof(RegresionLineal)
         << " bytes, o sea que los objetos estan pegados uno tras otro." << endl << endl;

    // --------------------------------------------------------
    // 5. Por que el objeto pesa lo que pesa
    // --------------------------------------------------------
    cout << "5. DE DONDE SALE EL TAMANO DEL OBJETO" << endl;
    cout << "   sizeof(RegresionLineal) = " << sizeof(RegresionLineal) << " bytes" << endl << endl;
    cout << "   Lo que tiene adentro:" << endl;
    cout << "     pesos         (vector) : " << sizeof(vector<double>) << " bytes" << endl;
    cout << "     bias          (double) : " << sizeof(double) << " bytes" << endl;
    cout << "     tasaAprendizaje(double): " << sizeof(double) << " bytes" << endl;
    cout << "     iteraciones   (int)    : " << sizeof(int) << " bytes" << endl;
    cout << "     medias        (vector) : " << sizeof(vector<double>) << " bytes" << endl;
    cout << "     desviaciones  (vector) : " << sizeof(vector<double>) << " bytes" << endl << endl;
    cout << "   Cada vector ocupa " << sizeof(vector<double>)
         << " bytes sin importar cuantos numeros guarde," << endl;
    cout << "   porque por dentro son 3 punteros. Los numeros estan en otro" << endl;
    cout << "   lado de la memoria, no dentro del objeto." << endl << endl;
    cout << "   Ademas el compilador rellena huecos para que todo quede" << endl;
    cout << "   alineado de a 8 bytes, asi que el total puede ser un poco" << endl;
    cout << "   mayor que la suma de las partes." << endl << endl;

    // --------------------------------------------------------
    // 6. Liberamos la memoria dinamica
    // --------------------------------------------------------
    delete modeloDinamico;     // new  va con delete
    delete[] arreglo;          // new[] va con delete[]

    modeloDinamico = NULL;
    arreglo = NULL;

    cout << "6. LIBERACION DE MEMORIA" << endl;
    cout << "   Memoria liberada. Cada new con su delete y cada new[] con su delete[]." << endl;
    cout << "   Los punteros quedan en NULL para no usarlos por accidente." << endl;
}

// ============================================================
// PROGRAMA PRINCIPAL
// ============================================================

int main() {
    cout << "PRACTICA 2 - PROGRAMACION ORIENTADA A OBJETOS" << endl;
    cout << "Regresion lineal simple y multiple en C++" << endl;

    // Parte 1: el reto de POO
    probarModelo("data/ice_cream_selling_data.csv", "REGRESION SIMPLE (venta de helados)");
    probarModelo("data/student_exam_scores.csv", "REGRESION MULTIPLE (notas de estudiantes)");

    // Parte 2: el analisis de memoria
    analisisDeMemoria();

    cout << endl << "Fin del programa." << endl;
    return 0;
}
