#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

double media(vector<double> datos) {
    double suma = 0;
    for (int i = 0; i < (int)datos.size(); i++) {
        suma = suma + datos[i];
    }
    return suma / datos.size();
}


double desviacion(vector<double> datos) {
    double m = media(datos);
    double suma = 0;
    for (int i = 0; i < (int)datos.size(); i++) {
        suma = suma + (datos[i] - m) * (datos[i] - m);
    }
    return sqrt(suma / datos.size());
}


double minimo(vector<double> datos) {
    double m = datos[0];
    for (int i = 0; i < (int)datos.size(); i++) {
        if (datos[i] < m) m = datos[i];
    }
    return m;
}


double maximo(vector<double> datos) {
    double m = datos[0];
    for (int i = 0; i < (int)datos.size(); i++) {
        if (datos[i] > m) m = datos[i];
    }
    return m;
}


vector<double> sacarColumna(vector<vector<double> > X, int col) {
    vector<double> columna;
    for (int i = 0; i < (int)X.size(); i++) {
        columna.push_back(X[i][col]);
    }
    return columna;
}


class RegresionLineal {
private:
    vector<double> pesos;          // un peso por cada caracteristica
    double bias;                   // termino independiente
    double tasaAprendizaje;        // que tan grande es cada paso
    int iteraciones;               // cuantas veces repetimos el ajuste

    vector<double> medias;         // para escalar los datos
    vector<double> desviaciones;

public:
    RegresionLineal(double tasa = 0.01, int iters = 1000) {
        bias = 0;
        tasaAprendizaje = tasa;
        iteraciones = iters;
    }

    void fit(vector<vector<double> > X, vector<double> y) {
        int m = X.size();          // numero de muestras
        int n = X[0].size();       // numero de caracteristicas

       
        pesos.clear();
        for (int j = 0; j < n; j++) {
            pesos.push_back(0.0);
        }
        bias = 0;

        for (int paso = 0; paso < iteraciones; paso++) {

            vector<double> gradientePesos(n, 0.0);
            double gradienteBias = 0;

            for (int i = 0; i < m; i++) {

                double prediccion = bias;
                for (int j = 0; j < n; j++) {
                    prediccion = prediccion + pesos[j] * X[i][j];
                }

                double error = prediccion - y[i];

                for (int j = 0; j < n; j++) {
                    gradientePesos[j] = gradientePesos[j] + error * X[i][j];
                }
                gradienteBias = gradienteBias + error;
            }

            for (int j = 0; j < n; j++) {
                pesos[j] = pesos[j] - tasaAprendizaje * (2.0 / m) * gradientePesos[j];
            }
            bias = bias - tasaAprendizaje * (2.0 / m) * gradienteBias;
        }
    }

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

    double score(vector<vector<double> > X, vector<double> y) {
        vector<double> predicciones = predict(X);
        double suma = 0;

        for (int i = 0; i < (int)y.size(); i++) {
            double error = predicciones[i] - y[i];
            suma = suma + error * error;
        }
        return suma / y.size();    
    }

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
                if (d == 0) d = 1;     
                desviaciones.push_back(d);
            }
        }

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

    vector<double> getPesos() {
        return pesos;
    }

    double getBias() {
        return bias;
    }
};


bool leerCSV(string ruta, vector<vector<double> > &X, vector<double> &y) {
    ifstream archivo(ruta.c_str());

    if (!archivo.is_open()) {
        cout << "ERROR: no se pudo abrir el archivo " << ruta << endl;
        return false;
    }

    string linea;
    bool primeraLinea = true;

    while (getline(archivo, linea)) {

        if (primeraLinea) {
            primeraLinea = false;
            continue;
        }

        if (linea.size() < 2) continue;   

        vector<double> numeros;
        stringstream ss(linea);
        string campo;

        while (getline(ss, campo, ',')) {
            numeros.push_back(atof(campo.c_str()));
        }

        if (numeros.size() < 2) continue;

        vector<double> fila;
        for (int j = 0; j < (int)numeros.size() - 1; j++) {
            fila.push_back(numeros[j]);
        }

        X.push_back(fila);
        y.push_back(numeros[numeros.size() - 1]);  
    }

    archivo.close();
    return true;
}


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


    RegresionLineal modelo(0.05, 5000);

    vector<vector<double> > X_train_escalado = modelo.data_scaling(X_train, true);
    vector<vector<double> > X_test_escalado  = modelo.data_scaling(X_test, false);

    modelo.fit(X_train_escalado, y_train);

    vector<double> pesos = modelo.getPesos();

    cout << "weights: ";
    for (int j = 0; j < (int)pesos.size(); j++) {
        cout << pesos[j] << "  ";
    }
    cout << endl;
    cout << "bias: " << modelo.getBias() << endl << endl;

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

RegresionLineal modeloGlobal(0.01, 1000);

void analisisDeMemoria() {
    cout << endl;
    cout << "==================================================" << endl;
    cout << "PARTE 2 - ANALISIS DE MEMORIA" << endl;
    cout << "==================================================" << endl << endl;


    cout << "1. MEMORIA GLOBAL" << endl;
    cout << "   modeloGlobal   -> direccion: " << &modeloGlobal
         << "   tamano: " << sizeof(modeloGlobal) << " bytes" << endl;
    cout << "   Se crea antes del main y vive hasta que termina el programa." << endl << endl;


    RegresionLineal modeloLocal1(0.02, 2000);
    RegresionLineal modeloLocal2(0.03, 3000);

    cout << "2. MEMORIA LOCAL (stack)" << endl;
    cout << "   modeloLocal1   -> direccion: " << &modeloLocal1
         << "   tamano: " << sizeof(modeloLocal1) << " bytes" << endl;
    cout << "   modeloLocal2   -> direccion: " << &modeloLocal2
         << "   tamano: " << sizeof(modeloLocal2) << " bytes" << endl;
    cout << "   Se borran solos cuando termina la funcion." << endl << endl;


    RegresionLineal *modeloDinamico = new RegresionLineal(0.04, 4000);

    cout << "3. MEMORIA DINAMICA (heap)" << endl;
    cout << "   modeloDinamico -> direccion: " << modeloDinamico
         << "   tamano: " << sizeof(*modeloDinamico) << " bytes" << endl;
    cout << "   El puntero esta en el stack, en: " << &modeloDinamico << endl;
    cout << "   El puntero ocupa " << sizeof(modeloDinamico)
         << " bytes (solo guarda la direccion)." << endl;
    cout << "   Hay que borrarlo con delete o queda ocupando memoria." << endl << endl;


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


    delete modeloDinamico;     
    delete[] arreglo;          

    modeloDinamico = NULL;
    arreglo = NULL;

    cout << "6. LIBERACION DE MEMORIA" << endl;
    cout << "   Memoria liberada. Cada new con su delete y cada new[] con su delete[]." << endl;
    cout << "   Los punteros quedan en NULL para no usarlos por accidente." << endl;
}


int main() {
    cout << "PRACTICA 2 - PROGRAMACION ORIENTADA A OBJETOS" << endl;
    cout << "Regresion lineal simple y multiple en C++" << endl;

    probarModelo("data/ice_cream_selling_data.csv", "REGRESION SIMPLE (venta de helados)");
    probarModelo("data/student_exam_scores.csv", "REGRESION MULTIPLE (notas de estudiantes)");


    analisisDeMemoria();

    cout << endl << "Fin del programa." << endl;
    return 0;
}
