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
