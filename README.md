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
