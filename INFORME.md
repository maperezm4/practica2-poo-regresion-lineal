# Práctica No. 2 — Programación Orientada a Objetos
### Programming Languages and Paradigms — Escuela de Ciencias Aplicadas e Ingeniería, EAFIT

**Integrantes**

| Nombre completo | ID | Correo institucional |
|---|---|---|
| Miguel Ángel Pérez Marín | 1018242716 | maperezm4@eafit.edu.co |
| Julio Cesar Llanes Cuello | 1118835249 | jcllanesc@eafit.edu.co |

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
