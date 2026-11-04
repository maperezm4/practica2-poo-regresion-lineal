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
