# Práctica No. 2 — Programación Orientada a Objetos
### Lenguajes y Paradigmas de la Computación — EAFIT

**Integrantes**

| Nombre completo | ID | Correo institucional |
|---|---|---|
| Miguel Ángel Pérez Marín | 1018242716 | maperezm4@eafit.edu.co |
| Julio Cesar Llanes Cuello | 1118835249 | jcllanesc@eafit.edu.co |

**Repositorio:** https://github.com/maperezm4/practica2-poo-regresion-lineal
**Video:** https://youtu.be/RnYFoRcU0tk

---

## 1. Qué hicimos

Implementamos en C++ una clase que calcula modelos de regresión lineal, tanto simple como
múltiple. No usamos ninguna librería externa: las operaciones de estadística y la lectura de los
archivos CSV las programamos nosotros.

Todo está en un solo archivo, `practica2.cpp`, que contiene las dos partes del enunciado: el reto
de programación orientada a objetos y el análisis de memoria.

**Para compilar y ejecutar:**

```bash
g++ practica2.cpp -o practica2
./practica2
```

Los archivos de datos van en la carpeta `data/`.

---

## 2. Cómo está organizado el código

| Parte | Qué hace |
|---|---|
| Funciones de estadística | `media`, `desviacion`, `minimo`, `maximo`, `sacarColumna` |
| Clase `RegresionLineal` | El modelo: `fit`, `predict`, `score`, `r2` y `data_scaling` |
| `leerCSV` | Lee un archivo CSV y lo convierte en una matriz y un vector |
| `probarModelo` | Entrena y evalúa el modelo con un dataset (parte 3.1) |
| `analisisDeMemoria` | Crea objetos en las tres zonas de memoria (parte 3.2) |
| `main` | Llama a las dos partes |

### Por qué una sola clase sirve para los dos casos

La regresión simple usa una variable y la múltiple usa varias, pero la fórmula es la misma:

```
y = w1*x1 + w2*x2 + ... + wn*xn + bias
```

Lo único que cambia es cuántos pesos hay. Por eso internamente siempre guardamos los datos en una
matriz de m filas (las muestras) por n columnas (las características), y la regresión simple es
simplemente el caso en que n vale 1. No hicieron falta dos clases.

### Encapsulamiento

Los atributos `pesos` y `bias` son **privados**. Desde fuera de la clase solo se pueden consultar
con `getPesos()` y `getBias()`, nunca modificar directamente. Así nadie puede dejar el modelo en
un estado inconsistente desde afuera.

---

## 3. Documentación de los métodos

### Funciones de estadística

| Función | Qué hace |
|---|---|
| `media(datos)` | Suma todos los valores y los divide por la cantidad |
| `desviacion(datos)` | Mide qué tan dispersos están los datos respecto a la media |
| `minimo(datos)` | Recorre la lista y devuelve el valor más pequeño |
| `maximo(datos)` | Recorre la lista y devuelve el valor más grande |
| `sacarColumna(X, col)` | Extrae una columna completa de la matriz. Lo usa `data_scaling` |

### Clase `RegresionLineal`

**Atributos**

| Atributo | Para qué sirve |
|---|---|
| `pesos` | Un peso por cada característica (lo que pide el enunciado como *weights*) |
| `bias` | El término independiente |
| `tasaAprendizaje` | Qué tan grande es cada paso al entrenar |
| `iteraciones` | Cuántas veces se repite el ajuste |
| `medias`, `desviaciones` | Valores guardados para escalar los datos de prueba igual que los de entrenamiento |

**`RegresionLineal(tasa, iteraciones)`** — Constructor. Deja el bias en cero y guarda los
hiperparámetros. Tiene valores por defecto porque `new RegresionLineal[5]` construye los objetos
sin pasarle parámetros.

**`fit(X, y)`** — Entrena el modelo por **descenso de gradiente**. Recibe la matriz de datos `X`
y el vector de respuestas correctas `y`.

Funciona así: empieza con todos los pesos en cero. En cada iteración recorre todas las muestras,
calcula qué predice el modelo con los pesos actuales, mide la diferencia contra el valor real y
acumula ese error. Al final de la pasada mueve los pesos un poquito en dirección contraria al
error. Repite ese proceso 5000 veces.

La idea es la de bajar una montaña con niebla: uno no ve el valle, pero sí siente hacia dónde
baja el piso, da un paso en esa dirección y repite.

**`predict(X)`** — Devuelve las predicciones del modelo para un conjunto de datos. Para cada fila
multiplica cada característica por su peso, suma todo y le agrega el bias.

**`score(X, y)`** — Calcula el error de las predicciones. Devuelve el **error cuadrático medio**:
para cada muestra calcula la diferencia entre lo predicho y lo real, la eleva al cuadrado, y
promedia. Entre más bajo, mejor.

**`r2(X, y)`** — Indica qué tan bien explica el modelo los datos. Va de 0 a 1: 1 es un ajuste
perfecto y 0 significa que el modelo no sirve de nada. Es más fácil de interpretar que el error
cuadrático porque no depende de las unidades.

**`data_scaling(X, guardar)`** — Pone todas las columnas en la misma escala con la fórmula
`z = (x - media) / desviacion`.

El parámetro `guardar` es importante. Con `true`, el método calcula la media y la desviación de
ese conjunto y **las guarda**. Con `false`, **reutiliza** las que ya guardó. Al conjunto de
entrenamiento se le pasa `true` y al de prueba `false`, porque los dos tienen que quedar en la
misma escala. Si a cada uno le calculáramos sus propias estadísticas, los pesos aprendidos con el
entrenamiento ya no servirían para el conjunto de prueba.

**`getPesos()` y `getBias()`** — Permiten consultar los resultados del entrenamiento desde fuera
de la clase, sin exponer los atributos.

### Otras funciones

**`leerCSV(ruta, X, y)`** — Abre el archivo, salta la primera línea (los nombres de las
columnas), y por cada línea separa los números por comas. Todas las columnas menos la última van
a `X`; la última va a `y`, porque es el valor que queremos predecir. Devuelve `false` si el
archivo no se puede abrir.

**`probarModelo(ruta, nombre)`** — Hace todo el proceso con un dataset: lo lee, muestra sus
estadísticas, lo divide en entrenamiento y prueba, escala, entrena, predice y muestra los
resultados.

**`analisisDeMemoria()`** — Es la parte 3.2, explicada más abajo.

---

## 4. Resultados (punto 3.1)

La salida completa está en `evidencias/salida_programa.txt`.

### 4.1 Regresión simple — venta de helados

**Entrada:** 49 muestras, 1 característica (temperatura → ventas).

```
x1 (temperatura): media = 0.2718   desviacion = 2.67    min = -4.66   max = 4.90
y  (ventas)     : media = 15.9053  desviacion = 12.1389

Entrenamiento: 40 muestras
Prueba: 9 muestras

weights: -2.09892
bias: 16.2385

  y_hat: 19.1906    y_real: 32.2845    error: -13.094
  y_hat: 18.478     y_real: 20.2797    error: -1.80166
  y_hat: 17.7149    y_real: 12.6152    error: 5.09976
  y_hat: 16.5528    y_real: 4.67364    error: 11.8791
  y_hat: 15.9078    y_real: 2.57678    error: 13.331
  ...

score (error) entrenamiento: 148.875
score (error) prueba:        116.565
R2 entrenamiento: 0.0287
R2 prueba:        0.0149
```

**Qué significa esto.** El R² da prácticamente **cero**, o sea que el modelo no explica nada. Al
principio pensamos que teníamos un error en el código, pero revisando los datos nos dimos cuenta
de que el problema es otro: **este dataset no tiene una relación lineal**.

Si uno mira los datos ordenados por temperatura, las ventas son altas en los dos extremos (unas
42 unidades a −4.66 °C y unas 41 a 4.90 °C) y caen casi a cero alrededor de los 0 °C. Los puntos
forman una **U**, no una recta. Y la mejor recta posible sobre una U es una recta casi
horizontal, que es justo lo que encontró el modelo: el bias (16.24) está muy cerca de la media de
las ventas (15.91).

O sea que un R² de cero no siempre significa que el programa esté mal; acá significa que la
regresión lineal no es el modelo adecuado para estos datos.

### 4.2 Regresión múltiple — notas de estudiantes

**Entrada:** 200 muestras, 4 características (horas de estudio, horas de sueño, porcentaje de
asistencia y notas previas → nota del examen).

```
x1 (horas de estudio): media = 6.3255   desviacion = 3.2192   min = 1.0    max = 12.0
x2 (horas de sueno)  : media = 6.6220   desviacion = 1.4934   min = 4.0    max = 9.0
x3 (% asistencia)    : media = 74.830   desviacion = 14.214   min = 50.3   max = 100.0
x4 (notas previas)   : media = 66.800   desviacion = 15.625   min = 40.0   max = 95.0
y  (nota del examen) : media = 33.955   desviacion = 6.7726

Entrenamiento: 160 muestras
Prueba: 40 muestras

weights: 4.88581  1.49737  1.52692  2.92372
bias: 34.3875

  y_hat: 40.4937    y_real: 40.3    error: 0.193732
  y_hat: 28.4178    y_real: 24.7    error: 3.71778
  y_hat: 41.6925    y_real: 41.1    error: 0.592547
  y_hat: 22.285     y_real: 17.1    error: 5.18502
  y_hat: 41.4991    y_real: 39.2    error: 2.29913
  ...

score (error) entrenamiento: 7.23996
score (error) prueba:        7.5781
R2 entrenamiento: 0.837611
R2 prueba:        0.83965
```

**Qué significa esto.** El modelo explica el **84 %** de los datos, que es un buen resultado.

Como los datos están escalados, los pesos se pueden comparar entre sí:

| Variable | Peso | Lectura |
|---|---|---|
| Horas de estudio | 4.89 | La que más influye |
| Notas previas | 2.92 | La segunda |
| % de asistencia | 1.53 | Influye poco |
| Horas de sueño | 1.50 | La que menos influye |

Sin escalar los datos esta comparación no se podría hacer, porque cada variable estaría en
unidades distintas (horas contra porcentajes contra puntos).

El bias (34.39) es la nota esperada de un estudiante que está en el promedio de las cuatro
variables, y queda muy cerca de la media real de las notas (33.96), que es lo esperado.

Otra cosa que vale la pena notar: el R² de entrenamiento (0.8376) y el de prueba (0.8397) son
casi iguales. Eso significa que el modelo funciona igual de bien con datos que nunca vio, o sea
que no está simplemente memorizando los de entrenamiento.

---

## 5. Análisis de memoria (punto 3.2)

Resultados de una ejecución en Linux de 64 bits.

### 5.1 Direcciones de los objetos

| Objeto | Zona | Dirección | Tamaño |
|---|---|---|---|
| `modeloGlobal` | Global | `0x55fb037ef160` | 96 bytes |
| `modeloLocal1` | Stack | `0x7ffe85612720` | 96 bytes |
| `modeloLocal2` | Stack | `0x7ffe85612780` | 96 bytes |
| `*modeloDinamico` | Heap | `0x55fb22f416e0` | 96 bytes |

**Qué se observa:**

- El objeto **global** se crea antes de que empiece `main` y vive hasta que el programa termina.
- Los objetos **locales** están en direcciones que empiezan en `0x7ffe`, mucho más altas que las
  demás. Se borran solos cuando la función termina.
- El objeto **dinámico** está en una dirección intermedia. El puntero que lo apunta está en el
  stack y ocupa solo 8 bytes, porque lo único que guarda es la dirección; el objeto está en otro
  lado.
- Las direcciones cambian en cada ejecución (por una protección del sistema operativo llamada
  ASLR), pero el orden entre las zonas siempre es el mismo: primero las globales, después el
  heap, y por último el stack.

### 5.2 Por qué el objeto ocupa 96 bytes

Lo que tiene adentro la clase:

| Atributo | Tipo | Bytes |
|---|---|---|
| `pesos` | `vector<double>` | 24 |
| `bias` | `double` | 8 |
| `tasaAprendizaje` | `double` | 8 |
| `iteraciones` | `int` | 4 |
| `medias` | `vector<double>` | 24 |
| `desviaciones` | `vector<double>` | 24 |
| **Suma** | | **92** |
| **`sizeof` real** | | **96** |

Hay dos cosas que explican el valor:

**Primera: cada `vector` ocupa 24 bytes sin importar cuántos números guarde.** Por dentro un
vector son tres punteros de 8 bytes (dónde empiezan los datos, dónde terminan y hasta dónde llega
la memoria reservada). Los números en sí **no están dentro del objeto**, están en otra parte de
la memoria. Por eso un vector con 2 números y uno con 1000 ocupan lo mismo según `sizeof`.

**Segunda: el relleno (padding).** El compilador acomoda los datos en direcciones múltiplos de 8
porque el procesador los lee más rápido así. Como la suma da 92, se rellena hasta 96 para
completar el múltiplo de 8 más cercano.

Lo importante de todo esto: **`sizeof` se calcula cuando se compila el programa**, así que mide
el tamaño fijo del objeto, no la memoria que ese objeto va a pedir mientras corre.

### 5.3 Vector de objetos en memoria dinámica

```
arreglo[0] -> 0x55fb22f457c8
arreglo[1] -> 0x55fb22f45828
arreglo[2] -> 0x55fb22f45888
arreglo[3] -> 0x55fb22f458e8
arreglo[4] -> 0x55fb22f45948

sizeof(arreglo) = 8 bytes
Tamano real = 5 * 96 = 480 bytes
```

`sizeof(arreglo)` da **8 bytes**, que al principio confunde. Da 8 porque `arreglo` es un
**puntero**, y lo único que guarda es una dirección. `sizeof` no tiene forma de saber cuántos
objetos se reservaron.

El tamaño real es **5 × 96 = 480 bytes**. Se puede comprobar mirando las direcciones: entre
`arreglo[0]` y `arreglo[1]` hay exactamente 96 bytes de diferencia, lo mismo entre todos los
demás. Eso quiere decir que los cinco objetos están guardados uno pegado al otro.

### 5.4 Liberación de la memoria

```cpp
delete modeloDinamico;    // new va con delete
delete[] arreglo;         // new[] va con delete[]
modeloDinamico = NULL;
arreglo = NULL;
```

Las reglas que seguimos: cada `new` tiene su `delete` y cada `new[]` tiene su `delete[]`
(mezclarlos da errores raros). Después de borrar, los punteros se dejan en `NULL` para que no
queden apuntando a memoria que ya no existe. Si no se libera, esa memoria queda ocupada hasta que
el programa termina, y eso es una **fuga de memoria**.

---

## 6. Tres fallos históricos por manejo de memoria dinámica

### 6.1 Heartbleed — OpenSSL (2014)

**Quiénes estuvieron implicados.** El error lo introdujo Robin Seggelmann, un desarrollador que
agregó una función nueva a OpenSSL, la librería de seguridad que usa buena parte de internet. Lo
descubrieron en 2014, de forma independiente, investigadores de Google y de la empresa finlandesa
Codenomicon. Como OpenSSL lo usaba Apache, que en ese momento corría cerca del 66 % de los sitios
web, el problema afectó a casi todo internet.

**Qué pasó.** El servidor reservaba un espacio de memoria del tamaño que le decía el cliente, y
después copiaba ahí los datos **sin verificar que ese tamaño fuera el real**. Si el cliente mentía
y pedía más de lo que había mandado, el servidor le devolvía lo que estuviera al lado en la
memoria.

**Consecuencias.** Cualquiera en internet podía leer hasta 64 KB de la memoria del servidor por
cada pedido, y repetirlo cuantas veces quisiera. Ahí podían salir contraseñas, cookies de sesión,
datos de tarjetas y hasta las claves privadas de los certificados de seguridad. El ataque no
dejaba ningún rastro en los registros del servidor. El costo de arreglarlo a nivel mundial se
estimó en unos 500 millones de dólares.

**Cómo se arregló.** El parche fue muy simple: verificar que el tamaño que declara el cliente
coincida con el que realmente mandó, y descartar el mensaje si no coincide. Salió en la versión
1.0.1g de OpenSSL, el 7 de abril de 2014. Se pudo haber evitado desde el principio con esa misma
comprobación, o con una revisión de seguridad del código.

**Fuentes:**
- OWASP — Heartbleed Bug: https://owasp.org/www-community/vulnerabilities/Heartbleed_Bug
- Black Duck — Heartbleed Bug: https://www.blackduck.com/blog/heartbleed-bug.html

### 6.2 Cloudbleed — Cloudflare (2017)

**Quiénes estuvieron implicados.** Cloudflare, una empresa que hace de intermediaria entre los
usuarios y millones de sitios web. Lo descubrió Tavis Ormandy, investigador de Google Project
Zero, en febrero de 2017, y le pusieron ese nombre por el parecido con Heartbleed.

**Qué pasó.** Un programa de Cloudflare que procesaba páginas HTML tenía un error: cuando el HTML
venía mal formado, el programa se pasaba del final del espacio de memoria que tenía asignado y
seguía leyendo lo que había después.

**Consecuencias.** Esa memoria de más terminaba **metida dentro de las páginas web** que los
usuarios recibían. Salían cookies, tokens de autenticación y datos de formularios de otros
usuarios. Y lo peor: como eran páginas web normales, los buscadores las guardaron en su caché, o
sea que información privada quedó indexada públicamente. Según Cloudflare, la fuga empezó en
septiembre de 2016 y duró hasta febrero de 2017, unos cinco meses.

**Cómo se arregló.** El error estaba en una comparación mal escrita: se verificaba si el puntero
era *igual* al final del espacio, cuando había que verificar si era *igual o mayor*. Por eso
cuando se lo pasaba de largo nunca se daba cuenta. Cloudflare desactivó las funciones afectadas
una hora después del reporte y después corrigió el código.

**Fuentes:**
- Cloudflare — Incident report on memory leak:
  https://blog.cloudflare.com/incident-report-on-memory-leak-caused-by-cloudflare-parser-bug/
- BitSight — Cloudbleed: Breakdown of Cloudflare's Memory Leak:
  https://bitsight.com/blog/cloudbleed-breakdown-of-cloudflares-memory-leak

### 6.3 Rover Spirit — NASA (2004)

**Quiénes estuvieron implicados.** La NASA y el Jet Propulsion Laboratory, con Pete Theisinger
como director del proyecto. El sistema operativo era VxWorks, de la empresa Wind River Systems.
La misión costó 820 millones de dólares.

**Qué pasó.** El 21 de enero de 2004, 17 días después de aterrizar en Marte, el rover Spirit dejó
de comunicarse con la Tierra. Se reiniciaba una y otra vez sin parar, y hasta había riesgo de que
se dañara el sistema de baterías por no apagarse en las noches.

El problema era de memoria: el rover tenía demasiados archivos guardados en su memoria flash,
muchos de ellos del viaje de siete meses desde la Tierra. Cuando arrancaba, el programa intentaba
reservar memoria RAM para cargar la lista de todos esos archivos, **no le alcanzaba**, el programa
se caía, y eso disparaba otro reinicio. Así quedó en un ciclo infinito.

**Consecuencias.** La misión estuvo diez días detenida y estuvo a punto de perderse.

**Cómo se arregló.** Existía un comando llamado `INIT_CRIPPLED` que permitía arrancar el
computador sin cargar el sistema de archivos. Lo había dejado puesto uno de los ingenieros años
antes, justamente por si algún día pasaba algo así. Con eso rompieron el ciclo de reinicios,
borraron archivos y después formatearon la memoria flash completa. El 6 de febrero el rover ya
funcionaba normal. Se pudo haber evitado verificando si la reserva de memoria había funcionado
antes de seguir, y borrando los archivos del viaje antes de aterrizar.

**Fuentes:**
- The Planetary Society — Spirit is "In Recovery":
  https://www.planetary.org/articles/0126_spirit_is_in_recovery
- Spaceflight Now: https://spaceflightnow.com/mars/mera/040126spirit.html

### 6.4 Lo que tienen en común

Los tres casos son el mismo error de fondo: **el programa confía en un número que no verificó**.
Heartbleed confía en el tamaño que le dice el cliente, Cloudbleed confía en que el puntero no se
va a pasar del límite, y Spirit confía en que la reserva de memoria siempre va a funcionar. En
C++ nada de eso se verifica solo; hay que escribir la comprobación.

---

## 7. Problemas que tuvimos y cómo los resolvimos

**1. El entrenamiento se iba a infinito.** Al entrenar el dataset de notas sin escalar los datos,
los pesos crecían hasta volverse números gigantes y el modelo no servía. El problema era que las
variables están en escalas muy distintas: las horas de estudio van de 1 a 12 pero la asistencia va
de 50 a 100, así que un paso del mismo tamaño en las dos es demasiado grande para una.
*Solución:* usar `data_scaling` para dejar todas las columnas en la misma escala antes de
entrenar.

**2. El conjunto de prueba daba resultados raros.** Al principio escalábamos el conjunto de prueba
calculándole sus propias medias y desviaciones, y los resultados no tenían sentido. El problema es
que así los dos conjuntos quedaban en escalas distintas y los pesos aprendidos ya no servían.
*Solución:* agregar el parámetro `guardar` a `data_scaling`, para que el conjunto de prueba use
las mismas medias y desviaciones del entrenamiento.

**3. Todas las temperaturas altas quedaban en el conjunto de prueba.** Dividíamos los datos
cortando por la mitad, pero el archivo de helados viene ordenado por temperatura, así que el
entrenamiento se quedaba solo con las temperaturas bajas. *Solución:* tomar una de cada cinco
muestras para prueba, en vez de cortar.

**4. Creímos que el código estaba mal por el R² tan bajo.** Con el dataset de helados nos daba
0.015 y revisamos todo buscando el error. *Solución:* mirar los datos en vez del código. Ahí nos
dimos cuenta de que la relación tiene forma de U y que el problema era el modelo, no el programa.

**5. `new RegresionLineal[5]` no compilaba.** Daba error porque el constructor pedía dos
parámetros y `new[]` crea los objetos sin pasarle ninguno. *Solución:* ponerle valores por defecto
al constructor.

**6. `sizeof` del arreglo dinámico daba 8.** Pensamos que algo estaba mal calculado. *Solución:*
entender que `sizeof` sobre un puntero mide el puntero, no lo que apunta, y calcular el tamaño
real multiplicando la cantidad de objetos por el tamaño de cada uno.

---

## 8. Conclusiones

**1. Una clase bien pensada sirve para más de un caso.** Al guardar los datos siempre como una
matriz, la misma clase resolvió la regresión simple y la múltiple sin cambiar nada. Si hubiéramos
escrito el código pensando solo en una variable, habríamos tenido que hacerlo todo de nuevo para
la múltiple.

**2. Escalar los datos no es opcional.** Sin escalar, el entrenamiento del dataset de notas
directamente no funcionaba. Con escalado, converge sin problemas. Además permite comparar los
pesos entre sí para saber qué variable influye más.

**3. Un mal resultado no siempre es culpa del código.** El R² de 0.015 del dataset de helados nos
hizo buscar errores en el programa durante un rato, cuando en realidad el programa estaba bien y
lo que no servía era el modelo lineal para esos datos. Distinguir las dos cosas fue lo que más nos
costó.

**4. `sizeof` no mide lo que uno cree que mide.** Mide el tamaño fijo del objeto en tiempo de
compilación, no la memoria que usa de verdad. Los vectores ocupan 24 bytes tengan 2 números o
1000, y un puntero ocupa 8 bytes apunte a lo que apunte.

**5. Cada zona de memoria funciona distinto.** Las variables globales viven todo el programa, las
locales se borran solas al terminar la función, y las dinámicas viven hasta que uno escriba
`delete`. Ver las direcciones reales impresas ayudó mucho más a entenderlo que verlo en un
diagrama.

**6. Olvidar verificar algo relacionado con la memoria tiene consecuencias serias.** Los tres
casos que investigamos —Heartbleed, Cloudbleed y el rover Spirit— pasaron por eso. Uno costó 500
millones de dólares, otro filtró datos privados durante cinco meses y el tercero casi pierde una
misión en Marte.
