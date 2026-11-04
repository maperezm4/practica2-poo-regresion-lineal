# Guion del video — Práctica 2

**Duración: unos 8 minutos** (el máximo permitido es 10)
Miguel Ángel Pérez Marín y Julio Cesar Llanes Cuello. Los dos con la cara visible.

> Antes de grabar: tener abierto el código y una terminal en la carpeta del proyecto.
> No hace falta leer esto palabra por palabra, es una guía de qué decir en cada parte.

---

## 1. Presentación (0:00 – 0:40) — los dos

**Miguel:**
> Hola, soy Miguel Ángel Pérez Marín.

**Julio:**
> Y yo soy Julio Cesar Llanes Cuello. Les vamos a mostrar la Práctica 2 de Lenguajes y
> Paradigmas, sobre programación orientada a objetos.

**Miguel:**
> Lo que hicimos fue una clase en C++ que calcula regresiones lineales, sin usar librerías
> externas. Está todo en un solo archivo que tiene las dos partes: el reto de POO y el análisis
> de memoria.

---

## 2. Qué es la regresión lineal (0:40 – 1:30) — Julio

**Julio:**
> Primero, qué estamos calculando. La regresión lineal es encontrar la recta que mejor pasa por
> una nube de puntos.
>
> Por ejemplo: temperatura contra ventas de helado. Si uno tiene esos datos, quiere encontrar la
> recta que diga cuántos helados se venden según la temperatura. Esa recta es `ventas = peso por
> temperatura, más el bias`. El peso es la pendiente y el bias es dónde cruza el eje.
>
> Cuando hay varias variables es lo mismo pero con más pesos. En el otro dataset tenemos cuatro:
> horas de estudio, horas de sueño, asistencia y notas previas, para predecir la nota del examen.
> Un peso por cada variable.
>
> Y acá está la decisión de diseño más importante: como la fórmula es la misma y solo cambia
> cuántos pesos hay, **una sola clase nos sirve para los dos casos**. La regresión simple es
> simplemente el caso en que hay una sola variable.

---

## 3. La clase y sus métodos (1:30 – 3:30) — Miguel
*Mostrar el código en pantalla mientras se explica.*

**Miguel:**
> Esta es la clase `RegresionLineal`. Tiene los dos atributos que pide el enunciado: `pesos`,
> que es el arreglo de pesos, y `bias`.
>
> Los dos son **privados**. Desde afuera solo se pueden leer con `getPesos` y `getBias`, no se
> pueden modificar. Eso es encapsulamiento: nadie puede dañar el estado del modelo desde fuera de
> la clase.
>
> **`fit`** es el que entrena. Empieza con todos los pesos en cero y los va corrigiendo de a
> poquito. En cada vuelta recorre todas las muestras, calcula qué predice el modelo con los pesos
> que tiene ahorita, compara con el valor real y acumula la diferencia. Al final de la vuelta
> mueve los pesos en dirección contraria al error. Eso lo repite cinco mil veces.
>
> La forma fácil de verlo: uno está en una montaña con mucha niebla y quiere bajar. No ve el
> valle, pero sí siente hacia dónde baja el piso bajo los pies, entonces da un paso en esa
> dirección y repite.
>
> **`predict`** usa el modelo ya entrenado: multiplica cada variable por su peso, suma todo, y le
> agrega el bias.
>
> **`score`** calcula el error de las predicciones. Es el error cuadrático medio: la diferencia
> entre lo que predice y lo real, elevada al cuadrado, promediada. Entre más bajo mejor.
>
> También hicimos **`r2`**, que es más fácil de interpretar: va de cero a uno, donde uno sería un
> ajuste perfecto.
>
> **`data_scaling`** pone todas las columnas en la misma escala. Julio explica por qué lo
> necesitamos.

---

## 4. Por qué hay que escalar (3:30 – 4:20) — Julio

**Julio:**
> Esto fue el primer problema serio que tuvimos. Cuando entrenamos el dataset de notas sin
> escalar, los pesos se volvían números gigantes y el modelo no servía para nada.
>
> La razón es que las variables están en escalas muy distintas: las horas de estudio van de 1 a
> 12, pero la asistencia va de 50 a 100. Como el algoritmo da pasos del mismo tamaño para todas,
> un paso que es razonable para las horas es enorme para la asistencia, y todo se descontrola.
>
> Escalar es restar la media y dividir por la desviación, así todas quedan en el mismo rango.
>
> Y hay un detalle en el método: el parámetro `guardar`. Cuando entrenamos le pasamos `true` y
> calcula las medias y desviaciones. Para el conjunto de prueba le pasamos `false`, y entonces
> reutiliza las que ya calculó. Eso es importante porque los dos conjuntos tienen que quedar en la
> misma escala; si a cada uno le calculáramos las suyas, los pesos aprendidos ya no servirían.

---

## 5. Ejecución (4:20 – 5:40) — Miguel
*Compilar y correr el programa en vivo.*

**Miguel:**
> Compilamos con `g++ practica2.cpp -o practica2` y lo corremos.
>
> El primer dataset es el de los helados. Y acá pasó algo interesante: miren el R cuadrado, da
> **cero punto cero uno**. O sea, el modelo no explica nada.
>
> Lo primero que pensamos fue que teníamos un error en el código. Pero cuando fuimos a mirar los
> datos nos dimos cuenta de que el problema era otro: **este dataset no es lineal**. Las ventas
> son altas cuando hace mucho frío y cuando hace calor, y caen casi a cero alrededor de los cero
> grados. Los puntos forman una U, no una recta.
>
> Y la mejor recta que uno puede trazar sobre una U es una recta casi horizontal. De hecho, el
> bias que encontró, dieciséis punto dos, es casi exactamente el promedio de las ventas. El
> modelo lo que hizo fue predecir el promedio siempre, porque no podía hacer nada mejor.
>
> La conclusión es que un resultado malo no siempre significa que el código esté mal. Acá el
> código estaba bien; lo que no servía era el modelo para esos datos.
>
> El segundo dataset sí funciona: doscientas muestras, cuatro variables, y el R cuadrado da
> **cero punto ochenta y cuatro**. El modelo explica el ochenta y cuatro por ciento de los datos.
>
> Y como los datos están escalados, los pesos se pueden comparar: las **horas de estudio** son la
> variable que más influye, con cuatro punto nueve. Después las notas previas con dos punto nueve.
> La asistencia y las horas de sueño quedan de últimas, casi empatadas. Sin escalar no podríamos
> comparar así, porque cada peso estaría en unidades distintas.

---

## 6. Análisis de memoria (5:40 – 7:20) — Julio
*Mostrar la segunda parte de la salida.*

**Julio:**
> La segunda parte del enunciado es el análisis de memoria. Creamos objetos de la misma clase en
> las tres zonas e imprimimos sus direcciones.
>
> El objeto **global** se crea antes del `main` y vive hasta que el programa termina. Los objetos
> **locales** están en direcciones mucho más altas, que empiezan en `0x7ffe`, y se borran solos
> cuando la función se acaba. El objeto **dinámico**, el que creamos con `new`, está en una
> dirección intermedia y vive hasta que uno escriba `delete`.
>
> Un detalle que vale la pena: el puntero y el objeto son dos cosas distintas. El puntero está en
> el stack y ocupa ocho bytes, porque lo único que guarda es una dirección. El objeto está en otro
> lado completamente.
>
> **Ahora el tamaño**, que es lo que pide explicar el enunciado. El objeto pesa noventa y seis
> bytes. Sumando los atributos a mano dan noventa y dos. ¿De dónde salen los cuatro que faltan?
>
> Primero, cada `vector` ocupa **veinticuatro bytes fijos**, tenga dos números o mil. Por dentro
> un vector son tres punteros: dónde empiezan los datos, dónde terminan y hasta dónde llega la
> memoria reservada. Los números en sí no están dentro del objeto, están en otra parte.
>
> Segundo, el compilador rellena huecos para que todo quede alineado de a ocho bytes, porque el
> procesador lee más rápido así. Noventa y dos se rellena hasta noventa y seis.
>
> Y lo más importante: **`sizeof` se calcula cuando se compila**, así que mide el tamaño fijo del
> objeto, no la memoria que va a usar después mientras corre.
>
> El arreglo dinámico muestra lo mismo. `sizeof` del arreglo da **ocho bytes**, y eso confunde al
> principio. Da ocho porque `arreglo` es un puntero. El tamaño real son cinco por noventa y seis,
> cuatrocientos ochenta bytes. Y se puede comprobar: entre un elemento y el siguiente hay
> exactamente noventa y seis bytes, o sea que los objetos están pegados uno tras otro.
>
> Al final liberamos todo: cada `new` con su `delete` y cada `new[]` con su `delete[]`.

---

## 7. Fallos históricos y cierre (7:20 – 8:00) — los dos

**Miguel:**
> El enunciado pedía investigar tres fallos famosos causados por mal manejo de memoria.
>
> **Heartbleed**, de 2014: el servidor reservaba memoria del tamaño que le decía el cliente, sin
> verificar que fuera cierto. Eso permitía leer sesenta y cuatro kilobytes de memoria ajena por
> pedido, incluidas contraseñas y claves privadas. Afectó a casi todo internet y costó unos
> quinientos millones de dólares.
>
> **Cloudbleed**, de 2017: un programa de Cloudflare se pasaba del final de un espacio de memoria
> y lo que leía de más terminaba metido dentro de las páginas web de los usuarios. Datos privados
> filtrados durante cinco meses.

**Julio:**
> Y el **rover Spirit** de la NASA, en 2004. Se quedó reiniciándose sin parar en Marte porque al
> arrancar no le alcanzaba la memoria RAM para cargar la lista de archivos que tenía guardados.
> El programa se caía, eso disparaba otro reinicio, y así infinitamente. Una misión de ochocientos
> veinte millones de dólares que casi se pierde.
>
> Los tres son el mismo error de fondo: **el programa confía en un número que no verificó**. Y en
> C++ eso no se verifica solo, uno tiene que escribir la comprobación.

**Miguel:**
> Esa fue la práctica. Gracias.

---

## Notas para grabar

- Repartido parejo: Miguel habla unos cuatro minutos y Julio otros cuatro.
- Grabar la terminal en vivo, no pegar capturas.
- Las dos caras visibles todo el video, es requisito de la rúbrica.
- Subir a YouTube como **No listado** (no privado, o el profesor no puede abrirlo).
- El profesor puede preguntarle a cualquiera de los dos sobre cualquier parte, así que conviene
  que cada uno entienda también lo del otro.
- Si se pasan de tiempo, lo más fácil de recortar es la parte 2 (qué es la regresión lineal).
