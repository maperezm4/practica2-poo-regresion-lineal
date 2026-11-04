# Práctica 2 — Programación Orientada a Objetos

Regresión lineal simple y múltiple en C++, sin librerías externas.

**Lenguajes y Paradigmas de la Computación — EAFIT**

- Miguel Ángel Pérez Marín — 1018242716 — maperezm4@eafit.edu.co
- Julio Cesar Llanes Cuello — 1118835249 — jcllanesc@eafit.edu.co

## Compilar y ejecutar

```bash
g++ practica2.cpp -o practica2
./practica2
```

En Windows con MinGW es lo mismo, cambiando `./practica2` por `practica2.exe`.

## Archivos

| Archivo | Qué es |
|---|---|
| `practica2.cpp` | Todo el código: la clase, el reto de POO y el análisis de memoria |
| `data/` | Los dos datasets en formato CSV |
| `evidencias/` | La salida del programa |
| `INFORME.md` | El informe de la práctica |

## Qué hace el programa

**Parte 1 — Reto POO.** Entrena un modelo de regresión lineal con dos datasets: uno de venta de
helados (una variable) y otro de notas de estudiantes (cuatro variables). Muestra los pesos, el
bias, las predicciones y el error.

**Parte 2 — Análisis de memoria.** Crea objetos de la misma clase en memoria global, local y
dinámica, muestra sus direcciones y tamaños, y explica de dónde sale cada valor.

## Los datos

Los archivos CSV van en `data/`. El programa asume que **la última columna es el valor a
predecir** y que la primera línea tiene los nombres de las columnas.
