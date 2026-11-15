# Guía: hacer un diagnóstico de vapor de principio a fin

Tres pasos. No hace falta programar ni encadenar herramientas.

## 1. Mandar la hoja a la planta

```bash
./vapor plantilla datos_planta.xlsx
```

También hay una copia lista en `ejemplos/plantilla_diagnostico_vapor.xlsx`. La planta llena la hoja «Datos» en sus unidades
(°C, kg/cm² del manómetro, t/h, caballos caldera, pesos). Lo que no sepa lo deja vacío. Las hojas «Fugas y trampas» y
«Tubería sin aislar» son opcionales y salen de un recorrido de la planta.

## 2. Correr el diagnóstico

```bash
./vapor diagnostico datos_planta.xlsx --salida reporte_planta
```

En la carpeta quedan `reporte.html` (una sola página, se puede mandar por correo), `reporte.pdf` (si el equipo tiene
Chrome), `reporte.md` y `resultado.json`.

## 3. Leer lo que dice la terminal antes de entregar

- **Errores de captura** (unidad fuera de lista, texto donde va un número): se listan todos con su celda y no se calcula.
- **Datos esenciales que faltan**: el motor se niega y dice cuáles son y para qué sirven.
- **Valores supuestos**: lo que no vino de la planta. Salen marcados en el reporte.
- **Datos por confirmar**: datos de la planta que parecen fuera de lo normal. El cálculo se hace, pero hay que confirmarlos.
- **Medidas no evaluadas**: les faltó el objetivo.

## Cuando la planta no tiene un dato

| No tiene | Qué hace en la hoja | Qué usa el diagnóstico |
|---|---|---|
| Análisis del gas natural | Deja vacía la composición y da el recibo con energía y volumen | Composición ajustada al poder calorífico del recibo |
| Pérdida por paredes de la caldera | Marca «No la conozco» | 1.0 % a plena carga, de una guía del Departamento de Energía de EE. UU. |
| Objetivo de oxígeno o de chimenea | Marca «usar el publicado» | 10 % de exceso de aire; chimenea a 121 °C (solo gas natural) |
| Barómetro | Da la altitud | Atmósfera estándar |
| Cotización de una medida | La deja vacía | Da el ahorro y el tope de inversión; no estima el retorno |

Todo lo demás se pide. Los detalles y las fuentes están en `docs/DATOS_PARA_DIAGNOSTICO.md` y en `MOCKS.md`.

## Ver un ejemplo completo

```bash
./vapor ejemplo carpeta_de_prueba
```

Escribe una hoja llena con una planta **inventada** y su reporte. Hay una copia en `ejemplos/diagnostico_vapor/`.

## Lo que no hace

- No mide nada: usa los datos que entrega la planta.
- Una caldera por hoja, y un solo punto de operación promedio.
- No estima inversiones ni costos de reparación.
- No modela el punto de rocío ácido, la química del agua ni la carga variable.
- No sustituye una auditoría en sitio ni al proveedor de la caldera.

Desde el agente o la app, las mismas funciones son las herramientas `steam_workbook_template` y
`steam_diagnostic_from_workbook`.
