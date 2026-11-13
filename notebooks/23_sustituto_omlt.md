# 23 · Sustituto de red neuronal sobre DWSIM, embebido con OMLT (bloque M8c)

Tool `optimize_design_surrogate` (`packages/eo/surrogate.py`). Recibe un flowsheet de DWSIM ya armado, las variables
con las cotas del usuario, el objetivo y las restricciones con la misma forma que `optimize_design_multi`. Hace seis pasos:
1. Muestrea DWSIM: hipercubo latino (scipy.stats.qmc, semilla fija) más las 2^n esquinas, y un hipercubo latino aparte
   para el conjunto apartado.
2. Entrena una red tanh pequeña por salida en numpy (L-BFGS de scipy). La arquitectura y la penalización L2 se eligen
   por validación cruzada de 5 partes, solo con el entrenamiento.
3. Embebe cada red en Pyomo con OMLT 1.2.2 (`NetworkDefinition` + `DenseLayer`, `ReducedSpaceSmoothNNFormulation`).
4. Resuelve con ipopt desde 6 arranques: las 5 mejores muestras factibles y el centro.
5. Corre DWSIM en el óptimo propuesto.
6. Sella el resultado solo si DWSIM lo confirma.

**Por qué esta formulación.** Con una red suave (tanh), las salidas son expresiones cerradas de las n variables:
- ipopt recibe derivadas exactas.
- No hay que inicializar neuronas ocultas.
- Con decenas de neuronas, el espacio completo solo agrega variables.

ReLU + MILP (cbc viene gratis con idaes-ext) daría el óptimo global de un sustituto lineal a trozos, pero ReLU pide
muchas más corridas de DWSIM para la misma precisión. Los arranques múltiples cubren la búsqueda global.

**Dependencias:** ninguna nueva. sklearn, torch y keras no están en el entorno y no hacen falta para redes de 25 a 105 pesos.

**Sello `verified`** (las seis condiciones):
- error apartado ≤ tolerancia;
- OMLT = la red de numpy (≤ 1e-6);
- DWSIM resuelve en el óptimo;
- error en el óptimo ≤ tolerancia;
- DWSIM cumple los límites dentro de la tolerancia;
- ninguna muestra de DWSIM factible es mejor que el valor de DWSIM en el óptimo.

Si falta cualquiera, el resultado sale `not_verified` y la nota pide no usar el número del sustituto.

**Tolerancia por defecto:** 0.5 % de la mediana del objetivo y de cada límite. Cada restricción puede traer su propia
`tolerance` absoluta: en las purezas se usó 0.002.

**Muestras por defecto:** 20 + 15·n más las esquinas, y max(10, n/4) apartadas. Es un supuesto del método y se declara.

## A · Compresión de aire en tres etapas, 1 → 27 bar (flowsheet del notebook 18, PR, η 75 %)

| Referencia | P1 (bar) | P2 (bar) | Potencia (kW) |
|---|---|---|---|
| **Sustituto (OMLT + ipopt)** | **3.0261** | **9.0755** | 892.074 (sustituto) |
| DWSIM en el óptimo del sustituto | — | — | **891.838** |
| `optimize_design_multi` directo sobre DWSIM | 3.0211 | 9.0867 | 891.837 |
| Malla densa de DWSIM 21×21 (paso 0.0225/0.07 bar) | 3.025 | 9.100 | 891.837 |
| Teoría de gas ideal (razones iguales) | 3 | 9 | — |

- **Error apartado** (12 puntos): máximo 0.34 kW, que es el 0.037 % (tolerancia 4.65 kW). El error en el óptimo es 0.24 kW (0.026 %).
- **Ubicación:** a 0.17 % y 0.12 % del óptimo directo. La malla muestra que el objetivo es plano: ±0.05 kW abarca
  P1 2.96-3.09 bar y P2 8.89-9.31 bar.
- **Costo:** 67 corridas de DWSIM (54 de entrenamiento, 12 apartadas y 1 de verificación) y 7 s.

## B · Igual, con el límite del HAZOP «descarga de la etapa 1 ≤ 430 K»

| Referencia | P1 (bar) | P2 (bar) | Potencia (kW) | T descarga 1 (K) |
|---|---|---|---|---|
| **Sustituto** | **2.6997** | **8.5792** | 892.903 | 430.000 |
| DWSIM en ese punto | — | — | **892.674** | 430.026 |
| `optimize_design_multi` directo | 2.6992 | 8.5908 | 892.676 | 430.000 |

- **Error apartado de T:** 0.075 K (tolerancia 2.15 K).
- **DWSIM excede el límite en 0.026 K:** cae dentro de la tolerancia, y el resultado lo reporta en `dwsim_constraint_violation`.

## C · Columna hexano-heptano (caso de M8b: 100 mol/s 50/50 a 355 K, Raoult, 12 etapas, alimentación en la 6)

Se minimiza la carga del rehervidor sobre RR ∈ [1.5, 4] y fondos ∈ [45, 55] mol/s, con ambas purezas ≥ 0.95. La
tolerancia de 0.002 en las purezas es un supuesto de la prueba (MOCKS.md).

| Referencia | RR | Fondos (mol/s) | Q_reb (kW) | x_D hexano / x_B heptano |
|---|---|---|---|---|
| **Sustituto** | **2.2314** | **50.001** | 4107.20 (sustituto) | 0.9500 / 0.9500 |
| DWSIM en ese punto | — | — | **4106.91** | 0.949992 / 0.949986 |
| `optimize_design_multi` directo sobre la misma columna | 2.2322 | 50.001 | 4107.94 | 0.95002 / 0.95001 |
| `eo_column_optimize` (IDAES, datos de `chemicals`) | 2.291 | 50.0 | 4164 (IDAES); DWSIM en ese punto 4187 | 0.952 / 0.951 en DWSIM |

- **Error apartado:** purezas 0.00086 y 0.00035 (tolerancia 0.002); Q_reb 4.19 kW (0.13 %).
- **Error en el óptimo:** Q_reb 0.29 kW; purezas 8e-6 y 1.4e-5.
- **Redes elegidas por validación cruzada:** [6] para Q_reb, [10] para x_D y [8, 8] para x_B.
- **Respecto al óptimo directo de DWSIM:** el sustituto está a 0.04 % en RR.
- **Respecto a la columna orientada a ecuaciones:** el sustituto está a 2.6 % en RR. La diferencia es de bases de
  propiedades: con el RR de IDAES, DWSIM sobrepurifica (0.952/0.951).

## Límites

- Solo variables continuas (1 a 6). Las enteras van a `optimize_design_multi`.
- El sustituto vale solo dentro de las cotas y donde DWSIM resuelve.
- ipopt da óptimos locales, aunque con arranques múltiples.
- **En estos casos DWSIM es rápido** (0.05-0.07 s por corrida), así que la optimización directa fue más rápida
  (0.7-1.8 s contra 6-9 s). El sustituto paga cuando cada corrida es lenta, cuando hay más variables, o cuando hace
  falta un modelo algebraico suave del flowsheet para meterlo en un NLP mayor.
- **No hay refinamiento adaptativo:** si sale `not_verified`, hay que volver a llamar con más muestras o con cotas más
  estrechas alrededor del óptimo.

Reproducir: `PYTHONPATH=. uv run python notebooks/23_sustituto_omlt.py A|B|C|grid|record`. La comparación lenta corre
con `ALCHEMAI_SLOW_SURROGATE=1 PYTHONPATH=. uv run pytest -q tests/test_surrogate.py`.
