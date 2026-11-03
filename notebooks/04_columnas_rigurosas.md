# Columnas rigurosas — DWSIM 10.2.12 vs FluxoEngine (Wang-Henke)

Semana 4 del plan: ¿dónde falla Fluxo en destilación rigurosa? Dos casos, diseñados para separar el error
del **algoritmo** del error de la **termodinámica**.

- **A. Depropanizadora** C2/C3/nC4/nC5 (0.05/0.45/0.40/0.10), 100 mol/s a 330 K y 17 bar, 22 etapas, alimentación
  en la 12, 16 → 17.5 bar, R = 1.84, D = 50 mol/s. **PR en ambos** → las diferencias son del algoritmo.
- **B. Etanol-agua** 50/50, 100 mol/s a 300 K y 1 atm, 20 etapas, alimentación en la 10, R = 2, B = 75 mol/s.
  DWSIM **NRTL** (verdad) vs DWSIM **PR** vs Fluxo **PR**: DWSIM-PR vs Fluxo-PR aísla el algoritmo; NRTL vs PR aísla
  lo que Fluxo pierde por no tener modelo de actividad.

Solver de DWSIM: `Wang-Henke (Bubble Point)` (el mismo método que Fluxo); tolerancias 1e-5.

**Hallazgo lateral**: `WithFeed(stream, n)` de la Fluent API cuenta etapas **desde 0** (`WithFeed(s, 2)` cae en
«Stage3»). Con la alimentación «en la 12» pasada tal cual, la desviación máxima del perfil era 3.8 K justo bajo el
plato de alimentación y la impureza clave difería 14 %; con el índice corregido (abajo) todo cae a < 1 K y ≈ 5 %.
Corregido en `tools.py` (feed_stage 1-based como en los libros), en el caso de referencia 9 y en las tarjetas.

| Caso | Variable | Fluxo | DWSIM | Desv. |
|---|---|---|---|---|
| A depro | D (mol/s) | 50  | 50  | +0.00 % |
| A depro | x_D Propane | 0.8797  | 0.8807  | -0.11 % |
| A depro | x_D N-butane | 0.02029  | 0.01933  | +4.97 % |
| A depro | x_B Propane | 0.02029  | 0.01933  | +4.98 % |
| A depro | x_B N-butane | 0.7797  | 0.7807  | -0.12 % |
| A depro | T destilado (K) | 311.4  | 311.3  | +0.04 % |
| A depro | T fondo (K) | 386.3  | 385.4  | +0.25 % |
| A depro | Qc (kW) | -1919  | -1917  | +0.08 % |
| A depro | Qr (kW) | 2313  | 2299  | +0.59 % |
| A depro | ΔT máx. del perfil (etapa 7) | 335.1 K | 334.1 K | +0.31 % |
| A depro | Δx máx. Propane líquido (etapa 6) | 0.6849  | 0.702  | -2.43 % |
| A depro | iteraciones | 46 | — | — |

## B en Fluxo: Wang-Henke **no converge** con PR

Fluxo no llegó a 2000 iteraciones con amortiguamiento 1, 0.5, 0.25 ni 0.1 (error de equilibrio 0.07–0.84).
La causa es termodinámica, no del algoritmo: la curva de burbuja a 1 atm que da el PR de Fluxo (k_ij = 0) para
etanol-agua no es monótona y el vapor sale **más pobre** que el líquido en el centro (LLE espurio del cúbico
con reglas de mezclado clásicas en una mezcla polar). Con esas K el punto de burbuja oscila.

| x etanol | T burbuja Fluxo-PR (K) | y etanol |
|---|---|---|
| 0.0 | 374.52 | 0.0000 |
| 0.1 | 328.15 | 0.8574 |
| 0.2 | 341.13 | 0.7112 |
| 0.3 | 345.66 | 0.6137 |
| 0.4 | 346.87 | 0.5679 |
| 0.5 | 346.99 | 0.5600 |
| 0.6 | 346.97 | 0.5817 |
| 0.7 | 347.23 | 0.6310 |
| 0.8 | 348.00 | 0.7109 |
| 0.9 | 349.45 | 0.8291 |
| 1.0 | 351.70 | 1.0000 |

Referencia: etanol puro hierve a 351.4 K y agua a 373.1 K; el azeótropo real está en x ≈ 0.89, 351.3 K (NRTL
en `02_nrtl_etanol_agua.py`). **Conclusión para Fluxo 1.1/1.2: hace falta NRTL (E4) para cualquier mezcla polar;
mientras tanto la app debe avisar que PR/SRK no son válidos para agua + alcoholes/ácidos.**

## B en DWSIM: NRTL converge en 2.3 s; **PR tampoco converge**

La misma columna con `PengRobinson` en DWSIM 10.2.12 (Wang-Henke, 100 iteraciones máx.) **no termina en 300 s**
(matada; > 6 min de CPU), mientras que con NRTL resuelve en 2.3 s y la depropanizadora PR en 2.2 s.
Es decir: el mismo algoritmo, en el simulador de referencia, falla con el mismo paquete en la misma mezcla. El
problema es la **ecuación de estado para una mezcla polar**, no Wang-Henke ni Fluxo.

Con `WithMaxIterations(30)` DWSIM termina y lanza `Solver reached the maximum number of iterations without
converging` desde `Column.SolveBubblePointRetry` (RigorousColumn.vb:6818): el solver reintenta con otras
estrategias y con el tope por defecto (100) ese bucle de reintentos es lo que pasa de 300 s. Ninguno de los dos
motores entrega una solución falsa: DWSIM lanza excepción y Fluxo deja la columna en rojo con el error de equilibrio.

Referencia NRTL (DWSIM): x_D etanol 0.8040, x_B 0.3987, T 351.59 → 353.73 K, Qc 2955.9 kW, Qr 3488.9 kW.

## Perfiles de temperatura por etapa (K)

| Etapa | A Fluxo | A DWSIM | B DWSIM-NRTL |
|---|---|---|---|
| 1 | 311.37 | 311.25 | 351.59 |
| 2 | 318.12 | 317.93 | 351.62 |
| 3 | 321.67 | 321.31 | 351.67 |
| 4 | 324.77 | 324.16 | 351.72 |
| 5 | 328.14 | 327.29 | 351.78 |
| 6 | 331.70 | 330.69 | 351.86 |
| 7 | 335.13 | 334.09 | 351.96 |
| 8 | 338.13 | 337.19 | 352.10 |
| 9 | 340.62 | 339.83 | 352.31 |
| 10 | 342.70 | 342.08 | 352.67 |
| 11 | 344.68 | 344.11 | 352.67 |
| 12 | 346.96 | 346.26 | 352.67 |
| 13 | 351.33 | 350.57 | 352.67 |
| 14 | 355.75 | 354.98 | 352.67 |
| 15 | 360.36 | 359.66 | 352.67 |
| 16 | 365.00 | 364.40 | 352.67 |
| 17 | 369.36 | 368.88 | 352.67 |
| 18 | 373.20 | 372.84 | 352.71 |
| 19 | 376.51 | 376.20 | 352.86 |
| 20 | 379.46 | 379.12 | 353.73 |
| 21 | 382.50 | 381.97 |  |
| 22 | 386.35 | 385.39 |  |
