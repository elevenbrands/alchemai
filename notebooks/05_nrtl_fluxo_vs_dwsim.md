# NRTL en FluxoEngine vs DWSIM 10 — etanol-agua (semana 5)

Objetivo del bloque 11: portar NRTL (E4) al motor de Fluxo y reproducir la columna etanol-agua que PR no podía
resolver (`04_columnas_rigurosas.md`). Fuente de verdad: DWSIM 10.2.12 con su paquete NRTL.

## Qué hace DWSIM-NRTL exactamente (volcado con `05_nrtl_dwsim_dump.py`)

- Vapor **ideal**: `DW_CalcFugCoeff(..., Vapor)` devuelve φ = 1 en toda la malla.
- Líquido: φ_i^L = γ_i·Psat_i/P, sin Poynting ni φ_sat.
- τ_ij = A_ij/(R·T) con A en cal/mol y R = 1.98721: etanol(1)-agua(2) A12 = -57.9601, A21 = 1241.7396, α = 0.2937
  (ajuste sobre 20 puntos de la malla: rms ln γ = 0.001; las otras formas de τ dan rms > 0.17).
- Psat DIPPR 101 propia (ChemSep): Tb etanol 351.81 K, agua 373.13 K.
  Perry's 8.ª (tabla 2-8, lo que lleva Fluxo): etanol 351.46 K, agua 373.17 K.
  El etanol de DWSIM hierve 0.33 K por encima del valor experimental (351.44 K); esa es la mayor fuente de
  diferencia entre «reproducir DWSIM» y «estar bien».

El conjunto NRTL de DWSIM tiene 894 pares; 16 caen entre los 24 componentes de Fluxo y quedaron transcritos en
`BaseDeComponentes.nrtl` (solo etanol-agua está validado; el resto se marca en MOCKS).

## Diseño en FluxoEngine

`PaqueteDeFluido.modeloDeLiquido: ModeloDeLiquido` (`.ecuacionDeEstado` por omisión, `.nrtl(parámetros, vapor:)`).
Con NRTL el líquido va por γ-φ y el vapor por la cúbica (`.cubica`, con φ_sat y Poynting para que un puro hierva
exactamente a su Psat) o por gas ideal (`.gasIdeal`, lo que hace DWSIM). `Componente.presionDeVapor` (DIPPR 101,
Perry's) alimenta Psat y ΔH_vap por Clapeyron; h^E y s^E salen de NRTL por derivada numérica. Flash, burbuja/rocío,
columna y balances de energía no cambiaron: solo piden `lnPhi` y `entalpiaRapida` al paquete.

## Txy a 1 atm

Reproducción (Psat de DWSIM + vapor ideal), 19 puntos x = 0.05…0.95: `TXY_NRTL_DWSIM maxΔT = 0.014309443241131703 K, maxΔy = 0.00024177482299791997`.

Datos propios (Perry's + vapor PR) frente a DWSIM:

| x etanol | T DWSIM (K) | T Fluxo (K) | ΔT (K) | y DWSIM | y Fluxo | Δy |
|---|---|---|---|---|---|---|
| 0.05 | 364.051 | 364.032 | -0.019 | 0.3146 | 0.3144 | -0.0002 |
| 0.10 | 359.826 | 359.739 | -0.087 | 0.4374 | 0.4382 | +0.0008 |
| 0.15 | 357.564 | 357.435 | -0.129 | 0.4998 | 0.5012 | +0.0014 |
| 0.20 | 356.213 | 356.058 | -0.156 | 0.5376 | 0.5393 | +0.0017 |
| 0.25 | 355.321 | 355.148 | -0.174 | 0.5638 | 0.5656 | +0.0019 |
| 0.30 | 354.673 | 354.485 | -0.187 | 0.5844 | 0.5863 | +0.0020 |
| 0.35 | 354.157 | 353.958 | -0.199 | 0.6024 | 0.6044 | +0.0020 |
| 0.40 | 353.718 | 353.509 | -0.209 | 0.6196 | 0.6217 | +0.0021 |
| 0.45 | 353.328 | 353.109 | -0.219 | 0.6371 | 0.6392 | +0.0021 |
| 0.50 | 352.973 | 352.745 | -0.228 | 0.6556 | 0.6578 | +0.0021 |
| 0.55 | 352.649 | 352.411 | -0.238 | 0.6757 | 0.6779 | +0.0021 |
| 0.60 | 352.357 | 352.109 | -0.249 | 0.6979 | 0.7000 | +0.0021 |
| 0.65 | 352.100 | 351.841 | -0.259 | 0.7224 | 0.7245 | +0.0021 |
| 0.70 | 351.883 | 351.613 | -0.270 | 0.7498 | 0.7518 | +0.0020 |
| 0.75 | 351.711 | 351.430 | -0.281 | 0.7803 | 0.7822 | +0.0019 |
| 0.80 | 351.593 | 351.300 | -0.292 | 0.8146 | 0.8163 | +0.0017 |
| 0.85 | 351.535 | 351.230 | -0.304 | 0.8530 | 0.8545 | +0.0014 |
| 0.90 | 351.545 | 351.228 | -0.317 | 0.8962 | 0.8973 | +0.0011 |
| 0.95 | 351.632 | 351.302 | -0.331 | 0.9449 | 0.9456 | +0.0006 |

Azeótropo Fluxo: x = 0.875, T = 351.22 K (experimental ≈ 0.894, 351.3 K). Las ΔT de −0.3 K en la zona rica
en etanol son la diferencia de Psat entre fuentes, no del modelo.

## Columna de 20 etapas (caso B de la semana 4)

| Variable | DWSIM NRTL | Fluxo NRTL (Psat DWSIM, vapor ideal) | Fluxo NRTL (Perry's, vapor PR) |
|---|---|---|---|
| x_D etanol | 0.8040 | 0.8043 | 0.8088 |
| x_B etanol | 0.3987 | 0.3986 | 0.3971 |
| T destilado (K) | 351.59 | 351.59 | 351.28 |
| T fondo (K) | 353.73 | 353.74 | 353.53 |
| Qc (kW) | 2955.9 | 3018.7 | 2961.3 |
| Qr (kW) | 3488.9 | 3534.4 | 3514.5 |
| iteraciones | — | 22 | 22 |

Perfil de T por etapa (K):

| Etapa | DWSIM | Fluxo (repro) | Fluxo (propio) |
|---|---|---|---|
| 1 | 351.59 | 351.59 | 351.28 |
| 2 | 351.62 | 351.62 | 351.32 |
| 3 | 351.67 | 351.67 | 351.37 |
| 4 | 351.72 | 351.72 | 351.42 |
| 5 | 351.78 | 351.78 | 351.49 |
| 6 | 351.86 | 351.86 | 351.57 |
| 7 | 351.96 | 351.96 | 351.68 |
| 8 | 352.10 | 352.10 | 351.82 |
| 9 | 352.31 | 352.31 | 352.04 |
| 10 | 352.67 | 352.67 | 352.42 |
| 11 | 352.67 | 352.67 | 352.42 |
| 12 | 352.67 | 352.67 | 352.42 |
| 13 | 352.67 | 352.67 | 352.42 |
| 14 | 352.67 | 352.67 | 352.42 |
| 15 | 352.67 | 352.67 | 352.42 |
| 16 | 352.67 | 352.67 | 352.42 |
| 17 | 352.67 | 352.68 | 352.43 |
| 18 | 352.71 | 352.71 | 352.46 |
| 19 | 352.86 | 352.86 | 352.62 |
| 20 | 353.73 | 353.74 | 353.53 |

## Conclusión

- Con los mismos datos que DWSIM, Fluxo reproduce la columna: x_D a 0.0004, temperaturas a 0.01 K, cargas a 2 %
  (la diferencia de cargas es el modelo de entalpía de líquido: Clapeyron + h^E en Fluxo vs el de DWSIM).
- Con datos propios la separación es ligeramente mejor (x_D 0.8088) porque el etanol de Perry's es 0.33 K más
  volátil que el de ChemSep; ambas corridas convergen en 22 iteraciones donde PR no convergía en 2000.
- Pendiente para Fluxo 1.1: exponer el modelo en la UI (Figma) y avisar cuando falten pares binarios.
