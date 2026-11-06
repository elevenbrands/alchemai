# 20 · Variables enteras: número de etapas × reflujo de costo mínimo (bloque 26, 2026-11-05)

**Pregunta.** ¿Puede el optimizador manejar una variable entera (etapas de una columna) junto con una continua
(reflujo), y el óptimo coincide con lo que dice la teoría económica de columnas?

**Qué se construyó.**
- `set_unit_param(..., "stages", n)` y `"feed_stage"` para columnas: DWSIM no deja renumerar las etapas de una columna
  conectada (la alimentación conserva su clave de etapa y `WithFeed` falla con «same key»), así que la columna se borra
  (`DeleteObject` del flowsheet real, no de la interfaz `IFlowsheet`) y se recrea con el mismo nombre, parámetros y
  conexiones; la etapa de alimentación conserva su posición relativa.
- `optimize_design_multi` acepta `"integer": true`: enumera cada valor entero (completo si son ≤ 12, si no ~6 valores y
  refinamiento ±2 alrededor del mejor hasta que quede interior) y optimiza las continuas en cada uno; devuelve
  `integer_curve` para ver qué tan plano es el óptimo.
- Costo de platos de Seider vía IDAES (`cost_vessel(number_of_trays=…)`): C_BT = 468·e^(0.1739·D_ft) por plato × F_NT
  (2.25/1.0414^N, mínimo 1) × F_TT × F_TM; reproducido a mano en dos casos (16 517 y 68 512 USD_CE500, exactos).
  Relleno: `cost_packing(volumen, precio_unitario)`; el precio unitario es entrada del usuario (no hay tabla pública verificada).
- `flowsheet_cost` costea columnas: coraza vertical con N−2 platos y altura (N−2)·espaciamiento + extra, o coraza + relleno;
  el condensador cuenta como enfriamiento y el rehervidor como calentamiento.

**Caso.** Columna etanol-agua de referencia (NRTL, 100 mol/s al 50 %, B = 75 mol/s), x_D(EtOH) ≥ 0.80. Variables: etapas
totales 8-60 (entera) y reflujo 1-5. Costo anual = capital anualizado (D = 1.2 m, platos perforados a 0.6 m, Lang 4.74,
10 %/10 años) + vapor 8 USD/GJ + agua 0.5 USD/GJ a 8000 h/año. Diámetro y precios ilustrativos (MOCKS.md).

## Resultados

| N | R mínimo que cumple x_D | Costo anual (USD/año) |
|---|---|---|
| 8 | — | infactible |
| 18 | 2.023 | 991 076 |
| 26 | 1.394 | 872 155 |
| 30 | 1.305 | 867 744 |
| **32** | **1.267** | **866 992** |
| 34 | 1.238 | 868 499 |
| 38 | 1.238 | 884 971 |
| 48 | 1.185 | 912 245 |
| 60 | 1.153 | 950 725 |

556 soluciones de DWSIM en 140 s. El óptimo está en 32 etapas con R = 1.267; entre 30 y 34 etapas el costo varía
menos de 0.2 % (óptimo plano, como describe Douglas).

**Contra la teoría.** R_min riguroso (80 etapas, bisección sobre la columna resuelta) = 1.134 → **R_opt/R_min = 1.12**,
dentro del rango heurístico 1.1-1.5 del óptimo económico (Seader, Henley y Roper). El R_min de FUG para el mismo corte
sale 0.319 (R/R_min = 3.98): Underwood con volatilidad relativa constante no sirve para etanol-agua cerca del azeótropo.
Es un hallazgo útil para el agente: el atajo FUG se puede usar para N_min pero no para R_min en sistemas no ideales.

**Verificación.** `tests/test_optimize.py::test_integer_stages_rebuild_and_enumeration` (reconstrucción con etapa de
alimentación escalada; con pureza fija, el reflujo necesario baja al subir N), `tests/test_costing.py::test_vessel_trays_match_seider_formula`.
Tarjeta bench v3 `optimo_etapas_reflujo` (oráculo 32 / 1.2665 / 866 992; tolerancias N ±3, R 8 %, costo 1 %).

**Límites.** Diámetro fijo (no se dimensiona por hidráulica de platos a cada N); la etapa de alimentación se escala, no
se optimiza; sin costo de internos de relleno más allá del volumen × precio; el método entero es enumerativo (bien para
1-2 enteras con rangos de decenas, no para MINLP grandes).
