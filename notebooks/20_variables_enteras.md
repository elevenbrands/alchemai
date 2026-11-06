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
| 18 | 2.333 | 1 065 894 |
| 26 | 1.495 | 896 523 |
| 27 | 1.439 | 887 346 |
| 28 | 1.439 | 891 655 |
| **29** | **1.394** | **885 080** |
| 30 | 1.394 | 889 338 |
| 31 | 1.359 | 885 126 |
| 38 | 1.284 | 896 069 |
| 48 | 1.207 | 917 553 |
| 60 | 1.169 | 954 586 |

**Re-corrida 2026-11-06 con la alimentación fija a la mitad.** 442 soluciones de DWSIM en 68 s. El óptimo está en 29 etapas
con R = 1.394 (885 080 USD/año); 31 etapas quedan a 0.005 %. La curva tiene dientes: con N par la etapa de alimentación
redondeada cae un poco más abajo y pide más reflujo, así que N impar gana. Es un efecto real de fijar la alimentación «a la mitad»,
no ruido del optimizador.

**Corrección.** La primera versión de este cuaderno (2026-11-05) reportó 32 etapas, R 1.267 y 866 992 USD/año. Ese óptimo no
era el del problema planteado: al reconstruir la columna en cada cambio de etapas, la alimentación se recalculaba desde la anterior
ya redondeada y se desplazaba. Al final la columna de 32 etapas tenía la alimentación en la 21 en lugar de la 16, y resolverla desde
cero daba x_D 0.796 < 0.80. Corregido en `tools_sweep._rebuild_column` (la posición relativa se guarda una vez). Lección: un óptimo
reportado se re-verifica en un flowsheet nuevo antes de publicarlo.

**Contra la teoría.** R_min riguroso (80 etapas, bisección sobre la columna resuelta) = 1.134 → **R_opt/R_min = 1.23**,
dentro del rango heurístico 1.1-1.5 del óptimo económico (Seader, Henley y Roper). El R_min de FUG para el mismo corte
sale 0.319 (R/R_min = 4.4): Underwood con volatilidad relativa constante no sirve para etanol-agua cerca del azeótropo.
Es un hallazgo útil para el agente: el atajo FUG se puede usar para N_min pero no para R_min en sistemas no ideales.

**Verificación.** `tests/test_optimize.py::test_integer_stages_rebuild_and_enumeration` (reconstrucción con etapa de
alimentación escalada; con pureza fija, el reflujo necesario baja al subir N), `tests/test_optimize.py::test_feed_stage_does_not_drift_over_rebuilds`, `tests/test_costing.py::test_vessel_trays_match_seider_formula`.
Tarjeta bench v3 `optimo_etapas_reflujo` (oráculo corregido 29 / 1.394 / 885 080; tolerancias N ±3, R 8 %, costo 1 %).

**Límites.** Diámetro fijo (no se dimensiona por hidráulica de platos a cada N); la etapa de alimentación se escala, no
se optimiza; sin costo de internos de relleno más allá del volumen × precio; el método entero es enumerativo (bien para
1-2 enteras con rangos de decenas, no para MINLP grandes).
