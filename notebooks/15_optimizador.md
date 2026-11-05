# 15 · Optimizador genérico `optimize_design` (bloque 21, 2026-11-05)

**Qué hace.** En una sola llamada optimiza una variable de diseño: un parámetro de equipo o una especificación de
corriente, entre dos límites. En cada punto resuelve DWSIM. La función objetivo puede ser:
- el costo total anual de `flowsheet_cost`;
- una suma ponderada de lecturas, por ejemplo la potencia de dos compresores.

Las restricciones son límites duros, típicamente del HAZOP: temperatura máxima de descarga, aproximación mínima o presión
máxima. Un punto que viola un límite, o que no cierra balances, es infactible. El método barre en escala lineal o
logarítmica y luego refina con sección dorada. Devuelve:
- la curva completa como evidencia;
- la restricción activa, si el óptimo está sobre un límite;
- el rango plano, los valores dentro de 1 % del óptimo, donde la decisión no es sensible.

Al terminar deja el flowsheet resuelto en el óptimo.

Para que el costo funcione dentro de una optimización, `flowsheet_cost` ahora acepta `exchanger_specs`. Con U y las
temperaturas de la utilidad, dimensiona cada calentador, enfriador e intercambiador a partir de la carga resuelta en
cada punto. 30 herramientas en total.

## A · Recuperación de calor rehecha solo con herramientas del agente

| | `optimize_design` | Lazo a mano del notebook 14 |
|---|---|---|
| Área de E-1 | 650.4 m² | 650.3 m² |
| Aproximación | 7.12 K | 7.12 K |
| Costo total anual | 129 992 USD/año | 129 992 USD/año |
| Rango plano (±1 %) | 598-742 m² | — |
| Evaluaciones | 27 | 30 |

Usa el intercambiador estándar de `add_unit` con UA. Coincide con el modo CalcBothTemp_UA del notebook 14.

## B · Compresión de aire en dos etapas, de 1 a 9 bar, con interenfriamiento a 300 K y η = 75 %

| Restricción | P intermedia (bar) | Potencia total (kW) | T descarga etapa 1 / 2 (K) | Restricción activa |
|---|---|---|---|---|
| Ninguna | **3.007** (teoría de gas ideal √(1·9) = 3.0) | 594.53 | 446.3 / 445.7 | — |
| Etapa 1 ≤ 440 K (HAZOP) | **2.885** | 594.67 (+0.15) | 440.0 / 452.1 | etapa 1 |
| Ambas etapas ≤ 440 K | infactible | — | — | ninguna presión intermedia sirve: hacen falta tres etapas |

El límite del HAZOP mueve el óptimo y cuesta solo 0.15 kW. El optimizador también reconoce cuando el problema no tiene
solución factible, en vez de devolver un número.

Reproducir: `PYTHONPATH=. uv run python notebooks/15_optimizador.py A` (o `B`). Salidas en `notebooks/15_optimizador_*.jsonl`.

## Límites
- Optimiza una sola variable. Varias variables requieren otro método, como búsqueda por coordenadas o bayesiana.
- Supone un óptimo unimodal dentro del intervalo refinado; el barrido inicial reduce el riesgo, no lo elimina.
- Las restricciones se evalúan solo en los puntos visitados; el óptimo con restricción activa queda a la precisión de la sección dorada.

## El agente con la herramienta nueva (Opus, `claude -p`, 30 tools)

| Tarjeta | Resultado | Tiempo | Antes (bloque 20, sin optimizador) |
|---|---|---|---|
| Compresión en dos etapas con límite del HAZOP | **1.00**: 2.885 bar, 594.67 kW, 452.05 K | 45 s | — |
| Área óptima de recuperación de calor | **1.00**: 648.9 m², 7.01 K, 130 595 USD/año | 111 s | 188 s y 49 llamadas |

Entre las dos tarjetas el agente hizo 48 llamadas, 3 de ellas a `optimize_design`. Traza en
`notebooks/bench_runs/v3_optimizador_cc_opus_run1.mcp_trace.jsonl`.
