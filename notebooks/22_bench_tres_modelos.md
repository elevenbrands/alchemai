# 22 · Bench completo con tres modelos (bloque 29, mes 2)

**Qué:** las 50 tarjetas del ChemE Design Bench (v0 20 · v1 10 trampas físicas · v1b 5 trampas de especificación · v2 4 polares · v3 11 transporte, costo y optimización) en una sola pasada por modelo, mismo agente `claude-code`, mismas 36 tools MCP, mismo prompt, timeout 600 s por tarjeta. Modelo por `ALCHEMAI_CC_MODEL`. 2026-11-06.

Comando: `ALCHEMAI_CC_MODEL=<m> PYTHONPATH=. uv run python packages/bench/runner.py --agent claude-code --out notebooks/bench_runs/m2/all_cc_<m>.jsonl` · resumen: `PYTHONPATH=. uv run python notebooks/22_bench_tres_modelos.py`.

## Resultado (una sola pasada, sin re-intentos)

| Modelo | Puntaje | v0 (20) | Trampas (15) | v2 (4) | v3 (11) | Tiempo total | Turnos | Costo reportado* |
|---|---|---|---|---|---|---|---|---|
| Haiku | **92.4 %** | 94 % | 15/15 | 100 % | 77 % | 3 665 s | 997 | 4.72 USD |
| Sonnet | **94.0 %** | 100 % | 14/15 | 100 % | 82 % | 2 755 s | 580 | 6.18 USD |
| Opus | **97.0 %** → **100 %** con las dos tarjetas corregidas | 95 % → 100 % | 15/15 | 100 % | 95 % → 100 % | 2 001 s | 604 | 8.28 USD |

\* Costo equivalente que reporta Claude Code; la corrida fue por suscripción, no se cobró por API.

## Fallos, uno por uno

| Modelo | Tarjeta | Puntaje | Qué pasó | Tipo |
|---|---|---|---|---|
| Opus | reciclo | 0 | Marcó la tarjeta como sub-especificada: al agua fresca le faltaban T y P (`dof_check`) y a 1 atm los 400 K deciden si el reciclo es vapor. **Tenía razón.** | Defecto del bench → corregido |
| Opus | optimo_etapas_reflujo | 0.5 → **1.0** | N 31, R 1.358, 884 934 USD/año. La tarjeta estaba mal: su óptimo (32, 1.267, 866 992) venía de un optimizador con la alimentación desplazada (ver abajo). Contra el óptimo corregido (29, 1.394, 885 080) la respuesta de Opus es correcta | Defecto del bench → corregido |
| Sonnet | trampa_azeotropo | 0 | Agotó los 600 s sin responder | Velocidad |
| Sonnet | optimo_etapas_reflujo | 0 | No devolvió JSON válido | Formato |
| Sonnet | optimo_recuperacion_calor | 0 | 317 m² y 385 k USD/año vs 650 m² y 130 k: **número equivocado reportado como verificado** | Agente, grave |
| Haiku | diseno_intercambiador_ua | 0 | Agotó los 600 s | Velocidad |
| Haiku | optimo_recuperacion_calor | 0 | No devolvió JSON válido | Formato |
| Haiku | columna_rigurosa | 0.7 | Qr 2 957 kW vs 3 489: equivocado reportado como verificado | Agente, grave |
| Haiku | optimo_compresion_tres_etapas | 0.5 | P1 2.649 vs 2.699 bar (fuera de 1 %) reportado como verificado | Agente |
| Haiku | optimo_etapas_reflujo | 0 | N 37 y R 1.25 contra 29 y 1.394 (re-calificada con la tarjeta corregida; el costo sí cae en 1 %): equivocado reportado como verificado | Agente, grave |

## Lectura

- **Las trampas aguantan en los tres.** Haiku y Opus 15/15; Sonnet falló una sola por tiempo, no por aceptar algo imposible. Ningún modelo reportó como verificado un caso infactible.
- **La diferencia está en v3** (optimización y costo), no en simulación: Haiku 77 %, Sonnet 82 %, Opus 95 %.
- **El fallo que importa es «equivocado reportado como verificado»**: Haiku 3 tarjetas, Sonnet 1, Opus ninguno una vez corregidas las tarjetas. Es lo que el bench castiga con 0 en la cantidad y lo que un ingeniero no detectaría sin re-simular.
- **Opus es el más rápido** (2 001 s, 40 s/tarjeta) aunque el más caro por token; Haiku usa casi el doble de turnos.
- **El agente encontró un defecto del bench**: la tarjeta `reciclo` llevaba desde v0 sin T ni P del agua fresca (la referencia usa 300 K y 1 atm). Se corrigió el enunciado en `cases_ref.py` y en la tarjeta; la re-corrida de esa tarjeta dio 100 % con los tres modelos (`reciclo_fix_cc_<m>.jsonl`).
- **Segundo defecto del bench, encontrado al diseñar la app de Mac (2026-11-06)**: al reconstruir la columna para cambiar el número de etapas, la etapa de alimentación se recalculaba desde la anterior ya redondeada y el error se acumulaba. Tras 556 evaluaciones, la columna de 32 etapas tenía la alimentación en la 21 y no en la 16 (mitad). Resolverla desde cero con 32 etapas y R 1.267 da x_D 0.796, no 0.80. Corregido en `tools_sweep._rebuild_column` (posición relativa guardada una vez) con prueba que falla con el código viejo. Óptimo corregido: 29 etapas, R 1.394, 885 080 USD/año, x_D 0.80001. La tarjeta `optimo_etapas_reflujo` se actualizó y las tres respuestas guardadas se re-calificaron sin volver a correr los modelos.
- Recomendación para el producto: Opus por defecto para diseño y optimización; Sonnet aceptable para simulación (v0 100 %); Haiku no para v3.

Datos: `notebooks/bench_runs/m2/all_cc_{haiku,sonnet,opus}.jsonl` y `reciclo_fix_cc_{haiku,sonnet,opus}.jsonl`.
