# 22 · Bench completo con tres modelos (bloque 29, mes 2)

**Qué:** las 50 tarjetas del ChemE Design Bench (v0 20 · v1 10 trampas físicas · v1b 5 trampas de especificación · v2 4 polares · v3 11 transporte, costo y optimización) en una sola pasada por modelo, mismo agente `claude-code`, mismas 36 tools MCP, mismo prompt, timeout 600 s por tarjeta. Modelo por `ALCHEMAI_CC_MODEL`. 2026-11-06.

Comando: `ALCHEMAI_CC_MODEL=<m> PYTHONPATH=. uv run python packages/bench/runner.py --agent claude-code --out notebooks/bench_runs/m2/all_cc_<m>.jsonl` · resumen: `PYTHONPATH=. uv run python notebooks/22_bench_tres_modelos.py`.

## Resultado (una sola pasada, sin re-intentos)

| Modelo | Puntaje | v0 (20) | Trampas (15) | v2 (4) | v3 (11) | Tiempo total | Turnos | Costo reportado* |
|---|---|---|---|---|---|---|---|---|
| Haiku | **92.4 %** | 94 % | 15/15 | 100 % | 77 % | 3 665 s | 997 | 4.72 USD |
| Sonnet | **94.0 %** | 100 % | 14/15 | 100 % | 82 % | 2 755 s | 580 | 6.18 USD |
| Opus | **97.0 %** → **99.0 %** con la tarjeta corregida | 95 % → 100 % | 15/15 | 100 % | 95 % | 2 001 s | 604 | 8.28 USD |

\* Costo equivalente que reporta Claude Code; la corrida fue por suscripción, no se cobró por API.

## Fallos, uno por uno

| Modelo | Tarjeta | Puntaje | Qué pasó | Tipo |
|---|---|---|---|---|
| Opus | reciclo | 0 | Marcó la tarjeta como sub-especificada: al agua fresca le faltaban T y P (`dof_check`) y a 1 atm los 400 K deciden si el reciclo es vapor. **Tenía razón.** | Defecto del bench → corregido |
| Opus | optimo_etapas_reflujo | 0.5 | N 31 ✓, costo +2.1 % sobre el óptimo reportado como verificado (igual que la corrida 1 del bloque 26) | Agente, marginal |
| Sonnet | trampa_azeotropo | 0 | Agotó los 600 s sin responder | Velocidad |
| Sonnet | optimo_etapas_reflujo | 0 | No devolvió JSON válido | Formato |
| Sonnet | optimo_recuperacion_calor | 0 | 317 m² y 385 k USD/año vs 650 m² y 130 k: **número equivocado reportado como verificado** | Agente, grave |
| Haiku | diseno_intercambiador_ua | 0 | Agotó los 600 s | Velocidad |
| Haiku | optimo_recuperacion_calor | 0 | No devolvió JSON válido | Formato |
| Haiku | columna_rigurosa | 0.7 | Qr 2 957 kW vs 3 489: equivocado reportado como verificado | Agente, grave |
| Haiku | optimo_compresion_tres_etapas | 0.5 | P1 2.649 vs 2.699 bar (fuera de 1 %) reportado como verificado | Agente |
| Haiku | optimo_etapas_reflujo | 0 | N 37 vs 32, costo +1.9 %: equivocado reportado como verificado | Agente, grave |

## Lectura

- **Las trampas aguantan en los tres.** Haiku y Opus 15/15; Sonnet falló una sola por tiempo, no por aceptar algo imposible. Ningún modelo reportó como verificado un caso infactible.
- **La diferencia está en v3** (optimización y costo), no en simulación: Haiku 77 %, Sonnet 82 %, Opus 95 %.
- **El fallo que importa es «equivocado reportado como verificado»**: Haiku 3 tarjetas, Sonnet 1, Opus 1 marginal (+2 % en costo). Es lo que el bench castiga con 0 en la cantidad y lo que un ingeniero no detectaría sin re-simular.
- **Opus es el más rápido** (2 001 s, 40 s/tarjeta) aunque el más caro por token; Haiku usa casi el doble de turnos.
- **El agente encontró un defecto del bench**: la tarjeta `reciclo` llevaba desde v0 sin T ni P del agua fresca (la referencia usa 300 K y 1 atm). Se corrigió el enunciado en `cases_ref.py` y en la tarjeta; la re-corrida de esa tarjeta dio 100 % con los tres modelos (`reciclo_fix_cc_<m>.jsonl`).
- Recomendación para el producto: Opus por defecto para diseño y optimización; Sonnet aceptable para simulación (v0 100 %); Haiku no para v3.

Datos: `notebooks/bench_runs/m2/all_cc_{haiku,sonnet,opus}.jsonl` y `reciclo_fix_cc_{haiku,sonnet,opus}.jsonl`.
