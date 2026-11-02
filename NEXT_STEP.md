# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 1, **bloque 4 cerrado en lo técnico (2026-11-02)**: el agente corre sobre la suscripción
(`claude -p` + MCP AlchemAI) y pasó el bench: corrida 2 78.1 % → corrida 3 97.8 % → las dos tarjetas
restantes 100 % tras corregir enunciado y añadir reintento MCP. Corrida 4 limpia en curso
(`notebooks/bench_runs/v0_cc_opus_run4.*`) para el número final. 27 tests verdes.

## Siguiente acción exacta
1. Anotar la corrida 4 en `BENCHMARK.md` (meta: ≥ 95 % limpio).
2. Bloque 5 (Fase 2): **HAZOP preliminar automático** en `packages/hazop/`: exportar el flowsheet
   resuelto a grafo (nodos = equipos/corrientes, con T, P, fases, duty) → reglas deterministas de
   desviación por tipo de nodo → tabla HAZOP; el LLM solo redacta causas/consecuencias/salvaguardas
   candidatas marcadas como propuesta. Primer caso: compresor + enfriador + separador.
3. Barato y útil: medir el bench con `--model sonnet` en `claude -p` (ALCHEMAI_CC_MODEL=sonnet) para
   comparar costo/calidad; publicar repo = OK del owner.

## Bloqueante actual
Ninguno.

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
