# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Fase 1 completa (2026-11-02).** Bloque 4 cerrado: el agente corre sobre la suscripción (`claude -p` +
MCP AlchemAI) y la corrida 4 limpia dio **100 % en las 20 tarjetas** (554 s). Historial: 78.1 % →
97.8 % → 100 %. 27 tests verdes. Pendiente de la Fase 1 solo lo que es decisión del owner: repo
público + primer post.

## Siguiente acción exacta
1. Bloque 5 (Fase 2): **HAZOP preliminar automático** en `packages/hazop/`: exportar el flowsheet
   resuelto a grafo (nodos = equipos/corrientes, con T, P, fases, duty) → reglas deterministas de
   desviación por tipo de nodo → tabla HAZOP; el LLM solo redacta causas/consecuencias/salvaguardas
   candidatas marcadas como propuesta. Primer caso: compresor + enfriador + separador.
2. Barato y útil: medir el bench con `--model sonnet` en `claude -p` (ALCHEMAI_CC_MODEL=sonnet) para
   comparar costo/calidad; publicar repo = OK del owner.

## Bloqueante actual
Ninguno.

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
