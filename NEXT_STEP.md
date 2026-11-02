# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Fase 2, bloque 5 cerrado (2026-11-02)**: HAZOP preliminar determinista en `packages/hazop/`
(grafo + reglas + reporte MD/Excel), 16 herramientas en MCP y agente (`hazop_preliminary`), 3 tests
nuevos (30 en total). Demo: `notebooks/hazop/gas_compresion.xlsx`.

## Siguiente acción exacta
Bloque 6: (1) **capa LLM del HAZOP**: a partir de las filas `rule` y el grafo, Claude propone causas,
consecuencias y salvaguardas adicionales marcadas `source=proposal`, nunca reemplaza filas `rule`;
vía `claude -p` con la suscripción. (2) **Bench v1**: 10 tarjetas "trampa" (especificaciones
inconsistentes: cruce de temperaturas, destilado imposible por azeótropo, bomba con descarga menor
que succión, reactor sin oxidante) para medir que el agente detecta errores en vez de inventar.
(3) Estudio semana 3: cinética y redes de reactores con Cantera (notebook 03) → base del reactor
cinético de Fluxo 1.1.

## Bloqueante actual
Ninguno.

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
