# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 1, **bloque 3 cerrado (2026-11-02)**: ChemE Design Bench v0 con 20 tarjetas yaml bilingües
(`packages/bench/cards/`), 7 casos de diseño con oráculo DWSIM (`design_cases.py`), runner con
protocolo JSON para cualquier agente (`runner.py`): oráculo 100 %, nulo 0 %; 22 tests verdes.
Pendiente de estudio: notebook 02 NRTL (etanol-agua) con `thermo`.

## Siguiente acción exacta
Bloque 4 (semanas 5-8): **el agente**. `packages/agent/`: agente con Claude (`claude-sonnet-5-5`
por defecto) que recibe la tarjeta pública, planea, construye el flowsheet con la Fluent API de
DWSIM (herramientas: add_compounds, add_stream, add_unit, connect, solve, verify, read), adjunta
el `Report` de `verify.py` como evidencia y marca `unverified` lo que no cerró. Primer objetivo:
≥ 80 % del bench. Decisión: exponer las herramientas vía MCP propio (MIT) sobre `fluent.py`,
en proceso separado (GPL de DWSIM), para que también lo usen Claude Desktop/Code.
Requiere ANTHROPIC_API_KEY en Keychain (nunca en archivos).

## Bloqueante actual
Ninguno. (Para el bloque 4: llave de Anthropic del owner en Keychain.)

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
