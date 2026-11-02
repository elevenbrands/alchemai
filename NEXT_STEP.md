# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 1, **bloque 4 en curso (2026-11-02)**: código del agente completo y verificado sin LLM:
`packages/dwsim_client/tools.py` (12 herramientas JSON), `packages/dwsim_mcp/server.py` (MCP propio,
12 tools por stdio ✅), `packages/agent/agent.py` (Claude Opus 5.5 + tool runner + protocolo del
bench), `runner.py --agent claude`. Tests: 26 (el live del agente se salta sin credenciales).

## Siguiente acción exacta
1. Owner guarda la llave: `security add-generic-password -a alchemai -s ANTHROPIC_API_KEY -w`
2. `ALCHEMAI_LIVE=1 uv run pytest -q tests/test_tools_and_mcp.py -k live` (1 tarjeta, ~1 min)
3. `uv run python packages/bench/runner.py --agent claude` → anotar % en BENCHMARK.md; meta ≥ 80 %.
   Iterar prompt/herramientas sobre las tarjetas que fallen (esperable: reactores adiabáticos con
   búsqueda, columna rigurosa). Luego probar `ALCHEMAI_MODEL=claude-sonnet-5-5` para costo.
4. Añadir `fallbacks` server-side al medir; registrar costo por tarjeta (usage).

## Bloqueante actual
**Llave de Anthropic en Keychain** (decisión/acción del owner). Todo lo demás está listo.

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
