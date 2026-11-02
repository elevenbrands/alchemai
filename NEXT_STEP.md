# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 0 (entorno y línea base), sub-bloque 1. Entorno Python ✅ (`notebooks/smoke_stack.py`).
DWSIM 10.2.12 extraído en `~/Applications/DWSIM10` ✅. .NET 10 SDK 10.0.403 en `~/.dotnet` ✅.
Fuente de DWSIM en `~/tools/dwsim10` ✅. Cliente stdio listo: `packages/dwsim_client/smoke.py` ✅.

## Siguiente acción exacta
1. El owner compila el MCP server (el clasificador de Claude Code bloqueó compilar código
   externo clonado):
   `cd ~/tools/dwsim10 && PATH=$HOME/.dotnet:$PATH dotnet build tools/DWSIM.MCPServer/DWSIM.MCPServer.csproj -c Release`
   Luego: `uv run python packages/dwsim_client/smoke.py` → debe listar 58 tools y crear un flowsheet.
2. Correr un flowsheet mínimo (mezclador + separador flash, C1/C2/C3, PR) por MCP y guardar la
   evidencia en `BENCHMARK.md`.
3. Línea base Fluxo vs DWSIM (3 casos, <1 %): flash PT gas natural, columna FUG, compresor +
   enfriador. Lo que no cuadre → `apps/fluxo/MOCKS.md`.

## Bloqueante actual
Build del MCP server pendiente del owner (ver acción 1). Riesgo: si `dotnet build` del MCP server falla en macOS arm64, plan B = `pythonnet`
sobre `DWSIM.Automation.dll` del tarball (self-contained) o DwsimPy.

## Decisiones pendientes del owner
- Nombre e identidad del producto/repo.
- Repo público (hoy local, sin remoto).
