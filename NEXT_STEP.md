# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 0 cerrada en lo técnico: entorno Python ✅, DWSIM 10 headless en el Mac ✅ vía Fluent API
(pythonnet + .NET 10 user-level). Evidencia en `BENCHMARK.md` (mezclador oficial; flash PR
DWSIM 0.7456 vs thermo 0.7426). Pendiente de Fase 0: línea base Fluxo vs DWSIM.

## Siguiente acción exacta
1. Línea base Fluxo vs DWSIM (3 casos, meta <1 %): flash PT gas natural, columna FUG, compresor +
   enfriador. Escribir `notebooks/baseline_fluxo_vs_dwsim.py` que corre DWSIM y deja una tabla;
   los valores de Fluxo salen de sus tests (`apps/fluxo/Packages/FluxoEngine/Tests`). Lo que no
   cuadre → `apps/fluxo/MOCKS.md`.
2. Fase 1, bloque 1: `packages/dwsim_client/` como capa de verificación (balances de masa/energía
   por equipo, convergencia) sobre la Fluent API; 10 flowsheets de referencia como tests.

## Bloqueante actual
Ninguno. (El build del MCP oficial ya no es necesario para fases 0-1; el clasificador de Claude
Code lo bloquea, el owner lo corre si llega a hacer falta.)

## Decisiones pendientes del owner
- Nombre e identidad del producto/repo.
- Repo público (hoy local, sin remoto).
