# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Fase 0 cerrada (2026-11-02).** Entorno ✅, DWSIM 10 headless ✅, línea base Fluxo vs DWSIM ✅:
balances, potencias, cargas térmicas y temperaturas dentro de 0.3 %; FUG (N_min/R_min) y k_ij de
CO₂ divergen y quedaron anotados como 🟠 en `apps/fluxo/MOCKS.md`. Tabla:
`notebooks/baseline_fluxo_vs_dwsim.md`.

## Siguiente acción exacta
Fase 1, bloque 1 (semana 1): **estudio** = flash y estabilidad (Michelsen) con `thermo`,
notebook que reproduce el caso 1 y explica la diferencia de k_ij; **construcción** =
`packages/dwsim_client/verify.py`: balances de masa/energía por equipo y chequeo de convergencia
sobre la Fluent API, con los 3 casos de la línea base como primeros tests (`tests/`).
Después: 10 flowsheets de referencia (bloque 2) → Bench v0 (bloque 3).

## Bloqueante actual
Ninguno.

## Decisiones pendientes del owner
- Nombre e identidad del producto/repo.
- Repo público (hoy local, sin remoto).
