# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 1, **bloque 1 cerrado (2026-11-02)**: `packages/dwsim_client/verify.py` (convergencia,
balances de masa, por compuesto y energía por equipo, leyendo el modelo de objetos de DWSIM) y
`packages/bench/cases_baseline.py`; 6 tests verdes (`uv run pytest -q`). Estudio semana 1:
`notebooks/01_flash_michelsen.py` → `thermo` reproduce a Fluxo a 4 decimales con sus k_ij; la
diferencia con DWSIM en CO₂ es la base de k_ij (anotado y pusheado en Fluxo).

## Siguiente acción exacta
Bloque 2: **10 flowsheets de referencia** en `packages/bench/cases_ref/` (yaml + builder):
mezclador, válvula JT, separador trifásico, intercambiador, bomba, compresor politrópico,
reactor de conversión (combustión CH₄), reactor de equilibrio, columna rigurosa (etanol-agua
NRTL), reciclo con Wegstein. Cada uno con respuesta esperada (DWSIM + literatura) y `verify()`
verde. Estudio semana 2: NRTL/UNIQUAC con `thermo` (notebook 02).

## Bloqueante actual
Ninguno.

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
