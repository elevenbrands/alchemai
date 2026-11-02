# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
Fase 1, **bloque 2 cerrado (2026-11-02)**: 10 flowsheets de referencia en `packages/bench/cases_ref.py`
(enunciado ES/EN, builder, readouts, chequeos de libro), runner `run_ref.py`, snapshot DWSIM en
`expected_ref.yaml`; 17 tests verdes. `verify.py` ya cierra reactores (término Σ Δn·ΔHf) e
intercambiadores. Hallazgo para Fluxo: η_p 0.75 → η_ad 0.7447 en DWSIM (gas ideal 0.744) para
r = 1.2; Fluxo documenta 0.722 → revisar en qué condiciones (no es necesariamente error).

## Siguiente acción exacta
Bloque 3 (semanas 3-4): **ChemE Design Bench v0**. Convertir los 13 casos (3 base + 10 ref) en
tarjetas `yaml` con enunciado, entradas, respuestas esperadas con tolerancia y rúbrica; añadir
7 casos de *diseño* (no solo simulación): dimensionar separador, elegir P de columna, estimar
etapas, etc. Runner que puntúa a cualquier agente MCP. Estudio semana 2: NRTL/UNIQUAC con
`thermo` (notebook 02) sobre el caso etanol-agua.

## Bloqueante actual
Ninguno.

## Decisiones pendientes del owner
- (ninguna) Nombre AlchemAI confirmado 2026-11-02; remoto `elevenbrands/alchemai` (privado).
- Hacer público el repo al cerrar Bench v0 (mes 3) = OK del owner.
