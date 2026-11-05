# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 19)

## Dónde estamos
**Bloque 19 (2026-11-05): costo de equipos** (`notebooks/13_costo_de_equipos.md`): costeo SSLW de IDAES (Seider) usado tal cual,
sin ipopt; reproduce los 4 valores de los tests de IDAES; tools `equipment_cost`, `costing_options`, `cost_unit` (28). Bench v3 = 5
tarjetas (3 transporte + 2 costo; 44 en total). **Bloque 18**: ΔP de tubería ≤ 1.4 % vs DWSIM, área a 0.2 %. Antes: 17 HAZOP vs libro.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
**Bloque 20 — costo instalado y de operación**: factores de módulo desnudo (verificar la fuente antes de usarlos), costo de
utilidades y electricidad por año, y una tool `flowsheet_cost` que sume CAPEX + OPEX de un flowsheet resuelto. Con eso, el primer
estudio de optimización (fase 3): barrer una variable de diseño (p. ej. aproximación de T de un intercambiador o reflujo de una
columna) y encontrar el costo anual mínimo con `sweep`.
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).
Después: conjunto apartado de HAZOP, DEXPI nivel P&ID, Fluxo 1.1 UI en Figma.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
