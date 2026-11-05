# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 20)

## Dónde estamos
**Bloque 20 (2026-11-05): costo anual + primera optimización** (`notebooks/14_optimizacion_recuperacion_calor.md`):
`flowsheet_cost` (29 tools) = compra × Lang (solo valores verificados) × ubicación (IDAES) × CRF + operación con precios del usuario;
corregido el costo de intercambiadores chicos (la correlación de Seider sube bajo ~9 m²). Óptimo de un intercambiador
alimentación-producto: 650 m², aproximación 7.1 K (precios ilustrativos). Bench v3 = 6 tarjetas (45 en total).
Bloques 18-19 del mismo día: transporte (ΔP ≤ 1.4 % vs DWSIM) y costo de compra (Seider vía IDAES, 4/4 tests de IDAES).
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
**Bloque 21 — optimizador genérico para el agente**: tool `optimize_design(flowsheet_id, variable, bounds, objective)` que
varíe un parámetro de equipo o corriente (sección dorada / barrido) y minimice `flowsheet_cost`, con restricciones del HAZOP
(p. ej. T máxima, aproximación mínima) y la curva completa como evidencia. Probar con la tarjeta `optimo_recuperacion_calor`
y con reflujo de columna vs costo.
Alternativas: transferencia de masa (HTU/NTU, kLa, columnas de relleno: el hueco que queda de fenómenos de transporte),
conjunto apartado de HAZOP, DEXPI nivel P&ID, Fluxo 1.1 UI en Figma.
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
