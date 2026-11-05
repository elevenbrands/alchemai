# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 21)

## Dónde estamos
**Bloque 21 (2026-11-05): optimizador genérico** (`notebooks/15_optimizador.md`): tool `optimize_design` (30 tools), una variable,
objetivo costo anual o suma de lecturas, límites del HAZOP como restricciones duras; reproduce el óptimo del notebook 14 y la
teoría de compresión en dos etapas (3.007 vs 3.0 bar); con descarga ≤ 440 K el óptimo pasa a 2.885 bar; detecta infactibilidad.
Bench v3 = 7 tarjetas (46 en total). Bloques 18-20 del mismo día: transporte, costo de compra, costo anual.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
**Bloque 22 — transferencia de masa** (lo que falta de fenómenos de transporte): columna de absorción/relleno con HTU/NTU
(Colburn para la línea de operación recta, NTU numérico en general), coeficientes kL/kG por correlación (Onda) desde
propiedades de DWSIM, altura de relleno; validar contra la columna de absorción de DWSIM y contra un ejemplo analítico.
Alternativas: optimización de dos variables (coordenadas), conjunto apartado de HAZOP, DEXPI nivel P&ID, Fluxo 1.1 UI en Figma.
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
