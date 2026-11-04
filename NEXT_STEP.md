# NEXT_STEP — cursor

Actualizado: 2026-11-03

## Dónde estamos
**Bloque 12 (2026-11-03): los 16 pares NRTL de Fluxo validados vs DWSIM** (`notebooks/06_nrtl_pares.md`): Txy a 1 atm, 11 puntos
por par; con la Psat y el vapor de DWSIM ΔT ≤ 0.12 K / Δy ≤ 0.004 (residuo = tolerancia del flash de DWSIM; réplica Python
independiente coincide con Fluxo a 0.001 K); con datos propios 0.23–1.08 K (fuente de Psat). `NRTLParesDWSIM.swift` se genera
desde `06_nrtl_pares_gen.py`. Fluxo: 57 tests / 16 suites verdes. Antes: bloque 11 NRTL en FluxoEngine (reproduce la columna
etanol-agua de DWSIM, x_D 0.8043 vs 0.8040), bloque 10 columnas rigurosas, bloque 9 reactor cinético, bloque 8 `dof_check`.
Bench = 35 tarjetas, agente 100 % en todo lo medido. 48 tests + 1 live skip (alchemai).

## Siguiente acción exacta
Bloque 13 (elegir uno):
1. **Fluxo 1.1 — UI en Figma** (gate obligatorio): reactor cinético + selector de modelo (PR/SRK/NRTL) + aviso de pares faltantes
   y de PR con polares; después `DefinicionCaso`/inspector; merge y envío solo con OK del owner (1.0.0 en App Review).
2. **Segunda fila de modelo**: bench completo (35) con `ALCHEMAI_CC_MODEL=sonnet` (~20 min).
3. **Export SFILES2/DEXPI** del flowsheet de la sesión (estudio semana 8 adelantado).
4. **NRTL en el agente**: `tools.py` ya acepta `NRTL` como paquete de DWSIM; añadir tarjetas de bench con mezclas polares
   (columna etanol-agua, metanol-agua) para que el agente aprenda a elegir paquete.
Recomendación: 1 si el owner abre 1.1 (requiere su participación en Figma); si no, 4 (sigue en motor/bench y usa lo validado).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); columna PR con mezcla polar puede no terminar (correr con `WithMaxIterations` o límite de tiempo); en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
