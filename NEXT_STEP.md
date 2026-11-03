# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Bloque 9 (2026-11-02): reactor cinético de Fluxo 1.1 implementado en el motor** (rama `feature/reactor-cinetico` de
`apps/fluxo`, 4 tests verdes, X dentro de 0.1 % de DWSIM; sin UI ni envío). Antes: **Fase 2, bloque 8 cerrado (2026-11-02)**: `dof_check` determinista (18 tools) + regla 9 del agente + Bench v1b (5 tarjetas de
especificación): agente 5/5, 0 falsas alarmas en 5 controles. Bench total = 35 tarjetas (20 factibles + 15 trampas), agente
100 % en todo lo medido. 48 tests + 1 live skip. Evidencia en `BENCHMARK.md`.

## Siguiente acción exacta
Bloque 10 (elegir uno):
1. **Fluxo 1.1 — UI del reactor cinético**: pantallas en Figma (gate obligatorio) → `DefinicionCaso`/`Reacciones.swift`/inspector;
   luego merge de `feature/reactor-cinetico` y TestFlight. Requiere OK del owner (1.0.0 en App Review).
2. **Estudio semana 4** — columnas rigurosas (inside-out / Wang-Henke): columna etanol-agua DWSIM vs Fluxo caso a caso.
3. **Segunda fila de modelo**: bench completo (35) con `ALCHEMAI_CC_MODEL=sonnet` (barato, ~20 min).
4. **Export SFILES2/DEXPI** del flowsheet de la sesión (texto/grafo intercambiable) — base de la UI y del whitepaper.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
