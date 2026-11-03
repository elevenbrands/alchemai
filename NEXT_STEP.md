# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Fase 2, bloque 8 cerrado (2026-11-02)**: `dof_check` determinista (18 tools) + regla 9 del agente + Bench v1b (5 tarjetas de
especificación): agente 5/5, 0 falsas alarmas en 5 controles. Bench total = 35 tarjetas (20 factibles + 15 trampas), agente
100 % en todo lo medido. 48 tests + 1 live skip. Evidencia en `BENCHMARK.md`.

## Siguiente acción exacta
Bloque 9 (elegir uno):
1. **Fluxo 1.1 — reactor cinético** siguiendo la sección D de `notebooks/03_cantera_reactores.md` (CSTR algebraico + PFR RK4,
   Arrhenius, validación obligatoria X_PFR 0.9807 / X_CSTR 0.7979). Requiere OK del owner (1.0.0 en App Review); se puede
   preparar en rama `feature/reactor-cinetico` sin enviar.
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
