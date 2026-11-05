# NEXT_STEP — cursor

Actualizado: 2026-11-04 (bloque 14)

## Dónde estamos
**Bloque 14 (2026-11-04): bench completo con Sonnet** (`notebooks/08_bench_sonnet.md`): 39 tarjetas con `ALCHEMAI_CC_MODEL=sonnet`.
Corrida 1: 95.1 % (15/15 trampas, 4/4 polares eligiendo NRTL solo → sin regla 10); los 5 fallos eran del prompt/bench (regla 5 de signo
ambigua: Sonnet negaba Qc aunque DWSIM lo da positivo; `reciclo` no pedía la T). Regla 5 reescrita + enunciado aclarado → corrida 2/3:
las 39 al 100 % entre corridas (no en una sola pasada). Arnés: `_kill_group` mata el grupo de procesos al vencer el timeout (antes
quedaban servidores MCP DWSIM huérfanos al 98 % CPU); `ZERO_ABS_TOL` 1e-4 para esperados = 0. Sonnet ≈ 40 % más lento que Opus.
Antes: bloque 13 tarjetas polares (v2), bloque 12 pares NRTL, bloque 11 NRTL en FluxoEngine, bloque 10 columnas rigurosas, bloque 9
reactor cinético. Fluxo: 57 tests / 16 suites, rama `feature/reactor-cinetico`, sin tocar en los bloques 13–14.

## Siguiente acción exacta
Bloque 15 (elegir uno):
1. **Fluxo 1.1 — UI en Figma** (gate obligatorio): reactor cinético + selector de modelo (PR/SRK/NRTL) + aviso de pares faltantes
   y de PR con polares; después `DefinicionCaso`/inspector; merge y envío solo con OK del owner (1.0.0 en App Review).
2. **Export SFILES2/DEXPI** del flowsheet de la sesión (estudio semana 8 adelantado; abre el camino al HAZOP sobre grafo estándar).
3. **Estudio semana 6 — Pyomo/IDAES** (modelado orientado a ecuaciones, tutoriales oficiales).
4. Opcional barato: una pasada única de las 39 con Sonnet y el prompt nuevo (~27 min) para tener la cifra en una sola corrida.
Recomendación: 1 si el owner abre 1.1 (requiere su participación en Figma); si no, 2.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
