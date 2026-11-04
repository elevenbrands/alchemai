# NEXT_STEP — cursor

Actualizado: 2026-11-04 (bloque 13)

## Dónde estamos
**Bloque 13 (2026-11-04): tarjetas polares, bench v2** (`notebooks/07_tarjetas_polares.md`): 4 tarjetas (burbuja/rocío y azeótropo
etanol-agua, flash y columna metanol-agua) sin paquete en el enunciado; esperado DWSIM-NRTL, PR falla en las 4. `sweep` acepta
`mole_fraction.<Compuesto>` y devuelve mínimo/máximo. Agente Opus 4/4 al 100 % en 139 s eligiendo NRTL solo (sin regla nueva).
Bench = 39 tarjetas (`--version v2`), 51 tests + 1 skip. Antes: bloque 12 (16 pares NRTL de Fluxo validados vs DWSIM), bloque 11 NRTL
en FluxoEngine, bloque 10 columnas rigurosas, bloque 9 reactor cinético. Fluxo: 57 tests / 16 suites, rama `feature/reactor-cinetico`.

## Siguiente acción exacta
Bloque 14 (elegir uno):
1. **Fluxo 1.1 — UI en Figma** (gate obligatorio): reactor cinético + selector de modelo (PR/SRK/NRTL) + aviso de pares faltantes
   y de PR con polares; después `DefinicionCaso`/inspector; merge y envío solo con OK del owner (1.0.0 en App Review).
2. **Segunda fila de modelo**: bench completo (39) con `ALCHEMAI_CC_MODEL=sonnet` (~25 min); si falla en v2, añadir la regla 10 de paquete.
3. **Export SFILES2/DEXPI** del flowsheet de la sesión (estudio semana 8 adelantado).
4. **Estudio semana 6 — Pyomo/IDAES** (modelado orientado a ecuaciones, tutoriales oficiales).
Recomendación: 1 si el owner abre 1.1 (requiere su participación en Figma); si no, 2 (mide al modelo barato con las trampas y las polares).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); columna PR con mezcla polar puede no terminar (correr con `WithMaxIterations` o límite de tiempo); en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
