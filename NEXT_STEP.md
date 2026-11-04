# NEXT_STEP — cursor

Actualizado: 2026-11-03

## Dónde estamos
**Bloque 11 (2026-11-03): NRTL en FluxoEngine cerrado** (`notebooks/05_nrtl_fluxo_vs_dwsim.md`, rama `feature/reactor-cinetico` de
Fluxo): `ModeloDeLiquido.nrtl` (γ-φ, vapor PR o ideal), Psat DIPPR 101 de Perry's en los 24 componentes, 16 pares NRTL del conjunto
de DWSIM. Volcado de DWSIM (`05_nrtl_dwsim_dump.py`): vapor ideal, φ_L = γ·Psat/P, τ = A/(1.98721·T). Reproducción: Txy a 0.014 K /
0.0002 en y; columna etanol-agua x_D 0.8043 vs 0.8040, T a 0.01 K, cargas +2 %, 22 iteraciones donde PR no convergía en 2000.
Con datos propios x_D 0.8088 (etanol de Perry's 0.33 K más volátil que el de ChemSep). Fluxo: 56 tests / 15 suites verdes.
Antes: bloque 10 columnas rigurosas (Fluxo = DWSIM en depro; `WithFeed` 0-based corregido), bloque 9 reactor cinético, bloque 8
`dof_check` + Bench v1b. Bench = 35 tarjetas, agente 100 % en todo lo medido. 48 tests + 1 live skip (alchemai).

## Siguiente acción exacta
Bloque 12 (elegir uno):
1. **Validar los 15 pares NRTL restantes** contra DWSIM: Txy por par con `05_nrtl_dwsim_dump.py` generalizado + test en Fluxo
   (barato, automatizable; cierra el 🟠 de MOCKS sin tocar UI).
2. **Fluxo 1.1 — UI en Figma** (gate obligatorio): reactor cinético + selector de modelo (PR/SRK/NRTL) + aviso de pares faltantes
   y de PR con polares; después `DefinicionCaso`/inspector; merge y envío solo con OK del owner (1.0.0 en App Review).
3. **Segunda fila de modelo**: bench completo (35) con `ALCHEMAI_CC_MODEL=sonnet` (~20 min).
4. **Export SFILES2/DEXPI** del flowsheet de la sesión.
Recomendación: 1 si el owner quiere seguir en motor; 2 si abre 1.1 (requiere su participación en Figma).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); columna PR con mezcla polar puede no terminar (correr con `WithMaxIterations` o límite de tiempo); en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
