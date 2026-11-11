# NEXT_STEP — cursor

Actualizado: 2026-11-10 (M1-M5, M6a y M6b cerrados)

## Dónde estamos
**Bloque M6b (2026-11-10): `stirred_tank_cfd`, 54 tools, ✅ (M6b-2 cerró el sello).** Rushton con mamparas por MRF k-ε: Np, P, P/V, θ95 (Grenville, correlación). Aspa a nivel 3 de refinamiento y `maxBoundarySkewness 4` (todas las mallas pasan checkMesh). El GCI no aplica (el aspa de 3 mm salta con la malla de fondo: 49 → 5.13 y 50 → 4.81 en el mismo caso), así que el sello es una **banda entre 3 mallas**: cada malla gruesa a < 10 % de la fina, reportada como incertidumbre. Corrida de punta a punta verificada: Np 4.85 ± 3.5 % (64/49/38 → 4.85/5.02/4.70), torque 0.7-2.4 %, 285 W/m³, 93 min.

**Bloque M6a (2026-11-09): `vessel_residence_time_cfd`, 53 tools.** OpenFOAM simpleFoam + edad media local en un tanque prismático; τ, σ²/τ², N, volumen estancado y conversión real vs CSTR; teorema edad de salida = V/Q y estudio de malla. Sin impulsor (M6b).

**Bloque M5 (2026-11-09): `pinch_analysis`, 52 tools.** Tabla del problema, MER, curvas compuestas y gran compuesta, unidades mínimas, barrido de ΔTmin y ahorro vs servicios actuales; Kemp (20/60 kW) y Smith (7.5/10 MW) reproducidos; evidencia = curvas desplazadas que se tocan sin cruzarse.

**Bloque M4c (2026-11-09): `shell_side_bell_delaware`, 51 tools.** Jc/Jl/Jb/Js/Jr y Rl/Rb/Rs con banco ideal Žukauskas (ESDU de contraste), claros TEMA por defecto declarados en `assumptions`, Kern de comparación. Evidencia = gráficas vs ajustes HEDH y dos bancos ideales; sin ejemplo publicado legible todavía.

**Bloque M4 (2026-11-09): fluids/ht como tools, 50 tools.** M4a: `relief_valve_sizing` (API 520 + API 526), `control_valve_sizing` (IEC 60534-2-1),
`fittings_resistance`, `compressible_gas_line` (isotérmico propio + EDO), `two_phase_pressure_drop`; M4b: `shell_side_kern`, `nucleate_boiling`, `film_condensation`.
Ejemplos de API 520 (1, 2, 4, 5), IEC (1-3) e Incropera 10.1 reproducidos; 16 tests; reglas 13-14 del agente; 163 tests en verde.
**Bloque M3 (2026-11-08): economía completa, 42 tools** (`bare_module_factors`, `capital_investment`, `cash_flow_analysis`): F_BM de Seider leídos
de BioSTEAM, inversión total con porcentajes del usuario, flujo de caja = fórmulas cerradas y = funciones de BioSTEAM; 15 tests; regla 12 del agente.
**Bloque M2 (2026-11-08): Cantera como tools, 39 tools** (`adiabatic_flame`, `kinetic_reactor_network`): notebook 03 reproducido por la tool y por MCP
(llama 2226 K vs completa 2327 K, PFR/CSTR = analítico, ignición 4.53 ms, apagado 0.2-0.5 ms, S_L 0.378 m/s); 7 tests; regla 11 del agente.
**Bloque M1 (2026-11-08): `tubular_reactor_field`, primer motor externo (FEniCSx como proceso aparte, 37 tools)**: Caso 1 reproducido por la
tool y por MCP (pistón 0.7940 = analítico, laminar 0.7819, penalización 1.2 puntos, Re 2000, sello verificado con 4 comprobaciones); 6 tests.
Mapa de 12 áreas en `docs/MAPA_MOTORES.md`.
**Bloque 29 (2026-11-06): bench completo con tres modelos** (`notebooks/22_bench_tres_modelos.md`): una pasada de las 50 tarjetas
por modelo: Opus 97.0 % (99.0 % con `reciclo` corregida), Sonnet 94.0 %, Haiku 92.4 %; trampas 15/15 · 14/15 · 15/15; la brecha está en v3
(95/82/77 %). Opus detectó que `reciclo` no daba T ni P del agua fresca: corregido. Datos en `notebooks/bench_runs/m2/`.
**Corrección 2026-11-06: deriva de la alimentación al reconstruir columnas** (óptimo del bloque 26 no reproducible; corregido a 29 etapas, R 1.394, 885 080 USD/año; tarjeta y calificaciones actualizadas: Opus 100 % en una pasada).
**Bloque 30 (2026-11-06): constantes de Fluxo** (`notebooks/21_fluxo_constantes.md`): 166/167 dentro de tolerancia; ω H₂S 0.090 → 0.100.
**Bloque 26 (2026-11-05): variables enteras + costo de platos/relleno** (`notebooks/20_variables_enteras.md`): etapas enteras × reflujo en la
columna etanol-agua → 32 etapas, R 1.267, R/R_min riguroso 1.12; platos de Seider exactos; tarjeta `optimo_etapas_reflujo` (50 tarjetas).
**Bloque 25 (2026-11-05): conjunto apartado de HAZOP** (`notebooks/19_hazop_apartado.md`): 5 HAZOP publicados nuevos; recall a ciegas
25/30 = 83 % → 30/30 con reglas de sustancia/régimen térmico; precisión útil 69 % → 76 %, espurias 2.6 % → 0 (revisor independiente,
rúbrica previa). El owner pidió seguir en orden: 26 variables enteras + costo de relleno, 27 Fluxo 1.1 UI Figma, 28 repo público + post.
**Bloque 24 (2026-11-05): optimización multivariable** (`notebooks/18_optimizacion_multivariable.md`): `optimize_design_multi`
(36 tools), malla + Nelder-Mead; compresión 3 etapas 3.021/9.087 bar vs teoría 3/9; con límite del HAZOP 2.699/8.591 bar;
absorbedor aceite + enfriamiento con óptimo de esquina sensible al precio. Bench v3 = 10 tarjetas (49 en total).
Bloques 18-23 del mismo día: transporte, costo, costo anual, optimizador 1D, masa, hidráulica.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Orden fijado por el owner (2026-11-05, «sigue con esos 4 en orden»):
1. ✅ Conjunto apartado de HAZOP (bloque 25).
2. ✅ Bloque 26: variables enteras + costo de platos y relleno.
3. ✅ **Bloque 27** (Figma aprobado por el owner e implementado en SwiftUI en `apps/fluxo` rama `feature/reactor-cinetico`, commit del 2026-11-05; falta TestFlight/QA físico). Antes:: pantallas Figma `iPhone/11 Nuevo reactor cinético` (23147:861) e `iPhone/12 Base · NRTL` (23147:979) en ZRNmiWDmd6NjWIU0bfW6Fi; falta revisión del owner + Premium Brand Gate. Apple rechazó Fluxo 1.0.0 (4.3.0 Spam, 2026-10-20); el owner respondió 2026-10-27; sin veredicto.
4. ✅ **Bloque 28**: repo PÚBLICO desde 2026-11-05 (OK del owner «Aprobado y sigue»); post listo en `docs/posts/01_lanzamiento_alchemai.md` con el enlace, lo publica el owner en LinkedIn.

**Decisión del owner 2026-11-06: se siguen AlchemAI (producto principal, sistema con IA en la Mac) y Fluxo (app iOS, canal móvil del motor) en paralelo.**

Siguiente (plan mes 2):
1. ✅ Bloque 29: tabla de tres modelos.
2. ✅ Bloque 30: constantes de Fluxo.
**Visión del owner 2026-11-08: una sola aplicación que cubra todo lo que hacen los software de ingeniería química** (calor, masa, momento,
balances, reacción, separación, costos, seguridad, diseño de planta); el agente elige el motor. Mapa por área en `docs/MAPA_MOTORES.md`
(12 áreas, estado ✅/🧪/📦/⬜ verificado, orden propuesto de 10 bloques). Primer caso de dos motores: `transporte-lab/reactor/` (DWSIM PFR
0.7945 vs FEniCSx laminar 0.7818). M1 `tubular_reactor_field` ✅, M2 Cantera ✅ y M3 economía ✅ 2026-11-08. M4 fluids/ht ✅ 2026-11-09 y M4c Bell-Delaware ✅ (51 tools; falta validarlo contra un ejemplo publicado con datos legibles). M5 pinch ✅ (52 tools). M6a tanque de flujo continuo CFD ✅ (53 tools). M6b viabilidad ✅ 2026-11-10 (caso 3 de `transporte-lab`: Rushton MRF, Np 4.67 vs ≈ 5, balance de torque 0.26 %, 19 min con `mpirun -np 4` desde `$TMPDIR`). M6b-2 ✅ 2026-11-10: sello por banda entre mallas, Np 4.85 ± 3.5 % verificado. **Siguiente (decide el owner): M5b (corrientes del pinch desde el flowsheet) o M7.**
3. **Bloque 31**: borrador del whitepaper ES/EN «LLM agents for process design: a verified benchmark» desde BENCHMARK.md (publicar = OK del owner).
4. Bloque 32: estudio de modelos sustitutos con OMLT (plan, tema 7).
5. Fluxo: exportar casos a JSON/SFILES.
**Decisión del owner 2026-11-06: la UI es una app de Mac en SwiftUI** («App Mac Swift»). Estructura en `docs/MAC_APP_ESTRUCTURA.md`
(agente en Python como proceso hijo, DWSIM aparte por GPL, Developer ID fuera de la App Store, repo privado `apps/alchemai-mac`).
Plantilla elegida por el owner 2026-11-06: macOS 27 + patrones del AI Agent UI Kit. **Diseño Figma listo** (Cb75xN1JHDXuPpUOzz37VQ, 7 pantallas, capturas en `docs/design/mac/`): **APROBADO por el owner 2026-11-06**. `alchemai serve` (`packages/serve/cli.py`) ✅ y app `apps/alchemai-mac` B1 ✅ 2026-11-06 (E2E real en la app). Siguiente en la app: B2 diagrama; luego Premium Brand Gate (ícono).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Publicar el post de lanzamiento en LinkedIn (lo hace el owner; repo ya público).
- App de Mac: plantilla de Figma (bloqueante), distribución fuera de la App Store (recomendada), cobro y precio.
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
