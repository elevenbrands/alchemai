# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloques 27-28 preparados)

## Dónde estamos
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

Siguiente (plan mes 2): bench v2 público con resultados de ≥ 2 modelos, whitepaper ES/EN, y cuando el owner lo decida la UI (web vs Mac).
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
