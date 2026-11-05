# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 24)

## Dónde estamos
**Bloque 24 (2026-11-05): optimización multivariable** (`notebooks/18_optimizacion_multivariable.md`): `optimize_design_multi`
(36 tools), malla + Nelder-Mead; compresión 3 etapas 3.021/9.087 bar vs teoría 3/9; con límite del HAZOP 2.699/8.591 bar;
absorbedor aceite + enfriamiento con óptimo de esquina sensible al precio. Bench v3 = 10 tarjetas (49 en total).
Bloques 18-23 del mismo día: transporte, costo, costo anual, optimizador 1D, masa, hidráulica.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Opciones (el owner elige):
1. **Conjunto apartado de HAZOP** para medir recall sin sesgo y calificar precisión.
2. **Variables enteras** (número de etapas) en la optimización + costo de relleno e internos.
3. **Fluxo 1.1 UI en Figma** (tras el veredicto de Apple de 1.0.0).
4. **Repo público + primer post** (decisión del owner) con la evidencia de los bloques 13-24.
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
