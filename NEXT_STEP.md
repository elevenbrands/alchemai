# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 23)

## Dónde estamos
**Bloque 23 (2026-11-05): hidráulica de columnas** (`notebooks/17_hidraulica_columnas.md`): inundación, ΔP y diámetro con
Stichlmair (reproduce el ejemplo del artículo), catálogo de 6 rellenos (fuente secundaria), 35 tools; primer absorbedor
dimensionado y costeado de punta a punta (Pall 25 mm: D 1.05 m, Z 5.2 m, casco 107 621 USD 2023). Bench v3 = 9 tarjetas
(48 en total). Bloques 18-22 del mismo día: transporte, costo de compra, costo anual, optimizador, transferencia de masa.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Opciones (el owner elige):
1. **Optimización de dos variables** (p. ej. flujo de aceite y etapas del absorbedor contra costo anual).
2. **Conjunto apartado de HAZOP** para medir recall sin sesgo y calificar precisión.
3. **Costo de relleno e internos** + absorbedor en `flowsheet_cost` (hoy solo casco).
4. **Fluxo 1.1 UI en Figma** (tras el veredicto de Apple de 1.0.0).
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
