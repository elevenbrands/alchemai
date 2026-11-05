# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 22)

## Dónde estamos
**Bloque 22 (2026-11-05): transferencia de masa** (`notebooks/16_transferencia_de_masa.md`): Kremser, Colburn, Onda 1968
(verificado contra el artículo), altura de relleno, difusividades; kind `absorption_column`; 33 tools. Kremser = absorbedor
riguroso de DWSIM dentro de 0.6 puntos en 9 casos. Bench v3 = 8 tarjetas (47 en total). Con esto fenómenos de transporte queda
cubierto en momento, calor y masa. Bloques 18-21 del mismo día: transporte, costo de compra, costo anual, optimizador.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Opciones (el owner elige):
1. **Hidráulica de columnas**: inundación y caída de presión de rellenos (Stichlmair de `fluids`, ya instalado) y diámetro de
   columna; cierra el dimensionamiento del absorbedor y permite costearlo.
2. **Optimización de dos variables** (coordenadas) sobre `optimize_design`.
3. **Conjunto apartado de HAZOP** para medir recall sin sesgo y calificar precisión.
4. **Fluxo 1.1 UI en Figma** (tras el veredicto de Apple de 1.0.0).
Decisiones del owner abiertas: repo público + primer post; UI (web vs Mac).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
