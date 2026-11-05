# NEXT_STEP — cursor

Actualizado: 2026-11-04 (bloque 15)

## Dónde estamos
**Bloque 15 (2026-11-04): exportación SFILES 2.0 + DEXPI Process** (`notebooks/09_export_sfiles_dexpi.md`): tool `export_flowsheet`
(19 tools) escribe el flowsheet resuelto como SFILES (una línea de topología) y DEXPI XML/JSON con T, P y flujo por corriente;
12/12 flowsheets de referencia con ida y vuelta exacta; 65 tests + 1 skip. Pendiente: XSD oficial de DEXPI y composición en DEXPI.
Antes: bloque 14 bench completo con Sonnet (39 al 100 % entre corridas, regla 5 de signos, kill del grupo de procesos), bloque 13
tarjetas polares, bloques 11–12 NRTL en FluxoEngine. Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Bloque 16 (elegir uno):
1. **Fluxo 1.1 — UI en Figma** (gate obligatorio; requiere al owner en Figma): reactor cinético + selector PR/SRK/NRTL + avisos.
   Conviene esperar el veredicto de Apple de 1.0.0 (en App Review desde 2026-10-19).
2. **HAZOP sobre DEXPI**: que `packages/hazop` lea el modelo DEXPI exportado (no solo el objeto DWSIM) → HAZOP de cualquier
   flowsheet que llegue en DEXPI, aunque no venga de DWSIM. Es el hueco sin competidor del plan (Fase 2).
3. **Estudio semana 6 — Pyomo/IDAES** (modelado orientado a ecuaciones).
4. Pequeño: validar el XML contra el XSD oficial de DEXPI 2.0 y añadir composición por corriente.
Recomendación: 2 mientras Apple decide 1.0.0; 1 cuando el owner tenga una hora para Figma.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
