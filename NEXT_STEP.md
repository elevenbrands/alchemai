# NEXT_STEP — cursor

Actualizado: 2026-11-04 (bloque 16)

## Dónde estamos
**Bloque 16 (2026-11-04): HAZOP preliminar sobre DEXPI** (`notebooks/10_hazop_sobre_dexpi.md`): tool `hazop_from_dexpi` (20 tools) hace el
HAZOP de un archivo DEXPI Process sin simular; 13/13 flowsheets dan el mismo HAZOP desde DWSIM que desde su DEXPI; funciona con un DEXPI
hecho a mano (°C, bar, kg/h) y reporta en `missing` lo que el archivo no trae. El DEXPI exportado ya lleva composición. Regla nueva de
sobreenfriamiento con hidratos en enfriadores. 81 tests + 1 skip. Antes: bloque 15 export SFILES2/DEXPI, bloque 14 bench Sonnet, bloque 13
tarjetas polares. Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Bloque 17 (elegir uno):
1. **Fluxo 1.1 — UI en Figma** (gate obligatorio; requiere al owner en Figma): reactor cinético + selector PR/SRK/NRTL + avisos.
   Conviene esperar el veredicto de Apple de 1.0.0 (en App Review desde 2026-10-19).
2. **HAZOP de 5 procesos de libro** contra HAZOP publicados (Turton cap. 26 / CCPS): es la verificación del mes 8 del plan y mide
   qué tan bueno es el HAZOP, no solo que corra.
3. **DEXPI de planta (nivel P&ID)**: leer equipos y tuberías de un P&ID exportado por un CAD, para aceptar archivos de otras herramientas.
4. **Estudio semana 6 — Pyomo/IDAES**.
Recomendación: 2 (convierte el HAZOP en evidencia publicable); 1 cuando el owner tenga una hora para Figma.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
