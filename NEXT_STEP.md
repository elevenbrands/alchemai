# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 17)

## Dónde estamos
**Bloque 17 (2026-11-05): HAZOP vs libro** (`notebooks/11_hazop_vs_libro.md`): 5 procesos con HAZOP publicado o investigación CSB,
escritos a mano como DEXPI (`packages/hazop/book/`). Referencias commiteadas antes de correr. Línea base a ciegas 9/22 = 41 %;
con 8 grupos de reglas genéricas nuevas, 22/22 (no independiente). 4 escenarios fuera de alcance de un DEXPI de proceso.
83 tests + 1 skip. Antes: bloque 16 HAZOP sobre DEXPI, 15 export SFILES2/DEXPI, 14 bench Sonnet. Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Lo que más mueve el proyecto ahora son decisiones del owner, no más bloques internos:
1. **Hacer público el repo + primer post** (bench + HAZOP vs libro como evidencia). Requiere OK del owner.
2. **Decidir la UI** (web Supabase + Stripe vs app Mac) y pasar por Figma.
Bloques técnicos que siguen si el owner prefiere seguir construyendo:
3. **HAZOP: conjunto apartado** de 3–5 HAZOP publicados nuevos, sin tocar reglas, para medir recall independiente; y calificar precisión (filas que sobran).
4. **DEXPI de planta (nivel P&ID)**: cubriría B4 (destino del alivio) y archivos de CAD.
5. **Fluxo 1.1 UI en Figma** tras el veredicto de Apple de 1.0.0.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
