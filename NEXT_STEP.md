# NEXT_STEP — cursor

Actualizado: 2026-11-05 (bloque 18)

## Dónde estamos
**Bloque 18 (2026-11-05): fenómenos de transporte** (`notebooks/12_fenomenos_transporte.md`): `packages/transport/hydraulics.py`
+ 5 tools (25 en total) + kind `pipe`. ΔP de tubería = Pipe Segment de DWSIM dentro de 1.4 % en 6 casos; el área de intercambiador
recupera la de DWSIM a 0.2 %. Bench v3 = 3 tarjetas de transporte (42 tarjetas). Antes: bloque 17 HAZOP vs libro (41 % a ciegas → 22/22),
16 HAZOP sobre DEXPI, 15 export SFILES2/DEXPI. Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
**Bloque 19 — costo de equipos** (fase 3, ya desbloqueado por el dimensionamiento): costo de compra y de módulo desnudo de bomba,
compresor, intercambiador, recipiente y columna a partir de área/potencia/volumen, con CEPCI. Usar correlaciones ya implementadas
en código abierto (costeo de IDAES basado en Seider, o BioSTEAM), no coeficientes de memoria; validar contra un ejemplo publicado.
Tool `equipment_cost` + tarjeta del bench. Cubre a la vez el estudio de Pyomo/IDAES (semana 5).
Decisiones del owner que siguen abiertas: repo público + primer post; UI (web vs Mac).
Después: conjunto apartado de HAZOP (recall independiente y precisión), DEXPI nivel P&ID, Fluxo 1.1 UI en Figma.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
