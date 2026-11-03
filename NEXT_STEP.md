# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Fase 2, bloque 7 cerrado (2026-11-02)**: estudio de cinética con Cantera (notebook 03: A→B analítico vs Cantera vs
DWSIM, llama GRI-Mech vs reactor de conversión +116 K, WSR blow-out, retardo de ignición) y **reactores cinéticos en
las herramientas** (`define_kinetic_reaction`, `cstr`, `pfr`; 17 tools; PFR/CSTR dentro de 0.03 % del analítico).
Sección D del notebook = especificación del reactor cinético de Fluxo 1.1. 43 tests + 1 live skip.

## Siguiente acción exacta
Bloque 8 (elegir uno):
1. **Agente v1 — `dof_check`**: herramienta determinista de grados de libertad por unidad (specs vs incógnitas) que avise
   «sobre/sub-especificado» antes de `solve`, + 5 tarjetas (v1b) que lo exijan; reutiliza el protocolo `infeasible`.
2. **Fluxo 1.1 — reactor cinético** siguiendo la sección D del notebook 03 (CSTR algebraico + PFR RK4, Arrhenius,
   validación obligatoria contra X_PFR 0.9807 / X_CSTR 0.7979). Requiere OK del owner para tocar Fluxo mientras 1.0.0
   está en App Review; se puede preparar en rama `feature/reactor-cinetico` sin enviar.
3. **Estudio semana 4** — columnas rigurosas (inside-out / Wang-Henke): comparar la columna etanol-agua de DWSIM
   con la de Fluxo caso a caso y localizar dónde falla Fluxo.
4. Barato: bench completo (30) con `ALCHEMAI_CC_MODEL=sonnet` como segunda fila de modelo.

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar (HAZOP de gas húmedo → paquete con agua libre);
`CSTRBuilder.WithIsothermalTemperature` roto en DWSIM 10.2.12 (usar `Isothermal()`).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
