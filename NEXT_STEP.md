# NEXT_STEP — cursor

Actualizado: 2026-11-03

## Dónde estamos
**Bloque 10 (2026-11-03): estudio de columnas rigurosas cerrado** (`notebooks/04_columnas_rigurosas.md`): el Wang-Henke de
Fluxo valida contra DWSIM en la depropanizadora (1 K / 5 % impureza clave); etanol-agua con PR no converge en ningún motor
(termodinámica: NRTL necesario en Fluxo). Hallazgo lateral corregido: `WithFeed` de DWSIM es 0-based → `tools.py` 1-based,
tarjeta `columna_rigurosa` reconstruida (x_D 0.8040). Antes: bloque 9 reactor cinético de Fluxo (rama `feature/reactor-cinetico`,
ahora con `LineaBaseColumnasTests`), bloque 8 `dof_check` + Bench v1b. Bench = 35 tarjetas, agente 100 % en todo lo medido.
48 tests + 1 live skip.

## Siguiente acción exacta
Bloque 11 (elegir uno):
1. **Fluxo 1.1 — NRTL en FluxoEngine (E4)**: portar el NRTL del notebook 02 (parámetros ChemSep etanol-agua, agua-metanol…) al
   paquete de Fluxo con K = γ·Psat/P (γ-φ) y repetir el caso B del notebook 04 hasta reproducir x_D 0.8040 de DWSIM. Solo motor;
   sin UI ni envío (1.0.0 en App Review).
2. **Fluxo 1.1 — UI del reactor cinético** en Figma (gate obligatorio) → `DefinicionCaso`/inspector; merge con OK del owner.
3. **Segunda fila de modelo**: bench completo (35) con `ALCHEMAI_CC_MODEL=sonnet` (~20 min).
4. **Export SFILES2/DEXPI** del flowsheet de la sesión.
Recomendación: 1 (cierra el hueco más grande que mostró el estudio y es solo motor).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); columna PR con mezcla polar puede no terminar (correr con `WithMaxIterations` o límite de tiempo).

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
