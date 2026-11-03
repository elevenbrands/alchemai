# NEXT_STEP — cursor

Actualizado: 2026-11-02

## Dónde estamos
**Fase 2, bloque 6 cerrado (2026-11-02)**: capa LLM del HAZOP (`packages/hazop/llm.py`, filas `proposal`
vía `claude -p`, 12 propuestas / 0 nodos inventados en la compresión de gas) + **Bench v1 con 10 trampas**
(`packages/bench/trap_cases.py`): el agente detecta 10/10 con evidencia del simulador y 0 falsas alarmas en
6 controles v0. 40 tests (39 + 1 live skip). Evidencia en `BENCHMARK.md` y `notebooks/bench_runs/v1_*.jsonl`.

## Siguiente acción exacta
Bloque 7 (elegir uno, en orden sugerido):
1. **Estudio semana 3 — cinética y redes de reactores con Cantera** (notebook 03: CSTR/PFR/batch, comparar
   con DWSIM y con el reactor de conversión; base del reactor cinético de Fluxo 1.1).
2. **Agente v1 — crítica de grados de libertad** antes de simular: contar especificaciones vs incógnitas por
   unidad y avisar «sobre/sub-especificado» como herramienta determinista (`dof_check`), con 5 tarjetas más.
3. **Fluxo 1.2 «Explica mi flowsheet»** con la capa `factory-ai` (gate `app_ai_gate.sh`) — requiere OK del
   owner para tocar Fluxo mientras 1.0.0 está en App Review.
4. Barato: medir el bench completo (30) con `ALCHEMAI_CC_MODEL=sonnet` para tener una segunda fila de modelo.

## Bloqueante actual
Ninguno. Limitación anotada: Peng-Robinson no condensa el 1 % de agua a 280 K / 60 bar (liq = 0 kg/s en V-1);
para HAZOP de gas húmedo conviene un paquete con agua libre (CPA / SRK con agua) — pendiente de evaluar.

## Decisiones pendientes del owner
- Hacer público el repo (plan: mes 3) y primer post.
- UI en mes 6: web (Supabase + Stripe) vs app Mac (SwiftUI + FluxoEngine).
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
