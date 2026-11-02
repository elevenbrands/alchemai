# Plan: ser el mejor en IA + Ingeniería Química (12 meses)

## Contexto

El owner es ingeniero químico con Python y modelado numérico sólidos, 20 h/semana disponibles, Mac Apple Silicon (sin .NET), Parallels posible pero sin licencia Aspen. Objetivo a 12 meses: **producto propio** que lo posicione como pionero en IA para ingeniería de procesos. Código en inglés, contenido y producto en español.

Activos existentes:
- **Fluxo** (`apps/fluxo`): simulador tipo HYSYS en Swift puro (`Packages/FluxoEngine`). PR/SRK, flash PT/PH/PS, 14 operaciones unitarias, columnas FUG y Wang-Henke, reciclo Wegstein. 1.0.0 (4) en App Review desde 2026-10-19. Huecos: solo reactor de conversión (sin cinética ni equilibrio), 24 componentes, sin NRTL/UNIQUAC, sin formato exportable (SFILES/JSON), constantes sin verificar contra NIST (MOCKS.md naranja).
- **Chemy+** (`apps/chemy_plus`): química escolar; ya usa la capa `factory-ai` (Gemini Flash-Lite, guard determinista).
- **Capa IA de la fábrica**: `app-factory/templates/ai/FactoryAI` + Edge Function `factory-ai`; gate `scripts/app_ai_gate.sh`. Regla: motor determinista = verdad, IA solo propone.
- Notas de investigación sobre HYSYS/DWSIM en `VS Code/research_notes/Cómo funciona Aspen HYSYS/`.
- Idea en portafolio (coach-system): «IA Especializada Ingeniería Química», sin competidor en español.
- Cero Python de ingeniería química en el workspace.

Mapa del terreno (verificado 2026-11-01/02):
- Abierto y maduro: DWSIM 10.x (ya con build Apple Silicon y paquete MCP headless en `DanWBR/dwsim10`), IDAES/Pyomo, OMLT, Cantera 3.2, `thermo`/`chemicals`/`fluids`/`ht`, BioSTEAM, SFILES2 (MIT), pyDEXPI 2.1, pc-gym, Summit.
- Frontera académica sin producto: agentes LLM sobre HYSYS/Aspen (ProcAgent 2026, Nature 2025), autocompletado de flowsheets con transformers (TU Delft), P&ID→grafo con modelos multimodales, ChatP&ID.
- **Huecos sin nadie en código abierto**: HAZOP asistido por LLM sobre grafo de proceso; ruteo de tubería/layout con IA; herramientas en español; móvil.
- Sobre Aspen/E3D/Smart 3D no se puede construir desde fuera (licencias, sin API abierta). Plant 3D sí (COM + SQLite), pero es fase tardía.

## Decisiones de diseño

1. **Producto insignia: agente de diseño de procesos.** Entrada en lenguaje natural → arma el flowsheet → lo corre en un motor determinista → reporta resultados verificados → genera HAZOP preliminar sobre el grafo. Meses 1-6 motor Python + servidor MCP (usable desde Claude Desktop/Claude Code/Cursor). Meses 7-12 UI (web o Mac, decisión del owner en mes 6). Fluxo recibe las mejoras del motor como canal móvil.
2. **Motor de simulación de la fase 1 = DWSIM 10 headless** (ya probado, compara bien con HYSYS). DWSIM es GPLv3: el agente vive en **proceso separado** (MCP), nunca enlazado. No usar `ol-dwsim-mcp-server` (AGPL) como base; escribir servidor MCP propio bajo MIT, aprendiendo de él y del paquete MCP oficial de DWSIM 10.
3. **Verdad = motor + reglas deterministas; la IA solo propone** (misma regla que `AI_FEATURES_PLAYBOOK.md`). Todo output del agente lleva evidencia (balances cerrados, convergencia) o se marca como no verificado.
4. **Benchmark público primero.** Quien publica el benchmark define el campo: 20 casos de diseño con respuesta conocida (separación flash, columna FUG, reactor CSTR/PFR, reciclo, intercambiador) en inglés con descripción en español. Mide a cualquier agente, incluido el propio.
5. **Repo nuevo fuera de `apps/`** (no es app iOS): `VS Code/<nombre>/` como monorepo Python con `uv`. El nombre del producto lo elige el owner (identidad = decisión del owner); placeholder en este plan: `proceso-ia`.
6. **Python 3.12 principal** (`uv python install 3.12`): cumple biosteam ≥3.12, pydexpi ≥3.12, cantera <3.15, SFILES2, idaes. `pcgym` exige 3.11 → venv aparte solo para fase 3.

## Ritmo semanal (20 h)

| Bloque | Horas | Qué |
|---|---|---|
| Estudio dirigido | 6 | Un tema por semana, siempre con un notebook que reproduce un resultado conocido |
| Construcción | 10 | El producto, en bloques cerrados con commit+push |
| Publicar | 2 | Un post en español/semana (LinkedIn o blog) con una evidencia real; README en inglés |
| Validar | 2 | Comparar contra DWSIM/literatura; registrar en `BENCHMARK.md` |

Regla de la fábrica aplicada: una sesión = un sub-bloque; `APP.md`/`NEXT_STEP.md` del repo nuevo como verdad técnica y cursor.

## Fase 0 · Entorno y línea base (semanas 1-2)

Hecho cuando:
- `VS Code/proceso-ia/` creado con `uv`, Python 3.12, deps: `thermo chemicals fluids ht cantera idaes-pse omlt biosteam SFILES2 pydexpi mcp fastmcp`. `uv run python -c "import cantera, thermo, idaes, SFILES2, pydexpi"` pasa.
- DWSIM 10.x Apple Silicon instalado desde `DanWBR/dwsim10` (verificar firma/procedencia; descargar a carpeta vacía propia). Paquete MCP headless de DWSIM corriendo un flowsheet de ejemplo desde Python. Si el headless no corre en macOS, plan B: DwsimPy (`volkan-a/dwsimpy`, .NET 10) o Windows 11 ARM en Parallels (compra = OK del owner).
- **Línea base Fluxo vs DWSIM**: 3 casos (flash PT de gas natural, columna FUG, compresor + enfriador) con desviación <1 % en flujos y temperaturas. Esto es a la vez el primer notebook y QA real de Fluxo; lo que no cuadre entra en `apps/fluxo/MOCKS.md`.
- `APP.md`, `NEXT_STEP.md`, `MOCKS.md`, `BENCHMARK.md` creados; repo en GitHub (privado hasta el primer release público).

## Fase 1 · Fundamentos + motor agentico (meses 1-4)

**Estudio** (un tema/semana, cada uno cierra con notebook reproducible):
1. Flash y estabilidad de fases (Michelsen) con `thermo`; comparar con Fluxo.
2. Modelos de actividad NRTL/UNIQUAC/Wilson con `thermo` → después portar a FluxoEngine (fase E4 de Fluxo).
3. Cinética y redes de reactores con Cantera (CSTR, PFR, batch; Fogler caps. 1-8 como texto).
4. Columnas rigurosas: inside-out y Wang-Henke; ver dónde falla Fluxo.
5. Modelado orientado a ecuaciones con Pyomo/IDAES (tutoriales oficiales IDAES).
6. Optimización de diseño: formulación NLP/MINLP (Biegler, *Nonlinear Programming*, caps. 1-6 y 11).
7. Modelos sustitutos: entrenar red sobre datos de DWSIM, embeber con OMLT en Pyomo.
8. Flowsheets como texto/grafo: SFILES2 y DEXPI (pyDEXPI); leer Vogel et al. 2023 y los papers de Schweidtmann.
9-16. Agentes: MCP, evaluación de agentes, tool-use determinista; leer ProcAgent (SSRN 2026) y el paper de Nature 2025.

**Construcción:**
- `packages/dwsim-mcp/`: servidor MCP propio (MIT) sobre DWSIM headless: `create_flowsheet`, `add_compound`, `add_unit`, `connect`, `set_spec`, `run`, `get_stream`, `mass_balance_check`, `export_sfiles`. Tests contra 10 flowsheets de referencia.
- `packages/bench/`: **ChemE Design Bench v0** con 20 casos (yaml: enunciado ES/EN, propiedades esperadas, tolerancias). Runner que puntúa a cualquier agente MCP.
- `packages/agent/`: agente con Claude (modelo por defecto `claude-sonnet-5-5`, escalar a Opus si el bench lo pide) que resuelve el bench. Objetivo mes 4: ≥80 % del bench con balances cerrados.
- Fluxo 1.1 (gates de la fábrica): reactor cinético (validado contra Cantera), NRTL, exportación JSON/SFILES del `DefinicionCaso`, componentes verificados contra NIST (cierra naranja de MOCKS.md).

**Publicar:** repo público al cerrar el bench v0 (mes 3). Post semanal en español con una evidencia (gráfica Fluxo vs DWSIM, caso del bench resuelto).

## Fase 2 · Diferenciación: el hueco vacío (meses 5-8)

**Estudio:** análisis de riesgo (HAZOP, LOPA; CCPS *Guidelines for Hazard Evaluation Procedures*), dinámica de procesos y control (Seborg caps. 1-12), P&ID como datos (DEXPI 1.3/2.0), redes de grafos para procesos.

**Construcción:**
- `packages/hazop/`: **HAZOP preliminar automático** sobre el grafo (SFILES/DEXPI): reglas deterministas de desviación (más flujo, menos flujo, más presión, más temperatura, reverso, contaminación) por tipo de nodo + LLM solo para redactar causas/consecuencias/salvaguardas candidatas, siempre marcadas como propuesta. Salida: tabla HAZOP en español, exportable a Excel. Nadie tiene esto abierto.
- Agente v1: diseño → simulación → crítica (grados de libertad, especificaciones imposibles, violaciones de pinch simples) → HAZOP preliminar. Bench v1 añade 10 casos con "trampas" (especificaciones inconsistentes) para medir que el agente detecta errores en vez de inventarlos.
- Fluxo 1.2: módulo «Explica mi flowsheet» con la capa `factory-ai` (guard = resultados del motor). Primer uso de IA dentro de Fluxo, respetando `app_ai_gate.sh`.
- Decisión del owner (mes 6): UI web (Supabase + Stripe, stack ya usado en Unifit/Portik) o app Mac (SwiftUI, reutilizando FluxoEngine como visor).

**Publicar:** whitepaper en inglés y español «LLM agents for process design: a verified benchmark»; video demo del HAZOP automático.

## Fase 3 · Pionero (meses 9-12)

**Estudio:** optimización bayesiana (Summit) y RL para control (pc-gym, venv 3.11); redes neuronales informadas por física (PINNs) para reactores; lectura selectiva de layout/ruteo (A*, MARL) para decidir si se abre ese frente.

**Construcción:**
- Optimización de diseño cerrada por el agente: barrido/BO sobre DWSIM para costo mínimo (CAPCOST en Python, basado en Turton) con restricciones de seguridad del HAZOP.
- UI elegida en mes 6, con cobro. Suscripción según estándar de la fábrica si es app; precios de SaaS = decisión del owner.
- Opcional si hay tiempo: exportador DEXPI → DXF/Plant 3D (Windows en Parallels, trial de Plant 3D) para cerrar el ciclo hasta el P&ID.

**Publicar:** bench v2 público con tabla de resultados de varios modelos; presentar en un foro (AMIDIQ, AIChE en español, IMIQ) = OK del owner.

## Lecturas base (comprar/consultar; no leer completas)
- Fogler, *Elements of Chemical Reaction Engineering* (reactores).
- Seader, Henley, Roper, *Separation Process Principles* (columnas, flash).
- Biegler, *Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes*.
- Turton et al., *Analysis, Synthesis and Design of Chemical Processes* (costos, HAZOP, Cap. 26).
- Seborg et al., *Process Dynamics and Control*.
- Papers: Vogel et al. 2023 (SFILES/transformer), Schweidtmann 2024-2026 (P&ID, pyDEXPI), ProcAgent 2026, Nature Comms Eng 2025 (agente + Aspen), PC-Gym 2024.

## Lo que requiere OK del owner (no decidir solo)
- Nombre e identidad del producto y del repo.
- Compra de Parallels/Windows 11 y cualquier licencia (Plant 3D trial es gratis 30 días).
- Hacer público el repo y publicar el whitepaper/bench.
- Precios y cobro de la UI (fase 3).
- Envío de Fluxo 1.1/1.2 a App Review.

## Verificación (cómo saber que el plan va bien)
- Fase 0: comando de imports pasa; DWSIM corre un flowsheet desde Python; tabla Fluxo vs DWSIM con 3 casos <1 %.
- Mes 4: bench v0 público, agente ≥80 % con balances cerrados; Fluxo 1.1 con reactor cinético validado contra Cantera en TestFlight.
- Mes 8: HAZOP automático genera una tabla correcta para 5 procesos de libro (comparar contra HAZOP publicados en Turton/CCPS); agente detecta ≥90 % de las trampas del bench v1.
- Mes 12: UI con al menos 10 usuarios reales probándola; bench v2 con resultados de ≥3 modelos; 40+ posts publicados.

## Primer paso al aprobar
Crear `VS Code/proceso-ia/` (nombre provisional) con `uv`, Python 3.12 y las dependencias; instalar DWSIM 10 Apple Silicon desde releases de `DanWBR/dwsim10`; correr el primer flowsheet headless desde Python. Un solo sub-bloque, cierra con commit.
