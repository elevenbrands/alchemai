# AlchemAI — verdad técnica

> Nombre confirmado por el owner el 2026-11-02: **AlchemAI** (alquimia + AI). Repo `elevenbrands/alchemai`.

## Qué es

Agente de diseño de procesos químicos: recibe una descripción en lenguaje natural, arma el
flowsheet, lo resuelve en un motor determinista (DWSIM 10), reporta resultados **verificados**
(balances cerrados, convergencia) y genera un HAZOP preliminar sobre el grafo del proceso.
Plan completo (12 meses): `docs/PLAN_12_MESES.md`.

Reglas:
- **Verdad = motor + reglas deterministas. La IA solo propone.** Todo resultado lleva evidencia
  o se marca «no verificado».
- Código y docs técnicas en inglés para la comunidad; contenido y producto en español.
- Licencia propia MIT. DWSIM es GPL-3.0 → siempre proceso separado (MCP/stdio), nunca enlazado.

## Stack

| Capa | Herramienta | Estado |
|---|---|---|
| Python | 3.12.13 vía `uv` (`.python-version`) | ✅ |
| Termodinámica | `thermo` 0.6.1, `chemicals`, `fluids`, `ht` | ✅ flash PR verificado (VF=0.7426 C1/C2/C3 230 K 30 bar) |
| Cinética | `cantera` 3.2.0 | ✅ GRI30 batch verificado |
| Modelado/optimización | `idaes-pse` 2.13.0 + Pyomo, `omlt` 1.2.2 | ✅ importa |
| TEA | `biosteam` 2.54.0 | ✅ importa |
| Flowsheet como texto/grafo | `SFILES2` 1.2.0 (módulo `Flowsheet_Class`), `pydexpi` 2.1.0 | ✅ importa |
| MCP | `mcp` 2.3.0, `fastmcp` 4.1.5 | ✅ |
| Motor de simulación | DWSIM 10.2.12 arm64 en `~/Applications/DWSIM10/`, **Fluent API en proceso vía pythonnet + coreclr 10** (`packages/dwsim_client/fluent.py`) | ✅ headless verificado (mezclador y flash PR) |
| Servidor MCP de DWSIM (opcional) | `~/tools/dwsim10/tools/DWSIM.MCPServer` (58 tools reales en fuente: flowsheet, thermo, streams, unitops, reactions, solve, dynamics, scenarios), stdio o SSE | ⏳ opcional: la Fluent API ya cubre fase 0-1; compilar solo si el agente lo necesita como proceso MCP |
| .NET 10 SDK | 10.0.403 user-level en `~/.dotnet` (dotnet-install.sh oficial, sin sudo) | ✅ |

Hallazgo clave (2026-11-02): el repo oficial `DanWBR/dwsim10` ya trae un servidor MCP oficial
(README dice 27 tools; en fuente hay 58 `[McpTool]`, incl. dinámica y escenarios). **No escribimos nuestro propio
servidor MCP sobre DWSIM**; nuestro código MIT es el agente, el bench y el HAZOP. El tarball de
macOS no incluye el servidor compilado (solo `.deb` Linux), por eso se compila desde fuente.

## Estructura

```
packages/dwsim_client/  fluent.py (carga DWSIM en proceso), verify.py (balances + convergencia), smoke*.py
packages/bench/         cases_baseline.py (3 casos), cases_ref.py (10 casos con enunciado ES/EN, readouts y chequeos de libro), run_ref.py, expected_ref.yaml (snapshot DWSIM)
packages/agent/         agente (Claude) que resuelve el bench vía MCP
packages/hazop/         HAZOP preliminar sobre grafo SFILES/DEXPI (fase 2)
notebooks/              estudio semanal, un notebook = un resultado reproducido
tests/
docs/                   plan, decisiones, lecturas
```

## Decisiones

- 2026-11-02: las entalpías de corriente de DWSIM excluyen la entalpía de formación (referencia: compuestos puros a 25 °C). Para reactores `verify.py` suma Σ Δn·ΔHf (de `ConstantProperties.IG_Enthalpy_of_Formation_25C`); con eso el balance cierra a 1e-13 kW. Intercambiador: Q es interno, el balance externo es ΔH ≈ 0.
- 2026-11-02: compresor politrópico requiere `WithProcessPath(Compressor.ProcessPathType.Polytropic)`; sin eso DWSIM calcula adiabático. Reactores: productos vapor puerto 0 y líquido puerto 1 obligatorios; energía como *feed* puerto 1. Reciclo: tolerancias por defecto (0.01 kg/s) dejan 0.04 % de error; se aprietan vía `ConvergenceParameters`.
- 2026-11-02: `verify.py` lee el modelo de objetos (`Inner.SimulationObjects`): el diccionario entrega `ISimulationObject`; las propiedades concretas (`DeltaQ`, `EnergyFlow`) se leen por reflexión y las corrientes por `fs.MaterialStream(tag).Object`. Tolerancia de balance 1e-4 (DWSIM cierra flash/shortcut a ~1e-5).
- 2026-11-02: nombre **AlchemAI** confirmado por el owner. Libre en PyPI y en `elevenbrands/`; en GitHub solo repos ajenos de 0-1 ★ (un juego, un newsletter). Marcas registradas no revisadas. Remoto privado `elevenbrands/alchemai` hasta el primer release público (OK del owner).

- 2026-11-02: Python 3.12 (biosteam y pydexpi exigen ≥3.12; cantera <3.15). `pcgym` (3.11) irá en
  venv aparte en fase 3.
- 2026-11-02: no usar `ol-dwsim-mcp-server` (AGPL). Usar el MCP oficial de DWSIM 10 (GPL, proceso
  separado).
- 2026-11-02: ruta principal al motor = Fluent API de DWSIM 10 cargada con pythonnet (runtime .NET 10 user-level, runtimeconfig propio con rollForward LatestMajor). El MCP oficial queda como alternativa cuando haga falta aislar el proceso.
- 2026-11-02: la Fluent API conecta energía del rehervidor de la Shortcut Column como *energy feed* puerto 1 y la del condensador como *energy product* puerto 0; el compresor no acepta energy feed puerto 0 → leer `builder.Object.DeltaQ` (kW).
- 2026-11-02: DWSIM instalado fuera del repo (`~/Applications/DWSIM10`), fuente en `~/tools/dwsim10`.

## Gates (adaptados; no es app iOS)

G0 Entorno ✅ · G1 Línea base Fluxo vs DWSIM ✅ (2026-11-02; balances/energía < 0.3 %, FUG N/R y k_ij CO₂ anotados en Fluxo) · G2 verify.py + 10 flowsheets ✅ (2026-11-02) ·
G3 Bench v0 (20 casos) ⏳ · G4 Agente ≥80 % bench ⏳ · G5 HAZOP ⏳ · G6 UI ⏳ · G7 Público/cobro (OK owner) ⏳
