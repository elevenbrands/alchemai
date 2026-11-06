# AlchemAI

An AI agent for chemical process design that **never guesses**: it builds the flowsheet, solves it in a
deterministic engine (DWSIM 10, headless, through MCP tools), reports only verified results (closed balances,
convergence), prices the equipment, optimizes the design under the limits of its own preliminary HAZOP, and drafts
that HAZOP on the process graph. Spanish-first content, English code. Status: month 1 of a 12-month plan
(`docs/PLAN_12_MESES.md`); the technical truth is `APP.md`, the cursor is `NEXT_STEP.md`, the evidence log is
`BENCHMARK.md`, and every notebook in `notebooks/` reproduces one result against a book, a paper or the simulator.

## What it does today

| Capability | Evidence |
|---|---|
| **36 MCP tools over DWSIM 10** (streams, 20 unit kinds incl. kinetic CSTR/PFR, rigorous columns, absorbers, pipes, recycle; `dof_check`, `sweep`, export to SFILES 2.0 / DEXPI) | `packages/dwsim_client/`, `packages/dwsim_mcp/server.py` |
| **ChemE Design Bench**: 50 bilingual cards — 20 feasible designs, 15 traps the agent must *detect* (impossible specs, under/over-specification), 4 polar systems without the package in the statement, 11 transport / costing / optimization cards | `packages/bench/`, `notebooks/bench_runs/` — Claude Opus and Sonnet at 100 % on every card measured (traps: 15/15 with simulator evidence, 0 false alarms) |
| **Preliminary HAZOP** from the solved flowsheet or from a third-party **DEXPI** file: deterministic rules instantiated with the numbers (hydrates, runaway, blocked-in liquid, vacuum, decomposition, condensation-induced hammer…), plus an optional LLM proposal layer that never overwrites a rule row | Validated against 10 published HAZOPs: 5 used to write rules (Lawley/Kletz, CCPS DAP, Crowl-Louvar, CSB Texas City, Williams) and 5 **held out** (blind recall 83 %, precision 76 % useful / 0 spurious rows after review) — `notebooks/11_*`, `notebooks/19_*` |
| **Transport**: pipe ΔP (Darcy-Colebrook), line sizing, exchanger area (LMTD·F, Gnielinski) | ≤ 1.4 % vs DWSIM pipe segments, area to 0.2 % — `notebooks/12_*` |
| **Mass transfer and column hydraulics**: Kremser, Colburn, Onda 1968, Stichlmair 1989 | Kremser = rigorous DWSIM absorber ±0.6 points; Stichlmair = the paper's example — `notebooks/16_*`, `17_*` |
| **Costing**: Seider et al. (SSLW) via IDAES for exchangers, pumps, compressors, vessels and trays; Lang factors; CEPCI; annualized cost | IDAES reference values exact; tray formula reproduced by hand — `notebooks/13_*`, `14_*`, `20_*` |
| **Design optimization**: 1-4 continuous variables (grid + Nelder-Mead) and integer ones (column stages, rebuilt in DWSIM) with HAZOP limits as hard constraints | 3-stage compression = theory ±1 %; ethanol column optimum at R/R_min = 1.12 (heuristic 1.1-1.5) — `notebooks/15_*`, `18_*`, `20_*` |

Prices, diameters and packing data used in the notebooks are illustrative and listed in `MOCKS.md`; the tools
require the user's own values and never ship defaults.

## Run it

```bash
uv sync                                   # Python 3.12; DWSIM 10 (arm64) installed separately, see APP.md
uv run python notebooks/smoke_stack.py
PYTHONPATH=. uv run pytest -q             # 110 tests (DWSIM headless required)
PYTHONPATH=. uv run python -m packages.bench.runner --agent oracle              # all 50 cards with the oracle
PYTHONPATH=. uv run python -m packages.bench.runner --agent claude-code --version v1   # the traps with the agent
```

**MCP server** (use DWSIM from Claude Desktop / Claude Code / Cursor):

```json
{"mcpServers": {"alchemai": {"command": "uv", "args": ["run", "--directory", "/path/to/alchemai", "python", "-m", "packages.dwsim_mcp.server"]}}}
```

**Agent**: `claude-code` runs through the Claude Code CLI; `claude` uses the SDK and needs `ANTHROPIC_API_KEY`
in the environment or in the macOS Keychain (service `ANTHROPIC_API_KEY`, account `alchemai`).

## Principles

- The engine and the rules are the truth; the model only proposes. Every number the agent reports carries its
  evidence (balances, convergence) or is marked unverified.
- Constants come from a cited source or are inputs. No invented data.
- References are written and committed **before** the first run (HAZOP sets, bench answers), so the first score
  is blind and stays in the log even after the rules improve.

License: MIT (this repository). DWSIM is GPL-3.0 and always runs as a separate process; nothing from it is bundled.
