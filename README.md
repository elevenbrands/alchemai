# AlchemAI

An AI agent for chemical process design that **never guesses**: it builds the flowsheet,
solves it in a deterministic engine (DWSIM 10 via MCP), reports only verified results
(closed balances, convergence) and drafts a preliminary HAZOP on the process graph.

Status: phase 2. DWSIM 10 runs headless on macOS via the Fluent API (pythonnet); `packages/dwsim_client/verify.py` closes mass, compound and energy balances per unit; **ChemE Design Bench** has 30 bilingual cards: 20 feasible (v0: 13 simulation, 7 design) + 10 **traps** (v1: infeasible specs the agent must detect, not solve). The AlchemAI agent (Claude + 16 DWSIM tools) scores 100 % on v0 and detects 10/10 traps with simulator evidence and 0 false alarms on controls. A rule-based **preliminary HAZOP** plus an LLM proposal layer runs on any solved flowsheet. See `APP.md` (technical truth) and `NEXT_STEP.md` (cursor).

```bash
uv sync
uv run python notebooks/smoke_stack.py
uv run pytest -q                                   # 22 tests
uv run python packages/bench/runner.py --agent oracle              # all 30 cards
uv run python packages/bench/runner.py --agent claude-code --version v1   # the 10 traps with the agent
```

**Preliminary HAZOP** (`packages/hazop/`): deterministic deviation rules per unit, instantiated with the solved numbers (hydrates, runaway, relief cases, blowby...), plus an optional LLM pass (`hazop_preliminary(..., with_proposals=True)`, rows tagged `proposal`, never replacing rule rows), exported to Markdown/Excel and exposed as the `hazop_preliminary` tool. A rule-based draft, not a substitute for a HAZOP team.

**MCP server** (use DWSIM from Claude Desktop / Claude Code / Cursor):

```json
{"mcpServers": {"alchemai": {"command": "uv", "args": ["run", "--directory", "/path/to/alchemai", "python", "-m", "packages.dwsim_mcp.server"]}}}
```

**Agent** (Claude + the same tools, Bench v0 protocol): `uv run python packages/bench/runner.py --agent claude bomba`
(needs `ANTHROPIC_API_KEY` in the environment or in the macOS Keychain, service `ANTHROPIC_API_KEY`, account `alchemai`).

License: MIT (this repo). DWSIM is GPL-3.0 and runs as a separate process.
