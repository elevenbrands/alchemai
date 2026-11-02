# AlchemAI

An AI agent for chemical process design that **never guesses**: it builds the flowsheet,
solves it in a deterministic engine (DWSIM 10 via MCP), reports only verified results
(closed balances, convergence) and drafts a preliminary HAZOP on the process graph.

Status: phase 1. DWSIM 10 runs headless on macOS via the Fluent API (pythonnet); `packages/dwsim_client/verify.py` closes mass, compound and energy balances per unit (reactors included); **ChemE Design Bench v0** has 20 bilingual cards (13 simulation, 7 design) with a JSON agent protocol and an oracle that scores 100 %. See `APP.md` (technical truth) and `NEXT_STEP.md` (cursor).

```bash
uv sync
uv run python notebooks/smoke_stack.py
uv run pytest -q                                   # 22 tests
uv run python packages/bench/runner.py --agent oracle   # Bench v0, 20 cards
```

**MCP server** (use DWSIM from Claude Desktop / Claude Code / Cursor):

```json
{"mcpServers": {"alchemai": {"command": "uv", "args": ["run", "--directory", "/path/to/alchemai", "python", "-m", "packages.dwsim_mcp.server"]}}}
```

**Agent** (Claude + the same tools, Bench v0 protocol): `uv run python packages/bench/runner.py --agent claude bomba`
(needs `ANTHROPIC_API_KEY` in the environment or in the macOS Keychain, service `ANTHROPIC_API_KEY`, account `alchemai`).

License: MIT (this repo). DWSIM is GPL-3.0 and runs as a separate process.
