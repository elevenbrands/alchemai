# AlchemAI

An AI agent for chemical process design that **never guesses**: it builds the flowsheet,
solves it in a deterministic engine (DWSIM 10 via MCP), reports only verified results
(closed balances, convergence) and drafts a preliminary HAZOP on the process graph.

Status: phase 0 (environment + baseline). See `APP.md` (technical truth) and `NEXT_STEP.md` (cursor).

```bash
uv sync
uv run python notebooks/smoke_stack.py
```

License: MIT (this repo). DWSIM is GPL-3.0 and runs as a separate process.
