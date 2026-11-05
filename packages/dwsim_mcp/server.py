"""AlchemAI MCP server (MIT) — exposes the DWSIM tool registry over stdio so Claude Desktop, Claude
Code, Cursor or any MCP client can build, solve and *verify* flowsheets.

DWSIM (GPL-3.0) is loaded in this process only; the MCP client lives in another process.

Run:   uv run python -m packages.dwsim_mcp.server          (stdio)
Claude Desktop config:
    {"mcpServers": {"alchemai": {"command": "uv", "args": ["run", "--directory", "<repo>", "python", "-m",
                                 "packages.dwsim_mcp.server"]}}}
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastmcp import FastMCP  # noqa: E402

from packages.dwsim_client import tools as T  # noqa: E402

mcp = FastMCP("alchemai", instructions=(
    "Deterministic chemical-process simulation (DWSIM 10). Build a flowsheet (new_flowsheet → streams → "
    "units → connect), then ALWAYS call solve(): it returns the balance verification. Report only numbers "
    "from a solve with ok=true; mark everything else as unverified."))

TRACE = os.environ.get("ALCHEMAI_MCP_TRACE")      # optional: append one JSON line per tool call (debugging hangs)


def _traced(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.time()
        with open(TRACE, "a") as f:
            f.write(json.dumps({"t": t0, "tool": fn.__name__, "args": k}, default=str)[:2000] + "\n")
        try:
            out = fn(*a, **k)
            res = "ok"
            return out
        except Exception as ex:
            res = f"{type(ex).__name__}: {ex}"[:500]
            raise
        finally:
            with open(TRACE, "a") as f:
                f.write(json.dumps({"tool": fn.__name__, "done_s": round(time.time() - t0, 2), "result": res}) + "\n")
    return w


for fn in T.TOOLS:
    mcp.tool(_traced(fn) if TRACE else fn)

if __name__ == "__main__":
    mcp.run()
