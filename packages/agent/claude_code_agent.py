"""Bench agent that runs on the owner's Claude Code subscription instead of the API.

It launches `claude -p` (headless Claude Code) with the AlchemAI MCP server attached and the same system
rules as agent.py, restricted to the MCP tools (no shell, no files), and parses the final JSON block.

    uv run python packages/agent/claude_code_agent.py < card.json
    uv run python packages/bench/runner.py --agent claude-code bomba
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.agent.agent import SYSTEM, _parse_answer  # noqa: E402

MODEL = os.environ.get("ALCHEMAI_CC_MODEL", "opus")
EFFORT = os.environ.get("ALCHEMAI_CC_EFFORT", "high")
MAX_TURNS = int(os.environ.get("ALCHEMAI_CC_MAX_TURNS", "60"))


def _mcp_config_path() -> str:
    cfg = {"mcpServers": {"alchemai": {"command": "uv", "args": ["run", "--directory", str(ROOT), "python", "-m", "packages.dwsim_mcp.server"]}}}
    p = Path(tempfile.gettempdir()) / "alchemai_mcp.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def solve_card(card: dict, log=None) -> dict:
    user = ("Card:\n```json\n" + json.dumps(card, ensure_ascii=False, indent=1) + "\n```\n"
            "Solve it with the alchemai MCP tools only and finish with the JSON block.")
    cmd = ["claude", "-p", user, "--model", MODEL, "--effort", EFFORT,
           "--mcp-config", _mcp_config_path(), "--strict-mcp-config",
           "--tools", "", "--allowedTools", "mcp__alchemai__*",
           "--append-system-prompt", SYSTEM, "--max-turns", str(MAX_TURNS),
           "--output-format", "json", "--no-session-persistence", "--permission-mode", "bypassPermissions"]
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT), env=env)
    if log:
        log(f"  claude -p exit={proc.returncode} stderr={proc.stderr.strip()[-300:]}")
    try:
        out = json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return {"answers": {}, "evidence": {"notes": f"claude -p produced no JSON (exit {proc.returncode}): {proc.stdout[-500:]} {proc.stderr[-500:]}"}, "unverified": []}
    text = out.get("result", "") if isinstance(out, dict) else ""
    ans = _parse_answer(text)
    ev = ans.setdefault("evidence", {})
    ev["model"] = f"claude-code:{MODEL}"
    ev["num_turns"] = out.get("num_turns") if isinstance(out, dict) else None
    ev["cost_usd_reported"] = out.get("total_cost_usd") if isinstance(out, dict) else None
    ev["duration_s"] = round((out.get("duration_ms") or 0) / 1000, 1) if isinstance(out, dict) else None
    if isinstance(out, dict) and out.get("is_error"):
        ev["notes"] = (ev.get("notes", "") + f" | claude error: {str(out.get('result'))[:300]}").strip(" |")
    return ans


if __name__ == "__main__":
    print(json.dumps(solve_card(json.load(sys.stdin), log=lambda s: print(s, file=sys.stderr)), ensure_ascii=False))
