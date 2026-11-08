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
MAX_TURNS = int(os.environ.get("ALCHEMAI_CC_MAX_TURNS", "40"))
TIMEOUT_S = int(os.environ.get("ALCHEMAI_CC_TIMEOUT_S", "600"))   # per card wall clock


_HOST_SESSION_PREFIXES = ("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_")
_HOST_SESSION_KEYS = {"CLAUDECODE", "CLAUDE_PID", "AI_AGENT", "CLAUDE_EFFORT", "BAGGAGE", "SENTRY_TRACE"}


def clean_env() -> dict[str, str]:
    """Environment for a child `claude -p` that must run on its own subscription login.
    Drops the variables of a parent Claude Code session (messaging socket, host auth refresh, session ids): when
    the app or this script is launched from inside Claude Code, the child latched onto the parent session and hung
    without making a single call (2026-11-08). Also drops ANTHROPIC_API_KEY so the subscription path never bills
    an API key that happens to be in the environment."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(_HOST_SESSION_PREFIXES) and k not in _HOST_SESSION_KEYS and k != "ANTHROPIC_API_KEY"}


def _mcp_config_path() -> str:
    cfg = {"mcpServers": {"alchemai": {"command": "uv", "args": ["run", "--directory", str(ROOT), "python", "-m", "packages.dwsim_mcp.server"]}}}
    p = Path(tempfile.gettempdir()) / "alchemai_mcp.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _tools_unavailable(ans: dict) -> bool:
    """claude -p occasionally starts before the MCP server (DWSIM takes a few seconds to load) and then
    answers from memory. Detect it so we can retry once instead of grading a hand estimate."""
    ev = ans.get("evidence", {}) or {}
    notes = str(ev.get("notes", "")).lower()
    return (ev.get("num_turns") in (0, 1) and not ev.get("converged")) or "not available" in notes or "no tools" in notes


def solve_card(card: dict, log=None, _retry: bool = True) -> dict:
    ans = _solve_card_once(card, log)
    if _retry and _tools_unavailable(ans):
        if log:
            log("  MCP tools were not available to claude -p; retrying once")
        ans = _solve_card_once(card, log)
        ans.setdefault("evidence", {})["retried"] = True
    return ans


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group of a timed-out `claude -p`, then SIGKILL whatever survives."""
    import signal
    import time
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def run_claude_p(prompt: str, system: str, *, model: str = MODEL, effort: str = EFFORT, max_turns: int = MAX_TURNS,
                 timeout_s: int = TIMEOUT_S, mcp: bool = True, log=None) -> dict:
    """Run one headless `claude -p` call on the owner's subscription and return claude's JSON envelope
    (`result`, `num_turns`, `duration_ms`, `is_error`...). With mcp=True the AlchemAI server is attached and
    only its tools are allowed; with mcp=False the model has no tools at all (pure text, e.g. HAZOP prose).
    Never raises: timeouts and bad output come back as {"is_error": True, "result": <why>}."""
    cmd = ["claude", "-p", prompt, "--model", model, "--effort", effort, "--tools", "",
           "--append-system-prompt", system, "--max-turns", str(max_turns),
           "--output-format", "json", "--no-session-persistence", "--permission-mode", "bypassPermissions"]
    if mcp:
        cmd += ["--mcp-config", _mcp_config_path(), "--strict-mcp-config", "--allowedTools", "mcp__alchemai__*"]
    env = clean_env()
    # Own process group: on timeout we must kill claude AND its MCP server child (a DWSIM solve that never
    # returns would otherwise keep spinning at 100 % CPU as an orphan; three such orphans were found 2026-11-04).
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT), env=env,
                            start_new_session=True)   # stdin cerrado: claude -p espera el EOF de un stdin heredado
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return {"is_error": True, "result": f"timeout after {timeout_s} s", "num_turns": None}
    proc.stdout, proc.stderr = stdout, stderr   # keep the attribute shape the code below expects
    if log:
        log(f"  claude -p exit={proc.returncode} stderr={proc.stderr.strip()[-300:]}")
    try:
        out = json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return {"is_error": True, "result": f"claude -p produced no JSON (exit {proc.returncode}): {proc.stdout[-500:]} {proc.stderr[-500:]}", "num_turns": None}
    return out if isinstance(out, dict) else {"is_error": True, "result": str(out)[:500]}


def _solve_card_once(card: dict, log=None) -> dict:
    user = ("Card:\n```json\n" + json.dumps(card, ensure_ascii=False, indent=1) + "\n```\n"
            "Solve it with the alchemai MCP tools only and finish with the JSON block.")
    out = run_claude_p(user, SYSTEM, mcp=True, log=log)
    if out.get("is_error") and out.get("num_turns") is None:   # timeout / no JSON: nothing to parse
        return {"answers": {}, "evidence": {"notes": str(out.get("result"))[:600], "model": f"claude-code:{MODEL}"}, "unverified": []}
    ans = _parse_answer(out.get("result", "") or "")
    ev = ans.setdefault("evidence", {})
    ev["model"] = f"claude-code:{MODEL}"
    ev["num_turns"] = out.get("num_turns")
    ev["cost_usd_reported"] = out.get("total_cost_usd")
    ev["duration_s"] = round((out.get("duration_ms") or 0) / 1000, 1)
    if out.get("is_error"):
        ev["notes"] = (ev.get("notes", "") + f" | claude error: {str(out.get('result'))[:300]}").strip(" |")
    return ans


if __name__ == "__main__":
    print(json.dumps(solve_card(json.load(sys.stdin), log=lambda s: print(s, file=sys.stderr)), ensure_ascii=False))
