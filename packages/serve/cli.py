"""`alchemai serve` — the engine process behind the Mac app (and any other front end).

One process, two channels, one DWSIM state:
  * stdio, JSON lines: the app sends requests ({"op": ...}) on stdin and reads events on stdout.
      {"op": "ping"}                               -> {"event": "ready", ...}
      {"op": "tool", "id", "name", "args"}         -> {"event": "result", "id", "ok", "data" | "error"}
      {"op": "agent", "id", "prompt", "model"?, "effort"?}
                                                   -> {"event": "agent", "id", "kind": "start|text|tool_call|tool_result|final|error", ...}
      {"op": "stop", "id"}                         -> kills that agent run (and its claude process group)
      {"op": "quit"}
  * HTTP MCP on 127.0.0.1:<port> with the same 55 tools, for the agent (`claude -p` on the owner's
    subscription). Because the MCP server lives in THIS process, the flowsheets the agent builds are the
    same objects the app edits afterwards ("Resolver sin el agente": no tokens spent).

DWSIM is GPL-3 and stays a separate installation; this process loads it through pythonnet exactly like the
bench does. All logs go to stderr; stdout carries only JSON events.

    uv run python -m packages.serve.cli            # or: alchemai serve (see pyproject scripts)
"""
from __future__ import annotations

import functools
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.agent.agent import SYSTEM, _parse_answer  # noqa: E402
from packages.agent.claude_code_agent import _kill_group, clean_env  # noqa: E402
from packages.dwsim_client import tools as T  # noqa: E402

VERSION = "0.1.0"
APP_SYSTEM = SYSTEM + """

You are now talking to an engineer inside the AlchemAI Mac app, not solving a bench card. Write your prose
answer in Spanish (short, for a chemical engineer). The flowsheet you build stays loaded in the app, so name
units and streams clearly (feed, dist, bot, T-101...). Finish with the same JSON block, where "answers" holds
the quantities you reported, and add "summary": one Spanish paragraph with the result and its evidence.
"""
_LOCK = threading.RLock()          # DWSIM is not thread-safe: app calls and agent (HTTP) calls are serialized
_TOOLS = {fn.__name__: fn for fn in T.TOOLS}
_RUNS: dict[str, subprocess.Popen] = {}
_OUT = threading.Lock()
# DWSIM/.NET prints to the real stdout ("resource pool: create resource pool..."), which would corrupt the JSON
# channel. So the JSON events go to a private copy of fd 1 and fd 1 itself is pointed at stderr before DWSIM loads.
_JSON_OUT = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with _OUT:
        _JSON_OUT.write(line + "\n")
        _JSON_OUT.flush()
    log(f"→ {obj.get('event')} {obj.get('kind') or obj.get('name') or ''} {len(line)} B")


def log(msg: str) -> None:
    print(f"[serve] {msg}", file=sys.stderr, flush=True)


def _locked(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        with _LOCK:
            return fn(*a, **k)
    return w


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mcp_http(port: int) -> threading.Thread:
    from fastmcp import FastMCP
    mcp = FastMCP("alchemai", instructions=(
        "Deterministic chemical-process simulation (DWSIM 10). Build a flowsheet (new_flowsheet → streams → "
        "units → connect), then ALWAYS call solve(): it returns the balance verification. Report only numbers "
        "from a solve with ok=true; mark everything else as unverified."))
    for fn in T.TOOLS:
        mcp.tool(_locked(fn))
    t = threading.Thread(target=lambda: mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False),
                         name="mcp-http", daemon=True)
    t.start()
    return t


def _wait_port(port: int, timeout_s: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def describe_flowsheet(flowsheet_id: str) -> dict:
    """App-only (not an agent tool): the graph as the session recorded it — units with kind, params and connections,
    stream specs, compounds. This is what the diagram draws and what the inspector edits."""
    s = T._s(flowsheet_id)
    units = []
    for name, kind in s["kinds"].items():
        c = s["conn"].get(name) or {}
        params = dict(s["params"].get(name) or {})
        if name in s["column_feeds"]:
            params["feed_stage"] = s["column_feeds"][name]
        units.append({"name": name, "kind": kind, "params": params, "feeds": c.get("feeds") or [], "products": c.get("products") or [],
                      "energy_feeds": c.get("energy_feeds") or [], "energy_products": c.get("energy_products") or []})
    return {"flowsheet_id": flowsheet_id, "compounds": list(s["compounds"]), "units": units, "streams": list(s["streams"]),
            "energy": list(s["energy"]), "stream_specs": {k: dict(v) for k, v in s["stream_specs"].items()}}


def list_bench_cards() -> dict:
    """App-only: the public ChemE Design Bench cards (id, version, kind, title and statement in ES/EN) as examples."""
    import yaml
    cards_dir = ROOT / "packages" / "bench" / "cards"
    index = yaml.safe_load((cards_dir / "index.yaml").read_text())
    out = []
    for c in index["cards"]:
        try:
            card = yaml.safe_load((cards_dir / f"{c['id']}.yaml").read_text())
        except FileNotFoundError:
            continue
        out.append({"id": c["id"], "version": c.get("version", "v0"), "kind": c.get("kind"), "title_es": card["title"]["es"],
                    "title_en": card["title"]["en"], "statement_es": card["statement"]["es"], "statement_en": card["statement"]["en"],
                    "expected": sorted(card.get("expected", {}).keys())})
    return {"n": len(out), "cards": out}


def snapshot_flowsheet(flowsheet_id: str) -> dict:
    from packages.dwsim_client.persist import snapshot_flowsheet as snap
    return snap(flowsheet_id)


def restore_flowsheet(snapshot: dict) -> dict:
    from packages.dwsim_client.persist import restore_flowsheet as restore
    return restore(snapshot)


_APP_TOOLS = {"describe_flowsheet": describe_flowsheet, "list_bench_cards": list_bench_cards,
              "snapshot_flowsheet": snapshot_flowsheet, "restore_flowsheet": restore_flowsheet}


def op_tool(req: dict) -> None:
    name, args, rid = req.get("name"), req.get("args") or {}, req.get("id")
    fn = _TOOLS.get(name) or _APP_TOOLS.get(name)
    if fn is None:
        emit({"event": "result", "id": rid, "ok": False, "error": f"unknown tool {name!r}", "tools": sorted(_TOOLS)})
        return
    t0 = time.time()
    try:
        with _LOCK:
            data = fn(**args)
        emit({"event": "result", "id": rid, "ok": True, "name": name, "seconds": round(time.time() - t0, 3), "data": data})
    except Exception as ex:  # noqa: BLE001
        emit({"event": "result", "id": rid, "ok": False, "name": name, "seconds": round(time.time() - t0, 3), "error": f"{type(ex).__name__}: {ex}"[:1000]})


def _mcp_config(port: int) -> str:
    p = Path(tempfile.gettempdir()) / f"alchemai_serve_{port}.json"
    p.write_text(json.dumps({"mcpServers": {"alchemai": {"type": "http", "url": f"http://127.0.0.1:{port}/mcp"}}}))
    return str(p)


def _forward_stream_event(rid: str, ev: dict, calls: dict) -> None:
    """Map one `claude -p --output-format stream-json` line to app events."""
    t = ev.get("type")
    if t == "assistant":
        for block in (ev.get("message") or {}).get("content") or []:
            bt = block.get("type")
            if bt == "text" and block.get("text"):
                emit({"event": "agent", "id": rid, "kind": "text", "text": block["text"]})
            elif bt == "thinking" and block.get("thinking"):
                emit({"event": "agent", "id": rid, "kind": "thinking", "text": block["thinking"]})
            elif bt == "tool_use":
                name = str(block.get("name", "")).replace("mcp__alchemai__", "")
                calls[block.get("id")] = {"name": name, "t0": time.time()}
                emit({"event": "agent", "id": rid, "kind": "tool_call", "call_id": block.get("id"), "name": name, "args": block.get("input")})
    elif t == "user":
        for block in (ev.get("message") or {}).get("content") or []:
            if block.get("type") == "tool_result":
                c = calls.pop(block.get("tool_use_id"), {})
                content = block.get("content")
                if isinstance(content, list):
                    content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
                emit({"event": "agent", "id": rid, "kind": "tool_result", "call_id": block.get("tool_use_id"), "name": c.get("name"),
                      "seconds": round(time.time() - c["t0"], 2) if c else None, "is_error": bool(block.get("is_error")), "content": str(content)[:20000]})
    elif t == "result":
        ans = _parse_answer(ev.get("result", "") or "")
        emit({"event": "agent", "id": rid, "kind": "final", "is_error": bool(ev.get("is_error")), "num_turns": ev.get("num_turns"),
              "duration_s": round((ev.get("duration_ms") or 0) / 1000, 1), "cost_usd_reported": ev.get("total_cost_usd"),
              "text": ev.get("result", ""), "answer": ans})


_API_MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}


def op_agent_api(req: dict) -> None:
    """Same events as the claude-code path, but through the Anthropic SDK with the key the app passes in the
    environment (ANTHROPIC_API_KEY, from the macOS Keychain). Tools run in this process under the DWSIM lock."""
    import anthropic
    from packages.agent.agent import AGENT_TOOLS
    rid = req.get("id") or f"run-{int(time.time())}"
    model = _API_MODELS.get(req.get("model") or "opus", req.get("model") or "claude-opus-5-5")
    effort = req.get("effort") or "high"
    emit({"event": "agent", "id": rid, "kind": "start", "model": model, "effort": effort, "backend": "api"})
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        emit({"event": "agent", "id": rid, "kind": "error", "error": "no hay llave de API de Anthropic (ANTHROPIC_API_KEY)"})
        return
    t0 = time.time()
    n_turns = 0
    try:
        client = anthropic.Anthropic(api_key=key)
        runner = client.beta.messages.tool_runner(model=model, max_tokens=16000, system=APP_SYSTEM, tools=AGENT_TOOLS,
                                                  output_config={"effort": effort},
                                                  messages=[{"role": "user", "content": req.get("prompt", "")}])
        final = None
        for message in runner:
            if rid in _STOPPED:
                break
            n_turns += 1
            final = message
            calls = {}
            for b in message.content:
                if b.type == "text" and b.text:
                    emit({"event": "agent", "id": rid, "kind": "text", "text": b.text})
                elif b.type == "tool_use":
                    calls[b.id] = time.time()
                    emit({"event": "agent", "id": rid, "kind": "tool_call", "call_id": b.id, "name": b.name, "args": b.input})
            if getattr(message, "stop_reason", None) == "tool_use":
                with _LOCK:
                    resp = runner.generate_tool_call_response()
                for b in getattr(resp, "content", []) or []:
                    if getattr(b, "type", None) != "tool_result":
                        continue
                    content = b.content
                    if isinstance(content, list):
                        content = "".join(getattr(x, "text", "") or (x.get("text", "") if isinstance(x, dict) else "") for x in content)
                    emit({"event": "agent", "id": rid, "kind": "tool_result", "call_id": b.tool_use_id, "name": None,
                          "seconds": round(time.time() - calls.get(b.tool_use_id, time.time()), 2),
                          "is_error": bool(getattr(b, "is_error", False)), "content": str(content)[:20000]})
        text = "".join(b.text for b in final.content if b.type == "text") if final else ""
        emit({"event": "agent", "id": rid, "kind": "final", "is_error": False, "num_turns": n_turns,
              "duration_s": round(time.time() - t0, 1), "cost_usd_reported": None, "text": text, "answer": _parse_answer(text)})
    except Exception as ex:  # noqa: BLE001
        emit({"event": "agent", "id": rid, "kind": "error", "error": f"{type(ex).__name__}: {str(ex)[:400]}"})
    finally:
        _STOPPED.discard(rid)


def op_agent(req: dict, port: int) -> None:
    if req.get("backend") == "api" or (req.get("backend") is None and os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ALCHEMAI_AGENT_BACKEND") == "api"):
        return op_agent_api(req)
    rid = req.get("id") or f"run-{int(time.time())}"
    model = req.get("model") or os.environ.get("ALCHEMAI_CC_MODEL", "opus")
    effort = req.get("effort") or os.environ.get("ALCHEMAI_CC_EFFORT", "high")
    timeout_s = int(req.get("timeout_s") or os.environ.get("ALCHEMAI_CC_TIMEOUT_S", "600"))
    cmd = ["claude", "-p", req.get("prompt", ""), "--model", model, "--effort", effort, "--tools", "",
           "--append-system-prompt", APP_SYSTEM, "--max-turns", str(req.get("max_turns") or 40),
           "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--setting-sources", "project,local", "--permission-mode", "bypassPermissions",
           "--mcp-config", _mcp_config(port), "--strict-mcp-config", "--allowedTools", "mcp__alchemai__*"]
    env = clean_env()
    emit({"event": "agent", "id": rid, "kind": "start", "model": model, "effort": effort})
    try:
        # stdin=DEVNULL: si hereda el pipe de la app (siempre abierto), `claude -p` espera el EOF de stdin para sumarlo al
        # prompt y nunca arranca (corrida colgada 5 min sin una sola llamada, 2026-11-08).
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT), env=env, start_new_session=True)
    except FileNotFoundError:
        emit({"event": "agent", "id": rid, "kind": "error", "error": "claude (Claude Code CLI) not found on PATH"})
        return
    _RUNS[rid] = proc
    calls: dict = {}
    got_final = False
    t0 = time.time()

    def killer():
        while proc.poll() is None:
            if time.time() - t0 > timeout_s:
                emit({"event": "agent", "id": rid, "kind": "error", "error": f"timeout after {timeout_s} s"})
                _kill_group(proc)
                return
            time.sleep(0.5)
    threading.Thread(target=killer, daemon=True).start()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                got_final = True
            _forward_stream_event(rid, ev, calls)
    finally:
        err = ""
        try:
            err = proc.stderr.read()[-1000:]
        except Exception:  # noqa: BLE001
            pass
        proc.wait()
        _RUNS.pop(rid, None)
        if not got_final:
            emit({"event": "agent", "id": rid, "kind": "error", "error": f"claude -p ended without a result (exit {proc.returncode}): {err.strip()[-500:]}", "stopped": rid in _STOPPED})
            _STOPPED.discard(rid)


_STOPPED: set[str] = set()


def op_stop(req: dict) -> None:
    rid = req.get("id")
    _STOPPED.add(rid)                       # the API runner checks this between turns
    proc = _RUNS.get(rid)
    if proc is not None:
        _kill_group(proc)
    emit({"event": "stopped", "id": rid, "ok": True})


def _watch_parent() -> None:
    """Exit when the app that launched us is gone. `uv run` sits between the app and us, so stdin EOF is not
    always enough (orphans with DWSIM loaded were found); the app passes its pid in ALCHEMAI_PARENT_PID."""
    pid = int(os.environ.get("ALCHEMAI_PARENT_PID") or 0)
    if not pid:
        return
    while True:
        time.sleep(2)
        try:
            os.kill(pid, 0)
        except OSError:
            log(f"parent {pid} gone; exiting")
            for proc in list(_RUNS.values()):
                _kill_group(proc)
            os._exit(0)


def main() -> None:
    port = int(os.environ.get("ALCHEMAI_SERVE_PORT") or _free_port())
    threading.Thread(target=_watch_parent, name="watch-parent", daemon=True).start()
    log(f"starting: python {sys.version.split()[0]}, mcp http port {port}")
    start_mcp_http(port)
    mcp_ok = _wait_port(port)
    with _LOCK:
        t0 = time.time()
        packages = T.list_property_packages()      # loads DWSIM through pythonnet
        dwsim_s = round(time.time() - t0, 1)
    ready = {"event": "ready", "version": VERSION, "mcp_url": f"http://127.0.0.1:{port}/mcp", "mcp_ok": mcp_ok,
             "dwsim_load_s": dwsim_s, "n_tools": len(_TOOLS), "property_packages": packages, "pid": os.getpid()}
    emit(ready)
    log(f"ready in {dwsim_s} s; {len(_TOOLS)} tools")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            emit({"event": "error", "error": "bad JSON line", "line": raw[:200]})
            continue
        op = req.get("op")
        if op == "ping":
            emit({**ready, "id": req.get("id")})
        elif op == "tool":
            op_tool(req)
        elif op == "agent":
            threading.Thread(target=op_agent, args=(req, port), daemon=True).start()
        elif op == "stop":
            op_stop(req)
        elif op == "quit":
            break
        else:
            emit({"event": "error", "id": req.get("id"), "error": f"unknown op {op!r}"})
    for proc in list(_RUNS.values()):
        _kill_group(proc)
    log("bye")
    # The HTTP server thread (uvicorn) is not a daemon in practice: without this the process outlives the app
    # (two orphans with DWSIM loaded were found after two test runs, 2026-11-06).
    _JSON_OUT.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
