"""`alchemai serve`: JSON-lines stdio + HTTP MCP in one process (Mac app engine). The agent op needs `claude`
and the owner's subscription: it runs only with ALCHEMAI_TEST_AGENT=1."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Serve:
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, "-m", "packages.serve.cli"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})

    def read(self, timeout_s=120):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("serve exited: " + self.p.stderr.read()[-800:])
            return json.loads(line)
        raise TimeoutError

    def send(self, **req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()

    def call(self, tool, **args):
        self.send(op="tool", id=tool, name=tool, args=args)
        ev = self.read()
        assert ev["event"] == "result" and ev["id"] == tool, ev
        assert ev["ok"], ev.get("error")
        return ev["data"]

    def close(self):
        self.send(op="quit"); self.p.wait(timeout=20)


@pytest.fixture(scope="module")
def serve():
    s = Serve()
    ready = s.read(timeout_s=180)
    assert ready["event"] == "ready" and ready["mcp_ok"] and ready["n_tools"] >= 36, ready
    assert "NRTL" in ready["property_packages"]
    yield s
    s.close()


def test_tool_calls_share_one_dwsim_state(serve):
    fid = serve.call("new_flowsheet", name="mix", compounds=["Water"], property_package="SteamTables")["flowsheet_id"]
    serve.call("add_material_stream", flowsheet_id=fid, name="a", temperature_K=300.0, pressure_Pa=101325.0, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    serve.call("add_material_stream", flowsheet_id=fid, name="b", temperature_K=350.0, pressure_Pa=101325.0, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    serve.call("add_material_stream", flowsheet_id=fid, name="m")
    serve.call("add_unit", flowsheet_id=fid, kind="mixer", name="MIX", params={})
    serve.call("connect", flowsheet_id=fid, unit="MIX", feeds=["a", "b"], products=["m"])
    r = serve.call("solve", flowsheet_id=fid)
    assert r["ok"]
    m = serve.call("read_stream", flowsheet_id=fid, name="m")
    assert 320 < m["T_K"] < 330 and abs(m["mass_flow_kg_s"] - 2.0) < 1e-6
    d = serve.call("describe_flowsheet", flowsheet_id=fid)
    assert [u["name"] for u in d["units"]] == ["MIX"] and d["units"][0]["feeds"] == ["a", "b"] and d["units"][0]["products"] == ["m"]
    assert d["stream_specs"]["a"]["temperature_K"] == 300.0 and d["compounds"] == ["Water"]
    snap = serve.call("snapshot_flowsheet", flowsheet_id=fid)
    r = serve.call("restore_flowsheet", snapshot=snap)
    m2 = serve.call("read_stream", flowsheet_id=r["flowsheet_id"], name="m")
    assert r["solve"]["ok"] and abs(m2["T_K"] - m["T_K"]) < 1e-6
    serve.send(op="tool", id="x", name="nope", args={})
    ev = serve.read(); assert ev["ok"] is False and "unknown tool" in ev["error"]


def test_bench_cards_are_listed_for_the_app(serve):
    d = serve.call("list_bench_cards")
    assert d["n"] >= 50 and any(c["id"] == "optimo_etapas_reflujo" for c in d["cards"])
    c = next(c for c in d["cards"] if c["id"] == "bomba")
    assert "mol" in c["statement_es"] or "kg" in c["statement_es"] and c["expected"]


def test_mcp_http_serves_the_same_tools(serve):
    import urllib.request
    serve.send(op="ping", id="p"); ready = serve.read()
    req = urllib.request.Request(ready["mcp_url"], data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    body = urllib.request.urlopen(req, timeout=20).read().decode()
    assert "alchemai" in body


@pytest.mark.skipif(not os.environ.get("ALCHEMAI_TEST_AGENT"), reason="needs claude CLI + subscription (ALCHEMAI_TEST_AGENT=1)")
def test_agent_streams_tool_calls_and_leaves_the_flowsheet_loaded(serve):
    serve.send(op="agent", id="r1", prompt="Mezcla 1 kg/s de agua a 300 K con 1 kg/s de agua a 350 K, ambas a 1 atm (SteamTables). ¿Temperatura de salida en K?", model="sonnet", effort="medium")
    kinds, final = [], None
    while final is None:
        ev = serve.read(timeout_s=300)
        assert ev["event"] == "agent" and ev["id"] == "r1", ev
        kinds.append(ev["kind"])
        if ev["kind"] in ("final", "error"):
            final = ev
    assert final["kind"] == "final" and not final["is_error"], final
    assert "tool_call" in kinds and "tool_result" in kinds, kinds
    assert any(320 < float(v) < 330 for v in final["answer"]["answers"].values() if isinstance(v, (int, float))), final["answer"]
    # the agent's flowsheet lives in this process: the app can read it without tokens
    serve.send(op="tool", id="pp", name="list_property_packages", args={}); assert serve.read()["ok"]


def test_child_claude_env_drops_host_session_and_api_key(monkeypatch):
    from packages.agent.claude_code_agent import clean_env
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_SOCKET", "/tmp/x.sock")
    monkeypatch.setenv("CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH", "1")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HOME_TEST_KEEP", "yes")
    env = clean_env()
    assert "CLAUDE_CODE_MESSAGING_SOCKET" not in env and "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH" not in env
    assert "CLAUDECODE" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1" and env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["HOME_TEST_KEEP"] == "yes"
