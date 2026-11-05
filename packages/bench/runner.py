"""ChemE Design Bench v0 runner — grades any agent against the yaml cards.

Agent protocol (JSON over stdin/stdout, so any language or MCP client can play):
    stdin : the card as JSON (id, kind, title, statement, expected quantity *names and units*
            but NOT the values, requires)
    stdout: {"answers": {quantity: number}, "evidence": {"converged": bool,
             "balances_closed": bool, "notes": str}, "unverified": [quantity, ...],
             "infeasible": {"detected": bool, "reason": str}}        # v1: optional, for inconsistent specs

Scoring per card (0..1):
    * each expected quantity within tolerance → 1/n; outside → 0; missing → 0
    * a quantity reported *and* listed in "unverified" is graded but flagged (honesty is free);
      a wrong value NOT flagged as unverified costs an extra 0.5/n (penalises confident errors)
    * if the card requires evidence and the agent claims it falsely (we cannot check from outside,
      so we only check presence), missing evidence caps the score at 0.8

Trap cards (v1, kind "trap"; the spec is impossible by construction):
    * detected + reason mentions the right wall (any card keyword) → 1.0; detected, reason off → 0.6
    * not detected but every reported number flagged unverified → 0.2 (honest, but missed the trap)
    * nothing reported and nothing detected → 0.0
    * not detected and numbers reported as verified → 0.0
    * a feasible (v0) card flagged infeasible → 0.0 (false alarm costs the whole card)

Built-in agents: `oracle` (re-solves with the case builders — should score 1.0) and `null`
(answers nothing — scores 0). External: `--agent-cmd "uv run python my_agent.py"`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CARDS_DIR = Path(__file__).with_name("cards")


def load_cards(ids=None, version=None):
    cards = [yaml.safe_load(p.read_text()) for p in sorted(CARDS_DIR.glob("*.yaml")) if p.name != "index.yaml"]
    def _v(c):
        v = c.get("version", "v0")
        return not version or version == "all" or v == version or (version == "traps" and c["kind"] == "trap")
    return [c for c in cards if (not ids or c["id"] in ids) and _v(c)]


def public_view(card):
    """What the agent sees: never the expected values, never the trap flag (kind is shown as 'simulation' for traps
    so the agent cannot pattern-match on it)."""
    kind = "simulation" if card["kind"] == "trap" else card["kind"]
    return {"id": card["id"], "kind": kind, "title": card["title"], "statement": card["statement"],
            "expected_quantities": {k: v["unit"] for k, v in card["expected"].items()}, "requires": card["requires"]}


def _infeasible(response):
    inf = response.get("infeasible") or {}
    return bool(inf.get("detected")), str(inf.get("reason", "") or "")


def grade_trap(card, response):
    detected, reason = _infeasible(response)
    answers = response.get("answers", {}) or {}
    unverified = set(response.get("unverified", []) or [])
    reported = {q for q, v in answers.items() if v is not None}
    if detected:
        hit = [k for k in card["trap"]["keywords"] if k.lower() in reason.lower()]
        score = 1.0 if hit else 0.6
        status = f"trap detected ({'keywords ' + ','.join(hit[:3]) if hit else 'reason does not name the wall'})"
    elif reported and reported <= unverified:
        score, status = 0.2, "trap missed; numbers reported but all flagged unverified"
    elif not reported:
        score, status = 0.0, "trap missed; nothing reported"
    else:
        score, status = 0.0, "TRAP MISSED: numbers reported as verified"
    rows = [(q, answers.get(q), None, status) for q in card["expected"]]
    return score, rows, []


def grade(card, response):
    if card["kind"] == "trap":
        return grade_trap(card, response)
    answers = response.get("answers", {}) or {}
    unverified = set(response.get("unverified", []) or [])
    evidence = response.get("evidence", {}) or {}
    if _infeasible(response)[0]:        # false alarm on a feasible card
        return 0.0, [(q, answers.get(q), spec["value"], "FALSE ALARM: feasible card flagged infeasible") for q, spec in card["expected"].items()], []
    n = len(card["expected"])
    score, rows = 0.0, []
    for q, spec in card["expected"].items():
        got = answers.get(q)
        v, tol = spec["value"], spec["tol"]
        if got is None:
            rows.append((q, None, v, "missing")); continue
        try:
            got = float(got)
        except (TypeError, ValueError):
            rows.append((q, got, v, "not a number")); continue
        band = tol.get("abs", 0.0) + tol.get("rel", 0.0) * abs(v)
        ok = abs(got - v) <= band + 1e-12
        if ok:
            score += 1.0 / n
            rows.append((q, got, v, "ok"))
        else:
            penalty = 0.0 if q in unverified else 0.5 / n
            score -= penalty
            rows.append((q, got, v, "wrong (flagged unverified)" if q in unverified else "WRONG, reported as verified"))
    score = max(0.0, score)
    missing_evidence = [r for r in card.get("requires", []) if not evidence.get(r)]
    if missing_evidence:
        score = min(score, 0.8)
    return score, rows, missing_evidence


# ----------------------------------------------------------------------------- built-in agents
def oracle_agent(card):
    """Re-solves the case with our builders/oracles. Upper bound for any agent."""
    from packages.bench.build_cards import BASELINE, BASELINE_META, REF, DESIGN_CASES, POLAR_SIM, POLAR_DESIGN
    from packages.dwsim_client.verify import verify
    cid = card["id"]
    if card["kind"] == "trap":
        return {"answers": {}, "evidence": {"notes": "oracle: trap by construction"}, "unverified": [],
                "infeasible": {"detected": True, "reason": card["trap"]["reason"]["es"]}}
    if cid in BASELINE:
        fs, h = BASELINE[cid](); rep = verify(fs)
        return {"answers": BASELINE_META[cid]["readouts"](h), "evidence": {"converged": rep.solved, "balances_closed": rep.ok}, "unverified": []}
    ref = {c.id: c for c in REF + POLAR_SIM}
    if cid in ref:
        fs, h = ref[cid].build(); rep = verify(fs)
        return {"answers": ref[cid].readouts(h), "evidence": {"converged": rep.solved, "balances_closed": rep.ok}, "unverified": []}
    from packages.bench.transport_cases import TRANSPORT_CASES
    des = {d.id: d for d in DESIGN_CASES + POLAR_DESIGN + TRANSPORT_CASES}
    if cid in des:
        a = des[cid].oracle()
        return {"answers": a, "evidence": {"converged": True, "balances_closed": True, "notes": "oracle: DWSIM search + analytic checks"}, "unverified": []}
    raise KeyError(cid)


def null_agent(card):
    return {"answers": {}, "evidence": {}, "unverified": []}


def claude_agent(card):
    """AlchemAI agent (Claude + DWSIM tools); sees only the public view of the card."""
    from packages.agent.agent import solve_card
    return solve_card(public_view(card), log=lambda s: print(s))


def claude_code_agent(card):
    """Same agent rules, run on the owner's Claude Code subscription via `claude -p` + the AlchemAI MCP server."""
    from packages.agent.claude_code_agent import solve_card
    return solve_card(public_view(card), log=lambda s: print(s))


def subprocess_agent(cmd):
    def run(card):
        proc = subprocess.run(cmd, shell=True, input=json.dumps(public_view(card), ensure_ascii=False), capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return {"answers": {}, "evidence": {"notes": f"agent failed: {proc.stderr[-500:]}"}, "unverified": []}
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as ex:  # noqa: BLE001
            return {"answers": {}, "evidence": {"notes": f"bad JSON: {ex}"}, "unverified": []}
    return run


def run_bench(agent, cards, verbose=True, out_jsonl=None):
    """Grade every card; optionally append one JSON line per card to `out_jsonl` as it finishes."""
    import time
    total, results = 0.0, []
    for card in cards:
        t0 = time.time()
        resp = agent(card)
        s, rows, missing = grade(card, resp)
        total += s
        results.append((card["id"], s, rows, missing))
        if verbose:
            print(f"\n## {card['id']} [{card['kind']}] — score {s:.2f}  ({time.time() - t0:.0f} s)", flush=True)
            for q, got, v, status in rows:
                exp = f"{v:<14.6g}" if isinstance(v, (int, float)) else f"{'—':<14s}"
                print(f"  {q:22s} got={got!s:>14s}  expected={exp} {status}", flush=True)
            if missing:
                print(f"  evidence missing: {missing}", flush=True)
            if resp.get("evidence", {}).get("notes"):
                print(f"  notes: {str(resp['evidence']['notes'])[:300]}", flush=True)
        if out_jsonl:
            with open(out_jsonl, "a") as f:
                f.write(json.dumps({"id": card["id"], "score": s, "seconds": round(time.time() - t0, 1),
                                    "rows": rows, "missing_evidence": missing, "response": resp}, ensure_ascii=False, default=str) + "\n")
    mean = total / max(len(cards), 1)
    traps = [s for cid, s, _, _ in results if cid.startswith("trampa_")]
    extra = f" · traps {sum(traps) / len(traps) * 100:.0f} % ({len(traps)})" if traps else ""
    print(f"\n### Bench: {mean * 100:.1f} % over {len(cards)} cards{extra}", flush=True)
    return mean, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="oracle", choices=["oracle", "null", "claude", "claude-code", "cmd"])
    ap.add_argument("--agent-cmd", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None, help="append per-card JSON lines here as cards finish")
    ap.add_argument("--version", default="all", choices=["v0", "v1", "v1b", "v2", "v3", "traps", "all"], help="v0 = 20 feasible, v1 = 10 physics traps, v1b = 5 specification traps, traps = both, v2 = 4 polar mixtures (package choice), v3 = 5 transport phenomena + cost")
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    cards = load_cards(a.ids, a.version)
    agent = {"oracle": oracle_agent, "null": null_agent, "claude": claude_agent, "claude-code": claude_code_agent}.get(a.agent) or subprocess_agent(a.agent_cmd)
    mean, _ = run_bench(agent, cards, verbose=not a.quiet, out_jsonl=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
