"""LLM layer of the preliminary HAZOP: Claude reads the deterministic `rule` rows plus the solved process graph
and PROPOSES additional causes, consequences and safeguards. Every proposal is tagged `source="proposal"`,
carries the model name, and is appended after the rule rows — it never edits, merges or removes a `rule` row.

Runs on the owner's Claude Code subscription via `claude -p` with no tools (pure text), so it costs no API key.
Proposals that name a node absent from the graph are dropped (hallucinated equipment is the first failure
mode of LLM HAZOPs). Severity of a proposal is inherited from the rule rows of the same node (max), or 1.

    rows = propose(rule_rows, graph_dict)          # -> list[HazopRow] with source="proposal"
    uv run python -m packages.hazop.llm            # demo on the wet-gas compression train
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.hazop.rules import HazopRow  # noqa: E402

MODEL = os.environ.get("ALCHEMAI_HAZOP_MODEL", os.environ.get("ALCHEMAI_CC_MODEL", "opus"))
EFFORT = os.environ.get("ALCHEMAI_HAZOP_EFFORT", "medium")
TIMEOUT_S = int(os.environ.get("ALCHEMAI_HAZOP_TIMEOUT_S", "300"))
MAX_PROPOSALS = 12

SYSTEM = """You are a senior process-safety engineer taking part in a HAZOP review. You receive (a) the solved
process graph (units, streams, T, P, phases, flows, duties, hazard flags) and (b) the deviation rows already
produced by deterministic rules. Your job is to PROPOSE what the rules missed: additional deviations, causes,
consequences and safeguards that a multidisciplinary team would raise for THIS process, using its numbers.

Rules:
1. Only name units that exist in the graph (exact names). Never invent equipment or streams.
2. Do not repeat a rule row. Add what is missing: common-cause failures, utility failures (cooling water,
   instrument air, power), start-up/shutdown, maintenance, human error, external events, contamination.
3. Quantify with the graph where you can (e.g. "settle-out pressure ~ X bar", "inventory Y kg of flammables").
   If you cannot quantify, say so; never make up a number.
4. Write causes, consequences and safeguards in Spanish, concise (one line each). Parameter in English
   (Flow, Pressure, Temperature, Level, Composition, Reaction, Utility, Procedure); deviation in English.
5. Return at most %d proposals, the most valuable first.

Finish with ONE fenced JSON block, nothing after it:
```json
{"proposals": [{"node": "<unit name>", "parameter": "<...>", "deviation": "<...>",
                "causes": ["..."], "consequences": ["..."], "safeguards": ["..."], "rationale": "<why the rules missed it>"}]}
```""" % MAX_PROPOSALS

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def _compact_graph(graph: dict) -> dict:
    """What the model needs, with rounded numbers (keeps the prompt short and the numbers legible)."""
    units = {k: {"kind": v["kind"], "inlets": v["inlets"], "outlets": v["outlets"], "duty_kW": _r(v.get("duty_kW")),
                 "T_in_K": _r(v.get("T_in_K")), "T_out_K": _r(v.get("T_out_K")),
                 "P_in_bar": _r(v.get("P_in_Pa"), 1e5), "P_out_bar": _r(v.get("P_out_Pa"), 1e5), "exothermic": v.get("exothermic")}
             for k, v in graph["units"].items()}
    streams = {k: {"T_K": _r(v["T_K"]), "P_bar": _r(v["P_Pa"], 1e5), "VF": _r(v["vapor_fraction"], 1, 3),
                   "kg_s": _r(v["mass_flow_kg_s"], 1, 3), "mol_s": _r(v["molar_flow_mol_s"], 1, 3),
                   "x": {c: round(x, 4) for c, x in v["mole_fractions"].items() if x > 1e-4},
                   "flammable": v["flammable"], "toxic": v["toxic"], "water": v["has_water"], "hydrate_risk": v["hydrate_risk"]}
               for k, v in graph["streams"].items()}
    return {"name": graph["name"], "units": units, "streams": streams}


def _r(x, scale: float = 1.0, nd: int = 1):
    return None if x is None else round(float(x) / scale, nd)


def build_prompt(rule_rows: list[HazopRow], graph: dict) -> str:
    rows = [{"node": r.node, "parameter": r.parameter, "deviation": r.deviation, "causes": r.causes,
             "consequences": r.consequences, "safeguards": r.safeguards, "severity": r.severity} for r in rule_rows]
    return ("Process graph (solved):\n```json\n" + json.dumps(_compact_graph(graph), ensure_ascii=False) + "\n```\n"
            "Rule rows already in the table:\n```json\n" + json.dumps(rows, ensure_ascii=False) + "\n```\n"
            "Propose what is missing. Finish with the JSON block.")


def parse_proposals(text: str, rule_rows: list[HazopRow], graph: dict, model: str = MODEL) -> tuple[list[HazopRow], list[dict]]:
    """Turn the model's JSON into HazopRow(source="proposal"). Returns (rows, dropped) where dropped lists the
    proposals rejected and why (unknown node, empty content)."""
    m = _JSON_BLOCK.findall(text or "")
    raw = m[-1] if m else (text[text.rfind("{"):] if text and "{" in text else "")
    try:
        props = json.loads(raw).get("proposals", [])
    except Exception:  # noqa: BLE001
        return [], [{"reason": "no valid JSON", "raw": (text or "")[-500:]}]
    sev_by_node: dict[str, int] = {}
    for r in rule_rows:
        sev_by_node[r.node] = max(sev_by_node.get(r.node, 1), r.severity)
    kinds = {k: v["kind"] for k, v in graph["units"].items()}
    rows, dropped = [], []
    for p in props[:MAX_PROPOSALS]:
        node = str(p.get("node", "")).strip()
        if node not in kinds:
            dropped.append({"reason": f"unknown node {node!r}", "proposal": p}); continue
        causes, cons, safe = [_lst(p.get(k)) for k in ("causes", "consequences", "safeguards")]
        if not (causes or cons):
            dropped.append({"reason": "empty proposal", "proposal": p}); continue
        rows.append(HazopRow(node=node, kind=kinds[node], parameter=str(p.get("parameter", "—")), deviation=str(p.get("deviation", "—")),
                             causes=causes, consequences=cons, safeguards=safe, severity=sev_by_node.get(node, 1),
                             evidence=f"propuesta LLM ({model}): {str(p.get('rationale', '')).strip()}"[:300], source="proposal"))
    return rows, dropped


def _lst(v) -> list[str]:
    if v is None:
        return []
    return [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]


def propose(rule_rows: list[HazopRow], graph: dict, model: str = MODEL, effort: str = EFFORT, log=None,
            _runner=None) -> tuple[list[HazopRow], dict]:
    """Ask Claude (subscription, `claude -p`, no tools) for proposals. Returns (proposal_rows, meta).
    `_runner` lets tests inject a fake `run_claude_p`."""
    from packages.agent.claude_code_agent import run_claude_p
    runner = _runner or run_claude_p
    out = runner(build_prompt(rule_rows, graph), SYSTEM, model=model, effort=effort, max_turns=1, timeout_s=TIMEOUT_S, mcp=False, log=log)
    meta = {"model": f"claude-code:{model}", "duration_s": round((out.get("duration_ms") or 0) / 1000, 1),
            "is_error": bool(out.get("is_error")), "error": str(out.get("result"))[:300] if out.get("is_error") else None}
    if out.get("is_error"):
        return [], {**meta, "dropped": []}
    rows, dropped = parse_proposals(out.get("result", "") or "", rule_rows, graph, model=model)
    meta["dropped"] = dropped
    meta["n_proposals"] = len(rows)
    return rows, meta


if __name__ == "__main__":   # demo: wet-gas compression train → rule rows + proposals → notebooks/hazop/
    from packages.dwsim_client import tools as T
    from packages.hazop.report import hazop_from_flowsheet, to_excel, to_markdown
    z = {"Methane": 0.84, "Ethane": 0.07, "Propane": 0.04, "N-butane": 0.02, "Carbon dioxide": 0.02, "Water": 0.01}
    fid = T.new_flowsheet("gas", list(z), "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300, pressure_Pa=50e5, molar_flow_mol_s=100, mole_fractions=z)
    for n in ("hp", "cold", "vap", "liq", "lp"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "compressor", "K-1", {"outlet_P_Pa": 60e5, "adiabatic_eff_pct": 75}); T.connect(fid, "K-1", feeds=["gas"], products=["hp"])
    T.add_unit(fid, "cooler", "E-1", {"outlet_T_K": 280}); T.connect(fid, "E-1", feeds=["hp"], products=["cold"])
    T.add_unit(fid, "separator", "V-1"); T.connect(fid, "V-1", feeds=["cold"], products=["vap", "liq"])
    T.add_unit(fid, "valve", "VLV-1", {"outlet_P_Pa": 15e5}); T.connect(fid, "VLV-1", feeds=["vap"], products=["lp"])
    assert T.solve(fid)["ok"]
    rule_rows, graph = hazop_from_flowsheet(T._s(fid)["fs"], "gas_compresion")
    props, meta = propose(rule_rows, graph, log=print)
    print(json.dumps({k: v for k, v in meta.items() if k != "dropped"}, ensure_ascii=False), f"dropped={len(meta['dropped'])}")
    for r in props:
        print(f"- [{r.node}] {r.parameter} / {r.deviation}: {r.consequences[:1]}")
    out = Path("notebooks/hazop"); out.mkdir(parents=True, exist_ok=True)
    (out / "gas_compresion_con_propuestas.md").write_text(to_markdown(rule_rows + props, "HAZOP preliminar — compresión de gas húmedo (reglas + propuestas LLM)"))
    to_excel(rule_rows + props, out / "gas_compresion_con_propuestas.xlsx")
    print("→", out / "gas_compresion_con_propuestas.md")
