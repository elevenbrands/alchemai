"""LLM layer of the HAZOP: proposals are parsed into `proposal` rows, unknown nodes are dropped, rule rows are
never touched, and the tool flag appends them. The model call is faked (no subscription needed); the live demo is
`uv run python -m packages.hazop.llm` (evidence in notebooks/hazop/gas_compresion_con_propuestas.md)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.hazop.llm import build_prompt, parse_proposals, propose  # noqa: E402
from packages.hazop.rules import HazopRow  # noqa: E402

GRAPH = {"name": "g", "units": {"K-1": {"kind": "Compressor", "inlets": ["gas"], "outlets": ["hp"], "duty_kW": 500.0, "T_in_K": 300, "T_out_K": 322,
                                        "P_in_Pa": 50e5, "P_out_Pa": 60e5, "exothermic": False}},
         "streams": {"gas": {"T_K": 300.0, "P_Pa": 50e5, "vapor_fraction": 1.0, "mass_flow_kg_s": 2.0, "molar_flow_mol_s": 100.0,
                             "mole_fractions": {"Methane": 0.98, "Water": 0.02}, "flammable": True, "toxic": False, "has_water": True, "hydrate_risk": False}}}
RULES = [HazopRow("K-1", "Compressor", "Flow", "Less flow", ["x"], ["Surge"], ["anti-surge"], 3, "P 50→60 bar")]
FAKE = {"result": "text before\n```json\n" + json.dumps({"proposals": [
    {"node": "K-1", "parameter": "Utility", "deviation": "Loss of power", "causes": ["falla eléctrica"], "consequences": ["settle-out"], "safeguards": ["SDV"], "rationale": "causa común"},
    {"node": "P-99", "parameter": "Flow", "deviation": "No flow", "causes": ["x"], "consequences": ["y"], "safeguards": []},
    {"node": "K-1", "parameter": "Flow", "deviation": "empty", "causes": [], "consequences": []},
]}) + "\n```", "duration_ms": 1200, "is_error": False}


def test_parse_keeps_known_nodes_only_and_inherits_severity():
    rows, dropped = parse_proposals(FAKE["result"], RULES, GRAPH, model="fake")
    assert len(rows) == 1 and rows[0].source == "proposal" and rows[0].node == "K-1" and rows[0].severity == 3
    assert rows[0].kind == "Compressor" and "fake" in rows[0].evidence
    assert [d["reason"] for d in dropped] == ["unknown node 'P-99'", "empty proposal"]


def test_propose_with_fake_runner_never_alters_rules():
    before = [r.to_dict() for r in RULES]
    captured = {}

    def fake_runner(prompt, system, **kw):
        captured["prompt"], captured["kw"] = prompt, kw
        return FAKE
    rows, meta = propose(RULES, GRAPH, _runner=fake_runner)
    assert [r.to_dict() for r in RULES] == before
    assert meta["n_proposals"] == 1 and len(meta["dropped"]) == 2 and not meta["is_error"]
    assert captured["kw"]["mcp"] is False and captured["kw"]["max_turns"] == 1
    assert "K-1" in captured["prompt"] and "Surge" in captured["prompt"]


def test_propose_error_path_is_empty_not_raising():
    rows, meta = propose(RULES, GRAPH, _runner=lambda *a, **k: {"is_error": True, "result": "timeout after 300 s"})
    assert rows == [] and meta["is_error"] and "timeout" in meta["error"]


def test_bad_json_is_dropped():
    rows, dropped = parse_proposals("no json here", RULES, GRAPH)
    assert rows == [] and dropped[0]["reason"] == "no valid JSON"


def test_prompt_is_compact():
    p = build_prompt(RULES, GRAPH)
    assert "P_bar" in p and '"P_Pa"' not in p and len(p) < 4000
