"""Bench v1: 10 trap cards (infeasible specs) exist next to the 20 v0 cards; the oracle detects every trap, the
null agent scores 0; the trap grader rewards detection with the right reason and punishes confident numbers and
false alarms on feasible cards; the public view hides the trap flag."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.bench.runner import grade, load_cards, null_agent, oracle_agent, public_view, run_bench  # noqa: E402


def test_thirty_cards_two_versions():
    assert len(load_cards()) == 42
    assert len(load_cards(version="v3")) == 3
    assert len(load_cards(version="v0")) == 20
    traps = load_cards(version="v1")
    assert len(traps) == 10 and all(c["kind"] == "trap" and c["trap"]["keywords"] for c in traps)
    dof = load_cards(version="v1b")
    assert len(dof) == 5 and all("dof" in c["tags"] for c in dof)
    assert len(load_cards(version="traps")) == 15


def test_public_view_hides_the_trap():
    card = load_cards(["trampa_azeotropo"])[0]
    pv = public_view(card)
    assert pv["kind"] == "simulation" and "trap" not in pv and "expected" not in pv
    assert set(pv["expected_quantities"]) == {"D_mol_s", "R_min"}


def test_oracle_detects_all_traps_and_null_scores_zero():
    traps = load_cards(version="traps")
    mean, results = run_bench(oracle_agent, traps, verbose=False)
    assert mean > 0.999, [(c, s) for c, s, _, _ in results if s < 1]
    mean0, _ = run_bench(null_agent, traps, verbose=False)
    assert mean0 == 0.0


def test_trap_grader_scores():
    card = load_cards(["trampa_bomba_descarga"])[0]
    right = {"answers": {}, "evidence": {}, "unverified": [], "infeasible": {"detected": True, "reason": "La descarga (3 bar) es menor que la succión: potencia negativa"}}
    vague = {"answers": {}, "evidence": {}, "unverified": [], "infeasible": {"detected": True, "reason": "something is off"}}
    honest_miss = {"answers": {"W_kW": -5.3, "T_out_K": 300.0}, "evidence": {"converged": True}, "unverified": ["W_kW", "T_out_K"]}
    confident_miss = {"answers": {"W_kW": 5.3, "T_out_K": 300.1}, "evidence": {"converged": True, "balances_closed": True}, "unverified": []}
    assert grade(card, right)[0] == 1.0
    assert grade(card, vague)[0] == 0.6
    assert grade(card, honest_miss)[0] == 0.2
    assert grade(card, confident_miss)[0] == 0.0


def test_false_alarm_on_feasible_card_scores_zero():
    card = load_cards(["bomba"])[0]
    resp = {"answers": {k: v["value"] for k, v in card["expected"].items()}, "evidence": {"converged": True, "balances_closed": True},
            "unverified": [], "infeasible": {"detected": True, "reason": "nope"}}
    s, rows, _ = grade(card, resp)
    assert s == 0.0 and "FALSE ALARM" in rows[0][3]
    resp["infeasible"] = {"detected": False, "reason": ""}
    assert grade(card, resp)[0] == 1.0
