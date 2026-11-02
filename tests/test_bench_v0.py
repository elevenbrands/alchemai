"""Bench v0: 20 cards exist, the oracle agent scores 100 %, the null agent 0 %, and the grader
penalises confident wrong answers more than flagged ones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.bench.runner import grade, load_cards, null_agent, oracle_agent, run_bench  # noqa: E402


def test_twenty_cards():
    cards = load_cards()
    assert len(cards) == 20
    kinds = {c["kind"] for c in cards}
    assert kinds == {"simulation", "design"}
    assert sum(c["kind"] == "design" for c in cards) == 7


def test_null_agent_scores_zero():
    mean, _ = run_bench(null_agent, load_cards(), verbose=False)
    assert mean == 0.0


def test_oracle_agent_scores_full():
    mean, results = run_bench(oracle_agent, load_cards(), verbose=False)
    bad = [(cid, s) for cid, s, _, _ in results if s < 0.999]
    assert not bad, bad
    assert mean > 0.999


def test_grader_penalises_confident_errors():
    card = load_cards(["bomba"])[0]
    good = {"answers": {"W_kW": card["expected"]["W_kW"]["value"], "T_out_K": 400.0}, "evidence": {"converged": True, "balances_closed": True}, "unverified": ["T_out_K"]}
    bad = {"answers": {"W_kW": card["expected"]["W_kW"]["value"], "T_out_K": 400.0}, "evidence": {"converged": True, "balances_closed": True}, "unverified": []}
    s_good, _, _ = grade(card, good)
    s_bad, _, _ = grade(card, bad)
    assert s_good == 0.5 and s_bad == 0.25


def test_grader_caps_without_evidence():
    card = load_cards(["bomba"])[0]
    resp = {"answers": {k: v["value"] for k, v in card["expected"].items()}, "evidence": {}, "unverified": []}
    s, _, missing = grade(card, resp)
    assert s == 0.8 and missing
