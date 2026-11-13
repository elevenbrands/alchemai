"""optimize_design_surrogate (bloque M8c): neural-network surrogate of a DWSIM flowsheet, OMLT + ipopt, re-checked in DWSIM.

Fast tests use an analytic flowsheet stand-in (ideal-gas three-stage compressor, exact optimum known) and DWSIM samples
recorded by notebooks/23_sustituto_omlt.py (tests/data). One DWSIM end-to-end case runs in the suite (≈ 6 s); the
column and the direct-optimisation comparisons run with ALCHEMAI_SLOW_SURROGATE=1.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.eo import column_optimize as C  # noqa: E402
from packages.eo import surrogate as S  # noqa: E402

needs_ipopt = pytest.mark.skipif(not C.ipopt_available(), reason="ipopt not installed (uv run idaes get-extensions)")
DATA = ROOT / "tests" / "data"
VARS = [{"kind": "unit", "name": "K-1", "param": "outlet_P_Pa", "lower": 1.5e5, "upper": 8e5, "log_scale": True},
        {"kind": "unit", "name": "K-2", "param": "outlet_P_Pa", "lower": 4e5, "upper": 20e5, "log_scale": True}]
A, ETA, T1 = 0.4 / 1.4, 0.75, 300.0


def ideal_gas_compressor(x):
    """Ideal-gas, constant cp, intercooled 1 → 27 bar: optimum = equal ratios (3, 9 bar) exactly."""
    p1, p2 = x
    rs = [p1 / 1e5, p2 / p1, 27e5 / p2]
    return {"solved_ok": True, "objective": sum(2.0 * 1.005 * T1 * (r ** A - 1) / ETA for r in rs),
            "constraints": {"s1.T_K": T1 * (1 + ((p1 / 1e5) ** A - 1) / ETA)}}


@needs_ipopt
def test_analytic_known_optimum_free_and_constrained():
    r = S.surrogate_optimize(ideal_gas_compressor, VARS, {})
    v = r["surrogate_optimum"]["values"]
    assert r["status"] == "verified" and r["evidence"]["verified"]
    assert v["K-1.outlet_P_Pa"] / 1e5 == pytest.approx(3.0, rel=0.005) and v["K-2.outlet_P_Pa"] / 1e5 == pytest.approx(9.0, rel=0.005)
    assert r["holdout_error"]["objective"]["within"] and r["embedding_max_rel_diff"] <= 1e-6
    # with stage-1 discharge ≤ 430 K: r1 = (1 + 0.75·(430/300 − 1))^(1/a), P2 = √(P1·27 bar)
    p1 = (1 + ETA * (430 / T1 - 1)) ** (1 / A)
    r = S.surrogate_optimize(ideal_gas_compressor, VARS, {"s1.T_K": {"max": 430.0}})
    v = r["surrogate_optimum"]["values"]
    assert r["status"] == "verified"
    assert v["K-1.outlet_P_Pa"] / 1e5 == pytest.approx(p1, rel=0.003)
    assert v["K-2.outlet_P_Pa"] / 1e5 == pytest.approx(math.sqrt(p1 * 27), rel=0.005)
    assert r["dwsim_constraint_violation"]["s1.T_K"] <= r["error_at_optimum"]["s1.T_K"]["tolerance_abs"]
    json.dumps(r)                                                  # the agent serialises the result


@needs_ipopt
def test_disagreement_at_optimum_is_not_verified():
    """The surrogate is never trusted: if the flowsheet disagrees at the proposed optimum, the result says so."""
    calls = {"n": 0}
    n_design = 20 + 15 * 2 + 4 + max(10, 50 // 4)

    def biased(x):
        calls["n"] += 1
        r = ideal_gas_compressor(x)
        if calls["n"] > n_design:                                  # only the verification run disagrees
            r["objective"] *= 1.05
            r["constraints"]["s1.T_K"] += 20.0
        return r

    r = S.surrogate_optimize(biased, VARS, {"s1.T_K": {"max": 430.0}})
    assert calls["n"] == n_design + 1
    assert r["status"] == "not_verified" and not r["verified"]
    ev = r["evidence"]
    assert ev["holdout_within_tolerance"] and not ev["error_at_optimum_within_tolerance"]
    assert not ev["dwsim_constraints_met_within_tolerance"] and "NO verificado" in r["note"]


@needs_ipopt
def test_unsolved_optimum_is_not_verified():
    calls = {"n": 0}

    def fails_last(x):
        calls["n"] += 1
        return ideal_gas_compressor(x) if calls["n"] <= 66 else {"solved_ok": False, "objective": None, "constraints": {}}

    r = S.surrogate_optimize(fails_last, VARS, {})
    assert r["status"] == "not_verified" and r["evidence"]["dwsim_solved_at_optimum"] is False


def _recorded(name):
    return json.loads((DATA / f"surrogate_samples_{name}.json").read_text())


@needs_ipopt
def test_recorded_dwsim_compressor_reaches_direct_optimum():
    """DWSIM samples (PR, η 75 %) recorded on 2026-11-13; direct optimize_design_multi on DWSIM: 3.021/9.087 bar,
    891.837 kW; dense DWSIM grid: 3.025/9.100 bar (step 0.0225/0.07 bar)."""
    rec = _recorded("A")
    r = S.surrogate_optimize(lambda x: {"solved_ok": True, **rec["dwsim_at_optimum"]}, VARS, {}, recorded=rec)
    v = r["surrogate_optimum"]["values"]
    assert r["holdout_error"]["objective"]["max_abs_error"] < 0.5 and r["holdout_error"]["objective"]["within"]
    assert v["K-1.outlet_P_Pa"] / 1e5 == pytest.approx(3.021, rel=0.005) and v["K-2.outlet_P_Pa"] / 1e5 == pytest.approx(9.087, rel=0.005)
    assert v == pytest.approx(rec["surrogate_optimum"]["values"], rel=1e-3)          # the recorded run is reproduced
    assert r["dwsim_at_optimum"]["objective"] == pytest.approx(891.837, abs=0.01)
    assert r["status"] == "verified"


@needs_ipopt
def test_recorded_dwsim_column_purities_and_optimum():
    """Hexane-heptane DWSIM column (Raoult, 12 stages): purities ≥ 0.95 with surrogate tolerance 0.002. Direct
    optimize_design_multi on the same DWSIM column: RR 2.2322, B 50.001 mol/s, 4107.9 kW."""
    rec = _recorded("C")
    limits = {"dist.mole_fractions.N-hexane": {"min": 0.95, "tolerance": 0.002},
              "bot.mole_fractions.N-heptane": {"min": 0.95, "tolerance": 0.002}}
    var = [{"kind": "unit", "name": "T-1", "param": "reflux_ratio", "lower": 1.5, "upper": 4.0},
           {"kind": "unit", "name": "T-1", "param": "bottoms_molar_flow_mol_s", "lower": 45.0, "upper": 55.0}]
    r = S.surrogate_optimize(lambda x: {"solved_ok": True, **rec["dwsim_at_optimum"]}, var, limits, recorded=rec)
    for k in limits:
        assert r["holdout_error"][k]["max_abs_error"] < 0.002
    v = r["surrogate_optimum"]["values"]
    assert v["T-1.reflux_ratio"] == pytest.approx(2.2322, rel=0.005)
    assert v["T-1.bottoms_molar_flow_mol_s"] == pytest.approx(50.0, abs=0.1)
    assert r["status"] == "verified"


def test_network_folding_and_omlt_definition():
    rng = np.random.default_rng(1)
    Z = rng.random((40, 2))
    y = np.sin(3 * Z[:, 0]) + Z[:, 1] ** 2
    net, info = S.fit_surrogate(Z, y)
    zz = rng.random((5, 2))
    a = zz
    layers = net.folded_layers()
    for i, (W, b) in enumerate(layers):
        a = a @ W + b
        if i < len(layers) - 1:
            a = np.tanh(a)
    assert np.allclose(a[:, 0], net.predict(zz), rtol=1e-12, atol=1e-12)
    nd = S.network_definition(net)
    assert len(list(nd.layers)) == len(layers) + 1 and info["cv_rmse"] < 0.05


def test_inputs_rejected_before_any_flowsheet_call():
    called = []
    ev = lambda x: called.append(x)
    with pytest.raises(ValueError, match="enteras"):
        S.surrogate_optimize(ev, [VARS[0] | {"integer": True}], {})
    with pytest.raises(ValueError, match="cota"):
        S.surrogate_optimize(ev, [VARS[0] | {"lower": 9e5}], {})
    with pytest.raises(ValueError, match="log"):
        S.surrogate_optimize(ev, [{"name": "a", "param": "b", "lower": 0.0, "upper": 1.0, "log_scale": True}], {})
    from packages.dwsim_client import tools as T  # noqa: F401  (tools first: tools_optimize imports it back)
    from packages.dwsim_client import tools_optimize as TO
    with pytest.raises(ValueError, match="max o min"):
        TO.optimize_design_surrogate("none", VARS, {"type": "sum", "terms": []}, [{"kind": "stream", "name": "s1", "key": "T_K"}])
    assert not called


def test_tool_is_wired():
    from packages.agent import agent as AG
    from packages.dwsim_client import tools_optimize as TO
    assert TO.optimize_design_surrogate in TO.OPTIMIZE_TOOLS
    assert any(getattr(t, "name", "") == "optimize_design_surrogate" for t in AG.AGENT_TOOLS)


def _nb23():
    import importlib.util
    spec = importlib.util.spec_from_file_location("nb23", ROOT / "notebooks" / "23_sustituto_omlt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@needs_ipopt
def test_dwsim_end_to_end_constrained_compressor():
    """Real DWSIM: stage-1 discharge ≤ 430 K. Direct optimize_design_multi: 2.699/8.591 bar, 892.68 kW."""
    nb = _nb23()
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_surrogate
    fid = nb.three_stage()
    r = optimize_design_surrogate(fid, nb.VARS_K, nb.OBJ_K, nb.LIMIT_B)
    assert r["status"] == "verified", r["evidence"]
    v = r["surrogate_optimum"]["values"]
    assert v["K-1.outlet_P_Pa"] / 1e5 == pytest.approx(2.699, rel=0.005) and v["K-2.outlet_P_Pa"] / 1e5 == pytest.approx(8.591, rel=0.005)
    assert r["dwsim_at_optimum"]["objective"] == pytest.approx(892.68, abs=0.1)
    assert T.read_stream(fid, "s1")["T_K"] == pytest.approx(r["dwsim_at_optimum"]["constraints"]["s1.T_K"])   # left at the optimum
    T.close_flowsheet(fid)


@pytest.mark.skipif(os.environ.get("ALCHEMAI_SLOW_SURROGATE") != "1" or not C.ipopt_available(),
                    reason="DWSIM surrogate vs direct optimisation (≈ 40 s): ALCHEMAI_SLOW_SURROGATE=1")
def test_dwsim_surrogate_vs_direct_compressor_and_column():
    nb = _nb23()
    a = nb.case("A")
    sa, da = a["surrogate"], a["direct"]
    assert sa["status"] == "verified"
    for k in ("K-1.outlet_P_Pa", "K-2.outlet_P_Pa"):
        assert sa["surrogate_optimum"]["values"][k] == pytest.approx(da["values"][k], rel=0.005)
    assert sa["dwsim_at_optimum"]["objective"] == pytest.approx(da["objective"], abs=0.05)
    c = nb.case("C")
    sc, dc = c["surrogate"], c["direct"]
    assert sc["status"] == "verified"
    assert sc["surrogate_optimum"]["values"]["T-1.reflux_ratio"] == pytest.approx(dc["values"]["T-1.reflux_ratio"], rel=0.005)
    assert sc["dwsim_at_optimum"]["objective"] == pytest.approx(dc["objective"], rel=0.005)
    # eo_column_optimize (IDAES, chemicals data) on the same column: RR 2.291; DWSIM's own optimum is 2.6 % lower in RR
    # and 1.4 % lower in duty (the two property bases differ; DWSIM at the EO point gives purities 0.952/0.951)
    assert sc["surrogate_optimum"]["values"]["T-1.reflux_ratio"] == pytest.approx(nb.EO_C["reflux_ratio"], rel=0.04)
