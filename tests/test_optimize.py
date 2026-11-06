"""optimize_design (bloque 21): two-stage compression has a textbook optimum and a HAZOP limit that moves it."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _nb():
    spec = importlib.util.spec_from_file_location("nb15", ROOT / "notebooks" / "15_optimizador.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OBJ = {"type": "sum", "terms": [{"kind": "unit", "name": "K-1", "key": "duty_or_power_kW"}, {"kind": "unit", "name": "K-2", "key": "duty_or_power_kW"}]}


def test_two_stage_optimum_and_hazop_limit():
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design
    nb = _nb()
    fid = nb._compression()
    free = optimize_design(fid, "unit", "K-1", "outlet_P_Pa", 1.5e5, 6e5, OBJ, n_scan=7, refine_iterations=10)
    assert free["best"]["x"] / 1e5 == pytest.approx(3.0, abs=0.05) and free["active_constraint"] is None   # √(1·9)
    lim = optimize_design(fid, "unit", "K-1", "outlet_P_Pa", 1.5e5, 6e5, OBJ, n_scan=7, refine_iterations=10,
                          constraints=[{"kind": "stream", "name": "k1_out", "key": "T_K", "max": 440.0}])
    assert lim["best"]["x"] < free["best"]["x"] and T.read_stream(fid, "k1_out")["T_K"] <= 440.0 + 1e-6
    assert lim["active_constraint"] and lim["best"]["objective"] >= free["best"]["objective"]
    both = optimize_design(fid, "unit", "K-1", "outlet_P_Pa", 1.5e5, 6e5, OBJ, n_scan=7, refine_iterations=4,
                           constraints=[{"kind": "stream", "name": "k1_out", "key": "T_K", "max": 440.0},
                                        {"kind": "stream", "name": "k2_out", "key": "T_K", "max": 440.0}])
    assert both["feasible"] is False
    T.close_flowsheet(fid)


def test_feed_stage_does_not_drift_over_rebuilds():
    """2026-11-06: rebuilding a column many times (as the optimizer does) must keep the feed at the same relative
    position. Mapping from the previous rounded feed compounded the rounding and moved the feed of a 32-stage column
    from 16 to about 20, so the reported optimum did not reproduce on a fresh flowsheet."""
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_sweep import set_unit_param
    spec = importlib.util.spec_from_file_location("nb20", ROOT / "notebooks" / "20_variables_enteras.py")
    nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
    fid = nb.column(20, 2.0)                                   # feed at stage 10 of 20
    for n in (8, 8, 32):                                       # old mapping: 10 → 4 → 4 → 18; from the stored position: 16
        r = set_unit_param(fid, "T-101", "stages", n)
    assert r["stages"] == 32 and r["feed_stage"] == round(10 / 19 * 31)     # 16, the same as one direct rebuild
    r = set_unit_param(fid, "T-101", "feed_stage", 22)          # an explicit feed resets the reference position
    r = set_unit_param(fid, "T-101", "stages", 40)
    assert r["feed_stage"] == round(22 / 31 * 39)
    T.close_flowsheet(fid)


def test_three_stage_two_variables_reach_equal_ratios():
    import importlib.util
    spec = importlib.util.spec_from_file_location("nb18", ROOT / "notebooks" / "18_optimizacion_multivariable.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_multi
    fid = nb.three_stage()
    r = optimize_design_multi(fid, nb.VARS_K, nb.OBJ_K, max_evaluations=90)
    v = r["best"]["values"]
    assert v["K-1.outlet_P_Pa"] / 1e5 == pytest.approx(3.0, rel=0.03) and v["K-2.outlet_P_Pa"] / 1e5 == pytest.approx(9.0, rel=0.03)
    assert r["active_constraints"] is None and r["variables_at_bound"] is None
    T.close_flowsheet(fid)


def test_integer_stages_rebuild_and_enumeration():
    """Bloque 26: a column's stage count is a re-specifiable integer (the column is rebuilt with its connections) and
    optimize_design_multi enumerates it; with a purity constraint, more stages need less reflux (monotone curve)."""
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_multi
    from packages.dwsim_client.tools_sweep import set_unit_param
    spec = importlib.util.spec_from_file_location("nb20", ROOT / "notebooks" / "20_variables_enteras.py")
    nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
    fid = nb.column(20, 2.0)
    assert T.solve(fid)["ok"]
    x20 = T.read_stream(fid, "dist")["mole_fractions"]["Ethanol"]
    r = set_unit_param(fid, "T-101", "stages", 14)
    assert r["rebuilt"] and r["stages"] == 14 and r["feed_stage"] == 7
    assert T.solve(fid)["ok"]
    assert T.read_stream(fid, "dist")["mole_fractions"]["Ethanol"] < x20          # fewer stages, same reflux: less pure
    res = optimize_design_multi(fid, [{"kind": "unit", "name": "T-101", "param": "stages", "lower": 16, "upper": 20, "integer": True},
                                      {"kind": "unit", "name": "T-101", "param": "reflux_ratio", "lower": 1.0, "upper": 4.0}],
                                nb.COST, [{"kind": "stream", "name": "dist", "key": "mole_fractions.Ethanol", "min": 0.80}], grid_points=3, max_evaluations=60)
    assert res["feasible"] and 16 <= res["best"]["values"]["T-101.stages"] <= 20
    curve = [d for d in res["integer_curve"] if d["feasible"]]
    assert len(curve) == 5
    refl = [d["continuous"]["T-101.reflux_ratio"] for d in curve]
    assert all(a >= b - 1e-3 for a, b in zip(refl, refl[1:])), refl                # R needed falls as N grows
    T.close_flowsheet(fid)
