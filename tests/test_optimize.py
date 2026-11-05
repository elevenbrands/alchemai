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
