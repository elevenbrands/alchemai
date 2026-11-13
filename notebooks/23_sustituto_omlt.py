"""Notebook 23 — surrogate optimisation over DWSIM (bloque M8c, `optimize_design_surrogate`, OMLT + ipopt).

A) Three-stage air compression 1 → 27 bar (notebook 18 flowsheet, PR, η = 75 %, intercooling to 300 K): minimise total
   power over the two intermediate pressures. Known optimum: optimize_design_multi directly on DWSIM (3.021/9.087 bar,
   891.84 kW), a dense DWSIM grid around it, and ideal-gas theory (3/9 bar).
B) Same with the HAZOP limit «stage-1 discharge ≤ 430 K» (direct: 2.699/8.591 bar, 892.68 kW).
C) Hexane-heptane column (M8b case: 100 mol/s 50/50 at 355 K, 1 atm, Raoult, 12 stages = condenser + 10 trays +
   reboiler, feed at stage 6): minimise reboiler duty over reflux ratio and bottoms flow with both purities ≥ 0.95.
   Known optimum: optimize_design_multi on the same DWSIM column and eo_column_optimize (IDAES + ipopt) on the same
   column (RR 2.291, B 50.0 mol/s, 4.164 MW; DWSIM at that point 4.187 MW).
Usage: PYTHONPATH=. uv run python notebooks/23_sustituto_omlt.py A|B|C|record
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_optimize import optimize_design_multi, optimize_design_surrogate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("nb18", ROOT / "notebooks" / "18_optimizacion_multivariable.py")
nb18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb18)
three_stage, VARS_K, OBJ_K = nb18.three_stage, nb18.VARS_K, nb18.OBJ_K
LIMIT_B = [{"kind": "stream", "name": "s1", "key": "T_K", "max": 430.0}]


def hexane_heptane_column():
    fid = T.new_flowsheet("hh", ["N-hexane", "N-heptane"], "Raoult")["flowsheet_id"]
    T.add_material_stream(fid, "feed", temperature_K=355.0, pressure_Pa=101325.0, molar_flow_mol_s=100.0,
                          compound_molar_flows_mol_s={"N-hexane": 50.0, "N-heptane": 50.0})
    for n in ("dist", "bot"):
        T.add_material_stream(fid, n)
    for n in ("cd", "rd"):
        T.add_energy_stream(fid, n)
    T.add_unit(fid, "distillation_column", "T-1", {"stages": 12, "feed_stage": 6, "reflux_ratio": 2.291,
                                                   "bottoms_molar_flow_mol_s": 50.0, "top_P_Pa": 101325.0})
    T.connect(fid, "T-1", feeds=["feed"], products=["dist", "bot"], energy_products=["cd"], energy_feeds=["rd"])
    return fid


VARS_C = [{"kind": "unit", "name": "T-1", "param": "reflux_ratio", "lower": 1.5, "upper": 4.0},
          {"kind": "unit", "name": "T-1", "param": "bottoms_molar_flow_mol_s", "lower": 45.0, "upper": 55.0}]
OBJ_C = {"type": "sum", "terms": [{"kind": "unit", "name": "T-1", "key": "energy_streams_kW.rd"}]}
# tolerance 0.002 on each purity = the accuracy asked of the surrogate in this test (assumption, MOCKS.md)
CONS_C = [{"kind": "stream", "name": "dist", "key": "mole_fractions.N-hexane", "min": 0.95, "tolerance": 0.002},
          {"kind": "stream", "name": "bot", "key": "mole_fractions.N-heptane", "min": 0.95, "tolerance": 0.002}]
EO_C = {"reflux_ratio": 2.291, "bottoms_mol_s": 50.0, "reboiler_MW": 4.164}


def _summary(r):
    keep = ("status", "surrogate_optimum", "dwsim_at_optimum", "holdout_error", "error_at_optimum", "best_dwsim_sample",
            "evidence", "ipopt", "variables_at_bound", "n_dwsim_solves", "elapsed_s", "dwsim_constraint_violation")
    return {k: r.get(k) for k in keep} | {"networks": r["method"]["networks"], "n_failed_solves": r["method"]["n_failed_solves"]}


def case(name, n_samples=None, samples=False):
    build, var, obj, cons = {"A": (three_stage, VARS_K, OBJ_K, None), "B": (three_stage, VARS_K, OBJ_K, LIMIT_B),
                             "C": (hexane_heptane_column, VARS_C, OBJ_C, CONS_C)}[name]
    fid = build()
    t = time.time()
    s = optimize_design_surrogate(fid, var, obj, cons, n_samples=n_samples, return_samples=samples)
    s_time = time.time() - t
    t = time.time()
    fid2 = build()
    d = optimize_design_multi(fid2, var, obj, constraints=cons, max_evaluations=150)
    d_time = time.time() - t
    T.close_flowsheet(fid); T.close_flowsheet(fid2)
    out = {"surrogate": _summary(s), "surrogate_s": round(s_time, 1), "direct": d["best"] | {"n_evaluations": d["n_evaluations"]},
           "direct_s": round(d_time, 1)}
    if samples:
        out["samples"] = s["samples"]
    return out


def dense_grid_A(n=21):
    """DWSIM on an n×n grid ±8 % around the direct optimum: the true optimum to grid resolution and how flat it is."""
    import numpy as np
    fid = three_stage()
    from packages.dwsim_client.tools_optimize import _set, _eval_current
    best = None
    rows = []
    for p1 in np.linspace(2.80e5, 3.25e5, n):
        for p2 in np.linspace(8.40e5, 9.80e5, n):
            _set(fid, "unit", "K-1", "outlet_P_Pa", p1); _set(fid, "unit", "K-2", "outlet_P_Pa", p2)
            r = _eval_current(fid, OBJ_K, [])
            rows.append((p1, p2, r["objective"]))
            if best is None or r["objective"] < best[2]:
                best = (p1, p2, r["objective"])
    T.close_flowsheet(fid)
    flat = [(a, b) for a, b, y in rows if y <= best[2] + 0.05]
    return {"best_P1_bar": best[0] / 1e5, "best_P2_bar": best[1] / 1e5, "best_kW": best[2], "grid_step_bar": [0.45 / (n - 1), 1.4 / (n - 1)],
            "within_0.05kW_P1_bar": [min(a for a, _ in flat) / 1e5, max(a for a, _ in flat) / 1e5],
            "within_0.05kW_P2_bar": [min(b for _, b in flat) / 1e5, max(b for _, b in flat) / 1e5]}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "A"
    if arg == "record":   # recorded DWSIM samples for the fast tests
        for name in ("A", "C"):
            fid = {"A": three_stage, "C": hexane_heptane_column}[name]()
            var, obj, cons = {"A": (VARS_K, OBJ_K, None), "C": (VARS_C, OBJ_C, CONS_C)}[name]
            r = optimize_design_surrogate(fid, var, obj, cons, return_samples=True)
            T.close_flowsheet(fid)
            json.dump(r["samples"] | {"case": name, "status": r["status"], "dwsim_at_optimum": r.get("dwsim_at_optimum"),
                                      "surrogate_optimum": r.get("surrogate_optimum")},
                      open(ROOT / "tests" / "data" / f"surrogate_samples_{name}.json", "w"), indent=0, default=float)
            print(name, r["status"])
    elif arg == "grid":
        print(json.dumps(dense_grid_A(), indent=1))
    else:
        print(json.dumps(case(arg), indent=1, default=float))
