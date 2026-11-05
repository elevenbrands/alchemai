"""Notebook 18 — multivariable optimizer `optimize_design_multi` (bloque 24).

A) Three-stage air compression 1 → 27 bar, intercooling to 300 K, η = 75 %, PR: minimize total power over the two
   intermediate pressures. Ideal-gas theory: equal ratios, 3 and 9 bar. The variables interact (a valley), which defeats
   one-at-a-time search.
B) Same with the HAZOP limit «stage-1 discharge ≤ 430 K».
C) Propane absorber (notebook 16 case, 6 stages): oil flow and oil chilling temperature vs an ILLUSTRATIVE annual cost
   = propane lost + solvent circulation + chilling duty (MOCKS.md).
Usage: PYTHONPATH=. uv run python notebooks/18_optimizacion_multivariable.py A|B|C
"""
import json
import math
import sys

sys.path.insert(0, ".")
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_optimize import optimize_design_multi  # noqa: E402

AIR = {"Nitrogen": 0.79, "Oxygen": 0.21}


def three_stage():
    fid = T.new_flowsheet("k3", ["Nitrogen", "Oxygen"], "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "s0", temperature_K=300.0, pressure_Pa=1e5, mass_flow_kg_s=2.0, mole_fractions=AIR)
    for n in ("s1", "c1", "s2", "c2", "s3"):
        T.add_material_stream(fid, n)
    for e in ("w1", "w2", "w3", "q1", "q2"):
        T.add_energy_stream(fid, e)
    for k, (i, o, w, p) in enumerate((("s0", "s1", "w1", 3e5), ("c1", "s2", "w2", 9e5), ("c2", "s3", "w3", 27e5)), 1):
        T.add_unit(fid, "compressor", f"K-{k}", {"outlet_P_Pa": p, "adiabatic_eff_pct": 75.0})
        T.connect(fid, f"K-{k}", feeds=[i], products=[o], energy_feeds=[w])
    for k, (i, o, q) in enumerate((("s1", "c1", "q1"), ("s2", "c2", "q2")), 1):
        T.add_unit(fid, "cooler", f"IC-{k}", {"outlet_T_K": 300.0})
        T.connect(fid, f"IC-{k}", feeds=[i], products=[o], energy_products=[q])
    return fid


OBJ_K = {"type": "sum", "terms": [{"kind": "unit", "name": f"K-{k}", "key": "duty_or_power_kW"} for k in (1, 2, 3)]}
VARS_K = [{"kind": "unit", "name": "K-1", "param": "outlet_P_Pa", "lower": 1.5e5, "upper": 8e5, "log_scale": True},
          {"kind": "unit", "name": "K-2", "param": "outlet_P_Pa", "lower": 4e5, "upper": 20e5, "log_scale": True}]


def case(constraints):
    fid = three_stage()
    r = optimize_design_multi(fid, VARS_K, OBJ_K, constraints=constraints)
    out = {"feasible": r["feasible"], "n_evaluations": r["n_evaluations"]}
    if r["feasible"]:
        v = r["best"]["values"]
        out.update(P1_bar=v["K-1.outlet_P_Pa"] / 1e5, P2_bar=v["K-2.outlet_P_Pa"] / 1e5, total_kW=r["best"]["objective"],
                   T_discharge_K=[T.read_stream(fid, s)["T_K"] for s in ("s1", "s2", "s3")], active=r["active_constraints"],
                   at_bound=r["variables_at_bound"], theory_bar=[3.0, 9.0])
    T.close_flowsheet(fid)
    return out


ILLUSTRATIVE = {"propane_usd_kg": 0.5, "solvent_usd_per_kg_circulated": 0.0005, "chill_usd_GJ": 10.0, "hours": 8000.0}


def absorber():
    from packages.costing.heat_recovery import ILLUSTRATIVE_PRICES  # noqa: F401  (kept for the MOCKS cross-reference)
    comps = ["Methane", "Propane", "N-decane"]
    fid = T.new_flowsheet("abs18", comps, "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=100.0,
                          mole_fractions={"Methane": 0.98, "Propane": 0.02, "N-decane": 0.0})
    T.add_material_stream(fid, "oil", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=120.0,
                          mole_fractions={"Methane": 0.0, "Propane": 0.0, "N-decane": 1.0})
    for n in ("oil_cold", "top", "bot"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "qc")
    T.add_unit(fid, "cooler", "CH-1", {"outlet_T_K": 290.0})
    T.connect(fid, "CH-1", feeds=["oil"], products=["oil_cold"], energy_products=["qc"])
    T.add_unit(fid, "absorption_column", "ABS", {"stages": 6, "top_P_Pa": 10e5})
    T.connect(fid, "ABS", feeds=["oil_cold", "gas"], products=["top", "bot"])
    p = ILLUSTRATIVE
    s = p["hours"] * 3600.0
    obj = {"type": "sum", "terms": [
        {"kind": "stream", "name": "top", "key": "component_flow.Propane", "weight": p["propane_usd_kg"] * 0.044097 * s},     # USD/y per mol/s lost
        {"kind": "stream", "name": "oil", "key": "mass_flow_kg_s", "weight": p["solvent_usd_per_kg_circulated"] * s},     # USD/y per kg/s of oil
        {"kind": "unit", "name": "CH-1", "key": "heat_removed_kW", "weight": p["chill_usd_GJ"] * s / 1e6}]}                 # USD/y per kW
    vars_ = [{"kind": "stream", "name": "oil", "param": "molar_flow_mol_s", "lower": 40.0, "upper": 300.0, "log_scale": True},
             {"kind": "unit", "name": "CH-1", "param": "outlet_T_K", "lower": 250.0, "upper": 299.0}]
    r = optimize_design_multi(fid, vars_, obj, max_evaluations=120)
    v = r["best"]["values"]
    out = {"oil_mol_s": v["oil.molar_flow_mol_s"], "oil_T_K": v["CH-1.outlet_T_K"], "annual_cost_usd_y": r["best"]["objective"],
           "terms_usd_y": r["best"]["terms"], "n_evaluations": r["n_evaluations"], "at_bound": r["variables_at_bound"],
           "propane_recovery": 1.0 - T.read_stream(fid, "top")["molar_flow_mol_s"] * T.read_stream(fid, "top")["mole_fractions"]["Propane"] / 2.0}
    T.close_flowsheet(fid)
    return out


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "A"
    res = {"A": lambda: case([]), "B": lambda: case([{"kind": "stream", "name": "s1", "key": "T_K", "max": 430.0}]),
           "C": absorber}[part]()
    print(json.dumps({"part": part, **res}, default=float), flush=True)
