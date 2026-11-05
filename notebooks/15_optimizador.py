"""Notebook 15 — generic optimizer `optimize_design` (bloque 21), two checks with an independent reference.

A) Heat recovery rebuilt with agent tools only (add_unit heat_exchanger UA, heater, cooler) and optimized on UA with
   objective total_annual_cost + exchanger_specs: must land where notebook 14's hand-written loop did (~650 m²).
B) Two-stage air compression 1 → 9 bar with intercooling to 300 K, η = 75 %: minimize total power over the intermediate
   pressure. Ideal-gas theory: P_mid = √(P1·P2) = 3 bar. Then add the HAZOP limit «compressor discharge ≤ 440 K» on
   stage 1 and see the optimum move to the limit (stage 1 alone at 3 bar would discharge ≈ 447 K).
Usage: PYTHONPATH=. uv run python notebooks/15_optimizador.py [A|B]   (JSON lines)
"""
import json
import sys

sys.path.insert(0, ".")
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_optimize import optimize_design  # noqa: E402
from packages.costing.heat_recovery import ILLUSTRATIVE_PRICES  # noqa: E402


def case_a():
    fid = T.new_flowsheet("opt_hr", ["Water"], "SteamTables")["flowsheet_id"]
    w = {"Water": 1.0}
    T.add_material_stream(fid, "feed", temperature_K=300.0, pressure_Pa=4e5, mass_flow_kg_s=10.0, mole_fractions=w)
    T.add_material_stream(fid, "prod", temperature_K=400.0, pressure_Pa=4e5, mass_flow_kg_s=10.0, mole_fractions=w)
    for n in ("feed_pre", "prod_pre", "feed_out", "prod_out"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "heat_exchanger", "E-1", {"UA_W_K": 85000.0})
    T.connect(fid, "E-1", feeds=["prod", "feed"], products=["prod_pre", "feed_pre"])
    T.add_unit(fid, "heater", "H-1", {"outlet_T_K": 395.0}); T.connect(fid, "H-1", feeds=["feed_pre"], products=["feed_out"])
    T.add_unit(fid, "cooler", "C-1", {"outlet_T_K": 305.0}); T.connect(fid, "C-1", feeds=["prod_pre"], products=["prod_out"])
    cs = {"material": "CarbonSteelCarbonSteel"}
    obj = {"type": "total_annual_cost", "prices": ILLUSTRATIVE_PRICES, "hours_per_year": 8000.0, "interest": 0.10, "years": 10,
           "exchanger_specs": {"E-1": {"U_W_m2K": 850.0}, "H-1": {"U_W_m2K": 1100.0, "utility_in_K": 450.0},
                               "C-1": {"U_W_m2K": 850.0, "utility_in_K": 290.0, "utility_out_K": 300.0}},
           "options": {"E-1": cs, "H-1": cs, "C-1": cs}}
    r = optimize_design(fid, "unit", "E-1", "UA_W_K", 8500.0, 765000.0, obj, log_scale=True, n_scan=12, refine_iterations=12)
    b = r["best"]
    return {"case": "A recuperación de calor", "UA_opt_W_K": b["x"], "area_opt_m2": b["x"] / 850.0, "TAC_usd_y": b["objective"],
            "approach_K": T.read_stream(fid, "prod_pre")["T_K"] - 300.0, "flat_range_m2": [x / 850.0 for x in r["flat_range"]],
            "n_eval": r["n_evaluations"], "at_bound": r["at_bound"], "reference_nb14": {"area_m2": 650.3, "TAC_usd_y": 129991.7, "approach_K": 7.12}}


def _compression():
    fid = T.new_flowsheet("opt_k", ["Nitrogen", "Oxygen"], "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "air", temperature_K=300.0, pressure_Pa=1e5, mass_flow_kg_s=2.0, mole_fractions={"Nitrogen": 0.79, "Oxygen": 0.21})
    for n in ("k1_out", "ic_out", "k2_out"):
        T.add_material_stream(fid, n)
    for e in ("w1", "w2", "q"):
        T.add_energy_stream(fid, e)
    T.add_unit(fid, "compressor", "K-1", {"outlet_P_Pa": 3e5, "adiabatic_eff_pct": 75.0})
    T.connect(fid, "K-1", feeds=["air"], products=["k1_out"], energy_feeds=["w1"])
    T.add_unit(fid, "cooler", "IC", {"outlet_T_K": 300.0})
    T.connect(fid, "IC", feeds=["k1_out"], products=["ic_out"], energy_products=["q"])
    T.add_unit(fid, "compressor", "K-2", {"outlet_P_Pa": 9e5, "adiabatic_eff_pct": 75.0})
    T.connect(fid, "K-2", feeds=["ic_out"], products=["k2_out"], energy_feeds=["w2"])
    return fid


def case_b():
    obj = {"type": "sum", "terms": [{"kind": "unit", "name": "K-1", "key": "duty_or_power_kW"}, {"kind": "unit", "name": "K-2", "key": "duty_or_power_kW"}]}
    out = []
    for label, cons in (("sin límite", []), ("T descarga K-1 ≤ 440 K (HAZOP)", [{"kind": "stream", "name": "k1_out", "key": "T_K", "max": 440.0}]),
                        ("ambas descargas ≤ 440 K", [{"kind": "stream", "name": "k1_out", "key": "T_K", "max": 440.0},
                                                     {"kind": "stream", "name": "k2_out", "key": "T_K", "max": 440.0}])):
        fid = _compression()
        r = optimize_design(fid, "unit", "K-1", "outlet_P_Pa", 1.5e5, 6e5, obj, constraints=cons, n_scan=10, refine_iterations=14)
        row = {"case": f"B compresión 2 etapas, {label}", "feasible": r["feasible"]}
        if r["feasible"]:
            b = r["best"]
            row.update(P_mid_bar=b["x"] / 1e5, total_power_kW=b["objective"], T_k1_out_K=T.read_stream(fid, "k1_out")["T_K"],
                       T_k2_out_K=T.read_stream(fid, "k2_out")["T_K"], active_constraint=r["active_constraint"], n_eval=r["n_evaluations"],
                       theory_P_mid_bar=3.0)
        else:
            row["note"] = r["note"]
        T.close_flowsheet(fid)
        out.append(row)
    return out


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "A"
    res = case_a() if part == "A" else case_b()
    for r in (res if isinstance(res, list) else [res]):
        print(json.dumps(r, ensure_ascii=False, default=float), flush=True)
