"""Notebook 20 — integer design variables (bloque 26): number of stages × reflux ratio of the ethanol-water column
(NRTL, reference case `columna_rigurosa`), minimum total annual cost with x_D(EtOH) ≥ 0.80.
Theory check: the optimal reflux should land at R/R_min ≈ 1.1-1.5 (Seader, Henley & Roper, heuristic for the
economic optimum) and the TAC-vs-N curve should be flat near the optimum (Douglas). Prices are illustrative (MOCKS.md).
Run: PYTHONPATH=. uv run python notebooks/20_variables_enteras.py  -> notebooks/20_variables_enteras.jsonl"""
from __future__ import annotations

import json
import time
from pathlib import Path

from packages.dwsim_client import tools as T
from packages.dwsim_client.tools_optimize import optimize_design_multi

PRICES = {"heating_usd_GJ": 8.0, "cooling_usd_GJ": 0.5, "electricity_usd_kWh": 0.10}   # illustrative (MOCKS: PRECIOS-ILUSTRATIVOS)
COST = {"type": "total_annual_cost", "prices": PRICES, "hours_per_year": 8000.0, "interest": 0.10, "years": 10,
        "vessel_sizes": {"T-101": {"diameter_m": 1.2, "tray_spacing_m": 0.6}}}
OUT = Path("notebooks/20_variables_enteras.jsonl")


def column(stages: int = 20, reflux: float = 2.0) -> str:
    fid = T.new_flowsheet("etoh_col", ["Water", "Ethanol"], "NRTL")["flowsheet_id"]
    T.add_material_stream(fid, "feed", temperature_K=300.0, pressure_Pa=101325.0, molar_flow_mol_s=100.0, mole_fractions={"Water": 0.5, "Ethanol": 0.5})
    T.add_unit(fid, "distillation_column", "T-101", {"stages": stages, "feed_stage": stages // 2, "reflux_ratio": reflux, "top_P_Pa": 101325.0,
                                                     "dP_Pa": 0.0, "bottoms_molar_flow_mol_s": 75.0})
    for n in ("dist", "bot"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "qc"); T.add_energy_stream(fid, "qr")
    T.connect(fid, "T-101", feeds=["feed"], products=["dist", "bot"], energy_feeds=["qr"], energy_products=["qc"])
    return fid


def r_min() -> dict:
    """FUG shortcut for the same split (x_D = 0.80 → x_B = 0.40 by balance) to get R_min and N_min."""
    fid = T.new_flowsheet("etoh_fug", ["Water", "Ethanol"], "NRTL")["flowsheet_id"]
    T.add_material_stream(fid, "feed", temperature_K=300.0, pressure_Pa=101325.0, molar_flow_mol_s=100.0, mole_fractions={"Water": 0.5, "Ethanol": 0.5})
    T.add_unit(fid, "shortcut_column", "SC", {"light_key": "Ethanol", "heavy_key": "Water", "lk_mole_frac_in_bottoms": 0.40,
                                            "hk_mole_frac_in_distillate": 0.20, "condenser_P_Pa": 101325.0, "reboiler_P_Pa": 101325.0, "reflux_ratio": 2.0})
    for n in ("d", "b"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "qc"); T.add_energy_stream(fid, "qr")
    T.connect(fid, "SC", feeds=["feed"], products=["d", "b"], energy_feeds=["qr"], energy_products=["qc"])
    T.solve(fid)
    u = T.read_unit(fid, "SC")
    T.close_flowsheet(fid)
    return {"R_min": u["R_min"], "N_min": u["N_min"], "N_theoretical_at_R2": u["N_theoretical"]}


def r_min_rigorous(fid: str) -> float:
    """Rigorous R_min: with many stages (80) the reflux that just gives x_D = 0.80, by bisection on the solved column.
    FUG's R_min is not trustworthy for ethanol-water (relative volatility collapses near the azeotrope)."""
    from packages.dwsim_client.tools_sweep import set_unit_param
    set_unit_param(fid, "T-101", "stages", 80)
    lo, hi = 0.3, 3.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        set_unit_param(fid, "T-101", "reflux_ratio", mid)
        ok = T.solve(fid)["ok"]
        xd = T.read_stream(fid, "dist")["mole_fractions"]["Ethanol"] if ok else 0.0
        lo, hi = (lo, mid) if xd >= 0.80 else (mid, hi)
    set_unit_param(fid, "T-101", "stages", 20); set_unit_param(fid, "T-101", "reflux_ratio", 2.0)
    return hi


def optimum(fid: str, max_evaluations: int = 260) -> dict:
    return optimize_design_multi(fid, [{"kind": "unit", "name": "T-101", "param": "stages", "lower": 8, "upper": 60, "integer": True},
                                       {"kind": "unit", "name": "T-101", "param": "reflux_ratio", "lower": 1.0, "upper": 5.0}],
                                 COST, [{"kind": "stream", "name": "dist", "key": "mole_fractions.Ethanol", "min": 0.80}],
                                 grid_points=4, max_evaluations=max_evaluations)


def main():
    rows = []
    fug = r_min()
    print("FUG:", fug)
    rows.append({"section": "fug", **fug})
    fid = column()
    rmin = r_min_rigorous(fid)
    print("R_min rigorous (80 stages):", round(rmin, 4))
    rows.append({"section": "r_min_rigorous_80_stages", "R_min": rmin})
    t0 = time.time()
    res = optimum(fid)
    res["seconds"] = time.time() - t0
    b = res["best"]
    print(f"best: {b['values']}  TAC {b['objective']:.0f} USD/y  in {res['n_evaluations']} solves, {res['seconds']:.0f} s")
    print("R/R_min (rigorous) =", round(b["values"]["T-101.reflux_ratio"] / rmin, 3), "| R/R_min (FUG) =", round(b["values"]["T-101.reflux_ratio"] / fug["R_min"], 3), "| at bound:", res.get("variables_at_bound"))
    for d in res["integer_curve"]:
        n = list(d["integers"].values())[0]
        tac = f"{d['objective']:.0f}" if d["objective"] is not None else "—"
        r = f"{d['continuous']['T-101.reflux_ratio']:.3f}" if d["continuous"] else "—"
        print(f"  N={n:>3}  TAC={tac:>8}  R={r}  feasible={d['feasible']}")
    rows.append({"section": "optimum", **{k: v for k, v in res.items() if k != "path"}})
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n")
    T.close_flowsheet(fid)


if __name__ == "__main__":
    main()
