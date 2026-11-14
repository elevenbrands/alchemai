"""M15c-2 evidence: the reference numbers of M15/M15b with the CONDITIONAL default of the excess-enthalpy term
(owner's decision of 2026-11-13, option C): excess_enthalpy is not passed, so every case below resolves it by itself.
With VLE-only parameters (ChemSep bank, the Kamihama regression) the term is now OFF; the like-for-like DWSIM mode is
«ExpData». Old numbers (term on, the default up to M15c) are read from the committed evidence of notebooks 25/26.

1. Methanol-water, published column of Simone (TU Wien 2015, Tables 9 and 10: Aspen Plus / gPROMS), ChemSep parameters.
2. Methanol-water optimum (x_D ≥ 0.9168, recovery ≥ 0.902) vs DWSIM «ExpData» with the same parameters.
3. Ethanol-water chain (Kamihama regression, 20 trays, x_D 0.70, 350 K feed) vs DWSIM «ExpData».
4. The M15b start-up cases (bubble-point feed: 15 trays x_D 0.70; 20 trays x_D 0.75, 0.80, 0.82) and ChemSep x_D 0.70.
5. A joint VLE + h^E fit that failed its verdict (363.15 K isotherm) → off by default, same numbers as its explicit off.
Run: PYTHONPATH=. uv run python -u notebooks/28_conditional_excess_default.py   (≈ 15 min) → .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemicals import MW  # noqa: E402
from packages.eo import column_optimize as C  # noqa: E402
from packages.eo import dwsim_nrtl as DN  # noqa: E402
from packages.eo import nrtl_properties as NP  # noqa: E402
from packages.vle.tool import vle_binary_regression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_suffix(".json")
P = 101325.0
EW, DW_EW = ["ethanol", "water"], {"ethanol": "Ethanol", "water": "Water"}
MWN, DW_MW = ["methanol", "water"], {"methanol": "Methanol", "water": "Water"}
Z = {"ethanol": 0.1, "water": 0.9}
res: dict = {"policy": NP.HE_POLICY}
old26 = json.loads((ROOT / "notebooks/26_eo_nrtl_ethanol_water.json").read_text())


def dev(a, b):
    return (a / b - 1) * 100


def save():
    OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))


def specs(xd, xbw=0.995):
    return [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": xd},
            {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": xbw}]


def run_eo(comps, F, T, z, n, f, sp, pressure=P, bp=None):
    """excess_enthalpy NOT passed: the conditional default decides."""
    t = time.time()
    r = C.eo_column_optimize(comps, F, T, pressure, z, n, f, pressure, sp, activity_model="NRTL", binary_parameters=bp)
    o, ev = r["optimum"], r["evidence"]
    a, b = comps
    he = r["properties"]["excess_enthalpy"]
    return {"RR": o["reflux_ratio"], "BR": o["boilup_ratio"], "x_D": o["distillate"]["mole_fractions"][a],
            "x_B_heavy": o["bottoms"]["mole_fractions"][b], "T_D_K": o["distillate"]["T_K"], "T_B_K": o["bottoms"]["T_K"],
            "D_mol_s": o["distillate"]["flow_mol_s"], "B_mol_s": o["bottoms"]["flow_mol_s"],
            "Q_reb_kW": r["reboiler_duty_MW"] * 1e3, "Q_cond_kW": -r["condenser_duty_MW"] * 1e3,
            "verified": ev["verified"], "reasons": ev["reasons"], "duals": r["duals_MW_per_unit_spec"],
            "dual_rel_diff": {q: v.get("rel_diff") for q, v in ev["dual_checks"].items()},
            "hE_decision": he["decision"], "hE_included": he["included_in_liquid_enthalpy"], "hE_class": he["source"]["class"],
            "hE_share_if_included": he.get("first_order_share_of_reboiler_duty"), "first_warning": r["warnings"][0],
            "hE_assumption_line": r["assumptions"][1], "seconds": time.time() - t}


def run_dw(e, comps_dw, names_map, F, T, z, stages, feed_stage, params, pressure=P):
    a, b = list(names_map)
    dw = DN.column(comps_dw, T, pressure, {names_map[a]: F * z[a], names_map[b]: F * z[b]}, stages, feed_stage, e["RR"],
                   pressure, bottoms_mol_s=e["B_mol_s"], params=params, dwsim_names=names_map, poynting=False,
                   enthalpy_mode="ExpData")
    if not dw["ok"]:
        return {"ok": False, "errors": dw["solver_errors"]}
    v = {"x_D": dw["distillate"]["mole_fractions"][names_map[a]], "x_B_heavy": dw["bottoms"]["mole_fractions"][names_map[b]],
         "T_D_K": dw["distillate"]["T_K"], "T_B_K": dw["bottoms"]["T_K"], "Q_reb_kW": dw["reboiler_duty_kW"],
         "Q_cond_kW": dw["condenser_duty_kW"]}
    return {"ok": True, "mode": "ExpData", "values": v, "dev_eo_vs_dwsim_pct": {q: dev(e[q], v[q]) for q in v}}


def brief(e):
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in e.items()
            if k in ("RR", "Q_reb_kW", "Q_cond_kW", "verified", "hE_included", "hE_class", "seconds")}


# ------------------------------------------------------------------ 1. Simone 2015, published methanol-water column
nm, nw = 500 / MW("67-56-1") / 3.6, 9500 / MW("7732-18-5") / 3.6
Fm = nm + nw
zm = {"methanol": nm / Fm, "water": nw / Fm}
feed_m = {"flow_mol_s": Fm, "T_K": 313.0, "P_Pa": 101300.0, "z": zm}
PUB = {"aspen": {"T_D_K": 339.0, "T_B_K": 373.0, "Q_reb_kW": 1503.0, "Q_cond_kW": 819.0, "methanol_top_kg_h": 451.0},
       "gproms": {"T_D_K": 339.0, "T_B_K": 373.0, "Q_reb_kW": 1522.0, "Q_cond_kW": 810.0, "methanol_top_kg_h": 450.0}}
res["1_simone_2015"] = {"published": PUB, "source": "Simone, TU Wien 2015, Tablas 9 y 10 (Aspen Plus RadFrac / gPROMS); "
                        "sus parámetros NRTL no están impresos: se usan los de ChemSep"}
for label, flag in (("conditional_default", None), ("explicit_on_as_M15", True)):
    t = time.time()
    st = C.simulate_column_nrtl(MWN, feed_m, 10, 6, 101300.0, 4.36, distillate_mass_flow_kg_s=474 / 3600,
                                excess_enthalpy=flag)
    d, b = st["distillate"], st["bottoms"]
    v = {"T_D_K": d["T_K"], "T_B_K": b["T_K"], "Q_reb_kW": st["reboiler_duty_W"] / 1e3, "Q_cond_kW": -st["condenser_duty_W"] / 1e3,
         "methanol_top_kg_h": d["flow_mol_s"] * d["mole_fractions"]["methanol"] * 32.04186 * 3.6}
    res["1_simone_2015"][label] = {"values": v, "decision": st["excess_enthalpy_decision"], "seconds": time.time() - t,
                                   "dev_vs_aspen_pct": {q: dev(v[q], PUB["aspen"][q]) for q in ("Q_reb_kW", "Q_cond_kW", "methanol_top_kg_h")},
                                   "dev_vs_gproms_pct": {q: dev(v[q], PUB["gproms"][q]) for q in ("Q_reb_kW", "Q_cond_kW", "methanol_top_kg_h")},
                                   "dT_vs_published_K": {q: v[q] - PUB["aspen"][q] for q in ("T_D_K", "T_B_K")}}
    print("1", label, {k: round(x, 3) for k, x in v.items()}, st["excess_enthalpy_decision"]["included"],
          json.dumps(res["1_simone_2015"][label]["dev_vs_aspen_pct"], default=float), flush=True)
save()

# ------------------------------------------------------------------ 2. methanol-water optimum vs DWSIM «ExpData»
spm = [{"stream": "distillate", "component": "methanol", "kind": "mole_fraction", "min": 0.9168},
       {"stream": "distillate", "component": "methanol", "kind": "recovery", "min": 0.902}]
pm = NP.resolve_parameters(MWN)
e = run_eo(MWN, Fm, 313.0, zm, 10, 6, spm, pressure=101300.0)
old_on = old26["2_methanol_water"]["hE_on_vs_LiqCp_Excess"]
res["2_methanol_water_optimum"] = {
    "new_default": e, "dwsim_ExpData": run_dw(e, ["Methanol", "Water"], DW_MW, Fm, 313.0, zm, 12, 7, pm, pressure=101300.0),
    "old_M15_term_on": {**{q: old_on["eo"][q] for q in ("RR", "BR", "Q_reb_kW", "Q_cond_kW")}, "duals": old_on["eo"]["duals"],
                        "dev_vs_dwsim_LiqCp_Excess_pct": old_on["dwsim"]["dev_eo_vs_dwsim_pct"]},
    "new_vs_old_pct": {q: dev(e[q], old_on["eo"][q]) for q in ("RR", "Q_reb_kW", "Q_cond_kW")}}
print("2", brief(e), json.dumps(res["2_methanol_water_optimum"]["dwsim_ExpData"].get("dev_eo_vs_dwsim_pct"), default=float), flush=True)
save()

# ------------------------------------------------------------------ 3. ethanol-water chain vs DWSIM «ExpData»
data = json.loads((ROOT / "tests/data/vle_ethanol_water_vlizard.json").read_text())
k = next(k for k in data["sets"] if "101" in k)
VLE = [{"label": k, "points": data["sets"][k], "citation": data["citation"]}]
reg = vle_binary_regression(data["components"], VLE, models=["NRTL"], compare_unifac=False)
p = NP.resolve_parameters(EW, reg)
T_BUB = C._feed_two_phase_T({"z": Z}, P, EW, p, 0.0)
e = run_eo(EW, 100.0, 350.0, Z, 20, 14, specs(0.70), bp=reg)
old_on = old26["2_chain_case"]["eo_hE_on"]
res["3_ethanol_water_chain"] = {
    "new_default": e, "dwsim_ExpData": run_dw(e, ["Ethanol", "Water"], DW_EW, 100.0, 350.0, Z, 22, 15, p),
    "old_M15b_term_on": {q: old_on[q] for q in ("RR", "BR", "Q_reb_kW", "Q_cond_kW")},
    "new_vs_old_pct": {q: dev(e[q], old_on[q]) for q in ("RR", "Q_reb_kW", "Q_cond_kW")}}
print("3", brief(e), json.dumps(res["3_ethanol_water_chain"]["dwsim_ExpData"].get("dev_eo_vs_dwsim_pct"), default=float), flush=True)
print("3 warning:", e["first_warning"], flush=True)
save()

# ------------------------------------------------------------------ 4. M15b start-up cases
cases = {"B_xD0.70_15trays_feed10_bubble": (0.70, 15, 10, T_BUB, reg), "C_xD0.75_20trays_feed14_bubble": (0.75, 20, 14, T_BUB, reg),
         "D_xD0.80_20trays_feed14_bubble": (0.80, 20, 14, T_BUB, reg), "E_xD0.82_20trays_feed14_bubble": (0.82, 20, 14, T_BUB, reg),
         "G_ChemSep_xD0.70_20trays_350K": (0.70, 20, 14, 350.0, None)}
res["4_startup"] = {"feed_T_bubble_K": T_BUB, "cases": {}}
for name, (xd, n, f, T, bp) in cases.items():
    c = {}
    try:
        c["new_default"] = run_eo(EW, 100.0, T, Z, n, f, specs(xd), bp=bp)
    except Exception as ex_:  # noqa: BLE001 - a failed start is a result to report
        c["new_default"] = {"error": f"{type(ex_).__name__}: {ex_}"[:600]}
    old = old26["3_startup"]["cases"][name]["hE_on"]
    c["old_M15b_term_on"] = {q: old[q] for q in ("RR", "Q_reb_kW", "Q_cond_kW")}
    if "error" not in c["new_default"]:
        c["new_vs_old_pct"] = {q: dev(c["new_default"][q], old[q]) for q in ("RR", "Q_reb_kW", "Q_cond_kW")}
        c["dwsim_ExpData"] = run_dw(c["new_default"], ["Ethanol", "Water"], DW_EW, 100.0, T, Z, n + 2, f + 1,
                                    NP.resolve_parameters(EW, bp))
    res["4_startup"]["cases"][name] = c
    print("4", name, brief(c["new_default"]) if "error" not in c["new_default"] else c["new_default"],
          json.dumps(c.get("dwsim_ExpData", {}).get("dev_eo_vs_dwsim_pct"), default=float), flush=True)
    save()

# ------------------------------------------------------------------ 5. a joint fit that failed its verdict → off
he = json.loads((ROOT / "tests/data/he_ethanol_water_thermoml.json").read_text())
s363 = [{"label": lab, "T_K": s["T_K"], "citation": he["sources"][s["source"]]["citation"],
         "points": [{"x1": q["x1"], "hE_J_mol": q["hE_J_mol"]} for q in s["points"]]} for lab, s in he["sets"].items() if s["T_K"] == 363.15]
j363 = vle_binary_regression(data["components"], VLE, models=["NRTL"], temperature_dependence="a+b/T", compare_unifac=False,
                             excess_enthalpy_sets=s363)
e = run_eo(EW, 100.0, 350.0, Z, 20, 14, specs(0.70), bp=j363)
ref27 = json.loads((ROOT / "notebooks/27_eo_nrtl_excess_enthalpy.json").read_text())["4_columns"]["J_363"]["hE_off"]
res["5_joint_fit_that_failed_the_verdict"] = {
    "new_default": e, "verdict": j363["modelos"]["NRTL"]["veredicto_conjunto"],
    "explicit_off_of_notebook_27": {q: ref27[q] for q in ("RR", "Q_reb_kW", "Q_cond_kW")},
    "new_vs_explicit_off_pct": {q: dev(e[q], ref27[q]) for q in ("RR", "Q_reb_kW", "Q_cond_kW")}}
print("5", brief(e), json.dumps(res["5_joint_fit_that_failed_the_verdict"]["new_vs_explicit_off_pct"], default=float), flush=True)
print("5 warning:", e["first_warning"], flush=True)
save()
print("done", flush=True)
