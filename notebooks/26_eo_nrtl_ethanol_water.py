"""M15b evidence: the two ethanol-water items M15 left open on the NRTL equation-oriented column.

1. Enthalpy audit EO vs DWSIM at identical T, P, x (pure liquids and vapours, feed, distillate, bottoms): the pure
   terms agree; DWSIM 10.2.12's «excess enthalpy» is −R·T²·Σ x·∂γ/∂T, not Gibbs-Helmholtz's −R·T²·Σ x·∂ln γ/∂T.
2. The chain case of M15 (Kamihama 2012 regression → 20 trays, x_D 0.70): Q_reb +6.5 % vs DWSIM «LiqCp_Excess» closed
   term by term from the point enthalpies; like-for-like (excess_enthalpy=False vs DWSIM «ExpData») for ethanol-water
   and for the methanol-water optimum of M15.
3. Start-up: McCabe-Thiele feasibility of every case M15 could not close, the cases through the tool (spec
   continuation on the square system + non-ideality homotopy), each re-built in DWSIM, and the refusals.
Run: PYTHONPATH=. uv run python -u notebooks/26_eo_nrtl_ethanol_water.py   (≈ 15 min) → notebooks/26_eo_nrtl_ethanol_water.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.eo import column_optimize as C  # noqa: E402
from packages.eo import dwsim_nrtl as DN  # noqa: E402
from packages.eo import nrtl_properties as NP  # noqa: E402
from packages.eo import shortcut as SC  # noqa: E402
from packages.vle.tool import vle_binary_regression  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
P = 101325.0
EW, DW_EW = ["ethanol", "water"], {"ethanol": "Ethanol", "water": "Water"}
res: dict = {}


def dev(a, b):
    return (a / b - 1) * 100


def save():
    OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))


data = json.loads((Path(__file__).parents[1] / "tests/data/vle_ethanol_water_vlizard.json").read_text())
k = next(k for k in data["sets"] if "101" in k)
reg = vle_binary_regression(data["components"], [{"label": k, "points": data["sets"][k], "citation": data["citation"]}],
                            models=["NRTL"], compare_unifac=False)
p = NP.resolve_parameters(EW, reg)
Z = {"ethanol": 0.1, "water": 0.9}
T_BUB = C._feed_two_phase_T({"z": Z}, P, EW, p, 0.0)


def specs(xd, xbw=0.995):
    return [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": xd},
            {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": xbw}]


def eo_point_enthalpies(states: dict, T_feed: float) -> dict:
    """h (J/mol) of the IDAES package for each (T, x_ethanol, phase), without solving anything."""
    from pyomo.environ import ConcreteModel, value, units as pyunits
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    cfg, _ = NP.build_configuration(EW, P, p, extra_T=[T_feed])
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.p = GenericParameterBlock(**cfg)
    m.fs.sb = m.fs.p.build_state_block([0], defined_state=True)
    s = m.fs.sb[0]
    out = {}
    for n, (T, x, ph) in states.items():
        ph_ = "Liq" if ph == "L" else "Vap"
        s.temperature.set_value(T)
        s.mole_frac_phase_comp[ph_, "ethanol"].set_value(max(x, 1e-20), skip_validation=True)
        s.mole_frac_phase_comp[ph_, "water"].set_value(max(1 - x, 1e-20), skip_validation=True)
        out[n] = value(pyunits.convert(s.enth_mol_phase[ph_], pyunits.J / pyunits.mol))
    return out


def run_eo(comps, F, T, z, n, f, sp, pressure=P, bp=None, hE=True):
    t = time.time()
    r = C.eo_column_optimize(comps, F, T, pressure, z, n, f, pressure, sp, activity_model="NRTL", binary_parameters=bp,
                             excess_enthalpy=hE)
    o, ev = r["optimum"], r["evidence"]
    a, b = comps
    return {"RR": o["reflux_ratio"], "BR": o["boilup_ratio"], "x_D": o["distillate"]["mole_fractions"][a],
            "x_B_heavy": o["bottoms"]["mole_fractions"][b], "T_D_K": o["distillate"]["T_K"], "T_B_K": o["bottoms"]["T_K"],
            "D_mol_s": o["distillate"]["flow_mol_s"], "B_mol_s": o["bottoms"]["flow_mol_s"],
            "Q_reb_kW": r["reboiler_duty_MW"] * 1e3, "Q_cond_kW": -r["condenser_duty_MW"] * 1e3,
            "verified": ev["verified"], "reasons": ev["reasons"], "smooth_eps_K": ev["smooth_eps_K"],
            "phantom_vapour": ev["phantom_vapour_fraction"], "balance": ev["component_balance_rel"],
            "duals": r["duals_MW_per_unit_spec"], "duals_source": ev["duals_source"],
            "dual_rel_diff": {q: v.get("rel_diff") for q, v in ev["dual_checks"].items()},
            "start": {q: r["initial_point"][q] for q in ("reflux_ratio", "boilup_ratio", "mode")},
            "starts_tried_before": len(r["initial_point"]["tried"]),
            "spec_continuation": r["initial_point"].get("spec_continuation"),
            "excess_enthalpy": r["properties"]["excess_enthalpy"], "shortcut": {q: r["shortcut"][q] for q in (
                "N_min_total_reflux", "R_min_CMO", "R_needed_CMO", "dT_dew_bubble_distillate_K") if q in r["shortcut"]},
            "seconds": time.time() - t}


def run_dw(e, comps_dw, names_map, F, T, z, stages, feed_stage, mode, params, pressure=P):
    a, b = list(names_map)
    dw = DN.column(comps_dw, T, pressure, {names_map[a]: F * z[a], names_map[b]: F * z[b]}, stages, feed_stage, e["RR"],
                   pressure, bottoms_mol_s=e["B_mol_s"], params=params, dwsim_names=names_map, poynting=False,
                   enthalpy_mode=mode)
    if not dw["ok"]:
        return {"ok": False, "errors": dw["solver_errors"]}
    v = {"x_D": dw["distillate"]["mole_fractions"][names_map[a]], "x_B_heavy": dw["bottoms"]["mole_fractions"][names_map[b]],
         "T_D_K": dw["distillate"]["T_K"], "T_B_K": dw["bottoms"]["T_K"], "Q_reb_kW": dw["reboiler_duty_kW"],
         "Q_cond_kW": dw["condenser_duty_kW"]}
    return {"ok": True, "mode": mode, "values": v, "dev_eo_vs_dwsim_pct": {q: dev(e[q], v[q]) for q in v}}


# ------------------------------------------------------------------ 2a. the M15 chain case, h^E on
A = run_eo(EW, 100.0, 350.0, Z, 20, 14, specs(0.70), bp=reg)
res["2_chain_case"] = {"eo_hE_on": A, "dwsim": {}}
for mode in ("LiqCp_Excess", "ExpData"):
    res["2_chain_case"]["dwsim"][mode] = run_dw(A, ["Ethanol", "Water"], DW_EW, 100.0, 350.0, Z, 22, 15, mode, p)
    print("2a", mode, json.dumps(res["2_chain_case"]["dwsim"][mode].get("dev_eo_vs_dwsim_pct")), flush=True)
save()

# ------------------------------------------------------------------ 1. enthalpy audit at identical T, P, x
dwL = res["2_chain_case"]["dwsim"]["LiqCp_Excess"]["values"]
st = {"feed": (350.0, 0.10, "L"), "dist_eo": (A["T_D_K"], A["x_D"], "L"), "bot_eo": (A["T_B_K"], 1 - A["x_B_heavy"], "L"),
      "dist_dw": (dwL["T_D_K"], dwL["x_D"], "L"), "bot_dw": (dwL["T_B_K"], 1 - dwL["x_B_heavy"], "L"),
      "x30_355": (355.0, 0.30, "L"), "x50_352": (352.0, 0.50, "L"),
      "w_298": (298.15, 0.0, "L"), "w_350": (350.0, 0.0, "L"), "w_371.68": (371.68, 0.0, "L"),
      "e_298": (298.15, 1.0, "L"), "e_350": (350.0, 1.0, "L"),
      "w_350_v": (350.0, 0.0, "V"), "w_371.68_v": (371.68, 0.0, "V"), "e_350_v": (350.0, 1.0, "V")}
h_eo = eo_point_enthalpies(st, 350.0)
liq = {n: (T, {"Ethanol": x, "Water": 1 - x}) for n, (T, x, ph) in st.items() if ph == "L"}
vap = {n: (T, {"Ethanol": x, "Water": 1 - x}) for n, (T, x, ph) in st.items() if ph == "V"}
h_dw = {mode: DN.liquid_enthalpies(["Ethanol", "Water"], liq, p, DW_EW, mode) for mode in ("LiqCp_Excess", "ExpData")}
h_dw_v = DN.liquid_enthalpies(["Ethanol", "Water"], vap, p, DW_EW, "LiqCp_Excess", pressure_Pa=2000.0)
mix = {}
for n, (T, x, ph) in st.items():
    if ph == "L" and 0 < x < 1:
        xs = [x, 1 - x]
        mix[n] = {"T_K": T, "x_ethanol": x, "hE_gibbs_helmholtz_J_mol": NP.excess_enthalpy_numeric(p, T, xs),
                  "hE_thermo_NRTL_HE_J_mol": float(NP.thermo_nrtl(p, T, xs).HE()),
                  "dwsim_excess_J_mol": h_dw["LiqCp_Excess"][n]["h_J_mol"] - h_dw["ExpData"][n]["h_J_mol"],
                  "minus_RT2_sum_x_dgamma_dT_J_mol": DN.dwsim_excess_term(p, T, xs), "gammas": NP.gammas(p, T, xs)}
        mix[n]["dwsim_over_dgamma_formula"] = mix[n]["dwsim_excess_J_mol"] / mix[n]["minus_RT2_sum_x_dgamma_dT_J_mol"]
        mix[n]["dwsim_over_gibbs_helmholtz"] = mix[n]["dwsim_excess_J_mol"] / mix[n]["hE_gibbs_helmholtz_J_mol"]
pure = {}
for a_, b_ in (("w_350", "w_298"), ("w_371.68", "w_350"), ("e_350", "e_298")):
    e_, d_ = h_eo[a_] - h_eo[b_], h_dw["ExpData"][a_]["h_J_mol"] - h_dw["ExpData"][b_]["h_J_mol"]
    pure[f"sensible_liq {a_} − {b_}"] = {"eo_J_mol": e_, "dwsim_J_mol": d_, "dev_pct": dev(e_, d_)}
for v_, l_ in (("w_350_v", "w_350"), ("w_371.68_v", "w_371.68"), ("e_350_v", "e_350")):
    e_, d_ = h_eo[v_] - h_eo[l_], h_dw_v[v_]["h_J_mol"] - h_dw["ExpData"][l_]["h_J_mol"]
    pure[f"latent {v_} − {l_}"] = {"eo_J_mol": e_, "dwsim_J_mol": d_, "dev_pct": dev(e_, d_)}
res["1_enthalpy_audit"] = {"mixtures": mix, "pure": pure,
                           "dwsim_vapour_fraction_check": {n: v["vapor_fraction"] for n, v in h_dw["ExpData"].items()}}
# closure of Q_reb − Q_cond = D·h_D + B·h_B − F·h_F between the two engines
F, D, B = 100.0, A["D_mol_s"], A["B_mol_s"]
hL = lambda n: h_dw["LiqCp_Excess"][n]["h_J_mol"]  # noqa: E731
hX = lambda n: h_dw["ExpData"][n]["h_J_mol"]  # noqa: E731
dwc = res["2_chain_case"]["dwsim"]["LiqCp_Excess"]["values"]
obs = (A["Q_reb_kW"] - A["Q_cond_kW"]) - (dwc["Q_reb_kW"] - dwc["Q_cond_kW"])
ex = {n: mix[n]["hE_gibbs_helmholtz_J_mol"] - mix[n]["dwsim_excess_J_mol"] for n in ("feed", "dist_eo", "bot_eo")}
t_ex = (D * ex["dist_eo"] + B * ex["bot_eo"] - F * ex["feed"]) / 1e3
ideal = lambda n: h_eo[n] - mix[n]["hE_gibbs_helmholtz_J_mol"]  # noqa: E731
ref = ideal("feed") - hX("feed")
t_pure = (D * (ideal("dist_eo") - hX("dist_eo") - ref) + B * (ideal("bot_eo") - hX("bot_eo") - ref)) / 1e3
t_state = (D * (hL("dist_eo") - hL("dist_dw")) + B * (hL("bot_eo") - hL("bot_dw"))) / 1e3
res["2_chain_case"]["closure_kW"] = {
    "eo_Qreb_minus_Qcond": A["Q_reb_kW"] - A["Q_cond_kW"],
    "eo_from_point_enthalpies": (D * h_eo["dist_eo"] + B * h_eo["bot_eo"] - F * h_eo["feed"]) / 1e3,
    "dwsim_Qreb_minus_Qcond": dwc["Q_reb_kW"] - dwc["Q_cond_kW"],
    "dwsim_from_point_enthalpies": (D * hL("dist_dw") + B * hL("bot_dw") - F * hL("feed")) / 1e3,
    "observed_difference": obs,
    "term_excess_formula": {"feed": -F * ex["feed"] / 1e3, "distillate": D * ex["dist_eo"] / 1e3,
                            "bottoms": B * ex["bot_eo"] / 1e3, "total": t_ex},
    "term_pure_component_enthalpies": t_pure, "term_different_product_states": t_state,
    "sum_of_terms": t_ex + t_pure + t_state, "unexplained": obs - (t_ex + t_pure + t_state),
    "Q_cond_difference": A["Q_cond_kW"] - dwc["Q_cond_kW"],
    "Q_cond_term_excess_in_reflux_and_distillate": -(A["RR"] + 1) * D * ex["dist_eo"] / 1e3,
    "Q_reb_difference": A["Q_reb_kW"] - dwc["Q_reb_kW"]}
print("1/2 closure", json.dumps(res["2_chain_case"]["closure_kW"], default=float), flush=True)
save()

# ------------------------------------------------------------------ 2b. like-for-like: no h^E vs DWSIM ExpData
A0 = run_eo(EW, 100.0, 350.0, Z, 20, 14, specs(0.70), bp=reg, hE=False)
res["2_chain_case"]["eo_hE_off"] = A0
res["2_chain_case"]["like_for_like_ExpData"] = run_dw(A0, ["Ethanol", "Water"], DW_EW, 100.0, 350.0, Z, 22, 15, "ExpData", p)
res["2_chain_case"]["hE_effect_on_Q_reb_pct"] = dev(A["Q_reb_kW"], A0["Q_reb_kW"])
print("2b", json.dumps(res["2_chain_case"]["like_for_like_ExpData"].get("dev_eo_vs_dwsim_pct")), flush=True)
save()

# methanol-water optimum of M15 (ChemSep parameters), both ways
from chemicals import MW  # noqa: E402
nm, nw = 500 / MW("67-56-1") / 3.6, 9500 / MW("7732-18-5") / 3.6
Fm = nm + nw
zm = {"methanol": nm / Fm, "water": nw / Fm}
spm = [{"stream": "distillate", "component": "methanol", "kind": "mole_fraction", "min": 0.9168},
       {"stream": "distillate", "component": "methanol", "kind": "recovery", "min": 0.902}]
pm = NP.resolve_parameters(["methanol", "water"])
DW_MW = {"methanol": "Methanol", "water": "Water"}
res["2_methanol_water"] = {}
for label, hE, mode in (("hE_on_vs_LiqCp_Excess", True, "LiqCp_Excess"), ("hE_off_vs_ExpData", False, "ExpData")):
    e = run_eo(["methanol", "water"], Fm, 313.0, zm, 10, 6, spm, pressure=101300.0, hE=hE)
    d = run_dw(e, ["Methanol", "Water"], DW_MW, Fm, 313.0, zm, 12, 7, mode, pm, pressure=101300.0)
    res["2_methanol_water"][label] = {"eo": e, "dwsim": d}
    print("2 MW", label, e["RR"], e["Q_reb_kW"], e["verified"], json.dumps(d.get("dev_eo_vs_dwsim_pct")), flush=True)
    save()

# ------------------------------------------------------------------ 3. start-up: feasibility, cases, DWSIM, refusals
az = NP.binary_azeotropes(p, P)
cases = {"B_xD0.70_15trays_feed10_bubble": (0.70, 15, 10, T_BUB), "C_xD0.75_20trays_feed14_bubble": (0.75, 20, 14, T_BUB),
         "D_xD0.80_20trays_feed14_bubble": (0.80, 20, 14, T_BUB), "E_xD0.82_20trays_feed14_bubble": (0.82, 20, 14, T_BUB)}
res["3_startup"] = {"azeotrope": az, "feed_T_bubble_K": T_BUB, "cases": {}}
for name, (xd, n, f, T) in cases.items():
    c = {}
    for label, hE in (("hE_on", True), ("hE_off", False)):
        try:
            c[label] = run_eo(EW, 100.0, T, Z, n, f, specs(xd), bp=reg, hE=hE)
        except Exception as ex_:  # noqa: BLE001
            c[label] = {"error": f"{type(ex_).__name__}: {ex_}"}
    if "error" not in c["hE_off"]:
        c["dwsim_ExpData_vs_hE_off"] = run_dw(c["hE_off"], ["Ethanol", "Water"], DW_EW, 100.0, T, Z, n + 2, f + 1, "ExpData", p)
    res["3_startup"]["cases"][name] = c
    print("3", name, {q: (c[q].get("verified"), c[q].get("RR"), c[q].get("Q_reb_kW"), c[q].get("seconds")) for q in ("hE_on", "hE_off")},
          json.dumps(c.get("dwsim_ExpData_vs_hE_off", {}).get("dev_eo_vs_dwsim_pct")), flush=True)
    save()
pc = NP.resolve_parameters(EW)
try:
    res["3_startup"]["cases"]["G_ChemSep_xD0.70_20trays_350K"] = {"hE_on": run_eo(EW, 100.0, 350.0, Z, 20, 14, specs(0.70))}
except Exception as ex_:  # noqa: BLE001
    res["3_startup"]["cases"]["G_ChemSep_xD0.70_20trays_350K"] = {"hE_on": {"error": str(ex_)}}
refusals = {}
for name, kw in {"near_azeotrope_xD0.85": dict(xd=0.85, n=20, f=14), "too_few_trays_xD0.80_4trays": dict(xd=0.80, n=4, f=3),
                 "reflux_bound_xD0.80_rrmax1.0": dict(xd=0.80, n=20, f=14, rr=(0.1, 1.0)),
                 "across_azeotrope_xD0.95": dict(xd=0.95, n=20, f=14)}.items():
    t = time.time()
    try:
        C.eo_column_optimize(EW, 100.0, T_BUB, P, Z, kw["n"], kw["f"], P, specs(kw["xd"]), activity_model="NRTL",
                             binary_parameters=reg, reflux_ratio_bounds=kw.get("rr", (0.1, 20.0)))
        refusals[name] = {"refused": False}
    except ValueError as ex_:
        refusals[name] = {"refused": True, "message": str(ex_), "seconds": time.time() - t}
res["3_startup"]["refusals"] = refusals
save()
print("wrote", OUT)
