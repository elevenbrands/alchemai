"""M15c evidence: how good is the h^E of ethanol-water in the NRTL equation-oriented column, against calorimetry.

Measured h^E: tests/data/he_ethanol_water_thermoml.json (Nagamachi y Francesconi 2006 at 298.15 and 323.15 K; Brandt et
al. 2014 at 333.15, 363.15 and 383.15 K; numbers read by script from the NIST ThermoML files).

1. Model h^E vs measured, per isotherm: (a) the VLE-only regression of Kamihama 2012 (the M15/M15b chain), (b) the
   ChemSep bank, (c) no excess term.
2. The same at the three streams of the M15b reference column (10 mol % ethanol at 350 K, 20 trays, x_D 0.70), with the
   measured value interpolated between isotherms, and its translation into duties: an ENTHALPY-BALANCE ESTIMATE
   (first order, at the solved column's flows), validated against the solved h^E on/off pair.
3. Joint VLE + h^E regression of ONE NRTL set (tau = a + b/T): which data, which weights, what it achieves (weight scan).
4. The joint sets through eo_column_optimize (FULL RE-SOLVES, h^E on and off) and DWSIM «ExpData» like-for-like with
   the same a + b/T parameters, plus DWSIM «ExpData» + measured h^E (estimate).
Run: PYTHONPATH=. uv run python -u notebooks/27_eo_nrtl_excess_enthalpy.py   (≈ 15 min) → notebooks/27_eo_nrtl_excess_enthalpy.json
     add --fast to stop after section 3 (no ipopt, no DWSIM; seconds).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.eo import nrtl_properties as NP  # noqa: E402
from packages.vle.tool import vle_binary_regression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_suffix(".json")
FAST = "--fast" in sys.argv
P = 101325.0
EW, DW_EW = ["ethanol", "water"], {"ethanol": "Ethanol", "water": "Water"}
Z = {"ethanol": 0.1, "water": 0.9}
F, T_FEED, N_TRAYS, FEED_TRAY, X_D, X_BW = 100.0, 350.0, 20, 14, 0.70, 0.995
res: dict = {}


def dev(a, b):
    return (a / b - 1) * 100


def save():
    OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))


# ------------------------------------------------------------------ data
vle = json.loads((ROOT / "tests/data/vle_ethanol_water_vlizard.json").read_text())
KAM = next(k for k in vle["sets"] if "101" in k)
VLE_SETS = [{"label": KAM, "points": vle["sets"][KAM], "citation": vle["citation"]}]
he = json.loads((ROOT / "tests/data/he_ethanol_water_thermoml.json").read_text())
ISO = {s["T_K"]: (np.array([q["x1"] for q in s["points"]]), np.array([q["hE_J_mol"] for q in s["points"]]),
                  np.array([q["U_hE_J_mol"] for q in s["points"]]), lab) for lab, s in he["sets"].items()}
T_ISO = sorted(ISO)


def he_sets(temps):
    return [{"label": lab, "T_K": s["T_K"], "citation": he["sources"][s["source"]]["citation"],
             "points": [{"x1": q["x1"], "hE_J_mol": q["hE_J_mol"]} for q in s["points"]]}
            for lab, s in he["sets"].items() if s["T_K"] in temps]


_PCHIP = {T: PchipInterpolator(np.r_[0.0, ISO[T][0], 1.0], np.r_[0.0, ISO[T][1], 0.0]) for T in T_ISO}


def measured(x: float, T: float) -> dict:
    """Measured h^E interpolated to (x, T): PCHIP in x on each isotherm (through h^E = 0 at x = 0 and 1), then LINEAR
    in T between the two bracketing isotherms. Spread = |quadratic through the 3 nearest isotherms − linear|: the size
    of the interpolation error, not a measurement uncertainty."""
    lo = max(t for t in T_ISO if t <= T)
    hi = min(t for t in T_ISO if t >= T)
    h_lo, h_hi = float(_PCHIP[lo](x)), float(_PCHIP[hi](x))
    lin = h_lo if hi == lo else h_lo + (h_hi - h_lo) * (T - lo) / (hi - lo)
    near = sorted(sorted(T_ISO, key=lambda t: abs(t - T))[:3])
    hs = [float(_PCHIP[t](x)) for t in near]
    quad = float(np.polyval(np.polyfit(near, hs, 2), T))
    return {"hE_J_mol": lin, "interpolation_spread_J_mol": abs(quad - lin), "quadratic_J_mol": quad,
            "isotherms_K": [lo, hi], "hE_at_isotherms_J_mol": [h_lo, h_hi]}


def model_he(params, x, T):
    return NP.excess_enthalpy_numeric(params, T, [x, 1 - x])


def by_isotherm(params) -> list[dict]:
    rows = []
    for T in T_ISO:
        x, h, U, lab = ISO[T]
        m = np.array([0.0] * len(x)) if params is None else np.array([model_he(params, xi, T) for xi in x])
        rows.append({"set": lab, "T_K": T, "n": len(x), "measured_min_max_J_mol": [float(h.min()), float(h.max())],
                     "model_min_max_J_mol": [float(m.min()), float(m.max())], "AAD_J_mol": float(np.mean(np.abs(m - h))),
                     "bias_J_mol": float(np.mean(m - h)),
                     "wrong_sign_points": None if params is None else int(np.sum(np.sign(m) != np.sign(h))),
                     "mean_expanded_uncertainty_of_data_J_mol": float(U.mean())})
    return rows


def overall(rows):
    n = sum(r["n"] for r in rows)
    return sum(r["AAD_J_mol"] * r["n"] for r in rows) / n


# ------------------------------------------------------------------ 1. model vs calorimetry, per isotherm
reg_a = vle_binary_regression(vle["components"], VLE_SETS, models=["NRTL"], compare_unifac=False)
PA = NP.resolve_parameters(EW, reg_a)           # (a) VLE-only regression, tau = b/T (the M15/M15b chain)
PB = NP.resolve_parameters(EW, None)            # (b) ChemSep bank
SETS = {"a_kamihama_vle_only": PA, "b_chemsep": PB, "c_no_excess_term": None}
res["1_by_isotherm"] = {}
for name, prm in SETS.items():
    rows = by_isotherm(prm)
    res["1_by_isotherm"][name] = {"rows": rows, "AAD_all_points_J_mol": overall(rows),
                                  "parameters": None if prm is None else {k: prm["pairs"][0][k] for k in (
                                      "a_ij", "a_ji", "b_ij_K", "b_ji_K", "alpha", "source_kind")},
                                  "hE_basis": None if prm is None else prm["pairs"][0]["hE_basis"]}
    print(name, "AAD all", round(overall(rows), 1), [(r["T_K"], round(r["AAD_J_mol"]), r["wrong_sign_points"]) for r in rows], flush=True)
save()

# ------------------------------------------------------------------ 2. the reference column's streams and duties
ref26 = json.loads((ROOT / "notebooks/26_eo_nrtl_ethanol_water.json").read_text())["2_chain_case"]


def stream_states(e):
    return {"feed": (Z["ethanol"], T_FEED, F), "distillate": (e["x_D"], e["T_D_K"], e["D_mol_s"]),
            "bottoms": (1 - e["x_B_heavy"], e["T_B_K"], e["B_mol_s"])}


def first_order(e, h):
    """Enthalpy-balance estimate of what an excess enthalpy h (J/mol per stream) adds to the duties of a solved column
    at fixed flows and reflux: ΔQ_cond = −(1 + RR)·D·h_D (the condensate leaves with h_D more enthalpy) and
    ΔQ_reb = ΔQ_cond + D·h_D + B·h_B − F·h_F (overall balance). kW."""
    st = stream_states(e)
    D, B = st["distillate"][2], st["bottoms"][2]
    dqc = -(1 + e["RR"]) * D * h["distillate"] / 1e3
    dqr = dqc + (D * h["distillate"] + B * h["bottoms"] - F * h["feed"]) / 1e3
    return {"dQ_reb_kW": dqr, "dQ_cond_kW": dqc}


def streams_table(e, params_by_name):
    st = stream_states(e)
    rows = {}
    for s, (x, T, flow) in st.items():
        rows[s] = {"x_ethanol": x, "T_K": T, "flow_mol_s": flow, "measured": measured(x, T),
                   **{n: (0.0 if p_ is None else model_he(p_, x, T)) for n, p_ in params_by_name.items()}}
    return rows


off, on = ref26["eo_hE_off"], ref26["eo_hE_on"]
tab = streams_table(off, SETS)
h_meas = {s: tab[s]["measured"]["hE_J_mol"] for s in tab}
h_quad = {s: tab[s]["measured"]["quadratic_J_mol"] for s in tab}
est = {n: first_order(off, {s: tab[s][n] for s in tab}) for n in SETS}
est["measured"] = first_order(off, h_meas)
est["measured_quadratic_in_T"] = first_order(off, h_quad)
duties = {n: {"Q_reb_kW": off["Q_reb_kW"] + v["dQ_reb_kW"], "Q_cond_kW": off["Q_cond_kW"] + v["dQ_cond_kW"],
              "dQ_reb_pct_of_no_excess": v["dQ_reb_kW"] / off["Q_reb_kW"] * 100,
              "dQ_cond_pct_of_no_excess": v["dQ_cond_kW"] / off["Q_cond_kW"] * 100} for n, v in est.items()}
res["2_reference_column"] = {
    "column": {"feed_mol_s": F, "z_ethanol": Z["ethanol"], "T_feed_K": T_FEED, "n_trays": N_TRAYS, "feed_tray": FEED_TRAY,
               "x_D": X_D, "x_B_water": X_BW, "parameters": "regresión de Kamihama 2012 (M15b)"},
    "solved_M15b": {"hE_on": {k: on[k] for k in ("Q_reb_kW", "Q_cond_kW", "RR", "D_mol_s", "B_mol_s", "T_D_K", "T_B_K")},
                    "hE_off": {k: off[k] for k in ("Q_reb_kW", "Q_cond_kW", "RR", "D_mol_s", "B_mol_s", "T_D_K", "T_B_K")},
                    "source": "notebooks/26_eo_nrtl_ethanol_water.json (re-solves completos de M15b)"},
    "streams_hE_J_mol": tab,
    "method": "ESTIMACIÓN por balance de entalpía a primer orden sobre la columna resuelta SIN h^E (flujos y reflujo "
              "fijos); no es un re-solve",
    "estimate_check_vs_solved_on_off": {
        "solved_dQ_reb_kW": on["Q_reb_kW"] - off["Q_reb_kW"], "estimated_dQ_reb_kW": est["a_kamihama_vle_only"]["dQ_reb_kW"],
        "solved_dQ_cond_kW": on["Q_cond_kW"] - off["Q_cond_kW"], "estimated_dQ_cond_kW": est["a_kamihama_vle_only"]["dQ_cond_kW"]},
    "duties_estimated": duties,
    "error_of_each_option_vs_measured_hE_pct": {
        n: {"Q_reb": dev(q["Q_reb_kW"], duties["measured"]["Q_reb_kW"]), "Q_cond": dev(q["Q_cond_kW"], duties["measured"]["Q_cond_kW"])}
        for n, q in {"a_solved_hE_on": on, "c_solved_hE_off": off, "b_chemsep_estimate": duties["b_chemsep"]}.items()}}
print("2 streams", {s: {k: (round(v, 1) if isinstance(v, float) else round(v["hE_J_mol"], 1)) for k, v in r.items()
                        if k in (*SETS, "measured")} for s, r in tab.items()}, flush=True)
print("2 duties", json.dumps({k: {q: round(v, 2) for q, v in d.items()} for k, d in duties.items()}), flush=True)
print("2 errors", json.dumps(res["2_reference_column"]["error_of_each_option_vs_measured_hE_pct"], default=float), flush=True)
save()

# ------------------------------------------------------------------ 3. joint regression VLE + h^E, one NRTL set
SCAN = [0, 0.01, 0.1, 1, 10, 100]
FITS = {  # data choice and alpha are declared here, before looking at any column result; weights are 1 and 1
    "J_all_T": {"temps": T_ISO, "alpha": None, "why": "todos los datos de h^E (298-383 K), α = 0.3, pesos 1 y 1"},
    "J_column_T": {"temps": [333.15, 363.15, 383.15], "alpha": None,
                   "why": "solo las isotermas que acotan la columna (333-383 K), α = 0.3, pesos 1 y 1"},
    "J_363": {"temps": [363.15], "alpha": None,
              "why": "solo la isoterma más cercana a la columna (363.15 K), α = 0.3, pesos 1 y 1"},
    "J_all_T_alpha_fit": {"temps": T_ISO, "alpha": "ajustar", "why": "todos los datos, α ajustado (5 parámetros), pesos 1 y 1"},
}
JOINT = {}
res["3_joint_fit"] = {"vle_only_reference": {"form": "tau = b/T", "stats": reg_a["modelos"]["NRTL"]["estadistica"],
                                             "azeotrope_x1": [a["x1"] for a in reg_a["modelos"]["NRTL"]["azeotropo_predicho"][KAM]]},
                      "fits": {}}
for name, cfg in FITS.items():
    t = time.time()
    r = vle_binary_regression(vle["components"], VLE_SETS, models=["NRTL"], temperature_dependence="a+b/T",
                              nrtl_alpha=cfg["alpha"], compare_unifac=False, excess_enthalpy_sets=he_sets(cfg["temps"]),
                              he_weight_scan=SCAN)
    assert r["ok"], r
    JOINT[name] = r
    m = r["modelos"]["NRTL"]
    prm = NP.resolve_parameters(EW, r)
    all_T = by_isotherm(prm)
    res["3_joint_fit"]["fits"][name] = {
        "why": cfg["why"], "parameters": m["parametros"], "fixed": m["fijos"], "weights": r["objetivo"]["pesos"],
        "sigmas": r["objetivo"]["sigmas"], "vle_stats": m["estadistica"], "hE_stats": {k: v for k, v in m["estadistica_hE"].items()},
        "verdict": m["veredicto_conjunto"], "azeotrope_x1": [a["x1"] for a in m["azeotropo_predicho"][KAM]],
        "starts": m["arranques_conjuntos"], "thermo_check": m["verificacion_thermo"]["max_dif_hE_J_mol"],
        "all_isotherms_including_not_fitted": all_T, "AAD_all_points_J_mol": overall(all_T),
        "weight_scan": m["compromiso_VLE_hE"]["barrido"], "seconds": time.time() - t}
    print(name, [(q["nombre"], round(q["valor"], 3), round(q["ic95"], 3)) for q in m["parametros"]],
          "VLE", {k: round(v, 4) for k, v in m["estadistica"].items() if k.startswith("AAD")},
          "hE AAD", round(m["estadistica_hE"]["AAD_hE_J_mol"], 1), "all-T AAD", round(overall(all_T), 1),
          "verdict", m["veredicto_conjunto"]["representa_VLE_y_hE"], flush=True)
    for sc in m["compromiso_VLE_hE"]["barrido"]:
        print("   w_hE", sc["peso_hE"], "AAD_T", round(sc["AAD_T_K"], 3), "AAD_y", round(sc["AAD_y"], 4), "AAD_hE",
              round(sc["AAD_hE_J_mol"], 1), sc["representa_VLE_y_hE"], flush=True)
save()
if FAST:
    sys.exit(0)

# ------------------------------------------------------------------ 4. the joint sets through the column (re-solves)
from packages.eo import column_optimize as C  # noqa: E402
from packages.eo import dwsim_nrtl as DN  # noqa: E402

SPECS = [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": X_D},
         {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": X_BW}]


def run_eo(bp, hE):
    t = time.time()
    try:
        r = C.eo_column_optimize(EW, F, T_FEED, P, Z, N_TRAYS, FEED_TRAY, P, SPECS, activity_model="NRTL",
                                 binary_parameters=bp, excess_enthalpy=hE)
    except Exception as ex:  # noqa: BLE001 - a refusal or a failed start is a result to report
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"[:600], "seconds": time.time() - t}
    o, ev = r["optimum"], r["evidence"]
    return {"ok": True, "RR": o["reflux_ratio"], "BR": o["boilup_ratio"], "x_D": o["distillate"]["mole_fractions"]["ethanol"],
            "x_B_heavy": o["bottoms"]["mole_fractions"]["water"], "T_D_K": o["distillate"]["T_K"], "T_B_K": o["bottoms"]["T_K"],
            "D_mol_s": o["distillate"]["flow_mol_s"], "B_mol_s": o["bottoms"]["flow_mol_s"],
            "Q_reb_kW": r["reboiler_duty_MW"] * 1e3, "Q_cond_kW": -r["condenser_duty_MW"] * 1e3,
            "verified": ev["verified"], "reasons": ev["reasons"], "excess_enthalpy": r["properties"]["excess_enthalpy"],
            "hE_assumption_line": r["assumptions"][1], "warnings": r["warnings"], "seconds": time.time() - t}


def run_dw(e, params, mode="ExpData"):
    dw = DN.column(["Ethanol", "Water"], T_FEED, P, {"Ethanol": F * Z["ethanol"], "Water": F * Z["water"]}, N_TRAYS + 2,
                   FEED_TRAY + 1, e["RR"], P, bottoms_mol_s=e["B_mol_s"], params=params, dwsim_names=DW_EW, poynting=False,
                   enthalpy_mode=mode)
    if not dw["ok"]:
        return {"ok": False, "errors": dw["solver_errors"]}
    v = {"x_D": dw["distillate"]["mole_fractions"]["Ethanol"], "x_B_heavy": dw["bottoms"]["mole_fractions"]["Water"],
         "T_D_K": dw["distillate"]["T_K"], "T_B_K": dw["bottoms"]["T_K"], "Q_reb_kW": dw["reboiler_duty_kW"],
         "Q_cond_kW": dw["condenser_duty_kW"], "RR": e["RR"], "D_mol_s": F - e["B_mol_s"], "B_mol_s": e["B_mol_s"]}
    return {"ok": True, "mode": mode, "values": v, "parameters_held_by_dwsim": dw.get("parameters"),
            "dev_eo_vs_dwsim_pct": {q: dev(e[q], v[q]) for q in ("x_D", "x_B_heavy", "T_D_K", "T_B_K", "Q_reb_kW", "Q_cond_kW")}}


res["4_columns"] = {}
for name in ("J_all_T", "J_column_T", "J_363"):
    reg = JOINT[name]
    prm = NP.resolve_parameters(EW, reg)
    case = {"hE_on": run_eo(reg, True), "hE_off": run_eo(reg, False)}
    res["4_columns"][name] = case
    save()
    a, b = case["hE_on"], case["hE_off"]
    print(name, "on", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k in ("ok", "RR", "Q_reb_kW", "Q_cond_kW", "verified", "error", "seconds")},
          "off", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items() if k in ("ok", "RR", "Q_reb_kW", "Q_cond_kW", "verified", "error", "seconds")}, flush=True)
    if not (a["ok"] and b["ok"]):
        continue
    tb = streams_table(a, {"model": prm})
    hm = {s: tb[s]["measured"]["hE_J_mol"] for s in tb}
    e_model = first_order(b, {s: model_he(prm, *stream_states(b)[s][:2]) for s in tb})
    e_meas = first_order(b, {s: measured(*stream_states(b)[s][:2])["hE_J_mol"] for s in tb})
    q_meas = {"Q_reb_kW": b["Q_reb_kW"] + e_meas["dQ_reb_kW"], "Q_cond_kW": b["Q_cond_kW"] + e_meas["dQ_cond_kW"]}
    case["streams_hE_J_mol"] = tb
    case["hE_effect_solved_pct"] = {"Q_reb": dev(a["Q_reb_kW"], b["Q_reb_kW"]), "Q_cond": dev(a["Q_cond_kW"], b["Q_cond_kW"])}
    case["estimate_check"] = {"solved_dQ_reb_kW": a["Q_reb_kW"] - b["Q_reb_kW"], "estimated_dQ_reb_kW": e_model["dQ_reb_kW"],
                              "solved_dQ_cond_kW": a["Q_cond_kW"] - b["Q_cond_kW"], "estimated_dQ_cond_kW": e_model["dQ_cond_kW"]}
    case["with_measured_hE_estimate"] = {**q_meas, "method": "re-solve sin h^E + balance de entalpía con el h^E medido"}
    case["hE_on_vs_measured_estimate_pct"] = {q: dev(a[q], q_meas[q]) for q in q_meas}
    case["hE_off_vs_measured_estimate_pct"] = {q: dev(b[q], q_meas[q]) for q in q_meas}
    case["vs_M15b_reference_pct"] = {"hE_on": {q: dev(a[q], on[q]) for q in ("Q_reb_kW", "Q_cond_kW", "RR")},
                                     "hE_off": {q: dev(b[q], off[q]) for q in ("Q_reb_kW", "Q_cond_kW", "RR")}}
    # DWSIM like-for-like with the same a + b/T parameters: «ExpData» (no excess) vs EO without h^E
    try:
        dw = run_dw(b, prm)
    except Exception as ex:  # noqa: BLE001
        dw = {"ok": False, "errors": f"{type(ex).__name__}: {ex}"[:400]}
    case["dwsim_ExpData_like_for_like"] = dw
    if dw["ok"]:
        v = dw["values"]
        st = {"feed": (Z["ethanol"], T_FEED), "distillate": (v["x_D"], v["T_D_K"]), "bottoms": (1 - v["x_B_heavy"], v["T_B_K"])}
        d_meas = first_order(v, {s: measured(*st[s])["hE_J_mol"] for s in st})
        q_dw = {"Q_reb_kW": v["Q_reb_kW"] + d_meas["dQ_reb_kW"], "Q_cond_kW": v["Q_cond_kW"] + d_meas["dQ_cond_kW"]}
        case["dwsim_ExpData_plus_measured_hE"] = {**q_dw, "method": "DWSIM «ExpData» + balance de entalpía con el h^E medido",
                                                  "eo_hE_on_dev_pct": {q: dev(a[q], q_dw[q]) for q in q_dw}}
    print("  ", json.dumps({k: case[k] for k in ("hE_effect_solved_pct", "estimate_check", "hE_on_vs_measured_estimate_pct",
                                                 "hE_off_vs_measured_estimate_pct")}, default=float), flush=True)
    print("   dwsim", json.dumps(dw.get("dev_eo_vs_dwsim_pct") or dw, default=float)[:400],
          json.dumps(case.get("dwsim_ExpData_plus_measured_hE", {}).get("eo_hE_on_dev_pct"), default=float), flush=True)
    save()
print("done", flush=True)
