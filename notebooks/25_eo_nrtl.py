"""M15 evidence: EO column (IDAES + ipopt) with an NRTL liquid, against a published column and DWSIM.

1. Published reference: Simone, C. (2015) "Development of thermal separation units and a tutorial…", TU Wien
   (repositum.tuwien.at, handle 20.500.12708/6251), Tables 7, 9, 10 (thesis pp. 48-51): methanol-water, 10 000 kg/h,
   5 wt% methanol, 313 K, 1013 mbar, 12 stages incl. condenser and reboiler, feed stage 7, RR 4.36 (molar), D 474 kg/h;
   Aspen Plus RadFrac and gPROMS, both NRTL (parameters not printed). Solved here at the same RR and D.
2. eo_column_optimize(activity_model="NRTL") on the same feed with the specs the thesis' products imply
   (x_D ≥ 0.9168, methanol recovery ≥ 0.902), then DWSIM (NRTL, SAME parameters written into its package) at the
   optimum's RR and bottoms flow, with the four DWSIM liquid-enthalpy modes and the Poynting factor on/off.
3. vle_binary_regression (Kamihama 2012, 101.3 kPa, via VLizard) → eo_column_optimize on ethanol-water below the
   azeotrope, then DWSIM with the regressed parameters (they differ from DWSIM's databank, so this checks the setter)
   and with DWSIM's own databank for contrast.
Run: PYTHONPATH=. uv run python -u notebooks/25_eo_nrtl.py   (≈ 6 min) → notebooks/25_eo_nrtl.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemicals import MW  # noqa: E402

from packages.eo import column_optimize as C  # noqa: E402
from packages.eo import nrtl_properties as NP  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
MWM, MWW, MWE = MW("67-56-1"), MW("7732-18-5"), MW("64-17-5")
res: dict = {}


def dev(a, b):
    return (a / b - 1) * 100


# ------------------------------------------------------------------ 1. published column
nm, nw = 500 / MWM / 3.6, 9500 / MWW / 3.6
F = nm + nw
feed = {"flow_mol_s": F, "T_K": 313.0, "P_Pa": 101300.0, "z": {"methanol": nm / F, "water": nw / F}}
t = time.time()
st = C.simulate_column_nrtl(["methanol", "water"], feed, 10, 6, 101300.0, 4.36, distillate_mass_flow_kg_s=474 / 3600)
d, b = st["distillate"], st["bottoms"]
mine = {"T_top_K": d["T_K"], "T_bottom_K": b["T_K"], "Q_reb_kW": st["reboiler_duty_W"] / 1e3,
        "Q_cond_kW": st["condenser_duty_W"] / 1e3,
        "MeOH_top_kg_h": d["flow_mol_s"] * d["mole_fractions"]["methanol"] * MWM * 3.6,
        "W_top_kg_h": d["flow_mol_s"] * d["mole_fractions"]["water"] * MWW * 3.6,
        "MeOH_bottom_kg_h": b["flow_mol_s"] * b["mole_fractions"]["methanol"] * MWM * 3.6,
        "W_bottom_kg_h": b["flow_mol_s"] * b["mole_fractions"]["water"] * MWW * 3.6}
aspen = {"T_top_K": 339, "T_bottom_K": 373, "Q_reb_kW": 1503, "Q_cond_kW": -819, "MeOH_top_kg_h": 451,
         "W_top_kg_h": 23, "MeOH_bottom_kg_h": 49, "W_bottom_kg_h": 9477}
gproms = {"T_top_K": 339, "T_bottom_K": 373, "Q_reb_kW": 1522, "Q_cond_kW": -810, "MeOH_top_kg_h": 450,
          "W_top_kg_h": 24, "MeOH_bottom_kg_h": 50, "W_bottom_kg_h": 9476}
res["1_published"] = {"source": "Simone 2015 (TU Wien), Tablas 7, 9, 10, pp. 48-51", "eo_nrtl": mine,
                      "aspen_plus": aspen, "gproms": gproms,
                      "dev_vs_aspen_pct": {k: dev(mine[k], aspen[k]) for k in aspen},
                      "dev_vs_gproms_pct": {k: dev(mine[k], gproms[k]) for k in gproms},
                      "x_D": d["mole_fractions"], "x_B": b["mole_fractions"], "boilup_ratio": st["boilup_ratio"],
                      "thermo_bubble_check": st["thermo_bubble_check"]["points"], "seconds": time.time() - t}
print("1", json.dumps(res["1_published"]["dev_vs_aspen_pct"]), flush=True)

# ------------------------------------------------------------------ 2. optimum vs DWSIM (same parameters)
from packages.eo import dwsim_nrtl as DN  # noqa: E402

specs = [{"stream": "distillate", "component": "methanol", "kind": "mole_fraction", "min": 0.9168},
         {"stream": "distillate", "component": "methanol", "kind": "recovery", "min": 0.902}]
t = time.time()
r = C.eo_column_optimize(["methanol", "water"], F, 313.0, 101300.0, feed["z"], 10, 6, 101300.0, specs,
                         activity_model="NRTL")
o = r["optimum"]
params = NP.resolve_parameters(["methanol", "water"])
eo = {"x_D_MeOH": o["distillate"]["mole_fractions"]["methanol"], "x_B_W": o["bottoms"]["mole_fractions"]["water"],
      "T_D_K": o["distillate"]["T_K"], "T_B_K": o["bottoms"]["T_K"], "Q_reb_kW": r["reboiler_duty_MW"] * 1e3,
      "Q_cond_kW": -r["condenser_duty_MW"] * 1e3}
res["2_optimum"] = {"reflux_ratio": o["reflux_ratio"], "boilup_ratio": o["boilup_ratio"], "eo": eo,
                    "duals": r["duals_MW_per_unit_spec"], "verified": r["evidence"]["verified"],
                    "reasons": r["evidence"]["reasons"], "screening": r["properties"]["screening"]["unifac_cross_check"],
                    "thermo_bubble_check": r["properties"]["thermo_bubble_check"], "warnings": r["warnings"],
                    "seconds": r["seconds"], "dwsim": {}}
print("2 eo", o["reflux_ratio"], eo, r["evidence"]["verified"], flush=True)
flows = {"Methanol": F * feed["z"]["methanol"], "Water": F * feed["z"]["water"]}
for mode in ("LiqCp_Excess", "ExpData", "Ideal", "Excess"):
    for poy in (False, True):
        dw = DN.column(["Methanol", "Water"], 313.0, 101300.0, flows, 12, 7, o["reflux_ratio"], 101300.0,
                       bottoms_mol_s=o["bottoms"]["flow_mol_s"], params=params,
                       dwsim_names={"methanol": "Methanol", "water": "Water"}, poynting=poy, enthalpy_mode=mode)
        key = f"{mode}{'_poynting' if poy else ''}"
        if not dw["ok"]:
            res["2_optimum"]["dwsim"][key] = {"ok": False, "errors": dw["solver_errors"]}
            continue
        v = {"x_D_MeOH": dw["distillate"]["mole_fractions"]["Methanol"],
             "x_B_W": dw["bottoms"]["mole_fractions"]["Water"], "T_D_K": dw["distillate"]["T_K"],
             "T_B_K": dw["bottoms"]["T_K"], "Q_reb_kW": dw["reboiler_duty_kW"], "Q_cond_kW": dw["condenser_duty_kW"]}
        res["2_optimum"]["dwsim"][key] = {"values": v, "dev_eo_vs_dwsim_pct": {k: dev(eo[k], v[k]) for k in v},
                                          "settings": dw["settings"], "parameters_used": dw["parameters_used"]}
        print("2 dwsim", key, json.dumps(res["2_optimum"]["dwsim"][key]["dev_eo_vs_dwsim_pct"]), flush=True)

# ------------------------------------------------------------------ 3. regression → column → DWSIM
from packages.vle.tool import vle_binary_regression  # noqa: E402

EW = json.loads((Path(__file__).parents[1] / "tests/data/vle_ethanol_water_vlizard.json").read_text())
k = next(k for k in EW["sets"] if "101" in k)
reg = vle_binary_regression(EW["components"], [{"label": k, "points": EW["sets"][k], "citation": EW["citation"]}],
                            models=["NRTL"], compare_unifac=False)
specs3 = [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": 0.70},
          {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": 0.995}]
r3 = C.eo_column_optimize(["ethanol", "water"], 100.0, 350.0, 101325.0, {"ethanol": 0.1, "water": 0.9}, 20, 14,
                          101325.0, specs3, activity_model="NRTL", binary_parameters=reg)
o3 = r3["optimum"]
p3 = NP.resolve_parameters(["ethanol", "water"], reg)
eo3 = {"x_D_EtOH": o3["distillate"]["mole_fractions"]["ethanol"], "x_B_W": o3["bottoms"]["mole_fractions"]["water"],
       "T_D_K": o3["distillate"]["T_K"], "T_B_K": o3["bottoms"]["T_K"], "Q_reb_kW": r3["reboiler_duty_MW"] * 1e3,
       "Q_cond_kW": -r3["condenser_duty_MW"] * 1e3}
res["3_regression_chain"] = {"pair": r3["properties"]["binary_parameters"], "reflux_ratio": o3["reflux_ratio"],
                             "boilup_ratio": o3["boilup_ratio"], "eo": eo3, "verified": r3["evidence"]["verified"],
                             "reasons": r3["evidence"]["reasons"], "warnings": r3["warnings"],
                             "azeotropes": r3["properties"]["screening"]["azeotropes_with_these_parameters"], "dwsim": {}}
print("3 eo", o3["reflux_ratio"], eo3, r3["evidence"]["verified"], flush=True)
for label, pp in (("mismos_parametros", p3), ("banco_propio_de_DWSIM", None)):
    dw = DN.column(["Ethanol", "Water"], 350.0, 101325.0, {"Ethanol": 10.0, "Water": 90.0}, 22, 15,
                   o3["reflux_ratio"], 101325.0, bottoms_mol_s=o3["bottoms"]["flow_mol_s"], params=pp,
                   dwsim_names={"ethanol": "Ethanol", "water": "Water"}, poynting=False, enthalpy_mode="LiqCp_Excess")
    if not dw["ok"]:
        res["3_regression_chain"]["dwsim"][label] = {"ok": False, "errors": dw["solver_errors"]}
        continue
    v = {"x_D_EtOH": dw["distillate"]["mole_fractions"]["Ethanol"], "x_B_W": dw["bottoms"]["mole_fractions"]["Water"],
         "T_D_K": dw["distillate"]["T_K"], "T_B_K": dw["bottoms"]["T_K"], "Q_reb_kW": dw["reboiler_duty_kW"],
         "Q_cond_kW": dw["condenser_duty_kW"]}
    res["3_regression_chain"]["dwsim"][label] = {"values": v, "dev_eo_vs_dwsim_pct": {q: dev(eo3[q], v[q]) for q in v},
                                                 "parameters_used": dw["parameters_used"]}
    print("3 dwsim", label, json.dumps(res["3_regression_chain"]["dwsim"][label]["dev_eo_vs_dwsim_pct"]), flush=True)

OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
print("wrote", OUT)
