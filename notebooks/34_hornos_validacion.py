"""Notebook 34 — Validación de hornos (bloque M18b): emisividad del gas, sección radiante y banco de convección.

Reproduce, con las mismas funciones que `tests/test_fired_heater_validation.py` (donde están las fuentes, la fecha de
lectura y las URL), la tabla de desviaciones contra literatura:
1. emisividad   WSGG de Smith 1982 (el motor) contra Leckner 1972 (independiente, coeficientes del manual SIERRA/Fuego de
                Sandia) en 70 puntos, y las dos lecturas de cartas de Kern (ej. 19.1 y 19.4).
2. radiante     Kern (1950) ej. 19.1 (Lobo-Evans) y 19.2 (Wilson-Lobo-Hottel): carga radiante y pared divisoria.
3. banco        Coeficiente de convección: Incropera ej. 7.7 (tresbolillo), Çengel ej. 7-7 (en línea), Grimison y ESDU 73031.
4. convección   Radiación del gas entre tubos (Kern ej. 19.4) y sección completa de Nelson (1941) ej. 68.
5. aletas       ESCOA revisada (Ganapathy 2015 ap. E): ejemplos E.1 (aleta sólida, también como sección) y E.10 (segmentada).

Ejecutar: PYTHONPATH=. uv run python -u notebooks/34_hornos_validacion.py   (≈ 10 s, sin DWSIM ni red) → .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_fired_heater_validation as V  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
res: dict = {}


def save() -> None:
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


# --------------------------------------------------------------------------------------------- 1. emisividad
grid = {k: V.emissivity_grid(k) for k in (V.H2O_PE_PRINTED, V.H2O_PE_RECALLED)}
summary = {}
for k, rows in grid.items():
    dev = [r["deviation"] for r in rows]
    box = [r["deviation"] for r in rows if r["pL_atm_m"] == 1.0 and 1000.0 <= r["T_K"] <= 1600.0]
    summary[f"Leckner con coeficiente de presión equivalente {k}"] = {
        "puntos": len(dev), "min": min(dev), "max": max(dev), "media": sum(dev) / len(dev),
        "media_absoluta": sum(abs(d) for d in dev) / len(dev), "hogar_pL_1_atm_m_max_abs": max(abs(d) for d in box)}
pw, pc, k191 = V.KERN_GAS["p_H2O"], V.KERN_GAS["p_CO2"], V.KERN_19_1
smith = V.FH.gas_emissivity(k191["T_K"], pw * V.ATM, pc * V.ATM, k191["L_m"])["emissivity"]
leck = V.leckner_mixture(k191["T_K"], pw, pc, k191["L_m"], pe_coefficient=V.H2O_PE_RECALLED)["emissivity"]
res["1_emisividad"] = {"smith_1982_vs_leckner_1972": summary,
                       "kern_19_1": {"kern_carta": k191["printed"], "smith_1982": smith, "leckner_1972": leck,
                                     "kern_sobre_leckner": k191["printed"] / leck - 1},
                       "veredicto": "Smith 1982 y Leckner 1972 concuerdan; las cartas de Kern (1950) son el valor atípico a "
                                    "trayectoria larga. Se conserva Smith 1982 como único modelo."}
for name, s in summary.items():
    print(f"1 {name}: {100*s['min']:+.1f} … {100*s['max']:+.1f} % (media abs. {100*s['media_absoluta']:.1f} %)")
print(f"1 Kern 19.1: carta {k191['printed']:.3f}, Smith {smith:.3f}, Leckner {leck:.3f}")
save()

# --------------------------------------------------------------------------------------------- 2. radiante
rad = {}
for tag, run, ans in (("kern_19_1_propio", V.kern_19_1, V.KERN_19_1_ANSWER), ("kern_19_2_propio", V.kern_19_2, V.KERN_19_2_ANSWER)):
    r = run()
    rad[tag] = {"carga_rel": r["radiant_duty_W"] / V.BTU_H / ans["Q_Btu_h"] - 1,
                "pared_divisoria_dif_K": r["bridgewall_T_K"] - V.F2K(ans["TG_F"]), "balance": r["evidence"]["energy_balance_error"]}
r = V.kern_19_1(geometry={"mean_beam_length_m": 15.0 * V.FT}, gas_emissivity_override=0.496)
rad["kern_19_1_con_emisividad_de_kern"] = {"carga_rel": r["radiant_duty_W"] / V.BTU_H / V.KERN_19_1_ANSWER["Q_Btu_h"] - 1,
                                           "pared_divisoria_dif_K": r["bridgewall_T_K"] - V.F2K(V.KERN_19_1_ANSWER["TG_F"])}
res["2_radiante"] = rad
for tag, x in rad.items():
    print(f"2 {tag}: carga {100*x['carga_rel']:+.1f} %, pared divisoria {x['pared_divisoria_dif_K']:+.1f} K")
save()

# --------------------------------------------------------------------------------------------- 3. banco
inc_b, inc_e, cen = V.incropera_7_7(False), V.incropera_7_7(True), V.cengel_7_7()
band = V.correlation_band()
devs = [r[k] for r in band for k in ("grimison_dev", "esdu_dev")]
res["3_banco"] = {"incropera_7_7_h_rel_props_libro": inc_b["h"] / V.INCROPERA_7_7["h"] - 1,
                  "incropera_7_7_h_rel_props_cantera": inc_e["h"] / V.INCROPERA_7_7["h"] - 1,
                  "cengel_7_7_Q_rel": cen["Q_W_m"] / V.CENGEL_7_7["Q_W_m"] - 1,
                  "grimison_esdu_vs_zukauskas": {"n": len(devs), "min": min(devs), "max": max(devs)}}
b = res["3_banco"]
print(f"3 Incropera 7.7 h {100*b['incropera_7_7_h_rel_props_libro']:+.1f} % (Cantera {100*b['incropera_7_7_h_rel_props_cantera']:+.1f} %); "
      f"Çengel 7-7 Q {100*b['cengel_7_7_Q_rel']:+.2f} %; Grimison/ESDU {100*min(devs):+.0f} … {100*max(devs):+.0f} %")
save()

# --------------------------------------------------------------------------------------------- 4. convección
k194, n68 = V.kern_19_4(), V.nelson_68()
x = V.NELSON_68
res["4_conveccion"] = {
    "kern_19_4_h_rad_rel_con_su_L": k194["h_kern_L"] / k194["kern_eq_19_18"] - 1,
    "kern_19_4_h_rad_rel_con_3_6_V_A": k194["h_engine_L"] / k194["kern_eq_19_18"] - 1,
    "nelson_68_U_rel": n68["U_lmtd_Btu"] / x["U"] - 1, "nelson_68_carga_rel_gas": n68["duty_Btu_h"] / x["gas_gives_Btu_h"] - 1,
    "nelson_68_carga_rel_aceite": n68["duty_Btu_h"] / x["absorbed_Btu_h"] - 1,
    "nelson_68_chimenea_dif_K": n68["stack_T_K"] - V.F2K(x["stack_F"])}
c = res["4_conveccion"]
print(f"4 Kern 19.4 h_rad {100*c['kern_19_4_h_rad_rel_con_su_L']:+.0f} % (3.6 V/A {100*c['kern_19_4_h_rad_rel_con_3_6_V_A']:+.0f} %); "
      f"Nelson 68 U {100*c['nelson_68_U_rel']:+.1f} %, carga {100*c['nelson_68_carga_rel_gas']:+.1f} % (gas) / "
      f"{100*c['nelson_68_carga_rel_aceite']:+.1f} % (aceite), chimenea {c['nelson_68_chimenea_dif_K']:+.0f} K")
save()

# --------------------------------------------------------------------------------------------- 5. tubos aletados (ESCOA)
e1, e10, sec = V.gan_coefficient(V.GAN_E1, "solid_fin"), V.gan_coefficient(V.GAN_E10, "serrated_fin"), V.gan_e1_section()
res["5_aletas"] = {"ganapathy_E1_hc_rel": e1["hc"] / V.GAN_E1["hc"] - 1, "ganapathy_E10_hc_rel": e10["hc"] / V.GAN_E10["hc"] - 1,
                   "ganapathy_E1_seccion_carga_rel": sec["duty_W"] / 1e6 / V.GAN_E1["duty_MW"] - 1,
                   "ganapathy_E1_seccion_U_rel": sec["U_lmtd"] / V.GAN_E1["U_W"] - 1,
                   "ganapathy_E1_seccion_gas_salida_dif_K": sec["stack_T_K"] - 273.15 - V.GAN_E1["gas_out_C"]}
f = res["5_aletas"]
print(f"5 ESCOA: E.1 h_c {100*f['ganapathy_E1_hc_rel']:+.2f} %, E.10 h_c {100*f['ganapathy_E10_hc_rel']:+.1f} %; E.1 sección "
      f"carga {100*f['ganapathy_E1_seccion_carga_rel']:+.1f} %, U {100*f['ganapathy_E1_seccion_U_rel']:+.1f} %, gas "
      f"{f['ganapathy_E1_seccion_gas_salida_dif_K']:+.1f} K")
res["bandas"] = {"fired_heater_radiant": V.FH.RADIANT_VALIDATION, "fired_heater_convection": V.FH.CONVECTION_VALIDATION,
                 "fired_heater_convection_aletas": V.FH.FINNED_VALIDATION}
save()
print("hecho")
