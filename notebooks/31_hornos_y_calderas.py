"""Notebook 31 — Hornos y calderas (bloque M18): `combustion_efficiency`, `fired_heater_radiant`, `fired_heater_convection`.

Qué demuestra, en orden:
1. doe        Eficiencia de combustión contra la tabla «Combustion Efficiency for Natural Gas» del U.S. DOE Steam Tip Sheet #4
              (DOE/GO-102012-3405, enero 2012): 25 valores y sus 5 pares exceso de aire ↔ O2. La hoja no dice qué gas ni qué
              aire usó: se mide la sensibilidad.
2. ornl       Pérdida por chimenea contra la Tabla B.1 «Natural gas stack loss (%)» de G. Harrell, Steam System Survey Guide,
              ORNL/TM-2001/263 (mayo 2002): 180 valores. Su Tabla 3 muestra que el O2 de esa guía está en base húmeda.
3. ibrahim    Horno de crudo con gas natural de Al-Haj Ibrahim y Al-Qassimi, Periodica Polytechnica Chem. Eng. 52/2 (2008)
              61-69, ejemplo 2: aire, gases y composición. Su poder calorífico neto no es consistente con su propio gas.
4. radiante   Sección radiante: alpha, factor de intercambio y emisividad del gas contra las lecturas de las cartas de Kern
              (cap. 19, ejemplos 19.1, 19.4 y 19.5, transcritos en el cuaderno del Textbook Companion de FOSSEE, IIT Bombay;
              el libro no se abrió). El ejemplo de Al-Haj Ibrahim 2010 (IntechOpen) no cierra consigo mismo. Comprobaciones
              internas: el balance cierra, la carga sube con el fuego, la fracción absorbida baja con el exceso de aire.
5. horno      Un horno de ejemplo de punta a punta: sección radiante → banco de convección → temperatura de chimenea como
              RESULTADO → eficiencia. Geometría y operación ILUSTRATIVAS (ver MOCKS.md).
6. servicios  Cómo pasa el usuario esa eficiencia a `utility_costs` (bloque M17) y qué pasa si mezcla las bases HHV y LHV.

Ejecutar: PYTHONPATH=. uv run python -u notebooks/31_hornos_y_calderas.py   (≈ 20 s, sin DWSIM ni red) → .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_combustion as T  # noqa: E402
from packages.dwsim_client import tools_utilities as TU  # noqa: E402
from packages.energy import combustion as CB  # noqa: E402
from packages.energy import fired_heater as FH  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
P = 101325.0
T70F = (70 + 459.67) / 1.8
res: dict = {}


def save() -> None:
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def stats(d) -> dict:
    d = np.array(d)
    return {"min": float(d.min()), "media": float(d.mean()), "max": float(d.max()), "rms": float(np.sqrt((d ** 2).mean())), "n": len(d)}


def gas(comp: dict, T_K: float = T70F) -> dict:
    return {"type": "gas", "composition": comp, "T_K": T_K}


def dry_air(T_K: float = T70F, o2: float | None = None) -> dict:
    a = {"T_K": T_K, "humidity_ratio_kg_kg": 0.0}
    if o2:
        a["dry_air_mole_fractions"] = {"O2": o2, "N2": 1 - o2}
    return a


def eff_hhv(fuel, air, stack_T, **kw) -> dict:
    return T.combustion_efficiency(fuel, air, P, stack_T_K=stack_T, setting_loss_fraction=0.0, setting_loss_basis="HHV", **kw)


# --------------------------------------------------------------------------------------------- 1. DOE Steam Tip Sheet #4
DOE = {(9.5, 2.0): (85.4, 83.1, 80.8, 78.4, 76.0), (15.0, 3.0): (85.2, 82.8, 80.4, 77.9, 75.4),
       (28.1, 5.0): (84.7, 82.1, 79.5, 76.7, 74.0), (44.9, 7.0): (84.1, 81.2, 78.2, 75.2, 72.1),
       (81.6, 10.0): (82.8, 79.3, 75.6, 71.9, 68.2)}
DT_DOE = (200, 300, 400, 500, 600)


def doe_run(comp: dict, T_air: float, o2: float | None):
    table, diffs, pairs = [], [], []
    for (ea, o2_pub), row in DOE.items():
        mine = []
        for dT, v in zip(DT_DOE, row):
            r = eff_hhv(gas(comp, T_air), dry_air(T_air, o2), T_air + dT / 1.8, excess_air_fraction=ea / 100)
            mine.append(100 * r["efficiency"]["HHV"]["efficiency"])
            diffs.append(mine[-1] - v)
        pairs.append({"exceso_%": ea, "O2_seco_publicado_%": o2_pub, "O2_seco_calculado_%": 100 * r["flue_gas"]["dry_mole_fractions"]["O2"]})
        table.append({"exceso_%": ea, "publicado": row, "calculado": [round(x, 2) for x in mine]})
    return table, stats(diffs), pairs


table, st, pairs = doe_run({"CH4": 1.0}, T70F, 0.209)
sens = {name: doe_run(comp, Ta, o2)[1] for name, comp, Ta, o2 in (
    ("CH4, aire 20.9 % O2, 70 °F (caso base)", {"CH4": 1.0}, T70F, 0.209),
    ("CH4, aire 21 % O2 + 1 % Ar (air.yaml de Cantera)", {"CH4": 1.0}, T70F, None),
    ("CH4, aire a 100 °F", {"CH4": 1.0}, (100 + 459.67) / 1.8, 0.209),
    ("gas 90/5/5 (CH4/C2H6/N2)", {"CH4": 0.90, "C2H6": 0.05, "N2": 0.05}, T70F, 0.209),
    ("gas 85/10/3/2 (CH4/C2H6/C3H8/N2)", {"CH4": 0.85, "C2H6": 0.10, "C3H8": 0.03, "N2": 0.02}, T70F, 0.209))}
res["1_doe_tip_sheet_4"] = {
    "fuente": "U.S. DOE, Steam Tip Sheet #4, «Improve Your Boiler's Combustion Efficiency», DOE/GO-102012-3405, enero 2012 "
              "(https://www.energy.gov/sites/prod/files/2014/05/f16/steam4_boiler_efficiency.pdf), leída 2026-11-14",
    "supuesto_para_comparar": "metano puro, aire seco 20.9 % O2 / 79.1 % N2 a 70 °F, combustión completa, sin pérdida por "
                              "paredes; la hoja solo dice «natural gas», aire sin vapor de agua y combustión completa",
    "tabla": table, "diferencia_puntos_calculado_menos_publicado": st, "pares_exceso_O2": pairs, "sensibilidad": sens}
print(f"1 DOE #4: 25 valores, calculado − publicado = {st['min']:+.2f} a {st['max']:+.2f} puntos (media {st['media']:+.2f}); "
      f"O2 seco {', '.join(f'{p_['O2_seco_calculado_%']:.2f}' for p_ in pairs)} contra 2.0, 3.0, 5.0, 7.0, 10.0", flush=True)
save()

# --------------------------------------------------------------------------------------------- 2. ORNL/TM-2001/263
DT_ORNL = (230, 250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450, 470, 490, 510)
B1 = """1 14.49 14.92 15.36 15.79 16.23 16.67 17.11 17.55 17.99 18.43 18.88 19.32 19.77 20.21 20.66
2 14.72 15.17 15.63 16.09 16.55 17.01 17.47 17.93 18.39 18.86 19.32 19.79 20.26 20.73 21.20
3 14.98 15.46 15.94 16.42 16.90 17.38 17.87 18.36 18.84 19.33 19.82 20.31 20.80 21.30 21.79
4 15.26 15.77 16.28 16.79 17.29 17.81 18.32 18.83 19.35 19.86 20.38 20.90 21.41 21.93 22.46
5 15.59 16.12 16.66 17.20 17.74 18.28 18.82 19.36 19.91 20.46 21.00 21.55 22.10 22.65 23.20
6 15.96 16.52 17.10 17.67 18.24 18.82 19.39 19.97 20.55 21.13 21.71 22.29 22.88 23.46 24.05
7 16.38 16.98 17.59 18.20 18.82 19.43 20.04 20.66 21.28 21.90 22.52 23.14 23.77 24.39 25.02
8 16.86 17.51 18.16 18.82 19.48 20.14 20.80 21.46 22.12 22.79 23.46 24.12 24.79 25.47 26.14
9 17.42 18.13 18.83 19.54 20.25 20.96 21.68 22.39 23.11 23.83 24.55 25.27 25.99 26.72 27.44
10 18.09 18.86 19.62 20.39 21.16 21.94 22.71 23.49 24.27 25.05 25.83 26.62 27.41 28.19 28.98
11 18.89 19.73 20.57 21.42 22.26 23.11 23.96 24.81 25.67 26.52 27.38 28.24 29.10 29.97 30.83
12 19.87 20.80 21.73 22.66 23.60 24.54 25.48 26.43 27.37 28.32 29.27 30.22 31.18 32.13 33.09"""
rows = [[float(x) for x in line.split()] for line in B1.splitlines()]


def ornl_run(comp: dict, basis: str) -> dict:
    d = []
    for row in rows:
        for dT, v in zip(DT_ORNL, row[1:]):
            r = eff_hhv(gas(comp), dry_air(), T70F + dT / 1.8, flue_O2={"mole_fraction": row[0] / 100, "basis": basis})
            d.append(100 * (1 - r["efficiency"]["HHV"]["efficiency"]) - v)
    return stats(d)


ornl = {"CH4, O2 en base húmeda": ornl_run({"CH4": 1.0}, "wet"), "CH4, O2 en base seca": ornl_run({"CH4": 1.0}, "dry"),
        "CH4 + 2.1 % N2 (23 000 Btu/lb, el gas de la guía), O2 húmedo": ornl_run({"CH4": 0.9789, "N2": 0.0211}, "wet")}
t3 = []
for o2, ea in ((1.5, 8.5), (3.0, 18.0), (7.0, 55.0)):
    t3.append({"O2_%": o2, "exceso_publicado_%": ea, **{f"exceso_calculado_base_{b}_%": 100 * CB.solve_combustion(
        gas({"CH4": 1.0}), dry_air(), P, flue_O2={"mole_fraction": o2 / 100, "basis": b}).excess_air for b in ("wet", "dry")}})
res["2_ornl_steam_survey_guide"] = {
    "fuente": "G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263, mayo 2002, Tabla B.1 y Tabla 3 "
              "(https://www.energy.gov/sites/prod/files/2014/04/f15/steam_survey_guide.pdf), leída 2026-11-14",
    "supuesto_para_comparar": "metano puro, aire seco de air.yaml a 70 °F (la temperatura del ejemplo de la guía); la guía "
                              "da un gas de 23 000 Btu/lb sin composición y no dice la base del O2",
    "diferencia_puntos_de_perdida_calculado_menos_publicado": ornl, "tabla_3_O2_contra_exceso": t3}
w = ornl["CH4, O2 en base húmeda"]
print(f"2 ORNL B.1: 180 valores, base húmeda {w['min']:+.2f} a {w['max']:+.2f} puntos de pérdida (media {w['media']:+.2f}); "
      f"en base seca {ornl['CH4, O2 en base seca']['min']:+.2f} a {ornl['CH4, O2 en base seca']['max']:+.2f}", flush=True)
save()

# --------------------------------------------------------------------------------------------- 3. Ibrahim y Al-Qassimi 2008
IBR = {"CH4": 0.8043, "C2H6": 0.0902, "C3H8": 0.0454, "i-C4H10": 0.0020, "n-C4H10": 0.0032, "i-C5H12": 0.0004,
       "n-C5H12": 0.0002, "CO2": 0.0352, "H2S": 0.0009, "N2": 0.01735}
s_ = sum(IBR.values())
ibr_fuel = gas({k: v / s_ for k, v in IBR.items()}, 298.15)
y_w = 0.015
r = T.combustion_efficiency(ibr_fuel, {"T_K": 298.15, "humidity_ratio_kg_kg": y_w / (1 - y_w) * 18.01528 / 28.9582}, P,
                            excess_air_fraction=0.25, stack_T_K=446 + 273.15, setting_loss_fraction=0.05, setting_loss_basis="LHV")
water = r["fuel"]["elements_mol_per_kg"]["H"] / 2 * r["fuel"]["molar_mass_kg_kmol"] / 1000
x = r["flue_gas"]["wet_mole_fractions"]
res["3_ibrahim_2008_ejemplo_2"] = {
    "fuente": "H. Al-Haj Ibrahim y M. Al-Qassimi, Periodica Polytechnica Chem. Eng. 52/2 (2008) 61-69, ejemplo 2; composición "
              "del gas de la Tabla 3 de Al-Haj Ibrahim, «Fired Process Heaters» (IntechOpen 2010), que suma 0.99915 y se normaliza",
    "comparacion": {
        "poder_calorifico_superior_kJ_kmol": {"publicado": 976029.6, "calculado": r["fuel"]["HHV_J_mol"]},
        "aire_humedo_kmol_h": {"publicado": 1589.014, "calculado": 120 * r["air"]["stoichiometric_dry_air_mol_per_mol_fuel"] * 1.25 / (1 - y_w)},
        "gases_kmol_h": {"publicado": 1720.9, "calculado": 120 * r["flue_gas"]["mol_per_mol_fuel"]},
        "CO2": {"publicado": 0.08234, "calculado": x["CO2"]}, "H2O": {"publicado": 0.15968, "calculado": x["H2O"]},
        "O2": {"publicado": 0.0382, "calculado": x["O2"]}, "N2_mas_Ar": {"publicado": 0.7179, "calculado": x["N2"] + x["Ar"]}},
    "inconsistencia_de_la_fuente": {
        "poder_calorifico_neto_publicado_kJ_kmol": 927844.41, "poder_calorifico_inferior_calculado_kJ_kmol": r["fuel"]["LHV_J_mol"],
        "agua_formada_mol_por_mol_de_gas": water,
        "superior_menos_neto_publicados_entre_calor_latente_del_agua_formada": (976029.6 - 927844.41) / (water * CB._latent_298()),
        "nota": "La diferencia entre los dos poderes caloríficos publicados equivale al calor latente de la mitad del agua que "
                "forma ese gas. Por eso su eficiencia (75.78 %) no se reproduce."},
    "eficiencia_sobre_el_calor_total_de_entrada_%": {"publicada": 75.78, "calculada": 100 * r["efficiency"]["LHV"]["efficiency_on_total_input"]}}
for k, v in res["3_ibrahim_2008_ejemplo_2"]["comparacion"].items():
    v["diferencia_%"] = 100 * (v["calculado"] / v["publicado"] - 1)
c3 = res["3_ibrahim_2008_ejemplo_2"]["comparacion"]
print(f"3 Ibrahim 2008: PCS {c3['poder_calorifico_superior_kJ_kmol']['diferencia_%']:+.2f} %, aire {c3['aire_humedo_kmol_h']['diferencia_%']:+.2f} %, "
      f"gases {c3['gases_kmol_h']['diferencia_%']:+.2f} %, H2O {c3['H2O']['diferencia_%']:+.2f} %; eficiencia "
      f"{res['3_ibrahim_2008_ejemplo_2']['eficiencia_sobre_el_calor_total_de_entrada_%']['calculada']:.1f} contra 75.78 "
      f"(su PCN no es consistente)", flush=True)
save()

# --------------------------------------------------------------------------------------------- 4. sección radiante
atm, ft = 101325.0, 0.3048
e191 = FH.gas_emissivity((1730 + 459.67) / 1.8, 0.1248 * atm, 0.1084 * atm, 15.0 * ft)
e194 = FH.gas_emissivity((1500 + 459.67) / 1.8, 0.1248 * atm, 0.1084 * atm, 0.565 * ft)
kern = {
    "fuente": "D. Q. Kern, Process Heat Transfer (1950), cap. 19, ejemplos 19.1, 19.4 y 19.5, según el cuaderno «Chapter 19: "
              "Furnace Calculations» del Python Textbook Companion de FOSSEE, IIT Bombay (tbc-python.fossee.in), leído "
              "2026-11-14. Son lecturas de cartas; el libro no se abrió.",
    "alpha_paso_8.5_in_tubo_5_in": {"carta_Fig_19.11": 0.937, "calculado": FH.tube_row_alpha(8.5, 5.0)["alpha"]},
    "factor_de_intercambio": [
        {"ejemplo": ej, "emisividad_del_gas": eg, "AR_entre_alphaAcp": ratio, "carta_Fig_19.15": chart,
         "calculado_tubo_0.9": FH.exchange_factor(eg, 1.0, 1.0 + ratio, 0.9),
         "calculado_tubo_negro": FH.exchange_factor(eg, 1.0, 1.0 + ratio, 1.0)}
        for ej, eg, ratio, chart in (("19.1", 0.496, 1.09, 0.635), ("19.5", 0.265, 2.49, 0.56))],
    "emisividad_del_gas": [
        {"ejemplo": "19.4 (banco, pL 0.13 atm·ft, 1500 °F)", "cartas_Figs_19.12_19.13": 2750 / (0.173 * 19.5967 ** 4),
         "WSGG_Smith_1982": e194["emissivity"]},
        {"ejemplo": "19.1 (hogar, pL 3.5 atm·ft, 1730 °F)", "cartas_Figs_19.12_19.13": 0.496, "WSGG_Smith_1982": e191["emissivity"]}],
    "coeficiente_de_radiacion_del_gas_ej_19.4_Btu_h_ft2_F": {
        "Kern": 2.52, "calculado": FH.gas_radiation_coefficient((1500 + 459.67) / 1.8, (650 + 459.67) / 1.8, e194["emissivity"], 0.9) / 5.678263},
    "nota_tubo": "La transcripción no dice qué emisividad de tubo usa la Fig. 19.15; se dan los dos extremos."}
root = brentq(lambda Tg: 9.0748e-5 * Tg ** 4 + 7.9153e4 * Tg - 1.5533e8, 500.0, 3000.0)
geo_i = {"tube_outside_diameter_m": 0.219, "tube_pitch_m": 0.394, "tube_effective_length_m": 20.024, "n_tubes": 60,
         "envelope_area_m2": 5000.0, "alpha": 0.835, "shield_cold_plane_area_m2": 8 * 0.394 * 20.024}
ri = T.fired_heater_radiant("rating", ibr_fuel, {"T_K": 298.15, "relative_humidity": 0.4}, P, 120e3 / 3600 * 20.046e-3, geo_i, 0.9,
                            30.66e3 / 3600, 0.05, "LHV", excess_air_fraction=0.25, tube_wall_T_K=100 + 0.5 * (483 + 628),
                            exchange_factor=0.97)
ibr10 = {"fuente": "H. Al-Haj Ibrahim, «Fired Process Heaters», cap. 16 de Matlab - Modelling, Programming and Simulations, "
                   "Sciyo/IntechOpen 2010 (CC BY-NC-SA 3.0), ec. (30), Tablas 2-6 y Apéndice 3",
         "temperatura_efectiva_que_dice_el_texto_K": 1278.0, "raiz_de_su_propia_ecuacion_30_K": root,
         "este_motor_con_sus_entradas_K": ri["bridgewall_T_K"], "calor_radiante_publicado_kJ_h": 6.182e7,
         "calor_radiante_de_este_motor_kJ_h": ri["radiant_duty_W"] * 3.6,
         "nota": "Con F = 0.97, alpha = 0.835, su temperatura de pared y su coeficiente de convección. El texto da 1278 K, pero la "
                 "ecuación que publica se anula en 966 K; además su tabla 6 dice 800 °C a la salida del hogar. No sirve como "
                 "validación cuantitativa."}
GEO = {"tube_outside_diameter_m": 0.1143, "tube_pitch_m": 0.2032, "tube_effective_length_m": 12.0, "n_tubes": 60,
       "envelope_area_m2": 2 * (12 * 6 + 12 * 7 + 6 * 7), "volume_m3": 12 * 6 * 7.0}
NG = gas({"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, 300.0)
AIR = {"T_K": 300.0, "humidity_ratio_kg_kg": 0.008}
RAD = dict(fuel=NG, air=AIR, pressure_Pa=P, geometry=GEO, tube_emissivity=0.85, convective_coefficient_W_m2K=10.0,
           setting_loss_fraction=0.02, setting_loss_basis="LHV", tube_wall_T_K=700.0)


def rad_row(label, **kw) -> dict:
    o = T.fired_heater_radiant("rating", **{**RAD, "fuel_flow_kg_s": 0.3, "excess_air_fraction": 0.15, **kw})
    return {"caso": label, "pared_divisoria_K": o["bridgewall_T_K"], "carga_radiante_MW": o["radiant_duty_W"] / 1e6,
            "fraccion_absorbida_del_PCI": o["absorbed_fraction_of_LHV"], "flujo_medio_kW_m2": o["average_flux_W_m2_tube_outside"] / 1e3,
            "emisividad_del_gas": o["radiation"]["gas_emissivity"], "factor_de_intercambio": o["radiation"]["exchange_factor"],
            "cierre_del_balance": o["evidence"]["energy_balance_error"]}


internal = [rad_row(f"combustible {m} kg/s", fuel_flow_kg_s=m) for m in (0.2, 0.3, 0.4)] + \
           [rad_row(f"exceso de aire {int(100 * e)} %", excess_air_fraction=e) for e in (0.05, 0.15, 0.30, 0.60)]
base_em = internal[1]["emisividad_del_gas"]
sens_em = [rad_row(f"emisividad × {f}", gas_emissivity=base_em * f) for f in (0.8, 1.0, 1.25)]
res["4_seccion_radiante"] = {"kern_cap_19": kern, "ibrahim_2010": ibr10, "comprobaciones_internas_horno_ilustrativo": internal,
                             "sensibilidad_a_la_emisividad_del_gas": sens_em}
print(f"4 Kern: alpha {kern['alpha_paso_8.5_in_tubo_5_in']['calculado']:.3f} contra 0.937; F "
      f"{kern['factor_de_intercambio'][0]['calculado_tubo_0.9']:.3f} / {kern['factor_de_intercambio'][1]['calculado_tubo_0.9']:.3f} contra 0.635 / 0.56; "
      f"emisividad {e194['emissivity']:.3f} contra {kern['emisividad_del_gas'][0]['cartas_Figs_19.12_19.13']:.3f} y {e191['emissivity']:.3f} contra 0.496")
print(f"4 Ibrahim 2010: el texto dice 1278 K, su ecuación da {root:.0f} K, este motor {ri['bridgewall_T_K']:.0f} K")
print(f"4 emisividad ×0.8 / ×1.25 → carga radiante {sens_em[0]['carga_radiante_MW']:.2f} / {sens_em[2]['carga_radiante_MW']:.2f} MW "
      f"(base {sens_em[1]['carga_radiante_MW']:.2f})", flush=True)
save()

# --------------------------------------------------------------------------------------------- 5. horno de punta a punta
rad = T.fired_heater_radiant("rating", **{**RAD, "fuel_flow_kg_s": 0.3, "excess_air_fraction": 0.15, "tube_wall_T_K": None,
                                          "process": {"bulk_T_K": 600.0, "inside_coefficient_W_m2K": 1200.0,
                                                      "tube_inside_diameter_m": 0.1023, "wall_conductivity_W_mK": 25.0,
                                                      "inside_fouling_m2K_W": 0.0005}})
BANK = {"tube_outside_diameter_m": 0.1143, "tube_inside_diameter_m": 0.1023, "tube_effective_length_m": 12.0,
        "tubes_per_row": 4, "n_rows": 12, "transverse_pitch_m": 0.2032, "longitudinal_pitch_m": 0.176, "arrangement": "staggered",
        "wall_conductivity_W_mK": 25.0, "outside_fouling_m2K_W": 0.0, "tube_emissivity": 0.85}
PROC = {"inlet_T_K": 520.0, "capacity_rate_W_K": 60e3, "inside_coefficient_W_m2K": 1200.0, "inside_fouling_m2K_W": 0.0005}
cv = T.fired_heater_convection(NG, AIR, P, 0.3, rad["bridgewall_T_K"], BANK, PROC, excess_air_fraction=0.15)
cv0 = T.fired_heater_convection(NG, AIR, P, 0.3, rad["bridgewall_T_K"], BANK, PROC, excess_air_fraction=0.15, gas_radiation=False)
ef = T.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.15, stack_T_K=cv["stack_T_K"], setting_loss_fraction=0.02,
                             setting_loss_basis="LHV", fuel_flow_kg_s=0.3, fuel_price_basis="HHV")
siz = T.fired_heater_radiant("sizing", required_radiant_duty_W=rad["radiant_duty_W"], allowable_average_flux_W_m2=32e3,
                             geometry={k: GEO[k] for k in ("tube_outside_diameter_m", "tube_effective_length_m", "tube_pitch_m")})
res["5_horno_de_ejemplo"] = {
    "aviso": "Geometría y operación ILUSTRATIVAS (MOCKS.md, ENTRADAS-ILUSTRATIVAS-M18): no describen ningún horno real.",
    "radiante": {k: rad[k] for k in ("bridgewall_T_K", "radiant_duty_W", "average_flux_W_m2_tube_outside", "tube_wall_T_K",
                                     "absorbed_fraction_of_LHV", "heat_release_LHV_W")} | {"radiation": rad["radiation"]},
    "conveccion": {k: cv[k] for k in ("stack_T_K", "duty_W", "process_outlet_T_K", "max_tube_wall_T_K", "mass_velocity_kg_m2s",
                                      "gas_radiation_share_of_duty", "warnings")} | {"filas": cv["rows"]},
    "conveccion_sin_radiacion_del_gas": {k: cv0[k] for k in ("stack_T_K", "duty_W")},
    "eficiencia_con_la_chimenea_calculada": ef["efficiency"], "combustion": {k: ef[k] for k in ("fuel", "air", "flue_gas", "adiabatic_flame")},
    "cierre_de_la_cadena": {"suma_de_secciones_W": rad["radiant_duty_W"] + rad["shield_radiant_duty_W"] + cv["duty_W"],
                            "absorbido_segun_combustion_efficiency_W": ef["firing"]["absorbed_duty_W"]},
    "dimensionamiento_con_flujo_admisible_del_usuario_32_kW_m2": siz}
print(f"5 horno: pared divisoria {rad['bridgewall_T_K']:.0f} K, radiante {rad['radiant_duty_W']/1e6:.2f} MW ({rad['average_flux_W_m2_tube_outside']/1e3:.1f} kW/m2, "
      f"pared de tubo {rad['tube_wall_T_K']:.0f} K); convección {cv['duty_W']/1e6:.2f} MW → chimenea {cv['stack_T_K']:.0f} K "
      f"(sin radiación del gas {cv0['stack_T_K']:.0f} K); eficiencia {100*ef['efficiency']['LHV']['efficiency']:.1f} % PCI / "
      f"{100*ef['efficiency']['HHV']['efficiency']:.1f} % PCS", flush=True)
save()

# --------------------------------------------------------------------------------------------- 6. enlace con utility_costs
link = ef["for_utility_costs"]
PRICE = {"fuel_per_GJ": 5.0, "fuel_basis": "HHV"}          # ILUSTRATIVO
good = TU.utility_costs("USD", prices=PRICE, fired_heat={"furnace_efficiency": link["efficiency_to_pass"]}, today="2026-11-14")
bad = TU.utility_costs("USD", prices=PRICE, fired_heat={"furnace_efficiency": link["boiler_or_furnace_efficiency"]["LHV"]},
                       today="2026-11-14")
res["6_enlace_con_utility_costs"] = {
    "precio_ilustrativo": PRICE, "regla": link["rule"], "eficiencia_a_pasar_base_HHV": link["efficiency_to_pass"],
    "calor_a_fuego_directo_USD_GJ": good["prices"]["heating_usd_GJ"],
    "con_la_base_equivocada_USD_GJ": bad["prices"]["heating_usd_GJ"], "aviso_de_la_herramienta": link["basis_mismatch_warning"],
    "nota": "utility_costs no cambia: la eficiencia sigue siendo un dato que el usuario pasa. Esta herramienta solo la calcula "
            "y dice en qué base está."}
print(f"6 utility_costs: eficiencia PCS {link['efficiency_to_pass']:.4f} → {good['prices']['heating_usd_GJ']:.3f} USD/GJ; con la base "
      f"equivocada {bad['prices']['heating_usd_GJ']:.3f} USD/GJ ({100*(bad['prices']['heating_usd_GJ']/good['prices']['heating_usd_GJ']-1):+.1f} %)")
save()
print("hecho")
