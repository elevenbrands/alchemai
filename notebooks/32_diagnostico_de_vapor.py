"""Notebook 32 — Diagnóstico de vapor (bloque M19): `steam_system_diagnostic` y `steam_diagnostic_report`.

Qué demuestra, en orden:
1. doe         Los ejemplos resueltos de cinco U.S. DOE Steam Tip Sheets (enero de 2012; PDF leídos el 2026-11-14 de
               energy.gov con curl + pdftotext), corridos por la MISMA herramienta que usará el agente:
                 #4  DOE/GO-102012-3405  afinar la combustión          29 482 MMBtu/año, 235 856 USD
                 #3  DOE/GO-102012-3412  economizador                  386 400 USD/año
                 #9  DOE/GO-102012-3406  bajar la purga                72 631 USD/año
                 #10 DOE/GO-102012-3408  recuperar el calor de purga   8 500 MMBtu/año, 68 000 USD
                 #8  DOE/GO-102012-3410  regresar condensado           121 760 USD/año
               Las hojas no dicen qué gas ni qué aire usan: metano puro y aire seco a 70 °F es NUESTRA lectura (la de las
               pruebas de M18); se mide cuánto cambia el resultado con otra. Donde la hoja dice «caldera de 80 %», se busca
               la temperatura de chimenea que da 0.80 exacto: eso reproduce su DATO, no su respuesta. Nada se ajusta.
2. measur      Una prueba unitaria de caldera de MEASUR-Tools-Suite (ORNL, rama develop, leída por la API de GitHub el
               2026-11-14): combustible de una caldera con purga. Qué son hoy las herramientas de vapor del DOE.
3. coherencia  La línea base es idéntica a llamar a mano `combustion_efficiency` y `utility_costs`; cada medida mueve el
               resultado hacia donde debe; el caso combinado ahorra menos que la suma de las medidas.
4. negativas   Qué pasa cuando falta un dato: se niega (esenciales), o degrada y lo dice (objetivos, inversión, operación).
5. planta      Un caso de punta a punta con el reporte completo. ES ILUSTRATIVO: no es una planta ni un cliente; todos sus
               datos están inventados y registrados en MOCKS.md (`ENTRADAS-ILUSTRATIVAS-M19`). La electricidad usa, nombrada
               a propósito, la referencia publicada `MX-CFE-GDMTH-GOLFO-NORTE-2026-11`, y el reporte lo marca.

Ejecutar: PYTHONPATH=. uv run python -u notebooks/32_diagnostico_de_vapor.py   (≈ 10 s, sin DWSIM ni red)
→ notebooks/32_diagnostico_de_vapor.json y notebooks/32_diagnostico_de_vapor_reporte.md
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.costing import utilities as U  # noqa: E402
from packages.dwsim_client import tools_combustion as TCB  # noqa: E402
from packages.dwsim_client import tools_steam as TS  # noqa: E402
from packages.dwsim_client import tools_utilities as TU  # noqa: E402
from packages.energy import combustion as CB  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
REPORT = Path(__file__).with_name("32_diagnostico_de_vapor_reporte.md")
TODAY = "2026-11-14"
MMBTU, LB, PSI, PATM, GAL_PER_M3 = 1.05505585262, 0.45359237, 6894.757293168, 101325.0, 264.172052
P150, P100 = (150 + 14.696) * PSI, (100 + 14.696) * PSI
res: dict = {}


def save() -> None:
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def F(t: float) -> float:
    return (t + 459.67) / 1.8


def measure(out: dict, name: str) -> dict:
    return next(m for m in out["measures"] if m["id"] == name)


def pct(mine: float, theirs: float) -> float:
    return 100.0 * (mine / theirs - 1.0)


# ------------------------------------------------------------------------------------------- 1. DOE Steam Tip Sheets
CH4 = {"type": "gas", "composition": {"CH4": 1.0}, "T_K": F(70)}
DRY = {"T_K": F(70), "humidity_ratio_kg_kg": 0.0}
USD_8 = {"price_per_GJ": 8.0 / MMBTU, "basis": "HHV"}
NO_WATER = {"price_per_m3": 0.0}       # the examples that do not charge water


def doe_comb(excess: float, stack_T: float, air: dict = DRY) -> dict:
    return {"site_pressure_Pa": PATM, "air": air, "excess_air_fraction": excess, "stack_T_K": stack_T,
            "setting_loss_fraction": 0.0, "setting_loss_basis": "HHV"}


def stack_for(eta: float, fuel: dict = CH4, air: dict = DRY) -> float:
    f = lambda T: CB.combustion_efficiency(fuel, air, PATM, excess_air_fraction=0.15, stack_T_K=T, setting_loss_fraction=0.0,  # noqa: E731
                                           setting_loss_basis="HHV", adiabatic_flame=False)["efficiency"]["HHV"]["efficiency"] - eta
    return brentq(f, 350.0, 900.0, xtol=1e-10)


def tip_4(fuel: dict = CH4, T_air_F: float = 70.0) -> dict:
    air = {"T_K": F(T_air_F), "humidity_ratio_kg_kg": 0.0}
    out = TS.steam_system_diagnostic(
        "USD", dict(fuel, T_K=F(T_air_F)), doe_comb(0.449, F(T_air_F + 400), air), {"steam_pressure_Pa": P150, "blowdown_fraction": 0.0},
        {"feedwater_heating": "not_charged", "feedwater_T_K": F(117), "makeup_water_T_K": F(60), "condensate_return_fraction": 0.0},
        USD_8, NO_WATER, operation={"hours_per_year": 8000, "fuel_consumption": {"energy_GJ_per_year": 500000 * MMBTU, "basis": "HHV"}},
        measures={"reduce_excess_air": {"target_excess_air_fraction": 0.095, "stack_T_K_after": F(T_air_F + 300)}}, today=TODAY)
    m = measure(out, "reduce_excess_air")
    return {"E1": 100 * out["baseline"]["boiler"]["efficiency_HHV"], "E2": 100 * m["result"]["boiler"]["efficiency_HHV"],
            "MMBtu": m["savings"]["annual"]["fuel_GJ"] / MMBTU, "USD": m["savings"]["annual"]["cost"]["total"],
            "verified": out["evidence"]["verified"]}


def run_doe() -> None:
    d: dict = {"lectura": "metano puro y aire seco a 70 °F (las hojas no lo dicen); precio 8 USD/MMBtu en base PCS",
               "fuentes": {"#3": "DOE/GO-102012-3412", "#4": "DOE/GO-102012-3405", "#8": "DOE/GO-102012-3410",
                           "#9": "DOE/GO-102012-3406", "#10": "DOE/GO-102012-3408",
                           "url": "https://www.energy.gov/sites/prod/files/2014/05/f16/ (steam3_recovery, steam4_boiler_efficiency, "
                                  "steam8_boiler, steam9_blowdown, steam10_boiler_blowdown).pdf", "leidas": TODAY}}
    # ---- #4
    t4 = tip_4()
    d["tip_4"] = {
        "datos": "500 000 MMBtu/año, 8 000 h; 44.9 % de exceso de aire y chimenea − aire = 400 °F → 9.5 % y 300 °F",
        "doe": {"E1": 78.2, "E2": 83.1, "MMBtu": 29482, "USD": 235856}, "motor": t4,
        "dif_pct": {"MMBtu": pct(t4["MMBtu"], 29482), "USD": pct(t4["USD"], 235856)},
        "explicacion": "La hoja lee E1 y E2 de su tabla; el motor los calcula y quedan 0.4-0.6 puntos por encima (el mismo "
                       "desfase medido en M18 contra las 25 celdas). El ahorro depende del cociente E1/E2.",
        "sensibilidad": {"gas natural 95/3/2 (CH4/C2H6/N2)": tip_4({"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}}),
                         "aire a 60 °F": tip_4(T_air_F=60.0), "aire a 80 °F": tip_4(T_air_F=80.0)}}
    print(f"#4  DOE 29 482 MMBtu / 235 856 USD   motor {t4['MMBtu']:,.0f} / {t4['USD']:,.0f}   ({d['tip_4']['dif_pct']['USD']:+.1f} %)"
          f"   E1 {t4['E1']:.2f} (78.2)  E2 {t4['E2']:.2f} (83.1)")

    # ---- #3
    out = TS.steam_system_diagnostic(
        "USD", CH4, doe_comb(0.15, F(500)), {"steam_pressure_Pa": P150, "blowdown_fraction": 0.0},
        {"feedwater_heating": "not_charged", "feedwater_T_K": F(117), "makeup_water_T_K": F(60), "condensate_return_fraction": 0.0},
        USD_8, NO_WATER, operation={"hours_per_year": 8400, "steam_generated_kg_s": 45000 * LB / 3600},
        measures={"reduce_stack_temperature": {"target_stack_T_K": F(250), "device": "economizer"}}, today=TODAY)
    b, m = out["baseline"], measure(out, "reduce_stack_temperature")
    duty = b["annual"]["steam_delivered_tonne"] * b["steam"]["boiler_duty_GJ_per_tonne"] / MMBTU / 8400
    usd = m["savings"]["annual"]["cost"]["total"]
    e0, e1 = 100 * b["boiler"]["efficiency_HHV"], 100 * m["result"]["boiler"]["efficiency_HHV"]
    recovered = (e1 - e0) / 100 * m["result"]["annual"]["fuel_GJ_HHV"] / MMBTU / 8400
    t500, t250 = 80.4 + 0.3 * (77.9 - 80.4), 85.2 + 0.2 * (85.2 - 82.8)          # tip sheet #4 table at 15 % excess air
    by_tip_4 = 50.0 / (t500 / 100) * (1 - t500 / t250) * 8400 * 8.0
    d["tip_3"] = {
        "datos": "45 000 lb/h de vapor a 150 psig, alimentación a 117 °F, chimenea 500 → 250 °F, 15 % de exceso de aire, 8 400 h",
        "doe": {"carga_MMBtu_h": 50.0, "eficiencia": 80.0, "recuperable_MMBtu_h": 4.6, "USD": 386400},
        "motor": {"carga_MMBtu_h": duty, "eficiencia_antes": e0, "eficiencia_despues": e1, "recuperado_MMBtu_h": recovered,
                  "ahorro_combustible_pct": 100 * m["savings"]["fuel_fraction"], "USD": usd, "checks": m["checks"],
                  "verified": out["evidence"]["verified"]},
        "dif_pct": {"USD": pct(usd, 386400)},
        "con_la_tabla_del_tip_4": {"eficiencia_500F": t500, "eficiencia_250F": t250, "USD": by_tip_4, "motor_vs_tabla_pct": pct(usd, by_tip_4),
                                   "nota": "tabla del Tip Sheet #4 a 15 % de exceso; 430 °F interpolado, 180 °F extrapolado 20 °F; "
                                           "aire a 70 °F es nuestra lectura"},
        "explicacion": "La diferencia es de la hoja, no del motor. Su tabla da 4.6 MMBtu/h recuperables y la hoja divide entre 0.80: "
                       "5.75 MMBtu/h de combustible, 9.2 % de los 62.5 MMBtu/h que quema la caldera. La tabla de combustión del propio "
                       "DOE (Tip Sheet #4) da para el mismo cambio 6.0 puntos de eficiencia, 7.0 % del combustible; el motor da 5.9 "
                       "puntos y 6.8 %. La hoja #3 no dice cómo se calculó su tabla."}
    print(f"#3  DOE 386 400 USD   motor {usd:,.0f} ({pct(usd, 386400):+.1f} %)   con la tabla del #4: {by_tip_4:,.0f} "
          f"(motor {pct(usd, by_tip_4):+.1f} %)   eficiencia {e0:.2f} → {e1:.2f}; carga {duty:.2f} MMBtu/h")

    # ---- #9
    T80 = stack_for(0.80)
    out = TS.steam_system_diagnostic(
        "USD", CH4, doe_comb(0.15, T80), {"steam_pressure_Pa": P150, "blowdown_fraction": 0.08},
        {"feedwater_heating": "not_charged", "feedwater_T_K": F(60), "makeup_water_T_K": F(60), "condensate_return_fraction": 0.0},
        USD_8, {"price_per_m3": 0.004 * GAL_PER_M3}, operation={"hours_per_year": 8760, "steam_generated_kg_s": 100000 * LB / 3600},
        measures={"reduce_blowdown": {"target_blowdown_fraction": 0.06}}, today=TODAY)
    s = measure(out, "reduce_blowdown")["savings"]["annual"]
    d["tip_9"] = {
        "datos": "100 000 lb/h a 150 psig, purga 8 % → 6 % de la alimentación, repuesto a 60 °F, caldera de 80 %, 8 760 h, agua 0.004 USD/gal",
        "doe": {"combustible_USD": 62913, "agua_USD": 9718, "total_USD": 72631, "repuesto_lb_h": 2313},
        "motor": {"combustible_USD": s["cost"]["fuel"], "agua_USD": s["cost"]["water"], "total_USD": s["cost"]["total"],
                  "repuesto_lb_h": s["makeup_water_m3"] * 1000 / LB / 8760, "chimenea_para_80_pct_K": T80,
                  "verified": out["evidence"]["verified"]},
        "dif_pct": {"combustible": pct(s["cost"]["fuel"], 62913), "agua": pct(s["cost"]["water"], 9718), "total": pct(s["cost"]["total"], 72631)},
        "explicacion": "Mismo volumen de control que la hoja (el agua entra a la temperatura del repuesto). La diferencia es el "
                       "redondeo de sus entalpías (338.5 y 28 Btu/lb) y de la densidad (8.34 lb/gal) frente a IAPWS-95."}
    print(f"#9  DOE 62 913 + 9 718 = 72 631 USD   motor {s['cost']['fuel']:,.0f} + {s['cost']['water']:,.0f} = {s['cost']['total']:,.0f} "
          f"({d['tip_9']['dif_pct']['total']:+.2f} %)")

    # ---- #10
    CP = U._cp()
    h_f, h_mu = CP.PropsSI("H", "P", P150, "Q", 0, "Water"), U._h_liq(F(60))
    T_drain = U._T_liq(h_mu + 0.10 * (h_f - h_mu))
    rows = {}
    for label, mode, r, T_c, T_fw in (("tanque sin calentador, sin retorno", "mixing_only", 0.0, None, None),
                                      ("desaireador a 227 °F, sin retorno", "deaerator_steam", 0.0, None, F(227)),
                                      ("desaireador a 227 °F, 50 % de retorno a 180 °F", "deaerator_steam", 0.5, F(180), F(227))):
        out = TS.steam_system_diagnostic(
            "USD", CH4, doe_comb(0.15, T80), {"steam_pressure_Pa": P150, "blowdown_fraction": 3200 / 53200},
            {"feedwater_heating": mode, "feedwater_T_K": T_fw, "makeup_water_T_K": F(60), "condensate_return_fraction": r, "condensate_T_K": T_c},
            USD_8, NO_WATER, operation={"hours_per_year": 8000, "steam_generated_kg_s": 50000 * LB / 3600},
            measures={"recover_blowdown_heat": {"drain_T_K": T_drain}}, today=TODAY)
        sv = measure(out, "recover_blowdown_heat")["savings"]["annual"]
        rows[label] = {"MMBtu": sv["fuel_GJ"] / MMBTU, "USD": sv["cost"]["total"], "verified": out["evidence"]["verified"]}
    first = rows["tanque sin calentador, sin retorno"]
    formula = 0.9 * 3200 * (338.5 - 28) / 0.80 * 8000 / 1e6
    d["tip_10"] = {
        "datos": "50 000 lb/h a 150 psig, purga 3 200 lb/h (6.0 % de la alimentación), se recupera 90 % de la energía de la purga hacia "
                 "el repuesto a 60 °F, caldera de 80 %, 8 000 h",
        "doe": {"tabla_MMBtu_h_por_100000_lb_h": 1.7, "MMBtu": 8500, "USD": 68000},
        "motor": rows, "drenaje_equivalente_a_90_pct_F": T_drain * 1.8 - 459.67,
        "su_formula_con_sus_entalpias_MMBtu": formula, "su_tabla_sin_redondear_MMBtu_h": 0.9 * 100000 * 0.06 / 0.94 * (338.5 - 28) / 1e6,
        "dif_pct": {"contra_la_hoja": pct(first["MMBtu"], 8500), "contra_su_formula": pct(first["MMBtu"], formula)},
        "explicacion": "La hoja lee 1.7 MMBtu/h de una tabla redondeada; con sus propias entalpías la celda vale 1.78. Con su fórmula "
                       "sin redondear salen 8 942 MMBtu/año, que es lo que da el motor. El 90 % se tradujo a una temperatura de "
                       "drenaje equivalente (91 °F); la hoja no dice cómo llega el agua a la caldera y el resultado cambia menos "
                       "de 0.3 % entre los tres arreglos probados."}
    print(f"#10 DOE 8 500 MMBtu / 68 000 USD   motor {first['MMBtu']:,.0f} / {first['USD']:,.0f} ({pct(first['MMBtu'], 8500):+.1f} %; "
          f"contra su fórmula sin redondear {pct(first['MMBtu'], formula):+.2f} %)")

    # ---- #8
    rows = {}
    for label, steam_lb_h, P, r0, beta in (("50 000 lb/h a 100 psig, 30 % de retorno, sin purga", 50000, P100, 0.3, 0.0),
                                           ("100 000 lb/h a 150 psig, 50 % de retorno, purga 5 %", 100000, P150, 0.5, 0.05)):
        out = TS.steam_system_diagnostic(
            "USD", CH4, doe_comb(0.15, T80), {"steam_pressure_Pa": P, "blowdown_fraction": beta},
            {"feedwater_heating": "mixing_only", "makeup_water_T_K": F(55), "condensate_return_fraction": r0, "condensate_T_K": F(180)},
            USD_8, {"price_per_m3": 0.004 * GAL_PER_M3}, operation={"hours_per_year": 8000, "steam_generated_kg_s": steam_lb_h * LB / 3600},
            measures={"increase_condensate_return": {"target_condensate_return_fraction": r0 + 8800 / steam_lb_h}}, today=TODAY)
        c = measure(out, "increase_condensate_return")["savings"]["annual"]["cost"]
        rows[label] = {"combustible_USD": c["fuel"], "agua_USD": c["water"], "total_USD": c["total"], "verified": out["evidence"]["verified"]}
    first = next(iter(rows.values()))
    d["tip_8"] = {
        "datos": "10 000 lb/h más de condensado a 180 °F con 12 % de pérdida por flasheo (8 800 lb/h netos), 8 000 h, caldera de 80 %, "
                 "repuesto a 55 °F, agua 0.004 USD/gal. La hoja no da el flujo de vapor ni la presión.",
        "doe": {"combustible_USD": 88000, "agua_USD": 33760, "total_USD": 121760}, "motor": rows,
        "dif_pct": {"combustible": pct(first["combustible_USD"], 88000), "agua": pct(first["agua_USD"], 33760),
                    "total": pct(first["total_USD"], 121760)},
        "explicacion": "La hoja usa cp = 1 Btu/lb·°F y 8.34 lb/gal; el motor IAPWS-95. El resultado no depende del flujo de vapor, de "
                       "la presión ni de la purga elegidos (dos juegos distintos dan lo mismo)."}
    print(f"#8  DOE 88 000 + 33 760 = 121 760 USD   motor {first['combustible_USD']:,.0f} + {first['agua_USD']:,.0f} = "
          f"{first['total_USD']:,.0f} ({d['tip_8']['dif_pct']['total']:+.2f} %)")
    res["doe"] = d
    save()


# ------------------------------------------------------------------------------------------------------ 2. MEASUR
def run_measur() -> None:
    T_da = U._cp().PropsSI("T", "P", 0.3631e6, "Q", 0, "Water")
    lv = U.steam_level({"name": "s", "pressure_Pa": 5.5766e6, "boiler_efficiency": 0.724, "feedwater_T_K": T_da,
                        "condensate_return_fraction": 0.0, "blowdown_fraction": 0.037, "makeup_water_T_K": 293.15,
                        "feedwater_heating": "not_charged"}, 1.0, 0.0)
    mine = lv["fuel_GJ_per_tonne"] * 1e6 * 10.864
    res["measur"] = {
        "caso": "Boiler(desaireador 0.3631 MPa, eficiencia 72.4 %, purga 3.7 %, vapor saturado a 5.5766 MPa, 10 864 kg/h)",
        "fuente": "ORNL-AMO/MEASUR-Tools-Suite, rama develop, tests/cpp/steamModeler/Boiler.unit.cpp, segundo caso (getFuelEnergy)",
        "measur_kJ_h": 33344279.4789710045, "motor_kJ_h": mine, "dif_pct": pct(mine, 33344279.4789710045),
        "explicacion": "Mismo balance de caldera (vapor + purga − alimentación) ÷ eficiencia. MEASUR usa IAPWS-IF97; este motor IAPWS-95.",
        "herramientas_del_doe": {
            "leido": TODAY,
            "MEASUR": "vigente: v1.10.0 del 2026-10-21 (github.com/ORNL-AMO/AMO-Tools-Desktop, último push 2026-11-13), web en "
                      "measur.ornl.gov y de escritorio; licencia propia de UT-Battelle «OPEN SOURCE LICENSE (Permissive)» (el "
                      "package.json dice MIT; GitHub no la reconoce como estándar). Su módulo de vapor es el sucesor de SSAT/SSMT: "
                      "modela caldera, turbinas, tanques de flasheo, desaireador y hasta tres cabezales y compara escenarios; "
                      "calculadoras sueltas de pérdida por chimenea, caldera, purga, tanque de flasheo, desaireador y economizador.",
            "SSMT_SSAT": "la página antigua del Steam System Modeler (energy.gov/node/903516) responde 404 el 2026-11-14; el "
                         "repositorio AMO-Tools-Suite está archivado y remite a MEASUR-Tools-Suite.",
            "no_verificado": "no se instaló ni se usó MEASUR: lo que hace se leyó de su README, su lista de calculadoras, la página "
                             "del DOE y su código de pruebas. En el árbol del repositorio no hay archivos de traducción.",
        }}
    print(f"MEASUR caldera: {33344279.479:,.0f} kJ/h   motor {mine:,.0f} kJ/h ({res['measur']['dif_pct']:+.4f} %)")
    save()


# ------------------------------------------------------------------------------------------- 5. illustrative plant
ILLUSTRATIVE_NOTICE = ("CASO ILUSTRATIVO. No es una planta real ni un cliente: todos los datos de operación, precios, objetivos "
                       "e inversiones de este documento son inventados para mostrar el formato (MOCKS.md, ENTRADAS-ILUSTRATIVAS-M19).")
ILLUSTRATIVE = dict(
    currency="MXN",
    fuel={"type": "gas", "composition": {"CH4": 0.92, "C2H6": 0.05, "C3H8": 0.01, "N2": 0.015, "CO2": 0.005}, "T_K": 298.15},
    combustion={"site_pressure_Pa": 95000.0, "air": {"T_K": 303.15, "relative_humidity": 0.50},
                "flue_O2": {"mole_fraction": 0.065, "basis": "dry"}, "stack_T_K": 503.15,
                "setting_loss_fraction": 0.015, "setting_loss_basis": "HHV"},
    boiler={"steam_pressure_Pa": 9.0e5, "blowdown_fraction": 0.08, "auxiliary_power_kWh_per_tonne": 5.0},
    water={"feedwater_heating": "deaerator_steam", "feedwater_T_K": 368.15, "makeup_water_T_K": 295.15,
           "condensate_return_fraction": 0.45, "condensate_T_K": 358.15},
    fuel_cost={"bill": {"amount": 10_925_000.0, "energy_GJ": 115_000.0, "basis": "HHV",
                        "period": "12 recibos mensuales (ilustrativos)"}},
    water_cost={"price_per_m3": 38.0},
    electricity_cost={"time_of_use_fractions": {"base": 0.30, "intermedia": 0.58, "punta": 0.12}, "load_factor": 0.75},
    reference="MX-CFE-GDMTH-GOLFO-NORTE-2026-11",
    operation={"hours_per_year": 6000.0, "steam_generated_kg_s": 6.0 / 3.6, "fuel_consumption": {"from_fuel_bill": True},
               "annual_basis": "steam"},
    measures={"reduce_excess_air": {"target_flue_O2": {"mole_fraction": 0.03, "basis": "dry"}, "stack_T_K_after": 493.15,
                                    "investment": 400_000.0},
              "reduce_stack_temperature": {"target_stack_T_K": 423.15, "device": "economizer", "investment": 2_800_000.0},
              "reduce_blowdown": {"target_blowdown_fraction": 0.05, "investment": 250_000.0},
              "recover_blowdown_heat": {"flash_pressure_Pa": 1.2e5, "drain_T_K": 313.15}},
    today=TODAY)


# ---------------------------------------------------------------------------------------------------- 3. coherence
def run_coherence() -> None:
    kw = copy.deepcopy(ILLUSTRATIVE)
    kw["measures"]["increase_condensate_return"] = {"target_condensate_return_fraction": 0.70}
    out = TS.steam_system_diagnostic(**kw)
    c = kw["combustion"]
    comb = TCB.combustion_efficiency(kw["fuel"], c["air"], c["site_pressure_Pa"], flue_O2=c["flue_O2"], stack_T_K=c["stack_T_K"],
                                     setting_loss_fraction=c["setting_loss_fraction"], setting_loss_basis=c["setting_loss_basis"],
                                     fuel_price_basis="HHV")
    eta = comb["for_utility_costs"]["efficiency_to_pass"]
    direct = TU.utility_costs(
        "MXN", {"fuel_per_GJ": 10_925_000.0 / 115_000.0, "fuel_basis": "HHV", "boiler_makeup_water_per_m3": 38.0},
        reference=kw["reference"], electricity=kw["electricity_cost"], hours_per_year=6000.0, today=TODAY,
        steam=[{"name": "v", "pressure_Pa": 9.0e5, "superheat_K": 0.0, "boiler_efficiency": eta, "feedwater_T_K": 368.15,
                "condensate_return_fraction": 0.45, "blowdown_fraction": 0.08, "makeup_water_T_K": 295.15,
                "feedwater_heating": "deaerator_steam", "condensate_T_K": 358.15, "auxiliary_power_kWh_per_tonne": 5.0}])["steam"][0]
    b = out["baseline"]
    same = {"eficiencia": abs(b["boiler"]["efficiency"] / eta - 1), "costo_por_tonelada": abs(b["steam"]["cost_per_tonne"] / direct["cost_per_tonne"] - 1),
            "combustible_por_tonelada": abs(b["steam"]["fuel_GJ_per_tonne"] / direct["fuel_GJ_per_tonne"] - 1),
            "agua_por_tonelada": abs(b["steam"]["makeup_m3_per_tonne"] / direct["makeup_m3_per_tonne"] - 1)}
    rows = {m["id"]: {"ahorro_MXN_año": m["savings"]["annual"]["cost"]["total"], "ahorro_GJ_año": m["savings"]["annual"]["fuel_GJ"],
                      "ahorro_agua_m3_año": m["savings"]["annual"]["makeup_water_m3"],
                      "puntos_de_eficiencia": m["savings"]["efficiency_gain_points"], "peor_residuo": max(m["residuals"].values())}
            for m in out["measures"]}
    cb = out["combined"]
    res["coherencia"] = {
        "linea_base_vs_llamadas_directas_dif_relativa": same,
        "medidas": rows, "todas_ahorran": all(r["ahorro_MXN_año"] > 0 and r["ahorro_GJ_año"] > 0 for r in rows.values()),
        "combinado_MXN_año": cb["savings"]["annual"]["cost"]["total"], "suma_individuales_MXN_año": cb["sum_of_individual"]["annual"]["cost_total"],
        "combinado_sobre_suma": cb["combined_over_sum"], "evidence": {k: v for k, v in out["evidence"].items() if k != "meaning"}}
    assert max(same.values()) < 1e-12 and res["coherencia"]["todas_ahorran"] and cb["combined_over_sum"] < 1 and out["evidence"]["verified"]
    print(f"Coherencia: línea base = llamadas directas (dif. máx. {max(same.values()):.1e}); 5 de 5 medidas ahorran; combinado "
          f"{cb['savings']['annual']['cost']['total']:,.0f} < suma {cb['sum_of_individual']['annual']['cost_total']:,.0f} MXN/año "
          f"({100 * cb['combined_over_sum']:.0f} %); peor residuo {out['evidence']['worst_residual_fraction']:.1e}")
    save()


# ---------------------------------------------------------------------------------------------------- 4. negatives
def run_negatives() -> None:
    neg: dict = {}

    def refuse(label: str, **changes) -> None:
        kw = copy.deepcopy(ILLUSTRATIVE)
        for k, v in changes.items():
            kw[k] = {**kw[k], **v} if isinstance(v, dict) and isinstance(kw.get(k), dict) else v
        try:
            TS.steam_system_diagnostic(**kw)
            neg[label] = "NO SE NEGÓ"
        except ValueError as e:
            neg[label] = "se niega: " + str(e)[:260]

    refuse("sin precio de combustible (la referencia mexicana no trae gas)", fuel_cost=None)
    refuse("sin precio del agua", water_cost=None)
    refuse("sin O2 ni exceso de aire", combustion={"flue_O2": None})
    refuse("sin temperatura de chimenea", combustion={"stack_T_K": None})
    refuse("sin pérdida por paredes", combustion={"setting_loss_fraction": None})
    refuse("sin purga", boiler={"blowdown_fraction": None})
    refuse("sin reparto horario para la tarifa de CFE", electricity_cost={"time_of_use_fractions": None})
    kw = copy.deepcopy(ILLUSTRATIVE)
    kw["operation"], kw["measures"] = None, {"reduce_blowdown": {"target_blowdown_fraction": 0.05}}
    kw["electricity_cost"] = {"time_of_use_fractions": kw["electricity_cost"]["time_of_use_fractions"]}   # sin horas no hay cómo repartir el cargo por demanda
    out = TS.steam_system_diagnostic(**kw)
    neg["sin horas ni producción"] = {"cifras_anuales": out["baseline"]["annual"], "costo_por_tonelada": out["baseline"]["steam"]["cost_per_tonne"],
                                      "ahorro_por_tonelada": measure(out, "reduce_blowdown")["savings"]["per_tonne"]["cost"]}
    neg["medidas sin objetivo"] = {m["id"]: m["reason"] for m in out["measures"] if m["status"] != "evaluated"}
    neg["retorno sin inversión"] = measure(out, "reduce_blowdown")["payback_note"]
    neg["datos faltantes listados"] = [f"{x['item']} → impide {x['prevents']}" for x in out["missing_data"]]
    dew = TS.steam_system_diagnostic(**{**copy.deepcopy(ILLUSTRATIVE), "measures": None})["baseline"]["boiler"]["water_dew_point_K"]
    kw = copy.deepcopy(ILLUSTRATIVE)
    kw["measures"] = {"reduce_stack_temperature": {"target_stack_T_K": dew - 2.0}}
    neg["chimenea bajo el punto de rocío del agua"] = {"punto_de_rocio_K": dew,
                                                       "resultado": measure(TS.steam_system_diagnostic(**kw), "reduce_stack_temperature")["reason"]}
    assert not any(v == "NO SE NEGÓ" for v in neg.values())
    res["negativas"] = neg
    print(f"Negativas: {sum(isinstance(v, str) and v.startswith('se niega') for v in neg.values())} de 7 datos esenciales faltantes se "
          f"niegan; sin operación → solo cifras por tonelada; {len(neg['medidas sin objetivo'])} medidas «no evaluada: falta el dato»")
    save()


# -------------------------------------------------------------------------------------------- 5. illustrative plant
def run_plant() -> None:
    rep = TS.steam_diagnostic_report(inputs=ILLUSTRATIVE, plant_label="planta ilustrativa (no es un cliente)", notice=ILLUSTRATIVE_NOTICE)
    d = rep["diagnostic"]
    REPORT.write_text(rep["markdown"])
    b, cb = d["baseline"], d["combined"]
    res["planta_ilustrativa"] = {
        "AVISO": ILLUSTRATIVE_NOTICE, "entradas": ILLUSTRATIVE, "titulares": rep["headline"],
        "linea_base": {"eficiencia_PCS": b["boiler"]["efficiency"], "costo_MXN_por_tonelada": b["steam"]["cost_per_tonne"],
                       "costo_MXN_por_GJ": b["steam"]["cost_per_GJ_delivered"], "desglose_MXN_por_tonelada": b["steam"]["breakdown_per_tonne"],
                       "anual": b["annual"], "perdidas": b["boiler"]["losses_fraction_of_heating_value"],
                       "punto_de_rocio_agua_K": b["boiler"]["water_dew_point_K"]},
        "precios": d["prices"],
        "medidas": [{"id": m["id"], "estado": m["status"], "razon": m["reason"],
                     **({"ahorro_MXN_año": m["savings"]["annual"]["cost"]["total"], "ahorro_GJ_año": m["savings"]["annual"]["fuel_GJ"],
                         "eficiencia_nueva": m["result"]["boiler"]["efficiency"], "costo_nuevo_MXN_t": m["result"]["steam"]["cost_per_tonne"],
                         "inversion_MXN": m["investment"], "retorno_años": m["simple_payback_years"], "nota_retorno": m["payback_note"]}
                        if m["status"] == "evaluated" else {})} for m in d["measures"]],
        "combinado": {"ahorro_MXN_año": cb["savings"]["annual"]["cost"]["total"], "ahorro_GJ_año": cb["savings"]["annual"]["fuel_GJ"],
                      "suma_individuales_MXN_año": cb["sum_of_individual"]["annual"]["cost_total"], "combinado_sobre_suma": cb["combined_over_sum"],
                      "eficiencia": cb["result"]["boiler"]["efficiency"]},
        "cruce_entradas_salidas": d["evidence"]["input_output_cross_check"], "datos_faltantes": d["missing_data"],
        "referencia_publicada": d["provenance"]["published_references"], "primer_aviso": d["warnings"][0],
        "evidence": {k: v for k, v in d["evidence"].items() if k not in ("meaning", "input_output_cross_check")},
        "reporte": {"archivo": REPORT.name, "secciones": rep["sections"], "caracteres": len(rep["markdown"]),
                    "marca_la_referencia": rep["uses_published_reference_price"], "verified": rep["verified"]}}
    assert rep["verified"] and rep["uses_published_reference_price"] and d["warnings"][0].startswith("PRECIO DE REFERENCIA PUBLICADO")
    print(f"Planta ILUSTRATIVA: vapor {b['annual']['cost']['total']:,.0f} MXN/año ({b['steam']['cost_per_tonne']:,.1f} MXN/t); eficiencia "
          f"{100 * b['boiler']['efficiency']:.1f} % (PCS); combinado {cb['savings']['annual']['cost']['total']:,.0f} MXN/año "
          f"({100 * cb['savings']['cost_fraction']:.1f} %) contra suma {cb['sum_of_individual']['annual']['cost_total']:,.0f}; "
          f"reporte de {len(rep['markdown']):,} caracteres → {REPORT.name}")
    save()


if __name__ == "__main__":
    run_doe()
    run_measur()
    run_coherence()
    run_negatives()
    run_plant()
    print(f"→ {OUT.name}")
