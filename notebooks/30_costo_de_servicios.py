"""Notebook 30 — Costo de servicios por región (bloque M17): `utility_costs` y `list_utility_price_references`.

Qué demuestra, en orden:
1. doe        Ruta física del vapor contra la hoja «Benchmark the Fuel Cost of Steam Generation» (U.S. DOE, Steam Tip
              Sheet #15, DOE/GO-102012-3391, enero 2012): los 15 valores de su Tabla 1 y su ejemplo ($9.39 por 1 000 lb).
2. uv         Segunda ruta (Ulrich y Vasudevan, Chemical Engineering, abril 2006, pp. 66-69, Tabla 1) contra los cuatro
              resultados de sus dos ejemplos resueltos: electricidad, agua de enfriamiento, vapor de 32 barg y refrigerante a −5 °C.
3. coherencia Comprobaciones independientes: el balance de la caldera cierra, el vapor nunca cuesta menos que
              combustible ÷ eficiencia, la refrigeración sube al bajar la temperatura, ida y vuelta de unidades.
4. planta     Una planta de ejemplo con SUS precios (entradas ILUSTRATIVAS, ver MOCKS.md) y la misma planta con precios de
              referencia publicados, nombrados explícitamente, por región. Sin precios ni referencia la herramienta se niega.
5. cadena     utility_costs → annual_cost sobre un caso existente: la recuperación de calor del bloque 20
              (`packages/costing/heat_recovery.py`), antes con precios ilustrativos fijos, ahora con los precios calculados.

Las entradas de ingeniería de la planta de ejemplo (eficiencias, retorno de condensado, ciclos de concentración, perfil de
carga, precios del agua) son ILUSTRATIVAS: sirven para mostrar el cálculo, no describen ninguna planta real.
Ejecutar: PYTHONPATH=. uv run python -u notebooks/30_costo_de_servicios.py   (≈ 2 min; la parte 5 usa DWSIM) → .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CoolProp.CoolProp as CP  # noqa: E402

from packages.costing import utilities as U  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
TODAY = "2026-11-14"                       # fecha fija para que la antigüedad de las referencias sea reproducible
BTU_LB, LB, MMBTU_GJ = 2326.0, 0.45359237, 1.05505585262
res: dict = {"fecha_de_referencia": TODAY}


def save() -> None:
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0


def K(F: float) -> float:
    return (F - 32.0) / 1.8 + 273.15


def psig(p: float) -> float:
    return (p + 14.696) * 6894.757


# ------------------------------------------------------------------------------------------- 1. hoja del DOE (vapor)
def doe_level(p: float, T_F: float, eff: float) -> dict:
    return dict(name="doe", pressure_Pa=psig(p), boiler_efficiency=eff, feedwater_T_K=K(T_F), condensate_return_fraction=1.0,
                blowdown_fraction=0.0, makeup_water_T_K=288.15, feedwater_heating="not_charged")


TABLE_1 = {150: [1178, 1128, 1078, 1028, 977], 450: [1187, 1137, 1087, 1037, 986], 600: [1184, 1134, 1084, 1034, 984]}
rows, worst = [], 0.0
for p, row in TABLE_1.items():
    for T_F, ref in zip((50, 100, 150, 200, 250), row):
        ours = U.steam_level(doe_level(p, T_F, 1.0), 1.0, 0.0)["fuel_GJ_per_tonne"] * 1e6 / BTU_LB
        rows.append({"psig": p, "agua_F": T_F, "publicado_Btu_lb": ref, "nuestro_Btu_lb": ours, "dif_pct": pct(ours, ref)})
        worst = max(worst, abs(pct(ours, ref)))
ex = U.steam_level(doe_level(450, 230, 0.857), 8.0 / MMBTU_GJ, 0.0)
res["1_doe_tip_sheet_15"] = {
    "fuente": "U.S. DOE, Advanced Manufacturing Office, Steam Tip Sheet #15 «Benchmark the Fuel Cost of Steam Generation», "
              "DOE/GO-102012-3391, enero 2012. https://www.energy.gov/sites/prod/files/2014/05/f16/steam15_benchmark.pdf",
    "tabla_1": rows, "tabla_1_peor_dif_pct": worst,
    "ejemplo": {"datos": "gas natural 8.00 USD/MMBtu, vapor saturado a 450 psig, agua de alimentación a 230 °F, eficiencia 85.7 %",
                "publicado_Btu_lb": 1006, "nuestro_Btu_lb": ex["fuel_GJ_per_tonne"] * 0.857 * 1e6 / BTU_LB,
                "publicado_usd_1000lb": 9.39, "nuestro_usd_1000lb": ex["cost_per_tonne"] * LB,
                "dif_pct": pct(ex["cost_per_tonne"] * LB, 9.39), "nuestro_usd_tonelada": ex["cost_per_tonne"],
                "nuestro_usd_GJ_entregado": ex["cost_per_GJ_delivered"]}}
print(f"1 DOE Tabla 1: 15 valores, peor diferencia {worst:.2f} %")
e = res["1_doe_tip_sheet_15"]["ejemplo"]
print(f"1 DOE ejemplo: publicado 9.39 USD/1000 lb, nuestro {e['nuestro_usd_1000lb']:.3f} ({e['dif_pct']:+.2f} %); "
      f"{e['nuestro_Btu_lb']:.1f} Btu/lb contra 1006")
save()

# ---------------------------------------------------------------------------------- 2. Ulrich y Vasudevan (ruta 2)
uv_cases = [
    ("electricidad comprada, USD/kWh", 0.106, U.ulrich_vasudevan("electricity", 470, 4.5)),
    ("agua de enfriamiento, USD/m3 (planta nueva, q = 10 m3/s)", 0.048,
     U.ulrich_vasudevan("cooling_water", 470, 4.5, "grass_roots", capacity_m3_s=10)),
    ("vapor de 32 barg, USD/kg (planta nueva, 40 kg/s)", 0.019,
     U.ulrich_vasudevan("steam", 470, 4.5, "grass_roots", pressure_barg=32, capacity_kg_s=40)),
    ("refrigerante a −5 °C (268 K), USD/kJ (planta nueva, 40 kJ/s)", 4.0e-6,
     U.ulrich_vasudevan("refrigeration", 470, 4.5, "grass_roots", T_K=268, capacity_kW=40)),
]
res["2_ulrich_vasudevan_2006"] = {
    "fuente": U.UV_SOURCE + ". Ejemplos «Utility cost estimation example» (p. 67) y «Refrigerant and wastewater treatment example» "
              "(p. 68): CE PCI 470 y combustible a 4.50 USD/GJ.",
    "casos": [{"caso": name, "publicado": pub, "nuestro": r["price"], "dif_pct": pct(r["price"], pub), "a": r["a"], "b": r["b"]}
              for name, pub, r in uv_cases],
    "nota": "Los valores publicados están redondeados a 2-3 cifras; la diferencia es de redondeo."}
for c in res["2_ulrich_vasudevan_2006"]["casos"]:
    print(f"2 U&V {c['caso']}: publicado {c['publicado']:g}, nuestro {c['nuestro']:.5g} ({c['dif_pct']:+.2f} %)")
save()

# ------------------------------------------------------------------------------------------------- 3. coherencia
# Planta de ejemplo: ENTRADAS ILUSTRATIVAS (MOCKS.md, ENTRADAS-ILUSTRATIVAS-M17)
BOILER = dict(boiler_efficiency=0.82, feedwater_T_K=378.15, condensate_return_fraction=0.70, condensate_T_K=363.15,
              blowdown_fraction=0.04, makeup_water_T_K=293.15, feedwater_heating="deaerator_steam")
STEAM = [dict(BOILER, name="baja (4 bar)", pressure_Pa=4e5), dict(BOILER, name="media (11 bar)", pressure_Pa=11e5),
         dict(BOILER, name="alta (42 bar)", pressure_Pa=42e5)]
CW = dict(supply_T_K=303.15, return_T_K=313.15, pump_head_m=30.0, pump_efficiency=0.70, evaporation_latent_fraction=0.85,
          drift_fraction=0.0005, cycles_of_concentration=5.0, fan_power_kWh_per_m3=0.04)
REFRIG = [dict(name=f"refrigeración {t} °C", evaporator_T_K=273.15 + t, condenser_T_K=313.15, second_law_efficiency=0.60,
               heat_rejection="cooling_water") for t in (5, -20, -40, -70)]
WATER = {"boiler_makeup_water_per_m3": 1.5, "cooling_makeup_water_per_m3": 0.6}
MINE = dict(WATER, fuel_per_GJ=5.0, fuel_basis="HHV", electricity_per_kWh=0.08)

mp = U.steam_level(STEAM[1], 5.0, 1.5)
m = mp["mass_per_kg_delivered"]
h = lambda T: CP.PropsSI("H", "T", T, "Q", 0, "Water")  # noqa: E731
h_s, h_f = CP.PropsSI("H", "P", 11e5, "Q", 1, "Water"), CP.PropsSI("H", "P", 11e5, "Q", 0, "Water")
to_steam = mp["fuel_GJ_per_tonne"] * 1e6 * 0.82
island_in = m["condensate_return"] * h(363.15) + m["makeup"] * h(293.15) + to_steam
island_out = h_s + m["blowdown"] * h_f
ladder = [(lv["name"], U.refrigeration_level(dict(lv, heat_rejection=None), 0.08)) for lv in REFRIG]
floors = []
for P in (3e5, 11e5, 42e5, 100e5):
    for mode in ("not_charged", "deaerator_steam"):
        r = U.steam_level(dict(BOILER, name="x", pressure_Pa=P, feedwater_heating=mode), 5.0, 0.0)
        floors.append({"P_bar": P / 1e5, "modo": mode, "combustible_usd_GJ_entregado": r["breakdown_per_GJ_delivered"]["fuel"],
                       "piso_combustible_entre_eficiencia": 5.0 / 0.82,
                       "cumple": r["breakdown_per_GJ_delivered"]["fuel"] >= 5.0 / 0.82})
cw0 = U.cooling_water(CW, 0.08, 0.6)
res["3_coherencia"] = {
    "balance_isla_de_vapor": {"entra_J_kg": island_in, "sale_J_kg": island_out, "residuo_relativo": (island_in - island_out) / island_out,
                              "nota": "recalculado aquí con entalpías de CoolProp, no con las del motor",
                              "masas_por_kg_entregado": m, "residuos_del_motor": mp["energy_balance"]},
    "vapor_no_baja_del_piso": floors,
    "refrigeracion_sube_al_enfriar": [{"nivel": n, "T_K": r["evaporator_T_K"], "COP": r["cop"],
                                       "usd_GJ": r["cost_per_GJ_removed"]} for n, r in ladder],
    "agua_de_enfriamiento": {"repuesto = evaporación + arrastre + purga": [cw0["water_kg_per_m3_circulated"]["makeup"],
                                                                         sum(cw0["water_kg_per_m3_circulated"][k] for k in ("evaporation", "drift", "blowdown"))],
                             "ciclos = repuesto / (purga + arrastre)": cw0["water_kg_per_m3_circulated"]["makeup"] /
                             (cw0["water_kg_per_m3_circulated"]["blowdown"] + cw0["water_kg_per_m3_circulated"]["drift"]),
                             "usd_m3 × m3_GJ = usd_GJ": [cw0["cost_per_m3"] * cw0["m3_per_GJ_removed"], cw0["cost_per_GJ_removed"]]},
    "unidades": {"8 USD/MMBtu → USD/GJ → USD/MMBtu": (8.0 / MMBTU_GJ) * MMBTU_GJ, "kWh por GJ": U.KWH_PER_GJ}}
costs = [x["usd_GJ"] for x in res["3_coherencia"]["refrigeracion_sube_al_enfriar"]]
assert costs == sorted(costs) and all(f["cumple"] for f in floors) and abs(res["3_coherencia"]["balance_isla_de_vapor"]["residuo_relativo"]) < 1e-9
print(f"3 balance de la isla de vapor: residuo {res['3_coherencia']['balance_isla_de_vapor']['residuo_relativo']:.1e}; "
      f"piso combustible/eficiencia cumplido en {len(floors)} casos; refrigeración {[round(c, 2) for c in costs]} USD/GJ")
save()


# ------------------------------------------------------------------------------- 4. la planta por región
def brief(o: dict) -> dict:
    return {"moneda": o["currency"],
            "vapor": {s["name"]: {"por_tonelada": s["cost_per_tonne"], "por_GJ": s["cost_per_GJ_delivered"],
                                  "desglose_por_tonelada": s["breakdown_per_tonne"]} for s in o.get("steam", [])},
            "agua_de_enfriamiento": {"por_1000_m3": o["cooling_water"]["cost_per_1000_m3"], "por_GJ": o["cooling_water"]["cost_per_GJ_removed"],
                                     "desglose_por_m3": o["cooling_water"]["breakdown_per_m3"]} if "cooling_water" in o else None,
            "refrigeracion_por_GJ": {r["name"]: r["cost_per_GJ_removed"] for r in o.get("refrigeration", [])},
            "electricidad_por_kWh": o["electricity"]["per_kWh"] if "electricity" in o else None,
            "origen_de_precios": {k: {"valor": v["value"], "origen": v["origin"], **({"periodo": v["period"], "antiguedad_dias": v["age_days"]}
                                                                                     if "period" in v else {})}
                                  for k, v in o["price_sources"].items()},
            "primer_aviso": o["warnings"][0] if o["warnings"] else None, "prices_para_annual_cost": o["prices"]}


ARGS = dict(steam=STEAM, cooling_water_spec=CW, refrigeration=REFRIG, electricity={}, today=TODAY,
            annual_cost_levels={"heating": "media (11 bar)", "cooling": "cooling_water"})
plant: dict = {"entradas_ilustrativas": {"caldera": BOILER, "agua_de_enfriamiento": CW, "refrigeracion": REFRIG,
                                         "precios_del_usuario": MINE}}
try:
    U.utility_costs("USD", **ARGS)
except ValueError as err:
    plant["sin_precios_ni_referencia"] = str(err)
    print("4 sin precios ni referencia →", str(err)[:150], "…")
full = U.utility_costs("USD", prices=MINE, **ARGS)
plant["precios_del_usuario_USD"] = brief(full)
plant["precios_del_usuario_USD"]["supuestos"] = full["assumptions"]
plant["precios_del_usuario_USD"]["no_incluido"] = full["not_included"]
for rid in ("US-EIA-2026-08", "US-TX-EIA-2026-08", "US-LA-EIA-2026-08"):
    plant[f"referencia_{rid}"] = brief(U.utility_costs("USD", prices=WATER, reference=rid, **ARGS))
for rid in ("EU27-EUROSTAT-2026-S1", "DE-EUROSTAT-2026-S1", "ES-EUROSTAT-2026-S1", "FR-EUROSTAT-2026-S1", "NL-EUROSTAT-2026-S1"):
    # agua en EUR: el mismo número ilustrativo, ahora en euros; sin tipo de cambio no hay dict prices en USD (a propósito)
    plant[f"referencia_{rid}"] = brief(U.utility_costs("EUR", prices=WATER, reference=rid, **ARGS))
TOU = {"base": 0.30, "intermedia": 0.55, "punta": 0.15}                      # perfil de carga ILUSTRATIVO
for rid in ("MX-CFE-GDMTH-ORIENTE-2026-11", "MX-CFE-GDMTH-GOLFO-NORTE-2026-11"):
    o = U.utility_costs("MXN", prices={"cooling_makeup_water_per_m3": 12.0}, reference=rid, cooling_water_spec=CW,
                        refrigeration=REFRIG, electricity={"time_of_use_fractions": TOU, "load_factor": 0.8},
                        hours_per_year=8000, today=TODAY, annual_cost_levels={"cooling": "cooling_water"})
    b = brief(o)
    b["electricidad"] = {k: o["electricity"][k] for k in ("energy_per_kWh", "demand_adder_per_kWh", "per_kWh")}
    b["nota"] = "Sin vapor: el banco no tiene gas natural de México (IPGN detrás de un reCAPTCHA); el combustible lo da el usuario."
    plant[f"referencia_{rid}"] = b
uv = {"cepci": 800.0, "basis": "grass_roots", "boiler_fuel_price_usd_GJ": 5.0, "power_fuel_price_usd_GJ": 5.0,
      "steam_capacity_kg_s": 20.0, "cooling_water_capacity_m3_s": 5.0, "refrigeration_capacity_kW": 500.0}
two = U.utility_costs("USD", prices=MINE, ulrich_vasudevan_spec=uv, **ARGS)["second_route"]
plant["contraste_ruta_2_ulrich_vasudevan"] = {
    "entradas_ilustrativas": uv, "nota": two["note"],
    "vapor_usd_tonelada": {s["name"]: {"ruta_1_costo_de_operacion": next(x["cost_per_tonne"] for x in full["steam"] if x["name"] == s["name"]),
                                       "ruta_2_precio_con_capital": s["price_per_tonne"]} for s in two["steam"]},
    "agua_usd_m3": {"ruta_1": full["cooling_water"]["cost_per_m3"], "ruta_2": two["cooling_water"]["price"]},
    "refrigeracion_usd_GJ": {r["name"]: {"ruta_1": next(x["cost_per_GJ_removed"] for x in full["refrigeration"] if x["name"] == r["name"]),
                                         "ruta_2": r["price_per_GJ_removed"]} for r in two["refrigeration"]},
    "omitidos": two["skipped"]}
res["4_planta_por_region"] = plant
for k, v in plant.items():
    if k.startswith(("precios", "referencia")):
        st = v["vapor"].get("media (11 bar)")
        print(f"4 {k} [{v['moneda']}]: vapor media {st['por_tonelada']:.2f}/t, {st['por_GJ']:.2f}/GJ; " if st else f"4 {k} [{v['moneda']}]: ",
              f"agua {v['agua_de_enfriamiento']['por_1000_m3']:.1f}/1000 m3, {v['agua_de_enfriamiento']['por_GJ']:.3f}/GJ; "
              f"refrig −20 °C {v['refrigeracion_por_GJ']['refrigeración -20 °C']:.2f}/GJ; electricidad {v['electricidad_por_kWh']:.4f}/kWh")
print("4 primer aviso con referencia:", plant["referencia_US-EIA-2026-08"]["primer_aviso"][:230], "…")
save()

# ---------------------------------------------------------------------------- 5. cadena utility_costs → annual_cost
# Caso existente del bloque 20: recuperación de calor (vapor a 450 K, agua de enfriamiento 290 → 300 K). Antes usaba
# ILLUSTRATIVE_PRICES fijos; aquí los precios salen de utility_costs para los niveles que el caso realmente usa.
from packages.costing import heat_recovery as HR  # noqa: E402

P_450 = CP.PropsSI("P", "T", HR.STEAM_K, "Q", 1, "Water")
chain_args = dict(steam=[dict(BOILER, name="vapor 450 K", pressure_Pa=P_450)],
                  cooling_water_spec=dict(CW, supply_T_K=HR.CW_IN, return_T_K=HR.CW_OUT), electricity={}, today=TODAY)
chain: dict = {"caso": "packages/costing/heat_recovery.py (bloque 20): intercambiador alimentación/producto, calentador de vapor y "
                       "enfriador de agua; costo total anual contra el área recuperadora",
               "precios_ilustrativos_del_bloque_20": HR.ILLUSTRATIVE_PRICES}
t0 = time.time()
for label, kw in (("precios_del_usuario", dict(prices=MINE)), ("referencia_US-EIA-2026-08", dict(prices=WATER, reference="US-EIA-2026-08")),
                  ("referencia_US-TX-EIA-2026-08", dict(prices=WATER, reference="US-TX-EIA-2026-08"))):
    u = U.utility_costs("USD", **kw, **chain_args)
    opt = HR.optimize(prices=u["prices"])
    b = opt["best"]
    chain[label] = {"prices": u["prices"], "primer_aviso": u["warnings"][0] if u["warnings"] else None,
                    "optimo": {k: b[k] for k in ("area_E1_m2", "Q_recovered_kW", "Q_steam_kW", "Q_cooling_kW", "approach_K", "TAC_usd_y",
                                                 "capital_usd_y", "operating_usd_y")},
                    "comprobacion_costo_de_operacion_usd_y": (b["Q_steam_kW"] * u["prices"]["heating_usd_GJ"]
                                                              + b["Q_cooling_kW"] * u["prices"]["cooling_usd_GJ"]) * 8000 * 3.6e-3}
    print(f"5 {label}: prices {json.dumps({k: round(v, 4) for k, v in u['prices'].items()})} → área óptima {b['area_E1_m2']:.0f} m2, "
          f"TAC {b['TAC_usd_y']:.0f} USD/año (operación {b['operating_usd_y']:.0f})", flush=True)
ill = HR.optimize()
chain["precios_ilustrativos_del_bloque_20"] = {"prices": HR.ILLUSTRATIVE_PRICES,
                                               "optimo": {k: ill["best"][k] for k in ("area_E1_m2", "TAC_usd_y", "operating_usd_y")}}
chain["segundos"] = time.time() - t0
res["5_cadena_utility_costs_annual_cost"] = chain
print(f"5 ilustrativos del bloque 20: área óptima {ill['best']['area_E1_m2']:.0f} m2, TAC {ill['best']['TAC_usd_y']:.0f} USD/año "
      f"({chain['segundos']:.0f} s)")
save()
print("hecho")
