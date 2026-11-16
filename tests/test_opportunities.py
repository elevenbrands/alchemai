"""Opportunity scanner (bloque M27_O): `scan_opportunities` and `opportunities_from_plant_data`.

Published worked examples reproduced end to end through the scanner (all opened on 2026-11-16 with curl + pdftotext):
  * NRCan CanmetENERGY, «Pinch Analysis: For the Efficient Use of Energy, Water & Hydrogen» (2003), Table 1 and Fig. 3.5:
    four streams, ΔTmin 10 °C -> Q_Hmin 900 kW, Q_Cmin 750 kW.
    https://natural-resources.canada.ca/sites/nrcan/files/canmetenergy/pdf/fichier.php/codectec/En/2009-052/2009-052_PM-FAC_404-DEPLOI_e.pdf
  * U.S. DOE AMO, Bandwidth Study … U.S. Petroleum Refining (2015, data 2010), Table Aa (CT/SOA Btu/bbl, throughput) and
    Table 5-4 (CT − SOA savings, TBtu/yr, rounded) for the nine processes.
    https://www.energy.gov/sites/default/files/2015/08/f26/petroleum_refining_bandwidth_report.pdf
  * U.S. DOE Steam Tip Sheet #4 «Improve Your Boiler's Combustion Efficiency», DOE/GO-102012-3405, January 2012: 500 000
    MMBtu/yr, 8 000 h/yr, 44.9 % excess air and flue − air 400 °F -> 9.5 % and 300 °F; E1 78.2 %, E2 83.1 %; 29 482 MMBtu/yr;
    235 856 USD at 8 USD/MMBtu. https://www.energy.gov/sites/prod/files/2014/05/f16/steam4_boiler_efficiency.pdf
Benchmarks of the other rules (ENERGY STAR refinery guide LBNL-56183, 10 CFR 431 pump standard, Stratiev 2025 FCC review,
DOE chemicals bandwidth) are only checked for transcription and for the arithmetic of the rule (consistency).
No network: prices are given, except the tests that check the bank fallback (local JSON)."""
from __future__ import annotations

import copy
import json
import math

import pytest

from packages.dwsim_client import tools_opportunities as TO
from packages.energy import combustion as CB
from packages.opportunities import scan as S

K = 273.15
TODAY = "2026-11-16"
MMBTU = 1.05505585262
PSI = 6894.757293168
PATM = 101325.0
P150 = (150 + 14.696) * PSI
PRICES = {"fuel_usd_GJ": 4.0, "electricity_usd_kWh": 0.10}


def F(t: float) -> float:
    return (t + 459.67) / 1.8


def dumps(r: dict) -> str:
    return json.dumps(r, default=float, ensure_ascii=False)


def opp(r: dict, oid: str) -> dict:
    return next(o for o in r["opportunities"] if o["id"] == oid)


def model(units: list[dict], kind: str = "described") -> dict:
    return {"name": "prueba", "kind": kind, "units": units, "streams": [], "sections": [], "source": "prueba"}


NRCAN = [dict(name="1", T_supply_K=200 + K, T_target_K=100 + K, CP_W_K=20e3),
         dict(name="2", T_supply_K=150 + K, T_target_K=60 + K, CP_W_K=40e3),
         dict(name="3", T_supply_K=80 + K, T_target_K=120 + K, CP_W_K=80e3),
         dict(name="4", T_supply_K=50 + K, T_target_K=220 + K, CP_W_K=15e3)]


# ------------------------------------------------------------------------------------------- published worked examples
def test_nrcan_pinch_example_end_to_end():
    """NRCan Table 1, ΔTmin 10 °C: Q_Hmin 900 kW, Q_Cmin 750 kW (exact). With no exchangers today the hot utility is the
    whole cold demand, 3200 + 2550 = 5750 kW, so 4850 kW are recoverable; valued at the given fuel price and hours."""
    pd = {"hours_per_year": 8000, "heat_integration": {"streams": NRCAN, "dT_min_K": 10.0, "current_hot_utility_W": 5750e3,
                                                         "current_cold_utility_W": 4650e3}}
    r = TO.scan_opportunities(model([{"id": "H1", "type": "fired_heater"}]), PRICES, "MX", pd, TODAY)
    o = opp(r, "a_heat_integration:pinch")
    ev = o["evidence"]
    assert ev["Q_hot_min_W"][1] == pytest.approx(900e3, rel=1e-12) and ev["Q_cold_min_W"][1] == pytest.approx(750e3, rel=1e-12)
    assert ev["verified"]
    GJ = 4850e3 * 8000 * 3600 / 1e9
    assert o["value_low_usd_y"] == pytest.approx(GJ * 4.0, rel=1e-12)                       # η = 1: the floor
    assert o["value_usd_y"] == pytest.approx(GJ / 0.90 * 4.0, rel=1e-12)                    # ENERGY STAR top of range
    assert o["value_high_usd_y"] == pytest.approx(GJ / 0.75 * 4.0, rel=1e-12)
    assert "H1" in ev["overlaps_with_hot_utility_units"]
    dumps(r)


BW = S.load_benchmarks()["entries"]["refining_bandwidth_2015"]["processes"]


@pytest.mark.parametrize("proc", sorted(BW))
def test_doe_refining_bandwidth_soa_savings_table_5_4(proc):
    """A unit running exactly at the U.S. typical intensity (CT) with the U.S. 2010 throughput of the process, measured, must
    give CT − SOA × throughput = Table 5-4 savings (rounded to 1 TBtu in the report), and nothing above CT."""
    b = BW[proc]
    bpd = b["throughput_2010_million_bbl"] * 1e6 / 365.0
    E_MW = b["CT"] * S.BTU_J * bpd / 86400 / 1e6
    rate_var = "production" if b["throughput_basis"] == "production" else "carga"
    table = [{"unit_id": "U", "variable": rate_var, "value": bpd, "uom": "bpd", "period": "2010"},
             {"unit_id": "U", "variable": "energía", "value": E_MW, "uom": "MW", "period": "2010"},
             {"unit_id": "site", "variable": "horas al año", "value": 8760, "uom": "h/y"}]
    r = TO.opportunities_from_plant_data(table, [{"id": "U", "type": "column", "label_es": b["label_es"],
                                                  "params": {"bandwidth_process": proc}}], PRICES, today=TODAY)
    o = opp(r, "d_energy_intensity:U")
    TBtu = o["value_usd_y"] / 4.0 * 1e9 / S.BTU_J / 1e12
    assert TBtu == pytest.approx((b["CT"] - b["SOA"]) * b["throughput_2010_million_bbl"] / 1e6, rel=1e-9)
    assert abs(TBtu - b["soa_saving_TBtu_y_table_5_4"]) <= 0.6                                    # report rounds to 1 TBtu
    assert o["value_low_usd_y"] == pytest.approx(0.0, abs=1e-6) and o["confidence"] == "media"
    assert o["evidence"]["intensity"] == pytest.approx(b["CT"], rel=1e-12)


def _tip4_inputs() -> dict:
    CH4 = {"type": "gas", "composition": {"CH4": 1.0}, "T_K": F(70)}
    return dict(currency="USD", fuel=CH4,
                combustion={"site_pressure_Pa": PATM, "air": {"T_K": F(70), "humidity_ratio_kg_kg": 0.0},
                            "excess_air_fraction": 0.449, "stack_T_K": F(470), "setting_loss_fraction": 0.0,
                            "setting_loss_basis": "HHV"},
                boiler={"steam_pressure_Pa": P150, "blowdown_fraction": 0.0},
                water={"feedwater_heating": "not_charged", "feedwater_T_K": F(117), "makeup_water_T_K": F(60),
                       "condensate_return_fraction": 0.0},
                fuel_cost={"price_per_GJ": 8.0 / MMBTU, "basis": "HHV"}, water_cost={"price_per_m3": 0.0},
                operation={"hours_per_year": 8000, "fuel_consumption": {"energy_GJ_per_year": 500000 * MMBTU, "basis": "HHV"}},
                measures={"reduce_excess_air": {"target_excess_air_fraction": 0.095, "stack_T_K_after": F(370)}}, today=TODAY)


def test_doe_steam_tip_sheet_4_through_the_scanner():
    """DOE: 235 856 USD/yr. The steam engine (M19) gives 227 064 USD (its efficiencies sit 0.4-0.6 points above the DOE
    table): −3.7 %, measured, not tuned. The scanner must carry that number unchanged and count it once in the total."""
    pd = {"steam_system": {"unit_id": "B1", "inputs": _tip4_inputs()}}
    r = TO.scan_opportunities(model([{"id": "B1", "type": "boiler"}]), PRICES, "US", pd, TODAY)
    o = opp(r, "c_steam:B1:reduce_excess_air")
    assert o["value_usd_y"] == pytest.approx(227064.0, rel=1e-4)
    assert o["value_usd_y"] / 235856.0 - 1 == pytest.approx(-0.0373, abs=0.001)
    assert o["evidence"]["verified"] and o["confidence"] == "alta"
    assert r["total_value_usd_y"] == pytest.approx(o["value_usd_y"], rel=1e-9)               # combined = the only measure
    dumps(r)


# ------------------------------------------------------------------------------------------- consistency of each rule
def _heater_combustion(O2: float) -> dict:
    return {"fuel": {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 298.15},
            "air": {"T_K": 298.15, "humidity_ratio_kg_kg": 0.0}, "pressure_Pa": PATM,
            "flue_O2": {"mole_fraction": O2, "basis": "dry"}, "stack_T_K": 573.15, "setting_loss_fraction": 0.02,
            "setting_loss_basis": "HHV"}


def test_excess_air_rule_equals_combustion_engine_identity():
    c = _heater_combustion(0.06)
    u = {"id": "H1", "type": "fired_heater", "params": {"combustion": c, "fuel_W": 30e6, "fuel_basis": "HHV"}}
    r = TO.scan_opportunities(model([u]), PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    o = opp(r, "b_excess_air:H1")
    kw = dict(stack_T_K=c["stack_T_K"], setting_loss_fraction=0.02, setting_loss_basis="HHV", adiabatic_flame=False)
    e0 = CB.combustion_efficiency(c["fuel"], c["air"], PATM, flue_O2=c["flue_O2"], **kw)["efficiency"]["HHV"]["efficiency"]
    e3 = CB.combustion_efficiency(c["fuel"], c["air"], PATM, flue_O2={"mole_fraction": 0.03, "basis": "dry"},
                                  **kw)["efficiency"]["HHV"]["efficiency"]
    assert o["value_low_usd_y"] == pytest.approx(30e6 * (1 - e0 / e3) * 8000 * 3600 / 1e9 * 4.0, rel=1e-12)
    assert o["value_low_usd_y"] < o["value_high_usd_y"]                                       # 2 % O2 saves more than 3 %
    assert opp(r, "b_furnace_efficiency:H1")["overlap_group"] == o["overlap_group"]            # same unit: one group
    assert r["total_value_usd_y"] == pytest.approx(max(o["value_usd_y"], opp(r, "b_furnace_efficiency:H1")["value_usd_y"]))
    r2 = TO.scan_opportunities(model([{**u, "params": {**u["params"], "combustion": _heater_combustion(0.025)}}]), PRICES,
                               "MX", {"hours_per_year": 8000}, TODAY)
    o2 = opp(r2, "b_excess_air:H1")                              # inside the 2-3 % band: conservative 0, high to 2 %
    assert o2["value_low_usd_y"] == o2["value_usd_y"] == 0.0 < o2["value_high_usd_y"] < o["value_high_usd_y"]


def test_boiler_excess_air_to_published_15_percent():
    c = {**_heater_combustion(0.0), "excess_air_fraction": 0.45}
    c.pop("flue_O2")
    r = TO.scan_opportunities(model([{"id": "B1", "type": "boiler", "params": {"combustion": c, "absorbed_W": 10e6}}]),
                              PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    o = opp(r, "b_excess_air:B1")
    assert o["value_low_usd_y"] == o["value_high_usd_y"] > 0
    assert o["evidence"]["benchmark"]["benchmark_id"] == "boiler_excess_air"
    assert any(n["rule"] == "c_steam" for n in r["not_assessed"])                              # no steam data: said so


def test_furnace_efficiency_gap_arithmetic():
    r = TO.scan_opportunities(model([{"id": "H1", "type": "fired_heater", "params": {"efficiency": 0.80, "fuel_W": 20e6}}]),
                              PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    o = opp(r, "b_furnace_efficiency:H1")
    GJ = 20e6 * 8000 * 3600 / 1e9
    assert o["value_usd_y"] == pytest.approx(GJ * (1 - 0.8 / 0.9) * 4.0, rel=1e-12)
    assert o["value_high_usd_y"] == pytest.approx(GJ * (1 - 0.8 / 0.92) * 4.0, rel=1e-12)
    r2 = TO.scan_opportunities(model([{"id": "H1", "type": "fired_heater", "params": {"efficiency": 0.91, "fuel_W": 20e6}}]),
                               PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    assert not any(x["id"].startswith("b_furnace") for x in r2["opportunities"])


def test_pump_rule_identity_and_doe_equation():
    p = {"flow_m3_s": 0.05, "head_m": 60.0, "density_kg_m3": 998.0, "shaft_power_W": 60e3, "speed_rpm": 3550, "stages": 1,
         "pump_class": "ESFM"}
    r = TO.scan_opportunities(model([{"id": "P1", "type": "pump", "params": p}]), PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    o = opp(r, "g_electricity:P1")
    Q_gpm, H_ft = 0.05 / S.GPM_M3S, 60.0 / 0.3048
    Ns = 3600 * math.sqrt(Q_gpm) / H_ft ** 0.75
    eta = (-0.85 * math.log(Q_gpm) ** 2 - 0.38 * math.log(Ns) * math.log(Q_gpm) - 11.48 * math.log(Ns) ** 2
           + 17.8 * math.log(Q_gpm) + 179.8 * math.log(Ns) - (130.99 + 555.6)) / 100
    assert o["evidence"]["reference"]["eta_std"]["ESFM.3600"] == pytest.approx(eta, rel=1e-12)
    P_hyd = 998.0 * 9.80665 * 0.05 * 60.0
    assert o["evidence"]["efficiency_now"] == pytest.approx(P_hyd / 60e3, rel=1e-12)
    assert o["value_usd_y"] == pytest.approx((60e3 - P_hyd / eta) / 1e3 * 8000 * 0.10, rel=1e-12)
    better = {**p, "shaft_power_W": P_hyd / eta * 0.999}
    r2 = TO.scan_opportunities(model([{"id": "P1", "type": "pump", "params": better}]), PRICES, "MX",
                               {"hours_per_year": 8000}, TODAY)
    assert not r2["opportunities"]


def test_fcc_conversion_valued_only_with_users_target_and_margin():
    u = {"id": "FCC", "type": "fluidized_bed_reactor", "label_es": "FCC", "params": {"conversion_pct": 48.0, "throughput_bpd": 40000}}
    r = TO.scan_opportunities(model([u]), PRICES, "MX", {"hours_per_year": 8400}, TODAY)
    o = opp(r, "e_yield:FCC")
    assert o["value_usd_y"] is None and o["evidence"]["below_range"] and len(o["missing"]) == 2
    u["params"].update(max_conversion_pct=72.0, conversion_value_usd_bbl_point=0.35)
    o = opp(TO.scan_opportunities(model([u]), PRICES, "MX", {"hours_per_year": 8400}, TODAY), "e_yield:FCC")
    assert o["value_usd_y"] == pytest.approx(40000 * 8400 / 24 * 24.0 * 0.35, rel=1e-12)


def test_flaring_and_co2_need_their_own_data():
    u = {"id": "FL", "type": "flare", "params": {"flare_kg_h": 500.0, "flare_HV_J_kg": 45e6, "flare_carbon_mass_fraction": 0.75}}
    r = TO.scan_opportunities(model([u]), PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    fuel, co2 = opp(r, "f_flaring:FL:fuel"), opp(r, "f_flaring:FL:co2")
    assert fuel["value_high_usd_y"] == pytest.approx(500 * 8000 * 45e6 / 1e9 * 4.0, rel=1e-12) and fuel["value_low_usd_y"] == 0
    assert not fuel["include_in_total"] and r["total_value_usd_y"] == 0
    assert co2["value_usd_y"] is None and co2["evidence"]["t_CO2_y"] == pytest.approx(500 * 8000 * 0.75 * 44.0095 / 12.011 / 1e3)
    u["params"]["flare_recoverable_fraction"] = 0.4
    r = TO.scan_opportunities(model([u]), {**PRICES, "carbon_usd_tCO2": 20.0}, "MX", {"hours_per_year": 8000}, TODAY)
    assert opp(r, "f_flaring:FL:co2")["value_usd_y"] == pytest.approx(opp(r, "f_flaring:FL:co2")["evidence"]["t_CO2_y"] * 0.4 * 20)
    assert r["total_value_usd_y"] == pytest.approx(opp(r, "f_flaring:FL:fuel")["value_usd_y"]
                                                   + opp(r, "f_flaring:FL:co2")["value_usd_y"])


def test_economics_sensitivity_becomes_confirm_items_outside_the_total():
    econ = {"operating_costs": {"total_usd_y": 5e6}, "sensitivity": {"ranked_by": "VPN", "note": "uno a la vez", "rows": [
        {"input": "prices.metanol", "label": "precio del metanol", "base_value": 500, "unit": "USD/t", "range": "±20%",
         "low_value": 400, "high_value": 600, "npv_swing_usd": 3e6, "operating_cost_usd_y_low": 4.8e6,
         "operating_cost_usd_y_high": 5.2e6, "rank": 1},
        {"input": "basis.plant_life_years", "label": "vida de la planta", "base_value": 30, "unit": "años", "range": "±20%",
         "low_value": 24, "high_value": 36, "npv_swing_usd": 1e6, "operating_cost_usd_y_low": 5e6,
         "operating_cost_usd_y_high": 5e6, "rank": 2}]}}
    m = model([{"id": "R1", "type": "reactor"}], "plant_model")
    m["economics"] = econ
    r = TO.scan_opportunities(m, PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    a, b = opp(r, "h_economics:prices.metanol"), opp(r, "h_economics:basis.plant_life_years")
    assert a["title_es"].startswith("Negociar") and a["value_usd_y"] == pytest.approx(2e5)
    assert b["title_es"].startswith("Confirmar") and b["value_usd_y"] is None
    assert r["total_value_usd_y"] is None or r["total_value_usd_y"] == 0


def test_model_energy_equal_to_benchmark_is_not_an_opportunity():
    b = BW["atmospheric_crude_distillation"]
    bpd = 300000.0
    u = {"id": "CDU", "type": "column", "label_es": "Destilación atmosférica", "params": {"throughput_bpd": bpd},
         "duties_W": {"energy_W": b["CT"] * S.BTU_J * bpd / 86400}}
    r = TO.scan_opportunities(model([u], "refinery"), PRICES, "MX", {"hours_per_year": 8000}, TODAY)
    na = [n for n in r["not_assessed"] if n["rule"] == "d_energy_intensity"]
    assert na and "mismo benchmark" in na[0]["reason"] and not r["opportunities"]
    u["duties_W"]["energy_W"] *= 1.2                                                             # model above CT: low confidence
    o = opp(TO.scan_opportunities(model([u], "refinery"), PRICES, "MX", {"hours_per_year": 8000}, TODAY), "d_energy_intensity:CDU")
    assert o["confidence"] == "baja" and "palabra clave" in o["evidence"]["process_identified"]


def test_overlap_total_and_ranking():
    pd = {"hours_per_year": 8000, "heat_integration": {"streams": NRCAN, "dT_min_K": 10.0, "current_hot_utility_W": 5750e3}}
    units = [{"id": "H1", "type": "fired_heater", "params": {"efficiency": 0.80, "fuel_W": 20e6}},
             {"id": "P1", "type": "pump", "params": {"flow_m3_s": 0.05, "head_m": 60.0, "density_kg_m3": 998.0,
                                                     "shaft_power_W": 60e3, "speed_rpm": 3550, "stages": 1}}]
    r = TO.scan_opportunities(model(units), PRICES, "MX", pd, TODAY)
    pinch, furn, pump = (opp(r, i) for i in ("a_heat_integration:pinch", "b_furnace_efficiency:H1", "g_electricity:P1"))
    assert pinch["overlap_group"] == furn["overlap_group"] != pump["overlap_group"]
    assert r["total_value_usd_y"] == pytest.approx(max(pinch["value_usd_y"], furn["value_usd_y"]) + pump["value_usd_y"])
    assert r["total_value_usd_y"] < r["naive_sum_usd_y"]
    vals = [o["value_usd_y"] for o in r["opportunities"]]
    assert vals == sorted(vals, reverse=True) and [o["rank"] for o in r["opportunities"]] == [1, 2, 3]
    assert r["evidence"]["verified"] and all(r["evidence"]["checks"].values())


def test_steam_in_mxn_needs_and_uses_an_exchange_rate():
    inp = _tip4_inputs()
    inp["currency"] = "MXN"
    pd = {"steam_system": {"unit_id": "B1", "inputs": inp}}
    r = TO.scan_opportunities(model([{"id": "B1", "type": "boiler"}]), {**PRICES, "usd_per_currency": {"MXN": 0.05}}, "MX",
                              pd, TODAY)
    o = opp(r, "c_steam:B1:reduce_excess_air")
    assert o["value_usd_y"] == pytest.approx(o["evidence"]["saving_in_currency"] * 0.05, rel=1e-12)


# ------------------------------------------------------------------------------------------- plant data and refusals
def test_plant_data_pump_in_field_units_and_rule_table():
    table = [{"unit_id": "P-101", "variable": "flujo", "value": 792.5, "uom": "gpm", "period": "2026-10"},
             {"unit_id": "P-101", "variable": "carga dinámica", "value": 196.85, "uom": "ft", "period": "2026-10"},
             {"unit_id": "P-101", "variable": "densidad", "value": 998, "uom": "kg/m3"},
             {"unit_id": "P-101", "variable": "potencia al eje", "value": 60, "uom": "kW", "period": "2026-10"},
             {"unit_id": "P-101", "variable": "potencia al eje", "value": 62, "uom": "kW", "period": "2026-11"},
             {"unit_id": "P-101", "variable": "rpm", "value": 3550, "uom": "rpm"},
             {"unit_id": "P-101", "variable": "etapas", "value": 1, "uom": ""},
             {"unit_id": "H-1", "variable": "O2", "value": 5, "uom": "%"},                       # no basis: rejected
             {"unit_id": "H-1", "variable": "temperatura de chimenea", "value": 300, "uom": "°C"},
             {"unit_id": "X", "variable": "color", "value": 1, "uom": ""},                       # unknown variable
             {"unit_id": "site", "variable": "horas al año", "value": 8000, "uom": "h/y"}]
    r = TO.opportunities_from_plant_data(table, prices=PRICES, today=TODAY)
    assert {i["unit_id"]: i["type"] for i in r["units_inferred"]} == {"P-101": "pump", "H-1": "fired_heater"}
    assert len(r["readings_rejected"]) == 2
    assert any("base" in x["reason"] for x in r["readings_rejected"])
    o = opp(r, "g_electricity:P-101")
    assert o["evidence"]["shaft_W"] == pytest.approx(61e3) and o["confidence"] == "alta"
    assert any(s["rule"] == "g_electricity" and s["status"] == "valorada" for s in r["rules_can_run"])
    assert any(s["rule"] == "b_excess_air" and s["unit_ids"] == ["H-1"] for s in r["rules_cannot_run"])
    dumps(r)


def test_missing_inputs_give_no_value_and_say_what_is_missing():
    r = TO.scan_opportunities(model([{"id": "P1", "type": "pump", "params": {"flow_m3_s": 0.05}}]), None, "MX", None, TODAY)
    o = opp(r, "g_electricity:P1")
    assert o["value_usd_y"] is None and o["value_low_usd_y"] is None and len(o["missing"]) == 5
    p = {"flow_m3_s": 0.05, "head_m": 60.0, "density_kg_m3": 998.0, "shaft_power_W": 60e3, "speed_rpm": 3550, "stages": 1}
    r = TO.scan_opportunities(model([{"id": "P1", "type": "pump", "params": p}]), None, "MX", {"hours_per_year": 8000}, TODAY)
    o = opp(r, "g_electricity:P1")                       # Mexico: the bank has time-of-use tariffs only, no single price
    assert o["value_usd_y"] is None and o["missing"] == ["precio de la electricidad"]
    assert r["prices_used"]["fuel_usd_GJ"]["origin"] == "referencia publicada"            # gas from the bank, flagged
    assert any("NO TU COTIZACIÓN" in a for a in r["assumptions"])
    assert r["evidence"]["checks"]["no_value_without_inputs"]


def test_hours_from_bank_are_flagged():
    r = TO.scan_opportunities(model([{"id": "H1", "type": "fired_heater", "params": {"efficiency": 0.8, "fuel_W": 1e6}}]),
                              PRICES, "MX", None, TODAY)
    assert r["hours_per_year"]["origin"] == "referencia publicada" and r["hours_per_year"]["url"]
    assert opp(r, "b_furnace_efficiency:H1")["value_usd_y"] > 0


def test_refusals():
    with pytest.raises(ValueError, match="ProcessModel"):
        TO.scan_opportunities({"units": "x"})
    with pytest.raises(ValueError, match="claves desconocidas"):
        TO.scan_opportunities(model([]), {"gas": 3.0}, today=TODAY)
    with pytest.raises(ValueError, match="id único"):
        TO.scan_opportunities(model([{"id": "A", "type": "pump"}, {"id": "A", "type": "pump"}]), PRICES, today=TODAY)
    with pytest.raises(ValueError, match="bandwidth_process"):
        TO.scan_opportunities(model([{"id": "U", "type": "column", "params": {"bandwidth_process": "magia"}}]), PRICES,
                              today=TODAY)
    with pytest.raises(ValueError, match="lista de lecturas"):
        TO.opportunities_from_plant_data({"unit_id": "P1"})
    with pytest.raises(ValueError, match="ninguna lectura válida"):
        TO.opportunities_from_plant_data([{"unit_id": "P1", "variable": "color", "value": 1, "uom": ""}])


def test_benchmarks_are_sourced_and_transcribed_consistently():
    entries = S.load_benchmarks()["entries"]
    for k, e in entries.items():
        assert e["source"] and e["url"].startswith("https://") and e["period"] and e["date_read"] == "2026-11-16", k
    for b in BW.values():                                       # bands ordered as in the study
        assert b["CT"] > b["SOA"] > b["PM"] > b["TM"]
    assert sum(b["soa_saving_TBtu_y_table_5_4"] for b in BW.values()) == 287       # report total 286: rounding of rows
    for c in entries["chemicals_bandwidth_2015"]["chemicals"].values():
        assert c["CT"] >= c["SOA"]
    assert S.EQUIPMENT_TYPES and "pump" in S.EQUIPMENT_TYPES


def test_wrappers_mirror_the_tools():
    from packages.agent import wrappers_opportunities as W
    assert [f.__name__ for f in TO.OPPORTUNITIES_TOOLS] == [w.name for w in W.AGENT_WRAPPERS]


def test_tula_refinery_no_intensity_false_positive_on_the_fcc_fractionator():
    """Regression (integrator, M27): rule (d) once matched the FCC fractionator by the «fcc» keyword and divided the whole
    conversion section's energy by that unit's feed (~9.4 M USD/y spurious). Now: only params.bandwidth_process when given,
    no keyword guessing on auxiliaries or on a process already assigned, own throughput and own energy only."""
    from packages.dwsim_client import tools_refinery as TR

    pm = TR.refinery_model("tula")["process_model"]
    units = {u["id"]: u for u in pm["units"]}
    r = TO.scan_opportunities(pm, None, "MX", None, TODAY)
    d = [o for o in r["opportunities"] if o["rule"] == "d_energy_intensity"]
    assert not any("fcc_fraccionadora" in o["unit_ids"] for o in d)
    assert not any("fcc_fraccionadora" in n["unit_ids"] for n in r["not_assessed"] if n["rule"] in ("d_energy_intensity",
                                                                                                    "e_yield"))
    for o in d:
        (uid,) = o["unit_ids"]
        m = {**(units[uid].get("params") or {}), **(units[uid].get("metrics") or {})}
        assert m.get("throughput_bpd") or m.get("production_bpd"), uid
    dumps(r)
