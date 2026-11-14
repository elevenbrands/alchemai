"""Utility costs by region (bloque M17): `utility_costs` and `list_utility_price_references`.

Published checks: U.S. DOE Steam Tip Sheet #15 «Benchmark the Fuel Cost of Steam Generation» (DOE/GO-102012-3391, January
2012), Table 1 and its example; Ulrich & Vasudevan, «How to Estimate Utility Costs», Chemical Engineering, April 2006,
pp. 66-69, Table 1 and its two worked examples. Both read on 2026-11-14. No network: the price bank is a file.
"""
from __future__ import annotations

import json
import math
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import CoolProp.CoolProp as CP  # noqa: E402

from packages.costing import utilities as U  # noqa: E402
from packages.costing.annual import annual_cost  # noqa: E402
from packages.dwsim_client import tools_utilities as TU  # noqa: E402

BTU_LB = 2326.0                     # J/kg per Btu/lb (exact by definition of the IT Btu and the pound)
LB = 0.45359237
MMBTU_GJ = 1.05505585262
TODAY = "2026-11-14"


def K(F: float) -> float:
    return (F - 32.0) / 1.8 + 273.15


def psig(p: float) -> float:
    return (p + 14.696) * 6894.757


def doe_level(p_psig: float, T_F: float, eff: float = 0.857) -> dict:
    return dict(name="x", pressure_Pa=psig(p_psig), boiler_efficiency=eff, feedwater_T_K=K(T_F),
                condensate_return_fraction=1.0, blowdown_fraction=0.0, makeup_water_T_K=288.15, feedwater_heating="not_charged")


MP = dict(name="mp", pressure_Pa=11e5, boiler_efficiency=0.82, feedwater_T_K=378.15, condensate_return_fraction=0.7,
          condensate_T_K=363.15, blowdown_fraction=0.04, makeup_water_T_K=293.15, feedwater_heating="deaerator_steam")
CW = dict(supply_T_K=303.15, return_T_K=313.15, pump_head_m=30.0, pump_efficiency=0.7, evaporation_latent_fraction=0.85,
          drift_fraction=0.0005, cycles_of_concentration=5.0)
R20 = dict(name="r-20", evaporator_T_K=253.15, condenser_T_K=313.15, second_law_efficiency=0.6)
MINE = {"fuel_per_GJ": 5.0, "fuel_basis": "HHV", "electricity_per_kWh": 0.08, "boiler_makeup_water_per_m3": 1.5,
        "cooling_makeup_water_per_m3": 0.6}


# ------------------------------------------------------------------------------------------- published: DOE tip sheet
DOE_TABLE_1 = {150: [1178, 1128, 1078, 1028, 977], 450: [1187, 1137, 1087, 1037, 986], 600: [1184, 1134, 1084, 1034, 984]}


def test_doe_tip_sheet_15_table_1_energy_per_pound_of_steam():
    worst = 0.0
    for p, row in DOE_TABLE_1.items():
        for T_F, ref in zip((50, 100, 150, 200, 250), row):
            r = U.steam_level(doe_level(p, T_F, eff=1.0), fuel_per_GJ=1.0, water_per_m3=0.0)
            btu_lb = r["fuel_GJ_per_tonne"] * 1e6 / BTU_LB
            worst = max(worst, abs(btu_lb - ref) / ref)
            assert btu_lb == pytest.approx(ref, rel=2e-3), (p, T_F, btu_lb)
    assert worst < 1.7e-3                                                       # 15 values, worst 1.6 Btu/lb at 600 psig


def test_doe_tip_sheet_15_example_fuel_cost_of_steam():
    """Natural gas $8.00/MMBtu, 450 psig saturated, 230 °F feed water, 85.7 % → 1,006 Btu/lb and $9.39 per 1,000 lb."""
    r = U.steam_level(doe_level(450, 230), fuel_per_GJ=8.0 / MMBTU_GJ, water_per_m3=0.0)
    assert r["fuel_GJ_per_tonne"] * 0.857 * 1e6 / BTU_LB == pytest.approx(1006, abs=1.0)
    assert r["cost_per_tonne"] * LB == pytest.approx(9.39, rel=1.5e-3)          # ours 9.398
    assert r["breakdown_per_tonne"] == {"fuel": pytest.approx(r["cost_per_tonne"]), "water": 0.0, "power": 0.0}
    assert any("not_charged" in a and "DOE" in a for a in r["assumptions"])


# ---------------------------------------------------------------------------------- published: Ulrich & Vasudevan 2006
def test_ulrich_vasudevan_2006_worked_examples():
    """CE PCI 470, fuel $4.50/GJ: electricity $0.106/kWh, cooling water $0.048/m³ (q = 10 m³/s, grass roots), 32 barg steam
    $0.019/kg (40 kg/s, grass roots), −5 °C refrigerant $4.0e-6/kJ (40 kJ/s, grass roots)."""
    e = U.ulrich_vasudevan("electricity", 470, 4.5)
    assert round(e["price"], 3) == 0.106 and e["inflation_term"] == pytest.approx(0.0611) and e["fuel_term"] == pytest.approx(0.045)
    cw = U.ulrich_vasudevan("cooling_water", 470, 4.5, "grass_roots", capacity_m3_s=10)
    assert cw["a"] == pytest.approx(0.0000725) and round(cw["price"], 3) == 0.048
    s = U.ulrich_vasudevan("steam", 470, 4.5, "grass_roots", pressure_barg=32, capacity_kg_s=40)
    assert round(s["price"], 3) == 0.019
    assert s["price"] == pytest.approx(2.3e-5 * 40 ** -0.9 * 470 + 0.0034 * 32 ** 0.05 * 4.5)
    r = U.ulrich_vasudevan("refrigeration", 470, 4.5, "grass_roots", T_K=268, capacity_kW=40)
    assert r["price"] == pytest.approx(4.0e-6, rel=0.01)
    assert round(1.2e9 * round(r["price"], 7), -2) == 4800                     # their annual expense, $4,800/yr
    assert "abril de 2006" in s["source"] and "Tabla 1" in s["source"]


def test_ulrich_vasudevan_ranges_and_basis():
    with pytest.raises(ValueError, match="fuera del intervalo"):
        U.ulrich_vasudevan("steam", 470, 4.5, pressure_barg=60, capacity_kg_s=10)
    with pytest.raises(ValueError, match="fuera del intervalo"):
        U.ulrich_vasudevan("refrigeration", 470, 4.5, T_K=320, capacity_kW=40)
    with pytest.raises(ValueError, match="basis"):
        U.ulrich_vasudevan("electricity", 470, 4.5, "turnkey")
    pm = U.ulrich_vasudevan("cooling_water", 470, 4.5, "process_module", capacity_m3_s=1)["price"]
    gr = U.ulrich_vasudevan("cooling_water", 470, 4.5, "grass_roots", capacity_m3_s=1)["price"]
    assert pm > gr


# ------------------------------------------------------------------------------------------------- steam: consistency
def test_steam_island_energy_and_water_balances_close_independently():
    r = U.steam_level(MP, fuel_per_GJ=5.0, water_per_m3=1.5)
    m = r["mass_per_kg_delivered"]
    P = MP["pressure_Pa"]
    h_s = CP.PropsSI("H", "P", P, "Q", 1, "Water")
    h_f = CP.PropsSI("H", "P", P, "Q", 0, "Water")
    h = lambda T: CP.PropsSI("H", "T", T, "Q", 0, "Water")  # noqa: E731
    fuel_to_steam = r["fuel_GJ_per_tonne"] * 1e6 * MP["boiler_efficiency"]        # J per kg delivered
    island_out = h_s + m["blowdown"] * h_f
    island_in = m["condensate_return"] * h(MP["condensate_T_K"]) + m["makeup"] * h(MP["makeup_water_T_K"]) + fuel_to_steam
    assert island_in == pytest.approx(island_out, rel=1e-10)
    boiler_in = m["feedwater"] * h(MP["feedwater_T_K"]) + fuel_to_steam
    assert boiler_in == pytest.approx(m["generated"] * h_s + m["blowdown"] * h_f, rel=1e-10)
    assert m["feedwater"] == pytest.approx(m["generated"] + m["blowdown"])
    assert m["blowdown"] / m["feedwater"] == pytest.approx(MP["blowdown_fraction"])
    assert m["makeup"] + m["condensate_return"] + m["deaerator_steam"] == pytest.approx(m["feedwater"])
    assert m["makeup"] == pytest.approx(1.0 - 0.7 + m["blowdown"])                # what leaves the island and does not return
    assert all(abs(v) < 1e-9 for k, v in r["energy_balance"].items() if k.endswith("residual_fraction"))
    assert m["generated"] > 1.0 and m["deaerator_steam"] > 0.0


@pytest.mark.parametrize("P", [3e5, 11e5, 42e5, 100e5])
def test_steam_cost_per_GJ_never_below_fuel_over_efficiency(P):
    for mode in ("not_charged", "deaerator_steam"):
        lv = dict(MP, pressure_Pa=P, feedwater_heating=mode)
        r = U.steam_level(lv, fuel_per_GJ=5.0, water_per_m3=0.0)
        assert r["breakdown_per_GJ_delivered"]["fuel"] >= 5.0 / 0.82
        assert r["floor_fuel_per_GJ_over_efficiency"] == pytest.approx(5.0 / 0.82)
        assert r["cost_per_tonne"] / r["heat_delivered_GJ_per_tonne"] == pytest.approx(r["cost_per_GJ_delivered"])


def test_steam_trends_and_units_round_trip():
    base = U.steam_level(MP, 5.0, 1.5)
    assert U.steam_level(dict(MP, boiler_efficiency=0.9), 5.0, 1.5)["cost_per_tonne"] < base["cost_per_tonne"]
    more_return = U.steam_level(dict(MP, condensate_return_fraction=0.9), 5.0, 1.5)
    assert more_return["breakdown_per_tonne"]["fuel"] < base["breakdown_per_tonne"]["fuel"]
    assert more_return["breakdown_per_tonne"]["water"] < base["breakdown_per_tonne"]["water"]
    assert U.steam_level(dict(MP, blowdown_fraction=0.10), 5.0, 1.5)["cost_per_tonne"] > base["cost_per_tonne"]
    assert U.steam_level(dict(MP, superheat_K=50.0), 5.0, 1.5)["heat_delivered_GJ_per_tonne"] > base["heat_delivered_GJ_per_tonne"]
    # the price scales linearly and $/MMBtu → $/GJ → $/MMBtu returns
    assert U.steam_level(MP, 10.0, 0.0)["cost_per_tonne"] == pytest.approx(2 * U.steam_level(MP, 5.0, 0.0)["cost_per_tonne"])
    assert (8.0 / MMBTU_GJ) * MMBTU_GJ == pytest.approx(8.0)
    with_aux = U.steam_level(dict(MP, auxiliary_power_kWh_per_tonne=5.0), 5.0, 1.5, electricity_per_kWh=0.1)
    assert with_aux["breakdown_per_tonne"]["power"] == pytest.approx(0.5)
    # charging the feed-water heating can only cost more than not charging it
    assert U.steam_level(dict(MP, feedwater_heating="not_charged"), 5.0, 0.0)["cost_per_tonne"] < U.steam_level(MP, 5.0, 0.0)["cost_per_tonne"]


def test_steam_refuses_missing_engineering_inputs_and_bad_values():
    for k in ("boiler_efficiency", "feedwater_T_K", "condensate_return_fraction", "blowdown_fraction", "makeup_water_T_K",
              "feedwater_heating"):
        lv = {a: b for a, b in MP.items() if a != k}
        with pytest.raises(ValueError, match=k):
            U.steam_level(lv, 5.0, 1.5)
    with pytest.raises(ValueError, match="condensate_T_K"):
        U.steam_level({a: b for a, b in MP.items() if a != "condensate_T_K"}, 5.0, 1.5)
    with pytest.raises(ValueError, match="boiler_efficiency"):
        U.steam_level(dict(MP, boiler_efficiency=1.2), 5.0, 1.5)
    with pytest.raises(ValueError, match="saturación"):
        U.steam_level(dict(MP, feedwater_T_K=500.0), 5.0, 1.5)
    with pytest.raises(ValueError, match="más caliente"):
        U.steam_level(dict(MP, feedwater_T_K=330.0, condensate_return_fraction=1.0), 5.0, 1.5)


def test_fired_heat():
    r = U.fired_heat({"furnace_efficiency": 0.8}, 5.0)
    assert r["cost_per_GJ_absorbed"] == pytest.approx(6.25)
    with pytest.raises(ValueError, match="furnace_efficiency"):
        U.fired_heat({}, 5.0)


# ---------------------------------------------------------------------------------------------------- cooling water
def test_cooling_water_balances_and_units():
    r = U.cooling_water(CW, electricity_per_kWh=0.08, water_per_m3=0.6)
    w = r["water_kg_per_m3_circulated"]
    assert w["makeup"] == pytest.approx(w["evaporation"] + w["drift"] + w["blowdown"])
    assert w["makeup"] / (w["blowdown"] + w["drift"]) == pytest.approx(CW["cycles_of_concentration"])
    assert w["makeup"] == pytest.approx(w["evaporation"] * 5.0 / 4.0)
    rho = CP.PropsSI("D", "T", 303.15, "P", 101325, "Water")
    q = rho * (CP.PropsSI("H", "T", 313.15, "Q", 0, "Water") - CP.PropsSI("H", "T", 303.15, "Q", 0, "Water"))
    assert r["heat_removed_GJ_per_m3"] == pytest.approx(q / 1e9)
    assert q == pytest.approx(rho * 4180.0 * 10.0, rel=5e-3)                      # ρ·cp·ΔT with cp ≈ 4.18 kJ/kg K
    assert r["power_kWh_per_m3"]["pump"] == pytest.approx(rho * 9.80665 * 30.0 / 0.7 / 3.6e6)
    assert r["cost_per_m3"] * r["m3_per_GJ_removed"] == pytest.approx(r["cost_per_GJ_removed"])
    assert r["cost_per_1000_m3"] == pytest.approx(1000 * r["cost_per_m3"])
    assert sum(r["breakdown_per_m3"].values()) == pytest.approx(r["cost_per_m3"])
    assert r["breakdown_per_m3"]["makeup_water"] == pytest.approx(w["makeup"] / rho * 0.6)
    assert r["breakdown_per_m3"]["fan_power"] == 0.0 and any("ventiladores" in x for x in r["not_included"])
    # all the duty by evaporation: evaporation = Q / h_fg, about 1.7 % of the circulation for a 10 K range
    full = U.cooling_water(dict(CW, evaporation_latent_fraction=1.0), 0.08, 0.6)
    assert full["water_kg_per_m3_circulated"]["evaporation"] == pytest.approx(q / full["latent_heat_J_kg"])
    assert 0.016 < full["evaporation_fraction_of_circulation"] < 0.018


def test_cooling_water_trends_options_and_refusals():
    base = U.cooling_water(CW, 0.08, 0.6)
    assert U.cooling_water(dict(CW, cycles_of_concentration=3.0), 0.08, 0.6)["breakdown_per_m3"]["makeup_water"] > base["breakdown_per_m3"]["makeup_water"]
    assert U.cooling_water(dict(CW, pump_head_m=60.0), 0.08, 0.6)["breakdown_per_m3"]["pump_power"] == pytest.approx(2 * base["breakdown_per_m3"]["pump_power"])
    opt = U.cooling_water(dict(CW, fan_power_kWh_per_m3=0.05, treatment_cost_per_m3_makeup=0.2), 0.08, 0.6)
    assert opt["breakdown_per_m3"]["fan_power"] == pytest.approx(0.004)
    assert opt["breakdown_per_m3"]["treatment"] == pytest.approx(opt["makeup_fraction_of_circulation"] * 0.2)
    drifty = U.cooling_water(dict(CW, drift_fraction=0.02), 0.08, 0.6)
    assert drifty["water_kg_per_m3_circulated"]["blowdown"] == 0.0 and drifty["cycles_of_concentration_effective"] < 5.0
    assert any("arrastre" in w for w in drifty["warnings"])
    for k in U.CW_REQUIRED:
        with pytest.raises(ValueError, match=k):
            U.cooling_water({a: b for a, b in CW.items() if a != k}, 0.08, 0.6)
    with pytest.raises(ValueError, match="cycles_of_concentration"):
        U.cooling_water(dict(CW, cycles_of_concentration=1.0), 0.08, 0.6)
    with pytest.raises(ValueError, match="supply_T_K"):
        U.cooling_water(dict(CW, supply_T_K=320.0), 0.08, 0.6)


# ---------------------------------------------------------------------------------------------------- refrigeration
def test_refrigeration_carnot_fraction_and_colder_costs_more():
    r = U.refrigeration_level(R20, electricity_per_kWh=0.08)
    assert r["cop_carnot"] == pytest.approx(253.15 / 60.0) and r["cop"] == pytest.approx(0.6 * 253.15 / 60.0)
    assert r["power_kWh_per_GJ_removed"] == pytest.approx(1e9 / 3.6e6 / r["cop"])
    assert r["cost_per_GJ_removed"] == pytest.approx(0.08 * 277.7777777778 / r["cop"])
    assert r["heat_rejected_GJ_per_GJ_removed"] == pytest.approx(1 + 1 / r["cop"])
    assert U.refrigeration_level(dict(R20, second_law_efficiency=1.0), 0.08)["cop"] == pytest.approx(r["cop_carnot"])
    costs = [U.refrigeration_level(dict(R20, evaporator_T_K=T), 0.08)["cost_per_GJ_removed"] for T in (278.15, 253.15, 233.15, 203.15)]
    assert costs == sorted(costs) and costs[0] < costs[-1]
    with_rej = U.refrigeration_level(dict(R20, heat_rejection=0.5), 0.08)
    assert with_rej["breakdown_per_GJ_removed"]["heat_rejection"] == pytest.approx(0.5 * (1 + 1 / r["cop"]))
    with pytest.raises(ValueError, match="cooling_water"):
        U.refrigeration_level(dict(R20, heat_rejection="cooling_water"), 0.08)
    with pytest.raises(ValueError, match="second_law_efficiency"):
        U.refrigeration_level({k: v for k, v in R20.items() if k != "second_law_efficiency"}, 0.08)
    with pytest.raises(ValueError, match="evaporator_T_K < condenser_T_K"):
        U.refrigeration_level(dict(R20, evaporator_T_K=320.0), 0.08)


# ---------------------------------------------------------------------------------------------------- the price bank
READ_ON_2026_11_14 = {                                   # id: (fuel published, electricity published)
    "US-EIA-2026-08": (4.34, 9.64), "US-TX-EIA-2026-08": (2.98, 7.21), "US-LA-EIA-2026-08": (3.94, 6.10),
    "EU27-EUROSTAT-2026-S1": (14.3195, 0.1559), "DE-EUROSTAT-2026-S1": (15.1064, 0.1742), "ES-EUROSTAT-2026-S1": (13.1056, 0.1177),
    "FR-EUROSTAT-2026-S1": (13.0639, 0.1239), "NL-EUROSTAT-2026-S1": (18.4644, 0.1824),
}
CFE_READ = {"MX-CFE-GDMTH-ORIENTE-2026-11": ({"base": 0.9018, "intermedia": 1.5799, "punta": 1.7739}, 202.45 + 346.20),
            "MX-CFE-GDMTH-GOLFO-NORTE-2026-11": ({"base": 0.9904, "intermedia": 1.5333, "punta": 1.6634}, 57.74 + 386.38)}


def test_bank_entries_are_complete_dated_and_equal_to_what_was_read():
    bank = json.loads(U.BANK_PATH.read_text(encoding="utf-8"))
    refs = bank["references"]
    assert set(refs) == set(READ_ON_2026_11_14) | set(CFE_READ)
    for rid, r in refs.items():
        assert r["retrieved"] == "2026-11-14" and r["region"] and r["label"] and r["currency"] in ("USD", "EUR", "MXN")
        for e in r["entries"]:
            for field in ("key", "commodity", "value", "unit", "currency", "period", "period_end", "source", "url", "caveat"):
                assert e.get(field), (rid, field)
            assert e["url"].startswith("https://") and e["key"] in U.PRICE_KEYS
    for rid, (fuel, el) in READ_ON_2026_11_14.items():
        by = {e["key"]: e for e in refs[rid]["entries"]}
        assert by["fuel_per_GJ"]["value"] == fuel and by["electricity_per_kWh"]["value"] == el
        assert by["fuel_per_GJ"]["fuel_basis"] == "HHV"
    for rid, (tou, demand) in CFE_READ.items():
        by = {e["key"]: e for e in refs[rid]["entries"]}
        assert by["electricity_time_of_use_per_kWh"]["value"] == tou
        assert by["demand_charge_per_kW_month"]["value"] == pytest.approx(demand)
    assert any(x["region"] == "MX" and "reCAPTCHA" in x["why"] for x in bank["not_included"])      # Mexican gas: pending


def test_list_references_converts_published_units_and_filters():
    out = U.list_references()
    assert len(out["references"]) == 10 and "no tu contrato" in out["rule"]
    us = {p["key"]: p for p in next(r for r in out["references"] if r["id"] == "US-EIA-2026-08")["prices"]}
    gj_per_mcf = 1036 * 1000 * 1055.05585262 / 1e9
    assert us["fuel_per_GJ"]["value"] == pytest.approx(4.34 / gj_per_mcf) == pytest.approx(3.9706, abs=1e-4)
    assert us["fuel_per_GJ"]["published_value"] == 4.34 and "Mcf" in us["fuel_per_GJ"]["published_unit"]
    assert us["electricity_per_kWh"]["value"] == pytest.approx(0.0964)
    assert [r["id"] for r in U.list_references("MX")["references"]] == sorted(CFE_READ, reverse=True) or \
        sorted(r["id"] for r in U.list_references("MX")["references"]) == sorted(CFE_READ)
    assert {r["id"] for r in U.list_references("US")["references"]} == {"US-EIA-2026-08", "US-TX-EIA-2026-08", "US-LA-EIA-2026-08"}
    assert U.list_references("ZZ")["references"] == []
    json.dumps(TU.list_utility_price_references("EU27"))


def test_no_network_is_used(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    U.load_bank.cache_clear()
    out = U.utility_costs("USD", prices={"boiler_makeup_water_per_m3": 1.5}, reference="US-EIA-2026-08", steam=[MP], today=TODAY)
    assert out["steam"][0]["cost_per_tonne"] > 0


# ----------------------------------------------------------------------------------------------- rules of use
def test_refuses_without_prices_or_reference_and_lists_references():
    with pytest.raises(ValueError) as e:
        U.utility_costs("USD", steam=[MP], cooling_water_spec=CW, refrigeration=[R20])
    msg = str(e.value)
    for key in ("fuel_per_GJ", "electricity_per_kWh", "boiler_makeup_water_per_m3", "cooling_makeup_water_per_m3"):
        assert key in msg
    assert "no se suponen" in msg and "US-EIA-2026-08" in msg and "MX-CFE-GDMTH-ORIENTE-2026-11" in msg
    assert "siempre es el de tu planta" in msg                                   # water is never in the bank
    with pytest.raises(ValueError, match="Referencias disponibles"):
        U.utility_costs("USD", prices=MINE, reference="US-EIA-1999-01", steam=[MP])
    with pytest.raises(ValueError, match="la referencia «US-EIA-2026-08» no lo trae|siempre es el de tu planta"):
        U.utility_costs("USD", reference="US-EIA-2026-08", steam=[MP])           # the reference has no water price
    with pytest.raises(ValueError, match="Nada que calcular"):
        U.utility_costs("USD", prices=MINE)
    with pytest.raises(ValueError, match="fuel_basis"):
        U.utility_costs("USD", prices={k: v for k, v in MINE.items() if k != "fuel_basis"}, steam=[MP])
    with pytest.raises(ValueError, match="claves desconocidas"):
        U.utility_costs("USD", prices=dict(MINE, steam_per_tonne=20.0), steam=[MP])
    # a misspelt optional key is refused, not silently dropped from the cost
    with pytest.raises(ValueError, match="claves desconocidas.*fan_power_kwh_per_m3"):
        U.utility_costs("USD", prices=MINE, cooling_water_spec=dict(CW, fan_power_kwh_per_m3=0.04))
    with pytest.raises(ValueError, match="claves desconocidas.*boiler_eff"):
        U.utility_costs("USD", prices=MINE, steam=[dict(MP, boiler_eff=0.9)])
    with pytest.raises(ValueError, match="claves desconocidas.*tou"):
        U.utility_costs("USD", prices=MINE, electricity={"tou": {}})


def test_user_prices_never_touch_the_bank():
    out = U.utility_costs("USD", prices=MINE, steam=[MP], cooling_water_spec=CW, refrigeration=[R20], electricity={}, today=TODAY)
    assert all(v["origin"] == "usuario" for v in out["price_sources"].values())
    assert not any(U.REFERENCE_FLAG in w for w in out["warnings"])
    assert out["steam"][0]["breakdown_per_tonne"]["fuel"] == pytest.approx(5.0 * out["steam"][0]["fuel_GJ_per_tonne"])
    assert out["inputs"]["prices"] == MINE and out["inputs"]["reference"] is None
    json.dumps(out)


def test_reference_is_flagged_first_with_source_period_and_age():
    water = {"boiler_makeup_water_per_m3": 1.5, "cooling_makeup_water_per_m3": 0.6}
    out = U.utility_costs("USD", prices=water, reference="US-EIA-2026-08", steam=[MP], cooling_water_spec=CW, electricity={},
                          today=TODAY)
    first = out["warnings"][0]
    assert first.startswith(U.REFERENCE_FLAG)
    assert "U.S. EIA" in first and "periodo 2026-08" in first and "75 días" in first and "2026-11-14" in first
    assert "4.34 USD/Mcf" in first and "9.64 centavos" in first
    src = out["price_sources"]
    assert src["fuel_per_GJ"]["origin"] == "referencia US-EIA-2026-08" and src["fuel_per_GJ"]["age_days"] == 75
    assert src["fuel_per_GJ"]["url"].startswith("https://www.eia.gov/") and src["fuel_per_GJ"]["fuel_basis"] == "HHV"
    assert src["boiler_makeup_water_per_m3"]["origin"] == "usuario"
    assert out["electricity"]["per_kWh"] == pytest.approx(0.0964)
    old = U.utility_costs("USD", prices=water, reference="US-EIA-2026-08", steam=[MP], today="2028-03-01")
    assert "más de un año" in old["warnings"][0]


def test_user_price_wins_over_the_reference_key_by_key():
    p = {"fuel_per_GJ": 7.0, "fuel_basis": "LHV", "boiler_makeup_water_per_m3": 1.5, "cooling_makeup_water_per_m3": 0.6}
    out = U.utility_costs("USD", prices=p, reference="US-EIA-2026-08", steam=[MP], cooling_water_spec=CW, today=TODAY)
    src = out["price_sources"]
    assert src["fuel_per_GJ"]["origin"] == "usuario" and src["fuel_per_GJ"]["value"] == 7.0
    assert src["electricity_per_kWh"]["origin"] == "referencia US-EIA-2026-08"
    assert "fuel_per_GJ" not in out["warnings"][0] and "electricity_per_kWh" in out["warnings"][0]
    assert out["steam"][0]["floor_fuel_per_GJ_over_efficiency"] == pytest.approx(7.0 / 0.82)
    everything = U.utility_costs("USD", prices=MINE, reference="US-EIA-2026-08", steam=[MP], cooling_water_spec=CW, today=TODAY)
    assert "no se usó ningún precio de referencia" in everything["warnings"][0]
    assert all(v["origin"] == "usuario" for v in everything["price_sources"].values())


def test_currency_is_explicit_and_never_converted_by_default():
    water = {"boiler_makeup_water_per_m3": 1.5}
    with pytest.raises(ValueError, match="tipo de cambio por defecto"):
        U.utility_costs("USD", prices=water, reference="EU27-EUROSTAT-2026-S1", steam=[MP], today=TODAY)
    out = U.utility_costs("USD", prices=water, reference="EU27-EUROSTAT-2026-S1", steam=[MP], exchange_rates={"EUR": 1.10}, today=TODAY)
    assert out["price_sources"]["fuel_per_GJ"]["value"] == pytest.approx(14.3195 * 1.10)
    assert out["price_sources"]["fuel_per_GJ"]["exchange_rate_applied"] == 1.10
    eur = U.utility_costs("EUR", prices=water, reference="EU27-EUROSTAT-2026-S1", steam=[MP], today=TODAY)
    assert eur["prices"] is None and "tipo de cambio por defecto" in eur["prices_note"]
    eur2 = U.utility_costs("EUR", prices=water, reference="EU27-EUROSTAT-2026-S1", steam=[MP], exchange_rates={"USD": 0.9}, today=TODAY)
    assert eur2["prices"]["heating_usd_GJ"] == pytest.approx(eur2["steam"][0]["cost_per_GJ_delivered"] / 0.9)


def test_mexican_tariff_needs_the_users_load_profile_and_load_factor():
    with pytest.raises(ValueError, match="time_of_use_fractions"):
        U.utility_costs("MXN", reference="MX-CFE-GDMTH-ORIENTE-2026-11", electricity={}, today=TODAY)
    frac = {"base": 0.30, "intermedia": 0.55, "punta": 0.15}
    out = U.utility_costs("MXN", reference="MX-CFE-GDMTH-ORIENTE-2026-11", electricity={"time_of_use_fractions": frac}, today=TODAY)
    energy = 0.30 * 0.9018 + 0.55 * 1.5799 + 0.15 * 1.7739
    assert out["electricity"]["energy_per_kWh"] == pytest.approx(energy) and out["electricity"]["demand_adder_per_kWh"] == 0.0
    assert any("Cargo por demanda" in x and "load_factor" in x for x in out["not_included"])
    assert out["warnings"][0].startswith(U.REFERENCE_FLAG) and "CFE" in out["warnings"][0]
    full = U.utility_costs("MXN", reference="MX-CFE-GDMTH-ORIENTE-2026-11",
                           electricity={"time_of_use_fractions": frac, "load_factor": 0.8}, hours_per_year=8000, today=TODAY)
    assert full["electricity"]["demand_adder_per_kWh"] == pytest.approx(12 * 548.65 / (0.8 * 8000))
    assert full["electricity"]["per_kWh"] == pytest.approx(energy + 12 * 548.65 / 6400)
    assert full["electricity"]["per_GJ"] == pytest.approx(full["electricity"]["per_kWh"] * 1e9 / 3.6e6)
    with pytest.raises(ValueError, match="hours_per_year"):
        U.utility_costs("MXN", reference="MX-CFE-GDMTH-ORIENTE-2026-11",
                        electricity={"time_of_use_fractions": frac, "load_factor": 0.8}, today=TODAY)
    with pytest.raises(ValueError, match="sumen 1"):
        U.utility_costs("MXN", reference="MX-CFE-GDMTH-ORIENTE-2026-11",
                        electricity={"time_of_use_fractions": {"base": 0.5, "intermedia": 0.2, "punta": 0.2}}, today=TODAY)
    usd = U.utility_costs("USD", reference="MX-CFE-GDMTH-ORIENTE-2026-11", electricity={"time_of_use_fractions": frac},
                          exchange_rates={"MXN": 0.05}, today=TODAY)
    assert usd["prices"]["electricity_usd_kWh"] == pytest.approx(energy * 0.05)
    # the user's flat price wins over the reference's time-of-use prices
    mine = U.utility_costs("MXN", prices={"electricity_per_kWh": 2.0}, reference="MX-CFE-GDMTH-ORIENTE-2026-11", electricity={}, today=TODAY)
    assert mine["electricity"]["energy_per_kWh"] == 2.0 and mine["electricity"]["origin"] == "usuario"


# ------------------------------------------------------------------------------------------------- chain and wiring
def test_chain_to_annual_cost_and_level_selection():
    out = U.utility_costs("USD", prices=MINE, steam=[MP, dict(MP, name="lp", pressure_Pa=4e5)], cooling_water_spec=CW,
                          refrigeration=[dict(R20, heat_rejection="cooling_water")], electricity={}, today=TODAY)
    assert set(out["prices"]) == {"electricity_usd_kWh"}                          # two heating and two cooling levels: ambiguous
    assert sum("elige cuál va a annual_cost" in w for w in out["warnings"]) == 2
    assert set(out["levels_per_GJ"]["heating"]) == {"mp", "lp"} and set(out["levels_per_GJ"]["cooling"]) == {"r-20", "cooling_water"}
    cw_gj = out["cooling_water"]["cost_per_GJ_removed"]
    r = out["refrigeration"][0]
    assert r["breakdown_per_GJ_removed"]["heat_rejection"] == pytest.approx(cw_gj * r["heat_rejected_GJ_per_GJ_removed"])
    assert out["levels_per_GJ"]["cooling"]["r-20"] > cw_gj
    out = U.utility_costs("USD", prices=MINE, steam=[MP, dict(MP, name="lp", pressure_Pa=4e5)], cooling_water_spec=CW,
                          refrigeration=[R20], electricity={}, annual_cost_levels={"heating": "lp", "cooling": "cooling_water"},
                          today=TODAY)
    p = out["prices"]
    assert p == {"electricity_usd_kWh": pytest.approx(0.08), "heating_usd_GJ": pytest.approx(out["steam"][1]["cost_per_GJ_delivered"]),
                 "cooling_usd_GJ": pytest.approx(cw_gj)}
    a = annual_cost({"E-1": 1e5}, {"P-1": ("electricity", 50.0), "E-1": ("heating", 1000.0), "E-2": ("cooling", 800.0)}, p,
                    hours_per_year=8000, interest=0.1, years=10)
    expect = 50 * 8000 * 0.08 + 1000 * 8000 * 3.6e-3 * p["heating_usd_GJ"] + 800 * 8000 * 3.6e-3 * p["cooling_usd_GJ"]
    assert a["operating_cost_usd_y"] == pytest.approx(expect)
    with pytest.raises(ValueError, match="no es un nivel calculado"):
        U.utility_costs("USD", prices=MINE, steam=[MP], annual_cost_levels={"heating": "hp"}, today=TODAY)
    with pytest.raises(ValueError, match="name distinto"):
        U.utility_costs("USD", prices=MINE, steam=[MP, MP], today=TODAY)


def test_second_route_is_reported_as_a_contrast_not_as_validation():
    uv = {"cepci": 800.0, "basis": "grass_roots", "boiler_fuel_price_usd_GJ": 5.0, "power_fuel_price_usd_GJ": 5.0,
          "steam_capacity_kg_s": 20.0, "cooling_water_capacity_m3_s": 5.0, "refrigeration_capacity_kW": 500.0}
    out = U.utility_costs("USD", prices=MINE, steam=[MP], cooling_water_spec=CW, refrigeration=[R20], electricity={},
                          ulrich_vasudevan_spec=uv, annual_cost_levels={"cooling": "cooling_water"}, today=TODAY)
    s2 = out["second_route"]
    assert "no de validación" in s2["note"] and "abril de 2006" in s2["source"] and s2["skipped"] == []
    st = s2["steam"][0]
    assert st["pressure_barg"] == pytest.approx((11e5 - 101325) / 1e5)
    assert st["price_per_tonne"] == pytest.approx(1000 * (2.3e-5 * 20 ** -0.9 * 800 + 0.0034 * st["pressure_barg"] ** 0.05 * 5.0))
    assert st["price_per_tonne"] > out["steam"][0]["cost_per_tonne"]             # a price with capital sits above the operating cost
    assert s2["refrigeration"][0]["price_per_GJ_removed"] == pytest.approx(1e6 * s2["refrigeration"][0]["price"])
    assert s2["cooling_water"]["price_per_GJ_removed"] == pytest.approx(s2["cooling_water"]["price"] / out["cooling_water"]["heat_removed_GJ_per_m3"])
    hot = U.utility_costs("USD", prices=MINE, steam=[dict(MP, pressure_Pa=80e5)], ulrich_vasudevan_spec=uv, today=TODAY)
    assert hot["second_route"]["steam"] == [] and "fuera del intervalo" in hot["second_route"]["skipped"][0]
    with pytest.raises(ValueError, match="cepci"):
        U.utility_costs("USD", prices=MINE, steam=[MP], ulrich_vasudevan_spec={"basis": "grass_roots"}, today=TODAY)


def test_tools_are_registered_with_spanish_descriptions():
    from packages.agent import agent as AG
    from packages.dwsim_client import tools as T
    names = {f.__name__ for f in T.TOOLS}
    assert {"utility_costs", "list_utility_price_references"} <= names and len(T.TOOLS) >= 73
    for name in ("utility_costs", "list_utility_price_references"):
        assert any(getattr(t, "name", "") == name for t in AG.AGENT_TOOLS)
    assert "no tu contrato" in TU.utility_costs.__doc__ and "nunca hay tipo de cambio por defecto" in TU.utility_costs.__doc__
    assert "Solo se usa" in TU.utility_costs.__doc__ and "PUBLICADOS" in TU.list_utility_price_references.__doc__
    out = TU.utility_costs("USD", prices=MINE, fired_heat={"furnace_efficiency": 0.8}, today=TODAY)
    assert out["prices"] == {"heating_usd_GJ": pytest.approx(6.25)} and math.isfinite(out["fired_heat"]["cost_per_GJ_absorbed"])
    assert "26. Utility costs" in AG.SYSTEM and "not their contract" in AG.SYSTEM
