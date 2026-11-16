"""Steam diagnostic (bloque M19): `steam_system_diagnostic` and `steam_diagnostic_report`.

Published numbers reproduced here were read on 2026-11-14 (curl + pdftotext of the PDFs on energy.gov; GitHub API for MEASUR):
  * U.S. DOE Steam Tip Sheet #3 «Use Feedwater Economizers for Waste Heat Recovery», DOE/GO-102012-3412, January 2012.
  * U.S. DOE Steam Tip Sheet #4 «Improve Your Boiler's Combustion Efficiency», DOE/GO-102012-3405, January 2012.
  * U.S. DOE Steam Tip Sheet #8 «Return Condensate to the Boiler», DOE/GO-102012-3410, January 2012.
  * U.S. DOE Steam Tip Sheet #9 «Minimize Boiler Blowdown», DOE/GO-102012-3406, January 2012.
  * U.S. DOE Steam Tip Sheet #10 «Recover Heat from Boiler Blowdown», DOE/GO-102012-3408, January 2012.
  * ORNL-AMO/MEASUR-Tools-Suite (branch develop), tests/cpp/steamModeler/Boiler.unit.cpp, second case.
The tip sheets do not say which gas or which air they used: pure methane and dry air at 70 °F are OUR reading (the one of
the M18 tests). Where a tip sheet states «80 % efficient boiler», the stack temperature that gives exactly 0.80 is solved
for: that reproduces their INPUT, not their answer. No network, no DWSIM."""
from __future__ import annotations

import copy
import json

import pytest
from scipy.optimize import brentq

from packages.costing import utilities as U
from packages.dwsim_client import tools_combustion as TCB
from packages.dwsim_client import tools_steam as TS
from packages.dwsim_client import tools_utilities as TU
from packages.energy import combustion as CB
from packages.energy import steam_diagnostic as SD

TODAY = "2026-11-14"
MMBTU = 1.05505585262                       # GJ per million Btu (IT)
LB = 0.45359237
PSI = 6894.757293168
PATM = 101325.0
GAL_PER_M3 = 264.172052
P150, P100 = (150 + 14.696) * PSI, (100 + 14.696) * PSI


def F(t: float) -> float:
    return (t + 459.67) / 1.8


CH4 = {"type": "gas", "composition": {"CH4": 1.0}, "T_K": F(70)}
DRY = {"T_K": F(70), "humidity_ratio_kg_kg": 0.0}
USD_8 = {"price_per_GJ": 8.0 / MMBTU, "basis": "HHV"}


def doe_combustion(excess: float, stack_T: float) -> dict:
    return {"site_pressure_Pa": PATM, "air": DRY, "excess_air_fraction": excess, "stack_T_K": stack_T,
            "setting_loss_fraction": 0.0, "setting_loss_basis": "HHV"}


@pytest.fixture(scope="module")
def stack_80() -> float:
    """Stack temperature at which methane with 15 % excess dry air gives exactly 80 % (HHV) with no setting loss."""
    f = lambda T: CB.combustion_efficiency(CH4, DRY, PATM, excess_air_fraction=0.15, stack_T_K=T, setting_loss_fraction=0.0,  # noqa: E731
                                           setting_loss_basis="HHV", adiabatic_flame=False)["efficiency"]["HHV"]["efficiency"] - 0.80
    return brentq(f, 350.0, 900.0, xtol=1e-10)


def measure(out: dict, name: str) -> dict:
    return next(m for m in out["measures"] if m["id"] == name)


# An invented plant, used only to exercise the code paths (not a client, not a validation).
NG = {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 298.15}
PLANT = dict(
    currency="MXN", fuel=NG,
    combustion={"site_pressure_Pa": 95000.0, "air": {"T_K": 298.15, "relative_humidity": 0.5},
                "flue_O2": {"mole_fraction": 0.07, "basis": "dry"}, "stack_T_K": 493.15, "setting_loss_fraction": 0.015,
                "setting_loss_basis": "HHV"},
    boiler={"steam_pressure_Pa": 9.0e5, "blowdown_fraction": 0.08, "auxiliary_power_kWh_per_tonne": 6.0},
    water={"feedwater_heating": "deaerator_steam", "feedwater_T_K": 378.15, "makeup_water_T_K": 293.15,
           "condensate_return_fraction": 0.4, "condensate_T_K": 353.15},
    fuel_cost={"price_per_GJ": 150.0, "basis": "HHV"}, water_cost={"price_per_m3": 45.0},
    electricity_cost={"price_per_kWh": 2.4},
    operation={"hours_per_year": 7200.0, "steam_generated_kg_s": 2.0},
    measures={"reduce_excess_air": {"target_flue_O2": {"mole_fraction": 0.03, "basis": "dry"}, "investment": 350000.0},
              "reduce_stack_temperature": {"target_stack_T_K": 423.15, "device": "economizer"},
              "reduce_blowdown": {"target_blowdown_fraction": 0.05},
              "recover_blowdown_heat": {"flash_pressure_Pa": 1.3e5, "drain_T_K": 313.15},
              "increase_condensate_return": {"target_condensate_return_fraction": 0.7}},
    today=TODAY)


def plant(**changes) -> dict:
    kw = copy.deepcopy(PLANT)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(kw.get(k), dict) and k != "measures":
            kw[k] = {**kw[k], **v}
        else:
            kw[k] = v
    return kw


@pytest.fixture(scope="module")
def full() -> dict:
    return TS.steam_system_diagnostic(**PLANT)


# ------------------------------------------------------------------------------------------------- published checks
def test_doe_tip_sheet_4_combustion_tuning_example():
    """500 000 MMBtu/yr, 8 000 h/yr; 44.9 % excess air and flue − air = 400 °F → 9.5 % and 300 °F. DOE: E1 78.2, E2 83.1,
    29 482 MMBtu/yr, 235 856 USD. The engine's efficiencies are 0.4-0.6 points above the table (the M18 offset), so the
    saving comes out 3.7 % lower. Not tuned."""
    out = TS.steam_system_diagnostic(
        "USD", CH4, doe_combustion(0.449, F(470)), {"steam_pressure_Pa": P150, "blowdown_fraction": 0.0},
        {"feedwater_heating": "not_charged", "feedwater_T_K": F(117), "makeup_water_T_K": F(60), "condensate_return_fraction": 0.0},
        USD_8, {"price_per_m3": 0.0},
        operation={"hours_per_year": 8000, "fuel_consumption": {"energy_GJ_per_year": 500000 * MMBTU, "basis": "HHV"}},
        measures={"reduce_excess_air": {"target_excess_air_fraction": 0.095, "stack_T_K_after": F(370)}}, today=TODAY)
    m = measure(out, "reduce_excess_air")
    e1, e2 = out["baseline"]["boiler"]["efficiency_HHV"], m["result"]["boiler"]["efficiency_HHV"]
    assert 100 * e1 == pytest.approx(78.2, abs=0.6) and 100 * e2 == pytest.approx(83.1, abs=0.6)
    saved = m["savings"]["annual"]["fuel_GJ"] / MMBTU
    assert saved == pytest.approx(500000 * (1 - e1 / e2), rel=1e-9)               # the tip sheet's own formula, our efficiencies
    assert saved == pytest.approx(28383.0, rel=1e-4) and saved / 29482 - 1 == pytest.approx(-0.0373, abs=0.001)
    assert m["savings"]["annual"]["cost"]["total"] == pytest.approx(227064.0, rel=1e-4)
    assert out["baseline"]["derived_quantity"].startswith("producción de vapor") and out["evidence"]["verified"]


def test_doe_tip_sheet_3_economizer_example_and_its_inconsistency_with_tip_sheet_4():
    """45 000 lb/h of 150 psig steam, 117 °F feed water, stack 500 → 250 °F, 15 % excess air, 8 400 h/yr, 8 USD/MMBtu.
    DOE: 50 MMBtu/h output, 4.6 MMBtu/h recoverable (table), 386 400 USD/yr. The engine gives 286 224 USD (−25.9 %). The
    difference is the tip sheet's: its own combustion table (#4) puts the same change at 6.0 efficiency points (296 887
    USD), 3.6 % above the engine, while 4.6 MMBtu/h ÷ 0.80 is 9.2 % of the fuel."""
    out = TS.steam_system_diagnostic(
        "USD", CH4, doe_combustion(0.15, F(500)), {"steam_pressure_Pa": P150, "blowdown_fraction": 0.0},
        {"feedwater_heating": "not_charged", "feedwater_T_K": F(117), "makeup_water_T_K": F(60), "condensate_return_fraction": 0.0},
        USD_8, {"price_per_m3": 0.0}, operation={"hours_per_year": 8400, "steam_generated_kg_s": 45000 * LB / 3600},
        measures={"reduce_stack_temperature": {"target_stack_T_K": F(250), "device": "economizer"}}, today=TODAY)
    b, m = out["baseline"], measure(out, "reduce_stack_temperature")
    duty = b["annual"]["steam_delivered_tonne"] * b["steam"]["boiler_duty_GJ_per_tonne"] / MMBTU / 8400
    assert duty == pytest.approx(50.0, rel=0.002)                                 # «Boiler heat output = 50 million Btu/hr»
    assert 100 * b["boiler"]["efficiency_HHV"] == pytest.approx(80.0, abs=0.2)    # «An 80% efficient boiler»
    usd = m["savings"]["annual"]["cost"]["total"]
    assert usd == pytest.approx(286224.0, rel=1e-4) and usd / 386400 - 1 == pytest.approx(-0.259, abs=0.002)
    # the same change read from the table of tip sheet #4 (15 % excess air; 430 °F interpolated, 180 °F extrapolated 20 °F)
    e_500, e_250 = 80.4 + 0.3 * (77.9 - 80.4), 85.2 + 0.2 * (85.2 - 82.8)
    by_tip_4 = 50.0 / (e_500 / 100) * (1 - e_500 / e_250) * 8400 * 8.0
    assert by_tip_4 == pytest.approx(296887, rel=1e-4) and usd / by_tip_4 - 1 == pytest.approx(-0.036, abs=0.002)
    assert m["checks"]["margin_over_water_dew_point_K"] > 60 and m["checks"]["feedwater_T_after_economizer_K"] < b["steam"]["saturation_T_K"]


def test_doe_tip_sheet_9_blowdown_example(stack_80):
    """100 000 lb/h, 150 psig, blow-down 8 % → 6 % of the feed water, make-up 60 °F, 80 %, 8 760 h, 8 USD/MMBtu, 0.004
    USD/gal. DOE: fuel 62 913, water 9 718, total 72 631 USD/yr."""
    out = TS.steam_system_diagnostic(
        "USD", CH4, doe_combustion(0.15, stack_80), {"steam_pressure_Pa": P150, "blowdown_fraction": 0.08},
        {"feedwater_heating": "not_charged", "feedwater_T_K": F(60), "makeup_water_T_K": F(60), "condensate_return_fraction": 0.0},
        USD_8, {"price_per_m3": 0.004 * GAL_PER_M3}, operation={"hours_per_year": 8760, "steam_generated_kg_s": 100000 * LB / 3600},
        measures={"reduce_blowdown": {"target_blowdown_fraction": 0.06}}, today=TODAY)
    s = measure(out, "reduce_blowdown")["savings"]["annual"]
    assert out["baseline"]["boiler"]["efficiency"] == pytest.approx(0.80, abs=1e-9)
    assert s["cost"]["fuel"] == pytest.approx(62913, rel=1e-3) and s["cost"]["water"] == pytest.approx(9718, rel=1e-3)
    assert s["cost"]["total"] == pytest.approx(72631, rel=5e-4)
    assert s["makeup_water_m3"] * 1000 / LB / 8760 == pytest.approx(2313, rel=2e-3)   # «Makeup Water Savings = 2,313 lb/hr»


def test_doe_tip_sheet_10_blowdown_heat_recovery_example(stack_80):
    """50 000 lb/h, 150 psig, blow-down 3 200 lb/h (6.0 % of the feed), 90 % of the blow-down energy recovered into the
    make-up at 60 °F, 80 %, 8 000 h. DOE: 8 500 MMBtu/yr, 68 000 USD, with 1.7 MMBtu/h read from its rounded table. Their
    own formula with their own enthalpies (338.5 and 28 Btu/lb) gives 8 942 MMBtu/yr; the engine 8 940-8 957."""
    CP = U._cp()
    h_f, h_mu = CP.PropsSI("H", "P", P150, "Q", 0, "Water"), U._h_liq(F(60))
    T_drain = U._T_liq(h_mu + 0.10 * (h_f - h_mu))                                # «recovers 90% of the blowdown energy losses»
    got = []
    for mode, r, T_c, T_fw in (("mixing_only", 0.0, None, None), ("deaerator_steam", 0.0, None, F(227)),
                               ("deaerator_steam", 0.5, F(180), F(227))):
        out = TS.steam_system_diagnostic(
            "USD", CH4, doe_combustion(0.15, stack_80), {"steam_pressure_Pa": P150, "blowdown_fraction": 3200 / 53200},
            {"feedwater_heating": mode, "feedwater_T_K": T_fw, "makeup_water_T_K": F(60), "condensate_return_fraction": r,
             "condensate_T_K": T_c},
            USD_8, {"price_per_m3": 0.0}, operation={"hours_per_year": 8000, "steam_generated_kg_s": 50000 * LB / 3600},
            measures={"recover_blowdown_heat": {"drain_T_K": T_drain}}, today=TODAY)
        assert out["evidence"]["verified"]
        got.append(measure(out, "recover_blowdown_heat")["savings"]["annual"]["fuel_GJ"] / MMBTU)
    their_formula = 0.9 * 3200 * (338.5 - 28) / 0.80 * 8000 / 1e6
    assert their_formula == pytest.approx(8942.4)
    assert got[0] == pytest.approx(their_formula, rel=5e-4)                       # feed tank without heater: their control volume
    assert max(got) / min(got) - 1 < 0.003                                        # the feed-water arrangement barely matters
    assert got[0] / 8500 - 1 == pytest.approx(0.052, abs=0.002)                   # against the rounded table value


def test_doe_tip_sheet_8_condensate_return_example(stack_80):
    """10 000 lb/h more condensate at 180 °F with a 12 % flash loss (8 800 lb/h net), 8 000 h, 80 %, make-up 55 °F, 0.004
    USD/gal, 8 USD/MMBtu. DOE: water 33 760, fuel 88 000, total 121 760 USD/yr. The example gives neither the steam rate nor
    the pressure: the result must not depend on them."""
    totals = []
    for steam_lb_h, P, r0, beta in ((50000, P100, 0.3, 0.0), (100000, P150, 0.5, 0.05)):
        out = TS.steam_system_diagnostic(
            "USD", CH4, doe_combustion(0.15, stack_80), {"steam_pressure_Pa": P, "blowdown_fraction": beta},
            {"feedwater_heating": "mixing_only", "makeup_water_T_K": F(55), "condensate_return_fraction": r0, "condensate_T_K": F(180)},
            USD_8, {"price_per_m3": 0.004 * GAL_PER_M3}, operation={"hours_per_year": 8000, "steam_generated_kg_s": steam_lb_h * LB / 3600},
            measures={"increase_condensate_return": {"target_condensate_return_fraction": r0 + 8800 / steam_lb_h}}, today=TODAY)
        s = measure(out, "increase_condensate_return")["savings"]["annual"]["cost"]
        assert s["fuel"] == pytest.approx(88000, rel=5e-4) and s["water"] == pytest.approx(33760, rel=5e-4)
        totals.append(s["total"])
        assert out["evidence"]["verified"]
    assert totals[0] == pytest.approx(121760, rel=5e-4) and totals[1] == pytest.approx(totals[0], rel=1e-9)


def test_measur_boiler_unit_test_case():
    """MEASUR-Tools-Suite, Boiler(deaerator 0.3631 MPa, 72.4 %, blow-down 3.7 %, 5.5766 MPa, quality 1, 10 864 kg/h):
    getFuelEnergy() == 33344279.4789710045 kJ/h. Same boiler balance; MEASUR uses IAPWS-IF97, this engine IAPWS-95."""
    T_da = U._cp().PropsSI("T", "P", 0.3631e6, "Q", 0, "Water")
    lv = U.steam_level({"name": "s", "pressure_Pa": 5.5766e6, "boiler_efficiency": 0.724, "feedwater_T_K": T_da,
                        "condensate_return_fraction": 0.0, "blowdown_fraction": 0.037, "makeup_water_T_K": 293.15,
                        "feedwater_heating": "not_charged"}, 1.0, 0.0)
    assert lv["fuel_GJ_per_tonne"] * 1e6 * 10.864 == pytest.approx(33344279.4789710045, rel=1e-4)
    assert lv["mass_per_kg_delivered"]["feedwater"] * 10864 == pytest.approx(10864 / (1 - 0.037), rel=1e-12)


# ------------------------------------------------------------------------------------------------- consistency
def test_baseline_equals_a_direct_call_of_the_two_tools(full):
    c = PLANT["combustion"]
    comb = TCB.combustion_efficiency(NG, c["air"], c["site_pressure_Pa"], flue_O2=c["flue_O2"], stack_T_K=c["stack_T_K"],
                                     setting_loss_fraction=c["setting_loss_fraction"], setting_loss_basis="HHV",
                                     fuel_price_basis="HHV")
    eta = comb["for_utility_costs"]["efficiency_to_pass"]
    cost = TU.utility_costs("MXN", {"fuel_per_GJ": 150.0, "fuel_basis": "HHV", "boiler_makeup_water_per_m3": 45.0,
                                    "electricity_per_kWh": 2.4},
                            steam=[{"name": "v", "pressure_Pa": 9.0e5, "superheat_K": 0.0, "boiler_efficiency": eta,
                                    "feedwater_T_K": 378.15, "condensate_return_fraction": 0.4, "blowdown_fraction": 0.08,
                                    "makeup_water_T_K": 293.15, "feedwater_heating": "deaerator_steam", "condensate_T_K": 353.15,
                                    "auxiliary_power_kWh_per_tonne": 6.0}], hours_per_year=7200.0, today=TODAY)["steam"][0]
    b = full["baseline"]
    assert b["boiler"]["efficiency"] == pytest.approx(eta, rel=1e-13)
    assert b["boiler"]["losses_fraction_of_heating_value"] == pytest.approx(comb["efficiency"]["HHV"]["losses_fraction_of_heating_value"])
    assert b["boiler"]["water_dew_point_K"] == pytest.approx(comb["flue_gas"]["water_dew_point_K"])
    for k in ("cost_per_tonne", "cost_per_GJ_delivered", "fuel_GJ_per_tonne", "makeup_m3_per_tonne"):
        assert b["steam"][k] == pytest.approx(cost[k], rel=1e-13), k
    assert b["steam"]["breakdown_per_tonne"] == pytest.approx(cost["breakdown_per_tonne"])
    delivered = 2.0 * 3.6 * 7200 / cost["mass_per_kg_delivered"]["generated"]
    assert b["annual"]["steam_delivered_tonne"] == pytest.approx(delivered) and b["annual"]["steam_generated_tonne"] == pytest.approx(51840.0)
    assert b["annual"]["cost"]["total"] == pytest.approx(delivered * cost["cost_per_tonne"])
    assert b["annual"]["fuel_GJ"] == pytest.approx(delivered * cost["fuel_GJ_per_tonne"])
    assert b["firing"]["heat_release_HHV_W"] * 7200 * 3600 / 1e9 == pytest.approx(b["annual"]["fuel_GJ"], rel=1e-10)
    # the two parts of the fuel that end up in water add to the efficiency
    assert b["boiler"]["useful_steam_heat_fraction_of_fuel"] + b["boiler"]["blowdown_heat_fraction_of_fuel"] == pytest.approx(eta, rel=1e-12)


def test_every_measure_moves_the_right_way_and_the_combined_case_is_not_additive(full):
    assert [m["id"] for m in full["measures"]] == list(SD.MEASURE_ORDER)
    assert all(m["status"] == "evaluated" for m in full["measures"])
    b = full["baseline"]
    for m in full["measures"]:
        s = m["savings"]
        assert s["annual"]["fuel_GJ"] > 0 and s["annual"]["cost"]["total"] > 0 and s["per_tonne"]["cost"] > 0, m["id"]
        assert s["annual"]["cost"]["total"] == pytest.approx(b["annual"]["cost"]["total"] - m["result"]["annual"]["cost"]["total"])
        assert m["result"]["annual"]["steam_delivered_tonne"] == pytest.approx(b["annual"]["steam_delivered_tonne"])   # same demand
        assert max(m["residuals"].values()) < 1e-9
    gain = {m["id"]: m["savings"]["efficiency_gain_points"] for m in full["measures"]}
    assert gain["reduce_excess_air"] > 1 and gain["reduce_stack_temperature"] > 2             # combustion-side measures
    assert all(abs(gain[k]) < 1e-9 for k in ("reduce_blowdown", "recover_blowdown_heat", "increase_condensate_return"))
    water = {m["id"]: m["savings"]["annual"]["makeup_water_m3"] for m in full["measures"]}
    assert water["reduce_blowdown"] > 0 and water["increase_condensate_return"] > 0 and water["recover_blowdown_heat"] > 0
    assert abs(water["reduce_excess_air"]) < 1e-6 and abs(water["reduce_stack_temperature"]) < 1e-6
    c = full["combined"]
    assert c["status"] == "evaluated" and c["measures"] == list(SD.MEASURE_ORDER)
    total, summed = c["savings"]["annual"]["cost"]["total"], c["sum_of_individual"]["annual"]["cost_total"]
    assert summed == pytest.approx(sum(m["savings"]["annual"]["cost"]["total"] for m in full["measures"]))
    assert max(m["savings"]["annual"]["cost"]["total"] for m in full["measures"]) < total < summed
    assert 0.5 < c["combined_over_sum"] < 1.0 and full["evidence"]["combined_not_above_sum"]["holds"]
    assert c["result"]["boiler"]["efficiency"] > max(m["result"]["boiler"]["efficiency"] for m in full["measures"])
    ev = full["evidence"]
    assert ev["verified"] and ev["worst_residual_fraction"] < 1e-9 and ev["annual_fuel_two_routes_difference_fraction"] < 1e-9
    json.dumps(full, allow_nan=False)


def test_payback_only_with_the_plants_investment(full):
    m = measure(full, "reduce_excess_air")
    assert m["simple_payback_years"] == pytest.approx(350000.0 / m["savings"]["annual"]["cost"]["total"])
    for name in SD.MEASURE_ORDER[1:]:
        e = measure(full, name)
        assert e["simple_payback_years"] is None and "falta la inversión" in e["payback_note"] and "No se estima" in e["payback_note"]
    assert full["combined"]["simple_payback_years"] is None and full["combined"]["investment"] is None
    kw = plant()
    for k, v in zip(SD.MEASURE_ORDER, (350e3, 2.5e6, 2e5, 9e5, 3e6)):
        kw["measures"][k]["investment"] = v
    out = TS.steam_system_diagnostic(**kw)
    assert out["combined"]["investment"] == pytest.approx(6.95e6)
    assert out["combined"]["simple_payback_years"] == pytest.approx(6.95e6 / out["combined"]["savings"]["annual"]["cost"]["total"])
    assert not [x for x in out["missing_data"] if "inversión" in x["item"]]


def test_a_measure_without_its_target_is_not_evaluated_and_says_what_is_needed():
    out = TS.steam_system_diagnostic(**plant(measures={"reduce_blowdown": {"target_blowdown_fraction": 0.05}}))
    for name in SD.MEASURE_ORDER:
        m = measure(out, name)
        if name == "reduce_blowdown":
            assert m["status"] == "evaluated"
            continue
        assert m["status"] == "not_evaluated" and m["reason"] == "no evaluada: falta el dato" and m["savings"] is None
        assert m["needs"] == SD.MEASURE_NEEDS[name]["plain"] and m["needs_parameter"].startswith(f"measures.{name}")
    assert out["combined"]["status"] == "not_applicable"
    prevents = [x["prevents"] for x in out["missing_data"]]
    assert sum(p.startswith("evaluar la medida") for p in prevents) == 4
    none = TS.steam_system_diagnostic(**plant(measures=None))
    assert none["combined"] is None and all(m["status"] == "not_evaluated" for m in none["measures"])


def test_missing_essentials_are_refused_once_with_everything_that_is_missing():
    with pytest.raises(ValueError) as e:
        TS.steam_system_diagnostic("MXN", NG, {"site_pressure_Pa": 95000.0, "air": {"T_K": 298.15, "relative_humidity": 0.5}},
                                   {"steam_pressure_Pa": 9e5}, {"feedwater_heating": "deaerator_steam"})
    msg = str(e.value)
    for needle in ("no se suponen", "combustion.flue_O2", "combustion.stack_T_K", "setting_loss_fraction", "blowdown_fraction",
                   "water.feedwater_T_K", "makeup_water_T_K", "condensate_return_fraction", "fuel_cost"):
        assert needle in msg, needle
    with pytest.raises(ValueError, match="condensate_T_K"):
        TS.steam_system_diagnostic(**plant(water={"condensate_T_K": None}))
    with pytest.raises(ValueError, match="fuel_cost.basis"):
        TS.steam_system_diagnostic(**plant(fuel_cost={"basis": None}))
    with pytest.raises(ValueError, match="electricity_per_kWh"):                 # auxiliary power given, no electricity price
        TS.steam_system_diagnostic(**plant(electricity_cost=None))
    with pytest.raises(ValueError, match="claves desconocidas"):
        TS.steam_system_diagnostic(**plant(boiler={"eficiencia": 0.8}))
    with pytest.raises(ValueError, match="fuel_per_GJ"):                         # the Mexican reference has no natural gas
        TS.steam_system_diagnostic(**plant(fuel_cost=None, reference="MX-CFE-GDMTH-GOLFO-NORTE-2026-11",
                                           electricity_cost={"time_of_use_fractions": {"base": 0.3, "intermedia": 0.6, "punta": 0.1}}))


def test_without_operating_data_only_per_tonne_figures(full):
    out = TS.steam_system_diagnostic(**plant(operation=None))
    assert out["baseline"]["annual"] is None and "firing" not in out["baseline"]
    assert out["baseline"]["steam"]["cost_per_tonne"] == pytest.approx(full["baseline"]["steam"]["cost_per_tonne"])
    m = measure(out, "reduce_excess_air")
    assert m["savings"]["annual"] is None and m["simple_payback_years"] is None and "cifras anuales" in m["payback_note"]
    assert m["savings"]["per_tonne"]["cost"] == pytest.approx(measure(full, "reduce_excess_air")["savings"]["per_tonne"]["cost"])
    assert any("cifras anuales" in x["prevents"] for x in out["missing_data"])
    assert out["combined"]["sum_of_individual"]["annual"] is None and out["evidence"]["verified"]
    with pytest.raises(ValueError, match="hours_per_year"):
        TS.steam_system_diagnostic(**plant(operation={"hours_per_year": None}))


def test_without_auxiliary_power_electricity_is_not_costed_and_it_is_listed(full):
    out = TS.steam_system_diagnostic(**plant(boiler={"auxiliary_power_kWh_per_tonne": None}))
    assert out["baseline"]["annual"]["electricity_kWh"] is None and out["baseline"]["steam"]["breakdown_per_tonne"]["power"] == 0.0
    assert out["baseline"]["steam"]["cost_per_tonne"] == pytest.approx(full["baseline"]["steam"]["cost_per_tonne"] - 6.0 * 2.4)
    assert any("eléctrico" in x["item"] for x in out["missing_data"])
    assert any("no entró al costo" in w for w in out["warnings"])


def test_stack_target_at_or_below_the_water_dew_point_is_not_evaluated():
    dew = TS.steam_system_diagnostic(**plant(measures=None))["baseline"]["boiler"]["water_dew_point_K"]
    assert 320 < dew < 335
    out = TS.steam_system_diagnostic(**plant(measures={"reduce_stack_temperature": {"target_stack_T_K": dew - 1.0}}))
    m = measure(out, "reduce_stack_temperature")
    assert m["status"] == "not_evaluated" and "punto de rocío del agua" in m["reason"] and m["savings"] is None
    assert any("rocío" in w for w in out["warnings"]) and any("ÁCIDO no se modela" in n for n in m["notes"])
    ok = measure(TS.steam_system_diagnostic(**plant(measures={"reduce_stack_temperature": {"target_stack_T_K": dew + 40.0}})),
                 "reduce_stack_temperature")
    assert ok["status"] == "evaluated" and ok["checks"]["margin_over_water_dew_point_K"] == pytest.approx(40.0)
    assert any("ÁCIDO no se modela" in n for n in ok["notes"])
    # an economizer cannot cool the gas below the water that enters it (feed water at 378 K)
    eco = measure(TS.steam_system_diagnostic(**plant(measures={"reduce_stack_temperature": {"target_stack_T_K": 370.0,
                                                                                            "device": "economizer"}})),
                  "reduce_stack_temperature")
    assert eco["status"] == "not_evaluated" and "economizador" in eco["reason"]
    sour = dict(NG, composition={"CH4": 0.94, "C2H6": 0.03, "N2": 0.02, "H2S": 0.01})
    m = measure(TS.steam_system_diagnostic(**plant(fuel=sour, measures={"reduce_stack_temperature": {"target_stack_T_K": 423.15}})),
                "reduce_stack_temperature")
    assert any("azufre" in n and "ATENCIÓN" in n for n in m["notes"])


def test_a_target_worse_than_today_is_reported_as_such():
    out = TS.steam_system_diagnostic(**plant(measures={"reduce_excess_air": {"target_flue_O2": {"mole_fraction": 0.09, "basis": "dry"}}}))
    m = measure(out, "reduce_excess_air")
    assert m["status"] == "evaluated" and m["savings"]["annual"]["cost"]["total"] < 0
    assert any("MÁS caro" in n for n in m["notes"]) and any("no ahorra" in w for w in out["warnings"])
    assert out["evidence"]["measures_direction"]["reduce_excess_air"]["saves"] is False


def test_prices_from_bills_and_consumption_from_bills(full):
    comb = TCB.combustion_efficiency(NG, PLANT["combustion"]["air"], 95000.0, excess_air_fraction=0.1)["fuel"]
    gj = 140000.0
    by_energy = TS.steam_system_diagnostic(**plant(fuel_cost={"price_per_GJ": None, "basis": None, "bill": {
        "amount": 150.0 * gj, "energy_GJ": gj, "basis": "HHV", "period": "2025-11 a 2026-10"}}))
    assert by_energy["prices"]["fuel_per_GJ"] == pytest.approx(150.0)
    assert by_energy["baseline"]["annual"]["cost"]["total"] == pytest.approx(full["baseline"]["annual"]["cost"]["total"])
    prov = by_energy["provenance"]["prices"][0]
    assert prov["origin"] == "calculado" and "2025-11 a 2026-10" in prov["detail"] and "precio medio del recibo" in prov["detail"]
    assert any("precio medio de los recibos" in a for a in by_energy["assumptions"])
    # a volume bill needs the reference conditions of the meter: 20 °C and 1 kgf/cm² here, given by the user
    ref = {"T_K": 293.15, "P_Pa": 98066.5}
    vol = gj * 1e9 / comb["HHV_J_mol"] * CB.R_GAS * ref["T_K"] / ref["P_Pa"]
    by_volume = TS.steam_system_diagnostic(**plant(fuel_cost={"price_per_GJ": None, "basis": None, "bill": {
        "amount": 150.0 * gj, "volume_m3": vol, "volume_reference": ref, "period": "2026"}}))
    assert by_volume["prices"]["fuel_per_GJ"] == pytest.approx(150.0, rel=1e-12) and by_volume["fuel_basis"] == "HHV"
    with pytest.raises(ValueError, match="volume_reference"):
        TS.steam_system_diagnostic(**plant(fuel_cost={"price_per_GJ": None, "basis": None,
                                                      "bill": {"amount": 1.0, "volume_m3": 10.0, "period": "x"}}))
    water = TS.steam_system_diagnostic(**plant(water_cost={"price_per_m3": None, "bill": {"amount": 90000.0, "volume_m3": 2000.0,
                                                                                         "period": "2026-10"}}))
    assert water["prices"]["boiler_makeup_water_per_m3"] == pytest.approx(45.0)
    # annual fuel taken from the bills: steam production is then the derived quantity
    from_bill = TS.steam_system_diagnostic(**plant(
        fuel_cost={"price_per_GJ": None, "basis": None, "bill": {"amount": 150.0 * gj, "energy_GJ": gj, "basis": "HHV", "period": "año"}},
        operation={"steam_generated_kg_s": None, "fuel_consumption": {"from_fuel_bill": True}}))
    assert from_bill["baseline"]["annual"]["fuel_GJ"] == pytest.approx(gj)
    assert from_bill["baseline"]["derived_quantity"].startswith("producción de vapor")
    assert any("otros equipos" in w for w in from_bill["warnings"])
    # a price on the lower heating value gives the same money
    lhv = TS.steam_system_diagnostic(**plant(fuel_cost={"price_per_GJ": 150.0 * comb["HHV_J_kg"] / comb["LHV_J_kg"], "basis": "LHV"}))
    assert lhv["fuel_basis"] == "LHV" and lhv["baseline"]["boiler"]["efficiency"] > full["baseline"]["boiler"]["efficiency"]
    assert lhv["baseline"]["annual"]["cost"]["total"] == pytest.approx(full["baseline"]["annual"]["cost"]["total"], rel=1e-10)
    assert lhv["baseline"]["annual"]["fuel_GJ_HHV"] == pytest.approx(full["baseline"]["annual"]["fuel_GJ"], rel=1e-10)


def test_steam_and_fuel_both_given_are_compared(full):
    fuel = full["baseline"]["annual"]["fuel_GJ"]
    with pytest.raises(ValueError, match="annual_basis"):
        TS.steam_system_diagnostic(**plant(operation={"fuel_consumption": {"energy_GJ_per_year": fuel, "basis": "HHV"}}))
    out = TS.steam_system_diagnostic(**plant(operation={"fuel_consumption": {"energy_GJ_per_year": fuel * 1.10, "basis": "HHV"},
                                                        "annual_basis": "steam"}))
    x = out["evidence"]["input_output_cross_check"]
    assert x["input_output_efficiency"] == pytest.approx(full["baseline"]["boiler"]["efficiency"] / 1.10)
    assert x["difference_points"] < -7 and x["steam_from_fuel_over_steam_measured"] == pytest.approx(1.10)
    assert out["baseline"]["annual"]["fuel_GJ"] == pytest.approx(fuel)                                # basis: steam
    by_fuel = TS.steam_system_diagnostic(**plant(operation={"fuel_consumption": {"energy_GJ_per_year": fuel * 1.10, "basis": "HHV"},
                                                            "annual_basis": "fuel"}))
    assert by_fuel["baseline"]["annual"]["fuel_GJ"] == pytest.approx(fuel * 1.10)


def test_feed_tank_without_heater_and_the_three_feedwater_modes(full):
    mix = TS.steam_system_diagnostic(**plant(water={"feedwater_heating": "mixing_only", "feedwater_T_K": 318.15}))
    s = mix["baseline"]["steam"]
    m = s["mass_per_kg_delivered"]
    h = lambda T: U._h_liq(T)                                                    # noqa: E731
    mixed = (0.4 * h(353.15) + m["makeup"] * h(293.15)) / m["feedwater"]         # independent energy balance of the tank
    assert h(s["feedwater_T_K"]) == pytest.approx(mixed, rel=1e-10) and m["deaerator_steam"] == pytest.approx(0.0, abs=1e-9)
    chk = mix["evidence"]["feed_tank_mixing_check"]
    assert chk["measured_minus_calculated_K"] == pytest.approx(318.15 - s["feedwater_T_K"]) and mix["evidence"]["verified"]
    # with blow-down heat recovery the tank is warmer and the fuel lower; flash steam returns water
    rec = measure(mix, "recover_blowdown_heat")
    assert rec["result"]["steam"]["feedwater_T_K"] > s["feedwater_T_K"] + 3 and rec["savings"]["annual"]["fuel_GJ"] > 0
    assert rec["result"]["steam"]["mass_per_kg_delivered"]["blowdown_flash_steam"] > 0 and max(rec["residuals"].values()) < 1e-9
    assert measure(mix, "increase_condensate_return")["result"]["steam"]["feedwater_T_K"] > s["feedwater_T_K"] + 10
    # heating not charged: blow-down heat recovery cannot be evaluated and condensate return only saves water
    nc = TS.steam_system_diagnostic(**plant(water={"feedwater_heating": "not_charged"}))
    assert measure(nc, "recover_blowdown_heat")["status"] == "not_evaluated" and "not_charged" in measure(nc, "recover_blowdown_heat")["reason"]
    cr = measure(nc, "increase_condensate_return")
    assert cr["savings"]["annual"]["cost"]["fuel"] == pytest.approx(0.0, abs=1e-6) and cr["savings"]["annual"]["cost"]["water"] > 0
    assert any("solo se cuenta el ahorro de AGUA" in n for n in cr["notes"]) and any("not_charged" in w for w in nc["warnings"])
    assert nc["baseline"]["annual"]["fuel_GJ"] == pytest.approx(full["baseline"]["annual"]["fuel_GJ"], rel=1e-10)   # same boiler duty


def test_utility_costs_blowdown_heat_recovery_closes_and_matches_an_independent_flash():
    CP = U._cp()
    base = dict(name="s", pressure_Pa=11.37e5, boiler_efficiency=0.8, feedwater_T_K=378.15, condensate_return_fraction=0.5,
                blowdown_fraction=0.08, makeup_water_T_K=293.15, feedwater_heating="deaerator_steam", condensate_T_K=353.15)
    plain = U.steam_level(base, 8.0, 1.0)
    fuels = {}
    for key, rec in (("hx", {"drain_T_K": 303.15}), ("flash", {"flash_pressure_Pa": 1.3e5}),
                     ("both", {"flash_pressure_Pa": 1.3e5, "drain_T_K": 303.15})):
        lv = U.steam_level({**base, "blowdown_heat_recovery": rec}, 8.0, 1.0)
        assert all(abs(v) < 1e-12 for k, v in lv["energy_balance"].items() if k.endswith("residual_fraction"))
        fuels[key] = lv["fuel_GJ_per_tonne"]
        m = lv["mass_per_kg_delivered"]
        assert m["makeup"] == pytest.approx(m["feedwater"] - m["condensate_return"] - m["deaerator_steam"] - m["blowdown_flash_steam"])
        if "flash_pressure_Pa" in rec:
            hf, hl, hg = (CP.PropsSI("H", "P", 11.37e5, "Q", 0, "Water"), CP.PropsSI("H", "P", 1.3e5, "Q", 0, "Water"),
                          CP.PropsSI("H", "P", 1.3e5, "Q", 1, "Water"))
            assert lv["blowdown_heat_recovery"]["flash_steam_fraction_of_blowdown"] == pytest.approx((hf - hl) / (hg - hl), rel=1e-12)
            # CoolProp's own isenthalpic flash gives the same vapour fraction
            assert CP.PropsSI("Q", "P", 1.3e5, "H", hf, "Water") == pytest.approx((hf - hl) / (hg - hl), rel=1e-9)
    assert fuels["both"] < fuels["hx"] < fuels["flash"] < plain["fuel_GJ_per_tonne"]
    assert "blowdown_heat_recovery" not in plain and any("sin recuperar su calor" in a for a in plain["assumptions"])
    with pytest.raises(ValueError, match="deaerator_steam"):
        U.steam_level({**base, "feedwater_heating": "not_charged", "blowdown_heat_recovery": {"drain_T_K": 303.15}}, 8.0, 1.0)
    with pytest.raises(ValueError, match="drain_T_K"):
        U.steam_level({**base, "blowdown_heat_recovery": {"drain_T_K": 290.0}}, 8.0, 1.0)       # colder than the make-up
    with pytest.raises(ValueError, match="flash_pressure_Pa"):
        U.steam_level({**base, "blowdown_heat_recovery": {"flash_pressure_Pa": 0.5e5}}, 8.0, 1.0)  # below the deaerator
    with pytest.raises(ValueError, match="no se suponen"):
        U.steam_level({**base, "blowdown_heat_recovery": {}}, 8.0, 1.0)


def test_a_named_reference_price_is_flagged_in_the_result_and_in_the_report():
    kw = plant(electricity_cost={"price_per_kWh": None, "time_of_use_fractions": {"base": 0.35, "intermedia": 0.55, "punta": 0.10},
                                 "load_factor": 0.8},
               reference="MX-CFE-GDMTH-GOLFO-NORTE-2026-11")
    out = TS.steam_system_diagnostic(**kw)
    assert out["warnings"][0].startswith("PRECIO DE REFERENCIA PUBLICADO, NO TU CONTRATO")
    refs = out["provenance"]["published_references"]
    assert len(refs) == 2 and all(r["origin"] == "referencia publicada" and "2026-11" in r["detail"] for r in refs)
    energy = 0.35 * 0.9904 + 0.55 * 1.5333 + 0.10 * 1.6634
    assert out["prices"]["electricity_per_kWh"] == pytest.approx(energy + 12 * 444.12 / (0.8 * 7200.0))
    rep = TS.steam_diagnostic_report(inputs=kw)
    assert rep["uses_published_reference_price"] and "REFERENCIA PUBLICADA, no el contrato de la planta" in rep["markdown"]
    assert "referencia publicada" in rep["markdown"] and "consultada el 2026-11-14" in rep["markdown"]
    assert not TS.steam_diagnostic_report(inputs=PLANT)["uses_published_reference_price"]


# ------------------------------------------------------------------------------------------------------ report
def test_report_is_deterministic_and_carries_only_engine_numbers(full):
    rep = TS.steam_diagnostic_report(inputs=PLANT, plant_label="Planta de prueba", notice="CASO DE PRUEBA")
    again = TS.steam_diagnostic_report(diagnostic=full, plant_label="Planta de prueba", notice="CASO DE PRUEBA")
    md = rep["markdown"]
    assert md == again["markdown"] and rep["integrity"]["rebuilt_from_inputs"] and again["integrity"]["checked_against_given_result"]
    assert rep["sections"] == ["Resumen", "1. Cómo opera hoy la caldera", "2. A dónde se va la energía del combustible",
                               "3. Medidas de ahorro", "4. Todas las medidas juntas", "5. Lo que no se evaluó y por qué",
                               "6. De dónde sale cada número", "7. Supuestos y límites", "8. Comprobaciones", "Glosario"]
    b, c = full["baseline"], full["combined"]
    assert md.startswith("# Diagnóstico de vapor — Planta de prueba") and "> **CASO DE PRUEBA**" in md
    for number in (f"{b['annual']['cost']['total']:,.0f} MXN al año", f"{100 * b['boiler']['efficiency']:.1f} %",
                   f"{c['savings']['annual']['cost']['total']:,.0f} MXN al año", f"{b['steam']['cost_per_tonne']:,.1f} MXN por tonelada",
                   f"{c['sum_of_individual']['annual']['cost_total']:,.0f} MXN al año"):
        assert number in md, number
    assert rep["headline"] == {"annual_cost": b["annual"]["cost"]["total"], "cost_per_tonne": b["steam"]["cost_per_tonne"],
                               "efficiency": b["boiler"]["efficiency"], "combined_annual_saving": c["savings"]["annual"]["cost"]["total"],
                               "combined_saving_per_tonne": c["savings"]["per_tonne"]["cost"]}
    # measures table ordered by annual saving, largest first
    table = md[md.index("## 3. Medidas de ahorro"):md.index("## 4.")]
    order = sorted(full["measures"], key=lambda m: -m["savings"]["annual"]["cost"]["total"])
    pos = [table.index(f"| {m['label']} |") for m in order]
    assert pos == sorted(pos)
    for m in full["measures"]:
        assert f"**{m['savings']['annual']['cost']['total']:,.0f}**" in table
    assert "0.6 años" in table and "sin cotización" in table
    assert "dato de la planta" in md and "calculado" in md and "combustion_efficiency (M18" in md and "utility_costs (M17" in md
    assert "nan" not in md.lower().replace("financiamiento", "").replace("manométricos", "") and "None" not in md
    client = md[:md.index("## 7.")]                                               # no parameter paths in the client sections
    assert not [w for w in ("measures.", "fuel_cost", "water_cost", "electricity_cost", "combustion.", "boiler.", "water.",
                                 "operation.", "e+0") if w in client]
    json.dumps(rep, allow_nan=False)


def test_report_refuses_a_result_that_is_not_the_engines(full):
    doctored = copy.deepcopy(full)
    doctored["combined"]["savings"]["per_tonne"]["cost"] *= 1.5
    with pytest.raises(ValueError, match="no coincide"):
        TS.steam_diagnostic_report(diagnostic=doctored)
    doctored = copy.deepcopy(full)
    doctored["baseline"]["boiler"]["efficiency"] = 0.90
    with pytest.raises(ValueError, match="baseline.efficiency"):
        TS.steam_diagnostic_report(diagnostic=doctored)
    with pytest.raises(ValueError, match="uno de los dos"):
        TS.steam_diagnostic_report()
    with pytest.raises(ValueError, match="uno de los dos"):
        TS.steam_diagnostic_report(diagnostic=full, inputs=PLANT)


def test_report_says_what_was_not_evaluated_and_degrades_without_annual_figures():
    rep = TS.steam_diagnostic_report(inputs=plant(measures={"reduce_blowdown": {"target_blowdown_fraction": 0.05}}, operation=None))
    md = rep["markdown"]
    assert rep["evaluated_measures"] == ["reduce_blowdown"] and len(rep["not_evaluated_measures"]) == 4
    assert "no hay cifras anuales" in md and "al año" not in md[md.index("## Resumen"):md.index("## 1.")]
    sec5 = md[md.index("## 5."):md.index("## 6.")]
    for name in SD.MEASURE_ORDER:
        if name != "reduce_blowdown":
            assert f"**{SD.MEASURE_LABEL[name]}**: no evaluada: falta el dato" in sec5 and SD.MEASURE_NEEDS[name]["plain"] in sec5
    assert "las horas de operación al año y la producción de vapor" in sec5 and "Solo se evaluó una medida" in md
    none = TS.steam_diagnostic_report(inputs=plant(measures=None))["markdown"]
    assert "**No se evaluó ninguna medida de ahorro**" in none and "No aplica: no se evaluó ninguna medida." in none


def test_tools_are_registered_with_spanish_descriptions():
    from packages.agent import agent as AG
    from packages.dwsim_client import tools as T
    names = [f.__name__ for f in T.TOOLS]
    assert {"steam_system_diagnostic", "steam_diagnostic_report"} <= set(names) and len(T.TOOLS) == 135 and len(set(names)) == 135
    for name in ("steam_system_diagnostic", "steam_diagnostic_report"):
        assert any(getattr(t, "name", "") == name for t in AG.AGENT_TOOLS)
    assert len(AG.AGENT_TOOLS) == 135
    doc = TS.steam_system_diagnostic.__doc__
    assert "Ningún dato de la planta, precio, objetivo ni inversión se supone" in doc and "punto de rocío ácido NO se modela" in doc
    assert "no evaluada: falta el dato" in doc and "Tip Sheets #3, #4, #8, #9 y #10" in doc
    assert "sin llamar a un modelo de lenguaje" in TS.steam_diagnostic_report.__doc__
    assert "28. Steam diagnostic of a plant" in AG.SYSTEM and "do not invent a target" in AG.SYSTEM
    assert "blowdown_heat_recovery" in TU.utility_costs.__doc__


def _unknown_loss() -> dict:
    kw = copy.deepcopy(PLANT)
    del kw["combustion"]["setting_loss_fraction"], kw["combustion"]["setting_loss_basis"]
    kw["combustion"]["setting_loss_unknown"] = True
    return kw


def test_unknown_setting_loss_uses_the_published_value_and_flags_it_everywhere(full):
    """Owner decision 2026-11-14: a plant that does not know its setting loss gets a published value, never silently."""
    R = SD.SETTING_LOSS_REFERENCE
    assert R["value"] == 0.010 and R["basis"] == "HHV" and "ORNL/TM-2001/263" in R["source"] and "4.2.5" in R["source"]
    kw = _unknown_loss()
    out = TS.steam_system_diagnostic(**kw)
    json.dumps(out)
    # same result as a plant that gives exactly the published number, so only the labelling differs
    same = TS.steam_system_diagnostic(**plant(combustion={"setting_loss_fraction": R["value"], "setting_loss_basis": "HHV"}))
    assert out["baseline"]["steam"]["cost_per_tonne"] == pytest.approx(same["baseline"]["steam"]["cost_per_tonne"], rel=1e-12)
    assert out["baseline"]["boiler"]["efficiency"] == pytest.approx(same["baseline"]["boiler"]["efficiency"], rel=1e-12)
    assert out["baseline"]["boiler"]["losses_fraction_of_heating_value"]["setting"] == pytest.approx(R["value"])
    # labelled: first warning, missing data, assumptions, provenance (published, not plant)
    assert out["warnings"][0].startswith(SD.SETTING_LOSS_FLAG) and R["url"] in out["warnings"][0]
    assert any("pérdida por paredes" in m["item"] and "supuesto" in m["prevents"] for m in out["missing_data"])
    assert any("NO es un dato de la planta" in a for a in out["assumptions"])
    sl = out["setting_loss"]
    assert sl["assumed"] is True and sl["origin"] == SD.REF and sl["reference"] == R["id"] and sl["limits"]
    pub = [p for p in out["provenance"]["published_references"] if p.get("kind") == "setting_loss"]
    assert len(pub) == 1 and pub[0]["origin"] == SD.REF and pub[0]["value"] == R["value"]
    assert not any("paredes" in p["item"] for p in out["provenance"]["plant_data"])
    # sensitivity: one more point of loss costs one point of efficiency and makes steam dearer
    sv = sl["sensitivity"]
    assert sv["setting_loss_fraction"] == pytest.approx(0.020) and sv["efficiency_change_points"] == pytest.approx(-1.0, abs=1e-9)
    assert sv["cost_per_tonne_change"] > 0 and 0.005 < sv["cost_change_fraction"] < 0.02
    two = TS.steam_system_diagnostic(**plant(combustion={"setting_loss_fraction": 0.020, "setting_loss_basis": "HHV"}))
    assert sv["cost_per_tonne"] == pytest.approx(two["baseline"]["steam"]["cost_per_tonne"], rel=1e-12)
    assert out["evidence"]["verified"] is True
    # a plant that gives its own number is not flagged
    assert full["setting_loss"] == {"origin": SD.PLANT, "assumed": False, "value": 0.015, "basis": "HHV"}
    assert not any(w.startswith(SD.SETTING_LOSS_FLAG) for w in full["warnings"])
    assert not any(p.get("kind") == "setting_loss" for p in full["provenance"]["published_references"])


def test_setting_loss_is_never_assumed_without_the_declaration():
    kw = _unknown_loss()
    del kw["combustion"]["setting_loss_unknown"]
    with pytest.raises(ValueError, match="setting_loss_unknown=true"):                    # still refused; the way out is named
        TS.steam_system_diagnostic(**kw)
    kw["combustion"]["setting_loss_unknown"] = False
    with pytest.raises(ValueError, match="setting_loss_fraction"):
        TS.steam_system_diagnostic(**kw)
    with pytest.raises(ValueError, match="también setting_loss_unknown"):                 # contradictory
        TS.steam_system_diagnostic(**plant(combustion={"setting_loss_unknown": True}))
    kw = _unknown_loss()
    kw["combustion"]["setting_loss_basis"] = "LHV"
    with pytest.raises(ValueError, match="la base la fija la publicación"):
        TS.steam_system_diagnostic(**kw)
    kw = _unknown_loss()
    kw["combustion"]["setting_loss_unknown"] = "yes"
    with pytest.raises(ValueError, match="true o false"):
        TS.steam_system_diagnostic(**kw)


def test_report_says_first_that_the_setting_loss_is_assumed(full):
    rep = TS.steam_diagnostic_report(inputs=_unknown_loss())
    md = rep["markdown"]
    assert rep["uses_assumed_setting_loss"] is True and rep["uses_published_reference_price"] is False
    notice = md.index("la pérdida por paredes de la caldera es un SUPUESTO")
    assert md.index("## Resumen") < notice < md.index("## 1.")
    assert "ORNL/TM-2001/263" in md and "Valores supuestos (no son datos de la planta)" in md
    assert "Valor supuesto de una publicación, no medido en la planta" in md             # in the table of losses too
    plain = TS.steam_diagnostic_report(inputs=PLANT)
    assert plain["uses_assumed_setting_loss"] is False and "SUPUESTO" not in plain["markdown"]


# ------------------------------------------------------------------------------------------------ bloque M20
BILL_EV = {"amount": 1.2e6, "period": "oct 2026", "energy_GJ": 8000.0, "basis": "HHV", "volume_m3": 208000.0,
           "volume_reference": {"T_K": 293.15, "P_Pa": 98066.5}}


def swap(**whole) -> dict:
    """PLANT with whole arguments replaced (plant() merges dicts key by key)."""
    return {**copy.deepcopy(PLANT), **whole}


def test_fitted_gas_against_the_reference_gases():
    """The numbers quoted in SD.FUEL_SURROGATE_NOTE are measured here: change both together."""
    from packages.energy import fuel_reference as FR
    errs = {g: FR.surrogate_error(g) for g in FR.REFERENCE_GASES}
    assert max(abs(e["efficiency_HHV_difference_points"]) for e in errs.values()) < 0.15
    assert max(abs(e["efficiency_LHV_difference_points"]) for e in errs.values()) < 0.05
    co2 = {g: 100 * e["co2_difference_fraction"] for g, e in errs.items()}
    assert -9.5 < co2["HIGH-CO2"] < -9.0
    assert all(-3.25 < v < -0.85 for g, v in co2.items() if g != "HIGH-CO2")
    assert "0.15 puntos" in SD.FUEL_SURROGATE_NOTE and "0.9 y 3.2 %" in SD.FUEL_SURROGATE_NOTE
    # the reference compositions are read from CoolProp, sum to one and are mostly methane
    gulf = FR.reference_gas("GULF-COAST")
    assert sum(gulf["composition"].values()) == pytest.approx(1.0) and gulf["composition"]["CH4"] == pytest.approx(0.965222)
    assert "CoolProp" in gulf["source"] and gulf["notes"]                                   # n-hexane lumped, and it says so
    # the fit is exact on the heating value, on both sides of pure methane
    h_m = FR.molar_heating_value({"CH4": 1.0}, "HHV")
    for hv in (0.95 * h_m, h_m, 1.08 * h_m):
        f = FR.fit_to_heating_value(hv, "HHV")
        assert FR.molar_heating_value(f["composition"], "HHV") == pytest.approx(hv, rel=1e-12)
    assert set(FR.fit_to_heating_value(0.95 * h_m, "HHV")["composition"]) == {"CH4", "N2"}
    with pytest.raises(ValueError, match="demasiado alto"):                                 # LP gas billed as natural gas
        FR.fit_to_heating_value(2.4 * h_m, "HHV")
    with pytest.raises(ValueError, match="demasiado bajo"):
        FR.fit_to_heating_value(0.6 * h_m, "HHV")


def test_unknown_gas_is_fitted_to_the_billed_heating_value_and_flagged():
    kw = swap(fuel={"type": "natural_gas_unknown", "T_K": 298.15}, fuel_cost={"bill": BILL_EV})
    out = TS.steam_system_diagnostic(**kw)
    json.dumps(out)
    fr = out["fuel_resolution"]
    assert fr["assumed"] is True and fr["method"] == "fit" and fr["origin"] == SD.CALC and set(fr["composition"]) == {"CH4", "C2H6"}
    assert out["evidence"]["fitted_heating_value_error"] < 1e-12 and out["evidence"]["verified"] is True
    assert out["prices"]["fuel_per_GJ"] == pytest.approx(1.2e6 / 8000.0)                   # energy rules the price
    assert out["warnings"][0].startswith("COMPOSICIÓN DEL GAS SUPUESTA") and "38.462 MJ/m³" in out["warnings"][0]
    assert any("análisis de composición del gas" in m["item"] for m in out["missing_data"])
    assert any(v["id"] == "fuel_composition" for v in out["assumed_values"])
    assert not any(p["item"] == "Combustible" for p in out["provenance"]["plant_data"])
    assert len(fr["sensitivity"]) == 1 and abs(fr["sensitivity"][0]["efficiency_change_points"]) < 0.5
    # same numbers as a plant that gives that composition itself
    same = TS.steam_system_diagnostic(**swap(fuel={"type": "gas", "composition": fr["composition"], "T_K": 298.15},
                                              fuel_cost={"price_per_GJ": 150.0, "basis": "HHV"}))
    assert out["baseline"]["boiler"]["efficiency"] == pytest.approx(same["baseline"]["boiler"]["efficiency"], rel=1e-12)
    assert same["fuel_resolution"] == {"origin": SD.PLANT, "assumed": False} and same["assumed_values"] == []
    # the supplier's heating value does the same job as the bill
    hv = TS.steam_system_diagnostic(**swap(fuel={"type": "natural_gas_unknown", "T_K": 298.15, "heating_value": {
        "value_J_m3": 8000.0e9 / 208000.0, "basis": "HHV", "volume_reference": {"T_K": 293.15, "P_Pa": 98066.5}}}))
    assert hv["fuel_resolution"]["composition"]["C2H6"] == pytest.approx(fr["composition"]["C2H6"], rel=1e-9)
    rep = TS.steam_diagnostic_report(inputs=kw)
    assert rep["uses_assumed_fuel_composition"] is True and "la composición del gas es un SUPUESTO" in rep["markdown"]
    assert "Valores supuestos (no son datos de la planta)" in rep["markdown"]


def test_unknown_gas_without_heating_value_uses_a_named_reference_gas():
    out = TS.steam_system_diagnostic(**swap(fuel={"type": "natural_gas_unknown", "T_K": 298.15}))
    fr = out["fuel_resolution"]
    assert fr["method"] == "reference" and fr["reference_gas"] == "GULF-COAST" and fr["origin"] == SD.REF
    assert len(fr["sensitivity"]) == 5 and max(abs(x["efficiency_change_points"]) for x in fr["sensitivity"]) < 0.5
    assert "GULF-COAST" in out["warnings"][0] and "Pide a tu proveedor" in out["warnings"][0]
    other = TS.steam_system_diagnostic(**swap(fuel={"type": "natural_gas_unknown", "T_K": 298.15, "reference_gas": "AMARILLO"}))
    assert other["fuel_resolution"]["reference_gas"] == "AMARILLO"
    with pytest.raises(ValueError, match="no existe"):
        TS.steam_system_diagnostic(**swap(fuel={"type": "natural_gas_unknown", "T_K": 298.15, "reference_gas": "X"}))
    with pytest.raises(ValueError, match="quita reference_gas"):
        TS.steam_system_diagnostic(**swap(fuel={"type": "natural_gas_unknown", "T_K": 298.15, "reference_gas": "AMARILLO"},
                                           fuel_cost={"bill": BILL_EV}))
    with pytest.raises(ValueError, match="fuel.T_K"):
        TS.steam_system_diagnostic(**swap(fuel={"type": "natural_gas_unknown"}))
    with pytest.raises(ValueError, match="natural_gas_unknown"):                            # no fuel at all: the way out is named
        TS.steam_system_diagnostic(**swap(fuel=None))


def test_co2_comes_from_the_carbon_balance(full):
    em = full["baseline"]["emissions"]
    assert em["carbon_balance_error"] < 1e-12 and full["evidence"]["carbon_balance_error"] < 1e-12
    ann = full["baseline"]["annual"]
    assert em["co2_tonne_per_year"] == pytest.approx(ann["fuel_kg"] * em["co2_kg_per_kg_fuel"] / 1000.0, rel=1e-12)
    methane = TS.steam_system_diagnostic(**swap(fuel={"type": "gas", "composition": {"CH4": 1.0}, "T_K": 298.15}))
    assert methane["baseline"]["emissions"]["co2_kg_per_kg_fuel"] == pytest.approx(44.0095 / 16.0425, rel=2e-4)
    for m in full["measures"]:
        if m["status"] == "evaluated":
            s = m["savings"]
            assert s["annual"]["co2_tonne"] == pytest.approx(s["annual"]["fuel_GJ"] * 1e9 / full["fuel"][full["fuel_basis"] + "_J_kg"]
                                                              * em["co2_kg_per_kg_fuel"] / 1000.0, rel=1e-9)
            assert s["annual"]["max_investment_for_payback"] == {str(n): n * s["annual"]["cost"]["total"] for n in (1, 2, 3)}
    rep = TS.steam_diagnostic_report(inputs=PLANT)["markdown"]
    assert "Emisiones directas de CO2" in rep and "Inversión máxima para recuperar en 2 años" in rep


def test_data_review_flags_but_never_refuses(full):
    assert full["data_review"]["flags"] == [] and "excess_air_high" in full["data_review"]["checks_run"]
    high = TS.steam_system_diagnostic(**plant(combustion={"flue_O2": {"mole_fraction": 0.115, "basis": "dry"}}))
    assert [f["id"] for f in high["data_review"]["flags"]] == ["excess_air_high"]
    assert any(w.startswith("REVISAR DATO:") for w in high["warnings"]) and high["evidence"]["verified"] is True
    cold = TS.steam_system_diagnostic(**plant(combustion={"stack_T_K": 430.0}, measures={}))
    assert [f["id"] for f in cold["data_review"]["flags"]] == ["stack_below_saturation"]
    purge = TS.steam_system_diagnostic(**plant(boiler={"blowdown_fraction": 0.14}, measures={}))
    assert [f["id"] for f in purge["data_review"]["flags"]] == ["blowdown_high"]
    # a bill with energy and volume is compared with the composition given
    off = TS.steam_system_diagnostic(**swap(fuel_cost={"bill": {**BILL_EV, "volume_m3": 190000.0}}, measures={}))
    assert off["evidence"]["billed_heating_value_check"]["difference_fraction"] > 0.02
    assert "billed_heating_value" in [f["id"] for f in off["data_review"]["flags"]]
    rep = TS.steam_diagnostic_report(inputs=plant(combustion={"flue_O2": {"mole_fraction": 0.115, "basis": "dry"}}))
    assert rep["data_flags"] == 1 and "Datos que conviene confirmar antes de decidir" in rep["markdown"]


def test_published_setting_loss_scales_with_the_declared_load():
    kw = _unknown_loss()
    kw["boiler"]["rated_steam_capacity_kg_s"] = 4.0                                        # production is 2.0 kg/s: 50 % load
    out = TS.steam_system_diagnostic(**kw)
    sl = out["setting_loss"]
    assert sl["load_fraction"] == pytest.approx(0.5) and sl["value"] == pytest.approx(0.020) and sl["full_load_value"] == 0.010
    assert "50 % de su capacidad nominal" in sl["value_choice"] and "2.00 %" in out["warnings"][0]
    assert out["baseline"]["boiler"]["losses_fraction_of_heating_value"]["setting"] == pytest.approx(0.020)
    # the plant's own figure is never rescaled
    own = TS.steam_system_diagnostic(**plant(boiler={"rated_steam_capacity_kg_s": 4.0}))
    assert own["baseline"]["boiler"]["losses_fraction_of_heating_value"]["setting"] == pytest.approx(0.015)
    low = TS.steam_system_diagnostic(**plant(boiler={"rated_steam_capacity_kg_s": 10.0}, measures={}))
    assert "low_load" in [f["id"] for f in low["data_review"]["flags"]]
    with pytest.raises(ValueError, match="capacidad nominal"):
        TS.steam_system_diagnostic(**plant(boiler={"rated_steam_capacity_kg_s": 1.0}))


def test_published_targets_only_by_name_and_only_for_natural_gas(full):
    pub = {"reduce_excess_air": {"use_published_target": True}, "reduce_stack_temperature": {"use_published_target": True}}
    out = TS.steam_system_diagnostic(**plant(measures=pub))
    json.dumps(out)
    air, stack = measure(out, "reduce_excess_air"), measure(out, "reduce_stack_temperature")
    assert air["status"] == stack["status"] == "evaluated"
    assert air["result"]["boiler"]["excess_air_fraction"] == pytest.approx(0.10, rel=1e-9)
    assert stack["result"]["boiler"]["stack_T_K"] == pytest.approx((250.0 - 32.0) / 1.8 + 273.15)
    assert "DOE/GO-102012-3405" in air["published_target"]["source"] and "DOE/GO-102012-3412" in stack["published_target"]["source"]
    assert any("NO lo dio la planta" in n for n in air["notes"]) and any("NO lo dio la planta" in n for n in stack["notes"])
    assert sum(w.startswith(SD.TARGET_FLAG) for w in out["warnings"]) == 2 and out["warnings"][0].startswith(SD.TARGET_FLAG)
    assert {v["id"] for v in out["assumed_values"]} == {"target_reduce_excess_air", "target_reduce_stack_temperature"}
    # same numbers as a plant that gives those targets itself; only the labelling differs
    own = TS.steam_system_diagnostic(**plant(measures={
        "reduce_excess_air": {"target_excess_air_fraction": 0.10},
        "reduce_stack_temperature": {"target_stack_T_K": (250.0 - 32.0) / 1.8 + 273.15, "device": "economizer"}}))
    assert out["combined"]["savings"]["per_tonne"]["cost"] == pytest.approx(own["combined"]["savings"]["per_tonne"]["cost"], rel=1e-12)
    assert own["assumed_values"] == [] and "published_target" not in measure(own, "reduce_excess_air")
    # not for a fuel the sources do not cover, not when already there, not together with the plant's own target
    lp = TS.steam_system_diagnostic(**swap(fuel={"type": "gas", "composition": {"C3H8": 0.6, "n-C4H10": 0.4}, "T_K": 298.15},
                                           measures=pub))
    assert all(measure(lp, k)["status"] == "not_evaluated" and "gas natural" in measure(lp, k)["reason"] for k in pub)
    tuned = TS.steam_system_diagnostic(**plant(combustion={"flue_O2": {"mole_fraction": 0.015, "basis": "dry"}}, measures=pub))
    assert "igual o\n" not in measure(tuned, "reduce_excess_air")["reason"] and "ya opera" in measure(tuned, "reduce_excess_air")["reason"]
    with pytest.raises(ValueError, match="no los dos"):
        TS.steam_system_diagnostic(**plant(measures={"reduce_excess_air": {"use_published_target": True,
                                                                           "target_excess_air_fraction": 0.12}}))
    with pytest.raises(ValueError, match="claves desconocidas"):
        TS.steam_system_diagnostic(**plant(measures={"reduce_blowdown": {"use_published_target": True}}))
    assert "Objetivo de la medida" in TS.steam_diagnostic_report(inputs=plant(measures=pub))["markdown"]


def test_the_data_checklist_names_every_input_of_the_engine():
    """docs/DATOS_PARA_DIAGNOSTICO.md is what the owner hands to a plant: it must not drift from the engine's inputs."""
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "DATOS_PARA_DIAGNOSTICO.md").read_text()
    keys = set(SD.COMBUSTION_KEYS + SD.BOILER_KEYS + SD.WATER_KEYS + SD.OPERATION_KEYS + SD.FUEL_COST_KEYS
               + SD.WATER_COST_KEYS + SD.ELECTRICITY_COST_KEYS + list(SD.MEASURE_ORDER) + list(SD.HEATING_MODES))
    for ks in SD.MEASURE_KEYS.values():
        keys |= set(ks)
    keys |= {"fuel", "fuel_cost", "water_cost", "electricity_cost", "reference", "exchange_rates", "energy_GJ_per_year",
             "volume_m3_per_year", "mass_kg_per_year", "from_fuel_bill", "volume_reference", "energy_GJ", "volume_m3", "mass_kg"}
    missing = sorted(k for k in keys if f"`{k}`" not in doc and f".{k}`" not in doc and f"`{k}" not in doc)
    assert not missing, missing
    assert "Si la planta no la conoce" in doc and "marcado como supuesto" in doc and "no evaluada: falta el dato" in doc and "la inversión nunca se estima" in doc
