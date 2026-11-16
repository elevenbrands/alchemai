"""Utilities capital (bloque M26-U): cooling tower, steam boiler + turbogenerator, wastewater treatment.

Every published number below was READ on 2026-11-16 in the open document cited next to it (PDF text extracted with
pdftotext); the asserted bands are the deviations MEASURED that day, not targets. Sources:
  [H11] Humbird et al. (2011), NREL/TP-5100-47764. https://docs.nlr.gov/docs/fy11osti/47764.pdf
        (sec. 3.6-3.9, PFD stream table of Area 900, equipment list of Appendix A)
  [A02] Aden et al. (2002), NREL/TP-510-32438. https://docs.nlr.gov/docs/fy02osti/32438.pdf
        (sec. II.6, II.9, individual equipment cost summary)
  [N02] Loh, Lyons & White (2002), DOE/NETL-2002/1169, Appendix B. https://www.osti.gov/servlets/purl/797810
  [S05] Smith (2005), Chemical Process Design and Integration, Table 2.1 (open copy hosted by fenix.ciencias.ulisboa.pt)
  [E15] U.S. EPA CHP Partnership (2015), Catalog of CHP Technologies, Section 4, Table 4-2.
        https://www.epa.gov/sites/default/files/2015-07/documents/catalog_of_chp_technologies_section_4._technology_characterization_-_steam_turbines.pdf
  [E07] U.S. EPA CHP Partnership (2007), Biomass CHP Catalog, ch. 5, Tables 5-4 and 5-7.
        https://www.epa.gov/sites/default/files/2015-07/documents/biomass_combined_heat_and_power_catalog_of_technologies_5._biomass_conversion_technologies.pdf
BioSTEAM's facility correlations are the quotes of [H11]: reproducing them is arithmetic, not validation."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import CoolProp.CoolProp as CP
import pytest

from packages.costing import utilities_capital as UC
from packages.dwsim_client import tools_utilities_capital as T

ATM = 101325.0
PSI = 6894.757293168
GCAL_H = 1e9 * 4.184 / 3600.0          # W per Gcal/h
MMBTU_H = 1055.05585262e6 / 3600.0     # W per MMBtu/h
LB_H = 0.45359237 / 3600.0             # kg/s per lb/h


def F(t):
    return (t - 32.0) / 1.8 + 273.15


def rel(a, b):
    return a / b - 1.0


# ---------------------------------------------------------------- cooling tower ---------------------------------------------
# [H11] PFD stream table of Area 900 (streams 940-951) and sec. 3.9.2: cooling duty QTTOTAL 97.401 Gcal/h, water 37 -> 28 °C,
# air in 6,549,403 kg/h at 25 °C carrying 128,274 kg/h of water, air out 6,687,697 kg/h at 37 °C carrying 266,568 kg/h;
# water-balance table: cooling tower evaporation 137,363 kg/h; make-up 155,041 kg/h; blowdown 17,678 kg/h; windage 0.005 %.
H11_DUTY_W = 97.401 * GCAL_H
H11_DRY_AIR_KG_H = 6_549_403 - 128_274
H11_W_IN = 128_274 / H11_DRY_AIR_KG_H
H11_EVAP_KG_H = 137_363.0


def test_tower_air_balance_reproduces_humbird_published_tower():
    air_in = UC._air_state(298.15, ATM, H11_W_IN)
    b = UC.tower_air_balance(H11_DUTY_W, 301.15, 310.15, air_in, ATM, air_outlet_T_K=310.15)
    assert -0.030 < rel(b["evaporation_kg_s"] * 3600, H11_EVAP_KG_H) < -0.015        # measured -2.3 %
    assert -0.030 < rel(b["dry_air_kg_s"] * 3600, H11_DRY_AIR_KG_H) < -0.010         # measured -2.0 %
    assert b["energy_closure"] < 1e-12
    # the report's stream table adds the pump work (WP902 = 1.374 Gcal/h) to the tower: with it the gap halves
    b2 = UC.tower_air_balance(H11_DUTY_W + 1.374 * GCAL_H, 301.15, 310.15, air_in, ATM, air_outlet_T_K=310.15)
    assert abs(rel(b2["evaporation_kg_s"] * 3600, H11_EVAP_KG_H)) < 0.012            # measured -0.9 %
    # given the published air flow instead, the outlet temperature comes out at the published 37 °C
    b3 = UC.tower_air_balance(H11_DUTY_W + 1.374 * GCAL_H, 301.15, 310.15, air_in, ATM, dry_air_kg_s=H11_DRY_AIR_KG_H / 3600)
    assert abs(b3["outlet"]["T_K"] - 310.15) < 0.15                                  # measured -0.06 K


def test_tool_reproduces_humbird_evaporation_and_makeup_from_user_level_inputs():
    cycles = 1 + H11_EVAP_KG_H / (17_678 + 0.00005 * 11_923_904)    # the report's own purge + windage
    r = T.cooling_tower_design(H11_DUTY_W, 301.15, 310.15, 298.15, cycles)
    assert -0.035 < rel(r["evaporation_kg_s"] * 3600, H11_EVAP_KG_H) < -0.015        # measured -2.5 %
    assert -0.035 < rel(r["makeup_kg_s"] * 3600, 155_041) < -0.010                   # measured -2.1 %
    assert -0.025 < rel(r["air"]["dry_air_kg_s"] * 3600, H11_DRY_AIR_KG_H) < 0.0     # measured -1.3 % (minimum-air limit)
    assert r["drift_kg_s"] == pytest.approx(0.00005 * r["circulation_kg_s"])
    assert any("drift_fraction" in s["input"] for s in r["suggested_inputs"])
    assert r["evidence"]["verified"] and r["evidence"]["energy_balance_closure"] < 1e-10
    assert r["evaporation_basis"].startswith("balance psicrométrico en el límite de aire mínimo")


def test_tower_capital_reproduces_the_published_scaled_costs_and_shows_biosteam_exponent():
    # [H11] equipment list: M-902 1,375,000 USD (2010) at 10,037,820 kg/h, exponent 0.60, scaled to 11,923,904 kg/h =
    # 1,524,650 USD; P-902 283,671 USD at 10,982,556 kg/h, exponent 0.80, scaled = 302,961 USD. Work streams WM902 1021 kW
    # and WP902 1598 kW at 11,923,904 kg/h.
    Q = 11_923_904 / 3600 * (UC._h_liq(310.15) - UC._h_liq(301.15))
    r = UC.cooling_tower_design(Q, 301.15, 310.15, 298.15, 8.5, cepci=550.8)
    est = r["capital"]["tower_estimates"]
    assert est["humbird2011_table"]["purchase_cost_usd"] == pytest.approx(1_524_650, rel=1e-5)
    assert rel(est["biosteam"]["purchase_cost_usd"], 1_524_650) == pytest.approx(0.017, abs=0.003)   # BioSTEAM uses 0.7
    assert r["capital"]["items"][1]["purchase_cost_usd"] == pytest.approx(302_961, rel=1e-3)
    assert r["capital"]["selected_tower_estimate"] == "biosteam"
    assert r["fan_power_W"] == pytest.approx(1.021e6) and r["pump_power_W"] == pytest.approx(1.598e6)
    assert rel(r["power_cross_check"]["biosteam_fan_plus_pump_W"], 2.619e6) == pytest.approx(0.148, abs=0.01)
    items = r["capital"]["items"]
    assert {"name", "F_BM", "purchase_cost_usd", "base_purchase_cost_usd", "basis"} <= set(items[0])
    assert items[0]["F_BM"] == 1.5 and items[1]["F_BM"] == 3.1


def test_tower_capital_independent_cross_check_aden_2002():
    # [A02] M-902: 1,659,000 USD (1998), induced draft, 71,000 gpm, 147.8e6 kcal/h, scaling exponent 0.78 on heat duty,
    # installation factor 1.2 -> 1,990,800 USD installed. Independent vendor quote, 12 years older than [H11].
    r = UC.cooling_tower_design(147.8e6 * 4184 / 3600, 301.15, 310.15, 298.15, 5.0, cepci=389.5)
    est = r["capital"]["tower_estimates"]
    assert est["aden2002"]["purchase_cost_usd"] == pytest.approx(1_659_000)
    assert rel(r["capital"]["circulation_gpm"], 71_000) == pytest.approx(0.023, abs=0.01)       # 72,650 gpm for 9 K range
    assert rel(est["biosteam"]["purchase_cost_usd"], 1_659_000) == pytest.approx(-0.173, abs=0.02)
    assert rel(est["humbird2011_table"]["purchase_cost_usd"], 1_659_000) == pytest.approx(-0.212, abs=0.02)
    assert rel(est["biosteam"]["installed_cost_usd"], 1_990_800) == pytest.approx(0.034, abs=0.02)
    assert rel(est["humbird2011_table"]["installed_cost_usd"], 1_990_800) == pytest.approx(-0.015, abs=0.02)


def test_small_tower_uses_the_netl_table_and_reports_the_spread():
    # [N02] Appendix B, cooling tower, 1Q1998: 1,000 gpm -> 18,000 USD purchased, 81,700 USD installed; 2,000 gpm -> 34,400 /
    # 106,100. [S05] Table 2.1: cooling tower 4.43e3 USD at 10 m3/h, exponent 0.63, 10-40 m3/h, CE 391.1.
    rho = CP.PropsSI("D", "T", 302.15, "Q", 0, "Water")
    for gpm, purchased, installed in ((1000, 18_000, 81_700), (2000, 34_400, 106_100)):
        m = gpm * UC.US_GAL_M3 / 60 * rho
        Q = m * (UC._h_liq(310.15) - UC._h_liq(302.15))
        r = UC.cooling_tower_design(Q, 302.15, 310.15, 297.15, 4.0, cepci=389.5)
        cap = r["capital"]
        assert cap["selected_tower_estimate"] == "netl2002"
        assert cap["items"][0]["purchase_cost_usd"] == pytest.approx(purchased, rel=1e-6)
        assert cap["items"][0]["installed_cost_usd"] == pytest.approx(installed, rel=1e-6)
    # measured spread at 2,000 gpm against the table: the extrapolated field-erected quotes are 2.2x-3.4x higher in purchase
    # (different scope: factory package vs erected system); installed +57 % (BioSTEAM), +114 % ([H11] exponent), +4 % ([A02])
    assert cap["tower_purchase_deviation_vs_selected"]["biosteam"] == pytest.approx(2.23, abs=0.1)
    inst = cap["tower_installed_deviation_vs_selected"]
    assert inst["biosteam"] == pytest.approx(0.57, abs=0.05) and inst["humbird2011_table"] == pytest.approx(1.14, abs=0.06)
    assert inst["aden2002"] == pytest.approx(0.04, abs=0.05)
    m = 10.0 / 3600 * rho
    r = UC.cooling_tower_design(m * (UC._h_liq(310.15) - UC._h_liq(302.15)), 302.15, 310.15, 297.15, 4.0, cepci=391.1)
    assert r["capital"]["selected_tower_estimate"] == "smith2005"
    assert r["capital"]["tower_estimates"]["smith2005"]["purchase_cost_usd"] == pytest.approx(4430.0, rel=1e-6)


def test_tower_physics_identities():
    base = dict(duty_W=5e6, supply_T_K=302.15, return_T_K=312.15, wet_bulb_T_K=297.15, cycles_of_concentration=4.0)
    r = UC.cooling_tower_design(**base, liquid_to_gas_ratio=1.2, pump_head_m=30.0, pump_efficiency=0.7,
                                fan_static_pressure_Pa=150.0, fan_efficiency=0.6)
    ev = r["evidence"]
    assert ev["verified"] and ev["energy_balance_closure"] < 1e-10 and ev["merkel_quadrature_vs_simpson"] < 1e-8
    assert r["makeup_kg_s"] == pytest.approx(r["evaporation_kg_s"] + r["drift_kg_s"] + r["blowdown_kg_s"])
    assert r["makeup_kg_s"] / (r["blowdown_kg_s"] + r["drift_kg_s"]) == pytest.approx(4.0)
    cp_dT = CP.PropsSI("H", "T", 312.15, "Q", 0, "Water") - CP.PropsSI("H", "T", 302.15, "Q", 0, "Water")
    assert r["circulation_kg_s"] == pytest.approx(5e6 / cp_dT)
    assert r["pump_power_W"] == pytest.approx(r["circulation_kg_s"] * 9.80665 * 30.0 / 0.7)
    assert 302.15 < r["air"]["outlet"]["T_K"] < 312.15 and r["air"]["merkel_number"] > 0
    assert r["evaporation_kg_s"] < r["evaporation_bounds_kg_s"]["all_duty_as_latent_heat"]      # saturated inlet air
    # linear in duty; more cycles, less blowdown; drier inlet air, more evaporation; less air, larger Merkel number
    r2 = UC.cooling_tower_design(**{**base, "duty_W": 1e7}, liquid_to_gas_ratio=1.2)
    assert r2["evaporation_kg_s"] == pytest.approx(2 * r["evaporation_kg_s"], rel=1e-9)
    r3 = UC.cooling_tower_design(**{**base, "cycles_of_concentration": 8.0}, liquid_to_gas_ratio=1.2)
    assert r3["blowdown_kg_s"] < r["blowdown_kg_s"] and r3["evaporation_kg_s"] == pytest.approx(r["evaporation_kg_s"])
    dry = UC.cooling_tower_design(**base, liquid_to_gas_ratio=1.2, dry_bulb_T_K=310.15)
    assert dry["evaporation_kg_s"] > r["evaporation_kg_s"]
    tight = UC.cooling_tower_design(**base, liquid_to_gas_ratio=0.95 * r["air"]["maximum_liquid_to_gas_ratio"])
    assert tight["air"]["merkel_number"] is None or tight["air"]["merkel_number"] > r["air"]["merkel_number"]
    # given the outlet temperature found with L/G, the air flow is recovered
    back = UC.cooling_tower_design(**base, air_outlet_T_K=r["air"]["outlet"]["T_K"])
    assert back["air"]["dry_air_kg_s"] == pytest.approx(r["air"]["dry_air_kg_s"], rel=1e-6)
    # saturation humidity of CoolProp's real-gas model against the ideal-gas formula 0.621945 p_ws / (P - p_ws)
    pws = CP.PropsSI("P", "T", 310.15, "Q", 0, "Water")
    assert rel(UC._air_state(310.15, ATM)["humidity_ratio_kg_kg"], 0.621945 * pws / (ATM - pws)) == pytest.approx(0.005, abs=0.003)


def test_merkel_number_matches_the_log_mean_limit_for_a_narrow_range():
    Tc, Th, lg = 303.15, 303.65, 1.0
    air = UC._air_state(297.15, ATM)
    m = UC.merkel_number(Tc, Th, air["enthalpy_J_kg_dry_air"], lg, ATM)
    d1 = UC._ha("H", T=Tc, P=ATM, R=1.0) - air["enthalpy_J_kg_dry_air"]
    d2 = UC._ha("H", T=Th, P=ATM, R=1.0) - (air["enthalpy_J_kg_dry_air"] + lg * (UC._h_liq(Th) - UC._h_liq(Tc)))
    analytic = (UC._h_liq(Th) - UC._h_liq(Tc)) / ((d1 - d2) / math.log(d1 / d2))
    assert m["merkel_number"] == pytest.approx(analytic, rel=5e-4)          # measured 1.3e-4


def test_tower_refuses_missing_or_impossible_inputs():
    ok = dict(duty_W=5e6, supply_T_K=302.15, return_T_K=312.15, wet_bulb_T_K=297.15, cycles_of_concentration=4.0)
    for key in ok:
        with pytest.raises(ValueError, match=key):
            UC.cooling_tower_design(**{**ok, key: None})
    with pytest.raises(ValueError, match="bulbo húmedo"):
        UC.cooling_tower_design(**{**ok, "wet_bulb_T_K": 303.15})
    with pytest.raises(ValueError, match="aire no alcanza"):
        UC.cooling_tower_design(**ok, liquid_to_gas_ratio=5.0)
    with pytest.raises(ValueError, match="no los dos"):
        UC.cooling_tower_design(**ok, liquid_to_gas_ratio=1.0, air_outlet_T_K=305.0)
    with pytest.raises(ValueError, match="pump_efficiency"):
        UC.cooling_tower_design(**ok, pump_head_m=30.0)
    with pytest.raises(ValueError, match="cepci"):
        UC.cooling_tower_design(**ok, year=2031)
    a = UC.cooling_tower_design(**ok, cepci=500.0)["capital"]["purchase_cost_usd"]
    b = UC.cooling_tower_design(**ok, cepci=1000.0)["capital"]["purchase_cost_usd"]
    assert b == pytest.approx(2 * a)


# ---------------------------------------------------------------- boiler and turbogenerator ---------------------------------
# [E15] Table 4-2, back-pressure steam turbines (Elliott Group data, 2014):
#   kW, isentropic eff., generator/gearbox eff., steam lb/h, inlet psig, inlet °F, outlet psig, outlet °F,
#   equipment $/kW, installed $/kW, fuel input MMBtu/h (boiler 80 % HHV), steam to process MMBtu/h
EPA_TURBINES = [(500, 0.525, 0.94, 20_050, 500, 550, 50, 298.0, 668, 1136, 27.2, 19.9),
                (3000, 0.612, 0.94, 152_600, 600, 575, 150, 373.0, 401, 682, 208.3, 155.7),
                (15000, 0.780, 0.96, 494_464, 700, 650, 150, 379.7, 392, 666, 700.1, 506.8)]


def _epa_case(row, **boiler):
    kW, eta, gen, lb_h, p_in, t_in, p_out = row[:7]
    return T.steam_boiler_design(
        [{"name": "proceso", "pressure_Pa": p_out * PSI + ATM, "flow_kg_s": lb_h * LB_H}], 373.15,
        {"type": "natural_gas", "HHV_J_kg": 52.2e6}, {"efficiency": 0.80, "efficiency_basis": "HHV", "blowdown_fraction": 0.0, **boiler},
        turbogenerator={"inlet_pressure_Pa": p_in * PSI + ATM, "inlet_T_K": F(t_in), "isentropic_efficiency": eta,
                        "mode": "back_pressure", "generator_efficiency": gen}, cepci=576.1)


@pytest.mark.parametrize("row", EPA_TURBINES)
def test_turbogenerator_power_and_exhaust_temperature_match_epa_table_4_2(row):
    r = _epa_case(row)
    tg = r["turbogenerator"]
    assert abs(rel(tg["electric_power_W"] / 1000, row[0])) < 0.002            # measured +0.11 %, +0.02 %, +0.01 %
    assert abs(r["steam_levels"][0]["T_K"] - F(row[7])) < 0.3                 # measured -0.20, -0.15, -0.01 K
    assert tg["energy_closure"] < 1e-12 and r["evidence"]["verified"]
    # CONSISTENCY with an inferred input: the table does not state the feedwater condition; with saturated liquid at 212 °F
    # and no blowdown the published fuel input and steam-to-process heat are reproduced (measured within 0.1 % and 0.3 %)
    assert abs(rel(r["fuel"]["heat_release_HHV_W"] / MMBTU_H, row[10])) < 0.002
    to_process = r["absorbed_duty_W"] - tg["shaft_power_W"]
    assert abs(rel(to_process / MMBTU_H, row[11])) < 0.004


def test_turbine_capital_table_nodes_and_cross_check():
    for kW, _, _, _, _, _, _, _, usd_kw, inst_kw, _, _ in EPA_TURBINES:
        c = UC._turbine_capital(kW * 1000.0, "back_pressure", 576.1, "CEPCI 576.1", [])
        assert c["selected_estimate"] == "epa_chp2015"
        assert c["purchase_cost_usd"] == pytest.approx(usd_kw * kW, rel=1e-9)
        assert c["installed_cost_usd"] == pytest.approx(inst_kw * kW, rel=1e-9)
    # BioSTEAM (Humbird's 42.2 MW extraction-condensing Siemens set, exponent 0.6) against the EPA back-pressure sets:
    # measured -9 % at 15 MW, +69 % at 3 MW, +108 % at 500 kW: the power law over-prices small machines
    dev = [UC._turbine_capital(kW * 1e3, "back_pressure", 576.1, "", [])["purchase_deviation_vs_selected"]["biosteam"]
           for kW in (15000, 3000, 500)]
    assert dev[0] == pytest.approx(-0.092, abs=0.02) and dev[1] == pytest.approx(0.69, abs=0.03)
    assert dev[2] == pytest.approx(1.08, abs=0.04)
    # [H11] M-811: 9,500,000 USD (2010) at 42,200 kW, exponent 0.6, scaled to 41,324 kW = 9,381,215 USD
    c = UC._turbine_capital(41_324e3, "condensing", 551.0, "", [])
    assert c["selected_estimate"] == "biosteam" and c["purchase_cost_usd"] == pytest.approx(9_381_215, rel=1e-4)
    assert c["items"][0]["F_BM"] == 1.8


def test_humbird_turbogenerator_is_only_a_plausibility_check():
    # [H11] sec. 3.8.2: 239,000 kg/h at 454 °C and 900 psig; 12 % extracted at 13 atm (268 °C), 35 % at 9.5 atm, the rest
    # condensed at 0.1 atm; "generator efficiency 85 %"; 41 MW generated. The isentropic efficiency is NOT stated (0.85 is
    # the value of [A02] for the same model), so this is plausibility, not validation: power +0.8 %, but the extraction comes
    # out 7 K colder than published and the condenser duty 25 % above the 52 % of 97.4 Gcal/h of Figure 14.
    m = 239_000 / 3600
    r = UC.steam_boiler_design(
        [{"name": "13 atm", "pressure_Pa": 13 * ATM, "flow_kg_s": 0.12 * m}, {"name": "9.5 atm", "pressure_Pa": 9.5 * ATM, "flow_kg_s": 0.35 * m}],
        450.15, {"type": "biomass", "LHV_J_kg": 9.0e6}, {"efficiency": 0.80, "efficiency_basis": "LHV"},
        turbogenerator={"inlet_pressure_Pa": 900 * PSI + ATM, "inlet_T_K": 727.15, "isentropic_efficiency": 0.85, "mode": "condensing",
                        "condensing_flow_kg_s": 0.53 * m, "condenser_pressure_Pa": 0.1 * ATM, "generator_efficiency": 0.85}, cepci=550.8)
    tg = r["turbogenerator"]
    assert rel(tg["electric_power_W"], 41e6) == pytest.approx(0.008, abs=0.01)
    assert r["steam_levels"][0]["T_K"] - (268 + 273.15) == pytest.approx(-7.4, abs=1.0)
    assert rel(tg["condenser_duty_W"] / GCAL_H, 0.52 * 97.401) == pytest.approx(0.25, abs=0.03)
    assert any("calidad" in w for w in r["warnings"]) and any("condensador" in w.lower() for w in r["warnings"])


def test_boiler_capital_reproduces_published_tables_and_quotes():
    # [N02] Appendix B, package steam boiler (250 psig, 100 °F superheat), 1Q1998: 50,000 lb/h -> 212,700 USD purchased,
    # 468,900 USD installed; 100,000 lb/h -> 305,700 / 607,300.
    for lb_h, purchased, installed in ((50_000, 212_700, 468_900), (100_000, 305_700, 607_300)):
        r = UC.steam_boiler_design([{"name": "x", "pressure_Pa": 250 * PSI + ATM, "flow_kg_s": lb_h * LB_H}], 380.0,
                                   {"type": "natural_gas", "LHV_J_kg": 47e6},
                                   {"efficiency": 0.9, "efficiency_basis": "LHV", "superheat_K": 100 / 1.8}, cepci=389.5)
        cap = r["capital"]
        assert cap["selected_estimate"] == "netl2002_package" and not r["warnings"]
        assert cap["purchase_cost_usd"] == pytest.approx(purchased, rel=1e-6)
        assert cap["installed_cost_usd"] == pytest.approx(installed, rel=1e-6)
    # [S05] Table 2.1 (January 2000, CE 391.1): package boiler 4.64e5 USD at 50,000 kg/h (exp. 0.96, 50,000-350,000 kg/h);
    # field-erected 3.28e5 USD at 20,000 kg/h (exp. 0.81, 10,000-800,000 kg/h)
    r = UC.steam_boiler_design([{"name": "x", "pressure_Pa": 18e5, "flow_kg_s": 50_000 / 3600}], 380.0,
                               {"type": "fuel_oil", "LHV_J_kg": 41e6}, {"efficiency": 0.9, "efficiency_basis": "LHV", "superheat_K": 0},
                               cepci=391.1)
    est = r["capital"]["estimates"]
    assert est["smith2005_package_boiler"]["purchase_cost_usd"] == pytest.approx(4.64e5, rel=1e-9)
    assert est["smith2005_field_erected_boiler"]["purchase_cost_usd"] == pytest.approx(3.28e5 * 2.5 ** 0.81, rel=1e-9)
    # independent cross-check of the gas/oil package boiler at 50 t/h: Smith's package boiler is +39 % over the NETL table
    assert r["capital"]["purchase_deviation_vs_selected"]["smith2005_package_boiler"] == pytest.approx(0.385, abs=0.03)
    # [H11] M-803 28,550,000 USD (2010) at 238,686 kg/h, exp. 0.6, scaled to 234,784 kg/h = 28,269,041 USD; T-826 deaerator
    # 305,000 at 235,803 kg/h -> 304,209; M-820 softener 78,000 -> 77,798; installation 1.8 / 3.0 / 1.8
    r = UC.steam_boiler_design([{"name": "x", "pressure_Pa": 13 * ATM, "flow_kg_s": 234_784 / 3600}], 450.15,
                               {"type": "biomass", "LHV_J_kg": 9.0e6},
                               {"efficiency": 0.8, "efficiency_basis": "LHV", "superheat_K": 0, "blowdown_fraction": 0.0}, cepci=551.0)
    items = r["capital"]["items"]
    assert r["capital"]["selected_estimate"] == "biosteam"
    assert [round(i["purchase_cost_usd"]) for i in items[:3]] == pytest.approx([28_269_041, 304_209, 77_798], rel=1e-4)
    assert [i["F_BM"] for i in items] == [1.8, 3.0, 1.8, 1.8]
    assert items[3]["purchase_cost_usd"] == pytest.approx(39_896, rel=1e-3)     # BioSTEAM scales the amine package; [H11] does not


def test_biomass_boiler_cross_check_against_epa_table_5_4():
    # [E07] Table 5-4 (nominal dollars, year not stated; band of the source +-35 %): stoker 20,000 lb/h equipment 1,195,000,
    # installed boiler system 1,990,000; 250,000 lb/h 10,790,000 / 23,250,000. Fluidized bed 20,000 lb/h 6,175,000 /
    # 6,970,000; 260,000 lb/h 19,790,000 / 32,250,000. BioSTEAM/Humbird (2010 dollars) falls between the two technologies.
    for lb_h in (20_000, 250_000):
        r = UC.steam_boiler_design([{"name": "x", "pressure_Pa": 20e5, "flow_kg_s": lb_h * LB_H}], 380.0,
                                   {"type": "biomass", "LHV_J_kg": 9.0e6}, {"efficiency": 0.8, "efficiency_basis": "LHV", "superheat_K": 0},
                                   cepci=550.8)
        est = r["capital"]["estimates"]
        stoker, fb, bst = est["epa2007_biomass_stoker"], est["epa2007_biomass_fluidized_bed"], est["biosteam"]
        assert stoker["in_range"] and stoker["purchase_cost_range_usd"][1] < bst["purchase_cost_usd"] < fb["purchase_cost_range_usd"][0]
        assert stoker["installed_cost_range_usd"][0] < bst["installed_cost_usd"] < fb["installed_cost_range_usd"][1]
    dev = r["capital"]["purchase_deviation_vs_selected"]        # at 250,000 lb/h, measured -30 % (stoker) and +25 % (bed)
    assert dev["epa2007_biomass_stoker"] == pytest.approx(-0.30, abs=0.04)
    assert dev["epa2007_biomass_fluidized_bed"] == pytest.approx(0.25, abs=0.04)
    # the stoker node of the table at its own (assumed) dollars
    lo = est["epa2007_biomass_stoker"]["purchase_cost_range_usd"][0]
    assert lo == pytest.approx(10_790_000 * 550.8 / 525.4, rel=1e-9)


def test_boiler_energy_identities_and_fuel_routes():
    lv = [{"name": "MP", "pressure_Pa": 11e5, "flow_kg_s": 3.0}, {"name": "LP", "pressure_Pa": 4e5, "flow_kg_s": 1.0}]
    fuel = {"type": "natural_gas", "LHV_J_kg": 47.1e6, "HHV_J_kg": 52.2e6}
    r = UC.steam_boiler_design(lv, 378.15, fuel, {"efficiency": 0.82, "efficiency_basis": "HHV", "superheat_K": 0, "blowdown_fraction": 0.0})
    hg = CP.PropsSI("H", "P", 11e5, "Q", 1, "Water")
    hf = CP.PropsSI("H", "T", 378.15, "Q", 0, "Water")
    assert r["absorbed_duty_W"] == pytest.approx(4.0 * (hg - hf), rel=1e-12)
    assert r["fuel"]["fuel_kg_s"] == pytest.approx(r["absorbed_duty_W"] / (0.82 * 52.2e6))
    assert r["fuel"]["efficiency_LHV"] == pytest.approx(0.82 * 52.2 / 47.1)
    lp = r["steam_levels"][1]
    assert lp["enthalpy_J_kg"] == pytest.approx(hg) and lp["superheat_K"] > 0          # isenthalpic let-down superheats
    assert r["evidence"]["verified"] and r["evidence"]["steam_side_energy_closure"] < 1e-12
    # blowdown raises the duty by m_bd (h_f,drum - h_fw); the suggested default is flagged
    r2 = UC.steam_boiler_design(lv, 378.15, fuel, {"efficiency": 0.82, "efficiency_basis": "HHV", "superheat_K": 0})
    bd = 0.03 / 1.03
    assert r2["blowdown_fraction"] == pytest.approx(bd) and r2["suggested_inputs"][0]["input"] == "boiler.blowdown_fraction"
    extra = r2["blowdown_kg_s"] * (CP.PropsSI("H", "P", 11e5, "Q", 0, "Water") - hf)
    assert r2["absorbed_duty_W"] == pytest.approx(r["absorbed_duty_W"] + extra, rel=1e-12)
    # combustion route: the heat-loss engine of the repo gives the fuel; fuel x LHV x efficiency = duty
    comb = {"fuel": {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 298.15},
            "air": {"T_K": 298.15, "relative_humidity": 0.5}, "pressure_Pa": ATM, "excess_air_fraction": 0.15,
            "stack_T_K": 450.0, "setting_loss_fraction": 0.01, "setting_loss_basis": "HHV"}
    r3 = UC.steam_boiler_design(lv[:1], 378.15, {"type": "natural_gas"}, {"superheat_K": 0, "blowdown_fraction": 0.0}, combustion=comb)
    f = r3["fuel"]
    assert f["fuel_kg_s"] * f["LHV_J_kg"] * f["efficiency_LHV"] == pytest.approx(r3["absorbed_duty_W"], rel=1e-9)
    assert 0.80 < f["efficiency_HHV"] < f["efficiency_LHV"] < 0.95 and f["combustion_evidence"]["verified"]
    # turbine: an ideal machine generates no entropy; power grows with the efficiency; energy closes
    tg = {"inlet_pressure_Pa": 42e5, "inlet_T_K": 673.15, "mode": "back_pressure", "generator_efficiency": 1.0}
    p = []
    for eta in (0.6, 0.8, 1.0):
        rt = UC.steam_boiler_design(lv, 378.15, fuel, {"efficiency": 0.82, "efficiency_basis": "HHV"},
                                    turbogenerator={**tg, "isentropic_efficiency": eta})
        p.append(rt["turbogenerator"]["shaft_power_W"])
        assert rt["turbogenerator"]["energy_closure"] < 1e-12
    assert p[0] < p[1] < p[2] and p[2] == pytest.approx(rt["turbogenerator"]["isentropic_power_W"])
    assert all(abs(s["entropy_generated_J_kgK"]) < 1e-6 for s in rt["turbogenerator"]["sections"])
    assert rt["absorbed_duty_W"] - p[2] == pytest.approx(
        sum(d["flow_kg_s"] * (d["enthalpy_J_kg"] - hf) for d in rt["steam_levels"]) + rt["blowdown_kg_s"] * (
            CP.PropsSI("H", "P", 42e5, "Q", 0, "Water") - hf), rel=1e-10)


def test_biomass_heating_values_as_fired():
    # [E07] Table 5-7: 8,500 Btu/lb dry; 50 % moisture -> 4,250 Btu/lb as received; 30 % -> 5,950 Btu/lb
    btu_lb = 2326.0
    for w, as_received in ((0.50, 4250), (0.30, 5950)):
        assert UC.biomass_heating_values(8500 * btu_lb, w)["HHV_J_kg"] == pytest.approx(as_received * btu_lb)
    hv = UC.biomass_heating_values(19.0e6, 0.5, 0.06)
    hfg = CP.PropsSI("H", "T", 298.15, "Q", 1, "Water") - CP.PropsSI("H", "T", 298.15, "Q", 0, "Water")
    assert hv["LHV_J_kg"] == pytest.approx(9.5e6 - hfg * (0.5 + 18.01528 / 2.01588 * 0.06 * 0.5))
    assert UC.biomass_heating_values(19.0e6, 0.0, 0.0)["LHV_J_kg"] == pytest.approx(19.0e6)   # dry, no hydrogen: LHV = HHV
    r = UC.steam_boiler_design([{"name": "x", "pressure_Pa": 22e5, "flow_kg_s": 10.0}], 378.15,
                               {"type": "biomass", "HHV_dry_J_kg": 19.0e6, "moisture_mass_fraction": 0.5, "hydrogen_dry_mass_fraction": 0.06},
                               {"efficiency": 0.8, "efficiency_basis": "LHV", "superheat_K": 0})
    assert r["fuel"]["LHV_J_kg"] == pytest.approx(hv["LHV_J_kg"]) and r["fuel"]["efficiency_HHV"] < 0.8


def test_boiler_refuses_missing_inputs():
    lv = [{"name": "MP", "pressure_Pa": 11e5, "flow_kg_s": 3.0}]
    fuel = {"type": "natural_gas", "LHV_J_kg": 47.1e6}
    ok = {"efficiency": 0.9, "efficiency_basis": "LHV", "superheat_K": 0}
    cases = [(dict(steam_demand=[], feedwater_T_K=378.15, fuel=fuel, boiler=ok), "steam_demand"),
             (dict(steam_demand=[{"pressure_Pa": 11e5}], feedwater_T_K=378.15, fuel=fuel, boiler=ok), "flow_kg_s"),
             (dict(steam_demand=lv, feedwater_T_K=None, fuel=fuel, boiler=ok), "feedwater_T_K"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel={"LHV_J_kg": 47e6}, boiler=ok), "type"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel={"type": "natural_gas"}, boiler=ok), "poder calorífico"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler={"superheat_K": 0}), "efficiency"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler={"efficiency": 0.9, "superheat_K": 0}), "efficiency_basis"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler={"efficiency": 0.9, "efficiency_basis": "HHV", "superheat_K": 0}),
              "HHV_J_kg"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler={"efficiency": 0.9, "efficiency_basis": "LHV"}), "superheat_K"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler=ok,
                   turbogenerator={"inlet_pressure_Pa": 42e5, "inlet_T_K": 673.15, "mode": "back_pressure"}), "isentropic_efficiency"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler=ok,
                   turbogenerator={"inlet_pressure_Pa": 42e5, "inlet_T_K": 673.15, "isentropic_efficiency": 0.7}), "mode"),
             (dict(steam_demand=lv, feedwater_T_K=378.15, fuel=fuel, boiler=ok,
                   turbogenerator={"inlet_pressure_Pa": 5e5, "inlet_T_K": 673.15, "isentropic_efficiency": 0.7, "mode": "back_pressure"}),
              "menor presión"),
             (dict(steam_demand=lv, feedwater_T_K=500.0, fuel=fuel, boiler=ok), "saturación")]
    for kwargs, word in cases:
        with pytest.raises(ValueError, match=word):
            UC.steam_boiler_design(**kwargs)
    r = UC.steam_boiler_design([{"name": "x", "pressure_Pa": 42e5, "flow_kg_s": 3.0}], 378.15, fuel, ok)
    assert any("250 psig" in w for w in r["warnings"])      # published pressure factors are quoted, never applied


# ---------------------------------------------------------------- wastewater ------------------------------------------------
HUMBIRD_REMOVAL = {"anaerobic": 0.91, "anaerobic_to_biomass": 0.05, "aerobic": 0.96, "aerobic_to_biomass": 0.22}   # [H11] 3.6.2


def test_methane_stoichiometry_against_the_published_maximum_yield():
    st = UC.methane_stoichiometry()
    assert st["kg_COD_per_kg_CH4"] == pytest.approx(2 * 31.9988 / 16.0425)
    # [A02] sec. II.6.2: "maximum yield of 350 L per kg COD removed"
    assert rel(st["Nm3_CH4_per_kg_COD"] * 1000, 350.0) == pytest.approx(0.0007, abs=0.0005)
    assert st["kg_CH4_per_kg_COD"] == pytest.approx(0.2507, abs=1e-4)


def test_wastewater_cod_balance_and_biogas_against_humbird():
    # [H11] sec. 3.6.2: COD to anaerobic digestion 26,500 kg/h (64 g/L); 91 % destroyed (86 % to biogas, 5 % to cells);
    # methane 228 g/kg COD removed; aerobic 96 % of the rest (74 % oxidised, 22 % cells); total COD reduction 99.6 %;
    # biogas "about 65 MMkcal/h".
    r = T.wastewater_treatment_cost(26_500 / 3600 / 64.0, 64.0, "anaerobic_aerobic", HUMBIRD_REMOVAL, True, True, cepci=550.8)
    an = r["stages"]["anaerobic"]
    assert rel(an["methane_kg_per_kg_COD_removed"], 0.228) == pytest.approx(0.039, abs=0.005)     # measured +3.9 %
    assert rel(r["biogas"]["energy_LHV_W"] / GCAL_H, 65.0) == pytest.approx(0.051, abs=0.01)   # measured +5.1 %
    assert r["overall_COD_removal"] == pytest.approx(0.996, abs=5e-4)
    assert r["evidence"]["verified"] and r["evidence"]["COD_balance_closure"] < 1e-12
    assert r["oxygen_demand_kg_s"] == pytest.approx(0.74 * 0.09 * r["COD_load_kg_s"])
    assert r["biogas"]["methane_LHV_J_kg"] == pytest.approx(50.03e6, rel=2e-3)                # NASA polynomials
    # [A02]: 93 % anaerobic then 90 % aerobic -> "total COD reduction is 99.4 %" (0.993 exactly)
    a = UC.wastewater_treatment_cost(100 / 3600, 16.0, "anaerobic_aerobic",
                                     {"anaerobic": 0.93, "anaerobic_to_biomass": 0.0, "aerobic": 0.90, "aerobic_to_biomass": 0.30})
    assert a["overall_COD_removal"] == pytest.approx(0.994, abs=1.5e-3)


def test_wastewater_capital_reproduces_the_source_and_shows_the_spread():
    # [H11] equipment list, Area 600: the 18 installed costs of 2010 add up to BioSTEAM's 50,280,080 USD; scaled in the
    # report from 393,100 to 411,178 kg/h with exponent 0.6 and taken to 2007 dollars: 49,389,478 USD.
    assert sum(c for _, _, c, _ in UC.HUMBIRD_WWT_ITEMS) == pytest.approx(50_280_080)
    assert UC.biosteam_cost_items()["wastewater/Wastewater system"]["cost"] == pytest.approx(50_280_080)
    rho = CP.PropsSI("D", "T", 298.15, "P", ATM, "Water")
    r = UC.wastewater_treatment_cost(411_178 / 3600 / rho, 64.0, "anaerobic_aerobic", HUMBIRD_REMOVAL, True, True, cepci=525.4)
    est = r["capital"]["estimates"]
    assert rel(est["humbird2011_itemized_flow_only"]["installed_cost_usd"], 49_389_478) == pytest.approx(-0.0024, abs=0.001)
    assert est["biosteam"]["installed_cost_usd"] == pytest.approx(est["humbird2011_itemized_flow_only"]["installed_cost_usd"], rel=1e-3)
    # at the design basis of the quote (393,100 kg/h, 2.15 MMgal/d x 87 g/L of COD) the COD-scaled reactors change nothing
    cod_kg_h = 2.15e6 * UC.US_GAL_M3 / 24 * 87.0
    V = 393_100 / 3600 / rho
    b = UC.wastewater_treatment_cost(V, cod_kg_h / 3600 / V, "anaerobic_aerobic", HUMBIRD_REMOVAL, True, True, cepci=550.8)
    assert b["capital"]["installed_cost_usd"] == pytest.approx(50_280_080, rel=1e-4)
    assert all(i["F_BM"] == 1.0 and i["purchase_cost_usd"] == i["installed_cost_usd"] for i in b["capital"]["items"])
    # [A02] A600 installed 3,252,573 USD (1998) for 1,600 kg/h of COD at 16 g/L, simpler scope (no membranes, RO, evaporator):
    # this module gives 2.4 times that and BioSTEAM's flow-only law 4.8 times. Order of magnitude only, and it says so.
    a = UC.wastewater_treatment_cost(100 / 3600, 16.0, "anaerobic_aerobic",
                                     {"anaerobic": 0.93, "anaerobic_to_biomass": 0.0, "aerobic": 0.90, "aerobic_to_biomass": 0.30},
                                     cepci=389.5)
    assert a["capital"]["installed_cost_usd"] / 3_252_573 == pytest.approx(2.38, abs=0.1)
    assert a["capital"]["estimates"]["biosteam"]["installed_cost_usd"] / 3_252_573 == pytest.approx(4.80, abs=0.15)
    assert a["capital"]["estimates"]["aden2002"]["installed_cost_usd"] == pytest.approx(3_252_573, rel=1e-6)
    assert any("orden de magnitud" in w for w in a["warnings"])


def test_wastewater_identities_scope_and_operating_cost():
    rem = {"aerobic": 0.95, "aerobic_to_biomass": 0.40}
    r = UC.wastewater_treatment_cost(20 / 3600, 5.0, "aerobic", rem, aeration_kg_O2_per_kWh=1.5,
                                     electricity_price_usd_kWh=0.10, operating_hours_per_year=8000.0)
    assert r["biogas"]["methane_kg_s"] == 0.0 and "anaerobic" not in r["stages"]
    assert r["oxygen_demand_kg_s"] == pytest.approx(0.55 * r["COD_load_kg_s"])
    assert r["power"]["aeration_power_W"] == pytest.approx(r["oxygen_demand_kg_s"] / 1.5 * 3.6e6)
    assert r["operating"]["electricity_cost_usd_y"] == pytest.approx(r["power"]["used_W"] / 1000 * 8000 * 0.10)
    assert any("solo aerobia" in w for w in r["warnings"])
    full = dict(flow_m3_s=0.05, cod_kg_m3=40.0, configuration="anaerobic_aerobic", cod_removal=HUMBIRD_REMOVAL)
    base = UC.wastewater_treatment_cost(**full)
    ro = UC.wastewater_treatment_cost(**full, include_reverse_osmosis=True)
    both = UC.wastewater_treatment_cost(**full, include_reverse_osmosis=True, include_brine_evaporator=True)
    assert base["capital"]["installed_cost_usd"] < ro["capital"]["installed_cost_usd"] < both["capital"]["installed_cost_usd"]
    double = UC.wastewater_treatment_cost(**{**full, "flow_m3_s": 0.10})          # same COD concentration, twice the plant
    assert double["capital"]["installed_cost_usd"] == pytest.approx(2 ** 0.6 * base["capital"]["installed_cost_usd"], rel=1e-9)
    assert double["biogas"]["methane_kg_s"] == pytest.approx(2 * base["biogas"]["methane_kg_s"])
    none = UC.wastewater_treatment_cost(**{**full, "cod_removal": {**HUMBIRD_REMOVAL, "anaerobic": 0.0, "anaerobic_to_biomass": 0.0}})
    assert none["biogas"]["methane_kg_s"] == 0.0
    credit = UC.wastewater_treatment_cost(**full, operating_hours_per_year=8000.0, biogas_value_usd_GJ=4.0)
    assert credit["operating"]["biogas_credit_usd_y"] == pytest.approx(credit["biogas"]["energy_LHV_W"] * 8000 * 3600 / 1e9 * 4.0)
    assert "note" in base["operating"]                                            # no prices, no cost invented


def test_wastewater_refuses_missing_inputs():
    ok = dict(flow_m3_s=0.05, cod_kg_m3=40.0, configuration="anaerobic_aerobic", cod_removal=HUMBIRD_REMOVAL)
    for key in ("flow_m3_s", "cod_kg_m3", "cod_removal"):
        with pytest.raises(ValueError, match=key):
            UC.wastewater_treatment_cost(**{**ok, key: None})
    with pytest.raises(ValueError, match="configuration"):
        UC.wastewater_treatment_cost(**{**ok, "configuration": "mbr"})
    for key in HUMBIRD_REMOVAL:
        with pytest.raises(ValueError, match=key):
            UC.wastewater_treatment_cost(**{**ok, "cod_removal": {k: v for k, v in HUMBIRD_REMOVAL.items() if k != key}})
    with pytest.raises(ValueError, match="no puede superar"):
        UC.wastewater_treatment_cost(**{**ok, "cod_removal": {**HUMBIRD_REMOVAL, "aerobic_to_biomass": 0.99}})
    with pytest.raises(ValueError, match="operating_hours_per_year"):
        UC.wastewater_treatment_cost(**ok, electricity_price_usd_kWh=0.1)


# ---------------------------------------------------------------- hygiene, JSON and wrappers --------------------------------
def test_biosteam_is_not_imported_and_global_state_is_left_untouched():
    src = Path(UC.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^(import|from) (biosteam|thermosteam|biorefineries)", src, flags=re.M)
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    before = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    loaded = {m for m in ("biosteam", "thermosteam", "biorefineries") if m in sys.modules}
    UC.biosteam_cost_items.cache_clear()
    items = UC.biosteam_cost_items()
    UC.cooling_tower_design(5e6, 302.15, 312.15, 297.15, 4.0)
    UC.steam_boiler_design([{"name": "x", "pressure_Pa": 11e5, "flow_kg_s": 3.0}], 378.15, {"type": "biomass", "LHV_J_kg": 8e6},
                           {"efficiency": 0.7, "efficiency_basis": "LHV", "superheat_K": 0},
                           turbogenerator={"inlet_pressure_Pa": 42e5, "inlet_T_K": 673.15, "isentropic_efficiency": 0.7,
                                           "mode": "back_pressure"})
    UC.wastewater_treatment_cost(0.05, 40.0, "anaerobic_aerobic", HUMBIRD_REMOVAL)
    # the correlations are read from BioSTEAM's source: the tools import nothing of BioSTEAM and flip no global flag
    assert {m for m in ("biosteam", "thermosteam", "biorefineries") if m in sys.modules} == loaded
    assert (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR) == before
    # they are BioSTEAM's own numbers (which are the quotes of [H11])
    assert items["cooling_tower/Cooling tower"] == {"S": 557183.0, "units": "kmol/hr", "basis": "Flow rate", "cost": 1375e3,
                                                    "CE": 551.0, "n": 0.7, "kW": 1598.0, "BM": 1.5}
    assert items["cooling_tower/Cooling water pump"]["BM"] == 3.1 and items["boiler_turbogenerator/Deaerator"]["BM"] == 3.0
    assert items["boiler_turbogenerator/Boiler"]["cost"] == 28550e3 and items["boiler_turbogenerator/Turbogenerator"]["S"] == 42200.0
    assert items["wastewater/Wastewater system"]["kW"] == pytest.approx(7139 / 1.05)


def test_source_reading_equals_the_imported_biosteam_classes():
    """The import of BioSTEAM takes 20-50 s on a loaded machine, so this check only runs when another test of the session
    has already imported it (it was run by hand on 2026-11-16 with BioSTEAM 2.54.0: identical dicts, state restored)."""
    if "biosteam" not in sys.modules:
        pytest.skip("BioSTEAM no está importado en esta sesión; la lectura de su código fuente se comprobó a mano")
    import biosteam as bst
    from thermo.utils.t_dependent_property import TDependentProperty
    state = (bst.CE, bst.PowerUtility.price, bst.main_flowsheet.get_flowsheet(), TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    assert UC.biosteam_cost_items(imported=True) == UC.biosteam_cost_items()
    assert (bst.CE, bst.PowerUtility.price, bst.main_flowsheet.get_flowsheet(),
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR) == state


def test_every_result_is_json_and_wrappers_match_tools():
    from packages.agent import wrappers_utilities_capital as W
    assert [f.__name__ for f in T.UTILITIES_CAPITAL_TOOLS] == [w.name for w in W.AGENT_WRAPPERS]
    results = [
        T.cooling_tower_design(5e6, 302.15, 312.15, 297.15, 4.0),
        T.cooling_tower_design(5e6, 302.15, 312.15, 297.15, 4.0, dry_bulb_T_K=308.15, liquid_to_gas_ratio=1.1, pressure_Pa=9.0e4,
                               pump_head_m=30.0, pump_efficiency=0.7, fan_static_pressure_Pa=150.0, fan_efficiency=0.6),
        T.steam_boiler_design([{"name": "MP", "pressure_Pa": 11e5, "flow_kg_s": 3.0}], 378.15, {"type": "natural_gas", "LHV_J_kg": 47.1e6},
                              {"efficiency": 0.9, "efficiency_basis": "LHV", "superheat_K": 0}),
        _epa_case(EPA_TURBINES[1]),
        T.wastewater_treatment_cost(0.05, 40.0, "anaerobic_aerobic", HUMBIRD_REMOVAL),
        T.wastewater_treatment_cost(0.01, 5.0, "aerobic", {"aerobic": 0.95, "aerobic_to_biomass": 0.4}),
    ]
    for r in results:
        back = json.loads(json.dumps(r))
        assert {"warnings", "assumptions", "not_included", "method", "evidence", "capital"} <= set(back)
        assert isinstance(back["evidence"]["verified"], bool)
        for item in back["capital"]["items"]:
            assert item["purchase_cost_usd"] > 0 and item["F_BM"] >= 1.0 and "basis" in item
    out = json.loads(W.cooling_tower_design.call({"duty_W": 5e6, "supply_T_K": 302.15, "return_T_K": 312.15, "wet_bulb_T_K": 297.15,
                                                  "cycles_of_concentration": 4.0}))
    assert out["circulation_kg_s"] == pytest.approx(results[0]["circulation_kg_s"])
    out = json.loads(W.wastewater_treatment_cost.call({"flow_m3_s": 0.05, "cod_kg_m3": 40.0, "configuration": "anaerobic_aerobic",
                                                       "cod_removal": HUMBIRD_REMOVAL}))
    assert out["biogas"]["methane_kg_s"] == pytest.approx(results[4]["biogas"]["methane_kg_s"])
    out = json.loads(W.steam_boiler_design.call({"steam_demand": [{"name": "MP", "pressure_Pa": 11e5, "flow_kg_s": 3.0}],
                                                 "feedwater_T_K": 378.15, "fuel": {"type": "natural_gas", "LHV_J_kg": 47.1e6},
                                                 "boiler": {"efficiency": 0.9, "efficiency_basis": "LHV", "superheat_K": 0}}))
    assert out["fuel"]["fuel_kg_s"] == pytest.approx(results[2]["fuel"]["fuel_kg_s"])
    # the items have the shape capital_investment takes (name, F_BM, purchase and base purchase cost, basis) and their
    # installed cost is its formula C_BM = C_P + C_P,base (F_BM - 1); checked by hand against EC.capital_investment (that
    # module imports BioSTEAM, 20-50 s, so it is not imported here)
    for r in results:
        for i in r["capital"]["items"]:
            assert i["installed_cost_usd"] == pytest.approx(i["purchase_cost_usd"] + i["base_purchase_cost_usd"] * (i["F_BM"] - 1))
        assert r["capital"]["installed_cost_usd"] == pytest.approx(sum(i["installed_cost_usd"] for i in r["capital"]["items"]))
