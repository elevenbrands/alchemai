"""Boilers and fired heaters (bloque M18): combustion and heat-loss efficiency, radiant section, convection bank.

Published numbers reproduced here were read on 2026-11-14:
  * U.S. DOE Steam Tip Sheet #4, «Improve Your Boiler's Combustion Efficiency», DOE/GO-102012-3405, January 2012: table
    «Combustion Efficiency for Natural Gas» (25 values, dry air, complete combustion).
  * G. Harrell, «Steam System Survey Guide», ORNL/TM-2001/263, May 2002: Table B.1 «Natural gas stack loss (%)» (180 values,
    sample below) and Table 3 (flue O2 versus excess air: the pairs show the O2 is on a wet basis).
  * H. Al-Haj Ibrahim and M. Al-Qassimi, Periodica Polytechnica Chem. Eng. 52/2 (2008) 61-69, example 2, and H. Al-Haj
    Ibrahim, «Fired Process Heaters», in Matlab - Modelling, Programming and Simulations, Sciyo/IntechOpen 2010, Tables 3-5.
  * D. Q. Kern, Process Heat Transfer, chapter 19, examples 19.1, 19.4 and 19.5, as transcribed in the FOSSEE (IIT Bombay)
    Python Textbook Companion notebook «Chapter 19: Furnace Calculations» (the book itself was not opened).
No network, no DWSIM."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from packages.dwsim_client import tools_combustion as TCB
from packages.energy import combustion as CB
from packages.energy import fired_heater as FH

P = 101325.0
T70F = (70 + 459.67) / 1.8
CH4 = {"type": "gas", "composition": {"CH4": 1.0}, "T_K": T70F}
DRY_209 = {"T_K": T70F, "humidity_ratio_kg_kg": 0.0, "dry_air_mole_fractions": {"O2": 0.209, "N2": 0.791}}
DRY = {"T_K": T70F, "humidity_ratio_kg_kg": 0.0}
NG = {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 300.0}
AIR = {"T_K": 300.0, "humidity_ratio_kg_kg": 0.008}

DOE_TIP_4 = {  # (excess air %, O2 % dry): efficiency at flue - air = 200, 300, 400, 500, 600 °F
    (9.5, 2.0): (85.4, 83.1, 80.8, 78.4, 76.0), (15.0, 3.0): (85.2, 82.8, 80.4, 77.9, 75.4),
    (28.1, 5.0): (84.7, 82.1, 79.5, 76.7, 74.0), (44.9, 7.0): (84.1, 81.2, 78.2, 75.2, 72.1),
    (81.6, 10.0): (82.8, 79.3, 75.6, 71.9, 68.2)}
ORNL_DT = (230, 250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450, 470, 490, 510)
ORNL_B1 = {  # flue O2 % (wet): stack loss % at the 15 temperature differences above
    1.0: (14.49, 14.92, 15.36, 15.79, 16.23, 16.67, 17.11, 17.55, 17.99, 18.43, 18.88, 19.32, 19.77, 20.21, 20.66),
    3.0: (14.98, 15.46, 15.94, 16.42, 16.90, 17.38, 17.87, 18.36, 18.84, 19.33, 19.82, 20.31, 20.80, 21.30, 21.79),
    5.0: (15.59, 16.12, 16.66, 17.20, 17.74, 18.28, 18.82, 19.36, 19.91, 20.46, 21.00, 21.55, 22.10, 22.65, 23.20),
    8.0: (16.86, 17.51, 18.16, 18.82, 19.48, 20.14, 20.80, 21.46, 22.12, 22.79, 23.46, 24.12, 24.79, 25.47, 26.14),
    12.0: (19.87, 20.80, 21.73, 22.66, 23.60, 24.54, 25.48, 26.43, 27.37, 28.32, 29.27, 30.22, 31.18, 32.13, 33.09)}
IBRAHIM_GAS = {"CH4": 0.8043, "C2H6": 0.0902, "C3H8": 0.0454, "i-C4H10": 0.0020, "n-C4H10": 0.0032, "i-C5H12": 0.0004,
               "n-C5H12": 0.0002, "CO2": 0.0352, "H2S": 0.0009, "N2": 0.01735}       # 2010 chapter, Table 3 (sums 0.99915)


def eff(fuel, air, stack, **kw):
    kw.setdefault("setting_loss_fraction", 0.0)
    kw.setdefault("setting_loss_basis", "HHV")
    return CB.combustion_efficiency(fuel, air, P, stack_T_K=stack, adiabatic_flame=False, **kw)


# ----------------------------------------------------------------------------------------------- part A: combustion

def test_heating_values_of_methane_against_an_independent_data_source():
    from chemicals.reaction import Hfg
    r = CB.combustion_efficiency(CH4, DRY, P, excess_air_fraction=0.1, adiabatic_flame=False)
    lhv_chem = Hfg("74-82-8") - Hfg("124-38-9") - 2 * Hfg("7732-18-5")
    assert r["fuel"]["LHV_J_mol"] == pytest.approx(lhv_chem, rel=2e-3)
    assert r["fuel"]["HHV_J_mol"] - r["fuel"]["LHV_J_mol"] == pytest.approx(2 * CB._latent_298())
    assert CB._latent_298() == pytest.approx(44004, abs=5)
    assert r["air"]["stoichiometric_dry_air_mol_per_mol_fuel"] == pytest.approx(2 / 0.21)
    assert r["flue_gas"]["mol_per_mol_fuel"] == pytest.approx(1 + 1.1 * 2 / 0.21)       # moles are conserved for CH4
    assert r["evidence"]["verified"] and r["evidence"]["mass_balance_error"] < 1e-12


def test_flue_gas_enthalpy_matches_coolprop_ideal_gas():
    import CoolProp.CoolProp as CP
    for sp, name in (("N2", "Nitrogen"), ("CO2", "CarbonDioxide"), ("H2O", "Water"), ("O2", "Oxygen")):
        st = CP.AbstractState("HEOS", name)
        for T in (500.0, 900.0, 1400.0):
            st.update(CP.DmolarT_INPUTS, 1e-6, T)
            h1 = st.hmolar()
            st.update(CP.DmolarT_INPUTS, 1e-6, CB.T_REF)
            assert CB.h_molar(sp, T) - CB.h_molar(sp, CB.T_REF) == pytest.approx(h1 - st.hmolar(), rel=4e-3)


def test_doe_steam_tip_sheet_4_table():
    diffs = []
    for (ea, o2), row in DOE_TIP_4.items():
        for dT, published in zip((200, 300, 400, 500, 600), row):
            r = eff(CH4, DRY_209, T70F + dT / 1.8, excess_air_fraction=ea / 100)
            diffs.append(100 * r["efficiency"]["HHV"]["efficiency"] - published)
            assert r["efficiency"]["stack_loss_only_efficiency_HHV"] == pytest.approx(r["efficiency"]["HHV"]["efficiency"])
        assert 100 * r["flue_gas"]["dry_mole_fractions"]["O2"] == pytest.approx(o2, abs=0.04)   # excess air <-> dry O2
    diffs = np.array(diffs)
    assert 0.25 < diffs.min() and diffs.max() < 0.70 and diffs.mean() == pytest.approx(0.44, abs=0.05)
    # the table does not say which gas; with a 85/10/3/2 natural gas the largest difference grows to about 1.2 points
    rich = {"type": "gas", "composition": {"CH4": 0.85, "C2H6": 0.10, "C3H8": 0.03, "N2": 0.02}, "T_K": T70F}
    worst = max(abs(100 * eff(rich, DRY, T70F + dT / 1.8, excess_air_fraction=ea / 100)["efficiency"]["HHV"]["efficiency"] - v)
                for (ea, _), row in DOE_TIP_4.items() for dT, v in zip((200, 300, 400, 500, 600), row))
    assert 0.9 < worst < 1.3


def test_ornl_steam_survey_guide_stack_loss_table():
    diffs = []
    for o2, row in ORNL_B1.items():
        for dT, published in zip(ORNL_DT, row):
            r = eff(CH4, DRY, T70F + dT / 1.8, flue_O2={"mole_fraction": o2 / 100, "basis": "wet"})
            diffs.append(100 * (1 - r["efficiency"]["HHV"]["efficiency"]) - published)
    diffs = np.array(diffs)
    assert 0.25 < diffs.min() and diffs.max() < 0.60 and diffs.mean() == pytest.approx(0.39, abs=0.05)
    # Table 3 of the guide: 1.5 / 3.0 / 7.0 % O2 <-> 8.5 / 18.0 / 55.0 % excess air, which only a wet basis gives
    for o2, ea in ((1.5, 8.5), (3.0, 18.0), (7.0, 55.0)):
        wet = CB.solve_combustion(CH4, DRY, P, flue_O2={"mole_fraction": o2 / 100, "basis": "wet"}).excess_air
        dry = CB.solve_combustion(CH4, DRY, P, flue_O2={"mole_fraction": o2 / 100, "basis": "dry"}).excess_air
        assert 100 * wet == pytest.approx(ea, abs=0.6) and abs(100 * dry - ea) > 1.0


def test_ibrahim_2008_natural_gas_heater_stoichiometry():
    s = sum(IBRAHIM_GAS.values())
    fuel = {"type": "gas", "composition": {k: v / s for k, v in IBRAHIM_GAS.items()}, "T_K": 298.15}
    y_w = 0.015                                                # the paper: «molar fraction of water ... of air .015»
    air = {"T_K": 298.15, "humidity_ratio_kg_kg": y_w / (1 - y_w) * 18.01528 / 28.9582}
    r = CB.combustion_efficiency(fuel, air, P, excess_air_fraction=0.25, stack_T_K=446 + 273.15, setting_loss_fraction=0.05,
                                 setting_loss_basis="LHV", adiabatic_flame=False)
    assert r["fuel"]["HHV_J_mol"] == pytest.approx(976029.6, rel=5e-4)                 # paper's gross calorific value
    wet_air = r["air"]["stoichiometric_dry_air_mol_per_mol_fuel"] * 1.25 / (1 - y_w) * 120
    assert wet_air == pytest.approx(1589.014, rel=2e-3)                                # kmol/h for 120 kmol/h of fuel
    assert 120 * r["flue_gas"]["mol_per_mol_fuel"] == pytest.approx(1720.9, rel=2e-3)
    x = r["flue_gas"]["wet_mole_fractions"]
    assert x["CO2"] == pytest.approx(0.08234, rel=5e-3) and x["O2"] == pytest.approx(0.0382, rel=5e-3)
    assert x["H2O"] == pytest.approx(0.15968, rel=1e-2) and x["N2"] + x["Ar"] == pytest.approx(0.7179, rel=6e-3)
    # the paper's net calorific value (927 844.41) is not its gross value minus the latent heat of the water this gas forms
    water = r["fuel"]["elements_mol_per_kg"]["H"] / 2 * r["fuel"]["molar_mass_kg_kmol"] / 1000
    assert water == pytest.approx(2.093, abs=0.005)
    assert (976029.6 - 927844.41) / (water * CB._latent_298()) == pytest.approx(0.523, abs=0.005)
    assert r["fuel"]["LHV_J_mol"] == pytest.approx(884100, rel=1e-3)
    assert 100 * r["efficiency"]["LHV"]["efficiency_on_total_input"] == pytest.approx(73.2, abs=0.2)   # paper: 75.78 with its NCV


def test_oxygen_reading_gives_back_the_excess_air_and_co_is_a_loss():
    for basis in ("dry", "wet"):
        a = CB.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.22, adiabatic_flame=False)
        y = a["flue_gas"]["dry_mole_fractions" if basis == "dry" else "wet_mole_fractions"]["O2"]
        b = CB.combustion_efficiency(NG, AIR, P, flue_O2={"mole_fraction": y, "basis": basis}, adiabatic_flame=False)
        assert b["air"]["excess_air_fraction"] == pytest.approx(0.22, abs=1e-9)
    clean = eff(NG, AIR, 480.0, flue_O2={"mole_fraction": 0.03, "basis": "dry"})
    dirty = eff(NG, AIR, 480.0, flue_O2={"mole_fraction": 0.03, "basis": "dry"}, flue_CO_dry_mole_fraction=2000e-6)
    assert dirty["flue_gas"]["dry_mole_fractions"]["CO"] == pytest.approx(2000e-6)
    assert dirty["flue_gas"]["dry_mole_fractions"]["O2"] == pytest.approx(0.03)
    loss = dirty["efficiency"]["LHV"]["losses_fraction_of_heating_value"]["unburned_CO"]
    n_co = dirty["flue_gas"]["wet_mole_fractions"]["CO"] * dirty["flue_gas"]["mol_per_kg_fuel"]
    co_heat = CB.h_molar("CO", CB.T_REF) - CB.h_molar("CO2", CB.T_REF)                # J/mol, O2 is the element reference
    assert co_heat == pytest.approx(282.98e3, rel=1e-3) and loss == pytest.approx(n_co * co_heat / dirty["fuel"]["LHV_J_kg"])
    assert dirty["efficiency"]["LHV"]["efficiency"] < clean["efficiency"]["LHV"]["efficiency"]
    assert dirty["evidence"]["verified"] and dirty["evidence"]["energy_balance_error"] < 1e-9


def test_losses_add_up_and_trends():
    r = eff(NG, AIR, 450.0, excess_air_fraction=0.15, setting_loss_fraction=0.02, setting_loss_basis="LHV")
    for basis in ("LHV", "HHV"):
        e = r["efficiency"][basis]
        assert e["check_one_minus_losses_plus_credits"] == pytest.approx(e["efficiency"], abs=1e-12)
    assert r["efficiency"]["HHV"]["efficiency"] < r["efficiency"]["LHV"]["efficiency"] < 1
    assert r["efficiency"]["HHV"]["efficiency"] * r["fuel"]["HHV_J_kg"] == pytest.approx(
        r["efficiency"]["LHV"]["efficiency"] * r["fuel"]["LHV_J_kg"])
    assert r["efficiency"]["LHV"]["losses_fraction_of_heating_value"]["setting"] == pytest.approx(0.02)
    assert r["efficiency"]["LHV"]["losses_fraction_of_heating_value"]["air_moisture_sensible"] > 0
    assert eff(NG, AIR, 550.0, excess_air_fraction=0.15)["efficiency"]["LHV"]["efficiency"] < \
        eff(NG, AIR, 450.0, excess_air_fraction=0.15)["efficiency"]["LHV"]["efficiency"]
    assert eff(NG, AIR, 450.0, excess_air_fraction=0.40)["efficiency"]["LHV"]["efficiency"] < \
        eff(NG, AIR, 450.0, excess_air_fraction=0.15)["efficiency"]["LHV"]["efficiency"]
    import CoolProp.CoolProp as CP
    fg = r["flue_gas"]
    assert CP.PropsSI("P", "T", fg["water_dew_point_K"], "Q", 0, "Water") == pytest.approx(fg["water_mole_fraction"] * P, rel=1e-6)
    cold = eff(NG, AIR, fg["water_dew_point_K"] - 5.0, excess_air_fraction=0.15)
    assert any("punto de rocío" in w for w in cold["warnings"])


def test_fuel_oil_by_ultimate_analysis():
    oil = {"type": "ultimate", "mass_fractions": {"C": 0.83, "H": 0.10, "S": 0.05, "N": 0.01, "O": 0.01}, "HHV_J_kg": 43.0e6}
    r = eff(oil, AIR, 600.0, excess_air_fraction=0.40, setting_loss_fraction=0.03, setting_loss_basis="LHV")
    # theoretical air of the 2008 paper, step 1: [11.43 C + 34.5 (H - O/8) + 4.32 S]/100 kg/kg with C, H, O, S in % weight
    assert r["air"]["stoichiometric_dry_air_kg_per_kg_fuel"] == pytest.approx((11.43 * 83 + 34.5 * (10 - 1 / 8) + 4.32 * 5) / 100,
                                                                              rel=1.5e-2)
    water = 0.10 / 1.008e-3 / 2
    assert r["fuel"]["HHV_J_kg"] - r["fuel"]["LHV_J_kg"] == pytest.approx(water * CB._latent_298(), rel=1e-3)
    assert r["flue_gas"]["dry_mole_fractions"]["SO2"] > 0 and r["evidence"]["verified"]
    assert any("Calor sensible del combustible NO incluido" in w for w in r["warnings"])
    lhv = eff(dict(oil, HHV_J_kg=None, LHV_J_kg=r["fuel"]["LHV_J_kg"]), AIR, 600.0, excess_air_fraction=0.40)
    assert lhv["fuel"]["HHV_J_kg"] == pytest.approx(43.0e6)
    wet = dict(oil, mass_fractions={"C": 0.75, "H": 0.09, "S": 0.04, "N": 0.01, "O": 0.01, "moisture": 0.08, "ash": 0.02})
    w = eff(wet, AIR, 600.0, excess_air_fraction=0.40)
    assert w["flue_gas"]["ash_kg_per_kg_fuel"] == pytest.approx(0.02) and w["evidence"]["mass_balance_error"] < 1e-12
    assert w["efficiency"]["LHV"]["losses_fraction_of_heating_value"]["fuel_moisture_sensible"] > 0


def test_adiabatic_flame_reuses_the_cantera_route():
    r = CB.combustion_efficiency(dict(NG, T_K=300.0), {"T_K": 300.0, "humidity_ratio_kg_kg": 0.0}, P, excess_air_fraction=0.10)
    fl = r["adiabatic_flame"]
    eq = fl["equilibrium"]
    assert eq["mechanism"] == "gri30.yaml" and eq["verified"]
    assert fl["T_adiabatic_complete_combustion_K"] == pytest.approx(eq["T_adiabatic_complete_combustion_K"], abs=3.0)
    assert 0 < fl["T_adiabatic_complete_combustion_K"] - eq["T_adiabatic_equilibrium_K"] < 80
    s = sum(IBRAHIM_GAS.values())
    heavy = CB.combustion_efficiency({"type": "gas", "composition": {k: v / s for k, v in IBRAHIM_GAS.items()}, "T_K": 300.0},
                                     {"T_K": 300.0, "humidity_ratio_kg_kg": 0.0}, P, excess_air_fraction=0.25)
    assert heavy["adiabatic_flame"]["equilibrium"] is None and "GRI-Mech" in heavy["adiabatic_flame"]["equilibrium_note"]
    assert heavy["adiabatic_flame"]["T_adiabatic_complete_combustion_K"] > 1900
    cold = CB.combustion_efficiency(CH4, DRY, P, excess_air_fraction=0.10)       # 294 K is below the GRI polynomials
    assert cold["adiabatic_flame"]["equilibrium"] is None and cold["adiabatic_flame"]["T_adiabatic_complete_combustion_K"] > 2000


def test_duty_fuel_and_the_link_to_utility_costs():
    from packages.dwsim_client import tools_utilities as U
    kw = dict(excess_air_fraction=0.15, setting_loss_fraction=0.02, setting_loss_basis="LHV")
    a = eff(NG, AIR, 450.0, absorbed_duty_W=10e6, fuel_price_basis="HHV", **kw)
    f = a["firing"]
    assert f["absorbed_duty_W"] == pytest.approx(10e6)
    assert f["fuel_kg_s"] * a["fuel"]["HHV_J_kg"] * a["efficiency"]["HHV"]["efficiency"] == pytest.approx(10e6)
    assert f["flue_gas_kg_s"] == pytest.approx(f["fuel_kg_s"] + f["moist_air_kg_s"])
    b = eff(NG, AIR, 450.0, fuel_flow_kg_s=f["fuel_kg_s"], **kw)
    assert b["firing"]["absorbed_duty_W"] == pytest.approx(10e6)
    link = a["for_utility_costs"]
    assert link["efficiency_to_pass"] == a["efficiency"]["HHV"]["efficiency"] and "MISMA base" in link["rule"]
    assert "+" in link["basis_mismatch_warning"] or "-" in link["basis_mismatch_warning"]
    cost = U.utility_costs("USD", prices={"fuel_per_GJ": 5.0, "fuel_basis": "HHV"},
                           fired_heat={"furnace_efficiency": link["efficiency_to_pass"]}, today="2026-11-14")
    assert cost["prices"]["heating_usd_GJ"] == pytest.approx(5.0 / a["efficiency"]["HHV"]["efficiency"])
    with pytest.raises(ValueError, match="no los dos"):
        eff(NG, AIR, 450.0, absorbed_duty_W=1e6, fuel_flow_kg_s=0.1, **kw)


def test_operating_data_are_required():
    with pytest.raises(ValueError, match="exceso de aire típico"):
        CB.combustion_efficiency(NG, AIR, P)
    with pytest.raises(ValueError, match="uno solo"):
        CB.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.1, flue_O2={"mole_fraction": 0.03, "basis": "dry"})
    with pytest.raises(ValueError, match="base no se supone"):
        CB.combustion_efficiency(NG, AIR, P, flue_O2={"mole_fraction": 0.03})
    with pytest.raises(ValueError, match="valor típico"):
        CB.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.1, stack_T_K=450.0)
    with pytest.raises(ValueError, match="valor típico"):
        CB.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.1, stack_T_K=450.0, setting_loss_fraction=0.02)
    with pytest.raises(ValueError, match="humidity_ratio_kg_kg"):
        CB.combustion_efficiency(NG, {"T_K": 300.0}, P, excess_air_fraction=0.1)
    with pytest.raises(ValueError, match="presión absoluta"):
        CB.combustion_efficiency(NG, AIR, None, excess_air_fraction=0.1)
    with pytest.raises(ValueError, match="no se supone"):
        CB.combustion_efficiency({"type": "gas", "composition": {"CH4": 1.0}}, AIR, P, excess_air_fraction=0.1)
    with pytest.raises(ValueError, match="no porcentajes"):
        CB.combustion_efficiency({"type": "gas", "composition": {"CH4": 95, "N2": 5}, "T_K": 300}, AIR, P, excess_air_fraction=0.1)
    with pytest.raises(ValueError, match="no se estima por correlación"):
        CB.combustion_efficiency({"type": "ultimate", "mass_fractions": {"C": 0.85, "H": 0.15}}, AIR, P, excess_air_fraction=0.1)
    with pytest.raises(ValueError, match="nasa_gas.yaml"):
        CB.combustion_efficiency({"type": "gas", "composition": {"unobtainium": 1.0}, "T_K": 300}, AIR, P, excess_air_fraction=0.1)
    with pytest.raises(ValueError, match="stack_T_K"):
        CB.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.1, absorbed_duty_W=1e6)


# ----------------------------------------------------------------------------------------------- part B: radiation

def _view_factor_numeric(ratio: float) -> float:
    """Diffuse plane -> one infinite row of tubes at pitch/diameter = ratio, by quadrature over position and angle."""
    C, R, h = ratio, 0.5, 1.0
    x = (np.arange(400) + 0.5) / 400 * C
    th = (np.arange(4000) + 0.5) / 4000 * math.pi - math.pi / 2
    s, c = np.sin(th)[None, :], np.cos(th)[None, :]
    u = (h * s + x[:, None] * c) / (C * c)
    dist = np.minimum(u - np.floor(u), np.ceil(u) - u) * C * c
    return float(((dist < R) * c).sum() / (np.ones_like(dist) * c).sum())


def test_tube_row_effectiveness_is_the_exact_view_factor_and_matches_kern():
    for ratio in (1.5, 1.7, 2.0, 3.0):
        row = FH.tube_row_alpha(ratio, 1.0)
        assert row["direct_view_factor"] == pytest.approx(_view_factor_numeric(ratio), abs=2e-3)
        assert row["alpha"] == pytest.approx(row["direct_view_factor"] * (2 - row["direct_view_factor"]))
    assert FH.tube_row_alpha(8.5, 5.0)["alpha"] == pytest.approx(0.937, abs=0.003)     # Kern ex. 19.1, Fig. 19.11
    assert FH.tube_row_alpha(1.0, 1.0)["alpha"] == pytest.approx(1.0)


def test_exchange_factor_limits_and_kern_chart_readings():
    assert FH.exchange_factor(0.4, 10.0, 10.0, 0.85) == pytest.approx(1 / (1 / 0.4 + 1 / 0.85 - 1))   # no refractory
    assert FH.exchange_factor(1 - 1e-12, 3.0, 10.0, 0.85) == pytest.approx(0.85, rel=1e-6)            # black gas
    assert FH.exchange_factor(0.3, 2.0, 10.0, 0.9) > FH.exchange_factor(0.3, 8.0, 10.0, 0.9)          # refractory helps
    # Kern Fig. 19.15 as read in examples 19.1 (0.635 at eG 0.496, AR/aAcp 1.09) and 19.5 (0.56 at 0.265, 2.49)
    for eg, ratio, chart in ((0.496, 1.09, 0.635), (0.265, 2.49, 0.56)):
        grey, black = (FH.exchange_factor(eg, 1.0, 1.0 + ratio, et) for et in (0.9, 1.0))
        assert grey == pytest.approx(chart, rel=0.07) and black == pytest.approx(chart, rel=0.07)
    assert FH.exchange_factor(0.496, 1.0, 2.09, 0.9) == pytest.approx(0.626, abs=0.001)


def test_gas_emissivity_wsgg():
    b = FH.SMITH_1982
    assert sum(sum(k) for k, _ in b.values()) == pytest.approx(0.3966 + 15.64 + 394.3 + 0.4303 + 7.055 + 178.1 + 0.4201 + 6.516
                                                                + 131.9 + 0.4098 + 6.325 + 120.5 + 0.4496 + 7.113 + 119.7)
    atm, ft = 101325.0, 0.3048
    for T in (600.0, 1200.0, 2400.0):                           # tabulated mixtures: 0.1/0.1 and 0.2/0.1 atm
        for pw, pc in ((0.1, 0.1), (0.2, 0.1)):
            vals = [FH.gas_emissivity(T, pw * atm, pc * atm, L) for L in (0.05, 0.5, 5.0, 30.0)]
            eps = [v["emissivity"] for v in vals]
            assert all(0 < e < 1 for e in eps) and eps == sorted(eps) and all(v["in_fitted_range"] for v in vals)
    node = FH.gas_emissivity(1200.0, 0.2 * atm, 0.1 * atm, 3.0)
    k, bb = b["ratio_2"]
    assert node["emissivity"] == pytest.approx(sum(sum(bb[i][j] * 1200.0 ** j for j in range(4)) * (1 - math.exp(-k[i] * 0.9))
                                                   for i in range(3)))
    assert not FH.gas_emissivity(500.0, 0.1 * atm, 0.1 * atm, 1.0)["in_fitted_range"]
    assert not FH.gas_emissivity(1200.0, 0.3 * atm, 0.3 * atm, 1.0, 3 * atm)["in_fitted_range"]
    # Kern's charts (Figs. 19.12-19.13) as read in two examples, pCO2 0.1084 and pH2O 0.1248 atm
    short = FH.gas_emissivity((1500 + 459.67) / 1.8, 0.1248 * atm, 0.1084 * atm, 0.565 * ft)["emissivity"]
    assert short == pytest.approx(2750 / (0.173 * 19.5967 ** 4), rel=0.08)             # ex. 19.4: within 8 %
    long_ = FH.gas_emissivity((1730 + 459.67) / 1.8, 0.1248 * atm, 0.1084 * atm, 15.0 * ft)["emissivity"]
    assert long_ / 0.496 == pytest.approx(0.80, abs=0.02)        # ex. 19.1: 20 % BELOW the chart value (declared limit)
    # gas radiation coefficient of ex. 19.4 (2.52 Btu/h ft2 °F with Kern's own beam length): this model gives 20 % more
    h = FH.gas_radiation_coefficient((1500 + 459.67) / 1.8, (650 + 459.67) / 1.8, short, 0.9) / 5.678263
    assert h / 2.52 == pytest.approx(1.20, abs=0.03)


GEO = {"tube_outside_diameter_m": 0.1143, "tube_pitch_m": 0.2032, "tube_effective_length_m": 12.0, "n_tubes": 60,
       "envelope_area_m2": 2 * (12 * 6 + 12 * 7 + 6 * 7), "volume_m3": 12 * 6 * 7.0}
RAD = dict(fuel=NG, air=AIR, pressure_Pa=P, geometry=GEO, tube_emissivity=0.85, convective_coefficient_W_m2K=10.0,
           setting_loss_fraction=0.02, setting_loss_basis="LHV")


def test_radiant_section_internal_consistency():
    runs = [FH.radiant_section(fuel_flow_kg_s=m, excess_air_fraction=0.15, tube_wall_T_K=700.0, **RAD) for m in (0.2, 0.3, 0.4)]
    for r in runs:
        assert r["evidence"]["energy_balance_error"] < 1e-9 and r["evidence"]["verified_internal"]
        assert 700.0 < r["bridgewall_T_K"] < r["adiabatic_complete_combustion_T_K"]
        total = r["radiant_duty_W"] + r["shield_radiant_duty_W"] + r["flue_gas_sensible_leaving_W"] + r["setting_loss_W"]
        assert total == pytest.approx(r["heat_release_LHV_W"], rel=0.01)       # LHV + the small sensible heat of fuel and air
        assert r["radiation"]["mean_beam_length_m"] == pytest.approx(3.6 * GEO["volume_m3"] / GEO["envelope_area_m2"])
        assert r["average_flux_W_m2_tube_outside"] == pytest.approx(r["radiant_duty_W"] / r["tube_area_m2"])
    assert runs[0]["radiant_duty_W"] < runs[1]["radiant_duty_W"] < runs[2]["radiant_duty_W"]       # more firing, more duty
    assert runs[0]["absorbed_fraction_of_LHV"] > runs[1]["absorbed_fraction_of_LHV"] > runs[2]["absorbed_fraction_of_LHV"]
    assert runs[0]["bridgewall_T_K"] < runs[1]["bridgewall_T_K"] < runs[2]["bridgewall_T_K"]
    fr = [FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=e, tube_wall_T_K=700.0, **RAD)["absorbed_fraction_of_LHV"]
          for e in (0.05, 0.15, 0.30, 0.60)]
    assert fr == sorted(fr, reverse=True)                                                          # more air, less absorbed
    proc = {"bulk_T_K": 600.0, "inside_coefficient_W_m2K": 1200.0, "tube_inside_diameter_m": 0.1023,
            "wall_conductivity_W_mK": 25.0, "inside_fouling_m2K_W": 0.0005}
    w = FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=0.15, process=proc, **RAD)
    resistance = 0.1143 / (0.1023 * 1200.0) + 0.0005 * 0.1143 / 0.1023 + 0.1143 * math.log(0.1143 / 0.1023) / 50.0
    assert w["tube_wall_T_K"] == pytest.approx(600.0 + resistance * w["average_flux_W_m2_tube_outside"], abs=1e-4)
    hot = FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=0.15, tube_wall_T_K=700.0, gas_emissivity_override=0.55, **RAD)
    assert hot["radiant_duty_W"] > runs[1]["radiant_duty_W"] and hot["radiant_duty_W"] / runs[1]["radiant_duty_W"] < 1.10
    with pytest.raises(ValueError, match="uno solo"):
        FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=0.15, **RAD)
    with pytest.raises(ValueError, match="no se suponen"):
        FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=0.15, tube_wall_T_K=700.0,
                           **dict(RAD, geometry={k: v for k, v in GEO.items() if k != "envelope_area_m2"}))
    with pytest.raises(ValueError, match="tube_emissivity"):
        FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=0.15, tube_wall_T_K=700.0, **dict(RAD, tube_emissivity=None))


def test_ibrahim_2010_radiant_example_is_internally_inconsistent():
    # eq. (30) of the chapter, F(Tg) = 9.0748e-5 Tg^4 + 7.9153e4 Tg - 1.5533e8; the text says its root is 1278 K
    root = brentq(lambda T: 9.0748e-5 * T ** 4 + 7.9153e4 * T - 1.5533e8, 500.0, 3000.0)
    assert root == pytest.approx(965.6, abs=0.2) and abs(root - 1278.0) > 300
    sigma_kJ_h = FH.SIGMA * 3.6
    assert sigma_kJ_h * 0.97 * (0.835 * 60 * 0.394 * 20.024 + 8 * 0.394 * 20.024) == pytest.approx(9.0748e-5, rel=1e-3)
    s = sum(IBRAHIM_GAS.values())
    fuel = {"type": "gas", "composition": {k: v / s for k, v in IBRAHIM_GAS.items()}, "T_K": 298.15}
    geo = {"tube_outside_diameter_m": 0.219, "tube_pitch_m": 0.394, "tube_effective_length_m": 20.024, "n_tubes": 60,
           "envelope_area_m2": 5000.0, "alpha": 0.835, "shield_cold_plane_area_m2": 8 * 0.394 * 20.024}
    m = 120e3 / 3600 * 20.046e-3
    r = FH.radiant_section(fuel, {"T_K": 298.15, "relative_humidity": 0.4}, P, m, geo, 0.9, 30.66e3 / 3600, 0.05, "LHV",
                           excess_air_fraction=0.25, tube_wall_T_K=100 + 0.5 * (483 + 628), exchange_factor_override=0.97)
    assert r["bridgewall_T_K"] == pytest.approx(root, abs=25.0)          # same inputs, NASA enthalpies and the true LHV
    assert r["radiant_duty_W"] * 3.6 / 6.182e7 == pytest.approx(0.91, abs=0.02)   # chapter's Table 6: 6.182e7 kJ/h


def test_radiant_sizing():
    s = FH.radiant_sizing(10e6, 35e3, 0.1143, 12.0, 0.2032)
    assert s["tube_area_required_m2"] == pytest.approx(10e6 / 35e3)
    assert s["n_tubes"] == math.ceil(10e6 / 35e3 / (math.pi * 0.1143 * 12.0)) and s["average_flux_installed_W_m2"] <= 35e3
    with pytest.raises(ValueError, match="no hay valor típico"):
        FH.radiant_sizing(10e6, None, 0.1143, 12.0, 0.2032)
    assert TCB.fired_heater_radiant("sizing", required_radiant_duty_W=10e6, allowable_average_flux_W_m2=35e3,
                                    geometry={"tube_outside_diameter_m": 0.1143, "tube_effective_length_m": 12.0,
                                              "tube_pitch_m": 0.2032})["n_tubes"] == s["n_tubes"]
    with pytest.raises(ValueError, match="rating"):
        TCB.fired_heater_radiant("design")


BANK = {"tube_outside_diameter_m": 0.1143, "tube_inside_diameter_m": 0.1023, "tube_effective_length_m": 12.0,
        "tubes_per_row": 8, "n_rows": 10, "transverse_pitch_m": 0.2032, "longitudinal_pitch_m": 0.176,
        "arrangement": "staggered", "wall_conductivity_W_mK": 25.0, "outside_fouling_m2K_W": 0.0, "tube_emissivity": 0.85}
PROC = {"inlet_T_K": 520.0, "capacity_rate_W_K": 60e3, "inside_coefficient_W_m2K": 1200.0, "inside_fouling_m2K_W": 0.0005}


def test_convection_section_and_the_whole_chain():
    rad = FH.radiant_section(fuel_flow_kg_s=0.3, excess_air_fraction=0.15, tube_wall_T_K=700.0, **RAD)
    args = (NG, AIR, P, 0.3, rad["bridgewall_T_K"], BANK, PROC)
    cv = FH.convection_section(*args, excess_air_fraction=0.15)
    assert cv["evidence"]["energy_balance_error"] < 1e-9
    # M18 asserted `validated_against_literature is False`; M18b validated the bank (tests/test_fired_heater_validation.py)
    assert cv["evidence"]["validated_against_literature"] == FH.CONVECTION_VALIDATION
    assert PROC["inlet_T_K"] < cv["stack_T_K"] < rad["bridgewall_T_K"]
    assert cv["duty_W"] == pytest.approx(PROC["capacity_rate_W_K"] * (cv["process_outlet_T_K"] - PROC["inlet_T_K"]), rel=1e-6)
    gas = [r["gas_out_K"] for r in cv["rows"]]
    assert gas == sorted(gas, reverse=True) and len(cv["rows"]) == 10
    assert all(r["process_K"] <= r["tube_wall_K"] <= r["gas_in_K"] for r in cv["rows"])
    assert cv["mean_beam_length_m"] == pytest.approx(FH.bank_beam_length(0.2032, 0.176, 0.1143))
    bare = FH.convection_section(*args, excess_air_fraction=0.15, gas_radiation=False)
    assert bare["stack_T_K"] > cv["stack_T_K"] and bare["gas_radiation_share_of_duty"] == 0 < cv["gas_radiation_share_of_duty"]
    # the convective coefficient of a row is Zukauskas with Cantera properties at the row's mean temperatures. For this
    # staggered bank (St/Sl = 1.15, Re 1e3-2e5) the engine's own function (M18b) and ht's coincide, so the assertion stays
    from ht.conv_tube_bank import Nu_Zukauskas_Bejan
    c = CB.solve_combustion(NG, AIR, P, excess_air_fraction=0.15)
    row = bare["rows"][3]
    Tg = 0.5 * (row["gas_in_K"] + row["gas_out_K"])
    pr, pw = FH._gas_properties(c, Tg, P), FH._gas_properties(c, row["tube_wall_K"], P)
    Re = bare["mass_velocity_kg_m2s"] * 0.1143 / pr["mu"]
    assert bare["mass_velocity_kg_m2s"] == pytest.approx(0.3 * c.flue_mass / (8 * (0.2032 - 0.1143) * 12.0))
    assert row["h_convection_W_m2K"] == pytest.approx(
        Nu_Zukauskas_Bejan(Re=Re, Pr=pr["Pr"], tube_rows=10, pitch_parallel=0.176, pitch_normal=0.2032, Pr_wall=pw["Pr"])
        * pr["k"] / 0.1143, rel=1e-9)
    more = FH.convection_section(NG, AIR, P, 0.3, rad["bridgewall_T_K"], dict(BANK, n_rows=16), PROC, excess_air_fraction=0.15)
    assert more["stack_T_K"] < cv["stack_T_K"] and more["duty_W"] > cv["duty_W"]
    boil = FH.convection_section(NG, AIR, P, 0.3, rad["bridgewall_T_K"], BANK,
                                 {"inlet_T_K": 520.0, "isothermal": True, "inside_coefficient_W_m2K": 5000.0,
                                  "inside_fouling_m2K_W": 0.0}, excess_air_fraction=0.15)
    assert boil["process_outlet_T_K"] == 520.0 and boil["evidence"]["energy_balance_error"] < 1e-9
    # chain: the stack temperature found here, given to combustion_efficiency, returns the duty of the two sections
    e = CB.combustion_efficiency(NG, AIR, P, excess_air_fraction=0.15, stack_T_K=cv["stack_T_K"], setting_loss_fraction=0.02,
                                 setting_loss_basis="LHV", fuel_flow_kg_s=0.3, adiabatic_flame=False)
    assert e["firing"]["absorbed_duty_W"] == pytest.approx(rad["radiant_duty_W"] + rad["shield_radiant_duty_W"] + cv["duty_W"],
                                                           rel=1e-7)
    with pytest.raises(ValueError, match="no se suponen"):
        FH.convection_section(NG, AIR, P, 0.3, 1100.0, {k: v for k, v in BANK.items() if k != "tube_emissivity"}, PROC,
                              excess_air_fraction=0.15)
    with pytest.raises(ValueError, match="capacity_rate_W_K"):
        FH.convection_section(NG, AIR, P, 0.3, 1100.0, BANK, {k: v for k, v in PROC.items() if k != "capacity_rate_W_K"},
                              excess_air_fraction=0.15)


def test_tools_are_registered_with_spanish_descriptions():
    from packages.agent import agent as AG
    from packages.dwsim_client import tools as T
    names = {f.__name__ for f in T.TOOLS}
    mine = ("combustion_efficiency", "fired_heater_radiant", "fired_heater_convection")
    assert set(mine) <= names and len(T.TOOLS) >= 76
    for name in mine:
        assert any(getattr(t, "name", "") == name for t in AG.AGENT_TOOLS)
    assert "no hay valor típico" in TCB.combustion_efficiency.__doc__ and "utility_costs" in TCB.combustion_efficiency.__doc__
    assert "±5 %" in TCB.fired_heater_radiant.__doc__ and "±35 K" in TCB.fired_heater_radiant.__doc__
    assert "Žukauskas" in TCB.fired_heater_convection.__doc__ and "ESCOA" in TCB.fired_heater_convection.__doc__
    assert "no se modelan" in TCB.fired_heater_convection.__doc__                     # studded tubes: said, not hidden
    assert "27. Boilers and fired heaters" in AG.SYSTEM and "SAME basis" in AG.SYSTEM
    out = TCB.combustion_efficiency(NG, AIR, P, flue_O2={"mole_fraction": 0.03, "basis": "dry"}, stack_T_K=450.0,
                                    setting_loss_fraction=0.02, setting_loss_basis="LHV", absorbed_duty_W=5e6)
    json.dumps(out, allow_nan=False)
    rad = TCB.fired_heater_radiant("rating", fuel_flow_kg_s=0.3, excess_air_fraction=0.15, tube_wall_T_K=700.0, **RAD)
    json.dumps(rad, allow_nan=False)
    json.dumps(TCB.fired_heater_convection(NG, AIR, P, 0.3, rad["bridgewall_T_K"], BANK, PROC, excess_air_fraction=0.15),
               allow_nan=False)
