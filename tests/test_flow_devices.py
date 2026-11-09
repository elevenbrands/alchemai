"""Flow devices (bloque M4a): relief valves against the API 520 Part I worked examples, control valves against the
IEC 60534-2-1 examples, fittings against Darby's 3-K formula, the isothermal gas line against its incompressible
limit and its choke condition, and two-phase ΔP against its single-phase limits. Every closed form here is written
independently of `fluids`."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_flow as TF  # noqa: E402

ATM = 101325.0
SET_670 = (670e3 - ATM) / 1.1               # set pressure (gauge) that gives P1 = 670 kPa abs at 10 % overpressure
GAS = dict(T_K=348.0, MW=51.0, Z=0.90, k=1.11)


def test_relief_gas_critical_api520_example_1():
    r = TF.relief_valve_sizing("gas", 24270 / 3600, SET_670, 0.10, 0.0, **GAS)
    k = GAS["k"]
    C = 0.03948 * math.sqrt(k * (2 / (k + 1)) ** ((k + 1) / (k - 1)))                # API 520 SI, critical flow
    A = 24270 / (C * 0.975 * 670.0) * math.sqrt(348.0 * 0.90 / 51.0)
    assert r["flow"] == "crítico" and r["required_area_mm2"] == pytest.approx(A, rel=1e-9)
    assert r["required_area_mm2"] == pytest.approx(3699, abs=1)                        # example 1 of API 520 Part I
    assert (r["orifice"], r["valves"]) == ("P", 1) and r["evidence"]["verified"]


def test_relief_gas_subcritical_and_back_pressure_warning():
    r = TF.relief_valve_sizing("gas", 24270 / 3600, SET_670, 0.10, 532e3 - ATM, **GAS)
    assert r["flow"] == "subcrítico" and r["required_area_mm2"] == pytest.approx(4248.4, abs=0.5)   # example 2
    assert r["evidence"]["verified"] and "fuelle balanceado" in r["warnings"][0]
    with pytest.raises(ValueError):                                                     # bellows chart stops at 50 %
        TF.relief_valve_sizing("gas", 24270 / 3600, SET_670, 0.10, 532e3 - ATM, valve_type="balanced_bellows", **GAS)
    big = TF.relief_valve_sizing("gas", 40.0, SET_670, 0.10, 0.0, **GAS)
    assert big["valves"] >= 2 and big["orifice_area_mm2"] * big["valves"] >= big["required_area_mm2"]


def test_relief_steam_and_liquid_api520_examples():
    st = TF.relief_valve_sizing("steam", 69615 / 3600, (12236e3 - ATM) / 1.1, 0.10, 0.0, T_K=707.0389)
    assert st["required_area_mm2"] == pytest.approx(1285.2, abs=0.5) and st["evidence"]["verified"]   # example 4, 10th ed.
    Q = 6814 / 6e4
    rho = 0.9 * 999
    li = TF.relief_valve_sizing("liquid", rho * Q, 1724e3, 0.10, 0.2 * 1724e3, density_kg_m3=rho,
                                valve_type="balanced_bellows")
    assert li["required_area_mm2"] == pytest.approx(3066, rel=0.005)                   # example 5 (printed Kw 0.97)
    assert li["evidence"]["verified"]
    visc = TF.relief_valve_sizing("liquid", rho * Q, 1724e3, 0.10, 0.2 * 1724e3, density_kg_m3=rho,
                                  viscosity_Pa_s=0.388, valve_type="balanced_bellows")
    assert visc["Kv"] < 1.0 and visc["required_area_mm2"] > li["required_area_mm2"] and visc["evidence"]["verified"]
    with pytest.raises(ValueError, match="needs T_K"):
        TF.relief_valve_sizing("gas", 1.0, 1e6, 0.10, 0.0)


WATER = dict(density_kg_m3=965.4, vapor_pressure_Pa=70.1e3, critical_pressure_Pa=22120e3, viscosity_Pa_s=3.1472e-4)
CO2 = dict(T_K=433.0, MW=44.01, gamma=1.30, Z=0.988, viscosity_Pa_s=1.4665e-4, xT=0.60)


def test_control_valve_iec_examples():
    l1 = TF.control_valve_sizing("liquid", 680e3, 220e3, 0.9, 0.46, 0.1, **WATER, pipe_inlet_diameter_m=0.15,
                                 pipe_outlet_diameter_m=0.15, valve_diameter_m=0.15)
    assert l1["Kv_m3_h"] == pytest.approx(165.0, abs=0.1) and not l1["choked"] and l1["evidence"]["verified"]  # ex. 1
    l2 = TF.control_valve_sizing("liquid", 680e3, 220e3, 0.6, 0.98, 0.1, **WATER, pipe_inlet_diameter_m=0.1,
                                 pipe_outlet_diameter_m=0.1, valve_diameter_m=0.1)
    assert l2["Kv_m3_h"] == pytest.approx(238.1, abs=0.1) and l2["choked"]                                      # ex. 2
    g3 = TF.control_valve_sizing("gas", 680e3, 310e3, 0.85, 0.42, 38 / 36, **CO2, pipe_inlet_diameter_m=0.08,
                                 pipe_outlet_diameter_m=0.1, valve_diameter_m=0.05)
    assert g3["Kv_m3_h"] == pytest.approx(72.6, abs=0.1) and g3["FP"] < 1 and g3["evidence"]["verified"]      # ex. 3
    assert g3["evidence"]["deviation_pct"] < 5 and g3["Cv_USgpm"] == pytest.approx(1.156 * g3["Kv_m3_h"])


def test_control_valve_closed_forms_without_reducers():
    g = TF.control_valve_sizing("gas", 680e3, 310e3, 0.85, 0.42, 38 / 36, **CO2)
    x, Fg = 370 / 680, 1.3 / 1.4
    Y = 1 - x / (3 * Fg * 0.60)
    assert g["Kv_m3_h"] == pytest.approx(38 / 36 * 3600 / (24.6 * 680 * Y) * math.sqrt(44.01 * 433 * 0.988 / x), rel=1e-9)
    c = TF.control_valve_sizing("liquid", 680e3, 100e3, 0.9, 0.46, 0.1, **WATER)
    FF = 0.96 - 0.28 * math.sqrt(70.1 / 22120)
    assert c["choked"] and c["warnings"]                                            # ΔP 580 kPa > FL²(P1 − FF·Pv)
    assert c["Kv_m3_h"] == pytest.approx(360 / (0.1 * 0.9) * math.sqrt(965.4 / 999.10329075702 / (680 - FF * 70.1)), rel=1e-9)
    with pytest.raises(ValueError, match="all three"):
        TF.control_valve_sizing("liquid", 680e3, 220e3, 0.9, 0.46, 0.1, **WATER, pipe_inlet_diameter_m=0.15)


def test_fittings_darby_3k():
    from fluids.fittings import Darby
    name = "Valve, Angle valve, 45°, full line size, β = 1"
    r = TF.fittings_resistance([{"name": name, "count": 2}], 0.05248, 1e4)                   # 2 in schedule 40
    K1, Ki, Kd = Darby[name]
    assert r["fittings"][0]["K_each"] == pytest.approx(K1 / 1e4 + Ki * (1 + Kd / 2.0 ** 0.3), rel=1e-6)
    assert r["K_total"] == pytest.approx(2 * 1.1572524, rel=1e-6)
    assert len(TF.fittings_resistance([], 0.05, 1e4)["available_fittings"]) == 34
    with pytest.raises(ValueError, match="closest"):
        TF.fittings_resistance([{"name": "codo 90"}], 0.05, 1e4)


METHANE = dict(T_K=300.0, MW=16.04, Z=0.92, k=1.3, viscosity_Pa_s=1.1e-5, inner_diameter_m=0.1)


def test_gas_line_small_drop_is_darcy_and_round_trips():
    small = TF.compressible_gas_line(50e5, **METHANE, length_m=50.0, mass_flow_kg_s=3.0)
    assert small["dP_fraction_of_P1"] < 0.02 and abs(small["incompressible_error_pct"]) < 1.0
    r = TF.compressible_gas_line(50e5, **METHANE, length_m=5000.0, mass_flow_kg_s=3.0)
    assert r["dP_fraction_of_P1"] > 0.4 and abs(r["incompressible_error_pct"]) > 15     # incompressible would be wrong
    assert r["evidence"]["verified"] and r["evidence"]["ode_match"]
    back = TF.compressible_gas_line(50e5, **METHANE, length_m=5000.0, P2_Pa=r["P2_Pa"])
    assert back["mass_flow_kg_s"] == pytest.approx(3.0, rel=1e-9)


def test_gas_line_chokes_at_isothermal_sonic_outlet():
    c = TF.compressible_gas_line(50e5, **METHANE, length_m=5000.0, P2_Pa=1e5)
    assert c["choked"] and not c["evidence"]["verified"] and "warning" in c
    assert c["Mach_out"] == pytest.approx(1 / math.sqrt(1.3), rel=1e-6)                 # v_out = √(ZRT/M)
    with pytest.raises(ValueError, match="chokes"):
        TF.compressible_gas_line(50e5, **METHANE, length_m=5000.0, mass_flow_kg_s=1.01 * c["mass_flow_kg_s"])


TP = dict(liquid_density_kg_m3=915.0, gas_density_kg_m3=2.67, gas_viscosity_Pa_s=14e-6, surface_tension_N_m=0.0487,
          inner_diameter_m=0.05, length_m=1.0)


def test_two_phase_methods_limits_and_regime():
    r = TF.two_phase_pressure_drop(0.6, 0.1, liquid_viscosity_Pa_s=180e-6, **TP)
    assert r["recommended_method"] == "Friedel" and r["evidence"]["verified"]
    assert r["dP_friction_Pa"] == pytest.approx(r["dP_by_method_Pa"]["Friedel"]) and len(r["dP_by_method_Pa"]) == 4
    assert r["flow_regime_Taitel_Dukler"] in ("annular", "bubbly", "intermittent", "stratified wavy", "stratified smooth")
    lo = TF.two_phase_pressure_drop(0.6, 0.01, liquid_viscosity_Pa_s=180e-6, **TP)["dP_friction_Pa"]
    assert lo < r["dP_friction_Pa"]                                                    # more vapor, faster flow, more ΔP
    viscous = TF.two_phase_pressure_drop(0.6, 0.1, liquid_viscosity_Pa_s=0.1, **TP)
    assert viscous["recommended_method"] == "Chisholm"                                 # μl/μg > 1000, G > 100
    with pytest.raises(ValueError, match="between 0 and 1"):
        TF.two_phase_pressure_drop(0.6, 1.0, liquid_viscosity_Pa_s=180e-6, **TP)


def test_results_are_plain_json():
    for r in (TF.relief_valve_sizing("gas", 24270 / 3600, SET_670, 0.10, 0.0, **GAS),
              TF.control_valve_sizing("gas", 680e3, 310e3, 0.85, 0.42, 38 / 36, **CO2, pipe_inlet_diameter_m=0.08,
                                      pipe_outlet_diameter_m=0.1, valve_diameter_m=0.05),
              TF.compressible_gas_line(50e5, **METHANE, length_m=5000.0, P2_Pa=1e5),
              TF.two_phase_pressure_drop(0.6, 0.1, liquid_viscosity_Pa_s=180e-6, **TP)):
        json.dumps(r, allow_nan=False)
