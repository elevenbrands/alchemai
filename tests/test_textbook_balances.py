"""fluid_state (steam-table lookup with any two properties), mechanical_energy_balance and solve_algebraic_system.

References are identities or closed forms: h = u + P·v, the lever rule, ideal-gas limits, Torricelli, a hand-written
Bernoulli sum, roots of polynomials. Two steam-table anchors are quoted from the IAPWS-IF97 verification tables as
I remember them (not looked up): saturated water at 100 °C, h_fg ≈ 2256–2257 kJ/kg and v_g ≈ 1.672–1.674 m³/kg.
"""
import math

import pytest

from packages.dwsim_client import tools_equations as TE
from packages.dwsim_client import tools_momentum as TM
from packages.dwsim_client import tools_properties as TPR


def test_fluid_state_two_phase_lever_rule_and_totals():
    o = TPR.fluid_state("Water", [{"label": "a", "P_Pa": 1e6, "quality": 0.3, "mass_kg": 2.0}])
    assert o["ok"]
    s = o["states"][0]
    sat = s["saturation_at_P"]
    assert s["phase"] == "twophase" and s["T_K"] == pytest.approx(sat["T_sat_K"])
    for k, f, g in (("v_m3_kg", "v_f_m3_kg", "v_g_m3_kg"), ("h_J_kg", "h_f_J_kg", "h_g_J_kg"), ("u_J_kg", "u_f_J_kg", "u_g_J_kg")):
        assert s[k] == pytest.approx(0.7 * sat[f] + 0.3 * sat[g], rel=1e-9)
    assert s["vapor_mass_kg"] == pytest.approx(0.6) and s["U_J"] == pytest.approx(2.0 * s["u_J_kg"])
    assert s["h_J_kg"] - s["u_J_kg"] == pytest.approx(s["P_Pa"] * s["v_m3_kg"], rel=1e-10)


def test_fluid_state_pairs_are_consistent_and_rigid_tank_balance():
    ref = TPR.fluid_state("Water", [{"T_K": 500.0, "P_Pa": 5e5}])["states"][0]
    for pair in (("P_Pa", "h_J_kg"), ("P_Pa", "s_J_kgK"), ("T_K", "v_m3_kg"), ("v_m3_kg", "u_J_kg"), ("P_Pa", "u_J_kg")):
        st = TPR.fluid_state("Water", [{k: ref[k] for k in pair}])["states"][0]
        assert st["T_K"] == pytest.approx(500.0, rel=1e-7) and st["P_Pa"] == pytest.approx(5e5, rel=1e-6)
    # rigid tank: same v, heated from a wet state to a hotter one — Q = m·Δu from consecutive_differences
    o = TPR.fluid_state("Water", [{"label": 1, "T_K": 400.0, "quality": 0.2, "mass_kg": 3.0}])
    v = o["states"][0]["v_m3_kg"]
    o = TPR.fluid_state("Water", [{"label": 1, "T_K": 400.0, "v_m3_kg": v, "mass_kg": 3.0},
                                  {"label": 2, "T_K": 450.0, "v_m3_kg": v, "mass_kg": 3.0}])
    d = o["consecutive_differences"][0]
    assert d["delta_v_m3_kg"] == pytest.approx(0.0, abs=1e-12)
    assert d["delta_U_J"] == pytest.approx(3.0 * (o["states"][1]["u_J_kg"] - o["states"][0]["u_J_kg"])) and d["delta_U_J"] > 0
    assert o["states"][1]["quality"] > o["states"][0]["quality"]


def test_fluid_state_anchors_and_limits():
    s = TPR.fluid_state("Water", [{"T_K": 373.15, "quality": 1.0}])["states"][0]
    sat = s["saturation_at_P"]
    assert s["P_Pa"] == pytest.approx(101.4e3, rel=5e-3)                      # 1 atm boils at ~373.12 K
    assert 2.255e6 < sat["h_g_J_kg"] - sat["h_f_J_kg"] < 2.258e6 and 1.670 < sat["v_g_m3_kg"] < 1.675
    n2 = TPR.fluid_state("Nitrogen", [{"T_K": 300.0, "P_Pa": 1e5}])
    st = n2["states"][0]                                                      # dilute gas: P·v = R·T/M within 0.1 %
    assert st["P_Pa"] * st["v_m3_kg"] == pytest.approx(8.314462618 * 300.0 / n2["molar_mass_kg_mol"], rel=1e-3)
    assert st["superheat_K"] > 200


def test_fluid_state_bad_input():
    assert "error" in TPR.fluid_state("Water", [{"T_K": 400.0}])
    assert "error" in TPR.fluid_state("Water", [{"T_K": 400.0, "P_Pa": 1e5, "quality": 0.5}])
    assert "error" in TPR.fluid_state("Unobtainium", [{"T_K": 400.0, "P_Pa": 1e5}])
    assert "error" in TPR.fluid_state("Water", [{"T_K": 400.0, "enthalpy": 1.0}])
    psat = TPR.fluid_state("Water", [{"T_K": 400.0, "quality": 0.0}])["states"][0]["P_Pa"]
    assert "saturación" in TPR.fluid_state("Water", [{"T_K": 400.0, "P_Pa": psat}])["error"]


def test_mechanical_energy_torricelli_pump_and_inverse_modes():
    tank = {"P_Pa": 101325.0, "z_m": 5.0, "large_surface": True}
    o = TM.mechanical_energy_balance(1000.0, tank, {"P_Pa": 101325.0, "z_m": 0.0, "diameter_m": 0.02}, "volumetric_flow_m3_s")
    assert o["ok"] and o["velocities_m_s"]["point_2"] == pytest.approx(math.sqrt(2 * 9.80665 * 5.0), rel=1e-9)
    q, d = 0.004, 0.05
    u = q / (math.pi * d ** 2 / 4)
    p2 = {"P_Pa": 3e5, "z_m": 10.0, "diameter_m": d}
    p1 = {"P_Pa": 1e5, "z_m": 0.0, "large_surface": True}
    o = TM.mechanical_energy_balance(1000.0, p1, p2, "shaft_power_W", volumetric_flow_m3_s=q, friction_loss_J_kg=20.0, efficiency=0.75)
    w = 1000.0 * q * (200.0 + u * u / 2 + 9.80665 * 10.0 + 20.0)
    assert o["value"] == pytest.approx(w, rel=1e-12) and o["drive_power_W"] == pytest.approx(w / 0.75)
    assert o["terms_J_kg"]["kinetic (u2²−u1²)/2"] == pytest.approx(u * u / 2)
    back = TM.mechanical_energy_balance(1000.0, p1, p2, "friction_loss_J_kg", volumetric_flow_m3_s=q, shaft_power_W=w)
    assert back["value"] == pytest.approx(20.0, rel=1e-9)
    p2b = {"z_m": 10.0, "diameter_m": d}
    back = TM.mechanical_energy_balance(1000.0, p1, p2b, "P2_Pa", mass_flow_kg_s=1000.0 * q, shaft_power_W=w, friction_loss_J_kg=20.0)
    assert back["value"] == pytest.approx(3e5, rel=1e-9)


def test_mechanical_energy_venturi_and_refusals():
    # venturi, horizontal, frictionless: Q = A2·sqrt(2ΔP/ρ / (1 − (A2/A1)²))
    a1, a2 = math.pi * 0.1 ** 2 / 4, math.pi * 0.05 ** 2 / 4
    o = TM.mechanical_energy_balance(1000.0, {"P_Pa": 2e5, "diameter_m": 0.1}, {"P_Pa": 1.6e5, "diameter_m": 0.05}, "volumetric_flow_m3_s")
    assert o["value"] == pytest.approx(a2 * math.sqrt(2 * 4e4 / 1000.0 / (1 - (a2 / a1) ** 2)), rel=1e-9)
    assert any("fricción F = 0" in a for a in o["assumptions"])
    up = TM.mechanical_energy_balance(1000.0, {"P_Pa": 1e5, "z_m": 0.0, "large_surface": True},
                                      {"P_Pa": 1e5, "z_m": 5.0, "diameter_m": 0.02}, "volumetric_flow_m3_s")
    assert up["ok"] is False                                   # no flow uphill without a pump
    assert "error" in TM.mechanical_energy_balance(1000.0, {"P_Pa": 1e5}, {"P_Pa": 2e5, "diameter_m": 0.1}, "shaft_power_W")
    assert "error" in TM.mechanical_energy_balance(1000.0, {"P_Pa": 1e5, "large_surface": True},
                                                   {"P_Pa": 2e5, "large_surface": True}, "voltage")


def test_algebraic_system_roots_calculator_and_refusals():
    o = TE.solve_algebraic_system(["x^2 + y^2 = 25", "x - y = 1"], {"x": {"lower": 0, "upper": 10}, "y": 1.0},
                                  outputs={"s": "x + y"})
    assert o["ok"] and o["solution"]["unknowns"]["x"] == pytest.approx(4.0) and o["solution"]["outputs"]["s"] == pytest.approx(7.0)
    cubic = TE.solve_algebraic_system(["x^3 - 6*x^2 + 11*x - 6"], {"x": {"guess": [0.5, 2.1, 3.5]}})
    roots = sorted([cubic["solution"]["unknowns"]["x"]] + [s["unknowns"]["x"] for s in cubic["other_solutions"]])
    assert roots == pytest.approx([1.0, 2.0, 3.0]) and cubic["warnings"]
    # Antoine inverted for T (log10 P = A − B/(T + C)) with a parameter and a definition
    ant = TE.solve_algebraic_system(["log10(P) = A - B/(T + C)"], {"T": 80.0}, parameters={"A": 8.07131, "B": 1730.63, "C": 233.426, "P": 760.0})
    assert ant["solution"]["unknowns"]["T"] == pytest.approx(1730.63 / (8.07131 - math.log10(760.0)) - 233.426, rel=1e-10)
    calc = TE.solve_algebraic_system(parameters={"m": 2.0, "v": 3.0}, definitions={"ek": "m*v^2/2"}, outputs={"kJ": "ek/1000"})
    assert calc["ok"] and calc["outputs"]["kJ"] == pytest.approx(0.009)
    none = TE.solve_algebraic_system(["exp(x) = -1"], {"x": 0.0})
    assert none["ok"] is False and "solution" not in none
    assert "error" in TE.solve_algebraic_system(["x = 1", "x = 2"], {"x": 0.0})
    assert "error" in TE.solve_algebraic_system(["__import__('os')"], {"x": 0.0})
    scaled = TE.solve_algebraic_system(["1e9*x = 3e9"], {"x": 0.0})
    assert scaled["solution"]["unknowns"]["x"] == pytest.approx(3.0)
