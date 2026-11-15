"""Validation of the own dynamic flowsheet engine (block dynflow): analytical solutions, textbook worked examples read on
2026-11-14, python-control, and identities (balances, refusals, JSON). Deviation bands are the ones MEASURED, with margin.

Sources (read 2026-11-14):
  [S23] Seborg, Edgar, Mellichamp & Doyle, Process Dynamics and Control, Table 2.3 (exothermic CSTR), as transcribed in
        https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/2_Time_domain_simulation/Nonlinear%20CSTR.html
        (derivatives at cA = 0.5 mol/L, T = 350 K) and https://apmonitor.com/pdc/index.php/Main/StirredReactor
        (steady state Ca_ss = 0.87725294608097, T_ss = 324.475443431599 at Tc = 300 K).
  [S21] Seborg et al. Example 2.1 (blending tank, ρ = 900 kg/m³, w1 = 500 kg/min x1 = 0.4, w2 = 200 kg/min x2 = 0.75,
        w = 650 kg/min, V0 = 2 m³, x0 = 0) as given in
        https://jckantor.github.io/CBE30338/02.03-Blending-Tank-Simulation.html
  [L51] [L52] [L53] Luyben, Process Modeling, Simulation and Control for Chemical Engineers, 2nd ed. (1990), §5.1
        gravity-flow
        tank (Table 5.1/5.2, p. 117-119), §5.2 three isothermal CSTRs (Table 5.3, p. 120-123), §5.3 nonisothermal CSTR with
        two P controllers (Tables 5.5/5.6, p. 125-127). The numbers were transcribed on 2026-11-14 from a copy of the
        book whose authorization could not be verified; the link is deliberately not kept here. Re-check them against a
        library copy before citing them as validation. Luyben's tables are explicit-Euler results (Δt = 1 s, 0.1 min, 0.01 h): the
        tests also replay his Euler program to show that the remaining deviation is his integration error.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

from packages.dwsim_client import tools_control as TC
from packages.dwsim_client import tools_dynamic_flowsheet as TDF
from packages.dynamics import flowsheet as F
from packages.dynamics import units as U


def S(r, name):
    return np.array(r["time_s"], dtype=float), np.array(r["series"][name], dtype=float)


def ok(r):
    json.dumps(r)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    return r


def tank(A=2.0, R=50.0, q=0.01, h0=0.5, extra=None):
    fs = {"units": [{"id": "T", "type": "tank", "area_m2": A, "initial": {"level_m": h0}}],
          "streams": [{"id": "F", "to": "T", "flow": {"type": "fixed", "volumetric_flow_m3_s": q}},
                      {"id": "out", "from": "T", "flow": {"type": "linear_resistance", "resistance_s_m2": R}}]}
    fs.update(extra or {})
    return fs


# ------------------------------------------------------------------------------------------------- analytical: liquid
def test_linear_tank_first_order():
    A, R, q0, q1 = 2.0, 50.0, 0.01, 0.015
    r = ok(F.simulate(tank(A, R, q0, q0 * R), 1500, events=[{"time_s": 50, "target": "F.volumetric_flow_m3_s",
                                                              "value": q1}]))
    t, h = S(r, "T.level_m")
    ex = np.where(t < 50, q0 * R, q1 * R + (q0 - q1) * R * np.exp(-(t - 50) / (A * R)))
    assert np.max(np.abs(h - ex) / ex) < 1e-7                                   # measured 3.7e-9
    # ±2 % band around the value at t_end (here 1.6e-6 of the change from the true steady state); metric grid 0.75 s
    assert r["metrics"]["T.level_m"]["settling_time_s"] == pytest.approx(-A * R * math.log(0.02), abs=1.0)
    assert all(e["hit_exactly"] for e in r["evidence"]["scheduled_events"])
    assert r["evidence"]["worst_balance_closure_rel"] < 1e-12


def test_torricelli_draining_time_and_dry_incident():
    A, Cd, a, h0 = 1.5, 0.6, 1e-3, 2.0
    fs = {"units": [{"id": "T", "type": "tank", "area_m2": A, "initial": {"level_m": h0}}],
          "streams": [{"id": "D", "from": "T", "flow": {"type": "gravity_orifice", "discharge_coefficient": Cd,
                                                        "area_m2": a}}]}
    td = A / (Cd * a) * math.sqrt(2 * h0 / U.G)
    r = ok(F.simulate(fs, 1.2 * td, outputs=["T.level_m"]))
    t, h = S(r, "T.level_m")
    ex = np.where(t < td, (math.sqrt(h0) - Cd * a / A * math.sqrt(U.G / 2) * np.minimum(t, td)) ** 2, 0.0)
    assert np.max(np.abs(h - ex)) < 1e-9                                         # measured 2e-12 m
    dry = [i for i in r["incidents"] if i["event"] == "dry"]
    assert len(dry) == 1 and dry[0]["time_s"] == pytest.approx(td, rel=2e-6)  # measured 1.0e-6 (empty threshold)


def test_two_tanks_non_interacting_and_interacting():
    q0, q1, A1, A2, R1, R2 = 0.01, 0.015, 1.0, 2.0, 100.0, 40.0
    fs = {"units": [{"id": "T1", "type": "tank", "area_m2": A1, "initial": {"level_m": q0 * R1}},
                    {"id": "T2", "type": "tank", "area_m2": A2, "initial": {"level_m": q0 * R2}}],
          "streams": [{"id": "F", "to": "T1", "flow": {"type": "fixed", "volumetric_flow_m3_s": q0}},
                      {"id": "S1", "from": "T1", "to": "T2", "flow": {"type": "linear_resistance", "resistance_s_m2": R1}},
                      {"id": "S2", "from": "T2", "flow": {"type": "linear_resistance", "resistance_s_m2": R2}}]}
    ev = [{"time_s": 0, "target": "F.volumetric_flow_m3_s", "value": q1}]
    r = ok(F.simulate(fs, 1200, events=ev))
    t, h2 = S(r, "T2.level_m")
    t1, t2 = A1 * R1, A2 * R2
    ex = q0 * R2 + (q1 - q0) * R2 * (1 - (t1 * np.exp(-t / t1) - t2 * np.exp(-t / t2)) / (t1 - t2))
    assert np.max(np.abs(h2 - ex) / ex) < 1e-7                                  # measured 3.5e-9
    fs["streams"][1]["flow"]["interacting"] = True
    fs["units"][0]["initial"]["level_m"] = q0 * (R1 + R2)
    r = ok(F.simulate(fs, 1200, events=ev))
    M = np.array([[-1 / (A1 * R1), 1 / (A1 * R1)], [1 / (A2 * R1), -1 / (A2 * R1) - 1 / (A2 * R2)]])
    xs = np.linalg.solve(M, -np.array([q1 / A1, 0.0]))
    x0 = np.array([q0 * (R1 + R2), q0 * R2])
    t, h1 = S(r, "T1.level_m")
    ex = np.array([xs + expm(M * tt) @ (x0 - xs) for tt in t])
    assert np.max(np.abs(h1 - ex[:, 0]) / ex[:, 0]) < 1e-7                      # measured 2.4e-9
    assert np.max(np.abs(S(r, "T2.level_m")[1] - ex[:, 1]) / ex[:, 1]) < 1e-7
    ss = F.steady_state_analysis(fs)
    assert ss["steady_state"]["T1.level_m"] == pytest.approx(q0 * (R1 + R2), rel=1e-10)   # initial flow is q0
    assert sorted(ss["time_constants_s"]) == pytest.approx(sorted(-1 / np.linalg.eigvals(M)), rel=1e-6)


def test_mixing_tank_step_in_feed_concentration():
    V, q, c0, cin = 3.0, 0.02, 100.0, 400.0
    fs = {"components": ["A"],
          "units": [{"id": "M", "type": "mixing_tank", "constant_volume": True, "volume_m3": V,
                     "initial": {"concentration_mol_m3": {"A": c0}}}],
          "streams": [{"id": "F", "to": "M", "flow": {"type": "fixed", "volumetric_flow_m3_s": q},
                       "concentration_mol_m3": {"A": c0}}, {"id": "P", "from": "M", "flow": {"type": "overflow"}}]}
    r = ok(F.simulate(fs, 900, events=[{"time_s": 100, "target": "F.concentration_mol_m3.A", "value": cin}]))
    t, c = S(r, "M.concentration_mol_m3.A")
    ex = np.where(t < 100, c0, cin + (c0 - cin) * np.exp(-(t - 100) / (V / q)))
    assert np.max(np.abs(c - ex) / ex) < 1e-7                                   # measured 4.2e-9


def test_seborg_example_2_1_blending_tank_with_rising_volume():
    """[S21] Two feeds, fixed outflow, volume rises 50/900 m³/min; x(t) = x̄[1 − (V/V0)^(−w_tot/(w_tot − w))]."""
    rho, w1, w2, w, x1, x2, V0 = 900.0, 500.0, 200.0, 650.0, 0.4, 0.75, 2.0
    fs = {"components": ["S"],
          "units": [{"id": "B", "type": "mixing_tank", "area_m2": 1.0,
                     "initial": {"volume_m3": V0, "concentration_mol_m3": {"S": 0.0}}}],
          "streams": [{"id": "W1", "to": "B", "flow": {"type": "fixed", "volumetric_flow_m3_s": w1 / rho / 60},
                       "concentration_mol_m3": {"S": x1 * rho}},
                      {"id": "W2", "to": "B", "flow": {"type": "fixed", "volumetric_flow_m3_s": w2 / rho / 60},
                       "concentration_mol_m3": {"S": x2 * rho}},
                      {"id": "W", "from": "B", "flow": {"type": "fixed", "volumetric_flow_m3_s": w / rho / 60}}]}
    r = ok(F.simulate(fs, 600.0))                     # 10 min as in the source's plot
    t, c = S(r, "B.concentration_mol_m3.S")
    xbar = (w1 * x1 + w2 * x2) / (w1 + w2)
    assert xbar == pytest.approx(0.5)                 # the source's steady state
    g = (w1 + w2 - w) / rho / 60
    ex = xbar * (1 - (1 + g * t / V0) ** (-(w1 + w2) / (w1 + w2 - w)))
    assert np.max(np.abs(c / rho - ex)) < 1e-7                                  # measured 1.7e-9
    assert S(r, "B.volume_m3")[1][-1] == pytest.approx(V0 + 50 / 900 * 10, rel=1e-10)


def test_heated_tank_step_and_ramp_in_duty():
    V, q, rho, cp, Ti, Q0, Q1 = 3.0, 0.02, 1000.0, 4184.0, 300.0, 2.0e5, 3.0e5
    Ts = Ti + Q0 / (rho * cp * q)
    fs = {"liquid": {"density_kg_m3": rho, "heat_capacity_J_kgK": cp},
          "units": [{"id": "H", "type": "heated_tank", "constant_volume": True, "volume_m3": V, "heat_duty_W": Q0,
                     "initial": {"temperature_K": Ts}}],
          "streams": [{"id": "F", "to": "H", "flow": {"type": "fixed", "volumetric_flow_m3_s": q}, "temperature_K": Ti},
                      {"id": "P", "from": "H", "flow": {"type": "overflow"}}]}
    r = ok(F.simulate(fs, 1200, events=[{"time_s": 60, "target": "H.heat_duty_W", "value": Q1}]))
    t, T = S(r, "H.temperature_K")
    Ts1, tau = Ti + Q1 / (rho * cp * q), V / q
    assert np.max(np.abs(T - np.where(t < 60, Ts, Ts1 + (Ts - Ts1) * np.exp(-(t - 60) / tau)))) < 1e-5  # meas. 9e-7 K
    assert r["evidence"]["balances"]["liquid_energy"]["closure_rel"] < 1e-12
    r = ok(F.simulate(fs, 1200, events=[{"time_s": 60, "target": "H.heat_duty_W", "value": Q1, "ramp_s": 300}]))
    t, T = S(r, "H.temperature_K")
    kk = (Q1 - Q0) / 300 / (rho * cp * q)

    def ramp(tt):
        if tt < 60:
            return Ts
        s = tt - 60
        if s <= 300:
            return Ts + kk * (s - tau * (1 - math.exp(-s / tau)))
        te = Ts + kk * (300 - tau * (1 - math.exp(-300 / tau)))
        return Ts1 + (te - Ts1) * math.exp(-(s - 300) / tau)

    assert np.max(np.abs(T - np.array([ramp(x) for x in t]))) < 1e-5            # measured 9.9e-7 K
    assert [e["hit_exactly"] for e in r["evidence"]["scheduled_events"]] == [True, True]


def test_isothermal_cstr_startup():
    V, q, k, cA0 = 3.0, 0.02, 0.004, 500.0
    fs = {"components": ["A", "B"],
          "units": [{"id": "R", "type": "cstr", "constant_volume": True, "volume_m3": V,
                     "reactions": [{"stoichiometry": {"A": -1, "B": 1}, "orders": {"A": 1}, "rate_constant": k}],
                     "initial": {"concentration_mol_m3": {"A": 0, "B": 0}}}],
          "streams": [{"id": "F", "to": "R", "flow": {"type": "fixed", "volumetric_flow_m3_s": q},
                       "concentration_mol_m3": {"A": cA0, "B": 0}}, {"id": "P", "from": "R", "flow": {"type": "overflow"}}]}
    tau = V / q
    r = ok(F.simulate(fs, 8 * tau / (1 + k * tau)))
    t, c = S(r, "R.concentration_mol_m3.A")
    css = cA0 / (1 + k * tau)
    assert np.max(np.abs(c - css * (1 - np.exp(-(1 + k * tau) * t / tau)))) / css < 1e-7    # measured 3.3e-9
    b = r["evidence"]["balances"]
    assert b["component_mol.A"]["closure_rel"] < 1e-9 and b["component_mol.B"]["generated"] > 0
    assert b["component_mol.A"]["generated"] == pytest.approx(-b["component_mol.B"]["generated"], rel=1e-12)


def test_valve_tank_and_pump_steady_states_and_pump_trip():
    Kv, op, qin = 20.0, 0.5, 0.002
    fs = {"liquid": {"density_kg_m3": 1000.0},
          "units": [{"id": "T", "type": "tank", "area_m2": 1.0, "pressure_Pa": 101325.0, "initial": {"level_m": 1.0}}],
          "streams": [{"id": "F", "to": "T", "flow": {"type": "fixed", "volumetric_flow_m3_s": qin}},
                      {"id": "V", "from": "T", "flow": {"type": "valve", "Kv_m3_h": Kv, "opening": op,
                                                        "characteristic": "linear", "downstream_pressure_Pa": 101325.0}}]}
    ss = F.steady_state_analysis(fs)
    hs = (qin * 3600 / (Kv * op)) ** 2 * 1e5 / (1000 * U.G)        # IEC 60534 with SG = 1
    assert ss["steady_state"]["T.level_m"] == pytest.approx(hs, rel=1e-10) and ss["stable"]
    assert ss["time_constants_s"][0] == pytest.approx(2 * 1.0 * hs / qin, rel=1e-6)   # linearised √h outflow
    # pump with a LINEAR curve (interpolation exact): 30 − 1000 q = (10 − h) + 2e5 q²
    qc = [0.0, 0.03]
    fs = {"liquid": {"density_kg_m3": 1000.0},
          "units": [{"id": "T", "type": "tank", "area_m2": 2.0, "pressure_Pa": 101325.0, "initial": {"level_m": 3.0}}],
          "streams": [{"id": "F", "to": "T", "flow": {"type": "fixed", "volumetric_flow_m3_s": 0.005}},
                      {"id": "P", "from": "T", "flow": {"type": "pump", "curve_flow_m3_s": qc, "curve_head_m": [30.0, 0.0],
                                                        "static_head_m": 10.0, "loss_coefficient_s2_m5": 2e5,
                                                        "downstream_pressure_Pa": 101325.0}}]}
    r = ok(F.simulate(fs, 100, outputs=["P.volumetric_flow_m3_s", "T.level_m"],
                      events=[{"time_s": 50, "target": "P.running", "value": False}]))
    q0 = (-1000 + math.sqrt(1000 ** 2 + 4 * 2e5 * 23)) / (2 * 2e5)
    assert r["series"]["P.volumetric_flow_m3_s"][0] == pytest.approx(q0, rel=1e-9)
    t, h = S(r, "T.level_m")
    assert r["series"]["P.volumetric_flow_m3_s"][-1] == 0.0
    assert h[-1] - np.interp(50, t, h) == pytest.approx(0.005 * 50 / 2.0, rel=1e-9)     # tripped: only inflow


def test_heat_exchanger_cells_converge_to_epsilon_ntu():
    """Countercurrent cells in series → ε-NTU of the ideal countercurrent exchanger (error O(1/n), Richardson)."""
    def hx(n):
        side = {"volume_m3": 0.05, "density_kg_m3": 1000.0, "heat_capacity_J_kgK": 4180.0}
        return {"units": [{"id": "X", "type": "heat_exchanger", "flow": "countercurrent", "UA_W_K": 5000.0, "cells": n,
                           "hot": side, "cold": side,
                           "initial": {"hot": {"temperature_K": 300.0}, "cold": {"temperature_K": 300.0}}}],
                "streams": [{"id": "H", "to": "X.hot", "flow": {"type": "fixed", "volumetric_flow_m3_s": 0.001},
                             "temperature_K": 360.0},
                            {"id": "C", "to": "X.cold", "flow": {"type": "fixed", "volumetric_flow_m3_s": 0.002},
                             "temperature_K": 290.0}, {"id": "Ho", "from": "X.hot"}, {"id": "Co", "from": "X.cold"}]}
    Ch, Cc = 4180.0, 8360.0
    ntu, cr = 5000 / Ch, Ch / Cc
    eps = (1 - math.exp(-ntu * (1 - cr))) / (1 - cr * math.exp(-ntu * (1 - cr)))
    e = {}
    for n in (10, 40):
        r = ok(F.steady_state_analysis(hx(n)))
        tho, tco = r["steady_state"]["X.hot_outlet_temperature_K"], r["steady_state"]["X.cold_outlet_temperature_K"]
        assert Ch * (360 - tho) == pytest.approx(Cc * (tco - 290), rel=1e-10)
        e[n] = (360 - tho) / 70
    assert e[10] < e[40] < eps
    assert (4 * e[40] - e[10]) / 3 == pytest.approx(eps, rel=2e-3)              # measured 4.9e-4 (n = 40 alone: 1.1 %)
    r = ok(F.simulate(hx(5), 300, outputs=["X.hot_outlet_temperature_K"]))
    assert r["evidence"]["balances"]["liquid_energy"]["closure_rel"] < 1e-12


# ------------------------------------------------------------------------------------------------------ analytical: gas
def _vessel(thermal, p0, flow, model=None):
    return {"units": [{"id": "V", "type": "gas_vessel", "volume_m3": 5.0, "thermal": thermal,
                       "gas": model or {"model": "ideal", "molar_mass_kg_mol": 0.02897, "heat_capacity_ratio": 1.4},
                       "initial": {"pressure_Pa": p0, "temperature_K": 300.0}}], "streams": [flow]}


def test_choked_blowdown_isothermal_and_adiabatic_closed_forms():
    k, M, T0, P0, Pb, CdA = 1.4, 0.02897, 300.0, 2.0e6, 1.0e5, 0.8e-4
    gam = (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    c0 = math.sqrt(k * U.R_GAS * T0 / M)
    tau = 5.0 / (CdA * c0 * gam)
    pch = Pb / U.critical_pressure_ratio(k)
    bd = {"id": "BD", "from": "V", "flow": {"type": "gas_orifice", "discharge_coefficient": 0.8, "area_m2": 1e-4,
                                            "downstream_pressure_Pa": Pb}}
    tch = tau * math.log(P0 / pch)
    r = ok(F.simulate(_vessel("isothermal", P0, bd), tch, limits=[{"variable": "V.pressure_Pa", "min": 5e5}]))
    t, P = S(r, "V.pressure_Pa")
    assert np.max(np.abs(P - P0 * np.exp(-t / tau)) / P) < 1e-7                  # measured 3.5e-9
    assert r["limit_violations"][0]["first_time_s"] == pytest.approx(tau * math.log(P0 / 5e5), rel=1e-8)
    tend = tau * 2 / (k - 1) * ((P0 / pch) ** ((k - 1) / (2 * k)) - 1)
    r = ok(F.simulate(_vessel("adiabatic", P0, bd), tend))
    t, P = S(r, "V.pressure_Pa")
    ex = P0 * (1 + (k - 1) / 2 * t / tau) ** (-2 * k / (k - 1))
    assert np.max(np.abs(P - ex) / ex) < 1e-7                                    # measured 3.5e-9
    assert np.max(np.abs(S(r, "V.temperature_K")[1] - T0 * (ex / P0) ** ((k - 1) / k))) < 1e-5
    assert r["evidence"]["balances"]["gas_energy_J"]["closure_rel"] < 1e-12
    r = ok(F.simulate(_vessel("isothermal", P0, bd), 6 * tch))                    # subsonic tail to back-pressure
    assert r["series"]["V.pressure_Pa"][-1] == pytest.approx(Pb, rel=1e-8) and r["at_steady_state"]


def test_choked_pressurization_is_linear_in_time():
    k, M, T0, Ps = 1.4, 0.02897, 300.0, 3.0e6
    gam = (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    feed = {"id": "IN", "to": "V", "flow": {"type": "gas_orifice", "discharge_coefficient": 0.8, "area_m2": 1e-4,
                                            "upstream_pressure_Pa": Ps, "upstream_temperature_K": T0}}
    md = 0.8e-4 * Ps * math.sqrt(k * M / (U.R_GAS * T0)) * gam
    rate = U.R_GAS * T0 / (M * 5.0) * md
    tch = (U.critical_pressure_ratio(k) * Ps - 1e5) / rate
    r = ok(F.simulate(_vessel("isothermal", 1e5, feed), tch))
    t, P = S(r, "V.pressure_Pa")
    assert np.max(np.abs(P - (1e5 + rate * t)) / P) < 1e-10


def test_coolprop_nitrogen_blowdown_close_to_ideal_at_low_pressure():
    """CONSISTENCY: N2 at ≤ 5 bar is nearly ideal; real-gas run must stay within 0.5 % of the ideal one."""
    pytest.importorskip("CoolProp")
    bd = {"id": "BD", "from": "V", "flow": {"type": "gas_orifice", "discharge_coefficient": 0.8, "area_m2": 1e-4,
                                            "downstream_pressure_Pa": 1e5}}
    ideal = ok(F.simulate(_vessel("adiabatic", 5e5, bd, {"model": "ideal", "molar_mass_kg_mol": 0.0280134,
                                                           "heat_capacity_ratio": 1.4}), 60, n_points=21))
    real = ok(F.simulate(_vessel("adiabatic", 5e5, bd, {"model": "coolprop", "fluid": "Nitrogen"}), 60, n_points=21,
                         refine=False))
    pi, pr = S(ideal, "V.pressure_Pa")[1], S(real, "V.pressure_Pa")[1]
    assert np.max(np.abs(pr - pi) / pi) < 5e-3                                   # measured 1.5e-3


# --------------------------------------------------------------------------------------------------------- control
def _loop(tau_i, ctl=None, extra=None):
    c = {"id": "LC", "measurement": "T.level_m", "setpoint": 0.5, "Kc": 0.08, "tau_I_s": tau_i,
         "manipulated": "F.volumetric_flow_m3_s"}
    c.update(ctl or {})
    return tank(extra={"controllers": [c], **(extra or {})})


def test_pi_loop_pole_zero_cancellation_closed_form():
    K, tau, Kc = 50.0, 100.0, 0.08
    r = ok(F.simulate(_loop(tau), 400, events=[{"time_s": 20, "target": "LC.setpoint", "value": 0.6}],
                      outputs=["T.level_m", "LC.output"]))
    t, h = S(r, "T.level_m")
    ex = np.where(t < 20, 0.5, 0.6 - 0.1 * np.exp(-(t - 20) / (tau / (Kc * K))))
    assert np.max(np.abs(h - ex)) / 0.1 < 1e-6                                   # measured 1.5e-8


def test_pi_and_pid_vs_python_control():
    control = pytest.importorskip("control")
    K, tau, Kc, ti, td, alpha = 50.0, 100.0, 0.08, 30.0, 5.0, 0.1
    G = control.tf([K], [tau, 1])
    Cpi = control.tf([Kc * ti, Kc], [ti, 0])
    ev = [{"time_s": 0, "target": "LC.setpoint", "value": 0.6}]
    r = ok(F.simulate(_loop(ti), 600, events=ev, outputs=["T.level_m"], n_points=401))
    t, h = S(r, "T.level_m")
    _, yy = control.step_response(control.feedback(Cpi * G), T=t)
    assert np.max(np.abs(h - (0.5 + 0.1 * yy))) / 0.1 < 1e-6                    # measured 1.4e-8
    assert r["metrics"]["T.level_m"]["overshoot_fraction"] == pytest.approx(yy.max() - 1, rel=1e-4)
    r = ok(F.simulate(_loop(ti, {"tau_D_s": td, "derivative_filter_alpha": alpha}), 600, events=ev,
                      outputs=["T.level_m"], n_points=401))
    t, h = S(r, "T.level_m")
    Cd = control.tf([Kc * td, 0], [alpha * td, 1])                # derivative on the filtered measurement
    _, yy = control.step_response(control.minreal(Cpi * G / (1 + (Cpi + Cd) * G), verbose=False), T=t)
    assert np.max(np.abs(h - (0.5 + 0.1 * yy))) / 0.1 < 1e-6                    # measured 1.1e-8


def test_saturation_with_anti_windup_limits_overshoot():
    ev = [{"time_s": 0, "target": "LC.setpoint", "value": 0.6}]
    sat = ok(F.simulate(_loop(30.0, {"output_min": 0.0, "output_max": 0.0125}), 900, events=ev,
                        outputs=["T.level_m", "LC.output"]))
    assert max(sat["series"]["LC.output"]) <= 0.0125 + 1e-15
    t, h = S(sat, "T.level_m")
    # while saturated the tank is open loop with q = 0.0125: h = 0.625 − 0.125 e^(−t/100) until it nears the setpoint
    early = t < 50
    assert np.max(np.abs(h[early] - (0.625 - 0.125 * np.exp(-t[early] / 100.0)))) < 1e-7
    assert sat["metrics"]["T.level_m"]["overshoot_fraction"] < 0.05 and sat["at_steady_state"]   # measured 0.031


def test_dead_time_sensor_open_and_closed_loop():
    blk = {"blocks": [{"id": "LT", "type": "lag", "input": "T.level_m", "dead_time_s": 15.0}]}
    fs = tank(extra=blk)
    r = ok(F.simulate(fs, 600, events=[{"time_s": 10, "target": "F.volumetric_flow_m3_s", "value": 0.012}],
                      outputs=["LT.output"]))
    t, y = S(r, "LT.output")
    assert np.max(np.abs(y - np.where(t < 25, 0.5, 0.5 + 0.1 * (1 - np.exp(-(t - 25) / 100.0))))) / 0.1 < 1e-6
    from packages.dynamics import loop as LP
    from packages.dynamics.process import Process
    fs = _loop(100.0, {"measurement": "LT.output", "Kc": 0.04}, blk)
    r = ok(F.simulate(fs, 1500, events=[{"time_s": 0, "target": "LC.setpoint", "value": 0.6}], outputs=["LT.output"],
                      n_points=601))
    t, y = S(r, "LT.output")
    sim = LP.simulate(Process(gain=50.0, lags=[100.0], delay=15.0), LP.Controller(kind="PI", Kc=0.04, tau_I=100.0),
                      0.1, 1.0, 0.0, t_end=1500.0)
    yi = np.interp(t, np.array(sim["t"]), np.array(sim["y"]))
    assert np.max(np.abs(y - (0.5 + 0.1 * yi))) / 0.1 < 5e-4                    # measured 2.0e-4 (loop.py's own grid)


def test_steady_start_keeps_delayed_signals_consistent():
    """Regression: after start_from_steady_state the history of a dead-time block is the steady state, not the
    user's initial state (previously the delayed output showed the old level for θ seconds)."""
    fs = tank(h0=0.2, extra={"blocks": [{"id": "LT", "input": "T.level_m", "dead_time_s": 15.0}]})
    r = ok(F.simulate(fs, 60, outputs=["LT.output"], start_from_steady_state=True, n_points=13))
    assert np.allclose(r["series"]["LT.output"], 0.5, rtol=1e-9)


# ------------------------------------------------------------------------------------------- textbook: Seborg CSTR
def _seborg(cA, T, Tc=300.0):
    return {"components": ["A"], "liquid": {"density_kg_m3": 1000.0, "heat_capacity_J_kgK": 239.0},
            "units": [{"id": "R", "type": "cstr", "constant_volume": True, "volume_m3": 0.1,
                       "jacket": {"UA_W_K": 5e4 / 60, "temperature_K": Tc},
                       "reactions": [{"stoichiometry": {"A": -1}, "orders": {"A": 1}, "pre_exponential": 7.2e10 / 60,
                                      "activation_energy_J_mol": 8750 * U.R_GAS, "heat_of_reaction_J_mol": -5e4}],
                       "initial": {"temperature_K": T, "concentration_mol_m3": {"A": cA * 1000}}}],
            "streams": [{"id": "F", "to": "R", "flow": {"type": "fixed", "volumetric_flow_m3_s": 0.1 / 60},
                         "temperature_K": 350.0, "concentration_mol_m3": {"A": 1000.0}},
                        {"id": "P", "from": "R", "flow": {"type": "overflow"}}]}


def test_seborg_table_2_3_cstr_derivatives_steady_state_and_multiplicity():
    m = F.Model(_seborg(0.5, 350.0))
    m.set_schedule(None, 1.0)
    dy = m.rhs(0.0, m.y0)
    assert dy[0] / 0.1 / 1000 * 60 == pytest.approx(3.40208612952253e-05, rel=1e-9)   # [S23] readthedocs, mol/(L min)
    assert dy[1] / 0.1 * 60 == pytest.approx(-0.007117334999003795, rel=1e-9)       # K/min; measured 1.3e-12
    low = ok(F.steady_state_analysis(_seborg(0.877, 324.5)))
    assert low["steady_state"]["R.concentration_mol_m3.A"] / 1000 == pytest.approx(0.87725294608097, rel=1e-9)
    assert low["steady_state"]["R.temperature_K"] == pytest.approx(324.475443431599, rel=1e-12)   # [S23] APMonitor
    assert low["stable"]
    mid = ok(F.steady_state_analysis(_seborg(0.5, 350.0)))
    high = ok(F.steady_state_analysis(_seborg(0.05, 400.0)))
    temps = sorted(r["steady_state"]["R.temperature_K"] for r in (low, mid, high))
    assert temps[0] < 330 < temps[1] < 360 < temps[2]                           # three distinct steady states
    assert mid["stable"] is False and mid["eigenvalues_per_s"][0]["imag"] == 0.0        # saddle (2.83, −0.45 1/min)
    assert high["stable"] is False and high["oscillation_periods_s"]                     # unstable focus → limit cycle
    for r in (low, mid, high):                                                  # each is a true zero of the RHS
        assert r["evidence"]["residual_max_per_s"] < 1e-10


def test_seborg_cstr_coolant_step_runaway_vs_quench():
    for tc, hot in ((290.0, False), (305.0, True)):
        r = ok(F.simulate(_seborg(0.5, 350.0), 600.0, start_from_steady_state=True,
                          events=[{"time_s": 0, "target": "R.jacket_temperature_K", "value": tc}],
                          outputs=["R.temperature_K"]))
        m = r["metrics"]["R.temperature_K"]
        assert (m["max"] > 430.0) if hot else (m["final"] < 315.0)                # measured 441.4 K / 312.7 K


# ------------------------------------------------------------------------------------------------- textbook: Luyben
def test_luyben_5_2_three_isothermal_cstrs_table_5_3():
    tab = {0.1: (0.450, 0.200, 0.100), 0.5: (0.605, 0.220, 0.101), 1.0: (0.726, 0.266, 0.109),
           1.5: (0.797, 0.313, 0.123), 2.0: (0.839, 0.352, 0.140), 2.5: (0.864, 0.382, 0.158), 2.9: (0.876, 0.400, 0.171)}
    units = [{"id": f"R{i + 1}", "type": "cstr", "constant_volume": True, "volume_m3": 1.0,
              "reactions": [{"stoichiometry": {"A": -1}, "orders": {"A": 1}, "rate_constant": 0.5 / 60}],
              "initial": {"concentration_mol_m3": {"A": c}}} for i, c in enumerate((400.0, 200.0, 100.0))]
    fs = {"components": ["A"], "units": units,
          "streams": [{"id": "F", "to": "R1", "flow": {"type": "fixed", "volumetric_flow_m3_s": 1 / 120},
                       "concentration_mol_m3": {"A": 1800.0}},
                      {"id": "S1", "from": "R1", "to": "R2", "flow": {"type": "overflow"}},
                      {"id": "S2", "from": "R2", "to": "R3", "flow": {"type": "overflow"}},
                      {"id": "S3", "from": "R3", "flow": {"type": "overflow"}}]}
    r = ok(F.simulate(fs, 2.9 * 60, n_points=30))
    t = np.array(r["time_s"])
    worst = 0.0
    for tm, row in tab.items():
        i = int(np.argmin(np.abs(t - tm * 60)))
        mine = [r["series"][f"R{j + 1}.concentration_mol_m3.A"][i] / 1000 for j in range(3)]
        worst = max(worst, max(abs(a - b) / b for a, b in zip(mine, row)))
    assert worst < 0.02                                                          # measured 1.37 % (his Euler Δt)
    c = [0.4, 0.2, 0.1]                                                          # Luyben's own Euler, Δt = 0.1 min
    for _ in range(29):
        d = [(1.8 - c[0]) / 2 - 0.5 * c[0], (c[0] - c[1]) / 2 - 0.5 * c[1], (c[1] - c[2]) / 2 - 0.5 * c[2]]
        c = [c[k] + 0.1 * d[k] for k in range(3)]
    assert np.allclose(c, tab[2.9], atol=6e-4)                                   # the table IS Euler (rounding)
    ss = F.steady_state_analysis(fs)["steady_state"]
    assert [ss[f"R{j}.concentration_mol_m3.A"] for j in (1, 2, 3)] == pytest.approx([900.0, 450.0, 225.0], rel=1e-10)


def _gravity(v0, h0):
    ft = 0.3048
    return {"units": [{"id": "T", "type": "tank", "area_m2": 113 * ft ** 2, "max_level_m": 7 * ft,
                       "initial": {"level_m": h0 * ft}}],
            "streams": [{"id": "F", "to": "T", "flow": {"type": "fixed", "volumetric_flow_m3_s": 35.1 * ft ** 3}},
                        {"id": "L", "from": "T", "flow": {"type": "inertial_pipe", "length_m": 3000 * ft,
                                                          "diameter_m": 3 * ft, "darcy_friction_factor": 0.0123,
                                                          "initial_velocity_m_s": v0 * ft}}]}


def test_luyben_5_1_gravity_flow_tank():
    ft = 0.3048
    case1 = [(0, 3.40, 2.05), (20, 3.55, 3.98), (40, 3.99, 5.52), (60, 4.54, 6.44), (80, 5.02, 6.69), (100, 5.33, 6.45),
             (120, 5.46, 5.92), (140, 5.44, 5.32), (160, 5.33, 4.81), (180, 5.19, 4.46), (200, 5.05, 4.29)]
    r = ok(F.simulate(_gravity(3.40, 2.05), 200.0, n_points=11, outputs=["T.level_m", "L.velocity_m_s"]))
    dv = max(abs(r["series"]["L.velocity_m_s"][i] / ft - v) for i, (_, v, _) in enumerate(case1))
    dh = max(abs(r["series"]["T.level_m"][i] / ft - h) for i, (_, _, h) in enumerate(case1))
    assert dv < 0.06 and dh < 0.15                                               # measured 0.041 ft/s, 0.109 ft
    v, h = 3.40, 2.05                                                            # Luyben's Euler, Δt = 1 s, eqs 5.3-5.4
    for _ in range(80):
        v, h = v + (0.0107 * h - 0.00205 * v * v), h + (0.311 - 0.0624 * v)
    assert abs(v - 5.02) < 0.01 and abs(h - 6.69) < 0.01                        # the table is that Euler run
    ss = F.steady_state_analysis(_gravity(3.4, 2.05))
    assert ss["steady_state"]["T.level_m"] / ft == pytest.approx(4.72, abs=0.01)          # Table 5.1
    assert ss["steady_state"]["L.velocity_m_s"] / ft == pytest.approx(4.97, abs=0.01)
    assert ss["stable"] and ss["oscillation_periods_s"]                          # underdamped, as in his plots
    r2 = ok(F.simulate(_gravity(2.50, 1.20), 200.0, outputs=["T.level_m"]))      # case 2: Luyben's level passes 7 ft
    assert [i["event"] for i in r2["incidents"]] == ["overflow", "back_in_range"]
    assert max(r2["series"]["T.level_m"]) <= 7 * ft * (1 + 1e-9)
    assert r2["evidence"]["balances"]["liquid_volume_m3"]["spilled"] > 0


def test_luyben_5_3_nonisothermal_cstr_with_two_p_controllers_table_5_6():
    """English units kept as given (ft³, h, °R, lb·mol): the engine is unit-consistent; E/R = 30000/1.99 °R."""
    tab = [(0.2, 0.252, 600.370, 594.822, 51.379), (0.41, 0.256, 601.138, 595.222, 54.453),
           (0.61, 0.257, 601.810, 595.572, 57.139), (0.81, 0.258, 602.267, 595.810, 58.967),
           (1.01, 0.257, 602.499, 595.932, 59.896), (1.21, 0.256, 602.565, 595.968, 60.161),
           (1.41, 0.256, 602.542, 595.957, 60.070), (1.81, 0.256, 602.442, 595.906, 59.668),
           (2.41, 0.256, 602.394, 595.881, 59.477), (4.01, 0.256, 602.407, 595.887, 59.529)]
    fs = {"components": ["A"],
          "units": [{"id": "R", "type": "cstr", "area_m2": 1.0, "density_kg_m3": 50.0, "heat_capacity_J_kgK": 0.75,
                     "reactions": [{"stoichiometry": {"A": -1}, "orders": {"A": 1}, "pre_exponential": 7.08e10,
                                    "activation_energy_J_mol": 30000 / 1.99 * U.R_GAS, "heat_of_reaction_J_mol": -30000.0}],
                     "initial": {"volume_m3": 48.0, "temperature_K": 600.0, "concentration_mol_m3": {"A": 0.245}}},
                    {"id": "J", "type": "heated_tank", "constant_volume": True, "volume_m3": 3.85, "density_kg_m3": 62.3,
                     "heat_capacity_J_kgK": 1.0, "initial": {"temperature_K": 594.59, "concentration_mol_m3": {"A": 0.0}}}],
          "heat_links": [{"id": "HX", "between": ["R", "J"], "UA_W_K": 150.0 * 250.0}],
          "streams": [{"id": "F0", "to": "R", "flow": {"type": "fixed", "volumetric_flow_m3_s": 40.0},
                       "temperature_K": 530.0,
                       "concentration_mol_m3": {"A": 0.50}},
                      {"id": "F", "from": "R", "flow": {"type": "fixed", "volumetric_flow_m3_s": 40.0}},
                      {"id": "FJ", "to": "J", "flow": {"type": "fixed", "volumetric_flow_m3_s": 49.9},
                       "temperature_K": 530.0,
                       "concentration_mol_m3": {"A": 0.0}},
                      {"id": "FJout", "from": "J", "flow": {"type": "overflow"}}],
          "controllers": [{"id": "LC", "measurement": "R.volume_m3", "setpoint": 48.0, "Kc": -10.0, "bias": 40.0,
                           "manipulated": "F.volumetric_flow_m3_s"},
                          {"id": "TC", "measurement": "R.temperature_K", "setpoint": 600.0, "Kc": -4.0, "bias": 49.9,
                           "manipulated": "FJ.volumetric_flow_m3_s"}]}
    r = ok(F.simulate(fs, 4.01, events=[{"time_s": 0.0, "target": "F0.concentration_mol_m3.A", "value": 0.55}],
                      outputs=["R.concentration_mol_m3.A", "R.temperature_K", "J.temperature_K", "TC.output"],
                      n_points=402))
    t = np.array(r["time_s"])
    dev = np.zeros(4)
    for tm, *row in tab:
        i = int(np.argmin(np.abs(t - tm)))
        assert abs(t[i] - tm) < 1e-9
        mine = [r["series"][k][i] for k in ("R.concentration_mol_m3.A", "R.temperature_K", "J.temperature_K", "TC.output")]
        dev = np.maximum(dev, np.abs(np.array(mine) - np.array(row)))
    assert dev[0] < 1e-3 and dev[1] < 0.03 and dev[2] < 0.03 and dev[3] < 0.1   # measured 5.5e-4, 0.016, 0.012, 0.063
    assert r["evidence"]["balances"]["liquid_energy"]["closure_rel"] < 1e-12


# --------------------------------------------------------------------------------------------------------- step test
def test_step_test_feeds_fopdt_identify_and_freezes_controller_in_manual():
    fs = tank(h0=0.3, extra={"blocks": [{"id": "LT", "input": "T.level_m", "dead_time_s": 15.0}],
                             "controllers": [{"id": "LC", "measurement": "LT.output", "setpoint": 0.6, "Kc": 0.04,
                                              "tau_I_s": 100.0, "manipulated": "F.volumetric_flow_m3_s"}]})
    r = ok(TDF.dynamic_step_test(fs, "F.volumetric_flow_m3_s", 0.002, "LT.output", 1000.0, 20.0))
    assert r["controllers_in_manual"] == ["LC"]
    assert r["input_initial"] == pytest.approx(0.012, rel=1e-9)                 # closed-loop steady output (h = 0.6)
    assert r["baseline_output"] == pytest.approx(0.6, rel=1e-9) and r["pre_step_drift"] < 1e-9
    fit = TC.fopdt_identify(**r["fopdt_identify_arguments"])["fopdt"]
    assert fit["K"] == pytest.approx(50.0, rel=1e-3)                            # K = R
    assert fit["tau"] == pytest.approx(100.0, rel=1e-3)                         # τ = A·R
    assert fit["theta"] == pytest.approx(15.0, abs=0.05)                        # sensor dead time
    json.dumps(r)


# ------------------------------------------------------------------------------------------ identities and refusals
def test_fill_from_empty_and_overflow_spill_balance():
    fs = {"units": [{"id": "T", "type": "tank", "area_m2": 1.0, "max_level_m": 2.0, "initial": {"level_m": 0.0}}],
          "streams": [{"id": "F", "to": "T", "flow": {"type": "fixed", "volumetric_flow_m3_s": 0.01}},
                      {"id": "out", "from": "T", "flow": {"type": "linear_resistance", "resistance_s_m2": 400.0}}]}
    r = ok(TDF.dynamic_flowsheet_run(fs, 2000.0, events=[{"time_s": 1500.0, "target": "F.volumetric_flow_m3_s",
                                                         "value": 0.0}]))
    assert [i["event"] for i in r["incidents"]][:1] == ["overflow"]
    t_full = r["incidents"][0]["time_s"]
    assert t_full == pytest.approx(-400.0 * math.log(1 - 2.0 / 4.0), rel=1e-8)   # h = qR(1 − e^(−t/AR)) reaches 2 m
    b = r["evidence"]["balances"]["liquid_volume_m3"]
    assert b["closure_rel"] < 1e-9 and b["spilled"] > 0                          # measured 5.5e-12
    assert r["series"]["T.level_m"][0] == 0.0


def test_refusals_on_missing_or_meaningless_inputs():
    with pytest.raises(ValueError, match="area_m2"):
        F.simulate({"units": [{"id": "T", "type": "tank", "initial": {"level_m": 1}}]}, 10)
    with pytest.raises(ValueError, match="initial.level_m"):
        F.simulate({"units": [{"id": "T", "type": "tank", "area_m2": 1}]}, 10)
    with pytest.raises(ValueError, match="t_end_s"):
        F.simulate(tank(), None)
    bad = _seborg(0.5, 350.0)
    del bad["units"][0]["reactions"][0]["heat_of_reaction_J_mol"]
    with pytest.raises(ValueError, match="heat_of_reaction_J_mol"):
        F.simulate(bad, 10)
    bad = _seborg(0.5, 350.0)
    del bad["streams"][0]["temperature_K"]
    with pytest.raises(ValueError, match="temperature_K"):
        F.simulate(bad, 10)
    with pytest.raises(ValueError, match="lo fija el controlador"):
        F.simulate(_loop(30.0), 10, events=[{"time_s": 1, "target": "F.volumetric_flow_m3_s", "value": 0.02}])
    v = {"liquid": {"density_kg_m3": 1000.0},
         "units": [{"id": "T", "type": "tank", "area_m2": 1.0, "initial": {"level_m": 1.0}}],
         "streams": [{"id": "V", "from": "T", "flow": {"type": "valve", "Kv_m3_h": 10, "opening": 1.5,
                                                       "characteristic": "linear", "downstream_pressure_Pa": 1e5}}]}
    with pytest.raises(ValueError, match="entre 0 y 1"):
        F.simulate(v, 10)
    with pytest.raises(ValueError, match="step_size"):
        F.step_test(tank(), "F.volumetric_flow_m3_s", 0.0, "T.level_m", 100, 10)
    with pytest.raises(ValueError, match="no es un parámetro"):
        F.step_test(tank(), "T.level_m", 0.1, "T.level_m", 100, 10)
    with pytest.raises(ValueError, match="pre_exponential"):
        bad = _seborg(0.5, 350.0)
        del bad["units"][0]["reactions"][0]["pre_exponential"]
        F.simulate(bad, 10)


def test_steady_state_tool_and_wrappers_return_json():
    from packages.agent import wrappers_dynamic_flowsheet as W
    r = TDF.dynamic_flowsheet_steady_state(tank(h0=0.1))
    assert r["steady_state"]["T.level_m"] == pytest.approx(0.5, rel=1e-10) and r["stable"]
    json.dumps(r)
    assert [w.name for w in W.AGENT_WRAPPERS] == [f.__name__ for f in TDF.DYNAMIC_FLOWSHEET_TOOLS]
    out = json.loads(W.dynamic_flowsheet_run(tank(), 100.0, None, ["T.level_m"]))
    assert out["evidence"]["verified"] and out["series"]["T.level_m"][-1] == pytest.approx(0.5, rel=1e-9)


# ----------------------------------------------------------------------------------- valve inherent characteristics
def test_valve_inherent_characteristics_at_constant_pressure_drop():
    """Inherent characteristic (constant ΔP = 1 bar, SG = 1, so q[m³/h] = Kv·f(x)). Equal percentage f = R^(x−1):
    worked example with R = 50 in https://instrumentationtools.com/control-valve-characteristics (read 2026-11-14):
    0.956, 1.414 and 2.091 m³/h at 40, 50 and 60 % lift for a valve passing 10 m³/h fully open. Linear f = x is the
    IEC/ISA definition; quick opening f = √x is the textbook convention (the same page notes that the quick-opening
    shape is not defined in the standards): CONSISTENCY only."""
    def fs(char, extra=None):
        law = {"type": "valve", "Kv_m3_h": 10.0, "opening": 0.4, "characteristic": char, "upstream_pressure_Pa": 2e5}
        law.update(extra or {})
        return {"liquid": {"density_kg_m3": 1000.0},
                "units": [{"id": "T", "type": "tank", "area_m2": 10.0, "pressure_Pa": 1e5, "initial": {"level_m": 1.0}}],
                "streams": [{"id": "V", "to": "T", "flow": law}]}
    ev = [{"time_s": 10, "target": "V.opening", "value": 0.5}, {"time_s": 20, "target": "V.opening", "value": 0.6}]
    expected = {"equal_percentage": ([0.956, 1.414, 2.091], 5e-4), "linear": ([4.0, 5.0, 6.0], 1e-12),
                "quick_opening": ([10 * math.sqrt(x) for x in (0.4, 0.5, 0.6)], 1e-12)}
    for char, (ref, tol) in expected.items():
        r = ok(F.simulate(fs(char, {"rangeability": 50} if char == "equal_percentage" else None), 30, events=ev,
                          outputs=["V.volumetric_flow_m3_s"], n_points=31))
        q = [r["series"]["V.volumetric_flow_m3_s"][i] * 3600 for i in (5, 15, 25)]
        assert q == pytest.approx(ref, abs=tol)                                  # eq. %: measured 3.5e-4 (rounding)
    assert U.valve_fraction(0.0, "equal_percentage", 50.0) == pytest.approx(1 / 50)  # f0 = 1/R (leakage at x = 0)
    assert U.valve_fraction(0.6, "equal_percentage", 50.0) / U.valve_fraction(0.5, "equal_percentage", 50.0) == \
        pytest.approx(50 ** 0.1)                                                 # equal % change per equal travel


# ------------------------------------------------------------------------------------ gas fixed-flow outlet guard
@pytest.mark.parametrize("thermal", ["isothermal", "adiabatic"])
def test_gas_fixed_outlet_cannot_drain_vessel_below_zero(thermal):
    fs = {"units": [{"id": "V", "type": "gas_vessel", "volume_m3": 2.0, "thermal": thermal,
                     "gas": {"model": "ideal", "molar_mass_kg_mol": 0.02897, "heat_capacity_ratio": 1.4},
                     "initial": {"pressure_Pa": 3e5, "temperature_K": 300.0}}],
          "streams": [{"id": "OUT", "from": "V", "flow": {"type": "fixed", "mass_flow_kg_s": 0.05}},
                      {"id": "IN", "to": "V", "flow": {"type": "fixed", "mass_flow_kg_s": 0.0,
                                                       "upstream_temperature_K": 300.0}}]}
    m0 = 3e5 * 2.0 / (U.R_GAS / 0.02897 * 300.0)
    r = ok(F.simulate(fs, 200.0, events=[{"time_s": 150.0, "target": "IN.mass_flow_kg_s", "value": 0.08}],
                      outputs=["V.mass_kg", "OUT.mass_flow_kg_s"]))
    dry = [i for i in r["incidents"] if i["event"] == "dry"]
    assert len(dry) == 1 and dry[0]["time_s"] == pytest.approx(m0 / 0.05, rel=1e-8)   # empty at m0/ṁ
    assert min(r["series"]["V.mass_kg"]) >= 0.0
    t, q = S(r, "OUT.mass_flow_kg_s")
    assert np.all(q[(t > m0 / 0.05 + 1) & (t < 150)] == 0.0)                     # outlet starved while empty
    assert r["series"]["V.mass_kg"][-1] == pytest.approx(50 * 0.03, rel=1e-6)    # refills at 0.08 − 0.05 kg/s
    assert r["evidence"]["balances"]["gas_mass_kg"]["closure_rel"] < 1e-9


# ------------------------------------------------------------------------------------------------- flash drum
from packages.dynamics import flash_drum as FD  # noqa: E402


def test_flash_drum_reaches_kantor_isothermal_flash():
    """Published steady flash (Kantor, CBE 20255 notebook 07.09, read 2026-11-14,
    https://github.com/jckantor/CBE20255/blob/master/notebooks/07.09-Isothermal-Flash-and-the-Rachford-Rice-Equation.ipynb):
    acetone/ethanol z = 0.6/0.4, 65 °C, 760 mmHg, Raoult with Antoine → K = 1.338/0.576, φ = 0.2317, x = 0.5565/0.4435,
    y = 0.7444/0.2556. Dynamic drum with the vapour draw fixed at φ·F: it must settle at 760 mmHg and those x, y."""
    mm = 101325 / 760
    pa = 10 ** (7.02447 - 1161.0 / (224 + 65)) * mm
    pb = 10 ** (8.04494 - 1554.3 / (222.65 + 65)) * mm
    assert pa / 101325 == pytest.approx(1.338, abs=5e-4) and pb / 101325 == pytest.approx(0.576, abs=5e-4)
    d = {"components": ["acetone", "ethanol"], "temperature_K": 338.15, "volume_m3": 0.6, "area_m2": 0.2,
         "liquid_molar_density_mol_m3": 13000.0, "liquid_density_kg_m3": 780.0,
         "equilibrium": {"model": "relative_volatility", "alpha": {"acetone": pa / pb, "ethanol": 1.0},
                         "reference_vapour_pressure_Pa": pb},
         "feed": {"molar_flow_mol_s": 10.0, "mole_fractions": {"acetone": 0.6, "ethanol": 0.4}},
         "vapour_outlet": {"type": "fixed", "molar_flow_mol_s": 2.317},
         "liquid_outlet": {"type": "valve", "Kv_m3_h": 10.0, "opening": 0.6, "characteristic": "linear",
                           "downstream_pressure_Pa": 1e5},
         "initial": {"liquid_level_m": 0.8, "mole_fractions": {"acetone": 0.6, "ethanol": 0.4}}}
    r = ok(TDF.dynamic_flash_drum(d, 12000.0))
    fsx = r["final_state"]
    assert r["at_steady_state"]
    assert fsx["pressure_Pa"] == pytest.approx(101325.0, rel=2e-5)              # measured 5.6e-6 (φ given to 4 digits)
    assert [fsx["x.acetone"], fsx["x.ethanol"], fsx["y.acetone"], fsx["y.ethanol"]] == \
        pytest.approx([0.5565, 0.4435, 0.7444, 0.2556], abs=6e-5)               # 4-decimal source
    assert r["rachford_rice_check"]["max_deviation"] < 1e-6                     # measured 5.6e-10
    assert r["evidence"]["worst_balance_closure_rel"] < 1e-9


def _kflash(C, Kv, zi):
    return {"components": ["A", "B", "C"], "temperature_K": 350.0, "volume_m3": 0.15, "area_m2": 0.05,
            "liquid_molar_density_mol_m3": 10000.0, "liquid_density_kg_m3": 800.0,
            "equilibrium": {"model": "k_values", "K": {"A": 3.0, "B": 1.2, "C": 0.3}},
            "feed": {"molar_flow_mol_s": 5.0, "mole_fractions": {"A": 0.3, "B": 0.3, "C": 0.4}},
            "vapour_outlet": {"type": "valve", "coefficient_mol_s_Pa05": C, "opening": 0.5,
                              "characteristic": "equal_percentage", "rangeability": 50, "downstream_pressure_Pa": 1e5},
            "liquid_outlet": {"type": "valve", "Kv_m3_h": Kv, "opening": 0.5, "characteristic": "linear",
                              "downstream_pressure_Pa": 1e5},
            "initial": {"liquid_level_m": 0.5, "mole_fractions": dict(zip("ABC", zi))}}


def test_flash_drum_constant_k_steady_states_are_closed_forms_and_balances_close():
    """Constant K: V/F = β(z_feed) by Rachford-Rice, then P = P_d + (βF/(C f))² from the vapour valve and the level from
    the liquid valve, all in closed form; a feed step and a vapour-valve step move the end state to the new closed form."""
    K, z, F0 = np.array([3.0, 1.2, 0.3]), np.array([0.3, 0.3, 0.4]), 5.0
    b = FD.rachford_rice(z, K)
    assert float(np.sum(z * (K - 1) / (1 + b * (K - 1)))) == pytest.approx(0.0, abs=1e-14)
    x = z / (1 + b * (K - 1))
    zi = 0.98 * x + 0.02 * K * x                              # start near the liquid of the flash (mostly liquid holdup)
    feq = lambda op: 50.0 ** (op - 1)                         # noqa: E731
    C = b * F0 / (feq(0.5) * math.sqrt(1e5))                  # the user's valve sized for 2 bar
    Kv = (1 - b) * F0 / 1e4 * 3600 / math.sqrt((1e5 + 800 * U.G * 0.5) / 1e5 / 0.8) / 0.5      # and for 0.5 m

    def closed(beta, F, op_v):
        P = 1e5 + (beta * F / (C * feq(op_v))) ** 2
        dp = ((1 - beta) * F / 1e4 * 3600 / (Kv * 0.5)) ** 2 * 0.8 * 1e5
        return P, (dp - (P - 1e5)) / (800 * U.G)

    r = ok(FD.simulate_flash_drum(_kflash(C, Kv, zi), 20000.0))
    P, h = closed(b, F0, 0.5)
    assert r["at_steady_state"] and P == pytest.approx(2e5) and h == pytest.approx(0.5)
    assert r["final_state"]["pressure_Pa"] == pytest.approx(P, rel=1e-6)
    assert r["final_state"]["liquid_level_m"] == pytest.approx(h, rel=1e-3)     # measured 2.2e-4 (slow level mode)
    # the steps are chosen so that a physical steady state exists (0 < h < drum height 3 m); others empty or flood it
    z2 = np.array([0.31, 0.30, 0.39])
    ev = [{"time_s": 1000, "target": "feed.molar_flow_mol_s", "value": 4.5},
          {"time_s": 1000, "target": "feed.mole_fractions", "value": dict(zip("ABC", z2))},
          {"time_s": 1500, "target": "vapour.opening", "value": 0.55}]
    r = ok(FD.simulate_flash_drum(_kflash(C, Kv, zi), 40000.0, events=ev))
    P2, h2 = closed(FD.rachford_rice(z2, K), 4.5, 0.55)
    assert 0.0 < h2 < 3.0
    assert r["at_steady_state"]
    assert r["final_state"]["pressure_Pa"] == pytest.approx(P2, rel=1e-6)          # measured 1.9e-8
    assert r["final_state"]["liquid_level_m"] == pytest.approx(h2, rel=1e-3)       # measured 3.5e-5 (slow level mode)
    assert r["rachford_rice_check"]["max_deviation"] < 1e-6
    assert all(e["hit_exactly"] for e in r["evidence"]["scheduled_events"])
    b_ = r["evidence"]["balances"]
    assert b_["total_mol"]["closure_rel"] < 1e-9 and all(b_[f"component_mol.{c}"]["closure_rel"] < 1e-9 for c in "ABC")
    s = FD.simulate_flash_drum(_kflash(C, Kv, zi), 50.0, start_from_steady_state=True, refine=False)
    assert s["series"]["pressure_Pa"][0] == pytest.approx(P, rel=1e-8)


def test_flash_drum_reports_leaving_two_phase_region_and_refuses_missing_data():
    K, z = np.array([3.0, 1.2, 0.3]), np.array([0.3, 0.3, 0.4])
    b = FD.rachford_rice(z, K)
    x = z / (1 + b * (K - 1))
    zi = 0.98 * x + 0.02 * K * x
    r = FD.simulate_flash_drum(_kflash(0.01, 5.0, zi), 3000.0)       # vapour valve far too small: liquid blown out
    assert not r["evidence"]["verified"] and "sin líquido" in " ".join(r["evidence"]["reasons"])
    assert r["t_reached_s"] < 3000.0
    json.dumps(r)
    bad = _kflash(0.01, 5.0, zi)
    del bad["equilibrium"]["K"]
    with pytest.raises(ValueError, match="'K'"):
        FD.simulate_flash_drum(bad, 10.0)
    bad = _kflash(0.01, 5.0, zi)
    bad["initial"]["mole_fractions"] = {"A": 0.0, "B": 0.0, "C": 1.0}          # pure heavy: subcooled, no vapour
    with pytest.raises(ValueError, match="bifásico"):
        FD.simulate_flash_drum(bad, 10.0)
    bad = _kflash(0.01, 5.0, zi)
    bad["vapour_outlet"]["opening"] = 1.2
    with pytest.raises(ValueError, match="entre 0 y 1"):
        FD.simulate_flash_drum(bad, 10.0)
    with pytest.raises(ValueError, match="t_end_s"):
        FD.simulate_flash_drum(_kflash(0.01, 5.0, zi), None)


def test_flash_drum_connected_by_streams_to_tanks_and_valves():
    """Feed tank → valve → flash drum → valve → product tank: products are the Rachford-Rice flash of the feed at the
    steady state, the drum pressure set by the vapour valve is the closed form, balances close (drum moles included),
    and a level loop on the liquid valve holds its setpoint."""
    K, z, rm, Fm = np.array([3.0, 1.2, 0.3]), np.array([0.3, 0.3, 0.4]), 1e4, 5.0
    b = FD.rachford_rice(z, K)
    x = z / (1 + b * (K - 1))
    zi = 0.98 * x + 0.02 * K * x
    C = b * Fm / (50 ** -0.5 * math.sqrt(1e5))                  # vapour valve sized for 2 bar
    fs = {"components": ["A", "B", "C"], "liquid": {"density_kg_m3": 800.0},
          "units": [{"id": "T1", "type": "tank", "area_m2": 1.0, "pressure_Pa": 3e5,
                     "initial": {"level_m": 1.0, "concentration_mol_m3": dict(zip("ABC", z * rm))}},
                    {"id": "FL", "type": "flash_drum", "temperature_K": 350.0, "volume_m3": 0.15, "area_m2": 0.05,
                     "liquid_molar_density_mol_m3": rm, "liquid_density_kg_m3": 800.0,
                     "equilibrium": {"model": "k_values", "K": {"A": 3.0, "B": 1.2, "C": 0.3}},
                     "vapour_outlet": {"type": "valve", "coefficient_mol_s_Pa05": C, "opening": 0.5,
                                       "characteristic": "equal_percentage", "rangeability": 50,
                                       "downstream_pressure_Pa": 1e5},
                     "initial": {"liquid_level_m": 0.5, "mole_fractions": dict(zip("ABC", zi))}},
                    {"id": "T2", "type": "tank", "area_m2": 2.0,
                     "initial": {"level_m": 0.5, "concentration_mol_m3": dict(zip("ABC", x * rm))}}],
          "streams": [{"id": "F", "to": "T1", "flow": {"type": "fixed", "volumetric_flow_m3_s": Fm / rm},
                       "concentration_mol_m3": dict(zip("ABC", z * rm))},
                      {"id": "S1", "from": "T1", "to": "FL", "flow": {"type": "valve", "Kv_m3_h": 3.10, "opening": 0.5,
                                                                      "characteristic": "linear"}},
                      {"id": "S2", "from": "FL", "to": "T2", "flow": {"type": "valve", "Kv_m3_h": 1.895, "opening": 0.5,
                                                                      "characteristic": "linear",
                                                                      "downstream_pressure_Pa": 1e5}},
                      {"id": "P", "from": "T2", "flow": {"type": "linear_resistance",
                                                         "resistance_s_m2": 0.5 / ((1 - b) * Fm / rm)}}]}
    r = ok(TDF.dynamic_flowsheet_run(fs, 20000.0, outputs=["FL.pressure_Pa", "FL.liquid_level_m", "FL.x.A"]))
    assert r["at_steady_state"] and r["series"]["FL.pressure_Pa"][-1] == pytest.approx(2e5, rel=1e-3)   # meas. 9e-5
    assert max(v["closure_rel"] for v in r["evidence"]["balances"].values()) < 1e-12
    ss = ok(TDF.dynamic_flowsheet_steady_state(fs))["steady_state"]
    v, lq = ss["FL.vapour_flow_mol_s"], ss["S2.volumetric_flow_m3_s"] * rm
    assert v / (v + lq) == pytest.approx(b, abs=1e-12) and ss["FL.pressure_Pa"] == pytest.approx(2e5, rel=1e-12)
    assert [ss[f"FL.x.{c}"] for c in "ABC"] == pytest.approx(list(x), abs=1e-12)
    assert [ss[f"FL.y.{c}"] for c in "ABC"] == pytest.approx(list(K * x), abs=1e-12)
    fs["controllers"] = [{"id": "LC", "measurement": "FL.liquid_level_m", "setpoint": 0.8, "Kc": -2.0, "tau_I_s": 200.0,
                          "manipulated": "S2.opening", "output_min": 0.0, "output_max": 1.0}]
    ss = ok(TDF.dynamic_flowsheet_steady_state(fs))
    assert ss["steady_state"]["FL.liquid_level_m"] == pytest.approx(0.8, rel=1e-10) and ss["stable"]
    with pytest.raises(ValueError, match="lo fija el equilibrio"):
        F.simulate(fs, 10, events=[{"time_s": 1, "target": "FL.pressure_Pa", "value": 3e5}])


# ------------------------------------------------------------------------------------- distillation: Skogestad A
# [SKA] Skogestad, "MATLAB Distillation column model (Column A)", https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html
#       and its files colamod.m, cola_lv.m, cola.dat (steady profile, 14 digits), read 2026-11-14. NT = 41 (1 = reboiler,
#       41 = total condenser), NF = 21, α = 1.5, F = 1 kmol/min, zF = 0.5, qF = 1, L = 2.70629, V = 3.20629, D = B = 0.5,
#       M0 = 0.5 kmol, τL = 0.063 min, λ = 0; yD = 0.99, xB = 0.01; open-loop eigenvalues 0, 0, −5.1562e-3 1/min
#       (dominant time constant 193.9 min); +1 % feed (nonlinear): yD 0.9925 (+0.0025), xB 0.0172 (+0.0072); reboiler
#       holdup rises like a ramp after ≈ 20·0.063 = 1.26 min. Converted to SI here (mol, s).
KMIN = 1000.0 / 60.0                                       # kmol/min → mol/s
COLA_X = [1.0e-2, 1.4260912285191e-2, 1.9723589240705e-2, 2.6693264811825e-2, 3.5531128682305e-2, 4.6650909897754e-2,
          6.0505364325745e-2, 7.7557807588356e-2, 9.8234185152883e-2, 0.12285384713922, 0.15154333129394,
          0.18414705878182, 0.22015928656995, 0.25870700564480, 0.29860668074760, 0.33849626436079, 0.37701507466589,
          0.41298329547589, 0.44553302705302, 0.47416387897878, 0.49872493204438, 0.52649475964320, 0.55776390242974,
          0.59216060127435, 0.62903916925421, 0.66750673731202, 0.70649840345461, 0.74489017363107, 0.78162559565357,
          0.81582677099606, 0.84686636753928, 0.87439107401281, 0.89830161503826, 0.91870393775007, 0.93584846972424,
          0.95007115492539, 0.96174449436761, 0.97124167454496, 0.97891332406394, 0.98507462686567, 0.99]
COLA_L, COLA_V = 2.70629295971439, 3.20629295971439


def _cola(config="LV", feed=1.0, x0=None, extra_ctl=None):
    u = {"id": "C", "type": "distillation_column", "stages": 41, "feed_stage": 21, "relative_volatility": 1.5,
         "hydraulics": {"tau_L_s": 0.063 * 60, "lambda": 0.0, "nominal_holdup_mol": 500.0,
                        "nominal_liquid_flow_top_mol_s": COLA_L * KMIN,
                        "nominal_liquid_flow_bottom_mol_s": (COLA_L + 1.0) * KMIN},
         "feed": {"molar_flow_mol_s": feed * KMIN, "mole_fraction_light": 0.5, "liquid_fraction": 1.0},
         "reflux_mol_s": COLA_L * KMIN, "boilup_mol_s": COLA_V * KMIN, "distillate_mol_s": 0.5 * KMIN,
         "bottoms_mol_s": 0.5 * KMIN, "initial": {"x": x0 or COLA_X, "holdup_mol": 500.0}}
    fs = {"units": [u]}
    kc = -10.0 / 60.0                                       # cola_lv.m: P level loops with gain 10 1/min
    if config == "LV":
        fs["controllers"] = [{"id": "LD", "measurement": "C.condenser_holdup_mol", "setpoint": 500.0, "Kc": kc,
                              "bias": 0.5 * KMIN, "manipulated": "C.distillate_mol_s"},
                             {"id": "LB", "measurement": "C.reboiler_holdup_mol", "setpoint": 500.0, "Kc": kc,
                              "bias": 0.5 * KMIN, "manipulated": "C.bottoms_mol_s"}]
    elif config == "DV":
        fs["controllers"] = [{"id": "LD", "measurement": "C.condenser_holdup_mol", "setpoint": 500.0, "Kc": kc,
                              "bias": COLA_L * KMIN, "manipulated": "C.reflux_mol_s"},
                             {"id": "LB", "measurement": "C.reboiler_holdup_mol", "setpoint": 500.0, "Kc": kc,
                              "bias": 0.5 * KMIN, "manipulated": "C.bottoms_mol_s"}]
    fs.setdefault("controllers", []).extend(extra_ctl or [])
    return fs


STAGES = [f"C.x.{i}" for i in range(1, 42)]


def test_column_a_steady_profile_eigenvalues_and_feed_step():
    r = ok(TDF.dynamic_flowsheet_steady_state(_cola(), outputs=STAGES))
    xs = [r["outputs"][n] for n in STAGES]
    assert xs == pytest.approx(COLA_X, abs=1e-10)                              # 41 values; measured 1.1e-12
    m = F.Model(_cola(None, x0=xs))                                            # open loop (no level control)
    m.set_schedule(None, 1.0)
    n, sc, y = m.n_dyn, m.scale[:m.n_dyn], m.y0.copy()

    def f(xx):
        y[:n] = xx * sc
        return m.rhs(0.0, y)[:n] / sc

    assert np.max(np.abs(f(m.y0[:n] / sc))) < 1e-12
    ev = np.sort(np.linalg.eigvals(F._jacobian(f, m.y0[:n] / sc, 1e-6)).real)[::-1] * 60.0     # 1/min
    assert abs(ev[0]) < 1e-9 and abs(ev[1]) < 1e-9                              # the two uncontrolled levels
    assert ev[2] == pytest.approx(-5.1562e-3, rel=5e-4)                        # measured −5.15545e-3 (1.5e-4)
    st = ok(TDF.dynamic_flowsheet_steady_state(_cola(feed=1.01, x0=xs), outputs=["C.distillate_x", "C.bottoms_x"]))
    assert st["outputs"]["C.distillate_x"] == pytest.approx(0.9925, abs=5e-5)  # measured 0.992495
    assert st["outputs"]["C.bottoms_x"] == pytest.approx(0.0172, abs=5e-5)     # measured 0.017162
    dv = ok(TDF.dynamic_flowsheet_steady_state(_cola("DV", x0=xs), outputs=STAGES))   # same flows → same state
    assert [dv["outputs"][n_] for n_ in STAGES] == pytest.approx(COLA_X, abs=1e-10)


def test_column_a_feed_step_reboiler_ramp_and_balances():
    """Open loop, +1 % feed of saturated liquid: the extra liquid crosses 20 linear tray lags (τL = 3.78 s) and the
    reboiler holdup follows the Erlang ramp ΔF·[t·P(20, t/τ) − 20τ·P(21, t/τ)] → ΔF·(t − 1.26 min)."""
    from scipy.special import gammainc
    r = ok(TDF.dynamic_flowsheet_run(_cola(None), 600.0, events=[{"time_s": 0.0, "target": "C.feed_mol_s",
                                                                  "value": 1.01 * KMIN}],
                                     outputs=["C.reboiler_holdup_mol", "C.bottoms_x", "C.distillate_x"], n_points=61))
    t, mb = S(r, "C.reboiler_holdup_mol")
    tau, dF = 3.78, 0.01 * KMIN
    ex = 500.0 + dF * (t * gammainc(20, t / tau) - 20 * tau * gammainc(21, t / tau))
    assert np.max(np.abs(mb - ex)) < 1e-4                                       # mol; measured 2.4e-6
    assert mb[-1] - 500.0 == pytest.approx(dF * (600.0 - 20 * tau), rel=1e-9)
    b = r["evidence"]["balances"]
    assert b["column_total_mol"]["closure_rel"] < 1e-12 and b["column_light_mol"]["closure_rel"] < 1e-12
    assert S(r, "C.bottoms_x")[1][-1] > 0.01                                    # xB starts to rise


def _mt_profile(alpha, n_stages, nf, F, zF, D, B, LT, VB, xb):
    """Independent stage-to-stage (McCabe-Thiele) profile from the reboiler up: equilibrium y = αx/(1 + (α − 1)x) on
    each stage and the operating lines of a saturated-liquid feed."""
    x = [xb]
    for i in range(1, n_stages):
        yv = alpha * x[-1] / (1 + (alpha - 1) * x[-1])
        x.append((VB * yv + B * xb) / (LT + F) if i < nf else (VB * yv - (F * zF - B * xb)) / LT)
    return x


def _mccabe_thiele(alpha, n_stages, nf, F, zF, D, B, LT, VB, bracket=(0.005, 0.02)):
    """Given the flows, xB such that the stepped condenser composition equals the overall balance (brentq)."""
    from scipy.optimize import brentq
    xb = brentq(lambda xb: _mt_profile(alpha, n_stages, nf, F, zF, D, B, LT, VB, xb)[-1] - (F * zF - B * xb) / D,
                *bracket, xtol=1e-15)
    return _mt_profile(alpha, n_stages, nf, F, zF, D, B, LT, VB, xb)


def test_column_mccabe_thiele_and_fenske_total_reflux():
    r = ok(TDF.dynamic_flowsheet_steady_state(_cola(), outputs=STAGES))
    mt = _mccabe_thiele(1.5, 41, 21, 1.0, 0.5, 0.5, 0.5, COLA_L, COLA_V)
    assert [r["outputs"][n] for n in STAGES] == pytest.approx(mt, abs=1e-9)
    # total reflux (F = D = B = 0, L = V): the stationary profile is x_{i+1} = y_i, i.e. Fenske with N = NT − 1
    a, xb, NT = 1.5, 0.02, 41
    x = [xb]
    for _ in range(NT - 1):
        x.append(a * x[-1] / (1 + (a - 1) * x[-1]))
    fs = _cola(None, x0=x)
    u = fs["units"][0]
    u["feed"]["molar_flow_mol_s"] = 0.0
    u["distillate_mol_s"] = u["bottoms_mol_s"] = 0.0
    u["reflux_mol_s"] = u["boilup_mol_s"] = COLA_V * KMIN
    u["hydraulics"]["nominal_liquid_flow_top_mol_s"] = u["hydraulics"]["nominal_liquid_flow_bottom_mol_s"] = COLA_V * KMIN
    n_min = math.log(x[-1] / (1 - x[-1]) * (1 - xb) / xb) / math.log(a)
    assert n_min == pytest.approx(NT - 1, abs=1e-9)
    rt = ok(TDF.dynamic_flowsheet_run(fs, 120.0, outputs=["C.distillate_x", "C.bottoms_x"], n_points=5))
    assert rt["at_steady_state"]
    assert rt["series"]["C.distillate_x"][-1] == pytest.approx(x[-1], abs=1e-10)


def test_column_composition_loops_reach_specs_given_by_mccabe_thiele():
    """LV with P level loops plus two PI composition loops (yD by L, xB by V): the closed-loop steady state meets both
    specs, and the reflux and boilup it settles on are the ones an independent McCabe-Thiele design needs. Newton
    starts from the design profile but with the nominal L and V (the loops' integral states must move to the design)."""
    from scipy.optimize import brentq
    # specs fix D = B = 0.5 by the overall balance; McCabe-Thiele then needs the reflux that steps 0.005 → 0.995
    lt = brentq(lambda lt: _mt_profile(1.5, 41, 21, 1.0, 0.5, 0.5, 0.5, lt, lt + 0.5, 0.005)[-1] - 0.995, 2.8, 4.0,
                xtol=1e-14)
    vb = lt + 0.5
    x_start = _mt_profile(1.5, 41, 21, 1.0, 0.5, 0.5, 0.5, lt, vb, 0.005)
    loops = [{"id": "XD", "measurement": "C.distillate_x", "setpoint": 0.995, "Kc": 50.0 * KMIN, "tau_I_s": 600.0,
              "manipulated": "C.reflux_mol_s"},
             {"id": "XB", "measurement": "C.bottoms_x", "setpoint": 0.005, "Kc": -50.0 * KMIN, "tau_I_s": 600.0,
              "manipulated": "C.boilup_mol_s"}]
    r = ok(TDF.dynamic_flowsheet_steady_state(_cola(x0=x_start, extra_ctl=loops), outputs=["C.distillate_x", "C.bottoms_x",
                                                                               "XD.output", "XB.output"]))
    o = r["outputs"]
    assert o["C.distillate_x"] == pytest.approx(0.995, abs=1e-10) and o["C.bottoms_x"] == pytest.approx(0.005, abs=1e-10)
    assert o["XD.output"] / KMIN == pytest.approx(lt, rel=1e-7) and o["XB.output"] / KMIN == pytest.approx(vb, rel=1e-7)
    d = ok(TDF.dynamic_flowsheet_run(_cola(x0=x_start, extra_ctl=loops), 30000.0, outputs=["C.distillate_x",
                                                                                         "C.bottoms_x"]))
    assert d["at_steady_state"] and d["series"]["C.distillate_x"][-1] == pytest.approx(0.995, abs=1e-6)   # 1.6e-7
    assert d["evidence"]["balances"]["column_light_mol"]["closure_rel"] < 1e-12


def test_column_refusals():
    fs = _cola(None)
    del fs["units"][0]["hydraulics"]["tau_L_s"]
    with pytest.raises(ValueError, match="tau_L_s"):
        F.simulate(fs, 10)
    fs = _cola(None)
    del fs["units"][0]["reflux_mol_s"]
    with pytest.raises(ValueError, match="reflux_mol_s"):
        F.simulate(fs, 10)
    fs = _cola(None)
    fs["units"][0]["feed_stage"] = 41
    with pytest.raises(ValueError, match="feed_stage"):
        F.simulate(fs, 10)
