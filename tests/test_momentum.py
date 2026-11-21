"""Momentum transfer in lines (bloque MG): Fanno flow, inclined two-phase flow, water hammer, pump selection.

Validation sources (read 2026-11-15):
  * Fanno table, γ = 1.4: Purdue University AAE propulsion pages, «Fanno Flow» table (M, T/T*, p/p*, ρ/ρ*, pt/pt*,
    4fL*/D; 4 significant digits; γ not printed, consistent with 1.4; 4f_Fanning = f_Darcy).
    https://engineering.purdue.edu/AAE/research/propulsion/Info/flow/fanno/
  * Fanno worked example: G. Bar-Meir, «Fluid Mechanics», LibreTexts 11.7.12 «More Examples of Fanno Flow» (air k = 1.4,
    4fL/D = 40, entrance stagnation 300 K and 3 bar, P2/P1 = 0.1, 0.3, 0.5, 0.8 → M1 = 0.12728 (choked), 0.12420, 0.11392,
    0.07975). The printed base mass flow 0.48 kg/s is off by a factor 10 versus the page's own formula and data
    (0.048 kg/s), so only the Mach numbers are used.
    https://eng.libretexts.org/Bookshelves/Civil_Engineering/Fluid_Mechanics_(Bar-Meir)/11%3A_Compressible_Flow_One_Dimensional/11.7%3A_Fanno_Flow/11.7.12%3A_More_Examples_of_Fanno_Flow
  * Pump operating point: E. Maurer, «Hydraulics and Water Resources: Examples Using R», Santa Clara University, ch. 7,
    examples 7.1-7.3 (CC BY-NC 4.0): system h = 30 + 0.160·Q² (K from D = 20 in, L = 3884 ft, ks = 0.0005 ft,
    ν = 1.23e-5 ft²/s, Q in ft³/s), pump (0, 81), (5000 gpm, 60), (7850 gpm, 20 ft) fitted «poly2» h = 82.5 − 0.201·Q²,
    operating point Q = 12.051 ft³/s, h = 53.285 ft.
    https://edm44.github.io/hydr-watres-book/pumps-and-how-they-operate-in-a-hydraulic-system.html
  * Two-phase: DWSIM pipe segment (Beggs-Brill, the existing DWSIM client); skipped when DWSIM is absent. DWSIM's source
    (github.com/DanWBR/dwsim, DWSIM.UnitOperations/FluidFlowCorrelations/BeggsBrill.vb, read 2026-11-15) computes the liquid
    velocity number as 1.938·vsl·(ρl/(32.2·σ))^0.25 (an extra g^0.25 inside the field-unit form) and applies the
    1 < y < 1.2 branch of S to ln(y) instead of y; both explain the measured deviations below.
  * Water hammer: analytical — Joukowsky ΔP = ρ·a·V0, the frictionless square wave of period 4L/a, and Allievi's chain
    equations ΔH(t) + ΔH(t − 2L/a) = (a/g)·[V(t − 2L/a) − V(t)] for a frictionless line with an orifice valve. Korteweg
    restraint factors as printed in A. S. Leon, «Water hammer flows», FIU course notes
    (https://web.eng.fiu.edu/arleon/courses/Transient_flows/Lectures/Waterhammer.pdf); no worked example with an answer
    could be opened.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_momentum as TM  # noqa: E402
from packages.transport import fanno as FA  # noqa: E402
from packages.transport import flow_devices as FD  # noqa: E402
from packages.transport import two_phase_inclined as TP  # noqa: E402
from packages.transport import water_hammer as WH  # noqa: E402
from packages.transport.pumps import FT, GPM, pump_selection  # noqa: E402

G = 9.80665
R_AIR = 287.0
MW_AIR = 8.314462618 / R_AIR


def _dumps(r):
    return json.dumps(r, default=float, ensure_ascii=False)


# ================================================================ Fanno
# Purdue AAE table, γ = 1.4: M, T/T*, p/p*, pt/pt*, 4fL*/D
PURDUE = [(0.10, 1.198, 10.94, 5.822, 66.92), (0.20, 1.190, 5.455, 2.964, 14.53), (0.30, 1.179, 3.619, 2.035, 5.299),
          (0.40, 1.163, 2.696, 1.590, 2.308), (0.50, 1.143, 2.138, 1.340, 1.069), (0.60, 1.119, 1.763, 1.188, 0.4908),
          (0.70, 1.093, 1.493, 1.094, 0.2081), (0.80, 1.064, 1.289, 1.038, 0.07229), (0.90, 1.033, 1.129, 1.009, 0.01451),
          (1.00, 1.000, 1.000, 1.000, 0.0), (1.50, 0.8276, 0.6065, 1.176, 0.1360), (2.00, 0.6667, 0.4082, 1.687, 0.3050),
          (3.00, 0.4286, 0.2182, 4.235, 0.5222)]


def test_fanno_table_purdue_gamma_14():
    worst = 0.0
    for M, tt, pp, p0, fld in PURDUE:
        r = FA.fanno_ratios(M, 1.4)
        for ours, ref in ((r["T_T_star"], tt), (r["P_P_star"], pp), (r["P0_P0_star"], p0), (r["fLD_star"], fld)):
            if ref == 0.0:
                assert abs(ours) < 1e-12
                continue
            worst = max(worst, abs(ours - ref) / ref)
    assert worst < 6e-4              # measured 4.0e-4: the table's 4-digit rounding (e.g. 10.94 vs 10.9435)


def test_bar_meir_example_inlet_mach():
    # 4fL/D = 4·0.05·4/0.02 = 40 (Fanning 0.05 → Darcy 0.2); M1 depends only on P2/P1 (static/static)
    cases = [(0.1, 0.12728, True), (0.3, 0.12420, False), (0.5, 0.11392, False), (0.8, 0.07975, False)]
    worst = 0.0
    for ratio, M_ref, choked in cases:
        r = FA.adiabatic_gas_pipe(3e5, 300.0, 0.02, 4.0, "static", P2_Pa=ratio * 3e5, gamma=1.4, molar_mass_kg_mol=MW_AIR,
                                  darcy_friction_factor=0.2, compare_isothermal=False)
        assert r["choked"] is choked and r["evidence"]["verified"]
        worst = max(worst, abs(r["inlet"]["Mach"] - M_ref) / M_ref)
        _dumps(r)
    assert worst < 1e-4               # measured 1.1e-5 to 3.9e-5 (printed to 5 digits)


def test_isothermal_limit_gamma_to_one():
    devs = []
    for g in (1.001, 1.0001, 1.00001):
        r = FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", mass_flow_kg_s=0.5, gamma=g, molar_mass_kg_mol=0.016,
                                  viscosity_Pa_s=1.1e-5)
        devs.append(abs(r["outlet"]["P_Pa"] / r["isothermal_comparison"]["P2_Pa"] - 1.0))
    assert devs[0] > devs[1] > devs[2] and devs[2] < 2e-7       # ∝ (γ − 1): measured 1.25e-5, 1.25e-6, 1.25e-7


def test_low_mach_matches_isothermal_and_round_trip():
    r = FA.adiabatic_gas_pipe(10e5, 300.0, 0.2, 100.0, "static", mass_flow_kg_s=1.0, gamma=1.3, molar_mass_kg_mol=0.016,
                              viscosity_Pa_s=1.1e-5)
    assert r["inlet"]["Mach"] < 0.02 and abs(r["isothermal_comparison"]["dP_ratio_adiabatic_over_isothermal"] - 1) < 1e-6
    a = FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", P2_Pa=5e5, gamma=1.3, molar_mass_kg_mol=0.016,
                              viscosity_Pa_s=1.1e-5)
    b = FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", mass_flow_kg_s=a["mass_flow_kg_s"], gamma=1.3,
                              molar_mass_kg_mol=0.016, viscosity_Pa_s=1.1e-5)
    assert b["outlet"]["P_Pa"] == pytest.approx(5e5, rel=1e-9)
    assert a["evidence"]["verified"] and b["evidence"]["ode_P_deviation_rel_P1"] < 1e-6
    # stagnation temperature is conserved and the outlet is colder (adiabatic expansion)
    assert b["outlet"]["stagnation_T_K"] == pytest.approx(b["inlet"]["stagnation_T_K"], rel=1e-12)
    assert b["outlet"]["T_K"] < b["inlet"]["T_K"] and b["stagnation_pressure_loss_Pa"] > 0


def test_choking_and_max_flow():
    kw = dict(gamma=1.3, molar_mass_kg_mol=0.016, viscosity_Pa_s=1.1e-5)
    c = FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 400.0, "static", P2_Pa=0.5e5, **kw)
    assert c["choked"] and c["outlet"]["Mach"] == pytest.approx(1.0, abs=1e-6) and c["evidence"]["verified"]
    assert c["critical_length_m"] == pytest.approx(400.0, rel=1e-8)
    over = FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 400.0, "static", mass_flow_kg_s=1.1 * c["mass_flow_kg_s"], **kw)
    assert over["feasible"] is False and over["outlet"] is None
    assert over["max_mass_flow_kg_s"] == pytest.approx(c["mass_flow_kg_s"], rel=1e-8)
    _dumps(c), _dumps(over)


def test_stagnation_basis_reservoir():
    r = FA.adiabatic_gas_pipe(3e5, 300.0, 0.02, 4.0, "stagnation", P2_Pa=0.3e5, gamma=1.4, molar_mass_kg_mol=MW_AIR,
                              darcy_friction_factor=0.2, compare_isothermal=False)
    M = r["inlet"]["Mach"]
    assert r["inlet"]["stagnation_T_K"] == pytest.approx(300.0, rel=1e-12)
    assert r["inlet"]["T_K"] == pytest.approx(300.0 / (1 + 0.2 * M * M), rel=1e-12)
    # choked here (P2/P0 = 0.1): the choked inlet Mach of Bar-Meir's example, mass flow from the page's formula
    assert r["choked"] and M == pytest.approx(0.12728, rel=1e-4)
    m_ref = 3e5 * math.pi * 0.02 ** 2 / 4 * M * math.sqrt(1.4 / (R_AIR * 300.0)) * (1 + 0.2 * M * M) ** -3.0
    assert r["mass_flow_kg_s"] == pytest.approx(m_ref, rel=1e-9)        # 0.0479 kg/s (page prints 0.48: typo ×10)


def test_gas_properties_from_compounds():
    r = TM.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", mass_flow_kg_s=0.5, compounds={"methane": 1.0})
    assert 1.28 < r["gamma"] < 1.32 and r["molar_mass_kg_mol"] == pytest.approx(0.016043, rel=1e-3)
    assert any("γ = Cp/(Cp − R)" in a for a in r["assumptions"]) and r["evidence"]["verified"]
    _dumps(r)


def test_fanno_refusals():
    with pytest.raises(ValueError, match="gamma"):
        FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", mass_flow_kg_s=0.5, viscosity_Pa_s=1e-5)
    with pytest.raises(ValueError, match="viscosity"):
        FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", mass_flow_kg_s=0.5, gamma=1.3, molar_mass_kg_mol=0.016)
    with pytest.raises(ValueError, match="inlet_basis"):
        FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "total", mass_flow_kg_s=0.5, gamma=1.3, molar_mass_kg_mol=0.016,
                              viscosity_Pa_s=1e-5)
    with pytest.raises(ValueError, match="exactamente uno"):
        FA.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", gamma=1.3, molar_mass_kg_mol=0.016, viscosity_Pa_s=1e-5)


# ================================================================ two-phase inclined
BASE = dict(mass_flow_kg_s=5.0, quality=0.147, liquid_density_kg_m3=996.0, gas_density_kg_m3=22.5,
            liquid_viscosity_Pa_s=8.6e-4, gas_viscosity_Pa_s=1.83e-5, surface_tension_N_m=0.0718, inner_diameter_m=0.1,
            length_m=100.0, inlet_pressure_Pa=2e6)


def test_beggs_brill_matches_fluids_across_regimes():
    seen = set()
    for m, x in ((0.3, 0.01), (2.0, 0.02), (5.0, 0.147), (6.0, 0.3), (15.0, 0.05), (1.0, 0.5), (30.0, 0.002)):
        for ang in (-90.0, -20.0, 0.0, 5.0, 45.0, 90.0):
            r = TP.two_phase_pipe_inclined(**{**BASE, "mass_flow_kg_s": m, "quality": x, "angle_deg": ang}, n_segments=4)
            seen.add(r["flow_pattern_inlet"])
            assert r["evidence"]["inlet_gradient_vs_fluids_Beggs_Brill_rel"] < 1e-9
    assert seen == set(TP.REGIMES)   # all four Beggs-Brill patterns exercised


def test_two_phase_physics():
    h = TP.two_phase_pipe_inclined(**BASE, angle_deg=0.0)
    assert h["dP_gravity_Pa"] == 0.0 and h["dP_friction_Pa"] > 0 and h["dP_acceleration_Pa"] > 0
    up = [TP.two_phase_pipe_inclined(**BASE, angle_deg=a)["dP_gravity_Pa"] for a in (10.0, 30.0, 60.0, 90.0)]
    assert all(b > a for a, b in zip(up, up[1:]))
    dn = TP.two_phase_pipe_inclined(**BASE, angle_deg=-90.0)
    assert dn["dP_gravity_Pa"] < 0 and dn["P_out_Pa"] > BASE["inlet_pressure_Pa"]
    v = TP.two_phase_pipe_inclined(**BASE, angle_deg=90.0)
    rho_s = v["dP_gravity_Pa"] / (G * BASE["length_m"])
    assert 22.5 < rho_s < 996.0 and v["evidence"]["verified"]
    assert v["evidence"]["segment_independence_rel"] < 1e-4
    assert v["separated_model"]["dP_total_Pa"] > 0 and v["method_spread_rel"] < 0.3
    _dumps(h), _dumps(dn), _dumps(v)


def test_two_phase_refusals():
    with pytest.raises(ValueError, match="quality"):
        TP.two_phase_pipe_inclined(**{**BASE, "quality": 0.0}, angle_deg=0.0)
    with pytest.raises(ValueError, match="faltan"):
        TP.two_phase_pipe_inclined(**{**BASE, "surface_tension_N_m": None}, angle_deg=0.0)
    with pytest.raises(ValueError, match="angle_deg"):
        TP.two_phase_pipe_inclined(**BASE, angle_deg=120.0)


def _dwsim_available() -> bool:
    try:
        from packages.dwsim_client import tools as T  # noqa: F401
        return True
    except Exception:
        return False


def _S(y: float, dwsim_branch: bool) -> float:
    ly = math.log(y)
    if (1.0 < ly < 1.2) if dwsim_branch else (1.0 < y < 1.2):
        return math.log(2.2 * (math.exp(ly) if dwsim_branch else y) - 1.2)
    return ly / (-0.0523 + 3.182 * ly - 0.8725 * ly ** 2 + 0.01853 * ly ** 4)


@pytest.mark.skipif(not _dwsim_available(), reason="DWSIM no disponible")
def test_against_dwsim_pipe_segment():
    from packages.dwsim_client import tools as T
    out = {}
    for dz in (0.0, 20.0, -20.0):
        fid = T.new_flowsheet(f"mg_tp_{dz}", ["Water", "Nitrogen"], "PengRobinson")["flowsheet_id"]
        T.add_material_stream(fid, "in", temperature_K=300.0, pressure_Pa=20e5, mass_flow_kg_s=5.0,
                              mole_fractions={"Water": 0.9, "Nitrogen": 0.1})
        T.add_material_stream(fid, "out")
        T.add_unit(fid, "pipe", "L-1", {"length_m": 100.0, "inner_diameter_m": 0.1, "elevation_change_m": dz,
                                        "increments": 20})
        T.connect(fid, "L-1", feeds=["in"], products=["out"])
        assert T.solve(fid)["solved"]
        u = T.read_unit(fid, "L-1")
        o = T._s(fid)["units"]["L-1"].Object
        o = getattr(o, "__implementation__", o)
        HL_dw = float(list(list(o.Profile.Sections.Values)[0].Results)[0].LiquidHoldup)
        ms = T._s(fid)["streams"]["in"].Object
        ms = getattr(ms, "__implementation__", ms)
        gp = lambda i, n: float(ms.Phases[i].Properties.GetType().GetProperty(n).GetValue(ms.Phases[i].Properties))
        r = TP.two_phase_pipe_inclined(5.0, gp(2, "massfraction"), gp(1, "density"), gp(2, "density"), gp(1, "viscosity"),
                                       gp(2, "viscosity"), gp(1, "surfaceTension"), 0.1, 100.0,
                                       math.degrees(math.asin(dz / 100.0)), 20e5)
        out[dz] = (u, HL_dw, r)
        T.close_flowsheet(fid)
    u, HL_dw, r = out[0.0]
    # horizontal: no inclination factor, both implementations agree (measured: friction −0.11 %, holdup 1.4e-5)
    assert r["dP_friction_Pa"] == pytest.approx(u["dP_friction_Pa"], rel=0.005)
    assert r["holdup_inlet"] == pytest.approx(HL_dw, rel=1e-4)
    # uphill 11.5°: DWSIM's holdup is reproduced exactly with its liquid velocity number NLv / g^0.25
    u, HL_dw, r = out[20.0]
    s = r["inlet_state"]
    HL_dw_formula = TP._bb_holdup(2, s["no_slip_holdup"], s["Froude"], math.asin(0.2), s["NLv"] / 32.174 ** 0.25)[0]
    assert HL_dw_formula == pytest.approx(HL_dw, rel=1e-4)
    assert r["holdup_inlet"] < HL_dw                                   # 0.267 vs 0.296
    assert r["dP_friction_Pa"] == pytest.approx(u["dP_friction_Pa"], rel=0.02)          # measured +1.2 %
    assert r["dP_gravity_Pa"] == pytest.approx(u["dP_static_Pa"], rel=0.10)             # measured −9.2 %
    assert r["dP_total_Pa"] == pytest.approx(u["dP_friction_Pa"] + u["dP_static_Pa"], rel=0.06)   # measured −4.9 %
    # downhill 11.5°: DWSIM's friction is ≈ 3.6× ours; the ratio is explained by its S branch on ln(y)
    u, HL_dw, r = out[-20.0]
    lam = r["inlet_state"]["no_slip_holdup"]
    y = lam / HL_dw ** 2
    ratio_expected = math.exp(_S(y, True)) / math.exp(_S(y, False))
    assert u["dP_friction_Pa"] / r["dP_friction_Pa"] == pytest.approx(ratio_expected, rel=0.1)
    assert r["dP_gravity_Pa"] == pytest.approx(u["dP_static_Pa"], rel=0.05)             # measured −2.9 %


# ================================================================ water hammer
def _allievi(a: float, L: float, D: float, V0: float, H_R: float, H_out: float, tau, dt: float, n: int) -> np.ndarray:
    """Head rise at the valve from Allievi's chain equations (frictionless line, orifice valve), at t = k·dt."""
    T2 = round(2.0 * L / a / dt)
    dH0 = H_R - H_out
    c = V0 / math.sqrt(dH0)
    dH, V = np.zeros(n + 1), np.full(n + 1, V0)
    for k in range(1, n + 1):
        dH_b, V_b = (dH[k - T2], V[k - T2]) if k - T2 >= 0 else (0.0, V0)
        t = tau(k * dt)
        b = a / G * c * t
        rhs = dH0 - dH_b + a / G * V_b
        s = 0.5 * (-b + math.sqrt(b * b + 4.0 * rhs))
        dH[k], V[k] = s * s - dH0, c * t * s
    return dH


def test_water_hammer_frictionless_instant_square_wave():
    L, D, a, rho = 1000.0, 0.25, 1000.0, 1000.0
    A = math.pi * D * D / 4
    r = WH.water_hammer(L, D, rho, 1.0 * A, 10e5, 1e5, closure_time_s=0.0, closure_exponent=1.0, wave_speed_m_s=a,
                        darcy_friction_factor=0.0, simulation_time_s=8.0)
    dPj = rho * a * 1.0
    assert r["joukowsky_dP_Pa"] == pytest.approx(dPj, rel=1e-14)
    assert r["surge_above_initial_Pa"] == pytest.approx(dPj, rel=1e-12)
    assert r["P_min_Pa"] == pytest.approx(10e5 - dPj, rel=1e-12)
    t = np.array(r["valve_history"]["time_s"])
    P = np.array(r["valve_history"]["P_valve_Pa"])
    period = r["period_4L_over_a_s"]
    phase = np.mod(t, period)
    hi = (phase > 0.01) & (phase < period / 2 - 0.01)
    lo = (phase > period / 2 + 0.01) & (phase < period - 0.01)
    assert np.allclose(P[hi], 10e5 + dPj, rtol=1e-12) and np.allclose(P[lo], 10e5 - dPj, rtol=1e-12)
    assert period == pytest.approx(4.0, rel=1e-14)
    _dumps(r)


def test_water_hammer_slow_closure_matches_allievi():
    L, D, a, rho, N = 1000.0, 0.25, 1000.0, 1000.0, 40
    A = math.pi * D * D / 4
    for tc, Em in ((4.0, 1.0), (8.0, 1.0), (6.0, 2.0)):
        r = WH.water_hammer(L, D, rho, 1.0 * A, 10e5, 1e5, closure_time_s=tc, closure_exponent=Em, wave_speed_m_s=a,
                            darcy_friction_factor=0.0, n_reaches=N, simulation_time_s=9.0)
        tau, _ = WH._tau_fn(tc, Em, None)
        dt = r["dt_s"]
        n = len(r["valve_history"]["time_s"]) - 1
        dH = _allievi(a, L, D, 1.0, 10e5 / (rho * G), 1e5 / (rho * G), tau, dt, n)
        P_moc = np.array(r["valve_history"]["P_valve_Pa"])
        dev = np.max(np.abs(P_moc - (10e5 + rho * G * dH))) / r["joukowsky_dP_Pa"]
        assert dev < 1e-9                                      # measured ~1e-15
        assert r["surge_above_initial_Pa"] < r["joukowsky_dP_Pa"] and r["closure"].startswith("lento")


def test_water_hammer_closure_time_and_friction():
    L, D, a, rho = 1000.0, 0.25, 1000.0, 1000.0
    A = math.pi * D * D / 4
    s = [WH.water_hammer(L, D, rho, A, 10e5, 1e5, closure_time_s=tc, closure_exponent=1.0, wave_speed_m_s=a,
                         darcy_friction_factor=0.0)["surge_above_initial_Pa"] for tc in (1.0, 2.0, 4.0, 8.0)]
    assert s[0] == pytest.approx(rho * a, rel=1e-12) and s[1] == pytest.approx(rho * a, rel=1e-12)   # t_c ≤ 2L/a
    assert s[1] > s[2] > s[3]
    # friction: line packing raises the valve pressure above Joukowsky after an instant closure of a long line
    f = WH.water_hammer(5000.0, D, rho, A, 30e5, 1e5, closure_time_s=0.0, closure_exponent=1.0, wave_speed_m_s=a,
                        darcy_friction_factor=0.03, vapor_pressure_Pa=2339.0)
    assert f["P_initial_valve_Pa"] == pytest.approx(30e5 - 0.03 * 5000 / D * rho / 2, rel=1e-12)
    assert f["surge_above_initial_Pa"] > 1.1 * rho * a and f["evidence"]["verified"] and f["cavitation"] is False
    _dumps(f)


def test_wave_speed_korteweg_and_cavitation_flag():
    rigid = WH.wave_speed(998.0, 2.19e9)
    assert rigid["wave_speed_m_s"] == pytest.approx(math.sqrt(2.19e9 / 998.0), rel=1e-14)
    w = WH.wave_speed(998.0, 2.19e9, 0.25, 0.01, 2.0e11, pipe_anchoring="anchored_both_ends", poisson_ratio=0.3)
    assert w["restraint_factor"] == pytest.approx(0.91)
    assert w["wave_speed_m_s"] == pytest.approx(math.sqrt(1 / (1 / 2.19e9 + 0.91 * 25 / 2e11) / 998.0), rel=1e-14)
    stiff = WH.wave_speed(998.0, 2.19e9, 0.25, 0.01, 2.0e15, restraint_factor=1.0)
    assert stiff["wave_speed_m_s"] == pytest.approx(rigid["wave_speed_m_s"], rel=1e-4)
    assert stiff["wave_speed_m_s"] > w["wave_speed_m_s"]
    A = math.pi * 0.25 ** 2 / 4
    r = WH.water_hammer(1000.0, 0.25, 998.0, 2.0 * A, 5e5, 1e5, closure_time_s=0.0, closure_exponent=1.0,
                        fluid_bulk_modulus_Pa=2.19e9, pipe_elastic_modulus_Pa=2e11, wall_thickness_m=0.01,
                        pipe_anchoring="expansion_joints", poisson_ratio=0.3, viscosity_Pa_s=1e-3, roughness_m=4.57e-5,
                        vapor_pressure_Pa=2339.0)
    assert r["cavitation"] is True and any("cavitación" in x for x in r["warnings"])
    _dumps(r)


def test_water_hammer_refusals():
    A = math.pi * 0.25 ** 2 / 4
    with pytest.raises(ValueError, match="ley de cierre"):
        WH.water_hammer(1000.0, 0.25, 998.0, A, 10e5, 1e5, wave_speed_m_s=1000.0, darcy_friction_factor=0.0)
    with pytest.raises(ValueError, match="fricción"):
        WH.water_hammer(1000.0, 0.25, 998.0, A, 10e5, 1e5, closure_time_s=1.0, closure_exponent=1.0, wave_speed_m_s=1000.0)
    with pytest.raises(ValueError, match="celeridad"):
        WH.water_hammer(1000.0, 0.25, 998.0, A, 10e5, 1e5, closure_time_s=1.0, closure_exponent=1.0,
                        darcy_friction_factor=0.0)
    with pytest.raises(ValueError, match="no es posible"):
        WH.water_hammer(1000.0, 0.25, 998.0, A, 1.5e5, 1e5, closure_time_s=1.0, closure_exponent=1.0,
                        wave_speed_m_s=1000.0, darcy_friction_factor=0.5)


# ================================================================ pumps
def _scu_K_SI() -> float:
    from fluids import friction_factor
    D, L, ks, nu, Q = 20 / 12 * FT, 3884 * FT, 0.0005 * FT, 1.23e-5 * FT ** 2, FT ** 3
    V = Q / (math.pi * D * D / 4)
    f = friction_factor(Re=V * D / nu, eD=ks / D)
    return f * L / D * V * V / (2 * G) / Q ** 2


def test_pump_operating_point_maurer_example():
    K = _scu_K_SI()
    assert K * FT ** 6 / FT == pytest.approx(0.160, abs=5e-4)          # printed K = 0.160 ft/(ft³/s)²; ours 0.16034
    curve = {"flow_m3_s": [0.0, 5000 * GPM, 7850 * GPM], "head_m": [81 * FT, 60 * FT, 20 * FT]}
    r = pump_selection(curve, 1800.0, 998.0, {"static_head_m": 30 * FT, "resistance_s2_m5": K}, curve_fit="parabola",
                       efficiency=0.8)
    op = r["operating_point"]
    assert op["flow_m3_s"] / FT ** 3 == pytest.approx(12.051, rel=1e-4)    # measured 12.0510 (−2e-6)
    assert op["head_m"] / FT == pytest.approx(53.285, rel=1e-4)            # measured 53.2860 (+2e-5)
    a, c = r["pump_curve_fit"]["coefficients"]
    assert a / FT == pytest.approx(82.5, abs=0.05) and -c * FT ** 5 == pytest.approx(0.201, abs=5e-4)
    assert r["evidence"]["verified"]
    _dumps(r)


CURVE = {"flow_m3_s": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05], "head_m": [60.0, 59.0, 56.0, 51.0, 44.0, 35.0],
         "efficiency": [0.05, 0.45, 0.66, 0.75, 0.74, 0.65], "NPSHr_m": [1.5, 1.8, 2.3, 3.0, 4.0, 5.3]}


def test_affinity_laws_and_specific_speed():
    friction_only = {"static_head_m": 0.0, "resistance_s2_m5": 51.0 / 0.03 ** 2}
    r = pump_selection(CURVE, 2950.0, 998.0, friction_only, new_speed_rpm=2400.0)
    op, new = r["operating_point"], r["affinity"]["new_speed"]["operating_point"]
    rn = 2400.0 / 2950.0
    # pure-friction system: the operating point moves along a homologous line, Q ∝ n, H ∝ n², P ∝ n³
    assert new["flow_m3_s"] == pytest.approx(op["flow_m3_s"] * rn, rel=1e-9)
    assert new["head_m"] == pytest.approx(op["head_m"] * rn ** 2, rel=1e-9)
    assert new["shaft_power_W"] == pytest.approx(op["shaft_power_W"] * rn ** 3, rel=1e-8)
    assert r["evidence"]["specific_speed_invariance_rel"] < 1e-12 and r["evidence"]["verified"]
    ns = r["specific_speed"]
    assert ns["Ns_metric_rpm_m3s_m"] == pytest.approx(2950.0 * math.sqrt(r["best_efficiency_point"]["flow_m3_s"])
                                                      / r["best_efficiency_point"]["head_m"] ** 0.75, rel=1e-12)
    assert ns["Ns_US_rpm_gpm_ft"] / ns["Ns_metric_rpm_m3s_m"] == pytest.approx(math.sqrt(1 / GPM) / (1 / FT) ** 0.75)
    assert ns["pump_type"] == ["radial (centrífuga)"]
    # speed for a target flow, then back: the new speed delivers the target on the system
    t = pump_selection(CURVE, 2950.0, 998.0, {"static_head_m": 20.0, "resistance_s2_m5": 2.0e4}, target_flow_m3_s=0.025)
    n_req = t["affinity"]["speed_for_target_flow"]["speed_rpm"]
    back = pump_selection(CURVE, 2950.0, 998.0, {"static_head_m": 20.0, "resistance_s2_m5": 2.0e4}, new_speed_rpm=n_req)
    assert back["affinity"]["new_speed"]["operating_point"]["flow_m3_s"] == pytest.approx(0.025, rel=1e-8)
    trim = pump_selection(CURVE, 2950.0, 998.0, friction_only, impeller_diameter_m=0.25, new_impeller_diameter_m=0.23)
    assert trim["affinity"]["impeller_trim"]["operating_point"]["flow_m3_s"] == pytest.approx(
        op["flow_m3_s"] * 0.23 / 0.25, rel=1e-9)
    _dumps(r), _dumps(t), _dumps(trim)


def test_npsh_motor_and_efficiency_policy():
    sysc = {"static_head_m": 20.0, "resistance_s2_m5": 2.0e4}
    suction = {"vessel_pressure_Pa": 101325.0, "liquid_level_above_pump_m": 2.0, "losses_m": 0.5}
    r = pump_selection(CURVE, 2950.0, 998.0, sysc, suction=suction, vapor_pressure_Pa=4246.0, motor_efficiency=0.92)
    n = r["npsh"]
    assert n["NPSHa_m"] == pytest.approx((101325.0 - 4246.0) / (998.0 * G) + 2.0 - 0.5, rel=1e-12)
    assert n["margin_m"] == pytest.approx(n["NPSHa_m"] - r["operating_point"]["NPSHr_m"], rel=1e-12)
    m = r["motor"]
    assert m["electric_power_W"] == pytest.approx(m["shaft_power_W"] / 0.92, rel=1e-12)
    assert m["NEMA_motor_for_operating_point_W"] >= m["shaft_power_W"]
    assert m["NEMA_motor_non_overloading_W"] >= m["max_shaft_power_on_curve_W"]
    hot = pump_selection(CURVE, 2950.0, 998.0, sysc, suction={**suction, "liquid_level_above_pump_m": -6.0},
                         vapor_pressure_Pa=7.0e4)
    assert any("cavita" in w for w in hot["warnings"])
    no_eta = {"flow_m3_s": CURVE["flow_m3_s"], "head_m": CURVE["head_m"]}
    with pytest.raises(ValueError, match="eficiencia"):
        pump_selection(no_eta, 2950.0, 998.0, sysc)
    est = pump_selection(no_eta, 2950.0, 998.0, sysc, allow_efficiency_estimate=True)
    q_gpm = est["operating_point"]["flow_m3_s"] / GPM
    eta_ref = -0.316 + 0.24015 * math.log(q_gpm) - 0.01199 * math.log(q_gpm) ** 2   # Corripio (1982), as in fluids
    assert est["operating_point"]["efficiency"] == pytest.approx(eta_ref, rel=1e-6)
    assert "ESTIMADA" in est["operating_point"]["efficiency_source"] and any("ESTIMADA" in a for a in est["assumptions"])
    with pytest.raises(ValueError, match="no cruza"):
        pump_selection(CURVE, 2950.0, 998.0, {"static_head_m": 70.0, "resistance_s2_m5": 1.0})
    _dumps(r), _dumps(hot), _dumps(est)


def test_tools_and_wrappers_registered():
    from packages.agent import wrappers_momentum as WM
    assert [f.__name__ for f in TM.MOMENTUM_TOOLS] == ["adiabatic_gas_pipe", "two_phase_pipe_inclined", "water_hammer",
                                                        "pump_selection", "mechanical_energy_balance"]
    assert [w.name for w in WM.AGENT_WRAPPERS] == [f.__name__ for f in TM.MOMENTUM_TOOLS]
    out = json.loads(WM.water_hammer.call({"length_m": 500.0, "inner_diameter_m": 0.2, "density_kg_m3": 998.0,
                                           "initial_flow_m3_s": 0.03, "upstream_pressure_Pa": 5e5,
                                           "downstream_pressure_Pa": 1e5, "closure_time_s": 0.0, "closure_exponent": 1.0,
                                           "wave_speed_m_s": 1200.0, "darcy_friction_factor": 0.02}))
    assert out["evidence"]["verified"]
    gl = FD.gas_line(10e5, 300.0, 16.0, 1.0, 1.3, 1.1e-5, 0.05, 200.0, mass_flow_kg_s=0.5)
    r = TM.adiabatic_gas_pipe(10e5, 300.0, 0.05, 200.0, "static", mass_flow_kg_s=0.5, gamma=1.3,
                              molar_mass_kg_mol=0.016, viscosity_Pa_s=1.1e-5)
    assert r["isothermal_comparison"]["P2_Pa"] == pytest.approx(gl["P2_Pa"], rel=1e-12)
