"""DWSIM dynamic mode driven headless (packages.dwsim_client.dynamic): transients with a known answer.

Reference solutions are ANALYTICAL (derived here, 2026-11-14), not taken from a book:
  * gravity drain of a tank through a valve, A dh/dt = -k sqrt(h)  ->  sqrt(h) = sqrt(h0) - k t / (2 A)   (Torricelli form);
  * filling against the same valve, A dh/dt = q - k sqrt(h)  ->  t = -(2A/k^2) [k s + q ln(1 - k s / q)], s = sqrt(h);
  * perfectly mixed tank at constant holdup, step in the feed temperature or composition  ->  first order with tau = V / q.
k comes from the definition of the flow coefficient, Q[m3/h] = Kv sqrt(dP[bar] / (rho / rho0)) with rho0 = 999.1 kg/m3 (water
at 15 C, IEC 60534-2-1) and the hydrostatic head dP = rho g h with standard gravity g = 9.80665 m/s2, so k = Kv/3600 *
sqrt(rho0 * g / 1e5) in m^2.5/s. DWSIM uses g = 9.8 inside its tank; that 0.03 % is part of the measured deviation.
Water density: IAPWS-95 through CoolProp (independent of DWSIM). Bands asserted are the deviations measured on 2026-11-14.

Tests that need DWSIM are skipped when it cannot be loaded; the refusal tests run without it."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client import dynamic as DY  # noqa: E402
from packages.dwsim_client import tools_dynamic as TDY  # noqa: E402

G = 9.80665
RHO0_KV = 999.1
AREA, HEIGHT, KV, W_FEED, T_FEED = 0.02, 2.0, 10.0, 1.0, 298.15


def k_valve(kv: float) -> float:
    """m^2.5/s: volumetric outflow = k sqrt(level) for a liquid Kv valve under the hydrostatic head of the tank."""
    return kv / 3600.0 * math.sqrt(RHO0_KV * G / 1e5)


@pytest.fixture(scope="module")
def dwsim():
    ok, why = DY.available()
    if not ok:
        pytest.skip(f"DWSIM not available: {why}")


@pytest.fixture(scope="module")
def rho_water():
    from CoolProp.CoolProp import PropsSI
    return PropsSI("D", "T", T_FEED, "P", 101325.0, "Water")


def tank_case(h0, events=None, compounds=("Water",), package="SteamTables", feed=None, valve=None, extra_units=(),
              monitors=("TK.level_m", "out.mass_flow_kg_s", "feed.mass_flow_kg_s"), **kw):
    feed = dict({"name": "feed", "dynamic_spec": "flow", "temperature_K": T_FEED, "pressure_Pa": 101325.0,
                 "mass_flow_kg_s": W_FEED}, **(feed or {}))
    tank = {"kind": "tank", "name": "TK", "feed": "feed", "product": "out", "volume_m3": AREA * HEIGHT, "height_m": HEIGHT,
            "initial_level_m": h0}
    if h0 > 0:
        tank["initial_content"] = "inlet"
    v = dict({"kind": "valve", "name": "V", "feed": "out", "product": "prod", "kv": KV, "opening_pct": 100.0,
              "characteristic": "linear"}, **(valve or {}))
    return dict(compounds=list(compounds), property_package=package,
                streams=[feed, {"name": "out", "dynamic_spec": "pressure"},
                         {"name": "prod", "dynamic_spec": "pressure", "pressure_Pa": 101325.0}],
                units=[tank, v, *extra_units],
                monitors=[dict(zip(("object", "variable"), m.split(".", 1))) for m in monitors], events=events, **kw)


# ----------------------------------------------------------------------------- 1. level: drain and fill (analytical)
@pytest.fixture(scope="module")
def drain(dwsim):
    """Shutdown: the feed is cut at t = 0 and the tank drains by gravity through its valve (stopped before it is empty)."""
    return TDY.dwsim_dynamic_run(**tank_case(1.5, events=[{"time_s": 0.0, "object": "feed", "variable": "mass_flow_kg_s",
                                                           "value": 0.0}], step_s=0.1, duration_s=40.0))


def test_drain_level_follows_the_analytical_square_root_law(drain):
    r, k = drain, k_valve(KV)
    json.dumps(r)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert r["time_s"][0] == 0.0 and r["series"]["TK.level_m"][0] == pytest.approx(1.5, rel=1e-6)   # the initial condition
    dev = [abs(h - (math.sqrt(1.5) - k * t / (2 * AREA)) ** 2) / 1.5 for t, h in zip(r["time_s"], r["series"]["TK.level_m"])]
    assert max(dev) < 1.0e-3                       # measured 5.0e-4 of h0 at step 0.1 s (1.15e-3 at 0.2 s, 2.5e-3 at 0.4 s)
    # the last level is small (0.126 m), so the same absolute error is 0.59 % of it (measured -5.9e-3)
    assert r["final"]["TK.level_m"] == pytest.approx((math.sqrt(1.5) - k * 40.0 / (2 * AREA)) ** 2, rel=0.01)
    lv = r["series"]["TK.level_m"]
    assert all(b < a for a, b in zip(lv, lv[1:]))                                                    # monotone
    assert r["final"]["feed.mass_flow_kg_s"] == 0.0 and r["events_applied"][0]["effective_time_s"] == 0.0


def test_drain_mass_balance_and_step_independence(drain, rho_water):
    t, ev = drain["tanks"]["TK"], drain["evidence"]
    assert t["initial_mass_kg"] == pytest.approx(1.5 * AREA * rho_water, rel=1e-5)       # measured 4e-7 (IAPWS density)
    assert t["fed_kg"] == 0.0 and t["drawn_kg"] == pytest.approx(t["initial_mass_kg"] - t["final_mass_kg"], rel=1e-9)
    assert ev["mass_balance_max_rel_residual"] < 1e-9
    assert t["mass_balance_trapezoid_rel_residual"] < 5e-3      # trapezoid on the series, measured 2.9e-3 (first order)
    assert ev["step_halving_worst_mean_rel_diff"] < 6e-4        # measured 3.1e-4 (6.2e-4 at 0.2 s, 1.25e-3 at 0.4 s)
    assert ev["steady_state_solved"] and ev["integration_completed"]


@pytest.fixture(scope="module")
def fill(dwsim):
    """Startup: an empty tank is fed at a constant rate and its level rises to where the valve passes the feed."""
    return TDY.dwsim_dynamic_run(**tank_case(0.0, step_s=0.5, duration_s=150.0))


def test_fill_level_follows_the_analytical_solution(fill, rho_water):
    from scipy.optimize import brentq

    r, k, q = fill, k_valve(KV), W_FEED / rho_water
    json.dumps(r)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    h_ss = (q / k) ** 2

    def t_of_s(s):
        return -(2 * AREA / k ** 2) * (k * s + q * math.log(1.0 - k * s / q))

    dev = []
    for t, h in zip(r["time_s"][1:], r["series"]["TK.level_m"][1:]):
        s = brentq(lambda x: t_of_s(x) - t, 0.0, (q / k) * (1 - 1e-13), xtol=1e-14)
        dev.append(abs(h - s * s) / h_ss)
    assert max(dev) < 6e-3                         # measured 3.8e-3 of the steady level at 0.5 s (7.8e-3 at 1 s, 1.9e-3 at 0.25 s)
    assert h_ss == pytest.approx(1.3306, rel=1e-3)                                      # the analytical steady level
    assert 0.9 * h_ss < r["final"]["TK.level_m"] < h_ss                                 # 150 s is 2.8 time constants
    assert r["final"]["out.mass_flow_kg_s"] < W_FEED and r["warnings"] == []
    lv = r["series"]["TK.level_m"]
    assert all(b > a for a, b in zip(lv, lv[1:]))
    t = r["tanks"]["TK"]
    assert t["initial_mass_kg"] == pytest.approx(0.0, abs=1e-5) and r["evidence"]["mass_balance_max_rel_residual"] < 1e-9
    assert t["fed_kg"] == pytest.approx(W_FEED * 150.0, rel=1e-12) and t["final_mass_kg"] == pytest.approx(t["fed_kg"] - t["drawn_kg"])


# ----------------------------------------------------------------------------- 2. mixing tank: first-order responses
def test_feed_temperature_step_is_first_order_with_tau_V_over_q(dwsim, rho_water):
    h0 = (W_FEED / rho_water / k_valve(KV)) ** 2                   # analytical steady level, so the holdup is constant
    r = TDY.dwsim_dynamic_run(**tank_case(
        h0, events=[{"time_s": 10.0, "object": "feed", "variable": "temperature_K", "value": T_FEED + 10.0}],
        monitors=("out.temperature_K", "TK.level_m", "feed.temperature_K", "out.density_kg_m3"),
        step_s=0.5, duration_s=140.0, settle_s=20.0))
    json.dumps(r)
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert ev["settle_max_rel_drift"] < 1e-3                       # t = 0 is a steady state: the level given was the steady one
    tau = AREA * r["initial"]["TK.level_m"] * rho_water / W_FEED   # V / q
    assert tau == pytest.approx(26.5, rel=0.01)
    t0 = r["initial"]["out.temperature_K"]
    assert t0 == pytest.approx(T_FEED, abs=0.01)
    dev = [abs(T - (t0 + 10.0 * (1 - math.exp(-(t - 10.0) / tau)) if t > 10.0 else t0))
           for t, T in zip(r["time_s"], r["series"]["out.temperature_K"])]
    assert max(dev) < 0.05                         # K on a 10 K step: measured 0.033 at 0.5 s (0.068 at 1 s, 0.016 at 0.25 s)
    i63 = min(range(len(r["time_s"])), key=lambda i: abs(r["time_s"][i] - (10.0 + tau)))
    assert (r["series"]["out.temperature_K"][i63] - t0) / 10.0 == pytest.approx(1 - math.exp(-1), abs=0.01)   # measured 0.635
    assert r["final"]["out.temperature_K"] == pytest.approx(T_FEED + 10.0 * (1 - math.exp(-130.0 / tau)), abs=0.01)
    assert r["final"]["feed.temperature_K"] == pytest.approx(T_FEED + 10.0) and r["events_applied"][0]["effective_time_s"] == 10.0
    assert r["time_s"][0] == 0.0 and r["time_s"][-1] == 140.0 and r["settle_s"] == 20.0


def test_feed_composition_step_is_first_order_with_tau_V_over_q(dwsim):
    # 1.3044 m is the steady level for the liquid density of DWSIM's NRTL package (1007.4 kg/m3, read from a run on
    # 2026-11-14); the settle evidence below is what certifies that t = 0 is steady
    r = TDY.dwsim_dynamic_run(**tank_case(
        1.3044, compounds=("Water", "Ethanol"), package="NRTL", feed={"mole_fractions": {"Water": 1.0, "Ethanol": 0.0}},
        events=[{"time_s": 10.0, "object": "feed", "mole_fractions": {"Water": 0.999, "Ethanol": 0.001}}],
        monitors=("out.mass_fraction:Ethanol", "feed.mass_fraction:Ethanol", "TK.level_m", "out.density_kg_m3",
                  "out.mole_fraction:Ethanol"), step_s=0.5, duration_s=140.0, settle_s=20.0))
    json.dumps(r)
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert ev["settle_max_rel_drift"] < 1e-3
    w_in = r["final"]["feed.mass_fraction:Ethanol"]
    assert w_in == pytest.approx(0.001 * 46.069 / (0.001 * 46.069 + 0.999 * 18.015), rel=1e-4)      # molar masses, g/mol
    tau = AREA * r["initial"]["TK.level_m"] * r["initial"]["out.density_kg_m3"] / W_FEED            # V / q
    dev = [abs(w - (w_in * (1 - math.exp(-(t - 10.0) / tau)) if t > 10.0 else 0.0)) / w_in
           for t, w in zip(r["time_s"], r["series"]["out.mass_fraction:Ethanol"])]
    assert max(dev) < 5e-3                         # of the step: measured 3.1e-3 at 0.5 s (6.7e-3 at 1 s, 1.4e-3 at 0.25 s)
    assert r["initial"]["out.mass_fraction:Ethanol"] == 0.0
    assert r["final"]["out.mole_fraction:Ethanol"] == pytest.approx(0.001 * (1 - math.exp(-130.0 / tau)), rel=2e-3)
    # the holdup is not exactly constant: the liquid gets lighter with ethanol and the level rises 2.1 mm (0.16 %)
    assert r["maximum"]["TK.level_m"] - r["minimum"]["TK.level_m"] < 3e-3


# ----------------------------------------------------------------------------- 3. valves and control (consistency)
def test_pressure_driven_feed_valve_passes_the_kv_flow_and_reacts_to_an_opening_step(dwsim, rho_water):
    r = TDY.dwsim_dynamic_run(
        compounds=["Water"], property_package="SteamTables",
        streams=[{"name": "header", "dynamic_spec": "pressure", "temperature_K": T_FEED, "pressure_Pa": 201325.0,
                  "mass_flow_kg_s": 0.3},
                 {"name": "fill", "dynamic_spec": "pressure", "pressure_Pa": 101325.0}, {"name": "out", "dynamic_spec": "pressure"},
                 {"name": "prod", "dynamic_spec": "pressure", "pressure_Pa": 101325.0}],
        units=[{"kind": "valve", "name": "FV", "feed": "header", "product": "fill", "kv": 4.0, "opening_pct": 50.0,
                "characteristic": "linear"},
               {"kind": "tank", "name": "TK", "feed": "fill", "product": "out", "volume_m3": AREA * HEIGHT, "height_m": HEIGHT,
                "initial_level_m": 0.3, "initial_content": "inlet"},
               {"kind": "valve", "name": "V", "feed": "out", "product": "prod", "kv": KV, "opening_pct": 100.0,
                "characteristic": "linear"}],
        step_s=0.5, duration_s=30.0, events=[{"time_s": 15.0, "object": "FV", "variable": "opening_pct", "value": 100.0}],
        monitors=[{"object": "fill", "variable": "mass_flow_kg_s"}, {"object": "FV", "variable": "opening_pct"},
                  {"object": "TK", "variable": "level_m"}, {"object": "FV", "variable": "pressure_drop_Pa"}],
        check_step_halving=False)
    json.dumps(r)
    assert not r["evidence"]["verified"] and "independencia del paso" in r["evidence"]["reasons"][0]   # the check was skipped
    w_half = 0.5 * 4.0 / 3600.0 * math.sqrt(rho_water * RHO0_KV * 1.0)                 # Kv definition, 1 bar across
    i = r["time_s"].index(15.0)
    assert r["series"]["fill.mass_flow_kg_s"][i] == pytest.approx(w_half, rel=1e-4)    # measured 2e-5
    assert r["final"]["fill.mass_flow_kg_s"] == pytest.approx(2 * w_half, rel=1e-4)
    assert r["series"]["FV.opening_pct"][i] == 50.0 and r["final"]["FV.opening_pct"] == 100.0
    assert r["final"]["FV.pressure_drop_Pa"] == pytest.approx(1e5)
    assert r["evidence"]["mass_balance_max_rel_residual"] < 1e-9


def header_case(valve, events=None):
    """A 2 bar(a) header feeding the tank through an inlet valve of Kv 4 with 1 bar across it."""
    return TDY.dwsim_dynamic_run(
        compounds=["Water"], property_package="SteamTables",
        streams=[{"name": "header", "dynamic_spec": "pressure", "temperature_K": T_FEED, "pressure_Pa": 201325.0,
                  "mass_flow_kg_s": 0.1},
                 {"name": "fill", "dynamic_spec": "pressure", "pressure_Pa": 101325.0}, {"name": "out", "dynamic_spec": "pressure"},
                 {"name": "prod", "dynamic_spec": "pressure", "pressure_Pa": 101325.0}],
        units=[dict({"kind": "valve", "name": "FV", "feed": "header", "product": "fill", "kv": 4.0, "opening_pct": 50.0}, **valve),
               {"kind": "tank", "name": "TK", "feed": "fill", "product": "out", "volume_m3": AREA * HEIGHT, "height_m": HEIGHT,
                "initial_level_m": 0.3, "initial_content": "inlet"},
               {"kind": "valve", "name": "V", "feed": "out", "product": "prod", "kv": KV, "opening_pct": 100.0,
                "characteristic": "linear"}],
        step_s=0.5, duration_s=10.0, events=events, check_step_halving=False,
        monitors=[{"object": "fill", "variable": "mass_flow_kg_s"}, {"object": "fill", "variable": "pressure_Pa"}])


@pytest.mark.parametrize("valve, kv_fraction", [
    ({"characteristic": "linear"}, 0.5), ({"characteristic": "quick_opening"}, 0.5 ** 0.5),
    ({"characteristic": "equal_percentage", "rangeability": 50.0}, 50.0 ** (0.5 - 1.0)),
    ({"characteristic": "linear", "opening_pct": 0.0}, 0.0)])
def test_valve_characteristics_give_the_kv_flow(dwsim, rho_water, valve, kv_fraction):
    r = header_case(valve)
    json.dumps(r)
    expected = kv_fraction * 4.0 / 3600.0 * math.sqrt(rho_water * RHO0_KV * 1.0)        # Kv definition, 1 bar across
    assert r["final"]["fill.mass_flow_kg_s"] == pytest.approx(expected, rel=1e-4, abs=1e-12)   # measured 2.3e-5
    assert r["minimum"]["fill.mass_flow_kg_s"] == pytest.approx(r["maximum"]["fill.mass_flow_kg_s"])


def test_back_pressure_step_changes_the_valve_flow(dwsim, rho_water):
    r = header_case({"characteristic": "linear"}, events=[{"time_s": 5.0, "object": "fill", "variable": "pressure_Pa",
                                                           "value": 151325.0}])
    w = 0.5 * 4.0 / 3600.0 * math.sqrt(rho_water * RHO0_KV)
    i = r["time_s"].index(5.0)
    assert r["series"]["fill.mass_flow_kg_s"][i] == pytest.approx(w, rel=1e-4)          # the sample at the event is the old value
    assert r["series"]["fill.mass_flow_kg_s"][i + 1] == pytest.approx(w * math.sqrt(0.5), rel=1e-4)   # 0.5 bar across
    assert r["final"]["fill.pressure_Pa"] == pytest.approx(151325.0)


def test_initial_content_different_from_the_inlet_washes_out_first_order(dwsim, rho_water):
    from CoolProp.CoolProp import PropsSI

    h0 = (W_FEED / rho_water / k_valve(KV)) ** 2
    case = tank_case(h0, monitors=("out.temperature_K", "TK.level_m"), step_s=0.5, duration_s=60.0)
    case["units"][0]["initial_content"] = {"temperature_K": 320.0}
    r = TDY.dwsim_dynamic_run(**case)
    json.dumps(r)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    m0 = r["tanks"]["TK"]["initial_mass_kg"]
    assert m0 == pytest.approx(h0 * AREA * PropsSI("D", "T", 320.0, "P", 101325.0, "Water"), rel=1e-4)   # measured 1.1e-5
    assert r["initial"]["out.temperature_K"] == pytest.approx(320.0, abs=0.01)
    tau = m0 / W_FEED
    dev = [abs(T - (T_FEED + (320.0 - T_FEED) * math.exp(-t / tau))) for t, T in zip(r["time_s"], r["series"]["out.temperature_K"])]
    assert max(dev) < 0.09                         # K on a 21.85 K step: measured 0.060 at step 0.5 s
    # composition of the initial content (two compounds): the first outlet sample is the content given
    case = tank_case(1.3, compounds=("Water", "Ethanol"), package="NRTL", feed={"mole_fractions": {"Water": 1.0, "Ethanol": 0.0}},
                     monitors=("out.mole_fraction:Ethanol", "out.mass_fraction:Ethanol"), step_s=0.5, duration_s=20.0)
    case["units"][0]["initial_content"] = {"temperature_K": T_FEED, "mole_fractions": {"Water": 0.9, "Ethanol": 0.1}}
    r = TDY.dwsim_dynamic_run(**case)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert r["initial"]["out.mole_fraction:Ethanol"] == pytest.approx(0.1, rel=1e-9)
    assert r["initial"]["out.mass_fraction:Ethanol"] == pytest.approx(0.1 * 46.069 / (0.1 * 46.069 + 0.9 * 18.015), rel=1e-4)
    x = r["series"]["out.mole_fraction:Ethanol"]
    assert all(b < a for a, b in zip(x, x[1:])) and 0.03 < r["final"]["out.mole_fraction:Ethanol"] < 0.06


def test_pid_level_loop_reaches_the_setpoint_and_the_steady_valve_opening(dwsim, rho_water):
    pid = {"kind": "pid", "name": "LC", "controlled": {"object": "TK", "variable": "level_m"},
           "manipulated": {"object": "V", "variable": "opening_pct"}, "setpoint": 1.0, "kp": 2.0, "ki": 0.1, "kd": 0.0,
           "reverse_acting": True, "output_bias": 50.0, "output_span": 100.0, "output_min": 0.0, "output_max": 100.0}
    r = TDY.dwsim_dynamic_run(**tank_case(
        1.0, valve={"kv": 20.0, "opening_pct": 50.0}, extra_units=[pid],
        events=[{"time_s": 60.0, "object": "LC", "variable": "setpoint", "value": 1.2}],
        monitors=("TK.level_m", "V.opening_pct", "LC.setpoint", "LC.output"), step_s=0.25, duration_s=150.0, max_points=1000))
    json.dumps(r)
    ev = r["evidence"]
    assert ev["steady_state_solved"] and ev["integration_completed"] and ev["mass_balance_max_rel_residual"] < 1e-9
    i = r["time_s"].index(60.0)
    opening = lambda h: 100.0 * W_FEED / (rho_water * k_valve(20.0) * math.sqrt(h))    # noqa: E731 steady mass balance
    # integral action: no offset, and the valve sits where the steady mass balance puts it
    assert r["series"]["TK.level_m"][i] == pytest.approx(1.0, abs=0.003)            # measured 1.0019, still closing in
    assert r["series"]["V.opening_pct"][i] == pytest.approx(opening(1.0), rel=3e-3)   # measured 57.78 vs 57.68 %
    assert r["final"]["TK.level_m"] == pytest.approx(1.2, abs=0.001)                 # measured 1.2003
    assert r["final"]["V.opening_pct"] == pytest.approx(opening(1.2), rel=1.5e-3)     # measured 52.68 vs 52.65 %
    assert r["final"]["LC.setpoint"] == 1.2 and r["series"]["LC.setpoint"][i] == 1.0
    assert r["final"]["LC.output"] == pytest.approx(r["final"]["V.opening_pct"])
    assert 0.0 <= r["minimum"]["V.opening_pct"] and r["maximum"]["V.opening_pct"] <= 100.0
    assert ev["verified"], ev["reasons"]
    # time-averaged change at step / 2: measured 3.6e-3 (7.3e-3 at 0.5 s, 1.8e-3 at 0.125 s); pointwise it is 0.99 of the
    # range at the proportional kick, which DWSIM applies two steps after the setpoint event whatever the step
    assert ev["step_halving_worst_mean_rel_diff"] < 6e-3 and ev["step_halving_pointwise_max_rel_diff"]["V.opening_pct"] > 0.5


def test_two_tanks_in_series_start_up_and_shutdown_until_empty(dwsim, rho_water):
    streams = [{"name": "feed", "dynamic_spec": "flow", "temperature_K": T_FEED, "pressure_Pa": 101325.0, "mass_flow_kg_s": W_FEED},
               {"name": "o1", "dynamic_spec": "pressure"}, {"name": "f2", "dynamic_spec": "pressure", "pressure_Pa": 101325.0},
               {"name": "o2", "dynamic_spec": "pressure"}, {"name": "prod", "dynamic_spec": "pressure", "pressure_Pa": 101325.0}]
    units = [{"kind": "tank", "name": "TK1", "feed": "feed", "product": "o1", "volume_m3": AREA * HEIGHT, "height_m": HEIGHT,
              "initial_level_m": 0.0},
             {"kind": "valve", "name": "V1", "feed": "o1", "product": "f2", "kv": 10.0, "opening_pct": 100.0, "characteristic": "linear"},
             {"kind": "tank", "name": "TK2", "feed": "f2", "product": "o2", "volume_m3": AREA * HEIGHT, "height_m": HEIGHT,
              "initial_level_m": 0.0},
             {"kind": "valve", "name": "V2", "feed": "o2", "product": "prod", "kv": 12.0, "opening_pct": 100.0, "characteristic": "linear"}]
    mons = [{"object": "TK1", "variable": "level_m"}, {"object": "TK2", "variable": "level_m"}, {"object": "prod", "variable": "mass_flow_kg_s"}]
    r = TDY.dwsim_dynamic_run(["Water"], "SteamTables", streams, units, 1.0, 420.0, mons,
                              events=[{"time_s": 300.0, "object": "feed", "variable": "mass_flow_kg_s", "value": 0.0}], max_points=1000)
    json.dumps(r)
    i = r["time_s"].index(300.0)
    for tank, kv in (("TK1", 10.0), ("TK2", 12.0)):     # startup closes in on the analytical steady levels: -0.18 %, -0.65 %
        assert r["series"][f"{tank}.level_m"][i] == pytest.approx((W_FEED / rho_water / k_valve(kv)) ** 2, rel=0.01)
    assert r["series"]["prod.mass_flow_kg_s"][i] == pytest.approx(W_FEED, rel=5e-3)          # measured 0.9964 kg/s at 300 s
    # shutdown: both tanks run empty; DWSIM clips the inventory and the tool says so instead of certifying the run
    assert r["tanks"]["TK1"]["emptied"] and r["tanks"]["TK2"]["emptied"] and r["tanks"]["TK2"]["final_mass_kg"] == 0.0
    assert not r["evidence"]["verified"] and any("se vació" in x for x in r["evidence"]["reasons"])
    assert sum("se vació" in w for w in r["warnings"]) == 2
    assert 1e-6 < r["evidence"]["mass_balance_max_rel_residual"] < 1e-4   # the clipped last step: measured 4.4e-5


def test_steady_state_tools_still_work_after_a_dynamic_run(dwsim):
    TDY.dwsim_dynamic_run(**tank_case(0.5, step_s=1.0, duration_s=5.0, check_step_halving=False))
    try:
        from packages.dwsim_client import tools as T
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"DWSIM tool registry not importable: {ex}")
    fid = T.new_flowsheet("pump", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "w", temperature_K=298.15, pressure_Pa=1e5, mass_flow_kg_s=10.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "hp")
    T.add_unit(fid, "pump", "P-1", {"outlet_P_Pa": 11e5, "efficiency_pct": 70})
    T.connect(fid, "P-1", feeds=["w"], products=["hp"])
    assert T.solve(fid)["ok"]
    assert T.read_unit(fid, "P-1")["power_kW"] == pytest.approx(14.328, abs=0.01)    # same reference as test_tools_and_mcp


# ----------------------------------------------------------------------------- 4. refusals (no DWSIM needed)
@pytest.mark.parametrize("where, key, fragment", [
    ("tank", "volume_m3", "volumen del tanque"), ("tank", "height_m", "altura del tanque"),
    ("tank", "initial_level_m", "condición inicial"), ("tank", "initial_content", "con qué está lleno"),
    ("valve", "kv", "relación presión-caudal"), ("valve", "opening_pct", "apertura"), ("valve", "characteristic", "característica"),
    ("feed", "temperature_K", "temperatura de la alimentación"), ("feed", "mass_flow_kg_s", "caudal"),
    ("feed", "dynamic_spec", "especificación dinámica"), ("prod", "pressure_Pa", "presión aguas abajo"),
    ("run", "step_s", "paso de integración"), ("run", "duration_s", "horizonte"), ("run", "monitors", "variables a registrar")])
def test_missing_inputs_are_refused_by_name(where, key, fragment):
    case = tank_case(1.0, step_s=1.0, duration_s=10.0)
    target = {"tank": case["units"][0], "valve": case["units"][1], "feed": case["streams"][0], "prod": case["streams"][2],
              "run": case}[where]
    if where == "run":
        target[key] = None
    else:
        del target[key]
    with pytest.raises(ValueError) as e:
        TDY.dwsim_dynamic_run(**case)
    assert key in str(e.value) and fragment in str(e.value)


def test_inconsistent_specifications_are_refused():
    pid = {"kind": "pid", "name": "LC", "controlled": {"object": "TK", "variable": "level_m"},
           "manipulated": {"object": "V", "variable": "opening_pct"}, "setpoint": 1.0, "kp": 2.0, "ki": 0.1, "kd": 0.0,
           "reverse_acting": True, "output_bias": 50.0, "output_span": 100.0, "output_min": 0.0}
    with pytest.raises(ValueError, match="output_max"):            # a controller setting is never defaulted
        TDY.dwsim_dynamic_run(**tank_case(1.0, extra_units=[pid], step_s=1.0, duration_s=10.0))
    with pytest.raises(ValueError, match="rangeability"):
        TDY.dwsim_dynamic_run(**tank_case(1.0, valve={"characteristic": "equal_percentage"}, step_s=1.0, duration_s=10.0))
    case = tank_case(1.0, step_s=1.0, duration_s=10.0)
    case["streams"][1]["dynamic_spec"] = "flow"
    with pytest.raises(ValueError, match="sale del tanque"):       # DWSIM only integrates a tank with a pressure outlet
        TDY.dwsim_dynamic_run(**case)
    with pytest.raises(ValueError, match="desconocida"):
        TDY.dwsim_dynamic_run(**tank_case(1.0, monitors=("TK.temperature_K",), step_s=1.0, duration_s=10.0))
    with pytest.raises(ValueError, match="solo aplican a alimentaciones"):
        TDY.dwsim_dynamic_run(**tank_case(1.0, events=[{"time_s": 1.0, "object": "out", "variable": "temperature_K", "value": 300.0}],
                                          step_s=1.0, duration_s=10.0))
    with pytest.raises(ValueError, match="no se puede escribir"):
        TDY.dwsim_dynamic_run(**tank_case(1.0, events=[{"time_s": 1.0, "object": "TK", "variable": "level_m", "value": 0.5}],
                                          step_s=1.0, duration_s=10.0))
    with pytest.raises(ValueError, match="sumar 1"):
        TDY.dwsim_dynamic_run(**tank_case(1.0, compounds=("Water", "Ethanol"), package="NRTL",
                                          feed={"mole_fractions": {"Water": 0.9, "Ethanol": 0.2}}, step_s=1.0, duration_s=10.0))
    with pytest.raises(ValueError, match="límite"):
        TDY.dwsim_dynamic_run(**tank_case(1.0, step_s=0.001, duration_s=1000.0))


def test_tool_and_wrapper_lists_match():
    from packages.agent import wrappers_dynamic as WD

    assert [f.__name__ for f in TDY.DYNAMIC_TOOLS] == ["dwsim_dynamic_run"]
    assert [getattr(w, "name", getattr(w, "__name__", None)) for w in WD.AGENT_WRAPPERS] == ["dwsim_dynamic_run"]
