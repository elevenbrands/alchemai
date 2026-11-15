"""Crystallization and drying (bloque solids): textbook worked examples, analytical limits, conservation and refusals.

Sources, all read on 2026-11-14:
  [GK9]  FOSSEE Python Textbook Companion, Geankoplis "Transport Processes and Unit Operations", chapter 9 (data of each
         example and the printed results of the companion):
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter09.ipynb
  [GK12] same companion, chapter 12 (examples 12.11-1 and 12.11-2):
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter12.ipynb
  [MC]   FOSSEE Scilab Textbook Companion, McCabe-Smith-Harriott "Unit Operations of Chemical Engineering" (book 839), examples
         24.1, 24.2, 24.4, 27.1 and 27.6 (data and formulas; the expected values below are those formulas evaluated here):
         https://raw.githubusercontent.com/FOSSEE/Scilab-TBC-Uploads/master/839/CH24/EX24.4/Example_24_4.sce (and siblings)
  [GZ]   Formula sheet with Geankoplis eqs. 9.6-9/9.6-10 (h = 0.0204 G^0.8, h = 1.17 G^0.37):
         https://akbis.gantep.edu.tr/yonetim/upload/files/12667-5234.pdf
Geankoplis 12.12-1 (MSMPR from a screen analysis) could not be opened in any source: the inverse is validated against exact
MSMPR cut fractions instead (analytical)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client import tools_solids as TS  # noqa: E402
from packages.solids import drying as DR  # noqa: E402

P = 101325.0
LB, FT, BTU = 0.45359237, 0.3048, 1055.05585


def C(t):
    return t + 273.15


def F(t):
    return (t - 32.0) / 1.8 + 273.15


def ok(r):
    json.dumps(r)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    for key in ("warnings", "assumptions", "not_included", "method"):
        assert key in r
    return r


# ------------------------------------------------------------------------------------------------------ psychrometrics
def test_humidity_from_vapor_pressure_geankoplis_9_3_1():
    """[GK9] 26.7 °C, 101.3 kPa, pA = 2.76 kPa -> H = 0.01742, HS = 0.02226, HP = 78.3 %, HR = 78.9 %. Measured: +0.00 %,
    +0.60 %, -0.65 %, -0.64 % (CoolProp: psat 3.506 kPa instead of 3.50 and the enhancement factor)."""
    a = DR.humid_air_state(C(26.7), 101300.0, water_partial_pressure_Pa=2760.0)
    assert a["humidity_ratio_kg_kg"] == pytest.approx(0.01742, rel=2e-4)
    assert a["saturation_humidity_ratio_kg_kg"] == pytest.approx(0.02226, rel=0.008)
    assert 100 * a["percentage_saturation"] == pytest.approx(78.3, rel=0.008)
    assert 100 * a["relative_humidity"] == pytest.approx(78.9, rel=0.008)


def test_humidity_chart_points_geankoplis_9_3_2_to_9_3_4():
    """[GK9] chart readings. 9.3-2: 60 °C, dew point 26.7 °C -> H = 0.0225, cS = 1.047 kJ/kg K, vH = 0.977 m3/kg (measured
    -0.50 %, +0.32 %, +0.06 %). 9.3-3: saturation at 40.5 °C -> H = 0.0505 (+0.16 %); the 90 %-saturation point (42.5 °C,
    H = 0.050) lies on the same adiabatic line (wet bulb 40.6 °C vs 40.5) but reads 88.1 % saturation, 2 points below the
    chart reading. 9.3-4: 60 °C dry bulb, 29.5 °C wet bulb -> H = 0.0135 (+0.43 %)."""
    a = DR.humid_air_state(C(60.0), P, dew_point_T_K=C(26.7))
    assert a["humidity_ratio_kg_kg"] == pytest.approx(0.0225, rel=0.006)
    assert a["humid_heat_J_kgK"] == pytest.approx(1047.0, rel=0.004)
    assert a["humid_volume_m3_kg"] == pytest.approx(0.977, rel=0.001)
    assert DR.humid_air_state(C(40.5), P, relative_humidity=1.0)["humidity_ratio_kg_kg"] == pytest.approx(0.0505, rel=0.003)
    b = DR.humid_air_state(C(42.5), P, humidity_ratio_kg_kg=0.050)
    assert b["wet_bulb_T_K"] == pytest.approx(C(40.5), abs=0.15)
    assert b["percentage_saturation"] == pytest.approx(0.90, abs=0.025)
    assert DR.humid_air_state(C(60.0), P, wet_bulb_T_K=C(29.5))["humidity_ratio_kg_kg"] == pytest.approx(0.0135, rel=0.006)


def test_psychrometric_identities():
    sat = DR.humid_air_state(C(35.0), P, relative_humidity=1.0)
    assert sat["wet_bulb_T_K"] == pytest.approx(C(35.0), abs=1e-6) and sat["dew_point_K"] == pytest.approx(C(35.0), abs=1e-6)
    assert sat["percentage_saturation"] == pytest.approx(1.0, rel=1e-9)
    dry = DR.humid_air_state(C(35.0), P, humidity_ratio_kg_kg=0.0)
    assert dry["relative_humidity"] == 0.0 and dry["dew_point_K"] is None and dry["wet_bulb_T_K"] < C(35.0)
    # the five ways of giving the humidity describe the same state
    ref = DR.humid_air_state(C(50.0), P, humidity_ratio_kg_kg=0.02)
    for key, out in (("relative_humidity", "relative_humidity"), ("wet_bulb_T_K", "wet_bulb_T_K"),
                     ("dew_point_T_K", "dew_point_K"), ("water_partial_pressure_Pa", "water_partial_pressure_Pa")):
        again = DR.humid_air_state(C(50.0), P, **{key: ref[out]})
        assert again["humidity_ratio_kg_kg"] == pytest.approx(0.02, rel=1e-7)
    # wet bulb rises with humidity; adiabatic saturation balance closes with IAPWS liquid water
    wb = [DR.humid_air_state(C(80.0), P, humidity_ratio_kg_kg=w) for w in (0.0, 0.01, 0.03, 0.1)]
    assert all(b["wet_bulb_T_K"] > a["wet_bulb_T_K"] for a, b in zip(wb, wb[1:]))
    assert all(abs(s["adiabatic_saturation_residual"]) < 1e-5 for s in wb)
    assert DR.humid_air_state(C(150.0), P, humidity_ratio_kg_kg=0.05)["saturation_humidity_ratio_kg_kg"] is None
    r = ok(TS.dryer_batch({"T_K": C(60.0), "pressure_Pa": P, "relative_humidity": 0.2}))
    assert set(r) >= {"air", "evidence"} and "drying" not in r


# ----------------------------------------------------------------------------------------------------------- batch drying
AIR = {"T_K": C(65.6), "pressure_Pa": P, "humidity_ratio_kg_kg": 0.010}
RC = 1.51 / 3600.0                                # kg/(m2 s), [GK9] Fig. 9.5-1b


def test_constant_rate_prediction_geankoplis_9_6_3():
    """[GK9] pan 0.457 x 0.457 m, air 6.1 m/s parallel, 65.6 °C, H = 0.010 (chart: Tw = 28.9 °C, λw = 2433 kJ/kg) -> total
    evaporation 0.708 kg/h. Measured +0.11 % (CoolProp wet bulb 28.85 °C, λ 2432.5 kJ/kg, G 22748 kg/h m2)."""
    r = ok(TS.dryer_batch(AIR, constant_rate={"configuration": "parallel", "velocity_m_s": 6.1}))
    c = r["constant_rate"]
    assert c["rate_kg_m2s"] * 3600 * 0.457 ** 2 == pytest.approx(0.708, rel=0.003)
    assert c["surface_T_K"] == pytest.approx(C(28.9), abs=0.1) and c["latent_heat_J_kg"] == pytest.approx(2433e3, rel=5e-4)
    assert c["h_W_m2K"] == pytest.approx(0.0204 * (c["mass_velocity_kg_m2s"] * 3600) ** 0.8, rel=1e-12)      # [GZ]
    assert any("ESTIMADO" in a for a in r["assumptions"]) and any("correlación" in w for w in r["warnings"])
    perp = TS.dryer_batch(AIR, constant_rate={"configuration": "perpendicular", "velocity_m_s": 3.0})["constant_rate"]
    assert perp["h_W_m2K"] == pytest.approx(1.17 * (perp["mass_velocity_kg_m2s"] * 3600) ** 0.37, rel=1e-12)   # [GZ]
    out = TS.dryer_batch(AIR, constant_rate={"configuration": "parallel", "velocity_m_s": 0.2})
    assert any("fuera del intervalo" in w for w in out["warnings"])


def test_constant_rate_time_geankoplis_9_6_2_and_mccabe_24_1():
    """[GK9] 9.6-2: X 0.38 -> 0.25, Ls/A = 21.5 kg/m2, Rc = 1.51 kg/h m2 -> 1.85 h (measured +0.05 %).
    [MC] 24.1: filter cake 24 in square, 2 in thick, both faces, air 120 °F with 80 °F wet bulb, h = 1.94 Btu/ft2 h °F,
    λ = 1049 Btu/lb, 120 lb/ft3, X 0.20 -> 0.10 (Xc = 0.09): Rc = 1.94·40/1049 lb/ft2 h and t = 8/(Rc·8) h. Measured Rc
    +0.09 %, time -0.09 % (IAPWS latent heat 1048.0 instead of 1049 Btu/lb)."""
    r = ok(TS.dryer_batch(AIR, solid={"initial_moisture": 0.38, "final_moisture": 0.25, "critical_moisture": 0.195,
                                      "dry_solid_per_area_kg_m2": 21.5}, constant_rate={"rate_kg_m2s": RC}))
    assert r["drying"]["total_time_h"] == pytest.approx(1.85, rel=0.002) and r["drying"]["falling_rate_time_s"] == 0.0
    rc_book = 1.94 * (120 - 80) / 1049.0
    t_book = 120.0 * (2.0 ** 2 * 2 / 12.0) * (0.20 - 0.10) / (rc_book * 2 * 2.0 ** 2)
    m = ok(TS.dryer_batch({"T_K": F(120), "pressure_Pa": P, "wet_bulb_T_K": F(80)},
                          solid={"initial_moisture": 0.20, "final_moisture": 0.10, "critical_moisture": 0.09,
                                 "dry_solid_mass_kg": 80 * LB, "drying_area_m2": 8 * FT ** 2},
                          constant_rate={"configuration": "parallel", "velocity_m_s": 3.5 * FT, "h_W_m2K": 1.94 * 5.678263}))
    assert m["constant_rate"]["rate_kg_m2s"] / LB * FT ** 2 * 3600 == pytest.approx(rc_book, rel=0.002)
    assert m["drying"]["total_time_h"] == pytest.approx(t_book, rel=0.002) and t_book == pytest.approx(13.52, abs=0.01)
    assert m["solid"]["water_removed_kg"] == pytest.approx(8 * LB, rel=1e-12)


def test_falling_rate_geankoplis_9_7_1_and_9_7_2():
    """[GK9] 9.7-2: Ls = 399 kg, A = 18.58 m2, Rc = 1.51, Xc = 0.195 -> X = 0.040, straight line through the origin: falling
    period 4.39 h (measured +0.07 %). 9.7-1: the same with the measured rate curve; the companion (Simpson on 1/R) prints a
    total of 6.536 h. Measured: 6.602 h with the exact integral of the linearly interpolated curve (+1.0 %) and 6.687 h with
    the trapezoid on 1/R (+2.3 %): the result depends on how the six points are joined, and the engine reports both."""
    solid = {"initial_moisture": 0.38, "final_moisture": 0.04, "critical_moisture": 0.195, "equilibrium_moisture": 0.0,
             "dry_solid_mass_kg": 399.0, "drying_area_m2": 18.58}
    lin = ok(TS.dryer_batch(AIR, solid=solid, constant_rate={"rate_kg_m2s": RC}, falling_rate={"model": "linear"}))
    assert lin["drying"]["falling_rate_time_s"] / 3600 == pytest.approx(4.39, rel=0.002)
    assert lin["drying"]["constant_rate_time_s"] / 3600 == pytest.approx(2.63, rel=0.002)
    cap = TS.dryer_batch(AIR, solid=solid, constant_rate={"rate_kg_m2s": RC}, falling_rate={"model": "capillary"})
    assert cap["drying"]["total_time_s"] == lin["drying"]["total_time_s"]
    curve = {"model": "curve", "moisture": [0.195, 0.150, 0.100, 0.065, 0.050, 0.040],
             "rate_kg_m2s": [x / 3600 for x in (1.51, 1.21, 0.90, 0.71, 0.37, 0.27)]}
    cur = ok(TS.dryer_batch(AIR, solid={k: v for k, v in solid.items() if k != "critical_moisture"},
                            constant_rate={"rate_kg_m2s": RC}, falling_rate=curve))
    d = cur["drying"]
    assert d["total_time_h"] == pytest.approx(6.536, rel=0.012)
    assert 6.536 < d["total_time_h"] < (d["constant_rate_time_s"] + d["falling_time_trapezoid_on_reciprocal_s"]) / 3600
    assert (d["constant_rate_time_s"] + d["falling_time_trapezoid_on_reciprocal_s"]) / 3600 == pytest.approx(6.687, abs=0.002)
    assert cur["evidence"]["time_quadrature_rel_diff"] < 1e-9
    # a curve that is exactly the straight line reproduces the linear model
    xs = [0.04, 0.10, 0.195]
    same = TS.dryer_batch(AIR, solid=solid, constant_rate={"rate_kg_m2s": RC},
                          falling_rate={"model": "curve", "moisture": xs, "rate_kg_m2s": [RC * x / 0.195 for x in xs]})
    assert same["drying"]["total_time_s"] == pytest.approx(lin["drying"]["total_time_s"], rel=1e-12)


def test_diffusion_controlled_geankoplis_9_9_1_and_mccabe_24_2():
    """[GK9] 9.9-1: wood plank 25.4 mm, both faces, DL = 2.97e-6 m2/h, total moisture 0.29 -> 0.09, equilibrium 0.04 -> 30.8 h
    (measured +0.00 %; full series vs the one-term formula of the book 1e-6). [MC] 24.2: 25.4 mm, D = 8.3e-6 cm2/s,
    X 0.25 -> 0.05 -> t = 4 s^2/(pi^2 D) ln(8 X1/(pi^2 X)) (30.6 h; measured +0.00 %)."""
    r = ok(TS.dryer_batch(AIR, solid={"initial_moisture": 0.29, "final_moisture": 0.09, "critical_moisture": 0.29,
                                      "equilibrium_moisture": 0.04},
                          falling_rate={"model": "diffusion", "diffusivity_m2_s": 2.97e-6 / 3600, "thickness_m": 0.0254,
                                        "drying_faces": 2}))
    assert r["drying"]["total_time_h"] == pytest.approx(30.8, rel=0.002)
    assert r["drying"]["diffusion"]["one_term_time_s"] == pytest.approx(r["drying"]["total_time_s"], rel=1e-5)
    s_cm = 2.54 / 2
    t_mccabe = 4 * s_cm ** 2 / (math.pi ** 2 * 8.3e-6) * math.log(8 * 0.25 / (math.pi ** 2 * 0.05)) / 3600
    base = {"initial_moisture": 0.25, "final_moisture": 0.05, "critical_moisture": 0.25, "equilibrium_moisture": 0.0}
    fr = {"model": "diffusion", "diffusivity_m2_s": 8.3e-10, "thickness_m": 0.0254, "drying_faces": 2}
    m = ok(TS.dryer_batch(AIR, solid=base, falling_rate=fr))
    assert m["drying"]["total_time_h"] == pytest.approx(t_mccabe, rel=1e-5) and t_mccabe == pytest.approx(30.6, abs=0.05)
    # identities: time scales with thickness squared; one face = double thickness; short times follow 1 - 2 sqrt(Fo/pi)
    thick = TS.dryer_batch(AIR, solid=base, falling_rate={**fr, "thickness_m": 0.0508})
    one_face = TS.dryer_batch(AIR, solid=base, falling_rate={**fr, "drying_faces": 1})
    assert thick["drying"]["total_time_s"] == pytest.approx(4 * m["drying"]["total_time_s"], rel=1e-12)
    assert one_face["drying"]["total_time_s"] == pytest.approx(4 * m["drying"]["total_time_s"], rel=1e-12)
    early = TS.dryer_batch(AIR, solid={**base, "final_moisture": 0.24}, falling_rate=fr)
    fo = early["drying"]["diffusion"]["fourier_number"]
    assert 1 - 2 * math.sqrt(fo / math.pi) == pytest.approx(0.96, rel=1e-12)
    assert any("un término" in w for w in early["warnings"])
    assert DR._slab_series_only(0.05, 400) == pytest.approx(DR._slab_unaccomplished(0.05 - 1e-12), abs=1e-9)


def test_through_circulation_geankoplis_9_10_1():
    """[GK9] bed 50.8 mm of cylinders 6.35 x 25.4 mm, bulk 641 and particle 1602 kg/m3, air 121.1 °C, H = 0.04, 0.811 m/s;
    total moisture 1.00 -> 0.10, critical 0.50, equilibrium 0.01 -> 0.628 h (2260 s). Measured 0.622 h (-1.0 %): the book
    takes a round mean humidity 0.05 and chart values (Tw 47.2 °C, µ 2.15e-5); here the mean humidity (0.054), the humid
    heat, the viscosity (2.21e-5) and the wet bulb (47.09 °C) are computed."""
    r = ok(TS.dryer_batch({"T_K": C(121.1), "pressure_Pa": P, "humidity_ratio_kg_kg": 0.04},
                          solid={"initial_moisture": 1.0, "final_moisture": 0.1, "critical_moisture": 0.5,
                                 "equilibrium_moisture": 0.01, "bed_depth_m": 0.0508, "bulk_density_kg_m3": 641.0},
                          constant_rate={"configuration": "through_circulation", "superficial_velocity_m_s": 0.811,
                                         "particle": {"shape": "cylinder", "diameter_m": 0.00635, "length_m": 0.0254,
                                                      "void_fraction": 1 - 641.0 / 1602.0}},
                          falling_rate={"model": "linear"}))
    c = r["constant_rate"]
    assert r["drying"]["total_time_h"] == pytest.approx(0.628, rel=0.015)
    assert c["specific_area_m2_m3"] == pytest.approx(283.5, rel=1e-3)
    assert c["equivalent_diameter_m"] == pytest.approx(0.01347, rel=1e-3)
    assert c["reynolds"] > 350 and c["surface_T_K"] < c["outlet_air_T_constant_period_K"] < C(121.1)
    # the air cooled across the bed pays exactly for the evaporation (adiabatic bed)
    heat = c["dry_air_mass_velocity_kg_m2s"] * c["humid_heat_mean_J_kgK"] * (C(121.1) - c["outlet_air_T_constant_period_K"])
    assert heat == pytest.approx(c["rate_kg_m2s"] * c["latent_heat_J_kg"], rel=1e-12)
    with pytest.raises(ValueError, match="Reynolds"):
        TS.dryer_batch({"T_K": C(121.1), "pressure_Pa": P, "humidity_ratio_kg_kg": 0.04},
                       solid={"bed_depth_m": 0.05, "bulk_density_kg_m3": 641.0},
                       constant_rate={"configuration": "through_circulation", "superficial_velocity_m_s": 0.05,
                                      "particle": {"shape": "sphere", "diameter_m": 0.002, "void_fraction": 0.4}})


def test_batch_drying_identities():
    solid = {"initial_moisture": 0.5, "final_moisture": 0.05, "critical_moisture": 0.2, "equilibrium_moisture": 0.02,
             "dry_solid_per_area_kg_m2": 10.0}
    base = ok(TS.dryer_batch(AIR, solid=solid, constant_rate={"rate_kg_m2s": 4e-4}, falling_rate={"model": "linear"}))
    twice = TS.dryer_batch(AIR, solid={**solid, "dry_solid_per_area_kg_m2": 20.0}, constant_rate={"rate_kg_m2s": 4e-4},
                           falling_rate={"model": "linear"})
    assert twice["drying"]["total_time_s"] == pytest.approx(2 * base["drying"]["total_time_s"], rel=1e-12)
    drier = TS.dryer_batch(AIR, solid={**solid, "final_moisture": 0.03}, constant_rate={"rate_kg_m2s": 4e-4},
                           falling_rate={"model": "linear"})
    assert drier["drying"]["total_time_s"] > base["drying"]["total_time_s"]
    curve = base["drying_curve"]
    assert all(b["time_s"] > a["time_s"] and b["moisture"] < a["moisture"] for a, b in zip(curve, curve[1:]))
    assert curve[-1]["time_s"] == pytest.approx(base["drying"]["total_time_s"], rel=1e-12)
    # hotter air dries faster; a faster stream dries faster
    def rc(T, v):
        return TS.dryer_batch({"T_K": T, "pressure_Pa": P, "humidity_ratio_kg_kg": 0.01},
                              constant_rate={"configuration": "parallel", "velocity_m_s": v})["constant_rate"]["rate_kg_m2s"]
    assert rc(C(90), 3.0) > rc(C(60), 3.0) and rc(C(60), 5.0) > rc(C(60), 3.0)


def test_batch_drying_refusals():
    with pytest.raises(ValueError, match="humedad"):
        TS.dryer_batch({"T_K": 330.0, "pressure_Pa": P})
    with pytest.raises(ValueError, match="pressure_Pa"):
        TS.dryer_batch({"T_K": 330.0, "humidity_ratio_kg_kg": 0.01})
    with pytest.raises(ValueError, match="sobresaturado"):
        TS.dryer_batch({"T_K": 300.0, "pressure_Pa": P, "humidity_ratio_kg_kg": 0.05})
    solid = {"initial_moisture": 0.4, "final_moisture": 0.05, "dry_solid_per_area_kg_m2": 10.0}
    with pytest.raises(ValueError, match="critical_moisture"):
        TS.dryer_batch(AIR, solid=solid, constant_rate={"rate_kg_m2s": 4e-4}, falling_rate={"model": "linear"})
    with pytest.raises(ValueError, match="equilibrium_moisture"):
        TS.dryer_batch(AIR, solid={**solid, "critical_moisture": 0.2}, constant_rate={"rate_kg_m2s": 4e-4},
                       falling_rate={"model": "linear"})
    with pytest.raises(ValueError, match="model"):
        TS.dryer_batch(AIR, solid={**solid, "critical_moisture": 0.2, "equilibrium_moisture": 0.0},
                       constant_rate={"rate_kg_m2s": 4e-4})
    with pytest.raises(ValueError, match="constant_rate"):
        TS.dryer_batch(AIR, solid={**solid, "critical_moisture": 0.2, "equilibrium_moisture": 0.0},
                       falling_rate={"model": "linear"})
    with pytest.raises(ValueError, match="velocity_m_s"):
        TS.dryer_batch(AIR, constant_rate={"configuration": "parallel"})
    with pytest.raises(ValueError, match="diffusivity_m2_s"):
        TS.dryer_batch(AIR, solid={"initial_moisture": 0.2, "final_moisture": 0.05, "critical_moisture": 0.2,
                                   "equilibrium_moisture": 0.0}, falling_rate={"model": "diffusion", "thickness_m": 0.01})
    with pytest.raises(ValueError, match="no se\\s+extrapola"):
        TS.dryer_batch(AIR, solid=solid, constant_rate={"rate_kg_m2s": 4e-4},
                       falling_rate={"model": "curve", "moisture": [0.1, 0.2], "rate_kg_m2s": [2e-4, 4e-4]})
    with pytest.raises(ValueError, match="equilibrium_moisture"):
        TS.dryer_batch(AIR, solid={**solid, "critical_moisture": 0.2, "equilibrium_moisture": 0.05},
                       constant_rate={"rate_kg_m2s": 4e-4}, falling_rate={"model": "linear"})


# ------------------------------------------------------------------------------------------------------ continuous dryer
SOLID_910 = {"dry_solid_flow_kg_s": 453.6 / 3600, "inlet_moisture": 0.04, "outlet_moisture": 0.002, "inlet_T_K": C(26.7),
             "outlet_T_K": C(62.8), "dry_solid_cp_J_kgK": 1465.0}
AIR_910 = {"T_K": C(93.3), "pressure_Pa": P, "humidity_ratio_kg_kg": 0.010}


def test_continuous_dryer_heat_balance_geankoplis_9_10_2():
    """[GK9] countercurrent dryer, 453.6 kg dry solid/h, X 0.04 -> 0.002, solid 26.7 -> 62.8 °C (cpS 1.465 kJ/kg K), air in
    93.3 °C H = 0.010, out 37.8 °C, no heat loss -> the companion prints 1170.92 kg dry air/h and H = 0.0247. Measured
    1166.0 kg/h (-0.42 %) and 0.02478 (+0.33 %): real humid-air enthalpy instead of cS = 1.005 + 1.88 H and λ0 = 2501."""
    r = ok(TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 0.0, air_outlet_T_K=C(37.8)))
    assert r["dry_air_flow_kg_s"] * 3600 == pytest.approx(1170.92, rel=0.006)
    assert r["air_out"]["humidity_ratio_kg_kg"] == pytest.approx(0.0247, rel=0.005)
    assert r["water_evaporated_kg_s"] == pytest.approx(453.6 / 3600 * 0.038, rel=1e-12)
    ev = r["evidence"]
    assert ev["water_balance_rel_error"] < 1e-12 and ev["enthalpy_balance_rel_error"] < 1e-10
    assert ev["heat_duty_vs_iapws_steam_rel_diff"] < 1e-3
    # the textbook balance itself (cS = 1.005 + 1.88 H, λ0 = 2501 kJ/kg, cpA = 4.187), solved exactly, as a second reference
    hs = lambda T, X: 1.465 * T + X * 4.187 * T  # noqa: E731
    hg2, dHs, mv = (1.005 + 1.88 * 0.01) * 93.3 + 2501 * 0.01, 453.6 * (hs(62.8, 0.002) - hs(26.7, 0.04)), 453.6 * 0.038
    G_book = (dHs + mv * (1.88 * 37.8 + 2501)) / (hg2 - (1.005 + 1.88 * 0.01) * 37.8 - 0.01 * 2501)
    assert G_book == pytest.approx(1170.92, abs=0.05)                  # the companion's iteration converges to this
    assert r["dry_air_flow_kg_s"] * 3600 == pytest.approx(G_book, rel=0.006)


def test_rotary_dryer_mccabe_24_4():
    """[MC] 2800 lb/h dry solid, X 0.15 -> 0.005, solid 80 -> 125 °F (cp 0.52), air 260 °F H = 0.01, Nt = 1.5, 700 lb/ft2 h,
    countercurrent. Companion formulas: Twb 102 °F (chart), Tha = 137.3 °F, qT = 502 140 Btu/h, 16 532 lb/h dry air,
    D = 5.51 ft, L = 35.2 ft. Measured: wet bulb 102.8 °F, Tha 137.9 °F, qT -0.02 %, air +1.1 % (McCabe divides by
    cs·(1 + H) with a chart cs = 0.245), D +0.6 %, L -0.1 %."""
    Twb, Nt, lam = 102.0, 1.5, 1036.0
    Tha = (260 - Twb) / math.exp(Nt) + Twb
    q_lb = 0.52 * (125 - 80) + 0.15 * (Twb - 80) + 0.145 * lam + 0.005 * (125 - Twb) + 0.145 * 0.45 * (Tha - Twb)
    qT = q_lb * 2800
    mg = qT / (0.245 * (260 - Tha) * 1.01)
    Dia = math.sqrt(4 * (qT / (0.245 * (260 - Tha)) / 700) / math.pi)
    dTL = ((260 - Twb) - (Tha - Twb)) / math.log((260 - Twb) / (Tha - Twb))
    L = qT / (0.125 * math.pi * Dia * 700 ** 0.67 * dTL)
    solid = {"dry_solid_flow_kg_s": 2800 * LB / 3600, "inlet_moisture": 0.15, "outlet_moisture": 0.005, "inlet_T_K": F(80),
             "outlet_T_K": F(125), "dry_solid_cp_J_kgK": 0.52 * 4186.8}
    air = {"T_K": F(260), "pressure_Pa": P, "humidity_ratio_kg_kg": 0.01}
    r = ok(TS.dryer_continuous(air, solid, "countercurrent", 0.0, transfer_units=Nt,
                               sizing={"gas_mass_velocity_kg_m2s": 700 * DR.LB_FT2_H, "correlation": "mccabe_rotary"}))
    assert (r["air_in"]["wet_bulb_T_K"] - 273.15) * 1.8 + 32 == pytest.approx(Twb, abs=1.0)
    assert (r["air_out"]["T_K"] - 273.15) * 1.8 + 32 == pytest.approx(Tha, abs=1.0)
    assert r["heat_transferred_to_solid_W"] / BTU * 3600 == pytest.approx(qT, rel=0.002)
    assert r["dry_air_flow_kg_s"] / LB * 3600 == pytest.approx(mg, rel=0.015)
    assert r["sizing"]["diameter_m"] / FT == pytest.approx(Dia, rel=0.01) and Dia == pytest.approx(5.51, abs=0.01)
    assert r["sizing"]["length_m"] / FT == pytest.approx(L, rel=0.005) and L == pytest.approx(35.2, abs=0.05)
    assert r["transfer_units"] == pytest.approx(Nt, rel=1e-12)
    assert any("ESTIMADO" in a for a in r["assumptions"])
    # the user's own Ua gives a length that scales inversely with it
    s = r["sizing"]
    r2 = TS.dryer_continuous(air, solid, "countercurrent", 0.0, transfer_units=Nt,
                             sizing={"diameter_m": s["diameter_m"], "volumetric_heat_transfer_coefficient_W_m3K":
                                     2 * s["volumetric_heat_transfer_coefficient_W_m3K"]})
    assert r2["sizing"]["length_m"] == pytest.approx(s["length_m"] / 2, rel=1e-9)


def test_continuous_dryer_identities_and_refusals():
    base = ok(TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 0.0, air_outlet_T_K=C(37.8)))
    # giving the air flow returns the same outlet temperature (inverse problem)
    inv = ok(TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 0.0, dry_air_flow_kg_s=base["dry_air_flow_kg_s"]))
    assert inv["air_out"]["T_K"] == pytest.approx(C(37.8), abs=1e-6)
    # heat losses and a wetter feed need more air; the heater duty covers the heat given to the solid plus the stack loss
    loss = TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 5000.0, air_outlet_T_K=C(37.8))
    wet = TS.dryer_continuous(AIR_910, {**SOLID_910, "inlet_moisture": 0.06}, "countercurrent", 0.0, air_outlet_T_K=C(37.8))
    assert loss["dry_air_flow_kg_s"] > base["dry_air_flow_kg_s"] and wet["dry_air_flow_kg_s"] > base["dry_air_flow_kg_s"]
    heated = TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 0.0, air_outlet_T_K=C(37.8), air_heater_inlet_T_K=C(25.0))
    assert heated["air_heater_duty_W"] > heated["heat_transferred_to_solid_W"] and 0 < heated["thermal_efficiency"] < 1
    # cocurrent: same balances, but here the product (62.8 °C) would leave hotter than the air (37.8 °C): flagged, not hidden
    co = TS.dryer_continuous(AIR_910, SOLID_910, "cocurrent", 0.0, air_outlet_T_K=C(37.8))
    json.dumps(co)
    assert co["dry_air_flow_kg_s"] == pytest.approx(base["dry_air_flow_kg_s"], rel=1e-12)
    assert co["evidence"]["temperature_cross"] and not co["evidence"]["verified"]
    with pytest.raises(ValueError, match="heat_loss_W"):
        TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", None, air_outlet_T_K=C(37.8))
    with pytest.raises(ValueError, match="exactamente uno"):
        TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 0.0)
    with pytest.raises(ValueError, match="flow"):
        TS.dryer_continuous(AIR_910, SOLID_910, "crossflow", 0.0, air_outlet_T_K=C(37.8))
    with pytest.raises(ValueError, match="outlet_T_K"):
        TS.dryer_continuous(AIR_910, {k: v for k, v in SOLID_910.items() if k != "outlet_T_K"}, "countercurrent", 0.0,
                            air_outlet_T_K=C(37.8))
    with pytest.raises(ValueError, match="dry_solid_cp_J_kgK"):
        TS.dryer_continuous(AIR_910, {k: v for k, v in SOLID_910.items() if k != "dry_solid_cp_J_kgK"}, "countercurrent", 0.0,
                            air_outlet_T_K=C(37.8))
    with pytest.raises(ValueError, match="sobresaturado|saturar"):
        TS.dryer_continuous(AIR_910, {**SOLID_910, "inlet_moisture": 0.5}, "countercurrent", 0.0,
                            dry_air_flow_kg_s=base["dry_air_flow_kg_s"])
    with pytest.raises(ValueError, match="volumetric_heat_transfer_coefficient_W_m3K"):
        TS.dryer_continuous(AIR_910, SOLID_910, "countercurrent", 0.0, air_outlet_T_K=C(50.0), sizing={"diameter_m": 1.0})


# ------------------------------------------------------------------------------------------------- crystallization: yield
DECA = {"anhydrous_molar_mass_kg_mol": 0.1060, "solvent_moles_per_mole": 10, "solvent_molar_mass_kg_mol": 0.01802}
SOL_NA2CO3 = {"T_K": [293.0], "kg_solute_per_kg_solvent": [0.215]}


def test_yield_of_a_hydrate_geankoplis_12_11_1():
    """[GK12] 10 000 kg of 30 wt % Na2CO3 cooled to 293 K, Na2CO3·10H2O, solubility 21.5 kg/100 kg water (MW 106.0 and
    10 x 18.02). Companion: (a) no evaporation 6361.7 kg crystals and 3638.3 kg solution; (b) 3 % of the feed evaporated
    6636.2 and 3063.8. Measured: equal to 1e-5 (the textbook rounds to 6370 and 6630; the companion notes its own figures are
    the exact ones)."""
    a = ok(TS.crystallizer_yield(10000.0, 293.0, SOL_NA2CO3, {"mode": "none"}, feed_solute_mass_fraction=0.30, hydrate=DECA))
    b = ok(TS.crystallizer_yield(10000.0, 293.0, SOL_NA2CO3, {"mode": "fraction_of_feed", "fraction": 0.03},
                                 feed_solute_mass_fraction=0.30, hydrate=DECA))
    assert a["crystals_kg"] == pytest.approx(6361.7, abs=0.06) and a["mother_liquor_kg"] == pytest.approx(3638.3, abs=0.06)
    assert b["crystals_kg"] == pytest.approx(6636.2, abs=0.06) and b["mother_liquor_kg"] == pytest.approx(3063.8, abs=0.06)
    assert b["solvent_evaporated_kg"] == pytest.approx(300.0)
    for r in (a, b):
        assert max(r["evidence"][k] for k in ("total_mass_balance_rel_error", "solute_balance_rel_error",
                                              "solvent_balance_rel_error", "mother_liquor_saturation_rel_error")) < 1e-12
        assert r["mother_liquor"]["kg_solute_per_kg_solvent"] == pytest.approx(0.215, rel=1e-12)
    # water molar mass read from CoolProp (18.01527 instead of the book's 18.02) moves the yield by -3.2e-4
    w = TS.crystallizer_yield(10000.0, 293.0, SOL_NA2CO3, {"mode": "none"}, feed_solute_mass_fraction=0.30, solvent="water",
                              hydrate={"anhydrous_molar_mass_kg_mol": 0.1060, "solvent_moles_per_mole": 10})
    assert w["crystals_kg"] == pytest.approx(a["crystals_kg"], rel=5e-4)


def test_heat_balance_geankoplis_12_11_2_and_mccabe_27_1():
    """[GK12] 2268 kg at 327.6 K with 48.2 kg MgSO4/100 kg water cooled to 293.2 K (35.5 kg/100 kg), MgSO4·7H2O (120.368 +
    126.107), cp 2.93 kJ/kg K, heat of solution 13.31e3 kJ/kmol heptahydrate. Companion: 633.65 kg crystals, 1634.35 kg
    solution, q = -262 814.27 kJ. Measured: equal to 1e-6. (The printed textbook answer, 616.9 kg and -261 912 kJ, is quoted
    by secondary pages; the companion states its own figures are the correct ones and the balance here confirms it.)
    [MC] 27.1: 1000 kg of 30 % MgSO4, mother liquor 24.5 % anhydrous, 5 % of the feed water evaporated, MgSO4·7H2O (120.4,
    246.5) -> 261 kg of crystals with the companion's arithmetic (261.15). Measured: equal to 1e-6."""
    hepta = {"anhydrous_molar_mass_kg_mol": 0.120368, "solvent_moles_per_mole": 7, "solvent_molar_mass_kg_mol": 0.126107 / 7}
    r = ok(TS.crystallizer_yield(2268.0, 293.2, {"T_K": [293.2], "kg_solute_per_kg_solvent": [0.355]}, {"mode": "none"},
                                 feed_solute_per_kg_solvent=0.482, hydrate=hepta,
                                 energy={"feed_T_K": 327.6, "solution_cp_J_kgK": 2930.0, "heat_of_solution_J_mol": 13.31e3}))
    assert r["crystals_kg"] == pytest.approx(633.65, abs=0.006) and r["mother_liquor_kg"] == pytest.approx(1634.35, abs=0.006)
    assert r["heat"]["heat_to_remove_J"] / 1e3 == pytest.approx(262814.27, rel=1e-6)
    assert r["heat"]["heat_of_crystallization_J_kg"] == pytest.approx(13.31e6 / 246.475, rel=1e-9)
    m = ok(TS.crystallizer_yield(1000.0, 288.71, {"T_K": [288.71], "kg_solute_per_kg_solvent": [0.245 / 0.755]},
                                 {"mode": "fraction_of_feed_solvent", "fraction": 0.05}, feed_solute_mass_fraction=0.30,
                                 hydrate={"anhydrous_molar_mass_kg_mol": 0.1204, "solvent_moles_per_mole": 7,
                                          "solvent_molar_mass_kg_mol": (0.2465 - 0.1204) / 7}))
    m2_in = 0.30 * 1000 * 246.5 / 120.4
    expected = m2_in - (24.5 * 246.5 / 120.4) / (100 - 24.5 * 246.5 / 120.4) * (1000 - 35.0 - m2_in)
    assert m["crystals_kg"] == pytest.approx(expected, rel=1e-9) and expected == pytest.approx(261.0, abs=0.2)


def test_crystallizer_yield_identities():
    table = {"T_K": [283.15, 293.15, 313.15, 333.15], "kg_solute_per_kg_solvent": [0.21, 0.32, 0.64, 1.10]}
    kw = dict(feed_solute_per_kg_solvent=0.9, crystal_solute_mass_fraction=1.0)
    cold = ok(TS.crystallizer_yield(100.0, 283.15, table, {"mode": "none"}, **kw))
    warm = ok(TS.crystallizer_yield(100.0, 303.15, table, {"mode": "none"}, **kw))
    assert cold["crystals_kg"] > warm["crystals_kg"] > 0                       # colder -> more crystals
    assert cold["crystals_kg"] == pytest.approx(100 / 1.9 * (0.9 - 0.21), rel=1e-12)   # anhydrous: solvent·(c0 − c*)
    assert warm["solubility_at_final_T"]["interpolation"] and any("INTERPOLADA" in a for a in warm["assumptions"])
    assert 0.32 < warm["solubility_at_final_T"]["kg_solute_per_kg_solvent"] < 0.64
    evap = TS.crystallizer_yield(100.0, 283.15, table, {"mode": "mass", "solvent_evaporated_kg": 10.0}, **kw)
    assert evap["crystals_kg"] == pytest.approx(cold["crystals_kg"] + 10.0 * 0.21, rel=1e-12)   # each kg evaporated drops s
    unsat = ok(TS.crystallizer_yield(100.0, 333.15, table, {"mode": "none"}, **kw))
    assert unsat["crystals_kg"] == 0.0 and not unsat["mother_liquor"]["saturated"] and unsat["warnings"]
    # adiabatic (vacuum): the evaporation closes the enthalpy balance, and feeding it back as a known mass gives q = 0
    energy = {"feed_T_K": 333.15, "solution_cp_J_kgK": 3000.0, "heat_of_crystallization_J_kg": 8.0e4}
    ad = ok(TS.crystallizer_yield(100.0, 293.15, table, {"mode": "adiabatic"}, solvent="water", energy=energy, **kw))
    assert ad["heat"]["heat_to_remove_J"] == 0.0 and ad["solvent_evaporated_kg"] > 0
    back = TS.crystallizer_yield(100.0, 293.15, table, {"mode": "mass", "solvent_evaporated_kg": ad["solvent_evaporated_kg"]},
                                 solvent="water", energy=energy, **kw)
    assert abs(back["heat"]["heat_to_remove_J"]) < 1e-6 * back["heat"]["sensible_heat_of_feed_J"]
    assert back["crystals_kg"] == pytest.approx(ad["crystals_kg"], rel=1e-12)
    cooled = TS.crystallizer_yield(100.0, 293.15, table, {"mode": "none"}, energy=energy, **kw)
    assert cooled["heat"]["heat_to_remove_J"] == pytest.approx(
        100 * 3000.0 * 40.0 + cooled["crystals_kg"] * 8.0e4, rel=1e-12)


ANH = {"feed_solute_mass_fraction": 0.4, "crystal_solute_mass_fraction": 1.0}


def test_crystallizer_yield_refusals():
    table = {"T_K": [283.15, 333.15], "kg_solute_per_kg_solvent": [0.21, 1.10]}
    with pytest.raises(ValueError, match="no se\\s+extrapola"):
        TS.crystallizer_yield(100.0, 273.15, table, {"mode": "none"}, **ANH)
    with pytest.raises(ValueError, match="qué cristaliza"):
        TS.crystallizer_yield(100.0, 283.15, table, {"mode": "none"}, feed_solute_mass_fraction=0.4)
    with pytest.raises(ValueError, match="concentración de la alimentación"):
        TS.crystallizer_yield(100.0, 283.15, table, {"mode": "none"}, crystal_solute_mass_fraction=1.0)
    with pytest.raises(ValueError, match="mode"):
        TS.crystallizer_yield(100.0, 283.15, table, {}, **ANH)
    with pytest.raises(ValueError, match="solvent_molar_mass_kg_mol"):
        TS.crystallizer_yield(100.0, 283.15, table, {"mode": "none"}, feed_solute_mass_fraction=0.4,
                              hydrate={"anhydrous_molar_mass_kg_mol": 0.12, "solvent_moles_per_mole": 7})
    with pytest.raises(ValueError, match="heat_of_crystallization_J_kg"):
        TS.crystallizer_yield(100.0, 283.15, table, {"mode": "none"}, **ANH,
                              energy={"feed_T_K": 330.0, "solution_cp_J_kgK": 3000.0})
    with pytest.raises(ValueError, match="energy"):
        TS.crystallizer_yield(100.0, 283.15, table, {"mode": "adiabatic"}, **ANH)
    with pytest.raises(ValueError, match="solvent_latent_heat_J_kg"):
        TS.crystallizer_yield(100.0, 283.15, table, {"mode": "adiabatic"}, **ANH,
                              energy={"feed_T_K": 330.0, "solution_cp_J_kgK": 3000.0, "heat_of_crystallization_J_kg": 5e4})
    with pytest.raises(ValueError, match="no es más rico"):
        TS.crystallizer_yield(100.0, 333.15, table, {"mode": "none"}, feed_solute_mass_fraction=0.4,
                              crystal_solute_mass_fraction=0.5)


# ------------------------------------------------------------------------------------------------------------------ MSMPR
def test_msmpr_analytical_identities():
    """Analytical: n = n0 exp(-L/Gτ); µk = n0 k! (Gτ)^(k+1); dominant size 3Gτ; mass-weighted CV 50 %; mass median
    3.672 Gτ; MT = 6 kv ρ n0 (Gτ)^4 (Randolph-Larson)."""
    G, B0, tau, rho, kv = 5e-8, 2e6, 7200.0, 1770.0, 0.6
    r = ok(TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=tau, growth_rate_m_s=G, nucleation_rate_m3_s=B0,
                                        suspension_volume_m3=4.0))
    Gt, n0 = G * tau, B0 / G
    assert r["nuclei_population_density_m4"] == pytest.approx(n0, rel=1e-12)
    for k in range(5):
        assert r["moments"][f"mu{k}"] == pytest.approx(n0 * math.factorial(k) * Gt ** (k + 1), rel=1e-12)
    assert r["magma_density_kg_m3"] == pytest.approx(rho * kv * r["moments"]["mu3"], rel=1e-12)
    assert r["sizes"]["dominant_mass_m"] == pytest.approx(3 * Gt, rel=1e-12)
    assert r["sizes"]["mass_median_m"] == pytest.approx(3.67206 * Gt, rel=1e-5)
    assert r["coefficient_of_variation"]["mass_weighted"] == pytest.approx(0.5, abs=1e-12)
    assert r["coefficient_of_variation"]["number_weighted"] == pytest.approx(1.0, abs=1e-12)
    assert r["production_rate_kg_s"] == pytest.approx(r["magma_density_kg_m3"] * 4.0 / tau, rel=1e-12)
    # the dominant size is the maximum of the mass density L^3 n(L); the median splits the mass in two
    mass_density = lambda L: L ** 3 * math.exp(-L / Gt)  # noqa: E731
    assert mass_density(3 * Gt) > max(mass_density(2.99 * Gt), mass_density(3.01 * Gt))
    med = TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=tau, growth_rate_m_s=G, nucleation_rate_m3_s=B0,
                                       sizes_m=[r["sizes"]["mass_median_m"], 50 * Gt])["distribution"]
    assert med[0]["cumulative_mass_undersize"] == pytest.approx(0.5, abs=1e-10)
    assert med[1]["cumulative_mass_undersize"] == pytest.approx(1.0, abs=1e-12)
    # the three ways of specifying the crystallizer are the same crystallizer
    MT = r["magma_density_kg_m3"]
    a = TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=tau, growth_rate_m_s=G, magma_density_kg_m3=MT)
    b = TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=tau, nucleation_rate_m3_s=B0, magma_density_kg_m3=MT)
    assert a["nucleation_rate_m3_s"] == pytest.approx(B0, rel=1e-12) and b["growth_rate_m_s"] == pytest.approx(G, rel=1e-12)
    law = {"k": B0 / (G ** 1.5 * MT), "growth_exponent": 1.5, "magma_exponent": 1.0}
    c = ok(TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=tau, magma_density_kg_m3=MT, nucleation_law=law))
    assert c["growth_rate_m_s"] == pytest.approx(G, rel=1e-10) and c["nucleation_rate_m3_s"] == pytest.approx(B0, rel=1e-10)
    # with B0 = k G^i and i > 1, doubling the residence time lowers G and enlarges the crystals (i = 1.5: L ~ τ^(1/9))
    law0 = {"k": law["k"] * MT, "growth_exponent": 1.5, "magma_exponent": 0.0}
    d = TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=2 * tau, magma_density_kg_m3=MT, nucleation_law=law0)
    assert d["sizes"]["dominant_mass_m"] / (3 * Gt) == pytest.approx(2 ** ((1.5 - 1) / (1.5 + 3)), rel=1e-10)


def test_msmpr_design_mccabe_27_6():
    """[MC] 27.6: G = 0.0018 ft/h, predominant size 20 mesh (0.00273 ft), 10 000 lb/h of crystals (105 lb/ft3) in
    44 520/82.5 ft3/h of mother liquor: drawdown time τ = Lpr/3G, liquid volume τ·Q and B0 = 9C/(2 ρc Vc Lpr^3) (eq. 27.44).
    Evaluating those formulas gives τ = 0.5056 h, Vc = 272.8 ft3, B0 = 7.72e7 nuclei/ft3 h; the engine reaches the same B0
    through n0 and the fourth moment (measured: equal to 1e-12). Basis: per volume of clear liquor, as in the book."""
    G_fps, L_pr, C_lb, rho_c, Q = 0.0018, 0.00273, 10000.0, 105.0, 44520.0 / 82.5
    tau_h = L_pr / (3 * G_fps)
    Vc = tau_h * Q
    B0_book = 9 * C_lb / (2 * rho_c * Vc * L_pr ** 3)
    assert tau_h == pytest.approx(0.5056, abs=1e-4) and Vc == pytest.approx(272.8, abs=0.05)
    assert B0_book == pytest.approx(7.72e7, rel=2e-3)
    r = ok(TS.crystal_size_distribution("msmpr", rho_c * LB / FT ** 3, 1.0, residence_time_s=tau_h * 3600,
                                        growth_rate_m_s=G_fps * FT / 3600, magma_density_kg_m3=C_lb * LB / (Q * FT ** 3),
                                        suspension_volume_m3=Vc * FT ** 3))
    assert r["nucleation_rate_m3_s"] * FT ** 3 * 3600 == pytest.approx(B0_book, rel=1e-12)
    assert r["sizes"]["dominant_mass_m"] / FT == pytest.approx(L_pr, rel=1e-12)
    assert r["production_rate_kg_s"] / LB * 3600 == pytest.approx(C_lb, rel=1e-12)
    assert r["nuclei_population_density_m4"] * FT ** 4 == pytest.approx(B0_book / G_fps, rel=1e-12)       # eq. 27.40


TYLER_MM = [0.074, 0.104, 0.147, 0.208, 0.295, 0.417, 0.589, 0.833, 1.168, 1.651]      # sqrt(2) series, 200 to 10 mesh


def _exact_sieve(Gt, edges):
    from scipy.special import gammainc
    cum = [float(gammainc(4.0, e / Gt)) for e in edges]
    w = [cum[0]] + [b - a for a, b in zip(cum, cum[1:])] + [1 - cum[-1]]
    return {"lower_m": [None] + edges, "upper_m": edges + [None], "mass_fraction": w}


def test_msmpr_from_sieve_recovers_exact_crystallizer():
    """Analytical round trip (Geankoplis 12.12-1 could not be read in an open source): exact MSMPR cut fractions on a sqrt(2)
    sieve series -> the textbook regression recovers G +0.53 %, n0 -0.57 %, B0 -0.05 % (the bias of the mid-size
    approximation, which the engine reports), and the exact-cut fit recovers them to 1e-8."""
    Gt, tau, rho, kv, n0 = 2e-4, 3600.0, 1500.0, 0.6, 1e12
    MT = 6 * kv * rho * n0 * Gt ** 4
    edges = [x * 1e-3 for x in TYLER_MM]
    r = ok(TS.crystal_size_distribution("msmpr_from_sieve", rho, kv, residence_time_s=tau, magma_density_kg_m3=MT,
                                        sieve=_exact_sieve(Gt, edges)))
    assert r["growth_rate_m_s"] == pytest.approx(Gt / tau, rel=0.007)
    assert r["nuclei_population_density_m4"] == pytest.approx(n0, rel=0.008)
    assert r["nucleation_rate_m3_s"] == pytest.approx(n0 * Gt / tau, rel=0.002)
    assert r["regression"]["r_squared"] > 0.9999 and r["regression"]["points"] == 9 and len(r["skipped_fractions"]) == 2
    assert r["mid_size_bias"]["growth_rate_rel"] == pytest.approx(r["growth_rate_m_s"] * tau / Gt - 1, abs=3e-4)
    assert r["exact_cut_fit"]["growth_rate_m_s"] == pytest.approx(Gt / tau, rel=1e-7)
    assert r["exact_cut_fit"]["nuclei_population_density_m4"] == pytest.approx(n0, rel=1e-6)
    assert r["magma_density_ratio_fit_to_given"] == pytest.approx(1.0, abs=0.03)
    # forward and inverse are consistent: the fitted parameters reproduce the given magma density through the fourth moment
    fwd = TS.crystal_size_distribution("msmpr", rho, kv, residence_time_s=tau, growth_rate_m_s=r["growth_rate_m_s"],
                                       nucleation_rate_m3_s=r["nucleation_rate_m3_s"])
    assert fwd["magma_density_kg_m3"] == pytest.approx(r["magma_density_implied_by_fit_kg_m3"], rel=1e-10)
    # narrower cuts (4th-root-of-2 series) shrink the bias; a non-exponential product is flagged
    fine_edges = [0.074e-3 * 2 ** (i / 4) for i in range(19)]
    fine = TS.crystal_size_distribution("msmpr_from_sieve", rho, kv, residence_time_s=tau, magma_density_kg_m3=MT,
                                        sieve=_exact_sieve(Gt, fine_edges))
    assert abs(fine["growth_rate_m_s"] * tau / Gt - 1) < abs(r["growth_rate_m_s"] * tau / Gt - 1)
    classified = {"lower_m": edges[:-1], "upper_m": edges[1:],
                  "mass_fraction": [0.01, 0.02, 0.03, 0.14, 0.50, 0.25, 0.03, 0.015, 0.005]}
    bad = TS.crystal_size_distribution("msmpr_from_sieve", rho, kv, residence_time_s=tau, magma_density_kg_m3=MT,
                                       sieve=classified)
    json.dumps(bad)
    # measured: R² is still 0.971 (n carries L^3 in the denominator), so the flag is the mass-fraction misfit, not R²
    assert bad["regression"]["r_squared"] < 0.99 and bad["exact_cut_fit"]["rms_mass_fraction_residual"] > 0.05
    assert any("MSMPR" in w for w in bad["warnings"]) and any("balance de masa" in w for w in bad["warnings"])
    assert not any("MSMPR ideal" in w for w in r["warnings"])


def test_crystal_size_distribution_refusals():
    with pytest.raises(ValueError, match="mode"):
        TS.crystal_size_distribution("fluidized", 1500.0, 0.5)
    with pytest.raises(ValueError, match="residence_time_s"):
        TS.crystal_size_distribution("msmpr", 1500.0, 0.5, growth_rate_m_s=1e-8, nucleation_rate_m3_s=1e6)
    with pytest.raises(ValueError, match="da dos de"):
        TS.crystal_size_distribution("msmpr", 1500.0, 0.5, residence_time_s=3600.0, growth_rate_m_s=1e-8)
    with pytest.raises(ValueError, match="growth_exponent"):
        TS.crystal_size_distribution("msmpr", 1500.0, 0.5, residence_time_s=3600.0, magma_density_kg_m3=100.0,
                                     nucleation_law={"k": 1e20})
    with pytest.raises(ValueError, match="sieve"):
        TS.crystal_size_distribution("msmpr_from_sieve", 1500.0, 0.5, residence_time_s=3600.0, magma_density_kg_m3=100.0)
    with pytest.raises(ValueError, match="suman"):
        TS.crystal_size_distribution("msmpr_from_sieve", 1500.0, 0.5, residence_time_s=3600.0, magma_density_kg_m3=100.0,
                                     sieve={"lower_m": [1e-4, 2e-4, 3e-4], "upper_m": [2e-4, 3e-4, 4e-4],
                                            "mass_fraction": [10, 50, 40]})
    with pytest.raises(ValueError, match="al menos 3"):
        TS.crystal_size_distribution("msmpr_from_sieve", 1500.0, 0.5, residence_time_s=3600.0, magma_density_kg_m3=100.0,
                                     sieve={"lower_m": [None, 2e-4, 3e-4], "upper_m": [2e-4, 3e-4, None],
                                            "mass_fraction": [0.3, 0.4, 0.3]})
    with pytest.raises(ValueError, match="batch"):
        TS.crystal_size_distribution("batch_seeded", 1500.0, 0.5)


# --------------------------------------------------------------------------------------------------- seeded batch cooling
SOL_TABLE = {"T_K": [293.15, 303.15, 313.15, 323.15], "kg_solute_per_kg_solvent": [0.10, 0.14, 0.20, 0.28]}
RHO, KV = 1800.0, 0.5


def _batch(**over):
    b = {"solubility": SOL_TABLE, "initial_concentration_kg_kg": 0.28,
         "cooling": {"time_s": [0, 3600.0], "T_K": [313.15, 293.15]},
         "growth": {"k_m_s": 2e-8, "order": 0}, "nucleation": None, "seed_mass_kg_per_kg_solvent": 0.002, "seed_size_m": 50e-6}
    b.update(over)
    return b


def test_batch_seeded_cooling_analytical_limits():
    """Analytical limits of the moment model (derived here): with constant G and no nucleation, monodisperse seeds grow to
    Ls + G t and the crystal mass is ρ kv N (Ls + G t)^3; adding a constant nucleation rate B adds ρ kv B G^3 t^4/4 and
    B t crystals. Measured: 1e-12 and 1e-11. With very fast growth the solution follows the solubility curve and the yield
    equals the equilibrium yield of `crystallizer_yield` (measured 3e-5)."""
    G, t, ms, Ls = 2e-8, 3600.0, 0.002, 50e-6
    N = ms / (RHO * KV * Ls ** 3)
    r = ok(TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch()))
    f = r["final"]
    assert f["crystal_mass_kg_per_kg_solvent"] == pytest.approx(RHO * KV * N * (Ls + G * t) ** 3, rel=1e-9)
    assert f["seed_final_size_m"] == pytest.approx(Ls + G * t, rel=1e-9)
    assert f["nucleated_mass_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert f["mass_mean_size_m"] == pytest.approx(Ls + G * t, rel=1e-9) and f["mass_weighted_cv"] == pytest.approx(0.0, abs=1e-4)
    assert r["evidence"]["solute_balance_rel_error"] < 1e-12 and r["evidence"]["step_independence_rel_diff"] < 1e-8
    B = 1e5
    rn = ok(TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch(
        nucleation={"k_per_kg_s": B, "order": 0, "magma_exponent": 0})))
    mass = RHO * KV * (N * (Ls + G * t) ** 3 + B * G ** 3 * t ** 4 / 4)
    assert rn["final"]["crystal_mass_kg_per_kg_solvent"] == pytest.approx(mass, rel=1e-8)
    assert rn["final"]["crystal_number_per_kg_solvent"] == pytest.approx(N + B * t, rel=1e-10)
    assert rn["final"]["nucleated_mass_fraction"] == pytest.approx(RHO * KV * B * G ** 3 * t ** 4 / 4 / mass, rel=1e-8)
    fast = ok(TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch(
        growth={"k_m_s": 1e-3, "order": 1}, cooling={"time_s": [0, 7200.0], "T_K": [323.15, 293.15]})))
    eq = TS.crystallizer_yield(1.28, 293.15, SOL_TABLE, {"mode": "none"}, feed_solute_per_kg_solvent=0.28,
                               crystal_solute_mass_fraction=1.0)["crystals_kg"]
    assert fast["final"]["crystallized_kg_per_kg_solvent"] == pytest.approx(eq, rel=1e-4)
    assert fast["final"]["approach_to_equilibrium_yield"] == pytest.approx(1.0, abs=1e-4)
    # slower kinetics leave more supersaturation; more seed area consumes it faster
    slow = TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch(growth={"k_m_s": 2e-6, "order": 1}))
    slower = TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch(growth={"k_m_s": 5e-7, "order": 1}))
    seeded = TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch(growth={"k_m_s": 5e-7, "order": 1},
                                                                                seed_mass_kg_per_kg_solvent=0.02))
    assert slower["final"]["supersaturation_kg_kg"] > slow["final"]["supersaturation_kg_kg"] > 0
    assert seeded["final"]["supersaturation_kg_kg"] < slower["final"]["supersaturation_kg_kg"]
    for x in (slow, slower, seeded):
        ok(x)


def test_batch_seeded_cooling_refusals():
    with pytest.raises(ValueError, match="nucleation"):
        TS.crystal_size_distribution("batch_seeded", RHO, KV, batch={k: v for k, v in _batch().items() if k != "nucleation"})
    with pytest.raises(ValueError, match="growth"):
        TS.crystal_size_distribution("batch_seeded", RHO, KV, batch={k: v for k, v in _batch().items() if k != "growth"})
    with pytest.raises(ValueError, match="no se\\s+extrapola"):
        TS.crystal_size_distribution("batch_seeded", RHO, KV,
                                     batch=_batch(cooling={"time_s": [0, 3600.0], "T_K": [313.15, 283.15]}))
    with pytest.raises(ValueError, match="sin siembra"):
        TS.crystal_size_distribution("batch_seeded", RHO, KV, batch=_batch(seed_mass_kg_per_kg_solvent=0.0))
    with pytest.raises(ValueError, match="seed_size_m"):
        TS.crystal_size_distribution("batch_seeded", RHO, KV, batch={k: v for k, v in _batch().items() if k != "seed_size_m"})


# --------------------------------------------------------------------------------------------------------------- wiring
def test_tools_and_agent_wrappers_are_aligned():
    from packages.agent import wrappers_solids as WS
    assert [f.__name__ for f in TS.SOLIDS_TOOLS] == [w.name for w in WS.AGENT_WRAPPERS]
    assert len(TS.SOLIDS_TOOLS) == 4 and all(f.__doc__ and len(f.__doc__) > 400 for f in TS.SOLIDS_TOOLS)
    out = json.loads(WS.dryer_batch.call({"air": {"T_K": 333.15, "pressure_Pa": P, "relative_humidity": 0.2}}))
    assert out["evidence"]["verified"] and out["air"]["wet_bulb_T_K"] < 333.15
    out = json.loads(WS.crystal_size_distribution.call({
        "mode": "msmpr", "crystal_density_kg_m3": 1500.0, "volume_shape_factor": 0.5, "residence_time_s": 3600.0,
        "growth_rate_m_s": 1e-8, "nucleation_rate_m3_s": 1e6}))
    assert out["sizes"]["dominant_mass_m"] == pytest.approx(3 * 1e-8 * 3600.0)
