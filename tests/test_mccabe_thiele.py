"""binary_distillation_design: McCabe-Thiele on the real equilibrium curve.

Analytic anchors (no literature numbers needed): with a CONSTANT relative volatility the stepped N_min equals Fenske and
the feed-pinch R_min equals the closed form for a saturated liquid feed, R_min = [x_D/x_F − α(1 − x_D)/(1 − x_F)]/(α − 1).
A constant α is obtained exactly with two Antoine equations that share B and C (ln α = (A1 − A2)·ln 10).
"""
import math

import pytest

from packages.dwsim_client import tools_properties as TPR

ALPHA = 2.5
PSAT = {"benzene": {"antoine": {"A": 6.9 + math.log10(ALPHA), "B": 1211.0, "C": 220.8}},
        "toluene": {"antoine": {"A": 6.9, "B": 1211.0, "C": 220.8}}}


def _ideal(**kw):
    return TPR.binary_distillation_design(["benzene", "toluene"], 101325.0, {"z1": 0.5, "q": 1.0, "molar_flow": 100.0},
                                          0.95, x_bottoms=0.05, model="ideal", psat=PSAT, **kw)


def test_constant_alpha_matches_fenske_and_underwood():
    o = _ideal()
    assert o["ok"] and o["azeotropes"] == []
    fenske = math.log(0.95 / 0.05 * 0.95 / 0.05) / math.log(ALPHA)
    # stepped stages hit x_B exactly only by chance: the fractional last stage is linear, Fenske is logarithmic
    assert o["N_min"] == pytest.approx(fenske, abs=0.15)
    assert o["constant_alpha_comparison"]["fenske_N_min_geometric_mean_alpha"] == pytest.approx(fenske, rel=1e-6)
    r_min = (0.95 / 0.5 - ALPHA * 0.05 / 0.5) / (ALPHA - 1)
    assert o["R_min"] == pytest.approx(r_min, rel=1e-4) and o["R_min_pinch"] == "alimentación"
    assert o["products"]["distillate_flow"] == pytest.approx(50.0) and o["products"]["bottoms_flow"] == pytest.approx(50.0)
    for k in ("at_x_distillate", "at_feed", "at_x_bottoms"):
        assert o["relative_volatility"][k] == pytest.approx(ALPHA, rel=1e-6)


def test_integer_stage_count_at_total_reflux_is_exact():
    """With α constant, n stages from x_D land exactly on x = x_D/(x_D + α^n (1 − x_D)): choose x_B there, N_min = n."""
    xD, n = 0.95, 6
    xB = xD / (xD + ALPHA ** n * (1 - xD))
    o = TPR.binary_distillation_design(["benzene", "toluene"], 101325.0, {"z1": 0.5}, xD, x_bottoms=xB, model="ideal", psat=PSAT)
    assert o["N_min"] == pytest.approx(n, abs=2e-3)


def test_reflux_and_stage_modes_are_consistent():
    o = _ideal(reflux_ratio=2.0)
    st = o["at_reflux_ratio"]
    assert st["feasible"] and st["stages"] > o["N_min"]
    back = _ideal(n_stages=st["stages"])["for_n_stages"]
    assert back["feasible"] and back["reflux_ratio_required"] == pytest.approx(2.0, rel=1e-4)
    assert back["optimal_feed_stage"] == st["optimal_feed_stage"]
    low = _ideal(reflux_ratio=0.5 * o["R_min"])["at_reflux_ratio"]
    assert low["feasible"] is False
    few = _ideal(n_stages=o["N_min"] - 1)["for_n_stages"]
    assert few["feasible"] is False and "N_min" in few["reason"]


def test_recovery_specification_and_feed_condition():
    o = TPR.binary_distillation_design(["benzene", "toluene"], 101325.0, {"z1": 0.4, "vapor_fraction": 0.5}, 0.9,
                                       recovery_in_distillate=0.9, model="ideal", psat=PSAT)
    p = o["products"]
    assert p["D_over_F"] == pytest.approx(0.9 * 0.4 / 0.9) and p["x_bottoms"] == pytest.approx(0.04 / 0.6)
    assert o["feed"]["q"] == pytest.approx(0.5)
    sat = TPR.binary_distillation_design(["benzene", "toluene"], 101325.0, {"z1": 0.4, "q": 1.0}, 0.9,
                                         recovery_in_distillate=0.9, model="ideal", psat=PSAT)
    assert o["R_min"] > sat["R_min"]                      # partly vaporised feed needs more reflux
    cold = TPR.binary_distillation_design(["benzene", "toluene"], 101325.0, {"z1": 0.4, "T_K": 300.0}, 0.9,
                                          recovery_in_distillate=0.9, model="ideal", psat=PSAT)
    assert cold["feed"]["q"] > 1.0 and any("calidad de la alimentación" in a for a in cold["assumptions"])


def test_ethanol_water_wilson_sees_azeotrope_and_tangent_pinch():
    kw = dict(components=["ethanol", "water"], pressure_Pa=101325.0, feed={"z1": 0.5, "q": 1.0}, x_bottoms=0.01)
    w = TPR.binary_distillation_design(x_distillate=0.84, model="Wilson", **kw)
    i = TPR.binary_distillation_design(x_distillate=0.84, model="ideal", **kw)
    assert w["ok"] and i["ok"]
    az = w["azeotropes"]
    # ethanol-water azeotrope at 1 atm, read on 2026-11-20: 95.6 wt % ethanol (4.40 wt % water) at 78.20 °C in the
    # Edinburgh azeotrope database (https://www.homepages.ed.ac.uk/jwp/Chemeng/azeotrope/EE.html) and 95.5 wt % at
    # 78.1 °C in the Lange / CRC table reproduced in https://en.wikipedia.org/wiki/Azeotrope_tables, i.e.
    # x1 = 0.892-0.895 (M = 46.07, 18.015) and 351.25-351.35 K. The library Wilson set puts it at x1 = 0.869, 351.25 K:
    # the temperature is right and the composition is 0.025 low (the y-x curve is almost tangent to the diagonal there,
    # so the azeotrope composition is very sensitive to the parameters). The window on x1 states that known deviation
    # of the default parameters; it is not a literature uncertainty.
    assert len(az) == 1 and az[0]["x1"] == pytest.approx(0.894, abs=0.03) and az[0]["T_K"] == pytest.approx(351.3, abs=0.3)
    assert w["R_min_pinch"].startswith("tangente") and i["R_min_pinch"] == "alimentación"
    assert w["N_min"] > i["N_min"] and w["R_min"] > 1.5 * i["R_min"]
    assert any("librería" in a for a in w["assumptions"]) and w["warnings"]
    beyond = TPR.binary_distillation_design(x_distillate=0.93, model="Wilson", **kw)
    assert beyond["ok"] is False and beyond["feasible"] is False and "azeótropo" in beyond["reason"]


def test_wilson_parameter_forms_agree_and_bad_input_is_reported():
    kw = dict(components=["ethanol", "water"], pressure_Pa=101325.0, feed={"z1": 0.5, "q": 1.0}, x_distillate=0.8,
              x_bottoms=0.02, model="Wilson")
    lib = TPR.binary_distillation_design(**kw)
    p = lib["thermo"]["parameters"]
    same = TPR.binary_distillation_design(parameters={k: p[k] for k in ("a12", "a21", "b12", "b21", "V")}, **kw)
    assert same["N_min"] == pytest.approx(lib["N_min"], rel=1e-12) and same["thermo"]["source"] == "del usuario"
    const = TPR.binary_distillation_design(parameters={"lambda12": 1.0, "lambda21": 1.0}, **kw)   # Λ = 1: ideal
    ideal = TPR.binary_distillation_design(**{**kw, "model": "ideal"})
    assert const["N_min"] == pytest.approx(ideal["N_min"], rel=1e-9)
    assert "error" in TPR.binary_distillation_design(**{**kw, "x_bottoms": None})
    assert "error" in TPR.binary_distillation_design(**{**kw, "x_bottoms": 0.6})
    assert "error" in TPR.binary_distillation_design(**{**kw, "model": "UNIQUAC"})
