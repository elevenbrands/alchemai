"""shortcut_distillation_design: Fenske-Underwood-Gilliland-Kirkbride with end-point volatilities from the own flash.

Checks are internal identities of the method (no literature numbers): Fenske reproduces the key split it was given,
Underwood's root closes, the modes (R ↔ N) invert each other, and for a binary at saturated-liquid feed Underwood
reduces to the closed form R_min = [x_D/z − α(1 − x_D)/(1 − z)]/(α − 1) with the SAME α the tool reports.
"""
import math

import pytest

from packages.dwsim_client import tools_properties as TPR
from packages.vle import fug

C4C5 = ["isobutane", "n-butane", "isopentane", "n-pentane"]


def _run(**kw):
    base = dict(components=C4C5, feed={"mole_fractions": [0.25, 0.25, 0.25, 0.25], "molar_flow": 100.0},
                pressure_Pa=2.5e5, light_key="n-butane", heavy_key="isopentane",
                recovery_light_key_in_distillate=0.99, recovery_heavy_key_in_bottoms=0.98, model="PR")
    base.update(kw)
    return TPR.shortcut_distillation_design(**base)


def test_fenske_split_and_balances():
    o = _run()
    assert o["ok"], o
    p = o["products"]
    assert p["distillate_flows"]["n-butane"] == pytest.approx(0.99 * 25) and p["bottoms_flows"]["isopentane"] == pytest.approx(0.98 * 25)
    for c in C4C5:
        assert p["distillate_flows"][c] + p["bottoms_flows"][c] == pytest.approx(25.0, rel=1e-12)
    a = o["relative_volatility_to_heavy_key"]["used_geometric_mean"]
    d, b = p["distillate_flows"], p["bottoms_flows"]
    assert o["N_min"] == pytest.approx(math.log(d["n-butane"] / b["n-butane"] * b["isopentane"] / d["isopentane"]) / math.log(a["n-butane"]))
    for c in ("isobutane", "n-pentane"):                     # non-keys follow the Fenske line
        assert d[c] / b[c] == pytest.approx(d["isopentane"] / b["isopentane"] * a[c] ** o["N_min"], rel=1e-6)
    assert d["isobutane"] > 24.9 and d["n-pentane"] < 0.2
    top, bot = (o["relative_volatility_to_heavy_key"][k]["n-butane"] for k in ("top", "bottom"))
    assert a["n-butane"] == pytest.approx(math.sqrt(top * bot))
    assert o["sensitivity"]["N_min_if_alpha_at_feed_only"] != pytest.approx(o["N_min"], rel=1e-3)


def test_modes_invert_each_other_and_duties_close_the_enthalpy_balance():
    a = _run(R_over_Rmin=1.3)
    op = a["operation"]
    assert op["feasible"] and op["reflux_ratio"] == pytest.approx(1.3 * a["R_min"]) and op["stages"] > a["N_min"]
    b = _run(n_stages=op["stages"])["operation"]
    assert b["reflux_ratio"] == pytest.approx(op["reflux_ratio"], rel=1e-6)
    c = _run(reflux_ratio=op["reflux_ratio"])["operation"]
    assert c["stages"] == pytest.approx(op["stages"], rel=1e-9)
    assert op["stages_above_feed_kirkbride"] + op["stages_below_feed_kirkbride"] == pytest.approx(op["stages"])
    # saturated-liquid feed, total condenser: reboiler ≈ condenser within the sensible-heat difference of the products
    assert 0.8 < op["reboiler_duty"] / op["condenser_duty"] < 1.25 and op["condenser_duty"] > 0
    assert _run(n_stages=a["N_min"] * 0.9)["operation"]["feasible"] is False
    assert _run(reflux_ratio=a["R_min"] * 0.9)["operation"]["feasible"] is False
    assert fug.gilliland_N(1e6, a["R_min"], a["N_min"]) == pytest.approx(a["N_min"], rel=1e-3)   # total reflux limit


def test_binary_underwood_closed_form_and_feed_condition():
    kw = dict(components=["benzene", "toluene"], pressure_Pa=101325.0, light_key="benzene", heavy_key="toluene",
              recovery_light_key_in_distillate=0.95, recovery_heavy_key_in_bottoms=0.95, model="PR")
    o = TPR.shortcut_distillation_design(feed={"mole_fractions": [0.5, 0.5], "q": 1.0}, **kw)
    al = o["relative_volatility_to_heavy_key"]["used_geometric_mean"]["benzene"]
    xD = o["products"]["x_distillate"]["benzene"]
    assert xD == pytest.approx(0.95)
    assert o["R_min"] == pytest.approx((xD / 0.5 - al * (1 - xD) / 0.5) / (al - 1), rel=1e-8)
    assert 2.2 < al < 2.7                                   # benzene/toluene at 1 atm: α ≈ 2.4–2.5 (textbook range)
    sat = TPR.shortcut_distillation_design(feed={"mole_fractions": [0.5, 0.5], "vapor_fraction": 0.0}, **kw)
    assert sat["feed"]["q"] == pytest.approx(1.0, abs=1e-9) and sat["R_min"] == pytest.approx(o["R_min"], rel=1e-6)
    vap = TPR.shortcut_distillation_design(feed={"mole_fractions": [0.5, 0.5], "vapor_fraction": 1.0}, **kw)
    assert vap["feed"]["q"] == pytest.approx(0.0, abs=1e-9) and vap["R_min"] > o["R_min"]
    # let-down: saturated liquid at 3 atm flashes partly at 1 atm
    hot = TPR.shortcut_distillation_design(feed={"mole_fractions": [0.5, 0.5], "vapor_fraction": 0.0, "P_Pa": 3 * 101325.0}, **kw)
    assert 0.0 < hot["feed"]["q"] < 1.0


def test_bad_input_is_reported():
    assert "error" in _run(light_key="isopentane", heavy_key="n-butane")
    assert "error" in _run(light_key="propane")
    assert "error" in _run(n_stages=30, reflux_ratio=2.0)
    assert "error" in _run(recovery_light_key_in_distillate=99.0)
