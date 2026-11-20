"""Wilson with constant Λ12, Λ21 (textbook form) and the textbook parameters reported by `vle_binary_regression`.

Checks:
  1. synthetic ISOBARIC T-x-y generated from known constant Λ12, Λ21 → `wilson_form="lambda_constante"` recovers them
     (exactly without noise; inside the 95 % interval with noise) and thermo's own Wilson class reproduces the fit;
  2. the energy form reports Λ12(T), Λ21(T) that equal (V2/V1)·exp(b12/T), (V1/V2)·exp(b21/T) of its own b_ij and V,
     i.e. the conversion the agent used to do by hand, and they change with T while the constant form does not;
  3. the real n-hexane + ethanol isobar at 101.3 kPa (Hartanto et al., J. Chem. Eng. Data 2025, 70, 4091, Table 2,
     already in tests/data) fitted both ways: both converge, the statistics differ (the point of the option), both
     stay within the quality bounds the pre-existing test uses, and the two forms agree on Λ inside the data T range
     to the extent a 2-parameter fit can;
  4. NRTL τ/G and UNIQUAC τ are reported too and equal their definitions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from packages.vle import models as M
from packages.vle.equilibrium import Psat, System, VaporPhase
from packages.vle.tool import vle_binary_regression

DATA = Path(__file__).parent / "data"
HX = json.loads((DATA / "vle_hexane_ethanol_101kPa.json").read_text())
HX_PSAT = {n: {"antoine_ext": HX["antoine_ext"][n]} for n in HX["components"]}
HX_SIGMA = {"T_K": 0.10, "P_Pa": 100.0, "x1": 0.003, "y1": 0.003}
HX_SETS = [{"label": "101.3 kPa", "points": HX["points"]}]

L_TRUE = (0.65, 0.40)          # constant Λ12, Λ21 of the synthetic acetone(1) + methanol(2) system
P_ISO = 101325.0
SIG = {"T_K": 0.05, "P_rel": 0.001, "x1": 0.001, "y1": 0.003}


def _isobaric(seed=None, n=17):
    """Isobaric T-x-y at 1 atm from constant Λ (chemicals Psat, ideal vapour), optional Gaussian noise of size SIG."""
    S = System((Psat("acetone", "67-64-1", None), Psat("methanol", "67-56-1", None)), VaporPhase("ideal", ("", "")))
    p = {"a12": float(np.log(L_TRUE[0])), "a21": float(np.log(L_TRUE[1])), "b12": 0.0, "b21": 0.0, "V": (1.0, 1.0)}
    x = np.linspace(0.04, 0.96, n)
    T, y = S.bubble_T(np.full(n, P_ISO), x, lambda T, x: M.ln_gamma("Wilson", T, x, p), T0=np.full(n, 330.0))
    if seed is not None:
        rng = np.random.default_rng(seed)
        T = T + SIG["T_K"] * rng.standard_normal(n)
        y = y + SIG["y1"] * rng.standard_normal(n)
    return [{"T_K": float(t), "P_Pa": P_ISO, "x1": float(a), "y1": float(b)} for t, a, b in zip(T, x, y)]


def _par(mo, name):
    return next(q for q in mo["parametros"] if q["nombre"] == name)


def test_constant_lambda_recovered_exactly_from_noise_free_isobar():
    r = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _isobaric()}], models=["Wilson"],
                              wilson_form="lambda_constante", sigma=SIG, compare_unifac=False)
    assert r["ok"], r
    mo = r["modelos"]["Wilson"]
    assert mo["forma_wilson"] == "lambda_constante" and r["objetivo"]["forma_wilson"] == "lambda_constante"
    assert _par(mo, "Lambda12")["valor"] == pytest.approx(L_TRUE[0], rel=1e-6)
    assert _par(mo, "Lambda21")["valor"] == pytest.approx(L_TRUE[1], rel=1e-6)
    assert _par(mo, "a12")["valor"] == pytest.approx(np.log(L_TRUE[0]), abs=1e-6)
    assert mo["estadistica"]["AAD_T_K"] < 1e-6 and mo["estadistica"]["AAD_y"] < 1e-7
    # no molar volumes involved, and none assumed
    assert mo["fijos"]["V"] == [1.0, 1.0] and not any("Wilson V" in s for s in r["supuestos"])
    # the textbook table is the same Λ at every temperature
    rows = mo["parametros_de_libro"]["a_T"]
    assert [q["origen"] for q in rows] == ["T mínima de los datos", "T media de los datos", "T máxima de los datos"]
    assert all(q["Lambda12"] == pytest.approx(L_TRUE[0], rel=1e-6) and q["Lambda21"] == pytest.approx(L_TRUE[1], rel=1e-6)
               for q in rows)
    # independent implementation (thermo.wilson.Wilson) gives the same γ
    assert mo["verificacion_thermo"]["max_dif_ln_gamma"] < 1e-12


def test_constant_lambda_noisy_isobar_inside_ic95_and_energy_form_differs():
    """Noisy replicates: the true Λ fall inside the reported 95 % interval in most of them (nominal 95 %; ≥ 75 % of 12
    replicates asked, a loose binomial bound). The energy form fitted to the same data is a different model on an
    isobar: its Λ(T) vary across the data range."""
    hits = {"Lambda12": 0, "Lambda21": 0}
    nrep = 12
    for seed in range(nrep):
        r = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _isobaric(seed)}],
                                  models=["Wilson"], wilson_form="lambda_constante", sigma=SIG, compare_unifac=False)
        assert r["ok"], r
        mo = r["modelos"]["Wilson"]
        for k, true in zip(("Lambda12", "Lambda21"), L_TRUE):
            q = _par(mo, k)
            hits[k] += abs(q["valor"] - true) <= q["ic95"]
            assert q["valor"] == pytest.approx(true, abs=0.06)
    assert all(h >= 0.75 * nrep for h in hits.values()), hits

    e = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _isobaric()}], models=["Wilson"],
                              sigma=SIG, compare_unifac=False)
    rows = e["modelos"]["Wilson"]["parametros_de_libro"]["a_T"]
    assert e["modelos"]["Wilson"]["forma_wilson"] == "energia"
    assert abs(rows[0]["Lambda12"] - rows[-1]["Lambda12"]) > 1e-4 or abs(rows[0]["Lambda21"] - rows[-1]["Lambda21"]) > 1e-4
    # data generated with constant Λ: the energy form cannot be exact, the constant form is
    assert e["modelos"]["Wilson"]["funcion_objetivo"] > 1e-6


def test_energy_form_reports_lambda_consistent_with_its_b():
    T_user = 350.0
    r = vle_binary_regression(HX["components"], HX_SETS, models=["Wilson"], sigma=HX_SIGMA, psat=HX_PSAT,
                              compare_unifac=False, report_temperature_K=T_user)
    assert r["ok"], r
    mo = r["modelos"]["Wilson"]
    b12, b21 = _par(mo, "b12"), _par(mo, "b21")
    V1, V2 = mo["fijos"]["V"]
    rows = mo["parametros_de_libro"]["a_T"]
    Ts = [q["T_K"] for q in HX["points"] if 0 < q["x1"] < 1]
    assert rows[0]["T_K"] == pytest.approx(min(Ts)) and rows[2]["T_K"] == pytest.approx(max(Ts))
    assert rows[1]["T_K"] == pytest.approx(np.mean(Ts)) and rows[3] == {**rows[3], "T_K": T_user,
                                                                         "origen": "T pedida por el usuario"}
    for q in rows:
        L12 = V2 / V1 * np.exp(b12["valor"] / q["T_K"])
        L21 = V1 / V2 * np.exp(b21["valor"] / q["T_K"])
        assert q["Lambda12"] == pytest.approx(L12, rel=1e-12) and q["Lambda21"] == pytest.approx(L21, rel=1e-12)
        # interval = Λ·IC95(b)/T (delta method on the only fitted parameter that enters)
        assert q["Lambda12_ic95"] == pytest.approx(L12 * b12["ic95"] / q["T_K"], rel=1e-9)
        assert q["Lambda21_ic95"] == pytest.approx(L21 * b21["ic95"] / q["T_K"], rel=1e-9)
    # Δλ12 = −R·b12 is still reported as before
    assert _par(mo, "Δλ12")["valor"] == pytest.approx(-M.R * b12["valor"])
    # the reported Λ reproduce the model's γ through the textbook Wilson equation written out here
    T0, x1 = rows[1]["T_K"], 0.3
    L12, L21, x2 = rows[1]["Lambda12"], rows[1]["Lambda21"], 0.7
    k = L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1)
    p = {"b12": b12["valor"], "b21": b21["valor"], "V": (V1, V2)}
    g1, g2 = M.ln_gamma("Wilson", T0, x1, p)
    assert float(g1) == pytest.approx(-np.log(x1 + L12 * x2) + x2 * k, abs=1e-12)
    assert float(g2) == pytest.approx(-np.log(x2 + L21 * x1) - x1 * k, abs=1e-12)


def test_hexane_ethanol_published_isobar_fitted_both_ways():
    """Hartanto et al. 2025, Table 2 (101.3 kPa). Bounds AAD_T < 0.7 K and AAD_y < 0.015 are those of the pre-existing
    test for the energy form; the constant form is held to the same quality on this 20 K wide isobar."""
    kw = dict(models=["Wilson"], sigma=HX_SIGMA, psat=HX_PSAT, compare_unifac=False)
    e = vle_binary_regression(HX["components"], HX_SETS, **kw)
    c = vle_binary_regression(HX["components"], HX_SETS, wilson_form="lambda_constante", **kw)
    assert e["ok"] and c["ok"], (e, c)
    me, mc = e["modelos"]["Wilson"], c["modelos"]["Wilson"]
    for mo in (me, mc):
        assert mo["convergencia"]["exito"]
        assert mo["estadistica"]["AAD_T_K"] < 0.7 and mo["estadistica"]["AAD_y"] < 0.015
        assert mo["verificacion_thermo"]["max_dif_ln_gamma"] < 1e-12
        (az,) = mo["azeotropo_predicho"]["101.3 kPa"]
        assert az["x1"] == pytest.approx(0.651, abs=0.02)          # azeotrope of the data (pre-existing test)
    # two different models on isobaric data: the fit statistics are not the same
    assert me["funcion_objetivo"] != pytest.approx(mc["funcion_objetivo"], rel=1e-3)
    assert me["grados_libertad"] == mc["grados_libertad"]
    # the constant Λ lie between the energy-form Λ(T) extremes widened by the intervals (same data, same equation)
    rows = me["parametros_de_libro"]["a_T"]
    for k in ("Lambda12", "Lambda21"):
        lo = min(q[k] - q[k + "_ic95"] for q in rows) - _par(mc, k)["ic95"]
        hi = max(q[k] + q[k + "_ic95"] for q in rows) + _par(mc, k)["ic95"]
        assert lo <= _par(mc, k)["valor"] <= hi, (k, lo, _par(mc, k), hi)
    assert "nota" in mc["energias"] and not any(k.startswith("Δλ") for k in mc["energias"])
    # published constant Λ can be compared in the same call: feed the fitted ones back, statistics must coincide
    pub = {"Wilson": {"Lambda12": _par(mc, "Lambda12")["valor"], "Lambda21": _par(mc, "Lambda21")["valor"], "source": "self"}}
    c2 = vle_binary_regression(HX["components"], HX_SETS, wilson_form="lambda_constante", published_parameters=pub, **kw)
    pp = c2["modelos"]["Wilson"]["parametros_publicados"]
    assert pp["estadistica_con_publicados"]["AAD_T_K"] == pytest.approx(mc["estadistica"]["AAD_T_K"], abs=1e-9)
    assert all(v["dentro_ic95"] for v in pp["comparacion"].values())


def test_nrtl_and_uniquac_textbook_parameters_reported():
    r = vle_binary_regression(HX["components"], HX_SETS, models=["NRTL", "UNIQUAC"], nrtl_alpha=0.47, sigma=HX_SIGMA,
                              psat=HX_PSAT, compare_unifac=False, report_temperature_K=[298.15])
    assert r["ok"], r
    n = r["modelos"]["NRTL"]
    row = n["parametros_de_libro"]["a_T"][-1]
    assert row["T_K"] == 298.15
    b12 = _par(n, "b12")
    assert row["tau12"] == pytest.approx(b12["valor"] / 298.15) and row["G12"] == pytest.approx(np.exp(-0.47 * row["tau12"]))
    assert row["tau12_ic95"] == pytest.approx(b12["ic95"] / 298.15, rel=1e-9)
    assert row["G12_ic95"] == pytest.approx(0.47 * row["G12"] * row["tau12_ic95"], rel=1e-9)
    u = r["modelos"]["UNIQUAC"]
    row = u["parametros_de_libro"]["a_T"][-1]
    assert row["tau21"] == pytest.approx(np.exp(_par(u, "b21")["valor"] / 298.15))
    assert "forma_wilson" not in r["objetivo"]


def test_bad_options_are_refused():
    r = vle_binary_regression(HX["components"], HX_SETS, models=["Wilson"], wilson_form="constante", psat=HX_PSAT)
    assert not r["ok"] and "wilson_form" in r["error"]
    r = vle_binary_regression(HX["components"], HX_SETS, models=["Wilson"], report_temperature_K=-5, psat=HX_PSAT)
    assert not r["ok"] and "report_temperature_K" in r["error"]
    r = vle_binary_regression(HX["components"], HX_SETS, models=["Wilson"], wilson_form="lambda_constante", psat=HX_PSAT,
                              published_parameters={"Wilson": {"d12_J_mol": 1.0, "d21_J_mol": 1.0}})
    assert not r["ok"] and "Lambda12" in r["error"]
