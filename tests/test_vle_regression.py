"""Bloque M14: binary interaction parameters from VLE data (`vle_binary_regression`).

Evidence:
  * Hartanto et al., J. Chem. Eng. Data 70 (2025) 4091 (Green Open Access copy, TU Delft): n-hexane + ethanol at
    101.3 kPa (Table 2), Aspen extended Antoine (Table 3), NRTL by Britt-Luecke maximum likelihood (Table 8).
  * Ethanol + water: Kamihama et al. 2012 (101.3 kPa) and Voutsas et al. 2011 (32.86 kPa), as shipped in VLizard
    (MIT), with the Redlich-Kister D published by Zbytovský et al., ACS Omega 10 (2025) 46809, Table 2.
  * Synthetic sets from KNOWN parameters (labelled as such): parameter recovery inside the 95 % intervals, and a set
    whose γ1 and γ2 come from different models (violates Gibbs-Duhem by construction) that the tests must reject.
  * Every fitted model re-evaluated with thermo's own NRTL / UNIQUAC / Wilson classes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from packages.vle import consistency as C
from packages.vle import models as M
from packages.vle.equilibrium import Psat, System, VaporPhase
from packages.vle.tool import vle_binary_regression

DATA = Path(__file__).parent / "data"
HX = json.loads((DATA / "vle_hexane_ethanol_101kPa.json").read_text())
EW = json.loads((DATA / "vle_ethanol_water_vlizard.json").read_text())
HX_PSAT = {n: {"antoine_ext": HX["antoine_ext"][n]} for n in HX["components"]}
EW_PSAT = {n: {"antoine_ext": EW["antoine_ext"][n]} for n in EW["components"]}
# Hartanto 2025 expanded uncertainties (k = 2) → standard deviations
HX_SIGMA = {"T_K": 0.10, "P_Pa": 100.0, "x1": 0.003, "y1": 0.003}


def _system(d, psat):
    return System(tuple(Psat(n, "-", psat[n]) for n in d["components"]), VaporPhase("ideal", ("", "")))


def _arrays(pts):
    pts = [q for q in pts if 0 < q["x1"] < 1]
    return tuple(np.array([q[c] for q in pts]) for c in ("T_K", "P_Pa", "x1", "y1"))


# ------------------------------------------------------------------ models vs thermo
@pytest.mark.parametrize("model,p", [
    ("NRTL", dict(b12=600.0, b21=150.0, a12=0.3, a21=-0.2, alpha=0.35)),
    ("UNIQUAC", dict(b12=-120.0, b21=40.0, a12=0.1, a21=-0.05, r=(2.5755, 0.92), q=(2.588, 1.4))),
    ("Wilson", dict(b12=-300.0, b21=-500.0, V=(5.9e-5, 1.8e-5))),
])
def test_models_match_thermo(model, p):
    for T in (300.0, 350.0):
        for x in (0.01, 0.3, 0.7, 0.99):
            a = M.ln_gamma(model, np.array([T]), np.array([x]), p)
            b = M.thermo_ln_gamma(model, T, x, p)
            assert a[0][0] == pytest.approx(b[0], abs=1e-12) and a[1][0] == pytest.approx(b[1], abs=1e-12)


def test_legendre_gibbs_duhem():
    """ln γ from the Legendre g^E satisfy x1 dlnγ1 + x2 dlnγ2 = 0 (finite difference)."""
    a = np.array([0.8, -0.2, 0.1, 0.05, -0.02])
    x = np.linspace(0.05, 0.95, 19)
    h = 1e-6
    l1p, l2p = C.legendre_ln_gamma(a, x + h)
    l1m, l2m = C.legendre_ln_gamma(a, x - h)
    gd = x * (l1p - l1m) / (2 * h) + (1 - x) * (l2p - l2m) / (2 * h)
    assert np.max(np.abs(gd)) < 1e-7


# ------------------------------------------------------------------ Hartanto 2025, n-hexane + ethanol
def test_hartanto_experimental_gammas_reproduced():
    """γ1, γ2 of Table 2 from y P / (x Psat) with the paper's extended Antoine (Table 3)."""
    S = _system(HX, HX_PSAT)
    T, P, x, y = _arrays(HX["points"])
    g1, g2 = C.experimental_gammas(S, T, P, x, y)
    table_g1 = [8.161, 8.490, 6.248, 5.416, 5.025, 4.847, 4.790, 4.461, 3.247, 1.568, 1.341, 1.159, 1.075, 1.025,
                1.036, 1.028]
    table_g2 = [1.031, 1.046, 1.035, 1.026, 1.043, 1.035, 1.028, 1.060, 1.169, 1.958, 2.597, 3.922, 6.218, 11.598,
                13.378, 10.275]
    # ≤ 1 % everywhere except γ2 at x1 = 0.990 (2.5 %: 1 - y1 = 0.061 is printed with 2 significant figures)
    assert np.max(np.abs(g1 / table_g1 - 1)) < 0.01
    assert np.max(np.abs(g2[:-1] / table_g2[:-1] - 1)) < 0.01 and abs(g2[-1] / table_g2[-1] - 1) < 0.03


def test_hartanto_ml_fit_contains_published_parameters():
    """Same objective as the paper (Britt-Luecke maximum likelihood, α = 0.47, τ = A + B/T, the paper's σ): every
    published parameter lies inside our 95 % interval. The paper fitted 101.3 + 61.3 kPa together (61.3 kPa data of
    Ortega & Espiau not reproduced here), so with one isobar A and B are almost perfectly correlated and the
    intervals are wide: the tool must say so."""
    r = vle_binary_regression(HX["components"], [{"label": "101.3 kPa", "points": HX["points"]}], models=["NRTL"],
                              nrtl_alpha=0.47, temperature_dependence="a+b/T", objective="ml", sigma=HX_SIGMA,
                              psat=HX_PSAT, published_parameters={"NRTL": HX["published_nrtl"]}, compare_unifac=False)
    assert r["ok"], r
    cmp_ = r["modelos"]["NRTL"]["parametros_publicados"]["comparacion"]
    assert set(cmp_) == {"a12", "a21", "b12", "b21"}
    assert all(v["dentro_ic95"] for v in cmp_.values()), cmp_
    assert any("correlacionados" in a for a in r["avisos"])
    assert r["modelos"]["NRTL"]["convergencia"]["exito"]


def test_hartanto_published_parameters_rmsd_discrepancy_is_reported():
    """The published Table 8 parameters on these 101.3 kPa points give RMSD(T) ≈ 0.79 K, larger than the 0.38 K
    of Table 9 (which pools both isobars). Our own fit does better on this isobar. Documented, not hidden."""
    r = vle_binary_regression(HX["components"], [{"label": "101.3 kPa", "points": HX["points"]}], models=["NRTL"],
                              nrtl_alpha=0.47, sigma=HX_SIGMA, psat=HX_PSAT,
                              published_parameters={"NRTL": HX["published_nrtl"]}, compare_unifac=False)
    pub = r["modelos"]["NRTL"]["parametros_publicados"]["estadistica_con_publicados"]
    fit = r["modelos"]["NRTL"]["estadistica"]
    assert pub["RMSD_T_K"] == pytest.approx(0.785, abs=0.01)
    assert fit["RMSD_T_K"] < pub["RMSD_T_K"] and fit["RMSD_T_K"] < 0.5


def test_hartanto_azeotrope_and_thermo_reproduction():
    r = vle_binary_regression(HX["components"], [{"label": "101.3 kPa", "points": HX["points"]}], nrtl_alpha=0.47,
                              sigma=HX_SIGMA, psat=HX_PSAT)
    assert r["ok"], r
    az_data = r["azeotropo"][0]["de_los_datos"]
    assert az_data["x1"] == pytest.approx(0.651, abs=0.002) and az_data["T_K"] == pytest.approx(331.30, abs=0.02)
    for m in ("NRTL", "UNIQUAC", "Wilson"):
        mo = r["modelos"][m]
        (az,) = mo["azeotropo_predicho"]["101.3 kPa"]
        assert az["x1"] == pytest.approx(az_data["x1"], abs=0.02) and az["T_K"] == pytest.approx(az_data["T_K"], abs=0.5)
        assert az["x1_con_thermo"] == pytest.approx(az["x1"], abs=1e-9)
        vt = mo["verificacion_thermo"]
        assert vt["max_dif_ln_gamma"] < 1e-12
        assert vt["estadistica_con_thermo"]["AAD_T_K"] == pytest.approx(mo["estadistica"]["AAD_T_K"], abs=1e-9)
        assert mo["estadistica"]["AAD_T_K"] < 0.7 and mo["estadistica"]["AAD_y"] < 0.015
    u = r["unifac_dortmund"]
    assert u["estadistica"]["AAD_T_K"] < 1.0 and u["azeotropo_x1_cerca_del_ajustado"]["101.3 kPa"]
    # pure-component check against the paper's Antoine: endpoints within 0.3 K
    pure = r["consistencia"][0]["componente_puro"]
    assert all(abs(p["desv_T_K"]) < 0.3 for p in pure)


# ------------------------------------------------------------------ ethanol + water, consistent published sets
def test_ethanol_water_consistency_reproduces_published_D():
    """Zbytovský 2025 Table 2 (Aspen extended Antoine): EtOH-H2O at 32.9 kPa D = 3.8 (Voutsas 2011 is the only set at
    that pressure). Kamihama 2012 at 101.3 kPa gives 1.65, matching row 'D' (1.7); the table does not name the source
    of each row, so that match is probable, not confirmed. Both sets are consistent by the area and point tests."""
    r = vle_binary_regression(EW["components"], [{"label": k, "points": v} for k, v in EW["sets"].items()],
                              psat=EW_PSAT, compare_unifac=False)
    assert r["ok"], r
    c = {e["conjunto"]: e for e in r["consistencia"]}
    assert c["Voutsas2011_32kPa"]["area_redlich_kister"]["D"] == pytest.approx(3.8, abs=0.06)
    assert c["Kamihama2012_101kPa"]["area_redlich_kister"]["D"] == pytest.approx(1.7, abs=0.06)
    for e in c.values():
        assert e["veredicto"] == "consistente"
        assert e["punto_fredenslund"]["dy_medio"] < 0.01
        assert e["van_ness_directa"]["indice"] <= 2
    # azeotrope of the 101.3 kPa data vs the fitted models
    az = {a["conjunto"]: a for a in r["azeotropo"]}["Kamihama2012_101kPa"]
    assert az["de_los_datos"]["x1"] == pytest.approx(0.895, abs=0.002)
    for m, pred in az["predicho_por_modelo"].items():
        assert pred and pred[0]["x1"] == pytest.approx(0.895, abs=0.02), (m, pred)


def test_virial_vapor_phase_runs_and_changes_little_at_low_pressure():
    sets = [{"label": k, "points": v} for k, v in EW["sets"].items()]
    a = vle_binary_regression(EW["components"], sets, models=["NRTL"], psat=EW_PSAT, compare_unifac=False)
    b = vle_binary_regression(EW["components"], sets, models=["NRTL"], psat=EW_PSAT, vapor_phase="virial",
                              compare_unifac=False)
    assert b["ok"] and "Tsonopoulos" in b["fase_vapor"]
    da = a["modelos"]["NRTL"]["energias"]["Δg21_J_mol"]
    db = b["modelos"]["NRTL"]["energias"]["Δg21_J_mol"]
    assert abs(da - db) / abs(da) < 0.05


# ------------------------------------------------------------------ synthetic: recovery and inconsistency
TRUE = {"b12": 450.0, "b21": 250.0, "alpha": 0.3}
SYN_SIG = {"T_K": 0.1, "P_rel": 0.002, "x1": 0.001, "y1": 0.004}


def _synthetic(seed, lng1=None, lng2=None, T=328.15, n=15):
    """Isothermal P-x-y of acetone(1) + methanol(2) at T from known NRTL parameters (chemicals Psat), plus noise."""
    S = System((Psat("acetone", "67-64-1", None), Psat("methanol", "67-56-1", None)), VaporPhase("ideal", ("", "")))
    x = np.linspace(0.05, 0.95, n)
    Tv = np.full(n, T)
    l1, l2 = M.ln_gamma("NRTL", Tv, x, TRUE)
    if lng1 is not None:
        l1 = lng1(x)
    if lng2 is not None:
        l2 = lng2(x)
    a1, a2 = x * np.exp(l1) * S.psat[0](Tv), (1 - x) * np.exp(l2) * S.psat[1](Tv)
    P, y = a1 + a2, a1 / (a1 + a2)
    rng = np.random.default_rng(seed)
    P = P * (1 + SYN_SIG["P_rel"] * rng.standard_normal(n))
    y = y + SYN_SIG["y1"] * rng.standard_normal(n)
    return [{"T_K": T, "P_Pa": float(p), "x1": float(a), "y1": float(b)} for p, a, b in zip(P, x, y)]


def test_synthetic_recovery_coverage():
    """30 noisy replicates with the true σ: each true parameter falls inside the 95 % interval in ≥ 80 % of them
    (expected 95 %; binomial 30 trials gives P(≤ 23) ≈ 0.2 %)."""
    hits = {"b12": 0, "b21": 0}
    N = 30
    for seed in range(N):
        r = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _synthetic(seed)}],
                                  models=["NRTL"], nrtl_alpha=0.3, sigma=SYN_SIG, compare_unifac=False)
        for p in r["modelos"]["NRTL"]["parametros"]:
            if p["nombre"] in hits:
                hits[p["nombre"]] += abs(p["valor"] - TRUE[p["nombre"]]) <= p["ic95"]
    assert all(h >= 24 for h in hits.values()), hits


def test_synthetic_ml_recovery():
    r = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _synthetic(7)}], models=["NRTL"],
                              nrtl_alpha=0.3, sigma=SYN_SIG, objective="ml", compare_unifac=False)
    assert r["ok"], r
    for p in r["modelos"]["NRTL"]["parametros"][:2]:
        assert abs(p["valor"] - TRUE[p["nombre"]]) <= p["ic95"], p
    assert len(r["modelos"]["NRTL"]["valores_verdaderos_estimados"]) == 15


def test_synthetic_consistent_set_passes():
    r = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _synthetic(3)}], models=["NRTL"],
                              sigma=SYN_SIG, compare_unifac=False)
    c = r["consistencia"][0]
    assert c["tipo"] == "isothermal" and c["veredicto"] == "consistente", c
    assert c["area_redlich_kister"]["D"] < 10 and c["punto_fredenslund"]["dy_medio"] < 0.01


def test_gibbs_duhem_violating_set_fails():
    """γ1 from the true NRTL, γ2 from a different function (ln γ2 × 3): the pair violates Gibbs-Duhem by
    construction, so the area and point tests must reject it (synthetic, labelled; no legible published
    inconsistent set was found, see APP.md)."""
    x = np.linspace(0.05, 0.95, 15)
    l2_true = M.ln_gamma("NRTL", np.full(15, 328.15), x, TRUE)[1]
    pts = _synthetic(3, lng2=lambda xx: 3.0 * l2_true)
    r = vle_binary_regression(["acetone", "methanol"], [{"label": "GD-roto", "points": pts}], models=["NRTL"],
                              sigma=SYN_SIG, compare_unifac=False)
    c = r["consistencia"][0]
    assert c["area_redlich_kister"]["consistente"] is False and c["punto_fredenslund"]["consistente"] is False, c
    assert c["veredicto"] == "inconsistente"
    ok = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _synthetic(3)}], models=["NRTL"],
                               sigma=SYN_SIG, compare_unifac=False)["consistencia"][0]
    assert c["van_ness_directa"]["rms_delta_ln_g1_g2"] > 3 * ok["van_ness_directa"]["rms_delta_ln_g1_g2"]


# ------------------------------------------------------------------ input handling
def test_missing_data_is_refused_not_invented():
    r = vle_binary_regression(["acetone", "methanol"], [{"label": "a", "points": [{"T_K": 300.0, "x1": 0.5}]}])
    assert not r["ok"] and "P_Pa" in r["error"]
    r = vle_binary_regression(["acetone", "methanol"], [])
    assert not r["ok"] and "no se inventan" in r["error"]
    r = vle_binary_regression(["acetone"], [{"points": [{"T_K": 300.0, "P_Pa": 1e5, "x1": 0.5}]}])
    assert not r["ok"]


def test_defaults_are_labelled_as_assumptions():
    r = vle_binary_regression(["acetone", "methanol"], [{"label": "syn", "points": _synthetic(1)}],
                              compare_unifac=False)
    s = " ".join(r["supuestos"])
    for key in ("α = 0.3", "σ por defecto", "fase vapor ideal", "thermo.VaporPressure", "UNIQUAC r, q", "Wilson V"):
        assert key in s, key
    json.dumps(r, default=float, ensure_ascii=False)  # serialisable for the agent / MCP
