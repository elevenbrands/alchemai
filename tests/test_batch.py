"""Batch operations (bloque M26_E2): batch / fed-batch fermentation with scheduling and cost, and batch distillation.

Reference data, all READ on 2026-11-16 (the bands asserted are the deviations measured that day, not targets):

[UMD]    N. S. Wang, CHBE482 (Univ. of Maryland), solutions to Shuler & Kargi Problem 9.2 run as a two-stage BATCH
         (Monod growth, then non-growth product formation): https://terpconnect.umd.edu/~nsw/chbe482/p9-2b.pdf (fixed-step
         RK4, 100 steps) and https://terpconnect.umd.edu/~nsw/chbe482/p9-2c.pdf (adaptive Odesolve). The two files give
         8.147e-3 and 8.121e-3 g/(L·h) for the same case: that 0.3 % is the numerical noise of the reference itself.
[WANKAT] P. C. Wankat, «Separation Process Engineering», chapter 9 (publisher's sample chapter):
         Example 9-1 https://www.informit.com/articles/article.aspx?p=2738308&seqNum=4
         Example 9-2 https://www.informit.com/articles/article.aspx?p=2738308&seqNum=7
         Example 9-3 and Table 9-2 https://www.informit.com/articles/article.aspx?p=2738308&seqNum=8
[GEANK]  Geankoplis, «Transport Processes and Unit Operations», Example 11.3-2 (p. 647), data and result as reproduced by
         the FOSSEE Python Textbook Companion:
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter11.ipynb
         (the companion's own numerical answer 0.276 / 0.836; the book's printed answer was not opened).
[CR]     Coulson & Richardson vol. 2, eq. 11.30 (Rayleigh with constant relative volatility), as reproduced in
         https://www.uomus.edu.iq/img/lectures21/MUCLecture_2022_42213779.pdf
[NREL]   Humbird et al. (2011), NREL/TP-5100-47764, equipment list: F-300 ethanol fermentor 1,000,000 gal, 12 units,
         $10,128,000 (2009$); A-300 agitator 30 hp $52,500; M-910 CIP system $421,000; agitation 0.03 hp/1000 gal
         (6 W/m3). https://www.osti.gov/servlets/purl/1013269
Analytical solutions (derived in `packages.bioprocess.batch` or here) are marked «analytical»."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from packages.agent import wrappers_batch as WB
from packages.bioprocess import batch as B
from packages.dwsim_client import tools_batch as TB

H = 3600.0
INIT = {"volume_m3": 1.0, "biomass_kg_m3": 0.1, "substrate_kg_m3": 5.0, "product_kg_m3": 0.0}
UMD = {"mu_max_1_s": 0.3 / H, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.4}                 # [UMD] growth stage
UMD2 = {"mu_max_1_s": 0.0, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.4, "beta_kg_kg_s": 0.02 / H, "Yps_kg_kg": 0.6}   # production stage


def _ok(r: dict) -> dict:
    json.dumps(r)
    assert r["evidence"]["verified"] is True, r["evidence"]
    for k in ("warnings", "assumptions", "not_included", "method"):
        assert k in r
    return r


# ------------------------------------------------------------------------------------------------ fermentation
def test_exponential_phase_analytical():
    """«analytical»: with S >> Ks growth is exponential, X = X0·exp(mu_max·t), and S = S0 − (X − X0)/Yxs."""
    k = {"mu_max_1_s": 0.5 / H, "Ks_kg_m3": 1e-7, "Yxs_kg_kg": 0.5}
    r = _ok(B.batch_fermentation(k, dict(INIT, substrate_kg_m3=50.0), 6 * H, n_points=7))
    for t, x, s in zip(r["profile"]["t_s"], r["profile"]["biomass_kg_m3"], r["profile"]["substrate_kg_m3"]):
        assert x == pytest.approx(0.1 * math.exp(0.5 * t / H), rel=1e-6)
        assert s == pytest.approx(50.0 - (x - 0.1) / 0.5, rel=1e-9)


def test_monod_closed_form_along_the_profile():
    """«analytical»: integrated Monod t(S) (formula in monod_batch_time_closed_form) at every profile point."""
    r = _ok(B.batch_fermentation(UMD, INIT, 12 * H, n_points=25))
    assert r["evidence"]["monod_closed_form"]["abs_diff_over_S0"] < 1e-8
    checked = 0
    for t, s in zip(r["profile"]["t_s"][1:], r["profile"]["substrate_kg_m3"][1:]):
        if s > 0.01:                 # t(S) is ill-conditioned once S << Ks: there the comparison is made in S (evidence)
            assert B.monod_batch_time_closed_form(0.3 / H, 0.1, 0.4, 0.1, 5.0, s) == pytest.approx(t, rel=1e-7)
            checked += 1
    assert checked >= 15
    # yield identity and asymptote X -> X0 + Yxs·S0
    f = r["final"]
    assert f["biomass_kg_m3"] == pytest.approx(0.1 + 0.4 * (5.0 - f["substrate_kg_m3"]), rel=1e-10)


def test_two_stage_batch_course_solution_umd():
    """[UMD] productivity = p_final/t_f2 for growth until t_f1 then production until t_f2 (x0 0.1, s0 5 g/L).
    prod(10 h, 50 h): 8.147e-3 (RK4 file) and 8.121e-3 (Odesolve file); optimum (9.318 h, 38.063 h): 0.023 g/(L·h)."""
    def prod(tf1: float, tf2: float) -> float:
        a = _ok(B.batch_fermentation(UMD, INIT, tf1 * H))["final"]
        b = B.batch_fermentation(UMD2, dict(INIT, biomass_kg_m3=a["biomass_kg_m3"], substrate_kg_m3=a["substrate_kg_m3"]),
                                 (tf2 - tf1) * H)
        assert b["evidence"]["verified"]
        return b["final"]["product_kg_m3"] / tf2
    p = prod(10.0, 50.0)
    assert p == pytest.approx(8.121e-3, rel=5e-4)          # measured −0.015 % vs the adaptive-solver file
    assert p == pytest.approx(8.147e-3, rel=4e-3)          # measured −0.33 % vs the fixed-step RK4 file
    assert round(prod(9.318, 38.063), 3) == 0.023          # measured 0.02293


def test_stage_two_stops_at_substrate_exhaustion():
    """«analytical»: no growth, zero-order uptake beta·X/Yps: S hits zero at t = S0·Yps/(beta·X) and P = Yps·S0."""
    r = B.batch_fermentation(UMD2, dict(INIT, biomass_kg_m3=2.0, substrate_kg_m3=0.5), 100 * H)
    assert r["stop_reason"] == "substrate_exhausted" and r["warnings"]
    assert r["simulated_time_s"] == pytest.approx(0.5 * 0.6 / (0.02 * 2.0) * H, rel=1e-6)
    assert r["final"]["product_kg_m3"] == pytest.approx(0.6 * 0.5, rel=1e-6)
    json.dumps(r)


def test_maintenance_and_death_limits_analytical():
    """«analytical»: mu_max = 0 → maintenance alone empties S linearly; death alone gives X = X0·exp(−kd·t)."""
    k = {"mu_max_1_s": 0.0, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.5, "maintenance_kg_kg_s": 0.05 / H}
    r = B.batch_fermentation(k, dict(INIT, biomass_kg_m3=10.0, substrate_kg_m3=2.0), 10 * H)
    assert r["simulated_time_s"] == pytest.approx(2.0 / (0.05 * 10.0) * H, rel=1e-6)
    k = {"mu_max_1_s": 0.0, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.5, "kd_1_s": 0.1 / H}
    r = _ok(B.batch_fermentation(k, INIT, 5 * H))
    assert r["final"]["biomass_kg_m3"] == pytest.approx(0.1 * math.exp(-0.5), rel=1e-8)
    assert r["final"]["dead_biomass_kg"] == pytest.approx(0.1 * (1 - math.exp(-0.5)), rel=1e-7)


def test_fed_batch_quasi_steady_state_limit():
    """«analytical» quasi-steady state of a constant-feed fed-batch (D = F/V < mu_max, Sf >> S):
    S ≈ Ks·D/(mu_max − D) and total biomass X·V = X0·V0 + F·Yxs·Sf·t (Pirt; Shuler & Kargi section 9.3.3)."""
    k = {"mu_max_1_s": 0.3 / H, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.5}
    r = _ok(B.batch_fermentation(k, {"volume_m3": 1e-3, "biomass_kg_m3": 50.0, "substrate_kg_m3": 0.2, "product_kg_m3": 0.0},
                                 5 * H, feed={"flow_m3_s": 0.2e-3 / H, "substrate_kg_m3": 100.0}, n_points=11))
    assert r["mode"] == "fed_batch"
    pr = r["profile"]
    for t, v, s, x in zip(pr["t_s"][1:], pr["volume_m3"][1:], pr["substrate_kg_m3"][1:], pr["biomass_kg_m3"][1:]):
        d = 0.2e-3 / v
        assert s == pytest.approx(0.1 * d / (0.3 - d), rel=7e-3)                 # measured ≤ 0.55 %
        assert x * v == pytest.approx(0.05 + 0.2e-3 * 0.5 * 100.0 * t / H, rel=6e-4)   # measured ≤ 0.05 %
    assert pr["volume_m3"][-1] == pytest.approx(2e-3, rel=1e-9)


def test_fed_batch_shuler_kargi_example_inputs():
    """Inputs of the fed-batch example of Shuler & Kargi (V = 1 L and F = 0.2 L/h at t = 2 h, Sf = 100 g/L, mu_max
    0.3 1/h, Ks 0.1 g/L, Yxs 0.5, 30 g of cells at t = 0, qP = 0.2 g/g/h). The book page was NOT opened: the reference
    values are the «analytical» quasi-steady-state formulas (V0 = 0.6 L, S = Ks·D/(mu_max − D) = 0.2 g/L,
    Xt = 30 + F·Y·Sf·t = 50 g, P·V = qP·(Xt0·t + F·Y·Sf·t²/2) = 16 g). The full ODE starts with D0 > mu_max, so it is
    not yet at quasi-steady state early on: the deviations below are those of the approximation, not of the solver."""
    k = {"mu_max_1_s": 0.3 / H, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.5, "beta_kg_kg_s": 0.2 / H}
    r = _ok(B.batch_fermentation(k, {"volume_m3": 0.6e-3, "biomass_kg_m3": 50.0, "substrate_kg_m3": 0.0, "product_kg_m3": 0.0},
                                 2 * H, feed={"flow_m3_s": 0.2e-3 / H, "substrate_kg_m3": 100.0}))
    f = r["final"]
    assert f["volume_m3"] == pytest.approx(1e-3, rel=1e-9)
    assert f["substrate_kg_m3"] == pytest.approx(0.2, rel=0.03)                       # measured +2.0 %
    assert f["biomass_kg_m3"] * f["volume_m3"] == pytest.approx(0.050, rel=3e-3)      # measured −0.2 %
    assert f["product_kg_m3"] == pytest.approx(16.0, rel=0.012)                       # measured −0.9 %


def test_exponential_feed_exact_solution():
    """«analytical», exact: with F = F0·exp(k·t), S0 = S* = Ks·k/(mu_max − k) and X0·V0 = F0·(Sf − S*)·Yxs/k the
    solution is S = S* and X·V = X0·V0·exp(k·t) for all t (substitute in the balances)."""
    mu, ks, y, sf, kset, f0 = 0.4 / H, 0.5, 0.5, 300.0, 0.15 / H, 0.002 / H
    s_star = ks * kset / (mu - kset)
    x0 = f0 * (sf - s_star) * y / kset
    r = _ok(B.batch_fermentation({"mu_max_1_s": mu, "Ks_kg_m3": ks, "Yxs_kg_kg": y},
                                 {"volume_m3": 1.0, "biomass_kg_m3": x0, "substrate_kg_m3": s_star, "product_kg_m3": 0.0}, 10 * H,
                                 feed={"flow_m3_s": f0, "substrate_kg_m3": sf, "exponential_rate_1_s": kset}, n_points=11))
    pr = r["profile"]
    for t, v, s, x in zip(pr["t_s"], pr["volume_m3"], pr["substrate_kg_m3"], pr["biomass_kg_m3"]):
        assert s == pytest.approx(s_star, rel=1e-6)
        assert x * v == pytest.approx(x0 * math.exp(kset * t), rel=1e-7)
        assert v == pytest.approx(1.0 + f0 * (math.exp(kset * t) - 1) / kset, rel=1e-9)


def test_feed_window_and_volume_limit():
    k = {"mu_max_1_s": 0.3 / H, "Ks_kg_m3": 0.1, "Yxs_kg_kg": 0.5}
    r = _ok(B.batch_fermentation(k, INIT, 10 * H, feed={"flow_m3_s": 0.1 / H, "substrate_kg_m3": 200.0, "start_s": 2 * H,
                                                        "stop_s": 6 * H, "max_volume_m3": 1.25}, n_points=21))
    v = r["profile"]["volume_m3"]
    assert v[4] == pytest.approx(1.0) and max(v) == pytest.approx(1.25, rel=1e-6) and v[-1] == pytest.approx(1.25, rel=1e-6)
    s = r["summary"]
    assert s["substrate_fed_kg"] == pytest.approx(0.25 * 200.0, rel=1e-6)
    assert s["biomass_formed_kg"] == pytest.approx(0.5 * s["substrate_consumed_kg"], rel=1e-9)


def test_inhibition_forms_and_monotonicity():
    base = {"mu_max_1_s": 0.4 / H, "Ks_kg_m3": 1.0, "Yxs_kg_kg": 0.1, "alpha_kg_kg": 4.0}
    init = dict(INIT, biomass_kg_m3=1.0, substrate_kg_m3=150.0)
    free = _ok(B.batch_fermentation(base, init, 20 * H, target={"substrate_conversion": 0.9}))
    lev = _ok(B.batch_fermentation(dict(base, inhibition=[{"model": "levenspiel", "P_max_kg_m3": 90.0, "n": 1.0}]), init,
                                   60 * H, target={"substrate_conversion": 0.9}))
    aiba = _ok(B.batch_fermentation(dict(base, inhibition={"model": "aiba", "k_m3_kg": 0.03}), init, 60 * H,
                                    target={"substrate_conversion": 0.9}))
    hyp = _ok(B.batch_fermentation(dict(base, inhibition=[{"model": "hyperbolic", "Kp_kg_m3": 30.0}]), init, 60 * H,
                                   target={"substrate_conversion": 0.9}))
    t0 = free["target"]["time_s"]
    assert all(r["target"]["reached"] and r["target"]["time_s"] > t0 for r in (lev, aiba, hyp))
    # same stoichiometry whatever the kinetics: P = alpha·(X − X0), X − X0 = Yxs·(S0 − S)
    for r in (free, lev, aiba, hyp):
        s = r["summary"]
        assert s["substrate_conversion"] == pytest.approx(0.9, abs=1e-9)
        assert s["product_kg_m3"] == pytest.approx(4.0 * 0.1 * 0.9 * 150.0, rel=1e-8)
    # the kinetic factors themselves («analytical»)
    p = B._parse_kinetics(dict(base, inhibition=[{"model": "andrews", "Ki_kg_m3": 100.0},
                                                {"model": "levenspiel", "P_max_kg_m3": 90.0, "n": 2.0}]))
    assert B._mu(p, 10.0, 45.0) == pytest.approx(0.4 / H * 10.0 / (1.0 + 10.0 + 1.0) * 0.25)
    s_opt = math.sqrt(1.0 * 100.0)                      # Andrews maximum at S = sqrt(Ks·Ki)
    assert B._mu(p, s_opt, 0.0) > max(B._mu(p, 0.8 * s_opt, 0.0), B._mu(p, 1.25 * s_opt, 0.0))
    assert B._mu(p, 10.0, 95.0) == 0.0
    # Levenspiel: growth stops at P_max, so the substrate is left unconverted
    stuck = B.batch_fermentation(dict(base, inhibition=[{"model": "levenspiel", "P_max_kg_m3": 40.0, "n": 1.0}]), init, 400 * H,
                                 target={"substrate_conversion": 0.9})
    assert not stuck["target"]["reached"] and stuck["final"]["product_kg_m3"] <= 40.0 * (1 + 1e-8) and stuck["warnings"]


CARBON = {"S": 6 * 12.011 / (6 * 12.011 + 12 * 1.008 + 6 * 15.999),
          "P": 2 * 12.011 / (2 * 12.011 + 6 * 1.008 + 15.999),
          "X": 12.011 / (12.011 + 1.8 * 1.008 + 0.5 * 15.999 + 0.2 * 14.007)}
ETHANOL = {"mu_max_1_s": 0.3 / H, "Ks_kg_m3": 1.0, "Yxs_kg_kg": 0.11, "alpha_kg_kg": 4.17,
           "inhibition": [{"model": "levenspiel", "P_max_kg_m3": 90.0, "n": 1.0}]}
FORMULAS = {"substrate_formula": "C6H12O6", "biomass_formula": "CH1.8O0.5N0.2", "product_formula": "C2H6O"}
BIG = {"volume_m3": 100.0, "biomass_kg_m3": 2.0, "substrate_kg_m3": 150.0, "product_kg_m3": 0.0}


def test_carbon_balance_and_co2():
    """Carbon identity with hand-computed carbon fractions (atomic masses typed here, independent of the engine)."""
    r = _ok(B.batch_fermentation(ETHANOL, BIG, 30 * H, stoichiometry=dict(FORMULAS, co2={"mol_per_mol_product": 1.0})))
    ds, dx, dp = (r["summary"][k] for k in ("substrate_consumed_kg", "biomass_formed_kg", "product_formed_kg"))
    mw_p = (2 * 12.011 + 6 * 1.008 + 15.999) / 1000.0
    assert r["co2"]["total_mol"] == pytest.approx(dp / mw_p, rel=5e-5)        # atomic-mass tables differ in the 5th digit
    rec = (dx * CARBON["X"] + dp * CARBON["P"] + r["co2"]["total_mol"] * 0.012011) / (ds * CARBON["S"])
    assert r["co2"]["carbon_recovery"] == pytest.approx(rec, rel=5e-5)
    assert 0.95 < rec < 1.05                                  # published yeast yields (0.11, 4.17) close carbon to +3 %
    c = _ok(B.batch_fermentation(ETHANOL, BIG, 30 * H, stoichiometry=dict(FORMULAS, co2="carbon_balance")))
    closed = dx * CARBON["X"] + dp * CARBON["P"] + c["co2"]["total_mol"] * 0.012011
    assert closed == pytest.approx(ds * CARBON["S"], rel=5e-5)
    assert np.trapezoid(c["profile"]["co2_rate_mol_s"], c["profile"]["t_s"]) == pytest.approx(c["co2"]["total_mol"], rel=2e-2)
    # yields that fix more carbon than the substrate has: flagged, not hidden
    bad = B.batch_fermentation(dict(ETHANOL, Yxs_kg_kg=0.5), BIG, 30 * H, stoichiometry=dict(FORMULAS, co2="carbon_balance"))
    assert bad["evidence"]["verified"] is False and bad["co2"]["total_mol"] < 0


def test_heat_and_cooling_duty():
    q = 5.0e5
    r = _ok(B.batch_fermentation(ETHANOL, BIG, 30 * H, heat={"basis": "substrate", "heat_J_per_kg_substrate": q},
                                 agitation_W_m3=50.0, n_points=241))
    assert r["heat"]["total_J"] == pytest.approx(q * r["summary"]["substrate_consumed_kg"], rel=1e-9)
    assert np.trapezoid(r["profile"]["heat_rate_W"], r["profile"]["t_s"]) == pytest.approx(r["heat"]["total_J"], rel=2e-3)
    assert r["cooling_duty"]["peak_W"] == pytest.approx(r["heat"]["peak_metabolic_W"] + 50.0 * 100.0, rel=1e-9)
    # Hess with heats of combustion: Q = hS·ΔS − hX·ΔX − hP·ΔP (inputs here are arbitrary test numbers)
    hc = {"substrate": 15.6e6, "biomass": 21.2e6, "product": 29.7e6}
    h = _ok(B.batch_fermentation(ETHANOL, BIG, 30 * H, heat={"basis": "combustion", "heats_of_combustion_J_kg": hc}))
    s = h["summary"]
    assert h["heat"]["total_J"] == pytest.approx(hc["substrate"] * s["substrate_consumed_kg"] - hc["biomass"]
                                                 * s["biomass_formed_kg"] - hc["product"] * s["product_formed_kg"], rel=1e-9)
    assert h["cooling_duty"]["includes"].endswith("agitation_W_m3)")


@pytest.fixture(scope="module")
def nrel_plant():
    """12 fermentors of exactly 3785 m3 (1 MM gal, the NREL quote) at the NREL cost index, 36 h reaction."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    before = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    cycle = (36 + 4 + 4 + 4) * H
    sch = {"broth_rate_m3_s": 12 * 3785.0 * 0.95 / cycle, "fill_time_s": 4 * H, "empty_time_s": 4 * H, "clean_time_s": 4 * H,
           "working_volume_fraction": 0.95, "n_vessels": 12}
    r = B.batch_fermentation(ETHANOL, BIG, 36 * H, heat={"basis": "substrate", "heat_J_per_kg_substrate": 5.0e5},
                             agitation_W_m3=5.9, schedule=sch, cepci=521.9)
    after = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    return r, before, after


def test_schedule_identities(nrel_plant):
    r, _, _ = nrel_plant
    s = _ok(r)["schedule"]
    assert s["cycle_time_s"] == pytest.approx(48 * H) and s["n_vessels"] == 12
    assert s["vessel_volume_m3"] == pytest.approx(3785.0, rel=1e-12)
    assert s["n_vessels"] * s["working_volume_m3"] / s["cycle_time_s"] == pytest.approx(s["broth_rate_m3_s"], rel=1e-12)
    assert s["vessels_in_reaction_on_average"] == pytest.approx(12 * 36 / 48)
    assert r["evidence"]["biosteam_size_batch_volume_rel_diff"] < 1e-12       # consistency with BioSTEAM, not validation
    assert s["per_vessel"]["agitator_power_kW"] == pytest.approx(5.9 * 3785.0 * 0.95 / 1000.0)
    # max_vessel_volume instead of n_vessels: ceil and re-sizing; product-rate basis
    sch = {"broth_rate_m3_s": 1000.0 / H, "fill_time_s": 0.0, "empty_time_s": 1 * H, "clean_time_s": 2 * H,
           "working_volume_fraction": 0.95, "max_vessel_volume_m3": 1000.0, "cost": False}
    q = _ok(B.batch_fermentation(ETHANOL, BIG, 30 * H, schedule=sch))["schedule"]
    # BioSTEAM's documented example of size_batch (F 1e3, tau 30, cleaning 3, V_wf 0.95, V_max 1e3, no loading):
    # 35 reactors of 992.48 (docstring of biosteam.units.design_tools.size_batch) — consistency with BioSTEAM
    assert q["n_vessels"] == 35 and q["vessel_volume_m3"] == pytest.approx(992.48, abs=0.005)
    sch2 = dict(sch, product_rate_kg_s=1.0)
    del sch2["broth_rate_m3_s"]
    q2 = _ok(B.batch_fermentation(ETHANOL, BIG, 30 * H, schedule=sch2))
    assert q2["schedule"]["broth_rate_m3_s"] == pytest.approx(1.0 / q2["final"]["product_kg_m3"], rel=1e-9)


def test_cost_at_the_nrel_reference_point(nrel_plant):
    """[NREL] at the quoted size and cost index the purchase costs must be the NREL quotes: 12 fermentors
    $10,128,000; agitator $52,500 each; CIP system $421,000 (all 2009$)."""
    r, before, after = nrel_plant
    c = r["schedule"]["cost"]
    eq = {e["biosteam_item"]: e for e in c["equipment"]}
    assert eq["Reactors"]["purchase_cost_usd"] == pytest.approx(10_128_000.0, rel=1e-9) and eq["Reactors"]["count"] == 12
    assert eq["Agitators"]["unit_purchase_cost_usd"] == pytest.approx(52_500.0, rel=1e-9)
    assert eq["Cleaning in place"]["purchase_cost_usd"] == pytest.approx(421_000.0, rel=1e-9)
    assert (eq["Reactors"]["F_BM"], eq["Agitators"]["F_BM"], eq["Cleaning in place"]["F_BM"]) == (1.5, 1.5, 1.8)   # [NREL]
    assert c["electricity_kW"]["agitators_from_user_P_V"] == pytest.approx(12 * 5.9 * 3785 * 0.95 / 1000)
    assert c["total_purchase_cost_usd"] == pytest.approx(sum(e["purchase_cost_usd"] for e in c["equipment"]))
    for e in c["equipment"]:                                   # shape capital_investment accepts
        assert {"name", "F_BM", "purchase_cost_usd", "base_purchase_cost_usd", "basis"} <= set(e)
    from packages.costing.economics import capital_investment
    ci = capital_investment(c["equipment"], {k: 0.0 for k in ("site_preparation", "service_facilities", "allocated_costs",
                                                              "contingency_and_fee", "land", "royalties", "startup",
                                                              "working_capital")})
    json.dumps(ci, default=float)
    # BioSTEAM hygiene: thermo flags and bst.CE as found
    assert before == after
    import biosteam as bst
    assert bst.CE == c["biosteam_cepci"]


def test_cost_scaling_and_range_warning(nrel_plant):
    r, _, _ = nrel_plant
    sch = {"broth_rate_m3_s": 50.0 / H, "fill_time_s": 2 * H, "empty_time_s": 2 * H, "clean_time_s": 2 * H,
           "working_volume_fraction": 0.9, "n_vessels": 4}
    a = B.batch_fermentation(ETHANOL, BIG, 36 * H, schedule=sch, year=2017)
    b = B.batch_fermentation(ETHANOL, BIG, 36 * H, schedule=sch, year=2023)
    ca, cb = a["schedule"]["cost"], b["schedule"]["cost"]
    assert cb["total_purchase_cost_usd"] / ca["total_purchase_cost_usd"] == pytest.approx(797.9 / 567.5, rel=1e-9)
    v = a["schedule"]["vessel_volume_m3"]
    reactors = next(e for e in ca["equipment"] if e["biosteam_item"] == "Reactors")
    assert reactors["unit_purchase_cost_usd"] == pytest.approx(844_000.0 * (v / 3785.0) ** 0.5 * 567.5 / 521.9, rel=1e-9)
    assert 378.5 < v < 3785.0 and not any("fuera de" in w for w in a["warnings"])
    small = B.batch_fermentation(ETHANOL, BIG, 36 * H, schedule=dict(sch, broth_rate_m3_s=2.0 / H), year=2017)
    assert small["schedule"]["vessel_volume_m3"] < 378.5 and any("fuera de" in w for w in small["warnings"])
    assert not any(e["biosteam_item"] == "Heat exchangers" for e in ca["equipment"])   # no heat given: not costed
    with pytest.raises(ValueError, match="cepci"):
        B.batch_fermentation(ETHANOL, BIG, 36 * H, schedule=sch, year=2026)


def test_fermentation_refuses_missing_inputs():
    good = dict(kinetics=UMD, initial=INIT, t_end_s=H)
    cases = [dict(kinetics=None), dict(kinetics={"mu_max_1_s": 1e-4}), dict(kinetics=dict(UMD, Yxs=0.5)),
             dict(initial={"volume_m3": 1.0}), dict(t_end_s=None), dict(feed={"flow_m3_s": 1e-6}),
             dict(heat={"basis": "combustion"}), dict(heat={"basis": "substrate"}),
             dict(stoichiometry={"substrate_formula": "C6H12O6"}), dict(stoichiometry=dict(FORMULAS)),
             dict(schedule={"fill_time_s": 1.0}), dict(target={"conversion": 0.9}),
             dict(kinetics=dict(UMD, inhibition=[{"model": "levenspiel"}])),
             dict(kinetics=dict(UMD, inhibition=[{"model": "monod"}])),
             dict(schedule={"fill_time_s": 0.0, "empty_time_s": 0.0, "clean_time_s": 0.0, "working_volume_fraction": 0.9,
                            "n_vessels": 2})]
    for kw in cases:
        with pytest.raises(ValueError):
            B.batch_fermentation(**dict(good, **kw))


def test_published_kinetic_sets_carry_their_source():
    p = B.published_kinetic_sets()
    json.dumps(p)
    assert p["date_read"] == "2026-11-16" and len(p["sets"]) >= 3
    for s in p["sets"]:
        assert s["url"].startswith("https://") and s["source"] and s["caveats"] and s["as_tool_arguments"]
    hojo = p["sets"][0]["as_tool_arguments"]
    r = _ok(B.batch_fermentation(dict(hojo, Yxs_kg_kg=0.11, alpha_kg_kg=4.17), dict(INIT, substrate_kg_m3=100.0), 40 * H))
    assert r["final"]["product_kg_m3"] < 80.0


# ------------------------------------------------------------------------------------------------ distillation
def test_rayleigh_constant_alpha_binary_analytical():
    """[CR] eq. 11.30: ln(S/S0) = [1/(alpha−1)]·ln[x·(1−x0)/(x0·(1−x))] + ln[(1−x0)/(1−x)]."""
    a, x0 = 2.41, 0.5
    for frac in (0.1, 0.4, 0.8):
        r = _ok(B.batch_distillation(["benzene", "toluene"], 100.0, [x0, 1 - x0], {"type": "constant_alpha", "alpha": [a, 1.0]},
                                     {"distilled_fraction": frac}))
        x = r["final"]["still_mole_fractions"]["benzene"]
        lhs = math.log(x * (1 - x0) / (x0 * (1 - x))) / (a - 1) + math.log((1 - x0) / (1 - x))
        assert lhs == pytest.approx(math.log(1 - frac), rel=1e-8)
        f = r["final"]
        assert f["still_mol"] * x + f["distillate_mol"] * f["average_distillate_mole_fractions"]["benzene"] == pytest.approx(50.0)
        assert r["evidence"]["rayleigh_analytical_max_abs_diff"] < 1e-9


def test_wankat_example_9_3_multicomponent():
    """[WANKAT] Ex. 9-3: benzene/toluene/cumene 0.70/0.20/0.10, alpha 2.4/1.0/0.21, 1 kmol, until xB = 0.40:
    area 1.1919, Wfinal 0.3036, Dtotal 0.6964, xT 0.3175, xC 0.2825, xD,avg 0.8308/0.1488/0.0204.
    Table 9-2: at xB = 0.30 the cumulative area is 1.4962 and W = 0.2239 kmol."""
    args = (["benzene", "toluene", "cumene"], 1.0, [0.7, 0.2, 0.1], {"type": "constant_alpha", "alpha": [2.4, 1.0, 0.21]})
    f = _ok(B.batch_distillation(*args, {"residue_mole_fraction": 0.4, "component": "benzene"}))["final"]
    assert f["rayleigh_integral_ln_W0_over_W"] == pytest.approx(1.1919, abs=1e-4)
    assert f["still_mol"] == pytest.approx(0.3036, abs=1e-4) and f["distillate_mol"] == pytest.approx(0.6964, abs=1e-4)
    assert f["still_mole_fractions"]["toluene"] == pytest.approx(0.3175, abs=1e-4)
    assert f["still_mole_fractions"]["cumene"] == pytest.approx(0.2825, abs=1e-4)
    for c, v in (("benzene", 0.8308), ("toluene", 0.1488), ("cumene", 0.0204)):
        assert f["average_distillate_mole_fractions"][c] == pytest.approx(v, abs=1e-4)
    g = _ok(B.batch_distillation(*args, {"residue_mole_fraction": 0.3, "component": "benzene"}))["final"]
    assert g["rayleigh_integral_ln_W0_over_W"] == pytest.approx(1.4962, abs=1e-4)
    assert g["still_mol"] == pytest.approx(0.2239, abs=1e-4)


METHANOL_WATER = {"type": "table", "x": [0.50, 0.60, 0.65, 0.70, 0.75, 0.80],      # [WANKAT] table of Example 9-1
                  "y": [0.780, 0.825, 0.845, 0.871, 0.895, 0.915]}


@pytest.mark.parametrize("xfin, area, wfinal, dcalc, xd", [
    (0.70, 0.7044, 24.72, 25.28, 0.898),       # first guess (graphical); Simpson gives W 24.79, D 25.21
    (0.60, 1.2084, 14.93, 35.07, 0.885),
    (0.65, 0.971, 18.94, 31.06, 0.891)])
def test_wankat_example_9_1_methanol_water(xfin, area, wfinal, dcalc, xd):
    """[WANKAT] Ex. 9-1: 50 mol of 80 % methanol, simple distillation to xfin; book values by graphical integration.
    Measured: area within 0.6 %, Wfinal within 0.6 %, xD,avg within 0.0007."""
    f = _ok(B.batch_distillation(["methanol", "water"], 50.0, [0.8, 0.2], METHANOL_WATER,
                                 {"residue_mole_fraction": xfin, "component": "methanol"}))["final"]
    assert f["rayleigh_integral_ln_W0_over_W"] == pytest.approx(area, rel=7e-3)
    assert f["still_mol"] == pytest.approx(wfinal, rel=7e-3) and f["distillate_mol"] == pytest.approx(dcalc, rel=4e-3)
    assert f["average_distillate_mole_fractions"]["methanol"] == pytest.approx(xd, abs=1e-3)


def test_wankat_example_9_1_specified_distillate():
    """Same example solved for its actual specification xD,avg = 0.892: the book stops its trial and error at
    xfin = 0.65 («xD is insensitive to xfin»); the converged root with the same table is 0.659 (W 19.7 vs 18.94)."""
    r = _ok(B.batch_distillation(["methanol", "water"], 50.0, [0.8, 0.2], METHANOL_WATER,
                                 {"average_distillate_mole_fraction": 0.892, "component": "methanol"}))
    assert r["final"]["still_mole_fractions"]["methanol"] == pytest.approx(0.65, abs=0.012)
    assert r["final"]["still_mol"] == pytest.approx(18.94, rel=0.05)
    # leaving the table is reported, never extrapolated
    out = B.batch_distillation(["methanol", "water"], 50.0, [0.8, 0.2], METHANOL_WATER,
                               {"residue_mole_fraction": 0.3, "component": "methanol"})
    assert out["evidence"]["verified"] is False and out["final"]["still_mole_fractions"]["methanol"] == pytest.approx(0.5, abs=1e-6)


def test_geankoplis_example_11_3_2_pentane_heptane():
    """[GEANK] 100 mol of 50 % n-pentane / n-heptane at 101.3 kPa, 40 mol distilled. Companion: residue 0.276, average
    distillate 0.836 (cubic interpolation of the 8-point table, 0.005 tolerance on the integral). Measured here with
    monotone PCHIP: 0.2788 and 0.8318 (+1.0 % and −0.5 %): the difference is the interpolation of a sparse table."""
    tab = {"type": "table", "x": [0.000, 0.059, 0.145, 0.254, 0.398, 0.594, 0.867, 1.000],
           "y": [0.000, 0.271, 0.521, 0.701, 0.836, 0.925, 0.984, 1.000]}
    r = _ok(B.batch_distillation(["n-pentane", "n-heptane"], 100.0, [0.5, 0.5], tab, {"distilled_fraction": 0.4},
                                 boilup_rate_mol_s=0.01, heat_of_vaporization_J_mol=30000.0))
    f = r["final"]
    assert f["still_mol"] == pytest.approx(60.0) and f["distillate_mol"] == pytest.approx(40.0)
    assert f["still_mole_fractions"]["n-pentane"] == pytest.approx(0.276, abs=0.004)
    assert f["average_distillate_mole_fractions"]["n-pentane"] == pytest.approx(0.836, abs=0.006)
    # time and heat identities for simple distillation: t = D/V, Q = V·lambda·t
    assert f["batch_time_s"] == pytest.approx(40.0 / 0.01)
    assert r["heat"]["reboiler_heat_J"] == pytest.approx(40.0 * 30000.0) and r["heat"]["reboiler_duty_W"] == pytest.approx(300.0)


def test_wankat_example_9_2_multistage_with_flash_engine():
    """[WANKAT] Ex. 9-2: 50 kmol of 32 mol% ethanol in water, still + 2 stages, L/D = 2/3, to 4.5 mol% in the still.
    Book (ethanol-water data of its Table 2-1, McCabe-Thiele on paper): graphical area 0.608, Wfinal 27.21, Dtotal 22.79,
    xD,avg 0.648; two-part Simpson: area 0.6196, Wfinal 26.91, Dtotal 23.09, xD,avg 0.640.
    Here the equilibrium is NOT the book's table (not openly available) but UNIFAC from the repo flash engine, so this
    tests the engine + thermo route against the book. Measured: area 0.6145, W 27.05, D 22.95, xD,avg 0.644 — between
    the book's two answers."""
    r = _ok(B.batch_distillation(["ethanol", "water"], 50e3, [0.32, 0.68], {"type": "thermo", "model": "UNIFAC", "P_Pa": 101325.0},
                                 {"residue_mole_fraction": 0.045, "component": "ethanol"}, mode="constant_reflux", n_stages=3,
                                 reflux_ratio=2 / 3, boilup_rate_mol_s=10.0, n_points=9))
    f = r["final"]
    lo, hi = sorted((0.608, 0.6196))
    assert lo < f["rayleigh_integral_ln_W0_over_W"] < hi
    assert 26.91e3 < f["still_mol"] < 27.21e3 and 22.79e3 < f["distillate_mol"] < 23.09e3
    assert 0.640 < f["average_distillate_mole_fractions"]["ethanol"] < 0.648
    assert r["evidence"]["flash_calls_not_verified"] == 0
    assert f["batch_time_s"] == pytest.approx(f["distillate_mol"] * (2 / 3 + 1) / 10.0)        # D·(R+1)/V
    assert r["heat"]["reboiler_heat_J"] == pytest.approx(10.0 * r["heat"]["heat_of_vaporization_J_mol"] * f["batch_time_s"])
    assert 3.9e4 < r["heat"]["heat_of_vaporization_J_mol"] < 4.4e4 and 355.0 < f["still_T_K"] < 373.15
    assert any("azeótropo" in w for w in r["warnings"])


def test_stage_limits_analytical():
    """«analytical»: one stage with any reflux is simple distillation (the reflux returns to the still); at total reflux
    the distillate follows Fenske, xD/(1−xD) = alpha^N · xW/(1−xW)."""
    eq = {"type": "constant_alpha", "alpha": [2.41, 1.0]}
    a = _ok(B.batch_distillation(["b", "t"], 100.0, [0.5, 0.5], eq, {"distilled_fraction": 0.5}))["final"]
    b = _ok(B.batch_distillation(["b", "t"], 100.0, [0.5, 0.5], eq, {"distilled_fraction": 0.5}, mode="constant_reflux",
                                 n_stages=1, reflux_ratio=3.0))["final"]
    assert b["still_mole_fractions"]["b"] == pytest.approx(a["still_mole_fractions"]["b"], rel=1e-10)
    c = _ok(B.batch_distillation(["b", "t"], 100.0, [0.5, 0.5], eq, {"distilled_fraction": 0.3}, mode="constant_reflux",
                                 n_stages=4, reflux_ratio=1e7))["final"]
    xw, xd = c["still_mole_fractions"]["b"], c["instantaneous_distillate_mole_fractions"]["b"]
    assert xd / (1 - xd) == pytest.approx(2.41 ** 4 * xw / (1 - xw), rel=1e-5)
    # monotonic: more reflux or more stages -> richer accumulated distillate
    prev = a["average_distillate_mole_fractions"]["b"]
    for n, rr in ((2, 1.0), (2, 3.0), (4, 3.0), (8, 3.0)):
        f = _ok(B.batch_distillation(["b", "t"], 100.0, [0.5, 0.5], eq, {"distilled_fraction": 0.5}, mode="constant_reflux",
                                     n_stages=n, reflux_ratio=rr))["final"]
        assert f["average_distillate_mole_fractions"]["b"] > prev
        prev = f["average_distillate_mole_fractions"]["b"]


def test_multicomponent_multistage_consistency():
    """CONSISTENCY ONLY: a trace third component must reproduce the binary column; balances close."""
    two = _ok(B.batch_distillation(["a", "b"], 1.0, [0.5, 0.5], {"type": "constant_alpha", "alpha": [2.4, 1.0]},
                                   {"distilled_fraction": 0.4}, mode="constant_reflux", n_stages=4, reflux_ratio=2.0))["final"]
    three = _ok(B.batch_distillation(["a", "b", "c"], 1.0, [0.5, 0.5 - 1e-9, 1e-9],
                                     {"type": "constant_alpha", "alpha": [2.4, 1.0, 0.21]}, {"distilled_fraction": 0.4},
                                     mode="constant_reflux", n_stages=4, reflux_ratio=2.0, boilup_rate_mol_s=1e-3))["final"]
    assert three["still_mole_fractions"]["a"] == pytest.approx(two["still_mole_fractions"]["a"], abs=1e-7)
    full = _ok(B.batch_distillation(["B", "T", "C"], 1.0, [0.7, 0.2, 0.1], {"type": "constant_alpha", "alpha": [2.4, 1.0, 0.21]},
                                    {"distilled_fraction": 0.5}, mode="constant_reflux", n_stages=4, reflux_ratio=2.0))["final"]
    for c, z in (("B", 0.7), ("T", 0.2), ("C", 0.1)):
        assert (0.5 * full["still_mole_fractions"][c] + 0.5 * full["average_distillate_mole_fractions"][c]
                == pytest.approx(z, abs=1e-9))
    assert full["average_distillate_mole_fractions"]["B"] > 0.9 and full["recovery_in_distillate"]["C"] < 1e-3
    with pytest.raises(ValueError, match="constant_alpha"):
        B.batch_distillation(["benzene", "toluene", "ethylbenzene"], 1.0, [0.4, 0.4, 0.2],
                             {"type": "thermo", "model": "IDEAL", "P_Pa": 101325.0}, {"distilled_fraction": 0.5},
                             mode="constant_reflux", n_stages=3, reflux_ratio=2.0)


def test_constant_distillate_two_stages_semi_analytical():
    """«analytical» for still + one stage and constant alpha: x1 = xD/(alpha − (alpha−1)·xD), the still vapour
    y*(x) = alpha·x/(1+(alpha−1)·x) lies on the operating line, so R(x) = (xD − y*)/(y* − x1); W = W0·(xD−x0)/(xD−x);
    vapour boiled = W0·(xD − x0)·∫(R+1)/(xD − x)² dx (independent adaptive quadrature here)."""
    a, xd, x0, w0, xend = 2.41, 0.80, 0.5, 100.0, 0.42
    x1 = xd / (a - (a - 1) * xd)

    def reflux(x: float) -> float:
        ys = a * x / (1 + (a - 1) * x)
        return (xd - ys) / (ys - x1)
    r = _ok(B.batch_distillation(["b", "t"], w0, [x0, 1 - x0], {"type": "constant_alpha", "alpha": [a, 1.0]},
                                 {"residue_mole_fraction": xend, "component": "b"}, mode="constant_distillate", n_stages=2,
                                 distillate_mole_fraction=xd, boilup_rate_mol_s=0.02, heat_of_vaporization_J_mol=32000.0))
    f = r["final"]
    assert f["reflux_ratio_initial"] == pytest.approx(reflux(x0), rel=1e-9)
    assert f["reflux_ratio_final"] == pytest.approx(reflux(xend), rel=1e-9)
    assert f["still_mol"] == pytest.approx(w0 * (xd - x0) / (xd - xend), rel=1e-12)
    vap = w0 * (xd - x0) * quad(lambda x: (reflux(x) + 1) / (xd - x) ** 2, xend, x0, epsabs=1e-13, epsrel=1e-13)[0]
    assert f["vapour_boiled_mol"] == pytest.approx(vap, rel=1e-8)
    assert f["batch_time_s"] == pytest.approx(vap / 0.02, rel=1e-8)
    assert r["heat"]["reboiler_heat_J"] == pytest.approx(vap * 32000.0, rel=1e-8)
    # Fenske limit of the still at total reflux, and stops by reflux ratio and by time agree with each other
    lim = (xd / (1 - xd)) / a ** 2
    assert r["evidence"]["total_reflux_still_limit_mole_fraction"] == pytest.approx(lim / (1 + lim), rel=1e-10)
    by_r = _ok(B.batch_distillation(["b", "t"], w0, [x0, 1 - x0], {"type": "constant_alpha", "alpha": [a, 1.0]},
                                    {"max_reflux_ratio": reflux(xend)}, mode="constant_distillate", n_stages=2,
                                    distillate_mole_fraction=xd, boilup_rate_mol_s=0.02))["final"]
    assert by_r["still_mole_fractions"]["b"] == pytest.approx(xend, abs=1e-9)
    by_t = _ok(B.batch_distillation(["b", "t"], w0, [x0, 1 - x0], {"type": "constant_alpha", "alpha": [a, 1.0]},
                                    {"time_s": vap / 0.02}, mode="constant_distillate", n_stages=2,
                                    distillate_mole_fraction=xd, boilup_rate_mol_s=0.02))["final"]
    assert by_t["still_mole_fractions"]["b"] == pytest.approx(xend, abs=1e-6)
    # an unreachable residue is reported as such
    far = B.batch_distillation(["b", "t"], w0, [x0, 1 - x0], {"type": "constant_alpha", "alpha": [a, 1.0]},
                               {"residue_mole_fraction": 0.05, "component": "b"}, mode="constant_distillate", n_stages=2,
                               distillate_mole_fraction=xd)
    assert far["evidence"]["verified"] is False and far["warnings"]


def test_rayleigh_multicomponent_with_flash_engine():
    """CONSISTENCY ONLY (thermo route, 3 components, Raoult): balance closure, lightest component depletes, the still
    heats up, and the result is close to the constant-alpha solution with the K-ratios of the charge."""
    comps = ["benzene", "toluene", "ethylbenzene"]
    r = _ok(B.batch_distillation(comps, 100.0, [0.4, 0.4, 0.2], {"type": "thermo", "model": "IDEAL", "P_Pa": 101325.0},
                                 {"distilled_fraction": 0.3}, boilup_rate_mol_s=0.02, n_points=5))
    f, pr = r["final"], r["profile"]
    for c, z in zip(comps, (0.4, 0.4, 0.2)):
        assert (70.0 * f["still_mole_fractions"][c] + 30.0 * f["average_distillate_mole_fractions"][c]
                == pytest.approx(100 * z, abs=1e-3))
    assert f["still_mole_fractions"]["benzene"] < 0.4 < f["average_distillate_mole_fractions"]["benzene"]
    assert pr["still_T_K"][-1] > pr["still_T_K"][0] and r["heat"]["heat_of_vaporization_J_mol"] > 3e4
    y0 = [pr["instantaneous_distillate_mole_fractions"][c][0] for c in comps]
    alpha = [y / z for y, z in zip(y0, (0.4, 0.4, 0.2))]
    ca = B.rayleigh_constant_alpha(alpha, [0.4, 0.4, 0.2], 100.0, 70.0)
    assert f["still_mole_fractions"]["benzene"] == pytest.approx(ca[0], abs=0.01)


def test_distillation_refuses_missing_or_impossible_inputs():
    good = dict(components=["a", "b"], charge_mol=100.0, charge_mole_fractions=[0.5, 0.5],
                equilibrium={"type": "constant_alpha", "alpha": [2.0, 1.0]}, stop={"distilled_fraction": 0.3})
    cases = [dict(equilibrium=None), dict(equilibrium={"type": "constant_alpha"}), dict(equilibrium={"type": "thermo", "P_Pa": 1e5}),
             dict(equilibrium={"type": "thermo", "model": "IDEAL"}), dict(equilibrium={"type": "table", "x": [0, 1], "y": [0, 1]}),
             dict(stop={}), dict(stop={"distilled_fraction": 0.3, "time_s": 5.0}), dict(stop={"time_s": 10.0}),
             dict(stop={"residue_mole_fraction": 0.3}), dict(stop={"residue_mole_fraction": 0.6, "component": "a"}),
             dict(stop={"average_distillate_mole_fraction": 0.4, "component": "a"}),
             dict(mode="constant_reflux"), dict(mode="constant_reflux", n_stages=3), dict(mode="rayleigh", n_stages=3),
             dict(mode="constant_distillate", n_stages=3),
             dict(mode="constant_distillate", n_stages=2, distillate_mole_fraction=0.999999),
             dict(mode="constant_distillate", n_stages=1, distillate_mole_fraction=0.6),
             dict(charge_mole_fractions=[0.5, 0.6]), dict(charge_mol=None), dict(mode="continuous")]
    for kw in cases:
        with pytest.raises(ValueError):
            B.batch_distillation(**dict(good, **kw))


# ------------------------------------------------------------------------------------------------ tools and wrappers
def test_tools_and_wrappers_registered_consistently():
    assert [f.__name__ for f in TB.BATCH_TOOLS] == ["batch_fermentation", "batch_distillation"]
    assert [w.name for w in WB.AGENT_WRAPPERS] == [f.__name__ for f in TB.BATCH_TOOLS]
    for f in TB.BATCH_TOOLS:
        assert f.__doc__ and len(f.__doc__) > 400
    out = json.loads(WB.batch_distillation(["a", "b"], 100.0, [0.5, 0.5], {"type": "constant_alpha", "alpha": [2.0, 1.0]},
                                           {"distilled_fraction": 0.3}))
    assert out["evidence"]["verified"] and "destilación" not in out["mode"]
    out = json.loads(WB.batch_fermentation(UMD, INIT, 14 * H, None, {"substrate_conversion": 0.5}))
    assert out["target"]["reached"] and out["evidence"]["verified"]
    assert TB.batch_fermentation(UMD, INIT, 5 * H)["final"] == B.batch_fermentation(UMD, INIT, 5 * H)["final"]
