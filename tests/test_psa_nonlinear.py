"""Block L1 (M25): Skarstrom PSA cycle with non-linear (Langmuir / extended-Langmuir) isotherms against equilibrium theory.

Reference (analytical, derived 2026-11-15; no published cycle could be opened legitimately, see below):
  Equilibrium theory of the four-step cycle with pressurisation by product (instantaneous uptake, plug flow, isothermal,
  uniform pressure), the non-linear extension of Knaebel & Hill (1985) re-derived from first principles:
    feed step     Rankine-Hugoniot jump of the heavy shock into a bed of pure light gas,
                  w = u_F C_H y_F / (eps C_H y_F + rho_b q_A,F),  t_bt = L / w,
                  u_out = u_F - w rho_b (q_A,F + q_B,F - q_B(0, C_H)) / C_H  (pure product),
    pressurisation  L [eps (C_H - C_L) + rho_b (q_B(0, C_H) - q_B(0, C_L))] of pure light gas,
    purge         slowest characteristic c_A -> 0 at u_P / (eps + rho_b dq_A/dc_A|0): t_P >= L (eps + rho_b q_max b) / u_P
                  cleans the bed (sufficient bound),
    recovery      R = (u_out C_H t_F - pressurisation - u_P C_L t_P) / (u_F C_H t_F (1 - y_F)).
  The closed form is coded twice: in the engine (`psa_equilibrium_theory_nonlinear`) and independently below (`_et`).
  Published PSA cycles searched 2026-11-15 (Farooq-Ruthven-Boniface 1989, Mendes et al. 2001, Kayser-Knaebel 1989,
  Waldron-Sircar 2000): paywalled or non-isothermal with pressure drop; Chen & Ahn, Ind. Eng. Chem. Res. 62 (2023) 5286
  (accepted manuscript https://www.pure.ed.ac.uk/ws/files/330693596/ie_2022_04654u.R1.pdf) uses linearised isotherms and
  gives recoveries only in figures, so it does not test the non-linear case. No published-cycle comparison here.
Every band asserted below is the deviation MEASURED on 2026-11-15 rounded up, not a tuned tolerance."""
from __future__ import annotations

import json
import math

import pytest

from packages.dwsim_client import tools_separations as TS
from packages.separations import adsorption as AD

EPS, RHO, T, PH, L_BED = 0.4, 500.0, 300.0, 5e5, 1.0
QM = 3.0                                                  # mol/kg, monolayer capacity of the heavy gas
B_MOD = 0.01                                              # m3/mol: b c_A,F = 0.60 at the feed (moderately non-linear)
KH2 = 8e-4                                                # m3/kg, linear light gas (same as tests/test_separations.py)
_CACHE: dict = {}


def lang(qm: float, b: float) -> dict:
    return {"model": "langmuir", "q_max_mol_kg": qm, "b_m3_mol": b}


def lin(K: float) -> dict:
    return {"model": "linear", "K_m3_kg": K}


def _et(isoA: dict, isoB: dict, PL: float, yF: float, uF: float, uP: float, f: float, g: float) -> dict:
    """Independent hand-coded closed form for a Langmuir heavy gas and a linear light gas (no competition)."""
    CH, CL = PH / (AD.R_GAS * T), PL / (AD.R_GAS * T)
    qm, b, KB = isoA["q_max_mol_kg"], isoA["b_m3_mol"], isoB["K_m3_kg"]
    cAF = CH * yF
    qAF = qm * b * cAF / (1 + b * cAF)
    t_bt = L_BED * (EPS * CH * yF + RHO * qAF) / (uF * CH * yF)
    u_out = uF - (L_BED / t_bt) * RHO * (qAF + KB * CH * (1 - yF) - KB * CH) / CH
    t_pb = L_BED * (EPS + RHO * qm * b) / uP
    tF, tP = f * t_bt, g * t_pb
    gross, pr, pu = u_out * CH * tF, L_BED * (EPS + RHO * KB) * (CH - CL), uP * CL * tP
    return {"t_bt": t_bt, "t_pb": t_pb, "tF": tF, "tP": tP, "gross": gross, "pr": pr,
            "R": (gross - pr - pu) / (uF * CH * tF * (1 - yF))}


def _run(key: str, isoA: dict, isoB: dict, *, PL: float = 1e5, yF: float = 0.3, uF: float = 0.05, uP: float = 0.1,
         f: float = 0.7, g: float = 1.2, k: float = 1.0, n: int = 40, press: str = "product", check_grid: bool = False,
         max_cycles: int = 200) -> dict:
    if key not in _CACHE:
        th = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [isoA, isoB], PH, PL, yF, uF, uP, 1.0, 1.0)
        tF, tP = f * th["feed_time_to_breakthrough_s"], g * th["purge_time_bound_to_clean_bed_s"]
        comps = [{"name": "CO2", "feed_mole_fraction": yF, "k_ldf_1_s": k, "isotherm": isoA},
                 {"name": "H2", "feed_mole_fraction": 1 - yF, "k_ldf_1_s": k, "isotherm": isoB}]
        _CACHE[key] = AD.psa_skarstrom(L_BED, EPS, RHO, T, comps, "H2", PH, PL, uF, uP, 20.0, tF, 20.0, tP, press,
                                       n_cells=n, max_cycles=max_cycles, check_grid=check_grid)
    return _CACHE[key]


def _pc(r: dict, what: str) -> float:
    return r["per_cycle_mol_m2"][what]["H2"]


# ------------------------------------------------------------------------------------------------- closed form itself
def test_closed_form_matches_independent_code_and_reduces_to_linear_theory():
    et = _et(lang(QM, B_MOD), lin(KH2), 1e5, 0.3, 0.05, 0.1, 0.7, 1.2)
    th = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [lang(QM, B_MOD), lin(KH2)], PH, 1e5, 0.3, 0.05, 0.1,
                                             et["tF"], et["tP"])
    json.dumps(th)
    assert th["feed_shock_admissible"]
    assert abs(th["feed_time_to_breakthrough_s"] - et["t_bt"]) < 1e-9 * et["t_bt"]
    assert abs(th["purge_time_bound_to_clean_bed_s"] - et["t_pb"]) < 1e-6 * et["t_pb"]   # slope at origin by a 1e-9 step
    assert abs(th["light_recovery_this_cycle"] - et["R"]) < 1e-7
    # linear isotherms: identical to the Knaebel-Hill-type closed form already in the module
    KA = 0.0152
    lt = AD.psa_equilibrium_theory(L_BED, EPS, RHO, KA, KH2, 5.0, 0.3, 0.05, 0.1)
    tn = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [lin(KA), lin(KH2)], PH, 1e5, 0.3, 0.05, 0.1,
                                             0.7 * lt["feed_time_to_breakthrough_s"], 1.2 * lt["purge_time_to_clean_bed_s"])
    r_lin = AD.psa_equilibrium_recovery(lt["beta"], 5.0, 0.3, 0.7, 1.2)
    assert abs(tn["light_recovery_this_cycle"] - r_lin) < 1e-12
    assert abs(tn["feed_time_to_breakthrough_s"] - lt["feed_time_to_breakthrough_s"]) < 1e-9
    assert abs(tn["product_to_feed_molar_ratio_during_feed_step"] - lt["product_to_feed_molar_ratio_during_feed_step"]) < 1e-12
    # more favourable isotherm (larger b) -> later breakthrough at the same capacity (monotonicity)
    t1 = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [lang(QM, 0.02), lin(KH2)], PH, 1e5, 0.3, 0.05, 0.1, 1, 1)
    assert t1["feed_time_to_breakthrough_s"] > th["feed_time_to_breakthrough_s"]


def test_closed_form_domain_and_refusals():
    # inverted roles (the "heavy" gas adsorbs less): the feed front is not an admissible shock -> no closed form
    th = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [lin(KH2), lang(QM, B_MOD)], PH, 1e5, 0.3, 0.05, 0.1, 10, 1e4)
    assert not th["feed_shock_admissible"] and th["light_recovery_this_cycle"] is None
    assert "choque" in th["light_recovery_this_cycle_note"]
    # purge below the cleaning bound, feed past breakthrough
    th = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [lang(QM, B_MOD), lin(KH2)], PH, 1e5, 0.3, 0.05, 0.1, 100, 50)
    assert th["light_recovery_this_cycle"] is None and "purga" in th["light_recovery_this_cycle_note"]
    th = AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [lang(QM, B_MOD), lin(KH2)], PH, 1e5, 0.3, 0.05, 0.1, 400, 500)
    assert th["light_recovery_this_cycle"] is None and "ruptura" in th["light_recovery_this_cycle_note"]
    with pytest.raises(ValueError, match="Langmuir"):
        AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [{"model": "freundlich", "K_F": 1, "n": 2}, lin(KH2)], PH, 1e5,
                                            0.3, 0.05, 0.1, 1, 1)
    with pytest.raises(ValueError, match="q_max_mol_kg"):
        AD.psa_equilibrium_theory_nonlinear(L_BED, EPS, RHO, T, [{"model": "langmuir", "b_m3_mol": 0.01}, lin(KH2)], PH, 1e5,
                                            0.3, 0.05, 0.1, 1, 1)


# ------------------------------------------------------------------------------------------------- simulator vs theory
def test_langmuir_cycle_vs_equilibrium_theory():
    r = _run("main", lang(QM, B_MOD), lin(KH2), check_grid=True)
    json.dumps(r)
    et = _et(lang(QM, B_MOD), lin(KH2), 1e5, 0.3, 0.05, 0.1, 0.7, 1.2)
    ev = r["evidence"]
    assert ev["verified"], ev
    assert ev["cycle_mass_balance_rel_error"] < 1e-5 and ev["cyclic_steady_state_residual"] < 1e-4
    assert abs(r["equilibrium_theory"]["light_recovery_this_cycle"] - et["R"]) < 1e-7
    assert abs(r["product_recovery"] - et["R"]) < 2e-4               # measured 1.34e-4 (k_LDF = 1/s, 40 cells), R = 0.1291
    assert abs(ev["equilibrium_theory_recovery_abs_diff"] - abs(r["product_recovery"] - et["R"])) < 1e-7
    assert r["product_purity_mole_fraction"] > 1 - 1e-6                # ET: pure product
    assert abs(_pc(r, "gross_product") / et["gross"] - 1) < 4e-3       # measured -3.4e-3 (finite LDF, see k test)
    assert abs(_pc(r, "pressurization_gas") / et["pr"] - 1) < 3e-2     # measured -2.5e-2 (finite LDF, see k test)
    assert r["validation_status"].startswith("comparable") and "Langmuir" in r["validation_status"]
    assert not any("NO VALIDADO" in w or "fuera del dominio" in w for w in r["warnings"])
    # identities: all the heavy gas fed leaves in the two exhausts; light-gas balance per cycle
    pc = r["per_cycle_mol_m2"]
    a_in, a_out = pc["feed"]["CO2"], pc["blowdown_exhaust"]["CO2"] + pc["purge_exhaust"]["CO2"]
    assert abs(a_out / a_in - 1) < 1e-5
    b_in = pc["feed"]["H2"] + pc["pressurization_gas"]["H2"] + pc["purge_gas_from_product"]["H2"]
    b_out = pc["gross_product"]["H2"] + pc["blowdown_exhaust"]["H2"] + pc["purge_exhaust"]["H2"]
    assert abs(b_out / b_in - 1) < 1e-5


def test_grid_refinement_converges_to_equilibrium_theory():
    et = _et(lang(QM, B_MOD), lin(KH2), 1e5, 0.3, 0.05, 0.1, 0.7, 1.2)
    d10 = abs(_run("n10", lang(QM, B_MOD), lin(KH2), n=10)["product_recovery"] - et["R"])
    d20 = abs(_run("n20", lang(QM, B_MOD), lin(KH2), n=20)["product_recovery"] - et["R"])
    d40 = abs(_run("main", lang(QM, B_MOD), lin(KH2), check_grid=True)["product_recovery"] - et["R"])
    assert d40 < d20 < d10                                             # measured 6.28e-4, 2.97e-4, 1.34e-4
    assert 1.6 < d10 / d20 < 3.0 and 1.6 < d20 / d40 < 3.0             # first order in the cell size (2.11, 2.22)
    # (outside the test, for time: 80 cells gave 5.8e-5, ratio 2.3)


def test_ldf_rate_limit_approaches_equilibrium_theory():
    et = _et(lang(QM, B_MOD), lin(KH2), 1e5, 0.3, 0.05, 0.1, 0.7, 1.2)
    runs = [_run(f"k{k}", lang(QM, B_MOD), lin(KH2), k=k) if k != 1.0 else _run("main", lang(QM, B_MOD), lin(KH2), check_grid=True)
            for k in (0.125, 0.25, 1.0)]
    dg = [abs(_pc(r, "gross_product") - et["gross"]) / et["gross"] for r in runs]
    dp = [abs(_pc(r, "pressurization_gas") - et["pr"]) / et["pr"] for r in runs]
    assert dg[0] > dg[1] > dg[2] and dp[0] > dp[1] > dp[2]             # measured 2.38e-2 > 1.29e-2 > 3.4e-3; 0.184 > 0.099 > 0.025
    # both deviations fall as 1/k (k * deviation nearly constant): measured k dg = 2.98e-3, 3.24e-3, 3.39e-3
    # (k = 4/s, outside the test for time, gave dg = 9.6e-4, dp = 6.3e-3, k dg = 3.8e-3)
    assert all(2.5e-3 < k * d < 4e-3 for k, d in zip((0.125, 0.25, 1.0), dg))
    assert all(1.8e-2 < k * d < 3e-2 for k, d in zip((0.125, 0.25, 1.0), dp))   # measured 2.30e-2, 2.48e-2, 2.51e-2
    # the recovery (a difference of those terms, whose LDF lags cancel) is limited by the grid: within 2e-4 for k >= 0.125
    assert max(abs(r["product_recovery"] - et["R"]) for r in runs) < 2e-4   # measured 9.2e-5 .. 1.34e-4


def test_inert_light_gas_and_extended_langmuir_and_strong_nonlinearity():
    # light gas not adsorbed (K = 0)
    r = _run("inert", lang(QM, B_MOD), lin(0.0), n=20)
    et = _et(lang(QM, B_MOD), lin(0.0), 1e5, 0.3, 0.05, 0.1, 0.7, 1.2)
    assert abs(r["product_recovery"] - et["R"]) < 1e-4                  # measured 8.1e-5 (40 cells: 5.2e-7), R = 0.1784
    # extended Langmuir, equal q_max (competition for the same sites; light gas almost linear)
    iso_b = lang(QM, KH2 / QM)
    r = _run("ext", lang(QM, B_MOD), iso_b)
    json.dumps(r)
    th = r["equilibrium_theory"]
    assert th["feed_shock_admissible"] and r["validation_status"].startswith("comparable")
    assert abs(r["product_recovery"] - th["light_recovery_this_cycle"]) < 1e-5     # measured 2.9e-6 (R = 0.1352)
    assert r["product_purity_mole_fraction"] > 1 - 1e-6
    # strongly non-linear (b c_A,F = 1.81), pressure ratio 10, shorter cycle at higher velocities
    r = _run("strong", lang(QM, 0.05), lin(KH2), PL=5e4, yF=0.15, uF=0.15, uP=0.6, n=20)
    th = r["equilibrium_theory"]
    assert th["heavy_loading_at_feed_mol_kg"] / QM > 0.6               # more than 60 % of the monolayer at the feed
    assert r["validation_status"].startswith("comparable")
    assert abs(r["product_recovery"] - th["light_recovery_this_cycle"]) < 1e-4    # measured 7.3e-5 (40 cells: 3.7e-5), R = 0.4639
    assert r["product_purity_mole_fraction"] > 1 - 1e-6


def test_outside_the_closed_form_stays_not_validated():
    # feed past the equilibrium breakthrough time: impure product, labelled NO VALIDADO with a warning
    r = _run("over", lang(QM, B_MOD), lin(KH2), f=1.15, n=12, max_cycles=6)
    json.dumps(r)
    assert r["equilibrium_theory"]["light_recovery_this_cycle"] is None
    assert r["validation_status"].startswith("NO VALIDADO")
    assert any("fuera del dominio" in w for w in r["warnings"])
    assert r["product_purity_mole_fraction"] < 0.95                    # measured 0.915: the shock has left the bed
    # pressurisation with feed: no closed form
    r = _run("pfeed", lang(QM, B_MOD), lin(KH2), press="feed", n=12, max_cycles=30)
    assert r["validation_status"].startswith("NO VALIDADO") and r["equilibrium_theory"]["light_recovery_this_cycle"] is None
    # Freundlich: still not validated
    comps = [{"name": "CO2", "feed_mole_fraction": 0.3, "k_ldf_1_s": 0.5, "isotherm": {"model": "freundlich", "K_F": 0.2, "n": 2.0}},
             {"name": "H2", "feed_mole_fraction": 0.7, "k_ldf_1_s": 0.5, "isotherm": lin(KH2)}]
    r = AD.psa_skarstrom(L_BED, EPS, RHO, T, comps, "H2", PH, 1e5, 0.05, 0.1, 20.0, 60.0, 20.0, 60.0, "product", n_cells=10,
                         max_cycles=5, check_grid=False)
    json.dumps(r)
    assert r["validation_status"].startswith("NO VALIDADO") and "Freundlich" in r["validation_status"]


def test_tool_docstring_no_longer_claims_the_limit():
    doc = TS.psa_cycle.__doc__
    assert "Langmuir" in doc and "con isotermas no lineales queda NO VALIDADO" not in doc
    assert math.isfinite(QM)
