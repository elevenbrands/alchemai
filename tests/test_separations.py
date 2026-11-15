"""Block membads: membrane gas permeation, reverse osmosis, fixed-bed adsorption (LDF), LUB scale-up and Skarstrom PSA.

Reference cases (read on 2026-11-14):
  [G13] Geankoplis, Transport Processes and Unit Operations 3rd ed., chap. 13, examples 13.4-1, 13.4-2, 13.4-3, 13.5-1,
        13.6-1, 13.9-1, 13.9-2, 13.10-1, as solved in the FOSSEE Python Textbook Companion (inputs and printed answers):
        https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter13.ipynb
  [G12] Same book and companion, chap. 12, example 12.3-1 (scale-up of a laboratory adsorption column, LUB):
        https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter12.ipynb
  [B06] H. Binous, "Gas permeation computations with Mathematica", Chem. Eng. Educ. 40(2) 140-144 (2006), air case of
        Geankoplis 13.4-2 in the four flow patterns and the ternary of 13.5-1:
        https://journals.flvc.org/cee/article/download/122538/121524
  Analytical: Anzelius/Thomas exact solution (linear isotherm, LDF, plug flow); Klinkenberg (1954) approximation; constant
  pattern of a Langmuir front with LDF; equilibrium theory of the Skarstrom cycle with linear isotherms (re-derived in
  `psa_equilibrium_recovery`, the f = gamma = 1 case is the Knaebel-Hill 1985 recovery).
Every band asserted below is the deviation MEASURED on 2026-11-14 rounded up, not a tuned tolerance."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from packages.dwsim_client import tools_separations as TS
from packages.separations import adsorption as AD
from packages.separations import membrane as MB

STP = 1.0 / 22414.0            # mol per cm3(STP)
CMHG = 1333.22368              # Pa per cmHg
ATM = 101325.0


def perm(P_cgs: float, t_cm: float) -> float:
    """cm3(STP) cm/(s cm2 cmHg) over a thickness in cm -> mol/(m2 s Pa)."""
    return P_cgs / t_cm * STP / (1e-4 * CMHG)


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def dumps(r: dict) -> str:
    s = json.dumps(r, default=float, ensure_ascii=False, allow_nan=False)
    assert isinstance(s, str)
    return s


# Geankoplis 13.4-2 / 13.6-1 and Binous: air, alpha = 10, P_O2 = 500e-10, t = 2.54e-3 cm, 190/19 cmHg, 1e6 cm3(STP)/s, theta 0.2
AIR_Q = {"O2": perm(500e-10, 2.54e-3), "N2": perm(50e-10, 2.54e-3)}
AIR = dict(feed_flow_mol_s=1e6 * STP, feed_composition={"O2": 0.209, "N2": 0.791}, feed_pressure_Pa=190 * CMHG,
           permeate_pressure_Pa=19 * CMHG, permeance_mol_m2_s_Pa=AIR_Q, stage_cut=0.2)
# (pattern, area cm2, y_p O2, x_o O2) from [B06]; complete mixing also [G13] 13.4-2: 3.231e8 cm2, 0.5068, 0.1345
BINOUS = [("complete_mixing", 3.228e8, 0.5067, 0.1346), ("cross_flow", 2.899e8, 0.5688, 0.1190),
          ("countercurrent", 2.859e8, 0.5763, 0.1171), ("cocurrent", 2.955e8, 0.5584, 0.1216)]


# ------------------------------------------------------------------------------------------------- gas permeation
@pytest.mark.parametrize("pattern,area,yp,xo", BINOUS)
def test_air_four_patterns_vs_binous(pattern, area, yp, xo):
    r = MB.gas_permeation(flow_pattern=pattern, **AIR)
    dumps(r)
    assert r["evidence"]["verified"], r["evidence"]
    # measured: area within 0.11 % (CM) and 0.03 % (others); compositions within 1e-4 (rounding of the paper)
    assert rel(r["area_m2"] * 1e4, area) < 0.0015
    assert abs(r["permeate"]["mole_fractions"]["O2"] - yp) < 1.5e-4
    assert abs(r["retentate"]["mole_fractions"]["O2"] - xo) < 1.5e-4
    assert r["evidence"]["component_balance_max_rel_error"] < 1e-9


def test_geankoplis_13_4_2_and_13_6_1():
    cm = MB.gas_permeation(flow_pattern="complete_mixing", **AIR)
    # [G13] 13.4-2: yp 0.5068, xo 0.1345, Am 3.231e8 cm2 (measured 0.01 %)
    assert abs(cm["permeate"]["mole_fractions"]["O2"] - 0.5068) < 6e-5
    assert abs(cm["retentate"]["mole_fractions"]["O2"] - 0.1345) < 6e-5
    assert rel(cm["area_m2"] * 1e4, 3.231e8) < 3e-4
    assert cm["evidence"]["binary_closed_form_quadratic_abs_diff"] < 1e-12
    cf = MB.gas_permeation(flow_pattern="cross_flow", **AIR)
    # [G13] 13.6-1 (companion integrates 5 interpolated points): xo 0.119, yp 0.569, Am 2.9135e8 cm2 -> measured -0.48 %
    assert abs(cf["retentate"]["mole_fractions"]["O2"] - 0.119) < 1e-3
    assert abs(cf["permeate"]["mole_fractions"]["O2"] - 0.569) < 1e-3
    assert rel(cf["area_m2"] * 1e4, 2.9135e8) < 0.006


def test_geankoplis_13_4_1_design_by_reject_composition():
    q = {"A": perm(50e-10, 2.54e-3), "B": perm(5e-10, 2.54e-3)}
    r = MB.gas_permeation(1e4 * STP, {"A": 0.5, "B": 0.5}, 80 * CMHG, 20 * CMHG, "complete_mixing", permeance_mol_m2_s_Pa=q,
                          target={"quantity": "retentate_mole_fraction", "component": "A", "value": 0.25})
    dumps(r)
    assert r["mode"] == "design" and r["evidence"]["verified"]
    assert rel(r["area_m2"] * 1e4, 2.734e8) < 3e-4                     # [G13] 2.734e8 cm2; measured 0.01 %
    assert abs(r["retentate"]["mole_fractions"]["A"] - 0.25) < 1e-9
    # 13.4-3: minimum reject composition at total permeation, 0.1932 (xf 0.50) and 0.28 (xf 0.65)
    assert abs(r["limits"]["complete_mixing_retentate_at_total_permeation"]["A"] - 0.1932) < 5e-5
    r65 = MB.gas_permeation(1e4 * STP, {"A": 0.65, "B": 0.35}, 80 * CMHG, 20 * CMHG, "complete_mixing", permeance_mol_m2_s_Pa=q,
                            stage_cut=0.3)
    assert abs(r65["limits"]["complete_mixing_retentate_at_total_permeation"]["A"] - 0.28) < 5e-4


def test_geankoplis_13_5_1_ternary_complete_mixing():
    q = {"A": perm(200e-10, 2.54e-3), "B": perm(50e-10, 2.54e-3), "C": perm(25e-10, 2.54e-3)}
    r = MB.gas_permeation(1e4 * STP, {"A": 0.25, "B": 0.55, "C": 0.20}, 300 * CMHG, 30 * CMHG, "complete_mixing",
                          permeance_mol_m2_s_Pa=q, stage_cut=0.25)
    # [G13] yp 0.4553/0.4503/0.0944, xo 0.1816/0.5832/0.2352, Am 3.5418e6 cm2; [B06] Am 3.54176e6 (measured < 1e-5)
    for k, v in {"A": 0.4553, "B": 0.4503, "C": 0.0944}.items():
        assert abs(r["permeate"]["mole_fractions"][k] - v) < 6e-5
    for k, v in {"A": 0.1816, "B": 0.5832, "C": 0.2352}.items():
        assert abs(r["retentate"]["mole_fractions"][k] - v) < 6e-5
    assert rel(r["area_m2"] * 1e4, 3.54176e6) < 1e-5
    # rating with that area in plug-flow patterns: more efficient, countercurrent best (same ranking as [B06] for air)
    cuts = {p: MB.gas_permeation(1e4 * STP, {"A": 0.25, "B": 0.55, "C": 0.20}, 300 * CMHG, 30 * CMHG, p, permeance_mol_m2_s_Pa=q,
                                 area_m2=r["area_m2"]) for p in ("cross_flow", "cocurrent", "countercurrent")}
    yA = {p: c["permeate"]["mole_fractions"]["A"] for p, c in cuts.items()}
    assert yA["countercurrent"] > yA["cross_flow"] > yA["cocurrent"] > r["permeate"]["mole_fractions"]["A"]
    assert all(c["evidence"]["verified"] for c in cuts.values())


def test_gas_permeation_identities():
    areas = {p: MB.gas_permeation(flow_pattern=p, **AIR)["area_m2"] for p in MB.PATTERNS}
    assert areas["countercurrent"] < areas["cross_flow"] < areas["cocurrent"] < areas["complete_mixing"]
    # vanishing area: permeate = local permeate of the feed for every pattern
    tiny = 1e-6 * areas["cross_flow"]
    for p in MB.PATTERNS:
        r = MB.gas_permeation(1e6 * STP, {"O2": 0.209, "N2": 0.791}, 190 * CMHG, 19 * CMHG, p, permeance_mol_m2_s_Pa=AIR_Q,
                              area_m2=tiny)
        y0 = r["limits"]["permeate_at_zero_stage_cut"]["O2"]
        assert abs(r["permeate"]["mole_fractions"]["O2"] - y0) < 1e-5
        assert r["permeate"]["mole_fractions"]["O2"] <= r["limits"]["permeate_ceiling_by_pressure_ratio"]["O2"] + 1e-12
    # equal permeances: no separation at any cut
    same = MB.gas_permeation(1.0, {"A": 0.3, "B": 0.7}, 5e5, 1e5, "countercurrent", permeance_mol_m2_s_Pa={"A": 1e-8, "B": 1e-8},
                             stage_cut=0.5)
    assert abs(same["permeate"]["mole_fractions"]["A"] - 0.3) < 1e-8
    # rating <-> design round trip
    cc = MB.gas_permeation(flow_pattern="countercurrent", **AIR)
    back = MB.gas_permeation(1e6 * STP, {"O2": 0.209, "N2": 0.791}, 190 * CMHG, 19 * CMHG, "countercurrent",
                             permeance_mol_m2_s_Pa=AIR_Q, area_m2=cc["area_m2"])
    assert abs(back["stage_cut"] - 0.2) < 1e-7
    # closed binary quadratic against the brute-force root of y/(1-y) = alpha (x - g y)/((1-x) - g (1-y))
    yq = MB.binary_complete_mixing_closed_form(0.209, 10.0, 10.0, 0.2)
    x = (0.209 - 0.2 * yq) / 0.8
    assert abs(yq / (1 - yq) - 10.0 * (x - 0.1 * yq) / ((1 - x) - 0.1 * (1 - yq))) < 1e-10
    # unattainable purity -> feasible False with the attainable range
    bad = MB.gas_permeation(1e6 * STP, {"O2": 0.209, "N2": 0.791}, 190 * CMHG, 19 * CMHG, "cross_flow", permeance_mol_m2_s_Pa=AIR_Q,
                            target={"quantity": "permeate_mole_fraction", "component": "O2", "value": 0.9})
    assert bad["feasible"] is False and not bad["evidence"]["verified"]
    dumps(bad)


def test_gas_permeation_refuses_missing_data():
    base = dict(AIR)
    with pytest.raises(ValueError, match="flow_pattern"):
        MB.gas_permeation(**base)
    with pytest.raises(ValueError, match="permeancias"):
        MB.gas_permeation(1.0, {"A": 0.5, "B": 0.5}, 5e5, 1e5, "cross_flow", stage_cut=0.2)
    with pytest.raises(ValueError, match="exactamente uno"):
        MB.gas_permeation(1.0, {"A": 0.5, "B": 0.5}, 5e5, 1e5, "cross_flow", permeance_mol_m2_s_Pa={"A": 1e-8, "B": 1e-9})
    with pytest.raises(ValueError, match="Falta feed_flow_mol_s"):
        MB.gas_permeation(None, {"A": 0.5, "B": 0.5}, 5e5, 1e5, "cross_flow", permeance_mol_m2_s_Pa={"A": 1e-8, "B": 1e-9},
                          stage_cut=0.2)
    with pytest.raises(ValueError, match=r"permeance_mol_m2_s_Pa\['B'\]"):
        MB.gas_permeation(1.0, {"A": 0.5, "B": 0.5}, 5e5, 1e5, "cross_flow", permeance_mol_m2_s_Pa={"A": 1e-8}, stage_cut=0.2)


# ------------------------------------------------------------------------------------------------- reverse osmosis
def test_geankoplis_13_9_1_osmotic_pressure():
    c = 0.1 * 997.0 * 58.45e-3                 # 0.1 mol NaCl / kg water at 997 kg/m3 -> kg/m3
    pi = MB.osmotic_pressure_Pa(c, 58.45e-3, 2, 298.15) / ATM
    assert abs(pi - 4.88) < 0.005              # [G13] 4.88 atm (van 't Hoff); measured 4.878


def test_geankoplis_13_10_1_and_13_9_2_local_ro():
    # 13.10-1: c1 2.5 kg/m3, Aw 4.81e-4 kg/(s m2 atm), As 4.42e-7 m/s, dP 27.2 atm, pi1 = 1.97 atm (table) -> phi
    phi = 1.97 * ATM / MB.osmotic_pressure_Pa(2.5, 58.45e-3, 2, 298.15)
    r = MB.reverse_osmosis(2.5, 58.45e-3, 2, 298.15, 28.2 * ATM, ATM, 4.81e-4 / ATM, 4.42e-7, osmotic_coefficient=phi,
                           permeate_density_kg_m3=997.0)
    dumps(r)
    loc = r["local_at_feed"]
    assert r["evidence"]["verified"]
    assert rel(loc["water_flux_kg_m2_s"], 1.2174e-2) < 6e-4          # [G13] 1.2174e-2; measured 0.04 % (book: c2 assumed 0.1)
    assert rel(loc["solute_flux_kg_m2_s"], 1.0664e-6) < 1e-4         # [G13] 1.0664e-6; measured < 0.01 %
    assert abs(loc["rejection_observed"] - 0.965) < 5e-4             # [G13] 0.965
    # 13.9-2 run forward with the constants the book extracted: measured Nw 9.5712e-3 (1.92e-8 m3/s * 997 / 2e-3 m2),
    # c2 = 0.39 kg/m3, R = 0.961; pi1 = 7.80 atm at 10 kg/m3 -> phi
    phi2 = 7.80 * ATM / MB.osmotic_pressure_Pa(10.0, 58.45e-3, 2, 298.15)
    r2 = MB.reverse_osmosis(10.0, 58.45e-3, 2, 298.15, 55.42 * ATM, ATM, 2.039e-4 / ATM, 3.8959e-7, osmotic_coefficient=phi2,
                            permeate_density_kg_m3=997.0)
    l2 = r2["local_at_feed"]
    assert rel(l2["water_flux_kg_m2_s"], 1.92e-8 * 997.0 / 2e-3) < 6e-4   # measured 0.03 %
    assert abs(l2["permeate_concentration_kg_m3"] - 0.39) < 2e-3
    assert abs(l2["rejection_observed"] - 0.961) < 5e-4


def test_ro_module_identities():
    common = (2.5, 58.45e-3, 2, 298.15, 28.2 * ATM, ATM, 4.81e-4 / ATM)
    d = MB.reverse_osmosis(*common, 4.42e-7, feed_flow_m3_s=1e-3, recovery=0.5, mass_transfer_coefficient_m_s=3e-5,
                           pump_efficiency=0.8, energy_recovery_efficiency=0.9, permeate_density_kg_m3=997.0)
    dumps(d)
    assert d["evidence"]["verified"] and abs(d["recovery"] - 0.5) < 1e-9
    # rating with the designed area returns the recovery
    r = MB.reverse_osmosis(*common, 4.42e-7, feed_flow_m3_s=1e-3, area_m2=d["area_m2"], mass_transfer_coefficient_m_s=3e-5,
                           permeate_density_kg_m3=997.0)
    assert abs(r["recovery"] - 0.5) < 1e-6
    # salt balance of the module
    assert abs(1e-3 * 2.5 - d["permeate"]["flow_m3_s"] * d["permeate"]["concentration_kg_m3"]
               - d["retentate"]["flow_m3_s"] * d["retentate"]["concentration_kg_m3"]) < 1e-12
    # polarisation lowers flux and rejection
    nop = MB.reverse_osmosis(*common, 4.42e-7, permeate_density_kg_m3=997.0)["local_at_feed"]
    pol = MB.reverse_osmosis(*common, 4.42e-7, mass_transfer_coefficient_m_s=3e-5, permeate_density_kg_m3=997.0)["local_at_feed"]
    assert pol["water_flux_kg_m2_s"] < nop["water_flux_kg_m2_s"] and pol["rejection_observed"] < nop["rejection_observed"]
    assert pol["polarization_modulus"] > 1.0
    # complete rejection: area equals its closed-form integral, and the osmotic limit 1 - pi_f/dP
    full = MB.reverse_osmosis(35.0, 58.45e-3, 2, 298.15, 40 * ATM, ATM, 4.81e-4 / ATM, 0.0, feed_flow_m3_s=1e-3, recovery=0.2,
                              permeate_density_kg_m3=997.0)
    assert full["evidence"]["area_closed_form_complete_rejection_rel_diff"] < 1e-6
    assert full["permeate"]["concentration_kg_m3"] == 0.0
    over = MB.reverse_osmosis(35.0, 58.45e-3, 2, 298.15, 40 * ATM, ATM, 4.81e-4 / ATM, 0.0, feed_flow_m3_s=1e-3, recovery=0.6,
                              permeate_density_kg_m3=997.0)
    assert over["feasible"] is False
    lim = over["max_recovery_osmotic_limit_complete_rejection"]
    assert abs(over["max_recovery_reached"] - lim) < 1e-3              # measured 1e-4 at 1e4 x the inlet area
    dumps(over)
    # specific energy: ideal hydraulic = dP / recovery; ERD reduces it
    assert abs(d["specific_energy"]["hydraulic_ideal_pump_J_m3"] - 27.2 * ATM / 0.5) < 1e-6 * 27.2 * ATM
    assert d["specific_energy"]["specific_energy_J_m3"] < 27.2 * ATM / (0.8 * 0.5)


def test_ro_refuses_missing_data():
    with pytest.raises(ValueError, match="water_permeability_kg_m2_s_Pa"):
        MB.reverse_osmosis(2.5, 58.45e-3, 2, 298.15, 28.2 * ATM, ATM, None, 4.42e-7)
    with pytest.raises(ValueError, match="ions_per_formula"):
        MB.reverse_osmosis(2.5, 58.45e-3, None, 298.15, 28.2 * ATM, ATM, 4.81e-4 / ATM, 4.42e-7)
    with pytest.raises(ValueError, match="feed_flow_m3_s"):
        MB.reverse_osmosis(2.5, 58.45e-3, 2, 298.15, 28.2 * ATM, ATM, 4.81e-4 / ATM, 4.42e-7, recovery=0.5)


# ------------------------------------------------------------------------------------------------- fixed bed
def _linear_bed(xi: float, n: int, **kw) -> dict:
    L, u, eps, rho_b, K = 1.0, 0.1, 0.4, 500.0, 0.02
    k = xi * u / (K * rho_b * L)
    comps = [{"name": "A", "feed_concentration_mol_m3": 1.0, "isotherm": {"model": "linear", "K_m3_kg": K}, "k_ldf_1_s": k}]
    return AD.fixed_bed_breakthrough(L, u, eps, rho_b, comps, 0.05, n_cells=n, **kw)


@pytest.mark.parametrize("xi,n,band", [(5.0, 60, 5e-4), (20.0, 60, 1.5e-3), (100.0, 100, 4e-3)])
def test_linear_ldf_vs_anzelius_exact(xi, n, band):
    r = _linear_bed(xi, n)
    dumps(r)
    e = r["evidence"]["per_component"]["A"]
    assert r["evidence"]["verified"], r["evidence"]
    assert e["anzelius_exact_max_abs_diff"] < band            # measured 1.6e-4, 6.6e-4, 1.8e-3
    assert e["mass_balance_rel_error"] < 1e-10
    assert e["stoichiometric_time_rel_diff"] < 1e-3            # integral of (1 - c/c0) = L (eps + rho_b K)/u
    c = r["components"][0]
    assert abs(c["stoichiometric_time_s"] - 1.0 * (0.4 + 500.0 * 0.02) / 0.1) < 1e-9
    assert 0 < c["bed_usage_at_breakthrough"] < 1 and c["length_of_unused_bed_m"] > 0


def test_klinkenberg_against_exact():
    for xi, tau in ((5, 5), (20, 15), (20, 25), (100, 90), (100, 110)):
        assert abs(AD.klinkenberg(xi, tau) - AD.anzelius_exact(xi, tau)) < 1.2e-3   # measured max 1.1e-3 at xi = 5
    assert AD.anzelius_exact(20, 0.0) == 0.0 and abs(AD.anzelius_exact(1e-12, 3.0) - 1.0) < 1e-9


def test_langmuir_constant_pattern_and_dispersion():
    comps = [{"name": "A", "feed_concentration_mol_m3": 1.0, "isotherm": {"model": "langmuir", "q_max_mol_kg": 2.0, "b_m3_mol": 4.0},
              "k_ldf_1_s": 0.004}]
    r = AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, comps, 0.05, n_cells=40)
    dumps(r)
    e = r["evidence"]["per_component"]["A"]
    assert r["evidence"]["verified"], r["evidence"]
    assert e["constant_pattern_width_rel_diff"] < 0.02          # closed form of the LDF constant pattern; measured 1.1 %
    assert abs(e["langmuir_separation_factor"] - 0.2) < 1e-12
    c = r["components"][0]
    assert abs(c["equilibrium_loading_at_feed_mol_kg"] - 1.6) < 1e-12
    assert c["breakthrough_time_s"] < c["stoichiometric_time_s"] < c["saturation_time_s"]
    # stronger dispersion widens the zone (lower usage at breakthrough)
    rd = AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, comps, 0.05, axial_dispersion_m2_s=1e-3, n_cells=40, check_grid=False)
    assert rd["components"][0]["bed_usage_at_breakthrough"] < c["bed_usage_at_breakthrough"]
    assert rd["evidence"]["per_component"]["A"]["mass_balance_rel_error"] < 1e-10
    assert not rd["evidence"]["verified"]                       # grid check skipped -> not verified, and said so


def test_extended_langmuir_roll_up_and_freundlich():
    comps = [{"name": "A", "feed_concentration_mol_m3": 1.0, "isotherm": {"model": "langmuir", "q_max_mol_kg": 2.0, "b_m3_mol": 4.0},
              "k_ldf_1_s": 0.002},
             {"name": "B", "feed_concentration_mol_m3": 1.0, "isotherm": {"model": "langmuir", "q_max_mol_kg": 2.0, "b_m3_mol": 0.5},
              "k_ldf_1_s": 0.002}]
    r = AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, comps, 0.05, n_cells=20, check_grid=False)
    dumps(r)
    a, b = r["components"]
    assert b["breakthrough_time_s"] < a["breakthrough_time_s"]
    assert b["roll_up_max_c_over_c0"] > 1.2                     # the weak component is displaced
    assert abs(a["equilibrium_loading_at_feed_mol_kg"] - 2 * 4 / 5.5) < 1e-12
    for e in r["evidence"]["per_component"].values():
        assert e["mass_balance_rel_error"] < 1e-10 and e["stoichiometric_time_rel_diff"] < 1e-3
    fr = AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, [{"name": "A", "feed_concentration_mol_m3": 1.0, "k_ldf_1_s": 0.002,
                                                           "isotherm": {"model": "freundlich", "K_F": 1.5, "n": 2.0}}],
                                   0.05, n_cells=20, check_grid=False)
    e = fr["evidence"]["per_component"]["A"]
    assert e["mass_balance_rel_error"] < 1e-10 and e["stoichiometric_time_rel_diff"] < 1e-3
    assert any("Freundlich" in s for s in fr["assumptions"])


def test_fixed_bed_refuses_missing_data():
    comps = [{"name": "A", "feed_concentration_mol_m3": 1.0, "isotherm": {"model": "linear", "K_m3_kg": 0.02}}]
    with pytest.raises(ValueError, match="k_ldf_1_s"):
        AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, comps, 0.05)
    with pytest.raises(ValueError, match="breakthrough_fraction"):
        AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, [dict(comps[0], k_ldf_1_s=0.1)])
    with pytest.raises(ValueError, match="q_max_mol_kg"):
        AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, 500.0, [{"name": "A", "feed_concentration_mol_m3": 1.0, "k_ldf_1_s": 0.1,
                                                          "isotherm": {"model": "langmuir", "b_m3_mol": 1.0}}], 0.05)
    with pytest.raises(ValueError, match="bulk_density_kg_m3"):
        AD.fixed_bed_breakthrough(1.0, 0.1, 0.4, None, [dict(comps[0], k_ldf_1_s=0.1)], 0.05)


# ------------------------------------------------------------------------------------------------- LUB
G12_T_H = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.2, 6.5, 6.8]
G12_X = [0.0, 0.0, 0.002, 0.030, 0.155, 0.396, 0.658, 0.903, 0.933, 0.975, 0.993]


def test_geankoplis_12_3_1_lub():
    rate = 754.0 * 0.00115e-3 * 600e-6           # 754 cm3/s air at 0.00115 g/cm3, 600 ppm alcohol -> kg/s
    r = AD.lub_scaleup([t * 3600 for t in G12_T_H], G12_X, 0.14, 0.01, new_breakthrough_time_s=6 * 3600,
                       solute_feed_rate_kg_s=rate, adsorbent_mass_kg=0.0792)
    dumps(r)
    lab, up = r["laboratory"], r["scale_up"]
    assert r["evidence"]["verified"]
    # [G12] companion (cubic interpolation): tb 3.78 h, tt 5.26 h, LUB 3.77 cm, new bed 20.0 cm, capacity 0.124 g/g.
    # Measured with linear interpolation and trapezoids: tb 3.643 h (-3.6 %), tt 5.189 h (-1.4 %), LUB 4.17 cm (+10.7 %),
    # bed 20.36 cm (+1.8 %), capacity 0.1227 (-1.0 %). The companion's tu = 3.84 h > tb is impossible (cubic overshoot below
    # c/c0 = 0), so its LUB is the less reliable number; the bands below are the measured deviations.
    assert rel(lab["breakthrough_time_s"] / 3600, 3.78) < 0.04
    assert rel(lab["total_capacity_time_s"] / 3600, 5.26) < 0.015
    assert rel(lab["length_of_unused_bed_m"] * 100, 3.77) < 0.11
    assert rel(up["bed_length_m"] * 100, 20.0) < 0.02
    assert rel(lab["saturation_capacity_kg_kg"], 0.124) < 0.011
    assert lab["usable_capacity_time_s"] <= lab["breakthrough_time_s"]


def test_lub_analytic_ramp_and_scaleup_identity():
    # linear ramp 0 -> 1 between 100 and 300 s: tt = 200 s exactly, tb at 0.1 = 120 s, tu = 120 - 0.5*0.1*20 = 119 s
    t = [0.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
    x = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    r = AD.lub_scaleup(t, x, 1.0, 0.1, new_bed_length_m=2.0)
    lab = r["laboratory"]
    assert abs(lab["total_capacity_time_s"] - 200.0) < 1e-9
    assert abs(lab["breakthrough_time_s"] - 120.0) < 1e-9
    assert abs(lab["usable_capacity_time_s"] - 119.0) < 1e-9
    assert abs(lab["length_of_unused_bed_m"] - (1 - 119.0 / 200.0)) < 1e-12
    up = r["scale_up"]
    assert abs(up["used_bed_length_m"] - (2.0 - lab["length_of_unused_bed_m"])) < 1e-12
    back = AD.lub_scaleup(t, x, 1.0, 0.1, new_breakthrough_time_s=up["breakthrough_time_s"])
    assert abs(back["scale_up"]["bed_length_m"] - 2.0) < 1e-12
    with pytest.raises(ValueError, match="no supera"):
        AD.lub_scaleup(t, x, 1.0, 0.1, new_bed_length_m=0.3)
    with pytest.raises(ValueError, match="time_s"):
        AD.lub_scaleup(None, x, 1.0, 0.1)
    short = AD.lub_scaleup(t[:4], x[:4], 1.0, 0.1)
    assert not short["evidence"]["verified"]                  # curve stops at c/c0 = 0.5: capacity undetermined


# ------------------------------------------------------------------------------------------------- PSA
EPS, RHO, KA, KB = 0.4, 500.0, 0.0152, 8e-4
BA, BB = EPS / (EPS + RHO * KA), EPS / (EPS + RHO * KB)
T_BT, T_PC = EPS / (BA * 0.05), EPS / (BA * 0.1)       # ET feed time to breakthrough (uF 0.05) and complete purge (uP 0.1)


def _psa(f: float, gam: float, n: int = 40, **kw) -> dict:
    comps = [{"name": "CO2", "feed_mole_fraction": 0.3, "isotherm": {"model": "linear", "K_m3_kg": KA}, "k_ldf_1_s": 1.0},
             {"name": "H2", "feed_mole_fraction": 0.7, "isotherm": {"model": "linear", "K_m3_kg": KB}, "k_ldf_1_s": 1.0}]
    return AD.psa_skarstrom(1.0, EPS, RHO, 300.0, comps, "H2", 5e5, 1e5, 0.05, 0.1, 20.0, f * T_BT, 20.0, gam * T_PC, "product",
                            n_cells=n, **kw)


def test_psa_linear_vs_equilibrium_theory():
    r = _psa(0.7, 1.2)
    dumps(r)
    ev = r["evidence"]
    assert ev["verified"], ev
    assert ev["cyclic_steady_state_residual"] < 1e-4 and ev["cycle_mass_balance_rel_error"] < 1e-5
    et = r["equilibrium_theory"]
    beta = BA / BB
    exact = (0.7 * (1 + (beta - 1) * 0.3) - 1.2 / 5 - beta * (1 - 1 / 5)) / (0.7 * 0.7)
    assert abs(et["light_recovery_this_cycle"] - exact) < 1e-12
    assert abs(r["product_recovery"] - exact) < 1e-3          # measured 1.7e-4 (k_LDF = 1/s, 40 cells)
    assert r["product_purity_mole_fraction"] > 0.9999         # ET: pure product
    assert "comparable" in r["validation_status"]


def test_psa_knaebel_hill_limit():
    r = _psa(1.0, 1.0, check_grid=False)
    beta = BA / BB
    kh = (1 - beta) * (1 - 1 / (5 * 0.7))
    assert abs(r["equilibrium_theory"]["light_recovery_complete_purge_product_pressurization"] - kh) < 1e-12
    assert abs(r["equilibrium_theory"]["light_recovery_this_cycle"] - kh) < 1e-12
    assert abs(r["product_recovery"] - kh) < 5e-3             # measured 2.7e-3; the shock just reaches the outlet
    assert r["product_purity_mole_fraction"] > 0.98           # measured 0.986


def test_psa_nonlinear_is_labelled_not_validated_and_refusals():
    comps = [{"name": "CO2", "feed_mole_fraction": 0.3, "k_ldf_1_s": 0.5,
              "isotherm": {"model": "langmuir", "q_max_mol_kg": 3.0, "b_m3_mol": 0.05}},
             {"name": "H2", "feed_mole_fraction": 0.7, "isotherm": {"model": "linear", "K_m3_kg": 8e-4}, "k_ldf_1_s": 0.5}]
    r = AD.psa_skarstrom(1.0, EPS, RHO, 300.0, comps, "H2", 5e5, 1e5, 0.05, 0.1, 20.0, 100.0, 20.0, 60.0, "product",
                         n_cells=12, max_cycles=30, check_grid=False)
    dumps(r)
    assert r["validation_status"].startswith("NO VALIDADO")
    assert r["evidence"]["cycle_mass_balance_rel_error"] < 1e-5
    with pytest.raises(ValueError, match="pressurization_with"):
        AD.psa_skarstrom(1.0, EPS, RHO, 300.0, comps, "H2", 5e5, 1e5, 0.05, 0.1, 20.0, 100.0, 20.0, 60.0, None)
    with pytest.raises(ValueError, match="product_component"):
        AD.psa_skarstrom(1.0, EPS, RHO, 300.0, comps, None, 5e5, 1e5, 0.05, 0.1, 20.0, 100.0, 20.0, 60.0, "product")
    with pytest.raises(ValueError, match="T_K"):
        AD.psa_skarstrom(1.0, EPS, RHO, None, comps, "H2", 5e5, 1e5, 0.05, 0.1, 20.0, 100.0, 20.0, 60.0, "product")


# ------------------------------------------------------------------------------------------------- tools / wrappers
def test_tools_and_wrappers_registry():
    from packages.agent import wrappers_separations as WS
    assert [f.__name__ for f in TS.SEPARATIONS_TOOLS] == ["gas_permeation_module", "reverse_osmosis", "adsorption_fixed_bed",
                                                          "psa_cycle"]
    assert [w.name for w in WS.AGENT_WRAPPERS] == [f.__name__ for f in TS.SEPARATIONS_TOOLS]
    out = json.loads(WS.gas_permeation_module.call({"feed_flow_mol_s": 1e6 * STP, "feed_composition": {"O2": 0.209, "N2": 0.791},
                                                    "feed_pressure_Pa": 190 * CMHG, "permeate_pressure_Pa": 19 * CMHG,
                                                    "flow_pattern": "cross_flow", "permeance_mol_m2_s_Pa": AIR_Q,
                                                    "stage_cut": 0.2}))
    assert abs(out["permeate"]["mole_fractions"]["O2"] - 0.5688) < 1.5e-4
    vac = TS.gas_permeation_module(1.0, {"A": 0.5, "B": 0.5}, 5e5, 0.0, "cross_flow", permeance_mol_m2_s_Pa={"A": 1e-8, "B": 1e-9},
                                   stage_cut=0.3)
    assert vac["pressure_ratio"] == "inf"
    dumps(vac)
    lub = TS.adsorption_fixed_bed("lub_scaleup", 0.14, 0.01, lab_curve={"time_s": [t * 3600 for t in G12_T_H], "c_over_c0": G12_X})
    assert lub["evidence"]["verified"]
    with pytest.raises(ValueError, match="mode"):
        TS.adsorption_fixed_bed("otro", 1.0, 0.05)
    with pytest.raises(ValueError, match="lab_curve"):
        TS.adsorption_fixed_bed("lub_scaleup", 0.14, 0.01)
    ro = json.loads(WS.reverse_osmosis.call({"feed_concentration_kg_m3": 2.5, "solute_molar_mass_kg_mol": 58.45e-3,
                                             "ions_per_formula": 2, "T_K": 298.15, "feed_pressure_Pa": 28.2 * ATM,
                                             "permeate_pressure_Pa": ATM, "water_permeability_kg_m2_s_Pa": 4.81e-4 / ATM,
                                             "solute_permeability_m_s": 4.42e-7}))
    assert math.isfinite(ro["local_at_feed"]["water_flux_kg_m2_s"])
    assert np.isclose(ro["permeate_density_kg_m3"], 997.05, atol=0.05)   # IAPWS-95 at 25 °C, 1 atm
    for w in WS.AGENT_WRAPPERS:
        assert len(w.description.strip().splitlines()) <= 8 and w.input_schema["type"] == "object"
