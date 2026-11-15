"""Block `catalysis`: catalyst pellet effectiveness, heterogeneous fixed bed, gas-liquid film theory.

Reference cases (sources opened and read 2026-11-14):
  [RE] Rawlings & Ekerdt, «Chemical Reactor Analysis and Design Fundamentals», ch. 7 slides,
       https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/ch7/slides-masswrxn.pdf
       Example 7.1 (pellet), Fig. 7.19 discussion (Weisz-Hicks Φ = 0.01, β = 0.4, γ = 30: three steady states),
       Examples 7.3, 7.4, 7.5 (fixed beds).
  [WU] P. A. Ramachandran, «Chapter 13 Gas-Liquid Systems», course notes, Washington University in St. Louis,
       https://classes.engineering.wustl.edu/eece503/Lecture_Notes/Gas%20Liquid%20Reactions.pdf
       Example 1 (physical absorption of H2S), backmixed liquid example (Ha = 1, αgl = 1.5, βr = 10), Problem 2 (H2S +
       Fe3+ EDTA, van Krevelen-Hoftijzer type), Problem 4 (H2S + amine, instantaneous).
Analytical: η first order (tanh φ/φ, 2 I1/(φ I0), 3(φ coth φ − 1)/φ²), zero-order dead core, film in series,
1/Φ asymptote, Prater ΔT_max = β·Ts, Ergun P² profile, adiabatic invariant, E = Ha/tanh Ha, E_i = 1 + q.
"""
from __future__ import annotations

import json
import math

import pytest

from packages.agent import wrappers_catalysis as W
from packages.dwsim_client import tools_catalysis as T
from packages.transport import multiphase_reactor as MR
from packages.transport import pellet as PL

ATM = 101325.0
RG = 8.314462618


def _first(k, n=1.0, **extra):
    return {"type": "power_law", "basis": "pellet_volume", "k": k, "order": n, **extra}


def _eff(**kw):
    r = T.catalyst_effectiveness(**kw)
    json.dumps(r)
    return r


# ---------------------------------------------------------------------------------------------------------------------
# pellet: analytical
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("geometry", ["slab", "cylinder", "sphere"])
@pytest.mark.parametrize("phi", [0.1, 1.0, 5.0, 30.0])
def test_first_order_analytical(geometry, phi):
    L, De = 1e-3, 1e-6
    r = _eff(geometry=geometry, characteristic_length_m=L, kinetics=_first(phi ** 2 * De / L ** 2),
             concentration_mol_m3=10.0, temperature_K=500.0, effective_diffusivity_m2_s=De)
    exact = PL.eta_first_order(geometry, phi)
    assert abs(r["eta_overall"] - exact) / exact < 1e-8          # measured ≤ 4e-11
    assert r["evidence"]["verified"]
    assert r["evidence"]["flux_balance_rel"] < 1e-4
    assert abs(r["thiele_modulus"] - phi) / phi < 1e-12


def test_closed_forms_themselves():
    assert abs(PL.eta_first_order("slab", 2.0) - math.tanh(2.0) / 2.0) < 1e-15
    assert abs(PL.eta_first_order("sphere", 3.0) - 3 * (3 / math.tanh(3) - 1) / 9) < 1e-15
    for g in ("slab", "cylinder", "sphere"):                     # small-φ series joins the closed form
        assert abs(PL.eta_first_order(g, 0.01 * (1 - 1e-12)) - PL.eta_first_order(g, 0.01 * (1 + 1e-12))) < 1e-11


@pytest.mark.parametrize("geometry", ["slab", "cylinder", "sphere"])
@pytest.mark.parametrize("phi", [1.0, 3.0, 10.0])
def test_zero_order_dead_core(geometry, phi):
    r = _eff(geometry=geometry, characteristic_length_m=1e-3, kinetics=_first(phi ** 2 * 10.0, 0.0),
             concentration_mol_m3=10.0, temperature_K=500.0, effective_diffusivity_m2_s=1e-6)
    exact = PL.eta_zero_order(geometry, phi)
    assert abs(r["eta_overall"] - exact) / exact < 1e-9          # measured ≤ 1e-11
    assert r["evidence"]["verified"]


def test_external_film_series_formula():
    r = _eff(geometry="sphere", characteristic_length_m=1e-3, kinetics=_first(25.0), concentration_mol_m3=10.0,
             temperature_K=500.0, effective_diffusivity_m2_s=1e-6, mass_transfer_coefficient_m_s=0.01)
    exact = PL.eta_first_order_with_film("sphere", 5.0, 10.0)
    assert abs(r["eta_overall"] - exact) / exact < 1e-8
    assert abs(r["eta_overall"] - r["eta_internal"] * r["eta_external"]) < 1e-12
    assert r["criteria"]["mears_mass"]["value"] > 0


def test_generalized_modulus_asymptote_second_order():
    # η → 1/Φ for large generalized modulus, any order (Rawlings Eq. 7.42); at Φ ≈ 100 the gap is O(1/Φ)
    r = _eff(geometry="sphere", characteristic_length_m=1e-3, kinetics=_first(3000.0, 2.0), concentration_mol_m3=10.0,
             temperature_K=500.0, effective_diffusivity_m2_s=1e-6)
    assert r["generalized_modulus"] > 50
    assert abs(r["eta_internal"] * r["generalized_modulus"] - 1.0) < 0.02
    assert r["evidence"]["verified"]


def test_monotonic_in_thiele():
    etas = [_eff(geometry="sphere", characteristic_length_m=1e-3, kinetics=_first(k, 2.0), concentration_mol_m3=10.0,
                 temperature_K=500.0, effective_diffusivity_m2_s=1e-6)["eta_overall"] for k in (0.1, 1.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(etas, etas[1:]))


def test_weisz_prater_identity_first_order():
    r = _eff(geometry="slab", characteristic_length_m=1e-3, kinetics=_first(4.0), concentration_mol_m3=10.0,
             temperature_K=500.0, effective_diffusivity_m2_s=1e-6)
    phi = 2.0
    assert abs(r["criteria"]["weisz_prater"]["value"] - r["eta_internal"] * phi ** 2) / (r["eta_internal"] * phi ** 2) < 1e-8


def test_langmuir_hinshelwood_converges_and_checks():
    kin = {"type": "langmuir_hinshelwood", "basis": "pellet_volume", "k": 10.0, "K_m3_mol": 5.0, "inhibition_exponent": 2}
    r = _eff(geometry="slab", characteristic_length_m=1e-3, kinetics=kin, concentration_mol_m3=10.0, temperature_K=500.0,
             effective_diffusivity_m2_s=1e-6)
    assert r["evidence"]["verified"]
    assert r["evidence"]["shooting_rel_dev"] < 1e-6


# ---------------------------------------------------------------------------------------------------------------------
# pellet: [RE] Example 7.1 — first order, sphere R = 0.3 cm, De = 0.007 cm²/s, ρp = 0.85 g/cm³, 0.7 atm, 450 K,
# R_Ap = −2.5e-5 mol/(g s) → Φ = 1.93 (a = R/3), k = 2.61 1/s; R = 0.15 cm → Φ = 0.964, η = 0.685, rate 3.38e-5
# ---------------------------------------------------------------------------------------------------------------------
def test_rawlings_example_7_1():
    cs = 0.7 * ATM / (RG * 450.0)                                # [RE] 1.90e-5 mol/cm³ = 19.0 mol/m³
    big = _eff(geometry="sphere", characteristic_length_m=0.3e-2, kinetics=_first(2.61), concentration_mol_m3=cs,
               temperature_K=450.0, effective_diffusivity_m2_s=0.007e-4)
    assert abs(big["thiele_modulus"] / 3.0 - 1.93) / 1.93 < 0.002          # measured +0.05 %
    rate_given = 2.5e-5 * 0.85 * 1e6                                         # mol/(m³ pellet s) = 21.25
    dev = (big["observed_rate_mol_m3_s"] - rate_given) / rate_given
    assert abs(dev) < 0.004                                                  # measured −0.23 % (k and cAs rounded)
    small = _eff(geometry="sphere", characteristic_length_m=0.15e-2, kinetics=_first(2.6),
                 concentration_mol_m3=1.90e-5 * 1e6,
                 temperature_K=450.0, effective_diffusivity_m2_s=0.007e-4)
    assert abs(small["eta_internal"] - 0.685) / 0.685 < 0.002                # measured +0.03 %
    assert abs(small["observed_rate_mol_m3_s"] - 33.8) / 33.8 < 0.003        # measured +0.14 %


# ---------------------------------------------------------------------------------------------------------------------
# pellet: non-isothermal (Weisz-Hicks)
# ---------------------------------------------------------------------------------------------------------------------
def _wh(gamma, beta, phi, branch="low", Tb=500.0, Cs=10.0, De=1e-6, lam=1.0, R=1e-3):
    kin = _first(phi ** 2 * De / R ** 2, activation_energy_J_mol=gamma * RG * Tb, reference_temperature_K=Tb)
    return _eff(geometry="sphere", characteristic_length_m=R, kinetics=kin, concentration_mol_m3=Cs, temperature_K=Tb,
                effective_diffusivity_m2_s=De, heat_of_reaction_J_mol=-beta * lam * Tb / (De * Cs),
                effective_conductivity_W_mK=lam, branch=branch)


def test_rawlings_weisz_hicks_three_states():
    # [RE] Fig. 7.19 / slide 102: Φ = 0.01 (generalized, Eq. 7.66), β = 0.4, γ = 30 → three steady states;
    # the ignited state C has c(0) ≈ 0 and T rise β (slide 104)
    g, b = 30.0, 0.4
    from scipy.integrate import quad
    I = math.sqrt(2 * quad(lambda c: c * math.exp(g * b * (1 - c) / (1 + b * (1 - c))), 0, 1)[0])
    phi = 3.0 * 0.01 * I                                          # φ = R (k/De)^½ = 3 Φ̃, Φ̃ = Φ·I
    r = _wh(g, b, phi, branch="high")
    assert abs(r["generalized_modulus"] - 0.01) < 1e-6
    assert len(r["steady_states"]) == 3
    assert r["evidence"]["verified"]
    hot = r["steady_states"][-1]
    assert hot["center_concentration_mol_m3"] / 10.0 < 1e-3
    assert abs((hot["max_temperature_K"] - 500.0) / 500.0 - b) < 1e-3       # Prater ΔT_max = β·Ts
    assert r["steady_states"][0]["eta_internal"] < r["steady_states"][1]["eta_internal"] < hot["eta_internal"]


def test_prater_limit_and_isothermal_limit():
    r = _wh(20.0, 0.3, 2.0)
    assert abs(r["internal_temperature_rise_K"] - 0.3 * 500.0) / 150.0 < 1e-3
    assert r["evidence"]["prater_invariant_rel"] < 1e-4
    iso = _wh(20.0, 1e-9, 1.0)                                    # β → 0: isothermal first order
    assert abs(iso["eta_internal"] - PL.eta_first_order("sphere", 1.0)) < 1e-6


# ---------------------------------------------------------------------------------------------------------------------
# pellet: refusals, estimated De, json
# ---------------------------------------------------------------------------------------------------------------------
def test_pellet_refusals():
    base = dict(geometry="sphere", characteristic_length_m=1e-3, kinetics=_first(1.0), concentration_mol_m3=1.0,
                temperature_K=500.0)
    with pytest.raises(ValueError, match="effective_diffusivity"):
        T.catalyst_effectiveness(**base)
    with pytest.raises(ValueError, match="basis"):
        T.catalyst_effectiveness(**{**base, "kinetics": {"type": "power_law", "k": 1.0, "order": 1}},
                                 effective_diffusivity_m2_s=1e-6)
    with pytest.raises(ValueError, match="heat_of_reaction"):
        T.catalyst_effectiveness(**base, effective_diffusivity_m2_s=1e-6, effective_conductivity_W_mK=0.5)
    with pytest.raises(ValueError, match="tortuosity"):
        T.catalyst_effectiveness(**base, pore={"porosity": 0.4, "mean_pore_radius_m": 5e-9,
                                               "molecular_diffusivity_m2_s": 1e-5, "molar_mass_kg_mol": 0.03})


def test_pore_estimate_flagged():
    r = _eff(geometry="sphere", characteristic_length_m=1e-3, kinetics=_first(1.0), concentration_mol_m3=1.0,
             temperature_K=500.0, pore={"porosity": 0.4, "tortuosity": 4.0, "mean_pore_radius_m": 5e-9,
                                        "molecular_diffusivity_m2_s": 1e-5, "molar_mass_kg_mol": 0.03})
    assert "ESTIMADA" in r["effective_diffusivity"]["source"]
    assert any("estimación" in w for w in r["warnings"])


# ---------------------------------------------------------------------------------------------------------------------
# fixed bed: [RE] Examples 7.3-7.5 (isothermal, isobaric; R_A = −(1 − ε_B) η k c_A; 1 − ε_B = ρB/ρp)
# ---------------------------------------------------------------------------------------------------------------------
BED_73 = dict(geometry="sphere", characteristic_length_m=0.3e-2, kinetics=_first(2.6), effective_diffusivity_m2_s=0.007e-4,
              pellet_density_kg_m3=850.0, bed_voidage=1 - 0.6 / 0.85, tube_diameter_m=1.0, target_conversion=0.97,
              feed_molar_flow_A_mol_s=12.0, feed_total_molar_flow_mol_s=12.0, moles_change_per_mol_A=0.0,
              inlet_temperature_K=450.0, inlet_pressure_Pa=1.5 * ATM, energy_mode="isothermal")


def _bed(**kw):
    r = T.packed_bed_catalytic_reactor(**kw)
    json.dumps(r)
    return r


def test_rawlings_example_7_3():
    r = _bed(**BED_73)                                            # [RE] VR = 1.32e6 cm³, Wc = 789 kg, η = 0.429
    assert r["evidence"]["verified"]
    assert abs(r["profile"]["eta_overall"][0] - 0.429) / 0.429 < 0.001       # measured +0.03 %
    assert abs(r["bed_volume_m3"] - 1.32) / 1.32 < 0.005                     # measured −0.36 % (source rounds)
    assert abs(r["catalyst_mass_kg"] - 789.0) / 789.0 < 0.001                # measured +0.02 %
    assert r["evidence"]["analytical_rel_dev"] < 1e-6


@pytest.mark.parametrize("kc, eta_src, mass_src, band", [(0.07e-2, 0.165, 2051.0, 0.004), (1.4e-2, 0.397, 852.0, 0.001)])
def test_rawlings_example_7_4(kc, eta_src, mass_src, band):
    r = _bed(**BED_73, mass_transfer_coefficient_m_s=kc)         # [RE] η = 0.165 / 0.397; 2051 kg / 852 kg
    assert abs(r["profile"]["eta_overall"][0] - eta_src) / eta_src < 0.003   # measured +0.27 % / +0.12 %
    assert abs(r["catalyst_mass_kg"] - mass_src) / mass_src < band           # measured −0.20 % / +0.002 %
    assert r["evidence"]["verified"]


def test_rawlings_example_7_5_second_order():
    # [RE] VR = 361 L, Wc = 216 kg computed WITH the first-order η approximation (Eq. 7.82); the rigorous second-order
    # η is slightly smaller, so the rigorous bed must be slightly larger (measured +1.4 % volume, +1.7 % mass)
    kw = dict(BED_73, characteristic_length_m=0.45e-2, kinetics=_first(2.25e5 * 1e-6, 2.0),
              effective_diffusivity_m2_s=0.008e-4,
              pellet_density_kg_m3=680.0, bed_voidage=1 - 0.6 / 0.68, target_conversion=0.75, feed_molar_flow_A_mol_s=10.0,
              feed_total_molar_flow_mol_s=20.0, inlet_temperature_K=550.0, inlet_pressure_Pa=4.0 * ATM, n_profile=3)
    r = _bed(**kw)
    assert r["evidence"]["verified"]
    dv = (r["bed_volume_m3"] - 0.361) / 0.361
    dm = (r["catalyst_mass_kg"] - 216.0) / 216.0
    assert 0.005 < dv < 0.025 and 0.005 < dm < 0.03
    assert r["bed_volume_m3"] > 0.333                             # [RE] large-Φ closed form gives 333 L (underestimate)


def test_ergun_analytical_and_constant_moles():
    # isothermal, δ = 0 → ρ ∝ P and dP/dz = −β0 P0/P: P = P0 (1 − 2 β0 z/P0)^½
    eps, dp, D, mdot, P0, Tk, mu, M = 0.4, 0.006, 0.05, 0.05 * 0.03, 2 * ATM, 450.0, 2.5e-5, 0.03
    r = _bed(**dict(BED_73, target_conversion=None, bed_length_m=2.0, bed_voidage=eps, tube_diameter_m=D,
                    feed_molar_flow_A_mol_s=0.01, feed_total_molar_flow_mol_s=0.05, inlet_pressure_Pa=P0,
                    gas_viscosity_Pa_s=mu, feed_molar_mass_kg_mol=M, pellet_density_kg_m3=None))
    A = math.pi * D * D / 4
    G, rho0 = mdot / A, P0 * M / (RG * Tk)
    b0 = G / (rho0 * dp) * (1 - eps) / eps ** 3 * (150 * (1 - eps) * mu / dp + 1.75 * G)
    exact = P0 * math.sqrt(1 - 2 * b0 * 2.0 / P0)
    assert abs(r["outlet_pressure_Pa"] - exact) / (P0 - exact) < 1e-6
    assert r["evidence"]["verified"]


KIN_E = _first(2.6, activation_energy_J_mol=80e3, reference_temperature_K=450.0)
BED_NI = dict(geometry="sphere", characteristic_length_m=0.3e-2, kinetics=KIN_E, effective_diffusivity_m2_s=0.007e-4,
              bed_voidage=0.4, tube_diameter_m=0.05, number_of_tubes=100, bed_length_m=3.0, feed_molar_flow_A_mol_s=0.5,
              feed_total_molar_flow_mol_s=5.0, moles_change_per_mol_A=1.0, inlet_temperature_K=440.0,
              inlet_pressure_Pa=3 * ATM, heat_of_reaction_J_mol=-60e3, gas_heat_capacity_J_molK=35.0)


def test_adiabatic_invariant():
    r = _bed(**BED_NI, energy_mode="adiabatic")
    X = r["conversion"]
    exact = 440.0 + 60e3 / 35.0 * math.log((5.0 + 0.5 * X) / 5.0)   # δ = 1
    assert abs(r["outlet_temperature_K"] - exact) < 1e-5
    assert r["evidence"]["energy_closure_rel"] < 1e-6 and r["evidence"]["verified"]


def test_wall_cooled_closure_and_limits():
    r = _bed(**BED_NI, energy_mode="wall_cooled", wall_heat_transfer_coefficient_W_m2K=150.0, coolant_temperature_K=440.0)
    assert r["evidence"]["energy_closure_rel"] < 1e-5 and r["evidence"]["verified"]
    assert 440.0 < r["hot_spot_temperature_K"] < 603.4
    adi = _bed(**BED_NI, energy_mode="adiabatic")
    assert r["hot_spot_temperature_K"] < adi["hot_spot_temperature_K"]
    # U → 0 recovers the adiabatic bed
    u0 = _bed(**BED_NI, energy_mode="wall_cooled", wall_heat_transfer_coefficient_W_m2K=1e-9, coolant_temperature_K=440.0)
    assert abs(u0["outlet_temperature_K"] - adi["outlet_temperature_K"]) < 1e-3


def test_coupled_pellet_in_bed_small():
    # endothermic, wall-heated, intraparticle gradient (coupled C-T pellet BVP at every axial position); endothermic
    # keeps the test fast (no multiplicity scan); the exothermic analogue was also verified (5.7 s)
    kw = dict(BED_NI, kinetics={**KIN_E, "order": 2, "k": 0.52}, bed_length_m=0.5, number_of_tubes=100,
              heat_of_reaction_J_mol=+60e3, effective_conductivity_W_mK=0.4, energy_mode="wall_cooled",
              wall_heat_transfer_coefficient_W_m2K=150.0, coolant_temperature_K=480.0, n_profile=3)
    r = _bed(**kw)
    assert r["evidence"]["verified"] and r["evidence"]["energy_closure_rel"] < 1e-5
    assert 440.0 < r["outlet_temperature_K"] < 480.0
    assert r["evidence"]["local_rate_max_rel_dev"] < 1e-4
    assert r["fraction_of_profile_diffusion_limited"] > 0


def test_bed_refusals():
    for drop, msg in (("bed_voidage", "bed_voidage"), ("energy_mode", "energy_mode"),
                      ("moles_change_per_mol_A", "moles_change")):
        kw = dict(BED_73)
        kw.pop(drop)
        with pytest.raises(ValueError, match=msg):
            T.packed_bed_catalytic_reactor(**kw)
    with pytest.raises(ValueError, match="gas_heat_capacity"):
        T.packed_bed_catalytic_reactor(**dict(BED_73, energy_mode="adiabatic", heat_of_reaction_J_mol=-1e4))
    with pytest.raises(ValueError, match="bed_length_m"):
        T.packed_bed_catalytic_reactor(**dict(BED_73, bed_length_m=1.0))


# ---------------------------------------------------------------------------------------------------------------------
# gas-liquid: [WU]
# ---------------------------------------------------------------------------------------------------------------------
def _gl(**kw):
    r = T.gas_liquid_reaction(**kw)
    json.dumps(r)
    return r


def test_wustl_example_1_physical_absorption():
    # [WU] kL = 2e-4 m/s, kg = 6e-6 mol/(m² Pa s), H = 1950 Pa m³/mol, p = 0.05·1e5 Pa → KG ≈ 1e-7, rate 5.09e-4 mol/(m² s)
    r = _gl(kinetics_type="first_order", liquid_mass_transfer_coefficient_m_s=2e-4, diffusivity_A_m2_s=1.44e-9,
            gas_partial_pressure_Pa=0.05e5, henry_constant_Pa_m3_mol=1950.0, gas_mass_transfer_coefficient_mol_m2_s_Pa=6e-6,
            rate_constant=1e-12, bulk_concentration_A_mol_m3=0.0)
    KG = r["absorption_flux_mol_m2_s"] / 0.05e5
    assert abs(KG - 1e-7) / 1e-7 < 0.01                                    # measured +0.8 % (source rounds to 1e-7)
    dev = (r["absorption_flux_mol_m2_s"] - 5.09e-4) / 5.09e-4
    assert -0.012 < dev < -0.005                                            # measured −0.94 %: source's own arithmetic
    assert r["regime"].startswith("lenta")


def test_wustl_backmixed_liquid():
    # [WU] Ha = 1, αgl = kL a V/QL = 1.5, βr = εL V k1/QL = 10 → a0 = 0.09841
    r = _gl(kinetics_type="first_order", liquid_mass_transfer_coefficient_m_s=1e-4, diffusivity_A_m2_s=1e-9,
            interface_concentration_A_mol_m3=1.0, rate_constant=10.0,
            contactor={"reactor_volume_m3": 1.0, "liquid_holdup": 0.5, "liquid_flow_m3_s": 0.5,
                       "inlet_concentration_A_mol_m3": 0.0, "kLa_1_s": 0.75})
    assert abs(r["hatta_number"] - 1.0) < 1e-12
    assert abs(r["bulk_concentration_A_mol_m3"] - 0.09841) < 1e-5            # measured 0.098413
    assert r["evidence"]["verified"] and r["evidence"]["contactor_balance_rel"] < 1e-9


def test_wustl_problem_2_second_order():
    # [WU] DA = 1.44e-9, DB = 0.54e-9, H = 1950, k2 = 9 m³/(mol s), kL = 2e-4, CB = 60, ν = 2, p = 0.05 atm (·1e5 Pa)
    # → Ha = 4.4, q = 4.3945, b_i = 0.5085, E = 3.12, rate 0.0016 mol/(m² s)
    r = _gl(kinetics_type="second_order", liquid_mass_transfer_coefficient_m_s=2e-4, diffusivity_A_m2_s=1.44e-9,
            gas_partial_pressure_Pa=0.05e5, henry_constant_Pa_m3_mol=1950.0, rate_constant=9.0,
            bulk_concentration_B_mol_m3=60.0, diffusivity_B_m2_s=0.54e-9, stoichiometric_ratio_B=2.0,
            bulk_concentration_A_mol_m3=0.0)
    assert abs(r["hatta_number"] - 4.4) / 4.4 < 0.003                       # measured +0.2 %
    assert abs(r["q_ratio"] - 4.3945) / 4.3945 < 0.003                      # measured −0.16 % (source rounds A* = 2.56)
    Evk, q = r["enhancement_van_krevelen_hoftijzer"], r["q_ratio"]
    assert abs((1 + q - Evk) / q - 0.5085) < 0.0005                          # vK-H b_i: measured 0.50859
    assert 0.005 < (Evk - 3.12) / 3.12 < 0.02                                # vK-H E: measured +1.2 % (source arithmetic)
    assert 0.02 < (r["enhancement_factor"] - 3.12) / 3.12 < 0.04             # numerical film: measured +2.9 %
    assert 0.0 < (r["absorption_flux_mol_m2_s"] - 0.0016) / 0.0016 < 0.04    # measured +2.9 % (source: 2 digits)
    assert r["evidence"]["verified"]


def test_wustl_problem_4_instantaneous():
    # [WU] A* = 2e-5 mol/cm³, B = 0.25 M, DA = 1.5 DB, ν = 1, kL·a = 0.030 1/s → q = 8.33, E = 9.33, 5.598e-6 mol/(cm³ s)
    r = _gl(kinetics_type="instantaneous", liquid_mass_transfer_coefficient_m_s=1e-4, diffusivity_A_m2_s=1.5e-9,
            interface_concentration_A_mol_m3=20.0, bulk_concentration_B_mol_m3=250.0, diffusivity_B_m2_s=1e-9,
            stoichiometric_ratio_B=1.0, bulk_concentration_A_mol_m3=0.0)
    assert abs(r["enhancement_factor"] - 9.33) / 9.33 < 0.001
    per_volume = r["absorption_flux_mol_m2_s"] * 0.030 / 1e-4                # a = kLa/kL
    assert abs(per_volume - 5.598) / 5.598 < 0.001                           # measured +0.04 %


@pytest.mark.parametrize("Ha", [0.2, 2.0, 8.0])
def test_first_order_film_closed_form(Ha):
    kL, DA = 1e-4, 1e-9
    r = _gl(kinetics_type="first_order", liquid_mass_transfer_coefficient_m_s=kL, diffusivity_A_m2_s=DA,
            interface_concentration_A_mol_m3=1.0, rate_constant=(Ha * kL) ** 2 / DA, bulk_concentration_A_mol_m3=0.0)
    assert abs(r["enhancement_factor"] - Ha / math.tanh(Ha)) < 1e-10
    assert r["evidence"]["numerical_flux_rel_dev_vs_closed_form"] < 1e-4 and r["evidence"]["verified"]


def test_second_order_limits_and_vkh_band():
    kL, DA = 1e-4, 1e-9
    devs = []
    for Ha in (0.1, 3.0, 30.0, 1000.0):
        r = _gl(kinetics_type="second_order", liquid_mass_transfer_coefficient_m_s=kL, diffusivity_A_m2_s=DA,
                interface_concentration_A_mol_m3=10.0, rate_constant=(Ha * kL) ** 2 / (DA * 100.0),
                bulk_concentration_B_mol_m3=100.0, diffusivity_B_m2_s=1e-9, stoichiometric_ratio_B=1.0,
                bulk_concentration_A_mol_m3=0.0)
        assert r["evidence"]["verified"]
        assert r["enhancement_factor"] <= r["instantaneous_enhancement_factor"] + 1e-9
        devs.append(r["van_krevelen_rel_dev_vs_numerical"])
        if Ha == 0.1:
            assert abs(r["enhancement_factor"] - 0.1 / math.tanh(0.1)) < 1e-4    # no depletion: pseudo-first order
        if Ha == 1000.0:
            assert abs(r["enhancement_factor"] / r["instantaneous_enhancement_factor"] - 1) < 1e-6
            assert r["regime"].startswith("casi instantánea")
    assert max(devs) < 0.02                                                  # measured ≤ 1.1 % in this sweep


def test_gas_film_consistency():
    kw = dict(kinetics_type="second_order", liquid_mass_transfer_coefficient_m_s=2e-4, diffusivity_A_m2_s=1.44e-9,
              gas_partial_pressure_Pa=5000.0, henry_constant_Pa_m3_mol=1950.0, rate_constant=9.0,
              bulk_concentration_B_mol_m3=60.0, diffusivity_B_m2_s=0.54e-9, stoichiometric_ratio_B=2.0,
              bulk_concentration_A_mol_m3=0.0)
    r = _gl(**kw, gas_mass_transfer_coefficient_mol_m2_s_Pa=6e-6)
    N, Ci = r["absorption_flux_mol_m2_s"], r["interface_concentration_A_mol_m3"]
    assert abs(N - 6e-6 * (5000.0 - 1950.0 * Ci)) / N < 1e-9
    assert N < _gl(**kw)["absorption_flux_mol_m2_s"]


def test_gas_liquid_refusals():
    with pytest.raises(ValueError, match="kinetics_type"):
        T.gas_liquid_reaction(liquid_mass_transfer_coefficient_m_s=1e-4, diffusivity_A_m2_s=1e-9)
    with pytest.raises(ValueError, match="bulk_concentration_A"):
        T.gas_liquid_reaction("first_order", 1e-4, 1e-9, interface_concentration_A_mol_m3=1.0, rate_constant=1.0)
    with pytest.raises(ValueError, match="diffusivity_B"):
        T.gas_liquid_reaction("second_order", 1e-4, 1e-9, interface_concentration_A_mol_m3=1.0, rate_constant=1.0,
                              bulk_concentration_B_mol_m3=10.0, stoichiometric_ratio_B=1.0, bulk_concentration_A_mol_m3=0.0)
    with pytest.raises(ValueError, match="henry"):
        T.gas_liquid_reaction("first_order", 1e-4, 1e-9, gas_partial_pressure_Pa=1e3, rate_constant=1.0,
                              bulk_concentration_A_mol_m3=0.0)


# ---------------------------------------------------------------------------------------------------------------------
# registration and wrappers
# ---------------------------------------------------------------------------------------------------------------------
def test_tool_lists_match():
    assert [f.__name__ for f in T.CATALYSIS_TOOLS] == [getattr(w, "name", None) or w.__name__ for w in W.AGENT_WRAPPERS]


def test_wrapper_returns_json():
    out = W.gas_liquid_reaction.call({"kinetics_type": "first_order", "liquid_mass_transfer_coefficient_m_s": 1e-4,
                                      "diffusivity_A_m2_s": 1e-9, "interface_concentration_A_mol_m3": 1.0,
                                      "rate_constant": 10.0, "bulk_concentration_A_mol_m3": 0.0})
    assert json.loads(out)["status"] == "ok"
    assert MR.enhancement_first_order(0.0) == 1.0


# ---------------------------------------------------------------------------------------------------------------------
# three-phase reactors. [FO] Fogler, «Elements of Chemical Reaction Engineering», 1st ed. §12.1-12.2, reproduced by the
# author's site (read 2026-11-14, scanned pages, OCR with macOS Vision):
#   slurry: https://public.websites.umich.edu/~elements/5e/15chap/pdf/slurry.pdf  (Examples 12-1, 12-2)
#   trickle bed: https://public.websites.umich.edu/~elements/5e/15chap/pdf/trickle.pdf  (Example 12-3)
# ---------------------------------------------------------------------------------------------------------------------
def _3p(**kw):
    r = T.three_phase_catalytic_reactor(**kw)
    json.dumps(r)
    return r


# Ex. 12-2: CSTR slurry, 2 m³, F_B0 = 0.7 kmol/min, p_H2 = 6 atm, H' = 0.00233 mol/(atm dm³), X = 0.30,
# r_b = 0.08 min, r_cr(60 µm) = 0.21 min·kg/m³ → m = 3.95 kg/m³
SLURRY_122 = dict(reactor_type="slurry", temperature_K=394.15, henry_constant_Pa_m3_mol=ATM / 2.33, kLa_1_s=1 / (0.08 * 60),
                  pellet_density_kg_m3=1000.0, apparent_rate_constant_m3_kg_s=1 / (0.21 * 60), stoichiometric_ratio_B=1.0,
                  liquid_reactant_concentration_mol_m3=1000.0, gas_partial_pressure_Pa=6 * ATM, liquid_volume_m3=2.0,
                  operation="continuous", liquid_flow_m3_s=0.7 / 60.0, target_conversion=0.3)


def test_fogler_slurry_example_12_2():
    r = _3p(**SLURRY_122)
    assert abs(r["catalyst_loading_kg_m3"] - 3.95) / 3.95 < 0.002          # measured +0.04 %
    assert r["evidence"]["verified"] and r["evidence"]["analytical_rel_dev"] < 1e-10
    assert r["controlling_step"] == "gas_liquid"


def test_fogler_slurry_example_12_1b_shares():
    # [FO] 80 µm, m = 0.4 kg/m³, r_b = 0.08 min, r_cr = 0.283 min·kg/m³: source reads 0.84 min from its figure and gives
    # 9.5 % absorption / 90.5 % diffusion; the series sum 0.08 + 0.283/0.4 = 0.7875 min gives 10.2 % / 89.8 %
    r = _3p(**dict(SLURRY_122, apparent_rate_constant_m3_kg_s=1 / (0.283 * 60), target_conversion=None,
                   catalyst_loading_kg_m3=0.4))
    assert abs(r["total_resistance"] / 60 - 0.7875) < 1e-9
    assert abs(r["shares"]["gas_liquid"] - 0.095) < 0.008                  # measured 0.1016 (figure-read source)
    assert r["controlling_step"] == "pellet"


def test_slurry_series_identities_and_limits():
    base = dict(SLURRY_122, target_conversion=None, catalyst_loading_kg_m3=2.0, apparent_rate_constant_m3_kg_s=None,
                geometry="sphere", characteristic_length_m=40e-6, liquid_solid_mass_transfer_coefficient_m_s=1e-4,
                effective_diffusivity_m2_s=1e-9)
    lin = _3p(**base, kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 0.05, "order": 1,
                                "other_reactant_order": 0})
    res = lin["resistances"]
    assert abs(lin["overall_rate_mol_m3_s"] - lin["interface_concentration_A_mol_m3"] / sum(res.values())) < 1e-12
    assert lin["evidence"]["pellet_rate_vs_bvp_rel"] < 1e-6 and lin["evidence"]["verified"]
    # Langmuir-Hinshelwood with K → 0 is first order: the nonlinear root reproduces the series sum
    lh = _3p(**base, kinetics={"type": "langmuir_hinshelwood", "basis": "catalyst_mass", "k": 0.05, "K_m3_mol": 0.0,
                               "inhibition_exponent": 1, "other_reactant_order": 0})
    assert abs(lh["overall_rate_mol_m3_s"] - lin["overall_rate_mol_m3_s"]) / lin["overall_rate_mol_m3_s"] < 1e-5
    # one resistance dominating: huge kLa and kc → rate = m·η·k·C*
    big = _3p(**dict(base, kLa_1_s=1e9, liquid_solid_mass_transfer_coefficient_m_s=1e6),
              kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 0.05, "order": 1, "other_reactant_order": 0})
    assert big["controlling_step"] == "pellet" and big["shares"]["pellet"] > 0.9999
    # tiny kLa: the rate tends to kLa·C*
    small = _3p(**dict(base, kLa_1_s=1e-6),
                kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 0.05, "order": 1, "other_reactant_order": 0})
    assert abs(small["overall_rate_mol_m3_s"] / (1e-6 * small["interface_concentration_A_mol_m3"]) - 1) < 1e-3


def test_slurry_batch_closed_form_and_nonlinear():
    kw = dict(SLURRY_122, operation="batch", target_conversion=None, batch_time_s=1800.0, catalyst_loading_kg_m3=5.0,
              apparent_rate_constant_m3_kg_s=None, geometry="sphere", characteristic_length_m=40e-6,
              liquid_solid_mass_transfer_coefficient_m_s=1e-4, effective_diffusivity_m2_s=1e-9,
              kinetics={"type": "langmuir_hinshelwood", "basis": "catalyst_mass", "k": 0.05, "K_m3_mol": 0.1,
                        "inhibition_exponent": 1, "other_reactant_order": 0})
    r = _3p(**kw)
    assert r["evidence"]["verified"] and r["evidence"]["analytical_rel_dev"] < 1e-8      # zero order in B: linear in t
    assert abs(sum(r["shares"].values()) - 1) < 1e-12
    assert 0 < r["conversion_B"] < 1


# Ex. 12-3: trickle bed, H2 + organic, 0.2 cm spheres, ρp = 1500, εb = 0.4, D = 1 m, 20 atm, 10 mol/s (50 % H2), 400 K,
# −dP/dL = 25 kPa/m, H' = 0.008 kmol/(m³ atm), kg·ai = 6.47 1/s, kl·ai = 2.17e-3 1/s, kc = 2.2e-5 m/s, k = 3e-5 m³/(kg s),
# De = 3.84e-10 m²/s, liquid 5 kg/(m² s) of 840 kg/m³ (M = 168), X_H2 = 0.55 → resistances 36.5 / 4.15e5 / 0.23e5 /
# 1.2e5 kg·s/m³ (0.01 / 74.37 / 4.12 / 21.50 %), η = 0.277 (large-φ asymptote 3/φ, φ = 10.8), W = 34,500 kg, V = 38.33 m³
TRICKLE_123 = dict(reactor_type="trickle_bed", temperature_K=400.0, henry_constant_Pa_m3_mol=ATM / 8.0, kLa_1_s=2.17e-3,
                   kGa_1_s=6.47, liquid_solid_mass_transfer_coefficient_m_s=2.2e-5, geometry="sphere",
                   characteristic_length_m=1e-3, pellet_density_kg_m3=1500.0, stoichiometric_ratio_B=1.0,
                   liquid_reactant_concentration_mol_m3=840 / 0.168, limiting_reactant="gas", wetting_efficiency=1.0,
                   bed_voidage=0.4, column_diameter_m=1.0, target_conversion=0.55, feed_molar_flow_A_mol_s=5.0,
                   feed_total_gas_molar_flow_mol_s=10.0, gas_moles_change_per_mol_A=-1.0, inlet_pressure_Pa=20 * ATM,
                   pressure_gradient_Pa_m=25e3, liquid_flow_m3_s=5 * math.pi / 4 / 840)


def test_fogler_trickle_example_12_3_source_eta():
    r = _3p(**TRICKLE_123, apparent_rate_constant_m3_kg_s=0.277 * 3e-5)
    res = r["inlet"]["resistances"]
    assert abs(res["gas_film"] - 36.5) / 36.5 < 0.002
    assert abs(res["gas_liquid"] - 4.15e5) / 4.15e5 < 0.002
    assert abs(res["liquid_solid"] - 0.23e5) / 0.23e5 < 0.02                  # measured −1.2 % (source: 2 digits)
    assert abs(res["pellet"] - 1.2e5) / 1.2e5 < 0.003
    sh = r["inlet"]["shares"]
    assert abs(sh["gas_liquid"] - 0.7437) < 0.001 and abs(sh["pellet"] - 0.2150) < 0.001
    assert abs(sh["liquid_solid"] - 0.0412) < 0.001
    # W: measured 33,026 kg = −4.3 %. The source rounds k_vg = 1/(H·R_T) = 1/(3.81·5.58e5) = 4.70e-7 to 4.6e-7 (its
    # Eq. E12-3.17); its own closed form with 4.70e-7 gives ≈ 33,000 kg. Our closed form is matched to 1e-9.
    dev = (r["catalyst_mass_kg"] - 34500.0) / 34500.0
    assert -0.05 < dev < -0.035
    assert r["evidence"]["analytical_rel_dev"] < 1e-7 and r["evidence"]["verified"]
    assert r["controlling_step"] == "gas_liquid"


def test_fogler_trickle_example_12_3_rigorous_pellet():
    # same case with η from the pellet engine: exact η = 3(φ coth φ − 1)/φ² = 0.2515 instead of the asymptote 0.277
    r = _3p(**TRICKLE_123, kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 3e-5, "order": 1,
                                     "other_reactant_order": 0}, effective_diffusivity_m2_s=3.84e-10)
    assert abs(r["evidence"]["pellet_eta_internal_bvp"] - 0.2515) < 5e-4
    assert r["evidence"]["pellet_rate_vs_bvp_rel"] < 1e-6 and r["evidence"]["verified"]
    assert 1.03 < r["catalyst_mass_kg"] / 33026.0 < 1.05                       # +3.8 %: the larger pellet resistance
    assert abs(r["conversion_B"] - 2.75 / 23.37) < 0.002                       # [FO] 2.75 mol/s of 23.36 mol/s organic


def test_trickle_liquid_limited_closed_form_and_wetting():
    kw = dict(TRICKLE_123, limiting_reactant="liquid", kLa_1_s=None, kGa_1_s=None, feed_molar_flow_A_mol_s=10.0,
              pressure_gradient_Pa_m=0.0, liquid_reactant_concentration_mol_m3=500.0, target_conversion=0.6,
              kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 1e-7, "order": 1, "other_reactant_order": 1},
              effective_diffusivity_m2_s=1e-9, liquid_solid_mass_transfer_coefficient_m_s=1e-5)
    full = _3p(**kw)
    assert full["evidence"]["verified"] and full["evidence"]["analytical_rel_dev"] < 1e-6     # [FO] Eq. 12-39
    half = _3p(**dict(kw, wetting_efficiency=0.5))
    # every resistance of the liquid-limited path scales with 1/f: the catalyst mass doubles exactly
    assert abs(half["catalyst_mass_kg"] / full["catalyst_mass_kg"] - 2.0) < 1e-6


def test_three_phase_refusals():
    with pytest.raises(ValueError, match="reactor_type"):
        T.three_phase_catalytic_reactor(temperature_K=400.0)
    with pytest.raises(ValueError, match="wetting_efficiency"):
        T.three_phase_catalytic_reactor(**dict(TRICKLE_123, wetting_efficiency=None), apparent_rate_constant_m3_kg_s=1e-5)
    with pytest.raises(ValueError, match="primer orden"):
        T.three_phase_catalytic_reactor(**TRICKLE_123, effective_diffusivity_m2_s=1e-9,
                                        kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 1e-5, "order": 2,
                                                  "other_reactant_order": 0})
    with pytest.raises(ValueError, match="other_reactant_order"):
        T.three_phase_catalytic_reactor(**TRICKLE_123, effective_diffusivity_m2_s=1e-9,
                                        kinetics={"type": "power_law", "basis": "catalyst_mass", "k": 1e-5, "order": 1})
    with pytest.raises(ValueError, match="operation"):
        T.three_phase_catalytic_reactor(**dict(SLURRY_122, operation=None))
