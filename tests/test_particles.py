"""Block MP: non-Newtonian pipe flow, particle settling, thickener area and fluidization.

Reference cases (all read on 2026-11-15 in open sources; deviations are the ones MEASURED with this code, never tuned):
  * Geankoplis, Transport Processes and Unit Operations (3rd ed.), examples 3.5-1, 3.5-2, 3.1-6, 14.3-1, 14.3-2, as
    reproduced (inputs and computed outputs) by the FOSSEE Python Textbook Companion (IIT Bombay):
    https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter03.ipynb
    https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter14.ipynb
  * de Nevers, Fluid Mechanics for Chemical Engineers, examples 13.1, 13.3, 13.5 (FOSSEE companion):
    https://tbc-python.fossee.in/convert-notebook/Fluid_Mechanics_For_Chemical_Engineers_by_N._D._Nevers/ch13.ipynb
  * GPSA Engineering Data Book 13th ed., example 7-1, as quoted in the docstring of fluids.drag.v_terminal (fluids 1.3.1).
  * Fluidized-bed case of arXiv:2211.01185v1 §3.2, https://arxiv.org/html/2211.01185v1
  * Formulas only (no worked numbers): Wheeler, «Non-Newtonian Fluid Math», BYU 2019, eqs. 8 and 14,
    https://www2.et.byu.edu/~wheeler/Non-Newtonian_Fluid_Math.pdf; Wikipedia «Bingham plastic» (Darby-Melson);
    arXiv:2502.00610 eq. 2.13 (Hanks 16800); arXiv:1007.0810 eq. 11 (Ryan-Johnson / Hanks 6464); Joseph et al. 2001,
    https://dept.aem.umn.edu/people/faculty/joseph/PL-correlations/docs-ln/expt-corr-pap.pdf eq. A2 (Richardson-Zaki n).
"""
from __future__ import annotations

import json
import math

import pytest
from fluids.friction import Colebrook

from packages.dwsim_client import tools_particles as TP
from packages.transport import non_newtonian as NN
from packages.transport import particles as PT

FT, LB = 0.3048, 0.45359237


def area(D):
    return math.pi * D * D / 4.0


def pl(K, n):
    return {"model": "power_law", "K_Pa_sn": K, "n": n}


def k_from_k_prime(Kp, n):
    """Geankoplis gives K' (the coefficient of (8V/D)^n'); the tool takes the true consistency K = K'/((3n+1)/4n)^n."""
    return Kp / ((3 * n + 1) / (4 * n)) ** n


# ------------------------------------------------------------------ non-Newtonian: textbook cases
def test_geankoplis_3_5_1_power_law_laminar():
    # rho 1041, L 14.9 m, D 0.0524 m, V 0.0728 m/s, K' 15.23, n' 0.40 -> Nre 1.1060, dP 45391.0 Pa
    D, V, n = 0.0524, 0.0728, 0.40
    r = TP.non_newtonian_pipe_flow(V * area(D), D, 14.9, 1041.0, pl(k_from_k_prime(15.23, n), n))
    assert r["regime"] == "laminar" and r["evidence"]["verified"]
    assert r["generalized_reynolds_metzner_reed"] == pytest.approx(1.1060, rel=1e-4)
    assert r["pressure_drop_Pa"] == pytest.approx(45391.0, rel=1e-5)          # measured: 2.7e-6


def test_geankoplis_3_5_2_power_law_turbulent_dodge_metzner():
    # rho 961, D 0.0508, V 6.1, K' 2.744, n' 0.30, L 30.5; f read from the Dodge-Metzner chart 0.0032 -> 137.4 kN/m2
    D, V, n = 0.0508, 6.1, 0.30
    r = TP.non_newtonian_pipe_flow(V * area(D), D, 30.5, 961.0, pl(k_from_k_prime(2.744, n), n))
    assert r["regime"] == "turbulento" and r["evidence"]["verified"]
    dev_f = r["f_fanning"] / 0.0032 - 1.0
    dev_dp = r["pressure_drop_Pa"] / 137.4e3 - 1.0
    assert -0.04 < dev_f < -0.02            # measured −3.0 % (Dodge-Metzner equation vs chart reading)
    assert -0.04 < dev_dp < -0.02           # measured −3.0 %


def test_de_nevers_13_1_and_13_3_power_law():
    n, K = 0.41, 0.66
    D, V = 0.5 / 3.281, 1.0 / 3.281                                            # 13.1: 1 ft/s in a 0.5 ft pipe
    r = NN.pipe_flow(V * area(D), D, 1.0, 1000.0, pl(K, n))
    assert r["pressure_gradient_Pa_m"] == pytest.approx(61.241859, rel=1e-6)
    D, V = 0.152, 0.305                                                       # 13.3: Re_pl 318.531771, f 0.050081
    r = NN.pipe_flow(V * area(D), D, 1.0, 1000.0, pl(K, n))
    assert r["generalized_reynolds_metzner_reed"] == pytest.approx(318.531771, rel=1e-6)
    assert r["regime"] == "laminar"
    assert abs(r["f_fanning"] / 0.050081 - 1.0) < 0.005   # measured +0.30 %: the book f uses the rounded 61.3 Pa/m


def test_de_nevers_13_5_bingham_turbulent_against_measured_gradient():
    # tau_y 3.8 Pa, mu_p 0.00686 Pa s, D 0.0206 m, rho 1530, V 3.47 m/s; measured 11069 Pa/m; He 52427.75, Re 15942.78
    D, V = 0.0206, 3.47
    r = TP.non_newtonian_pipe_flow(V * area(D), D, 1.0, 1530.0,
                                   {"model": "bingham", "yield_stress_Pa": 3.8, "plastic_viscosity_Pa_s": 0.00686})
    assert r["hedstrom_number"] == pytest.approx(52427.752042, rel=1e-9)
    assert r["bingham_reynolds"] == pytest.approx(15942.778426, rel=1e-9)
    assert r["regime"] == "turbulento"
    dev = r["pressure_gradient_Pa_m"] / 11069.0 - 1.0
    assert -0.26 < dev < -0.22              # measured −23.9 %: Darby under-predicts this data point (reported, not tuned)
    assert any("Darby" in w for w in r["warnings"])
    assert r["turbulent"]["newtonian_smooth_f_fanning_at_Re_B"] > r["f_fanning"]


# ------------------------------------------------------------------ non-Newtonian: analytical identities
def test_newtonian_limit_is_hagen_poiseuille():
    mu, D, L, Q = 0.05, 0.03, 7.0, 2e-4
    r = NN.pipe_flow(Q, D, L, 900.0, pl(mu, 1.0))
    assert r["regime"] == "laminar"
    assert r["pressure_drop_Pa"] == pytest.approx(128 * mu * L * Q / (math.pi * D ** 4), rel=1e-12)
    assert r["laminar_solution"]["centerline_to_mean"] == pytest.approx(2.0, rel=1e-12)


def test_power_law_velocity_profile_analytical():
    n, K, D, Q = 0.5, 2.0, 0.05, 1e-3
    r = NN.pipe_flow(Q, D, 3.0, 1000.0, pl(K, n), profile_points=11)
    V = Q / area(D)
    R = D / 2
    prof = r["laminar_solution"]["velocity_profile"]
    for rr, u in zip(prof["r_m"], prof["u_m_s"]):
        exact = V * (3 * n + 1) / (n + 1) * (1 - (rr / R) ** ((n + 1) / n))
        assert u == pytest.approx(exact, rel=1e-10, abs=1e-14)
    # Wheeler eq. 8: dP = K (4L/D) ((3n+1)/4n)^n (32 Q / (pi D^3))^n
    assert r["pressure_drop_Pa"] == pytest.approx(K * 4 * 3.0 / D * ((3 * n + 1) / (4 * n)) ** n
                                                 * (32 * Q / (math.pi * D ** 3)) ** n, rel=1e-12)


def test_bingham_buckingham_reiner_and_plug():
    tau_y, mu_p, D, L, Q, rho = 10.0, 0.02, 0.05, 5.0, 4e-4, 1200.0
    r = NN.pipe_flow(Q, D, L, rho, {"model": "bingham", "yield_stress_Pa": tau_y, "plastic_viscosity_Pa_s": mu_p})
    assert r["regime"] == "laminar" and r["evidence"]["verified"]
    tw = r["wall_shear_stress_Pa"]
    phi = tau_y / tw
    assert Q == pytest.approx(math.pi * (D / 2) ** 3 * tw / (4 * mu_p) * (1 - 4 * phi / 3 + phi ** 4 / 3), rel=1e-12)
    Re, He, f = r["bingham_reynolds"], r["hedstrom_number"], r["f_fanning"]
    assert f == pytest.approx(16 / Re * (1 + He / (6 * Re) - He ** 4 / (3 * f ** 3 * Re ** 7)), rel=1e-10)  # Fanning form
    assert r["laminar_solution"]["plug_radius_m"] == pytest.approx(phi * D / 2, rel=1e-12)
    hb = NN.pipe_flow(Q, D, L, rho, {"model": "herschel_bulkley", "yield_stress_Pa": tau_y, "K_Pa_sn": mu_p, "n": 1.0})
    assert hb["pressure_drop_Pa"] == pytest.approx(r["pressure_drop_Pa"], rel=1e-12)


def test_herschel_bulkley_closed_form_vs_quadrature_and_limits():
    for tau_y, K, n in ((2.0, 0.8, 0.45), (30.0, 5.0, 0.7), (0.5, 0.01, 1.4)):
        r = NN.pipe_flow(5e-4, 0.04, 2.0, 1100.0,
                         {"model": "herschel_bulkley", "yield_stress_Pa": tau_y, "K_Pa_sn": K, "n": n})
        if r["regime"] == "laminar":
            assert r["evidence"]["verified"]
            assert r["evidence"]["checks"]["laminar_closed_form_vs_quadrature_rel"] < 1e-10
            assert r["evidence"]["checks"]["laminar_profile_integral_vs_Q_rel"] < 1e-10
    a = NN.pipe_flow(5e-4, 0.04, 2.0, 1100.0,
                     {"model": "herschel_bulkley", "yield_stress_Pa": 0.0, "K_Pa_sn": 0.8, "n": 0.45})
    b = NN.pipe_flow(5e-4, 0.04, 2.0, 1100.0, pl(0.8, 0.45))
    assert a["pressure_drop_Pa"] == pytest.approx(b["pressure_drop_Pa"], rel=1e-12)


def test_critical_reynolds_limits():
    assert NN.critical_re_power_law(1.0) == pytest.approx(2100.0, rel=1e-3)     # 6464·3^1.5/16 = 2099.2
    assert NN.critical_re_power_law(0.5) > NN.critical_re_power_law(1.0)          # shear thinning delays transition
    assert NN.hanks_critical(1e-4)[1] == pytest.approx(2100.0, rel=1e-6)          # He -> 0: Newtonian
    xi, rec = NN.hanks_critical(1e5)
    assert xi / (1 - xi) ** 3 == pytest.approx(1e5 / 16800.0, rel=1e-12)
    assert NN.hanks_critical(1e6)[1] > rec > NN.hanks_critical(1e3)[1]


def test_dodge_metzner_newtonian_limit_vs_colebrook_smooth():
    for Re in (4e3, 1e4, 1e5, 1e6):
        f, res = NN.dodge_metzner_f(Re, 1.0)
        assert res < 1e-12
        assert abs(f / (Colebrook(Re, 0.0) / 4) - 1) < 1e-3   # measured +0.05 to +0.09 % (−0.40 vs −0.396 constant)


def test_darby_low_hedstrom_limit_is_flagged():
    # consistency, not validation: Darby at He -> 0 gives about half the Newtonian smooth-pipe f (measured −50 to −58 %)
    for Re in (1e4, 1e5):
        d = NN.darby_bingham_f(Re, 0.0, 16 / Re)
        assert -0.60 < d["f_fanning"] / (Colebrook(Re, 0.0) / 4) - 1 < -0.45


def test_pressure_drop_monotonic_in_flow():
    for rh in (pl(0.3, 0.6), {"model": "bingham", "yield_stress_Pa": 5.0, "plastic_viscosity_Pa_s": 0.01},
               {"model": "herschel_bulkley", "yield_stress_Pa": 5.0, "K_Pa_sn": 0.05, "n": 0.6}):
        dps = [NN.pipe_flow(q, 0.05, 10.0, 1200.0, rh)["pressure_drop_Pa"] for q in (1e-4, 1e-3, 5e-3, 2e-2)]
        assert all(b > a for a, b in zip(dps, dps[1:]))


def test_herschel_bulkley_turbulent_is_flagged_estimate():
    r = NN.pipe_flow(2e-2, 0.05, 10.0, 1200.0,
                     {"model": "herschel_bulkley", "yield_stress_Pa": 5.0, "K_Pa_sn": 0.05, "n": 0.6})
    assert r["regime"] == "turbulento (estimado)"
    assert r["evidence"]["verified"] and any("ESTIMACIÓN" in w for w in r["warnings"])
    near_pl = NN.pipe_flow(2e-2, 0.05, 10.0, 1200.0, {"model": "herschel_bulkley", "yield_stress_Pa": 1e-9, "K_Pa_sn": 0.05,
                                                      "n": 0.6})
    assert near_pl["pressure_drop_Pa"] == pytest.approx(NN.pipe_flow(2e-2, 0.05, 10.0, 1200.0, pl(0.05, 0.6))
                                                        ["pressure_drop_Pa"], rel=1e-8)


def test_non_newtonian_refusals():
    with pytest.raises(ValueError, match="rheology"):
        NN.pipe_flow(1e-3, 0.05, 1.0, 1000.0, None)
    with pytest.raises(ValueError, match="K_Pa_sn"):
        NN.pipe_flow(1e-3, 0.05, 1.0, 1000.0, {"model": "power_law", "n": 0.5})
    with pytest.raises(ValueError, match="yield_stress_Pa"):
        NN.pipe_flow(1e-3, 0.05, 1.0, 1000.0, {"model": "bingham", "plastic_viscosity_Pa_s": 0.01})
    with pytest.raises(ValueError, match="model"):
        NN.pipe_flow(1e-3, 0.05, 1.0, 1000.0, {"model": "carreau"})
    with pytest.raises(ValueError, match="density_kg_m3"):
        NN.pipe_flow(1e-3, 0.05, 1.0, None, pl(1.0, 0.5))
    with pytest.raises(ValueError, match="diameter_m"):
        NN.pipe_flow(1e-3, -0.05, 1.0, 1000.0, pl(1.0, 0.5))


# ------------------------------------------------------------------ settling
def test_geankoplis_14_3_1_stokes_droplet():
    # oil droplet Dp 2e-5 m, rho 900, air rho 1.137, mu 1.9e-5 -> vt 0.0103 m/s, Re 0.012
    r = TP.particle_settling(2.0e-5, 900.0, 1.137, 1.9e-5, 1.0)
    assert r["evidence"]["verified"] and r["regime"] == "Stokes"
    assert abs(r["terminal_velocity_m_s"] / 0.0103 - 1) < 0.005    # measured −0.14 % (book rounds to 3 digits)
    assert r["reynolds_terminal"] == pytest.approx(0.012, abs=0.0005)


def test_gpsa_7_1_droplet_in_gas():
    # 150 µm, rho_p 31.2 lb/ft3, rho 2.07 lb/ft3, mu 1.2e-5 Pa s -> 0.46 ft/s (GPSA, quoted by fluids)
    r = PT.terminal_velocity(150e-6, 31.2 * LB / FT ** 3, 2.07 * LB / FT ** 3, 1.2e-5, 1.0)
    dev = r["terminal_velocity_m_s"] / FT / 0.46 - 1
    assert -0.03 < dev < -0.01              # measured −2.1 % (Clift); spread of 4 correlations 3.9 %
    assert r["regime"] == "intermedio"


def test_stokes_analytical_limit_and_force_balance():
    r = PT.terminal_velocity(2e-6, 2500.0, 1000.0, 1e-3, 1.0)
    assert r["reynolds_terminal"] < 1e-4
    assert r["terminal_velocity_m_s"] == pytest.approx(r["stokes_velocity_m_s"], rel=1e-4)
    assert r["evidence"]["checks"]["force_balance_rel"] < 1e-12
    up = PT.terminal_velocity(1e-3, 800.0, 1000.0, 1e-3, 1.0)      # buoyant particle rises with the same physics
    assert up["direction"].startswith("asciende")


def test_newton_regime_and_monotonic_in_diameter():
    vs = [PT.terminal_velocity(d, 2650.0, 1.2, 1.8e-5, 1.0) for d in (1e-5, 1e-4, 1e-3, 5e-3)]
    assert all(b["terminal_velocity_m_s"] > a["terminal_velocity_m_s"] for a, b in zip(vs, vs[1:]))
    assert vs[-1]["regime"] == "Newton" and 0.35 < vs[-1]["drag_coefficient"] < 0.5


def test_nonspherical_needs_area_ratio_and_falls_slower():
    with pytest.raises(ValueError, match="projected_area_ratio"):
        PT.terminal_velocity(5e-4, 2650.0, 998.0, 1e-3, 0.8)
    ns = PT.terminal_velocity(5e-4, 2650.0, 998.0, 1e-3, 0.8, projected_area_ratio=1.0)
    sp = PT.terminal_velocity(5e-4, 2650.0, 998.0, 1e-3, 1.0)
    assert ns["terminal_velocity_m_s"] < sp["terminal_velocity_m_s"] and ns["evidence"]["verified"]
    sx = PT.terminal_velocity(5e-4, 2650.0, 998.0, 1e-3, 1.0, drag_method="Song_Xu")
    dev = sx["terminal_velocity_m_s"] / sp["terminal_velocity_m_s"] - 1     # Song-Xu at phi = 1 vs Clift, Re ≈ 38
    assert -0.08 < dev < -0.05              # measured −6.8 %: bias inherited by the non-spherical results


def test_richardson_zaki_exponent_table():
    assert PT.richardson_zaki_exponent(0.1) == 4.65
    assert PT.richardson_zaki_exponent(0.1, 0.01) == pytest.approx(4.65 + 0.195)
    assert PT.richardson_zaki_exponent(0.5) == pytest.approx(4.35 * 0.5 ** -0.03)
    assert PT.richardson_zaki_exponent(10.0) == pytest.approx(4.45 * 10 ** -0.1)
    assert PT.richardson_zaki_exponent(300.0) == pytest.approx(4.45 * 300 ** -0.1)
    assert PT.richardson_zaki_exponent(1000.0) == 2.39


def test_geankoplis_14_3_2_hindered_comparison_steinour_vs_richardson_zaki():
    # glass spheres 1.554e-4 m, rho 2467, water 998 / 1.005e-3, 60 wt % solids -> Steinour (book) 1.530e-3 m/s.
    # COMPARISON of two correlations, not a validation: Richardson-Zaki gives +43.6 % (measured).
    e = (0.4 / 998) / (0.4 / 998 + 0.6 / 2467)
    r = TP.particle_settling(1.554e-4, 2467.0, 998.0, 1.005e-3, 1.0, solids_volume_fraction=1 - e)
    dev = r["hindered"]["hindered_settling_velocity_m_s"] / 1.530e-3 - 1
    assert 0.40 < dev < 0.47


def test_coe_clevenger_constant_velocity_is_clarifier_identity():
    # u(C) = u0 constant: A = Q0 (1 − C0/Cu) / u0 = overflow / u0 (analytical)
    Q0, C0, Cu, u0 = 0.02, 40.0, 600.0, 2e-4
    r = PT.settling(5e-5, 2650.0, 998.0, 1e-3, 1.0, thickener={
        "feed_flow_m3_s": Q0, "feed_solids_kg_m3": C0, "underflow_solids_kg_m3": Cu,
        "batch_tests": {"concentration_kg_m3": [40.0, 100.0, 200.0, 400.0], "settling_velocity_m_s": [u0] * 4}})
    th = r["thickener"]
    assert th["area_m2"] == pytest.approx(Q0 * (1 - C0 / Cu) / u0, rel=1e-12)
    assert th["underflow_m3_s"] + th["overflow_m3_s"] == pytest.approx(Q0, rel=1e-12)
    assert th["solids_flux_kg_m2_s"] * th["area_m2"] == pytest.approx(Q0 * C0, rel=1e-12)


def test_coe_clevenger_batch_data_picks_maximum_and_rz_estimate_flagged():
    C = [50.0, 100.0, 150.0, 200.0, 300.0]
    u = [3e-4, 1.2e-4, 5e-5, 2.5e-5, 1e-5]
    th = PT.coe_clevenger_area(0.01, 50.0, 400.0, C, u)
    best = max((1 / c - 1 / 400.0) / v for c, v in zip(C, u))
    assert th["area_m2"] == pytest.approx(0.01 * 50.0 * best, rel=1e-12)
    r = PT.settling(5e-5, 2650.0, 998.0, 1e-3, 1.0, thickener={"feed_flow_m3_s": 0.01, "feed_solids_kg_m3": 50.0,
                                                                "underflow_solids_kg_m3": 800.0})
    assert "ESTIMADO" in r["thickener"]["settling_data_source"] and r["evidence"]["verified"]


# ------------------------------------------------------------------ fluidization
def test_geankoplis_3_1_6_minimum_fluidization():
    # dp 1.2e-4 m, phi 0.88, eps_mf 0.42, air 2.374 kg/m3 / 1.85e-5, rho_p 1000, 300 kg on 0.30 m2
    # -> L_mf 1.724 m, dP 9783.4 Pa, u_mf Ergun 0.005015 m/s (Re 0.07723), Wen-Yu 0.004605 m/s
    r = TP.fluidization(1.2e-4, 1000.0, 2.374, 1.85e-5, 0.88, voidage_mf=0.42, bed_mass_kg=300.0,
                        column_diameter_m=math.sqrt(4 * 0.30 / math.pi))
    assert r["evidence"]["verified"]
    assert r["u_mf_ergun_m_s"] == pytest.approx(0.005015, rel=1e-4)   # measured 2e-5
    assert r["re_mf_ergun"] == pytest.approx(0.07723, rel=1e-4)
    assert r["u_mf_wen_yu_m_s"] == pytest.approx(0.004605, rel=1e-4)
    assert r["fluidized_bed_pressure_drop_Pa"] == pytest.approx(9783.4, rel=1e-5)
    assert r["bed_height_mf_m"] == pytest.approx(1.724, rel=1e-3)
    assert r["terminal"] is None and any("projected_area_ratio" in w for w in r["warnings"])


def test_arxiv_2211_01185_fluidized_bed():
    # 2e5 spheres of 500 µm, rho_p 1000, air 1 kg/m3 / 1e-5 Pa s, bed radius 10 mm -> dP 408.34 Pa, Wen-Yu 0.135 m/s
    M = 2e5 * math.pi / 6 * (5e-4) ** 3 * 1000.0
    r = TP.fluidization(5e-4, 1000.0, 1.0, 1e-5, 1.0, bed_mass_kg=M, column_diameter_m=0.02, superficial_velocity_m_s=0.2)
    assert abs(r["fluidized_bed_pressure_drop_Pa"] / 408.34 - 1) < 1e-3   # measured −0.034 % (g = 9.80665 here)
    assert abs(r["u_mf_wen_yu_m_s"] / 0.135 - 1) < 0.005                  # measured −0.16 % (paper rounds)
    assert r["operating_point"]["state"].startswith("fluidizado")


def test_liquid_fluidization_identities():
    kw = dict(voidage_mf=0.40, bed_mass_kg=10.0, column_diameter_m=0.1)
    base = PT.fluidization(1e-3, 2500.0, 998.0, 1e-3, 1.0, **kw)
    assert base["evidence"]["checks"]["ergun_at_umf_vs_bed_weight_rel"] < 1e-12
    umf, ut = base["u_mf_ergun_m_s"], base["operating_window_m_s"]["u_t"]
    eps = [PT.fluidization(1e-3, 2500.0, 998.0, 1e-3, 1.0, superficial_velocity_m_s=U, **kw)["operating_point"]
           ["voidage_richardson_zaki"] for U in (1.5 * umf, 3 * umf, 0.5 * ut, 0.95 * ut)]
    assert all(b > a for a, b in zip(eps, eps[1:])) and eps[-1] < 1.0
    fixed = PT.fluidization(1e-3, 2500.0, 998.0, 1e-3, 1.0, superficial_velocity_m_s=0.5 * umf, **kw)["operating_point"]
    assert fixed["state"] == "lecho fijo" and fixed["pressure_drop_Pa"] < base["fluidized_bed_pressure_drop_Pa"]
    gone = PT.fluidization(1e-3, 2500.0, 998.0, 1e-3, 1.0, superficial_velocity_m_s=1.1 * ut, **kw)["operating_point"]
    assert gone["state"].startswith("transporte")


def test_particle_refusals():
    with pytest.raises(ValueError, match="fluid_viscosity_Pa_s"):
        PT.terminal_velocity(1e-4, 2500.0, 1000.0, None, 1.0)
    with pytest.raises(ValueError, match="sphericity"):
        PT.terminal_velocity(1e-4, 2500.0, 1000.0, 1e-3, 1.2)
    with pytest.raises(ValueError, match="más densa"):
        PT.fluidization(1e-4, 900.0, 1000.0, 1e-3, 1.0)
    with pytest.raises(ValueError, match="column_diameter_m"):
        PT.fluidization(1e-4, 2500.0, 1000.0, 1e-3, 1.0, bed_mass_kg=5.0)
    with pytest.raises(ValueError, match="underflow_solids_kg_m3"):
        PT.settling(1e-4, 2500.0, 1000.0, 1e-3, 1.0, thickener={"feed_flow_m3_s": 0.01, "feed_solids_kg_m3": 50.0})


def test_every_result_is_json_and_wrappers_match_tools():
    results = [
        TP.non_newtonian_pipe_flow(1e-3, 0.05, 10.0, 1200.0, pl(0.3, 0.6)),
        TP.non_newtonian_pipe_flow(2e-2, 0.05, 10.0, 1200.0, {"model": "bingham", "yield_stress_Pa": 5.0,
                                                              "plastic_viscosity_Pa_s": 0.01}),
        TP.non_newtonian_pipe_flow(2e-2, 0.05, 10.0, 1200.0, {"model": "herschel_bulkley", "yield_stress_Pa": 5.0,
                                                              "K_Pa_sn": 0.05, "n": 0.6}),
        TP.particle_settling(5e-5, 2650.0, 998.0, 1e-3, 1.0, solids_volume_fraction=0.2, column_diameter_m=0.1,
                             thickener={"feed_flow_m3_s": 0.01, "feed_solids_kg_m3": 50.0, "underflow_solids_kg_m3": 800.0}),
        TP.particle_settling(5e-4, 2650.0, 998.0, 1e-3, 0.7, projected_area_ratio=1.1),
        TP.fluidization(1e-3, 2500.0, 998.0, 1e-3, 1.0, voidage_mf=0.4, bed_mass_kg=10.0, column_diameter_m=0.1,
                        superficial_velocity_m_s=0.03),
    ]
    for r in results:
        s = json.dumps(r, ensure_ascii=False)
        assert json.loads(s)["evidence"]["verified"] in (True, False)
        for k in ("warnings", "assumptions", "not_included", "method"):
            assert k in r
    from packages.agent import wrappers_particles as W
    assert [f.__name__ for f in TP.PARTICLES_TOOLS] == [w.name for w in W.AGENT_WRAPPERS]
    out = json.loads(W.AGENT_WRAPPERS[0](volumetric_flow_m3_s=1e-3, diameter_m=0.05, length_m=10.0, density_kg_m3=1200.0,
                                         rheology={"model": "power_law", "K_Pa_sn": 0.3, "n": 0.6}))
    assert out["regime"] == "laminar"


# ------------------------------------------------------------------ cyclone (Lapple, Shepherd-Lapple)
# Penn State ME433 «Lapple Cyclones» (Cimbala), https://www.me.psu.edu/cimbala/me433/Lesson_Notes/
# ME433_Lesson_12_B_Lapple_Cyclones.pdf, read 2026-11-15: standard Lapple, D2 = 0.450 m, rho_p 730, Q 0.550 m3/s,
# air 1.184 kg/m3 / 1.849e-5 Pa s -> D_p,cut = 5.59987 µm, eta(10 µm) = 76.13 %, dP (their H&C formula
# 40.96 rho (Q/WH)^2) = 22896 Pa. Slide deck «Cyclones» (slideshare 47222507, read 2026-11-15): d_pc = 5.79 µm and the PSD
# below give per-class efficiencies 2.9 ... 99.4 % and an overall 70.6 %.
from packages.transport import cyclone as CY  # noqa: E402

PSU = dict(gas_flow_m3_s=0.550, gas_density_kg_m3=1.184, gas_viscosity_Pa_s=1.849e-5, particle_density_kg_m3=730.0,
           body_diameter_m=0.450)


def test_psu_lapple_cut_diameter_and_grade_efficiency():
    r = TP.cyclone_performance(**PSU, geometry="lapple", diameters_m=[10e-6])
    assert r["evidence"]["verified"] and r["effective_turns"] == pytest.approx(6.0, rel=1e-12)
    assert r["cut_diameter_m"] == pytest.approx(5.59987e-6, rel=1e-5)               # measured −7e-7
    assert r["grade_efficiency"][0]["efficiency"] == pytest.approx(0.7613, abs=5e-5)
    D = PSU["body_diameter_m"]                                                        # PSU closed form for the standard
    assert r["cut_diameter_m"] == pytest.approx(math.sqrt(3 * 1.849e-5 * D ** 3 / (128 * math.pi * 0.550 * (730 - 1.184))),
                                                rel=1e-12)


def test_shepherd_lapple_velocity_heads_and_psu_pressure_drop_discrepancy():
    r = CY.cyclone_performance(**PSU, geometry="lapple")
    assert r["pressure_drop_velocity_heads"] == pytest.approx(8.0, rel=1e-12)        # 16 · (D/2·D/4)/(D/2)²
    Vi = r["inlet_velocity_m_s"]
    assert Vi == pytest.approx(0.550 / (0.225 * 0.1125), rel=1e-12)
    assert r["pressure_drop_Pa"] == pytest.approx(4 * 1.184 * Vi ** 2, rel=1e-12)
    # PSU uses the same inlet-velocity basis and SI units: 40.96 rho Vi² = 22896 Pa, 10.24 x Shepherd-Lapple (documented)
    assert 40.96 * 1.184 * Vi ** 2 == pytest.approx(22896.0, rel=1e-4)
    assert 40.96 * 1.184 * Vi ** 2 / r["pressure_drop_Pa"] == pytest.approx(10.24, rel=1e-12)
    assert CY.cyclone_performance(**PSU, geometry="stairmand_he")["pressure_drop_velocity_heads"] == pytest.approx(6.4)


def test_overall_efficiency_slide_example():
    o = CY.overall_efficiency(5.79e-6, [1e-6, 3e-6, 5e-6, 8e-6, 14e-6, 24e-6, 40e-6, 75e-6],
                              [0.01, 0.09, 0.10, 0.30, 0.30, 0.14, 0.05, 0.01])
    assert o["overall_efficiency"] == pytest.approx(0.706, abs=6e-4)                  # measured 70.63 %
    printed = [2.9, 21.1, 42.7, 65.6, 85.4, 94.5, 97.9, 99.4]
    for row, p in zip(o["table"], printed):
        assert abs(100 * row["grade_efficiency"] - p) < 0.1    # measured ≤ 0.07 points (slide truncates 21.17)


def test_cyclone_identities_limits_and_trends():
    r = CY.cyclone_performance(**PSU, geometry="lapple", diameters_m=[1e-9, 1e-2])
    assert CY.grade_efficiency(r["cut_diameter_m"], r["cut_diameter_m"]) == 0.5
    assert r["grade_efficiency"][0]["efficiency"] < 1e-6 and r["grade_efficiency"][1]["efficiency"] > 0.999999
    big = CY.cyclone_performance(**{**PSU, "gas_flow_m3_s": 1.1}, geometry="lapple")
    assert big["cut_diameter_m"] == pytest.approx(r["cut_diameter_m"] / math.sqrt(2), rel=1e-12)   # d50 ∝ Q^-1/2
    assert big["pressure_drop_Pa"] == pytest.approx(4 * r["pressure_drop_Pa"], rel=1e-12)           # dP ∝ Q²
    he = CY.cyclone_performance(**PSU, geometry="stairmand_he")
    ht = CY.cyclone_performance(**PSU, geometry="stairmand_ht")
    assert he["cut_diameter_m"] < ht["cut_diameter_m"] and any("voluta" in w for w in ht["warnings"])
    custom = CY.cyclone_performance(**PSU, geometry={k: CY.PRESETS["lapple"][k] for k in CY.RATIO_KEYS})
    assert custom["cut_diameter_m"] == pytest.approx(r["cut_diameter_m"], rel=1e-15)


def test_cyclone_refusals_and_json():
    with pytest.raises(ValueError, match="gas_viscosity_Pa_s"):
        CY.cyclone_performance(0.5, 1.2, None, 2000.0, 0.5)
    with pytest.raises(ValueError, match="body_diameter_m"):
        CY.cyclone_performance(0.5, 1.2, 1.8e-5, 2000.0, None)
    with pytest.raises(ValueError, match="faltan"):
        CY.cyclone_performance(0.5, 1.2, 1.8e-5, 2000.0, 0.5, geometry={"inlet_height": 0.5})
    with pytest.raises(ValueError, match="suman"):
        CY.cyclone_performance(0.5, 1.2, 1.8e-5, 2000.0, 0.5,
                               particle_size_distribution={"diameter_m": [1e-6, 1e-5], "mass_fraction": [0.5, 0.3]})
    r = CY.cyclone_performance(0.5, 1.2, 1.8e-5, 2000.0, 0.5, particle_size_distribution={
        "diameter_m": [2e-6, 1e-5], "mass_fraction": [0.4, 0.6]})
    assert json.loads(json.dumps(r))["evidence"]["verified"]
    assert any("particle_size_distribution" in w for w in CY.cyclone_performance(0.5, 1.2, 1.8e-5, 2000.0, 0.5)["warnings"])
    from packages.agent import wrappers_particles as W
    assert [w.name for w in W.AGENT_WRAPPERS][-1] == "cyclone_performance"
    out = json.loads(W.cyclone_performance(**PSU))
    assert out["cut_diameter_m"] == pytest.approx(5.59987e-6, rel=1e-5)
