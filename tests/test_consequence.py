"""Consequence analysis (bloque M12): source terms, Gaussian dispersion, fires, explosions and effects.

Evidence (sources read 2026-11-13; numbers below are the PRINTED ones unless a comment says otherwise):
* Crowl & Louvar, Chemical Process Safety, 4th ed. (2020): Ex. 4-1, 4-2, 4-4, 4-6, 3-9, 5-1, 5-2, 6-10, 6-11, 6-13,
  2-2, tables 2-4, 2-5, 5-2, 5-3. Errata found in the print are asserted as such (Ex. 5-1(b) is 10x too high).
* TNO Yellow Book CPR 14E (2005): §6.6.3 and §6.6.4 pool fires, §6.6.5 BLEVE fireball (its final product is 0.5 %
  off its own factors: erratum), app. 6.1 view factors.
* NUREG-1805 Example 5.11-1 (SFPE solid flame, Shokri-Beyler); NISTIR 6546 Problem 3 (point source).
* PGS 1 (2003, Dutch revision of the Green Book CPR 16E): thermal-dose table 4-4, eardrum ex. 7.2, windows ex. 2.
Not reproduced (no legible copy): CCPS CPQRA examples, the 3rd-ed. Crowl & Louvar nitrogen 0.1-in example, any
Baker-Strehlow-Tang example, multi-energy curve fits.
Limits covered: choked flow at its boundary, Britter-McQuaid dense-gas flag at its threshold, Pasquill classes at the
wind-speed edges, view factor of the numerical integral against Mudan's analytic one.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.consequence import dispersion as Dm  # noqa: E402
from packages.consequence import effects as Ef  # noqa: E402
from packages.consequence import explosion as Ex  # noqa: E402
from packages.consequence import fire as Fi  # noqa: E402
from packages.consequence import source as So  # noqa: E402
from packages.consequence import tool as T  # noqa: E402

FT, IN, LBM = 0.3048, 0.0254, 0.45359237
ATM = 101325.0


# ----------------------------------------------------------------------------------------------- source terms
def test_cl_ex4_1_benzene_pipeline_hole():
    r = So.liquid_hole(0.02, 7e5, 879.4, Cd=0.61)                   # 7 barg, 2 cm, SG 0.8794
    assert r["flujo_masico_kg_s"] == pytest.approx(6.72, rel=2e-3)   # printed 6.72 kg/s
    assert r["flujo_masico_kg_s"] * 90 * 60 == pytest.approx(3.63e4, rel=3e-3)   # printed 3.63e4 kg


def test_cl_ex4_2_padded_tank_draining():
    At = math.pi * (8 * FT) ** 2 / 4
    r = So.liquid_hole(1 * IN, ATM, 879.4, liquid_head_m=12 * FT, Cd=0.61, tank_area_m2=At)
    # printed with pi = 3.14 and rounded intermediates: 3386 s, 10.4 lbm/s, 4506 gal (book's rounding ≈ 0.5 %)
    assert r["tiempo_vaciado_s"] == pytest.approx(3386, rel=5e-3)
    assert r["flujo_masico_kg_s"] / LBM == pytest.approx(10.4, rel=5e-3)
    assert r["masa_drenada_kg"] / 879.4 / 3.785411784e-3 == pytest.approx(4506, rel=5e-3)


def test_cl_ex4_4_nitrogen_choked():
    r = So.gas_hole(0.002, 14e5 + 1.013e5, 298.0, 0.028, 1.41, ATM, Cd=1.0)
    assert r["ahogado"]
    assert r["razon_presion_critica"] == pytest.approx(0.527, abs=6e-4)
    assert r["presion_critica_Pa"] / 1e5 == pytest.approx(7.91, abs=6e-3)
    assert r["flujo_masico_kg_s"] == pytest.approx(1.09e-2, rel=3e-3)


@pytest.mark.parametrize("gamma,ratio", [(1.67, 0.487), (1.40, 0.528), (1.32, 0.542)])   # C&L §4-5 table
def test_choked_pressure_ratio_table(gamma, ratio):
    assert So.choked_pressure(1.0, gamma) == pytest.approx(ratio, abs=6e-4)


def test_choked_flow_at_its_boundary_is_continuous():
    P0, T0, M, g = 5e5, 300.0, 0.029, 1.4
    ps = So.choked_pressure(P0, g)
    at = So.gas_hole(0.01, P0, T0, M, g, ps)
    above = So.gas_hole(0.01, P0, T0, M, g, ps * (1 + 1e-9))
    assert at["ahogado"] and not above["ahogado"]                 # boundary itself is choked
    assert above["flujo_masico_kg_s"] == pytest.approx(at["flujo_masico_kg_s"], rel=1e-6)
    lower = So.gas_hole(0.01, P0, T0, M, g, ps * 0.5)            # below P*: flow no longer grows
    assert lower["flujo_masico_kg_s"] == pytest.approx(at["flujo_masico_kg_s"], rel=1e-12)
    sub = So.gas_hole(0.01, P0, T0, M, g, ps * 1.3)
    assert sub["flujo_masico_kg_s"] < at["flujo_masico_kg_s"]
    # C&L: an air leak to 14.7 psia is choked above 27.8 psia
    assert 14.7 / So.choked_pressure(1.0, 1.4) == pytest.approx(27.8, abs=0.05)


def test_cl_ex4_6_flashing_water():
    # Cp(T - Tb)/ΔHv only needs the temperature difference: 350 °F → 212 °F, Btu/lbm units cancel
    assert So.flash_fraction_cp(1.01, 350, 212, 970.3)["fraccion_flash"] == pytest.approx(0.144, abs=5e-4)
    assert So.flash_fraction_cp(1.04, 350, 212, 920.7)["fraccion_flash_integrada"] == pytest.approx(0.144, abs=5e-4)
    # steam tables (printed 0.146) against CoolProp's IAPWS-95: saturated water at 350 °F flashed to 1 atm
    T = (350 - 32) / 1.8 + 273.15
    assert So.flash_fraction_coolprop("Water", T)["fraccion_flash"] == pytest.approx(0.146, abs=1e-3)


def test_cl_ex3_9_toluene_pool_evaporation():
    r = So.pool_evaporation_cl(0.092, 0.0371 * ATM, 298.0, math.pi * 1.5 ** 2 / 4)
    assert r["K_m_s"] * 100 == pytest.approx(0.482, abs=5e-4)                 # printed 0.482 cm/s
    assert r["flujo_masico_kg_s"] * 60 == pytest.approx(0.0714, rel=3e-3)       # printed 0.0714 kg/min


# ----------------------------------------------------------------------------------------------- dispersion
def test_cl_table_5_2_and_5_3_spot_values():
    sy, sz = Dm.sigmas_plume(1000.0, "B", "rural")
    assert (float(sy), float(sz)) == (pytest.approx(152.5, abs=0.1), pytest.approx(120.0))   # Ex. 5-1: 153, 120
    sy, sz = Dm.sigmas_puff(100.0, "B")
    assert (float(sy), float(sz)) == (pytest.approx(9.7, abs=0.05), pytest.approx(15.3, abs=0.05))   # Ex. 5-2


def test_cl_ex5_1_elevated_h2s_plume():
    Q, u, H, M, T = 10.0, 3.5, 100.0, 0.03408, 303.0
    assert Dm.stability_class(3.5, insolation="fuerte") == "B"
    c = float(Dm.plume_concentration(Q, u, 1000.0, H=H, cls="B"))
    assert c == pytest.approx(3.50e-5, rel=4e-3)                              # (a) 35.0 mg/m³
    assert float(Dm.kgm3_to_ppm(c, M, T)) == pytest.approx(25.6, abs=0.06)    # (a) 25.6 ppm
    # (b) maximum ground concentration: printed x = 589 m; printed Cmax 5.17e-4 kg/m³ is 10x too high (the book
    # substitutes 100 kg/s); with Q = 10 kg/s it is 5.16e-5 kg/m³ = 37.7 ppm, as part (c) confirms.
    # The book's method: σz = Hr/√2 (eq. 5-22) → x, then Cmax = 2Q/(e π u Hr²)·σz/σy (eq. 5-21).
    from scipy.optimize import brentq, minimize_scalar
    xb = brentq(lambda x: float(Dm.sigmas_plume(x, "B")[1]) - H / math.sqrt(2), 100, 2000)
    assert xb == pytest.approx(589, abs=0.5)
    sy, sz = (float(s) for s in Dm.sigmas_plume(xb, "B"))
    assert sy == pytest.approx(91.6, abs=0.05)
    cmax_book = 2 * Q / (math.e * math.pi * u * H ** 2) * sz / sy
    assert cmax_book == pytest.approx(5.16e-5, rel=3e-3)
    assert cmax_book != pytest.approx(5.17e-4, rel=0.5)                      # the printed value is the erratum
    # the exact maximum of the Gaussian (σz/σy is not constant) is 0.75 % farther and within 0.1 % in value
    opt = minimize_scalar(lambda x: -float(Dm.plume_concentration(Q, u, x, H=H, cls="B")), bounds=(100, 5000),
                          method="bounded", options={"xatol": 0.01})
    assert opt.x == pytest.approx(593.4, abs=0.5)
    assert -opt.fun == pytest.approx(cmax_book, rel=1e-3)
    # (c) release giving 10 ppm at the maximum: printed 2.64 kg/s (13.7 mg/m³)
    c10 = Dm.ppm_to_kgm3(10.0, M, T)
    assert c10 * 1e6 == pytest.approx(13.7, abs=0.05)
    assert Q * c10 / cmax_book == pytest.approx(2.64, abs=0.015)


def test_cl_ex5_2_ground_puff():
    c = float(Dm.puff_center_concentration(10.0, 100.0, cls="B"))
    assert c == pytest.approx(8.85e-4, rel=2e-3)                              # printed 8.85e-4 kg/m³
    assert float(Dm.kgm3_to_ppm(c, 0.03408, 303.0)) == pytest.approx(647, abs=1.5)
    assert 100.0 / 3.5 == pytest.approx(28.6, abs=0.05)                      # arrival time


@pytest.mark.parametrize("u,ins,cls", [(1.99, "fuerte", "A"), (2.0, "fuerte", "A-B"), (2.99, "fuerte", "A-B"),
                                        (3.0, "fuerte", "B"), (4.99, "fuerte", "B"), (5.0, "fuerte", "C"),
                                        (6.0, "fuerte", "C"), (5.99, "moderada", "C-D"), (6.0, "moderada", "D"),
                                        (2.99, "ligera", "C"), (1.99, "ligera", "B"), (5.0, "ligera", "D")])
def test_pasquill_day_edges(u, ins, cls):
    assert Dm.stability_class(u, insolation=ins) == cls


@pytest.mark.parametrize("u,cloud,cls", [(1.99, "despejado", "F"), (2.99, "despejado", "F"), (3.0, "despejado", "E"),
                                          (4.99, "despejado", "E"), (5.0, "despejado", "D"), (2.0, "nublado", "E"),
                                          (3.0, "nublado", "D"), (9.0, "despejado", "D")])
def test_pasquill_night_edges(u, cloud, cls):
    assert Dm.stability_class(u, night_cloud=cloud) == cls


def test_overcast_is_D_and_bad_input_raises():
    assert Dm.stability_class(1.0, insolation="cubierto") == "D"
    with pytest.raises(ValueError):
        Dm.stability_class(3.0)
    with pytest.raises(ValueError):
        Dm.sigmas_plume(100.0, "G")


def test_mixed_class_is_average_of_its_two_classes():
    a, b, ab = (Dm.sigmas_plume(500.0, k) for k in ("A", "B", "A-B"))
    assert float(ab[0]) == pytest.approx((float(a[0]) + float(b[0])) / 2)
    assert float(ab[1]) == pytest.approx((float(a[1]) + float(b[1])) / 2)


@pytest.mark.parametrize("cls", ["A", "B", "C", "D", "E", "F"])
def test_sigmas_continuous_and_increasing_across_classes(cls):
    xs = [100.0, 1000.0, 10000.0]
    for terrain in ("rural", "urban"):
        sy = [float(Dm.sigmas_plume(x, cls, terrain)[0]) for x in xs]
        assert sy == sorted(sy)
    # more stable class → narrower plume at the same distance (rural)
    order = [float(Dm.sigmas_plume(1000.0, k)[1]) for k in "ABCDEF"]
    assert order == sorted(order, reverse=True)


def test_distance_to_threshold_lands_on_the_threshold():
    conc = lambda x: Dm.plume_concentration(1.0, 2.0, x, cls="D")  # noqa: E731
    d = Dm.distance_to_concentration(conc, 1e-5)
    assert float(conc(d)) == pytest.approx(1e-5, rel=1e-5)
    # elevated source: concentration rises then falls; the farthest crossing is returned
    conc_h = lambda x: Dm.plume_concentration(10.0, 3.5, x, H=100.0, cls="B")  # noqa: E731
    dh = Dm.distance_to_concentration(conc_h, 2e-5)
    assert dh > 589 and float(conc_h(dh)) == pytest.approx(2e-5, rel=1e-5)
    assert Dm.distance_to_concentration(conc_h, 1e-3) is None


def test_dense_gas_flag_at_its_threshold():
    # continuous: criterion³ = g0·sqrt(q0)·u^-2.5 → q0 exactly at 0.15
    M, T, u = 0.0709, 239.0, 2.0
    chk = Dm.dense_gas_check(M, T, 293.15, u, Q_kg_s=1.0)
    g0, rho0 = chk["g0_m_s2"], chk["densidad_nube_kg_m3"]
    q0 = (0.15 ** 3 * u ** 2.5 / g0) ** 2
    at = Dm.dense_gas_check(M, T, 293.15, u, Q_kg_s=q0 * rho0 * (1 + 1e-9))
    below = Dm.dense_gas_check(M, T, 293.15, u, Q_kg_s=q0 * rho0 * 0.99)
    assert at["denso"] and at["criterio"] == pytest.approx(0.15, rel=1e-6)
    assert not below["denso"]
    # instantaneous: criterion = sqrt(g0 V0^(1/3))/u = 0.20
    V0 = (0.20 * u) ** 6 / g0 ** 3
    assert Dm.dense_gas_check(M, T, 293.15, u, Qstar_kg=V0 * rho0 * (1 + 1e-9))["denso"]
    assert not Dm.dense_gas_check(M, T, 293.15, u, Qstar_kg=V0 * rho0 * 0.98)["denso"]
    # lighter than air: never dense
    assert not Dm.dense_gas_check(0.016, 293.15, 293.15, 2.0, Q_kg_s=100.0)["denso"]


def test_dispersion_tool_refuses_gaussian_for_a_dense_cloud():
    o = T.gas_dispersion("continua", wind_speed_m_s=2.0, ambient_temperature_K=293.15, stability_class="F",
                         release_rate_kg_s=10.0, compound_name="chlorine", release_temperature_K=239.0,
                         thresholds=[{"nombre": "ERPG-2 (dato del usuario)", "ppm": 3}])
    assert o["ok"] and o["valido_gaussiano"] is False and o["gas_denso"]["denso"]
    assert "umbrales" not in o and "concentracion_eje" not in o
    assert "NO SE CALCULA" in o["resultado"]


def test_dispersion_tool_passive_cloud_is_valid():
    o = T.gas_dispersion("instantanea", wind_speed_m_s=5.0, ambient_temperature_K=293.15, night_cloud="nublado",
                         release_mass_kg=50.0, compound_name="methane", release_height_m=0.0,
                         thresholds=[{"nombre": "50 % LFL", "fraccion_lfl": 0.5}])
    assert o["ok"] and o["valido_gaussiano"] is True and o["clase_estabilidad"] == "D"
    d = o["umbrales"][0]["distancia_m"]
    assert d is not None and o["umbrales"][0]["C_ppm"] == pytest.approx(0.5 * 0.044e6, rel=1e-6)
    assert "dosis_kg_s_m3" in o["concentracion_eje"][0]


def test_dispersion_tool_never_defaults_the_weather():
    o = T.gas_dispersion("continua", release_rate_kg_s=1.0, compound_name="hydrogen sulfide")
    assert not o["ok"] and "wind_speed_m_s" in o["error"] and "ambient_temperature_K" in o["error"]


def test_dispersion_tool_ex5_1_and_lopa_link():
    o = T.gas_dispersion("continua", wind_speed_m_s=3.5, ambient_temperature_K=303.0, insolation="fuerte",
                         release_rate_kg_s=10.0, molar_mass_kg_mol=0.03408, release_height_m=100.0,
                         release_temperature_K=303.0,
                         thresholds=[{"nombre": "10 ppm (usuario)", "ppm": 10}, {"nombre": "100 ppm", "ppm": 100}],
                         hazop_row=4, severity_criterion=[{"severidad": 4, "endpoint": "10 ppm (usuario)",
                                                           "distancia_min_m": 500}])
    assert o["ok"] and o["clase_estabilidad"] == "B"
    # H2S is heavier than air and 10 kg/s is dense at the source by Britter-McQuaid; elevated → computed WITH a warning
    assert o["gas_denso"]["denso"] and o["valido_gaussiano"] == "con aviso" and "SUBESTIMAR" in o["aviso_gas_denso"]
    row = next(r for r in o["concentracion_eje"] if r["x_m"] == 1000.0)
    assert row["C_ppm"] == pytest.approx(25.6, abs=0.06)
    ten = next(t for t in o["umbrales"] if t["nombre"].startswith("10 ppm"))
    assert ten["distancia_m"] > 1000 and o["umbrales"][1]["distancia_m"] is None   # 100 ppm never reached
    link = o["para_lopa"]
    assert link["severidad_calculada"] == 4
    assert link["overrides_lopa"] == {"4": {"severidad": 4, "severidad_fuente":
                                            "calculada (análisis de consecuencias M12, criterio del usuario)"}}
    assert any("Cribado preliminar" in a for a in o["assumptions"])


# ----------------------------------------------------------------------------------------------- fires
@pytest.mark.parametrize("D,H,x", [(10, 20, 10), (10, 20, 30), (20, 30, 15), (5, 15, 100), (1.03, 2.305, 3.564)])
def test_numeric_view_factor_matches_mudan_analytic(D, H, x):
    a = Fi.view_factor_vertical_cylinder(D, H, x)
    n = Fi.view_factor_cylinder_numeric(D, H, 0.0, x)
    for k in ("F_vertical", "F_horizontal", "F_max"):
        assert n[k] == pytest.approx(a[k], rel=2e-4, abs=1e-6)


def test_yellow_book_tilted_view_factors():
    # YB §6.6.3 step 12 and §6.6.4 step 21 (Mudan tilted cylinder, printed)
    a = Fi.view_factor_cylinder_numeric(42.445, 46.7725, 50.8286, 100.0, n_phi=720, n_s=400)
    assert (a["F_vertical"], a["F_horizontal"], a["F_max"]) == (pytest.approx(0.091915, rel=2e-4),
                                                                pytest.approx(0.029146, rel=2e-4),
                                                                pytest.approx(0.0964, abs=1e-4))
    b = Fi.view_factor_cylinder_numeric(31.97, 37.91, 51.9, 100.0, n_phi=720, n_s=400)
    assert (b["F_vertical"], b["F_horizontal"], b["F_max"]) == (pytest.approx(0.0466, rel=4e-3),
                                                                pytest.approx(0.00977, rel=4e-3),
                                                                pytest.approx(0.0476, rel=4e-3))


def test_yellow_book_6_6_3_confined_benzene_pool_fire():
    pf = Fi.pool_fire(42.445, 4.015e7, 1.0, 353.206, 288.15, 1.2243, 1.0, burning_rate=0.085, wind_speed=5.0,
                      model="tno", radiant_fraction=0.40, nu_air=7.5133e-6, tau_user=0.71474)
    g = pf["geometria"]
    assert g["u_c_m_s"] == pytest.approx(3.06866, rel=5e-5)
    assert g["u_estrella"] == pytest.approx(1.62937, rel=5e-5)
    assert g["L_D"] == pytest.approx(1.101938, rel=1e-4)
    assert g["longitud_llama_m"] == pytest.approx(46.7725, rel=1e-4)
    assert g["tan_cos"] == pytest.approx(1.94315, rel=1e-5)
    assert g["inclinacion_grados"] == pytest.approx(50.8286, abs=1e-3)
    assert g["SEP_max_kW_m2"] * 1e3 == pytest.approx(25.24e4, rel=2e-4)
    assert pf["E_kW_m2"] * 1e3 == pytest.approx(6.648e4, rel=2e-4)          # printed rounded to 6.6e4
    q, F, tau = pf["_flux"](100.0)
    assert F == pytest.approx(0.0964, abs=1e-4)
    assert q * 1e3 == pytest.approx(4.581e3, rel=2e-3)                       # printed 4.581e3 J/m²s


def test_yellow_book_6_6_4_unconfined_benzene_pool_fire():
    pf = Fi.pool_fire(31.97, 4.015e7, 1.0, 353.206, 288.15, 1.2243, 1.0, burning_rate=0.085, wind_speed=5.0,
                      model="tno", radiant_fraction=0.40, nu_air=7.513e-6, tau_user=0.71)
    g = pf["geometria"]
    assert g["u_c_m_s"] == pytest.approx(2.792, abs=1e-3)
    assert g["u_estrella"] == pytest.approx(1.79, abs=5e-3)
    assert g["longitud_llama_m"] == pytest.approx(37.91, rel=3e-3)          # print 37.91; its own L/D gives 37.98
    assert g["inclinacion_grados"] == pytest.approx(51.9, abs=0.05)
    assert pf["E_kW_m2"] * 1e3 == pytest.approx(6.34e4, rel=3e-3)
    # print 2.141e3 uses L = 37.91, which its own L/D does not give (37.98): 0.8 % from the chain recomputed
    assert pf["_flux"](100.0)[0] * 1e3 == pytest.approx(2.141e3, rel=1e-2)


def test_nureg_1805_ex_5_11_1_solid_flame_shokri_beyler():
    D = math.sqrt(4 * 9.0 * FT ** 2 / math.pi)                              # 9 ft² spill → D = 1.03 m
    pf = Fi.pool_fire(D, 46000e3, 1.0, 600.0, 298.0, 1.18, 1.0, burning_rate_inf=0.039, k_beta=0.7,
                      flame_height="heskestad", sep_method="shokri_beyler")
    assert D == pytest.approx(1.03, abs=5e-3)
    assert pf["geometria"]["Q_kW"] == pytest.approx(771.52, rel=2e-3)
    assert pf["geometria"]["altura_llama_m"] == pytest.approx(2.305, abs=5e-3)
    assert pf["E_kW_m2"] == pytest.approx(56.88, abs=0.01)
    assert pf["_flux"](10 * FT + D / 2)[0] == pytest.approx(3.05, abs=0.02)   # printed 3.05 kW/m²


def test_yellow_book_6_6_5_propane_bleve_fireball():
    m = 45 * 0.85 * 517
    fb = Fi.fireball_tno(m, 46.013e6, P_burst_Pa=16e5, Hvap_J_kg=0.426134e6, Cp_l_J_kgK=2.582e3, T_air_K=283.15)
    assert m == pytest.approx(19775, abs=1)
    assert fb["radio_m"] == pytest.approx(80.68, abs=0.01)
    assert fb["duracion_s"] == pytest.approx(11.15, abs=0.01)
    assert fb["altura_centro_m"] == pytest.approx(161.366, abs=0.01)
    assert fb["fraccion_radiada"] == pytest.approx(0.314, abs=5e-4)
    assert fb["dH_J_kg"] == pytest.approx(41.153e6, rel=1e-4)
    assert fb["E_kW_m2"] * 1e3 == pytest.approx(28.02e4, rel=2e-4)
    q, F, tau = Fi.fireball_flux(fb, 200.0, None, 283.15, tau_user=0.7335)   # τ read from the YB graphs
    assert F == pytest.approx(0.09856, abs=2e-5)
    # printed 20.361e3, but 28.02e4 × 0.09856 × 0.7335 = 20.26e3: the print's last product is the erratum
    assert q * 1e3 == pytest.approx(20.26e3, rel=1e-3)
    assert q * 1e3 != pytest.approx(20.361e3, rel=2e-3)
    tau_b, note = Fi.transmissivity_tno(0.7, 288.15, 176.296)               # YB step 13a: Bagster τ = 0.67
    assert tau_b == pytest.approx(0.67, abs=5e-3) and "fuera de 1e4–1e5" in note   # 21.04e4 N/m, outside its range


def test_nistir_6546_problem_3_point_source():
    Hc = 46e6
    Qm = 41.4e9 / Hc
    assert Fi.jet_fire_point_source(Qm, Hc, 0.20, 260.0) == pytest.approx(9.7, abs=0.06)
    d = Fi.distance_to_flux(lambda r: Fi.jet_fire_point_source(Qm, Hc, 0.20, r), 1.4, 1.0)
    assert d == pytest.approx(690, abs=5)


def test_fire_tool_pool_needs_user_data_and_labels_examples():
    assert not T.fire_radiation("charco", compound_name="hexane", pool_diameter_m=10.0)["ok"]   # no temperature
    o = T.fire_radiation("charco", ambient_temperature_K=298.15, compound_name="hexane", pool_diameter_m=10.0,
                         wind_speed_m_s=3.0, relative_humidity=0.5, example_endpoints="epa_rmp",
                         endpoints_kW_m2=[{"nombre": "dolor (usuario)", "kW_m2": 4.73}], receptor_distance_m=40.0,
                         exposure_time_s=40.0, hazop_row=2,
                         severity_criterion=[{"severidad": 5, "endpoint": "EJEMPLO epa_rmp", "distancia_min_m": 20}])
    assert o["ok"] and o["modelo"]["SEP_usada"] in ("mudan_croce", "shokri_beyler")
    eps = {e["nombre"]: e["distancia_m"] for e in o["puntos_finales"]}
    assert eps["dolor (usuario)"] > eps["EJEMPLO epa_rmp"] > 5.0
    assert any("EJEMPLO" in a and "40 CFR 68.22" in a for a in o["assumptions"])
    assert o["para_lopa"]["severidad_calculada"] == 5
    assert 0 <= o["para_lopa"]["modificador_sugerido"]["probabilidad"] <= 1


def test_fire_tool_fireball_from_compound():
    o = T.fire_radiation("bola_fuego", ambient_temperature_K=283.15, compound_name="propane", mass_kg=19775,
                         burst_pressure_Pa=16e5, transmissivity=0.7335, distances_m=[200.0])
    assert o["ok"]
    assert o["modelo"]["radio_m"] == pytest.approx(80.68, abs=0.01)
    # ΔHc, ΔHv and Cp from thermo instead of the YB inputs: same order, within 5 % of the printed 20.26 kW/m²
    assert o["radiacion"][0]["q_kW_m2"] == pytest.approx(20.26, rel=0.05)


# ----------------------------------------------------------------------------------------------- explosions
def test_kinney_graham_equals_twice_the_free_air_form():
    for ze in (0.1, 1.0, 8.36, 30.0, 100.0):
        free = 808 * (1 + (ze / 4.5) ** 2) / (math.sqrt(1 + (ze / 0.048) ** 2) * math.sqrt(1 + (ze / 0.32) ** 2)
                                               * math.sqrt(1 + (ze / 1.35) ** 2))
        assert Ex.kinney_graham_ratio(ze) == pytest.approx(2 * free, rel=1e-12)


def test_cl_ex6_10_one_kg_tnt_at_30_m():
    r = Ex.tnt_overpressure(1.0, 30.0)
    assert r["ze_m_kg13"] == pytest.approx(30.0)
    assert r["ps_pa"] == pytest.approx(0.055, rel=0.03)        # printed 0.055 read from Fig. 6-24; eq. gives 0.0563
    assert r["sobrepresion_Pa"] / 1e3 == pytest.approx(5.6, rel=0.03)


def test_cl_ex6_11_methane_vce_tnt_equivalent():
    m_tnt = Ex.tnt_mass(1000.0, 802.3e3 / 0.016, 0.02)
    assert m_tnt == pytest.approx(214, abs=0.5)
    r = Ex.tnt_overpressure(m_tnt, 50.0)
    assert r["ze_m_kg13"] == pytest.approx(8.4, abs=0.05)
    assert r["ps_pa"] == pytest.approx(0.25, rel=0.02)
    assert r["sobrepresion_Pa"] / 1e3 == pytest.approx(25, rel=0.03)


def test_cl_ex6_13_tnt_and_overpressure_probits():
    r = Ex.tnt_overpressure(5000.0, 150.0)
    assert r["ze_m_kg13"] == pytest.approx(8.80, abs=0.04)      # printed 8.80 (exact 8.77)
    # the book then reads 0.20 off the chart (eq. 6-29 gives 0.235); the probits use its 20 300 N/m²
    assert Ef.overpressure_probit(20300, "muerte_hemorragia_pulmonar")["Y"] == pytest.approx(-8.56, abs=0.01)
    ear = Ef.overpressure_probit(20300, "ruptura_timpano")
    assert ear["Y"] == pytest.approx(3.54, abs=0.01) and ear["probabilidad"] == pytest.approx(0.072, abs=0.002)


def test_tnt_distance_inverts_the_overpressure():
    for p in (1e3, 6.895e3, 20e3, 100e3):
        d = Ex.tnt_distance_to(214.0, p)
        assert Ex.tnt_overpressure(214.0, d)["sobrepresion_Pa"] == pytest.approx(p, rel=1e-6)


def test_explosion_tool_ex6_11_and_receptor():
    o = T.explosion_overpressure("tnt", flammable_mass_kg=1000.0, efficiency=0.02, heat_of_combustion_J_kg=802.3e3 / 0.016,
                                 distances_m=[50.0], example_endpoints="epa_rmp", receptor_distance_m=50.0,
                                 hazop_row=1)
    assert o["ok"] and o["masa_TNT_kg"] == pytest.approx(214, abs=0.5)
    assert o["sobrepresion"][0]["sobrepresion_kPa"] == pytest.approx(25.5, abs=0.1)
    assert o["puntos_finales"][0]["nombre"] == "EJEMPLO epa_rmp"
    assert o["receptor"]["probits"]["ruptura_timpano"]["probabilidad"] > 0.05
    assert "sin criterio" in o["para_lopa"]["nota"].lower() or "Sin criterio" in o["para_lopa"]["nota"]
    assert not T.explosion_overpressure("tnt", flammable_mass_kg=1000.0, compound_name="methane")["ok"]   # η is the user's
    assert "pendientes" in T.explosion_overpressure("multi_energia")["error"]


# ----------------------------------------------------------------------------------------------- effects
def test_cl_ex2_2_and_table_2_4():
    assert Ef.probability_from_probit(4.39) == pytest.approx(0.271, abs=5e-4)
    for pct, y in ((0.01, 2.67), (0.10, 3.72), (0.27, 4.39), (0.50, 5.00), (0.90, 6.28), (0.99, 7.33),
                   (0.999, 8.09), (0.001, 1.91)):
        assert Ef.probit_from_probability(pct) == pytest.approx(y, abs=0.006)


@pytest.mark.parametrize("dose,p", [(1.2e7, 0.04), (1.4e7, 0.09), (1.6e7, 0.16), (1.8e7, 0.24), (2.0e7, 0.33),
                                    (2.5e7, 0.55), (3.0e7, 0.73), (3.2e7, 0.78), (4.5e7, 0.95)])
def test_pgs1_table_4_4_eisenberg_lethality(dose, p):
    # dose = t·q^(4/3) [(W/m²)^(4/3)·s]; first row (1e7 → printed 0.015, eq. 0.0133) is a print erratum, left out
    q = 10.0                                                          # kW/m², t chosen to give the dose
    t = dose / (q * 1000) ** (4 / 3)
    assert Ef.thermal_probit(q, t)["probabilidad"] == pytest.approx(p, abs=0.006)


def test_pgs1_eardrum_and_window_probits():
    assert Ef.overpressure_probit(3e5, "ruptura_timpano_pgs1")["Y"] == pytest.approx(6.52, abs=0.01)    # ex. 7.2
    assert Ef.overpressure_probit(5e3, "rotura_ventanas_pre1975_pgs1")["Y"] == pytest.approx(6.086, abs=0.002)
    w = Ef.overpressure_probit(5e3, "rotura_ventanas_post1975_pgs1")
    assert w["Y"] == pytest.approx(4.968, abs=0.002) and w["probabilidad"] == pytest.approx(0.49, abs=0.01)


def test_inverse_effect_functions_round_trip():
    for m in Ef.THERMAL:
        q = Ef.flux_for_probability(0.01, 40.0, m)
        assert Ef.thermal_probit(q, 40.0, m)["probabilidad"] == pytest.approx(0.01, rel=1e-6)
    p = Ef.overpressure_for_probability(0.5, "ruptura_timpano")
    assert p == pytest.approx(43500, rel=0.05)                       # C&L Ex. 2-3: 50 % at 43 500 N/m² (Eisenberg data)


# ----------------------------------------------------------------------------------------------- source tool + LOPA
def test_source_tool_gas_from_compound_and_missing_data():
    o = T.release_source_term("gas", hole_diameter_m=0.002, pressure_Pa=15.01e5, temperature_K=298.0,
                              compound_name="nitrogen")
    assert o["ok"] and o["resultado"]["ahogado"]
    assert o["resultado"]["flujo_masico_kg_s"] == pytest.approx(1.09e-2, rel=0.01)   # γ from thermo ≈ 1.40
    miss = T.release_source_term("liquido", density_kg_m3=879.4)
    assert not miss["ok"] and "hole_diameter_m" in miss["error"] and "pressure_Pa" in miss["error"]


def test_calculated_severity_feeds_lopa(tmp_path):
    from packages.hazop.book.cases import CASES, write_all
    from packages.lopa.tool import lopa_from_hazop
    files = write_all(tmp_path / "dexpi", CASES)
    path = next(iter(files.values()))
    base = lopa_from_hazop(dexpi_path=str(path), tolerable_by_severity={"5": 1e-6})
    row = base["hoja_lopa"][0]
    import re
    n = re.search(r"#(\d+)", row["escenario"]).group(1)
    sev = T._lopa_link(int(n), {"x": 500.0}, [{"severidad": 5, "endpoint": "x", "distancia_min_m": 100}], None)
    o = lopa_from_hazop(dexpi_path=str(path), tolerable_by_severity={"5": 1e-6}, overrides=sev["overrides_lopa"])
    r = next(x for x in o["hoja_lopa"] if x["escenario"] == row["escenario"])
    assert r["severidad_usada"] == 5 and r["origen_severidad"].startswith("calculada")
    assert r["frecuencia_tolerable_por_año"] == 1e-6 and "calculada" in r["origen_tolerable"]
