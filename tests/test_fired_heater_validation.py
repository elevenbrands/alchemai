"""Validation of the fired-heater tools against literature (bloque M18b): gas emissivity, radiant section, convection bank.

Every published number below was READ on 2026-11-14 in a source that opened without login:
  [KERN]   D. Q. Kern, Process Heat Transfer, McGraw-Hill 1950, chapter 19, examples 19.1, 19.2, 19.4, 19.5 and Table 19.1.
           Full text of the scan at the Internet Archive: https://archive.org/details/unset0000unse_z6e9 (the book itself,
           not only the FOSSEE transcription used in M18:
           https://tbc-python.fossee.in/convert-notebook/Process_Heat_Transfer_by_D._Q._Kern/Chapter_19_Furnace_Calculations_4.ipynb).
  [NELSON] W. L. Nelson, Petroleum Refinery Engineering, 2nd ed., McGraw-Hill 1941, chapter «Tubestill heaters», Table 64 and
           example 68 (design of a pipestill): https://archive.org/details/dli.ernet.476244
  [ZUK]    A. Žukauskas, «Heat Transfer from Tubes in Crossflow», Advances in Heat Transfer 8 (1972) 93-160, eqs. (39)-(44):
           https://archive.org/details/AdvancesInHeatTransfer
  [INC]    Incropera, DeWitt, Bergman, Lavine, Fundamentals of Heat and Mass Transfer, example 7.7 (staggered bank), as
           transcribed in the FOSSEE Python Textbook Companion:
           https://tbc-python.fossee.in/convert-notebook/Fundamentals_of_Heat_and_Mass_Transfer/Chapter_7.ipynb
  [CEN]    Y. A. Çengel, Heat and Mass Transfer: A Practical Approach, examples 7-7 (in-line bank), 13-13 and 13-14 (Hottel
           chart readings), as transcribed in the FOSSEE companion:
           https://tbc-python.fossee.in/convert-notebook/Heat_And_Mass_Transfer_-_A_Practical_Approach_by_Y._A._Cengel/ch7.ipynb
           (and ch13.ipynb)
  [AHTT]   J. H. Lienhard IV and V, A Heat Transfer Textbook, 6th ed. (2024), example 10.11 (readings of the charts of
           Alberti et al. 2015 for CO2): https://ahtt.mit.edu/wp-content/uploads/2024/04/AHTTv600.pdf
  [LECK]   B. Leckner, Combustion and Flame 19 (1972) 33-48, total emissivity of H2O and CO2; coefficients and formulas as
           listed in the SIERRA/Fuego theory manual of Sandia (SAND2024-07984O), section «Absorptivity model», Tables 3.8-3.11:
           https://www.sandia.gov/files/sierra/Fuego_Users_5_20/theory/mathmodels/absorptivity.html
           (the original paper was not opened).
  [GAN]    V. Ganapathy, Steam Generators and Waste Heat Boilers, CRC Press 2015, Appendix E «Calculations with Finned
           Tubes»: Table E.2 (revised ESCOA, «Source: Fintube Technologies, Tulsa»), eqs. E.1-E.24, examples E.1 and E.10,
           transcribed 2026-11-14 from an online copy of the book whose authorization could not be verified (link
           deliberately not kept). Re-check against a licensed copy before citing as validation.
  [EST]    Esteem Projects, «Free Fired Heater Design Tutorial - Convection Section Design» (procedure «taken from the Escoa
           manual»), fin-tube sample bank, read 2026-11-14:
           https://www.esteemprojects.com/library/Fired-Heater-Design/page4_3.htm
No network, no DWSIM. Old examples are in Btu, ft and °F: they are converted here, never retyped in SI by hand."""
from __future__ import annotations

import json
import math
import types

import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from packages.energy import combustion as CB
from packages.energy import fired_heater as FH

P = 101325.0
ATM = 101325.0
FT, FT2, BTU_H, BTU_LB, LB, HBTU = 0.3048, 0.09290304, 0.29307107, 2326.0, 0.45359237, 5.678263   # HBTU: Btu/(h ft2 °F) in W/m2K


def F2K(t_f: float) -> float:
    return (t_f + 459.67) / 1.8


def K2F(t_k: float) -> float:
    return t_k * 1.8 - 459.67


# ----------------------------------------------------------------------------------------------------------------------------
# Reference implementation of Leckner (1972), independent of the WSGG of the engine. [LECK]
#   ln(eps0) = sum_i sum_j c_ij theta^j lambda^i, theta = T/1000 K, lambda = log10(p_a L / bar cm)
#   eps/eps0 = 1 + exp(-xi (lambda_max - lambda)^2) [(A PE + B)/(PE + A + B - 1) - 1]
#   overlap  = [zeta/(10.7 + 101 zeta) - 0.0089 zeta^10.4] (log10((pw + pc) L))^2.76,  zeta = pw/(pw + pc)
# The manual prints the equivalent pressure of H2O as p (1 + 1.49 (p_CO2/p) sqrt(273/T)); the partial pressure there can only
# be that of H2O itself. The coefficient of Leckner's paper as this author recalls it is 4.9 (NOT read today): both are run.
LECKNER_CO2 = ((-3.9781, 2.7353, -1.9882, 0.31054, 0.015719), (1.9326, -3.5932, 3.7247, -1.4535, 0.20132),
               (-0.35366, 0.61766, -0.84207, 0.39859, -0.063356), (-0.080181, 0.31466, -0.19973, 0.046532, -0.0033086))
LECKNER_H2O = ((-2.2118, -1.1987, 0.035596), (0.85667, 0.93048, -0.14391), (-0.10838, -0.17156, 0.045915))
H2O_PE_PRINTED, H2O_PE_RECALLED = 1.49, 4.9


def _leckner_scale(c, T_K: float, paL_bar_cm: float) -> float:
    theta, lam = T_K / 1000.0, math.log10(paL_bar_cm)
    return math.exp(sum(sum(cij * theta ** j for j, cij in enumerate(ci)) * lam ** i for i, ci in enumerate(c)))


def leckner_co2(T_K: float, p_co2_bar: float, L_cm: float, p_bar: float = 1.01325) -> float:
    theta, lam = T_K / 1000.0, math.log10(p_co2_bar * L_cm)
    PE = p_bar * (1.0 + 0.28 * p_co2_bar / p_bar)
    lam_max = math.log10(0.225 * theta ** 2) if T_K > 700.0 else math.log10(0.054 / theta ** 2)
    A, B, xi = 1.0 + 0.1 * theta ** -1.45, 0.23, 1.47
    return _leckner_scale(LECKNER_CO2, T_K, p_co2_bar * L_cm) * \
        (1.0 + math.exp(-xi * (lam_max - lam) ** 2) * ((A * PE + B) / (PE + A + B - 1.0) - 1.0))


def leckner_h2o(T_K: float, p_h2o_bar: float, L_cm: float, p_bar: float = 1.01325, pe_coefficient: float = H2O_PE_PRINTED) -> float:
    theta, lam = T_K / 1000.0, math.log10(p_h2o_bar * L_cm)
    PE = p_bar * (1.0 + pe_coefficient * p_h2o_bar / p_bar * math.sqrt(273.0 / T_K))
    lam_max = math.log10(13.2 * theta ** 2)
    A = 1.888 - 2.053 * math.log10(theta) if T_K > 750.0 else 1.888 - 2.053 * math.log10(0.75)
    B, xi = 1.1 * theta ** -1.4, 0.5
    return _leckner_scale(LECKNER_H2O, T_K, p_h2o_bar * L_cm) * \
        (1.0 + math.exp(-xi * (lam_max - lam) ** 2) * ((A * PE + B) / (PE + A + B - 1.0) - 1.0))


def leckner_mixture(T_K: float, p_h2o_atm: float, p_co2_atm: float, L_m: float, p_atm: float = 1.0,
                    pe_coefficient: float = H2O_PE_PRINTED) -> dict:
    pw, pc, L, p = p_h2o_atm * 1.01325, p_co2_atm * 1.01325, L_m * 100.0, p_atm * 1.01325
    ew, ec = leckner_h2o(T_K, pw, L, p, pe_coefficient), leckner_co2(T_K, pc, L, p)
    zeta, lam = pw / (pw + pc), math.log10((pw + pc) * L)
    overlap = (zeta / (10.7 + 101.0 * zeta) - 0.0089 * zeta ** 10.4) * lam ** 2.76 if lam > 0 else 0.0
    return {"emissivity": ew + ec - overlap, "H2O": ew, "CO2": ec, "overlap": overlap}


def _dilute(fn, T_K: float, pL_atm_m: float, **kw) -> float:
    """Standard emissivity of the charts: 1 bar total, vanishing partial pressure of the absorber."""
    pa = 1e-6
    return fn(T_K, pa, pL_atm_m * 1.01325 * 100.0 / pa, 1.0, **kw)


# ----------------------------------------------------------------------------------------------------------------------------
# 1. Gas emissivity
# ----------------------------------------------------------------------------------------------------------------------------
KERN_GAS = {"p_H2O": 0.1248, "p_CO2": 0.1084}                 # [KERN] ex. 19.1 and 19.4, atm
KERN_19_1 = {"T_K": F2K(1730), "L_m": 15.0 * FT, "q_CO2": 6500.0, "q_H2O": 14500.0, "q_black": 39000.0, "correction": 0.08,
             "printed": 0.496}                                # fluxes read by Kern in his Figs. 19.12-19.13, Btu/h ft2
KERN_19_4 = {"T_K": F2K(1500), "L_m": 0.565 * FT, "q_CO2": 1700.0, "q_H2O": 1050.0, "correction": 0.02}
EMISSIVITY_GRID = [(T, pL) for T in (800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0) for pL in (0.03, 0.1, 0.3, 1.0, 3.0)]


def emissivity_grid(pe_coefficient: float = H2O_PE_PRINTED) -> list:
    """Smith 1982 (engine) against Leckner 1972 at the two compositions of the Smith fit: pw/pc = 1 and 2 with pc = 0.1 atm."""
    rows = []
    for pw, pc in ((0.1, 0.1), (0.2, 0.1)):
        for T, pL in EMISSIVITY_GRID:
            L = pL / (pw + pc)
            smith = FH.gas_emissivity(T, pw * ATM, pc * ATM, L)["emissivity"]
            leck = leckner_mixture(T, pw, pc, L, pe_coefficient=pe_coefficient)["emissivity"]
            rows.append({"pw_atm": pw, "pc_atm": pc, "T_K": T, "pL_atm_m": pL, "smith_1982": smith, "leckner_1972": leck,
                         "deviation": smith / leck - 1.0})
    return rows


def test_leckner_reference_reproduces_published_chart_readings():
    # [AHTT] ex. 10.11: CO2 0.2 atm in N2, 1 atm, L = 0.40 m: 0.084 (text also rounds 0.085) at 1473 K; 0.073 at 573 K and
    # pL scaled by Tw/Tg (3.16 bar cm). Charts of Alberti et al. 2015.
    assert leckner_co2(1473.0, 0.2 * 1.01325, 40.0) == pytest.approx(0.084, rel=0.04)            # measured +3.0 %
    assert leckner_co2(573.0, 0.2 * 1.01325 * 573.0 / 1473.0, 40.0) == pytest.approx(0.073, rel=0.02)   # measured -1.1 %
    # [CEN] ex. 13-13 and 13-14, Hottel's charts at 1 atm: CO2 0.16 (1200 K, 0.30 m atm) and 0.11 (600 K, 0.15 m atm);
    # H2O 0.23 (1200 K, 0.48 m atm) and 0.25 (600 K, 0.24 m atm). Chart readings of a textbook: two significant figures.
    assert _dilute(leckner_co2, 1200.0, 0.30) == pytest.approx(0.16, rel=0.16)                   # measured -14 %
    assert _dilute(leckner_co2, 600.0, 0.15) == pytest.approx(0.11, rel=0.03)                    # measured -1 %
    for k in (H2O_PE_PRINTED, H2O_PE_RECALLED):
        assert _dilute(leckner_h2o, 1200.0, 0.48, pe_coefficient=k) == pytest.approx(0.23, rel=0.17)   # measured +15 %
        assert _dilute(leckner_h2o, 600.0, 0.24, pe_coefficient=k) == pytest.approx(0.25, rel=0.03)    # measured -2 %
        # same example at 2 atm total with 8 % H2O and 5 % CO2, L = 3 m: 1.1*0.16 + 1.4*0.23 - 0.048 = 0.45
        mix = leckner_mixture(1200.0, 0.16, 0.10, 3.0, p_atm=2.0, pe_coefficient=k)
        assert mix["emissivity"] == pytest.approx(0.45, rel=0.10)                                # measured -7.5 / -9.3 %
        assert mix["H2O"] == pytest.approx(1.4 * 0.23, rel=0.03) and mix["overlap"] == pytest.approx(0.048, abs=0.002)


def test_smith_wsgg_of_the_engine_against_single_gas_chart_readings():
    """The limits of the WSGG (one absorber only) against the same chart readings: this is the engine, not the reference."""
    one = lambda T, pw, pc, pL: FH.gas_emissivity(T, pw * ATM, pc * ATM, pL / (pw + pc))["emissivity"]   # noqa: E731
    assert one(1473.0, 0.0, 0.2, 0.08) == pytest.approx(0.084, rel=0.07)          # [AHTT] measured +5.6 %
    assert one(1200.0, 0.0, 0.05, 0.30) == pytest.approx(0.16, rel=0.13)          # [CEN] measured -12 %
    assert one(600.0, 0.0, 0.05, 0.15) == pytest.approx(0.11, rel=0.08)           # [CEN] measured +6 %
    assert one(600.0, 0.05, 0.0, 0.24) == pytest.approx(0.25, rel=0.05)           # [CEN] measured +4 %
    # water vapour alone at 1200 K is the worst point found: +25 % over the reading of Hottel's chart (Leckner: +15 %)
    assert one(1200.0, 0.05, 0.0, 0.48) / 0.23 == pytest.approx(1.25, abs=0.02)


def test_smith_wsgg_against_leckner_over_the_furnace_range():
    # measured: as printed (1.49) -12.3 % ... +17.6 %, mean +4.2 %, mean absolute 7.7 %; with 4.9: -12.4 % ... +14.4 %, mean
    # +1.5 %, mean absolute 6.3 %. Where a firebox works (pL = 1 atm m, 1000-1600 K): -0.4 ... +8.4 % and -2.4 ... +3.1 %
    for k, lo, hi, mean_abs, box_max in ((H2O_PE_PRINTED, -0.13, 0.18, 0.08, 0.09), (H2O_PE_RECALLED, -0.13, 0.15, 0.065, 0.035)):
        grid = emissivity_grid(k)
        dev = [r["deviation"] for r in grid]
        assert len(dev) == 70 and lo < min(dev) and max(dev) < hi
        assert sum(abs(d) for d in dev) / len(dev) < mean_abs
        box = [r["deviation"] for r in grid if r["pL_atm_m"] == 1.0 and 1000.0 <= r["T_K"] <= 1600.0]
        assert max(abs(d) for d in box) < box_max
    # the two forms of the equivalent pressure of H2O change Leckner by at most 6 %
    a, b = emissivity_grid(H2O_PE_PRINTED), emissivity_grid(H2O_PE_RECALLED)
    assert max(abs(x["leckner_1972"] / y["leckner_1972"] - 1.0) for x, y in zip(a, b)) < 0.065


def test_kern_charts_are_the_outlier_at_long_path_length():
    pw, pc = KERN_GAS["p_H2O"], KERN_GAS["p_CO2"]
    k = KERN_19_1
    kern_at_TG = (k["q_CO2"] + k["q_H2O"]) * (1 - k["correction"]) / k["q_black"]            # 0.495: what his charts say
    assert kern_at_TG == pytest.approx(k["printed"], abs=0.002)
    smith = FH.gas_emissivity(k["T_K"], pw * ATM, pc * ATM, k["L_m"])["emissivity"]
    assert smith == pytest.approx(0.398, abs=0.002)
    for coeff, expected in ((H2O_PE_RECALLED, 0.4025), (H2O_PE_PRINTED, 0.390)):
        leck = leckner_mixture(k["T_K"], pw, pc, k["L_m"], pe_coefficient=coeff)["emissivity"]
        assert leck == pytest.approx(expected, abs=0.002)
        assert smith / leck == pytest.approx(1.0, abs=0.025)          # two independent models: within 2.5 %
        assert k["printed"] / leck > 1.22                              # Kern's chart: 23-27 % above both
    # of Kern's 0.538 before the overlap, 0.372 is water vapour at 0.57 m atm and 1216 K; Hottel's later chart read by
    # Çengel gives 0.23 at 0.48 m atm and 1200 K, Leckner 0.27-0.30 with the pressure correction: the excess is in Fig. 19.13
    assert k["q_H2O"] / k["q_black"] == pytest.approx(0.372, abs=0.001)
    assert leckner_mixture(k["T_K"], pw, pc, k["L_m"], pe_coefficient=H2O_PE_RECALLED)["H2O"] == pytest.approx(0.30, abs=0.01)
    # short path (ex. 19.4): here Kern's charts are the LOW ones
    s = KERN_19_4
    black = 0.173 * ((1500 + 459.67) / 100.0) ** 4
    kern_short = (s["q_CO2"] + s["q_H2O"]) * (1 - s["correction"]) / black
    assert kern_short == pytest.approx(0.1056, abs=0.0005)
    smith_s = FH.gas_emissivity(s["T_K"], pw * ATM, pc * ATM, s["L_m"])["emissivity"]
    leck_s = leckner_mixture(s["T_K"], pw, pc, s["L_m"], pe_coefficient=H2O_PE_RECALLED)["emissivity"]
    assert smith_s / kern_short == pytest.approx(1.093, abs=0.01) and leck_s / kern_short == pytest.approx(1.14, abs=0.015)


# ----------------------------------------------------------------------------------------------------------------------------
# 2. Radiant section: Kern's furnace (examples 19.1 and 19.2, Fig. 19.16)
# ----------------------------------------------------------------------------------------------------------------------------
# Given by Kern: 50e6 Btu/h at 75 % efficiency (LHV) -> 66.67e6 Btu/h fired; oil 17 130 Btu/lb LHV; 25 % excess air = 17.44 lb
# air/lb; air at 400 °F; 0.3 lb atomising steam/lb; humidity 50 % of saturation at 60 °F; wall loss 2 % of the heat fired;
# 60 tubes 5 in OD on 8.5 in centres, 38.5 ft exposed, one row; tube 800 °F; box 38.5 x 20.46 x 14.92 ft, A_T = 3138 ft2;
# L = 15 ft; pCO2 0.1084, pH2O 0.1248; eps_G 0.496; F 0.635; alpha 0.937; convection: hc = 2.0 Btu/h ft2 °F on the tube area.
# His answer: «a balance will be struck at about 1700 °F», duty 37 050 000 Btu/h, 12 280 Btu/h ft2.
# NOT given by Kern: the analysis of the oil. It is back-calculated here from his own partial pressures and air ratio
# (C 0.8514, H 0.1203, rest taken as S): it returns his 17.44 lb air/lb within 0.4 % and his partial pressures within 0.5 %.
# The atomising steam enters this model as humidity of the combustion air (0.3/17.44 kg/kg): it then carries the sensible heat
# of the air preheat, 0.1 % of the heat fired more than Kern's «negligible».
KERN_OIL = {"type": "ultimate", "mass_fractions": {"C": 0.8514, "H": 0.1203, "S": 0.0283}, "LHV_J_kg": 17130.0 * BTU_LB}
KERN_AIR = {"T_K": F2K(400.0), "humidity_ratio_kg_kg": 0.00548 + 0.3 / 17.44}
KERN_BOX = {"tube_outside_diameter_m": 5.0 / 12 * FT, "tube_pitch_m": 8.5 / 12 * FT, "tube_effective_length_m": 38.5 * FT,
            "n_tubes": 60, "envelope_area_m2": 3138.0 * FT2, "volume_m3": 38.5 * 20.46 * 14.92 * FT ** 3}
KERN_19_1_ANSWER = {"TG_F": 1700.0, "Q_Btu_h": 37.05e6, "flux_Btu_h_ft2": 12280.0}
KERN_19_2_ANSWER = {"TG_F": 1540.0, "Q_Btu_h": 25.3e6, "flux_Btu_h_ft2": 8350.0}
# Example 19.2: the same furnace «fired with cracked gas fuel at 40 per cent excess air, using no air preheat», 50e6 Btu/h,
# 22.36 lb air/lb fuel; answer by the empirical equation of Wilson, Lobo and Hottel (19.15), fitted to tests of furnaces.
# NOT given: the gas (a C2H6/C2H4 surrogate with the same air demand is used: 22.5 lb/lb) nor the tube temperature
# (800 °F of example 19.1 is kept).
CRACKED_GAS = {"type": "gas", "composition": {"C2H6": 0.935, "C2H4": 0.065}, "T_K": F2K(60.0)}
AIR_60F = {"T_K": F2K(60.0), "humidity_ratio_kg_kg": 0.00548}


def kern_19_1(**overrides) -> dict:
    geometry = dict(KERN_BOX, **overrides.pop("geometry", {}))
    args = dict(fuel=KERN_OIL, air=KERN_AIR, pressure_Pa=P, fuel_flow_kg_s=66.67e6 * BTU_H / KERN_OIL["LHV_J_kg"],
                geometry=geometry, tube_emissivity=0.9, convective_coefficient_W_m2K=2.0 * HBTU, setting_loss_fraction=0.02,
                setting_loss_basis="LHV", excess_air_fraction=0.25, tube_wall_T_K=F2K(800.0))
    args.update(overrides)
    return FH.radiant_section(**args)


def kern_19_2(**overrides) -> dict:
    c = CB.solve_combustion(CRACKED_GAS, AIR_60F, P, excess_air_fraction=0.40)
    return kern_19_1(fuel=CRACKED_GAS, air=AIR_60F, excess_air_fraction=0.40, fuel_flow_kg_s=50e6 * BTU_H / c.LHV, **overrides)


def wilson_lobo_hottel(Q_fired_Btu_h: float, air_fuel_lb_lb: float, alpha_Acp_ft2: float) -> float:
    """[KERN] eq. (19.15): radiant duty, Btu/h."""
    return Q_fired_Btu_h / (1.0 + air_fuel_lb_lb / 4200.0 * math.sqrt(Q_fired_Btu_h / alpha_Acp_ft2))


def test_kern_19_1_inputs_are_reproduced_by_the_combustion_engine():
    c = CB.solve_combustion(KERN_OIL, KERN_AIR, P, excess_air_fraction=0.25)
    assert c.air_mass == pytest.approx(17.44 * 1.00548 + 0.3, rel=0.005)             # air + its humidity + steam, lb/lb
    assert c.flue["CO2"] / c.flue_mol == pytest.approx(0.1084, rel=0.005)
    assert c.flue["H2O"] / c.flue_mol == pytest.approx(0.1248, rel=0.005)
    # Kern: 476 Btu/lb of flue gas at 1730 °F above 60 °F, and 82 Btu/lb of air at 400 °F
    h_flue = (c.flue_sensible(F2K(1730.0)) - c.flue_sensible(F2K(60.0))) / c.flue_mass / BTU_LB
    assert h_flue == pytest.approx(476.0, rel=0.012) and h_flue < 476.0              # measured 471.1 (-1.0 %)
    assert c.credits / BTU_LB == pytest.approx(82.0 * 17.44, rel=0.02)
    # geometry: his cold plane and refractory ratio
    row = FH.tube_row_alpha(8.5, 5.0)
    assert row["alpha"] * 60 * 8.5 / 12 * 38.5 == pytest.approx(1500.0, rel=0.03)    # he rounds 25.7 * 0.937 to 25 ft2/tube
    assert 2.0 / 3.0 * (38.5 * 20.46 * 14.92) ** (1 / 3) == pytest.approx(15.0, rel=0.02)


def test_kern_19_1_lobo_evans_radiant_section():
    ans = KERN_19_1_ANSWER
    # (a) his exchange factor and his alpha: only the heat balance and the enthalpies are this engine's
    a = kern_19_1(geometry={"alpha": 0.937}, exchange_factor_override=0.635)
    assert K2F(a["bridgewall_T_K"]) == pytest.approx(ans["TG_F"], abs=5.0)                       # measured 1696.8 °F
    assert a["radiant_duty_W"] / BTU_H == pytest.approx(ans["Q_Btu_h"], rel=0.008)               # measured +0.6 %
    # (b) his gas emissivity, this engine's network for F (0.622 against his 0.635) and its exact alpha (0.935)
    b = kern_19_1(geometry={"mean_beam_length_m": 15.0 * FT}, gas_emissivity_override=0.496)
    assert b["radiation"]["exchange_factor"] == pytest.approx(0.635, rel=0.025)
    assert K2F(b["bridgewall_T_K"]) == pytest.approx(ans["TG_F"], abs=6.0)                       # measured 1704.1 °F
    assert b["radiant_duty_W"] / BTU_H == pytest.approx(ans["Q_Btu_h"], rel=0.004)               # measured +0.2 %
    # (c) everything of this engine with his beam length, and (d) with the engine's own 3.6 V/A: the WSGG emissivity (0.396
    # and 0.385 against his 0.496) leaves the gas 52-59 °F hotter and the duty 3 % lower
    c_ = kern_19_1(geometry={"mean_beam_length_m": 15.0 * FT})
    d = kern_19_1()
    assert c_["radiation"]["gas_emissivity"] == pytest.approx(0.396, abs=0.002)
    assert d["radiation"]["gas_emissivity"] == pytest.approx(0.385, abs=0.002) and d["radiation"]["mean_beam_length_m"] / FT == \
        pytest.approx(13.48, abs=0.01)
    assert c_["radiant_duty_W"] / BTU_H / ans["Q_Btu_h"] - 1 == pytest.approx(-0.028, abs=0.003)
    assert d["radiant_duty_W"] / BTU_H / ans["Q_Btu_h"] - 1 == pytest.approx(-0.032, abs=0.003)
    assert K2F(d["bridgewall_T_K"]) - ans["TG_F"] == pytest.approx(58.6, abs=2.0)                # +33 K
    assert d["average_flux_W_m2_tube_outside"] / BTU_H * FT2 == pytest.approx(ans["flux_Btu_h_ft2"], rel=0.04)
    for r in (a, b, c_, d):
        assert r["evidence"]["energy_balance_error"] < 1e-9
        json.dumps(r)
    # the trace is the emissivity alone: -20 % of emissivity -> -3 % of duty
    assert (d["radiant_duty_W"] - b["radiant_duty_W"]) / b["radiant_duty_W"] == pytest.approx(-0.033, abs=0.004)
    # Kern's own two equations, closed without charts (19.10 and his heat balance with 476 Btu/lb at 1730 °F), give 1690 °F
    # and 37.0e6: his «about 1700 °F» is this root
    W = 3890.0 + 67900.0 + 1170.0

    def residual(TG):
        q = 1500.0 * 0.635 * (0.173 * (((TG + 460.0) / 100) ** 4 - 12.6 ** 4) + 7.0 * (TG - 800.0))
        return 70.9e6 - W * 476.0 * (TG - 60.0) / 1670.0 - q
    TG = brentq(residual, 1400.0, 1900.0)
    assert TG == pytest.approx(1690.0, abs=2.0) and 70.9e6 - W * 476.0 * (TG - 60.0) / 1670.0 == pytest.approx(37.0e6, rel=0.002)
    # and the empirical equation (19.15) of the same chapter, applied to example 19.1, says 35.5e6: this engine with its own
    # emissivity (35.9e6) falls between the two methods of the book
    wlh = wilson_lobo_hottel(66.67e6, 17.44, 1500.0)
    assert wlh == pytest.approx(35.55e6, rel=0.002) and wlh < d["radiant_duty_W"] / BTU_H < ans["Q_Btu_h"]


def test_kern_19_2_wilson_lobo_hottel_same_furnace_on_gas():
    ans = KERN_19_2_ANSWER
    assert wilson_lobo_hottel(50e6, 22.36, 1500.0) == pytest.approx(ans["Q_Btu_h"], rel=0.003)   # his arithmetic
    c = CB.solve_combustion(CRACKED_GAS, AIR_60F, P, excess_air_fraction=0.40)
    assert c.air_mass == pytest.approx(22.36, rel=0.01)
    r = kern_19_2()
    assert r["radiant_duty_W"] / BTU_H / ans["Q_Btu_h"] - 1 == pytest.approx(-0.017, abs=0.004)  # measured -1.7 %
    assert K2F(r["bridgewall_T_K"]) - ans["TG_F"] == pytest.approx(20.6, abs=3.0)                # +11 K
    assert r["average_flux_W_m2_tube_outside"] / BTU_H * FT2 == pytest.approx(ans["flux_Btu_h_ft2"], rel=0.02)
    # the tube temperature is not in the example: 100 °F less moves the duty +1.7 % and the gas -24 °F
    cooler = kern_19_2(tube_wall_T_K=F2K(700.0))
    assert cooler["radiant_duty_W"] / r["radiant_duty_W"] - 1 == pytest.approx(0.017, abs=0.003)
    assert r["evidence"]["energy_balance_error"] < 1e-9


def test_kern_19_5_radiant_flux_at_given_temperatures():
    # eps_G 0.265, A_R/alpha A_cp 2.49, cold surface 0.9: F = 0.56 from Fig. 19.15; radiation only at T_G 1174 °F, T_S 500 °F:
    # 10 610 * 0.56 = 5 940 Btu/h ft2 of alpha A_cp
    F = FH.exchange_factor(0.265, 1.0, 3.49, 0.9)
    assert F == pytest.approx(0.56, rel=0.07) and F == pytest.approx(0.525, abs=0.002)
    flux = FH.SIGMA * F * (F2K(1174.0) ** 4 - F2K(500.0) ** 4) / BTU_H * FT2
    assert flux / 5940.0 - 1 == pytest.approx(-0.049, abs=0.01)          # -6.3 % in F, +1.5 % in his chart reading of 19.14


def test_radiant_validation_statement_and_refusals():
    r = kern_19_1()
    text = r["evidence"]["validated_against_literature"]
    assert isinstance(text, str) and text.startswith("Validado") and "Kern" in text and "±5 %" in text
    assert r["evidence"]["verified"] is True and r["evidence"]["reasons"] == []
    with pytest.raises(ValueError, match="volume_m3"):
        kern_19_1(geometry={"volume_m3": None})
    with pytest.raises(ValueError, match="uno solo"):
        kern_19_1(tube_wall_T_K=None)


# ----------------------------------------------------------------------------------------------------------------------------
# 3. Convection: the bank coefficient
# ----------------------------------------------------------------------------------------------------------------------------
AIR_AS_FLUE = types.SimpleNamespace(flue={"O2": 0.21, "N2": 0.79})      # what _gas_properties needs to give air
# [INC] ex. 7.7: staggered, D 16.4 mm, S_T 31.3, S_L 34.3 mm, 7 rows of 8 tubes, air 6 m/s at 15 °C, tubes at 70 °C.
# Table A.4 at 288 K: rho 1.217, cp 1007, nu 14.82e-6, k 0.0253, Pr 0.710; Pr_s 0.701; C2 = 0.95 for 7 rows.
# FOSSEE transcription (read 2026-11-14, file saved): Vmax = St V / (St - D), C = 0.35 (St/Sl)^0.2, m 0.6, C2 = 0.95, 56 tubes,
# printed results h = 137.0 W/m2K and q' = 19.6 kW/m. The book's own row (V_max 12.6 m/s, Re 13 943, Nu 87.9, h 135.6, q'
# 19.4) was used by M18b from a source that could NOT be reopened on 2026-11-14 (search found no open copy): it is kept only
# as a comment, the assertions are against the transcription. Its Nu 87.9 is 1 % under what its own formula gives (88.8).
INCROPERA_7_7 = {"D": 0.0164, "St": 0.0313, "Sl": 0.0343, "rows": 7, "per_row": 8, "V": 6.0, "T_in": 288.15, "T_s": 343.15,
                 "rho": 1.217, "cp": 1007.0, "nu": 14.82e-6, "k": 0.0253, "Pr": 0.710, "Pr_s": 0.701,
                 "h": 137.0, "q_kW_m": 19.6, "C2": 0.95, "Vmax_unconfirmed": 12.6, "Re_unconfirmed": 13943.0}
# [CEN] ex. 7-7: in-line, D 15 mm, S_L = S_T = 50 mm, 6 rows of 10 tubes, air 4.5 m/s at 20 °C and 1 atm, tubes 120 °C.
# Properties at 60 °C: k 0.02808, rho 1.059, cp 1007, Pr 0.7202, mu 2.008e-5; Pr_s 0.7073; rho_in 1.204; F = 0.945.
# Transcription: Re 5086, Nu 52.1 (inner rows), exit 29.11 °C, 24 863 W per metre of tube length.
CENGEL_7_7 = {"D": 0.015, "St": 0.05, "Sl": 0.05, "rows": 6, "per_row": 10, "V": 4.5, "rho": 1.059, "cp": 1007.0,
              "mu": 2.008e-5, "k": 0.02808, "Pr": 0.7202, "Pr_s": 0.7073, "rho_in": 1.204,
              "Re": 5086.0, "Nu_inner": 52.12, "F": 0.945, "T_out_C": 29.11, "Q_W_m": 24863.0}


def incropera_7_7(engine_properties: bool) -> dict:
    x = INCROPERA_7_7
    D, St, Sl, n, nt = x["D"], x["St"], x["Sl"], x["rows"], x["per_row"]
    vmax = x["V"] * nt * St / FH.bank_free_width(nt * St, nt, D, St, Sl, "staggered")
    if engine_properties:                                   # Cantera, as the tool computes them
        pr, pw = FH._gas_properties(AIR_AS_FLUE, x["T_in"], P), FH._gas_properties(AIR_AS_FLUE, x["T_s"], P)
        rho = P * 0.02885 / (8.314462 * x["T_in"])
        nu, k, Pr, Prs, cp = pr["mu"] / rho, pr["k"], pr["Pr"], pw["Pr"], pr["cp"]
    else:
        rho, nu, k, Pr, Prs, cp = x["rho"], x["nu"], x["k"], x["Pr"], x["Pr_s"], x["cp"]
    Re = vmax * D / nu
    z = FH.zukauskas_bank_nusselt(Re, Pr, Prs, n, St, Sl, "staggered")
    h = z["Nu"] * k / D
    dT_out = (x["T_s"] - x["T_in"]) * math.exp(-math.pi * D * n * nt * h / (rho * x["V"] * nt * St * cp))
    lmtd = ((x["T_s"] - x["T_in"]) - dT_out) / math.log((x["T_s"] - x["T_in"]) / dT_out)
    return {"Vmax": vmax, "Re": Re, "Nu": z["Nu"], "h": h, "q_kW_m": n * nt * h * math.pi * D * lmtd / 1e3, "zk": z, "nu": nu, "k": k}


def cengel_7_7() -> dict:
    x = CENGEL_7_7
    D, St, Sl, n, nt = x["D"], x["St"], x["Sl"], x["rows"], x["per_row"]
    vmax = x["V"] * nt * St / FH.bank_free_width(nt * St, nt, D, St, Sl, "inline")
    Re = x["rho"] * vmax * D / x["mu"]
    z = FH.zukauskas_bank_nusselt(Re, x["Pr"], x["Pr_s"], n, St, Sl, "inline")
    h = z["Nu"] * x["k"] / D
    area, m = n * nt * math.pi * D, x["rho_in"] * x["V"] * nt * St
    T_out = 120.0 - 100.0 * math.exp(-area * h / (m * x["cp"]))
    lmtd = ((120.0 - T_out) - 100.0) / math.log((120.0 - T_out) / 100.0)
    return {"Re": Re, "Nu_inner": z["Nu_inner_rows"], "F": z["row_factor"], "h": h, "T_out_C": T_out, "Q_W_m": h * area * lmtd, "zk": z}


def test_bank_coefficient_incropera_7_7_staggered():
    x = INCROPERA_7_7
    book = incropera_7_7(engine_properties=False)
    # the transcription's own arithmetic: V_max on the transverse gap (the diagonal gap is wider here)
    assert book["Vmax"] == pytest.approx(x["St"] * x["V"] / (x["St"] - x["D"]), rel=1e-9)
    assert book["Vmax"] == pytest.approx(x["Vmax_unconfirmed"], rel=0.002)
    assert book["Re"] == pytest.approx(x["Re_unconfirmed"], rel=0.001)
    assert book["zk"]["equation"] == "ec. (40)" and book["zk"]["row_factor"] == pytest.approx(x["C2"], abs=0.008)
    assert book["h"] / x["h"] - 1 == pytest.approx(0.0072, abs=0.002)    # +0.7 %: row factor 0.957 (ht's table) vs 0.95
    assert book["q_kW_m"] / x["q_kW_m"] - 1 == pytest.approx(0.0047, abs=0.002)    # +0.5 %
    eng = incropera_7_7(engine_properties=True)                                   # Cantera air instead of the book's table
    assert eng["nu"] == pytest.approx(x["nu"], rel=0.005) and eng["k"] == pytest.approx(x["k"], rel=0.02)
    assert eng["h"] / x["h"] - 1 == pytest.approx(0.0185, abs=0.003)              # +1.9 % (conductivity +1.6 %)
    assert eng["q_kW_m"] / x["q_kW_m"] - 1 == pytest.approx(0.0152, abs=0.003)


def test_bank_coefficient_cengel_7_7_in_line():
    x, r = CENGEL_7_7, cengel_7_7()
    assert r["zk"]["equation"] == "ec. (39)"
    assert r["Re"] == pytest.approx(x["Re"], rel=0.001) and r["Nu_inner"] == pytest.approx(x["Nu_inner"], rel=0.001)
    assert r["F"] == pytest.approx(x["F"], abs=0.002)
    assert r["T_out_C"] == pytest.approx(x["T_out_C"], abs=0.03) and r["Q_W_m"] / x["Q_W_m"] - 1 == pytest.approx(0.0015, abs=0.001)


def test_zukauskas_equations_as_printed_and_why_ht_is_not_called():
    f = lambda Re, a, b, arr, rows=20, Pr=0.7: FH.zukauskas_bank_nusselt(Re, Pr, Pr, rows, a, b, arr)       # noqa: E731
    pr = 0.7 ** 0.36
    assert f(1e4, 2.0, 2.0, "inline")["Nu"] == pytest.approx(0.27 * 1e4 ** 0.63 * pr)                       # [ZUK] eq. (39)
    assert f(1e4, 2.0, 1.5, "staggered")["Nu"] == pytest.approx(0.35 * (2.0 / 1.5) ** 0.2 * 1e4 ** 0.60 * pr)   # eq. (40)
    assert f(1e4, 3.0, 1.2, "staggered")["Nu"] == pytest.approx(0.40 * 1e4 ** 0.60 * pr)                    # eq. (41), a/b > 2
    assert f(5e5, 2.0, 2.0, "inline")["Nu"] == pytest.approx(0.021 * 5e5 ** 0.84 * pr)                      # eq. (42)
    assert f(5e5, 2.0, 1.5, "staggered")["Nu"] == pytest.approx(0.022 * 5e5 ** 0.84 * pr)                   # eq. (43)
    assert 0.022 * pr == pytest.approx(0.019, abs=0.0005)                          # eq. (44), air: 0.019 Re^0.84
    assert f(1e4, 2.0, 1.5, "staggered", rows=3)["row_factor"] < f(1e4, 2.0, 1.5, "staggered", rows=10)["row_factor"] < 1.0
    assert f(1e4, 2.0, 1.5, "staggered", Pr=0.7)["in_source_range"] and not f(500.0, 2.0, 1.5, "staggered")["in_source_range"]
    assert not f(3e6, 2.0, 2.0, "inline")["in_source_range"] and not f(1e4, 1.2, 2.0, "inline")["in_source_range"]
    with pytest.raises(ValueError, match="staggered"):
        f(1e4, 2.0, 1.5, "triangular")
    # ht 1.2.0: (1) arrangement guessed from the pitch ratio, (2) exponent 0.05 for in-line banks at 100 < Re < 1000
    from ht.conv_tube_bank import Nu_Zukauskas_Bejan
    as_ht = Nu_Zukauskas_Bejan(Re=2e4, Pr=0.7, tube_rows=20, pitch_parallel=1.5, pitch_normal=2.0)
    assert as_ht == pytest.approx(f(2e4, 2.0, 1.5, "staggered")["Nu"]) and as_ht != pytest.approx(f(2e4, 2.0, 1.5, "inline")["Nu"])
    assert Nu_Zukauskas_Bejan(Re=500.0, Pr=0.7, tube_rows=20, pitch_parallel=1.5, pitch_normal=1.5) < 1.0 < \
        f(500.0, 1.5, 1.5, "inline")["Nu"] / 5.0


def correlation_band() -> list:
    """This module's Žukauskas against two correlations built on other data, as coded in `ht`: Grimison (1937) and ESDU 73031.

    Only layouts where `ht` guesses the arrangement right (staggered with different pitches, in-line with equal pitches)."""
    from ht.conv_tube_bank import Nu_ESDU_73031, Nu_Grimison_tube_bank
    rows = []
    for arr, a, b in (("staggered", 1.78, 1.54), ("staggered", 2.0, 1.732), ("staggered", 1.5, 1.3), ("inline", 2.0, 2.0),
                      ("inline", 1.5, 1.5)):
        for Re in (3e3, 1e4, 3e4):
            ours = FH.zukauskas_bank_nusselt(Re, 0.72, 0.72, 20, a, b, arr)["Nu"]
            grim = Nu_Grimison_tube_bank(Re=Re, Pr=0.72, Do=1.0, tube_rows=20, pitch_parallel=b, pitch_normal=a)
            esdu = Nu_ESDU_73031(Re=Re, Pr=0.72, tube_rows=20, pitch_parallel=b, pitch_normal=a)
            rows.append({"arrangement": arr, "St_D": a, "Sl_D": b, "Re": Re, "zukauskas": ours, "grimison": grim, "esdu_73031": esdu,
                         "grimison_dev": grim / ours - 1, "esdu_dev": esdu / ours - 1})
    return rows


def test_other_tube_bank_correlations_fall_within_ten_per_cent():
    band = correlation_band()
    devs = [r[k] for r in band for k in ("grimison_dev", "esdu_dev")]
    assert len(devs) == 30 and -0.08 < min(devs) and max(devs) < 0.11             # measured -7 % ... +10 %


# ----------------------------------------------------------------------------------------------------------------------------
# 4. Convection: gas radiation between the tubes, and a whole convection section
# ----------------------------------------------------------------------------------------------------------------------------
def kern_19_4() -> dict:
    """[KERN] ex. 19.4: convection bank, 5 in OD on 8.5 in triangular pitch, gas 1500 °F, tube 650 °F, eps_S 0.90 (assumed by
    Kern), L = 0.4 * 8.5 - 0.567 * 5 = 0.565 ft (Table 19.1). His result: 2.51 Btu/h ft2 °F by eq. (19.20)."""
    Tg, Ts, pw, pc = F2K(1500.0), F2K(650.0), KERN_GAS["p_H2O"], KERN_GAS["p_CO2"]
    D, pitch = 5.0 / 12 * FT, 8.5 / 12 * FT
    L_kern, L_engine = 0.565 * FT, FH.bank_beam_length(pitch, pitch * math.sqrt(3) / 2, D)
    out = {"L_kern_ft": 0.565, "L_engine_ft": L_engine / FT, "kern_shortcut": 0.9 * (2750.0 - 325.0) * 0.98 / 850.0}
    black = 0.173 * (((1500 + 459.67) / 100) ** 4 - ((650 + 459.67) / 100) ** 4)
    eps_kern = (2750.0 - 325.0) * 0.98 / black                         # his eq. (19.5) with his chart readings
    out["kern_eq_19_18"] = black / (1 / eps_kern + 1 / 0.9 - 1) / 850.0  # his eq. (19.18), which (19.19)-(19.20) shorten
    for tag, L in (("kern_L", L_kern), ("engine_L", L_engine)):
        e = FH.gas_emissivity(Tg, pw * ATM, pc * ATM, L)["emissivity"]
        out["eps_" + tag] = e
        out["h_" + tag] = FH.gas_radiation_coefficient(Tg, Ts, e, 0.9) / HBTU
    return out


def test_gas_radiation_between_tubes_kern_19_4():
    r = kern_19_4()
    assert r["kern_shortcut"] == pytest.approx(2.51, abs=0.01) and r["kern_eq_19_18"] == pytest.approx(2.76, abs=0.01)
    # same beam length as Kern: +10 % over his own eq. (19.18) (the WSGG emissivity is 9 % above his charts at short path and
    # the gas is taken grey), +21 % over the shortcut he prints, which alone takes 9 % away
    assert r["h_kern_L"] / r["kern_eq_19_18"] - 1 == pytest.approx(0.10, abs=0.01)
    assert r["h_kern_L"] / r["kern_shortcut"] - 1 == pytest.approx(0.21, abs=0.01)
    # beam length: 3.6 V/A against Kern's Table 19.1, L(ft) = 0.4 P(in) - 0.567 D(in). At pitch 2 D and 3 D (the two layouts
    # of Hottel that the linear rule joins) this module is 10 % and 6 % longer; at 1.7 D the linear rule falls to 0.62 x 4V/A
    # and this module is 45 % longer, so the coefficient is +32 % over eq. (19.18)
    for ratio, expected in ((2.0, 0.098), (3.0, 0.057), (1.7, 0.451)):
        ours = FH.bank_beam_length(ratio, ratio * math.sqrt(3) / 2, 1.0)
        kern = (0.4 * ratio - 0.567) * 12.0
        assert ours / kern - 1 == pytest.approx(expected, abs=0.003)
        four_v_a = 4.0 * (ratio * ratio * math.sqrt(3) / 2 - math.pi / 4) / math.pi
        assert ours / four_v_a == pytest.approx(0.9)
    assert (0.4 * 1.7 - 0.567) * 12.0 / (4.0 * (1.7 ** 2 * math.sqrt(3) / 2 - math.pi / 4) / math.pi) == pytest.approx(0.62, abs=0.01)
    assert r["L_engine_ft"] == pytest.approx(0.82, abs=0.005) and r["h_engine_L"] / r["kern_eq_19_18"] - 1 == pytest.approx(0.32, abs=0.01)


# [NELSON] example 68, convection section of a pipestill. Given: 30 000 lb/h of oil enters the convection section at 300 °F
# with specific heat 0.6 (he writes 2 860 000 = 30 000 (t - 300) 0.6: leaves at 460 °F); fuel oil 17 600 Btu/lb net, 14.7 lb of
# flue gas per lb with no excess air, 25 % excess air; 13 330 000 Btu/h fired; gas enters at 1700 °F and leaves at 810 °F (his
# Fig. 56: 43.5 % and 20 % of the heat fired); 2 % of the heat fired is lost through the walls of the convection section, so the
# gas gives up 3 130 000 and the oil takes 2 860 000 Btu/h; «3- by 4-in. tubes»; gas velocity 12 ft/s; transfer rate 6.95
# Btu/h ft2 °F (his Table 64 gives 7.3 at 12 ft/s, 1200 °F, 4 in and 7.7 at 1400 °F; it includes gas radiation and the wall
# effect and neglects the oil film and the metal); answer: 477 ft of tubing.
# NOT given, taken here: C/H of the oil (from 13.7 lb air/lb with C + H = 1), staggered triangular pitch of two diameters, 12
# rows of 4 tubes (gives his 12 ft/s), tube emissivity 0.9, dry air at 60 °F.
NELSON_OIL = {"type": "ultimate", "mass_fractions": {"C": 0.9035, "H": 0.0965}, "LHV_J_kg": 17600.0 * BTU_LB}
NELSON_AIR = {"T_K": F2K(60.0), "humidity_ratio_kg_kg": 0.0}
NELSON_68 = {"fired_Btu_h": 13.33e6, "gas_in_F": 1700.0, "stack_F": 810.0, "absorbed_Btu_h": 2.86e6, "gas_gives_Btu_h": 3.13e6,
             "U": 6.95, "table_64_1200F": 7.3, "table_64_1400F": 7.7, "tubing_ft": 477.0, "velocity_ft_s": 12.0, "oil_out_F": 460.0}


def nelson_68(gas_in_K: float | None = None, **bank_overrides) -> dict:
    x = NELSON_68
    rows, per_row = bank_overrides.pop("n_rows", 12), bank_overrides.pop("tubes_per_row", 4)
    pitch = 8.0 / 12 * FT
    bank = {"tube_outside_diameter_m": 4.0 / 12 * FT, "tube_inside_diameter_m": 3.0 / 12 * FT,
            "tube_effective_length_m": x["tubing_ft"] / (rows * per_row) * FT, "tubes_per_row": per_row, "n_rows": rows,
            "transverse_pitch_m": pitch, "longitudinal_pitch_m": pitch * math.sqrt(3) / 2, "arrangement": "staggered",
            "wall_conductivity_W_mK": 1e6, "outside_fouling_m2K_W": 0.0, "tube_emissivity": 0.9}
    bank.update(bank_overrides)
    process = {"inlet_T_K": F2K(300.0), "capacity_rate_W_K": 30000.0 * 0.6 * BTU_H * 1.8, "inside_coefficient_W_m2K": 1e6,
               "inside_fouling_m2K_W": 0.0}                            # «the oil film and the metal wall are negligible»
    fuel_kg_s = x["fired_Btu_h"] * BTU_H / NELSON_OIL["LHV_J_kg"]
    T_in = F2K(x["gas_in_F"]) if gas_in_K is None else gas_in_K
    r = FH.convection_section(NELSON_OIL, NELSON_AIR, P, fuel_kg_s, T_in, bank, process, excess_air_fraction=0.25)
    hot, cold = T_in - r["process_outlet_T_K"], r["stack_T_K"] - process["inlet_T_K"]
    lmtd = (hot - cold) / math.log(hot / cold)
    T_mean = 0.5 * (T_in + r["stack_T_K"])
    return dict(r, U_lmtd_Btu=r["duty_W"] / (r["outside_area_m2"] * lmtd) / HBTU, duty_Btu_h=r["duty_W"] / BTU_H,
                velocity_ft_s=r["mass_velocity_kg_m2s"] / (P * 0.029 / (8.314462 * T_mean)) / FT)


def test_nelson_68_convection_section_of_a_pipestill():
    x = NELSON_68
    c = CB.solve_combustion(NELSON_OIL, NELSON_AIR, P, excess_air_fraction=0.25)
    assert CB.solve_combustion(NELSON_OIL, NELSON_AIR, P, excess_air_fraction=0.0).flue_mass == pytest.approx(14.7, rel=0.003)
    heat_left = lambda t_f: (c.flue_sensible(F2K(t_f)) - c.flue_sensible(F2K(60.0))) / c.LHV    # noqa: E731
    # his enthalpy chart against the NASA enthalpies: equal at the stack, 2.8 points of the heat fired apart at 1700 °F
    assert heat_left(810.0) == pytest.approx(0.20, abs=0.003) and heat_left(1700.0) == pytest.approx(0.463, abs=0.003)
    r = nelson_68()
    assert r["velocity_ft_s"] == pytest.approx(x["velocity_ft_s"], rel=0.08)
    assert r["outside_area_m2"] / FT2 == pytest.approx(x["tubing_ft"] * math.pi * 4 / 12)
    # overall coefficient on the log-mean difference: +5 % over the 6.95 he uses, -1 % under his Table 64 (7.4 at 1250 °F)
    assert r["U_lmtd_Btu"] / x["U"] - 1 == pytest.approx(0.053, abs=0.01)
    assert r["U_lmtd_Btu"] == pytest.approx(0.75 * x["table_64_1200F"] + 0.25 * x["table_64_1400F"], rel=0.02)
    # duty with his 477 ft: +2.0 % over what his gas gives up, +11.6 % over what his oil takes (he charges 2 % of the heat fired
    # to the casing; this model has no loss in the bank and says so in not_included)
    assert r["duty_Btu_h"] / x["gas_gives_Btu_h"] - 1 == pytest.approx(0.020, abs=0.005)
    assert r["duty_Btu_h"] / x["absorbed_Btu_h"] - 1 == pytest.approx(0.116, abs=0.006)
    assert K2F(r["stack_T_K"]) - x["stack_F"] == pytest.approx(86.0, abs=4.0)                  # +48 K: his chart at 1700 °F
    assert 0.25 < r["gas_radiation_share_of_duty"] < 0.40                                      # Kern: «5 to 30 per cent»
    assert any("pérdida por paredes del banco" in s for s in r["not_included"])
    # entering with HIS heat content (43.5 % of the heat fired: 1609 °F by NASA) the duty is +2.3 % over his 2.86e6
    T435 = brentq(lambda t: heat_left(K2F(t)) - 0.435, 900.0, 1400.0)
    same_heat = nelson_68(gas_in_K=T435)
    assert K2F(T435) == pytest.approx(1609.0, abs=3.0)
    assert same_heat["duty_Btu_h"] / x["absorbed_Btu_h"] - 1 == pytest.approx(0.023, abs=0.006)
    # what was not given matters little: tube emissivity 0.8 instead of 0.9, or the same tubes in line
    for other in ({"tube_emissivity": 0.8}, {"arrangement": "inline", "longitudinal_pitch_m": 8.0 / 12 * FT}):
        assert nelson_68(**other)["duty_W"] == pytest.approx(r["duty_W"], rel=0.01)
    json.dumps(r)


def test_row_by_row_march_against_the_continuous_solution():
    """Numerics of the tool: isothermal tubes, no gas radiation, no wall resistance. The continuous problem
    m dh = -U(T) (T - Tp) dA is integrated with scipy and compared with the row-by-row arithmetic-mean march."""
    bank_args = dict(tube_outside_diameter_m=0.1143, tube_inside_diameter_m=0.1023, tube_effective_length_m=12.0, tubes_per_row=8,
                     n_rows=10, transverse_pitch_m=0.2032, longitudinal_pitch_m=0.176, arrangement="staggered",
                     wall_conductivity_W_mK=1e7, outside_fouling_m2K_W=0.0, tube_emissivity=0.85)
    fuel = {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 300.0}
    air = {"T_K": 300.0, "humidity_ratio_kg_kg": 0.008}
    Tp, T_in, m = 520.0, 1100.0, 0.6
    proc = {"inlet_T_K": Tp, "isothermal": True, "inside_coefficient_W_m2K": 1e7, "inside_fouling_m2K_W": 0.0}
    r = FH.convection_section(fuel, air, P, m, T_in, bank_args, proc, excess_air_fraction=0.15, gas_radiation=False)
    c = CB.solve_combustion(fuel, air, P, excess_air_fraction=0.15)
    Do, G = 0.1143, r["mass_velocity_kg_m2s"]
    prw = FH._gas_properties(c, Tp, P)["Pr"]

    def dT_dA(_, y):
        T = y[0]
        pr = FH._gas_properties(c, T, P)
        h = FH.zukauskas_bank_nusselt(G * Do / pr["mu"], pr["Pr"], prw, 10, 0.2032, 0.176, "staggered")["Nu"] * pr["k"] / Do
        cp_flow = m * (c.flue_sensible(T + 0.5) - c.flue_sensible(T - 0.5))
        return [-h * (T - Tp) / cp_flow]
    sol = solve_ivp(dT_dA, (0.0, r["outside_area_m2"]), [T_in], rtol=1e-9, atol=1e-9)
    assert r["stack_T_K"] == pytest.approx(sol.y[0][-1], abs=0.2)                 # measured difference below 0.1 K
    assert r["evidence"]["energy_balance_error"] < 1e-9 and r["method"]["equations_used"] == ["ec. (40)"]


def test_duty_sensitivity_to_the_coefficient_and_statement(monkeypatch):
    fuel = {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 300.0}
    air = {"T_K": 300.0, "humidity_ratio_kg_kg": 0.008}
    bank = {"tube_outside_diameter_m": 0.1143, "tube_inside_diameter_m": 0.1023, "tube_effective_length_m": 12.0,
            "tubes_per_row": 8, "n_rows": 10, "transverse_pitch_m": 0.2032, "longitudinal_pitch_m": 0.176,
            "arrangement": "staggered", "wall_conductivity_W_mK": 25.0, "outside_fouling_m2K_W": 0.0, "tube_emissivity": 0.85}
    proc = {"inlet_T_K": 520.0, "capacity_rate_W_K": 60e3, "inside_coefficient_W_m2K": 1200.0, "inside_fouling_m2K_W": 0.0005}
    run = lambda b=bank: FH.convection_section(fuel, air, P, 0.3, 1129.0, b, proc, excess_air_fraction=0.15)   # noqa: E731
    base = run()
    text = base["evidence"]["validated_against_literature"]
    assert isinstance(text, str) and "Nelson" in text and "Incropera" in text
    assert base["evidence"]["verified"] is True and base["evidence"]["reasons"] == []
    assert base["method"]["convection_correlation"] == FH.ZUKAUSKAS_SOURCE and not any("Reynolds" in w for w in base["warnings"])
    original = FH.zukauskas_bank_nusselt
    out = {}
    for factor in (0.85, 1.15):
        monkeypatch.setattr(FH, "zukauskas_bank_nusselt", lambda *a, _f=factor, **k: dict(original(*a, **k), Nu=original(*a, **k)["Nu"] * _f))
        out[factor] = run()["duty_W"] / base["duty_W"] - 1
    monkeypatch.setattr(FH, "zukauskas_bank_nusselt", original)
    assert out[0.85] == pytest.approx(-0.032, abs=0.004) and out[1.15] == pytest.approx(0.029, abs=0.004)
    # the beam length of the bank can be given: Kern's rule instead of 3.6 V/A lowers the radiant part
    kern_L = (0.4 * 8.0 - 0.567 * 4.5) * FT
    shorter = run(dict(bank, mean_beam_length_m=kern_L))
    assert shorter["mean_beam_length_m"] == pytest.approx(kern_L) and shorter["duty_W"] < base["duty_W"]
    assert any("bank.mean_beam_length_m" in s for s in shorter["assumptions"])
    # a low Reynolds number is flagged, not hidden
    slow = FH.convection_section(fuel, air, P, 0.05, 900.0, bank, proc, excess_air_fraction=0.15)
    assert any("Reynolds" in w and "Bejan" in w for w in slow["warnings"])
    json.dumps(base), json.dumps(slow)


# ----------------------------------------------------------------------------------------------------------------------------
# 5. Finned tubes (bloque M18c): revised ESCOA as printed by [GAN], fin efficiency, a whole finned section, bare-tube limit
# ----------------------------------------------------------------------------------------------------------------------------
KCAL = 1.163                                   # kcal/(m2 h °C) -> W/(m2 K); also kcal/(m h °C) -> W/(m K)
# [GAN] ex. E.1: solid fins, in-line, 50.8 x 44 mm, 78.7 fins/m, h 12.7 mm, b 1.52 mm, 101.6 mm square pitch, 18 tubes/row,
# 6 rows, 3.1 m, gas 100 000 kg/h 550 -> 479 °C (CO2 3, H2O 7, N2 75, O2 15 % vol), steam 25 000 kg/h 265 -> 369 °C, 4804 kPa,
# supplier's surface 188 m2. His properties at 514 °C: cp 0.2755 kcal/kg °C, mu 0.127 kg/m h, k 0.0468 kcal/m h °C; mean fin
# temperature assumed 400 °C; fin and wall conductivity 30 kcal/m h °C (he converts it as 25.8 W/m K; his arithmetic uses 30).
# His results: Ao 0.0538, Af 0.427, At 0.567 m2/m, s 0.0112 m, G 37 491 kg/m2 h, Re 14 996, C1 0.006059, C3 0.6900, C5 1.0016,
# hc 69.6 kcal (80.93 W/m2K), beam 0.0707 m, hn 3.58 kcal, ho 73.13, m 56.6, E 0.827 (simplified) / 0.856 (Bessel), eta 0.87 /
# 0.89, hi 1720 kcal, At/Ai 4.1, fouling 0.0002 m2 h °C/kcal both sides, Uo 51.9 kcal (60.35 W/m2K), duty 2.25 MW, LMTD 197 °C,
# area required 189 m2.
GAN_E1 = {"d": 0.0508, "di": 0.044, "h": 0.0127, "b": 0.00152, "n": 78.7, "St": 0.1016, "Sl": 0.1016, "per_row": 18, "rows": 6,
          "L": 3.1, "W_kg_h": 100000.0, "cp": 0.2755, "mu": 0.127, "k": 0.0468, "Tg_C": 514.0, "Tf_C": 400.0, "kfin": 30.0,
          "Ao": 0.0538, "Af": 0.427, "At": 0.567, "s": 0.0112, "G": 37491.0, "Re": 14996.0, "C1": 0.006059, "C3": 0.6900,
          "C5": 1.0016, "hc": 69.6, "hc_W": 80.93, "beam": 0.0707, "ho": 73.13, "m": 56.6, "E_simplified": 0.827,
          "E_bessel": 0.856, "hi": 1720.0, "ratio": 4.1, "U": 51.9, "U_W": 60.35, "duty_MW": 2.25, "gas_out_C": 479.0,
          "steam_out_C": 369.0, "steam_kJ_kg": 324.3, "steam_kg_s": 6.94}
# [GAN] ex. E.10: serrated fins, in-line, 50.8 x 46.4 mm, 197 fins/m, h 15 mm, b 1.5 mm, ws 4 mm, ST 107, SL 90 mm, 50 tubes/row,
# 56 deep, 2.4 m, 200 000 kg/h of air; properties at 103 °C: cp 0.2455, mu 0.0806, k 0.0282; fin temperature assumed 135 °C.
# His results: Ao 0.05967, Af 1.343, At 1.456, G 35 213 kg/m2 h, Re 22 193, C1 0.00435, C3 0.4516, C5 0.9605, hc 24.93 kcal,
# m 39.0, mh 0.585, E 0.896, eta 0.904.
GAN_E10 = {"d": 0.0508, "h": 0.015, "b": 0.0015, "n": 197.0, "ws": 0.004, "St": 0.107, "Sl": 0.090, "per_row": 50, "rows": 56,
           "L": 2.4, "W_kg_h": 200000.0, "cp": 0.2455, "mu": 0.0806, "k": 0.0282, "Tg_C": 103.0, "Tf_C": 135.0, "kfin": 30.0,
           "Ao": 0.05967, "Af": 1.343, "At": 1.456, "G": 35213.0, "Re": 22193.0, "C1": 0.00435, "C3": 0.4516, "C5": 0.9605,
           "hc": 24.93, "m": 39.0, "E": 0.896, "eta": 0.904}
# [EST] fin-tube sample: staggered (equilateral) 4.5 in tubes on 8 in, 8 wide, 13 ft, 6 rows, serrated fins 0.75 in high,
# 0.05 in thick, 6 fins/in, segment 0.3125 in; 100 000 lb/h, gas 1000 -> 591 °F, mean fin 755 °F; kb 0.0290, cp 0.2858,
# mu 0.0767 lb/h ft. Its constants (an earlier ESCOA version): C1 = 0.25 Re^-0.35, C3 = 0.55 + 0.45 exp(-0.35 lf/sf), C5 as
# Table E.2. Printed: Ac 0.4125 ft2/ft, An 26.4333 ft2, Gn 3783.1069, Re 18 496.2854, C1 0.0080, C3 0.5975, C5 0.9929,
# j 0.0055 (rounded), hc 7.1732 Btu/h ft2 °F.
ESTEEM = {"d": 4.5, "h": 0.75, "t": 0.05, "n": 6.0, "ws": 0.3125, "pitch": 8.0, "wide": 8, "L_ft": 13.0, "rows": 6, "W": 100000.0,
          "kb": 0.0290, "cp": 0.2858, "mu": 0.0767, "Tb_F": 795.5, "Ts_F": 755.0,
          "Ac": 0.4125, "An": 26.4333, "Gn": 3783.1069, "Re": 18496.2854, "C1": 0.0080, "C3": 0.5975, "C5": 0.9929, "j": 0.0055,
          "hc": 7.1732}
INCH = 0.0254


def gan_coefficient(x: dict, fin_type: str) -> dict:
    """His step-by-step hand calculation, with this module's functions and HIS gas properties and fin temperature."""
    g = FH.finned_tube_geometry(x["d"], fin_type, x["h"], x["b"], x["n"], x.get("ws"))
    G = x["W_kg_h"] / (x["per_row"] * x["L"] * (x["St"] - g["obstruction_m"]))
    Re = G * x["d"] / x["mu"]
    ec = FH.escoa_colburn(Re, x["rows"], x["St"], x["Sl"], x["d"], x["h"], g["fin_gap_m"], fin_type, "inline")
    hc = ec["j_geometric"] * ((x["Tg_C"] + 273.0) / (x["Tf_C"] + 273.0)) ** 0.5 * G * x["cp"] * (x["k"] / x["mu"] / x["cp"]) ** 0.67
    return {"geo": g, "G": G, "Re": Re, "ec": ec, "hc": hc}


def test_ganapathy_e1_solid_fins_coefficient_and_fin_efficiency():
    x = GAN_E1
    r = gan_coefficient(x, "solid_fin")
    g = r["geo"]
    assert g["obstruction_m"] == pytest.approx(x["Ao"], rel=0.001) and g["fin_area_m2_per_m"] == pytest.approx(x["Af"], rel=0.002)
    assert g["total_area_m2_per_m"] == pytest.approx(x["At"], rel=0.002) and g["fin_gap_m"] == pytest.approx(x["s"], rel=0.002)
    assert r["G"] == pytest.approx(x["G"], rel=0.001) and r["Re"] == pytest.approx(x["Re"], rel=0.001)
    for key in ("C1", "C3", "C5"):
        assert r["ec"][key] == pytest.approx(x[key], rel=0.0015)
    assert r["hc"] == pytest.approx(x["hc"], rel=0.001) and r["hc"] * KCAL == pytest.approx(x["hc_W"], rel=0.001)   # -0.01 %
    assert FH.finned_bank_beam_length(x["St"], x["Sl"], x["d"], x["h"]) == pytest.approx(x["beam"], rel=0.001)
    fe = FH.fin_efficiency("solid_fin", x["ho"] * KCAL, x["kfin"] * KCAL, x["d"], x["h"], x["b"])
    assert fe["m_1_m"] == pytest.approx(x["m"], rel=0.002)
    # exact Bessel: 0.830. His Bessel 0.856 uses table values at m ro = 1.437 that are not those of the functions (I1 0.849 for
    # 0.921, I0 1.516 for 1.587); his own simplified formula gives 0.827, 0.4 % away from the exact value
    assert fe["E"] == pytest.approx(0.8304, abs=0.0005)
    assert fe["E"] / x["E_bessel"] - 1 == pytest.approx(-0.030, abs=0.002) and fe["E"] == pytest.approx(x["E_simplified"], rel=0.005)
    from scipy.special import i0, i1
    assert i1(1.437) == pytest.approx(0.921, abs=0.001) and i0(1.437) == pytest.approx(1.587, abs=0.001)
    eta = 1 - (1 - fe["E"]) * g["fin_area_m2_per_m"] / g["total_area_m2_per_m"]
    assert eta == pytest.approx(0.872, abs=0.001)
    # his overall coefficient with this effectiveness (his other terms unchanged): -1.6 %
    inv = 1 / (eta * x["ho"]) + 0.0002 + 0.0005 + 0.0002 * x["ratio"] + x["ratio"] / x["hi"]
    assert (1 / inv) / x["U"] - 1 == pytest.approx(-0.016, abs=0.003)
    json.dumps(r)


def test_ganapathy_e10_serrated_fins_coefficient_and_fin_efficiency():
    x = GAN_E10
    r = gan_coefficient(x, "serrated_fin")
    g = r["geo"]
    assert g["obstruction_m"] == pytest.approx(x["Ao"], rel=0.001) and g["fin_area_m2_per_m"] == pytest.approx(x["Af"], rel=0.001)
    assert g["total_area_m2_per_m"] == pytest.approx(x["At"], rel=0.001)
    assert r["G"] == pytest.approx(x["G"], rel=0.001) and r["Re"] == pytest.approx(x["Re"], rel=0.001)
    for key in ("C1", "C3", "C5"):
        assert r["ec"][key] == pytest.approx(x[key], rel=0.001)
    assert r["hc"] / x["hc"] - 1 == pytest.approx(0.005, abs=0.002)                     # +0.5 %
    fe = FH.fin_efficiency("serrated_fin", x["hc"] * KCAL, x["kfin"] * KCAL, x["d"], x["h"], x["b"], x["ws"])
    assert fe["m_1_m"] == pytest.approx(x["m"], rel=0.002) and fe["E"] / x["E"] - 1 == pytest.approx(0.004, abs=0.002)
    eta = 1 - (1 - fe["E"]) * g["fin_area_m2_per_m"] / g["total_area_m2_per_m"]
    assert eta / x["eta"] - 1 == pytest.approx(0.004, abs=0.002)


def test_esteem_staggered_serrated_example_and_the_two_escoa_versions():
    x = ESTEEM
    g = FH.finned_tube_geometry(x["d"] * INCH, "serrated_fin", x["h"] * INCH, x["t"] * INCH, x["n"] / INCH, x["ws"] * INCH)
    Ac = g["obstruction_m"] / FT
    An = x["wide"] * x["L_ft"] * x["pitch"] / 12 - Ac * x["L_ft"] * x["wide"]
    Gn = x["W"] / An
    Re = Gn * x["d"] / (12 * x["mu"])
    assert Ac == pytest.approx(x["Ac"], rel=1e-4) and An == pytest.approx(x["An"], rel=1e-5) and Gn == pytest.approx(x["Gn"], rel=1e-5)
    assert Re == pytest.approx(x["Re"], rel=1e-5)
    Pl = math.sqrt(x["pitch"] ** 2 - (x["pitch"] / 2) ** 2)
    ec = FH.escoa_colburn(Re, x["rows"], x["pitch"] * INCH, Pl * INCH, x["d"] * INCH, x["h"] * INCH, g["fin_gap_m"],
                          "serrated_fin", "staggered")
    assert ec["C5"] == pytest.approx(x["C5"], abs=5e-5)                                 # C5: same in both versions
    sf = 1 / x["n"] - x["t"]
    C1_old, C3_old = 0.25 * Re ** -0.35, 0.55 + 0.45 * math.exp(-0.35 * x["h"] / sf)
    assert C1_old == pytest.approx(x["C1"], abs=5e-5) and C3_old == pytest.approx(x["C3"], abs=1e-4)   # printed to 4 d.
    temp = ((x["Tb_F"] + 460) / (x["Ts_F"] + 460)) ** 0.25
    props = x["cp"] * (x["kb"] / (x["cp"] * x["mu"])) ** 0.67
    j_old = C1_old * C3_old * x["C5"] * (6 / 4.5) ** 0.5 * temp
    assert j_old == pytest.approx(x["j"], abs=5e-5) and x["j"] * Gn * props == pytest.approx(x["hc"], rel=1e-4)
    # the revised constants of the engine (C1 0.0078 instead of 0.0080, C3 0.568 instead of 0.597) on the same bank: -6.9 %
    hc_rev = ec["j_geometric"] * temp * Gn * props
    assert hc_rev / x["hc"] - 1 == pytest.approx(-0.069, abs=0.003)


def gan_e1_section(**bank_overrides) -> dict:
    """[GAN] ex. E.1 run as a section by the tool. The gas analysis is reproduced with methane, excess air and air humidity
    (CO2 3.0, H2O 7.0, O2 14.2, N2 74.9, Ar 0.96 %); steam as a constant capacity rate from his 324.3 kJ/kg over 265-369 °C."""
    x = GAN_E1
    fuel = {"type": "gas", "composition": {"CH4": 1.0}, "T_K": 298.15}
    air = {"T_K": 298.15, "humidity_ratio_kg_kg": 0.0065083}
    c = CB.solve_combustion(fuel, air, P, excess_air_fraction=2.35984)
    bank = {"tube_outside_diameter_m": x["d"], "tube_inside_diameter_m": x["di"], "tube_effective_length_m": x["L"],
            "tubes_per_row": x["per_row"], "n_rows": x["rows"], "transverse_pitch_m": x["St"], "longitudinal_pitch_m": x["Sl"],
            "arrangement": "inline", "wall_conductivity_W_mK": x["kfin"] * KCAL, "outside_fouling_m2K_W": 0.0002 / KCAL,
            "tube_emissivity": 0.9, "tube_type": "solid_fin", "fin_height_m": x["h"], "fin_thickness_m": x["b"],
            "fins_per_m": x["n"], "fin_conductivity_W_mK": x["kfin"] * KCAL}
    bank.update(bank_overrides)
    proc = {"inlet_T_K": 265 + 273.15, "capacity_rate_W_K": x["steam_kg_s"] * x["steam_kJ_kg"] * 1e3 / (369 - 265),
            "inside_coefficient_W_m2K": x["hi"] * KCAL, "inside_fouling_m2K_W": 0.0002 / KCAL}
    r = FH.convection_section(fuel, air, P, x["W_kg_h"] / 3600 / c.flue_mass, 550 + 273.15, bank, proc, excess_air_fraction=2.35984)
    hot, cold = 550 + 273.15 - r["process_outlet_T_K"], r["stack_T_K"] - (265 + 273.15)
    return dict(r, U_lmtd=r["duty_W"] / r["outside_area_m2"] / ((hot - cold) / math.log(hot / cold)), flue=c.flue, flue_mol=c.flue_mol)


def test_ganapathy_e1_as_a_whole_finned_section():
    x, r = GAN_E1, gan_e1_section()
    assert r["flue"]["CO2"] / r["flue_mol"] == pytest.approx(0.03, abs=1e-4) and r["flue"]["H2O"] / r["flue_mol"] == pytest.approx(0.07, abs=1e-4)
    assert r["outside_area_m2"] == pytest.approx(188.0, rel=0.012)                          # 190.1 m2 from his At
    assert r["duty_W"] / 1e6 / x["duty_MW"] - 1 == pytest.approx(0.029, abs=0.005)          # +2.9 %
    assert r["stack_T_K"] - 273.15 - x["gas_out_C"] == pytest.approx(-1.6, abs=1.0)
    assert r["process_outlet_T_K"] - 273.15 - x["steam_out_C"] == pytest.approx(2.9, abs=1.0)
    assert r["U_lmtd"] / x["U_W"] - 1 == pytest.approx(0.036, abs=0.006)                    # +3.6 %
    # where the difference comes from: Cantera's conductivity of this gas is 7 % above his Appendix F (0.0502 vs 0.0468)
    pr = FH._gas_properties(types.SimpleNamespace(flue=r["flue"]), x["Tg_C"] + 273.15, P)
    assert pr["k"] / KCAL / x["k"] - 1 == pytest.approx(0.072, abs=0.01)
    assert all(row["h_convection_W_m2K"] / x["hc_W"] - 1 == pytest.approx(0.056, abs=0.01) for row in r["rows"])
    assert all(0.81 < row["fin_efficiency"] < 0.83 for row in r["rows"]) and r["max_fin_tip_T_K"] < 550 + 273.15
    assert r["evidence"]["verified"] and r["evidence"]["energy_balance_error"] < 1e-9 and r["method"]["tube_type"] == "solid_fin"
    assert r["evidence"]["validated_against_literature"] == FH.FINNED_VALIDATION and "ESCOA" in r["method"]["convection_correlation"]
    assert any("con pernos" in s for s in r["not_included"])
    # the tool's own fin-base / fin-tip temperatures obey the source's relation (E.22): tip between base and gas
    assert all(row["tube_wall_K"] < row["fin_tip_T_K"] < row["gas_in_K"] for row in r["rows"])
    json.dumps({k: v for k, v in r.items() if k not in ("flue",)})


def test_finned_bank_bare_tube_limit_and_identities():
    """Fin height -> 0 with one thin fin per metre: the area becomes the bare tube's and the coefficient is ESCOA's bare
    limit. Against this module's bare-tube result (Žukauskas) the difference is that of the two correlations, measured:
    staggered -9 % ... +18 %, in-line -25 % ... -58 % (Re 5e3-2e4, Pr 0.72, isothermal, banks of [GAN], [EST] and Nelson)."""
    Pl = math.sqrt(8.0 ** 2 - 4.0 ** 2)
    devs = {"staggered": [], "inline": []}
    for D, St, Sl, arr, rows in ((0.0508, 0.1016, 0.1016, "inline", 6), (0.0508, 0.107, 0.090, "inline", 56),
                                 (4.5 * INCH, 8 * INCH, Pl * INCH, "staggered", 6), (0.1016, 0.2032, 0.176, "staggered", 12),
                                 (0.1, 0.2, 0.1732, "staggered", 20)):
        for Re in (5e3, 2e4):
            zk = FH.zukauskas_bank_nusselt(Re, 0.72, 0.72, rows, St, Sl, arr)["Nu"]
            for ft in FH.FIN_TYPES:
                e = FH.escoa_colburn(Re, rows, St, Sl, D, 1e-9, 0.003, ft, arr)
                devs[arr].append(e["j_geometric"] * Re * 0.72 ** 0.33 / zk - 1)
    assert -0.095 < min(devs["staggered"]) and max(devs["staggered"]) < 0.19
    assert -0.58 < min(devs["inline"]) and max(devs["inline"]) < -0.24
    # through the tool: Nelson-type staggered bank, bare against fins of 10 micrometres
    fuel = {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 300.0}
    air = {"T_K": 300.0, "humidity_ratio_kg_kg": 0.008}
    bank = {"tube_outside_diameter_m": 0.1143, "tube_inside_diameter_m": 0.1023, "tube_effective_length_m": 12.0,
            "tubes_per_row": 8, "n_rows": 10, "transverse_pitch_m": 0.2032, "longitudinal_pitch_m": 0.176,
            "arrangement": "staggered", "wall_conductivity_W_mK": 25.0, "outside_fouling_m2K_W": 0.0, "tube_emissivity": 0.85}
    proc = {"inlet_T_K": 520.0, "capacity_rate_W_K": 60e3, "inside_coefficient_W_m2K": 1200.0, "inside_fouling_m2K_W": 0.0005}
    run = lambda b: FH.convection_section(fuel, air, P, 0.3, 1129.0, b, proc, excess_air_fraction=0.15)    # noqa: E731
    bare = run(bank)
    tiny = run(dict(bank, tube_type="solid_fin", fin_height_m=1e-5, fin_thickness_m=1e-4, fins_per_m=1.0, fin_conductivity_W_mK=40.0))
    assert tiny["outside_area_m2"] == pytest.approx(bare["outside_area_m2"], rel=1e-3)
    hc_ratio = tiny["rows"][0]["h_convection_W_m2K"] / bare["rows"][0]["h_convection_W_m2K"]
    assert hc_ratio - 1 == pytest.approx(-0.067, abs=0.01)             # this bank (Re ~4e3): ESCOA -6.7 % under Žukauskas
    # duty -3.0 %: the coefficient, and Ganapathy's finned beam length (0.216 m) instead of 3.6 V/A (0.256 m)
    assert tiny["duty_W"] / bare["duty_W"] - 1 == pytest.approx(-0.030, abs=0.005)
    assert tiny["mean_beam_length_m"] == pytest.approx(0.216, abs=0.001)
    assert tiny["rows"][0]["fin_efficiency"] == pytest.approx(1.0, abs=1e-3)
    # fins raise the duty, more fins more duty; the effectiveness falls as fins grow
    fins = lambda n, h=0.019: run(dict(bank, tube_type="serrated_fin", fin_height_m=h, fin_thickness_m=0.0013, fins_per_m=n,  # noqa: E731
                                       fin_segment_width_m=0.004, fin_conductivity_W_mK=40.0))
    a, b2 = fins(118.0), fins(197.0)
    assert bare["duty_W"] < a["duty_W"] < b2["duty_W"]
    assert fins(118.0, h=0.025)["rows"][0]["fin_effectiveness"] < a["rows"][0]["fin_effectiveness"]
    assert any("tresbolillo" in w for w in a["warnings"]) and a["evidence"]["verified"]


def test_finned_bank_refusals():
    base = gan_e1_section
    with pytest.raises(ValueError, match="fin_segment_width_m"):
        base(tube_type="serrated_fin")
    with pytest.raises(ValueError, match="fin_conductivity_W_mK"):
        base(fin_conductivity_W_mK=None)
    with pytest.raises(ValueError, match="pernos"):
        base(tube_type="studded")
    with pytest.raises(ValueError, match="no cabe"):
        base(fin_height_m=0.03)
    with pytest.raises(ValueError, match="espesor"):
        base(fins_per_m=700.0)
