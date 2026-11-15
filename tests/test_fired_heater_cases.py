"""Fired heaters, bloque L3: two more published radiant sections and studded convection tubes.

Every published number below was READ on 2026-11-15 in a source that opened without login (saved under the scratchpad of the
block, nothing downloaded is used at run time):
  [EST-R]  Esteem Projects, «Free Fired Heater Design Tutorial - Radiant Section Design», section «Radiant Section Rating»:
           https://www.esteemprojects.com/library/Fired-Heater-Design/page4_1.htm
  [EST-S]  same tutorial, «Shield Section Design» (states the converged state of the same radiant example):
           https://www.esteemprojects.com/library/Fired-Heater-Design/page4_2.htm
  [EST-C]  same tutorial, «Fuel Gas Combustion» calculator (the default fuel and air of the example) and «Flue Gas Properties»
           calculator (its enthalpy polynomials, transcribed below from the page's script):
           https://www.esteemprojects.com/library/Fired-Heater-Design/combust2.htm and .../gasprops.htm
  [EST-T]  same tutorial, «Convection Section Design», section «Convection Transfer, Stud Tubes» (formulas and the default
           inputs of its stud script; the script prints no results and the page has no worked stud example):
           https://www.esteemprojects.com/library/Fired-Heater-Design/page4_3.htm
  [HTC]    Heat Transfer Consult, «A fired heater example» (cylindrical reboiler, «Calculations according to the Lobo-Evans
           method put the bridgewall temperature (BWT) at about 867 degrees centigrade») and its API 560 data sheets 1-3:
           http://www.heattransferconsult.nl/heater.html , .../Htr-Example.sht01 , .../Htr_Example.sht02 , .../Htr_Example.sht03
Searched and NOT usable today: Nelson (1941) ex. 66 and 68 (Wilson-Lobo-Hottel only, no firebox geometry); Al-Haj Ibrahim and
Al-Qassimi, Per. Pol. Chem. Eng. 54 (2010) 33 (Homs crude heater: no inside firebox dimensions); Hofmann et al., WSEAS Trans.
Heat Mass Transfer 2 (2007) 72 (finned staggered banks: fitted constants only in figures); no legitimate open copy of
Ganapathy's appendix E and no open worked example of studded tubes was found.
No network, no DWSIM. Old examples are in Btu, ft and °F: converted here, never retyped in SI by hand."""
from __future__ import annotations

import json
import math

import pytest
from scipy.optimize import brentq

from packages.dwsim_client import tools_combustion as TCB
from packages.energy import combustion as CB
from packages.energy import fired_heater as FH

P = 101325.0
FT, FT2, IN, LB = 0.3048, 0.09290304, 0.0254, 0.45359237
BTU_H, BTU_LB, HBTU, KBTU = 0.29307107, 2326.0, 5.678263, 1.7307347     # HBTU W/m2K, KBTU W/mK per English unit


def F2K(t_f: float) -> float:
    return (t_f - 32.0) / 1.8 + 273.15


def K2F(t_k: float) -> float:
    return (t_k - 273.15) * 1.8 + 32.0


# ----------------------------------------------------------------------------------------------------------------------------
# 1. Esteem Projects box heater [EST-R], [EST-S], [EST-C]
# ----------------------------------------------------------------------------------------------------------------------------
# Given: heat absorbed 9 500 000 Btu/h, tube wall 600 °F, «previous example fuel» = gas, 15 % excess air; 30 tubes 4.5 in on
# 8 in, 26 ft effective, box 8 x 10.333 x 26 ft, flue-gas exit 42 ft2; A_cp 520.0, alpha 0.915 (chart), A_r 1076.644 (box minus
# exit), A_w/alpha A_cp 1.2628, beam 2/3 V^(1/3) = 8.6012 ft, CO2 0.085586 + H2O 0.172186; at T_g 1500 °F: emissivity 0.427,
# F 0.597, q_r 6 631 794, q_c = 1.5 x 918.918 x 900 = 1 240 539 Btu/h; loss 1.5 % of the release; W_fuel = 9 500 000 / (20345 -
# (20345 x 0.015 + 18.7964 x 443.1019 + 443.1019)) = 843.0953 lb/h. [EST-S] states the converged state: F 0.585, T_g 2063 °R.
# The fuel of [EST-C] (mole %): CH4 92.8607, C2H6 3.3911, C3H8 0.8220, n-C4 0.1760, i-C4 0.1760, n-C5 0.0430, i-C5 0.0710,
# C6H14 0.1890, N2 1.0111, CO2 1.2641; air N2 77.2240, O2 20.7705, CO2 0.0297, H2O 1.0456, Ar 0.9302.
# NOT a species of the engine: n-hexane, replaced by n-pentane with the same carbon (0.1890 x 6/5).
EST_FUEL_RAW = {"CH4": 92.8607, "C2H6": 3.3911, "C3H8": 0.8220, "n-C4H10": 0.1760, "i-C4H10": 0.1760, "n-C5H12": 0.0430 + 0.1890 * 6 / 5,
                "i-C5H12": 0.0710, "N2": 1.0111, "CO2": 1.2641}
EST_FUEL = {"type": "gas", "composition": {k: v / sum(EST_FUEL_RAW.values()) for k, v in EST_FUEL_RAW.items()}, "T_K": F2K(60.0)}
_DRY = {"N2": 77.2240, "O2": 20.7705, "CO2": 0.0297, "Ar": 0.9302}
_DRY = {k: v / sum(_DRY.values()) for k, v in _DRY.items()}
_MDRY = _DRY["N2"] * 28.0134 + _DRY["O2"] * 31.998 + _DRY["CO2"] * 44.0095 + _DRY["Ar"] * 39.948
EST_AIR = {"T_K": F2K(60.0), "humidity_ratio_kg_kg": 0.010456 / (1 - 0.010456) * 18.015 / _MDRY, "dry_air_mole_fractions": _DRY}
EST_AR_FT2 = 8 * 26 * 2 + 8 * 10.333 * 2 + 10.333 * 26 * 2 - 42
EST_BOX = {"tube_outside_diameter_m": 4.5 * IN, "tube_pitch_m": 8 * IN, "tube_effective_length_m": 26 * FT, "n_tubes": 30,
           "envelope_area_m2": EST_AR_FT2 * FT2, "volume_m3": 8 * 10.333 * 26 * FT ** 3, "mean_beam_length_m": 8.6012 * FT}
EST_W_LB_H = 843.0953
# [EST-C] gasprops script: molar cp of N2, O2, CO2, H2O, Ar, SO2, CO = a + b t + c t^2 + d t^3 (Btu/lbmol °F, t in °F),
# enthalpy integrated from 60 °F, mole weights of the script
EST_CP = ((6.921525928892, 0.000264324135, 0.000000433542006, -1.17644032e-10),
          (6.905598931009, 0.00156659667, -0.000000453431059, 5.37208504e-11),
          (8.44682086499, 0.00575849121, -0.00000215927489, 3.05898491e-10),
          (7.967187267909, 0.000876904142, 0.000000575161121, -1.47239368e-10),
          (4.964694915254, 0.0, 0.0, 0.0),
          (9.13051799475, 0.00533077398, -0.00000232346409, 3.5270919e-10),
          (6.902864194523, 0.000529496335, 0.00000027057429, -9.09979423e-11))
EST_MW = (28.016, 32.0, 44.01, 18.016, 39.95, 64.06, 28.01)


def esteem_enthalpy_Btu_lb(y: list, t_F: float) -> float:
    w = [a * m for a, m in zip(y, EST_MW)]
    w = [x / sum(w) for x in w]
    return sum(sum(cf[j] / (j + 1) * (t_F ** (j + 1) - 60.0 ** (j + 1)) for j in range(4)) * w[i] / EST_MW[i]
               for i, cf in enumerate(EST_CP))


def esteem_box(fuel_lb_h: float = EST_W_LB_H, **overrides) -> dict:
    geometry = dict(EST_BOX, **overrides.pop("geometry", {}))
    args = dict(fuel=EST_FUEL, air=EST_AIR, pressure_Pa=P, fuel_flow_kg_s=fuel_lb_h * LB / 3600, geometry=geometry,
                tube_emissivity=0.9, convective_coefficient_W_m2K=1.5 * HBTU, setting_loss_fraction=0.015,
                setting_loss_basis="LHV", excess_air_fraction=0.15, tube_wall_T_K=F2K(600.0))
    args.update(overrides)
    return FH.radiant_section(**args)


def test_esteem_fuel_and_flue_gas_are_reproduced():
    c = CB.solve_combustion(EST_FUEL, EST_AIR, P, excess_air_fraction=0.15)
    assert c.LHV / BTU_LB / 20345.0 - 1 == pytest.approx(-0.0006, abs=0.0005)          # 20 334 Btu/lb
    assert c.air_mass / 18.7964 - 1 == pytest.approx(0.0, abs=0.0005)
    assert c.flue["CO2"] / c.flue_mol == pytest.approx(0.085586, rel=0.001)
    assert c.flue["H2O"] / c.flue_mol == pytest.approx(0.172186, rel=0.001)
    # the source's own enthalpy polynomials and the engine (NASA) agree on this gas: -0.28 % at 500 °F, -0.01 % at 1500 °F
    y = [c.flue.get(k, 0.0) / c.flue_mol for k in ("N2", "O2", "CO2", "H2O", "Ar", "SO2", "CO")]
    for t, dev in ((500.0, -0.0028), (1000.0, -0.0021), (1500.0, -0.0001), (1600.0, 0.0), (2000.0, -0.0002)):
        h_eng = (c.flue_sensible(F2K(t)) - c.flue_sensible(F2K(60.0))) / c.flue_mass / BTU_LB
        assert h_eng / esteem_enthalpy_Btu_lb(y, t) - 1 == pytest.approx(dev, abs=0.0003)
    # the printed 443.1019 Btu/lb is the enthalpy at 1596.5 °F, not at 1500 °F (412.7): the fuel of the example is the one
    # that holds the converged gas temperature (2063 °R = 1603 °F in [EST-S]), which is what the rating below compares
    t_443 = brentq(lambda t: esteem_enthalpy_Btu_lb(y, t) - 443.1019, 1500.0, 1700.0)
    assert t_443 == pytest.approx(1596.5, abs=0.2) and esteem_enthalpy_Btu_lb(y, 1500.0) == pytest.approx(412.75, abs=0.05)
    # and his arithmetic of the fuel
    W = 9.5e6 / (20345 - (20345 * 0.015 + 18.7964 * 443.1019 + 443.1019))
    assert W == pytest.approx(EST_W_LB_H, rel=1e-6)


def test_esteem_radiation_terms_at_fixed_temperatures():
    Acp = 30 * 8 / 12 * 26
    assert Acp == pytest.approx(520.0) and EST_AR_FT2 == pytest.approx(1076.644, abs=1e-3)
    assert FH.tube_row_alpha(8.0, 4.5)["alpha"] == pytest.approx(0.922, abs=0.001)            # chart 0.915 (+0.8 %)
    # exchange factor of the engine's network with HIS emissivity and alpha, against Mekler-Fairall's chart 0.597: -1.7 %
    assert FH.exchange_factor(0.427, 0.915 * Acp, EST_AR_FT2, 0.9) / 0.597 - 1 == pytest.approx(-0.017, abs=0.003)
    # WSGG emissivity at 1500 °F and pL 2.2174 atm ft: 0.400 against his Lobo-Evans curve 0.427 (-6 %)
    c = CB.solve_combustion(EST_FUEL, EST_AIR, P, excess_air_fraction=0.15)
    pw, pc = c.flue["H2O"] / c.flue_mol * P, c.flue["CO2"] / c.flue_mol * P
    eg = FH.gas_emissivity(F2K(1500.0), pw, pc, 8.6012 * FT)
    assert eg["pL_atm_m"] / FT == pytest.approx(2.2174, rel=0.001) and eg["emissivity"] / 0.427 - 1 == pytest.approx(-0.062, abs=0.005)
    # his transfer equation with his numbers (arithmetic) and the converged state of [EST-S]: 9.497e6 against 9.5e6
    qr = 0.173e-8 * 475.8 * 0.597 * (1960.0 ** 4 - 1060.0 ** 4)
    assert qr == pytest.approx(6631794.0, rel=1e-6) and 1.5 * 918.918 * 900 == pytest.approx(1240539.0, abs=1.0)
    conv = 0.173e-8 * 475.8 * 0.585 * (2063.0 ** 4 - 1060.0 ** 4) + 1.5 * 918.918 * (2063.0 - 1060.0)
    assert conv / 9.5e6 - 1 == pytest.approx(-0.0003, abs=0.0005)
    assert 30 * math.pi * 4.5 / 12 * 26 == pytest.approx(918.918, abs=0.01)


def test_esteem_box_heater_rating():
    # (a) everything of the engine, with his beam length: at his firing the duty is 1.4 % low and the gas 18-24 °F hot
    r = esteem_box()
    assert r["radiant_duty_W"] / BTU_H / 9.5e6 - 1 == pytest.approx(-0.014, abs=0.003)          # measured -1.43 %
    assert K2F(r["bridgewall_T_K"]) == pytest.approx(1620.9, abs=1.0)
    # +9.9 K over his converged 2063 °R and +13.5 K over the 1596.5 °F at which his printed enthalpy holds
    assert (K2F(r["bridgewall_T_K"]) - 1603.0) / 1.8 == pytest.approx(9.9, abs=0.6)
    assert (K2F(r["bridgewall_T_K"]) - 1596.5) / 1.8 == pytest.approx(13.5, abs=0.6)
    assert r["radiation"]["exchange_factor"] == pytest.approx(0.555, abs=0.002)                 # his 0.585: -5 %
    assert r["average_flux_W_m2_tube_outside"] / BTU_H * FT2 == pytest.approx(9.5e6 / 918.918, rel=0.016)
    assert r["evidence"]["verified"] and r["evidence"]["energy_balance_error"] < 1e-9
    # (b) his alpha and his converged F imposed: only the heat balance and the enthalpies are the engine's: +4.5 °F, -0.8 % fuel
    W = brentq(lambda w: esteem_box(w, geometry={"alpha": 0.915}, exchange_factor_override=0.585)["radiant_duty_W"] / BTU_H
               - 9.5e6, 700.0, 1000.0)
    b = esteem_box(W, geometry={"alpha": 0.915}, exchange_factor_override=0.585)
    assert K2F(b["bridgewall_T_K"]) - 1603.0 == pytest.approx(5.0, abs=1.5) and W / EST_W_LB_H - 1 == pytest.approx(0.007, abs=0.002)
    # (c) the engine's default beam length 3.6 V/A (7.19 ft against his 8.60 ft): duty -2.1 % at his firing
    d = esteem_box(geometry={"mean_beam_length_m": None})
    assert d["radiation"]["mean_beam_length_m"] / FT == pytest.approx(7.187, abs=0.005)
    assert d["radiant_duty_W"] / BTU_H / 9.5e6 - 1 == pytest.approx(-0.021, abs=0.003)
    json.dumps(r)


# ----------------------------------------------------------------------------------------------------------------------------
# 2. Heat Transfer Consult cylindrical reboiler [HTC]
# ----------------------------------------------------------------------------------------------------------------------------
# Data sheets (API 560, SI): cylindrical with overhead convection, natural draft; methane 100 %, 15 % excess air, heat release
# 26.26 MW, efficiency 80.24 %, heat loss 1.28 % of liberation (radiant section 299 kW), flue gas 11.05 kg/s, ambient 15 °C;
# radiant 14.91 MW, flue gas leaving radiant 830 °C (the author's imposed «educated guess»; his Lobo-Evans run gives ~867 °C);
# radiant coil per pass 4 x 8 in (219.08 mm, on 406.4 mm, 14.012 m exposed) + 2 x 6 in (168.27 mm, on 304.8 mm, 14.005 m),
# 8 passes, wall tubes 304.8 mm from the refractory, 308.6 + 118.5 m2; fluid vaporising at 288 °C; max tube metal 377/360 °C;
# refractory ID 6.30 m, radiant internal height 14.5 m; shock rows 8 tubes of 6 in on 254 mm, 6.24 m, three rows.
# NOT given, taken here: the convective coefficient in the firebox, 3.0 Btu/h ft2 °F (the rule of [EST-R] for vertical
# cylinders with L/D > 2: here 2.30); the mean tube wall, 330 °C (fluid 288 °C plus the max metal excess scaled by average to
# peak flux); air dry; beam length = diameter (Wimpress' rule of [EST-R] for L/D >= 2); first shock row as a shield cold plane
# with alpha 1 and that opening taken out of the refractory (the shield rule of [EST-S]). The two tube sizes enter as one
# equivalent tube with the same total tube area, cold-plane area and area-weighted exact alpha.
HTC_FUEL = {"type": "gas", "composition": {"CH4": 1.0}, "T_K": 288.15}
HTC_AIR = {"T_K": 288.15, "humidity_ratio_kg_kg": 0.0}
_ACP8, _ACP6 = 32 * 0.4064 * 14.012, 16 * 0.3048 * 14.005
_AT8, _AT6 = 32 * math.pi * 0.21908 * 14.012, 16 * math.pi * 0.16827 * 14.005
HTC_ALPHA = (FH.tube_row_alpha(0.4064, 0.21908)["alpha"] * _ACP8 + FH.tube_row_alpha(0.3048, 0.16827)["alpha"] * _ACP6) / \
    (_ACP8 + _ACP6)
HTC_SHIELD = 8 * 0.254 * 6.24
HTC_GEO = {"tube_outside_diameter_m": (_AT8 + _AT6) / (48 * math.pi * 14.0), "tube_pitch_m": (_ACP8 + _ACP6) / (48 * 14.0),
           "tube_effective_length_m": 14.0, "n_tubes": 48, "alpha": HTC_ALPHA, "shield_cold_plane_area_m2": HTC_SHIELD,
           "envelope_area_m2": math.pi * 6.30 * 14.5 + 2 * math.pi / 4 * 6.30 ** 2 - HTC_SHIELD,
           "volume_m3": math.pi / 4 * 6.30 ** 2 * 14.5, "mean_beam_length_m": 6.30}


def htc(**overrides) -> dict:
    c = CB.solve_combustion(HTC_FUEL, HTC_AIR, P, excess_air_fraction=0.15)
    geometry = dict(HTC_GEO, **overrides.pop("geometry", {}))
    args = dict(fuel=HTC_FUEL, air=HTC_AIR, pressure_Pa=P, fuel_flow_kg_s=26.26e6 / c.LHV, geometry=geometry,
                tube_emissivity=0.9, convective_coefficient_W_m2K=3.0 * HBTU, setting_loss_fraction=0.299 / 26.26,
                setting_loss_basis="LHV", excess_air_fraction=0.15, tube_wall_T_K=330.0 + 273.15)
    args.update(overrides)
    return FH.radiant_section(**args)


def test_htc_data_sheet_is_consistent_with_the_combustion_engine():
    assert _AT8 == pytest.approx(308.6, rel=0.001) and _AT6 == pytest.approx(118.5, rel=0.001)
    assert 14.91e6 / (_AT8 + _AT6) == pytest.approx(34910.0, rel=0.001)                      # his average radiant flux
    assert _ACP8 + _ACP6 == pytest.approx(math.pi * 5.69 * 14.0, rel=0.002)                    # his tube circle 5.69 m
    c = CB.solve_combustion(HTC_FUEL, HTC_AIR, P, excess_air_fraction=0.15)
    m = 26.26e6 / c.LHV
    assert m * c.flue_mass / 11.05 - 1 == pytest.approx(-0.013, abs=0.003)                     # dry air; humid air -0.7 %
    # his radiant balance at 830 °C closes with a flue-gas enthalpy 2.6 % above the engine's: 15.31 MW against his 14.91
    q = 26.26e6 + m * c.credits - 0.299e6 - m * c.flue_sensible(830.0 + 273.15)
    assert q / 14.91e6 - 1 == pytest.approx(0.027, abs=0.003)


def test_htc_cylindrical_heater_against_its_lobo_evans_bridgewall():
    r = htc()
    assert r["bridgewall_T_K"] - 273.15 - 867.0 == pytest.approx(-5.4, abs=1.5)               # measured 861.6 °C
    assert r["radiant_duty_W"] / 1e6 == pytest.approx(14.28, abs=0.05)                          # his imposed 14.91 at 830 °C
    assert r["shield_radiant_duty_W"] / 1e6 == pytest.approx(0.58, abs=0.02)
    assert r["radiation"]["gas_emissivity"] == pytest.approx(0.477, abs=0.003)
    assert r["evidence"]["verified"]
    # what the unpublished inputs are worth: tube wall ±30 K -> ∓7 K; firebox convection 2 instead of 3 Btu/h ft2 °F -> +22 K;
    # without the shock row +6 K; beam 3.6 V/A (4.83 m) +9 K
    T = r["bridgewall_T_K"]
    assert htc(tube_wall_T_K=300.0 + 273.15)["bridgewall_T_K"] - T == pytest.approx(-6.5, abs=1.0)
    assert htc(tube_wall_T_K=360.0 + 273.15)["bridgewall_T_K"] - T == pytest.approx(6.9, abs=1.0)
    assert htc(convective_coefficient_W_m2K=2.0 * HBTU)["bridgewall_T_K"] - T == pytest.approx(21.6, abs=1.5)
    assert htc(geometry={"shield_cold_plane_area_m2": 0.0})["bridgewall_T_K"] - T == pytest.approx(5.8, abs=1.0)
    assert htc(geometry={"mean_beam_length_m": None})["bridgewall_T_K"] - T == pytest.approx(8.9, abs=1.0)
    json.dumps(r)


def test_radiant_validation_statement_names_the_new_cases():
    text = htc()["evidence"]["validated_against_literature"]
    assert "Esteem" in text and "Heat Transfer Consult" in text and "±5 %" in text and "±35 K" in text
    doc = TCB.fired_heater_radiant.__doc__
    assert "Esteem" in doc and "API 560" in doc and "±5 %" in doc and "±35 K" in doc


# ----------------------------------------------------------------------------------------------------------------------------
# 3. Studded tubes [EST-T]
# ----------------------------------------------------------------------------------------------------------------------------
# Default inputs of the stud script: tube 6.625 in, 12 wide, longitudinal pitch 9.922 in, transverse 13.5 in, staggered, 4
# rows, 33.25 ft; studs 1.75 in long, 0.50 in diameter, 22 per plane, 19.2 planes per ft, k 24.8 Btu/h ft °F; outside fouling
# 0.01 h ft2 °F/Btu; gas 1 510 947 lb/h at 1500 °F. The script prints nothing: the reference below is the page's formulas
# coded literally in its units (an independent transcription of what the engine codes in SI).
STUD = {"od": 6.625, "nt": 12, "pl": 9.922, "st": 13.5, "nr": 4, "le": 33.25, "lf": 1.75, "sp": 22, "tf": 0.50, "pf": 19.2,
        "ks": 24.8, "rfo": 0.01, "wg": 1510947.0, "tb": 1500.0}


def esteem_stud_script(x: dict, layout: str = "staggered") -> dict:
    od, nt, st, le, lf, sp, tf, pf, ks, rfo, wg, tb = (x[k] for k in ("od", "nt", "st", "le", "lf", "sp", "tf", "pf", "ks", "rfo",
                                                                    "wg", "tb"))
    an = le * nt * (st - od - (lf * tf * pf) / 12) / 12
    gn = wg / an
    ht = (0.717 / od ** 0.333) * (gn / 1000) ** 0.67 * (tb + 460) ** 0.3
    hs = 0.936 * (gn / 1000) ** 0.67 * (tb + 460) ** 0.3
    hso = 1 / (1 / hs + rfo)
    X = lf / 12 * ((2 * hso) / (ks * tf / 12)) ** 0.5
    E = 1 / ((math.exp(X) + math.exp(-X)) / 1.950)
    apo = math.pi * od / 12 - ((tf * tf * math.pi / 4) / 144 * sp * pf)
    afo = (tf * math.pi * lf + tf * tf * math.pi / 4) / 144 * sp * pf
    he = (hso * E * afo + ht * apo) / (apo + afo)
    if layout == "inline":
        he *= (od / x["pl"]) ** 0.333
    return {"an": an, "gn": gn, "ht": ht, "hs": hs, "hso": hso, "X": X, "E": E, "apo": apo, "afo": afo, "he": he}


def engine_stud(x: dict, layout: str = "staggered") -> dict:
    g = FH.studded_tube_geometry(x["od"] * IN, x["lf"] * IN, x["tf"] * IN, x["sp"], x["pf"] / FT)
    G = x["wg"] * LB / 3600 / (x["le"] * FT * x["nt"] * (x["st"] * IN - g["obstruction_m"]))
    return dict(FH.stud_coefficients(G, F2K(x["tb"]), x["od"] * IN, x["lf"] * IN, x["tf"] * IN, x["ks"] * KBTU, x["rfo"] / HBTU,
                                     g, layout, x["pl"] * IN), geo=g, G=G)


def test_stud_correlation_is_coded_as_printed():
    for layout in ("staggered", "inline"):
        ref, eng = esteem_stud_script(STUD, layout), engine_stud(STUD, layout)
        assert eng["Gn_lb_h_ft2"] == pytest.approx(ref["gn"], rel=1e-9)
        assert eng["geo"]["tube_area_m2_per_m"] * FT / FT2 == pytest.approx(ref["apo"], rel=1e-9)
        assert eng["geo"]["stud_area_m2_per_m"] * FT / FT2 == pytest.approx(ref["afo"], rel=1e-9)
        for k, kr in (("h_t", "ht"), ("h_s", "hs"), ("h_so", "hso"), ("h_e", "he")):
            assert eng[k] / HBTU == pytest.approx(ref[kr], rel=1e-6)
        assert eng["X"] == pytest.approx(ref["X"], rel=1e-6) and eng["E"] == pytest.approx(ref["E"], rel=1e-6)
    ref = esteem_stud_script(STUD)
    assert ref["an"] == pytest.approx(182.044, abs=0.001) and ref["he"] == pytest.approx(16.333, abs=0.001)
    assert engine_stud(STUD, "inline")["inline_factor"] == pytest.approx((6.625 / 9.922) ** 0.333)
    # the source's efficiency 0.975 sech(X) is a fit of the pin fin with adiabatic tip, tanh(mL)/(mL), m = (4h/(kD))^0.5, i.e.
    # mL = X sqrt(2): -0.08 % at the default stud (X 1.06), within ±2.4 % for X <= 1.2, -9.5 % at 1.5, -26 % at 2 (warned)
    e = engine_stud(STUD)
    pin = lambda mL: math.tanh(mL) / mL                                                       # noqa: E731
    mL = 1.75 * IN * math.sqrt(4 * e["h_so"] / (24.8 * KBTU * 0.5 * IN))
    assert mL == pytest.approx(e["X"] * math.sqrt(2)) and e["E"] / pin(mL) - 1 == pytest.approx(-0.0008, abs=0.0005)
    devs = [1.950 / (math.exp(X) + math.exp(-X)) / pin(X * math.sqrt(2)) - 1 for X in (0.1, 0.3, 0.5, 0.8, 1.0, 1.2)]
    assert max(abs(d) for d in devs) < 0.024
    assert 1.950 / (math.exp(1.5) + math.exp(-1.5)) / pin(1.5 * math.sqrt(2)) - 1 == pytest.approx(-0.095, abs=0.002)
    assert 1.950 / (math.exp(2.0) + math.exp(-2.0)) / pin(2.0 * math.sqrt(2)) - 1 == pytest.approx(-0.262, abs=0.002)
    assert engine_stud(dict(STUD, lf=1e-6))["E"] == pytest.approx(0.975, abs=1e-6)


FUEL = {"type": "gas", "composition": {"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}, "T_K": 300.0}
AIR = {"T_K": 300.0, "humidity_ratio_kg_kg": 0.008}
BANK = {"tube_outside_diameter_m": 6.625 * IN, "tube_inside_diameter_m": 6.065 * IN, "tube_effective_length_m": 33.25 * FT,
        "tubes_per_row": 12, "n_rows": 4, "transverse_pitch_m": 13.5 * IN, "longitudinal_pitch_m": 9.922 * IN,
        "arrangement": "staggered", "wall_conductivity_W_mK": 40.0, "outside_fouling_m2K_W": 0.01 / HBTU, "tube_emissivity": 0.9}
STUDS = {"tube_type": "studded", "stud_length_m": 1.75 * IN, "stud_diameter_m": 0.5 * IN, "studs_per_plane": 22,
         "stud_planes_per_m": 19.2 / FT, "stud_conductivity_W_mK": 24.8 * KBTU}
PROC = {"inlet_T_K": 600.0, "isothermal": True, "inside_coefficient_W_m2K": 3000.0, "inside_fouling_m2K_W": 0.0002}


def stud_section(**bank_overrides) -> dict:
    c = CB.solve_combustion(FUEL, AIR, P, excess_air_fraction=0.15)
    b = dict(BANK, **bank_overrides)
    gas_radiation = b.pop("gas_radiation", True)
    return FH.convection_section(FUEL, AIR, P, STUD["wg"] * LB / 3600 / c.flue_mass, F2K(1500.0) + 40.0, b, PROC,
                                 excess_air_fraction=0.15, gas_radiation=gas_radiation)


def test_studded_section_limits_identities_and_flags():
    bare, studs = stud_section(), stud_section(**STUDS)
    tiny = stud_section(**dict(STUDS, stud_length_m=1e-4, stud_diameter_m=1e-4, studs_per_plane=1, stud_planes_per_m=1.0))
    # no-stud limit: the source's base-tube coefficient (radiation «built in») equals Žukauskas convection alone (-1 %) and is
    # 31 % under this module's bare tube with WSGG gas radiation; the section duty is 16 % lower -> possible underestimate
    hb, ht = bare["rows"][0], tiny["rows"][0]
    assert ht["h_base_tube_W_m2K"] / hb["h_convection_W_m2K"] - 1 == pytest.approx(-0.010, abs=0.01)
    assert ht["h_convection_W_m2K"] / (hb["h_convection_W_m2K"] + hb["h_radiation_W_m2K"]) - 1 == pytest.approx(-0.31, abs=0.02)
    assert tiny["duty_W"] / bare["duty_W"] - 1 == pytest.approx(-0.156, abs=0.01)
    assert tiny["outside_area_m2"] == pytest.approx(bare["outside_area_m2"], rel=1e-6)
    # studs: 5.6 times the area, 3.8 times the duty of the bare bank; closure, flags and no WSGG
    assert studs["outside_area_m2"] / bare["outside_area_m2"] == pytest.approx(5.65, abs=0.05)
    assert studs["duty_W"] / bare["duty_W"] == pytest.approx(3.80, abs=0.05)
    for r in (studs, tiny):
        assert r["evidence"]["verified"] and r["evidence"]["energy_balance_error"] < 1e-9
        assert r["evidence"]["validated_against_literature"] == FH.STUDDED_VALIDATION
        assert r["evidence"]["validated_against_literature"].startswith("NO validado")
        assert r["method"]["tube_type"] == "studded" and "Esteem" in r["method"]["convection_correlation"]
        assert all(row["h_radiation_W_m2K"] == 0.0 and row["gas_emissivity"] is None for row in r["rows"])
        assert any("pernos" in w for w in r["warnings"]) and r["mean_beam_length_m"] is None and r["max_fin_tip_T_K"] is None
        json.dumps(r, allow_nan=False)
    assert studs["rows"][0]["stud_efficiency"] == pytest.approx(engine_stud(STUD)["E"], abs=0.01)
    # monotonic: more planes more duty; longer studs lower efficiency; a better conductor higher efficiency
    more = stud_section(**dict(STUDS, stud_planes_per_m=22.0 / FT))
    longer = stud_section(**dict(STUDS, stud_length_m=2.5 * IN))
    copper = stud_section(**dict(STUDS, stud_conductivity_W_mK=200.0))
    assert not any("X =" in w for w in studs["warnings"]) and any("X =" in w for w in longer["warnings"])
    assert more["duty_W"] > studs["duty_W"] and longer["rows"][0]["stud_efficiency"] < studs["rows"][0]["stud_efficiency"]
    assert copper["rows"][0]["stud_efficiency"] > studs["rows"][0]["stud_efficiency"] and copper["duty_W"] > studs["duty_W"]
    # in-line: the (d/P_l)^0.333 factor lowers the coefficient
    inline = stud_section(**dict(STUDS, arrangement="inline"))
    assert inline["duty_W"] < studs["duty_W"]
    # gas_radiation=False cannot remove what is inside the correlation: same duty and a warning
    nogr = stud_section(**dict(STUDS, gas_radiation=False))
    assert nogr["duty_W"] == pytest.approx(studs["duty_W"], rel=1e-12) and any("gas_radiation=False" in w for w in nogr["warnings"])


def test_studded_refusals_and_tool_text():
    with pytest.raises(ValueError, match="pernos"):
        stud_section(tube_type="studded")
    with pytest.raises(ValueError, match="stud_conductivity_W_mK"):
        stud_section(**dict(STUDS, stud_conductivity_W_mK=None))
    with pytest.raises(ValueError, match="no cabe"):
        stud_section(**dict(STUDS, stud_length_m=3.5 * IN))
    with pytest.raises(ValueError, match="cubren"):
        stud_section(**dict(STUDS, studs_per_plane=200, stud_planes_per_m=200.0, stud_diameter_m=0.005))
    with pytest.raises(ValueError, match="'studded'"):
        stud_section(tube_type="pins")
    doc = TCB.fired_heater_convection.__doc__
    assert "studded" in doc and "stud_planes_per_m" in doc and "NO validado" in doc and "cotejar" in doc
    finned = FH.FINNED_VALIDATION
    assert "cotejar" in finned and "Sin tubos con pernos" not in finned
