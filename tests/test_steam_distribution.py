"""Steam distribution losses (bloque M20b): `steam_leak_loss`, `failed_trap_loss`, `bare_surface_heat_loss`.

Published numbers reproduced here were read on 2026-11-14 (curl + pdftotext -layout of the PDFs; nothing was tuned to match):
  * U.S. DOE Steam Tip Sheet #1 «Inspect and Repair Steam Traps», DOE/GO-102012-3401, January 2012,
    https://www.energy.gov/sites/prod/files/2014/05/f16/steam1_traps.pdf  (table «Leaking Steam Trap Discharge Rate»).
    Its footnote: «From the Boiler Efficiency Institute. Steam is discharging to atmospheric pressure through a re-entrant
    orifice with a coefficient of discharge equal to 0.72.» It does NOT name the flow formula. The table is linear in
    absolute pressure: lb/h = 29.46 * D[in]^2 * P[psia] within its rounding, which is 1.3 % above 0.72 * Napier.
  * U.S. DOE Steam Tip Sheet #2 «Insulate Steam Distribution and Condensate Return Lines», January 2012,
    https://www.energy.gov/sites/prod/files/2014/05/f16/steam2_insulate.pdf  (table «Heat Loss Per 100 Feet of Uninsulated
    Steam Line»). Its footnote: «Based on horizontal steel pipe, 75°F ambient air, no wind velocity, and 8,760 operating
    hours per year.» It does NOT state the emissivity, the surface temperature, the real outer diameters, the barometric
    pressure or the program that produced it. Saturation temperature as surface temperature, ASME B36.10 diameters and
    101 325 Pa are OUR reading; the emissivity is a sensitivity, not a validation.
  * G. Harrell, «Steam System Survey Guide», ORNL/TM-2001/263, May 2002 (local text copy of the report), Table 6 «Steam leak
    rates» (sharp-edged orifice, steam at 500 °F; the text says a sharp-edged orifice passes «approximately 60 %» of the
    well-rounded flow, the table itself states no coefficient), eq. 56 (Napier) and Table 7 «Pipe surface heat transfer»
    (bare pipe, 10 mile/h wind, 70 °F ambient; no emissivity stated).
  * M.M. Yovanovich, Maple worksheet EX9P4.MWS, «Example 9.4 of 4th edition of Incropera and DeWitt», University of Waterloo,
    https://www.mhtlab.uwaterloo.ca/old/courses/me353/maple/examples/ex9p41.html : horizontal steam pipe, every input stated
    (D = 0.1 m, 165 °C, air and surroundings at 23 °C, emissivity 0.85; then 25 mm of insulation, k = 0.026 W/(m K)). The
    worksheet uses tabulated air properties at 367 K for both cases; the engine uses the film temperature of each case.
No network, no DWSIM."""
from __future__ import annotations

import json
import math

import CoolProp.CoolProp as CP
import pytest

from packages.energy import steam_distribution as SDN

PSI = 6894.757293168
ATM = 101325.0
ATM_PSI = ATM / PSI
INCH = 0.0254
FOOT = 0.3048
LB = 0.45359237
BTU = 1055.05585262                     # J (IT)
MPH = 0.44704


def F(t_f: float) -> float:
    return (t_f - 32.0) / 1.8 + 273.15


def psig(p: float) -> float:
    return (p + ATM_PSI) * PSI


def lb_h(kg_s: float) -> float:
    return kg_s / LB * 3600.0


# DOE Steam Tip Sheet #1 (January 2012), steam loss in lb/h; columns 15, 100, 150 and 300 psig; «–» in the sheet = None
TIP1_PSIG = (15, 100, 150, 300)
TIP1_CD = 0.72
TIP1 = {1 / 32: (0.85, 3.3, 4.8, None), 1 / 16: (3.4, 13.2, 18.9, 36.2), 1 / 8: (13.7, 52.8, 75.8, 145),
        3 / 16: (30.7, 119, 170, 326), 1 / 4: (54.7, 211, 303, 579), 3 / 8: (123, 475, 682, 1303)}

# ORNL/TM-2001/263 Table 6, leak rate in lbm/h at 500 °F through a sharp-edged orifice; columns in psig
T6_PSIG = (50, 100, 150, 200, 250, 300, 350)
T6_CD = 0.60       # «approximately 60 %» in the text of the guide; the table does not state it
T6 = {1 / 8: (23, 41, 59, 77, 96, 119, 134), 1 / 4: (91, 163, 235, 308, 382, 478, 536),
      3 / 8: (206, 366, 529, 693, 860, 1075, 1207), 1 / 2: (366, 651, 940, 1232, 1528, 1912, 2145),
      3 / 4: (822, 1465, 2115, 2773, 3438, 4302, 4826), 1.0: (1462, 2605, 3761, 4929, 6112, 7648, 8580),
      1.25: (2285, 4071, 5876, 7702, 9551, 11949, 13406), 1.5: (3290, 5862, 8462, 11091, 13753, 17207, 19305)}

# ASME B36.10 outer diameter (in) by nominal pipe size
OD_IN = {0.5: 0.840, 1: 1.315, 2: 2.375, 3: 3.5, 4: 4.5, 5: 5.563, 6: 6.625, 8: 8.625, 10: 10.75, 12: 12.75, 16: 16.0,
         20: 20.0, 24: 24.0}

# DOE Steam Tip Sheet #2 (January 2012), MMBtu/yr per 100 ft of bare line; columns 15, 150, 300 and 600 psig
TIP2_PSIG = (15, 150, 300, 600)
TIP2 = {1: (140, 285, 375, 495), 2: (235, 480, 630, 840), 4: (415, 850, 1120, 1500), 8: (740, 1540, 2030, 2725),
        12: (1055, 2200, 2910, 3920)}

# ORNL/TM-2001/263 Table 7, Btu/(h linear ft), bare pipe, 10 mile/h wind, 70 °F ambient; columns = fluid temperature (°F)
T7_F = (200, 400, 600, 800, 1000, 1200)
T7 = {0.5: (274, 731, 1279, 1963, 2865, 4030), 1: (354, 959, 1712, 2694, 3995, 5708), 2: (514, 1416, 2591, 4167, 6324, 9247),
      3: (708, 1849, 3425, 5605, 8619, 12728), 4: (845, 2352, 4132, 6838, 10605, 15776),
      5: (982, 2751, 5126, 8105, 12671, 18938), 6: (1107, 3128, 5868, 9726, 14692, 22055),
      8: (1336, 3824, 7237, 12089, 18973, 27785), 10: (1575, 4532, 8642, 14543, 22945, 34498),
      12: (1792, 5183, 9932, 16826, 26678, 40274), 16: (2135, 6210, 12009, 20491, 32705, 49623),
      20: (2534, 7443, 14509, 24932, 40034, 61039), 24: (2934, 8699, 17078, 29315, 47283, 72352)}

LEAK = dict(steam_pressure_Pa=8.0e5, hole_diameter_m=0.005, discharge_coefficient=0.8, back_pressure_Pa=ATM)
PIPE = {"label": "2 in", "outer_diameter_m": 0.0603, "length_m": 30.0, "steam_pressure_Pa": 1.1e6,
        "surface_emissivity": 0.8, "insulation_thickness_m": 0.05, "insulation_conductivity_W_m_K": 0.045,
        "jacket_emissivity": 0.1}
AMBIENT = dict(ambient_T_K=298.15, wind_speed_m_s=0.0, ambient_pressure_Pa=78000.0)


def _common(result: dict) -> None:
    """Every result is a JSON-serializable dict with the contract keys."""
    json.loads(json.dumps(result))
    for key in ("warnings", "assumptions", "not_included"):
        assert isinstance(result[key], list) and all(isinstance(x, str) for x in result[key])
    assert result["assumptions"] and result["not_included"]
    assert isinstance(result["method"], dict) and result["method"]["module"] == "packages/energy/steam_distribution.py"
    assert result["evidence"]["verified"] is True and result["evidence"]["reasons"] == []


# ---------------------------------------------------------------------------------------------------------- saturation
def test_saturation_temperature():
    assert SDN.saturation_T_K(ATM) == pytest.approx(373.124, abs=0.002)        # IAPWS-95 normal boiling point
    assert SDN.saturation_T_K(psig(150)) == pytest.approx(F(365.9), abs=0.1)   # 150 psig steam: 366 °F
    with pytest.raises(ValueError, match="presión absoluta"):
        SDN.saturation_T_K(5.0)
    with pytest.raises(ValueError, match="falta pressure_Pa"):
        SDN.saturation_T_K(None)


# -------------------------------------------------------------------------------------------------- leaks: DOE tables
def _tip1_deviations():
    engine, napier = {}, {}
    for d_in, row in TIP1.items():
        for p, published in zip(TIP1_PSIG, row):
            if published is None:
                continue
            r = SDN.steam_leak_loss(psig(p), d_in * INCH, TIP1_CD, ATM)
            assert r["choked"] and r["evidence"]["verified"] and r["napier"]["applicable"]
            engine[(d_in, p)] = lb_h(r["mass_flow_kg_s"]) / published - 1.0
            napier[(d_in, p)] = lb_h(r["napier"]["mass_flow_kg_s"]) / published - 1.0
    return engine, napier


def test_doe_tip_sheet_1_whole_table():
    """23 values, Cd = 0.72 and atmospheric discharge as the sheet states: the engine stays within -3.8 % / +2.9 %."""
    engine, napier = _tip1_deviations()
    assert len(engine) == 23
    assert -0.039 < min(engine.values()) < -0.036 and 0.027 < max(engine.values()) < 0.030
    # the deviation is a pressure trend, not scatter: the table is linear in pressure and real steam is not
    by_pressure = {p: [v for (_, q), v in engine.items() if q == p] for p in TIP1_PSIG}
    assert all(v > 0.02 for v in by_pressure[15]) and all(v < -0.035 for v in by_pressure[300])
    # 0.72 * Napier is 0.8 % to 2.6 % BELOW the sheet in every cell: the sheet's formula is not exactly Napier
    assert -0.026 < min(napier.values()) and max(napier.values()) < -0.007


def test_doe_tip_sheet_1_table_is_linear_in_absolute_pressure():
    constants = [published / (d_in ** 2 * (p + ATM_PSI))
                 for d_in, row in TIP1.items() for p, published in zip(TIP1_PSIG, row) if published is not None]
    mean = sum(constants) / len(constants)
    assert mean == pytest.approx(29.455, abs=0.01)
    assert max(abs(c / mean - 1.0) for c in constants) < 0.014                  # rounding of the smallest entries
    napier_072 = TIP1_CD * 3600.0 / 70.0 * math.pi / 4.0                         # lb/h per in^2 of D^2 and psia
    assert mean / napier_072 == pytest.approx(1.0128, abs=0.001)


def test_doe_tip_sheet_1_example():
    """The example of the sheet: 1/8 in orifice on a 150 psig line, 75.8 lb/h in the table."""
    r = SDN.steam_leak_loss(psig(150), INCH / 8, TIP1_CD, ATM)
    assert lb_h(r["mass_flow_kg_s"]) == pytest.approx(75.8, rel=0.025)
    assert lb_h(r["mass_flow_kg_s"]) == pytest.approx(73.96, abs=0.02)
    _common(r)


def _table6_deviations():
    out = {}
    for d_in, row in T6.items():
        for p, published in zip(T6_PSIG, row):
            superheat = F(500.0) - SDN.saturation_T_K(psig(p))
            r = SDN.steam_leak_loss(psig(p), d_in * INCH, T6_CD, ATM, superheat_K=superheat)
            assert r["choked"] and r["evidence"]["verified"] and not r["napier"]["applicable"]
            out[(d_in, p)] = lb_h(r["mass_flow_kg_s"]) / published - 1.0
    return out


def test_ornl_table_6_superheated_leaks():
    """56 values at 500 °F. With Cd = 0.60: 48 values within -0.8 % / +0.4 %; the source's 300 psig column is 4 % off."""
    dev = _table6_deviations()
    assert len(dev) == 56
    regular = [v for (_, p), v in dev.items() if p != 300]
    odd = [v for (_, p), v in dev.items() if p == 300]
    assert len(regular) == 48 and -0.008 < min(regular) and max(regular) < 0.004
    assert len(odd) == 8 and -0.044 < min(odd) and max(odd) < -0.039


def test_ornl_table_6_300_psig_column_breaks_the_trend_of_the_source():
    """Not an engine result: per unit of D^2 the published 300 psig column sits 4 % above its own neighbours."""
    per_area = [T6[1.0][i] for i in range(len(T6_PSIG))]
    interpolated = 0.5 * (per_area[T6_PSIG.index(250)] + per_area[T6_PSIG.index(350)])
    assert per_area[T6_PSIG.index(300)] / interpolated == pytest.approx(1.041, abs=0.002)
    assert per_area[T6_PSIG.index(200)] / (0.5 * (per_area[2] + per_area[4])) == pytest.approx(1.0, abs=0.002)


def test_ornl_one_inch_orifice_note():
    """«A 1-in.-diam orifice will pass 1000 lbm/h of 25-psig steam» (sharp-edged, ~60 %): 989 lb/h with Cd = 0.60."""
    r = SDN.steam_leak_loss(psig(25), INCH, T6_CD, ATM)
    assert lb_h(r["mass_flow_kg_s"]) == pytest.approx(1000.0, rel=0.015)


def test_napier_cross_check_matches_ornl_equation_56():
    r = SDN.steam_leak_loss(psig(100), INCH / 4, 1.0, ATM)
    area_in2 = math.pi / 4.0 * 0.25 ** 2
    assert lb_h(r["napier"]["mass_flow_kg_s"]) == pytest.approx(51.43 * area_in2 * (100 + ATM_PSI), rel=1e-4)
    assert r["napier"]["difference_fraction"] == pytest.approx(r["napier"]["mass_flow_kg_s"] / r["mass_flow_kg_s"] - 1.0)
    # Napier against the real-steam engine, saturated steam to atmosphere: -3.6 % at 15 psig, +3.0 % at 600 psig
    diff = {p: SDN.steam_leak_loss(psig(p), INCH, 1.0, ATM)["napier"]["difference_fraction"] for p in (15, 100, 300, 600)}
    assert diff[15] == pytest.approx(-0.0356, abs=0.001) and diff[100] == pytest.approx(0.0028, abs=0.001)
    assert diff[300] == pytest.approx(0.0251, abs=0.001) and diff[600] == pytest.approx(0.0301, abs=0.001)


# ------------------------------------------------------------------------------------------------- leaks: the physics
def test_choked_flow_does_not_depend_on_back_pressure():
    flows = [SDN.steam_leak_loss(**{**LEAK, "back_pressure_Pa": pb}) for pb in (700.0, 5.0e4, ATM, 3.0e5, 4.5e5)]
    assert all(r["choked"] for r in flows)
    assert all(r["mass_flow_kg_s"] == pytest.approx(flows[0]["mass_flow_kg_s"], rel=1e-9) for r in flows)
    assert flows[0]["critical_pressure_ratio"] == pytest.approx(0.5765, abs=0.0005)   # wet steam: ~0.58, not 0.545
    assert flows[0]["throat"]["vapor_quality"] == pytest.approx(0.963, abs=0.001)
    for r in flows:
        assert r["evidence"]["mach_upstream_side_of_throat"] == pytest.approx(1.0, abs=1e-3)
        assert r["evidence"]["mach_downstream_side_of_throat"] == pytest.approx(1.0, abs=1e-3)
        _common(r)


def test_unchoked_flow_falls_with_back_pressure_and_is_continuous_at_the_critical_ratio():
    choked = SDN.steam_leak_loss(**LEAK)
    critical = choked["critical_pressure_ratio"] * LEAK["steam_pressure_Pa"]
    just_above = SDN.steam_leak_loss(**{**LEAK, "back_pressure_Pa": critical * 1.002})
    assert not just_above["choked"] and just_above["mass_flow_kg_s"] == pytest.approx(choked["mass_flow_kg_s"], rel=1e-4)
    previous = choked["mass_flow_kg_s"]
    for pb in (5.0e5, 6.0e5, 7.0e5, 7.9e5, 7.999e5):
        r = SDN.steam_leak_loss(**{**LEAK, "back_pressure_Pa": pb})
        assert not r["choked"] and r["critical_pressure_ratio"] is None
        assert r["mass_flow_kg_s"] < previous
        assert r["throat"]["pressure_Pa"] == pytest.approx(pb)
        assert r["evidence"]["mach_upstream_side_of_throat"] < 1.0
        assert r["napier"]["applicable"] is False and r["napier"]["mass_flow_kg_s"] is None
        assert any("subcrítico" in w for w in r["warnings"])
        _common(r)
        previous = r["mass_flow_kg_s"]
    assert previous < 0.03 * choked["mass_flow_kg_s"]            # the flow vanishes with the pressure difference


def test_mass_flow_is_proportional_to_area_and_to_the_discharge_coefficient():
    base = SDN.steam_leak_loss(**LEAK)
    double = SDN.steam_leak_loss(**{**LEAK, "hole_diameter_m": 2 * LEAK["hole_diameter_m"]})
    half_cd = SDN.steam_leak_loss(**{**LEAK, "discharge_coefficient": 0.4})
    assert double["mass_flow_kg_s"] == pytest.approx(4.0 * base["mass_flow_kg_s"], rel=1e-12)
    assert half_cd["mass_flow_kg_s"] == pytest.approx(0.5 * base["mass_flow_kg_s"], rel=1e-12)
    assert base["mass_flow_kg_s"] == pytest.approx(base["discharge_coefficient"] * base["hole_area_m2"]
                                                   * base["mass_flux_ideal_kg_m2_s"], rel=1e-12)
    assert base["mass_flux_ideal_kg_m2_s"] == pytest.approx(base["throat"]["density_kg_m3"] * base["throat"]["velocity_m_s"])


def test_enthalpy_and_latent_heat_of_the_lost_steam_are_iapws95():
    r = SDN.steam_leak_loss(**LEAK)
    h_g = CP.PropsSI("H", "P", 8.0e5, "Q", 1, "Water")
    h_f = CP.PropsSI("H", "P", 8.0e5, "Q", 0, "Water")
    assert r["steam_enthalpy_J_kg"] == pytest.approx(h_g) and r["latent_heat_J_kg"] == pytest.approx(h_g - h_f)
    assert r["steam"]["superheat_K"] == 0.0 and r["steam"]["T_K"] == pytest.approx(SDN.saturation_T_K(8.0e5))
    # energy conservation of the expansion, checked with an independent (p, s) flash of the throat
    h_throat = CP.PropsSI("H", "P", r["throat"]["pressure_Pa"], "S", r["steam"]["entropy_J_kg_K"], "Water")
    assert h_g - h_throat == pytest.approx(0.5 * r["throat"]["velocity_m_s"] ** 2, rel=1e-9)


@pytest.mark.parametrize("pressure_Pa,superheat_K", [(2.0e5, 300.0), (1.0e5, 400.0), (5.0e4, 300.0), (2.0e5, 150.0)])
def test_highly_superheated_low_pressure_steam_recovers_the_perfect_gas_nozzle(pressure_Pa, superheat_K):
    T0 = SDN.saturation_T_K(pressure_Pa) + superheat_K
    k = CP.PropsSI("Cpmass", "P", pressure_Pa, "T", T0, "Water") / CP.PropsSI("Cvmass", "P", pressure_Pa, "T", T0, "Water")
    rho = CP.PropsSI("D", "P", pressure_Pa, "T", T0, "Water")
    perfect_gas = math.sqrt(k * rho * pressure_Pa) * (2.0 / (k + 1.0)) ** ((k + 1.0) / (2.0 * (k - 1.0)))
    r = SDN.steam_leak_loss(pressure_Pa, 0.01, 1.0, 0.1 * pressure_Pa, superheat_K=superheat_K)
    assert r["choked"] and r["throat"]["vapor_quality"] is None
    assert r["mass_flux_ideal_kg_m2_s"] == pytest.approx(perfect_gas, rel=0.002)
    assert r["critical_pressure_ratio"] == pytest.approx((2.0 / (k + 1.0)) ** (k / (k - 1.0)), abs=0.003)
    _common(r)


def test_superheat_sweep_including_the_throat_on_the_saturation_line():
    """Between ~35 and ~39 K of superheat at 8 bar the isentrope meets the saturation line at the throat (sonic kink)."""
    flows = {}
    for sh in (0.0, 5.0, 20.0, 30.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 80.0, 300.0):
        r = SDN.steam_leak_loss(**LEAK, superheat_K=sh)
        assert r["choked"] and r["evidence"]["verified"], sh
        assert r["evidence"]["mach_upstream_side_of_throat"] <= 1.0 + 5e-3
        assert r["evidence"]["mach_downstream_side_of_throat"] >= 1.0 - 5e-3
        assert r["steam"]["T_K"] == pytest.approx(r["steam"]["saturation_T_K"] + sh)
        flows[sh] = r["mass_flow_kg_s"]
        json.dumps(r)
    kink = SDN.steam_leak_loss(**LEAK, superheat_K=36.0)
    mach = kink["evidence"]
    assert mach["mach_upstream_side_of_throat"] < 0.97 and mach["mach_downstream_side_of_throat"] > 1.01
    assert flows[300.0] < flows[80.0] < flows[40.0] < flows[0.0]              # hotter steam is lighter: less mass
    assert max(flows.values()) / min(v for k, v in flows.items() if k <= 40.0) < 1.01


@pytest.mark.parametrize("pressure_Pa", [1200.0, 1.2e5, 2.0e6, 1.0e7, 2.0e7])
def test_whole_saturation_range_is_verified(pressure_Pa):
    r = SDN.steam_leak_loss(pressure_Pa, 0.005, 0.8, max(700.0, min(ATM, 0.5 * pressure_Pa)))
    assert r["choked"] and 0.57 < r["critical_pressure_ratio"] < 0.63
    _common(r)


@pytest.mark.parametrize("override,fragment", [
    ({"steam_pressure_Pa": None}, "falta steam_pressure_Pa"),
    ({"hole_diameter_m": None}, "falta hole_diameter_m"),
    ({"discharge_coefficient": None}, "falta discharge_coefficient"),
    ({"back_pressure_Pa": None}, "falta back_pressure_Pa"),
    ({"discharge_coefficient": 1.2}, "entre 0 y 1"),
    ({"discharge_coefficient": 0.0}, "mayor que cero"),
    ({"hole_diameter_m": -0.001}, "mayor que cero"),
    ({"hole_diameter_m": "3 mm"}, "debe ser un número"),
    ({"back_pressure_Pa": 9.0e5}, "debe ser menor que steam_pressure_Pa"),
    ({"back_pressure_Pa": 100.0}, "punto triple"),
    ({"steam_pressure_Pa": 8.0}, "presión absoluta"),
    ({"steam_pressure_Pa": 2.3e7}, "intervalo de saturación"),
    ({"superheat_K": -5.0}, "superheat_K"),
])
def test_leak_inputs_are_required_and_checked_in_spanish(override, fragment):
    with pytest.raises(ValueError, match=fragment):
        SDN.steam_leak_loss(**{**LEAK, **override})


def test_leak_has_no_default_for_any_plant_datum():
    with pytest.raises(ValueError, match="falta steam_pressure_Pa: presión absoluta del vapor"):
        SDN.steam_leak_loss()
    with pytest.raises(ValueError, match="para saber si el flujo es crítico"):
        SDN.steam_leak_loss(8.0e5, 0.003, 0.7)


# -------------------------------------------------------------------------------------------------------------- traps
TRAPS = [{"label": "termodinámicas 1/2", "count": 3, "orifice_diameter_m": INCH / 8, "steam_pressure_Pa": 1.1e6,
          "back_pressure_Pa": 2.0e5},
         {"count": 1, "orifice_diameter_m": 0.006, "steam_pressure_Pa": 4.0e5, "back_pressure_Pa": 3.0e5}]


def test_failed_traps_are_leaks_times_the_surveyed_count():
    r = SDN.failed_trap_loss(TRAPS, 0.72)
    _common(r)
    single = [SDN.steam_leak_loss(g["steam_pressure_Pa"], g["orifice_diameter_m"], 0.72, g["back_pressure_Pa"])
              for g in TRAPS]
    for group, leak, spec in zip(r["groups"], single, TRAPS):
        assert group["mass_flow_per_trap_kg_s"] == pytest.approx(leak["mass_flow_kg_s"], rel=1e-12)
        assert group["mass_flow_kg_s"] == pytest.approx(spec["count"] * leak["mass_flow_kg_s"], rel=1e-12)
        assert group["count"] == spec["count"] and group["label"] == spec.get("label")
    assert r["groups"][0]["choked"] is True and r["groups"][1]["choked"] is False     # 3/4 bar: return header too high
    assert r["total_count"] == 4
    assert r["total_mass_flow_kg_s"] == pytest.approx(sum(g["mass_flow_kg_s"] for g in r["groups"]), rel=1e-12)
    enthalpy_flow = sum(g["mass_flow_kg_s"] * g["steam_enthalpy_J_kg"] for g in r["groups"])
    assert r["total_enthalpy_flow_W"] == pytest.approx(enthalpy_flow)
    assert any("groups[1]" in w and "subcrítico" in w for w in r["warnings"])
    assert any("censo" in a for a in r["assumptions"])


def test_failed_traps_reproduce_the_doe_tip_sheet_1_example():
    """One trap, 1/8 in orifice, 150 psig, to atmosphere, Cd 0.72: the sheet reads 75.8 lb/h; ten of them, ten times."""
    one = SDN.failed_trap_loss([{"count": 1, "orifice_diameter_m": INCH / 8, "steam_pressure_Pa": psig(150),
                                 "back_pressure_Pa": ATM}], TIP1_CD)
    ten = SDN.failed_trap_loss([{"count": 10, "orifice_diameter_m": INCH / 8, "steam_pressure_Pa": psig(150),
                                 "back_pressure_Pa": ATM}], TIP1_CD)
    assert lb_h(one["total_mass_flow_kg_s"]) == pytest.approx(75.8, rel=0.025)
    assert ten["total_mass_flow_kg_s"] == pytest.approx(10 * one["total_mass_flow_kg_s"], rel=1e-12)
    assert one["groups"][0]["napier_difference_fraction"] == pytest.approx(0.012, abs=0.001)


@pytest.mark.parametrize("groups,cd,fragment", [
    (None, 0.72, "falta groups"),
    ([], 0.72, "falta groups"),
    (TRAPS, None, "falta discharge_coefficient"),
    ([{k: v for k, v in TRAPS[0].items() if k != "count"}], 0.72, r"falta groups\[0\].count"),
    ([{**TRAPS[0], "count": 0}], 0.72, "entero mayor que cero"),
    ([{**TRAPS[0], "count": 2.5}], 0.72, "entero mayor que cero"),
    ([{**TRAPS[0], "count": True}], 0.72, "entero mayor que cero"),
    ([{k: v for k, v in TRAPS[0].items() if k != "orifice_diameter_m"}], 0.72, r"falta groups\[0\].orifice_diameter_m"),
    ([{k: v for k, v in TRAPS[0].items() if k != "steam_pressure_Pa"}], 0.72, r"falta groups\[0\].steam_pressure_Pa"),
    ([TRAPS[0], {k: v for k, v in TRAPS[1].items() if k != "back_pressure_Pa"}], 0.72,
     r"falta groups\[1\].back_pressure_Pa"),
    ([{**TRAPS[0], "back_pressure_Pa": 2.0e6}], 0.72, r"groups\[0\]: back_pressure_Pa"),
    ([{**TRAPS[0], "failed_fraction": 0.2}], 0.72, "claves no reconocidas"),
    (["trampa"], 0.72, "debe ser un diccionario"),
])
def test_trap_inputs_are_required_and_checked_in_spanish(groups, cd, fragment):
    with pytest.raises(ValueError, match=fragment):
        SDN.failed_trap_loss(groups, cd)


# ------------------------------------------------------------------------------------------ heat loss: published cases
def test_incropera_example_9_4_every_input_stated():
    """Yovanovich's worksheet: bare 763.48 W/m (322.46 convection + 441.02 radiation, Nu 23.094); with 25 mm of insulation
    the surface drops to 35.654 °C and the loss to 52.24 W/m. Measured: +0.41 %, +0.79 %, +0.13 %, -0.42 K and +0.08 %."""
    r = SDN.bare_surface_heat_loss([{"outer_diameter_m": 0.1, "length_m": 1.0, "surface_T_K": 165.0 + 273.15,
                                     "surface_emissivity": 0.85, "insulation_thickness_m": 0.025,
                                     "insulation_conductivity_W_m_K": 0.026, "jacket_emissivity": 0.85}],
                                   23.0 + 273.15, 0.0, ATM)
    _common(r)
    bare, ins = r["segments"][0]["bare"], r["segments"][0]["insulated"]
    assert bare["heat_loss_W_m"] / 763.48 - 1.0 == pytest.approx(0.0041, abs=0.0005)
    assert bare["convection_W_m"] / 322.46 - 1.0 == pytest.approx(0.0079, abs=0.0005)
    assert bare["radiation_W_m"] / 441.02 - 1.0 == pytest.approx(0.0013, abs=0.0005)
    assert bare["Nu"] / 23.094 - 1.0 == pytest.approx(0.011, abs=0.001)
    assert ins["surface_T_K"] - 273.15 - 35.654 == pytest.approx(-0.42, abs=0.02)
    assert ins["heat_loss_W_m"] / 52.24 - 1.0 == pytest.approx(0.0008, abs=0.0005)
    assert r["saving_fraction"] == pytest.approx(1.0 - 52.24 / 763.48, abs=0.001)


def _tip2_deviations(emissivity: float) -> dict:
    out = {}
    for nps, row in TIP2.items():
        for p, published in zip(TIP2_PSIG, row):
            r = SDN.bare_surface_heat_loss([{"outer_diameter_m": OD_IN[nps] * INCH, "length_m": 100 * FOOT,
                                             "steam_pressure_Pa": psig(p), "surface_emissivity": emissivity}],
                                           F(75.0), 0.0, ATM)
            assert r["evidence"]["verified"]
            out[(nps, p)] = r["bare_heat_loss_W"] * 8760 * 3600 / BTU / 1e6 / published - 1.0
    return out


def test_doe_tip_sheet_2_whole_table_is_a_sensitivity_to_the_emissivity_the_sheet_does_not_state():
    """20 values, 75 °F, no wind, 8 760 h (stated). Emissivity NOT stated: 0.8 gives +2.4 % / +16.2 %, 0.7 gives -5.1 % /
    +8.1 %. At any emissivity the deviation grows with the diameter (8 to 12 points from 1 in to 12 in): no single emissivity
    reproduces the table, so this is recorded as a sensitivity and NOT as a validation."""
    e08, e07 = _tip2_deviations(0.8), _tip2_deviations(0.7)
    assert len(e08) == 20
    assert 0.023 < min(e08.values()) < 0.026 and 0.160 < max(e08.values()) < 0.165
    assert -0.052 < min(e07.values()) < -0.049 and 0.079 < max(e07.values()) < 0.083
    assert abs(sum(e07.values()) / 20) < 0.01 and sum(e08.values()) / 20 == pytest.approx(0.088, abs=0.002)
    for dev in (e08, e07):
        for p in TIP2_PSIG:
            column = [dev[(nps, p)] for nps in TIP2]
            assert column == sorted(column)                           # grows with diameter in every pressure column
            assert 0.08 < column[-1] - column[0] < 0.13


def _table7_deviations(emissivity: float) -> dict:
    out = {}
    for nps, row in T7.items():
        for t_f, published in zip(T7_F, row):
            r = SDN.bare_surface_heat_loss([{"outer_diameter_m": OD_IN[nps] * INCH, "length_m": FOOT, "surface_T_K": F(t_f),
                                             "surface_emissivity": emissivity}], F(70.0), 10 * MPH, ATM)
            assert r["evidence"]["verified"] and r["method"]["forced_convection"]
            out[(nps, t_f)] = r["bare_heat_loss_W"] * 3600 / BTU / published - 1.0
    return out


def test_ornl_table_7_wind_is_a_sensitivity_to_the_emissivity_the_guide_does_not_state():
    """78 values, 10 mile/h wind and 70 °F (stated). Emissivity NOT stated: 0.8 gives -3.4 % / +9.8 % (mean +3.5 %), 0.7
    gives -7.2 % / +3.0 % (mean -3.0 %). Sensitivity, not validation."""
    e08, e07 = _table7_deviations(0.8), _table7_deviations(0.7)
    assert len(e08) == 78
    assert -0.035 < min(e08.values()) < -0.033 and 0.096 < max(e08.values()) < 0.099
    assert sum(e08.values()) / 78 == pytest.approx(0.035, abs=0.002)
    assert -0.074 < min(e07.values()) < -0.071 and 0.029 < max(e07.values()) < 0.032
    # the worked example of the guide: 100 ft of 8 in pipe at 600 °F, 723,700 Btu/h
    assert T7[8][T7_F.index(600)] * 100 == 723700
    assert e08[(8, 600)] == pytest.approx(0.006, abs=0.002)


def test_correlations_of_ht_are_the_published_equations():
    """Churchill-Chu (1975) and Churchill-Bernstein (1977) as printed in the papers, against the library functions."""
    Pr, Gr, Re = 0.70, 7.5e6, 3.0e4
    Ra = Pr * Gr
    chu = (0.6 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
    bernstein = (0.3 + 0.62 * Re ** 0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25
                 * (1 + (Re / 282000) ** (5 / 8)) ** 0.8)
    assert SDN.Nu_horizontal_cylinder_Churchill_Chu(Pr, Gr) == pytest.approx(chu, rel=1e-12)
    assert SDN.Nu_cylinder_Churchill_Bernstein(Re, Pr) == pytest.approx(bernstein, rel=1e-12)


# -------------------------------------------------------------------------------------------- heat loss: the physics
def test_bare_loss_exceeds_insulated_loss_and_the_layer_balance_closes():
    second = {**PIPE, "label": "4 in", "outer_diameter_m": 0.1143, "length_m": 12.0, "surface_T_K": 440.0}
    for wind in (0.0, 1.0, 5.0):
        r = SDN.bare_surface_heat_loss([PIPE, second], **{**AMBIENT, "wind_speed_m_s": wind})
        _common(r)
        assert r["evidence"]["max_insulation_energy_balance_residual"] < 1e-10
        assert r["evidence"]["bare_exceeds_insulated"] is True
        for seg in r["segments"]:
            bare, ins = seg["bare"], seg["insulated"]
            assert bare["heat_loss_W"] > ins["heat_loss_W"] > 0
            assert r["ambient_T_K"] < ins["surface_T_K"] < seg["surface_T_K"]
            assert ins["conduction_W_m"] == pytest.approx(ins["convection_W_m"] + ins["radiation_W_m"], rel=1e-10)
            conduction = 2 * math.pi * 0.045 * (seg["surface_T_K"] - ins["surface_T_K"]) / math.log(
                ins["outer_diameter_m"] / seg["outer_diameter_m"])
            assert ins["heat_loss_W_m"] == pytest.approx(conduction, rel=1e-10)
            assert bare["heat_loss_W"] == pytest.approx(bare["heat_loss_W_m"] * seg["length_m"])
            assert bare["heat_loss_W_m"] == pytest.approx(bare["convection_W_m"] + bare["radiation_W_m"])
            assert seg["saving_W"] == pytest.approx(bare["heat_loss_W"] - ins["heat_loss_W"])
            assert 0.85 < seg["saving_fraction"] < 0.99
        assert r["bare_heat_loss_W"] == pytest.approx(sum(s["bare"]["heat_loss_W"] for s in r["segments"]))
        assert r["insulated_heat_loss_W"] == pytest.approx(sum(s["insulated"]["heat_loss_W"] for s in r["segments"]))
        assert r["saving_W"] == pytest.approx(r["bare_heat_loss_W"] - r["insulated_heat_loss_W"])
        assert r["total_length_m"] == pytest.approx(42.0)
    assert r["segments"][0]["surface_T_K"] == pytest.approx(SDN.saturation_T_K(1.1e6))    # from the steam pressure
    assert r["segments"][1]["surface_T_K"] == 440.0                                       # measured: it wins
    assert any("temperatura de saturación" in a and "segments[0]" in a for a in r["assumptions"])


def test_equivalent_steam_uses_the_latent_heat_at_the_line_pressure():
    r = SDN.bare_surface_heat_loss([PIPE], **AMBIENT)
    latent = CP.PropsSI("H", "P", 1.1e6, "Q", 1, "Water") - CP.PropsSI("H", "P", 1.1e6, "Q", 0, "Water")
    seg = r["segments"][0]
    assert seg["latent_heat_J_kg"] == pytest.approx(latent)
    assert r["steam_condensed_bare_kg_s"] == pytest.approx(r["bare_heat_loss_W"] / latent)
    assert r["steam_condensed_insulated_kg_s"] == pytest.approx(r["insulated_heat_loss_W"] / latent)
    assert r["steam_saved_kg_s"] == pytest.approx(r["saving_W"] / latent)
    no_pressure = {k: v for k, v in PIPE.items() if k != "steam_pressure_Pa"} | {"surface_T_K": 450.0}
    r2 = SDN.bare_surface_heat_loss([PIPE, no_pressure], **AMBIENT)
    assert r2["steam_condensed_bare_kg_s"] is None and r2["segments"][1]["steam_condensed_bare_kg_s"] is None
    assert r2["segments"][0]["steam_saved_kg_s"] == pytest.approx(seg["steam_saved_kg_s"])
    assert any("steam_pressure_Pa" in w for w in r2["warnings"])
    json.dumps(r2)


def test_heat_loss_goes_to_zero_as_the_surface_approaches_ambient():
    previous = None
    for dT in (50.0, 5.0, 0.5, 0.05, 0.0):
        seg = {"outer_diameter_m": 0.1, "length_m": 1.0, "surface_T_K": 300.0 + dT, "surface_emissivity": 0.9,
               "insulation_thickness_m": 0.03, "insulation_conductivity_W_m_K": 0.04, "jacket_emissivity": 0.2}
        r = SDN.bare_surface_heat_loss([seg], 300.0, 0.0, ATM)
        _common(r)
        assert r["bare_heat_loss_W"] >= r["insulated_heat_loss_W"] >= 0.0
        if previous is not None:
            assert r["bare_heat_loss_W"] < previous
        previous = r["bare_heat_loss_W"]
    assert r["bare_heat_loss_W"] == 0.0 and r["insulated_heat_loss_W"] == 0.0


def test_wind_adds_loss_and_is_continuous_with_natural_convection():
    bare_only = {k: v for k, v in PIPE.items() if not k.startswith(("insulation", "jacket"))}
    losses = [SDN.bare_surface_heat_loss([bare_only], **{**AMBIENT, "wind_speed_m_s": w})
              for w in (0.0, 1e-3, 0.5, 2.0, 8.0)]
    values = [r["bare_heat_loss_W"] for r in losses]
    assert values == sorted(values)
    assert values[1] == pytest.approx(values[0], rel=1e-4)                  # no jump when the wind starts
    assert values[-1] > 2.0 * values[0]
    assert losses[0]["method"]["forced_convection"] is None and losses[0]["segments"][0]["bare"]["Nu_forced"] is None
    assert "Churchill-Bernstein" in losses[-1]["method"]["forced_convection"]
    assert "Churchill-Chu" in losses[0]["method"]["natural_convection"]
    b = losses[-1]["segments"][0]["bare"]
    assert b["Nu"] == pytest.approx((b["Nu_forced"] ** 4 + b["Nu_natural"] ** 4) ** 0.25)
    assert losses[0]["insulated_heat_loss_W"] is None and losses[0]["saving_W"] is None
    assert losses[0]["segments"][0]["insulated"] is None and losses[0]["evidence"]["bare_exceeds_insulated"] is None
    for r in losses:
        _common(r)


def test_altitude_and_emissivity_move_the_loss_the_right_way():
    bare_only = {k: v for k, v in PIPE.items() if not k.startswith(("insulation", "jacket"))}
    sea = SDN.bare_surface_heat_loss([bare_only], 298.15, 0.0, ATM)
    high = SDN.bare_surface_heat_loss([bare_only], 298.15, 0.0, 78000.0)        # Mexico City plateau
    sb, hb = sea["segments"][0]["bare"], high["segments"][0]["bare"]
    assert hb["convection_W_m"] < sb["convection_W_m"] and hb["radiation_W_m"] == pytest.approx(sb["radiation_W_m"])
    assert 0.85 < hb["convection_W_m"] / sb["convection_W_m"] < 0.92
    shiny = SDN.bare_surface_heat_loss([{**bare_only, "surface_emissivity": 0.4}], 298.15, 0.0, ATM)
    assert shiny["segments"][0]["bare"]["radiation_W_m"] == pytest.approx(0.5 * sb["radiation_W_m"])
    assert shiny["segments"][0]["bare"]["convection_W_m"] == pytest.approx(sb["convection_W_m"])
    sigma = 5.670374419e-8
    expected = 0.8 * sigma * math.pi * 0.0603 * (SDN.saturation_T_K(1.1e6) ** 4 - 298.15 ** 4)
    assert sb["radiation_W_m"] == pytest.approx(expected, rel=1e-9)


def test_partial_insulation_data_leaves_the_insulated_totals_empty():
    bare_only = {k: v for k, v in PIPE.items() if not k.startswith(("insulation", "jacket"))}
    r = SDN.bare_surface_heat_loss([PIPE, bare_only], **AMBIENT)
    _common(r)
    assert r["insulated_heat_loss_W"] is None and r["saving_W"] is None and r["steam_saved_kg_s"] is None
    assert r["segments"][0]["saving_W"] > 0 and r["segments"][1]["insulated"] is None
    assert any("solo algunos tramos" in w for w in r["warnings"])


def test_insulation_that_increases_the_loss_is_reported_not_hidden():
    """A thin, poor layer with a black jacket on a polished small tube: below the critical radius the loss goes up."""
    seg = {"outer_diameter_m": 0.006, "length_m": 1.0, "surface_T_K": 400.0, "surface_emissivity": 0.05,
           "insulation_thickness_m": 0.002, "insulation_conductivity_W_m_K": 0.5, "jacket_emissivity": 0.95}
    r = SDN.bare_surface_heat_loss([seg], 300.0, 0.0, ATM)
    assert r["saving_W"] < 0 and r["evidence"]["bare_exceeds_insulated"] is False
    assert any("AUMENTA" in w for w in r["warnings"])
    assert r["evidence"]["max_insulation_energy_balance_residual"] < 1e-10
    json.dumps(r)


@pytest.mark.parametrize("segments,overrides,fragment", [
    (None, {}, "falta segments"),
    ([], {}, "falta segments"),
    ([PIPE], {"ambient_T_K": None}, "falta ambient_T_K"),
    ([PIPE], {"wind_speed_m_s": None}, "falta wind_speed_m_s"),
    ([PIPE], {"ambient_pressure_Pa": None}, "falta ambient_pressure_Pa: presión barométrica local"),
    ([PIPE], {"wind_speed_m_s": -1.0}, "mayor o igual a cero"),
    ([{k: v for k, v in PIPE.items() if k != "outer_diameter_m"}], {}, r"falta segments\[0\].outer_diameter_m"),
    ([{k: v for k, v in PIPE.items() if k != "length_m"}], {}, r"falta segments\[0\].length_m"),
    ([{k: v for k, v in PIPE.items() if k != "surface_emissivity"}], {}, r"falta segments\[0\].surface_emissivity"),
    ([{k: v for k, v in PIPE.items() if k != "steam_pressure_Pa"}], {}, r"falta segments\[0\].surface_T_K o"),
    ([{k: v for k, v in PIPE.items() if k != "jacket_emissivity"}], {}, r"falta segments\[0\].jacket_emissivity"),
    ([{k: v for k, v in PIPE.items() if k != "insulation_conductivity_W_m_K"}], {},
     r"falta segments\[0\].insulation_conductivity_W_m_K"),
    ([{k: v for k, v in PIPE.items() if k != "insulation_thickness_m"}], {}, r"falta segments\[0\].insulation_thickness_m"),
    ([{**PIPE, "surface_emissivity": 1.3}], {}, "entre 0 y 1"),
    ([{**PIPE, "surface_T_K": 280.0}], {}, "más fría que el ambiente"),
    ([{**PIPE, "emissivity": 0.8}], {}, "claves no reconocidas"),
    ([PIPE, "tramo"], {}, r"segments\[1\] debe ser un diccionario"),
])
def test_heat_loss_inputs_are_required_and_checked_in_spanish(segments, overrides, fragment):
    with pytest.raises(ValueError, match=fragment):
        SDN.bare_surface_heat_loss(segments, **{**AMBIENT, **overrides})


def test_every_user_facing_string_is_spanish_and_serializable():
    results = [SDN.steam_leak_loss(**LEAK), SDN.steam_leak_loss(**LEAK, superheat_K=60.0),
               SDN.steam_leak_loss(**{**LEAK, "back_pressure_Pa": 7.0e5}), SDN.failed_trap_loss(TRAPS, 0.72),
               SDN.bare_surface_heat_loss([PIPE], **AMBIENT),
               SDN.bare_surface_heat_loss([PIPE], **{**AMBIENT, "wind_speed_m_s": 3.0})]
    for r in results:
        text = json.dumps(r, ensure_ascii=False)
        assert "NaN" not in text and "Infinity" not in text
        _common(r)
        for sentence in r["assumptions"] + r["not_included"] + r["warnings"]:
            assert not any(word in sentence.lower().split() for word in ("the", "steam", "with", "and", "loss"))
    assert any("vapor saturado seco" in a for a in results[0]["assumptions"])
    assert any("sobrecalentado 60.0 K" in a for a in results[1]["assumptions"])
    assert any("0.8 dado por quien llama" in a for a in results[0]["assumptions"])
