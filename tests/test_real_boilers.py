"""The steam diagnostic against PUBLISHED measurements of real boilers, and bare pipe against 3E Plus and ASHRAE (M22).

Sources read on 2026-11-14; every number below was checked against the text or the table of its source (see
notebooks/33_calderas_reales_publicadas.py for the citations). No network. Nothing is tuned to match.
"""
import importlib

import pytest

from packages.dwsim_client import tools_steam as TS
from packages.energy import combustion as CB
from packages.energy import steam_audit as SA
from packages.energy import steam_distribution as SDI

NB = importlib.import_module("notebooks.33_calderas_reales_publicadas")


@pytest.fixture(scope="module")
def cases() -> dict:
    return {"A": NB.case_a(), "B": NB.case_b(), "C": NB.case_c()}


def test_mexican_fire_tube_boiler_measured_with_a_flue_gas_analyzer(cases):
    """UNAM 2021: O2 2.7 %, stack 211 °C, air 22.5 °C, 2 250 m. Analyzer: efficiency 82.26 %, excess air 13.18 %, CO2 10.28 %."""
    a = cases["A"]
    c = a["comparison"]
    assert abs(c["combustion_efficiency_HHV"]["difference_points"]) < 0.15
    lo, hi = c["combustion_efficiency_HHV"]["engine_range"]                     # three reference gases, dry and humid air
    assert 0.8220 < lo and hi < 0.8260 and hi - lo < 0.004
    assert c["combustion_efficiency_LHV_for_reference"] > 0.90                   # the analyzer figure is not a PCI figure
    assert abs(c["excess_air"]["difference_points"]) < 0.3 and abs(c["CO2_dry"]["difference_points"]) < 0.15
    # the source's direct method is reproduced, with its enthalpies and with IAPWS-95
    assert a["direct_method"]["with_source_enthalpies"] == pytest.approx(0.8068, abs=2e-4)
    assert a["direct_method"]["with_iapws"] == pytest.approx(0.8068, abs=1e-3)
    # the full diagnostic, as a plant without gas analysis or setting loss would get it
    d = a["diagnostic"]
    assert d["verified"] is True and d["data_flags"] == [] and set(d["assumed"]) == {"fuel_composition", "setting_loss"}
    assert d["load_fraction"] == pytest.approx(3345.81 / (600 * 15.65), rel=1e-6)
    assert abs(d["difference_points"]) < 2.0                                     # losses vs input-output, on real data
    # what this boiler says about the published setting loss: at 36 % load it is about half of what is assumed
    implied = c["combustion_efficiency_HHV"]["engine"] - d["input_output_efficiency"]
    assert 0.010 < implied < 0.018 and d["setting_loss_used"] == pytest.approx(0.010 / d["load_fraction"])


def test_argentinian_water_tube_boiler_with_its_gas_analysis(cases):
    """Tucumán 2008, Table 6 B: with the supplier's gas analysis and the measured O2, the measured CO2 and lambda follow."""
    b = cases["B"]["comparison"]
    assert abs(b["CO2_dry"]["difference_points"]) < 0.15 and abs(b["lambda"]["difference"]) < 0.01
    assert abs(b["direct_efficiency_LHV"]["difference_points"]) < 2.5
    # losses and input-output disagree by far more than any setting loss: the diagnostic must flag it, not average it
    assert b["losses_minus_direct_points"] > 10.0
    comp = {"N2": 0.0066, "CO2": 0.0222, "CH4": 0.9039, "C2H6": 0.0535, "C3H8": 0.0089, "n-C4H10": 0.0033, "n-C5H12": 0.0016}
    lhv = cases["B"]["heating_values"]["LHV_MJ_Nm3"] * 1e6
    out = TS.steam_system_diagnostic(
        "USD", fuel={"type": "gas", "composition": comp, "T_K": 309.05},
        combustion={"site_pressure_Pa": 101325.0, "air": {"T_K": 309.05, "relative_humidity": 0.6},
                    "flue_O2": {"mole_fraction": 0.018, "basis": "dry"}, "flue_CO_dry_mole_fraction": 0.00015,
                    "stack_T_K": 590.05, "setting_loss_unknown": True},
        boiler={"steam_pressure_Pa": 17.5 * 98066.5, "superheat_K": 590.05 - 478.8, "blowdown_fraction": 0.0},
        water={"feedwater_heating": "not_charged", "feedwater_T_K": 379.95, "makeup_water_T_K": 293.15,
               "condensate_return_fraction": 0.0},
        fuel_cost={"price_per_GJ": 1.0, "basis": "LHV"}, water_cost={"price_per_m3": 1.0},     # prices are not in the source
        operation={"hours_per_year": 1.0, "steam_generated_kg_s": 22200.0 / 3600.0, "annual_basis": "steam",
                   "fuel_consumption": {"energy_GJ_per_year": 2156.0 * lhv / 1e9, "basis": "LHV"}}, today="2026-11-14")
    assert "efficiency_gap" in [f["id"] for f in out["data_review"]["flags"]]
    assert any(w.startswith("REVISAR DATO") and "no cuadra" in w for w in out["warnings"])


def test_worked_fuel_oil_example_loss_by_loss(cases):
    """BEE (India) worked example, not an identified boiler: constant specific heats and 584 kcal/kg in the source."""
    c = cases["C"]
    L = c["losses_percent"]
    assert abs(L["L2 + L3 agua del hidrógeno y humedad del combustible"]["difference_points"]) < 0.05
    assert abs(L["L4 humedad del aire"]["difference_points"]) < 0.02 and abs(L["L6 radiación"]["difference_points"]) < 1e-9
    assert 0.2 < L["L1 gases secos"]["difference_points"] < 0.5                  # real cp of the gases against 0.23 kcal/kg °C
    assert abs(c["efficiency_HHV"]["difference_points"]) < 0.2
    # the source takes 23.0 % of oxygen by mass in air; 21 % by volume with 1 % argon is 23.3 %, hence 1.2 % less air
    assert c["stoichiometric_air_kg_kg"]["engine"] == pytest.approx(13.92 * 0.230 / 0.2327, rel=0.002)


def test_fuel_gas_colder_than_the_nasa_fits_is_handled():
    """Found with the real Mexican boiler: gas at 22.5 °C and a reference gas with pentanes stopped the engine."""
    from packages.energy import fuel_reference as FR
    gulf = FR.reference_gas("GULF-COAST")["composition"]
    kw = dict(pressure_Pa=101325.0, flue_O2={"mole_fraction": 0.03, "basis": "dry"}, stack_T_K=480.0,
              setting_loss_fraction=0.0, setting_loss_basis="HHV", adiabatic_flame=False)
    cold = CB.combustion_efficiency({"type": "gas", "composition": gulf, "T_K": 283.15}, {"T_K": 283.15, "relative_humidity": 0.5}, **kw)
    ref = CB.combustion_efficiency({"type": "gas", "composition": gulf, "T_K": 298.15}, {"T_K": 283.15, "relative_humidity": 0.5}, **kw)
    assert cold["evidence"]["mass_balance_error"] < 1e-9
    assert 0 < ref["efficiency"]["HHV"]["efficiency"] - cold["efficiency"]["HHV"]["efficiency"] < 0.002   # colder fuel, tiny debit
    # continuity at the lower limit of a fit, and a check of the constant-cp continuation on a species fitted below it
    k = "C5H12,i-pentane"
    assert CB.h_molar(k, 296.0) < CB.h_molar(k, 298.15) and CB.h_molar(k, 296.2) == pytest.approx(CB.h_molar(k, 296.1), rel=1e-3)
    sp = CB._nasa()["CH4"].thermo
    true, linear = (sp.h(298.15) - sp.h(258.15)) / 1000.0, sp.cp(298.15) / 1000.0 * 40.0
    assert abs(linear / true - 1.0) < 0.03
    with pytest.raises(ValueError, match="fuera del intervalo"):
        CB.h_molar(k, 298.15 - CB.LOW_T_EXTRAPOLATION_K - 5.0)


# ------------------------------------------------------------------------------------------------------ bare pipe
F = lambda t: (t - 32.0) / 1.8 + 273.15                                         # noqa: E731
BTU_H_FT = 3.412141633 * 0.3048                                                  # W/m -> Btu/h·ft
EMISSIVITY = SA.MODEL_CONSTANTS["surface_emissivity"]["value"]


def _bare(od_in: float, surface_F: float, ambient_F: float) -> float:
    r = SDI.bare_surface_heat_loss([{"outer_diameter_m": od_in * 0.0254, "length_m": 1.0, "surface_emissivity": EMISSIVITY,
                                     "surface_T_K": F(surface_F)}], F(ambient_F), 0.0, 101325.0)
    return r["bare_heat_loss_W"] * BTU_H_FT


def test_bare_pipe_against_3e_plus():
    """NAIMA 3E Plus v4.1 User's Guide (Aug 2012), pp. 1-13/1-14: horizontal NPS 12 steel pipe (emittance 0.8), ambient
    90 °F, no wind; bare surface 794.8 °F -> 14 020 Btu/h·ft and 597.5 °F -> 7 455 Btu/h·ft."""
    assert EMISSIVITY == 0.80
    for surface, published in ((794.8, 14020.0), (597.5, 7455.0)):
        assert _bare(12.75, surface, 90.0) == pytest.approx(published, rel=0.005)


# ASHRAE table as reproduced in the NIA Mechanical Insulation Design Guide, Design Data, Table 6: heat loss from bare
# steel pipe to still air at 80 °F, Btu/h·ft; columns 180, 280, 380, 480, 580 °F. Outside diameters of ASME B36.10.
ASHRAE_T_F = (180, 280, 380, 480, 580)
ASHRAE = {1.315: (82.5, 203.0, 360.0, 560.0, 813.0), 2.375: (141.0, 350.0, 623.0, 974.0, 1420.0),
          4.5: (254.0, 631.0, 1130.0, 1770.0, 2590.0), 6.625: (368.0, 915.0, 1640.0, 2580.0, 3770.0)}


def test_bare_pipe_against_the_ashrae_table():
    dev = [_bare(od, t, 80.0) / published - 1.0 for od, row in ASHRAE.items() for t, published in zip(ASHRAE_T_F, row)]
    assert len(dev) == 20 and -0.025 < min(dev) and max(dev) < 0.008             # the table does not state its emissivity
