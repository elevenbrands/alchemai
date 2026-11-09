"""Cantera tools (bloque M2): adiabatic_flame and kinetic_reactor_network reproduce notebook 03 (analytic first order,
GRI-Mech 3.0 flame, WSR blow-out, ignition delay) and the evidence turns the seal off when it should."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_kinetics as TK  # noqa: E402
from packages.kinetics import cantera_engine as CE  # noqa: E402

R = 8.314462618
K0, EA, TEMP, P, F, V = 1.0e8, 100_000.0, 600.0, 1.0e5, 10.0, 10.0
k = K0 * math.exp(-EA / (R * TEMP))
TAU = V / (F * R * TEMP / P)
SPECIES = [{"name": "A", "composition": {"C": 4, "H": 10}}, {"name": "B", "composition": {"C": 4, "H": 10}}]
REACTIONS = [{"equation": "A => B", "A": K0, "Ea_J_mol": EA}]
AIR_CH4 = "CH4:1, O2:2, N2:7.52"


def test_custom_first_order_matches_analytic_pfr_and_cstr():
    pfr = TK.kinetic_reactor_network("pfr", "A:1", TEMP, P, energy="isothermal", end_time_s=TAU,
                                     custom_species=SPECIES, custom_reactions=REACTIONS)
    wsr = TK.kinetic_reactor_network("wsr", "A:1", TEMP, P, energy="isothermal", residence_times_s=[TAU],
                                     initial_state="feed", custom_species=SPECIES, custom_reactions=REACTIONS)
    assert pfr["final"]["conversion"]["A"] == pytest.approx(1 - math.exp(-k * TAU), abs=1e-6)       # 0.98071
    assert wsr["sweep"][0]["conversion"]["A"] == pytest.approx(k * TAU / (1 + k * TAU), abs=1e-6)  # 0.79791
    assert pfr["evidence"]["verified"] and wsr["evidence"]["verified"]
    assert len(pfr["profile"]["t_s"]) == 60 and pfr["profile"]["X_A"][-1] == pytest.approx(0.98071, abs=1e-4)


def test_custom_kinetics_refuse_adiabatic_and_reversible():
    with pytest.raises(ValueError, match="isothermal"):
        TK.kinetic_reactor_network("pfr", "A:1", TEMP, P, energy="adiabatic", end_time_s=1.0,
                                   custom_species=SPECIES, custom_reactions=REACTIONS)
    with pytest.raises(ValueError, match="irreversible"):
        CE.custom_mechanism_yaml(SPECIES, [{"equation": "A <=> B", "A": 1.0}])


def test_methane_air_flame_matches_notebook_03():
    r = TK.adiabatic_flame("CH4", 1.0, T_K=300.0, P_Pa=101325.0)
    assert r["T_adiabatic_equilibrium_K"] == pytest.approx(2226, abs=1)                 # notebook 03 B1
    assert r["products_mole_fractions"]["CO"] == pytest.approx(0.0090, abs=2e-4)
    assert r["products_mole_fractions"]["NO"] == pytest.approx(0.0019, abs=1e-4)
    # no dissociation: ideal-gas Cantera 2327 K (DWSIM PR conversion reactor gave 2341 K); always above equilibrium
    assert r["T_adiabatic_complete_combustion_K"] == pytest.approx(2327, abs=2)
    assert r["dissociation_penalty_K"] > 90 and r["evidence"]["verified"]
    assert TK.adiabatic_flame("CH4", 1.5, T_K=300.0)["T_adiabatic_complete_combustion_K"] is None   # rich: no single answer


def test_wsr_blow_out_and_no_like_notebook_03():
    r = TK.kinetic_reactor_network("wsr", AIR_CH4, 300.0, 101325.0, residence_times_s=[0.2e-3, 0.5e-3, 100e-3])
    rows = {round(x["residence_time_s"] * 1e3, 1): x for x in r["sweep"]}
    assert rows[0.2]["state"] == "apagado" and rows[0.5]["state"] == "encendido"
    assert r["blow_out_between_s"] == [0.2e-3, 0.5e-3]
    assert rows[0.5]["T_K"] == pytest.approx(1947, abs=3) and rows[100.0]["T_K"] == pytest.approx(2208, abs=3)
    assert rows[0.5]["mole_fractions"]["NO"] * 1e6 == pytest.approx(97, abs=3)
    assert rows[100.0]["mole_fractions"]["NO"] * 1e6 == pytest.approx(851, abs=10)
    assert r["evidence"]["verified"] and r["evidence"]["steady_state"]


def test_ignition_delay_matches_notebook_03():
    r = TK.kinetic_reactor_network("batch_constant_volume", AIR_CH4, 1200.0, 10e5, end_time_s=0.02)
    assert r["ignition_delay_s"] == pytest.approx(4.520e-3, rel=0.01)                  # notebook 03 C
    assert r["evidence"]["verified"] and r["evidence"]["energy_conserved"]
    cold = TK.kinetic_reactor_network("batch_constant_volume", AIR_CH4, 800.0, 10e5, end_time_s=0.01)
    assert not cold["evidence"]["verified"] and "fuera del rango validado" in cold["evidence"]["reasons"][0]


def test_bad_inputs_give_clear_errors():
    with pytest.raises(ValueError, match="not a Cantera data file"):
        TK.adiabatic_flame("CH4", 1.0, mechanism="../../etc/passwd")
    with pytest.raises(ValueError, match="species not in the mechanism"):
        TK.adiabatic_flame("C8H18", 1.0)
    with pytest.raises(ValueError, match="end_time_s"):
        TK.kinetic_reactor_network("pfr", AIR_CH4, 1200.0, 1e5)
    with pytest.raises(ValueError, match="residence_times_s"):
        TK.kinetic_reactor_network("wsr", AIR_CH4, 300.0, 1e5)


def test_methane_air_flame_speed():
    r = TK.adiabatic_flame("CH4", 1.0, T_K=300.0, P_Pa=101325.0, laminar_flame_speed=True)
    # GRI-Mech 3.0 gives ~0.38 m/s; measured stoichiometric methane-air speeds are ~0.36-0.37 m/s
    assert 0.36 <= r["laminar_flame_speed_m_s"] <= 0.40
    assert r["evidence"]["grid_independent"] and r["evidence"]["verified"]
