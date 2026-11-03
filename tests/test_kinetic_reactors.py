"""Kinetic reactors (bloque 7): DWSIM CSTR/PFR through the tools layer and Cantera reactor networks both reproduce
the first-order analytic conversions X_PFR = 1−e^(−kτ), X_CSTR = kτ/(1+kτ); the DWSIM adiabatic conversion reactor
overestimates the methane flame temperature vs GRI-Mech equilibrium (no dissociation). Evidence: notebooks/03_cantera_reactores.md."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools as T  # noqa: E402

R = 8.314462618
K0, EA, TEMP, P, F, V = 1.0e8, 100_000.0, 600.0, 1.0e5, 10.0, 10.0
k = K0 * math.exp(-EA / (R * TEMP))
tau = V / (F * R * TEMP / P)
X_PFR, X_CSTR = 1 - math.exp(-k * tau), k * tau / (1 + k * tau)


def _dwsim(kind):
    fid = T.new_flowsheet(kind, ["N-butane", "Isobutane"], "PengRobinson")["flowsheet_id"]
    T.define_kinetic_reaction(fid, "ISO", {"N-butane": -1, "Isobutane": 1}, "N-butane", {"N-butane": 1}, K0, EA)
    T.add_material_stream(fid, "feed", temperature_K=TEMP, pressure_Pa=P, molar_flow_mol_s=F, mole_fractions={"N-butane": 1.0})
    T.add_material_stream(fid, "prod"); T.add_energy_stream(fid, "q")
    T.add_unit(fid, kind, "R", {"volume_m3": V, "reaction_set": "KIN"})
    T.connect(fid, "R", feeds=["feed"], products=["prod"], energy_feeds=["q"])
    out = T.solve(fid)
    X = 1 - T.read_stream(fid, "prod")["mole_fractions"]["N-butane"]
    T.close_flowsheet(fid)
    return out, X


@pytest.mark.parametrize("kind,expected", [("pfr", X_PFR), ("cstr", X_CSTR)])
def test_dwsim_kinetic_reactor_matches_first_order_analytic(kind, expected):
    out, X = _dwsim(kind)
    assert out["ok"] and out["units"][0]["kind"] == f"RCT_{kind.upper()}"
    assert X == pytest.approx(expected, rel=1e-3)          # PR density vs ideal gas: +0.014 % / +0.028 %
    assert out["units"][0]["duty_kW"] < 0                  # isomerisation to isobutane is exothermic


def test_cantera_network_matches_analytic():
    ct = pytest.importorskip("cantera")
    yaml = f"""
units: {{length: m, quantity: mol, activation-energy: J/mol}}
phases: [{{name: gas, thermo: ideal-gas, species: [A, B], kinetics: gas, reactions: all, state: {{T: {TEMP}, P: {P}}}}}]
species:
- {{name: A, composition: {{C: 4, H: 10}}, thermo: {{model: constant-cp, T0: 298.15 K, h0: 0 kJ/mol, s0: 0 J/mol/K, cp0: 120 J/mol/K}}}}
- {{name: B, composition: {{C: 4, H: 10}}, thermo: {{model: constant-cp, T0: 298.15 K, h0: -8 kJ/mol, s0: 0 J/mol/K, cp0: 120 J/mol/K}}}}
reactions: [{{equation: A => B, rate-constant: {{A: {K0}, b: 0, Ea: {EA}}}}}]
"""
    gas = ct.Solution(yaml=yaml); gas.TPX = TEMP, P, "A:1"
    batch = ct.IdealGasConstPressureReactor(gas, energy="off", clone=False)
    ct.ReactorNet([batch]).advance(tau)
    assert 1 - batch.phase["A"].X[0] == pytest.approx(X_PFR, rel=1e-4)
    gas2 = ct.Solution(yaml=yaml); gas2.TPX = TEMP, P, "A:1"
    inlet, r, out = ct.Reservoir(gas2, clone=False), ct.IdealGasReactor(gas2, energy="off", clone=False), ct.Reservoir(gas2, clone=False)
    r.volume = V
    mfc = ct.MassFlowController(inlet, r, mdot=r.mass / tau)
    ct.PressureController(r, out, primary=mfc, K=0.01)
    ct.ReactorNet([r]).advance_to_steady_state()
    assert 1 - r.phase["A"].X[0] == pytest.approx(X_CSTR, rel=1e-4)


def test_conversion_reactor_overestimates_flame_temperature():
    ct = pytest.importorskip("cantera")
    g = ct.Solution("gri30.yaml"); g.TPX = 300.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"; g.equilibrate("HP")
    fid = T.new_flowsheet("comb", ["Methane", "Oxygen", "Nitrogen", "Carbon dioxide", "Water"], "PengRobinson")["flowsheet_id"]
    T.define_conversion_reaction(fid, "COMB", {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}, "Methane", 100, "Comb")
    T.add_material_stream(fid, "mix", temperature_K=300.0, pressure_Pa=ct.one_atm, molar_flow_mol_s=10.52,
                          mole_fractions={"Methane": 1 / 10.52, "Oxygen": 2 / 10.52, "Nitrogen": 7.52 / 10.52})
    for n in ("flue", "liq"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "conversion_reactor", "R-1", {"reaction_set": "Comb", "mode": "adiabatic"})
    T.connect(fid, "R-1", feeds=["mix"], products=["flue", "liq"], energy_feeds=["Q"])
    assert T.solve(fid)["ok"]
    T_dw = T.read_stream(fid, "flue")["T_K"]
    T.close_flowsheet(fid)
    assert g.T == pytest.approx(2226, abs=5)
    assert 80 < T_dw - g.T < 160                            # ~+116 K: no dissociation in the conversion reactor
