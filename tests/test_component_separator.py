"""component_separator: DWSIM's ideal compound splitter exposed through add_unit, for mass balances with fixed recoveries."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client import tools as T  # noqa: E402

COMPS = ["Methane", "Hydrogen", "Toluene"]
Z = {"Methane": 0.3, "Hydrogen": 0.5, "Toluene": 0.2}


def _base(recoveries, connect_energy=True):
    fid = T.new_flowsheet("cs", COMPS, "PengRobinson1978")["flowsheet_id"]
    T.add_material_stream(fid, "F", 350.0, 2e5, molar_flow_mol_s=100.0, mole_fractions=Z)
    T.add_material_stream(fid, "A")
    T.add_material_stream(fid, "B")
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "component_separator", "CS", {"recoveries": recoveries})
    T.connect(fid, "CS", ["F"], ["A", "B"], energy_products=["Q"] if connect_energy else None)
    return fid


def _flows(fid, name):
    s = T.read_stream(fid, name)
    return {c: s["molar_flow_mol_s"] * x for c, x in s["mole_fractions"].items()}


def test_recoveries_are_exact_and_balance_closes():
    fid = _base({"Hydrogen": 0.999, "Methane": 0.001})
    sol = T.solve(fid)
    assert sol["ok"], sol
    a, b = _flows(fid, "A"), _flows(fid, "B")
    assert a["Hydrogen"] == pytest.approx(0.999 * 50.0, rel=1e-9)
    assert a["Methane"] == pytest.approx(0.001 * 30.0, rel=1e-9)
    assert a["Toluene"] == pytest.approx(0.0, abs=1e-12)          # left out -> all to products[1]
    for c in COMPS:
        assert a[c] + b[c] == pytest.approx(100.0 * Z[c], rel=1e-9)


def test_energy_stream_accepted_under_either_name():
    fid = T.new_flowsheet("cs2", COMPS, "PengRobinson1978")["flowsheet_id"]
    T.add_material_stream(fid, "F", 350.0, 2e5, molar_flow_mol_s=100.0, mole_fractions=Z)
    T.add_material_stream(fid, "A")
    T.add_material_stream(fid, "B")
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "component_separator", "CS", {"recoveries": {"Toluene": 1.0}})
    T.connect(fid, "CS", ["F"], ["A", "B"], energy_feeds=["Q"])
    assert T.solve(fid)["ok"]
    assert _flows(fid, "A")["Toluene"] == pytest.approx(20.0, rel=1e-9)
    assert T.read_stream(fid, "A")["molar_flow_mol_s"] == pytest.approx(20.0, rel=1e-9)


def test_recycle_with_purge_matches_analytic_balance():
    """Fresh feed 100 mol/s (10 % methane inert). The separator sends all toluene and hydrogen to product and all methane
    to a loop with a 10 % purge: methane into the separator = 10 / 0.1 = 100 mol/s, recycle = 90 mol/s."""
    fid = T.new_flowsheet("loop", COMPS, "PengRobinson1978")["flowsheet_id"]
    T.add_material_stream(fid, "feed", 350.0, 2e5, molar_flow_mol_s=100.0,
                          mole_fractions={"Methane": 0.1, "Hydrogen": 0.4, "Toluene": 0.5})
    for n in ("mixed", "product", "loop", "purge", "tear", "recycled"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "mixer", "M", {})
    T.add_unit(fid, "component_separator", "CS", {"recoveries": {"Methane": 1.0}})
    T.add_unit(fid, "splitter", "SP", {"split_ratios": [0.1, 0.9]})
    T.add_unit(fid, "recycle", "R", {})
    T.connect(fid, "M", ["feed", "recycled"], ["mixed"])
    T.connect(fid, "CS", ["mixed"], ["loop", "product"], energy_products=["Q"])
    T.connect(fid, "SP", ["loop"], ["purge", "tear"])
    T.connect(fid, "R", ["tear"], ["recycled"])
    sol = T.solve(fid)
    assert sol["ok"], sol
    assert T.read_stream(fid, "loop")["molar_flow_mol_s"] == pytest.approx(100.0, rel=1e-4)
    assert T.read_stream(fid, "recycled")["molar_flow_mol_s"] == pytest.approx(90.0, rel=1e-4)
    assert T.read_stream(fid, "purge")["molar_flow_mol_s"] == pytest.approx(10.0, rel=1e-4)
    assert T.read_stream(fid, "product")["molar_flow_mol_s"] == pytest.approx(90.0, rel=1e-6)


def test_input_errors():
    fid = T.new_flowsheet("bad", COMPS, "PengRobinson1978")["flowsheet_id"]
    with pytest.raises(ValueError, match="not in the flowsheet"):
        T.add_unit(fid, "component_separator", "X", {"recoveries": {"Water": 0.5}})
    with pytest.raises(ValueError, match="between 0 and 1"):
        T.add_unit(fid, "component_separator", "Y", {"recoveries": {"Methane": 99.0}})
    with pytest.raises(ValueError, match="recoveries"):
        T.add_unit(fid, "component_separator", "Z", {})


def test_dof_check_requires_the_energy_stream():
    fid = _base({"Hydrogen": 1.0}, connect_energy=False)
    rep = T.dof_check(fid)
    assert not rep["ok"]
    assert any("energy_products" in p for p in rep["problems"])
    assert T.dof_check(_base({"Hydrogen": 1.0}))["ok"]
