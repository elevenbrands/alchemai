"""Saving a flowsheet (snapshot) and rebuilding it in a fresh session gives the same results: mixer, the ethanol-water
column after an edit (reflux and a stage-count rebuild), and a kinetic CSTR with its reaction definition."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client import persist  # noqa: E402
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_sweep import set_unit_param  # noqa: E402


def _roundtrip(fid, streams):
    before = {n: T.read_stream(fid, n) for n in streams}
    snap = json.loads(json.dumps(persist.snapshot_flowsheet(fid)))   # must survive JSON (the app writes it to disk)
    T.close_flowsheet(fid)
    r = persist.restore_flowsheet(snap)
    assert r["solve"]["ok"], r["solve"]
    nid = r["flowsheet_id"]
    after = {n: T.read_stream(nid, n) for n in streams}
    for n in streams:
        assert after[n]["T_K"] == pytest.approx(before[n]["T_K"], rel=1e-6), n
        assert after[n]["molar_flow_mol_s"] == pytest.approx(before[n]["molar_flow_mol_s"], rel=1e-6), n
        for c, x in before[n]["mole_fractions"].items():
            assert after[n]["mole_fractions"][c] == pytest.approx(x, abs=1e-6), (n, c)
    return nid, snap


def test_mixer_roundtrip_keeps_zero_vapor_fraction():
    fid = T.new_flowsheet("mix", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "a", temperature_K=300.0, pressure_Pa=101325.0, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "b", vapor_fraction=0.0, pressure_Pa=101325.0, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "m")
    T.add_unit(fid, "mixer", "MIX", {})
    T.connect(fid, "MIX", ["a", "b"], ["m"])
    assert T.solve(fid)["ok"]
    nid, snap = _roundtrip(fid, ["a", "b", "m"])
    assert snap["streams"][1]["spec"]["vapor_fraction"] == 0.0     # bubble-point feed kept (was dropped as falsy)
    T.close_flowsheet(nid)


def test_column_roundtrip_after_edits():
    spec = importlib.util.spec_from_file_location("nb20", ROOT / "notebooks" / "20_variables_enteras.py")
    nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
    fid = nb.column(20, 2.0)
    set_unit_param(fid, "T-101", "stages", 29)
    set_unit_param(fid, "T-101", "reflux_ratio", 1.394)
    assert T.solve(fid)["ok"]
    nid, snap = _roundtrip(fid, ["feed", "dist", "bot"])
    u = snap["units"][0]
    assert u["params"]["stages"] == 29 and u["params"]["reflux_ratio"] == 1.394 and u["params"]["feed_stage"] == 15
    assert snap["property_package"] == "NRTL" and snap["energy"] == ["qc", "qr"]
    T.close_flowsheet(nid)


def test_kinetic_cstr_roundtrip_keeps_the_reaction():
    fid = T.new_flowsheet("cstr", ["N-butane", "Isobutane"], "PengRobinson")["flowsheet_id"]
    T.define_kinetic_reaction(fid, "ISO", {"N-butane": -1, "Isobutane": 1}, "N-butane", {"N-butane": 1}, 1.0e8, 100_000.0)
    T.add_material_stream(fid, "feed", temperature_K=600.0, pressure_Pa=1.0e5, molar_flow_mol_s=10.0, mole_fractions={"N-butane": 1.0})
    T.add_material_stream(fid, "prod"); T.add_energy_stream(fid, "q")
    T.add_unit(fid, "cstr", "R", {"volume_m3": 10.0, "reaction_set": "KIN"})
    T.connect(fid, "R", feeds=["feed"], products=["prod"], energy_feeds=["q"])
    assert T.solve(fid)["ok"]
    nid, snap = _roundtrip(fid, ["feed", "prod"])
    assert snap["reaction_defs"][0]["tool"] == "define_kinetic_reaction" and snap["reaction_defs"][0]["args"]["A_forward"] == 1.0e8
    T.close_flowsheet(nid)


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError):
        persist.restore_flowsheet({"version": 99})
