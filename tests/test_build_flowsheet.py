"""build_flowsheet (whole flowsheet in one call) and flowsheet_goal_seek (design specifications), against analytic balances.

Loop used throughout: fresh feed F with 10 % methane (inert), 40 % hydrogen, 50 % toluene; an ideal component separator
sends all methane to a loop with purge fraction p and everything else to product. Steady state: methane entering the
separator = 0.1·F / p, recycle = 0.1·F·(1 − p)/p, product = 0.9·F.
"""
import pytest

from packages.dwsim_client import tools as T
from packages.dwsim_client import tools_build as TB

COMPS = ["Methane", "Hydrogen", "Toluene"]


def _spec(purge=0.1, feed=100.0):
    return dict(name="loop", compounds=COMPS, property_package="PengRobinson1978",
                streams=[{"name": "feed", "temperature_K": 350.0, "pressure_Pa": 2e5, "molar_flow_mol_s": feed,
                          "mole_fractions": {"Methane": 0.1, "Hydrogen": 0.4, "Toluene": 0.5}}],
                units=[{"kind": "mixer", "name": "M", "feeds": ["feed", "recycled"], "products": ["mixed"]},
                       {"kind": "component_separator", "name": "CS", "params": {"recoveries": {"Methane": 1.0}},
                        "feeds": ["mixed"], "products": ["loop", "product"], "energy_products": ["Q"]},
                       {"kind": "splitter", "name": "SP", "params": {"split_ratios": [purge, 1 - purge]},
                        "feeds": ["loop"], "products": ["purge", "tear"]},
                       {"kind": "recycle", "name": "R", "feeds": ["tear"], "products": ["recycled"]}])


def test_one_call_builds_and_solves_the_loop():
    out = TB.build_flowsheet(**_spec())
    assert out["built"] and out["ok"], out
    assert out["created"] == {"reactions": 0, "material_streams": 7, "energy_streams": 1, "units": 4}
    assert out["dof"]["ok"] is True
    st = out["solve"]["streams"]
    assert st["loop"]["molar_flow_mol_s"] == pytest.approx(100.0, rel=1e-4)
    assert st["recycled"]["molar_flow_mol_s"] == pytest.approx(90.0, rel=1e-4)
    assert st["product"]["molar_flow_mol_s"] == pytest.approx(90.0, rel=1e-6)
    T.close_flowsheet(out["flowsheet_id"])


def test_reactions_are_defined_from_the_specification():
    """Toluene + H2 -> Benzene + CH4 at 75 % toluene conversion: 100 mol/s toluene leaves 25, makes 75 benzene."""
    out = TB.build_flowsheet(
        "hda", ["Toluene", "Hydrogen", "Benzene", "Methane"], "PengRobinson1978",
        streams=[{"name": "in", "temperature_K": 900.0, "pressure_Pa": 3e6,
                  "compound_molar_flows_mol_s": {"Toluene": 100.0, "Hydrogen": 500.0}}],
        units=[{"kind": "conversion_reactor", "name": "RX", "params": {"reaction_set": "RS1", "mode": "isothermal"}, "feeds": ["in"],
                "products": ["out", "liq"], "energy_feeds": ["Qr"]}],
        reactions=[{"type": "conversion", "name": "hda", "base_compound": "Toluene", "conversion_percent": 75.0,
                    "stoichiometry": {"Toluene": -1, "Hydrogen": -1, "Benzene": 1, "Methane": 1}}])
    assert out["ok"], out
    o = out["solve"]["streams"]["out"]
    assert o["molar_flow_mol_s"] * o["mole_fractions"]["Benzene"] == pytest.approx(75.0, rel=1e-6)
    assert o["molar_flow_mol_s"] * o["mole_fractions"]["Toluene"] == pytest.approx(25.0, rel=1e-6)
    T.close_flowsheet(out["flowsheet_id"])


def test_rejected_element_is_named_and_nothing_is_kept():
    spec = _spec()
    spec["units"][2]["kind"] = "centrifuge"
    before = set(T.SESSIONS)
    out = TB.build_flowsheet(**spec)
    assert out["built"] is False and out["failed_at"] == "units" and out["item"]["name"] == "SP"
    assert set(T.SESSIONS) == before
    spec = _spec()
    spec["units"][0]["inlets"] = ["feed"]
    out = TB.build_flowsheet(**spec)
    assert out["built"] is False and "inlets" in out["error"]
    out = TB.build_flowsheet("x", COMPS, "PengRobinson1978", reactions=[{"type": "magic", "name": "r"}])
    assert out["built"] is False and out["failed_at"] == "reactions"


def test_goal_seek_one_specification():
    """Methane/toluene = 1 in the separator feed needs 50 mol/s methane there: purge fraction = 10/50 = 0.2."""
    fid = TB.build_flowsheet(**_spec())["flowsheet_id"]
    out = TB.flowsheet_goal_seek(
        fid, [{"kind": "unit", "name": "SP", "param": "split_ratio_0", "lower": 0.05, "upper": 0.9}],
        [{"kind": "stream", "name": "mixed", "key": "component_flow.Methane", "value": 1.0,
          "divide_by": {"kind": "stream", "name": "mixed", "key": "component_flow.Toluene"}}])
    assert out["converged"], out
    assert out["variables"][0]["value"] == pytest.approx(0.2, rel=1e-3)
    assert T.read_stream(fid, "purge")["molar_flow_mol_s"] == pytest.approx(10.0, rel=1e-3)
    T.close_flowsheet(fid)


def test_goal_seek_two_specifications_together():
    """Product = 180 mol/s needs F = 200; methane/toluene = 1 then needs 100 mol/s methane with 20 fresh: purge = 0.2."""
    fid = TB.build_flowsheet(**_spec())["flowsheet_id"]
    out = TB.flowsheet_goal_seek(
        fid, [{"kind": "unit", "name": "SP", "param": "split_ratio_0", "lower": 0.05, "upper": 0.9},
              {"kind": "stream", "name": "feed", "param": "molar_flow_mol_s", "lower": 50.0, "upper": 400.0}],
        [{"kind": "stream", "name": "mixed", "key": "component_flow.Methane", "value": 1.0,
          "divide_by": {"kind": "stream", "name": "mixed", "key": "component_flow.Toluene"}},
         {"kind": "stream", "name": "product", "key": "molar_flow_mol_s", "value": 180.0}], tolerance=1e-4)
    assert out["converged"], out
    assert out["variables"][0]["value"] == pytest.approx(0.2, rel=2e-3)
    assert out["variables"][1]["value"] == pytest.approx(200.0, rel=2e-3)
    T.close_flowsheet(fid)


def test_goal_seek_reports_unreachable_target_and_bad_input():
    fid = TB.build_flowsheet(**_spec())["flowsheet_id"]
    out = TB.flowsheet_goal_seek(
        fid, [{"kind": "unit", "name": "SP", "param": "split_ratio_0", "lower": 0.5, "upper": 0.9}],
        [{"kind": "stream", "name": "product", "key": "molar_flow_mol_s", "value": 500.0}], max_solves=15)
    assert out["converged"] is False and "note" in out
    with pytest.raises(ValueError):
        TB.flowsheet_goal_seek(fid, [], [])
    with pytest.raises(ValueError):
        TB.flowsheet_goal_seek(fid, [{"kind": "unit", "name": "SP", "param": "split_ratio_0", "lower": 0.9, "upper": 0.1}],
                               [{"kind": "stream", "name": "product", "key": "molar_flow_mol_s", "value": 90.0}])
    T.close_flowsheet(fid)
