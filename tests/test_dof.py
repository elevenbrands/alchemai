"""dof_check: specification bookkeeping before solve(). Pure `analyse()` on synthetic session records (no DWSIM) plus one
live flowsheet through the tools layer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client.dof import analyse  # noqa: E402


def _session(streams, specs, kinds, params, conn):
    return {"streams": {n: None for n in streams}, "stream_specs": specs, "kinds": kinds, "params": params, "conn": conn}


def test_well_posed_pump_is_ok():
    s = _session(["w", "w2"], {"w": {"temperature_K": 300, "pressure_Pa": 1e5, "mass_flow_kg_s": 10, "mole_fractions": {"Water": 1.0}}},
                 {"P": "pump"}, {"P": {"outlet_P_Pa": 5e5, "efficiency_pct": 75}},
                 {"P": {"feeds": ["w"], "products": ["w2"], "energy_feeds": ["e"], "energy_products": []}})
    r = analyse(s)
    assert r["ok"] and r["units"]["P"]["status"] == "ok" and r["streams"]["w2"]["produced_by"] == "P"


def test_under_specified_feed_and_produced_stream_with_specs():
    s = _session(["gas", "lp"], {"gas": {"temperature_K": 300, "molar_flow_mol_s": 10, "mole_fractions": {"Methane": 0.9, "Ethane": 0.1}},
                                 "lp": {"temperature_K": 280}},
                 {"V": "valve"}, {"V": {"outlet_P_Pa": 2e5}}, {"V": {"feeds": ["gas"], "products": ["lp"], "energy_feeds": [], "energy_products": []}})
    r = analyse(s)
    assert not r["ok"]
    assert r["streams"]["gas"]["status"] == "under-specified" and "state" in r["streams"]["gas"]["missing"][0]
    assert r["streams"]["lp"]["status"] == "over-specified"


def test_over_specified_heater_and_mutually_exclusive_efficiencies():
    s = _session(["a", "b", "c"], {"a": {"temperature_K": 300, "pressure_Pa": 1e5, "mass_flow_kg_s": 1, "mole_fractions": {"Water": 1.0}}},
                 {"H": "heater", "K": "compressor"},
                 {"H": {"outlet_T_K": 350, "heat_added_kW": 100}, "K": {"outlet_P_Pa": 5e5, "adiabatic_eff_pct": 75, "polytropic_eff_pct": 80}},
                 {"H": {"feeds": ["a"], "products": ["b"], "energy_feeds": [], "energy_products": []},
                  "K": {"feeds": ["b"], "products": ["c"], "energy_feeds": [], "energy_products": []}})
    r = analyse(s)
    assert r["units"]["H"]["status"] == "over-specified" and "mutually exclusive" in r["units"]["H"]["extra"][0]
    assert r["units"]["K"]["status"] == "over-specified"


def test_ports_splitter_ratios_and_unconnected():
    s = _session(["f", "p1", "p2", "x"], {"f": {"temperature_K": 300, "pressure_Pa": 1e5, "molar_flow_mol_s": 1, "mole_fractions": {"Methane": 1.0}}},
                 {"S": "splitter", "R": "conversion_reactor", "Z": "valve"},
                 {"S": {"split_ratios": [0.7, 0.5]}, "R": {"reaction_set": "RS1"}, "Z": {"outlet_P_Pa": 1e5}},
                 {"S": {"feeds": ["f"], "products": ["p1", "p2"], "energy_feeds": [], "energy_products": []},
                  "R": {"feeds": ["p1"], "products": ["x"], "energy_feeds": [], "energy_products": []}})
    r = analyse(s)
    assert any("sum to 1.2000" in e for e in r["units"]["S"]["extra"])
    assert r["units"]["R"]["status"] == "under-specified" and any("products: 1 connected, 2 required" in m for m in r["units"]["R"]["missing"])
    assert any("energy_feeds: 0 connected, 1 required" in m for m in r["units"]["R"]["missing"])
    assert r["units"]["Z"]["status"] == "unconnected"


def test_live_dof_check_through_tools():
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_sweep import set_unit_param
    fid = T.new_flowsheet("d", ["Water"], "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "w", temperature_K=300, pressure_Pa=1e5, mass_flow_kg_s=10, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "w2"); T.add_energy_stream(fid, "e")
    T.add_unit(fid, "heater", "H", {"outlet_T_K": 350, "heat_added_kW": 100})
    T.connect(fid, "H", feeds=["w"], products=["w2"], energy_feeds=["e"])
    r = T.dof_check(fid)
    assert not r["ok"] and "over-specified" in r["problems"][0]
    set_unit_param(fid, "H", "outlet_T_K", 350.0)          # resolves the conflict in the record too
    assert T.dof_check(fid)["ok"]
    T.close_flowsheet(fid)
