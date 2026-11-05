"""Generic tool registry (shared by the agent and the MCP server) reproduces the reference results, and
the MCP server starts over stdio and exposes the 12 tools. The live Claude agent test runs only when
credentials exist (env or Keychain) and ALCHEMAI_LIVE=1."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client import tools as T  # noqa: E402


def test_tools_pump_matches_reference():
    fid = T.new_flowsheet("pump", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "w", temperature_K=298.15, pressure_Pa=1e5, mass_flow_kg_s=10.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "hp")
    T.add_unit(fid, "pump", "P-1", {"outlet_P_Pa": 11e5, "efficiency_pct": 70})
    T.connect(fid, "P-1", feeds=["w"], products=["hp"])
    sol = T.solve(fid)
    assert sol["ok"], sol
    assert T.read_unit(fid, "P-1")["power_kW"] == pytest.approx(14.328, abs=0.01)


def test_tools_shortcut_column_matches_reference():
    fid = T.new_flowsheet("fug", ["Ethane", "Propane", "N-butane", "N-pentane"], "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "alim", pressure_Pa=17e5, vapor_fraction=0.0, molar_flow_mol_s=100,
                          mole_fractions={"Ethane": 0.05, "Propane": 0.45, "N-butane": 0.40, "N-pentane": 0.10})
    for n in ("d", "b"):
        T.add_material_stream(fid, n)
    for n in ("qc", "qr"):
        T.add_energy_stream(fid, n)
    T.add_unit(fid, "shortcut_column", "T-1", {"light_key": "Propane", "heavy_key": "N-butane", "lk_mole_frac_in_bottoms": 0.02,
                                              "hk_mole_frac_in_distillate": 0.02, "condenser_P_Pa": 16e5, "reboiler_P_Pa": 17.5e5, "reflux_ratio": 1.8403542708971303})
    T.connect(fid, "T-1", feeds=["alim"], products=["d", "b"], energy_products=["qc"], energy_feeds=["qr"])
    assert T.solve(fid)["ok"]
    u = T.read_unit(fid, "T-1")
    assert u["N_min"] == pytest.approx(9.16, abs=0.05) and u["R_min"] == pytest.approx(1.281, abs=0.01)


def test_tools_reactor_and_column():
    fid = T.new_flowsheet("comb", ["Methane", "Oxygen", "Nitrogen", "Carbon dioxide", "Water"], "PengRobinson")["flowsheet_id"]
    T.define_conversion_reaction(fid, "COMB", {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}, "Methane", 100, "Comb")
    T.add_material_stream(fid, "feed", temperature_K=500, pressure_Pa=101325, molar_flow_mol_s=10.52,
                          compound_molar_flows_mol_s={"Methane": 1.0, "Oxygen": 2.0, "Nitrogen": 7.52})
    for n in ("flue", "liq"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "conversion_reactor", "R-1", {"reaction_set": "Comb", "mode": "isothermal"})
    T.connect(fid, "R-1", feeds=["feed"], products=["flue", "liq"], energy_feeds=["Q"])
    assert T.solve(fid)["ok"]
    assert T.read_unit(fid, "R-1")["heat_duty_kW"] == pytest.approx(-800.9, abs=1.0)

    fid = T.new_flowsheet("col", ["Water", "Ethanol"], "NRTL")["flowsheet_id"]
    T.add_material_stream(fid, "feed", temperature_K=300, pressure_Pa=101325, molar_flow_mol_s=100, compound_molar_flows_mol_s={"Water": 50, "Ethanol": 50})
    for n in ("dist", "bot"):
        T.add_material_stream(fid, n)
    for n in ("cd", "rd"):
        T.add_energy_stream(fid, n)
    T.add_unit(fid, "distillation_column", "T-101", {"stages": 20, "feed_stage": 10, "reflux_ratio": 2.0, "bottoms_molar_flow_mol_s": 75, "top_P_Pa": 101325})
    T.connect(fid, "T-101", feeds=["feed"], products=["dist", "bot"], energy_products=["cd"], energy_feeds=["rd"])
    assert T.solve(fid)["ok"]
    assert T.read_stream(fid, "dist")["mole_fractions"]["Ethanol"] == pytest.approx(0.8040, abs=0.002)


def test_mcp_server_lists_tools():
    out = subprocess.run([sys.executable, str(ROOT / "packages/dwsim_mcp/smoke_mcp.py")], capture_output=True, text=True, timeout=300)
    assert "36 tools" in out.stdout and "solve ok: True" in out.stdout, out.stdout[-800:] + out.stderr[-800:]


def _have_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    r = subprocess.run(["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-a", "alchemai"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not (_have_credentials() and os.environ.get("ALCHEMAI_LIVE") == "1"), reason="needs Anthropic credentials and ALCHEMAI_LIVE=1")
def test_live_agent_solves_pump_card():
    from packages.bench.runner import grade, load_cards, public_view
    from packages.agent.agent import solve_card
    card = load_cards(["bomba"])[0]
    resp = solve_card(public_view(card))
    score, rows, _ = grade(card, resp)
    assert score >= 0.99, (score, rows, resp)
