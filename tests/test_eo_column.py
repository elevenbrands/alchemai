"""Equation-oriented column optimisation (bloque M8a): IDAES TrayColumn + ipopt (idaes-ext).

Evidence: IDAES's own test case reproduced with its stock configuration; the corrected liquid cp gives ΔHvap within 3 %
of DIPPR (thermo); the optimum is verified (neighbours, multipliers vs finite differences); DWSIM (Raoult) at the
IDAES optimum gives the same duties within 1.5 % and purities within 0.005.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.eo import column_optimize as C  # noqa: E402

needs_ipopt = pytest.mark.skipif(not C.ipopt_available(), reason="ipopt not installed (uv run idaes get-extensions)")
FEED = {"flow_mol_s": 100.0, "T_K": 368.0, "P_Pa": 101325.0, "x_benzene": 0.5}


def test_inputs_are_validated():
    args = dict(system="benzene-toluene ideal", feed_flow_mol_s=100, feed_T_K=368, feed_P_Pa=101325, feed_x_benzene=0.5,
                n_trays=10, feed_tray=5, pressure_Pa=101325, x_benzene_distillate_min=0.95, x_toluene_bottoms_min=0.95)
    for bad in ({"system": "ethanol-water"}, {"feed_x_benzene": 1.2}, {"feed_tray": 11}, {"pressure_Pa": 0}):
        with pytest.raises(ValueError):
            C.eo_column_optimize(**(args | bad))


def test_tool_is_registered():
    from packages.dwsim_client import tools as T  # noqa: F401
    from packages.dwsim_client import tools_optimize as TO
    from packages.agent import agent as AG
    assert TO.eo_column_optimize in TO.OPTIMIZE_TOOLS
    assert any(getattr(t, "name", "") == "eo_column_optimize" for t in AG.AGENT_TOOLS)


@needs_ipopt
def test_reproduces_the_idaes_reference_case():
    """idaes/models_extra/column_models/tests/test_tray_column.py::TestBTXIdealGeneric (stock configuration)."""
    s = C.simulate_column(10, 5, FEED, 101325.0, reflux_ratio=1.4, boilup_ratio=1.3, corrected=False)
    assert s["distillate"]["flow_mol_s"] == pytest.approx(49.28, rel=1e-2)
    assert s["distillate"]["x_benzene"] == pytest.approx(0.8784, rel=1e-2)
    assert s["distillate"]["T_K"] == pytest.approx(355.853, rel=1e-4)
    assert s["bottoms"]["flow_mol_s"] == pytest.approx(50.711, rel=1e-2)
    assert s["bottoms"]["x_toluene"] == pytest.approx(0.8677, rel=1e-3)
    assert s["bottoms"]["T_K"] == pytest.approx(378.0433, rel=1e-4)


@needs_ipopt
@pytest.mark.parametrize("corrected,max_rel", [(True, 0.03), (False, None)])
def test_liquid_cp_correction_fixes_the_latent_heat(corrected, max_rel):
    from pyomo.environ import ConcreteModel, value
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    from thermo import Chemical
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.p = GenericParameterBlock(**C.bt_ideal_configuration(corrected))
    m.fs.sb = m.fs.p.build_state_block([0], defined_state=True)
    s = m.fs.sb[0]
    s.flow_mol.fix(1)
    s.pressure.fix(101325)
    s.mole_frac_comp["benzene"].fix(0.5)
    s.mole_frac_comp["toluene"].fix(0.5)
    worst = 0.0
    for T in (298.15, 354.0, 381.5):
        s.temperature.fix(T)
        for c in C.COMPONENTS:
            idaes_dh = value(s.enth_mol_phase_comp["Vap", c]) - value(s.enth_mol_phase_comp["Liq", c])
            worst = max(worst, abs(idaes_dh / Chemical(c, T=T).Hvapm - 1))
    if corrected:
        assert worst < max_rel
    else:
        assert worst > 0.25            # the stock configuration is off by 26-43 % above 298 K


@needs_ipopt
def test_optimum_is_verified_and_matches_dwsim():
    r = C.eo_column_optimize("benzene-toluene ideal", 100, 368, 101325, 0.5, 10, 5, 101325, 0.95, 0.95)
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert set(r["active_specs"]) == {"x_benzene_distillate", "x_toluene_bottoms"}
    o = r["optimum"]
    assert o["reflux_ratio"] == pytest.approx(2.452, abs=0.005) and o["boilup_ratio"] == pytest.approx(2.411, abs=0.005)
    assert r["reboiler_duty_MW"] == pytest.approx(4.105, abs=0.005)
    for chk in ev["dual_checks"].values():
        assert chk["consistent"] and chk["rel_diff"] < 0.01
    # DWSIM rigorous column (Raoult, ChemSep data) at the IDAES optimum: condenser + 10 trays + reboiler = 12 stages
    try:
        from packages.dwsim_client import tools as T
        fid = T.new_flowsheet("eo_bt", ["Benzene", "Toluene"], "Raoult")["flowsheet_id"]
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"DWSIM not available: {ex}")
    T.add_material_stream(fid, "feed", temperature_K=368, pressure_Pa=101325, molar_flow_mol_s=100,
                          compound_molar_flows_mol_s={"Benzene": 50, "Toluene": 50})
    for n in ("dist", "bot"):
        T.add_material_stream(fid, n)
    for n in ("cd", "rd"):
        T.add_energy_stream(fid, n)
    T.add_unit(fid, "distillation_column", "T-1", {"stages": 12, "feed_stage": 6, "reflux_ratio": o["reflux_ratio"],
                                                   "bottoms_molar_flow_mol_s": o["bottoms"]["flow_mol_s"], "top_P_Pa": 101325})
    T.connect(fid, "T-1", feeds=["feed"], products=["dist", "bot"], energy_products=["cd"], energy_feeds=["rd"])
    assert T.solve(fid)["ok"]
    d, b, u = T.read_stream(fid, "dist"), T.read_stream(fid, "bot"), T.read_unit(fid, "T-1")
    assert d["mole_fractions"]["Benzene"] == pytest.approx(0.95, abs=0.005)
    assert b["mole_fractions"]["Toluene"] == pytest.approx(0.95, abs=0.005)
    assert d["T_K"] == pytest.approx(o["distillate"]["T_K"], abs=0.5) and b["T_K"] == pytest.approx(o["bottoms"]["T_K"], abs=0.5)
    assert u["energy_streams_kW"]["rd"] / 1e3 == pytest.approx(r["reboiler_duty_MW"], rel=0.015)
    assert u["energy_streams_kW"]["cd"] / 1e3 == pytest.approx(-r["condenser_duty_MW"], rel=0.015)
    T.close_flowsheet(fid)
