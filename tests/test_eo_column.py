"""Equation-oriented column optimisation (bloques M8a and M8b): IDAES TrayColumn + ipopt (idaes-ext).

M8a evidence: IDAES's own test case reproduced with its stock configuration; the corrected liquid cp gives ΔHvap within
3 % of DIPPR (thermo); the optimum is verified (neighbours, multipliers vs finite differences); DWSIM (Raoult) at the
IDAES optimum gives the same duties within 1.5 % and purities within 0.005.
M8b evidence: the property package generated from chemicals reproduces chemicals/DIPPR (same-source to 1e-9, so no unit
error); non-ideal mixtures are refused; with benzene-toluene it reproduces the M8a optimum; hexane-heptane (no IDAES
reference case) at the optimum matches DWSIM within the M8a tolerance.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.eo import column_optimize as C  # noqa: E402

needs_ipopt = pytest.mark.skipif(not C.ipopt_available(), reason="ipopt not installed (uv run idaes get-extensions)")
FEED = {"flow_mol_s": 100.0, "T_K": 368.0, "P_Pa": 101325.0, "x_benzene": 0.5}


BT_SPECS = [{"stream": "distillate", "component": "benzene", "kind": "mole_fraction", "min": 0.95},
            {"stream": "bottoms", "component": "toluene", "kind": "mole_fraction", "min": 0.95}]
HH_SPECS = [{"stream": "distillate", "component": "n-hexane", "kind": "mole_fraction", "min": 0.95},
            {"stream": "bottoms", "component": "n-heptane", "kind": "mole_fraction", "min": 0.95}]


def test_inputs_are_validated():
    args = dict(components=["benzene", "toluene"], feed_flow_mol_s=100, feed_T_K=368, feed_P_Pa=101325,
                feed_mole_fractions={"benzene": 0.5, "toluene": 0.5}, n_trays=10, feed_tray=5, pressure_Pa=101325,
                specs=BT_SPECS)
    for bad in ({"components": ["benzene"]}, {"feed_mole_fractions": {"benzene": 0.6, "toluene": 0.5}},
                {"feed_mole_fractions": {"benzene": 1.0}}, {"feed_tray": 11}, {"pressure_Pa": 0}, {"feed_T_K": None},
                {"specs": []}, {"specs": [BT_SPECS[0] | {"kind": "purity"}]}, {"specs": [BT_SPECS[0] | {"component": "water"}]},
                {"property_package": "idaes_BT_ideal", "components": ["n-hexane", "n-heptane"],
                 "feed_mole_fractions": {"n-hexane": 0.5, "n-heptane": 0.5}, "specs": HH_SPECS}):
        with pytest.raises(ValueError):
            C.eo_column_optimize(**(args | bad))


@pytest.mark.parametrize("pair", [("ethanol", "water"), ("benzene", "cyclohexane"), ("acetone", "chloroform"),
                                  ("n-heptane", "toluene")])
def test_nonideal_mixtures_are_refused(pair):
    from packages.eo import ideal_properties as IP
    chk = IP.ideality_check(list(pair), 101325)
    assert not chk["ideal"] and chk["reasons"]
    with pytest.raises(ValueError, match="no ideal"):
        C.eo_column_optimize(list(pair), 100, 350, 101325, {pair[0]: 0.5, pair[1]: 0.5}, 10, 5, 101325,
                             [{"stream": "distillate", "component": pair[0], "kind": "mole_fraction", "min": 0.9}])


@pytest.mark.parametrize("names", [["benzene", "toluene"], ["n-hexane", "n-heptane"], ["benzene", "toluene", "p-xylene"]])
def test_ideal_mixtures_pass_the_screen(names):
    from packages.eo import ideal_properties as IP
    chk = IP.ideality_check(names, 101325)
    assert chk["ideal"], chk["reasons"]
    assert all(p["max_ln_gamma_ratio"] < 0.06 and not p["azeotrope_predicted"] for p in chk["pairs"])


@pytest.mark.parametrize("names", [["benzene", "toluene"], ["n-hexane", "n-heptane"]])
def test_generated_package_reproduces_chemicals(names):
    """Values computed by the IDAES package itself vs chemicals/thermo: same-source tables agree to 1e-9 (a unit slip
    like BT_ideal's J/kmol would show as 1000×), DIPPR/TRC within a few %, and the magnitudes are physical."""
    from packages.eo import ideal_properties as IP
    cfg, prov = IP.build_configuration(names, 101325)
    ts = [IP.saturation_T(n, 101325) for n in names]
    pc = IP.property_check(cfg, names, [min(ts), 0.5 * (min(ts) + max(ts)), max(ts)])
    assert pc["verified"], pc["max_rel_diff"]
    for k in ("psat_same_source", "cp_ig_same_source", "dens_liq_same_source"):
        assert pc["max_rel_diff"][k] < 1e-9
    assert pc["max_rel_diff"]["dhvap_vs_dippr106"] < 0.02
    for r in pc["rows"]:
        assert 100 < r["cp_liq_J_mol_K"] < 300 and 80 < r["cp_ig_J_mol_K"] < 250
        assert 6000 < r["dens_liq_mol_m3"] < 12000 and 25 < r["dhvap_kJ_mol"] < 40
    for n in names:   # at its own boiling point the package's ΔHvap is DIPPR 106 and its Psat is ~1 atm
        row = [r for r in IP.property_check(cfg, names, [prov["components"][n]["dhvap_anchor_T_K"]])["rows"]
               if r["component"] == n][0]
        assert row["dhvap_kJ_mol"] == pytest.approx(row["dhvap_dippr106_kJ_mol"], rel=1e-4)
        assert row["psat_Pa"] == pytest.approx(101325, rel=0.01)


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
    r = C.eo_column_optimize(["benzene", "toluene"], 100, 368, 101325, {"benzene": 0.5, "toluene": 0.5}, 10, 5, 101325,
                             BT_SPECS, property_package="idaes_BT_ideal")
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert set(r["active_specs"]) == {"mole_fraction:benzene@distillate", "mole_fraction:toluene@bottoms"}
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


def _dwsim_column(name, compounds, feed_T, flows, stages, feed_stage, rr, bottoms_mol_s):
    from packages.dwsim_client import tools as T
    fid = T.new_flowsheet(name, compounds, "Raoult")["flowsheet_id"]
    T.add_material_stream(fid, "feed", temperature_K=feed_T, pressure_Pa=101325, molar_flow_mol_s=sum(flows.values()),
                          compound_molar_flows_mol_s=flows)
    for n in ("dist", "bot"):
        T.add_material_stream(fid, n)
    for n in ("cd", "rd"):
        T.add_energy_stream(fid, n)
    T.add_unit(fid, "distillation_column", "T-1", {"stages": stages, "feed_stage": feed_stage, "reflux_ratio": rr,
                                                   "bottoms_molar_flow_mol_s": bottoms_mol_s, "top_P_Pa": 101325})
    T.connect(fid, "T-1", feeds=["feed"], products=["dist", "bot"], energy_products=["cd"], energy_feeds=["rd"])
    assert T.solve(fid)["ok"]
    out = T.read_stream(fid, "dist"), T.read_stream(fid, "bot"), T.read_unit(fid, "T-1")
    T.close_flowsheet(fid)
    return out


@needs_ipopt
def test_generated_package_reproduces_m8a():
    """Same column and specs as M8a, but the property package generated from chemicals instead of IDAES's BT_ideal:
    the optimum moves only by the data difference (Wagner-McGarry vs RPP4 Psat, ΔHvap anchored to DIPPR)."""
    r = C.eo_column_optimize(["benzene", "toluene"], 100, 368, 101325, {"benzene": 0.5, "toluene": 0.5}, 10, 5, 101325,
                             BT_SPECS)
    ev, o = r["evidence"], r["optimum"]
    assert ev["verified"], ev["reasons"]
    assert ev["properties_verified"] and r["properties"]["ideality_check"]["ideal"]
    # M8a (tests above): RR 2.452, BR 2.411, Q_reb 4.105 MW, Q_cond 5.381 MW, multipliers 26.8 / 22.1 MW per unit
    assert o["reflux_ratio"] == pytest.approx(2.452, rel=0.01) and o["boilup_ratio"] == pytest.approx(2.411, rel=0.01)
    assert r["reboiler_duty_MW"] == pytest.approx(4.105, rel=0.02)
    assert -r["condenser_duty_MW"] == pytest.approx(5.381, rel=0.02)
    assert r["duals_MW_per_unit_spec"]["mole_fraction:benzene@distillate"] == pytest.approx(26.79, rel=0.02)
    assert r["duals_MW_per_unit_spec"]["mole_fraction:toluene@bottoms"] == pytest.approx(22.12, rel=0.02)
    assert o["distillate"]["T_K"] == pytest.approx(354.33, abs=0.2) and o["bottoms"]["T_K"] == pytest.approx(381.58, abs=0.2)


@needs_ipopt
def test_hexane_heptane_optimum_matches_dwsim():
    """A mixture with no IDAES reference case: the generated-package optimum, rebuilt in DWSIM (Raoult, ChemSep data,
    12 stages = condenser + 10 trays + reboiler) at the same RR and bottoms flow, within the M8a tolerance."""
    r = C.eo_column_optimize(["n-hexane", "n-heptane"], 100, 355, 101325, {"n-hexane": 0.5, "n-heptane": 0.5}, 10, 5,
                             101325, HH_SPECS)
    ev, o = r["evidence"], r["optimum"]
    assert ev["verified"], ev["reasons"]
    for chk in ev["dual_checks"].values():
        assert chk["consistent"] and chk["rel_diff"] < 0.01
    assert o["reflux_ratio"] == pytest.approx(2.291, abs=0.005) and r["reboiler_duty_MW"] == pytest.approx(4.164, abs=0.005)
    try:
        from packages.dwsim_client import tools as T  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"DWSIM not available: {ex}")
    d, b, u = _dwsim_column("eo_hh", ["N-hexane", "N-heptane"], 355, {"N-hexane": 50, "N-heptane": 50}, 12, 6,
                            o["reflux_ratio"], o["bottoms"]["flow_mol_s"])
    assert d["mole_fractions"]["N-hexane"] == pytest.approx(0.95, abs=0.005)
    assert b["mole_fractions"]["N-heptane"] == pytest.approx(0.95, abs=0.005)
    assert d["T_K"] == pytest.approx(o["distillate"]["T_K"], abs=0.5) and b["T_K"] == pytest.approx(o["bottoms"]["T_K"], abs=0.5)
    assert u["energy_streams_kW"]["rd"] / 1e3 == pytest.approx(r["reboiler_duty_MW"], rel=0.015)
    assert u["energy_streams_kW"]["cd"] / 1e3 == pytest.approx(-r["condenser_duty_MW"], rel=0.015)
