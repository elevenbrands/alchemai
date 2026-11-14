"""M15: equation-oriented column with an NRTL liquid (packages/eo/nrtl_properties.py).

M15b: McCabe-Thiele feasibility and refusals, the excess-enthalpy switch, DWSIM's excess term, and (slow) the
ethanol-water columns M15 left open plus the like-for-like DWSIM comparison.

Fast tests: parameter plumbing (user > vle_binary_regression > ChemSep bank, one labelled source per pair), refusal
without parameters, azeotrope and ideal-vapour refusals, the IDAES NRTL expressions vs thermo.nrtl.NRTL, the ideal path
unchanged. Slow tests (ALCHEMAI_SLOW_EO_NRTL=1): the published methanol-water column of Simone (TU Wien 2015, Tables
7, 9, 10: Aspen Plus RadFrac and gPROMS, NRTL) reproduced, the optimum re-built in DWSIM with the SAME parameters, and
the vle_binary_regression → eo_column_optimize chain on ethanol-water (Kamihama 2012 data).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from packages.eo import column_optimize as C
from packages.eo import ideal_properties as IP
from packages.eo import nrtl_properties as NP

DATA = Path(__file__).parent / "data"
needs_ipopt = pytest.mark.skipif(not C.ipopt_available(), reason="ipopt not installed (uv run idaes get-extensions)")
slow = pytest.mark.skipif(os.environ.get("ALCHEMAI_SLOW_EO_NRTL") != "1" or not C.ipopt_available(),
                          reason="slow M15 reference/DWSIM tests: ALCHEMAI_SLOW_EO_NRTL=1")
MW_SPECS = [{"stream": "distillate", "component": "methanol", "kind": "mole_fraction", "min": 0.9168},
            {"stream": "distillate", "component": "methanol", "kind": "recovery", "min": 0.902}]


def _thesis_feed():
    """Simone 2015, Table 7: 10 000 kg/h, 5 wt% methanol, 313 K after the preheater, 1013 mbar."""
    from chemicals import MW
    nm, nw = 500 / MW("67-56-1") / 3.6, 9500 / MW("7732-18-5") / 3.6
    F = nm + nw
    return {"flow_mol_s": F, "T_K": 313.0, "P_Pa": 101300.0, "z": {"methanol": nm / F, "water": nw / F}}


# ------------------------------------------------------------------------------------------------ parameter sources
def test_chemsep_bank_is_the_one_dwsim_loads():
    p = NP.resolve_parameters(["methanol", "water"])
    pair = p["pairs"][0]
    assert pair["source_kind"] == "banco ChemSep NRTL (thermo)" and "ChemSep" in pair["source"]
    # DWSIM 10 NRTL databank, read from its NRTLPropertyPackage (m_uni.InteractionParameters["Methanol"]["Water"]):
    # A12 = -189.0469, A21 = 792.802 cal/mol, alpha12 = 0.2999; tau = A/(1.98721 T)
    assert pair["b_ij_K"] * 1.98721 == pytest.approx(-189.0469, abs=0.01)
    assert pair["b_ji_K"] * 1.98721 == pytest.approx(792.802, abs=0.01)
    assert pair["alpha"] == 0.2999 and pair["a_ij"] == pair["a_ji"] == 0.0


def test_user_parameters_win_and_need_a_citation():
    user = {"components": ["water", "methanol"], "dg_ij_cal_mol": 800.0, "dg_ji_cal_mol": -190.0, "alpha": 0.3,
            "source": "valores de prueba del usuario"}
    p = NP.resolve_parameters(["methanol", "water"], [user])
    pair = p["pairs"][0]
    assert pair["source_kind"] == "usuario" and (pair["i"], pair["j"]) == ("water", "methanol")
    # stored in the column's order: tau_b[methanol][water] = dg_ji/R
    assert p["tau_b_K"][0][1] == pytest.approx(-190.0 * 4.184 / 8.314462618)
    assert p["tau_b_K"][1][0] == pytest.approx(800.0 * 4.184 / 8.314462618)
    with pytest.raises(ValueError, match="source"):
        NP.resolve_parameters(["methanol", "water"], [{k: v for k, v in user.items() if k != "source"}])
    with pytest.raises(ValueError, match="alpha"):
        NP.resolve_parameters(["methanol", "water"], [{k: v for k, v in user.items() if k != "alpha"}])
    with pytest.raises(ValueError, match="a_ij y a_ji"):
        NP.resolve_parameters(["methanol", "water"], [user | {"a_ij": 1.0}])


def _ew_regression():
    from packages.vle.tool import vle_binary_regression
    EW = json.loads((DATA / "vle_ethanol_water_vlizard.json").read_text())
    k = next(k for k in EW["sets"] if "101" in k)
    return vle_binary_regression(EW["components"], [{"label": k, "points": EW["sets"][k], "citation": EW["citation"]}],
                                 models=["NRTL"], compare_unifac=False)


def test_regression_result_is_accepted_as_is_and_beats_the_bank():
    r = _ew_regression()
    assert r["ok"]
    p = NP.resolve_parameters(["ethanol", "water"], r)
    pair = p["pairs"][0]
    vals = {q["nombre"]: q["valor"] for q in r["modelos"]["NRTL"]["parametros"]}
    assert pair["source_kind"] == "vle_binary_regression" and "Kamihama" in pair["source"]
    assert pair["b_ij_K"] == pytest.approx(vals["b12"]) and pair["b_ji_K"] == pytest.approx(vals["b21"])
    assert pair["T_range_K"][0] < pair["T_range_K"][1]
    # the user's pair outranks the regression, which outranks the bank
    user = {"components": ["ethanol", "water"], "b_ij_K": -30.0, "b_ji_K": 620.0, "alpha": 0.3, "source": "prueba"}
    assert NP.resolve_parameters(["ethanol", "water"], [r, user])["pairs"][0]["source_kind"] == "usuario"
    with pytest.raises(ValueError, match="no de un par"):
        NP.resolve_parameters(["methanol", "water"], r)


def test_missing_pair_is_refused_and_points_to_the_regression():
    with pytest.raises(NP.MissingParameters, match="vle_binary_regression"):
        NP.resolve_parameters(["n-hexane", "n-heptane"])
    with pytest.raises(ValueError, match="vle_binary_regression"):
        C.eo_column_optimize(["n-hexane", "n-heptane"], 100, 355, 101325, {"n-hexane": 0.5, "n-heptane": 0.5}, 10, 5,
                             101325, [{"stream": "distillate", "component": "n-hexane", "kind": "mole_fraction",
                                       "min": 0.95}], activity_model="NRTL")


# ------------------------------------------------------------------------------------------------ screening
def test_azeotrope_with_these_parameters():
    p = NP.resolve_parameters(["ethanol", "water"])
    az = NP.binary_azeotropes(p, 101325)
    assert len(az) == 1 and az[0]["kind"] == "mínimo punto de ebullición"
    # ChemSep NRTL: x_EtOH 0.881 at 351.21 K (experimental ≈ 0.894 / 351.3 K, M14 handover)
    assert az[0]["x_az"]["ethanol"] == pytest.approx(0.8812, abs=0.002) and az[0]["T_az_K"] == pytest.approx(351.21, abs=0.05)


def test_specs_across_the_azeotrope_are_refused():
    z = {"ethanol": 0.1, "water": 0.9}
    bad = [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": 0.95}]
    with pytest.raises(ValueError, match="azeótropo"):
        C.eo_column_optimize(["ethanol", "water"], 100, 350, 101325, z, 15, 10, 101325, bad, activity_model="NRTL")
    p = NP.resolve_parameters(["ethanol", "water"])
    ok_specs = [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": 0.80}]
    assert NP.azeotrope_refusals(p, 101325, z, ok_specs)[1] == []
    # feed on the ethanol-rich side: pure water in the bottoms is across the azeotrope
    rich = {"ethanol": 0.95, "water": 0.05}
    w = [{"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": 0.9}]
    assert NP.azeotrope_refusals(p, 101325, rich, w)[1]


def test_non_ideal_vapour_is_refused():
    with pytest.raises(ValueError, match="gas ideal"):
        C.eo_column_optimize(["methanol", "water"], 100, 450, 2e6, {"methanol": 0.3, "water": 0.7}, 10, 5, 2e6,
                             MW_SPECS, activity_model="NRTL")


def test_ideal_path_unchanged_and_arguments_checked():
    with pytest.raises(ValueError, match="activity_model='NRTL'"):
        C.eo_column_optimize(["ethanol", "water"], 100, 350, 101325, {"ethanol": 0.1, "water": 0.9}, 10, 5, 101325,
                             [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": 0.8}])
    with pytest.raises(ValueError, match="binary_parameters"):
        C.eo_column_optimize(["benzene", "toluene"], 100, 368, 101325, {"benzene": 0.5, "toluene": 0.5}, 10, 5, 101325,
                             [{"stream": "distillate", "component": "benzene", "kind": "mole_fraction", "min": 0.9}],
                             binary_parameters=[{}])
    with pytest.raises(ValueError, match="activity_model"):
        C.eo_column_optimize(["benzene", "toluene"], 100, 368, 101325, {"benzene": 0.5, "toluene": 0.5}, 10, 5, 101325,
                             [{"stream": "distillate", "component": "benzene", "kind": "mole_fraction", "min": 0.9}],
                             activity_model="UNIQUAC")


def test_unifac_is_only_a_cross_check():
    cc = NP.unifac_cross_check(NP.resolve_parameters(["ethanol", "water"]), 101325)
    row = cc["pairs"][0]
    assert row["azeotrope"] == {"NRTL": True, "UNIFAC_Do": True}
    g = row["gamma_inf_water_en_ethanol"]
    assert g["NRTL"] == pytest.approx(2.64, abs=0.02) and g["UNIFAC_Do"] == pytest.approx(2.58, abs=0.02)


def test_idaes_nrtl_matches_thermo_and_gibbs_helmholtz():
    names = ["methanol", "water"]
    p = NP.resolve_parameters(names)
    cfg, prov = NP.build_configuration(names, 101300, p, extra_T=[313])
    chk = NP.nrtl_check(cfg, p, [320.0, 350.0, 373.0])
    assert chk["verified"], chk
    assert chk["max_abs_ln_gamma_diff"] < 1e-12 and chk["max_rel_hE_diff"] < 1e-5
    # water has no DIPPR 105 entry: cubic fit to VDI-PPDS, residual reported and within tolerance
    fit = prov["components"]["water"]["dens_liq_fit"]
    assert fit["method"] == "VDI_PPDS" and fit["max_rel_residual"] < IP.DENS_FIT_TOL
    assert IP.property_check(cfg, names, [338.0, 355.0, 372.0])["verified"]


def test_thermo_bubble_points_are_independent_of_idaes():
    p = NP.resolve_parameters(["methanol", "water"])
    b = NP.thermo_bubble_points(p, 101300, {"d": {"methanol": 0.9168, "water": 0.0832},
                                            "b": {"methanol": 0.0028, "water": 0.9972}})["points"]
    assert b["d"]["T_bubble_K"] == pytest.approx(338.93, abs=0.01)     # Simone 2015 Table 10: top 339 K
    assert b["b"]["T_bubble_K"] == pytest.approx(372.6, abs=0.1)       # bottom 373 K


def test_tool_signature_exposes_the_activity_model():
    import inspect
    from packages.dwsim_client import tools as T  # noqa: F401  (import order, see M8c handover)
    from packages.dwsim_client import tools_optimize as TO
    sig = inspect.signature(TO.eo_column_optimize)
    assert sig.parameters["activity_model"].default == "ideal" and "binary_parameters" in sig.parameters
    assert "vle_binary_regression" in TO.eo_column_optimize.__doc__


# ------------------------------------------------------------------------------------------------ M15b (fast)
EW_SPECS = lambda xd: [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": xd},  # noqa: E731
                       {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": 0.995}]
EW_Z = {"ethanol": 0.1, "water": 0.9}


def _ew_shortcut(xd, n, f, T=359.124, rr=(0.1, 20.0)):
    from packages.eo import shortcut as SC
    p = NP.resolve_parameters(["ethanol", "water"], _ew_regression())
    az = NP.binary_azeotropes(p, 101325.0)
    return SC.binary_shortcut(p, 101325.0, {"z": EW_Z, "T_K": T}, EW_SPECS(xd), n, f, rr, (0.1, 20.0), az)


def test_mccabe_thiele_says_the_m15_open_cases_are_feasible():
    """The four ethanol-water spec sets M15 could not close are physically reachable (same NRTL, own code): what
    failed was the start-up. R_needed (constant molar overflow) sits 1-5 % under the rigorous optimum of notebook 26."""
    for xd, n, f, T, rr_opt in ((0.70, 20, 14, 350.0, 0.7008), (0.70, 15, 10, 359.124, 0.7482),
                                (0.75, 20, 14, 359.124, 0.8948), (0.80, 20, 14, 359.124, 1.5585)):
        s = _ew_shortcut(xd, n, f, T)
        assert not s["refusals"], s["refusals"]
        assert s["N_min_total_reflux"] <= n + 1
        assert s["R_min_CMO"] <= s["R_needed_CMO"] <= rr_opt
        assert s["R_needed_CMO"] == pytest.approx(rr_opt, rel=0.06)
        assert s["starts"] and all(0.1 <= br <= 20 for _, br in s["starts"])
    s = _ew_shortcut(0.70, 20, 14, 350.0)
    assert s["light"] == "ethanol" and s["feed"]["state"] == "líquido subenfriado" and s["feed"]["q"] > 1.0
    assert s["D_over_F"] == pytest.approx((0.1 - 0.005) / (0.70 - 0.005))


def test_spec_pairs_follow_the_component_balance():
    from packages.eo import shortcut as SC
    z = {"methanol": 0.03, "water": 0.97}
    pair = SC.spec_pair(z, MW_SPECS, "methanol", "water")           # purity + recovery in the distillate
    dF = 0.902 * 0.03 / 0.9168
    assert pair["D_over_F"] == pytest.approx(dF) and pair["x_B"] == pytest.approx(0.03 * (1 - 0.902) / (1 - dF))
    both = [{"stream": "distillate", "component": "methanol", "kind": "recovery", "min": 0.9},
            {"stream": "bottoms", "component": "water", "kind": "recovery", "min": 0.99}]
    pr = SC.spec_pair(z, both, "methanol", "water")
    assert pr["x_D"] == pytest.approx(0.9 * 0.03 / (0.9 * 0.03 + 0.01 * 0.97))
    assert SC.spec_pair(z, MW_SPECS[:1], "methanol", "water") is None            # one spec: no pair, nothing refused
    bad = SC.spec_pair(z, [{"stream": "distillate", "component": "methanol", "kind": "mole_fraction", "min": 0.02},
                           {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": 0.99}],
                       "methanol", "water")
    assert bad["consistent"] is False


def test_unreachable_columns_are_refused_in_spanish_before_the_nlp():
    reg = _ew_regression()
    kw = dict(activity_model="NRTL", binary_parameters=reg)
    with pytest.raises(ValueError, match="ni a reflujo total") as e1:        # 4 trays + reboiler < N_min = 7
        C.eo_column_optimize(["ethanol", "water"], 100.0, 359.124, 101325.0, EW_Z, 4, 3, 101325.0, EW_SPECS(0.80), **kw)
    assert "hacen falta 7 etapas" in str(e1.value)
    with pytest.raises(ValueError, match="T_rocío − T_burbuja") as e2:       # next to the azeotrope: cannot be sealed
        C.eo_column_optimize(["ethanol", "water"], 100.0, 359.124, 101325.0, EW_Z, 20, 14, 101325.0, EW_SPECS(0.85), **kw)
    assert "azeótropo predicho" in str(e2.value) and "DWSIM" in str(e2.value) and "0.82" in str(e2.value)
    with pytest.raises(ValueError, match="reflujo máximo"):                  # reflux bound far under what the trays need
        C.eo_column_optimize(["ethanol", "water"], 100.0, 359.124, 101325.0, EW_Z, 20, 14, 101325.0, EW_SPECS(0.80),
                             reflux_ratio_bounds=(0.1, 1.0), **kw)
    # the measured limit: 0.030 K seals (x_D 0.82), 0.021 K does not (0.83)
    assert _ew_shortcut(0.82, 20, 14)["dT_dew_bubble_distillate_K"] > 0.03 > _ew_shortcut(0.83, 20, 14)["dT_dew_bubble_distillate_K"]
    assert not _ew_shortcut(0.82, 20, 14)["refusals"] and _ew_shortcut(0.83, 20, 14)["refusals"]


def test_excess_enthalpy_switch_and_homotopy_scale():
    """excess_enthalpy=False removes exactly h^E from the liquid enthalpy; nrtl_scale = 0 is the ideal liquid."""
    from pyomo.environ import ConcreteModel, value, units as pyunits
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    names = ["ethanol", "water"]
    p = NP.resolve_parameters(names)
    h = {}
    for flag in (True, False):
        cfg, prov = NP.build_configuration(names, 101325.0, p, extra_T=[350.0], excess_enthalpy=flag)
        assert ("sin h^E" in prov["liquid_model"]["excess_enthalpy"]) is (not flag)
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.p = GenericParameterBlock(**cfg)
        m.fs.sb = m.fs.p.build_state_block([0], defined_state=True)
        s = m.fs.sb[0]
        s.temperature.set_value(350.0)
        s.mole_frac_phase_comp["Liq", "ethanol"].set_value(0.1)
        s.mole_frac_phase_comp["Liq", "water"].set_value(0.9)
        h[flag] = value(pyunits.convert(s.enth_mol_phase["Liq"], pyunits.J / pyunits.mol))
        if flag:
            m.fs.p.Liq.nrtl_scale.set_value(0.0)
            assert value(s.enth_mol_phase["Liq"]) == pytest.approx(
                value(pyunits.convert(s.enth_mol_phase["Liq"], pyunits.J / pyunits.mol)))
            h["ideal"] = value(pyunits.convert(s.enth_mol_phase["Liq"], pyunits.J / pyunits.mol))
            assert value(NP._ln_gamma_expr(p, names, {"ethanol": 0.1, "water": 0.9}, s.temperature, "ethanol",
                                           m.fs.p.Liq.nrtl_scale)) == pytest.approx(0.0, abs=1e-14)
    he = NP.excess_enthalpy_numeric(p, 350.0, [0.1, 0.9])
    assert he == pytest.approx(float(NP.thermo_nrtl(p, 350.0, [0.1, 0.9]).HE()), rel=1e-6)
    assert h[True] - h[False] == pytest.approx(he, rel=1e-5)
    assert h["ideal"] == pytest.approx(h[False], rel=1e-12)
    import inspect
    from packages.dwsim_client import tools as T  # noqa: F401
    from packages.dwsim_client import tools_optimize as TO
    assert inspect.signature(TO.eo_column_optimize).parameters["excess_enthalpy"].default is True


def test_dwsim_excess_term_is_the_derivative_of_gamma_not_gibbs_helmholtz():
    """DWSIM 10.2.12: h(LiqCp_Excess) − h(ExpData) of a liquid stream = −R·T²·Σ x·∂γ/∂T, 2.8× the Gibbs-Helmholtz h^E
    of the same NRTL parameters at 10 % ethanol in water. This is why an NRTL column with h^E is compared with DWSIM
    in its «ExpData» mode and excess_enthalpy=False (notebook 26)."""
    try:
        from packages.dwsim_client import tools as T  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"DWSIM not available: {ex}")
    from packages.eo import dwsim_nrtl as DN
    p = NP.resolve_parameters(["ethanol", "water"], _ew_regression())
    names = {"ethanol": "Ethanol", "water": "Water"}
    states = {"feed": (350.0, {"Ethanol": 0.10, "Water": 0.90}), "mid": (352.0, {"Ethanol": 0.50, "Water": 0.50}),
              "bottoms": (371.68, {"Ethanol": 0.005, "Water": 0.995})}
    h = {mode: DN.liquid_enthalpies(["Ethanol", "Water"], states, p, names, mode) for mode in ("LiqCp_Excess", "ExpData")}
    for k_, (T_, x) in states.items():
        xs = [x["Ethanol"], x["Water"]]
        assert h["ExpData"][k_]["vapor_fraction"] == pytest.approx(0.0, abs=1e-9)
        dw = h["LiqCp_Excess"][k_]["h_J_mol"] - h["ExpData"][k_]["h_J_mol"]
        assert dw == pytest.approx(DN.dwsim_excess_term(p, T_, xs), rel=2e-3)
        assert dw > 1.4 * NP.excess_enthalpy_numeric(p, T_, xs)
    assert (h["LiqCp_Excess"]["feed"]["h_J_mol"] - h["ExpData"]["feed"]["h_J_mol"]) / NP.excess_enthalpy_numeric(
        p, 350.0, [0.1, 0.9]) == pytest.approx(2.79, abs=0.03)


# ------------------------------------------------------------------------------------------------ slow: reference + DWSIM
@slow
def test_reproduces_the_published_methanol_water_column():
    """Simone 2015 Table 9: 12 stages incl. condenser and reboiler (gPROMS: 10 + feed on 6), RR 4.36 (molar),
    distillate 474 kg/h. Table 10 (Aspen / gPROMS): T 339 / 373 K, Q_reb 1503 / 1522 kW, Q_cond -819 / -810 kW,
    methanol 451 / 450 kg/h top and 49 / 50 kg/h bottom. Their NRTL parameters are not printed: ChemSep's are used.
    Measured (2026-11-13): T 339.06 / 372.59 K, Q_reb 1481.7 kW (−1.4 % vs Aspen; Aspen and gPROMS differ by 1.3 % and
    DWSIM gives 1474-1512 kW depending on its liquid-enthalpy mode), Q_cond 820.6 kW, methanol top 448.7 kg/h. The
    duty tolerance (3 %) is the published models' own spread plus the enthalpy-model choice, not a fit."""
    st = C.simulate_column_nrtl(["methanol", "water"], _thesis_feed(), 10, 6, 101300.0, 4.36,
                                distillate_mass_flow_kg_s=474 / 3600)
    d, b = st["distillate"], st["bottoms"]
    assert d["mass_flow_kg_s"] * 3600 == pytest.approx(474, rel=1e-8)
    assert d["T_K"] == pytest.approx(339, abs=0.5) and b["T_K"] == pytest.approx(373, abs=0.5)
    assert st["reboiler_duty_W"] / 1e3 == pytest.approx(1503, rel=0.03)
    assert -st["condenser_duty_W"] / 1e3 == pytest.approx(819, rel=0.03)
    meoh_top = d["flow_mol_s"] * d["mole_fractions"]["methanol"] * 32.04186 * 3.6
    assert meoh_top == pytest.approx(451, rel=0.02)


@slow
def test_nrtl_optimum_matches_dwsim_with_the_same_parameters():
    try:
        from packages.dwsim_client import tools as T  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"DWSIM not available: {ex}")
    from packages.eo import dwsim_nrtl as DN
    f = _thesis_feed()
    r = C.eo_column_optimize(["methanol", "water"], f["flow_mol_s"], f["T_K"], f["P_Pa"], f["z"], 10, 6, 101300.0,
                             MW_SPECS, activity_model="NRTL")
    ev, o = r["evidence"], r["optimum"]
    assert ev["verified"], ev["reasons"]
    assert ev["nrtl_verified"] and ev["thermo_bubble_points_match"]
    assert ev["component_balance_closed"] and ev["phantom_vapour_fraction"] < 1e-4
    params = NP.resolve_parameters(["methanol", "water"])
    dw = DN.column(["Methanol", "Water"], f["T_K"], f["P_Pa"], {"Methanol": f["flow_mol_s"] * f["z"]["methanol"],
                   "Water": f["flow_mol_s"] * f["z"]["water"]}, 12, 7, o["reflux_ratio"], 101300.0,
                   bottoms_mol_s=o["bottoms"]["flow_mol_s"], params=params,
                   dwsim_names={"methanol": "Methanol", "water": "Water"}, poynting=False, enthalpy_mode="LiqCp_Excess")
    assert dw["ok"], dw["solver_errors"]
    held = dw["parameters_used"]["Methanol/Water"]
    assert held["b_ij_K"] == pytest.approx(params["tau_b_K"][0][1], rel=1e-9)
    xd = dw["distillate"]["mole_fractions"]["Methanol"]
    xb = dw["bottoms"]["mole_fractions"]["Water"]
    assert xd == pytest.approx(o["distillate"]["mole_fractions"]["methanol"], rel=0.01)
    assert xb == pytest.approx(o["bottoms"]["mole_fractions"]["water"], rel=0.01)
    assert dw["distillate"]["T_K"] == pytest.approx(o["distillate"]["T_K"], rel=0.01)
    assert dw["bottoms"]["T_K"] == pytest.approx(o["bottoms"]["T_K"], rel=0.01)
    assert dw["reboiler_duty_kW"] / 1e3 == pytest.approx(r["reboiler_duty_MW"], rel=0.01)
    # measured 2026-11-13 (LiqCp_Excess, no Poynting): x_D +0.06 %, Q_reb +0.29 %, Q_cond -0.62 %. Before SmoothVLE's
    # eps was continued to 1e-4 K, Q_cond was off by 1.23 % (phantom vapour in saturated liquids, see column_optimize)
    assert dw["condenser_duty_kW"] / 1e3 == pytest.approx(-r["condenser_duty_MW"], rel=0.01)


@slow
def test_regression_chains_into_the_column():
    """vle_binary_regression (Kamihama 2012, 101.3 kPa) → eo_column_optimize: the pair is labelled as regressed and
    the optimum is verified; ethanol below the azeotrope. The harder columns M15 left open (15 trays, x_D 0.75 and
    0.80) are in test_ethanol_water_columns_m15_left_open_converge_and_seal (M15b)."""
    reg = _ew_regression()
    specs = [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": 0.70},
             {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": 0.995}]
    r = C.eo_column_optimize(["ethanol", "water"], 100.0, 350.0, 101325.0, {"ethanol": 0.1, "water": 0.9}, 20, 14,
                             101325.0, specs, activity_model="NRTL", binary_parameters=reg)
    assert r["properties"]["binary_parameters"][0]["source_kind"] == "vle_binary_regression"
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert ev["component_balance_closed"] and ev["phantom_vapour_ok"]
    assert any("Psat" in w for w in r["warnings"])
    # M15 numbers, unchanged by the M15b start-up (spec continuation on the square system, then the NLP)
    assert r["optimum"]["reflux_ratio"] == pytest.approx(0.70081, rel=1e-4)
    assert r["reboiler_duty_MW"] == pytest.approx(1.030023, rel=1e-4)
    assert r["initial_point"]["spec_continuation"]["used"]
    he = r["properties"]["excess_enthalpy"]
    assert he["included_in_liquid_enthalpy"] and he["first_order_share_of_reboiler_duty"] == pytest.approx(-0.0313, abs=5e-4)


@slow
@pytest.mark.parametrize("xd, n, f, rr, q_reb_kW", [(0.70, 15, 10, 0.74817, 971.665), (0.75, 20, 14, 0.89482, 979.835),
                                                    (0.80, 20, 14, 1.55851, 1229.630)])
def test_ethanol_water_columns_m15_left_open_converge_and_seal(xd, n, f, rr, q_reb_kW):
    """M15b: bubble-point feed, 10 % ethanol. M15: 15 trays converged without the seal, x_D 0.75 did not converge from
    any start, x_D 0.80 kept phantom vapour. Now each one starts (non-ideality homotopy when the first start fails),
    reaches the specs by continuation on the square system and passes the whole seal. Reference values = notebook 26,
    where DWSIM («ExpData», same parameters, excess_enthalpy=False) agrees within 0.7 % on both duties."""
    reg = _ew_regression()
    p = NP.resolve_parameters(["ethanol", "water"], reg)
    T_bub = C._feed_two_phase_T({"z": EW_Z}, 101325.0, ["ethanol", "water"], p, 0.0)
    r = C.eo_column_optimize(["ethanol", "water"], 100.0, T_bub, 101325.0, EW_Z, n, f, 101325.0, EW_SPECS(xd),
                             activity_model="NRTL", binary_parameters=reg)
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert ev["component_balance_closed"] and ev["phantom_vapour_ok"] and ev["duals_consistent"]
    assert r["optimum"]["reflux_ratio"] == pytest.approx(rr, rel=1e-3)
    assert r["reboiler_duty_MW"] * 1e3 == pytest.approx(q_reb_kW, rel=1e-3)
    assert r["shortcut"]["R_needed_CMO"] < r["optimum"]["reflux_ratio"]
    assert all(v > 0 for v in r["duals_MW_per_unit_spec"].values())


@slow
def test_ethanol_water_matches_dwsim_like_for_like():
    """The M15 residual (Q_reb +6.5 % vs DWSIM «LiqCp_Excess») was DWSIM's excess term (−R·T²·Σ x·∂γ/∂T). With the same
    enthalpy model on both sides (no excess: excess_enthalpy=False vs «ExpData») the duties agree within 1 %."""
    try:
        from packages.dwsim_client import tools as T  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"DWSIM not available: {ex}")
    from packages.eo import dwsim_nrtl as DN
    reg = _ew_regression()
    p = NP.resolve_parameters(["ethanol", "water"], reg)
    r = C.eo_column_optimize(["ethanol", "water"], 100.0, 350.0, 101325.0, EW_Z, 20, 14, 101325.0, EW_SPECS(0.70),
                             activity_model="NRTL", binary_parameters=reg, excess_enthalpy=False)
    o = r["optimum"]
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert not r["properties"]["excess_enthalpy"]["included_in_liquid_enthalpy"]
    dw = DN.column(["Ethanol", "Water"], 350.0, 101325.0, {"Ethanol": 10.0, "Water": 90.0}, 22, 15, o["reflux_ratio"],
                   101325.0, bottoms_mol_s=o["bottoms"]["flow_mol_s"], params=p,
                   dwsim_names={"ethanol": "Ethanol", "water": "Water"}, poynting=False, enthalpy_mode="ExpData")
    assert dw["ok"], dw["solver_errors"]
    # measured 2026-11-13: Q_reb +0.20 %, Q_cond +0.09 %, x_D +1.13 % (pinched column; DWSIM's own Psat: T_D +0.30 K)
    assert dw["reboiler_duty_kW"] == pytest.approx(r["reboiler_duty_MW"] * 1e3, rel=0.01)
    assert dw["condenser_duty_kW"] == pytest.approx(-r["condenser_duty_MW"] * 1e3, rel=0.01)
    assert dw["distillate"]["mole_fractions"]["Ethanol"] == pytest.approx(o["distillate"]["mole_fractions"]["ethanol"], rel=0.015)
    assert dw["bottoms"]["mole_fractions"]["Water"] == pytest.approx(o["bottoms"]["mole_fractions"]["water"], rel=0.005)
