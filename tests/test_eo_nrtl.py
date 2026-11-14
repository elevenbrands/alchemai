"""M15: equation-oriented column with an NRTL liquid (packages/eo/nrtl_properties.py).

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
    the optimum is verified; ethanol below the azeotrope. x_D 0.70, not 0.80: at 0.80 the top trays sit in the
    azeotrope's pinch (Tdew − Tbub ≈ 0.01 K) and SmoothVLE leaves 0.14 % phantom vapour at eps 1e-4 K, so the tool
    (correctly) withholds the seal there. 20 trays / feed on 14: with 15 trays the start-up does not converge at 0.70
    (open item, NEXT_STEP)."""
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
