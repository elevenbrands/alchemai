"""Bell-Delaware shell side (bloque M4c). No worked example with legible inputs was available, so the checks are
independent of the implementation in other ways: the J factors read from the digitized Delaware charts against the
HEDH closed-form fits, the ideal bank by two correlations, the no-leakage/no-bypass limit that must collapse to
h_ideal·Jc, the TEMA clearance tables, and the direction every correction must move."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_heat as TH  # noqa: E402

# water on the shell side of a 584 mm (23 in) shell, 19 mm tubes on a 25.4 mm triangular pitch, fixed tubesheet
FLUID = dict(mass_flow_kg_s=11.0, density_kg_m3=995.0, viscosity_Pa_s=0.000803, thermal_conductivity_W_mK=0.62,
             cp_J_kgK=4180.0, wall_viscosity_Pa_s=0.000657)
SHELL = dict(shell_diameter_m=0.584, tube_od_m=0.019, tube_pitch_m=0.0254, layout_angle=30, baffle_cut_fraction=0.25,
             baffle_spacing_m=0.2, n_baffles=20, outer_tube_limit_diameter_m=0.572, n_tubes=370)


def bd(**kw):
    return TH.shell_side_bell_delaware(**{**FLUID, **SHELL, **kw})


def test_standard_design_and_independent_checks():
    r = bd()
    ev = r["evidence"]
    assert ev["verified"] and not ev["reasons"]
    assert all(abs(v) < 10 for v in ev["J_chart_vs_HEDH_fit_pct"].values())          # charts vs HEDH fits
    assert abs(ev["ideal_bank_Zukauskas_vs_ESDU_pct"]) < 30
    J = r["J"]
    assert J["product"] == pytest.approx(J["Jc_baffle_cut"] * J["Jl_leakage"] * J["Jb_bypass"] * J["Js_end_spacing"]
                                         * J["Jr_laminar"])
    assert 0.5 < J["product"] < 0.9 and r["h_W_m2K"] == pytest.approx(r["h_ideal_tube_bank_W_m2K"] * J["product"])
    d = r["dP_breakdown_Pa"]
    assert r["dP_Pa"] == pytest.approx(d["crossflow"] + d["windows"] + d["end_zones"])
    # crossflow area at the centreline written out: B·[(Ds − Dotl) + (Dotl − Do)/Pt·(Pt − Do)]
    Sm = 0.2 * ((0.584 - 0.572) + (0.572 - 0.019) / 0.0254 * (0.0254 - 0.019))
    assert r["geometry"]["crossflow_area_m2"] == pytest.approx(Sm, rel=1e-12)
    assert r["Re"] == pytest.approx(0.019 * 11.0 / Sm / 0.000803, rel=1e-12)
    assert r["kern"]["h_W_m2K"] > 0 and r["kern"]["kern_over_bell_delaware_dP"] > 1      # Kern overpredicts ΔP here


def test_no_leakage_no_bypass_collapses_to_ideal_bank_times_Jc():
    r = bd(outer_tube_limit_diameter_m=0.584, shell_baffle_clearance_m=0.0, tube_baffle_clearance_m=0.0)
    J, R = r["J"], r["R"]
    assert (J["Jl_leakage"], J["Jb_bypass"], J["Js_end_spacing"], J["Jr_laminar"]) == (1.0, 1.0, 1.0, 1.0)
    assert (R["Rl_leakage"], R["Rb_bypass"], R["Rs_end_zones"]) == (1.0, 1.0, 2.0)
    assert r["h_W_m2K"] == pytest.approx(r["h_ideal_tube_bank_W_m2K"] * J["Jc_baffle_cut"], rel=1e-12)
    assert 0.52 < J["Jc_baffle_cut"] < 1.15


def test_tema_clearance_defaults_are_listed():
    r = bd()
    g = r["geometry"]
    assert g["shell_baffle_clearance_m"] == pytest.approx(0.0048)        # TEMA RCB-4.3, shell 18-39 in: 3/16 in
    assert g["tube_baffle_clearance_m"] == pytest.approx(0.0008)         # TEMA RCB-4.2, span <= 36 in: 1/32 in
    assert len(r["assumptions"]) == 2 and all("TEMA" in a for a in r["assumptions"])
    est = bd(n_tubes=None)
    assert est["geometry"]["n_tubes"] > 300 and "estimado" in est["assumptions"][0]


def test_corrections_move_the_right_way():
    base = bd()
    loose = bd(shell_baffle_clearance_m=0.0096, tube_baffle_clearance_m=0.0016)
    assert loose["J"]["Jl_leakage"] < base["J"]["Jl_leakage"] and loose["dP_Pa"] < base["dP_Pa"]
    bypass = bd(outer_tube_limit_diameter_m=0.584 - 0.09)                      # pull-through floating head gap
    sealed = bd(outer_tube_limit_diameter_m=0.584 - 0.09, sealing_strip_pairs=2)
    assert bypass["J"]["Jb_bypass"] < sealed["J"]["Jb_bypass"] < 1 and bypass["R"]["Rb_bypass"] < sealed["R"]["Rb_bypass"]
    assert any("tiras de sello" in w for w in bypass["warnings"])
    wide_ends = bd(inlet_baffle_spacing_m=0.3, outlet_baffle_spacing_m=0.3)
    assert wide_ends["J"]["Js_end_spacing"] < 1 and wide_ends["R"]["Rs_end_zones"] < 2


def test_laminar_and_range_flags():
    lam = bd(viscosity_Pa_s=0.2, wall_viscosity_Pa_s=None)
    assert lam["regime"].startswith("laminar") and lam["J"]["Jr_laminar"] < 1
    assert not bd(baffle_cut_fraction=0.10)["evidence"]["verified"]
    wide = bd(tube_pitch_m=0.019 * 1.6)
    assert not wide["evidence"]["verified"] and "1.25-1.5" in wide["evidence"]["reasons"][0]
    with pytest.raises(ValueError, match="layout_angle"):
        bd(layout_angle=60)
    with pytest.raises(ValueError, match="outer_tube_limit"):
        bd(outer_tube_limit_diameter_m=0.6)


def test_all_layouts_plain_json():
    for angle in (30, 45, 90):
        r = bd(layout_angle=angle)
        assert r["evidence"]["verified"], (angle, r["evidence"]["reasons"])
        json.dumps(r, allow_nan=False)
    assert math.isfinite(bd(layout_angle=45)["h_W_m2K"])
