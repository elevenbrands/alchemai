"""Tubular reactor as a FEniCSx field (bloque M1): evidence logic without FEniCSx, and the Case 1 run with it."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.transport import reactor_field as RF  # noqa: E402

# Case 1 of transporte-lab: n-butane isomerisation, A = 1e8 1/s, E = 1e5 J/mol, 600 K → k = 0.1970 1/s; 10 mol/s at 1 bar
# → Q = 0.4988 m³/s; V = 4 m³ as 1 228 tubes × 2 cm (Re 2000 with ρ 1.16, μ 1.5e-5); DWSIM PFR X = 0.7945.
CASE = dict(volumetric_flow_m3_s=10 * 8.314462618 * 600 / 1e5, tube_diameter_m=0.02, n_tubes=1228, volume_m3=4.0,
            A_1_s=1e8, E_J_mol=1e5, T_K=600.0, diffusivity_m2_s=1e-5, density_kg_m3=1.16, viscosity_Pa_s=1.5e-5)
X_DWSIM_PFR = 0.7945

GOOD = {"X_plug_analytic": 0.7940, "X_plug_fenicsx": 0.79400, "X_laminar_fenicsx": 0.7820,
        "X_laminar_fenicsx_fine_mesh": 0.7819, "X_laminar_segregated_analytic": 0.7073}


def test_arrhenius_and_bundle_geometry():
    k = RF.arrhenius_k(1e8, 1e5, 600.0)
    assert k == pytest.approx(0.1970, abs=5e-4)
    g = RF.bundle_geometry(CASE["volumetric_flow_m3_s"], 0.02, 1228, volume_m3=4.0)
    assert g["tube_length_m"] == pytest.approx(10.37, abs=0.01) and g["u_mean_m_s"] == pytest.approx(1.293, abs=0.002)
    assert g["volume_m3"] == pytest.approx(4.0) and g["tau_s"] == pytest.approx(8.02, abs=0.01)
    with pytest.raises(ValueError):
        RF.bundle_geometry(1.0, 0.02, 10)                       # neither length nor volume
    with pytest.raises(ValueError):
        RF.bundle_geometry(1.0, 0.02, 10, tube_length_m=1.0, volume_m3=1.0)


def test_evidence_verified_only_with_all_four_checks():
    ev = RF.evidence_from(GOOD, Re=2000.0)
    assert ev["verified"] and ev["reasons"] == [] and ev["laminar_regime"]
    assert not RF.evidence_from(GOOD, Re=None)["verified"]                       # regime unknown
    assert "Re = 5000" in RF.evidence_from(GOOD, Re=5000.0)["reasons"][0]        # turbulent: Poiseuille does not apply
    bad_plug = dict(GOOD, X_plug_fenicsx=0.80)
    assert not RF.evidence_from(bad_plug, Re=2000.0)["plug_matches_analytic"]
    bad_mesh = dict(GOOD, X_laminar_fenicsx_fine_mesh=0.79)
    assert not RF.evidence_from(bad_mesh, Re=2000.0)["mesh_independent"]
    out_of_bounds = dict(GOOD, X_laminar_fenicsx=0.60)
    assert not RF.evidence_from(out_of_bounds, Re=2000.0)["within_bounds"]


def test_tool_requires_kinetics_and_positive_inputs():
    with pytest.raises(ValueError):
        RF.tubular_reactor_field(1.0, 0.02, 10, 1e-5, tube_length_m=1.0)          # no k, no Arrhenius
    with pytest.raises(ValueError):
        RF.tubular_reactor_field(1.0, 0.02, 10, -1e-5, tube_length_m=1.0, k_1_s=0.1)


def test_missing_fenicsx_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(RF, "fenicsx_python", lambda: None)
    with pytest.raises(RuntimeError, match="FEniCSx no está instalado"):
        RF.run_solver({"nz": 10, "nr": 2})


def test_timeout_kills_the_solver_group(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import time, sys; sys.stdin.read(); time.sleep(30)")
    orig = RF.SOLVER
    RF.SOLVER = slow
    try:
        with pytest.raises(RuntimeError, match="no terminó"):
            RF.run_solver({"nz": 1, "nr": 1}, timeout_s=1, python=sys.executable)
    finally:
        RF.SOLVER = orig


@pytest.mark.skipif(not RF.fenicsx_available(), reason="FEniCSx environment not installed")
def test_case1_bundle_matches_transporte_lab(tmp_path):
    fig = tmp_path / "reactor.png"
    r = RF.tubular_reactor_field(**CASE, nz=150, nr=15, figure_path=str(fig))
    assert r["X_plug_flow"] == pytest.approx(1 - math.exp(-r["k_1_s"] * r["geometry"]["tau_s"]))
    assert r["X_plug_flow"] == pytest.approx(X_DWSIM_PFR, abs=1e-3)              # same ideal reactor as DWSIM
    assert r["X_laminar"] == pytest.approx(0.7820, abs=2e-3)                       # transporte-lab Case 1
    assert r["X_laminar_segregated"] == pytest.approx(0.7073, abs=1e-3)
    assert r["X_laminar_segregated"] < r["X_laminar"] < r["X_plug_flow"]
    assert r["Re"] == pytest.approx(2000, abs=5) and r["evidence"]["verified"], r["evidence"]
    assert r["laminar_penalty_points"] == pytest.approx(1.2, abs=0.2)
    prof = r["outlet_profile"]
    assert len(prof["r_m"]) == 60 and prof["X"][0] < prof["X"][-1]              # centre-line less converted than the wall
    assert fig.is_file() and fig.stat().st_size > 10_000
    assert r["engine"].startswith("FEniCSx dolfinx")
