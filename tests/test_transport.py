"""Transport phenomena (bloque 18): pure correlations + one DWSIM pipe cross-check."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.transport.hydraulics import hx_area, lmtd, overall_U, pipe_dp, size_pipe, tube_film_coefficient  # noqa: E402


def test_laminar_is_hagen_poiseuille():
    r = pipe_dp(rho=900.0, mu=0.5, mass_flow_kg_s=0.5, D_m=0.05, L_m=10.0)
    v = r["velocity_m_s"]
    assert r["regime"] == "laminar" and r["dP_friction_Pa"] == pytest.approx(32 * 0.5 * v * 10.0 / 0.05 ** 2, rel=1e-12)


def test_turbulent_colebrook_and_static_head():
    r = pipe_dp(998.2, 1.002e-3, 10.0, 0.10226, 100.0, elevation_change_m=10.0)
    f, eD, Re = r["friction_factor_darcy"], 4.57e-5 / 0.10226, r["Re"]
    assert 1 / math.sqrt(f) == pytest.approx(-2 * math.log10(eD / 3.7 + 2.51 / (Re * math.sqrt(f))), rel=1e-9)
    assert r["dP_static_Pa"] == pytest.approx(998.2 * 9.80665 * 10.0)


def test_size_pipe_skips_uncommon_sizes():
    s = size_pipe(998.2, 1e-3, 10.0, 2.0)          # needs Di ≥ 79.9 mm: 3 in (77.9) too small, 3½ in skipped → 4 in
    assert s["NPS_in"] == 4 and s["velocity_m_s"] <= 2.0


def test_hx_area_and_cross():
    a = hx_area(1.5e6, 360, 320, 300, 315, 850, shell_passes=0)
    assert a["F"] == 1.0 and a["area_m2"] == pytest.approx(1.5e6 / (850 * lmtd(360, 320, 300, 315)))
    a2 = hx_area(1.5e6, 360, 320, 300, 315, 850, shell_passes=1)
    assert 0.85 < a2["F"] < 0.9 and a2["area_m2"] > a["area_m2"]
    with pytest.raises(ValueError):
        lmtd(350, 300, 310, 355)


def test_film_coefficient_and_U():
    h = tube_film_coefficient(998.2, 1.0e-3, 0.6, 4182.0, 1.5, 0.02)
    assert h["correlation"] == "Gnielinski" and 4000 < h["h_W_m2K"] < 9000
    U = overall_U(h["h_W_m2K"], 1500.0, 0.02, 0.025, fouling_inner_m2K_W=1.8e-4, fouling_outer_m2K_W=1.8e-4)
    assert 400 < U < 1500


def test_dwsim_pipe_matches_darcy():
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_transport import pipe_hydraulics
    fid = T.new_flowsheet("t_pipe", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "in", temperature_K=293.15, pressure_Pa=5e5, mass_flow_kg_s=10.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "out")
    T.add_unit(fid, "pipe", "L-1", {"length_m": 100.0, "inner_diameter_m": 0.10226, "elevation_change_m": 5.0})
    T.connect(fid, "L-1", feeds=["in"], products=["out"])
    assert T.solve(fid)["solved"]
    u = T.read_unit(fid, "L-1")
    ours = pipe_hydraulics(fid, "in", 100.0, 0.10226, elevation_change_m=5.0)
    assert ours["dP_friction_Pa"] == pytest.approx(u["dP_friction_Pa"], rel=0.01)
    assert ours["dP_static_Pa"] == pytest.approx(u["dP_static_Pa"], rel=0.01)
    T.close_flowsheet(fid)
