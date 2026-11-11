"""Stirred tank (bloque M6b): geometry, case files, post-processing and the correlation are checked fast; the full
MRF run (3 meshes, ~2 h) only with ALCHEMAI_SLOW_CFD=1 (its numbers are recorded in APP.md)."""
import math
import os
import tempfile
from pathlib import Path

import pytest

from packages.transport import stirred_tank as ST
from packages.transport.vessel_cfd import openfoam_available


def test_standard_geometry_matches_rushton_proportions():
    g = ST.rushton_geometry(0.3, 0.3, 0.1, 0.1)
    assert g["blade_height"] == pytest.approx(0.02) and g["blade_length"] == pytest.approx(0.025)
    assert g["disc_diameter"] == pytest.approx(0.075) and g["baffle_width"] == pytest.approx(0.03)
    assert g["blade_thickness"] == pytest.approx(0.003)
    # rotating zone wraps the blades, stays clear of the baffles and inside the liquid
    assert 0.05 < g["mrf_radius"] < 0.15 - 0.03
    assert g["C"] - g["mrf_height"] / 2 > 0 and g["C"] + g["mrf_height"] / 2 < g["H"]
    assert len(g["assumptions"]) == 2


def test_geometry_rejects_parts_that_do_not_fit():
    with pytest.raises(ValueError, match="no cabe"):
        ST.rushton_geometry(0.3, 0.3, 0.24, 0.1)
    with pytest.raises(ValueError, match="fondo"):
        ST.rushton_geometry(0.3, 0.3, 0.1, 0.005)
    with pytest.raises(ValueError, match="n_baffles"):
        ST.rushton_geometry(0.3, 0.3, 0.1, 0.1, n_baffles=5)
    g = ST.rushton_geometry(0.3, 0.3, 0.1, 0.1, baffle_width_m=0.025, blade_thickness_m=0.002)
    assert g["assumptions"] == []


def test_case_files_carry_the_speed_and_both_torque_monitors():
    g = ST.rushton_geometry(0.6, 0.6, 0.2, 0.2)
    with tempfile.TemporaryDirectory() as d:
        case = Path(d) / "c"
        ST._write_case(case, g, 40, 2.5, 1e-6, 998.0, 2000, 4)
        mrf = (case / "constant/MRFProperties").read_text()
        assert f"omega {2 * math.pi * 2.5}" in mrf
        ctl = (case / "system/controlDict").read_text()
        assert "patches (impeller shaft)" in ctl and "patches (tank baffles top)" in ctl and "endTime 2000" in ctl
        imp = (case / "constant/triSurface/impeller.stl").read_text()
        assert imp.count("facet normal") == 2 * (96 * 4) + 6 * 12      # disc + hub (side+caps) + 6 blade boxes
        assert (case / "constant/triSurface/baffles.stl").read_text().count("facet normal") == 4 * 12
        bm = (case / "system/blockMeshDict").read_text()
        assert "(43 43 43)" in bm                                     # (T + 2T/30) / (T/40) cells per side
        ST._write_case(Path(d) / "u", ST.rushton_geometry(0.6, 0.6, 0.2, 0.2, n_baffles=0), 40, 2.5, 1e-6, 998, 2000, 1)
        assert not (Path(d) / "u/constant/triSurface/baffles.stl").exists()
        assert "baffles" not in (Path(d) / "u/system/snappyHexMeshDict").read_text()


def test_power_number_from_a_moment_file():
    N, D, rho, Np = 5.0, 0.1, 998.0, 5.0
    Mz = Np * rho * N ** 3 * D ** 5 / (2 * math.pi * N)
    with tempfile.TemporaryDirectory() as d:
        dat = Path(d) / "moment.dat"
        lines = ["# Moment", "# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z"]
        for it in range(10, 1010, 10):
            wob = 1 + 0.02 * math.sin(it / 50)
            lines.append(f"{it} ({0.001} {0.002} {-Mz * wob}) (0 0 {-Mz * wob}) (0 0 0)")
        dat.write_text("\n".join(lines) + "\n")
        rows = ST.read_moment_z(dat)
    assert len(rows) == 100 and rows[0][0] == 10
    r = ST.power_number_from(rows, N, D, rho)
    assert r["Np"] == pytest.approx(Np, rel=0.02) and 0 < r["oscillation"] < 0.05
    assert r["iterations"] == 1000


def test_grenville_mixing_time_and_its_limits():
    m = ST.grenville_mixing_time(5.0, 5.0, 0.1, 0.3, 0.3, 5e4)
    assert m["N_theta95"] == pytest.approx(5.20 * 5.0 ** (-1 / 3) * 9)
    assert m["theta95_s"] == pytest.approx(m["N_theta95"] / 5.0) and "correlación" in m["source"]
    assert ST.grenville_mixing_time(5.0, 5.0, 0.1, 0.3, 0.45, 5e4)["theta95_s"] is None
    assert ST.grenville_mixing_time(5.0, 5.0, 0.1, 0.3, 0.3, 2000)["theta95_s"] is None


def test_tool_is_registered():
    from packages.dwsim_client import tools as T  # noqa: F401  (tools imports tools_transport)
    from packages.dwsim_client import tools_transport as TT
    assert TT.stirred_tank_cfd in TT.TRANSPORT_TOOLS
    with pytest.raises(ValueError):
        if not openfoam_available():
            raise ValueError("sin OpenFOAM")
        ST.stirred_tank_rushton(0.3, 0.3, 0.1, 0.1, 5.0, 998.0, 1e-3, cells_across_tank=40)


@pytest.mark.skipif(os.environ.get("ALCHEMAI_SLOW_CFD") != "1" or not openfoam_available(),
                    reason="corrida MRF completa con 3 mallas (~2 h): ALCHEMAI_SLOW_CFD=1")
def test_standard_rushton_power_number_is_verified():
    r = ST.stirred_tank_rushton(0.3, 0.3, 0.1, 0.1, 5.0, 998.0, 1.0e-3)
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert 4.0 < r["Np"] < 6.0 and ev["torque_imbalance"] < 0.03
    assert ev["mesh_ok"] and r["Np_uncertainty_fraction"] < ST.BAND_TOL


def test_grid_convergence_recovers_an_exact_second_order_series():
    cells = [8000, 1000, 125]
    phi = [1 + (n ** (-1 / 3)) ** 2 for n in cells]
    g = ST.grid_convergence(phi, cells)
    assert g["monotonic"] and g["p_observed"] == pytest.approx(2.0, abs=1e-6)
    assert g["phi_extrapolated"] == pytest.approx(1.0, abs=1e-9)


def test_grid_convergence_flags_oscillation_and_caps_the_order():
    assert not ST.grid_convergence([4.9, 4.7, 4.95], [885278, 457727, 220360])["monotonic"]
    # recorded standard-Rushton series (APP.md, 2026-11-10): 64 / 50 / 38 cells across the tank
    g = ST.grid_convergence([4.9202, 4.7340, 4.2438], [885278, 457727, 220360])
    assert g["p_observed"] > 3 and g["p_used"] == 2.0
    assert g["gci_fine"] == pytest.approx(0.0856, abs=0.001) and g["phi_extrapolated"] == pytest.approx(5.257, abs=0.005)


def test_recorded_end_to_end_series_is_rejected():
    # end-to-end tool run 2026-11-10, 64 / 49 / 38 cells: blade snapping makes Np jump (49 → 4.52, 50 → 4.73), so the
    # series is not in the asymptotic range (R > 1) and the seal must not be given
    g = ST.grid_convergence([4.8944, 4.5216, 4.2876], [885278, 431656, 220360])
    assert not g["monotonic"] and g["R"] > 1 and g["gci_fine"] is None


def test_mesh_band_is_the_largest_deviation_from_the_fine_mesh():
    b = ST.mesh_band([5.0, 4.8, 5.3])
    assert b["uncertainty"] == pytest.approx(0.06) and b["min"] == 4.8 and b["max"] == 5.3
    # level-2 blade series of 2026-11-10 (64 / 49 / 38): 12.4 % scatter, no seal
    assert ST.mesh_band([4.8944, 4.5216, 4.2876])["uncertainty"] > ST.BAND_TOL
    # level-3 blade, verified end-to-end run of 2026-11-10 (APP.md): 3.5 % band, sealed
    assert ST.mesh_band([4.8531, 5.0223, 4.7038])["uncertainty"] == pytest.approx(0.0349, abs=0.001)
