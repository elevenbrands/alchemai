"""Blasius laminar boundary layer (geometry='flat_plate' of `non_newtonian_pipe_flow`).

Reference values (Blasius 1908; Howarth 1938; White, Viscous Fluid Flow, table 4-1; Schlichting): f''(0) = 0.33206,
delta_99 = 4.91 x/sqrt(Re_x), delta* = 1.7208 x/sqrt(Re_x), theta = 0.664 x/sqrt(Re_x), c_f = 0.664/sqrt(Re_x),
C_D = 1.328/sqrt(Re_L); tabulated profile f'(1) = 0.32979, f'(2) = 0.62977, f'(3) = 0.84605, f'(5) = 0.99155."""
import json
import math

import pytest

from packages.dwsim_client import tools_particles as TP
from packages.transport import boundary_layer as BL

AIR = {"model": "newtonian", "viscosity_Pa_s": 1.8e-5}


def test_blasius_constants():
    b = BL.blasius_solution()
    assert b["fpp0"] == pytest.approx(0.33206, abs=5e-6)
    assert b["eta_99"] == pytest.approx(4.91, abs=5e-3)
    assert b["displacement_coefficient"] == pytest.approx(1.7208, abs=5e-5)
    assert b["momentum_coefficient"] == pytest.approx(0.664, abs=5e-4)
    assert 2 * b["momentum_coefficient"] == pytest.approx(1.328, abs=5e-4)
    assert b["displacement_coefficient"] / b["momentum_coefficient"] == pytest.approx(2.59, abs=5e-3)   # shape factor
    for eta, fp in ((1.0, 0.32979), (2.0, 0.62977), (3.0, 0.84605), (5.0, 0.99155)):
        assert BL.similarity_profile(eta)[1] == pytest.approx(fp, abs=2e-5)
    assert BL.similarity_profile(0.0) == (0.0, 0.0, pytest.approx(0.33206, abs=5e-6))
    assert BL.similarity_profile(50.0)[1] == 1.0


def test_flat_plate_thicknesses_friction_and_drag():
    U, rho, mu, L, W, x = 2.0, 1.2, 1.8e-5, 0.6, 0.4, 0.3
    r = TP.non_newtonian_pipe_flow(density_kg_m3=rho, rheology=AIR, geometry="flat_plate", free_stream_velocity_m_s=U,
                                   length_m=L, width_m=W, positions_m=[x])
    assert r["evidence"]["verified"] and r["warnings"] == []
    p = r["points"][0]
    re_x = rho * U * x / mu
    assert p["reynolds_x"] == pytest.approx(re_x)
    assert p["delta_99_m"] == pytest.approx(4.91 * x / math.sqrt(re_x), rel=1e-3)
    assert p["displacement_thickness_m"] == pytest.approx(1.7208 * x / math.sqrt(re_x), rel=1e-4)
    assert p["momentum_thickness_m"] == pytest.approx(0.664 * x / math.sqrt(re_x), rel=1e-3)
    assert p["local_skin_friction_coefficient"] == pytest.approx(0.664 / math.sqrt(re_x), rel=1e-3)
    assert p["wall_shear_stress_Pa"] == pytest.approx(0.332 * rho * U ** 2 / math.sqrt(re_x), rel=1e-3)
    pl = r["plate"]
    re_L = rho * U * L / mu
    assert pl["reynolds_L"] == pytest.approx(re_L) and re_L < 5e5
    assert pl["average_skin_friction_coefficient"] == pytest.approx(1.328 / math.sqrt(re_L), rel=1e-3)
    drag = 1.328 / math.sqrt(re_L) * 0.5 * rho * U ** 2 * L * W
    assert pl["drag_one_side_N"] == pytest.approx(drag, rel=1e-3)
    assert pl["drag_both_sides_N"] == pytest.approx(2 * drag, rel=1e-3)
    # momentum balance: drag per unit width of one side = rho U^2 theta(L)
    assert pl["drag_per_unit_width_one_side_N_m"] == pytest.approx(rho * U ** 2 * pl["trailing_edge"]["momentum_thickness_m"],
                                                                   rel=1e-10)
    # average coefficient is twice the local one at the trailing edge
    assert pl["average_skin_friction_coefficient"] == pytest.approx(
        2 * pl["trailing_edge"]["local_skin_friction_coefficient"], rel=1e-12)
    json.dumps(r, ensure_ascii=False)


def test_velocity_at_a_point_from_the_similarity_profile():
    U, rho, mu, x = 1.5, 998.0, 1.0e-3, 0.2
    nu = mu / rho
    ys = [eta * math.sqrt(nu * x / U) for eta in (1.0, 3.0, 5.0)]
    r = BL.flat_plate(U, rho, mu, positions_m=[[x, y] for y in ys] + [[x, 0.0], [x, 1.0]])
    pts = r["points"]
    for p, eta, fp in zip(pts, (1.0, 3.0, 5.0), (0.32979, 0.84605, 0.99155)):
        assert p["eta"] == pytest.approx(eta, rel=1e-12)
        assert p["u_over_U"] == pytest.approx(fp, abs=2e-5) and p["u_m_s"] == pytest.approx(fp * U, abs=4e-5)
    assert pts[3]["u_m_s"] == 0.0 and pts[3]["v_m_s"] == 0.0                              # no slip at the wall
    assert pts[3]["shear_stress_Pa"] == pytest.approx(pts[3]["wall_shear_stress_Pa"], rel=1e-10)
    assert pts[4]["u_over_U"] == 1.0                                                      # far outside the layer
    # outer normal velocity v/U -> 0.8604/sqrt(Re_x) (= delta* coefficient / 2)
    assert pts[4]["v_m_s"] / U * math.sqrt(pts[4]["reynolds_x"]) == pytest.approx(1.7208 / 2, abs=5e-5)
    # u = 0.99 U exactly at y = delta_99
    edge = BL.flat_plate(U, rho, mu, positions_m=[[x, pts[0]["delta_99_m"]]])["points"][0]
    assert edge["u_over_U"] == pytest.approx(0.99, abs=1e-9)


def test_transition_warning_and_refusals():
    r = BL.flat_plate(20.0, 1.2, 1.8e-5, length_m=1.0, positions_m=[0.1, 0.9])
    assert r["plate"]["reynolds_L"] > 5e5 and not r["evidence"]["verified"]
    assert r["transition"]["x_transition_m"] == pytest.approx(5e5 * 1.8e-5 / 1.2 / 20.0)
    assert r["points"][0]["laminar"] and not r["points"][1]["laminar"]
    assert any("5e5" in w for w in r["warnings"])
    assert "drag_one_side_N" not in r["plate"] and "drag_per_unit_width_one_side_N_m" in r["plate"]   # no width given
    with pytest.raises(ValueError, match="free_stream_velocity_m_s"):
        TP.non_newtonian_pipe_flow(density_kg_m3=1.2, rheology=AIR, geometry="flat_plate", length_m=1.0)
    with pytest.raises(ValueError, match="newtoniano"):
        TP.non_newtonian_pipe_flow(density_kg_m3=1.2, rheology={"model": "power_law", "K_Pa_sn": 1.0, "n": 0.5},
                                   geometry="flat_plate", free_stream_velocity_m_s=1.0, length_m=1.0)
    with pytest.raises(ValueError, match="positions_m"):
        BL.flat_plate(1.0, 1.2, 1.8e-5)
    with pytest.raises(ValueError, match="y debe ser"):
        BL.flat_plate(1.0, 1.2, 1.8e-5, positions_m=[[0.1, -0.01]])
    with pytest.raises(ValueError, match="density_kg_m3"):
        BL.flat_plate(1.0, None, 1.8e-5, length_m=1.0)


def test_wrapper_flat_plate():
    from packages.agent import wrappers_particles as W
    out = json.loads(W.AGENT_WRAPPERS[0](geometry="flat_plate", free_stream_velocity_m_s=2.0, density_kg_m3=998.0,
                                         rheology={"model": "newtonian", "viscosity_Pa_s": 1e-3}, length_m=0.2,
                                         width_m=0.5, positions_m=[[0.1, 0.0005]]))
    re_L = 998.0 * 2.0 * 0.2 / 1e-3
    assert out["plate"]["drag_one_side_N"] == pytest.approx(1.328 / math.sqrt(re_L) * 0.5 * 998.0 * 4.0 * 0.1, rel=1e-3)
    assert 0.0 < out["points"][0]["u_over_U"] < 1.0 and out["evidence"]["verified"]
