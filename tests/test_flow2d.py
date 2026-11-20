"""laminar_flow_2d (FEniCSx Taylor-Hood Navier-Stokes on user geometry): every case against an exact or published value.

Vortex shedding at Re = 100 takes ~5 min and runs only with ALCHEMAI_SLOW_FLOW2D=1."""
import json
import math
import os

import pytest

from packages.transport import flow2d as F

needs_fenicsx = pytest.mark.skipif(not F.fenicsx_available(), reason="FEniCSx environment not installed")
slow = pytest.mark.skipif(not (F.fenicsx_available() and os.environ.get("ALCHEMAI_SLOW_FLOW2D") == "1"),
                          reason="slow transient: set ALCHEMAI_SLOW_FLOW2D=1 (needs FEniCSx)")

IN_OUT = [{"name": "in", "type": "inlet", "on": {"side": "left"}, "mean_velocity": 0.01},
          {"name": "out", "type": "outlet", "on": {"side": "right"}}]
WATER = {"density_kg_m3": 1000.0, "viscosity_Pa_s": 1e-3}


# ---------------------------------------------------------------- validation (no FEniCSx needed)
def test_rejects_bad_requests_before_launching():
    geo = {"mode": "planar", "rectangles": [[0, 0, 1, 0.1]]}
    with pytest.raises(ValueError, match="mode"):
        F.laminar_flow_2d({"mode": "3d", "rectangles": [[0, 0, 1, 1]]}, IN_OUT, **WATER)
    with pytest.raises(ValueError, match="y < 0"):
        F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[0, -1, 1, 1]]}, IN_OUT, **WATER)
    with pytest.raises(ValueError, match="density_kg_m3"):
        F.laminar_flow_2d(geo, IN_OUT)
    with pytest.raises(ValueError, match="no hay ninguna frontera"):
        F.laminar_flow_2d(geo, [{"name": "out", "type": "outlet", "on": {"side": "right"}}], **WATER)
    with pytest.raises(ValueError, match="selector"):
        F.laminar_flow_2d(geo, [{"name": "in", "type": "inlet", "on": {"where": "left"}, "mean_velocity": 1}], **WATER)
    # inlet expressions go through the AST whitelist: no calls to anything but the maths functions, no attributes
    for bad in ("__import__('os').system('true')", "x.real", "open('f')", "[1][0]"):
        with pytest.raises(ValueError):
            F.laminar_flow_2d(geo, [{"name": "in", "type": "inlet", "on": {"side": "left"}, "velocity": [bad, 0]}], **WATER)


def test_solver_side_expression_evaluator_is_a_whitelist():
    """The solver's own evaluator (it cannot import the repo) is loaded by path: pure numpy/ast, no dolfinx at import."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("flow2d_geometry", F.SOLVER.with_name("flow2d_geometry.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    tree = G.parse_expr("1.5*(1-(y/0.5)^2) + sin(pi*x)*0", {"x", "y"})
    assert G.eval_expr(tree, {"x": 2.0, "y": 0.25}) == pytest.approx(1.125)
    for bad in ("__import__('os')", "x.real", "(lambda: 1)()", "abs(x)", "q + 1"):
        with pytest.raises((ValueError, SyntaxError)):
            G.parse_expr(bad, {"x", "y"})


def test_missing_fenicsx_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(F.RF, "fenicsx_python", lambda: None)
    with pytest.raises(RuntimeError, match="FEniCSx no está instalado"):
        F.laminar_flow_2d({"mode": "planar", "rectangles": [[0, 0, 1, 0.1]]}, IN_OUT, **WATER)


def test_timeout_is_an_honest_failure(tmp_path, monkeypatch):
    slow_script = tmp_path / "slow.py"
    slow_script.write_text("import time, sys; sys.stdin.read(); time.sleep(30)")
    monkeypatch.setattr(F, "SOLVER", slow_script)
    monkeypatch.setattr(F.RF, "fenicsx_python", lambda: __import__("sys").executable)
    with pytest.raises(RuntimeError, match="no terminó en 1 s"):
        F.laminar_flow_2d({"mode": "planar", "rectangles": [[0, 0, 1, 0.1]]}, IN_OUT, **WATER, timeout_s=1)


def test_tool_and_wrapper_share_signature_and_docstring():
    import inspect

    from packages.agent import wrappers_flow2d as W
    from packages.dwsim_client import tools_flow2d as T
    assert T.FLOW2D_TOOLS == [T.laminar_flow_2d] and len(W.AGENT_WRAPPERS) == 1
    w = W.AGENT_WRAPPERS[0]
    assert w.name == "laminar_flow_2d"
    assert list(w.to_dict()["input_schema"]["properties"]) == list(inspect.signature(T.laminar_flow_2d).parameters)
    assert "geometry" in T.laminar_flow_2d.__doc__ and "Example" in T.laminar_flow_2d.__doc__


# ---------------------------------------------------------------- exact solutions
@needs_fenicsx
def test_plane_poiseuille_is_exact():
    """Plane Poiseuille (exact Navier-Stokes solution): −dp/dx = 12 μ ū / H², u_centre = 1.5 ū."""
    H, L, U = 0.01, 0.1, 0.01
    r = F.laminar_flow_2d({"mode": "planar", "rectangles": [[0, 0, L, H]]}, IN_OUT, **WATER, points=[[0.08, H / 2], [0.08, H / 4]])
    g = 12 * 1e-3 * U / H ** 2
    assert r["Re"] == pytest.approx(100)
    assert r["pressure_drops"]["in->out"] == pytest.approx(g * L, rel=1e-6)
    s = r["straight_sections"][0]
    assert s["pressure_drop_per_length"] == pytest.approx(g, rel=1e-6) and s["developed"]
    assert s["sides"] == ["wall", "wall"] and s["u_max_over_mean_at_end"] == pytest.approx(1.5, rel=1e-6)
    assert r["points"][0]["velocity"][0] == pytest.approx(1.5 * U, rel=1e-6)
    assert r["points"][1]["velocity"][0] == pytest.approx(1.5 * U * 0.75, rel=1e-6)          # 1 − (1/2)²
    assert r["boundaries"]["out"]["flow_out"] == pytest.approx(U * H, rel=1e-9)               # per unit depth
    ev = r["evidence"]
    assert ev["verified"] and ev["poiseuille_self_check"] and ev["mass_imbalance_rel"] < 1e-10 and ev["mesh_converged"]
    assert not r["recirculation"]["exists"]


@needs_fenicsx
def test_hagen_poiseuille_is_exact(tmp_path):
    """Hagen-Poiseuille in a tube (axisymmetric): −dp/dz = 8 μ ū / R², u_axis = 2 ū, Q = π R² ū."""
    R, L, U = 0.005, 0.1, 0.01
    fig = tmp_path / "tube.png"
    r = F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[0, 0, L, R]]}, IN_OUT, **WATER, points=[[0.08, 0.0]],
                          figure_path=str(fig))
    g = 8 * 1e-3 * U / R ** 2
    assert r["Re"] == pytest.approx(100) and "diámetro" in r["reference"]["length_definition"]
    assert r["pressure_drops"]["in->out"] == pytest.approx(g * L, rel=1e-5)
    s = r["straight_sections"][0]
    assert s["pressure_drop_per_length"] == pytest.approx(g, rel=1e-5) and s["sides"] == ["symmetry", "wall"]
    assert r["points"][0]["velocity"][0] == pytest.approx(2 * U, rel=1e-5)
    assert r["boundaries"]["out"]["flow_out"] == pytest.approx(math.pi * R ** 2 * U, rel=1e-9)
    assert r["evidence"]["verified"] and fig.is_file() and fig.stat().st_size > 10_000
    json.dumps(r, default=float)


@needs_fenicsx
def test_entry_flow_develops_to_poiseuille():
    """Uniform inlet in a tube at Re = 100: the profile develops to Poiseuille; the 99 % centre-line entry length follows
    Durst et al., J. Fluids Eng. 127 (2005) 1154: Le/D = [0.619^1.6 + (0.0567 Re)^1.6]^(1/1.6) = 5.77 (Langhaar: 0.0575 Re)."""
    D, U, Re = 0.01, 0.01, 100
    b = [dict(IN_OUT[0], profile="uniform"), IN_OUT[1]]
    r = F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[0, 0, 12 * D, D / 2]]}, b, **WATER,
                          points=[[0.0005, 0.0], [11 * D, 0.0]])
    s = r["straight_sections"][0]
    durst = (0.619 ** 1.6 + (0.0567 * Re) ** 1.6) ** (1 / 1.6)
    assert s["entry_length_99"] / D == pytest.approx(durst, rel=0.08)
    assert s["centreline_velocity_start_over_developed"] < 0.6                    # starts flat: u_axis ≈ ū = 0.5·(2 ū)
    assert r["points"][1]["velocity"][0] == pytest.approx(2 * U, rel=5e-3)        # developed at 11 D
    assert s["developed"] and s["poiseuille_error_rel"] == pytest.approx(0.0, abs=5e-3)
    assert r["boundaries"]["in"]["flow_out"] == pytest.approx(-math.pi * D ** 2 / 4 * U, rel=1e-9)   # discrete inlet flow exact
    assert r["pressure_drops"]["in->out"] > 8 * 1e-3 * U / (D / 2) ** 2 * 12 * D   # entry loss on top of Poiseuille
    assert r["evidence"]["poiseuille_self_check"]


@needs_fenicsx
def test_tee_junction_conserves_mass_and_recovers_developed_gradient():
    """Two inlets merging in a T (planar, H = 1): outlet flow = sum of inlets (continuity) and, downstream, the fully
    developed gradient is the plane-Poiseuille value for the SUM of the flows, 12 μ (Q1 + Q2) / H³."""
    b = [{"name": "main", "type": "inlet", "on": {"side": "left"}, "mean_velocity": 1.0},
         {"name": "branch", "type": "inlet", "on": {"side": "top"}, "mean_velocity": 0.5},
         {"name": "out", "type": "outlet", "on": {"side": "right"}}]
    r = F.laminar_flow_2d({"mode": "planar", "rectangles": [[0, 0, 12, 1], [3, 1, 4, 4]]}, b, reynolds=20,
                          sections=[{"name": "down", "x": 10.0, "y": [0, 1]}])
    mu = r["reference"]["viscosity_Pa_s"]
    assert mu == pytest.approx(1 / 20)
    bd = r["boundaries"]
    assert bd["main"]["flow_out"] == pytest.approx(-1.0, rel=1e-9) and bd["branch"]["flow_out"] == pytest.approx(-0.5, rel=1e-9)
    assert bd["out"]["flow_out"] == pytest.approx(1.5, rel=1e-9) and r["mass_balance"]["imbalance_rel"] < 1e-10
    assert r["sections"]["down"]["flow_positive_axis"] == pytest.approx(1.5, rel=1e-4)
    down = next(s for s in r["straight_sections"] if s["axis"] == "x" and s["from"] > 3.9)
    assert down["developed"] and down["pressure_drop_per_length"] == pytest.approx(12 * mu * 1.5, rel=1e-4)
    assert {s["flow_direction"] for s in r["straight_sections"]} == {"+x", "-y"}             # the branch flows down
    assert set(r["pressure_drops"]) == {"main->out", "branch->out"}
    assert r["evidence"]["poiseuille_self_check"] and r["evidence"]["verified"], r["evidence"]["reasons"]


# ---------------------------------------------------------------- published values
@needs_fenicsx
def test_backward_facing_step_reattachment():
    """Armaly, Durst, Pereira & Schönung, J. Fluid Mech. 127 (1983) 473: step S = 4.9 mm, inlet height h = 5.2 mm
    (expansion ratio 1.94), Re = ū·2h/ν = 100 → reattachment x_r/S ≈ 3 (2-D computations: 2.8–3.0)."""
    h, S = 1.0, 0.9423
    r = F.laminar_flow_2d({"mode": "planar", "rectangles": [[-5, S, 0, S + h], [0, 0, 30, S + h]]},
                          [dict(IN_OUT[0], mean_velocity=1.0), IN_OUT[1]], reynolds=100, reference_length_m=2 * h,
                          mesh={"boxes": [{"box": [-0.5, 0, 6, S + h], "size": 0.06}]})
    rc = r["recirculation"]
    assert rc["exists"] and rc["zones"][0]["wall"]["coordinate"] == pytest.approx(0.0, abs=1e-9)   # on the lower wall
    assert 2.7 < rc["zones"][0]["wall"]["to"] / S < 3.1
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


@needs_fenicsx
def test_sudden_pipe_expansion_reattachment_length():
    """1:2 sudden expansion in a pipe, developed inlet. Macagno & Hung, J. Fluid Mech. 28 (1967) 43 (computation and
    experiment, Re_d ≤ 200): the reattachment length grows linearly, L_r/h ≈ 0.088–0.09 Re_d (h = step height, Re_d on the
    inlet diameter and mean velocity)."""
    Re, h = 50, 0.5
    r = F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[0, 0, 2, 0.5], [2, 0, 12, 1]]},
                          [dict(IN_OUT[0], mean_velocity=1.0), IN_OUT[1]], reynolds=Re)
    assert r["reference"]["length_m"] == pytest.approx(1.0) and r["Re"] == pytest.approx(Re)     # inlet diameter
    z = r["recirculation"]["zones"][0]
    assert z["wall"]["coordinate"] == pytest.approx(1.0) and z["wall"]["from"] == pytest.approx(2.0, abs=0.1)   # on the pipe wall, from the step
    assert 0.082 < (z["wall"]["to"] - 2.0) / h / Re < 0.095
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


@needs_fenicsx
def test_orifice_plate_in_a_tube_creeping_flow():
    """Sampson (1891): creeping flow through a circular hole of radius a in a thin wall, Δp = 3 μ Q / a³; with a plate of
    thickness t, Dagan, Weinbaum & Pfeffer, J. Fluid Mech. 115 (1982) 505: Δp = (μ Q / a³)(3 + 8 t / (π a)) within 1 %.
    Here the hole is in a tube (a/R = 0.2), which lowers the loss by ~2 %; the Poiseuille loss of the tube is removed."""
    R, a, t, Lh = 1.0, 0.2, 0.02, 6.0
    r = F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[-Lh, 0, Lh, R]],
                           "obstacles": [{"name": "plate", "type": "rectangle", "box": [-t / 2, a, t / 2, R + 0.5]}]},
                          [dict(IN_OUT[0], mean_velocity=1.0), IN_OUT[1]], reynolds=0.01, mesh={"obstacle_size": 0.01, "growth": 0.15})
    mu, Q = r["reference"]["viscosity_Pa_s"], math.pi * R ** 2
    assert r["Re"] == pytest.approx(0.01) and r["reference"]["length_m"] == pytest.approx(2 * R)   # plate is not 'immersed'
    extra = r["pressure_drops"]["in->out"] - 8 * mu * 1.0 / R ** 2 * (2 * Lh - t)
    assert extra == pytest.approx(mu * Q / a ** 3 * (3 + 8 * t / (math.pi * a)), rel=0.04)
    plate = r["obstacles"]["plate"]
    assert plate["drag_pressure_fraction"] > 0.95                                   # the plate is held by pressure
    assert plate["drag"] == pytest.approx(extra * math.pi * R ** 2, rel=0.03)       # momentum balance: force ≈ Δp_extra·A
    assert r["evidence"]["mass_balanced"] and r["evidence"]["poiseuille_self_check"]


@needs_fenicsx
def test_lid_driven_cavity_re1000_needs_continuation():
    """Lid-driven cavity, Re = 1000 (moving wall, no outlet → pressure with zero mean). u on the vertical centre-line:
    Botella & Peyret, Comput. Fluids 27 (1998) 421 (spectral): −0.38857, −0.28037, −0.10820, −0.06206, 0.18867 at
    y = 0.1719, 0.2813, 0.4531, 0.5, 0.7344; Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387 gives −0.38289 … 0.18719.
    Newton from rest fails at this Re: the answer must come from the continuation in Re."""
    ys = (0.1719, 0.2813, 0.4531, 0.5, 0.7344)
    r = F.laminar_flow_2d({"mode": "planar", "rectangles": [[0, 0, 1, 1]]},
                          [{"name": "lid", "type": "wall", "on": {"side": "top"}, "velocity": [1, 0]}], reynolds=1000,
                          points=[[0.5, y] for y in ys], mesh={"size": 0.025})
    u = [p["velocity"][0] for p in r["points"]]
    assert u == pytest.approx([-0.38857, -0.28037, -0.10820, -0.06206, 0.18867], rel=0.01)
    ev = r["evidence"]
    assert ev["newton_converged"] and ev["continuation_in_Re"] and not ev["newton_path"][0]["converged"]
    assert ev["newton_path"][-1]["Re"] == pytest.approx(1000) and ev["newton_residual_norm"] < 1e-8
    assert r["boundaries"]["lid"]["type"] == "wall" and any("media nula" in w for w in r["warnings"])


@needs_fenicsx
def test_newton_failure_is_reported_not_hidden():
    """Cavity at Re = 2·10⁵ on a very coarse mesh: no steady solution is reached; the tool says so and gives no numbers."""
    r = F.laminar_flow_2d({"mode": "planar", "rectangles": [[0, 0, 1, 1]]},
                          [{"name": "lid", "type": "wall", "on": {"side": "top"}, "velocity": [1, 0]}], reynolds=2e5,
                          points=[[0.5, 0.5]], mesh={"size": 0.1})
    assert r["converged"] is False and "points" not in r and "boundaries" not in r
    ev = r["evidence"]
    assert not ev["verified"] and not ev["newton_converged"] and "Newton no convergió" in ev["reasons"][0]
    assert ev["continuation_in_Re"] and ev["newton_path"][-1]["Re"] < 2e5


@pytest.fixture(scope="module")
def sphere():
    """Sphere of radius a = 1 on the axis of a 'wide' stream: tube of radius 50 a, 100 a up- and downstream, uniform
    stream imposed on the inlet and the side; Re = 0.01 on the diameter. One run per Péclet number."""
    def run(pe):
        return F.laminar_flow_2d(
            {"mode": "axisymmetric", "rectangles": [[-100, 0, 100, 50]],
             "obstacles": [{"name": "sphere", "type": "circle", "center": [0, 0], "radius": 1}]},
            [{"name": "in", "type": "stream", "on": {"side": "left"}, "velocity": [1, 0]},
             {"name": "side", "type": "stream", "on": {"side": "top"}, "velocity": [1, 0]},
             {"name": "out", "type": "outlet", "on": {"side": "right"}}],
            reynolds=0.01, scalar={"peclet": pe, "values": {"sphere": 1.0, "in": 0.0, "side": 0.0}})
    return {pe: run(pe) for pe in (0.01, 1.0)}


@needs_fenicsx
def test_stokes_drag_on_a_sphere(sphere):
    """Stokes: C_D·Re = 24, one third pressure and two thirds viscous. In a finite stream bounded at R = 50 a by the
    moving 'wall' the exact creeping-flow drag is 24·K with K = 1/(1 − 2.10444 λ + 2.08877 λ³ − …), λ = a/R (Haberman &
    Sayre 1958; Happel & Brenner §7-3): K = 1.0439 for λ = 0.02. So 24 is approached only as the domain grows."""
    r = sphere[0.01]
    ob = r["obstacles"]["sphere"]
    lam = 0.02
    K = 1 / (1 - 2.10444 * lam + 2.08877 * lam ** 3 - 0.94813 * lam ** 5)
    cd_re = ob["drag_coefficient"] * ob["reynolds_obstacle"]
    assert ob["reynolds_obstacle"] == pytest.approx(0.01)
    assert cd_re == pytest.approx(24 * K, rel=5e-3)                # the exact value for THIS domain
    assert cd_re == pytest.approx(24, rel=0.05)                    # Stokes, within the declared wall effect
    assert ob["drag_pressure_fraction"] == pytest.approx(1 / 3, abs=3e-3)
    assert ob["drag_viscous"] / ob["drag"] == pytest.approx(2 / 3, abs=3e-3)
    assert ob["drag_direct_vs_consistent_rel"] < 5e-3              # stress integral agrees with the reaction force
    assert not r["recirculation"]["exists"]                        # no wake in creeping flow
    assert r["domain"]["sphere"]["radius_ratio_obstacle_to_domain"] == pytest.approx(0.02)
    assert any("depende del tamaño del dominio" in w for w in r["warnings"])
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


@needs_fenicsx
def test_sphere_heat_transfer_tends_to_two(sphere):
    """Conduction limit Nu = 2 (sphere in an infinite medium). In the finite domain used (outer boundary between 50 a
    and 112 a from the centre) pure conduction gives 2/(1 − a/R_o) = 2.018–2.041. At Pe = 1 (creeping flow) Clift, Grace
    & Weber, Bubbles, Drops and Particles (1978) eq. 3-49: Sh = 1 + (1 + Pe)^(1/3) = 2.26 (±2 %)."""
    lo = sphere[0.01]["scalar"]["surfaces"]["sphere"]
    assert lo["peclet_on_characteristic_length"] == pytest.approx(0.01)
    assert 2 / (1 - 1 / math.hypot(100, 50)) < lo["nusselt_sherwood_mean"] < 2 / (1 - 1 / 50) + 0.005
    assert lo["nusselt_sherwood_mean"] == pytest.approx(2.0, rel=0.025)
    assert lo["nusselt_direct_gradient"] == pytest.approx(lo["nusselt_sherwood_mean"], rel=5e-3)
    one = sphere[1.0]["scalar"]["surfaces"]["sphere"]["nusselt_sherwood_mean"]
    assert one == pytest.approx(1 + 2 ** (1 / 3), rel=0.04) and one > lo["nusselt_sherwood_mean"]
    assert "scalar.sphere.nusselt_sherwood_mean" in sphere[1.0]["evidence"]["mesh_refinement"]
    assert sphere[1.0]["evidence"]["mesh_converged"]


CYLINDER = dict(
    geometry={"mode": "planar", "rectangles": [[-30, -30, 60, 30]],
              "obstacles": [{"name": "cyl", "type": "circle", "center": [0, 0], "radius": 0.5}]},
    boundaries=[{"name": "in", "type": "stream", "on": {"side": "left"}, "velocity": [1, 0]},
                {"name": "top", "type": "stream", "on": {"side": "top"}, "velocity": [1, 0]},
                {"name": "bot", "type": "stream", "on": {"side": "bottom"}, "velocity": [1, 0]},
                {"name": "out", "type": "outlet", "on": {"side": "right"}}],
    mesh={"boxes": [{"box": [-1.5, -1.5, 6, 1.5], "size": 0.1}]})


@needs_fenicsx
@pytest.mark.parametrize("Re, cd_range, wake_range", [(20, (1.99, 2.10), (0.87, 0.97)), (40, (1.49, 1.60), (2.1, 2.4))])
def test_cylinder_steady_wake(Re, cd_range, wake_range):
    """Steady flow past a circular cylinder. Dennis & Chang, J. Fluid Mech. 42 (1970) 471: C_D = 2.045 (Re 20), 1.522
    (Re 40), wake length L/D = 0.94, 2.35; Fornberg, J. Fluid Mech. 98 (1980) 819: 2.000, 1.498 and 0.91, 2.24; Tritton's
    experiments (1959) 2.0–2.1 and 1.5–1.6; Coutanceau & Bouard (1977) L/D = 0.93, 2.13. Attached twin vortices."""
    r = F.laminar_flow_2d(**CYLINDER, reynolds=Re)
    ob = r["obstacles"]["cyl"]
    assert cd_range[0] < ob["drag_coefficient"] < cd_range[1]
    assert abs(ob["lift_coefficient"]) < 1e-3                                       # symmetric
    assert ob["drag_coefficient_pressure"] + ob["drag_coefficient_viscous"] == pytest.approx(ob["drag_coefficient"])
    z = r["recirculation"]["zones"][0]
    assert r["recirculation"]["exists"] and z["attached_to_obstacle"] == "cyl"
    assert wake_range[0] < z["length_over_obstacle_size"] < wake_range[1]
    assert r["domain"]["cyl"]["blockage_ratio"] == pytest.approx(1 / 60)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert r["evidence"]["mesh_refinement"]["obstacle.cyl.drag"]["change_rel"] < 0.01


@needs_fenicsx
def test_unconverged_mesh_is_flagged():
    """A mesh far too coarse for the wake must not be sealed: the drag changes between the two meshes and is listed."""
    r = F.laminar_flow_2d(CYLINDER["geometry"], CYLINDER["boundaries"], reynolds=40,
                          mesh={"size": 6.0, "obstacle_size": 0.5, "growth": 0.6, "tolerance": 0.005})
    ev = r["evidence"]
    assert not ev["verified"] and not ev["mesh_converged"]
    assert "obstacle.cyl.drag" in ev["not_mesh_converged"] and "no convergidos en malla" in ev["reasons"][0]


@slow
def test_cylinder_vortex_shedding_re100():
    """Re = 100: periodic shedding. Williamson, Annu. Rev. Fluid Mech. 28 (1996): St = 0.164; Norberg (2003) St = 0.164;
    2-D computations (Henderson 1995; Park, Kwon & Choi 1998): mean C_D = 1.33–1.35, lift amplitude ≈ 0.33. A domain of
    ±15 D with slip sides (blockage 3 %) raises both slightly: accepted St 0.16–0.17, C_D 1.3–1.4."""
    r = F.laminar_flow_2d(
        {"mode": "planar", "rectangles": [[-15, -15, 30, 15]],
         "obstacles": [{"name": "cyl", "type": "circle", "center": [0, 0], "radius": 0.5}]},
        [{"name": "in", "type": "stream", "on": {"side": "left"}, "velocity": [1, 0]},
         {"name": "top", "type": "symmetry", "on": {"side": "top"}}, {"name": "bot", "type": "symmetry", "on": {"side": "bottom"}},
         {"name": "out", "type": "outlet", "on": {"side": "right"}, "normal_flow": False}],
        reynolds=100, mesh={"size": 2.0, "obstacle_size": 0.05, "growth": 0.2, "refinement_check": False,
                            "boxes": [{"box": [-1, -1.2, 10, 1.2], "size": 0.15}]},
        transient={"dt_s": 0.05, "t_end_s": 150}, timeout_s=1500)
    tr = r["transient"]
    assert tr["state"] == "periodic" and tr["signal"] == "lift_coefficient"
    assert 0.160 < tr["strouhal"] < 0.170
    assert 1.30 < tr["drag_coefficient_mean"] < 1.40
    assert 0.25 < tr["lift_coefficient_amplitude"] < 0.40
    assert r["obstacles"]["cyl"]["drag_coefficient"] < tr["drag_coefficient_mean"]   # the steady (unstable) branch has less drag
    assert not r["evidence"]["verified"] and any("transitorio sin estudio" in s for s in r["evidence"]["reasons"])
