"""laminar_flow_2d (FEniCSx Taylor-Hood Navier-Stokes on user geometry): every case against an exact or published value.

Non-Newtonian cases against the analytic developed flow of a power-law fluid; the transient against the exact
start-up of Couette flow. Vortex shedding at Re = 100 (several minutes) runs only with ALCHEMAI_SLOW_FLOW2D=1."""
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


def test_rejects_bad_rheology_transient_and_budget_requests():
    geo = {"mode": "axisymmetric", "rectangles": [[0, 0, 0.1, 0.005]]}
    pl = {"model": "power_law", "K": 0.01, "n": 0.5}
    with pytest.raises(ValueError, match="rheology.model"):
        F.laminar_flow_2d(geo, IN_OUT, density_kg_m3=1000, rheology={"model": "bingham", "tau_y": 1})
    with pytest.raises(ValueError, match="no se admite reynolds ni viscosity_Pa_s"):
        F.laminar_flow_2d(geo, IN_OUT, **WATER, rheology=pl)
    with pytest.raises(ValueError, match="no se admite reynolds"):
        F.laminar_flow_2d(geo, IN_OUT, reynolds=10, rheology=pl)
    with pytest.raises(ValueError, match="claves no reconocidas"):
        F.laminar_flow_2d(geo, IN_OUT, density_kg_m3=1000, rheology=dict(pl, eta0=1))
    with pytest.raises(ValueError, match="rheology.n"):
        F.laminar_flow_2d(geo, IN_OUT, density_kg_m3=1000, rheology=dict(pl, n=0))
    with pytest.raises(ValueError, match="gamma_reg_1_s"):
        F.laminar_flow_2d(geo, IN_OUT, density_kg_m3=1000, rheology=dict(pl, gamma_reg_1_s=0))
    with pytest.raises(ValueError, match="transient no está soportado"):
        F.laminar_flow_2d(geo, IN_OUT, density_kg_m3=1000, rheology=pl, transient={})
    with pytest.raises(ValueError, match="refinement"):
        F.laminar_flow_2d(geo, IN_OUT, **WATER, transient={"refinement": "triple"})
    with pytest.raises(ValueError, match="max_wall_s"):
        F.laminar_flow_2d(geo, IN_OUT, **WATER, max_wall_s=0)


def test_transient_defaults_choose_time_step_and_wake_mesh():
    """Request-side defaults of a transient (no FEniCSx): dt left to the solver (Courant-capped), a wake box behind the
    obstacle when the user gives no mesh, and the wall budget passed on with a kill margin above it."""
    geo = {"mode": "planar", "rectangles": [[-15, -15, 30, 15]],
           "obstacles": [{"name": "cyl", "type": "circle", "center": [0, 0], "radius": 0.5}]}
    bnd = [{"name": "in", "type": "stream", "on": {"side": "left"}, "velocity": [1, 0]},
           {"name": "out", "type": "outlet", "on": {"side": "right"}, "normal_flow": False}]
    args = dict(density_kg_m3=None, viscosity_Pa_s=None, reynolds=100, reference_length_m=None, reference_velocity_m_s=None,
                stream_direction="+x", points=None, sections=None, pressure_drop_between=None, scalar=None)
    p, ctx = F._prepare(geo, bnd, **args, transient={}, mesh=None)
    tr = p["transient"]
    assert tr["dt"] is None and tr["t_end"] == pytest.approx(150.0) and tr["refinement"] == "dt" and tr["initial"] == "steady"
    assert p["mesh"]["boxes"] == [{"box": [pytest.approx(-1.2), pytest.approx(-1.2), pytest.approx(10.0), pytest.approx(1.2)],
                                   "size": pytest.approx(1 / 7)}]
    assert any("caja de estela" in n for n in ctx["notes"])
    p, ctx = F._prepare(geo, bnd, **args, transient={"dt_s": 0.02}, mesh={"boxes": [{"box": [-1, -1, 5, 1], "size": 0.2}]})
    assert p["transient"]["dt"] == 0.02 and len(p["mesh"]["boxes"]) == 1 and not ctx["notes"]     # the user's mesh is respected
    p, _ = F._prepare(geo, bnd, **args, transient=None, mesh=None)
    assert "transient" not in p and p["mesh"]["boxes"] == []


def test_apparent_viscosity_laws():
    """The viscosity laws the solver assembles, in plain Python: limits that must hold exactly."""
    car = F._rheology({"model": "carreau", "eta0": 2.0, "eta_inf": 0.1, "lambda_s": 3.0, "n": 0.4}, 1.0)
    cy = F._rheology({"model": "carreau_yasuda", "eta0": 2.0, "eta_inf": 0.1, "lambda_s": 3.0, "n": 0.4, "a": 2.0, "gamma_reg_1_s": 1e-12}, 1.0)
    for g in (0.0, 0.01, 1.0, 50.0):
        assert F.apparent_viscosity(cy, g) == pytest.approx(F.apparent_viscosity(car, g), rel=1e-9)     # Yasuda a = 2 is Carreau
        assert 0.1 <= F.apparent_viscosity(car, g) <= 2.0
    assert F.apparent_viscosity(car, 0.0) == 2.0 and F.apparent_viscosity(car, 1e9) == pytest.approx(0.1, rel=1e-3)
    assert F.apparent_viscosity(car, 2.0) == pytest.approx(0.1 + 1.9 * 37.0 ** -0.3)
    for kw in ({"n": 1.0}, {"lambda_s": 0.0}, {"eta_inf": 2.0}):                                        # Newtonian members
        r = F._rheology({"model": "carreau", "eta0": 2.0, "eta_inf": 0.1, "lambda_s": 3.0, "n": 0.4, **kw}, 1.0)
        assert r["newtonian_equivalent_Pa_s"] == 2.0 and F.apparent_viscosity(r, 7.0) == pytest.approx(2.0)
    assert car["newtonian_equivalent_Pa_s"] is None
    pl = F._rheology({"model": "power_law", "K": 0.3, "n": 0.5}, 20.0)
    assert pl["gamma_reg_1_s"] == pytest.approx(0.02)                                                   # default 1e-3·U_ref/L_ref
    assert F.apparent_viscosity(pl, 4.0) == pytest.approx(0.3 * 4.0 ** -0.5, rel=1e-4)
    assert F.apparent_viscosity(pl, 0.0) == pytest.approx(0.3 * 0.02 ** -0.5)                           # bounded at rest


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
    assert w.to_dict()["description"].strip() == inspect.cleandoc(T.laminar_flow_2d.__doc__).strip() or \
        inspect.getdoc(w.func) == inspect.getdoc(T.laminar_flow_2d)
    for word in ("rheology", "power_law", "carreau", "gamma_reg_1_s", "max_wall_s", "provisional_not_final", "refinement"):
        assert word in T.laminar_flow_2d.__doc__


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


@needs_fenicsx
@pytest.mark.parametrize("n", [1.0, 0.6])
def test_entry_length_field_agrees_with_probes(n):
    """Creeping flow between parallel plates from a uniform inlet. Internal consistency: a probe placed at the reported
    entry_length_99 must read 99 % of the developed centre-line velocity measured in the same solution (it read
    98.9 % when the length was measured from the detection grid instead of the inlet and normalised analytically).
    Newtonian reference L_e ≈ 0.63 gaps as Re → 0: VALUE FROM MEMORY, NOT CHECKED AGAINST A SOURCE, hence only a
    wide window. No reference is asserted for the power-law case."""
    H, L, U, rho, Re = 0.01, 0.06, 0.01, 1000.0, 0.01
    kw = dict(reynolds=Re) if n == 1.0 else dict(density_kg_m3=rho, rheology={"model": "power_law", "n": n,
                                                                              "K": rho * U ** (2 - n) * H ** n / Re})
    geo = {"mode": "planar", "rectangles": [[0, 0, L, H]]}
    s = F.laminar_flow_2d(geo, UNIFORM_IN, **kw)["straight_sections"][0]
    le = s["entry_length_99"]
    assert s["developed"] and s["entry_length_reference_measured"] and s["entry_length_measured_from"] == pytest.approx(0.0, abs=1e-12)
    assert s["entry_length_98"] < le < s["entry_length_995"] and s["entry_length_99_sensitivity_rel_per_0p1pct_reference"] > 0.005
    assert s["entry_length_99_mesh_check"]["converged"] and "entry_length_99_status" not in s
    r = F.laminar_flow_2d(geo, UNIFORM_IN, **kw, points=[[le, H / 2], [L - H, H / 2]])
    u = [q["velocity"][0] for q in r["points"]]
    assert u[0] / u[1] == pytest.approx(0.99, abs=2e-4)
    if n == 1.0:
        assert 0.58 < le / H < 0.68                                               # memory value 0.63, not verified


# ---------------------------------------------------------------- published values
@needs_fenicsx
def test_backward_facing_step_reattachment():
    """Armaly, Durst, Pereira & Schönung, J. Fluid Mech. 127 (1983) 473: step S = 4.9 mm, inlet height h = 5.2 mm
    (expansion ratio 1.94), Re = ū·2h/ν = 100 → reattachment x_r/S ≈ 3 (2-D computations: 2.8–3.0).
    NOT CONFIRMED AGAINST THE SOURCE: the Re = 100 value and the 2.8–3.0 interval were not re-checked in the paper."""
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
    inlet diameter and mean velocity).
    NOT CONFIRMED AGAINST THE SOURCE: the slope 0.088–0.09 was not re-checked in the paper."""
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
    y = 0.1719, 0.2813, 0.4531, 0.5, 0.7344 — this table is NOT CONFIRMED AGAINST THE SOURCE (not re-checked in the
    paper). Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387 gives −0.38289 … 0.18719 at the same points (Re = 1000
    column, confirmed); the asserted values are within 1.5 % of Ghia's at the two points quoted here.
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


# ---------------------------------------------------------------- non-Newtonian (generalised Newtonian) fluids
UNIFORM_IN = [dict(IN_OUT[0], profile="uniform"), IN_OUT[1]]


@needs_fenicsx
@pytest.mark.parametrize("mode", ["axisymmetric", "planar"])
@pytest.mark.parametrize("n", [0.5, 1.5])
def test_power_law_developed_flow_is_analytic(mode, n):
    """Developed flow of a power-law fluid τ = K γ̇ⁿ from a UNIFORM inlet (so the profile is computed, not imposed).
    Exact (Bird, Armstrong & Hassager, Dynamics of Polymeric Liquids vol. 1; derivable by integrating τ_rz = −(dp/dz) r/2):
    tube  u_max/ū = (3n+1)/(n+1),  −dp/dz = (2K/R)·[(3n+1)/n · ū/R]ⁿ
    plates (gap H = 2h)  u_max/ū = (2n+1)/(n+1),  −dp/dx = (K/h)·[(2n+1)/n · ū/h]ⁿ."""
    K, U, L, rho = 0.01, 0.01, 0.1, 1000.0
    if mode == "axisymmetric":
        R = 0.005
        geo, centre = {"mode": mode, "rectangles": [[0, 0, L, R]]}, [0.09, 0.0]
        ratio, grad = (3 * n + 1) / (n + 1), 2 * K / R * ((3 * n + 1) / n * U / R) ** n
    else:
        H = 0.01
        geo, centre = {"mode": mode, "rectangles": [[0, 0, L, H]]}, [0.09, H / 2]
        ratio, grad = (2 * n + 1) / (n + 1), K / (H / 2) * ((2 * n + 1) / n * U / (H / 2)) ** n
    r = F.laminar_flow_2d(geo, UNIFORM_IN, density_kg_m3=rho, rheology={"model": "power_law", "K": K, "n": n}, points=[centre])
    s = r["straight_sections"][0]
    assert r["converged"] and s["developed"]
    assert s["u_max_over_mean_at_end"] == pytest.approx(ratio, rel=2e-3)
    assert r["points"][0]["velocity"][0] == pytest.approx(ratio * U, rel=2e-3)
    assert s["pressure_drop_per_length"] == pytest.approx(grad, rel=2e-3)
    assert s["poiseuille_pressure_drop_per_length"] == pytest.approx(grad, rel=1e-4)       # built-in reference: same formula, measured flow
    assert "power-law analytic" in s["developed_reference"] and r["evidence"]["poiseuille_self_check"]
    # entry length is reported (99 % of the developed centre-line velocity); no reference value is asserted: the
    # correlations for power-law entry lengths I could quote are from memory and NOT verified
    # normalised with the developed centre-line velocity MEASURED at the section end (the analytic one belongs to the
    # unregularised model and the crossing is flat: 0.1 % in the reference moves the length by several %)
    assert 0 < s["entry_length_99"] < 0.5 * L and s["entry_length_reference_measured"]
    assert s["entry_length_98"] < s["entry_length_99"] < s["entry_length_995"]
    assert s["developed_centreline_measured_over_analytic"] == pytest.approx(1.0, abs=2e-3)
    assert s["entry_length_99_mesh_check"]["converged"] == (s.get("entry_length_99_status") is None)
    assert s["entry_length_99_over_width"] == pytest.approx(s["entry_length_99"] / 0.01)
    assert s["centreline_velocity_start_over_developed"] < 0.8
    rh = r["rheology"]
    assert rh["model"] == "power_law" and rh["parameters"] == {"n": n, "K": K}
    assert rh["regularisation_shear_rate_1_s"] == pytest.approx(1e-3 * U / 0.01)           # declared: 1e-3·U_ref/L_ref
    assert rh["reference_viscosity_Pa_s"] == pytest.approx(K * (U / 0.01) ** (n - 1), rel=1e-5) and r["Re"] == pytest.approx(10, rel=1e-5)
    assert rh["field"]["cells_below_regularisation_rate_fraction"] == 0.0
    assert rh["field"]["apparent_viscosity_max_Pa_s"] > rh["field"]["apparent_viscosity_min_Pa_s"] > 0
    if mode == "axisymmetric":
        assert rh["Re_metzner_reed"] == pytest.approx(rho * U ** (2 - n) * 0.01 ** n / (K * 8 ** (n - 1) * ((3 * n + 1) / (4 * n)) ** n))
    ev = r["evidence"]
    assert ev["newton_converged"] and ev["newton_residual_norm"] < 1e-8 and ev["newton_tolerances"]["snes_rtol"] == 1e-10
    assert ev["mass_balanced"] and "newtoniano generalizado" in r["assumptions"][0]


@needs_fenicsx
def test_power_law_developed_inlet_profile_is_a_fixed_point():
    """With the developed power-law profile at the inlet (profile "parabolic" = developed for the model) nothing changes
    along the tube: entry length 0, exact gradient from the first diameter, and the whole result is mesh-verified."""
    K, n, U, R, L = 0.01, 0.5, 0.01, 0.005, 0.1
    r = F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[0, 0, L, R]]}, IN_OUT, density_kg_m3=1000.0,
                          rheology={"model": "power_law", "K": K, "n": n})
    s = r["straight_sections"][0]
    grad = 2 * K / R * ((3 * n + 1) / n * U / R) ** n
    assert s["pressure_drop_per_length"] == pytest.approx(grad, rel=1e-3)
    assert r["pressure_drops"]["in->out"] == pytest.approx(grad * L, rel=2e-3)
    assert s["entry_length_99"] == 0.0 and s["u_max_over_mean_at_end"] == pytest.approx(5 / 3, rel=1e-3)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


@needs_fenicsx
@pytest.mark.parametrize("newtonian", [{"n": 1.0, "lambda_s": 3.0}, {"n": 0.4, "lambda_s": 0.0}])
def test_carreau_reduces_to_newtonian(newtonian):
    """Carreau with n = 1 or λ = 0 is Newtonian with η0: Hagen-Poiseuille must come out of the generalised-Newtonian
    code path (stress-divergence form, viscosity evaluated from γ̇), −dp/dz = 8 η0 ū / R², u_axis = 2 ū."""
    R, L, U = 0.005, 0.1, 0.01
    r = F.laminar_flow_2d({"mode": "axisymmetric", "rectangles": [[0, 0, L, R]]}, IN_OUT, density_kg_m3=1000.0,
                          rheology={"model": "carreau", "eta0": 1e-3, "eta_inf": 0.0, **newtonian}, points=[[0.08, 0.0]])
    g = 8 * 1e-3 * U / R ** 2
    assert r["Re"] == pytest.approx(100) and r["rheology"]["newtonian_equivalent_Pa_s"] == 1e-3
    assert r["pressure_drops"]["in->out"] == pytest.approx(g * L, rel=1e-5)
    assert r["straight_sections"][0]["pressure_drop_per_length"] == pytest.approx(g, rel=1e-5)
    assert r["points"][0]["velocity"][0] == pytest.approx(2 * U, rel=1e-5)
    f = r["rheology"]["field"]
    assert f["apparent_viscosity_min_Pa_s"] == pytest.approx(1e-3) and f["apparent_viscosity_max_Pa_s"] == pytest.approx(1e-3)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


@needs_fenicsx
def test_carreau_power_law_limit_and_bounds():
    """Two analytic anchors for Carreau with n ≠ 1. (a) For λγ̇ ≫ 1 and η∞ = 0 it IS the power law K = η0 λ^(n−1): with
    λ = 1000 s (λγ̇_wall = 10⁴) the tube gradient must equal the power-law one. (b) A fluid with η∞ ≤ η ≤ η0 has a
    pressure gradient between the two Hagen-Poiseuille values and a profile between plug and parabola; no closed form,
    so the entry length is referred to the measured developed velocity."""
    R, L, U, n, K, lam = 0.005, 0.1, 0.01, 0.5, 0.01, 1000.0
    geo = {"mode": "axisymmetric", "rectangles": [[0, 0, L, R]]}
    r = F.laminar_flow_2d(geo, UNIFORM_IN, density_kg_m3=1000.0,
                          rheology={"model": "carreau", "eta0": K * lam ** (1 - n), "lambda_s": lam, "n": n})
    s = r["straight_sections"][0]
    assert s["pressure_drop_per_length"] == pytest.approx(2 * K / R * ((3 * n + 1) / n * U / R) ** n, rel=2e-3)
    assert s["u_max_over_mean_at_end"] == pytest.approx((3 * n + 1) / (n + 1), rel=2e-3)
    r = F.laminar_flow_2d(geo, UNIFORM_IN, density_kg_m3=1000.0,
                          rheology={"model": "carreau", "eta0": 5e-3, "eta_inf": 1e-3, "lambda_s": 1.0, "n": 0.4})
    s = r["straight_sections"][0]
    assert 8 * 1e-3 * U / R ** 2 < s["pressure_drop_per_length"] < 8 * 5e-3 * U / R ** 2
    assert 1.0 < s["u_centre_over_mean_developed_measured"] < 2.0 and "poiseuille_error_rel" not in s
    assert s["entry_length_reference_measured"] and 0 < s["entry_length_99"] < 0.5 * L
    f = r["rheology"]["field"]
    assert 1e-3 <= f["apparent_viscosity_min_Pa_s"] < f["apparent_viscosity_max_Pa_s"] <= 5e-3
    assert r["rheology"]["regularisation_shear_rate_1_s"] is None and r["evidence"]["poiseuille_self_check"] is None


# ---------------------------------------------------------------- transient
COUETTE = dict(
    geometry={"mode": "planar", "rectangles": [[0, 0, 0.02, 0.01]]},
    boundaries=[{"name": "plate", "type": "wall", "on": {"side": "bottom"}, "velocity": [0.01, 0]},
                {"name": "l", "type": "outlet", "on": {"side": "left"}}, {"name": "r", "type": "outlet", "on": {"side": "right"}}],
    **WATER)


@needs_fenicsx
def test_transient_couette_startup_is_analytic():
    """Start-up of Couette flow (fluid at rest between plates a distance H apart, the lower one set in motion at t = 0):
    u/U = (1 − y/H) − (2/π) Σ_k (1/k) exp(−k²π² ν t / H²) sin(kπ y / H)  (separation of variables; e.g. Bird, Stewart &
    Lightfoot, Transport Phenomena, unsteady flow between parallel plates). Compared at ν t / H² = 0.05, far from steady."""
    H, U, nu = 0.01, 0.01, 1e-6
    t_end = 0.05 * H ** 2 / nu
    ys = [0.1 * H, 0.25 * H, 0.5 * H]
    r = F.laminar_flow_2d(**COUETTE, points=[[0.01, y] for y in ys],
                          transient={"initial": "rest", "dt_s": t_end / 100, "t_end_s": t_end, "stop_when_settled": False,
                                     "refinement": "both"})
    tr = r["transient"]

    def exact(y, t):
        return U * ((1 - y / H) - 2 / math.pi * sum(math.exp(-k * k * math.pi ** 2 * nu * t / H ** 2) * math.sin(k * math.pi * y / H) / k
                                                    for k in range(1, 400)))
    assert tr["complete"] and r["partial"] is False and tr["stopped_by"] == "t_end" and tr["steps"] == 100
    assert tr["time_simulated"] == pytest.approx(t_end) and tr["initial_condition"] == "rest"
    for y, q in zip(ys, tr["points_at_end"]):
        assert q["velocity"][0] == pytest.approx(exact(y, t_end), rel=2e-3)
        assert abs(q["velocity"][1]) < 1e-6 * U
        assert abs(q["velocity"][0] - U * (1 - y / H)) > 0.05 * U                 # really unsteady: far from the steady profile
    assert tr["state"] == "not_settled" and tr["field_rate_of_change_final"] > 0   # the field is still evolving at t_end
    ref = tr["refinement"]
    assert ref["time_step"]["done"] and ref["time_step"]["converged"] and ref["time_step"]["dt"] == pytest.approx(2 * t_end / 100)
    assert ref["mesh"]["done"] and ref["mesh"]["converged"] and ref["mesh"]["mesh"]["cells"] < r["transient"]["mesh"]["cells"]
    assert ref["time_step"]["table"]["point1.velocity"]["change_rel"] < 1e-3
    assert tr["time_step"]["chosen_by"].startswith("user") and tr["performance"]["lu_factorisations"] < 10
    ev = r["evidence"]
    assert ev["transient_time_step_converged"] and ev["transient_mesh_converged"] and ev["verified"], ev["reasons"]
    json.dumps(r, default=float)


@needs_fenicsx
def test_transient_out_of_wall_budget_returns_a_flagged_partial_result():
    """200 000 steps cannot fit in 5 s: the run must come back PARTIAL, with nothing presented as a final number."""
    import time
    t0 = time.perf_counter()
    r = F.laminar_flow_2d(**COUETTE, points=[[0.01, 0.005]], max_wall_s=5,
                          transient={"initial": "rest", "dt_s": 0.002, "t_end_s": 400.0, "stop_when_settled": False})
    assert time.perf_counter() - t0 < 60
    tr = r["transient"]
    assert r["partial"] is True and "PARCIAL" in r["status"] and "presupuesto" in r["status"]
    assert tr["complete"] is False and tr["stopped_by"] == "time_budget" and 0 < tr["steps"] < 200000
    assert tr["time_simulated"] < 400.0 and "provisional_not_final" in tr and "points_at_interruption" in tr
    for key in ("strouhal", "drag_coefficient_mean", "points_at_end", "signal_final"):
        assert key not in tr
    assert tr["refinement"]["time_step"]["done"] is False
    ev = r["evidence"]
    assert not ev["verified"] and ev["transient_complete"] is False and any("PARCIAL" in s for s in ev["reasons"])


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
    """Steady flow past a circular cylinder, attached twin vortices. References and their status:
      * Fornberg, J. Fluid Mech. 98 (1980) 819: C_D = 2.000 (Re 20), 1.498 (Re 40); wake length L/D = 0.91, 2.24
        (confirmed as quoted in arXiv:2205.15886, Table 1).
      * Dennis & Chang, J. Fluid Mech. 42 (1970) 471, Re 40: C_D = 1.522, L/D = 2.345 (confirmed as quoted in
        arXiv:1310.6641, Table 1). Re 20: 2.045 and 0.94 — NOT CONFIRMED AGAINST THE SOURCE.
      * Coutanceau & Bouard (1977), experiment: L/D = 2.13 at Re 40 (confirmed); 0.93 at Re 20 — NOT CONFIRMED.
      * Tritton (1959), experiment: NOT used as support. The earlier attribution "2.0–2.1 and 1.5–1.6" is not
        confirmed (arXiv:1310.6641 attributes C_D = 1.48 at Re 40 to Tritton).
    The windows rest on the confirmed values only: Re 20 C_D (1.99, 2.10) contains Fornberg 2.000 and L/D (0.87, 0.97)
    contains Fornberg 0.91; Re 40 C_D (1.49, 1.60) contains Fornberg 1.498 and Dennis & Chang 1.522, and L/D (2.1, 2.4)
    contains 2.13, 2.24 and 2.345. The finite domain (blockage 1/60, slip sides) raises C_D slightly."""
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
    """Re = 100: periodic shedding, with the time-step and mesh checks as convergence evidence, inside the wall budget.

    References and their status:
      * St = 0.164 (Williamson, Annu. Rev. Fluid Mech. 28, 1996) and mean C_D = 1.35 (Henderson 1995): confirmed as
        quoted in arXiv:1609.04364, Table 3.
      * mean C_D = 1.33 (Park, Kwon & Choi 1998): confirmed as quoted in arXiv:2205.15886.
      * lift amplitude: other 2-D computations give 0.32–0.34 (arXiv:1803.09380, Table 1). The value 0.33 formerly
        attributed here to Park et al. could NOT be attributed to that paper.
    The domain is ±15 d with slip sides (blockage 3.3 %, inlet at 14.5 d): at that level C_D and St depend on the
    domain (measured with this solver: ±20 d → ±40 d lowered C_D 1.3 % and St 0.9 %), so the windows are those
    references widened upwards, not a claim of agreement to the last digit; the result must carry the domain warning.
    The assertions that do not depend on references are the refinement ones. Time step: St and mean C_D change < 1 %
    on doubling dt; the lift amplitude changes 2.3 % (a real discretisation error: sampling of the peak is 0.01 % and
    saturation 0.03 %), so it is sealed only through the three-level Richardson rule (observed order ≈ 2, error
    estimate of the dt value < 2 %), never by loosening the tolerance."""
    r = F.laminar_flow_2d(
        {"mode": "planar", "rectangles": [[-15, -15, 30, 15]],
         "obstacles": [{"name": "cyl", "type": "circle", "center": [0, 0], "radius": 0.5}]},
        [{"name": "in", "type": "stream", "on": {"side": "left"}, "velocity": [1, 0]},
         {"name": "top", "type": "symmetry", "on": {"side": "top"}}, {"name": "bot", "type": "symmetry", "on": {"side": "bottom"}},
         {"name": "out", "type": "outlet", "on": {"side": "right"}, "normal_flow": False}],
        reynolds=100, mesh={"size": 2.0, "obstacle_size": 0.05, "growth": 0.2,
                            "boxes": [{"box": [-1, -1.2, 10, 1.2], "size": 0.15}]},
        transient={"refinement": "both"}, max_wall_s=900)
    tr = r["transient"]
    assert tr["state"] == "periodic" and tr["signal"] == "lift_coefficient" and tr["complete"] and r["partial"] is False
    assert r["timing_s"]["total"] < 600                                             # the 10-minute target, checks included
    assert tr["time_step"]["chosen_by"].startswith("accuracy") and tr["dt"] == pytest.approx(0.05)
    assert 0.160 < tr["strouhal"] < 0.172                                           # 0.164 (Williamson) + blockage 3.3 %
    assert 1.32 < tr["drag_coefficient_mean"] < 1.40                                # 1.33 (Park), 1.35 (Henderson) + blockage
    assert 0.31 < tr["lift_coefficient_amplitude"] < 0.35                           # 0.32–0.34 (arXiv:1803.09380 Table 1)
    # the peak is interpolated; the largest sample is lower by less than (π/N)²/2, and the amplitude has stopped growing
    assert 0 <= tr["lift_coefficient_amplitude"] - tr["lift_coefficient_amplitude_sampled"] < 1e-3 * tr["lift_coefficient_amplitude"]
    assert abs(tr["lift_amplitude_growth_per_cycle"]) < 1e-3
    assert r["obstacles"]["cyl"]["drag_coefficient"] < tr["drag_coefficient_mean"]   # the steady (unstable) branch has less drag
    ref = tr["refinement"]
    for key in ("time_step", "mesh"):
        assert ref[key]["done"], ref[key]
        assert ref[key]["table"]["strouhal"]["change_rel"] < 0.01
        assert ref[key]["table"]["drag_coefficient_mean"]["change_rel"] < 0.01
        assert ref[key]["table"]["lift_coefficient_amplitude"]["change_rel"] < 0.05
    assert ref["mesh"]["mesh"]["cells"] < tr["mesh"]["cells"] and ref["time_step"]["dt"] == pytest.approx(0.1)
    assert ref["mesh"]["converged"] and ref["mesh"]["table"]["lift_coefficient_amplitude"]["change_rel"] < 0.02
    # time step: three levels, observed order of BDF2; rows above the tolerance are sealed by Richardson only
    ts, rich = ref["time_step"]["table"], ref["time_step"]["richardson"]
    assert rich["done"] and rich["dt_levels"] == pytest.approx([0.05, 0.1, 0.2])
    for key in ("strouhal", "drag_coefficient_mean", "lift_coefficient_amplitude"):
        assert 1.5 <= ts[key]["observed_order"] <= 2.5 and ts[key]["error_estimate_rel"] < 0.02
    assert ts["strouhal"]["converged_by"].startswith("change") and ts["drag_coefficient_mean"]["converged_by"].startswith("change")
    amp = ts["lift_coefficient_amplitude"]
    if amp["change_rel"] >= 0.02:
        assert amp["converged_by"].startswith("Richardson") and amp["extrapolated_dt_to_zero"] < amp["reported"]
        assert "lift_coefficient_amplitude" in r["evidence"]["transient_time_step_sealed_by_richardson"]
        assert any("Richardson" in w for w in r["warnings"])
    ev = r["evidence"]
    assert ev["transient_complete"] and ev["transient_time_step_converged"] and ev["transient_mesh_converged"]
    assert ev["verified"], ev["reasons"]
    # verified covers mesh and time step, not the domain: the blockage warning must be there
    assert r["domain"]["cyl"]["blockage_ratio"] == pytest.approx(1 / 30) and "domain_dependence" in r["domain"]["cyl"]
    assert any("DEPENDEN DEL DOMINIO" in w for w in r["warnings"])
