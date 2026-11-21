"""solve_pde_2d against independent exact results (stated in each test) and published heat-transfer values.

References used as independent data:
- Separable solutions of Laplace's equation (rectangle, polar sector): elementary.
- First-order reaction in a finite cylinder: eigenfunction (double) series and the equivalent single series in modified
  Bessel functions, both evaluated here; limits 2 I1(phi)/(phi I0(phi)) (long cylinder), tanh(phi)/phi (slab) and
  3 (phi coth(phi) - 1)/phi^2 (sphere).
- Graetz problem (parabolic profile, constant wall temperature, no axial conduction): eigenvalues and coefficients as
  tabulated in Kays & Crawford / Shah & London: theta_m = 8 sum G_n/lambda_n^2 exp(-lambda_n^2 x+), x+ = (x/R)/(Re Pr),
  lambda_n^2 = 7.3136, 44.61, 113.9, 215.2; G_n = 0.7488, 0.544, 0.463, 0.414; Nu_inf = lambda_0^2 / 2 = 3.657.
- Sphere in creeping flow (Nu and Pe based on the diameter): Acrivos & Taylor, Phys. Fluids 5 (1962) 387, low-Pe
  expansion Nu = 2 + Pe/2 + Pe^2 ln(Pe)/4 + 0.03404 Pe^2 + Pe^3 ln(Pe)/16; Acrivos & Goddard, J. Fluid Mech. 23 (1965)
  273, high-Pe Nu = 0.991 Pe^(1/3) + 0.922; Clift, Grace & Weber, "Bubbles, Drops and Particles" (1978), ch. 3,
  correlation of the numerical solutions for all Pe: Nu = 1 + (1 + Pe)^(1/3) (a few per cent).
  The formulas were written from memory of those sources (not re-read for this test); the three agree with the solver
  to 0.2-1.5 % in their own ranges, which is the check made here.
"""
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq
from scipy.special import ive, jn_zeros

from packages.agent import wrappers_pde2d
from packages.dwsim_client import tools_pde2d
from packages.transport.pde2d import solve_pde_2d

PI = math.pi
SYM = {"type": "symmetry"}


def one(v=1):
    return {"type": "dirichlet", "value": v}


def Q(res, label):
    return res["evidence"]["grid_refinement"]["quantities"][label]


def check_evidence(res, linear=True):
    ev = res["evidence"]
    assert res["converged"] and all(ev["nonlinear"]["converged_by_grid"])
    assert max(ev["nonlinear"]["residual_norm_by_grid"]) < 1e-7
    for v in res["variables"]:
        assert ev["global_balance"][v]["imbalance_relative"] < 1e-8
    assert len(ev["grid_refinement"]["grids"]) == 3
    assert ev["nonlinear"]["linear_problem"] is linear


# ---------------------------------------------------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------------------------------------------------
def test_laplace_rectangle_dirichlet():
    """u = sin(pi x) sinh(pi y)/sinh(pi): u = 0 on three sides, sin(pi x) on y = 1."""
    res = solve_pde_2d("cartesian", {"x": [0, 1], "y": [0, 1]}, {"u": {"D": 1}},
                       {"x_min": one(0), "x_max": one(0), "y_min": one(0), "y_max": one("sin(pi*x)")},
                       outputs={"points": [[0.5, 0.5], [0.25, 0.75]], "derivative_points": [[0.5, 0.5], [0.5, 1.0]],
                                "side_fluxes": [{"name": "top", "side": "y_max"}]}, grid={"n": [32, 32]})
    check_evidence(res)
    assert res["grid_converged"] and res["evidence"]["convection"]["scheme"] == "central"
    ex = lambda x, y: math.sin(PI * x) * math.sinh(PI * y) / math.sinh(PI)
    assert res["points"][0]["u"]["value"] == pytest.approx(ex(0.5, 0.5), abs=2e-5)
    assert res["points"][1]["u"]["value"] == pytest.approx(ex(0.25, 0.75), abs=2e-4)
    d = res["derivative_points"]
    assert d[0]["du_dy"]["value"] == pytest.approx(PI * math.cosh(PI / 2) / math.sinh(PI), rel=1e-3)
    assert abs(d[0]["du_dx"]["value"]) < 1e-10
    assert d[1]["du_dy"]["value"] == pytest.approx(PI / math.tanh(PI), rel=3e-3)
    # mean outward normal derivative on y = 1: 2 coth(pi); the flux leaves through the other sides -> balance
    top = res["side_fluxes"]["top"]
    assert top["mean_normal_derivative"]["value"] == pytest.approx(2 / math.tanh(PI), rel=1e-3)
    assert top["mean_flux_out"]["value"] == pytest.approx(-2 / math.tanh(PI), rel=1e-3)
    assert Q(res, "u(0.25,0.75)")["observed_order"] == pytest.approx(2.0, abs=0.15)
    assert Q(res, "u(0.25,0.75)")["richardson_extrapolated"] == pytest.approx(ex(0.25, 0.75), abs=2e-6)
    assert res["extrema"]["u"]["max"]["value"] == pytest.approx(1.0, abs=2e-3)
    assert res["extrema"]["u"]["max"]["x"] == pytest.approx(0.5, abs=0.02)


def test_laplace_rectangle_mixed_and_poisson():
    """u = cos(pi x) cosh(pi y)/cosh(pi) (insulated x sides and y = 0); Poisson -lap(u) = 2 pi^2 sin(pi x) sin(pi y)."""
    res = solve_pde_2d("cartesian", {"x": [0, 1], "y": [0, 1]}, {"u": {"D": 1}},
                       {"x_min": SYM, "x_max": {"type": "neumann", "value": 0}, "y_min": SYM,
                        "y_max": one("cos(pi*x)")}, outputs={"points": [[0.2, 0.3], [0.0, 0.0]]}, grid={"n": [32, 32]})
    check_evidence(res)
    ex = lambda x, y: math.cos(PI * x) * math.cosh(PI * y) / math.cosh(PI)
    assert res["points"][0]["u"]["value"] == pytest.approx(ex(0.2, 0.3), abs=1e-4)
    assert res["points"][1]["u"]["value"] == pytest.approx(ex(0, 0), abs=3e-4)
    bcs = {s: one(0) for s in ("x_min", "x_max", "y_min", "y_max")}
    res = solve_pde_2d("cartesian", {"x": [0, 1], "y": [0, 1]}, {"u": {"D": "k", "source": "2*k*pi^2*sin(pi*x)*sin(pi*y)"}},
                       bcs, parameters={"k": 3.0}, grid={"n": 32},
                       outputs={"points": [[0.5, 0.5]], "averages": [{"name": "m", "expression": "u"}]})
    check_evidence(res)
    assert res["points"][0]["u"]["value"] == pytest.approx(1.0, abs=5e-4)
    assert res["averages"]["m"]["average"]["value"] == pytest.approx(4 / PI ** 2, rel=1e-3)
    bal = res["evidence"]["global_balance"]["u"]
    assert bal["source_integral"] == pytest.approx(3.0 * 8.0, rel=1e-3)          # 2 k pi^2 (2/pi)^2
    assert sum(bal["diffusive_flux_out_by_side"].values()) == pytest.approx(bal["source_integral"], rel=1e-9)


def test_polar_sector_singular_corner():
    """u = r^lam sin(lam theta), lam = pi/(2 alpha): u = 0 at theta = 0, du/dtheta = 0 at theta = alpha, u = sin(lam
    theta) at r = 1. With alpha = 3 pi/2 (re-entrant corner) lam = 1/3 < 1 and du/dr ~ r^(lam-1) is unbounded at r = 0."""
    al = 1.5 * PI
    lam = PI / (2 * al)
    res = solve_pde_2d("polar", {"r": [0, 1], "theta": [0, al]}, {"u": {"D": 1}},
                       {"theta_min": one(0), "theta_max": SYM, "r_max": one("sin(lam*theta)")},
                       parameters={"lam": lam}, grid={"n": [48, 48], "grading": {"r": {"cluster": "min", "strength": 3}}},
                       outputs={"points": [[0.5, al / 2], [0.1, al], [0.01, al]],
                                "derivative_points": [[0.5, al / 2], [0.0, al / 2], [0.05, al]]})
    check_evidence(res)
    ex = lambda r, t: r ** lam * math.sin(lam * t)
    for p, (r, t) in zip(res["points"], [(0.5, al / 2), (0.1, al), (0.01, al)]):
        assert p["u"]["grid_converged"]
        assert p["u"]["value"] == pytest.approx(ex(r, t), abs=1.5e-3)
    d = res["derivative_points"]
    assert d[0]["du_dr"]["value"] == pytest.approx(lam * 0.5 ** (lam - 1) * math.sin(lam * al / 2), rel=5e-3)
    assert d[0]["du_dtheta"]["value"] == pytest.approx(lam * 0.5 ** lam * math.cos(lam * al / 2), rel=5e-3)
    assert d[2]["du_dr"]["grid_converged"]
    assert d[2]["du_dr"]["value"] == pytest.approx(lam * 0.05 ** (lam - 1), rel=1e-2)
    # the corner gradient: grows on every refinement and must be flagged, not presented
    c = Q(res, f"du_dr(0,{al / 2:g})")
    assert not d[1]["du_dr"]["grid_converged"] and "no converge" in d[1]["du_dr"]["note"]
    assert abs(c["value"]) > 1.5 * abs(c["base_grid"]) > 2.0 * abs(c["coarse_grid"])
    assert not res["grid_converged"] and c is not None and f"du_dr(0,{al / 2:g})" in res["not_grid_converged"]
    assert any("NO han convergido" in w for w in res["warnings"])


def test_polar_sector_regular_corner_and_default_grading():
    """alpha = pi/4 -> lam = 2: smooth solution u = r^2 sin(2 theta); du/dr -> 0 at the corner and converges."""
    al = PI / 4
    res = solve_pde_2d("polar", {"r": [0, 1], "theta": [0, al]}, {"u": {"D": 1}},
                       {"theta_min": one(0), "theta_max": SYM, "r_max": one("sin(2*theta)")}, grid={"n": [64, 32]},
                       outputs={"points": [[0.5, al / 2]], "derivative_points": [[0.5, al / 2], [0.0, al / 2]]})
    check_evidence(res)
    assert res["grid_converged"]
    assert res["evidence"]["grid_refinement"]["grading"]["r"] == {"type": "power", "cluster": "min", "strength": 2.0}
    assert res["points"][0]["u"]["value"] == pytest.approx(0.25 * math.sin(al), abs=3e-4)
    assert res["derivative_points"][0]["du_dr"]["value"] == pytest.approx(math.sin(al), rel=3e-3)
    assert abs(res["derivative_points"][1]["du_dr"]["value"]) < 5e-3


# ---------------------------------------------------------------------------------------------------------------------
# diffusion with reaction: effectiveness factors
# ---------------------------------------------------------------------------------------------------------------------
def eta_cylinder_single(phi, H):
    """Finite cylinder radius 1, half-length H, c = 1 on the whole surface: slab solution + Bessel correction."""
    mu = (np.arange(20000) + 0.5) * PI / H
    k = np.sqrt(phi ** 2 + mu ** 2)
    return math.tanh(phi * H) / (phi * H) + float(np.sum(4 * phi ** 2 / (H ** 2 * mu ** 2 * k ** 2) * ive(1, k) / (k * ive(0, k))))


def eta_cylinder_double(phi, H, n=1500):
    """Same quantity from the eigenfunction double series: 1 - sum_mn phi^2/(a_m^2+mu_n^2+phi^2) (4/a_m^2)(2/(mu_n H)^2)."""
    a = jn_zeros(0, n)[:, None]
    mu = ((np.arange(n) + 0.5) * PI / H)[None, :]
    return 1.0 - float(np.sum(phi ** 2 / (a ** 2 + mu ** 2 + phi ** 2) * 4 / a ** 2 * 2 / (mu * H) ** 2))


def pellet(phi, H, bcs, n=(32, 32)):
    return solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, H]}, {"c": {"D": 1, "source": "-phi^2*c"}}, bcs,
                        parameters={"phi": phi, "H": H}, grid={"n": list(n)},
                        outputs={"averages": [{"name": "eta", "expression": "phi^2*c/phi^2"}],
                                 "side_fluxes": [{"name": "side", "side": "r_max"}, {"name": "end", "side": "z_max"}],
                                 "derived": {"eta_flux": "-(side_integral + end_integral)/(phi^2*pi*H)"}})


def test_finite_cylinder_effectiveness_factor():
    for phi, H in ((2.0, 1.0), (5.0, 0.5)):
        exact = eta_cylinder_single(phi, H)
        assert eta_cylinder_double(phi, H) == pytest.approx(exact, abs=2e-3)      # the two series agree
        res = pellet(phi, H, {"r_max": one(), "z_max": one(), "z_min": SYM})
        check_evidence(res)
        eta = res["averages"]["eta"]["average"]
        assert eta["grid_converged"] and eta["value"] == pytest.approx(exact, rel=5e-4)
        assert Q(res, "average:eta")["richardson_extrapolated"] == pytest.approx(exact, rel=2e-5)
        # the same number from the flux through the surface (volume of the half pellet = pi H)
        assert res["derived"]["eta_flux"]["value"] == pytest.approx(eta["value"], rel=1e-9)
        assert res["domain_measure"] == pytest.approx(PI * H, rel=1e-12)


def test_cylinder_limits_infinite_cylinder_and_slab():
    phi = 3.0
    res = pellet(phi, 1.0, {"r_max": one(), "z_max": SYM, "z_min": SYM}, n=(48, 8))        # sealed ends
    assert res["averages"]["eta"]["average"]["value"] == pytest.approx(2 * ive(1, phi) / (phi * ive(0, phi)), rel=5e-4)
    res = pellet(phi, 1.0, {"r_max": SYM, "z_max": one(), "z_min": SYM}, n=(8, 48))        # sealed side: slab
    assert res["averages"]["eta"]["average"]["value"] == pytest.approx(math.tanh(phi) / phi, rel=5e-4)
    # long open cylinder tends to the infinite one, short one to the slab
    assert pellet(phi, 40.0, {"r_max": one(), "z_max": one(), "z_min": SYM}, n=(32, 64))["averages"]["eta"]["average"][
        "value"] == pytest.approx(eta_cylinder_single(phi, 40.0), rel=2e-3)


def test_sphere_effectiveness_factor_spherical_system():
    phi = 3.0
    res = solve_pde_2d("spherical_axisymmetric", {"r": [0, 1]}, {"c": {"D": 1, "source": "-phi^2*c"}},
                       {"r_max": one()}, parameters={"phi": phi}, grid={"n": [48, 8]},
                       outputs={"averages": [{"name": "eta", "expression": "c"}], "points": [[0.0, 1.0], [0.5, 2.0]]})
    check_evidence(res)
    assert res["averages"]["eta"]["average"]["value"] == pytest.approx(3 / phi ** 2 * (phi / math.tanh(phi) - 1), rel=5e-4)
    assert res["points"][0]["c"]["value"] == pytest.approx(phi / math.sinh(phi), rel=3e-3)
    assert res["points"][1]["c"]["value"] == pytest.approx(math.sinh(phi / 2) / (0.5 * math.sinh(phi)), rel=1e-3)
    assert res["domain_measure"] == pytest.approx(4 * PI / 3, rel=1e-12)


def test_nonlinear_reaction_robin_and_surface_flux():
    """Second-order reaction in a slab (reference: scipy solve_bvp on the 1-D problem), Biot condition, nonlinear flux."""
    phi = 2.0
    ref = solve_bvp(lambda x, y: np.vstack([y[1], phi ** 2 * y[0] ** 2]), lambda ya, yb: np.array([ya[1], yb[0] - 1]),
                    np.linspace(0, 1, 200), np.ones((2, 200)), tol=1e-10)
    res = solve_pde_2d("cartesian", {"x": [0, 1], "y": [0, 1]}, {"c": {"D": 1, "source": "-phi^2*c^2"}},
                       {"x_min": SYM, "x_max": one(), "y_min": SYM, "y_max": SYM}, parameters={"phi": phi},
                       grid={"n": [32, 8]}, outputs={"averages": [{"name": "eta", "expression": "c^2"}], "points": [[0, 0.5]]})
    check_evidence(res, linear=False)
    assert res["evidence"]["nonlinear"]["iterations_by_grid"][0] > 1
    assert any("no lineal" in w for w in res["warnings"])
    assert res["averages"]["eta"]["average"]["value"] == pytest.approx(ref.sol(1.0)[1] / phi ** 2, rel=5e-4)
    assert res["points"][0]["c"]["value"] == pytest.approx(ref.sol(0.0)[0], rel=5e-4)
    # conduction with a film: u(1) = 1/(1 + Bi)
    bc = {"x_min": one(), "y_min": SYM, "y_max": SYM}
    res = solve_pde_2d("cartesian", {"x": [0, 1], "y": [0, 1]}, {"u": {"D": 1}},
                       {**bc, "x_max": {"type": "robin", "h": "Bi", "u_ext": 0}}, parameters={"Bi": 2.0},
                       grid={"n": 16}, outputs={"points": [[1, 0.5]]})
    assert res["points"][0]["u"]["value"] == pytest.approx(1 / 3, rel=1e-9)
    # surface reaction -du/dx = k u^2 at x = 1 with u(0) = 1: 1 - w = k w^2 -> w = 1/2 for k = 2
    res = solve_pde_2d("cartesian", {"x": [0, 1], "y": [0, 1]}, {"u": {"D": 1}},
                       {**bc, "x_max": {"type": "flux", "value": "k*u^2"}}, parameters={"k": 2.0}, grid={"n": 16},
                       outputs={"points": [[1, 0.5]], "side_fluxes": [{"name": "s", "side": "x_max"}]})
    check_evidence(res, linear=False)
    assert res["points"][0]["u"]["value"] == pytest.approx(0.5, rel=1e-8)
    assert res["side_fluxes"]["s"]["mean_flux_out"]["value"] == pytest.approx(0.5, rel=1e-8)


def test_coupled_variables():
    """A -> B with equal diffusivities, a = 1 and b = 0 on the surface: a + b = 1 everywhere."""
    surf = {"a": one(1), "b": one(0)}
    res = solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, 1]},
                       {"a": {"D": 1, "source": "-4*a"}, "b": {"D": 1, "source": "4*a"}},
                       {"r_max": surf, "z_max": surf, "z_min": SYM}, grid={"n": 16},
                       outputs={"points": [[0, 0]], "averages": [{"name": "s", "expression": "a+b"}]})
    check_evidence(res)
    assert res["averages"]["s"]["average"]["value"] == pytest.approx(1.0, abs=1e-10)
    assert res["points"][0]["a"]["value"] + res["points"][0]["b"]["value"] == pytest.approx(1.0, abs=1e-10)
    assert 0 < res["points"][0]["a"]["value"] < 1


# ---------------------------------------------------------------------------------------------------------------------
# convection-diffusion
# ---------------------------------------------------------------------------------------------------------------------
def graetz_theta_m(xp):
    lam2 = (7.3136, 44.61, 113.9, 215.2)
    G = (0.7488, 0.544, 0.463, 0.414)
    return sum(8 * g / l * math.exp(-l * xp) for g, l in zip(G, lam2))


def test_graetz_problem():
    """Dimensionless: r in [0,1], v_z = 2 (1 - r^2), radial diffusivity 1, no axial conduction -> x+ = z/2."""
    zs = [0.1, 0.2, 0.5]
    out = {"mixing_cup": [{"name": "Tb", "expression": "T", "coordinate": "z", "at": zs},
                          {"name": "Tend", "expression": "T", "coordinate": "z", "at": 1.0}],
           "side_fluxes": [{"name": "qw", "side": "r_max", "at": 1.0}], "derived": {"Nu": "2*qw/Tend"}}
    bcs = {"r_max": one(0), "z_min": one(1), "z_max": {"type": "neumann", "value": 0}}
    grid = {"n": [32, 64], "grading": {"z": {"cluster": "min", "strength": 2}, "r": {"cluster": "max", "strength": 1.5}}}
    res = solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, 1.2]}, {"T": {"D": {"r": 1, "z": 0}}}, bcs,
                       velocity={"z": "2*(1-r^2)"}, outputs=out, grid=grid)
    check_evidence(res)                      # linear PDE; the TVD limiter is iterated (semismooth Newton)
    assert min(res["evidence"]["nonlinear"]["iterations_by_grid"]) > 1
    conv = res["evidence"]["convection"]
    assert conv["scheme"] == "tvd" and math.isinf(conv["max_cell_peclet_base_grid"])
    assert "van Leer" in conv["scheme_description"] and conv["velocity_divergence_relative"] < 1e-12
    for item, z in zip(res["mixing_cup"], zs):
        assert item["mixing_cup"]["grid_converged"]
        assert item["mixing_cup"]["value"] == pytest.approx(graetz_theta_m(z / 2), abs=6e-4)
        assert item["flow_through_section"] == pytest.approx(PI, rel=2e-3)          # mean velocity 1 on area pi
    nu = res["derived"]["Nu"]
    assert nu["grid_converged"] and nu["value"] == pytest.approx(3.657, rel=2e-3)
    # energy balance: what enters by convection minus what leaves = heat to the wall
    bal = res["evidence"]["global_balance"]["T"]
    assert bal["convection_term_integral"] == pytest.approx(-bal["diffusive_flux_out_by_side"]["r_max"], rel=1e-9)
    # true problem at high Peclet (isotropic D = 1, i.e. Pe_R -> axial conduction negligible when z is scaled): the
    # first-order upwind scheme is declared and is visibly less accurate than TVD on the same grid
    up = solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, 1.2]}, {"T": {"D": {"r": 1, "z": 0}}}, bcs,
                      velocity={"z": "2*(1-r^2)"}, outputs=out, grid=grid, convection_scheme="upwind")
    assert up["evidence"]["convection"]["scheme"] == "upwind" and any("difusión numérica" in w for w in up["warnings"])
    err = lambda r: abs(r["mixing_cup"][0]["mixing_cup"]["value"] - graetz_theta_m(0.05))
    assert err(up) > 3 * err(res)


def graetz_modes(n_modes=6):
    """Independent reference for the Graetz problem, computed here (not taken from a table): eigenpairs of
    (r R')' + lam^2 r (1 - r^2) R = 0, R'(0) = 0, R(1) = 0, by shooting (scipy DOP853 from the series start
    R = 1 - lam^2 r^2/4 at r = 1e-6, root of R(1) with brentq). Returns [(lam, C, R'(1), I1)] with
    I1 = int r (1 - r^2) R dr and C = I1 / int r (1 - r^2) R^2 dr (expansion coefficient of the inlet value 1).
    Then theta = sum C R exp(-lam^2 z/2) for 2 (1 - r^2) dT/dz = (1/r) d/dr(r dT/dr), the cup-mixing value is
    4 sum C I1 exp(..) and Nu(z) = -2 dT/dr(1) / theta_m; Nu_inf = lam_1^2 / 2."""
    def shoot(lam, full=False):
        e = 1e-6
        f = lambda r, y: [y[1] / r, -lam ** 2 * r * (1 - r * r) * y[0], r * (1 - r * r) * y[0], r * (1 - r * r) * y[0] ** 2]
        sol = solve_ivp(f, (e, 1.0), [1 - lam ** 2 * e * e / 4, -lam ** 2 * e * e / 2, e * e / 2, e * e / 2],
                        method="DOP853", rtol=1e-12, atol=1e-14)
        return sol.y[:, -1] if full else sol.y[0, -1]
    grid = np.arange(1.0, 4.0 * n_modes + 2.0, 0.5)
    vals = [shoot(x) for x in grid]
    modes = []
    for x0, x1, v0, v1 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if v0 * v1 < 0 and len(modes) < n_modes:
            lam = brentq(shoot, x0, x1, xtol=1e-13)
            _, dR, I1, I2 = shoot(lam, full=True)             # y[1] = r R' = R'(1) at r = 1
            modes.append((lam, I1 / I2, dR, I1))
    return modes


def graetz_nu(z, modes):
    e = [math.exp(-lam ** 2 * z / 2) for lam, *_ in modes]
    return (-2 * sum(C * dR * x for (_, C, dR, _), x in zip(modes, e))
            / (4 * sum(C * I1 * x for (_, C, _, I1), x in zip(modes, e))))


NU_TUBE = "2*wall_gradient(T, r_max, z)/(wall_value(T, r_max, z) - mixing_cup(T, z, z))"
GRAETZ_GRID = {"n": [32, 64], "grading": {"z": {"cluster": "min", "strength": 2}, "r": {"cluster": "max", "strength": 1.5}}}


def test_local_nusselt_in_one_derived_expression_graetz_tube():
    """Graetz problem (Poiseuille flow, constant wall temperature, no axial conduction). The local Nusselt number
    Nu(z) = (dT/dn)|wall D / (T_wall - T_mix(z)) (n = outward normal) is ONE derived expression that mixes a wall-point normal derivative, a
    wall value and a cup-mixing average. Reference: eigenfunction series with the eigenpairs computed in graetz_modes
    (lam_1 = 2.7043644, Nu_inf = lam_1^2/2 = 3.6567935; the literature value 3.657 is the same number rounded).
    Tolerances from the refinement evidence of this grid (32x64 -> 64x128): the reported change on refinement is 4e-4
    for z >= 0.4 and 3e-3 at z = 0.05; the Richardson value agrees with the series to 1e-6."""
    modes = graetz_modes()
    lam1 = modes[0][0]
    assert lam1 == pytest.approx(2.7043644, abs=2e-7) and lam1 ** 2 / 2 == pytest.approx(3.657, abs=5e-4)
    assert [m[0] for m in modes[1:3]] == pytest.approx([6.679, 10.673], abs=2e-3)       # Kays & Crawford table
    zs = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
    res = solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, 2]}, {"T": {"D": {"r": 1, "z": 0}}},
                       {"r_max": one(0), "z_min": one(1), "z_max": {"type": "neumann", "value": 0}},
                       velocity={"z": "2*(1-r^2)"}, grid=GRAETZ_GRID,
                       outputs={"mixing_cup": [{"name": "Tend", "expression": "T", "coordinate": "z", "at": 2.0}],
                                "side_fluxes": [{"name": "qw", "side": "r_max", "at": 2.0}],
                                "derived": {"Nu": {"expression": NU_TUBE, "along": "z", "at": zs},
                                            "Nu_end": "2*wall_gradient(T, r_max, 2.0)/(0 - mixing_cup(T, z, 2.0))",
                                            "Nu_old_way": "2*qw/Tend"}})
    check_evidence(res)
    prof = res["derived"]["Nu"]
    assert prof["along"] == "z" and [v["z"] for v in prof["values"]] == zs and res["grid_converged"]
    for v in prof["values"]:
        ref = graetz_nu(v["z"], modes)
        assert v["grid_converged"] and v["value"] == pytest.approx(ref, rel=1.5e-3 if v["z"] < 0.2 else 5e-4), v["z"]
    nu_inf = lam1 ** 2 / 2
    assert prof["values"][-1]["value"] == pytest.approx(nu_inf, rel=5e-4)
    assert Q(res, "derived:Nu@z=2")["richardson_extrapolated"] == pytest.approx(nu_inf, rel=1e-5)
    assert prof["values"][0]["value"] > prof["values"][1]["value"] > prof["values"][2]["value"] > nu_inf   # entrance
    # the inline form and the two-named-outputs form are the same number
    assert res["derived"]["Nu_end"]["value"] == pytest.approx(prof["values"][-1]["value"], rel=1e-12)
    assert res["derived"]["Nu_old_way"]["value"] == pytest.approx(prof["values"][-1]["value"], rel=1e-12)


def test_local_nusselt_in_one_derived_expression_parallel_plates():
    """Same problem between parallel plates (half gap 1, u = 1.5 (1 - y^2), symmetry at y = 0): with the hydraulic
    diameter 4 (twice the gap) Nu = 4 (dT/dn)|wall / (T_wall - T_mix) -> (8/3) lam_1^2 = 7.5407, lam_1 = 1.6815953 the
    first root of Y'' + lam^2 (1 - y^2) Y = 0, Y'(0) = 0, Y(1) = 0, found here by shooting. (Energy balance of the
    asymptotic mode T = Y(y) exp(-b x) with lam^2 = 1.5 b: -dT/dy(1)/T_mix = b.) Literature: 7.541."""
    def shoot(lam):
        return solve_ivp(lambda y, w: [w[1], -lam ** 2 * (1 - y * y) * w[0]], (0, 1), [1.0, 0.0], method="DOP853",
                         rtol=1e-12, atol=1e-14).y[0, -1]
    lam1 = brentq(shoot, 1.0, 2.5, xtol=1e-13)
    nu_inf = 8 * lam1 ** 2 / 3
    assert lam1 == pytest.approx(1.6815953, abs=2e-7) and nu_inf == pytest.approx(7.541, abs=5e-4)
    res = solve_pde_2d("cartesian", {"x": [0, 3], "y": [0, 1]}, {"T": {"D": {"x": 0, "y": 1}}},
                       {"y_max": one(0), "y_min": SYM, "x_min": one(1), "x_max": {"type": "neumann", "value": 0}},
                       velocity={"x": "1.5*(1-y^2)"},
                       grid={"n": [64, 32], "grading": {"x": {"cluster": "min", "strength": 2},
                                                        "y": {"cluster": "max", "strength": 1.5}}},
                       outputs={"derived": {"Nu": {"expression": "4*wall_gradient(T, y_max, x)/(0 - mixing_cup(T, x, x))",
                                                   "along": "x", "at": [2.0, 3.0]}}})
    check_evidence(res)
    for v in res["derived"]["Nu"]["values"]:                 # reported change on refinement: 3e-4
        assert v["grid_converged"] and v["value"] == pytest.approx(nu_inf, rel=5e-4)
    assert Q(res, "derived:Nu@x=3")["richardson_extrapolated"] == pytest.approx(nu_inf, rel=1e-5)


def test_inline_measurements_equal_the_named_outputs():
    res = solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, 1]}, {"T": {"D": {"r": 1, "z": 0}}},
                       {"r_max": one(0), "z_min": one(1), "z_max": {"type": "neumann"}}, velocity={"z": "2*(1-r^2)"},
                       parameters={"zc": 0.5}, grid={"n": [16, 32]},
                       outputs={"points": [[0.3, 0.5]], "side_fluxes": [{"name": "qw", "side": "r_max", "at": 0.5}],
                                "mixing_cup": [{"name": "Tb", "expression": "T^2", "coordinate": "z", "at": 0.5}],
                                "derived": {"a": "value(T*r, 0.3, zc)", "b": "wall_flux(T, r_max, zc)",
                                            "c": "wall_gradient(T, r_max, 0.5)", "d": "wall_value(T, r_max, zc)",
                                            "e": "mixing_cup(T^2, z, zc)", "f": "area_average(T^2, z, 2*zc - 0.5)",
                                            "g": "qw_gradient/(Tb - area_average(T^2, z, zc)) + value(T, 0.3, 0.5)"}})
    d = {k: v["value"] for k, v in res["derived"].items()}
    T = res["points"][0]["T"]["value"]
    q, mc = res["side_fluxes"]["qw"], res["mixing_cup"][0]
    assert d["a"] == pytest.approx(0.3 * T, rel=1e-13) and d["b"] == pytest.approx(q["flux_out"]["value"], rel=1e-13)
    assert d["c"] == pytest.approx(q["normal_derivative"]["value"], rel=1e-13) and d["d"] == pytest.approx(0.0, abs=1e-14)
    assert d["e"] == pytest.approx(mc["mixing_cup"]["value"], rel=1e-13)
    assert d["f"] == pytest.approx(mc["area_average"]["value"], rel=1e-13)
    assert d["g"] == pytest.approx(d["c"] / (d["e"] - d["f"]) + T, rel=1e-12)


def test_wall_temperature_jump_needs_grading_towards_it():
    """Wall at T = 1 (as the inlet) up to z = 1, then 0: downstream of the jump this IS the Graetz problem with the
    origin at z = 1, so Nu(1 + s) must equal the series value Nu(s) of graetz_modes. On the default uniform grid the
    local values near the jump are not grid-converged and the warning says why and what to do; with the grid graded
    towards the jump they converge to the series (tolerance = refinement_tol of the run, 1 %; observed 0.1-0.3 %)."""
    modes = graetz_modes(8)
    kw = dict(coordinates="axisymmetric", domain={"r": [0, 1], "z": [0, 2]}, equations={"T": {"D": {"r": 1, "z": 0}}},
              boundary_conditions={"r_max": [{"from": 0, "to": 1, **one(1)}, {"from": 1, "to": 2, **one(0)}],
                                   "z_min": one(1), "z_max": {"type": "neumann"}}, velocity={"z": "2*(1-r^2)"},
              outputs={"derived": {"Nu": {"expression": NU_TUBE, "along": "z", "at": [1.05, 1.1, 1.5]}}})
    plain = solve_pde_2d(**kw)
    bad = [v["z"] for v in plain["derived"]["Nu"]["values"] if not v["grid_converged"]]
    assert bad == [1.05, 1.1] and not plain["grid_converged"]
    jump = plain["evidence"]["grid_refinement"]["boundary_condition_jumps"]
    assert len(jump) == 1 and jump[0]["position"] == 1.0 and jump[0]["values"] == [1.0, 0.0]
    assert not jump[0]["graded_towards_it"]
    assert any("'cluster': 'at', 'position': 1" in w and "discontinua" in w for w in plain["warnings"])
    graded = solve_pde_2d(**kw, grid={"grading": {"z": {"cluster": "at", "position": 1.0, "strength": 2}}})
    assert graded["grid_converged"] and not any("discontinua" in w for w in graded["warnings"])
    j = graded["evidence"]["grid_refinement"]["boundary_condition_jumps"][0]
    assert j["graded_towards_it"] and j["cell_size_fine_grid"] < jump[0]["cell_size_fine_grid"] / 30
    for v in graded["derived"]["Nu"]["values"]:
        assert v["value"] == pytest.approx(graetz_nu(v["z"] - 1.0, modes), rel=1e-2), v["z"]
    # the grading is nested (z = 1 is a face of the three grids) and monotone
    from packages.transport.pde2d import _Problem
    P = _Problem(kw["coordinates"], kw["domain"], kw["equations"], kw["boundary_conditions"], kw["velocity"], None, {},
                 {"grading": {"z": {"cluster": "at", "position": 1.0, "strength": 2}}}, None, "auto", 0.01, 60)
    f32, f64 = P.faces(1, 32), P.faces(1, 64)
    assert np.all(np.diff(f64) > 0) and np.allclose(f64[::2], f32, atol=1e-15) and np.any(f32 == 1.0)


def test_wall_temperature_changing_along_z():
    """Piecewise Dirichlet wall (1 for z < 1, 0.5 after) with low cell Peclet: central scheme; far downstream of each
    step the cup-mixing temperature tends to the local wall value and stays between the bounds."""
    res = solve_pde_2d("axisymmetric", {"r": [0, 1], "z": [0, 4]}, {"T": {"D": 0.5}},
                       {"r_max": [{"from": 0, "to": 1, **one(1)}, {"from": 1, "to": 4, **one(0.5)}],
                        "z_min": one(0), "z_max": {"type": "neumann"}}, velocity={"z": "2*(1-r^2)"}, grid={"n": [16, 64]},
                       outputs={"mixing_cup": [{"name": "Tb", "expression": "T", "coordinate": "z", "at": [1.0, 4.0]}]})
    check_evidence(res)
    assert res["evidence"]["convection"]["scheme"] == "central"
    t1, t4 = (m["mixing_cup"]["value"] for m in res["mixing_cup"])
    assert 0.5 < t1 < 1.0 and t4 == pytest.approx(0.5, abs=5e-3)
    assert res["extrema"]["T"]["max"]["value"] <= 1.0 + 1e-9 and res["extrema"]["T"]["min"]["value"] >= -1e-9


STOKES = {"r": "cos(theta)*(1-1.5/r+0.5/r^3)", "theta": "-sin(theta)*(1-0.75/r-0.25/r^3)"}


def sphere_nu(pe_d, r_out, n):
    """Sphere of radius 1 at T = 1 in a uniform stream (U = 1 along +z) at T = 0; D = 1/Pe_radius. The outer boundary
    is T = 0 where the stream enters (theta > pi/2) and zero gradient where it leaves."""
    bcs = {"r_min": one(1)}
    if pe_d == 0:
        bcs["r_max"] = one(0)
        return solve_pde_2d("spherical_axisymmetric", {"r": [1, r_out]}, {"T": {"D": 1}}, bcs, grid={"n": n},
                            outputs={"side_fluxes": [{"name": "q", "side": "r_min"}], "derived": {"Nu": "-2*q"}})
    bcs["r_max"] = [{"from": 0, "to": PI / 2, "type": "neumann", "value": 0}, {"from": PI / 2, "to": 4, **one(0)}]
    return solve_pde_2d("spherical_axisymmetric", {"r": [1, r_out]}, {"T": {"D": "2/Pe"}}, bcs, velocity=STOKES,
                        parameters={"Pe": pe_d}, grid={"n": n},
                        outputs={"side_fluxes": [{"name": "q", "side": "r_min"}], "derived": {"Nu": "-q*Pe"}})


def test_sphere_conduction_to_infinity():
    """Exact for the truncated shell: Nu = 2/(1 - R/R_out); the truncation error is R/R_out (5 % at R_out = 20 R) and
    a linear extrapolation in 1/R_out gives Nu = 2."""
    nus = {}
    for r_out in (10.0, 20.0, 40.0):
        res = sphere_nu(0, r_out, [48, 8])
        check_evidence(res)
        assert res["evidence"]["grid_refinement"]["grading"]["r"] == {"type": "log"}
        nu = res["derived"]["Nu"]
        assert nu["grid_converged"] and nu["value"] == pytest.approx(2 / (1 - 1 / r_out), rel=2e-3)
        nus[r_out] = Q(res, "derived:Nu")["richardson_extrapolated"]
        assert nus[r_out] == pytest.approx(2 / (1 - 1 / r_out), rel=2e-5)
    assert nus[40.0] - 2 == pytest.approx(2 / 39, rel=1e-3)                        # the truncation effect itself
    extrap = nus[40.0] + (nus[40.0] - nus[20.0]) * (0 - 1 / 40) / (1 / 40 - 1 / 20)
    assert extrap == pytest.approx(2.0, abs=3e-3)
    assert res["side_fluxes"]["q"]["area"] == pytest.approx(4 * PI, rel=1e-12)


def test_sphere_creeping_flow_heat_transfer():
    # low Pe: Acrivos & Taylor (1962)
    pe = 0.2
    at = 2 + pe / 2 + pe ** 2 * math.log(pe) / 4 + 0.03404 * pe ** 2 + pe ** 3 * math.log(pe) / 16
    res = sphere_nu(pe, 400.0, [64, 32])
    check_evidence(res)
    assert res["derived"]["Nu"]["grid_converged"]
    assert res["derived"]["Nu"]["value"] == pytest.approx(at, rel=5e-3)
    # moderate Pe: Clift, Grace & Weber (1978) correlation of numerical solutions
    for pe, r_out in ((10.0, 100.0), (100.0, 50.0)):
        res = sphere_nu(pe, r_out, [64, 32])
        check_evidence(res)
        nu = res["derived"]["Nu"]
        assert nu["grid_converged"] and res["evidence"]["convection"]["scheme"] == "tvd"
        assert nu["value"] == pytest.approx(1 + (1 + pe) ** (1 / 3), rel=0.02)
    # high Pe: Acrivos & Goddard (1965) boundary-layer asymptote with its first correction
    res = sphere_nu(1000.0, 30.0, [96, 64])
    assert res["derived"]["Nu"]["value"] == pytest.approx(0.991 * 1000 ** (1 / 3) + 0.922, rel=0.01)
    assert res["evidence"]["convection"]["velocity_divergence_relative"] < 1e-3


# ---------------------------------------------------------------------------------------------------------------------
# input errors and wrappers
# ---------------------------------------------------------------------------------------------------------------------
BASE = dict(coordinates="cartesian", domain={"x": [0, 1], "y": [0, 1]}, equations={"u": {"D": 1}},
            boundary_conditions={"x_min": one(0), "x_max": one(1), "y_min": SYM, "y_max": SYM}, grid={"n": 8})


@pytest.mark.parametrize("change, message", [
    (dict(coordinates="cilindricas"), "no reconocido; opciones"),
    (dict(domain={"x": [0, 1]}), "falta 'y'"),
    (dict(domain={"x": [1, 0], "y": [0, 1]}), "debe ser mayor que el mínimo"),
    (dict(equations={"u": {"D": "k"}}), "el nombre 'k' no está declarado"),
    (dict(equations={"u": {"D": "__import__('os')"}}), "no admitida"),
    (dict(equations={"u": {"D": "x.real"}}), "elemento no permitido"),
    (dict(equations={"u": {"D": "x[0]"}}), "elemento no permitido"),
    (dict(equations={"u": {"D": "-1"}}), "negativa"),
    (dict(equations={"u": {"D": "1/(x-x)"}}), "no da un valor finito"),
    (dict(equations={"u": {}}), "falta 'D'"),
    (dict(boundary_conditions={"x_min": one(0)}), "falta la condición de frontera"),
    (dict(boundary_conditions={"r_max": one(0)}), "los lados son x_min, x_max, y_min, y_max"),
    (dict(boundary_conditions={s: SYM for s in ("x_min", "x_max", "y_min", "y_max")}), "no queda determinado"),
    (dict(boundary_conditions={"x_min": [{"from": 0, "to": 0.4, **one(0)}], "x_max": SYM, "y_min": SYM, "y_max": SYM}),
     "no cubren todo el lado"),
    (dict(boundary_conditions={"x_min": {"type": "robin", "h": 1}, "x_max": SYM, "y_min": SYM, "y_max": SYM}),
     "Robin necesita"),
    (dict(boundary_conditions={"x_min": {"type": "periodic"}, "x_max": SYM, "y_min": SYM, "y_max": SYM}), "tipo 'periodic'"),
    (dict(outputs={"points": [[2, 2]]}), "fuera del dominio"),
    (dict(outputs={"mixing_cup": [{"name": "m", "expression": "u", "coordinate": "x", "at": 0.5}]}), "necesita velocity"),
    (dict(outputs={"side_fluxes": [{"name": "q", "side": "top"}]}), "no es un lado"),
    (dict(outputs={"derived": {"z": "nada*2"}}), "no está declarado"),
    (dict(outputs={"derived": {"z": "mixing_cup(u, x, 0.5)"}}), "mixing_cup necesita velocity"),
    (dict(outputs={"derived": {"z": "wall_gradient(u, top, 0.5)"}}), "el lado debe ser uno de x_min"),
    (dict(outputs={"derived": {"z": "wall_gradient(w, x_max, 0.5)"}}), "la variable debe ser uno de u"),
    (dict(outputs={"derived": {"z": "value(u, 0.5, 7)"}}), "fuera del dominio"),
    (dict(outputs={"derived": {"z": "area_average(u, x)"}}), "se espera area_average(expresión"),
    (dict(outputs={"derived": {"z": "value(nada, 0.5, 0.5)"}}), "no está declarado"),
    (dict(outputs={"derived": {"z": {"expression": "value(u, x, 0.5)", "along": "x", "at": [0.5, 3]}}}), "posiciones dentro"),
    (dict(outputs={"derived": {"z": {"expression": "value(u, x, 0.5)", "along": "q", "at": [0.5]}}}), "'along'"),
    (dict(outputs={"derived": {"z": "integral(u)"}}), "las mediciones area_average"),
    (dict(outputs={"averages": [{"name": "m", "expression": "value(u, 0.5, 0.5)"}]}), "no admitida"),
    (dict(grid={"grading": {"x": {"cluster": "at", "strength": 2}}}), "'position' se da con"),
    (dict(grid={"grading": {"x": {"cluster": "at", "position": 5}}}), "debe estar dentro"),
    (dict(grid={"n": [4000, 4000]}), "incógnitas"),
    (dict(grid={"n": [2, 2]}), "mínimo 8"),
    (dict(convection_scheme="quick"), "no reconocido"),
    (dict(velocity={"z": "1"}), "velocity: se espera"),
    (dict(parameters={"x": 1.0}), "nombres repetidos"),
])
def test_bad_input_messages(change, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        solve_pde_2d(**{**BASE, **change})


def test_polar_and_spherical_domain_errors():
    with pytest.raises(ValueError, match="negativo"):
        solve_pde_2d("polar", {"r": [-1, 1], "theta": [0, 1]}, {"u": {"D": 1}}, {})
    with pytest.raises(ValueError, match="2·pi"):
        solve_pde_2d("polar", {"r": [0, 1], "theta": [0, 7]}, {"u": {"D": 1}}, {})
    with pytest.raises(ValueError, match="de 0 a pi"):
        solve_pde_2d("spherical_axisymmetric", {"r": [1, 2], "theta": [0, 4]}, {"u": {"D": 1}}, {})
    with pytest.raises(ValueError, match="'log' necesita"):
        solve_pde_2d("spherical_axisymmetric", {"r": [0, 2]}, {"u": {"D": 1}}, {"r_max": one(1)},
                     grid={"grading": {"r": "log"}})


def test_tool_and_wrapper():
    assert tools_pde2d.PDE2D_TOOLS == [tools_pde2d.solve_pde_2d]
    assert len(wrappers_pde2d.AGENT_WRAPPERS) == 1 and wrappers_pde2d.AGENT_WRAPPERS[0].name == "solve_pde_2d"
    doc = tools_pde2d.solve_pde_2d.__doc__
    for word in ("spherical_axisymmetric", "theta_max", "mixing_cup", "grid_converged", "robin"):
        assert word in doc
    kw = {**BASE, "outputs": {"points": [[0.25, 0.5]]}}
    out = json.loads(wrappers_pde2d.solve_pde_2d(**kw))
    assert out["points"][0]["u"]["value"] == pytest.approx(0.25, abs=1e-12)
    assert set(out["evidence"]) >= {"grid_refinement", "nonlinear", "global_balance", "convection"}
