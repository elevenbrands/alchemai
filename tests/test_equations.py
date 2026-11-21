"""General equation solvers (bloque EQ): solve_ode_system, solve_bvp_system, solve_pde_1d.

Every number is checked against an independent exact or published result, computed in the test itself from its closed
form or series (never from the solver under test):

  ODE
  * First-order decay c' = −k·c: c = e^{−kt}; half-life ln 2 / k (event).
  * Harmonic oscillator: x = sin t, maximum 1 at t = π/2.
  * Damped oscillator x'' + 2ζx' + x = 0: period 2π/√(1−ζ²), amplitude ratio per cycle exp(−2πζ/√(1−ζ²)).
  * van der Pol, μ = 1: limit-cycle period 6.6632868593 and amplitude 2.00861986 (high-precision values tabulated e.g.
    in A. Buonomo, SIAM J. Appl. Math. 59 (1998) 156, and reproduced by every stiff-solver test set).
  * Robertson stiff kinetics at t = 40: y = (0.7158270687, 9.185534764e-6, 0.2841637457) (Hairer & Wanner, «Solving
    Ordinary Differential Equations II», ROBER reference values).
  * y' = y², y(0) = 1: y = 1/(1 − t), blows up at t = 1.
  BVP
  * Axial dispersion + first-order reaction with Danckwerts conditions: Wehner & Wilhelm (1956) closed form
    c(1) = 4q·e^{Pe/2} / [(1+q)²e^{qPe/2} − (1−q)²e^{−qPe/2}], q = √(1 + 4Da/Pe) (Levenspiel, Fogler).
  * First-order reaction in a pellet: η = tanh φ/φ (slab), η = (3/φ²)(φ·coth φ − 1) (sphere) (Thiele; Aris).
  * Second-order reaction in a sphere: integral balance φ²∫c²x²dx = c'(1) (divergence theorem).
  * Blasius: f''(0) = 0.33206 (Howarth 1938; Schlichting).
  * Bratu y'' + λe^y = 0, y(0) = y(1) = 0: y(½) = 2·ln cosh(θ/4) with θ = √(2λ)·cosh(θ/4); two roots θ for
    λ < λc = 3.513830719 (Frank-Kamenetskii slab), none above.
  * Eigenvalue y'' + k²y = 0, y(0) = y(1) = 0: k = π.
  PDE
  * Slab and sphere, uniform initial temperature, convection (Biot) at the surface: one-term-and-beyond Fourier series
    (Carslaw & Jaeger; Incropera eqs. 5.42, 5.51) with eigenvalues ζ·tan ζ = Bi and 1 − ζ·cot ζ = Bi.
  * Slab with two different fixed faces: linear steady profile, flux D·Δu/L.
  * Slab with an oscillating face u(0,t) = sin ωt, u(L) = 0: periodic state amplitude |sinh κ(L−x)/sinh κL|,
    κ = √(iω/D), period 2π/ω (Carslaw & Jaeger §3.6).
  * Packed bed marching in z with radial dispersion, c and T coupled, Robin wall: with an impermeable wall
    c = e^{−Da·z}; T = Σ a_n(z)·J0(λ_n r), λ_n·J1(λ_n) = Bi·J0(λ_n), a_n = β·Da·g_n·(e^{−Da z} − e^{−μ_n z})/(μ_n − Da),
    μ_n = λ_n²/Pe_h, g_n = 2·J1(λ_n)/[λ_n(J0² + J1²)] (eigenfunction expansion, exact).
  * Transient axial-dispersion reactor run to steady state: the Wehner-Wilhelm outlet value above."""
import json
import math

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.special import j0, j1, jn_zeros

from packages.dwsim_client import tools_equations as TE


# ---------------------------------------------------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------------------------------------------------
def test_ode_first_order_decay_with_event_and_definitions():
    r = TE.solve_ode_system({"c": "-r"}, {"c": 1.0}, 5.0, parameters={"k": 0.7}, definitions={"r": "k*c"},
                            output_times=[0, 1, 2.5, 5], events=[{"expression": "c - 0.5", "name": "half"}])
    assert r["success"] and r["evidence"]["verified"]
    exact = np.exp(-0.7 * np.array([0, 1, 2.5, 5]))
    assert np.allclose(r["outputs"]["c"], exact, rtol=1e-6)
    assert np.allclose(r["outputs"]["r"], 0.7 * exact, rtol=1e-6)
    assert r["events"][0]["times"] == pytest.approx([math.log(2) / 0.7], rel=1e-7)
    assert r["extrema"]["c"]["max"] == pytest.approx(1.0) and r["extrema"]["c"]["t_at_max"] == 0.0


def test_ode_terminal_event_and_extrema():
    r = TE.solve_ode_system({"x": "v", "v": "-x"}, {"x": 0.0, "v": 1.0}, 10.0,
                            events=[{"expression": "x", "direction": -1, "terminal": True}])
    assert r["t_final"] == pytest.approx(math.pi, rel=1e-8)            # x = sin t falls through zero at π
    assert r["extrema"]["x"]["max"] == pytest.approx(1.0, abs=1e-7)
    assert r["extrema"]["x"]["t_at_max"] == pytest.approx(math.pi / 2, abs=1e-3)


def test_ode_damped_oscillation_is_diagnosed_as_decaying():
    z = 0.05
    r = TE.solve_ode_system({"x": "v", "v": "-2*z*v - x"}, {"x": 1.0, "v": 0.0}, 60.0, parameters={"z": z})
    d = r["oscillation"]["x"]
    wd = math.sqrt(1 - z * z)
    assert d["kind"] == "decaying_oscillation"
    assert d["period"] == pytest.approx(2 * math.pi / wd, rel=1e-4)
    assert d["amplitude_ratio_per_cycle"] == pytest.approx(math.exp(-2 * math.pi * z / wd), rel=1e-3)
    assert not r["steady_state"]["reached"]


def test_ode_van_der_pol_limit_cycle():
    r = TE.solve_ode_system({"x": "v", "v": "mu*(1 - x^2)*v - x"}, {"x": 0.5, "v": 0.0}, 120.0, parameters={"mu": 1.0})
    d = r["oscillation"]["x"]
    assert d["kind"] == "sustained_oscillation"
    assert d["period"] == pytest.approx(6.6632868593, rel=1e-5)
    assert d["amplitude"] == pytest.approx(2.00861986, rel=1e-5)
    assert r["evidence"]["verified"]


def test_ode_decay_to_steady_state_is_not_an_oscillation():
    r = TE.solve_ode_system({"c": "(1 - c)/tau - k*c"}, {"c": 0.0}, 200.0, parameters={"tau": 2.0, "k": 0.5})
    assert r["steady_state"]["reached"] and r["oscillation"]["c"]["kind"] == "steady"
    assert r["final"]["c"] == pytest.approx(1 / (1 + 0.5 * 2.0), rel=1e-8)          # CSTR: c = 1/(1 + k·tau)


def test_ode_stiff_robertson():
    r = TE.solve_ode_system({"a": "-0.04*a + 1e4*b*c", "b": "0.04*a - 1e4*b*c - 3e7*b^2", "c": "3e7*b^2"},
                            {"a": 1.0, "b": 0.0, "c": 0.0}, 40.0, rtol=1e-9, atol=1e-13)
    assert r["final"]["a"] == pytest.approx(0.7158270687, rel=1e-7)
    assert r["final"]["b"] == pytest.approx(9.185534764e-6, rel=1e-6)
    assert r["final"]["c"] == pytest.approx(0.2841637457, rel=1e-7)
    assert r["evidence"]["verified"]


def test_ode_blow_up_is_reported_plainly():
    r = TE.solve_ode_system({"y": "y^2"}, {"y": 1.0}, 2.0)
    assert r["success"] is False and not r["evidence"]["verified"]
    assert r["t_final"] == pytest.approx(1.0, abs=1e-4)
    assert "crece sin límite" in r["message"]


def test_ode_input_errors_are_clear():
    with pytest.raises(ValueError, match="el nombre 'k2' no está declarado"):
        TE.solve_ode_system({"c": "-k2*c"}, {"c": 1.0}, 1.0, parameters={"k": 1.0})
    with pytest.raises(ValueError, match="faltan \\['b'\\]"):
        TE.solve_ode_system({"a": "-a", "b": "a"}, {"a": 1.0}, 1.0)
    with pytest.raises(ValueError, match="elemento no permitido"):
        TE.solve_ode_system({"c": "c.real"}, {"c": 1.0}, 1.0)
    with pytest.raises(ValueError, match="función no admitida"):
        TE.solve_ode_system({"c": "__import__('os').getpid()"}, {"c": 1.0}, 1.0)
    with pytest.raises(ValueError, match="no dan un número finito en el estado inicial"):
        TE.solve_ode_system({"c": "1/c"}, {"c": 0.0}, 1.0)


def test_expression_functions():
    from packages.numerics.expressions import compile_expression
    env = {"x": np.array([-1.0, 0.5, 2.0]), "k": 3.0}
    f = compile_expression("max(x, 0) + min(x, 1, k) + where(x > 1, 10, 0) + heaviside(x)*abs(x) + 2^2 + pi*0",
                           {"x", "k"}, "prueba")
    assert np.allclose(f(env), [0 - 1 + 0 + 0 + 4, 0.5 + 0.5 + 0 + 0.5 + 4, 2 + 1 + 10 + 2 + 4])


# ---------------------------------------------------------------------------------------------------------------------
# BVP
# ---------------------------------------------------------------------------------------------------------------------
def _wehner_wilhelm(Pe, Da):
    q = math.sqrt(1 + 4 * Da / Pe)
    return 4 * q * math.exp(Pe / 2) / ((1 + q) ** 2 * math.exp(q * Pe / 2) - (1 - q) ** 2 * math.exp(-q * Pe / 2))


@pytest.mark.parametrize("Pe, Da", [(5.0, 2.0), (40.0, 3.0)])
def test_bvp_dispersion_reactor_danckwerts(Pe, Da):
    r = TE.solve_bvp_system({"c": "z", "z": "Pe*(z + Da*c)"}, [0, 1], ["c_a - z_a/Pe - 1", "z_b"],
                            parameters={"Pe": Pe, "Da": Da}, initial_guess={"c": 1.0})
    assert r["converged"] and r["evidence"]["verified"]
    assert r["boundary_values"]["c_b"] == pytest.approx(_wehner_wilhelm(Pe, Da), rel=1e-5)
    assert r["evidence"]["max_boundary_residual"] < 1e-8


@pytest.mark.parametrize("phi", [0.5, 2.0, 8.0])
def test_bvp_thiele_effectiveness_slab_and_sphere(phi):
    slab = TE.solve_bvp_system({"c": "z", "z": "phi^2*c"}, [0, 1], ["z_a", "c_b - 1"], parameters={"phi": phi},
                               initial_guess={"c": 1.0}, integrals={"eta": "c"})
    assert slab["integrals"]["eta"]["average"] == pytest.approx(math.tanh(phi) / phi, rel=1e-5)
    sphere = TE.solve_bvp_system({"c": "z", "z": "phi^2*c"}, [0, 1], ["z_a", "c_b - 1"], parameters={"phi": phi},
                                 singular_terms={"z": "-2*z"}, initial_guess={"c": 1.0}, integrals={"eta": "c"},
                                 integral_weight_power=2, output_points=[0.0, 1.0])
    assert sphere["converged"] and sphere["evidence"]["verified"]
    assert sphere["integrals"]["eta"]["average"] == pytest.approx(3 / phi ** 2 * (phi / math.tanh(phi) - 1), rel=1e-5)
    assert sphere["points"]["c"][0] == pytest.approx(phi / math.sinh(phi), rel=1e-5)        # centre: φ/sinh φ
    assert sphere["points"]["dc_dx"][1] == pytest.approx(phi / math.tanh(phi) - 1, rel=1e-5)  # surface gradient


def test_bvp_nonlinear_pellet_satisfies_the_integral_balance():
    phi = 3.0
    r = TE.solve_bvp_system({"c": "z", "z": "phi^2*c^2"}, [0, 1], ["z_a", "c_b - 1"], parameters={"phi": phi},
                            singular_terms={"z": "-2*z"}, initial_guess={"c": 1.0}, integrals={"rate": "c^2"},
                            integral_weight_power=2)
    assert r["converged"] and r["evidence"]["verified"]
    assert phi ** 2 * r["integrals"]["rate"]["integral"] == pytest.approx(r["boundary_values"]["z_b"], rel=1e-5)
    assert 0 < r["integrals"]["rate"]["average"] < 1


def test_bvp_singular_term_written_inline_is_explained():
    with pytest.raises(ValueError, match="singular_terms"):
        TE.solve_bvp_system({"c": "z", "z": "phi^2*c - 2/x*z"}, [0, 1], ["z_a", "c_b - 1"], parameters={"phi": 1.0})
    with pytest.raises(ValueError, match="combinación LINEAL"):
        TE.solve_bvp_system({"c": "z", "z": "c"}, [0, 1], ["z_a", "c_b - 1"], singular_terms={"z": "-2*z^2"})


def test_bvp_blasius():
    r = TE.solve_bvp_system({"f": "g", "g": "h", "h": "-0.5*f*h"}, [0, 10], ["f_a", "g_a", "g_b - 1"],
                            initial_guess={"f": "eta", "g": 1.0}, independent_variable="eta", output_points=[0, 5, 10])
    assert r["converged"] and r["evidence"]["verified"]
    assert r["boundary_values"]["h_a"] == pytest.approx(0.33206, abs=1e-5)
    assert r["points"]["g"][1] == pytest.approx(0.99155, abs=1e-5)             # f'(5), Howarth's table
    longer = TE.solve_bvp_system({"f": "g", "g": "h", "h": "-0.5*f*h"}, [0, 15], ["f_a", "g_a", "g_b - 1"],
                                 initial_guess={"f": "eta", "g": 1.0}, independent_variable="eta")
    assert longer["boundary_values"]["h_a"] == pytest.approx(r["boundary_values"]["h_a"], abs=1e-6)   # truncation


def _bratu_midpoints(lam):
    g = lambda th: th - math.sqrt(2 * lam) * math.cosh(th / 4)      # noqa: E731
    top = minimize_scalar(lambda th: -g(th), bounds=(0, 20), method="bounded").x
    return [2 * math.log(math.cosh(brentq(g, lo, hi) / 4)) for lo, hi in ((0, top), (top, 40))]


def test_bvp_bratu_two_solutions_from_two_guesses():
    low_exact, high_exact = _bratu_midpoints(1.0)
    assert (low_exact, high_exact) == (pytest.approx(0.140539, abs=1e-6), pytest.approx(4.091467, abs=1e-6))
    eq, bc = {"y": "z", "z": "-lam*exp(y)"}, ["y_a", "y_b"]
    low = TE.solve_bvp_system(eq, [0, 1], bc, parameters={"lam": 1.0}, output_points=[0.5])
    high = TE.solve_bvp_system(eq, [0, 1], bc, parameters={"lam": 1.0}, output_points=[0.5],
                               initial_guess={"y": "16*x*(1 - x)", "z": "16*(1 - 2*x)"})
    assert low["converged"] and high["converged"] and high["evidence"]["verified"]
    assert low["points"]["y"][0] == pytest.approx(low_exact, rel=1e-5)
    assert high["points"]["y"][0] == pytest.approx(high_exact, rel=1e-5)
    table = TE.solve_bvp_system(eq, [0, 1], bc, parameters={"lam": 1.0}, output_points=[0.5],
                                initial_guess={"y": {"x": [0, 0.5, 1], "values": [0, 4, 0]}})
    assert table["points"]["y"][0] == pytest.approx(high_exact, rel=1e-5)


def test_bvp_continuation_and_plain_non_convergence():
    eq, bc = {"y": "z", "z": "-lam*exp(y)"}, ["y_a", "y_b"]
    ok = TE.solve_bvp_system(eq, [0, 1], bc, parameters={"lam": 0.0}, output_points=[0.5],
                             continuation={"parameter": "lam", "values": [1, 2, 3, 3.4]})
    assert ok["converged"] and [s["lam"] for s in ok["continuation"]["steps"]] == [1, 2, 3, 3.4]
    assert ok["points"]["y"][0] == pytest.approx(_bratu_midpoints(3.4)[0], rel=1e-5)
    bad = TE.solve_bvp_system(eq, [0, 1], bc, parameters={"lam": 0.0}, max_nodes=2000,
                              continuation={"parameter": "lam", "values": [3.0, 4.0]})     # λ > λc = 3.5138: no solution
    assert bad["converged"] is False and "NO CONVERGIÓ" in bad["message"] and "points" not in bad
    last = [s for s in bad["continuation"]["steps"] if s["converged"]][-1]["lam"]
    assert 3.0 <= last < 3.513830719


def test_bvp_unknown_parameter_eigenvalue():
    r = TE.solve_bvp_system({"y": "z", "z": "-k^2*y"}, [0, 1], ["y_a", "y_b", "z_a - 1"], unknown_parameters={"k": 3.0},
                            initial_guess={"y": "sin(3*x)", "z": "cos(3*x)"})
    assert r["unknown_parameters"]["k"] == pytest.approx(math.pi, rel=1e-6)


def test_bvp_input_errors_are_clear():
    with pytest.raises(ValueError, match="se necesitan exactamente 2 condiciones de frontera .* se dieron 1"):
        TE.solve_bvp_system({"y": "z", "z": "-y"}, [0, 1], ["y_a"])
    with pytest.raises(ValueError, match="3 condiciones de frontera \\(2 variables \\+ 1 parámetros desconocidos\\)"):
        TE.solve_bvp_system({"y": "z", "z": "-k*y"}, [0, 1], ["y_a", "y_b"], unknown_parameters={"k": 1.0})
    with pytest.raises(ValueError, match="el nombre 'y_0' no está declarado"):
        TE.solve_bvp_system({"y": "z", "z": "-y"}, [0, 1], ["y_0", "y_b - 1"])
    with pytest.raises(ValueError, match="el nombre 'Pe' no está declarado"):
        TE.solve_bvp_system({"c": "z", "z": "Pe*z"}, [0, 1], ["c_a - 1", "z_b"])
    with pytest.raises(ValueError, match="a < b"):
        TE.solve_bvp_system({"y": "z", "z": "-y"}, [1, 0], ["y_a", "y_b"])


# ---------------------------------------------------------------------------------------------------------------------
# PDE
# ---------------------------------------------------------------------------------------------------------------------
def _slab_series(x, fo, Bi, n=40):
    out = np.zeros_like(np.asarray(x, float))
    for k in range(n):
        z = brentq(lambda z: z * math.tan(z) - Bi, k * math.pi + 1e-12, k * math.pi + math.pi / 2 - 1e-12)
        out += 4 * math.sin(z) / (2 * z + math.sin(2 * z)) * math.exp(-z * z * fo) * np.cos(z * x)
    return out


def _sphere_series(r, fo, Bi, n=40):
    """Returns (θ(r), volume average)."""
    r = np.asarray(r, float)
    out, avg = np.zeros_like(r), 0.0
    for k in range(n):
        z = brentq(lambda z: z * math.cos(z) - (1 - Bi) * math.sin(z), k * math.pi + 1e-9, (k + 1) * math.pi - 1e-9)
        C = 4 * (math.sin(z) - z * math.cos(z)) / (2 * z - math.sin(2 * z))
        out += C * math.exp(-z * z * fo) * np.sinc(z * r / math.pi)
        avg += C * math.exp(-z * z * fo) * 3 * (math.sin(z) - z * math.cos(z)) / z ** 3
    return out, avg


def test_pde_slab_with_biot_condition_matches_series():
    Bi, pts, times = 2.0, [0.0, 0.5, 1.0], [0.1, 0.5]
    r = TE.solve_pde_1d(["T"], [0, 1], 0.5, {"T": 1.0},
                        {"left": {"T": {"type": "symmetry"}}, "right": {"T": {"type": "flux", "inflow": "-Bi*T"}}},
                        {"T": 1.0}, parameters={"Bi": Bi}, output_times=times, output_points=pts)
    assert r["success"] and r["evidence"]["verified"]
    for i, t in enumerate(times):
        assert np.allclose(r["profiles"]["T"][i], _slab_series(np.array(pts), t, Bi), atol=2e-5)
    assert r["boundary"]["right"]["T"]["inflow"][1] == pytest.approx(-Bi * _slab_series(np.array([1.0]), 0.5, Bi)[0],
                                                                    rel=1e-4)
    assert not r["steady_state"]["reached"]


def test_pde_sphere_with_biot_condition_matches_series():
    Bi, pts = 1.5, [0.0, 0.5, 1.0]
    r = TE.solve_pde_1d(["T"], [0, 0.02], 60.0, {"T": 1.0},
                        {"right": {"T": {"type": "robin", "inflow": "h*(0 - T)"}}}, {"T": "alpha"}, geometry="sphere",
                        parameters={"alpha": 1e-6, "h": 1.5 * 1e-6 / 0.02}, output_times=[20.0, 60.0],
                        output_points=[0.0, 0.01, 0.02], space_variable="r")
    assert r["success"] and r["evidence"]["verified"]
    for i, t in enumerate([20.0, 60.0]):
        exact, avg = _sphere_series(np.array(pts), 1e-6 * t / 0.02 ** 2, Bi)
        assert np.allclose(r["profiles"]["T"][i], exact, atol=5e-5)
        assert r["averages"]["T"][i] == pytest.approx(avg, abs=5e-5)


def test_pde_two_dirichlet_faces_reach_the_linear_steady_state():
    r = TE.solve_pde_1d(["u"], [0, 2], 40.0, {"u": 0.0},
                        {"left": {"u": {"type": "dirichlet", "value": 1.0}},
                         "right": {"u": {"type": "dirichlet", "value": 3.0}}}, {"u": 0.5}, output_times=[40.0])
    x = np.array(r["profiles"]["x"])
    assert np.allclose(r["profiles"]["u"][0], 1 + x, atol=1e-7)
    assert r["steady_state"]["reached"] and 0 < r["steady_state"]["settling_time"] < 40
    assert r["oscillation"]["average[u]"]["kind"] == "steady"
    assert r["averages"]["u"][0] == pytest.approx(2.0, abs=1e-7)
    assert r["boundary"]["left"]["u"]["inflow"][0] == pytest.approx(-0.5 * 1.0, rel=1e-6)   # D·Δu/L leaves through x = 0
    assert r["boundary"]["right"]["u"]["inflow"][0] == pytest.approx(0.5, rel=1e-6)


def test_pde_oscillating_face_gives_sustained_oscillation_with_exact_amplitude():
    D, L, w, xp = 0.05, 1.0, 2 * math.pi, 0.2
    r = TE.solve_pde_1d(["u"], [0, L], 20.0, {"u": 0.0},
                        {"left": {"u": {"type": "dirichlet", "value": "sin(w*t)"}},
                         "right": {"u": {"type": "dirichlet", "value": 0.0}}}, {"u": D}, parameters={"w": w},
                        output_points=[xp], n_points=201)
    kappa = np.sqrt(1j * w / D)
    amplitude = abs(np.sinh(kappa * (L - xp)) / np.sinh(kappa * L))
    d = r["oscillation"][f"u@x={xp:g}"]
    assert d["kind"] == "sustained_oscillation"
    assert d["period"] == pytest.approx(2 * math.pi / w, rel=1e-4)
    assert d["amplitude"] == pytest.approx(amplitude, rel=2e-3)
    assert not r["steady_state"]["reached"]
    assert r["extrema"]["u"]["max"] == pytest.approx(1.0, abs=1e-4) and r["extrema"]["u"]["x_at_max"] == 0.0


def _packed_bed_exact(r, z, Da, beta, Peh, Bi, n=60):
    """T(r, z) and its cross-section average 2∫T r dr for the linear packed bed of the module docstring."""
    brackets = np.concatenate([[1e-9], jn_zeros(1, n)])        # one root of λJ1 − BiJ0 between consecutive zeros of J1
    T, avg = np.zeros_like(np.asarray(r, float)), 0.0
    for k in range(n):
        lam = brentq(lambda s: s * j1(s) - Bi * j0(s), brackets[k], brackets[k + 1])
        g = 2 * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
        mu = lam ** 2 / Peh
        a = beta * Da * g * (math.exp(-Da * z) - math.exp(-mu * z)) / (mu - Da)
        T += a * j0(lam * np.asarray(r, float))
        avg += a * 2 * j1(lam) / lam
    return T, avg


def test_pde_packed_bed_marching_with_radial_dispersion_two_variables():
    Da, beta, Peh, Pem, Bi = 1.2, 0.8, 6.0, 9.0, 2.5
    zs, pts = [0.5, 1.0, 2.0], [0.0, 0.5, 1.0]
    r = TE.solve_pde_1d(["c", "T"], [0, 1], 2.0, {"c": 1.0, "T": 0.0},
                        {"right": {"c": {"type": "symmetry"}, "T": {"type": "flux", "outward_gradient": "-Bi*T"}}},
                        {"c": "1/Pem", "T": "1/Peh"}, source={"c": "-rate", "T": "beta*rate"},
                        definitions={"rate": "Da*c"}, geometry="cylinder",
                        parameters={"Da": Da, "beta": beta, "Peh": Peh, "Pem": Pem, "Bi": Bi}, output_times=zs,
                        output_points=pts, space_variable="r", time_variable="z", n_points=201)
    assert r["success"] and r["evidence"]["verified"]
    for i, z in enumerate(zs):
        T, avg = _packed_bed_exact(np.array(pts), z, Da, beta, Peh, Bi)
        assert np.allclose(r["profiles"]["c"][i], math.exp(-Da * z), rtol=1e-5)
        assert np.allclose(r["profiles"]["T"][i], T, atol=3e-5)
        assert r["averages"]["T"][i] == pytest.approx(avg, abs=3e-5)
    hot = minimize_scalar(lambda z: -_packed_bed_exact(np.array([0.0]), z, Da, beta, Peh, Bi)[0][0], bounds=(0.1, 2),
                          method="bounded")
    assert r["extrema"]["T"]["max"] == pytest.approx(-hot.fun, abs=3e-5)          # hot spot on the axis
    assert r["extrema"]["T"]["r_at_max"] == 0.0
    assert r["extrema"]["T"]["z_at_max"] == pytest.approx(hot.x, abs=5e-3)


def test_pde_transient_dispersion_reactor_reaches_the_danckwerts_steady_state():
    Pe, Da = 8.0, 1.5
    r = TE.solve_pde_1d(["c"], [0, 1], 12.0, {"c": 0.0},
                        {"left": {"c": {"type": "flux", "inflow": "v*(1 - c)"}}, "right": {"c": "zero_flux"}},
                        {"c": "1/Pe"}, source={"c": "-Da*c"}, velocity={"c": "v"},
                        parameters={"Pe": Pe, "Da": Da, "v": 1.0}, output_times=[12.0], n_points=201)
    assert r["success"] and r["steady_state"]["reached"] and r["evidence"]["verified"]
    assert r["profiles"]["c"][0][-1] == pytest.approx(_wehner_wilhelm(Pe, Da), rel=2e-4)


def test_pde_graetz_problem_with_velocity_profile_as_time_coefficient():
    """Laminar tube, fixed wall temperature: (1 − r²)·∂T/∂z = (1/r)·∂/∂r(r·∂T/∂r); fully developed Nu = 3.657
    (Graetz-Nusselt; Incropera table 8.1; first eigenvalue λ1 = 2.7043644, Nu = λ1²/2 = 3.6568)."""
    r = TE.solve_pde_1d(["T"], [0, 1], 1.0, {"T": 1.0}, {"right": {"T": {"type": "dirichlet", "value": 0.0}}}, {"T": 1.0},
                        time_coefficient={"T": "1 - r^2"}, geometry="cylinder", space_variable="r", time_variable="z",
                        averages={"num": "(1 - r^2)*T", "den": "1 - r^2"}, output_times=[1.0], n_points=201)
    mixing_cup = r["averages"]["num"][0] / r["averages"]["den"][0]
    nusselt = -2 * r["boundary"]["right"]["T"]["inflow"][0] / mixing_cup       # Nu = h·D/k, D = 2R
    assert r["success"] and r["evidence"]["verified"]
    assert nusselt == pytest.approx(2.7043644 ** 2 / 2, rel=2e-4)


_IGNITION = {"T": "B*Da*c*exp(g*(1 - 1/T))", "c": "-Da*c*exp(g*(1 - 1/T))"}       # first order, Arrhenius, T/T0 and c/c0


def test_pde_ignition_is_reported_as_a_verified_steep_front_not_a_blow_up():
    """Adiabatic, uniform reacting slab (zero flux at both faces, uniform start): the PDE reduces to the batch-reactor
    ODEs T' = B·Da·c·e^{γ(1−1/T)}, c' = −Da·c·e^{γ(1−1/T)}. With γ = 40 the temperature sits near 1 and then jumps to the
    adiabatic value 1 + B = 1.6 in ~1e-5 of the interval. Reference: the same two ODEs integrated here with scipy
    Radau (a different integrator from the tool's BDF) at rtol = 1e-12, plus the exact invariant T + B·c = 1 + B."""
    from scipy.integrate import solve_ivp
    P = {"B": 0.6, "Da": 0.5, "g": 40.0}
    sym = {"type": "symmetry"}
    t_out = [0.05, 0.09, 0.0924, 0.093, 0.2, 0.5]
    r = TE.solve_pde_1d(["T", "c"], [0, 1], 0.5, {"T": 1.0, "c": 1.0},
                        {"left": {"T": sym, "c": sym}, "right": {"T": sym, "c": sym}}, {"T": 0.01, "c": 0.01},
                        source=_IGNITION, parameters=P, output_times=t_out, rtol=1e-9, atol=1e-12)

    def f(t, y):
        k = P["Da"] * y[1] * math.exp(P["g"] * (1 - 1 / y[0]))
        return [P["B"] * k, -k]
    ref = solve_ivp(f, (0, 0.5), [1.0, 1.0], method="Radau", rtol=1e-12, atol=1e-14, dense_output=True)
    assert r["success"] and r["ignition_or_steep_front"] is True
    sf = r["steep_front"]
    assert sf["bounded"] and sf["survives_refinement"] and r["evidence"]["verified"]
    assert sf["peak_scaled_rate"] > 1e3                       # the whole rise takes < 1/1000 of the interval
    i_peak = int(np.argmax(np.abs(np.gradient(ref.sol(np.linspace(0.092, 0.093, 200001))[0]))))
    assert sf["t_at_peak_rate"] == pytest.approx(0.092 + 1e-3 * i_peak / 200000, abs=3e-4)   # sampled every 2.5e-4
    assert r["extrema"]["T"]["max"] == pytest.approx(1 + P["B"], rel=1e-8)
    assert r["extrema"]["T"]["max"] == pytest.approx(float(np.max(ref.y[0])), rel=1e-8)
    # t = 0.0924 is in the middle of the jump, where T rises at ≈ 7e4 per unit time: an error of 1e-9 in the ignition
    # time (the tolerance asked for) moves T by ≈ 1e-4 there; everywhere else the agreement is ≈ 1e-7
    for m, t in enumerate(t_out):
        for k, v in enumerate(("T", "c")):
            err = float(np.max(np.abs(np.array(r["profiles"][v][m]) - ref.sol(t)[k])))
            assert err < (3e-4 if t == 0.0924 else 1e-6), (v, t, err)
    assert 1.2 < r["profiles"]["T"][2][0] < 1.5


def test_pde_ignition_with_fixed_surface_values_is_not_mistaken_for_a_blow_up():
    """Regression: catalyst sphere with fixed surface T and c and γ = 40. The integrator's finite-difference Jacobian
    perturbs the Dirichlet nodes ten times more at each evaluation (they do not affect the right-hand side), so after
    ~108 Jacobian evaluations, which an ignition needs, the raw state vector passed 1e100 and the run was reported as a
    finite-time blow-up at t ≈ 0.416. The solution is bounded: T + B·c obeys the pure diffusion equation (equal
    diffusivities) with surface value 1 + B and start 1, so by the maximum principle T ≤ 1 + B everywhere."""
    from packages.numerics import pde1d
    B = 0.6
    m = pde1d._Model(["T", "c"], [0, 1], "sphere", {"B": B, "Da": 3.0, "g": 40.0}, None, {"T": 1.0, "c": 1.0}, _IGNITION,
                     None, None, {"T": 1.0, "c": 0.0},
                     {"left": {}, "right": {"T": {"type": "dirichlet", "value": 1.0},
                                            "c": {"type": "dirichlet", "value": 1.0}}}, "x", "t", "central")
    g, sol, state, _ = pde1d._run(m, 101, 0.0, 1.0, "BDF", 1e-6, 1e-9, pde1d.Deadline())
    assert sol.status == 0 and not sol.blew_up and sol.njev > 100
    Y = state(np.linspace(0, 1, 401))
    assert np.max(Y[:, 0]) <= (1 + B) * (1 + 1e-5) and np.max(Y[:, 0]) > 1.59
    assert np.max(Y[:, 0] + B * Y[:, 1]) <= (1 + B) * (1 + 1e-5)


def test_pde_blow_up_is_reported_plainly():
    r = TE.solve_pde_1d(["u"], [0, 1], 5.0, {"u": 5.0}, {"left": {"u": "symmetry"}, "right": {"u": "symmetry"}},
                        {"u": 1.0}, source={"u": "u^2"})                 # uniform: u' = u², blows up at t = 0.2
    assert r["success"] is False and "profiles" not in r
    assert "crece sin límite" in r["message"] and r["ignition_or_steep_front"] is False
    assert r["blow_up"]["confirmed_by_refinement"]
    assert [run["t_failure"] for run in r["blow_up"]["runs"]] == pytest.approx([0.2, 0.2], abs=2e-5)


def test_pde_finite_time_blow_up_with_diffusion_is_still_a_blow_up():
    """u_t = u_xx + u², u = 0 at both faces, u(x, 0) = 20·sin(πx): the diffusion loss (π²·u) is smaller than the source
    from the start (u² > π²·u where u > 9.87), so the solution blows up in finite time (Kaplan's argument). Without
    diffusion the peak would blow up at t = 1/20 = 0.05, so the blow-up time must be later than that; both grids must
    fail at the same time and no numbers may be reported."""
    zero = {"type": "dirichlet", "value": 0.0}
    r = TE.solve_pde_1d(["u"], [0, 1], 1.0, {"u": "20*sin(pi*x)"}, {"left": {"u": zero}, "right": {"u": zero}},
                        {"u": 1.0}, source={"u": "u^2"})
    assert r["success"] is False and "profiles" not in r and "extrema" not in r
    assert "crece sin límite" in r["message"] and r["ignition_or_steep_front"] is False
    runs = r["blow_up"]["runs"]
    assert r["blow_up"]["confirmed_by_refinement"] and len(runs) == 2
    assert 0.05 < runs[0]["t_failure"] < 0.2 and runs[1]["t_failure"] == pytest.approx(runs[0]["t_failure"], rel=1e-3)
    assert min(run["max_abs_value_reached"] for run in runs) > 1e9


def test_pde_input_errors_are_clear():
    bc = {"left": {"u": "symmetry"}, "right": {"u": {"type": "dirichlet", "value": 0.0}}}
    with pytest.raises(ValueError, match="el nombre 'kk' no está declarado"):
        TE.solve_pde_1d(["u"], [0, 1], 1.0, {"u": 1.0}, bc, {"u": 1.0}, source={"u": "-kk*u"})
    with pytest.raises(ValueError, match="falta la condición de frontera de 'u' en la cara 'left'"):
        TE.solve_pde_1d(["u"], [0, 1], 1.0, {"u": 1.0}, {"right": bc["right"]}, {"u": 1.0})
    with pytest.raises(ValueError, match="solo cabe la simetría"):
        TE.solve_pde_1d(["u"], [0, 1], 1.0, {"u": 1.0},
                        {"left": {"u": {"type": "dirichlet", "value": 1.0}}, "right": bc["right"]}, {"u": 1.0},
                        geometry="sphere")
    with pytest.raises(ValueError, match="'inflow' .* o 'outward_gradient'"):
        TE.solve_pde_1d(["u"], [0, 1], 1.0, {"u": 1.0},
                        {"left": {"u": "symmetry"}, "right": {"u": {"type": "flux", "value": 1.0}}}, {"u": 1.0})
    with pytest.raises(ValueError, match="diffusivity: falta la expresión de \\['T'\\]"):
        TE.solve_pde_1d(["c", "T"], [0, 1], 1.0, {"c": 1.0, "T": 1.0},
                        {"left": "symmetry", "right": "symmetry"}, {"c": 1.0})


# ---------------------------------------------------------------------------------------------------------------------
# registration surface
# ---------------------------------------------------------------------------------------------------------------------
def test_tool_lists_and_wrappers_return_json():
    from packages.agent import wrappers_equations as WE
    assert [f.__name__ for f in TE.EQUATIONS_TOOLS] == ["solve_ode_system", "solve_bvp_system", "solve_pde_1d"]
    assert [w.name for w in WE.AGENT_WRAPPERS] == [f.__name__ for f in TE.EQUATIONS_TOOLS]
    for plain, wrapper in zip(TE.EQUATIONS_TOOLS, WE.AGENT_WRAPPERS):
        assert wrapper.to_dict()["description"].split()[:6] == plain.__doc__.split()[:6]
    out = json.loads(WE.solve_ode_system.call({"equations": {"c": "-c"}, "initial_conditions": {"c": 1.0}, "t_end": 1.0}))
    assert out["final"]["c"] == pytest.approx(math.exp(-1), rel=1e-7)
    out = json.loads(WE.solve_pde_1d.call({
        "variables": ["u"], "domain": [0, 1], "t_end": 0.1, "initial_conditions": {"u": 1},
        "boundary_conditions": {"left": {"u": {"type": "symmetry"}}, "right": {"u": {"type": "dirichlet", "value": 0}}},
        "diffusivity": {"u": 1}}))
    assert out["success"] and out["evidence"]["verified"] is not None
