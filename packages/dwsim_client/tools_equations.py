"""Agent tools that integrate equations the USER writes as text (bloque EQ): initial-value ODE systems, two-point boundary
value problems and 1-D parabolic PDE systems by the method of lines. No DWSIM flowsheet needed. Engines:
`packages.numerics.ode`, `packages.numerics.bvp`, `packages.numerics.pde1d`; expressions are compiled by the safe AST
whitelist of `packages.numerics.expressions` (never eval/exec).

The docstrings below are what the model reads (MCP server and agent wrappers share them): keep them precise and compact."""
from __future__ import annotations

from packages.numerics import bvp as BV
from packages.numerics import ode as OD
from packages.numerics import pde1d as PD


def solve_ode_system(equations: dict[str, str | float], initial_conditions: dict[str, float], t_end: float,
                     t_start: float = 0.0, parameters: dict[str, float] | None = None,
                     definitions: dict[str, str | float] | None = None, output_times: list[float] | None = None,
                     n_output: int = 11, events: list[dict] | None = None, method: str = "LSODA", rtol: float = 1e-8,
                     atol: float = 1e-10, independent_variable: str = "t", oscillation_window_fraction: float = 0.5,
                     max_step: float | None = None) -> dict:
    """Integrate a user-written initial-value problem dy_i/dt = f_i(t, y, params) (stiff-safe). Use it for any ODE model
    the user states (reactors, kinetics, population/oscillator models, batch balances) instead of doing arithmetic by hand.

    equations: {variable: expression of d(variable)/dt}; initial_conditions: {variable: value at t_start}.
    Expressions (text): numbers, declared names, + - * / ** (or ^), comparisons < <= > >= (worth 1 or 0), pi, and
      exp log log10 sqrt sin cos tan sinh cosh tanh asin acos atan abs sign erf erfc floor heaviside(x) min(a,b,..)
      max(a,b,..) atan2(y,x) pow(a,b) where(cond,a,b). Names: the variables, `parameters` {name: number}, the independent
      variable (default 't'), and `definitions` {name: expression}: intermediate quantities (rates, Arrhenius constants)
      evaluated in the given order, each may use the previous ones; they are also tabulated in the outputs.
    output_times: times to report (default n_output evenly spaced). events: [{expression, name?, terminal?, direction?}]
      — the event occurs when the expression crosses zero (direction 1: rising, -1: falling, 0: both); terminal=True
      stops the integration there (e.g. 'T - 400' to find when T reaches 400).
    method: LSODA (default, automatic stiffness switching), Radau, BDF (very stiff), RK45, DOP853. rtol/atol: tolerances.

    Returns success + message (says plainly if the solution blows up or hits a singularity before t_end, and where),
    outputs {t: [...], variable: [...], definition: [...]}, final, extrema {variable: {min, t_at_min, max, t_at_max}},
    events [{name, n_crossings, times, states}], steady_state {reached, max_scaled_rate}, oscillation {variable: {kind:
    sustained_oscillation | decaying_oscillation | growing_oscillation | irregular_oscillation | steady | not_settled,
    period, amplitude (half peak-to-trough), amplitude_ratio_per_cycle, mean, n_cycles}} over the last
    oscillation_window_fraction of the run, evidence (same run with tolerances 100x tighter: max_rel_diff, verified),
    warnings. A limit cycle needs t_end long enough for the transient to die (several periods inside the window).
    Not for: DAEs, delays."""
    return OD.solve_ode_system(equations, initial_conditions, t_end, t_start, parameters, definitions, output_times,
                               n_output, events, method, rtol, atol, independent_variable, oscillation_window_fraction,
                               max_step)


def solve_bvp_system(equations: dict[str, str | float], domain: list[float], boundary_conditions: list[str],
                     parameters: dict[str, float] | None = None, unknown_parameters: dict[str, float] | None = None,
                     initial_guess: dict | None = None, definitions: dict[str, str | float] | None = None,
                     singular_terms: dict[str, str] | None = None, output_points: list[float] | None = None,
                     n_output: int = 11, integrals: dict[str, str] | None = None, integral_weight_power: float = 0.0,
                     continuation: dict | None = None, tol: float = 1e-6, n_mesh: int = 41, max_nodes: int = 20000,
                     independent_variable: str = "x") -> dict:
    """Solve a user-written two-point boundary value problem: first-order system dy_i/dx = f_i(x, y, params) on
    domain [a, b]. Rewrite higher order as first order (y'' = g → equations {'y': 'z', 'z': 'g'}). Typical uses:
    axial-dispersion reactor with Danckwerts conditions, reaction-diffusion in a catalyst pellet (effectiveness factor),
    boundary layers (Blasius) on a truncated domain, fins, eigenvalue problems.

    boundary_conditions: list of RESIDUAL expressions that must equal 0, exactly (number of variables + number of
      unknown_parameters) of them; the value of variable y at x=a is `y_a`, at x=b `y_b`. Examples: y(a)=1 → 'y_a - 1';
      Danckwerts inlet c − c'/Pe = 1 with z=c' → 'c_a - z_a/Pe - 1'; zero gradient at the outlet → 'z_b'.
    Expressions: same syntax and functions as solve_ode_system; names = variables, x (independent_variable), parameters,
      unknown_parameters {name: guess} (solved for; one extra condition each), definitions {name: expression}.
    singular_terms: for a term (k/x)·z singular at x=a (symmetry centre of a cylinder k=1 or sphere k=2): do NOT write
      it in equations; give its numerator, linear in the variables: z' = phi^2*c − (2/x)*z → equations {'c': 'z',
      'z': 'phi^2*c'}, singular_terms {'z': '-2*z'}, and still list the symmetry condition 'z_a'.
    initial_guess: {variable: number | expression of x | {'x': [...], 'values': [...]}} (default 0). Nonlinear problems
      with several solutions converge to the one nearest the guess: change the guess to reach the other branch.
    continuation: {'parameter': name in parameters, 'values': [easy, ..., target]} solves at each value starting from the
      previous solution (auto-bisection); the last value is the reported problem.
    output_points (default n_output evenly spaced); integrals {name: expression} → ∫ x^w · expr dx and its average
      (divided by ∫ x^w dx) with w = integral_weight_power (0 slab, 1 cylinder, 2 sphere), e.g. effectiveness factor =
      average of rate/rate_surface. Semi-infinite domain: truncate at a large b and check the result does not change.

    Returns converged (False + message and NO numbers when it did not converge), points {x, variable, d<variable>_dx,
    definitions}, boundary_values {y_a, y_b}, unknown_parameters, integrals {name: {integral, average}}, continuation
    steps, evidence {max_rms_residual of scipy solve_bvp, max_boundary_residual, re-solve with tol/100: max_rel_diff,
    verified}, warnings."""
    return BV.solve_bvp_system(equations, domain, boundary_conditions, parameters, unknown_parameters, initial_guess,
                               definitions, singular_terms, output_points, n_output, integrals, integral_weight_power,
                               continuation, tol, n_mesh, max_nodes, independent_variable)


def solve_pde_1d(variables: list[str], domain: list[float], t_end: float, initial_conditions: dict[str, str | float],
                 boundary_conditions: dict[str, dict], diffusivity: dict[str, str | float],
                 source: dict[str, str | float] | None = None, velocity: dict[str, str | float] | None = None,
                 time_coefficient: dict[str, str | float] | None = None, geometry: str = "slab",
                 parameters: dict[str, float] | None = None, definitions: dict[str, str | float] | None = None,
                 t_start: float = 0.0, output_times: list[float] | None = None, n_output_times: int = 6,
                 output_points: list[float] | None = None, averages: dict[str, str] | None = None, n_points: int = 101,
                 rtol: float = 1e-6, atol: float = 1e-9, method: str = "BDF", convection_scheme: str = "central",
                 space_variable: str = "x", time_variable: str = "t", oscillation_window_fraction: float = 0.5) -> dict:
    """Solve user-written parabolic PDEs in one space coordinate (method of lines, stiff BDF), one equation per variable:
        c·∂u/∂t = (1/x^s)·∂/∂x(x^s·D·∂u/∂x) − v·∂u/∂x + R
    geometry 'slab' (s=0), 'cylinder' (s=1, x=radius) or 'sphere' (s=2). Per variable: diffusivity D (required), source R,
    velocity v, time_coefficient c (default 0, 0, 1) as numbers or expressions of x, t, the variables, parameters and
    definitions (same syntax as solve_ode_system). Uses: transient conduction/diffusion with reaction, pellets,
    dispersion with convection. "t" can be an AXIAL coordinate (steady packed bed or tube with radial dispersion, coupled
    c and T: divide by the axial velocity or put it in time_coefficient); rename with time_variable / space_variable.

    initial_conditions: {variable: number or expression of x}. domain: [x_min, x_max].
    boundary_conditions: {'left': {variable: cond}, 'right': {variable: cond}} (left = x_min, right = x_max); cond is
      {'type': 'dirichlet', 'value': expr} (may depend on t; the two faces may differ);
      {'type': 'flux', 'inflow': expr}: D·∂u/∂n = inflow with n the OUTWARD normal, i.e. what ENTERS the domain per unit
        area. Constant = Neumann; expression of the surface value = Robin: wall convection 'h*(Tinf - T)' (units of
        D·∂u/∂x: use h/(rho*cp) if D is thermal diffusivity), Danckwerts inlet 'v*(c_in - c)'. Or give 'outward_gradient'
        (∂u/∂n) instead of 'inflow', e.g. dimensionless Biot '-Bi*T';
      {'type': 'symmetry'} = zero flux / insulated. At x=0 of a cylinder or sphere symmetry is automatic (omit it).
      Inside boundary expressions the variable names are the values at that face.
    output_times (default n_output_times evenly spaced), output_points (x positions, default 11 evenly spaced),
    averages {name: expression} (default each variable): section average ∫e·x^s dx/∫x^s dx at every output time
    (cylinder: 2∫e·r dr/R²). n_points grid nodes. convection_scheme 'central' or 'upwind'.

    Returns success + message, profiles {t, x, variable[i_t][j_x]},
    averages {t, name: [...]}, extrema {variable: {max, x_at_max, t_at_max, min, ...}} over all x and t (hot spot),
    boundary {face: {variable: {value, inflow}}} at the output times, steady_state {reached, settling_time},
    oscillation {signal: {kind, period, amplitude, ...}} for each average and each user output point (kinds as in
    solve_ode_system), evidence (re-run with twice the grid intervals and 10x tighter tolerances: max_rel_diff,
    verified), warnings.
    Automatic refinement: if that check fails, the grid is doubled (up to 3 more times, while time allows) and the
    results REPORTED are those of the finest verified pair: read 'n_points' (grid actually used) and
    evidence.automatic_refinement {n_points_requested, n_points_used, levels}. If it still fails,
    evidence.recommended_n_points is the grid estimated to pass.
    Ignition vs blow-up: ignition_or_steep_front = true when a variable jumps at an isolated, self-accelerating rate
    (Arrhenius source with reactant exhaustion, travelling front); the solution is bounded and IS reported, with
    steep_front {variable, t_at_peak_rate, peak_scaled_rate, extrema, extrema_refined, survives_refinement}: trust it
    when survives_refinement is true (profiles at fixed times near the jump converge slower than the extrema). A run
    the integrator cannot finish is repeated on the finer grid; success = false with blow_up {confirmed_by_refinement,
    runs: failure time and max |u| of each} and NO numbers only when it fails again (unbounded growth, e.g. u_t = u_xx +
    u^2, or a singularity). Not for pure convection (hyperbolic) fronts or more than one space dimension."""
    return PD.solve_pde_1d(variables, domain, t_end, initial_conditions, boundary_conditions, diffusivity, source,
                           velocity, time_coefficient, geometry, parameters, definitions, t_start, output_times,
                           n_output_times, output_points, averages, n_points, rtol, atol, method, convection_scheme,
                           space_variable, time_variable, oscillation_window_fraction)


def solve_algebraic_system(equations: list[str] | None = None, unknowns: dict | None = None,
                           parameters: dict | None = None, definitions: dict | None = None,
                           outputs: dict | None = None, tolerance: float = 1e-9) -> dict:
    """Sistema de ecuaciones ALGEBRAICAS no lineales escrito por el usuario, o evaluación directa de fórmulas (calculadora
    con evidencia). Úsala para aritmética de balances, ecuaciones implícitas (Antoine inversa, factor de fricción,
    raíces de un polinomio, n ecuaciones con n incógnitas) en vez de calcular a mano o de forzar solve_ode_system.

    equations: lista de textos "lhs = rhs" (o una expresión que debe valer 0), tantas como incógnitas.
    unknowns: {nombre: estimación inicial} o {nombre: {"guess": x o [x1, x2...], "lower": a, "upper": b}}; con varios
      guess o con límites se prueban varios arranques y se reportan TODAS las raíces distintas halladas.
    parameters: {nombre: número}. definitions: {nombre: expresión} intermedias, en orden (pueden usar incógnitas).
    outputs: {nombre: expresión} a evaluar en la solución. Sin equations ni unknowns evalúa definitions y outputs.
    Expresiones: + − * / ^, exp, log, log10, sqrt, sin, cos, tan, abs, min, max, where(cond, a, b), pi. Sin unidades:
    el usuario cuida la coherencia.

    Devuelve solution {unknowns, residuals, definitions, outputs, at_bound}, other_solutions si hay más de una raíz,
    warnings y evidence. converged=false: no hay raíz con esos arranques o límites; no reportes números de ese intento."""
    from packages.numerics import algebraic as AL
    try:
        return AL.solve(equations, unknowns, parameters, definitions, outputs, tolerance)
    except ValueError as ex:
        return {"ok": False, "error": str(ex)}


EQUATIONS_TOOLS = [solve_ode_system, solve_bvp_system, solve_pde_1d, solve_algebraic_system]
