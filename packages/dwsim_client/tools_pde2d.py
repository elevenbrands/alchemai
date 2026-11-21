"""Agent tool for steady 2-D scalar field problems written by the user (Laplace/Poisson, diffusion-reaction,
convection-diffusion with a given velocity field). No DWSIM flowsheet needed. Engine: `packages.transport.pde2d`.
This module is imported by `packages.dwsim_client.tools`; it must not import it back."""
from __future__ import annotations

from packages.transport import pde2d as PDE


def solve_pde_2d(coordinates: str, domain: dict[str, list[float]], equations: dict[str, dict[str, object]],
                 boundary_conditions: dict[str, object], velocity: dict[str, str] | None = None,
                 parameters: dict[str, float] | None = None, outputs: dict[str, object] | None = None,
                 grid: dict[str, object] | None = None, initial_guess: dict[str, object] | None = None,
                 convection_scheme: str = "auto", refinement_tol: float = 0.01, max_iterations: int = 60) -> dict:
    """Steady 2-D field solver (finite volumes) for one or several coupled variables u_i:
        div(D_i grad u_i) - v . grad u_i + R_i(coords, u, params) = 0
    Use it for Laplace/Poisson, diffusion with (nonlinear) reaction in a pellet, conduction, and convection-diffusion
    with a KNOWN velocity field (Graetz, flow past a sphere). Work in consistent (preferably dimensionless) units.

    coordinates / domain / side names ('<coord>_min', '<coord>_max'):
      'cartesian'  {'x': [x0, x1], 'y': [y0, y1]}          sides x_min x_max y_min y_max (per unit depth)
      'axisymmetric'  {'r': [r0 >= 0, r1], 'z': [z0, z1]}  sides r_min (axis if r0 = 0) r_max z_min z_max
      'polar'  {'r': [r_in >= 0, r_out], 'theta': [t0, t1]}  sector; sides r_min r_max theta_min theta_max
      'spherical_axisymmetric'  {'r': [R >= 0, R_out]} (theta 0..pi from the +z axis, or give 'theta')  sides
        r_min (sphere surface) r_max theta_min theta_max.
      Sides that collapse (r = 0, the poles) need no condition.
    Expressions are text (or numbers) of the coordinate names, parameters, pi, + - * / ** and exp log log10 sqrt sin cos
      tan sinh cosh tanh asin acos atan abs erf erfc step(x) min(a,b) max(a,b) atan2 pow. No comparisons: piecewise
      data go in boundary segments.
    equations: {var: {'D': number | expr(coords) | {'r': .., 'z': ..} (per direction; 0 allowed in one, e.g. no axial
      conduction), 'source': expr(coords, all variables, params) = R_i, 'velocity_factor': optional multiplier of v}}.
    velocity: {coord: expr(coords)} physical components (e.g. {'z': '2*(1-r^2)'}; Stokes flow past a sphere, stream
      along +z: {'r': 'cos(theta)*(1-1.5/r+0.5/r^3)', 'theta': '-sin(theta)*(1-0.75/r-0.25/r^3)'}).
    boundary_conditions: {side: bc | [segments] | {var: bc | [segments]}}; n = outward normal. bc:
      {'type': 'dirichlet', 'value': expr}; {'type': 'neumann', 'value': du/dn (default 0)}; {'type': 'symmetry'};
      {'type': 'robin', 'h': expr, 'u_ext': expr} meaning -D du/dn = h (u - u_ext);
      {'type': 'flux', 'value': expr(coords, variables at the wall, params)} meaning -D du/dn = value (outward
      diffusive flux, may be nonlinear: surface reaction).
      Segments: the same dicts with 'from'/'to' along the side coordinate (e.g. a wall temperature changing along z).
    outputs (all optional):
      'points': [[c1, c2], ..] values; 'derivative_points': [[c1, c2], ..] du/dc1 and du/dc2 (coordinate derivatives);
      'averages': [{'name', 'expression'}] volume average (true geometric weight) and integral of an expression of
        coords/variables, e.g. effectiveness factor = average of 'rate(c)/rate_surface';
      'mixing_cup': [{'name', 'expression', 'coordinate': 'z', 'at': value or list}] velocity-weighted (cup-mixing) and
        area average over the section coordinate = at;
      'side_fluxes': [{'name', 'side', 'variable', 'from', 'to'}] area-averaged and integrated OUTWARD diffusive flux
        -D du/dn, mean du/dn and mean u over the side or a stretch; or with 'at': position, the local values;
      'derived': {name: expr} combines, in ONE expression, params, the named outputs above (<average>,
        <average>_integral, <mixing_cup> (scalar 'at'), <side_flux> (mean or local flux out), <side_flux>_integral,
        <side_flux>_gradient (du/dn), <side_flux>_mean) and these measurements written inline (side and coordinate
        names without quotes; positions are numbers or expressions of params):
          value(expr, c1, c2)             expression of coords/variables at the point (c1, c2)
          wall_gradient(var, side, s)     OUTWARD normal derivative du/dn on the side, at position s along it
          wall_flux(var, side, s)         -D du/dn there;   wall_value(var, side, s)  boundary value there
          area_average(expr, coord, pos)  area average over the section coord = pos
          mixing_cup(expr, coord, pos)    velocity-weighted (cup-mixing) average over that section
        Profile along a coordinate: {name: {'expression': expr, 'along': 'z', 'at': [z1, z2, ..]}}; inside expr the name
        z is each position. Local Nusselt number of a tube (r in [0,1], wall r_max, diameter 2), in one expression:
          'Nu': {'expression': '2*wall_gradient(T, r_max, z)/(wall_value(T, r_max, z) - mixing_cup(T, z, z))',
                 'along': 'z', 'at': [0.1, 0.5, 1.0]}
        i.e. Nu = (dT/dn)|wall * D/(T_wall - T_mix) > 0, dT/dn = +dT/dr at r_max. Returned as derived[name] = {expression,
        along, values: [{z, value, relative_change_on_refinement, grid_converged}]}. Sphere at T = 1 (r_min):
        Nu = '-2*q' with q a side_flux.
    grid: {'n': [n1, n2] base cells (default 64 x 64), 'grading': {coord: 'uniform' | 'log' | {'cluster': 'min' | 'max'
      | 'both', 'strength': 1..6} | {'cluster': 'at', 'position': x0, 'strength': 1..6} (cells shrink towards x0:
      use it where a wall condition jumps, local fluxes near a jump do not converge on a uniform grid; a warning
      names the jump and this option when that happens)}}. Defaults: polar from r = 0 is graded towards r = 0; a
      spherical shell with R_out/R > 3 is 'log' in r. initial_guess: {var: expr} for nonlinear problems.
    convection_scheme: 'auto' (central if cell Peclet <= 2, else van Leer TVD), 'central', 'upwind', 'tvd'.

    Returns points, derivative_points, averages, mixing_cup, side_fluxes, derived, extrema (min/max and where); every
    number is {value, relative_change_on_refinement, grid_converged, note}. The problem is solved on n/2, n and 2n;
    values are from 2n. grid_converged / not_grid_converged list what still changes more than refinement_tol (e.g. a
    gradient at a singular corner, which is unbounded): do NOT report those as results. evidence: grid_refinement
    (three grids, observed order, Richardson value when regular), nonlinear (iterations, residual), global_balance
    (diffusive flux out by side, convection term, source integral, imbalance), convection (scheme used, cell Peclet).
    Integrals are per unit depth (cartesian, polar) or over the full turn (axisymmetric, spherical). Limits: steady,
    structured domain in the chosen coordinates, D independent of u, no periodic theta, a truncated outer radius
    stands for 'infinity' (vary R_out to see its effect)."""
    return PDE.solve_pde_2d(coordinates, domain, equations, boundary_conditions, velocity, parameters, outputs, grid,
                            initial_guess, convection_scheme, refinement_tol, max_iterations)


PDE2D_TOOLS = [solve_pde_2d]
