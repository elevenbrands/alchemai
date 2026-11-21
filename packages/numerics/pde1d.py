"""Systems of parabolic equations in one space coordinate by the method of lines, written by the user as text expressions.

Equation for every variable u_i(x, t):

    c_i · ∂u_i/∂t = (1/x^s) · ∂/∂x( x^s · D_i · ∂u_i/∂x )  −  v_i · ∂u_i/∂x  +  R_i

with s = 0 (slab), 1 (cylinder, x = radius) or 2 (sphere). c_i (default 1), D_i, v_i (default 0) and R_i (default 0) are
numbers or expressions of x, t, the variables, the parameters and the `definitions`. For constant D_i this is
∂u/∂t = (D/x^s) ∂/∂x(x^s ∂u/∂x) − v ∂u/∂x + R. "t" is whatever coordinate the problem marches along: time, or the axial
position of a steady packed bed / tube with radial dispersion (divide the equation by the axial velocity, or put a
velocity profile in c_i).

Boundary conditions, per face ('left' = x_min, 'right' = x_max) and per variable:
  * {'type': 'dirichlet', 'value': expr}                     u = value (may depend on t);
  * {'type': 'flux', 'inflow': expr}                         D·∂u/∂n = inflow, n = OUTWARD normal, so `inflow` is the
        amount entering the domain per unit area, in the units of D·u/x. Constant → Neumann; an expression of the surface
        value → Robin: convection (Biot) at a wall is `h*(u_inf - u)` (same units as D·∂u/∂x: divide h by ρ·cp if D is a
        thermal diffusivity), a Danckwerts inlet is `v*(u_in - u)`. `outward_gradient` may be given instead of `inflow`
        (∂u/∂n = expr; dimensionless Biot: `-Bi*u`);
  * {'type': 'symmetry'} (= zero flux, insulated, impermeable). At x = 0 of a cylinder or sphere it is the only possible
        condition and is applied automatically.
  In boundary expressions the variable names mean the values AT THAT FACE.

Discretisation: uniform vertex grid of n_points nodes, conservative finite volumes (face areas x^s and exact shell volumes,
so x = 0 needs no special formula and averages are conserved to round-off), second order in space; convection by central
differences (second order; warns when the cell Péclet number exceeds 2) or first-order upwind on request. Time integration
by scipy BDF (or Radau) with the block-tridiagonal sparsity pattern of the Jacobian.

Evidence of correctness: the problem is solved again with twice the grid intervals and tolerances 10 times tighter and the
largest difference on every reported number is returned. While that check fails and there is time, the fine run becomes
the reported one and is checked against a finer one (automatic refinement, at most MAX_AUTO_REFINEMENTS doublings); if it
still fails, the grid that would pass is estimated from the observed order.

Ignition versus blow-up: a reaction with an Arrhenius source can raise a variable by orders of magnitude in rate and then
saturate (reactant exhausted). That is a bounded, physical solution and it is reported, flagged
`ignition_or_steep_front` with the refinement evidence (extrema of the two grids, time of the peak rate). A run the
integrator cannot finish is repeated on the finer grid with tighter tolerances: only if that fails too (same place,
values growing without bound) is it reported as a blow-up, with both failure times."""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from packages.numerics import diagnostics as DG
from packages.numerics import expressions as EX
from packages.numerics.common import (EVIDENCE_TOL, MAX_SECONDS, TIMEOUT_TEXT, Deadline, Timeout, full, number, number_list,
                                      rel_diff)

GEOMETRY = {"slab": 0, "cylinder": 1, "sphere": 2}
METHODS = ("BDF", "Radau")
MAX_VARIABLES = 12
MAX_POINTS = 2001
MAX_OUTPUT_TIMES = 200
MAX_OUTPUT_POINTS = 200
MAX_DIAGNOSED_POINTS = 6
N_SAMPLE = 2001
BLOWUP = 1e100
GROWTH = 1e6            # see _run: growth factor that classifies a failed integration as unbounded
STEADY_TOL = 1e-4       # |∂u/∂t|·(t_end − t_start)/scale below this at every node → steady state reached
SETTLE_BAND = 1e-3      # settling time: last time a monitored signal is farther than this fraction of its total change
STEEP_RATE = 50.0       # peak |∂u/∂t|·(interval)/(range of u) above this → ignition / steep front (see _steep_front)
STEEP_FRACTION = 0.05
MAX_AUTO_REFINEMENTS = 3   # extra grid doublings tried automatically when the refinement check does not pass
AUTO_REFINE_BUDGET = 0.5   # ... while the projected time stays below this fraction of the wall-clock budget
RTOL_FLOOR = 1e-11
_FLUX_TYPES = ("flux", "robin", "neumann")
_ZERO_TYPES = ("symmetry", "zero_flux", "insulated")


class _Blowup(Exception):
    pass


class _Model:
    """Compiled problem; `rhs` is the semi-discrete system on a given grid."""

    def __init__(self, variables, domain, geometry, parameters, definitions, diffusivity, source, velocity,
                 time_coefficient, initial_conditions, boundary_conditions, xname, tname, scheme):
        if not isinstance(variables, (list, tuple)) or not variables:
            raise ValueError("falta variables: lista de nombres de las variables dependientes, p. ej. ['T'] o ['c', 'T']")
        self.vars = [str(v) for v in variables]
        if len(self.vars) > MAX_VARIABLES:
            raise ValueError(f"demasiadas variables (máximo {MAX_VARIABLES})")
        if geometry not in GEOMETRY:
            raise ValueError("geometry debe ser 'slab' (placa), 'cylinder' (cilindro, x = radio) o 'sphere' (esfera)")
        self.s = GEOMETRY[geometry]
        try:
            self.a, self.b = (number(v, "domain") for v in domain)
        except (TypeError, ValueError):
            raise ValueError("domain debe ser [x_min, x_max] con x_min < x_max") from None
        if not self.a < self.b:
            raise ValueError("domain debe ser [x_min, x_max] con x_min < x_max")
        if self.s and self.a < 0:
            raise ValueError("en cilindro y esfera la coordenada es el radio: x_min debe ser ≥ 0")
        self.xname, self.tname, self.scheme = xname, tname, scheme
        self.params = EX.parse_parameters(parameters)
        base = EX.declare([("coordenadas", [xname, tname]), ("variables", self.vars), ("parameters", list(self.params))])
        self.defs, self.names = EX.compile_definitions(definitions, base)
        nv = len(self.vars)

        def per_variable(arg, label, default, required=False):
            if arg is None and not required:
                arg = {}
            if isinstance(arg, (int, float, str)) and not isinstance(arg, bool):
                arg = dict.fromkeys(self.vars, arg)
            if not isinstance(arg, dict) or set(arg) - set(self.vars):
                raise ValueError(f"{label} debe ser un número/expresión común o {{variable: expresión}} con variables de "
                                 f"{self.vars}")
            if required and set(arg) != set(self.vars):
                raise ValueError(f"{label}: falta la expresión de {sorted(set(self.vars) - set(arg))}")
            return [EX.compile_expression(arg.get(v, default), self.names, f"{label}['{v}']") for v in self.vars]

        self.D = per_variable(diffusivity, "diffusivity", 0.0, required=True)
        self.R = per_variable(source, "source", 0.0)
        self.v = per_variable(velocity, "velocity", 0.0)
        self.c = per_variable(time_coefficient, "time_coefficient", 1.0)
        self.has_v = velocity is not None
        ic_names = {xname} | set(self.params)
        if isinstance(initial_conditions, (int, float, str)) and not isinstance(initial_conditions, bool):
            initial_conditions = dict.fromkeys(self.vars, initial_conditions)
        if not isinstance(initial_conditions, dict) or set(initial_conditions) != set(self.vars):
            raise ValueError(f"initial_conditions debe dar, para cada variable de {self.vars}, un número o una expresión "
                             f"de {xname}")
        self.ic = [EX.compile_expression(initial_conditions[v], ic_names, f"initial_conditions['{v}']") for v in self.vars]

        # boundary conditions → per face, per variable: ('dirichlet', fn) | ('flux', fn) | ('gradient', fn) | ('zero',)
        if not isinstance(boundary_conditions, dict) or set(boundary_conditions) - {"left", "right"}:
            raise ValueError("boundary_conditions debe ser {'left': {variable: condición}, 'right': {variable: condición}}; "
                             f"'left' es la cara {xname} = x_min y 'right' la cara {xname} = x_max")
        self.bc = {}
        for face, xf in (("left", self.a), ("right", self.b)):
            given = boundary_conditions.get(face) or {}
            if isinstance(given, str):
                given = dict.fromkeys(self.vars, given)
            if not isinstance(given, dict) or set(given) - set(self.vars):
                raise ValueError(f"boundary_conditions['{face}'] debe ser {{variable: condición}} con variables de {self.vars}")
            axis = self.s > 0 and xf == 0.0
            specs = []
            for v in self.vars:
                where = f"boundary_conditions['{face}']['{v}']"
                spec = given.get(v)
                if isinstance(spec, str):
                    spec = {"type": spec}
                if spec is None:
                    if not axis:
                        raise ValueError(f"falta la condición de frontera de '{v}' en la cara '{face}' ({xname} = {xf:g}): "
                                         "{'type': 'dirichlet', 'value': ...}, {'type': 'flux', 'inflow': ...} o "
                                         "{'type': 'symmetry'}")
                    spec = {"type": "symmetry"}
                kind = str(spec.get("type", "")).lower() if isinstance(spec, dict) else ""
                if axis and kind not in _ZERO_TYPES:
                    raise ValueError(f"{where}: en {xname} = 0 de un cilindro o esfera solo cabe la simetría (el área de la "
                                     "cara es cero); quitar esa condición o usar {'type': 'symmetry'}")
                if kind == "dirichlet":
                    if set(spec) != {"type", "value"}:
                        raise ValueError(f"{where}: una condición dirichlet lleva exactamente 'value'")
                    specs.append(("dirichlet", EX.compile_expression(spec["value"], self.names, where)))
                elif kind in _FLUX_TYPES:
                    keys = set(spec) - {"type"}
                    if keys == {"inflow"}:
                        specs.append(("flux", EX.compile_expression(spec["inflow"], self.names, where)))
                    elif keys == {"outward_gradient"}:
                        specs.append(("gradient", EX.compile_expression(spec["outward_gradient"], self.names, where)))
                    else:
                        raise ValueError(f"{where}: una condición de flujo lleva 'inflow' (D·∂u/∂n = inflow, n normal "
                                         "exterior: lo que ENTRA al dominio) o 'outward_gradient' (∂u/∂n), solo uno")
                elif kind in _ZERO_TYPES:
                    specs.append(("zero",))
                else:
                    raise ValueError(f"{where}: type debe ser 'dirichlet', 'flux' (Neumann/Robin) o 'symmetry'")
            self.bc[face] = specs
        self.nv = nv

    # ------------------------------------------------------------------------------------------------------------
    def grid(self, n):
        x = np.linspace(self.a, self.b, n)
        faces = 0.5 * (x[:-1] + x[1:])
        edges = np.concatenate([[x[0]], faces, [x[-1]]])
        p = self.s + 1
        return {"x": x, "h": x[1] - x[0], "A": faces ** self.s, "V": (edges[1:] ** p - edges[:-1] ** p) / p,
                "A_left": x[0] ** self.s, "A_right": x[-1] ** self.s, "n": n}

    def env(self, g, t, U):
        env = dict(self.params)
        env[self.xname], env[self.tname] = g["x"], t
        env.update(zip(self.vars, U))
        return EX.apply_definitions(self.defs, env)

    def face_env(self, g, t, U, j):
        env = dict(self.params)
        env[self.xname], env[self.tname] = float(g["x"][j]), t
        env.update(zip(self.vars, (float(u) for u in U[:, j])))
        return EX.apply_definitions(self.defs, env)

    def impose(self, g, t, U):
        """Overwrite the Dirichlet nodes with their boundary values (in place)."""
        for face, j in (("left", 0), ("right", -1)):
            if any(s[0] == "dirichlet" for s in self.bc[face]):
                env = self.face_env(g, t, U, j)
                for k, spec in enumerate(self.bc[face]):
                    if spec[0] == "dirichlet":
                        U[k, j] = float(spec[1](env))
        return U

    def initial(self, g):
        env = {**self.params, self.xname: g["x"]}
        U = np.array([full(f(env), g["x"]) for f in self.ic], float)
        if not np.all(np.isfinite(U)):
            raise ValueError("initial_conditions no da valores finitos en toda la malla")
        return U

    def coefficients(self, g, t, U):
        env = self.env(g, t, U)
        x = g["x"]
        return (np.array([full(f(env), x) for f in self.D]), np.array([full(f(env), x) for f in self.R]),
                np.array([full(f(env), x) for f in self.v]), np.array([full(f(env), x) for f in self.c]))

    def inflows(self, g, t, U, D):
        """D·∂u/∂n at both faces for the non-Dirichlet variables (nan where Dirichlet)."""
        out = np.full((2, self.nv), np.nan)
        for m, (face, j) in enumerate((("left", 0), ("right", -1))):
            env = None
            for k, spec in enumerate(self.bc[face]):
                if spec[0] == "zero":
                    out[m, k] = 0.0
                elif spec[0] != "dirichlet":
                    env = env or self.face_env(g, t, U, j)
                    val = float(spec[1](env))
                    out[m, k] = val if spec[0] == "flux" else D[k, j] * val
        return out

    def rhs(self, g, t, U):
        """dU/dt on the grid (U of shape (nv, n), already with the Dirichlet values imposed)."""
        h = g["h"]
        D, R, v, c = self.coefficients(g, t, U)
        dU = np.diff(U, axis=1) / h
        F = g["A"] * 0.5 * (D[:, :-1] + D[:, 1:]) * dU                      # A·D·∂u/∂x at the interior faces
        q = self.inflows(g, t, U, D)
        ql, qr = np.nan_to_num(q[0]), np.nan_to_num(q[1])
        Fall = np.concatenate([(-g["A_left"] * ql)[:, None], F, (g["A_right"] * qr)[:, None]], axis=1)
        out = np.diff(Fall, axis=1) / g["V"] + R
        if self.has_v:
            grad = np.empty_like(U)
            if self.scheme == "upwind":
                back = np.concatenate([dU[:, :1], dU], axis=1)
                fwd = np.concatenate([dU, dU[:, -1:]], axis=1)
                grad = np.where(v >= 0, back, fwd)
            else:
                grad[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * h)
            with np.errstate(all="ignore"):                               # boundary nodes: gradient from the condition
                gl, gr = -ql / D[:, 0], qr / D[:, -1]
            grad[:, 0] = np.where(D[:, 0] > 0, gl, dU[:, 0])
            grad[:, -1] = np.where(D[:, -1] > 0, gr, dU[:, -1])
            out = out - v * grad
        with np.errstate(all="ignore"):      # c may vanish only at a Dirichlet node, overwritten just below
            out = out / c
        for k in range(self.nv):
            if self.bc["left"][k][0] == "dirichlet":
                out[k, 0] = 0.0
            if self.bc["right"][k][0] == "dirichlet":
                out[k, -1] = 0.0
        return out

    def sparsity(self, n):
        nv = self.nv
        S = lil_matrix((n * nv, n * nv))
        for i in range(n):
            S[i * nv:(i + 1) * nv, max(0, i - 1) * nv:min(n, i + 2) * nv] = 1
        return S.tocsr()


def _run(model: _Model, n, t0, tf, method, rtol, atol, deadline):
    g = model.grid(n)
    U0 = model.impose(g, t0, model.initial(g))
    nv = model.nv
    last = [t0, 0.0]

    def f(t, y):
        deadline.tick()
        U = model.impose(g, t, y.reshape(n, nv).T.copy())
        # Checked AFTER imposing the Dirichlet values: the integrator's finite-difference Jacobian perturbs every
        # unknown, and for the Dirichlet nodes (which the right-hand side ignores) it enlarges the perturbation tenfold
        # at every Jacobian evaluation without limit. Checking the raw vector made a run with many Jacobian evaluations
        # (an ignition) look like a blow-up although every state was bounded.
        m = np.max(np.abs(U))
        if not m < BLOWUP:
            raise _Blowup()
        last[0], last[1] = t, float(m)
        return model.rhs(g, t, U).T.ravel()

    with np.errstate(all="ignore"):
        D, R, v, c = model.coefficients(g, t0, U0)
        f0 = model.rhs(g, t0, U0.copy())
    free = np.ones((nv, n), bool)
    for k in range(nv):
        free[k, 0] = model.bc["left"][k][0] != "dirichlet"
        free[k, -1] = model.bc["right"][k][0] != "dirichlet"
    if np.any(~(c[free] > 0)):
        raise ValueError("time_coefficient debe ser > 0 en todos los nodos que se integran (solo puede anularse en una "
                         "cara con condición dirichlet, p. ej. velocidad cero en la pared con temperatura de pared fija)")
    if np.any(D < 0) or not np.all(np.isfinite(D)):
        raise ValueError("diffusivity debe ser finita y ≥ 0 en todo el dominio con las condiciones iniciales")
    if not np.all(np.isfinite(f0)):
        raise ValueError("las expresiones no dan valores finitos con las condiciones iniciales (división entre cero, "
                         "logaritmo o raíz de un negativo): revisar source, velocity, coeficientes y condiciones de "
                         "frontera")
    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(f, (t0, tf), U0.T.ravel(), method=method, rtol=rtol, atol=atol, dense_output=True,
                            jac_sparsity=model.sparsity(n))
        sol.blew_up = False
    except _Blowup:
        sol = type("Failed", (), {"status": -1, "blew_up": True, "t": np.array([t0, last[0]]), "message": ""})()
    except (RuntimeError, np.linalg.LinAlgError) as exc:     # singular iteration matrix: non-finite Jacobian
        sol = type("Failed", (), {"status": -1, "blew_up": False, "t": np.array([t0, last[0]]), "message": str(exc)})()
    sol.last_max = last[1]
    if sol.status != 0 and not sol.blew_up and getattr(sol, "y", None) is not None and sol.y.shape[1] > 1:
        # the integrator gave up (step below round-off): unbounded growth if the accepted solution ended at least a
        # million times larger than it was over the first half of the time it covered
        size = np.max(np.abs(sol.y), axis=0)
        ref = float(np.max(size[sol.t <= t0 + 0.5 * (sol.t[-1] - t0)]))
        sol.blew_up = bool(size[-1] > GROWTH * max(ref, 1e-300))
        sol.last_max = float(size[-1])

    def state(times):
        """U at the given times, shape (len(times), nv, n), Dirichlet values imposed."""
        Y = sol.sol(np.asarray(times, float)).T.reshape(len(times), n, nv).transpose(0, 2, 1).copy()
        for m, t in enumerate(times):
            model.impose(g, float(t), Y[m])
        return Y
    return g, sol, state, {"cell_peclet": float(np.max(np.abs(v) * g["h"] / np.where(D > 0, D, np.inf))) if model.has_v else 0.0,
                           "pure_convection": bool(model.has_v and np.any((D == 0) & (v != 0)))}


def _steep_front(tm, dt, rates, names, ext, scale, t0, span):
    """Largest isolated, self-accelerating rate of change: the signature of an ignition (or of a front crossing the
    domain). tm, dt: mid-times and lengths of the sampled intervals; rates[m, k] = max over the nodes of |Δu_k|/Δt.
    Returns None unless some variable changes at a peak rate that would cover its whole range in less than
    1/STEEP_RATE of the interval, the rate was at least 10 times lower at some EARLIER time (so a jump between the
    initial and the boundary condition, whose rate only decays, does not count) and it stays above a tenth of the peak
    for less than STEEP_FRACTION of the interval (so a sustained oscillation does not count)."""
    ok = dt > 1e-9 * span
    if not np.any(ok):
        return None
    tm, dt, rates = tm[ok], dt[ok], rates[ok]
    best = None
    for k, v in enumerate(names):
        rng = ext[v]["max"] - ext[v]["min"]
        if not rng > 1e-9 * max(scale[k], 1e-300):
            continue
        r = rates[:, k] * span / rng
        i = int(np.argmax(r))
        if (r[i] > STEEP_RATE and tm[i] - t0 > 1e-3 * span and i > 0 and np.min(r[:i]) < 0.1 * r[i]
                and np.sum(dt[r > 0.1 * r[i]]) < STEEP_FRACTION * span and (best is None or r[i] > best["peak_scaled_rate"])):
            best = {"variable": v, "t_at_peak_rate": float(tm[i]), "peak_scaled_rate": float(r[i]),
                    "fraction_of_interval_above_tenth_of_peak": float(np.sum(dt[r > 0.1 * r[i]]) / span)}
    return best


def solve_pde_1d(variables: list, domain: list, t_end: float, initial_conditions: dict, boundary_conditions: dict,
                 diffusivity: dict, source: dict | None = None, velocity: dict | None = None,
                 time_coefficient: dict | None = None, geometry: str = "slab", parameters: dict | None = None,
                 definitions: dict | None = None, t_start: float = 0.0, output_times: list | None = None,
                 n_output_times: int = 6, output_points: list | None = None, averages: dict | None = None,
                 n_points: int = 101, rtol: float = 1e-6, atol: float = 1e-9, method: str = "BDF",
                 convection_scheme: str = "central", space_variable: str = "x", time_variable: str = "t",
                 oscillation_window_fraction: float = 0.5) -> dict:
    """Method-of-lines solution of the parabolic system described in the module docstring."""
    xname, tname = EX.check_name(space_variable, "space_variable"), EX.check_name(time_variable, "time_variable")
    if convection_scheme not in ("central", "upwind"):
        raise ValueError("convection_scheme debe ser 'central' (segundo orden) o 'upwind' (primer orden, sin oscilaciones)")
    if method not in METHODS:
        raise ValueError(f"method debe ser uno de {', '.join(METHODS)}")
    model = _Model(variables, domain, geometry, parameters, definitions, diffusivity, source, velocity, time_coefficient,
                   initial_conditions, boundary_conditions, xname, tname, convection_scheme)
    t0, tf = number(t_start, "t_start"), number(t_end, "t_end")
    if not tf > t0:
        raise ValueError("t_end debe ser mayor que t_start")
    span = tf - t0
    rtol, atol = number(rtol, "rtol", True), number(atol, "atol", True)
    if rtol > 1e-2:
        raise ValueError("rtol debe ser ≤ 1e-2")
    if not isinstance(n_points, int) or not 11 <= n_points <= MAX_POINTS:
        raise ValueError(f"n_points debe ser un entero entre 11 y {MAX_POINTS}")
    if not 0.05 <= oscillation_window_fraction <= 1.0:
        raise ValueError("oscillation_window_fraction debe estar entre 0.05 y 1")
    if output_times is None:
        if not isinstance(n_output_times, int) or not 2 <= n_output_times <= MAX_OUTPUT_TIMES:
            raise ValueError(f"n_output_times debe ser un entero entre 2 y {MAX_OUTPUT_TIMES}")
        t_out = np.linspace(t0, tf, n_output_times)
    else:
        t_out = number_list(output_times, "output_times")
        if not 1 <= len(t_out) <= MAX_OUTPUT_TIMES or np.any(t_out < t0 - 1e-12 * span) or np.any(t_out > tf + 1e-12 * span):
            raise ValueError(f"output_times: entre 1 y {MAX_OUTPUT_TIMES} tiempos dentro de [t_start, t_end]")
        t_out = np.clip(t_out, t0, tf)
    a, b = model.a, model.b
    user_points = output_points is not None
    if user_points:
        x_out = number_list(output_points, "output_points")
        if not 1 <= len(x_out) <= MAX_OUTPUT_POINTS or np.any(x_out < a - 1e-12 * (b - a)) or np.any(
                x_out > b + 1e-12 * (b - a)):
            raise ValueError(f"output_points: entre 1 y {MAX_OUTPUT_POINTS} posiciones dentro del dominio")
        x_out = np.clip(x_out, a, b)
    else:
        x_out = np.linspace(a, b, 11)
    if averages is None:
        averages = {v: v for v in model.vars}
    if not isinstance(averages, dict):
        raise ValueError("averages debe ser {nombre: expresión a promediar sobre la sección}")
    avg_fns = {str(k): EX.compile_expression(t, model.names, f"averages['{k}']") for k, t in averages.items()}
    nv, s = model.nv, model.s
    warnings: list[str] = []
    deadline = Deadline()

    def analyse(n, rt, at):
        g, sol, state, info = _run(model, n, t0, tf, method, rt, at, deadline)
        if sol.status != 0:
            return {"ok": False, "sol": sol, "g": g, "info": info}
        x, V = g["x"], g["V"]
        U_out = state(t_out)
        prof = np.array([[np.interp(x_out, x, U_out[m, k]) for k in range(nv)] for m in range(len(t_out))])

        def average_series(times, states):
            series = {k: np.empty(len(times)) for k in avg_fns}
            for m, t in enumerate(times):
                env = model.env(g, float(t), states[m])
                for k, fn in avg_fns.items():
                    series[k][m] = np.sum(V * full(fn(env), x)) / np.sum(V)
            return series

        avg = average_series(t_out, U_out)
        # extrema over all nodes and a dense set of times
        ts = np.unique(np.concatenate([np.linspace(t0, tf, N_SAMPLE), sol.t, t_out]))
        ext = {v: {"max": -np.inf, "min": np.inf} for v in model.vars}
        scale = np.zeros(nv)
        sig_t = np.linspace(tf - oscillation_window_fraction * span, tf, N_SAMPLE)
        prev, r_t, r_dt, r_val = None, [], [], []
        for chunk in np.array_split(ts, max(1, len(ts) * n * nv // 2_000_000 + 1)):
            Y = state(chunk)
            tc, Yc = (chunk, Y) if prev is None else (np.concatenate([[prev[0]], chunk]), np.concatenate([prev[1][None], Y]))
            r_t.append(0.5 * (tc[1:] + tc[:-1]))
            r_dt.append(np.diff(tc))
            r_val.append(np.max(np.abs(np.diff(Yc, axis=0)), axis=2) / np.diff(tc)[:, None])
            prev = (chunk[-1], Y[-1])
            for k, v in enumerate(model.vars):
                scale[k] = max(scale[k], float(np.max(np.abs(Y[:, k]))))
                for label, fn in (("max", np.argmax), ("min", np.argmin)):
                    idx = np.unravel_index(fn(Y[:, k]), Y[:, k].shape)
                    val = float(Y[:, k][idx])
                    if (label == "max" and val > ext[v]["max"]) or (label == "min" and val < ext[v]["min"]):
                        ext[v].update({label: val, f"{xname}_at_{label}": float(x[idx[1]]),
                                       f"{tname}_at_{label}": float(chunk[idx[0]])})
        front = _steep_front(np.concatenate(r_t), np.concatenate(r_dt), np.concatenate(r_val), model.vars, ext, scale,
                             t0, span)
        # signals for the oscillation / settling diagnosis: averages and (user) point values
        Yw = state(sig_t)
        signals = {f"average[{k}]": val for k, val in average_series(sig_t, Yw).items()}
        if user_points:
            for xp in x_out[:MAX_DIAGNOSED_POINTS]:
                for k, v in enumerate(model.vars):
                    signals[f"{v}@{xname}={xp:g}"] = np.array([np.interp(xp, x, Yw[m, k]) for m in range(len(sig_t))])
        # boundary values and fluxes at the output times
        bnd = {"left": {v: {"value": [], "inflow": []} for v in model.vars},
               "right": {v: {"value": [], "inflow": []} for v in model.vars}}
        h = g["h"]
        for m, t in enumerate(t_out):
            U = U_out[m]
            D = model.coefficients(g, float(t), U)[0]
            q = model.inflows(g, float(t), U, D)
            one_sided = np.array([-(-3 * U[:, 0] + 4 * U[:, 1] - U[:, 2]) / (2 * h) * D[:, 0],
                                  (3 * U[:, -1] - 4 * U[:, -2] + U[:, -3]) / (2 * h) * D[:, -1]])
            q = np.where(np.isnan(q), one_sided, q)
            for fi, (face, j) in enumerate((("left", 0), ("right", -1))):
                for k, v in enumerate(model.vars):
                    bnd[face][v]["value"].append(float(U[k, j]))
                    bnd[face][v]["inflow"].append(float(q[fi, k]))
        U_end = U_out[-1] if t_out[-1] == tf else state([tf])[0]
        rate = np.abs(model.rhs(g, tf, U_end.copy())) * span / np.maximum(scale, atol)[:, None]
        return {"ok": True, "g": g, "sol": sol, "info": info, "profiles": prof, "averages": avg, "extrema": ext,
                "scale": np.maximum(scale, atol), "signals": signals, "sig_t": sig_t, "boundary": bnd,
                "steady_rate": float(np.max(rate)), "state": state, "front": front}

    def attempt(n, rt, at):
        """analyse, with the time spent; None when the wall-clock budget runs out."""
        start = time.monotonic()
        try:
            return analyse(n, rt, at), time.monotonic() - start
        except Timeout:
            return None, time.monotonic() - start

    def finer(n, rt, at):
        return 2 * n - 1, max(rt * 0.1, min(rt, RTOL_FLOOR)), at * 0.1

    def failure_text(sol):
        t_fail = float(sol.t[-1])
        return (f"la solución crece sin límite cerca de {tname} = {t_fail:.6g} (llegó a |u| = {sol.last_max:.3g}: "
                "explosión en tiempo finito, p. ej. embalamiento térmico sin consumo de reactivo, o inestabilidad del "
                "modelo)" if sol.blew_up else
                f"el integrador no pudo pasar de {tname} = {t_fail:.6g} ({sol.message}): expresiones no finitas o "
                "singularidad")

    started = time.monotonic()
    res, spent = attempt(n_points, rtol, atol)
    if res is None:
        raise ValueError(TIMEOUT_TEXT)
    used = (n_points, rtol, atol)
    recovered = None
    if not res["ok"]:
        # Suspected blow-up. A numerical instability does not survive refinement, a real ignition does: run again with
        # twice the intervals and tighter tolerances before deciding.
        runs = [{"n_points": n_points, "rtol": rtol, "atol": atol, f"{tname}_failure": float(res["sol"].t[-1]),
                 "max_abs_value_reached": res["sol"].last_max, "failure": failure_text(res["sol"])}]
        trial = finer(*used)
        retry, spent = attempt(*trial)
        if retry is not None and retry["ok"]:
            recovered = runs[0]
            res, used = retry, trial
        else:
            if retry is not None:
                runs.append({"n_points": trial[0], "rtol": trial[1], "atol": trial[2],
                             f"{tname}_failure": float(retry["sol"].t[-1]),
                             "max_abs_value_reached": retry["sol"].last_max, "failure": failure_text(retry["sol"])})
            out: dict = {"success": False, "variables": model.vars, "geometry": geometry, "space_variable": xname,
                         "time_variable": tname, "n_points": n_points, "method": method, "rtol": rtol, "atol": atol,
                         "ignition_or_steep_front": False}
            confirmed = len(runs) == 2
            out["blow_up"] = {"confirmed_by_refinement": confirmed, "runs": runs,
                              "check": "la corrida que falla se repite con el doble de intervalos y tolerancias 10 veces "
                                       "más estrictas: una ignición real (acotada) se completa al refinar; una explosión "
                                       "o singularidad vuelve a fallar en el mismo punto"}
            out["message"] = failure_text(res["sol"]) + (
                f". Al refinar ({trial[0]} nodos, tolerancias 10 veces más estrictas) vuelve a fallar en {tname} = "
                f"{runs[1][f'{tname}_failure']:.6g}: no es un artefacto de la malla ni del paso" if confirmed else
                ". No se pudo repetir en malla fina (tiempo límite)") + ". NO se reporta ninguna solución"
            out["warnings"] = [out["message"]]
            return out
    # evidence: finer grid and tighter tolerances; while it does not pass and there is time, the fine run becomes the
    # reported one and is checked against a finer one still (automatic refinement)
    levels: list[dict] = []
    ev: dict = {}
    while True:
        trial = finer(*used)
        ev = {"check": f"mismo problema con {trial[0]} nodos (el doble de intervalos) y tolerancias 10 veces más estrictas",
              "n_points_check": trial[0]}
        fine, spent = (None, 0.0) if trial[0] > 2 * MAX_POINTS else attempt(*trial)
        if fine is None:
            ev.update(verified=False, note="la corrida en malla fina excedió el tiempo límite")
            warnings.append("no se pudo completar la comprobación de malla (tiempo límite): resultado sin verificar")
            break
        if not fine["ok"]:
            ev.update(verified=False, note="la corrida en la malla fina no terminó: " + failure_text(fine["sol"]))
            warnings.append("la comprobación en malla fina falló: resultado sin verificar")
            break
        sc = res["scale"]
        diffs = {"profiles": rel_diff(res["profiles"], fine["profiles"], sc[None, :, None]),
                 "averages": max(rel_diff(res["averages"][k], fine["averages"][k],
                                          max(float(np.max(np.abs(fine["averages"][k]))), 1e-300)) for k in avg_fns)
                 if avg_fns else 0.0,
                 "extrema": float(max(abs(res["extrema"][v][m] - fine["extrema"][v][m]) / sc[k]
                                      for k, v in enumerate(model.vars) for m in ("max", "min")))}
        ev.update(rel_diff=diffs, max_rel_diff=float(max(diffs.values())),
                  relative_to="máximo valor absoluto de cada variable (o de cada promedio) en la corrida")
        ev["verified"] = bool(ev["max_rel_diff"] < EVIDENCE_TOL)
        ev["extrema_check"] = {v: {m: fine["extrema"][v][m] for m in ("max", "min")} for v in model.vars}
        levels.append({"n_points": used[0], "n_points_check": trial[0], "max_rel_diff": ev["max_rel_diff"]})
        projected = time.monotonic() - started + 4.0 * spent      # one more level: ≈ 2× nodes, more steps
        if (ev["verified"] or len(levels) > MAX_AUTO_REFINEMENTS or trial[0] > MAX_POINTS
                or projected > AUTO_REFINE_BUDGET * MAX_SECONDS):
            break
        res, used = fine, trial
    n_used = used[0]
    if "max_rel_diff" in ev and not ev["verified"]:
        d = [lv["max_rel_diff"] for lv in levels]
        order = float(np.clip(np.log2(d[-2] / d[-1]), 1.0, 2.0)) if len(d) > 1 and d[-1] > 0 and d[-2] > d[-1] else 2.0
        n_rec = int(np.ceil((n_used - 1) * (ev["max_rel_diff"] / EVIDENCE_TOL) ** (1.0 / order))) + 1
        ev["observed_order"] = order
        ev["recommended_n_points"] = n_rec
        warnings.append(
            f"al duplicar la malla ({n_used} → {ev['n_points_check']} nodos) los resultados cambian "
            f"{ev['max_rel_diff']:.2e} (relativo): NO son independientes de la malla. Para bajar de {EVIDENCE_TOL:g} con "
            f"el orden observado ({order:.2g}) hacen falta ≈ {n_rec} nodos"
            + (f" (más que el máximo de {MAX_POINTS}: acortar el dominio o el intervalo)" if n_rec > MAX_POINTS else
               f": repetir con n_points = {n_rec}")
            + "; con salto inicial entre la condición inicial y la de frontera, o con un frente de ignición cuyo instante "
              "se desplaza con la malla, los perfiles a tiempo fijo convergen más despacio que los extremos")
    if len(levels) > 1 or n_used != n_points:
        ev["automatic_refinement"] = {"n_points_requested": n_points, "n_points_used": n_used, "levels": levels}
        warnings.append(f"refinamiento automático: la malla pedida ({n_points} nodos) no "
                        + ("se pudo integrar" if recovered and len(levels) <= 1 else "pasó la comprobación")
                        + f"; los resultados reportados son los de {n_used} nodos (rtol = {used[1]:.1e})")
    front = res["front"]
    out = {"success": True, "variables": model.vars, "geometry": geometry, "space_variable": xname,
           "time_variable": tname, "n_points": n_used, "method": method, "rtol": used[1], "atol": used[2]}
    out["ignition_or_steep_front"] = bool(front is not None or recovered is not None)
    if out["ignition_or_steep_front"]:
        sf: dict = dict(front or {})
        sf["bounded"] = True
        sf["extrema"] = {v: {m: res["extrema"][v][m] for m in ("max", "min")} for v in model.vars}
        if "extrema_check" in ev:
            sf["extrema_refined"] = ev["extrema_check"]
            sf["survives_refinement"] = bool(ev["rel_diff"]["extrema"] < EVIDENCE_TOL)
            sf["extrema_rel_diff"] = ev["rel_diff"]["extrema"]
            if front and fine is not None and fine["ok"] and fine["front"]:
                sf[f"{tname}_at_peak_rate_refined"] = fine["front"]["t_at_peak_rate"]
        else:
            sf["survives_refinement"] = False
        if front:
            sf[f"{tname}_at_peak_rate"] = sf.pop("t_at_peak_rate")
        if recovered is not None:
            sf["first_run_failed"] = recovered
        sf["criterion"] = (f"una variable cambia a un ritmo máximo que recorrería todo su intervalo de valores en menos de "
                           f"1/{STEEP_RATE:g} del intervalo integrado, tras acelerarse al menos 10 veces, durante menos del "
                           f"{100 * STEEP_FRACTION:g} % del intervalo; es física (no explosión numérica) si la solución "
                           "queda acotada y sus extremos coinciden al refinar malla y tolerancias")
        out["steep_front"] = sf
        warnings.append(
            ("ignición o frente pronunciado" + (f" en '{front['variable']}' cerca de {tname} = {front['t_at_peak_rate']:.6g}"
                                               if front else "")
             + ": la solución queda acotada"
             + (" y sus extremos coinciden al refinar (no es una explosión numérica)" if sf["survives_refinement"] else
                ", pero sus extremos NO coinciden todavía al refinar: tratar el resultado como no verificado"))
            + (f". La primera corrida ({recovered['n_points']} nodos) falló y la refinada se completó" if recovered else ""))
    out["message"] = f"integración completa hasta {tname} = {tf:.6g}"
    out["n_time_steps"] = int(len(res["sol"].t) - 1)
    info = res["info"]
    if info["pure_convection"]:
        warnings.append("hay nodos con difusividad cero y velocidad distinta de cero (convección pura): el método es para "
                        "problemas parabólicos; los frentes se difunden numéricamente (upwind) u oscilan (central)")
    if info["cell_peclet"] > 2 and convection_scheme == "central":
        warnings.append(f"Péclet de celda |v|·Δx/D = {info['cell_peclet']:.3g} > 2 con diferencias centrales: pueden "
                        "aparecer oscilaciones espurias; aumentar n_points o usar convection_scheme='upwind'")
    if convection_scheme == "upwind" and model.has_v:
        warnings.append("convección upwind de primer orden: añade difusión numérica ≈ |v|·Δx/2; ver la comprobación de malla")

    def pack(r):
        prof = {tname: [float(v) for v in t_out], xname: [float(v) for v in x_out]}
        for k, v in enumerate(model.vars):
            prof[v] = [[float(u) for u in r["profiles"][m, k]] for m in range(len(t_out))]
        return prof

    out["profiles"] = pack(res)
    out["profiles_layout"] = f"profiles[variable][i][j] = valor en {tname} = profiles['{tname}'][i], {xname} = profiles['{xname}'][j]"
    weight = {0: "1", 1: xname, 2: f"{xname}^2"}[s]
    out["averages"] = {tname: [float(v) for v in t_out], **{k: [float(u) for u in val] for k, val in res["averages"].items()}}
    out["average_definition"] = (f"∫ expresión · {weight} d{xname} / ∫ {weight} d{xname} sobre el dominio (promedio de "
                                 "sección con el peso geométrico)")
    out["extrema"] = res["extrema"]
    out["boundary"] = res["boundary"]
    out["boundary_definition"] = ("inflow = D·∂u/∂n (n normal exterior) en cada tiempo de salida: lo que entra al dominio "
                                  "por unidad de área; en caras dirichlet se calcula con una diferencia lateral de 2.º orden")
    steady = res["steady_rate"] < STEADY_TOL
    out["steady_state"] = {"reached": bool(steady), "max_scaled_rate": res["steady_rate"],
                           "criterion": f"|∂u/∂{tname}|·(intervalo integrado)/max|u| < {STEADY_TOL:g} en todos los nodos"}
    out["oscillation"] = {k: DG.diagnose(res["sig_t"], y, float(np.max(np.abs(y))), rtol)
                          for k, y in res["signals"].items()}
    if steady:
        ts = np.linspace(t0, tf, N_SAMPLE)
        Y = res["state"](ts)
        dev = np.max(np.abs(Y - Y[-1]) / np.maximum(np.max(np.abs(Y - Y[-1]), axis=(0, 2), keepdims=True), 1e-300),
                     axis=(1, 2))
        late = np.nonzero(dev > SETTLE_BAND)[0]
        out["steady_state"]["settling_time"] = float(ts[min(late[-1] + 1, len(ts) - 1)]) if len(late) else t0
        out["steady_state"]["settling_definition"] = (f"primer {tname} a partir del cual todo el perfil queda a menos de "
                                                      f"{SETTLE_BAND:g} de su cambio total respecto del perfil final")

    out["evidence"] = ev
    out["warnings"] = warnings
    out["assumptions"] = ["ecuaciones, coeficientes y condiciones son los escritos por el usuario; no se comprueban unidades",
                          "malla uniforme, volúmenes finitos de 2.º orden; perfiles interpolados linealmente entre nodos",
                          "los extremos se buscan sobre los nodos y sobre los pasos del integrador"]
    out["not_included"] = ["ecuaciones hiperbólicas puras (frentes sin difusión) y choques", "más de una coordenada espacial",
                           "condiciones de frontera que acoplan las dos caras (reciclo, periódicas) y fronteras móviles",
                           "ecuaciones sin derivada temporal (c = 0: algebraicas o elípticas acopladas)"]
    return out
