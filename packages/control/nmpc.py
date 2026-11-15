"""Nonlinear model predictive control on the user's ODE (or difference-equation) model, solved with pyomo + ipopt
(bloque advcontrol, part A2).

What it answers: «with my nonlinear model, these horizons, weights and limits, how does an NMPC drive the plant from this
initial state through these setpoints, which constraints become active, and how much does it cost?»

Model as DATA, never code: {states: [names], inputs: [names], parameters: {name: number}, rhs: {state: expression}}.
The expressions are parsed with `ast` into a whitelist (numbers, the declared names, + − * / **, unary ±, and the calls
exp log log10 sqrt sin cos tan sinh cosh tanh asin acos atan); anything else (attributes, subscripts, other calls,
comparisons, lambdas...) is refused. Nothing is ever passed to eval/exec. The same parsed tree is evaluated twice: with
pyomo functions to build the optimisation problem and with numpy for the plant simulation.

Optimal control problem at every sampling instant t_k (direct collocation, the simultaneous approach of Rawlings, Mayne &
Diehl 2017, §8.1.3; Biegler, Nonlinear Programming, 2010, ch. 10):

    min  Σ_{j=1..P} Σ_i q_i (x_i(t_k+jT) − r_i)² + Σ_i p_i (x_i(t_k+PT) − r_i)²
         + Σ_{j=0..M−1} Σ_m s_m Δu_m,j² + Σ_{j=0..P−1} Σ_m w_m (u_m,j − u_ref,m)² + Σ ρ ε²
    s.t. ẋ = f(x, u, θ) on each interval (u held constant: zero-order hold), x(t_k) = measured state,
         u_min ≤ u ≤ u_max, |Δu| ≤ Δu_max, x_min − ε ≤ x ≤ x_max + ε (ε = 0 when hard), |x(t_k+PT) − r| ≤ terminal tol,
         u_j = u_{M−1} for j ≥ M.

Each sampling interval is split into `elements_per_sample` finite elements with Radau IIA collocation of
`collocation_points` stages (stiffly accurate; order 2s−1; the nodes and the Butcher matrix are computed here from their
definition and checked against the simplifying conditions). r is held constant over the horizon (no preview), as in the
linear MPC. The plant is integrated independently with scipy's Radau (rtol 1e-10) under the applied input, so the one-step
prediction error of the collocation is measured at every step. Full-state feedback, no estimator: with plant-model
mismatch there is no offset-free mechanism. ipopt finds LOCAL optima; the problem is warm-started from the previous
solution shifted by one interval."""
from __future__ import annotations

import ast
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

FUNCTIONS = ("exp", "log", "log10", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan")
MAX_EXPR_CHARS = 4000
MAX_EXPR_DEPTH = 60
MAX_COLLOCATION_POINTS = 5
MAX_NLP_VARIABLES = 60000
STEP_PRED_TOL = 1e-4          # one-step collocation prediction vs plant integration, relative to the state scale
REFINE_TOL = 1e-3             # first OCP: objective and first move with twice the finite elements
CONS_TOL = 1e-6               # relative violation accepted for a hard state constraint at the sampling instants
IPOPT_OPTIONS = {"tol": 1e-10, "acceptable_tol": 1e-8, "max_iter": 3000, "print_level": 0, "mu_strategy": "adaptive"}
OK_TERMINATIONS = ("optimal", "locallyOptimal")


# ---------------------------------------------------------------------------------------------------------------------
# safe expression parsing
# ---------------------------------------------------------------------------------------------------------------------
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


def _check(node: ast.AST, names: set[str], where: str, depth: int = 0) -> None:
    if depth > MAX_EXPR_DEPTH:
        raise ValueError(f"{where}: expresión demasiado anidada")
    if isinstance(node, ast.Expression):
        return _check(node.body, names, where, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"{where}: solo se admiten constantes numéricas")
        return None
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"{where}: el nombre '{node.id}' no es un estado, entrada ni parámetro declarado")
        return None
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        _check(node.left, names, where, depth + 1)
        return _check(node.right, names, where, depth + 1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _check(node.operand, names, where, depth + 1)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords or len(node.args) != 1:
            raise ValueError(f"{where}: solo se admiten las funciones {', '.join(FUNCTIONS)} con un argumento")
        return _check(node.args[0], names, where, depth + 1)
    raise ValueError(f"{where}: elemento no permitido en la expresión ({type(node).__name__}); se admiten números, "
                     "nombres declarados, + − * / **, signo y las funciones " + ", ".join(FUNCTIONS))


def parse_expression(text: str, names: set[str], where: str) -> ast.AST:
    """Parse one right-hand side into a checked AST (raises ValueError in Spanish)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where}: falta la expresión (texto)")
    if len(text) > MAX_EXPR_CHARS:
        raise ValueError(f"{where}: expresión demasiado larga (máximo {MAX_EXPR_CHARS} caracteres)")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError as ex:
        raise ValueError(f"{where}: expresión con sintaxis inválida ({ex.msg})") from None
    _check(tree, names, where)
    return tree.body


def evaluate(node: ast.AST, env: dict, funcs: dict):
    """Evaluate a checked AST with the given name values and function table (pyomo or numpy)."""
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](evaluate(node.left, env, funcs), evaluate(node.right, env, funcs))
    if isinstance(node, ast.UnaryOp):
        v = evaluate(node.operand, env, funcs)
        return -v if isinstance(node.op, ast.USub) else v
    return funcs[node.func.id](evaluate(node.args[0], env, funcs))


NUMPY_FUNCS = {f: getattr(np, {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}.get(f, f)) for f in FUNCTIONS}


def parse_model(model: dict) -> dict:
    if not isinstance(model, dict):
        raise ValueError("falta model: {states, inputs, parameters, rhs, time ('continuous' o 'discrete')}")
    unknown = set(model) - {"states", "inputs", "parameters", "rhs", "time"}
    if unknown:
        raise ValueError(f"model: claves desconocidas {sorted(unknown)}")
    for k, why in (("states", "lista de nombres de los estados"), ("inputs", "lista de nombres de las entradas manipuladas"),
                   ("rhs", "una expresión por estado: dx/dt (continuo) o x[k+1] (discreto)")):
        if not model.get(k):
            raise ValueError(f"model: falta '{k}' ({why}); el modelo es del usuario, no hay uno por defecto")
    states, inputs = [str(s) for s in model["states"]], [str(u) for u in model["inputs"]]
    params = {str(k): float(v) for k, v in (model.get("parameters") or {}).items()}
    names = states + inputs + list(params)
    if len(set(names)) != len(names):
        raise ValueError("model: estados, entradas y parámetros deben tener nombres distintos")
    for n in names:
        if not n.isidentifier() or n in FUNCTIONS:
            raise ValueError(f"model: '{n}' no es un nombre válido (identificador, sin usar nombres de funciones)")
    if any(not math.isfinite(v) for v in params.values()):
        raise ValueError("model: los parámetros deben ser números finitos")
    rhs = model["rhs"]
    if not isinstance(rhs, dict) or set(rhs) != set(states):
        raise ValueError(f"model.rhs debe tener exactamente una expresión por estado {states}")
    kind = str(model.get("time", "continuous"))
    if kind not in ("continuous", "discrete"):
        raise ValueError("model.time debe ser 'continuous' (dx/dt = rhs) o 'discrete' (x[k+1] = rhs)")
    trees = [parse_expression(rhs[s], set(names), f"model.rhs['{s}']") for s in states]
    return {"states": states, "inputs": inputs, "parameters": params, "trees": trees, "discrete": kind == "discrete"}


# ---------------------------------------------------------------------------------------------------------------------
# Radau IIA collocation
# ---------------------------------------------------------------------------------------------------------------------
def radau_iia(s: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes c (zeros of d^{s−1}/dτ^{s−1}[τ^{s−1}(τ−1)^s], last = 1) and A_jl = ∫_0^{c_j} L_l(τ) dτ."""
    P = np.polynomial.Polynomial
    p = (P([0, 1]) ** (s - 1)) * (P([-1, 1]) ** s)
    c = np.sort(p.deriv(s - 1).roots().real)
    c[-1] = 1.0
    A = np.zeros((s, s))
    for li in range(s):
        L = P([1.0])
        for m in range(s):
            if m != li:
                L = L * P([-c[m], 1.0]) / (c[li] - c[m])
        Li = L.integ()
        A[:, li] = [Li(cj) - Li(0.0) for cj in c]
    return c, A


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def _named(v, names: list[str], what: str, default=None, required: bool = False) -> np.ndarray:
    """dict {name: value} or list in model order (None = default) → array; unknown names refused."""
    if v is None:
        if required:
            raise ValueError(f"falta {what}")
        return np.full(len(names), np.nan if default is None else float(default))
    if isinstance(v, dict):
        bad = set(v) - set(names)
        if bad:
            raise ValueError(f"{what}: nombres desconocidos {sorted(bad)}; válidos {names}")
        vals = [v.get(n) for n in names]
    elif isinstance(v, (list, tuple)):
        if len(v) != len(names):
            raise ValueError(f"{what}: necesita {len(names)} valores en el orden {names} (o un dict por nombre)")
        vals = list(v)
    else:
        raise ValueError(f"{what}: da un dict {{nombre: valor}} o una lista en el orden {names}")
    out = np.array([np.nan if x is None else float(x) for x in vals], float)
    if default is not None:
        out[np.isnan(out)] = float(default)
    if required and np.any(np.isnan(out)):
        missing = [n for n, x in zip(names, out) if np.isnan(x)]
        raise ValueError(f"{what}: faltan valores para {missing}")
    return out


def _ipopt():
    import pyomo.environ as pyo
    exe = Path.home() / ".idaes" / "bin" / "ipopt"
    opt = pyo.SolverFactory("ipopt", executable=str(exe)) if exe.is_file() else pyo.SolverFactory("ipopt")
    if not opt.available(exception_flag=False):
        raise RuntimeError("no se encontró ipopt (se busca en ~/.idaes/bin/ipopt y en el PATH): instala los binarios con "
                           "`idaes get-extensions` en el entorno del repositorio")
    for k, v in IPOPT_OPTIONS.items():
        opt.options[k] = v
    return opt


def build_ocp(pm: dict, P: int, M: int, Ts: float, ne: int, s: int, q, p_term, sw, iw, uref, u_lo, u_hi, du_max,
              x_lo, x_hi, soft_rho, term_tol):
    """Pyomo model of the OCP with mutable parameters x0, u_prev and r (built once, re-solved every step)."""
    import pyomo.environ as pyo
    nx, nu = len(pm["states"]), len(pm["inputs"])
    c, A = radau_iia(s)
    funcs = {f: getattr(pyo, f) for f in FUNCTIONS}
    m = pyo.ConcreteModel()
    m.I, m.Mu = pyo.RangeSet(0, nx - 1), pyo.RangeSet(0, nu - 1)
    m.K = pyo.RangeSet(0, P)
    m.KU = pyo.RangeSet(0, M - 1)
    m.x0 = pyo.Param(m.I, mutable=True, initialize=0.0)
    m.r = pyo.Param(m.I, mutable=True, initialize=0.0)
    m.up = pyo.Param(m.Mu, mutable=True, initialize=0.0)
    m.X = pyo.Var(m.K, m.I)
    m.U = pyo.Var(m.KU, m.Mu)
    for k in range(M):
        for j in range(nu):
            if math.isfinite(u_lo[j]):
                m.U[k, j].setlb(float(u_lo[j]))
            if math.isfinite(u_hi[j]):
                m.U[k, j].setub(float(u_hi[j]))
    m.c = pyo.ConstraintList()
    for i in range(nx):
        m.c.add(m.X[0, i] == m.x0[i])

    def U(k, j):
        return m.U[min(k, M - 1), j]

    def f(xv, k):
        env = dict(zip(pm["states"], xv))
        env.update({n: U(k, j) for j, n in enumerate(pm["inputs"])})
        env.update(pm["parameters"])
        return [evaluate(t, env, funcs) for t in pm["trees"]]
    soft_states = [i for i in range(nx) if (math.isfinite(x_lo[i]) or math.isfinite(x_hi[i])) and soft_rho is not None]
    if soft_states:
        m.E = pyo.Var(soft_states, bounds=(0, None), initialize=0.0)

    def bound_state(expr, i):
        e = m.E[i] if i in soft_states else 0.0
        if math.isfinite(x_lo[i]):
            m.c.add(expr >= float(x_lo[i]) - e)
        if math.isfinite(x_hi[i]):
            m.c.add(expr <= float(x_hi[i]) + e)
    if pm["discrete"]:
        for k in range(P):
            fx = f([m.X[k, i] for i in range(nx)], k)
            for i in range(nx):
                m.c.add(m.X[k + 1, i] == fx[i])
    else:
        h = Ts / ne
        m.Z = pyo.Var(range(P), range(ne), range(s), m.I)
        for k in range(P):
            for e in range(ne):
                start = [m.X[k, i] if e == 0 else m.Z[k, e - 1, s - 1, i] for i in range(nx)]
                fz = [f([m.Z[k, e, l, i] for i in range(nx)], k) for l in range(s)]
                for jj in range(s):
                    for i in range(nx):
                        m.c.add(m.Z[k, e, jj, i] == start[i] + h * sum(float(A[jj, l]) * fz[l][i] for l in range(s)))
                        if jj < s - 1 or e < ne - 1:
                            bound_state(m.Z[k, e, jj, i], i)
            for i in range(nx):
                m.c.add(m.X[k + 1, i] == m.Z[k, ne - 1, s - 1, i])
    for k in range(1, P + 1):
        for i in range(nx):
            bound_state(m.X[k, i], i)
    for j in range(nu):
        if math.isfinite(du_max[j]):
            for k in range(M):
                prev = m.up[j] if k == 0 else m.U[k - 1, j]
                m.c.add(m.U[k, j] - prev <= float(du_max[j]))
                m.c.add(prev - m.U[k, j] <= float(du_max[j]))
    for i in range(nx):
        if math.isfinite(term_tol[i]):
            m.c.add(m.X[P, i] - m.r[i] <= float(term_tol[i]))
            m.c.add(m.r[i] - m.X[P, i] <= float(term_tol[i]))
    obj = sum(float(q[i]) * (m.X[k, i] - m.r[i]) ** 2 for k in range(1, P + 1) for i in range(nx) if q[i] > 0)
    obj += sum(float(p_term[i]) * (m.X[P, i] - m.r[i]) ** 2 for i in range(nx) if p_term[i] > 0)
    obj += sum(float(sw[j]) * (m.U[k, j] - (m.up[j] if k == 0 else m.U[k - 1, j])) ** 2
               for k in range(M) for j in range(nu) if sw[j] > 0)
    obj += sum(float(iw[j]) * (U(k, j) - float(uref[j])) ** 2 for k in range(P) for j in range(nu) if iw[j] > 0)
    if soft_states:
        obj += sum(float(soft_rho[i]) * m.E[i] ** 2 for i in soft_states)
    m.obj = pyo.Objective(expr=obj if not isinstance(obj, (int, float)) else 0.0 * m.U[0, 0])
    m._meta = {"nx": nx, "nu": nu, "P": P, "M": M, "ne": ne, "s": s, "continuous": not pm["discrete"]}
    return m


def _set_params(m, x0, up, r):
    for i, v in enumerate(x0):
        m.x0[i] = float(v)
        m.r[i] = float(r[i]) if math.isfinite(r[i]) else 0.0
    for j, v in enumerate(up):
        m.up[j] = float(v)


def _initialise(m, x0, u0):
    nx, nu, P, M = m._meta["nx"], m._meta["nu"], m._meta["P"], m._meta["M"]
    for k in range(P + 1):
        for i in range(nx):
            m.X[k, i].value = float(x0[i])
    if m._meta["continuous"]:
        for idx in m.Z:
            m.Z[idx].value = float(x0[idx[3]])
    for k in range(M):
        for j in range(nu):
            lo, hi = m.U[k, j].lb, m.U[k, j].ub
            v = float(u0[j])
            m.U[k, j].value = min(max(v, lo if lo is not None else -math.inf), hi if hi is not None else math.inf)


def _shift(m):
    """Warm start: previous solution moved one interval earlier (last interval repeated)."""
    nx, nu, P, M = m._meta["nx"], m._meta["nu"], m._meta["P"], m._meta["M"]
    for k in range(P):
        for i in range(nx):
            m.X[k, i].value = m.X[k + 1, i].value
    if m._meta["continuous"]:
        ne, s = m._meta["ne"], m._meta["s"]
        for k in range(P - 1):
            for e in range(ne):
                for jj in range(s):
                    for i in range(nx):
                        m.Z[k, e, jj, i].value = m.Z[k + 1, e, jj, i].value
    for k in range(M - 1):
        for j in range(nu):
            m.U[k, j].value = m.U[k + 1, j].value


def _solve(m, opt) -> tuple[str, float]:
    t0 = time.time()
    res = opt.solve(m, load_solutions=False)
    tc = str(res.solver.termination_condition)
    if tc in OK_TERMINATIONS:
        m.solutions.load_from(res)
    return tc, time.time() - t0


def _rhs_numpy(pm: dict, params: dict):
    def rhs(x, u):
        env = dict(zip(pm["states"], x))
        env.update(zip(pm["inputs"], u))
        env.update(params)
        return np.array([evaluate(t, env, NUMPY_FUNCS) for t in pm["trees"]], float)
    return rhs


def _plant_step(rhs, x, u, Ts, discrete, scale):
    if discrete:
        return rhs(x, u), True
    sol = solve_ivp(lambda _, z: rhs(z, u), (0.0, Ts), x, method="Radau", rtol=1e-10, atol=1e-10 * scale)
    return sol.y[:, -1], bool(sol.success)


# ---------------------------------------------------------------------------------------------------------------------
# public engine
# ---------------------------------------------------------------------------------------------------------------------
def nmpc_closed_loop(*, model: dict | None, sampling_period: float | None, prediction_horizon: int | None,
                     simulation_time: float | None, initial_state=None, setpoints: dict | None = None,
                     state_weights=None, control_horizon: int | None = None, move_weights=None, input_weights=None,
                     input_reference=None, initial_input=None, input_min=None, input_max=None, input_rate_max=None,
                     state_min=None, state_max=None, state_soft_weight=None, terminal_weights=None,
                     terminal_tolerance=None, plant_parameters: dict | None = None, collocation_points: int = 3,
                     elements_per_sample: int = 2, check_refinement: bool = True,
                     return_trajectories: bool = True) -> dict:
    """Closed-loop NMPC on the user's model. See the module docstring. Raises ValueError (Spanish) on missing data."""
    import pyomo.environ as pyo
    pm = parse_model(model)
    S_, U_ = pm["states"], pm["inputs"]
    nx, nu = len(S_), len(U_)
    if sampling_period is None or not float(sampling_period) > 0:
        raise ValueError("falta sampling_period (> 0): periodo del NMPC en las unidades de tiempo del modelo (decisión de "
                         "diseño del usuario)")
    if prediction_horizon is None or int(prediction_horizon) < 1:
        raise ValueError("falta prediction_horizon (≥ 1, en periodos de muestreo): decisión de diseño del usuario")
    if simulation_time is None or not float(simulation_time) > 0:
        raise ValueError("falta simulation_time (> 0): duración del escenario en lazo cerrado")
    Ts, P = float(sampling_period), int(prediction_horizon)
    M = P if control_horizon is None else int(control_horizon)
    if not 1 <= M <= P:
        raise ValueError("debe cumplirse 1 ≤ control_horizon ≤ prediction_horizon")
    n_steps = int(round(float(simulation_time) / Ts))
    if n_steps < 1 or n_steps > 2000:
        raise ValueError("simulation_time debe cubrir entre 1 y 2000 periodos de muestreo")
    if initial_state is None:
        raise ValueError(f"falta initial_state: valor inicial de cada estado {S_} (dato del usuario)")
    x0 = _named(initial_state, S_, "initial_state", required=True)
    if not setpoints or not isinstance(setpoints, dict):
        raise ValueError("faltan setpoints: {estado: valor} o {estado: lista de un valor por periodo} (consignas)")
    bad = set(setpoints) - set(S_)
    if bad:
        raise ValueError(f"setpoints: estados desconocidos {sorted(bad)}")
    R = np.full((n_steps, nx), np.nan)
    for name, v in setpoints.items():
        i = S_.index(name)
        if isinstance(v, (int, float)):
            R[:, i] = float(v)
        else:
            v = [float(z) for z in v]
            if len(v) != n_steps:
                raise ValueError(f"setpoints['{name}']: necesita un valor por periodo ({n_steps}) o un escalar")
            R[:, i] = v
    if state_weights is None:
        raise ValueError("falta state_weights: peso q de cada estado con consigna (dict {estado: q}); es la sintonía y "
                         "depende de las unidades, no se supone")
    q = _named(state_weights, S_, "state_weights", default=0.0)
    if any(q[S_.index(n)] <= 0 for n in setpoints):
        raise ValueError("state_weights: cada estado con consigna necesita un peso > 0")
    if any(q[i] > 0 and np.isnan(R[0, i]) for i in range(nx)):
        raise ValueError("state_weights: un estado con peso > 0 necesita consigna en setpoints")
    p_term = _named(terminal_weights, S_, "terminal_weights", default=0.0)
    sw = _named(move_weights, U_, "move_weights", default=0.0)
    iw = _named(input_weights, U_, "input_weights", default=0.0)
    if np.any(q < 0) or np.any(p_term < 0) or np.any(sw < 0) or np.any(iw < 0):
        raise ValueError("los pesos deben ser ≥ 0")
    uref = _named(input_reference, U_, "input_reference")
    if np.any((iw > 0) & np.isnan(uref)):
        raise ValueError("input_weights > 0 necesita input_reference (la entrada objetivo que se penaliza; dato del usuario)")
    u_lo, u_hi = _named(input_min, U_, "input_min", default=-math.inf), _named(input_max, U_, "input_max", default=math.inf)
    du_max = _named(input_rate_max, U_, "input_rate_max", default=math.inf)
    if np.any(u_lo >= u_hi) or np.any(du_max <= 0):
        raise ValueError("input_min < input_max y input_rate_max > 0")
    up0 = _named(initial_input, U_, "initial_input")
    if (np.any(sw > 0) or np.any(np.isfinite(du_max))) and np.any(np.isnan(up0)):
        raise ValueError("move_weights o input_rate_max necesitan initial_input (la entrada aplicada antes de t = 0, para "
                         "el primer Δu): dato del usuario")
    if np.any(np.isfinite(up0) & ((up0 < u_lo) | (up0 > u_hi))):
        raise ValueError("initial_input fuera de [input_min, input_max]")
    x_lo, x_hi = _named(state_min, S_, "state_min", default=-math.inf), _named(state_max, S_, "state_max", default=math.inf)
    if np.any(x_lo >= x_hi):
        raise ValueError("state_min < state_max")
    rho = None
    if state_soft_weight is not None:
        rho = _named(state_soft_weight, S_, "state_soft_weight", default=0.0)
        if np.any(rho[np.isfinite(x_lo) | np.isfinite(x_hi)] <= 0):
            raise ValueError("state_soft_weight debe ser > 0 en cada estado con límite")
    term_tol = _named(terminal_tolerance, S_, "terminal_tolerance", default=math.inf)
    if np.any(term_tol < 0) or np.any(np.isfinite(term_tol) & np.isnan(R[0])):
        raise ValueError("terminal_tolerance ≥ 0 y solo en estados con consigna")
    s, ne = int(collocation_points), int(elements_per_sample)
    if not 1 <= s <= MAX_COLLOCATION_POINTS or not 1 <= ne <= 20:
        raise ValueError(f"collocation_points entre 1 y {MAX_COLLOCATION_POINTS}, elements_per_sample entre 1 y 20")
    if (0 if pm["discrete"] else P * ne * s * nx) + P * nx + M * nu > MAX_NLP_VARIABLES:
        raise ValueError("problema demasiado grande: reduce prediction_horizon o la colocación")
    plant_params = dict(pm["parameters"])
    if plant_parameters:
        bad = set(plant_parameters) - set(plant_params)
        if bad:
            raise ValueError(f"plant_parameters: parámetros desconocidos {sorted(bad)}")
        plant_params.update({k: float(v) for k, v in plant_parameters.items()})
    assumptions, warnings, reasons = [], [], []
    if np.all(np.isinf(u_lo)) and np.all(np.isinf(u_hi)):
        warnings.append("no hay límites de entrada: el NMPC puede pedir entradas sin sentido físico")
    rhs_model, rhs_plant = _rhs_numpy(pm, pm["parameters"]), _rhs_numpy(pm, plant_params)
    try:
        f0 = rhs_model(x0, np.array([up0[j] if math.isfinite(up0[j]) else (
            (u_lo[j] + u_hi[j]) / 2 if math.isfinite(u_lo[j]) and math.isfinite(u_hi[j]) else 0.0) for j in range(nu)]))
    except (ZeroDivisionError, FloatingPointError, OverflowError, ValueError) as ex:
        raise ValueError(f"el modelo no se puede evaluar en initial_state: {ex}") from None
    if not np.all(np.isfinite(f0)):
        raise ValueError("el modelo da valores no finitos en initial_state (revisa expresiones y parámetros)")
    scale = np.maximum(np.abs(x0), 1e-8 + np.max(np.abs(x0)) * 1e-6)
    opt = _ipopt()
    m = build_ocp(pm, P, M, Ts, ne, s, q, p_term, sw, iw, uref, u_lo, u_hi, du_max, x_lo, x_hi, rho, term_tol)
    u_init = up0.copy()                   # numerical initial guess only: the given u_prev, else mid-range / a bound / 0
    for j in range(nu):
        if np.isnan(u_init[j]):
            lo, hi = u_lo[j], u_hi[j]
            u_init[j] = (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else (
                lo if math.isfinite(lo) else (hi if math.isfinite(hi) else 0.0))
    up = np.where(np.isnan(up0), 0.0, up0)
    _set_params(m, x0, up, R[0])
    _initialise(m, x0, u_init)
    X, Uapp = np.empty((n_steps + 1, nx)), np.empty((n_steps, nu))
    X[0] = x0
    x = x0.copy()
    stats = {"solves": 0, "failed": 0, "terminations": {}, "max_solve_seconds": 0.0, "total_solve_seconds": 0.0}
    pred_err, first, objs = 0.0, None, []
    stopped = None
    for k in range(n_steps):
        _set_params(m, x, up, R[k])
        tc, dt_s = _solve(m, opt)
        stats["solves"] += 1
        stats["terminations"][tc] = stats["terminations"].get(tc, 0) + 1
        stats["max_solve_seconds"] = max(stats["max_solve_seconds"], dt_s)
        stats["total_solve_seconds"] += dt_s
        if tc not in OK_TERMINATIONS:
            stats["failed"] += 1
            stopped = {"step": k, "time": k * Ts, "termination": tc}
            break
        objs.append(float(pyo.value(m.obj)))
        u = np.array([m.U[0, j].value for j in range(nu)], float)
        xpred = np.array([m.X[1, i].value for i in range(nx)], float)
        if first is None:
            first = {"objective": objs[-1], "u_plan": [[m.U[min(j, M - 1), jj].value for jj in range(nu)] for j in range(P)],
                     "x_pred": [[m.X[j, i].value for i in range(nx)] for j in range(P + 1)]}
        xn, ok = _plant_step(rhs_plant, x, u, Ts, pm["discrete"], scale)
        if not ok or not np.all(np.isfinite(xn)):
            stopped = {"step": k, "time": k * Ts, "termination": "fallo de la integración de la planta"}
            Uapp[k] = u
            break
        if not plant_parameters:
            pred_err = max(pred_err, float(np.max(np.abs(xpred - xn) / scale)))
        Uapp[k], X[k + 1] = u, xn
        x, up = xn, u
        _shift(m)
    n_done = k + 1 if stopped is None else stopped["step"]
    X, Uapp, Rr = X[:n_done + 1], Uapp[:n_done], R[:n_done]
    # refinement check of the first OCP --------------------------------------------------------------------------------
    refine = None
    if check_refinement and not pm["discrete"] and first is not None:
        m2 = build_ocp(pm, P, M, Ts, 2 * ne, s, q, p_term, sw, iw, uref, u_lo, u_hi, du_max, x_lo, x_hi, rho, term_tol)
        _set_params(m2, x0, np.where(np.isnan(up0), 0.0, up0), R[0])
        _initialise(m2, x0, u_init)
        tc2, _ = _solve(m2, opt)
        if tc2 in OK_TERMINATIONS:
            u2 = np.array([m2.U[0, j].value for j in range(nu)], float)
            urange = np.where(np.isfinite(u_hi - u_lo), u_hi - u_lo, np.maximum(np.abs(u2), 1.0))
            refine = {"elements_per_sample": 2 * ne,
                      "objective_rel_diff": abs(float(pyo.value(m2.obj)) - first["objective"])
                      / max(abs(first["objective"]), 1e-300),
                      "first_move_rel_diff": float(np.max(np.abs(u2 - np.array(first["u_plan"][0])) / urange))}
            if refine["objective_rel_diff"] > REFINE_TOL or refine["first_move_rel_diff"] > REFINE_TOL:
                reasons.append("el primer problema cambia al duplicar los elementos de colocación "
                               f"(objetivo {refine['objective_rel_diff']:.2e}, primer movimiento "
                               f"{refine['first_move_rel_diff']:.2e}): aumenta elements_per_sample o collocation_points")
        else:
            refine = {"termination": tc2}
            reasons.append(f"el problema refinado no convergió ({tc2})")
    # indices ----------------------------------------------------------------------------------------------------------
    err = np.where(np.isnan(Rr), 0.0, Rr - X[1:])
    dU = np.diff(np.vstack([np.where(np.isnan(up0), Uapp[:1] if len(Uapp) else 0.0, up0), Uapp]), axis=0) if len(Uapp) else Uapp
    stage = (np.sum(q * err ** 2, axis=1) + np.sum(sw * dU ** 2, axis=1)
             + (np.sum(iw * (Uapp - np.where(np.isnan(uref), 0.0, uref)) ** 2, axis=1) if np.any(iw > 0) else 0.0))
    viol_lo = np.max(np.maximum(x_lo - X[1:], 0) / scale, initial=0.0) if len(X) > 1 else 0.0
    viol_hi = np.max(np.maximum(X[1:] - x_hi, 0) / scale, initial=0.0) if len(X) > 1 else 0.0
    tolu = 1e-9 * np.maximum(1.0, np.where(np.isfinite(u_hi - u_lo), u_hi - u_lo, 1.0))
    act = {"inputs": [{"input": U_[j],
                       "fraction_at_min": float(np.mean(Uapp[:, j] <= u_lo[j] + tolu[j])) if len(Uapp) else 0.0,
                       "fraction_at_max": float(np.mean(Uapp[:, j] >= u_hi[j] - tolu[j])) if len(Uapp) else 0.0,
                       "fraction_at_rate_limit": float(np.mean(np.abs(dU[:, j]) >= du_max[j] * (1 - 1e-7)))
                       if len(Uapp) and math.isfinite(du_max[j]) else 0.0} for j in range(nu)],
           "state_max_rel_violation_at_samples": float(max(viol_lo, viol_hi))}
    if rho is None and act["state_max_rel_violation_at_samples"] > CONS_TOL:
        if plant_parameters:
            warnings.append("la planta (con plant_parameters) viola un límite duro de estado que el modelo respeta: "
                            f"{act['state_max_rel_violation_at_samples']:.2e} (relativo)")
        else:
            reasons.append(f"un límite duro de estado se viola en la planta ({act['state_max_rel_violation_at_samples']:.2e})")
    indices = {"closed_loop_cost": float(np.sum(stage)), "cost_per_step": stage.tolist(),
               "IAE": (Ts * np.sum(np.abs(err), axis=0)).tolist(),
               "final_error": (err[-1] if len(err) else np.zeros(nx)).tolist(),
               "TV_u": np.sum(np.abs(dU), axis=0).tolist() if len(Uapp) else [0.0] * nu}
    # evidence ---------------------------------------------------------------------------------------------------------
    if stopped is not None:
        reasons.append(f"el lazo se detuvo en el paso {stopped['step']}: {stopped['termination']}")
        warnings.append(f"ipopt no resolvió el problema en t = {stopped['time']:g} ({stopped['termination']}): restricciones "
                        "incompatibles (p. ej. límites de estado o tolerancia terminal inalcanzables) o mala inicialización")
    if not plant_parameters and pred_err > STEP_PRED_TOL and not pm["discrete"]:
        reasons.append(f"la predicción de un paso de la colocación difiere {pred_err:.2e} de la integración de la planta")
    warnings.append("ipopt encuentra óptimos LOCALES: con un modelo no lineal puede haber otros mejores (Rawlings, Mayne y "
                    "Diehl, ej. 8.1); el arranque en caliente sigue la solución anterior")
    if not np.any(np.isfinite(term_tol)):
        warnings.append("sin restricción terminal: la estabilidad en lazo cerrado no está garantizada por construcción")
    if plant_parameters:
        assumptions.append(f"la planta usa plant_parameters {plant_parameters} distintos del modelo del NMPC (prueba de "
                           "robustez); sin estimador, el desajuste deja offset")
    assumptions += [
        "realimentación del estado completo medido sin ruido, en cada instante de muestreo (sin estimador)",
        "entrada constante durante cada periodo (retenedor de orden cero); la consigna se mantiene constante en el horizonte "
        "(sin anticipación de cambios futuros)",
        "unidades = las del modelo del usuario; los pesos actúan sobre esas unidades",
        f"colocación de Radau IIA con {s} punto(s) y {ne} elemento(s) por periodo" if not pm["discrete"]
        else "modelo discreto: x[k+1] = rhs evaluado exactamente",
        "IAE con la suma de |r − x| en los instantes de muestreo × Ts (no integra entre muestras)",
    ]
    out = {"model": {"states": S_, "inputs": U_, "parameters": pm["parameters"], "time": "discrete" if pm["discrete"]
                     else "continuous"},
           "controller": {"sampling_period": Ts, "prediction_horizon": P, "control_horizon": M,
                          "state_weights": dict(zip(S_, q.tolist())), "terminal_weights": dict(zip(S_, p_term.tolist())),
                          "move_weights": dict(zip(U_, sw.tolist())), "input_weights": dict(zip(U_, iw.tolist())),
                          "collocation_points": s, "elements_per_sample": ne,
                          "nlp_variables": int(m.nvariables()), "nlp_constraints": int(m.nconstraints())},
           "nmpc": {"steps_completed": int(n_done), "stopped": stopped, "indices": indices, "constraint_activity": act,
                    "solver": {**stats, "ipopt_options": IPOPT_OPTIONS},
                    "first_plan": first, "objective_per_step": objs}}
    if return_trajectories:
        out["trajectories"] = {"time": (np.arange(n_done + 1) * Ts).tolist(),
                               "states": {n: X[:, i].tolist() for i, n in enumerate(S_)},
                               "inputs": {n: Uapp[:, j].tolist() for j, n in enumerate(U_)},
                               "setpoints": {n: Rr[:, i].tolist() for i, n in enumerate(S_) if not np.all(np.isnan(Rr[:, i]))},
                               "note": "states en t = 0, Ts, …; inputs y setpoints = valor aplicado/vigente durante cada "
                                       "periodo [k·Ts, (k+1)·Ts)"}
    out.update(warnings=warnings, assumptions=assumptions,
               not_included=["estimación de estado / perturbaciones (MHE, EKF) y acción integral ante desajuste",
                             "ruido de medición e incertidumbre (NMPC robusto o estocástico)",
                             "búsqueda de óptimo global (multiarranque)",
                             "restricciones de estado entre los nodos de colocación"],
               method=("NMPC por colocación directa (Radau IIA, enfoque simultáneo) con pyomo + ipopt "
                       f"{'(modelo discreto)' if pm['discrete'] else ''}; planta integrada aparte con scipy Radau "
                       "(rtol 1e-10); modelo dado como datos y analizado con un analizador de expresiones restringido "
                       "(sin eval)"),
               evidence={"verified": not reasons, "reasons": reasons,
                         "one_step_prediction_max_rel_diff": None if plant_parameters else pred_err,
                         "tolerance_prediction": STEP_PRED_TOL, "refinement": refine,
                         "solves": stats["solves"], "failed_solves": stats["failed"],
                         "certifies": "la numérica de esta corrida (ipopt óptimo local en cada paso, predicción de la "
                                      "colocación igual a la integración de la planta, resultado independiente de la malla "
                                      "de colocación, límites duros respetados en los muestreos); NO certifica óptimo "
                                      "global, estabilidad ni que el modelo represente a la planta"})
    return out
