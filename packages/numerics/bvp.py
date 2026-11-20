"""Two-point boundary value problems written by the user as text expressions.

Form: first-order system  dy_i/dx = f_i(x, y, params)  on [a, b]  (a higher-order equation is rewritten as a first-order
system by the user: y'' = g  →  y' = z, z' = g), with as many boundary conditions as variables plus unknown parameters.

Conventions
  * Boundary conditions are RESIDUAL expressions that must vanish; the value of variable `y` at x = a is written `y_a` and
    at x = b `y_b`. Danckwerts inlet `c − (1/Pe)·c' = 1` with c' = z is the residual `c_a - z_a/Pe - 1`.
  * Singular term at x = a (symmetry axis or centre, a = 0 normally): the term `(s/x)·z` cannot be evaluated at x = 0, so
    it is NOT written in `equations`; it goes in `singular_terms` as the numerator, a linear combination of the variables
    with constant coefficients: z' = φ²·R(c) − (2/x)·z  →  equations {'z': 'phi^2*R'} and singular_terms {'z': '-2*z'}.
    The solver then integrates y' = f + S·y/(x − a) (scipy's treatment of a regular singular point), which enforces the
    regularity condition S·y(a) = 0 (here z(0) = 0); that condition must still be listed among the boundary conditions.
  * Unknown parameters (eigenvalues, a free length...) are given with a guess; one extra boundary condition each.
  * Initial guess per variable: a number, an expression of x, or a table {'x': [...], 'values': [...]}. Nonlinear problems
    with several solutions converge to the one nearest to the guess.
  * Continuation: {'parameter': name, 'values': [v1, ..., v_target]} solves the problem at each value in turn, starting
    each solve from the previous solution (with automatic bisection of a step that fails); the reported solution is the
    one at the last value.

Solver: scipy `solve_bvp` (4th-order collocation, residual-controlled adaptive mesh, damped Newton). Evidence returned:
the collocation residual (max relative RMS residual over the mesh intervals), the boundary-condition residuals, and the
problem solved again with a tolerance 100 times tighter (finer mesh) compared on every reported number."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from packages.numerics import expressions as EX
from packages.numerics.common import (EVIDENCE_TOL, TIMEOUT_TEXT, Deadline, Timeout, full, number, number_list, rel_diff)

MAX_VARIABLES = 60
MAX_OUTPUT_POINTS = 2000
MAX_CONTINUATION_VALUES = 200
MAX_BISECTIONS = 6
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(5)
_STATUS = {1: "se alcanzó el máximo de nodos de la malla sin cumplir la tolerancia (max_nodes): la solución tiene una capa "
              "muy delgada, el dominio truncado es demasiado largo o la aproximación inicial queda lejos",
           2: "el jacobiano de la colocación es singular: las condiciones de frontera no determinan la solución (faltan, "
              "sobran o son dependientes), o la aproximación inicial es inadecuada (por ejemplo, todo ceros)",
           3: "el residuo de las condiciones de frontera no bajó de la tolerancia"}


def _setup(equations, domain, boundary_conditions, parameters, unknown_parameters, definitions, singular_terms, xname):
    if not isinstance(equations, dict) or not equations:
        raise ValueError(f"falta equations: {{variable: expresión de d(variable)/d{xname}}}; sistema de PRIMER orden")
    names = list(equations)
    if len(names) > MAX_VARIABLES:
        raise ValueError(f"demasiadas ecuaciones (máximo {MAX_VARIABLES})")
    try:
        a, b = (number(v, "domain") for v in domain)
    except (TypeError, ValueError):
        raise ValueError("domain debe ser [a, b] con a < b") from None
    if not a < b:
        raise ValueError("domain debe ser [a, b] con a < b")
    params = EX.parse_parameters(parameters)
    unknown = EX.parse_parameters(unknown_parameters, "unknown_parameters")
    edge = [f"{n}_a" for n in names] + [f"{n}_b" for n in names]
    base = EX.declare([("variable independiente", [xname]), ("equations", names), ("parameters", list(params)),
                       ("unknown_parameters", list(unknown)), ("valores de frontera", edge)]) - set(edge)
    defs, allnames = EX.compile_definitions(definitions, base)
    rhs = [EX.compile_expression(equations[n], allnames, f"equations['{n}']") for n in names]
    n_bc = len(names) + len(unknown)
    if not isinstance(boundary_conditions, (list, tuple)) or len(boundary_conditions) != n_bc:
        got = len(boundary_conditions) if isinstance(boundary_conditions, (list, tuple)) else 0
        raise ValueError(f"se necesitan exactamente {n_bc} condiciones de frontera ({len(names)} variables"
                         + (f" + {len(unknown)} parámetros desconocidos" if unknown else "") + f") y se dieron {got}; "
                         "cada una es una expresión residual que debe valer 0, con los nombres "
                         f"{names[0]}_a (valor en {xname} = a) y {names[0]}_b (valor en {xname} = b)")
    bc_names = set(edge) | set(params) | set(unknown)
    bcs = [EX.compile_expression(t, bc_names, f"boundary_conditions[{k}]") for k, t in enumerate(boundary_conditions)]
    sing = {}
    if singular_terms:
        if not isinstance(singular_terms, dict) or set(singular_terms) - set(names):
            raise ValueError("singular_terms debe ser {variable: numerador}, con variables de equations; el término que se "
                             f"añade a esa ecuación es numerador/({xname} − a)")
        lin_names = set(names) | set(params)
        sing = {n: EX.compile_expression(t, lin_names, f"singular_terms['{n}']") for n, t in singular_terms.items()}
    return names, a, b, params, unknown, defs, rhs, bcs, sing


def _singular_matrix(names, sing, params):
    """S of y' = f + S·y/(x − a) from the linear numerators; refuses anything not linear with constant coefficients."""
    n = len(names)
    S = np.zeros((n, n))
    rng = np.random.default_rng(7)
    probe = rng.uniform(0.5, 1.5, n)
    for var, fn in sing.items():
        i = names.index(var)
        zero = float(fn({**params, **dict.fromkeys(names, 0.0)}))
        for j, other in enumerate(names):
            S[i, j] = float(fn({**params, **{m: (1.0 if m == other else 0.0) for m in names}})) - zero
        value = float(fn({**params, **dict(zip(names, probe))}))
        if abs(zero) > 1e-12 or abs(value - S[i] @ probe) > 1e-9 * (1 + abs(value)):
            raise ValueError(f"singular_terms['{var}'] debe ser una combinación LINEAL de las variables con coeficientes "
                             "constantes (por ejemplo '-2*z'); la parte no lineal va en equations")
    return S


def _guess_functions(initial_guess, names, params, xname):
    fns = {}
    if initial_guess is None:
        return fns
    if not isinstance(initial_guess, dict) or set(initial_guess) - set(names):
        raise ValueError("initial_guess debe ser {variable: número | expresión de la variable independiente | "
                         "{'x': [...], 'values': [...]}} con variables de equations")
    gnames = {xname} | set(params)
    for n, g in initial_guess.items():
        if isinstance(g, dict):
            xs, vs = number_list(g.get("x"), f"initial_guess['{n}']['x']"), number_list(g.get("values"),
                                                                                        f"initial_guess['{n}']['values']")
            if len(xs) < 2 or len(xs) != len(vs) or np.any(np.diff(xs) <= 0):
                raise ValueError(f"initial_guess['{n}']: la tabla necesita 'x' creciente y 'values' de la misma longitud")
            fns[n] = lambda env, xs=xs, vs=vs: np.interp(env[xname], xs, vs)
        else:
            fns[n] = EX.compile_expression(g, gnames, f"initial_guess['{n}']")
    return fns


class _Problem:
    def __init__(self, names, a, b, defs, rhs, bcs, sing, unknown_names, xname, deadline):
        self.names, self.a, self.b, self.defs, self.rhs, self.bcs, self.sing = names, a, b, defs, rhs, bcs, sing
        self.unknown_names, self.xname, self.deadline = unknown_names, xname, deadline

    def env(self, x, y, p, params):
        env = dict(params)
        env.update(zip(self.unknown_names, p))
        env[self.xname] = x
        env.update(zip(self.names, y))
        return EX.apply_definitions(self.defs, env)

    def f(self, x, y, p, params):
        env = self.env(x, y, p, params)
        return np.array([full(r(env), x) for r in self.rhs])

    def bc(self, ya, yb, p, params):
        env = dict(params)
        env.update(zip(self.unknown_names, p))
        env.update({f"{n}_a": ya[i] for i, n in enumerate(self.names)})
        env.update({f"{n}_b": yb[i] for i, n in enumerate(self.names)})
        return np.array([float(c(env)) for c in self.bcs])

    def solve(self, params, x, y, p, tol, max_nodes):
        S = _singular_matrix(self.names, self.sing, params) if self.sing else None
        has_p = bool(self.unknown_names)

        def fun(xx, yy, pp=()):
            self.deadline.tick()
            return self.f(xx, yy, pp, params)

        def bcf(ya, yb, pp=()):
            return self.bc(ya, yb, pp, params)

        with np.errstate(all="ignore"):
            sol = solve_bvp(fun, bcf, x, y, p=np.asarray(p, float) if has_p else None, S=S, tol=tol, max_nodes=max_nodes,
                            bc_tol=min(tol, 1e-8))
        sol.S = S
        if not has_p:
            sol.p = np.array([])
        if sol.status == 0 and not (np.all(np.isfinite(sol.y)) and np.all(np.isfinite(sol.rms_residuals))):
            sol.status = 2
        return sol


def solve_bvp_system(equations: dict, domain: list, boundary_conditions: list, parameters: dict | None = None,
                     unknown_parameters: dict | None = None, initial_guess: dict | None = None,
                     definitions: dict | None = None, singular_terms: dict | None = None,
                     output_points: list | None = None, n_output: int = 11, integrals: dict | None = None,
                     integral_weight_power: float = 0.0, continuation: dict | None = None, tol: float = 1e-6,
                     n_mesh: int = 41, max_nodes: int = 20000, independent_variable: str = "x") -> dict:
    """Solve dy/dx = f(x, y, params) on [a, b] with residual boundary conditions; see the module docstring."""
    xname = EX.check_name(independent_variable, "independent_variable")
    names, a, b, params, unknown, defs, rhs, bcs, sing = _setup(
        equations, domain, boundary_conditions, parameters, unknown_parameters, definitions, singular_terms, xname)
    tol = number(tol, "tol", True)
    if not 1e-12 <= tol <= 1e-2:
        raise ValueError("tol debe estar entre 1e-12 y 1e-2")
    if not isinstance(n_mesh, int) or not 5 <= n_mesh <= 5000:
        raise ValueError("n_mesh (nodos de la malla inicial) debe ser un entero entre 5 y 5000")
    if not isinstance(max_nodes, int) or not n_mesh <= max_nodes <= 200000:
        raise ValueError("max_nodes debe ser un entero entre n_mesh y 200000")
    if output_points is None:
        if not isinstance(n_output, int) or not 2 <= n_output <= MAX_OUTPUT_POINTS:
            raise ValueError(f"n_output debe ser un entero entre 2 y {MAX_OUTPUT_POINTS}")
        x_out = np.linspace(a, b, n_output)
    else:
        x_out = number_list(output_points, "output_points")
        if len(x_out) > MAX_OUTPUT_POINTS or np.any(x_out < a - 1e-12 * (b - a)) or np.any(x_out > b + 1e-12 * (b - a)):
            raise ValueError(f"output_points: hasta {MAX_OUTPUT_POINTS} puntos dentro de [a, b]")
        x_out = np.clip(x_out, a, b)
    w = number(integral_weight_power, "integral_weight_power")
    if w < 0 or (w != 0 and a < 0):
        raise ValueError("integral_weight_power debe ser ≥ 0 (0 placa, 1 cilindro, 2 esfera) y, si no es 0, a ≥ 0")
    int_fns = {}
    if integrals:
        if not isinstance(integrals, dict):
            raise ValueError("integrals debe ser {nombre: expresión a integrar sobre el dominio}")
        allnames = {xname} | set(names) | set(params) | set(unknown) | {k for k, _ in defs}
        int_fns = {k: EX.compile_expression(t, allnames, f"integrals['{k}']") for k, t in integrals.items()}

    # continuation schedule
    steps = [dict(params)]
    cont_name = None
    if continuation is not None:
        if not isinstance(continuation, dict) or set(continuation) != {"parameter", "values"}:
            raise ValueError("continuation debe ser {'parameter': nombre, 'values': [v1, ..., v_final]}")
        cont_name = continuation["parameter"]
        values = number_list(continuation["values"], "continuation['values']")
        if cont_name not in params:
            raise ValueError(f"continuation: '{cont_name}' debe ser uno de los parameters (su valor se sustituye por los "
                             "de la lista; el último es el del problema que se reporta)")
        if not 1 <= len(values) <= MAX_CONTINUATION_VALUES:
            raise ValueError(f"continuation['values'] necesita entre 1 y {MAX_CONTINUATION_VALUES} valores")
        steps = [{**params, cont_name: float(v)} for v in values]

    deadline = Deadline()
    prob = _Problem(names, a, b, defs, rhs, bcs, sing, list(unknown), xname, deadline)
    guess = _guess_functions(initial_guess, names, steps[0], xname)
    x0 = np.linspace(a, b, n_mesh)
    genv = {**steps[0], xname: x0}
    y0 = np.array([full(guess[n](genv), x0) if n in guess else np.zeros_like(x0) for n in names], float)
    p0 = np.array(list(unknown.values()), float)
    warnings: list[str] = []
    if not np.all(np.isfinite(y0)):
        raise ValueError("initial_guess no da valores finitos en todo el dominio")
    with np.errstate(all="ignore"):
        f0 = prob.f(x0, y0, p0, steps[0])
    if not np.all(np.isfinite(f0)):
        if not np.all(np.isfinite(f0[:, 0])) and np.all(np.isfinite(f0[:, 1:])):
            raise ValueError(f"las ecuaciones no son finitas en {xname} = a = {a:g} (término singular tipo 1/{xname}). No "
                             f"escribir ese término en equations: ponerlo en singular_terms como numerador lineal, p. ej. "
                             f"z' = g − (2/{xname})·z  →  equations {{'z': 'g'}}, singular_terms {{'z': '-2*z'}}")
        raise ValueError("las ecuaciones no dan valores finitos con la aproximación inicial (división entre cero, "
                         "logaritmo o raíz de un negativo): dar un initial_guess dentro del dominio físico de las "
                         "expresiones" + (f"; un término 1/({b:g} − {xname}) singular en b no se admite: invertir la "
                                          "coordenada para que la singularidad quede en a" if not np.all(
                                              np.isfinite(f0[:, -1])) else ""))
    if initial_guess is None:
        warnings.append("sin initial_guess: se partió de cero en todas las variables; en problemas no lineales con varias "
                        "soluciones se obtiene la más cercana a esa aproximación")

    # solve (with continuation and bisection of failed steps)
    history = []
    try:
        x, y, p = x0, y0, p0
        sol = None
        last_good = None
        for k, target in enumerate(steps):
            queue, depth = [target], 0
            while queue:
                current = queue[0]
                trial = prob.solve(current, x, y, p, tol, max_nodes)
                if trial.status == 0:
                    queue.pop(0)
                    sol, last_good = trial, current
                    x, y, p = trial.x, trial.y, trial.p
                    if cont_name:
                        history.append({cont_name: current[cont_name], "converged": True, "n_mesh_nodes": int(len(x)),
                                        "inserted_by_bisection": current is not target})
                elif cont_name and last_good is not None and depth < MAX_BISECTIONS:
                    depth += 1
                    queue.insert(0, {**current, cont_name: 0.5 * (last_good[cont_name] + current[cont_name])})
                else:
                    sol = trial
                    if cont_name:
                        history.append({cont_name: current[cont_name], "converged": False})
                    queue = None
                    break
            if queue is None:
                break
    except Timeout:
        raise ValueError(TIMEOUT_TEXT) from None
    final = steps[-1]
    converged = sol.status == 0 and (not cont_name or last_good is final)
    out: dict = {"converged": bool(converged), "independent_variable": xname, "variables": names, "domain": [a, b],
                 "tol": tol}
    if cont_name:
        out["continuation"] = {"parameter": cont_name, "steps": history, "target": final[cont_name]}
    if not converged:
        reason = _STATUS.get(sol.status, str(sol.message))
        where = (f" en el paso de continuación {cont_name} = {history[-1][cont_name]:g}"
                 + (f" (último valor resuelto: {last_good[cont_name]:g})" if last_good else "")) if cont_name else ""
        out["message"] = (f"NO CONVERGIÓ{where}: {reason}. No se reporta ninguna solución. Probar otra aproximación "
                          "inicial (initial_guess con la forma esperada del perfil), continuación desde un valor fácil "
                          "del parámetro, o más nodos (n_mesh, max_nodes)")
        out["warnings"] = warnings + [out["message"]]
        return out

    def report(s):
        """Every number reported from a solution, as a flat dict of arrays for the refinement comparison."""
        ys = s.sol(x_out)
        with np.errstate(all="ignore"):
            dys = prob.f(x_out, ys, s.p, final)
            if s.S is not None:
                dys = dys + (s.S @ ys) / np.where(x_out > a, x_out - a, np.nan)
        spline = s.sol(x_out, 1)
        dys = np.where(np.isfinite(dys), dys, spline)
        env = prob.env(x_out, ys, s.p, final)
        ints = {}
        if int_fns:
            h = np.diff(s.x)
            xg = (s.x[:-1, None] + 0.5 * h[:, None] * (_GAUSS_X[None, :] + 1)).ravel()
            wg = (0.5 * h[:, None] * _GAUSS_W[None, :]).ravel() * xg ** w
            envg = prob.env(xg, s.sol(xg), s.p, final)
            for k, fn in int_fns.items():
                ints[k] = float(np.sum(wg * full(fn(envg), xg)))
        return ys, dys, env, ints

    ys, dys, env, ints = report(sol)
    table = {xname: [float(v) for v in x_out]}
    for i, n in enumerate(names):
        table[n] = [float(v) for v in ys[i]]
    for i, n in enumerate(names):
        table[f"d{n}_d{xname}"] = [float(v) for v in dys[i]]
    for k, _ in defs:
        table[k] = [float(v) for v in full(env[k], x_out)]
    out["message"] = "convergió"
    out["n_mesh_nodes"] = int(len(sol.x))
    out["points"] = table
    out["boundary_values"] = {**{f"{n}_a": float(sol.y[i, 0]) for i, n in enumerate(names)},
                              **{f"{n}_b": float(sol.y[i, -1]) for i, n in enumerate(names)}}
    if unknown:
        out["unknown_parameters"] = {n: float(v) for n, v in zip(unknown, sol.p)}
    if int_fns:
        volume = (b ** (w + 1) - a ** (w + 1)) / (w + 1)
        out["integrals"] = {k: {"integral": v, "average": v / volume} for k, v in ints.items()}
        out["integral_definition"] = (f"integral = ∫ {xname}^{w:g} · expresión d{xname} sobre [a, b]; average = integral / "
                                      f"∫ {xname}^{w:g} d{xname}")
    bc_res = prob.bc(sol.y[:, 0], sol.y[:, -1], sol.p, final)
    scale = np.maximum(np.max(np.abs(sol.y), axis=1), 1e-12)
    evidence: dict = {"max_rms_residual": float(np.max(sol.rms_residuals)),
                      "residual_definition": "RMS por intervalo de (y' − f)/(1 + |f|) de la colocación de scipy.solve_bvp",
                      "max_boundary_residual": float(np.max(np.abs(bc_res))), "n_iterations": int(sol.niter)}
    try:
        tol2 = max(tol * 0.01, 1e-12)
        fine = prob.solve(final, sol.x, sol.y, sol.p, tol2, max(max_nodes, 20 * len(sol.x)))
        evidence.update(check="mismo problema con tolerancia 100 veces más estricta, partiendo de esta solución",
                        tol_check=tol2)
        if fine.status != 0:
            evidence.update(verified=False, note="la corrida de comprobación no convergió")
            warnings.append("la comprobación con tolerancia más estricta no convergió: solución sin verificar")
        else:
            ys2, dys2, _, ints2 = report(fine)
            diffs = {"values": rel_diff(ys, ys2, scale[:, None]),
                     "derivatives": rel_diff(dys, dys2, np.maximum(np.max(np.abs(dys2), axis=1), 1e-12)[:, None])}
            if ints:
                diffs["integrals"] = max(abs(ints[k] - ints2[k]) / max(abs(ints2[k]), 1e-300) for k in ints)
            if unknown:
                diffs["unknown_parameters"] = rel_diff(sol.p, fine.p, np.maximum(np.abs(fine.p), 1e-300))
            evidence.update(n_mesh_nodes_check=int(len(fine.x)), rel_diff=diffs, max_rel_diff=float(max(diffs.values())))
            evidence["verified"] = bool(evidence["max_rel_diff"] < EVIDENCE_TOL)
            if not evidence["verified"]:
                warnings.append(f"al refinar, los resultados cambian {evidence['max_rel_diff']:.2e} (relativo): reducir tol")
    except Timeout:
        evidence.update(verified=False, note="la comprobación excedió el tiempo límite")
        warnings.append("no se pudo completar la comprobación de refinamiento (tiempo límite)")
    out["evidence"] = evidence
    out["warnings"] = warnings
    out["assumptions"] = ["ecuaciones, parámetros y condiciones de frontera son los escritos por el usuario; no se "
                          "comprueban unidades",
                          "si el problema tiene varias soluciones, se reporta la alcanzada desde la aproximación inicial "
                          "(o la rama seguida por la continuación); no se buscan las demás"]
    out["not_included"] = ["estabilidad de la solución encontrada", "condiciones en puntos interiores (multipunto)",
                           "singularidad en x = b (invertir la coordenada)", "dominios infinitos: truncar y comprobar "
                           "que el resultado no cambia al alargar el dominio"]
    return out
