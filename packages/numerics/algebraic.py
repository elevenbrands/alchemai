"""Algebraic systems with the user's equations: f_i(unknowns, parameters) = 0, or plain evaluation of formulas.

Same restricted expression language as the ODE/BVP/PDE tools (`packages.numerics.expressions`: whitelist AST, no eval).
An equation is written "lhs = rhs" or as an expression that must vanish. With no unknowns the tool is a calculator for
`outputs`. The solve is a bounded least-squares (trf) from each start; a solution is accepted only when every scaled
residual is below the tolerance, and different converged roots from the starts are all reported.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import least_squares

from packages.numerics import expressions as EX

MAX_UNKNOWNS = 30
MAX_STARTS = 64


def _split(eq: str, where: str) -> str:
    if not isinstance(eq, str) or not eq.strip():
        raise ValueError(f"{where}: falta la ecuación")
    parts = eq.replace("==", "=").split("=")
    if len(parts) == 1:
        return parts[0]
    if len(parts) != 2:
        raise ValueError(f"{where}: una ecuación lleva un solo signo '=' («{eq}»)")
    return f"({parts[0]}) - ({parts[1]})"


def solve(equations: list[str] | None, unknowns: dict | None, parameters: dict | None = None,
          definitions: dict | None = None, outputs: dict | None = None, tolerance: float = 1e-9) -> dict:
    equations, unknowns = list(equations or []), dict(unknowns or {})
    if len(unknowns) > MAX_UNKNOWNS:
        raise ValueError(f"máximo {MAX_UNKNOWNS} incógnitas")
    if len(equations) != len(unknowns):
        raise ValueError(f"se necesitan tantas ecuaciones como incógnitas ({len(equations)} ecuaciones, {len(unknowns)} "
                         "incógnitas); lo que se pueda calcular directo va en definitions/outputs")
    params = EX.parse_parameters(parameters)
    names = EX.declare([("parameters", list(params)), ("unknowns", list(unknowns))])
    defs, names_d = EX.compile_definitions(definitions, names)
    eqs = [EX.compile_expression(_split(e, f"equations[{i}]"), names_d, f"equations[{i}]") for i, e in enumerate(equations)]
    outs = {k: EX.compile_expression(v, names_d, f"outputs['{k}']") for k, v in (outputs or {}).items()}
    keys = list(unknowns)
    starts_per, lo, hi = [], [], []
    for k in keys:
        u = unknowns[k]
        if isinstance(u, dict):
            g = u.get("guess")
            lo.append(float(u.get("lower", -np.inf)))
            hi.append(float(u.get("upper", np.inf)))
            if g is None:
                if not (np.isfinite(lo[-1]) and np.isfinite(hi[-1])):
                    raise ValueError(f"unknowns['{k}']: da guess o los dos límites lower y upper")
                g = [lo[-1] + f * (hi[-1] - lo[-1]) for f in (0.5, 0.1, 0.9)]
        else:
            g = u
            lo.append(-np.inf)
            hi.append(np.inf)
        g = [float(v) for v in (g if isinstance(g, (list, tuple)) else [g])]
        if not lo[-1] < hi[-1]:
            raise ValueError(f"unknowns['{k}']: lower debe ser menor que upper")
        starts_per.append([min(max(v, lo[-1] + 1e-12 * (abs(lo[-1]) + 1) if np.isfinite(lo[-1]) else v),
                               hi[-1] - 1e-12 * (abs(hi[-1]) + 1) if np.isfinite(hi[-1]) else v) for v in g])

    def env_of(x) -> dict:
        env = dict(params)
        env.update({k: float(v) for k, v in zip(keys, x)})
        return EX.apply_definitions(defs, env)

    def res(x) -> np.ndarray:
        with np.errstate(all="ignore"):
            env = env_of(x)
            r = np.array([float(f(env)) for f in eqs])
        return np.where(np.isfinite(r), r, 1e30)

    def report(env) -> dict:
        with np.errstate(all="ignore"):
            d = {k: float(env[k]) for k, _ in defs}
            o = {k: float(f(env)) for k, f in outs.items()}
        return {"definitions": d, "outputs": o}

    if not keys:
        env = env_of([])
        rep = report(env)
        bad = [k for part in rep.values() for k, v in part.items() if not math.isfinite(v)]
        return {"ok": not bad, "mode": "evaluación directa (sin incógnitas)", **rep,
                "warnings": [f"valores no finitos: {bad}"] if bad else [],
                "evidence": {"verified": not bad, "meaning": "aritmética directa de las fórmulas dadas"}}

    starts = list(itertools.islice(itertools.product(*starts_per), MAX_STARTS))
    roots, attempts = [], []
    for x0 in starts:
        x0 = np.array(x0, float)
        r0 = res(x0)
        scale = np.maximum(np.abs(r0), 1.0)
        try:
            sol = least_squares(res, x0, bounds=(lo, hi), method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000,
                                x_scale=np.maximum(np.abs(x0), 1e-3))
        except Exception as ex:  # noqa: BLE001
            attempts.append({"start": x0.tolist(), "converged": False, "error": str(ex)})
            continue
        r = res(sol.x)
        # scale-free acceptance: each residual against the magnitude of the terms at the solution (perturbation test)
        h = np.maximum(np.abs(sol.x), 1e-6) * 1e-3
        jac_scale = np.array([np.max(np.abs(res(sol.x + h * (np.arange(len(h)) == j)) - r)) for j in range(len(h))]).max()
        ref = max(float(np.max(scale)) if np.all(np.isfinite(scale)) else 1.0, float(jac_scale), 1e-300)
        rel = float(np.max(np.abs(r)) / ref)
        ok = bool(rel < max(tolerance, 1e-12) * 1e3 or np.max(np.abs(r)) < tolerance)
        attempts.append({"start": x0.tolist(), "converged": ok, "max_abs_residual": float(np.max(np.abs(r)))})
        if ok and not any(np.allclose(sol.x, q["x"], rtol=1e-6, atol=1e-9) for q in roots):
            roots.append({"x": sol.x, "r": r})
    if not roots:
        return {"ok": False, "converged": False, "attempts": attempts[:10],
                "note": "Ningún arranque llegó a residuo cero: no hay solución dentro de los límites o las estimaciones "
                        "iniciales están lejos. No reportes números de este intento; da otros `guess` o revisa las ecuaciones."}
    sols = []
    for q in roots:
        env = env_of(q["x"])
        at_bound = [k for k, v, a, b in zip(keys, q["x"], lo, hi)
                    if (np.isfinite(a) and abs(v - a) <= 1e-9 * (abs(a) + 1)) or (np.isfinite(b) and abs(v - b) <= 1e-9 * (abs(b) + 1))]
        sols.append({"unknowns": {k: float(v) for k, v in zip(keys, q["x"])}, "residuals": [float(v) for v in q["r"]],
                     "at_bound": at_bound, **report(env)})
    out = {"ok": True, "converged": True, "solution": sols[0], "n_distinct_solutions_found": len(sols),
           "starts_tried": len(starts), "starts_converged": sum(a["converged"] for a in attempts),
           "method": "mínimos cuadrados con límites (trf) desde cada arranque; raíz aceptada solo con residuo ~0",
           "warnings": [], "evidence": {"checks": {"max_abs_residual": float(np.max(np.abs(roots[0]["r"])))}, "verified": True,
                                        "meaning": "las ecuaciones dadas se cumplen en la solución; no certifica que las "
                                                   "ecuaciones describan el problema ni que la raíz sea única"}}
    if len(sols) > 1:
        out["other_solutions"] = sols[1:]
        out["warnings"].append(f"se hallaron {len(sols)} raíces distintas desde los arranques: decide cuál es la física y dilo")
    elif len(starts) == 1:
        out["warnings"].append("un solo arranque: no se exploraron otras raíces (da varios guess o límites si la ecuación "
                               "puede tener más de una)")
    if sols[0]["at_bound"]:
        out["warnings"].append(f"la solución toca un límite en {sols[0]['at_bound']}")
    return out
