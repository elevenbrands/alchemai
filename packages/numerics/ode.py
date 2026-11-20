"""Initial-value problems dy/dt = f(t, y, params) written by the user as text expressions.

Integration with scipy `solve_ivp` (LSODA by default: Adams/BDF with automatic stiffness switching; Radau and BDF for
very stiff systems; RK45/DOP853 for non-stiff ones) and dense output. Besides the trajectory at the requested times the
solver reports the extrema of every variable with the time they occur (located on the dense interpolant), the zero
crossings of user `events`, whether the final state is steady (‖f‖ against the scale of the variables) and a diagnosis of
the last part of the run (sustained oscillation / decaying oscillation / steady; period and amplitude).

Evidence of correctness: the whole problem is integrated again with tolerances 100 times tighter and the largest
difference on the reported numbers, relative to the scale of each variable, is returned."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from packages.numerics import diagnostics as DG
from packages.numerics import expressions as EX
from packages.numerics.common import (EVIDENCE_TOL, TIMEOUT_TEXT, Deadline, Timeout, full, number, number_list, rel_diff)

METHODS = ("LSODA", "Radau", "BDF", "RK45", "DOP853")
MAX_VARIABLES = 200
MAX_OUTPUT_TIMES = 2000
MAX_EVENT_TIMES = 100
N_SAMPLE = 4001
BLOWUP = 1e100          # |y| beyond this stops the run: the solution is treated as unbounded (finite-time blow-up)
STALL_CALLS = 20000     # right-hand-side evaluations without advancing 1e-9 of the interval → stuck at a singularity
STEADY_TOL = 1e-6       # |dy/dt|·(t_end − t_start) / scale below this for every variable → steady state reached


class _Blowup(Exception):
    pass


class _Stall(Exception):
    pass


def _problem(equations, initial_conditions, parameters, definitions, events, tname):
    if not isinstance(equations, dict) or not equations:
        raise ValueError("falta equations: {variable: expresión de d(variable)/dt}, una por variable de estado")
    if len(equations) > MAX_VARIABLES:
        raise ValueError(f"demasiadas ecuaciones (máximo {MAX_VARIABLES})")
    names = list(equations)
    if not isinstance(initial_conditions, dict) or set(initial_conditions) != set(names):
        missing = sorted(set(names) - set(initial_conditions or {}))
        extra = sorted(set(initial_conditions or {}) - set(names))
        raise ValueError("initial_conditions debe dar un valor inicial por cada variable de equations"
                         + (f"; faltan {missing}" if missing else "") + (f"; sobran {extra}" if extra else ""))
    params = EX.parse_parameters(parameters)
    base = EX.declare([("variable independiente", [tname]), ("equations", names), ("parameters", list(params))])
    defs, allnames = EX.compile_definitions(definitions, base)
    rhs = [EX.compile_expression(equations[n], allnames, f"equations['{n}']") for n in names]
    y0 = np.array([number(initial_conditions[n], f"initial_conditions['{n}']") for n in names])
    evs = []
    for k, ev in enumerate(events or []):
        if isinstance(ev, str):
            ev = {"expression": ev}
        if not isinstance(ev, dict) or "expression" not in ev:
            raise ValueError("events: cada evento es {expression, name?, terminal?, direction?}; el evento ocurre cuando "
                             "la expresión cruza por cero")
        unknown = set(ev) - {"expression", "name", "terminal", "direction"}
        if unknown:
            raise ValueError(f"events[{k}]: claves desconocidas {sorted(unknown)}")
        direction = ev.get("direction", 0)
        if direction not in (-1, 0, 1):
            raise ValueError(f"events[{k}]: direction debe ser 0 (cualquier cruce), 1 (de negativo a positivo) o -1")
        evs.append({"name": str(ev.get("name", f"event_{k + 1}")), "terminal": bool(ev.get("terminal", False)),
                    "direction": int(direction), "text": ev["expression"],
                    "fn": EX.compile_expression(ev["expression"], allnames, f"events[{k}]")})
    return names, params, defs, rhs, y0, evs


def _env_factory(names, params, defs, tname):
    def env_of(t, y):
        env = dict(params)
        env[tname] = t
        for i, n in enumerate(names):
            env[n] = y[i]
        return EX.apply_definitions(defs, env)
    return env_of


def _integrate(names, params, defs, rhs, y0, evs, tname, t0, tf, method, rtol, atol, max_step, deadline):
    env_of = _env_factory(names, params, defs, tname)

    def f(t, y):
        deadline.tick()
        m = np.max(np.abs(y))
        if m > BLOWUP:
            raise _Blowup()
        mark[0] += 1
        if mark[0] >= STALL_CALLS:      # no progress in many evaluations: the integrator is stuck at a singularity
            if abs(t - mark[1]) < 1e-9 * abs(tf - t0):
                raise _Stall()
            mark[0], mark[1] = 0, t
        if m < np.sqrt(BLOWUP) and (t - track[0]) * (tf - t0) > 0:
            track[0] = t
        env = env_of(t, y)
        return np.array([float(r(env)) for r in rhs])

    def make_event(ev):
        def g(t, y):
            return float(ev["fn"](env_of(t, y)))
        g.terminal, g.direction = ev["terminal"], ev["direction"]
        return g

    track, mark = [t0], [0, t0]
    sign = 1.0 if tf > t0 else -1.0
    kw = {"max_step": max_step} if max_step else {}
    with np.errstate(all="ignore"):
        blew_up = stalled = False
        for _ in range(6):
            try:
                sol = solve_ivp(f, (t0, tf), y0, method=method, rtol=rtol, atol=atol, dense_output=True,
                                events=[make_event(e) for e in evs] or None, **kw)
                break
            except _Blowup:      # unbounded growth: integrate again only up to the last time with moderate values
                blew_up = True
                tf = t0 + (track[0] - t0) * (1 - 1e-9)
                track[0] = t0
                if tf == t0:
                    raise ValueError("la solución crece sin límite desde el primer paso: revisar las ecuaciones") from None
            except _Stall:
                stalled = True
                tf = track[0] - 1e-6 * (tf - t0) if track[0] != t0 else t0
                track[0], mark[0], mark[1] = t0, 0, t0
                if (tf - t0) * sign <= 0:
                    raise ValueError("el integrador no avanza desde el inicio (singularidad en el estado inicial o "
                                     "ecuaciones no finitas): revisar las ecuaciones") from None
        else:
            raise ValueError("la solución crece sin límite o es singular y no se pudo aislar el punto donde ocurre")
    sol.blew_up, sol.stalled = blew_up, stalled
    return sol, env_of


def solve_ode_system(equations: dict, initial_conditions: dict, t_end: float, t_start: float = 0.0,
                     parameters: dict | None = None, definitions: dict | None = None,
                     output_times: list | None = None, n_output: int = 11, events: list | None = None,
                     method: str = "LSODA", rtol: float = 1e-8, atol: float = 1e-10,
                     independent_variable: str = "t", oscillation_window_fraction: float = 0.5,
                     max_step: float | None = None) -> dict:
    """Integrate dy_i/dt = f_i(t, y, params) from t_start to t_end; see the module docstring."""
    tname = EX.check_name(independent_variable, "independent_variable")
    names, params, defs, rhs, y0, evs = _problem(equations, initial_conditions, parameters, definitions, events, tname)
    t0, tf = number(t_start, "t_start"), number(t_end, "t_end")
    if tf == t0:
        raise ValueError("t_end debe ser distinto de t_start")
    if method not in METHODS:
        raise ValueError(f"method debe ser uno de {', '.join(METHODS)} (LSODA, Radau y BDF son seguros con rigidez)")
    rtol, atol = number(rtol, "rtol", True), number(atol, "atol", True)
    if rtol > 1e-2:
        raise ValueError("rtol debe ser ≤ 1e-2")
    if not 0.05 <= oscillation_window_fraction <= 1.0:
        raise ValueError("oscillation_window_fraction debe estar entre 0.05 y 1")
    span, sign = abs(tf - t0), (1.0 if tf > t0 else -1.0)
    if output_times is None:
        if not isinstance(n_output, int) or not 2 <= n_output <= MAX_OUTPUT_TIMES:
            raise ValueError(f"n_output debe ser un entero entre 2 y {MAX_OUTPUT_TIMES}")
        t_out = np.linspace(t0, tf, n_output)
    else:
        t_out = number_list(output_times, "output_times")
        if len(t_out) > MAX_OUTPUT_TIMES:
            raise ValueError(f"output_times admite hasta {MAX_OUTPUT_TIMES} tiempos")
        if np.any((t_out - t0) * sign < -1e-12 * span) or np.any((t_out - tf) * sign > 1e-12 * span):
            raise ValueError("output_times debe estar dentro de [t_start, t_end]")
        t_out = np.clip(t_out, min(t0, tf), max(t0, tf))
    warnings: list[str] = []
    deadline = Deadline()
    try:
        with np.errstate(all="ignore"):
            y_start = _env_factory(names, params, defs, tname)(t0, y0)
            bad = [n for n, r in zip(names, rhs) if not np.isfinite(float(r(y_start)))]
        if bad:
            raise ValueError(f"las ecuaciones de {bad} no dan un número finito en el estado inicial (división entre "
                             "cero, logaritmo o raíz de un negativo): revisar expresiones y condiciones iniciales")
        sol, env_of = _integrate(names, params, defs, rhs, y0, evs, tname, t0, tf, method, rtol, atol, max_step, deadline)
    except Timeout:
        raise ValueError(TIMEOUT_TEXT) from None
    except (TypeError, OverflowError) as ex:
        raise ValueError(f"las ecuaciones no pudieron evaluarse ({ex})") from None
    t_last = float(sol.t[-1])
    reached_end = sol.status in (0, 1) and not sol.blew_up and not sol.stalled
    out: dict = {"success": bool(reached_end), "independent_variable": tname, "variables": names, "method": method,
                 "rtol": rtol, "atol": atol, "t_start": t0, "t_final": t_last, "n_steps": int(len(sol.t) - 1),
                 "n_rhs_evaluations": int(sol.nfev)}
    if sol.blew_up:
        out["message"] = (f"la solución crece sin límite (|y| supera {np.sqrt(BLOWUP):g}) cerca de {tname} = {t_last:.8g}, antes de "
                          f"llegar a {tf:.6g}: explosión en tiempo finito o inestabilidad. No hay solución más allá de "
                          "ese punto; los resultados cubren solo hasta ahí y no deben usarse como valores finales")
        warnings.append(out["message"])
    elif sol.stalled:
        out["message"] = (f"el integrador deja de avanzar cerca de {tname} = {t_last:.8g}, antes de llegar a {tf:.6g}: "
                          "las ecuaciones tienen ahí una singularidad (división entre cero, término no finito) o la "
                          "solución deja de existir. Los resultados cubren solo hasta ese punto")
        warnings.append(out["message"])
    elif not reached_end:
        out["message"] = (f"la integración se detuvo en {tname} = {t_last:.6g} sin llegar a {tf:.6g}: el integrador no "
                          f"pudo avanzar ({sol.message}). Suele indicar una singularidad (la solución explota), una "
                          "expresión que deja de ser finita o una discontinuidad; los resultados cubren solo hasta ahí")
        warnings.append(out["message"])
    elif sol.status == 1:
        out["message"] = f"integración terminada por un evento terminal en {tname} = {t_last:.8g}"
    else:
        out["message"] = f"integración completa hasta {tname} = {tf:.6g}"
    if len(sol.t) < 2:
        out.update(warnings=warnings)
        return out

    keep = (t_out - t_last) * sign <= 1e-12 * span
    if not np.all(keep):
        warnings.append(f"{int(np.sum(~keep))} tiempos de salida posteriores a {tname} = {t_last:.6g} se omitieron")
    t_out = t_out[keep]

    def table(solution, env_fn, times):
        ys = solution.sol(times) if len(times) else np.zeros((len(names), 0))
        tab = {tname: [float(v) for v in times]}
        for i, n in enumerate(names):
            tab[n] = [float(v) for v in ys[i]]
        if defs and len(times):
            env = env_fn(times, ys)
            for k, _ in defs:
                tab[k] = [float(v) for v in full(env[k], times)]
        return tab

    out["outputs"] = table(sol, env_of, t_out)
    y_end = sol.y[:, -1]
    env_end = env_of(t_last, y_end)
    out["final"] = {n: float(y_end[i]) for i, n in enumerate(names)}
    if defs:
        out["final_definitions"] = {k: float(env_end[k]) for k, _ in defs}

    # extrema on the dense interpolant
    lo_t, hi_t = min(t0, t_last), max(t0, t_last)
    ts = np.unique(np.concatenate([np.linspace(lo_t, hi_t, N_SAMPLE), sol.t]))
    ys = sol.sol(ts)
    scale = np.maximum(np.max(np.abs(ys), axis=1), atol)
    extrema = {}
    for i, n in enumerate(names):
        item = {}
        for label, sgn in (("min", 1.0), ("max", -1.0)):
            k = int(np.argmin(sgn * ys[i]))
            tk, vk = ts[k], ys[i, k]
            if 0 < k < len(ts) - 1:
                res = minimize_scalar(lambda t: sgn * sol.sol(t)[i], bounds=(ts[k - 1], ts[k + 1]), method="bounded",
                                      options={"xatol": 1e-10 * span})
                if res.fun < sgn * vk:
                    tk, vk = float(res.x), sgn * float(res.fun)
            item[label] = float(vk)
            item[f"{tname}_at_{label}"] = float(tk)
        extrema[n] = item
    out["extrema"] = extrema

    # events
    if evs:
        out["events"] = []
        for k, ev in enumerate(evs):
            te, ye = sol.t_events[k], sol.y_events[k]
            item = {"name": ev["name"], "expression": ev["text"], "terminal": ev["terminal"], "n_crossings": int(len(te)),
                    "times": [float(v) for v in te[:MAX_EVENT_TIMES]],
                    "states": [{n: float(y[i]) for i, n in enumerate(names)} for y in ye[:MAX_EVENT_TIMES]]}
            if len(te) > MAX_EVENT_TIMES:
                item["note"] = f"se listan los primeros {MAX_EVENT_TIMES} cruces"
            if len(te) == 0:
                item["note"] = "la expresión no cruzó por cero en el intervalo integrado"
            out["events"].append(item)

    # steady state at the end and diagnosis of the tail
    f_end = np.array([float(r(env_end)) for r in rhs])
    rate = np.abs(f_end) * span / scale
    worst = int(np.argmax(rate))
    out["steady_state"] = {"reached": bool(np.all(rate < STEADY_TOL)), "max_scaled_rate": float(rate[worst]),
                           "slowest_variable": names[worst],
                           "criterion": f"|dy/d{tname}|·(intervalo integrado)/max|y| < {STEADY_TOL:g} en todas las variables"}
    w0 = t_last - sign * oscillation_window_fraction * abs(t_last - t0)
    tw = np.linspace(min(w0, t_last), max(w0, t_last), N_SAMPLE)
    yw = sol.sol(tw)
    out["oscillation"] = {n: DG.diagnose(tw, yw[i], scale[i], rtol) for i, n in enumerate(names)}
    kinds = {d["kind"] for d in out["oscillation"].values()}
    if "irregular_oscillation" in kinds:
        warnings.append("la oscilación de la ventana final es irregular: periodo y amplitud reportados son promedios, no "
                        "valores de un ciclo límite; alargar la integración o revisar si el sistema es caótico")

    # evidence: tighter tolerances
    rt, at = max(rtol * 0.01, 1e-13), atol * 0.01
    ev_out: dict = {"check": "misma integración con tolerancias 100 veces más estrictas", "rtol_check": rt,
                    "atol_check": at}
    try:
        sol2, env2 = _integrate(names, params, defs, rhs, y0, evs, tname, t0, tf, method, rt, at, max_step, deadline)
        t2 = float(sol2.t[-1])
        common = t_out[(t_out - t2) * sign <= 0]
        t_cmp = np.append(common, t_last if (t_last - t2) * sign <= 0 else t2)
        d_traj = rel_diff(sol.sol(t_cmp), sol2.sol(t_cmp), scale[:, None])
        ev_out["max_rel_diff_outputs"] = d_traj
        diffs = [d_traj]
        if abs(t2 - t_last) > 0:
            ev_out["rel_diff_final_time"] = abs(t2 - t_last) / span
            diffs.append(ev_out["rel_diff_final_time"])
        if evs:
            d_ev, same = 0.0, True
            for k in range(len(evs)):
                a, b = sol.t_events[k], sol2.t_events[k]
                if len(a) != len(b):
                    same = False
                else:
                    d_ev = max(d_ev, rel_diff(a, b, span))
            ev_out["max_rel_diff_event_times"] = d_ev
            ev_out["same_number_of_events"] = same
            diffs.append(d_ev if same else 1.0)
        ev_out["max_rel_diff"] = float(max(diffs))
        ev_out["relative_to"] = "máximo valor absoluto de cada variable en la corrida (tiempos: intervalo integrado)"
        ev_out["verified"] = bool(ev_out["max_rel_diff"] < EVIDENCE_TOL and reached_end)
        if not ev_out["verified"] and reached_end:
            warnings.append(f"la corrida con tolerancias más estrictas difiere {ev_out['max_rel_diff']:.2e} (relativo): "
                            "los números NO son independientes de la tolerancia. Reducir rtol/atol; si la diferencia no "
                            "baja, el sistema es sensible a las condiciones iniciales (caótico) o la solución explota")
    except Timeout:
        ev_out.update(verified=False, note="la corrida de comprobación excedió el tiempo límite: resultado sin verificar")
        warnings.append("no se pudo completar la comprobación con tolerancias más estrictas (tiempo límite)")
    out["evidence"] = ev_out
    out["warnings"] = warnings
    out["assumptions"] = ["las ecuaciones, parámetros y condiciones iniciales son los escritos por el usuario; no se "
                          "comprueban unidades ni significado físico",
                          "salidas, extremos y eventos se leen del interpolante continuo del integrador"]
    out["not_included"] = ["ecuaciones algebraicas implícitas (DAE), retardos y ecuaciones con memoria",
                           "discontinuidades en el lado derecho se integran sin reinicio: usar events terminales y "
                           "encadenar corridas si hace falta precisión en el salto"]
    return out
