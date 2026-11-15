"""Membrane modules by the solution-diffusion model: multicomponent gas permeation and reverse osmosis.

Gas permeation (`gas_permeation`). Local flux of each species, with the permeance Q_i given by the user (or permeability
over thickness):
    J_i = Q_i (p_h x_i - p_l y_i)                    [mol/(m2 s)]
and four idealised flow patterns in one module, all with constant pressure on each side:
  * complete_mixing: both sides perfectly mixed (algebraic; a binary has a closed quadratic, used as a check);
  * cross_flow: feed side in plug flow, permeate withdrawn locally without mixing (Weller-Steiner);
  * cocurrent / countercurrent: both sides in plug flow (the countercurrent case is a boundary-value problem solved by
    shooting from the retentate end).
Rating (area given), or design (stage cut, purity or recovery given -> area). Working variables are dimensionless:
component flows over the feed flow and the area as a = A Q_max p_h / F, so the equations are dr_i/da = -q_i (x_i - g y_i)
with q_i = Q_i / Q_max and g = p_l / p_h.

Reverse osmosis (`reverse_osmosis`). Water and solute by solution-diffusion in the mass units of Geankoplis (chap. 13):
    N_w = A_w (dP - (pi_m - pi_p))      N_s = A_s (c_m - c_p)      c_p = N_s / J_v,  J_v = N_w / rho_p
osmotic pressure pi = phi nu c R T / M (van 't Hoff when phi = 1, the user's osmotic coefficient otherwise), film-model
concentration polarisation c_m = c_p + (c_b - c_p) exp(J_v / k) when the user gives k, and a plug-flow feed channel
integrated over the area for a finite recovery. Nothing here is a membrane property: A_w, A_s, k and Q_i are the user's.

Every result carries the component balances with the permeate integrated independently of the retentate, the change of
the answer when the integration tolerance is relaxed a thousandfold, and the closed forms where they exist."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root
from scipy.special import expit

R_GAS = 8.314462618            # J/(mol K)
PATTERNS = ("complete_mixing", "cross_flow", "cocurrent", "countercurrent")
TARGETS = ("permeate_mole_fraction", "retentate_mole_fraction", "recovery_in_permeate", "recovery_in_retentate")
RTOL_TIGHT, RTOL_LOOSE = 1e-10, 1e-7      # the answer must not move between the two
BALANCE_TOL = 1e-6                        # component balance closure, relative to the feed of the component
STEP_TOL = 1e-5                           # change of mole fractions / stage cut between the two tolerances
TOTAL_PERMEATION = 1.0 - 1e-9             # stage cut treated as «everything permeates»
N_PROFILE = 21


# ---------------------------------------------------------------------------------------------------------------
# Gas permeation
# ---------------------------------------------------------------------------------------------------------------
def _need(value, name: str, what_for: str):
    if value is None:
        raise ValueError(f"Falta {name}: {what_for}. No se supone ningún valor.")
    return value


def _positive(value, name: str, what_for: str, allow_zero: bool = False) -> float:
    v = float(_need(value, name, what_for))
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} debe ser {'no negativo' if allow_zero else 'positivo'} (recibido {value!r}).")
    return v


def _local_permeate(x: np.ndarray, q: np.ndarray, g: float) -> tuple[np.ndarray, float]:
    """Composition y' and dimensionless total flux J of the permeate that leaves a point where the feed side is x and the
    permeate is not mixed with anything else: y'_i J = q_i (x_i - g y'_i), sum y' = 1. Newton from J = 0 on a convex
    decreasing function: monotone convergence."""
    if g <= 0.0:
        J = float(np.dot(q, x))
        return (q * x / J if J > 0 else x.copy()), J
    J = 0.0
    for _ in range(200):
        den = J + q * g
        h = float(np.sum(q * x / den)) - 1.0
        dh = -float(np.sum(q * x / den ** 2))
        step = h / dh
        J -= step
        if abs(step) <= 1e-15 * max(J, 1e-300):
            break
    J = max(J, 0.0)
    y = q * x / (J + q * g)
    return y / y.sum(), J


def _complete_mixing_y(theta: float, a: float, z: np.ndarray, q: np.ndarray, g: float) -> np.ndarray:
    return q * a * z / (theta * (1.0 - theta) + q * a * (theta + g * (1.0 - theta)))


def _complete_mixing(z: np.ndarray, q: np.ndarray, g: float, a: float) -> dict:
    a_total = float(np.sum(z / q)) / (1.0 - g) if g < 1.0 else math.inf      # area at which everything has permeated
    if a >= a_total:
        return {"r": np.zeros_like(z), "p": z.copy(), "p_flux": z.copy(), "total_permeation": True, "converged": True,
                "residual": 0.0, "profile": None}
    f = lambda th: float(np.sum(_complete_mixing_y(th, a, z, q, g))) - 1.0   # noqa: E731
    lo, hi = 1e-15, 1.0 - 1e-13
    if f(lo) <= 0.0:
        theta = 0.0
    else:
        theta = brentq(f, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=500)
    y = _complete_mixing_y(theta, a, z, q, g)
    y = y / y.sum()
    p = theta * y
    r = np.clip(z - p, 0.0, None)
    x = r / r.sum()
    p_flux = q * a * (x - g * y)                       # the same permeate from the rate equations: independent of z - r
    return {"r": r, "p": p, "p_flux": p_flux, "total_permeation": False, "converged": True,
            "residual": float(np.max(np.abs(p_flux - p))), "profile": None}


def _plug_rhs(pattern: str, q: np.ndarray, g: float):
    n = len(q)

    def rhs(_a, s):
        r = np.clip(s[:n], 0.0, None)
        tot = r.sum()
        if tot <= 1e-300:
            return np.zeros(2 * n)
        x = r / tot
        if pattern == "cross_flow":
            y, _ = _local_permeate(x, q, g)
        else:
            # permeate flowing past this point = the integrated flux itself (no cancellation, unlike z - r): s[n:] going
            # downstream in cocurrent; minus it when the countercurrent case is integrated backwards from the retentate end
            p = s[n:] if pattern == "cocurrent" else -s[n:]
            ptot = p.sum()
            y = p / ptot if ptot > 1e-200 else _local_permeate(x, q, g)[0]
        flux = q * (x - g * y)
        return np.concatenate([-flux, flux])

    return rhs


def _integrate(pattern: str, z: np.ndarray, q: np.ndarray, g: float, a: float, rtol: float,
               r_out: np.ndarray | None = None) -> dict:
    """Cross-flow and cocurrent: initial-value problem from the feed end. Countercurrent with a trial retentate r_out: the
    same equations integrated from the retentate end back to the feed end (s = a_total - a)."""
    n = len(q)
    grid = np.linspace(0.0, a, N_PROFILE)
    if pattern == "countercurrent":
        base = _plug_rhs(pattern, q, g)
        sol = solve_ivp(lambda s_, w: -base(s_, w), (0.0, a), np.concatenate([r_out, np.zeros(n)]), method="LSODA",
                        rtol=rtol, atol=1e-15, t_eval=grid)
        r_feed_end = sol.y[:n, -1]
        return {"ok": bool(sol.success), "r_feed_end": r_feed_end, "p_flux": -sol.y[n:, -1], "grid": a - sol.t,
                "r_profile": sol.y[:n]}
    sol = solve_ivp(_plug_rhs(pattern, q, g), (0.0, a), np.concatenate([z, np.zeros(n)]), method="LSODA",
                    rtol=rtol, atol=1e-15, t_eval=grid)
    return {"ok": bool(sol.success), "r": np.clip(sol.y[:n, -1], 0.0, None), "p_flux": sol.y[n:, -1], "grid": sol.t,
            "r_profile": sol.y[:n]}


def _countercurrent(z: np.ndarray, q: np.ndarray, g: float, a: float, rtol: float, guess: np.ndarray | None) -> dict:
    """Shooting on the fraction of each component that stays in the retentate (logit variables keep it inside 0-1)."""
    if guess is None:
        guess = _integrate("cross_flow", z, q, g, a, 1e-8)["r"]
    frac0 = np.clip(guess / z, 1e-12, 1.0 - 1e-12)

    def shoot(u):
        frac = expit(u)
        return _integrate("countercurrent", z, q, g, a, rtol, frac * z)

    def res(u):
        return shoot(u)["r_feed_end"] / z - 1.0

    sol = root(res, np.log(frac0 / (1.0 - frac0)), method="hybr", tol=1e-13)
    out = shoot(sol.x)
    r = z * expit(sol.x)
    residual = float(np.max(np.abs(out["r_feed_end"] / z - 1.0)))
    return {"r": r, "p": z - r, "p_flux": out["p_flux"], "ok": out["ok"], "converged": residual < 1e-8,
            "residual": residual, "grid": out["grid"], "r_profile": out["r_profile"], "r_out": r}


def _solve(pattern: str, z: np.ndarray, q: np.ndarray, g: float, a: float, rtol: float = RTOL_TIGHT,
           guess: np.ndarray | None = None) -> dict:
    if pattern == "complete_mixing":
        return _complete_mixing(z, q, g, a)
    if pattern == "countercurrent":
        return _countercurrent(z, q, g, a, rtol, guess)
    s = _integrate(pattern, z, q, g, a, rtol)
    return {"r": s["r"], "p": z - s["r"], "p_flux": s["p_flux"], "converged": s["ok"], "residual": 0.0,
            "grid": s["grid"], "r_profile": s["r_profile"]}


def _metric(kind: str, idx: int | None, z: np.ndarray, s: dict) -> float:
    r, p = s["r"], s["p"]
    if kind == "stage_cut":
        return float(p.sum())
    if kind == "permeate_mole_fraction":
        return float(p[idx] / p.sum()) if p.sum() > 0 else math.nan
    if kind == "retentate_mole_fraction":
        return float(r[idx] / r.sum()) if r.sum() > 0 else math.nan
    if kind == "recovery_in_permeate":
        return float(p[idx] / z[idx])
    return float(r[idx] / z[idx])


def _design_area(pattern: str, z: np.ndarray, q: np.ndarray, g: float, kind: str, idx: int | None, value: float) -> dict:
    """Smallest dimensionless area that meets the target: log scan upwards, then Brent inside the first sign change."""
    a_hi = (float(np.sum(z / q)) / (1.0 - g) * (1.0 - 1e-9)) if pattern == "complete_mixing" else 60.0 / float(q.min())
    grid = np.geomspace(1e-7, a_hi, 41)
    state = {"guess": None}

    def f(a):
        s = _solve(pattern, z, q, g, a, 1e-9, state["guess"])
        state["cut"] = float(s["p"].sum())
        if not s["converged"]:
            return math.nan
        if pattern == "countercurrent":
            state["guess"] = s["r"]
        return _metric(kind, idx, z, s) - value

    prev_a, prev_f, seen = None, None, []
    for a in grid:
        if state.get("cut", 0.0) > 1.0 - 1e-6:          # everything already permeates: larger areas add nothing
            break
        fa = f(a)
        if not math.isfinite(fa):
            continue
        seen.append(fa + value)
        if fa == 0.0:
            return {"a": float(a), "attainable": (min(seen), max(seen))}
        if prev_f is not None and prev_f * fa < 0.0:
            state["guess"] = None
            a_star = brentq(f, prev_a, a, xtol=1e-13 * a, rtol=1e-12, maxiter=200)
            return {"a": float(a_star), "attainable": (min(seen), max(seen))}
        prev_a, prev_f = a, fa
    return {"a": None, "attainable": (min(seen), max(seen)) if seen else (math.nan, math.nan)}


def binary_complete_mixing_closed_form(z_fast: float, alpha: float, pressure_ratio: float, stage_cut: float) -> float:
    """Permeate mole fraction of the fast gas of a binary in complete mixing: positive root of the quadratic that results
    from y/(1-y) = alpha (x - g y)/((1-x) - g (1-y)) with x = (z - theta y)/(1 - theta), g = 1/pressure_ratio
    (Weller-Steiner case I; Geankoplis eq. 13.4-10/11)."""
    g = 1.0 / pressure_ratio
    k = stage_cut + g * (1.0 - stage_cut)
    a1 = k * (1.0 - alpha)
    b1 = (1.0 - stage_cut) * (1.0 - g) - z_fast + alpha * k + alpha * z_fast
    c1 = -alpha * z_fast
    if abs(a1) < 1e-14:
        return -c1 / b1
    return (-b1 + math.sqrt(b1 * b1 - 4.0 * a1 * c1)) / (2.0 * a1)


def gas_permeation(feed_flow_mol_s: float | None = None, feed_composition: dict | None = None,
                   feed_pressure_Pa: float | None = None, permeate_pressure_Pa: float | None = None,
                   flow_pattern: str | None = None, permeance_mol_m2_s_Pa: dict | None = None,
                   permeability_mol_m_m2_s_Pa: dict | None = None, thickness_m: float | None = None,
                   area_m2: float | None = None, stage_cut: float | None = None, target: dict | None = None) -> dict:
    """One membrane module separating a gas mixture by solution-diffusion. See the module docstring for the model.

    Exactly one of `area_m2` (rating), `stage_cut` or `target` = {'quantity': one of TARGETS, 'component', 'value'}
    (design: the area that meets it). Returns flows and compositions of permeate and retentate, stage cut, recoveries,
    area, the pressure-ratio and selectivity limits, a short profile, and the evidence."""
    F = _positive(feed_flow_mol_s, "feed_flow_mol_s", "caudal molar de alimentación del módulo")
    comp = _need(feed_composition, "feed_composition", "fracciones molares de la alimentación {componente: fracción}")
    ph = _positive(feed_pressure_Pa, "feed_pressure_Pa", "presión absoluta del lado de alimentación/retenido")
    pl = _positive(permeate_pressure_Pa, "permeate_pressure_Pa", "presión absoluta del lado de permeado", allow_zero=True)
    if flow_pattern not in PATTERNS:
        raise ValueError(f"Falta flow_pattern o no es válido ({flow_pattern!r}): elige uno de {', '.join(PATTERNS)}. El patrón "
                         "de flujo del módulo cambia el área y la pureza; no se supone.")
    if pl >= ph:
        raise ValueError("permeate_pressure_Pa debe ser menor que feed_pressure_Pa: sin diferencia de presión no hay permeación "
                         "neta.")
    if not isinstance(comp, dict) or len(comp) < 2:
        raise ValueError("feed_composition debe ser un diccionario con al menos dos componentes.")
    names = list(comp)
    zraw = np.array([float(comp[k]) for k in names])
    if np.any(zraw <= 0) or abs(zraw.sum() - 1.0) > 1e-6:
        raise ValueError(f"feed_composition: fracciones molares positivas que sumen 1 (suman {zraw.sum():.6g}); quita los "
                         "componentes ausentes.")
    z = zraw / zraw.sum()
    assumptions = ["Solución-difusión con permeancia constante de cada gas (sin plastificación, sin acoplamiento entre gases, "
                   "sin dependencia con presión ni composición).",
                   "Presión uniforme en cada lado del módulo: no se calcula caída de presión en fibras ni canales.",
                   "Gas ideal (fuerza impulsora = diferencia de presiones parciales), módulo isotérmico, sin polarización de "
                   "concentración en la capa límite del gas."]
    if (permeance_mol_m2_s_Pa is None) == (permeability_mol_m_m2_s_Pa is None):
        raise ValueError("Da las permeancias (permeance_mol_m2_s_Pa) O las permeabilidades con el espesor "
                         "(permeability_mol_m_m2_s_Pa y thickness_m), una sola opción: son datos medidos de tu membrana y no se "
                         "estiman.")
    if permeance_mol_m2_s_Pa is not None:
        src = permeance_mol_m2_s_Pa
        Q = np.array([_positive(src.get(k), f"permeance_mol_m2_s_Pa['{k}']", "permeancia del componente en la membrana")
                      for k in names])
    else:
        t = _positive(thickness_m, "thickness_m", "espesor de la capa selectiva, para pasar de permeabilidad a permeancia")
        src = permeability_mol_m_m2_s_Pa
        Q = np.array([_positive(src.get(k), f"permeability_mol_m_m2_s_Pa['{k}']", "permeabilidad del componente") for k in names]) / t
        assumptions.append("Permeancia = permeabilidad / espesor de la capa selectiva (sin resistencia del soporte poroso).")
    given = [v is not None for v in (area_m2, stage_cut, target)]
    if sum(given) != 1:
        raise ValueError("Da exactamente uno de area_m2 (evaluación de un módulo existente), stage_cut o target (diseño: se "
                         "calcula el área).")
    Qref = float(Q.max())
    q = Q / Qref
    g = pl / ph
    area_scale = F / (Qref * ph)                      # A = a * area_scale
    warnings: list[str] = []
    design = None
    if area_m2 is not None:
        a = _positive(area_m2, "area_m2", "área de membrana instalada") / area_scale
        mode = "rating"
    else:
        mode = "design"
        if stage_cut is not None:
            sc = float(stage_cut)
            if not 0.0 < sc < 1.0:
                raise ValueError("stage_cut debe estar entre 0 y 1 (fracción molar de la alimentación que permea).")
            kind, idx, value = "stage_cut", None, sc
        else:
            kind = target.get("quantity")
            if kind not in TARGETS:
                raise ValueError(f"target['quantity'] debe ser uno de {', '.join(TARGETS)}.")
            if target.get("component") not in names:
                raise ValueError(f"target['component'] debe ser uno de los componentes de la alimentación: {names}.")
            idx = names.index(target["component"])
            value = float(_need(target.get("value"), "target['value']", "valor objetivo de pureza o recuperación (0-1)"))
            if not 0.0 < value < 1.0:
                raise ValueError("target['value'] debe estar entre 0 y 1.")
        design = _design_area(flow_pattern, z, q, g, kind, idx, value)
        if design["a"] is None:
            lo, hi = design["attainable"]
            return {"feasible": False, "mode": mode, "flow_pattern": flow_pattern,
                    "message": (f"El objetivo {kind} = {value:g} no se alcanza con un solo módulo en {flow_pattern}: con estas "
                                f"permeancias y presiones el intervalo alcanzable es {lo:.4g} a {hi:.4g}."),
                    "attainable_range": [lo, hi], "limits": _limits(names, z, q, Q, g),
                    "warnings": ["Objetivo inalcanzable: revisa relación de presiones, selectividad o usa etapas en serie/recirculación "
                                 "(no incluidas)."],
                    "assumptions": assumptions, "not_included": _not_included_gas(),
                    "method": "Barrido logarítmico del área adimensional con el modelo del patrón pedido (scipy).",
                    "evidence": {"verified": False, "reasons": ["el objetivo no tiene solución en el intervalo de áreas barrido"]}}
        a = design["a"]
    tight = _solve(flow_pattern, z, q, g, a, RTOL_TIGHT)
    loose = _solve(flow_pattern, z, q, g, a, RTOL_LOOSE, tight.get("r_out"))
    r, p = tight["r"], tight["p"]
    theta = float(p.sum())
    total = theta >= TOTAL_PERMEATION or tight.get("total_permeation", False)
    y = p / p.sum() if p.sum() > 0 else _local_permeate(z, q, g)[0]
    x = r / r.sum() if r.sum() > 1e-300 else np.full_like(z, math.nan)
    if total:
        warnings.append("Con esta área permea prácticamente toda la alimentación: no hay separación (el permeado tiene la "
                        "composición de la alimentación).")
    balance = float(np.max(np.abs(z - r - tight["p_flux"]) / z))
    yl = loose["p"] / loose["p"].sum() if loose["p"].sum() > 0 else y
    step = max(float(np.max(np.abs(yl - y))), abs(float(loose["p"].sum()) - theta))
    evidence = {"component_balance_max_rel_error": balance, "integration_tolerance_change_max_abs": step,
                "converged": bool(tight["converged"]), "boundary_or_equation_residual": tight["residual"]}
    reasons = []
    if not tight["converged"]:
        reasons.append("el sistema de ecuaciones del patrón de flujo no convergió (disparo desde el retenido o integración)")
    if balance > BALANCE_TOL:
        reasons.append(f"el balance por componente (permeado integrado aparte) no cierra: {balance:.2e} relativo")
    if step > STEP_TOL:
        reasons.append(f"el resultado cambia {step:.2e} al relajar la tolerancia de integración")
    if len(names) == 2 and flow_pattern == "complete_mixing" and not total and theta > 0:
        i = int(np.argmax(q))
        alpha = float(q[i] / q[1 - i])
        if alpha > 1.0 + 1e-12 and g > 0:
            y_cf = binary_complete_mixing_closed_form(float(z[i]), alpha, 1.0 / g, theta)
            evidence["binary_closed_form_quadratic_abs_diff"] = abs(y_cf - float(y[i]))
            if evidence["binary_closed_form_quadratic_abs_diff"] > 1e-8:
                reasons.append("la composición del permeado no coincide con la raíz de la cuadrática cerrada del binario")
    evidence["verified"] = not reasons
    if reasons:
        evidence["reasons"] = reasons
    area = a * area_scale
    out = {"feasible": True, "mode": mode, "flow_pattern": flow_pattern, "area_m2": area, "stage_cut": theta,
           "pressure_ratio": ph / pl if pl > 0 else "inf",
           "permeate": {"flow_mol_s": theta * F, "mole_fractions": dict(zip(names, map(float, y))), "pressure_Pa": pl,
                        "component_flows_mol_s": dict(zip(names, map(float, p * F)))},
           "retentate": {"flow_mol_s": float(r.sum()) * F,
                         "mole_fractions": {k: (float(v) if math.isfinite(v) else None) for k, v in zip(names, x)},
                         "pressure_Pa": ph, "component_flows_mol_s": dict(zip(names, map(float, r * F)))},
           "recovery_in_permeate": dict(zip(names, map(float, p / z))),
           "recovery_in_retentate": dict(zip(names, map(float, r / z))),
           "permeance_mol_m2_s_Pa": dict(zip(names, map(float, Q))),
           "limits": _limits(names, z, q, Q, g)}
    if tight.get("grid") is not None:
        rp = np.clip(tight["r_profile"], 0.0, None)
        tot = rp.sum(axis=0)
        order = np.argsort(tight["grid"])
        xs = np.where(tot > 0, rp / np.where(tot > 0, tot, 1.0), np.nan)
        out["profile"] = {"area_m2": [float(v * area_scale) for v in np.asarray(tight["grid"])[order]],
                          "retentate_side_flow_mol_s": [float(v * F) for v in tot[order]],
                          "retentate_side_mole_fractions": {k: [None if not math.isfinite(v) else float(v) for v in xs[i][order]]
                                                            for i, k in enumerate(names)},
                          "note": "área medida desde la entrada de la alimentación"}
    lim = out["limits"]
    fast = lim["most_permeable"]
    if ph / max(pl, 1e-300) < lim["ideal_selectivity_vs_slowest"][fast]:
        warnings.append(f"Módulo limitado por la relación de presiones ({ph / pl:.3g}) más que por la selectividad "
                        f"({lim['ideal_selectivity_vs_slowest'][fast]:.3g}): más selectividad casi no mejora la pureza del permeado.")
    out.update(warnings=warnings, assumptions=assumptions, not_included=_not_included_gas(),
               method=_method_gas(flow_pattern), evidence=evidence)
    return out


def _limits(names: list, z: np.ndarray, q: np.ndarray, Q: np.ndarray, g: float) -> dict:
    slow = int(np.argmin(q))
    fast = int(np.argmax(q))
    y0, _ = _local_permeate(z, q, g)
    y_inf = q * z / float(np.dot(q, z))
    # retentate of complete mixing when everything permeates (y -> z): x_i = z_i (lam + q_i g) / q_i, sum x = 1
    x_min = z * ((1.0 - g) / float(np.sum(z / q)) + q * g) / q
    return {"most_permeable": names[fast], "least_permeable": names[slow],
            "ideal_selectivity_vs_slowest": {k: float(Q[i] / Q[slow]) for i, k in enumerate(names)},
            "permeate_at_zero_stage_cut": dict(zip(names, map(float, y0))),
            "permeate_at_infinite_pressure_ratio_zero_stage_cut": dict(zip(names, map(float, y_inf))),
            "permeate_ceiling_by_pressure_ratio": ({k: float(min(1.0, zi / g)) for k, zi in zip(names, z)} if g > 0 else
                                                   {k: 1.0 for k in names}),
            "complete_mixing_retentate_at_total_permeation": dict(zip(names, map(float, x_min))),
            "note": ("permeate_at_zero_stage_cut es el permeado más rico posible en el gas más permeable para un módulo sin "
                     "recirculación; el techo por relación de presiones es y_i <= x_i p_alta/p_baja.")}


def _not_included_gas() -> list[str]:
    return ["Caída de presión en el lado de alimentación y en el de permeado (fibras huecas, espiral).",
            "Dependencia de la permeancia con presión, composición o temperatura; plastificación por CO2 o hidrocarburos.",
            "Efecto Joule-Thomson y no idealidad del gas (fugacidades).",
            "Gas de arrastre en el permeado, recirculación de permeado y cascadas de dos o más etapas: no están validadas "
            "aquí; encadena llamadas de un módulo si las necesitas y cierra tú el reciclo.",
            "Compresión, costo de membrana y vida útil."]


def _method_gas(pattern: str) -> str:
    how = {"complete_mixing": "sistema algebraico (suma de fracciones del permeado = 1) resuelto con scipy.optimize.brentq",
           "cross_flow": "ecuaciones diferenciales de Weller-Steiner integradas en el área con scipy.integrate.solve_ivp (LSODA); "
                         "composición local del permeado por Newton",
           "cocurrent": "balances diferenciales de ambos lados integrados en el área con scipy.integrate.solve_ivp (LSODA)",
           "countercurrent": "problema de valores en la frontera resuelto por disparo desde el extremo del retenido "
                             "(scipy.optimize.root hybr sobre solve_ivp LSODA)"}[pattern]
    return f"Solución-difusión multicomponente, patrón {pattern}: {how}. Diseño: barrido y Brent sobre el área."


# ---------------------------------------------------------------------------------------------------------------
# Reverse osmosis
# ---------------------------------------------------------------------------------------------------------------
def osmotic_pressure_Pa(concentration_kg_m3: float, molar_mass_kg_mol: float, ions_per_formula: float, T_K: float,
                        osmotic_coefficient: float = 1.0) -> float:
    """pi = phi nu c R T / M (van 't Hoff for phi = 1)."""
    return osmotic_coefficient * ions_per_formula * concentration_kg_m3 / molar_mass_kg_mol * R_GAS * T_K


def _ro_local(cb: float, dP: float, Aw: float, As: float, rho_p: float, pi_per_c: float, k: float | None) -> dict:
    """Fluxes at a point of the membrane where the bulk feed-side concentration is cb (kg/m3)."""
    def parts(Jv):
        E = math.exp(min(Jv / k, 700.0)) if k else 1.0
        if As == 0.0:                                   # complete rejection
            return 0.0, cb * E
        cp = As * cb * E / (Jv + As * E)
        cm = cb * E * (Jv + As) / (Jv + As * E)
        return cp, cm

    def f(Jv):
        cp, cm = parts(Jv)
        return Jv * rho_p - Aw * (dP - pi_per_c * (cm - cp))

    hi = Aw * dP / rho_p
    if f(0.0) >= 0.0:                                   # applied pressure does not exceed the osmotic difference
        Jv = 0.0
    else:
        Jv = brentq(f, 0.0, hi, xtol=1e-18, rtol=1e-14, maxiter=300) if f(hi) > 0 else hi
    cp, cm = parts(Jv)
    return {"Jv": Jv, "Nw": Jv * rho_p, "Ns": Jv * cp, "cp": cp, "cm": cm, "residual": abs(f(Jv)) / (Aw * dP)}


def reverse_osmosis(feed_concentration_kg_m3: float | None = None, solute_molar_mass_kg_mol: float | None = None,
                    ions_per_formula: float | None = None, T_K: float | None = None, feed_pressure_Pa: float | None = None,
                    permeate_pressure_Pa: float | None = None, water_permeability_kg_m2_s_Pa: float | None = None,
                    solute_permeability_m_s: float | None = None, feed_flow_m3_s: float | None = None,
                    area_m2: float | None = None, recovery: float | None = None, osmotic_coefficient: float | None = None,
                    mass_transfer_coefficient_m_s: float | None = None, permeate_density_kg_m3: float | None = None,
                    pump_efficiency: float | None = None, energy_recovery_efficiency: float | None = None) -> dict:
    """Reverse osmosis of one dissolved solute. Without `area_m2` and `recovery`: local fluxes and rejection at feed
    conditions (a differential element, recovery -> 0). With `feed_flow_m3_s` and `area_m2` (rating) or `recovery`
    (design): plug-flow feed channel integrated over the membrane area."""
    cf = _positive(feed_concentration_kg_m3, "feed_concentration_kg_m3", "concentración de soluto en la alimentación")
    M = _positive(solute_molar_mass_kg_mol, "solute_molar_mass_kg_mol", "masa molar del soluto, para la presión osmótica")
    nu = _positive(ions_per_formula, "ions_per_formula", "número de iones por fórmula del soluto (2 para NaCl, 1 si no se disocia), "
                   "para la presión osmótica de van 't Hoff")
    T = _positive(T_K, "T_K", "temperatura de la solución, para la presión osmótica")
    pf = _positive(feed_pressure_Pa, "feed_pressure_Pa", "presión absoluta del lado de alimentación")
    pp = _positive(permeate_pressure_Pa, "permeate_pressure_Pa", "presión absoluta del permeado")
    Aw = _positive(water_permeability_kg_m2_s_Pa, "water_permeability_kg_m2_s_Pa", "constante de permeabilidad al agua A_w de tu "
                   "membrana (kg de agua por m2, s y Pa), medida")
    As = _positive(solute_permeability_m_s, "solute_permeability_m_s", "constante de permeabilidad al soluto A_s de tu membrana "
                   "(m/s), medida; 0 si el rechazo es total", allow_zero=True)
    dP = pf - pp
    if dP <= 0:
        raise ValueError("feed_pressure_Pa debe ser mayor que permeate_pressure_Pa.")
    assumptions = ["Solución-difusión: flujo de agua proporcional a (ΔP − Δπ) y flujo de soluto proporcional a la diferencia de "
                   "concentración, con A_w y A_s constantes (sin compactación ni dependencia con concentración o temperatura).",
                   "El flujo volumétrico de permeado se toma como N_w/ρ del permeado (permeado diluido), como en Geankoplis."]
    if osmotic_coefficient is None:
        phi = 1.0
        assumptions.append("Presión osmótica de van 't Hoff ideal (coeficiente osmótico 1): π = ν c R T / M. Sobrestima π de "
                           "electrolitos reales unos puntos porcentuales; da osmotic_coefficient si lo tienes.")
        osm = "van 't Hoff ideal (coeficiente osmótico = 1)"
    else:
        phi = _positive(osmotic_coefficient, "osmotic_coefficient", "coeficiente osmótico de la solución")
        assumptions.append(f"Presión osmótica π = φ ν c R T / M con el coeficiente osmótico del usuario φ = {phi:g}, constante "
                           "en todo el intervalo de concentración.")
        osm = f"van 't Hoff corregida con el coeficiente osmótico del usuario ({phi:g})"
    if permeate_density_kg_m3 is None:
        from CoolProp.CoolProp import PropsSI
        rho_p = float(PropsSI("D", "T", T, "P", pp, "Water"))
        assumptions.append(f"Densidad del permeado = agua pura a T y presión del permeado por IAPWS-95 (CoolProp): {rho_p:.2f} kg/m3.")
    else:
        rho_p = _positive(permeate_density_kg_m3, "permeate_density_kg_m3", "densidad del permeado")
    k = None
    if mass_transfer_coefficient_m_s is not None:
        k = _positive(mass_transfer_coefficient_m_s, "mass_transfer_coefficient_m_s", "coeficiente de transferencia de masa del "
                      "lado de alimentación, para la polarización de concentración")
        assumptions.append("Polarización de concentración por el modelo de película: c_m = c_p + (c_b − c_p) exp(J_v/k), con k "
                           "del usuario, constante a lo largo del módulo.")
    else:
        assumptions.append("Sin polarización de concentración (no se dio mass_transfer_coefficient_m_s): la concentración en la "
                           "pared de la membrana se toma igual a la del seno. Optimista si el flujo es alto.")
    pi_per_c = phi * nu * R_GAS * T / M
    pi_f = pi_per_c * cf
    warnings: list[str] = []
    if dP <= pi_f:
        warnings.append(f"La diferencia de presión ({dP:.4g} Pa) no supera la presión osmótica de la alimentación ({pi_f:.4g} Pa): "
                        "el flujo de agua es marginal y el rechazo pobre.")
    not_included = ["Caída de presión a lo largo del canal de alimentación y del permeado.",
                    "Varios solutos, ensuciamiento, incrustación y compactación de la membrana.",
                    "Efecto de la temperatura sobre A_w y A_s; arreglo de etapas y pasos (aquí es un solo banco).",
                    "Ultrafiltración (capa de gel, presión osmótica de macromoléculas): no está incluida."]
    method = (f"Solución-difusión con presión osmótica {osm}; flujo local por scipy.optimize.brentq"
              + ("" if area_m2 is None and recovery is None else "; canal de alimentación en flujo pistón integrado en el área con "
                 "scipy.integrate.solve_ivp (LSODA)") + ".")

    def point(cb):
        return _ro_local(cb, dP, Aw, As, rho_p, pi_per_c, k)

    loc = point(cf)
    inlet = {"water_flux_kg_m2_s": loc["Nw"], "volumetric_flux_m_s": loc["Jv"], "solute_flux_kg_m2_s": loc["Ns"],
             "permeate_concentration_kg_m3": loc["cp"], "wall_concentration_kg_m3": loc["cm"],
             "polarization_modulus": loc["cm"] / cf, "rejection_observed": 1.0 - loc["cp"] / cf,
             "rejection_intrinsic": 1.0 - loc["cp"] / loc["cm"],
             "osmotic_pressure_feed_Pa": pi_f, "osmotic_pressure_wall_Pa": pi_per_c * loc["cm"],
             "osmotic_pressure_permeate_Pa": pi_per_c * loc["cp"],
             "net_driving_pressure_Pa": dP - pi_per_c * (loc["cm"] - loc["cp"])}
    base = {"osmotic_pressure_model": osm, "transmembrane_pressure_Pa": dP, "permeate_density_kg_m3": rho_p}
    if area_m2 is None and recovery is None:
        # closed form of the same model without polarisation: R = B (dP - dpi)/(1 + B (dP - dpi)), B = A_w/(A_s rho_p)
        ev = {"flux_equation_residual_rel": loc["residual"]}
        reasons = []
        if As > 0 and k is None:
            B = Aw / (As * rho_p)
            ndp = inlet["net_driving_pressure_Pa"]
            ev["rejection_closed_form_abs_diff"] = abs(B * ndp / (1.0 + B * ndp) - inlet["rejection_observed"])
            if ev["rejection_closed_form_abs_diff"] > 1e-9:
                reasons.append("el rechazo no coincide con la forma cerrada R = B(ΔP−Δπ)/(1+B(ΔP−Δπ))")
        if k is not None:
            ev["film_model_identity_rel_residual"] = abs((loc["cm"] - loc["cp"]) - (cf - loc["cp"]) * math.exp(loc["Jv"] / k)) / cf
        if loc["residual"] > 1e-9:
            reasons.append("la ecuación local de flujo no convergió")
        ev["verified"] = not reasons
        if reasons:
            ev["reasons"] = reasons
        return {"mode": "local", **base, "local_at_feed": inlet,
                "warnings": warnings + ["Resultado local (recuperación → 0): da feed_flow_m3_s y area_m2 o recovery para un módulo "
                                        "con recuperación finita."],
                "assumptions": assumptions, "not_included": not_included, "method": method, "evidence": ev}
    Qf = _positive(feed_flow_m3_s, "feed_flow_m3_s", "caudal volumétrico de alimentación, para el balance del módulo")
    if (area_m2 is None) == (recovery is None):
        raise ValueError("Da area_m2 (evaluación) O recovery (diseño: se calcula el área), no ambos.")
    assumptions.append("Canal de alimentación en flujo pistón, permeado recogido y mezclado a presión uniforme; densidad del "
                       "líquido constante (volúmenes aditivos).")

    def rhs(_A, s):
        Q, ms = max(s[0], 1e-300), max(s[1], 0.0)
        p_ = point(ms / Q)
        return [-p_["Jv"], -p_["Ns"], p_["Jv"], p_["Ns"]]

    s0 = [Qf, Qf * cf, 0.0, 0.0]
    if area_m2 is not None:
        A = _positive(area_m2, "area_m2", "área de membrana instalada")
        mode = "rating"
    else:
        mode = "design"
        rec = float(recovery)
        if not 0.0 < rec < 1.0:
            raise ValueError("recovery debe estar entre 0 y 1 (permeado / alimentación, en volumen).")
        ev_hit = lambda _A, s: s[0] - Qf * (1.0 - rec)          # noqa: E731
        ev_hit.terminal, ev_hit.direction = True, -1
        A_max = 1e4 * Qf * rec / loc["Jv"]
        sol = solve_ivp(rhs, (0.0, A_max), s0, method="LSODA", rtol=RTOL_TIGHT, atol=[1e-14 * Qf, 1e-14 * Qf * cf] * 2,
                        events=ev_hit)
        if not len(sol.t_events[0]):
            reached = 1.0 - sol.y[0, -1] / Qf
            if As == 0.0:      # complete rejection: the concentrate reaches pi = dP at recovery 1 - pi_f/dP (pi linear in c)
                base["max_recovery_osmotic_limit_complete_rejection"] = 1.0 - pi_f / dP
            return {"feasible": False, "mode": mode, **base, "local_at_feed": inlet,
                    "message": (f"La recuperación {rec:g} no se alcanza: con {A_max:.3g} m2 (10 000 veces el área que daría el "
                                f"flujo de entrada) solo se llega a {reached:.4g}; la presión osmótica del concentrado iguala a la "
                                "presión aplicada."),
                    "max_recovery_reached": reached, "warnings": warnings, "assumptions": assumptions,
                    "not_included": not_included, "method": method,
                    "evidence": {"verified": False, "reasons": ["la recuperación pedida está fuera del límite osmótico"]}}
        A = float(sol.t_events[0][0])
    atol = [1e-14 * Qf, 1e-14 * Qf * cf] * 2
    grid = np.linspace(0.0, A, N_PROFILE)
    tight = solve_ivp(rhs, (0.0, A), s0, method="LSODA", rtol=RTOL_TIGHT, atol=atol, t_eval=grid)
    loose = solve_ivp(rhs, (0.0, A), s0, method="LSODA", rtol=RTOL_LOOSE, atol=atol)
    Qr, msr, Qp_int, msp_int = (float(v) for v in tight.y[:, -1])
    Qp, msp = Qf - Qr, Qf * cf - msr
    rec = Qp / Qf
    cp_avg = msp / Qp if Qp > 0 else loc["cp"]
    cr = msr / Qr
    out_loc = point(cr)
    bal_w = abs(Qp - Qp_int) / Qf
    bal_s = abs(msp - msp_int) / (Qf * cf)
    step = max(abs(float(loose.y[0, -1]) - Qr) / Qf, abs(float(loose.y[1, -1]) - msr) / (Qf * cf))
    ev = {"water_balance_rel_error": bal_w, "solute_balance_rel_error": bal_s,
          "integration_tolerance_change_max_rel": step, "integrator_success": bool(tight.success)}
    reasons = []
    if not tight.success:
        reasons.append("la integración a lo largo del módulo no terminó")
    if max(bal_w, bal_s) > BALANCE_TOL:
        reasons.append("el balance de agua o de soluto (permeado integrado aparte) no cierra")
    if step > STEP_TOL:
        reasons.append("el resultado cambia al relajar la tolerancia de integración")
    if As == 0.0 and k is None:
        u_in, u_out = dP * Qf - pi_f * Qf, dP * Qr - pi_f * Qf
        if u_out > 0:
            A_cf = rho_p / Aw * (dP * (Qf - Qr) + pi_f * Qf * math.log(u_in / u_out)) / dP ** 2
            ev["area_closed_form_complete_rejection_rel_diff"] = abs(A_cf - A) / A
            if ev["area_closed_form_complete_rejection_rel_diff"] > 1e-6:
                reasons.append("el área no coincide con la integral cerrada del caso de rechazo total")
    ev["verified"] = not reasons
    if reasons:
        ev["reasons"] = reasons
    sec_h = dP / rec                                     # J per m3 of permeate, ideal pump, no energy recovery
    energy = {"hydraulic_ideal_pump_J_m3": sec_h, "hydraulic_ideal_pump_kWh_m3": sec_h / 3.6e6,
              "minimum_reversible_work_J_m3": pi_f * math.log(1.0 / (1.0 - rec)) / rec,
              "basis": ("por m3 de permeado; la bomba eleva toda la alimentación desde la presión del permeado hasta la de "
                        "alimentación; el mínimo reversible supone rechazo total y π lineal en la concentración")}
    if pump_efficiency is not None:
        eta = _positive(pump_efficiency, "pump_efficiency", "eficiencia global de la bomba de alta presión")
        if eta > 1:
            raise ValueError("pump_efficiency debe estar entre 0 y 1.")
        sec = dP / (eta * rec)
        if energy_recovery_efficiency is not None:
            erd = _positive(energy_recovery_efficiency, "energy_recovery_efficiency", "eficiencia del recuperador de energía del "
                            "concentrado", allow_zero=True)
            if erd > 1:
                raise ValueError("energy_recovery_efficiency debe estar entre 0 y 1.")
            sec -= erd * dP * (1.0 - rec) / rec
            energy["energy_recovery_efficiency"] = erd
        energy.update(specific_energy_J_m3=sec, specific_energy_kWh_m3=sec / 3.6e6, pump_efficiency=eta)
    else:
        warnings.append("Sin pump_efficiency solo se da la energía hidráulica ideal (bomba 100 %, sin recuperación de energía).")
    if 1.0 - out_loc["cp"] / cr < 0.9 * inlet["rejection_observed"]:
        warnings.append(f"El rechazo local cae de {inlet['rejection_observed']:.3f} a la entrada a {1.0 - out_loc['cp'] / cr:.3f} a la "
                        "salida: el flujo neto de agua se acerca a cero y el soluto pasa; la recuperación pedida está cerca del "
                        "límite osmótico.")
    if out_loc["cm"] / cr > 1.2 or loc["cm"] / cf > 1.2:
        warnings.append("Módulo de polarización mayor que 1.2: el flujo está limitado por la capa límite; revisa velocidad de "
                        "flujo cruzado o el coeficiente k.")
    return {"feasible": True, "mode": mode, **base, "area_m2": A, "recovery": rec,
            "permeate": {"flow_m3_s": Qp, "concentration_kg_m3": cp_avg}, "retentate": {"flow_m3_s": Qr, "concentration_kg_m3": cr},
            "rejection_observed": 1.0 - cp_avg / cf, "average_water_flux_kg_m2_s": Qp * rho_p / A,
            "average_volumetric_flux_m_s": Qp / A, "local_at_feed": inlet,
            "local_at_outlet": {"volumetric_flux_m_s": out_loc["Jv"], "permeate_concentration_kg_m3": out_loc["cp"],
                                "wall_concentration_kg_m3": out_loc["cm"], "polarization_modulus": out_loc["cm"] / cr,
                                "osmotic_pressure_bulk_Pa": pi_per_c * cr,
                                "net_driving_pressure_Pa": dP - pi_per_c * (out_loc["cm"] - out_loc["cp"])},
            "specific_energy": energy,
            "profile": {"area_m2": [float(v) for v in tight.t], "feed_side_flow_m3_s": [float(v) for v in tight.y[0]],
                        "feed_side_concentration_kg_m3": [float(m / max(q_, 1e-300)) for q_, m in zip(tight.y[0], tight.y[1])]},
            "warnings": warnings, "assumptions": assumptions, "not_included": not_included, "method": method, "evidence": ev}
