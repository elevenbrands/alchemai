"""Adsorption in packed beds: isotherms, breakthrough of an isothermal fixed bed, scale-up by the length of unused bed,
and a four-step pressure-swing (Skarstrom) cycle of a binary gas on one bed.

Isotherms (parameters always from the user; any consistent amount unit, mol is assumed in the key names):
    linear       q* = K c                        K in m3/kg
    langmuir     q* = q_max b c / (1 + b c)      q_max in mol/kg, b in m3/mol; with several Langmuir components the
                                                 denominator becomes 1 + sum_j b_j c_j (extended Langmuir)
    freundlich   q* = K_F c**(1/n)               K_F in (mol/kg)/(mol/m3)**(1/n)

Fixed bed (`fixed_bed_breakthrough`), per component, constant superficial velocity u (dilute adsorbate or liquid):
    eps dc/dt + u dc/dz = eps D_ax d2c/dz2 - rho_b dq/dt        dq/dt = k_LDF (q*(c) - q)
finite volumes in z with a van Leer slope limiter, Danckwerts boundaries, time by scipy's LSODA with a banded Jacobian.
Checks carried in the result: component balance, the stoichiometric time against its closed form, the same run on half
the cells, and for a single linear isotherm without dispersion the exact Anzelius/Thomas solution
    c/c0 = 1 - integral_0^xi exp(-(tau + s)) I0(2 sqrt(tau s)) ds,   xi = k K rho_b L / u,   tau = k (t - eps L / u)
(Klinkenberg's error-function approximation of it is `klinkenberg`).

Scale-up (`lub_scaleup`): the user's laboratory breakthrough curve gives t_b, t_t = integral (1 - c/c0) dt, t_u (the same
integral up to t_b) and the length of unused bed LUB = L (1 - t_u/t_t), which is carried unchanged to another bed length at
the same superficial velocity (Geankoplis sec. 12.3D; McCabe-Smith-Harriott chap. 25).

PSA (`psa_skarstrom`): pressurisation (with feed or with product), high-pressure feed, countercurrent blowdown and
countercurrent purge with product, repeated until the cyclic steady state. Bulk binary: the velocity along the bed
follows from the total balance at uniform pressure,
    d(uC)/dz = -eps dC/dt - rho_b sum_i dq_i/dt ,   C = P(t)/(R T)
with LDF uptake and linear pressure ramps. The result is set against the equilibrium theory (method of characteristics,
no dispersion, instantaneous uptake): for linear isotherms the binary analysis of Shendalman & Mitchell 1972 and
Knaebel & Hill 1985, re-derived in `psa_equilibrium_theory`; for Langmuir / extended-Langmuir isotherms the same cycle
with pressurisation by product, re-derived here from the shock (Rankine-Hugoniot) conditions of the feed step and the
slowest (c_A -> 0) characteristic of the purge in `psa_equilibrium_theory_nonlinear`."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import erf, ive

R_GAS = 8.314462618            # J/(mol K)
ISOTHERMS = ("linear", "langmuir", "freundlich")
BALANCE_TOL = 1e-5             # component balance closure of a run, relative to what was fed
GRID_TOL = 0.02                # change of the breakthrough time (over the stoichiometric time) on half the cells
ANALYTIC_TOL = 0.02            # max |c/c0 - exact| against the Anzelius solution
PSA_GRID_TOL = 0.02            # change of purity / recovery (absolute) on half the cells
BED_ATOL_REL = 1e-8             # absolute tolerance of the bed integration, relative to the largest feed concentration
FREUNDLICH_LINEAR_BELOW = 1e-4  # Freundlich made linear below this fraction of the feed concentration (numerical only)


def _need(value, name: str, what_for: str):
    if value is None:
        raise ValueError(f"Falta {name}: {what_for}. No se supone ningún valor.")
    return value


def _positive(value, name: str, what_for: str, allow_zero: bool = False) -> float:
    v = float(_need(value, name, what_for))
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} debe ser {'no negativo' if allow_zero else 'positivo'} (recibido {value!r}).")
    return v


# ---------------------------------------------------------------------------------------------------------------
# Isotherms
# ---------------------------------------------------------------------------------------------------------------
def _parse_isotherms(components: list, where: str, c_ref: np.ndarray | None = None) -> dict:
    """Validated parameters and a vectorised q*(c) for c of shape (n_components, n_cells). A Freundlich isotherm with n > 1
    has an infinite slope at c = 0, which makes the clean bed ahead of the front unboundedly stiff; below
    FREUNDLICH_LINEAR_BELOW * c_ref it is continued by the chord to the origin (declared in the notes)."""
    nc = len(components)
    models, par = [], []
    for i, comp in enumerate(components):
        iso = _need(comp.get("isotherm"), f"{where}[{i}]['isotherm']", "isoterma de adsorción del componente, medida")
        m = iso.get("model")
        if m not in ISOTHERMS:
            raise ValueError(f"{where}[{i}]['isotherm']['model'] debe ser uno de {', '.join(ISOTHERMS)} (recibido {m!r}).")
        if m == "linear":
            par.append((_positive(iso.get("K_m3_kg"), "isotherm['K_m3_kg']", "constante de Henry de la isoterma lineal q = K c",
                                  allow_zero=True),))
        elif m == "langmuir":
            par.append((_positive(iso.get("q_max_mol_kg"), "isotherm['q_max_mol_kg']", "capacidad de monocapa de Langmuir"),
                        _positive(iso.get("b_m3_mol"), "isotherm['b_m3_mol']", "constante de afinidad de Langmuir")))
        else:
            par.append((_positive(iso.get("K_F"), "isotherm['K_F']", "constante de Freundlich en q = K_F c^(1/n)"),
                        _positive(iso.get("n"), "isotherm['n']", "exponente n de Freundlich en q = K_F c^(1/n)")))
        models.append(m)
    lang = [i for i, m in enumerate(models) if m == "langmuir"]
    b = np.zeros(nc)
    for i in lang:
        b[i] = par[i][1]
    notes = []
    if len(lang) > 1:
        notes.append("Langmuir extendido para los componentes con isoterma de Langmuir: compiten por los mismos sitios "
                     "(denominador 1 + Σ b_j c_j); es termodinámicamente consistente solo si sus q_max son iguales.")
    c_lin = np.zeros(nc)
    for i, m in enumerate(models):
        if m == "freundlich" and par[i][1] > 1.0 and c_ref is not None:
            c_lin[i] = FREUNDLICH_LINEAR_BELOW * float(c_ref[i])
    if np.any(c_lin > 0):
        notes.append(f"Freundlich con n > 1 tiene pendiente infinita en c = 0: por debajo de {FREUNDLICH_LINEAR_BELOW:g} veces la "
                     "concentración de alimentación se sustituye por la recta al origen (ajuste numérico; la carga a esas "
                     "concentraciones es despreciable).")
    if nc > 1 and len(lang) < nc:
        notes.append("Los componentes con isoterma lineal o de Freundlich se adsorben de forma independiente (sin competencia "
                     "con los demás).")

    def qstar(c: np.ndarray) -> np.ndarray:
        c = np.clip(c, 0.0, None)
        out = np.empty_like(c)
        den = 1.0 + np.tensordot(b, c, axes=(0, 0)) if lang else None
        for i, m in enumerate(models):
            if m == "linear":
                out[i] = par[i][0] * c[i]
            elif m == "langmuir":
                out[i] = par[i][0] * par[i][1] * c[i] / den
            elif c_lin[i] > 0.0:
                ci = c[i]
                out[i] = np.where(ci >= c_lin[i], par[i][0] * np.maximum(ci, c_lin[i]) ** (1.0 / par[i][1]),
                                  par[i][0] * c_lin[i] ** (1.0 / par[i][1]) * ci / c_lin[i])
            else:
                out[i] = par[i][0] * c[i] ** (1.0 / par[i][1])
        return out

    return {"models": models, "par": par, "qstar": qstar, "notes": notes}


def _harmonic(dm: np.ndarray, dp: np.ndarray) -> np.ndarray:
    prod = dm * dp
    return np.where(prod > 0.0, 2.0 * prod / np.where(prod > 0.0, dm + dp, 1.0), 0.0)


def _outlet_slope(c_last: np.ndarray, c_prev: np.ndarray) -> np.ndarray:
    """Slope of the last cell from a ghost cell that continues the profile linearly but never below zero, so the outlet
    face c_last + slope/2 is second order on smooth profiles and stays non-negative ahead of a steep front."""
    dm = c_last - c_prev
    return _harmonic(dm, np.maximum(2.0 * c_last - c_prev, 0.0) - c_last)


def _limited_slopes(c: np.ndarray, outlet: bool = True) -> np.ndarray:
    """van Leer (harmonic) slope per cell along the last axis; first cell first order, last cell by `_outlet_slope` (or
    first order too when `outlet` is False)."""
    d = np.diff(c, axis=-1)
    zero = np.zeros(c.shape[:-1] + (1,))
    s = _harmonic(np.concatenate([zero, d], axis=-1), np.concatenate([d, zero], axis=-1))
    if outlet and c.shape[-1] > 1:
        s[..., -1] = _outlet_slope(c[..., -1], c[..., -2])
    return s


# ---------------------------------------------------------------------------------------------------------------
# Analytical solutions used as checks
# ---------------------------------------------------------------------------------------------------------------
def anzelius_exact(xi: float, tau: float) -> float:
    """Exact c/c0 at the outlet of a clean bed with a linear isotherm, LDF uptake and plug flow (Anzelius 1926; Thomas
    1944; the J function of Hiester & Vermeulen): 1 - int_0^xi exp(-(sqrt(tau) - sqrt(s))^2) ive(0, 2 sqrt(tau s)) ds."""
    if tau <= 0.0:
        return 0.0
    if xi <= 0.0:
        return 1.0
    f = lambda s: math.exp(-(math.sqrt(tau) - math.sqrt(s)) ** 2) * ive(0, 2.0 * math.sqrt(tau * s))   # noqa: E731
    pts = [p for p in (tau,) if 0.0 < p < xi]
    val, _ = quad(f, 0.0, xi, points=pts or None, limit=400, epsabs=1e-12, epsrel=1e-12)
    return min(1.0, max(0.0, 1.0 - val))


def klinkenberg(xi: float, tau: float) -> float:
    """Klinkenberg's (1954) approximation of the same solution: 0.5 [1 + erf(sqrt(tau) - sqrt(xi) + 1/(8 sqrt(tau)) +
    1/(8 sqrt(xi)))], stated accurate to < 0.6 % for xi > 2."""
    if tau <= 0.0:
        return 0.0
    return 0.5 * (1.0 + float(erf(math.sqrt(tau) - math.sqrt(xi) + 0.125 / math.sqrt(tau) + 0.125 / math.sqrt(xi))))


def constant_pattern_langmuir_ldf_time(x1: float, x2: float, k_ldf_1_s: float, separation_factor: float) -> float:
    """Time for the outlet to go from c/c0 = x1 to x2 once the front of a Langmuir isotherm has reached its constant
    pattern (plug flow, LDF): k (1 - R)(t2 - t1) = R ln(x2/x1) + ln((1 - x1)/(1 - x2)), R = 1/(1 + b c0). It follows from
    q/q0 = c/c0 along the front and dq/dt = k (q* - q)."""
    r = separation_factor
    return (r * math.log(x2 / x1) + math.log((1.0 - x1) / (1.0 - x2))) / (k_ldf_1_s * (1.0 - r))


# ---------------------------------------------------------------------------------------------------------------
# Fixed bed
# ---------------------------------------------------------------------------------------------------------------
def _bed_run(L: float, u: float, eps: float, rho_b: float, D: float, c0: np.ndarray, k: np.ndarray, qstar, n_cells: int,
             t_chunks: list, rtol: float, extend_until_saturated: int = 0) -> dict:
    """Integrates over the time marks `t_chunks`; with `extend_until_saturated` = n it keeps adding marks of the same
    spacing (at most n in total) until every outlet has reached the feed concentration."""
    nc = len(c0)
    N = n_cells
    dz = L / N
    w = 2 * nc

    def rhs(_t, S):
        cell = S[:N * w].reshape(N, w)
        c = cell[:, :nc].T
        q = cell[:, nc:].T
        dq = k[:, None] * (qstar(c) - q)
        face = c + 0.5 * _limited_slopes(c)                     # value at the downstream face of each cell
        F = np.empty((nc, N + 1))
        F[:, 0] = u * c0
        F[:, 1:] = u * face
        if D > 0.0:
            F[:, 1:N] -= eps * D * np.diff(c, axis=1) / dz
        dc = (-(F[:, 1:] - F[:, :-1]) / dz - rho_b * dq) / eps
        out = np.empty(N * w + nc)
        out[:N * w] = np.concatenate([dc.T, dq.T], axis=1).ravel()
        out[N * w:] = F[:, N]
        return out

    S = np.zeros(N * w + nc)
    band = 3 * w - 1
    ts, xs = [np.array([0.0])], [np.zeros((nc, 1))]
    t0, ok = 0.0, True
    marks = list(t_chunks)
    i_mark = 0
    while i_mark < len(marks):
        t1 = marks[i_mark]
        i_mark += 1
        grid = np.linspace(t0, t1, 401)
        sol = solve_ivp(rhs, (t0, t1), S, method="LSODA", rtol=rtol, atol=BED_ATOL_REL * float(c0.max()), t_eval=grid,
                        lband=band, uband=band)
        ok &= bool(sol.success)
        S = sol.y[:, -1]
        cells = sol.y[:N * w].reshape(N, w, -1)
        c_last, c_prev = cells[N - 1, :nc, :], cells[N - 2, :nc, :]
        xs.append(((c_last + 0.5 * _outlet_slope(c_last, c_prev)) / c0[:, None])[:, 1:])
        ts.append(sol.t[1:])
        t0 = t1
        if not ok:
            break
        if i_mark == len(marks) < extend_until_saturated and float(np.min(xs[-1][:, -1])) < 0.999:
            marks.append(t1 + marks[0])
    cell = S[:N * w].reshape(N, w)
    hold = dz * (eps * cell[:, :nc].sum(axis=0) + rho_b * cell[:, nc:].sum(axis=0))
    return {"t": np.concatenate(ts), "X": np.concatenate(xs, axis=1), "out": S[N * w:], "holdup": hold, "ok": ok, "t_end": t0,
            "marks": marks[:i_mark]}


def _first_crossing(t: np.ndarray, x: np.ndarray, level: float) -> float | None:
    idx = np.nonzero(x >= level)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - x[i - 1]) * (t[i] - t[i - 1]) / (x[i] - x[i - 1]))


def fixed_bed_breakthrough(bed_length_m: float | None = None, superficial_velocity_m_s: float | None = None,
                           bed_void_fraction: float | None = None, bulk_density_kg_m3: float | None = None,
                           components: list | None = None, breakthrough_fraction: float | None = None,
                           saturation_fraction: float | None = None, axial_dispersion_m2_s: float | None = None,
                           t_end_s: float | None = None, n_cells: int = 200, check_grid: bool = True) -> dict:
    """Breakthrough curve of an initially clean isothermal bed. components: [{'name', 'feed_concentration_mol_m3',
    'isotherm': {...}, 'k_ldf_1_s'}]. Returns the curve, breakthrough and saturation times, stoichiometric time,
    mass-transfer-zone length, LUB, bed usage at breakthrough and the evidence."""
    L = _positive(bed_length_m, "bed_length_m", "longitud del lecho empacado")
    u = _positive(superficial_velocity_m_s, "superficial_velocity_m_s", "velocidad superficial del fluido (caudal/área del lecho)")
    eps = _positive(bed_void_fraction, "bed_void_fraction", "fracción de huecos del lecho (entre partículas)")
    rho_b = _positive(bulk_density_kg_m3, "bulk_density_kg_m3", "densidad aparente del lecho (kg de adsorbente por m3 de lecho)")
    if eps >= 1.0:
        raise ValueError("bed_void_fraction debe ser menor que 1.")
    comps = _need(components, "components", "lista de adsorbatos con concentración de alimentación, isoterma y k_LDF")
    if not isinstance(comps, list) or not comps:
        raise ValueError("components debe ser una lista con al menos un adsorbato.")
    bf = _positive(breakthrough_fraction, "breakthrough_fraction", "c/c0 que defines como punto de ruptura (por ejemplo 0.05); es un "
                   "criterio de operación tuyo")
    if not 0.0 < bf < 0.5:
        raise ValueError("breakthrough_fraction debe estar entre 0 y 0.5.")
    assumptions = ["Lecho inicialmente limpio (sin adsorbato) e isotérmico; velocidad superficial constante (adsorbato diluido "
                   "o líquido).",
                   "Transferencia de masa por fuerza impulsora lineal (LDF) sobre la fase sólida: dq/dt = k (q* − q), con k del "
                   "usuario, constante."]
    if saturation_fraction is None:
        sf = 1.0 - bf
        assumptions.append(f"Punto de saturación tomado en c/c0 = 1 − breakthrough_fraction = {sf:g} (definición simétrica de la "
                           "zona de transferencia de masa).")
    else:
        sf = float(saturation_fraction)
        if not bf < sf < 1.0:
            raise ValueError("saturation_fraction debe estar entre breakthrough_fraction y 1.")
    if axial_dispersion_m2_s is None:
        D = 0.0
        assumptions.append("Flujo pistón sin dispersión axial (no se dio axial_dispersion_m2_s).")
    else:
        D = _positive(axial_dispersion_m2_s, "axial_dispersion_m2_s", "coeficiente de dispersión axial", allow_zero=True)
        if D > 0:
            assumptions.append("Dispersión axial con condiciones de frontera de Danckwerts.")
    names = [str(c.get("name", f"componente_{i + 1}")) for i, c in enumerate(comps)]
    c0 = np.array([_positive(c.get("feed_concentration_mol_m3"), f"components[{i}]['feed_concentration_mol_m3']",
                             "concentración del adsorbato en la alimentación") for i, c in enumerate(comps)])
    k = np.array([_positive(c.get("k_ldf_1_s"), f"components[{i}]['k_ldf_1_s']", "coeficiente LDF de transferencia de masa "
                            "(1/s), medido o ajustado a una curva de ruptura") for i, c in enumerate(comps)])
    iso = _parse_isotherms(comps, "components", c0)
    assumptions += iso["notes"]
    n_cells = int(n_cells)
    if n_cells < 20:
        raise ValueError("n_cells debe ser al menos 20.")
    q0 = iso["qstar"](c0[:, None])[:, 0]
    t_st = L * (eps + rho_b * q0 / c0) / u                       # closed form of integral (1 - c/c0) dt
    warnings: list[str] = []
    if t_end_s is not None:
        chunks = [_positive(t_end_s, "t_end_s", "tiempo final de la simulación")]
    else:
        chunks = [1.5 * float(t_st.max())]
    run = _bed_run(L, u, eps, rho_b, D, c0, k, iso["qstar"], n_cells, chunks, 1e-8, 0 if t_end_s is not None else 10)
    t, X = run["t"], run["X"]
    half = None
    if check_grid:
        half = _bed_run(L, u, eps, rho_b, D, c0, k, iso["qstar"], max(10, n_cells // 2), run["marks"], 1e-8)
    results, reasons = [], []
    ev = {"integrator_success": bool(run["ok"]), "n_cells": n_cells, "per_component": {}}
    if not run["ok"]:
        reasons.append("la integración en el tiempo no terminó")
    for i, name in enumerate(names):
        x = X[i]
        saturated = bool(x[-1] >= 0.999)
        tb = _first_crossing(t, x, bf)
        ts = _first_crossing(t, x, sf)
        fed = u * c0[i] * run["t_end"]
        t_t = run["t_end"] - run["out"][i] / (u * c0[i])
        closure = abs(fed - run["out"][i] - run["holdup"][i]) / fed
        e = {"mass_balance_rel_error": float(closure), "outlet_reached_feed_concentration": saturated}
        res = {"name": name, "feed_concentration_mol_m3": float(c0[i]), "equilibrium_loading_at_feed_mol_kg": float(q0[i]),
               "stoichiometric_time_s": float(t_st[i]), "breakthrough_time_s": tb, "saturation_time_s": ts}
        if closure > BALANCE_TOL:
            reasons.append(f"{name}: el balance de materia no cierra ({closure:.2e} relativo)")
        if saturated:
            e["stoichiometric_time_rel_diff"] = float(abs(t_t - t_st[i]) / t_st[i])
            if e["stoichiometric_time_rel_diff"] > 5e-3:
                reasons.append(f"{name}: la integral de (1 − c/c0) no coincide con el tiempo estequiométrico")
        else:
            warnings.append(f"{name}: la salida no llegó a la concentración de alimentación en el tiempo simulado "
                            f"(c/c0 final = {x[-1]:.4f}); t_t y LUB no se dan.")
        if tb is None:
            warnings.append(f"{name}: no se alcanzó el punto de ruptura c/c0 = {bf:g} en el tiempo simulado.")
        if tb is not None and saturated:
            m = t <= tb
            tt_, xx_ = np.append(t[m], tb), np.append(x[m], bf)
            t_u = float(np.trapezoid(1.0 - xx_, tt_))
            res.update(usable_capacity_time_s=t_u, total_capacity_time_s=float(t_t), bed_usage_at_breakthrough=t_u / float(t_t),
                       length_of_unused_bed_m=L * (1.0 - t_u / float(t_t)))
            if ts is not None:
                mtz = L * (ts - tb) / float(t_st[i])
                res["mass_transfer_zone_length_m"] = mtz
                if mtz > L:
                    warnings.append(f"{name}: la zona de transferencia de masa ({mtz:.3g} m) es más larga que el lecho: el "
                                    "frente no se desarrolla y el lecho apenas se aprovecha.")
        if float(x.max()) > 1.0 + 1e-3:
            res["roll_up_max_c_over_c0"] = float(x.max())
            warnings.append(f"{name}: sobreconcentración a la salida (c/c0 máx = {x.max():.3f}) por desplazamiento de un "
                            "componente que se adsorbe con más fuerza.")
        if half is not None and tb is not None:
            tb_h = _first_crossing(half["t"], half["X"][i], bf)
            e["breakthrough_time_change_on_half_grid_over_stoich"] = (float(abs(tb - tb_h) / t_st[i]) if tb_h is not None
                                                                      else None)
            e["curve_max_abs_change_on_half_grid"] = float(np.max(np.abs(np.interp(t, half["t"], half["X"][i]) - x)))
            if tb_h is None or e["breakthrough_time_change_on_half_grid_over_stoich"] > GRID_TOL:
                reasons.append(f"{name}: el tiempo de ruptura cambia más de {GRID_TOL:.0%} del tiempo estequiométrico al usar la "
                               "mitad de las celdas: el frente es más delgado que la malla (sube n_cells)")
        if iso["models"][i] == "linear" and D == 0.0 and iso["par"][i][0] > 0:
            xi = k[i] * iso["par"][i][0] * rho_b * L / u
            sel = np.unique(np.linspace(1, len(t) - 1, 60).astype(int))
            exact = np.array([anzelius_exact(xi, k[i] * (tt - eps * L / u)) for tt in t[sel]])
            e["anzelius_exact_max_abs_diff"] = float(np.max(np.abs(exact - x[sel])))
            e["dimensionless_bed_length_xi"] = float(xi)
            if e["anzelius_exact_max_abs_diff"] > ANALYTIC_TOL:
                reasons.append(f"{name}: la curva se aparta {e['anzelius_exact_max_abs_diff']:.3f} de la solución exacta de "
                               "Anzelius/Thomas (sube n_cells)")
        if iso["models"][i] == "langmuir" and D == 0.0 and len(names) == 1 and tb is not None and ts is not None:
            r_sep = 1.0 / (1.0 + iso["par"][i][1] * c0[i])
            cp = constant_pattern_langmuir_ldf_time(bf, sf, k[i], r_sep)
            e["constant_pattern_width_s"] = float(cp)
            e["constant_pattern_width_rel_diff"] = float(abs((ts - tb) - cp) / cp)
            e["langmuir_separation_factor"] = float(r_sep)
        ev["per_component"][name] = e
        results.append(res)
    if not check_grid:
        reasons.append("no se comprobó la independencia de malla (check_grid=False)")
    ev["verified"] = not reasons
    if reasons:
        ev["reasons"] = reasons
    keep = np.unique(np.linspace(0, len(t) - 1, 121).astype(int))
    return {"components": results,
            "breakthrough_curve": {"time_s": [float(v) for v in t[keep]],
                                   "c_over_c0": {n_: [float(v) for v in X[i][keep]] for i, n_ in enumerate(names)}},
            "breakthrough_fraction": bf, "saturation_fraction": sf, "simulated_time_s": float(run["t_end"]),
            "definitions": {"stoichiometric_time_s": "L (ε + ρ_b q*(c0)/c0)/u: instante del frente ideal en escalón",
                            "mass_transfer_zone_length_m": "L (t_saturación − t_ruptura)/t_estequiométrico",
                            "length_of_unused_bed_m": "L (1 − t_u/t_t), con t_u = ∫(1−c/c0)dt hasta la ruptura y t_t hasta saturar",
                            "bed_usage_at_breakthrough": "t_u/t_t: fracción de la capacidad usada al llegar a la ruptura"},
            "warnings": warnings, "assumptions": assumptions,
            "not_included": ["Efectos térmicos (calor de adsorción), caída de presión y cambio de velocidad por adsorción de un "
                             "componente mayoritario.",
                             "Resistencia de película externa y difusión intrapartícula por separado (van agrupadas en k_LDF).",
                             "Isotermas IAST o dependientes de la temperatura; regeneración del lecho."],
            "method": ("Volúmenes finitos (limitador de van Leer) en z y scipy.integrate.solve_ivp (LSODA, jacobiano en banda) en "
                       "el tiempo; tiempos por interpolación lineal de la curva de salida."),
            "evidence": ev}


# ---------------------------------------------------------------------------------------------------------------
# LUB scale-up from laboratory data
# ---------------------------------------------------------------------------------------------------------------
def lub_scaleup(time_s: list | None = None, c_over_c0: list | None = None, bed_length_m: float | None = None,
                breakthrough_fraction: float | None = None, new_breakthrough_time_s: float | None = None,
                new_bed_length_m: float | None = None, solute_feed_rate_kg_s: float | None = None,
                adsorbent_mass_kg: float | None = None) -> dict:
    """Length of unused bed from a measured breakthrough curve and its transfer to another bed length at the same
    superficial velocity, particle and feed (constant-pattern assumption)."""
    t = np.asarray(_need(time_s, "time_s", "tiempos de la curva de ruptura medida en el laboratorio"), dtype=float)
    x = np.asarray(_need(c_over_c0, "c_over_c0", "c/c0 medido a la salida en esos tiempos"), dtype=float)
    L = _positive(bed_length_m, "bed_length_m", "longitud del lecho de laboratorio")
    bf = _positive(breakthrough_fraction, "breakthrough_fraction", "c/c0 que defines como punto de ruptura")
    if t.ndim != 1 or t.shape != x.shape or len(t) < 4:
        raise ValueError("time_s y c_over_c0 deben ser listas de la misma longitud con al menos 4 puntos.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s debe ser estrictamente creciente.")
    if np.any(x < -1e-9) or np.any(x > 1.5):
        raise ValueError("c_over_c0 debe ser el cociente c/c0 (entre 0 y ~1).")
    if not 0.0 < bf < 1.0:
        raise ValueError("breakthrough_fraction debe estar entre 0 y 1.")
    if x[0] >= bf:
        raise ValueError("La curva empieza por encima del punto de ruptura: faltan los datos anteriores a la ruptura.")
    warnings, reasons = [], []
    tb = _first_crossing(t, x, bf)
    if tb is None:
        raise ValueError(f"La curva medida no alcanza el punto de ruptura c/c0 = {bf:g}.")
    t_t = float(np.trapezoid(1.0 - x, t)) + float(t[0])           # the bed adsorbs everything before the first sample
    m = t <= tb
    t_u = float(np.trapezoid(1.0 - np.append(x[m], bf), np.append(t[m], tb))) + float(t[0])
    pch = PchipInterpolator(t, x)
    t_t_p = float(quad(lambda s: 1.0 - float(pch(s)), t[0], t[-1], limit=400)[0]) + float(t[0])
    quad_diff = abs(t_t_p - t_t) / t_t
    if x[-1] < 0.99:
        warnings.append(f"La curva termina en c/c0 = {x[-1]:.3f}: la cola que falta hasta la saturación no se integra y t_t queda "
                        "subestimado (LUB subestimada).")
    if x[-1] < 0.95:
        reasons.append("la curva medida no llega a la saturación (c/c0 final < 0.95): la capacidad total no está determinada")
    if np.any(np.diff(x) < -0.02):
        warnings.append("La curva medida no es monótona (ruido o desplazamiento competitivo): revisa los datos.")
    if quad_diff > 0.02:
        reasons.append("la integral depende del esquema de cuadratura (trapecio vs. PCHIP) más de 2 %: hacen falta más puntos")
    frac = t_u / t_t
    lub = L * (1.0 - frac)
    out = {"laboratory": {"bed_length_m": L, "breakthrough_time_s": tb, "total_capacity_time_s": t_t,
                          "usable_capacity_time_s": t_u, "bed_usage_at_breakthrough": frac, "used_bed_length_m": L * frac,
                          "length_of_unused_bed_m": lub}}
    if (solute_feed_rate_kg_s is None) != (adsorbent_mass_kg is None):
        raise ValueError("Para la capacidad de saturación da solute_feed_rate_kg_s y adsorbent_mass_kg juntos.")
    if solute_feed_rate_kg_s is not None:
        rate = _positive(solute_feed_rate_kg_s, "solute_feed_rate_kg_s", "flujo másico de soluto alimentado al lecho de laboratorio")
        mass = _positive(adsorbent_mass_kg, "adsorbent_mass_kg", "masa de adsorbente del lecho de laboratorio")
        out["laboratory"]["saturation_capacity_kg_kg"] = rate * t_t / mass
        out["laboratory"]["capacity_used_at_breakthrough_kg_kg"] = rate * t_u / mass
    if new_breakthrough_time_s is not None and new_bed_length_m is not None:
        raise ValueError("Da new_breakthrough_time_s (para obtener la longitud) O new_bed_length_m (para obtener el tiempo de "
                         "ruptura), no ambos.")
    if new_breakthrough_time_s is not None:
        tb2 = _positive(new_breakthrough_time_s, "new_breakthrough_time_s", "tiempo de ruptura deseado en el lecho nuevo")
        used = L * frac * tb2 / tb
        out["scale_up"] = {"breakthrough_time_s": tb2, "used_bed_length_m": used, "bed_length_m": used + lub,
                           "bed_usage_at_breakthrough": used / (used + lub)}
    elif new_bed_length_m is not None:
        L2 = _positive(new_bed_length_m, "new_bed_length_m", "longitud del lecho nuevo")
        if L2 <= lub:
            raise ValueError(f"new_bed_length_m ({L2:g} m) no supera la longitud de lecho no usado ({lub:.4g} m): rompería de "
                             "inmediato.")
        out["scale_up"] = {"bed_length_m": L2, "used_bed_length_m": L2 - lub, "breakthrough_time_s": tb * (L2 - lub) / (L * frac),
                           "bed_usage_at_breakthrough": (L2 - lub) / L2}
    ev = {"quadrature_trapezoid_vs_pchip_rel_diff": quad_diff, "final_c_over_c0": float(x[-1]), "verified": not reasons}
    if reasons:
        ev["reasons"] = reasons
    out.update(warnings=warnings,
               assumptions=["La zona de transferencia de masa tiene forma constante (isoterma favorable): la longitud de lecho no "
                            "usado no depende de la longitud del lecho.",
                            "El lecho nuevo trabaja con la misma velocidad superficial, el mismo adsorbente y tamaño de partícula, la "
                            "misma alimentación y temperatura que el ensayo; el diámetro se escala con el caudal.",
                            "La longitud usada escala con el tiempo de ruptura (Geankoplis ec. 12.3-5): H_B2 = H_B1 t_b2/t_b1.",
                            "Antes del primer dato el lecho retiene todo el soluto (c/c0 = 0)."],
               not_included=["Cambio de velocidad superficial, de partícula o de concentración entre el ensayo y la planta (cambian "
                             "la LUB).", "Caída de presión, efectos térmicos y mala distribución en lechos de gran diámetro."],
               method="Integración por trapecios de 1 − c/c0 sobre los datos del usuario; ruptura por interpolación lineal (numpy).",
               evidence=ev)
    return out


# ---------------------------------------------------------------------------------------------------------------
# PSA: four-step Skarstrom cycle of a binary on one bed
# ---------------------------------------------------------------------------------------------------------------
def psa_equilibrium_theory(bed_length_m: float, bed_void_fraction: float, bulk_density_kg_m3: float, K_heavy_m3_kg: float,
                           K_light_m3_kg: float, pressure_ratio: float, heavy_feed_mole_fraction: float,
                           feed_velocity_m_s: float | None = None, purge_velocity_m_s: float | None = None) -> dict:
    """Equilibrium theory of the four-step cycle with pressurisation by product, feed up to breakthrough and complete
    purge, for linear isotherms (instantaneous uptake, plug flow, isothermal, no pressure drop).

    beta_i = eps/(eps + rho_b K_i) is the fraction of component i in the gas of a bed element; beta = beta_A/beta_B
    (A heavy). At constant pressure u (1 + (beta - 1) y_A) is uniform, a shock of y_A = 0 -> y_F travels at beta_A u_in/eps
    and a composition y -> 0 at beta_A u/eps. Per unit of bed cross-section, with C = P/(R T):
        feed to breakthrough       eps L C_H / beta_A               product out   (1 + (beta - 1) y_F) times that
        purge to clean the bed     eps L C_L / beta_A               pressurisation with product  eps L (C_H - C_L)/beta_B
        recovery of the light gas  (1 - beta)(1 - 1/(P_ratio (1 - y_F)))
    which is the binary-linear-isotherm result of Knaebel & Hill (1985); the purge needs C_L u_L t_P = C_H u_H t_F / P_ratio.
    """
    eps, rho_b, L = bed_void_fraction, bulk_density_kg_m3, bed_length_m
    bA = eps / (eps + rho_b * K_heavy_m3_kg)
    bB = eps / (eps + rho_b * K_light_m3_kg)
    beta = bA / bB
    yF = heavy_feed_mole_fraction
    out = {"beta_heavy": bA, "beta_light": bB, "beta": beta,
           "light_recovery_complete_purge_product_pressurization": (1.0 - beta) * (1.0 - 1.0 / (pressure_ratio * (1.0 - yF))),
           "max_light_recovery_infinite_pressure_ratio": 1.0 - beta,
           "product_to_feed_molar_ratio_during_feed_step": 1.0 + (beta - 1.0) * yF,
           "feed_volume_to_breakthrough_m3_m2": eps * L / bA, "purge_volume_to_clean_bed_m3_m2": eps * L / bA,
           "note": ("volúmenes a la presión del paso, por m2 de sección; alimentar más allá de ese volumen contamina el producto y "
                    "purgar menos deja componente pesado en el lecho")}
    if feed_velocity_m_s:
        out["feed_time_to_breakthrough_s"] = eps * L / (bA * feed_velocity_m_s)
    if purge_velocity_m_s:
        out["purge_time_to_clean_bed_s"] = eps * L / (bA * purge_velocity_m_s)
    return out


def psa_equilibrium_recovery(beta: float, pressure_ratio: float, heavy_feed_mole_fraction: float, feed_fraction: float,
                             purge_fraction: float) -> float | None:
    """Light-gas recovery of the same cycle (pressurisation with product) when the feed step admits only a fraction
    f <= 1 of the volume to breakthrough and the purge is gamma >= 1 times the complete purge. The shock of the feed step
    moves at constant speed, so the product of that step is f times its full value; the purge still leaves the bed clean
    (the y = 0 characteristic crosses the bed in eps L/(beta_A u_purge) whatever the profile left by the blowdown), so
        R = [f (1 + (beta - 1) y_F) - gamma / P_ratio - beta (1 - 1/P_ratio)] / [f (1 - y_F)]
    and the product is pure. f = gamma = 1 recovers (1 - beta)(1 - 1/(P_ratio (1 - y_F))). None outside f <= 1, gamma >= 1
    (the product is no longer pure and the closed form does not hold)."""
    f, g = feed_fraction, purge_fraction
    if not (0.0 < f <= 1.0 + 1e-12 and g >= 1.0 - 1e-12):
        return None
    yF, inv = heavy_feed_mole_fraction, 1.0 / pressure_ratio
    return (f * (1.0 + (beta - 1.0) * yF) - g * inv - beta * (1.0 - inv)) / (f * (1.0 - yF))


def psa_equilibrium_theory_nonlinear(bed_length_m: float, bed_void_fraction: float, bulk_density_kg_m3: float, T_K: float,
                                     isotherms: list, high_pressure_Pa: float, low_pressure_Pa: float,
                                     heavy_feed_mole_fraction: float, feed_velocity_m_s: float, purge_velocity_m_s: float,
                                     feed_time_s: float, purge_time_s: float) -> dict:
    """Equilibrium theory of the four-step cycle with pressurisation by product for linear, Langmuir or extended-Langmuir
    isotherms (isotherms = [heavy, light] isotherm dicts; instantaneous uptake, plug flow, isothermal, uniform pressure).

    Feed step at C_H = P_H/(R T) into a bed of pure light gas B: the heavy front is a shock whose jump conditions
        A:      w (eps C_H y_F + rho_b q_A,F) = u_F C_H y_F
        total:  w rho_b (q_A,F + q_B,F - q_B(0, C_H)) = C_H (u_F - u_out)
    give the breakthrough time t_bt = L / w and the velocity u_out of the pure product (q_*,F at the feed mixture). The
    shock is admissible (Oleinik/Liu) when w(y_F -> 0) <= w(y_F -> y) for every 0 < y < y_F, with
        w(y_F -> y) = C u_F dy / (eps C dy + rho_b (dq_A - y dQ)),   d(.) = (.)(y_F) - (.)(y),  Q = q_A + q_B.
    Purge at C_L with pure B: the slowest characteristic is c_A -> 0, speed u_P / (eps + rho_b dq_A/dc_A|0), so a purge
    of t_P >= L (eps + rho_b dq_A/dc_A|0) / u_P leaves the bed clean whatever the blowdown left (sufficient bound; for a
    favourable isotherm the desorption wave is a simple wave and never overtakes it). Per unit cross-section and cycle
        gross product  u_out C_H t_F            pressurisation  L [eps (C_H - C_L) + rho_b (q_B(0, C_H) - q_B(0, C_L))]
        purge          u_P C_L t_P              recovery R = (gross - pressurisation - purge) / (u_F C_H t_F (1 - y_F))
    with a pure product, valid for t_F <= t_bt, t_P >= the bound and an admissible shock; for linear isotherms it reduces
    exactly to `psa_equilibrium_recovery`."""
    eps, rho_b, L = bed_void_fraction, bulk_density_kg_m3, bed_length_m
    if len(isotherms) != 2:
        raise ValueError("isotherms debe tener dos isotermas: [gas pesado, gas ligero].")
    for i, s in enumerate(isotherms):
        if (s or {}).get("model") not in ("linear", "langmuir"):
            raise ValueError(f"isotherms[{i}]: la teoría de equilibrio no lineal solo está derivada para isotermas lineales o de "
                             f"Langmuir (recibido {(s or {}).get('model')!r}).")
    iso = _parse_isotherms([{"isotherm": s} for s in isotherms], "isotherms")
    RT = R_GAS * T_K
    CH, CL = high_pressure_Pa / RT, low_pressure_Pa / RT
    yF, uF, uP = heavy_feed_mole_fraction, feed_velocity_m_s, purge_velocity_m_s

    def q(cA, cB) -> np.ndarray:
        cA, cB = np.atleast_1d(np.asarray(cA, float)), np.atleast_1d(np.asarray(cB, float))
        return iso["qstar"](np.vstack([cA, cB]))

    qF = q(CH * yF, CH * (1.0 - yF))[:, 0]
    qBH, qBL = float(q(0.0, CH)[1, 0]), float(q(0.0, CL)[1, 0])
    w = uF * CH * yF / (eps * CH * yF + rho_b * qF[0])
    t_bt = L / w
    u_out = uF - w * rho_b * (qF[0] + qF[1] - qBH) / CH
    y = np.linspace(0.0, yF, 401)[1:-1]
    qy = q(CH * y, CH * (1.0 - y))
    dy, dqA, dQ = yF - y, qF[0] - qy[0], (qF[0] + qF[1]) - (qy[0] + qy[1])
    den = eps * CH * dy + rho_b * (dqA - y * dQ)
    shock_ok = bool(np.all(den > 0.0) and np.all(CH * uF * dy / den >= w * (1.0 - 1e-9)))
    dc = 1e-9 * CH * yF
    slope0 = float(q(dc, CL)[0, 0]) / dc                  # dq_A/dc_A at c_A -> 0 in pure B at C_L
    t_pb = L * (eps + rho_b * slope0) / uP if uP > 0 else None
    n_feed = uF * CH * feed_time_s
    n_gross = u_out * CH * feed_time_s
    n_pr = L * (eps * (CH - CL) + rho_b * (qBH - qBL))
    n_pu = uP * CL * purge_time_s
    f = feed_time_s / t_bt
    g = purge_time_s / t_pb if t_pb else 0.0
    ok = shock_ok and u_out > 0 and f <= 1.0 + 1e-12 and g >= 1.0 - 1e-12
    out = {"beta_heavy_at_origin": eps / (eps + rho_b * slope0), "heavy_loading_at_feed_mol_kg": float(qF[0]),
           "light_loading_at_feed_mol_kg": float(qF[1]), "feed_shock_velocity_m_s": w, "feed_time_to_breakthrough_s": t_bt,
           "feed_shock_admissible": shock_ok, "product_velocity_during_feed_m_s": u_out,
           "product_to_feed_molar_ratio_during_feed_step": u_out / uF, "purge_time_bound_to_clean_bed_s": t_pb,
           "this_cycle_feed_time_over_breakthrough": f, "this_cycle_purge_time_over_bound": g,
           "per_cycle_mol_m2": {"feed": n_feed, "gross_product_light": n_gross, "pressurization_gas_light": n_pr,
                                "purge_gas_light": n_pu},
           "light_recovery_this_cycle": float((n_gross - n_pr - n_pu) / (n_feed * (1.0 - yF))) if ok else None,
           "note": ("forma cerrada para presurización con producto: vale con alimentación hasta la ruptura o menos, purga mayor "
                    "o igual que la cota de limpieza y frente de alimentación de choque; el producto es puro")}
    if not ok:
        why = [] if shock_ok else ["el frente de alimentación no es un choque admisible"]
        why += [] if f <= 1.0 + 1e-12 else ["la alimentación pasa del tiempo de ruptura"]
        why += [] if g >= 1.0 - 1e-12 else ["la purga es menor que la cota de limpieza del lecho"]
        why += [] if u_out > 0 else ["no sale producto durante la alimentación"]
        out["light_recovery_this_cycle_note"] = "fuera del dominio de la forma cerrada: " + "; ".join(why)
    return out


def _psa_cycle_run(p: dict, N: int, max_cycles: int, css_tol: float) -> dict:
    L, eps, rho_b, T = p["L"], p["eps"], p["rho_b"], p["T"]
    PH, PL = p["PH"], p["PL"]
    k = p["k"]
    qstar = p["qstar"]
    dz = L / N
    RT = R_GAS * T
    CH, CL = PH / RT, PL / RT
    yF = p["yF"]

    def make_rhs(P0: float, P1: float, dur: float, bc: str, F_bc: float, y_in: float):
        dC = (P1 - P0) / dur / RT

        def rhs(t, S):
            y = S[:N]
            q = S[N:3 * N].reshape(2, N)
            C = (P0 + (P1 - P0) * t / dur) / RT
            c = np.vstack([C * y, C * (1.0 - y)])
            dq = k[:, None] * (qstar(c) - q)
            sink = dz * (eps * dC + rho_b * (dq[0] + dq[1]))
            F = np.empty(N + 1)
            if bc == "z0":
                F[0] = F_bc
                F[1:] = F_bc - np.cumsum(sink)
            else:
                F[N] = F_bc
                F[:N] = F_bc + np.cumsum(sink[::-1])[::-1]
            s = _limited_slopes(y, outlet=False)
            yl, yr = y + 0.5 * s, y - 0.5 * s                 # downstream-face value for flow in +z / in -z
            yf = np.empty(N + 1)
            yf[1:N] = np.where(F[1:N] >= 0.0, yl[:-1], yr[1:])
            yf[0] = y_in if F[0] >= 0.0 else y[0]
            yf[N] = y[N - 1] if F[N] >= 0.0 else y_in
            FA = F * yf
            dy = (-(FA[1:] - FA[:-1]) - rho_b * dz * dq[0] - eps * dz * y * dC) / (eps * C * dz)
            return np.concatenate([dy, dq[0], dq[1], [F[0], FA[0], F[N], FA[N]]])

        return rhs

    def step(S, P0, P1, dur, bc, F_bc, y_in):
        S = S.copy()
        S[3 * N:] = 0.0
        sol = solve_ivp(make_rhs(P0, P1, dur, bc, F_bc, y_in), (0.0, dur), S, method="LSODA", rtol=1e-7, atol=1e-9)
        return sol.y[:, -1], bool(sol.success)

    def inventory(S, P):
        y = S[:N]
        C = P / RT
        tot = dz * (eps * C * N + rho_b * (S[N:2 * N].sum() + S[2 * N:3 * N].sum()))
        a = dz * (eps * C * y.sum() + rho_b * S[N:2 * N].sum())
        return np.array([a, tot - a])

    # start: bed at low pressure filled with the pure second component (the product), in equilibrium
    y0 = np.zeros(N)
    c0 = np.vstack([CL * y0, CL * (1.0 - y0)])
    S = np.concatenate([y0, qstar(c0).ravel(), np.zeros(4)])
    y_prod = 0.0
    hist, ok, cyc, prev = [], True, None, None
    for n in range(1, max_cycles + 1):
        inv0 = inventory(S, PL)
        S_start = S.copy()
        if p["press_with"] == "feed":
            S, o1 = step(S, PL, PH, p["t_pr"], "zL", 0.0, yF)
            pr_in = np.array([S[3 * N + 1], S[3 * N] - S[3 * N + 1]])              # entered at z = 0
        else:
            S, o1 = step(S, PL, PH, p["t_pr"], "z0", 0.0, y_prod)
            pr_in = -np.array([S[3 * N + 3], S[3 * N + 2] - S[3 * N + 3]])         # entered at z = L
        S, o2 = step(S, PH, PH, p["t_feed"], "z0", p["uF"] * CH, yF)
        feed_in = np.array([S[3 * N + 1], S[3 * N] - S[3 * N + 1]])
        gross = np.array([S[3 * N + 3], S[3 * N + 2] - S[3 * N + 3]])
        y_prod_new = float(gross[0] / gross.sum()) if gross.sum() > 0 else y_prod
        S, o3 = step(S, PH, PL, p["t_bd"], "zL", 0.0, y_prod_new)
        bd_out = -np.array([S[3 * N + 1], S[3 * N] - S[3 * N + 1]])
        S, o4 = step(S, PL, PL, p["t_pu"], "zL", -p["uP"] * CL, y_prod_new)
        pu_in = -np.array([S[3 * N + 3], S[3 * N + 2] - S[3 * N + 3]])
        pu_out = -np.array([S[3 * N + 1], S[3 * N] - S[3 * N + 1]])
        ok &= o1 and o2 and o3 and o4
        inv1 = inventory(S, PL)
        total_in = pr_in + feed_in + pu_in
        total_out = gross + bd_out + pu_out
        closure = float(np.max(np.abs(total_in - total_out - (inv1 - inv0)) / np.maximum(total_in, 1e-300)))
        drift = float(np.max(np.abs(inv1 - inv0) / np.maximum(total_in, 1e-300)))
        q_scale = max(float(np.max(np.abs(S[N:3 * N]))), 1e-300)
        resid = max(float(np.max(np.abs(S[:N] - S_start[:N]))), float(np.max(np.abs(S[N:3 * N] - S_start[N:3 * N]))) / q_scale)
        cyc = {"pr_in": pr_in, "feed_in": feed_in, "gross": gross, "bd_out": bd_out, "pu_in": pu_in, "pu_out": pu_out,
               "closure": closure, "drift": drift, "resid": resid, "y_prod": y_prod_new, "y_prod_prev": y_prod}
        hist.append(resid)
        y_prod = y_prod_new
        if not ok:
            break
        if n >= 3 and resid < css_tol and drift < css_tol:
            break
    cyc.update(n_cycles=n, ok=ok, converged=bool(ok and cyc["resid"] < css_tol and cyc["drift"] < css_tol),
               history=hist[-5:], y_profile=S[:N].copy())
    return cyc


def _psa_performance(c: dict, press_with: str, prod_idx: int) -> dict:
    used = c["pu_in"] + (c["pr_in"] if press_with == "product" else 0.0)
    net = c["gross"] - used
    fed = c["feed_in"] + (c["pr_in"] if press_with == "feed" else 0.0)
    return {"net": net, "fed": fed, "purity": float(c["gross"][prod_idx] / c["gross"].sum()) if c["gross"].sum() > 0 else math.nan,
            "recovery": float(net[prod_idx] / fed[prod_idx])}


def psa_skarstrom(bed_length_m: float | None = None, bed_void_fraction: float | None = None,
                  bulk_density_kg_m3: float | None = None, T_K: float | None = None, components: list | None = None,
                  product_component: str | None = None, high_pressure_Pa: float | None = None,
                  low_pressure_Pa: float | None = None, feed_velocity_m_s: float | None = None,
                  purge_velocity_m_s: float | None = None, pressurization_time_s: float | None = None,
                  feed_time_s: float | None = None, blowdown_time_s: float | None = None, purge_time_s: float | None = None,
                  pressurization_with: str | None = None, bed_diameter_m: float | None = None, n_cells: int = 40,
                  max_cycles: int = 200, css_tolerance: float = 1e-4, check_grid: bool = True) -> dict:
    """Four-step PSA cycle (pressurisation, feed, countercurrent blowdown, countercurrent purge with product) of a binary
    gas on one isothermal bed, run to the cyclic steady state. components: two dicts {'name', 'feed_mole_fraction',
    'isotherm', 'k_ldf_1_s'}; product_component is the weakly adsorbed gas taken at the product end."""
    L = _positive(bed_length_m, "bed_length_m", "longitud del lecho")
    eps = _positive(bed_void_fraction, "bed_void_fraction", "fracción de huecos del lecho")
    rho_b = _positive(bulk_density_kg_m3, "bulk_density_kg_m3", "densidad aparente del lecho")
    T = _positive(T_K, "T_K", "temperatura del lecho (isotérmico)")
    comps = _need(components, "components", "los dos gases con fracción molar de alimentación, isoterma y k_LDF")
    if not isinstance(comps, list) or len(comps) != 2:
        raise ValueError("components debe tener exactamente dos gases (ciclo PSA binario).")
    names = [str(c.get("name", f"componente_{i + 1}")) for i, c in enumerate(comps)]
    if product_component not in names:
        raise ValueError(f"Falta product_component o no coincide con un componente ({names}): es el gas poco adsorbido que se "
                         "recoge como producto.")
    PH = _positive(high_pressure_Pa, "high_pressure_Pa", "presión absoluta del paso de adsorción")
    PL = _positive(low_pressure_Pa, "low_pressure_Pa", "presión absoluta de despresurización y purga")
    if PL >= PH:
        raise ValueError("low_pressure_Pa debe ser menor que high_pressure_Pa.")
    uF = _positive(feed_velocity_m_s, "feed_velocity_m_s", "velocidad superficial de la alimentación a la presión alta")
    uP = _positive(purge_velocity_m_s, "purge_velocity_m_s", "velocidad superficial del gas de purga a la presión baja (0 = sin "
                   "purga)", allow_zero=True)
    t_pr = _positive(pressurization_time_s, "pressurization_time_s", "duración de la presurización")
    t_f = _positive(feed_time_s, "feed_time_s", "duración del paso de alimentación a presión alta")
    t_bd = _positive(blowdown_time_s, "blowdown_time_s", "duración de la despresurización")
    t_pu = _positive(purge_time_s, "purge_time_s", "duración de la purga")
    if pressurization_with not in ("feed", "product"):
        raise ValueError("Falta pressurization_with: 'feed' (por el extremo de alimentación) o 'product' (por el extremo de "
                         "producto). Cambia la recuperación; no se supone.")
    z = np.array([_positive(c.get("feed_mole_fraction"), f"components[{i}]['feed_mole_fraction']", "fracción molar en la "
                            "alimentación") for i, c in enumerate(comps)])
    if abs(z.sum() - 1.0) > 1e-6:
        raise ValueError(f"Las fracciones molares de alimentación deben sumar 1 (suman {z.sum():.6g}).")
    kk = np.array([_positive(c.get("k_ldf_1_s"), f"components[{i}]['k_ldf_1_s']", "coeficiente LDF (1/s)") for i, c in enumerate(comps)])
    prod = names.index(product_component)
    order = [1 - prod, prod]                              # internal order: [the other gas (heavy), product]
    iso = _parse_isotherms([comps[i] for i in order], "components", PH / (R_GAS * T) * z[order])
    n_cells = int(n_cells)
    if n_cells < 10 or int(max_cycles) < 3:
        raise ValueError("n_cells debe ser al menos 10 y max_cycles al menos 3.")
    par = {"L": L, "eps": eps, "rho_b": rho_b, "T": T, "PH": PH, "PL": PL, "k": kk[order], "qstar": iso["qstar"],
           "yF": float(z[order[0]]), "uF": uF, "uP": uP, "t_pr": t_pr, "t_feed": t_f, "t_bd": t_bd, "t_pu": t_pu,
           "press_with": pressurization_with}
    run = _psa_cycle_run(par, n_cells, int(max_cycles), float(css_tolerance))
    perf = _psa_performance(run, pressurization_with, 1)
    warnings, reasons = [], []
    ev = {"cyclic_steady_state_residual": run["resid"], "cycle_inventory_drift_over_feed": run["drift"],
          "cycle_mass_balance_rel_error": run["closure"], "n_cycles": run["n_cycles"], "css_tolerance": float(css_tolerance),
          "residual_last_cycles": [float(v) for v in run["history"]], "integrator_success": run["ok"], "n_cells": n_cells}
    if not run["ok"]:
        reasons.append("la integración de un paso del ciclo no terminó")
    if not run["converged"]:
        reasons.append(f"no se alcanzó el estado cíclico estacionario en {run['n_cycles']} ciclos (residuo {run['resid']:.2e})")
    if run["closure"] > BALANCE_TOL:
        reasons.append(f"el balance de materia del último ciclo no cierra ({run['closure']:.2e})")
    if check_grid:
        half = _psa_cycle_run(par, max(5, n_cells // 2), int(max_cycles), float(css_tolerance))
        ph_ = _psa_performance(half, pressurization_with, 1)
        ev["purity_change_on_half_grid"] = abs(ph_["purity"] - perf["purity"])
        ev["recovery_change_on_half_grid"] = abs(ph_["recovery"] - perf["recovery"])
        if max(ev["purity_change_on_half_grid"], ev["recovery_change_on_half_grid"]) > PSA_GRID_TOL:
            reasons.append(f"pureza o recuperación cambian más de {PSA_GRID_TOL} al usar la mitad de las celdas (sube n_cells)")
    else:
        reasons.append("no se comprobó la independencia de malla (check_grid=False)")
    ev["verified"] = not reasons
    if reasons:
        ev["reasons"] = reasons
    net, fed = perf["net"], perf["fed"]
    feasible = bool(net[1] > 0 and net.sum() > 0)
    if not feasible:
        warnings.append("El ciclo consume en purga y presurización más producto del que genera: no hay producto neto. Reduce la "
                        "purga o alarga la alimentación.")
    t_cycle = t_pr + t_f + t_bd + t_pu
    area = math.pi * _positive(bed_diameter_m, "bed_diameter_m", "diámetro del lecho") ** 2 / 4.0 if bed_diameter_m is not None else None

    def pair(v):                                           # back to the user's component order
        return {names[order[0]]: float(v[0]), names[order[1]]: float(v[1])}

    out = {"feasible": feasible, "product_component": product_component, "product_purity_mole_fraction": perf["purity"],
           "product_recovery": perf["recovery"],
           "productivity_mol_kg_s": float(net[1] / (t_cycle * rho_b * L)),
           "purge_to_feed_molar_ratio": float(run["pu_in"].sum() / run["feed_in"].sum()),
           "purge_to_feed_actual_volume_ratio": float(uP * t_pu / (uF * t_f)),
           "pressure_ratio": PH / PL, "cycle_time_s": t_cycle,
           "per_cycle_mol_m2": {"feed": pair(run["feed_in"]), "pressurization_gas": pair(run["pr_in"]),
                                "gross_product": pair(run["gross"]), "purge_gas_from_product": pair(run["pu_in"]),
                                "net_product": pair(net), "blowdown_exhaust": pair(run["bd_out"]),
                                "purge_exhaust": pair(run["pu_out"]),
                                "basis": "moles por ciclo y por m2 de sección transversal del lecho"},
           "bed_profile_end_of_cycle": {"z_m": [float((j + 0.5) * L / n_cells) for j in range(n_cells)],
                                        f"gas_mole_fraction_{names[order[0]]}": [float(v) for v in run["y_profile"]]}}
    if area is not None:
        out["per_bed"] = {"cross_section_m2": area, "adsorbent_mass_kg": rho_b * L * area,
                          "net_product_mol_s_cycle_average": float(net.sum() * area / t_cycle),
                          "feed_mol_s_cycle_average": float(fed.sum() * area / t_cycle)}
    if all(m == "linear" for m in iso["models"]):
        KA, KB = iso["par"][0][0], iso["par"][1][0]
        if KA > KB:
            th = psa_equilibrium_theory(L, eps, rho_b, KA, KB, PH / PL, float(z[order[0]]), uF, uP if uP > 0 else None)
            th["this_cycle_feed_time_over_breakthrough"] = t_f / th["feed_time_to_breakthrough_s"]
            if uP > 0:
                th["this_cycle_purge_time_over_complete"] = t_pu / th["purge_time_to_clean_bed_s"]
            r_et = None
            if pressurization_with == "product" and uP > 0:
                r_et = psa_equilibrium_recovery(th["beta"], PH / PL, float(z[order[0]]), th["this_cycle_feed_time_over_breakthrough"],
                                                th["this_cycle_purge_time_over_complete"])
            th["light_recovery_this_cycle"] = r_et
            if r_et is None:
                th["light_recovery_this_cycle_note"] = ("la forma cerrada solo vale con presurización con producto, alimentación hasta "
                                                        "la ruptura o menos y purga completa o mayor")
            else:
                ev["equilibrium_theory_recovery_abs_diff"] = abs(perf["recovery"] - r_et)
                ev["equilibrium_theory_note"] = ("informativo: la teoría de equilibrio supone transferencia instantánea y sin "
                                                 "dispersión; con LDF finito la diferencia no anula la verificación numérica")
            out["equilibrium_theory"] = th
            validation = ("comparable con la teoría de equilibrio (isotermas lineales); ver equilibrium_theory y "
                          "evidence.equilibrium_theory_recovery_abs_diff" if r_et is not None else
                          "isotermas lineales, pero este ciclo cae fuera de la forma cerrada de la teoría de equilibrio: "
                          "NO VALIDADO contra una solución conocida")
        else:
            validation = "NO VALIDADO: el producto pedido no es el gas menos adsorbido"
            warnings.append(f"{product_component} no es el gas menos adsorbido según las constantes de Henry dadas: el producto del "
                            "extremo de salida no se enriquece en él.")
    elif all(m in ("linear", "langmuir") for m in iso["models"]):
        th = psa_equilibrium_theory_nonlinear(L, eps, rho_b, T, [comps[order[0]]["isotherm"], comps[order[1]]["isotherm"]], PH,
                                              PL, float(z[order[0]]), uF, uP, t_f, t_pu)
        r_et = th["light_recovery_this_cycle"] if pressurization_with == "product" else None
        if pressurization_with != "product":
            th["light_recovery_this_cycle"] = None
            th["light_recovery_this_cycle_note"] = "la forma cerrada solo vale con presurización con producto"
        if r_et is not None:
            ev["equilibrium_theory_recovery_abs_diff"] = abs(perf["recovery"] - r_et)
            ev["equilibrium_theory_note"] = ("informativo: la teoría de equilibrio supone transferencia instantánea y sin "
                                             "dispersión; con LDF finito la diferencia no anula la verificación numérica")
            validation = ("comparable con la teoría de equilibrio para isotermas de Langmuir (choque de alimentación y purga "
                          "por encima de la cota de limpieza); ver equilibrium_theory y "
                          "evidence.equilibrium_theory_recovery_abs_diff")
        else:
            validation = ("NO VALIDADO: con isotermas no lineales la forma cerrada de la teoría de equilibrio solo vale con "
                          "presurización con producto, alimentación hasta la ruptura o menos, purga por encima de la cota de "
                          "limpieza y frente de alimentación de choque; este ciclo cae fuera (ver equilibrium_theory)")
            warnings.append("Ciclo PSA con isotermas no lineales fuera del dominio comparado con la teoría de equilibrio: solo se "
                            "certifican balance, estado cíclico estacionario y malla; úsalo para tendencias, no para diseño.")
        out["equilibrium_theory"] = th
    else:
        validation = ("NO VALIDADO: con isoterma de Freundlich no hay solución conocida con la que comparar; solo se certifican "
                      "el cierre de balance, el estado cíclico estacionario y la independencia de malla")
        warnings.append("Ciclo PSA con isoterma de Freundlich: NO VALIDADO contra una solución de referencia; úsalo para "
                        "tendencias, no para diseño.")
    if perf["purity"] < float(z[prod]):
        warnings.append("La pureza del producto es menor que la de la alimentación: el ciclo no separa en estas condiciones.")
    assumptions = ["Un solo lecho, isotérmico, sin caída de presión axial; gas ideal; mezcla binaria en la que ambos gases pueden "
                   "adsorberse (la velocidad varía a lo largo del lecho por la adsorción).",
                   "La presión sube y baja linealmente en el tiempo durante presurización y despresurización.",
                   "El gas de purga y, en su caso, el de presurización tienen la composición media del producto (tanque de "
                   "producto bien mezclado); despresurización y purga a contracorriente.",
                   "Transferencia de masa LDF con k constante; equilibrio por la isoterma del usuario en concentraciones de gas.",
                   "El estado inicial (lecho a presión baja lleno de producto puro) solo es el arranque numérico: se reporta el "
                   "estado cíclico estacionario."] + iso["notes"]
    out["validation_status"] = validation
    out.update(warnings=warnings, assumptions=assumptions,
               not_included=["Pasos de igualación de presión, enjuague con producto pesado, vacío (VSA) y ciclos de varios lechos.",
                             "Efectos térmicos (calor de adsorción), caída de presión (Ergun) y dinámica de válvulas.",
                             "Más de dos componentes; isotermas dependientes de la temperatura.",
                             "No es una herramienta de diseño de unidades industriales: sirve para explorar tendencias de un ciclo "
                             "Skarstrom ideal."],
               method=("Volúmenes finitos (limitador de van Leer) con velocidad obtenida del balance total a presión uniforme; "
                       "cada paso integrado con scipy.integrate.solve_ivp (LSODA); ciclos sucesivos hasta el estado cíclico "
                       "estacionario."),
               evidence=ev)
    return out
