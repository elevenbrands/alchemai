"""Parameter regression for binary VLE data, with confidence intervals.

Objectives (weights are standard deviations σ; the user's, or the labelled defaults of tool.DEFAULT_SIGMA):
  "bubble"  (default) — isobaric sets: S = Σ[(T_calc - T)/σT]² + Σ[(y_calc - y)/σy]², with T_calc, y_calc the bubble
            point at the measured (P, x); isothermal or P-T-x-y sets: same with (P_calc - P)/(σP_rel·P). This is the
            classic "Barker + y" least squares (Van Ness & Abbott 1982; Gmehling & Onken DECHEMA introduction).
  "ml"      — maximum likelihood with errors in all measured variables (Britt & Luecke 1973; Anderson, Abrams & Grens
            1978): unknown true T̂_k, x̂_k per point, P̂_k and ŷ_k from the bubble point at (T̂_k, x̂_k), and
            S = Σ[(T̂-T)/σT]² + [(x̂-x)/σx]² + [(P̂-P)/σP]² + [(ŷ-y)/σy]². Same form as Aspen Plus "Maximum-likelihood".
Covariance of the parameters: s²·(JᵀJ)⁻¹ at the optimum (Gauss-Newton), s² = S/(m - n), parameter block when latent
variables are present (Bard 1974, ch. 7); 95 % intervals with Student t(m - n). Points with x1 = 0 or 1 carry no
information on the binary parameters and are excluded from the fit (they enter the pure-component check).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from packages.vle.models import ln_gamma

MAX_NFEV_ML = 3000
INIT = {"NRTL": [(0.0, 0.0), (300.0, 300.0), (800.0, -200.0), (-200.0, 800.0), (1500.0, 500.0), (500.0, 1500.0)],
        "UNIQUAC": [(0.0, 0.0), (-150.0, -150.0), (150.0, -300.0), (-300.0, 150.0), (-500.0, 100.0), (100.0, -500.0)],
        "Wilson": [(0.0, 0.0), (-300.0, -300.0), (-800.0, 100.0), (100.0, -800.0), (-1200.0, -100.0), (-100.0, -1200.0)]}


class Data:
    """Flattened points of all sets. kind per point: 'isobaric' | 'isothermal' | 'PTxy'."""

    def __init__(self, sets: list[dict]):
        T, P, x, y, kind, setid = [], [], [], [], [], []
        for i, s in enumerate(sets):
            for q in s["points"]:
                T.append(q["T_K"]); P.append(q["P_Pa"]); x.append(q["x1"])
                y.append(np.nan if q.get("y1") is None else q["y1"])
                kind.append(s["kind"]); setid.append(i)
        self.T, self.P, self.x, self.y = (np.array(v, float) for v in (T, P, x, y))
        self.kind, self.set = np.array(kind), np.array(setid)
        self.inner = (self.x > 0.0) & (self.x < 1.0)
        self.has_y = np.isfinite(self.y)


def _params(names, theta, fixed):
    """theta → model parameters. With fixed['_T_ref'] the a_ij slots hold c_ij = a_ij + b_ij/T_ref."""
    p = {k: v for k, v in fixed.items() if k != "_T_ref"}
    for n, v in zip(names, theta):
        p[n] = float(v)
    if "_T_ref" in fixed:
        for ij in ("12", "21"):
            if "a" + ij in p:
                p["a" + ij] -= p["b" + ij] / fixed["_T_ref"]
    return p


def predict(system, model, p, d: Data, mask=None):
    """Bubble point at the measured coordinates: T_calc for isobaric points (at P, x), P_calc otherwise (at T, x)."""
    m = d.inner if mask is None else mask
    lng = lambda T, x: ln_gamma(model, T, x, p)
    Tc, Pc, yc = d.T.copy(), d.P.copy(), np.full_like(d.T, np.nan)
    ib = m & (d.kind == "isobaric")
    it = m & (d.kind != "isobaric")
    if ib.any():
        Tc[ib], yc[ib] = system.bubble_T(d.P[ib], d.x[ib], lng, T0=d.T[ib])
    if it.any():
        Pc[it], yc[it] = system.bubble_P(d.T[it], d.x[it], lng)
    return Tc, Pc, yc


def _res_bubble(theta, names, fixed, model, system, d, sig):
    p = _params(names, theta, fixed)
    Tc, Pc, yc = predict(system, model, p, d)
    m = d.inner
    ib = m & (d.kind == "isobaric")
    it = m & (d.kind != "isobaric")
    r = [(Tc[ib] - d.T[ib]) / sig["T_K"], (Pc[it] / d.P[it] - 1.0) / sig["P_rel"]]
    yy = m & d.has_y
    r.append((yc[yy] - d.y[yy]) / sig["y1"])
    out = np.concatenate(r)
    return np.where(np.isfinite(out), out, 1e3)


def _res_ml(z, npar, names, fixed, model, system, d, sig):
    m = d.inner
    n = int(m.sum())
    theta, Th, xh = z[:npar], z[npar:npar + n], z[npar + n:]
    p = _params(names, theta, fixed)
    Ph, yh = system.bubble_P(Th, xh, lambda T, x: ln_gamma(model, T, x, p))
    P, T, x, y, hy = d.P[m], d.T[m], d.x[m], d.y[m], d.has_y[m]
    r = np.concatenate([(Th - T) / sig["T_K"], (xh - x) / sig["x1"], (Ph / P - 1.0) / sig["P_rel"],
                        (yh[hy] - y[hy]) / sig["y1"]])
    return np.where(np.isfinite(r), r, 1e3)


def _gamma_start(model, fixed, names, d, system):
    """Cheap start: least squares on experimental ln γ (modified Raoult) from several initial guesses."""
    m = d.inner & d.has_y
    ps1, ps2 = system.psat[0](d.T[m]), system.psat[1](d.T[m])
    lg1 = np.log(d.y[m] * d.P[m] / (d.x[m] * ps1))
    lg2 = np.log((1 - d.y[m]) * d.P[m] / ((1 - d.x[m]) * ps2))
    T, x = d.T[m], d.x[m]

    def r(th):
        p = _params(names, th, fixed)
        a, b = ln_gamma(model, T, x, p)
        v = np.concatenate([a - lg1, b - lg2])
        return np.where(np.isfinite(v), v, 1e2)
    best = None
    for g in INIT[model]:
        th0 = np.zeros(len(names))
        th0[names.index("b12")], th0[names.index("b21")] = g
        if "alpha" in names:
            th0[names.index("alpha")] = 0.3
        if "_T_ref" in fixed:  # start from a_ij = 0
            th0[names.index("a12")], th0[names.index("a21")] = g[0] / fixed["_T_ref"], g[1] / fixed["_T_ref"]
        try:
            s = least_squares(r, th0, bounds=_bounds(names), x_scale="jac", max_nfev=400)
        except Exception:  # noqa: BLE001 - one bad start must not stop the others
            continue
        if best is None or s.cost < best.cost:
            best = s
    return best.x if best is not None else np.zeros(len(names))


def _bounds(names):
    lo = [-np.inf if n != "alpha" else 0.05 for n in names]
    hi = [np.inf if n != "alpha" else 1.0 for n in names]
    return lo, hi


def fit(system, model: str, d: Data, fixed: dict, names: list[str], objective: str, sig: dict) -> dict:
    """With a_ij in `names` the regression runs on c_ij = a_ij + b_ij/T_ref (T_ref = mean T of the fitted points),
    which removes most of the a-b correlation; values and covariance are mapped back linearly (a = c - b/T_ref)."""
    if "a12" not in names:
        return _fit(system, model, d, fixed, names, objective, sig)
    Tref = float(np.mean(d.T[d.inner]))
    n = len(names)
    L = np.eye(n)  # theta_real = L @ theta_internal
    for ij in ("12", "21"):
        L[names.index("a" + ij), names.index("b" + ij)] = -1.0 / Tref
    r = _fit(system, model, d, {**fixed, "_T_ref": Tref}, names, objective, sig)
    theta = L @ r["theta"]
    cov = L @ r["cov"] @ L.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(se, se)
    p = _params(names, theta, fixed)
    r.update(theta=theta, p=p, cov=cov, se=se, ci95=r["t_quantile"] * se, corr=corr, T_ref=Tref,
             cond_corr=float(np.linalg.cond(corr)) if np.all(np.isfinite(corr)) else None)
    return r


def _fit(system, model: str, d: Data, fixed: dict, names: list[str], objective: str, sig: dict) -> dict:
    if not (d.inner & d.has_y).any() and objective == "ml":
        raise ValueError("máxima verosimilitud necesita y medida")
    th0 = _gamma_start(model, fixed, names, d, system) if (d.inner & d.has_y).any() else np.zeros(len(names))
    lo, hi = _bounds(names)
    if objective == "bubble":
        s = least_squares(_res_bubble, th0, args=(names, fixed, model, system, d, sig), bounds=(lo, hi),
                          x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
        J, npar = s.jac, len(names)
        theta = s.x
        latent = None
    elif objective == "ml":
        m = d.inner
        n = int(m.sum())
        s0 = least_squares(_res_bubble, th0, args=(names, fixed, model, system, d, sig), bounds=(lo, hi),
                           x_scale="jac", max_nfev=600)
        z0 = np.concatenate([s0.x, d.T[m], d.x[m]])
        npar = len(names)
        rows = 3 * n + int(d.has_y[m].sum())
        sp = np.zeros((rows, npar + 2 * n), int)
        sp[:, :npar] = 1
        idx = np.arange(n)
        hy = np.where(d.has_y[m])[0]
        for blk, base in ((0, 0), (1, n), (2, 2 * n)):
            sp[base + idx, npar + idx] = 1
            sp[base + idx, npar + n + idx] = 1
        sp[3 * n + np.arange(len(hy)), npar + hy] = 1
        sp[3 * n + np.arange(len(hy)), npar + n + hy] = 1
        lo2 = np.concatenate([lo, np.full(n, 100.0), np.full(n, 1e-9)])
        hi2 = np.concatenate([hi, np.full(n, 1000.0), np.full(n, 1 - 1e-9)])
        z0 = np.clip(z0, lo2 + 1e-12, hi2 - 1e-12)
        s = least_squares(_res_ml, z0, args=(npar, names, fixed, model, system, d, sig), bounds=(lo2, hi2),
                          jac_sparsity=sp, x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=MAX_NFEV_ML)
        J = s.jac.toarray() if hasattr(s.jac, "toarray") else s.jac
        theta = s.x[:npar]
        latent = {"T_K": s.x[npar:npar + n], "x1": s.x[npar + n:]}
    else:
        raise ValueError("objective debe ser 'bubble' o 'ml'")
    m_res, n_var = len(s.fun), J.shape[1]
    dof = m_res - n_var
    S = float(np.sum(s.fun ** 2))
    s2 = S / dof if dof > 0 else np.nan
    JTJ = J.T @ J
    cov_full = np.linalg.pinv(JTJ) * s2
    cov = cov_full[:npar, :npar]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tq = float(student_t.ppf(0.975, dof)) if dof > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(se, se)
    return {"theta": theta, "names": names, "p": _params(names, theta, fixed), "cost": S, "dof": dof, "s2": s2,
            "se": se, "ci95": tq * se, "t_quantile": tq, "cov": cov, "corr": corr,
            "cond_corr": float(np.linalg.cond(corr)) if np.all(np.isfinite(corr)) else None,
            "success": bool(s.success), "message": str(s.message), "nfev": int(s.nfev), "latent": latent}
