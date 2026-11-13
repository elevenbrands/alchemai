"""Step-response identification: FOPDT (and SOPDT) by nonlinear least squares, cross-checked by two-point methods.

Data: an open-loop step test (controller in manual), input changed by Δu at t_step, output sampled y(t).
    FOPDT  y = y0 + K Δu (1 − e^{−(t−t_s−θ)/τ})                              for t > t_s + θ
    SOPDT  y = y0 + K Δu (1 − (τ1 e^{−t'/τ1} − τ2 e^{−t'/τ2})/(τ1 − τ2)),     t' = t − t_s − θ
Two-point methods (on the normalised response y_n = (y − y0)/(y_∞ − y0)):
    Smith (1972):                     t1 at 28.3 %, t2 at 63.2 %:  τ = 1.5 (t2 − t1),  θ = t2 − τ
    Sundaresan & Krishnaswamy (1977): t1 at 35.3 %, t2 at 85.3 %:  τ = 0.67 (t2 − t1), θ = 1.3 t1 − 0.29 t2
(times measured from the step). The least-squares fit starts from the Smith estimate; disagreement between the fit
and the two-point estimates is reported, not hidden (noise biases the two-point methods; a higher-order process
makes all of them approximations)."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t


def _fopdt(p, ts):
    K, tau, th = p
    tt = np.clip(ts - th, 0, None)
    return K * (1 - np.exp(-tt / tau))


def _sopdt(p, ts):
    K, t1, t2, th = p
    tt = np.clip(ts - th, 0, None)
    if abs(t1 - t2) < 1e-6 * max(t1, t2):
        tm = 0.5 * (t1 + t2)
        return K * (1 - (1 + tt / tm) * np.exp(-tt / tm))
    return K * (1 - (t1 * np.exp(-tt / t1) - t2 * np.exp(-tt / t2)) / (t1 - t2))


def _cross(ts, yn, level):
    """First time the normalised response reaches `level` (linear interpolation)."""
    i = np.nonzero(yn >= level)[0]
    if not len(i):
        return None
    i = i[0]
    if i == 0:
        return float(ts[0])
    return float(ts[i - 1] + (level - yn[i - 1]) * (ts[i] - ts[i - 1]) / (yn[i] - yn[i - 1]))


def two_point(ts: np.ndarray, yn: np.ndarray, gain: float) -> dict:
    out = {}
    a, b = _cross(ts, yn, 0.283), _cross(ts, yn, 0.632)
    if a is not None and b is not None:
        tau = 1.5 * (b - a)
        out["Smith"] = {"K": gain, "tau": tau, "theta": max(b - tau, 0.0), "t28_3": a, "t63_2": b}
    a, b = _cross(ts, yn, 0.353), _cross(ts, yn, 0.853)
    if a is not None and b is not None:
        out["Sundaresan_Krishnaswamy"] = {"K": gain, "tau": 0.67 * (b - a), "theta": max(1.3 * a - 0.29 * b, 0.0),
                                          "t35_3": a, "t85_3": b}
    return out


def _fit(fun, p0, lb, ub, ts, dy):
    r = least_squares(lambda p: fun(p, ts) - dy, p0, bounds=(lb, ub), x_scale="jac", max_nfev=5000)
    res = r.fun
    n, k = len(dy), len(p0)
    rss = float(res @ res)
    sst = float(((dy - dy.mean()) ** 2).sum())
    out = {"params": r.x, "rmse": math.sqrt(rss / n), "r2": 1 - rss / sst if sst > 0 else float("nan"),
           "max_abs_residual": float(np.max(np.abs(res))), "success": bool(r.success)}
    # parameter standard errors from the Jacobian (linearised, valid for small white noise)
    try:
        J = r.jac
        s2 = rss / max(n - k, 1)
        cov = np.linalg.inv(J.T @ J) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        q = float(student_t.ppf(0.975, max(n - k, 1)))
        out["ci95_halfwidth"] = q * se
    except np.linalg.LinAlgError:
        out["ci95_halfwidth"] = np.full(k, np.nan)
    # lag-1 autocorrelation of the residuals: near 0 for white noise, near 1 for systematic model mismatch
    rr = res - res.mean()
    out["residual_autocorr_lag1"] = float((rr[1:] @ rr[:-1]) / (rr @ rr)) if rr @ rr > 0 else 0.0
    return out


def identify(time_s: list[float], output: list[float], input_change: float, step_time_s: float,
             baseline_output: float | None = None, final_output: float | None = None) -> dict:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(output, dtype=float)
    if t.shape != y.shape or t.ndim != 1 or len(t) < 8:
        raise ValueError("time_s y output deben ser listas del mismo largo (al menos 8 puntos)")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s debe ser estrictamente creciente")
    if not math.isfinite(input_change) or input_change == 0:
        raise ValueError("input_change (Δu del escalón) es dato del usuario y debe ser ≠ 0")
    if not (t[0] <= step_time_s < t[-1]):
        raise ValueError("step_time_s debe estar dentro del intervalo de datos")
    assumptions, warnings = [], []
    pre = t < step_time_s
    if baseline_output is not None:
        y0 = float(baseline_output)
    elif pre.sum() >= 3:
        y0 = float(y[pre].mean())
        assumptions.append(f"y0 = promedio de los {int(pre.sum())} puntos antes del escalón ({y0:.6g})")
    else:
        y0 = float(y[0])
        assumptions.append(f"y0 = primer punto ({y0:.6g}): no hay datos suficientes antes del escalón")
    post = ~pre
    ts, yp = t[post] - step_time_s, y[post]
    dy = yp - y0
    n_tail = max(3, len(ts) // 10)
    if final_output is not None:
        yinf = float(final_output)
    else:
        yinf = float(yp[-n_tail:].mean())
        assumptions.append(f"y∞ = promedio del último 10 % de los datos ({yinf:.6g}), solo para los métodos de dos puntos "
                           "y el arranque del ajuste")
    span = yinf - y0
    if span == 0:
        raise ValueError("la salida no cambió: no hay respuesta al escalón que identificar")
    # data-quality diagnostics
    tail = slice(int(len(ts) * 0.8), None)
    slope = np.polyfit(ts[tail], yp[tail], 1)[0] if len(ts[tail]) >= 3 else 0.0
    drift = abs(slope) * (ts[-1] - ts[int(len(ts) * 0.8)]) / abs(span)
    if drift > 0.02:
        warnings.append(f"la salida sigue cambiando al final ({drift * 100:.1f} % de Δy en el último 20 % del registro): "
                        "no llegó a estado estacionario; K y τ son inciertos. Si nunca se estabiliza, el proceso puede ser "
                        "integrante (modelo k'·e^(−θs)/s)")
    yn = (yp - y0) / span
    if np.max(yn) > 1.05:
        warnings.append(f"sobrepaso de {100 * (np.max(yn) - 1):.0f} % sobre el valor final: respuesta subamortiguada; "
                        "un FOPDT/SOPDT no la representa")
    early = yn[: max(3, len(yn) // 5)]
    if np.min(early) < -0.05:
        warnings.append("la salida se mueve primero en sentido contrario: respuesta inversa (cero en el semiplano derecho); "
                        "el tiempo muerto ajustado la absorbe (regla de la mitad, Skogestad 2003)")
    K0 = span / input_change
    tp = two_point(ts, yn, K0)
    s = tp.get("Smith") or {"tau": ts[-1] / 5, "theta": 0.0}
    tau0, th0 = max(s["tau"], 1e-3 * ts[-1]), min(max(s["theta"], 0.0), 0.9 * ts[-1])
    T = ts[-1]
    f1 = _fit(_fopdt, [K0 * input_change, tau0, th0], [-np.inf, 1e-6 * T, 0], [np.inf, 100 * T, T], ts, dy)
    K, tau, th = f1["params"]
    K /= input_change
    ci = f1["ci95_halfwidth"]
    fop = {"K": K, "tau": tau, "theta": th, "theta_over_tau": th / tau,
           "ci95": {"K": ci[0] / abs(input_change), "tau": ci[1], "theta": ci[2]},
           "fit": {"r2": f1["r2"], "rmse": f1["rmse"], "rmse_pct_of_step": 100 * f1["rmse"] / abs(span),
                   "max_abs_residual": f1["max_abs_residual"], "residual_autocorr_lag1": f1["residual_autocorr_lag1"],
                   "converged": f1["success"]}}
    # SOPDT from a spread of the FOPDT time constant
    best = None
    for frac in (0.5, 0.3, 0.1):
        g = [K * input_change, tau * (1 - frac), tau * frac, max(th * 0.8, 0.0)]
        f2 = _fit(_sopdt, g, [-np.inf, 1e-6 * T, 1e-6 * T, 0], [np.inf, 100 * T, 100 * T, T], ts, dy)
        if best is None or f2["rmse"] < best["rmse"]:
            best = f2
    K2, a, b, th2 = best["params"]
    t1, t2 = max(a, b), min(a, b)
    sop = {"K": K2 / input_change, "tau1": t1, "tau2": t2, "theta": th2,
           "fit": {"r2": best["r2"], "rmse": best["rmse"], "rmse_pct_of_step": 100 * best["rmse"] / abs(span),
                   "max_abs_residual": best["max_abs_residual"],
                   "residual_autocorr_lag1": best["residual_autocorr_lag1"], "converged": best["success"]}}
    # cross-check FOPDT fit vs two-point methods
    cross = {}
    for name, v in tp.items():
        cross[name] = {"tau_rel_diff_pct": 100 * (v["tau"] - tau) / tau,
                       "theta_abs_diff_over_tau": (v["theta"] - th) / tau}
    agree = all(abs(c["tau_rel_diff_pct"]) <= 20 and abs(c["theta_abs_diff_over_tau"]) <= 0.1 for c in cross.values())
    if not agree:
        warnings.append("los métodos de dos puntos difieren del ajuste en más de 20 % en τ o 0.1τ en θ: ruido fuerte o "
                        "proceso de orden alto; revisa la gráfica antes de sintonizar")
    sop_better = sop["fit"]["rmse"] < 0.7 * fop["fit"]["rmse"] and t2 > 0.1 * t1
    rec = ("SOPDT (τ2 > θ: PID SIMC con τD = τ2 tiene sentido)" if sop_better and t2 > th2 else
           "SOPDT describe mejor los datos pero τ2 < θ: PI con el FOPDT es suficiente" if sop_better else
           "FOPDT (el SOPDT no mejora el ajuste de forma significativa)")
    return {"fopdt": fop, "sopdt": sop, "two_point": tp, "two_point_vs_fit": cross, "two_point_agrees": agree,
            "baseline_output": y0, "final_output_estimate": yinf, "input_change": input_change,
            "step_time_s": step_time_s, "n_points_after_step": int(post.sum()), "recommended_model": rec,
            "warnings": warnings, "assumptions": assumptions}
