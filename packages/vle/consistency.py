"""Thermodynamic consistency tests for one binary VLE data set (constant T or constant P).

Experimental activity coefficients: γ_i = y_i Φ_i P / (x_i Psat_i) (Φ_i = 1 with an ideal gas). Points with x1 = 0 or 1
are left to the pure-component check.

  area        Redlich & Kister (1948) area test: D = 100·|∫₀¹ ln(γ1/γ2) dx1| / ∫₀¹ |ln(γ1/γ2)| dx1 on a cubic smoothing
              spline of the data (scipy UnivariateSpline, s = n), as in VLizard / Zbytovský et al. (ACS Omega 2025).
              Criterion D ≤ 10. Isobaric sets add Herington (1951): J = 150·(Tmax - Tmin)/Tmin, consistent if D - J < 10.
  point       Fredenslund, Gmehling & Rasmussen (1977) point test: Barker's method with g^E/RT = x1x2 Σ_k a_k L_k(2x1-1)
              (shifted Legendre, order 4 by default) fitted to P (isothermal) or T (isobaric) ONLY; y is predicted.
              Consistent if the mean |y_calc - y_exp| ≤ 0.01.
  van_ness    Van Ness (1995, Pure Appl. Chem. 67, 859) direct test: g^E/RT (same Legendre basis) fitted to minimise
              Σ[δ ln(γ1/γ2)]²; RMS of the residuals gives the index 1 (≤ 0.025) … 10 (> 0.225); 1 is excellent, 5 doubtful,
              10 unacceptable.
  model_dev   Deviation check "of Van Ness" as used in recent J. Chem. Eng. Data papers (e.g. Hartanto et al. 2025):
              bubble P at the measured (T, x) with the FITTED activity model, ΔP = mean 100|ΔP|/P and Δy = mean 100|Δy|;
              consistent if both < 1.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares


def shifted_legendre(n: int, u: np.ndarray) -> list[np.ndarray]:
    """L_0..L_n of argument u = 2x1 - 1 (ordinary Legendre on [-1, 1]); derivatives d/dx1 alongside."""
    P = [np.ones_like(u), u.copy()]
    for k in range(1, n):
        P.append(((2 * k + 1) * u * P[k] - k * P[k - 1]) / (k + 1))
    dP = [np.zeros_like(u), np.ones_like(u)]
    for k in range(1, n):
        dP.append(((2 * k + 1) * (P[k] + u * dP[k]) - k * dP[k - 1]) / (k + 1))
    return P[:n + 1], [2.0 * d for d in dP[:n + 1]]  # chain rule du/dx1 = 2


def legendre_ln_gamma(a: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ln γ from g = g^E/RT = x1 x2 Σ a_k L_k: ln γ1 = g + x2 dg/dx1, ln γ2 = g - x1 dg/dx1."""
    x2 = 1 - x1
    L, dL = shifted_legendre(len(a) - 1, 2 * x1 - 1)
    s = sum(ak * l for ak, l in zip(a, L))
    ds = sum(ak * d for ak, d in zip(a, dL))
    g = x1 * x2 * s
    dg = (x2 - x1) * s + x1 * x2 * ds
    return g + x2 * dg, g - x1 * dg


def experimental_gammas(system, T, P, x1, y1):
    ps1, ps2 = system.psat[0](T), system.psat[1](T)
    F1, F2 = system.vapor.phis(T, P, y1, ps1, ps2)
    return y1 * F1 * P / (x1 * ps1), (1 - y1) * F2 * P / ((1 - x1) * ps2)


def area_test(system, T, P, x1, y1, isobaric: bool) -> dict:
    g1, g2 = experimental_gammas(system, T, P, x1, y1)
    o = np.argsort(x1)
    xs, f = x1[o], np.log(g1 / g2)[o]
    if len(np.unique(xs)) < len(xs):  # spline needs strictly increasing x: average duplicates
        u = np.unique(xs)
        f = np.array([f[xs == v].mean() for v in u])
        xs = u
    if len(xs) < 4:
        return {"aplicable": False, "motivo": "la spline cúbica necesita ≥ 4 puntos con 0 < x1 < 1"}
    sp = UnivariateSpline(xs, f)
    dif = abs(quad(sp, 0, 1, limit=200)[0])
    tot = quad(lambda z: abs(sp(z)), 0, 1, limit=200)[0]
    D = 100 * dif / tot if tot > 0 else float("nan")
    out = {"aplicable": True, "D": D, "criterio": "D ≤ 10", "consistente": bool(D <= 10.0),
           "cobertura_x1": [float(xs.min()), float(xs.max())],
           "ln_g1_g2_vs_x1": [{"x1": float(a), "ln_g1_g2": float(b)} for a, b in zip(xs, f)]}
    if isobaric:
        J = 150 * (T.max() - T.min()) / T.min()
        out["herington"] = {"J": float(J), "D_menos_J": float(D - J), "criterio": "D − J < 10",
                            "consistente": bool(D - J < 10.0)}
    if xs.min() > 0.1 or xs.max() < 0.9:
        out["aviso"] = "los datos no cubren bien los extremos de x1; la integral depende de la extrapolación"
    return out


def _legendre_bubble(system, a, T, P, x1, isobaric):
    lng = lambda TT, xx: legendre_ln_gamma(a, xx)
    if isobaric:
        return system.bubble_T(P, x1, lng, T0=T)
    Pc, yc = system.bubble_P(T, x1, lng)
    return Pc, yc


def point_test(system, T, P, x1, y1, isobaric: bool, order: int = 4) -> dict:
    n = len(x1)
    if n < order + 3:
        return {"aplicable": False, "motivo": f"Legendre de orden {order} necesita ≥ {order + 3} puntos"}

    def res(a):
        c, _ = _legendre_bubble(system, a, T, P, x1, isobaric)
        r = (c - T) if isobaric else (c / P - 1) * 100
        return np.where(np.isfinite(r), r, 1e3)
    s = least_squares(res, np.zeros(order + 1), x_scale="jac", max_nfev=2000)
    c, yc = _legendre_bubble(system, s.x, T, P, x1, isobaric)
    dy = yc - y1
    signs = np.sign(dy[np.abs(dy) > 1e-12])
    runs = int(1 + np.sum(signs[1:] != signs[:-1])) if len(signs) else 0
    out = {"aplicable": True, "orden_legendre": order, "dy_medio": float(np.mean(np.abs(dy))),
           "criterio": "media |Δy| ≤ 0.01", "consistente": bool(np.mean(np.abs(dy)) <= 0.01),
           "rachas_de_signo_dy": runs,
           "residuos": [{"x1": float(a), "dy": float(b)} for a, b in zip(x1, dy)]}
    if isobaric:
        out["dT_medio_K"] = float(np.mean(np.abs(c - T)))
    else:
        out["dP_medio_pct"] = float(np.mean(np.abs(c / P - 1)) * 100)
    return out


def van_ness_direct(system, T, P, x1, y1, order: int = 4) -> dict:
    g1, g2 = experimental_gammas(system, T, P, x1, y1)
    f = np.log(g1 / g2)
    if len(x1) < order + 2:
        return {"aplicable": False, "motivo": f"Legendre de orden {order} necesita ≥ {order + 2} puntos"}
    s = least_squares(lambda a: np.subtract(*legendre_ln_gamma(a, x1)) - f, np.zeros(order + 1))
    r = np.subtract(*legendre_ln_gamma(s.x, x1)) - f
    rms = float(np.sqrt(np.mean(r ** 2)))
    idx = int(min(10, max(1, math.ceil(rms / 0.025))))
    return {"aplicable": True, "rms_delta_ln_g1_g2": rms, "indice": idx,
            "escala": "1 excelente … 5 dudoso … 10 inaceptable (Van Ness 1995, intervalos de 0.025)",
            "residuos": [{"x1": float(a), "delta_ln_g1_g2": float(b)} for a, b in zip(x1, r)]}


def model_deviation(system, lng, T, P, x1, y1) -> dict:
    Pc, yc = system.bubble_P(T, x1, lng)
    dP = float(np.mean(np.abs(Pc / P - 1)) * 100)
    dy = float(np.mean(np.abs(yc - y1)) * 100)
    return {"dP_pct": dP, "dy_pct": dy, "criterio": "ΔP < 1 y Δy < 1 (en %)", "consistente": bool(dP < 1 and dy < 1)}


def pure_component_check(system, sets_pure: list[dict]) -> list[dict]:
    out = []
    for q in sets_pure:
        i = 0 if q["x1"] >= 1.0 else 1
        ps = system.psat[i]
        out.append({"componente": ps.name, "T_K": q["T_K"], "P_medida_Pa": q["P_Pa"],
                    "Psat_modelo_Pa": float(ps(q["T_K"])), "desv_pct": float((q["P_Pa"] / ps(q["T_K"]) - 1) * 100),
                    "Tsat_modelo_K": float(ps.Tsat(q["P_Pa"])), "desv_T_K": float(q["T_K"] - ps.Tsat(q["P_Pa"]))})
    return out
