"""Binary activity-coefficient models, vectorised over data points, in the SAME matrix convention as `thermo`
(tau[i][j], 0-based), so that `thermo.nrtl.NRTL`, `thermo.uniquac.UNIQUAC` and `thermo.wilson.Wilson` can re-evaluate
any fitted set independently (see `thermo_ln_gamma`).

  model    temperature dependence (i != j)                         energy parameter reported
  NRTL     tau_ij = a_ij + b_ij/T ; G_ij = exp(-alpha tau_ij)          Δg_ij = R·b_ij  (Renon & Prausnitz 1968)
  UNIQUAC  tau_ij = exp(a_ij + b_ij/T)                                  Δu_ij = -R·b_ij (Abrams & Prausnitz 1975)
  Wilson   Λ_ij = (V_j/V_i)·exp(a_ij + b_ij/T)                          Δλ_ij = -R·b_ij (Wilson 1964)

With a_ij = 0 (the default) these are the classic temperature-independent energy parameters of the DECHEMA Chemistry
Data Series (Gmehling & Onken): tau_12 = Δg_12/RT for NRTL, tau_12 = exp(-Δu_12/RT) for UNIQUAC, Λ_12 = (V_2/V_1)
exp(-Δλ_12/RT) for Wilson. UNIQUAC uses z = 10.

Excess enthalpy (M15c): h^E = −R·T²·∂(g^E/RT)/∂T at constant x (Gibbs-Helmholtz), analytic for the three models with
d(a + b/T)/dT = −b/T² (`excess_enthalpy`); `thermo_excess_enthalpy` is the same quantity from thermo's GibbsExcess.HE().
"""
from __future__ import annotations

import numpy as np

R = 8.314462618  # J/(mol K), CODATA 2018
CAL = 4.184      # J per thermochemical cal
MODELS = ("NRTL", "UNIQUAC", "Wilson")


def _pair(a12: float, a21: float, T: np.ndarray, b12: float, b21: float) -> tuple[np.ndarray, np.ndarray]:
    return a12 + b12 / T, a21 + b21 / T


def ln_gamma(model: str, T, x1, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """ln γ1, ln γ2 at temperatures T [K] and liquid fractions x1 (arrays of the same shape).
    p: {b12, b21, a12=0, a21=0, alpha (NRTL), r (2), q (2) (UNIQUAC), V (2, m³/mol) (Wilson)}."""
    T = np.asarray(T, float)
    x1 = np.asarray(x1, float)
    x2 = 1.0 - x1
    e12, e21 = _pair(p.get("a12", 0.0), p.get("a21", 0.0), T, p["b12"], p["b21"])
    if model == "NRTL":
        al = p["alpha"]
        t12, t21 = e12, e21
        G12, G21 = np.exp(-al * t12), np.exp(-al * t21)
        d1, d2 = x1 + x2 * G21, x2 + x1 * G12
        l1 = x2 ** 2 * (t21 * (G21 / d1) ** 2 + t12 * G12 / d2 ** 2)
        l2 = x1 ** 2 * (t12 * (G12 / d2) ** 2 + t21 * G21 / d1 ** 2)
        return l1, l2
    if model == "UNIQUAC":
        r1, r2 = p["r"]
        q1, q2 = p["q"]
        t12, t21 = np.exp(e12), np.exp(e21)
        z = 10.0
        sr = x1 * r1 + x2 * r2
        sq = x1 * q1 + x2 * q2
        ph1, ph2 = x1 * r1 / sr, x2 * r2 / sr
        th1, th2 = x1 * q1 / sq, x2 * q2 / sq
        l1_, l2_ = z / 2 * (r1 - q1) - (r1 - 1), z / 2 * (r2 - q2) - (r2 - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            c1 = np.log(ph1 / x1) + z / 2 * q1 * np.log(th1 / ph1) + l1_ - ph1 / x1 * (x1 * l1_ + x2 * l2_)
            c2 = np.log(ph2 / x2) + z / 2 * q2 * np.log(th2 / ph2) + l2_ - ph2 / x2 * (x1 * l1_ + x2 * l2_)
        # pure-component limits of the combinatorial part (x_i → 1 gives 0)
        c1 = np.where(x1 >= 1.0, 0.0, c1)
        c2 = np.where(x2 >= 1.0, 0.0, c2)
        if np.any(x1 <= 0.0):
            c1 = np.where(x1 <= 0.0, np.log(r1 / r2) + z / 2 * q1 * np.log(q1 * r2 / (q2 * r1)) + l1_ - r1 / r2 * l2_, c1)
        if np.any(x2 <= 0.0):
            c2 = np.where(x2 <= 0.0, np.log(r2 / r1) + z / 2 * q2 * np.log(q2 * r1 / (q1 * r2)) + l2_ - r2 / r1 * l1_, c2)
        s1 = th1 + th2 * t21
        s2 = th2 + th1 * t12
        res1 = q1 * (1 - np.log(s1) - th1 / s1 - th2 * t12 / s2)
        res2 = q2 * (1 - np.log(s2) - th2 / s2 - th1 * t21 / s1)
        return c1 + res1, c2 + res2
    if model == "Wilson":
        V1, V2 = p["V"]
        L12 = V2 / V1 * np.exp(e12)
        L21 = V1 / V2 * np.exp(e21)
        d1, d2 = x1 + L12 * x2, x2 + L21 * x1
        k = L12 / d1 - L21 / d2
        return -np.log(d1) + x2 * k, -np.log(d2) - x1 * k
    raise ValueError(f"modelo desconocido: {model}")


def excess_enthalpy(model: str, T, x1, p: dict) -> np.ndarray:
    """h^E [J/mol] = −R·T²·∂(g^E/RT)/∂T at temperatures T [K] and liquid fractions x1 (same parameters as ln_gamma).
    Only the energy part depends on T: the UNIQUAC combinatorial term and the Wilson volume ratio drop out."""
    T = np.asarray(T, float)
    x1 = np.asarray(x1, float)
    x2 = 1.0 - x1
    e12, e21 = _pair(p.get("a12", 0.0), p.get("a21", 0.0), T, p["b12"], p["b21"])
    d12, d21 = -p["b12"] / T ** 2, -p["b21"] / T ** 2          # d e_ij / dT
    if model == "NRTL":
        al = p["alpha"]
        G12, G21 = np.exp(-al * e12), np.exp(-al * e21)
        D1, D2 = x1 + x2 * G21, x2 + x1 * G12
        # d/dT of tau·G/D with G' = −alpha·G·tau', D' = x_other·G'
        t21 = d21 * G21 * (1 - al * e21) / D1 + e21 * G21 * al * x2 * G21 * d21 / D1 ** 2
        t12 = d12 * G12 * (1 - al * e12) / D2 + e12 * G12 * al * x1 * G12 * d12 / D2 ** 2
        dge = x1 * x2 * (t21 + t12)
    elif model == "UNIQUAC":
        q1, q2 = p["q"]
        sq = x1 * q1 + x2 * q2
        th1, th2 = x1 * q1 / sq, x2 * q2 / sq
        t12, t21 = np.exp(e12), np.exp(e21)
        dge = -q1 * x1 * th2 * t21 * d21 / (th1 + th2 * t21) - q2 * x2 * th1 * t12 * d12 / (th2 + th1 * t12)
    elif model == "Wilson":
        V1, V2 = p["V"]
        L12, L21 = V2 / V1 * np.exp(e12), V1 / V2 * np.exp(e21)
        dge = -x1 * x2 * L12 * d12 / (x1 + L12 * x2) - x2 * x1 * L21 * d21 / (x2 + L21 * x1)
    else:
        raise ValueError(f"modelo desconocido: {model}")
    return -R * T ** 2 * dge


def _thermo_ge(model: str, T: float, x1: float, p: dict):
    xs = [float(x1), 1.0 - float(x1)]
    a = [[0.0, p.get("a12", 0.0)], [p.get("a21", 0.0), 0.0]]
    b = [[0.0, p["b12"]], [p["b21"], 0.0]]
    if model == "NRTL":
        from thermo.nrtl import NRTL
        al = p["alpha"]
        ge = NRTL(T=float(T), xs=xs, tau_as=a, tau_bs=b, alpha_cs=[[0.0, al], [al, 0.0]])
    elif model == "UNIQUAC":
        from thermo.uniquac import UNIQUAC
        ge = UNIQUAC(T=float(T), xs=xs, rs=list(p["r"]), qs=list(p["q"]), tau_as=a, tau_bs=b)
    elif model == "Wilson":
        from thermo.wilson import Wilson
        V1, V2 = p["V"]
        aw = [[0.0, a[0][1] + np.log(V2 / V1)], [a[1][0] + np.log(V1 / V2), 0.0]]
        ge = Wilson(T=float(T), xs=xs, lambda_as=aw, lambda_bs=b)
    else:
        raise ValueError(model)
    return ge


def thermo_ln_gamma(model: str, T: float, x1: float, p: dict) -> tuple[float, float]:
    """Same quantity computed by `thermo`'s own GibbsExcess classes (independent implementation)."""
    g = _thermo_ge(model, T, x1, p).gammas()
    return float(np.log(g[0])), float(np.log(g[1]))


def thermo_excess_enthalpy(model: str, T: float, x1: float, p: dict) -> float:
    """h^E [J/mol] from thermo's GibbsExcess.HE() (independent implementation)."""
    return float(_thermo_ge(model, T, x1, p).HE())


def energy_parameters(model: str, p: dict) -> dict:
    """Δg/Δu/Δλ (J/mol and cal/mol) for the reported table; sign convention in the module docstring."""
    sign = 1.0 if model == "NRTL" else -1.0
    name = {"NRTL": "Δg", "UNIQUAC": "Δu", "Wilson": "Δλ"}[model]
    out = {}
    for ij in ("12", "21"):
        J = sign * R * p["b" + ij]
        out[f"{name}{ij}_J_mol"] = J
        out[f"{name}{ij}_cal_mol"] = J / CAL
    return out


def b_from_energy(model: str, d12_J: float, d21_J: float) -> tuple[float, float]:
    sign = 1.0 if model == "NRTL" else -1.0
    return sign * d12_J / R, sign * d21_J / R
