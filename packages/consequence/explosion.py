"""Blast overpressure.

TNT equivalence (Crowl & Louvar ch. 6): m_TNT = η·m·ΔHc / E_TNT, E_TNT = 1120 cal/g = 4686 kJ/kg; scaled distance
ze = r / m_TNT^(1/3) (m/kg^(1/3)); side-on overpressure from the Kinney & Graham (1985) fit
  ps/pa = 1616 [1 + (ze/4.5)²] / sqrt([1 + (ze/0.048)²][1 + (ze/0.32)²][1 + (ze/1.35)²]).
η (explosion efficiency) is a user choice (Crowl & Louvar quote 1-10 %); it is never assumed here.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

from packages.consequence.props import P_ATM

E_TNT_J_KG = 4.686e6


def tnt_mass(m_fuel_kg: float, Hc_J_kg: float, efficiency: float) -> float:
    if not 0 < efficiency <= 1:
        raise ValueError("la eficiencia de explosión η va en (0, 1]")
    return efficiency * m_fuel_kg * Hc_J_kg / E_TNT_J_KG


def kinney_graham_ratio(ze: float) -> float:
    """ps/pa as a function of scaled distance ze (m/kg^(1/3))."""
    return 1616.0 * (1 + (ze / 4.5) ** 2) / math.sqrt((1 + (ze / 0.048) ** 2) * (1 + (ze / 0.32) ** 2)
                                                       * (1 + (ze / 1.35) ** 2))


def tnt_overpressure(m_tnt_kg: float, r_m: float, p_ambient: float = P_ATM) -> dict:
    ze = r_m / m_tnt_kg ** (1.0 / 3.0)
    ratio = kinney_graham_ratio(ze)
    return {"ze_m_kg13": ze, "ps_pa": ratio, "sobrepresion_Pa": ratio * p_ambient}


def tnt_distance_to(m_tnt_kg: float, p_target_Pa: float, p_ambient: float = P_ATM) -> float:
    """Distance (m) at which the TNT-equivalent overpressure falls to p_target (monotone in ze)."""
    target = p_target_Pa / p_ambient
    ze = brentq(lambda z: kinney_graham_ratio(z) - target, 1e-3, 1e4, xtol=1e-9)
    return ze * m_tnt_kg ** (1.0 / 3.0)


# --------------------------------------------------------------------------- TNO multi-energy (Díaz Alonso 2006 fits)
SRC_ME = ("TNO multi-energía, ajustes P' = C1·R'^C2 (tabla 4.3) e i' = C3·R'^C4 (tabla 4.4) de Díaz Alonso (2006), "
          "tesis doctoral, Univ. de Murcia (sobre las curvas del Yellow Book 1997); R' = r/(E/p0)^(1/3)")
# class: list of (R'_low, R'_high, C1, C2); the last interval is closed above.
_ME_P = {
    1: [(0.23, 0.6, 1e-2, 0.0), (0.6, 7, 6.40e-3, -0.97)],
    2: [(0.23, 0.7, 2e-2, 0.0), (0.7, 12, 1.32e-2, -0.98)],
    # class 3 far field: printed C1 = 6.05e-2 is an erratum (×2 jump at R' = 0.6, inconsistent with the thesis' own
    # table 4.6 and with Johansson's curves); 3.10e-2 / −0.97 is back-solved from table 4.6.
    3: [(0.23, 0.6, 5e-2, 0.0), (0.6, 30, 3.10e-2, -0.97)],
    4: [(0.23, 0.5, 1e-1, 0.0), (0.5, 70, 6.44e-2, -0.99)],
    5: [(0.23, 0.6, 2e-1, 0.0), (0.6, 90, 1.17e-1, -0.99)],
    6: [(0.23, 0.6, 5e-1, 0.0), (0.6, 100, 3.01e-1, -1.11)],
    7: [(0.23, 0.5, 1.0, 0.0), (0.5, 100, 4.06e-1, -1.20)],
    8: [(0.23, 0.5, 2.0, 0.0), (0.5, 1, 4.76e-1, -2.08), (1, 2, 4.67e-1, -1.58), (2, 100, 3.18e-1, -1.13)],
    9: [(0.23, 0.35, 5.0, 0.0), (0.35, 1, 4.87e-1, -2.03), (1, 2, 4.67e-1, -1.58), (2, 100, 3.18e-1, -1.13)],
    10: [(0.23, 1, 4.41e-1, -2.39), (1, 2, 4.67e-1, -1.58), (2, 100, 3.18e-1, -1.13)],
}
_ME_I = {
    1: [(0.23, 0.6, 4.41e-2, -0.20), (0.6, 7, 2.96e-2, -0.94)],
    2: [(0.23, 0.7, 5.22e-2, -0.27), (0.7, 12, 4.03e-2, -1.05)],
    3: [(0.23, 0.6, 8.74e-2, -0.20), (0.6, 30, 6.05e-2, -0.99)],
    4: [(0.23, 0.5, 1.4e-1, 0.0), (0.5, 70, 6.77e-2, -0.97)],
    5: [(0.23, 0.6, 1.25e-1, -0.26), (0.6, 90, 8.46e-2, -1.00)],
    6: [(0.23, 0.8, 1.28e-1, -0.45), (0.8, 100, 1.14e-1, -1.03)],
    7: [(0.23, 0.6, 1.98e-1, -0.49), (0.6, 100, 1.14e-1, -1.03)],
    8: [(0.23, 0.6, 1.66e-1, -0.90), (0.6, 100, 1.14e-1, -1.03)],
    9: [(0.23, 0.3, 1.11, 0.89), (0.3, 0.4, 3.08e-1, -1.08), (0.4, 0.8, 8.08e-2, -2.26), (0.8, 100, 1.14e-1, -1.03)],
    10: [(0.23, 0.3, 10.82, 1.14), (0.3, 0.4, 3.15e-1, -1.79), (0.4, 0.5, 1.30e-3, -7.52), (0.5, 100, 1.14e-1, -1.03)],
}
ME_ENERGY_DENSITY_J_M3 = 3.5e6   # stoichiometric hydrocarbon-air (TNO Yellow Book; C&L ex. 6-12)


def _piecewise(table: list, Rp: float) -> float | None:
    for k, (lo, hi, c, e) in enumerate(table):
        last = k == len(table) - 1
        if lo <= Rp < hi or (last and lo <= Rp <= hi):
            return c * Rp ** e
    return None


def me_scaled_overpressure(Rp: float, strength: int) -> float | None:
    """P' of a multi-energy charge of class 1-10 at scaled distance R' (None outside 0.23 ≤ R' ≤ range end). Class 4
    keeps its plateau until the far-field fit falls below it (continuity; the printed joint at 0.5 jumps ×1.28)."""
    if strength not in _ME_P:
        raise ValueError("clase de fuerza multi-energía: 1 a 10")
    if Rp < 0.23 and strength != 10:
        return _ME_P[strength][0][2]                 # inside the charge: the class plateau
    p = _piecewise(_ME_P[strength], Rp)
    if strength == 4 and p is not None:
        p = min(0.1, 6.44e-2 * Rp ** -0.99)
    return p


def me_scaled_impulse(Rp: float, strength: int) -> float | None:
    """i' = P'·tp' (Díaz Alonso eq. 5.22, twice the usual scaled impulse)."""
    return _piecewise(_ME_I[strength], Rp)


def multi_energy(E_J: float, strength: int, r_m: float, p0: float = P_ATM, c0: float = 340.0) -> dict:
    s = (E_J / p0) ** (1.0 / 3.0)
    Rp = r_m / s
    P = me_scaled_overpressure(Rp, strength)
    ip = me_scaled_impulse(Rp, strength) if Rp >= 0.23 else None
    out = {"R_escalada": Rp, "P_escalada": P, "sobrepresion_Pa": P * p0 if P is not None else None}
    if ip is not None and P:
        tp = ip / P * s / c0
        out.update({"tp_escalado": ip / P, "duracion_s": tp, "impulso_Pa_s": 0.5 * P * p0 * tp})
    if P is None:
        out["nota"] = "fuera del rango del ajuste"
    return out


def me_distance_to(E_J: float, strength: int, p_target_Pa: float, p0: float = P_ATM) -> float | None:
    """Farthest distance (m) with overpressure ≥ target (P' decreases with R' beyond the plateau)."""
    target = p_target_Pa / p0
    hi_R = _ME_P[strength][-1][1]
    if (me_scaled_overpressure(0.23, strength) or 0) < target:
        return None
    if (me_scaled_overpressure(hi_R, strength) or 0) >= target:
        return float("inf")
    import numpy as np
    xs = np.geomspace(0.23, hi_R, 600)
    vals = [me_scaled_overpressure(x, strength) - target for x in xs]
    k = max(i for i, v in enumerate(vals) if v >= 0)
    Rp = brentq(lambda x: me_scaled_overpressure(x, strength) - target, xs[k], xs[k + 1], xtol=1e-10) \
        if k < len(xs) - 1 else hi_R
    return Rp * (E_J / p0) ** (1.0 / 3.0)
