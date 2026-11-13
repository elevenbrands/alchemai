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
