"""Gaussian plume and puff with Pasquill-Gifford stability (Crowl & Louvar ch. 5; Briggs 1974 dispersion
coefficients as tabulated by Turner/Griffiths), and the Britter-McQuaid dense-gas criterion.

The Gaussian model is for passive (neutrally buoyant) clouds. When the Britter-McQuaid Richardson-type criterion says
a ground-level release is dense, the Gaussian is not used: `britter_mcquaid_distance` (workbook curve fits) gives the
distances instead (M12b).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from packages.consequence.props import G, P_ATM, R

CLASSES = ("A", "B", "C", "D", "E", "F")
SRC_PG = ("Pasquill-Gifford con coeficientes de Briggs (1974, vía Griffiths 1994) para pluma: Crowl & Louvar "
          "4.ª ed., tabla 5-2 (rural/urbano)")
SRC_PUFF = "coeficientes de puff de Crowl & Louvar 4.ª ed., tabla 5-3 (σx = σy, σz; el libro avisa que se basan en pocos datos)"
X_RANGE_PLUME = (100.0, 10_000.0)   # m, range of the Briggs fits

# Briggs (1974) open-country (rural) and urban fits: sigma = a·x·(1 + b·x)^c   (x in m)
_BRIGGS = {
    "rural": {
        "A": ((0.22, 1e-4, -0.5), (0.20, 0.0, 1.0)),
        "B": ((0.16, 1e-4, -0.5), (0.12, 0.0, 1.0)),
        "C": ((0.11, 1e-4, -0.5), (0.08, 2e-4, -0.5)),
        "D": ((0.08, 1e-4, -0.5), (0.06, 1.5e-3, -0.5)),
        "E": ((0.06, 1e-4, -0.5), (0.03, 3e-4, -1.0)),
        "F": ((0.04, 1e-4, -0.5), (0.016, 3e-4, -1.0)),
    },
    "urban": {
        "A": ((0.32, 4e-4, -0.5), (0.24, 1e-3, 0.5)),
        "B": ((0.32, 4e-4, -0.5), (0.24, 1e-3, 0.5)),
        "C": ((0.22, 4e-4, -0.5), (0.20, 0.0, 1.0)),
        "D": ((0.16, 4e-4, -0.5), (0.14, 3e-4, -0.5)),
        "E": ((0.11, 4e-4, -0.5), (0.08, 1.5e-3, -0.5)),
        "F": ((0.11, 4e-4, -0.5), (0.08, 1.5e-3, -0.5)),
    },
}
# Puff: sigma_y = sigma_x = a x^b ; sigma_z = c x^d   (x = downwind distance of the puff centre, m)
_PUFF = {"A": (0.18, 0.92, 0.60, 0.75), "B": (0.14, 0.92, 0.53, 0.73), "C": (0.10, 0.92, 0.34, 0.71),
         "D": (0.06, 0.92, 0.15, 0.70), "E": (0.04, 0.92, 0.10, 0.65), "F": (0.02, 0.89, 0.05, 0.61)}

# Pasquill (1961) stability from 10 m wind speed and insolation / night cloud cover, as in C&L table 5-1 (bands from Turner 1970;
# the 4th-ed. copy read has corrupted row labels).
# Wind bands: u < 2 | 2 ≤ u < 3 | 3 ≤ u < 5 | 5 ≤ u < 6 | u ≥ 6  (lower bound inclusive).
_PASQUILL_DAY = {"fuerte": ("A", "A-B", "B", "C", "C"), "moderada": ("A-B", "B", "B-C", "C-D", "D"),
                 "ligera": ("B", "C", "C", "D", "D")}
_PASQUILL_NIGHT = {"nublado": ("F", "E", "D", "D", "D"), "despejado": ("F", "F", "E", "D", "D")}
_WIND_EDGES = (2.0, 3.0, 5.0, 6.0)


def stability_class(wind_speed_10m: float, insolation: str | None = None, night_cloud: str | None = None) -> str:
    """Pasquill class from the user's weather. insolation: fuerte | moderada | ligera (day); night_cloud: nublado
    (≥ 4/8 low cloud) | despejado (≤ 3/8). Overcast day or night → D (Pasquill)."""
    if wind_speed_10m < 0:
        raise ValueError("velocidad de viento negativa")
    band = sum(wind_speed_10m >= e for e in _WIND_EDGES)
    if insolation == "cubierto" or night_cloud == "cubierto":
        return "D"
    if insolation:
        return _PASQUILL_DAY[insolation][band]
    if night_cloud:
        return _PASQUILL_NIGHT[night_cloud][band]
    raise ValueError("da la insolación (día) o la nubosidad (noche), o la clase de estabilidad directamente")


def _split(cls: str) -> tuple[str, ...]:
    parts = tuple(p.strip().upper() for p in cls.replace("–", "-").split("-"))
    if not all(p in CLASSES for p in parts):
        raise ValueError(f"clase de estabilidad no válida: {cls}")
    return parts


def sigmas_plume(x: float | np.ndarray, cls: str, terrain: str = "rural") -> tuple:
    """σy, σz (m) for a continuous plume. Mixed classes (A-B…) average the two, as Crowl & Louvar advise."""
    x = np.asarray(x, dtype=float)
    sy = sz = 0.0
    parts = _split(cls)
    for p in parts:
        (a, b, c), (d, e, f) = _BRIGGS[terrain][p]
        sy = sy + a * x * (1 + b * x) ** c
        sz = sz + d * x * (1 + e * x) ** f
    return sy / len(parts), sz / len(parts)


def sigmas_puff(x: float | np.ndarray, cls: str) -> tuple:
    x = np.asarray(x, dtype=float)
    sy = sz = 0.0
    parts = _split(cls)
    for p in parts:
        a, b, c, d = _PUFF[p]
        sy = sy + a * x ** b
        sz = sz + c * x ** d
    return sy / len(parts), sz / len(parts)


def plume_concentration(Q_kg_s: float, u: float, x, y=0.0, z=0.0, H: float = 0.0, cls: str = "D",
                        terrain: str = "rural"):
    """Continuous release with ground reflection (C&L ec. de pluma con fuente elevada H y reflexión): kg/m³."""
    sy, sz = sigmas_plume(x, cls, terrain)
    return (Q_kg_s / (2 * math.pi * sy * sz * u) * np.exp(-0.5 * (np.asarray(y) / sy) ** 2)
            * (np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(-0.5 * ((z + H) / sz) ** 2)))


def puff_center_concentration(Qstar_kg: float, x, y=0.0, z=0.0, H: float = 0.0, cls: str = "D"):
    """Peak concentration at the passage of the puff centre (x = u·t), with ground reflection: kg/m³."""
    sy, sz = sigmas_puff(x, cls)
    return (Qstar_kg / ((2 * math.pi) ** 1.5 * sy * sy * sz) * np.exp(-0.5 * (np.asarray(y) / sy) ** 2)
            * (np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(-0.5 * ((z + H) / sz) ** 2)))


def puff_dose(Qstar_kg: float, u: float, x, y=0.0, z=0.0, H: float = 0.0, cls: str = "D"):
    """Time-integrated concentration ∫C dt (kg·s/m³) at a fixed point as the puff passes (σx = σy)."""
    sy, sz = sigmas_puff(x, cls)
    return (Qstar_kg / (2 * math.pi * sy * sz * u) * np.exp(-0.5 * (np.asarray(y) / sy) ** 2)
            * (np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(-0.5 * ((z + H) / sz) ** 2)))


def kgm3_to_ppm(c_kg_m3, M_kg_mol: float, T_K: float, P_Pa: float = P_ATM):
    return np.asarray(c_kg_m3) / M_kg_mol * R * T_K / P_Pa * 1e6


def ppm_to_kgm3(ppm: float, M_kg_mol: float, T_K: float, P_Pa: float = P_ATM) -> float:
    return ppm * 1e-6 * P_Pa * M_kg_mol / (R * T_K)


def dense_gas_check(M_kg_mol: float, T_release_K: float, T_ambient_K: float, wind_speed: float,
                    Q_kg_s: float | None = None, Qstar_kg: float | None = None, P_Pa: float = P_ATM,
                    rho_release: float | None = None) -> dict:
    """Britter & McQuaid (1988) Richardson-type criterion, as given by Crowl & Louvar ch. 5 (dense gas dispersion):
    continuous (g0 q0/(u³ D))^(1/3) ≥ 0.15 with D = sqrt(q0/u); instantaneous sqrt(g0 V0^(1/3))/u ≥ 0.20, where
    g0 = g(ρ0 − ρa)/ρa and q0, V0 the release volume rate / volume at ρ0."""
    rho_a = P_Pa * 0.028964 / (R * T_ambient_K)
    rho0 = rho_release if rho_release else P_Pa * M_kg_mol / (R * T_release_K)
    g0 = G * (rho0 - rho_a) / rho_a
    out = {"densidad_nube_kg_m3": rho0, "densidad_aire_kg_m3": rho_a, "g0_m_s2": g0,
           "fuente": "Britter & McQuaid (1988) Workbook on the dispersion of dense gases, criterio citado en Crowl & "
                     "Louvar cap. 5"}
    if g0 <= 0:
        out.update({"denso": False, "criterio": None,
                    "nota": "nube más ligera o igual que el aire: flotante o pasiva; el gaussiano es aplicable"})
        return out
    if Q_kg_s is not None:
        q0 = Q_kg_s / rho0
        D = math.sqrt(q0 / wind_speed)
        crit = (g0 * q0 / (wind_speed ** 3 * D)) ** (1.0 / 3.0)
        out.update({"tipo": "continua", "criterio": crit, "umbral": 0.15, "denso": crit >= 0.15, "D_m": D})
    elif Qstar_kg is not None:
        V0 = Qstar_kg / rho0
        crit = math.sqrt(g0 * V0 ** (1.0 / 3.0)) / wind_speed
        out.update({"tipo": "instantánea", "criterio": crit, "umbral": 0.20, "denso": crit >= 0.20, "V0_m3": V0})
    else:
        raise ValueError("da Q_kg_s (continua) o Qstar_kg (instantánea)")
    return out


def _last_crossing(f, x_lo: float, x_hi: float, n: int = 400) -> float | None:
    """Largest x in [x_lo, x_hi] where f(x) crosses 0 from above (f(x) = C(x) − C*). None if C < C* everywhere."""
    xs = np.geomspace(x_lo, x_hi, n)
    vals = np.array([f(x) for x in xs])
    above = vals >= 0
    if not above.any():
        return None
    k = int(np.nonzero(above)[0][-1])
    if k == n - 1:
        return float("inf")
    return float(brentq(f, xs[k], xs[k + 1], xtol=1e-6))


def distance_to_concentration(conc_of_x, threshold: float, x_lo: float = 1.0, x_hi: float = 100_000.0) -> float | None:
    return _last_crossing(lambda x: float(conc_of_x(x)) - threshold, x_lo, x_hi)


# --------------------------------------------------------------------------- Britter-McQuaid dense gas (workbook fits)
SRC_BM = ("Britter & McQuaid (1988) Workbook, ajustes de sus curvas impresos en Crowl & Louvar 2.ª ed. (2002) "
          "(leídos en la reproducción de la tesis de Souza, UFRJ); erratas de signo E1 (pluma 0.1) y E2 (puff 0.005) "
          "corregidas por continuidad, igual que la transcripción de CCPS")
_INF = float("inf")
# Cm/C0: [(alpha_low, alpha_high, slope, intercept)] with alpha_low < α ≤ alpha_high; β = slope·α + intercept
_BM_PLUME = {
    0.1: [(-_INF, -0.55, 0, 1.75), (-0.55, -0.14, 0.24, 1.88), (-0.14, 1.0, -0.50, 1.78)],
    0.05: [(-_INF, -0.68, 0, 1.92), (-0.68, -0.29, 0.36, 2.16), (-0.29, -0.18, 0, 2.06), (-0.18, 1.0, -0.56, 1.96)],
    0.02: [(-_INF, -0.69, 0, 2.08), (-0.69, -0.31, 0.45, 2.39), (-0.31, -0.16, 0, 2.25), (-0.16, 1.0, -0.54, 2.16)],
    0.01: [(-_INF, -0.70, 0, 2.25), (-0.70, -0.29, 0.49, 2.59), (-0.29, -0.20, 0, 2.45), (-0.20, 1.0, -0.52, 2.35)],
    0.005: [(-_INF, -0.67, 0, 2.40), (-0.67, -0.28, 0.59, 2.80), (-0.28, -0.15, 0, 2.63), (-0.15, 1.0, -0.49, 2.56)],
    0.002: [(-_INF, -0.69, 0, 2.6), (-0.69, -0.25, 0.39, 2.87), (-0.25, -0.13, 0, 2.77), (-0.13, 1.0, -0.50, 2.71)],
}
_BM_PUFF = {
    0.1: [(-_INF, -0.44, 0, 0.70), (-0.44, 0.43, 0.26, 0.81), (0.43, 1.0, 0, 0.93)],
    0.05: [(-_INF, -0.56, 0, 0.85), (-0.56, 0.31, 0.26, 1.0), (0.31, 1.0, -0.12, 1.12)],
    0.02: [(-_INF, -0.66, 0, 0.95), (-0.66, 0.32, 0.36, 1.19), (0.32, 1.0, -0.26, 1.38)],
    0.01: [(-_INF, -0.71, 0, 1.15), (-0.71, 0.37, 0.34, 1.39), (0.37, 1.0, -0.38, 1.66)],
    0.005: [(-_INF, -0.52, 0, 1.48), (-0.52, 0.24, 0.26, 1.62), (0.24, 1.0, -0.30, 1.75)],
    0.002: [(-_INF, 0.27, 0, 1.83), (0.27, 1.0, -0.32, 1.92)],
    0.001: [(-_INF, -0.10, 0, 2.075), (-0.10, 1.0, -0.27, 2.05)],
}


def _bm_beta(table: dict, ratio: float, alpha: float) -> float:
    def beta(r):
        for lo, hi, a, b in table[r]:
            if lo < alpha <= hi:
                return a * alpha + b
        raise ValueError(f"α = {alpha:.3f} fuera del rango de las curvas de Britter-McQuaid (α ≤ 1)")
    ks = sorted(table)
    if not ks[0] <= ratio <= ks[-1]:
        raise ValueError(f"Cm/C0 = {ratio:.4g} fuera de las curvas ({ks[0]} – {ks[-1]})")
    if ratio in table:
        return beta(ratio)
    lo = max(k for k in ks if k < ratio)
    hi = min(k for k in ks if k > ratio)
    w = (math.log10(ratio) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return beta(lo) + w * (beta(hi) - beta(lo))


def britter_mcquaid_distance(mode: str, rho0: float, T_ambient_K: float, wind_speed: float, conc_ratio: float,
                             Q_kg_s: float | None = None, Qstar_kg: float | None = None,
                             T_release_K: float | None = None, P_Pa: float = P_ATM) -> dict:
    """Downwind distance to the ground-level concentration ratio Cm/C0 for a dense release (continua: plume from
    Q_kg_s; instantanea: puff from Qstar_kg). With T_release ≠ T_ambient, the target ratio is corrected for the
    cloud's heat-up, Cm* = Cm/(Cm + (1 − Cm)·Ta/T0) (form given by secondary sources citing CCPS; not verified in a
    printed copy). β interpolated linearly in log(Cm/C0) between tabulated curves."""
    rho_a = P_Pa * 0.028964 / (R * T_ambient_K)
    g0 = G * (rho0 - rho_a) / rho_a
    if g0 <= 0:
        raise ValueError("la nube no es más densa que el aire")
    ratio = conc_ratio
    if T_release_K and abs(T_release_K - T_ambient_K) > 1e-6:
        ratio = conc_ratio / (conc_ratio + (1 - conc_ratio) * T_ambient_K / T_release_K)
    if mode == "continua":
        q0 = Q_kg_s / rho0
        D = math.sqrt(q0 / wind_speed)
        alpha = 0.2 * math.log10(g0 ** 2 * q0 / wind_speed ** 5)
        beta = _bm_beta(_BM_PLUME, ratio, alpha)
    else:
        V0 = Qstar_kg / rho0
        D = V0 ** (1.0 / 3.0)
        alpha = 0.5 * math.log10(g0 * D / wind_speed ** 2)
        beta = _bm_beta(_BM_PUFF, ratio, alpha)
    return {"g0_m_s2": g0, "D_m": D, "alpha": alpha, "beta": beta, "Cm_C0_corregida": ratio,
            "distancia_m": D * 10 ** beta, "fuente": SRC_BM}
