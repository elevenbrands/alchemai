"""Effect models: probits and dose (Crowl & Louvar 4th ed. table 2-5, after Eisenberg et al. 1975 / Lees 1986; PGS 1,
2003 Dutch revision of the TNO Green Book CPR 16E, read in Dutch: the 1992 English edition was not available).

Probability = Φ(Y − 5). Probit constants for toxics are substance-specific and come from the user (or from a
labelled example). End points (kW/m², kPa, ppm) are always the user's or the company's criterion.
"""
from __future__ import annotations

import math

# name: (k1, k2, variable description, source)
PROBITS = {
    "quemaduras_fatales_eisenberg": (-14.9, 2.56, "V = t·I^(4/3)/1e4, I en W/m², t en s",
                                     "Eisenberg et al. (1975): Crowl & Louvar 4.ª ed. tabla 2-5; PGS 1 ec. 4.19"),
    "quemaduras_fatales_hidrocarburos_pgs1": (-36.38, 2.56, "V = t·q^(4/3), q en W/m², t en s (fuego de "
                                              "hidrocarburo, sin ropa protectora)", "PGS 1 (2003) deel 1A ec. 4.5"),
    "quemaduras_2o_grado_pgs1": (-43.14, 3.0186, "V = t·q^(4/3), q en W/m², t en s", "PGS 1 (2003) deel 1A ec. 4.7"),
    "muerte_hemorragia_pulmonar": (-77.1, 6.91, "V = p° en N/m²", "Eisenberg et al. (1975), Crowl & Louvar tabla 2-5"),
    "ruptura_timpano": (-15.6, 1.93, "V = p° en N/m²", "Eisenberg et al. (1975), Crowl & Louvar tabla 2-5"),
    "ruptura_timpano_pgs1": (-12.9, 1.54, "V = Ps en Pa", "PGS 1 (2003) deel 2A ec. 12"),
    "dano_estructural": (-23.8, 2.92, "V = p° en N/m²", "Eisenberg et al. (1975), Crowl & Louvar tabla 2-5"),
    "rotura_vidrios": (-18.1, 2.79, "V = p° en N/m²", "Eisenberg et al. (1975), Crowl & Louvar tabla 2-5"),
    "rotura_ventanas_pre1975_pgs1": (-11.97, 2.12, "V = Ps en Pa", "PGS 1 (2003) deel 2B ec. 70"),
    "rotura_ventanas_post1975_pgs1": (-16.58, 2.53, "V = Ps en Pa", "PGS 1 (2003) deel 2B ec. 71"),
}
THERMAL = ("quemaduras_fatales_eisenberg", "quemaduras_fatales_hidrocarburos_pgs1", "quemaduras_2o_grado_pgs1")


def probability_from_probit(Y: float) -> float:
    return 0.5 * (1.0 + math.erf((Y - 5.0) / math.sqrt(2.0)))


def probit_from_probability(P: float) -> float:
    from scipy.stats import norm
    return 5.0 + float(norm.ppf(P))


def thermal_probit(q_kW_m2: float, t_s: float, model: str = "quemaduras_fatales_eisenberg") -> dict:
    k1, k2, _, src = PROBITS[model]
    q = q_kW_m2 * 1000.0
    V = t_s * q ** (4.0 / 3.0) / (1e4 if model == "quemaduras_fatales_eisenberg" else 1.0)
    Y = k1 + k2 * math.log(V)
    return {"modelo": model, "Y": Y, "probabilidad": probability_from_probit(Y), "fuente": src,
            "dosis_termica_(W/m2)^(4/3)·s": t_s * q ** (4.0 / 3.0)}


def overpressure_probit(p_Pa: float, model: str) -> dict:
    k1, k2, _, src = PROBITS[model]
    Y = k1 + k2 * math.log(p_Pa)
    return {"modelo": model, "Y": Y, "probabilidad": probability_from_probit(Y), "fuente": src}


def toxic_probit(conc: float, t: float, a: float, b: float, n: float) -> dict:
    """Y = a + b ln(C^n t), with C and t in the units the constants were fitted with (user's)."""
    Y = a + b * math.log(conc ** n * t)
    return {"Y": Y, "probabilidad": probability_from_probit(Y)}


def flux_for_probability(P: float, t_s: float, model: str = "quemaduras_fatales_eisenberg") -> float:
    """Heat flux (kW/m²) giving probability P after t_s seconds of exposure (inverse thermal probit)."""
    k1, k2, _, _ = PROBITS[model]
    V = math.exp((probit_from_probability(P) - k1) / k2)
    scale = 1e4 if model == "quemaduras_fatales_eisenberg" else 1.0
    return (V * scale / t_s) ** 0.75 / 1000.0


def overpressure_for_probability(P: float, model: str) -> float:
    k1, k2, _, _ = PROBITS[model]
    return math.exp((probit_from_probability(P) - k1) / k2)
