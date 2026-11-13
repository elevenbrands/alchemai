"""Source terms (Crowl & Louvar, *Chemical Process Safety*, ch. 4; Mackay & Matsugu 1973 for pool evaporation).

All SI. Each function returns a dict with the result, the equation used and its source; the caller adds weather and
inventory, which are always the user's.
"""
from __future__ import annotations

import math

from packages.consequence.props import G, P_ATM, R

SRC_CL = "Crowl & Louvar, Chemical Process Safety, 4.ª ed. (2020), cap. 4"


def _area(d: float) -> float:
    if d <= 0:
        raise ValueError("hole_diameter_m debe ser > 0")
    return math.pi * d * d / 4.0


def liquid_hole(hole_diameter_m: float, gauge_pressure_Pa: float, density_kg_m3: float, liquid_head_m: float = 0.0,
                Cd: float = 0.61, tank_area_m2: float | None = None) -> dict:
    """Liquid through a hole: Qm = A·Cd·sqrt(2ρ(Pg + ρ g hL)) (C&L 4th ed. eq. 4-7, with the liquid head of eq. 4-19).

    With `tank_area_m2` (constant Pg), the tank drains: Qm(t) falls linearly and the time to drain the head above the
    hole is t_e = (A_t/(Cd A g))·[sqrt(2(Pg/ρ + g hL0)) − sqrt(2 Pg/ρ)] (C&L 4th ed. eq. 4-20)."""
    A = _area(hole_diameter_m)
    dp = gauge_pressure_Pa + density_kg_m3 * G * liquid_head_m
    if dp <= 0:
        raise ValueError("sin fuerza impulsora: presión manométrica + carga de líquido ≤ 0")
    u = Cd * math.sqrt(2.0 * dp / density_kg_m3)
    out = {"modelo": "líquido por orificio (Bernoulli con coeficiente de descarga)", "fuente": f"{SRC_CL}, ec. 4-7 (y 4-19 con carga de líquido)",
           "area_m2": A, "Cd": Cd, "velocidad_m_s": u, "flujo_masico_kg_s": density_kg_m3 * A * u}
    if tank_area_m2 and liquid_head_m > 0:
        pg = max(gauge_pressure_Pa, 0.0) / density_kg_m3
        te = tank_area_m2 / (Cd * A * G) * (math.sqrt(2 * (pg + G * liquid_head_m)) - math.sqrt(2 * pg))
        out.update({"tiempo_vaciado_s": te, "masa_drenada_kg": density_kg_m3 * tank_area_m2 * liquid_head_m,
                    "fuente_vaciado": f"{SRC_CL}, ec. 4-20 (Pg constante)"})
    return out


def choked_pressure(P0_Pa: float, gamma: float) -> float:
    """Downstream pressure at which a gas jet chokes: P* = P0 (2/(γ+1))^(γ/(γ−1)) (C&L 4th ed. eq. 4-49)."""
    return P0_Pa * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def gas_hole(hole_diameter_m: float, P0_Pa: float, T0_K: float, M_kg_mol: float, gamma: float,
             P_ambient_Pa: float = P_ATM, Cd: float = 1.0) -> dict:
    """Gas/vapour through a hole, isentropic nozzle with discharge coefficient (C&L 4th ed. eq. 4-48 and 4-50).

    Choked when P_ambient ≤ P* (the boundary is included: both equations give the same flow there)."""
    if P0_Pa <= P_ambient_Pa:
        raise ValueError("P0 debe ser mayor que la presión ambiente")
    if gamma <= 1:
        raise ValueError("gamma debe ser > 1")
    A = _area(hole_diameter_m)
    p_star = choked_pressure(P0_Pa, gamma)
    choked = P_ambient_Pa <= p_star
    k = gamma * M_kg_mol / (R * T0_K)
    if choked:
        q = Cd * A * P0_Pa * math.sqrt(k * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))
        eq = "ec. 4-50 (flujo crítico, ahogado)"
    else:
        r = P_ambient_Pa / P0_Pa
        q = Cd * A * P0_Pa * math.sqrt(2.0 * M_kg_mol / (R * T0_K) * gamma / (gamma - 1.0)
                                        * (r ** (2.0 / gamma) - r ** ((gamma + 1.0) / gamma)))
        eq = "ec. 4-48 (subcrítico)"
    return {"modelo": "gas por orificio (tobera isentrópica)", "fuente": f"{SRC_CL}, {eq}", "area_m2": A, "Cd": Cd,
            "regimen": "ahogado (sónico)" if choked else "subsónico", "ahogado": choked,
            "presion_critica_Pa": p_star, "razon_presion_critica": p_star / P0_Pa,
            "flujo_masico_kg_s": q}


def flash_fraction_cp(Cp_J_kgK: float, T_K: float, Tb_K: float, Hvap_J_kg: float) -> dict:
    """Flashing fraction of a superheated liquid: fv = Cp (T − Tb)/ΔHv (C&L 4th ed. eq. 4-87) and the integrated form
    fv = 1 − exp(−Cp (T − Tb)/ΔHv) (C&L 4th ed. eq. 4-91)."""
    if T_K <= Tb_K:
        return {"fraccion_flash": 0.0, "fraccion_flash_integrada": 0.0, "nota": "T ≤ Tb: el líquido no flashea",
                "fuente": f"{SRC_CL}, ec. 4-87/4-91"}
    x = Cp_J_kgK * (T_K - Tb_K) / Hvap_J_kg
    return {"fraccion_flash": min(x, 1.0), "fraccion_flash_integrada": 1.0 - math.exp(-x),
            "fuente": f"{SRC_CL}, ec. 4-87 (Cp y ΔHv constantes) y 4-91 (integrada)"}


def flash_fraction_coolprop(fluid: str, T_K: float, P_storage_Pa: float | None = None,
                            P_ambient_Pa: float = P_ATM) -> dict:
    """Isenthalpic flash from storage (saturated liquid at T, or compressed liquid at P_storage) to ambient pressure,
    with CoolProp (MIT): fv = (h1 − hL(Pa)) / (hV(Pa) − hL(Pa)). Same energy balance as C&L eq. 4-87, without constant Cp."""
    import CoolProp.CoolProp as CP
    if P_storage_Pa is None:
        h1 = CP.PropsSI("H", "T", T_K, "Q", 0, fluid)
        P_storage_Pa = CP.PropsSI("P", "T", T_K, "Q", 0, fluid)
    else:
        h1 = CP.PropsSI("H", "T", T_K, "P", P_storage_Pa, fluid)
    hl = CP.PropsSI("H", "P", P_ambient_Pa, "Q", 0, fluid)
    hv = CP.PropsSI("H", "P", P_ambient_Pa, "Q", 1, fluid)
    tb = CP.PropsSI("T", "P", P_ambient_Pa, "Q", 0, fluid)
    x = (h1 - hl) / (hv - hl)
    return {"fraccion_flash": min(max(x, 0.0), 1.0), "T_ebullicion_ambiente_K": tb, "P_almacenamiento_Pa": P_storage_Pa,
            "fuente": "balance isentálpico (el de C&L ec. 4-87 sin Cp constante) con entalpías de CoolProp 8 (MIT)",
            "nota": "sin flash: líquido subenfriado a presión ambiente" if x <= 0 else ""}


def pool_evaporation_cl(M_kg_mol: float, Psat_Pa: float, T_liquid_K: float, area_m2: float,
                        K0_m_s: float = 0.0083, M0_kg_mol: float = 0.018) -> dict:
    """Evaporation of a non-boiling pool: Qm = M K A Psat/(Rg T_L), K = K0 (M0/M)^(1/3), K0 = 0.83 cm/s for water
    (C&L 4th ed. eq. 3-12 and 3-18). Still air; for wind use `pool_evaporation_mackay`."""
    K = K0_m_s * (M0_kg_mol / M_kg_mol) ** (1.0 / 3.0)
    q = M_kg_mol * K * area_m2 * Psat_Pa / (R * T_liquid_K)
    return {"modelo": "evaporación de charco (coeficiente de transferencia referido al agua)",
            "fuente": "Crowl & Louvar 4.ª ed., ec. 3-12/3-18 (= 4-106; K0 agua = 0.83 cm/s)", "K_m_s": K,
            "flujo_masico_kg_s": q}


def pool_evaporation_mackay(M_kg_mol: float, Psat_Pa: float, T_liquid_K: float, pool_diameter_m: float,
                            wind_speed_m_s: float, schmidt: float) -> dict:
    """Mackay & Matsugu (1973): K = 0.0048 u^(7/9) Z^(−1/9) Sc^(−2/3) (m/s; u at 10 m in m/s, Z pool diameter in m),
    Qm = M K A Psat/(R T_L) (CCPS CPQRA 2000 §2.1.2)."""
    K = 0.0048 * wind_speed_m_s ** (7.0 / 9.0) * pool_diameter_m ** (-1.0 / 9.0) * schmidt ** (-2.0 / 3.0)
    A = math.pi * pool_diameter_m ** 2 / 4.0
    q = M_kg_mol * K * A * Psat_Pa / (R * T_liquid_K)
    return {"modelo": "evaporación de charco con viento (Mackay y Matsugu 1973)",
            "fuente": "Mackay & Matsugu (1973), Can. J. Chem. Eng. 51:434; forma de CCPS CPQRA (2000)", "K_m_s": K,
            "area_m2": A, "flujo_masico_kg_s": q}


def boiling_pool(area_m2: float, T_ground_K: float, T_boil_K: float, Hvap_J_kg: float, k_soil_W_mK: float,
                 alpha_soil_m2_s: float, t_s: float, penetration_factor: float = 1.0) -> dict:
    """Boiling (cryogenic) pool on ground, heat from the soil only: q_g = k_s (T_g − T)/sqrt(π α_s t),
    Qm = q_g A/ΔHv (C&L 4th ed. eq. 4-107/4-108). Mass evaporated up to t: ∫Qm dt = 2 k_s (T_g − T) A sqrt(t/(π α_s))/ΔHv.
    Soil properties are the user's (or TNO Yellow Book table 3.5). penetration_factor = 8 for dry porous soil (liquid
    soaks in; TNO YB eq. 3.120), CR = 1 + 1.4 m − 3.0 m² for moist soil (eq. 3.124); 1 = impermeable ground (C&L)."""
    if t_s <= 0:
        raise ValueError("t_s debe ser > 0 (el flujo inicial es singular en t = 0)")
    dT = T_ground_K - T_boil_K
    if dT <= 0:
        raise ValueError("el suelo no está más caliente que el punto de ebullición: no es un charco en ebullición")
    qg = penetration_factor * k_soil_W_mK * dT / math.sqrt(math.pi * alpha_soil_m2_s * t_s)
    q = qg * area_m2 / Hvap_J_kg
    m = 2.0 * penetration_factor * k_soil_W_mK * dT * area_m2 * math.sqrt(t_s / (math.pi * alpha_soil_m2_s)) / Hvap_J_kg
    return {"modelo": "charco en ebullición calentado por el suelo (conducción semi-infinita)",
            "fuente": f"{SRC_CL}, ec. 4-107/4-108; TNO YB 2005 ec. 3.120-3.122", "factor_penetracion": penetration_factor,
            "flujo_calor_W_m2": qg, "flujo_masico_kg_s": q,
            "masa_evaporada_kg": m, "t_s": t_s}


# TNO Yellow Book CPR 14E (2005) table 3.5: λs (W/m K), as (m²/s)
SOILS_TNO = {"hormigon_aislante": (0.207, 2.5e-7), "hormigon_ligero": (0.418, 2.5e-7), "hormigon_pesado": (1.3, 5.9e-7),
             "escoria": (0.7, 4.2e-7), "subsuelo_promedio_8pct_humedad": (0.9, 4.3e-7),
             "subsuelo_arenoso_seco": (0.3, 2.0e-7), "arena_humeda_8pct_arcilla": (0.6, 3.3e-7), "madera": (0.2, 1.6e-7),
             "grava": (2.5, 11e-7), "acero_al_carbono": (46.0, 128e-7)}
