"""Transport phenomena for sizing: pipe hydraulics (momentum) and heat-exchanger area (heat).

Pure functions on SI inputs, no simulator: they take fluid properties (from a solved DWSIM stream or by hand) and
return the numbers a sizing step needs. Correlations come from Caleb Bell's MIT-licensed `fluids` and `ht`:
Darcy friction factor by Clamond's exact solution of Colebrook (laminar: 64/Re), nominal pipe sizes from ASME B36.10M,
LMTD correction factor F for 1-N shell-and-tube by Fakheri, and Gnielinski for tube-side film coefficients.
"""
from __future__ import annotations

import math

STEEL_ROUGHNESS_M = 4.57e-5           # commercial steel (Moody chart value)
# Common nominal sizes: 1¼, 2½, 3½ and 5 in exist in B36.10M but are avoided in process piping.
COMMON_NPS = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
G = 9.80665


def pipe_dp(rho: float, mu: float, mass_flow_kg_s: float, D_m: float, L_m: float, roughness_m: float = STEEL_ROUGHNESS_M,
            elevation_change_m: float = 0.0, fittings_K: float = 0.0) -> dict:
    """Single-phase, incompressible pressure drop in a straight pipe (Darcy-Weisbach). Gas lines are valid while
    ΔP stays below ~10 % of the inlet pressure; beyond that, split the line or use a compressible model."""
    from fluids import friction_factor
    A = math.pi * D_m ** 2 / 4.0
    v = mass_flow_kg_s / rho / A
    Re = rho * v * D_m / mu
    f = 64.0 / Re if Re < 2300.0 else friction_factor(Re=Re, eD=roughness_m / D_m)
    dyn = 0.5 * rho * v ** 2
    dp_f = f * L_m / D_m * dyn
    dp_k = fittings_K * dyn
    dp_z = rho * G * elevation_change_m
    regime = "laminar" if Re < 2300.0 else ("transición" if Re < 4000.0 else "turbulento")
    return {"velocity_m_s": v, "Re": Re, "friction_factor_darcy": f, "regime": regime, "dP_friction_Pa": dp_f,
            "dP_fittings_Pa": dp_k, "dP_static_Pa": dp_z, "dP_total_Pa": dp_f + dp_k + dp_z,
            "dP_friction_per_100m_Pa": dp_f / L_m * 100.0 if L_m > 0 else None, "dynamic_pressure_Pa": dyn}


def size_pipe(rho: float, mu: float, mass_flow_kg_s: float, max_velocity_m_s: float, schedule: str = "40",
              roughness_m: float = STEEL_ROUGHNESS_M) -> dict:
    """Smallest common nominal pipe of the schedule whose velocity stays at or below the limit (ASME B36.10M)."""
    from fluids.piping import nearest_pipe
    q = mass_flow_kg_s / rho
    Di_min = math.sqrt(4.0 * q / (math.pi * max_velocity_m_s))
    for nps in COMMON_NPS:
        NPS, Di, Do, t = nearest_pipe(NPS=nps, schedule=schedule)
        if Di >= Di_min:
            break
    else:
        raise ValueError(f"needs D > {Di_min:.3f} m: above 36 in, split the flow into parallel lines")
    h = pipe_dp(rho, mu, mass_flow_kg_s, Di, 100.0, roughness_m)
    return {"NPS_in": NPS, "schedule": schedule, "inner_diameter_m": Di, "outer_diameter_m": Do, "wall_m": t,
            "min_inner_diameter_m": Di_min, "velocity_m_s": h["velocity_m_s"], "Re": h["Re"],
            "dP_friction_per_100m_Pa": h["dP_friction_Pa"]}


def lmtd(hot_in_K: float, hot_out_K: float, cold_in_K: float, cold_out_K: float) -> float:
    """Counter-current log-mean temperature difference."""
    d1, d2 = hot_in_K - cold_out_K, hot_out_K - cold_in_K
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"temperature cross: ΔT1 = {d1:.2f} K, ΔT2 = {d2:.2f} K (counter-current needs both > 0)")
    return d1 if abs(d1 - d2) < 1e-9 else (d1 - d2) / math.log(d1 / d2)


def hx_area(duty_W: float, hot_in_K: float, hot_out_K: float, cold_in_K: float, cold_out_K: float, U_W_m2K: float,
            shell_passes: int = 0) -> dict:
    """Area = Q / (U·F·LMTD). shell_passes = 0 → pure counter-current (F = 1); N ≥ 1 → 1-N shell with 2N tube passes."""
    lm = lmtd(hot_in_K, hot_out_K, cold_in_K, cold_out_K)
    F = 1.0
    if shell_passes >= 1:
        from ht import F_LMTD_Fakheri
        F = float(F_LMTD_Fakheri(Thi=hot_in_K, Tho=hot_out_K, Tci=cold_in_K, Tco=cold_out_K, shells=shell_passes))
        if not math.isfinite(F) or F <= 0:
            raise ValueError("F undefined for these temperatures with this shell arrangement: add shells or use counter-current")
    A = abs(duty_W) / (U_W_m2K * F * lm)
    out = {"LMTD_K": lm, "F": F, "area_m2": A, "UA_W_K": U_W_m2K * A, "U_W_m2K": U_W_m2K, "duty_kW": abs(duty_W) / 1e3}
    if shell_passes >= 1 and F < 0.75:
        out["warning"] = f"F = {F:.3f} < 0.75: diseño ineficiente o sensible; considerar más corazas"
    return out


def tube_film_coefficient(rho: float, mu: float, k: float, cp_J_kgK: float, velocity_m_s: float, D_m: float,
                          roughness_m: float = STEEL_ROUGHNESS_M) -> dict:
    """Tube-side film coefficient h = Nu·k/D. Turbulent: Gnielinski (valid 3000 < Re < 5e6, 0.5 < Pr < 2000);
    laminar fully developed, constant wall T: Nu = 3.66."""
    from fluids import friction_factor
    from ht.conv_internal import turbulent_Gnielinski
    Re = rho * velocity_m_s * D_m / mu
    Pr = cp_J_kgK * mu / k
    if Re < 2300.0:
        Nu, corr = 3.66, "laminar, T de pared constante (Nu = 3.66)"
    else:
        Nu, corr = turbulent_Gnielinski(Re=Re, Pr=Pr, fd=friction_factor(Re=Re, eD=roughness_m / D_m)), "Gnielinski"
    return {"Re": Re, "Pr": Pr, "Nu": Nu, "h_W_m2K": Nu * k / D_m, "correlation": corr}


def overall_U(h_inner_W_m2K: float, h_outer_W_m2K: float, D_inner_m: float, D_outer_m: float, k_wall_W_mK: float = 45.0,
              fouling_inner_m2K_W: float = 0.0, fouling_outer_m2K_W: float = 0.0) -> float:
    """U referred to the outer area, thin-wall cylindrical resistances in series."""
    ro = (1.0 / h_outer_W_m2K + fouling_outer_m2K_W + D_outer_m * math.log(D_outer_m / D_inner_m) / (2.0 * k_wall_W_mK)
          + fouling_inner_m2K_W * D_outer_m / D_inner_m + D_outer_m / (D_inner_m * h_inner_W_m2K))
    return 1.0 / ro
