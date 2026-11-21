"""Mechanical energy balance (extended Bernoulli) between two points of an incompressible stream.

    (P2 − P1)/rho + (alpha2·u2² − alpha1·u1²)/2 + g·(z2 − z1) + F = W_on_fluid / m        [J/kg]

F is the friction loss (≥ 0), W_on_fluid the shaft power delivered TO the fluid (pump > 0, turbine < 0) and m the mass
flow. Velocities come from the volumetric flow and the diameters, are given directly, or are zero at a large free
surface. Exactly one quantity is left as the unknown. No friction correlation is used here: F is the user's (or zero,
stated); pipe_hydraulics / pipe_network compute it from roughness and fittings.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

G = 9.80665
_UNKNOWNS = ("P1_Pa", "P2_Pa", "z1_m", "z2_m", "shaft_power_W", "friction_loss_J_kg", "volumetric_flow_m3_s")


def _velocity(pt: dict, Q: float | None, label: str) -> tuple[float, str]:
    if pt.get("velocity_m_s") is not None:
        return float(pt["velocity_m_s"]), "dada"
    if pt.get("diameter_m") is not None or pt.get("area_m2") is not None:
        A = float(pt["area_m2"]) if pt.get("area_m2") is not None else math.pi * float(pt["diameter_m"]) ** 2 / 4.0
        if A <= 0:
            raise ValueError(f"{label}: el área debe ser positiva")
        if Q is None:
            raise ValueError(f"{label}: con diámetro o área hace falta volumetric_flow_m3_s (o mass_flow_kg_s)")
        return Q / A, "Q/A"
    if pt.get("large_surface"):
        return 0.0, "superficie libre grande (u ≈ 0)"
    raise ValueError(f"{label}: da velocity_m_s, diameter_m, area_m2 o large_surface: true")


def balance(density_kg_m3: float, point_1: dict, point_2: dict, solve_for: str, volumetric_flow_m3_s: float | None = None,
            mass_flow_kg_s: float | None = None, shaft_power_W: float | None = None,
            friction_loss_J_kg: float | None = None, efficiency: float | None = None, g_m_s2: float = G,
            kinetic_energy_factor: float = 1.0) -> dict:
    rho = float(density_kg_m3)
    if rho <= 0:
        raise ValueError("density_kg_m3 debe ser positiva")
    if solve_for not in _UNKNOWNS:
        raise ValueError(f"solve_for debe ser uno de {_UNKNOWNS}")
    if volumetric_flow_m3_s is not None and mass_flow_kg_s is not None:
        raise ValueError("da volumetric_flow_m3_s o mass_flow_kg_s, no ambos")
    Q = float(volumetric_flow_m3_s) if volumetric_flow_m3_s is not None else (
        float(mass_flow_kg_s) / rho if mass_flow_kg_s is not None else None)
    p1, p2 = dict(point_1), dict(point_2)
    assumptions = []
    vals = {"P1_Pa": p1.get("P_Pa"), "P2_Pa": p2.get("P_Pa"), "z1_m": p1.get("z_m"), "z2_m": p2.get("z_m"),
            "shaft_power_W": shaft_power_W, "friction_loss_J_kg": friction_loss_J_kg, "volumetric_flow_m3_s": Q}
    if vals[solve_for] is not None:
        raise ValueError(f"{solve_for} es la incógnita: no la des también como dato")
    if solve_for != "friction_loss_J_kg" and vals["friction_loss_J_kg"] is None:
        vals["friction_loss_J_kg"] = 0.0
        assumptions.append("pérdidas por fricción F = 0 (no se dio friction_loss_J_kg)")
    if solve_for != "shaft_power_W" and vals["shaft_power_W"] is None:
        vals["shaft_power_W"] = 0.0
        assumptions.append("sin bomba ni turbina: trabajo de flecha = 0 (no se dio shaft_power_W)")
    for k in ("z1_m", "z2_m"):
        if k != solve_for and vals[k] is None:
            vals[k] = 0.0
            assumptions.append(f"{k} = 0 (no se dio la elevación)")
    missing = [k for k, v in vals.items() if v is None and k != solve_for]
    if missing:
        raise ValueError(f"faltan datos: {missing} (solo {solve_for} puede quedar sin valor)")
    a = float(kinetic_energy_factor)

    def terms(v: dict) -> dict:
        q = v["volumetric_flow_m3_s"]
        u1, _ = _velocity(p1, q, "point_1")
        u2, _ = _velocity(p2, q, "point_2")
        pressure = (v["P2_Pa"] - v["P1_Pa"]) / rho
        kinetic = a * (u2 ** 2 - u1 ** 2) / 2.0
        potential = g_m_s2 * (v["z2_m"] - v["z1_m"])
        return {"u1": u1, "u2": u2, "pressure_J_kg": pressure, "kinetic_J_kg": kinetic, "potential_J_kg": potential,
                "friction_J_kg": v["friction_loss_J_kg"]}

    def residual(x: float) -> float:
        v = dict(vals)
        v[solve_for] = x
        t = terms(v)
        m = rho * v["volumetric_flow_m3_s"] if v["volumetric_flow_m3_s"] else None
        w = v["shaft_power_W"] / m if (v["shaft_power_W"] and m) else 0.0
        if v["shaft_power_W"] and not m:
            raise ValueError("con shaft_power_W hace falta el flujo (volumetric_flow_m3_s o mass_flow_kg_s)")
        return t["pressure_J_kg"] + t["kinetic_J_kg"] + t["potential_J_kg"] + t["friction_J_kg"] - w

    if solve_for == "volumetric_flow_m3_s":
        f = lambda lq: residual(math.exp(lq))
        grid = [math.log(1e-12) + i * (math.log(1e6) - math.log(1e-12)) / 720 for i in range(721)]
        fv = [f(x) for x in grid]
        roots = [math.exp(brentq(f, grid[i], grid[i + 1], xtol=1e-14)) for i in range(720) if fv[i] * fv[i + 1] < 0]
        if not roots:
            return {"ok": False, "error": "no hay un caudal positivo que cierre el balance con estos datos (revisa signos: "
                                          "la energía disponible debe superar las pérdidas)"}
        x = roots[0]
        extra = {"other_roots_m3_s": roots[1:]} if len(roots) > 1 else {}
    else:
        if solve_for == "shaft_power_W" and not Q:
            raise ValueError("para shaft_power_W hace falta el flujo (volumetric_flow_m3_s o mass_flow_kg_s)")
        r0, r1 = residual(0.0), residual(1.0)                     # the balance is linear in every other unknown
        if r1 == r0:
            raise ValueError(f"el balance no depende de {solve_for} con estos datos")
        x = -r0 / (r1 - r0)
        extra = {}
    vals[solve_for] = x
    t = terms(vals)
    m = rho * vals["volumetric_flow_m3_s"] if vals["volumetric_flow_m3_s"] else None
    w = vals["shaft_power_W"] / m if (vals["shaft_power_W"] and m) else 0.0
    out = {"ok": True, "solved_for": solve_for, "value": x, "density_kg_m3": rho,
           "volumetric_flow_m3_s": vals["volumetric_flow_m3_s"], "mass_flow_kg_s": m,
           "velocities_m_s": {"point_1": t["u1"], "point_2": t["u2"]},
           "terms_J_kg": {"pressure (P2−P1)/rho": t["pressure_J_kg"], "kinetic (u2²−u1²)/2": t["kinetic_J_kg"],
                          "potential g(z2−z1)": t["potential_J_kg"], "friction F": t["friction_J_kg"],
                          "shaft work on fluid W/m": w},
           "shaft_power_on_fluid_W": vals["shaft_power_W"], "assumptions": assumptions, "warnings": [], **extra,
           "sign_convention": "shaft_power_W > 0 = potencia entregada AL fluido (bomba); < 0 = extraída (turbina)",
           "not_included": ["fricción calculada (usa pipe_hydraulics/pipe_network y pásala como friction_loss_J_kg)",
                            "flujo compresible", "cambios de temperatura"]}
    if vals["friction_loss_J_kg"] < 0:
        out["warnings"].append("la pérdida por fricción resulta negativa: los datos son incompatibles con el sentido del flujo o falta una bomba")
    if efficiency is not None and vals["shaft_power_W"]:
        e = float(efficiency)
        if not 0 < e <= 1:
            raise ValueError("efficiency es una fracción entre 0 y 1")
        out["drive_power_W"] = vals["shaft_power_W"] / e if vals["shaft_power_W"] > 0 else vals["shaft_power_W"] * e
        out["efficiency"] = e
    res = t["pressure_J_kg"] + t["kinetic_J_kg"] + t["potential_J_kg"] + t["friction_J_kg"] - w
    scale = max(abs(v) for v in (t["pressure_J_kg"], t["kinetic_J_kg"], t["potential_J_kg"], t["friction_J_kg"], w, 1e-12))
    out["evidence"] = {"checks": {"balance_residual_rel": abs(res) / scale}, "verified": abs(res) / scale < 1e-9,
                       "meaning": "el balance cierra con los datos dados; no certifica la fricción ni que el flujo sea incompresible"}
    out["ok"] = out["evidence"]["verified"]
    return out
