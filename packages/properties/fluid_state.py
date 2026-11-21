"""Thermodynamic state of a PURE fluid from any two independent properties (CoolProp reference equations of state).

Why: closed-system and steam-table problems need the internal energy, the specific volume and states fixed by pairs such
as (v, u), (P, v), (T, v) or (P, h) — a rigid tank that is heated, a throttle, an adiabatic turbine. A flowsheet stream
gives T, P and vapour fraction only; this module is the steam-table lookup, for water (IAPWS-95) and any CoolProp fluid.

Reference state: CoolProp's default for each fluid (water: u = 0 and s = 0 for the saturated liquid at the triple point,
as in the usual steam tables; refrigerants: IIR). Absolute h, u, s of two different sources may differ by a constant —
only DIFFERENCES between states of the same call are physically meaningful, and the tool reports them.
"""
from __future__ import annotations

import math

_KEYS = {"T_K": "T", "P_Pa": "P", "h_J_kg": "Hmass", "s_J_kgK": "Smass", "u_J_kg": "Umass", "rho_kg_m3": "Dmass",
         "quality": "Q"}
_PAIRS_OK = "T_K, P_Pa, h_J_kg, s_J_kgK, u_J_kg, v_m3_kg (o rho_kg_m3), quality"


def _one(CP, fluid: str, spec: dict) -> dict:
    given = {k: float(v) for k, v in spec.items() if k not in ("label", "mass_kg", "mass_flow_kg_s") and v is not None}
    if "v_m3_kg" in given:
        if given["v_m3_kg"] <= 0:
            raise ValueError("v_m3_kg debe ser positivo")
        given["rho_kg_m3"] = 1.0 / given.pop("v_m3_kg")
    unknown = set(given) - set(_KEYS)
    if unknown:
        raise ValueError(f"propiedades no reconocidas {sorted(unknown)}; usa dos de: {_PAIRS_OK}")
    if len(given) != 2:
        raise ValueError(f"cada estado necesita exactamente DOS propiedades independientes (se dieron {len(given)}): {_PAIRS_OK}")
    (k1, v1), (k2, v2) = given.items()
    if {k1, k2} == {"T_K", "P_Pa"}:
        psat = None
        try:
            if CP.PropsSI("Ttriple", fluid) < given["T_K"] < CP.PropsSI("Tcrit", fluid):
                psat = CP.PropsSI("P", "T", given["T_K"], "Q", 0, fluid)
        except Exception:  # noqa: BLE001
            psat = None
        if psat is not None and abs(given["P_Pa"] / psat - 1.0) < 1e-6:
            raise ValueError("T y P caen sobre la curva de saturación: no fijan el estado; da quality, v, h, u o s junto con T o P")
    get = lambda out: float(CP.PropsSI(out, _KEYS[k1], v1, _KEYS[k2], v2, fluid))
    T, P, rho = get("T"), get("P"), get("Dmass")
    h, u, s = get("Hmass"), get("Umass"), get("Smass")
    q = get("Q")
    phase = CP.PhaseSI(_KEYS[k1], v1, _KEYS[k2], v2, fluid)
    st = {"T_K": T, "P_Pa": P, "phase": phase, "v_m3_kg": 1.0 / rho, "rho_kg_m3": rho, "h_J_kg": h, "u_J_kg": u,
          "s_J_kgK": s, "quality": q if 0.0 <= q <= 1.0 else None}
    two_phase = st["quality"] is not None and phase == "twophase"
    if not two_phase:
        for name, key in (("cp_J_kgK", "Cpmass"), ("cv_J_kgK", "Cvmass")):
            try:
                st[name] = get(key)
            except Exception:  # noqa: BLE001
                st[name] = None
    Tc, Pc = CP.PropsSI("Tcrit", fluid), CP.PropsSI("Pcrit", fluid)
    if P < Pc and P >= CP.PropsSI("ptriple", fluid):
        sat = lambda out, x: float(CP.PropsSI(out, "P", P, "Q", x, fluid))
        st["saturation_at_P"] = {"T_sat_K": sat("T", 0), "v_f_m3_kg": 1 / sat("Dmass", 0), "v_g_m3_kg": 1 / sat("Dmass", 1),
                                 "h_f_J_kg": sat("Hmass", 0), "h_g_J_kg": sat("Hmass", 1), "u_f_J_kg": sat("Umass", 0),
                                 "u_g_J_kg": sat("Umass", 1), "s_f_J_kgK": sat("Smass", 0), "s_g_J_kgK": sat("Smass", 1)}
        if not two_phase:
            st["superheat_K" if T > st["saturation_at_P"]["T_sat_K"] else "subcooling_K"] = abs(T - st["saturation_at_P"]["T_sat_K"])
    if T < Tc and T > CP.PropsSI("Ttriple", fluid):
        st["P_sat_at_T_Pa"] = float(CP.PropsSI("P", "T", T, "Q", 0, fluid))
    st["check_h_minus_u_minus_Pv_J_kg"] = h - u - P / rho
    if spec.get("label") is not None:
        st["label"] = spec["label"]
    for k in ("mass_kg", "mass_flow_kg_s"):
        if spec.get(k) is not None:
            m = float(spec[k])
            st[k] = m
            unit = "J" if k == "mass_kg" else "W"
            st.update({f"H_{unit}": m * h, f"U_{unit}": m * u, ("V_m3" if k == "mass_kg" else "V_m3_s"): m / rho})
            if two_phase:
                st["vapor_" + k], st["liquid_" + k] = m * q, m * (1 - q)
    return st


def fluid_state(fluid: str, states: list[dict]) -> dict:
    import CoolProp.CoolProp as CP
    if not isinstance(states, list) or not states:
        raise ValueError("states debe ser una lista con al menos un estado {dos propiedades}")
    try:
        CP.PropsSI("Tcrit", fluid)
    except Exception as ex:  # noqa: BLE001
        raise ValueError(f"CoolProp no reconoce el fluido {fluid!r} (usa nombres como Water, Ammonia, R134a, CarbonDioxide, "
                         f"Nitrogen, Methane, n-Butane): {ex}") from None
    out_states = []
    for i, spec in enumerate(states):
        try:
            out_states.append(_one(CP, fluid, dict(spec)))
        except ValueError as ex:
            raise ValueError(f"states[{i}]: {ex}") from None
    diffs = []
    for i in range(1, len(out_states)):
        a, b = out_states[i - 1], out_states[i]
        d = {"from": a.get("label", i - 1), "to": b.get("label", i), "delta_h_J_kg": b["h_J_kg"] - a["h_J_kg"],
             "delta_u_J_kg": b["u_J_kg"] - a["u_J_kg"], "delta_s_J_kgK": b["s_J_kgK"] - a["s_J_kgK"],
             "delta_v_m3_kg": b["v_m3_kg"] - a["v_m3_kg"], "delta_T_K": b["T_K"] - a["T_K"], "delta_P_Pa": b["P_Pa"] - a["P_Pa"]}
        for k, unit in (("mass_kg", "J"), ("mass_flow_kg_s", "W")):
            if a.get(k) is not None and b.get(k) is not None and math.isclose(a[k], b[k], rel_tol=1e-12):
                d[f"delta_H_{unit}"] = a[k] * d["delta_h_J_kg"]
                d[f"delta_U_{unit}"] = a[k] * d["delta_u_J_kg"]
        diffs.append(d)
    worst = max(abs(s["check_h_minus_u_minus_Pv_J_kg"]) / max(abs(s["h_J_kg"]), 1e3) for s in out_states)
    return {"ok": worst < 1e-8, "fluid": fluid, "critical_point": {"T_K": float(CP.PropsSI("Tcrit", fluid)),
                                                                   "P_Pa": float(CP.PropsSI("Pcrit", fluid))},
            "molar_mass_kg_mol": float(CP.PropsSI("M", fluid)), "states": out_states, "consecutive_differences": diffs,
            "reference_state": "la de CoolProp para el fluido (agua: u = 0 y s = 0 en el líquido saturado del punto triple, "
                               "como en las tablas de vapor; refrigerantes: IIR). Solo las DIFERENCIAS entre estados son "
                               "comparables con otra fuente",
            "method": f"ecuación de estado de referencia de CoolProp {__import__('CoolProp').__version__} para {fluid}",
            "evidence": {"checks": {"max_rel_h_minus_u_minus_Pv": worst}, "verified": worst < 1e-8,
                         "meaning": "h = u + P·v se cumple en cada estado; certifica la coherencia de la ecuación de "
                                    "estado, no que el fluido real sea puro"},
            "not_included": ["mezclas (usa independent_flash o un flowsheet)", "propiedades de transporte (compound_properties)"]}
