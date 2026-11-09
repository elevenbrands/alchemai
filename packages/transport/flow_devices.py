"""Flow devices and non-ideal lines (bloque M4a): relief valves (API 520 Part I / API 526), control valves
(IEC 60534-2-1), fittings (Darby 3-K, Hooper 2-K), compressible gas lines (isothermal) and two-phase pressure drop.

The correlations come from Caleb Bell's MIT-licensed `fluids`. Each result carries `evidence` recomputed here by an
independent route: the closed-form API/IEC equations in their printed SI units, the isothermal momentum ODE
integrated numerically, and the single-phase limits of the two-phase correlations. All pressures are absolute Pa
unless the name ends in `_Pa_g` (gauge).
"""
from __future__ import annotations

import difflib
import math

from packages.transport.hydraulics import STEEL_ROUGHNESS_M

ATM = 101325.0
R = 8.314462618


def _darcy_f(Re: float, eD: float) -> float:
    from fluids import friction_factor
    return 64.0 / Re if Re < 2300.0 else float(friction_factor(Re=Re, eD=eD))


# ---------------------------------------------------------------- relief valves (API 520 Part I, API 526 orifices)
def _api526(area_m2: float) -> dict:
    from fluids.safety_valve import API526_A, API526_letters
    n = max(1, math.ceil(area_m2 / API526_A[-1]))
    per = area_m2 / n
    for letter, a in zip(API526_letters, API526_A):
        if a >= per * (1 - 1e-12):
            return {"orifice": letter, "orifice_area_mm2": a * 1e6, "valves": n}
    raise AssertionError("unreachable")


def _api_C(k: float) -> float:
    return 0.03948 * math.sqrt(k * (2.0 / (k + 1.0)) ** ((k + 1.0) / (k - 1.0)))


def _api_F2(k: float, r: float) -> float:
    return math.sqrt(k / (k - 1.0) * r ** (2.0 / k) * (1.0 - r ** ((k - 1.0) / k)) / (1.0 - r))


def relief_valve(service: str, relieving_rate_kg_s: float, set_pressure_Pa_g: float, overpressure_fraction: float,
                 back_pressure_Pa_g: float, T_K: float | None = None, MW: float | None = None, Z: float | None = None,
                 k: float | None = None, density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None,
                 valve_type: str = "conventional", rupture_disk_upstream: bool = False, Kd: float | None = None) -> dict:
    from fluids import safety_valve as sv
    if service not in ("gas", "steam", "liquid"):
        raise ValueError("service must be 'gas', 'steam' or 'liquid'")
    if valve_type not in ("conventional", "balanced_bellows"):
        raise ValueError("valve_type must be 'conventional' or 'balanced_bellows'")
    P1 = set_pressure_Pa_g * (1.0 + overpressure_fraction) + ATM
    P2 = back_pressure_Pa_g + ATM
    Pset = set_pressure_Pa_g + ATM
    Kc = 0.9 if rupture_disk_upstream else 1.0
    W, kPa = relieving_rate_kg_s * 3600.0, 1e-3                  # API SI equations: W kg/h, P kPa, A mm²
    bp_frac = back_pressure_Pa_g / set_pressure_Pa_g
    out: dict = {"service": service, "relieving_pressure_Pa": P1, "back_pressure_Pa": P2, "Kc": Kc,
                 "back_pressure_fraction_of_set": bp_frac, "warnings": []}
    if valve_type == "conventional" and bp_frac > 0.10:
        out["warnings"].append(f"contrapresión {bp_frac:.0%} del set > 10 %: una válvula convencional pierde capacidad y "
                               "estabilidad; usar fuelle balanceado o piloto (API 520 Parte I)")
    if service in ("gas", "steam"):
        Kd = 0.975 if Kd is None else Kd
        Kb = sv.API520_B(Pset, P2, overpressure_fraction) if valve_type == "balanced_bellows" else 1.0
    if service == "gas":
        if None in (T_K, MW, Z, k):
            raise ValueError("gas service needs T_K, MW, Z and k (Cp/Cv) at relieving conditions")
        r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        critical = P2 / P1 <= r_crit
        A = sv.API520_A_g(m=relieving_rate_kg_s, T=T_K, Z=Z, MW=MW, k=k, P1=P1, P2=P2, Kd=Kd, Kb=Kb, Kc=Kc)
        if critical:
            A_chk = W / (_api_C(k) * Kd * P1 * kPa * Kb * Kc) * math.sqrt(T_K * Z / MW)
        else:
            A_chk = 17.9 * W / (_api_F2(k, P2 / P1) * Kd * Kc) * math.sqrt(Z * T_K / (MW * P1 * kPa * (P1 - P2) * kPa))
        out.update(flow="crítico" if critical else "subcrítico", critical_pressure_ratio=r_crit, Kd=Kd, Kb=Kb)
        check = "API 520 Parte I ec. 9 (crítico, C de k) / ec. 19 (subcrítico, F2)"
    elif service == "steam":
        if T_K is None:
            raise ValueError("steam service needs T_K (relieving temperature; saturated or superheated)")
        A = sv.API520_A_steam(m=relieving_rate_kg_s, T=T_K, P1=P1, Kd=Kd, Kb=Kb, Kc=Kc)
        KN, KSH = sv.API520_N(P1), sv.API520_SH(T_K, P1)
        A_chk = 190.5 * W / (P1 * kPa * Kd * Kb * Kc * KN * KSH)
        out.update(flow="crítico (Napier)", Kd=Kd, Kb=Kb, KN=KN, KSH=KSH)
        check = "API 520 Parte I ec. de Napier con KN y KSH"
    else:
        if density_kg_m3 is None:
            raise ValueError("liquid service needs density_kg_m3 (and viscosity_Pa_s for the viscosity correction Kv)")
        Kd = 0.65 if Kd is None else Kd
        Kw = None if valve_type == "balanced_bellows" else 1.0
        A = sv.API520_A_l(m=relieving_rate_kg_s, rho=density_kg_m3, P1=P1, P2=P2, overpressure=overpressure_fraction,
                          Kd=Kd, Kc=Kc, Kw=Kw, mu=viscosity_Pa_s,
                          Kv=None if viscosity_Pa_s is not None else 1.0)
        Kw = sv.API520_W(Pset, P2) if Kw is None else Kw
        Q_Lmin = relieving_rate_kg_s / density_kg_m3 * 6e4
        G = density_kg_m3 / 999.0
        A0 = 11.78 * Q_Lmin / (Kd * Kw * Kc) * math.sqrt(G / ((P1 - P2) * kPa))       # Kv = 1
        Kv = A0 / (A * 1e6) if viscosity_Pa_s is not None else 1.0
        A_chk = A0 / Kv
        out.update(flow="líquido", Kd=Kd, Kw=Kw, Kv=Kv)
        check = "API 520 Parte I ec. 27 (líquido, Kd 0.65 certificado) con Kv recalculado"
        if overpressure_fraction != 0.10:
            out["warnings"].append("API 520 dimensiona válvulas de líquido certificadas con 10 % de sobrepresión")
    A = float(A)
    match = abs(A * 1e6 - A_chk) / A_chk < 2e-3
    out.update(required_area_mm2=A * 1e6, mass_flux_kg_m2_s=relieving_rate_kg_s / A, **_api526(A))
    if out["valves"] > 1:
        out["warnings"].append(f"el área supera la T de API 526 ({16774:.0f} mm²): {out['valves']} válvulas en paralelo")
    out["evidence"] = {"independent_formula_area_mm2": A_chk, "independent_formula": check, "formula_match": match,
                       "verified": match,
                       "reasons": [] if match else ["el área de fluids no coincide con la ecuación impresa de API 520"]}
    out["notes"] = ("Kd preliminar API (gas/vapor 0.975, líquido 0.65): con la válvula elegida usar el Kd certificado "
                    "del fabricante. La tasa de alivio sale del escenario (API 521: fuego, bloqueo, falla de "
                    "enfriamiento); esta tool no la decide.")
    return out


# ---------------------------------------------------------------- control valves (IEC 60534-2-1)
N1, N2, N5, N9 = 0.1, 1.6e-3, 1.8e-3, 24.6          # m³/h, kPa, mm; N9 for Q at 0 °C and 1 atm
RHO0 = 999.10329075702                              # water at 15 °C, the IEC reference


def _iec_turbulent(fluid: str, Q_h: float, P1: float, P2: float, FL: float, dims: tuple | None, liq: dict, gas: dict) -> dict:
    """IEC 60534-2-1 turbulent sizing with the piping factors FP, FLP, xTP solved to a converged fixed point
    (P in kPa, Q in m³/h, diameters in mm). Returns C and the factors at convergence."""
    from fluids.control_valve import loss_coefficient_piping
    if dims:
        D1, D2, d = (x * 1000.0 for x in dims)
        z_all, z_up = loss_coefficient_piping(d, D1, D2), loss_coefficient_piping(d, D1)
    dP = P1 - P2

    def solve(C: float | None) -> dict:
        FP = FLP = None
        if C is not None and dims:
            FP = 1.0 / math.sqrt(1.0 + z_all / N2 * (C / d ** 2) ** 2)
        fp = FP or 1.0
        if fluid == "liquid":
            if C is not None and dims:
                FLP = FL / math.sqrt(1.0 + FL ** 2 / N2 * z_up * (C / d ** 2) ** 2)
            flp = FLP or FL
            dP_max = (flp / fp) ** 2 * (P1 - liq["FF"] * liq["Pv"])
            choked = dP >= dP_max
            C_new = (Q_h / (N1 * flp) * math.sqrt(liq["rr"] / (P1 - liq["FF"] * liq["Pv"])) if choked
                     else Q_h / (N1 * fp) * math.sqrt(liq["rr"] / dP))
            return {"C": C_new, "FP": FP, "FLP": FLP, "choked": choked, "dP_choked_kPa": dP_max}
        xT, Fg = gas["xT"], gas["Fg"]
        xTP = xT / fp ** 2 / (1.0 + xT * z_up / N5 * (C / d ** 2) ** 2) if (C is not None and dims) else xT
        x = dP / P1
        choked = x >= Fg * xTP
        xe = Fg * xTP if choked else x
        Y = 1.0 - xe / (3.0 * Fg * xTP)
        C_new = Q_h / (N9 * fp * P1 * Y) * math.sqrt(gas["MTZ"] / xe)
        return {"C": C_new, "FP": FP, "xTP": xTP if dims else None, "choked": choked, "Y": Y, "x": x,
                "x_choked": Fg * xTP}

    s = solve(None)
    if dims:
        for _ in range(200):
            nxt = solve(s["C"])
            done = abs(nxt["C"] - s["C"]) / nxt["C"] < 1e-12
            s = nxt
            if done:
                break
        s["fixed_point_residual"] = abs(solve(s["C"])["C"] - s["C"]) / s["C"]
    return s


def control_valve(fluid: str, P1_Pa: float, P2_Pa: float, FL: float, Fd: float, flow_m3_s: float,
                  density_kg_m3: float | None = None, vapor_pressure_Pa: float | None = None,
                  critical_pressure_Pa: float | None = None, viscosity_Pa_s: float | None = None,
                  T_K: float | None = None, MW: float | None = None, gamma: float | None = None, Z: float | None = None,
                  xT: float | None = None, pipe_inlet_diameter_m: float | None = None,
                  pipe_outlet_diameter_m: float | None = None, valve_diameter_m: float | None = None) -> dict:
    from fluids import control_valve as cv
    if P2_Pa >= P1_Pa:
        raise ValueError("P2 must be below P1")
    dims = (pipe_inlet_diameter_m, pipe_outlet_diameter_m, valve_diameter_m)
    if any(v is not None for v in dims) and None in dims:
        raise ValueError("give all three of pipe_inlet_diameter_m, pipe_outlet_diameter_m, valve_diameter_m, or none")
    geo = dict(D1=dims[0], D2=dims[1], d=dims[2])
    reducers = dims[0] is not None and not (dims[0] == dims[1] == dims[2])
    warnings, liq, gas = [], {}, {}
    if fluid == "liquid":
        if None in (density_kg_m3, vapor_pressure_Pa, critical_pressure_Pa, viscosity_Pa_s):
            raise ValueError("liquid needs density_kg_m3, vapor_pressure_Pa, critical_pressure_Pa, viscosity_Pa_s")
        r = cv.size_control_valve_l(rho=density_kg_m3, Psat=vapor_pressure_Pa, Pc=critical_pressure_Pa,
                                    mu=viscosity_Pa_s, P1=P1_Pa, P2=P2_Pa, Q=flow_m3_s, FL=FL, Fd=Fd, full_output=True, **geo)
        liq = {"FF": 0.96 - 0.28 * math.sqrt(vapor_pressure_Pa / critical_pressure_Pa), "Pv": vapor_pressure_Pa / 1e3,
               "rr": density_kg_m3 / RHO0}
    elif fluid == "gas":
        if None in (T_K, MW, gamma, Z, viscosity_Pa_s, xT):
            raise ValueError("gas needs T_K, MW, gamma (Cp/Cv), Z, viscosity_Pa_s and xT; flow_m3_s at 0 °C, 1 atm")
        r = cv.size_control_valve_g(T=T_K, MW=MW, mu=viscosity_Pa_s, gamma=gamma, Z=Z, P1=P1_Pa, P2=P2_Pa, Q=flow_m3_s,
                                    FL=FL, Fd=Fd, xT=xT, full_output=True, **geo)
        gas = {"xT": xT, "Fg": gamma / 1.4, "MTZ": MW * T_K * Z}
    else:
        raise ValueError("fluid must be 'liquid' or 'gas'")
    laminar = bool(r.get("laminar"))
    s = _iec_turbulent(fluid, flow_m3_s * 3600.0, P1_Pa / 1e3, P2_Pa / 1e3, FL, dims if reducers else None, liq, gas)
    Kv = float(r["Kv"])                     # IEC 60534-2-1 procedure (matches the standard's examples 1-3)
    if fluid == "liquid":
        extra = {"dP_choked_Pa": s["dP_choked_kPa"] * 1e3, "FF": liq["FF"], "flashing": P2_Pa < vapor_pressure_Pa,
                 "cavitation_index_sigma": (P1_Pa - vapor_pressure_Pa) / (P1_Pa - P2_Pa), "FLP": s.get("FLP")}
        if P2_Pa < vapor_pressure_Pa:
            warnings.append("P2 < Psat: flasheo a la salida; materiales endurecidos y velocidad de salida a revisar")
        elif s["choked"]:
            warnings.append("flujo estrangulado por cavitación: más ΔP no da más flujo; considerar trim anticavitación")
    else:
        extra = {"x": s["x"], "x_choked": s["x_choked"], "Y": s["Y"], "F_gamma": gas["Fg"], "xTP": s.get("xTP")}
        if s["choked"]:
            warnings.append("flujo crítico (x ≥ Fγ·xTP): el flujo ya no crece al bajar P2; revisar ruido (IEC 60534-8-3)")
    agree = abs(float(r["Kv"]) - s["C"]) / s["C"]
    converged = s.get("fixed_point_residual", 0.0) < 1e-9
    reasons = []
    if laminar:
        reasons.append("régimen laminar (Re_v bajo): Kv de fluids con corrección FR, sin verificación independiente")
    if not converged:
        reasons.append("los factores de tubería FP/xTP no convergieron")
    if not laminar and agree > (0.05 if reducers else 1e-6):
        reasons.append("el Kv no coincide con la ecuación de IEC 60534-2-1 recalculada" + (
            " con FP/xTP convergidos (> 5 %)" if reducers else ""))
    ok = not reasons
    return {"fluid": fluid, "Kv_m3_h": Kv, "Cv_USgpm": Kv * 1.156, "choked": bool(s["choked"]), "laminar": laminar,
            "Re_valve": r.get("Rev"), "FP": s.get("FP"), "dP_Pa": P1_Pa - P2_Pa, **extra, "warnings": warnings,
            "evidence": {"Kv_independent": s["C"], "deviation_pct": agree * 100.0,
                         "piping_factors_converged": converged, "verified": ok, "reasons": reasons},
            "notes": ("Kv por el procedimiento de IEC 60534-2-1 (fluids; reproduce los ejemplos 1-3 de la norma). La "
                      "evidencia recalcula las ecuaciones de la norma por fuera: exactas sin reductores; con reductores, "
                      "FP/FLP/xTP a punto fijo convergido (la norma corta su iteración en C/Ci ≥ 0.99 y queda unos % "
                      "del lado conservador). FL, Fd y xT son del estilo de válvula (catálogo o IEC 60534-2-1 tabla 2) y los da el "
                      "usuario. Elegir la válvula para que el Kv requerido caiga entre 20 % y 80 % de su apertura.")}

# ---------------------------------------------------------------- fittings
def fittings_K(fittings: list[dict], inner_diameter_m: float, Re: float, method: str = "Darby3K") -> dict:
    from fluids import fittings as ft
    table = {"Darby3K": ft.Darby, "Hooper2K": ft.Hooper}.get(method)
    if table is None:
        raise ValueError("method must be 'Darby3K' or 'Hooper2K'")
    if not fittings:
        return {"method": method, "available_fittings": list(table)}
    rows, total = [], 0.0
    for item in fittings:
        name, n = item["name"], int(item.get("count", 1))
        if name not in table:
            near = difflib.get_close_matches(name, list(table), n=5, cutoff=0.3)
            raise ValueError(f"fitting '{name}' not in {method}; closest: {near} (call with fittings=[] for the list)")
        K = (ft.Darby3K(Di=inner_diameter_m, Re=Re, name=name) if method == "Darby3K"
             else ft.Hooper2K(Di=inner_diameter_m / 0.0254, Re=Re, name=name))
        rows.append({"name": name, "count": n, "K_each": float(K), "K": float(K) * n})
        total += float(K) * n
    return {"method": method, "Re": Re, "inner_diameter_m": inner_diameter_m, "fittings": rows, "K_total": total,
            "notes": "Pasar K_total como fittings_K a pipe_hydraulics. Darby 3-K (Darby 2001) o Hooper 2-K (1981): "
                     "K depende de Re y del diámetro, a diferencia de una K fija de tabla."}


# ---------------------------------------------------------------- compressible gas line (isothermal)
def _ode_P2(P1: float, G: float, c2: float, fd: float, D: float, L: float) -> float:
    """Momentum balance dP(1 − G²c²/P²) = −fd·G²c²/(2·D·P)·dx integrated numerically (independent of the closed form)."""
    from scipy.integrate import solve_ivp
    rhs = lambda x, P: [-(fd * G ** 2 * c2 / (2.0 * D * P[0])) / (1.0 - G ** 2 * c2 / P[0] ** 2)]
    s = solve_ivp(rhs, (0.0, L), [P1], rtol=1e-10, atol=1e-6, method="LSODA")
    return float(s.y[0, -1])


def _iso_residual(P1: float, P2: float, G: float, c2: float, fLD: float) -> float:
    """Isothermal ideal-gas line, closed form: P1² − P2² = G²c²·(fd·L/D + 2·ln(P1/P2))."""
    return P1 ** 2 - P2 ** 2 - G ** 2 * c2 * (fLD + 2.0 * math.log(P1 / P2))


def gas_line(P1_Pa: float, T_K: float, MW: float, Z: float, k: float, viscosity_Pa_s: float, inner_diameter_m: float,
             length_m: float, mass_flow_kg_s: float | None = None, P2_Pa: float | None = None,
             roughness_m: float = STEEL_ROUGHNESS_M) -> dict:
    from scipy.optimize import brentq
    if (mass_flow_kg_s is None) == (P2_Pa is None):
        raise ValueError("give exactly one of mass_flow_kg_s (→ outlet pressure) or P2_Pa (→ mass flow)")
    D, L = inner_diameter_m, length_m
    A, c2 = math.pi * D ** 2 / 4.0, Z * R * T_K / (MW / 1000.0)
    c, rho1, eD = math.sqrt(c2), P1_Pa / c2, roughness_m / D
    fd_of = lambda m: _darcy_f(4.0 * m / (math.pi * D * viscosity_Pa_s), eD)
    choke_P = lambda fLD: brentq(lambda p: _iso_residual(P1_Pa, p, p / c, c2, fLD), 1e-9 * P1_Pa, P1_Pa / math.sqrt(1.0 + 1e-12))
    choked = False
    if mass_flow_kg_s is not None:
        m = mass_flow_kg_s
        G, fd = m / A, fd_of(m)
        P_star = G * c                                     # isothermal choke: v = c at the outlet (Ma = 1/√k)
        if P_star >= P1_Pa or _iso_residual(P1_Pa, P_star, G, c2, fd * L / D) < 0:
            raise ValueError(f"the line chokes: {m:.4g} kg/s needs an outlet below the isothermal choke pressure "
                             f"{P_star:.4g} Pa (Ma = 1/√k at the outlet); enlarge D or shorten the line")
        P2 = brentq(lambda p: _iso_residual(P1_Pa, p, G, c2, fd * L / D), P_star, P1_Pa)
        P_crit = P_star
    else:
        fd, m = 0.02, 0.0
        for _ in range(100):
            fLD = fd * L / D
            P_crit = choke_P(fLD)
            P2 = max(P2_Pa, P_crit)
            G = math.sqrt((P1_Pa ** 2 - P2 ** 2) / (c2 * (fLD + 2.0 * math.log(P1_Pa / P2))))
            m_new = G * A
            fd_new = fd_of(m_new)
            if abs(m_new - m) / m_new < 1e-12 and abs(fd_new - fd) < 1e-14:
                break
            m, fd = m_new, fd_new
        m, choked = m_new, P2_Pa < P_crit
    G = m / A
    v1, v2 = G / rho1, G / (P2 / c2)
    sound = math.sqrt(k * c2)
    from packages.transport.hydraulics import pipe_dp
    dP = P1_Pa - P2
    inc = pipe_dp(rho1, viscosity_Pa_s, m, D, L, roughness_m)["dP_friction_Pa"]
    P2_ode = _ode_P2(P1_Pa, G, c2, fd, D, L) if not choked else float("nan")
    match = (not choked) and abs(P2_ode - P2) / P1_Pa < 1e-5
    reasons = ([] if match or choked else ["la integral numérica de la ecuación de momento no coincide"]) + (
        ["línea estrangulada: el flujo reportado es el máximo; P2 pedida no se alcanza en la tubería"] if choked else [])
    out = {"mass_flow_kg_s": m, "P2_Pa": P2, "dP_Pa": dP, "dP_fraction_of_P1": dP / P1_Pa, "friction_factor_darcy": fd,
           "Re": 4.0 * m / (math.pi * D * viscosity_Pa_s), "velocity_in_m_s": v1, "velocity_out_m_s": v2,
           "Mach_in": v1 / sound, "Mach_out": v2 / sound, "choke_pressure_Pa": P_crit, "choked": bool(choked),
           "dP_incompressible_Pa": inc, "incompressible_error_pct": (inc - dP) / dP * 100.0 if dP > 0 else None,
           "model": "isotérmico, gas ideal con Z constante: P1² − P2² = G²·ZRT/M·(f·L/D + 2·ln(P1/P2))",
           "evidence": {"P2_momentum_ode_Pa": None if choked else P2_ode, "ode_match": match, "verified": match,
                        "reasons": reasons},
           "notes": ("Isotérmico es lo correcto para líneas largas sin aislar; para tramos cortos y rápidos "
                     "(Ma > 0.3) el adiabático (Fanno) da un poco menos de flujo. El estrangulamiento isotérmico ocurre "
                     "cuando la velocidad de salida llega a √(ZRT/M), Ma = 1/√k.")}
    if choked:
        out["warning"] = (f"P2 pedida está bajo la presión de estrangulamiento {P_crit:.4g} Pa: el flujo reportado es el "
                          "máximo de la línea")
    return out

# ---------------------------------------------------------------- two-phase frictional pressure drop
TWO_PHASE_METHODS = ("Friedel", "Muller_Steinhagen_Heck", "Chisholm", "Lockhart_Martinelli")


def two_phase_line(mass_flow_kg_s: float, quality: float, liquid_density_kg_m3: float, gas_density_kg_m3: float,
                   liquid_viscosity_Pa_s: float, gas_viscosity_Pa_s: float, surface_tension_N_m: float,
                   inner_diameter_m: float, length_m: float, roughness_m: float = STEEL_ROUGHNESS_M) -> dict:
    from fluids.two_phase import Taitel_Dukler_regime, two_phase_dP
    if not 0.0 < quality < 1.0:
        raise ValueError("quality (vapor mass fraction) must be strictly between 0 and 1; use pipe_hydraulics if single phase")
    D, L, m = inner_diameter_m, length_m, mass_flow_kg_s
    kw = dict(m=m, rhol=liquid_density_kg_m3, rhog=gas_density_kg_m3, mul=liquid_viscosity_Pa_s,
              mug=gas_viscosity_Pa_s, sigma=surface_tension_N_m, D=D, L=L, roughness=roughness_m)
    dps = {M: float(two_phase_dP(x=quality, Method=M, **kw)) for M in TWO_PHASE_METHODS}
    G = m / (math.pi * D ** 2 / 4.0)
    ratio = liquid_viscosity_Pa_s / gas_viscosity_Pa_s
    rec = "Friedel" if ratio < 1000.0 else ("Chisholm" if G > 100.0 else "Lockhart_Martinelli")
    regime = Taitel_Dukler_regime(m=m, x=quality, rhol=liquid_density_kg_m3, rhog=gas_density_kg_m3,
                                  mul=liquid_viscosity_Pa_s, mug=gas_viscosity_Pa_s, D=D, angle=0.0,
                                  roughness=roughness_m)[0]
    # single-phase limits of the recommended method, against Darcy-Weisbach with Clamond's friction factor
    lim = {}
    for name, x, rho, mu in (("liquid", 1e-9, liquid_density_kg_m3, liquid_viscosity_Pa_s),
                             ("gas", 1 - 1e-9, gas_density_kg_m3, gas_viscosity_Pa_s)):
        v = G / rho
        darcy = _darcy_f(G * D / mu, roughness_m / D) * L / D * 0.5 * rho * v ** 2
        lim[name] = abs(float(two_phase_dP(x=x, Method=rec, **kw)) - darcy) / darcy
    ok = max(lim.values()) < 0.02
    spread = (max(dps.values()) - min(dps.values())) / dps[rec] * 100.0
    return {"dP_friction_Pa": dps[rec], "recommended_method": rec, "dP_by_method_Pa": dps, "method_spread_pct": spread,
            "mass_flux_kg_m2_s": G, "viscosity_ratio": ratio, "flow_regime_Taitel_Dukler": regime,
            "evidence": {"single_phase_limit_error": lim, "limits_consistent": ok, "verified": ok,
                         "reasons": [] if ok else ["el método no reduce a Darcy en los límites de una fase"]},
            "notes": ("Solo fricción, tubo horizontal y calidad constante (sin flasheo a lo largo). Criterio de "
                      "Whalley (1980): μl/μg < 1000 → Friedel; > 1000 con G > 100 kg/m²s → Chisholm; si no, "
                      "Lockhart-Martinelli. Estas correlaciones tienen ±30-50 % de dispersión: la columna "
                      "dP_by_method_Pa da el intervalo. Para gradiente estático y aceleración usar la tubería de DWSIM "
                      "(Beggs-Brill).")}
