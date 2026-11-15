"""Adiabatic compressible gas flow in a constant-area pipe with wall friction: Fanno flow (bloque MG).

What it answers: for an ideal gas entering a pipe of diameter D and length L, the outlet state (P, T, Mach, velocity)
for a given mass flow, or the mass flow for a given outlet pressure; whether the line chokes (Mach 1 at the outlet),
the critical length L* from the inlet Mach, the maximum flow of the line, and the same line solved isothermally
(`packages.transport.flow_devices.gas_line`) for comparison.

Model (calorically perfect ideal gas, γ and molar mass constant, Darcy friction factor f constant along the pipe):
    f·L*/D = (1 − M²)/(γ·M²) + (γ + 1)/(2γ)·ln[(γ + 1)·M² / (2 + (γ − 1)·M²)]
    T/T* = (γ + 1)/(2 + (γ − 1)·M²)        P/P* = (1/M)·√(T/T*)
    P0/P0* = (1/M)·[(2 + (γ − 1)·M²)/(γ + 1)]^((γ+1)/(2(γ−1)))
with f the Darcy factor (= 4·Fanning, so f·L*/D here is the «4fL*/D» of tables written with the Fanning factor).
Between inlet 1 and outlet 2: f·L/D = f·L*/D(M1) − f·L*/D(M2). Only the subsonic branch is solved (process lines).

Inlet basis: 'static' means P1, T1 are the static pressure and temperature in the pipe at the inlet (what a gauge in the
line reads); 'stagnation' means they are the reservoir (rest) conditions upstream of an isentropic, loss-free entrance,
the textbook reservoir-pipe problem.

Evidence of THIS run (independent of the closed form): the primitive equations (continuity ρv = G, energy cp·dT + v·dv
= 0, momentum dP + G·dv = −f·G·v/(2D)·dx) integrated as an ODE in x with scipy must reproduce the outlet pressure and
keep the stagnation temperature constant. Friction factor: Clamond's solution of Colebrook through `fluids` (laminar
64/Re) with the inlet viscosity; Re = G·D/μ changes only through μ(T), which is ignored (said so).
"""
from __future__ import annotations

import math

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from packages.transport.hydraulics import STEEL_ROUGHNESS_M

R_GAS = 8.314462618                  # J/(mol K)
ODE_RTOL = 1e-10
ODE_MATCH_TOL = 1e-6                 # relative to P1
T0_MATCH_TOL = 1e-7
M_CHECK_MAX = 0.98                   # the ODE is singular at M = 1: a choked run is checked up to this Mach
M_MIN = 1e-7


# ---------------------------------------------------------------- closed-form Fanno functions
def fanno_fLD(M: float, gamma: float) -> float:
    """Darcy f·L*/D from Mach M to the sonic point (both branches)."""
    g, M2 = gamma, M * M
    return (1.0 - M2) / (g * M2) + (g + 1.0) / (2.0 * g) * math.log((g + 1.0) * M2 / (2.0 + (g - 1.0) * M2))


def fanno_ratios(M: float, gamma: float) -> dict:
    """Static and stagnation ratios to the sonic (starred) state at the same mass flux and stagnation temperature."""
    g = gamma
    t = (g + 1.0) / (2.0 + (g - 1.0) * M * M)
    p = math.sqrt(t) / M
    p0 = (1.0 / M) * ((2.0 + (g - 1.0) * M * M) / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    return {"T_T_star": t, "P_P_star": p, "rho_rho_star": p / t, "P0_P0_star": p0, "v_v_star": t / p,
            "fLD_star": fanno_fLD(M, g)}


def mach_from_fLD(fLD: float, gamma: float) -> float:
    """Subsonic Mach whose f·L*/D equals `fLD` (fanno_fLD decreases monotonically from ∞ at M → 0 to 0 at M = 1)."""
    if fLD <= 0.0:
        return 1.0
    return brentq(lambda M: fanno_fLD(M, gamma) - fLD, M_MIN, 1.0, xtol=1e-15, rtol=4e-15, maxiter=500)


def _stag_T(T: float, M: float, g: float) -> float:
    return T * (1.0 + 0.5 * (g - 1.0) * M * M)


def _stag_P(P: float, M: float, g: float) -> float:
    return P * (1.0 + 0.5 * (g - 1.0) * M * M) ** (g / (g - 1.0))


def _darcy_f(Re: float, eD: float) -> float:
    from fluids import friction_factor
    return 64.0 / Re if Re < 2300.0 else float(friction_factor(Re=Re, eD=eD))


# ---------------------------------------------------------------- independent ODE in x
def _ode_outlet(P1: float, T1: float, G: float, Rs: float, cp: float, fd: float, D: float, x_end: float) -> tuple:
    """Integrate continuity + energy + momentum in (P, T) from the inlet to x_end; returns (P, T, ok)."""
    def rhs(_x, y):
        P, T = y
        v = G * Rs * T / P
        # unknowns dP, dT per unit length:  cp·dT + v²(dT/T − dP/P) = 0 ;  dP + G·v(dT/T − dP/P) = −fd·G·v/(2D)
        a11, a12 = -v * v / P, cp + v * v / T
        a21, a22 = 1.0 - G * v / P, G * v / T
        b2 = -fd * G * v / (2.0 * D)
        det = a11 * a22 - a12 * a21
        return [(-a12 * b2) / det, (a11 * b2) / det]
    s = solve_ivp(rhs, (0.0, x_end), [P1, T1], rtol=ODE_RTOL, atol=1e-9, method="LSODA")
    return float(s.y[0, -1]), float(s.y[1, -1]), bool(s.success)


# ---------------------------------------------------------------- properties of the gas
def _gas_from_compounds(compounds: dict, T_K: float, P_Pa: float) -> dict:
    """Ideal-gas γ, molar mass and viscosity of a gas (mixture) from `packages.properties.pure` (chemicals/thermo)."""
    from packages.properties import pure as PU
    tot = sum(float(v) for v in compounds.values())
    if not compounds or tot <= 0.0:
        raise ValueError("compounds debe ser {nombre: fracción molar} con fracciones positivas")
    names, ys, MWs, cps, mus, notes = [], [], [], [], [], []
    for name, y in compounds.items():
        r = PU.compound_properties(name, T_K, P_Pa)
        cp = r["properties_at_T"]["Cp_ideal_gas_J_molK"]
        mu = r["properties_at_T"]["mu_gas_Pa_s"]
        if cp.get("value") is None:
            raise ValueError(f"no hay Cp de gas ideal para {name} a {T_K:.1f} K: dar gamma y molar_mass_kg_mol")
        names.append(name)
        ys.append(float(y) / tot)
        MWs.append(r["identity"]["molar_mass_kg_mol"])
        cps.append(cp["value"])
        mus.append(mu.get("value"))
        mu_txt = "sin dato" if mu.get("value") is None else format(mu["value"], ".4g")
        notes.append(f"{name}: Cp gas ideal {cp['value']:.4g} J/(mol K) por {cp.get('method')} "
                     f"({'estimado' if cp.get('estimated') else cp.get('basis')}); μ gas {mu_txt} Pa·s por "
                     f"{mu.get('method')}{' (estimado)' if mu.get('estimated') else ''}")
    MW = sum(y * m for y, m in zip(ys, MWs))
    cp_mix = sum(y * c for y, c in zip(ys, cps))
    mu = None
    if all(m is not None for m in mus):
        if len(mus) == 1:
            mu = mus[0]
        else:
            from chemicals.viscosity import Herning_Zipperer
            mu = float(Herning_Zipperer(ys, mus, [m * 1000.0 for m in MWs]))
    return {"molar_mass_kg_mol": MW, "cp_J_molK": cp_mix, "gamma": cp_mix / (cp_mix - R_GAS), "viscosity_Pa_s": mu,
            "mole_fractions": dict(zip(names, ys)), "notes": notes, "mixture": len(names) > 1}


# ---------------------------------------------------------------- main engine
def adiabatic_gas_pipe(P1_Pa: float, T1_K: float, inner_diameter_m: float, length_m: float, inlet_basis: str,
                       mass_flow_kg_s: float | None = None, P2_Pa: float | None = None, gamma: float | None = None,
                       molar_mass_kg_mol: float | None = None, viscosity_Pa_s: float | None = None,
                       compounds: dict | None = None, roughness_m: float = STEEL_ROUGHNESS_M,
                       darcy_friction_factor: float | None = None, compare_isothermal: bool = True) -> dict:
    if inlet_basis not in ("static", "stagnation"):
        raise ValueError("inlet_basis debe ser 'static' (P1 y T1 estáticas en la entrada del tubo) o 'stagnation' "
                         "(P1 y T1 de reposo en el recipiente aguas arriba, entrada isentrópica sin pérdidas)")
    if (mass_flow_kg_s is None) == (P2_Pa is None):
        raise ValueError("dar exactamente uno: mass_flow_kg_s (→ condiciones de salida) o P2_Pa (→ flujo másico)")
    for name, val in (("P1_Pa", P1_Pa), ("T1_K", T1_K), ("inner_diameter_m", inner_diameter_m), ("length_m", length_m)):
        if val is None or not val > 0:
            raise ValueError(f"falta {name} (> 0): define el estado de entrada y la geometría del tubo")
    if P2_Pa is not None and not 0 < P2_Pa < P1_Pa:
        raise ValueError("P2_Pa debe estar entre 0 y P1_Pa")
    if mass_flow_kg_s is not None and not mass_flow_kg_s > 0:
        raise ValueError("mass_flow_kg_s debe ser > 0")
    assumptions = ["gas ideal caloríficamente perfecto: γ y masa molar constantes a lo largo del tubo",
                   "flujo adiabático (sin calor por la pared), estacionario, unidimensional, área constante",
                   "factor de fricción de Darcy constante, evaluado con la viscosidad de entrada (μ(T) se ignora)",
                   "solo la rama subsónica (entrada con Mach < 1)"]
    warnings: list[str] = []
    gas_src = "usuario"
    props = None
    if compounds:
        if gamma is not None or molar_mass_kg_mol is not None:
            raise ValueError("dar gamma y molar_mass_kg_mol, o compounds, no ambos")
        T_eval, P_eval = T1_K, P1_Pa
        props = _gas_from_compounds(compounds, T_eval, P_eval)
        gamma, molar_mass_kg_mol = props["gamma"], props["molar_mass_kg_mol"]
        if viscosity_Pa_s is None:
            viscosity_Pa_s = props["viscosity_Pa_s"]
            if props["mixture"] and viscosity_Pa_s is not None:
                assumptions.append("viscosidad de la mezcla por Herning-Zipperer (regla de mezclado, estimación)")
        gas_src = "compuestos (packages.properties.pure sobre chemicals/thermo)"
        assumptions.append(f"γ = Cp/(Cp − R) del gas ideal a T1 = {T_eval:.2f} K ({gamma:.5g}); se mantiene constante")
    if gamma is None or molar_mass_kg_mol is None:
        raise ValueError("faltan gamma (Cp/Cv) y molar_mass_kg_mol del gas, o compounds {nombre: fracción molar} para "
                         "leerlos de la base de propiedades: definen la velocidad del sonido y la densidad")
    if not 1.0 < gamma < 1.7:
        raise ValueError("gamma debe estar entre 1 y 1.7 para un gas")
    if darcy_friction_factor is None and viscosity_Pa_s is None:
        raise ValueError("falta viscosity_Pa_s del gas (o darcy_friction_factor): define el número de Reynolds y la "
                         "fricción")
    if roughness_m == STEEL_ROUGHNESS_M and darcy_friction_factor is None:
        assumptions.append(f"rugosidad de acero comercial {STEEL_ROUGHNESS_M} m (pasar roughness_m si el tubo es otro)")
    g, D, L = float(gamma), float(inner_diameter_m), float(length_m)
    Rs = R_GAS / molar_mass_kg_mol
    cp = g * Rs / (g - 1.0)
    A = math.pi * D * D / 4.0

    def inlet_state(M1: float) -> tuple:
        """Static inlet P, T for an inlet Mach (stagnation basis: isentropic entrance from the reservoir)."""
        if inlet_basis == "static":
            return P1_Pa, T1_K
        k = 1.0 + 0.5 * (g - 1.0) * M1 * M1
        return P1_Pa / k ** (g / (g - 1.0)), T1_K / k

    def mflow(M1: float) -> float:
        P, T = inlet_state(M1)
        return P * A * M1 * math.sqrt(g / (Rs * T))

    def fd_of(m: float) -> float:
        if darcy_friction_factor is not None:
            return float(darcy_friction_factor)
        return _darcy_f(m / A * D / viscosity_Pa_s, roughness_m / D)

    def M2_of(M1: float, fLD: float) -> float:
        return mach_from_fLD(fanno_fLD(M1, g) - fLD, g)

    def P2_of(M1: float, fLD: float) -> float:
        P, _ = inlet_state(M1)
        M2 = M2_of(M1, fLD)
        return P * fanno_ratios(M2, g)["P_P_star"] / fanno_ratios(M1, g)["P_P_star"]

    def M1_max(fLD: float) -> float:
        return mach_from_fLD(fLD, g)

    feasible, choked = True, False
    if mass_flow_kg_s is not None:
        m = float(mass_flow_kg_s)
        if inlet_basis == "static":
            M1 = m / (P1_Pa * A) * math.sqrt(Rs * T1_K / g)
            if M1 >= 1.0:
                raise ValueError(f"el Mach de entrada sería {M1:.3f} ≥ 1 con P1 y T1 estáticas: fuera del alcance subsónico")
        else:
            if m >= mflow(1.0):
                raise ValueError(f"{m:.4g} kg/s supera el flujo isentrópico máximo del recipiente ({mflow(1.0):.4g} kg/s "
                                 "con Mach 1 en la entrada)")
            M1 = brentq(lambda M: mflow(M) - m, M_MIN, 1.0, xtol=1e-15, rtol=4e-15)
        fd = fd_of(m)
        fLD = fd * L / D
        if fLD > fanno_fLD(M1, g):
            feasible = False
            choked = True
    else:
        fd = 0.02 if darcy_friction_factor is None else float(darcy_friction_factor)
        m = None
        for _ in range(200):
            fLD = fd * L / D
            M1m = M1_max(fLD)
            P_choke = inlet_state(M1m)[0] / fanno_ratios(M1m, g)["P_P_star"]
            if P2_Pa <= P_choke * (1.0 + 1e-12):
                M1, choked = M1m, True
            else:
                M1 = brentq(lambda M: P2_of(M, fLD) - P2_Pa, M_MIN, M1m, xtol=1e-15, rtol=4e-15, maxiter=500)
            m_new = mflow(M1)
            fd_new = fd_of(m_new)
            done = m is not None and abs(m_new - m) <= 1e-13 * m_new and abs(fd_new - fd) <= 1e-14
            m, fd = m_new, fd_new
            if done or darcy_friction_factor is not None:
                break
        fLD = fd * L / D
        if not choked:
            M1 = brentq(lambda M: P2_of(M, fLD) - P2_Pa, M_MIN, M1_max(fLD), xtol=1e-15, rtol=4e-15, maxiter=500)
        else:
            M1 = M1_max(fLD)
        m = mflow(M1)

    Re = m / A * D / viscosity_Pa_s if viscosity_Pa_s is not None else None
    Pin, Tin = inlet_state(M1)
    r1 = fanno_ratios(M1, g)
    a1 = math.sqrt(g * Rs * Tin)
    T0 = _stag_T(Tin, M1, g)
    P_star, T_star = Pin / r1["P_P_star"], Tin / r1["T_T_star"]
    L_star = r1["fLD_star"] * D / fd
    inlet = {"P_Pa": Pin, "T_K": Tin, "Mach": M1, "velocity_m_s": M1 * a1, "density_kg_m3": Pin / (Rs * Tin),
             "stagnation_P_Pa": _stag_P(Pin, M1, g), "stagnation_T_K": T0}
    out = {"mode": "flujo dado → salida" if mass_flow_kg_s is not None else "P2 dada → flujo",
           "inlet_basis": inlet_basis, "gamma": g, "molar_mass_kg_mol": molar_mass_kg_mol, "gas_properties_source": gas_src,
           "mass_flow_kg_s": m, "mass_flux_kg_m2_s": m / A, "Re": Re, "friction_factor_darcy": fd,
           "friction_source": "usuario" if darcy_friction_factor is not None else "Colebrook (Clamond, fluids) / 64/Re",
           "fL_D": fLD, "inlet": inlet, "fLmax_D_inlet": r1["fLD_star"], "critical_length_m": L_star,
           "sonic_state": {"P_Pa": P_star, "T_K": T_star}, "choked": bool(choked), "feasible": feasible}
    if props:
        out["gas_properties"] = props
    reasons: list[str] = []
    if not feasible:
        # the asked flow cannot pass: maximum flow of this line with the same inlet state definition
        fd_m = fd
        for _ in range(200):
            M1m = M1_max(fd_m * L / D)
            if inlet_basis == "static":
                m_max = mflow(M1m)
            else:
                m_max = mflow(M1m)
            fd_new = fd_of(m_max)
            if abs(fd_new - fd_m) <= 1e-14 or darcy_friction_factor is not None:
                break
            fd_m = fd_new
        out.update(outlet=None, max_mass_flow_kg_s=m_max, max_flow_inlet_Mach=M1m)
        warnings.append(f"el tubo se estrangula: con el Mach de entrada {M1:.4f} la longitud crítica es {L_star:.4g} m < "
                        f"L = {L:.4g} m; {m:.4g} kg/s no pasa con este estado de entrada. Flujo máximo de la línea: "
                        f"{m_max:.4g} kg/s (Mach 1 a la salida). Agrandar D, acortar o subir P1")
        resid = abs(fanno_fLD(M1m, g) - fd_m * L / D) / (fd_m * L / D)
        out["evidence"] = {"max_flow_fLD_residual": resid, "verified": resid < 1e-9,
                           "reasons": [] if resid < 1e-9 else ["el flujo máximo no cierra f·L*/D = f·L/D"]}
    else:
        M2 = M2_of(M1, fLD)
        r2 = fanno_ratios(M2, g)
        P2 = Pin * r2["P_P_star"] / r1["P_P_star"]
        T2 = Tin * r2["T_T_star"] / r1["T_T_star"]
        a2 = math.sqrt(g * Rs * T2)
        outlet = {"P_Pa": P2, "T_K": T2, "Mach": M2, "velocity_m_s": M2 * a2, "density_kg_m3": P2 / (Rs * T2),
                  "stagnation_P_Pa": _stag_P(P2, M2, g), "stagnation_T_K": _stag_T(T2, M2, g)}
        out.update(outlet=outlet, dP_Pa=Pin - P2, dP_fraction_of_P1=(Pin - P2) / Pin,
                   stagnation_pressure_loss_Pa=inlet["stagnation_P_Pa"] - outlet["stagnation_P_Pa"])
        if choked and P2_Pa is not None:
            warnings.append(f"P2 pedida ({P2_Pa:.4g} Pa) está bajo la presión sónica de salida {P2:.4g} Pa: el tubo se "
                            "estrangula (Mach 1 a la salida), el flujo reportado es el máximo y la expansión hasta P2 "
                            "ocurre fuera del tubo")
        # ---- independent check: ODE of the primitive equations
        G = m / A
        x_end = L
        if M2 > M_CHECK_MAX:
            x_end = (r1["fLD_star"] - fanno_fLD(M_CHECK_MAX, g)) * D / fd
            P_ref = Pin * fanno_ratios(M_CHECK_MAX, g)["P_P_star"] / r1["P_P_star"]
        else:
            P_ref = P2
        P_ode, T_ode, ok = _ode_outlet(Pin, Tin, G, Rs, cp, fd, D, x_end)
        v_ode = G * Rs * T_ode / P_ode
        T0_ode = T_ode + v_ode * v_ode / (2.0 * cp)
        dev = abs(P_ode - P_ref) / Pin
        dT0 = abs(T0_ode - T0) / T0
        if not ok:
            reasons.append("la integración de la ODE de momento y energía no terminó")
        if dev > ODE_MATCH_TOL:
            reasons.append(f"la ODE de las ecuaciones primitivas no reproduce la presión de salida ({dev:.2e} de P1)")
        if dT0 > T0_MATCH_TOL:
            reasons.append("la temperatura de estancamiento no se conserva en la ODE")
        out["evidence"] = {"ode_checked_length_m": x_end, "P_closed_form_Pa": P_ref, "P_ode_Pa": P_ode,
                           "ode_P_deviation_rel_P1": dev, "stagnation_T_residual_rel": dT0,
                           "verified": not reasons, "reasons": reasons}
        if M2 > M_CHECK_MAX:
            out["evidence"]["note"] = (f"la ODE es singular en Mach 1: se comparó hasta Mach {M_CHECK_MAX} "
                                       f"(x = {x_end:.4g} m)")
        if M2 > 0.7 and not choked:
            warnings.append(f"Mach de salida {M2:.3f}: cerca del estrangulamiento, el resultado es muy sensible a f y L")
    if Re is not None and darcy_friction_factor is None and 2300.0 <= Re < 4000.0:
        warnings.append("Re en transición laminar-turbulento: factor de fricción incierto")
    # ---- isothermal comparison (existing engine)
    if compare_isothermal and viscosity_Pa_s is not None and feasible:
        from packages.transport import flow_devices as FD
        try:
            iso = FD.gas_line(Pin, Tin, molar_mass_kg_mol * 1000.0, 1.0, g, viscosity_Pa_s, D, L,
                              mass_flow_kg_s=m if mass_flow_kg_s is not None else None,
                              P2_Pa=P2_Pa if P2_Pa is not None else None, roughness_m=roughness_m)
            out["isothermal_comparison"] = {"P2_Pa": iso["P2_Pa"], "mass_flow_kg_s": iso["mass_flow_kg_s"],
                                            "choked": iso["choked"], "Mach_out": iso["Mach_out"],
                                            "note": "misma línea resuelta isotérmica a T de entrada (gas_line, Z = 1)"}
            if mass_flow_kg_s is not None:
                out["isothermal_comparison"]["dP_ratio_adiabatic_over_isothermal"] = (
                    (Pin - out["outlet"]["P_Pa"]) / (Pin - iso["P2_Pa"]))
            else:
                out["isothermal_comparison"]["flow_ratio_adiabatic_over_isothermal"] = m / iso["mass_flow_kg_s"]
            if darcy_friction_factor is not None:
                out["isothermal_comparison"]["note"] += "; ahí f sale de Colebrook, no del f dado"
        except ValueError as ex:
            out["isothermal_comparison"] = {"error": str(ex)}
    out.update(
        warnings=warnings, assumptions=assumptions,
        not_included=["gas real (Z ≠ 1) y variación de γ con T", "transferencia de calor por la pared (Rayleigh / "
                      "no adiabático)", "rama supersónica y ondas de choque", "pérdidas en accesorios y en la entrada",
                      "cambio de elevación"],
        method=("flujo de Fanno: f·L*/D, T/T* y P/P* en forma cerrada resueltos con brentq (scipy); verificación por "
                "integración de las ecuaciones primitivas con solve_ivp (LSODA)"))
    return out
