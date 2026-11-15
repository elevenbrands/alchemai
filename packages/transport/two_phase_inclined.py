"""Gas-liquid flow in inclined and vertical pipes: Beggs-Brill with inclination, and a separated-flow model (bloque MG).

What it answers: pressure profile and outlet pressure of a two-phase line at any angle (−90° downhill to +90° uphill),
the Beggs-Brill flow pattern (segregated / transition / intermittent / distributed) and liquid holdup along the pipe,
and the split of the pressure drop into gravity (hydrostatic of the holdup-weighted density), friction and acceleration.

Methods:
  * Beggs & Brill (1973, J. Petrol. Technol. 25, 607) with the transition-zone interpolation of Brill & Beggs (1991), as
    written in Shoham (2006): horizontal holdup HL(0) = a·λL^b/Fr^c (HL(0) ≥ λL), inclination factor
    ψ = 1 + C·[sin(1.8θ) − sin³(1.8θ)/3], C = (1 − λL)·ln(d·λL^e·NLv^f·Fr^h) ≥ 0, two-phase friction
    f_tp = f_n·e^S with f_n of the no-slip mixture and S from y = λL/HL², acceleration through Ek = vsg·vm·ρs/P.
    Implemented here; the normalizing friction factor is Colebrook (Clamond, `fluids`) with the user's roughness instead of
    the original smooth-pipe fit, which is the common extension. The Payne et al. (1979) holdup corrections are NOT applied.
  * Separated model: void fraction of Woldesemayat & Ghajar (2007, inclination-dependent drift flux, `fluids`) for the
    gravity term, frictional gradient of the correlation chosen by Whalley's criterion (Friedel / Chisholm /
    Lockhart-Martinelli, `fluids.two_phase_dP`) and the momentum-flux acceleration term (`two_phase_dP_acceleration`).
The line is marched in segments; each segment is solved at its mean pressure (fixed-point on the outlet pressure). The gas
density follows the pressure as an isothermal ideal gas at constant quality (no flashing, no mass transfer); the liquid is
incompressible.

Evidence of THIS run: (1) the Beggs-Brill gradient of this implementation at the inlet state against the independent
implementation in Caleb Bell's `fluids.two_phase.Beggs_Brill` (total gradient including acceleration); (2) segment
independence: the outlet pressure with 2·n segments against n.
"""
from __future__ import annotations

import math

from packages.transport.hydraulics import STEEL_ROUGHNESS_M

G_ACC = 9.80665
FLUIDS_MATCH_TOL = 1e-6          # relative, gradient of this implementation vs fluids.Beggs_Brill
SEGMENT_TOL = 2e-3               # relative change of the total ΔP between n and 2n segments
EK_LIMIT = 0.9

REGIMES = ("segregado", "transición", "intermitente", "distribuido")
_HOLDUP = {0: (0.98, 0.4846, 0.0868), 2: (0.845, 0.5351, 0.0173), 3: (1.065, 0.5824, 0.0609)}
_UPHILL = {0: (0.011, -3.768, 3.539, -1.614), 2: (2.96, 0.305, -0.4473, 0.0978)}
_DOWNHILL = (4.70, -0.3692, 0.1244, -0.5056)


def _darcy_f(Re: float, eD: float) -> float:
    from fluids import friction_factor
    return 64.0 / Re if Re < 2300.0 else float(friction_factor(Re=Re, eD=eD))


def _bb_regime(lam: float, Fr: float) -> int:
    L1 = 316.0 * lam ** 0.302
    L2 = 0.0009252 * lam ** -2.4684
    L3 = 0.1 * lam ** -1.4516
    L4 = 0.5 * lam ** -6.738
    if (lam < 0.01 and Fr < L1) or (lam >= 0.01 and Fr < L2):
        return 0
    if lam >= 0.01 and L2 <= Fr <= L3:
        return 1
    if (0.01 <= lam < 0.4 and L3 < Fr <= L1) or (lam >= 0.4 and L3 < Fr <= L4):
        return 2
    return 3


def _bb_holdup(regime: int, lam: float, Fr: float, theta: float, NLv: float) -> tuple:
    a, b, c = _HOLDUP[regime]
    HL0 = max(a * lam ** b / Fr ** c, lam)
    if theta == 0.0 or (theta > 0.0 and regime == 3):
        return HL0, 1.0, 0.0
    d, e, f, h = _UPHILL[regime] if theta > 0.0 else _DOWNHILL
    C = max(0.0, (1.0 - lam) * math.log(d * lam ** e * NLv ** f * Fr ** h))
    s = math.sin(1.8 * theta)
    psi = 1.0 + C * (s - s ** 3 / 3.0)
    return HL0 * psi, psi, C


def beggs_brill_gradient(m: float, x: float, rhol: float, rhog: float, mul: float, mug: float, sigma: float, P: float,
                         D: float, angle_deg: float, roughness: float, clamp_holdup: bool = True) -> dict:
    """Local Beggs-Brill state and pressure gradients (Pa/m, positive = pressure falls along the flow). The correlation
    can return HL > 1 at very low gas rates; with clamp_holdup it is cut to 1 (flagged in `holdup_clamped`)."""
    A = math.pi * D * D / 4.0
    vsg, vsl = m * x / (rhog * A), m * (1.0 - x) / (rhol * A)
    vm = vsg + vsl
    lam = vsl / vm
    Fr = vm * vm / (G_ACC * D)
    NLv = vsl * (rhol / (G_ACC * sigma)) ** 0.25
    theta = math.radians(angle_deg)
    reg = _bb_regime(lam, Fr)
    if reg == 1:
        L2, L3 = 0.0009252 * lam ** -2.4684, 0.1 * lam ** -1.4516
        Aw = (L3 - Fr) / (L3 - L2)
        h_s, psi_s, _ = _bb_holdup(0, lam, Fr, theta, NLv)
        h_i, psi_i, _ = _bb_holdup(2, lam, Fr, theta, NLv)
        HL, psi = Aw * h_s + (1.0 - Aw) * h_i, None
    else:
        HL, psi, _ = _bb_holdup(reg, lam, Fr, theta, NLv)
    clamped = clamp_holdup and HL > 1.0
    if clamped:
        HL = 1.0
    rho_s = rhol * HL + rhog * (1.0 - HL)
    rho_ns = rhol * lam + rhog * (1.0 - lam)
    mu_ns = mul * lam + mug * (1.0 - lam)
    Re_ns = rho_ns * vm * D / mu_ns
    fn = _darcy_f(Re_ns, roughness / D)
    y = lam / (HL * HL)
    if 1.0 < y < 1.2:
        S = math.log(2.2 * y - 1.2)
    else:
        ly = math.log(y)
        S = ly / (-0.0523 + 3.182 * ly - 0.8725 * ly ** 2 + 0.01853 * ly ** 4)
    S = min(S, 7.0)
    ftp = fn * math.exp(S)
    grav = rho_s * G_ACC * math.sin(theta)
    fric = ftp * rho_ns * vm * vm / (2.0 * D)
    Ek = vsg * vm * rho_s / P
    return {"regime_index": reg, "regime": REGIMES[reg], "no_slip_holdup": lam, "Froude": Fr, "NLv": NLv, "holdup": HL,
            "inclination_factor_psi": psi, "holdup_clamped": clamped, "slip_density_kg_m3": rho_s,
            "no_slip_density_kg_m3": rho_ns, "Re_no_slip": Re_ns,
            "fn": fn, "S": S, "f_tp": ftp, "vsg_m_s": vsg, "vsl_m_s": vsl, "vm_m_s": vm, "Ek": Ek,
            "dPdz_gravity_Pa_m": grav, "dPdz_friction_Pa_m": fric,
            "dPdz_total_Pa_m": (grav + fric) / (1.0 - Ek) if Ek < 1.0 else math.inf}


def _whalley_method(mul: float, mug: float, G: float) -> str:
    ratio = mul / mug
    return "Friedel" if ratio < 1000.0 else ("Chisholm" if G > 100.0 else "Lockhart_Martinelli")


def _march(n: int, method: str, kw: dict, P_in: float, rhog_in: float, L: float) -> dict:
    """March the line in n segments; each segment at its mean pressure (fixed point on the outlet pressure)."""
    from fluids.two_phase import two_phase_dP, two_phase_dP_acceleration
    from fluids.two_phase_voidage import Woldesemayat_Ghajar
    m, x, rhol, D, ang = kw["m"], kw["x"], kw["rhol"], kw["D"], kw["angle"]
    dL = L / n
    A = math.pi * D * D / 4.0
    P = P_in
    prof = [{"x_m": 0.0, "P_Pa": P_in}]
    tot = {"gravity": 0.0, "friction": 0.0, "acceleration": 0.0}
    clamped = False
    fric_method = _whalley_method(kw["mul"], kw["mug"], m / A)

    def alpha(Pk: float) -> float:
        return float(Woldesemayat_Ghajar(x, rhol, rhog_in * Pk / P_in, kw["sigma"], m, D, Pk, ang))

    for i in range(n):
        Pout = P
        for _ in range(50):
            Pm = 0.5 * (P + Pout)
            if Pm <= 0.0:
                raise ValueError("la presión cae a cero dentro de la línea: el flujo no es posible con estos datos")
            rg = rhog_in * Pm / P_in
            if method == "beggs_brill":
                s = beggs_brill_gradient(m, x, rhol, rg, kw["mul"], kw["mug"], kw["sigma"], Pm, D, ang, kw["roughness"])
                clamped = clamped or s["holdup_clamped"]
                if s["Ek"] >= 1.0:
                    raise ValueError("término de aceleración de Beggs-Brill Ek ≥ 1: flujo crítico en la línea")
                dg, df = s["dPdz_gravity_Pa_m"] * dL, s["dPdz_friction_Pa_m"] * dL
                da = (dg + df) * s["Ek"] / (1.0 - s["Ek"])
            else:
                a_m = alpha(Pm)
                dg = (a_m * rg + (1.0 - a_m) * rhol) * G_ACC * math.sin(math.radians(ang)) * dL
                df = float(two_phase_dP(m=m, x=x, rhol=rhol, D=D, L=dL, rhog=rg, mul=kw["mul"], mug=kw["mug"],
                                        sigma=kw["sigma"], P=Pm, roughness=kw["roughness"], Method=fric_method))
                Pe = max(Pout, 1e-6 * P)
                da = float(two_phase_dP_acceleration(m=m, D=D, xi=x, xo=x, alpha_i=alpha(P), alpha_o=alpha(Pe),
                                                     rho_li=rhol, rho_gi=rhog_in * P / P_in, rho_lo=rhol,
                                                     rho_go=rhog_in * Pe / P_in))
                s = {"holdup": 1.0 - a_m, "regime": None}
            new = P - (dg + df + da)
            if abs(new - Pout) <= 1e-12 * P_in:
                Pout = new
                break
            Pout = new
        else:
            raise ValueError("el punto fijo de presión de un segmento no convergió")
        tot["gravity"] += dg
        tot["friction"] += df
        tot["acceleration"] += da
        P = Pout
        if P <= 0.0:
            raise ValueError("la presión cae a cero dentro de la línea: el flujo no es posible con estos datos")
        prof.append({"x_m": (i + 1) * dL, "P_Pa": P, "holdup": s["holdup"], "regime": s["regime"]})
    return {"P_out_Pa": P, "terms": tot, "profile": prof, "friction_method": fric_method, "holdup_clamped": clamped}


def _thin(profile: list, keep: int = 41) -> list:
    if len(profile) <= keep:
        return profile
    step = (len(profile) - 1) / (keep - 1)
    return [profile[round(i * step)] for i in range(keep)]


def two_phase_pipe_inclined(mass_flow_kg_s: float, quality: float, liquid_density_kg_m3: float,
                            gas_density_kg_m3: float, liquid_viscosity_Pa_s: float, gas_viscosity_Pa_s: float,
                            surface_tension_N_m: float, inner_diameter_m: float, length_m: float, angle_deg: float,
                            inlet_pressure_Pa: float, roughness_m: float = STEEL_ROUGHNESS_M, n_segments: int = 20) -> dict:
    data = {"mass_flow_kg_s": mass_flow_kg_s, "quality": quality, "liquid_density_kg_m3": liquid_density_kg_m3,
            "gas_density_kg_m3": gas_density_kg_m3, "liquid_viscosity_Pa_s": liquid_viscosity_Pa_s,
            "gas_viscosity_Pa_s": gas_viscosity_Pa_s, "surface_tension_N_m": surface_tension_N_m,
            "inner_diameter_m": inner_diameter_m, "length_m": length_m, "angle_deg": angle_deg,
            "inlet_pressure_Pa": inlet_pressure_Pa}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        raise ValueError(f"faltan datos del flujo bifásico: {missing} (propiedades de ambas fases a la entrada, geometría, "
                         "ángulo sobre la horizontal y presión de entrada)")
    if not 0.0 < quality < 1.0:
        raise ValueError("quality (fracción másica de gas) debe estar estrictamente entre 0 y 1; si es una sola fase usar "
                         "la hidráulica monofásica")
    if not -90.0 <= angle_deg <= 90.0:
        raise ValueError("angle_deg debe estar entre −90 (vertical descendente) y +90 (vertical ascendente)")
    for k in ("mass_flow_kg_s", "liquid_density_kg_m3", "gas_density_kg_m3", "liquid_viscosity_Pa_s", "gas_viscosity_Pa_s",
              "surface_tension_N_m", "inner_diameter_m", "length_m", "inlet_pressure_Pa"):
        if not data[k] > 0:
            raise ValueError(f"{k} debe ser > 0")
    if gas_density_kg_m3 >= liquid_density_kg_m3:
        raise ValueError("la densidad del gas debe ser menor que la del líquido")
    n = int(n_segments)
    if n < 1:
        raise ValueError("n_segments debe ser ≥ 1")
    kw = dict(m=mass_flow_kg_s, x=quality, rhol=liquid_density_kg_m3, mul=liquid_viscosity_Pa_s,
              mug=gas_viscosity_Pa_s, sigma=surface_tension_N_m, D=inner_diameter_m, angle=float(angle_deg),
              roughness=roughness_m)
    P1, rg1, L = float(inlet_pressure_Pa), float(gas_density_kg_m3), float(length_m)
    warnings: list[str] = []
    bb = _march(n, "beggs_brill", kw, P1, rg1, L)
    bb2 = _march(2 * n, "beggs_brill", kw, P1, rg1, L)
    sep = _march(n, "separated", kw, P1, rg1, L)
    inlet = beggs_brill_gradient(mass_flow_kg_s, quality, liquid_density_kg_m3, rg1, liquid_viscosity_Pa_s,
                                 gas_viscosity_Pa_s, surface_tension_N_m, P1, inner_diameter_m, angle_deg, roughness_m)
    rg_out = rg1 * bb["P_out_Pa"] / P1
    outlet = beggs_brill_gradient(mass_flow_kg_s, quality, liquid_density_kg_m3, rg_out, liquid_viscosity_Pa_s,
                                  gas_viscosity_Pa_s, surface_tension_N_m, bb["P_out_Pa"], inner_diameter_m, angle_deg,
                                  roughness_m)
    # ---- evidence 1: independent implementation (fluids, which does not clip HL) at the inlet state
    from fluids.two_phase import Beggs_Brill, Taitel_Dukler_regime
    raw = beggs_brill_gradient(mass_flow_kg_s, quality, liquid_density_kg_m3, rg1, liquid_viscosity_Pa_s,
                               gas_viscosity_Pa_s, surface_tension_N_m, P1, inner_diameter_m, angle_deg, roughness_m,
                               clamp_holdup=False)
    ref = float(Beggs_Brill(m=mass_flow_kg_s, x=quality, rhol=liquid_density_kg_m3, rhog=rg1, mul=liquid_viscosity_Pa_s,
                            mug=gas_viscosity_Pa_s, sigma=surface_tension_N_m, P=P1, D=inner_diameter_m,
                            angle=float(angle_deg), roughness=roughness_m, L=1.0, acceleration=True))
    dev_fluids = abs(raw["dPdz_total_Pa_m"] - ref) / max(abs(ref), 1e-300)
    if bb["holdup_clamped"]:
        warnings.append("Beggs-Brill dio holdup de líquido > 1 (gas muy escaso): se recortó a 1; la correlación está fuera "
                        "de su zona útil, tratar el flujo como casi monofásico")
    dP_bb = P1 - bb["P_out_Pa"]
    dP_bb2 = P1 - bb2["P_out_Pa"]
    seg_dev = abs(dP_bb - dP_bb2) / max(abs(dP_bb2), 1e-9 * P1)
    reasons = []
    if dev_fluids > FLUIDS_MATCH_TOL:
        reasons.append(f"el gradiente de Beggs-Brill no coincide con fluids.Beggs_Brill ({dev_fluids:.2e})")
    if seg_dev > SEGMENT_TOL:
        reasons.append(f"el ΔP cambia {seg_dev:.2%} al duplicar los segmentos: aumentar n_segments")
    dP_sep = P1 - sep["P_out_Pa"]
    spread = abs(dP_sep - dP_bb) / max(abs(dP_bb), 1e-9 * P1)
    td = None
    if abs(angle_deg) <= 10.0:
        td = Taitel_Dukler_regime(m=mass_flow_kg_s, x=quality, rhol=liquid_density_kg_m3, rhog=rg1,
                                  mul=liquid_viscosity_Pa_s, mug=gas_viscosity_Pa_s, D=inner_diameter_m,
                                  angle=float(angle_deg), roughness=roughness_m)[0]
    if max(inlet["Ek"], outlet["Ek"]) > 0.1:
        warnings.append(f"Ek = {max(inlet['Ek'], outlet['Ek']):.2f}: la aceleración pesa; cerca de Ek → 1 el flujo es "
                        "crítico y Beggs-Brill pierde validez")
    if dP_bb > 0.2 * P1:
        warnings.append(f"ΔP = {dP_bb / P1:.0%} de P de entrada: la calidad constante (sin flasheo) y el gas ideal "
                        "isotérmico dejan de ser buenas hipótesis; usar la tubería de DWSIM que hace flash por incremento")
    if angle_deg < 0:
        warnings.append("Woldesemayat-Ghajar se ajustó a flujo horizontal y ascendente: en descendente el modelo separado "
                        "es solo orientativo")
    if spread > 0.3:
        warnings.append(f"los dos métodos difieren {spread:.0%}: dispersión típica de correlaciones bifásicas (±30-50 %); "
                        "reportar el intervalo")
    if bb["P_out_Pa"] > P1:
        warnings.append("la presión sube a lo largo de la línea (descendente con ganancia hidrostática mayor que la "
                        "fricción)")
    sep_terms = sep["terms"]
    return {
        "P_out_Pa": bb["P_out_Pa"], "dP_total_Pa": dP_bb, "dP_gravity_Pa": bb["terms"]["gravity"],
        "dP_friction_Pa": bb["terms"]["friction"], "dP_acceleration_Pa": bb["terms"]["acceleration"],
        "recommended_method": "Beggs-Brill",
        "flow_pattern_inlet": inlet["regime"], "flow_pattern_outlet": outlet["regime"],
        "flow_pattern_Taitel_Dukler_inlet": td if td is not None else "no evaluado (|ángulo| > 10°: mapa de tubo casi "
                                                                       "horizontal)",
        "holdup_inlet": inlet["holdup"], "holdup_outlet": outlet["holdup"],
        "no_slip_holdup_inlet": inlet["no_slip_holdup"], "inlet_state": inlet, "outlet_state": outlet,
        "profile": _thin(bb["profile"]),
        "separated_model": {"P_out_Pa": sep["P_out_Pa"], "dP_total_Pa": dP_sep, "dP_gravity_Pa": sep_terms["gravity"],
                            "dP_friction_Pa": sep_terms["friction"], "dP_acceleration_Pa": sep_terms["acceleration"],
                            "friction_method": sep["friction_method"],
                            "void_fraction_method": "Woldesemayat-Ghajar (2007)",
                            "holdup_inlet": sep["profile"][1]["holdup"]},
        "method_spread_rel": spread,
        "evidence": {"inlet_gradient_vs_fluids_Beggs_Brill_rel": dev_fluids, "compared_unclipped_holdup": True,
                     "fluids_gradient_Pa_m": ref,
                     "segments": n, "dP_with_2n_segments_Pa": dP_bb2, "segment_independence_rel": seg_dev,
                     "verified": not reasons, "reasons": reasons},
        "warnings": warnings,
        "assumptions": ["calidad constante a lo largo de la línea (sin flasheo ni transferencia de masa)",
                        "densidad del gas proporcional a la presión (gas ideal isotérmico, Z constante); líquido "
                        "incompresible; viscosidades y tensión superficial constantes",
                        "ángulo constante en todo el tramo (positivo = ascendente)",
                        "factor de fricción normalizante de Beggs-Brill por Colebrook con la rugosidad dada (el original "
                        "usa tubo liso)" + (f"; rugosidad de acero comercial {STEEL_ROUGHNESS_M} m por omisión"
                                            if roughness_m == STEEL_ROUGHNESS_M else "")],
        "not_included": ["correcciones de holdup de Payne et al. (1979)", "balance de energía (temperatura constante)",
                         "slugging severo o terreno ondulado (un solo ángulo por tramo)",
                         "mapas de patrón de flujo para tubo vertical (el de Beggs-Brill es el horizontal)"],
        "method": ("Beggs-Brill (1973) con transición de Brill-Beggs (1991), implementado aquí y verificado contra "
                   "fluids.two_phase.Beggs_Brill; modelo separado Woldesemayat-Ghajar + "
                   f"{sep['friction_method']} + aceleración (fluids); marcha en {n} segmentos a presión media"),
    }
