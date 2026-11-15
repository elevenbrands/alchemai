"""Steam distribution losses as physical quantities (bloque M20b): leaks, failed-open traps and bare pipe.

What it answers (the caller prices the result with its own cost of steam):
  * `steam_leak_loss`: steam lost through a hole, kg/s, from the absolute pressure, the hole diameter, the back pressure and
    a discharge coefficient.
  * `failed_trap_loss`: the same physics for groups of steam traps that failed open (the survey gives the count).
  * `bare_surface_heat_loss`: heat lost by horizontal bare pipe, by the same pipe insulated, the saving and the steam that
    condenses to supply it.
  * `saturation_T_K`: saturation temperature of water at a pressure.

Property data (nothing is typed here): water and steam from IAPWS-95, dry air (density, viscosity, conductivity, cp,
expansion coefficient) from the Lemmon pseudo-pure fluid, both through CoolProp. Stefan-Boltzmann constant and g from
scipy.constants.

Flow through the hole: isentropic expansion from the stagnation state in the line (velocity in the line neglected) to the
throat, in thermodynamic equilibrium (the steam that crosses the saturation line condenses: homogeneous equilibrium model).
    G(p) = rho(p, s0) * sqrt(2 * (h0 - h(p, s0)))            mass flux at a throat pressure p
    throat pressure = the p in [back pressure, line pressure] that maximises G; choked when that p is above the back pressure
    mass flow = discharge_coefficient * area * G(throat)
Napier's formula (ORNL/TM-2001/263 eq. 56: lb/h = 51.43 * A[in2] * P[psia], i.e. lb/s = A * P / 70) is evaluated only as a
cross-check, with the same discharge coefficient, when the steam is saturated and the flow is choked.

Bare and insulated pipe, per segment, with the film properties of air at the mean of surface and ambient temperature:
    natural convection: Churchill-Chu (1975) for a horizontal cylinder (`ht.Nu_horizontal_cylinder_Churchill_Chu`)
    wind > 0: Churchill-Bernstein (1977) for a cylinder in cross flow (`ht.Nu_cylinder_Churchill_Bernstein`), combined
              with the natural one as Nu^4 = Nu_forced^4 + Nu_natural^4 (Bergman-Lavine-Incropera-DeWitt eq. 9.64, the
              exponent they give for transverse flow over cylinders)
    radiation: emissivity * sigma * area * (Ts^4 - Tambient^4), surroundings at ambient temperature
    insulated: the jacket temperature that makes 2*pi*k*L*(Tpipe - Tjacket)/ln(D2/D1) equal convection + radiation outside

Operating data and model constants have no defaults: pressures, diameters, temperatures, wind, ambient pressure, the
discharge coefficient and the emissivities are given by the caller; a missing one raises ValueError saying what it is for.

Measured against published numbers (tests/test_steam_distribution.py, sources read on 2026-11-14, nothing tuned):
  * U.S. DOE Steam Tip Sheet #1, leaking trap table, Cd 0.72 to atmosphere as stated: 23 values, -3.8 % to +2.9 % (the table
    is linear in pressure, 1.3 % above 0.72 * Napier; real steam is not linear).
  * ORNL/TM-2001/263 Table 6, steam at 500 F, Cd 0.60 (the «approximately 60 %» of its text): 48 values within -0.8 % /
    +0.4 %; its 300 psig column (8 values) is 4 % above its own neighbours.
  * Incropera and DeWitt example 9.4 as worked by Yovanovich (every input stated): bare pipe +0.4 %, insulated +0.1 % and
    -0.4 K in the jacket temperature.
  * NOT validations, only sensitivities, because the emissivity is not stated: DOE Steam Tip Sheet #2 (20 values, no wind:
    +2 % to +16 % with 0.8, -5 % to +8 % with 0.7, growing with diameter) and ORNL/TM-2001/263 Table 7 (78 values, 10 mile/h:
    -3 % to +10 % with 0.8)."""
from __future__ import annotations

import math

import CoolProp
import CoolProp.CoolProp as CP
from ht import Nu_cylinder_Churchill_Bernstein, Nu_horizontal_cylinder_Churchill_Chu
from scipy import constants
from scipy.optimize import brentq, minimize_scalar

SIGMA = constants.Stefan_Boltzmann
G_ACCEL = constants.g
NAPIER_DIVISOR = 70.0   # Napier: lb/s = A[in2] * P[psia] / 70 (ORNL/TM-2001/263 eq. 56 writes it as 51.43 lb/h)
NAPIER_SI = constants.pound / (NAPIER_DIVISOR * constants.inch ** 2 * constants.psi)   # kg/s per (m2 * Pa)
MIXED_EXPONENT = 4.0    # Bergman et al., Fundamentals of Heat and Mass Transfer, eq. 9.64: transverse flow over cylinders
GRID_POINTS = 80        # throat-pressure grid before the local refinement
FD_STEP = 1e-4          # relative pressure step of the finite-difference speed of sound (Mach check)
MACH_TOL = 5e-3
BALANCE_TOL = 1e-8
P_TRIPLE = CP.PropsSI("ptriple", "Water")
P_CRIT = CP.PropsSI("pcrit", "Water")
WATER_SOURCE = f"IAPWS-95 a través de CoolProp {CoolProp.__version__}"
AIR_SOURCE = f"aire seco (Lemmon et al.) a través de CoolProp {CoolProp.__version__}"
MODULE = "packages/energy/steam_distribution.py"
ENTHALPY_REFERENCE = "IAPWS-95: energía interna y entropía cero en el líquido saturado del punto triple"

SEGMENT_KEYS = {"label", "outer_diameter_m", "length_m", "surface_T_K", "steam_pressure_Pa", "surface_emissivity",
                "insulation_thickness_m", "insulation_conductivity_W_m_K", "jacket_emissivity"}
TRAP_KEYS = {"label", "count", "orifice_diameter_m", "steam_pressure_Pa", "back_pressure_Pa"}


def _positive(value, name: str, purpose: str, allow_zero: bool = False) -> float:
    """Required numeric input: Spanish error naming what is missing and what it is for."""
    if value is None:
        raise ValueError(f"falta {name}: {purpose}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} debe ser un número ({purpose}); se recibió {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} debe ser {'mayor o igual a' if allow_zero else 'mayor que'} cero ({purpose}); "
                         f"se recibió {value!r}")
    return v


def _fraction(value, name: str, purpose: str) -> float:
    v = _positive(value, name, purpose)
    if v > 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1 ({purpose}); se recibió {value!r}")
    return v


def _steam_pressure(value, name: str, purpose: str) -> float:
    p = _positive(value, name, purpose)
    if not P_TRIPLE < p < P_CRIT:
        raise ValueError(f"{name} = {p:.6g} Pa fuera del intervalo de saturación del agua ({P_TRIPLE:.1f} Pa a "
                         f"{P_CRIT:.0f} Pa absolutos); revisa que sea presión absoluta en Pa")
    return p


def saturation_T_K(pressure_Pa: float) -> float:
    """Saturation temperature of water (IAPWS-95) at an absolute pressure, K."""
    p = _steam_pressure(pressure_Pa, "pressure_Pa", "presión absoluta a la que se quiere la temperatura de saturación")
    return float(CP.PropsSI("T", "P", p, "Q", 1.0, "Water"))


def _latent_heat(pressure_Pa: float) -> float:
    return float(CP.PropsSI("H", "P", pressure_Pa, "Q", 1.0, "Water") - CP.PropsSI("H", "P", pressure_Pa, "Q", 0.0, "Water"))


# ----------------------------------------------------------------------------------------------------------------- leaks
def _nozzle(steam_pressure_Pa: float, back_pressure_Pa: float, superheat_K: float) -> dict:
    """Isentropic equilibrium expansion of steam from the line to the throat; mass flux per unit of open area."""
    st = CP.AbstractState("HEOS", "Water")
    st.update(CP.PQ_INPUTS, steam_pressure_Pa, 1.0)
    T_sat, h_g = st.T(), st.hmass()
    st.update(CP.PQ_INPUTS, steam_pressure_Pa, 0.0)
    h_f = st.hmass()
    if superheat_K > 0:
        st.update(CP.PT_INPUTS, steam_pressure_Pa, T_sat + superheat_K)
    else:
        st.update(CP.PQ_INPUTS, steam_pressure_Pa, 1.0)
    T0, h0, s0, rho0 = st.T(), st.hmass(), st.smass(), st.rhomass()

    def state(p: float) -> tuple:
        st.update(CP.PSmass_INPUTS, p, s0)
        return st.hmass(), st.rhomass()

    def flux(p: float) -> float:
        h, rho = state(p)
        return rho * math.sqrt(max(2.0 * (h0 - h), 0.0))

    r_back = back_pressure_Pa / steam_pressure_Pa
    ratios = [r_back + (1.0 - r_back) * i / (GRID_POINTS - 1) for i in range(GRID_POINTS)]
    grid = [flux(r * steam_pressure_Pa) for r in ratios]
    i_max = max(range(GRID_POINTS), key=grid.__getitem__)
    lo, hi = ratios[max(i_max - 1, 0)], ratios[min(i_max + 1, GRID_POINTS - 1)]
    best = minimize_scalar(lambda r: -flux(r * steam_pressure_Pa), bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-12})
    G_back = grid[0]
    r_throat, G_throat = (float(best.x), float(-best.fun)) if -best.fun > grid[i_max] else (ratios[i_max], grid[i_max])
    choked = bool(G_throat > G_back * (1.0 + 1e-10) and r_throat > r_back * (1.0 + 1e-9))
    if not choked:
        r_throat, G_throat = r_back, G_back
    p_t = r_throat * steam_pressure_Pa

    h_t, rho_t = state(p_t)
    T_t = st.T()
    quality = st.Q()
    quality = float(quality) if 0.0 <= quality <= 1.0 else None
    velocity = math.sqrt(max(2.0 * (h0 - h_t), 0.0))
    st.update(CP.HmassP_INPUTS, h_t, p_t)                     # independent flash of the throat state: must return s0
    entropy_residual = abs(st.smass() - s0) / abs(s0)
    # Mach number on each side of the throat from a finite-difference (dp/drho)_s: <= 1 upstream, >= 1 downstream
    dp = FD_STEP * p_t
    mach_up = velocity / math.sqrt(dp / (state(p_t + dp)[1] - rho_t))
    mach_down = velocity / math.sqrt(dp / (rho_t - state(p_t - dp)[1])) if choked else None
    return {"T_sat": T_sat, "T0": T0, "h0": h0, "s0": s0, "rho0": rho0, "latent": h_g - h_f, "choked": choked,
            "p_throat": p_t, "G": G_throat, "G_back": G_back, "T_throat": T_t, "rho_throat": rho_t, "quality": quality,
            "velocity": velocity, "entropy_residual": entropy_residual, "mach_up": mach_up, "mach_down": mach_down,
            "grid_excess": max(0.0, max(grid) / G_throat - 1.0)}


def steam_leak_loss(steam_pressure_Pa: float | None = None, hole_diameter_m: float | None = None,
                    discharge_coefficient: float | None = None, back_pressure_Pa: float | None = None,
                    superheat_K: float = 0.0) -> dict:
    """Steam lost through a hole or a leaking fitting, kg/s; see the module docstring.

    Pressures are absolute. `discharge_coefficient` is the ratio of the real flow to the ideal isentropic nozzle flow through
    the hole area (it depends on the shape of the hole; it has no default). `superheat_K` = 0 means dry saturated steam."""
    p0 = _steam_pressure(steam_pressure_Pa, "steam_pressure_Pa", "presión absoluta del vapor en la línea que fuga")
    d = _positive(hole_diameter_m, "hole_diameter_m", "diámetro del orificio de la fuga, para el área de paso")
    cd = _fraction(discharge_coefficient, "discharge_coefficient",
                   "coeficiente de descarga del orificio (flujo real / flujo ideal de tobera isentrópica)")
    pb = _positive(back_pressure_Pa, "back_pressure_Pa",
                   "presión absoluta a la que descarga la fuga (atmósfera local o cabezal de retorno), para saber si el "
                   "flujo es crítico")
    sh = _positive(superheat_K, "superheat_K", "sobrecalentamiento del vapor sobre la saturación (0 = saturado)",
                   allow_zero=True)
    if pb < P_TRIPLE:
        raise ValueError(f"back_pressure_Pa = {pb:.6g} Pa está por debajo del punto triple del agua ({P_TRIPLE:.1f} Pa); "
                         "el modelo de expansión en equilibrio no cubre descarga a hielo")
    if pb >= p0:
        raise ValueError(f"back_pressure_Pa ({pb:.6g} Pa) debe ser menor que steam_pressure_Pa ({p0:.6g} Pa): sin "
                         "diferencia de presión no hay fuga")
    n = _nozzle(p0, pb, sh)
    area = math.pi * d ** 2 / 4.0
    ideal = area * n["G"]
    flow = cd * ideal

    warnings, reasons = [], []
    saturated = sh == 0.0
    if not n["choked"]:
        warnings.append(f"flujo subcrítico (contrapresión/presión = {pb / p0:.3f}): el caudal depende de la contrapresión; "
                        "si la contrapresión real es menor, la fuga es mayor")
    if saturated and n["choked"]:
        napier_flow = cd * NAPIER_SI * area * p0
        napier = {"applicable": True, "mass_flow_kg_s": napier_flow, "difference_fraction": napier_flow / flow - 1.0,
                  "reason": None}
    else:
        why = ("la fórmula de Napier es para vapor saturado" if not saturated else
               "la fórmula de Napier supone flujo crítico (contrapresión menor que ~0.58 de la presión)")
        napier = {"applicable": False, "mass_flow_kg_s": None, "difference_fraction": None, "reason": why}
    napier["formula"] = "lb/s = Cd * A[in2] * P[psia] / 70 (ORNL/TM-2001/263 ec. 56: 51.43 lb/h por in2 y psia)"

    if n["entropy_residual"] > 1e-8:
        reasons.append("el estado de garganta no conserva la entropía de la línea")
    if n["grid_excess"] > 1e-9:
        reasons.append("la garganta no es el máximo de flujo másico en el intervalo de presiones")
    if n["mach_up"] > 1.0 + MACH_TOL:
        reasons.append("Mach mayor que 1 aguas arriba de la garganta")
    if n["choked"] and n["mach_down"] < 1.0 - MACH_TOL:
        reasons.append("flujo declarado crítico con Mach menor que 1 aguas abajo de la garganta")
    evidence = {"verified": not reasons, "reasons": reasons, "entropy_residual": n["entropy_residual"],
                "mass_flux_excess_on_grid": n["grid_excess"], "mach_upstream_side_of_throat": n["mach_up"],
                "mach_downstream_side_of_throat": n["mach_down"],
                "mass_flux_at_back_pressure_over_throat": n["G_back"] / n["G"]}

    assumptions = [
        ("vapor saturado seco (título 1) a la presión de la línea: no se dio sobrecalentamiento" if saturated else
         f"vapor sobrecalentado {sh:.1f} K sobre la saturación"),
        "expansión isentrópica desde el reposo en la línea hasta la garganta, en equilibrio termodinámico (el vapor que "
        "cruza la línea de saturación condensa: modelo homogéneo de equilibrio)",
        f"coeficiente de descarga {cd:g} dado por quien llama: multiplica el flujo ideal de tobera sobre el área del "
        "orificio",
        "orificio circular del diámetro dado; presión de la línea constante (la fuga no la abate)",
    ]
    return {
        "mass_flow_kg_s": flow, "choked": n["choked"], "ideal_mass_flow_kg_s": ideal, "mass_flux_ideal_kg_m2_s": n["G"],
        "hole_area_m2": area, "hole_diameter_m": d, "discharge_coefficient": cd,
        "steam_pressure_Pa": p0, "back_pressure_Pa": pb, "pressure_ratio": pb / p0,
        "critical_pressure_ratio": n["p_throat"] / p0 if n["choked"] else None,
        "steam_enthalpy_J_kg": n["h0"], "latent_heat_J_kg": n["latent"],
        "steam": {"T_K": n["T0"], "saturation_T_K": n["T_sat"], "superheat_K": sh, "density_kg_m3": n["rho0"],
                  "enthalpy_J_kg": n["h0"], "entropy_J_kg_K": n["s0"]},
        "throat": {"pressure_Pa": n["p_throat"], "T_K": n["T_throat"], "density_kg_m3": n["rho_throat"],
                   "velocity_m_s": n["velocity"], "vapor_quality": n["quality"]},
        "napier": napier,
        "evidence": evidence,
        "warnings": warnings,
        "assumptions": assumptions,
        "not_included": ["flujo metaestable (vapor sobresaturado que no alcanza a condensar): daría un caudal algo mayor",
                         "fricción y longitud del paso (grietas, empaques, roscas): entran solo por el coeficiente de "
                         "descarga",
                         "condensado arrastrado o título menor que 1 en la línea",
                         "valor económico: se calcula con el costo del vapor de quien llama y sus horas de operación"],
        "method": {"name": "expansión isentrópica en equilibrio con verificación de flujo crítico",
                   "module": MODULE, "property_data": WATER_SOURCE,
                   "enthalpy_reference": ENTHALPY_REFERENCE,
                   "cross_check": "fórmula de Napier (solo comparación; no es el motor)"},
    }


def failed_trap_loss(groups: list | None = None, discharge_coefficient: float | None = None) -> dict:
    """Live steam lost through steam traps that failed open; groups of equal traps, same physics as `steam_leak_loss`.

    Each group: {"count", "orifice_diameter_m", "steam_pressure_Pa", "back_pressure_Pa"} and an optional "label". Pressures
    are absolute. The count is the number of traps found failed open by the plant's survey (no failure rate is assumed)."""
    if not groups:
        raise ValueError("falta groups: lista de grupos de trampas falladas abiertas, cada uno con count, "
                         "orifice_diameter_m, steam_pressure_Pa y back_pressure_Pa (salen del censo de trampas)")
    cd = _fraction(discharge_coefficient, "discharge_coefficient",
                   "coeficiente de descarga del orificio de la trampa (flujo real / flujo ideal de tobera isentrópica)")
    rows, reasons, warnings = [], [], []
    for i, g in enumerate(groups):
        where = f"groups[{i}]"
        if not isinstance(g, dict):
            raise ValueError(f"{where} debe ser un diccionario con count, orifice_diameter_m, steam_pressure_Pa y "
                             "back_pressure_Pa")
        unknown = sorted(set(g) - TRAP_KEYS)
        if unknown:
            raise ValueError(f"{where}: claves no reconocidas {unknown}; las válidas son {sorted(TRAP_KEYS)}")
        count = g.get("count")
        if count is None:
            raise ValueError(f"falta {where}.count: número de trampas falladas abiertas de ese grupo según el censo de la "
                             "planta (no se supone un porcentaje de falla)")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{where}.count debe ser un entero mayor que cero (trampas falladas abiertas); se recibió "
                             f"{count!r}")
        d = _positive(g.get("orifice_diameter_m"), f"{where}.orifice_diameter_m",
                      "diámetro del orificio de la trampa, para el área de paso")
        p0 = _steam_pressure(g.get("steam_pressure_Pa"), f"{where}.steam_pressure_Pa",
                             "presión absoluta del vapor en la línea que drena la trampa")
        pb = _positive(g.get("back_pressure_Pa"), f"{where}.back_pressure_Pa",
                       "presión absoluta del retorno de condensado (o atmósfera) a la que descarga la trampa")
        try:
            r = steam_leak_loss(p0, d, cd, pb)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        if not r["evidence"]["verified"]:
            reasons.extend(f"{where}: {x}" for x in r["evidence"]["reasons"])
        warnings.extend(f"{where}: {w}" for w in r["warnings"])
        rows.append({"label": g.get("label"), "count": count, "orifice_diameter_m": r["hole_diameter_m"],
                     "steam_pressure_Pa": r["steam_pressure_Pa"], "back_pressure_Pa": r["back_pressure_Pa"],
                     "choked": r["choked"], "mass_flow_per_trap_kg_s": r["mass_flow_kg_s"],
                     "mass_flow_kg_s": r["mass_flow_kg_s"] * count, "steam_enthalpy_J_kg": r["steam_enthalpy_J_kg"],
                     "latent_heat_J_kg": r["latent_heat_J_kg"], "saturation_T_K": r["steam"]["saturation_T_K"],
                     "napier_mass_flow_kg_s": (r["napier"]["mass_flow_kg_s"] * count if r["napier"]["applicable"] else None),
                     "napier_difference_fraction": r["napier"]["difference_fraction"],
                     "evidence": r["evidence"]})
    total = sum(r["mass_flow_kg_s"] for r in rows)
    return {
        "total_mass_flow_kg_s": total, "total_count": sum(r["count"] for r in rows), "discharge_coefficient": cd,
        "total_enthalpy_flow_W": sum(r["mass_flow_kg_s"] * r["steam_enthalpy_J_kg"] for r in rows),
        "groups": rows,
        "evidence": {"verified": not reasons, "reasons": reasons,
                     "sum_residual": abs(total - math.fsum(r["mass_flow_per_trap_kg_s"] * r["count"] for r in rows)),
                     "max_entropy_residual": max(r["evidence"]["entropy_residual"] for r in rows)},
        "warnings": warnings,
        "assumptions": ["cada trampa fallada deja pasar vapor vivo saturado seco por todo su orificio (falla totalmente "
                        "abierta): es una cota superior si el orificio está parcialmente obstruido o pasa condensado",
                        f"coeficiente de descarga {cd:g} dado por quien llama, igual para todos los grupos",
                        "el número de trampas falladas es el del censo de la planta; no se supone un porcentaje de falla",
                        "expansión isentrópica en equilibrio hasta la garganta, la misma de steam_leak_loss"],
        "not_included": ["trampas que fugan sin estar totalmente abiertas y trampas falladas cerradas",
                         "condensado que comparte el orificio con el vapor",
                         "vapor recuperado aguas abajo si el retorno alimenta un tanque de flasheo o un cabezal de baja",
                         "valor económico: se calcula con el costo del vapor de quien llama y sus horas de operación"],
        "method": {"name": "flujo por orificio de trampas falladas abiertas (expansión isentrópica en equilibrio)",
                   "module": MODULE, "property_data": WATER_SOURCE,
                   "enthalpy_reference": ENTHALPY_REFERENCE},
    }


# ------------------------------------------------------------------------------------------------------------ heat loss
def _outside(diameter_m: float, surface_T_K: float, ambient_T_K: float, wind_m_s: float, pressure_Pa: float,
             emissivity: float) -> dict:
    """Convection (natural, or mixed with wind) plus radiation from a horizontal cylinder, per metre of length."""
    T_film = 0.5 * (surface_T_K + ambient_T_K)
    air = CP.AbstractState("HEOS", "Air")
    air.update(CP.PT_INPUTS, pressure_Pa, T_film)
    rho, mu, k, cp = air.rhomass(), air.viscosity(), air.conductivity(), air.cpmass()
    beta = air.isobaric_expansion_coefficient()
    nu = mu / rho
    Pr = cp * mu / k
    dT = surface_T_K - ambient_T_K
    Gr = G_ACCEL * beta * dT * diameter_m ** 3 / nu ** 2
    Ra = Gr * Pr
    Nu_natural = Nu_horizontal_cylinder_Churchill_Chu(Pr, Gr)
    in_range = Ra <= 1e12
    Re, Nu_forced = 0.0, None
    if wind_m_s > 0:
        Re = wind_m_s * diameter_m / nu
        Nu_forced = Nu_cylinder_Churchill_Bernstein(Re, Pr)
        in_range = in_range and Re * Pr >= 0.2
        Nu = (Nu_forced ** MIXED_EXPONENT + Nu_natural ** MIXED_EXPONENT) ** (1.0 / MIXED_EXPONENT)
    else:
        Nu = Nu_natural
    h = Nu * k / diameter_m
    perimeter = math.pi * diameter_m
    q_conv = h * perimeter * dT
    q_rad = emissivity * SIGMA * perimeter * (surface_T_K ** 4 - ambient_T_K ** 4)
    return {"heat_loss_W_m": q_conv + q_rad, "convection_W_m": q_conv, "radiation_W_m": q_rad,
            "convection_coefficient_W_m2_K": h, "Nu": Nu, "Nu_natural": Nu_natural, "Nu_forced": Nu_forced,
            "Rayleigh": Ra, "Reynolds": Re, "Prandtl": Pr, "film_T_K": T_film, "correlations_in_range": bool(in_range)}


def _segment(i: int, seg, ambient_T_K: float, wind_m_s: float, pressure_Pa: float) -> dict:
    where = f"segments[{i}]"
    if not isinstance(seg, dict):
        raise ValueError(f"{where} debe ser un diccionario con outer_diameter_m, length_m, surface_emissivity y "
                         "surface_T_K o steam_pressure_Pa")
    unknown = sorted(set(seg) - SEGMENT_KEYS)
    if unknown:
        raise ValueError(f"{where}: claves no reconocidas {unknown}; las válidas son {sorted(SEGMENT_KEYS)}")
    D = _positive(seg.get("outer_diameter_m"), f"{where}.outer_diameter_m", "diámetro exterior real del tubo desnudo")
    L = _positive(seg.get("length_m"), f"{where}.length_m", "longitud de tubería desnuda de ese diámetro")
    eps = _fraction(seg.get("surface_emissivity"), f"{where}.surface_emissivity",
                    "emisividad de la superficie del tubo desnudo, para la pérdida por radiación")
    assumptions = []
    p_steam = seg.get("steam_pressure_Pa")
    if p_steam is not None:
        p_steam = _steam_pressure(p_steam, f"{where}.steam_pressure_Pa", "presión absoluta del vapor de la línea")
    if seg.get("surface_T_K") is not None:
        Ts = _positive(seg["surface_T_K"], f"{where}.surface_T_K", "temperatura de la superficie del tubo")
    elif p_steam is not None:
        Ts = saturation_T_K(p_steam)
        assumptions.append(f"{where}: superficie del tubo a la temperatura de saturación del vapor ({Ts:.2f} K); se "
                           "desprecian la resistencia de la película interior y de la pared de acero (cota superior)")
    else:
        raise ValueError(f"falta {where}.surface_T_K o {where}.steam_pressure_Pa: temperatura de la superficie del tubo "
                         "(medida) o presión absoluta del vapor para tomar su temperatura de saturación")
    if Ts < ambient_T_K:
        raise ValueError(f"{where}: la superficie ({Ts:.2f} K) está más fría que el ambiente ({ambient_T_K:.2f} K); este "
                         "módulo es para líneas calientes")

    bare = _outside(D, Ts, ambient_T_K, wind_m_s, pressure_Pa, eps)
    bare["surface_T_K"] = Ts
    bare["heat_loss_W"] = bare["heat_loss_W_m"] * L
    latent = _latent_heat(p_steam) if p_steam is not None else None
    out = {"label": seg.get("label"), "outer_diameter_m": D, "length_m": L, "surface_T_K": Ts,
           "surface_emissivity": eps, "steam_pressure_Pa": p_steam, "latent_heat_J_kg": latent,
           "bare": bare, "insulated": None, "saving_W": None, "saving_fraction": None,
           "steam_condensed_bare_kg_s": bare["heat_loss_W"] / latent if latent else None,
           "steam_condensed_insulated_kg_s": None, "steam_saved_kg_s": None, "assumptions": assumptions, "warnings": []}

    ins_keys = ("insulation_thickness_m", "insulation_conductivity_W_m_K", "jacket_emissivity")
    given = [k for k in ins_keys if seg.get(k) is not None]
    if given:
        t = _positive(seg.get("insulation_thickness_m"), f"{where}.insulation_thickness_m",
                      "espesor del aislamiento, para el caso aislado")
        k_ins = _positive(seg.get("insulation_conductivity_W_m_K"), f"{where}.insulation_conductivity_W_m_K",
                          "conductividad térmica del aislamiento a su temperatura media, para el caso aislado")
        eps_j = _fraction(seg.get("jacket_emissivity"), f"{where}.jacket_emissivity",
                          "emisividad de la chaqueta exterior del aislamiento, para la radiación del caso aislado")
        D2 = D + 2.0 * t
        conductance = 2.0 * math.pi * k_ins / math.log(D2 / D)          # W/(m K) of the layer

        def balance(T2: float) -> float:
            return conductance * (Ts - T2) - _outside(D2, T2, ambient_T_K, wind_m_s, pressure_Pa, eps_j)["heat_loss_W_m"]

        T2 = brentq(balance, ambient_T_K, Ts, xtol=1e-12, rtol=1e-14) if Ts > ambient_T_K else Ts
        ins = _outside(D2, T2, ambient_T_K, wind_m_s, pressure_Pa, eps_j)
        conducted = conductance * (Ts - T2)
        ins.update({"surface_T_K": T2, "outer_diameter_m": D2, "insulation_thickness_m": t,
                    "insulation_conductivity_W_m_K": k_ins, "jacket_emissivity": eps_j,
                    "conduction_W_m": conducted, "heat_loss_W": ins["heat_loss_W_m"] * L,
                    "energy_balance_residual": abs(conducted - ins["heat_loss_W_m"]) / conducted if conducted > 0 else 0.0})
        out["insulated"] = ins
        out["saving_W"] = bare["heat_loss_W"] - ins["heat_loss_W"]
        out["saving_fraction"] = out["saving_W"] / bare["heat_loss_W"] if bare["heat_loss_W"] > 0 else None
        if latent:
            out["steam_condensed_insulated_kg_s"] = ins["heat_loss_W"] / latent
            out["steam_saved_kg_s"] = out["saving_W"] / latent
        if out["saving_W"] < 0:
            out["warnings"].append(f"{where}: el aislamiento dado AUMENTA la pérdida (espesor por debajo del radio crítico "
                                   "o chaqueta más emisiva que el tubo)")
        assumptions.append(f"{where}: cara interior del aislamiento a la temperatura de la superficie del tubo; "
                           f"conductividad {k_ins:g} W/(m·K) constante en la capa (dada a su temperatura media)")
    return out


def bare_surface_heat_loss(segments: list | None = None, ambient_T_K: float | None = None,
                           wind_speed_m_s: float | None = None, ambient_pressure_Pa: float | None = None) -> dict:
    """Heat lost by bare horizontal steam pipe and by the same pipe insulated; see the module docstring.

    Each segment: {"outer_diameter_m", "length_m", "surface_emissivity"} plus "surface_T_K" (measured) and/or
    "steam_pressure_Pa" (absolute; gives the saturation temperature as surface temperature when no surface_T_K is given, and
    the latent heat for the equivalent steam). The insulated case is computed for the segments that also carry
    "insulation_thickness_m", "insulation_conductivity_W_m_K" and "jacket_emissivity" (the three together). Optional "label".
    `wind_speed_m_s` = 0 is natural convection (indoor line). `ambient_pressure_Pa` is the local barometric pressure: it sets
    the density of the air, which matters at altitude."""
    if not segments:
        raise ValueError("falta segments: lista de tramos de tubería desnuda, cada uno con outer_diameter_m, length_m, "
                         "surface_emissivity y surface_T_K o steam_pressure_Pa (salen del levantamiento de la planta)")
    Ta = _positive(ambient_T_K, "ambient_T_K", "temperatura del aire alrededor de la tubería")
    wind = _positive(wind_speed_m_s, "wind_speed_m_s",
                     "velocidad del viento sobre la tubería (0 si está bajo techo: convección natural)", allow_zero=True)
    p_amb = _positive(ambient_pressure_Pa, "ambient_pressure_Pa",
                      "presión barométrica local, para la densidad del aire (cambia la convección con la altitud)")
    rows = [_segment(i, s, Ta, wind, p_amb) for i, s in enumerate(segments)]

    warnings, assumptions, reasons = [], [], []
    for r in rows:
        warnings.extend(r.pop("warnings"))
        assumptions.extend(r.pop("assumptions"))
    all_insulated = all(r["insulated"] is not None for r in rows)
    any_insulated = any(r["insulated"] is not None for r in rows)
    all_steam = all(r["latent_heat_J_kg"] is not None for r in rows)
    if any_insulated and not all_insulated:
        warnings.append("solo algunos tramos traen datos de aislamiento: los totales del caso aislado y del ahorro quedan "
                        "en None; cada tramo con datos trae su propio ahorro")
    if not all_steam:
        warnings.append("hay tramos sin steam_pressure_Pa: sin el calor latente no se da el vapor condensado equivalente "
                        "total")
    bare_total = math.fsum(r["bare"]["heat_loss_W"] for r in rows)
    ins_total = math.fsum(r["insulated"]["heat_loss_W"] for r in rows) if all_insulated else None
    residuals = [r["insulated"]["energy_balance_residual"] for r in rows if r["insulated"] is not None]
    in_range = all(r["bare"]["correlations_in_range"] for r in rows) and \
        all(r["insulated"]["correlations_in_range"] for r in rows if r["insulated"] is not None)
    if residuals and max(residuals) > BALANCE_TOL:
        reasons.append("el balance de energía de la capa de aislamiento no cierra")
    if not in_range:
        reasons.append("algún tramo queda fuera del intervalo de las correlaciones (Ra > 1e12 o Re·Pr < 0.2)")
    evidence = {"verified": not reasons, "reasons": reasons,
                "max_insulation_energy_balance_residual": max(residuals) if residuals else None,
                "correlations_in_range": bool(in_range),
                "bare_exceeds_insulated": (all(r["saving_W"] >= 0 for r in rows if r["insulated"] is not None)
                                           if any_insulated else None)}

    assumptions = [
        "tubería horizontal, tramos rectos; superficie isoterma a la temperatura dada",
        "los alrededores radian a la temperatura ambiente (recinto grande o cielo cubierto)",
        ("sin viento: convección natural" if wind == 0 else
         f"viento de {wind:g} m/s perpendicular al tubo (flujo cruzado: es la orientación de mayor pérdida)"),
        "propiedades del aire seco a la temperatura de película y a la presión barométrica dada",
        "emisividades dadas por quien llama, constantes con la temperatura",
    ] + assumptions
    if wind > 0:
        assumptions.append(f"convección forzada y natural combinadas con exponente {MIXED_EXPONENT:g} (Nu^4 = Nu_forzada^4 "
                           "+ Nu_natural^4): valor de Bergman, Lavine, Incropera y DeWitt, ec. 9.64, para flujo transversal "
                           "sobre cilindros; es constante del modelo, no dato de la planta")
    if all_steam:
        assumptions.append("vapor condensado equivalente = pérdida de calor / calor latente a la presión de la línea "
                           "(vapor saturado que condensa sin subenfriar)")
    natural = "Churchill-Chu (1975), cilindro horizontal, convección natural (ht.Nu_horizontal_cylinder_Churchill_Chu)"
    forced = ("Churchill-Bernstein (1977), cilindro en flujo cruzado (ht.Nu_cylinder_Churchill_Bernstein), combinada con la "
              "natural como Nu^4 = Nu_forzada^4 + Nu_natural^4 (Bergman, Lavine, Incropera y DeWitt, ec. 9.64)")
    return {
        "bare_heat_loss_W": bare_total, "insulated_heat_loss_W": ins_total,
        "saving_W": bare_total - ins_total if all_insulated else None,
        "saving_fraction": (1.0 - ins_total / bare_total) if all_insulated and bare_total > 0 else None,
        "total_length_m": math.fsum(r["length_m"] for r in rows),
        "steam_condensed_bare_kg_s": math.fsum(r["steam_condensed_bare_kg_s"] for r in rows) if all_steam else None,
        "steam_condensed_insulated_kg_s": (math.fsum(r["steam_condensed_insulated_kg_s"] for r in rows)
                                           if all_steam and all_insulated else None),
        "steam_saved_kg_s": math.fsum(r["steam_saved_kg_s"] for r in rows) if all_steam and all_insulated else None,
        "ambient_T_K": Ta, "wind_speed_m_s": wind, "ambient_pressure_Pa": p_amb,
        "segments": rows,
        "evidence": evidence,
        "warnings": warnings,
        "assumptions": assumptions,
        "not_included": ["válvulas, bridas, trampas, soportes y demás accesorios sin aislar (dan su propia superficie)",
                         "tubería vertical y superficies planas (tanques, cabezales de caldera)",
                         "radiación solar, lluvia, aislamiento mojado o dañado y humedad del aire",
                         "resistencia de la película interior y de la pared del tubo; variación de la conductividad del "
                         "aislamiento con la temperatura",
                         "valor económico: se calcula con el costo del vapor de quien llama y sus horas de operación"],
        "method": {"name": "convección + radiación en cilindro horizontal; capa cilíndrica de aislamiento por balance",
                   "module": MODULE, "natural_convection": natural, "forced_convection": forced if wind > 0 else None,
                   "radiation": "cuerpo gris pequeño en recinto grande: emisividad·sigma·área·(Ts^4 - Tamb^4)",
                   "property_data": f"{AIR_SOURCE}; calor latente y saturación: {WATER_SOURCE}"},
    }
