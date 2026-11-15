"""Reverse-flow gas cyclone by the Lapple method (bloque MP).

What it answers: inlet velocity, number of effective turns, cut size (50 % grade efficiency), grade-efficiency curve,
overall collection efficiency over the user's particle size distribution, and pressure drop.

    Ne  = (L_body + L_cone/2) / a                                   (Lapple; a = inlet height)
    d50 = sqrt(9 mu b / (2 pi Ne V_i (rho_p - rho_g)))               (b = inlet width, V_i = Q/(a b))
    eta(d) = 1 / (1 + (d50/d)^2)                                     (Lapple grade efficiency)
    eta_overall = sum_i x_i eta(d_i)                                 (user's mass fractions at representative sizes)
    H_v = K a b / De^2,  dP = H_v · ½ rho_g V_i^2                   (Shepherd-Lapple; K = 16 for a tangential inlet)

Standard geometries are ratios to the body diameter D only (the user gives D): Lapple general purpose, Stairmand high
efficiency and high throughput, Swift high efficiency, general purpose and high throughput (Leith and Licht table, as
reproduced by Utikar et al., IntechOpen 2010, DOI 10.5772/7106; Lapple body length 2D as in Heinsohn & Cimbala via Penn
State ME433 notes and the powderprocess.net table). Body length h and total height H: cone length = H - h.

Not included: Barth-Muschelknautz / Muschelknautz method (wall friction, mass-loading effect), saltation, natural
vortex length, re-entrainment, the effect of solids loading on dP and efficiency, scroll or axial inlets (Shepherd-Lapple
K = 16 is for a tangential inlet), cyclones in series or multicyclones."""
from __future__ import annotations

import math

PRESETS = {   # a/D, b/D, De/D, S/D, h/D (body), H/D (total), B/D (dust outlet)
    "lapple": {"inlet_height": 0.5, "inlet_width": 0.25, "outlet_diameter": 0.5, "outlet_length": 0.625,
               "body_length": 2.0, "total_height": 4.0, "dust_outlet": 0.25, "inlet": "tangencial",
               "label": "Lapple (1951) propósito general"},
    "stairmand_he": {"inlet_height": 0.5, "inlet_width": 0.2, "outlet_diameter": 0.5, "outlet_length": 0.5,
                     "body_length": 1.5, "total_height": 4.0, "dust_outlet": 0.375, "inlet": "tangencial",
                     "label": "Stairmand (1951) alta eficiencia"},
    "stairmand_ht": {"inlet_height": 0.75, "inlet_width": 0.375, "outlet_diameter": 0.75, "outlet_length": 0.875,
                     "body_length": 1.5, "total_height": 4.0, "dust_outlet": 0.375, "inlet": "voluta",
                     "label": "Stairmand (1951) alto caudal (entrada en voluta)"},
    "swift_he": {"inlet_height": 0.44, "inlet_width": 0.21, "outlet_diameter": 0.4, "outlet_length": 0.5,
                 "body_length": 1.4, "total_height": 3.9, "dust_outlet": 0.4, "inlet": "tangencial",
                 "label": "Swift (1969) alta eficiencia"},
    "swift_gp": {"inlet_height": 0.5, "inlet_width": 0.25, "outlet_diameter": 0.5, "outlet_length": 0.6,
                 "body_length": 1.75, "total_height": 3.75, "dust_outlet": 0.4, "inlet": "tangencial",
                 "label": "Swift (1969) propósito general"},
    "swift_ht": {"inlet_height": 0.8, "inlet_width": 0.35, "outlet_diameter": 0.75, "outlet_length": 0.85,
                 "body_length": 1.7, "total_height": 3.7, "dust_outlet": 0.4, "inlet": "voluta",
                 "label": "Swift (1969) alto caudal (entrada en voluta)"},
}
RATIO_KEYS = ("inlet_height", "inlet_width", "outlet_diameter", "outlet_length", "body_length", "total_height",
              "dust_outlet")
SHEPHERD_LAPPLE_K_TANGENTIAL = 16.0
PSD_SUM_TOL = 1e-3
CHECK_TOL = 1e-12


def _pos(name: str, value, what: str) -> float:
    if value is None:
        raise ValueError(f"falta {name}: {what}")
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} debe ser un número > 0 ({what})")
    return v


def lapple_cut_diameter(mu: float, b: float, Ne: float, Vi: float, rho_p: float, rho_g: float) -> float:
    return math.sqrt(9.0 * mu * b / (2.0 * math.pi * Ne * Vi * (rho_p - rho_g)))


def grade_efficiency(d: float, d50: float) -> float:
    return 1.0 / (1.0 + (d50 / d) ** 2)


def overall_efficiency(d50: float, diameters_m: list[float], mass_fractions: list[float]) -> dict:
    """Mass-weighted Lapple efficiency over a discrete PSD (representative diameter of each size class)."""
    if not diameters_m or len(diameters_m) != len(mass_fractions):
        raise ValueError("particle_size_distribution necesita listas 'diameter_m' y 'mass_fraction' de igual longitud")
    if any(float(d) <= 0 for d in diameters_m) or any(float(x) < 0 for x in mass_fractions):
        raise ValueError("los diámetros deben ser > 0 y las fracciones másicas ≥ 0")
    total = sum(float(x) for x in mass_fractions)
    if abs(total - 1.0) > PSD_SUM_TOL:
        raise ValueError(f"las fracciones másicas de la distribución suman {total:.6g}, no 1")
    rows = [{"diameter_m": float(d), "mass_fraction": float(x), "grade_efficiency": grade_efficiency(float(d), d50),
             "contribution": float(x) * grade_efficiency(float(d), d50)} for d, x in zip(diameters_m, mass_fractions)]
    eta = sum(r["contribution"] for r in rows) / total
    return {"overall_efficiency": eta, "mass_fraction_sum": total, "table": rows}


def cyclone_performance(gas_flow_m3_s: float, gas_density_kg_m3: float, gas_viscosity_Pa_s: float,
                        particle_density_kg_m3: float, body_diameter_m: float, geometry: str | dict = "lapple",
                        particle_size_distribution: dict | None = None, diameters_m: list[float] | None = None,
                        pressure_drop_K: float = SHEPHERD_LAPPLE_K_TANGENTIAL) -> dict:
    """Lapple cut size, grade and overall efficiency, Shepherd-Lapple pressure drop (module docstring)."""
    Q = _pos("gas_flow_m3_s", gas_flow_m3_s, "caudal volumétrico de gas a condiciones de operación")
    rho_g = _pos("gas_density_kg_m3", gas_density_kg_m3, "densidad del gas a condiciones de operación")
    mu = _pos("gas_viscosity_Pa_s", gas_viscosity_Pa_s, "viscosidad del gas")
    rho_p = _pos("particle_density_kg_m3", particle_density_kg_m3, "densidad real de las partículas (no aparente)")
    D = _pos("body_diameter_m", body_diameter_m, "diámetro del cuerpo cilíndrico del ciclón")
    K = _pos("pressure_drop_K", pressure_drop_K, "constante de Shepherd-Lapple")
    if rho_p <= rho_g:
        raise ValueError("la partícula debe ser más densa que el gas")
    warnings: list[str] = []
    assumptions = ["método de Lapple: partículas esféricas en régimen de Stokes, sin interacción entre partículas, "
                   "carga de sólidos baja", "Ne = (h + (H − h)/2)/a de Lapple (deducido para la geometría estándar 2D-2D)"]
    if isinstance(geometry, str):
        key = geometry.strip().lower()
        if key not in PRESETS:
            raise ValueError(f"geometry desconocida {geometry!r}; presets: {sorted(PRESETS)} o un dict con las relaciones "
                             f"{RATIO_KEYS} respecto a D")
        ratios = {k: PRESETS[key][k] for k in RATIO_KEYS}
        label, inlet = PRESETS[key]["label"], PRESETS[key]["inlet"]
    elif isinstance(geometry, dict):
        missing = [k for k in RATIO_KEYS if geometry.get(k) is None]
        if missing:
            raise ValueError(f"geometry personalizada: faltan las relaciones {missing} (cada una dividida entre D)")
        ratios = {k: _pos(f"geometry.{k}", geometry[k], "relación geométrica respecto a D") for k in RATIO_KEYS}
        label, inlet = "personalizada", str(geometry.get("inlet", "tangencial"))
    else:
        raise ValueError("geometry debe ser el nombre de un preset o un dict de relaciones")
    if ratios["total_height"] <= ratios["body_length"]:
        raise ValueError("total_height debe ser mayor que body_length (el cono mide H − h)")
    if ratios["outlet_length"] < ratios["inlet_height"]:
        warnings.append("el buscador de vórtice es más corto que la entrada: cortocircuito del gas, eficiencia "
                        "sobreestimada")
    dims = {k + "_m": v * D for k, v in ratios.items()}
    a, b, De = dims["inlet_height_m"], dims["inlet_width_m"], dims["outlet_diameter_m"]
    Lb, Lc = dims["body_length_m"], dims["total_height_m"] - dims["body_length_m"]
    dims["cone_length_m"] = Lc
    Vi = Q / (a * b)
    Ne = (Lb + Lc / 2.0) / a
    d50 = lapple_cut_diameter(mu, b, Ne, Vi, rho_p, rho_g)
    Hv = K * a * b / De ** 2
    dp = Hv * 0.5 * rho_g * Vi ** 2
    if inlet != "tangencial":
        warnings.append(f"entrada {inlet}: la K = {K:g} de Shepherd-Lapple es para entrada tangencial; ΔP aproximada")
    if K != SHEPHERD_LAPPLE_K_TANGENTIAL:
        assumptions.append(f"K de Shepherd-Lapple dada por el usuario: {K:g}")
    else:
        assumptions.append("Shepherd-Lapple con K = 16 (entrada tangencial sin álabes)")
    if not (isinstance(geometry, str) and geometry.strip().lower() == "lapple"):
        warnings.append("Ne de Lapple fuera de la geometría estándar 2D-2D: el número de vueltas y d50 son aproximados")
    checks = {"eta_at_d50_minus_half": abs(grade_efficiency(d50, d50) - 0.5),
              "d50_closed_form_rel": abs(9.0 * mu * b / (2.0 * math.pi * Ne * Vi * (rho_p - rho_g)) / d50 ** 2 - 1.0)}
    out: dict = {"geometry": {"name": label, "inlet": inlet, "ratios_to_D": ratios, "dimensions": dims},
                 "inlet_velocity_m_s": Vi, "effective_turns": Ne, "cut_diameter_m": d50,
                 "pressure_drop_velocity_heads": Hv, "pressure_drop_Pa": dp, "pressure_drop_K": K}
    if diameters_m:
        out["grade_efficiency"] = [{"diameter_m": float(d), "efficiency": grade_efficiency(_pos("diameters_m", d,
                                    "diámetro de partícula"), d50)} for d in diameters_m]
    if particle_size_distribution is not None:
        psd = particle_size_distribution
        ov = overall_efficiency(d50, psd.get("diameter_m"), psd.get("mass_fraction"))
        checks["overall_sum_rel"] = abs(sum(r["contribution"] for r in ov["table"]) / ov["mass_fraction_sum"]
                                        - ov["overall_efficiency"])
        out["overall"] = ov
        assumptions.append("eficiencia global = Σ x_i·η(d_i) con el diámetro representativo de cada clase (sin integrar "
                           "dentro de la clase)")
    else:
        warnings.append("sin particle_size_distribution no se calcula la eficiencia global (no hay distribución "
                        "por defecto)")
    reasons = [k for k, v in checks.items() if v > 1e-9]
    out.update(method=("Lapple (Ne, d50, η = 1/(1 + (d50/d)²)) y Shepherd-Lapple (H_v = K·a·b/De²); fórmulas cerradas, "
                       "sin iteración"),
               warnings=warnings, assumptions=assumptions,
               not_included=["método de Barth-Muschelknautz (fricción de pared, efecto de la carga de sólidos)",
                             "efecto de la carga de sólidos en ΔP y eficiencia", "reentrada y velocidad de saltación",
                             "longitud natural del vórtice", "ciclones en serie o multiciclones"],
               evidence={"verified": not reasons, "checks": checks, "tolerance": 1e-9,
                         "reasons": [f"identidad no cumplida: {r}" for r in reasons]})
    return out
