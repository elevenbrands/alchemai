"""Psychrometrics and drying of solids (bloque solids): batch dryers and steady continuous dryers as 1-D models.

What it answers: the state of humid air (humidity, wet bulb = adiabatic saturation, dew point, humid heat and volume,
enthalpy); the drying time of a batch (constant-rate period from heat transfer to a surface at the wet-bulb temperature,
falling-rate period linear, diffusion-controlled or from the user's rate curve; cross-circulation and through-circulation);
and the balances of a continuous dryer (air needed or outlet air state, heat duty, transfer units, length).

Property data (nothing is typed here):
  * Humid air: CoolProp `HumidAirProp` (real-gas humid air of ASHRAE RP-1485). Its wet bulb is the THERMODYNAMIC wet bulb
    (adiabatic saturation temperature); for air-water it is taken as the psychrometric wet bulb (Lewis relation ~ 1).
  * Water (latent heat, liquid enthalpy): IAPWS-95 through CoolProp.

Correlations (named in the output, never silent): Geankoplis, Transport Processes and Unit Operations, eqs. 9.6-9 and 9.6-10
(air parallel / perpendicular to the surface, h = 0.0204 G^0.8 and h = 1.17 G^0.37 with G in kg/(h m2)), eq. 9.10-13 branch
Re > 350 (through-circulation, h = 0.151 Gt^0.59 / Dp^0.41) and McCabe-Smith-Harriott eq. 24.29 (rotary dryer,
Ua = 0.5 G^0.67 / D in fps units). The user may always give the coefficient instead.

Moistures are dry basis, kg water / kg dry solid, TOTAL (the equilibrium moisture is given separately; 0 if the numbers are
already free moisture). Critical and equilibrium moisture, the rate curve and every operating datum are the user's: no
defaults. Spray and flash dryers are out of scope."""
from __future__ import annotations

import math

import CoolProp
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
from scipy.integrate import quad
from scipy.optimize import brentq

SOURCE = (f"CoolProp {CoolProp.__version__}: HumidAirProp (aire húmedo real, ASHRAE RP-1485) e IAPWS-95 para el agua")
HUMIDITY_INPUTS = {"humidity_ratio_kg_kg": "W", "relative_humidity": "R", "wet_bulb_T_K": "B", "dew_point_T_K": "D",
                   "water_partial_pressure_Pa": "P_w"}
T_ZERO = 273.15
# Geankoplis eqs. 9.6-9 / 9.6-10: h [W/m2K] = c * G^n with G in kg/(h m2); validity as published
FLAT_SURFACE = {"parallel": {"c": 0.0204, "n": 0.8, "G_h": (2450.0, 29300.0), "T_C": (45.0, 150.0),
                             "name": "Geankoplis ec. 9.6-9, aire paralelo a la superficie: h = 0.0204·G^0.8 (G en kg/h·m²)"},
                "perpendicular": {"c": 1.17, "n": 0.37, "G_h": (3900.0, 19500.0), "T_C": None,
                                  "name": "Geankoplis ec. 9.6-10, aire perpendicular a la superficie: h = 1.17·G^0.37 "
                                          "(G en kg/h·m²)"}}
THROUGH_RE_MIN = 350.0                    # Geankoplis eq. 9.10-13 is used only above this Reynolds number
BTU_FT3_H_F = 5.678263 / 0.3048           # W/(m3 K) per Btu/(ft3 h °F)
LB_FT2_H = 0.45359237 / 0.3048 ** 2 / 3600.0   # kg/(m2 s) per lb/(ft2 h)
PSY_TOL = 1e-5
QUAD_TOL = 1e-6
BALANCE_TOL = 1e-8
IAPWS_CROSS_TOL = 5e-3                    # humid-air enthalpy scale vs IAPWS steam (independent cross-check)


def _need(d: dict | None, key: str, what: str, where: str):
    if d is None or d.get(key) is None:
        raise ValueError(f"falta {where}['{key}']: {what}")
    return d[key]


def _positive(x, name: str, allow_zero: bool = False) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un número") from None
    if not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        raise ValueError(f"{name} debe ser {'no negativo' if allow_zero else 'positivo'} (recibido {x})")
    return x


def _ha(out: str, T: float, P: float, key: str, val: float) -> float:
    try:
        v = HAPropsSI(out, "T", T, "P", P, key, val)
    except ValueError as exc:
        raise ValueError(f"estado de aire húmedo fuera del rango de CoolProp HumidAirProp (T={T:.2f} K, P={P:.0f} Pa, "
                         f"{key}={val:g}): {str(exc)[:160]}") from None
    if not math.isfinite(v):
        raise ValueError(f"CoolProp HumidAirProp no devolvió un valor finito de {out} en T={T:.2f} K, {key}={val:g}")
    return float(v)


def _hf(T: float) -> float:
    """Enthalpy of saturated liquid water, J/kg (IAPWS-95, zero at the triple point)."""
    return float(PropsSI("H", "T", T, "Q", 0, "Water"))


def _latent(T: float) -> float:
    return float(PropsSI("H", "T", T, "Q", 1, "Water")) - _hf(T)


def humid_air_state(T_K: float, pressure_Pa: float, humidity_ratio_kg_kg: float | None = None,
                    relative_humidity: float | None = None, wet_bulb_T_K: float | None = None,
                    dew_point_T_K: float | None = None, water_partial_pressure_Pa: float | None = None) -> dict:
    """Psychrometric state from dry-bulb temperature, total pressure and exactly one humidity datum. Plain dict of floats;
    `adiabatic_saturation_residual` is the energy balance of the adiabatic saturator evaluated with IAPWS liquid water."""
    T = _positive(T_K, "T_K")
    P = _positive(pressure_Pa, "pressure_Pa")
    given = {k: v for k, v in (("humidity_ratio_kg_kg", humidity_ratio_kg_kg), ("relative_humidity", relative_humidity),
                               ("wet_bulb_T_K", wet_bulb_T_K), ("dew_point_T_K", dew_point_T_K),
                               ("water_partial_pressure_Pa", water_partial_pressure_Pa)) if v is not None}
    if len(given) != 1:
        raise ValueError("el aire necesita exactamente un dato de humedad: humidity_ratio_kg_kg (kg agua/kg aire seco), "
                         "relative_humidity (0-1), wet_bulb_T_K, dew_point_T_K o water_partial_pressure_Pa; "
                         f"se recibieron {len(given)}")
    name, value = next(iter(given.items()))
    value = _positive(value, name, allow_zero=name in ("humidity_ratio_kg_kg", "relative_humidity",
                                                       "water_partial_pressure_Pa"))
    if name == "relative_humidity" and value > 1.0:
        raise ValueError("relative_humidity va de 0 a 1 (0.5 = 50 %)")
    if name in ("wet_bulb_T_K", "dew_point_T_K") and value > T + 1e-9:
        raise ValueError(f"{name} no puede superar la temperatura de bulbo seco")
    W = value if name == "humidity_ratio_kg_kg" else _ha("W", T, P, HUMIDITY_INPUTS[name], value)
    if W < 0:
        raise ValueError("la humedad resultante es negativa: revisa el dato de humedad")
    T_boil = float(PropsSI("T", "P", P, "Q", 0, "Water"))
    W_sat = _ha("W", T, P, "R", 1.0) if T < T_boil - 1.5 else None
    if W_sat is not None and W > W_sat * (1 + 1e-9):
        raise ValueError(f"el aire estaría sobresaturado: humedad {W:.5f} > saturación {W_sat:.5f} kg/kg a {T:.2f} K")
    saturated = W_sat is not None and W >= W_sat * (1 - 1e-7)     # CoolProp rejects RH = 1 + rounding: handle the limit here
    Twb = T if saturated else _ha("B", T, P, "W", W)
    W_wb = W_sat if saturated else _ha("W", Twb, P, "R", 1.0)
    h = _ha("H", T, P, "W", W)
    h_wb = _ha("H", Twb, P, "R", 1.0)
    lam = _latent(Twb)
    v = _ha("V", T, P, "W", W)
    h_liquid = float(PropsSI("H", "T", Twb, "P", P, "Water"))          # make-up water at the wet bulb and total pressure
    residual = (h + (W_wb - W) * h_liquid - h_wb) / (abs(h_wb) + lam * W_wb)
    return {"T_K": T, "pressure_Pa": P, "humidity_ratio_kg_kg": W,
            "relative_humidity": 1.0 if saturated else (_ha("R", T, P, "W", W) if W > 0 else 0.0),
            "saturation_humidity_ratio_kg_kg": W_sat,
            "percentage_saturation": (W / W_sat if W_sat else None),
            "water_partial_pressure_Pa": _ha("P_w", T, P, "W", W) if W > 0 else 0.0,
            "dew_point_K": T if saturated else (_ha("D", T, P, "W", W) if W > 0 else None),
            "wet_bulb_T_K": Twb, "saturation_humidity_at_wet_bulb_kg_kg": W_wb,
            "latent_heat_at_wet_bulb_J_kg": lam,
            "humid_heat_J_kgK": _ha("C", T, P, "W", W), "humid_volume_m3_kg": v, "density_kg_m3": (1.0 + W) / v,
            "enthalpy_J_kg": h, "viscosity_Pa_s": _ha("mu", T, P, "W", W),
            "adiabatic_saturation_residual": residual}


def _air(air: dict | None, where: str = "air") -> dict:
    if not isinstance(air, dict):
        raise ValueError(f"falta {where}: {{'T_K', 'pressure_Pa' y un dato de humedad (humidity_ratio_kg_kg, "
                         "relative_humidity, wet_bulb_T_K, dew_point_T_K o water_partial_pressure_Pa)}")
    T = _need(air, "T_K", "temperatura de bulbo seco del aire", where)
    P = _need(air, "pressure_Pa", "presión absoluta total (define la humedad de saturación)", where)
    return humid_air_state(T, P, **{k: air.get(k) for k in HUMIDITY_INPUTS})


BASE_ASSUMPTIONS = ["bulbo húmedo = temperatura de saturación adiabática (bulbo húmedo termodinámico de CoolProp); válido "
                    "para aire-agua, no para otros disolventes",
                    "humedades por kg de aire seco; humid_heat y enthalpy también por kg de aire seco"]


def _seal(out: dict, evidence: dict, reasons: list[str]) -> dict:
    evidence["verified"] = not reasons
    evidence["reasons"] = reasons
    out["evidence"] = evidence
    return out


# ------------------------------------------------------------------------------------------------ constant-rate period
def _flat_surface_rate(state: dict, cr: dict, warnings: list, assumptions: list) -> dict:
    conf = cr["configuration"]
    rho = state["density_kg_m3"]
    if cr.get("mass_velocity_kg_m2s") is not None and cr.get("velocity_m_s") is not None:
        raise ValueError("da velocity_m_s o mass_velocity_kg_m2s del aire, no ambos")
    G = cr.get("mass_velocity_kg_m2s")
    if G is None and cr.get("velocity_m_s") is not None:
        G = _positive(cr["velocity_m_s"], "constant_rate['velocity_m_s']") * rho
    h = cr.get("h_W_m2K")
    info: dict = {"configuration": conf}
    if G is not None:
        info["mass_velocity_kg_m2s"] = G = _positive(G, "constant_rate['mass_velocity_kg_m2s']")
    if h is not None:
        h = _positive(h, "constant_rate['h_W_m2K']")
        info["h_source"] = "dato del usuario"
    else:
        if G is None:
            raise ValueError("falta constant_rate['velocity_m_s'] (o 'mass_velocity_kg_m2s', o directamente 'h_W_m2K'): la "
                             "velocidad del aire sobre el sólido fija el coeficiente de transferencia de calor")
        c = FLAT_SURFACE[conf]
        G_h = G * 3600.0
        h = c["c"] * G_h ** c["n"]
        info["h_source"] = c["name"]
        assumptions.append(f"coeficiente de transferencia de calor ESTIMADO con {c['name']}; no es un dato medido")
        warnings.append("h estimado con una correlación de libro: su incertidumbre es grande (en el ejemplo 24.1 de McCabe "
                        "la correlación de Geankoplis da un h 46 % mayor que el que usa McCabe); mide la velocidad de "
                        "secado o da h_W_m2K si lo tienes")
        lo, hi = c["G_h"]
        if not lo <= G_h <= hi:
            warnings.append(f"G = {G_h:.0f} kg/h·m² está fuera del intervalo de la correlación ({lo:.0f}-{hi:.0f})")
        if c["T_C"] and not c["T_C"][0] <= state["T_K"] - T_ZERO <= c["T_C"][1]:
            warnings.append(f"la temperatura del aire está fuera del intervalo de la correlación "
                            f"({c['T_C'][0]:.0f}-{c['T_C'][1]:.0f} °C)")
    dT = state["T_K"] - state["wet_bulb_T_K"]
    info.update(h_W_m2K=h, driving_force_K=dT, surface_T_K=state["wet_bulb_T_K"],
                latent_heat_J_kg=state["latent_heat_at_wet_bulb_J_kg"],
                rate_kg_m2s=h * dT / state["latent_heat_at_wet_bulb_J_kg"], rate_basis="por m² de superficie expuesta")
    assumptions.append("periodo de velocidad constante: solo convección desde el aire; la superficie está a la temperatura "
                       "de bulbo húmedo: Rc = h·(T − Tw)/λw")
    return info


def _particle_geometry(part: dict) -> tuple[float, float, dict]:
    """Specific surface a [m2 per m3 of bed] and equivalent diameter Dp [m] of the packing (Geankoplis 9.10)."""
    a = part.get("specific_area_m2_m3")
    Dp = part.get("equivalent_diameter_m")
    info = {}
    shape = part.get("shape")
    if a is None or Dp is None:
        if shape not in ("sphere", "cylinder"):
            raise ValueError("falta constant_rate['particle']: da 'specific_area_m2_m3' (m² de partícula por m³ de lecho) y "
                             "'equivalent_diameter_m', o 'shape' ('sphere' con 'diameter_m', o 'cylinder' con 'diameter_m' "
                             "y 'length_m') junto con 'void_fraction'")
        eps = _need(part, "void_fraction", "fracción de huecos del lecho (para el área específica)",
                    "constant_rate['particle']")
        if not 0 < eps < 1:
            raise ValueError("void_fraction debe estar entre 0 y 1")
        D = _positive(_need(part, "diameter_m", "diámetro de la partícula", "constant_rate['particle']"), "diameter_m")
        if shape == "sphere":
            a_c, Dp_c = 6.0 * (1 - eps) / D, D
        else:
            L = _positive(_need(part, "length_m", "longitud del cilindro", "constant_rate['particle']"), "length_m")
            a_c, Dp_c = 4.0 * (1 - eps) * (L + 0.5 * D) / (D * L), math.sqrt(D * L + 0.5 * D * D)
        info["geometry"] = (f"{shape}: a y Dp (diámetro de la esfera de igual área) según Geankoplis ecs. 9.10-15/16")
        a = a_c if a is None else a
        Dp = Dp_c if Dp is None else Dp
    return _positive(a, "specific_area_m2_m3"), _positive(Dp, "equivalent_diameter_m"), info


def _through_circulation_rate(state: dict, cr: dict, solid: dict | None, warnings: list, assumptions: list) -> dict:
    T1, Tw, lam, P, H1 = (state["T_K"], state["wet_bulb_T_K"], state["latent_heat_at_wet_bulb_J_kg"], state["pressure_Pa"],
                          state["humidity_ratio_kg_kg"])
    if cr.get("dry_air_mass_velocity_kg_m2s") is not None and cr.get("superficial_velocity_m_s") is not None:
        raise ValueError("da superficial_velocity_m_s o dry_air_mass_velocity_kg_m2s, no ambos")
    G = cr.get("dry_air_mass_velocity_kg_m2s")
    if G is None:
        v = _need(cr, "superficial_velocity_m_s", "velocidad superficial del aire que atraviesa el lecho (o "
                  "'dry_air_mass_velocity_kg_m2s')", "constant_rate")
        G = _positive(v, "superficial_velocity_m_s") / state["humid_volume_m3_kg"]
    G = _positive(G, "dry_air_mass_velocity_kg_m2s")
    x1 = _positive(_need(solid, "bed_depth_m", "espesor del lecho que atraviesa el aire", "solid"), "bed_depth_m")
    a, Dp, info = _particle_geometry(_need(cr, "particle", "geometría del relleno", "constant_rate"))
    h_user = cr.get("h_W_m2K")
    # humid heat and total mass velocity at the mean humidity of the air across the bed (constant-rate period)
    cs, H_mean, h, T2 = state["humid_heat_J_kgK"], H1, None, None
    for _ in range(50):
        Gt = G * (1.0 + H_mean)
        mu = _ha("mu", T1, P, "W", H_mean)
        Re = Dp * Gt / mu
        if h_user is not None:
            h = _positive(h_user, "constant_rate['h_W_m2K']")
        elif Re > THROUGH_RE_MIN:
            h = 0.151 * (Gt * 3600.0) ** 0.59 / Dp ** 0.41
        else:
            raise ValueError(f"Reynolds del lecho Dp·Gt/μ = {Re:.0f} ≤ {THROUGH_RE_MIN:.0f}: la correlación incluida "
                             "(Geankoplis ec. 9.10-13) solo vale por encima; da constant_rate['h_W_m2K']")
        T2 = Tw + (T1 - Tw) * math.exp(-h * a * x1 / (G * cs))
        H2 = H1 + cs * (T1 - T2) / lam
        new = 0.5 * (H1 + H2)
        cs_new = _ha("C", 0.5 * (T1 + T2), P, "W", new)
        done = abs(new - H_mean) < 1e-12 and abs(cs_new - cs) < 1e-9
        H_mean, cs = new, cs_new
        if done:
            break
    if h_user is None:
        assumptions.append("coeficiente gas-partícula ESTIMADO con Geankoplis ec. 9.10-13 (Re > 350): h = "
                           "0.151·Gt^0.59/Dp^0.41, Gt en kg/h·m² total; no es un dato medido")
    assumptions.append("secado con circulación a través del lecho: el aire se enfría y humidifica adiabáticamente a lo "
                       "largo del espesor; velocidad media sobre el lecho con fuerza impulsora media logarítmica "
                       "(Geankoplis ecs. 9.10-11 y 9.10-12); calor húmedo y Gt a la humedad media entrada-salida del periodo "
                       "constante")
    rate = G * cs * (T1 - Tw) * (1.0 - math.exp(-h * a * x1 / (G * cs))) / lam
    info.update(configuration="through_circulation", dry_air_mass_velocity_kg_m2s=G, total_mass_velocity_kg_m2s=Gt,
                reynolds=Re, h_W_m2K=h, h_source=("dato del usuario" if h_user is not None else "Geankoplis ec. 9.10-13"),
                specific_area_m2_m3=a, equivalent_diameter_m=Dp, humid_heat_mean_J_kgK=cs,
                bed_transfer_units=h * a * x1 / (G * cs), outlet_air_T_constant_period_K=T2,
                outlet_air_humidity_constant_period_kg_kg=H2, surface_T_K=Tw, latent_heat_J_kg=lam, rate_kg_m2s=rate,
                rate_basis="por m² de sección transversal del lecho")
    return info


def _constant_rate(state: dict, cr: dict, solid: dict | None, warnings: list, assumptions: list) -> dict:
    if cr.get("rate_kg_m2s") is not None:
        if cr.get("configuration") is not None:
            raise ValueError("da constant_rate['rate_kg_m2s'] (medida) o 'configuration' (calculada), no ambos")
        return {"rate_kg_m2s": _positive(cr["rate_kg_m2s"], "constant_rate['rate_kg_m2s']"), "h_source": None,
                "configuration": "measured", "rate_basis": "dato del usuario (por m² de superficie de secado)"}
    conf = cr.get("configuration")
    if conf in FLAT_SURFACE:
        return _flat_surface_rate(state, cr, warnings, assumptions)
    if conf == "through_circulation":
        return _through_circulation_rate(state, cr, solid, warnings, assumptions)
    raise ValueError("falta constant_rate['rate_kg_m2s'] (velocidad de secado medida, kg/m²·s) o constant_rate['configuration'] "
                     "= 'parallel' | 'perpendicular' | 'through_circulation' para calcularla")


# --------------------------------------------------------------------------------------------------- diffusion in a slab
def _slab_unaccomplished(tau: float) -> float:
    """Average unaccomplished moisture change E of a slab with uniform initial moisture and surface at equilibrium,
    tau = D t / x1^2. Short-time (error-function) form below 0.05, eigenfunction series above."""
    if tau <= 0:
        return 1.0
    if tau < 0.05:
        return 1.0 - 2.0 * math.sqrt(tau / math.pi)
    return 8.0 / math.pi ** 2 * math.fsum(math.exp(-(2 * n + 1) ** 2 * math.pi ** 2 * tau / 4.0) / (2 * n + 1) ** 2
                                          for n in range(60))


def _slab_series_only(tau: float, terms: int) -> float:
    return 8.0 / math.pi ** 2 * math.fsum(math.exp(-(2 * n + 1) ** 2 * math.pi ** 2 * tau / 4.0) / (2 * n + 1) ** 2
                                          for n in range(terms))


def _diffusion_time(E: float, D: float, x1: float) -> dict:
    tau = brentq(lambda s: _slab_unaccomplished(s) - E, 0.0, 50.0, xtol=1e-15, rtol=1e-13)
    switch = abs(_slab_series_only(0.05, 400) - (1.0 - 2.0 * math.sqrt(0.05 / math.pi)))
    one_term = 4.0 / math.pi ** 2 * math.log(8.0 / (math.pi ** 2 * E)) if E < 8.0 / math.pi ** 2 else None
    return {"time_s": tau * x1 * x1 / D, "fourier_number": tau,
            "one_term_time_s": (one_term * x1 * x1 / D if one_term is not None else None),
            "series_forms_abs_diff_at_switch": switch,
            "series_truncation_abs_diff": abs(_slab_series_only(max(tau, 0.05), 60) - _slab_series_only(max(tau, 0.05), 120))}


# ------------------------------------------------------------------------------------------------------------ batch dryer
def _curve(fr: dict) -> tuple[list[float], list[float]]:
    X = _need(fr, "moisture", "humedades (base seca) de la curva de velocidad de secado", "falling_rate")
    R = _need(fr, "rate_kg_m2s", "velocidades de secado medidas en esas humedades", "falling_rate")
    if len(X) != len(R) or len(X) < 2:
        raise ValueError("falling_rate['moisture'] y ['rate_kg_m2s'] deben tener la misma longitud (≥ 2 puntos)")
    pts = sorted((float(x), float(r)) for x, r in zip(X, R))
    if any(b[0] - a[0] <= 0 for a, b in zip(pts, pts[1:])):
        raise ValueError("la curva de velocidad de secado tiene humedades repetidas")
    if any(r <= 0 for _, r in pts):
        raise ValueError("la curva de velocidad de secado debe tener velocidades positivas en todos los puntos (el tiempo "
                         "diverge donde la velocidad es cero)")
    return [p[0] for p in pts], [p[1] for p in pts]


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    for i in range(len(xs) - 1):
        if x <= xs[i + 1] or i == len(xs) - 2:
            return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[-1]


def _segment_time(xa: float, ra: float, xb: float, rb: float) -> float:
    """Exact ∫ dX / R between two points of a rate that is linear in X."""
    if abs(rb - ra) <= 1e-12 * max(ra, rb):
        return (xb - xa) / (0.5 * (ra + rb))
    return (xb - xa) * math.log(rb / ra) / (rb - ra)


def dryer_batch(air: dict, solid: dict | None = None, constant_rate: dict | None = None,
                falling_rate: dict | None = None) -> dict:
    """Batch drying time. `air` alone returns the psychrometric state; with `constant_rate` the constant drying rate; with
    `solid` the times of the constant-rate and falling-rate periods. See the tools file for the argument dictionaries."""
    warnings: list[str] = []
    assumptions = list(BASE_ASSUMPTIONS)
    not_included = ["calor por radiación y por conducción a través de la bandeja (suben la temperatura de la superficie por "
                    "encima del bulbo húmedo y la velocidad de secado)", "encogimiento del sólido y cambio de sus propiedades",
                    "calentamiento inicial del sólido hasta el bulbo húmedo", "secadores por aspersión (spray) y flash",
                    "calor de sorción del agua ligada"]
    state = _air(air)
    out: dict = {"air": state, "warnings": warnings, "assumptions": assumptions, "not_included": not_included,
                 "method": f"Estado psicrométrico con {SOURCE}"}
    evidence: dict = {"adiabatic_saturation_residual": state["adiabatic_saturation_residual"]}
    reasons: list[str] = []
    if abs(state["adiabatic_saturation_residual"]) > PSY_TOL:
        reasons.append("el balance de saturación adiabática del estado del aire no cierra")
    if solid is None and constant_rate is None:
        if falling_rate is not None:
            raise ValueError("falta solid: humedades inicial, final, crítica y de equilibrio y la carga de sólido seco")
        return _seal(out, evidence, reasons)

    cr = _constant_rate(state, constant_rate, solid, warnings, assumptions) if constant_rate is not None else None
    if cr is not None:
        out["constant_rate"] = cr
    if solid is None:
        out["method"] += "; velocidad de secado constante por transferencia de calor a la superficie a bulbo húmedo"
        return _seal(out, evidence, reasons)

    X1 = _positive(_need(solid, "initial_moisture", "humedad inicial, kg agua/kg sólido seco", "solid"), "initial_moisture")
    X2 = _positive(_need(solid, "final_moisture", "humedad final deseada, kg agua/kg sólido seco", "solid"), "final_moisture",
                   allow_zero=True)
    if X2 >= X1:
        raise ValueError("final_moisture debe ser menor que initial_moisture")
    model = (falling_rate or {}).get("model")
    if model == "capillary":
        model = "linear"
        assumptions.append("flujo capilar: velocidad proporcional a la humedad libre (misma ley que el modelo lineal; el "
                           "tiempo crece con el espesor, no con su cuadrado)")
    through = cr is not None and cr["configuration"] == "through_circulation"
    # dry-solid loading per unit of the area the rate refers to
    loading = solid.get("dry_solid_per_area_kg_m2")
    if through:
        if loading is not None or solid.get("dry_solid_mass_kg") is not None:
            raise ValueError("con circulación a través del lecho la carga sale de solid['bed_depth_m'] × "
                             "solid['bulk_density_kg_m3']: no des además dry_solid_per_area_kg_m2 ni dry_solid_mass_kg")
        rho_b = _positive(_need(solid, "bulk_density_kg_m3", "kg de sólido seco por m³ de lecho", "solid"),
                          "bulk_density_kg_m3")
        loading = rho_b * float(solid["bed_depth_m"])
    elif loading is None and solid.get("dry_solid_mass_kg") is not None:
        area = _positive(_need(solid, "drying_area_m2", "superficie expuesta al aire", "solid"), "drying_area_m2")
        loading = _positive(solid["dry_solid_mass_kg"], "dry_solid_mass_kg") / area
    if loading is not None:
        loading = _positive(loading, "dry_solid_per_area_kg_m2")

    curve = _curve(falling_rate) if model == "curve" else None
    Xc = solid.get("critical_moisture")
    if curve is not None:
        if curve[0][0] > X2 + 1e-12:
            raise ValueError(f"la curva de velocidad de secado no llega a la humedad final ({curve[0][0]:g} > {X2:g}): no "
                             "se extrapola")
        if Xc is not None and abs(float(Xc) - curve[0][-1]) > 1e-9 and X1 > curve[0][-1]:
            warnings.append("critical_moisture no coincide con el punto más alto de la curva: se usa la curva hasta su "
                            "punto más alto y la velocidad constante por encima")
        Xc = curve[0][-1]
    Xc = _positive(_need({"critical_moisture": Xc}, "critical_moisture", "humedad crítica (fin del periodo de velocidad "
                         "constante); es un dato experimental del material", "solid"), "critical_moisture")
    need_falling = X2 < Xc
    need_constant = X1 > Xc
    Xe = solid.get("equilibrium_moisture")
    if need_falling and curve is None:
        Xe = _positive(_need(solid, "equilibrium_moisture", "humedad de equilibrio con este aire (0 si las humedades ya son "
                             "humedad libre)", "solid"), "equilibrium_moisture", allow_zero=True)
        if model not in ("linear", "diffusion"):
            raise ValueError("falta falling_rate['model'] = 'linear' (velocidad proporcional a la humedad libre; 'capillary' "
                             "es la misma ley), 'diffusion' (difusión de líquido en una placa) o 'curve' (tu curva de "
                             "velocidad): la humedad final está por debajo de la crítica")
        if X2 <= Xe:
            raise ValueError("final_moisture debe ser mayor que equilibrium_moisture: el equilibrio solo se alcanza a "
                             "tiempo infinito")
        if Xc <= Xe:
            raise ValueError("critical_moisture debe ser mayor que equilibrium_moisture")
    rate_based = need_constant or (need_falling and model in ("linear", "curve"))
    Rc = cr["rate_kg_m2s"] if cr is not None else None
    if (need_constant or (need_falling and model == "linear")) and Rc is None:
        raise ValueError("falta constant_rate: la velocidad de secado del periodo constante (medida, 'rate_kg_m2s', o "
                         "calculada con 'configuration') fija el tiempo de secado")
    if rate_based and loading is None:
        raise ValueError("falta solid['dry_solid_per_area_kg_m2'] (o 'dry_solid_mass_kg' y 'drying_area_m2'): kg de sólido "
                         "seco por m² de superficie de secado")
    if curve is not None and need_constant and abs(curve[1][-1] - Rc) > 0.05 * Rc:
        warnings.append(f"la curva termina en {curve[1][-1]:.3e} kg/m²·s y la velocidad constante es {Rc:.3e}: no empalman "
                        "(diferencia > 5 %)")

    def rate(X: float) -> float:
        if X > Xc or (curve is None and X >= Xc):
            return Rc
        if curve is not None:
            return _interp(curve[0], curve[1], X)
        return Rc * (X - Xe) / (Xc - Xe)

    def time_between(Xa: float, Xb: float) -> float:
        """Closed-form time to go from Xa down to Xb (rate-based models), s."""
        t = 0.0
        if Xa > Xc:
            t += loading * (Xa - max(Xb, Xc)) / Rc
        lo, hi = Xb, min(Xa, Xc)
        if hi > lo:
            if curve is not None:
                nodes = [lo] + [x for x in curve[0] if lo < x < hi] + [hi]
                t += loading * sum(_segment_time(a, _interp(curve[0], curve[1], a), b, _interp(curve[0], curve[1], b))
                                   for a, b in zip(nodes, nodes[1:]))
            else:
                t += loading * (Xc - Xe) / Rc * math.log((hi - Xe) / (lo - Xe))
        return t

    periods: dict = {}
    t_const = loading * (X1 - max(X2, Xc)) / Rc if need_constant else 0.0
    periods["constant_rate_time_s"] = t_const
    t_fall = 0.0
    method = ["periodo constante: t = (Ls/A)·(X1 − Xc)/Rc"] if need_constant else []
    X_top = min(X1, Xc)
    if need_falling and model == "diffusion":
        fr = falling_rate
        D = _positive(_need(fr, "diffusivity_m2_s", "difusividad efectiva del agua líquida en el sólido (dato experimental)",
                            "falling_rate"), "diffusivity_m2_s")
        thick = _positive(_need(fr, "thickness_m", "espesor de la placa de sólido", "falling_rate"), "thickness_m")
        faces = _need(fr, "drying_faces", "caras por las que seca la placa (1 o 2)", "falling_rate")
        if faces not in (1, 2):
            raise ValueError("falling_rate['drying_faces'] debe ser 1 o 2")
        x1 = thick / faces
        E = (X2 - Xe) / (X_top - Xe)
        dif = _diffusion_time(E, D, x1)
        t_fall = dif["time_s"]
        periods["diffusion"] = {"unaccomplished_change": E, "half_thickness_m": x1, **dif}
        evidence["diffusion_series_forms_abs_diff"] = dif["series_forms_abs_diff_at_switch"]
        evidence["diffusion_series_truncation_abs_diff"] = dif["series_truncation_abs_diff"]
        if dif["series_forms_abs_diff_at_switch"] > 1e-8 or dif["series_truncation_abs_diff"] > 1e-12:
            reasons.append("la serie de difusión no convergió")
        assumptions.append("periodo decreciente por difusión de líquido en una placa (2.ª ley de Fick, difusividad "
                           "constante, superficie en la humedad de equilibrio, perfil inicial uniforme, sin encogimiento); "
                           "serie completa, no solo el primer término")
        if need_constant:
            warnings.append("la difusión arranca de un perfil uniforme en la humedad crítica; tras un periodo constante el "
                            "perfil real no es uniforme, así que el tiempo decreciente es aproximado")
        if E > 0.6:
            warnings.append("X/X1 > 0.6: la fórmula de un término de los libros no es válida aquí; se usa la serie completa")
        method.append("periodo decreciente: difusión en placa, serie de Fourier completa resuelta en el tiempo (brentq de "
                      "SciPy)")
    elif need_falling:
        t_fall = time_between(X_top, X2)
        if curve is not None:
            assumptions.append("periodo decreciente con TU curva de velocidad, interpolada linealmente entre puntos; el "
                               "tiempo es la integral exacta de dX/R de esa poligonal")
            nodes = [X2] + [x for x in curve[0] if X2 < x < X_top] + [X_top]
            recip = sum(0.5 * (1 / _interp(*curve, a) + 1 / _interp(*curve, b)) * (b - a) for a, b in zip(nodes, nodes[1:]))
            periods["falling_time_trapezoid_on_reciprocal_s"] = loading * recip
            method.append("periodo decreciente: t = (Ls/A)·∫dX/R sobre la curva del usuario (integral exacta por tramos)")
        else:
            assumptions.append("periodo decreciente lineal: R = Rc·(X − X*)/(Xc − X*) (recta desde el punto crítico hasta "
                               "velocidad cero en el equilibrio)")
            method.append("periodo decreciente: t = (Ls/A)·(Xc − X*)/Rc·ln[(Xc − X*)/(X2 − X*)]")
    periods["falling_rate_time_s"] = t_fall
    periods["total_time_s"] = t_const + t_fall
    periods["total_time_h"] = (t_const + t_fall) / 3600.0
    out["drying"] = periods
    out["solid"] = {"initial_moisture": X1, "final_moisture": X2, "critical_moisture": Xc, "equilibrium_moisture": Xe,
                    "dry_solid_per_area_kg_m2": loading,
                    "water_removed_kg_m2": (loading * (X1 - X2) if loading is not None else None)}
    if solid.get("dry_solid_mass_kg") is not None:
        out["solid"]["water_removed_kg"] = float(solid["dry_solid_mass_kg"]) * (X1 - X2)
    if not need_falling:
        assumptions.append("todo el secado ocurre en el periodo de velocidad constante (humedad final ≥ crítica)")

    # independent check of the rate-based closed forms: adaptive quadrature of dX/R
    if rate_based:
        lo = Xc if (need_falling and model == "diffusion") else X2
        breaks = sorted({Xc, *(curve[0] if curve is not None else [])})
        breaks = [b for b in breaks if lo < b < X1]
        numeric, err = quad(lambda X: 1.0 / rate(X), lo, X1, points=breaks or None, epsabs=0.0, epsrel=1e-11, limit=400)
        closed = time_between(X1, lo)
        evidence["time_quadrature_rel_diff"] = abs(loading * numeric - closed) / closed
        evidence["quadrature_error_estimate_rel"] = abs(err / numeric)
        if evidence["time_quadrature_rel_diff"] > QUAD_TOL:
            reasons.append("el tiempo en forma cerrada no coincide con la integración numérica de dX/R")
        grid = [X1 - (X1 - lo) * i / 10 for i in range(11)]
        out["drying_curve"] = [{"moisture": x, "time_s": time_between(X1, x), "rate_kg_m2s": rate(x)} for x in grid]
    out["method"] = "; ".join([out["method"]] + method + ["comprobado con cuadratura adaptativa de SciPy (QUADPACK)"])
    return _seal(out, evidence, reasons)


# ------------------------------------------------------------------------------------------------------- continuous dryer
def dryer_continuous(air_in: dict, solid: dict, flow: str, heat_loss_W: float, air_outlet_T_K: float | None = None,
                     dry_air_flow_kg_s: float | None = None, transfer_units: float | None = None,
                     sizing: dict | None = None, air_heater_inlet_T_K: float | None = None) -> dict:
    """Steady continuous dryer (rotary, tunnel, belt) as a 1-D air-solid contactor: water and enthalpy balances, air needed or
    outlet air state, heat transferred, transfer units and, with a volumetric coefficient, length."""
    warnings: list[str] = []
    assumptions = list(BASE_ASSUMPTIONS)
    not_included = ["secadores por aspersión (spray) y flash", "tiempo de residencia y retención del sólido en el tambor",
                    "perfil de humedad del sólido y zonas de precalentamiento y de humedad ligada (el número de unidades de "
                    "transferencia supone la superficie del sólido a bulbo húmedo en todo el equipo)",
                    "calor de sorción del agua ligada", "arrastre de finos y caída de presión"]
    if flow not in ("countercurrent", "cocurrent"):
        raise ValueError("falta flow = 'countercurrent' o 'cocurrent': fija qué extremos del aire y del sólido se enfrentan")
    if heat_loss_W is None:
        raise ValueError("falta heat_loss_W: pérdida de calor del secador al ambiente (0 si es adiabático)")
    Q = _positive(heat_loss_W, "heat_loss_W", allow_zero=True)
    specs = [k for k, v in (("air_outlet_T_K", air_outlet_T_K), ("dry_air_flow_kg_s", dry_air_flow_kg_s),
                            ("transfer_units", transfer_units)) if v is not None]
    if len(specs) != 1:
        raise ValueError("da exactamente uno: air_outlet_T_K (sale el aire necesario), dry_air_flow_kg_s (sale la "
                         "temperatura de salida) o transfer_units (unidades de transferencia de calor del diseño)")
    a_in = _air(air_in, "air_in")
    P, T_in, H_in, Twb = a_in["pressure_Pa"], a_in["T_K"], a_in["humidity_ratio_kg_kg"], a_in["wet_bulb_T_K"]
    where = "solid"
    Ls = _positive(_need(solid, "dry_solid_flow_kg_s", "caudal de sólido seco", where), "dry_solid_flow_kg_s")
    X1 = _positive(_need(solid, "inlet_moisture", "humedad de entrada, kg agua/kg sólido seco", where), "inlet_moisture")
    X2 = _positive(_need(solid, "outlet_moisture", "humedad de salida, kg agua/kg sólido seco", where), "outlet_moisture",
                   allow_zero=True)
    Ts1 = _positive(_need(solid, "inlet_T_K", "temperatura de entrada del sólido", where), "inlet_T_K")
    Ts2 = _positive(_need(solid, "outlet_T_K", "temperatura de salida del sólido (dato o especificación; no se supone igual "
                          "al bulbo húmedo)", where), "outlet_T_K")
    cps = _positive(_need(solid, "dry_solid_cp_J_kgK", "calor específico del sólido seco", where), "dry_solid_cp_J_kgK")
    if X2 >= X1:
        raise ValueError("outlet_moisture debe ser menor que inlet_moisture")
    mv = Ls * (X1 - X2)
    dHs = Ls * (cps * (Ts2 - Ts1) + X2 * _hf(Ts2) - X1 * _hf(Ts1))      # W, enthalpy gained by the solid stream
    h_in = a_in["enthalpy_J_kg"]
    evidence: dict = {"adiabatic_saturation_residual": a_in["adiabatic_saturation_residual"]}
    reasons: list[str] = []

    def residual(G: float, T_out: float) -> float:
        return G * (h_in - _ha("H", T_out, P, "W", H_in + mv / G)) - dHs - Q

    if transfer_units is not None:
        N = _positive(transfer_units, "transfer_units")
        T_out = Twb + (T_in - Twb) * math.exp(-N)
        assumptions.append("temperatura de salida del aire por unidades de transferencia: Nt = ln[(T_ent − Tw)/(T_sal − "
                           "Tw)] (McCabe-Smith-Harriott ec. 24.8), con Tw el bulbo húmedo del aire de entrada")
    elif air_outlet_T_K is not None:
        T_out = _positive(air_outlet_T_K, "air_outlet_T_K")
    if dry_air_flow_kg_s is None:
        if T_out >= T_in:
            raise ValueError("el aire debe salir más frío de lo que entra (air_outlet_T_K < T de entrada)")
        T_boil = float(PropsSI("T", "P", P, "Q", 0, "Water"))
        if T_out < T_boil - 1.5:
            W_sat = _ha("W", T_out, P, "R", 1.0)
            if W_sat <= H_in:
                raise ValueError("a esa temperatura de salida el aire de entrada ya estaría saturado")
            G_lo = mv / (W_sat - H_in) * (1 + 1e-9)
        else:
            G_lo = mv / 5.0
        if residual(G_lo, T_out) > 0:
            raise ValueError("balance imposible con esa temperatura de salida: el aire saldría sobresaturado (el calor "
                             "disponible evapora más agua de la que el aire admite); sube air_outlet_T_K")
        G_hi = G_lo * 2.0
        for _ in range(200):
            if residual(G_hi, T_out) > 0:
                break
            G_hi *= 2.0
        else:
            raise ValueError("no hay caudal de aire que cierre el balance de entalpía: el aire no trae calor suficiente "
                             "(revisa temperaturas y pérdidas)")
        G, r = brentq(residual, G_lo, G_hi, args=(T_out,), xtol=1e-14, rtol=1e-14, full_output=True)
        converged = r.converged
    else:
        G = _positive(dry_air_flow_kg_s, "dry_air_flow_kg_s")
        H_out_try = H_in + mv / G
        try:
            T_lo = _ha("D", T_in, P, "W", H_out_try) + 1e-6
        except ValueError:
            raise ValueError("con ese caudal de aire la humedad de salida supera lo que el aire puede llevar: sube "
                             "dry_air_flow_kg_s") from None
        if residual(G, T_lo) < 0:
            raise ValueError("balance imposible con ese caudal de aire: el aire se saturaría antes de ceder el calor "
                             "necesario; sube dry_air_flow_kg_s o la temperatura de entrada")
        if residual(G, T_in) > 0:
            raise ValueError("balance imposible: el aire tendría que salir más caliente de lo que entra")
        T_out, r = brentq(lambda T: residual(G, T), T_lo, T_in, xtol=1e-12, rtol=1e-14, full_output=True)
        converged = r.converged
    H_out = H_in + mv / G
    a_out = humid_air_state(T_out, P, humidity_ratio_kg_kg=H_out)
    h_out = a_out["enthalpy_J_kg"]
    q_gas = G * (h_in - _ha("H", T_out, P, "W", H_in)) - Q          # heat given up by the entering gas, net of losses
    h_steam = float(PropsSI("H", "T", T_out, "P", max(a_out["water_partial_pressure_Pa"], 1.0), "Water"))
    q_iapws = dHs + mv * h_steam                                     # same duty from the solid side with IAPWS steam
    enthalpy_in = G * h_in + Ls * (cps * (Ts1 - T_ZERO) + X1 * _hf(Ts1))
    enthalpy_out = G * h_out + Ls * (cps * (Ts2 - T_ZERO) + X2 * _hf(Ts2)) + Q
    evidence.update(
        solver_converged=bool(converged),
        water_balance_rel_error=abs(G * (H_out - H_in) - mv) / mv,
        enthalpy_balance_rel_error=abs(enthalpy_in - enthalpy_out) / max(abs(q_gas), abs(dHs), 1.0),
        heat_duty_vs_iapws_steam_rel_diff=abs(q_gas - q_iapws) / abs(q_iapws),
        outlet_relative_humidity=a_out["relative_humidity"])
    if not converged:
        reasons.append("el solver del balance de entalpía no convergió")
    if evidence["water_balance_rel_error"] > BALANCE_TOL or evidence["enthalpy_balance_rel_error"] > BALANCE_TOL:
        reasons.append("los balances de agua o de entalpía no cierran")
    if evidence["heat_duty_vs_iapws_steam_rel_diff"] > IAPWS_CROSS_TOL:
        reasons.append("el calor transferido no coincide con el calculado con vapor IAPWS (escalas de entalpía incoherentes)")
    if abs(a_in["adiabatic_saturation_residual"]) > PSY_TOL:
        reasons.append("el balance de saturación adiabática del aire de entrada no cierra")
    # second law: the gas must be hotter than the solid it meets at each end
    ends = ((T_in, Ts2, "entrada del aire / salida del sólido"), (T_out, Ts1, "salida del aire / entrada del sólido")) \
        if flow == "countercurrent" else ((T_in, Ts1, "entrada del aire / entrada del sólido"),
                                          (T_out, Ts2, "salida del aire / salida del sólido"))
    crosses = [f"{name}: aire {tg:.1f} K ≤ sólido {ts:.1f} K" for tg, ts, name in ends if tg <= ts]
    evidence["temperature_cross"] = bool(crosses)
    if crosses:
        reasons.append("cruce de temperaturas (el aire no puede calentar a un sólido más caliente): " + "; ".join(crosses))
    if a_out["relative_humidity"] > 0.9:
        warnings.append(f"el aire sale al {100 * a_out['relative_humidity']:.0f} % de humedad relativa: riesgo de "
                        "condensación")
    if T_out <= Twb:
        warnings.append("el aire sale por debajo del bulbo húmedo de entrada: el secador no es adiabático o los datos del "
                        "sólido piden más calor del que da un contacto adiabático; no se calculan unidades de transferencia")
    out: dict = {"air_in": a_in, "air_out": a_out, "dry_air_flow_kg_s": G, "moist_air_in_flow_kg_s": G * (1 + H_in),
                 "water_evaporated_kg_s": mv, "heat_transferred_to_solid_W": q_gas,
                 "solid_enthalpy_gain_W": dHs, "heat_loss_W": Q,
                 "heat_per_kg_water_evaporated_J_kg": q_gas / mv, "flow": flow}
    if air_heater_inlet_T_K is not None:
        T_amb = _positive(air_heater_inlet_T_K, "air_heater_inlet_T_K")
        duty = G * (h_in - _ha("H", T_amb, P, "W", H_in))
        out["air_heater_duty_W"] = duty
        out["thermal_efficiency"] = (mv * a_in["latent_heat_at_wet_bulb_J_kg"] / duty) if duty > 0 else None
        assumptions.append("air_heater_duty_W: calentar el aire a humedad constante desde air_heater_inlet_T_K; "
                           "thermal_efficiency = calor latente del agua evaporada (a bulbo húmedo) / calor del calentador")
    if T_out > Twb:
        N_t = math.log((T_in - Twb) / (T_out - Twb))
        dT_lm = (T_in - T_out) / N_t
        out["transfer_units"] = N_t
        out["log_mean_temperature_difference_K"] = dT_lm
        out["solid_surface_T_K"] = Twb
        assumptions.append("unidades de transferencia y ΔT media logarítmica con la superficie del sólido a la temperatura "
                           "de bulbo húmedo del aire de entrada en todo el secador (modelo de una zona de "
                           "McCabe-Smith-Harriott); aproximado si hay mucho precalentamiento o humedad ligada")
        if sizing is not None:
            out["sizing"] = _continuous_sizing(sizing, G * (1 + H_in), q_gas, dT_lm, warnings, assumptions)
    elif sizing is not None:
        warnings.append("no se dimensiona: la temperatura de salida del aire no supera el bulbo húmedo")
    assumptions.append("entalpía del sólido = cp del sólido seco·(T − 0 °C) + humedad·entalpía del agua líquida saturada "
                       "(IAPWS); el agua evaporada sale como vapor a la temperatura de salida del aire")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method=f"Balances de agua y de entalpía aire-sólido resueltos con brentq (SciPy); propiedades de {SOURCE}")
    return _seal(out, evidence, reasons)


def _continuous_sizing(sz: dict, gas_flow: float, q: float, dT_lm: float, warnings: list, assumptions: list) -> dict:
    given = [k for k in ("diameter_m", "cross_section_area_m2", "gas_mass_velocity_kg_m2s") if sz.get(k) is not None]
    if len(given) != 1:
        raise ValueError("sizing necesita exactamente uno: 'diameter_m', 'cross_section_area_m2' o 'gas_mass_velocity_kg_m2s' "
                         "(velocidad másica admisible del aire húmedo de entrada, límite por arrastre de polvo)")
    if given[0] == "gas_mass_velocity_kg_m2s":
        Gv = _positive(sz["gas_mass_velocity_kg_m2s"], "gas_mass_velocity_kg_m2s")
        S = gas_flow / Gv
    else:
        S = (_positive(sz["cross_section_area_m2"], "cross_section_area_m2") if given[0] == "cross_section_area_m2"
             else math.pi / 4.0 * _positive(sz["diameter_m"], "diameter_m") ** 2)
        Gv = gas_flow / S
    D = math.sqrt(4.0 * S / math.pi)
    Ua = sz.get("volumetric_heat_transfer_coefficient_W_m3K")
    corr = sz.get("correlation")
    if (Ua is None) == (corr is None):
        raise ValueError("sizing necesita 'volumetric_heat_transfer_coefficient_W_m3K' (coeficiente volumétrico Ua, dato "
                         "tuyo) o 'correlation': 'mccabe_rotary' para estimarlo; uno solo")
    if corr is not None:
        if corr != "mccabe_rotary":
            raise ValueError("sizing['correlation'] solo admite 'mccabe_rotary'")
        Ua = 0.5 * (Gv / LB_FT2_H) ** 0.67 / (D / 0.3048) * BTU_FT3_H_F
        source = ("McCabe-Smith-Harriott ec. 24.29, secador rotatorio con elevadores: Ua = 0.5·G^0.67/D (Btu/ft³·h·°F, G en "
                  "lb/ft²·h, D en ft)")
        assumptions.append(f"coeficiente volumétrico ESTIMADO con {source}; es una correlación empírica, no un dato medido")
    else:
        Ua = _positive(Ua, "volumetric_heat_transfer_coefficient_W_m3K")
        source = "dato del usuario"
    L = q / (Ua * S * dT_lm)
    return {"cross_section_area_m2": S, "diameter_m": D, "gas_mass_velocity_kg_m2s": Gv,
            "volumetric_heat_transfer_coefficient_W_m3K": Ua, "coefficient_source": source, "length_m": L,
            "length_to_diameter": L / D, "volume_m3": S * L}
