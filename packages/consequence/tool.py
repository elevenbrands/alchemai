"""Agent / MCP tools of bloque M12 (consequence analysis): source term, Gaussian dispersion, fire radiation and blast
overpressure, each linked to a HAZOP/LOPA row through `para_lopa`.

Weather, release size, inventory, exposure time and end points are the user's: missing ones are asked for, never
defaulted. Example end points are only applied by name and labelled as EXAMPLES.
"""
from __future__ import annotations

import math

import numpy as np

from packages.consequence import dispersion as Dm
from packages.consequence import effects as Ef
from packages.consequence import explosion as Ex
from packages.consequence import fire as Fi
from packages.consequence import source as So
from packages.consequence.props import P_ATM, R, air_density, compound

SCREENING = ("Cribado preliminar con modelos publicados (cada uno con su fuente). No sustituye a una herramienta "
             "validada (Phast, EFFECTS, ALOHA…) para un QRA real.")
# EPA 40 CFR 68.22(a)(2) end points (read on Cornell LII 2026-11-13) for the worst-case / alternative scenarios of the Risk Management Program: EXAMPLE only.
EXAMPLE_ENDPOINTS = {
    "epa_rmp": {"sobrepresion_kPa": 6.895, "radiacion_kW_m2": 5.0, "tiempo_exposicion_s": 40.0, "inflamable": "LFL",
                "fuente": "EPA 40 CFR 68.22(a)(2): 1 psi; 5 kW/m² durante 40 s; LFL (EJEMPLO regulatorio de EE. UU., "
                          "no criterio del owner)"},
}
STD_DISTANCES = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)


class MissingData(ValueError):
    pass


def _need(**kw):
    miss = [k for k, v in kw.items() if v is None]
    if miss:
        raise MissingData("faltan datos del usuario (no se suponen): " + ", ".join(miss))


def _err(e: Exception) -> dict:
    return {"ok": False, "error": str(e), "aviso": SCREENING}


def _lopa_link(hazop_row: int | None, distances: dict[str, float | None], severity_criterion: list[dict] | None,
               modifier: dict | None) -> dict | None:
    """Turn end-point distances into a calculated severity with the USER's criterion, ready for lopa_from_hazop."""
    if hazop_row is None:
        return None
    out: dict = {"fila_hazop": hazop_row, "distancias_m": distances}
    sev = None
    used = []
    for c in severity_criterion or []:
        d = distances.get(c["endpoint"])
        if d is not None and d >= float(c["distancia_min_m"]):
            if sev is None or int(c["severidad"]) > sev:
                sev = int(c["severidad"])
            used.append(c)
    if severity_criterion:
        ov = {}
        if sev is not None:
            ov = {"severidad": sev, "severidad_fuente": "calculada (análisis de consecuencias M12, criterio del usuario)"}
        out.update({"severidad_calculada": sev, "criterio_usuario_cumplido": used,
                    "overrides_lopa": {str(hazop_row): ov} if ov else {},
                    "nota": "Pasa overrides_lopa a lopa_from_hazop; la severidad queda etiquetada como calculada."
                    if ov else "Ningún umbral del criterio del usuario alcanza su distancia: la severidad del HAZOP no cambia."})
    else:
        out["nota"] = ("Sin criterio de severidad del usuario (severity_criterion: [{severidad, endpoint, "
                       "distancia_min_m}]): no se calcula severidad.")
    if modifier:
        out["modificador_sugerido"] = {**modifier, "filas": [hazop_row]}
        out["nota_modificador"] = ("Probabilidad calculada en el receptor del usuario; úsala en conditional_modifiers "
                                   "solo si el owner la acepta.")
    return out


def _props(compound_name: str | None, M: float | None = None) -> tuple[object | None, float | None, list[str]]:
    notes = []
    c = compound(compound_name) if compound_name else None
    if c is not None:
        notes.append(f"Propiedades de {compound_name} (CAS {c.cas}) de thermo/chemicals (ChEDL): M = {c.M * 1000:.2f} "
                     f"g/mol, Tb = {c.Tb:.1f} K, ΔHc neto = {c.Hc_lower / 1e6:.2f} MJ/kg, LFL = {c.LFL}.")
        if M is None:
            M = c.M
    return c, M, notes


# =========================================================================== 1. source term
def release_source_term(kind: str, hole_diameter_m: float | None = None, pressure_Pa: float | None = None,
                        temperature_K: float | None = None, compound_name: str | None = None,
                        density_kg_m3: float | None = None, molar_mass_kg_mol: float | None = None,
                        gamma: float | None = None, liquid_head_m: float = 0.0, discharge_coefficient: float | None = None,
                        tank_area_m2: float | None = None, ambient_pressure_Pa: float | None = None,
                        coolprop_fluid: str | None = None, cp_liquid_J_kgK: float | None = None,
                        boiling_point_K: float | None = None, hvap_J_kg: float | None = None,
                        pool_diameter_m: float | None = None, pool_area_m2: float | None = None,
                        wind_speed_m_s: float | None = None, schmidt: float | None = None,
                        vapour_pressure_Pa: float | None = None) -> dict:
    """Source term of a loss of containment (consequence analysis, Crowl & Louvar 4th ed.). kind: liquido (liquid
    through a hole, Cd 0.61 unless given; tank_area_m2 + liquid_head_m give the draining time), gas (gas through a hole,
    choked or subsonic, Cd 1.0 unless given; γ from thermo if compound_name), flash (flashing fraction: CoolProp
    isenthalpic with coolprop_fluid, and/or Cp(T−Tb)/ΔHv with compound_name), charco (pool evaporation: Crowl & Louvar
    still air, plus Mackay-Matsugu with the user's wind). pressure_Pa is the ABSOLUTE storage pressure. Hole size,
    pressure, temperature and pool size are the user's: missing ones are reported, never assumed. Screening only."""
    try:
        assumptions = [SCREENING]
        pa = ambient_pressure_Pa if ambient_pressure_Pa else P_ATM
        if not ambient_pressure_Pa:
            assumptions.append("Presión ambiente 101 325 Pa (atmósfera estándar; da la del sitio si difiere).")
        c, M, notes = _props(compound_name, molar_mass_kg_mol)
        assumptions += notes
        if kind == "liquido":
            _need(hole_diameter_m=hole_diameter_m, pressure_Pa=pressure_Pa)
            if density_kg_m3 is None and c is not None:
                from thermo import Chemical
                density_kg_m3 = float(Chemical(compound_name, T=temperature_K or 298.15, P=max(pressure_Pa, pa)).rhol)
                assumptions.append(f"Densidad del líquido {density_kg_m3:.1f} kg/m³ (thermo).")
            _need(density_kg_m3=density_kg_m3)
            cd = discharge_coefficient if discharge_coefficient else 0.61
            if not discharge_coefficient:
                assumptions.append("Cd = 0.61 (orificio de arista viva, Re > 30 000, Crowl & Louvar); usa 1.0 para el "
                                   "flujo máximo conservador.")
            r = So.liquid_hole(hole_diameter_m, pressure_Pa - pa, density_kg_m3, liquid_head_m, cd, tank_area_m2)
        elif kind == "gas":
            _need(hole_diameter_m=hole_diameter_m, pressure_Pa=pressure_Pa, temperature_K=temperature_K)
            if gamma is None and c is not None:
                cpm = c.cp_ideal_gas_molar(temperature_K)
                gamma = cpm / (cpm - R)
                assumptions.append(f"γ = {gamma:.3f} de gas ideal (Cp de thermo a {temperature_K:.1f} K).")
            _need(molar_mass_kg_mol=M, gamma=gamma)
            cd = discharge_coefficient if discharge_coefficient else 1.0
            if not discharge_coefficient:
                assumptions.append("Cd = 1.0 (flujo máximo, conservador, Crowl & Louvar).")
            r = So.gas_hole(hole_diameter_m, pressure_Pa, temperature_K, M, gamma, pa, cd)
            assumptions.append("Gas ideal, expansión isentrópica, sin pérdidas en tubería aguas arriba; caudal "
                               "inicial (el recipiente se despresuriza con el tiempo).")
        elif kind == "flash":
            _need(temperature_K=temperature_K)
            r = {}
            if coolprop_fluid:
                r["coolprop"] = So.flash_fraction_coolprop(coolprop_fluid, temperature_K, pressure_Pa, pa)
            if c is not None or (cp_liquid_J_kgK and boiling_point_K and hvap_J_kg):
                tb = boiling_point_K or c.Tb
                hv = hvap_J_kg or c.Hvap_Tb
                cp = cp_liquid_J_kgK or c.Cp_liquid(0.5 * (temperature_K + tb))
                _need(cp_liquid_J_kgK=cp)
                r["cp_constante"] = So.flash_fraction_cp(cp, temperature_K, tb, hv)
                assumptions.append(f"Cp del líquido {cp:.0f} J/kg K a la temperatura media, ΔHv en Tb.")
            if not r:
                raise MissingData("da coolprop_fluid o compound_name (o Cp, Tb y ΔHv) para el flash")
            assumptions.append("Aerosol y lluvia de líquido (rainout) no se calculan: la fracción flasheada es solo "
                               "vapor; el arrastre de gotas puede aumentar la masa en la nube.")
        elif kind == "charco":
            _need(temperature_K=temperature_K)
            if vapour_pressure_Pa is None and c is not None:
                vapour_pressure_Pa = c.psat(temperature_K)
                assumptions.append(f"Psat = {vapour_pressure_Pa:.0f} Pa a {temperature_K:.1f} K (thermo).")
            _need(molar_mass_kg_mol=M, vapour_pressure_Pa=vapour_pressure_Pa)
            if pool_area_m2 is None and pool_diameter_m:
                pool_area_m2 = math.pi * pool_diameter_m ** 2 / 4
            _need(pool_area_m2=pool_area_m2)
            if c is not None and temperature_K >= c.Tb:
                assumptions.append("El líquido está en o sobre su punto de ebullición: es un charco en ebullición "
                                   "(limitado por el calor del suelo), fuera del alcance de estos modelos.")
            r = {"crowl_louvar": So.pool_evaporation_cl(M, vapour_pressure_Pa, temperature_K, pool_area_m2)}
            if wind_speed_m_s is not None and pool_diameter_m:
                sc = schmidt if schmidt else 0.8
                if not schmidt:
                    assumptions.append("Sc = 0.8 (valor típico de vapores en aire, SUPUESTO; da el del compuesto).")
                r["mackay_matsugu"] = So.pool_evaporation_mackay(M, vapour_pressure_Pa, temperature_K, pool_diameter_m,
                                                                 wind_speed_m_s, sc)
            assumptions.append("Charco de área fija (la del dique o la del usuario), sin enfriamiento evaporativo.")
        else:
            raise ValueError("kind: liquido | gas | flash | charco")
        return {"ok": True, "tipo": kind, "resultado": r, "assumptions": assumptions, "aviso": SCREENING}
    except (ValueError, KeyError) as e:
        return _err(e)


# =========================================================================== 2. dispersion
def gas_dispersion(mode: str, wind_speed_m_s: float | None = None, ambient_temperature_K: float | None = None,
                   stability_class: str | None = None, insolation: str | None = None, night_cloud: str | None = None,
                   release_rate_kg_s: float | None = None, release_mass_kg: float | None = None,
                   compound_name: str | None = None, molar_mass_kg_mol: float | None = None,
                   release_temperature_K: float | None = None, release_height_m: float | None = None,
                   terrain: str = "rural", thresholds: list[dict] | None = None, receptor_distance_m: float | None = None,
                   toxic_probit: dict | None = None, exposure_time_s: float | None = None,
                   cloud_density_kg_m3: float | None = None, hazop_row: int | None = None,
                   severity_criterion: list[dict] | None = None) -> dict:
    """Gaussian dispersion of a release (Pasquill-Gifford, Briggs coefficients, Crowl & Louvar 4th ed. tables
    5-2/5-3). mode continua (plume, release_rate_kg_s) | instantanea (puff, release_mass_kg). Weather is the user's:
    wind_speed_m_s (10 m), ambient_temperature_K and stability_class A-F (or insolation fuerte|moderada|ligera|cubierto
    by day, night_cloud nublado|despejado by night). terrain rural|urban. thresholds [{nombre, ppm | mg_m3 |
    fraccion_lfl}] are the user's end points → farthest distance and max half-width. Britter-McQuaid dense-gas check:
    a dense ground-level cloud is NOT computed (Gaussian invalid); an elevated one is computed with an explicit
    warning. Optional receptor_distance_m + toxic_probit {a, b, n, unidad_c ppm, unidad_t min} (+ exposure_time_s for
    a plume) → probability; hazop_row + severity_criterion [{severidad, endpoint, distancia_min_m}] → para_lopa."""
    try:
        _need(wind_speed_m_s=wind_speed_m_s, ambient_temperature_K=ambient_temperature_K)
        if mode not in ("continua", "instantanea"):
            raise ValueError("mode: continua | instantanea")
        if mode == "continua":
            _need(release_rate_kg_s=release_rate_kg_s)
        else:
            _need(release_mass_kg=release_mass_kg)
        assumptions = [SCREENING]
        c, M, notes = _props(compound_name, molar_mass_kg_mol)
        assumptions += notes
        _need(molar_mass_kg_mol=M)
        if stability_class is None:
            stability_class = Dm.stability_class(wind_speed_m_s, insolation, night_cloud)
            assumptions.append(f"Clase de Pasquill {stability_class} desde el viento y la insolación/nubosidad del "
                               "usuario (tabla de Pasquill, bordes de viento 2, 3, 5 y 6 m/s inclusivos por abajo).")
        H = release_height_m if release_height_m is not None else 0.0
        if release_height_m is None:
            assumptions.append("Fuente a nivel del suelo (H = 0): da la altura real de la fuga si es elevada.")
        t_rel = release_temperature_K or ambient_temperature_K
        if release_temperature_K is None:
            assumptions.append("Temperatura de la nube = ambiente (para el criterio de gas denso); en fugas que "
                               "flashean usa la de ebullición.")
        dense = Dm.dense_gas_check(M, t_rel, ambient_temperature_K, wind_speed_m_s,
                                   Q_kg_s=release_rate_kg_s if mode == "continua" else None,
                                   Qstar_kg=release_mass_kg if mode == "instantanea" else None,
                                   rho_release=cloud_density_kg_m3)
        base = {"modo": mode, "clase_estabilidad": stability_class, "terreno": terrain, "altura_fuente_m": H,
                "gas_denso": dense, "aviso": SCREENING}
        if dense["denso"] and H > 0:
            base["valido_gaussiano"] = "con aviso"
            base["aviso_gas_denso"] = (
                f"Nube más pesada que el aire con criterio de Britter-McQuaid {dense['criterio']:.3f} ≥ "
                f"{dense['umbral']} en la fuente. El criterio está deducido para fuentes a nivel del suelo; en esta fuga "
                f"elevada (H = {H:g} m) se calcula el gaussiano, pero ignora el descenso de la nube: puede SUBESTIMAR la "
                "concentración a nivel del suelo cerca de la fuente. Usa un modelo de gas denso para decidir.")
            assumptions.append(base["aviso_gas_denso"])
        elif dense["denso"]:
            base.update({"ok": True, "valido_gaussiano": False, "assumptions": assumptions,
                         "resultado": "NO SE CALCULA: la nube es densa según Britter-McQuaid (criterio "
                                      f"{dense['criterio']:.3f} ≥ {dense['umbral']}); el modelo gaussiano no es válido "
                                      "y no se usa en silencio. Hace falta un modelo de gas denso (Britter-McQuaid, "
                                      "DEGADIS, SLAB): pendiente en AlchemAI."})
            return base
        if mode == "continua":
            conc = lambda x: Dm.plume_concentration(release_rate_kg_s, wind_speed_m_s, x, 0.0, 0.0, H,  # noqa: E731
                                                    stability_class, terrain)
            sy_fn = lambda x: Dm.sigmas_plume(x, stability_class, terrain)[0]  # noqa: E731
            rng = Dm.X_RANGE_PLUME
            assumptions.append("Pluma gaussiana estacionaria con reflexión en el suelo, viento constante, tiempo de "
                               "promediado ≈ 10 min (Pasquill-Gifford), terreno plano; la concentración es la media, "
                               "no el pico instantáneo.")
        else:
            conc = lambda x: Dm.puff_center_concentration(release_mass_kg, x, 0.0, 0.0, H, stability_class)  # noqa: E731
            sy_fn = lambda x: Dm.sigmas_puff(x, stability_class)[0]  # noqa: E731
            rng = (100.0, 10_000.0)
            assumptions.append("Puff gaussiano instantáneo (centro a x = u·t) con reflexión en el suelo; los "
                               "coeficientes de puff no distinguen rural/urbano.")
        to_ppm = lambda ckg: float(Dm.kgm3_to_ppm(ckg, M, ambient_temperature_K))  # noqa: E731
        dist = sorted(set(STD_DISTANCES) | ({receptor_distance_m} if receptor_distance_m else set()))
        table = [{"x_m": x, "C_kg_m3": float(conc(x)), "C_ppm": to_ppm(conc(x)),
                  "fuera_de_rango": not (rng[0] <= x <= rng[1])} for x in dist]
        if mode == "instantanea":
            for row in table:
                row["dosis_kg_s_m3"] = float(Dm.puff_dose(release_mass_kg, wind_speed_m_s, row["x_m"], H=H,
                                                          cls=stability_class))
        res_thr = []
        dist_map: dict[str, float | None] = {}
        for t in thresholds or []:
            name = t.get("nombre", "umbral")
            if "ppm" in t:
                ck = Dm.ppm_to_kgm3(float(t["ppm"]), M, ambient_temperature_K)
            elif "mg_m3" in t:
                ck = float(t["mg_m3"]) * 1e-6
            elif "fraccion_lfl" in t:
                if c is None or c.LFL is None:
                    raise MissingData("LFL: da compound_name de un inflamable o el umbral en ppm")
                ck = Dm.ppm_to_kgm3(float(t["fraccion_lfl"]) * c.LFL * 1e6, M, ambient_temperature_K)
            else:
                raise ValueError("cada umbral: {nombre, ppm | mg_m3 | fraccion_lfl}")
            d = Dm.distance_to_concentration(conc, ck)
            half = None
            if d and math.isfinite(d):
                xs = np.geomspace(1.0, d, 300)
                hw = [float(sy_fn(x)) * math.sqrt(2 * math.log(float(conc(x)) / ck)) if float(conc(x)) > ck else 0.0
                      for x in xs]
                half = max(hw)
            res_thr.append({"nombre": name, "C_kg_m3": ck, "C_ppm": to_ppm(ck), "distancia_m": d,
                            "semiancho_max_m": half,
                            "fuera_de_rango": d is not None and not (rng[0] <= d <= rng[1])})
            dist_map[name] = d
        out = {"valido_gaussiano": True, **base, "ok": True, "concentracion_eje": table, "umbrales": res_thr}
        modifier = None
        if receptor_distance_m and toxic_probit:
            tp = toxic_probit
            cr = to_ppm(conc(receptor_distance_m)) if tp.get("unidad_c", "ppm") == "ppm" else float(conc(receptor_distance_m)) * 1e6
            if mode == "continua":
                _need(exposure_time_s=exposure_time_s)
                t_exp = exposure_time_s
            else:
                sy_r = float(Dm.sigmas_puff(receptor_distance_m, stability_class)[0])
                t_exp = math.sqrt(2 * math.pi) * sy_r / wind_speed_m_s
                assumptions.append(f"Exposición al puff: duración equivalente √(2π)·σx/u = {t_exp:.0f} s con la "
                                   "concentración pico (conservador).")
            t_units = t_exp / 60.0 if tp.get("unidad_t", "min") == "min" else t_exp
            pr = Ef.toxic_probit(cr, t_units, float(tp["a"]), float(tp["b"]), float(tp["n"]))
            out["receptor"] = {"distancia_m": receptor_distance_m, "C": cr, "t": t_units, **pr,
                               "fuente_probit": tp.get("fuente", "usuario")}
            modifier = {"nombre": "probabilidad de efecto tóxico en el receptor (calculada, probit del usuario)",
                        "probabilidad": pr["probabilidad"], "fuente": "calculada por gas_dispersion (M12)"}
        if any(r["fuera_de_rango"] for r in res_thr):
            assumptions.append("Alguna distancia cae fuera de 100 m – 10 km, el rango de las correlaciones de σ: es "
                               "extrapolación.")
        out["assumptions"] = assumptions
        link = _lopa_link(hazop_row, dist_map, severity_criterion, modifier)
        if link:
            out["para_lopa"] = link
        return out
    except (ValueError, KeyError) as e:
        return _err(e)


# =========================================================================== 3. fire radiation
def _endpoints_list(endpoints_kW_m2: list[dict] | None, example: str | None) -> tuple[list[dict], list[str]]:
    notes = []
    eps = list(endpoints_kW_m2 or [])
    if example:
        ex = EXAMPLE_ENDPOINTS[example]
        eps.append({"nombre": f"EJEMPLO {example}", "kW_m2": ex["radiacion_kW_m2"]})
        notes.append("Punto final de EJEMPLO: " + ex["fuente"])
    return eps, notes


def fire_radiation(kind: str, ambient_temperature_K: float | None = None, compound_name: str | None = None,
                   heat_of_combustion_J_kg: float | None = None, pool_diameter_m: float | None = None,
                   burning_rate_kg_m2s: float | None = None, wind_speed_m_s: float | None = None,
                   relative_humidity: float | None = None, sep_method: str = "both",
                   mass_kg: float | None = None, burst_pressure_Pa: float | None = None,
                   pool_model: str = "sfpe", burning_rate_inf_kg_m2s: float | None = None, k_beta_m: float | None = None,
                   flame_height: str = "thomas", transmissivity: float | None = None,
                   air_kinematic_viscosity_m2_s: float | None = None,
                   radiant_fraction: float | None = None, release_rate_kg_s: float | None = None,
                   distances_m: list[float] | None = None, endpoints_kW_m2: list[dict] | None = None,
                   example_endpoints: str | None = None, exposure_time_s: float | None = None,
                   receptor_distance_m: float | None = None, target_height_m: float | None = None,
                   hazop_row: int | None = None, severity_criterion: list[dict] | None = None) -> dict:
    """Thermal radiation vs distance. kind charco: solid-flame pool fire, pool_model sfpe (Burgess burning rate or
    m''∞ + kβ, Thomas or Heskestad height, AGA tilt, Mudan-Croce/Shokri-Beyler SEP) or tno (TNO Yellow Book, needs
    radiant_fraction Fs); bola_fuego: TNO fireball from mass_kg and burst_pressure_Pa (PSV set, N/m²); chorro: point
    source with the user's radiant_fraction. ambient_temperature_K is the user's; relative_humidity (0-1) gives the
    Bagster transmissivity, otherwise τ = 1 (conservative) or the user's transmissivity. endpoints_kW_m2
    [{nombre, kW_m2}] are the user's; example_endpoints "epa_rmp" adds a labelled EXAMPLE. receptor_distance_m +
    exposure_time_s → Eisenberg burn-death probability; hazop_row + severity_criterion → para_lopa. Screening only."""
    try:
        _need(ambient_temperature_K=ambient_temperature_K)
        assumptions = [SCREENING]
        c, _, notes = _props(compound_name)
        assumptions += notes
        hc = heat_of_combustion_J_kg or (c.Hc_lower if c else None)
        _need(heat_of_combustion_J_kg=hc)
        if hc <= 0:
            raise ValueError(f"{compound_name} no es combustible (ΔHc = 0)")
        if transmissivity is not None:
            assumptions.append(f"Transmisividad τ = {transmissivity:g} dada por el usuario.")
        elif relative_humidity is None:
            assumptions.append("Sin humedad relativa del usuario: transmisividad τ = 1 (conservador).")
        else:
            assumptions.append("Transmisividad de Bagster (TNO YB ec. 6.29), válida para 1e4 < pw·x < 1e5 N/m; fuera de "
                               "ese rango es extrapolación (el Yellow Book prefiere sus gráficas de H2O y CO2).")
        eps, n2 = _endpoints_list(endpoints_kW_m2, example_endpoints)
        assumptions += n2
        th = target_height_m if target_height_m is not None else 0.0
        if target_height_m is None and kind == "charco":
            assumptions.append("Blanco a nivel del suelo, viento hacia el blanco (peor caso).")
        info: dict
        if kind == "charco":
            _need(pool_diameter_m=pool_diameter_m)
            rho_a = air_density(ambient_temperature_K)
            tb = c.Tb if c else ambient_temperature_K
            rho_v = P_ATM * (c.M if c else 0.029) / (R * tb)
            cp = c.Cp_liquid(0.5 * (tb + ambient_temperature_K)) if (c and tb > ambient_temperature_K) else None
            if c is None and not (burning_rate_kg_m2s or (burning_rate_inf_kg_m2s and k_beta_m)):
                raise MissingData("da compound_name o la tasa de quemado (burning_rate_kg_m2s o m''∞ + kβ)")
            nu = air_kinematic_viscosity_m2_s
            if pool_model == "tno" and nu is None:
                import CoolProp.CoolProp as CP
                nu = CP.PropsSI("V", "T", ambient_temperature_K, "P", P_ATM, "Air") / rho_a
                assumptions.append(f"ν del aire {nu:.3g} m²/s (CoolProp) para el número de Reynolds de la inclinación.")
            pf = Fi.pool_fire(pool_diameter_m, hc, c.Hvap_Tb if c else 1.0, tb, ambient_temperature_K, rho_a, rho_v, cp,
                              burning_rate_kg_m2s, wind_speed_m_s, relative_humidity, sep_method, target_height=th,
                              model=pool_model, burning_rate_inf=burning_rate_inf_kg_m2s, k_beta=k_beta_m,
                              flame_height=flame_height, radiant_fraction=radiant_fraction, nu_air=nu,
                              tau_user=transmissivity)
            if pf["x_bajo_llama_m"] > pf["x_min_m"] * 1.01:
                assumptions.append(f"La llama inclinada cubre el suelo hasta {pf['x_bajo_llama_m']:.1f} m del centro: "
                                   "ahí el blanco está bajo la llama (considéralo envuelto; el factor de vista es "
                                   "solo indicativo).")
            if pool_model == "tno":
                assumptions.append("Modelo TNO: fracción de la llama cubierta de hollín ς = 0.8 y SEP del hollín 20 kW/m² "
                                   "(valores del Yellow Book para combustibles que humean).")
            flux = lambda x: pf["_flux"](x)[0]  # noqa: E731
            x_lo = pf["x_min_m"]
            info = {k: v for k, v in pf.items() if not k.startswith("_")}
            if not wind_speed_m_s:
                assumptions.append("Sin viento del usuario: llama vertical (Thomas sin viento).")
            assumptions.append("Llama cilíndrica sólida con SEP uniforme; charco circular de diámetro fijo (el del "
                               "dique o el del usuario); sin humo que tape la llama más allá de lo que ya incluye la SEP.")
            t_exp = exposure_time_s
        elif kind == "bola_fuego":
            _need(mass_kg=mass_kg)
            hv = c.Hvap_Tb if c else None
            cpl = c.Cp_liquid(min(ambient_temperature_K, c.Tb)) if c else None
            fb = Fi.fireball_tno(mass_kg, hc, burst_pressure_Pa, radiant_fraction, hv, cpl, ambient_temperature_K)
            flux = lambda x: Fi.fireball_flux(fb, x, relative_humidity, ambient_temperature_K, transmissivity)[0]  # noqa: E731
            x_lo = 0.0 + 1e-3
            info = fb
            t_exp = exposure_time_s or fb["duracion_s"]
            assumptions.append("Bola de fuego esférica de diámetro máximo, centro a una altura D, blanco orientado al "
                               "centro; exposición = duración de la bola (sin escape) si no das otra.")
        elif kind == "chorro":
            _need(release_rate_kg_s=release_rate_kg_s, radiant_fraction=radiant_fraction)
            def flux(x):
                tau, _ = Fi.transmissivity_tno(relative_humidity, ambient_temperature_K, x, transmissivity)
                return Fi.jet_fire_point_source(release_rate_kg_s, hc, radiant_fraction, x, tau)
            x_lo = 1.0
            info = {"modelo": "fuente puntual (API 521 / Hajek-Ludwig)", "fraccion_radiada": radiant_fraction}
            assumptions.append("Chorro de fuego como fuente puntual en el centro de la llama: válido lejos de la "
                               "llama (r ≫ longitud de llama); el modelo de llama sólida (Chamberlain) está pendiente.")
            t_exp = exposure_time_s
        else:
            raise ValueError("kind: charco | bola_fuego | chorro")
        dist = sorted(set(distances_m or STD_DISTANCES) | ({receptor_distance_m} if receptor_distance_m else set()))
        table = [{"x_m": x, "q_kW_m2": float(flux(x))} for x in dist if x > x_lo]
        res_ep, dist_map = [], {}
        for e in eps:
            d = Fi.distance_to_flux(flux, float(e["kW_m2"]), x_lo)
            res_ep.append({**e, "distancia_m": d, "desde": "centro del charco" if kind == "charco" else
                           ("proyección del centro de la bola en el suelo" if kind == "bola_fuego" else "centro de la llama")})
            dist_map[e["nombre"]] = d
        out = {"ok": True, "tipo": kind, "modelo": info, "radiacion": table, "puntos_finales": res_ep, "aviso": SCREENING}
        modifier = None
        if receptor_distance_m:
            q = float(flux(receptor_distance_m))
            rec = {"distancia_m": receptor_distance_m, "q_kW_m2": q}
            if t_exp:
                pr = Ef.thermal_probit(q, t_exp)
                rec.update({"tiempo_exposicion_s": t_exp, "probit": pr})
                modifier = {"nombre": "probabilidad de fatalidad por radiación en el receptor (calculada, probit "
                                      "Eisenberg)", "probabilidad": pr["probabilidad"],
                            "fuente": "calculada por fire_radiation (M12)"}
            else:
                rec["nota"] = "da exposure_time_s para la probabilidad (probit)"
            out["receptor"] = rec
        out["assumptions"] = assumptions
        link = _lopa_link(hazop_row, dist_map, severity_criterion, modifier)
        if link:
            out["para_lopa"] = link
        return out
    except (ValueError, KeyError) as e:
        return _err(e)


# =========================================================================== 4. explosion
def explosion_overpressure(method: str, flammable_mass_kg: float | None = None, efficiency: float | None = None,
                           compound_name: str | None = None, heat_of_combustion_J_kg: float | None = None,
                           tnt_mass_kg: float | None = None, ambient_pressure_Pa: float | None = None,
                           distances_m: list[float] | None = None, endpoints_kPa: list[dict] | None = None,
                           example_endpoints: str | None = None, receptor_distance_m: float | None = None,
                           hazop_row: int | None = None, severity_criterion: list[dict] | None = None) -> dict:
    """Blast overpressure vs distance by TNT equivalence (Crowl & Louvar 4th ed. eq. 6-29/6-30, Kinney-Graham
    surface burst): flammable_mass_kg in the cloud and the explosion efficiency (the user's, typically 1-10 %; never
    assumed), heat of combustion from compound_name or heat_of_combustion_J_kg; or tnt_mass_kg directly.
    endpoints_kPa [{nombre, kPa}] are the user's; example_endpoints "epa_rmp" adds 1 psi as a labelled EXAMPLE.
    receptor_distance_m → Eisenberg probits (lung hemorrhage, eardrum, structures, glass). TNO multi-energy and
    Baker-Strehlow-Tang are pending (no legible curve fits). hazop_row + severity_criterion → para_lopa."""
    try:
        assumptions = [SCREENING]
        pa = ambient_pressure_Pa or P_ATM
        if not ambient_pressure_Pa:
            assumptions.append("Presión ambiente 101 325 Pa (atmósfera estándar).")
        if method != "tnt":
            raise ValueError("method: tnt (multi-energía TNO y Baker-Strehlow-Tang: pendientes, ver NEXT_STEP)")
        c, _, notes = _props(compound_name)
        assumptions += notes
        if tnt_mass_kg is None:
            _need(flammable_mass_kg=flammable_mass_kg, efficiency=efficiency)
            hc = heat_of_combustion_J_kg or (c.Hc_lower if c else None)
            _need(heat_of_combustion_J_kg=hc)
            tnt_mass_kg = Ex.tnt_mass(flammable_mass_kg, hc, efficiency)
            assumptions.append(f"Eficiencia de explosión η = {efficiency:g} (dato del usuario); E_TNT = 4686 kJ/kg "
                               "(1120 cal/g, Crowl & Louvar).")
        assumptions.append("Equivalencia TNT: explosión puntual en campo libre sobre el suelo; ignora congestión y "
                           "confinamiento reales (por eso η es incierta, 1-10 %) y sobreestima cerca y subestima lejos "
                           "frente a multi-energía/BST.")
        eps = list(endpoints_kPa or [])
        if example_endpoints:
            ex = EXAMPLE_ENDPOINTS[example_endpoints]
            eps.append({"nombre": f"EJEMPLO {example_endpoints}", "kPa": ex["sobrepresion_kPa"]})
            assumptions.append("Punto final de EJEMPLO: " + ex["fuente"])
        dist = sorted(set(distances_m or STD_DISTANCES) | ({receptor_distance_m} if receptor_distance_m else set()))
        table = [{"x_m": x, **Ex.tnt_overpressure(tnt_mass_kg, x, pa)} for x in dist]
        for row in table:
            row["sobrepresion_kPa"] = row.pop("sobrepresion_Pa") / 1000.0
        res_ep, dist_map = [], {}
        for e in eps:
            d = Ex.tnt_distance_to(tnt_mass_kg, float(e["kPa"]) * 1000.0, pa)
            res_ep.append({**e, "distancia_m": d})
            dist_map[e["nombre"]] = d
        out = {"ok": True, "metodo": "equivalencia TNT (Kinney y Graham 1985)", "masa_TNT_kg": tnt_mass_kg,
               "sobrepresion": table, "puntos_finales": res_ep, "aviso": SCREENING}
        modifier = None
        if receptor_distance_m:
            p = Ex.tnt_overpressure(tnt_mass_kg, receptor_distance_m, pa)["sobrepresion_Pa"]
            probs = {m: Ef.overpressure_probit(p, m) for m in ("muerte_hemorragia_pulmonar", "ruptura_timpano",
                                                                "dano_estructural", "rotura_vidrios")}
            out["receptor"] = {"distancia_m": receptor_distance_m, "sobrepresion_kPa": p / 1000.0, "probits": probs}
            modifier = {"nombre": "probabilidad de fatalidad por sobrepresión en el receptor (calculada, probit "
                                  "hemorragia pulmonar, Eisenberg)",
                        "probabilidad": probs["muerte_hemorragia_pulmonar"]["probabilidad"],
                        "fuente": "calculada por explosion_overpressure (M12)"}
            assumptions.append("El probit de hemorragia pulmonar solo cuenta el efecto directo de la onda; las "
                               "muertes por proyectiles o colapso de edificios suelen dominar a menor sobrepresión.")
        out["assumptions"] = assumptions
        link = _lopa_link(hazop_row, dist_map, severity_criterion, modifier)
        if link:
            out["para_lopa"] = link
        return out
    except (ValueError, KeyError) as e:
        return _err(e)
