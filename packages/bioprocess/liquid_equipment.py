"""Liquid-side equipment of a bioprocess / agro-industrial plant (bloque M26_E1): liquid-liquid extraction and decanters,
forward-feed multiple-effect evaporators and storage tanks, each with sizing and purchase cost.

What is solved here and with what:
  * Liquid-liquid extraction. Two routes, never mixed:
      - thermodynamic: every stage is an isothermal two-liquid flash of the repo's own engine
        (`packages.properties.flash.independent_flash`, thermo; models NRTL / UNIQUAC with the user's parameters or the
        predictive UNIFAC family). A single decanter is one flash; an N-stage countercurrent cascade is solved by stage-to-
        stage successive substitution until the component flows stop changing, and the stage balances are the evidence.
      - constant distribution coefficients given by the user with immiscible carrier and solvent: the stage balances are a
        linear tridiagonal system (numpy); the Kremser equation is the independent closed form it is checked against.
  * Gravity settler: Stokes terminal velocity of the user's drop size, plug flow of the continuous phase through the
    interface (u_c = Q_c / A_i <= u_d), vessel from the user's hold-up time; the larger of the two governs.
  * Multiple-effect evaporator, forward feed: mass, solids and enthalpy balance of every effect plus q = U A dT, solved
    simultaneously (scipy.optimize.root) for equal areas or for given effect pressures. Steam and vapour properties are
    IAPWS-95 (CoolProp); boiling-point rise, solution heat capacity and U are the user's.
  * Storage tank: volume from flow x time / fill fraction, cylinder from the user's H/D.

Costs. Tank and liquid-liquid centrifuge correlations are BioSTEAM's own objects, read at run time (never retyped):
`storage_tank_purchase_cost_algorithms`, `mix_tank_purchase_cost_algorithms`, `LiquidsCentrifuge.cost_items`. The gravity
settler is a Seider pressure vessel through the repo's `packages.costing.sslw.cost_vessel` (IDAES). Evaporator bodies are
interpolated in the purchased-cost table of DOE/NETL-2002/1169 (Loh, Lyons & White, «Process Equipment Cost Estimation»,
1st quarter 1998 USD, read 2026-11-16, https://www.osti.gov/servlets/purl/797810): BioSTEAM 2.54's evaporator cost
functions return about a tenth of that table for the same area, so they are reported as a second opinion only.
Every cost is PURCHASE cost scaled to the caller's `year` / `cepci` with the repo convention (`CEPCI_IDAES`).

BioSTEAM hygiene: `biosteam` / `thermosteam` are imported lazily inside `_biosteam_session`, which restores thermo's
RAISE_PROPERTY_CALCULATION_ERROR flags, the thermosteam default thermo, `bst.CE`, the active flowsheet, the electricity
price and the heat-utility prices, and captures BioSTEAM's Cost/Design warnings (returned in Spanish)."""
from __future__ import annotations

import math
import warnings as _pywarnings
from contextlib import contextmanager

G = 9.80665                       # m/s2
T_ZERO_C = 273.15                 # K, reference of the solution enthalpy h = cp(x)·(T − 273.15)
GAL_M3 = 0.003785411784           # US gallon
FT2_M2 = 0.09290304
_MISSING = object()

FLASH_MODELS = ("NRTL", "UNIQUAC", "WILSON", "UNIFAC", "UNIFAC_DORTMUND", "UNIFAC_LLE", "PR", "SRK")

# DOE/NETL-2002/1169, appendix table «Evaporator», 1st quarter 1998 USD, carbon steel (A285C shell, CS tubes).
NETL_SOURCE = ("DOE/NETL-2002/1169, Loh, Lyons & White, «Process Equipment Cost Estimation, Final Report» (enero de 2002), "
               "dólares del 1.er trimestre de 1998; https://www.osti.gov/servlets/purl/797810 (leído el 2026-11-16)")
NETL_CEPCI_1998 = 389.5           # Table 11 of the same report (annual average 1998)
NETL_EVAPORATOR_1998 = {
    "vertical_tube": {"area_ft2": [100, 500, 1000, 2000, 3000, 4000, 5000, 6000],
                      "purchased_usd": [62600, 151600, 221900, 324700, 405700, 475200, 537100, 593700],
                      "installed_usd": [120800, 273500, 388400, 555200, 689100, 803300, 904700, 997000],
                      "biosteam_type": "Long-tube vertical", "label": "tubos verticales estándar"},
    "horizontal_tube": {"area_ft2": [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                        "purchased_usd": [34500, 81100, 117100, 169000, 209500, 244100, 274400, 302600, 328300, 352400,
                                          375100, 396600],
                        "installed_usd": [73300, 161300, 226300, 317100, 386300, 444300, 496800, 545600, 590500, 632400,
                                          671900, 709200],
                        "biosteam_type": "Horizontal tube", "label": "tubos horizontales estándar"},
}

TANK_TYPES = {"cone_roof": ("storage", "Cone roof", "Seider et al. (2017), techo cónico"),
              "floating_roof": ("storage", "Floating roof", "Seider et al. (2017), techo flotante"),
              "field_erected": ("storage", "Field erected", "Apostolakou et al. (2009), tanque montado en campo"),
              "mix_tank": ("mix", "Conventional", "Apostolakou et al. (2009), tanque de proceso con agitación")}
TANK_MATERIALS = {"carbon_steel": "Carbon steel", "stainless_steel": "Stainless steel", "copper": "Copper",
                  "nickel": "Nickel", "titanium_clad": "Titanium clad", "titanium": "Titanium"}
TANK_TO_VESSEL_MATERIAL = {"carbon_steel": "CarbonSteel", "stainless_steel": "StainlessSteel304", "nickel": "Nickel200",
                           "titanium": "Titanium"}

# Published rules of thumb for gravity decanters (never applied here: the caller passes the values explicitly).
DECANTER_TYPICAL_VALUES = {
    "droplet_diameter_m": {"value": 150e-6, "note": "tamaño de gota de diseño, «muy por debajo de lo habitual en la "
                                                    "alimentación de un decantador»"},
    "residence_time_s": {"low": 120.0, "high": 600.0, "note": "retención de 2 a 10 min si no se forman emulsiones"},
    "length_to_diameter": {"low": 3.0, "high": 5.0, "note": "relación largo/diámetro de un decantador horizontal"},
    "max_settling_velocity_m_s": {"value": 4e-3, "note": "si Stokes da más, la regla publicada es diseñar con 4e-3 m/s"},
    "source": "S. D. Al-Bayati, «Liquid-liquid separators» (apuntes de la Universidad de Bagdad; método de Coulson & "
              "Richardson vol. 6), https://uomus.edu.iq/img/lectures21/MUCLecture_2024_4232530.pdf (leído el 2026-11-16)",
    "flag": "VALORES TÍPICOS PUBLICADOS, NO DATOS DE TU MEZCLA: la herramienta no los aplica; pásalos tú si los aceptas",
}


# ----------------------------------------------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------------------------------------------
def _num(value, name: str, purpose: str, minimum: float | None = 0.0, strict: bool = True) -> float:
    """A required finite number; a missing one is refused in Spanish saying what it is for."""
    if value is None:
        raise ValueError(f"Falta '{name}': {purpose}. No se asume ningún valor por defecto.")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"'{name}' debe ser un número finito ({purpose}).")
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        raise ValueError(f"'{name}' = {value} no es válido: debe ser {'mayor que' if strict else 'al menos'} {minimum:g} "
                         f"({purpose}).")
    return float(value)


def _clean(x):
    """Plain JSON types only (numpy scalars and booleans become Python ones; a non-finite float becomes None)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, str)) or x is None:
        return x
    if hasattr(x, "item") and not isinstance(x, (int, float)):
        x = x.item()
    if isinstance(x, float):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def _cepci(year: int, cepci: float | None) -> tuple[float, str]:
    """Index value and the basis label, identical to the labels of `packages.costing.sslw` so that capital_investment sees
    one single basis."""
    from packages.costing.sslw import CEPCI_IDAES
    if cepci is not None:
        _num(cepci, "cepci", "valor del índice CEPCI al que se quiere el costo")
        return float(cepci), f"CEPCI {cepci} (dado por el usuario)"
    if year not in CEPCI_IDAES:
        raise ValueError(f"El repositorio tiene CEPCI de 2015 a 2023 (tabla de IDAES); para {year} pasa cepci=<valor del "
                         "índice de ese año>.")
    return CEPCI_IDAES[year], f"CEPCI {CEPCI_IDAES[year]} ({year}, tabla de IDAES)"


def _translate_warning(w) -> str:
    kind = getattr(w.category, "__name__", "Warning")
    return f"Aviso de BioSTEAM ({kind}, texto original): {str(w.message)}"


@contextmanager
def _biosteam_session():
    """Import BioSTEAM lazily and give it back together with the list where its warnings are recorded. On exit every global
    that BioSTEAM / thermosteam touch is put back exactly as it was found (see the module docstring)."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    flags = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    saved = None
    try:
        with _pywarnings.catch_warnings(record=True) as caught:
            _pywarnings.simplefilter("always")
            import biosteam as bst
            import thermosteam as tmo
            agents = list(bst.HeatUtility.heating_agents) + list(bst.HeatUtility.cooling_agents)
            saved = {"CE": bst.CE, "power": bst.PowerUtility.price, "flowsheet": bst.main_flowsheet.get_flowsheet(),
                     "thermo": getattr(tmo.settings, "_thermo", _MISSING),
                     "heating": list(bst.HeatUtility.heating_agents), "cooling": list(bst.HeatUtility.cooling_agents),
                     "agents": [(a, a.regeneration_price, a.heat_transfer_price) for a in agents]}
            n_import = len(caught)
            yield bst, caught, n_import
    finally:
        if saved is not None:
            import biosteam as bst
            import thermosteam as tmo
            bst.CE = saved["CE"]
            bst.PowerUtility.price = saved["power"]
            if bst.main_flowsheet.get_flowsheet() is not saved["flowsheet"]:
                bst.main_flowsheet.set_flowsheet(saved["flowsheet"])
            if saved["thermo"] is _MISSING:
                if hasattr(tmo.settings, "_thermo"):
                    del tmo.settings._thermo
            elif getattr(tmo.settings, "_thermo", _MISSING) is not saved["thermo"]:
                tmo.settings._thermo = saved["thermo"]
            bst.HeatUtility.heating_agents[:] = saved["heating"]
            bst.HeatUtility.cooling_agents[:] = saved["cooling"]
            for agent, regeneration, transfer in saved["agents"]:
                agent.regeneration_price, agent.heat_transfer_price = regeneration, transfer
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flags


def _bst_warnings(caught: list, start: int) -> list[str]:
    """BioSTEAM's own Cost/Design warnings raised after the import (import-time SyntaxWarnings are not the user's business)."""
    return [_translate_warning(w) for w in caught[start:] if "biosteam" in getattr(w.category, "__module__", "")]


def _vessel_cost(diameter_m: float, length_m: float, vertical: bool, material: str, year: int, cepci: float | None) -> dict:
    """Seider pressure vessel through the repo's IDAES wrapper, with its carbon-steel base (what F_BM refers to)."""
    from packages.costing import sslw
    full = sslw.cost_vessel(diameter_m, length_m, vertical=vertical, material=material, year=year, cepci=cepci)
    base = sslw.cost_vessel(diameter_m, length_m, vertical=vertical, material="CarbonSteel", year=year, cepci=cepci)
    return {"purchase_cost_usd": full["purchase_cost_usd"], "base_purchase_cost_usd": base["purchase_cost_usd"],
            "platforms_ladders_purchase_cost_usd": full.get("platforms_ladders_purchase_cost_usd", 0.0),
            "shell_weight_lb": full["detail"].get("weight"), "basis": full["basis"], "method": full["method"]}


# ----------------------------------------------------------------------------------------------------------------------
# storage tank
# ----------------------------------------------------------------------------------------------------------------------
def _tank_algorithm(bst, tank_type: str):
    from biosteam.units.design_tools import tank_design as TD
    family, key, _ = TANK_TYPES[tank_type]
    table = TD.storage_tank_purchase_cost_algorithms if family == "storage" else TD.mix_tank_purchase_cost_algorithms
    return table[key], TD.compute_number_of_tanks_and_purchase_cost


def storage_tank(flow_kg_s: float | None = None, density_kg_m3: float | None = None, fill_fraction: float | None = None,
                 holdup_time_s: float | None = None, inventory_days: float | None = None, tank_type: str | None = None,
                 material: str | None = None, aspect_ratio_H_D: float | None = None, number_of_tanks: int | None = None,
                 year: int = 2023, cepci: float | None = None) -> dict:
    """Storage / surge tank: working volume = flow × time, total volume = working / fill_fraction, vertical cylinder from the
    caller's H/D, and purchase cost with BioSTEAM's tank correlations (their volume range is stated and enforced as the
    maximum per tank; below the minimum BioSTEAM warns and so does this function). Nothing is defaulted."""
    purpose = "dimensionar el tanque"
    m = _num(flow_kg_s, "flow_kg_s", "caudal másico que entra o sale del tanque")
    rho = _num(density_kg_m3, "density_kg_m3", "densidad del líquido almacenado, para pasar de masa a volumen")
    fill = _num(fill_fraction, "fill_fraction", "fracción del volumen total que ocupa el líquido (volumen útil / total)")
    if fill > 1.0:
        raise ValueError("'fill_fraction' es una fracción entre 0 y 1 (volumen útil / volumen total).")
    if (holdup_time_s is None) == (inventory_days is None):
        raise ValueError("Da exactamente uno de 'holdup_time_s' (tiempo de retención) o 'inventory_days' (días de "
                         "inventario): es lo que fija el volumen del tanque y no se supone.")
    time_s = (_num(holdup_time_s, "holdup_time_s", purpose) if holdup_time_s is not None
              else 86400.0 * _num(inventory_days, "inventory_days", purpose))
    if (tank_type is None) != (material is None):
        raise ValueError("Para el costo hacen falta 'tank_type' y 'material' a la vez (o ninguno, y solo se dimensiona): "
                         f"tank_type en {sorted(TANK_TYPES)}, material en {sorted(TANK_MATERIALS)}.")
    if tank_type is not None and tank_type not in TANK_TYPES:
        raise ValueError(f"tank_type '{tank_type}' no existe; opciones: {sorted(TANK_TYPES)}.")
    if material is not None and material not in TANK_MATERIALS:
        raise ValueError(f"material '{material}' no existe; opciones (factores de material de BioSTEAM): "
                         f"{sorted(TANK_MATERIALS)}.")
    if aspect_ratio_H_D is not None:
        _num(aspect_ratio_H_D, "aspect_ratio_H_D", "relación altura/diámetro del cilindro")
    if number_of_tanks is not None and (isinstance(number_of_tanks, bool) or not isinstance(number_of_tanks, int)
                                        or number_of_tanks < 1):
        raise ValueError("'number_of_tanks' debe ser un entero ≥ 1.")

    warnings: list[str] = []
    assumptions = ["Cilindro vertical de fondo y techo planos: el volumen total es π·D²·H/4 (sin cabezas ni volumen del techo).",
                   "El volumen útil es caudal × tiempo; fill_fraction reparte el resto (cámara de vapor, fondo no bombeable)."]
    not_included = ["Tamaños normalizados API 650 (no se abrió ninguna tabla de tamaños estándar: la geometría sale solo de "
                    "tu H/D).", "Cimentación, dique de contención, tuberías, bombas, instrumentación, aislamiento y "
                    "serpentines de calentamiento.", "Espesor de pared, viento, sismo y presión de diseño.",
                    "Emisiones por respiración y pérdidas de producto."]
    q = m / rho
    working = q * time_s
    total = working / fill
    cost_block = None
    equipment: list[dict] = []
    n_tanks = number_of_tanks or 1
    in_range = None
    if tank_type is not None:
        index, basis = _cepci(year, cepci)
        with _biosteam_session() as (bst, caught, n0):
            alg, compute = _tank_algorithm(bst, tank_type)
            to_m3 = float(alg.V_units.conversion_factor("m^3"))
            v_min_m3, v_max_m3 = alg.V_min * to_m3, alg.V_max * to_m3
            n_min = max(1, math.ceil(total / v_max_m3 - 1e-12))
            if number_of_tanks is None:
                n_tanks = n_min
            elif number_of_tanks < n_min:
                raise ValueError(f"La correlación '{tank_type}' llega a {v_max_m3:.4g} m³ por tanque: para {total:.4g} m³ "
                                 f"hacen falta al menos {n_min} tanques (pediste {number_of_tanks}).")
            each = total / n_tanks
            # one tank, default material, at bst.CE (the tiny shrink keeps BioSTEAM from splitting a tank that sits
            # exactly on the upper bound because of round-off in the unit conversion)
            _, cp_default = compute(min(each, v_max_m3 * (1.0 - 1e-12)), alg)
            ce_library = float(bst.CE)
            from biosteam.units.design_tools.specification_factors import vessel_material_factors as FM
            f_default, f_material = FM.get(alg.material, 1.0), FM[TANK_MATERIALS[material]]
            direct = float(alg.f_Cp(each / to_m3)) * ce_library / alg.CE   # the same correlation evaluated directly
            f_bm = float(bst.StorageTank._F_BM_default["Tank"])
            warnings += _bst_warnings(caught, n0)
        base_unit = cp_default / f_default * index / ce_library
        unit = base_unit * f_material
        in_range = bool(v_min_m3 * (1 - 1e-12) <= each <= v_max_m3 * (1 + 1e-12))
        if each < v_min_m3:
            warnings.append(f"Volumen por tanque {each:.4g} m³ por debajo del mínimo de la correlación ({v_min_m3:.4g} m³): "
                            "el costo es una EXTRAPOLACIÓN hacia abajo y suele quedar bajo.")
        if n_tanks > 1 and number_of_tanks is None:
            warnings.append(f"El volumen total supera el máximo de la correlación ({v_max_m3:.4g} m³ por tanque): se reparte "
                            f"en {n_tanks} tanques iguales, como hace BioSTEAM.")
        label = TANK_TYPES[tank_type][2]
        cost_block = {"correlation": label, "library": "BioSTEAM (biosteam.units.design_tools.tank_design), leída en "
                      "ejecución", "volume_range_m3": [v_min_m3, v_max_m3], "correlation_cepci": float(alg.CE),
                      "correlation_material": alg.material, "material_factor": f_material,
                      "material_factor_source": "BioSTEAM vessel_material_factors",
                      "unit_purchase_cost_usd": unit, "unit_base_purchase_cost_usd": base_unit,
                      "purchase_cost_usd": unit * n_tanks, "base_purchase_cost_usd": base_unit * n_tanks,
                      "number_of_tanks": n_tanks, "cost_year": year if cepci is None else None, "cepci": index,
                      "basis": basis, "biosteam_CE": ce_library, "F_BM": f_bm,
                      "F_BM_source": "BioSTEAM StorageTank (igual que el tipo 'storage_tank' de bare_module_factors)",
                      "within_correlation_range": in_range,
                      "direct_evaluation_rel_diff": abs(direct - cp_default) / max(abs(cp_default), 1e-300)}
        equipment.append({"name": f"Tanque {tank_type} ({material}) ×{n_tanks}", "type": "storage_tank",
                          "purchase_cost_usd": unit * n_tanks, "base_purchase_cost_usd": base_unit * n_tanks,
                          "basis": basis, "number_of_units": n_tanks, "bare_module_factor": f_bm,
                          "bare_module_factor_source": "BioSTEAM StorageTank"})
        assumptions.append(f"Costo de COMPRA (no instalado) con la correlación de BioSTEAM «{label}» en {alg.material} "
                           f"(CEPCI {alg.CE:g}), llevada a {basis} y multiplicada por el factor de material de BioSTEAM "
                           f"({f_material:g}).")
    else:
        each = total / n_tanks
        not_included.append("Costo: no se pidió (faltan tank_type y material).")
    geometry = None
    cross = None
    if aspect_ratio_H_D is not None:
        d = (4.0 * each / (math.pi * aspect_ratio_H_D)) ** (1.0 / 3.0)
        h = aspect_ratio_H_D * d
        geometry = {"diameter_m": d, "height_m": h, "aspect_ratio_H_D": aspect_ratio_H_D, "liquid_height_m": fill * h,
                    "volume_m3": math.pi * d * d * h / 4.0}
        if cost_block is not None:
            vessel_material = TANK_TO_VESSEL_MATERIAL.get(material)
            if vessel_material is None:
                cross = {"available": False, "reason": f"no hay material equivalente a '{material}' en la correlación de "
                                                       "recipientes a presión de Seider/IDAES"}
            else:
                try:
                    full = _vessel_cost(d, h, True, vessel_material, year, cepci)
                    weight = full["shell_weight_lb"]
                    cross = {"available": True, "purchase_cost_usd_per_tank": full["purchase_cost_usd"],
                             "ratio_to_tank_correlation": full["purchase_cost_usd"] / cost_block["unit_purchase_cost_usd"],
                             "shell_weight_lb": weight,
                             "within_seider_weight_range": bool(weight is not None and 4200.0 <= weight <= 1.0e6),
                             "note": "recipiente A PRESIÓN vertical de Seider (equipment_cost 'vessel' del repositorio) con "
                                     "las mismas dimensiones: es una cota superior, no el costo de un tanque atmosférico"}
                except Exception as e:  # noqa: BLE001 — a cross-check that cannot be evaluated is reported, not hidden
                    cross = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    else:
        not_included.append("Diámetro y altura: falta aspect_ratio_H_D (no se supone una relación H/D).")
    closure = abs(n_tanks * each * fill - q * time_s) / (q * time_s)
    geo = abs(geometry["volume_m3"] - each) / each if geometry else 0.0
    direct_rel = cost_block["direct_evaluation_rel_diff"] if cost_block else 0.0
    reasons = []
    if closure > 1e-9:
        reasons.append("el volumen útil no cierra con caudal × tiempo")
    if geo > 1e-9:
        reasons.append("la geometría no reproduce el volumen")
    if direct_rel > 1e-9:
        reasons.append("la correlación evaluada directamente no coincide con la función de BioSTEAM")
    return _clean({"volumetric_flow_m3_s": q, "storage_time_s": time_s, "working_volume_m3": working, "total_volume_m3": total,
            "fill_fraction": fill, "number_of_tanks": n_tanks, "volume_per_tank_m3": each,
            "volume_per_tank_gal": each / GAL_M3, "geometry_per_tank": geometry, "cost": cost_block, "equipment": equipment,
            "cross_check_pressure_vessel": cross, "warnings": warnings, "assumptions": assumptions,
            "not_included": not_included,
            "method": "Balance de volumen (caudal × tiempo / fracción de llenado), cilindro con la relación H/D del usuario y "
                      "costo de compra con las correlaciones de tanques de BioSTEAM evaluadas por la propia librería.",
            "evidence": {"verified": not reasons, "reasons": reasons, "working_volume_closure_rel": closure,
                         "geometry_closure_rel": geo, "cost_direct_evaluation_rel_diff": direct_rel,
                         "within_correlation_range": in_range}})


# ----------------------------------------------------------------------------------------------------------------------
# multiple-effect evaporator
# ----------------------------------------------------------------------------------------------------------------------
def _water():
    from CoolProp.CoolProp import PropsSI

    def clip(t):
        return min(max(t, 273.17), 640.0)

    def psat(t):
        return PropsSI("P", "T", clip(t), "Q", 0, "Water")

    def tsat(p):
        return PropsSI("T", "P", p, "Q", 0, "Water")

    def hf(t):
        return PropsSI("H", "T", clip(t), "Q", 0, "Water")

    def hg(t):
        return PropsSI("H", "T", clip(t), "Q", 1, "Water")

    def hv(t_sat, t):
        """Vapour at the saturation pressure of t_sat, superheated to t (the boiling-point rise)."""
        if t - t_sat < 1e-7:
            return hg(t_sat)
        return PropsSI("H", "T|gas", clip(t), "P", psat(t_sat), "Water")

    return psat, tsat, hf, hg, hv


def _poly(coefficients, name: str, first_power: int):
    if not isinstance(coefficients, (list, tuple)) or not coefficients or any(
            isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for c in coefficients):
        raise ValueError(f"'{name}' debe ser una lista no vacía de coeficientes numéricos.")
    cs = [float(c) for c in coefficients]
    return lambda x: sum(c * x ** (k + first_power) for k, c in enumerate(cs))


def _bpr_function(spec, n: int):
    """BPR(effect index, solids mass fraction, saturation temperature of the vapour space) in kelvin, from the user's data."""
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError("Falta 'boiling_point_rise' (elevación del punto de ebullición de TU solución; no se estima): da "
                         "exactamente una de {'polynomial_K': [c1, c2, ...]} (EPE = c1·x + c2·x² + …, x = fracción másica de "
                         "sólidos), {'per_effect_K': [...]} (una por efecto) o {'duhring': {'solids_mass_fraction': [...], "
                         "'slope': [...], 'intercept_K': [...]}} (T_ebullición = intercept + slope·T_sat del agua, en K). "
                         "Para una solución sin elevación apreciable pasa {'polynomial_K': [0.0]}.")
    kind, data = next(iter(spec.items()))
    if kind == "polynomial_K":
        f = _poly(data, "boiling_point_rise.polynomial_K", 1)
        return (lambda i, x, ts: f(x)), None, "polinomio en la fracción de sólidos (dato del usuario)"
    if kind == "per_effect_K":
        if not isinstance(data, (list, tuple)) or len(data) != n:
            raise ValueError(f"'boiling_point_rise.per_effect_K' necesita {n} valores (uno por efecto).")
        vals = [_num(v, "boiling_point_rise.per_effect_K", "elevación del punto de ebullición", 0.0, strict=False)
                for v in data]
        return (lambda i, x, ts: vals[i]), None, "valor fijo por efecto (dato del usuario)"
    if kind == "duhring":
        try:
            xs, sl, ic = (list(map(float, data[k])) for k in ("solids_mass_fraction", "slope", "intercept_K"))
        except (KeyError, TypeError, ValueError):
            raise ValueError("'boiling_point_rise.duhring' necesita las listas 'solids_mass_fraction', 'slope' e "
                             "'intercept_K' (rectas de Dühring: T_ebullición = intercept + slope·T_sat del agua, K).") from None
        if not (len(xs) == len(sl) == len(ic) >= 2) or any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("'boiling_point_rise.duhring': las tres listas deben tener la misma longitud (≥ 2) y "
                             "'solids_mass_fraction' debe ser creciente.")
        import numpy as np

        def f(i, x, ts):
            return float(np.interp(x, xs, ic)) + (float(np.interp(x, xs, sl)) - 1.0) * ts
        return f, (xs[0], xs[-1]), "rectas de Dühring del usuario, interpoladas linealmente en la fracción de sólidos"
    raise ValueError(f"'boiling_point_rise': clave '{kind}' desconocida (polynomial_K, per_effect_K o duhring).")


def _evaporator_cost(areas_m2: list[float], costing: dict, year: int, cepci: float | None, warnings: list[str]) -> dict:
    kind = costing.get("type")
    if kind not in NETL_EVAPORATOR_1998:
        raise ValueError(f"costing['type'] debe ser uno de {sorted(NETL_EVAPORATOR_1998)} (tipos de la tabla de costo "
                         "publicada); no se elige por ti.")
    factor = costing.get("material_factor")
    if factor is not None:
        _num(factor, "costing['material_factor']", "factor de material respecto al acero al carbón (tu dato)")
        if factor < 1.0:
            raise ValueError("costing['material_factor'] no puede ser menor que 1 (la base es acero al carbón).")
    f_m = 1.0 if factor is None else float(factor)
    index, basis = _cepci(year, cepci)
    table = NETL_EVAPORATOR_1998[kind]
    la = [math.log(a) for a in table["area_ft2"]]
    lp = [math.log(c) for c in table["purchased_usd"]]
    li = [math.log(c) for c in table["installed_usd"]]

    def loglog(a_ft2, ly):
        x = math.log(a_ft2)
        j = min(max(sum(1 for v in la if v <= x) - 1, 0), len(la) - 2)
        return math.exp(ly[j] + (ly[j + 1] - ly[j]) * (x - la[j]) / (la[j + 1] - la[j]))

    with _biosteam_session() as (bst, caught, n0):
        from biosteam.units._multi_effect_evaporator import evaporators as bst_evaporators
        bst_range, bst_cost, _, bst_material = bst_evaporators[table["biosteam_type"]]
        f_bm = float(bst.MultiEffectEvaporator._F_BM_default["Evaporators"])
        second = [float(bst_cost(a, index)) for a in areas_m2]
        warnings += _bst_warnings(caught, n0)
    rows, equipment = [], []
    lo, hi = table["area_ft2"][0], table["area_ft2"][-1]
    for i, a in enumerate(areas_m2):
        a_ft2 = a / FT2_M2
        inside = lo * (1 - 1e-9) <= a_ft2 <= hi * (1 + 1e-9)
        base = loglog(a_ft2, lp) * index / NETL_CEPCI_1998
        ratio_installed = loglog(a_ft2, li) / loglog(a_ft2, lp)
        if not inside:
            warnings.append(f"Efecto {i + 1}: área {a:.4g} m² ({a_ft2:.4g} ft²) fuera del intervalo de la tabla de costo "
                            f"({lo}-{hi} ft²): costo EXTRAPOLADO con la pendiente log-log del extremo.")
        rows.append({"effect": i + 1, "area_m2": a, "area_ft2": a_ft2, "purchase_cost_usd": base * f_m,
                     "base_purchase_cost_usd": base, "within_table_range": inside,
                     "table_installed_to_purchased_ratio": ratio_installed,
                     "biosteam_second_opinion_usd": second[i], "biosteam_to_table_ratio": second[i] / base})
        equipment.append({"name": f"Evaporador, efecto {i + 1} ({table['label']})", "type": "evaporator",
                          "purchase_cost_usd": base * f_m, "base_purchase_cost_usd": base, "basis": basis,
                          "bare_module_factor": f_bm, "bare_module_factor_source": "BioSTEAM MultiEffectEvaporator"})
    worst = min(r["biosteam_to_table_ratio"] for r in rows)
    if worst < 1 / 3 or max(r["biosteam_to_table_ratio"] for r in rows) > 3:
        warnings.append(f"La función de costo de evaporadores de BioSTEAM ('{table['biosteam_type']}') da entre "
                        f"{worst:.2f} y {max(r['biosteam_to_table_ratio'] for r in rows):.2f} veces el valor de la tabla "
                        "DOE/NETL para la misma área: NO se usa; el costo devuelto es el de la tabla publicada.")
    return {"type": kind, "source": NETL_SOURCE, "table_cepci": NETL_CEPCI_1998, "cepci": index, "basis": basis,
            "cost_year": year if cepci is None else None, "material_factor": f_m,
            "material_factor_source": "acero al carbón (base de la tabla)" if factor is None else "dado por el usuario",
            "effects": rows, "purchase_cost_usd": sum(r["purchase_cost_usd"] for r in rows),
            "base_purchase_cost_usd": sum(r["base_purchase_cost_usd"] for r in rows), "F_BM": f_bm,
            "F_BM_source": "BioSTEAM MultiEffectEvaporator (igual que el tipo 'evaporator' de bare_module_factors)",
            "interpolation": "log-log entre los puntos publicados", "equipment": equipment,
            "biosteam_second_opinion": {"type": table["biosteam_type"], "area_range_m2": list(bst_range),
                                        "material": bst_material, "purchase_cost_usd": sum(second)}}


def multi_effect_evaporator(feed_kg_s: float | None = None, feed_solids_mass_fraction: float | None = None,
                            feed_T_K: float | None = None, product_solids_mass_fraction: float | None = None,
                            n_effects: int | None = None, U_W_m2K: list[float] | None = None,
                            solution_cp_J_kgK: list[float] | None = None, boiling_point_rise: dict | None = None,
                            steam_T_K: float | None = None, steam_P_Pa: float | None = None,
                            last_effect_P_Pa: float | None = None, last_effect_saturation_T_K: float | None = None,
                            equal_area: bool = True, effect_P_Pa: list[float] | None = None, costing: dict | None = None,
                            year: int = 2023, cepci: float | None = None, tolerance: float = 1e-10) -> dict:
    """Forward-feed N-effect evaporator of a solution of a non-volatile solute. Either equal areas (the usual design: the
    intermediate pressures are solved) or the caller's effect pressures (areas follow). See the module docstring."""
    import numpy as np
    from scipy.optimize import root
    F = _num(feed_kg_s, "feed_kg_s", "caudal de alimentación")
    xF = _num(feed_solids_mass_fraction, "feed_solids_mass_fraction", "fracción másica de sólidos de la alimentación")
    xP = _num(product_solids_mass_fraction, "product_solids_mass_fraction", "fracción másica de sólidos del producto")
    TF = _num(feed_T_K, "feed_T_K", "temperatura de la alimentación (entra en el balance de entalpía del primer efecto)")
    if not xF < xP <= 1.0:
        raise ValueError("Se necesita feed_solids_mass_fraction < product_solids_mass_fraction ≤ 1 (se concentra evaporando).")
    if isinstance(n_effects, bool) or not isinstance(n_effects, int) or n_effects < 1:
        raise ValueError("Falta 'n_effects' (número de efectos, entero ≥ 1): es una decisión de diseño y no se supone.")
    N = n_effects
    if not isinstance(U_W_m2K, (list, tuple)) or len(U_W_m2K) != N:
        raise ValueError(f"Falta 'U_W_m2K': una lista con el coeficiente global de transferencia de cada uno de los {N} "
                         "efectos, W/(m²·K). Depende de tu solución y del tipo de evaporador; no se estima.")
    U = [_num(u, "U_W_m2K", "coeficiente global de un efecto") for u in U_W_m2K]
    if solution_cp_J_kgK is None:
        raise ValueError("Falta 'solution_cp_J_kgK': calor específico de TU solución como polinomio en la fracción másica de "
                         "sólidos, [c0, c1, ...] con cp = c0 + c1·x + … en J/(kg·K). No se estima.")
    cp = _poly(solution_cp_J_kgK, "solution_cp_J_kgK", 0)
    bpr, duhring_range, bpr_label = _bpr_function(boiling_point_rise, N)
    psat, tsat, hf, hg, hv = _water()
    if (steam_T_K is None) == (steam_P_Pa is None):
        raise ValueError("Da exactamente uno de 'steam_T_K' o 'steam_P_Pa': el vapor de calefacción saturado del primer "
                         "efecto.")
    Ts = _num(steam_T_K, "steam_T_K", "vapor de calefacción") if steam_T_K is not None else tsat(
        _num(steam_P_Pa, "steam_P_Pa", "vapor de calefacción"))
    fixed_pressures = effect_P_Pa is not None
    if fixed_pressures:
        if not isinstance(effect_P_Pa, (list, tuple)) or len(effect_P_Pa) != N:
            raise ValueError(f"'effect_P_Pa' necesita {N} presiones (una por efecto, Pa).")
        if last_effect_P_Pa is not None or last_effect_saturation_T_K is not None:
            raise ValueError("Con 'effect_P_Pa' la presión del último efecto ya está dada: no pases además "
                             "last_effect_P_Pa ni last_effect_saturation_T_K.")
        tsat_fixed = [tsat(_num(p, "effect_P_Pa", "presión del espacio de vapor de un efecto")) for p in effect_P_Pa]
        tsat_last = tsat_fixed[-1]
    else:
        if (last_effect_P_Pa is None) == (last_effect_saturation_T_K is None):
            raise ValueError("Da exactamente uno de 'last_effect_P_Pa' o 'last_effect_saturation_T_K': el vacío del último "
                             "efecto (lo fija tu condensador) y no se supone.")
        tsat_last = (tsat(_num(last_effect_P_Pa, "last_effect_P_Pa", "presión del último efecto"))
                     if last_effect_P_Pa is not None
                     else _num(last_effect_saturation_T_K, "last_effect_saturation_T_K", "saturación del último efecto"))
        if not equal_area:
            raise ValueError("Con equal_area=False hay que dar 'effect_P_Pa' (la presión de cada efecto): sin áreas iguales "
                             "ni presiones el problema queda indeterminado.")
    if costing is not None and not isinstance(costing, dict):
        raise ValueError("'costing' debe ser un dict, p. ej. {'type': 'vertical_tube'}.")
    LN = F * xF / xP
    lam_ref = hg(Ts) - hf(Ts)
    scale = F * lam_ref

    def h_liq(x, t):
        return cp(x) * (t - T_ZERO_C)

    def state(ts_list, liquids, steam):
        """All effect quantities for given vapour-space saturation temperatures, liquid flows and steam."""
        out = []
        l_prev, x_prev, t_prev = F, xF, TF
        for i in range(N):
            li = liquids[i]
            xi = F * xF / li
            ti = ts_list[i] + bpr(i, xi, ts_list[i])
            vi = l_prev - li
            if i == 0:
                t_heat, q = Ts, steam * lam_ref
            else:
                t_heat = ts_list[i - 1]
                q = out[-1]["V"] * (hv(ts_list[i - 1], out[-1]["T"]) - hf(ts_list[i - 1]))
            residual = l_prev * h_liq(x_prev, t_prev) + q - li * h_liq(xi, ti) - vi * hv(ts_list[i], ti)
            out.append({"L": li, "x": xi, "T": ti, "Tsat": ts_list[i], "V": vi, "q": q, "T_heat": t_heat,
                        "dT": t_heat - ti, "energy_residual_W": residual})
            l_prev, x_prev, t_prev = li, xi, ti
        return out

    def unpack(v):
        if fixed_pressures:
            return tsat_fixed, list(v[:N - 1] * F) + [LN], v[N - 1] * F, None
        return list(v[:N - 1]) + [tsat_last], list(v[N - 1:2 * N - 2] * F) + [LN], v[2 * N - 2] * F, v[2 * N - 1]

    def residuals(v):
        ts_list, liquids, steam, a_scaled = unpack(v)
        st = state(ts_list, liquids, steam)
        res = [s["energy_residual_W"] / scale for s in st]
        if not fixed_pressures:
            res += [(s["q"] - U[i] * a_scaled * a_ref * s["dT"]) / scale for i, s in enumerate(st)]
        return res

    def initial(weights):
        liquids = [F - (F - LN) * (i + 1) / N for i in range(N)]
        rises = [bpr(i, F * xF / liquids[i], tsat_last) for i in range(N)]
        available = Ts - tsat_last - sum(rises)
        if available <= 0:
            raise ValueError(f"No hay fuerza impulsora: T del vapor ({Ts:.2f} K) − saturación del último efecto "
                             f"({tsat_last:.2f} K) − suma de elevaciones del punto de ebullición ({sum(rises):.2f} K) ≤ 0. "
                             "Sube la presión del vapor, baja la del último efecto o usa menos efectos.")
        ts_list, t_heat = [], Ts
        for i in range(N):
            t_boil = t_heat - available * weights[i] / sum(weights)
            ts_list.append(t_boil - rises[i])
            t_heat = ts_list[-1]
        steam = (F - LN) / N * 1.15
        area = steam * lam_ref / (U[0] * available * weights[0] / sum(weights))
        return ts_list, liquids, steam, area

    ts0, l0, s0, a_ref = initial([1.0 / u for u in U])
    if fixed_pressures:
        v0 = np.array([li / F for li in l0[:-1]] + [s0 / F])
    else:
        v0 = np.array(ts0[:-1] + [li / F for li in l0[:-1]] + [s0 / F, 1.0])

    def solve(start):
        best = None
        for method in ("hybr", "lm"):
            try:
                sol = root(residuals, start, method=method, tol=1e-13)
            except Exception as e:  # noqa: BLE001 — reported as non-convergence below
                best = best or (None, method, f"{type(e).__name__}: {str(e)[:160]}")
                continue
            norm = float(np.max(np.abs(residuals(sol.x))))
            if best is None or best[0] is None or norm < best[3]:
                best = (sol.x, method, None, norm)
            if norm <= tolerance:
                break
        return best

    best = solve(v0)
    warnings: list[str] = []
    assumptions = [
        "Alimentación en serie (forward feed): la solución pasa del efecto 1 al N y el vapor de cada efecto calienta el "
        "siguiente; sin extracciones de vapor, sin precalentadores ni flash de condensados.",
        "Entalpía de la solución h = cp(x)·(T − 273.15 K) con el cp del usuario; calor de disolución despreciado.",
        "El vapor que sale de cada efecto está sobrecalentado en la elevación del punto de ebullición y condensa a su "
        "temperatura de saturación en el efecto siguiente; el condensado sale saturado. Propiedades del agua y del vapor: "
        "IAPWS-95 (CoolProp).",
        "Soluto no volátil y sin arrastre; sin pérdidas de calor al ambiente; sin caída de presión entre efectos.",
        f"Elevación del punto de ebullición: {bpr_label}. U de cada efecto: dato del usuario, constante.",
    ]
    not_included = ["Condensador barométrico o de superficie y sistema de vacío (se informa el calor a retirar en el "
                    "condensador, no su equipo).", "Bombas, precalentadores, separadores de arrastre y recompresión de vapor.",
                    "Ensuciamiento, incrustación y variación de U con la concentración o la temperatura.",
                    "Límites de temperatura del producto (degradación, caramelización) y de viscosidad.",
                    "Alimentación en contracorriente, mixta o en paralelo."]
    method = ("Balances de materia, de sólidos y de entalpía de cada efecto más q = U·A·ΔT, resueltos simultáneamente con "
              "scipy.optimize.root" + (" para áreas iguales (incógnitas: presiones intermedias, caudales, vapor y área)."
                                        if not fixed_pressures else " con las presiones de efecto dadas (las áreas resultan)."))
    if best is None or best[0] is None:
        return {"status": "not_converged", "effects": None, "warnings": ["El sistema de ecuaciones no se resolvió."],
                "assumptions": assumptions, "not_included": not_included, "method": method,
                "evidence": {"verified": False, "reasons": [f"scipy.optimize.root falló: {best[2] if best else 'sin dato'}"]}}
    ts_list, liquids, steam, a_scaled = unpack(best[0])
    st = state(ts_list, liquids, steam)
    areas = [s["q"] / (U[i] * s["dT"]) if s["dT"] > 0 else float("nan") for i, s in enumerate(st)]
    reasons = []
    if best[3] > tolerance:
        reasons.append(f"el solver no alcanzó la tolerancia: residuo máximo {best[3]:.2e} (tolerancia {tolerance:.1e})")
    if any(s["dT"] <= 0 for s in st):
        reasons.append("algún efecto queda sin fuerza impulsora (ΔT ≤ 0): la especificación no es operable")
    if any(s["V"] <= 0 for s in st):
        reasons.append("algún efecto no evapora (V ≤ 0): la especificación no es operable con esta alimentación")
    if steam <= 0:
        reasons.append("el consumo de vapor resultó no positivo")
    if duhring_range and any(not duhring_range[0] <= s["x"] <= duhring_range[1] for s in st):
        reasons.append(f"alguna concentración sale del intervalo de las rectas de Dühring ({duhring_range[0]}-"
                       f"{duhring_range[1]}): no se extrapola")
    independence = None
    if not fixed_pressures and not reasons:
        ts1, l1, s1, _ = initial([1.0] * N)                       # a different starting point: equal ΔT, lower steam
        other = solve(np.array(ts1[:-1] + [li / F for li in l1[:-1]] + [0.8 * s1 / F, 1.3]))
        if other is not None and other[0] is not None and other[3] <= tolerance:
            independence = float(abs(other[0][-1] - best[0][-1]) / abs(best[0][-1]))
            if independence > 1e-6:
                reasons.append("dos puntos de partida distintos dan áreas distintas")
        else:
            reasons.append("no se pudo repetir la solución desde otro punto de partida")
    total_v = sum(s["V"] for s in st)
    condensates = sum(st[i]["V"] * hf(st[i]["Tsat"]) for i in range(N - 1))
    # overall: feed + steam latent heat = product + vapour of the last effect + condensates of effects 2..N (the vapour
    # of effect i leaves as saturated liquid at its own saturation temperature)
    overall_rel = abs(F * h_liq(xF, TF) + steam * lam_ref - LN * h_liq(st[-1]["x"], st[-1]["T"])
                      - st[-1]["V"] * hv(st[-1]["Tsat"], st[-1]["T"]) - condensates) / (steam * lam_ref)
    mass_rel = abs(F - LN - total_v) / F
    solids_rel = max(abs(s["L"] * s["x"] - F * xF) for s in st) / (F * xF)
    energy_rel = max(abs(s["energy_residual_W"]) for s in st) / scale
    finite = [a for a in areas if math.isfinite(a)]
    spread = (max(finite) - min(finite)) / (sum(finite) / len(finite)) if finite and not fixed_pressures else None
    if overall_rel > 1e-8:
        reasons.append(f"el balance global de entalpía no cierra ({overall_rel:.2e})")
    if spread is not None and spread > 1e-7:
        reasons.append(f"las áreas no quedaron iguales (dispersión {spread:.2e})")
    effects = [{"effect": i + 1, "pressure_Pa": psat(s["Tsat"]), "vapor_saturation_T_K": s["Tsat"], "boiling_T_K": s["T"],
                "boiling_point_rise_K": s["T"] - s["Tsat"], "heating_medium_T_K": s["T_heat"], "delta_T_K": s["dT"],
                "U_W_m2K": U[i], "duty_W": s["q"], "area_m2": areas[i], "liquid_out_kg_s": s["L"],
                "solids_mass_fraction_out": s["x"], "vapor_kg_s": s["V"]} for i, s in enumerate(st)]
    condenser = st[-1]["V"] * (hv(st[-1]["Tsat"], st[-1]["T"]) - hf(st[-1]["Tsat"]))
    result = {"status": "ok" if not reasons else "not_verified", "n_effects": N, "feed_kg_s": F, "product_kg_s": LN,
              "total_evaporation_kg_s": total_v, "steam_kg_s": steam, "steam_T_K": Ts, "steam_P_Pa": psat(Ts),
              "steam_latent_heat_J_kg": lam_ref, "steam_economy": total_v / steam if steam > 0 else None,
              "specific_steam_kg_per_kg_evaporated": steam / total_v, "effects": effects,
              "total_area_m2": sum(finite) if len(finite) == N else None,
              "mean_area_m2": sum(finite) / N if len(finite) == N else None, "equal_area": not fixed_pressures,
              "condenser_duty_W": condenser, "cost": None, "equipment": [], "warnings": warnings,
              "assumptions": assumptions, "not_included": not_included, "method": method + f" Solver: {best[1]}."}
    if costing is not None:
        if reasons:
            warnings.append("No se calcula el costo porque el balance no quedó verificado.")
        else:
            result["cost"] = _evaporator_cost(areas, costing, year, cepci, warnings)
            result["equipment"] = result["cost"].pop("equipment")
            assumptions.append("Costo de COMPRA de cada cuerpo por interpolación log-log en la tabla publicada DOE/NETL "
                               "(acero al carbón, 1998), llevado al CEPCI pedido; el factor de material, si no es 1, es el "
                               "del usuario.")
    else:
        not_included.append("Costo: no se pidió (falta costing = {'type': 'vertical_tube' | 'horizontal_tube'}).")
    if reasons:
        warnings.append("Resultado NO verificado: " + "; ".join(reasons) + ".")
    result["evidence"] = {"verified": not reasons, "reasons": reasons, "solver_max_residual_rel": best[3],
                          "mass_balance_rel": mass_rel, "solids_balance_rel": solids_rel,
                          "effect_enthalpy_residual_rel": energy_rel, "overall_enthalpy_balance_rel": overall_rel,
                          "equal_area_spread_rel": spread, "initial_guess_independence_rel": independence,
                          "min_delta_T_K": min(s["dT"] for s in st)}
    return _clean(result)


# ----------------------------------------------------------------------------------------------------------------------
# gravity settler and separator cost
# ----------------------------------------------------------------------------------------------------------------------
def _segment_area(h: float, d: float) -> float:
    r = d / 2.0
    h = min(max(h, 0.0), d)
    return r * r * math.acos((r - h) / r) - (r - h) * math.sqrt(max(2.0 * r * h - h * h, 0.0))


def _level_for_area_fraction(fraction: float, d: float) -> float:
    from scipy.optimize import brentq
    if fraction >= 1.0:
        return d
    target = fraction * math.pi * d * d / 4.0
    return brentq(lambda h: _segment_area(h, d) - target, 0.0, d, xtol=1e-14, rtol=1e-14)


def size_gravity_settler(q_light_m3_s: float, q_heavy_m3_s: float, rho_light: float, rho_heavy: float, sizing: dict) -> dict:
    """Gravity decanter. Vessel from the hold-up time; settling by Stokes with the user's drop size and the plug-flow
    criterion u_c = Q_c / A_i <= u_d; the larger diameter governs. Every design datum comes from `sizing`."""
    if not isinstance(sizing, dict):
        raise ValueError("'sizing' debe ser un dict con los datos de diseño del decantador.")
    orientation = sizing.get("orientation")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("Falta sizing['orientation']: 'horizontal' o 'vertical' (decisión de diseño, no se supone).")
    d_drop = _num(sizing.get("droplet_diameter_m"), "sizing['droplet_diameter_m']",
                  "diámetro de gota de la fase dispersa para la ley de Stokes (dato tuyo; ver published_typical_values)")
    tau = _num(sizing.get("residence_time_s"), "sizing['residence_time_s']", "tiempo de retención total del líquido")
    ld = _num(sizing.get("length_to_diameter"), "sizing['length_to_diameter']",
              "relación largo/diámetro (altura/diámetro si es vertical)")
    fill = _num(sizing.get("fill_fraction"), "sizing['fill_fraction']", "fracción del volumen del recipiente ocupada por líquido")
    fi = _num(sizing.get("interface_height_fraction"), "sizing['interface_height_fraction']",
              "altura de la interfase / diámetro (horizontal) o / altura (vertical); la fijan las piernas de rebose")
    continuous = sizing.get("continuous_phase")
    if continuous not in ("light", "heavy"):
        raise ValueError("Falta sizing['continuous_phase']: 'light' o 'heavy' (cuál es la fase continua; la otra va en gotas).")
    mu_c = _num(sizing.get("continuous_viscosity_Pa_s"), "sizing['continuous_viscosity_Pa_s']",
                "viscosidad de la fase continua para la ley de Stokes (no se estima)")
    mu_d = sizing.get("dispersed_viscosity_Pa_s")
    if mu_d is not None:
        mu_d = _num(mu_d, "sizing['dispersed_viscosity_Pa_s']", "viscosidad de la otra fase (comprobación de arrastre)")
    if fill > 1.0 or not 0.0 < fi < 1.0:
        raise ValueError("sizing: 'fill_fraction' debe estar en (0, 1] e 'interface_height_fraction' en (0, 1).")
    d_rho = rho_heavy - rho_light
    if d_rho <= 0:
        raise ValueError("Las densidades de las dos fases coinciden o están invertidas: sin diferencia de densidad no hay "
                         "decantación por gravedad.")
    q_c, q_d = (q_light_m3_s, q_heavy_m3_s) if continuous == "light" else (q_heavy_m3_s, q_light_m3_s)
    rho_c = rho_light if continuous == "light" else rho_heavy
    q_tot = q_light_m3_s + q_heavy_m3_s
    u_t = G * d_drop ** 2 * d_rho / (18.0 * mu_c)
    reynolds = rho_c * u_t * d_drop / mu_c
    area_required = q_c / u_t
    volume_tau = q_tot * tau / fill
    if orientation == "horizontal":
        level_fraction = _level_for_area_fraction(fill, 1.0)             # liquid level / D for this fill fraction
        if fi >= level_fraction:
            raise ValueError(f"sizing: la interfase ({fi:.3f}·D) queda por encima del nivel de líquido ({level_fraction:.3f}"
                             f"·D para fill_fraction = {fill}).")
        width_factor = 2.0 * math.sqrt(fi - fi * fi)                     # interface width / D
        d_tau = (4.0 * volume_tau / (math.pi * ld)) ** (1.0 / 3.0)
        d_settle = math.sqrt(area_required / (ld * width_factor))
    else:
        if fi >= fill:
            raise ValueError("sizing: la interfase queda por encima del nivel de líquido (interface_height_fraction ≥ "
                             "fill_fraction).")
        d_tau = (4.0 * volume_tau / (math.pi * ld)) ** (1.0 / 3.0)
        d_settle = math.sqrt(4.0 * area_required / math.pi)
    d = max(d_tau, d_settle)
    length = ld * d
    volume = math.pi * d * d * length / 4.0
    if orientation == "horizontal":
        interface_area = width_factor * d * length
        a_heavy = _segment_area(fi * d, d)
        a_liquid = fill * math.pi * d * d / 4.0
        v_heavy, v_light = a_heavy * length, (a_liquid - a_heavy) * length
        depth_heavy, depth_light = fi * d, (level_fraction - fi) * d
    else:
        interface_area = math.pi * d * d / 4.0
        v_heavy, v_light = interface_area * fi * length, interface_area * (fill - fi) * length
        depth_heavy, depth_light = fi * length, (fill - fi) * length
    u_c = q_c / interface_area
    d_min = math.sqrt(18.0 * mu_c * u_c / (G * d_rho))
    out = {"orientation": orientation, "diameter_m": d, "length_m": length, "vessel_volume_m3": volume,
           "governed_by": "settling" if d_settle > d_tau else "residence_time",
           "diameter_from_residence_time_m": d_tau, "diameter_from_settling_m": d_settle,
           "liquid_residence_time_s": fill * volume / q_tot, "interface_area_m2": interface_area,
           "interface_area_required_m2": area_required, "continuous_phase": continuous,
           "stokes_settling_velocity_m_s": u_t, "droplet_reynolds": reynolds, "stokes_regime": bool(reynolds <= 1.0),
           "continuous_phase_velocity_m_s": u_c, "settling_margin": u_t / u_c, "droplets_settle": bool(u_t >= u_c * (1 - 1e-12)),
           "smallest_droplet_separated_m": d_min, "density_difference_kg_m3": d_rho,
           "heavy_layer": {"volume_m3": v_heavy, "depth_m": depth_heavy,
                           "residence_time_s": v_heavy / q_heavy_m3_s if q_heavy_m3_s > 0 else None},
           "light_layer": {"volume_m3": v_light, "depth_m": depth_light,
                           "residence_time_s": v_light / q_light_m3_s if q_light_m3_s > 0 else None},
           "volumetric_flow_light_m3_s": q_light_m3_s, "volumetric_flow_heavy_m3_s": q_heavy_m3_s,
           "published_typical_values": DECANTER_TYPICAL_VALUES, "warnings": [], "entrainment_check": None}
    if mu_d is not None:
        u_other = q_d / interface_area
        out["entrainment_check"] = {
            "dispersed_source_phase_velocity_m_s": u_other,
            "largest_continuous_droplet_carried_m": math.sqrt(18.0 * mu_d * u_other / (G * d_rho)),
            "note": "gotas de la fase continua que la otra fase podría arrastrar (misma ley de Stokes con su viscosidad)"}
    if reynolds > 1.0:
        out["warnings"].append(f"Reynolds de la gota = {reynolds:.2f} > 1: fuera del régimen de Stokes; la velocidad real es "
                               "menor que la calculada y el decantador queda corto.")
    if u_t > DECANTER_TYPICAL_VALUES["max_settling_velocity_m_s"]["value"]:
        out["warnings"].append(f"Stokes da {u_t:.2e} m/s, por encima de 4e-3 m/s: la regla publicada (ver "
                               "published_typical_values) es diseñar con 4e-3 m/s; aquí NO se recorta.")
    return out


def _separator_cost(costing: dict, sizing_result: dict | None, q_total_m3_s: float, n_units: int, year: int,
                    cepci: float | None, warnings: list[str]) -> tuple[dict, list[dict]]:
    if not isinstance(costing, dict) or costing.get("equipment") not in ("gravity_decanter", "centrifuge"):
        raise ValueError("costing['equipment'] debe ser 'gravity_decanter' (recipiente de Seider con el dimensionamiento de "
                         "'sizing') o 'centrifuge' (centrífuga líquido-líquido, correlación de BioSTEAM).")
    index, basis = _cepci(year, cepci)
    if costing["equipment"] == "gravity_decanter":
        if sizing_result is None:
            raise ValueError("Para costear un decantador por gravedad hace falta 'sizing' (da su diámetro y su longitud).")
        material = costing.get("material")
        if not isinstance(material, str):
            raise ValueError("Falta costing['material'] (nombre de material de recipientes de IDAES, p. ej. 'CarbonSteel', "
                             "'StainlessSteel304', 'StainlessSteel316'): no se elige por ti.")
        vertical = sizing_result["orientation"] == "vertical"
        c = _vessel_cost(sizing_result["diameter_m"], sizing_result["length_m"], vertical, material, year, cepci)
        weight = c["shell_weight_lb"]
        lo, hi = (4200.0, 1.0e6) if vertical else (1000.0, 9.2e5)
        in_range = bool(weight is not None and lo <= weight <= hi)
        if not in_range:
            warnings.append(f"Peso del casco {weight:.4g} lb fuera del intervalo de la correlación de Seider ({lo:g}-{hi:g} "
                            "lb): costo extrapolado.")
        kind = "vessel_vertical" if vertical else "vessel_horizontal"
        block = {"equipment": "gravity_decanter", "correlation": c["method"], "material": material,
                 "unit_purchase_cost_usd": c["purchase_cost_usd"], "unit_base_purchase_cost_usd": c["base_purchase_cost_usd"],
                 "number_of_units": n_units, "purchase_cost_usd": c["purchase_cost_usd"] * n_units,
                 "base_purchase_cost_usd": c["base_purchase_cost_usd"] * n_units, "shell_weight_lb": weight,
                 "within_correlation_range": in_range, "cepci": index, "basis": c["basis"],
                 "cost_year": year if cepci is None else None,
                 "note": "recipiente a presión de Seider (equipment_cost 'vessel' del repositorio) con internos no incluidos"}
        item = {"name": f"Decantador por gravedad ({material}) ×{n_units}", "type": kind,
                "purchase_cost_usd": block["purchase_cost_usd"], "base_purchase_cost_usd": block["base_purchase_cost_usd"],
                "platforms_ladders_purchase_cost_usd": c["platforms_ladders_purchase_cost_usd"] * n_units,
                "basis": c["basis"], "number_of_units": n_units}
        return block, [item]
    q_h = q_total_m3_s * 3600.0
    with _biosteam_session() as (bst, caught, n0):
        from biosteam.units.liquid_liquid_extraction import LiquidsCentrifuge
        item_bst = LiquidsCentrifuge.cost_items["Liquids centrifuge"]
        ub, ce, c0, s0, n_exp, kw = (float(item_bst.ub), float(item_bst.CE), float(item_bst.cost), float(item_bst.S),
                                     float(item_bst.n), float(item_bst.kW))
        f_bm = float(LiquidsCentrifuge._F_BM_default["Liquids centrifuge"])
        warnings += _bst_warnings(caught, n0)
    parallel = max(1, math.ceil(q_h / ub - 1e-12))
    unit = c0 * (q_h / parallel / s0) ** n_exp * index / ce
    total_units = parallel * n_units
    if parallel > 1:
        warnings.append(f"Caudal {q_h:.4g} m³/h por encima del máximo de la correlación ({ub:g} m³/h por centrífuga): "
                        f"{parallel} centrífugas en paralelo por etapa, como hace BioSTEAM.")
    block = {"equipment": "centrifuge", "correlation": f"C = {c0:g}·Q^{n_exp:g} USD (Q en m³/h, máximo {ub:g} m³/h, CEPCI "
             f"{ce:g}); BioSTEAM LiquidsCentrifuge, de Apostolakou et al. (2009)", "flow_m3_h": q_h,
             "parallel_units_per_stage": parallel, "number_of_units": total_units, "unit_purchase_cost_usd": unit,
             "unit_base_purchase_cost_usd": unit, "purchase_cost_usd": unit * total_units,
             "base_purchase_cost_usd": unit * total_units, "cepci": index, "basis": basis,
             "cost_year": year if cepci is None else None, "F_BM": f_bm, "F_BM_source": "BioSTEAM LiquidsCentrifuge",
             "power_W": kw * q_h * 1000.0 * n_units, "power_source": f"BioSTEAM: {kw:g} kW por m³/h",
             "within_correlation_range": bool(q_h / parallel <= ub),
             "note": "la correlación no distingue material; sin límite inferior de caudal declarado"}
    item = {"name": f"Centrífuga líquido-líquido ×{total_units}", "F_BM": f_bm, "F_BM_source": "BioSTEAM LiquidsCentrifuge",
            "purchase_cost_usd": unit * total_units, "base_purchase_cost_usd": unit * total_units, "basis": basis,
            "number_of_units": total_units}
    return block, [item]


# ----------------------------------------------------------------------------------------------------------------------
# liquid-liquid extraction
# ----------------------------------------------------------------------------------------------------------------------
def _stream(flows, name: str, required: bool) -> dict[str, float]:
    if flows is None:
        if required:
            raise ValueError(f"Falta '{name}': dict {{componente: caudal}} de esa corriente.")
        return {}
    if not isinstance(flows, dict) or not flows:
        raise ValueError(f"'{name}' debe ser un dict no vacío {{componente: caudal}}.")
    out = {}
    for k, v in flows.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"'{name}': el caudal de '{k}' debe ser un número ≥ 0.")
        out[str(k)] = float(v)
    if sum(out.values()) <= 0:
        raise ValueError(f"'{name}' no tiene caudal.")
    return out


def kremser_raffinate_ratio(x0: float, x_star: float, extraction_factor: float, n: int) -> float:
    """Kremser: X_N = X* + (X_0 − X*)·(E − 1)/(E^(N+1) − 1) for N ideal countercurrent stages, immiscible phases and a
    constant distribution coefficient (X* = solute ratio in equilibrium with the entering solvent, E = K·S/F)."""
    e = extraction_factor
    if abs(e - 1.0) < 1e-12:
        frac = 1.0 / (n + 1)
    elif e > 1.0:
        frac = (1.0 - 1.0 / e) / (e ** n - 1.0 / e)
    else:
        frac = (e - 1.0) / (e ** (n + 1) - 1.0)
    return x_star + (x0 - x_star) * frac


def kremser_stages(x0: float, xn: float, x_star: float, extraction_factor: float) -> dict:
    """Ideal stages for a target raffinate ratio; says when the target cannot be reached."""
    e = extraction_factor
    if not x0 > x_star:
        return {"reachable": False, "reason": "el disolvente entra más cargado que el equilibrio con la alimentación"}
    ratio = (xn - x_star) / (x0 - x_star)
    if ratio <= 0:
        return {"reachable": False, "reason": "el objetivo está por debajo del equilibrio con el disolvente que entra"}
    if ratio >= 1:
        return {"reachable": True, "stages": 0.0}
    if abs(e - 1.0) < 1e-12:
        return {"reachable": True, "stages": 1.0 / ratio - 1.0}
    if e < 1.0 and ratio <= 1.0 - e:
        return {"reachable": False, "reason": f"con factor de extracción {e:.4g} < 1 la fracción no extraída no baja de "
                                              f"{1.0 - e:.4g} ni con infinitas etapas"}
    return {"reachable": True, "stages": math.log((1.0 / ratio) * (1.0 - 1.0 / e) + 1.0 / e) / math.log(e)}


def _lle_constant(mode, feed, solvent, flow_basis, coefficients, carrier, extraction_solvent, n_stages, target):
    import numpy as np
    unit = "kg/s" if flow_basis == "kg_s" else "mol/s"
    if not isinstance(carrier, str) or feed.get(carrier, 0.0) <= 0:
        raise ValueError("Falta 'carrier': el componente de la alimentación que NO pasa al disolvente (diluyente), con "
                         "caudal > 0 en 'feed'. Hace falta para el modelo de fases inmiscibles.")
    if not isinstance(extraction_solvent, str) or extraction_solvent == carrier:
        raise ValueError("Falta 'extraction_solvent': el componente disolvente, distinto de 'carrier', que no pasa al refinado.")
    fc = feed[carrier] + (solvent.get(carrier, 0.0) if mode == "decanter" else 0.0)
    s = solvent.get(extraction_solvent, 0.0) + (feed.get(extraction_solvent, 0.0) if mode == "decanter" else 0.0)
    if mode == "countercurrent" and (solvent.get(carrier, 0.0) > 0 or feed.get(extraction_solvent, 0.0) > 0):
        raise ValueError("Contracorriente con fases inmiscibles: 'solvent' no puede llevar 'carrier' ni 'feed' llevar "
                         "'extraction_solvent' (cada uno sale íntegro por su extremo).")
    if s <= 0:
        raise ValueError(f"No hay caudal de '{extraction_solvent}' en la corriente de disolvente.")
    names = list(dict.fromkeys(list(feed) + list(solvent)))
    solutes = [c for c in names if c not in (carrier, extraction_solvent)]
    if not isinstance(coefficients, dict):
        raise ValueError("'distribution_coefficients' debe ser un dict {soluto: K}.")
    missing = [c for c in solutes if c not in coefficients]
    if missing:
        raise ValueError(f"Falta el coeficiente de distribución de {missing}: K = (soluto por unidad de disolvente en el "
                         "extracto) / (soluto por unidad de diluyente en el refinado), en las unidades de los caudales. Es "
                         "un dato de equilibrio de tu sistema; no se estima.")
    N = 1 if mode == "decanter" else n_stages
    profiles = [{"stage": j + 1, "raffinate_ratio": {}, "extract_ratio": {}} for j in range(N)]
    raffinate, extract = {carrier: fc}, {extraction_solvent: s}
    detail, worst_kremser, worst_stage, worst_balance = {}, 0.0, 0.0, 0.0
    for c in solutes:
        k = _num(coefficients[c], f"distribution_coefficients['{c}']", "coeficiente de distribución", 0.0, strict=False)
        f_in, s_in = feed.get(c, 0.0), solvent.get(c, 0.0)
        a = np.zeros((N, N))
        b = np.zeros(N)
        for j in range(N):
            a[j, j] = -(fc + s * k)
            if j > 0:
                a[j, j - 1] = fc
            if j < N - 1:
                a[j, j + 1] = s * k
        b[0] -= f_in
        b[N - 1] -= s_in
        x = np.linalg.solve(a, b)
        y = k * x
        for j in range(N):
            profiles[j]["raffinate_ratio"][c] = float(x[j])
            profiles[j]["extract_ratio"][c] = float(y[j])
            inflow = (f_in if j == 0 else fc * x[j - 1]) + (s_in if j == N - 1 else s * y[j + 1])
            worst_stage = max(worst_stage, abs(inflow - fc * x[j] - s * y[j]) / max(f_in + s_in, 1e-300))
        raffinate[c], extract[c] = float(fc * x[-1]), float(s * y[0])
        worst_balance = max(worst_balance, abs(f_in + s_in - raffinate[c] - extract[c]) / max(f_in + s_in, 1e-300))
        e = k * s / fc
        row = {"distribution_coefficient": k, "extraction_factor": e, "feed_ratio": f_in / fc,
               "raffinate_ratio": float(x[-1]), "extract_ratio": float(y[0]),
               "fraction_extracted": extract[c] / (f_in + s_in) if f_in + s_in > 0 else None}
        if k > 0:
            x_star = s_in / (s * k)
            closed = kremser_raffinate_ratio(f_in / fc, x_star, e, N)
            rel = abs(closed - float(x[-1])) / max(abs(closed), abs(f_in / fc), 1e-300)
            row["kremser"] = {"raffinate_ratio": closed, "rel_diff": rel, "equilibrium_ratio_with_entering_solvent": x_star}
            worst_kremser = max(worst_kremser, rel)
        detail[c] = row
    required = None
    if target is not None:
        if not isinstance(target, dict) or target.get("solute") not in solutes or target["solute"] not in detail:
            raise ValueError(f"'target' debe ser {{'solute': uno de {solutes}, 'raffinate_ratio': X_N}}.")
        xn = _num(target.get("raffinate_ratio"), "target['raffinate_ratio']",
                  f"relación objetivo del soluto en el refinado ({unit} de soluto por {unit} de diluyente)")
        row = detail[target["solute"]]
        if row["distribution_coefficient"] <= 0:
            required = {"reachable": False, "reason": "K = 0: el soluto no pasa al disolvente"}
        else:
            required = kremser_stages(row["feed_ratio"], xn, row["kremser"]["equilibrium_ratio_with_entering_solvent"],
                                      row["extraction_factor"])
        required |= {"solute": target["solute"], "raffinate_ratio_target": xn, "method": "ecuación de Kremser"}
        if required.get("reachable"):
            required["ideal_stages_integer"] = int(math.ceil(required["stages"] - 1e-9))
    reasons = []
    if worst_balance > 1e-9 or worst_stage > 1e-9:
        reasons.append("los balances de etapa no cierran")
    if worst_kremser > 1e-8:
        reasons.append("la solución numérica no coincide con la ecuación de Kremser")
    return {"basis": "constant_distribution_coefficients", "flow_unit": unit, "n_stages": N, "carrier": carrier,
            "extraction_solvent": extraction_solvent,
            "raffinate": {"flows": raffinate, "total": sum(raffinate.values())},
            "extract": {"flows": extract, "total": sum(extract.values())}, "solutes": detail, "stage_profiles": profiles,
            "stages_required": required,
            "assumptions": [
                f"Diluyente ('{carrier}') y disolvente ('{extraction_solvent}') totalmente inmiscibles: cada uno sale íntegro "
                "por su fase.",
                "Coeficiente de distribución constante (dato del usuario), definido en relaciones libres de soluto: Y = K·X "
                f"con X = {unit} de soluto por {unit} de diluyente e Y por {unit} de disolvente. En disoluciones diluidas "
                "coincide con el cociente de fracciones.",
                "Etapas ideales (equilibrio en cada una), isotermas; los solutos no interactúan entre sí."],
            "method": "Balances de soluto por etapa (sistema lineal tridiagonal, numpy.linalg.solve) contrastados con la "
                      "ecuación de Kremser.",
            "evidence": {"verified": not reasons, "reasons": reasons, "solute_balance_closure_rel": worst_balance,
                         "stage_balance_residual_rel": worst_stage, "kremser_rel_diff": worst_kremser}}


def _molar_masses(names: list[str]) -> dict[str, float]:
    from packages.properties.pure import identify
    return {n: float(identify(n)[0].MW) / 1000.0 for n in names}


def _phase(flows_mol: dict, mw: dict, rho) -> dict:
    total = sum(flows_mol.values())
    kg = {c: v * mw[c] for c, v in flows_mol.items()}
    total_kg = sum(kg.values())
    return {"flows_mol_s": dict(flows_mol), "flows_kg_s": kg, "total_mol_s": total, "total_kg_s": total_kg,
            "mole_fractions": {c: v / total for c, v in flows_mol.items()} if total > 0 else None,
            "mass_fractions": {c: v / total_kg for c, v in kg.items()} if total_kg > 0 else None,
            "density_kg_m3": rho, "volumetric_flow_m3_s": total_kg / rho if rho else None}


def _two_liquid_flash(names, flows_mol, model, t, p, activity_parameters) -> dict:
    """One isothermal flash of the repo engine, reduced to what a stage needs. Never raises for a physical outcome."""
    from packages.properties import flash as FL
    total = sum(flows_mol[c] for c in names)
    z = [max(flows_mol[c], 0.0) / total for c in names]
    z = [v / sum(z) for v in z]
    r = FL.independent_flash(names, z, model, T_K=t, P_Pa=p, activity_parameters=activity_parameters)
    out = {"status": r.get("status"), "verified": bool(r["evidence"].get("verified")), "reasons": r["evidence"].get("reasons", []),
           "warnings": r.get("warnings", []), "method": r.get("method"), "binary_parameters": r.get("binary_parameters"),
           "pure_component_data": r.get("pure_component_data"), "phases": []}
    if r.get("status") != "ok":
        out["kind"] = "not_converged"
        return out
    rows = r["phases"]
    if any(row["type"] != "liquid" for row in rows):
        out["kind"] = "vapor_present"
    else:
        out["kind"] = "two_liquids" if len(rows) == 2 else "single_liquid"
    for row in sorted(rows, key=lambda q: q["rho_kg_m3"]):
        out["phases"].append({"flows": {c: row["mole_fraction_of_feed"] * total * row["mole_fractions"].get(c, 0.0)
                                        for c in names}, "rho": row["rho_kg_m3"], "type": row["type"]})
    return out


def _lle_thermo(mode, feed, solvent, flow_basis, model, activity_parameters, t, p, extraction_solvent, n_stages, max_sweeps,
                tolerance):
    names = list(dict.fromkeys(list(feed) + list(solvent)))
    mw = _molar_masses(names)
    conv = (lambda d: {c: d.get(c, 0.0) / mw[c] for c in names}) if flow_basis == "kg_s" else (
        lambda d: {c: d.get(c, 0.0) for c in names})
    f_mol, s_mol = conv(feed), conv(solvent)
    total_in = sum(f_mol.values()) + sum(s_mol.values())
    warnings: list[str] = []
    assumptions = ["Cada etapa es una etapa ideal: las dos fases líquidas salen en equilibrio a T y P dadas (isoterma).",
                   "Equilibrio líquido-líquido calculado con el motor de flash del repositorio (thermo) y el modelo elegido; "
                   "las solubilidades mutuas dependen por completo del modelo y de sus parámetros.",
                   "Densidades de las fases: estimación de thermo para la mezcla líquida (no son datos medidos)."]
    key = extraction_solvent
    if solvent and key is None:
        key = max(s_mol, key=s_mol.get)
        assumptions.append(f"Fase extracto = la más rica en '{key}' (componente mayoritario de la corriente de disolvente).")
    if key is not None and key not in names:
        raise ValueError(f"'extraction_solvent' = '{key}' no está entre los componentes {names}.")
    if mode == "decanter":
        mix = {c: f_mol[c] + s_mol[c] for c in names}
        fl = _two_liquid_flash(names, mix, model, t, p, activity_parameters)
        base = {"basis": "thermodynamic_model", "model": model, "T_K": t, "P_Pa": p, "n_stages": 1,
                "binary_parameters": fl["binary_parameters"], "molar_masses_kg_mol": mw}
        method = "Un flash isotérmico de dos líquidos. " + (fl["method"] or "")
        warnings += fl["warnings"]
        if fl["kind"] != "two_liquids":
            text = {"single_liquid": "la mezcla NO se separa en dos líquidos a esta T y P con este modelo",
                    "vapor_present": "aparece fase vapor a esta T y P: no es una decantación líquido-líquido",
                    "not_converged": "el flash no convergió"}[fl["kind"]]
            warnings.append("Sin separación: " + text + ".")
            return base | {"status": fl["kind"], "light_phase": None, "heavy_phase": None, "warnings": warnings,
                           "assumptions": assumptions, "method": method, "_phases": None,
                           "evidence": {"verified": False, "reasons": [text] + list(fl["reasons"]),
                                        "flash_verified": fl["verified"]}}
        light, heavy = (_phase(ph["flows"], mw, ph["rho"]) for ph in fl["phases"])
        closure = max(abs(mix[c] - light["flows_mol_s"][c] - heavy["flows_mol_s"][c]) for c in names) / total_in
        split = {c: (light["flows_mol_s"][c] / mix[c] if mix[c] > 0 else None) for c in names}
        k_mole = {c: (light["mole_fractions"][c] / heavy["mole_fractions"][c] if heavy["mole_fractions"][c] > 0 else None)
                  for c in names}
        k_mass = {c: (light["mass_fractions"][c] / heavy["mass_fractions"][c] if heavy["mass_fractions"][c] > 0 else None)
                  for c in names}
        roles = None
        if key is not None:
            ext = "light" if light["mole_fractions"][key] > heavy["mole_fractions"][key] else "heavy"
            roles = {"extract": ext, "raffinate": "heavy" if ext == "light" else "light", "solvent_key": key}
        reasons = []
        if not fl["verified"]:
            reasons.append("el flash no quedó verificado: " + " ".join(fl["reasons"]))
        if closure > 1e-8:
            reasons.append("el balance de materia de las dos fases no cierra")
        return base | {"status": "ok", "light_phase": light, "heavy_phase": heavy, "split_to_light_phase": split,
                       "K_light_over_heavy_mole": k_mole, "K_light_over_heavy_mass": k_mass, "roles": roles,
                       "warnings": warnings, "assumptions": assumptions, "method": method,
                       "_phases": [(light, heavy)],
                       "evidence": {"verified": not reasons, "reasons": reasons, "flash_verified": fl["verified"],
                                    "component_balance_closure_rel": closure}}
    # countercurrent cascade: raffinate flows 1 -> N, extract N -> 1
    N = n_stages
    raff = [dict(f_mol) for _ in range(N)]
    extr = [dict(s_mol) for _ in range(N)]
    last = [None] * N
    change, sweeps, failure = math.inf, 0, None
    z_feed, z_solv = f_mol[key] / sum(f_mol.values()), s_mol[key] / sum(s_mol.values())
    while sweeps < max_sweeps and change > tolerance:
        sweeps += 1
        change = 0.0
        for j in range(N):
            r_in = f_mol if j == 0 else raff[j - 1]
            e_in = s_mol if j == N - 1 else extr[j + 1]
            mix = {c: r_in[c] + e_in[c] for c in names}
            fl = _two_liquid_flash(names, mix, model, t, p, activity_parameters)
            last[j] = fl
            if fl["kind"] == "two_liquids":
                a, b = fl["phases"]
                xa, xb = a["flows"][key] / sum(a["flows"].values()), b["flows"][key] / sum(b["flows"].values())
                new_e, new_r = (a, b) if xa > xb else (b, a)
                new_e, new_r = new_e["flows"], new_r["flows"]
            elif fl["kind"] == "single_liquid":
                zk = mix[key] / sum(mix.values())
                empty = {c: 0.0 for c in names}
                new_e, new_r = (mix, empty) if abs(zk - z_solv) < abs(zk - z_feed) else (empty, mix)
            else:
                failure = (j + 1, fl["kind"], fl["reasons"])
                break
            change = max(change, max(abs(new_r[c] - raff[j][c]) + abs(new_e[c] - extr[j][c]) for c in names) / total_in)
            raff[j], extr[j] = new_r, new_e
        if failure:
            break
    base = {"basis": "thermodynamic_model", "model": model, "T_K": t, "P_Pa": p, "n_stages": N, "solvent_key": key,
            "binary_parameters": last[0]["binary_parameters"] if last[0] else None, "molar_masses_kg_mol": mw}
    method = (f"Cascada de {N} etapas ideales en contracorriente por sustitución sucesiva etapa a etapa: cada etapa es un "
              "flash isotérmico de dos líquidos del motor del repositorio. " + ((last[0]["method"] or "") if last[0] else ""))
    if failure:
        text = (f"etapa {failure[0]}: " + {"vapor_present": "aparece fase vapor a esta T y P",
                                           "not_converged": "el flash no convergió"}[failure[1]])
        warnings.append("Cascada no resuelta: " + text + ".")
        return base | {"status": failure[1], "raffinate": None, "extract": None, "stages": None, "warnings": warnings,
                       "assumptions": assumptions, "method": method, "_phases": None,
                       "evidence": {"verified": False, "reasons": [text] + list(failure[2]), "sweeps": sweeps}}
    stages, pairs, residual = [], [], 0.0
    rho = []
    for j in range(N):
        fl = last[j]
        if fl["kind"] == "two_liquids":
            a, b = fl["phases"]
            xa, xb = a["flows"][key] / sum(a["flows"].values()), b["flows"][key] / sum(b["flows"].values())
            rho_e, rho_r = (a["rho"], b["rho"]) if xa > xb else (b["rho"], a["rho"])
        else:
            rho_e = rho_r = fl["phases"][0]["rho"] if fl["phases"] else None
        rho.append((rho_r, rho_e))
        r_in = f_mol if j == 0 else raff[j - 1]
        e_in = s_mol if j == N - 1 else extr[j + 1]
        residual = max(residual, max(abs(r_in[c] + e_in[c] - raff[j][c] - extr[j][c]) for c in names) / total_in)
        pr, pe = _phase(raff[j], mw, rho_r), _phase(extr[j], mw, rho_e)
        stages.append({"stage": j + 1, "two_liquid_phases": fl["kind"] == "two_liquids", "raffinate": pr, "extract": pe,
                       "flash_verified": fl["verified"]})
        if fl["kind"] == "two_liquids":
            pairs.append((pr, pe) if rho_r <= rho_e else (pe, pr))
    overall = max(abs(f_mol[c] + s_mol[c] - raff[-1][c] - extr[0][c]) for c in names) / total_in
    recovery = {c: (extr[0][c] / (f_mol[c] + s_mol[c]) if f_mol[c] + s_mol[c] > 0 else None) for c in names}
    reasons = []
    if change > tolerance:
        reasons.append(f"la sustitución sucesiva no convergió en {sweeps} barridos (último cambio {change:.2e})")
    single = [s["stage"] for s in stages if not s["two_liquid_phases"]]
    if single:
        reasons.append(f"en las etapas {single} la mezcla no se separa en dos líquidos: la cascada no es operable así")
    if any(not s["flash_verified"] for s in stages):
        reasons.append("el flash de alguna etapa no quedó verificado")
    if residual > 1e-7 or overall > 1e-7:
        reasons.append(f"los balances no cierran (etapa {residual:.2e}, global {overall:.2e})")
    for w in last[0]["warnings"]:
        if w not in warnings:
            warnings.append(w)
    return base | {"status": "ok" if not reasons else "not_verified", "raffinate": stages[-1]["raffinate"],
                   "extract": stages[0]["extract"], "recovery_to_extract": recovery, "stages": stages, "warnings": warnings,
                   "assumptions": assumptions, "method": method, "_phases": pairs or None,
                   "evidence": {"verified": not reasons, "reasons": reasons, "sweeps": sweeps, "last_sweep_change_rel": change,
                                "stage_balance_residual_rel": residual, "overall_balance_closure_rel": overall,
                                "all_stage_flashes_verified": all(s["flash_verified"] for s in stages)}}


def liquid_liquid_extraction(mode: str | None = None, feed: dict | None = None, flow_basis: str | None = None,
                             solvent: dict | None = None, T_K: float | None = None, P_Pa: float | None = None,
                             model: str | None = None, activity_parameters: list[dict] | None = None,
                             distribution_coefficients: dict | None = None, carrier: str | None = None,
                             extraction_solvent: str | None = None, n_stages: int | None = None, target: dict | None = None,
                             sizing: dict | None = None, costing: dict | None = None, year: int = 2023,
                             cepci: float | None = None, max_sweeps: int = 400, tolerance: float = 1e-9) -> dict:
    """Decanter / mixer-settler split (mode 'decanter') or N ideal countercurrent stages (mode 'countercurrent'), with the
    repo flash engine (`model`) or with constant distribution coefficients; optional settler sizing and purchase cost."""
    if mode not in ("decanter", "countercurrent"):
        raise ValueError("Falta 'mode': 'decanter' (una etapa: decantador o mezclador-sedimentador) o 'countercurrent' "
                         "(N etapas en contracorriente).")
    if flow_basis not in ("kg_s", "mol_s"):
        raise ValueError("Falta 'flow_basis': 'kg_s' o 'mol_s', las unidades de los caudales de 'feed' y 'solvent'.")
    f = _stream(feed, "feed", True)
    s = _stream(solvent, "solvent", mode == "countercurrent")
    if (model is None) == (distribution_coefficients is None):
        raise ValueError("Da exactamente uno de 'model' (equilibrio con el motor de flash: NRTL o UNIQUAC con tus "
                         "parámetros, o UNIFAC / UNIFAC_LLE predictivos) o 'distribution_coefficients' ({soluto: K} "
                         "constantes, con 'carrier' y 'extraction_solvent'). El equilibrio no se supone.")
    if mode == "countercurrent" and (isinstance(n_stages, bool) or not isinstance(n_stages, int) or n_stages < 1):
        raise ValueError("Falta 'n_stages' (número de etapas ideales, entero ≥ 1) para la cascada en contracorriente.")
    if mode == "decanter" and n_stages not in (None, 1):
        raise ValueError("mode='decanter' es una sola etapa: no pases n_stages (usa mode='countercurrent').")
    if model is not None:
        if model not in FLASH_MODELS:
            raise ValueError(f"model '{model}' no reconocido; opciones: {list(FLASH_MODELS)}.")
        t = _num(T_K, "T_K", "temperatura de la extracción (el equilibrio depende de ella)")
        p = _num(P_Pa, "P_Pa", "presión de la extracción")
        if target is not None:
            raise ValueError("'target' (etapas necesarias por Kremser) solo existe con 'distribution_coefficients'; con un "
                             "modelo termodinámico prueba varios n_stages.")
        res = _lle_thermo(mode, f, s, flow_basis, model, activity_parameters, t, p, extraction_solvent, n_stages,
                          max_sweeps, tolerance)
    else:
        res = _lle_constant(mode, f, s, flow_basis, distribution_coefficients, carrier, extraction_solvent, n_stages, target)
        res["warnings"] = []
    pairs = res.pop("_phases", None)
    res = {"mode": mode} | res
    res.setdefault("status", "ok" if res["evidence"]["verified"] else "not_verified")
    not_included = ["Eficiencia de etapa, arrastre de una fase en la otra y tiempo de coalescencia.",
                    "Emulsiones estables, tensoactivos, sólidos en la interfase y efecto de la agitación en el tamaño de gota.",
                    "Efectos térmicos (calor de mezcla) y reacciones (p. ej. jabones en la decantación de biodiésel).",
                    "Mezcladores (agitador y su potencia) de un mezclador-sedimentador; columnas de extracción (altura, "
                    "inundación)."]
    res["sizing"], res["cost"], res["equipment"] = None, None, []
    n_units = res["n_stages"]
    q_total = None
    if sizing is not None or costing is not None:
        over = sizing or {}
        if model is not None:
            if not pairs:
                res["warnings"].append("No se dimensiona ni se costea: el equilibrio no dio dos fases líquidas.")
                sizing = costing = None
                light = heavy = None
            else:
                light, heavy = max(pairs, key=lambda pr: pr[0]["volumetric_flow_m3_s"] + pr[1]["volumetric_flow_m3_s"])
        if model is not None and light is not None:
            rho_l = (_num(over["light_density_kg_m3"], "sizing['light_density_kg_m3']", "densidad de la fase ligera")
                     if "light_density_kg_m3" in over else light["density_kg_m3"])
            rho_h = (_num(over["heavy_density_kg_m3"], "sizing['heavy_density_kg_m3']", "densidad de la fase pesada")
                     if "heavy_density_kg_m3" in over else heavy["density_kg_m3"])
            q_l, q_h = light["total_kg_s"] / rho_l, heavy["total_kg_s"] / rho_h
            density_source = ("dadas por el usuario" if "light_density_kg_m3" in over or "heavy_density_kg_m3" in over
                              else "estimadas por thermo para cada fase")
        elif model is None:
            if flow_basis != "kg_s":
                raise ValueError("Con coeficientes de distribución constantes, el dimensionamiento y el costo necesitan "
                                 "caudales másicos (flow_basis='kg_s') para pasar a caudal volumétrico.")
            rho_r = _num(over.get("raffinate_density_kg_m3"), "sizing['raffinate_density_kg_m3']",
                         "densidad de la fase refinado (no hay modelo que la estime)")
            rho_e = _num(over.get("extract_density_kg_m3"), "sizing['extract_density_kg_m3']",
                         "densidad de la fase extracto (no hay modelo que la estime)")
            m_r, m_e = res["raffinate"]["total"], res["extract"]["total"]
            if mode == "countercurrent":       # the largest flow of each phase along the cascade (its loaded end)
                m_r, m_e = max(m_r, sum(f.values())), max(m_e, sum(s.values()))
            (rho_l, q_l), (rho_h, q_h) = sorted([(rho_r, m_r / rho_r), (rho_e, m_e / rho_e)])
            density_source = "dadas por el usuario"
        if sizing is not None or costing is not None:
            q_total = q_l + q_h
        if sizing is not None:
            res["sizing"] = size_gravity_settler(q_l, q_h, rho_l, rho_h, sizing)
            res["sizing"] |= {"light_density_kg_m3": rho_l, "heavy_density_kg_m3": rho_h, "density_source": density_source,
                              "number_of_units": n_units}
            res["warnings"] += res["sizing"].pop("warnings")
            res["assumptions"] += [
                "Decantador: velocidad terminal de la gota por la ley de Stokes (gota rígida aislada, sin coalescencia ni "
                "sedimentación impedida) y flujo pistón de la fase continua a través de la interfase (u_c = Q_c/A_i ≤ u_d).",
                "El recipiente es el mayor de los dos criterios: tiempo de retención del usuario y área de interfase "
                "necesaria; cilindro sin cabezas. Con varias etapas, todos los sedimentadores se dimensionan iguales con "
                "la etapa de mayor caudal volumétrico."]
            if not res["sizing"]["stokes_regime"]:
                res["evidence"]["verified"] = False
                res["evidence"].setdefault("reasons", []).append("la gota sale del régimen de Stokes (Re > 1)")
            res["evidence"]["settler_volume_closure_rel"] = abs(
                math.pi * res["sizing"]["diameter_m"] ** 2 * res["sizing"]["length_m"] / 4.0
                - res["sizing"]["vessel_volume_m3"]) / res["sizing"]["vessel_volume_m3"]
        if costing is not None:
            res["cost"], res["equipment"] = _separator_cost(costing, res["sizing"], q_total, n_units, year, cepci,
                                                            res["warnings"])
            res["assumptions"].append("Costo de COMPRA (no instalado) de " + str(n_units) + " unidad(es) iguales, una por "
                                      "etapa; ver cost.correlation.")
    if sizing is None:
        not_included.append("Dimensionamiento del decantador: no se pidió (falta sizing).")
    if costing is None:
        not_included.append("Costo: no se pidió (falta costing).")
    res["not_included"] = not_included
    if not res["evidence"]["verified"] and res["status"] == "ok":
        res["status"] = "not_verified"
    if not res["evidence"]["verified"]:
        res["warnings"].append("Resultado NO verificado: " + "; ".join(res["evidence"].get("reasons", [])) + ".")
    return _clean(res)
