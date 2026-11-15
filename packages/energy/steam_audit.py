"""Distribution losses of the steam diagnostic (bloque M20): leaks, failed traps and bare pipe, priced with the plant's steam.

The physics lives in `steam_distribution` (validated there against the DOE tables). This module takes what a walk-through of
the plant produces (counts, hole sizes, pipe lengths), calls that engine and values each loss with figures the diagnostic
already computed: the cost and the CO2 of a tonne of steam, and the fuel cost of a GJ of heat delivered.

What the plant gives: every count, diameter, length, pressure, ambient temperature and wind. What it is not asked for: the
four model constants in `MODEL_CONSTANTS`, each with its source; any of them can be overridden by name.
"""
from __future__ import annotations

from packages.energy import steam_distribution as SDI

TOOL = "steam_distribution (flujo por orificio con IAPWS-95; convección natural o forzada más radiación)"
KEYS = ["leaks", "failed_traps", "bare_pipes", "ambient_T_K", "wind_speed_m_s", "leak_discharge_coefficient",
        "trap_discharge_coefficient", "surface_emissivity", "jacket_emissivity"]
ROW_KEYS = ["description", "count", "orifice_diameter_m", "steam_pressure_Pa", "back_pressure_Pa"]
PIPE_KEYS = ["description", "outer_diameter_m", "length_m", "steam_pressure_Pa", "insulation_thickness_m",
             "insulation_conductivity_W_m_K"]
LOSS_SHARE_FLAG = 0.30         # flag when the listed losses exceed this share of the steam delivered (plausibility, not a limit)
MODEL_CONSTANTS = {
    "leak_discharge_coefficient": {
        "value": 0.60, "what": "coeficiente de descarga de una fuga (orificio de borde vivo)",
        "source": "G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), 2002: un orificio de borde vivo deja "
                  "pasar aproximadamente 60 % del flujo de una tobera bien redondeada. Con 0.60 el motor reproduce su Tabla 6 "
                  "entre −0.8 % y +0.4 % (la columna de 300 psig, −4 %)"},
    "trap_discharge_coefficient": {
        "value": 0.72, "what": "coeficiente de descarga de una trampa fallada abierta (orificio reentrante)",
        "source": "U.S. DOE, Steam Tip Sheet #1, DOE/GO-102012-3401, enero 2012: su tabla usa 0.72. Con 0.72 el motor "
                  "reproduce esa tabla entre −3.8 % y +2.9 %"},
    "surface_emissivity": {
        "value": 0.80, "what": "emisividad de la superficie del tubo de acero sin aislar",
        "source": "valor de «Steel» en la tabla de metales base del programa 3E Plus v4.1 de NAIMA (guía del usuario, agosto "
                  "2012), que implementa ASTM C680; «Iron or steel 0.8» en la Mechanical Insulation Design Guide de la NIA. "
                  "Con 0.80 el motor reproduce dos resultados de 3E Plus a −0.3 % y la tabla de ASHRAE de tubería de acero "
                  "desnuda (115 valores) entre −2.3 % y +0.5 %"},
    "jacket_emissivity": {
        "value": 0.90, "what": "emisividad de la cubierta del aislamiento",
        "source": "elección conservadora de este estudio (una cubierta opaca pierde más que una de aluminio brillante, así "
                  "que el ahorro no queda inflado); cambia poco el resultado porque la pérdida ya aislada es pequeña"},
}


def _known(d: dict, allowed: list[str], where: str) -> None:
    extra = sorted(k for k in d if k not in allowed)
    if extra:
        raise ValueError(f"{where}: claves desconocidas {extra}. Válidas: {allowed}.")


def distribution_losses(distribution: dict, site_pressure_Pa: float, currency: str, hours: float | None,
                        cost_per_tonne: float, fuel_cost_per_GJ_delivered: float, co2_kg_per_tonne: float,
                        heat_delivered_GJ_per_tonne: float, delivered_tonne_per_year: float | None,
                        cost_per_tonne_after_measures: float | None = None) -> dict:
    """Steam and heat lost in the distribution, per item and in total, valued with the baseline figures of the diagnostic."""
    _known(distribution, KEYS, "distribution")
    leaks, traps, pipes = (list(distribution.get(k) or []) for k in ("leaks", "failed_traps", "bare_pipes"))
    if not (leaks or traps or pipes):
        raise ValueError("distribution: da al menos una fila en leaks (fugas), failed_traps (trampas falladas abiertas) o "
                         "bare_pipes (tubería sin aislar).")
    constants, assumptions, warnings = {}, [], []
    for name, spec in MODEL_CONSTANTS.items():
        given = distribution.get(name)
        constants[name] = {"value": float(given) if given is not None else spec["value"],
                           "origin": "dato del usuario" if given is not None else "constante del modelo", "what": spec["what"],
                           "source": None if given is not None else spec["source"]}
    year_s = None if hours is None else float(hours) * 3600.0

    def money(tonnes_per_year: float | None, GJ_per_year: float | None = None) -> dict:
        if tonnes_per_year is None:
            return {"steam_tonne_per_year": None, "cost_per_year": None, "co2_tonne_per_year": None}
        if GJ_per_year is None:                                                   # steam that leaves the system
            return {"steam_tonne_per_year": tonnes_per_year, "cost_per_year": tonnes_per_year * cost_per_tonne,
                    "co2_tonne_per_year": tonnes_per_year * co2_kg_per_tonne / 1000.0}
        return {"steam_tonne_per_year": tonnes_per_year, "heat_GJ_per_year": GJ_per_year,    # heat only: fuel, not water
                "cost_per_year": GJ_per_year * fuel_cost_per_GJ_delivered,
                "co2_tonne_per_year": GJ_per_year / heat_delivered_GJ_per_tonne * co2_kg_per_tonne / 1000.0}

    def orifices(rows: list[dict], where: str, cd: float, label: str) -> dict | None:
        if not rows:
            return None
        groups = []
        for i, row in enumerate(rows):
            _known(row, ROW_KEYS, f"distribution.{where}[{i}]")
            miss = [k for k in ("count", "orifice_diameter_m", "steam_pressure_Pa") if row.get(k) is None]
            if miss:
                raise ValueError(f"distribution.{where}[{i}]: faltan {miss}. Salen del recorrido de la planta: cuántas hay, el "
                                 "diámetro del orificio y la presión del vapor en ese punto. No se suponen.")
            if float(row["count"]) != int(float(row["count"])) or float(row["count"]) < 1:
                raise ValueError(f"distribution.{where}[{i}].count = {row['count']}: debe ser un número entero de piezas, 1 o más.")
            groups.append({"label": row.get("description") or f"{label} {i + 1}", "count": int(float(row["count"])),
                           "orifice_diameter_m": float(row["orifice_diameter_m"]),
                           "steam_pressure_Pa": float(row["steam_pressure_Pa"]),
                           "back_pressure_Pa": float(row["back_pressure_Pa"]) if row.get("back_pressure_Pa") is not None
                           else float(site_pressure_Pa)})
        r = SDI.failed_trap_loss(groups, cd)
        kg_s = r["total_mass_flow_kg_s"]
        items = [{"description": g["label"], "count": g["count"], "orifice_diameter_m": g["orifice_diameter_m"],
                  "steam_pressure_Pa": g["steam_pressure_Pa"], "choked": g["choked"], "steam_kg_h": g["mass_flow_kg_s"] * 3600.0,
                  **money(None if year_s is None else g["mass_flow_kg_s"] * year_s / 1000.0)} for g in r["groups"]]
        return {"steam_kg_h": kg_s * 3600.0, "items": items, "discharge_coefficient": cd,
                **money(None if year_s is None else kg_s * year_s / 1000.0), "engine_verified": r["evidence"]["verified"],
                "engine_warnings": r["warnings"]}

    out_leaks = orifices(leaks, "leaks", constants["leak_discharge_coefficient"]["value"], "fuga")
    out_traps = orifices(traps, "failed_traps", constants["trap_discharge_coefficient"]["value"], "trampa")

    out_pipes = None
    if pipes:
        if distribution.get("ambient_T_K") is None or distribution.get("wind_speed_m_s") is None:
            raise ValueError("distribution: con bare_pipes da ambient_T_K (temperatura del aire junto a las tuberías) y "
                             "wind_speed_m_s (viento; 0 si están bajo techo). Son datos de la planta; no se suponen.")
        segments, proposed = [], []
        for i, row in enumerate(pipes):
            _known(row, PIPE_KEYS, f"distribution.bare_pipes[{i}]")
            miss = [k for k in ("outer_diameter_m", "length_m", "steam_pressure_Pa") if row.get(k) is None]
            if miss:
                raise ValueError(f"distribution.bare_pipes[{i}]: faltan {miss} (diámetro exterior, longitud del tramo sin "
                                 "aislar y presión del vapor que lleva). No se suponen.")
            seg = {"label": row.get("description") or f"tramo {i + 1}", "outer_diameter_m": float(row["outer_diameter_m"]),
                   "length_m": float(row["length_m"]), "steam_pressure_Pa": float(row["steam_pressure_Pa"]),
                   "surface_emissivity": constants["surface_emissivity"]["value"]}
            t, k = row.get("insulation_thickness_m"), row.get("insulation_conductivity_W_m_K")
            if (t is None) != (k is None):
                raise ValueError(f"distribution.bare_pipes[{i}]: para evaluar el aislamiento da el espesor y la conductividad "
                                 "térmica del material (ficha del proveedor); con solo uno no se puede. Sin ninguno se "
                                 "reporta solo lo que el tramo pierde hoy.")
            if t is not None:
                seg.update(insulation_thickness_m=float(t), insulation_conductivity_W_m_K=float(k),
                           jacket_emissivity=constants["jacket_emissivity"]["value"])
            proposed.append(t is not None)
            segments.append(seg)
        r = SDI.bare_surface_heat_loss(segments, float(distribution["ambient_T_K"]), float(distribution["wind_speed_m_s"]),
                                       float(site_pressure_Pa))
        items = []
        saving_W = 0.0
        for seg, has_ins in zip(r["segments"], proposed):
            W = seg["bare"]["heat_loss_W"]
            item = {"description": seg["label"], "outer_diameter_m": seg["outer_diameter_m"], "length_m": seg["length_m"],
                    "surface_T_K": seg["surface_T_K"], "heat_loss_W": W, "heat_loss_W_per_m": W / seg["length_m"],
                    "steam_condensed_kg_h": seg["steam_condensed_bare_kg_s"] * 3600.0,
                    **money(None if year_s is None else seg["steam_condensed_bare_kg_s"] * year_s / 1000.0,
                            None if year_s is None else W * year_s / 1e9), "insulation": None}
            if has_ins:
                s_W = seg["saving_W"]
                saving_W += s_W
                item["insulation"] = {
                    "thickness_m": seg["insulated"]["insulation_thickness_m"] if "insulation_thickness_m" in seg["insulated"]
                    else None, "heat_loss_W": seg["insulated"]["heat_loss_W"], "jacket_T_K": seg["insulated"]["surface_T_K"],
                    "saving_W": s_W, "saving_fraction": seg["saving_fraction"],
                    **{("saving_" + k2): v for k2, v in money(
                        None if year_s is None else seg["steam_saved_kg_s"] * year_s / 1000.0,
                        None if year_s is None else s_W * year_s / 1e9).items()}}
            else:
                item["insulation_note"] = ("ahorro no evaluado: falta el espesor y la conductividad del aislamiento propuesto "
                                           "(ficha del proveedor)")
            items.append(item)
        total_W = r["bare_heat_loss_W"]
        out_pipes = {"heat_loss_W": total_W, "items": items, "ambient_T_K": float(distribution["ambient_T_K"]),
                     "wind_speed_m_s": float(distribution["wind_speed_m_s"]),
                     **money(None if year_s is None else r["steam_condensed_bare_kg_s"] * year_s / 1000.0,
                             None if year_s is None else total_W * year_s / 1e9),
                     "insulation_saving_W": saving_W if any(proposed) else None,
                     "insulation_saving_cost_per_year": (saving_W * year_s / 1e9 * fuel_cost_per_GJ_delivered
                                                         if any(proposed) and year_s is not None else None),
                     "engine_verified": r["evidence"]["verified"], "engine_warnings": r["warnings"]}

    parts = [p for p in (out_leaks, out_traps, out_pipes) if p]
    total_cost = None if year_s is None else sum(p["cost_per_year"] for p in parts)
    total_co2 = None if year_s is None else sum(p["co2_tonne_per_year"] for p in parts)
    lost_steam = None if year_s is None else sum(p["steam_tonne_per_year"] for p in parts if p is not out_pipes)
    share = (None if lost_steam is None or not delivered_tonne_per_year
             else (lost_steam + (out_pipes["steam_tonne_per_year"] if out_pipes else 0.0)) / delivered_tonne_per_year)
    if share is not None and share > LOSS_SHARE_FLAG:
        warnings.append(f"REVISAR DATO: las fugas, trampas y tubería sin aislar listadas suman {100 * share:.0f} % del vapor "
                        "que entrega la caldera. Es mucho: revisa los diámetros de orificio (se pide el del orificio, no el "
                        "de la tubería) y las cantidades.")
    for p in parts:                                   # per-segment savings are reported here, so the engine's note on totals is moot
        warnings.extend(w for w in p["engine_warnings"] if w not in warnings and "solo algunos tramos traen datos" not in w)
    used = [n for n in MODEL_CONSTANTS if (n == "leak_discharge_coefficient" and leaks)
            or (n == "trap_discharge_coefficient" and traps) or (n == "surface_emissivity" and pipes)
            or (n == "jacket_emissivity" and out_pipes and out_pipes["insulation_saving_W"] is not None)]
    for n in used:
        c = constants[n]
        assumptions.append(f"{c['what'].capitalize()}: {c['value']:.2f}"
                           + (f". Fuente: {c['source']}." if c["source"] else " (dado por el usuario)."))
    assumptions += [
        "El vapor de fugas y trampas falladas se valora al costo de una tonelada de vapor de la línea base (combustible, agua y "
        "electricidad), como vapor que sale del sistema. Si una trampa descarga a un retorno de condensado cerrado, el agua no "
        "se pierde y el costo real es algo menor.",
        "El calor que pierde la tubería sin aislar se valora solo con el combustible (costo de combustible por GJ de calor "
        "entregado), porque el condensado que se forma no sale del sistema.",
        "Las pérdidas se suponen constantes durante todas las horas de operación declaradas.",
        "Flujo por orificio con expansión en equilibrio y vapor saturado a la presión dada.",
    ]
    if hours is None:
        warnings.append("Sin operation.hours_per_year las pérdidas de distribución se dan solo en kg/h y kW, sin cifras al año.")
    return {
        "currency": currency, "hours_per_year": hours, "leaks": out_leaks, "failed_traps": out_traps, "bare_pipes": out_pipes,
        "total": {"cost_per_year": total_cost, "co2_tonne_per_year": total_co2, "share_of_delivered_steam": share,
                  "max_investment_for_payback": (None if not total_cost else {str(n): n * total_cost for n in (1, 2, 3)})},
        "valuation": {"cost_per_tonne_steam": cost_per_tonne, "fuel_cost_per_GJ_delivered": fuel_cost_per_GJ_delivered,
                      "co2_kg_per_tonne_steam": co2_kg_per_tonne,
                      "cost_per_tonne_after_boiler_measures": cost_per_tonne_after_measures,
                      "note": "Valorado con la línea base. Si además se aplican las medidas de la caldera, cada tonelada de "
                              "vapor cuesta menos y estas pérdidas valen proporcionalmente menos."},
        "model_constants": {n: constants[n] for n in used}, "warnings": warnings, "assumptions": assumptions,
        "not_included": ["fugas en válvulas, bridas y empaques que no se listaron", "trampas que fallan cerradas (no pierden "
                         "vapor; inundan equipos)", "válvulas, bridas y accesorios sin aislar", "tubería de condensado sin "
                         "aislar", "el costo de reparar o aislar: sale de una cotización"],
        "evidence": {"verified": all(p["engine_verified"] for p in parts),
                     "meaning": "verified = los balances internos del motor de pérdidas cierran. Los conteos y diámetros son "
                                "datos del recorrido de la planta y no se comprueban."},
        "method": {"module": "packages/energy/steam_audit.py", "engine": TOOL,
                   "validated_against": "tablas del U.S. DOE Steam Tip Sheet #1 y de la guía ORNL/TM-2001/263 (fugas y "
                                        "trampas); para tubería, el programa 3E Plus v4.1 de NAIMA, la tabla de ASHRAE de "
                                        "tubería de acero desnuda y un ejemplo resuelto de Incropera y DeWitt"},
    }
