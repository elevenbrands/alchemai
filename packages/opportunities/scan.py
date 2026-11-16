"""Opportunity scanner (bloque M27_O): deterministic, explainable improvement and optimization opportunities for a
ProcessModel (the common JSON of M27) and for raw plant readings an engineer pastes from a historian or a spreadsheet.

Nothing here re-implements an engine. Each rule is one function that gathers its inputs, calls the engine of the repo
that already answers the physics, compares with a PUBLISHED benchmark (packages/opportunities/benchmarks.json: value,
source, URL, period, date read) and turns the gap into USD per year at a price that is either the caller's or a sourced
reference of the price banks (`packages.costing.reference_prices`, which delegates gas and electricity to the utility bank
of `packages.costing.utilities`):

  (a) heat integration     -> `packages.energy.pinch.pinch` (Problem Table Algorithm) vs the current hot utility
  (b) fired heaters/boilers -> `packages.energy.combustion.combustion_efficiency` at the measured and at the published
                              excess air / O2; and the efficiency gap to the published furnace efficiency
  (c) steam system          -> `packages.energy.steam_diagnostic.steam_system_diagnostic` (its measures and leaks/traps)
  (d) energy intensity      -> DOE bandwidth studies (refining per process, chemicals per product): CT / SOA / PM bands
  (e) yield / conversion    -> FCC conversion vs the published range of maximum conversion; valued only with the
                              caller's achievable conversion and margin per point
  (f) flaring and CO2       -> flared gas valued as fuel; CO2 valued only with the caller's carbon price
  (g) pumps                 -> measured efficiency vs the U.S. DOE minimum efficiency of a new clean-water pump
  (h) plant economics       -> the ranked sensitivity of `plant_economics` as «confirmar / negociar» items (not savings)

Every opportunity says what it is anchored to (unit and stream ids), its value with a low/high range, the basis, the
assumptions, what data would confirm it, its confidence and the evidence of the engine run. An input that is missing is
NEVER filled: the opportunity comes back with value None and the list of what is missing, or goes to `not_assessed`.
Values are ESTIMATES against references, never measured savings.

Overlap: opportunities of the energy kind that touch the same unit (or, for an energy-intensity gap, the same section)
are merged into one overlap group and the group counts with its largest member only; the steam measures of one boiler
count with the engine's combined case (they are not additive). Economics items never enter the total.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

try:                                    # closed vocabulary of block D
    from packages.diagram.symbols import EQUIPMENT_TYPES
except ImportError:                     # pragma: no cover - only before block D exists
    EQUIPMENT_TYPES = ["pump", "compressor", "fan", "heat_exchanger", "air_cooler", "heater", "cooler", "fired_heater",
                       "boiler", "cooling_tower", "vessel", "flash_drum", "separator_3phase", "decanter", "tank", "column",
                       "packed_column", "reactor", "cstr", "pfr", "fixed_bed_reactor", "fluidized_bed_reactor", "fermenter",
                       "mixer", "splitter", "valve", "filter", "centrifuge", "screw_press", "mill", "conveyor", "dryer",
                       "evaporator", "crystallizer", "cyclone", "membrane", "adsorber", "flare", "stack", "generic"]

BENCH_PATH = Path(__file__).parent / "benchmarks.json"
BTU_J = 1055.05585262                   # J per Btu (IT)
MMBTU_GJ = 1.05505585262
BBL_M3 = 0.158987294928                 # m3 per U.S. barrel (exact: 42 U.S. gal)
LB_KG = 0.45359237
GPM_M3S = 3.785411784e-3 / 60.0         # m3/s per U.S. gpm
FT_M = 0.3048
G0 = 9.80665
CO2_PER_C = 44.0095 / 12.011
CONFIDENCE_ORDER = {"alta": 0, "media": 1, "baja": 2}
ENERGY_CATEGORIES = ("energy", "utilities")
METHOD = ("Reglas deterministas, una función por regla, que llaman a los motores existentes del repositorio (pinch, combustión, "
          "diagnóstico de vapor) y comparan con referencias publicadas y fechadas (benchmarks.json); precios del usuario o de "
          "los bancos de referencia (reference_prices / utilities). Python estándar; sin modelo de lenguaje.")
NOT_INCLUDED = [
    "Inversión, costo de implementación y retorno: ninguna oportunidad estima su costo (cotízalo).",
    "Restricciones de operación, seguridad, confiabilidad y espacio que impidan una medida.",
    "Interacciones entre unidades más allá de la regla de traslape (por ejemplo, menos vapor de proceso que cambia el balance "
    "de la caldera).",
    "Benchmarks de reformación (octano, rendimiento C5+), HDS, compresores y ventiladores: no se abrió una fuente abierta citable.",
]


# --------------------------------------------------------------------------- small helpers

def _norm(text) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9%/]+", " ", t).strip()


@lru_cache(maxsize=1)
def load_benchmarks() -> dict:
    return json.loads(BENCH_PATH.read_text(encoding="utf-8"))


def _bench(key: str) -> dict:
    return load_benchmarks()["entries"][key]


def _cite(key: str) -> dict:
    e = _bench(key)
    return {"benchmark_id": key, "source": e["source"], "url": e["url"], "period": e["period"], "date_read": e["date_read"]}


def _cite_text(key: str) -> str:
    c = _cite(key)
    return f"{c['source']} ({c['period']}; {c['url']}; leído {c['date_read']})"


def _usd(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f} USD/año"


def _num(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: debe ser un número (llegó {x!r}).") from None
    if not math.isfinite(v):
        raise ValueError(f"{what}: debe ser un número finito.")
    return v


# --------------------------------------------------------------------------- plant readings (historian / Excel)

_VAR_ALIASES = {
    "throughput": ["throughput", "feed", "feed rate", "charge", "carga", "alimentacion", "flujo de carga", "procesamiento",
                   "crudo procesado", "carga a la unidad"],
    "production": ["production", "produccion", "producto", "product"],
    "energy": ["energy", "energia", "energia total", "consumo de energia", "onsite energy", "energy total"],
    "fuel": ["fuel", "combustible", "fuel fired", "combustible quemado", "gas combustible"],
    "steam": ["steam", "vapor", "vapor consumido", "steam energy"],
    "electricity": ["electricity", "electricidad", "energia electrica", "consumo electrico", "power consumption"],
    "absorbed": ["absorbed", "duty", "carga termica", "calor absorbido", "absorbed duty"],
    "efficiency": ["efficiency", "eficiencia", "eficiencia termica"],
    "flue_O2": ["flue o2", "o2", "oxigeno en chimenea", "oxigeno", "o2 en humos", "o2 humos"],
    "stack_T": ["stack t", "stack temperature", "temperatura de chimenea", "t chimenea", "temperatura de humos"],
    "flow": ["flow", "flujo", "caudal", "gasto"],
    "head": ["head", "cabeza", "carga dinamica", "carga dinamica total", "tdh", "altura"],
    "density": ["density", "densidad"],
    "shaft_power": ["shaft power", "potencia al eje", "potencia en flecha", "bhp"],
    "electric_power": ["electric power", "potencia electrica", "potencia"],
    "motor_efficiency": ["motor efficiency", "eficiencia del motor"],
    "speed": ["speed", "velocidad", "rpm"],
    "stages": ["stages", "etapas", "pasos"],
    "flare": ["flare", "flare flow", "gas a quemador", "flujo a quemador", "flujo al flare", "gas quemado", "gas a flare"],
    "flare_heating_value": ["flare heating value", "poder calorifico del gas de quemador", "poder calorifico flare",
                            "flare hv"],
    "flare_carbon_fraction": ["flare carbon fraction", "carbono en gas de quemador", "fraccion de carbono flare"],
    "flare_recoverable_fraction": ["flare recoverable fraction", "fraccion recuperable", "fraccion recuperable flare"],
    "conversion": ["conversion", "conversion fcc"],
    "max_conversion": ["max conversion", "conversion maxima", "conversion alcanzable"],
    "conversion_value": ["conversion value", "valor por punto de conversion", "margen por punto de conversion"],
    "hours": ["hours", "horas", "horas de operacion", "horas al ano", "on stream hours"],
    "fuel_price": ["fuel price", "precio del combustible", "precio del gas", "precio gas"],
    "electricity_price": ["electricity price", "precio de la electricidad", "precio electricidad"],
    "carbon_price": ["carbon price", "precio del carbono", "precio co2"],
    "cooling_price": ["cooling price", "precio del enfriamiento", "precio agua de enfriamiento"],
    "T_in": ["t in", "temperatura de entrada", "t entrada"],
    "T_out": ["t out", "temperatura de salida", "t salida"],
}
_ALIAS = {_norm(a): base for base, names in _VAR_ALIASES.items() for a in names + [base]}

_DIMS = {
    "vol_daily": {"bpd": 1.0, "bbl/d": 1.0, "b/d": 1.0, "bbl/dia": 1.0, "barriles/dia": 1.0, "kbpd": 1e3, "mbpd": 1e3,
                  "m3/d": 1.0 / BBL_M3, "m3/h": 24.0 / BBL_M3},
    "power": {"w": 1.0, "kw": 1e3, "mw": 1e6, "gj/h": 1e9 / 3600, "mj/h": 1e6 / 3600, "mmbtu/h": MMBTU_GJ * 1e9 / 3600,
              "gcal/h": 4.1868e9 / 3600},
    "mass_flow": {"kg/h": 1.0, "t/h": 1e3, "kg/s": 3600.0, "t/d": 1e3 / 24, "lb/h": LB_KG},
    "fraction": {"%": 0.01, "fraction": 1.0, "fraccion": 1.0, "-": 1.0, "": 1.0},
    "percent": {"%": 1.0, "% peso": 1.0, "wt%": 1.0, "% wt": 1.0, "%peso": 1.0},
    "vol_flow": {"m3/s": 1.0, "m3/h": 1 / 3600, "gpm": GPM_M3S, "l/s": 1e-3, "bpd": BBL_M3 / 86400},
    "length": {"m": 1.0, "ft": FT_M},
    "density": {"kg/m3": 1.0, "g/cm3": 1e3, "kg/l": 1e3},
    "speed": {"rpm": 1.0},
    "count": {"": 1.0, "-": 1.0, "etapas": 1.0, "stages": 1.0},
    "heating_value": {"j/kg": 1.0, "kj/kg": 1e3, "mj/kg": 1e6, "btu/lb": BTU_J / LB_KG},
    "hours": {"h/y": 1.0, "h/ano": 1.0, "h/año": 1.0, "h": 1.0, "horas": 1.0},
    "price_energy": {"usd/gj": 1.0, "usd/mmbtu": 1.0 / MMBTU_GJ},
    "price_elec": {"usd/kwh": 1.0, "usd/mwh": 1e-3},
    "price_carbon": {"usd/t": 1.0, "usd/tco2": 1.0, "usd/t co2": 1.0},
    "usd_point": {"usd/bbl/punto": 1.0, "usd/bbl/pt": 1.0, "usd/bbl": 1.0},
}
# base variable -> [(dimension, canonical key)]
_ROUTES = {
    "throughput": [("vol_daily", "throughput_bpd")], "production": [("vol_daily", "production_bpd"),
                                                                     ("mass_flow", "production_kg_h")],
    "energy": [("power", "energy_W")], "fuel": [("power", "fuel_W")], "steam": [("power", "steam_W")],
    "electricity": [("power", "electricity_W")], "absorbed": [("power", "absorbed_W")],
    "efficiency": [("fraction", "efficiency")], "flue_O2": [("fraction", "flue_O2")],
    "stack_T": [("temperature", "stack_T_K")], "flow": [("vol_flow", "flow_m3_s")], "head": [("length", "head_m")],
    "density": [("density", "density_kg_m3")], "shaft_power": [("power", "shaft_power_W")],
    "electric_power": [("power", "electric_power_W")], "motor_efficiency": [("fraction", "motor_efficiency")],
    "speed": [("speed", "speed_rpm")], "stages": [("count", "stages")], "flare": [("mass_flow", "flare_kg_h")],
    "flare_heating_value": [("heating_value", "flare_HV_J_kg")],
    "flare_carbon_fraction": [("fraction", "flare_carbon_mass_fraction")],
    "flare_recoverable_fraction": [("fraction", "flare_recoverable_fraction")],
    "conversion": [("percent", "conversion_pct")], "max_conversion": [("percent", "max_conversion_pct")],
    "conversion_value": [("usd_point", "conversion_value_usd_bbl_point")], "hours": [("hours", "hours_per_year")],
    "fuel_price": [("price_energy", "fuel_usd_GJ")], "electricity_price": [("price_elec", "electricity_usd_kWh")],
    "carbon_price": [("price_carbon", "carbon_usd_tCO2")], "cooling_price": [("price_energy", "cooling_usd_GJ")],
    "T_in": [("temperature", "T_in_K")], "T_out": [("temperature", "T_out_K")],
}
SITE_IDS = {"", "site", "sitio", "planta", "plant", "*", "general"}
_SITE_KEYS = {"hours_per_year", "fuel_usd_GJ", "electricity_usd_kWh", "carbon_usd_tCO2", "cooling_usd_GJ"}


def _temperature(v: float, uom: str) -> float | None:
    u = uom.replace("°", "").replace("º", "").strip().lower()
    if u in ("k", "kelvin"):
        return v
    if u in ("c", "degc", "celsius"):
        return v + 273.15
    if u in ("f", "degf", "fahrenheit"):
        return (v + 459.67) / 1.8
    return None


def _basis_from(reading: dict, uom: str) -> str | None:
    text = _norm(f"{uom} {reading.get('basis') or ''}")
    for key, names in (("dry", ("dry", "seco", "seca")), ("wet", ("wet", "humedo", "humeda")),
                       ("HHV", ("hhv", "pcs", "superior")), ("LHV", ("lhv", "pci", "inferior"))):
        if any(n in text.split() for n in names):
            return key
    return None


def parse_readings(table: list[dict]) -> dict:
    """Readings [{unit_id, variable, value, uom, period?, basis?}] -> canonical values per unit (mean of repeats)."""
    if isinstance(table, dict) or not isinstance(table, list):
        raise ValueError("table: da una lista de lecturas [{unit_id, variable, value, uom, period}], una por renglón.")
    acc: dict[str, dict[str, list]] = {}
    accepted, rejected, warnings = [], [], []
    for i, r in enumerate(table):
        if not isinstance(r, dict):
            rejected.append({"row": i, "reason": "el renglón no es un objeto {unit_id, variable, value, uom, period}"})
            continue
        var, uom_raw = r.get("variable"), str(r.get("uom") if r.get("uom") is not None else "")
        base = _ALIAS.get(_norm(var or ""))
        if base is None:
            rejected.append({"row": i, "variable": var, "reason": f"variable desconocida «{var}». Válidas: "
                                                                  f"{sorted(_VAR_ALIASES)} (y sus alias en español)."})
            continue
        try:
            v = _num(r.get("value"), f"renglón {i} ({var})")
        except ValueError as exc:
            rejected.append({"row": i, "variable": var, "reason": str(exc)})
            continue
        basis = _basis_from(r, uom_raw)
        u = re.sub(r"\b(dry|seco|seca|wet|humedo|humeda|hhv|lhv|pcs|pci|base)\b", "", uom_raw.lower()).strip()
        u = re.sub(r"\s+", " ", u).replace("³", "3").strip()
        key, val = None, None
        for dim, canon in _ROUTES[base]:
            if dim == "temperature":
                t = _temperature(v, u)
                if t is not None:
                    key, val = canon, t
                    break
                continue
            if u in _DIMS[dim]:
                key, val = canon, v * _DIMS[dim][u]
                break
        if key is None:
            allowed = sorted({x for dim, _ in _ROUTES[base] for x in (_DIMS.get(dim) or {"K": 1, "°C": 1, "°F": 1})})
            rejected.append({"row": i, "variable": var, "uom": uom_raw,
                             "reason": f"unidad «{uom_raw}» no válida para «{var}». Usa una de {allowed}."})
            continue
        if key == "flue_O2" and basis not in ("dry", "wet"):
            rejected.append({"row": i, "variable": var, "uom": uom_raw,
                             "reason": "el O2 en humos necesita su base: escribe '% dry' / '% base seca' o '% wet' / "
                                       "'% base húmeda' (o el campo basis)."})
            continue
        uid = str(r.get("unit_id") if r.get("unit_id") is not None else "").strip()
        uid = "site" if _norm(uid) in SITE_IDS else uid
        acc.setdefault(uid, {}).setdefault(key, []).append((val, r.get("period"), basis))
        accepted.append({"row": i, "unit_id": uid, "variable": var, "canonical": key, "value_SI": val, "period": r.get("period"),
                         **({"basis": basis} if basis else {})})
    units: dict[str, dict] = {}
    for uid, d in acc.items():
        out = {}
        for key, vals in d.items():
            xs = [x for x, _, _ in vals]
            mean = math.fsum(xs) / len(xs)
            periods = sorted({str(p) for _, p, _ in vals if p})
            bases = sorted({b for _, _, b in vals if b})
            if len(bases) > 1:
                warnings.append(f"{uid}.{key}: lecturas con bases distintas {bases}; se promedian: revisa.")
            if len(xs) > 1 and mean != 0 and (max(xs) - min(xs)) / abs(mean) > 0.10:
                warnings.append(f"{uid}.{key}: {len(xs)} lecturas con dispersión de "
                                f"{100 * (max(xs) - min(xs)) / abs(mean):.0f} %; se usa el promedio.")
            out[key] = {"value": mean, "n": len(xs), "periods": periods, "basis": bases[0] if bases else None}
        units[uid] = out
    return {"units": units, "accepted": accepted, "rejected": rejected, "warnings": warnings}


# --------------------------------------------------------------------------- context: prices, hours, unit data

_PRICE_KEYS = ("fuel_usd_GJ", "electricity_usd_kWh", "carbon_usd_tCO2", "cooling_usd_GJ", "usd_per_currency")


def _resolve_prices(prices: dict | None, region: str, today: date, site: dict) -> tuple[dict, list[str], list[str]]:
    from packages.costing import reference_prices as RP

    prices = dict(prices or {})
    bad = [k for k in prices if k not in _PRICE_KEYS]
    if bad:
        raise ValueError(f"prices: claves desconocidas {bad}. Válidas: {list(_PRICE_KEYS)} (combustible en USD/GJ base PCS, "
                         "electricidad USD/kWh, carbono USD/t CO2, enfriamiento USD/GJ, usd_per_currency {'MXN': USD por MXN}).")
    out, assumptions, warnings = {}, [], []
    for key in ("fuel_usd_GJ", "electricity_usd_kWh", "carbon_usd_tCO2", "cooling_usd_GJ"):
        if prices.get(key) is not None:
            out[key] = {"value": _num(prices[key], f"prices.{key}"), "origin": "dato del usuario"}
        elif key in site:
            out[key] = {"value": site[key]["value"], "origin": f"lectura de planta {site[key]['periods'] or ''}".strip()}
    for key, item in (("fuel_usd_GJ", "natural gas"), ("electricity_usd_kWh", "electricity")):
        if key in out:
            continue
        r = RP.lookup(item, region, "USD", today=today)
        if r.get("found") and r.get("value_converted") is not None:
            out[key] = {"value": float(r["value_converted"]), "origin": "referencia publicada", "source": r["source"],
                        "url": r["url"], "period": r["period"], "date_read": r["date_read"], "region_used": r["region_used"],
                        "basis": r.get("basis"), "flag": RP.FLAG}
            assumptions.append(f"Precio de {'gas natural' if key == 'fuel_usd_GJ' else 'electricidad'}: "
                               f"{r['value_converted']:.4g} {r['unit_converted']} ({r['basis']}; {r['source']}; {r['url']}; "
                               f"periodo {r['period']['start']} a {r['period']['end']}; región {r['region_used']}) — {RP.FLAG}.")
            if r.get("fallback_region_used"):
                warnings.append(f"{key}: no hay valor para {region}; se usa {r['region_used']}: {r.get('fallback_note', '')}")
        else:
            out[key] = None
            warnings.append(f"Sin precio de {'combustible' if key == 'fuel_usd_GJ' else 'electricidad'} para {region}: "
                            f"{r.get('reason', 'no está en el banco')} Las oportunidades que lo necesitan salen sin valor.")
    out.setdefault("carbon_usd_tCO2", None)
    out.setdefault("cooling_usd_GJ", None)
    out["usd_per_currency"] = dict(prices.get("usd_per_currency") or {})
    return out, assumptions, warnings


def _hours(process_model: dict, plant_data: dict, site: dict, region: str, today: date) -> dict:
    from packages.costing import reference_prices as RP

    for where, v in (("plant_data", plant_data.get("hours_per_year")),
                     ("lectura de planta", (site.get("hours_per_year") or {}).get("value")),
                     ("process_model", process_model.get("hours_per_year")),
                     ("process_model.basis", (process_model.get("basis") or {}).get("on_stream_hours"))):
        if v is not None:
            h = _num(v, "hours_per_year")
            if not 0 < h <= 8784:
                raise ValueError("hours_per_year: debe estar entre 0 y 8784 h/año.")
            return {"value": h, "origin": where}
    db = RP.design_basis(region, today=today)["values"].get("on_stream_hours")
    if db is None:
        return {"value": None, "origin": None}
    return {"value": float(db["value"]), "origin": "referencia publicada", "source": db["source"], "url": db["url"],
            "flag": RP.FLAG, "text": f"Horas de operación: {db['value']:g} h/año ({db['source']}; {db['url']}) — {RP.FLAG}; "
                                    "sustitúyelas por las de tu planta."}


_DUTY_KEYS = {"fuel_W": "fuel_W", "fired_W": "fuel_W", "steam_W": "steam_W", "electricity_W": "electricity_W",
              "power_W": "electricity_W", "energy_W": "energy_W", "onsite_energy_W": "energy_W", "absorbed_W": "absorbed_W",
              "duty_W": "duty_W", "heat_W": "duty_W", "cooling_W": "duty_W"}


def _unit_data(unit: dict, model: dict, readings: dict) -> dict:
    """Canonical values of one unit: model params/metrics/duties first, plant readings override. Each value has origin."""
    d: dict[str, dict] = {}
    params, metrics = unit.get("params") or {}, unit.get("metrics") or {}
    for src in (params, metrics):
        for k, v in src.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                d[k] = {"value": float(v), "origin": "modelo"}
            elif isinstance(v, (str, dict, list)):
                d[k] = {"value": v, "origin": "modelo"}
    for k, v in (unit.get("duties_W") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            d[_DUTY_KEYS.get(k, k)] = {"value": abs(float(v)), "origin": "modelo"}
    uid = unit["id"]
    if "throughput_bpd" not in d:
        feeds = [s for s in model.get("streams") or [] if s.get("to") == uid and s.get("kind", "material") == "material"
                 and s.get("flow_bpd") is not None]
        if feeds:
            d["throughput_bpd"] = {"value": math.fsum(float(s["flow_bpd"]) for s in feeds), "origin": "modelo (corrientes de carga)",
                                   "stream_ids": [s["id"] for s in feeds]}
    if "production_bpd" not in d:
        prods = [s for s in model.get("streams") or [] if s.get("from") == uid and s.get("kind", "material") == "material"
                 and s.get("flow_bpd") is not None]
        if prods:
            d["production_bpd"] = {"value": math.fsum(float(s["flow_bpd"]) for s in prods),
                                   "origin": "modelo (corrientes de producto)", "stream_ids": [s["id"] for s in prods]}
    for k, rec in (readings.get(uid) or {}).items():
        d[k] = {"value": rec["value"], "origin": f"dato de planta ({rec['n']} lectura(s), periodo "
                                                  f"{', '.join(rec['periods']) or 'sin periodo'})", "basis": rec.get("basis")}
    return d


def _val(ud: dict, key: str):
    return (ud.get(key) or {}).get("value")


def _measured(ud: dict, *keys: str) -> bool:
    """True when at least one of keys is present and every present one comes from plant readings."""
    present = [k for k in keys if k in ud]
    return bool(present) and all(str(ud[k].get("origin", "")).startswith("dato de planta") for k in present)


def _onsite_energy(ud: dict) -> tuple[float | None, str]:
    if _val(ud, "energy_W") is not None:
        return float(_val(ud, "energy_W")), "energía en sitio total (energy_W)"
    parts = [(k, _val(ud, k)) for k in ("fuel_W", "steam_W", "electricity_W") if _val(ud, k) is not None]
    if parts:
        return math.fsum(v for _, v in parts), "suma de " + " + ".join(k for k, _ in parts)
    return None, ""


# --------------------------------------------------------------------------- opportunity record

def _opp(ctx: dict, rule: str, key: str, unit_ids: list[str], category: str, title: str, explanation: str,
         value: float | None, low: float | None, high: float | None, basis: str, assumptions: list[str], evidence: dict,
         confidence: str, confirm: list[str], missing: list[str] | None = None, stream_ids: list[str] | None = None,
         overlap_units: list[str] | None = None, include_in_total: bool = True) -> dict:
    if value is None:
        low = high = None
    else:
        lo, hi = min(x for x in (low, value, high) if x is not None), max(x for x in (low, value, high) if x is not None)
        low, high = lo, hi
    o = {"id": f"{rule}:{key}", "rule": rule, "unit_ids": list(unit_ids), "stream_ids": list(stream_ids or []),
         "category": category, "title_es": title, "explanation_es": explanation, "value_usd_y": value, "value_low_usd_y": low,
         "value_high_usd_y": high, "basis_es": basis, "assumptions": assumptions, "evidence": evidence,
         "confidence": confidence, "data_to_confirm": confirm, "missing": list(missing or []),
         "is_estimate": True, "include_in_total": include_in_total and value is not None,
         "_overlap_units": list(overlap_units or unit_ids)}
    ctx["opportunities"].append(o)
    _status(ctx, rule, unit_ids, "valorada" if value is not None else "sin valor: faltan datos", missing)
    return o


def _skip(ctx: dict, rule: str, unit_ids: list[str], reason: str, missing: list[str] | None = None,
          status: str = "no evaluada") -> None:
    ctx["not_assessed"].append({"rule": rule, "unit_ids": list(unit_ids), "reason": reason, "missing": list(missing or [])})
    _status(ctx, rule, unit_ids, status, missing, reason)


def _status(ctx: dict, rule: str, unit_ids: list[str], status: str, missing: list[str] | None, note: str | None = None) -> None:
    ctx["status"].append({"rule": rule, "unit_ids": list(unit_ids), "status": status, "missing": list(missing or []),
                          **({"note": note} if note else {})})


def _price(ctx: dict, key: str) -> float | None:
    p = ctx["prices"].get(key)
    return None if p is None else p["value"]


def _price_text(ctx: dict, key: str) -> str:
    p = ctx["prices"].get(key)
    if p is None:
        return "sin precio"
    unit = {"fuel_usd_GJ": "USD/GJ", "electricity_usd_kWh": "USD/kWh", "carbon_usd_tCO2": "USD/t CO2",
            "cooling_usd_GJ": "USD/GJ"}[key]
    return f"{p['value']:.4g} {unit} ({p['origin']}{', ' + p['url'] if p.get('url') else ''})"


def _annual_GJ(W: float, hours: float) -> float:
    return W * hours * 3600.0 / 1e9


# --------------------------------------------------------------------------- (a) heat integration

def rule_heat_integration(model: dict, units: dict, ctx: dict) -> None:
    """Pinch (Problem Table) on the hot/cold streams the model exposes vs the current hot utility -> fuel at sourced price."""
    from packages.energy import pinch as P

    hi = dict(ctx["plant_data"].get("heat_integration") or model.get("heat_integration") or {})
    streams, unit_ids, cur_hot, cur_cold = [], [], 0.0, 0.0
    hot_known, cold_known = True, True
    explicit = hi.get("streams") or model.get("heat_streams")
    if explicit:
        for s in explicit:
            streams.append({k: s[k] for k in ("name", "T_supply_K", "T_target_K", "CP_W_K", "heat_load_W", "dT_contribution_K")
                            if s.get(k) is not None})
            if s.get("unit_id"):
                unit_ids.append(s["unit_id"])
        cur_hot, cur_cold = hi.get("current_hot_utility_W"), hi.get("current_cold_utility_W")
        hot_known, cold_known = cur_hot is not None, cur_cold is not None
    else:
        skipped = []
        for uid, (u, ud) in units.items():
            t = u.get("type")
            if t not in ("heater", "cooler", "air_cooler", "fired_heater", "heat_exchanger"):
                continue
            if t == "heat_exchanger":
                keys = ("hot_T_in_K", "hot_T_out_K", "cold_T_in_K", "cold_T_out_K")
                Q = _val(ud, "duty_W") or _val(ud, "absorbed_W")
                if Q is None or any(_val(ud, k) is None for k in keys):
                    skipped.append(uid)
                    continue
                streams += [{"name": f"{uid}/caliente", "T_supply_K": _val(ud, "hot_T_in_K"), "T_target_K": _val(ud, "hot_T_out_K"),
                             "heat_load_W": Q},
                            {"name": f"{uid}/fría", "T_supply_K": _val(ud, "cold_T_in_K"), "T_target_K": _val(ud, "cold_T_out_K"),
                             "heat_load_W": Q}]
                unit_ids.append(uid)
                continue
            Q = _val(ud, "absorbed_W") if t == "fired_heater" else (_val(ud, "duty_W") or _val(ud, "absorbed_W"))
            Ti, To = _val(ud, "T_in_K"), _val(ud, "T_out_K")
            if Q is None or Ti is None or To is None or abs(Ti - To) < 1e-9:
                skipped.append(uid)
                continue
            streams.append({"name": uid, "T_supply_K": Ti, "T_target_K": To, "heat_load_W": Q})
            unit_ids.append(uid)
            if t in ("heater", "fired_heater"):
                cur_hot += Q
            else:
                cur_cold += Q
        if skipped:
            ctx["warnings"].append(f"Integración térmica: sin temperaturas de entrada/salida o carga, quedaron fuera {skipped}.")
    if not streams or not any(s["T_supply_K"] > s["T_target_K"] for s in streams) or not any(
            s["T_supply_K"] < s["T_target_K"] for s in streams):
        _skip(ctx, "a_heat_integration", unit_ids, "Integración térmica: hace falta al menos una corriente caliente y una fría con "
              "T_in_K, T_out_K y carga (o heat_integration.streams).", ["corrientes calientes y frías con temperaturas y carga"])
        return
    bench = _bench("pinch_dT_min_typical")
    dT_user = hi.get("dT_min_K", model.get("dT_min_K"))
    if dT_user is not None:
        dts, dt_src = (float(dT_user),) * 3, "ΔTmin del usuario"
    else:
        rng = bench["fouling_or_low_h_K"] if model.get("kind") == "refinery" else bench["chemical_K"]
        dts = (rng[1], (rng[0] + rng[1]) / 2, rng[0])
        dt_src = (f"ΔTmin no dado: rango típico publicado {rng[0]:g}-{rng[1]:g} K "
                  f"({'servicio con ensuciamiento / coeficientes bajos' if model.get('kind') == 'refinery' else 'procesos químicos'}"
                  f"; {_cite_text('pinch_dT_min_typical')}); el valor central (punto medio) es nuestro")
    runs = [P.pinch(streams, dt, None, cur_hot if hot_known else None, cur_cold if cold_known else None) for dt in dts]
    targets = [r["Q_hot_min_W"] for r in runs]
    ev = {"engine": "packages.energy.pinch.pinch", "verified": all(r["evidence"]["verified"] for r in runs),
          "dT_min_K_low_central_high_value": list(dts), "Q_hot_min_W": targets, "Q_cold_min_W": [r["Q_cold_min_W"] for r in runs],
          "pinch_evidence_central": runs[1]["evidence"], "current_hot_utility_W": cur_hot if hot_known else None,
          "current_cold_utility_W": cur_cold if cold_known else None}
    missing = []
    if not hot_known:
        missing.append("consumo actual de servicio caliente (current_hot_utility_W)")
    fuel = _price(ctx, "fuel_usd_GJ")
    hours = ctx["hours"]["value"]
    if fuel is None:
        missing.append("precio del combustible")
    if hours is None:
        missing.append("horas de operación al año")
    saving_W = [max((cur_hot or 0.0) - q, 0.0) for q in targets] if hot_known else None
    eff_user = hi.get("heater_efficiency")
    furnace = _bench("refinery_furnace_efficiency")
    title = "Integración térmica: recuperar el calor que hoy cruza el pinch"
    if hot_known and saving_W[2] <= 0:
        _skip(ctx, "a_heat_integration", unit_ids, "El servicio caliente actual ya está en el blanco mínimo del pinch "
              f"({cur_hot / 1e6:.3f} MW frente a {targets[1] / 1e6:.3f} MW).", status="sin oportunidad")
        return
    value = low = high = None
    if not missing:
        def usd(W, eta):
            return _annual_GJ(W / eta, hours) * fuel
        if eff_user is not None:
            eta = _num(eff_user, "heat_integration.heater_efficiency")
            value, low, high = usd(saving_W[1], eta), usd(saving_W[0], eta), usd(saving_W[2], eta)
        else:
            value, low, high = usd(saving_W[1], furnace["value_high"]), usd(saving_W[0], 1.0), usd(saving_W[2], furnace["value_low"])
    cold_note = ""
    if cold_known and _price(ctx, "cooling_usd_GJ") is not None and value is not None:
        cold_note = (f" El servicio frío baja también {max(cur_cold - ev['Q_cold_min_W'][1], 0) / 1e6:.3f} MW (no sumado: "
                     "queda en evidence).")
        ev["cold_utility_saving_usd_y"] = _annual_GJ(max(cur_cold - ev["Q_cold_min_W"][1], 0), hours) * _price(ctx, "cooling_usd_GJ")
    expl = (f"El análisis pinch de {len(streams)} corrientes da un mínimo de servicio caliente de {targets[1] / 1e6:.3f} MW con "
            f"ΔTmin {dts[1]:g} K" + (f" frente a {cur_hot / 1e6:.3f} MW que la planta usa hoy: hasta "
                                     f"{saving_W[1] / 1e6:.3f} MW de calor recuperable entre corrientes." if hot_known else
                                     "; sin el consumo actual de servicio caliente no hay brecha que valorar.") + cold_note)
    assumptions = [dt_src,
                   "El blanco MER es el máximo termodinámico para ese ΔTmin: un retrofit real recupera una parte y necesita área "
                   "y una red de intercambio nueva (no costeadas).",
                   ("Eficiencia del equipo que da el servicio caliente: la del usuario." if eff_user is not None else
                    f"Sin eficiencia del horno: central con {furnace['value_high']:.0%} y alto con {furnace['value_low']:.0%} "
                    f"(rango promedio de hornos de refinería, {_cite_text('refinery_furnace_efficiency')}); bajo con 100 % "
                    "(cada GJ de calor ahorrado ahorra al menos 1 GJ de combustible).")]
    if ctx["hours"].get("text"):
        assumptions.append(ctx["hours"]["text"])
    measured = bool(explicit) and hi.get("from_plant_data", False)
    suppliers = [uid for uid, (u, _) in units.items() if u.get("type") in ("heater", "fired_heater", "boiler")]
    ev["overlaps_with_hot_utility_units"] = suppliers
    _opp(ctx, "a_heat_integration", "pinch", sorted(set(unit_ids)), "energy", title, expl, value, low, high,
         f"(servicio caliente actual − blanco pinch) ÷ eficiencia × {hours or '—'} h/año × {_price_text(ctx, 'fuel_usd_GJ')}",
         assumptions, ev, "media" if measured else "baja",
         ["temperaturas y flujos reales de las corrientes (balance cerrado)", "consumo medido de servicio caliente",
          "ΔTmin acordado con el diseño", "eficiencia de los hornos"], missing,
         overlap_units=sorted(set(unit_ids)) + suppliers)


# --------------------------------------------------------------------------- (b) fired heaters and boilers

def _combustion_inputs(uid: str, ud: dict, ctx: dict) -> dict | None:
    c = (ctx["plant_data"].get("combustion") or {}).get(uid) or _val(ud, "combustion")
    if not isinstance(c, dict):
        return None
    c = dict(c)
    if _val(ud, "stack_T_K") is not None:
        c["stack_T_K"] = _val(ud, "stack_T_K")
    if _val(ud, "flue_O2") is not None and isinstance(_val(ud, "flue_O2"), float):
        c["flue_O2"] = {"mole_fraction": _val(ud, "flue_O2"), "basis": (ud["flue_O2"].get("basis") or "dry")}
    return c


def rule_fired_equipment(model: dict, units: dict, ctx: dict) -> None:
    """(b) Excess air / O2 at the published value with the combustion engine; furnace efficiency gap to the published one."""
    from packages.energy import combustion as CB

    fuel_price, hours = _price(ctx, "fuel_usd_GJ"), ctx["hours"]["value"]
    for uid, (u, ud) in units.items():
        t = u.get("type")
        if t not in ("fired_heater", "boiler"):
            continue
        c = _combustion_inputs(uid, ud, ctx)
        eff_now = None
        eff_basis = None
        if c is not None:
            need = [k for k in ("fuel", "air", "stack_T_K") if c.get(k) is None]
            if c.get("pressure_Pa") is None and c.get("site_pressure_Pa") is None:
                need.append("pressure_Pa")
            if c.get("flue_O2") is None and c.get("excess_air_fraction") is None:
                need.append("flue_O2 o excess_air_fraction")
            if c.get("setting_loss_fraction") is None or c.get("setting_loss_basis") is None:
                need.append("setting_loss_fraction + setting_loss_basis (pérdida por paredes)")
            if need:
                _skip(ctx, "b_excess_air", [uid], f"{uid}: datos de combustión incompletos.", need)
                c = None
        else:
            _skip(ctx, "b_excess_air", [uid], f"{uid}: sin datos de combustión (combustible, aire, O2 en humos, temperatura de "
                  "chimenea, pérdida por paredes) no se puede calcular la eficiencia ni el efecto del exceso de aire.",
                  ["combustion {fuel, air, pressure_Pa, flue_O2|excess_air_fraction, stack_T_K, setting_loss_fraction, "
                   "setting_loss_basis}"])
        if c is not None:
            P = c.get("pressure_Pa", c.get("site_pressure_Pa"))
            kw = dict(stack_T_K=c["stack_T_K"], setting_loss_fraction=c["setting_loss_fraction"],
                      setting_loss_basis=c["setting_loss_basis"], adiabatic_flame=False)
            now = CB.combustion_efficiency(c["fuel"], c["air"], P, excess_air_fraction=c.get("excess_air_fraction"),
                                           flue_O2=c.get("flue_O2"), **kw)
            eta0 = now["efficiency"]["HHV"]["efficiency"]
            eff_now, eff_basis = eta0, "HHV (motor de combustión)"
            ratio = now["fuel"]["HHV_J_kg"] / now["fuel"]["LHV_J_kg"]
            if t == "fired_heater":
                b = _bench("refinery_heater_excess_O2")
                targets = {"low": {"mole_fraction": b["value_high"], "basis": "dry"},
                           "high": {"mole_fraction": b["value_low"], "basis": "dry"}}
                O2_now = now["flue_gas"]["dry_mole_fractions"].get("O2", 0.0)
                runs = {k: CB.combustion_efficiency(c["fuel"], c["air"], P, flue_O2=v, **kw) for k, v in targets.items()
                        if O2_now > v["mole_fraction"]}
                cite, tgt_text = "refinery_heater_excess_O2", f"O2 en humos {b['value_low']:.0%}-{b['value_high']:.0%} (base seca)"
                now_text = f"O2 {O2_now:.1%} base seca"
            else:
                b = _bench("boiler_excess_air")
                ea_now = now["air"]["excess_air_fraction"]
                runs = ({"low": CB.combustion_efficiency(c["fuel"], c["air"], P, excess_air_fraction=b["value"], **kw)}
                        if ea_now > b["value"] else {})
                if runs:
                    runs["high"] = runs["low"]
                cite, tgt_text = "boiler_excess_air", f"exceso de aire {b['value']:.0%}"
                now_text = f"exceso de aire {ea_now:.0%}"
            if not runs:
                _skip(ctx, "b_excess_air", [uid], f"{uid}: {now_text}, ya dentro de la referencia ({tgt_text}).",
                      status="sin oportunidad")
            else:
                fuel_W, fuel_origin = _fuel_HHV_W(ud, eta0, ratio)
                missing = [m for m, ok in (("combustible quemado (fuel_W) o carga absorbida (absorbed_W)", fuel_W is not None),
                                           ("precio del combustible", fuel_price is not None),
                                           ("horas de operación", hours is not None)) if not ok]
                etas = {k: r["efficiency"]["HHV"]["efficiency"] for k, r in runs.items()}
                frac = {k: 1.0 - eta0 / e for k, e in etas.items()}
                vals = {k: _annual_GJ(fuel_W * f, hours) * fuel_price for k, f in frac.items()} if not missing else {}
                ev = {"engine": "packages.energy.combustion.combustion_efficiency", "efficiency_HHV_now": eta0,
                      "efficiency_HHV_target": etas, "fuel_saving_fraction": frac,
                      "verified": bool(now.get("evidence", {}).get("verified", True)) and all(
                          r.get("evidence", {}).get("verified", True) for r in runs.values()),
                      "benchmark": _cite(cite), "fuel_HHV_W": fuel_W, "fuel_origin": fuel_origin}
                _opp(ctx, "b_excess_air", uid, [uid], "energy",
                     f"{u.get('label_es') or uid}: afinar la combustión (bajar el exceso de aire)",
                     f"Hoy {now_text}; eficiencia {eta0:.1%} (PCS). A {tgt_text} — la referencia publicada — la eficiencia sube "
                     f"a {max(etas.values()):.1%} con la misma temperatura de chimenea: {100 * max(frac.values()):.1f} % menos "
                     "combustible.",
                     vals.get("low", 0.0 if vals else None), vals.get("low", 0.0 if vals else None), vals.get("high"),
                     f"combustible × (1 − η_hoy/η_meta) × horas × {_price_text(ctx, 'fuel_usd_GJ')}",
                     [f"Referencia: {_cite_text(cite)}.", "La temperatura de chimenea se deja igual (si baja al afinar, el "
                      "ahorro crece).", "El objetivo de O2 que aguanta el quemador lo fija la planta (CO, estabilidad de flama)."]
                     + ([ctx["hours"]["text"]] if ctx["hours"].get("text") else []),
                     ev, "alta" if _measured(ud, "stack_T_K", "flue_O2") and "stack_T_K" in ud else "media",
                     ["O2 y CO en humos con analizador calibrado", "consumo de combustible medido", "límite del quemador"],
                     missing)
        if t != "fired_heater":
            continue
        if eff_now is None and _val(ud, "efficiency") is not None:
            eff_now = float(_val(ud, "efficiency"))
            eff_basis = (ud["efficiency"].get("basis") or _val(ud, "efficiency_basis") or "desconocida")
        if eff_now is None and _val(ud, "absorbed_W") and _val(ud, "fuel_W"):
            eff_now = _val(ud, "absorbed_W") / _val(ud, "fuel_W")
            eff_basis = _val(ud, "fuel_basis") or "la del combustible reportado (desconocida)"
        if eff_now is None:
            _skip(ctx, "b_furnace_efficiency", [uid], f"{uid}: sin eficiencia (medida, calculada o absorbida/combustible).",
                  ["eficiencia del horno o datos de combustión"])
            continue
        fb = _bench("refinery_furnace_efficiency")
        if eff_now >= fb["value_high"]:
            _skip(ctx, "b_furnace_efficiency", [uid], f"{uid}: eficiencia {eff_now:.1%}, ya en el tope del rango publicado "
                  f"({fb['value_high']:.0%}).", status="sin oportunidad")
            continue
        fuel_W, fuel_origin = _fuel_HHV_W(ud, eff_now, None)
        missing = [m for m, ok in (("combustible quemado (fuel_W) o carga absorbida (absorbed_W)", fuel_W is not None),
                                   ("precio del combustible", fuel_price is not None),
                                   ("horas de operación", hours is not None)) if not ok]
        f90, f92 = 1 - eff_now / fb["value_high"], 1 - eff_now / fb["max_theoretical_HHV"]
        v90 = v92 = None
        if not missing:
            v90, v92 = _annual_GJ(fuel_W * f90, hours) * fuel_price, _annual_GJ(fuel_W * f92, hours) * fuel_price
        warn = [] if str(eff_basis).upper().startswith("HHV") else [
            f"La eficiencia está en base {eff_basis}; la referencia de 92 % es PCS: si tu eficiencia es PCI la brecha real es "
            "mayor (el valor queda conservador)."]
        _opp(ctx, "b_furnace_efficiency", uid, [uid], "energy",
             f"{u.get('label_es') or uid}: eficiencia del horno bajo la referencia",
             f"Eficiencia {eff_now:.1%} frente al tope del rango promedio de hornos de refinería ({fb['value_high']:.0%}) y al "
             f"máximo práctico de {fb['max_theoretical_HHV']:.0%} PCS: precalentar aire, recuperar calor de humos, quemadores y "
             "control.", v90, v90, v92,
             f"combustible × (1 − η/{fb['value_high']:.2f}) [alto: η/{fb['max_theoretical_HHV']:.2f}] × horas × "
             f"{_price_text(ctx, 'fuel_usd_GJ')}",
             [f"Referencia: {_cite_text('refinery_furnace_efficiency')}.", *warn,
              "El bajo es igual al central: el 90 % ya es el tope de lo que reportan hornos promedio."]
             + ([ctx["hours"]["text"]] if ctx["hours"].get("text") else []),
             {"efficiency_now": eff_now, "efficiency_basis": eff_basis, "benchmark": _cite("refinery_furnace_efficiency"),
              "fuel_W": fuel_W, "fuel_origin": fuel_origin, "verified": True},
             "media" if _measured(ud, "efficiency", "fuel_W", "absorbed_W") else "baja",
             ["prueba de eficiencia del horno (pérdidas por chimenea y paredes)", "consumo de combustible medido"], missing)


def _fuel_HHV_W(ud: dict, eta_HHV: float, hhv_over_lhv: float | None) -> tuple[float | None, str]:
    f = _val(ud, "fuel_W")
    if f is not None:
        basis = str((ud.get("fuel_W") or {}).get("basis") or _val(ud, "fuel_basis") or "HHV").upper()
        if basis == "LHV" and hhv_over_lhv:
            return f * hhv_over_lhv, "combustible dado en PCI, pasado a PCS con el PCS/PCI del motor"
        return f, f"combustible dado (base {basis})"
    a = _val(ud, "absorbed_W")
    if a is not None and eta_HHV:
        return a / eta_HHV, "carga absorbida ÷ eficiencia PCS"
    return None, ""


# --------------------------------------------------------------------------- (c) steam system

def rule_steam(model: dict, units: dict, ctx: dict) -> None:
    """(c) Steam measures (condensate, blowdown, excess air, stack, leaks/traps) through steam_system_diagnostic."""
    from packages.energy import steam_diagnostic as SD

    sys_in = ctx["plant_data"].get("steam_system")
    entries = sys_in if isinstance(sys_in, list) else ([sys_in] if sys_in else [])
    done = set()
    for k, e in enumerate(entries):
        uid = e.get("unit_id") or f"vapor_{k + 1}"
        inputs = dict(e.get("inputs") or {})
        if not inputs.get("currency"):
            raise ValueError("steam_system.inputs: da currency y los argumentos de steam_system_diagnostic.")
        inputs.setdefault("today", ctx["today"].isoformat())
        r = SD.steam_system_diagnostic(**inputs)
        cur = r["currency"]
        fx, fx_text = _usd_per(cur, ctx)
        done.add(uid)
        group = f"steam:{uid}"
        evaluated = [m for m in r["measures"] if m.get("status") == "evaluated" and m.get("savings")]
        for m in r["measures"]:
            if m not in evaluated:
                _skip(ctx, "c_steam", [uid], f"{uid} / {m['label']}: {m.get('reason') or 'no evaluada'}", m.get("needs") or [])
        comb = r.get("combined") or {}
        comb_usd = None
        if comb.get("status") == "evaluated" and fx is not None:
            comb_usd = comb["savings"]["annual"]["cost"]["total"] * fx
        for m in evaluated:
            ann = m["savings"]["annual"]["cost"]["total"]
            v = ann * fx if fx is not None else None
            o = _opp(ctx, "c_steam", f"{uid}:{m['id']}", [uid], "utilities", f"Vapor — {m['label']}",
                     f"Diagnóstico de vapor: {m['label'].lower()} ahorra {m['savings']['annual']['fuel_GJ']:,.0f} GJ/año de "
                     f"combustible ({ann:,.0f} {cur}/año). Las medidas no se suman entre sí: el conjunto vale "
                     f"{(comb['savings']['annual']['cost']['total'] if comb.get('status') == 'evaluated' else float('nan')):,.0f} "
                     f"{cur}/año.", v, v, v,
                     f"steam_system_diagnostic con los datos de la planta; {fx_text}",
                     list(r.get("assumptions") or [])[:6],
                     {"engine": "packages.energy.steam_diagnostic.steam_system_diagnostic",
                      "verified": bool(r["evidence"]["verified"]), "reasons": r["evidence"].get("reasons"),
                      "residuals": m.get("residuals"), "saving_in_currency": ann, "currency": cur},
                     "alta" if r["evidence"]["verified"] else "media",
                     ["datos de la hoja de vapor (purga, retorno de condensado, O2, chimenea) medidos"],
                     [] if fx is not None else [f"tipo de cambio {cur}→USD (prices.usd_per_currency)"])
            o["_group"], o["_group_total"] = group, comb_usd
        dist = r.get("distribution")
        if dist and (dist.get("total") or {}).get("cost_per_year") is not None:
            ann = dist["total"]["cost_per_year"]
            v = ann * fx if fx is not None else None
            _opp(ctx, "c_steam", f"{uid}:distribution", [uid], "utilities", "Vapor — fugas, trampas falladas y tubería sin aislar",
                 f"Pérdidas de distribución valoradas con el costo del vapor de la línea base: {ann:,.0f} {cur}/año.", v, v, v,
                 f"steam_system_diagnostic.distribution; {fx_text}", ["Valoradas al costo del vapor ANTES de las medidas de "
                 "caldera: si se aplican también, el valor real es algo menor."],
                 {"engine": "packages.energy.steam_diagnostic (distribution)", "verified": bool(dist.get("evidence", {}).get(
                     "verified", r["evidence"]["verified"])), "saving_in_currency": ann, "currency": cur},
                 "media", ["recorrido de trampas y fugas con su presión y diámetro"],
                 [] if fx is not None else [f"tipo de cambio {cur}→USD"])
    for uid, (u, ud) in units.items():
        if u.get("type") == "boiler" and uid not in done:
            _skip(ctx, "c_steam", [uid], f"{uid}: sin datos del sistema de vapor no se evalúan condensado, purga ni trampas.",
                  ["plant_data.steam_system {unit_id, inputs: argumentos de steam_system_diagnostic}"])


def _usd_per(currency: str, ctx: dict) -> tuple[float | None, str]:
    if currency.upper() == "USD":
        return 1.0, "en USD"
    user = ctx["prices"]["usd_per_currency"].get(currency.upper())
    if user is not None:
        return _num(user, f"usd_per_currency.{currency}"), f"tipo de cambio del usuario: {user} USD/{currency}"
    if currency.upper() == "MXN":
        from packages.costing import reference_prices as RP

        r = RP.lookup("fx_fix_mxn_usd", "MX", today=ctx["today"])
        if r.get("found"):
            ctx["assumptions"].append(f"Tipo de cambio {r['value']} MXN/USD ({r['source']}; {r['url']}; {r['period']}) — "
                                      f"{RP.FLAG}.")
            return 1.0 / float(r["value"]), f"FIX {r['value']} MXN/USD ({r['source']})"
    return None, f"sin tipo de cambio {currency}→USD"


# --------------------------------------------------------------------------- (d) energy intensity vs bandwidth

_REFINING_KEYWORDS = [   # order matters: hydrocracking before cracking, vacuum before atmospheric
    ("catalytic_hydrocracking", ("hydrocrack", "hidrocraq", "hidrodesintegr", "hcu")),
    ("fluid_catalytic_cracking", ("fcc", "catalytic crack", "desintegracion catalitica", "craqueo catalitico")),
    ("vacuum_crude_distillation", ("vacuum", "vacio", "vdu", "destilacion al vacio")),
    ("atmospheric_crude_distillation", ("atmospheric", "atmosferica", "cdu", "destilacion primaria", "crude distillation",
                                        "topping", "destilacion de crudo")),
    ("catalytic_reforming", ("reform", "ccr", "platform")),
    ("hydrotreating", ("hydrotreat", "hidrotrat", "hds", "hidrodesulf", "nht", "dht")),
    ("coking_visbreaking", ("coker", "coqui", "coque", "visbreak", "reductora de viscosidad")),
    ("alkylation", ("alkyl", "alquil")),
    ("isomerization", ("isomer",)),
]


# keyword inference only on these types (the primary unit of a process), never on auxiliaries of another unit
_PROCESS_TYPES = ("column", "packed_column", "reactor", "fixed_bed_reactor", "fluidized_bed_reactor", "cstr", "pfr",
                  "fired_heater")
_AUXILIARY_WORDS = ("fraccion", "fractionat", "stripper", "agotador", "splitter", "estabiliz", "stabiliz", "regenerad",
                    "scrubber", "lavador", "absorbedor", "absorber")
INFERRED = "proceso inferido por palabra clave (confírmalo con params.bandwidth_process)"


def _explicit_processes(units: dict) -> set[str]:
    return {str(_val(ud, "bandwidth_process")) for _, ud in units.values() if _val(ud, "bandwidth_process")}


def _refining_process(u: dict, ud: dict, explicit_procs: set[str] | None = None) -> tuple[str | None, str]:
    """bandwidth_process when given (only that); else keyword inference on a primary process unit whose process is not
    already assigned explicitly to another unit of the model."""
    procs = _bench("refining_bandwidth_2015")["processes"]
    explicit = _val(ud, "bandwidth_process")
    if explicit:
        if explicit not in procs:
            raise ValueError(f"{u['id']}: bandwidth_process «{explicit}» no existe. Válidos: {sorted(procs)}.")
        return explicit, "dado"
    if u.get("type") not in _PROCESS_TYPES:
        return None, ""
    text = " " + _norm(" ".join(str(x) for x in (u.get("id"), u.get("label_es"), _val(ud, "process") or ""))) + " "
    if any(w in text for w in _AUXILIARY_WORDS):
        return None, ""
    for key, words in _REFINING_KEYWORDS:
        if any((" " + w + " ") in text or (len(w) > 4 and w in text) for w in words):
            if key in (explicit_procs or set()):
                return None, ""
            return key, INFERRED
    return None, ""


def _chemical(u: dict, ud: dict, model: dict) -> str | None:
    chems = _bench("chemicals_bandwidth_2015")["chemicals"]
    name = _val(ud, "bandwidth_chemical")
    if not name:
        return None
    n = _norm(name).replace(" ", "_")
    for k, c in chems.items():
        if n == k or _norm(name) in {_norm(a) for a in c["aliases"]}:
            return k
    raise ValueError(f"{u['id']}: bandwidth_chemical «{name}» no está en el estudio. Válidos: {sorted(chems)}.")


def rule_energy_intensity(model: dict, units: dict, ctx: dict) -> None:
    """(d) Energy intensity of a process unit vs the DOE bandwidth CT / SOA / PM bands -> gap x throughput x fuel price."""
    fuel, hours = _price(ctx, "fuel_usd_GJ"), ctx["hours"]["value"]
    by_section: dict[str, list[str]] = {}
    for uid, (u, _) in units.items():
        if u.get("section"):
            by_section.setdefault(u["section"], []).append(uid)
    explicit_procs = _explicit_processes(units)
    for uid, (u, ud) in units.items():
        chem = _chemical(u, ud, model)
        proc, how = (None, "") if chem else _refining_process(u, ud, explicit_procs)
        if not chem and not proc:
            continue
        if chem:
            b = _bench("chemicals_bandwidth_2015")["chemicals"][chem]
            cite, label, unit = "chemicals_bandwidth_2015", b["label_es"], "Btu/lb"
            rate, rate_key = _val(ud, "production_kg_h"), "production_kg_h"
        else:
            b = _bench("refining_bandwidth_2015")["processes"][proc]
            cite, label, unit = "refining_bandwidth_2015", b["label_es"], "Btu/bbl"
            rate_key = "production_bpd" if b["throughput_basis"] == "production" else "throughput_bpd"
            rate = _val(ud, rate_key)
        if rate is not None and str((ud.get(rate_key) or {}).get("origin", "")).startswith("modelo (corrientes"):
            rate = None                          # a stream sum is not the unit's own throughput: never benchmark on it
        E, how_E = _onsite_energy(ud)            # only the unit's own energy (no section sums)
        missing = [m for m, ok in ((f"{'producción' if rate_key.startswith('production') else 'carga'} propia de la unidad "
                                    f"({rate_key} en params/metrics o lectura)", rate is not None and rate > 0),
                                   ("energía en sitio de la unidad (energy_W o fuel_W + steam_W + electricity_W)", E is not None))
                   if not ok]
        if missing:
            _opp(ctx, "d_energy_intensity", uid, [uid], "energy", f"{u.get('label_es') or uid}: intensidad energética vs {label}",
                 "No se puede comparar con la referencia sin carga y energía de la unidad.", None, None, None,
                 f"intensidad (energía/{unit}) vs bandas CT/SOA de {_cite_text(cite)}", [], {"benchmark": _cite(cite)}, "baja",
                 ["carga o producción y energía en sitio medidas"], missing, overlap_units=[uid] + by_section.get(u.get("section"), []))
            continue
        if chem:
            I = E * 3600.0 / BTU_J / (rate / LB_KG)          # Btu per lb of product
            annual = rate / LB_KG * (hours or 0.0)          # lb/y
        else:
            I = E * 86400.0 / BTU_J / rate                   # Btu per bbl
            annual = rate * (hours or 0.0) / 24.0            # bbl/y
        gaps = {"CT": max(I - b["CT"], 0.0), "SOA": max(I - b["SOA"], 0.0), "PM": max(I - b["PM"], 0.0)}
        ev = {"intensity": I, "intensity_unit": unit, "bands": {k: b[k] for k in ("CT", "SOA", "PM", "TM")},
              "energy_W": E, "energy_from": how_E, "rate": rate, "rate_key": rate_key, "benchmark": _cite(cite),
              "process_identified": how or "dado", "verified": True}
        from_model = not _measured(ud, *(k for k in ("energy_W", "fuel_W", "steam_W", "electricity_W", rate_key) if k in ud))
        if from_model and abs(I - b["CT"]) <= 0.005 * abs(b["CT"]):
            _skip(ctx, "d_energy_intensity", [uid], f"{uid}: la energía del modelo da exactamente la intensidad típica (CT) del "
                  "estudio: parece salir del mismo benchmark, así que no es una brecha medida. Da la energía medida.",
                  ["energía medida de la unidad"])
            continue
        if gaps["SOA"] <= 0:
            _skip(ctx, "d_energy_intensity", [uid], f"{uid}: {I:,.0f} {unit}, ya en o bajo el estado del arte ({b['SOA']:,} {unit}).",
                  status="sin oportunidad")
            continue
        miss = [m for m, ok in (("precio del combustible", fuel is not None), ("horas de operación", hours is not None)) if not ok]
        val = {k: (g * annual * BTU_J / 1e9 * fuel if not miss else None) for k, g in gaps.items()}
        _opp(ctx, "d_energy_intensity", uid, [uid], "energy",
             f"{u.get('label_es') or uid}: intensidad energética sobre el estado del arte",
             f"{label}: {I:,.0f} {unit} frente a {b['CT']:,} típico de EE. UU. (CT) y {b['SOA']:,} estado del arte (SOA). La "
             f"brecha al SOA es {gaps['SOA']:,.0f} {unit}.",
             val["SOA"], val["CT"], val["PM"],
             f"(intensidad − banda) × {'producción' if chem else 'barriles'}/año × {_price_text(ctx, 'fuel_usd_GJ')}; bajo = sobre "
             "CT, central = sobre SOA, alto = sobre PM (incluye tecnologías en I+D)",
             [f"Referencia: {_cite_text(cite)}; promedios de EE. UU. 2010, no de tu refinería ni de tu crudo.",
              "Toda la brecha se valora como combustible (gas natural): la electricidad y el vapor de la unidad pueden costar "
              "más por GJ.", f"Energía: {how_E}; proceso {how or 'dado'}."]
             + ([ctx["hours"]["text"]] if ctx["hours"].get("text") else []),
             ev, "baja" if (from_model or how == INFERRED) else "media",
             ["energía medida de la unidad en el mismo periodo que la carga", "límites de batería iguales a los del estudio"],
             miss, overlap_units=[uid] + by_section.get(u.get("section"), []))


# --------------------------------------------------------------------------- (e) yield / conversion

def rule_yield(model: dict, units: dict, ctx: dict) -> None:
    """(e) FCC conversion vs the published range of maximum conversion; valued with the caller's target and margin."""
    rng = _bench("fcc_max_conversion_range")
    hours = ctx["hours"]["value"]
    explicit_procs = _explicit_processes(units)
    for uid, (u, ud) in units.items():
        proc, _ = (_refining_process(u, ud, explicit_procs) if not _val(ud, "bandwidth_chemical") else (None, ""))
        if proc in ("catalytic_reforming", "hydrotreating"):
            _skip(ctx, "e_yield", [uid], f"{uid}: no hay un rango abierto y citable de octano/rendimiento ({proc}) en "
                  "benchmarks.json.", ["rango publicado abierto"])
            continue
        if proc != "fluid_catalytic_cracking":
            continue
        conv = _val(ud, "conversion_pct")
        if conv is None:
            _skip(ctx, "e_yield", [uid], f"{uid}: sin conversión de la FCC.", ["conversion_pct (% peso)"])
            continue
        tgt, margin, feed = _val(ud, "max_conversion_pct"), _val(ud, "conversion_value_usd_bbl_point"), _val(ud, "throughput_bpd")
        below = conv < rng["value_low"]
        missing = [m for m, ok in (("conversión máxima de TU carga (max_conversion_pct: prueba MAT/ACE o licenciante)", tgt),
                                   ("valor por barril de carga y por punto de conversión (conversion_value_usd_bbl_point)",
                                    margin is not None),
                                   ("carga de la FCC (throughput_bpd)", feed), ("horas de operación", hours)) if not ok]
        if tgt is not None and conv >= tgt:
            _skip(ctx, "e_yield", [uid], f"{uid}: conversión {conv:.1f} % ya en la máxima de su carga ({tgt:.1f} %).",
                  status="sin oportunidad")
            continue
        val = None if missing else feed * hours / 24.0 * (tgt - conv) * margin
        _opp(ctx, "e_yield", uid, [uid], "yield", f"{u.get('label_es') or uid}: conversión de la FCC",
             f"Conversión {conv:.1f} % peso. El rango publicado de conversión máxima es {rng['value_low']}-{rng['value_high']} % "
             f"según la carga{' — estás DEBAJO del rango de todas las cargas estudiadas' if below else ''}. Sin la conversión "
             "máxima de tu carga no se puede afirmar una brecha.", val, val, val,
             "carga × horas/24 × (conversión máxima de tu carga − actual) × valor por punto (dato del usuario)",
             [f"Referencia: {_cite_text('fcc_max_conversion_range')}.", "El valor por punto (diferencia de precio de productos "
              "contra carga no convertida) es del usuario: no se supone."],
             {"conversion_pct": conv, "published_range_pct": [rng["value_low"], rng["value_high"]], "below_range": below,
              "benchmark": _cite("fcc_max_conversion_range"), "verified": True},
             "media" if not missing and _measured(ud, "conversion_pct") else "baja",
             ["prueba de craqueabilidad de la carga", "balance de rendimientos cerrado"], missing)


# --------------------------------------------------------------------------- (f) flaring and CO2

def rule_flaring(model: dict, units: dict, ctx: dict) -> None:
    """(f) Flared gas recovered as fuel at the sourced fuel price; CO2 valued only with the caller's carbon price."""
    fuel, carbon, hours = _price(ctx, "fuel_usd_GJ"), _price(ctx, "carbon_usd_tCO2"), ctx["hours"]["value"]
    for uid, (u, ud) in units.items():
        if u.get("type") != "flare" and _val(ud, "flare_kg_h") is None:
            continue
        m, hv = _val(ud, "flare_kg_h"), _val(ud, "flare_HV_J_kg")
        rec, xc = _val(ud, "flare_recoverable_fraction"), _val(ud, "flare_carbon_mass_fraction")
        missing = [x for x, ok in (("flujo a quemador (flare_kg_h)", m is not None),
                                   ("poder calorífico del gas a quemador (flare_HV_J_kg)", hv is not None),
                                   ("precio del combustible", fuel is not None), ("horas de operación", hours is not None))
                   if not ok]
        v = lo = hi = None
        if not missing:
            full = m * hours * hv / 1e9 * fuel
            v, lo, hi = (full * rec, full * rec, full * rec) if rec is not None else (full, 0.0, full)
        _opp(ctx, "f_flaring", f"{uid}:fuel", [uid], "emissions", f"{u.get('label_es') or uid}: recuperar gas de quemador",
             "El gas que va al quemador en operación normal (no el de emergencia) puede comprimirse y usarse como combustible "
             "en vez de gas comprado." + ("" if rec is not None else " Sin la fracción recuperable el central es la cota "
                                          "superior (todo el flujo medido)."), v, lo, hi,
             f"flujo × horas × poder calorífico × {_price_text(ctx, 'fuel_usd_GJ')}" + (f" × {rec:.0%} recuperable" if rec else ""),
             ["Con el poder calorífico en PCI y el precio en PCS el valor queda conservador.",
              "La guía ENERGY STAR de refinerías (LBNL-56183, sec. 7.1) describe la recuperación de gas de quemador pero no da "
              "una fracción típica: la fracción recuperable es de la planta."],
             {"engine": "aritmética directa: flujo × poder calorífico × precio", "flare_kg_h": m, "verified": True},
             "media" if _measured(ud, "flare_kg_h") and rec is not None else "baja",
             ["medidor de flujo a quemador (ultrasónico) y cromatografía", "separar quema base de quema por emergencia"],
             missing, include_in_total=rec is not None)
        miss2 = [x for x, ok in (("flujo a quemador", m is not None), ("fracción de carbono del gas (flare_carbon_mass_fraction)",
                                                                       xc is not None),
                                 ("horas de operación", hours is not None)) if not ok]
        t_co2 = None if miss2 else m * hours * xc * CO2_PER_C / 1000.0
        if carbon is None:
            miss2.append("precio del carbono (prices.carbon_usd_tCO2): no hay uno abierto y fechado para México en el banco")
        vco2 = None if (miss2 or t_co2 is None) else t_co2 * (rec if rec is not None else 1.0) * carbon
        _opp(ctx, "f_flaring", f"{uid}:co2", [uid], "emissions", f"{u.get('label_es') or uid}: CO2 de la quema",
             (f"La quema emite hasta {t_co2:,.0f} t CO2/año (combustión completa)." if t_co2 is not None else
              "Sin flujo y carbono del gas no se calcula el CO2.") + (" Sin precio del carbono no se valora." if carbon is None
                                                                      else ""),
             vco2, vco2, vco2, "t CO2 evitadas × precio del carbono del usuario",
             ["CO2 = flujo × fracción de carbono × 44/12 (combustión completa: cota superior)."],
             {"engine": "balance de carbono: flujo × fracción de carbono × 44/12", "t_CO2_y": t_co2, "verified": True},
             "baja", ["cromatografía del gas a quemador"], miss2,
             overlap_units=[f"{uid}:co2"])


# --------------------------------------------------------------------------- (g) pumps

def pump_min_efficiency(Q_m3_s: float, H_m: float, speed_rpm: float, stages: int, pump_class: str | None = None) -> dict:
    """DOE minimally compliant pump efficiency at BEP (10 CFR 431 subpart Y, App. A II.B.1.1.1) for one class or all classes."""
    b = _bench("pump_min_efficiency_doe")
    Q_gpm = Q_m3_s / GPM_M3S
    H_ft_stage = H_m / FT_M / stages
    nominal = 1800 if speed_rpm < 2700 else 3600
    Ns = nominal * math.sqrt(Q_gpm) / H_ft_stage ** 0.75
    lq, ln = math.log(Q_gpm), math.log(Ns)
    base = -0.8500 * lq ** 2 - 0.3800 * ln * lq - 11.480 * ln ** 2 + 17.800 * lq + 179.80 * ln - 555.60
    classes = {k: c for k, c in b["C_values"].items() if k.endswith(str(nominal))}
    if pump_class:
        key = f"{pump_class.upper()}.{nominal}"
        if key not in b["C_values"]:
            raise ValueError(f"pump_class «{pump_class}»: usa ESCC, ESFM, IL, RSV o ST.")
        classes = {key: b["C_values"][key]}
    eta = {k: (base - c) / 100.0 for k, c in classes.items()}
    scope = []
    if Q_gpm < 25:
        scope.append(f"flujo {Q_gpm:.1f} gpm < 25 gpm")
    if H_m / FT_M > 459:
        scope.append(f"carga {H_m / FT_M:.0f} ft > 459 ft")
    if Ns > 5000:
        scope.append(f"Ns {Ns:.0f} > 5000")
    return {"Q_gpm": Q_gpm, "H_ft_per_stage": H_ft_stage, "nominal_speed_rpm": nominal, "Ns_US": Ns, "eta_std": eta,
            "out_of_scope": scope}


def rule_pumps(model: dict, units: dict, ctx: dict) -> None:
    """(g) Pump efficiency from flow, head and power vs the DOE minimum efficiency of a new clean-water pump."""
    elec, hours = _price(ctx, "electricity_usd_kWh"), ctx["hours"]["value"]
    for uid, (u, ud) in units.items():
        if u.get("type") == "compressor":
            _skip(ctx, "g_electricity", [uid], f"{uid}: compresor — sin banda publicada abierta de eficiencia en benchmarks.json.",
                  ["banda publicada"])
            continue
        if u.get("type") != "pump":
            continue
        Q, H, rho = _val(ud, "flow_m3_s"), _val(ud, "head_m"), _val(ud, "density_kg_m3")
        Ps, Pe, me = _val(ud, "shaft_power_W"), _val(ud, "electric_power_W"), _val(ud, "motor_efficiency")
        n, S = _val(ud, "speed_rpm"), _val(ud, "stages")
        missing = [x for x, ok in (("flujo (flow_m3_s)", Q), ("carga (head_m)", H), ("densidad (density_kg_m3)", rho),
                                   ("potencia al eje, o potencia eléctrica + eficiencia del motor", Ps or (Pe and me)),
                                   ("velocidad (speed_rpm)", n), ("número de etapas (stages)", S)) if not ok]
        if missing:
            _opp(ctx, "g_electricity", uid, [uid], "energy", f"{u.get('label_es') or uid}: eficiencia de la bomba",
                 "Sin flujo, carga, densidad, potencia, velocidad y etapas no se calcula la eficiencia.", None, None, None,
                 "η = ρ·g·Q·H / potencia al eje vs mínimo DOE", [], {"benchmark": _cite("pump_min_efficiency_doe")}, "baja",
                 ["medición simultánea de flujo, presiones y potencia"], missing)
            continue
        shaft = Ps if Ps else Pe * me
        P_hyd = rho * G0 * Q * H
        eta = P_hyd / shaft
        ref = pump_min_efficiency(Q, H, n, int(round(S)), _val(ud, "pump_class"))
        etas = sorted(ref["eta_std"].values())
        eta_low_ref, eta_high_ref = etas[0], etas[-1]
        if eta >= eta_low_ref:
            _skip(ctx, "g_electricity", [uid], f"{uid}: eficiencia {eta:.1%} ≥ mínimo DOE de una bomba nueva ({eta_low_ref:.1%}).",
                  status="sin oportunidad")
            continue
        mot = me if me else 1.0
        sav = {"low": (shaft - P_hyd / eta_low_ref) / mot, "high": (shaft - P_hyd / eta_high_ref) / mot}
        miss = [x for x, ok in (("precio de la electricidad", elec is not None), ("horas de operación", hours is not None)) if not ok]
        val = {k: s / 1e3 * hours * elec for k, s in sav.items()} if not miss else {}
        warn = []
        if ref["out_of_scope"]:
            warn.append("Fuera del alcance de la norma (" + "; ".join(ref["out_of_scope"]) + "): la referencia es indicativa.")
        if not me:
            warn.append("Sin eficiencia del motor el ahorro eléctrico se iguala al ahorro al eje (cota inferior).")
        _opp(ctx, "g_electricity", uid, [uid], "energy", f"{u.get('label_es') or uid}: bomba bajo la eficiencia mínima de una nueva",
             f"Eficiencia {eta:.1%} (potencia hidráulica {P_hyd / 1e3:.1f} kW, al eje {shaft / 1e3:.1f} kW) frente a "
             f"{eta_low_ref:.1%}-{eta_high_ref:.1%}, la mínima que la norma de EE. UU. exige a una bomba NUEVA de agua limpia "
             "de ese flujo y velocidad específica: reparar (anillos de desgaste, impulsor) o reemplazar.",
             val.get("low"), val.get("low"), val.get("high"),
             f"(P_eje − P_hidráulica/η_mín) ÷ η_motor × horas × {_price_text(ctx, 'electricity_usd_kWh')}",
             [f"Referencia: {_cite_text('pump_min_efficiency_doe')}.",
              "El punto medido se trata como el de mejor eficiencia (BEP); fuera del BEP la norma aplica α = 0.947 a 75 % y 0.985 "
              "a 110 % del flujo.", *warn] + ([ctx["hours"]["text"]] if ctx["hours"].get("text") else []),
             {"efficiency_now": eta, "hydraulic_W": P_hyd, "shaft_W": shaft, "reference": ref,
              "benchmark": _cite("pump_min_efficiency_doe"), "verified": True},
             "alta" if _measured(ud, "flow_m3_s", "head_m", "shaft_power_W", "electric_power_W") and not ref["out_of_scope"]
             else "media", ["prueba de campo de la bomba (flujo, presión de succión y descarga, kW)"], miss)


# --------------------------------------------------------------------------- (h) plant economics sensitivity

def rule_economics(model: dict, units: dict, ctx: dict) -> None:
    """(h) The ranked sensitivity of plant_economics turned into «confirmar / negociar» items (excluded from the total)."""
    econ = ctx["plant_data"].get("plant_economics") or model.get("economics")
    if not econ:
        if model.get("kind") == "plant_model":
            _skip(ctx, "h_economics", [], "Modelo de planta sin resultado de plant_economics: no hay sensibilidad que ordenar.",
                  ["process_model.economics (resultado de plant_economics)"])
        return
    sens = econ.get("sensitivity") or {}
    rows = sens.get("rows") or []
    if not rows:
        _skip(ctx, "h_economics", [], "plant_economics sin filas de sensibilidad.", ["sensitivity_span > 0"])
        return
    base_opex = ((econ.get("operating_costs") or {}).get("total_usd_y"))
    for r in rows[:8]:
        negotiate = str(r["input"]).startswith("prices.")
        lo, hi = r.get("operating_cost_usd_y_low"), r.get("operating_cost_usd_y_high")
        v = abs(hi - lo) / 2.0 if (lo is not None and hi is not None and abs(hi - lo) > 0) else None
        _opp(ctx, "h_economics", str(r["input"]), [], "economics",
             f"{'Negociar' if negotiate else 'Confirmar'}: {r['label']}",
             f"Es el supuesto n.º {r['rank']} que más mueve el resultado ({sens.get('ranked_by')}): de "
             f"{r.get('low_value')} a {r.get('high_value')} {r.get('unit') or ''} el VPN cambia "
             f"{(r.get('npv_swing_usd') or 0):,.0f} USD." + (f" En costo de operación son ±{v:,.0f} USD/año." if v else ""),
             v, v, v, f"sensibilidad de plant_economics ({r.get('range')}); no es un ahorro: es qué confirmar o negociar primero",
             [sens.get("note", "")], {"engine": "packages.costing.plant.plant_economics (sensitivity)", "row": r,
                                       "base_operating_cost_usd_y": base_opex, "verified": True},
             "baja", [f"cotización o dato real de {r['label']}"], [] if v is not None else
             ["(no aplica: el renglón no cambia el costo de operación; su efecto está en el VPN)"], include_in_total=False)


RULES = [rule_heat_integration, rule_fired_equipment, rule_steam, rule_energy_intensity, rule_yield, rule_flaring, rule_pumps,
         rule_economics]


# --------------------------------------------------------------------------- overlap, ranking, total

def _groups(opps: list[dict], units_section: dict[str, str | None]) -> dict[str, list[dict]]:
    parent = list(range(len(opps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(opps):
        for j in range(i + 1, len(opps)):
            b = opps[j]
            same_steam = a.get("_group") and a.get("_group") == b.get("_group")
            energy = a["category"] in ENERGY_CATEGORIES and b["category"] in ENERGY_CATEGORIES
            if same_steam or (energy and set(a["_overlap_units"]) & set(b["_overlap_units"])):
                parent[find(i)] = find(j)
    out: dict[str, list[dict]] = {}
    for i, o in enumerate(opps):
        out.setdefault(str(find(i)), []).append(o)
    return out


def _total(opps: list[dict], units_section: dict) -> dict:
    counted = [o for o in opps if o["include_in_total"]]
    groups = _groups(counted, units_section)
    rows, tot = [], {"value": 0.0, "low": 0.0, "high": 0.0}
    for members in groups.values():
        steam_total = next((m["_group_total"] for m in members if m.get("_group_total") is not None), None)
        steam = [m for m in members if m.get("_group")]
        others = [m for m in members if not m.get("_group")]
        cands = []
        if steam:
            sv = steam_total if steam_total is not None else max(m["value_usd_y"] for m in steam)
            cands.append(("vapor (caso combinado del motor)" if steam_total is not None else "vapor (mayor medida)",
                          sv, sv, sv, [m["id"] for m in steam]))
        for m in others:
            cands.append((m["id"], m["value_usd_y"], m["value_low_usd_y"], m["value_high_usd_y"], [m["id"]]))
        best = max(cands, key=lambda c: c[1])
        tot["value"] += best[1]
        tot["low"] += max(c[2] for c in cands)
        tot["high"] += max(c[3] for c in cands)
        rows.append({"members": [m["id"] for m in members], "counted": best[4], "value_usd_y": best[1]})
    return {"total": tot, "groups": rows}


# --------------------------------------------------------------------------- public entry points

def scan_opportunities(process_model: dict, prices: dict | None = None, region: str = "MX", plant_data=None,
                       today: str | None = None) -> dict:
    """All rules on one ProcessModel; see the module docstring and tools_opportunities.scan_opportunities."""
    if not isinstance(process_model, dict) or not isinstance(process_model.get("units"), list):
        raise ValueError("process_model: da el JSON ProcessModel {name, kind, units: [{id, type, ...}], streams: [...]}.")
    day = date.fromisoformat(today) if today else date.today()
    if isinstance(plant_data, list):
        plant_data = {"readings": plant_data}
    plant_data = dict(plant_data or {})
    parsed = parse_readings(plant_data["readings"]) if plant_data.get("readings") else {"units": {}, "accepted": [],
                                                                                      "rejected": [], "warnings": []}
    site = parsed["units"].get("site", {})
    price_res, p_assump, p_warn = _resolve_prices(prices, region, day, site)
    ctx = {"plant_data": plant_data, "today": day, "region": region, "prices": price_res, "opportunities": [],
           "not_assessed": [], "status": [], "warnings": list(parsed["warnings"]) + p_warn, "assumptions": list(p_assump)}
    ctx["hours"] = _hours(process_model, plant_data, site, region, day)
    if ctx["hours"].get("text"):
        ctx["assumptions"].append(ctx["hours"]["text"])
    ids = [u.get("id") for u in process_model["units"]]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("process_model.units: cada unidad necesita un id único.")
    unknown_readings = [k for k in parsed["units"] if k not in ids and k != "site"]
    if unknown_readings:
        ctx["warnings"].append(f"Lecturas de unidades que no están en el modelo (ignoradas): {unknown_readings}.")
    for u in process_model["units"]:
        if u.get("type") not in EQUIPMENT_TYPES:
            ctx["warnings"].append(f"{u['id']}: tipo «{u.get('type')}» fuera del vocabulario de equipos; se trata como genérico.")
    units = {u["id"]: (u, _unit_data(u, process_model, parsed["units"])) for u in process_model["units"]}
    for rule in RULES:
        rule(process_model, units, ctx)
    opps = ctx["opportunities"]
    opps.sort(key=lambda o: (o["value_usd_y"] is None, not o["include_in_total"], -(o["value_usd_y"] or 0.0),
                             CONFIDENCE_ORDER[o["confidence"]]))
    for i, o in enumerate(opps, 1):
        o["rank"] = i
    tot = _total(opps, {uid: u.get("section") for uid, (u, _) in units.items()})
    for o in opps:
        o.pop("_overlap_units", None)
        o.pop("_group_total", None)
        o["overlap_group"] = o.pop("_group", None)
    group_of = {mid: n for n, g in enumerate(tot["groups"], 1) for mid in g["members"]}
    for o in opps:
        o["overlap_group"] = group_of.get(o["id"]) if o["include_in_total"] else None
    valued = [o for o in opps if o["value_usd_y"] is not None]
    sum_naive = math.fsum(o["value_usd_y"] for o in valued if o["include_in_total"])
    reasons = [f"{o['id']}: {o['evidence'].get('reasons') or 'el motor no verificó'}" for o in valued
               if o["evidence"].get("verified") is False]
    checks = {"total_not_above_naive_sum": tot["total"]["value"] <= sum_naive + 1e-6,
              "low_le_value_le_high": all(o["value_low_usd_y"] <= o["value_usd_y"] <= o["value_high_usd_y"] for o in valued),
              "no_value_without_inputs": all(o["value_usd_y"] is None for o in opps if o["missing"] and o["category"] != "economics"),
              "every_opportunity_cited_or_engine": all(o["evidence"].get("benchmark") or o["evidence"].get("engine")
                                                       for o in opps)}
    if not all(checks.values()):
        reasons.append(f"comprobaciones internas fallidas: {[k for k, v in checks.items() if not v]}")
    out = {
        "name": process_model.get("name"), "kind": process_model.get("kind"), "region": region, "today": day.isoformat(),
        "opportunities": opps, "total_value_usd_y": tot["total"]["value"] if valued else None,
        "total_value_low_usd_y": tot["total"]["low"] if valued else None,
        "total_value_high_usd_y": tot["total"]["high"] if valued else None,
        "total_method_es": ("Suma de oportunidades valoradas que NO se traslapan: las de energía/servicios que tocan la misma "
                            "unidad (o la misma sección, en la brecha de intensidad energética) forman un grupo que cuenta solo "
                            "con su mayor valor; las medidas de vapor de una caldera cuentan con el caso combinado del motor; "
                            "las de economía (sensibilidad) y las de cota superior sin fracción recuperable no se suman."),
        "overlap_groups": tot["groups"], "naive_sum_usd_y": sum_naive if valued else None,
        "not_assessed": ctx["not_assessed"], "rule_status": ctx["status"],
        "prices_used": price_res, "hours_per_year": ctx["hours"],
        "readings": {"accepted": len(parsed["accepted"]), "rejected": parsed["rejected"]},
        "warnings": ctx["warnings"], "assumptions": list(dict.fromkeys(ctx["assumptions"])),
        "not_included": list(NOT_INCLUDED), "method": METHOD,
        "evidence": {"verified": not reasons, "reasons": reasons, "checks": checks, "n_opportunities": len(opps),
                     "n_valued": len(valued), "benchmarks_version": load_benchmarks()["version"],
                     "meaning": "verified certifica que los motores llamados cerraron sus balances y que el total respeta el "
                                "traslape; no certifica los datos de la planta ni que la oportunidad sea realizable."},
    }
    json.dumps(out, default=float)
    return out


_INFER = [("pump", ("flow_m3_s", "head_m")), ("flare", ("flare_kg_h",)), ("fired_heater", ("flue_O2",)),
          ("fired_heater", ("stack_T_K",))]


def opportunities_from_plant_data(table: list[dict], units: list[dict] | None = None, prices: dict | None = None,
                                  region: str = "MX", today: str | None = None, plant_data: dict | None = None) -> dict:
    """Pasted readings -> which rules can run with them, and their result (scan_opportunities on a minimal model)."""
    parsed = parse_readings(table)
    given = {u["id"]: dict(u) for u in (units or []) if isinstance(u, dict) and u.get("id")}
    inferred = []
    for uid, vals in parsed["units"].items():
        if uid == "site" or uid in given:
            continue
        t = next((t for t, keys in _INFER if all(k in vals for k in keys)), "generic")
        given[uid] = {"id": uid, "label_es": uid, "type": t}
        inferred.append({"unit_id": uid, "type": t, "why": "inferido de las variables" if t != "generic" else
                         "sin variables que indiquen el equipo: genérico (da units=[{id, type, label_es}])"})
    if not given:
        raise ValueError("table: ninguna lectura válida de una unidad; revisa unit_id, variable y uom (ver rejected). "
                         f"Rechazadas: {parsed['rejected'][:5]}")
    model = {"name": "datos de planta", "kind": "described", "units": list(given.values()), "streams": [], "sections": [],
             "source": "lecturas pegadas por el usuario"}
    pd = dict(plant_data or {})
    pd["readings"] = table
    scan = scan_opportunities(model, prices, region, pd, today)
    can_run = [s for s in scan["rule_status"] if s["status"] in ("valorada", "sin oportunidad")]
    cannot = [s for s in scan["rule_status"] if s["status"] not in ("valorada", "sin oportunidad")]
    return {"readings_accepted": parsed["accepted"], "readings_rejected": parsed["rejected"], "units_inferred": inferred,
            "rules_can_run": can_run, "rules_cannot_run": cannot, **scan}
