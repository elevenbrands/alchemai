"""Adapters to the shared ProcessModel JSON (block M27_D).

(a) a DWSIM flowsheet as the session records it (``packages.serve.cli.describe_flowsheet``) plus, when solved, the stream
    results of ``read_stream``; (b) a whole-plant result of ``reference_biorefinery`` / ``biodiesel_plant``
    (``packages/plants``: units in ``equipment`` and ``sections``, streams in ``streams`` with ``from_unit``/``to_unit``);
    (c) a hand description in the shape of ``hazop_from_process_description``. A refinery ProcessModel (block R) is native.

The adapters only translate: no flow, temperature or pressure is invented. A stream without data is drawn and its table
row shows «—».
"""

from __future__ import annotations

import re

from packages.diagram.symbols import EQUIPMENT_TYPES, LABELS_ES

# --------------------------------------------------------------------------------------------- type vocabularies

DWSIM_TYPE_MAP: dict[str, str] = {
    # session kinds of packages/dwsim_client/tools.py (add_unit)
    "mixer": "mixer", "splitter": "splitter", "heater": "heater", "cooler": "cooler", "valve": "valve", "pump": "pump",
    "compressor": "compressor", "expander": "expander", "separator": "flash_drum", "heat_exchanger": "heat_exchanger",
    "shortcut_column": "column", "distillation_column": "column", "conversion_reactor": "reactor",
    "equilibrium_reactor": "reactor", "gibbs_reactor": "reactor", "cstr": "cstr", "pfr": "pfr", "pipe": "generic",
    "absorption_column": "packed_column",
    # DWSIM ObjectType names (packages/hazop/graph.py build_graph)
    "Mixer": "mixer", "NodeIn": "mixer", "Splitter": "splitter", "NodeOut": "splitter", "Heater": "heater",
    "Cooler": "cooler", "Valve": "valve", "Pump": "pump", "Compressor": "compressor", "Expander": "expander",
    "Vessel": "flash_drum", "Separator": "flash_drum", "Tank": "tank", "HeatExchanger": "heat_exchanger",
    "ShortcutColumn": "column", "DistillationColumn": "column", "AbsorptionColumn": "packed_column",
    "RefluxedAbsorber": "column", "ReboiledAbsorber": "column", "RCT_Conversion": "reactor",
    "RCT_Equilibrium": "reactor", "RCT_Gibbs": "reactor", "RCT_CSTR": "cstr", "RCT_PFR": "pfr", "RCT_Batch": "cstr",
    "RCT_Generic": "reactor", "Pipe": "generic", "ComponentSeparator": "generic", "SolidsSeparator": "filter",
    "Filter": "filter", "AirCooler": "air_cooler", "AirCooler2": "air_cooler", "Membrane": "membrane",
}
DWSIM_LOGICAL = {"recycle", "Recycle", "OT_Recycle", "EnergyRecycle", "Adjust", "Spec", "OT_Adjust", "OT_Spec"}

# hand descriptions: the hazop vocabulary + Spanish aliases (lower case keys) + the closed vocabulary itself
DESCRIPTION_TYPE_MAP: dict[str, str] = {
    "pump": "pump", "bomba": "pump", "compressor": "compressor", "compresor": "compressor", "heater": "heater",
    "calentador": "heater", "cooler": "cooler", "enfriador": "cooler", "heatexchanger": "heat_exchanger",
    "intercambiador": "heat_exchanger", "valve": "valve", "valvula": "valve", "válvula": "valve", "vessel": "vessel",
    "recipiente": "vessel", "separator": "flash_drum", "separador": "flash_drum", "flash": "flash_drum", "tank": "tank",
    "tanque": "tank", "mixer": "mixer", "mezclador": "mixer", "splitter": "splitter", "divisor": "splitter",
    "pipe": "generic", "tuberia": "generic", "tubería": "generic", "distillationcolumn": "column",
    "absorptioncolumn": "packed_column", "shortcutcolumn": "column", "columna": "column", "torre": "column",
    "absorbedor": "packed_column", "rct_cstr": "cstr", "rct_pfr": "pfr", "rct_batch": "cstr",
    "rct_conversion": "reactor", "rct_equilibrium": "reactor", "rct_gibbs": "reactor", "rct_generic": "reactor",
    "reactor": "reactor", "fermentador": "fermenter", "fermenter": "fermenter", "centrifuga": "centrifuge",
    "centrífuga": "centrifuge", "filtro": "filter", "prensa": "screw_press", "molino": "mill", "secador": "dryer",
    "evaporador": "evaporator", "cristalizador": "crystallizer", "ciclon": "cyclone", "ciclón": "cyclone",
    "caldera": "boiler", "horno": "fired_heater", "decantador": "decanter", "membrana": "membrane",
    "adsorbedor": "adsorber", "transportador": "conveyor", "turbina": "expander", "ventilador": "fan",
    "soplador": "fan", "chimenea": "stack", "quemador": "flare",
}
DESCRIPTION_TYPE_MAP.update({t: t for t in EQUIPMENT_TYPES})

# BioSTEAM unit «line» (u.line) -> closed vocabulary; matched by lower-case keyword, first match wins
BIOSTEAM_LINE_MAP: list[tuple[str, str]] = [
    ("flash", "flash_drum"), ("heat exchanger network", "heat_exchanger"), ("waste vapor condenser", "cooler"),
    ("jet cooker", "heater"), ("thermal oxidizer", "fired_heater"), ("vent scrubber", "packed_column"),
    ("reverse osmosis", "membrane"), ("digestion", "fermenter"), ("seed train", "fermenter"), ("handling", "conveyor"),
    ("hopper", "tank"), ("storage", "tank"), ("magnetic separator", "filter"), ("cip", "generic"),
    ("storage tank", "tank"), ("mix tank", "agitated_tank"), ("slurry mix", "agitated_tank"), ("tank", "tank"),
    ("pump", "pump"),
    ("fermentation", "fermenter"), ("fermentor", "fermenter"), ("bioreactor", "fermenter"), ("saccharification", "cstr"),
    ("transesterification", "cstr"), ("liquefaction", "cstr"), ("neutralization", "cstr"), ("reactor", "reactor"),
    ("hydrolysis", "reactor"), ("pretreatment", "reactor"), ("centrifuge", "centrifuge"), ("clarifier", "decanter"),
    ("decanter", "decanter"), ("settler", "decanter"), ("dryer", "dryer"), ("drum dryer", "dryer"),
    ("heat exchanger", "heat_exchanger"), ("evaporator", "evaporator"), ("flash", "flash_drum"),
    ("divided distillation", "column"), ("distillation", "column"), ("stripper", "column"), ("absorption", "packed_column"),
    ("scrubber", "packed_column"), ("molecular sieve", "adsorber"), ("adsorption", "adsorber"),
    ("boiler", "boiler"), ("turbogenerator", "boiler"), ("cooling tower", "cooling_tower"), ("chilled water", "cooler"),
    ("compressor", "compressor"), ("blower", "fan"), ("fan", "fan"), ("air distribution", "compressor"),
    ("filter", "filter"), ("vibrating screen", "filter"), ("screen", "filter"), ("press", "screw_press"),
    ("mill", "mill"), ("shredder", "mill"), ("crusher", "mill"), ("grinder", "mill"), ("hammer", "mill"),
    ("conveyor", "conveyor"), ("conveying", "conveyor"), ("screw feeder", "conveyor"), ("feeder", "conveyor"),
    ("crystallizer", "crystallizer"), ("cyclone", "cyclone"), ("membrane", "membrane"), ("mixer", "mixer"),
    ("splitter", "splitter"), ("valve", "valve"), ("flare", "flare"), ("stack", "stack"), ("process water", "tank"),
    ("fire water", "tank"), ("wastewater", "generic"), ("waste water", "generic"), ("clean-in-place", "generic"),
    ("mass balance", "generic"),
]
# BioSTEAM unit ID prefix convention (letter before the 3-digit area code) for units that carry no line we know
BIOSTEAM_PREFIX_MAP = {"T": "tank", "P": "pump", "M": "mixer", "S": "splitter", "H": "heat_exchanger", "R": "reactor",
                       "C": "centrifuge", "D": "column", "F": "flash_drum", "E": "evaporator", "K": "compressor",
                       "U": "generic", "B": "generic", "V": "flash_drum", "X": "generic"}
BIOSTEAM_LINE_ES = {
    "Storage tank": "Tanque de almacenamiento", "Mix tank": "Tanque de mezcla", "Pump": "Bomba",
    "Transesterification": "Transesterificación", "Liquids centrifuge": "Centrífuga de líquidos",
    "Vacuum dryer": "Secador al vacío", "Heat exchanger": "Intercambiador de calor",
    "Divided Distillation Column": "Columna de destilación dividida", "Distillation Column": "Columna de destilación",
    "Mass balance": "Balance de reposición", "Splitter": "Divisor", "Mixer": "Mezclador", "Flash": "Tanque flash",
    "Fermentation": "Fermentación", "Boiler turbogenerator": "Caldera y turbogenerador", "Cooling tower": "Torre de enfriamiento",
    "Multi-effect evaporator": "Evaporador de múltiple efecto", "Molecular sieve": "Tamiz molecular",
    "Pressure filter": "Filtro prensa", "Vibrating screen": "Criba vibratoria", "Clarifier": "Clarificador",
    "Conveying belt": "Banda transportadora", "Screw feeder": "Alimentador de tornillo", "Crushing mill": "Molino",
    "Shredder": "Desfibradora", "Process water center": "Centro de agua de proceso", "Fire water tank": "Tanque contra incendio",
    "Chilled water package": "Paquete de agua helada", "Air distribution package": "Distribución de aire",
    "Clean-in-place system": "Sistema CIP", "Wastewater system cost": "Tratamiento de aguas residuales",
    "Saccharification": "Sacarificación", "Liquefaction": "Licuefacción", "Hydrolysis": "Hidrólisis",
    "Pretreatment reactor": "Reactor de pretratamiento", "Reactor": "Reactor", "Stripper": "Agotador",
    "Rotary vacuum filter": "Filtro rotatorio al vacío", "Mixer settler": "Mezclador-sedimentador",
    "Heat exchanger network": "Red de intercambio de calor", "Vent scrubber": "Lavador de venteos",
    "Thermal oxidizer": "Oxidador térmico", "Jet cooker": "Cocedor de chorro", "Hammer mill": "Molino de martillos",
    "Drum dryer": "Secador de tambor", "Solids centrifuge": "Centrífuga de sólidos", "Magnetic separator": "Separador magnético",
    "Reverse osmosis": "Ósmosis inversa", "Anaerobic digestion": "Digestión anaerobia", "Aerobic digestion": "Digestión aerobia",
    "Pretreatment flash": "Flash de pretratamiento", "Pretreatment reactor system": "Reactor de pretratamiento",
    "Seed train": "Tren de inóculo", "Seed hold tank": "Tanque de inóculo", "Beer tank": "Tanque de vino",
    "Simultaneous saccharification fermentation": "Sacarificación y fermentación", "Steam mixer": "Mezclador de vapor",
    "Sludge centrifuge": "Centrífuga de lodos", "Waste vapor condenser": "Condensador de vapores", "Grain handling":
    "Manejo de grano", "Feed stock handling": "Manejo de materia prima", "Membrane bioreactor": "Biorreactor de membrana",
}


def biosteam_type(line: str | None, unit_id: str = "") -> str:
    low = (line or "").lower()
    for key, t in BIOSTEAM_LINE_MAP:
        if key in low:
            return t
    m = re.match(r"([A-Za-z]+)\d", unit_id or "")
    if m and m.group(1)[0].upper() in BIOSTEAM_PREFIX_MAP:
        return BIOSTEAM_PREFIX_MAP[m.group(1)[0].upper()]
    return "generic"


# ------------------------------------------------------------------------------------------------- (a) DWSIM

def from_dwsim_description(desc: dict, stream_results: dict | None = None, name: str | None = None) -> dict:
    """ProcessModel of a DWSIM flowsheet description (shape of ``describe_flowsheet``). ``stream_results`` maps stream
    name -> ``read_stream`` dict (solved values). Logical blocks (recycle, adjust) are bridged: the stream that enters a
    recycle block and the one that leaves it are drawn as one stream from the real source to the real sink."""
    units_in = desc.get("units") or []
    if not units_in:
        raise ValueError("La descripción del flowsheet no trae equipos («units»): no hay diagrama que dibujar.")
    stream_results = stream_results or {}
    specs = desc.get("stream_specs") or {}
    warnings: list[str] = []
    producer: dict[str, str] = {}
    consumer: dict[str, str] = {}
    e_prod: dict[str, str] = {}
    e_cons: dict[str, str] = {}
    logical: dict[str, dict] = {}
    units = []
    for u in units_in:
        nm, kind = str(u["name"]), str(u.get("kind") or "")
        if kind in DWSIM_LOGICAL:
            logical[nm] = u
            continue
        t = DWSIM_TYPE_MAP.get(kind) or DWSIM_TYPE_MAP.get(kind.lower())
        if t is None:
            warnings.append(f"Equipo «{nm}» de tipo DWSIM «{kind}» sin símbolo propio: se dibuja como genérico.")
            t = "generic"
        label = LABELS_ES[t] if kind != "pipe" else "Tubería"
        units.append({"id": nm, "label_es": label, "type": t, "section": "proceso", "params": dict(u.get("params") or {}),
                      "duties_W": {}, "metrics": {}, "dwsim_kind": kind})
        for s in u.get("products") or []:
            producer[s] = nm
        for s in u.get("feeds") or []:
            consumer[s] = nm
        for s in u.get("energy_products") or []:
            e_prod[s] = nm
        for s in u.get("energy_feeds") or []:
            e_cons[s] = nm
    # bridge logical blocks: inlet stream continues as the outlet stream
    bridged: dict[str, str] = {}
    for nm, u in logical.items():
        fe, pr = list(u.get("feeds") or []), list(u.get("products") or [])
        if len(fe) == 1 and len(pr) == 1:
            bridged[fe[0]] = pr[0]
    streams = []
    for s in desc.get("streams") or []:
        if s in bridged.values() and s not in producer:
            continue                                   # drawn as the continuation of its bridged inlet
        to = consumer.get(s)
        out_name = s
        if s in bridged:
            out_name = bridged[s]
            to = consumer.get(out_name)
        res = stream_results.get(out_name) or stream_results.get(s) or {}
        spec = specs.get(s) or {}
        row = {"id": s, "label_es": s if out_name == s else f"{s} (recirculación → {out_name})", "from": producer.get(s),
               "to": to, "kind": "material"}
        T = res.get("T_K", spec.get("temperature_K"))
        P = res.get("P_Pa", spec.get("pressure_Pa"))
        m = res.get("mass_flow_kg_s", spec.get("mass_flow_kg_s"))
        if T is not None:
            row["T_K"] = float(T)
        if P is not None:
            row["P_Pa"] = float(P)
        if m is not None:
            row["flow_kg_h"] = float(m) * 3600.0
        if res.get("vapor_fraction") is not None:
            row["vapor_fraction"] = float(res["vapor_fraction"])
            if row["vapor_fraction"] > 0.999:
                row["phase"] = "g"
        if res.get("mole_fractions"):
            row["composition"] = {k: float(v) for k, v in res["mole_fractions"].items() if v and v > 1e-6}
        if row["from"] is None and row["to"] is None:
            warnings.append(f"La corriente «{s}» no está conectada a ningún equipo: no se dibuja.")
            continue
        streams.append(row)
    for s in desc.get("energy") or []:
        fr, to = e_prod.get(s), e_cons.get(s)
        if fr is None and to is None:
            continue
        streams.append({"id": s, "label_es": s, "from": fr, "to": to, "kind": "energy"})
    if not stream_results:
        warnings.append("Sin resultados resueltos de corrientes: la tabla muestra solo lo especificado en las alimentaciones.")
    return {"name": name or f"Flowsheet {desc.get('flowsheet_id', '')}".strip(), "kind": "dwsim", "units": units,
            "streams": streams, "sections": [{"id": "proceso", "label_es": "Proceso (flowsheet de DWSIM)"}],
            "source": f"flowsheet de DWSIM {desc.get('flowsheet_id', '')}".strip(), "warnings": warnings}


def dwsim_process_model(flowsheet_id: str) -> dict:
    """Read a live DWSIM session (lazy import; needs the DWSIM runtime) and adapt it."""
    from packages.dwsim_client import tools as T
    s = T._s(flowsheet_id)
    desc = {"flowsheet_id": flowsheet_id, "units": [], "streams": list(s["streams"]), "energy": list(s["energy"]),
            "stream_specs": {k: dict(v) for k, v in s["stream_specs"].items()}}
    for nm, kind in s["kinds"].items():
        c = s["conn"].get(nm) or {}
        desc["units"].append({"name": nm, "kind": kind, "params": dict(s["params"].get(nm) or {}),
                              "feeds": c.get("feeds") or [], "products": c.get("products") or [],
                              "energy_feeds": c.get("energy_feeds") or [], "energy_products": c.get("energy_products") or []})
    results = {}
    for nm in s["streams"]:
        try:
            results[nm] = T.read_stream(flowsheet_id, nm)
        except Exception:             # unsolved stream: the table shows the feed spec or «—»
            pass
    return from_dwsim_description(desc, results, name=f"Flowsheet {flowsheet_id}")


# --------------------------------------------------------------------------------------------- (b) plant results

def biosteam_streams(system) -> list[dict]:
    """Every unit-to-unit stream of a LIVE BioSTEAM System, in the row shape of ``packages/plants`` streams
    (id, flow_kg_h, T_K, P_Pa, phase, mass_fractions, from_unit, to_unit). Lets a plant result carry its internal
    topology (``plant_result['internal_streams']``) so the full view is connected unit by unit."""
    rows = []
    for s in system.streams:
        src, snk = getattr(s, "source", None), getattr(s, "sink", None)
        if src is None or snk is None:
            continue
        total = float(s.F_mass)
        comp = {}
        if total > 0:
            for c in s.available_chemicals:
                w = float(s.imass[c.ID]) / total
                if w >= 1e-3:
                    comp[c.ID] = round(w, 5)
        comp = dict(sorted(comp.items(), key=lambda kv: -kv[1])[:8])
        rows.append({"id": s.ID, "flow_kg_h": total, "T_K": float(s.T), "P_Pa": float(s.P), "phase": s.phase,
                     "mass_fractions": comp, "from_unit": src.ID, "to_unit": snk.ID, "role": "interna"})
    return rows


def from_plant_result(result: dict, internal_streams: list[dict] | None = None) -> dict:
    """ProcessModel of a ``reference_biorefinery`` / ``biodiesel_plant`` result. Those results list feeds, products and
    the main streams BETWEEN sections, not every unit-to-unit stream: without ``internal_streams`` (see
    ``biosteam_streams``) the units inside a section are drawn unconnected and that is said in ``warnings``."""
    if not isinstance(result, dict) or not (result.get("equipment") or result.get("sections")):
        raise ValueError("plant_result debe ser el resultado de reference_biorefinery o biodiesel_plant (con «equipment» y "
                         "«sections»).")
    warnings: list[str] = []
    line_of = {e["name"]: e.get("description") for e in result.get("equipment") or []}
    section_of = {e["name"]: str(e.get("section")) for e in result.get("equipment") or []}
    order: list[str] = []
    for sec in result.get("sections") or []:
        for uid in sec.get("unit_ids") or []:
            section_of.setdefault(uid, str(sec["section"]))
            if uid not in order:
                order.append(uid)
    for uid in line_of:
        if uid not in order:
            order.append(uid)
    units = []
    for uid in order:
        line = line_of.get(uid)
        t = biosteam_type(line, uid)
        label = BIOSTEAM_LINE_ES.get(line or "", line or LABELS_ES[t])
        units.append({"id": uid, "label_es": label, "type": t, "section": section_of.get(uid, "otros"), "params": {},
                      "duties_W": {}, "metrics": {}, "biosteam_line": line})
    known = {u["id"] for u in units}
    streams, seen = [], set()
    rows = list(result.get("streams") or []) + list(internal_streams or result.get("internal_streams") or [])
    for r in rows:
        sid = str(r.get("id"))
        if sid in seen:
            continue
        seen.add(sid)
        fr, to = r.get("from_unit"), r.get("to_unit")
        fr = fr if fr in known else None
        to = to if to in known else None
        if fr is None and to is None:
            continue
        s = {"id": sid, "label_es": sid.replace("_", " "), "from": fr, "to": to, "kind": "material"}
        for k in ("flow_kg_h", "T_K", "P_Pa", "phase"):
            if r.get(k) is not None:
                s[k] = r[k]
        if r.get("mass_fractions"):
            s["composition"] = dict(r["mass_fractions"])
        if r.get("price_usd_kg") is not None:
            s["price"] = {"usd_kg": r["price_usd_kg"]}
        streams.append(s)
    connected = {s["from"] for s in streams} | {s["to"] for s in streams}
    loose = [u["id"] for u in units if u["id"] not in connected]
    if loose and not (internal_streams or result.get("internal_streams")):
        warnings.append(f"El resultado de la planta no trae las corrientes internas entre equipos: {len(loose)} equipo(s) "
                        "se dibujan sin conexión dentro de su sección. Use la vista «sections» o pase internal_streams.")
    secs = []
    for sec in result.get("sections") or []:
        sid = str(sec["section"])
        secs.append({"id": sid, "label_es": f"Área {sid}" if sid.isdigit() else sid.capitalize()})
    names_es = {"biodiesel": "Planta de biodiésel", "biodiesel_plant": "Planta de biodiésel",
                "sugarcane_ethanol": "Etanol de caña de azúcar", "corn_ethanol": "Etanol de maíz (molienda seca)",
                "cornstover_ethanol": "Etanol celulósico de rastrojo de maíz"}
    key = str(result.get("plant") or "")
    name = names_es.get(key) or names_es.get(key.replace(" ", "_")) or key.replace("_", " ").capitalize() or "Planta"
    oil = (result.get("oil_model") or {}).get("oil") if isinstance(result.get("oil_model"), dict) else None
    if oil:
        name += f" (aceite: {oil})"
    return {"name": str(name), "kind": "plant_model", "units": units, "streams": streams, "sections": secs,
            "source": str(result.get("method") or "resultado de planta de AlchemAI (BioSTEAM)")[:160], "warnings": warnings}


# -------------------------------------------------------------------------------------------- (c) hand description

def from_description(units: list[dict], streams: list[dict] | None = None, name: str | None = None,
                     sections: list[dict] | None = None) -> dict:
    """ProcessModel of a hand description in the shape of ``hazop_from_process_description``: units [{name, type, inlets,
    outlets, duty_kW?, section?, label_es?}], streams [{name, T_K?, P_Pa?, composition?, mass_flow_kg_s?}]. For a drawing
    the stream data are optional (missing ones are shown as «—»; nothing is supposed)."""
    if not units:
        raise ValueError("Da units [{name, type, inlets, outlets}]: sin equipos no hay diagrama que dibujar.")
    warnings: list[str] = []
    producer: dict[str, str] = {}
    consumer: dict[str, str] = {}
    out_units = []
    for i, u in enumerate(units):
        nm = str(u.get("name") or u.get("id") or "")
        if not nm:
            raise ValueError(f"units[{i}] no tiene «name».")
        raw = str(u.get("type") or "")
        t = DESCRIPTION_TYPE_MAP.get(raw) or DESCRIPTION_TYPE_MAP.get(raw.lower())
        if t is None:
            warnings.append(f"Equipo «{nm}»: tipo «{raw}» sin símbolo; se dibuja como genérico.")
            t = "generic"
        duties = {}
        if isinstance(u.get("duty_kW"), (int, float)):
            duties["duty"] = float(u["duty_kW"]) * 1e3
        out_units.append({"id": nm, "label_es": str(u.get("label_es") or LABELS_ES[t]), "type": t,
                          "section": str(u.get("section") or "proceso"), "params": {}, "duties_W": duties, "metrics": {}})
        for s in u.get("outlets") or []:
            if s in producer:
                raise ValueError(f"La corriente «{s}» sale de dos equipos («{producer[s]}» y «{nm}»).")
            producer[s] = nm
        for s in u.get("inlets") or []:
            if s in consumer:
                raise ValueError(f"La corriente «{s}» entra a dos equipos («{consumer[s]}» y «{nm}»).")
            consumer[s] = nm
    data = {str(s.get("name") or s.get("id")): s for s in streams or []}
    names = list(dict.fromkeys(list(data) + list(producer) + list(consumer)))
    out_streams = []
    nodata = []
    for s in names:
        d = data.get(s) or {}
        if s not in data:
            nodata.append(s)
        row = {"id": s, "label_es": str(d.get("label_es") or s), "from": producer.get(s), "to": consumer.get(s),
               "kind": d.get("kind") or "material"}
        if row["from"] is None and row["to"] is None:
            warnings.append(f"La corriente «{s}» no entra ni sale de ningún equipo: no se dibuja.")
            continue
        for k in ("T_K", "P_Pa", "flow_kg_h", "flow_m3_d", "flow_bpd"):
            if d.get(k) is not None:
                row[k] = float(d[k])
        if d.get("mass_flow_kg_s") is not None:
            row["flow_kg_h"] = float(d["mass_flow_kg_s"]) * 3600.0
        if d.get("composition"):
            row["composition"] = dict(d["composition"])
        out_streams.append(row)
    if nodata:
        warnings.append("Corrientes sin datos en «streams» (se dibujan sin flujo, T ni P): " + ", ".join(nodata[:10])
                        + (" …" if len(nodata) > 10 else ""))
    secs = sections or [{"id": s, "label_es": "Proceso" if s == "proceso" else s}
                        for s in dict.fromkeys(u["section"] for u in out_units)]
    return {"name": name or "Proceso descrito a mano", "kind": "described", "units": out_units, "streams": out_streams,
            "sections": secs, "source": "descripción a mano del usuario", "warnings": warnings}
