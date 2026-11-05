"""Deterministic HAZOP rules (CCPS guide words × parameters) per unit kind, instantiated with the solved
numbers of the process graph. Each rule yields rows; every row is tagged `source="rule"` so a reader can
tell it apart from anything an LLM later proposes.

Severity (1-5) is a preliminary ranking from inventory/energy/hazard flags, NOT a risk assessment:
  +1 flammable in the node, +1 toxic, +1 P > 15 bar, +1 T > 450 K or T < 220 K, +1 exothermic reactor,
  base 1. Capped at 5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from packages.hazop.graph import TOXIC, ProcessGraph, StreamNode, UnitNode

BAR = 1e5


@dataclass
class HazopRow:
    node: str
    kind: str
    parameter: str            # Flow, Pressure, Temperature, Level, Composition, Reaction
    deviation: str            # "More pressure", "No flow", ...
    causes: list[str]
    consequences: list[str]
    safeguards: list[str]
    severity: int
    evidence: str             # the numbers that triggered or sized the row
    source: str = "rule"

    def to_dict(self) -> dict:
        return asdict(self)


def _streams(g: ProcessGraph, names: list[str]) -> list[StreamNode]:
    return [g.streams[n] for n in names if n in g.streams]


def _flags(g: ProcessGraph, u: UnitNode) -> dict:
    ss = _streams(g, u.inlets + u.outlets)
    pmax = max([s.P_Pa for s in ss] + [0.0])
    tmax = max([s.T_K for s in ss] + [0.0])
    tmin = min([s.T_K for s in ss] + [1e9])
    return {"flammable": any(s.flammable for s in ss), "toxic": any(s.toxic for s in ss), "water": any(s.has_water for s in ss),
            "hydrate": any(s.hydrate_risk for s in ss), "pmax": pmax, "tmax": tmax, "tmin": tmin,
            "mass": max([s.mass_flow_kg_s for s in ss] + [0.0]), "pmin": min([s.P_Pa for s in ss] + [1e12]),
            "liquid": any(_is_liquid(s) for s in ss), "toxic_in": sorted({c for s in _streams(g, u.inlets) for c, x in s.mole_fractions.items()
                                                                          if x > 1e-4 and c in TOXIC})}


def _is_liquid(s: StreamNode) -> bool:
    """Known liquid (vapor fraction below 0.5). Unknown vapor fraction (NaN, e.g. a DEXPI file without it) is not liquid."""
    return s.vapor_fraction == s.vapor_fraction and s.vapor_fraction < 0.5


def _downstream(g: ProcessGraph, u: UnitNode) -> list[str]:
    return [v.name for v in g.units.values() if set(v.inlets) & set(u.outlets)]


def _blocked_in_row(u: UnitNode, f: dict, sev: int, ev: str) -> HazopRow:
    return HazopRow(u.name, u.kind, "Pressure", "More pressure (blocked-in liquid, thermal expansion)",
                    ["Sección con líquido aislada entre válvulas y calentada por sol, fuego o trazado"],
                    ["Expansión térmica del líquido atrapado: la presión sube rápido → fuga en bridas o rotura de la línea"],
                    ["Válvula de alivio térmico en cada sección aislable", "Procedimiento: drenar antes de aislar"], sev, ev)


def _severity(f: dict, exo: bool = False) -> int:
    sev = 1 + f["flammable"] + f["toxic"] + (f["pmax"] > 15 * BAR) + (f["tmax"] > 450.0 or f["tmin"] < 220.0) + exo
    return min(5, int(sev))


def _ev(u: UnitNode, f: dict) -> str:
    parts = []
    if u.P_in_Pa is not None and u.P_out_Pa is not None:
        parts.append(f"P {u.P_in_Pa / BAR:.1f}→{u.P_out_Pa / BAR:.1f} bar")
    if u.T_in_K is not None and u.T_out_K is not None:
        parts.append(f"T {u.T_in_K:.0f}→{u.T_out_K:.0f} K")
    if u.duty_kW is not None:
        parts.append(f"duty {u.duty_kW:.1f} kW")
    parts.append(f"{'inflamable' if f['flammable'] else 'no inflamable'}, {f['mass']:.2f} kg/s")
    return "; ".join(parts)


def _common_flow_rows(u: UnitNode, f: dict, sev: int, ev: str) -> list[HazopRow]:
    rows = [HazopRow(u.name, u.kind, "Flow", "No / less flow",
                     ["Válvula de bloqueo cerrada aguas arriba", "Falla de equipo aguas arriba", "Taponamiento / hidratos" if f["hydrate"] else "Taponamiento"],
                     ["Pérdida de producción", "Equipo aguas abajo fuera de especificación"] + (["Daño al equipo rotatorio por operar en vacío"] if u.kind in ("Compressor", "Pump") else []),
                     ["Alarma de bajo flujo (FAL)", "Paro por bajo flujo en rotatorios"], max(1, sev - 1), ev)]
    if f["flammable"] or f["toxic"]:
        rows.append(HazopRow(u.name, u.kind, "Flow", "Reverse flow",
                             ["Paro del equipo con presión aguas abajo mayor", "Falla de válvula check"],
                             ["Retroceso de fluido inflamable/tóxico a sección de menor presión", "Sobrepresión de equipo aguas arriba"],
                             ["Válvula check en descarga", "Interlock de paro con cierre de válvula de descarga"], sev, ev))
    if u.kind in ("Pump", "Valve", "Compressor") and (f["flammable"] or f["toxic"]):
        rows.append(HazopRow(u.name, u.kind, "Flow", "Less flow (leak / loss of containment)",
                             ["Fuga en sello, bridas o derivaciones sin tapón", "Corrosión de la línea"],
                             [f"Liberación de material {'inflamable' if f['flammable'] else 'tóxico'}: incendio o exposición del personal; pérdida de contención en la línea"],
                             ["Inspección y recorridos periódicos de la línea", "Detectores de gas", "Tapones en derivaciones y drenajes"], sev, ev))
    if u.kind in ("Pump", "Valve") and f["liquid"]:
        rows.append(_blocked_in_row(u, f, sev, ev))
    return rows


def rules_for_unit(g: ProcessGraph, u: UnitNode) -> list[HazopRow]:
    f = _flags(g, u)
    sev = _severity(f, u.exothermic)
    ev = _ev(u, f)
    k = u.kind
    rows: list[HazopRow] = []

    if k == "Compressor":
        rows += _common_flow_rows(u, f, sev, ev)
        rows.append(HazopRow(u.name, k, "Pressure", "More pressure (discharge)",
                             ["Válvula de descarga cerrada / bloqueo aguas abajo", "Falla del control antisurge"],
                             [f"Sobrepresión de la descarga por encima de {u.P_out_Pa / BAR:.1f} bar de diseño" if u.P_out_Pa else "Sobrepresión de la descarga", "Surge del compresor"],
                             ["PSV en descarga dimensionada al flujo del compresor", "Alarma PAH + paro por alta presión", "Control antisurge"], sev, ev))
        rows.append(HazopRow(u.name, k, "Temperature", "More temperature (discharge)",
                             ["Alta relación de compresión por baja presión de succión", "Falla del enfriador aguas abajo", "Baja eficiencia por desgaste"],
                             [f"T de descarga > {u.T_out_K:.0f} K de diseño; límite típico 420-450 K para lubricante/sellos" if u.T_out_K else "Alta T de descarga", "Daño a sellos y lubricante"],
                             ["Alarma TAH y paro por alta T de descarga", "Monitoreo de vibración"], sev, ev))
    elif k == "Pump":
        rows += _common_flow_rows(u, f, sev, ev)
        rows.append(HazopRow(u.name, k, "Pressure", "More pressure (deadhead)",
                             ["Válvula de descarga cerrada con bomba operando"],
                             ["Calentamiento del fluido en la carcasa, daño a sellos", "Sobrepresión de la tubería de descarga"],
                             ["Recirculación mínima", "PSV o paro por alta presión / bajo flujo"], sev, ev))
        rows.append(HazopRow(u.name, k, "Pressure", "Less pressure (suction)",
                             ["Bajo nivel en el recipiente de succión", "Filtro de succión tapado"],
                             ["Cavitación y daño al impulsor"], ["Alarma LAL en succión", "ΔP en filtro"], max(1, sev - 1), ev))
    elif k in ("Cooler", "Heater"):
        rows += _common_flow_rows(u, f, sev, ev)
        if k == "Cooler":
            rows.append(HazopRow(u.name, k, "Temperature", "More temperature (less cooling)",
                                 ["Falla de la utilidad fría (agua/aire)", "Ensuciamiento del intercambiador", "Bypass abierto"],
                                 [f"Salida por encima de {u.T_out_K:.0f} K; equipo aguas abajo fuera de diseño" if u.T_out_K else "Alta T aguas abajo",
                                  "Mayor presión en recipientes aguas abajo por mayor fracción vapor"],
                                 ["Alarma TAH en salida", "Control de T con válvula en utilidad"], sev, ev))
            # Overcooling: cold utility at full duty or low process flow drives the outlet below design.
            lt = ["Condensación de pesados / cambio de fase aguas abajo"]
            if f["hydrate"]:
                lt.insert(0, f"Formación de hidratos en la salida (agua + hidrocarburos ligeros/CO₂ a {u.T_out_K:.0f} K y {f['pmax'] / BAR:.0f} bar) → taponamiento del enfriador o del recipiente aguas abajo" if u.T_out_K else "Formación de hidratos en la salida")
            if u.T_out_K is not None and u.T_out_K < 273.15:
                lt.append("Congelamiento del agua; T por debajo de la MDMT del material")
            rows.append(HazopRow(u.name, k, "Temperature", "Less temperature (overcooling)",
                                 ["Utilidad fría a máxima capacidad (ventiladores/agua sin control)", "Bajo flujo de proceso con enfriamiento constante", "Baja T ambiente"],
                                 lt,
                                 (["Inyección de inhibidor (MEG/metanol) o deshidratación aguas arriba", "Control de T de salida con alarma TAL"] if f["hydrate"]
                                  else ["Control de T de salida con alarma TAL", "Verificar MDMT de materiales"]), sev, ev))
        else:
            rows.append(HazopRow(u.name, k, "Temperature", "More temperature (more heating)",
                                 ["Falla del control de T", "Bajo flujo de proceso con calentamiento constante"],
                                 ["Vaporización / degradación térmica", "Sobrepresión si hay bloqueo"],
                                 ["Alarma TAH y corte de calentamiento", "PSV por expansión térmica/bloqueo"], sev, ev))
    elif k in ("Vessel", "Separator", "Tank"):
        rows += _common_flow_rows(u, f, sev, ev)
        rows.append(HazopRow(u.name, k, "Level", "More level",
                             ["Falla del control de nivel / válvula de líquido cerrada", "Slug de líquido en la alimentación"],
                             ["Arrastre de líquido al gas (carryover) → daño a compresor aguas abajo" if f["flammable"] else "Arrastre de líquido al gas",
                              "Rebose"], ["LAH + LAHH con paro", "Demister"], sev, ev))
        rows.append(HazopRow(u.name, k, "Level", "Less level",
                             ["Válvula de líquido abierta en falla", "Baja alimentación"],
                             ["Paso de gas a la línea de líquido (gas blowby) → sobrepresión del sistema de baja presión aguas abajo"],
                             ["LAL + LALL con cierre de válvula de líquido", "PSV aguas abajo dimensionada para blowby"], sev, ev))
        rows.append(HazopRow(u.name, k, "Pressure", "More pressure",
                             ["Salida de gas bloqueada", "Blowby desde aguas arriba", "Fuego externo"],
                             [f"Sobrepresión del recipiente (operación {f['pmax'] / BAR:.1f} bar)"],
                             ["PSV dimensionada (bloqueo, fuego)", "PAH"], sev, ev))
        if f["liquid"]:
            rows.append(HazopRow(u.name, k, "Temperature", "More temperature",
                                 ["Alta T en la alimentación o en el almacenamiento (sol, falla de enfriamiento aguas arriba)", "Fuego externo"],
                                 ["Mayor presión de vapor del líquido → más presión en el recipiente y en las líneas conectadas"],
                                 ["TAH", "PSV dimensionada para la presión de vapor a la T máxima y para fuego"], max(1, sev - 1), ev))
        if f["water"] and f["tmin"] < 323.15:
            rows.append(HazopRow(u.name, k, "Temperature", "Less temperature (winter)",
                                 ["Baja T ambiente"],
                                 ["Congelamiento del agua en sumidero, drenajes e instrumentos → drenaje bloqueado o tubería rota"],
                                 ["Trazado eléctrico o de vapor y aislamiento en sumidero, drenajes e instrumentos"], max(1, sev - 1), ev))
        liq_out = [s for s in _streams(g, u.outlets) if _is_liquid(s)]
        if k == "Separator" and f["water"] and f["flammable"] and len(liq_out) >= 2:
            dn = _downstream(g, u)
            rows.append(HazopRow(u.name, k, "Composition", "Part of (water carryover)",
                                 ["Alto nivel de interfase: drenaje de agua tardío o alta carga de agua en la alimentación"],
                                 [f"El agua pasa aguas abajo con la fase orgánica{' (' + ', '.join(dn) + ')' if dn else ''}"],
                                 ["Alarma de alto nivel de interfase", "Drenaje automático de agua con control de interfase"], sev, ev))
        if k == "Tank" and f["toxic"] and f["pmin"] < 1.3 * BAR:
            rows.append(HazopRow(u.name, k, "Composition", "Other (toxic vapors from atmospheric tank)",
                                 ["Componente tóxico residual en el líquido almacenado"],
                                 ["Vapores tóxicos salen del tanque atmosférico al área de trabajo → exposición del personal"],
                                 ["Detectores de gas con alarma en el área", "Venteo a lugar seguro o a lavador", "Ventilación del área"], sev, ev))
    elif k == "Valve":
        rows += _common_flow_rows(u, f, sev, ev)
        rows.append(HazopRow(u.name, k, "Pressure", "More pressure (downstream)",
                             ["Válvula abierta en falla con aguas abajo de menor presión de diseño"],
                             [f"Sistema aguas abajo expuesto a {u.P_in_Pa / BAR:.1f} bar de aguas arriba" if u.P_in_Pa else "Sobrepresión aguas abajo"],
                             ["PSV aguas abajo dimensionada para válvula totalmente abierta", "Clase de presión homogénea"], sev, ev))
        if u.T_out_K is not None and u.T_in_K is not None and u.T_out_K < u.T_in_K - 5:
            lt = []
            if f["hydrate"]:
                lt.append("Formación de hidratos (agua + hidrocarburos ligeros/CO₂ a baja T y alta P)")
            if u.T_out_K < 244.0:
                lt.append("T por debajo de −29 °C: acero al carbono fuera de rango (fragilización)")
            rows.append(HazopRow(u.name, k, "Temperature", "Less temperature (Joule-Thomson)",
                                 ["Expansión isoentálpica normal; mayor ΔP por baja presión aguas abajo"],
                                 [f"Salida a {u.T_out_K:.1f} K (ΔT = {u.T_out_K - u.T_in_K:+.1f} K)"] + lt,
                                 ["Inyección de inhibidor / deshidratación aguas arriba" if f["hydrate"] else "Verificar MDMT de materiales", "TAL aguas abajo"],
                                 sev, ev))
    elif k == "HeatExchanger":
        rows += _common_flow_rows(u, f, sev, ev)
        rows.append(HazopRow(u.name, k, "Composition", "Other (tube leak)",
                             ["Fuga tubo-coraza por corrosión o vibración"],
                             ["Contaminación del lado de baja presión con el de alta", "Sobrepresión del lado de baja presión"],
                             ["PSV en el lado de baja presión (caso rotura de tubo)", "Monitoreo de composición / presión"], sev, ev))
        rows.append(HazopRow(u.name, k, "Temperature", "Less heat transfer",
                             ["Ensuciamiento", "Bajo flujo de un lado"],
                             ["Salidas fuera de especificación de temperatura"], ["TAH/TAL en salidas", "Programa de limpieza"], max(1, sev - 1), ev))
        ins = _streams(g, u.inlets)
        if len(ins) >= 2:
            cold = min(ins, key=lambda s: s.T_K)
            if _is_liquid(cold):
                rows.append(HazopRow(u.name, k, "Pressure", "More pressure (cold side blocked in)",
                                     ["Válvulas del lado frío cerradas (equipo en espera o mantenimiento) con el lado caliente en servicio"],
                                     ["El líquido atrapado se calienta: expansión térmica y vaporización → sobrepresión y ruptura"]
                                     + (["BLEVE si el líquido es un gas licuado inflamable"] if cold.flammable else []),
                                     ["PSV del lado frío que no pueda aislarse (válvulas candado-abierto)",
                                      "Procedimiento: aislar primero el lado caliente y drenar el lado frío"], min(5, sev + cold.flammable), ev))
    elif k.startswith("RCT"):
        rows += _common_flow_rows(u, f, sev, ev)
        if u.exothermic:
            rows.append(HazopRow(u.name, k, "Temperature", "More temperature (runaway)",
                                 ["Pérdida de enfriamiento", "Exceso de reactivo limitante", "Falla del control de T"],
                                 [f"Reacción exotérmica ({u.duty_kW:.0f} kW retirados en diseño): aumento de T y P, posible descontrol" if u.duty_kW else "Descontrol térmico",
                                  "Reacciones secundarias / descomposición"],
                                 ["TAHH con paro de alimentación", "Sistema de enfriamiento de emergencia", "PSV/disco de ruptura dimensionado para runaway", "Quench"], sev, ev))
        rows.append(HazopRow(u.name, k, "Composition", "Wrong ratio / composition",
                             ["Falla del control de relación de alimentación", "Contaminación de la alimentación"],
                             ["Baja conversión", "Mezcla inflamable dentro del reactor" if f["flammable"] else "Producto fuera de especificación", "Generación de subproductos"]
                             + ([f"Exceso de {', '.join(f['toxic_in'])}: reactivo tóxico sin reaccionar pasa aguas abajo"
                                 + (f" ({', '.join(_downstream(g, u))})" if _downstream(g, u) else "") + " y puede liberarse"] if f["toxic_in"] else []),
                             ["Analizador en línea / control de relación", "Interlock por relación fuera de rango"], sev, ev))
        rows.append(HazopRow(u.name, k, "Pressure", "More pressure",
                             ["Generación de gas por reacción", "Salida bloqueada", "Runaway"],
                             ["Sobrepresión del reactor"], ["PSV / disco de ruptura", "PAH con paro"], sev, ev))
        if u.exothermic:
            rows.append(HazopRow(u.name, k, "Temperature", "Less temperature (more cooling)",
                                 ["Válvula del refrigerante abierta de más", "Falla del control de T"],
                                 ["Baja T: se acumula reactivo sin reaccionar, que puede descontrolarse al recuperar la T"],
                                 ["TAL con corte de alimentación", "Control de T en cascada sobre el refrigerante"], sev, ev))
            rows.append(HazopRow(u.name, k, "Flow", "Reverse flow (cooling medium)",
                                 ["Falla de la fuente de agua de enfriamiento con el retorno a mayor presión"],
                                 ["Menos enfriamiento del reactor → aumento de T, posible descontrol"],
                                 ["Válvula check en la entrada del refrigerante", "FAL en el refrigerante con alarma"], sev, ev))
        if k in ("RCT_CSTR", "RCT_Batch"):
            rows.append(HazopRow(u.name, k, "Agitation", "No agitation",
                                 ["Falla del motor o del acoplamiento del agitador"],
                                 (["Sin mezcla: acumulación de reactivo y puntos calientes; descontrol al reanudar la agitación"] if u.exothermic
                                  else ["Sin mezcla: baja conversión, producto fuera de especificación"]),
                                 ["Interlock: sin agitación se corta la alimentación", "Indicación de corriente/velocidad del agitador"], sev, ev))
    elif k in ("ShortcutColumn", "DistillationColumn", "AbsorptionColumn"):
        rows += _common_flow_rows(u, f, sev, ev)
        rows.append(HazopRow(u.name, k, "Pressure", "More pressure",
                             ["Falla del condensador (utilidad fría)", "Salida de vapor bloqueada", "Fuego externo"],
                             ["Sobrepresión de la columna; alivio de vapor inflamable" if f["flammable"] else "Sobrepresión de la columna"],
                             ["PSV en el domo dimensionada para falla de condensador", "PAH con corte de rehervidor"], sev, ev))
        rows.append(HazopRow(u.name, k, "Temperature", "More temperature (reboiler)",
                             ["Falla del control de vapor al rehervidor"],
                             ["Inundación, arrastre de pesados al destilado", "Degradación térmica de fondos"],
                             ["TAH en fondos", "Control de ΔP de la columna"], max(1, sev - 1), ev))
        rows.append(HazopRow(u.name, k, "Level", "More level (bottoms)",
                             ["Salida de fondos cerrada o válvula de nivel que falla cerrada", "Transmisor de nivel con lectura falsa"],
                             ["Llenado de la torre: líquido a la línea de domo y al sistema de alivio"]
                             + (["Liberación de líquido inflamable por el alivio"] if f["flammable"] else []),
                             ["Medición de nivel independiente con LAHH", "Balance de entradas y salidas durante el arranque"], sev, ev))
        rows.append(HazopRow(u.name, k, "Flow", "Less reflux",
                             ["Falla de la bomba de reflujo", "Bajo nivel en el acumulador"],
                             ["Pérdida de pureza del destilado"], ["FAL en reflujo", "LAL en acumulador"], max(1, sev - 1), ev))
    elif k in ("Mixer", "Splitter", "OT_Recycle", "NodeIn", "NodeOut"):
        is_mixer = k in ("Mixer", "NodeIn")        # DWSIM names the mixer ObjectType "NodeIn"
        if is_mixer:
            ps = [s.P_Pa for s in _streams(g, u.inlets)]
            if ps and (max(ps) - min(ps)) > 0.5 * BAR:
                rows.append(HazopRow(u.name, k, "Pressure", "Different inlet pressures",
                                     [f"Entradas a {min(ps) / BAR:.1f} y {max(ps) / BAR:.1f} bar"],
                                     ["Flujo inverso hacia la línea de menor presión si falla el control"],
                                     ["Válvulas check en cada entrada", "Control de presión de la entrada de baja"], sev, ev))
        if is_mixer and f["flammable"] and any(s.mole_fractions.get("Oxygen", 0) > 0.01 for s in _streams(g, u.inlets)):
            rows.append(HazopRow(u.name, k, "Composition", "Flammable mixture",
                                 ["Mezcla de combustible y oxidante antes del reactor/quemador"],
                                 ["Mezcla dentro de límites de inflamabilidad en tubería; retroceso de llama"],
                                 ["Control de relación aire/combustible", "Arrestallamas", "Analizador de O₂"], 5, ev))
    else:
        rows += _common_flow_rows(u, f, sev, ev)
    return rows


def hazop_rows(g: ProcessGraph) -> list[HazopRow]:
    rows: list[HazopRow] = []
    for u in g.units.values():
        rows += rules_for_unit(g, u)
    return rows
