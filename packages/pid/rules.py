"""Deterministic instrumentation rules for a minimal P&ID (bloque M10).

Input: the solved `ProcessGraph` (from DWSIM or from DEXPI Process) and its deterministic HAZOP rows
(`packages.hazop.rules.hazop_rows`). Output: a `PidSpec` with lines, piping items (control, check, relief and
isolation valves) and instrument functions, each traced to the rule and, when the HAZOP caused it, to the HAZOP rows.

Two kinds of rules:
* control-structure rules (R-*): inventory, flow and temperature loops that do not depend on the HAZOP;
* HAZOP rules (H-*): relief valves, check valves, alarms and trips, placed only where a HAZOP row asks for them.

Nothing here invents a design value: setpoints, alarm limits, PSV set pressures and relief loads stay
"por definir"; the sizing tools (`relief_valve_sizing`, `control_valve_sizing`, `pid_tuning`) fill them later with
the user's data. Tags follow ISA-5.1 letters; manual block valves and check valves are piping items that ISA-5.1
does not tag, so they get the internal tags BV-nn and CHK-nn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from packages.export.flowsheet_export import PASS_THROUGH, export_graph
from packages.hazop.graph import ProcessGraph, StreamNode, UnitNode
from packages.hazop.rules import HazopRow, _is_liquid, hazop_rows

TBD = "por definir (dato del usuario)"
ALARM_SEVERITY = 3        # preliminary HAZOP severity (1-5) from which a deviation gets an alarm
TRIP_SEVERITY = 4         # ... and from which it also gets an independent trip (SIS)

LUYBEN = ("Luyben, Tyréus y Luyben (1998), Plantwide Process Control, McGraw-Hill: procedimiento de 9 pasos, "
          "paso 6 (fijar un flujo en cada reciclo y controlar inventarios: niveles y presiones)")
LUYBEN8 = "Luyben, Tyréus y Luyben (1998), Plantwide Process Control: paso 8 (control de cada operación unitaria)"
SKO97 = ("Skogestad (1997), Dynamics and control of distillation columns — a tutorial introduction, Trans IChemE 75A, "
         "Fig. 1: configuración LV (VT → presión, D → nivel del acumulador MD, B → nivel de fondos MB)")
SKO07 = ("Skogestad (2007), The dos and don'ts of distillation column control, Chem Eng Res Des 85: "
         "configuración LV con un lazo rápido de temperatura (recomendada para la mayoría de las columnas)")
TOWLER = ("Towler y Sinnott (2013), Chemical Engineering Design, 2ª ed., cap. 5 (instrumentación y control; "
          "cita a nivel de capítulo)")
SEBORG_CASCADE = "Seborg, Edgar, Mellichamp y Doyle, Process Dynamics and Control: control en cascada (lazo maestro → consigna del esclavo)"
API521 = ("API STD 521 (causas de sobrepresión y escenarios de alivio) + API STD 520 (dimensionamiento; tool "
          "relief_valve_sizing) + ASME BPVC VIII-1 (protección contra sobrepresión de todo recipiente a presión)")
ISA182 = "ANSI/ISA-18.2 (gestión de alarmas): una alarma por desviación con consecuencia y acción del operador"
IEC61511 = ("IEC 61511-1 / ISA-84: función instrumentada de seguridad con sensor independiente del BPCS; "
            "SIL por LOPA (pendiente)")
ISA51 = "ANSI/ISA-5.1-2009 (letras de identificación de instrumentos)"

RULES: dict[str, tuple[str, str]] = {
    "R-L1": ("Todo recipiente con inventario de líquido lleva un lazo de nivel que manipula su salida de líquido "
             "(una por fase líquida; la segunda es de interfase)", LUYBEN),
    "R-P1": ("Todo recipiente con espacio de gas y salida de vapor lleva control de presión sobre esa salida", LUYBEN),
    "R-C1": ("Columna: presión con el condensador (VT), nivel del acumulador con el destilado D, nivel de fondos con B",
             SKO97),
    "R-C2": ("Columna de destilación: lazo de temperatura de plato con el rehervidor (V)", SKO07),
    "R-F1": ("Bomba: control de flujo con válvula en la descarga (bomba centrífuga; una de desplazamiento positivo "
             "se controla con recirculación o velocidad)", TOWLER),
    "R-F2": ("Compresor: control de flujo (capacidad) en la descarga; el elemento final real (velocidad, álabes, "
             "recirculación) lo decide el ingeniero", TOWLER),
    "R-T1": ("Calentador o enfriador: control de temperatura de salida del proceso con la utilidad", LUYBEN8),
    "R-T2": ("Reactor con carga térmica: control de temperatura con el medio de enfriamiento o calentamiento", LUYBEN8),
    "R-K1": ("Si la salida de un recipiente alimenta una bomba o un compresor con lazo de flujo, el lazo de inventario "
             "fija la consigna de ese flujo (cascada) en vez de poner otra válvula", SEBORG_CASCADE),
    "R-V1": ("Una válvula del flowsheet es el elemento final del lazo que necesita válvula en esa línea; si ningún "
             "lazo la reclama queda como válvula manual HV", ISA51),
    "R-I1": ("Válvula de aislamiento en cada límite de batería (alimentación y producto) y en succión y descarga de "
             "equipo rotatorio", TOWLER),
    "H-PSV": ("Dispositivo de alivio donde una fila del HAZOP pide PSV, alivio térmico o disco de ruptura; un "
              "dispositivo por punto protegido, con todos sus escenarios", API521),
    "H-CHK": ("Válvula check donde una fila del HAZOP la pide (flujo inverso o entradas a presiones distintas)",
              "Salvaguarda propuesta por la fila del HAZOP (packages/hazop/rules.py)"),
    "H-ALM": (f"Alarma alta o baja para cada desviación de nivel, presión, temperatura o flujo con severidad ≥ "
              f"{ALARM_SEVERITY}", ISA182),
    "H-TRIP": (f"Disparo con sensor independiente para cada desviación de nivel, presión, temperatura o flujo con "
               f"severidad ≥ {TRIP_SEVERITY}", IEC61511),
}

VESSELS = {"Vessel", "Separator", "Tank"}
REACTORS_WITH_INVENTORY = {"RCT_CSTR", "RCT_Batch", "RCT_Conversion", "RCT_Equilibrium", "RCT_Gibbs"}
COLUMNS = {"DistillationColumn", "ShortcutColumn"}
ABSORBERS = {"AbsorptionColumn"}
INLINE = {"Valve", "Mixer", "NodeIn", "Splitter", "NodeOut"}     # piping components, not equipment, on the P&ID
ROTATING = {"Pump", "Compressor"}
VAR = {"Level": "L", "Pressure": "P", "Temperature": "T", "Flow": "F"}
VAR_ES = {"L": "nivel", "P": "presión", "T": "temperatura", "F": "flujo"}
# HAZOP deviations that are handled by relief, check valves or procedures, not by a process alarm
NOT_ALARMED = ("reverse", "leak", "blocked in", "blocked-in", "different inlet", "hammer")


@dataclass
class Line:
    name: str                    # first stream name, or "util-<unit>-<medium>" / "alivio-<tag>"
    kind: str                    # "proceso" | "utilidad" | "alivio"
    src: str | None              # unit name; None = battery limit / utility header / relief header
    dst: str | None
    streams: list[str] = field(default_factory=list)
    T_K: float | None = None
    P_Pa: float | None = None
    vapor_fraction: float | None = None
    note: str = ""


@dataclass
class Item:
    """A piping item on a line (rank orders it from source to target) or a relief device on a unit."""
    tag: str
    isa: str                     # BV, CHK, LV/PV/FV/TV, HV, PSV
    category: str                # aislamiento | check | valvula_control | valvula_manual | alivio
    line: str | None
    unit: str | None             # protected / served unit
    rank: int
    rule: str
    hazop: list[str] = field(default_factory=list)
    origin: str = "regla"        # regla | hazop | propuesta
    detail: str = ""
    dwsim: str | None = None     # flowsheet valve this item is
    setpoint: str = ""           # PSV set pressure etc.


@dataclass
class Instrument:
    """One ISA-5.1 instrument function (controller, alarm, trip or indicator) with its transmitter."""
    tag: str                     # LIC-101, LAH-101, LSHH-501
    sensor: str                  # LT-101
    variable: str                # L P T F
    function: str                # IC, AH, AL, SHH, SLL, I
    loop: str                    # loop number "101"
    category: str                # lazo | alarma | disparo
    unit: str
    sensing_line: str | None     # None = on the unit (instrument nozzle)
    final_element: str           # item tag, "cascada → FIC-1xx", or a text
    rule: str
    hazop: list[str] = field(default_factory=list)
    origin: str = "regla"
    detail: str = ""
    setpoint: str = TBD
    cascade_to: str | None = None
    action: str = ""             # trip action from the HAZOP safeguard


@dataclass
class PidSpec:
    name: str
    units: dict[str, str]                       # equipment + inline units: name -> DWSIM kind
    lines: dict[str, Line]
    items: list[Item]
    instruments: list[Instrument]
    hazop_rows: list[dict]
    not_instrumented: list[dict]                # HAZOP rows with a safeguard this tool does not draw, and why
    undetermined: list[str]                     # rules that could not be decided (e.g. unknown phase)
    assumptions: list[str]

    def to_dict(self) -> dict:
        return {"name": self.name, "units": self.units, "lines": {k: asdict(v) for k, v in self.lines.items()},
                "items": [asdict(i) for i in self.items], "instruments": [asdict(i) for i in self.instruments],
                "not_instrumented": self.not_instrumented, "undetermined": self.undetermined,
                "assumptions": self.assumptions}


def row_ref(i: int, r: HazopRow) -> str:
    return f"HAZOP #{i + 1}: {r.node} · {r.parameter} · {r.deviation} (sev {r.severity})"


class _Builder:
    def __init__(self, g: ProcessGraph, rows: list[HazopRow], alarm_sev: int, trip_sev: int):
        self.g, self.rows, self.alarm_sev, self.trip_sev = g, rows, alarm_sev, trip_sev
        self.units = {u.name: u.kind for u in g.units.values() if u.kind not in PASS_THROUGH}
        self.lines: dict[str, Line] = {}
        self.items: list[Item] = []
        self.instruments: list[Instrument] = []
        self.not_instrumented: list[dict] = []
        self.undetermined: list[str] = []
        self.loop_no = 100
        self.sis_no = 500
        self.n = {"BV": 0, "CHK": 0, "PSV": 300, "HV": 0}
        self.claimed: dict[str, str] = {}      # flowsheet valve -> instrument tag
        self.flow_loop: dict[str, Instrument] = {}   # rotating unit -> its FIC
        self._lines_from_graph()

    # ------------------------------------------------------------- topology
    def _lines_from_graph(self):
        eg = export_graph(self.g)
        dw = {n: a["dwsim"] for n, a in eg.nodes.items()}
        for e in eg.edges:
            s: StreamNode = e.data
            self.lines[e.streams[0]] = Line(e.streams[0], "proceso", dw.get(e.src), dw.get(e.dst), list(e.streams),
                                            s.T_K, s.P_Pa, s.vapor_fraction)

    def outlets(self, unit: str) -> list[Line]:
        return [l for l in self.lines.values() if l.kind == "proceso" and l.src == unit]

    def inlets(self, unit: str) -> list[Line]:
        return [l for l in self.lines.values() if l.kind == "proceso" and l.dst == unit]

    def utility(self, unit: str, medium: str, note: str) -> Line:
        name = f"util-{unit}-{medium}"
        if name not in self.lines:
            self.lines[name] = Line(name, "utilidad", None, unit, note=note)
        return self.lines[name]

    # ------------------------------------------------------------- tags
    def next_loop(self) -> str:
        self.loop_no += 1
        return str(self.loop_no)

    def next_sis(self) -> str:
        self.sis_no += 1
        return str(self.sis_no)

    def piping_tag(self, isa: str) -> str:
        self.n[isa] += 1
        return f"{isa}-{self.n[isa]:02d}" if isa in ("BV", "CHK", "HV") else f"{isa}-{self.n[isa]}"

    def add_item(self, **kw) -> Item:
        it = Item(**kw)
        self.items.append(it)
        return it

    # ------------------------------------------------------------- final elements
    def final_element(self, line: Line, var: str, loop: str, rule: str, unit: str) -> str:
        """Valve for a loop acting on a process line: reuse a flowsheet valve the line runs into, cascade to the
        flow loop of a pump/compressor it feeds, or add a new control valve."""
        dst_kind = self.units.get(line.dst or "", "")
        if dst_kind in ROTATING and line.dst in self.flow_loop:
            return f"cascada → {self.flow_loop[line.dst].tag}"
        if dst_kind == "Valve" and line.dst not in self.claimed:
            tag = f"{var}V-{loop}"
            self.claimed[line.dst] = tag
            self.add_item(tag=tag, isa=f"{var}V", category="valvula_control", line=line.name, unit=unit, rank=10,
                          rule="R-V1", dwsim=line.dst, detail=f"válvula {line.dst} del flowsheet como elemento final de {var}IC-{loop} ({rule})")
            return tag
        tag = f"{var}V-{loop}"
        self.add_item(tag=tag, isa=f"{var}V", category="valvula_control", line=line.name, unit=unit, rank=4, rule=rule,
                      detail=f"válvula de control de {VAR_ES[var]} en la línea {line.name}")
        return tag

    def loop(self, var: str, unit: str, rule: str, detail: str, sensing_line: str | None, final: str | None = None,
             line_for_valve: Line | None = None) -> Instrument:
        no = self.next_loop()
        fe = final or (self.final_element(line_for_valve, var, no, rule, unit) if line_for_valve else TBD)
        ins = Instrument(tag=f"{var}IC-{no}", sensor=f"{var}T-{no}", variable=var, function="IC", loop=no,
                         category="lazo", unit=unit, sensing_line=sensing_line, final_element=fe, rule=rule, detail=detail)
        if fe.startswith("cascada → "):
            ins.cascade_to = fe.split("→ ")[1]
        self.instruments.append(ins)
        return ins

    # ------------------------------------------------------------- control-structure rules
    def control_rules(self):
        g = self.g
        for u in g.units.values():          # flow loops first: inventory loops cascade onto them
            if u.kind in ROTATING:
                outs = self.outlets(u.name)
                if not outs:
                    continue
                d = outs[0]
                rule = "R-F1" if u.kind == "Pump" else "R-F2"
                ins = self.loop("F", u.name, rule, f"flujo de descarga de {u.name}", d.name, line_for_valve=d)
                if u.kind == "Compressor":
                    ins.detail += "; FV es un marcador: el elemento final (velocidad, álabes, recirculación) lo decide el ingeniero"
                self.flow_loop[u.name] = ins
        for u in g.units.values():
            k = u.kind
            if k in VESSELS or k in REACTORS_WITH_INVENTORY:
                self._vessel(u)
            elif k in COLUMNS or k in ABSORBERS:
                self._column(u)
            elif k in ("Heater", "Cooler"):
                outs = self.outlets(u.name)
                medium = "calentamiento" if k == "Heater" else "enfriamiento"
                ut = self.utility(u.name, medium, f"medio de {medium} de {u.name} (utilidad no modelada en el flowsheet)")
                tv = f"TV-{self.loop_no + 1}"
                self.add_item(tag=tv, isa="TV", category="valvula_control", line=ut.name, unit=u.name, rank=4, rule="R-T1",
                              detail=f"válvula del medio de {medium}")
                self.loop("T", u.name, "R-T1", f"temperatura de salida de {u.name}", outs[0].name if outs else None, final=tv)
            if k.startswith("RCT") and (u.exothermic or (u.duty_kW not in (None, 0.0) and abs(u.duty_kW) > 1e-6)):
                medium = "enfriamiento" if (u.exothermic or (u.duty_kW or 0) < 0) else "calentamiento"
                ut = self.utility(u.name, medium, f"medio de {medium} del reactor {u.name} (utilidad no modelada)")
                tv = f"TV-{self.loop_no + 1}"
                self.add_item(tag=tv, isa="TV", category="valvula_control", line=ut.name, unit=u.name, rank=4, rule="R-T2",
                              detail=f"válvula del medio de {medium} del reactor")
                self.loop("T", u.name, "R-T2", f"temperatura del reactor {u.name}", None, final=tv)
        # flowsheet valves nobody claimed stay as hand valves
        for name, kind in self.units.items():
            if kind == "Valve" and name not in self.claimed:
                tag = self.piping_tag("HV")
                self.claimed[name] = tag
                ins = self.inlets(name)
                self.add_item(tag=tag, isa="HV", category="valvula_manual", line=ins[0].name if ins else None, unit=name,
                              rank=10, rule="R-V1", dwsim=name, detail=f"válvula {name} del flowsheet sin lazo asignado")

    def _vessel(self, u: UnitNode):
        outs = self.outlets(u.name)
        known = [l for l in outs if l.vapor_fraction == l.vapor_fraction and l.vapor_fraction is not None]
        if len(known) < len(outs):
            self.undetermined.append(f"{u.name}: fase de salida desconocida; no se decide el lazo de nivel/presión (R-L1, R-P1)")
        liq = [l for l in known if l.vapor_fraction < 0.5]
        vap = [l for l in known if l.vapor_fraction >= 0.5]
        if u.kind.startswith("RCT") and not liq:
            return                                           # gas-phase reactor: no liquid inventory
        liq_in = [l for l in self.inlets(u.name) if l.vapor_fraction == l.vapor_fraction and l.vapor_fraction is not None
                  and l.vapor_fraction < 0.5]
        if liq:
            water = lambda x: self.g.streams[x.name].mole_fractions.get("Water", 0.0) if x.name in self.g.streams else 0.0
            for i, l in enumerate(sorted(liq, key=lambda x: (water(x), x.name))):     # the aqueous phase is the interface
                what = "nivel" if i == 0 else "nivel de interfase"
                self.loop("L", u.name, "R-L1", f"{what} de {u.name}; manipula la salida de líquido {l.name}", None,
                          line_for_valve=l)
        elif liq_in and u.kind in VESSELS:
            self.loop("L", u.name, "R-L1", f"nivel de {u.name} sin salida de líquido: manipula la alimentación {liq_in[0].name}",
                      None, line_for_valve=liq_in[0])
        for l in vap:
            if (l.P_Pa or 0) < 1.3e5 and u.kind == "Tank":
                continue                                     # atmospheric tank: vented, no pressure loop
            self.loop("P", u.name, "R-P1", f"presión de {u.name}; manipula la salida de vapor {l.name}", None, line_for_valve=l)

    def _column(self, u: UnitNode):
        outs = sorted(self.outlets(u.name), key=lambda l: (l.T_K if l.T_K == l.T_K and l.T_K is not None else 0.0))
        if len(outs) < 2:
            self.undetermined.append(f"{u.name}: columna con menos de dos productos; no se aplica R-C1")
            return
        top, bottom = outs[0], outs[-1]
        if u.kind in COLUMNS:
            ut = self.utility(u.name, "condensador", f"agua de enfriamiento del condensador de {u.name} (utilidad no modelada)")
            pv = f"PV-{self.loop_no + 1}"
            self.add_item(tag=pv, isa="PV", category="valvula_control", line=ut.name, unit=u.name, rank=4, rule="R-C1",
                          detail="válvula del refrigerante del condensador (VT)")
            self.loop("P", u.name, "R-C1", f"presión de domo de {u.name} con el condensador (VT)", None, final=pv)
            self.loop("L", u.name, "R-C1", f"nivel del acumulador de reflujo (MD) de {u.name} con el destilado D ({top.name}); "
                      "el acumulador va dentro del bloque de columna", None, line_for_valve=top)
            self.loop("L", u.name, "R-C1", f"nivel de fondos (MB) de {u.name} con el producto de fondos B ({bottom.name})",
                      None, line_for_valve=bottom)
            ut = self.utility(u.name, "rehervidor", f"vapor de calentamiento del rehervidor de {u.name} (utilidad no modelada)")
            tv = f"TV-{self.loop_no + 1}"
            self.add_item(tag=tv, isa="TV", category="valvula_control", line=ut.name, unit=u.name, rank=4, rule="R-C2",
                          detail="válvula del medio de calentamiento del rehervidor (V)")
            self.loop("T", u.name, "R-C2", f"temperatura de un plato sensible de {u.name} con el rehervidor (V); plato por elegir",
                      None, final=tv)
        else:
            self.loop("P", u.name, "R-P1", f"presión de domo de {u.name} con la salida de gas {top.name}", None, line_for_valve=top)
            self.loop("L", u.name, "R-L1", f"nivel de fondos de {u.name} con el líquido {bottom.name}", None, line_for_valve=bottom)

    def isolation_rules(self):
        for l in list(self.lines.values()):
            if l.kind != "proceso":
                continue
            if l.src is None:
                self.add_item(tag=self.piping_tag("BV"), isa="BV", category="aislamiento", line=l.name, unit=l.dst, rank=0,
                              rule="R-I1", detail=f"aislamiento en el límite de batería (alimentación {l.name})")
            if l.dst is None:
                self.add_item(tag=self.piping_tag("BV"), isa="BV", category="aislamiento", line=l.name, unit=l.src, rank=9,
                              rule="R-I1", detail=f"aislamiento en el límite de batería (producto {l.name})")
            if self.units.get(l.src or "") in ROTATING:
                self.add_item(tag=self.piping_tag("BV"), isa="BV", category="aislamiento", line=l.name, unit=l.src, rank=2,
                              rule="R-I1", detail=f"aislamiento de la descarga de {l.src}")
            if self.units.get(l.dst or "") in ROTATING:
                self.add_item(tag=self.piping_tag("BV"), isa="BV", category="aislamiento", line=l.name, unit=l.dst, rank=8,
                              rule="R-I1", detail=f"aislamiento de la succión de {l.dst}")

    # ------------------------------------------------------------- HAZOP rules
    def hazop_rules(self):
        for i, r in enumerate(self.rows):
            ref = row_ref(i, r)
            origin = "hazop" if r.source == "rule" else "propuesta"
            sg = " | ".join(r.safeguards).lower()
            drawn = False
            if any(k in sg for k in ("psv", "alivio", "disco de ruptura")):
                self._relief(r, ref, origin, sg)
                drawn = True
            if "check" in sg:
                self._check(r, ref, origin, sg)
                drawn = True
            if self._alarm_trip(r, ref, origin):
                drawn = True
            iface = next((x for x in self.instruments if x.unit == r.node and "interfase" in x.detail), None)
            if not drawn and "interfase" in sg and iface is not None:   # answered by the R-L1 interface loop
                iface.hazop.append(ref)
                drawn = True
            if not drawn:
                self.not_instrumented.append({"hazop": ref, "safeguards": r.safeguards, "razon": self._why_not(r)})

    def _why_not(self, r: HazopRow) -> str:
        if r.parameter not in VAR:
            return f"parámetro {r.parameter}: necesita analizador o procedimiento; lo decide el ingeniero"
        if any(k in r.deviation.lower() for k in NOT_ALARMED):
            return "se cubre con procedimiento, detección de gas o interlock; no es una alarma de proceso"
        return f"severidad {r.severity} < {self.alarm_sev}: salvaguardas existentes (lazo de control, procedimiento)"

    def _psv(self, key: tuple, unit: str, line: str | None, rank: int, ref: str, origin: str, scenario: str):
        for it in self.items:
            if it.isa == "PSV" and (it.unit, it.line, it.rank) == key[1:]:
                if ref not in it.hazop:
                    it.hazop.append(ref)
                    it.detail += f"; {scenario}"
                return
        self.add_item(tag=self.piping_tag("PSV"), isa="PSV", category="alivio", line=line, unit=unit, rank=rank,
                      rule="H-PSV", hazop=[ref], origin=origin, detail=f"escenarios: {scenario}",
                      setpoint=f"presión de ajuste {TBD} (MAWP del equipo); carga de alivio con relief_valve_sizing")

    def _relief(self, r: HazopRow, ref: str, origin: str, sg: str):
        u = r.node
        kind = self.units.get(u, r.kind)
        scen = r.deviation
        if kind in ROTATING or ("aguas abajo" in sg and kind == "Valve") or "alivio térmico" in sg or kind == "Valve":
            for l in self.outlets(u) or self.inlets(u):
                self._psv(("PSV", u, l.name, 3), u, l.name, 3, ref, origin, scen)
            return
        if "aguas abajo" in sg:                       # gas blowby through the liquid outlet
            liq = [l for l in self.outlets(u) if l.vapor_fraction is not None and l.vapor_fraction == l.vapor_fraction
                   and l.vapor_fraction < 0.5]
            if kind in COLUMNS or kind in ABSORBERS:      # blowby leaves through the bottoms only
                liq = sorted(self.outlets(u), key=lambda x: x.T_K or 0.0)[-1:]
            for l in liq or self.outlets(u):
                if self.units.get(l.dst or "") == "Valve":
                    nxt = self.outlets(l.dst)
                    if nxt:
                        self._psv(("PSV", u, nxt[0].name, 3), u, nxt[0].name, 3, ref, origin, scen + " (aguas abajo de la válvula)")
                        continue
                self._psv(("PSV", u, l.name, 5), u, l.name, 5, ref, origin, scen + " (aguas abajo de la válvula de nivel)")
            return
        self._psv(("PSV", u, None, 0), u, None, 0, ref, origin, scen)

    def _check(self, r: HazopRow, ref: str, origin: str, sg: str):
        u = r.node
        if "refrigerante" in sg:
            lines = [self.utility(u, "enfriamiento", f"medio de enfriamiento de {u} (utilidad no modelada)")]
            rank = 2
        elif "cada entrada" in sg:
            lines, rank = self.inlets(u), 7
        else:
            lines, rank = self.outlets(u), 1
        for l in lines:
            ex = [it for it in self.items if it.isa == "CHK" and it.line == l.name]
            if ex:
                if ref not in ex[0].hazop:
                    ex[0].hazop.append(ref)
                continue
            self.add_item(tag=self.piping_tag("CHK"), isa="CHK", category="check", line=l.name, unit=u, rank=rank,
                          rule="H-CHK", hazop=[ref], origin=origin, detail=f"evita flujo inverso en {l.name}")

    def _alarm_trip(self, r: HazopRow, ref: str, origin: str) -> bool:
        var = VAR.get(r.parameter)
        dev = r.deviation.lower()
        if var is None or r.severity < self.alarm_sev or any(k in dev for k in NOT_ALARMED):
            return False
        hl = "H" if dev.startswith(("more", "high")) else ("L" if dev.startswith(("less", "no")) else None)
        if hl is None:
            return False
        u = r.node
        line = None
        if "(discharge)" in dev:
            line = (self.outlets(u) or [None])[0]
        elif "(suction)" in dev or (var == "F" and "heating medium" not in dev):
            line = (self.inlets(u) or [None])[0]
        elif "heating medium" in dev:
            line = self.lines.get(f"util-{u}-rehervidor")
        if line is None and self.units.get(u) in INLINE:      # a valve or tee has no body to mount a sensor on
            line = (self.outlets(u) or self.inlets(u) or [None])[0]
        line_name = line.name if line else None
        where = "fondos" if "(bottoms)" in dev else ""          # a column has two level loops: pick the right one
        host = next((x for x in self.instruments if x.category == "lazo" and x.unit == u and x.variable == var
                     and where in x.detail and (line_name is None or x.sensing_line in (None, line_name))), None)
        if host is None:
            host = next((x for x in self.instruments if x.category == "indicador" and x.unit == u and x.variable == var
                         and x.sensing_line == line_name), None)
        if host is None:
            no = self.next_loop()
            host = Instrument(tag=f"{var}I-{no}", sensor=f"{var}T-{no}", variable=var, function="I", loop=no,
                              category="indicador", unit=u, sensing_line=line_name, final_element="—", rule="H-ALM",
                              hazop=[ref], origin=origin, detail=f"transmisor de {VAR_ES[var]} {('en la línea ' + line_name) if line_name else 'en ' + u} para la alarma", setpoint="—")
            self.instruments.append(host)
        self._merge(Instrument(tag=f"{var}A{hl}-{host.loop}", sensor=host.sensor, variable=var, function=f"A{hl}",
                               loop=host.loop, category="alarma", unit=u, sensing_line=host.sensing_line, final_element="operador",
                               rule="H-ALM", hazop=[ref], origin=origin, detail=f"alarma {'alta' if hl == 'H' else 'baja'} de {VAR_ES[var]} ({r.deviation})",
                               setpoint=f"límite de alarma {TBD}"))
        if r.severity >= self.trip_sev:
            action = next((s for s in r.safeguards if any(k in s.lower() for k in ("paro", "corte", "cierre", "interlock", "sis", "hh", "ll"))),
                          "acción del disparo por definir")
            key = (u, var, hl, line_name)
            ex = next((x for x in self.instruments if x.category == "disparo" and (x.unit, x.variable, x.function[1], x.sensing_line) == key), None)
            if ex:
                if ref not in ex.hazop:
                    ex.hazop.append(ref)
            else:
                no = self.next_sis()
                self.instruments.append(Instrument(tag=f"{var}S{hl}{hl}-{no}", sensor=f"{var}T-{no}", variable=var, function=f"S{hl}{hl}",
                                                   loop=no, category="disparo", unit=u, sensing_line=line_name,
                                                   final_element="elemento final del SIS por definir", rule="H-TRIP", hazop=[ref], origin=origin,
                                                   detail=f"disparo {'muy alto' if hl == 'H' else 'muy bajo'} de {VAR_ES[var]} con sensor independiente del BPCS",
                                                   setpoint=f"punto de disparo {TBD}", action=action))
        return True

    def _merge(self, ins: Instrument):
        ex = next((x for x in self.instruments if x.tag == ins.tag), None)
        if ex:
            for h in ins.hazop:
                if h not in ex.hazop:
                    ex.hazop.append(h)
        else:
            self.instruments.append(ins)


def build_pid(g: ProcessGraph, rows: list[HazopRow] | None = None, alarm_severity: int = ALARM_SEVERITY,
              trip_severity: int = TRIP_SEVERITY) -> PidSpec:
    """Apply the rules to a solved process graph. `rows` defaults to the deterministic HAZOP of `g`; rows whose
    `source` is not "rule" (LLM proposals) produce items marked origin="propuesta"."""
    rows = hazop_rows(g) if rows is None else rows
    rows = [r for r in rows if r.kind not in PASS_THROUGH]
    b = _Builder(g, rows, alarm_severity, trip_severity)
    b.control_rules()
    b.isolation_rules()
    b.hazop_rules()
    assumptions = [
        f"Severidad del HAZOP preliminar (1-5): alarma desde {alarm_severity}, disparo con sensor independiente desde {trip_severity}; ajustables.",
        "Consignas, límites de alarma, puntos de disparo, presiones de ajuste de PSV y diámetros quedan por definir: no se inventan.",
        "Condensador, acumulador de reflujo y rehervidor van dentro del bloque de columna (como en DWSIM); sus utilidades se dibujan como líneas de utilidad.",
        "Las utilidades (vapor, agua de enfriamiento) no están en el flowsheet: sus válvulas van en líneas de utilidad que nacen en un conector de página.",
        "Bombas tratadas como centrífugas (control de flujo con válvula en la descarga); una de desplazamiento positivo cambia la regla.",
        "Compresor: FV en la descarga es un marcador del elemento final de capacidad (velocidad, álabes o recirculación antisurge).",
        "Columna: configuración LV (Skogestad); con reflujo L/D > 6 conviene controlar el nivel del acumulador con L (no se conoce L/D en el grafo).",
        "Intercambiadores proceso-proceso sin lazo propio; el control por bypass es decisión del ingeniero.",
        "Sin lazo de flujo de alimentación: el manipulador de producción (Luyben paso 4) es decisión del ingeniero.",
        "Las válvulas de bloqueo y check no tienen letra ISA-5.1: se etiquetan BV-nn y CHK-nn.",
    ]
    return PidSpec(g.name, b.units, b.lines, sorted(b.items, key=lambda i: (i.line or "", i.rank, i.tag)), b.instruments,
                   [r.to_dict() for r in rows], b.not_instrumented, b.undetermined, assumptions)
