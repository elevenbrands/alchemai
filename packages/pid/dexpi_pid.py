"""Write a `PidSpec` as a DEXPI 2.0.1 P&ID (PlantModel) with pyDEXPI 2.1.0, read it back and check it.

Encoding follows the official DEXPI reference P&ID (`reference_pid.xml` of gitlab.com/dexpi/Specification):
* equipment = TaggedPlantItems with ProcessNozzles (one per line end) and InstrumentNozzles (one per sensor on the
  equipment); a DWSIM valve is a GlobeValve and a DWSIM mixer/splitter is a PipeTee, inside the piping;
* each line = one PipingNetworkSystem (lineNumber = stream name) with one segment: source → items → target, the items
  joined by Pipes; feeds and products start/end in off-page connectors; utility lines start in an off-page connector;
* a line PSV sits on a PipeTee branch, an equipment PSV on its own nozzle; both discharge to an off-page connector;
* each instrument function = ProcessInstrumentationFunction (category = ISA first letter, functions = succeeding
  letters, number = loop number) inside an InstrumentationLoopFunction; the transmitter is a
  ProcessSignalGeneratingFunction, the control valve an ActuatingFunction → ActuatingSystem → OperatedValveReference;
* every element carries a Note with its rule, source and HAZOP rows (the trace).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from packages.pid.rules import INLINE, RULES, PidSpec

EQUIPMENT = {"Vessel": "PressureVessel", "Separator": "PressureVessel", "Tank": "Tank", "Pump": "CentrifugalPump",
             "Compressor": "Compressor", "HeatExchanger": "TubularHeatExchanger", "Heater": "HeatExchanger",
             "Cooler": "HeatExchanger", "DistillationColumn": "ProcessColumn", "ShortcutColumn": "ProcessColumn",
             "AbsorptionColumn": "ProcessColumn"}
VALVES = {"BV": "GateValve", "CHK": "CheckValve", "PSV": "SpringLoadedGlobeSafetyValve"}   # control valves: GlobeValve

# Official DEXPI 2.0.1 XML schema, pinned to the commit of gitlab.com/dexpi/Specification read on 2026-11-13.
XSD_COMMIT = "eef0f0c6e8edfe84582e7a45d109c396bdc544f8"
XSD_URL = ("https://gitlab.com/api/v4/projects/dexpi%2FSpecification/repository/files/"
           "src%2Fdocumentation%2F_static%2FDEXPI_XML_Schema.xsd/raw?ref=" + XSD_COMMIT)
XSD_CACHE = Path.home() / ".cache" / "alchemai" / "dexpi" / f"DEXPI_XML_Schema_{XSD_COMMIT[:8]}.xsd"


def _pc():
    from pydexpi.pydexpi_classes import pydexpi_classes as pc
    return pc


def _mls(pc, s: str):
    return pc.MultiLanguageString(singleLanguageStrings=[pc.SingleLanguageString(language="es", value=s)])


class _Writer:
    def __init__(self, spec: PidSpec):
        self.pc = _pc()
        self.spec = spec
        self.notes = []
        self.equip: dict[str, object] = {}
        self.inline: dict[str, object] = {}       # DWSIM valve / mixer / splitter -> piping component
        self.inline_owner: dict[str, str] = {}    # inline unit -> the line whose segment owns it
        self.inline_port: dict[str, dict[str, int]] = {}   # inline unit -> line name -> node index
        self.item_obj: dict[str, object] = {}     # item tag -> DEXPI object
        self.segment_of: dict[str, object] = {}   # line name -> segment
        self.segment_of_item: dict[str, object] = {}
        self.systems = []
        self.nozzle_n: Counter = Counter()

    def note(self, text: str):
        n = self.pc.Note(noteText=_mls(self.pc, text), noteClassification="trazabilidad AlchemAI")
        self.notes.append(n)
        return [n]

    def trace(self, rule: str, hazop: list[str], origin: str, detail: str) -> str:
        txt, src = RULES.get(rule, ("", ""))
        parts = [f"[{origin}] {rule}: {txt}", f"Fuente: {src}"] + ([f"Detalle: {detail}"] if detail else []) + list(hazop)
        return " | ".join(parts)

    def nozzle(self, unit: str, cls: str = "ProcessNozzle", sub: str | None = None):
        pc = self.pc
        self.nozzle_n[unit] += 1
        nz = getattr(pc, cls)(subTagName=sub or f"N{self.nozzle_n[unit]}", nodes=[pc.PipingNode()])
        self.equip[unit].nozzles.append(nz)
        return nz

    # ------------------------------------------------------------- build
    def build(self):
        pc, spec = self.pc, self.spec
        for name, kind in spec.units.items():
            if kind in INLINE:
                ins = [l for l in spec.lines.values() if l.dst == name]
                outs = [l for l in spec.lines.values() if l.src == name]
                ports = {l.name: i for i, l in enumerate(ins)} | {l.name: len(ins) + i for i, l in enumerate(outs)}
                self.inline_port[name] = ports
                self.inline_owner[name] = ins[0].name if ins else (outs[0].name if outs else "")
                if kind == "Valve":
                    it = next((i for i in spec.items if i.dwsim == name), None)
                    obj = pc.GlobeValve(pipingComponentName=it.tag if it else name, pipingComponentNumber=name,
                                        nodes=[pc.PipingNode(), pc.PipingNode()],
                                        referencedNotes=self.note(self.trace(it.rule, it.hazop, it.origin, it.detail) if it else name))
                    if it:
                        self.item_obj[it.tag] = obj
                else:
                    obj = pc.PipeTee(pipingComponentName=name, nodes=[pc.PipingNode() for _ in range(max(3, len(ports)))],
                                     referencedNotes=self.note(f"{kind} {name} del flowsheet como unión de tubería"))
                self.inline[name] = obj
                continue
            cls = EQUIPMENT.get(kind, "PressureVessel")
            desc = f"{kind} {name}" + (" (reactor)" if kind.startswith("RCT") else "")
            self.equip[name] = getattr(pc, cls)(tagName=name, nozzles=[], equipmentDescription=_mls(pc, desc),
                                                referencedNotes=self.note(f"Equipo del flowsheet: {desc}"))
        for line in spec.lines.values():
            self._line(line)
        for it in spec.items:                      # equipment PSVs (no line)
            if it.isa == "PSV" and it.line is None:
                nz = self.nozzle(it.unit, sub=f"N-{it.tag}")
                self._relief_branch(it, nz, nz.nodes[0])
        pifs, loops, acts = self._instruments()
        plant = pc.PlantModel(taggedPlantItems=list(self.equip.values()), pipingNetworkSystems=self.systems,
                              processInstrumentationFunctions=pifs, instrumentationLoopFunctions=loops,
                              actuatingSystems=acts, notes=self.notes)
        return pc.EngineeringModel(conceptualModel=plant, originatingSystemName="AlchemAI",
                                   originatingSystemVendorName="Elevenbrands", originatingSystemVersion="M10")

    def _line(self, line):
        pc, spec = self.pc, self.spec
        chain: list[tuple[object, int, int]] = []     # (object, node in, node out)
        owned: list[object] = []
        # source
        if line.src is None:
            src = pc.FlowInPipeOffPageConnector(nodes=[pc.PipingNode()],
                                                referencedNotes=self.note(line.note or f"límite de batería: alimentación {line.name}"))
            owned.append(src)
            chain.append((src, 0, 0))
        elif line.src in self.inline:
            chain.append((self.inline[line.src], -1, self.inline_port[line.src][line.name]))
        else:
            nz = self.nozzle(line.src)
            chain.append((nz, 0, 0))
        # items
        for it in sorted((i for i in spec.items if i.line == line.name and not i.dwsim), key=lambda i: (i.rank, i.tag)):
            note = self.note(self.trace(it.rule, it.hazop, it.origin, it.detail) + (f" | {it.setpoint}" if it.setpoint else ""))
            if it.isa == "PSV":
                tee = pc.PipeTee(pipingComponentName=f"T-{it.tag}", nodes=[pc.PipingNode() for _ in range(3)])
                owned.append(tee)
                chain.append((tee, 0, 1))
                self._pending_branch = getattr(self, "_pending_branch", []) + [(it, tee, tee.nodes[2], note)]
                continue
            cls = VALVES.get(it.isa, "GlobeValve")
            obj = getattr(pc, cls)(pipingComponentName=it.tag, nodes=[pc.PipingNode(), pc.PipingNode()], referencedNotes=note)
            self.item_obj[it.tag] = obj
            owned.append(obj)
            chain.append((obj, 0, 1))
        # target
        if line.dst is None:
            dst = pc.FlowOutPipeOffPageConnector(nodes=[pc.PipingNode()],
                                                 referencedNotes=self.note(f"límite de batería: producto {line.name}"))
            owned.append(dst)
            chain.append((dst, 0, 0))
        elif line.dst in self.inline:
            obj = self.inline[line.dst]
            if self.inline_owner[line.dst] == line.name:
                owned.append(obj)
            chain.append((obj, self.inline_port[line.dst][line.name], -1))
        else:
            nz = self.nozzle(line.dst)
            chain.append((nz, 0, 0))
        # A tee may only start or end a segment (pyDEXPI rule segment_linear_order, as in the reference P&ID):
        # split the line at every PSV tee.
        tees = {id(t) for _, t, _, _ in getattr(self, "_pending_branch", [])}
        cuts = [0] + [i for i, (o, _, _) in enumerate(chain) if id(o) in tees and 0 < i < len(chain) - 1] + [len(chain) - 1]
        segs, assigned = [], set()
        for k, (a_i, b_i) in enumerate(zip(cuts, cuts[1:])):
            part = chain[a_i:b_i + 1]
            pipes = [pc.Pipe(sourceItem=a, sourceNode=a.nodes[ao], targetItem=b, targetNode=b.nodes[bi])
                     for (a, _, ao), (b, bi, _) in zip(part, part[1:])]
            mine = [o for o, _, _ in part if any(o is x for x in owned) and id(o) not in assigned]
            assigned |= {id(o) for o in mine}
            first, last = part[0], part[-1]
            seg = pc.PipingNetworkSegment(segmentNumber=f"S{k + 1}", items=mine, connections=pipes, sourceItem=first[0],
                                          sourceNode=first[0].nodes[first[2]], targetItem=last[0], targetNode=last[0].nodes[last[1]],
                                          flowDirection=pc.PipingNetworkSegmentFlowClassification.SINGLEFLOWPIPINGNETWORKSEGMENT)
            segs.append(seg)
            for o in mine:
                self.segment_of_item[o.id] = seg
        seg = segs[0]
        state = "" if line.T_K is None else f"T {line.T_K:.2f} K, P {line.P_Pa / 1e5:.3f} bar, VF {line.vapor_fraction}"
        sysnote = self.note(f"Línea {line.kind} {line.name}: {line.src or '—'} → {line.dst or '—'}"
                            + (f" (corrientes {', '.join(line.streams)})" if line.streams else "") + (f"; {state}" if state else "")
                            + (f"; {line.note}" if line.note else ""))
        self.systems.append(pc.PipingNetworkSystem(lineNumber=line.name, segments=segs, referencedNotes=sysnote))
        self.segment_of[line.name] = seg
        for it, tee, node, note in getattr(self, "_pending_branch", []):
            self._relief_branch(it, tee, node, note)
        self._pending_branch = []

    def _relief_branch(self, it, src, src_node, note=None):
        pc = self.pc
        note = note or self.note(self.trace(it.rule, it.hazop, it.origin, it.detail) + f" | {it.setpoint}")
        psv = pc.SpringLoadedGlobeSafetyValve(positionNumber=it.tag, nodes=[pc.PipingNode(), pc.PipingNode()],
                                              referencedNotes=note)
        out = pc.FlowOutPipeOffPageConnector(nodes=[pc.PipingNode()],
                                             referencedNotes=self.note("a sistema de alivio (antorcha o atmósfera segura; destino por definir)"))
        pipes = [pc.Pipe(sourceItem=src, sourceNode=src_node, targetItem=psv, targetNode=psv.nodes[0]),
                 pc.Pipe(sourceItem=psv, sourceNode=psv.nodes[1], targetItem=out, targetNode=out.nodes[0])]
        seg = pc.PipingNetworkSegment(segmentNumber="S1", items=[psv, out], connections=pipes, sourceItem=src,
                                      sourceNode=src_node, targetItem=out, targetNode=out.nodes[0])
        self.item_obj[it.tag] = psv
        self.segment_of_item[psv.id] = seg
        self.systems.append(pc.PipingNetworkSystem(lineNumber=f"alivio-{it.tag}", segments=[seg],
                                                   referencedNotes=self.note(f"Línea de alivio de {it.tag}")))

    def _instruments(self):
        pc, spec = self.pc, self.spec
        by_loop: dict[str, list] = {}
        for ins in spec.instruments:
            by_loop.setdefault(ins.loop, []).append(ins)
        pif_of: dict[str, object] = {}
        pifs, loops, acts, cascades = [], [], [], []
        for no, group in by_loop.items():
            psgf = None
            members = []
            for ins in group:
                note = self.note(self.trace(ins.rule, ins.hazop, ins.origin, ins.detail)
                                 + f" | consigna/límite: {ins.setpoint}" + (f" | acción: {ins.action}" if ins.action else ""))
                pif = pc.ProcessInstrumentationFunction(processInstrumentationFunctionCategory=ins.variable,
                                                        processInstrumentationFunctions=ins.function,
                                                        processInstrumentationFunctionNumber=no,
                                                        safetyRelevanceClass="SIS" if ins.category == "disparo" else None,
                                                        referencedNotes=note)
                if psgf is None:
                    loc = self.segment_of.get(ins.sensing_line) if ins.sensing_line else None
                    if loc is None:
                        loc = self.nozzle(ins.unit, cls="InstrumentNozzle", sub=f"N-{ins.sensor}")
                    psgf = pc.ProcessSignalGeneratingFunction(processSignalGeneratingFunctionNumber=ins.sensor, sensingLocation=loc)
                    pif.processSignalGeneratingFunctions.append(psgf)
                pif.signalConveyingFunctions.append(pc.MeasuringLineFunction(source=psgf, target=pif))
                obj = self.item_obj.get(ins.final_element)
                if obj is not None and ins.category == "lazo":
                    sys_ = pc.ActuatingSystem(actuatingSystemNumber=ins.final_element,
                                              operatedValveReference=pc.OperatedValveReference(valve=obj))
                    acts.append(sys_)
                    af = pc.ActuatingFunction(actuatingFunctionNumber=ins.final_element, systems=sys_,
                                              actuatingLocation=self.segment_of_item.get(obj.id))
                    pif.actuatingFunctions.append(af)
                    pif.signalConveyingFunctions.append(pc.SignalLineFunction(source=pif, target=af))
                if ins.cascade_to:
                    cascades.append((pif, ins.cascade_to))
                pif_of[ins.tag] = pif
                members.append(pif)
                pifs.append(pif)
            loops.append(pc.InstrumentationLoopFunction(instrumentationLoopFunctionNumber=no, processInstrumentationFunctions=members))
        for master, slave in cascades:
            if slave in pif_of:
                master.signalConveyingFunctions.append(pc.SignalLineFunction(source=master, target=pif_of[slave]))
        return pifs, loops, acts


def to_dexpi_pid(spec: PidSpec):
    return _Writer(spec).build()


def save(model, out_dir: str | Path, stem: str) -> dict[str, str]:
    from pydexpi.loaders.public import DexpiXmlSerializer, JsonSerializer
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    DexpiXmlSerializer().save(model, out, stem)
    JsonSerializer().save(model, out, stem)
    return {"dexpi_pid_xml": str(out / f"{stem}.xml"), "dexpi_pid_json": str(out / f"{stem}.json")}


def load(path: str | Path):
    from pydexpi.loaders.public import DexpiXmlSerializer, JsonSerializer
    p = Path(path)
    return (JsonSerializer() if p.suffix == ".json" else DexpiXmlSerializer()).load(p.parent, p.stem)


def digest(model) -> dict:
    """Comparable summary of a DEXPI P&ID: equipment, piping items, lines, instrument functions, loops, actuators,
    signal links and the cascade links."""
    pm = model.conceptualModel
    items = Counter()
    for s in pm.pipingNetworkSystems:
        for seg in s.segments:
            for it in seg.items:
                items[(type(it).__name__, getattr(it, "pipingComponentName", None) or getattr(it, "positionNumber", None))] += 1
    pif_tag = lambda f: f"{f.processInstrumentationFunctionCategory}{f.processInstrumentationFunctions}-{f.processInstrumentationFunctionNumber}"
    signals = Counter()
    for f in pm.processInstrumentationFunctions:
        for s in f.signalConveyingFunctions:
            tgt = s.target
            name = pif_tag(tgt) if type(tgt).__name__ == "ProcessInstrumentationFunction" else getattr(tgt, "actuatingFunctionNumber", None)
            src = s.source
            sname = pif_tag(src) if type(src).__name__ == "ProcessInstrumentationFunction" else getattr(src, "processSignalGeneratingFunctionNumber", None)
            signals[(type(s).__name__, sname, name)] += 1
    return {
        "equipment": sorted((e.tagName, type(e).__name__, len(e.nozzles)) for e in pm.taggedPlantItems),
        "lines": sorted((s.lineNumber, len(s.segments), sum(len(g.connections) for g in s.segments)) for s in pm.pipingNetworkSystems),
        "items": sorted((k[0], k[1], v) for k, v in items.items()),
        "instruments": sorted((pif_tag(f), f.safetyRelevanceClass or "",
                               tuple(g.processSignalGeneratingFunctionNumber for g in f.processSignalGeneratingFunctions),
                               tuple(a.actuatingFunctionNumber for a in f.actuatingFunctions))
                              for f in pm.processInstrumentationFunctions),
        "loops": sorted((l.instrumentationLoopFunctionNumber, tuple(sorted(pif_tag(f) for f in l.processInstrumentationFunctions)))
                        for l in pm.instrumentationLoopFunctions),
        "actuators": sorted((a.actuatingSystemNumber, type(a.operatedValveReference.valve).__name__,
                             a.operatedValveReference.valve.pipingComponentName) for a in pm.actuatingSystems),
        "signals": sorted((k[0], k[1], k[2], v) for k, v in signals.items()),
        "notes": len(pm.notes),
    }


def spec_digest(spec: PidSpec) -> dict:
    """The same summary computed from the spec alone (what the DEXPI file must contain)."""
    return {"instruments": sorted(f"{i.variable}{i.function}-{i.loop}" for i in spec.instruments),
            "loops": sorted({i.loop for i in spec.instruments}),
            "items": sorted(i.tag for i in spec.items),
            "equipment": sorted(n for n, k in spec.units.items() if k not in INLINE)}


def model_matches_spec(model, spec: PidSpec) -> dict:
    d = digest(model)
    got_items = sorted(n for _, n, c in d["items"] for _ in range(c) if n and not n.startswith("T-PSV") and n not in
                       {k for k, v in spec.units.items() if v in INLINE and v != "Valve"})
    got = {"instruments": sorted(t for t, *_ in d["instruments"]), "loops": sorted(n for n, _ in d["loops"]),
           "items": got_items, "equipment": sorted(t for t, *_ in d["equipment"])}
    want = spec_digest(spec)
    return {k: got[k] == want[k] for k in want} | {"diff": {k: {"falta": sorted(set(want[k]) - set(got[k])),
                                                                 "sobra": sorted(set(got[k]) - set(want[k]))}
                                                             for k in want if got[k] != want[k]}}


def official_xsd(fetch: bool = True) -> Path | None:
    """Path of the official DEXPI XML schema (cached outside the repo: the specification repository states no
    license, so it is not vendored). Downloads it once, pinned to XSD_COMMIT."""
    if XSD_CACHE.exists():
        return XSD_CACHE
    if not fetch:
        return None
    import urllib.request
    try:
        with urllib.request.urlopen(XSD_URL, timeout=20) as r:
            data = r.read()
    except Exception:
        return None
    if b"<xs:schema" not in data and b"<xsd:schema" not in data:
        return None
    XSD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    XSD_CACHE.write_bytes(data)
    return XSD_CACHE


def validate_xsd(xml_path: str | Path, fetch: bool = True) -> dict:
    xsd = official_xsd(fetch)
    if xsd is None:
        return {"status": "pendiente", "reason": "no se pudo obtener el XSD oficial de DEXPI (sin red)"}
    from lxml import etree
    schema = etree.XMLSchema(etree.parse(str(xsd)))
    ok = schema.validate(etree.parse(str(xml_path)))
    return {"status": "valido" if ok else "invalido", "xsd": f"gitlab.com/dexpi/Specification @ {XSD_COMMIT[:8]}",
            "xsd_sha256": hashlib.sha256(xsd.read_bytes()).hexdigest(), "errors": [str(e) for e in schema.error_log][:10]}


def validate_semantic(model) -> list[str]:
    """pyDEXPI's built-in DEXPI rules (mandatory attributes, segment connectivity, one loop per instrument, ...)."""
    from pydexpi.validation.default_validator import validate
    return [str(i) for i in validate(model)]
