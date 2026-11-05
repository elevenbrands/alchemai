"""Export a solved DWSIM flowsheet to open process-engineering standards.

* SFILES 2.0 (Vogel et al. 2023): a one-line text notation of the flowsheet topology, the format used to train
  flowsheet autocompletion models. Built with the `SFILES2` package (MIT) from a networkx graph.
* DEXPI Process (DEXPI 2.0 process model, the PFD level of DEXPI): process steps with ports and the streams
  between them, carrying the solved temperature, pressure and mass flow. Built with `pyDEXPI` (MIT) and written
  as DEXPI XML and pyDEXPI JSON.

Source of truth is the same `ProcessGraph` the HAZOP reads (`packages.hazop.graph.build_graph`), so every export
describes exactly the flowsheet DWSIM solved. Logical blocks with no physical meaning (DWSIM's recycle block) are
dropped and the two streams around them become one. Feeds with no upstream unit become `raw-k` sources and
products with no downstream unit become `prod-k` sinks, as SFILES requires.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from packages.hazop.graph import ProcessGraph, StreamNode

# DWSIM ObjectType -> (SFILES abbreviation, DEXPI process-step class, required enum field or None)
KINDS: dict[str, tuple[str, str, tuple[str, str, str] | None]] = {
    "NodeIn": ("mix", "Mixing", None),
    "Mixer": ("mix", "Mixing", None),
    "NodeOut": ("splt", "SplittingMaterial", None),
    "Splitter": ("splt", "SplittingMaterial", None),
    "Valve": ("v", "RegulatingFlow", None),
    "Vessel": ("flash", "SeparatingByFlash", None),
    "Separator": ("flash", "SeparatingByFlash", None),
    "Tank": ("tank", "StoringFluids", None),
    "HeatExchanger": ("hex", "ExchangingThermalEnergy", ("method", "HeatExchangeMethod", "GENERIC")),
    "Heater": ("hex", "SupplyingThermalEnergy", None),
    "Cooler": ("hex", "RemovingThermalEnergy", ("method", "HeatExchangeMethod", "GENERIC")),
    "Pump": ("pp", "Pumping", None),
    "Compressor": ("comp", "Compressing", ("method", "CompressionMethod", "UNSPECIFIED")),
    "Expander": ("turb", "ProcessStep", None),
    "DistillationColumn": ("dist", "Distilling", None),
    "ShortcutColumn": ("dist", "Distilling", None),
    "AbsorptionColumn": ("abs", "Absorbing", None),
    "RCT_CSTR": ("r", "ReactingChemicals", ("method", "ReactionProcessType", "TANK")),
    "RCT_PFR": ("r", "ReactingChemicals", ("method", "ReactionProcessType", "TUBULAR")),
    "RCT_Conversion": ("r", "ReactingChemicals", ("method", "ReactionProcessType", "UNSPECIFIED")),
    "RCT_Equilibrium": ("r", "ReactingChemicals", ("method", "ReactionProcessType", "UNSPECIFIED")),
    "RCT_Gibbs": ("r", "ReactingChemicals", ("method", "ReactionProcessType", "UNSPECIFIED")),
}
PASS_THROUGH = {"OT_Recycle"}
UNKNOWN = ("X", "ProcessStep", None)


@dataclass
class Edge:
    """One physical stream between two nodes (after removing logical blocks)."""
    src: str                     # SFILES node name
    dst: str
    streams: list[str]           # DWSIM stream names it covers (2 when it crossed a recycle block)
    data: StreamNode             # solved state (taken from the first stream)
    tags: dict = field(default_factory=lambda: {"he": [], "col": []})


@dataclass
class ExportGraph:
    name: str
    nodes: dict[str, dict]       # SFILES node -> {"dwsim": name|None, "kind": DWSIM kind|"raw"|"prod", "duty_kW": ...}
    edges: list[Edge]
    dropped: list[str]           # logical blocks removed

    def to_nx(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for n, a in self.nodes.items():
            g.add_node(n)
        for e in self.edges:
            g.add_edge(e.src, e.dst, tags={"he": list(e.tags["he"]), "col": list(e.tags["col"])})
        return g

    def summary(self) -> dict:
        return {"nodes": {n: a["kind"] if a["dwsim"] is None else f'{a["dwsim"]} ({a["kind"]})' for n, a in self.nodes.items()},
                "edges": [{"from": e.src, "to": e.dst, "streams": e.streams, "T_K": round(e.data.T_K, 3),
                           "P_Pa": round(e.data.P_Pa, 1), "mass_flow_kg_s": round(e.data.mass_flow_kg_s, 6)} for e in self.edges],
                "dropped_logical_blocks": self.dropped}


def export_graph(g: ProcessGraph) -> ExportGraph:
    """Map the solved ProcessGraph onto SFILES node names and physical edges."""
    src_of: dict[str, str] = {}
    dst_of: dict[str, str] = {}
    for u in g.units.values():
        for s in u.outlets:
            src_of[s] = u.name
        for s in u.inlets:
            dst_of[s] = u.name
    count: Counter = Counter()
    name_of: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    dropped = []
    for u in g.units.values():
        if u.kind in PASS_THROUGH:
            dropped.append(u.name)
            continue
        abbr = KINDS.get(u.kind, UNKNOWN)[0]
        count[abbr] += 1
        n = f"{abbr}-{count[abbr]}"
        name_of[u.name] = n
        nodes[n] = {"dwsim": u.name, "kind": u.kind, "duty_kW": u.duty_kW}

    def downstream(stream: str) -> tuple[str | None, list[str]]:
        """Follow a stream through logical blocks; return the physical unit it feeds and the streams crossed."""
        crossed = [stream]
        unit = dst_of.get(stream)
        while unit is not None and g.units[unit].kind in PASS_THROUGH:
            outs = g.units[unit].outlets
            if not outs:
                return None, crossed
            crossed.append(outs[0])
            unit = dst_of.get(outs[0])
        return unit, crossed

    edges: list[Edge] = []
    for s in g.streams:
        up = src_of.get(s)
        if up is not None and g.units[up].kind in PASS_THROUGH:
            continue                      # covered when walking from the stream that enters the block
        down, crossed = downstream(s)
        if up is None:
            count["raw"] += 1
            src = f"raw-{count['raw']}"
            nodes[src] = {"dwsim": None, "kind": "raw", "duty_kW": None}
        else:
            src = name_of[up]
        if down is None:
            count["prod"] += 1
            dst = f"prod-{count['prod']}"
            nodes[dst] = {"dwsim": None, "kind": "prod", "duty_kW": None}
        else:
            dst = name_of[down]
        e = Edge(src, dst, crossed, g.streams[s])
        # Tags: two-sided exchangers pair inlet i with outlet i; columns mark top/bottom products.
        if up is not None:
            uu = g.units[up]
            i = uu.outlets.index(s)
            if KINDS.get(uu.kind, UNKNOWN)[0] == "hex" and len(uu.inlets) > 1:
                e.tags["he"].append(f"{i + 1}_out")
            if KINDS.get(uu.kind, UNKNOWN)[0] in ("dist", "abs") and len(uu.outlets) >= 2:
                e.tags["col"].append("tout" if i == 0 else "bout")
        if down is not None:
            dd = g.units[down]
            last = crossed[-1]
            if KINDS.get(dd.kind, UNKNOWN)[0] == "hex" and len(dd.inlets) > 1:
                e.tags["he"].append(f"{dd.inlets.index(last) + 1}_in")
        edges.append(e)
    return ExportGraph(g.name, nodes, edges, dropped)


# ------------------------------------------------------------------ SFILES 2.0
def to_sfiles(eg: ExportGraph, canonical: bool = True) -> str:
    from Flowsheet_Class.flowsheet import Flowsheet
    f = Flowsheet()
    f.state = eg.to_nx()
    f.convert_to_sfiles(version="v2", remove_hex_tags=False, canonical=canonical)
    return f.sfiles


def sfiles_topology(sfiles: str) -> nx.MultiDiGraph:
    """Parse an SFILES string back to a graph (SFILES2 parser), for round-trip checks."""
    from Flowsheet_Class.flowsheet import Flowsheet
    return Flowsheet(sfiles_in=sfiles).state


def same_topology(a: nx.MultiDiGraph, b: nx.MultiDiGraph) -> bool:
    """Isomorphic as multigraphs with nodes matched by unit type (the part before '-')."""
    t = lambda n: n.split("-")[0]
    for g in (a, b):
        for n in g.nodes:
            g.nodes[n]["_t"] = t(n)
    return nx.is_isomorphic(a, b, node_match=lambda x, y: x["_t"] == y["_t"])


# ------------------------------------------------------------------ DEXPI Process
def _pc():
    from pydexpi.pydexpi_classes import pydexpi_classes as pc
    return pc


def _port(pc, direction, ident: str):
    """Ports and streams reference each other; build the port first without its connection, link it later."""
    defaults = {k: f.get_default(call_default_factory=True) for k, f in pc.MaterialPort.model_fields.items()
                if k not in ("identifier", "nominalDirection", "connectorReference") and not f.is_required()}
    return pc.MaterialPort.model_construct(identifier=ident, nominalDirection=direction, **defaults)


def _text(pc, s: str):
    return pc.MultiLanguageString(singleLanguageStrings=[pc.SingleLanguageString(language="en", value=s)])


def to_dexpi(eg: ExportGraph, compounds: list[str] | None = None):
    """Build a pyDEXPI EngineeringModel whose conceptual model is a DEXPI ProcessModel."""
    pc = _pc()
    D = pc.PortDirection
    qv = lambda v, unit: pc.QualifiedValue(value=pc.PhysicalQuantity(value=float(v), unit=unit))
    steps, ports_out, ports_in = {}, {}, {}
    for n, a in eg.nodes.items():
        if a["kind"] == "raw":
            cls, enum = "Source", None
        elif a["kind"] == "prod":
            cls, enum = "Sink", None
        else:
            _, cls, enum = KINDS.get(a["kind"], UNKNOWN)
        kw = {"identifier": n}
        if enum is not None:
            kw[enum[0]] = getattr(getattr(pc, enum[1]), enum[2])
        if a["dwsim"] is not None:
            d = f"DWSIM {a['kind']} {a['dwsim']}"
            if a["duty_kW"] is not None:
                d += f", duty {a['duty_kW']:.3f} kW (DWSIM sign)"
            kw["description"] = _text(pc, d)
        steps[n] = getattr(pc, cls)(**kw)
        ports_out[n], ports_in[n] = [], []
    conns = []
    for k, e in enumerate(eg.edges, 1):
        po = _port(pc, D.OUTLET, f"{e.src}.out{len(ports_out[e.src]) + 1}")
        pi = _port(pc, D.INLET, f"{e.dst}.in{len(ports_in[e.dst]) + 1}")
        ports_out[e.src].append(po)
        ports_in[e.dst].append(pi)
        s = pc.Stream(identifier=f"S{k}", label=e.streams[0], source=po, target=pi,
                      description=_text(pc, " -> ".join(e.streams) + f"; VF {e.data.vapor_fraction:.4f}; molar flow {e.data.molar_flow_mol_s:.6f} mol/s"),
                      temperature=[qv(e.data.T_K, pc.TemperatureUnit.KELVIN)],
                      pressure=[qv(e.data.P_Pa, pc.PressureAbsoluteUnit.PASCAL)],
                      massFlow=[qv(e.data.mass_flow_kg_s, pc.MassFlowRateUnit.KILOGRAMPERSECOND)])
        po.connectorReference = s
        pi.connectorReference = s
        conns.append(s)
    for n, st in steps.items():
        st.ports = ports_in[n] + ports_out[n]
    comps = [pc.PureMaterialComponent(identifier=c, label=c) for c in (compounds or [])]
    pm = pc.ProcessModel(processSteps=list(steps.values()), processConnections=conns, materialComponents=comps)
    return pc.EngineeringModel(conceptualModel=pm, originatingSystemName="AlchemAI",
                               originatingSystemVendorName="AlchemAI (MIT)", originatingSystemVersion="0.1")


def save_dexpi(model, out_dir: str | Path, stem: str) -> dict[str, str]:
    from pydexpi.loaders.public import DexpiXmlSerializer, JsonSerializer
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    DexpiXmlSerializer().save(model, out, stem)
    JsonSerializer().save(model, out, stem)
    return {"dexpi_xml": str(out / f"{stem}.xml"), "dexpi_json": str(out / f"{stem}.json")}


def load_dexpi_xml(path: str | Path):
    from pydexpi.loaders.public import DexpiXmlSerializer
    p = Path(path)
    return DexpiXmlSerializer().load(p.parent, p.stem)


def dexpi_digest(model) -> dict:
    """Comparable summary of a DEXPI process model: step types, connections and stream temperatures."""
    cm = model.conceptualModel
    owner = {}
    for st in cm.processSteps:
        for p in st.ports:
            owner[p.identifier] = st.identifier
    return {"steps": sorted(f"{type(s).__name__}:{s.identifier}" for s in cm.processSteps),
            "connections": sorted((owner.get(c.source.identifier), owner.get(c.target.identifier), c.label,
                                   round(c.temperature[0].value.value, 6) if c.temperature else None) for c in cm.processConnections)}


# ------------------------------------------------------------------ one call
def export_all(g: ProcessGraph, out_dir: str | Path | None, compounds: list[str] | None = None) -> dict:
    """SFILES string + DEXPI files for a solved ProcessGraph, with round-trip checks on both."""
    eg = export_graph(g)
    sf = to_sfiles(eg)
    out = {"sfiles": sf, "graph": eg.summary(),
           "sfiles_roundtrip_ok": same_topology(eg.to_nx(), sfiles_topology(sf))}
    model = to_dexpi(eg, compounds)
    if out_dir is not None:
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in g.name) or "flowsheet"
        paths = save_dexpi(model, out_dir, stem)
        Path(out_dir, f"{stem}.sfiles").write_text(sf + "\n")
        Path(out_dir, f"{stem}.graph.json").write_text(json.dumps(eg.summary(), indent=1, ensure_ascii=False))
        out["files"] = {**paths, "sfiles": str(Path(out_dir, f"{stem}.sfiles")), "graph_json": str(Path(out_dir, f"{stem}.graph.json"))}
        out["dexpi_roundtrip_ok"] = dexpi_digest(load_dexpi_xml(paths["dexpi_xml"])) == dexpi_digest(model)
    return out


# ------------------------------------------------------------------ tool (agent / MCP)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "exports"


def export_flowsheet(flowsheet_id: str, out_dir: str | None = None) -> dict:
    """Export a SOLVED flowsheet to open standards: an SFILES 2.0 string (topology in one line, e.g.
    "(raw)(mix)<1(hex)(splt)1(prod)") and DEXPI Process files (XML + JSON) whose streams carry the solved T, P and
    mass flow. Both are re-read and compared with the source (sfiles_roundtrip_ok, dexpi_roundtrip_ok). Files go to
    out_dir (default: the repo's exports/<flowsheet_id>/). Call solve() first. Use when the user wants to share,
    archive or hand the flowsheet to another tool, or asks for SFILES / DEXPI / a P&ID-ready model."""
    from packages.dwsim_client import tools as T
    from packages.hazop.graph import build_graph
    s = T._s(flowsheet_id)
    g = build_graph(s["fs"], flowsheet_id)
    return export_all(g, out_dir or (DEFAULT_OUT / flowsheet_id), s.get("compounds"))
