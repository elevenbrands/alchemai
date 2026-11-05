"""Process graph from a solved DWSIM flowsheet: the substrate for HAZOP rules.

Nodes are units and material streams; edges are the connections. Every node carries the solved state
(T, P, vapor fraction, flows, composition, duty) so rules can reason with numbers, not adjectives.
Built from `Flowsheet.Inner.SimulationObjects` exactly like verify.py — works for flowsheets built by
anyone (agent, MCP client, file).
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client.verify import _prop, _stream_view  # noqa: E402

FLAMMABLE = {"Methane", "Ethane", "Propane", "N-butane", "Isobutane", "N-pentane", "N-hexane", "N-heptane", "N-octane",
             "1-Butene", "Isobutene", "Hydrogen", "Carbon monoxide", "Ethanol", "Methanol", "Benzene", "Toluene", "Ethylene", "Propylene", "Ammonia"}
TOXIC = {"Carbon monoxide", "Hydrogen sulfide", "Ammonia", "Methanol", "Benzene", "Chlorine", "Sulfur dioxide"}
HYDRATE_FORMERS = {"Methane", "Ethane", "Propane", "Carbon dioxide", "Nitrogen", "Hydrogen sulfide"}


@dataclass
class StreamNode:
    name: str
    T_K: float
    P_Pa: float
    vapor_fraction: float
    molar_flow_mol_s: float
    mass_flow_kg_s: float
    mole_fractions: dict[str, float]
    flammable: bool
    toxic: bool
    has_water: bool
    hydrate_risk: bool          # water + hydrate formers + T < 300 K + P > 10 bar


@dataclass
class UnitNode:
    name: str
    kind: str                    # DWSIM ObjectType string, e.g. "Compressor", "Cooler", "Vessel", "RCT_Conversion"
    inlets: list[str]
    outlets: list[str]
    energy_in: list[str]
    energy_out: list[str]
    duty_kW: float | None        # DeltaQ / Q by reflection (sign: DWSIM convention)
    T_in_K: float | None
    T_out_K: float | None
    P_in_Pa: float | None
    P_out_Pa: float | None
    exothermic: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessGraph:
    name: str
    streams: dict[str, StreamNode]
    units: dict[str, UnitNode]
    edges: list[tuple[str, str]]      # (from, to) over stream and unit names

    def to_dict(self) -> dict:
        return {"name": self.name, "streams": {k: asdict(v) for k, v in self.streams.items()},
                "units": {k: asdict(v) for k, v in self.units.items()}, "edges": self.edges}


def stream_node(name: str, T_K: float, P_Pa: float, vapor_fraction: float, molar_flow_mol_s: float,
                mass_flow_kg_s: float, mole_fractions: dict[str, float]) -> StreamNode:
    """StreamNode with the hazard flags derived from composition, T and P (same rule for every source)."""
    present = {c for c, x in mole_fractions.items() if x > 1e-4}
    water = "Water" in present
    return StreamNode(name=name, T_K=T_K, P_Pa=P_Pa, vapor_fraction=vapor_fraction, molar_flow_mol_s=molar_flow_mol_s,
                      mass_flow_kg_s=mass_flow_kg_s, mole_fractions=mole_fractions,
                      flammable=bool(present & FLAMMABLE), toxic=bool(present & TOXIC), has_water=water,
                      hydrate_risk=water and bool(present & HYDRATE_FORMERS) and T_K < 300.0 and P_Pa > 10e5)


def _obj_type(o) -> str:
    return str(o.GraphicObject.ObjectType)


def build_graph(fs, name: str = "flowsheet") -> ProcessGraph:
    inner = fs.Inner
    objects = {kv.Key: kv.Value for kv in inner.SimulationObjects}
    by_name = {o.GraphicObject.Name: o for o in objects.values()}
    streams: dict[str, StreamNode] = {}
    for o in objects.values():
        if _obj_type(o) != "MaterialStream":
            continue
        v = _stream_view(fs.MaterialStream(o.GraphicObject.Tag).Object)
        fr = {c: (f / v.molar_mol_s if v.molar_mol_s else 0.0) for c, f in v.compounds_mol_s.items()}
        streams[v.name] = stream_node(v.name, v.temperature_k, v.pressure_pa, v.vapor_fraction, v.molar_mol_s, v.mass_kg_s, fr)

    units: dict[str, UnitNode] = {}
    edges: list[tuple[str, str]] = []
    for o in objects.values():
        kind = _obj_type(o)
        if kind in ("MaterialStream", "EnergyStream"):
            continue
        go = o.GraphicObject
        me = go.Name
        inl, outl, ein, eout = [], [], [], []
        for c in list(go.InputConnectors):
            if not c.IsAttached:
                continue
            other = by_name.get(c.AttachedConnector.AttachedFrom.Name)
            if other is None or other.GraphicObject.Name == me:
                other = by_name.get(c.AttachedConnector.AttachedTo.Name)
            if other is None or other.GraphicObject.Name == me:
                continue
            (inl if _obj_type(other) == "MaterialStream" else ein).append(other.GraphicObject.Tag)
        for c in list(go.OutputConnectors):
            if not c.IsAttached:
                continue
            other = by_name.get(c.AttachedConnector.AttachedTo.Name)
            if other is None or other.GraphicObject.Name == me:
                other = by_name.get(c.AttachedConnector.AttachedFrom.Name)
            if other is None or other.GraphicObject.Name == me:
                continue
            (outl if _obj_type(other) == "MaterialStream" else eout).append(other.GraphicObject.Tag)
        ec = getattr(go, "EnergyConnector", None)
        if ec is not None and ec.IsAttached:
            other = by_name.get(ec.AttachedConnector.AttachedTo.Name) or by_name.get(ec.AttachedConnector.AttachedFrom.Name)
            if other is not None and _obj_type(other) == "EnergyStream":
                eout.append(other.GraphicObject.Tag)
        duty = None
        for pn in ("DeltaQ", "Q"):
            dq = _prop(o, pn)
            if dq is not None:
                try:
                    duty = float(dq)
                    break
                except TypeError:
                    pass
        tin = [streams[s].T_K for s in inl if s in streams]
        tout = [streams[s].T_K for s in outl if s in streams]
        pin = [streams[s].P_Pa for s in inl if s in streams]
        pout = [streams[s].P_Pa for s in outl if s in streams]
        u = UnitNode(name=go.Tag, kind=kind, inlets=inl, outlets=outl, energy_in=ein, energy_out=eout, duty_kW=duty,
                     T_in_K=max(tin) if tin else None, T_out_K=max(tout) if tout else None,
                     P_in_Pa=max(pin) if pin else None, P_out_Pa=max(pout) if pout else None)
        if kind.startswith("RCT") and duty is not None:
            u.exothermic = duty < 0
        units[u.name] = u
        edges += [(s, u.name) for s in inl] + [(u.name, s) for s in outl]
    return ProcessGraph(name=name, streams=streams, units=units, edges=edges)
