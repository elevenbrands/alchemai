"""Process graph from a DEXPI Process model: HAZOP for flowsheets that never touched DWSIM.

Reads the DEXPI 2.0 process model (pyDEXPI, XML or JSON) and builds the same `ProcessGraph` the DWSIM reader
builds, so `rules.hazop_rows` runs unchanged. What it reads:

* process steps → units. The unit kind comes from the DEXPI class (Compressing → Compressor, Distilling →
  DistillationColumn, ...). Files exported by AlchemAI also carry the DWSIM kind and duty in the step
  description ("DWSIM Compressor K-1, duty 52.820 kW"), which is used when present.
* streams → material streams: temperature, pressure and mass flow from the stream; composition (moleFractions in
  the order of the material template), molar flow (sum of component moleFlow) and vapor fraction from the
  referenced MaterialState. Missing values are left at neutral defaults and reported in `missing`, never invented.
* Sources and Sinks are boundaries, not units: they get no HAZOP rows.
"""
from __future__ import annotations

import re
from pathlib import Path

from packages.hazop.graph import ProcessGraph, UnitNode, stream_node

# DEXPI process-step class → unit kind understood by the rules (DWSIM ObjectType vocabulary)
DEXPI_KIND = {
    "Mixing": "Mixer", "MixingSimple": "Mixer", "StaticMixing": "Mixer", "RotaryMixing": "Mixer",
    "SplittingMaterial": "Splitter", "Splitting": "Splitter",
    "RegulatingFlow": "Valve", "LimitingFlow": "Valve", "ShuttingOffFlow": "Valve",
    "SeparatingByFlash": "Separator", "SeparatingByPhaseSeparation": "Separator", "SeparatingByGravity": "Separator",
    "StoringFluids": "Tank", "StoringInTank": "Tank", "StoringInPressureVessel": "Vessel",
    "ExchangingThermalEnergy": "HeatExchanger",
    "SupplyingThermalEnergy": "Heater", "HeatingElectrical": "Heater", "HeatingInFurnace": "Heater",
    "RemovingThermalEnergy": "Cooler", "Cooling": "Cooler",
    "Pumping": "Pump", "Compressing": "Compressor",
    "Distilling": "DistillationColumn", "StabilizingDistilling": "DistillationColumn",
    "StrippingDistilling": "DistillationColumn", "VacuumDistilling": "DistillationColumn",
    "Absorbing": "AbsorptionColumn",
    "ReactingChemicals": "RCT_Generic",
}
BOUNDARY = {"Source", "Sink"}
_DWSIM = re.compile(r"^DWSIM (\S+) ")
_DUTY = re.compile(r"duty (-?[0-9.eE+-]+) kW")
_VF = re.compile(r"vapor_fraction=(-?[0-9.eE+-]+)")


def _text(mls) -> str:
    if mls is None or not getattr(mls, "singleLanguageStrings", None):
        return ""
    return " ".join(str(s.value) for s in mls.singleLanguageStrings if s.value)


def _q(values) -> float | None:
    """First QualifiedValue as SI float (K, Pa, kg/s); converts the common non-SI units."""
    if not values:
        return None
    v = values[0].value
    if isinstance(v, (int, float)):
        return float(v)
    x, u = float(v.value), getattr(v.unit, "name", str(v.unit))
    return {"DEGREECELSIUS": x + 273.15, "DEGREEFAHRENHEIT": (x - 32) * 5 / 9 + 273.15, "BAR": x * 1e5,
            "KILOPASCAL": x * 1e3, "MEGAPASCAL": x * 1e6, "MILLIBAR": x * 100.0, "KILOGRAMPERHOUR": x / 3600.0,
            "KILOGRAMPERMINUTE": x / 60.0, "KILOMOLEPERSECOND": x * 1000.0}.get(u, x)


def load_dexpi(path: str | Path):
    """Load a DEXPI model from .xml (DEXPI XML) or .json (pyDEXPI JSON)."""
    from pydexpi.loaders.public import DexpiXmlSerializer, JsonSerializer
    p = Path(path)
    ser = JsonSerializer() if p.suffix.lower() == ".json" else DexpiXmlSerializer()
    return ser.load(p.parent, p.stem)


def graph_from_dexpi(model, name: str = "dexpi") -> tuple[ProcessGraph, list[str]]:
    """ProcessGraph (+ list of what the file did not provide) from a pyDEXPI EngineeringModel or ProcessModel."""
    pm = getattr(model, "conceptualModel", None) or model
    missing: list[str] = []
    owner: dict[str, object] = {}
    for st in pm.processSteps:
        for p in st.ports:
            owner[p.identifier] = st

    streams = {}
    src_of, dst_of = {}, {}
    for c in pm.processConnections:
        if type(c).__name__ != "Stream":
            continue
        sname = c.label if isinstance(c.label, str) and c.label else c.identifier
        T, P, m = _q(c.temperature), _q(c.pressure), _q(c.massFlow)
        fr, molar, vf = {}, 0.0, None
        ms = c.materialStateReference
        if ms is not None and ms.state is not None and ms.state.composition is not None:
            comp = ms.state.composition
            tmpl = c.source.materialTemplateReference or c.target.materialTemplateReference
            names = [x.label or x.identifier for x in tmpl.listOfComponents.component] if tmpl is not None else []
            xs = [_q([q]) for q in comp.moleFractions]
            if names and len(names) == len(xs):
                fr = {n: x for n, x in zip(names, xs) if x is not None}
            molar = sum(_q([q]) or 0.0 for q in comp.moleFlow)
            if m is None:
                m = _q(ms.state.massFlow)
            mv = _VF.search(ms.description or "")
            vf = float(mv.group(1)) if mv else None
        for label, val in (("T", T), ("P", P), ("mass flow", m)):
            if val is None:
                missing.append(f"{sname}: {label}")
        if not fr:
            missing.append(f"{sname}: composition (hazard flags off)")
        streams[sname] = stream_node(sname, T or 0.0, P or 0.0, vf if vf is not None else float("nan"), molar, m or 0.0, fr)
        so, ta = owner.get(c.source.identifier), owner.get(c.target.identifier)
        if so is not None:
            src_of.setdefault(so.identifier, []).append(sname)
        if ta is not None:
            dst_of.setdefault(ta.identifier, []).append(sname)

    units, edges = {}, []
    for st in pm.processSteps:
        cls = type(st).__name__
        if cls in BOUNDARY:
            continue
        desc = _text(st.description)
        md = _DWSIM.search(desc)
        kind = md.group(1) if md else DEXPI_KIND.get(cls, cls)
        if not md and kind == "RCT_Generic" and str(getattr(getattr(st, "method", None), "value", "")) == "Tank":
            kind = "RCT_CSTR"           # stirred tank reactor (DWSIM vocabulary), so agitation rows apply
        mq = _DUTY.search(desc)
        duty = float(mq.group(1)) if mq else None
        inl, outl = dst_of.get(st.identifier, []), src_of.get(st.identifier, [])
        tin = [streams[s].T_K for s in inl]
        tout = [streams[s].T_K for s in outl]
        pin = [streams[s].P_Pa for s in inl]
        pout = [streams[s].P_Pa for s in outl]
        u = UnitNode(name=st.identifier, kind=kind, inlets=inl, outlets=outl, energy_in=[], energy_out=[], duty_kW=duty,
                     T_in_K=max(tin) if tin else None, T_out_K=max(tout) if tout else None,
                     P_in_Pa=max(pin) if pin else None, P_out_Pa=max(pout) if pout else None)
        if kind.startswith("RCT"):
            if duty is not None:
                u.exothermic = duty < 0
            else:
                u.notes.append("duty not in the DEXPI file: exothermicity not assessed")
                missing.append(f"{st.identifier}: duty (exothermicity not assessed)")
        if kind not in DEXPI_KIND.values() and kind != "RCT_CSTR" and not md:
            missing.append(f"{st.identifier}: DEXPI class {cls} has no HAZOP rule set (generic flow rows only)")
        units[u.name] = u
        edges += [(s, u.name) for s in inl] + [(u.name, s) for s in outl]
    return ProcessGraph(name=name, streams=streams, units=units, edges=edges), missing


def hazop_from_dexpi(path: str, excel_path: str | None = None) -> dict:
    """Preliminary HAZOP of a process described in a DEXPI file (.xml DEXPI XML or .json pyDEXPI JSON), with no
    simulator involved: the same deterministic rules as hazop_preliminary, instantiated with the T, P, flows and
    compositions written in the file. Returns rows, a Markdown table and `missing` (data the file lacked, e.g. a
    composition, so some hazard flags could not be raised). Optionally writes an .xlsx."""
    from packages.hazop.report import to_excel, to_markdown
    from packages.hazop.rules import hazop_rows
    g, missing = graph_from_dexpi(load_dexpi(path), Path(path).stem)
    rows = hazop_rows(g)
    out = {"n_rows": len(rows), "rows": [r.to_dict() for r in rows], "markdown": to_markdown(rows), "missing": missing,
           "units": {k: v.kind for k, v in g.units.items()}}
    if excel_path:
        out["excel"] = to_excel(rows, excel_path)
    return out
