"""The five reference processes as hand-built DEXPI Process files (no simulator involved).

Numbers are illustrative design-basis values typed by hand: the published HAZOPs give the process and the hazards,
rarely the operating point. Units in °C, bar and kg/h on purpose, as a third-party tool would write them.
Each case's node identifiers match `reference.yaml`.
"""
from __future__ import annotations

from pathlib import Path

# (id, DEXPI class, description, extra kwargs as (enum class name, member) or plain values)
Step = tuple[str, str, str, dict]
# (label, from, to, T °C, P bar, kg/h, mole fractions, vapor fraction)
Conn = tuple[str, str, str, float, float, float, list[float], float]


def build_model(name: str, components: list[str], steps: list[Step], conns: list[Conn]):
    from pydexpi.pydexpi_classes import pydexpi_classes as pc
    D = pc.PortDirection
    C, BAR, KGH = pc.TemperatureUnit.DEGREECELSIUS, pc.PressureAbsoluteUnit.BAR, pc.MassFlowRateUnit.KILOGRAMPERHOUR
    qv = lambda v, u: pc.QualifiedValue(value=pc.PhysicalQuantity(value=float(v), unit=u))
    txt = lambda s: pc.MultiLanguageString(singleLanguageStrings=[pc.SingleLanguageString(language="en", value=s)])
    comps = [pc.PureMaterialComponent(identifier=n, label=n) for n in components]
    lst = pc.ListOfMaterialComponents(component=comps)
    tmpl = pc.MaterialTemplate(identifier="MT-1", label=name, listOfComponents=lst, description=txt(name),
                               numberOfMaterialComponents=len(components), numberOfPhases=2, phaseLabel=["Vapor", "Liquid"])

    def port(d, ident):
        defaults = {k: f.get_default(call_default_factory=True) for k, f in pc.MaterialPort.model_fields.items()
                    if k not in ("identifier", "nominalDirection", "connectorReference") and not f.is_required()}
        p = pc.MaterialPort.model_construct(identifier=ident, nominalDirection=d, **defaults)
        p.materialTemplateReference = tmpl
        return p

    objs = {}
    for ident, cls, desc, kw in steps:
        kw = {k: (getattr(getattr(pc, v[0]), v[1]) if isinstance(v, tuple) else v) for k, v in kw.items()}
        if desc:
            kw["description"] = txt(desc)
        objs[ident] = getattr(pc, cls)(identifier=ident, **kw)
    ports = {k: [] for k in objs}
    streams, states = [], []
    for lab, a, b, T, P, m, x, vf in conns:
        po, pi = port(D.OUTLET, f"{a}.out{len(ports[a]) + 1}"), port(D.INLET, f"{b}.in{len(ports[b]) + 1}")
        ports[a].append(po); ports[b].append(pi)
        st = pc.MaterialStateType(identifier=f"MST-{lab}", label=lab, description=txt("design basis"),
                                  composition=pc.Composition(moleFractions=[pc.QualifiedValue(value=v) for v in x]))
        ms = pc.MaterialState(identifier=f"MS-{lab}", label=lab, state=st, description=f"vapor_fraction={vf}")
        states.append(ms)
        s = pc.Stream(identifier=f"S-{lab}", label=lab, source=po, target=pi, temperature=[qv(T, C)],
                      pressure=[qv(P, BAR)], massFlow=[qv(m, KGH)], materialStateReference=ms)
        po.connectorReference = pi.connectorReference = s
        streams.append(s)
    for k, o in objs.items():
        o.ports = ports[k]
    pm = pc.ProcessModel(processSteps=list(objs.values()), processConnections=streams, materialComponents=comps,
                         listsOfMaterialComponents=[lst], materialTemplates=[tmpl], materialStates=states)
    return pc.EngineeringModel(conceptualModel=pm, originatingSystemName=f"AlchemAI book case: {name}")


HX = {"method": ("HeatExchangeMethod", "TUBULAR")}
TANK_RX = {"method": ("ReactionProcessType", "TANK")}


def lawley_dimerizacion():
    comps = ["1-Butene", "N-butane", "Water"]
    feed = [0.600, 0.398, 0.002]
    steps = [("SRC-1", "Source", "", {}), ("T-1", "StoringInTank", "Intermediate storage", {}),
             ("J-1", "Pumping", "Transfer pump", {}), ("LCV-1", "RegulatingFlow", "Half-mile line + settling-tank level valve", {}),
             ("T-2", "SeparatingByGravity", "Buffer/settling tank, manual water draw-off", {}),
             ("SNK-RX", "Sink", "", {}), ("SNK-H2O", "Sink", "", {})]
    conns = [("1", "SRC-1", "T-1", 20, 4.0, 20000, feed, 0.0), ("2", "T-1", "J-1", 20, 4.0, 20000, feed, 0.0),
             ("3", "J-1", "LCV-1", 20.5, 12.0, 20000, feed, 0.0), ("4", "LCV-1", "T-2", 20.5, 6.0, 20000, feed, 0.0),
             ("5", "T-2", "SNK-RX", 20, 6.0, 19960, [0.6012, 0.39875, 0.00005], 0.0),
             ("6", "T-2", "SNK-H2O", 20, 6.0, 40, [0.0005, 0.0005, 0.999], 0.0)]
    return build_model("lawley_dimerizacion", comps, steps, conns)


def ccps_dap():
    comps = ["Ammonia", "Water", "Phosphoric acid", "Diammonium phosphate"]
    nh3, acid = [0.29, 0.71, 0.0, 0.0], [0.0, 0.75, 0.25, 0.0]
    steps = [("SRC-NH3", "Source", "", {}), ("SRC-ACID", "Source", "", {}),
             ("T-1", "StoringInTank", "Ammonia solution storage", {}), ("T-2", "StoringInTank", "Phosphoric acid storage", {}),
             ("CV-A", "RegulatingFlow", "Ammonia feed valve A", {}), ("CV-B", "RegulatingFlow", "Phosphoric acid feed valve B", {}),
             ("R-1", "ReactingChemicals", "Agitated DAP reactor, duty -1500 kW", TANK_RX),
             ("T-3", "StoringInTank", "Open-top DAP storage tank in enclosed work area", {}), ("SNK-LOAD", "Sink", "", {})]
    conns = [("1", "SRC-NH3", "T-1", 25, 1.5, 3000, nh3, 0.0), ("2", "T-1", "CV-A", 25, 3.0, 3000, nh3, 0.0),
             ("3", "CV-A", "R-1", 25, 2.0, 3000, nh3, 0.0),
             ("4", "SRC-ACID", "T-2", 25, 1.5, 4000, acid, 0.0), ("5", "T-2", "CV-B", 25, 3.0, 4000, acid, 0.0),
             ("6", "CV-B", "R-1", 25, 2.0, 4000, acid, 0.0),
             ("7", "R-1", "T-3", 95, 2.0, 7000, [0.001, 0.85, 0.0, 0.149], 0.0),
             ("8", "T-3", "SNK-LOAD", 60, 1.013, 7000, [0.001, 0.85, 0.0, 0.149], 0.0)]
    return build_model("ccps_dap", comps, steps, conns)


def crowl_louvar_reactor():
    comps = ["Reactant A", "Solvent", "Product B"]
    steps = [("SRC-1", "Source", "", {}), ("FV-1", "RegulatingFlow", "Feed valve", {}),
             ("R-1", "ReactingChemicals", "Exothermic stirred reactor with cooling-water coil, duty -400 kW", TANK_RX),
             ("SNK-1", "Sink", "", {})]
    conns = [("1", "SRC-1", "FV-1", 30, 3.0, 5000, [0.3, 0.7, 0.0], 0.0), ("2", "FV-1", "R-1", 30, 2.0, 5000, [0.3, 0.7, 0.0], 0.0),
             ("3", "R-1", "SNK-1", 80, 2.0, 5000, [0.03, 0.7, 0.27], 0.0)]
    return build_model("crowl_louvar_reactor", comps, steps, conns)


def csb_texas_city():
    comps = ["N-pentane", "N-hexane", "N-heptane"]
    feed, top, bot = [0.30, 0.40, 0.30], [0.65, 0.33, 0.02], [0.05, 0.45, 0.50]
    steps = [("SRC-1", "Source", "", {}), ("E-1", "ExchangingThermalEnergy", "Feed/bottoms exchanger", HX),
             ("T-1", "Distilling", "Raffinate splitter tower", {}), ("F-20", "SeparatingByGravity", "Blowdown drum with stack", {}),
             ("SNK-LR", "Sink", "", {}), ("SNK-HR", "Sink", "", {}), ("SNK-STACK", "Sink", "", {}), ("SNK-SEWER", "Sink", "", {})]
    conns = [("1", "SRC-1", "E-1", 60, 4.0, 60000, feed, 0.0), ("2", "E-1", "T-1", 110, 3.5, 60000, feed, 0.0),
             ("3", "T-1", "SNK-LR", 75, 2.4, 25000, top, 0.0), ("4", "T-1", "E-1", 140, 2.6, 35000, bot, 0.0),
             ("5", "E-1", "SNK-HR", 85, 2.5, 35000, bot, 0.0), ("6", "T-1", "F-20", 120, 2.5, 0.0, top, 1.0),
             ("7", "F-20", "SNK-STACK", 60, 1.013, 0.0, top, 1.0), ("8", "F-20", "SNK-SEWER", 60, 1.013, 0.0, bot, 0.0)]
    return build_model("csb_texas_city", comps, steps, conns)


def csb_williams():
    comps = ["Propane", "Propylene", "Water"]
    steps = [("SRC-QW", "Source", "", {}), ("SRC-FEED", "Source", "", {}),
             ("E-B", "ExchangingThermalEnergy", "Standby reboiler B, quench-water heated", HX),
             ("T-100", "Distilling", "Propylene fractionator", {}),
             ("SNK-QW", "Sink", "", {}), ("SNK-C3=", "Sink", "", {}), ("SNK-C3", "Sink", "", {})]
    conns = [("1", "SRC-QW", "E-B", 85, 5.0, 200000, [0.0, 0.0, 1.0], 0.0), ("2", "E-B", "SNK-QW", 70, 4.5, 200000, [0.0, 0.0, 1.0], 0.0),
             ("3", "T-100", "E-B", 50, 19.0, 30000, [0.95, 0.05, 0.0], 0.0), ("4", "E-B", "T-100", 52, 19.0, 30000, [0.95, 0.05, 0.0], 0.4),
             ("5", "SRC-FEED", "T-100", 40, 20.0, 40000, [0.30, 0.70, 0.0], 0.0),
             ("6", "T-100", "SNK-C3=", 45, 18.5, 28000, [0.01, 0.99, 0.0], 0.0), ("7", "T-100", "SNK-C3", 50, 19.0, 12000, [0.95, 0.05, 0.0], 0.0)]
    return build_model("csb_williams", comps, steps, conns)


CASES = {f.__name__: f for f in (lawley_dimerizacion, ccps_dap, crowl_louvar_reactor, csb_texas_city, csb_williams)}


def write_all(out_dir: str | Path, cases: dict | None = None) -> dict[str, str]:
    from packages.hazop.dexpi_examples import write
    return {k: write(f(), out_dir, k) for k, f in (cases or CASES).items()}
