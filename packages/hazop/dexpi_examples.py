"""Hand-built DEXPI Process models (no simulator involved) to exercise HAZOP-from-DEXPI.

`gas_compression()` is a small wet-gas compression train as a third-party tool would write it: engineering tags,
temperatures in °C, pressures in bar, flows in kg/h, compositions in a material template. Values are typical
design-basis numbers typed by hand, not simulated.
"""
from __future__ import annotations

from pathlib import Path


def gas_compression(with_composition: bool = True):
    from pydexpi.pydexpi_classes import pydexpi_classes as pc
    D = pc.PortDirection
    C, BAR, KGH = pc.TemperatureUnit.DEGREECELSIUS, pc.PressureAbsoluteUnit.BAR, pc.MassFlowRateUnit.KILOGRAMPERHOUR
    qv = lambda v, u: pc.QualifiedValue(value=pc.PhysicalQuantity(value=float(v), unit=u))
    names = ["Methane", "Ethane", "Propane", "N-butane", "Carbon dioxide", "Water"]
    comps = [pc.PureMaterialComponent(identifier=n, label=n) for n in names]
    lst = pc.ListOfMaterialComponents(component=comps)
    tmpl = pc.MaterialTemplate(identifier="MT-GAS", label="wet gas", listOfComponents=lst,
                               description=pc.MultiLanguageString(singleLanguageStrings=[pc.SingleLanguageString(language="en", value="wet natural gas")]),
                               numberOfMaterialComponents=len(names), numberOfPhases=2, phaseLabel=["Vapor", "Liquid"])
    txt = lambda s: pc.MultiLanguageString(singleLanguageStrings=[pc.SingleLanguageString(language="en", value=s)])

    def port(d, ident):
        defaults = {k: f.get_default(call_default_factory=True) for k, f in pc.MaterialPort.model_fields.items()
                    if k not in ("identifier", "nominalDirection", "connectorReference") and not f.is_required()}
        p = pc.MaterialPort.model_construct(identifier=ident, nominalDirection=d, **defaults)
        p.materialTemplateReference = tmpl
        return p

    steps = {
        "SRC-1": pc.Source(identifier="SRC-1"),
        "K-100": pc.Compressing(identifier="K-100", method=pc.CompressionMethod.CENTRIFUGALMOTION, description=txt("Export gas compressor")),
        "E-100": pc.Cooling(identifier="E-100", method=pc.HeatExchangeMethod.TUBULAR, description=txt("Discharge air cooler")),
        "V-100": pc.SeparatingByFlash(identifier="V-100", description=txt("Discharge scrubber")),
        "SNK-GAS": pc.Sink(identifier="SNK-GAS"), "SNK-LIQ": pc.Sink(identifier="SNK-LIQ"),
    }
    ports = {k: [] for k in steps}
    # (label, from, to, T °C, P bar, kg/h, mole fractions)
    wet = [0.84, 0.07, 0.04, 0.02, 0.02, 0.01]
    data = [("100", "SRC-1", "K-100", 25.0, 30.0, 18000.0, wet),
            ("101", "K-100", "E-100", 105.0, 65.0, 18000.0, wet),
            ("102", "E-100", "V-100", 15.0, 64.0, 18000.0, wet),
            ("103", "V-100", "SNK-GAS", 15.0, 64.0, 17850.0, [0.849, 0.0707, 0.0400, 0.0193, 0.0202, 0.0008]),
            ("104", "V-100", "SNK-LIQ", 15.0, 64.0, 150.0, [0.02, 0.01, 0.04, 0.09, 0.0, 0.84])]
    conns, states = [], []
    for k, (lab, a, b, T, P, m, x) in enumerate(data, 1):
        po, pi = port(D.OUTLET, f"{a}.out{len(ports[a]) + 1}"), port(D.INLET, f"{b}.in{len(ports[b]) + 1}")
        ports[a].append(po); ports[b].append(pi)
        kw = {}
        if with_composition:
            st = pc.MaterialStateType(identifier=f"MST-{lab}", label=lab, description=txt("design basis"),
                                      composition=pc.Composition(moleFractions=[pc.QualifiedValue(value=v) for v in x]))
            ms = pc.MaterialState(identifier=f"MS-{lab}", label=lab, state=st, description="design basis")
            states.append(ms)
            kw["materialStateReference"] = ms
        s = pc.Stream(identifier=f"S-{lab}", label=lab, source=po, target=pi, temperature=[qv(T, C)],
                      pressure=[qv(P, BAR)], massFlow=[qv(m, KGH)], **kw)
        po.connectorReference = pi.connectorReference = s
        conns.append(s)
    for k, st in steps.items():
        st.ports = ports[k]
    pm = pc.ProcessModel(processSteps=list(steps.values()), processConnections=conns, materialComponents=comps,
                         listsOfMaterialComponents=[lst], materialTemplates=[tmpl], materialStates=states)
    return pc.EngineeringModel(conceptualModel=pm, originatingSystemName="hand-built example")


def write(model, out_dir: str | Path, stem: str) -> str:
    from pydexpi.loaders.public import DexpiXmlSerializer
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    DexpiXmlSerializer().save(model, Path(out_dir), stem)
    return str(Path(out_dir, f"{stem}.xml"))
