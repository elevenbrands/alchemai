"""Tool registry over DWSIM's Fluent API — the single surface the agent and the MCP server share.

Every function takes/returns plain JSON-friendly values so it can be exposed through
`@beta_tool` (Anthropic SDK), FastMCP, or called directly. State lives in `SESSIONS`
(one headless flowsheet per session id). Nothing here reasons; DWSIM computes, `verify.py`
checks, and the caller decides what to report.

Units: K, Pa, mol/s, kg/s, kW. Compositions are mole fractions.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client.fluent import fluent  # noqa: E402
from packages.dwsim_client.verify import verify, _prop  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
import System  # noqa: E402
from System import Double, String  # noqa: E402
from System.Collections.Generic import Dictionary  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # noqa: E402

SESSIONS: dict[str, dict[str, Any]] = {}
_ids = itertools.count(1)

UNIT_KINDS = ["mixer", "splitter", "heater", "cooler", "valve", "pump", "compressor", "expander", "separator",
              "heat_exchanger", "shortcut_column", "distillation_column", "conversion_reactor",
              "equilibrium_reactor", "gibbs_reactor", "recycle"]


def _s(fs_id: str) -> dict:
    if fs_id not in SESSIONS:
        raise KeyError(f"unknown flowsheet id {fs_id!r}; call new_flowsheet first")
    return SESSIONS[fs_id]


def _stoich(d: dict[str, float]):
    out = Dictionary[String, Double]()
    for k, v in d.items():
        out[k] = float(v)
    return out


def _composition(fracs: dict[str, float]):
    pairs = [(k, float(v)) for k, v in fracs.items()]

    def _set(cb):
        for name, z in pairs:
            cb.Mole(name, z)

    return System.Action[CompositionBuilder](_set)


# ----------------------------------------------------------------------------- flowsheet
def list_property_packages() -> list[str]:
    """List the thermodynamic property packages available in DWSIM (e.g. PengRobinson, NRTL, SteamTables)."""
    return sorted(p for p in dir(PropertyPackages) if p[0].isupper()
                  and p not in ("Equals", "Finalize", "GetHashCode", "GetType", "MemberwiseClone", "Overloads",
                                "ReferenceEquals", "ToString", "Plus", "RequiresPlus", "CapeOpen"))


def new_flowsheet(name: str, compounds: list[str], property_package: str) -> dict:
    """Create an empty headless DWSIM flowsheet with the given compounds (DWSIM names, e.g. "Methane",
    "N-butane", "Carbon dioxide", "Water", "Ethanol") and property package. Returns {"flowsheet_id"}."""
    if not hasattr(PropertyPackages, property_package):
        raise ValueError(f"unknown property package {property_package!r}; see list_property_packages()")
    fs = Flowsheet.Create(name)
    for c in compounds:
        fs = fs.WithCompound(c)
    fs = fs.WithPropertyPackage(getattr(PropertyPackages, property_package))
    fs_id = f"fs{next(_ids)}"
    SESSIONS[fs_id] = {"fs": fs, "streams": {}, "energy": {}, "units": {}, "kinds": {}, "compounds": list(compounds),
                       "reactions": {}, "column_feeds": {}}
    return {"flowsheet_id": fs_id, "compounds": list(compounds), "property_package": property_package}


def add_material_stream(flowsheet_id: str, name: str, temperature_K: float | None = None, pressure_Pa: float | None = None,
                        vapor_fraction: float | None = None, molar_flow_mol_s: float | None = None,
                        mass_flow_kg_s: float | None = None, mole_fractions: dict[str, float] | None = None,
                        compound_molar_flows_mol_s: dict[str, float] | None = None) -> dict:
    """Add a material stream. Feed streams need two of (T, P, vapor_fraction), one flow, and a composition
    (mole_fractions summing to 1, or compound_molar_flows_mol_s). Product streams: just the name."""
    s = _s(flowsheet_id)
    b = s["fs"].AddMaterialStream(name)
    if temperature_K is not None:
        b = b.WithTemperature(Q.Kelvin(float(temperature_K)))
    if pressure_Pa is not None:
        b = b.WithPressure(Q.Pascal(float(pressure_Pa)))
    if vapor_fraction is not None:
        b = b.WithVaporFraction(float(vapor_fraction))
    if molar_flow_mol_s is not None:
        b = b.WithMolarFlow(Q.MolPerSecond(float(molar_flow_mol_s)))
    if mass_flow_kg_s is not None:
        b = b.WithMassFlow(Q.KgPerSecond(float(mass_flow_kg_s)))
    if mole_fractions:
        b = b.WithComposition(_composition(mole_fractions))
    if compound_molar_flows_mol_s:
        for k, v in compound_molar_flows_mol_s.items():
            b = b.SetCompoundMolarFlow(k, float(v))
    s["streams"][name] = b
    return {"ok": True, "stream": name}


def add_energy_stream(flowsheet_id: str, name: str) -> dict:
    """Add an energy stream (duty or power carrier) to connect to a unit's energy port."""
    s = _s(flowsheet_id)
    s["energy"][name] = s["fs"].AddEnergyStream(name)
    return {"ok": True, "energy_stream": name}


def define_conversion_reaction(flowsheet_id: str, name: str, stoichiometry: dict[str, float], base_compound: str,
                               conversion_percent: float, reaction_set: str = "RS1") -> dict:
    """Define a conversion reaction (stoichiometry: negative for reactants, positive for products; e.g.
    {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}) and add it to a reaction set."""
    s = _s(flowsheet_id)
    r = s["fs"].DefineConversionReaction(name, _stoich(stoichiometry), base_compound, "Vapor", str(float(conversion_percent)))
    s["fs"].ReactionSet(reaction_set).Add(r)
    s["reactions"][name] = reaction_set
    return {"ok": True, "reaction": name, "reaction_set": reaction_set}


def define_equilibrium_reaction(flowsheet_id: str, name: str, stoichiometry: dict[str, float], base_compound: str,
                                lnK_expression_of_T: str, reaction_set: str = "RS1") -> dict:
    """Define an equilibrium reaction with ln K as an expression of T in kelvin, e.g. "4577.8/T - 4.33"
    (activity basis, vapor phase), and add it to a reaction set."""
    s = _s(flowsheet_id)
    r = s["fs"].DefineEquilibriumReaction(name, _stoich(stoichiometry), base_compound, "Vapor", "Activity", "",
                                          lnK_expression_of_T, 0.0)
    s["fs"].ReactionSet(reaction_set).Add(r)
    s["reactions"][name] = reaction_set
    return {"ok": True, "reaction": name, "reaction_set": reaction_set}


def add_unit(flowsheet_id: str, kind: str, name: str, params: dict[str, Any] | None = None) -> dict:
    """Add a unit operation. kind ∈ mixer, splitter, heater, cooler, valve, pump, compressor, expander,
    separator, heat_exchanger, shortcut_column, distillation_column, conversion_reactor,
    equilibrium_reactor, gibbs_reactor, recycle. params by kind:
      splitter: split_ratios [..]; heater: outlet_T_K | heat_added_kW; cooler: outlet_T_K | heat_removed_kW;
      valve: outlet_P_Pa; pump: outlet_P_Pa, efficiency_pct; compressor/expander: outlet_P_Pa,
      adiabatic_eff_pct | polytropic_eff_pct; heat_exchanger: UA_W_K (pinch-point mode), hot_dP_Pa, cold_dP_Pa;
      shortcut_column: light_key, heavy_key, lk_mole_frac_in_bottoms, hk_mole_frac_in_distillate,
      condenser_P_Pa, reboiler_P_Pa, reflux_ratio; distillation_column: stages, feed_stage, reflux_ratio,
      bottoms_molar_flow_mol_s | distillate_molar_flow_mol_s, top_P_Pa, dP_Pa; *_reactor: reaction_set,
      mode ("isothermal" | "adiabatic"), dP_Pa. Connect streams afterwards with connect()."""
    s = _s(flowsheet_id)
    fs, p = s["fs"], dict(params or {})
    kind = kind.lower()
    if kind not in UNIT_KINDS:
        raise ValueError(f"unknown kind {kind!r}; use one of {UNIT_KINDS}")

    if kind == "mixer":
        b = fs.AddMixer(name)
    elif kind == "splitter":
        b = fs.AddSplitter(name).WithSplitRatios(*[float(x) for x in p["split_ratios"]])
    elif kind == "heater":
        b = fs.AddHeater(name)
        if "outlet_T_K" in p:
            b = b.WithOutletTemperature(Q.Kelvin(float(p["outlet_T_K"])))
        elif "heat_added_kW" in p:
            b = b.WithHeatAdded(Q.Kilowatt(float(p["heat_added_kW"])))
    elif kind == "cooler":
        b = fs.AddCooler(name)
        if "outlet_T_K" in p:
            b = b.WithOutletTemperature(Q.Kelvin(float(p["outlet_T_K"])))
        elif "heat_removed_kW" in p:
            b = b.WithHeatRemoved(Q.Kilowatt(float(p["heat_removed_kW"])))
    elif kind == "valve":
        b = fs.AddValve(name).WithOutletPressure(Q.Pascal(float(p["outlet_P_Pa"])))
    elif kind == "pump":
        b = fs.AddPump(name).WithOutletPressure(Q.Pascal(float(p["outlet_P_Pa"]))).WithEfficiencyPercent(float(p.get("efficiency_pct", 75.0)))
    elif kind in ("compressor", "expander"):
        b = (fs.AddCompressor(name) if kind == "compressor" else fs.AddExpander(name)).WithOutletPressure(Q.Pascal(float(p["outlet_P_Pa"])))
        if "polytropic_eff_pct" in p:
            from DWSIM.UnitOperations.UnitOperations import Compressor
            b = b.WithProcessPath(Compressor.ProcessPathType.Polytropic).WithPolytropicEfficiencyPercent(float(p["polytropic_eff_pct"]))
        else:
            b = b.WithAdiabaticEfficiencyPercent(float(p.get("adiabatic_eff_pct", 75.0)))
    elif kind == "separator":
        b = fs.AddSeparator(name)
    elif kind == "heat_exchanger":
        from DWSIM.UnitOperations.UnitOperations import HeatExchangerCalcMode
        b = (fs.AddHeatExchanger(name).WithCalculationMode(HeatExchangerCalcMode.PinchPoint).WithGlobalUA(float(p["UA_W_K"]))
             .WithHotSidePressureDrop(Q.Pascal(float(p.get("hot_dP_Pa", 0.0)))).WithColdSidePressureDrop(Q.Pascal(float(p.get("cold_dP_Pa", 0.0)))))
    elif kind == "shortcut_column":
        b = (fs.AddShortcutColumn(name).WithKeys(p["light_key"], p["heavy_key"])
             .WithLightKeyInBottoms(float(p["lk_mole_frac_in_bottoms"])).WithHeavyKeyInDistillate(float(p["hk_mole_frac_in_distillate"]))
             .WithPressures(Q.Pascal(float(p["condenser_P_Pa"])), Q.Pascal(float(p["reboiler_P_Pa"]))).WithRefluxRatio(float(p["reflux_ratio"])))
    elif kind == "distillation_column":
        b = (fs.AddDistillationColumn(name).WithNumberOfStages(int(p["stages"]))
             .WithCondenserSpec("Reflux Ratio", float(p["reflux_ratio"]), "")
             .WithTopPressure(Q.Pascal(float(p["top_P_Pa"]))).WithColumnPressureDrop(Q.Pascal(float(p.get("dP_Pa", 0.0)))))
        if "bottoms_molar_flow_mol_s" in p:
            b = b.WithReboilerSpec("Product Molar Flow Rate", float(p["bottoms_molar_flow_mol_s"]), "mol/s")
        elif "distillate_molar_flow_mol_s" in p:
            b = b.WithCondenserSpec("Product Molar Flow Rate", float(p["distillate_molar_flow_mol_s"]), "mol/s")
        s["column_feeds"][name] = int(p.get("feed_stage", max(1, int(p["stages"]) // 2)))
    elif kind in ("conversion_reactor", "equilibrium_reactor", "gibbs_reactor"):
        b = {"conversion_reactor": fs.AddConversionReactor, "equilibrium_reactor": fs.AddEquilibriumReactor,
             "gibbs_reactor": fs.AddGibbsReactor}[kind](name)
        b = b.Adiabatic() if str(p.get("mode", "isothermal")).lower().startswith("adiab") else b.Isothermal()
        if "reaction_set" in p:
            b = b.WithReactionSet(p["reaction_set"])
        b = b.WithPressureDrop(Q.Pascal(float(p.get("dP_Pa", 0.0))))
    elif kind == "recycle":
        b = fs.AddUnitOperation(ObjectType.OT_Recycle, name)
        r = b.Object.__implementation__
        cp = r.ConvergenceParameters
        cp.VazaoMassica, cp.VazaoMassicaRelativa, cp.Temperatura, cp.Entalpia, cp.Composicao = 1e-6, 1e-8, 1e-4, 1e-4, 1e-7
        r.MaximumIterations = 500
    s["units"][name] = b
    s["kinds"][name] = kind
    return {"ok": True, "unit": name, "kind": kind}


def connect(flowsheet_id: str, unit: str, feeds: list[str] | None = None, products: list[str] | None = None,
            energy_feeds: list[str] | None = None, energy_products: list[str] | None = None) -> dict:
    """Connect streams to a unit, in port order. Conventions: separator products = [vapor, liquid];
    shortcut/distillation column products = [distillate, bottoms], energy_products = [condenser duty],
    energy_feeds = [reboiler duty]; reactors products = [vapor, liquid] (both required), energy_feeds =
    [heat duty]; heat_exchanger feeds = [hot_in, cold_in], products = [hot_out, cold_out];
    recycle feeds = [tear stream], products = [recycled stream]."""
    s = _s(flowsheet_id)
    b, kind = s["units"][unit], s["kinds"][unit]
    st, en = s["streams"], s["energy"]
    feeds, products = feeds or [], products or []
    energy_feeds, energy_products = energy_feeds or [], energy_products or []

    if kind == "distillation_column":
        for f in feeds:
            b = b.WithFeed(st[f], s["column_feeds"][unit])
        if products:
            b = b.WithDistillate(st[products[0]])
        if len(products) > 1:
            b = b.WithBottoms(st[products[1]])
        if energy_products:
            b = b.WithCondenserDuty(en[energy_products[0]])
        if energy_feeds:
            b = b.WithReboilerDuty(en[energy_feeds[0]])
        return {"ok": True, "unit": unit}

    for i, f in enumerate(feeds):
        b.ConnectFeed(st[f], i)
    for i, pr in enumerate(products):
        b.ConnectProduct(st[pr], i)
    if kind == "shortcut_column":
        for e in energy_products:
            b.ConnectEnergyProduct(en[e], 0)
        for e in energy_feeds:
            b.ConnectEnergyFeed(en[e], 1)
    elif kind.endswith("_reactor"):
        for e in energy_feeds:
            b.ConnectEnergyFeed(en[e], 1)
    else:
        for i, e in enumerate(energy_feeds):
            _try_ports(b.ConnectEnergyFeed, en[e], (1, 0, 2))
        for i, e in enumerate(energy_products):
            _try_ports(b.ConnectEnergyProduct, en[e], (0, 1, 2))
    return {"ok": True, "unit": unit}


def _try_ports(fn, stream, ports):
    last = None
    for port in ports:
        try:
            fn(stream, port)
            return
        except Exception as ex:  # noqa: BLE001
            last = ex
    raise RuntimeError(f"no energy port accepted the connection: {last}")


def solve(flowsheet_id: str) -> dict:
    """Solve the flowsheet and run the deterministic verifier. Returns convergence, per-unit mass/compound/
    energy balance residuals, solver errors and a table of every stream. Nothing is trustworthy until
    this returns ok=true; report anything else as unverified."""
    s = _s(flowsheet_id)
    s["fs"].AutoLayout()
    rep = verify(s["fs"])
    s["report"] = rep
    return {"solved": rep.solved, "ok": rep.ok, "solver_errors": rep.solver_errors,
            "units": [{"name": u.name, "kind": u.kind, "calculated": u.calculated, "error": u.error,
                       "mass_rel_err": u.mass_rel_err, "compound_max_rel_err": u.compound_max_rel_err,
                       "energy_rel_err": u.energy_rel_err, "duty_kW": u.duty_kw, "ok": u.ok, "notes": u.notes} for u in rep.units],
            "streams": {k: {"T_K": v.temperature_k, "P_Pa": v.pressure_pa, "vapor_fraction": v.vapor_fraction,
                            "molar_flow_mol_s": v.molar_mol_s, "mass_flow_kg_s": v.mass_kg_s,
                            "mole_fractions": {c: (f / v.molar_mol_s if v.molar_mol_s else 0.0) for c, f in v.compounds_mol_s.items()}}
                        for k, v in rep.streams.items()}}


def read_stream(flowsheet_id: str, name: str) -> dict:
    """Read a solved material stream: T, P, vapor fraction, flows and overall mole fractions."""
    s = _s(flowsheet_id)
    b = s["streams"][name]
    return {"name": name, "T_K": b.TemperatureK, "P_Pa": b.PressurePa, "vapor_fraction": b.VaporFraction,
            "molar_flow_mol_s": b.MolarFlowMolPerSecond, "mass_flow_kg_s": b.MassFlowKgPerSecond,
            "mole_fractions": {c: b.OverallMoleFraction(c) for c in s["compounds"]}}


def read_unit(flowsheet_id: str, name: str) -> dict:
    """Read unit results: duty/power (kW), compressor efficiencies, shortcut-column N_min/R_min/N/feed stage,
    heat-exchanger Q, reactor heat duty, recycle convergence, and the energy flows of attached streams."""
    s = _s(flowsheet_id)
    b, kind = s["units"][name], s["kinds"][name]
    o = b.Object
    out: dict[str, Any] = {"name": name, "kind": kind}
    for prop, key in (("DeltaQ", "duty_or_power_kW"), ("Q", "Q_kW"), ("AdiabaticEfficiency", "adiabatic_eff_pct"),
                      ("PolytropicEfficiency", "polytropic_eff_pct")):
        v = _prop(o, prop)
        if v is not None:
            try:
                out[key] = float(v)
            except TypeError:
                pass
    if kind == "shortcut_column":
        out.update(N_min=b.MinimumStages, R_min=b.MinimumRefluxRatio, N_theoretical=b.TheoreticalStages,
                   feed_stage=b.OptimumFeedStage, condenser_duty_kW=b.CondenserDutyKW, reboiler_duty_kW=b.ReboilerDutyKW)
    if kind.endswith("_reactor"):
        out["heat_duty_kW"] = b.HeatDutyKW
    if kind == "pump":
        out["power_kW"] = b.PowerKW
    if kind == "cooler":
        out["heat_removed_kW"] = b.HeatRemovedKW
    if kind == "recycle":
        r = o.__implementation__
        out.update(converged=bool(r.Converged), iterations=int(r.IterationsTaken))
    out["energy_streams_kW"] = {k: e.EnergyFlowKW for k, e in s["energy"].items()}
    return out


def close_flowsheet(flowsheet_id: str) -> dict:
    """Discard a flowsheet session."""
    SESSIONS.pop(flowsheet_id, None)
    return {"ok": True}


from packages.dwsim_client.tools_sweep import SWEEP_TOOLS  # noqa: E402  (imports this module; placed last on purpose)

TOOLS = [list_property_packages, new_flowsheet, add_material_stream, add_energy_stream, define_conversion_reaction,
         define_equilibrium_reaction, add_unit, connect, solve, read_stream, read_unit, *SWEEP_TOOLS, close_flowsheet]
