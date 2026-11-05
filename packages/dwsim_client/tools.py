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
              "equilibrium_reactor", "gibbs_reactor", "cstr", "pfr", "recycle", "pipe"]


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
                       "reactions": {}, "column_feeds": {}, "stream_specs": {}, "params": {}, "conn": {}}
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
    s["stream_specs"][name] = {k: v for k, v in dict(temperature_K=temperature_K, pressure_Pa=pressure_Pa, vapor_fraction=vapor_fraction,
                                                      molar_flow_mol_s=molar_flow_mol_s, mass_flow_kg_s=mass_flow_kg_s, mole_fractions=mole_fractions,
                                                      compound_molar_flows_mol_s=compound_molar_flows_mol_s).items() if v}
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


def define_kinetic_reaction(flowsheet_id: str, name: str, stoichiometry: dict[str, float], base_compound: str,
                            direct_orders: dict[str, float], A_forward: float, E_forward_J_mol: float,
                            reverse_orders: dict[str, float] | None = None, A_reverse: float = 0.0, E_reverse_J_mol: float = 0.0,
                            reaction_set: str = "KIN") -> dict:
    """Define an Arrhenius kinetic reaction for CSTR/PFR units: r = A·exp(−E/RT)·Π C_i^order (vapor phase, molar
    concentration basis in mol/m³, rate in mol/[m³·s]; so for a first-order reaction A_forward is in 1/s). E in
    J/mol. direct_orders gives the forward orders per compound (compounds not listed get 0); reverse_orders
    likewise for the reverse rate (A_reverse=0 → irreversible). Add it to a reaction set and bind that set to the
    reactor via add_unit(kind="cstr"|"pfr", params={"reaction_set": ...}). Validated: first-order PFR/CSTR match
    the analytic X = 1−e^(−kτ) / kτ/(1+kτ) within 0.05 %."""
    s = _s(flowsheet_id)
    comps = list(stoichiometry)
    fwd = {c: float((direct_orders or {}).get(c, 0.0)) for c in comps}
    rev = {c: float((reverse_orders or {}).get(c, 0.0)) for c in comps}
    r = s["fs"].DefineKineticReaction(name, _stoich(stoichiometry), _stoich(fwd), _stoich(rev), base_compound, "Vapor",
                                      "Molar Concentration", "mol/m3", "mol/[m3.s]", float(A_forward), float(E_forward_J_mol),
                                      float(A_reverse), float(E_reverse_J_mol), "", "", "")
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
      condenser_P_Pa, reboiler_P_Pa, reflux_ratio; distillation_column: stages, feed_stage (1-based, 1 = condenser), reflux_ratio,
      bottoms_molar_flow_mol_s | distillate_molar_flow_mol_s, top_P_Pa, dP_Pa; *_reactor: reaction_set,
      mode ("isothermal" | "adiabatic"), dP_Pa; cstr / pfr (kinetic reactions): volume_m3, reaction_set,
      mode ("isothermal" at the feed T | "adiabatic"), dP_Pa; pipe (adiabatic, Beggs-Brill, handles two-phase): length_m,
      inner_diameter_m, elevation_change_m (outlet minus inlet), increments (default 10); commercial steel.
      Connect streams afterwards with connect()."""
    try:
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
        elif kind in ("cstr", "pfr"):
            b = (fs.AddCSTR if kind == "cstr" else fs.AddPFR)(name).WithVolume(Q.CubicMeters(float(p["volume_m3"])))
            # NOTE: CSTRBuilder.WithIsothermalTemperature breaks the PH flash in DWSIM 10.2.12; Isothermal() = feed T.
            b = b.Adiabatic() if str(p.get("mode", "isothermal")).lower().startswith("adiab") else b.Isothermal()
            if "reaction_set" in p:
                b = b.WithReactionSet(p["reaction_set"])
            b = b.WithPressureDrop(Q.Pascal(float(p.get("dP_Pa", 0.0))))
        elif kind == "pipe":
            import DWSIM.UnitOperations.UnitOperations.Auxiliary.Pipe as AP
            b = fs.AddPipe(name)
            o = getattr(b.Object, "__implementation__", b.Object)
            di_in = float(p["inner_diameter_m"]) / 0.0254          # DWSIM pipe sections take diameters in inches
            sec = AP.PipeSection(1, "Tubulaosimples", 1, int(p.get("increments", 10)), "Aço Comum", float(p["length_m"]),
                                 float(p.get("elevation_change_m", 0.0)), di_in * 1.12, di_in)
            o.Profile.Sections.Add(1, sec)
            o.Profile.Status = type(o.Profile.Status).OK
            tp = o.ThermalProfile
            tp.TipoPerfil = type(tp.TipoPerfil).Definir_Q
            tp.Calor_trocado = 0.0
        elif kind == "recycle":
            b = fs.AddUnitOperation(ObjectType.OT_Recycle, name)
            r = b.Object.__implementation__
            cp = r.ConvergenceParameters
            cp.VazaoMassica, cp.VazaoMassicaRelativa, cp.Temperatura, cp.Entalpia, cp.Composicao = 1e-6, 1e-8, 1e-4, 1e-4, 1e-7
            r.MaximumIterations = 500
    except KeyError as ex:
        raise ValueError(f"{kind} {name!r}: missing required param {ex.args[0]!r} (see add_unit docstring); dof_check() lists what each kind needs") from None
    s["units"][name] = b
    s["kinds"][name] = kind
    s["params"][name] = dict(p)
    return {"ok": True, "unit": name, "kind": kind}


def connect(flowsheet_id: str, unit: str, feeds: list[str] | None = None, products: list[str] | None = None,
            energy_feeds: list[str] | None = None, energy_products: list[str] | None = None) -> dict:
    """Connect streams to a unit, in port order. Conventions: separator products = [vapor, liquid];
    shortcut/distillation column products = [distillate, bottoms], energy_products = [condenser duty],
    energy_feeds = [reboiler duty]; conversion/equilibrium/gibbs reactors products = [vapor, liquid] (both
    required), energy_feeds = [heat duty]; cstr/pfr products = [outlet], energy_feeds = [heat duty] (required,
    also in isothermal mode); heat_exchanger feeds = [hot_in, cold_in], products = [hot_out, cold_out];
    recycle feeds = [tear stream], products = [recycled stream]."""
    s = _s(flowsheet_id)
    b, kind = s["units"][unit], s["kinds"][unit]
    st, en = s["streams"], s["energy"]
    feeds, products = feeds or [], products or []
    energy_feeds, energy_products = energy_feeds or [], energy_products or []
    s["conn"][unit] = {"feeds": list(feeds), "products": list(products), "energy_feeds": list(energy_feeds), "energy_products": list(energy_products)}

    if kind == "distillation_column":
        for f in feeds:
            # DWSIM's WithFeed index is 0-based (2 -> "Stage3"); feed_stage is 1-based like every textbook (1 = condenser).
            b = b.WithFeed(st[f], s["column_feeds"][unit] - 1)
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
    elif kind.endswith("_reactor") or kind in ("cstr", "pfr"):
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
    if kind == "pipe":
        r = getattr(o, "__implementation__", o)
        sec = list(r.Profile.Sections.Values)[0]
        res = list(sec.Results)
        out.update(dP_friction_Pa=float(r.PressureDrop_Friction), dP_static_Pa=float(r.PressureDrop_Static),
                   outlet_P_Pa=float(r.OutletPressure) if r.OutletPressure else None,
                   liquid_velocity_m_s=float(res[0].LiqVel) if res else None, liquid_Re=float(res[0].LiqRe) if res else None,
                   flow_regime=str(res[-1].FlowRegime) if res else None)
    if kind == "recycle":
        r = o.__implementation__
        out.update(converged=bool(r.Converged), iterations=int(r.IterationsTaken))
    out["energy_streams_kW"] = {k: e.EnergyFlowKW for k, e in s["energy"].items()}
    return out


def close_flowsheet(flowsheet_id: str) -> dict:
    """Discard a flowsheet session."""
    SESSIONS.pop(flowsheet_id, None)
    return {"ok": True}


from packages.dwsim_client.tools_sweep import SWEEP_TOOLS
from packages.dwsim_client.tools_transport import TRANSPORT_TOOLS
from packages.dwsim_client.tools_costing import COSTING_TOOLS
from packages.dwsim_client.dof import dof_check  # noqa: E402  (imports this module; placed last on purpose)
from packages.hazop.report import hazop_preliminary  # noqa: E402
from packages.export.flowsheet_export import export_flowsheet  # noqa: E402
from packages.hazop.dexpi_graph import hazop_from_dexpi  # noqa: E402

TOOLS = [list_property_packages, new_flowsheet, add_material_stream, add_energy_stream, define_conversion_reaction,
         define_equilibrium_reaction, define_kinetic_reaction, add_unit, connect, dof_check, solve, read_stream, read_unit,
         *SWEEP_TOOLS, *TRANSPORT_TOOLS, *COSTING_TOOLS, hazop_preliminary, export_flowsheet, hazop_from_dexpi, close_flowsheet]
