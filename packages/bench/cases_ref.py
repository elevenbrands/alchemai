"""Ten reference flowsheets (block 2). Each case is a card + a builder + readouts + literature checks.

* `build()`      -> (flowsheet, handles)   built with the Fluent API, not yet solved
* `readouts(h)`  -> {name: float}          numbers worth comparing (DWSIM reference snapshot in
                                           expected_ref.yaml; tests regress against it)
* `checks(h, r)` -> [(name, ok, detail)]   independent sanity checks from textbook physics, so a
                                           wrong engine result is caught even if the snapshot
                                           was wrong too.

Statements are given in Spanish and English because the bench will be bilingual.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client.fluent import fluent  # noqa: E402
from packages.bench.cases_baseline import composition  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
from System import Double, String  # noqa: E402
from System.Collections.Generic import Dictionary  # noqa: E402
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # noqa: E402


def stoich(d: dict[str, float]):
    out = Dictionary[String, Double]()
    for k, v in d.items():
        out[k] = float(v)
    return out


def _prop(obj, name, default=None):
    p = obj.GetType().GetProperty(name)
    if p is None:
        return default
    v = p.GetValue(obj)
    return default if v is None else v


@dataclass
class Case:
    id: str
    title_es: str
    title_en: str
    statement_es: str
    statement_en: str
    build: Callable
    readouts: Callable
    checks: Callable
    tags: list[str] = field(default_factory=list)


def _with(fs, names):
    for n in names:
        fs = fs.WithCompound(n)
    return fs


# --------------------------------------------------------------------------- 1 mixer
def b_mixer():
    names = ["Methane", "Ethane", "Propane"]
    fs = _with(Flowsheet.Create("mixer"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    a = fs.AddMaterialStream("a").At(Q.Kelvin(280.0), Q.Bar(20.0)).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, [0.9, 0.08, 0.02])))
    b = fs.AddMaterialStream("b").At(Q.Kelvin(340.0), Q.Bar(22.0)).WithMolarFlow(Q.MolPerSecond(50.0)).WithComposition(composition(zip(names, [0.5, 0.3, 0.2])))
    out = fs.AddMaterialStream("out")
    fs.AddMixer("MIX-1").ConnectFeed(a, 0).ConnectFeed(b, 1).ConnectProduct(out, 0)
    fs.AutoLayout()
    return fs, {"a": a, "b": b, "out": out}


def r_mixer(h):
    return {"T_out_K": h["out"].TemperatureK, "P_out_bar": h["out"].PressurePa / 1e5,
            "x_CH4_out": h["out"].OverallMoleFraction("Methane"), "F_out_mol_s": h["out"].MolarFlowMolPerSecond}


def c_mixer(h, rep):
    t = h["out"].TemperatureK
    x = h["out"].OverallMoleFraction("Methane")
    return [("T_out entre las entradas", 280.0 < t < 340.0, f"T={t:.2f} K"),
            ("x_CH4 = promedio molar (0.7667)", abs(x - (100 * 0.9 + 50 * 0.5) / 150) < 1e-6, f"x={x:.5f}"),
            ("P_out = mínima entrada (20 bar)", abs(h["out"].PressurePa / 1e5 - 20.0) < 1e-6, f"P={h['out'].PressurePa / 1e5:.3f} bar")]


# --------------------------------------------------------------------------- 2 Joule-Thomson valve
def b_valve():
    names = ["Methane", "Ethane", "Propane", "N-butane", "Carbon dioxide"]
    fs = _with(Flowsheet.Create("valve"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    gas = fs.AddMaterialStream("gas").At(Q.Kelvin(300.0), Q.Bar(50.0)).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, [0.85, 0.07, 0.04, 0.02, 0.02])))
    low = fs.AddMaterialStream("low")
    fs.AddValve("VLV-1").WithOutletPressure(Q.Bar(10.0)).ConnectFeed(gas, 0).ConnectProduct(low, 0)
    fs.AutoLayout()
    return fs, {"gas": gas, "low": low}


def r_valve(h):
    return {"T_out_K": h["low"].TemperatureK, "dT_K": h["low"].TemperatureK - 300.0}


def c_valve(h, rep):
    dt = 300.0 - h["low"].TemperatureK
    return [("Enfría por Joule-Thomson (ΔT entre 10 y 40 K para 40 bar)", 10.0 < dt < 40.0, f"ΔT={dt:.2f} K"),
            ("Isoentálpica (ΔH − Q ≈ 0)", abs(rep.units[0].energy_residual_kw) < 1e-3, f"res={rep.units[0].energy_residual_kw:.2e} kW")]


# --------------------------------------------------------------------------- 3 separator (wet gas)
def b_separator():
    names = ["Methane", "Ethane", "N-pentane", "N-heptane"]
    fs = _with(Flowsheet.Create("sep"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    feed = fs.AddMaterialStream("feed").At(Q.Kelvin(290.0), Q.Bar(20.0)).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, [0.70, 0.10, 0.10, 0.10])))
    vap, liq = fs.AddMaterialStream("vap"), fs.AddMaterialStream("liq")
    fs.AddSeparator("V-1").ConnectFeed(feed, 0).ConnectProduct(vap, 0).ConnectProduct(liq, 1)
    fs.AutoLayout()
    return fs, {"feed": feed, "vap": vap, "liq": liq}


def r_separator(h):
    return {"VF_feed": h["feed"].VaporFraction, "F_vap_mol_s": h["vap"].MolarFlowMolPerSecond,
            "y_CH4": h["vap"].OverallMoleFraction("Methane"), "x_C7": h["liq"].OverallMoleFraction("N-heptane")}


def c_separator(h, rep):
    vf = h["feed"].VaporFraction
    return [("Bifásico (0 < VF < 1)", 0.0 < vf < 1.0, f"VF={vf:.4f}"),
            ("Metano se concentra en el vapor", h["vap"].OverallMoleFraction("Methane") > 0.70, f"y_CH4={h['vap'].OverallMoleFraction('Methane'):.4f}"),
            ("Heptano se concentra en el líquido", h["liq"].OverallMoleFraction("N-heptane") > 0.10, f"x_C7={h['liq'].OverallMoleFraction('N-heptane'):.4f}"),
            ("Vapor sale como vapor y líquido como líquido", h["vap"].VaporFraction > 0.999 and h["liq"].VaporFraction < 1e-3, "")]


# --------------------------------------------------------------------------- 4 heat exchanger
def b_hx():
    fs = Flowsheet.Create("hx").WithCompound("Water").WithPropertyPackage(PropertyPackages.SteamTables)
    hot_in = fs.AddMaterialStream("hot-in").At(Q.Kelvin(450.0), Q.Bar(10.0)).WithMassFlow(Q.KgPerSecond(5.0))
    cold_in = fs.AddMaterialStream("cold-in").At(Q.Kelvin(298.0), Q.Bar(10.0)).WithMassFlow(Q.KgPerSecond(8.0))
    hot_out, cold_out = fs.AddMaterialStream("hot-out"), fs.AddMaterialStream("cold-out")
    from DWSIM.UnitOperations.UnitOperations import HeatExchangerCalcMode
    hx = (fs.AddHeatExchanger("E-1").WithCalculationMode(HeatExchangerCalcMode.PinchPoint).WithGlobalUA(2500.0)
          .WithHotSidePressureDrop(Q.Bar(0.2)).WithColdSidePressureDrop(Q.Bar(0.1))
          .ConnectFeed(hot_in, 0).ConnectProduct(hot_out, 0).ConnectFeed(cold_in, 1).ConnectProduct(cold_out, 1))
    fs.AutoLayout()
    return fs, {"hot_in": hot_in, "cold_in": cold_in, "hot_out": hot_out, "cold_out": cold_out, "hx": hx}


def r_hx(h):
    return {"T_hot_out_K": h["hot_out"].TemperatureK, "T_cold_out_K": h["cold_out"].TemperatureK,
            "Q_kW": float(_prop(h["hx"].Object, "Q", 0.0))}


def c_hx(h, rep):
    th, tc = h["hot_out"].TemperatureK, h["cold_out"].TemperatureK
    q = float(_prop(h["hx"].Object, "Q", 0.0))
    q_cold = 8.0 * 4.18 * (tc - 298.0)   # kW, cp agua líquida ≈ 4.18 kJ/kg K
    return [("Caliente se enfría, frío se calienta", th < 450.0 and tc > 298.0, f"T_hot_out={th:.1f} K, T_cold_out={tc:.1f} K"),
            ("Sin cruce: T_hot_out > T_cold_in y T_cold_out < T_hot_in", th > 298.0 and tc < 450.0, ""),
            ("Q ≈ m·cp·ΔT del lado frío (±5 %)", abs(q - q_cold) / max(q_cold, 1e-9) < 0.05, f"Q={q:.1f} kW vs {q_cold:.1f} kW")]


# --------------------------------------------------------------------------- 5 pump
def b_pump():
    fs = Flowsheet.Create("pump").WithCompound("Water").WithPropertyPackage(PropertyPackages.SteamTables)
    w = fs.AddMaterialStream("water").At(Q.Kelvin(298.15), Q.Bar(1.0)).WithMassFlow(Q.KgPerSecond(10.0))
    hp = fs.AddMaterialStream("hp")
    p = fs.AddPump("P-1").WithOutletPressure(Q.Bar(11.0)).WithEfficiencyPercent(70.0).ConnectFeed(w, 0).ConnectProduct(hp, 0)
    fs.AutoLayout()
    return fs, {"water": w, "hp": hp, "p": p}


def r_pump(h):
    return {"W_kW": h["p"].PowerKW, "T_out_K": h["hp"].TemperatureK}


def c_pump(h, rep):
    w_ideal = 10e5 * (10.0 / 997.0) / 0.7 / 1e3   # ΔP·Q/η en kW ≈ 14.33
    return [("W ≈ ΔP·Q/η (±3 %)", abs(h["p"].PowerKW - w_ideal) / w_ideal < 0.03, f"W={h['p'].PowerKW:.3f} kW vs {w_ideal:.3f} kW"),
            ("T sube menos de 1 K", 0.0 <= h["hp"].TemperatureK - 298.15 < 1.0, f"ΔT={h['hp'].TemperatureK - 298.15:.3f} K")]


# --------------------------------------------------------------------------- 6 polytropic compressor
def b_comp_poly():
    names = ["Methane", "Ethane", "Propane", "N-butane", "Carbon dioxide"]
    fs = _with(Flowsheet.Create("comp-poly"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    gas = fs.AddMaterialStream("gas").At(Q.Kelvin(300.0), Q.Bar(50.0)).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, [0.85, 0.07, 0.04, 0.02, 0.02])))
    hp = fs.AddMaterialStream("hp")
    from DWSIM.UnitOperations.UnitOperations import Compressor
    k = (fs.AddCompressor("K-1").WithOutletPressure(Q.Bar(60.0)).WithProcessPath(Compressor.ProcessPathType.Polytropic)
         .WithPolytropicEfficiencyPercent(75.0).ConnectFeed(gas, 0).ConnectProduct(hp, 0))
    fs.AutoLayout()
    return fs, {"gas": gas, "hp": hp, "k": k}


def r_comp_poly(h):
    o = h["k"].Object
    return {"W_kW": float(_prop(o, "DeltaQ", 0.0)), "T_out_K": h["hp"].TemperatureK,
            "eta_ad": float(_prop(o, "AdiabaticEfficiency", 0.0)) / 100.0,      # DWSIM stores percent
            "eta_poly": float(_prop(o, "PolytropicEfficiency", 0.0)) / 100.0}


def c_comp_poly(h, rep):
    o = h["k"].Object
    w = float(_prop(o, "DeltaQ", 0.0))
    eta_ad = float(_prop(o, "AdiabaticEfficiency", 0.0)) / 100.0
    # Ideal-gas relation between polytropic and adiabatic efficiency for r = 1.2, k ≈ 1.30:
    # η_ad = (r^((k-1)/k) − 1) / (r^((k-1)/(k·η_p)) − 1) ≈ 0.744
    r, k_ = 60.0 / 50.0, 1.30
    eta_ad_ideal = (r ** ((k_ - 1) / k_) - 1) / (r ** ((k_ - 1) / (k_ * 0.75)) - 1)
    return [("η_ad < η_poly = 0.75 (Schultz)", 0.60 < eta_ad < 0.75, f"η_ad={eta_ad:.4f}"),
            ("Potencia mayor que el caso adiabático η=0.75 (52.8 kW)", 52.8 < w < 60.0, f"W={w:.2f} kW"),
            ("η_ad ≈ relación gas ideal con k=1.30 (±0.01)", abs(eta_ad - eta_ad_ideal) < 0.01, f"η_ad={eta_ad:.4f} vs {eta_ad_ideal:.4f}")]


# --------------------------------------------------------------------------- 7 conversion reactor (CH4 combustion)
def b_conv_reactor():
    names = ["Methane", "Oxygen", "Nitrogen", "Carbon dioxide", "Water"]
    fs = _with(Flowsheet.Create("conv"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    r1 = fs.DefineConversionReaction("COMB", stoich({"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}), "Methane", "Vapor", "100")
    fs.ReactionSet("Comb").Add(r1)
    feed = (fs.AddMaterialStream("feed").WithTemperature(Q.Kelvin(500.0)).WithPressure(Q.Bar(1.01325)).WithMolarFlow(Q.MolPerSecond(10.52))
            .SetCompoundMolarFlow("Methane", 1.0).SetCompoundMolarFlow("Oxygen", 2.0).SetCompoundMolarFlow("Nitrogen", 7.52))
    gas, liq = fs.AddMaterialStream("flue"), fs.AddMaterialStream("liq")
    heat = fs.AddEnergyStream("Q")
    r = (fs.AddConversionReactor("R-1").Isothermal().WithReactionSet("Comb").WithPressureDrop(Q.Pascal(0.0))
         .ConnectFeed(feed, 0).ConnectProduct(gas, 0).ConnectProduct(liq, 1).ConnectEnergyFeed(heat, 1))
    fs.AutoLayout()
    return fs, {"feed": feed, "flue": gas, "liq": liq, "heat": heat, "r": r}


def r_conv_reactor(h):
    return {"Q_kW": h["r"].HeatDutyKW, "y_CO2": h["flue"].OverallMoleFraction("Carbon dioxide"), "y_CH4": h["flue"].OverallMoleFraction("Methane")}


def c_conv_reactor(h, rep):
    q = h["r"].HeatDutyKW
    return [("Isotérmico 500 K: Q ≈ −ΔH_c(500 K) ≈ −800 kJ/mol CH₄ (±2 %)", abs(abs(q) - 800.0) / 800.0 < 0.02, f"Q={q:.1f} kW por 1 mol/s"),
            ("Conversión 100 %: sin CH₄ en humos", h["flue"].OverallMoleFraction("Methane") < 1e-9, f"y_CH4={h['flue'].OverallMoleFraction('Methane'):.2e}"),
            ("y_CO2 = 1/10.52 = 0.0951", abs(h["flue"].OverallMoleFraction("Carbon dioxide") - 1 / 10.52) < 1e-3, f"y_CO2={h['flue'].OverallMoleFraction('Carbon dioxide'):.4f}")]


# --------------------------------------------------------------------------- 8 equilibrium reactor (WGS)
def b_eq_reactor():
    names = ["Carbon monoxide", "Water", "Carbon dioxide", "Hydrogen"]
    fs = _with(Flowsheet.Create("wgs"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    wgs = fs.DefineEquilibriumReaction("WGS", stoich({"Carbon monoxide": -1, "Water": -1, "Carbon dioxide": 1, "Hydrogen": 1}),
                                       "Carbon monoxide", "Vapor", "Activity", "", "4577.8/T - 4.33", 0.0)
    fs.ReactionSet("WGSset").Add(wgs)
    feed = (fs.AddMaterialStream("syngas").WithTemperature(Q.Kelvin(600.0)).WithPressure(Q.Bar(1.01325)).WithMolarFlow(Q.MolPerSecond(10.0))
            .SetCompoundMolarFlow("Carbon monoxide", 5.0).SetCompoundMolarFlow("Water", 5.0))
    out, liq = fs.AddMaterialStream("shifted"), fs.AddMaterialStream("liq")
    heat = fs.AddEnergyStream("Q")
    r = (fs.AddEquilibriumReactor("R-WGS").Isothermal().WithReactionSet("WGSset")
         .ConnectFeed(feed, 0).ConnectProduct(out, 0).ConnectProduct(liq, 1).ConnectEnergyFeed(heat, 1))
    fs.AutoLayout()
    return fs, {"feed": feed, "out": out, "liq": liq, "heat": heat, "r": r}


def r_eq_reactor(h):
    return {"y_H2": h["out"].OverallMoleFraction("Hydrogen"), "Q_kW": h["r"].HeatDutyKW}


def c_eq_reactor(h, rep):
    # ln K = 4577.8/T − 4.33 at 600 K → K ≈ 27.1; equimolar ideal gas: x/(1−x) = √K → x ≈ 0.839 → y_H2 ≈ 0.42
    k = math.exp(4577.8 / 600.0 - 4.33)
    x = math.sqrt(k) / (1 + math.sqrt(k))
    y = h["out"].OverallMoleFraction("Hydrogen")
    return [("y_H2 ≈ conversión de equilibrio analítica (±0.02)", abs(y - x / 2) < 0.02, f"y_H2={y:.4f} vs {x / 2:.4f} (K={k:.1f})"),
            ("WGS exotérmica: Q < 0 isotérmico", h["r"].HeatDutyKW < 0, f"Q={h['r'].HeatDutyKW:.1f} kW")]


# --------------------------------------------------------------------------- 9 rigorous column (ethanol-water, NRTL)
def b_column():
    fs = Flowsheet.Create("col").WithCompounds("Water", "Ethanol").WithPropertyPackage(PropertyPackages.NRTL)
    feed = (fs.AddMaterialStream("feed").WithTemperature(Q.Kelvin(300.0)).WithPressure(Q.Pascal(101325.0)).WithMolarFlow(Q.MolPerSecond(100.0))
            .SetCompoundMolarFlow("Water", 50.0).SetCompoundMolarFlow("Ethanol", 50.0))
    dist, bot = fs.AddMaterialStream("dist"), fs.AddMaterialStream("bot")
    cd, rd = fs.AddEnergyStream("cd"), fs.AddEnergyStream("rd")
    # WithFeed is 0-based (WithFeed(s, 2) lands on "Stage3"): index 9 = stage 10 of the statement.
    col = (fs.AddDistillationColumn("T-101").WithNumberOfStages(20).WithFeed(feed, 10 - 1).WithDistillate(dist).WithBottoms(bot)
           .WithCondenserDuty(cd).WithReboilerDuty(rd).WithCondenserSpec("Reflux Ratio", 2.0, "")
           .WithReboilerSpec("Product Molar Flow Rate", 75.0, "mol/s").WithTopPressure(Q.Pascal(101325.0)).WithColumnPressureDrop(Q.Pascal(0.0)))
    fs.AutoLayout()
    return fs, {"feed": feed, "dist": dist, "bot": bot, "cd": cd, "rd": rd, "col": col}


def r_column(h):
    return {"x_EtOH_dist": h["dist"].OverallMoleFraction("Ethanol"), "x_EtOH_bot": h["bot"].OverallMoleFraction("Ethanol"),
            "D_mol_s": h["dist"].MolarFlowMolPerSecond, "Qc_kW": h["cd"].EnergyFlowKW, "Qr_kW": h["rd"].EnergyFlowKW}


def c_column(h, rep):
    xd, xb = h["dist"].OverallMoleFraction("Ethanol"), h["bot"].OverallMoleFraction("Ethanol")
    return [("Destilado más rico que la alimentación y bajo el azeótropo (0.894)", 0.5 < xd < 0.894, f"x_D={xd:.4f}"),
            ("Fondo más pobre que la alimentación", xb < 0.5, f"x_B={xb:.4f}"),
            ("D = F − B = 25 mol/s", abs(h["dist"].MolarFlowMolPerSecond - 25.0) < 0.05, f"D={h['dist'].MolarFlowMolPerSecond:.3f}"),
            ("Balance de etanol D·x_D + B·x_B = 50", abs(25.0 * xd + 75.0 * xb - 50.0) < 0.1, f"{25 * xd + 75 * xb:.3f}")]


# --------------------------------------------------------------------------- 10 recycle loop (Wegstein)
def b_recycle():
    fs = Flowsheet.Create("recycle").WithCompound("Water").WithPropertyPackage(PropertyPackages.SteamTables)
    fresh = fs.AddMaterialStream("fresh").At(Q.Kelvin(300.0), Q.Atm(1.0)).WithMassFlow(Q.KgPerSecond(10.0))
    mixed, heated, purge = fs.AddMaterialStream("mixed"), fs.AddMaterialStream("heated"), fs.AddMaterialStream("purge")
    recycle, recycle_in = fs.AddMaterialStream("recycle"), fs.AddMaterialStream("recycle-in")
    fs.AddMixer("MIX-1").ConnectFeed(fresh, 0).ConnectFeed(recycle_in, 1).ConnectProduct(mixed)
    fs.AddHeater("H-1").WithOutletTemperature(Q.Kelvin(400.0)).ConnectFeed(mixed).ConnectProduct(heated)
    fs.AddSplitter("SPL-1").WithSplitRatios(0.2, 0.8).ConnectFeed(heated).ConnectProduct(purge, 0).ConnectProduct(recycle, 1)
    rec = fs.AddUnitOperation(ObjectType.OT_Recycle, "REC-1").ConnectFeed(recycle).ConnectProduct(recycle_in)
    # Default Wegstein tolerances (0.01 kg/s, 1e-4 relative) stop 0.04 % short of steady state;
    # tighten them so the analytic solution (purge = fresh) is reproduced.
    r = rec.Object.__implementation__
    cp = r.ConvergenceParameters
    cp.VazaoMassica, cp.VazaoMassicaRelativa = 1e-6, 1e-8
    cp.Temperatura, cp.Pressao, cp.Entalpia, cp.Composicao = 1e-4, 1e-3, 1e-4, 1e-7
    r.MaximumIterations = 500
    fs.AutoLayout()
    return fs, {"fresh": fresh, "purge": purge, "recycle_in": recycle_in, "mixed": mixed, "rec": rec}


def r_recycle(h):
    r = h["rec"].Object.__implementation__
    return {"recycle_kg_s": h["recycle_in"].MassFlowKgPerSecond, "purge_kg_s": h["purge"].MassFlowKgPerSecond,
            "iterations": float(r.IterationsTaken), "T_mixed_K": h["mixed"].TemperatureK}


def c_recycle(h, rep):
    r = h["rec"].Object.__implementation__
    return [("Convergió", bool(r.Converged), f"iter={r.IterationsTaken}"),
            ("Estado estacionario: purga = fresco = 10 kg/s (±1e-3)", abs(h["purge"].MassFlowKgPerSecond - 10.0) < 1e-3, f"purge={h['purge'].MassFlowKgPerSecond:.5f}"),
            ("Reciclo = 0.8/0.2 × fresco = 40 kg/s (±1e-2)", abs(h["recycle_in"].MassFlowKgPerSecond - 40.0) < 1e-2, f"rec={h['recycle_in'].MassFlowKgPerSecond:.4f}")]


CASES: list[Case] = [
    Case("mezclador", "Mezclador de dos corrientes de gas", "Two-stream gas mixer",
         "Mezcla 100 mol/s de gas (90/8/2 % C1/C2/C3) a 280 K y 20 bar con 50 mol/s (50/30/20 %) a 340 K y 22 bar. Reporta T, P y composición de salida.",
         "Mix 100 mol/s of gas (90/8/2 % C1/C2/C3) at 280 K, 20 bar with 50 mol/s (50/30/20 %) at 340 K, 22 bar. Report outlet T, P and composition.",
         b_mixer, r_mixer, c_mixer, ["basico", "balance"]),
    Case("valvula_jt", "Válvula Joule-Thomson", "Joule-Thomson valve",
         "Gas natural 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂, 100 mol/s, a 300 K y 50 bar se expande a 10 bar en una válvula (Peng-Robinson). T de salida y ΔT = T_out − 300 K.",
         "Natural gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂, 100 mol/s, at 300 K and 50 bar expands to 10 bar through a valve (Peng-Robinson). Outlet T and ΔT = T_out − 300 K.",
         b_valve, r_valve, c_valve, ["basico", "termo"]),
    Case("separador", "Separador bifásico de gas húmedo", "Two-phase wet-gas separator",
         "100 mol/s de C1/C2/nC5/nC7 (70/10/10/10 %) a 290 K y 20 bar entran a un separador. Flujos y composiciones de vapor y líquido.",
         "100 mol/s of C1/C2/nC5/nC7 (70/10/10/10 %) at 290 K, 20 bar enter a separator. Vapor and liquid flows and compositions.",
         b_separator, r_separator, c_separator, ["basico", "flash"]),
    Case("intercambiador", "Intercambiador agua-agua (UA)", "Water-water heat exchanger (UA)",
         "5 kg/s de agua a 450 K y 10 bar calientan 8 kg/s a 298 K; UA = 2500 W/K, ΔP 0.2/0.1 bar. Temperaturas de salida y carga.",
         "5 kg/s of water at 450 K, 10 bar heat 8 kg/s at 298 K; UA = 2500 W/K, ΔP 0.2/0.1 bar. Outlet temperatures and duty.",
         b_hx, r_hx, c_hx, ["equipos", "calor"]),
    Case("bomba", "Bomba de agua", "Water pump",
         "10 kg/s de agua a 25 °C de 1 a 11 bar con η = 70 %. Potencia al eje.",
         "10 kg/s of water at 25 °C from 1 to 11 bar, η = 70 %. Shaft power.",
         b_pump, r_pump, c_pump, ["equipos"]),
    Case("compresor_politropico", "Compresor politrópico", "Polytropic compressor",
         "Gas natural 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂, 100 mol/s, 300 K, de 50 a 60 bar con η politrópica 75 % (Peng-Robinson). Potencia, T de salida, y η adiabática y politrópica como fracciones.",
         "Natural gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂, 100 mol/s, 300 K, 50 to 60 bar with 75 % polytropic efficiency (Peng-Robinson). Power, outlet T, and adiabatic and polytropic efficiencies as fractions.",
         b_comp_poly, r_comp_poly, c_comp_poly, ["equipos", "rotatorios"]),
    Case("reactor_conversion", "Combustión de metano (reactor de conversión)", "Methane combustion (conversion reactor)",
         "1 mol/s de CH₄ con aire estequiométrico (2 O₂ + 7.52 N₂) a 500 K y 1 atm, conversión 100 %, isotérmico (Peng-Robinson). Carga térmica del reactor Q_kW con signo del simulador (negativo = calor retirado) y composición de humos.",
         "1 mol/s CH₄ with stoichiometric air (2 O₂ + 7.52 N₂) at 500 K, 1 atm, 100 % conversion, isothermal (Peng-Robinson). Reactor duty Q_kW with the simulator's sign (negative = heat removed) and flue-gas composition.",
         b_conv_reactor, r_conv_reactor, c_conv_reactor, ["reactores"]),
    Case("reactor_equilibrio", "Water-gas shift en equilibrio", "Water-gas shift at equilibrium",
         "CO + H₂O equimolar, 10 mol/s a 600 K y 1 atm, ln K = 4577.8/T − 4.33, isotérmico (Peng-Robinson). Fracción molar de H₂ y carga térmica Q_kW (negativo = calor retirado).",
         "Equimolar CO + H₂O, 10 mol/s at 600 K, 1 atm, ln K = 4577.8/T − 4.33, isothermal (Peng-Robinson). H₂ mole fraction and duty Q_kW (negative = heat removed).",
         b_eq_reactor, r_eq_reactor, c_eq_reactor, ["reactores", "equilibrio"]),
    Case("columna_rigurosa", "Columna etanol-agua (NRTL)", "Ethanol-water column (NRTL)",
         "100 mol/s etanol-agua 50/50 a 300 K y 1 atm, 20 etapas, alimentación en la 10, R = 2, fondo 75 mol/s. Purezas y cargas.",
         "100 mol/s ethanol-water 50/50 at 300 K, 1 atm, 20 stages, feed at 10, R = 2, bottoms 75 mol/s. Purities and duties.",
         b_column, r_column, c_column, ["columnas", "nrtl"]),
    Case("reciclo", "Lazo de reciclo con Wegstein", "Recycle loop (Wegstein)",
         "10 kg/s de agua fresca a 300 K y 1 atm se mezclan con el reciclo, se calientan a 400 K y un divisor purga 20 % y recicla 80 %. Flujos de reciclo y purga en estado estacionario y temperatura de la corriente mezclada (salida del mezclador, antes del calentador).",
         "10 kg/s fresh water at 300 K and 1 atm mixes with the recycle, is heated to 400 K and a splitter purges 20 % and recycles 80 %. Steady-state recycle and purge flows and the temperature of the mixed stream (mixer outlet, before the heater).",
         b_recycle, r_recycle, c_recycle, ["flowsheet", "reciclo"]),
]

BY_ID = {c.id: c for c in CASES}
