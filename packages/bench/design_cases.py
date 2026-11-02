"""Seven *design* cases: the answer is a decision (a pressure, a number of stages, a motor size),
not a simulation readout. Each has an `oracle()` that finds the answer deterministically with
DWSIM (bisection or enumeration) plus analytic sanity checks, so the bench can grade any agent.
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
from packages.bench.cases_ref import stoich, _with, _prop  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # noqa: E402


@dataclass
class DesignCase:
    id: str
    title_es: str
    title_en: str
    statement_es: str
    statement_en: str
    oracle: Callable[[], dict]            # -> {quantity: value}
    tolerances: dict                      # quantity -> {"rel": x} | {"abs": x}
    units: dict                           # quantity -> unit string
    checks: Callable[[dict], list]        # analytic sanity on the oracle's own answer
    tags: list[str] = field(default_factory=list)


def bisect(f, lo, hi, target, tol=1e-4, maxit=60):
    """Find x in [lo, hi] with f(x) = target (f monotone)."""
    flo, fhi = f(lo) - target, f(hi) - target
    if flo * fhi > 0:
        raise ValueError(f"no sign change: f({lo})={flo + target}, f({hi})={fhi + target}")
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - target
        if abs(fm) <= tol or (hi - lo) / max(abs(mid), 1e-12) < 1e-7:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- 1 separator temperature for VF = 0.5
GAS7 = (["Methane", "Ethane", "Propane", "N-butane", "N-pentane", "Nitrogen", "Carbon dioxide"],
        [0.80, 0.08, 0.04, 0.02, 0.02, 0.02, 0.02])


def _vf_at_T(t_k, p_bar=30.0):
    names, z = GAS7
    fs = _with(Flowsheet.Create("vf"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    s = fs.AddMaterialStream("s").At(Q.Kelvin(t_k), Q.Bar(p_bar)).WithMolarFlow(Q.MolPerSecond(1.0)).WithComposition(composition(zip(names, z)))
    fs.Solve()
    return s.VaporFraction


def o_flash_temperatura():
    # At 30 bar the mixture is two-phase between its bubble (~180 K) and dew (~291 K) points;
    # VF rises monotonically with T, so bisection on T is well posed. (Pressure is not: at 230 K
    # VF stays above 0.75 up to 60 bar — retrograde region.)
    t = bisect(_vf_at_T, 185.0, 290.0, 0.5, tol=1e-5)
    return {"T_K": t, "VF_at_T": _vf_at_T(t)}


def c_flash_temperatura(a):
    return [("T entre burbuja (180 K) y 230 K (VF 0.876)", 180.0 < a["T_K"] < 230.0, f"T={a['T_K']:.2f} K"),
            ("VF(T) = 0.5 ± 1e-3", abs(a["VF_at_T"] - 0.5) < 1e-3, f"VF={a['VF_at_T']:.5f}")]


# --------------------------------------------------------------------------- 2 compression stages with T_max
def _stage_T_and_W(n_stages, p_in=1.0, p_out=50.0, t_in=300.0, eta=75.0):
    ratio = (p_out / p_in) ** (1.0 / n_stages)
    fs = Flowsheet.Create("stage").WithCompound("Methane").WithPropertyPackage(PropertyPackages.PengRobinson)
    inlet = fs.AddMaterialStream("in").At(Q.Kelvin(t_in), Q.Bar(p_in)).WithMolarFlow(Q.MolPerSecond(10.0)).WithComposition(composition([("Methane", 1.0)]))
    out = fs.AddMaterialStream("out")
    k = fs.AddCompressor("K").WithOutletPressure(Q.Bar(p_in * ratio)).WithAdiabaticEfficiencyPercent(eta).ConnectFeed(inlet, 0).ConnectProduct(out, 0)
    fs.Solve()
    # Every stage starts at t_in (intercooled) and sees the same ratio; stage W differs slightly
    # with pressure (real gas) — report first-stage T and n × first-stage W as the estimate.
    return out.TemperatureK, float(_prop(k.Object, "DeltaQ", 0.0)) * n_stages, ratio


def o_compresor_etapas():
    for n in range(1, 9):
        t, w, ratio = _stage_T_and_W(n)
        if t <= 420.0:
            return {"N_stages": float(n), "T_stage_out_K": t, "ratio_per_stage": ratio, "W_total_kW": w}
    raise RuntimeError("no stage count met T_max")


def c_compresor_etapas(a):
    n = a["N_stages"]
    t_prev, _, _ = _stage_T_and_W(int(n) - 1) if n > 1 else (1e9, 0, 0)
    return [("Con una etapa menos se excede 420 K", t_prev > 420.0, f"T(N-1)={t_prev:.1f} K"),
            ("T de etapa ≤ 420 K", a["T_stage_out_K"] <= 420.0, f"T={a['T_stage_out_K']:.1f} K"),
            ("Relación por etapa = 50^(1/N)", abs(a["ratio_per_stage"] - 50 ** (1 / n)) < 1e-9, "")]


# --------------------------------------------------------------------------- 3 FUG stages at R = 1.5 Rmin
def _fug(reflux):
    names, z = ["Ethane", "Propane", "N-butane", "N-pentane"], [0.05, 0.45, 0.40, 0.10]
    fs = _with(Flowsheet.Create("fug"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    al = fs.AddMaterialStream("alim").WithPressure(Q.Bar(17.0)).WithVaporFraction(0.0).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, z)))
    d, b = fs.AddMaterialStream("d"), fs.AddMaterialStream("b")
    qc, qr = fs.AddEnergyStream("qc"), fs.AddEnergyStream("qr")
    sc = (fs.AddShortcutColumn("T").WithKeys("Propane", "N-butane").WithLightKeyInBottoms(0.02).WithHeavyKeyInDistillate(0.02)
          .WithPressures(Q.Bar(16.0), Q.Bar(17.5)).WithRefluxRatio(reflux).ConnectFeed(al, 0).ConnectProduct(d, 0).ConnectProduct(b, 1))
    sc.ConnectEnergyProduct(qc, 0)
    sc.ConnectEnergyFeed(qr, 1)
    fs.Solve()
    return sc, d, qc, qr


def o_columna_etapas():
    sc, *_ = _fug(2.0)
    rmin = sc.MinimumRefluxRatio
    sc2, d, qc, qr = _fug(1.5 * rmin)
    return {"R_min": rmin, "R": 1.5 * rmin, "N_min": sc2.MinimumStages, "N_theoretical": sc2.TheoreticalStages,
            "feed_stage": sc2.OptimumFeedStage, "D_mol_s": d.MolarFlowMolPerSecond, "Qr_kW": qr.EnergyFlowKW}


def c_columna_etapas(a):
    return [("N > N_min", a["N_theoretical"] > a["N_min"], f"N={a['N_theoretical']:.2f}, Nmin={a['N_min']:.2f}"),
            ("R = 1.5 R_min", abs(a["R"] - 1.5 * a["R_min"]) < 1e-9, ""),
            ("Plato de alimentación dentro de la columna", 1 < a["feed_stage"] < a["N_theoretical"], f"feed={a['feed_stage']:.2f}"),
            ("D ≈ 50 mol/s (balance de claves)", abs(a["D_mol_s"] - 50.0) < 0.5, f"D={a['D_mol_s']:.2f}")]


# --------------------------------------------------------------------------- 4 exchanger UA for a target outlet
def _cold_out_T(ua):
    fs = Flowsheet.Create("hx").WithCompound("Water").WithPropertyPackage(PropertyPackages.SteamTables)
    hi = fs.AddMaterialStream("hi").At(Q.Kelvin(450.0), Q.Bar(10.0)).WithMassFlow(Q.KgPerSecond(5.0))
    ci = fs.AddMaterialStream("ci").At(Q.Kelvin(298.0), Q.Bar(10.0)).WithMassFlow(Q.KgPerSecond(8.0))
    ho, co = fs.AddMaterialStream("ho"), fs.AddMaterialStream("co")
    from DWSIM.UnitOperations.UnitOperations import HeatExchangerCalcMode
    hx = (fs.AddHeatExchanger("E").WithCalculationMode(HeatExchangerCalcMode.PinchPoint).WithGlobalUA(ua)
          .WithHotSidePressureDrop(Q.Bar(0.0)).WithColdSidePressureDrop(Q.Bar(0.0))
          .ConnectFeed(hi, 0).ConnectProduct(ho, 0).ConnectFeed(ci, 1).ConnectProduct(co, 1))
    fs.Solve()
    return co.TemperatureK, float(_prop(hx.Object, "Q", 0.0)), ho.TemperatureK


def o_intercambiador_ua():
    ua = bisect(lambda u: _cold_out_T(u)[0], 500.0, 30000.0, 330.0, tol=1e-3)
    tc, q, th = _cold_out_T(ua)
    return {"UA_W_K": ua, "Q_kW": q, "T_hot_out_K": th}


def c_intercambiador_ua(a):
    q_cold = 8.0 * 4.18 * (330.0 - 298.0)
    # LMTD check: Q = UA·LMTD with the oracle's outlet temperatures
    dt1, dt2 = 450.0 - 330.0, a["T_hot_out_K"] - 298.0
    lmtd = (dt1 - dt2) / math.log(dt1 / dt2) if abs(dt1 - dt2) > 1e-9 else dt1
    return [("Q ≈ m·cp·ΔT frío (±5 %)", abs(a["Q_kW"] - q_cold) / q_cold < 0.05, f"Q={a['Q_kW']:.1f} vs {q_cold:.1f} kW"),
            ("Q ≈ UA·LMTD (±5 %)", abs(a["Q_kW"] * 1e3 - a["UA_W_K"] * lmtd) / (a["Q_kW"] * 1e3) < 0.05, f"UA·LMTD={a['UA_W_K'] * lmtd / 1e3:.1f} kW")]


# --------------------------------------------------------------------------- 5 pump motor size
IEC_MOTORS_KW = [5.5, 7.5, 11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0]


def o_bomba_motor():
    fs = Flowsheet.Create("pump").WithCompound("Water").WithPropertyPackage(PropertyPackages.SteamTables)
    w = fs.AddMaterialStream("w").At(Q.Kelvin(298.15), Q.Bar(1.0)).WithMassFlow(Q.KgPerSecond(10.0))
    hp = fs.AddMaterialStream("hp")
    p = fs.AddPump("P").WithOutletPressure(Q.Bar(11.0)).WithEfficiencyPercent(70.0).ConnectFeed(w, 0).ConnectProduct(hp, 0)
    fs.Solve()
    shaft = p.PowerKW
    required = shaft * 1.15
    motor = next(m for m in IEC_MOTORS_KW if m >= required)
    return {"W_shaft_kW": shaft, "W_required_kW": required, "motor_kW": motor}


def c_bomba_motor(a):
    return [("Motor ≥ 1.15 × eje", a["motor_kW"] >= a["W_required_kW"], f"{a['motor_kW']} ≥ {a['W_required_kW']:.2f}"),
            ("Motor es el menor estándar que cumple", all(m < a["W_required_kW"] for m in IEC_MOTORS_KW if m < a["motor_kW"]), ""),
            ("Eje ≈ ΔP·Q/η = 14.33 kW (±3 %)", abs(a["W_shaft_kW"] - 14.33) / 14.33 < 0.03, f"W={a['W_shaft_kW']:.3f}")]


# --------------------------------------------------------------------------- 6 excess air for a target adiabatic flame T
def _flame_T(air_mult):
    names = ["Methane", "Oxygen", "Nitrogen", "Carbon dioxide", "Water"]
    fs = _with(Flowsheet.Create("flame"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    r1 = fs.DefineConversionReaction("COMB", stoich({"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}), "Methane", "Vapor", "100")
    fs.ReactionSet("Comb").Add(r1)
    o2, n2 = 2.0 * air_mult, 7.52 * air_mult
    feed = (fs.AddMaterialStream("feed").WithTemperature(Q.Kelvin(298.15)).WithPressure(Q.Bar(1.01325)).WithMolarFlow(Q.MolPerSecond(1.0 + o2 + n2))
            .SetCompoundMolarFlow("Methane", 1.0).SetCompoundMolarFlow("Oxygen", o2).SetCompoundMolarFlow("Nitrogen", n2))
    gas, liq = fs.AddMaterialStream("flue"), fs.AddMaterialStream("liq")
    (fs.AddConversionReactor("R").Adiabatic().WithReactionSet("Comb").WithPressureDrop(Q.Pascal(0.0))
       .ConnectFeed(feed, 0).ConnectProduct(gas, 0).ConnectProduct(liq, 1))
    fs.Solve()
    return gas.TemperatureK


def o_aire_exceso():
    t_stoich = _flame_T(1.0)
    lam = bisect(_flame_T, 1.0, 4.0, 1800.0, tol=0.05)
    return {"excess_air_pct": (lam - 1.0) * 100.0, "lambda": lam, "T_adiabatic_stoich_K": t_stoich, "T_at_lambda_K": _flame_T(lam)}


def c_aire_exceso(a):
    return [("T adiabática estequiométrica entre 2000 y 2400 K (sin disociación)", 2000.0 < a["T_adiabatic_stoich_K"] < 2400.0, f"T={a['T_adiabatic_stoich_K']:.0f} K"),
            ("T(λ) = 1800 ± 1 K", abs(a["T_at_lambda_K"] - 1800.0) < 1.0, f"T={a['T_at_lambda_K']:.1f} K"),
            ("Exceso de aire entre 20 y 60 %", 20.0 < a["excess_air_pct"] < 60.0, f"{a['excess_air_pct']:.1f} %")]


# --------------------------------------------------------------------------- 7 purge fraction for a target recycle
def _recycle_flow(purge_frac):
    fs = Flowsheet.Create("rec").WithCompound("Water").WithPropertyPackage(PropertyPackages.SteamTables)
    fresh = fs.AddMaterialStream("fresh").At(Q.Kelvin(300.0), Q.Atm(1.0)).WithMassFlow(Q.KgPerSecond(10.0))
    mixed, heated, purge = fs.AddMaterialStream("mixed"), fs.AddMaterialStream("heated"), fs.AddMaterialStream("purge")
    recycle, recycle_in = fs.AddMaterialStream("recycle"), fs.AddMaterialStream("recycle-in")
    fs.AddMixer("MIX").ConnectFeed(fresh, 0).ConnectFeed(recycle_in, 1).ConnectProduct(mixed)
    fs.AddHeater("H").WithOutletTemperature(Q.Kelvin(400.0)).ConnectFeed(mixed).ConnectProduct(heated)
    fs.AddSplitter("SPL").WithSplitRatios(purge_frac, 1.0 - purge_frac).ConnectFeed(heated).ConnectProduct(purge, 0).ConnectProduct(recycle, 1)
    rec = fs.AddUnitOperation(ObjectType.OT_Recycle, "REC").ConnectFeed(recycle).ConnectProduct(recycle_in)
    r = rec.Object.__implementation__
    cp = r.ConvergenceParameters
    cp.VazaoMassica, cp.VazaoMassicaRelativa, cp.Temperatura, cp.Entalpia, cp.Composicao = 1e-6, 1e-8, 1e-4, 1e-4, 1e-7
    r.MaximumIterations = 500
    fs.Solve()
    return recycle_in.MassFlowKgPerSecond


def o_reciclo_purga():
    p_analytic = 10.0 / (10.0 + 25.0)        # recycle = fresh·(1−p)/p = 25 → p = 10/35
    return {"purge_fraction": p_analytic, "recycle_kg_s_dwsim": _recycle_flow(p_analytic), "purge_kg_s": 10.0}


def c_reciclo_purga(a):
    return [("DWSIM reproduce 25 kg/s con p = 10/35 (±0.01)", abs(a["recycle_kg_s_dwsim"] - 25.0) < 0.01, f"rec={a['recycle_kg_s_dwsim']:.4f}"),
            ("Purga = fresco en estado estacionario", abs(a["purge_kg_s"] - 10.0) < 1e-9, "")]


DESIGN_CASES: list[DesignCase] = [
    DesignCase("diseno_flash_temperatura", "Temperatura de separador para VF = 0.5", "Separator temperature for VF = 0.5",
               "Gas natural (80/8/4/2/2 % C1/C2/C3/nC4/nC5, 2 % N₂, 2 % CO₂) a 30 bar. ¿A qué temperatura (K) un separador flash deja exactamente la mitad molar en vapor? Peng-Robinson.",
               "Natural gas (80/8/4/2/2 % C1/C2/C3/nC4/nC5, 2 % N₂, 2 % CO₂) at 30 bar. At what temperature (K) does a flash separator leave exactly half the moles as vapor? Peng-Robinson.",
               o_flash_temperatura, {"T_K": {"abs": 1.0}, "VF_at_T": {"abs": 0.01}}, {"T_K": "K", "VF_at_T": "-"}, c_flash_temperatura, ["diseno", "flash"]),
    DesignCase("diseno_compresor_etapas", "Número de etapas de compresión", "Number of compression stages",
               "Metano, 10 mol/s, a 300 K de 1 a 50 bar, η adiabática 75 %, enfriamiento entre etapas a 300 K, relación igual por etapa. Mínimo número de etapas para que ninguna salga arriba de 420 K, T de salida de etapa y potencia total aproximada (N × potencia de la primera etapa).",
               "Methane, 10 mol/s, at 300 K from 1 to 50 bar, 75 % adiabatic efficiency, intercooling to 300 K, equal ratio per stage. Minimum stages so no stage exceeds 420 K, stage outlet T and approximate total power (N × first-stage power).",
               o_compresor_etapas, {"N_stages": {"abs": 0.0}, "T_stage_out_K": {"abs": 2.0}, "W_total_kW": {"rel": 0.05}}, {"N_stages": "-", "T_stage_out_K": "K", "W_total_kW": "kW"}, c_compresor_etapas, ["diseno", "rotatorios"]),
    DesignCase("diseno_columna_etapas", "Etapas de una depropanizadora a 1.5 R_min", "Depropanizer stages at 1.5 R_min",
               "Alimentación líquida saturada 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) a 17 bar; 2 % de nC4 en destilado y 2 % de C3 en fondo; condensador 16 bar, rehervidor 17.5 bar. Con R = 1.5 R_min: R_min, N_min, N teórico y plato de alimentación (FUG).",
               "Saturated-liquid feed 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) at 17 bar; 2 % nC4 in distillate and 2 % C3 in bottoms; condenser 16 bar, reboiler 17.5 bar. At R = 1.5 R_min: R_min, N_min, theoretical N and feed stage (FUG).",
               o_columna_etapas, {"R_min": {"rel": 0.05}, "N_min": {"rel": 0.05}, "N_theoretical": {"rel": 0.08}, "feed_stage": {"abs": 1.5}, "D_mol_s": {"rel": 0.01}}, {"R_min": "-", "N_min": "-", "N_theoretical": "-", "feed_stage": "-", "D_mol_s": "mol/s"}, c_columna_etapas, ["diseno", "columnas"]),
    DesignCase("diseno_intercambiador_ua", "UA para calentar agua a 330 K", "UA to heat water to 330 K",
               "5 kg/s de agua a 450 K y 10 bar deben calentar 8 kg/s de 298 a 330 K (contracorriente, sin ΔP). ¿Qué UA (W/K) se requiere y cuál es la carga?",
               "5 kg/s of water at 450 K, 10 bar must heat 8 kg/s from 298 to 330 K (counter-current, no ΔP). What UA (W/K) is required and what is the duty?",
               o_intercambiador_ua, {"UA_W_K": {"rel": 0.03}, "Q_kW": {"rel": 0.02}}, {"UA_W_K": "W/K", "Q_kW": "kW"}, c_intercambiador_ua, ["diseno", "calor"]),
    DesignCase("diseno_bomba_motor", "Motor de una bomba de agua", "Motor for a water pump",
               "10 kg/s de agua a 25 °C de 1 a 11 bar, η 70 %. Potencia al eje y motor IEC estándar (5.5, 7.5, 11, 15, 18.5, 22, 30 kW) con 15 % de margen.",
               "10 kg/s of water at 25 °C from 1 to 11 bar, η 70 %. Shaft power and standard IEC motor (5.5, 7.5, 11, 15, 18.5, 22, 30 kW) with 15 % margin.",
               o_bomba_motor, {"W_shaft_kW": {"rel": 0.02}, "motor_kW": {"abs": 0.0}}, {"W_shaft_kW": "kW", "motor_kW": "kW"}, c_bomba_motor, ["diseno", "equipos"]),
    DesignCase("diseno_aire_exceso", "Exceso de aire para 1800 K de flama", "Excess air for an 1800 K flame",
               "CH₄ y aire a 298.15 K y 1 atm, combustión completa adiabática sin disociación (Peng-Robinson). ¿Qué % de exceso de aire da 1800 K en los humos?",
               "CH₄ and air at 298.15 K, 1 atm, complete adiabatic combustion without dissociation (Peng-Robinson). What % excess air gives 1800 K flue gas?",
               o_aire_exceso, {"excess_air_pct": {"abs": 2.0}, "T_adiabatic_stoich_K": {"abs": 25.0}}, {"excess_air_pct": "%", "T_adiabatic_stoich_K": "K"}, c_aire_exceso, ["diseno", "reactores"]),
    DesignCase("diseno_reciclo_purga", "Fracción de purga para un reciclo de 25 kg/s", "Purge fraction for a 25 kg/s recycle",
               "10 kg/s de agua fresca se mezclan con el reciclo, se calientan a 400 K y un divisor purga una fracción p. ¿Qué p deja un reciclo de 25 kg/s en estado estacionario?",
               "10 kg/s fresh water mixes with the recycle, is heated to 400 K and a splitter purges a fraction p. Which p gives a 25 kg/s recycle at steady state?",
               o_reciclo_purga, {"purge_fraction": {"abs": 0.003}, "purge_kg_s": {"abs": 0.05}}, {"purge_fraction": "-", "purge_kg_s": "kg/s"}, c_reciclo_purga, ["diseno", "flowsheet"]),
]

DESIGN_BY_ID = {c.id: c for c in DESIGN_CASES}
