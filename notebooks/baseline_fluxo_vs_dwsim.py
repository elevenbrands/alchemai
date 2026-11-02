"""Baseline: FluxoEngine (Swift, apps/fluxo) vs DWSIM 10 on three reference cases.

Fluxo numbers come from `swift test --filter LineaBaseDWSIM` (Tests/FluxoEngineTests/
LineaBaseDWSIMTests.swift), captured in notebooks/fluxo_baseline_raw.jsonl.
DWSIM runs here through the Fluent API (packages/dwsim_client/fluent.py). Output: a Markdown
table in notebooks/baseline_fluxo_vs_dwsim.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client.fluent import fluent  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
import System  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402

fluxo = {json.loads(l)["caso"]: json.loads(l) for l in (ROOT / "notebooks/fluxo_baseline_raw.jsonl").read_text().splitlines()}
rows = []  # (case, variable, fluxo, dwsim, unit)


def comp(pairs):
    def _set(cb):
        for name, z in pairs:
            cb.Mole(name, z)
    return System.Action[CompositionBuilder](_set)


def add(case, var, f, d, unit=""):
    rows.append((case, var, f, d, unit))


# ---------------------------------------------------------------- Caso 1: flash gas natural
names = ["Methane", "Ethane", "Propane", "N-butane", "N-pentane", "Nitrogen", "Carbon dioxide"]
z = [0.80, 0.08, 0.04, 0.02, 0.02, 0.02, 0.02]
fs = Flowsheet.Create("flash")
for n in names:
    fs = fs.WithCompound(n)
fs = fs.WithPropertyPackage(PropertyPackages.PengRobinson)
feed = fs.AddMaterialStream("feed").At(Q.Kelvin(230.0), Q.Bar(30.0)).WithMolarFlow(Q.MolPerSecond(1.0)).WithComposition(comp(zip(names, z)))
vap, liq = fs.AddMaterialStream("vap"), fs.AddMaterialStream("liq")
fs.AddSeparator("V-1").ConnectFeed(feed, 0).ConnectProduct(vap, 0).ConnectProduct(liq, 1)
fs.AutoLayout(); fs.Solve()
f1 = fluxo["flash"]
add("flash", "VF", f1["vf"], feed.VaporFraction)
for i, n in enumerate(names):
    add("flash", f"y {n}", f1["y"][i], vap.OverallMoleFraction(n))
for i, n in enumerate(names):
    add("flash", f"x {n}", f1["x"][i], liq.OverallMoleFraction(n))

# ---------------------------------------------------------------- Caso 2: compresor + enfriador
names2 = ["Methane", "Ethane", "Propane", "N-butane", "Carbon dioxide"]
z2 = [0.85, 0.07, 0.04, 0.02, 0.02]
fs2 = Flowsheet.Create("comp")
for n in names2:
    fs2 = fs2.WithCompound(n)
fs2 = fs2.WithPropertyPackage(PropertyPackages.PengRobinson)
gas = fs2.AddMaterialStream("gas").At(Q.Kelvin(300.0), Q.Bar(50.0)).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(comp(zip(names2, z2)))
hp, cold = fs2.AddMaterialStream("hp"), fs2.AddMaterialStream("cold")
k = fs2.AddCompressor("K-1").WithOutletPressure(Q.Bar(60.0)).WithAdiabaticEfficiencyPercent(75.0).ConnectFeed(gas, 0).ConnectProduct(hp, 0)
e = fs2.AddCooler("E-1").WithOutletTemperature(Q.Kelvin(300.0)).ConnectFeed(hp, 0).ConnectProduct(cold, 0)
fs2.AutoLayout(); fs2.Solve()
f2 = fluxo["compresor"]
add("compresor", "T salida compresor", f2["tOutK"], hp.TemperatureK, "K")
add("compresor", "Potencia", f2["potenciaW"] / 1e3, k.Object.DeltaQ, "kW")  # DeltaQ: kW
add("compresor", "Q enfriador (retirado)", -f2["qEnfriadorW"] / 1e3, e.HeatRemovedKW, "kW")

# ---------------------------------------------------------------- Caso 3: depropanizadora FUG
names3 = ["Ethane", "Propane", "N-butane", "N-pentane"]
z3 = [0.05, 0.45, 0.40, 0.10]
f3 = fluxo["fug"]
fs3 = Flowsheet.Create("fug")
for n in names3:
    fs3 = fs3.WithCompound(n)
fs3 = fs3.WithPropertyPackage(PropertyPackages.PengRobinson)
al = fs3.AddMaterialStream("alim").WithPressure(Q.Bar(17.0)).WithVaporFraction(0.0).WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(comp(zip(names3, z3)))
dist, bot = fs3.AddMaterialStream("dest"), fs3.AddMaterialStream("fondo")
sc = (fs3.AddShortcutColumn("T-1").WithKeys("Propane", "N-butane")
      .WithLightKeyInBottoms(0.02).WithHeavyKeyInDistillate(0.02)
      .WithPressures(Q.Bar(16.0), Q.Bar(17.5)).WithRefluxRatio(f3["reflujo"])
      .ConnectFeed(al, 0).ConnectProduct(dist, 0).ConnectProduct(bot, 1))
qc, qr = fs3.AddEnergyStream("Qc"), fs3.AddEnergyStream("Qr")
sc.ConnectEnergyProduct(qc, 0)   # condenser duty leaves the column
sc.ConnectEnergyFeed(qr, 1)      # reboiler duty enters the column (energy feed port 1)
fs3.AutoLayout()
fs3.Solve()
add("fug", "T alimentación (burbuja)", f3["tAlimK"], al.TemperatureK, "K")
add("fug", "N mínimo", f3["nMin"], sc.MinimumStages)
add("fug", "R mínimo", f3["rMin"], sc.MinimumRefluxRatio)
add("fug", "N (R = 1.3 Rmin de Fluxo)", f3["n"], sc.TheoreticalStages)
add("fug", "Plato de alimentación", f3["platoAlim"], sc.OptimumFeedStage)
add("fug", "Destilado", f3["destiladoMolS"], dist.MolarFlowMolPerSecond, "mol/s")
add("fug", "Q condensador", f3["qCondW"] / 1e3, -abs(qc.EnergyFlowKW), "kW")
add("fug", "Q rehervidor", f3["qRehW"] / 1e3, qr.EnergyFlowKW, "kW")
add("fug", "T destilado", f3["tDestK"], dist.TemperatureK, "K")
add("fug", "T fondo", f3["tFondoK"], bot.TemperatureK, "K")

# ---------------------------------------------------------------- tabla
lines = ["| Caso | Variable | Fluxo | DWSIM 10 | Desv. |", "|---|---|---|---|---|"]
for case, var, f, d, unit in rows:
    try:
        f, d = float(f), float(d)
        dev = (d - f) / f * 100 if abs(f) > 1e-9 else float("nan")
        lines.append(f"| {case} | {var} | {f:.4g} {unit} | {d:.4g} {unit} | {dev:+.2f} % |")
    except Exception as ex:  # noqa: BLE001
        lines.append(f"| {case} | {var} | {f} | {d} | error: {ex} |")
md = "\n".join(lines)
print(md)
(ROOT / "notebooks/baseline_fluxo_vs_dwsim.md").write_text("# Línea base Fluxo vs DWSIM 10.2.12 (PR)\n\n" + md + "\n")
