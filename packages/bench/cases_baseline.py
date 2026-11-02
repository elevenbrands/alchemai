"""The three baseline flowsheets (Fluxo vs DWSIM) as reusable builders.

Each function returns (flowsheet, handles) where `handles` is a dict of the builders you may
want to read back. Used by notebooks/baseline_fluxo_vs_dwsim.py and tests/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client.fluent import fluent  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
import System  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402


def composition(pairs):
    pairs = list(pairs)

    def _set(cb):
        for name, z in pairs:
            cb.Mole(name, float(z))

    return System.Action[CompositionBuilder](_set)


def _with_compounds(fs, names):
    for n in names:
        fs = fs.WithCompound(n)
    return fs


def flash_gas_natural():
    names = ["Methane", "Ethane", "Propane", "N-butane", "N-pentane", "Nitrogen", "Carbon dioxide"]
    z = [0.80, 0.08, 0.04, 0.02, 0.02, 0.02, 0.02]
    fs = _with_compounds(Flowsheet.Create("flash"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    feed = (fs.AddMaterialStream("feed").At(Q.Kelvin(230.0), Q.Bar(30.0))
            .WithMolarFlow(Q.MolPerSecond(1.0)).WithComposition(composition(zip(names, z))))
    vap, liq = fs.AddMaterialStream("vap"), fs.AddMaterialStream("liq")
    fs.AddSeparator("V-1").ConnectFeed(feed, 0).ConnectProduct(vap, 0).ConnectProduct(liq, 1)
    fs.AutoLayout()
    return fs, {"feed": feed, "vap": vap, "liq": liq, "names": names}


def compresor_enfriador():
    names = ["Methane", "Ethane", "Propane", "N-butane", "Carbon dioxide"]
    z = [0.85, 0.07, 0.04, 0.02, 0.02]
    fs = _with_compounds(Flowsheet.Create("comp"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    gas = (fs.AddMaterialStream("gas").At(Q.Kelvin(300.0), Q.Bar(50.0))
           .WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, z))))
    hp, cold = fs.AddMaterialStream("hp"), fs.AddMaterialStream("cold")
    k = (fs.AddCompressor("K-1").WithOutletPressure(Q.Bar(60.0)).WithAdiabaticEfficiencyPercent(75.0)
         .ConnectFeed(gas, 0).ConnectProduct(hp, 0))
    e = fs.AddCooler("E-1").WithOutletTemperature(Q.Kelvin(300.0)).ConnectFeed(hp, 0).ConnectProduct(cold, 0)
    fs.AutoLayout()
    return fs, {"gas": gas, "hp": hp, "cold": cold, "k": k, "e": e}


def depropanizadora_fug(reflux_ratio: float = 1.8403542708971303):
    names = ["Ethane", "Propane", "N-butane", "N-pentane"]
    z = [0.05, 0.45, 0.40, 0.10]
    fs = _with_compounds(Flowsheet.Create("fug"), names).WithPropertyPackage(PropertyPackages.PengRobinson)
    al = (fs.AddMaterialStream("alim").WithPressure(Q.Bar(17.0)).WithVaporFraction(0.0)
          .WithMolarFlow(Q.MolPerSecond(100.0)).WithComposition(composition(zip(names, z))))
    dist, bot = fs.AddMaterialStream("dest"), fs.AddMaterialStream("fondo")
    qc, qr = fs.AddEnergyStream("Qc"), fs.AddEnergyStream("Qr")
    sc = (fs.AddShortcutColumn("T-1").WithKeys("Propane", "N-butane")
          .WithLightKeyInBottoms(0.02).WithHeavyKeyInDistillate(0.02)
          .WithPressures(Q.Bar(16.0), Q.Bar(17.5)).WithRefluxRatio(reflux_ratio)
          .ConnectFeed(al, 0).ConnectProduct(dist, 0).ConnectProduct(bot, 1))
    sc.ConnectEnergyProduct(qc, 0)   # condenser duty leaves the column
    sc.ConnectEnergyFeed(qr, 1)      # reboiler duty enters the column (energy feed port 1)
    fs.AutoLayout()
    return fs, {"alim": al, "dest": dist, "fondo": bot, "qc": qc, "qr": qr, "sc": sc}


ALL = {"flash": flash_gas_natural, "compresor": compresor_enfriador, "fug": depropanizadora_fug}
