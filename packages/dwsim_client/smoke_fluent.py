"""Headless smoke test of DWSIM 10 via the Fluent API (pythonnet, in-process).

Case 1: the documented water mixer (steam tables).
Case 2: PT flash of C1/C2/C3 (0.7/0.2/0.1) at 230 K, 30 bar, Peng-Robinson — the same case
        solved with `thermo` in notebooks/smoke_stack.py (VF = 0.7426) to cross-check engines.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client.fluent import fluent  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
print("Fluent API loaded")

# --- Case 1: mixer -------------------------------------------------------------------------
fs = (Flowsheet.Create("PyMixer")
      .WithCompound("Water")
      .WithPropertyPackage(PropertyPackages.SteamTables))
in1 = fs.AddMaterialStream("inlet1").At(Q.Kelvin(300.0), Q.Pascal(101325.0)).WithMassFlow(Q.KgPerSecond(100.0))
in2 = fs.AddMaterialStream("inlet2").At(Q.Kelvin(348.0), Q.Pascal(101325.0)).WithMassFlow(Q.KgPerSecond(50.0))
out = fs.AddMaterialStream("outlet")
fs.AddMixer("MIX-1").ConnectFeed(in1, 0).ConnectFeed(in2, 1).ConnectProduct(out, 0)
fs.AutoLayout()
fs.Solve()
print(f"[mixer] outlet T = {out.TemperatureK:.3f} K, mass flow = {out.MassFlowKgPerSecond:.3f} kg/s")

# --- Case 2: PR flash C1/C2/C3 -----------------------------------------------------------
import System  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402

fs2 = (Flowsheet.Create("PRFlash")
       .WithCompound("Methane").WithCompound("Ethane").WithCompound("Propane")
       .WithPropertyPackage(PropertyPackages.PengRobinson))


def _composition(cb):
    for name, z in (("Methane", 0.7), ("Ethane", 0.2), ("Propane", 0.1)):
        cb.Mole(name, z)


feed = (fs2.AddMaterialStream("feed")
        .At(Q.Kelvin(230.0), Q.Bar(30.0))
        .WithMolarFlow(Q.MolPerSecond(1.0))
        .WithComposition(System.Action[CompositionBuilder](_composition)))
fs2.AutoLayout()
fs2.Solve()
print(f"[flash] DWSIM PR VF = {feed.VaporFraction:.4f}   (thermo PRMIX same case: 0.7426)")
