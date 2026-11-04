"""Dump what DWSIM's NRTL package actually uses for ethanol-water, so FluxoEngine can reproduce it:
NRTL binary parameters, vapor-pressure constants, ln(phi) of both phases on a grid and a Txy at 1 atm.
Writes notebooks/05_nrtl_raw/dwsim_nrtl.json. Run: uv run python -u notebooks/05_nrtl_dwsim_dump.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client.fluent import fluent  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
import clr  # noqa: E402
clr.AddReference("DWSIM.Thermodynamics")
import System  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402
from DWSIM.Thermodynamics.PropertyPackages import State  # noqa: E402

EW = ["Ethanol", "Water"]
CAS = {"Ethanol": "64-17-5", "Water": "7732-18-5"}
OUT = Path(__file__).parent / "05_nrtl_raw"


def comp(pairs):
    def _set(cb):
        for c, v in pairs:
            cb.Mole(c, v)
    return System.Action[CompositionBuilder](_set)


def arr(v):
    a = System.Array.CreateInstance(System.Double, len(v))
    for i, x in enumerate(v):
        a[i] = float(x)
    return a


fs = Flowsheet.Create("nrtl_dump")
for c in EW:
    fs = fs.WithCompound(c)
fs = fs.WithPropertyPackage(PropertyPackages.NRTL)
s = fs.AddMaterialStream("s").WithPressure(Q.Pascal(101325)).WithMolarFlow(Q.MolPerSecond(1)).WithTemperature(Q.Kelvin(350)) \
    .WithComposition(comp(zip(EW, [0.5, 0.5])))
fs.AutoLayout()
errs = list(fs.TrySolve())
print("solve errors:", [str(e) for e in errs])
fso = fs.Inner
from DWSIM.Thermodynamics.PropertyPackages import NRTLPropertyPackage  # noqa: E402
pp = list(fso.PropertyPackages.Values)[0]
pp = getattr(pp, "__implementation__", pp)   # pythonnet returns the interface; unwrap to the concrete class
pp.CurrentMaterialStream = s.Object
print("pp:", type(pp), pp.ComponentName)

out = {"package": str(pp.ComponentName)}

# --- NRTL binary parameters as stored by DWSIM
ip = {}
try:
    ipd = pp.m_uni.InteractionParameters   # Auxiliary.NRTL holds the binary set (keyed by CAS)
    for k1 in ipd.Keys:
        for k2 in ipd[k1].Keys:
            d = ipd[k1][k2]
            ip[f"{k1}|{k2}"] = {f: getattr(d, f) for f in ("A12", "A21", "B12", "B21", "C12", "C21", "alpha12")}
except Exception as e:  # noqa: BLE001
    print("IP dump failed:", e)
out["nrtl_ip"] = ip
# gamma straight from the model on the same grid (removes any Psat ambiguity when fitting tau)
gam = []
for T in (330.0, 350.0, 373.15):
    for x1 in (0.1, 0.3, 0.5, 0.7, 0.9):
        g = list(pp.m_uni.GAMMA_MR(float(T), arr([x1, 1 - x1]), System.Array[System.String]([CAS[c] for c in EW])))
        gam.append({"T": T, "x1": x1, "gamma": g})
out["gamma_grid"] = gam
print("gamma sample:", gam[7])
print("NRTL IP:", json.dumps(ip, indent=1))

# --- vapor pressure constants of each compound
vp = {}
for name in EW:
    c = fso.SelectedCompounds[name]
    vp[name] = {f: getattr(c, f) for f in ("Vapor_Pressure_Constant_A", "Vapor_Pressure_Constant_B", "Vapor_Pressure_Constant_C",
                                           "Vapor_Pressure_Constant_D", "Vapor_Pressure_Constant_E", "Vapor_Pressure_TMIN",
                                           "Vapor_Pressure_TMAX", "VaporPressureEquation", "Critical_Temperature",
                                           "Critical_Pressure", "Acentric_Factor", "Normal_Boiling_Point", "Molar_Weight")}
    vp[name]["Psat_grid"] = {str(T): pp.AUX_PVAPi(EW.index(name), float(T)) for T in range(300, 376, 5)}
out["vapor_pressure"] = vp
print("VP:", json.dumps(vp, indent=1, default=str))

# --- ln(phi) of both phases on a grid at 1 atm
grid = []
for T in (330.0, 350.0, 360.0, 373.15):
    for x1 in (0.0001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9999):
        x = arr([x1, 1 - x1])
        lnL = list(pp.DW_CalcFugCoeff(x, T, 101325.0, State.Liquid))
        lnV = list(pp.DW_CalcFugCoeff(x, T, 101325.0, State.Vapor))
        grid.append({"T": T, "x1": x1, "lnphi_L": lnL, "lnphi_V": lnV})
out["lnphi_grid"] = grid
print("lnphi sample:", grid[10])

# --- Txy at 1 atm by bubble-point streams (VF = 0)
txy = []
for i in range(0, 21):
    x1 = min(max(i / 20, 1e-6), 1 - 1e-6)
    fs2 = Flowsheet.Create(f"txy{i}")
    for c in EW:
        fs2 = fs2.WithCompound(c)
    fs2 = fs2.WithPropertyPackage(PropertyPackages.NRTL)
    st = fs2.AddMaterialStream("b").WithPressure(Q.Pascal(101325)).WithMolarFlow(Q.MolPerSecond(1)).WithVaporFraction(0.0) \
        .WithComposition(comp(zip(EW, [x1, 1 - x1])))
    fs2.AutoLayout()
    e2 = list(fs2.TrySolve())
    if e2:
        print("txy error", x1, [str(e) for e in e2])
        continue
    so = st.Object
    try:
        y1 = so.Phases[2].Compounds["Ethanol"].MoleFraction
    except Exception:  # noqa: BLE001
        y1 = None
    txy.append({"x1": x1, "T": st.TemperatureK, "y1": y1})
    print(f"TXY x={x1:.3f} T={st.TemperatureK:.3f} y={y1}")
out["txy_1atm"] = txy

OUT.mkdir(exist_ok=True)
(OUT / "dwsim_nrtl.json").write_text(json.dumps(out, indent=1, default=str))
print("written", OUT / "dwsim_nrtl.json")
