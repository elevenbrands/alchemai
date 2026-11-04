"""Reference Txy (1 atm) from DWSIM-NRTL for every NRTL pair that FluxoEngine carries, plus the DIPPR-101
vapor-pressure constants DWSIM used, so the Swift test can separate "same algorithm + parameters" from
"different Psat source". Writes notebooks/06_nrtl_raw/dwsim_pares.json.
Run: uv run python -u notebooks/06_nrtl_pares_dwsim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client.fluent import fluent  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
import System  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402

# Fluxo id -> DWSIM compound name
NAMES = {"etanol": "Ethanol", "agua": "Water", "metanol": "Methanol", "benceno": "Benzene", "tolueno": "Toluene",
         "n-butano": "N-butane", "n-pentano": "N-pentane", "n-hexano": "N-hexane", "n-heptano": "N-heptane", "n-octano": "N-octane"}
# Same keys and order as BaseDeComponentes.nrtl in FluxoEngine (i|j).
PAIRS = ["etanol|agua", "metanol|agua", "metanol|etanol", "etanol|benceno", "etanol|tolueno", "etanol|n-heptano",
         "etanol|n-octano", "n-pentano|etanol", "n-hexano|etanol", "metanol|benceno", "metanol|tolueno", "metanol|n-hexano",
         "metanol|n-heptano", "metanol|n-octano", "n-butano|metanol", "benceno|tolueno"]
XS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
OUT = Path(__file__).parent / "06_nrtl_raw"


def comp(pairs):
    def _set(cb):
        for name, z in pairs:
            cb.Mole(name, z)
    return System.Action[CompositionBuilder](_set)


out = {"pressure_Pa": 101325.0, "xs": XS, "vapor_pressure": {}, "pairs": {}}

# Vapor-pressure constants of every compound involved (one flowsheet with all of them).
fs0 = Flowsheet.Create("vp")
for n in NAMES.values():
    fs0 = fs0.WithCompound(n)
fs0 = fs0.WithPropertyPackage(PropertyPackages.NRTL)
for fid, n in NAMES.items():
    c = fs0.Inner.SelectedCompounds[n]
    out["vapor_pressure"][fid] = {k: getattr(c, k) for k in ("Vapor_Pressure_Constant_A", "Vapor_Pressure_Constant_B",
                                                            "Vapor_Pressure_Constant_C", "Vapor_Pressure_Constant_D",
                                                            "Vapor_Pressure_Constant_E", "Vapor_Pressure_TMIN", "Vapor_Pressure_TMAX",
                                                            "VaporPressureEquation", "Normal_Boiling_Point")}
    print("VP", fid, out["vapor_pressure"][fid]["VaporPressureEquation"], out["vapor_pressure"][fid]["Normal_Boiling_Point"])

for pair in PAIRS:
    a, b = pair.split("|")
    na, nb = NAMES[a], NAMES[b]
    pts = []
    for x in XS:
        fs = Flowsheet.Create(f"{pair}-{x}").WithCompound(na).WithCompound(nb).WithPropertyPackage(PropertyPackages.NRTL)
        st = fs.AddMaterialStream("b").WithPressure(Q.Pascal(101325)).WithMolarFlow(Q.MolPerSecond(1)).WithVaporFraction(0.0) \
            .WithComposition(comp([(na, x), (nb, 1 - x)]))
        fs.AutoLayout()
        errs = list(fs.TrySolve())
        if errs:
            print("ERR", pair, x, [str(e) for e in errs][:1])
            pts.append({"x": x, "T": None, "y": None, "error": str(errs[0])[:200]})
            continue
        so = st.Object
        try:
            y = so.Phases[2].Compounds[na].MoleFraction
        except Exception:  # noqa: BLE001
            y = None
        pts.append({"x": x, "T": st.TemperatureK, "y": y})
    out["pairs"][pair] = pts
    ok = [p for p in pts if p["T"] is not None]
    print(f"{pair:20s} {len(ok)}/{len(XS)} puntos; T {min(p['T'] for p in ok):.2f}–{max(p['T'] for p in ok):.2f} K" if ok else f"{pair}: sin puntos")

OUT.mkdir(exist_ok=True)
(OUT / "dwsim_pares.json").write_text(json.dumps(out, indent=1, default=str))
print("written", OUT / "dwsim_pares.json")
