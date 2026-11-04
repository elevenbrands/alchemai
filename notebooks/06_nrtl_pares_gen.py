"""Turn notebooks/06_nrtl_raw/dwsim_pares.json into the Swift reference file used by NRTLParesTests
(apps/fluxo/Packages/FluxoEngine/Tests/FluxoEngineTests/NRTLParesDWSIM.swift). Generated code: do not edit by hand."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
d = json.loads((HERE / "06_nrtl_raw" / "dwsim_pares.json").read_text())
dst = HERE.parents[1] / "apps" / "fluxo" / "Packages" / "FluxoEngine" / "Tests" / "FluxoEngineTests" / "NRTLParesDWSIM.swift"

L = ["// Generado por alchemai/notebooks/06_nrtl_pares_gen.py (2026-11-03) a partir de DWSIM 10.2.12, paquete NRTL,",
     "// Txy a 101325 Pa por punto de burbuja (VF = 0). No editar a mano.",
     "import Foundation", "import Testing", "@testable import FluxoEngine", "",
     "struct ParDWSIM: Sendable, CustomTestStringConvertible {", "    let id: String", "    let i: String", "    let j: String",
     "    /// (x_i, T K, y_i) de DWSIM-NRTL a 1 atm.", "    let puntos: [(Double, Double, Double)]",
     "    var testDescription: String { id }", "}", "",
     "enum NRTLParesDWSIM {",
     "    /// Constantes DIPPR 101 que usa DWSIM (ChemSep) para los componentes de los pares.",
     "    static let psat: [String: PresionDeVaporDIPPR101] = ["]
for fid, v in d["vapor_pressure"].items():
    L.append(f'        "{fid}": .init(a: {v["Vapor_Pressure_Constant_A"]}, b: {v["Vapor_Pressure_Constant_B"]}, c: {v["Vapor_Pressure_Constant_C"]}, '
             f'd: {v["Vapor_Pressure_Constant_D"]}, e: {v["Vapor_Pressure_Constant_E"]}, tMin: {v["Vapor_Pressure_TMIN"]}, tMax: {v["Vapor_Pressure_TMAX"]}),')
L += ["    ]", "", "    static let pares: [ParDWSIM] = ["]
for pair, pts in d["pairs"].items():
    i, j = pair.split("|")
    body = ", ".join(f"({p['x']}, {p['T']:.4f}, {p['y']:.6f})" for p in pts if p["T"] is not None)
    L.append(f'        ParDWSIM(id: "{pair}", i: "{i}", j: "{j}", puntos: [{body}]),')
L += ["    ]", "}", ""]
dst.write_text("\n".join(L))
print("written", dst, len(L), "lines")
