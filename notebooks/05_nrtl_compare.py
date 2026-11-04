"""Week-5 study: NRTL in FluxoEngine vs DWSIM 10 (ethanol-water).

Reads notebooks/05_nrtl_raw/{dwsim_nrtl.json, fluxo_nrtl_columnas.jsonl, fluxo_txy_nrtl.txt,
fluxo_txy_nrtl_dwsim.txt} and 04_dwsim_columnas_raw/B_nrtl.json, writes notebooks/05_nrtl_fluxo_vs_dwsim.md.
Fluxo lines come from `swift test` in apps/fluxo/Packages/FluxoEngine (NRTLTests, LineaBaseColumnasTests).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "05_nrtl_raw"
dw = json.loads((RAW / "dwsim_nrtl.json").read_text())
col_dw = json.loads((HERE / "04_dwsim_columnas_raw" / "B_nrtl.json").read_text())
fluxo = {}
for line in (RAW / "fluxo_nrtl_columnas.jsonl").read_text().splitlines():
    if line.startswith("{"):
        d = json.loads(line)
        fluxo[d["caso"]] = d
txy_fluxo = [tuple(float(v) for v in re.findall(r"=([-\d.eE+]+)", ln)) for ln in (RAW / "fluxo_txy_nrtl.txt").read_text().splitlines()]
txy_dw = [(p["x1"], p["T"], p["y1"]) for p in dw["txy_1atm"]]
repro = (RAW / "fluxo_txy_nrtl_dwsim.txt").read_text().strip()

ip = next(v for k, v in dw["nrtl_ip"].items() if k.lower() in ("etanol|agua", "ethanol|water"))
vp = dw["vapor_pressure"]


def psat_dw(name, T):
    v = vp[name]
    return math.exp(v["Vapor_Pressure_Constant_A"] + v["Vapor_Pressure_Constant_B"] / T + v["Vapor_Pressure_Constant_C"] * math.log(T)
                    + v["Vapor_Pressure_Constant_D"] * T ** v["Vapor_Pressure_Constant_E"])


def psat_perry(c, T):
    # Perry's 8th, table 2-8 (as transcribed into FluxoEngine's BaseDeComponentes.psatDIPPR)
    A, B, C, D, E = {"Ethanol": (73.304, -7122.3, -7.1424, 2.8853e-06, 2), "Water": (73.649, -7258.2, -7.3037, 4.1653e-06, 2)}[c]
    return math.exp(A + B / T + C * math.log(T) + D * T ** E)


def tb(f, c):
    lo, hi = 300.0, 400.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(c, mid) < 101325:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


L = ["# NRTL en FluxoEngine vs DWSIM 10 — etanol-agua (semana 5)", "",
     "Objetivo del bloque 11: portar NRTL (E4) al motor de Fluxo y reproducir la columna etanol-agua que PR no podía",
     "resolver (`04_columnas_rigurosas.md`). Fuente de verdad: DWSIM 10.2.12 con su paquete NRTL.", "",
     "## Qué hace DWSIM-NRTL exactamente (volcado con `05_nrtl_dwsim_dump.py`)", "",
     "- Vapor **ideal**: `DW_CalcFugCoeff(..., Vapor)` devuelve φ = 1 en toda la malla.",
     "- Líquido: φ_i^L = γ_i·Psat_i/P, sin Poynting ni φ_sat.",
     f"- τ_ij = A_ij/(R·T) con A en cal/mol y R = 1.98721: etanol(1)-agua(2) A12 = {ip['A12']}, A21 = {ip['A21']}, α = {ip['alpha12']}",
     "  (ajuste sobre 20 puntos de la malla: rms ln γ = 0.001; las otras formas de τ dan rms > 0.17).",
     f"- Psat DIPPR 101 propia (ChemSep): Tb etanol {tb(psat_dw, 'Ethanol'):.2f} K, agua {tb(psat_dw, 'Water'):.2f} K.",
     f"  Perry's 8.ª (tabla 2-8, lo que lleva Fluxo): etanol {tb(psat_perry, 'Ethanol'):.2f} K, agua {tb(psat_perry, 'Water'):.2f} K.",
     "  El etanol de DWSIM hierve 0.33 K por encima del valor experimental (351.44 K); esa es la mayor fuente de",
     "  diferencia entre «reproducir DWSIM» y «estar bien».", "",
     "El conjunto NRTL de DWSIM tiene 894 pares; 16 caen entre los 24 componentes de Fluxo y quedaron transcritos en",
     "`BaseDeComponentes.nrtl` (solo etanol-agua está validado; el resto se marca en MOCKS).", "",
     "## Diseño en FluxoEngine", "",
     "`PaqueteDeFluido.modeloDeLiquido: ModeloDeLiquido` (`.ecuacionDeEstado` por omisión, `.nrtl(parámetros, vapor:)`).",
     "Con NRTL el líquido va por γ-φ y el vapor por la cúbica (`.cubica`, con φ_sat y Poynting para que un puro hierva",
     "exactamente a su Psat) o por gas ideal (`.gasIdeal`, lo que hace DWSIM). `Componente.presionDeVapor` (DIPPR 101,",
     "Perry's) alimenta Psat y ΔH_vap por Clapeyron; h^E y s^E salen de NRTL por derivada numérica. Flash, burbuja/rocío,",
     "columna y balances de energía no cambiaron: solo piden `lnPhi` y `entalpiaRapida` al paquete.", "",
     "## Txy a 1 atm", "",
     f"Reproducción (Psat de DWSIM + vapor ideal), 19 puntos x = 0.05…0.95: `{repro}`.", "",
     "Datos propios (Perry's + vapor PR) frente a DWSIM:", "",
     "| x etanol | T DWSIM (K) | T Fluxo (K) | ΔT (K) | y DWSIM | y Fluxo | Δy |", "|---|---|---|---|---|---|---|"]
fx = {round(x, 3): (T, y) for x, T, y in txy_fluxo}
for x, T, y in txy_dw:
    k = round(x, 3)
    if k in fx:
        L.append(f"| {x:.2f} | {T:.3f} | {fx[k][0]:.3f} | {fx[k][0] - T:+.3f} | {y:.4f} | {fx[k][1]:.4f} | {fx[k][1] - y:+.4f} |")
az = min(txy_fluxo, key=lambda p: p[1])
L += ["", f"Azeótropo Fluxo: x = {az[0]:.3f}, T = {az[1]:.2f} K (experimental ≈ 0.894, 351.3 K). Las ΔT de −0.3 K en la zona rica",
      "en etanol son la diferencia de Psat entre fuentes, no del modelo.", "",
      "## Columna de 20 etapas (caso B de la semana 4)", "",
      "| Variable | DWSIM NRTL | Fluxo NRTL (Psat DWSIM, vapor ideal) | Fluxo NRTL (Perry's, vapor PR) |", "|---|---|---|---|"]
a, b = fluxo["etoh_nrtl_dwsim"], fluxo["etoh_nrtl"]
rows = [("x_D etanol", col_dw["xD"][0], a["xD"][0], b["xD"][0], ".4f"), ("x_B etanol", col_dw["xB"][0], a["xB"][0], b["xB"][0], ".4f"),
        ("T destilado (K)", col_dw["T_dist_K"], a["tDestK"], b["tDestK"], ".2f"), ("T fondo (K)", col_dw["T_bot_K"], a["tFondoK"], b["tFondoK"], ".2f"),
        ("Qc (kW)", col_dw["Qc_kW"], abs(a["qCondW"]) / 1000, abs(b["qCondW"]) / 1000, ".1f"),
        ("Qr (kW)", col_dw["Qr_kW"], a["qRehW"] / 1000, b["qRehW"] / 1000, ".1f"),
        ("iteraciones", "—", a["iter"], b["iter"], "")]
for name, d, f1, f2, fmt in rows:
    f = (lambda v: format(v, fmt)) if fmt else str
    L.append(f"| {name} | {f(d) if d != '—' else d} | {f(f1)} | {f(f2)} |")
L += ["", "Perfil de T por etapa (K):", "", "| Etapa | DWSIM | Fluxo (repro) | Fluxo (propio) |", "|---|---|---|---|"]
for j in range(20):
    L.append(f"| {j + 1} | {col_dw['profile']['T'][j]:.2f} | {a['T'][j]:.2f} | {b['T'][j]:.2f} |")
L += ["", "## Conclusión", "",
      "- Con los mismos datos que DWSIM, Fluxo reproduce la columna: x_D a 0.0004, temperaturas a 0.01 K, cargas a 2 %",
      "  (la diferencia de cargas es el modelo de entalpía de líquido: Clapeyron + h^E en Fluxo vs el de DWSIM).",
      "- Con datos propios la separación es ligeramente mejor (x_D 0.8088) porque el etanol de Perry's es 0.33 K más",
      "  volátil que el de ChemSep; ambas corridas convergen en 22 iteraciones donde PR no convergía en 2000.",
      "- Pendiente para Fluxo 1.1: exponer el modelo en la UI (Figma) y avisar cuando falten pares binarios.", ""]
(HERE / "05_nrtl_fluxo_vs_dwsim.md").write_text("\n".join(L))
print("\n".join(L[-28:-24]))
print("written", HERE / "05_nrtl_fluxo_vs_dwsim.md")
