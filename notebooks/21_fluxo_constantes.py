"""Notebook 21 — Fluxo's 24 components checked against NIST WebBook and other cited sources (bloque 30).

Reads the component table straight from FluxoEngine's Swift source and compares, per compound:
  MW (from the formula, IUPAC atomic weights via `chemicals`); Tc and Pc against IUPAC critically evaluated data
  (Ambrose et al.), then the reference equations of state (HEOS = CoolProp, NIST-equivalent), then NIST WebBook;
  Tb and ω against HEOS, then CRC / PSRK; ΔHf°(gas, 298 K) against ATcT, then NIST WebBook; and the ideal-gas Cp
  polynomial (Reid-Prausnitz-Poling) against TRC ideal-gas tables, then JANAF, then CoolProp, at 300, 500 and 800 K.
Run 1 (2026-11-06) used NIST WebBook first and HEOS_FIT for Cp: 150/168 in tolerance. Every flag was checked against
two or three other sources: WebBook entries inside `chemicals` were the outliers (Ar Pc 30.65 bar, CO Tc 134.45 K,
CH4 Tb 111.0 K, iC4 Tb 262 K, MeOH ΔHf −205 kJ/mol) and HEOS_FIT Cp is invalid at 800 K (C2H4 69 107 J/mol/K).
That run is kept in 21_fluxo_constantes_corrida1.json; this ordering is the one reported.
Tolerances are fixed before reading the result (below). Output: notebooks/21_fluxo_constantes.md + .json.
Run: PYTHONPATH=. uv run python notebooks/21_fluxo_constantes.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import chemicals as C
from thermo import HeatCapacityGas

SWIFT = Path(__file__).resolve().parents[2] / "apps/fluxo/Packages/FluxoEngine/Sources/FluxoEngine/Componentes/BaseDeComponentes.swift"
OUT = Path(__file__).with_suffix("")
TOL = {"MW": ("abs", 0.02), "Tc": ("rel", 0.005), "Pc": ("rel", 0.01), "omega": ("abs", 0.01), "Tb": ("abs", 0.5),
       "Hf": ("abs", 1.0), "Cp": ("rel", 0.02)}
ROW = re.compile(r'c\("([^"]+)", "[^"]+", "([^"]+)", "([^"]+)", ([-\d.e]+), ([-\d.e]+), ([-\d.e]+), ([-\d.e]+), ([-\d.e]+), ([-\d.e]+), '
                 r'\(([-\d.e]+), ([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\)\)')


def first(fn, cas, methods):
    for m in methods:
        try:
            v = fn(cas, method=m)
            if v is not None:
                return v, m
        except Exception:
            continue
    return None, None


def main():
    rows = ROW.findall(SWIFT.read_text())
    assert len(rows) == 24, len(rows)
    out, flags = [], []
    for r in rows:
        cid, formula, cas = r[0], r[1], r[2]
        mw, tc, pc, om, tb, hf = map(float, r[3:9])
        cp = tuple(map(float, r[9:13]))
        ref = {}
        ref["MW"] = (C.molecular_weight(C.nested_formula_parser(formula)), "IUPAC (fórmula)")
        ref["Tc"] = first(C.Tc, cas, ["IUPAC", "HEOS", "WEBBOOK", "CRC"])
        p, m = first(C.Pc, cas, ["IUPAC", "HEOS", "WEBBOOK", "CRC"]); ref["Pc"] = (p / 1e5 if p else None, m)
        ref["omega"] = first(C.omega, cas, ["HEOS", "PSRK", "PD", "YAWS"])
        ref["Tb"] = first(C.Tb, cas, ["HEOS", "CRC_ORG", "WEBBOOK", "COMMON_CHEMISTRY"])
        h, m = first(C.Hfg, cas, ["ATCT_G", "WEBBOOK", "CRC"]); ref["Hf"] = (h / 1000 if h is not None else None, m)
        ours = {"MW": mw, "Tc": tc, "Pc": pc, "omega": om, "Tb": tb, "Hf": hf}
        line = {"id": cid, "cas": cas}
        for k, v in ours.items():
            rv, src = ref[k]
            if rv is None:
                line[k] = {"fluxo": v, "ref": None, "src": None, "ok": None}
                continue
            kind, tol = TOL[k]
            dev = (v - rv) / rv if kind == "rel" else v - rv
            ok = abs(dev) <= tol
            line[k] = {"fluxo": v, "ref": round(rv, 5), "src": src, "dev": round(dev, 5), "ok": ok}
            if not ok:
                flags.append(f"{cid}.{k}: Fluxo {v} vs {src} {rv:.5g} (desv {dev:+.4g})")
        hc = HeatCapacityGas(CASRN=cas)
        src = next((m for m in ("TRCIG", "JANAF", "COOLPROP", "POLING_POLY") if m in hc.all_methods), None)
        cpdev = []
        for T in (300.0, 500.0, 800.0):
            ours_cp = cp[0] + cp[1] * T + cp[2] * T ** 2 + cp[3] * T ** 3
            try:
                rv = hc.calculate(T, src) if src else None
            except Exception:
                rv = None
            if rv:
                cpdev.append(round((ours_cp - rv) / rv, 4))
        worst = max(cpdev, key=abs) if cpdev else None
        line["Cp"] = {"dev_300_500_800": cpdev, "src": src, "ok": worst is not None and abs(worst) <= TOL["Cp"][1]}
        if worst is not None and abs(worst) > TOL["Cp"][1]:
            flags.append(f"{cid}.Cp: desviación máx {worst:+.1%} vs {src} (300/500/800 K: {cpdev})")
        out.append(line)
    OUT.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    n_checks = sum(1 for l in out for k in TOL if isinstance(l.get(k), dict) and l[k].get("ok") is not None)
    srcs = sorted({f"{k}:{l[k]['src']}" for l in out for k in TOL if isinstance(l.get(k), dict) and l[k].get("src")})
    print("fuentes usadas:", ", ".join(srcs))
    n_ok = sum(1 for l in out for k in TOL if isinstance(l.get(k), dict) and l[k].get("ok"))
    print(f"{n_ok}/{n_checks} constantes dentro de tolerancia")
    for f in flags:
        print(" ", f)


if __name__ == "__main__":
    main()
