"""Generate the ChemE Design Bench v0 cards (yaml) from the Python case definitions.

One card per case, 20 cards: 3 baseline + 10 reference (simulation) + 7 design. The expected
values are DWSIM 10.2.12 results (or analytic oracles) and are the single source of truth for
grading; regenerate with `uv run python packages/bench/build_cards.py`.

Card schema (v0):
    id, kind (simulation|design), title: {es, en}, statement: {es, en}, tags
    expected: {quantity: {value, unit, tol: {rel|abs}}}
    requires: [balances_closed, converged]      # evidence the agent must attach
    source: "DWSIM 10.2.12 PR/NRTL/IAPWS-IF97 + analytic checks"
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.bench.cases_baseline import ALL as BASELINE  # noqa: E402
from packages.bench.cases_ref import CASES as REF  # noqa: E402
from packages.bench.design_cases import DESIGN_CASES  # noqa: E402
from packages.dwsim_client.verify import verify  # noqa: E402

CARDS_DIR = Path(__file__).with_name("cards")
SOURCE = "DWSIM 10.2.12 (Peng-Robinson / NRTL / IAPWS-IF97) + analytic checks"
DEFAULT_TOL = {"rel": 0.01}

# Baseline cases: statements and readouts live here (the builders are shared with the tests).
BASELINE_META = {
    "flash": dict(
        title=("Flash PT de gas natural", "Natural-gas PT flash"),
        statement=("Gas natural 80/8/4/2/2 % C1/C2/C3/nC4/nC5 + 2 % N₂ + 2 % CO₂, 1 mol/s, a 230 K y 30 bar en un separador. Fracción vapor y composición del vapor y del líquido (Peng-Robinson).",
                   "Natural gas 80/8/4/2/2 % C1/C2/C3/nC4/nC5 + 2 % N₂ + 2 % CO₂, 1 mol/s, at 230 K and 30 bar in a separator. Vapor fraction and vapor/liquid compositions (Peng-Robinson)."),
        readouts=lambda h: {"VF": h["feed"].VaporFraction, "y_CH4": h["vap"].OverallMoleFraction("Methane"),
                            "x_CO2": h["liq"].OverallMoleFraction("Carbon dioxide"), "x_C3": h["liq"].OverallMoleFraction("Propane")},
        units={"VF": "-", "y_CH4": "-", "x_CO2": "-", "x_C3": "-"}, tol={"x_CO2": {"abs": 0.003}}),
    "compresor": dict(
        title=("Compresor adiabático + enfriador", "Adiabatic compressor + cooler"),
        statement=("Gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂, 100 mol/s, 300 K y 50 bar, se comprime a 60 bar con η adiabática 75 % y se enfría a 300 K. T de descarga, potencia y calor retirado.",
                   "Gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂, 100 mol/s, 300 K, 50 bar, compressed to 60 bar at 75 % adiabatic efficiency and cooled to 300 K. Discharge T, power and heat removed."),
        readouts=lambda h: {"T_out_K": h["hp"].TemperatureK, "W_kW": _duty(h["k"]), "Q_cooler_kW": h["e"].HeatRemovedKW},
        units={"T_out_K": "K", "W_kW": "kW", "Q_cooler_kW": "kW"}, tol={"T_out_K": {"abs": 0.5}}),
    "fug": dict(
        title=("Depropanizadora por FUG", "Depropanizer by FUG shortcut"),
        statement=("Alimentación líquida saturada 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) a 17 bar; 2 % nC4 en destilado, 2 % C3 en fondo; condensador 16 bar, rehervidor 17.5 bar, R = 1.8404. Destilado, cargas, N_min, R_min y N.",
                   "Saturated-liquid feed 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) at 17 bar; 2 % nC4 in distillate, 2 % C3 in bottoms; condenser 16 bar, reboiler 17.5 bar, R = 1.8404. Distillate, duties, N_min, R_min and N."),
        readouts=lambda h: {"D_mol_s": h["dest"].MolarFlowMolPerSecond, "Qc_kW": h["qc"].EnergyFlowKW, "Qr_kW": h["qr"].EnergyFlowKW,
                            "N_min": h["sc"].MinimumStages, "R_min": h["sc"].MinimumRefluxRatio, "N": h["sc"].TheoreticalStages},
        units={"D_mol_s": "mol/s", "Qc_kW": "kW", "Qr_kW": "kW", "N_min": "-", "R_min": "-", "N": "-"},
        tol={"N_min": {"rel": 0.08}, "R_min": {"rel": 0.1}, "N": {"rel": 0.2}}),   # FUG conventions differ across simulators
}


def _duty(builder):
    p = builder.Object.GetType().GetProperty("DeltaQ")
    return float(p.GetValue(builder.Object))


def _card(cid, kind, title, statement, tags, expected, units, tol):
    exp = {k: {"value": float(v), "unit": units.get(k, "-"), "tol": tol.get(k, DEFAULT_TOL)} for k, v in expected.items()}
    return {"id": cid, "kind": kind, "version": "v0", "title": {"es": title[0], "en": title[1]},
            "statement": {"es": statement[0], "en": statement[1]}, "tags": tags, "expected": exp,
            "requires": ["converged", "balances_closed"], "source": SOURCE}


def build_all():
    cards = []
    for cid, build in BASELINE.items():
        m = BASELINE_META[cid]
        fs, h = build()
        rep = verify(fs)
        assert rep.ok, f"{cid}: {rep.markdown()}"
        cards.append(_card(cid, "simulation", m["title"], m["statement"], ["baseline"], m["readouts"](h), m["units"], m.get("tol", {})))
    for c in REF:
        fs, h = c.build()
        rep = verify(fs)
        assert rep.ok, f"{c.id}: {rep.markdown()}"
        r = c.readouts(h)
        units = {k: _unit_from_name(k) for k in r}
        tol = {k: {"abs": 0.0} if k in ("iterations",) else DEFAULT_TOL for k in r}
        if "iterations" in r:        # solver-dependent, not a physics answer
            r = {k: v for k, v in r.items() if k != "iterations"}
        cards.append(_card(c.id, "simulation", (c.title_es, c.title_en), (c.statement_es, c.statement_en), c.tags, r, units, tol))
    for d in DESIGN_CASES:
        a = d.oracle()
        checks = d.checks(a)
        assert all(ok for _, ok, _ in checks), f"{d.id}: {[c for c in checks if not c[1]]}"
        exp = {k: v for k, v in a.items() if k in d.tolerances}
        cards.append(_card(d.id, "design", (d.title_es, d.title_en), (d.statement_es, d.statement_en), d.tags, exp, d.units, d.tolerances))
    return cards


def _unit_from_name(k: str) -> str:
    for suf, u in (("_K", "K"), ("_bar", "bar"), ("_kW", "kW"), ("_mol_s", "mol/s"), ("_kg_s", "kg/s"), ("_W_K", "W/K"), ("_pct", "%")):
        if k.endswith(suf):
            return u
    return "-"


def main():
    CARDS_DIR.mkdir(exist_ok=True)
    cards = build_all()
    for c in cards:
        (CARDS_DIR / f"{c['id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=110))
    index = [{"id": c["id"], "kind": c["kind"], "title_es": c["title"]["es"], "n_expected": len(c["expected"])} for c in cards]
    (CARDS_DIR / "index.yaml").write_text(yaml.safe_dump({"version": "v0", "source": SOURCE, "cards": index}, sort_keys=False, allow_unicode=True))
    print(f"{len(cards)} cards → {CARDS_DIR}")


if __name__ == "__main__":
    main()
