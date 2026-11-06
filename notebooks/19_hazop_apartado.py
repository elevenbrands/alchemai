"""Notebook 19 — held-out HAZOP set: blind recall + precision labels (bloque 25).

Usage:
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py run    # blind run -> notebooks/hazop_apartado/rows.json + recall.md
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py review # for the reviewer: process descriptions + rows, no rules, no reference
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py score  # merge labels.yaml (reviewer output) -> precision tables
Labels follow packages/hazop/book/PRECISION_RUBRIC.md: coincide | plausible | ruido | espuria.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from packages.hazop.book.heldout_cases import HELDOUT, DESCRIPTIONS
from packages.hazop.book.score import score, to_markdown

OUT = Path("notebooks/hazop_apartado")
REF = Path("packages/hazop/book/heldout.yaml")


def row_key(r: dict) -> str:
    return f"{r['node']} · {r['parameter']} · {r['deviation']}"


def run():
    OUT.mkdir(exist_ok=True)
    s = score(OUT / "dexpi", ref_path=REF, builders=HELDOUT)
    (OUT / "recall.md").write_text(to_markdown(s))
    (OUT / "rows.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(to_markdown(s))


def review():
    s = json.loads((OUT / "rows.json").read_text())
    out = ["# Filas para el revisor (sin reglas, sin referencia publicada)", ""]
    for c in s["cases"]:
        out += [f"## {c['id']} — {c['titulo']}", "", DESCRIPTIONS[c["id"]], "", "| # | Fila | Causas | Consecuencias | Salvaguardas |", "|---|---|---|---|---|"]
        for i, r in enumerate(c["rows"], 1):
            cell = lambda xs: "; ".join(xs).replace("|", "/")
            out.append(f"| {c['id']}:{i} | {row_key(r)} | {cell(r['causes'])} | {cell(r['consequences'])} | {cell(r['safeguards'])} |")
        out.append("")
    (OUT / "review_input.md").write_text("\n".join(out))
    print(f"{sum(len(c['rows']) for c in s['cases'])} filas en {OUT / 'review_input.md'}")


def score_labels():
    s = json.loads((OUT / "rows.json").read_text())
    labels = yaml.safe_load((OUT / "labels.yaml").read_text())
    tot, per_case, lines = Counter(), {}, []
    for c in s["cases"]:
        matched = {i["row"] for i in c["items"] if i["row"]}
        cnt = Counter()
        for i, r in enumerate(c["rows"], 1):
            k = row_key(r)
            lab = "coincide" if k in matched else labels.get(f"{c['id']}:{i}", {}).get("etiqueta", "sin etiqueta")
            cnt[lab] += 1
        per_case[c["id"]] = cnt
        tot += cnt
    n = sum(tot.values())
    lines += ["| Caso | Filas | coincide | plausible | ruido | espuria | sin etiqueta |", "|---|---|---|---|---|---|---|"]
    for cid, cnt in per_case.items():
        m = sum(cnt.values())
        lines.append(f"| {cid} | {m} | " + " | ".join(str(cnt[k]) for k in ("coincide", "plausible", "ruido", "espuria", "sin etiqueta")) + " |")
    lines.append(f"| **Total** | {n} | " + " | ".join(str(tot[k]) for k in ("coincide", "plausible", "ruido", "espuria", "sin etiqueta")) + " |")
    lines += ["", f"Precisión estricta = {tot['coincide']}/{n} = {tot['coincide'] / n:.0%}",
              f"Precisión útil = {tot['coincide'] + tot['plausible']}/{n} = {(tot['coincide'] + tot['plausible']) / n:.0%}",
              f"Tasa de espurias = {tot['espuria']}/{n} = {tot['espuria'] / n:.1%}"]
    (OUT / "precision.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": run, "review": review, "score": score_labels}[sys.argv[1]]()
