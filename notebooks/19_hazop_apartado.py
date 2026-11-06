"""Notebook 19 — held-out HAZOP set: blind recall + precision labels (bloque 25).

Usage:
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py run    # blind run -> notebooks/hazop_apartado/rows.json + recall.md
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py review # for the reviewer: process descriptions + rows, no rules, no reference
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py score rows_blind.json  # precision of the blind run
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py review_new            # rows added by new rules, for the reviewer
  PYTHONPATH=. uv run python notebooks/19_hazop_apartado.py score                 # precision of the current run
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


def row_text(r: dict) -> str:
    """Full text of a row: a blind label is reused only when the row did not change."""
    return row_key(r) + "|" + "|".join(r["causes"] + r["consequences"] + r["safeguards"])


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


def _load_labels(rows_file: str) -> dict:
    """Labels keyed by (case, row_key). Blind labels are by index into rows_blind.json; new rows are labelled by key."""
    blind = json.loads((OUT / "rows_blind.json").read_text())
    by_idx = yaml.safe_load((OUT / "labels.yaml").read_text())
    out = {}
    for c in blind["cases"]:
        for i, r in enumerate(c["rows"], 1):
            lab = by_idx.get(f"{c['id']}:{i}")
            if lab:
                out[(c["id"], row_text(r))] = lab["etiqueta"]
    if (OUT / "labels_new.yaml").exists():
        for k, v in yaml.safe_load((OUT / "labels_new.yaml").read_text()).items():
            cid, key = k.split(":", 1)
            out[(cid, key)] = v["etiqueta"]
    return out


def _label_for(labels: dict, cid: str, r: dict) -> str:
    return labels.get((cid, row_text(r))) or labels.get((cid, row_key(r))) or "sin etiqueta"


def review_new():
    """Rows of the current run that the blind run did not have: the reviewer labels only these, keyed by case:row_key."""
    s = json.loads((OUT / "rows.json").read_text())
    blind = json.loads((OUT / "rows_blind.json").read_text())
    labels = _load_labels("rows.json")
    out, n = ["# Filas NUEVAS o MODIFICADAS para el revisor (sin reglas, sin referencia publicada)", ""], 0
    for c in s["cases"]:
        new = [r for r in c["rows"] if _label_for(labels, c["id"], r) == "sin etiqueta"]
        if not new:
            continue
        out += [f"## {c['id']} — {c['titulo']}", "", DESCRIPTIONS[c["id"]], "", "| id | Causas | Consecuencias | Salvaguardas |", "|---|---|---|---|"]
        cell = lambda xs: "; ".join(xs).replace("|", "/")
        for r in new:
            out.append(f"| {c['id']}:{row_key(r)} | {cell(r['causes'])} | {cell(r['consequences'])} | {cell(r['safeguards'])} |")
            n += 1
        out.append("")
    (OUT / "review_input_new.md").write_text("\n".join(out))
    print(f"{n} filas nuevas en {OUT / 'review_input_new.md'}")


def score_labels():
    rows_file = sys.argv[2] if len(sys.argv) > 2 else "rows.json"
    s = json.loads((OUT / rows_file).read_text())
    labels = _load_labels(rows_file)
    tot, per_case, lines = Counter(), {}, []
    for c in s["cases"]:
        matched = {i["row"] for i in c["items"] if i["row"]}
        cnt = Counter()
        for i, r in enumerate(c["rows"], 1):
            k = row_key(r)
            lab = "coincide" if k in matched else _label_for(labels, c["id"], r)
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
    (OUT / ("precision_blind.md" if "blind" in rows_file else "precision.md")).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": run, "review": review, "review_new": review_new, "score": score_labels}[sys.argv[1]]()
