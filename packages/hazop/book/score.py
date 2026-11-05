"""Score the rule-based HAZOP against published HAZOPs (`reference.yaml`).

Each case is written as a DEXPI file and read back through `hazop_from_dexpi`, the same path a third-party file takes.
A reference item is found when one row has a node in `nodes`, "parameter deviation" matching `dev` and
causes + consequences + safeguards matching `concept` (lower-case, accents stripped). Recall is reported separately
for items the process data can reveal (`alcance: datos`) and for all items.
"""
from __future__ import annotations

import re
import tempfile
import unicodedata
from pathlib import Path

import yaml

from packages.hazop.book.cases import CASES, write_all
from packages.hazop.dexpi_graph import hazop_from_dexpi

REF = Path(__file__).with_name("reference.yaml")


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def find(item: dict, rows: list[dict]) -> dict | None:
    m = item["match"]
    for r in rows:
        if r["node"] not in m["nodes"]:
            continue
        if not re.search(m["dev"], norm(f"{r['parameter']} {r['deviation']}")):
            continue
        if re.search(m["concept"], norm(" ".join(r["causes"] + r["consequences"] + r["safeguards"]))):
            return r
    return None


def score(out_dir: str | Path | None = None) -> dict:
    ref = yaml.safe_load(REF.read_text())
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="hazop_book_"))
    paths = write_all(out_dir)
    cases, tot = [], {"datos": [0, 0], "todos": [0, 0]}
    for c in ref["casos"]:
        h = hazop_from_dexpi(paths[c["id"]])
        items = []
        for it in c["items"]:
            r = find(it, h["rows"])
            items.append({"id": it["id"], "alcance": it["alcance"], "desviacion": it["desviacion"], "found": r is not None,
                          "row": f"{r['node']} · {r['parameter']} · {r['deviation']}" if r else None})
            for k in ("todos",) + (("datos",) if it["alcance"] == "datos" else ()):
                tot[k][0] += r is not None
                tot[k][1] += 1
        used = {i["row"] for i in items if i["row"]}
        extra = [f"{r['node']} · {r['parameter']} · {r['deviation']}" for r in h["rows"]
                 if f"{r['node']} · {r['parameter']} · {r['deviation']}" not in used]
        cases.append({"id": c["id"], "titulo": c["titulo"], "n_rows": h["n_rows"], "missing": h["missing"], "items": items, "extra_rows": extra})
    return {"cases": cases, "recall_datos": tot["datos"][0] / tot["datos"][1], "found_datos": tot["datos"],
            "recall_todos": tot["todos"][0] / tot["todos"][1], "found_todos": tot["todos"], "dexpi_dir": str(out_dir)}


def to_markdown(s: dict) -> str:
    out = [f"Recall (alcance datos): {s['found_datos'][0]}/{s['found_datos'][1]} = {s['recall_datos']:.0%} · "
           f"todos: {s['found_todos'][0]}/{s['found_todos'][1]} = {s['recall_todos']:.0%}", ""]
    for c in s["cases"]:
        f = sum(i["found"] for i in c["items"])
        out += [f"### {c['titulo']} — {f}/{len(c['items'])} ({c['n_rows']} filas)", "", "| Ref | Alcance | Desviación publicada | Fila de AlchemAI |", "|---|---|---|---|"]
        out += [f"| {i['id']} | {i['alcance']} | {i['desviacion']} | {i['row'] or '**no encontrada**'} |" for i in c["items"]]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import json
    import sys
    s = score(sys.argv[1] if len(sys.argv) > 1 else None)
    print(to_markdown(s))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(s, ensure_ascii=False, indent=1))
