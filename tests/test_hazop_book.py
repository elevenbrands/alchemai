"""HAZOP preliminar contra HAZOP publicados (bloque 17): 5 procesos como DEXPI hechos a mano, sin simulador."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.hazop.book.score import find, score  # noqa: E402


def test_book_recall(tmp_path):
    s = score(tmp_path)
    assert s["found_datos"] == [22, 22], [i for c in s["cases"] for i in c["items"] if not i["found"] and i["alcance"] == "datos"]
    fuera = [i["id"] for c in s["cases"] for i in c["items"] if i["alcance"] == "fuera"]
    assert sorted(fuera) == ["B4", "C5", "L10", "L9"]
    assert all(not c["missing"] for c in s["cases"]), "los 5 DEXPI traen T, P, flujo y composición"


def test_matcher_is_strict():
    item = {"match": {"nodes": ["T-1"], "dev": "less temperature", "concept": "congel"}}
    row = {"node": "T-1", "parameter": "Temperature", "deviation": "Less temperature (winter)", "causes": [], "consequences": ["Congelamiento del agua"], "safeguards": []}
    assert find(item, [row])
    assert not find(item, [{**row, "node": "T-2"}])
    assert not find({"match": {**item["match"], "concept": r"\bagua\b.{0,60}aguas abajo"}}, [{**row, "consequences": ["presión aguas abajo; PSV aguas abajo"]}])
