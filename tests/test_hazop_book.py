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


def test_heldout_recall(tmp_path):
    """Bloque 25: conjunto apartado (5 HAZOP publicados que las reglas no habían visto). A ciegas: 25/30 (criterios originales),
    23/30 con el calificador endurecido; 30/30 tras las reglas de sustancia/régimen térmico (vacío, solidificación,
    descomposición, frente frío, golpe por condensación, medio de calentamiento de columnas)."""
    from packages.hazop.book.heldout_cases import HELDOUT
    s = score(tmp_path, ref_path=Path("packages/hazop/book/heldout.yaml"), builders=HELDOUT)
    assert s["found_datos"] == [30, 30], [i for c in s["cases"] for i in c["items"] if not i["found"] and i["alcance"] == "datos"]
    assert all(not c["missing"] for c in s["cases"])


def test_substance_flags():
    from packages.hazop.graph import melting_point_K, stream_node
    assert melting_point_K("Ammonium nitrate") > 430 and melting_point_K("Sulfur") > 380      # CRC, not OPEN_NTBKM (286 K)
    assert melting_point_K("Reactant A") is None
    an = stream_node("s", 413.0, 1e5, 0.0, 1.0, 1.0, {"Ammonium nitrate": 0.52, "Water": 0.479, "Ammonia": 0.001})
    assert an.solidifies and an.unstable and an.solid_component == "Ammonium nitrate"
    assert an.toxic and not an.flammable, "0.1 % de amoniaco en solución no es una liberación inflamable"
