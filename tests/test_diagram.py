"""Tests of the process flow diagram engine (block M27_D): packages/diagram/*, tools_diagram, wrappers_diagram.

Validation level: CONSISTENCY ONLY / analytical. A drawing has no published «known answer»; what is asserted are the
geometric identities a correct PFD must satisfy (no two equipment footprints overlap, every stream starts on its source
symbol and ends on its target symbol, every routed segment is orthogonal, no routed segment passes through a symbol, the
SVG parses as XML) plus two analytical layering cases whose answer is known by hand (a chain of n units has n layers and
zero crossings; a 2x2 «crossed» bipartite graph is untangled to zero crossings by the barycentre pass).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from packages.diagram import adapters as AD
from packages.diagram import layout as LY
from packages.diagram import render as RD
from packages.diagram.symbols import EQUIPMENT_TYPES, NOMINAL_SIZE, SYMBOLS, symbol_svg, vocabulary
from packages.dwsim_client import tools_diagram as TD


# ------------------------------------------------------------------------------------------------- synthetic models

def synthetic_refinery() -> dict:
    """41-unit refinery-like ProcessModel (synthetic: topology of a conventional high-conversion refinery; the flows,
    temperatures and pressures are illustrative drawing data, NOT data of any real refinery)."""
    S = [("crudo", "Destilación atmosférica"), ("vacio", "Destilación al vacío"), ("hdt", "Hidrotratamiento"),
         ("reformacion", "Reformación e isomerización"), ("fcc", "Craqueo catalítico (FCC)"),
         ("coquizadora", "Coquización retardada"), ("azufre", "Aminas y azufre"), ("servicios", "Servicios auxiliares"),
         ("mezclado", "Mezclado y productos")]
    U = [("TK-101", "tank", "crudo", "Tanque de crudo"), ("P-101", "pump", "crudo", "Bomba de carga"),
         ("E-101", "heat_exchanger", "crudo", "Tren de precalentamiento"), ("V-101", "vessel", "crudo", "Desaladora"),
         ("F-101", "fired_heater", "crudo", "Horno atmosférico"), ("T-101", "column", "crudo", "Torre atmosférica"),
         ("E-102", "air_cooler", "crudo", "Condensador de domo"), ("V-102", "flash_drum", "crudo", "Acumulador de domo"),
         ("F-201", "fired_heater", "vacio", "Horno de vacío"), ("T-201", "column", "vacio", "Torre de vacío"),
         ("V-201", "vessel", "vacio", "Pozo caliente"),
         ("R-301", "fixed_bed_reactor", "hdt", "HDT de naftas"), ("R-302", "fixed_bed_reactor", "hdt", "HDS de diésel"),
         ("V-301", "flash_drum", "hdt", "Separador de alta presión"), ("K-301", "compressor", "hdt", "Compresor de recirculación"),
         ("T-301", "column", "hdt", "Agotador de naftas"),
         ("F-401", "fired_heater", "reformacion", "Hornos de reformación"),
         ("R-401", "fixed_bed_reactor", "reformacion", "Reformadora"), ("R-402", "fixed_bed_reactor", "reformacion",
                                                                          "Isomerizadora"),
         ("T-401", "column", "reformacion", "Estabilizadora"),
         ("R-501", "fluidized_bed_reactor", "fcc", "Reactor FCC (riser)"),
         ("R-502", "fluidized_bed_reactor", "fcc", "Regenerador"), ("T-501", "column", "fcc", "Fraccionadora principal"),
         ("K-501", "compressor", "fcc", "Compresor de gas húmedo"), ("T-502", "column", "fcc", "Planta de gases"),
         ("F-601", "fired_heater", "coquizadora", "Horno de coquizadora"), ("R-601", "vessel", "coquizadora",
                                                                             "Tambores de coque"),
         ("T-601", "column", "coquizadora", "Fraccionadora de coquizadora"),
         ("T-701", "packed_column", "azufre", "Absorbedor de aminas"), ("T-702", "column", "azufre", "Regeneradora de aminas"),
         ("R-701", "reactor", "azufre", "Planta Claus"),
         ("B-801", "boiler", "servicios", "Caldera de vapor"), ("CT-801", "cooling_tower", "servicios",
                                                                 "Torre de enfriamiento"),
         ("FL-801", "flare", "servicios", "Quemador elevado"), ("R-802", "reactor", "servicios", "Planta de hidrógeno"),
         ("M-901", "mixer", "mezclado", "Mezcla de gasolinas"), ("M-902", "mixer", "mezclado", "Mezcla de diésel"),
         ("TK-901", "tank", "mezclado", "Gasolinas"), ("TK-902", "tank", "mezclado", "Diésel"),
         ("TK-903", "tank", "mezclado", "Combustóleo"), ("TK-904", "tank", "mezclado", "Turbosina")]
    L = [  # id, from, to, label, bpd, T_C, P_bar, kind
        ("crudo", None, "TK-101", "Crudo (mezcla Maya/Istmo)", 250000, 30, 1.0, "material"),
        ("s2", "TK-101", "P-101", "Crudo a bombeo", 250000, 30, 1.0, "material"),
        ("s3", "P-101", "E-101", "Crudo a precalentamiento", 250000, 30, 18, "material"),
        ("s4", "E-101", "V-101", "Crudo precalentado", 250000, 130, 12, "material"),
        ("agua_lavado", None, "V-101", "Agua de lavado", 12000, 120, 12, "material"),
        ("salmuera", "V-101", None, "Salmuera", 12500, 130, 11, "material"),
        ("s7", "V-101", "F-101", "Crudo desalado", 249500, 250, 10, "material"),
        ("s8", "F-101", "T-101", "Carga a torre", 249500, 365, 3, "material"),
        ("domo", "T-101", "E-102", "Vapores de domo", 90000, 120, 1.5, "material"),
        ("s10", "E-102", "V-102", "Domo condensado", 90000, 45, 1.3, "material"),
        ("reflujo", "V-102", "T-101", "Reflujo", 40000, 45, 3, "material"),
        ("nafta", "V-102", "R-301", "Nafta", 45000, 45, 30, "material"),
        ("gas_cdu", "V-102", "T-701", "Gas amargo", 5000, 45, 1.3, "material"),
        ("kero", "T-101", "TK-904", "Turbosina", 25000, 180, 2, "material"),
        ("diesel", "T-101", "R-302", "Diésel primario", 50000, 250, 2, "material"),
        ("resid_atm", "T-101", "F-201", "Residuo atmosférico", 84500, 350, 2, "material"),
        ("s17", "F-201", "T-201", "Carga a vacío", 84500, 400, 0.1, "material"),
        ("vapores_vacio", "T-201", "V-201", "Vapores a eyectores", 500, 60, 0.05, "material"),
        ("gov", "T-201", "R-501", "Gasóleo de vacío", 45000, 300, 1, "material"),
        ("resid_vac", "T-201", "F-601", "Residuo de vacío", 39000, 380, 1, "material"),
        ("agua_amarga", "V-201", None, "Agua amarga", 800, 45, 1, "material"),
        ("s22", "R-301", "V-301", "Efluente HDT", 45000, 330, 28, "material"),
        ("gas_recirc", "V-301", "K-301", "Gas de recirculación", 3000, 50, 27, "material"),
        ("h2_recirc", "K-301", "R-301", "H2 recirculado", 3000, 70, 32, "material"),
        ("s25", "V-301", "T-301", "Nafta a agotador", 44000, 50, 27, "material"),
        ("nafta_pesada", "T-301", "F-401", "Nafta pesada HDT", 30000, 150, 8, "material"),
        ("nafta_ligera", "T-301", "R-402", "Nafta ligera HDT", 14000, 60, 8, "material"),
        ("s28", "F-401", "R-401", "Carga a reactores", 30000, 520, 6, "material"),
        ("s29", "R-401", "T-401", "Efluente de reformación", 30000, 500, 5, "material"),
        ("reformado", "T-401", "M-901", "Reformado", 26000, 40, 2, "material"),
        ("glp_ccr", "T-401", "T-502", "GLP de reformación", 3000, 40, 15, "material"),
        ("isomerizado", "R-402", "M-901", "Isomerizado", 13500, 40, 2, "material"),
        ("diesel_uba", "R-302", "M-902", "Diésel UBA", 62000, 45, 3, "material"),
        ("gas_hds", "R-302", "T-701", "Gas ácido de HDS", 2000, 45, 5, "material"),
        ("h2_hds", "R-802", "R-302", "Hidrógeno a HDS", 1500, 40, 60, "material"),
        ("h2_hdt", "R-802", "R-301", "Hidrógeno a HDT", 800, 40, 35, "material"),
        ("gas_natural", None, "R-802", "Gas natural", 900, 25, 30, "material"),
        ("s38", "R-501", "T-501", "Efluente del reactor", 45000, 520, 2.4, "material"),
        ("cat_gastado", "R-501", "R-502", "Catalizador gastado", 0, 520, 2.5, "material"),
        ("cat_regen", "R-502", "R-501", "Catalizador regenerado", 0, 700, 2.6, "material"),
        ("gases_comb", "R-502", None, "Gases de combustión", 0, 700, 1.1, "material"),
        ("gas_humedo", "T-501", "K-501", "Gas húmedo", 9000, 40, 1.5, "material"),
        ("s43", "K-501", "T-502", "Gas comprimido", 9000, 90, 15, "material"),
        ("gasolina_fcc", "T-502", "M-901", "Gasolina FCC", 24000, 40, 10, "material"),
        ("glp", "T-502", None, "GLP a ventas", 7000, 40, 15, "material"),
        ("aac", "T-501", "M-902", "Aceite cíclico ligero", 8000, 200, 2, "material"),
        ("aceite_dec", "T-501", "TK-903", "Aceite decantado", 4000, 340, 2, "material"),
        ("s48", "F-601", "R-601", "Carga a tambores", 39000, 495, 4, "material"),
        ("vapores_coque", "R-601", "T-601", "Vapores de coque", 30000, 440, 1.5, "material"),
        ("coque", "R-601", None, "Coque", 9000, 50, 1, "material"),
        ("gok", "T-601", "R-302", "Gasóleo de coquizadora", 14000, 300, 2, "material"),
        ("gas_coque", "T-601", "T-701", "Gas de coquizadora", 4000, 40, 1.5, "material"),
        ("amina_rica", "T-701", "T-702", "Amina rica", 6000, 55, 6, "material"),
        ("amina_pobre", "T-702", "T-701", "Amina pobre", 6000, 45, 7, "material"),
        ("gas_acido", "T-702", "R-701", "Gas ácido", 900, 50, 1.8, "material"),
        ("azufre", "R-701", None, "Azufre", 600, 140, 1, "material"),
        ("gas_comb", "T-701", "B-801", "Gas combustible dulce", 10000, 40, 5, "utility"),
        ("agua_caldera", None, "B-801", "Agua de alimentación", 20000, 105, 50, "utility"),
        ("vapor", "B-801", "T-201", "Vapor de agotamiento", 2000, 300, 10, "utility"),
        ("agua_enf", "CT-801", "V-201", "Agua de enfriamiento", 60000, 30, 4, "utility"),
        ("desfogues", "T-502", "FL-801", "Desfogues", 100, 40, 1.2, "material"),
        ("potencia", None, "K-501", "Potencia eléctrica", None, None, None, "energy"),
        ("gasolina", "M-901", "TK-901", "Gasolina", 63500, 40, 3, "material"),
        ("diesel_mez", "M-902", "TK-902", "Diésel", 70000, 45, 3, "material"),
        ("venta_gasolina", "TK-901", None, "Gasolinas a ventas", 63500, 30, 1, "material"),
        ("venta_diesel", "TK-902", None, "Diésel a ventas", 70000, 30, 1, "material"),
        ("venta_combustoleo", "TK-903", None, "Combustóleo a ventas", 4000, 60, 1, "material"),
        ("venta_turbosina", "TK-904", None, "Turbosina a ventas", 25000, 30, 1, "material"),
    ]
    streams = []
    for sid, fr, to, lab, bpd, T, P, kind in L:
        s = {"id": sid, "label_es": lab, "from": fr, "to": to, "kind": kind}
        if bpd:
            s["flow_bpd"] = float(bpd)
        if T is not None:
            s["T_K"] = T + 273.15
        if P is not None:
            s["P_Pa"] = P * 1e5
        streams.append(s)
    units = [{"id": i, "type": t, "section": s, "label_es": lab, "params": {}, "duties_W": {}, "metrics": {}}
             for i, t, s, lab in U]
    units[4]["duties_W"] = {"fired": 95e6}
    return {"name": "Refinería sintética de prueba (41 equipos)", "kind": "refinery", "units": units, "streams": streams,
            "sections": [{"id": a, "label_es": b} for a, b in S],
            "source": "modelo sintético de prueba (topología típica; números ilustrativos)"}


def refinery_opportunities() -> list[dict]:
    def o(i, units, streams, cat, title, v, lo, hi, conf, rank):
        return {"id": i, "unit_ids": units, "stream_ids": streams, "category": cat, "title_es": title,
                "explanation_es": "", "value_usd_y": v, "value_low_usd_y": lo, "value_high_usd_y": hi,
                "basis_es": "valor ilustrativo de prueba", "assumptions": [], "evidence": {}, "confidence": conf,
                "rank": rank}
    return [o("op1", ["E-101"], [], "energy", "Limpiar el tren de precalentamiento (ensuciamiento)", 6.2e6, 3.1e6, 9.5e6,
              "media", 1),
            o("op2", ["FL-801"], ["desfogues"], "emissions", "Recuperar gas de quemador", 1.4e6, 0.7e6, 2.0e6, "baja", 3),
            o("op3", ["R-501"], [], "yield", "Subir conversión del FCC con catalizador", 9.0e6, 4.0e6, 14e6, "media", 2),
            o("op4", ["B-801", "CT-801"], [], "utilities", "Purgas de caldera y trampas de vapor", 0.8e6, 0.4e6, 1.2e6,
              "alta", 4),
            o("op5", [], ["h2_recirc"], "economics", "Pureza de H2 de recirculación", None, None, None, "baja", 5)]


# ------------------------------------------------------------------------------------------------------ geometry

def _boxes_overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def assert_geometry(lay: dict) -> None:
    nodes = {k: v for k, v in lay["nodes"].items() if v["kind"] != "dummy"}
    ids = list(nodes)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = nodes[ids[i]]["foot_box"], nodes[ids[j]]["foot_box"]
            assert not _boxes_overlap(a, b), (ids[i], ids[j], a, b)
    for e in lay["edges"]:
        pts = e["points"]
        assert len(pts) >= 2
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            assert abs(x0 - x1) < 1e-6 or abs(y0 - y1) < 1e-6, (e["id"], pts)     # orthogonal
        for end, nid in ((pts[0], e["src"]), (pts[-1], e["dst"])):
            x, y, w, h = nodes[nid]["sym_box"]
            assert x - 1.0 <= end[0] <= x + w + 1.0 and y - 1.0 <= end[1] <= y + h + 1.0, (e["id"], nid, end)
        # interior grid points never inside a symbol (except the two end symbols)
        for (x, y) in pts[1:-1]:
            for nid, nd in nodes.items():
                if nid in (e["src"], e["dst"]):
                    continue
                bx, by, bw, bh = nd["sym_box"]
                assert not (bx + 0.5 < x < bx + bw - 0.5 and by + 0.5 < y < by + bh - 0.5), (e["id"], nid)


def _chain(types: list[str]) -> dict:
    units = [{"id": f"U{i + 1}", "type": t, "section": "s", "label_es": t} for i, t in enumerate(types)]
    streams = [{"id": "f", "from": None, "to": "U1", "flow_kg_h": 100.0, "T_K": 300.0, "P_Pa": 1e5}]
    streams += [{"id": f"s{i + 1}", "from": f"U{i + 1}", "to": f"U{i + 2}"} for i in range(len(types) - 1)]
    streams += [{"id": "p", "from": f"U{len(types)}", "to": None}]
    return {"name": "cadena", "units": units, "streams": streams, "sections": [{"id": "s", "label_es": "S"}]}


# ------------------------------------------------------------------------------------------------------- symbols

def test_vocabulary_is_closed_superset_of_contract():
    contract = ("pump, compressor, fan, heat_exchanger, air_cooler, heater, cooler, fired_heater, boiler, cooling_tower, "
                "vessel, flash_drum, separator_3phase, decanter, tank, column, packed_column, reactor, cstr, pfr, "
                "fixed_bed_reactor, fluidized_bed_reactor, fermenter, mixer, splitter, valve, filter, centrifuge, "
                "screw_press, mill, conveyor, dryer, evaporator, crystallizer, cyclone, membrane, adsorber, flare, stack, "
                "generic").split(", ")
    assert EQUIPMENT_TYPES[:len(contract)] == contract          # never removed nor reordered
    assert set(SYMBOLS) == set(EQUIPMENT_TYPES) == set(NOMINAL_SIZE)
    assert len(vocabulary()) == len(EQUIPMENT_TYPES)
    json.dumps(TD.diagram_symbols())


@pytest.mark.parametrize("t", EQUIPMENT_TYPES)
def test_every_symbol_renders_as_valid_svg(t):
    frag = symbol_svg(t, 10, 10, 60, 80)
    root = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{frag}</svg>')
    assert len(list(root.iter())) > 2


def test_unknown_symbol_type_refused():
    with pytest.raises(ValueError, match="EQUIPMENT_TYPES"):
        symbol_svg("teleporter", 0, 0, 10, 10)


def test_every_type_in_one_diagram():
    lay = LY.layout(_chain(EQUIPMENT_TYPES), "full")
    assert_geometry(lay)
    out = RD.render(lay)
    ET.fromstring(out["svg"])
    assert {n["type"] for n in out["scene"]["nodes"] if n["kind"] == "unit"} == set(EQUIPMENT_TYPES)


# ------------------------------------------------------------------------------------------- analytical layering

def test_chain_has_one_layer_per_unit_and_straight_streams():
    """Analytical: a chain of n identical units -> n layers left to right, all on one line, every stream a straight
    2-point segment, zero crossings."""
    n = 6
    lay = LY.layout(_chain(["pump"] * n), "full")
    xs = [lay["nodes"][f"U{i + 1}"]["sym_box"][0] for i in range(n)]
    ys = {lay["nodes"][f"U{i + 1}"]["sym_box"][1] for i in range(n)}
    assert xs == sorted(xs) and len(set(xs)) == n and len(ys) == 1
    for e in lay["edges"]:
        assert len(e["points"]) == 2, e


def test_barycentre_untangles_crossed_bipartite():
    """Analytical: a1->b2, a2->b1 with b1 listed first has 1 crossing in model order; the barycentre pass must reach 0,
    i.e. put b2 above b1."""
    nodes = {k: {"id": k, "kind": "unit", "type": "vessel", "group": "g", "label_es": k, "note": None, "unit_ids": [k]}
             for k in ("a1", "a2", "b1", "b2")}
    for nd in nodes.values():
        LY._size(nd)
    pos = LY._layout_group(nodes, ["a1", "a2", "b1", "b2"], [("a1", "b2"), ("a2", "b1")])
    assert pos["a1"][1] < pos["a2"][1]
    assert pos["b2"][1] < pos["b1"][1]


def test_recycle_detected_and_routed():
    pm = _chain(["pump", "heat_exchanger", "flash_drum"])
    pm["streams"].append({"id": "rec", "from": "U3", "to": "U1", "label_es": "Recirculación"})
    lay = LY.layout(pm, "full")
    rec = [e for e in lay["edges"] if e["id"] == "rec"][0]
    assert rec["recycle"] and sum(e["recycle"] for e in lay["edges"]) == 1
    assert_geometry(lay)


# ---------------------------------------------------------------------------------------- refinery (40+ units)

@pytest.mark.parametrize("view", ["full", "sections", "section:fcc", "section:crudo"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_refinery_views(view, theme):
    r = TD.process_diagram(process_model=synthetic_refinery(), opportunities=refinery_opportunities(), view=view,
                           theme=theme, today="2026-11-16")
    assert r["evidence"]["verified"], r["evidence"]
    root = ET.fromstring(r["svg"])
    assert root.tag.endswith("svg") and "<script" not in r["svg"] and "@import" not in r["svg"]
    assert "Diagrama generado; no es un P&amp;ID" in r["svg"]
    from packages.diagram.render import PALETTES
    assert PALETTES[theme]["bg"] in r["svg"]
    json.dumps(r)
    if view == "full":
        assert r["n_units"] == 41
        drawn = {s for e in r["scene"]["edges"] for s in e["stream_ids"]}
        assert drawn == {s["id"] for s in synthetic_refinery()["streams"]}      # every stream drawn


def test_refinery_full_geometry_both_modes():
    for link in (False, True):
        lay = LY.layout(synthetic_refinery(), "full", link_sections=link)
        assert_geometry(lay)


def test_markers_land_on_the_right_unit():
    r = TD.process_diagram(process_model=synthetic_refinery(), opportunities=refinery_opportunities(), view="full")
    sc = r["scene"]
    nodes = {n["id"]: n for n in sc["nodes"]}
    m = [x for x in sc["markers"] if x["opportunity_id"] == "op1"]
    assert len(m) == 1 and m[0]["target"] == "E-101" and m[0]["n"] == 1        # rank 1
    x, y, w, h = nodes["E-101"]["sym_box"]
    assert abs(m[0]["pos"][0] - (x + w)) < 15 and abs(m[0]["pos"][1] - y) < 15
    assert {x["target"] for x in sc["markers"] if x["opportunity_id"] == "op4"} == {"B-801", "CT-801"}
    assert any(x["target"] == "h2_recirc" for x in sc["markers"] if x["opportunity_id"] == "op5")
    assert [o["n"] for o in sc["opportunities"]] == [1, 2, 3, 4, 5]
    assert [o["id"] for o in sc["opportunities"]] == ["op1", "op3", "op2", "op4", "op5"]
    # collapsed view: marker on the section block, one badge per block even with two units in it
    s = TD.process_diagram(process_model=synthetic_refinery(), opportunities=refinery_opportunities(), view="sections")
    t4 = [x["target"] for x in s["scene"]["markers"] if x["opportunity_id"] == "op4"]
    assert t4 == ["__sec__servicios"]
    assert not [o for o in s["scene"]["opportunities"] if o["id"] == "op5"][0]["drawn"]      # internal stream


def test_opportunity_with_unknown_ids_warns():
    opp = refinery_opportunities()[:1]
    opp[0]["unit_ids"] = ["E-101", "NO-EXISTE"]
    r = TD.process_diagram(process_model=synthetic_refinery(), opportunities=opp, view="full")
    assert any("NO-EXISTE" in w for w in r["warnings"])


def test_path_writes_file(tmp_path):
    p = tmp_path / "pfd.svg"
    r = TD.process_diagram(process_model=synthetic_refinery(), view="sections", path=str(p))
    assert "svg" not in r and r["path"] == str(p)
    ET.parse(p)


# ------------------------------------------------------------------------------------------------------ adapters

def test_description_adapter_hazop_shape():
    desc = {"name": "Agotador", "units": [
        {"name": "B-1", "type": "bomba", "inlets": ["alim"], "outlets": ["s1"]},
        {"name": "E-1", "type": "HeatExchanger", "inlets": ["s1"], "outlets": ["s2"], "duty_kW": 120.0},
        {"name": "C-1", "type": "DistillationColumn", "inlets": ["s2"], "outlets": ["vapor_domo", "fondos"]},
        {"name": "R-1", "type": "RCT_CSTR", "inlets": ["fondos"], "outlets": ["prod"]}],
        "streams": [{"name": "alim", "T_K": 300.0, "P_Pa": 2e5, "composition": {"Water": 1.0}, "mass_flow_kg_s": 1.0}]}
    r = TD.process_diagram(description=desc, view="full")
    assert r["evidence"]["verified"]
    pm = AD.from_description(desc["units"], desc["streams"])
    types = {u["id"]: u["type"] for u in pm["units"]}
    assert types == {"B-1": "pump", "E-1": "heat_exchanger", "C-1": "column", "R-1": "cstr"}
    st = {s["id"]: s for s in pm["streams"]}
    assert st["alim"]["flow_kg_h"] == pytest.approx(3600.0) and st["alim"]["from"] is None
    assert "T_K" not in st["s1"] and "flow_kg_h" not in st["s1"]                 # nothing invented
    assert any("sin datos" in w for w in pm["warnings"])
    lay = LY.layout(pm, "full")
    vap = [e for e in lay["edges"] if e["id"] == "vapor_domo"][0]
    assert vap["d0"] == (0, -1)                                                   # overhead leaves from the top
    with pytest.raises(ValueError, match="dos equipos"):
        AD.from_description([{"name": "a", "type": "pump", "outlets": ["x"]}, {"name": "b", "type": "pump", "outlets": ["x"]}])


def test_dwsim_adapter_bridges_recycle_and_reads_results():
    desc = {"flowsheet_id": "fs1", "streams": ["feed", "mix_out", "h_out", "v", "l", "rec_in", "rec_out"],
            "energy": ["q1"], "stream_specs": {"feed": {"temperature_K": 300.0, "pressure_Pa": 1e5, "mass_flow_kg_s": 2.0}},
            "units": [{"name": "MIX", "kind": "mixer", "feeds": ["feed", "rec_out"], "products": ["mix_out"]},
                      {"name": "HTR", "kind": "heater", "feeds": ["mix_out"], "products": ["h_out"], "energy_feeds": ["q1"]},
                      {"name": "FL", "kind": "separator", "feeds": ["h_out"], "products": ["v", "l"]},
                      {"name": "SPL", "kind": "splitter", "feeds": ["l"], "products": ["rec_in", "prod"]},
                      {"name": "REC", "kind": "recycle", "feeds": ["rec_in"], "products": ["rec_out"]}]}
    res = {"v": {"T_K": 350.0, "P_Pa": 1e5, "mass_flow_kg_s": 0.5, "vapor_fraction": 1.0, "mole_fractions": {"A": 1.0}}}
    pm = AD.from_dwsim_description(desc, res)
    st = {s["id"]: s for s in pm["streams"]}
    assert "REC" not in {u["id"] for u in pm["units"]}
    assert st["rec_in"]["from"] == "SPL" and st["rec_in"]["to"] == "MIX" and "rec_out" not in st
    assert st["v"]["flow_kg_h"] == pytest.approx(1800.0) and st["feed"]["flow_kg_h"] == pytest.approx(7200.0)
    assert st["q1"]["kind"] == "energy" and st["q1"]["to"] == "HTR"
    assert {u["id"]: u["type"] for u in pm["units"]}["FL"] == "flash_drum"
    lay = LY.layout(pm, "full")
    assert_geometry(lay)
    assert [e for e in lay["edges"] if e["id"] == "rec_in"][0]["recycle"]


@pytest.fixture(scope="module")
def castor():
    from packages.plants import biodiesel as BD
    return BD.biodiesel_plant(oil_feed_kg_h=1500.0, oil="castor", ffa_mass_fraction=0.002, water_mass_fraction=0.0,
                              operating_hours_y=7920.0)


def test_castor_biodiesel_plant_adapter(castor):
    pm = AD.from_plant_result(castor)
    ids = {u["id"] for s in castor["sections"] for u in [{"id": x} for x in s["unit_ids"]]}
    assert {u["id"] for u in pm["units"]} == ids
    types = {u["id"]: u["type"] for u in pm["units"]}
    assert types["T403"] == "tank" and types["P403"] == "pump" and types["R401"] == "cstr"
    assert types["C401"] == "centrifuge" and types["D401"] == "column" and types["F401"] == "dryer"
    # plant results now carry internal_streams (integrator, M27): the full view is connected, no "missing" warning
    assert castor.get("internal_streams") and not any("corrientes internas" in w for w in pm["warnings"])
    opps = [{"id": "x", "unit_ids": ["D401"], "stream_ids": [], "category": "energy", "title_es": "t",
             "value_usd_y": 1.0, "confidence": "baja", "rank": 1}]
    for view in ("sections", "full"):
        r = TD.process_diagram(plant_result=castor, opportunities=opps, view=view)
        assert r["evidence"]["verified"], r["evidence"]
        ET.fromstring(r["svg"])
        json.dumps(r)
    assert r["scene"]["markers"][0]["target"] == "D401"


def test_castor_with_internal_topology_from_live_system():
    """biosteam_streams on a live BioSTEAM system gives the unit-to-unit streams; drawn connected and verified."""
    from packages.plants import biodiesel as BD
    from packages.plants import reference as R
    orig = R.extract_results

    def wrapped(bst, system, *a, **k):
        out = orig(bst, system, *a, **k)
        out["internal_streams"] = AD.biosteam_streams(system)
        return out
    R.extract_results = wrapped
    try:
        res = BD.biodiesel_plant(oil_feed_kg_h=1500.0, oil="castor", ffa_mass_fraction=0.002, water_mass_fraction=0.0,
                                 operating_hours_y=7920.0)
    finally:
        R.extract_results = orig
    assert len(res["internal_streams"]) > 30
    pm = AD.from_plant_result(res)
    assert not pm["warnings"]
    lay = LY.layout(pm, "full")
    assert_geometry(lay)


# ------------------------------------------------------------------------------------------------------ refusals

def test_refusals():
    with pytest.raises(ValueError, match="UNA fuente"):
        TD.process_diagram()
    with pytest.raises(ValueError, match="UNA fuente"):
        TD.process_diagram(process_model=synthetic_refinery(), description={"units": []})
    with pytest.raises(ValueError, match="view"):
        TD.process_diagram(process_model=synthetic_refinery(), view="isometric")
    with pytest.raises(ValueError, match="no existe"):
        TD.process_diagram(process_model=synthetic_refinery(), view="section:luna")
    with pytest.raises(ValueError, match="theme"):
        TD.process_diagram(process_model=synthetic_refinery(), theme="sepia")
    bad = synthetic_refinery()
    bad["streams"][0]["to"] = "NADA"
    with pytest.raises(ValueError, match="NADA"):
        TD.process_diagram(process_model=bad)
    dup = synthetic_refinery()
    dup["units"].append(dict(dup["units"][0]))
    with pytest.raises(ValueError, match="repetido"):
        TD.process_diagram(process_model=dup)
    with pytest.raises(ValueError, match="units"):
        TD.process_diagram(process_model={"name": "x", "units": []})


def test_unknown_type_drawn_generic_with_warning():
    pm = _chain(["pump", "warp_core"])
    r = TD.process_diagram(process_model=pm, view="full")
    assert any("warp_core" in w for w in r["warnings"])
    assert {n["id"]: n["type"] for n in r["scene"]["nodes"]}["U2"] == "generic"


def test_wrappers_match_tools():
    from packages.agent import wrappers_diagram as W
    assert [w.name for w in W.AGENT_WRAPPERS] == [f.__name__ for f in TD.DIAGRAM_TOOLS]
    out = json.loads(W.diagram_symbols(**{}) if callable(W.diagram_symbols) else "{}")
    assert out["n_types"] == len(EQUIPMENT_TYPES)
