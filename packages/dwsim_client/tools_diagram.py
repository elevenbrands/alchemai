"""Agent tools of the process flow diagram engine (bloque M27_D): draw any ProcessModel (refinery of block R, DWSIM
flowsheet, BioSTEAM plant result, hand description) as a standalone SVG PFD + a JSON scene for the Mac app, with the
opportunities of block O marked on it. Engines: `packages.diagram.layout`, `packages.diagram.render`,
`packages.diagram.adapters`, `packages.diagram.symbols`."""
from __future__ import annotations

import time


def process_diagram(process_model: dict | None = None, flowsheet_id: str | None = None, plant_result: dict | None = None,
                    description: dict | None = None, opportunities: list | None = None, view: str = "sections",
                    theme: str = "light", path: str | None = None, table_streams: list | None = None,
                    today: str | None = None) -> dict:
    """Diagrama de flujo de proceso (PFD) profesional en SVG con símbolos de equipo según la convención ISO 10628-2 /
    ISA (bomba, compresor, intercambiador, horno, columna de platos, reactor de lecho fijo/fluidizado, tanque…), corrientes
    ortogonales con flechas y número de corriente, tabla de corrientes (flujo, T en °C, P en bar), leyenda, cuadro de
    título («diagrama generado; no es un P&ID») y MARCADORES de oportunidades numerados (color por categoría) con su lista
    de título y USD/año. Dé UNA fuente: process_model (ProcessModel JSON: {name, kind, units [{id, label_es, type,
    section, params, duties_W, metrics}], streams [{id, label_es, from, to, kind material|energy|utility, flow_kg_h |
    flow_m3_d | flow_bpd, T_K, P_Pa}], sections [{id, label_es}], source}; `type` del vocabulario de diagram_symbols),
    flowsheet_id (flowsheet de DWSIM abierto en esta sesión), plant_result (el dict que devolvió reference_biorefinery o
    biodiesel_plant) o description ({units: [{name, type, inlets, outlets, section?, duty_kW?}], streams: [{name, T_K?,
    P_Pa?, mass_flow_kg_s?}], name?}, la misma forma que hazop_from_process_description). opportunities: lista de
    oportunidades {id, unit_ids, stream_ids, category energy|yield|utilities|reliability|emissions|economics, title_es,
    value_usd_y, value_low_usd_y, value_high_usd_y, confidence, rank} que se marcan sobre el equipo o la corriente.
    view: "sections" (cada sección colapsada en un bloque; la vista para plantas grandes), "full" (todos los equipos
    agrupados por sección) o "section:<id>" (una sección; las corrientes que cruzan su frontera se dibujan como conectores
    que nombran la otra sección). theme "light" | "dark". path: si se da, escribe ahí el .svg y devuelve la ruta en vez del
    texto. table_streams: id de las corrientes de la tabla (None = todas las dibujadas).
    Devuelve svg (texto) o path, scene (JSON con nodos, cajas, polilíneas de corrientes, marcadores, tabla y primitivas de
    los símbolos para dibujar nativo), legend, n_units, n_streams, warnings, assumptions, not_included, method y evidence
    (verified = ningún equipo se traslapa, cada corriente empieza y termina en su equipo, todos los tramos ortogonales y
    el SVG es XML válido)."""
    from packages.diagram import adapters as AD
    from packages.diagram import layout as LY
    from packages.diagram import render as RD
    t0 = time.time()
    given = [k for k, v in (("process_model", process_model), ("flowsheet_id", flowsheet_id),
                            ("plant_result", plant_result), ("description", description)) if v]
    if len(given) != 1:
        raise ValueError("Dé exactamente UNA fuente del diagrama: process_model, flowsheet_id, plant_result o description "
                         f"(recibí {given or 'ninguna'}).")
    if process_model:
        pm = process_model
    elif flowsheet_id:
        pm = AD.dwsim_process_model(flowsheet_id)
    elif plant_result:
        pm = AD.from_plant_result(plant_result)
    else:
        if not isinstance(description, dict):
            raise ValueError("description debe ser {units: [...], streams: [...], name?}.")
        pm = AD.from_description(description.get("units") or [], description.get("streams") or [],
                                 description.get("name"), description.get("sections"))
    lay = LY.layout(pm, view)
    out = RD.render(lay, opportunities, theme, today=today, table_streams=table_streams)
    ev = check_geometry(lay, out["svg"])
    warnings = list(pm.get("warnings") or []) + out["warnings"]
    res = {"n_units": sum(1 for n in lay["nodes"].values() if n["kind"] == "unit") or len(lay["model"]["units"]),
           "n_units_model": len(lay["model"]["units"]), "n_streams": len(lay["edges"]),
           "n_streams_model": len(lay["model"]["streams"]), "view": view, "theme": theme,
           "scene": out["scene"], "legend": out["legend"], "warnings": warnings,
           "assumptions": ["Los símbolos siguen la CONVENCIÓN de ISO 10628-2 / ISA para PFD con dibujos propios; no son una "
                           "biblioteca de símbolos certificada.",
                           "La posición de cada equipo la decide el algoritmo de capas (izquierda a derecha por flujo), no "
                           "la planta real (no es un plano de arreglo).",
                           "Las salidas de vapor/gas de columnas, tanques flash y hornos se dibujan por arriba si su nombre "
                           "o fase lo indica; las demás salidas por la derecha."],
           "not_included": ["No es un P&ID: sin instrumentación, lazos de control, válvulas de seguridad, diámetros de línea "
                            "ni especificación de tuberías.",
                            "No calcula nada del proceso: dibuja los números que trae el modelo."],
           "method": "Diagrama por capas tipo Sugiyama (ruptura de ciclos por DFS, capas por camino más largo, nodos "
                     "ficticios en aristas largas, reducción de cruces por baricentro) por sección, secciones en columnas "
                     "por su propio orden de flujo; enrutado ortogonal A* sobre rejilla de 10 px con penalización de "
                     "quiebres, traslapes y cruces; SVG autónomo (sin scripts ni fuentes externas).",
           "evidence": ev, "seconds": round(time.time() - t0, 3)}
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out["svg"])
        res["path"] = path
    else:
        res["svg"] = out["svg"]
    return res


def check_geometry(lay: dict, svg: str) -> dict:
    """Evidence of the drawing: overlaps, stream endpoints on their units, orthogonality, routing through symbols, XML."""
    import xml.etree.ElementTree as ET
    nodes = {k: v for k, v in lay["nodes"].items() if v["kind"] != "dummy"}
    ids = list(nodes)
    overlaps = []
    for i in range(len(ids)):
        a = nodes[ids[i]]["foot_box"]
        for j in range(i + 1, len(ids)):
            b = nodes[ids[j]]["foot_box"]
            if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                overlaps.append([ids[i], ids[j]])
    bad_ends, non_orth, through = [], [], []
    for e in lay["edges"]:
        pts = e["points"]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if abs(x0 - x1) > 1e-6 and abs(y0 - y1) > 1e-6:
                non_orth.append(e["id"])
                break
        for end, nid in ((pts[0], e["src"]), (pts[-1], e["dst"])):
            x, y, w, h = nodes[nid]["sym_box"]
            if not (x - 1 <= end[0] <= x + w + 1 and y - 1 <= end[1] <= y + h + 1):
                bad_ends.append(e["id"])
        for (px, py) in pts[1:-1]:
            for nid, nd in nodes.items():
                if nid in (e["src"], e["dst"]):
                    continue
                bx, by, bw, bh = nd["sym_box"]
                if bx + 0.5 < px < bx + bw - 0.5 and by + 0.5 < py < by + bh - 0.5:
                    through.append([e["id"], nid])
    try:
        ET.fromstring(svg)
        xml_ok = True
    except ET.ParseError:
        xml_ok = False
    verified = not overlaps and not bad_ends and not non_orth and not through and xml_ok
    ev = {"verified": verified, "overlapping_units": overlaps, "streams_off_their_units": sorted(set(bad_ends)),
          "non_orthogonal_streams": non_orth, "streams_through_symbols": through[:20], "svg_parses": xml_ok}
    if not verified:
        ev["reasons"] = [r for r, bad in (("equipos traslapados", overlaps), ("corrientes fuera de su equipo", bad_ends),
                                          ("tramos no ortogonales", non_orth), ("corrientes que cruzan un símbolo", through),
                                          ("SVG no es XML válido", not xml_ok)) if bad]
    return ev


def diagram_symbols() -> dict:
    """Vocabulario CERRADO de tipos de equipo del diagrama (lo que acepta `type` en units de un ProcessModel): para cada
    tipo su nombre en español, el tamaño nominal del símbolo en px y su caja de vista; además el mapa de tipos de DWSIM y
    de las descripciones a mano a este vocabulario. Úselo antes de armar un ProcessModel a mano."""
    from packages.diagram import adapters as AD
    from packages.diagram.symbols import EQUIPMENT_TYPES, vocabulary
    return {"n_types": len(EQUIPMENT_TYPES), "types": vocabulary(), "dwsim_type_map": dict(AD.DWSIM_TYPE_MAP),
            "description_aliases": {k: v for k, v in AD.DESCRIPTION_TYPE_MAP.items() if k not in EQUIPMENT_TYPES},
            "views": ["sections", "full", "section:<id>"], "themes": ["light", "dark"],
            "opportunity_categories": ["energy", "yield", "utilities", "reliability", "emissions", "economics"]}


DIAGRAM_TOOLS = [process_diagram, diagram_symbols]
