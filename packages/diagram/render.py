"""Standalone SVG and JSON scene of a laid-out process flow diagram (block M27_D).

The SVG has no scripts and no external fonts (system sans-serif stack), so it opens anywhere. It carries: section frames
(dash-dot area limits), equipment symbols with tag and Spanish label, orthogonal streams with arrowheads and line hops at
crossings, stream numbers in diamonds (the PFD convention) linked to a stream table (flow, T in °C, P in bar), a legend
of the line types, opportunity categories and the symbols used, a title block («diagrama generado; no es un P&ID») and
numbered OPPORTUNITY markers (colour = category) with a side list of title and USD/año. Light and dark palettes.
"""

from __future__ import annotations

import datetime as _dt
from xml.sax.saxutils import escape

from packages.diagram.layout import _wrap
from packages.diagram.symbols import EQUIPMENT_TYPES, LABELS_ES, SYMBOLS, symbol_svg

FONT = "-apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"

PALETTES: dict[str, dict[str, str]] = {
    "light": {"bg": "#ffffff", "ink": "#1b1f24", "muted": "#5b6470", "faint": "#8a929c", "lane": "#f5f7fa",
              "frame": "#9aa3ad", "energy": "#c2410c", "utility": "#1d4ed8", "panel": "#f8fafc", "rule": "#d6dbe1",
              "header": "#0f172a", "badge_text": "#ffffff", "block": "#eef2f7"},
    "dark": {"bg": "#0f1216", "ink": "#e6e9ed", "muted": "#a3acb7", "faint": "#7d8794", "lane": "#161b22",
             "frame": "#4b5563", "energy": "#fb923c", "utility": "#60a5fa", "panel": "#151a21", "rule": "#2d3540",
             "header": "#f1f5f9", "badge_text": "#0f1216", "block": "#1b222c"},
}
CATEGORY_COLORS: dict[str, dict[str, str]] = {
    "light": {"energy": "#d97706", "yield": "#15803d", "utilities": "#2563eb", "reliability": "#7c3aed",
              "emissions": "#0f766e", "economics": "#be185d", "other": "#6b7280"},
    "dark": {"energy": "#fbbf24", "yield": "#4ade80", "utilities": "#60a5fa", "reliability": "#c4b5fd",
             "emissions": "#2dd4bf", "economics": "#f472b6", "other": "#9ca3af"},
}
CATEGORY_ES = {"energy": "Energía", "yield": "Rendimiento", "utilities": "Servicios auxiliares",
               "reliability": "Confiabilidad", "emissions": "Emisiones", "economics": "Economía", "other": "Otra"}
KIND_ES = {"material": "Corriente de proceso", "energy": "Corriente de energía", "utility": "Servicio auxiliar"}

HEADER_H = 64
PANEL_W = 400
ROW_H = 16
TABLE_COLS = [("N°", 34), ("Corriente", 200), ("De → A", 220), ("Flujo", 118), ("T (°C)", 62), ("P (bar)", 64)]
TABLE_W = sum(w for _, w in TABLE_COLS)
MAX_TABLE_ROWS = 160


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _t(x: float, y: float, s: str, size: float = 10, fill: str = "#000", anchor: str = "start", weight: str = "normal",
       style: str = "") -> str:
    extra = f' font-style="{style}"' if style else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{extra}>{escape(str(s))}</text>')


def _num(x: float, digits: int = 3) -> str:
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a == 0:
        return "0"
    return f"{x:.{digits - 1}g}" if a < 1 else f"{x:.2f}"


def flow_text(s: dict) -> str:
    for key, unit in (("flow_kg_h", "kg/h"), ("flow_bpd", "bbl/d"), ("flow_m3_d", "m³/d"), ("flow_m3_h", "m³/h"),
                      ("flow_W", "W"), ("duty_W", "W"), ("flow_kmol_h", "kmol/h")):
        v = s.get(key)
        if isinstance(v, (int, float)):
            if unit == "W":
                return f"{_num(v / 1e6)} MW" if abs(v) >= 1e6 else f"{_num(v / 1e3)} kW"
            if unit == "kg/h" and abs(v) >= 1e5:
                return f"{_num(v / 1e3)} t/h"
            return f"{_num(v)} {unit}"
    return "—"


def money(v: float | None) -> str:
    if v is None:
        return "sin cuantificar"
    a = abs(v)
    if a >= 1e9:
        return f"USD {v / 1e9:.2f} mil M"
    if a >= 1e6:
        return f"USD {v / 1e6:.1f} M"
    if a >= 1e3:
        return f"USD {v / 1e3:.0f} k"
    return f"USD {v:.0f}"


def _arrow_defs(pal: dict) -> str:
    out = []
    for k, col in (("material", pal["ink"]), ("energy", pal["energy"]), ("utility", pal["utility"])):
        out.append(f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="8" markerHeight="8" '
                   f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L10,5 L0,9 Z" fill="{col}"/></marker>')
    return "<defs>" + "".join(out) + "</defs>"


def _edge_path(pts: list[list[float]], verticals: list[tuple[float, float, float, int]], idx: int) -> str:
    """Path with small hops where a horizontal segment crosses a vertical segment of another edge."""
    d = [f"M{_f(pts[0][0])},{_f(pts[0][1])}"]
    r = 4.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if abs(y0 - y1) < 1e-6 and abs(x1 - x0) > 2 * r + 6:
            sgn = 1 if x1 > x0 else -1
            xs = sorted({vx for vx, ya, yb, k in verticals if k != idx and ya + 1 < y0 < yb - 1
                         and min(x0, x1) + r + 3 < vx < max(x0, x1) - r - 3}, reverse=sgn < 0)
            last = None
            for vx in xs:
                if last is not None and abs(vx - last) < 2 * r + 2:
                    continue
                d.append(f"L{_f(vx - sgn * r)},{_f(y0)} A{r},{r} 0 0 {1 if sgn > 0 else 0} {_f(vx + sgn * r)},{_f(y0)}")
                last = vx
        d.append(f"L{_f(x1)},{_f(y1)}")
    return " ".join(d)


def _label_spot(pts: list[list[float]], taken: list[tuple[float, float, float, float]], w: float, h: float
                ) -> tuple[float, float]:
    segs = sorted(zip(pts, pts[1:]), key=lambda s: -(abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1])))
    for (a, b) in segs:
        for t in (0.5, 0.3, 0.7, 0.15, 0.85):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            box = (x - w / 2, y - h / 2, w, h)
            if not any(_overlap(box, o) for o in taken):
                return x, y
    a, b = segs[0]
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _overlap(a: tuple, b: tuple, pad: float = 1.0) -> bool:
    return a[0] < b[0] + b[2] + pad and b[0] < a[0] + a[2] + pad and a[1] < b[1] + b[3] + pad and b[1] < a[1] + a[3] + pad


def _normalize_opps(opps: list[dict] | None, model: dict, warnings: list[str]) -> list[dict]:
    if not opps:
        return []
    uids = {u["id"] for u in model["units"]}
    sids = {s["id"] for s in model["streams"]}
    out = []
    order = sorted(range(len(opps)), key=lambda i: (opps[i].get("rank") if isinstance(opps[i].get("rank"), (int, float))
                                                     else 1e9, i))
    for k, i in enumerate(order, start=1):
        o = dict(opps[i])
        cat = o.get("category")
        if cat not in CATEGORY_ES or cat == "other":
            warnings.append(f"Oportunidad «{o.get('id', i)}»: categoría «{cat}» fuera del contrato; se dibuja en gris.")
            cat = "other"
        bad_u = [u for u in o.get("unit_ids") or [] if u not in uids]
        bad_s = [s for s in o.get("stream_ids") or [] if s not in sids]
        if bad_u or bad_s:
            warnings.append(f"Oportunidad «{o.get('id', i)}» cita equipos/corrientes que no están en el modelo: "
                            f"{', '.join(bad_u + bad_s)}.")
        o.update(n=k, category=cat, unit_ids=[u for u in o.get("unit_ids") or [] if u in uids],
                 stream_ids=[s for s in o.get("stream_ids") or [] if s in sids])
        out.append(o)
    return out


def render(lay: dict, opportunities: list[dict] | None = None, theme: str = "light", today: str | None = None,
           table_streams: list[str] | None = None) -> dict:
    """SVG string + JSON scene + legend of a layout produced by ``layout.layout``."""
    if theme not in PALETTES:
        raise ValueError(f"theme «{theme}» no válido: use «light» o «dark».")
    pal, cat_col = PALETTES[theme], CATEGORY_COLORS[theme]
    warnings: list[str] = list(lay["warnings"])
    model = lay["model"]
    nodes = {k: v for k, v in lay["nodes"].items() if v["kind"] != "dummy"}
    edges = lay["edges"]
    opps = _normalize_opps(opportunities, model, warnings)
    date = today or _dt.date.today().isoformat()
    draw_w, draw_h = lay["width"], lay["height"]
    panel_w = PANEL_W if opps else 0
    W = max(draw_w + panel_w, 1180.0)
    oy = HEADER_H
    body: list[str] = []

    # sections ---------------------------------------------------------------------------------------------------
    for s in lay["sections"]:
        x, y, w, h = s["box"]
        body.append(f'<rect x="{_f(x)}" y="{_f(y + oy)}" width="{_f(w)}" height="{_f(h)}" rx="8" fill="{pal["lane"]}" '
                    f'stroke="{pal["frame"]}" stroke-width="1" stroke-dasharray="10,3,2,3"/>')
        title = s["label_es"]
        body.append(_t(x + 12, y + oy + 18, title, 12.5, pal["muted"], weight="bold"))

    # edges --------------------------------------------------------------------------------------------------------
    verticals = []
    for k, e in enumerate(edges):
        for (x0, y0), (x1, y1) in zip(e["points"], e["points"][1:]):
            if abs(x0 - x1) < 1e-6:
                verticals.append((x0, min(y0, y1), max(y0, y1), k))
    edge_svg, label_svg, taken = [], [], []
    for nd in nodes.values():
        taken.append((nd["sym_box"][0], nd["sym_box"][1] + oy, nd["sym_box"][2], nd["sym_box"][3]))
        if nd["label_box"]:
            lb = nd["label_box"]
            taken.append((lb[0], lb[1] + oy, lb[2], lb[3]))
    scene_edges = []
    for k, e in enumerate(edges):
        col = {"material": pal["ink"], "energy": pal["energy"], "utility": pal["utility"]}[e["kind"]]
        dash = {"material": "", "energy": ' stroke-dasharray="7,4"', "utility": ' stroke-dasharray="9,3,2,3"'}[e["kind"]]
        width = 1.5 if e["kind"] == "material" else 1.2
        pts = [[p[0], p[1] + oy] for p in e["points"]]
        edge_svg.append(f'<path d="{_edge_path(pts, [(v[0], v[1] + oy, v[2] + oy, v[3]) for v in verticals], k)}" '
                        f'fill="none" stroke="{col}" stroke-width="{width}"{dash} stroke-linejoin="round" '
                        f'marker-end="url(#ah-{e["kind"]})" data-stream="{escape(e["id"])}"/>')
        nums = e["numbers"]
        text = str(nums[0]) if len(nums) == 1 else (", ".join(map(str, nums)) if len(nums) <= 3 else
                                                    f"{nums[0]}, {nums[1]} +{len(nums) - 2}")
        tw = max(20.0, len(text) * 6.4 + 10)
        lx, ly = _label_spot(pts, taken, tw, 16)
        taken.append((lx - tw / 2, ly - 8, tw, 16))
        if len(nums) == 1 and len(text) <= 3:
            r = 10.5
            label_svg.append(f'<path d="M{_f(lx)},{_f(ly - r)} L{_f(lx + r)},{_f(ly)} L{_f(lx)},{_f(ly + r)} '
                             f'L{_f(lx - r)},{_f(ly)} Z" fill="{pal["bg"]}" stroke="{col}" stroke-width="1"/>')
        else:
            label_svg.append(f'<rect x="{_f(lx - tw / 2)}" y="{_f(ly - 8)}" width="{_f(tw)}" height="16" rx="8" '
                             f'fill="{pal["bg"]}" stroke="{col}" stroke-width="1"/>')
        label_svg.append(_t(lx, ly + 3.4, text, 9.5, col, "middle", "bold"))
        scene_edges.append({"id": e["id"], "stream_ids": e["stream_ids"], "numbers": nums, "kind": e["kind"],
                            "from": e["src"], "to": e["dst"], "recycle": e["recycle"],
                            "points": [[round(p[0], 1), round(p[1], 1)] for p in pts],
                            "label_pos": [round(lx, 1), round(ly, 1)], "label_text": text})
    body += edge_svg

    # nodes ----------------------------------------------------------------------------------------------------------
    scene_nodes = []
    for nd in nodes.values():
        x, y, w, h = nd["sym_box"]
        y += oy
        if nd["kind"] == "unit":
            body.append(symbol_svg(nd["type"], x, y, w, h, ink=pal["ink"], bg=pal["bg"], stroke_width=1.5))
            lx, ly = x + w / 2, y + h + 14
            body.append(_t(lx, ly, nd["id"], 11, pal["ink"], "middle", "bold"))
            for i, line in enumerate(nd["lines"]):
                body.append(_t(lx, ly + 12.5 + 12 * i, line, 10, pal["muted"], "middle"))
            if nd.get("note"):
                body.append(_t(lx, ly + 12.5 + 12 * len(nd["lines"]), nd["note"], 9.5, pal["faint"], "middle",
                               style="italic"))
        elif nd["kind"] in ("terminal_in", "terminal_out"):
            k = nd.get("stream_kind", "material")
            col = {"material": pal["ink"], "energy": pal["energy"], "utility": pal["utility"]}[k]
            dash = "" if k == "material" else ' stroke-dasharray="4,3"'
            pts = [(x, y), (x + w - 9, y), (x + w, y + h / 2), (x + w - 9, y + h), (x, y + h)]
            body.append(f'<polygon points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="{pal["bg"]}" '
                        f'stroke="{col}" stroke-width="1.2"{dash}/>')
            body.append(_t(x + (w - 9) / 2 + 2, y + h / 2 + 3.5, nd["label_es"], 9.5, col, "middle"))
        elif nd["kind"] == "section_block":
            body.append(f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" rx="8" fill="{pal["block"]}" '
                        f'stroke="{pal["ink"]}" stroke-width="1.5"/>')
            for i, line in enumerate(_wrap(nd["label_es"], 26, 2)):
                body.append(_t(x + w / 2, y + 18 + 14 * i, line, 12, pal["ink"], "middle", "bold"))
            body.append(_t(x + w / 2, y + 48, nd["note"], 9.5, pal["muted"], "middle"))
            tops = nd.get("top_types") or []
            for i, t in enumerate(tops):
                bw = w / max(1, len(tops))
                body.append(symbol_svg(t, x + i * bw + bw / 2 - 15, y + 56, 30, 40, ink=pal["muted"], bg=pal["block"],
                                       stroke_width=1.1))
        scene_nodes.append({"id": nd["id"], "type": nd["type"], "kind": nd["kind"], "section": nd["group"],
                            "label_es": nd["label_es"], "label_lines": nd["lines"], "note": nd.get("note"),
                            "unit_ids": nd["unit_ids"], "sym_box": [round(v, 1) for v in (x, y, w, h)],
                            "label_box": None if not nd["label_box"] else
                            [round(nd["label_box"][0], 1), round(nd["label_box"][1] + oy, 1),
                             round(nd["label_box"][2], 1), round(nd["label_box"][3], 1)]})
    body += label_svg

    # opportunity markers ---------------------------------------------------------------------------------------------
    unit_node = {}
    for nd in nodes.values():
        for u in nd["unit_ids"]:
            unit_node[u] = nd
    stream_edge = {}
    for se in scene_edges:
        for s in se["stream_ids"]:
            stream_edge[s] = se
    badges_on: dict[str, int] = {}
    scene_markers = []
    for o in opps:
        col = cat_col[o["category"]]
        spots = []
        done_nodes: set = set()
        for u in o["unit_ids"]:
            nd = unit_node.get(u)
            if nd is None or nd["id"] in done_nodes:
                continue
            done_nodes.add(nd["id"])
            k = badges_on.get(nd["id"], 0)
            badges_on[nd["id"]] = k + 1
            x, y, w, h = nd["sym_box"]
            spots.append((nd["id"], x + w + 4 - 22 * k, y + oy - 4))
        for s in o["stream_ids"]:
            se = stream_edge.get(s)
            if se is None:
                continue
            k = badges_on.get("s:" + se["id"], 0)
            badges_on["s:" + se["id"]] = k + 1
            spots.append((se["id"], se["label_pos"][0] + 18 + 22 * k, se["label_pos"][1] - 14))
        seen = set()
        for target, bx, by in spots:
            if (round(bx), round(by)) in seen:
                continue
            seen.add((round(bx), round(by)))
            body.append(f'<g data-opportunity="{escape(str(o.get("id", o["n"])))}"><circle cx="{_f(bx)}" cy="{_f(by)}" '
                        f'r="10" fill="{col}" stroke="{pal["bg"]}" stroke-width="2"/>'
                        + _t(bx, by + 3.8, str(o["n"]), 10.5, pal["badge_text"], "middle", "bold") + "</g>")
            scene_markers.append({"opportunity_id": o.get("id"), "n": o["n"], "category": o["category"], "color": col,
                                  "target": target, "pos": [round(bx, 1), round(by, 1)]})
        o["drawn"] = bool(spots)

    # side panel of opportunities ------------------------------------------------------------------------------------
    panel = []
    panel_h = 0.0
    if opps:
        px, py = draw_w + 10, oy + 10
        items = []
        yy = py + 34
        for o in opps:
            col = cat_col[o["category"]]
            title = _wrap(str(o.get("title_es") or o.get("id") or "Oportunidad"), 50, 3)
            lo, hi = o.get("value_low_usd_y"), o.get("value_high_usd_y")
            val = money(o.get("value_usd_y"))
            if o.get("value_usd_y") is not None:
                val += "/año"
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    val += f"  (rango {money(lo).replace('USD ', '')}–{money(hi).replace('USD ', '')})"
            meta = f"{CATEGORY_ES[o['category']]} · confianza {o.get('confidence') or 'sin dato'}"
            if not o["drawn"]:
                meta += " · fuera de esta vista"
            basis = _wrap(str(o.get("basis_es") or ""), 62, 2) if o.get("basis_es") else []
            items.append(f'<circle cx="{_f(px + 18)}" cy="{_f(yy + 2)}" r="10" fill="{col}"/>'
                         + _t(px + 18, yy + 5.8, str(o["n"]), 10.5, pal["badge_text"], "middle", "bold"))
            for i, line in enumerate(title):
                items.append(_t(px + 36, yy + 6 + 13 * i, line, 11, pal["ink"], weight="bold"))
            y2 = yy + 6 + 13 * len(title)
            items.append(_t(px + 36, y2 + 1, val, 11, col, weight="bold"))
            items.append(_t(px + 36, y2 + 14, meta, 9.5, pal["muted"]))
            for i, line in enumerate(basis):
                items.append(_t(px + 36, y2 + 27 + 11.5 * i, line, 9, pal["faint"]))
            yy = y2 + 27 + 11.5 * len(basis) + 12
        total = sum(o["value_usd_y"] for o in opps if isinstance(o.get("value_usd_y"), (int, float)))
        items.append(f'<line x1="{_f(px + 12)}" y1="{_f(yy - 4)}" x2="{_f(px + PANEL_W - 22)}" y2="{_f(yy - 4)}" '
                     f'stroke="{pal["rule"]}"/>')
        items.append(_t(px + 14, yy + 12, f"Suma simple de lo cuantificado: {money(total)}/año", 10.5, pal["ink"],
                        weight="bold"))
        items.append(_t(px + 14, yy + 25, "Las oportunidades pueden no ser aditivas (interactúan entre sí).", 9,
                        pal["faint"]))
        panel_h = yy + 36 - py
        panel.append(f'<rect x="{_f(px)}" y="{_f(py)}" width="{PANEL_W - 20}" height="{_f(panel_h)}" rx="8" '
                     f'fill="{pal["panel"]}" stroke="{pal["rule"]}"/>')
        panel.append(_t(px + 14, py + 22, "Oportunidades de mejora", 13, pal["header"], weight="bold"))
        panel += items

    # bottom: legend, stream table, title block ----------------------------------------------------------------------
    y0 = oy + max(draw_h, panel_h + 20) + 10
    bottom: list[str] = []
    legend = {"line_types": [], "categories": [], "symbols": []}
    lx = 30.0
    bottom.append(_t(lx, y0 + 16, "Leyenda", 12.5, pal["header"], weight="bold"))
    ly = y0 + 34
    kinds_used = sorted({e["kind"] for e in edges}, key=["material", "energy", "utility"].index) or ["material"]
    for k in kinds_used:
        col = {"material": pal["ink"], "energy": pal["energy"], "utility": pal["utility"]}[k]
        dash = {"material": "", "energy": ' stroke-dasharray="7,4"', "utility": ' stroke-dasharray="9,3,2,3"'}[k]
        bottom.append(f'<line x1="{_f(lx)}" y1="{_f(ly)}" x2="{_f(lx + 40)}" y2="{_f(ly)}" stroke="{col}" '
                      f'stroke-width="1.5"{dash} marker-end="url(#ah-{k})"/>')
        bottom.append(_t(lx + 50, ly + 3.5, KIND_ES[k], 10, pal["ink"]))
        legend["line_types"].append({"kind": k, "label_es": KIND_ES[k]})
        ly += 17
    r = 9
    bottom.append(f'<path d="M{_f(lx + 20)},{_f(ly - r)} L{_f(lx + 20 + r)},{_f(ly)} L{_f(lx + 20)},{_f(ly + r)} '
                  f'L{_f(lx + 20 - r)},{_f(ly)} Z" fill="{pal["bg"]}" stroke="{pal["ink"]}"/>')
    bottom.append(_t(lx + 20, ly + 3.4, "n", 9, pal["ink"], "middle", "bold"))
    bottom.append(_t(lx + 50, ly + 3.5, "Número de corriente (ver tabla)", 10, pal["ink"]))
    ly += 20
    bottom.append(f'<path d="M{_f(lx)},{_f(ly)} L{_f(lx + 16)},{_f(ly)} A4,4 0 0 1 {_f(lx + 24)},{_f(ly)} '
                  f'L{_f(lx + 40)},{_f(ly)} M{_f(lx + 20)},{_f(ly - 9)} L{_f(lx + 20)},{_f(ly + 9)}" fill="none" '
                  f'stroke="{pal["ink"]}" stroke-width="1.3"/>')
    bottom.append(_t(lx + 50, ly + 3.5, "Cruce sin conexión", 10, pal["ink"]))
    ly += 20
    pts = [(lx, ly - 8), (lx + 31, ly - 8), (lx + 40, ly), (lx + 31, ly + 8), (lx, ly + 8)]
    bottom.append(f'<polygon points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="{pal["bg"]}" '
                  f'stroke="{pal["ink"]}" stroke-width="1.1"/>')
    bottom.append(_t(lx + 50, ly + 3.5, "Alimentación / producto (límite de batería)", 10, pal["ink"]))
    ly += 22
    if opps:
        cats = [c for c in CATEGORY_ES if any(o["category"] == c for o in opps)]
        for c in cats:
            bottom.append(f'<circle cx="{_f(lx + 20)}" cy="{_f(ly)}" r="7" fill="{cat_col[c]}"/>')
            bottom.append(_t(lx + 50, ly + 3.5, f"Oportunidad: {CATEGORY_ES[c]}", 10, pal["ink"]))
            legend["categories"].append({"category": c, "label_es": CATEGORY_ES[c], "color": cat_col[c]})
            ly += 17
    used = [t for t in EQUIPMENT_TYPES if any(n["type"] == t for n in nodes.values())]
    for nd in nodes.values():
        for t in nd.get("top_types") or []:
            if t not in used:
                used.append(t)
    used = [t for t in EQUIPMENT_TYPES if t in used]
    sx, sy = lx + 300, y0 + 34
    per_col = max(1, int(-(-len(used) // 2)))
    for i, t in enumerate(used):
        cx = sx + (i // per_col) * 190
        cy = sy + (i % per_col) * 30
        bottom.append(symbol_svg(t, cx, cy - 12, 26, 24, ink=pal["ink"], bg=pal["bg"], stroke_width=1.1))
        bottom.append(_t(cx + 34, cy + 3.5, LABELS_ES[t], 10, pal["ink"]))
        legend["symbols"].append({"type": t, "label_es": LABELS_ES[t]})
    legend_h = max(ly - y0, 34 + per_col * 30)
    legend_w = 300 + (2 if len(used) > 1 else 1) * 190 + 20

    # stream table
    streams = model["streams"]
    shown_ids = {s for e in edges for s in e["stream_ids"]}
    rows = [s for s in streams if s["id"] in shown_ids and (table_streams is None or s["id"] in table_streams)]
    if len(rows) > MAX_TABLE_ROWS:
        warnings.append(f"La tabla de corrientes muestra {MAX_TABLE_ROWS} de {len(rows)}; pase table_streams para elegir.")
        rows = rows[:MAX_TABLE_ROWS]
    numbers = {s["id"]: i + 1 for i, s in enumerate(streams)}
    unit_label = {u["id"]: u["id"] for u in model["units"]}
    table_rows = []
    for s in rows:
        fr = unit_label.get(s["from"]) if s["from"] else "límite"
        to = unit_label.get(s["to"]) if s["to"] else "límite"
        T = f"{s['T_K'] - 273.15:.0f}" if isinstance(s.get("T_K"), (int, float)) else "—"
        P = _num(s["P_Pa"] / 1e5) if isinstance(s.get("P_Pa"), (int, float)) else "—"
        table_rows.append({"n": numbers[s["id"]], "id": s["id"], "label_es": s["label_es"], "from": s["from"],
                           "to": s["to"], "flow": flow_text(s), "T_C": T, "P_bar": P, "kind": s["kind"]})
    tx0 = lx + legend_w + 10
    W = max(W, tx0 + TABLE_W + 30)                 # at least one full table block fits on the sheet
    title_w, title_h = 430.0, 128.0
    avail = W - tx0 - 30
    nblocks = max(1, int(avail // (TABLE_W + 18)))
    if table_rows:
        nblocks = min(nblocks, len(table_rows))
        per = -(-len(table_rows) // nblocks)
    else:
        per = 0
    table_h = 40 + per * ROW_H
    bottom.append(_t(tx0, y0 + 16, "Tabla de corrientes", 12.5, pal["header"], weight="bold"))
    for b in range(nblocks if table_rows else 0):
        bx = tx0 + b * (TABLE_W + 18)
        by = y0 + 26
        x = bx
        bottom.append(f'<rect x="{_f(bx)}" y="{_f(by)}" width="{TABLE_W}" height="{ROW_H}" fill="{pal["panel"]}" '
                      f'stroke="{pal["rule"]}"/>')
        for name, w in TABLE_COLS:
            bottom.append(_t(x + 5, by + 11.5, name, 9.5, pal["muted"], weight="bold"))
            x += w
        for i, row in enumerate(table_rows[b * per:(b + 1) * per]):
            ry = by + ROW_H * (i + 1)
            if i % 2:
                bottom.append(f'<rect x="{_f(bx)}" y="{_f(ry)}" width="{TABLE_W}" height="{ROW_H}" fill="{pal["lane"]}"/>')
            name = row["label_es"] if len(row["label_es"]) <= 34 else row["label_es"][:33] + "…"
            frto = f"{row['from'] or 'límite'} → {row['to'] or 'límite'}"
            frto = frto if len(frto) <= 38 else frto[:37] + "…"
            cells = [str(row["n"]), name, frto, row["flow"], row["T_C"], row["P_bar"]]
            x = bx
            for (cname, w), c in zip(TABLE_COLS, cells):
                anchor = "end" if cname in ("Flujo", "T (°C)", "P (bar)") else "start"
                tx = x + w - 6 if anchor == "end" else x + 5
                bottom.append(_t(tx, ry + 11.5, c, 9.5, pal["ink"], anchor))
                x += w
        bottom.append(f'<line x1="{_f(bx)}" y1="{_f(by + ROW_H * (1 + min(per, len(table_rows) - b * per)))}" '
                      f'x2="{_f(bx + TABLE_W)}" y2="{_f(by + ROW_H * (1 + min(per, len(table_rows) - b * per)))}" '
                      f'stroke="{pal["rule"]}"/>')
    if not table_rows:
        bottom.append(_t(tx0, y0 + 40, "Sin corrientes con datos en esta vista.", 10, pal["muted"]))
    content_h = max(legend_h, table_h)
    # title block (bottom right)
    tb_y = y0 + content_h + 20
    H = tb_y + title_h + 20
    tbx = W - title_w - 20
    src = str(model.get("source") or "—")
    view_es = {"full": "completa (todas las secciones)", "sections": "por secciones (colapsada)"}.get(
        lay["view"], f"sección {lay['view'].split(':', 1)[-1]}")
    rows_tb = [("Proceso", model["name"]), ("Fuente", src), ("Vista", view_es), ("Fecha", date),
               ("Equipos / corrientes", f"{sum(1 for n in nodes.values() if n['kind'] == 'unit') or len(model['units'])}"
                                        f" / {len(rows)}")]
    bottom.append(f'<rect x="{_f(tbx)}" y="{_f(tb_y)}" width="{_f(title_w)}" height="{_f(title_h)}" fill="{pal["bg"]}" '
                  f'stroke="{pal["ink"]}" stroke-width="1.4"/>')
    for i, (k, v) in enumerate(rows_tb):
        ry = tb_y + 18 * i
        if i:
            bottom.append(f'<line x1="{_f(tbx)}" y1="{_f(ry)}" x2="{_f(tbx + title_w)}" y2="{_f(ry)}" '
                          f'stroke="{pal["rule"]}"/>')
        bottom.append(_t(tbx + 8, ry + 13, k, 9, pal["muted"]))
        v = v if len(v) <= 52 else v[:51] + "…"
        bottom.append(_t(tbx + 120, ry + 13, v, 10, pal["ink"], weight="bold" if i == 0 else "normal"))
    bottom.append(f'<line x1="{_f(tbx + 112)}" y1="{_f(tb_y)}" x2="{_f(tbx + 112)}" y2="{_f(tb_y + 90)}" '
                  f'stroke="{pal["rule"]}"/>')
    bottom.append(f'<rect x="{_f(tbx)}" y="{_f(tb_y + 90)}" width="{_f(title_w)}" height="{_f(title_h - 90)}" '
                  f'fill="{pal["panel"]}" stroke="{pal["ink"]}" stroke-width="1.4"/>')
    bottom.append(_t(tbx + title_w / 2, tb_y + 106, "Diagrama generado; no es un P&ID", 11, pal["ink"], "middle", "bold"))
    bottom.append(_t(tbx + title_w / 2, tb_y + 120, "AlchemAI · PFD automático · símbolos según convención ISO 10628-2",
                     8.5, pal["muted"], "middle"))
    for i, w in enumerate(warnings[:4]):
        bottom.append(_t(30, tb_y + 14 + 13 * i, "⚠ " + (w if len(w) <= 120 else w[:119] + "…"), 9, pal["energy"]))

    header = [_t(30, 30, model["name"], 19, pal["header"], weight="bold"),
              _t(30, 50, f"Diagrama de flujo de proceso (PFD) · vista {view_es} · {date}", 11, pal["muted"]),
              f'<line x1="30" y1="{HEADER_H - 4}" x2="{_f(W - 30)}" y2="{HEADER_H - 4}" stroke="{pal["rule"]}"/>']
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(W)}" height="{_f(H)}" viewBox="0 0 {_f(W)} {_f(H)}" '
           f'font-family="{FONT}">'
           f'<title>{escape(model["name"])} — PFD</title>{_arrow_defs(pal)}'
           f'<rect width="100%" height="100%" fill="{pal["bg"]}"/>' + "".join(header) + "".join(body) + "".join(panel)
           + "".join(bottom) + "</svg>")
    used_types = sorted({n["type"] for n in nodes.values() if n["type"] in SYMBOLS} | set(used))
    scene = {
        "canvas": {"width": round(W, 1), "height": round(H, 1), "header_h": HEADER_H, "grid_px": 10},
        "theme": theme, "palette": pal, "category_colors": cat_col, "view": lay["view"],
        "title_block": {"name": model["name"], "source": src, "date": date, "view_es": view_es,
                        "note_es": "Diagrama generado; no es un P&ID"},
        "sections": [{"id": s["id"], "label_es": s["label_es"], "box": [round(s["box"][0], 1), round(s["box"][1] + oy, 1),
                                                                         round(s["box"][2], 1), round(s["box"][3], 1)]}
                     for s in lay["sections"]],
        "nodes": scene_nodes, "edges": scene_edges, "markers": scene_markers,
        "opportunities": [{"n": o["n"], "id": o.get("id"), "category": o["category"], "title_es": o.get("title_es"),
                           "value_usd_y": o.get("value_usd_y"), "value_low_usd_y": o.get("value_low_usd_y"),
                           "value_high_usd_y": o.get("value_high_usd_y"), "confidence": o.get("confidence"),
                           "unit_ids": o["unit_ids"], "stream_ids": o["stream_ids"], "drawn": o["drawn"]} for o in opps],
        "stream_table": table_rows,
        "symbols": {t: {"viewbox": [SYMBOLS[t][0], SYMBOLS[t][1]], "primitives": SYMBOLS[t][2]} for t in used_types},
    }
    return {"svg": svg, "scene": scene, "legend": legend, "warnings": warnings}

