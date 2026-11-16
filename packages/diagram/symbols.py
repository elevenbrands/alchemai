"""Equipment vocabulary of AlchemAI process flow diagrams (PFD) and their SVG symbols.

``EQUIPMENT_TYPES`` is the CLOSED vocabulary every ProcessModel ``units[].type`` must use (blocks R and O import it; types
may be added, never removed). Each type has a symbol drawn here from scratch (MIT, own drawings) following the CONVENTIONS
of ISO 10628-2 / ISA-5.1 process flow diagrams: a circle with an inscribed triangle for a pump, a circle with converging
lines for a compressor, a circle crossed by a zig-zag for a heat exchanger, a capsule with trays for a column, a bow-tie
for a valve, a vessel with an agitator for a stirred reactor, and so on. Those are conventions shared by every textbook PFD
(Turton, Towler-Sinnott, Seider), not anyone's artwork; no symbol library was copied.

A symbol is a list of primitives in its own view box (``SYMBOLS[type] = (vw, vh, primitives)``). Primitives are plain
dicts (``el`` = path | circle | ellipse | rect | line | polyline | text) with ``fill`` in {"none", "bg", "ink"} (bg = the
diagram background, so the symbol hides lines behind it; ink = the stroke colour) and an optional ``dash``. The same list
is shipped in the JSON scene so the Mac app can draw the symbols natively; ``symbol_svg`` turns it into an SVG fragment
scaled uniformly into a box.
"""

from __future__ import annotations

EQUIPMENT_TYPES: list[str] = [
    "pump", "compressor", "fan", "heat_exchanger", "air_cooler", "heater", "cooler", "fired_heater", "boiler",
    "cooling_tower", "vessel", "flash_drum", "separator_3phase", "decanter", "tank", "column", "packed_column",
    "reactor", "cstr", "pfr", "fixed_bed_reactor", "fluidized_bed_reactor", "fermenter", "mixer", "splitter", "valve",
    "filter", "centrifuge", "screw_press", "mill", "conveyor", "dryer", "evaporator", "crystallizer", "cyclone",
    "membrane", "adsorber", "flare", "stack", "generic",
    # added by block D (M27): stirred mix tank without reaction (BioSTEAM «Mix tank», wash tanks, blending)
    "agitated_tank",
    # added by block D (M27): turbine / expander (DWSIM «expander», steam turbines)
    "expander",
    # added by block N (M27): power-plant equipment (nuclear steam cycle)
    "nuclear_reactor", "steam_turbine", "generator", "condenser", "moisture_separator_reheater",
]

LABELS_ES: dict[str, str] = {
    "pump": "Bomba", "compressor": "Compresor", "fan": "Ventilador / soplador", "heat_exchanger": "Intercambiador de calor",
    "air_cooler": "Enfriador por aire", "heater": "Calentador (servicio)", "cooler": "Enfriador (servicio)",
    "fired_heater": "Horno de proceso", "boiler": "Caldera", "cooling_tower": "Torre de enfriamiento",
    "vessel": "Recipiente", "flash_drum": "Tanque flash", "separator_3phase": "Separador trifásico",
    "decanter": "Decantador", "tank": "Tanque de almacenamiento", "column": "Columna de platos",
    "packed_column": "Columna empacada", "reactor": "Reactor", "cstr": "Reactor agitado (CSTR)",
    "pfr": "Reactor tubular (PFR)", "fixed_bed_reactor": "Reactor de lecho fijo",
    "fluidized_bed_reactor": "Reactor de lecho fluidizado", "fermenter": "Fermentador", "mixer": "Mezclador",
    "splitter": "Divisor de corriente", "valve": "Válvula", "filter": "Filtro", "centrifuge": "Centrífuga",
    "screw_press": "Prensa de tornillo", "mill": "Molino", "conveyor": "Transportador", "dryer": "Secador",
    "evaporator": "Evaporador", "crystallizer": "Cristalizador", "cyclone": "Ciclón", "membrane": "Membrana",
    "adsorber": "Adsorbedor", "flare": "Quemador elevado (flare)", "stack": "Chimenea", "generic": "Equipo genérico",
    "agitated_tank": "Tanque agitado", "expander": "Turbina / expansor",
    "nuclear_reactor": "Reactor nuclear (vasija)", "steam_turbine": "Turbina de vapor", "generator": "Generador eléctrico",
    "condenser": "Condensador de superficie", "moisture_separator_reheater": "Separador de humedad y recalentador (MSR)",
}

# nominal drawing box (px) of each symbol on the diagram; the view box aspect is preserved inside it
NOMINAL_SIZE: dict[str, tuple[float, float]] = {}


def _p(d: str, fill: str = "bg", dash: list[float] | None = None) -> dict:
    out = {"el": "path", "d": d, "fill": fill}
    if dash:
        out["dash"] = dash
    return out


def _c(cx: float, cy: float, r: float, fill: str = "bg") -> dict:
    return {"el": "circle", "cx": cx, "cy": cy, "r": r, "fill": fill}


def _e(cx: float, cy: float, rx: float, ry: float, fill: str = "bg") -> dict:
    return {"el": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "fill": fill}


def _r(x: float, y: float, w: float, h: float, rx: float = 0.0, fill: str = "bg", dash: list[float] | None = None) -> dict:
    out = {"el": "rect", "x": x, "y": y, "w": w, "h": h, "rx": rx, "fill": fill}
    if dash:
        out["dash"] = dash
    return out


def _l(x1: float, y1: float, x2: float, y2: float, dash: list[float] | None = None) -> dict:
    out = {"el": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "fill": "none"}
    if dash:
        out["dash"] = dash
    return out


def _pl(pts: list[tuple[float, float]], fill: str = "none", dash: list[float] | None = None) -> dict:
    out = {"el": "polyline", "points": [list(p) for p in pts], "fill": fill}
    if dash:
        out["dash"] = dash
    return out


def _capsule_v(x: float, y: float, w: float, h: float, head: float) -> str:
    """Vertical vessel with 2:1 elliptical heads (outline path)."""
    r = w / 2
    return (f"M{x},{y + head} A{r},{head} 0 0 1 {x + w},{y + head} L{x + w},{y + h - head} "
            f"A{r},{head} 0 0 1 {x},{y + h - head} Z")


def _capsule_h(x: float, y: float, w: float, h: float, head: float) -> str:
    """Horizontal vessel with elliptical heads."""
    r = h / 2
    return (f"M{x + head},{y} L{x + w - head},{y} A{head},{r} 0 0 1 {x + w - head},{y + h} L{x + head},{y + h} "
            f"A{head},{r} 0 0 1 {x + head},{y} Z")


def _zigzag(x0: float, x1: float, y: float, amp: float, n: int = 2) -> list[tuple[float, float]]:
    pts = [(x0, y)]
    step = (x1 - x0) / (2 * n + 2)
    x = x0 + step
    pts.append((x, y))
    for i in range(2 * n):
        x += step
        pts.append((x, y + (amp if i % 2 == 0 else -amp)))
    x += step
    pts.append((x, y))
    pts.append((x1, y))
    return pts


def _agitator(cx: float, top: float, bottom: float, blade: float) -> list[dict]:
    return [_r(cx - 5, top - 9, 10, 8, 1.5), _l(cx, top - 1, cx, bottom),
            _p(f"M{cx - blade},{bottom - 4} L{cx + blade},{bottom + 4} M{cx - blade},{bottom + 4} L{cx + blade},{bottom - 4}",
               fill="none")]


def _flame(cx: float, base: float, h: float) -> dict:
    w = h * 0.45
    return _p(f"M{cx},{base - h} C{cx + w},{base - h * 0.45} {cx + w * 0.6},{base} {cx},{base} "
              f"C{cx - w * 0.6},{base} {cx - w},{base - h * 0.45} {cx},{base - h} Z", fill="none")


def _build() -> dict[str, tuple[float, float, list[dict]]]:
    S: dict[str, tuple[float, float, list[dict]]] = {}
    # rotating machines -------------------------------------------------------------------------------------------
    S["pump"] = (40, 44, [
        _p("M8,42 L13,33 M32,42 L27,33 M5,42 L35,42", fill="none"),
        _c(20, 20, 17), _p("M8.5,7.5 L37,20 L8.5,32.5", fill="none")])
    S["compressor"] = (44, 40, [
        _c(22, 20, 18), _l(7, 9, 38, 15), _l(7, 31, 38, 25)])
    S["expander"] = (44, 40, [
        _c(22, 20, 18), _l(6, 15, 37, 9), _l(6, 25, 37, 31)])
    S["fan"] = (40, 40, [
        _c(20, 20, 17), _e(20, 12, 4, 7.5, fill="none"), _e(20, 28, 4, 7.5, fill="none"), _c(20, 20, 2, fill="ink")])
    # heat transfer -----------------------------------------------------------------------------------------------
    S["heat_exchanger"] = (46, 46, [_c(23, 23, 20), _pl(_zigzag(0, 46, 23, 9, 2))])
    S["heater"] = (46, 46, [_c(23, 23, 20), _pl(_zigzag(0, 46, 23, 9, 2)),
                            _p("M17,40 L23,32 L29,40 Z", fill="ink")])
    S["cooler"] = (46, 46, [_c(23, 23, 20), _pl(_zigzag(0, 46, 23, 9, 2)),
                            _p("M17,6 L23,14 L29,6 Z", fill="ink")])
    S["air_cooler"] = (76, 50, [
        _r(2, 4, 72, 16), _pl(_zigzag(2, 74, 12, 4, 4)),
        _l(38, 20, 38, 30), _e(38, 34, 18, 4, fill="none"), _l(30, 46, 46, 46), _l(38, 38, 38, 46)])
    S["fired_heater"] = (72, 96, [
        _r(28, 0, 16, 18), _p("M4,94 L4,40 L20,18 L52,18 L68,40 L68,94 Z"), _l(4, 40, 68, 40, dash=[3, 2]),
        _pl([(14, 46), (14, 84), (24, 84), (24, 50), (34, 50), (34, 84), (44, 84), (44, 50), (54, 50), (54, 84),
             (58, 84)]), _flame(19, 93, 9), _flame(36, 93, 9), _flame(53, 93, 9)])
    S["boiler"] = (62, 84, [
        _r(10, 0, 42, 14, 7), _r(4, 14, 54, 68, 4), _pl([(16, 20), (16, 54), (24, 54), (24, 24), (32, 24), (32, 54),
                                                        (40, 54), (40, 24), (46, 24), (46, 54)]),
        _flame(23, 79, 11), _flame(39, 79, 11)])
    S["cooling_tower"] = (70, 84, [
        _p("M8,82 Q22,44 14,10 L56,10 Q48,44 62,82 Z"), _e(35, 6, 13, 4, fill="none"), _l(35, 2, 35, 10),
        _p("M16,62 L54,62 M18,68 L52,68", fill="none", dash=[3, 2])])
    # vessels -----------------------------------------------------------------------------------------------------
    S["vessel"] = (44, 84, [_p(_capsule_v(2, 2, 40, 80, 10))])
    S["flash_drum"] = (44, 84, [_p(_capsule_v(2, 2, 40, 80, 10)), _l(4, 24, 40, 24, dash=[2.5, 2]),
                                _l(4, 27, 40, 27, dash=[2.5, 2])])
    S["separator_3phase"] = (104, 50, [
        _r(78, 40, 14, 9), _p(_capsule_h(2, 2, 100, 40, 12)), _l(70, 42, 70, 18), _l(4, 30, 70, 30, dash=[3, 2])])
    S["decanter"] = (96, 44, [_p(_capsule_h(2, 2, 92, 40, 11)), _l(6, 24, 90, 24, dash=[4, 2]), _l(62, 42, 62, 28)])
    S["tank"] = (72, 62, [_p("M4,60 L4,16 Q36,0 68,16 L68,60 Z"), _l(4, 16, 68, 16, dash=[2, 2])])
    S["agitated_tank"] = (60, 72, [_p("M4,14 L4,64 Q4,70 10,70 L50,70 Q56,70 56,64 L56,14 Z")] + _agitator(30, 14, 56, 14))
    S["column"] = (40, 156, [_p(_capsule_v(2, 2, 36, 152, 9))] + [
        _l(2 if i % 2 == 0 else 10, 26 + i * 13, 30 if i % 2 == 0 else 38, 26 + i * 13) for i in range(9)])
    S["packed_column"] = (40, 156, [
        _p(_capsule_v(2, 2, 36, 152, 9)), _r(2, 30, 36, 40, fill="none"), _l(2, 30, 38, 70), _l(38, 30, 2, 70),
        _r(2, 86, 36, 40, fill="none"), _l(2, 86, 38, 126), _l(38, 86, 2, 126)])
    # reactors ----------------------------------------------------------------------------------------------------
    S["reactor"] = (50, 88, [_p(_capsule_v(5, 4, 40, 80, 10)), _p("M2,30 L2,70 Q2,82 12,84 M48,30 L48,70 Q48,82 38,84",
                                                                 fill="none"),
                             _l(2, 30, 5, 30), _l(48, 30, 45, 30)])
    S["cstr"] = (50, 92, [_p(_capsule_v(5, 10, 40, 80, 10))] + _agitator(25, 10, 66, 13))
    S["fermenter"] = (54, 96, [_p(_capsule_v(5, 10, 44, 84, 10))] + _agitator(27, 10, 58, 14) + [
        _l(12, 80, 42, 80)] + [_c(14 + 5 * i, 75 - 3 * (i % 2), 1.4, fill="none") for i in range(7)])
    S["pfr"] = (104, 40, [_p(_capsule_h(2, 2, 100, 36, 10)), _l(14, 12, 90, 12), _l(14, 20, 90, 20), _l(14, 28, 90, 28)])
    S["fixed_bed_reactor"] = (48, 92, [_p(_capsule_v(4, 2, 40, 88, 10)), _r(4, 26, 40, 40, fill="none"),
                                       _l(4, 26, 44, 66), _l(44, 26, 4, 66)])
    S["fluidized_bed_reactor"] = (50, 104, [
        _p(_capsule_v(5, 2, 40, 100, 10)), _p("M15,14 L35,14 L27,34 L23,34 Z", fill="none"), _l(5, 54, 45, 54, dash=[3, 2])]
        + [_c(11 + 6 * (i % 5) + 3 * ((i // 5) % 2), 62 + 7 * (i // 5), 1.4, fill="ink") for i in range(20)])
    # flow elements -----------------------------------------------------------------------------------------------
    S["mixer"] = (36, 36, [_p("M2,2 L34,18 L2,34 Z")])
    S["splitter"] = (36, 36, [_p("M34,2 L2,18 L34,34 Z")])
    S["valve"] = (40, 30, [_p("M2,8 L38,28 L38,8 L2,28 Z"), _l(20, 18, 20, 4), _l(13, 4, 27, 4)])
    # solids and mechanical separation ----------------------------------------------------------------------------
    S["filter"] = (56, 50, [_r(2, 2, 52, 46, 3), _l(4, 46, 52, 4, dash=[4, 2.5])])
    S["centrifuge"] = (56, 60, [
        _r(2, 10, 52, 48, 6), _p("M12,18 L44,18 L38,50 L18,50 Z", fill="none"), _l(28, 2, 28, 54),
        _p("M18,6 A12,4 0 0 0 38,6", fill="none"), _p("M38,6 L33,3 M38,6 L33,9", fill="none")])
    S["screw_press"] = (104, 40, [
        _r(2, 6, 84, 28, 3), _p("M86,6 L102,14 L102,26 L86,34", fill="none")] + [
        _l(10 + 10 * i, 32, 18 + 10 * i, 8) for i in range(7)] + [_l(2, 20, 100, 20, dash=[2, 2])])
    S["mill"] = (60, 60, [
        _p("M6,2 L54,2 L40,18 L20,18 Z"), _r(4, 18, 52, 40, 4), _c(19, 38, 11, fill="none"), _c(41, 38, 11, fill="none"),
        _c(19, 38, 1.5, fill="ink"), _c(41, 38, 1.5, fill="ink")])
    S["conveyor"] = (108, 30, [
        _p("M14,4 L94,4 A11,11 0 0 1 94,26 L14,26 A11,11 0 0 1 14,4 Z"), _c(14, 15, 7, fill="none"),
        _c(94, 15, 7, fill="none"), _c(54, 15, 3, fill="none")])
    S["dryer"] = (100, 50, [
        _p("M4,14 L92,4 L96,34 L8,44 Z"), _p("M28,11 L32,41 M64,7 L68,37", fill="none"),
        _p("M30,8 L34,44 M66,4 L70,40", fill="none", dash=[1.5, 1.5])])
    S["evaporator"] = (54, 104, [
        _p(_capsule_v(2, 2, 50, 100, 12)), _l(2, 54, 52, 54), _l(2, 88, 52, 88)] + [
        _l(10 + 7 * i, 54, 10 + 7 * i, 88) for i in range(6)] + [_l(6, 26, 48, 26, dash=[2.5, 2])])
    S["crystallizer"] = (56, 100, [
        _p("M4,14 Q4,4 14,4 L42,4 Q52,4 52,14 L52,64 L32,96 L24,96 L4,64 Z"), _l(18, 30, 18, 66), _l(38, 30, 38, 66)]
        + _agitator(28, 4, 74, 9))
    S["cyclone"] = (40, 88, [
        _p("M4,8 L36,8 L36,40 L23,82 L17,82 L4,40 Z"), _r(14, 0, 12, 26, fill="none"), _p("M8,16 L32,16", fill="none")])
    S["membrane"] = (84, 46, [_r(2, 2, 80, 42, 3), _l(2, 23, 82, 23, dash=[5, 2.5]),
                              _p("M10,12 L74,12 M10,34 L74,34", fill="none", dash=[1, 3])])
    S["adsorber"] = (46, 96, [_p(_capsule_v(3, 2, 40, 92, 10)), _l(3, 26, 43, 26), _l(3, 70, 43, 70)] + [
        _c(10 + 7 * (i % 5) + 3 * ((i // 5) % 2), 32 + 8 * (i // 5), 1.3, fill="ink") for i in range(25)])
    S["flare"] = (34, 116, [
        _p("M17,4 C26,12 24,24 17,26 C10,24 8,12 17,4 Z"), _r(13, 26, 8, 80), _p("M5,114 L13,100 M29,114 L21,100",
                                                                                 fill="none"), _l(3, 114, 31, 114)])
    S["stack"] = (36, 108, [_p("M10,4 L26,4 L32,106 L4,106 Z"), _l(9, 12, 27, 12), _l(7, 46, 29, 46, dash=[2, 2])])
    S["generic"] = (64, 46, [_r(2, 2, 60, 42, 5), _c(32, 23, 3, fill="none")])
    return S


SYMBOLS = _build()


def _build_power() -> dict[str, tuple[float, float, list[dict]]]:
    """Power-plant symbols (block N, M27), same conventions: reactor vessel with core and fuel, steam turbine as a
    trapezoid widening towards the exhaust, generator as a circle with a sine wave, surface condenser as a shell with a
    tube bundle and hotwell, moisture separator-reheater as a horizontal vessel with chevron vanes and a heating bundle."""
    S: dict[str, tuple[float, float, list[dict]]] = {}
    S["nuclear_reactor"] = (56, 108, [
        _p(_capsule_v(4, 2, 48, 104, 14)), _l(4, 34, 52, 34, dash=[3, 2]), _r(12, 52, 32, 40, fill="none")]
        + [_l(17 + 5.5 * i, 54, 17 + 5.5 * i, 90) for i in range(5)] + [
        _p("M16,24 L22,18 L28,24 L34,18 L40,24", fill="none")])
    S["steam_turbine"] = (64, 56, [
        _p("M6,18 L58,2 L58,54 L6,38 Z"), _l(0, 28, 6, 28), _l(58, 28, 64, 28)])
    S["generator"] = (46, 46, [
        _c(23, 23, 20), _p("M9,23 C13,11 19,11 23,23 C27,35 33,35 37,23", fill="none")])
    S["condenser"] = (104, 60, [
        _r(2, 2, 100, 40, 8), _r(40, 42, 24, 16, 2),
        _pl([(12, 12), (90, 12), (90, 22), (12, 22), (12, 32), (90, 32)])])
    S["moisture_separator_reheater"] = (110, 44, [
        _p(_capsule_h(2, 2, 106, 40, 12)), _l(14, 30, 96, 30)] + [
        _p(f"M{18 + 12 * i},8 L{24 + 12 * i},14 L{18 + 12 * i},20", fill="none") for i in range(7)])
    return S


SYMBOLS.update(_build_power())
for _t, (_vw, _vh, _) in SYMBOLS.items():
    NOMINAL_SIZE[_t] = (float(_vw), float(_vh))
# a few symbols drawn at a size that reads better next to the others
NOMINAL_SIZE.update({"pump": (38, 42), "valve": (32, 24), "mixer": (30, 30), "splitter": (30, 30), "column": (40, 156),
                     "flare": (30, 104), "tank": (70, 60)})

assert set(SYMBOLS) == set(EQUIPMENT_TYPES), set(EQUIPMENT_TYPES) ^ set(SYMBOLS)


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _fill(f: str, bg: str, ink: str) -> str:
    return {"bg": bg, "ink": ink}.get(f, "none")


def primitives_svg(prims: list[dict], ink: str, bg: str, stroke_width: float) -> str:
    """SVG elements of a primitive list in its own coordinates (no transform)."""
    out = []
    sw = _fmt(stroke_width)
    for p in prims:
        el = p["el"]
        common = f'fill="{_fill(p.get("fill", "none"), bg, ink)}" stroke="{ink}" stroke-width="{sw}"'
        if p.get("dash"):
            common += f' stroke-dasharray="{",".join(_fmt(d * stroke_width / 1.2) for d in p["dash"])}"'
        if el == "path":
            out.append(f'<path d="{p["d"]}" {common} stroke-linejoin="round"/>')
        elif el == "circle":
            out.append(f'<circle cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" r="{_fmt(p["r"])}" {common}/>')
        elif el == "ellipse":
            out.append(f'<ellipse cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" rx="{_fmt(p["rx"])}" ry="{_fmt(p["ry"])}" '
                       f'{common}/>')
        elif el == "rect":
            out.append(f'<rect x="{_fmt(p["x"])}" y="{_fmt(p["y"])}" width="{_fmt(p["w"])}" height="{_fmt(p["h"])}" '
                       f'rx="{_fmt(p.get("rx", 0))}" {common}/>')
        elif el == "line":
            out.append(f'<line x1="{_fmt(p["x1"])}" y1="{_fmt(p["y1"])}" x2="{_fmt(p["x2"])}" y2="{_fmt(p["y2"])}" '
                       f'{common}/>')
        elif el == "polyline":
            pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in p["points"])
            out.append(f'<polyline points="{pts}" {common} stroke-linejoin="round"/>')
    return "".join(out)


def fit(type_: str, x: float, y: float, w: float, h: float) -> tuple[float, float, float]:
    """Uniform scale and offset that fit the symbol's view box centred in the box (x, y, w, h)."""
    vw, vh, _ = SYMBOLS[type_]
    s = min(w / vw, h / vh)
    return s, x + (w - vw * s) / 2, y + (h - vh * s) / 2


def symbol_svg(type_: str, x: float, y: float, w: float, h: float, ink: str = "#1b1f24", bg: str = "#ffffff",
               stroke_width: float = 1.6) -> str:
    """SVG fragment (a <g>) of the symbol of ``type_`` scaled uniformly and centred into the box (x, y, w, h).
    ``stroke_width`` is in diagram pixels whatever the scale."""
    if type_ not in SYMBOLS:
        raise ValueError(f"tipo de equipo «{type_}» desconocido; use uno de EQUIPMENT_TYPES: {', '.join(EQUIPMENT_TYPES)}")
    s, ox, oy = fit(type_, x, y, w, h)
    body = primitives_svg(SYMBOLS[type_][2], ink, bg, stroke_width / s)
    return f'<g data-type="{type_}" transform="translate({_fmt(ox)},{_fmt(oy)}) scale({s:.4f})">{body}</g>'


def symbol_bounds(type_: str, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    """The box actually covered by the drawn symbol (after uniform fit) inside (x, y, w, h)."""
    vw, vh, _ = SYMBOLS[type_]
    s, ox, oy = fit(type_, x, y, w, h)
    return ox, oy, vw * s, vh * s


def vocabulary() -> list[dict]:
    """The closed vocabulary with Spanish names and nominal sizes (what ``diagram_symbols`` returns)."""
    return [{"type": t, "label_es": LABELS_ES[t], "nominal_size_px": list(NOMINAL_SIZE[t]),
             "viewbox": [SYMBOLS[t][0], SYMBOLS[t][1]]} for t in EQUIPMENT_TYPES]

