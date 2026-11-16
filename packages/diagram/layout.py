"""Layered left-to-right layout and orthogonal routing of a ProcessModel (block M27_D).

Pipeline (all deterministic, pure Python, no graphviz):
1. ``normalize`` checks a ProcessModel and fills only DRAWING defaults (an unknown type is drawn as ``generic`` and said so;
   no physical datum is ever invented).
2. ``build_view`` picks what is drawn: ``full`` (every unit, framed by section), ``sections`` (each section collapsed into
   one block, inter-section streams aggregated) or ``section:<id>`` (one section; streams that cross its border become
   off-sheet connectors naming the other section). Feeds and products become off-sheet connectors.
3. Inside each group a Sugiyama-style layered layout: cycle breaking by depth-first search (the reversed edges are the
   recycles), longest-path layering with sources pulled next to their first consumer, dummy nodes on long edges, barycentre
   crossing reduction (alternating sweeps, best ordering kept) and a median-based vertical alignment snapped to the grid.
4. Groups (sections) are themselves layered left to right by their inter-section streams and stacked in columns.
5. Every stream is routed orthogonally by A* on a 10 px grid with a bend penalty, equipment and labels as obstacles, a
   penalty for running on top of another stream and a small one for crossing it. Ports: inlets on the left, outlets on the
   right, overhead/vapour outlets of columns, drums, heaters and boilers on the top.
"""

from __future__ import annotations

import heapq
import re

from packages.diagram.symbols import EQUIPMENT_TYPES, LABELS_ES, NOMINAL_SIZE, SYMBOLS, fit

G = 10                      # routing grid (px)
GAP_X = 80                 # horizontal gap between layers inside a section
GAP_Y = 34                  # vertical gap between nodes of a layer
SECTION_PAD = 30            # frame padding around a section's content
SECTION_TITLE_H = 28        # title band of a section frame
SECTION_GAP_X = 110         # gap between section columns
SECTION_GAP_Y = 60          # gap between stacked sections
LABEL_CHARS = 20            # wrap width of the Spanish label (characters)
FONT_W = 5.6                # mean glyph width of the 10 px label font (Helvetica/Arial metrics, approx.)
BLOCK_SIZE = (190.0, 104.0)  # collapsed section block
MAX_ISOLATED_PER_ROW = 8

KINDS = ("material", "energy", "utility")
_OVERHEAD = re.compile(r"vapo|\bgas(es)?\b|gas_|overhead|destilado|\btope\b|\btop\b|\bdomo\b|off.?gas|venteo|\bvent\b|"
                       r"\bglp\b|\blpg\b|h2s|hidr[oó]geno|\bhumos?\b|flue|incondensable|desfogue", re.I)
_TOP_OUT_TYPES = {"column", "packed_column", "flash_drum", "separator_3phase", "evaporator", "vessel", "fired_heater",
                  "boiler", "cooling_tower", "adsorber", "fluidized_bed_reactor", "fixed_bed_reactor", "reactor", "cyclone",
                  "crystallizer", "fermenter", "cstr"}
_ROUND = {"pump", "compressor", "fan", "heat_exchanger", "heater", "cooler"}


# --------------------------------------------------------------------------------------------------- model checking

def normalize(pm: dict) -> tuple[dict, list[str]]:
    """Validated copy of a ProcessModel with drawing defaults. Raises ValueError (Spanish) on what cannot be drawn."""
    if not isinstance(pm, dict):
        raise ValueError("process_model debe ser un dict con la forma ProcessModel: {name, kind, units, streams, sections}.")
    units_in = pm.get("units") or []
    if not units_in:
        raise ValueError("El ProcessModel no trae «units»: sin equipos no hay diagrama que dibujar.")
    warnings: list[str] = []
    units, seen = [], set()
    no_section, unknown = [], []
    for i, u in enumerate(units_in):
        uid = str(u.get("id") or "").strip()
        if not uid:
            raise ValueError(f"units[{i}] no tiene «id»: cada equipo necesita un identificador (etiqueta) único.")
        if uid in seen:
            raise ValueError(f"El id de equipo «{uid}» está repetido; los id deben ser únicos.")
        seen.add(uid)
        t = str(u.get("type") or "generic")
        if t not in EQUIPMENT_TYPES:
            unknown.append(f"{uid} ({t})")
            t = "generic"
        sec = str(u.get("section") or "").strip()
        if not sec:
            no_section.append(uid)
            sec = "general"
        units.append({**u, "id": uid, "type": t, "section": sec, "label_es": str(u.get("label_es") or LABELS_ES[t])})
    if unknown:
        warnings.append("Tipos de equipo fuera del vocabulario cerrado, dibujados como «generic»: " + ", ".join(unknown[:12])
                        + (" …" if len(unknown) > 12 else ""))
    if no_section:
        warnings.append(f"{len(no_section)} equipo(s) sin «section» se agruparon en la sección «general».")
    secs, sec_ids = [], set()
    for s in pm.get("sections") or []:
        sid = str(s.get("id") if isinstance(s, dict) else s)
        if sid not in sec_ids:
            sec_ids.add(sid)
            secs.append({"id": sid, "label_es": str((s.get("label_es") if isinstance(s, dict) else None) or sid)})
    for u in units:
        if u["section"] not in sec_ids:
            sec_ids.add(u["section"])
            secs.append({"id": u["section"], "label_es": "General" if u["section"] == "general" else u["section"]})
    used = {u["section"] for u in units}
    secs = [s for s in secs if s["id"] in used]
    streams, sids = [], set()
    for i, s in enumerate(pm.get("streams") or []):
        sid = str(s.get("id") or f"S{i + 1}")
        if sid in sids:
            raise ValueError(f"El id de corriente «{sid}» está repetido; los id deben ser únicos.")
        sids.add(sid)
        fr, to = s.get("from"), s.get("to")
        for end, val in (("from", fr), ("to", to)):
            if val is not None and str(val) not in seen:
                raise ValueError(f"La corriente «{sid}» tiene «{end}» = «{val}», que no es ningún equipo de units.")
        if fr is None and to is None:
            warnings.append(f"La corriente «{sid}» no tiene origen ni destino: no se dibuja.")
            continue
        kind = s.get("kind") or "material"
        if kind not in KINDS:
            warnings.append(f"Corriente «{sid}»: kind «{kind}» desconocido; se dibuja como material.")
            kind = "material"
        streams.append({**s, "id": sid, "from": None if fr is None else str(fr), "to": None if to is None else str(to),
                        "kind": kind, "label_es": str(s.get("label_es") or sid)})
    out = {**pm, "units": units, "streams": streams, "sections": secs, "name": str(pm.get("name") or "Proceso")}
    return out, warnings


# ------------------------------------------------------------------------------------------------------- the view

def _wrap(text: str, width: int = LABEL_CHARS, max_lines: int = 3) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(w) > width:
            w = w[:width - 1] + "…"
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1].rstrip() + "…"
    return lines or [""]


def _fmt_power(w: float) -> str:
    a = abs(w)
    if a >= 1e6:
        return f"{w / 1e6:.3g} MW"
    if a >= 1e3:
        return f"{w / 1e3:.3g} kW"
    return f"{w:.3g} W"


def _unit_note(u: dict) -> str | None:
    if u.get("note_es"):
        return str(u["note_es"])[:LABEL_CHARS + 6]
    duties = u.get("duties_W") or {}
    for k, v in duties.items():
        if isinstance(v, (int, float)) and v:
            sym = "W" if any(x in k.lower() for x in ("power", "potencia", "electric", "shaft")) else "Q"
            return f"{sym} {_fmt_power(float(v))}"
    return None


def _terminal_label(s: dict, prefix: str = "") -> str:
    lab = prefix + s["label_es"]
    return lab if len(lab) <= 46 else lab[:45] + "…"


def build_view(pm: dict, view: str = "full", link_sections: bool | None = None
               ) -> tuple[dict, list[dict], list[dict], list[str]]:
    """Nodes, edges and groups of the requested view of a normalized model. In the full view a stream between two
    sections is either routed as one line (``link_sections`` True) or drawn as a PAIR of off-sheet connectors carrying
    the same stream number (the multi-area PFD convention); default: routed only with 3 sections or fewer."""
    warnings: list[str] = []
    units = {u["id"]: u for u in pm["units"]}
    sec_label = {s["id"]: s["label_es"] for s in pm["sections"]}
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    numbers = {s["id"]: i + 1 for i, s in enumerate(pm["streams"])}

    def unit_node(u: dict) -> dict:
        return {"id": u["id"], "type": u["type"], "kind": "unit", "group": u["section"], "label_es": u["label_es"],
                "note": _unit_note(u), "unit_ids": [u["id"]]}

    def terminal(nid: str, kind: str, group: str, label: str, skind: str) -> dict:
        return {"id": nid, "type": kind, "kind": kind, "group": group, "label_es": label, "note": None, "unit_ids": [],
                "stream_kind": skind}

    if view == "full" or view.startswith("section:"):
        only = None
        if view.startswith("section:"):
            only = view.split(":", 1)[1]
            if only not in sec_label:
                raise ValueError(f"La sección «{only}» no existe; secciones disponibles: {', '.join(sec_label)}.")
        for u in pm["units"]:
            if only is None or u["section"] == only:
                nodes[u["id"]] = unit_node(u)
        for s in pm["streams"]:
            fr, to = s["from"], s["to"]
            fr_in = fr is not None and fr in nodes
            to_in = to is not None and to in nodes
            if not fr_in and not to_in:
                continue
            src, dst = fr, to
            linked = link_sections if link_sections is not None else len(sec_label) <= 3
            if fr_in and to_in and nodes[fr]["group"] != nodes[to]["group"] and not linked:
                gs, gd = nodes[fr]["group"], nodes[to]["group"]
                a, b = f"__out__{s['id']}", f"__in__{s['id']}"
                nodes[a] = terminal(a, "terminal_out", gs, _terminal_label(s, f"a {sec_label[gd]}: "), s["kind"])
                nodes[b] = terminal(b, "terminal_in", gd, _terminal_label(s, f"de {sec_label[gs]}: "), s["kind"])
                common = {"stream_ids": [s["id"]], "numbers": [numbers[s["id"]]], "kind": s["kind"],
                          "label_es": s["label_es"], "to_port_hint": None}
                edges.append({**common, "id": f"{s['id']}", "src": fr, "dst": a, "port_hint": s.get("from_port"),
                              "overhead": _is_overhead(s, units.get(fr)), "connector_pair": True})
                edges.append({**common, "id": f"{s['id']}·2", "src": b, "dst": to, "port_hint": None,
                              "to_port_hint": s.get("to_port"), "overhead": False, "connector_pair": True})
                continue
            if not fr_in:
                prefix = "" if fr is None else f"desde {sec_label[units[fr]['section']]}: "
                src = f"__in__{s['id']}"
                nodes[src] = terminal(src, "terminal_in", nodes[to]["group"], _terminal_label(s, prefix), s["kind"])
            if not to_in:
                prefix = "" if to is None else f"hacia {sec_label[units[to]['section']]}: "
                dst = f"__out__{s['id']}"
                nodes[dst] = terminal(dst, "terminal_out", nodes[fr]["group"], _terminal_label(s, prefix), s["kind"])
            edges.append({"id": s["id"], "stream_ids": [s["id"]], "numbers": [numbers[s["id"]]], "kind": s["kind"],
                          "src": src, "dst": dst, "label_es": s["label_es"], "port_hint": s.get("from_port"),
                          "to_port_hint": s.get("to_port"), "overhead": _is_overhead(s, units.get(fr))})
        groups = [{"id": sid, "label_es": sec_label[sid]} for sid in sec_label if only is None or sid == only]
        groups = [g for g in groups if any(n["group"] == g["id"] for n in nodes.values())]
    elif view == "sections":
        for sec in pm["sections"]:
            members = [u for u in pm["units"] if u["section"] == sec["id"]]
            counts: dict[str, int] = {}
            for u in members:
                counts[u["type"]] = counts.get(u["type"], 0) + 1
            top = sorted(counts, key=lambda t: (-counts[t], EQUIPMENT_TYPES.index(t)))[:4]
            nodes[f"__sec__{sec['id']}"] = {
                "id": f"__sec__{sec['id']}", "type": "section_block", "kind": "section_block", "group": "__all__",
                "label_es": sec["label_es"], "note": f"{len(members)} equipo(s)", "unit_ids": [u["id"] for u in members],
                "section": sec["id"], "top_types": top}
        agg: dict[tuple, dict] = {}
        for s in pm["streams"]:
            fr, to = s["from"], s["to"]
            fs = None if fr is None else units[fr]["section"]
            ts = None if to is None else units[to]["section"]
            if fs == ts:
                continue
            if fs is None:
                src = f"__in__{s['id']}"
                nodes[src] = terminal(src, "terminal_in", "__all__", _terminal_label(s), s["kind"])
                key = (src, f"__sec__{ts}", s["kind"])
            elif ts is None:
                dst = f"__out__{s['id']}"
                nodes[dst] = terminal(dst, "terminal_out", "__all__", _terminal_label(s), s["kind"])
                key = (f"__sec__{fs}", dst, s["kind"])
            else:
                key = (f"__sec__{fs}", f"__sec__{ts}", s["kind"])
            e = agg.setdefault(key, {"id": "+".join([s["id"]]), "stream_ids": [], "numbers": [], "kind": s["kind"],
                                     "src": key[0], "dst": key[1], "label_es": s["label_es"], "port_hint": None,
                                     "to_port_hint": None, "overhead": False})
            e["stream_ids"].append(s["id"])
            e["numbers"].append(numbers[s["id"]])
        for e in agg.values():
            if len(e["stream_ids"]) > 1:
                e["id"] = f"{e['stream_ids'][0]}+{len(e['stream_ids']) - 1}"
                e["label_es"] = f"{len(e['stream_ids'])} corrientes"
            edges.append(e)
        groups = [{"id": "__all__", "label_es": ""}]
    else:
        raise ValueError(f"view «{view}» no válido: use «sections», «full» o «section:<id>».")
    return nodes, edges, groups, warnings


def _is_overhead(s: dict, src_unit: dict | None) -> bool:
    if s.get("from_port") in ("top", "bottom", "right"):
        return s["from_port"] == "top"
    if src_unit is None or src_unit["type"] not in _TOP_OUT_TYPES:
        return False
    text = f"{s.get('id', '')} {s.get('label_es', '')} {s.get('phase', '')}"
    return bool(_OVERHEAD.search(text)) or str(s.get("phase", "")).lower() in ("g", "v", "vapor", "gas")


# ------------------------------------------------------------------------------------------------- node geometry

def _size(n: dict) -> None:
    if n["kind"] == "section_block":
        sw, sh = BLOCK_SIZE
        n["lines"], n["label_h"] = [], 0.0
    elif n["kind"] in ("terminal_in", "terminal_out"):
        sw = max(70.0, min(290.0, len(n["label_es"]) * 5.3 + 24))
        sh = 20.0
        n["lines"], n["label_h"] = [], 0.0
    elif n["kind"] == "dummy":
        sw, sh = 0.0, 10.0
        n["lines"], n["label_h"] = [], 0.0
    else:
        sw, sh = NOMINAL_SIZE[n["type"]]
        lines = _wrap(n["label_es"])
        n["lines"] = lines
        n["label_h"] = 15.0 + 12.0 * len(lines) + (12.0 if n.get("note") else 0.0) + 2
    n["sym_w"], n["sym_h"] = float(sw), float(sh)
    lw = 0.0
    if n["lines"]:
        lw = max(len(n["id"]) * 6.6, max(len(x) for x in n["lines"]) * FONT_W, len(n.get("note") or "") * FONT_W)
    n["label_w"] = lw
    n["foot_w"] = max(n["sym_w"], lw)
    n["foot_h"] = n["sym_h"] + n["label_h"]


# ---------------------------------------------------------------------------------------- Sugiyama inside a group

def _break_cycles(order: list[str], succ: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Back edges of an iterative DFS that starts from sources (in model order): the feedback (recycle) set."""
    state: dict[str, int] = {}
    back: set[tuple[str, str]] = set()
    indeg = {n: 0 for n in order}
    for u in order:
        for v in succ[u]:
            indeg[v] += 1
    roots = [n for n in order if indeg[n] == 0] + [n for n in order if indeg[n] > 0]
    for r in roots:
        if r in state:
            continue
        stack = [(r, iter(succ[r]))]
        state[r] = 1
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[u] = 2
                stack.pop()
                continue
            st = state.get(nxt, 0)
            if st == 1:
                back.add((u, nxt))
            elif st == 0:
                state[nxt] = 1
                stack.append((nxt, iter(succ[nxt])))
    return back


def _isotonic_tops(lay: list[str], want_centres: list[float], nodes: dict) -> list[float]:
    """Tops of the nodes of a layer closest (least squares) to the wanted centres, keeping their order and the gaps:
    pool-adjacent-violators isotonic regression on t_i = top_i - (sum of the heights + gaps above i)."""
    off, acc = [], 0.0
    for n in lay:
        off.append(acc)
        acc += nodes[n]["foot_h"] + GAP_Y
    target = [w - nodes[n]["sym_h"] / 2 - o for n, w, o in zip(lay, want_centres, off)]
    blocks: list[list[float]] = []          # [mean, count]
    for v in target:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, c2 = blocks.pop()
            m1, c1 = blocks.pop()
            blocks.append([(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2])
    t = []
    for m, c in blocks:
        t += [m] * int(c)
    return [ti + o for ti, o in zip(t, off)]


def _crossings(layers: list[list[str]], succ: dict[str, list[str]], layer_of: dict[str, int]) -> int:
    total = 0
    for li in range(len(layers) - 1):
        pos_b = {n: i for i, n in enumerate(layers[li + 1])}
        segs = [(i, pos_b[v]) for i, u in enumerate(layers[li]) for v in succ[u] if layer_of.get(v) == li + 1]
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                if (segs[a][0] - segs[b][0]) * (segs[a][1] - segs[b][1]) < 0:
                    total += 1
    return total


def _layout_group(nodes: dict[str, dict], ids: list[str], dag_edges: list[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    """Positions (x of the footprint's left, y of the symbol centre) of the nodes of one group, relative to (0, 0).
    Adds dummy nodes for long edges into ``nodes`` (kind 'dummy')."""
    succ = {n: [] for n in ids}
    pred = {n: [] for n in ids}
    for u, v in dag_edges:
        succ[u].append(v)
        pred[v].append(u)
    connected = [n for n in ids if succ[n] or pred[n]]
    isolated = [n for n in ids if not succ[n] and not pred[n]]
    # longest-path layering (Kahn order)
    layer: dict[str, int] = {}
    indeg = {n: len(pred[n]) for n in connected}
    queue = [n for n in connected if indeg[n] == 0]
    topo = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    for u in topo:
        layer[u] = max((layer[p] + 1 for p in pred[u]), default=0)
    for u in reversed(topo):          # pull sources right, next to their first consumer
        if not pred[u] and succ[u]:
            layer[u] = max(0, min(layer[v] for v in succ[u]) - 1)
    # dummies on long edges
    d_succ = {n: list(succ[n]) for n in connected}
    for u in list(connected):
        for v in list(succ[u]):
            span = layer[v] - layer[u]
            if span > 1:
                d_succ[u].remove(v)
                prev = u
                for k in range(1, span):
                    did = f"__d__{u}__{v}__{k}"
                    nodes[did] = {"id": did, "kind": "dummy", "type": "dummy", "group": nodes[u]["group"],
                                  "label_es": "", "note": None, "unit_ids": []}
                    _size(nodes[did])
                    layer[did] = layer[u] + k
                    d_succ.setdefault(prev, []).append(did)
                    d_succ[did] = []
                    prev = did
                d_succ[prev].append(v)
    d_pred: dict[str, list[str]] = {n: [] for n in d_succ}
    for u, vs in d_succ.items():
        for v in vs:
            d_pred.setdefault(v, []).append(u)
    nl = max(layer.values(), default=-1) + 1
    layers: list[list[str]] = [[] for _ in range(nl)]
    for n in d_succ:                  # model order first, then dummies
        layers[layer[n]].append(n)
    # barycentre crossing reduction
    best = [list(x) for x in layers]
    best_c = _crossings(layers, d_succ, layer)
    for it in range(12):
        pos = {n: i for lay in layers for i, n in enumerate(lay)}
        rng = range(1, nl) if it % 2 == 0 else range(nl - 2, -1, -1)
        for li in rng:
            nb = d_pred if it % 2 == 0 else d_succ
            bc = {}
            for n in layers[li]:
                ps = [pos[p] for p in nb.get(n, [])]
                bc[n] = sum(ps) / len(ps) if ps else pos[n]
            layers[li].sort(key=lambda n: (bc[n], pos[n]))
            for i, n in enumerate(layers[li]):
                pos[n] = i
        c = _crossings(layers, d_succ, layer)
        if c < best_c:
            best_c, best = c, [list(x) for x in layers]
    layers = best
    # x of each layer
    xs, x = [], 0.0
    for lay in layers:
        xs.append(x)
        w = max((nodes[n]["foot_w"] for n in lay), default=0.0)
        x += max(w, 20.0) + GAP_X
    # y: stack, then median alignment with the neighbours, keeping order and gaps
    yc: dict[str, float] = {}
    for lay in layers:
        y = 0.0
        for n in lay:
            nd = nodes[n]
            yc[n] = y + nd["sym_h"] / 2
            y += nd["foot_h"] + GAP_Y
    for it in range(10):
        forward = it % 2 == 0
        rng = range(nl) if forward else range(nl - 1, -1, -1)
        for li in rng:
            lay = layers[li]
            want = []
            for n in lay:
                nb = (d_pred if forward else d_succ).get(n, []) or ((d_succ if forward else d_pred).get(n, [])
                                                                     if it >= 6 else [])
                if nb:
                    vals = sorted(yc[m] for m in nb)
                    k = len(vals)
                    want.append(vals[k // 2] if k % 2 else (vals[k // 2 - 1] + vals[k // 2]) / 2)
                else:
                    want.append(yc[n])
            for n, t in zip(lay, _isotonic_tops(lay, want, nodes)):
                yc[n] = t + nodes[n]["sym_h"] / 2
    pos_out: dict[str, tuple[float, float]] = {}
    for li, lay in enumerate(layers):
        lw = max((nodes[n]["foot_w"] for n in lay), default=0.0)
        for n in lay:
            pos_out[n] = (xs[li] + (lw - nodes[n]["foot_w"]) / 2, yc[n])
    # isolated nodes: rows under the layered part
    if isolated:
        base = max((yc[n] - nodes[n]["sym_h"] / 2 + nodes[n]["foot_h"] for n in pos_out), default=-GAP_Y) + GAP_Y + 10
        x = 0.0
        row_h = 0.0
        for i, n in enumerate(isolated):
            if i and i % MAX_ISOLATED_PER_ROW == 0:
                base += row_h + GAP_Y
                x, row_h = 0.0, 0.0
            pos_out[n] = (x, base + nodes[n]["sym_h"] / 2)
            x += nodes[n]["foot_w"] + 40
            row_h = max(row_h, nodes[n]["foot_h"])
    # normalise to top-left (0, 0) and snap symbol centres to the grid
    if pos_out:
        min_top = min(y - nodes[n]["sym_h"] / 2 for n, (x, y) in pos_out.items())
        min_x = min(x for x, _ in pos_out.values())
        pos_out = {n: (round((x - min_x) / G) * G, round((y - min_top) / G) * G) for n, (x, y) in pos_out.items()}
    return pos_out


# ------------------------------------------------------------------------------------------------------ the layout

def _group_order(groups: list[dict], links: list[tuple[str, str]]) -> list[list[str]]:
    gid = [g["id"] for g in groups]
    succ: dict[str, list[str]] = {g: [] for g in gid}
    for a, b in links:
        if a != b and a in succ and b in succ and b not in succ[a]:
            succ[a].append(b)
    back = _break_cycles(gid, succ)
    pred: dict[str, list[str]] = {g: [] for g in gid}
    for a in gid:
        for b in succ[a]:
            if (a, b) not in back:
                pred[b].append(a)
    connected = [g for g in gid if pred[g] or any((a, b) not in back for a in [g] for b in succ[g])]
    layer: dict[str, int] = {}
    remaining = list(connected)
    while remaining:
        progressed = False
        for g in list(remaining):
            if all(p in layer for p in pred[g]):
                layer[g] = max((layer[p] + 1 for p in pred[g]), default=0)
                remaining.remove(g)
                progressed = True
        if not progressed:              # should not happen after cycle breaking
            for g in remaining:
                layer[g] = 0
            break
    cols: list[list[str]] = [[] for _ in range(max(layer.values(), default=-1) + 1)]
    for g in gid:
        if g in layer:
            cols[layer[g]].append(g)
    for ci in range(1, len(cols)):     # barycentre ordering of stacked sections
        prev = {g: i for i, g in enumerate(cols[ci - 1])}
        cols[ci].sort(key=lambda g: (sum(prev.get(p, 0) for p in pred[g]) / max(1, len(pred[g])), gid.index(g)))
    isolated = [g for g in gid if g not in layer]
    if isolated:
        cols.append(["__row__"] + isolated)
    return cols


def _ports(nodes: dict, edges: list[dict], pos: dict) -> None:
    """Port point and outward direction of every edge end, on grid lines, ordered to avoid self-crossings."""
    outs: dict[str, dict[str, list]] = {}
    ins: dict[str, list] = {}
    for e in edges:
        side = "top" if e["overhead"] else "right"
        if e.get("port_hint") in ("top", "right", "bottom"):
            side = "right" if e["port_hint"] == "bottom" else e["port_hint"]
        outs.setdefault(e["src"], {}).setdefault(side, []).append(e)
        tside = "top" if e.get("to_port_hint") == "top" else "left"
        ins.setdefault(e["dst"], []).append((tside, e))
    center = {n: (pos[n][0] + nodes[n]["foot_w"] / 2, pos[n][1]) for n in pos}

    def sym_box(n: str) -> tuple[float, float, float, float]:
        nd = nodes[n]
        cx, cy = center[n]
        if nd["kind"] == "unit":
            s, ox, oy = fit(nd["type"], cx - nd["sym_w"] / 2, cy - nd["sym_h"] / 2, nd["sym_w"], nd["sym_h"])
            vw, vh, _ = SYMBOLS[nd["type"]]
            return ox, oy, vw * s, vh * s
        return cx - nd["sym_w"] / 2, cy - nd["sym_h"] / 2, nd["sym_w"], nd["sym_h"]

    def side_points(n: str, side: str, k: int) -> list[tuple[float, float]]:
        x, y, w, h = sym_box(n)
        nd = nodes[n]
        if side in ("left", "right"):
            lo, hi = y + 4, y + h - 4
            if nd["type"] in _ROUND:
                lo, hi = y + h * 0.25, y + h * 0.75
            if nd["type"] in ("mixer",) and side == "right" or nd["type"] == "splitter" and side == "left":
                lo = hi = y + h / 2
            cands = [g * G for g in range(int(-(-lo // G)), int(hi // G) + 1)] or [round((y + h / 2) / G) * G]
            ys = _spread(cands, k, y + h / 2)
            pts = []
            for yy in ys:
                xx = x if side == "left" else x + w
                if nd["type"] in _ROUND:
                    r = min(w, h) * 0.43
                    cy0 = y + h / 2 if nd["type"] != "pump" else y + h * 20 / 44
                    dx = max(0.0, r * r - (yy - cy0) ** 2) ** 0.5
                    cx0 = x + w / 2
                    xx = cx0 - dx if side == "left" else cx0 + dx
                elif nd["type"] == "mixer" and side == "left":
                    xx = x + 1
                elif nd["type"] == "splitter" and side == "right":
                    xx = x + w - 1
                pts.append((xx, yy))
            return pts
        lo, hi = x + 4, x + w - 4
        cands = [g * G for g in range(int(-(-lo // G)), int(hi // G) + 1)] or [round((x + w / 2) / G) * G]
        xs = _spread(cands, k, x + w / 2)
        pts = []
        for xx in xs:
            yy = y
            if nd["kind"] == "unit" and nd["type"] in ("column", "packed_column", "vessel", "flash_drum", "evaporator",
                                                        "fixed_bed_reactor", "adsorber", "fluidized_bed_reactor"):
                r = w / 2
                t = min(1.0, abs(xx - (x + r)) / r)
                yy = y + 0.11 * h * (1 - (1 - t * t) ** 0.5) if nd["type"] not in ("column", "packed_column") else \
                    y + 9 / 156 * h * (1 - (1 - t * t) ** 0.5)
            pts.append((xx, yy))
        return pts

    for n, by_side in outs.items():
        for side, es in by_side.items():
            other = sorted(es, key=lambda e: (center[e["dst"]][1] if side != "top" else center[e["dst"]][0]))
            pts = side_points(n, side, len(other))
            for e, p in zip(other, pts):
                e["p0"], e["d0"] = p, {"right": (1, 0), "top": (0, -1)}[side]
    for n, lst in ins.items():
        for side in ("left", "top"):
            es = [e for s, e in lst if s == side]
            if not es:
                continue
            es.sort(key=lambda e: center[e["src"]][1] if side == "left" else center[e["src"]][0])
            pts = side_points(n, side, len(es))
            for e, p in zip(es, pts):
                e["p1"], e["d1"] = p, {"left": (-1, 0), "top": (0, -1)}[side]


def _spread(cands: list[float], k: int, mid: float) -> list[float]:
    if k <= 0:
        return []
    if k == 1:
        return [min(cands, key=lambda c: abs(c - mid))]
    if k >= len(cands):
        return [cands[min(len(cands) - 1, i * len(cands) // k)] for i in range(k)]
    return [cands[round(i * (len(cands) - 1) / (k - 1))] for i in range(k)]


class _Router:
    """A* on grid vertices with direction in the state (bend penalty) and soft occupancy costs."""

    BEND, OVERLAP, CROSS, NEAR = 4.0, 14.0, 2.5, 0.6

    def __init__(self, width: float, height: float, obstacles: list[tuple[float, float, float, float]]):
        self.nx, self.ny = int(width // G) + 2, int(height // G) + 2
        self.blocked = bytearray(self.nx * self.ny)
        self.near = bytearray(self.nx * self.ny)
        for (x, y, w, h) in obstacles:
            self._mark(x, y, w, h, self.blocked)
            self._mark(x - G, y - G, w + 2 * G, h + 2 * G, self.near)
        self.occ_h = bytearray(self.nx * self.ny)
        self.occ_v = bytearray(self.nx * self.ny)

    def _mark(self, x: float, y: float, w: float, h: float, arr: bytearray) -> None:
        i0, i1 = max(0, int(-(-x // G))), min(self.nx - 1, int((x + w) // G))
        j0, j1 = max(0, int(-(-y // G))), min(self.ny - 1, int((y + h) // G))
        for j in range(j0, j1 + 1):
            base = j * self.nx
            for i in range(i0, i1 + 1):
                arr[base + i] = 1

    def free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[j * self.nx + i]

    def route(self, a: tuple[int, int], da: tuple[int, int], b: tuple[int, int], db_in: tuple[int, int],
              window: int | None = 30) -> list[tuple[int, int]] | None:
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        di_a, di_b = dirs.index(da), dirs.index(db_in)
        if window is not None:
            i0 = max(0, min(a[0], b[0]) - window)
            i1 = min(self.nx - 1, max(a[0], b[0]) + window)
            j0 = max(0, min(a[1], b[1]) - window)
            j1 = min(self.ny - 1, max(a[1], b[1]) + window)
        else:
            i0, i1, j0, j1 = 0, self.nx - 1, 0, self.ny - 1
        nx = self.nx
        start = (a[1] * nx + a[0]) * 4 + di_a
        dist = {start: 0.0}
        prev: dict[int, int] = {}
        heap = [(abs(a[0] - b[0]) + abs(a[1] - b[1]), 0.0, start)]
        goal_cell = b[1] * nx + b[0]
        while heap:
            f, g, st = heapq.heappop(heap)
            if g > dist.get(st, 1e18):
                continue
            cell, d = divmod(st, 4)
            if cell == goal_cell:
                if d != di_b:
                    g += self.BEND
                path = [st]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                path.reverse()
                return [((s // 4) % nx, (s // 4) // nx) for s in path]
            ci, cj = cell % nx, cell // nx
            for nd, (dx, dy) in enumerate(dirs):
                if nd == (d + 2) % 4:
                    continue
                ni, nj = ci + dx, cj + dy
                if not (i0 <= ni <= i1 and j0 <= nj <= j1):
                    continue
                ncell = nj * nx + ni
                if self.blocked[ncell] and ncell != goal_cell:
                    continue
                cost = 1.0 + (self.BEND if nd != d else 0.0) + (self.NEAR if self.near[ncell] else 0.0)
                horiz = dx != 0
                if (self.occ_h if horiz else self.occ_v)[ncell]:
                    cost += self.OVERLAP
                if (self.occ_v if horiz else self.occ_h)[ncell]:
                    cost += self.CROSS
                ng = g + cost
                ns = ncell * 4 + nd
                if ng < dist.get(ns, 1e18):
                    dist[ns] = ng
                    prev[ns] = st
                    heapq.heappush(heap, (ng + abs(ni - b[0]) + abs(nj - b[1]), ng, ns))
        return None

    def occupy(self, pts: list[tuple[int, int]]) -> None:
        for (i0, j0), (i1, j1) in zip(pts, pts[1:]):
            arr = self.occ_h if j0 == j1 else self.occ_v
            for i in range(min(i0, i1), max(i0, i1) + 1):
                for j in range(min(j0, j1), max(j0, j1) + 1):
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        arr[j * self.nx + i] = 1


def _simplify(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in pts:
        if out and abs(out[-1][0] - p[0]) < 1e-9 and abs(out[-1][1] - p[1]) < 1e-9:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            if (abs(a[0] - b[0]) < 1e-9 and abs(b[0] - p[0]) < 1e-9) or (abs(a[1] - b[1]) < 1e-9 and abs(b[1] - p[1]) < 1e-9):
                out[-1] = p
                continue
        out.append(p)
    return out


def layout(pm: dict, view: str = "full", link_sections: bool | None = None) -> dict:
    """Geometry of the diagram: nodes with symbol/label boxes, section frames, routed edges, drawing size, warnings."""
    pmn, warnings = normalize(pm)
    nodes, edges, groups, w2 = build_view(pmn, view, link_sections)
    warnings += w2
    for n in nodes.values():
        _size(n)
    # feedback set over the whole view (recycles), then group-local layering on forward edges
    order = list(nodes)
    succ: dict[str, list[str]] = {n: [] for n in order}
    for e in edges:
        if e["dst"] not in succ[e["src"]]:
            succ[e["src"]].append(e["dst"])
    back = _break_cycles(order, succ)
    for e in edges:
        e["recycle"] = (e["src"], e["dst"]) in back or e["src"] == e["dst"]
    local: dict[str, dict[str, tuple[float, float]]] = {}
    gsize: dict[str, tuple[float, float]] = {}
    for g in groups:
        ids = [n for n in order if nodes[n]["group"] == g["id"]]
        dag = sorted({(e["src"], e["dst"]) for e in edges if not e["recycle"] and nodes[e["src"]]["group"] == g["id"]
                      and nodes[e["dst"]]["group"] == g["id"]}, key=lambda t: (order.index(t[0]), order.index(t[1])))
        p = _layout_group(nodes, ids, dag)
        local[g["id"]] = p
        w = max((x + nodes[n]["foot_w"] for n, (x, y) in p.items()), default=0.0)
        h = max((y - nodes[n]["sym_h"] / 2 + nodes[n]["foot_h"] for n, (x, y) in p.items()), default=0.0)
        gsize[g["id"]] = (w, h)
    framed = view != "sections"
    pad, title = (SECTION_PAD, SECTION_TITLE_H) if framed else (0.0, 0.0)
    sec_of = {u["id"]: u["section"] for u in pmn["units"]}
    links = [(sec_of[s["from"]], sec_of[s["to"]]) for s in pmn["streams"] if s["from"] and s["to"]]
    cols = _group_order(groups, links) if len(groups) > 1 else [[groups[0]["id"]]]
    frames: dict[str, tuple[float, float, float, float]] = {}
    margin = 40.0
    fsize = {g: (gsize[g][0] + 2 * pad, gsize[g][1] + 2 * pad + title) for g in gsize}
    routed_between = any(nodes[e["src"]]["group"] != nodes[e["dst"]]["group"] for e in edges)
    if routed_between:
        # sections stacked in columns by their flow order (lines between sections are routed)
        row = None
        x = margin
        col_boxes = []
        for col in cols:
            if col and col[0] == "__row__":
                row = col[1:]
                continue
            cw = max(fsize[g][0] for g in col)
            ch = sum(fsize[g][1] for g in col) + SECTION_GAP_Y * (len(col) - 1)
            col_boxes.append((col, x, cw, ch))
            x += cw + SECTION_GAP_X
        max_h = max((c[3] for c in col_boxes), default=0.0)
        for col, cx, cw, ch in col_boxes:
            y = margin + 20 + (max_h - ch) / 2
            for g in col:
                frames[g] = (cx + (cw - fsize[g][0]) / 2, y, *fsize[g])
                y += fsize[g][1] + SECTION_GAP_Y
        x, y = margin, margin + 20 + max_h + SECTION_GAP_Y
        for g in row or []:
            frames[g] = (x, y, *fsize[g])
            x += fsize[g][0] + SECTION_GAP_X
    else:
        # independent sections (off-sheet connectors between them): reading order = flow order, packed in rows
        order_g = [g for col in cols for g in (col[1:] if col and col[0] == "__row__" else col)]
        area = sum((w + SECTION_GAP_X) * (h + SECTION_GAP_Y) for w, h in fsize.values())
        target = max(1600.0, (1.6 * area) ** 0.5, max(w for w, _ in fsize.values()))
        rows: list[list[str]] = [[]]
        acc = 0.0
        for g in order_g:
            if rows[-1] and acc + fsize[g][0] > target:
                rows.append([])
                acc = 0.0
            rows[-1].append(g)
            acc += fsize[g][0] + SECTION_GAP_X
        y = margin + 20
        for r in rows:
            x = margin
            rh = max(fsize[g][1] for g in r)
            for g in r:
                frames[g] = (x, y, *fsize[g])
                x += fsize[g][0] + SECTION_GAP_X
            y += rh + SECTION_GAP_Y
    pos: dict[str, tuple[float, float]] = {}
    for g, p in local.items():
        fx, fy, _, _ = frames[g]
        ox = round((fx + pad) / G) * G
        oy = round((fy + pad + title) / G) * G
        for n, (px, py) in p.items():
            pos[n] = (px + ox, py + oy)
    width = max((fx + fw for fx, fy, fw, fh in frames.values()), default=0.0) + margin
    height = max((fy + fh for fx, fy, fw, fh in frames.values()), default=0.0) + margin + 30
    # boxes
    for n, nd in nodes.items():
        x0, yc = pos[n]
        cx = x0 + nd["foot_w"] / 2
        nd["sym_box"] = [cx - nd["sym_w"] / 2, yc - nd["sym_h"] / 2, nd["sym_w"], nd["sym_h"]]
        nd["label_box"] = [cx - nd["label_w"] / 2, yc + nd["sym_h"] / 2 + 2, nd["label_w"], nd["label_h"]] \
            if nd["label_h"] else None
        nd["foot_box"] = [x0, yc - nd["sym_h"] / 2, nd["foot_w"], nd["foot_h"]]
    # routing
    _ports(nodes, edges, pos)
    obstacles = []
    for nd in nodes.values():
        if nd["kind"] == "dummy":
            continue
        obstacles.append(tuple(nd["sym_box"]))
        if nd["label_box"]:
            obstacles.append(tuple(nd["label_box"]))
    router = _Router(width + 200, height + 200, obstacles)
    failed = []

    def escape(p: tuple[float, float], d: tuple[int, int]) -> tuple[int, int]:
        i, j = (p[0] / G, p[1] / G)
        i = int(i // 1) + (1 if d[0] > 0 else 0) if d[0] else round(i)
        j = int(j // 1) + (1 if d[1] > 0 else 0) if d[1] else round(j)
        for _ in range(60):
            ahead = (i * G - p[0]) * d[0] + (j * G - p[1]) * d[1]
            if router.free(i, j) and ahead >= G * 0.99:
                return i, j
            i, j = i + d[0], j + d[1]
        return i, j

    def span(e: dict) -> float:
        return abs(e["p0"][0] - e["p1"][0]) + abs(e["p0"][1] - e["p1"][1]) + (1e5 if e["recycle"] else 0)

    for e in sorted(edges, key=span):
        a = escape(e["p0"], e["d0"])
        b = escape(e["p1"], e["d1"])
        din = (-e["d1"][0], -e["d1"][1])
        path = router.route(a, e["d0"], b, din, window=30) or router.route(a, e["d0"], b, din, window=None)
        if path is None:
            failed.append(e["id"])
            mid = ((a[0] + b[0]) // 2, a[1])
            path = [a, mid, (mid[0], b[1]), b]
        else:
            router.occupy(path)
        pts = [e["p0"]] + [(i * G, j * G) for i, j in path] + [e["p1"]]
        # keep the stubs orthogonal: the first and last grid points share the port's perpendicular coordinate
        e["points"] = [list(p) for p in _simplify(pts)]
    if failed:
        warnings.append("No se encontró ruta ortogonal libre para: " + ", ".join(failed) + " (se trazó en L).")
    sections = [{"id": g["id"], "label_es": g["label_es"], "box": list(frames[g["id"]])} for g in groups] if framed else []
    return {"name": pmn["name"], "view": view, "model": pmn, "nodes": nodes, "edges": edges, "sections": sections,
            "width": width, "height": height, "warnings": warnings}
