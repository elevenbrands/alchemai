"""Stream extraction for pinch analysis (bloque M5b): turn the heaters, coolers and heat exchangers of a solved
flowsheet into a pinch stream table, with the heat-capacity flow rate segmented where it varies.

Pure Python. The flowsheet side (which units, which streams, the enthalpy flash at a given T and P) is passed in, so
this module can be tested without DWSIM: `packages.dwsim_client.tools_energy.pinch_from_flowsheet` supplies a DWSIM
flash.

Method, per piece (one side of one unit, inlet → outlet):
  * the enthalpy curve H(T) between inlet and outlet is sampled by flashing the piece's own composition and flow at
    intermediate temperatures, with the pressure linear in T between inlet and outlet; the endpoints are pinned to the
    enthalpies the flowsheet reports, so the extracted duty equals ΔH of the real streams;
  * each interval is bisected while its midpoint departs from the chord by more than `tol_pct` % of the piece duty
    (this is what finds a phase change: an isothermal step makes the midpoint test fail until the interval is narrower
    than `MIN_SPAN_K`);
  * the sampled curve is simplified to the fewest chords that stay within the same tolerance (each chord = one
    segment with constant CP = ΔH/ΔT).
Pieces whose outlet stream is the inlet of another piece in the same direction (heat exchanger hot side → cooler) are
chained into one process stream: pinch targets do not change by splitting a stream, but the stream table and the
minimum-units count are those of the real streams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

BALANCE_REL_TOL = 1e-3     # extracted duty (ΔH of the unit's streams) vs the duty DWSIM reports, relative to the duty
CLOSURE_REL_TOL = 1e-9     # Σ segment duties vs the piece duty (exact by construction; guards the code)
ISOTHERMAL_DT_K = 0.05     # a piece with |ΔT| below this is an isothermal phase change
ISOTHERMAL_SPAN_K = 1.0    # ... and is given this span (pinch convention: latent heat over 1 K)
MIN_SPAN_K = 0.1           # bisection stops below this interval width
GRID = 8                   # initial intervals per piece
MAX_LEVELS = 9             # bisection levels (8 · 2⁹ intervals at most per piece)

Flash = Callable[[list[tuple[int, float, float]]], list[float]]
"""flash([(piece_index, T_K, P_Pa), ...]) → enthalpy in W of that piece's material at (T, P), same order."""


@dataclass
class Piece:
    unit: str
    kind: str                       # heater | cooler | heat_exchanger
    side: str                       # "" for heater/cooler, "hot"/"cold" for an exchanger
    inlet: str
    outlet: str
    T_in: float
    T_out: float
    P_in: float
    P_out: float
    H_in: float                     # W, from the solved flowsheet
    H_out: float
    duty_reported_W: float          # |duty| DWSIM reports for the unit (exchanger: Q, same for both sides)
    points: list[tuple[float, float]] = field(default_factory=list)   # (T, H) along the path, inlet → outlet
    aux_endpoint_err_W: float | None = None

    @property
    def Q(self) -> float:
        return abs(self.H_out - self.H_in)

    @property
    def hot(self) -> bool:          # a hot stream gives heat (its enthalpy falls)
        return self.H_out < self.H_in

    @property
    def isothermal(self) -> bool:
        return abs(self.T_out - self.T_in) < ISOTHERMAL_DT_K

    def P_at(self, T: float) -> float:
        if self.isothermal:
            return self.P_in
        return self.P_in + (self.P_out - self.P_in) * (T - self.T_in) / (self.T_out - self.T_in)


def sample(pieces: list[Piece], flash: Flash, tol_pct: float) -> None:
    """Fill piece.points with an adaptive sampling of H(T); endpoints pinned to the flowsheet's enthalpies."""
    # initial grid (endpoints included so the flash can be checked against the flowsheet)
    req = []
    for i, p in enumerate(pieces):
        if p.isothermal or p.Q <= 0:
            p.points = [(p.T_in, p.H_in), (p.T_out, p.H_out)]
            continue
        for k in range(GRID + 1):
            req.append((i, p.T_in + (p.T_out - p.T_in) * k / GRID))
    H = flash([(i, T, pieces[i].P_at(T)) for i, T in req]) if req else []
    grid: dict[int, list[tuple[float, float]]] = {}
    for (i, T), h in zip(req, H):
        grid.setdefault(i, []).append((T, h))
    for i, pts in grid.items():
        p = pieces[i]
        p.aux_endpoint_err_W = max(abs(pts[0][1] - p.H_in), abs(pts[-1][1] - p.H_out))
        p.points = [(p.T_in, p.H_in), *pts[1:-1], (p.T_out, p.H_out)]
    # bisection where the midpoint leaves the chord
    todo = {i: [(a, b) for a, b in zip(range(len(pieces[i].points) - 1), range(1, len(pieces[i].points)))]
            for i in grid}
    for _ in range(MAX_LEVELS):
        mids = []
        for i, ivs in todo.items():
            pts = pieces[i].points
            for a, b in ivs:
                (Ta, _), (Tb, _) = pts[a], pts[b]
                if abs(Tb - Ta) > MIN_SPAN_K:
                    mids.append((i, Ta, Tb))
        if not mids:
            break
        Hm = flash([(i, 0.5 * (Ta + Tb), pieces[i].P_at(0.5 * (Ta + Tb))) for i, Ta, Tb in mids])
        new_todo: dict[int, list[tuple[float, float]]] = {}
        added: dict[int, list[tuple[float, float]]] = {}
        for (i, Ta, Tb), hm in zip(mids, Hm):
            p = pieces[i]
            Ha = dict(p.points)[Ta]
            Hb = dict(p.points)[Tb]
            Tm = 0.5 * (Ta + Tb)
            added.setdefault(i, []).append((Tm, hm))
            if abs(hm - 0.5 * (Ha + Hb)) > tol_pct / 100.0 * p.Q:
                new_todo.setdefault(i, []).extend([(Ta, Tm), (Tm, Tb)])
        for i, extra in added.items():
            p = pieces[i]
            rev = p.T_out < p.T_in
            p.points = sorted(p.points + extra, key=lambda x: x[0], reverse=rev)
        # translate intervals back to indices on the updated point lists
        todo = {}
        for i, ivs in new_todo.items():
            idx = {T: k for k, (T, _) in enumerate(pieces[i].points)}
            todo[i] = [(idx[a], idx[b]) for a, b in ivs]


def simplify(points: list[tuple[float, float]], tol_W: float) -> list[tuple[float, float]]:
    """Fewest breakpoints (greedy) such that every sampled point lies within tol_W of its chord."""
    if len(points) <= 2:
        return list(points)
    out, i = [points[0]], 0
    while i < len(points) - 1:
        j = i + 1
        while j + 1 < len(points) and _chord_ok(points, i, j + 1, tol_W):
            j += 1
        out.append(points[j])
        i = j
    return out


def _chord_ok(pts, i, j, tol):
    (T0, H0), (T1, H1) = pts[i], pts[j]
    if T1 == T0:
        return False
    for T, Hk in pts[i + 1:j]:
        if abs(Hk - (H0 + (H1 - H0) * (T - T0) / (T1 - T0))) > tol:
            return False
    return True


def chain(pieces: list[Piece]) -> list[list[int]]:
    """Group pieces into process streams: piece B follows A when A.outlet is B.inlet and both go the same way."""
    by_inlet = {p.inlet: k for k, p in enumerate(pieces)}
    has_pred = set()
    nxt: dict[int, int] = {}
    for k, p in enumerate(pieces):
        j = by_inlet.get(p.outlet)
        if j is not None and j != k and pieces[j].hot == p.hot and j not in has_pred:
            nxt[k] = j
            has_pred.add(j)
    paths = []
    for k in range(len(pieces)):
        if k in has_pred:
            continue
        path, seen = [k], {k}
        while path[-1] in nxt and nxt[path[-1]] not in seen:
            path.append(nxt[path[-1]])
            seen.add(path[-1])
        paths.append(path)
    return paths


def segments(piece: Piece, tol_pct: float) -> list[tuple[float, float, float]]:
    """(T_supply, T_target, duty_W) segments of one piece, inlet → outlet."""
    if piece.isothermal:
        T = 0.5 * (piece.T_in + piece.T_out)
        h = 0.5 * ISOTHERMAL_SPAN_K
        return [(T + h, T - h, piece.Q) if piece.hot else (T - h, T + h, piece.Q)]
    br = simplify(piece.points, tol_pct / 100.0 * piece.Q)
    return [(Ta, Tb, abs(Hb - Ha)) for (Ta, Ha), (Tb, Hb) in zip(br[:-1], br[1:]) if abs(Hb - Ha) > 0 and Ta != Tb]


def stream_table(pieces: list[Piece], tol_pct: float, dT_contribution_K: dict[str, float] | None = None) -> dict:
    """Process streams (chained pieces), their segments as pinch input, the same streams with one linear CP each,
    and the per-piece balance checks."""
    dtc = dict(dT_contribution_K or {})
    paths = chain(pieces)
    rows, seg_streams, lin_streams, parent_of = [], [], [], {}
    for n, path in enumerate(paths, 1):
        ps = [pieces[k] for k in path]
        hot = ps[0].hot
        name = f"{'H' if hot else 'C'}{n}: {ps[0].inlet}→{ps[-1].outlet}"
        segs = [s for p in ps for s in segments(p, tol_pct)]
        Q = sum(p.Q for p in ps)
        extra = {"dT_contribution_K": float(dtc[name])} if name in dtc else {}
        for k, (Ta, Tb, q) in enumerate(segs, 1):
            sname = name if len(segs) == 1 else f"{name} [{k}/{len(segs)}]"
            parent_of[sname] = name
            seg_streams.append({"name": sname, "T_supply_K": Ta, "T_target_K": Tb, "heat_load_W": q, **extra})
        Ts, Tt = ps[0].T_in, ps[-1].T_out
        if abs(Tt - Ts) < ISOTHERMAL_DT_K:
            Ts, Tt = segs[0][0], segs[-1][1]
        lin_streams.append({"name": name, "T_supply_K": Ts, "T_target_K": Tt, "heat_load_W": Q, **extra})
        cps = [q / abs(Tb - Ta) for Ta, Tb, q in segs]
        rows.append({"name": name, "type": "hot" if hot else "cold", "T_supply_K": Ts, "T_target_K": Tt,
                     "duty_W": Q, "CP_linear_W_K": Q / abs(Tt - Ts), "n_segments": len(segs),
                     "CP_min_W_K": min(cps), "CP_max_W_K": max(cps),
                     "segments": [{"T_supply_K": Ta, "T_target_K": Tb, "duty_W": q, "CP_W_K": q / abs(Tb - Ta)}
                                  for Ta, Tb, q in segs],
                     "units": [f"{p.unit}{'(' + p.side + ')' if p.side else ''}" for p in ps],
                     "path_streams": [ps[0].inlet, *[p.outlet for p in ps]]})
    checks = []
    for p in pieces:
        segQ = sum(q for _, _, q in segments(p, tol_pct))
        rel = abs(p.Q - p.duty_reported_W) / max(p.duty_reported_W, 1.0)
        checks.append({"unit": p.unit, "side": p.side or None, "inlet": p.inlet, "outlet": p.outlet,
                       "dH_streams_W": p.Q, "duty_reported_W": p.duty_reported_W, "rel_err": rel,
                       "ok": rel <= BALANCE_REL_TOL, "segments_sum_W": segQ,
                       "segments_close": abs(segQ - p.Q) <= CLOSURE_REL_TOL * max(p.Q, 1.0),
                       "aux_flash_endpoint_err_W": p.aux_endpoint_err_W})
    return {"rows": rows, "segmented": seg_streams, "linear": lin_streams, "parent_of": parent_of, "checks": checks}


def units_by_parent(pinch_result: dict, parent_of: dict[str, str]) -> dict | None:
    """Euler minimum units counted on the real process streams (a segmented stream is one stream, not several)."""
    if pinch_result.get("threshold_problem") or not pinch_result.get("pinch"):
        return None
    Pm = max(p["T_shifted_K"] for p in pinch_result["pinch"])
    above, below = set(), set()
    for s in pinch_result["streams"]:
        par = parent_of.get(s["name"], s["name"])
        if s["T_shifted_high_K"] > Pm + 1e-9:
            above.add(par)
        if s["T_shifted_low_K"] < Pm - 1e-9:
            below.add(par)
    QH, QC = pinch_result["Q_hot_min_W"], pinch_result["Q_cold_min_W"]
    n_above = max(len(above) + (QH > 1e-9) - 1, 0)
    n_below = max(len(below) + (QC > 1e-9) - 1, 0)
    n = len(set(parent_of.values()))
    return {"above_pinch": n_above, "below_pinch": n_below, "total_MER": n_above + n_below,
            "without_pinch_split": n + (QH > 1e-9) + (QC > 1e-9) - 1}
