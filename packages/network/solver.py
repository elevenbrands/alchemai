"""Steady incompressible pipe networks (bloque M13): global gradient algorithm of Todini & Pilati (1988), the method
EPANET uses, with Hardy Cross (1936) loop corrections as an independent cross-check.

Unknowns are the link flows q (m³/s, positive from `a` to `b`) and the heads H (m) of the junctions; fixed-head nodes
(reservoirs, tanks at their level, fixed-pressure boundaries) close the system. Each Newton step linearises every
link, h_a − h_b = r(q) + r'(q)·Δq, and solves the symmetric node system A·H = F; the flow update then satisfies
continuity at every junction exactly (to round-off), so the mass balance closes on every iteration, not only at the
end. Status changes (check valves, pumps that cannot deliver, PRV/FCV active-open-closed) are decided between Newton
solves, as EPANET does.

Friction is the SAME as the single-pipe tool `pipe_hydraulics`: Darcy-Weisbach with `fluids.friction_factor`
(Clamond's exact Colebrook) above Re 2300 and 64/Re below (`flow_devices._darcy_f`). Hazen-Williams is offered only
because water networks are often specified with C (EPANET's SI coefficient). Pure numerics: no simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from packages.transport.flow_devices import _darcy_f

G = 9.80665
# Continuity tolerance per junction, stated here and reported with every result: |Σq_in − Σq_out − demand| ≤ this.
MASS_TOL_M3_S = 1e-9
FLOW_TOL_REL = 1e-10            # Newton stops when max|Δq| ≤ FLOW_TOL_REL·max|q| + 1e-14
ENERGY_TOL_M = 1e-6             # |h_a − h_b − r(q)| per open link at the solution
RQ_MIN = 1e-7                   # floor of dh/dq (s/m²) so a link never gets an infinite conductance (EPANET RQtol idea)
H_TOL = 1e-4                    # m, hysteresis of status checks
Q_TOL = 1e-9                    # m³/s, reverse-flow threshold of status checks
MAX_NEWTON = 300
MAX_STATUS = 40
# EPANET 2.2 Hazen-Williams: h = 4.727 C^-1.852 d^-4.871 L q^1.852 in ft and cfs, converted exactly to m and m³/s.
HW_SI = 4.727 * 0.3048 ** 4.871 * (0.3048 ** 3) ** -1.852


class NetworkError(ValueError):
    pass


@dataclass
class Fluid:
    rho: float
    mu: float

    @property
    def nu(self) -> float:
        return self.mu / self.rho


@dataclass
class Link:
    id: str
    kind: str                      # pipe | pump | prv | fcv
    a: str
    b: str
    D: float = 0.0
    L: float = 0.0
    eps: float = 0.0
    K: float = 0.0
    K_of_Re: object = None         # callable Re -> K (Darby 3-K fittings) or None
    cv: bool = False
    hw_C: float | None = None
    res_K: float | None = None     # generic resistance h = K·|q|^(n−1)·q (SI: m and m³/s), e.g. textbook K·Q^n data
    res_n: float = 2.0
    pump: object = None            # PumpCurve
    setting: float | None = None   # PRV: head (m) at b; FCV: flow (m³/s)
    status: str = "open"           # open | closed | active

    @property
    def area(self) -> float:
        return math.pi * self.D ** 2 / 4.0


@dataclass
class PumpCurve:
    """Head gain of one pump at speed ratio w: gain(q) = w²·h(q/w) (affinity laws)."""
    kind: str                      # power: h0 − r·q^n | poly: c0 + c1 q + c2 q²
    coef: tuple
    w: float = 1.0
    q_range: tuple = (0.0, math.inf)
    source: str = ""

    def base(self, q: float) -> tuple[float, float]:
        if self.kind == "power":
            h0, r, n = self.coef
            qa = max(q, 0.0)
            return h0 - r * qa ** n, -r * n * qa ** (n - 1.0) if qa > 0 else (-r if n == 1 else 0.0)
        c0, c1, c2 = self.coef
        return c0 + c1 * q + c2 * q * q, c1 + 2.0 * c2 * q

    def gain(self, q: float) -> tuple[float, float]:
        h, dh = self.base(q / self.w)
        return self.w ** 2 * h, self.w * dh

    @property
    def shutoff(self) -> float:
        return self.gain(0.0)[0]


def fit_pump_curve(points: list[list[float]] | None = None, coefficients: dict | None = None,
                   speed_ratio: float = 1.0) -> PumpCurve:
    """Curve from the user's points (q m³/s, h m) or coefficients {h0_m, r, n}. 1 point: EPANET's rule (shut-off at
    133 % of the design head, zero head at 200 % of the design flow); 3 points starting at q = 0: EPANET's exact
    power fit h = A − B·q^C; otherwise least squares h = c0 + c1·q + c2·q²."""
    if coefficients:
        h0, r, n = float(coefficients["h0_m"]), float(coefficients["r"]), float(coefficients.get("n", 2.0))
        return PumpCurve("power", (h0, r, n), speed_ratio, (0.0, (h0 / r) ** (1 / n) if r > 0 else math.inf),
                         "coeficientes del usuario h = h0 − r·q^n")
    pts = sorted((float(q), float(h)) for q, h in points)
    if len(pts) == 1:
        q1, h1 = pts[0]
        return PumpCurve("power", (4.0 / 3.0 * h1, h1 / 3.0 / q1 ** 2, 2.0), speed_ratio, (0.0, 2.0 * q1),
                         "1 punto: regla de EPANET 2.2 (h0 = 4/3·h, caudal máximo = 2·q); SUPUESTO de forma, no dato")
    if len(pts) == 2:
        raise NetworkError("curva de bomba con 2 puntos: dar 1, 3 o más puntos, o coeficientes {h0_m, r, n}")
    if len(pts) == 3 and pts[0][0] == 0.0:
        (_, h0), (q1, h1), (q2, h2) = pts
        if not (h0 > h1 > h2):
            raise NetworkError("curva de bomba: la carga debe bajar con el caudal")
        C = math.log((h0 - h2) / (h0 - h1)) / math.log(q2 / q1)
        B = (h0 - h1) / q1 ** C
        return PumpCurve("power", (h0, B, C), speed_ratio, (0.0, pts[-1][0]),
                         "3 puntos con q = 0: ajuste exacto h = A − B·q^C (como EPANET)")
    q = np.array([p[0] for p in pts])
    h = np.array([p[1] for p in pts])
    c2, c1, c0 = np.polyfit(q, h, 2)
    dev = float(np.max(np.abs(c0 + c1 * q + c2 * q * q - h)))
    return PumpCurve("poly", (float(c0), float(c1), float(c2)), speed_ratio, (float(q[0]), float(q[-1])),
                     f"{len(pts)} puntos: mínimos cuadrados cuadráticos (desviación máx. {dev:.3g} m)")


# ------------------------------------------------------------------------------------------------ link physics
def pipe_loss(lk: Link, q: float, fl: Fluid) -> tuple[float, float, dict]:
    """Signed head loss r(q) (m) of a pipe or open valve and dr/dq, plus Re and f (same f as pipe_hydraulics)."""
    if lk.res_K is not None:
        h = lk.res_K * q * abs(q) ** (lk.res_n - 1.0)
        return h, max(lk.res_n * lk.res_K * abs(q) ** (lk.res_n - 1.0), RQ_MIN), {"Re": None, "f": None, "K": 0.0}
    A = lk.area
    v = q / A
    Re = abs(v) * lk.D / fl.nu
    K = lk.K + (lk.K_of_Re(Re) if (lk.K_of_Re and Re > 0) else 0.0)
    hK = K * v * abs(v) / (2 * G)
    dK = 2.0 * K * abs(v) / (2 * G) / A
    if lk.L <= 0.0:
        return hK, max(dK, RQ_MIN), {"Re": Re, "f": None, "K": K}
    if lk.hw_C is not None:
        r = HW_SI * lk.L / (lk.hw_C ** 1.852 * lk.D ** 4.871)
        hf = r * q * abs(q) ** 0.852
        df = 1.852 * r * abs(q) ** 0.852
        return hf + hK, max(df + dK, RQ_MIN), {"Re": Re, "f": None, "K": K}
    if Re < 2300.0:
        lam = 32.0 * fl.nu * lk.L / (G * lk.D ** 2 * A)       # h_f = lam·q exactly when f = 64/Re
        f = 64.0 / Re if Re > 0 else None
        return lam * q + hK, max(lam + dK, RQ_MIN), {"Re": Re, "f": f, "K": K}
    eD = lk.eps / lk.D
    f = _darcy_f(Re, eD)
    s = math.log(_darcy_f(Re * 1.0001, eD) / f) / math.log(1.0001)     # d ln f / d ln Re within the branch
    hf = f * lk.L / lk.D * v * abs(v) / (2 * G)
    return hf + hK, max((2.0 + s) * abs(hf / q) + dK, RQ_MIN), {"Re": Re, "f": f, "K": K}


def link_loss(lk: Link, q: float, fl: Fluid) -> tuple[float, float]:
    if lk.kind == "pump":
        g, dg = lk.pump.gain(q)
        return -g, max(-dg, RQ_MIN)
    h, d, _ = pipe_loss(lk, q, fl)
    return h, d


# ------------------------------------------------------------------------------------------------ GGA solver
@dataclass
class Network:
    fluid: Fluid
    junctions: dict            # id -> {"z": m, "d": m³/s}
    fixed: dict                # id -> {"z": m, "H": m}
    links: list = field(default_factory=list)


def _components(net: Network, active: list[Link], pinned: set) -> tuple[set, set]:
    """Junctions connected (through non-closed links) to a fixed head or a PRV-pinned node, and the isolated ones."""
    parent = {n: n for n in [*net.junctions, *net.fixed]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for lk in active:
        parent[find(lk.a)] = find(lk.b)
    roots = {find(n) for n in [*net.fixed, *pinned]}
    conn = {n for n in net.junctions if find(n) in roots}
    return conn, set(net.junctions) - conn


def _newton(net: Network, q: dict, H: dict) -> dict:
    """Newton (Todini-Pilati) at fixed statuses. Updates q and H in place; returns iteration info."""
    fl = net.fluid
    links = net.links
    prv = [lk for lk in links if lk.kind == "prv" and lk.status == "active"]
    pinned = {lk.b: lk.setting for lk in prv}
    flowing = [lk for lk in links if lk.status != "closed"]
    conn, iso = _components(net, flowing, set(pinned))
    for lk in links:
        if lk.status == "closed":
            q[lk.id] = 0.0
    for n in iso:
        if abs(net.junctions[n]["d"]) > 0:
            raise NetworkError(f"el nodo {n} queda aislado de toda carga fija (enlaces cerrados) y tiene demanda")
        H[n] = None
    unknown = [n for n in net.junctions if n in conn and n not in pinned]
    idx = {n: i for i, n in enumerate(unknown)}
    known = {**{n: v["H"] for n, v in net.fixed.items()}, **pinned}
    for n, h in pinned.items():
        H[n] = h
    fixed_flow = {lk.id for lk in links if lk.status == "active"}      # active FCV (setting) and PRV (lagged)
    newton_links = [lk for lk in flowing if lk.id not in fixed_flow]
    damp, hist = 1.0, []
    for it in range(1, MAX_NEWTON + 1):
        n = len(unknown)
        A = np.zeros((n, n))
        F = np.zeros(n)
        for j in unknown:
            F[idx[j]] -= net.junctions[j]["d"]
        coeff = {}
        for lk in newton_links:
            r, dr = link_loss(lk, q[lk.id], fl)
            p = 1.0 / dr
            y = p * r
            coeff[lk.id] = (p, y)
            for node, sgn in ((lk.a, -1.0), (lk.b, 1.0)):     # flow (q − y) leaves a, enters b
                if node in idx:
                    i = idx[node]
                    A[i, i] += p
                    F[i] += sgn * (q[lk.id] - y)
                    other = lk.b if node == lk.a else lk.a
                    if other in idx:
                        A[i, idx[other]] -= p
                    elif other in known:
                        F[i] += p * known[other]
        for lk in links:
            if lk.id in fixed_flow:
                for node, sgn in ((lk.a, -1.0), (lk.b, 1.0)):
                    if node in idx:
                        F[idx[node]] += sgn * q[lk.id]
        try:
            Hs = np.linalg.solve(A, F) if n else np.zeros(0)
        except np.linalg.LinAlgError as e:
            raise NetworkError(f"sistema singular (¿nodo sin camino a una carga fija?): {e}")
        for j in unknown:
            H[j] = float(Hs[idx[j]])
        head = lambda m: H[m] if m in H and H[m] is not None else known[m]
        dq_max, q_max = 0.0, 0.0
        for lk in newton_links:
            p, y = coeff[lk.id]
            qn = q[lk.id] - y + p * (head(lk.a) - head(lk.b))
            dq = qn - q[lk.id]
            q[lk.id] += damp * dq
            dq_max = max(dq_max, abs(dq))
            q_max = max(q_max, abs(q[lk.id]))
        for lk in prv:                     # lagged PRV flow: whatever node b still needs
            qn = _node_need(net, lk.b, q, exclude=lk.id)
            dq_max = max(dq_max, abs(qn - q[lk.id]))
            q[lk.id] = qn
        hist.append(dq_max)
        if dq_max <= FLOW_TOL_REL * q_max + 1e-14:
            return {"iterations": it, "max_dq": dq_max}
        if it > 40 and it % 20 == 0 and hist[-1] > 0.5 * hist[-20]:
            damp = max(0.25, damp * 0.5)            # f(Re) jump at Re 2300 can make Newton cycle: relax
    raise NetworkError(f"Newton no convergió en {MAX_NEWTON} iteraciones (max|Δq| = {hist[-1]:.3g} m³/s)")


def _node_need(net: Network, node: str, q: dict, exclude: str) -> float:
    need = net.junctions[node]["d"]
    for lk in net.links:
        if lk.id == exclude or lk.status == "closed":
            continue
        if lk.a == node:
            need += q[lk.id]
        elif lk.b == node:
            need -= q[lk.id]
    return need


def _update_status(net: Network, q: dict, H: dict) -> list[str]:
    changes = []
    known = {n: v["H"] for n, v in net.fixed.items()}
    head = lambda m: H[m] if H.get(m) is not None else known.get(m)

    def set_(lk, st, why):
        if lk.status != st:
            changes.append(f"{lk.id}: {lk.status} → {st} ({why})")
            lk.status = st
    for lk in net.links:
        ha, hb = head(lk.a), head(lk.b)
        if ha is None or hb is None:
            continue
        if lk.kind == "pipe" and lk.cv:
            if lk.status == "open" and q[lk.id] < -Q_TOL:
                set_(lk, "closed", "flujo inverso en la check")
            elif lk.status == "closed" and ha - hb > H_TOL:
                set_(lk, "open", "carga a favor")
        elif lk.kind == "pump":
            if lk.status == "open" and q[lk.id] < -Q_TOL:
                set_(lk, "closed", "la bomba no vence la carga (flujo inverso)")
            elif lk.status == "closed" and hb - ha < lk.pump.shutoff - H_TOL:
                set_(lk, "open", "carga requerida bajo la de cierre")
        elif lk.kind == "prv":
            s = lk.setting
            if lk.status == "active":
                if q[lk.id] < -Q_TOL:
                    set_(lk, "closed", "aguas abajo ya está sobre el ajuste")
                elif ha < s - H_TOL:
                    set_(lk, "open", "aguas arriba bajo el ajuste")
            elif lk.status == "open":
                if q[lk.id] < -Q_TOL:
                    set_(lk, "closed", "flujo inverso")
                elif hb > s + H_TOL:
                    set_(lk, "active", "aguas abajo sobre el ajuste")
            else:
                if ha > s + H_TOL and hb < s - H_TOL:
                    set_(lk, "active", "puede regular")
                elif ha > hb + H_TOL and ha < s:
                    set_(lk, "open", "aguas arriba bajo el ajuste")
        elif lk.kind == "fcv":
            if lk.status == "active" and ha - hb < -H_TOL:
                set_(lk, "open", "no hay carga para dar el caudal de ajuste")
            elif lk.status == "open" and q[lk.id] > lk.setting + Q_TOL:
                set_(lk, "active", "el caudal supera el ajuste")
    return changes


def solve(net: Network, q0: dict | None = None) -> dict:
    """Solve the network. Returns q, H, statuses and convergence evidence."""
    if not net.fixed:
        raise NetworkError("la red necesita al menos un nodo de carga fija (reservorio, tanque o presión fija)")
    q = dict(q0 or {})
    for lk in net.links:
        if lk.id in q:
            continue
        if lk.kind == "pump":
            q[lk.id] = 0.5 * min(lk.pump.q_range[1], 1.0) * lk.pump.w if math.isfinite(lk.pump.q_range[1]) else 0.01
        elif lk.kind == "fcv":
            q[lk.id] = lk.setting
        else:
            q[lk.id] = 0.3 * lk.area if lk.D > 0 else 1e-3  # 0.3 m/s starting guess (numerical only)
    H: dict = {}
    log, total_it = [], 0
    for lk in net.links:
        if lk.kind in ("prv", "fcv"):
            lk.status = "active"
    seen = set()
    for outer in range(1, MAX_STATUS + 1):
        info = _newton(net, q, H)
        total_it += info["iterations"]
        ch = _update_status(net, q, H)
        log += ch
        if not ch:
            break
        key = tuple(lk.status for lk in net.links)
        if key in seen:
            raise NetworkError("los estados de válvulas/bombas ciclan sin converger: " + "; ".join(ch))
        seen.add(key)
    else:
        raise NetworkError("estados sin converger en " + str(MAX_STATUS) + " pasadas")
    return {"q": q, "H": H, "status": {lk.id: lk.status for lk in net.links}, "newton_iterations": total_it,
            "status_passes": outer, "status_log": log}


def node_residuals(net: Network, q: dict) -> dict:
    """Continuity residual per junction (m³/s): Σq_in − Σq_out − demand."""
    res = {n: -v["d"] for n, v in net.junctions.items()}
    for lk in net.links:
        if lk.a in res:
            res[lk.a] -= q[lk.id]
        if lk.b in res:
            res[lk.b] += q[lk.id]
    return res


def energy_residuals(net: Network, q: dict, H: dict) -> dict:
    """|h_a − h_b − r(q)| (m) for every open, Newton-solved link at the solution."""
    known = {n: v["H"] for n, v in net.fixed.items()}
    head = lambda m: H[m] if H.get(m) is not None else known.get(m)
    out = {}
    for lk in net.links:
        if lk.status != "open" or head(lk.a) is None or head(lk.b) is None:
            continue
        r, _ = link_loss(lk, q[lk.id], net.fluid)
        out[lk.id] = abs(head(lk.a) - head(lk.b) - r)
    return out


# ------------------------------------------------------------------------------------------------ Hardy Cross
def hardy_cross(net: Network, tol: float = 1e-11, max_iter: int = 5000) -> dict:
    """Classic Hardy Cross with loop flow corrections ΔQ = −Σh / Σ(dh/dq), sequential over a fundamental loop set
    (and pseudo-loops between fixed-head nodes). Only for networks of pipes and pumps with known open links: it is
    the independent cross-check of the GGA solution, not the production solver."""
    fl = net.fluid
    links = [lk for lk in net.links if lk.status != "closed"]
    if any(lk.kind in ("prv", "fcv") for lk in links):
        return {"aplica": False, "motivo": "Hardy Cross no maneja PRV/FCV activas; solo se contrasta sin ellas"}
    ROOT = "__root__"
    adj: dict = {n: [] for n in [*net.junctions, *net.fixed, ROOT]}
    for lk in links:
        adj[lk.a].append((lk.b, lk))
        adj[lk.b].append((lk.a, lk))
    parent, order, seen = {ROOT: None}, [ROOT], {ROOT}
    for f in net.fixed:
        parent[f] = (ROOT, None)
        seen.add(f)
        order.append(f)
    i = 1
    while i < len(order):
        u = order[i]
        i += 1
        for v, lk in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v] = (u, lk)
                order.append(v)
    if len(seen) < len(adj):
        return {"aplica": False, "motivo": "hay nodos sin camino a una carga fija"}
    tree = {p[1].id for n, p in parent.items() if p and p[1] is not None}
    chords = [lk for lk in links if lk.id not in tree]
    q = {lk.id: 0.0 for lk in links}
    for lk in chords:
        q[lk.id] = (0.5 * lk.pump.q_range[1] if lk.kind == "pump" and math.isfinite(lk.pump.q_range[1])
                    else (0.3 * lk.area if lk.D > 0 else 1e-3))
    # tree flows from the leaves so that every junction balances with the chord guesses
    excess = {n: (net.junctions[n]["d"] if n in net.junctions else 0.0) for n in adj}
    for lk in chords:
        excess[lk.a] += q[lk.id]
        excess[lk.b] -= q[lk.id]
    for n in reversed(order[1:]):
        if n in net.fixed:
            continue
        u, lk = parent[n]
        need = excess[n]                  # flow that must arrive at n through its tree link
        q[lk.id] = need if lk.b == n else -need
        excess[u] += need
    depth = {ROOT: 0}
    for n in order[1:]:
        depth[n] = depth[parent[n][0]] + 1

    def path_up(n):
        out = []
        while parent[n] is not None and parent[n][1] is not None:
            u, lk = parent[n]
            out.append((lk, 1.0 if lk.a == u else -1.0))   # orientation u -> n
            n = u
        return out, n
    loops = []
    for c in chords:                       # loop: a -> b along the chord, then b -> a back through the tree
        pa, ra = path_up(c.a)
        pb, rb = path_up(c.b)
        ida = {lk.id for lk, _ in pa}
        common = [lk.id for lk, _ in pb if lk.id in ida]
        pa = [x for x in pa if x[0].id not in common]
        pb = [x for x in pb if x[0].id not in common]
        # going b -> up -> ... -> down -> a: up the b branch reverses its downward orientation
        terms = [(c, 1.0)] + [(lk, -s) for lk, s in pb] + [(lk, s) for lk, s in reversed(pa)]
        dH = net.fixed[ra]["H"] - net.fixed[rb]["H"] if ra != rb else 0.0   # pseudo-loop: Σh = H_topA − H_topB
        loops.append((terms, dH))
    for it in range(1, max_iter + 1):
        worst = 0.0
        for terms, dH in loops:
            E, dE = -dH, 0.0
            for lk, s in terms:
                r, dr = link_loss(lk, q[lk.id], fl)
                E += s * r
                dE += dr
            dQ = -E / dE
            for lk, s in terms:
                q[lk.id] += s * dQ
            worst = max(worst, abs(dQ))
        if worst < tol:
            break
    H = {n: v["H"] for n, v in net.fixed.items()}
    for n in order[1:]:
        if n in net.fixed:
            continue
        u, lk = parent[n]
        r, _ = link_loss(lk, q[lk.id], fl)
        H[n] = H[u] - r if lk.a == u else H[u] + r
    return {"aplica": True, "q": q, "H": H, "iterations": it, "loops": len(loops), "converged": worst < tol}
