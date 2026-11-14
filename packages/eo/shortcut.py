"""McCabe-Thiele feasibility of a binary NRTL column (bloque M15b), independent of the IDAES model.

Why: `eo_column_optimize` minimises the reboiler duty, so with a fixed number of trays its optimum sits close to minimum
reflux, where a cold start of the equation-oriented model is fragile. Before the NLP is built this module answers, with
the same NRTL parameters and Psat but its own code (thermo.nrtl.NRTL γ through nrtl_properties._bubble_binary):

  * the product pair (x_D, x_B) the two specs imply (component balance);
  * N_min, the equilibrium stages at total reflux (operating line y = x: no enthalpy assumption, so rigorous for this
    VLE) — fewer stages than that and the specs are unreachable at any reflux;
  * R_min under constant molar overflow (feed pinch or tangent pinch), and R_needed, the reflux at which the user's
    trays and feed tray just make the pair (stage stepping from the top, feed mixed into the feed tray);
  * a consistent start (RR, BR) a margin above R_needed, whose products are purer than the specs: an interior point.

Constant molar overflow is an approximation (the molar latent heats of the two components are taken as equal and the
heat of mixing is ignored): R_min and R_needed are estimates and are labelled so. Only the N_min refusal is rigorous.
Stage numbering = the IDAES TrayColumn: total condenser, trays 1..N from the top, partial reboiler = stage N + 1.
"""
from __future__ import annotations

import numpy as np

N_GRID = 401
MAX_STAGES_TOTAL_REFLUX = 500
START_MARGINS = (1.25, 1.6, 1.1, 2.5)     # multiples of R_needed reported as consistent (RR, BR) pairs
# Limit of the EO tool, measured (M15b, ethanol-water, Kamihama 2012 parameters, 20 trays, 101325 Pa): a distillate with
# T_dew − T_bubble = 0.030 K (x 0.82) converges and passes the balance/phantom-vapour seal; 0.021 K (x 0.83) does not.
MIN_DEW_BUBBLE_GAP_K = 0.03
# R_needed assumes constant molar overflow (measured against the rigorous optimum on ethanol-water: −1 % to −5 %), so a
# reflux bound is only called insufficient when not even this multiple of it reaches the products on the diagram.
CMO_REFUSAL_MARGIN = 1.25


def spec_pair(feed_z: dict, specs: list[dict], light: str, heavy: str) -> dict | None:
    """(x_D, x_B) of the light component and D/F implied by exactly two specs taken as equalities; None when the specs
    are not one of the combinations below (the shortcut is then skipped, nothing is refused)."""
    if len(specs) != 2:
        return None
    z = float(feed_z[light])
    k = {(s["stream"], s["component"], s["kind"]): float(s["min"]) for s in specs}
    xd = k.get(("distillate", light, "mole_fraction"))
    xb_h = k.get(("bottoms", heavy, "mole_fraction"))
    rl = k.get(("distillate", light, "recovery"))
    rh = k.get(("bottoms", heavy, "recovery"))
    try:
        if xd is not None and xb_h is not None:
            xb = 1.0 - xb_h
        elif xd is not None and rl is not None:
            dF = rl * z / xd
            xb = z * (1.0 - rl) / (1.0 - dF)
        elif xb_h is not None and rh is not None:
            xb = 1.0 - xb_h
            bF = rh * (1.0 - z) / xb_h
            xd = (z - bF * xb) / (1.0 - bF)
        elif rl is not None and xb_h is not None:
            xb = 1.0 - xb_h
            bF = (1.0 - rl) * z / xb
            xd = rl * z / (1.0 - bF)
        elif rl is not None and rh is not None:
            dF = rl * z + (1.0 - rh) * (1.0 - z)
            xd = rl * z / dF
            xb = (1.0 - rl) * z / (1.0 - dF)
        else:
            return None
    except ZeroDivisionError:
        return None
    if not (0.0 < xb < z < xd < 1.0):
        return {"x_D": xd, "x_B": xb, "consistent": False}
    return {"x_D": xd, "x_B": xb, "D_over_F": (z - xb) / (xd - xb), "consistent": True}


class BinaryVLE:
    """y(x), T(x) of the light component at P on the feed's side of any azeotrope (monotone there)."""

    def __init__(self, params: dict, P_Pa: float, light: str, heavy: str, x_lo: float, x_hi: float):
        from scipy.interpolate import PchipInterpolator
        from packages.eo import nrtl_properties as NP
        sub = NP.sub_parameters(params, [light, heavy])
        ps = NP._psat([light, heavy])
        self.x_lo, self.x_hi = x_lo, x_hi
        xs = np.unique(np.concatenate([np.linspace(x_lo, x_hi, N_GRID),
                                       x_lo + (x_hi - x_lo) * np.geomspace(1e-6, 1e-2, 40),
                                       x_hi - (x_hi - x_lo) * np.geomspace(1e-6, 1e-2, 40)]))
        xs = xs[np.concatenate([[True], np.diff(xs) > 1e-9 * (x_hi - x_lo)])]     # drop round-off duplicates
        T, y = [], []
        for x in xs:
            xx = min(max(float(x), 1e-12), 1 - 1e-12)
            t, _, g = NP._bubble_binary(sub, [light, heavy], ps, P_Pa, xx)
            T.append(t)
            y.append(xx * g[0] * ps[light](t) / P_Pa)
        self.xs, self.ys, self.Ts = xs, np.array(y), np.array(T)
        self.monotone = bool(np.all(np.diff(self.ys) > 0))
        self.y = PchipInterpolator(xs, self.ys)
        self.T = PchipInterpolator(xs, self.Ts)
        self._x_of_y = PchipInterpolator(self.ys, xs) if self.monotone else None

    def x_of_y(self, y: float) -> float:
        y = min(max(float(y), float(self.ys[0])), float(self.ys[-1]))
        return float(self._x_of_y(y))


def feed_quality(vle: BinaryVLE, light: str, heavy: str, z: float, T_feed: float) -> dict:
    """q = moles of liquid added to the stripping section per mole of feed. Two-phase: lever rule on the T-x-y curve.
    Subcooled/superheated: 1 + cp_L·(T_b − T_F)/λ or −cp_V·(T_F − T_d)/λ with thermo's liquid/ideal-gas cp and
    DIPPR ΔHvap, mole-averaged at the feed composition."""
    from scipy.optimize import brentq
    from thermo import EnthalpyVaporization, HeatCapacityGas, HeatCapacityLiquid
    from packages.eo.nrtl_properties import _cas
    Tb = float(vle.T(z))
    # dew temperature of the feed: the liquid x whose equilibrium vapour is z
    xdew = vle.x_of_y(z)
    Td = float(vle.T(xdew))
    zs = {light: z, heavy: 1.0 - z}
    lam = sum(zs[c] * float(EnthalpyVaporization(CASRN=_cas(c))(Tb)) for c in zs)
    if T_feed <= Tb:
        cp = sum(zs[c] * float(HeatCapacityLiquid(CASRN=_cas(c))(0.5 * (Tb + T_feed))) for c in zs)
        q = 1.0 + cp * (Tb - T_feed) / lam
        state = "líquido subenfriado" if T_feed < Tb - 1e-6 else "líquido saturado"
    elif T_feed >= Td:
        cp = sum(zs[c] * float(HeatCapacityGas(CASRN=_cas(c))(0.5 * (Td + T_feed))) for c in zs)
        q = -cp * (T_feed - Td) / lam
        state = "vapor sobrecalentado" if T_feed > Td + 1e-6 else "vapor saturado"
    else:
        xl = brentq(lambda x: float(vle.T(x)) - T_feed, xdew, z, xtol=1e-12)
        yl = float(vle.y(xl))
        q = (yl - z) / (yl - xl)
        state = "dos fases"
    return {"q": float(q), "T_bubble_K": Tb, "T_dew_K": Td, "state": state}


def min_stages_total_reflux(vle: BinaryVLE, xd: float, xb: float) -> int | None:
    """Equilibrium stages (trays + reboiler) at total reflux from y_1 = x_D down to x ≤ x_B; None if the stepping
    stalls (pinch at an azeotrope or tangency)."""
    y = xd
    for n in range(1, MAX_STAGES_TOTAL_REFLUX + 1):
        x = vle.x_of_y(y)
        if x <= xb:
            return n
        if y - x < 1e-9:
            return None
        y = x
    return None


def _lines(R: float, xd: float, xb: float, z: float, q: float, dF: float):
    """Rectifying and stripping operating lines (slope, intercept) and V̄/B under constant molar overflow, per unit feed."""
    D, B = dF, 1.0 - dF
    L, V = R * D, (R + 1.0) * D
    Ls, Vs = L + q, V - (1.0 - q)
    return (L / V, D * xd / V), (Ls / Vs if Vs > 0 else float("inf"), -B * xb / Vs if Vs > 0 else 0.0), Vs / B


def _feasible(vle: BinaryVLE, R: float, xd: float, xb: float, z: float, q: float, dF: float) -> bool:
    """Both operating lines strictly below the equilibrium curve between x_B and x_D (covers tangent pinches)."""
    (ar, br), (as_, bs), vsb = _lines(R, xd, xb, z, q, dF)
    if vsb <= 0:
        return False
    # intersection of the two lines
    xi = (bs - br) / (ar - as_) if abs(ar - as_) > 1e-14 else z
    xs = np.linspace(xb, xd, 1200)[1:-1]
    yop = np.where(xs >= xi, ar * xs + br, as_ * xs + bs)
    return bool(np.all(vle.y(xs) - yop > 1e-9))


def min_reflux(vle: BinaryVLE, xd: float, xb: float, z: float, q: float, dF: float, r_hi: float = 200.0) -> float | None:
    if not _feasible(vle, r_hi, xd, xb, z, q, dF):
        return None
    lo, hi = 1e-6, r_hi
    if _feasible(vle, lo, xd, xb, z, q, dF):
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _feasible(vle, mid, xd, xb, z, q, dF):
            hi = mid
        else:
            lo = mid
    return hi


def step_stages(vle: BinaryVLE, R: float, n_trays: int, feed_tray: int, xd: float, xb: float, z: float, q: float,
                dF: float) -> dict:
    """Liquid composition leaving each stage (trays 1..N, reboiler N + 1) for reflux ratio R with the operating lines
    drawn to the target pair. `x_reboiler` ≤ x_B means the trays are enough at this reflux."""
    (ar, br), (as_, bs), vsb = _lines(R, xd, xb, z, q, dF)
    if vsb <= 0:
        return {"x": [], "x_reboiler": 1.0, "boilup_ratio": vsb}
    xs, y = [], xd
    for n in range(1, n_trays + 2):
        if y <= float(vle.ys[0]):
            xs.append(float(vle.x_lo))
            y = float(vle.ys[0])
            continue
        x = vle.x_of_y(y)
        xs.append(x)
        a, b = (ar, br) if n < feed_tray else (as_, bs)
        y = a * x + b
    return {"x": xs, "x_reboiler": xs[-1], "boilup_ratio": vsb}


def reflux_needed(vle: BinaryVLE, n_trays: int, feed_tray: int, xd: float, xb: float, z: float, q: float, dF: float,
                  r_min: float, r_hi: float) -> float | None:
    """Smallest R in (r_min, r_hi] at which the stepping reaches x_B in the reboiler; None if not even r_hi does."""
    f = lambda R: step_stages(vle, R, n_trays, feed_tray, xd, xb, z, q, dF)["x_reboiler"] - xb  # noqa: E731
    if f(r_hi) > 0:
        return None
    lo, hi = r_min, r_hi
    if f(lo * (1 + 1e-9)) <= 0:
        return lo
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return hi


def binary_shortcut(params: dict, P_Pa: float, feed: dict, specs: list[dict], n_trays: int, feed_tray: int,
                    rr_bounds: tuple[float, float], br_bounds: tuple[float, float],
                    azeotropes: list[dict] | None = None) -> dict | None:
    """McCabe-Thiele summary for a binary NRTL column, or None when it does not apply (not binary, specs that do not
    fix a product pair). Keys: light, heavy, x_D, x_B, D_over_F, feed (q…), N_min, R_min_CMO, R_needed_CMO, starts
    [(RR, BR)], refusals [Spanish reasons]. `refusals` non-empty = physically unreachable with this column."""
    names = list(params["names"])
    if len(names) != 2:
        return None
    from packages.eo import nrtl_properties as NP
    ps = NP._psat(names)
    z0 = float(feed["z"][names[0]])
    alpha = NP._bubble_binary(params, names, ps, P_Pa, min(max(z0, 1e-9), 1 - 1e-9))[1]
    light, heavy = (names[0], names[1]) if alpha > 1.0 else (names[1], names[0])
    z = float(feed["z"][light])
    pair = spec_pair(feed["z"], specs, light, heavy)
    if pair is None:
        return None
    out = {"method": "McCabe-Thiele con el mismo NRTL y Psat (thermo), flujo molar constante para R_min y R_needed",
           "light": light, "heavy": heavy, "x_D": pair["x_D"], "x_B": pair["x_B"], "refusals": [], "starts": []}
    if not pair["consistent"]:
        out["refusals"].append(
            f"las dos specs implican x_D({light}) = {pair['x_D']:.4f} y x_B({light}) = {pair['x_B']:.4f} con una "
            f"alimentación de {z:.4f}: el balance por componente no admite esos productos")
        return out
    # the feed's side of any azeotrope (azeotrope_refusals has already refused specs across it)
    x_lo, x_hi = 0.0, 1.0
    for a in azeotropes or []:
        xa = a["x_az"][light]
        if xa > z:
            x_hi = min(x_hi, xa)
        else:
            x_lo = max(x_lo, xa)
    vle = BinaryVLE(params, P_Pa, light, heavy, x_lo, x_hi)
    if not vle.monotone:
        return None
    xd, xb, dF = pair["x_D"], pair["x_B"], pair["D_over_F"]
    out["D_over_F"] = dF
    fq = feed_quality(vle, light, heavy, z, float(feed["T_K"]))
    out["feed"] = fq
    q = fq["q"]
    gap = lambda x: float(vle.T(vle.x_of_y(x)) - vle.T(x))  # noqa: E731
    out["dT_dew_bubble_distillate_K"] = gap(xd)
    out["dT_dew_bubble_bottoms_K"] = gap(xb)
    out["min_dew_bubble_gap_K"] = MIN_DEW_BUBBLE_GAP_K
    for name, x, toward in (("destilado", xd, 1), ("fondos", xb, -1)):
        if gap(x) >= MIN_DEW_BUBBLE_GAP_K:
            continue
        # purest product on this side that still has the gap (bisection between the feed and the requested product)
        lo, hi = z, x
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(mid) >= MIN_DEW_BUBBLE_GAP_K else (lo, mid)
        az = [a for a in (azeotropes or []) if abs(a["x_az"][light] - (x_hi if toward > 0 else x_lo)) < 1e-9]
        where = (f"junto al azeótropo predicho (x_{light} = {az[0]['x_az'][light]:.4f})" if az else
                 "en un pellizco del equilibrio")
        out["refusals"].append(
            f"el {name} pedido (x_{light} = {x:.4f}) tiene T_rocío − T_burbuja = {gap(x):.4f} K, {where}: por debajo "
            f"de {MIN_DEW_BUBBLE_GAP_K} K el equilibrio suavizado (SmoothVLE) de esta tool deja vapor ficticio en los "
            f"líquidos saturados y no cierra el balance (límite medido de la tool, no de la física). Con estos "
            f"parámetros el {name} más puro que se puede sellar es x_{light} ≈ {lo:.3f}; para ir más allá usa la "
            f"columna de DWSIM (optimize_design_multi)")
    if out["refusals"]:
        return out
    nmin = min_stages_total_reflux(vle, xd, xb)
    out["N_min_total_reflux"] = nmin
    stages = n_trays + 1
    if nmin is None or nmin > stages:
        need = "no se alcanza ni con 500 etapas (pellizco)" if nmin is None else f"hacen falta {nmin} etapas de equilibrio"
        out["refusals"].append(
            f"con {n_trays} platos + rehervidor ({stages} etapas de equilibrio) no se llega de x_D({light}) = {xd:.4f} "
            f"a x_B({light}) = {xb:.5f} ni a reflujo total: {need} (McCabe-Thiele con estos parámetros NRTL a "
            f"{P_Pa:.0f} Pa). Añade platos o relaja las purezas")
        return out
    rmin = min_reflux(vle, xd, xb, z, q, dF)
    out["R_min_CMO"] = rmin
    if rmin is None:
        return out
    r_hi = float(rr_bounds[1]) * CMO_REFUSAL_MARGIN
    rneed = reflux_needed(vle, n_trays, feed_tray, xd, xb, z, q, dF, rmin, max(r_hi, rmin * 1.0001))
    out["R_needed_CMO"] = rneed
    if rneed is None:
        # is it the feed tray or the tray count? best feed tray at the reflux bound
        best = min(range(1, n_trays + 1),
                   key=lambda f: step_stages(vle, r_hi, n_trays, f, xd, xb, z, q, dF)["x_reboiler"])
        xr = step_stages(vle, r_hi, n_trays, best, xd, xb, z, q, dF)["x_reboiler"]
        out["best_feed_tray_at_reflux_bound"] = {"feed_tray": best, "x_reboiler": xr, "reaches": bool(xr <= xb)}
        out["refusals"].append(
            f"con {n_trays} platos y alimentación en el plato {feed_tray}, ni {CMO_REFUSAL_MARGIN:g}× el reflujo máximo "
            f"permitido ({rr_bounds[1]:g}) alcanza x_D({light}) = {xd:.4f} con x_B({light}) = {xb:.5f} (McCabe-Thiele "
            f"de flujo molar constante; R_min ≈ {rmin:.3f}, N_min = {nmin}). " +
            (f"Con la alimentación en el plato {best} sí alcanza" if xr <= xb else
             "Ningún plato de alimentación lo logra con ese reflujo") + ": mueve la alimentación, añade platos o sube "
            "el límite de reflujo")
        return out
    for mlt in START_MARGINS:
        rr = min(max(rneed * mlt, rr_bounds[0]), rr_bounds[1])
        br = _lines(rr, xd, xb, z, q, dF)[2]
        if br_bounds[0] <= br <= br_bounds[1] and (rr, br) not in out["starts"]:
            out["starts"].append((float(rr), float(br)))
    prof = step_stages(vle, out["starts"][0][0] if out["starts"] else rneed, n_trays, feed_tray, xd, xb, z, q, dF)
    out["profile_at_first_start"] = {"x_light": prof["x"], "T_K": [float(vle.T(x)) for x in prof["x"]]}
    return out
