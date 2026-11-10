"""Pinch analysis (bloque M5): energy targets by the Problem Table Algorithm (Linnhoff & Flower 1978), composite and
grand composite curves, minimum number of units, ΔTmin sweep and the saving against the plant's current utilities.

Pure Python, no dependencies. Temperatures in K, heat-capacity flow rates CP = m·cp in W/K, heats in W. Each stream
is shifted by its ΔT contribution (ΔTmin/2 unless the stream gives its own): hot streams down, cold streams up.
`evidence` checks the cascade against an independent geometric construction: the shifted composite curves, placed
with the cold curve starting at the minimum cold utility, must touch (zero vertical gap) and never cross, and the
overall energy balance Q_cold − Q_hot = Σ hot duty − Σ cold duty must close.
"""
from __future__ import annotations

import bisect

_TOL = 1e-9


def _parse(streams: list[dict], dT_min: float) -> list[dict]:
    if not streams:
        raise ValueError("give at least one hot and one cold stream")
    out = []
    for i, s in enumerate(streams):
        name = str(s.get("name") or f"S{i + 1}")
        Ts, Tt = float(s["T_supply_K"]), float(s["T_target_K"])
        if abs(Ts - Tt) < 1e-12:
            raise ValueError(f"{name}: T_supply_K equals T_target_K (no duty); drop it or give a phase change as a "
                             "1 K span with its latent heat as heat_load_W")
        if "CP_W_K" in s and s["CP_W_K"] is not None:
            CP = float(s["CP_W_K"])
        elif "heat_load_W" in s and s["heat_load_W"] is not None:
            CP = abs(float(s["heat_load_W"])) / abs(Ts - Tt)
        else:
            raise ValueError(f"{name}: needs CP_W_K (m·cp) or heat_load_W")
        if CP <= 0:
            raise ValueError(f"{name}: CP_W_K must be positive")
        hot = Ts > Tt
        dTc = float(s["dT_contribution_K"]) if s.get("dT_contribution_K") is not None else dT_min / 2.0
        shift = -dTc if hot else dTc
        out.append(dict(name=name, hot=hot, Ts=Ts, Tt=Tt, CP=CP, dTc=dTc, S_hi=max(Ts, Tt) + shift,
                        S_lo=min(Ts, Tt) + shift, Q=CP * abs(Ts - Tt)))
    if not any(s["hot"] for s in out) or all(s["hot"] for s in out):
        raise ValueError("pinch analysis needs at least one hot stream (cools down) and one cold stream (heats up)")
    return out


def _cascade(ss: list[dict]) -> dict:
    temps = sorted({t for s in ss for t in (s["S_hi"], s["S_lo"])}, reverse=True)
    rows, R, cum = [], 0.0, [0.0]
    for hi, lo in zip(temps[:-1], temps[1:]):
        cp_net = sum(s["CP"] * (1 if s["hot"] else -1) for s in ss if s["S_hi"] >= hi - _TOL and s["S_lo"] <= lo + _TOL)
        dH = cp_net * (hi - lo)
        R += dH
        cum.append(R)
        rows.append({"T_shifted_high_K": hi, "T_shifted_low_K": lo, "net_CP_W_K": cp_net, "surplus_W": dH,
                     "streams": [s["name"] for s in ss if s["S_hi"] >= hi - _TOL and s["S_lo"] <= lo + _TOL]})
    QH = max(0.0, -min(cum))
    feasible = [QH + c for c in cum]
    QC = feasible[-1]
    scale = max(1.0, max(abs(f) for f in feasible))
    pinches = [t for t, f in zip(temps, feasible) if abs(f) <= 1e-9 * scale]
    for r, f_in, f_out in zip(rows, feasible[:-1], feasible[1:]):
        r["heat_in_W"], r["heat_out_W"] = f_in, f_out
    return dict(temps=temps, rows=rows, feasible=feasible, QH=QH, QC=QC, pinches=pinches)


def _composite(ss: list[dict], hot: bool, shifted: bool) -> list[tuple[float, float]]:
    """(H, T) breakpoints of the hot or cold composite from its coldest end, H cumulative in W."""
    key_hi, key_lo = ("S_hi", "S_lo") if shifted else (None, None)
    st = [s for s in ss if s["hot"] == hot]
    hi_of = (lambda s: s[key_hi]) if shifted else (lambda s: max(s["Ts"], s["Tt"]))
    lo_of = (lambda s: s[key_lo]) if shifted else (lambda s: min(s["Ts"], s["Tt"]))
    temps = sorted({t for s in st for t in (hi_of(s), lo_of(s))})
    pts, H = [(0.0, temps[0])], 0.0
    for lo, hi in zip(temps[:-1], temps[1:]):
        cp = sum(s["CP"] for s in st if lo_of(s) <= lo + _TOL and hi_of(s) >= hi - _TOL)
        H += cp * (hi - lo)
        pts.append((H, hi))
    return pts


def _T_at(curve: list[tuple[float, float]], H: float) -> float:
    Hs = [p[0] for p in curve]
    i = bisect.bisect_left(Hs, H - 1e-12)
    i = min(max(i, 1), len(curve) - 1)
    (H0, T0), (H1, T1) = curve[i - 1], curve[i]
    if H1 - H0 < 1e-15:            # vertical-free: flat segment means CP = 0 gap, take the higher temperature
        return T1
    return T0 + (T1 - T0) * (H - H0) / (H1 - H0)


def _min_gap(hot: list[tuple[float, float]], cold: list[tuple[float, float]], offset: float) -> tuple[float, float]:
    """Minimum vertical gap T_hot − T_cold over the overlap, with the cold curve shifted right by offset."""
    cold_s = [(H + offset, T) for H, T in cold]
    lo, hi = max(hot[0][0], cold_s[0][0]), min(hot[-1][0], cold_s[-1][0])
    if hi <= lo:
        return float("inf"), lo
    Hs = sorted({lo, hi, *[H for H, _ in hot if lo <= H <= hi], *[H for H, _ in cold_s if lo <= H <= hi]})
    best = (float("inf"), lo)
    for H in Hs:
        g = _T_at(hot, H) - _T_at(cold_s, H)
        if g < best[0]:
            best = (g, H)
    return best


def _targets(ss: list[dict]) -> dict:
    c = _cascade(ss)
    hot_sh, cold_sh = _composite(ss, True, True), _composite(ss, False, True)
    gap, H_touch = _min_gap(hot_sh, cold_sh, c["QC"])
    return {**c, "hot_sh": hot_sh, "cold_sh": cold_sh, "gap": gap, "H_touch": H_touch}


def pinch(streams: list[dict], dT_min_K: float, dT_min_sweep_K: list[float] | None = None,
          current_hot_utility_W: float | None = None, current_cold_utility_W: float | None = None) -> dict:
    if dT_min_K < 0:
        raise ValueError("dT_min_K must be >= 0")
    ss = _parse(streams, dT_min_K)
    t = _targets(ss)
    QH, QC = t["QH"], t["QC"]
    Q_hot_streams = sum(s["Q"] for s in ss if s["hot"])
    Q_cold_streams = sum(s["Q"] for s in ss if not s["hot"])
    balance_err = abs((QC - QH) - (Q_hot_streams - Q_cold_streams))
    balance_ok = balance_err <= 1e-9 * max(1.0, Q_hot_streams, Q_cold_streams)
    scale = max(1.0, max(abs(T) for _, T in t["hot_sh"] + t["cold_sh"]))
    no_cross = t["gap"] >= -1e-7 * scale
    # a lower cold utility than the target must make the shifted curves cross (the target is a minimum)
    if QC > 0:
        tighter = _min_gap(t["hot_sh"], t["cold_sh"], QC * (1 - 1e-6) - 1e-6)[0]
        minimum_ok = tighter < 0 or QH == 0
    else:
        minimum_ok = True
    interior = [p for p in t["pinches"] if t["temps"][0] - _TOL > p > t["temps"][-1] + _TOL]
    threshold = not interior
    # with a process pinch the shifted curves must touch; in a threshold problem they only must not cross
    touch_ok = threshold or abs(t["gap"]) <= 1e-7 * scale
    pinch_info = []
    for p in interior:
        dTs = sorted({s["dTc"] for s in ss})
        pinch_info.append({"T_shifted_K": p,
                           "T_hot_side_K": p + (dTs[0] if len(dTs) == 1 else float("nan")),
                           "T_cold_side_K": p - (dTs[0] if len(dTs) == 1 else float("nan"))})
    pinch_info = [{k: (v if v == v else None) for k, v in d.items()} for d in pinch_info]   # NaN → None
    # minimum number of units (Euler, per side of the main pinch): N = streams + utilities − 1
    if not threshold:
        Pm = interior[0] if len(interior) == 1 else max(interior)
        above = [s for s in ss if s["S_hi"] > Pm + _TOL]
        below = [s for s in ss if s["S_lo"] < Pm - _TOL]
        n_above = len(above) + (1 if QH > _TOL else 0) - 1
        n_below = len(below) + (1 if QC > _TOL else 0) - 1
        units = {"above_pinch": max(n_above, 0), "below_pinch": max(n_below, 0),
                 "total_MER": max(n_above, 0) + max(n_below, 0),
                 "without_pinch_split": len(ss) + (QH > _TOL) + (QC > _TOL) - 1}
    else:
        units = {"total_MER": len(ss) + (QH > _TOL) + (QC > _TOL) - 1}
    # curves in real temperatures, cold composite placed at the minimum cold utility
    hot_c = _composite(ss, True, False)
    cold_c = [(H + QC, T) for H, T in _composite(ss, False, False)]
    gcc = [{"T_shifted_K": T, "heat_flow_W": f} for T, f in zip(t["temps"], t["feasible"])]
    out = {"Q_hot_min_W": QH, "Q_cold_min_W": QC, "heat_recovery_W": Q_hot_streams - QC,
           "pinch": pinch_info, "threshold_problem": threshold,
           "dT_min_K": dT_min_K, "minimum_units": units,
           "problem_table": t["rows"], "grand_composite_curve": gcc,
           "hot_composite_curve": [{"H_W": H, "T_K": T} for H, T in hot_c],
           "cold_composite_curve": [{"H_W": H, "T_K": T} for H, T in cold_c],
           "streams": [{"name": s["name"], "type": "hot" if s["hot"] else "cold", "CP_W_K": s["CP"], "duty_W": s["Q"],
                        "T_shifted_high_K": s["S_hi"], "T_shifted_low_K": s["S_lo"],
                        "crosses_pinch": (not threshold) and s["S_hi"] > interior[0] + _TOL > s["S_lo"] + 2 * _TOL}
                       for s in ss],
           "warnings": [], "notes": ("Reglas de diseño MER: no transferir calor a través del pinch, no usar servicio "
                                     "frío arriba ni caliente abajo. Cada W cruzado se paga dos veces (caliente y frío). "
                                     "El GCC indica a qué nivel colocar servicios (vapor de baja vs alta, etc.).")}
    if threshold:
        out["warnings"].append("problema umbral: solo hace falta un servicio en el rango; el ΔTmin no limita hasta "
                               "el ΔT umbral (ver dT_min_sweep_K)")
    if current_hot_utility_W is not None:
        excess = current_hot_utility_W - QH
        out["current_vs_target"] = {"hot_excess_W": excess, "hot_saving_pct": 100 * excess / current_hot_utility_W
                                    if current_hot_utility_W else 0.0, "cross_pinch_estimate_W": max(excess, 0.0)}
        if current_cold_utility_W is not None:
            out["current_vs_target"]["cold_excess_W"] = current_cold_utility_W - QC
            mism = abs((current_cold_utility_W - QC) - excess)
            out["current_vs_target"]["balance_consistent"] = mism <= 1e-3 * max(1.0, abs(excess)) + 1.0
        if excess < -1e-6 * max(1.0, QH):
            out["warnings"].append("el servicio caliente actual es MENOR que el mínimo teórico: los datos de corrientes "
                                   "o de servicio no son consistentes (ΔTmin real menor, corriente faltante o error)")
    if dT_min_sweep_K:
        sweep = []
        for d in sorted(dT_min_sweep_K):
            tt = _targets(_parse(streams, float(d)))
            inner = [p for p in tt["pinches"] if tt["temps"][0] - _TOL > p > tt["temps"][-1] + _TOL]
            sweep.append({"dT_min_K": float(d), "Q_hot_min_W": tt["QH"], "Q_cold_min_W": tt["QC"],
                          "pinch_shifted_K": inner[0] if inner else None})
        out["dT_min_sweep"] = sweep
    reasons = []
    if not balance_ok:
        reasons.append(f"balance global no cierra ({balance_err:.3g} W)")
    if not no_cross:
        reasons.append(f"las curvas compuestas desplazadas se cruzan (gap {t['gap']:.3g} K)")
    if not touch_ok:
        reasons.append(f"las curvas compuestas desplazadas no se tocan en el pinch (gap {t['gap']:.3g} K)")
    if not minimum_ok:
        reasons.append("bajar el servicio frío no cruza las curvas: el objetivo no es mínimo")
    out["evidence"] = {"energy_balance_closed": balance_ok, "composite_min_gap_K": t["gap"],
                       "composite_touch_at_H_W": t["H_touch"], "curves_touch_without_crossing": no_cross and touch_ok,
                       "target_is_minimum": minimum_ok, "verified": not reasons, "reasons": reasons}
    return out
