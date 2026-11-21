"""Binary distillation on the McCabe-Thiele diagram with the REAL equilibrium curve of an activity-coefficient model.

Why: a Fenske-Underwood shortcut assumes a constant relative volatility. On a non-ideal pair (ethanol-water towards its
azeotrope) alpha falls from ~9 to 1 along the column and the minimum reflux is set by a TANGENT pinch, not by the feed:
the constant-alpha shortcut is then wrong by tens of percent and blind to the azeotrope. This module steps stages on the
curve y(x) computed at the column pressure with gamma-phi (modified Raoult, ideal vapour by default):

  * N_min: equilibrium stages at total reflux (operating line y = x). No enthalpy assumption: exact for this VLE.
  * R_min: largest slope of a rectifying line through (x_D, x_D) touching the curve between the feed pinch and x_D
    (feed pinch or tangent pinch), under constant molar overflow (CMO).
  * N(R) with the feed on its optimal stage, and R needed for a given number of stages, both CMO.

Stages are equilibrium stages below a TOTAL condenser and include the partial reboiler. The last stage is fractional:
(x_in - x_B) / (x_in - x_out). Component 1 is the first of `components` and must be the more volatile one between x_B
and x_D.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from packages.vle import models as M
from packages.vle.equilibrium import DataError, Psat, System, VaporPhase

N_GRID = 2001
MAX_STAGES = 400
R_CEILING = 1e4


class Curve:
    """y(x), T(x) of component 1 at pressure P on a uniform grid, with azeotropes."""

    def __init__(self, system: System, lng, P_Pa: float, n: int = N_GRID):
        self.x = np.linspace(1e-6, 1 - 1e-6, n)
        self.T, self.y = system.bubble_T(np.full(n, float(P_Pa)), self.x, lng)
        d = self.y - self.x
        self.azeotropes = []
        for i in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
            xa = self.x[i] + d[i] * (self.x[i + 1] - self.x[i]) / (d[i] - d[i + 1])
            self.azeotropes.append({"x1": float(xa), "T_K": float(np.interp(xa, self.x, self.T)),
                                    "kind": "temperatura mínima" if d[i] > 0 else "temperatura máxima"})

    def yeq(self, x):
        return np.interp(x, self.x, self.y)

    def window(self, xB: float, xD: float):
        """Monotonic branch of the curve covering [xB, xD] (bounded by azeotropes), for the inverse x(y)."""
        lo, hi = 0.0, 1.0
        for a in self.azeotropes:
            if a["x1"] <= xB:
                lo = max(lo, a["x1"])
            if a["x1"] >= xD:
                hi = min(hi, a["x1"])
        m = (self.x >= lo) & (self.x <= hi)
        return self.x[m], self.y[m]


def _n_min(curve: Curve, xB: float, xD: float) -> float:
    xs, ys = curve.window(xB, xD)
    n, x = 0.0, xD
    while n < MAX_STAGES:
        xn = float(np.interp(x, ys, xs))
        if xn <= xB:
            return n + (x - xB) / (x - xn)
        if x - xn < 1e-12:
            break
        x, n = xn, n + 1
    return float("inf")


def _feed_pinch(curve: Curve, z: float, q: float) -> float:
    """x where the q-line meets the equilibrium curve."""
    if abs(q - 1.0) < 1e-9:
        return z
    f = lambda x: float(curve.yeq(x)) - (q / (q - 1.0) * x - z / (q - 1.0))
    return brentq(f, 1e-6, 1 - 1e-6)


def _r_min(curve: Curve, z: float, q: float, xD: float) -> dict:
    xq = _feed_pinch(curve, z, q)
    xx = curve.x[(curve.x >= xq) & (curve.x < xD - 1e-4)]
    xx = np.concatenate(([xq], xx))
    slopes = (xD - curve.yeq(xx)) / (xD - xx)
    i = int(np.argmax(slopes))
    LV = float(slopes[i])
    return {"R_min": LV / (1.0 - LV) if LV < 1 else float("inf"), "x_pinch": float(xx[i]),
            "pinch": "alimentación" if i == 0 else "tangente (curva de equilibrio, no la alimentación)",
            "x_feed_pinch": float(xq), "y_feed_pinch": float(curve.yeq(xq))}


def _stages(curve: Curve, z: float, q: float, xD: float, xB: float, R: float) -> dict:
    """Stages from the top at reflux R (CMO), feed on the optimal stage. Returns n (fractional, incl. reboiler) and
    feed_stage (1 = top stage below the total condenser), or n = inf when the lines pinch."""
    xs, ys = curve.window(xB, xD)
    # intersection of the rectifying line with the q-line
    if abs(q - 1.0) < 1e-9:
        xi = z
    else:
        xi = (z / (q - 1.0) + xD / (R + 1.0)) / (q / (q - 1.0) - R / (R + 1.0))
    yi = R / (R + 1.0) * xi + xD / (R + 1.0)
    if not xB < xi < xD:
        return {"n": float("inf"), "feed_stage": None}
    strip = lambda x: xB + (yi - xB) / (xi - xB) * (x - xB)
    rect = lambda x: R / (R + 1.0) * x + xD / (R + 1.0)
    n, y, feed, x_prev = 0, xD, None, xD
    while n < MAX_STAGES:
        x = float(np.interp(y, ys, xs))
        if feed is None and x <= xi:
            feed = n + 1
        if x <= xB:
            return {"n": n + (x_prev - xB) / (x_prev - x), "feed_stage": feed or n + 1}
        y_next = strip(x) if feed is not None else rect(x)
        if y - y_next < 1e-12:
            break
        y, n, x_prev = y_next, n + 1, x
    return {"n": float("inf"), "feed_stage": feed}


def _r_for_stages(curve, z, q, xD, xB, n_target: float, r_min: float) -> float | None:
    f = lambda R: _stages(curve, z, q, xD, xB, R)["n"] - n_target
    lo = r_min * (1 + 1e-6) + 1e-9
    hi = max(2 * lo, 1.0)
    while f(hi) > 0:
        hi *= 2
        if hi > R_CEILING:
            return None
    while not np.isfinite(f(lo)):
        lo = lo + 0.02 * (hi - lo)
        if hi - lo < 1e-9:
            return None
    if f(lo) < 0:
        return lo
    return brentq(f, lo, hi, xtol=1e-9, rtol=1e-10)


# ------------------------------------------------------------------------------------------------ model set-up
def _library(model: str, cas: list[str]) -> tuple[dict, str]:
    from thermo.interaction_parameters import IPDB
    table = f"ChemSep {model}"
    key = "bij"
    if not IPDB.has_ip_specific(table, cas, key):
        raise DataError(f"la librería (thermo, tabla '{table}') no tiene parámetros para este par: da `parameters`")
    b = IPDB.get_ip_asymmetric_matrix(table, cas, "bij")
    if model == "Wilson":
        a = IPDB.get_ip_asymmetric_matrix(table, cas, "aij")
        p = {"a12": a[0][1], "a21": a[1][0], "b12": b[0][1], "b21": b[1][0], "V": (1.0, 1.0)}
        form = "Λ_ij = exp(a_ij + b_ij/T), sin cociente de volúmenes"
    else:
        al = IPDB.get_ip_asymmetric_matrix(table, cas, "alphaij")
        p = {"a12": 0.0, "a21": 0.0, "b12": b[0][1], "b21": b[1][0], "alpha": al[0][1]}
        form = "τ_ij = b_ij/T"
    return p, f"thermo.interaction_parameters IPDB, tabla '{table}' (base de datos de ChemSep): {form}; NO son del usuario"


def setup(components: list[str], model: str, parameters: dict | None, psat: dict | None, vapor_phase: str,
          wilson_volumes_m3_mol: dict | None, uniquac_rq: dict | None):
    from packages.vle.tool import _cas, _rq, _vol
    if len(components) != 2:
        raise DataError("se necesitan exactamente dos componentes")
    cas = [_cas(c) for c in components]
    psat = psat or {}
    ps = (Psat(components[0], cas[0], psat.get(components[0])), Psat(components[1], cas[1], psat.get(components[1])))
    vap = VaporPhase(vapor_phase, tuple(cas))
    system = System(ps, vap)
    assumptions = [f"Psat de {p.name}: {p.source}" for p in ps if p.kind == "thermo"]
    key = {"ideal": "ideal", "raoult": "ideal", "wilson": "Wilson", "nrtl": "NRTL", "uniquac": "UNIQUAC"}.get(str(model).lower())
    if key is None:
        raise DataError("model debe ser 'ideal', 'Wilson', 'NRTL' o 'UNIQUAC'")
    if key == "ideal":
        lng = lambda T, x: (np.zeros_like(np.asarray(x, float)), np.zeros_like(np.asarray(x, float)))
        return system, lng, {"model": "ideal (ley de Raoult, γ = 1)", "parameters": None, "source": None}, assumptions, cas
    if parameters is None:
        if key == "UNIQUAC":
            raise DataError("UNIQUAC necesita `parameters` {b12, b21 [K], a12, a21}; la librería solo se usa para Wilson y NRTL")
        p, source = _library(key, cas)
        assumptions.append(f"parámetros binarios de {key} tomados de la librería ({source}): supuesto, no dato del enunciado")
    else:
        p = dict(parameters)
        source = "del usuario"
        if "lambda12" in p or "lambda21" in p:                       # Wilson with constant Λ (textbook form)
            if key != "Wilson":
                raise DataError("lambda12/lambda21 solo aplican a Wilson")
            p = {"a12": float(np.log(p["lambda12"])), "a21": float(np.log(p["lambda21"])), "b12": 0.0, "b21": 0.0,
                 "V": (1.0, 1.0)}
            source = "del usuario (Λ12, Λ21 constantes)"
        elif "energy_J_mol" in p:
            d12, d21 = p.pop("energy_J_mol")
            p["b12"], p["b21"] = M.b_from_energy(key, float(d12), float(d21))
            source = "del usuario (energías Δg/Δu/Δλ en J/mol)"
        for k in ("b12", "b21"):
            if k not in p:
                raise DataError(f"parameters: falta {k} (o da energy_J_mol [d12, d21], o lambda12/lambda21 para Wilson)")
        if key == "NRTL" and "alpha" not in p:
            raise DataError("NRTL: falta alpha en parameters")
        if key == "Wilson" and "V" not in p:
            v = [_vol(c, s, wilson_volumes_m3_mol) for c, s in zip(components, cas)]
            p["V"] = (v[0][0], v[1][0])
            assumptions += [f"volumen molar líquido de {c} para Wilson: {s_}" for c, (_, s_) in zip(components, v)
                            if s_ != "del usuario"]
        if key == "UNIQUAC" and ("r" not in p or "q" not in p):
            rq = [_rq(c, s, uniquac_rq) for c, s in zip(components, cas)]
            p["r"], p["q"] = (rq[0][0], rq[1][0]), (rq[0][1], rq[1][1])
            assumptions += [f"r y q de {c} para UNIQUAC: {s_[2]}" for c, s_ in zip(components, rq) if s_[2] != "del usuario"]
    lng = system.gammas_fn(key, p)
    info = {"model": key, "parameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
            "source": source, "convention": "la de vle_binary_regression: NRTL τ_ij = a_ij + b_ij/T; UNIQUAC τ_ij = "
            "exp(a_ij + b_ij/T); Wilson Λ_ij = (V_j/V_i)·exp(a_ij + b_ij/T); b en K"}
    return system, lng, info, assumptions, cas


def feed_quality(components, cas, z: float, T_feed_K: float, T_bubble: float, T_dew: float) -> tuple[float, str]:
    """q of a subcooled liquid or superheated vapour from mole-averaged pure-component Cp and ΔHvap (thermo)."""
    from thermo import EnthalpyVaporization, HeatCapacityGas, HeatCapacityLiquid
    zz = (z, 1.0 - z)
    hv = sum(w * EnthalpyVaporization(CASRN=c).T_dependent_property(T_bubble) for w, c in zip(zz, cas))
    cpv_span = sum(w * HeatCapacityGas(CASRN=c).T_dependent_property(0.5 * (T_bubble + T_dew)) for w, c in zip(zz, cas))
    span = hv + cpv_span * (T_dew - T_bubble)
    if T_feed_K <= T_bubble:
        Tm = 0.5 * (T_feed_K + T_bubble)
        cp = sum(w * HeatCapacityLiquid(CASRN=c).T_dependent_property(Tm) for w, c in zip(zz, cas))
        return 1.0 + cp * (T_bubble - T_feed_K) / span, (
            "q = (H_vapor de rocío − H_alim)/(H_vapor de rocío − H_líquido de burbuja) = 1 + Cp_L·(T_burbuja − T_alim)/"
            "[ΔH_vap(T_burbuja) + Cp_V·(T_rocío − T_burbuja)], con Cp y ΔH_vap de los puros (thermo) promediados en "
            "moles; se ignora el calor de mezcla")
    if T_feed_K >= T_dew:
        Tm = 0.5 * (T_feed_K + T_dew)
        cp = sum(w * HeatCapacityGas(CASRN=c).T_dependent_property(Tm) for w, c in zip(zz, cas))
        return -cp * (T_feed_K - T_dew) / span, (
            "q = −Cp_V·(T_alim − T_rocío)/ΔH_vap con Cp de gas ideal y ΔH_vap de los puros (thermo) promediados en moles")
    raise DataError("la temperatura de alimentación cae entre burbuja y rocío: da feed.q (o feed.vapor_fraction) en vez de T_K")


# ------------------------------------------------------------------------------------------------ tool body
def design(components: list[str], pressure_Pa: float, feed: dict, x_distillate: float, x_bottoms: float | None = None,
           recovery_in_distillate: float | None = None, model: str = "ideal", parameters: dict | None = None,
           n_stages: float | None = None, reflux_ratio: float | None = None, psat: dict | None = None,
           vapor_phase: str = "ideal", wilson_volumes_m3_mol: dict | None = None, uniquac_rq: dict | None = None) -> dict:
    system, lng, info, assumptions, cas = setup(components, model, parameters, psat, vapor_phase, wilson_volumes_m3_mol,
                                                uniquac_rq)
    P = float(pressure_Pa)
    z, xD = float(feed["z1"]), float(x_distillate)
    F = feed.get("molar_flow")
    if (x_bottoms is None) == (recovery_in_distillate is None):
        raise DataError("da x_bottoms o recovery_in_distillate (uno de los dos)")
    if x_bottoms is None:
        r = float(recovery_in_distillate)
        D_F = r * z / xD
        if not 0 < D_F < 1:
            raise DataError("la recuperación y x_distillate no dejan un fondo: D/F fuera de (0, 1)")
        xB = (1 - r) * z / (1 - D_F)
    else:
        xB = float(x_bottoms)
        D_F = (z - xB) / (xD - xB)
    if not 0 < xB < z < xD < 1:
        raise DataError(f"se requiere 0 < x_bottoms ({xB:.4g}) < z1 ({z:.4g}) < x_distillate ({xD:.4g}) < 1, en fracción "
                        f"mol de {components[0]} (el componente 1 debe ser el más volátil)")
    curve = Curve(system, lng, P)
    out: dict = {"components": components, "pressure_Pa": P, "thermo": {**info, "vapor_phase": system.vapor.source,
                                                                         "psat": [p.source for p in system.psat]},
                 "basis": f"fracciones mol de {components[0]}; etapas de equilibrio bajo un condensador TOTAL, "
                          "incluyendo el rehervidor parcial; última etapa fraccionaria",
                 "azeotropes": curve.azeotropes, "assumptions": assumptions, "warnings": []}
    seg = (curve.x >= xB) & (curve.x <= xD)
    if np.any(curve.y[seg] <= curve.x[seg]) or any(xB < a["x1"] < xD or a["x1"] <= xD + 1e-9 and a["x1"] >= xD - 1e-9
                                                   for a in curve.azeotropes):
        az = ", ".join(f"x1 = {a['x1']:.4f} ({a['T_K']:.2f} K)" for a in curve.azeotropes) or "ninguno"
        out.update(ok=False, feasible=False, reason=(
            f"con este modelo {components[0]} no es el más volátil en todo el intervalo x_B–x_D, o hay un azeótropo "
            f"entre ambos (azeótropos: {az}): una sola columna a esta presión no puede cruzarlo"))
        return out
    alpha = lambda x: float(curve.yeq(x) / (1 - curve.yeq(x)) * (1 - x) / x)
    aD, aB, aF = alpha(xD), alpha(xB), alpha(z)
    Tb = float(np.interp(z, curve.x, curve.T))
    # feed condition
    if feed.get("q") is not None:
        q, q_note = float(feed["q"]), "del usuario"
    elif feed.get("vapor_fraction") is not None:
        q, q_note = 1.0 - float(feed["vapor_fraction"]), "q = 1 − fracción vaporizada"
    elif feed.get("T_K") is not None:
        Td = float(np.interp(z, curve.y, curve.T)) if np.all(np.diff(curve.y) > 0) else Tb
        q, q_note = feed_quality(components, cas, z, float(feed["T_K"]), Tb, Td)
        assumptions.append(f"calidad de la alimentación estimada: {q_note}")
    else:
        q, q_note = 1.0, "líquido saturado (no se dio q, vapor_fraction ni T_K)"
        assumptions.append("alimentación tomada como líquido saturado (q = 1)")
    n_min = _n_min(curve, xB, xD)
    rm = _r_min(curve, z, q, xD)
    coarse = Curve(system, lng, P, (N_GRID - 1) // 2 + 1)
    n_min_c, rm_c = _n_min(coarse, xB, xD), _r_min(coarse, z, q, xD)["R_min"]
    fenske = np.log(xD / (1 - xD) * (1 - xB) / xB) / np.log(np.sqrt(aD * aB))
    out.update({
        "products": {"x_distillate": xD, "x_bottoms": xB, "D_over_F": D_F, "B_over_F": 1 - D_F,
                     "distillate_flow": None if F is None else F * D_F, "bottoms_flow": None if F is None else F * (1 - D_F),
                     "recovery_1_in_distillate": D_F * xD / z, "recovery_2_in_bottoms": (1 - D_F) * (1 - xB) / (1 - z),
                     "T_distillate_bubble_K": float(np.interp(xD, curve.x, curve.T)),
                     "T_bottoms_bubble_K": float(np.interp(xB, curve.x, curve.T))},
        "feed": {"z1": z, "q": q, "q_basis": q_note, "T_bubble_K": Tb},
        "relative_volatility": {"at_x_distillate": aD, "at_feed": aF, "at_x_bottoms": aB},
        "N_min": n_min, "R_min": rm["R_min"], "R_min_pinch": rm["pinch"], "x_pinch": rm["x_pinch"],
        "constant_alpha_comparison": {
            "fenske_N_min_geometric_mean_alpha": float(fenske),
            "note": "Fenske con α constante (media geométrica de cabeza y fondo): solo para ver cuánto se aparta el "
                    "atajo de α constante de la curva real; NO es el resultado"},
    })
    if max(aD, aB) / min(aD, aB) > 1.5:
        out["warnings"].append(f"α varía de {aB:.2f} (fondo) a {aD:.2f} (cabeza): un atajo de α constante "
                               "(Fenske-Underwood, shortcut_column de DWSIM) no es fiable para este par")
    if reflux_ratio is not None:
        R = float(reflux_ratio)
        if R <= rm["R_min"]:
            out["at_reflux_ratio"] = {"reflux_ratio": R, "feasible": False,
                                      "reason": f"R = {R:g} no supera R_min = {rm['R_min']:.4g}: etapas infinitas"}
        else:
            st = _stages(curve, z, q, xD, xB, R)
            out["at_reflux_ratio"] = {"reflux_ratio": R, "R_over_R_min": R / rm["R_min"], "feasible": bool(np.isfinite(st["n"])),
                                      "stages": st["n"], "optimal_feed_stage": st["feed_stage"]}
    if n_stages is not None:
        n = float(n_stages)
        if n <= n_min:
            out["for_n_stages"] = {"n_stages": n, "feasible": False, "reason": (
                f"{n:g} etapas no alcanzan: a reflujo total hacen falta N_min = {n_min:.2f}; con este modelo los productos "
                "pedidos son inalcanzables con esa columna a cualquier reflujo")}
        else:
            R = _r_for_stages(curve, z, q, xD, xB, n, rm["R_min"])
            if R is None:
                out["for_n_stages"] = {"n_stages": n, "feasible": False,
                                       "reason": f"no se encontró reflujo ≤ {R_CEILING:g} que dé los productos con {n:g} etapas"}
            else:
                st = _stages(curve, z, q, xD, xB, R)
                out["for_n_stages"] = {"n_stages": n, "feasible": True, "reflux_ratio_required": R,
                                       "R_over_R_min": R / rm["R_min"], "stages_check": st["n"],
                                       "optimal_feed_stage": st["feed_stage"]}
    checks = {"N_min_grid_rel_diff": abs(n_min - n_min_c) / n_min, "R_min_grid_rel_diff": abs(rm["R_min"] - rm_c) / rm["R_min"],
              "component_balance": abs(D_F * xD + (1 - D_F) * xB - z)}
    out["evidence"] = {"checks": checks,
                       "verified": bool(checks["N_min_grid_rel_diff"] < 5e-3 and checks["R_min_grid_rel_diff"] < 5e-3
                                        and checks["component_balance"] < 1e-12 and np.isfinite(n_min)),
                       "meaning": "N_min y R_min repetidos con la mitad de puntos en la curva y balance por componente; "
                                  "certifica el cálculo con ESTE modelo y parámetros, no que el modelo represente la mezcla"}
    out["not_included"] = ["balances de entalpía por etapa (R_min, etapas a R y R requerido suponen flujo molar constante)",
                           "eficiencias de plato, caída de presión, columnas con salidas laterales",
                           "mezclas de más de dos componentes"]
    out["ok"] = out["evidence"]["verified"]
    return out
