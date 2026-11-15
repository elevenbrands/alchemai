"""Dynamic isothermal flash drum: liquid and vapour holdup in equilibrium at every instant, pressure set by the vapour
holdup and the vapour outlet, level set by the liquid holdup and the liquid outlet. Independent of DWSIM.

State: moles of each component in the drum, M_i (conservative form):
    dM_i/dt = F z_i − V y_i − L x_i
Algebraic equilibrium at every instant (drum volume V_d, temperature T fixed by the user):
    Rachford-Rice  Σ z_i (K_i − 1)/(1 + β(K_i − 1)) = 0,   x_i = z_i/(1 + β(K_i − 1)),  y_i = K_i x_i,  z = M/ΣM
    volume         P (V_d − (1 − β) N / ρ_m) = β N R T      (ideal vapour, liquid of constant molar density ρ_m)
    K-values       constant (the user's, at the drum T and design P), or constant relative volatility with Raoult's
                   law K_i = α_i P_ref / P (P_ref = vapour pressure of the reference component at T)
Outlets: vapour valve V = C·f(x)·√(P − P_down) [mol/s, C the user's], liquid valve q = Kv·f(x)·√(ΔP[bar]/SG) with
ΔP = P + ρ g h − P_down (IEC 60534 liquid sizing equation), or fixed molar flows. f(x) as in `packages.dynamics.units`.

At steady state the products are the Rachford-Rice flash of the FEED at the drum pressure (F z = V y + L x with
y = K x); the run checks it. The drum may not leave the two-phase region (no liquid, or full of liquid): that stops
the integration and is reported, never patched. Energy balance not included: the drum is isothermal (a heater or
cooler holds T)."""
from __future__ import annotations

import math

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from packages.dynamics import units as U
from packages.dynamics.flowsheet import (BALANCE_TOL, N_FINE, REFINE_TOL, STEADY_RATE_TOL, SETTLE_BAND, _metrics, _num,
                                         _req, clean)

RR_MATCH_TOL = 1e-6          # products vs Rachford-Rice of the feed, at a steady end state
METHODS = ("Radau", "BDF", "LSODA")


class FlashOutOfRange(ValueError):
    """The drum left the two-phase region (all liquid, all vapour, overfilled)."""


def rachford_rice_extended(z: np.ndarray, K: np.ndarray) -> float:
    """Rachford-Rice root on its whole monotone branch 1/(1 − K_max) < β < 1/(1 − K_min) (the «negative flash» of
    Whitson & Michelsen): smooth across β = 0 and β = 1, so finite-difference Jacobians never fall off the region.
    Physical only for 0 < β < 1 (checked by the integrator's events)."""
    kmax, kmin = float(np.max(K[z > 0])), float(np.min(K[z > 0]))
    if kmax <= 1.0 or kmin >= 1.0:
        raise FlashOutOfRange("todos los K están del mismo lado de 1: no hay dos fases posibles")
    km1 = K - 1.0
    lo, hi = 1.0 / (1.0 - kmax), 1.0 / (1.0 - kmin)
    eps = 1e-12 * (hi - lo)

    def g(b):
        return float(np.sum(z * km1 / (1.0 + b * km1)))

    return float(brentq(g, lo + eps, hi - eps, xtol=1e-15, rtol=1e-15, maxiter=500))


def rachford_rice(z: np.ndarray, K: np.ndarray) -> float:
    """Vapour fraction β in (0, 1); raises FlashOutOfRange outside the two-phase region."""
    if float(np.dot(z, K)) <= 1.0:
        raise FlashOutOfRange("líquido subenfriado: Σ z·K ≤ 1, no hay vapor en equilibrio")
    if float(np.dot(z, 1.0 / K)) <= 1.0:
        raise FlashOutOfRange("vapor sobrecalentado: Σ z/K ≤ 1, no hay líquido en equilibrio")
    km1 = K - 1.0

    def g(b):
        return float(np.sum(z * km1 / (1.0 + b * km1)))

    return float(brentq(g, 0.0, 1.0, xtol=1e-15, rtol=1e-15, maxiter=500))


class Drum:
    """Compiled drum: parameters, equilibrium and right-hand side."""

    def __init__(self, d: dict, components: list[str] | None = None):
        """Standalone drum (feed and liquid outlet inside `d`) or, with `components` given, a unit embedded in the
        flowsheet engine: then feed and liquid product are the engine's streams and are not read here."""
        if not isinstance(d, dict):
            raise ValueError("falta 'flash_drum': diccionario con los datos del tambor")
        embedded = components is not None
        comp = components if embedded else _req(d, "components", "el tambor", "los componentes (al menos dos)")
        self.comp = [str(c) for c in comp]
        if len(self.comp) < 2:
            raise ValueError("'components' necesita al menos dos componentes")
        nc = self.nc = len(self.comp)
        w = "el tambor"
        self.T = _num(_req(d, "temperature_K", w, "la temperatura del tambor (isotérmico)"), "'temperature_K'",
                      positive=True)
        self.Vd = _num(_req(d, "volume_m3", w, "el volumen total del tambor"), "'volume_m3'", positive=True)
        self.area = _num(_req(d, "area_m2", w, "convertir volumen de líquido en nivel"), "'area_m2'", positive=True)
        self.rhom = _num(_req(d, "liquid_molar_density_mol_m3", w, "el volumen del líquido retenido"),
                         "'liquid_molar_density_mol_m3'", positive=True)
        eq = _req(d, "equilibrium", w, "el equilibrio líquido-vapor ({'model': 'k_values', 'K': {…}} o "
                                       "{'model': 'relative_volatility', 'alpha': {…}, 'reference_vapour_pressure_Pa'})")
        model = str(_req(eq, "model", "'equilibrium'", "elegir 'k_values' o 'relative_volatility'"))
        if model == "k_values":
            kd = _req(eq, "K", "'equilibrium'", "los valores K de cada componente a la T y P de diseño")
            self.a = np.array([_num(_req(kd, c, "'equilibrium.K'", "el valor K"), f"K de {c}", positive=True)
                               for c in self.comp])
            self.e = 0.0
        elif model == "relative_volatility":
            al = _req(eq, "alpha", "'equilibrium'", "las volatilidades relativas (constantes)")
            pref = _num(_req(eq, "reference_vapour_pressure_Pa", "'equilibrium'",
                             "la presión de vapor del componente de referencia a la T del tambor"),
                        "'reference_vapour_pressure_Pa'", positive=True)
            self.a = np.array([_num(_req(al, c, "'equilibrium.alpha'", "la volatilidad relativa"), f"alfa de {c}",
                                    positive=True) for c in self.comp]) * pref
            self.e = 1.0
        else:
            raise ValueError("'equilibrium.model' debe ser 'k_values' o 'relative_volatility'")
        self.model = model
        self.P = {}
        self.zf = np.zeros(nc)
        if not embedded:
            fd = _req(d, "feed", w, "la alimentación (molar_flow_mol_s, mole_fractions)")
            self.P["feed.molar_flow_mol_s"] = _num(_req(fd, "molar_flow_mol_s", "'feed'", "el caudal de alimentación"),
                                                   "'feed.molar_flow_mol_s'", nonneg=True)
            zf = _req(fd, "mole_fractions", "'feed'", "la composición de la alimentación")
            self.zf = self._fractions(zf, "'feed.mole_fractions'")
        self.out = {}
        for side in (("vapour",) if embedded else ("vapour", "liquid")):
            od = _req(d, f"{side}_outlet", w, f"la salida de {'vapor' if side == 'vapour' else 'líquido'}")
            self.out[side] = self._outlet(side, od)
        if embedded:
            self.rho = None if d.get("liquid_density_kg_m3") is None else _num(
                d["liquid_density_kg_m3"], "'liquid_density_kg_m3'", positive=True)
        elif self.out["liquid"]["type"] == "valve":
            self.rho = _num(_req(d, "liquid_density_kg_m3", w, "la válvula de líquido (carga y densidad relativa)"),
                            "'liquid_density_kg_m3'", positive=True)
        ini = _req(d, "initial", w, "el estado inicial ({'liquid_level_m', 'mole_fractions'} del contenido total)")
        h0 = _num(_req(ini, "liquid_level_m", "'initial'", "el líquido retenido al inicio"), "'initial.liquid_level_m'",
                  positive=True)
        z0 = self._fractions(_req(ini, "mole_fractions", "'initial'", "la composición global del contenido"),
                             "'initial.mole_fractions'")
        self.M0 = self._initial_moles(h0, z0)
        self.scale = float(np.sum(self.M0))
        self.base = dict(self.P)
        self.zf_base = self.zf.copy()
        self.n = nc
        self.oor = ""

    # ---------------------------------------------------------------------------------------------------- input
    def _fractions(self, zd: dict, where: str) -> np.ndarray:
        z = np.array([_num(_req(zd, c, where, "la fracción molar de cada componente (0 si falta)"), f"{where}[{c}]",
                           nonneg=True) for c in self.comp])
        if abs(z.sum() - 1.0) > 1e-9:
            raise ValueError(f"{where} debe sumar 1 (suma {z.sum():.12g})")
        return z

    def _outlet(self, side: str, od: dict) -> dict:
        where = f"'{side}_outlet'"
        typ = str(_req(od, "type", where, "'valve' o 'fixed'"))
        o = {"type": typ}
        if typ == "fixed":
            self.P[f"{side}.molar_flow_mol_s"] = _num(_req(od, "molar_flow_mol_s", where, "el caudal molar impuesto"),
                                                      f"{where}.molar_flow_mol_s", nonneg=True)
            return o
        if typ != "valve":
            raise ValueError(f"{where}.type debe ser 'valve' o 'fixed'")
        if side == "vapour":
            o["C"] = _num(_req(od, "coefficient_mol_s_Pa05", where, "la capacidad de la válvula de vapor "
                               "V = C·f(x)·√(P − P_abajo)"), f"{where}.coefficient_mol_s_Pa05", positive=True)
        else:
            o["kv"] = _num(_req(od, "Kv_m3_h", where, "el coeficiente de la válvula de líquido"), f"{where}.Kv_m3_h",
                           positive=True)
        op = _num(_req(od, "opening", where, "la apertura (0 a 1)"), f"{where}.opening")
        if not 0.0 <= op <= 1.0:
            raise ValueError(f"{where}.opening debe estar entre 0 y 1")
        self.P[f"{side}.opening"] = op
        o["char"] = str(_req(od, "characteristic", where, "la característica inherente"))
        if o["char"] not in U.CHARACTERISTICS:
            raise ValueError(f"{where}.characteristic debe ser una de {', '.join(U.CHARACTERISTICS)}")
        o["rang"] = None
        if o["char"] == "equal_percentage":
            o["rang"] = _num(_req(od, "rangeability", where, "f = R^(x−1)"), f"{where}.rangeability", positive=True)
        self.P[f"{side}.downstream_pressure_Pa"] = _num(_req(od, "downstream_pressure_Pa", where, "la presión aguas abajo"),
                                                        f"{where}.downstream_pressure_Pa", nonneg=True)
        return o

    # ---------------------------------------------------------------------------------------------- equilibrium
    def K(self, P: float) -> np.ndarray:
        return self.a / P ** self.e

    def _split(self, P: float, z: np.ndarray, N: float) -> tuple[float, float]:
        b = rachford_rice(z, self.K(P))
        return b, P * (self.Vd - (1.0 - b) * N / self.rhom) - b * N * U.R_GAS * self.T

    def equilibrium(self, M: np.ndarray) -> dict:
        N = float(np.sum(M))
        if not np.all(np.isfinite(M)) or N <= 0.0 or np.any(M < -1e-12 * self.scale):
            raise FlashOutOfRange("el contenido del tambor se agotó")
        z = np.clip(M, 0.0, None) / N
        if self.e == 0.0:
            b = rachford_rice_extended(z, self.a)
            vl = (1.0 - b) * N / self.rhom
            if vl >= self.Vd:
                raise FlashOutOfRange("el tambor se llenó de líquido")
            P = b * N * U.R_GAS * self.T / (self.Vd - vl)
        else:
            p_bub, p_dew = float(np.dot(z, self.a)), 1.0 / float(np.dot(z, 1.0 / self.a))
            if N / self.rhom >= self.Vd:
                raise FlashOutOfRange("el tambor se llenó de líquido")
            if p_dew * self.Vd >= N * U.R_GAS * self.T:
                raise FlashOutOfRange("todo el contenido es vapor: no queda líquido")
            lo, hi = p_dew * (1 + 1e-13), p_bub * (1 - 1e-13)
            P = float(brentq(lambda p: self._split(p, z, N)[1], lo, hi, xtol=1e-12 * p_bub, rtol=1e-15, maxiter=500))
            b = rachford_rice(z, self.K(P))
        K = self.K(P)
        x = z / (1.0 + b * (K - 1.0))
        y = K * x
        L = (1.0 - b) * N
        return {"P": P, "beta": b, "x": x, "y": y, "N": N, "L": L, "level": L / self.rhom / self.area}

    def vapour_flow(self, eq: dict) -> float:
        P = self.P
        o = self.out["vapour"]
        if o["type"] == "fixed":
            return P["vapour.molar_flow_mol_s"]
        f = U.valve_fraction(P["vapour.opening"], o["char"], o["rang"])
        return o["C"] * f * math.sqrt(max(eq["P"] - P["vapour.downstream_pressure_Pa"], 0.0))

    def flows(self, eq: dict) -> tuple[float, float]:
        P = self.P
        o = self.out["vapour"]
        if o["type"] == "fixed":
            V = P["vapour.molar_flow_mol_s"]
        else:
            f = U.valve_fraction(P["vapour.opening"], o["char"], o["rang"])
            V = o["C"] * f * math.sqrt(max(eq["P"] - P["vapour.downstream_pressure_Pa"], 0.0))
        o = self.out["liquid"]
        if o["type"] == "fixed":
            L = P["liquid.molar_flow_mol_s"]
        else:
            f = U.valve_fraction(P["liquid.opening"], o["char"], o["rang"])
            dp = eq["P"] + self.rho * U.G * eq["level"] - P["liquid.downstream_pressure_Pa"]
            L = U.valve_flow(o["kv"], f, dp, self.rho, False) * self.rhom
        return V, L

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Out of the two-phase region the derivative is NaN: Radau then rejects the trial step and shrinks it; if the
        true trajectory leaves the region the integration stops and the reason is reported."""
        n = self.n
        try:
            eq = self.equilibrium(y[:n])
        except FlashOutOfRange as e:
            self.oor = str(e)
            return np.full(len(y), np.nan)
        V, L = self.flows(eq)
        F = self.P["feed.molar_flow_mol_s"]
        dy = np.zeros(len(y))
        dy[:n] = F * self.zf - V * eq["y"] - L * eq["x"]
        dy[n:2 * n] = F * self.zf                       # ∫ fed
        dy[2 * n:3 * n] = V * eq["y"]                   # ∫ vapour product
        dy[3 * n:4 * n] = L * eq["x"]                   # ∫ liquid product
        return dy

    def _initial_moles(self, h0: float, z0: np.ndarray) -> np.ndarray:
        L = h0 * self.area * self.rhom
        if L >= self.rhom * self.Vd:
            raise ValueError("el nivel inicial llena el tambor de líquido")
        try:
            if self.e == 0.0:
                b = rachford_rice(z0, self.a)
            else:
                def g(p):
                    bb = rachford_rice(z0, self.K(p))
                    N = L / (1.0 - bb)
                    return p * (self.Vd - L / self.rhom) - bb * N * U.R_GAS * self.T
                p_bub, p_dew = float(np.dot(z0, self.a)), 1.0 / float(np.dot(z0, 1.0 / self.a))
                p = brentq(g, p_dew * (1 + 1e-12), p_bub * (1 - 1e-12), xtol=1e-12 * p_bub, rtol=1e-15)
                b = rachford_rice(z0, self.K(p))
        except FlashOutOfRange as e:
            raise ValueError(f"el estado inicial no es bifásico: {e}") from None
        return z0 * L / (1.0 - b)

    def outputs(self, y: np.ndarray) -> dict:
        eq = self.equilibrium(y[:self.n])
        V, L = self.flows(eq)
        out = {"pressure_Pa": eq["P"], "liquid_level_m": eq["level"], "holdup_vapour_fraction": eq["beta"],
               "vapour_flow_mol_s": V, "liquid_flow_mol_s": L, "feed_flow_mol_s": self.P["feed.molar_flow_mol_s"]}
        for i, c in enumerate(self.comp):
            out[f"x.{c}"] = eq["x"][i]
            out[f"y.{c}"] = eq["y"][i]
        return out


def _compile_events(dr: Drum, events: list[dict] | None, t_end: float) -> list[dict]:
    out = []
    for k, e in enumerate(events or []):
        where = f"el evento {k + 1}"
        t = _num(_req(e, "time_s", where, "saber cuándo ocurre"), f"{where}: 'time_s'", nonneg=True)
        tgt = str(_req(e, "target", where, "saber qué cambia"))
        val = _req(e, "value", where, "el nuevo valor")
        if tgt == "feed.mole_fractions":
            val = dr._fractions(val, f"{where}: 'value'")
        elif tgt in dr.P:
            val = _num(val, f"{where}: 'value'")
            if tgt.endswith(".opening") and not 0.0 <= val <= 1.0:
                raise ValueError(f"{where}: la apertura debe estar entre 0 y 1")
        else:
            raise ValueError(f"{where}: destino '{tgt}' no válido. Destinos: feed.mole_fractions, {', '.join(sorted(dr.P))}")
        out.append({"time_s": t, "target": tgt, "value": val, "inside_horizon": t < t_end})
    out.sort(key=lambda e: e["time_s"])
    return out


def _apply(dr: Drum, evs: list[dict], t: float):
    dr.P.update(dr.base)
    dr.zf = dr.zf_base.copy()
    for e in evs:
        if e["time_s"] <= t:
            if e["target"] == "feed.mole_fractions":
                dr.zf = np.array(e["value"], dtype=float)
            else:
                dr.P[e["target"]] = e["value"]


REGION_EDGES = ("sin vapor (β = 0: líquido subenfriado o tambor lleno de líquido)", "sin líquido (β = 1)",
                "el líquido llena el volumen del tambor")


def _region_events(dr: Drum) -> list:
    """Terminal events at the edges of the physical region (only needed with constant K, whose equilibrium is extended
    smoothly beyond them; with Raoult the equilibrium itself is bounded and leaving the region fails loudly)."""
    if dr.e != 0.0:
        return []

    def eq(y):
        try:
            return dr.equilibrium(y[:dr.n])
        except FlashOutOfRange:
            return None

    def e_vap(t, y):
        r = eq(y)
        return r["beta"] if r else -1.0

    def e_liq(t, y):
        r = eq(y)
        return 1.0 - r["beta"] if r else -1.0

    def e_full(t, y):
        r = eq(y)
        return (dr.Vd - r["L"] / dr.rhom) / dr.Vd if r else -1.0

    fns = [e_vap, e_liq, e_full]
    for f in fns:
        f.terminal, f.direction = True, -1.0
    return fns


def _integrate(dr: Drum, y0: np.ndarray, evs: list[dict], t_end: float, method: str, rtol: float, atol: np.ndarray):
    bps = sorted({0.0, t_end} | {e["time_s"] for e in evs if 0.0 < e["time_s"] < t_end})
    segs, info, y = [], {"success": True, "message": "", "restarts": []}, y0.copy()
    for ta, tb in zip(bps[:-1], bps[1:]):
        _apply(dr, evs, ta)
        info["restarts"].append(ta)
        snap = (dict(dr.P), dr.zf.copy())
        dr.oor = ""
        try:
            sol = solve_ivp(dr.rhs, (ta, tb), y, method=method, rtol=rtol, atol=atol, dense_output=True,
                            events=_region_events(dr))
        except (FlashOutOfRange, ValueError, np.linalg.LinAlgError) as e:   # NaN Jacobian at the edge of the region
            why = dr.oor or str(e)[:120]
            info.update(success=False, message=f"el tambor salió de la región de dos fases después de t = {ta:.6g} s: "
                                               f"{why}", t_reached=ta)
            return segs, info
        if not sol.success:
            why = f" (último estado de prueba fuera de la región de dos fases: {dr.oor})" if dr.oor else ""
            info.update(success=False, message=str(sol.message)[:200] + why, t_reached=float(sol.t[-1]))
            if len(sol.t) > 1:
                segs.append((ta, float(sol.t[-1]), sol.sol, snap, len(sol.t) - 1))
            return segs, info
        if sol.status == 1:                         # the accepted trajectory reached the edge of the two-phase region
            te = float(sol.t[-1])
            segs.append((ta, te, sol.sol, snap, len(sol.t) - 1))
            which = [REGION_EDGES[i] for i, tt in enumerate(sol.t_events) if len(tt)]
            info.update(success=False, t_reached=te, message=f"el tambor sale de la región de dos fases en t = {te:.6g} s: "
                                                             + "; ".join(which))
            return segs, info
        segs.append((ta, tb, sol.sol, snap, len(sol.t) - 1))
        y = sol.y[:, -1].copy()
    info["t_reached"] = t_end
    return segs, info


def _eval(dr: Drum, segs, t: float) -> tuple[np.ndarray, dict]:
    for ta, tb, s, snap, _ in segs:
        if ta <= t <= tb and (t < tb or (ta, tb) == segs[-1][:2]):
            dr.P.update(snap[0])
            dr.zf = snap[1]
            y = s(t)
            return y, dr.outputs(y)
    raise ValueError("instante fuera de la trayectoria")


def steady_state(dr: Drum) -> dict:
    n, sc = dr.n, dr.scale

    def f(x):
        y = np.concatenate([x * sc, np.zeros(3 * n)])
        r = dr.rhs(0.0, y)[:n] / sc
        return r if np.all(np.isfinite(r)) else np.full(n, 1e3)

    sol = root(f, dr.M0 / sc, method="hybr", options={"xtol": 1e-14})
    res = f(sol.x)
    worst = float(np.max(np.abs(res)))
    return {"M": sol.x * sc, "residual_rel_per_s": worst, "converged": bool(worst < 1e-10),
            "message": str(sol.message)}


def simulate_flash_drum(flash_drum: dict, t_end_s: float, events: list[dict] | None = None,
                        start_from_steady_state: bool = False, n_points: int = 201, method: str = "Radau",
                        rtol: float = 1e-8, refine: bool = True) -> dict:
    """Transient of the isothermal flash drum. Returns series of pressure, level, flows and compositions, metrics, final
    state, the Rachford-Rice check of the products and the evidence."""
    if method not in METHODS:
        raise ValueError(f"'method' debe ser uno de {', '.join(METHODS)}")
    if not (1e-12 <= rtol <= 1e-3):
        raise ValueError("'rtol' debe estar entre 1e-12 y 1e-3")
    if t_end_s is None or not math.isfinite(float(t_end_s)) or float(t_end_s) <= 0:
        raise ValueError("falta 't_end_s' (> 0): el horizonte de simulación no se supone")
    if not (2 <= int(n_points) <= 5000):
        raise ValueError("'n_points' debe estar entre 2 y 5000")
    t_end = float(t_end_s)
    dr = Drum(flash_drum)
    evs = _compile_events(dr, events, t_end)
    n = dr.n
    ss_info = None
    M0 = dr.M0
    if start_from_steady_state:
        ss = steady_state(dr)
        ss_info = {k: ss[k] for k in ("converged", "residual_rel_per_s")}
        if not ss["converged"]:
            raise ValueError(f"no se encontró el estado estacionario inicial ({ss['message'][:120]})")
        M0 = ss["M"]
    y0 = np.concatenate([M0, np.zeros(3 * n)])
    atol = 1e-12 * dr.scale * np.ones(4 * n)
    segs, info = _integrate(dr, y0, evs, t_end, method, rtol, atol)
    reasons = []
    if not segs:
        raise ValueError(f"la integración falló al inicio: {info['message']}")
    if not info["success"]:
        reasons.append(info["message"])
    t_reached = segs[-1][1]
    marks = [e["time_s"] for e in evs if e["time_s"] <= t_reached]
    grid = np.unique(np.concatenate([np.linspace(0.0, t_reached, N_FINE), marks]))
    sgrid = np.unique(np.concatenate([np.linspace(0.0, t_reached, int(n_points)), marks]))
    names = list(dr.outputs(y0).keys())

    def series(g):
        rows = [_eval(dr, segs, float(t))[1] for t in g]
        return {k: np.array([r[k] for r in rows]) for k in names}

    fine, coarse = series(grid), series(sgrid)
    t_last = max([e["time_s"] for e in evs if e["time_s"] < t_reached] + [0.0])
    metrics = {k: _metrics(grid, fine[k], t_last) for k in ("pressure_Pa", "liquid_level_m", "vapour_flow_mol_s",
                                                          "liquid_flow_mol_s")}
    y_end, out_end = _eval(dr, segs, t_reached)
    # ---- balances
    M_start, M_end = y0[:n], y_end[:n]
    fed, vout, lout = y_end[n:2 * n], y_end[2 * n:3 * n], y_end[3 * n:]
    bal = {}
    worst = 0.0
    for i, c in enumerate(dr.comp):
        acc, net = M_end[i] - M_start[i], fed[i] - vout[i] - lout[i]
        sc = max(abs(acc), abs(net), fed[i], vout[i], lout[i], M_start[i] * 1e-9, 1e-300)
        bal[f"component_mol.{c}"] = {"accumulated": acc, "fed": fed[i], "vapour_product": vout[i],
                                     "liquid_product": lout[i], "closure_rel": abs(acc - net) / sc}
        worst = max(worst, abs(acc - net) / sc)
    acc, net = float(M_end.sum() - M_start.sum()), float(fed.sum() - vout.sum() - lout.sum())
    bal["total_mol"] = {"accumulated": acc, "net_in_minus_out": net,
                        "closure_rel": abs(acc - net) / max(abs(acc), abs(net), float(fed.sum()), 1e-300)}
    if worst > BALANCE_TOL:
        reasons.append(f"un balance de componente no cierra ({worst:.2e})")
    # ---- steady end state and Rachford-Rice of the feed at the final pressure
    dM = dr.rhs(t_reached, y_end)[:n]
    steady_change = float(np.max(np.abs(dM) / np.maximum(M_end, 1e-6 * dr.scale)) * max(t_reached - t_last, 1e-300))
    at_steady = bool(steady_change < STEADY_RATE_TOL) and info["success"]
    F, V, L = out_end["feed_flow_mol_s"], out_end["vapour_flow_mol_s"], out_end["liquid_flow_mol_s"]
    rr = None
    try:
        Kf = dr.K(out_end["pressure_Pa"])
        bf = rachford_rice(dr.zf, Kf)
        xf = dr.zf / (1.0 + bf * (Kf - 1.0))
        yp = np.array([out_end[f"y.{c}"] for c in dr.comp])
        xp = np.array([out_end[f"x.{c}"] for c in dr.comp])
        dev = max(abs(V / F - bf) if F > 0 else 0.0, float(np.max(np.abs(yp - Kf * xf))), float(np.max(np.abs(xp - xf))))
        rr = {"feed_vapour_fraction": bf, "product_vapour_fraction": V / F if F > 0 else None,
              "x": dict(zip(dr.comp, xf)), "y": dict(zip(dr.comp, Kf * xf)), "max_deviation": dev,
              "pressure_Pa": out_end["pressure_Pa"]}
        if at_steady and dev > RR_MATCH_TOL:
            reasons.append(f"el estado final estacionario no coincide con el flash de la alimentación (desviación "
                           f"{dev:.2e})")
    except FlashOutOfRange as e:
        rr = {"error": f"la alimentación no es bifásica a la presión final: {e}"}
    # ---- tolerance refinement
    refinement = None
    if refine and info["success"]:
        segs2, info2 = _integrate(dr, y0, evs, t_end, method, max(rtol / 100.0, 1e-13), atol / 100.0)
        if info2["success"]:
            rows2 = [_eval(dr, segs2, float(t))[1] for t in grid]
            diffs = {}
            for k in names:
                v2 = np.array([r[k] for r in rows2])
                span = max(np.ptp(v2), 1e-9 * max(np.max(np.abs(v2)), 1e-300))
                diffs[k] = float(np.max(np.abs(fine[k] - v2)) / span)
            wk = max(diffs, key=diffs.get)
            refinement = {"rtol": rtol, "max_difference_rel_to_span": diffs[wk], "worst_output": wk}
            if diffs[wk] > REFINE_TOL:
                reasons.append(f"la solución cambia {diffs[wk]:.2e} del rango de '{wk}' al apretar la tolerancia")
        else:
            reasons.append(f"la corrida de tolerancia apretada falló: {info2['message']}")
    warnings = []
    if not at_steady and info["success"]:
        warnings.append("el estado final todavía cambia: no es estacionario (alargue t_end_s si lo busca); la comparación "
                        "con Rachford-Rice es solo informativa")
    if any(not e["inside_horizon"] for e in evs):
        warnings.append("hay eventos en o después de t_end_s: no se aplicaron")
    assumptions = [f"tambor isotérmico a {dr.T:g} K (un calentador o enfriador mantiene T; balance de energía no incluido)",
                   "equilibrio líquido-vapor instantáneo en el tambor; vapor gas ideal; líquido de densidad molar constante "
                   "(la del usuario); retención de vapor incluida en el balance",
                   "K constantes (los del usuario) independientes de P y de la composición" if dr.model == "k_values" else
                   "volatilidad relativa constante con ley de Raoult K_i = α_i·P_ref/P (mezcla ideal)",
                   "válvula de vapor V = C·f(x)·√(P − P_abajo) en mol/s (forma de caída de presión moderada, C del usuario)"
                   if dr.out["vapour"]["type"] == "valve" else "caudal de vapor impuesto",
                   "sección del tambor constante (nivel = volumen de líquido / área)"]
    events_out = [{**e, "value": dict(zip(dr.comp, e["value"])) if e["target"] == "feed.mole_fractions" else e["value"]}
                  for e in evs]
    return clean({
        "time_s": sgrid, "series": coarse, "metrics": metrics, "final_state": out_end, "at_steady_state": at_steady,
        "initial_steady_state": ss_info, "events": events_out,
        "rachford_rice_check": rr, "settling_band": SETTLE_BAND, "t_reached_s": t_reached,
        "warnings": warnings, "assumptions": assumptions,
        "not_included": ["balance de energía (flash adiabático), calor de vaporización y cambio de T",
                         "dependencia de K con T y composición (modelos de actividad o ecuaciones de estado)",
                         "arrastre de líquido, espuma, hidráulica de la válvula de vapor compresible (IEC 60534 para gas)",
                         "control PID integrado (use valores de apertura por eventos o el motor de flowsheet para lazos)"],
        "method": (f"EDO de moles por componente con equilibrio algebraico (Rachford-Rice y volumen, scipy brentq) en cada "
                   f"evaluación, integradas con scipy {scipy.__version__} solve_ivp/{method} (rtol {rtol:g}); reinicio en "
                   "cada evento; integrales de alimentación y productos junto con el estado"),
        "evidence": {"verified": not reasons, "reasons": reasons, "balances": bal, "worst_balance_closure_rel": worst,
                     "final_normalized_change_over_window": steady_change, "tolerance_refinement": refinement,
                     "scheduled_events": [{"time_s": e["time_s"], "hit_exactly": e["time_s"] in info["restarts"]}
                                          for e in evs if e["inside_horizon"] and e["time_s"] > 0.0],
                     "integrator": {"segments": len(segs), "steps": sum(s[4] for s in segs), "success": info["success"]}},
    })
