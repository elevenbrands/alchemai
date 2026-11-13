"""PID controllers and closed-loop evaluation with an exact dead time.

Controller implementation (Skogestad 2003, eq. 2): series PID with the derivative on the filtered measurement only
(no derivative kick) and filter τF = α·τD,

    u = (Kc + KI/s) · [ r − (τD s + 1)/(α τD s + 1) · y ],      KI = Kc/τI,

PI and P are the special cases τD = 0 and KI = 0; a pure integral controller (SIMC on a pure delay) is Kc = 0. An
ideal-form PID that cannot be written in series form (τI < 4 τD) is run as an ideal two-degree-of-freedom PID
(P and I on the error, filtered D on the measurement). Every controller is one small state-space model, and both the
time simulation and the loop frequency response are built from it, so they cannot disagree.

Frequency analysis is exact: L(jω) = Cy(jω)·g(jω) with e^{-jωθ}; crossings are bracketed on a dense log grid and
refined with brentq. python-control (BSD) recomputes GM, PM and Ms on a Padé approximant as the independent check.

Time simulation: the delay-free plant and the controller form one LTI system discretised exactly (zero-order hold,
matrix exponential); the dead time is an exact shift of an integer number of steps h = θ/N. The load disturbance
enters at the process input before the delay (gd = g), as in Skogestad (2003) and Seborg et al. (Ex. 12.4). Checks:
the integrated error of a load step must equal d/KI (and of a setpoint step u_ss/KI + (1-α)τD), and python-control's
step response of the Padé loop must give the same IAE."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.signal as ss
from scipy.optimize import brentq, minimize_scalar

from packages.dynamics.process import Process

PADE_ORDER = 8
SETTLE_BAND = 0.02          # ±2 % band for settling times (stated in assumptions)


@dataclass
class Controller:
    """Native-form controller. form = 'series' (Skogestad eq. 1) or 'ideal' (ISA). tau_I = inf → no integral."""
    kind: str                 # 'P' | 'PI' | 'PID' | 'I'
    Kc: float
    tau_I: float = math.inf
    tau_D: float = 0.0
    form: str = "series"
    KI: float | None = None   # integral gain; required when Kc == 0 (pure I)

    def __post_init__(self):
        if self.KI is None:
            self.KI = 0.0 if math.isinf(self.tau_I) else self.Kc / self.tau_I
        if self.kind not in ("P", "PI", "PID", "I"):
            raise ValueError(f"tipo de controlador desconocido: {self.kind}")
        if self.tau_D < 0 or (not math.isinf(self.tau_I) and self.tau_I <= 0 and self.kind != "I"):
            raise ValueError("τI debe ser > 0 y τD ≥ 0")

    # conversions (Skogestad 2003 eqs. 35-36) ---------------------------------------------------------------------
    def ideal(self) -> dict:
        if self.form == "ideal" or self.tau_D == 0:
            return {"Kc": self.Kc, "tau_I": self.tau_I, "tau_D": self.tau_D, "KI": self.KI}
        f = 1 + self.tau_D / self.tau_I
        return {"Kc": self.Kc * f, "tau_I": self.tau_I + self.tau_D,
                "tau_D": self.tau_I * self.tau_D / (self.tau_I + self.tau_D), "KI": self.KI}

    def series(self) -> dict | None:
        if self.form == "series" or self.tau_D == 0:
            return {"Kc": self.Kc, "tau_I": self.tau_I, "tau_D": self.tau_D, "KI": self.KI}
        q = 1 - 4 * self.tau_D / self.tau_I
        if q < -1e-9:
            return None
        r = math.sqrt(max(q, 0.0))
        return {"Kc": self.Kc / 2 * (1 + r), "tau_I": self.tau_I / 2 * (1 + r), "tau_D": self.tau_I / 2 * (1 - r),
                "KI": self.KI}

    def as_dict(self) -> dict:
        d = {"type": self.kind, "form": self.form, "Kc": self.Kc, "tau_I": None if math.isinf(self.tau_I) else self.tau_I,
             "tau_D": self.tau_D, "KI": self.KI}
        if self.tau_D > 0:
            s, i = self.series(), self.ideal()
            d["series_equivalent"] = s and {k: v for k, v in s.items() if k != "KI"}
            d["ideal_equivalent"] = {k: v for k, v in i.items() if k != "KI"}
        return d

    # state space: xc' = Ac xc + by y + br r ;  u = cx xc + dy y + dr r ---------------------------------------------
    def ss(self, alpha: float):
        KI = self.KI or 0.0
        if self.tau_D > 0:
            if alpha <= 0:
                raise ValueError("el filtro derivativo α debe ser > 0 en la simulación")
            s = self.series()
            if s is not None:
                Kc, tF = s["Kc"], alpha * s["tau_D"]
                # xc = [xi, xf]; ỹ = y/α + (1-1/α) xf
                Ac = np.array([[0.0, -(1 - 1 / alpha)], [0.0, -1 / tF]])
                by = np.array([-1 / alpha, 1 / tF]); br = np.array([1.0, 0.0])
                cx = np.array([KI, -Kc * (1 - 1 / alpha)]); dy = -Kc / alpha; dr = Kc
                return Ac, by, br, cx, dy, dr
            i = self.ideal(); kp, kd = i["Kc"], i["Kc"] * i["tau_D"]; tF = alpha * i["tau_D"]
            Ac = np.array([[0.0, 0.0], [0.0, -1 / tF]])
            by = np.array([-1.0, 1 / tF]); br = np.array([1.0, 0.0])
            cx = np.array([KI, kd / tF]); dy = -kp - kd / tF; dr = kp
            return Ac, by, br, cx, dy, dr
        Kc = self.ideal()["Kc"]
        if KI > 0:
            return np.zeros((1, 1)), np.array([-1.0]), np.array([1.0]), np.array([KI]), -Kc, Kc
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0), np.zeros(0), -Kc, Kc

    def freq(self, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Cy(jω) (feedback path, u = Cr r − Cy y) and Cr(jω). alpha = 0 → unfiltered derivative (analytic)."""
        if alpha == 0:
            jw = 1j * np.asarray(w, dtype=float)
            s = self.series()
            if s is not None:
                pi = s["Kc"] + (self.KI or 0.0) / jw
                return pi * (s["tau_D"] * jw + 1), pi
            i = self.ideal()
            pi = i["Kc"] + (self.KI or 0.0) / jw
            return pi + i["Kc"] * i["tau_D"] * jw, pi
        Ac, by, br, cx, dy, dr = self.ss(alpha)
        n = Ac.shape[0]
        w = np.asarray(w, dtype=float)
        if n == 0:
            return -dy * np.ones_like(w, dtype=complex), dr * np.ones_like(w, dtype=complex)
        cy, cr = np.empty(w.shape, complex), np.empty(w.shape, complex)
        I = np.eye(n)
        for k, wk in enumerate(w.ravel()):
            M = 1j * wk * I - Ac
            cy.ravel()[k] = -(cx @ np.linalg.solve(M, by) + dy)
            cr.ravel()[k] = cx @ np.linalg.solve(M, br) + dr
        return cy, cr


def controller_from_dict(d: dict) -> Controller:
    """Tool-facing controller: {'Kc', 'tau_I' (omit/None = no integral), 'tau_D' (default 0), 'form' series|ideal}."""
    if "Kc" not in d:
        raise ValueError("el controlador necesita 'Kc' (y 'tau_I', 'tau_D' si aplica)")
    tI = d.get("tau_I")
    tI = math.inf if tI in (None, 0) and d.get("KI") is None else (math.inf if tI is None else float(tI))
    tD = float(d.get("tau_D") or 0.0)
    Kc = float(d["Kc"])
    KI = d.get("KI")
    kind = "PID" if tD > 0 else ("PI" if (not math.isinf(tI) or KI) else "P")
    if Kc == 0 and KI:
        kind = "I"
    return Controller(kind, Kc, tI, tD, d.get("form", "series"), None if KI is None else float(KI))


# ----------------------------------------------------------------------------------------------------------------
# frequency domain
# ----------------------------------------------------------------------------------------------------------------
def _grid(p: Process, c: Controller) -> np.ndarray:
    sc = p.time_scales() + [t for t in (c.tau_I, c.tau_D) if 0 < t < math.inf]
    lo, hi = 1e-4 / max(sc), 1e3 / min(sc)
    return np.logspace(math.log10(lo), math.log10(hi), 30000)


def margins(p: Process, c: Controller, alpha: float) -> dict:
    """GM, PM, crossover frequencies, Ms, Mt and allowed delay error from the exact loop L(jω) = Cy·g."""
    def L(w):
        return c.freq(np.atleast_1d(w), alpha)[0] * p.freq(np.atleast_1d(w))

    w = _grid(p, c)
    Lw = L(w)
    mag = np.abs(Lw)
    # gain crossovers (|L| = 1)
    f = np.log(mag)
    wc = [brentq(lambda x: math.log(abs(L(x)[0])), w[i], w[i + 1]) for i in np.nonzero(np.diff(np.sign(f)))[0]]
    pms = []
    for x in wc:
        ph = math.degrees(np.angle(L(x)[0]))
        pms.append(((ph + 180.0 + 180.0) % 360.0) - 180.0)
    # phase crossovers (Im L = 0, Re L < 0)
    im = Lw.imag
    wp = []
    for i in np.nonzero(np.diff(np.sign(im)))[0]:
        if Lw.real[i] < 0 and Lw.real[i + 1] < 0:
            x = brentq(lambda z: L(z)[0].imag, w[i], w[i + 1])
            wp.append(x)
    gms = [1.0 / abs(L(x)[0]) for x in wp]
    upper = [(g, x) for g, x in zip(gms, wp) if g > 1]
    lower = [(g, x) for g, x in zip(gms, wp) if g <= 1]
    # sensitivity peaks
    S, T = np.abs(1 / (1 + Lw)), np.abs(Lw / (1 + Lw))

    def peak(arr, fun):
        i = int(np.argmax(arr))
        a, b = w[max(i - 1, 0)], w[min(i + 1, len(w) - 1)]
        r = minimize_scalar(lambda z: -fun(z), bounds=(a, b), method="bounded", options={"xatol": a * 1e-9})
        return max(float(-r.fun), float(arr[i])), float(r.x)

    Ms, wMs = peak(S, lambda z: abs(1 / (1 + L(z)[0])))
    Mt, _ = peak(T, lambda z: abs(L(z)[0] / (1 + L(z)[0])))
    k = int(np.argmin(pms)) if pms else None
    gm = min(upper)[0] if upper else math.inf
    out = {"gain_margin": gm, "gain_margin_dB": 20 * math.log10(gm) if math.isfinite(gm) else math.inf,
           "phase_crossover_frequency": min(upper)[1] if upper else None,
           "phase_margin_deg": pms[k] if k is not None else math.inf,
           "gain_crossover_frequency": wc[k] if k is not None else None,
           "Ms": Ms, "Ms_frequency": wMs, "Mt": Mt,
           "allowed_delay_error": (math.radians(pms[k]) / wc[k]) if k is not None and pms[k] > 0 else None}
    if lower:
        out["lower_gain_margin"] = max(lower)[0]     # gain may only be reduced by this factor (conditionally stable)
    return out


def _pc_loop(p: Process, c: Controller, alpha: float, pade_order: int = PADE_ORDER):
    """python-control transfer functions of Cy, Cr and g (delay as Padé) — used only for the independent checks."""
    import control as ct
    if alpha == 0 and c.tau_D > 0:        # unfiltered PID: improper Cy, written directly as a polynomial ratio
        KI, srs = c.KI or 0.0, c.series()
        if srs is not None:
            pi = ct.tf([srs["Kc"], KI], [1, 0])
            Cy, Cr = pi * ct.tf([srs["tau_D"], 1], [1]), pi
        else:
            i = c.ideal()
            Cr = ct.tf([i["Kc"], KI], [1, 0])
            Cy = ct.tf([i["Kc"] * i["tau_D"], i["Kc"], KI], [1, 0])
        num, den = p.polys()
        G = ct.tf(num, den)
        if p.delay > 0:
            n, d = ct.pade(p.delay, pade_order)
            G = G * ct.tf(n, d)
        return Cy, Cr, G
    Ac, by, br, cx, dy, dr = c.ss(alpha)
    if Ac.shape[0]:
        Cy = -ct.ss2tf(ct.ss(Ac, by.reshape(-1, 1), cx.reshape(1, -1), [[dy]]))
        Cr = ct.ss2tf(ct.ss(Ac, br.reshape(-1, 1), cx.reshape(1, -1), [[dr]]))
    else:
        Cy, Cr = ct.tf([-dy], [1]), ct.tf([dr], [1])
    num, den = p.polys()
    G = ct.tf(num, den)
    if p.delay > 0:
        n, d = ct.pade(p.delay, pade_order)
        G = G * ct.tf(n, d)
    return Cy, Cr, G


def margins_python_control(p: Process, c: Controller, alpha: float) -> dict:
    import control as ct
    Cy, _, G = _pc_loop(p, c, alpha)
    gm, pm, sm, wpc, wgc, wms = ct.stability_margins(Cy * G)
    return {"gain_margin": float(gm), "phase_margin_deg": float(pm), "Ms": 1 / float(sm) if sm > 0 else math.inf,
            "pade_order": PADE_ORDER}


def stable(p: Process, c: Controller, alpha: float) -> bool:
    """Closed-loop stability from the poles of the Padé loop (high order; confirmed by the simulation decaying)."""
    import control as ct
    Cy, _, G = _pc_loop(p, c, alpha, pade_order=10)
    poles = ct.poles(ct.feedback(G * Cy, 1))
    return bool(np.all(np.real(poles) < -1e-9 * max(1.0, np.max(np.abs(poles)))))


# ----------------------------------------------------------------------------------------------------------------
# time domain
# ----------------------------------------------------------------------------------------------------------------
def _plant_ss(p: Process):
    num, den = p.polys()
    A, B, C, D = ss.tf2ss(num, den)
    return A, B.ravel(), C.ravel(), float(D.ravel()[0])


def simulate(p: Process, c: Controller, alpha: float, r: float, d: float, t_end: float | None = None,
             max_steps: int = 800_000) -> dict:
    """Closed-loop response to a setpoint step r and a load step d (both at t = 0, deviation variables). Runs until
    y and u have settled (or t_end / max_steps). Returns t, y, u, e, the step h and whether it settled."""
    Ap, Bp, Cp, Dp = _plant_ss(p)
    Ac, by, br, cx, dy, dr = c.ss(alpha)
    n_p, n_c = Ap.shape[0], Ac.shape[0]
    n = n_p + n_c
    scales = p.time_scales() + [t for t in (c.tau_I, c.tau_D) if 0 < t < math.inf]
    h_dyn = min(scales) / 20
    theta = p.delay
    N = max(200, int(math.ceil(theta / h_dyn))) if theta > 0 else 0
    h = theta / N if N else h_dyn
    # x' = A x + B [v, r]; v = process input (u + d, delayed by θ); y = Cp xp + Dp v; u = Cu x + Du_v v + Du_r r
    A = np.zeros((n, n)); B = np.zeros((n, 2))
    A[:n_p, :n_p] = Ap; B[:n_p, 0] = Bp
    A[n_p:, n_p:] = Ac; A[n_p:, :n_p] = np.outer(by, Cp); B[n_p:, 0] = by * Dp; B[n_p:, 1] = br
    Cu = np.concatenate([dy * Cp, cx]); Du_v, Du_r = dy * Dp, dr
    Cy_ = np.concatenate([Cp, np.zeros(n_c)])
    if N == 0:   # no delay: v = u + d inside the loop → v (1 − Du_v) = Cu x + Du_r r + d
        den = 1 - Du_v
        if abs(den) < 1e-12:
            raise ValueError("lazo algebraico sin solución (ganancia directa del lazo = 1)")
        Kx, Kr, Kd = Cu / den, Du_r / den, 1 / den
        A = A + np.outer(B[:, 0], Kx)
        B = np.column_stack([B[:, 0] * Kd, B[:, 1] + B[:, 0] * Kr])        # inputs [d, r]
    M = sla.expm(np.block([[A, B], [np.zeros((2, n + 2))]]) * h)
    Ad, B0, B1 = M[:n, :n], M[:n, n], M[:n, n + 1]
    tmax = t_end if t_end is not None else math.inf
    t_min = 10 * max(scales + [theta])
    cap = max_steps + 1
    Y, U = np.empty(cap), np.empty(cap)
    buf = np.zeros(max(N, 1))
    x = np.zeros(n)
    k, settled = 0, False
    while True:
        if N == 0:
            v = Kx @ x + Kr * r + Kd * d
            u = v - d
            x_in = d
        else:
            v = buf[k % N]
            u = Cu @ x + Du_v * v + Du_r * r
            buf[k % N] = u + d
            x_in = v
        Y[k] = Cy_ @ x + Dp * v
        U[k] = u
        if k and k % 2000 == 0 and k * h >= t_min:
            settled = _settled(Y[:k + 1], U[:k + 1])
            if settled:
                break
        if k >= max_steps or k * h >= tmax:
            break
        x = Ad @ x + B0 * x_in + B1 * r
        k += 1
    t = np.arange(k + 1) * h
    y, u = Y[:k + 1].copy(), U[:k + 1].copy()
    return {"t": t, "y": y, "u": u, "e": r - y, "h": h, "settled": settled or _settled(y, u), "steps": k}


def _settled(y: np.ndarray, u: np.ndarray) -> bool:
    k0 = int(len(y) * 0.8)
    if len(y) - k0 < 50:
        return False
    sy = max(float(np.max(np.abs(y))), 1e-12)
    su = max(float(np.max(np.abs(u))), 1e-12)
    return bool(np.ptp(y[k0:]) < 1e-5 * sy and np.ptp(u[k0:]) < 1e-5 * su)


def _first_cross(t, y, level):
    i = np.nonzero(y >= level)[0]
    if not len(i):
        return None
    i = i[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


def setpoint_metrics(sim: dict, r: float) -> dict:
    t, y, u, e = sim["t"], sim["y"] / r, sim["u"], sim["e"] / r
    over = max(0.0, float(np.max(y) - 1.0)) * 100
    t10, t90 = _first_cross(t, y, 0.1), _first_cross(t, y, 0.9)
    out_band = np.nonzero(np.abs(y - 1) > SETTLE_BAND)[0]
    return {"overshoot_pct": over, "rise_time_10_90": (t90 - t10) if t10 is not None and t90 is not None else None,
            "peak_time": float(t[int(np.argmax(y))]),
            "settling_time_2pct": float(t[out_band[-1] + 1]) if len(out_band) and out_band[-1] + 1 < len(t) else
            (0.0 if not len(out_band) else None),
            "IAE": float(np.trapezoid(np.abs(e), t)) * abs(r), "IE": float(np.trapezoid(e, t)) * r,
            "ISE": float(np.trapezoid(e ** 2, t)) * r * r,
            "TV_u": float(np.sum(np.abs(np.diff(u)))), "u_max_abs": float(np.max(np.abs(u)))}


def load_metrics(sim: dict, d: float) -> dict:
    t, y, u = sim["t"], sim["y"], sim["u"]
    i = int(np.argmax(np.abs(y)))
    peak = float(y[i])
    out_band = np.nonzero(np.abs(y) > SETTLE_BAND * abs(peak))[0]
    return {"peak_deviation": peak, "peak_time": float(t[i]),
            "settling_time_2pct_of_peak": float(t[out_band[-1] + 1]) if len(out_band) and out_band[-1] + 1 < len(t) else None,
            "IAE": float(np.trapezoid(np.abs(y), t)), "IE": float(np.trapezoid(-y, t)),
            "ISE": float(np.trapezoid(y ** 2, t)),
            "TV_u": float(np.sum(np.abs(np.diff(u)))), "u_max_abs": float(np.max(np.abs(u)))}


def iae_python_control(p: Process, c: Controller, alpha: float, r: float, d: float, t: np.ndarray) -> dict:
    """IAE of the Padé loop from python-control's step responses on the same time grid (independent check)."""
    import control as ct
    Cy, Cr, G = _pc_loop(p, c, alpha)
    S = ct.feedback(G, Cy)                         # y/d = G/(1+G Cy) (d at the process input)
    out = {}
    if r:
        T = Cr * S
        y = ct.step_response(T, T=t).outputs * r
        out["setpoint_IAE"] = float(np.trapezoid(np.abs(r - y), t))
    if d:
        y = ct.step_response(S, T=t).outputs * d
        out["load_IAE"] = float(np.trapezoid(np.abs(y), t))
    return out


def downsample(sim: dict, n: int = 150) -> dict:
    t = sim["t"]
    idx = np.unique(np.linspace(0, len(t) - 1, n).round().astype(int))
    return {"t": t[idx].tolist(), "y": sim["y"][idx].tolist(), "u": sim["u"][idx].tolist()}
