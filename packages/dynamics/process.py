"""Linear process models in the factored form used by process-control texts, and the SIMC half rule.

    g(s) = k · Π(T_j s + 1) / ( s^n · Π(τ_i s + 1) ) · e^{-θ s},   n = 1 for an integrating process (k = slope k')

Lags τ_i > 0 (stable), leads T_j of either sign (T_j < 0 = inverse response, right-half-plane zero), dead time θ ≥ 0.
The frequency response is evaluated exactly (the delay is e^{-jωθ}, never a Padé approximant); a rational realisation
of the delay-free part feeds the time simulation in `loop`.

Half rule (Skogestad 2003, J. Process Control 13, 291-309, eqs. 10-11): the largest neglected lag is split evenly
between the effective delay and the smallest retained time constant; smaller lags, inverse-response time constants
and half the sampling period add to the delay. Positive numerator time constants need rules T1-T3 (eq. 12), which are
not automated here: the caller gets a clear error and may pass the approximation directly."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Process:
    gain: float
    lags: list[float] = field(default_factory=list)
    leads: list[float] = field(default_factory=list)
    delay: float = 0.0
    integrating: bool = False

    def __post_init__(self):
        if not math.isfinite(self.gain) or self.gain == 0:
            raise ValueError("la ganancia del proceso debe ser finita y distinta de cero")
        if any((not math.isfinite(t)) or t <= 0 for t in self.lags):
            raise ValueError("las constantes de tiempo (atrasos) deben ser > 0; procesos inestables no están cubiertos")
        if any((not math.isfinite(t)) or t == 0 for t in self.leads):
            raise ValueError("las constantes de adelanto deben ser ≠ 0 (negativas = respuesta inversa)")
        if not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError("el tiempo muerto debe ser ≥ 0")
        if len(self.leads) > len(self.lags) + int(self.integrating):
            raise ValueError("modelo impropio: más adelantos que atrasos")
        self.lags = sorted((float(t) for t in self.lags), reverse=True)
        self.leads = [float(t) for t in self.leads]

    # --- frequency domain -------------------------------------------------------------------------------------
    def freq(self, w: np.ndarray, with_delay: bool = True) -> np.ndarray:
        jw = 1j * np.asarray(w, dtype=float)
        g = self.gain * np.ones_like(jw)
        for T in self.leads:
            g = g * (T * jw + 1)
        for t in self.lags:
            g = g / (t * jw + 1)
        if self.integrating:
            g = g / jw
        if with_delay and self.delay > 0:
            g = g * np.exp(-jw * self.delay)
        return g

    # --- rational part ----------------------------------------------------------------------------------------
    def polys(self) -> tuple[np.ndarray, np.ndarray]:
        num = np.array([self.gain])
        for T in self.leads:
            num = np.polymul(num, [T, 1.0])
        den = np.array([1.0])
        for t in self.lags:
            den = np.polymul(den, [t, 1.0])
        if self.integrating:
            den = np.polymul(den, [1.0, 0.0])
        return num, den

    def time_scales(self) -> list[float]:
        return [t for t in (*self.lags, *(abs(T) for T in self.leads), self.delay) if t > 0]

    def describe(self) -> dict:
        return {"gain": self.gain, "time_constants": self.lags, "lead_time_constants": self.leads,
                "dead_time": self.delay, "integrating": self.integrating}


def from_dict(d: dict) -> Process:
    """Parse the tool-facing dict: gain, time_constants, lead_time_constants, dead_time, integrating."""
    unknown = set(d) - {"gain", "time_constants", "lead_time_constants", "dead_time", "integrating"}
    if unknown:
        raise ValueError(f"claves desconocidas en el modelo del proceso: {sorted(unknown)}")
    if "gain" not in d or "dead_time" not in d:
        raise ValueError("el modelo del proceso necesita 'gain' y 'dead_time' (0 si no hay); son datos del usuario")
    return Process(float(d["gain"]), list(d.get("time_constants") or []), list(d.get("lead_time_constants") or []),
                   float(d["dead_time"]), bool(d.get("integrating", False)))


def half_rule(p: Process, order: int, sampling_period: float = 0.0) -> dict:
    """FOPDT (order 1) or SOPDT (order 2) approximation of p by the half rule. Returns k, tau1, tau2, theta and the
    contribution of every neglected term to theta (so the user can audit it)."""
    if order not in (1, 2):
        raise ValueError("order debe ser 1 (FOPDT) o 2 (SOPDT)")
    if any(T > 0 for T in p.leads):
        raise ValueError("el modelo tiene constantes de adelanto positivas (T·s+1 en el numerador): la regla de la "
                         "mitad sola no aplica; usa las reglas T1-T3 de Skogestad (2003, ec. 12) y pasa la aproximación "
                         "(gain, time_constant, dead_time, second_time_constant) directamente")
    inv = [-T for T in p.leads]                      # inverse-response time constants (> 0)
    lags = list(p.lags)                              # sorted descending
    parts = [("tiempo muerto original", p.delay)] + [(f"respuesta inversa T={t:g}", t) for t in inv]
    if sampling_period > 0:
        parts.append(("mitad del periodo de muestreo", sampling_period / 2))
    keep = order - (1 if p.integrating else 0)      # rational time constants retained besides the integrator
    retained = lags[:keep]
    neglected = lags[keep:]
    tau = list(retained)
    if neglected:
        big = neglected[0]
        parts.append((f"mitad del mayor atraso despreciado ({big:g})", big / 2))
        if tau:
            tau[-1] += big / 2                      # smallest retained constant gets the other half
        # (integrating, nothing retained: the other half goes to the integrator, i.e. is absorbed)
        parts += [(f"atraso despreciado {t:g}", t) for t in neglected[1:]]
    tau.sort(reverse=True)                           # τ1 ≥ τ2 (E4: 1 + 1/2 becomes τ1 = 1.5, τ2 = 1)
    theta = sum(v for _, v in parts)
    out = {"k": p.gain, "theta": theta, "integrating": p.integrating,
           "theta_contributions": [{"term": n, "value": v} for n, v in parts if v > 0]}
    if p.integrating:
        out["tau1"] = math.inf
        out["tau2"] = tau[0] if (order == 2 and tau) else 0.0
    else:
        out["tau1"] = tau[0] if tau else 0.0
        out["tau2"] = tau[1] if (order == 2 and len(tau) > 1) else 0.0
    return out
