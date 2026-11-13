"""PID tuning rules from a FOPDT (or SOPDT) model, with each rule's validity range and source.

Model: g(s) = k e^{-θs} / ((τ1 s + 1)(τ2 s + 1)); integrating: k' e^{-θs} / (s (τ2 s + 1)) with k' = slope.

Rules
- SIMC (Skogestad 2003, J. Process Control 13:291, eqs. 23-25 and Table 1), series form, default τc = θ (eq. 28):
  Kc = τ1 / (k (τc+θ)), τI = min(τ1, 4(τc+θ)), τD = τ2; integrating: Kc = 1/(k'(τc+θ)), τI = 4(τc+θ).
- IMC / direct synthesis (Seborg et al., Table 12.1, from Chien & Fruehauf 1990; Rivera, Morari & Skogestad 1986):
  FOPDT PI  Kc = τ/(k(τc+θ)), τI = τ  (Seborg Ex. 12.4a);
  FOPDT PID (Padé) Kc = (τ+θ/2)/(k(τc+θ/2)), τI = τ+θ/2, τD = τθ/(2τ+θ)  (ideal form; equals Skogestad eq. 34);
  SOPDT PID Kc = (τ1+τ2)/(k(τc+θ)), τI = τ1+τ2, τD = τ1τ2/(τ1+τ2);
  integrating PI (integrator approximation) Kc = (2τc+θ)/(k'(τc+θ)²), τI = 2τc+θ  (Seborg Ex. 12.4b).
- Ziegler-Nichols ultimate cycle (1942; Skogestad 2003 §5.3): from Ku, Pu of the model, P 0.5Ku; PI 0.45Ku, Pu/1.2;
  PID ideal 0.6Ku, Pu/2, Pu/8 (= series 0.3Ku, Pu/4, Pu/4).
- Ziegler-Nichols reaction curve (1942 open-loop): slope R = k/τ (= k'), L = θ: P 1/(Rθ); PI 0.9/(Rθ), 3.33θ;
  PID 1.2/(Rθ), 2θ, 0.5θ (ideal form).
- Cohen-Coon (1953), ideal form, r = θ/τ: P (τ/kθ)(1+r/3); PI (τ/kθ)(0.9+r/12), τI = θ(30+3r)/(9+20r);
  PID (τ/kθ)(4/3+r/4), τI = θ(32+6r)/(13+8r), τD = 4θ/(11+2r).
- Tyreus-Luyben (1992) PI: Kc = Ku/3.2, τI = 2.2 Pu (Skogestad 2003 §5.4).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from packages.dynamics.loop import Controller
from packages.dynamics.process import Process

RULES = ("SIMC", "IMC", "ZN", "ZN_reaction_curve", "Cohen_Coon", "Tyreus_Luyben")

SOURCES = {
    "SIMC": "Skogestad (2003) J. Process Control 13:291-309, ecs. 23-28 y Tabla 1",
    "IMC": "Seborg, Edgar, Mellichamp y Doyle, Tabla 12.1 (Chien y Fruehauf 1990); Rivera, Morari y Skogestad (1986)",
    "ZN": "Ziegler y Nichols (1942), ciclo último; forma ideal → serie según Skogestad (2003) §5.3",
    "ZN_reaction_curve": "Ziegler y Nichols (1942), curva de reacción (lazo abierto)",
    "Cohen_Coon": "Cohen y Coon (1953)",
    "Tyreus_Luyben": "Tyreus y Luyben (1992), citado en Skogestad (2003) §5.4",
}


def ultimate(p: Process) -> dict | None:
    """Ultimate gain and period of the process (P-only loop at the stability limit): first ω with ∠g = −180°."""
    sc = p.time_scales()
    if not sc:
        return None
    w = np.logspace(math.log10(1e-4 / max(sc)), math.log10(1e3 / min(sc)), 20000)
    g = p.freq(w)
    for i in np.nonzero(np.diff(np.sign(g.imag)))[0]:
        if g.real[i] < 0 and g.real[i + 1] < 0:
            wu = brentq(lambda x: p.freq(np.array([x]))[0].imag, w[i], w[i + 1])
            return {"Ku": 1 / abs(p.freq(np.array([wu]))[0]), "Pu": 2 * math.pi / wu, "w180": wu}
    return None


def _ratio(m: dict) -> float:
    return 0.0 if m["integrating"] else (math.inf if m["tau1"] == 0 else m["theta"] / m["tau1"])


def simc(m: dict, ctype: str, tau_c: float) -> tuple[Controller | None, str]:
    k, th, t1, t2 = m["k"], m["theta"], m["tau1"], m.get("tau2", 0.0)
    if ctype == "PID" and not t2 > 0:
        return None, ("SIMC-PID necesita un modelo de segundo orden (τ2): la derivada cancela la segunda constante "
                      "de tiempo; con un FOPDT SIMC recomienda PI (Skogestad 2003 §3.3)")
    s = tau_c + th
    if m["integrating"]:
        Kc, tI = 1 / (k * s), 4 * s
    else:
        Kc, tI = t1 / (k * s), min(t1, 4 * s)
    if ctype == "P":
        return Controller("P", Kc), ""
    if Kc == 0:                                       # pure delay: integral-only controller, KI = 1/(k(τc+θ))
        return Controller("I", 0.0, math.inf, 0.0, "series", 1 / (k * s)), ""
    return Controller(ctype, Kc, tI, t2 if ctype == "PID" else 0.0, "series"), ""


def imc(m: dict, ctype: str, tau_c: float) -> tuple[Controller | None, str]:
    k, th, t1, t2 = m["k"], m["theta"], m["tau1"], m.get("tau2", 0.0)
    if m["integrating"]:
        if ctype != "PI":
            return None, "IMC para proceso integrante implementado solo como PI (Seborg Tabla 12.1, aprox. integrador)"
        return Controller("PI", (2 * tau_c + th) / (k * (tau_c + th) ** 2), 2 * tau_c + th, 0.0, "ideal"), ""
    if ctype == "P":
        return None, "IMC no da un controlador P para FOPDT (siempre tiene acción integral)"
    if ctype == "PI":
        if t1 == 0:
            return Controller("I", 0.0, math.inf, 0.0, "ideal", 1 / (k * (tau_c + th))), ""
        return Controller("PI", t1 / (k * (tau_c + th)), t1, 0.0, "ideal"), ""
    if t2 > 0:
        return Controller("PID", (t1 + t2) / (k * (tau_c + th)), t1 + t2, t1 * t2 / (t1 + t2), "ideal"), ""
    if th == 0:
        return None, "IMC-PID con aproximación de Padé necesita θ > 0"
    return Controller("PID", (t1 + th / 2) / (k * (tau_c + th / 2)), t1 + th / 2, t1 * th / (2 * t1 + th), "ideal"), ""


def zn_ultimate(u: dict | None, ctype: str) -> tuple[Controller | None, str]:
    if u is None:
        return None, "el modelo nunca llega a −180° de fase: no hay ganancia última (ZN no aplica)"
    Ku, Pu = u["Ku"], u["Pu"]
    if ctype == "P":
        return Controller("P", 0.5 * Ku, form="ideal"), ""
    if ctype == "PI":
        return Controller("PI", 0.45 * Ku, Pu / 1.2, form="ideal"), ""
    return Controller("PID", 0.6 * Ku, Pu / 2, Pu / 8, "ideal"), ""


def zn_reaction(m: dict, ctype: str) -> tuple[Controller | None, str]:
    th = m["theta"]
    if th <= 0:
        return None, "la curva de reacción de ZN necesita tiempo muerto θ > 0"
    R = m["k"] if m["integrating"] else (m["k"] / m["tau1"] if m["tau1"] > 0 else math.inf)
    if not math.isfinite(R):
        return None, "proceso de retardo puro (τ = 0): la pendiente es infinita"
    if ctype == "P":
        return Controller("P", 1 / (R * th), form="ideal"), ""
    if ctype == "PI":
        return Controller("PI", 0.9 / (R * th), th / 0.3, form="ideal"), ""
    return Controller("PID", 1.2 / (R * th), 2 * th, 0.5 * th, "ideal"), ""


def cohen_coon(m: dict, ctype: str) -> tuple[Controller | None, str]:
    if m["integrating"] or m["tau1"] == 0 or m["theta"] <= 0:
        return None, "Cohen-Coon necesita un FOPDT autorregulado con τ > 0 y θ > 0"
    k, th, t = m["k"], m["theta"], m["tau1"]
    r = th / t
    base = t / (k * th)
    if ctype == "P":
        return Controller("P", base * (1 + r / 3), form="ideal"), ""
    if ctype == "PI":
        return Controller("PI", base * (0.9 + r / 12), th * (30 + 3 * r) / (9 + 20 * r), form="ideal"), ""
    return Controller("PID", base * (4 / 3 + r / 4), th * (32 + 6 * r) / (13 + 8 * r), 4 * th / (11 + 2 * r), "ideal"), ""


def tyreus_luyben(u: dict | None, ctype: str) -> tuple[Controller | None, str]:
    if ctype != "PI":
        return None, "Tyreus-Luyben implementado solo como PI (la versión PID no se verificó contra su fuente)"
    if u is None:
        return None, "sin ganancia última: Tyreus-Luyben no aplica"
    return Controller("PI", u["Ku"] / 3.2, 2.2 * u["Pu"], form="ideal"), ""


def validity(rule: str, m: dict, tau_c: float | None, ctype: str) -> dict:
    """Validity range of each rule and whether this model falls inside it."""
    r = _ratio(m)
    th, t1 = m["theta"], m["tau1"]
    if rule == "SIMC":
        note = ("válida para todo θ/τ (de retardo puro a integrante); con τc = θ: Ms 1.59-1.70, GM 2.96-3.14, "
                "PM 46.9-61.4° (Skogestad 2003, Tabla 2). τc < θ = más rápido y menos robusto")
        if ctype == "PID" and m.get("tau2", 0) and m["tau2"] < th:
            note += "; τ2 < θ: la derivada mejora poco frente al PI (Skogestad 2003 §4.2)"
        return {"range": "todo θ/τ; τc ≥ θ recomendado", "in_range": tau_c is None or tau_c >= 0.5 * th, "note": note}
    if rule == "IMC":
        ok = tau_c is not None and tau_c > 0.8 * th and (m["integrating"] or tau_c > 0.1 * t1)
        note = "guía τc/θ > 0.8 y τc > 0.1τ (Rivera et al. 1986, citado por Seborg)"
        if not m["integrating"] and r < 0.1:
            note += ("; proceso dominado por atraso (θ/τ < 0.1): τI = τ da un rechazo de carga muy lento "
                     "(Seborg §12.3); usa SIMC (τI acotado) o la aproximación de integrador")
        return {"range": "τc/θ > 0.8 y τc > 0.1τ", "in_range": bool(ok), "note": note}
    if rule in ("ZN", "ZN_reaction_curve", "Cohen_Coon"):
        ok = 0.1 <= r <= 1.0
        return {"range": "0.1 ≤ θ/τ ≤ 1 (rango orientativo habitual en textos; no tomado de la publicación original)",
                "in_range": bool(ok),
                "note": "diseñadas para razón de decaimiento 1/4: agresivas y poco robustas para la mayoría de los "
                        "lazos de proceso (Skogestad 2003 §5)"}
    return {"range": "pensada para procesos integrantes / lazos lentos (Tyreus y Luyben 1992)", "in_range": None,
            "note": "más conservadora que ZN; necesita Ku y Pu"}


def tune(m: dict, ctype: str, rules: list[str], tau_c: float | None, u: dict | None) -> list[dict]:
    """Settings for every requested rule. tau_c None → default per rule (θ for SIMC and IMC)."""
    out = []
    for rule in rules:
        tc = tau_c if tau_c is not None else (m["theta"] if rule in ("SIMC", "IMC") else None)
        if rule in ("SIMC", "IMC") and (tc is None or tc + m["theta"] <= 0):
            c, why = None, "θ = 0: SIMC/IMC necesitan τc > 0 dado por el usuario (con τc = θ = 0 la ganancia es infinita)"
        elif rule == "SIMC":
            c, why = simc(m, ctype, tc)
        elif rule == "IMC":
            c, why = imc(m, ctype, tc)
        elif rule == "ZN":
            c, why = zn_ultimate(u, ctype)
        elif rule == "ZN_reaction_curve":
            c, why = zn_reaction(m, ctype)
        elif rule == "Cohen_Coon":
            c, why = cohen_coon(m, ctype)
        elif rule == "Tyreus_Luyben":
            c, why = tyreus_luyben(u, ctype)
        else:
            raise ValueError(f"regla desconocida: {rule}; opciones {RULES}")
        row = {"rule": rule, "source": SOURCES[rule], "tau_c": tc if rule in ("SIMC", "IMC") else None,
               "validity": validity(rule, m, tc, ctype)}
        if c is None:
            row.update(applicable=False, reason=why)
        else:
            row.update(applicable=True, controller=c)
        out.append(row)
    return out
