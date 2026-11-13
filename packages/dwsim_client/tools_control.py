"""Agent tools for process control (bloque M9): step-test identification, PID tuning and closed-loop evaluation.
Engines and evidence live in `packages.dynamics` (python-control, BSD, is the independent check; the dead time is
exact everywhere, Padé only inside the check)."""
from __future__ import annotations

import math

import numpy as np

from packages.dynamics import identify as ID
from packages.dynamics import loop as LP
from packages.dynamics import tuning as TU
from packages.dynamics.process import Process, from_dict, half_rule

MARGIN_REL_TOL = 0.01        # exact margins vs python-control on a Padé(8) loop
IE_REL_TOL = 0.005           # integrated error of the simulation vs its closed form
PADE_IAE_REL_TOL = 0.05      # informative: python-control step response of the Padé loop


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def _rel(a, b, scale: float = 0.0):
    """|a − b| relative to max(|b|, scale): `scale` (e.g. the IAE) keeps a closed form of 0 comparable."""
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return 0.0 if a == b else math.inf
    return abs(a - b) / max(abs(b), abs(scale), 1e-12)


def evaluate(p: Process, c: LP.Controller, alpha: float, r: float, d: float, margins_alpha: float = 0.0,
             trajectories: bool = True) -> dict:
    """Margins (exact + python-control), stability, setpoint and load responses, and the checks that seal them."""
    out = {"controller": c.as_dict()}
    m = LP.margins(p, c, margins_alpha)
    pc = LP.margins_python_control(p, c, margins_alpha)
    checks = {"margins_vs_python_control": {
        "gain_margin_rel_diff": _rel(m["gain_margin"], pc["gain_margin"]),
        "phase_margin_rel_diff": _rel(m["phase_margin_deg"], pc["phase_margin_deg"]),
        "Ms_rel_diff": _rel(m["Ms"], pc["Ms"]), "python_control": pc}}
    ok_m = all(checks["margins_vs_python_control"][k] <= MARGIN_REL_TOL
               for k in ("gain_margin_rel_diff", "phase_margin_rel_diff", "Ms_rel_diff"))
    out["margins"] = m
    st = LP.stable(p, c, alpha)
    out["closed_loop_stable"] = st
    if not st:
        out.update(checks=checks, status="unstable",
                   message="el lazo cerrado es inestable con estos ajustes (polos del lazo con Padé de orden 10)")
        return out
    sims = {}
    ok_ie = True
    if r:
        s = LP.simulate(p, c, alpha, r, 0.0)
        sm = LP.setpoint_metrics(s, r)
        out["setpoint"] = sm
        sims["setpoint"] = s
        if c.KI:
            u_ss = 0.0 if p.integrating else r / p.gain
            srs = c.series() if c.tau_D > 0 else None
            extra = (1 - alpha) * srs["tau_D"] * r if srs is not None else 0.0
            ie = u_ss / c.KI + extra
            checks["setpoint_IE_closed_form"] = {"simulated": sm["IE"], "closed_form": ie,
                                                 "rel_diff": _rel(sm["IE"], ie, sm["IAE"])}
            ok_ie &= checks["setpoint_IE_closed_form"]["rel_diff"] <= IE_REL_TOL
        ok_ie &= s["settled"]
    if d:
        s = LP.simulate(p, c, alpha, 0.0, d)
        lm = LP.load_metrics(s, d)
        out["load"] = lm
        sims["load"] = s
        if c.KI:
            ie = -d / c.KI
            checks["load_IE_closed_form"] = {"simulated": lm["IE"], "closed_form": ie, "rel_diff": _rel(lm["IE"], ie)}
            ok_ie &= checks["load_IE_closed_form"]["rel_diff"] <= IE_REL_TOL
        else:
            lm["steady_state_offset"] = float(s["y"][-1])
        ok_ie &= s["settled"]
    # informative: same IAE from python-control's step response of the Padé loop
    try:
        pade = {}
        if "setpoint" in sims:
            pade["setpoint_IAE"] = LP.iae_python_control(p, c, alpha, r, 0.0, sims["setpoint"]["t"])["setpoint_IAE"]
        if "load" in sims:
            pade["load_IAE"] = LP.iae_python_control(p, c, alpha, 0.0, d, sims["load"]["t"])["load_IAE"]
        checks["IAE_vs_python_control_pade"] = {
            k: {"pade": v, "exact_delay": out["setpoint" if k.startswith("setpoint") else "load"]["IAE"],
                "rel_diff": _rel(v, out["setpoint" if k.startswith("setpoint") else "load"]["IAE"])}
            for k, v in pade.items()}
    except Exception as e:  # noqa: BLE001 — informative check only
        checks["IAE_vs_python_control_pade"] = {"error": str(e)[:200]}
    if trajectories:
        out["trajectories"] = {k: LP.downsample(v) for k, v in sims.items()}
    out["simulation"] = {k: {"step": v["h"], "steps": v["steps"], "settled": v["settled"]} for k, v in sims.items()}
    out["checks"] = checks
    out["status"] = "verified" if (ok_m and ok_ie) else "not_verified"
    return out


# ---------------------------------------------------------------------------------------------------------------
def fopdt_identify(time_s: list[float], output: list[float], input_change: float, step_time_s: float,
                   baseline_output: float | None = None, final_output: float | None = None) -> dict:
    """Identify a process model from an open-loop step test (controller in manual): fits FOPDT
    K·e^(−θs)/(τs+1) and SOPDT K·e^(−θs)/((τ1s+1)(τ2s+1)) by least squares and reports R², RMSE (also as % of the
    step), max residual, residual lag-1 autocorrelation (≈0 = only noise left; ≈1 = model mismatch) and 95 %
    confidence half-widths of K, τ, θ. Cross-checks with the two-point methods of Smith (28.3/63.2 %) and
    Sundaresan-Krishnaswamy (35.3/85.3 %) and flags disagreement, unfinished (integrating?) responses, overshoot and
    inverse response. time_s/output = the user's samples; input_change = Δu of the step; step_time_s = when it was
    applied (all the user's data: never assume them). baseline_output/final_output optional (otherwise estimated
    from the data and listed in assumptions). Next: pass K, τ, θ (and τ2 for PID) to pid_tuning."""
    r = ID.identify(time_s, output, input_change, step_time_s, baseline_output, final_output)
    f = r["fopdt"]
    r["next_step"] = (f"pid_tuning(gain={f['K']:.6g}, time_constant={f['tau']:.6g}, dead_time={f['theta']:.6g}) "
                      "— o con second_time_constant del SOPDT si se quiere PID")
    r["evidence"] = ("ajuste por mínimos cuadrados (scipy least_squares) con IC 95 % linealizados; contraste con Smith "
                     "(28.3/63.2 %) y Sundaresan-Krishnaswamy (35.3/85.3 %); autocorrelación de residuos ≈ 0 indica "
                     "que el modelo explica los datos y lo que queda es ruido")
    r["assumptions"].append("unidades de tiempo = las de time_s; K en unidades de salida por unidad de entrada")
    return _clean(r)


def pid_tuning(gain: float | None = None, time_constant: float | None = None, dead_time: float | None = None,
               second_time_constant: float | None = None, integrating: bool = False,
               process_model: dict | None = None, controller_type: str = "PI", rules: list[str] | None = None,
               tau_c: float | None = None, evaluate_closed_loop: bool = True, setpoint_step: float = 1.0,
               load_step: float = 1.0, derivative_filter_alpha: float = 0.01,
               sampling_period: float | None = None) -> dict:
    """PID tuning from a process model with standard rules, each with its source and validity range, and every
    result evaluated in closed loop. Give the FOPDT gain, time_constant, dead_time (from the user or fopdt_identify;
    second_time_constant = τ2 of a SOPDT for SIMC-PID; integrating=true with gain = slope k' for level-type
    processes), OR process_model = {'gain', 'time_constants' [lags], 'lead_time_constants' [numerator; negative =
    inverse response], 'dead_time', 'integrating'} which is reduced by Skogestad's half rule (FOPDT for P/PI, SOPDT
    for PID) and used in full for Ku/Pu and the closed-loop evaluation. No process value has a default.
    controller_type P|PI|PID. rules ⊆ SIMC (default recommendation, Skogestad 2003), IMC (Seborg Table 12.1 /
    Rivera 1986), ZN (ultimate cycle, Ku/Pu from the model), ZN_reaction_curve, Cohen_Coon, Tyreus_Luyben (PI).
    tau_c = closed-loop time constant for SIMC/IMC (default θ, stated in assumptions). Each rule returns Kc, τI, τD
    (native form plus series/ideal equivalents), validity (range, in_range, note), and with evaluate_closed_loop:
    Ms, Mt, GM, PM, crossover frequencies, setpoint (overshoot, rise/settling time, IAE, TV) and load-step metrics
    (peak, IAE), and status 'verified' only if the exact margins match python-control and the simulated integrated
    error matches its closed form. summary = one row per rule for a comparison table."""
    ctype = controller_type.upper()
    if ctype not in ("P", "PI", "PID"):
        raise ValueError("controller_type debe ser P, PI o PID")
    rules = list(rules or TU.RULES)
    assumptions = []
    full = from_dict(process_model) if process_model else None
    given = gain is not None
    if given:
        if dead_time is None or (time_constant is None and not integrating):
            raise ValueError("faltan datos del proceso: gain, time_constant (o integrating=true) y dead_time son del usuario")
        m = {"k": float(gain), "theta": float(dead_time), "integrating": bool(integrating),
             "tau1": math.inf if integrating else float(time_constant), "tau2": float(second_time_constant or 0.0)}
        approx = None
    elif full is not None:
        h = float(sampling_period or 0.0)
        approx = {"FOPDT": half_rule(full, 1, h), "SOPDT": half_rule(full, 2, h)}
        a = approx["SOPDT" if ctype == "PID" else "FOPDT"]
        m = {"k": a["k"], "theta": a["theta"], "integrating": a["integrating"], "tau1": a["tau1"], "tau2": a["tau2"]}
        assumptions.append("modelo reducido por la regla de la mitad (Skogestad 2003, ecs. 10-11); "
                           + ("SOPDT para PID" if ctype == "PID" else "FOPDT para P/PI"))
    else:
        raise ValueError("da el modelo del proceso: gain/time_constant/dead_time (FOPDT, del usuario o de fopdt_identify) "
                         "o process_model completo; no hay valores por defecto")
    if tau_c is None:
        assumptions.append("τc = θ (regla SIMC de Skogestad 2003 ec. 28; también usada para IMC): decisión de sintonía, "
                           "no dato del proceso; más grande = más lento y robusto")
    if m["theta"] == 0 and tau_c is None:
        assumptions.append("θ = 0: SIMC/IMC necesitan τc del usuario")
    # process used for Ku/Pu and for the closed-loop evaluation
    if full is not None:
        p_eval = full
    elif m["integrating"]:
        p_eval = Process(m["k"], [m["tau2"]] if m["tau2"] > 0 else [], [], m["theta"], True)
    else:
        p_eval = Process(m["k"], [t for t in (m["tau1"], m["tau2"]) if t > 0], [], m["theta"])
    u = TU.ultimate(p_eval)
    rows = TU.tune(m, ctype, rules, tau_c, u)
    for row in rows:
        if not row["applicable"]:
            continue
        c = row.pop("controller")
        if evaluate_closed_loop:
            ev = evaluate(p_eval, c, derivative_filter_alpha, setpoint_step, load_step, trajectories=False)
            row.update(ev)
        else:
            row["controller"] = c.as_dict()
    summary = []
    for row in rows:
        if not row["applicable"]:
            summary.append({"rule": row["rule"], "applicable": False, "reason": row["reason"]})
            continue
        c = row["controller"]
        s = {"rule": row["rule"], "Kc": c["Kc"], "tau_I": c["tau_I"], "tau_D": c["tau_D"], "form": c["form"],
             "in_validity_range": row["validity"]["in_range"]}
        if "margins" in row:
            s.update(Ms=row["margins"]["Ms"], GM=row["margins"]["gain_margin"], PM_deg=row["margins"]["phase_margin_deg"],
                     stable=row["closed_loop_stable"], status=row["status"])
            if "setpoint" in row:
                s.update(setpoint_IAE=row["setpoint"]["IAE"], overshoot_pct=row["setpoint"]["overshoot_pct"],
                         settling_time=row["setpoint"]["settling_time_2pct"])
            if "load" in row:
                s.update(load_IAE=row["load"]["IAE"], load_peak=row["load"]["peak_deviation"])
        summary.append(s)
    simc = next((r for r in rows if r["rule"] == "SIMC"), None)
    if simc is not None and simc["applicable"]:
        rec = "SIMC"
        why = ("SIMC con τc = θ: buen compromiso velocidad/robustez (Ms ≈ 1.6-1.7) y rechazo de carga razonable; "
               "ZN y Cohen-Coon apuntan a decaimiento 1/4 y suelen dar Ms > 2")
    elif simc is not None and ctype == "PID":
        rec = "SIMC-PI"
        why = simc["reason"] + "; pide controller_type='PI' o da second_time_constant"
    else:
        rec, why = None, "SIMC no aplica a este caso; compara las demás reglas por Ms y IAE"
    assumptions += [
        "escalón de consigna y de carga unitarios en variables de desviación salvo que se den setpoint_step/load_step",
        "la carga entra a la entrada del proceso antes del retardo (gd = g), como en Skogestad 2003 y Seborg Ej. 12.4",
        f"PID en forma serie con derivada solo sobre la medición filtrada, τF = α·τD, α = {derivative_filter_alpha} en la "
        "simulación; márgenes con α = 0 (como Skogestad 2003)",
        f"tiempo de asentamiento con banda de ±{LP.SETTLE_BAND * 100:.0f} %",
    ]
    out = {"model_used_for_tuning": m, "theta_over_tau": TU._ratio(m), "half_rule": approx,
           "evaluated_on": "process_model completo" if full is not None else "modelo FOPDT/SOPDT dado",
           "ultimate": u, "controller_type": ctype, "recommended_rule": rec, "recommendation_reason": why,
           "summary": summary, "rules": rows, "assumptions": assumptions,
           "evidence": ("márgenes exactos (retardo e^{-jωθ}) contra python-control con Padé(8) "
                        f"(tolerancia {MARGIN_REL_TOL * 100:.0f} %); error integrado de cada simulación contra su forma "
                        f"cerrada (d/KI y u∞/KI + τD, tolerancia {IE_REL_TOL * 100:.1f} %); status = verified solo si "
                        "ambas pasan")}
    return _clean(out)


def pid_closed_loop(process_model: dict, controller: dict, setpoint_step: float = 1.0, load_step: float = 1.0,
                    derivative_filter_alpha: float = 0.01) -> dict:
    """Closed-loop evaluation of a GIVEN controller on a GIVEN process (e.g. the plant's current settings, or a
    FOPDT-tuned controller on the full higher-order model). process_model = {'gain', 'time_constants',
    'lead_time_constants', 'dead_time', 'integrating'} (dead time exact, never Padé); controller = {'Kc', 'tau_I'
    (omit = no integral), 'tau_D', 'form': 'series'|'ideal'} or {'Kc': 0, 'KI'} for pure integral. Returns stability,
    GM, PM, Ms, Mt, allowed delay error, setpoint and load-step responses (overshoot, settling time, IAE, ISE, peak,
    TV of the input, downsampled trajectories t/y/u) and the checks (margins vs python-control on a Padé loop,
    integrated error vs closed form). setpoint_step/load_step in the user's deviation units (default 1 = per unit)."""
    p = from_dict(process_model)
    c = LP.controller_from_dict(controller)
    ev = evaluate(p, c, derivative_filter_alpha, setpoint_step, load_step)
    ev["process"] = p.describe()
    ev["ultimate"] = TU.ultimate(p)
    ev["assumptions"] = [
        "la carga entra a la entrada del proceso antes del retardo (gd = g)",
        f"PID serie con derivada solo sobre la medición filtrada (τF = α·τD, α = {derivative_filter_alpha}); "
        "una forma ideal con τI < 4τD se simula como PID ideal de 2 grados de libertad; márgenes con α = 0",
        f"tiempo de asentamiento con banda de ±{LP.SETTLE_BAND * 100:.0f} %",
    ]
    return _clean(ev)


CONTROL_TOOLS = [fopdt_identify, pid_tuning, pid_closed_loop]
