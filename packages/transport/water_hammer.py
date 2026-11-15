"""Water hammer in a reservoir-pipe-valve line: Joukowsky surge, wave speed with pipe elasticity, transient by the method
of characteristics (bloque MG).

What it answers: the pressure-wave speed a (fluid bulk modulus corrected for pipe-wall elasticity), the Joukowsky surge
ΔP = ρ·a·ΔV of a sudden closure, whether the valve closure is «fast» (t_c ≤ 2L/a: full Joukowsky surge) or «slow», and the
full transient after the valve starts closing: maximum and minimum pressure anywhere in the line, the pressure history at
the valve, and whether the pressure falls below the vapour pressure (cavitation / column separation flag).

Wave speed (thin-walled elastic pipe, the FIU lecture-notes form of Korteweg's formula):
    1/K_eff = 1/K + c1·D/(E·e)      a = √(K_eff/ρ)
with c1 the restraint factor of the pipe: given directly, or from the anchoring with Poisson's ratio ν
(free to move longitudinally 5/4 − ν, anchored at both ends 1 − ν², with expansion joints 1 − ν/2; A. S. Leon, «Water hammer
flows», FIU course notes — textbooks differ on these factors, give c1 yourself when your reference does).

Transient (Wylie-Streeter MOC, fixed grid with Courant number 1, dt = dx/a, so no interpolation):
    C+: H_P = C_P − B·Q_P,  C_P = H_A + B·Q_A − R·Q_A|Q_A|      C−: H_P = C_M + B·Q_P,  C_M = H_B − B·Q_B + R·Q_B|Q_B|
    B = a/(g·A),  R = f·Δx/(2·g·D·A²)  (quasi-steady Darcy friction, first-order explicit)
Upstream boundary: reservoir at constant head (entrance loss and velocity head neglected). Downstream: valve discharging to a
constant pressure, Q = τ(t)·Q0·√(ΔH/ΔH0), with τ the relative opening (1 = initial). Initial state: steady flow Q0.

Evidence of THIS run: the initial steady state is preserved by one MOC step with the valve open, and the extreme
pressures do not change (relative to the Joukowsky surge) when the grid is refined from N to 2N reaches.
"""
from __future__ import annotations

import math

import numpy as np

G_ACC = 9.80665
GRID_TOL = 0.01                  # change of the extreme surge, relative to ρ·a·V0, between N and 2N reaches
STEADY_TOL = 1e-9
ANCHORING = {"free_longitudinal": lambda nu: 1.25 - nu, "anchored_both_ends": lambda nu: 1.0 - nu * nu,
             "expansion_joints": lambda nu: 1.0 - 0.5 * nu}


def wave_speed(density_kg_m3: float, fluid_bulk_modulus_Pa: float, inner_diameter_m: float | None = None,
               wall_thickness_m: float | None = None, pipe_elastic_modulus_Pa: float | None = None,
               restraint_factor: float | None = None, pipe_anchoring: str | None = None,
               poisson_ratio: float | None = None) -> dict:
    """Pressure-wave speed in a liquid-filled thin-walled elastic pipe (rigid pipe if E, e are not given → √(K/ρ))."""
    a_rigid = math.sqrt(fluid_bulk_modulus_Pa / density_kg_m3)
    if pipe_elastic_modulus_Pa is None and wall_thickness_m is None:
        return {"wave_speed_m_s": a_rigid, "rigid_wave_speed_m_s": a_rigid, "restraint_factor": None,
                "method": "tubo rígido: a = √(K/ρ)"}
    if None in (pipe_elastic_modulus_Pa, wall_thickness_m, inner_diameter_m):
        raise ValueError("para la elasticidad del tubo faltan pipe_elastic_modulus_Pa, wall_thickness_m e inner_diameter_m")
    c1 = restraint_factor
    if c1 is None:
        if pipe_anchoring not in ANCHORING:
            raise ValueError(f"falta restraint_factor (c1) o pipe_anchoring ∈ {sorted(ANCHORING)} con poisson_ratio: "
                             "define cuánto se deforma el tubo axialmente")
        if poisson_ratio is None:
            raise ValueError("falta poisson_ratio del material del tubo para el factor de anclaje")
        c1 = ANCHORING[pipe_anchoring](poisson_ratio)
    D, e, E, K = inner_diameter_m, wall_thickness_m, pipe_elastic_modulus_Pa, fluid_bulk_modulus_Pa
    if D / e < 25.0:
        warn = f"D/e = {D / e:.1f} < 25: tubo de pared gruesa, la fórmula de pared delgada pierde exactitud"
    else:
        warn = None
    K_eff = 1.0 / (1.0 / K + c1 * D / (E * e))
    out = {"wave_speed_m_s": math.sqrt(K_eff / density_kg_m3), "rigid_wave_speed_m_s": a_rigid, "restraint_factor": c1,
           "effective_bulk_modulus_Pa": K_eff, "D_over_e": D / e,
           "method": "Korteweg (pared delgada): 1/K_eff = 1/K + c1·D/(E·e), a = √(K_eff/ρ)"}
    if warn:
        out["warning"] = warn
    return out


def _tau_fn(closure_time_s: float | None, closure_exponent: float | None, closure_curve: dict | None):
    if closure_curve is not None:
        t = np.asarray(closure_curve.get("time_s", []), float)
        tau = np.asarray(closure_curve.get("tau", []), float)
        if len(t) < 2 or len(t) != len(tau):
            raise ValueError("closure_curve debe ser {'time_s': [...], 'tau': [...]} de igual longitud (≥ 2)")
        if t[0] != 0.0 or abs(tau[0] - 1.0) > 1e-12 or np.any(np.diff(t) <= 0) or np.any(tau < 0) or np.any(tau > 1):
            raise ValueError("closure_curve: time_s creciente desde 0 y tau entre 0 y 1 con tau(0) = 1 (apertura inicial)")
        return (lambda s: float(np.interp(s, t, tau))), float(t[-1])
    if closure_time_s is None or closure_exponent is None:
        raise ValueError("falta la ley de cierre de la válvula: closure_time_s y closure_exponent (τ = (1 − t/tc)^Em) o "
                         "closure_curve {'time_s', 'tau'}")
    if closure_time_s < 0 or closure_exponent <= 0:
        raise ValueError("closure_time_s ≥ 0 y closure_exponent > 0")
    tc, Em = float(closure_time_s), float(closure_exponent)
    if tc == 0.0:
        return (lambda s: 0.0 if s > 0.0 else 1.0), 0.0
    return (lambda s: 0.0 if s >= tc else (1.0 - s / tc) ** Em), tc


def _moc(N: int, L: float, D: float, a: float, f: float, rho: float, Q0: float, H_R: float, P_down: float, dz: float,
         tau, t_end: float, keep_history: bool) -> dict:
    A = math.pi * D * D / 4.0
    dx = L / N
    dt = dx / a
    B = a / (G_ACC * A)
    R = f * dx / (2.0 * G_ACC * D * A * A)
    z = np.linspace(0.0, dz, N + 1)
    H = H_R - R * Q0 * Q0 * np.arange(N + 1)
    Q = np.full(N + 1, Q0)
    H_out = P_down / (rho * G_ACC) + dz
    dH0 = H[-1] - H_out
    if dH0 <= 0.0:
        raise ValueError(f"con estos datos la carga antes de la válvula ({H[-1]:.4g} m) no supera la de descarga "
                         f"({H_out:.4g} m): el flujo inicial Q0 no es posible; revisar presiones, Q0 y la fricción")
    Cv0 = Q0 * Q0 / dH0

    def step(H, Q, tau_t):
        Hn, Qn = np.empty_like(H), np.empty_like(Q)
        CP = H[:-1] + B * Q[:-1] - R * Q[:-1] * np.abs(Q[:-1])
        CM = H[1:] - B * Q[1:] + R * Q[1:] * np.abs(Q[1:])
        Hn[1:-1] = 0.5 * (CP[:-1] + CM[1:])
        Qn[1:-1] = (CP[:-1] - CM[1:]) / (2.0 * B)
        Hn[0] = H_R
        Qn[0] = (H_R - CM[0]) / B
        cp, cv = CP[-1], Cv0 * tau_t ** 2
        if cv <= 0.0:
            qv = 0.0
        else:
            d = cp - H_out
            if d >= 0.0:
                qv = 0.5 * (-B * cv + math.sqrt((B * cv) ** 2 + 4.0 * cv * d))
            else:
                qv = 0.5 * (B * cv - math.sqrt((B * cv) ** 2 - 4.0 * cv * d))
        Qn[-1] = qv
        Hn[-1] = cp - B * qv
        return Hn, Qn

    H1, Q1 = step(H, Q, 1.0)                      # valve fully open: the steady state must be preserved
    steady_res = max(float(np.max(np.abs(H1 - H))) / max(abs(H_R), 1.0), float(np.max(np.abs(Q1 - Q))) / Q0)
    pg = rho * G_ACC
    P = pg * (H - z)
    Pmax, Pmin = P.copy(), P.copy()
    tmax, tmin = np.zeros(N + 1), np.zeros(N + 1)
    n_steps = int(math.ceil(t_end / dt))
    hist_t, hist_P, hist_Q = [0.0], [float(P[-1])], [Q0]
    t = 0.0
    for k in range(1, n_steps + 1):
        t = k * dt
        H, Q = step(H, Q, tau(t))
        P = pg * (H - z)
        up, dn = P > Pmax, P < Pmin
        Pmax[up], tmax[up] = P[up], t
        Pmin[dn], tmin[dn] = P[dn], t
        if keep_history:
            hist_t.append(t)
            hist_P.append(float(P[-1]))
            hist_Q.append(float(Q[-1]))
    i_max, i_min = int(np.argmax(Pmax)), int(np.argmin(Pmin))
    return {"dt_s": dt, "steps": n_steps, "P_max_Pa": float(Pmax[i_max]), "x_P_max_m": i_max * dx,
            "t_P_max_s": float(tmax[i_max]), "P_min_Pa": float(Pmin[i_min]), "x_P_min_m": i_min * dx,
            "t_P_min_s": float(tmin[i_min]), "envelope_x_m": (np.arange(N + 1) * dx).tolist(),
            "envelope_P_max_Pa": Pmax.tolist(), "envelope_P_min_Pa": Pmin.tolist(),
            "P_valve_initial_Pa": float(pg * (H_R - R * Q0 * Q0 * N - dz)), "steady_state_residual": steady_res,
            "history": {"time_s": hist_t, "P_valve_Pa": hist_P, "Q_valve_m3_s": hist_Q}, "valve_dH0_m": dH0}


def _thin(hist: dict, keep: int = 400) -> dict:
    n = len(hist["time_s"])
    if n <= keep:
        return hist
    idx = np.unique(np.linspace(0, n - 1, keep).round().astype(int))
    return {k: [v[i] for i in idx] for k, v in hist.items()}


def water_hammer(length_m: float, inner_diameter_m: float, density_kg_m3: float, initial_flow_m3_s: float,
                 upstream_pressure_Pa: float, downstream_pressure_Pa: float, closure_time_s: float | None = None,
                 closure_exponent: float | None = None, closure_curve: dict | None = None,
                 wave_speed_m_s: float | None = None, fluid_bulk_modulus_Pa: float | None = None,
                 pipe_elastic_modulus_Pa: float | None = None, wall_thickness_m: float | None = None,
                 restraint_factor: float | None = None, pipe_anchoring: str | None = None,
                 poisson_ratio: float | None = None, darcy_friction_factor: float | None = None,
                 viscosity_Pa_s: float | None = None, roughness_m: float | None = None,
                 vapor_pressure_Pa: float | None = None, elevation_change_m: float = 0.0, n_reaches: int = 40,
                 simulation_time_s: float | None = None) -> dict:
    req = {"length_m": length_m, "inner_diameter_m": inner_diameter_m, "density_kg_m3": density_kg_m3,
           "initial_flow_m3_s": initial_flow_m3_s, "upstream_pressure_Pa": upstream_pressure_Pa,
           "downstream_pressure_Pa": downstream_pressure_Pa}
    for k, v in req.items():
        if v is None or not v > 0:
            raise ValueError(f"falta {k} (> 0): geometría, densidad, caudal inicial y presiones absolutas del depósito "
                             "aguas arriba y de la descarga de la válvula definen el estado inicial")
    tau, tc = _tau_fn(closure_time_s, closure_exponent, closure_curve)
    L, D, rho, Q0 = float(length_m), float(inner_diameter_m), float(density_kg_m3), float(initial_flow_m3_s)
    A = math.pi * D * D / 4.0
    V0 = Q0 / A
    warnings: list[str] = []
    assumptions = ["fricción cuasi-estacionaria de Darcy (sin fricción transitoria)", "depósito aguas arriba a carga "
                   "constante, sin pérdida de entrada ni carga de velocidad", "líquido de una sola fase; el modelo no "
                   "representa separación de columna ni cavitación distribuida", "elevación lineal a lo largo del tubo"
                   + ("" if elevation_change_m else " (tubería horizontal: elevation_change_m = 0)")]
    if wave_speed_m_s is not None:
        a = float(wave_speed_m_s)
        ws = {"wave_speed_m_s": a, "method": "dada por el usuario"}
    else:
        if fluid_bulk_modulus_Pa is None:
            raise ValueError("falta wave_speed_m_s o fluid_bulk_modulus_Pa (con los datos del tubo) para la celeridad")
        ws = wave_speed(rho, fluid_bulk_modulus_Pa, D, wall_thickness_m, pipe_elastic_modulus_Pa, restraint_factor,
                        pipe_anchoring, poisson_ratio)
        a = ws["wave_speed_m_s"]
        if ws.get("warning"):
            warnings.append(ws["warning"])
        if pipe_elastic_modulus_Pa is None:
            assumptions.append("tubo rígido (sin E ni espesor): a = √(K/ρ) es una cota superior")
    if darcy_friction_factor is not None:
        f, f_src = float(darcy_friction_factor), "usuario"
    else:
        if viscosity_Pa_s is None or roughness_m is None:
            raise ValueError("falta la fricción: darcy_friction_factor (0 = sin fricción) o viscosity_Pa_s y roughness_m")
        from fluids import friction_factor
        Re = rho * V0 * D / viscosity_Pa_s
        f = 64.0 / Re if Re < 2300.0 else float(friction_factor(Re=Re, eD=roughness_m / D))
        f_src = f"Colebrook (Clamond, fluids) a Re = {Re:.4g}"
    H_R = upstream_pressure_Pa / (rho * G_ACC)
    two_L_a = 2.0 * L / a
    t_end = float(simulation_time_s) if simulation_time_s is not None else tc + 10.0 * two_L_a
    N = int(n_reaches)
    if N < 4:
        raise ValueError("n_reaches debe ser ≥ 4")
    run = _moc(N, L, D, a, f, rho, Q0, H_R, downstream_pressure_Pa, float(elevation_change_m), tau, t_end, True)
    fine = _moc(2 * N, L, D, a, f, rho, Q0, H_R, downstream_pressure_Pa, float(elevation_change_m), tau, t_end, False)
    dPj = rho * a * V0
    grid_dev = max(abs(run["P_max_Pa"] - fine["P_max_Pa"]), abs(run["P_min_Pa"] - fine["P_min_Pa"])) / dPj
    reasons = []
    if run["steady_state_residual"] > STEADY_TOL:
        reasons.append("el paso MOC con la válvula abierta no conserva el estado estacionario inicial")
    if grid_dev > GRID_TOL:
        reasons.append(f"las presiones extremas cambian {grid_dev:.2%} del golpe de Joukowsky al duplicar la malla: "
                       "aumentar n_reaches")
    fast = tc <= two_L_a
    cav = None
    if vapor_pressure_Pa is not None:
        cav = run["P_min_Pa"] < vapor_pressure_Pa
        if cav:
            warnings.append(f"la presión mínima {run['P_min_Pa']:.4g} Pa cae bajo la presión de vapor "
                            f"{vapor_pressure_Pa:.4g} Pa (x = {run['x_P_min_m']:.4g} m, t = {run['t_P_min_s']:.4g} s): "
                            "cavitación / separación de columna; desde ahí el MOC de una fase no es válido y el golpe "
                            "real al colapsar la cavidad puede ser mayor")
    else:
        warnings.append("sin vapor_pressure_Pa: no se evaluó la cavitación")
    if run["P_min_Pa"] < 0.0:
        warnings.append("presión absoluta negativa en el cálculo: físicamente habría cavitación; el resultado posterior no "
                        "es físico")
    if not fast:
        warnings.append("cierre lento (t_c > 2L/a): la sobrepresión depende de la ley de cierre; la estimación de columna "
                        "rígida ρ·L·V0/t_c es solo orientativa")
    out = {"wave_speed": ws, "wave_speed_m_s": a, "velocity_m_s": V0, "darcy_friction_factor": f, "friction_source": f_src,
           "joukowsky_dP_Pa": dPj, "joukowsky_dH_m": a * V0 / G_ACC, "two_L_over_a_s": two_L_a,
           "period_4L_over_a_s": 2.0 * two_L_a, "closure_time_s": tc,
           "closure": "rápido (t_c ≤ 2L/a): golpe pleno de Joukowsky" if fast else "lento (t_c > 2L/a)",
           "rigid_column_dP_Pa": rho * L * V0 / tc if tc > 0 else None,
           "P_initial_valve_Pa": run["P_valve_initial_Pa"], "P_max_Pa": run["P_max_Pa"], "x_P_max_m": run["x_P_max_m"],
           "t_P_max_s": run["t_P_max_s"], "P_min_Pa": run["P_min_Pa"], "x_P_min_m": run["x_P_min_m"],
           "t_P_min_s": run["t_P_min_s"], "surge_above_initial_Pa": run["P_max_Pa"] - run["P_valve_initial_Pa"],
           "cavitation": cav, "envelope": {"x_m": run["envelope_x_m"], "P_max_Pa": run["envelope_P_max_Pa"],
                                           "P_min_Pa": run["envelope_P_min_Pa"]},
           "valve_history": _thin(run["history"]), "simulation_time_s": t_end, "n_reaches": N, "dt_s": run["dt_s"],
           "evidence": {"steady_state_residual": run["steady_state_residual"], "P_max_2N_Pa": fine["P_max_Pa"],
                        "P_min_2N_Pa": fine["P_min_Pa"], "grid_change_rel_joukowsky": grid_dev, "verified": not reasons,
                        "reasons": reasons},
           "warnings": warnings, "assumptions": assumptions,
           "not_included": ["separación de columna y cavitación transitoria (DVCM/DGCM)", "fricción transitoria (Brunone, "
                            "Zielke)", "interacción fluido-estructura y amortiguamiento viscoelástico del tubo",
                            "bombas, chimeneas de equilibrio, válvulas de alivio o tanques hidroneumáticos",
                            "redes con más de un tubo"],
           "method": ("Joukowsky ΔP = ρ·a·V0; celeridad de Korteweg (pared delgada); transitorio por el método de "
                      f"características (Wylie-Streeter) con {N} tramos, Courant 1, fricción cuasi-estacionaria (numpy)")}
    return out
