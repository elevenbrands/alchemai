"""Centrifugal pump on a system: operating point, specific speed and pump type, affinity laws, NPSH and motor (bloque MG).

What it answers: where the user's pump curve crosses the user's system curve (flow, head, efficiency, shaft and motor
power), the specific speed at the best-efficiency point and the impeller type it implies, the curve and operating point
after a speed change or an impeller trim (affinity laws), the speed needed for a target flow, and NPSH available against
NPSH required.

Data policy: the pump curve (Q-H, and optionally efficiency and NPSHr) and the system (static head plus a resistance, or a
Q-H table, e.g. from the pipe network tool) are the user's. Curves are interpolated (PCHIP, monotone, never extrapolated) or
fitted by least squares (`quadratic` H = a + b·Q + c·Q², `parabola` H = a + c·Q²). Efficiency is NEVER invented: from the
user's curve or value, or, only when the caller opts in, from the published Corripio, Chrien & Evans (1982) correlation
for centrifugal pumps (as given in Seider, Seader & Lewin, implemented in `fluids.pump`; valid 50-5000 US gpm), said so.
Motor efficiency: the user's, or the Corripio motor correlation under the same opt-in; motor size: next NEMA rating
(`fluids.pump.motor_round_size`).

Specific speed: n_s = n·√Q/H^0.75 at the best-efficiency point, per stage, in US units (rpm, US gpm, ft), metric (rpm, m³/s,
m) and dimensionless ω_s = ω·√Q/(g·H)^0.75. Type ranges (US units, Wikipedia «Specific speed», read 2026-11-15): radial
500-4000, mixed flow 2000-8000, axial 7000-20000, below 500 positive displacement; overlapping ranges are reported as such.

Affinity laws (geometrically similar, same efficiency): Q ∝ n·D, H ∝ n²·D², P ∝ n³·D³; NPSHr is scaled with n² (an
approximation). For impeller trimming the laws are approximate. Evidence: the operating point closes H_pump = H_system, the
specific speed is invariant under the speed scaling, and the hydraulic power equals ρ·g·Q·H.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

G_ACC = 9.80665
GPM = 6.30901964e-5          # m³/s per US gallon per minute
FT = 0.3048
NS_RANGES_US = {"radial (centrífuga)": (500.0, 4000.0), "flujo mixto": (2000.0, 8000.0), "axial": (7000.0, 20000.0)}
CORRIPIO_RANGE_GPM = (50.0, 5000.0)
OP_TOL = 1e-9


def _curve(q: np.ndarray, y: np.ndarray, fit: str):
    if fit == "pchip":
        return PchipInterpolator(q, y, extrapolate=False), None
    if fit == "quadratic":
        X = np.vstack([np.ones_like(q), q, q * q]).T
    elif fit == "parabola":
        X = np.vstack([np.ones_like(q), q * q]).T
    else:
        raise ValueError("curve_fit debe ser 'pchip', 'quadratic' o 'parabola'")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if fit == "quadratic":
        fn = lambda Q: coef[0] + coef[1] * Q + coef[2] * Q * Q
    else:
        fn = lambda Q: coef[0] + coef[1] * Q * Q
    resid = float(np.max(np.abs(fn(q) - y)))
    return fn, {"coefficients": coef.tolist(), "max_abs_residual": resid,
                "form": "H = a + b·Q + c·Q²" if fit == "quadratic" else "H = a + c·Q²"}


def _arr(d: dict, key: str, what: str) -> np.ndarray:
    v = d.get(key)
    if v is None:
        raise ValueError(f"falta {what}['{key}']")
    return np.asarray(v, float)


def _system_fn(system: dict):
    if "static_head_m" not in system:
        raise ValueError("system necesita 'static_head_m' (carga estática: elevación + diferencia de presión, m) y "
                         "'resistance_s2_m5' (H = Hs + k·Q²), o 'flow_m3_s' y 'head_m' (tabla de la curva del sistema)")
    Hs = float(system["static_head_m"])
    if "resistance_s2_m5" in system:
        k = float(system["resistance_s2_m5"])
        return (lambda Q: Hs + k * Q * Q), None, f"H = {Hs:.6g} + {k:.6g}·Q²"
    if "flow_m3_s" in system and "head_m" in system:
        q, h = np.asarray(system["flow_m3_s"], float), np.asarray(system["head_m"], float)
        if len(q) < 2 or np.any(np.diff(q) <= 0):
            raise ValueError("system: flow_m3_s creciente con ≥ 2 puntos")
        f = PchipInterpolator(q, h, extrapolate=False)
        return (lambda Q: float(f(Q))), (float(q[0]), float(q[-1])), "tabla del usuario (PCHIP)"
    raise ValueError("system necesita 'resistance_s2_m5' o la tabla 'flow_m3_s' + 'head_m'")


def _ns(Q: float, H: float, n_rpm: float) -> dict:
    us = n_rpm * math.sqrt(Q / GPM) / (H / FT) ** 0.75
    types = [k for k, (lo, hi) in NS_RANGES_US.items() if lo <= us <= hi]
    if us < 500.0:
        types = ["desplazamiento positivo (n_s < 500 en unidades EE. UU.)"]
    elif us > 20000.0:
        types = ["fuera de rango (> 20000): axial de muy baja carga"]
    return {"Ns_US_rpm_gpm_ft": us, "Ns_metric_rpm_m3s_m": n_rpm * math.sqrt(Q) / H ** 0.75,
            "omega_s_dimensionless": n_rpm * 2.0 * math.pi / 60.0 * math.sqrt(Q) / (G_ACC * H) ** 0.75,
            "pump_type": types}


def _operating_point(Hp, Hsys, q_lo: float, q_hi: float) -> float | None:
    qs = np.linspace(q_lo, q_hi, 401)
    d = np.array([Hp(q) - Hsys(q) for q in qs], float)
    ok = np.isfinite(d)
    qs, d = qs[ok], d[ok]
    for i in range(len(qs) - 1, 0, -1):                     # highest-flow crossing
        if d[i] == 0.0:
            return float(qs[i])
        if d[i - 1] * d[i] < 0.0:
            return brentq(lambda q: Hp(q) - Hsys(q), qs[i - 1], qs[i], xtol=1e-15, rtol=4e-15)
    return None


def pump_selection(pump_curve: dict, speed_rpm: float, density_kg_m3: float, system: dict, curve_fit: str = "pchip",
                   stages: int = 1, efficiency: float | None = None, motor_efficiency: float | None = None,
                   allow_efficiency_estimate: bool = False, suction: dict | None = None,
                   vapor_pressure_Pa: float | None = None, new_speed_rpm: float | None = None,
                   impeller_diameter_m: float | None = None, new_impeller_diameter_m: float | None = None,
                   target_flow_m3_s: float | None = None) -> dict:
    if speed_rpm is None or not speed_rpm > 0 or density_kg_m3 is None or not density_kg_m3 > 0:
        raise ValueError("faltan speed_rpm (velocidad de la curva) y density_kg_m3 del líquido")
    q = _arr(pump_curve, "flow_m3_s", "pump_curve")
    h = _arr(pump_curve, "head_m", "pump_curve")
    if len(q) < 3 or len(q) != len(h) or np.any(np.diff(q) <= 0) or np.any(q < 0):
        raise ValueError("pump_curve: ≥ 3 puntos con flow_m3_s creciente (≥ 0) y head_m de igual longitud")
    eta_pts = pump_curve.get("efficiency")
    npshr_pts = pump_curve.get("NPSHr_m")
    warnings: list[str] = []
    assumptions = [f"curva Q-H por {curve_fit}" + (" (interpolación monótona, sin extrapolar)" if curve_fit == "pchip"
                                                   else " (mínimos cuadrados)"),
                   "líquido incompresible de densidad constante; viscosidad baja (sin corrección de curva por viscosidad, "
                   "ANSI/HI 9.6.7)"]
    if stages != 1:
        assumptions.append(f"{stages} etapas iguales: n_s por etapa con H/{stages}")
    Hp_base, fit_info = _curve(q, h, curve_fit)
    eta_fn = None
    if eta_pts is not None:
        e = np.asarray(eta_pts, float)
        if len(e) != len(q) or np.any(e <= 0) or np.any(e > 1):
            raise ValueError("pump_curve['efficiency'] como fracción (0-1] en los mismos caudales")
        eta_fn = PchipInterpolator(q, e, extrapolate=False)
    npsh_fn = PchipInterpolator(q, np.asarray(npshr_pts, float), extrapolate=False) if npshr_pts is not None else None
    Hsys, sys_range, sys_desc = _system_fn(system)

    def scaled(r_n: float, r_d: float):
        """Affinity-scaled curve: Q' = Q·r_n·r_d, H' = H·(r_n·r_d)²."""
        s = r_n * r_d
        return (lambda Q: float(Hp_base(Q / s)) * s * s), (float(q[0]) * s, float(q[-1]) * s), s

    def solve(r_n: float = 1.0, r_d: float = 1.0) -> dict | None:
        Hp, (lo, hi), s = scaled(r_n, r_d)
        if sys_range:
            lo, hi = max(lo, sys_range[0]), min(hi, sys_range[1])
        if lo >= hi:
            return None
        Qop = _operating_point(Hp, Hsys, lo, hi)
        if Qop is None:
            return None
        Hop = Hp(Qop)
        res = abs(Hop - Hsys(Qop)) / max(abs(Hop), 1e-12)
        eta, src = None, None
        if eta_fn is not None:
            eta, src = float(eta_fn(Qop / s)), "curva del usuario (misma eficiencia en el punto homólogo, afinidad)"
        elif efficiency is not None:
            eta, src = float(efficiency), "valor del usuario"
        elif allow_efficiency_estimate:
            from fluids.pump import Corripio_pump_efficiency
            eta, src = float(Corripio_pump_efficiency(Qop)), "ESTIMADA: Corripio, Chrien y Evans (1982), vía fluids"
            gpm = Qop / GPM
            if not CORRIPIO_RANGE_GPM[0] <= gpm <= CORRIPIO_RANGE_GPM[1]:
                warnings.append(f"Q = {gpm:.0f} gpm fuera del rango de Corripio (50-5000 gpm): eficiencia estimada "
                                "no fiable")
        P_hyd = density_kg_m3 * G_ACC * Qop * Hop
        out = {"flow_m3_s": Qop, "head_m": Hop, "residual_rel": res, "hydraulic_power_W": P_hyd, "efficiency": eta,
               "efficiency_source": src, "speed_rpm": speed_rpm * r_n}
        if eta:
            out["shaft_power_W"] = P_hyd / eta
        if npsh_fn is not None:
            out["NPSHr_m"] = float(npsh_fn(Qop / s)) * r_n * r_n
        return out

    op = solve()
    if op is None:
        raise ValueError("la curva de la bomba no cruza la curva del sistema dentro del rango de caudal de la curva "
                         f"({q[0]:.4g}-{q[-1]:.4g} m³/s): la bomba no da la carga del sistema o la curva es corta; no se "
                         "extrapola")
    if op["efficiency"] is None:
        raise ValueError("falta la eficiencia: pump_curve['efficiency'] o efficiency del fabricante, o "
                         "allow_efficiency_estimate=True para usar la correlación publicada de Corripio (estimación)")
    if "estimada" in str(op["efficiency_source"]).lower():
        assumptions.append("eficiencia de bomba ESTIMADA por Corripio, Chrien y Evans (1982): η = −0.316 + 0.24015·ln(Q) "
                           "− 0.01199·ln(Q)² con Q en gpm; no es dato del equipo")
    # ---- best-efficiency point and specific speed
    if eta_fn is not None:
        qq = np.linspace(q[0], q[-1], 2001)
        ee = np.asarray(eta_fn(qq), float)
        i = int(np.nanargmax(ee))
        Q_bep, H_bep, eta_bep = float(qq[i]), float(Hp_base(qq[i])), float(ee[i])
        bep_src = "máximo de la curva de eficiencia del usuario"
    else:
        Q_bep, H_bep, eta_bep = op["flow_m3_s"], op["head_m"], op["efficiency"]
        bep_src = "punto de operación (sin curva de eficiencia no hay BEP: n_s es solo indicativa)"
        warnings.append("sin curva de eficiencia: n_s se evaluó en el punto de operación, no en el BEP")
    ns = _ns(Q_bep, H_bep / stages, speed_rpm)
    if len(ns["pump_type"]) > 1:
        warnings.append(f"n_s = {ns['Ns_US_rpm_gpm_ft']:.0f} cae donde se traslapan los rangos {ns['pump_type']}")
    ratio_bep = op["flow_m3_s"] / Q_bep
    if eta_fn is not None and not 0.7 <= ratio_bep <= 1.2:
        warnings.append(f"opera a {ratio_bep:.0%} del caudal del BEP: lejos del BEP crecen vibración, recirculación y "
                        "carga radial; revisar la región de operación preferida del fabricante")
    # ---- motor
    motor = None
    if op.get("shaft_power_W"):
        from fluids.pump import Corripio_motor_efficiency, motor_round_size
        Psh = op["shaft_power_W"]
        if motor_efficiency is not None:
            me, me_src = float(motor_efficiency), "valor del usuario"
        elif allow_efficiency_estimate:
            me, me_src = float(Corripio_motor_efficiency(Psh)), "ESTIMADA: Corripio, Chrien y Evans (1982), vía fluids"
        else:
            me, me_src = None, None
        Pmax_curve = None
        if eta_fn is not None:
            qq = np.linspace(q[0] if q[0] > 0 else q[1] * 1e-3, q[-1], 400)
            pw = density_kg_m3 * G_ACC * qq * np.asarray([Hp_base(x) for x in qq]) / np.asarray(eta_fn(qq))
            Pmax_curve = float(np.nanmax(pw))
        try:
            nema = float(motor_round_size(Psh))
            nema_max = float(motor_round_size(Pmax_curve)) if Pmax_curve else None
        except ValueError:
            nema, nema_max = None, None
            warnings.append("la potencia supera el mayor motor NEMA tabulado")
        motor = {"shaft_power_W": Psh, "motor_efficiency": me, "motor_efficiency_source": me_src,
                 "electric_power_W": Psh / me if me else None, "NEMA_motor_for_operating_point_W": nema,
                 "max_shaft_power_on_curve_W": Pmax_curve, "NEMA_motor_non_overloading_W": nema_max}
    # ---- NPSH
    npsh = None
    if suction is not None:
        need = ("vessel_pressure_Pa", "liquid_level_above_pump_m")
        miss = [k for k in need if k not in suction]
        if miss or vapor_pressure_Pa is None or ("losses_m" not in suction and "loss_coefficient_s2_m5" not in suction):
            raise ValueError("NPSH disponible: suction necesita vessel_pressure_Pa (absoluta), liquid_level_above_pump_m "
                             "(negativo si está debajo) y losses_m (pérdidas de succión al caudal de operación) o "
                             "loss_coefficient_s2_m5; además vapor_pressure_Pa del líquido a la T de bombeo")
        hl = (float(suction["losses_m"]) if "losses_m" in suction
              else float(suction["loss_coefficient_s2_m5"]) * op["flow_m3_s"] ** 2)
        npsha = ((float(suction["vessel_pressure_Pa"]) - vapor_pressure_Pa) / (density_kg_m3 * G_ACC)
                 + float(suction["liquid_level_above_pump_m"]) - hl)
        npsh = {"NPSHa_m": npsha, "suction_losses_m": hl, "NPSHr_m": op.get("NPSHr_m")}
        if op.get("NPSHr_m") is not None:
            npsh["margin_m"] = npsha - op["NPSHr_m"]
            npsh["ratio"] = npsha / op["NPSHr_m"] if op["NPSHr_m"] > 0 else None
            if npsha <= op["NPSHr_m"]:
                warnings.append(f"NPSHa {npsha:.2f} m ≤ NPSHr {op['NPSHr_m']:.2f} m: la bomba cavita (NPSHr es el de 3 % "
                                "de caída de carga); subir el nivel, la presión del recipiente o reducir pérdidas")
            else:
                warnings.append("compara el cociente NPSHa/NPSHr con el margen que exige tu norma o el fabricante "
                                "(NPSHr ya implica 3 % de caída de carga)")
        else:
            warnings.append("sin NPSHr en la curva: solo se calculó el NPSH disponible")
    # ---- affinity laws
    affinity = {}
    ns_invariance = None
    if new_speed_rpm is not None:
        rn = float(new_speed_rpm) / speed_rpm
        new = solve(r_n=rn)
        Qb, Hb = Q_bep * rn, H_bep * rn * rn
        ns_new = _ns(Qb, Hb / stages, new_speed_rpm)["Ns_US_rpm_gpm_ft"]
        ns_invariance = abs(ns_new - ns["Ns_US_rpm_gpm_ft"]) / ns["Ns_US_rpm_gpm_ft"]
        affinity["new_speed"] = {"speed_rpm": new_speed_rpm, "ratio": rn, "operating_point": new,
                                 "law": "Q ∝ n, H ∝ n², P ∝ n³, NPSHr ∝ n² (aprox.)"}
        if new is None:
            warnings.append("con la nueva velocidad la curva no cruza el sistema en su rango")
    if new_impeller_diameter_m is not None:
        if impeller_diameter_m is None:
            raise ValueError("para el recorte de impulsor falta impeller_diameter_m (diámetro de la curva dada)")
        rd = float(new_impeller_diameter_m) / float(impeller_diameter_m)
        affinity["impeller_trim"] = {"impeller_diameter_m": new_impeller_diameter_m, "ratio": rd,
                                     "operating_point": solve(r_d=rd), "law": "Q ∝ D, H ∝ D², P ∝ D³ (aproximada)"}
        warnings.append("las leyes de afinidad para recorte de impulsor son aproximadas: confirmar con la curva del "
                        "fabricante para el diámetro recortado")
    if target_flow_m3_s is not None:
        Qt = float(target_flow_m3_s)
        Ht = Hsys(Qt)

        def gap(r: float) -> float:
            Hp, (lo, hi), _ = scaled(r, 1.0)
            v = Hp(Qt)
            return (v if math.isfinite(v) else math.nan) - Ht
        r_lo, r_hi = max(Qt / q[-1], 1e-3) * (1 + 1e-9), 3.0
        rs = np.linspace(r_lo, r_hi, 600)
        gs = np.array([gap(r) for r in rs])
        sol = None
        for i in range(len(rs) - 1):
            if np.isfinite(gs[i]) and np.isfinite(gs[i + 1]) and gs[i] * gs[i + 1] <= 0:
                sol = brentq(gap, rs[i], rs[i + 1], xtol=1e-14)
                break
        affinity["speed_for_target_flow"] = ({"target_flow_m3_s": Qt, "system_head_m": Ht, "speed_rpm": speed_rpm * sol,
                                              "ratio": sol} if sol else
                                             {"target_flow_m3_s": Qt, "error": "ninguna velocidad entre "
                                              f"{r_lo:.3f} y 3 veces la nominal da ese caudal en este sistema"})
    P_check = abs(op["hydraulic_power_W"] - density_kg_m3 * G_ACC * op["flow_m3_s"] * Hsys(op["flow_m3_s"])) / \
        op["hydraulic_power_W"]
    reasons = []
    if op["residual_rel"] > OP_TOL:
        reasons.append("el punto de operación no cierra H_bomba = H_sistema")
    if ns_invariance is not None and ns_invariance > 1e-9:
        reasons.append("n_s no se conserva con la ley de afinidad de velocidad")
    if P_check > 1e-8:
        reasons.append("la potencia hidráulica no coincide con ρ·g·Q·H del sistema")
    return {"operating_point": op, "system_curve": sys_desc, "pump_curve_fit": fit_info or "PCHIP",
            "best_efficiency_point": {"flow_m3_s": Q_bep, "head_m": H_bep, "efficiency": eta_bep, "source": bep_src},
            "flow_fraction_of_bep": ratio_bep, "specific_speed": ns, "motor": motor, "npsh": npsh, "affinity": affinity,
            "evidence": {"operating_point_residual_rel": op["residual_rel"], "specific_speed_invariance_rel": ns_invariance,
                         "hydraulic_power_check_rel": P_check, "verified": not reasons, "reasons": reasons},
            "warnings": warnings, "assumptions": assumptions,
            "not_included": ["corrección de curvas por viscosidad (ANSI/HI 9.6.7)", "bombas en serie o en paralelo",
                             "transitorios de arranque y paro (ver water_hammer)", "selección de catálogo de fabricante",
                             "estabilidad de curvas con máximo (cruces múltiples: se toma el de mayor caudal)"],
            "method": ("intersección de curvas con brentq (scipy) sobre PCHIP o ajuste por mínimos cuadrados (numpy); "
                       "n_s en el BEP; leyes de afinidad; eficiencias solo del usuario o Corripio (fluids) si se permite")}
