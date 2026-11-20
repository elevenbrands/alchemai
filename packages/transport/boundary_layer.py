"""Laminar boundary layer on a flat plate at zero incidence (Blasius), solved numerically, never quoted from memory.

What it answers: for a uniform stream U of a Newtonian fluid (rho, mu) along a thin flat plate with zero pressure
gradient, at any distance x from the leading edge: Re_x, the thicknesses (delta_99, displacement delta*, momentum theta),
the local wall shear stress and skin-friction coefficient; the velocity components u, v at any point (x, y) from the
similarity profile; and the friction drag of a plate of length L (and width W), one side and both sides.

Method: the similarity variable eta = y sqrt(U/(nu x)) and stream function psi = sqrt(nu U x) f(eta) turn the
boundary-layer equations into the Blasius problem
    f''' + f f''/2 = 0,   f(0) = f'(0) = 0,   f'(inf) = 1
integrated with scipy.integrate.solve_ivp (DOP853). The missing wall curvature follows from the scaling invariance of the
equation: if F solves it with F''(0) = 1 and F'(inf) = c, then f''(0) = c^(-3/2). Results (all computed here):
    u/U = f'(eta);  v/U = (eta f' - f) / (2 sqrt(Re_x));  tau_w = f''(0) rho U^2 / sqrt(Re_x)
    delta_99 = eta_99 x / sqrt(Re_x)        (eta_99 ~ 4.91, where f' = 0.99)
    delta*   = lim(eta - f) x / sqrt(Re_x)  (~ 1.7208)
    theta    = 2 f''(0) x / sqrt(Re_x)      (~ 0.664)
    c_f = 2 f''(0) / sqrt(Re_x) (~ 0.664/sqrt(Re_x));  C_D(L) = 4 f''(0) / sqrt(Re_L) (~ 1.328/sqrt(Re_L))
Evidence returned with every result: f'(eta_max) = 1, f''(0) against the literature value 0.33206 (Blasius 1908; Howarth
1938; White, Viscous Fluid Flow; Schlichting, Boundary-Layer Theory), the momentum-integral identity
theta = integral f'(1 - f') d eta = 2 f''(0) by an independent quadrature state, the solution re-integrated with a looser
tolerance, and the plate drag as the numerical integral of tau_w(x) against the closed form.

Limits: laminar only. Transition on a smooth plate with a quiet stream starts near Re_x ~ 5e5 (it can be delayed to ~3e6
or triggered earlier by roughness or free-stream turbulence): above it the numbers do not apply. Boundary-layer theory
also fails very close to the leading edge (Re_x below ~1e3 the error is noticeable). Not included: pressure gradients
(Falkner-Skan), turbulent or mixed boundary layers, heat and mass transfer, compressibility, suction/blowing, non-Newtonian
fluids, finite-width edge effects."""
from __future__ import annotations

import math
from functools import lru_cache

from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

RE_TRANSITION = 5.0e5          # usual engineering value for the onset of transition on a smooth flat plate
RE_LOW = 1.0e3                 # below it the boundary-layer approximation itself is poor (leading-edge region)
ETA_MAX = 12.0
F2_LITERATURE = 0.33206        # f''(0), Howarth (1938) / White / Schlichting
CHECK_TOL = 1e-6


def _pos(name: str, value, what: str) -> float:
    if value is None:
        raise ValueError(f"falta {name}: {what}")
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} debe ser un número finito > 0 ({what})")
    return v


def _rhs(_eta, y):
    f, fp, fpp, _ = y
    return [fp, fpp, -0.5 * f * fpp, fp * (1.0 - fp)]


def _integrate(fpp0: float, rtol: float, dense: bool = False):
    return solve_ivp(_rhs, (0.0, ETA_MAX), [0.0, 0.0, fpp0, 0.0], method="DOP853", rtol=rtol, atol=1e-15,
                     dense_output=dense)


def _wall_curvature(rtol: float) -> float:
    c = _integrate(1.0, rtol).y[1, -1]
    return float(c) ** -1.5


@lru_cache(maxsize=1)
def blasius_solution() -> dict:
    """Numerical Blasius solution: constants, the dense interpolant and the checks of its own correctness."""
    fpp0 = _wall_curvature(1e-13)
    sol = _integrate(fpp0, 1e-13, dense=True)
    f_end, fp_end, _, theta_int = (float(v) for v in sol.y[:, -1])
    eta99 = brentq(lambda e: sol.sol(e)[1] - 0.99, 1.0, 8.0, xtol=1e-14, rtol=1e-14)
    checks = {"fprime_at_eta_max_minus_1": abs(fp_end - 1.0),
              "fpp0_vs_literature_0.33206_rel": abs(fpp0 - F2_LITERATURE) / F2_LITERATURE,
              "momentum_integral_vs_2fpp0_rel": abs(theta_int - 2.0 * fpp0) / (2.0 * fpp0),
              "fpp0_tolerance_refinement_rel": abs(_wall_curvature(1e-9) - fpp0) / fpp0}
    return {"fpp0": fpp0, "eta_99": eta99, "displacement_coefficient": ETA_MAX - f_end,
            "momentum_coefficient": 2.0 * fpp0, "sol": sol.sol, "checks": checks}


def similarity_profile(eta: float) -> tuple[float, float, float]:
    """(f, f', f'') of Blasius at eta (beyond eta_max the outer solution f' = 1 is used)."""
    b = blasius_solution()
    if eta >= ETA_MAX:
        return eta - b["displacement_coefficient"], 1.0, 0.0
    f, fp, fpp, _ = b["sol"](max(eta, 0.0))
    return float(f), float(fp), float(fpp)


def flat_plate(free_stream_velocity_m_s: float, density_kg_m3: float, viscosity_Pa_s: float,
               length_m: float | None = None, width_m: float | None = None, positions_m: list | None = None) -> dict:
    """Blasius laminar boundary layer on a flat plate (module docstring). positions_m: x values or [x, y] pairs."""
    U = _pos("free_stream_velocity_m_s", free_stream_velocity_m_s, "velocidad de la corriente libre paralela a la placa")
    rho = _pos("density_kg_m3", density_kg_m3, "densidad del fluido")
    mu = _pos("viscosity_Pa_s", viscosity_Pa_s, "viscosidad dinámica del fluido newtoniano")
    if length_m is None and not positions_m:
        raise ValueError("placa plana: da length_m (longitud de la placa, para el arrastre) y/o positions_m "
                         "(lista de x o de pares [x, y] en metros, medidos desde el borde de ataque y desde la pared)")
    nu = mu / rho
    b = blasius_solution()
    fpp0, eta99, c_disp, c_mom = b["fpp0"], b["eta_99"], b["displacement_coefficient"], b["momentum_coefficient"]
    checks = dict(b["checks"])
    warnings: list[str] = []
    reasons: list[str] = []
    x_tr = RE_TRANSITION * nu / U
    q = 0.5 * rho * U * U

    def station(x: float) -> dict:
        re_x = U * x / nu
        s = math.sqrt(re_x)
        return {"x_m": x, "reynolds_x": re_x, "laminar": re_x <= RE_TRANSITION,
                "delta_99_m": eta99 * x / s, "displacement_thickness_m": c_disp * x / s,
                "momentum_thickness_m": c_mom * x / s, "shape_factor": c_disp / c_mom,
                "local_skin_friction_coefficient": 2.0 * fpp0 / s, "wall_shear_stress_Pa": 2.0 * fpp0 / s * q}

    out: dict = {"geometry": "flat_plate", "free_stream_velocity_m_s": U, "kinematic_viscosity_m2_s": nu,
                 "blasius_constants": {"f_second_derivative_at_wall": fpp0, "eta_99": eta99,
                                       "displacement_coefficient": c_disp, "momentum_coefficient": c_mom,
                                       "local_cf_coefficient": 2.0 * fpp0, "average_cf_coefficient": 4.0 * fpp0},
                 "transition": {"criterion": "Re_x ≈ 5e5 (placa lisa, corriente poco turbulenta; valor de ingeniería)",
                                "critical_reynolds": RE_TRANSITION, "x_transition_m": x_tr}}
    points = []
    for i, p in enumerate(positions_m or []):
        if isinstance(p, (list, tuple)):
            if len(p) not in (1, 2):
                raise ValueError(f"positions_m[{i}] debe ser x o un par [x, y] (m)")
            x, y = _pos(f"positions_m[{i}][0]", p[0], "distancia x desde el borde de ataque"), (
                float(p[1]) if len(p) == 2 else None)
        else:
            x, y = _pos(f"positions_m[{i}]", p, "distancia x desde el borde de ataque"), None
        if y is not None and (not math.isfinite(y) or y < 0.0):
            raise ValueError(f"positions_m[{i}]: y debe ser ≥ 0 (distancia normal a la placa)")
        st = station(x)
        if y is not None:
            eta = y * math.sqrt(U / (nu * x))
            f, fp, fpp = similarity_profile(eta)
            st.update(y_m=y, eta=eta, u_over_U=fp, u_m_s=fp * U,
                      v_m_s=0.5 * (eta * fp - f) / math.sqrt(st["reynolds_x"]) * U,
                      shear_stress_Pa=mu * U * math.sqrt(U / (nu * x)) * fpp, y_over_delta_99=y / st["delta_99_m"])
        if not st["laminar"]:
            warnings.append(f"x = {x:.4g} m: Re_x = {st['reynolds_x']:.4g} > 5e5, la capa límite ya no es laminar ahí "
                            f"(la transición empieza cerca de x = {x_tr:.4g} m); los valores de Blasius no aplican")
            reasons.append(f"Re_x = {st['reynolds_x']:.4g} fuera del intervalo laminar en x = {x:.4g} m")
        elif st["reynolds_x"] < RE_LOW:
            warnings.append(f"x = {x:.4g} m: Re_x = {st['reynolds_x']:.4g} < 1e3, muy cerca del borde de ataque la teoría "
                            "de capa límite es solo aproximada")
        points.append(st)
    if points:
        out["points"] = points
    if length_m is not None:
        L = _pos("length_m", length_m, "longitud de la placa en la dirección del flujo")
        re_L = U * L / nu
        cd = 4.0 * fpp0 / math.sqrt(re_L)
        per_width = cd * q * L
        num, _ = quad(lambda x: 2.0 * fpp0 / math.sqrt(U * x / nu) * q, 0.0, L, epsabs=0.0, epsrel=1e-10, limit=200)
        checks["drag_integral_of_tau_w_vs_closed_form_rel"] = abs(num - per_width) / per_width
        plate = {"length_m": L, "reynolds_L": re_L, "average_skin_friction_coefficient": cd,
                 "trailing_edge": station(L), "drag_per_unit_width_one_side_N_m": per_width,
                 "drag_per_unit_width_both_sides_N_m": 2.0 * per_width}
        if width_m is not None:
            W = _pos("width_m", width_m, "ancho de la placa (perpendicular al flujo)")
            plate.update(width_m=W, drag_one_side_N=per_width * W, drag_both_sides_N=2.0 * per_width * W,
                         wetted_area_one_side_m2=L * W)
        if re_L > RE_TRANSITION:
            warnings.append(f"Re_L = {re_L:.4g} > 5e5: la capa límite deja de ser laminar hacia x = {x_tr:.4g} m; el "
                            "arrastre laminar de toda la placa subestima el real (capa límite mixta o turbulenta, no "
                            "incluida)")
            reasons.append(f"Re_L = {re_L:.4g} por encima del límite laminar 5e5")
        elif re_L < RE_LOW:
            warnings.append(f"Re_L = {re_L:.4g} < 1e3: placa tan corta que la teoría de capa límite subestima el "
                            "arrastre (efectos del borde de ataque)")
        out["plate"] = plate
    if checks["fprime_at_eta_max_minus_1"] > 1e-9 or checks["fpp0_vs_literature_0.33206_rel"] > 1e-4 or any(
            v > CHECK_TOL for k, v in checks.items() if k not in ("fprime_at_eta_max_minus_1",
                                                                   "fpp0_vs_literature_0.33206_rel")):
        reasons.append("la solución numérica de Blasius no cierra sus comprobaciones")
    out.update(
        method="ecuación de Blasius f''' + f·f''/2 = 0 integrada con scipy.integrate.solve_ivp (DOP853); f''(0) por la "
               "invariancia de escala; perfil u/U = f'(η), η = y·√(U/(ν·x))",
        warnings=warnings,
        assumptions=["placa plana delgada, lisa, alineada con la corriente (gradiente de presión nulo)",
                     "flujo laminar, estacionario, incompresible, fluido newtoniano de propiedades constantes",
                     "capa límite delgada (Re_x ≫ 1) que nace en el borde de ataque",
                     "arrastre solo de fricción; 'one_side' es una cara mojada, 'both_sides' las dos"],
        not_included=["capa límite turbulenta o mixta (Re_x > 5e5)", "gradiente de presión (Falkner-Skan), separación",
                      "transferencia de calor o masa en la capa límite", "efectos de borde de una placa de ancho finito",
                      "fluidos no newtonianos"],
        evidence={"verified": not reasons, "checks": checks, "tolerance": CHECK_TOL, "reasons": reasons,
                  "source": "Blasius (1908); valores de referencia f''(0) = 0.33206, η99 = 4.91, δ* = 1.7208, θ = 0.664 "
                            "(White, Viscous Fluid Flow; Schlichting, Boundary-Layer Theory), reproducidos aquí "
                            "numéricamente"})
    return out
