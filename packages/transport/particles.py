"""Particles in fluids (bloque MP): terminal settling, hindered settling, thickener area and fluidization.

What it answers:
  * terminal velocity of a single particle (sphere or, with sphericity, non-spherical) in a still Newtonian fluid, with
    its drag coefficient, Reynolds number and regime (Stokes / intermediate / Newton);
  * hindered settling of a uniform suspension by Richardson-Zaki (1954), u = u_t (1 - phi)^n_RZ, with the stepwise
    exponent of Richardson and Zaki including the d/D wall term:
        n = 4.65 + 19.5 d/D (Re_t < 0.2); (4.35 + 17.5 d/D) Re_t^-0.03 (0.2-1); (4.45 + 18 d/D) Re_t^-0.1 (1-200);
        4.45 Re_t^-0.1 (200-500); 2.39 (> 500)
    (as reproduced by Barree in Joseph et al., «Power law correlations for sediment transport...», U. Minnesota 2001);
  * thickener / clarifier area by Coe-Clevenger, from a solids balance on a settling layer:
        A = Q0 C0 max_C [(1/C - 1/Cu) / u(C)]   for C0 <= C < Cu
    with u(C) from the user's batch settling tests or, as a flagged estimate, from Richardson-Zaki;
  * fluidization: bed pressure drop (buoyant weight over area), Ergun pressure drop of the fixed bed (fluids.packed_bed),
    minimum fluidization by Ergun, Ar = 150 (1-e)/(phi^2 e^3) Re + 1.75/(phi e^3) Re^2, and by Wen-Yu,
    Re_mf = sqrt(33.7^2 + 0.0408 Ar) - 33.7, terminal velocity, operating window and (liquid) bed expansion.

Drag of a sphere: the correlations of the `fluids` library (Caleb Bell, MIT), default 'Clift' (the piecewise standard drag
curve of Clift, Grace and Weber 1978, valid to Re = 1e6); 'Barati', 'Haider_Levenspiel' and 'Morrison' are evaluated as
well and their spread is reported as model uncertainty. Non-spherical particles: `fluids.drag.Song_Xu` (sphericity and the
ratio of equivalent-sphere area to projected area, data for spheres, cubes and cylinders, 0.001 < Re < 100).
Terminal velocity: brentq on Re of Cd(Re) Re^2 = 4/3 Ar, Ar = g d^3 rho_f |rho_p - rho_f| / mu^2.

Wall effect on one sphere falling along the axis of a cylindrical tube (lam = d/D, wall factor f = u_wall/u_unbounded):
  * creeping flow, Haberman and Sayre (1958, David Taylor Model Basin Report 1143), drag ratio
        F/F_Stokes = (1 - 0.75857 lam^5) / (1 - 2.1050 lam + 2.0865 lam^3 - 1.7068 lam^5 + 0.72603 lam^6),  f = 1/that,
    valid lam < 0.8 (coefficients and range as in Wham et al. 1996 and in Eqs. 20-22 of «Studies on falling ball
    viscometry», arXiv:1202.1400, whose Table 9 compares it with the exact theory of Haberman/Bohlin/Paine-Scherr:
    equal to 3 decimals up to lam = 0.3, -0.4 % at 0.4, -1.7 % at 0.5, -5 % at 0.6);
  * creeping flow, Francis (1933, Physics 4, 403): f = ((1 - lam)/(1 - 0.475 lam))^4, Re < 0.1, lam < 0.8 (as given in
    Braz. J. Chem. Eng. 16(4), 1999, Eq. 7);
  * any Reynolds number, Di Felice (1996, Int. J. Multiphase Flow 22, 527): f = ((1 - lam)/(1 - 0.33 lam))^alpha with
    (3.3 - alpha)/(alpha - 0.85) = 0.1 Re_unbounded (arXiv:1202.1400, Eq. 18; the same source notes that it saturates
    too early in the intermediate regime): reported as an ESTIMATE;
  * Newton regime, Munroe (1888): f = 1 - lam^1.5, Re > 1000 (Braz. J. Chem. Eng. 16(4), 1999, Eq. 6).
The `fluids` library has no wall correction for single particles. Kehlenbeck-Di Felice (1999) and Wham et al. (1996) are
not implemented.

Stokes drag split (exact creeping-flow solution around a sphere, e.g. Bird-Stewart-Lightfoot §2.6): F = 3 pi mu d U, of
which pi mu d U (1/3) is form drag (pressure) and 2 pi mu d U (2/3) friction drag (viscous shear). Reported only when
Re_t < 0.1; above it the split needs the actual flow field.

Not included: wall effect off the tube axis or near the bottom, non-Newtonian liquids, Brownian particles, drops and bubbles that
deform or circulate internally, polydisperse suspensions, flocculation (for flocculated slurries use measured batch data),
bubbling-bed hydrodynamics of gas fluidization (bubble size, expansion), entrainment and TDH, Geldart classification."""
from __future__ import annotations

import math

import numpy as np
from fluids import drag as FD
from fluids.packed_bed import Ergun as fluids_ergun
from scipy.optimize import brentq, minimize_scalar

G = 9.80665
SPHERE_METHODS = ("Clift", "Barati", "Haider_Levenspiel", "Morrison")
STOKES_RE_LIMIT = 0.1         # label only: below it Clift's Cd is within 2 % of 24/Re
NEWTON_RE_LIMIT = 1000.0      # label only: above it Cd of a sphere is nearly flat until the drag crisis
DRAG_CRISIS_RE = 2.0e5
SONG_XU_RANGE = (1e-3, 100.0)
WEN_YU = (33.7, 0.0408)
ROOT_RTOL = 1e-13
CHECK_TOL = 1e-8


def _req(name: str, value, what: str) -> float:
    if value is None:
        raise ValueError(f"falta {name}: {what}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} no es un número finito")
    return v


def _pos(name: str, value, what: str) -> float:
    v = _req(name, value, what)
    if v <= 0.0:
        raise ValueError(f"{name} debe ser > 0 ({what})")
    return v


def _frac(name: str, value, what: str, lo: float = 0.0, hi: float = 1.0) -> float:
    v = _req(name, value, what)
    if not lo < v < hi:
        raise ValueError(f"{name} debe estar entre {lo} y {hi} ({what})")
    return v


def archimedes(d: float, rho_p: float, rho_f: float, mu: float) -> float:
    return G * d ** 3 * rho_f * abs(rho_p - rho_f) / (mu * mu)


def drag_coefficient(Re: float, method: str, sphericity: float = 1.0, area_ratio: float = 1.0) -> float:
    if method == "Song_Xu":
        return FD.Song_Xu(Re, sphericity=sphericity, S=area_ratio)
    return FD.drag_sphere(Re, Method=method)


def _solve_re(Ar: float, method: str, sphericity: float, area_ratio: float) -> tuple[float, float]:
    """Re_t with Cd(Re) Re^2 = 4/3 Ar, and the relative residual of that balance."""
    target = 4.0 * Ar / 3.0

    def g(lnre):
        re = math.exp(lnre)
        return math.log(drag_coefficient(re, method, sphericity, area_ratio) * re * re / target)
    lo, hi = math.log(1e-12), math.log(1e6 * 0.999)
    if g(hi) < 0:
        raise ArithmeticError("Re terminal > 1e6: fuera del rango de las correlaciones de arrastre")
    if g(lo) > 0:
        return 1e-12, 0.0
    lnre = brentq(g, lo, hi, xtol=1e-14, rtol=ROOT_RTOL, maxiter=500)
    return math.exp(lnre), abs(math.expm1(g(lnre)))


WALL_LAMBDA_MAX = 0.8        # Haberman-Sayre and Francis: d/D < 0.8
MUNROE_RE_MIN = 1000.0
# exact creeping-flow theory (Haberman 1956; Bohlin 1960; Paine & Scherr 1975) as tabulated in arXiv:1202.1400, Table 9
EXACT_WALL_FACTOR = {0.1: 0.792, 0.2: 0.595, 0.3: 0.422, 0.4: 0.278, 0.5: 0.168, 0.6: 0.090}


def wall_factor_haberman_sayre(lam: float) -> float:
    """u_wall/u_unbounded in creeping flow, sphere on the axis of a cylinder (Haberman & Sayre 1958), lam = d/D < 0.8."""
    drag_ratio = (1.0 - 0.75857 * lam ** 5) / (1.0 - 2.1050 * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5
                                               + 0.72603 * lam ** 6)
    return 1.0 / drag_ratio


def wall_factor_francis(lam: float) -> float:
    """Francis (1933), creeping flow (Re < 0.1), lam < 0.8."""
    return ((1.0 - lam) / (1.0 - 0.475 * lam)) ** 4


def wall_factor_di_felice(lam: float, re_unbounded: float) -> float:
    """Di Felice (1996), any flow regime; the exponent goes from 3.3 (viscous) to 0.85 (inertial)."""
    alpha = (3.3 + 0.085 * re_unbounded) / (1.0 + 0.1 * re_unbounded)
    return ((1.0 - lam) / (1.0 - 0.33 * lam)) ** alpha


def wall_factor_munroe(lam: float) -> float:
    """Munroe (1888), Newton regime (Re > 1000)."""
    return 1.0 - lam ** 1.5


def wall_effect(particle_diameter_m: float, tube_diameter_m: float, unbounded_velocity_m_s: float,
                reynolds_unbounded: float, fluid_density_kg_m3: float, fluid_viscosity_Pa_s: float) -> dict:
    """Wall-corrected terminal velocity of a sphere on the axis of a cylindrical tube (module docstring)."""
    d, D = float(particle_diameter_m), float(tube_diameter_m)
    lam = d / D
    if lam >= 1.0:
        raise ValueError(f"la partícula (d = {d:.4g} m) no cabe en el tubo (column_diameter_m = {D:.4g} m): d/D debe "
                         "ser < 1")
    Re = reynolds_unbounded
    creeping = Re < STOKES_RE_LIMIT
    in_lam = lam < WALL_LAMBDA_MAX
    table = {
        "Haberman_Sayre_1958": {"wall_factor": wall_factor_haberman_sayre(lam) if in_lam else None,
                                "validity": "flujo reptante (Re < 0.1 aquí; la fuente da Re < 2), d/D < 0.8",
                                "applicable": creeping and in_lam},
        "Francis_1933": {"wall_factor": wall_factor_francis(lam) if in_lam else None,
                         "validity": "Re < 0.1, d/D < 0.8", "applicable": creeping and in_lam},
        "Di_Felice_1996": {"wall_factor": wall_factor_di_felice(lam, Re),
                           "validity": "cualquier Re (empírica; intervalo de d/D no verificado; imprecisa en el régimen "
                                       "intermedio)", "applicable": True},
        "Munroe_1888": {"wall_factor": wall_factor_munroe(lam), "validity": "régimen de Newton, Re > 1000",
                        "applicable": Re > MUNROE_RE_MIN}}
    warnings: list[str] = []
    checks: dict = {}
    if creeping and in_lam:
        used, estimate = "Haberman_Sayre_1958", False
        if lam > 0.45:
            warnings.append(f"d/D = {lam:.3g}: Haberman-Sayre se aparta de la teoría exacta (−1.7 % en 0.5, −5 % en 0.6 "
                            "y más al acercarse a 0.8)")
    elif creeping:
        used, estimate = None, False
        warnings.append(f"d/D = {lam:.3g} ≥ 0.8: fuera del intervalo de Haberman-Sayre y Francis; no hay corrección de "
                        "pared verificada para este caso (se reporta Di Felice solo como referencia)")
    else:
        used, estimate = "Di_Felice_1996", True
        warnings.append(f"efecto de pared con Re = {Re:.4g} fuera del flujo reptante: se usa Di Felice (1996), "
                        "correlación empírica; trátalo como ESTIMACIÓN"
                        + (" (en régimen de Newton compárala con Munroe, también reportada)" if Re > MUNROE_RE_MIN
                           else ""))
    applicable = [v["wall_factor"] for v in table.values() if v["applicable"]]
    out: dict = {"diameter_ratio_d_over_D": lam, "reynolds_unbounded": Re, "correlation_used": used,
                 "is_estimate": estimate, "wall_factor_by_correlation": table}
    if used:
        f = table[used]["wall_factor"]
        uw = f * unbounded_velocity_m_s
        out.update(wall_factor=f, terminal_velocity_with_wall_m_s=uw,
                   reynolds_with_wall=fluid_density_kg_m3 * uw * d / fluid_viscosity_Pa_s,
                   velocity_reduction_percent=100.0 * (1.0 - f))
        checks["wall_factor_spread_between_applicable_rel"] = (max(applicable) - min(applicable)) / f
    else:
        out.update(wall_factor=None, terminal_velocity_with_wall_m_s=None)
    # the coded Haberman-Sayre formula against the exact-theory table it was published with
    checks["haberman_sayre_vs_exact_theory_table_max_abs"] = max(abs(wall_factor_haberman_sayre(x) - v)
                                                                 for x, v in EXACT_WALL_FACTOR.items())
    out.update(warnings=warnings, checks=checks,
               sources=["Haberman & Sayre (1958), David Taylor Model Basin Rep. 1143; coeficientes e intervalo según "
                        "arXiv:1202.1400 (Ecs. 20-22 y Tabla 9, teoría exacta)",
                        "Francis (1933), Physics 4, 403; forma según Braz. J. Chem. Eng. 16(4) 1999, Ec. 7",
                        "Di Felice (1996), Int. J. Multiphase Flow 22, 527; forma según arXiv:1202.1400, Ec. 18",
                        "Munroe (1888); forma según Braz. J. Chem. Eng. 16(4) 1999, Ec. 6"])
    return out


def stokes_drag_split(d: float, mu: float, velocity: float, Re: float, net_weight_N: float) -> dict:
    """Form (pressure) and friction (viscous) parts of the drag of a sphere from the exact Stokes solution."""
    if Re >= STOKES_RE_LIMIT:
        return {"available": False,
                "reason": f"Re = {Re:.4g} ≥ {STOKES_RE_LIMIT}: fuera del flujo reptante el reparto entre arrastre de forma "
                          "y de fricción no es 1/3-2/3 y requiere resolver el campo de flujo (no hay fórmula cerrada)"}
    total = 3.0 * math.pi * mu * d * velocity
    return {"available": True, "reynolds_limit_used": STOKES_RE_LIMIT, "stokes_drag_N": total,
            "form_drag_N": total / 3.0, "friction_drag_N": 2.0 * total / 3.0, "form_fraction": 1.0 / 3.0,
            "friction_fraction": 2.0 / 3.0, "net_weight_minus_buoyancy_N": net_weight_N,
            "stokes_drag_vs_net_weight_rel": abs(total - net_weight_N) / net_weight_N,
            "note": "solución exacta de Stokes para esfera en fluido infinito: F = 3πμdU = πμdU (forma, presión) + "
                    "2πμdU (fricción, esfuerzo viscoso); la pequeña diferencia con el peso neto es la inercia que la "
                    "curva de arrastre sí incluye"}


def _regime(Re: float) -> str:
    if Re < STOKES_RE_LIMIT:
        return "Stokes"
    if Re < NEWTON_RE_LIMIT:
        return "intermedio"
    if Re < DRAG_CRISIS_RE:
        return "Newton"
    return "crisis de arrastre"


def terminal_velocity(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
                      fluid_viscosity_Pa_s: float, sphericity: float, projected_area_ratio: float | None = None,
                      drag_method: str = "Clift") -> dict:
    """Terminal velocity of one particle in an unbounded still Newtonian fluid (module docstring)."""
    d = _pos("particle_diameter_m", particle_diameter_m,
             "diámetro de la partícula (esfera de igual volumen si no es esfera)")
    rho_p = _pos("particle_density_kg_m3", particle_density_kg_m3, "densidad de la partícula")
    rho_f = _pos("fluid_density_kg_m3", fluid_density_kg_m3, "densidad del fluido")
    mu = _pos("fluid_viscosity_Pa_s", fluid_viscosity_Pa_s, "viscosidad dinámica del fluido")
    phi = _req("sphericity", sphericity,
               "esfericidad de Wadell (1 para esfera): área de la esfera de igual volumen / área real")
    if not 0.0 < phi <= 1.0:
        raise ValueError("sphericity debe estar en (0, 1]")
    if rho_p == rho_f:
        raise ValueError("la partícula tiene la misma densidad que el fluido: no sedimenta ni flota")
    warnings: list[str] = []
    assumptions = ["partícula aislada en fluido newtoniano quieto e infinito (sin efecto de pared ni de otras partículas)",
                   "partícula rígida, lisa, con su orientación de caída estable"]
    sphere = phi >= 1.0 - 1e-12
    if sphere:
        if drag_method not in FD.drag_sphere_correlations:
            raise ValueError(f"drag_method desconocido {drag_method!r}; opciones: {sorted(FD.drag_sphere_correlations)}")
        method, S = drag_method, 1.0
    else:
        S = _pos("projected_area_ratio", projected_area_ratio, "para partícula no esférica: área de la esfera de igual "
                 "volumen / área proyectada en la dirección de caída (Song et al. 2017)")
        method = "Song_Xu"
    Ar = archimedes(d, rho_p, rho_f, mu)
    reasons: list[str] = []
    try:
        Re, res = _solve_re(Ar, method, phi, S)
    except ArithmeticError as exc:
        raise ValueError(f"no hay solución de velocidad terminal: {exc}") from exc
    Cd = drag_coefficient(Re, method, phi, S)
    ut = Re * mu / (rho_f * d)
    balance = math.sqrt(4.0 * G * d * abs(rho_p - rho_f) / (3.0 * rho_f * Cd))
    stokes = G * d * d * abs(rho_p - rho_f) / (18.0 * mu)
    checks = {"force_balance_rel": abs(balance - ut) / ut, "drag_equation_residual_rel": res}
    spread = {}
    if sphere:
        for m in SPHERE_METHODS:
            try:
                rm, _ = _solve_re(Ar, m, 1.0, 1.0)
                spread[m] = rm * mu / (rho_f * d)
            except (ArithmeticError, ValueError):
                spread[m] = None
        vals = [v for v in spread.values() if v]
        checks["correlation_spread_rel"] = (max(vals) - min(vals)) / ut if vals else None
        if Re > DRAG_CRISIS_RE:
            warnings.append("Re por encima de 2e5: crisis de arrastre de la esfera, el Cd depende de la rugosidad y la "
                            "turbulencia libre; resultado muy incierto")
    else:
        lo, hi = SONG_XU_RANGE
        if not lo <= Re <= hi:
            reasons.append(f"Re = {Re:.4g} fuera del rango de datos de Song-Xu ({lo}-{hi})")
            warnings.append("correlación no esférica fuera de su rango de datos: velocidad terminal no confiable")
        try:
            rs, _ = _solve_re(Ar, "Clift", 1.0, 1.0)
            spread["sphere_same_volume_Clift"] = rs * mu / (rho_f * d)
        except ArithmeticError:
            pass
        assumptions.append("arrastre no esférico de Song et al. (2017) vía fluids.drag.Song_Xu (esfericidad y relación de "
                           "áreas S son datos del usuario)")
    if Re < STOKES_RE_LIMIT:
        checks["stokes_limit_rel"] = abs(ut - stokes) / stokes
    if any(v > CHECK_TOL for k, v in checks.items() if k in ("force_balance_rel", "drag_equation_residual_rel")):
        reasons.append("el balance de fuerzas no cierra")
    if sphere:
        split = stokes_drag_split(d, mu, ut, Re, math.pi / 6.0 * d ** 3 * abs(rho_p - rho_f) * G)
    else:
        split = {"available": False, "reason": "el reparto exacto de Stokes (1/3 forma, 2/3 fricción) es solo para esferas"}
    return {"drag_split": split, "terminal_velocity_m_s": ut, "direction": "sedimenta" if rho_p > rho_f else "asciende (flota)",
            "reynolds_terminal": Re, "drag_coefficient": Cd, "archimedes_number": Ar, "regime": _regime(Re),
            "stokes_velocity_m_s": stokes, "drag_correlation": method,
            "terminal_velocity_by_correlation_m_s": spread,
            "method": f"Cd(Re)·Re² = 4/3·Ar resuelto con scipy.optimize.brentq; Cd de fluids.drag ({method})",
            "warnings": warnings, "assumptions": assumptions,
            "not_included": ["efecto de pared (se calcula solo para esfera si das column_diameter_m)",
                             "fluidos no newtonianos", "partículas brownianas", "gotas o burbujas "
                             "deformables"],
            "evidence": {"verified": not reasons, "checks": checks, "tolerance": CHECK_TOL, "reasons": reasons}}


def richardson_zaki_exponent(Re_t: float, d_over_D: float = 0.0) -> float:
    """Stepwise exponent of Richardson and Zaki (1954) with the d/D wall term."""
    x = d_over_D
    if Re_t < 0.2:
        return 4.65 + 19.5 * x
    if Re_t < 1.0:
        return (4.35 + 17.5 * x) * Re_t ** -0.03
    if Re_t < 200.0:
        return (4.45 + 18.0 * x) * Re_t ** -0.1
    if Re_t < 500.0:
        return 4.45 * Re_t ** -0.1
    return 2.39


def coe_clevenger_area(feed_flow_m3_s: float, feed_solids_kg_m3: float, underflow_solids_kg_m3: float,
                       concentrations_kg_m3: list[float] | None = None, settling_velocities_m_s: list[float] | None = None,
                       rz: dict | None = None) -> dict:
    """Thickener area by Coe-Clevenger from tabulated u(C) (user's tests) or from a Richardson-Zaki law (estimate)."""
    Q0 = _pos("thickener.feed_flow_m3_s", feed_flow_m3_s, "caudal de alimentación del espesador")
    C0 = _pos("thickener.feed_solids_kg_m3", feed_solids_kg_m3, "concentración de sólidos en la alimentación")
    Cu = _pos("thickener.underflow_solids_kg_m3", underflow_solids_kg_m3, "concentración de sólidos deseada en la descarga")
    if Cu <= C0:
        raise ValueError("thickener.underflow_solids_kg_m3 debe ser mayor que la concentración de la alimentación")
    warnings: list[str] = []
    if concentrations_kg_m3 is not None or settling_velocities_m_s is not None:
        if (not concentrations_kg_m3 or not settling_velocities_m_s
                or len(concentrations_kg_m3) != len(settling_velocities_m_s)):
            raise ValueError("thickener.batch_tests necesita listas 'concentration_kg_m3' y 'settling_velocity_m_s' "
                             "de igual "
                             "longitud (velocidad inicial de sedimentación por lotes a cada concentración)")
        C = np.asarray(concentrations_kg_m3, float)
        u = np.asarray(settling_velocities_m_s, float)
        if np.any(C <= 0) or np.any(u <= 0):
            raise ValueError("concentraciones y velocidades de sedimentación deben ser > 0")
        mask = (C >= C0 * (1 - 1e-12)) & (C < Cu)
        if not mask.any():
            raise ValueError("ninguna prueba por lotes tiene concentración entre la de alimentación y la de descarga")
        if not np.isclose(C[mask].min(), C0, rtol=0.05):
            warnings.append("no hay prueba a la concentración de alimentación: el máximo puede estar fuera de los datos")
        unit = (1.0 / C[mask] - 1.0 / Cu) / u[mask]
        i = int(np.argmax(unit))
        table = [{"concentration_kg_m3": float(c), "settling_velocity_m_s": float(v),
                  "unit_area_m2_s_kg": float((1 / c - 1 / Cu) / v)} for c, v in zip(C[mask], u[mask])]
        lim_C, lim_unit = float(C[mask][i]), float(unit[i])
        source, checks = "pruebas de sedimentación por lotes del usuario", {}
        if i == len(unit) - 1 and mask.sum() > 1:
            warnings.append("el área máxima está en la concentración más alta probada: prueba concentraciones mayores")
    else:
        ut, n, rho_s = rz["u_t"], rz["n"], rz["rho_s"]

        def unit_of(c):
            return (1.0 / c - 1.0 / Cu) / (ut * (1.0 - c / rho_s) ** n)
        grid = np.geomspace(C0, Cu * (1 - 1e-9), 4001)
        vals = np.array([unit_of(c) for c in grid])
        j = int(np.argmax(vals))
        if 0 < j < len(grid) - 1:
            opt = minimize_scalar(lambda c: -unit_of(c), bounds=(grid[j - 1], grid[j + 1]), method="bounded",
                                  options={"xatol": 1e-12 * Cu})
            lim_C, lim_unit = float(opt.x), float(-opt.fun)
        else:
            lim_C, lim_unit = float(grid[j]), float(vals[j])
        checks = {"grid_vs_optimizer_rel": abs(lim_unit - vals[j]) / lim_unit}
        table = [{"concentration_kg_m3": float(c), "settling_velocity_m_s": float(ut * (1 - c / rho_s) ** n)}
                 for c in np.linspace(C0, Cu, 6)[:-1]]
        source = "ESTIMADO con Richardson-Zaki (suspensión monodispersa no floculada)"
        warnings.append("área de espesador estimada con Richardson-Zaki: los lodos reales suelen flocular; "
                        "diseña con pruebas "
                        "por lotes medidas")
    A = Q0 * C0 * lim_unit
    return {"area_m2": A, "diameter_m": math.sqrt(4 * A / math.pi), "limiting_concentration_kg_m3": lim_C,
            "solids_flux_kg_m2_s": Q0 * C0 / A, "underflow_m3_s": Q0 * C0 / Cu, "overflow_m3_s": Q0 * (1 - C0 / Cu),
            "settling_data_source": source, "table": table, "warnings": warnings, "checks": checks}


def settling(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
             fluid_viscosity_Pa_s: float, sphericity: float, projected_area_ratio: float | None = None,
             drag_method: str = "Clift", solids_volume_fraction: float | None = None, column_diameter_m: float | None = None,
             thickener: dict | None = None) -> dict:
    """Terminal velocity, optional hindered settling (Richardson-Zaki) and optional thickener area (Coe-Clevenger)."""
    out = terminal_velocity(particle_diameter_m, particle_density_kg_m3, fluid_density_kg_m3, fluid_viscosity_Pa_s,
                            sphericity, projected_area_ratio, drag_method)
    ut, Re = out["terminal_velocity_m_s"], out["reynolds_terminal"]
    d = float(particle_diameter_m)
    dD = 0.0
    if column_diameter_m is not None:
        dD = d / _pos("column_diameter_m", column_diameter_m, "diámetro de la columna o recipiente, para el término d/D")
    if column_diameter_m is not None:
        if float(sphericity) >= 1.0 - 1e-12:
            we = wall_effect(d, float(column_diameter_m), ut, Re, float(fluid_density_kg_m3), float(fluid_viscosity_Pa_s))
            out["warnings"].extend(we.pop("warnings"))
            out["evidence"]["checks"].update({f"wall_{k}": v for k, v in we.pop("checks").items()})
            out["evidence"]["wall_effect_sources"] = we.pop("sources")
            out["terminal_velocity_with_wall_m_s"] = we["terminal_velocity_with_wall_m_s"]
            out["wall_effect"] = we
            out["assumptions"].append("efecto de pared: esfera rígida que cae por el EJE de un tubo cilíndrico largo; "
                                      "terminal_velocity_m_s sigue siendo la del fluido infinito y "
                                      "terminal_velocity_with_wall_m_s la corregida")
            out["method"] += "; efecto de pared por " + str(we["correlation_used"] or "ninguna correlación aplicable")
        else:
            out["warnings"].append("efecto de pared no calculado: las correcciones implementadas son solo para esferas")
    n = richardson_zaki_exponent(Re, dD)
    out["richardson_zaki_exponent"] = n
    out["richardson_zaki_d_over_D"] = dD
    if column_diameter_m is None:
        out["assumptions"].append("exponente de Richardson-Zaki con d/D = 0 (recipiente mucho mayor que la partícula)")
    if solids_volume_fraction is not None:
        phi_s = _frac("solids_volume_fraction", solids_volume_fraction, "fracción volumétrica de sólidos de la suspensión")
        uh = ut * (1.0 - phi_s) ** n
        out["hindered"] = {"solids_volume_fraction": phi_s, "hindered_settling_velocity_m_s": uh,
                           "ratio_to_terminal": uh / ut, "solids_flux_kg_m2_s": uh * phi_s * float(particle_density_kg_m3)}
        out["assumptions"].append("sedimentación obstaculizada de Richardson-Zaki: u = u_t·(1 − φ)^n, intercepto u_i = u_t "
                                  "(sin corrección de pared del intercepto), suspensión monodispersa no floculada")
    if thickener is not None:
        bt = thickener.get("batch_tests") or {}
        rz = None
        if not bt:
            rz = {"u_t": ut, "n": n, "rho_s": float(particle_density_kg_m3)}
        th = coe_clevenger_area(thickener.get("feed_flow_m3_s"), thickener.get("feed_solids_kg_m3"),
                                thickener.get("underflow_solids_kg_m3"), bt.get("concentration_kg_m3"),
                                bt.get("settling_velocity_m_s"), rz)
        out["warnings"].extend(th.pop("warnings"))
        out["evidence"]["checks"].update({f"thickener_{k}": v for k, v in th.pop("checks").items()})
        out["thickener"] = th
        out["assumptions"].append("área de Coe-Clevenger sin factor de seguridad (aplícalo tú) y sin zona de compresión")
        out["not_included"].append("profundidad del espesador y zona de compresión (método de Talmage-Fitch / Kynch)")
    out["method"] += "; Richardson-Zaki escalonado (1954); Coe-Clevenger por balance de sólidos en la capa límite"
    return out


def fluidization(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
                 fluid_viscosity_Pa_s: float, sphericity: float, voidage_mf: float | None = None,
                 bed_mass_kg: float | None = None, column_diameter_m: float | None = None,
                 superficial_velocity_m_s: float | None = None, projected_area_ratio: float | None = None,
                 drag_method: str = "Clift") -> dict:
    """Minimum fluidization (Ergun, Wen-Yu), terminal velocity, bed pressure drop, operating point and expansion."""
    d = _pos("particle_diameter_m", particle_diameter_m, "diámetro de partícula (esfera de igual volumen)")
    rho_p = _pos("particle_density_kg_m3", particle_density_kg_m3, "densidad de la partícula")
    rho_f = _pos("fluid_density_kg_m3", fluid_density_kg_m3, "densidad del fluido")
    mu = _pos("fluid_viscosity_Pa_s", fluid_viscosity_Pa_s, "viscosidad del fluido")
    phi = _req("sphericity", sphericity, "esfericidad de Wadell (1 para esfera), para Ergun y el arrastre")
    if not 0.0 < phi <= 1.0:
        raise ValueError("sphericity debe estar en (0, 1]")
    if rho_p <= rho_f:
        raise ValueError("la partícula debe ser más densa que el fluido para fluidizar hacia arriba")
    gas = rho_f < 100.0
    warnings: list[str] = []
    assumptions = [f"fase del fluido tomada como {'gas' if gas else 'líquido'} por su densidad ({rho_f:.4g} kg/m³)",
                   "partículas monodispersas del diámetro dado; lecho isotermo; distribuidor uniforme"]
    not_included = ["hidrodinámica de burbujas, expansión de lechos gas-sólido burbujeantes, arrastre y TDH",
                    "clasificación de Geldart", "lechos polidispersos y segregación", "pérdida de carga del distribuidor"]
    Ar = archimedes(d, rho_p, rho_f, mu)
    c1, c2 = WEN_YU
    re_wy = math.sqrt(c1 * c1 + c2 * Ar) - c1
    umf_wy = re_wy * mu / (rho_f * d)
    out: dict = {"archimedes_number": Ar, "u_mf_wen_yu_m_s": umf_wy, "re_mf_wen_yu": re_wy}
    checks: dict = {}
    reasons: list[str] = []
    umf_ergun = None
    if voidage_mf is not None:
        e = _frac("voidage_mf", voidage_mf, "porosidad del lecho en mínima fluidización")
        a = 1.75 / (phi * e ** 3)
        b = 150.0 * (1 - e) / (phi * phi * e ** 3)
        re_e = (-b + math.sqrt(b * b + 4 * a * Ar)) / (2 * a)
        umf_ergun = re_e * mu / (rho_f * d)
        per_len = (1 - e) * (rho_p - rho_f) * G
        u_root = brentq(lambda U: fluids_ergun(dp=phi * d, voidage=e, vs=U, rho=rho_f, mu=mu, L=1.0) - per_len,
                        umf_ergun * 1e-3, umf_ergun * 1e3, xtol=1e-300, rtol=ROOT_RTOL)
        checks["ergun_closed_form_vs_fluids_root_rel"] = abs(u_root - umf_ergun) / umf_ergun
        out.update(u_mf_ergun_m_s=umf_ergun, re_mf_ergun=re_e, voidage_mf=e,
                   ergun_vs_wen_yu_rel=(umf_ergun - umf_wy) / umf_wy)
        assumptions.append("Ergun con diámetro efectivo φ·d (150 y 1.75) vía fluids.packed_bed.Ergun; el lecho fijo bajo "
                           "u_mf se toma con la porosidad de mínima fluidización")
    else:
        warnings.append("sin voidage_mf no se calcula u_mf por Ergun ni la ΔP del lecho fijo; se usa Wen-Yu "
                        "(estima ε_mf y φ "
                        "implícitos, ±30 % típico)")
    umf = umf_ergun if umf_ergun is not None else umf_wy
    out["u_mf_used_m_s"] = umf
    out["u_mf_source"] = "Ergun (con ε_mf y φ del usuario)" if umf_ergun is not None else "Wen-Yu"
    tv = None
    if phi < 1.0 - 1e-12 and projected_area_ratio is None:
        ut = re_t = None
        warnings.append("partícula no esférica sin projected_area_ratio: no se calcula u_t "
                        "(arrastre de Song-Xu lo necesita); "
                        "la ventana de operación queda sin límite superior")
        out["terminal"] = None
    else:
        tv = terminal_velocity(d, rho_p, rho_f, mu, phi, projected_area_ratio, drag_method)
        ut, re_t = tv["terminal_velocity_m_s"], tv["reynolds_terminal"]
        checks.update({f"terminal_{k}": v for k, v in tv["evidence"]["checks"].items()})
        reasons += tv["evidence"]["reasons"]
        out["terminal"] = {k: tv[k] for k in ("terminal_velocity_m_s", "reynolds_terminal", "drag_coefficient", "regime",
                                             "drag_correlation", "terminal_velocity_by_correlation_m_s")}
    out["operating_window_m_s"] = {"u_mf": umf, "u_t": ut, "ratio_ut_umf": ut / umf if ut else None}
    if ut is not None and ut <= umf:
        warnings.append("u_t ≤ u_mf: incoherente (revisa ε_mf, φ o el diámetro); no hay ventana de fluidización")
    area = None
    if column_diameter_m is not None:
        Dc = _pos("column_diameter_m", column_diameter_m, "diámetro interno de la columna")
        area = math.pi * Dc * Dc / 4.0
        out["column_area_m2"] = area
        if d / Dc > 0.01:
            warnings.append(f"d/D = {d / Dc:.3g} > 0.01: efecto de pared apreciable")
    if bed_mass_kg is not None:
        M = _pos("bed_mass_kg", bed_mass_kg, "masa de sólidos en el lecho")
        if area is None:
            raise ValueError("falta column_diameter_m: diámetro de la columna, para la ΔP del lecho (peso/área)")
        dp_fl = M * G * (1.0 - rho_f / rho_p) / area
        out["fluidized_bed_pressure_drop_Pa"] = dp_fl
        if umf_ergun is not None:
            e = out["voidage_mf"]
            H_mf = M / (rho_p * (1 - e) * area)
            dp_erg = fluids_ergun(dp=phi * d, voidage=e, vs=umf_ergun, rho=rho_f, mu=mu, L=H_mf)
            checks["ergun_at_umf_vs_bed_weight_rel"] = abs(dp_erg - dp_fl) / dp_fl
            out["bed_height_mf_m"] = H_mf
    n_rz = richardson_zaki_exponent(re_t, d / math.sqrt(4 * area / math.pi) if area else 0.0) if re_t is not None else None
    out["richardson_zaki_exponent"] = n_rz
    if superficial_velocity_m_s is not None:
        U = _pos("superficial_velocity_m_s", superficial_velocity_m_s, "velocidad superficial de operación")
        op: dict = {"superficial_velocity_m_s": U, "u_over_umf": U / umf}
        if U < umf:
            op["state"] = "lecho fijo"
            if umf_ergun is not None:
                e = out["voidage_mf"]
                op["pressure_gradient_Pa_m"] = fluids_ergun(dp=phi * d, voidage=e, vs=U, rho=rho_f, mu=mu, L=1.0)
                if "bed_height_mf_m" in out:
                    op["pressure_drop_Pa"] = op["pressure_gradient_Pa_m"] * out["bed_height_mf_m"]
        elif ut is None or U < ut:
            op["state"] = "fluidizado"
            if "fluidized_bed_pressure_drop_Pa" in out:
                op["pressure_drop_Pa"] = out["fluidized_bed_pressure_drop_Pa"]
            if gas:
                op["state"] = "fluidizado (gas: burbujeante o particulado según el sólido)"
                warnings.append("fluidización con gas: la expansión del lecho no se calcula (burbujeo); ΔP = peso/área")
            elif ut is None:
                warnings.append("sin u_t no se calcula la expansión del lecho por Richardson-Zaki")
            else:
                eps = (U / ut) ** (1.0 / n_rz)
                op["voidage_richardson_zaki"] = eps
                op["check_u_from_voidage_rel"] = abs(ut * eps ** n_rz - U) / U
                checks["rz_expansion_inverse_rel"] = op["check_u_from_voidage_rel"]
                if umf_ergun is not None and eps < out["voidage_mf"]:
                    warnings.append("Richardson-Zaki da una porosidad menor que ε_mf a esta velocidad: las dos fuentes son "
                                    "incoherentes cerca de u_mf")
                if bed_mass_kg is not None:
                    op["bed_height_m"] = float(bed_mass_kg) / (rho_p * (1 - eps) * area)
                assumptions.append("expansión de lecho líquido (particulado) por Richardson-Zaki, U = u_t·ε^n")
        else:
            op["state"] = "transporte (U ≥ u_t): las partículas se arrastran fuera del lecho"
            warnings.append("velocidad de operación ≥ u_t: arrastre total del lecho")
        out["operating_point"] = op
    if any(v is not None and v > CHECK_TOL for k, v in checks.items()
           if k in ("ergun_closed_form_vs_fluids_root_rel", "ergun_at_umf_vs_bed_weight_rel", "rz_expansion_inverse_rel")):
        reasons.append("las identidades de Ergun / Richardson-Zaki no cierran")
    out.update(method=("u_mf: cuadrática de Ergun (forma cerrada, verificada con brentq sobre fluids.packed_bed.Ergun) y "
                       "Wen-Yu (33.7, 0.0408); u_t con fluids.drag; expansión líquida por Richardson-Zaki"),
               warnings=warnings + (tv["warnings"] if tv else []),
               assumptions=assumptions + (tv["assumptions"] if tv else []),
               not_included=not_included,
               evidence={"verified": not reasons, "checks": checks, "tolerance": CHECK_TOL, "reasons": reasons})
    return out
