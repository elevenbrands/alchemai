"""Pressure drop of time-independent non-Newtonian liquids in straight circular pipes (bloque MP).

What it answers: for a volumetric flow of a power-law, Bingham or Herschel-Bulkley fluid through a straight, smooth,
horizontal pipe, the frictional pressure drop, the wall shear stress, the friction factor, the flow regime and (laminar
only) the velocity profile and the plug radius.

Rheology (always the user's, never defaulted): tau = K·gamma^n (power law), tau = tau_y + mu_p·gamma (Bingham),
tau = tau_y + K·gamma^n (Herschel-Bulkley) for tau > tau_y, unsheared plug otherwise.

Laminar (exact, no correlation): the Rabinowitsch-Mooney integral of the constitutive law over the cross-section,
    Q = pi R^3 / tau_w^3 · ∫_{tau_y}^{tau_w} tau^2 gamma(tau) dtau
which for Herschel-Bulkley has the closed form (phi = tau_y/tau_w, m = 1/n)
    Q = pi R^3 (tau_w/K)^m [(1-phi)^(3+m)/(3+m) + 2 phi (1-phi)^(2+m)/(2+m) + phi^2 (1-phi)^(1+m)/(1+m)]
and reduces to the Metzner-Reed power-law relation and to the Buckingham-Reiner equation. The closed form is solved for
tau_w (brentq) and checked against the integral done by quadrature and against the velocity profile integrated over the
area. Generalised Reynolds number of Metzner and Reed: Re_MR = 8 rho V^2 / tau_w,laminar (so that f_Fanning = 16/Re_MR in
laminar flow for any inelastic fluid); for a power-law fluid it is rho V^(2-n) D^n / (K ((3n+1)/4n)^n 8^(n-1)).

Laminar-turbulent transition and turbulence (empirical, smooth pipe):
  * power law: critical Re_MR of Ryan-Johnson / Hanks, 6464 n (2+n)^((2+n)/(1+n)) / (1+3n)^2 (2100 at n = 1);
    turbulent Fanning factor of Dodge and Metzner (1959): 1/sqrt(f) = 4/n^0.75 log10(Re_MR f^(1-n/2)) - 0.4/n^1.2.
  * Bingham: critical Reynolds of Hanks from the Hedstrom number, xi_c/(1-xi_c)^3 = He/16800 and
    Re_c = He/(8 xi_c) (1 - 4/3 xi_c + xi_c^4/3) (the factor 8 follows from the Buckingham-Reiner equation itself);
    above it the Darby correlation (Darby and Melson 1981, Darby et al. 1992): f_T = 10^a Re_B^-0.193,
    a = -1.47 [1 + 0.146 exp(-2.9e-5 He)], f = (f_L^m + f_T^m)^(1/m), m = 1.7 + 40000/Re_B.
  * Herschel-Bulkley: no validated turbulent correlation here; above the estimated transition the tool gives an ESTIMATE:
    Dodge-Metzner with the local flow-curve slope n' and K' at the wall (generalised Metzner-Reed approach), flagged.

Not included: roughness effects on non-Newtonian turbulence, elevation and fittings, entrance length, wall slip,
thixotropy and viscoelasticity (drag reduction), settling slurries (heterogeneous regime), non-circular ducts."""
from __future__ import annotations

import math

import numpy as np
from fluids.friction import Colebrook
from scipy.integrate import quad
from scipy.optimize import brentq

HANKS_CONSTANT = 16800.0            # He = 16800 xi_c/(1 - xi_c)^3 at the onset of turbulence (Hanks)
NEWTONIAN_RE_CRIT = 2100.0
CLOSURE_TOL = 1e-6                  # closed form vs Rabinowitsch-Mooney quadrature and vs integrated profile
ROOT_RTOL = 1e-13
TRANSITION_BAND = 0.2               # ±20 % of the critical Reynolds number is reported as the uncertain band
MODELS = ("power_law", "bingham", "herschel_bulkley")


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


def parse_rheology(rheology: dict | None) -> dict:
    """Validate the user's rheology and return {model, tau_y, K, n} in SI (Bingham: K = mu_p, n = 1)."""
    if not rheology:
        raise ValueError("falta rheology: {'model': 'power_law' | 'bingham' | 'herschel_bulkley', ...} con los parámetros "
                         "reológicos medidos del fluido (no hay valores por defecto)")
    model = str(rheology.get("model", "")).strip().lower()
    if model not in MODELS:
        raise ValueError(f"rheology.model debe ser uno de {MODELS}; se recibió {rheology.get('model')!r}")
    if model == "power_law":
        K = _pos("rheology.K_Pa_sn", rheology.get("K_Pa_sn"), "índice de consistencia K (Pa·s^n) de τ = K·γ̇^n")
        n = _pos("rheology.n", rheology.get("n"), "índice de flujo n de τ = K·γ̇^n")
        tau_y = 0.0
    elif model == "bingham":
        tau_y = _req("rheology.yield_stress_Pa", rheology.get("yield_stress_Pa"), "esfuerzo de cedencia τ_y de Bingham")
        K = _pos("rheology.plastic_viscosity_Pa_s", rheology.get("plastic_viscosity_Pa_s"),
                 "viscosidad plástica μ_p de τ = τ_y + μ_p·γ̇")
        n = 1.0
    else:
        tau_y = _req("rheology.yield_stress_Pa", rheology.get("yield_stress_Pa"), 
                     "esfuerzo de cedencia τ_y de Herschel-Bulkley")
        K = _pos("rheology.K_Pa_sn", rheology.get("K_Pa_sn"), "índice de consistencia K de τ = τ_y + K·γ̇^n")
        n = _pos("rheology.n", rheology.get("n"), "índice de flujo n de τ = τ_y + K·γ̇^n")
    if tau_y < 0.0:
        raise ValueError("rheology.yield_stress_Pa no puede ser negativo")
    return {"model": model, "tau_y": tau_y, "K": K, "n": n}


# ------------------------------------------------------------------ laminar, exact
def laminar_flow_rate(tau_w: float, R: float, tau_y: float, K: float, n: float) -> float:
    """Closed-form Herschel-Bulkley volumetric flow for wall stress tau_w (0 below the yield stress)."""
    if tau_w <= tau_y:
        return 0.0
    m = 1.0 / n
    phi = tau_y / tau_w
    a = 1.0 - phi
    br = a ** (3 + m) / (3 + m) + 2 * phi * a ** (2 + m) / (2 + m) + phi * phi * a ** (1 + m) / (1 + m)
    return math.pi * R ** 3 * (tau_w / K) ** m * br


def laminar_flow_rate_quadrature(tau_w: float, R: float, tau_y: float, K: float, n: float) -> float:
    """Rabinowitsch-Mooney integral by adaptive quadrature: the independent check of the closed form."""
    if tau_w <= tau_y:
        return 0.0
    val, _ = quad(lambda t: t * t * ((t - tau_y) / K) ** (1.0 / n), tau_y, tau_w, epsabs=0.0, epsrel=1e-12, limit=200)
    return math.pi * R ** 3 / tau_w ** 3 * val


def laminar_velocity(r: float, R: float, tau_w: float, tau_y: float, K: float, n: float) -> float:
    """Axial velocity at radius r of the laminar profile (no slip at the wall, plug where tau < tau_y)."""
    tau = tau_w * r / R
    e = (n + 1.0) / n
    sw = (tau_w - tau_y) ** e
    s = max(tau - tau_y, 0.0) ** e
    return (R / tau_w) * K ** (-1.0 / n) * (n / (n + 1.0)) * (sw - s)


def laminar_wall_stress(Q: float, R: float, tau_y: float, K: float, n: float) -> float:
    """Wall shear stress that carries Q in laminar flow (inverse of laminar_flow_rate, brentq)."""
    lo = tau_y
    hi = max(tau_y, 1e-12) * 2.0 + K * (4.0 * Q / (math.pi * R ** 3)) ** n + 1e-12
    while laminar_flow_rate(hi, R, tau_y, K, n) < Q:
        hi *= 2.0
    return brentq(lambda t: laminar_flow_rate(t, R, tau_y, K, n) - Q, lo, hi, xtol=1e-300, rtol=ROOT_RTOL, maxiter=500)


def flow_curve_slope(tau_w: float, R: float, tau_y: float, K: float, n: float) -> tuple[float, float]:
    """n' = dln tau_w / dln(8V/D) and K' = tau_w/(8V/D)^n' of the laminar flow curve at tau_w (Rabinowitsch-Mooney)."""
    Q = laminar_flow_rate(tau_w, R, tau_y, K, n)
    V = Q / (math.pi * R * R)
    g_app = 8.0 * V / (2.0 * R)
    g_wall = ((tau_w - tau_y) / K) ** (1.0 / n)
    # gamma_w = (8V/D)(3n'+1)/(4n')  ->  n' = (8V/D) / (4 gamma_w - 3 (8V/D))
    n_p = g_app / (4.0 * g_wall - 3.0 * g_app)
    return n_p, tau_w / g_app ** n_p


# ------------------------------------------------------------------ transition and turbulence
def critical_re_power_law(n: float) -> float:
    """Ryan-Johnson / Hanks critical Metzner-Reed Reynolds number of a power-law fluid."""
    return 6464.0 * n * (2.0 + n) ** ((2.0 + n) / (1.0 + n)) / (1.0 + 3.0 * n) ** 2


def hanks_critical(He: float) -> tuple[float, float]:
    """(xi_c, Re_B,c) of Hanks for a Bingham plastic of Hedstrom number He."""
    if He < 1e-8:
        return 0.0, NEWTONIAN_RE_CRIT
    xi = brentq(lambda x: x / (1.0 - x) ** 3 - He / HANKS_CONSTANT, 1e-16, 1.0 - 1e-12, xtol=1e-300, rtol=ROOT_RTOL)
    return xi, He / (8.0 * xi) * (1.0 - 4.0 * xi / 3.0 + xi ** 4 / 3.0)


def dodge_metzner_f(Re: float, n: float) -> tuple[float, float]:
    """Fanning friction factor of Dodge-Metzner (smooth pipe) and the residual of the implicit equation."""
    def g(x):          # x = 1/sqrt(f)
        f = 1.0 / (x * x)
        return 4.0 / n ** 0.75 * math.log10(Re * f ** (1.0 - n / 2.0)) - 0.4 / n ** 1.2 - x
    x = brentq(g, 1e-3, 1e4, xtol=1e-300, rtol=ROOT_RTOL, maxiter=500)
    return 1.0 / (x * x), abs(g(x)) / x


def darby_bingham_f(Re_B: float, He: float, f_L: float) -> dict:
    """Darby correlation (Fanning) for a Bingham plastic over the whole Reynolds range."""
    a = -1.47 * (1.0 + 0.146 * math.exp(-2.9e-5 * He))
    f_T = 10.0 ** a * Re_B ** -0.193
    m = 1.7 + 40000.0 / Re_B
    f = (f_L ** m + f_T ** m) ** (1.0 / m)
    return {"f_fanning": f, "f_turbulent_fanning": f_T, "f_laminar_fanning": f_L, "exponent_m": m, "a": a}


# ------------------------------------------------------------------ public engine
def pipe_flow(volumetric_flow_m3_s: float, diameter_m: float, length_m: float, density_kg_m3: float, rheology: dict,
              profile_points: int = 21) -> dict:
    """Frictional pressure drop of a non-Newtonian liquid in a straight smooth pipe (see the module docstring)."""
    Q = _pos("volumetric_flow_m3_s", volumetric_flow_m3_s, "caudal volumétrico del fluido")
    D = _pos("diameter_m", diameter_m, "diámetro interno del tubo")
    L = _pos("length_m", length_m, "longitud recta del tubo")
    rho = _pos("density_kg_m3", density_kg_m3, "densidad del fluido, para el número de Reynolds y el régimen")
    rh = parse_rheology(rheology)
    model, tau_y, K, n = rh["model"], rh["tau_y"], rh["K"], rh["n"]
    if profile_points < 3:
        raise ValueError("profile_points debe ser ≥ 3")
    R = D / 2.0
    A = math.pi * R * R
    V = Q / A
    warnings: list[str] = []
    assumptions = ["flujo isotérmico, completamente desarrollado, en tubo recto horizontal y circular; ΔP solo por fricción",
                   "fluido inelástico e independiente del tiempo (sin tixotropía ni viscoelasticidad), sin deslizamiento en "
                   "la pared",
                   "la reología dada vale en todo el intervalo de esfuerzos de corte del tubo (0 a τ_w); fuera "
                   "del intervalo "
                   "en que se midió es extrapolación"]
    not_included = ["rugosidad de la pared en turbulencia no newtoniana (las correlaciones son de tubo liso)",
                    "cambios de elevación, accesorios, longitud de entrada", "lodos con sedimentación (régimen heterogéneo)",
                    "reducción de arrastre por viscoelasticidad", "ductos no circulares", "cálculo inverso (caudal dado ΔP)"]

    # laminar solution (always computed: it defines Re_MR and is the answer when the flow is laminar)
    tw_lam = laminar_wall_stress(Q, R, tau_y, K, n)
    q_closed = laminar_flow_rate(tw_lam, R, tau_y, K, n)
    q_quad = laminar_flow_rate_quadrature(tw_lam, R, tau_y, K, n)
    rr = np.linspace(0.0, R, profile_points)
    u = np.array([laminar_velocity(r, R, tw_lam, tau_y, K, n) for r in rr])
    q_prof, _ = quad(lambda r: 2.0 * math.pi * r * laminar_velocity(r, R, tw_lam, tau_y, K, n), 0.0, R,
                     points=[min(R * tau_y / tw_lam, R)] if tau_y > 0 else None, epsabs=0.0, epsrel=1e-12, limit=200)
    re_mr = 8.0 * rho * V * V / tw_lam
    f_lam = 16.0 / re_mr
    n_p, K_p = flow_curve_slope(tw_lam, R, tau_y, K, n)
    checks = {"laminar_closed_form_vs_quadrature_rel": abs(q_closed - q_quad) / Q,
              "laminar_closed_form_vs_Q_rel": abs(q_closed - Q) / Q,
              "laminar_profile_integral_vs_Q_rel": abs(q_prof - Q) / Q}
    out: dict = {"model": model, "mean_velocity_m_s": V,
                 "rheology": {"yield_stress_Pa": tau_y, "K_Pa_sn": K, "n": n},
                 "generalized_reynolds_metzner_reed": re_mr,
                 "flow_curve_at_wall": {"n_prime": n_p, "K_prime_Pa_sn": K_p,
                                        "apparent_wall_shear_rate_8V_D_1_s": 8.0 * V / D}}

    if model == "power_law":
        re_formula = rho * V ** (2 - n) * D ** n / (K * ((3 * n + 1) / (4 * n)) ** n * 8 ** (n - 1))
        checks["re_mr_identity_rel"] = abs(re_formula - re_mr) / re_mr
        re_c = critical_re_power_law(n)
        crit = {"criterion": "Ryan-Johnson / Hanks (Re_MR crítico)", "critical_reynolds": re_c}
    elif model == "bingham":
        Re_B = rho * V * D / K
        He = rho * D * D * tau_y / (K * K)
        xi_c, re_c_b = hanks_critical(He)
        out.update(bingham_reynolds=Re_B, hedstrom_number=He)
        crit = {"criterion": "Hanks (Re de Bingham crítico desde el número de Hedström)", "critical_reynolds": re_c_b,
                "xi_c": xi_c, "critical_wall_stress_Pa": tau_y / xi_c if xi_c > 0 else None}
        re_c = re_c_b
    else:
        He = rho * D * D / K * (tau_y / K) ** ((2.0 - n) / n) if tau_y > 0 else 0.0
        re_c = critical_re_power_law(n_p)
        out["hedstrom_number_HB"] = He
        crit = {"criterion": "ESTIMADO: Ryan-Johnson / Hanks con el n' local de la curva de flujo en la pared "
                             "(no validado para Herschel-Bulkley)", "critical_reynolds": re_c}
    re_cmp = out.get("bingham_reynolds", re_mr)
    turbulent = re_cmp >= re_c
    if abs(re_cmp - re_c) <= TRANSITION_BAND * re_c:
        warnings.append(f"Reynolds {re_cmp:.4g} dentro de ±{TRANSITION_BAND:.0%} del crítico {re_c:.4g}: "
                        "zona de transición, "
                        "el régimen y la ΔP son inciertos; diseña con el mayor de los dos valores reportados")
    out["transition"] = crit
    laminar_block = {"wall_shear_stress_Pa": tw_lam, "pressure_drop_Pa": 4.0 * L * tw_lam / D, "f_fanning": f_lam,
                     "wall_shear_rate_1_s": ((tw_lam - tau_y) / K) ** (1.0 / n),
                     "plug_radius_m": min(R * tau_y / tw_lam, R),
                     "velocity_profile": {"r_m": rr.tolist(), "u_m_s": u.tolist()},
                     "centerline_velocity_m_s": float(u[0]), "centerline_to_mean": float(u[0]) / V}
    out["laminar_solution"] = laminar_block

    turb_ok = True
    if not turbulent:
        regime, tw, f = "laminar", tw_lam, f_lam
        method = ("laminar exacto: integral de Rabinowitsch-Mooney del modelo " + model
                  + " en forma cerrada resuelta para τ_w "
                  "con scipy.optimize.brentq, verificada con scipy.integrate.quad")
    elif model == "power_law":
        f, res = dodge_metzner_f(re_mr, n)
        checks["dodge_metzner_residual"] = res
        turb_ok = res < 1e-10
        regime, tw = "turbulento", f * rho * V * V / 2.0
        method = "turbulento, tubo liso: Dodge-Metzner (1959) con Re de Metzner-Reed, resuelto con brentq"
        out["turbulent"] = {"correlation": "Dodge-Metzner", "f_fanning": f}
        if n > 1.0:
            warnings.append("n > 1 (dilatante) en turbulencia: Dodge-Metzner se ajustó con fluidos pseudoplásticos; "
                            "extrapolación")
    elif model == "bingham":
        d = darby_bingham_f(out["bingham_reynolds"], out["hedstrom_number"], f_lam)
        f = d["f_fanning"]
        regime, tw = "turbulento", f * rho * V * V / 2.0
        method = "Bingham por encima del crítico de Hanks: correlación de Darby (Darby y Melson 1981; Darby et al. 1992)"
        f_newt = Colebrook(out["bingham_reynolds"], 0.0) / 4.0
        out["turbulent"] = {"correlation": "Darby", **d, "newtonian_smooth_f_fanning_at_Re_B": f_newt}
        if f < f_newt:
            warnings.append(f"Darby da f = {f:.4g}, menor que la de un fluido newtoniano de viscosidad μ_p en tubo liso "
                            f"({f_newt:.4g}, Colebrook): la correlación subestima la fricción con Hedström bajo "
                            "(en He → 0 da cerca de la mitad del valor newtoniano) y contra datos de de Nevers "
                            "13.5 quedó −24 %; considera "
                            "el valor newtoniano como cota inferior conservadora")
    else:
        regime = "turbulento (estimado)"

        def resid(tw_try):
            npp, Kpp = flow_curve_slope(tw_try, R, tau_y, K, n)
            re_g = rho * V ** (2 - npp) * D ** npp / (Kpp * 8 ** (npp - 1))
            ff, _ = dodge_metzner_f(re_g, npp)
            return ff * rho * V * V / 2.0 - tw_try
        try:
            lo = hi = tw_lam
            if resid(tw_lam) > 0:
                while resid(hi) > 0:
                    lo, hi = hi, hi * 1.5
            else:
                while resid(lo) <= 0:
                    hi, lo = lo, tau_y + (lo - tau_y) / 1.5
            tw = brentq(resid, lo, hi, xtol=1e-300, rtol=1e-12, maxiter=500)
            npp, Kpp = flow_curve_slope(tw, R, tau_y, K, n)
            re_g = rho * V ** (2 - npp) * D ** npp / (Kpp * 8 ** (npp - 1))
            f = 2.0 * tw / (rho * V * V)
            checks["hb_turbulent_residual_rel"] = abs(resid(tw)) / tw
            out["turbulent"] = {"correlation": "ESTIMADO: Dodge-Metzner con n', K' locales (Metzner-Reed generalizado)",
                                "n_prime_at_wall": npp, "K_prime_Pa_sn": Kpp, "generalized_reynolds": re_g, "f_fanning": f}
        except ValueError:
            turb_ok, tw, f = False, float("nan"), float("nan")
        warnings.append("Herschel-Bulkley en turbulencia: no hay correlación validada en esta herramienta; el valor es una "
                        "ESTIMACIÓN (Dodge-Metzner con la pendiente local de la curva de flujo)")
        method = ("Herschel-Bulkley turbulento ESTIMADO: Dodge-Metzner con n' y K' de la curva de flujo laminar en τ_w "
                  "(brentq)")
    if turbulent:
        not_included.append("perfil de velocidad turbulento (solo se reporta el laminar, como referencia)")
        if math.isfinite(tw) and tw < tw_lam:
            warnings.append("la correlación turbulenta da menos fricción que la solución laminar al mismo caudal: estás en "
                            "transición; usa la ΔP laminar como cota conservadora")

    dp = 4.0 * L * tw / D
    out.update(regime=regime, wall_shear_stress_Pa=tw, pressure_drop_Pa=dp, pressure_gradient_Pa_m=dp / L, f_fanning=f,
               f_darcy=4.0 * f, hydraulic_power_W=dp * Q)
    if tau_y > 0 and not turbulent:
        assumptions.append("tapón central sin cizalla donde τ < τ_y (plug_radius_m)")
    lam_ok = all(checks[k] <= CLOSURE_TOL for k in ("laminar_closed_form_vs_quadrature_rel", "laminar_closed_form_vs_Q_rel",
                                                     "laminar_profile_integral_vs_Q_rel"))
    reasons = []
    if not lam_ok:
        reasons.append("la forma cerrada laminar no cierra con la cuadratura o con el perfil integrado")
    if not turb_ok:
        reasons.append("la ecuación turbulenta no convergió")
    if checks.get("re_mr_identity_rel", 0.0) > 1e-10:
        reasons.append("el Re de Metzner-Reed no coincide con su forma explícita")
    out.update(method=method, warnings=warnings, assumptions=assumptions, not_included=not_included,
               evidence={"verified": not reasons, "checks": checks, "tolerance": CLOSURE_TOL, "reasons": reasons})
    return out
