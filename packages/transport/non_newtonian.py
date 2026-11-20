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
thixotropy and viscoelasticity (drag reduction), settling slurries (heterogeneous regime).

Extensions (same tool, see `flow`):
  * Generalised Newtonian models without yield stress and without closed form: Carreau
    eta = eta_inf + (eta0 - eta_inf)[1 + (lambda g)^2]^((n-1)/2), Carreau-Yasuda (exponent a instead of 2:
    [1 + (lambda g)^a]^((n-1)/a)) and Cross, eta = eta_inf + (eta0 - eta_inf)/(1 + (lambda g)^m); plus 'newtonian'.
  * Slit (two wide parallel plates, gap h) besides the tube, laminar. With s the distance from the axis / mid-plane,
    S = R or h/2, and the linear stress tau = tau_w s/S (tau_w = S·G/2 in the tube, S·G in the slit, G = -dp/dz):
        V = S J_k / tau_w^k,  J_k = ∫_0^tau_w tau^(k-1) g(tau) dtau,  k = 3 (tube), 2 (slit)
        u(s) = (S/tau_w) ∫_tau(s)^tau_w g(tau) dtau
    Herschel-Bulkley family: J_k and u in closed form, checked by quadrature. Carreau family: g(tau) by root finding
    (brentq) and Gauss-Legendre quadrature at N and 2N nodes, checked against the integral by parts in the shear-rate
    variable, J_k = tau_w^k g_w/k - (1/k)∫_0^g_w tau(g)^k dg, which needs no inversion. Both directions: flow -> gradient
    (brentq on tau_w) and gradient -> flow (direct).
  * Apparent viscosity eta = tau/g of one or two rheologies at given shear rates and the shear rate(s) where they cross.
  * Blasius laminar boundary layer on a flat plate: `packages.transport.boundary_layer`.
Not included in the extensions: turbulence of Carreau-type fluids and of any fluid in a slit (laminar solution only, with
an indicative Reynolds check), side-wall effects of a slit of finite width, annuli and other ducts."""
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
MODELS = ("power_law", "bingham", "herschel_bulkley", "newtonian", "carreau", "carreau_yasuda", "cross")
GNF_MODELS = ("carreau", "carreau_yasuda", "cross")     # no yield stress, no closed form: numerical integration
GEOMETRIES = {"tube": "tube", "pipe": "tube", "tubo": "tube", "circular": "tube",
              "slit": "slit", "parallel_plates": "slit", "plates": "slit", "placas": "slit", "placas_paralelas": "slit",
              "rendija": "slit", "flat_plate": "flat_plate", "placa_plana": "flat_plate", "plate": "flat_plate",
              "boundary_layer": "flat_plate"}
INDICATIVE_RE_CRIT = 2100.0         # laminar check where no validated transition criterion exists (flagged)
GL_NODES = 48                       # Gauss-Legendre nodes of the numerical path (compared with 2x)
CROSSING_RANGE = (1e-8, 1e10)       # shear rates (1/s) scanned for viscosity crossings


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
        raise ValueError("falta rheology: {'model': 'power_law' | 'bingham' | 'herschel_bulkley' | 'newtonian' | 'carreau' "
                         "| 'carreau_yasuda' | 'cross', ...} con los parámetros reológicos medidos del fluido (no hay "
                         "valores por defecto)")
    model = str(rheology.get("model", "")).strip().lower()
    if model not in MODELS:
        raise ValueError(f"rheology.model debe ser uno de {MODELS}; se recibió {rheology.get('model')!r}")
    if model in GNF_MODELS:
        return _parse_gnf(model, rheology)
    if model == "newtonian":
        K = _pos("rheology.viscosity_Pa_s", rheology.get("viscosity_Pa_s"), "viscosidad μ del modelo newtoniano τ = μ·γ̇")
        n, tau_y = 1.0, 0.0
    elif model == "power_law":
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
    if rh["model"] in GNF_MODELS:
        return _laminar_duct("tube", rh, D / 2.0, Q=Q, length_m=L, density=rho, profile_points=profile_points)
    model, tau_y, K, n = rh["model"], rh["tau_y"], rh["K"], rh["n"]
    model_label = model
    if model == "newtonian":            # Newtonian = power law with n = 1 (same exact solution and Dodge-Metzner limit)
        model = "power_law"
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
                    "reducción de arrastre por viscoelasticidad", "cálculo inverso en régimen turbulento"]

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
    out: dict = {"model": model_label, "mean_velocity_m_s": V,
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


# ------------------------------------------------------------------ generalised Newtonian models (Carreau family)
def _parse_gnf(model: str, rheology: dict) -> dict:
    """Carreau / Carreau-Yasuda / Cross parameters (SI). eta_inf defaults to 0 only if the user omits it."""
    name = {"carreau": "Carreau", "carreau_yasuda": "Carreau-Yasuda", "cross": "Cross"}[model]
    eta0 = _pos("rheology.eta0_Pa_s", rheology.get("eta0_Pa_s"), f"viscosidad a corte cero η0 del modelo de {name}")
    eta_inf = _req("rheology.eta_inf_Pa_s", rheology.get("eta_inf_Pa_s", 0.0),
                   f"viscosidad a corte infinito η∞ del modelo de {name}")
    lam = _req("rheology.lambda_s", rheology.get("lambda_s"), f"constante de tiempo λ (s) del modelo de {name}")
    if eta_inf < 0.0 or eta_inf > eta0:
        raise ValueError("rheology.eta_inf_Pa_s debe cumplir 0 ≤ η∞ ≤ η0")
    if lam < 0.0:
        raise ValueError("rheology.lambda_s no puede ser negativo")
    rh = {"model": model, "tau_y": 0.0, "eta0": eta0, "eta_inf": eta_inf, "lam": lam,
          "eta_inf_defaulted": "eta_inf_Pa_s" not in rheology}
    if model == "cross":
        rh["m"] = _pos("rheology.m", rheology.get("m"), "exponente m del modelo de Cross η = η∞ + (η0 − η∞)/(1 + (λγ̇)^m)")
        if rh["m"] > 1.0 and lam > 0.0:      # tau(g) may stop being monotonic: then the duct problem is ill-posed
            g = np.geomspace(1e-8, 1e8, 3201) / lam
            tau = np.array([shear_stress(rh, float(x)) for x in g])
            if np.any(np.diff(tau) <= 0.0):
                raise ValueError("rheology (Cross): con m > 1 y esta η∞ la curva τ(γ̇) no es monótona (el esfuerzo baja al "
                                 "subir la rapidez de corte): el flujo en ducto no tiene solución única; revisa m y η∞")
    else:
        rh["n"] = _pos("rheology.n", rheology.get("n"), f"índice n del modelo de {name} (pendiente n − 1 de la zona de "
                       "ley de potencia)")
        rh["a"] = 2.0 if model == "carreau" else _pos("rheology.a", rheology.get("a"),
                                                       "exponente a de Yasuda (a = 2 recupera Carreau)")
    return rh


def apparent_viscosity(rh: dict, g: float) -> float:
    """eta = tau/g of a parsed rheology at shear rate g > 0 (infinite at g -> 0 with a yield stress)."""
    model = rh["model"]
    if model in GNF_MODELS:
        x = rh["lam"] * g
        d = rh["eta0"] - rh["eta_inf"]
        if model == "cross":
            return rh["eta_inf"] + d / (1.0 + x ** rh["m"])
        a = rh["a"]
        return rh["eta_inf"] + d * (1.0 + x ** a) ** ((rh["n"] - 1.0) / a)
    return (rh["tau_y"] / g if rh["tau_y"] > 0.0 else 0.0) + rh["K"] * g ** (rh["n"] - 1.0)


def shear_stress(rh: dict, g: float) -> float:
    """tau(g) of a parsed rheology (tau_y at g = 0)."""
    if g <= 0.0:
        return rh["tau_y"]
    if rh["model"] in GNF_MODELS:
        return apparent_viscosity(rh, g) * g
    return rh["tau_y"] + rh["K"] * g ** rh["n"]


def shear_rate_from_stress(rh: dict, tau: float) -> float:
    """g(tau): closed form for the Herschel-Bulkley family, brentq on tau(g) = tau for the Carreau family."""
    if tau <= rh["tau_y"]:
        return 0.0
    if rh["model"] not in GNF_MODELS:
        return ((tau - rh["tau_y"]) / rh["K"]) ** (1.0 / rh["n"])
    lo = hi = tau / rh["eta0"]
    while shear_stress(rh, lo) > tau:
        lo *= 0.25
    while shear_stress(rh, hi) < tau:
        hi *= 4.0
    if lo == hi:
        return lo
    return brentq(lambda g: shear_stress(rh, g) - tau, lo, hi, xtol=1e-300, rtol=ROOT_RTOL, maxiter=500)


# ------------------------------------------------------------------ laminar flow in a tube (k = 3) or a slit (k = 2)
def _J_closed(rh: dict, tau_w: float, k: int) -> float:
    """J_k = ∫ tau^(k-1) g dtau in closed form (Herschel-Bulkley family)."""
    ty, m = rh["tau_y"], 1.0 / rh["n"]
    a = tau_w - ty
    if a <= 0.0:
        return 0.0
    if k == 2:
        br = a ** (m + 2) / (m + 2) + ty * a ** (m + 1) / (m + 1)
    else:
        br = a ** (m + 3) / (m + 3) + 2.0 * ty * a ** (m + 2) / (m + 2) + ty * ty * a ** (m + 1) / (m + 1)
    return rh["K"] ** -m * br


def _J_gauss(rh: dict, tau_w: float, k: int, nodes: int) -> float:
    """J_k by Gauss-Legendre in the stress variable, shear rate from the stress (root finding where needed)."""
    ty = rh["tau_y"]
    if tau_w <= ty:
        return 0.0
    x, w = np.polynomial.legendre.leggauss(nodes)
    half = 0.5 * (tau_w - ty)
    return float(sum(wi * (ty + half * (xi + 1.0)) ** (k - 1) * shear_rate_from_stress(rh, ty + half * (xi + 1.0))
                     for xi, wi in zip(x, w)) * half)


def _J_by_parts(rh: dict, tau_w: float, k: int, g_w: float) -> float:
    """J_k = tau_w^k g_w/k − (1/k)∫_0^g_w tau(g)^k dg: quadrature in the shear-rate variable, no inversion needed."""
    if g_w <= 0.0:
        return 0.0
    lam = rh.get("lam", 0.0)
    pts = [1.0 / lam] if lam > 0.0 and 1.0 / lam < g_w else None
    val, _ = quad(lambda g: shear_stress(rh, g) ** k, 0.0, g_w, points=pts, epsabs=0.0, epsrel=1e-12, limit=400)
    return (tau_w ** k * g_w - val) / k


def _J(rh: dict, tau_w: float, k: int) -> float:
    if rh["model"] in GNF_MODELS:
        return _J_by_parts(rh, tau_w, k, shear_rate_from_stress(rh, tau_w))
    return _J_closed(rh, tau_w, k)


def duct_velocity(rh: dict, s: float, S: float, tau_w: float) -> float:
    """Laminar velocity at distance s from the axis (tube) or the mid-plane (slit); S = R or h/2."""
    if rh["model"] not in GNF_MODELS:
        return laminar_velocity(s, S, tau_w, rh["tau_y"], rh["K"], rh["n"])
    tau_s = tau_w * s / S
    g_w, g_s = shear_rate_from_stress(rh, tau_w), shear_rate_from_stress(rh, tau_s)
    lam = rh["lam"]
    pts = [1.0 / lam] if lam > 0.0 and g_s < 1.0 / lam < g_w else None
    val, _ = quad(lambda g: shear_stress(rh, g), g_s, g_w, points=pts, epsabs=0.0, epsrel=1e-12, limit=400)
    return (S / tau_w) * (tau_w * g_w - tau_s * g_s - val)


def _wall_stress_for_velocity(rh: dict, V: float, S: float, k: int) -> float:
    """tau_w that gives the mean velocity V = S J_k/tau_w^k (brentq; V grows monotonically with tau_w)."""
    def f(tw):
        return S * _J(rh, tw, k) / tw ** k - V
    ty = rh["tau_y"]
    t0 = shear_stress(rh, (k + 1.0) * V / S)        # Newtonian wall shear rate as the first guess
    lo = hi = max(t0, ty * (1.0 + 1e-9), 1e-300)
    if ty > 0.0:
        lo = ty
    else:
        while f(lo) > 0.0:
            lo *= 0.5
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, lo, hi, xtol=1e-300, rtol=ROOT_RTOL, maxiter=500)


def _positions(rh: dict, positions_m, S: float, tau_w: float, key: str, what: str) -> list[dict]:
    rows = []
    for i, p in enumerate(positions_m or []):
        s = abs(_req(f"positions_m[{i}]", p, what))
        if s > S * (1.0 + 1e-12):
            raise ValueError(f"positions_m[{i}] = {p} queda fuera del ducto: {what}, entre 0 y {S:.6g} m")
        s = min(s, S)
        tau = tau_w * s / S
        rows.append({key: s, "distance_from_wall_m": S - s, "u_m_s": duct_velocity(rh, s, S, tau_w),
                     "shear_stress_Pa": tau, "shear_rate_1_s": shear_rate_from_stress(rh, tau)})
    return rows


def _laminar_duct(geometry: str, rh: dict, S: float, Q: float | None = None, gradient: float | None = None,
                  length_m: float | None = None, density: float | None = None, profile_points: int = 21,
                  positions_m: list | None = None, width_m: float | None = None) -> dict:
    """Laminar fully developed flow in a tube (S = R, Q in m³/s) or a slit (S = h/2, Q per unit width in m²/s) for any
    supported rheology, in either direction (Q -> gradient or gradient -> Q). See the module docstring."""
    if profile_points < 3:
        raise ValueError("profile_points debe ser ≥ 3")
    tube = geometry == "tube"
    k = 3 if tube else 2
    area = math.pi * S * S if tube else 2.0 * S          # slit: per unit width
    gnf = rh["model"] in GNF_MODELS
    ty = rh["tau_y"]
    key = "r_m" if tube else "y_m"
    what = "distancia al eje del tubo (m)" if tube else "distancia al plano medio entre las placas (m)"
    warnings: list[str] = []
    reasons: list[str] = []
    checks: dict = {}
    if gradient is not None:
        tau_w = S * gradient / 2.0 if tube else S * gradient
        direction = "gradiente de presión -> caudal"
        V = S * _J(rh, tau_w, k) / tau_w ** k if tau_w > ty else 0.0
    else:
        V = Q / area
        tau_w = _wall_stress_for_velocity(rh, V, S, k)
        direction = "caudal -> gradiente de presión"
        checks["mean_velocity_closure_rel"] = abs(S * _J(rh, tau_w, k) / tau_w ** k - V) / V
    G = 2.0 * tau_w / S if tube else tau_w / S
    out: dict = {"geometry": "tubo circular" if tube else "placas paralelas (rendija)", "model": rh["model"],
                 "direction": direction, "wall_shear_stress_Pa": tau_w, "pressure_gradient_Pa_m": G}
    out["half_gap_m" if not tube else "radius_m"] = S
    assumptions = ["flujo laminar, isotérmico, estacionario y completamente desarrollado; sin deslizamiento en la pared",
                   "fluido inelástico e independiente del tiempo; la reología dada vale de 0 a τ_w",
                   "τ varía linealmente con la distancia al " + ("eje" if tube else "plano medio") + " (balance de "
                   "cantidad de movimiento); positions_m se mide desde ahí"]
    if not tube:
        assumptions.append("placas mucho más anchas que la separación (W ≫ h): sin efecto de las paredes laterales; "
                           "sin width_m los caudales son por unidad de ancho (m²/s)")
    if rh.get("eta_inf_defaulted"):
        assumptions.append("η∞ no se dio: se tomó η∞ = 0")
    not_included = ["régimen turbulento (solo solución laminar; el Reynolds crítico es indicativo)",
                    "longitud de entrada, accesorios, cambios de elevación", "viscoelasticidad, tixotropía, "
                    "deslizamiento en la pared"]
    if V <= 0.0:
        out.update(regime="sin flujo", mean_velocity_m_s=0.0, max_velocity_m_s=0.0, wall_shear_rate_1_s=0.0)
        warnings.append(f"τ_w = {tau_w:.4g} Pa no supera el esfuerzo de cedencia τ_y = {ty:.4g} Pa: el fluido no fluye "
                        "con ese gradiente de presión")
        out.update(method="balance de cantidad de movimiento: τ_w ≤ τ_y", warnings=warnings, assumptions=assumptions,
                   not_included=not_included,
                   evidence={"verified": True, "checks": {"tau_w_over_tau_y": tau_w / ty}, "tolerance": CLOSURE_TOL,
                             "reasons": []})
        return out
    g_w = shear_rate_from_stress(rh, tau_w)
    # independent evaluations of the flow integral
    j_ref = _J(rh, tau_w, k)
    if gnf:
        j_n, j_2n = _J_gauss(rh, tau_w, k, GL_NODES), _J_gauss(rh, tau_w, k, 2 * GL_NODES)
        checks["by_parts_vs_stress_quadrature_rel"] = abs(j_ref - j_2n) / j_ref
        checks["quadrature_refinement_N_vs_2N_rel"] = abs(j_n - j_2n) / j_2n
        checks["shear_rate_root_residual_rel"] = abs(shear_stress(rh, g_w) - tau_w) / tau_w
    else:
        j_q, _ = quad(lambda t: t ** (k - 1) * shear_rate_from_stress(rh, t), ty, tau_w, epsabs=0.0, epsrel=1e-12,
                      limit=200)
        checks["closed_form_vs_quadrature_rel"] = abs(j_ref - j_q) / j_ref
    # profile and its integral over the cross-section
    ss = np.linspace(0.0, S, profile_points)
    u = [float(duct_velocity(rh, float(s), S, tau_w)) for s in ss]
    u_max = u[0]
    s_plug = min(S * ty / tau_w, S)

    def profile_mean(nodes: int) -> float:
        total = 0.0
        for a, b in ((0.0, s_plug), (s_plug, S)):
            if b <= a:
                continue
            x, w = np.polynomial.legendre.leggauss(nodes)
            for xi, wi in zip(x, w):
                s = a + 0.5 * (b - a) * (xi + 1.0)
                total += wi * 0.5 * (b - a) * duct_velocity(rh, s, S, tau_w) * (2.0 * s / (S * S) if tube else 1.0 / S)
        return float(total)
    v_prof = profile_mean(24 if gnf else 64)
    checks["profile_integral_vs_mean_velocity_rel"] = abs(v_prof - V) / V
    out.update(mean_velocity_m_s=V, max_velocity_m_s=u_max, max_to_mean=u_max / V, wall_shear_rate_1_s=g_w,
               apparent_wall_viscosity_Pa_s=tau_w / g_w)
    if tube:
        out.update(volumetric_flow_m3_s=V * area, nominal_shear_rate_8V_D_1_s=4.0 * V / S, plug_radius_m=s_plug)
    else:
        out.update(flow_per_unit_width_m2_s=V * area, gap_m=2.0 * S, nominal_shear_rate_6V_h_1_s=3.0 * V / S,
                   plug_half_thickness_m=s_plug)
        if width_m is not None:
            out.update(width_m=width_m, volumetric_flow_m3_s=V * area * width_m)
            if width_m < 10.0 * 2.0 * S:
                warnings.append(f"ancho/separación = {width_m / (2 * S):.3g} < 10: las paredes laterales frenan el flujo "
                                "y la solución de placas infinitas sobrestima el caudal")
    if length_m is not None:
        out.update(length_m=length_m, pressure_drop_Pa=G * length_m)
        if "volumetric_flow_m3_s" in out:
            out["hydraulic_power_W"] = G * length_m * out["volumetric_flow_m3_s"]
    out["velocity_profile"] = {key: ss.tolist(), "u_m_s": u}
    if positions_m:
        out["velocity_at_positions"] = _positions(rh, positions_m, S, tau_w, key, what)
    # laminar check
    if density is not None:
        re_g = (8.0 if tube else 12.0) * density * V * V / tau_w
        f = 2.0 * tau_w / (density * V * V)
        re_c, crit, re_cmp = INDICATIVE_RE_CRIT, "INDICATIVO: 2100, valor newtoniano (sin criterio validado para este caso)", re_g
        if tube and rh["model"] in ("power_law", "newtonian"):
            re_c, crit = critical_re_power_law(rh["n"]), "Ryan-Johnson / Hanks (Re_MR crítico)"
        elif tube and rh["model"] == "bingham":
            re_cmp = density * V * 2.0 * S / rh["K"]
            He = density * 4.0 * S * S * ty / rh["K"] ** 2
            re_c, crit = hanks_critical(He)[1], "Hanks (Re de Bingham crítico desde el número de Hedström)"
            out.update(bingham_reynolds=re_cmp, hedstrom_number=He)
        out["generalized_reynolds" + ("_metzner_reed" if tube else "_slit_hydraulic_diameter")] = re_g
        out.update(f_fanning=f, f_darcy=4.0 * f, transition={"criterion": crit, "critical_reynolds": re_c})
        if re_cmp >= re_c:
            out["regime"] = "laminar NO aplicable (Reynolds por encima del crítico)"
            warnings.append(f"Reynolds {re_cmp:.4g} ≥ crítico {re_c:.4g}: el flujo no sería laminar y esta solución no "
                            "aplica (no hay correlación turbulenta para este modelo/geometría en la herramienta)")
            reasons.append("Reynolds por encima del crítico: la solución laminar no aplica")
        else:
            out["regime"] = "laminar"
            if re_cmp >= (1.0 - TRANSITION_BAND) * re_c:
                warnings.append(f"Reynolds {re_cmp:.4g} a menos de {TRANSITION_BAND:.0%} del crítico {re_c:.4g}: zona "
                                "de transición, régimen incierto")
    else:
        out["regime"] = "laminar (supuesto: sin densidad no se comprobó el Reynolds)"
        warnings.append("sin density_kg_m3 no se puede comprobar que el flujo sea laminar: se supuso laminar")
    if any(v > CLOSURE_TOL for kk, v in checks.items() if kk.endswith("_rel")):
        reasons.append("las evaluaciones independientes de la integral de flujo o del perfil no cierran")
    if gnf:
        method = ("laminar por integración numérica del balance de cantidad de movimiento: γ̇(τ) por brentq sobre "
                  f"η(γ̇)·γ̇ = τ, cuadratura de Gauss-Legendre ({GL_NODES} y {2 * GL_NODES} nodos) e integral por partes "
                  "en γ̇ con scipy.integrate.quad")
    else:
        method = ("laminar exacto: integral de τ^(k−1)·γ̇(τ) del modelo " + rh["model"] + " en forma cerrada "
                  "(k = 3 tubo, Rabinowitsch-Mooney; k = 2 placas paralelas), verificada con scipy.integrate.quad")
    out.update(method=method, warnings=warnings, assumptions=assumptions, not_included=not_included,
               evidence={"verified": not reasons, "checks": checks, "tolerance": CLOSURE_TOL, "reasons": reasons})
    return out


# ------------------------------------------------------------------ apparent viscosity and crossing of two rheologies
def _asymptotes(rh: dict) -> dict | None:
    """Low- and high-shear asymptotes of a Carreau-type model and where its plateau meets its power-law branch."""
    if rh["model"] not in GNF_MODELS or rh["lam"] <= 0.0:
        return None
    slope = -rh["m"] if rh["model"] == "cross" else rh["n"] - 1.0
    d = rh["eta0"] - rh["eta_inf"]
    out = {"zero_shear_viscosity_Pa_s": rh["eta0"], "infinite_shear_viscosity_Pa_s": rh["eta_inf"],
           "power_law_branch": {"n": slope + 1.0, "K_Pa_sn": d * rh["lam"] ** slope,
                                "note": "η − η∞ → K·γ̇^(n−1) para λγ̇ ≫ 1"},
           "plateau_meets_power_law_at_shear_rate_1_s": (d / rh["eta0"]) ** (-1.0 / slope) / rh["lam"] if slope < 0 and d > 0
           else None}
    return out


def viscosity_curve(rheology: dict, shear_rates_1_s: list | None = None, compare_rheology: dict | None = None) -> dict:
    """Apparent viscosity of one or two rheologies at given shear rates and the shear rate(s) at which they cross."""
    ra = parse_rheology(rheology)
    rb = parse_rheology(compare_rheology) if compare_rheology else None
    if not shear_rates_1_s and rb is None:
        raise ValueError("falta shear_rates_1_s (lista de rapideces de corte en 1/s) y/o compare_rheology")
    rates = []
    for i, g in enumerate(shear_rates_1_s or []):
        rates.append(_pos(f"shear_rates_1_s[{i}]", g, "rapidez de corte en 1/s"))
    table = []
    for g in rates:
        row = {"shear_rate_1_s": g, "viscosity_Pa_s": apparent_viscosity(ra, g), "shear_stress_Pa": shear_stress(ra, g)}
        if rb:
            eb = apparent_viscosity(rb, g)
            row.update(viscosity_compare_Pa_s=eb, shear_stress_compare_Pa=shear_stress(rb, g),
                       ratio_compare_to_rheology=eb / row["viscosity_Pa_s"])
        table.append(row)
    out: dict = {"rheology": {"model": ra["model"]}, "table": table}
    checks: dict = {}
    warnings: list[str] = []
    for name, r in (("rheology", ra), ("compare_rheology", rb)):
        if r and _asymptotes(r):
            out.setdefault("asymptotes", {})[name] = _asymptotes(r)
    if rb:
        out["compare_rheology"] = {"model": rb["model"]}

        def h(lg):
            g = math.exp(lg)
            return math.log(apparent_viscosity(rb, g) / apparent_viscosity(ra, g))
        grid = np.linspace(math.log(CROSSING_RANGE[0]), math.log(CROSSING_RANGE[1]), 3601)
        vals = [h(x) for x in grid]
        crossings, worst = [], 0.0
        tol = 1e-9          # |ln(η_b/η_a)| below it = the curves coincide to rounding (asymptotic merge, not a crossing)
        live = [(x, v) for x, v in zip(grid, vals) if abs(v) > tol]
        if not live:
            warnings.append("las dos reologías dan la misma viscosidad en todo el intervalo: no hay un cruce aislado")
        else:
            if len(live) < len(vals):
                merged = [math.exp(x) for x, v in zip(grid, vals) if abs(v) <= tol]
                warnings.append(f"las dos curvas coinciden (diferencia relativa < 1e-9) entre γ̇ ≈ {min(merged):.3g} y "
                                f"{max(merged):.3g} 1/s: ahí se funden asintóticamente, no se reporta como cruce")
            step = grid[1] - grid[0]
            for (x0, v0), (x1, v1) in zip(live[:-1], live[1:]):
                if v0 * v1 < 0.0 and x1 - x0 < 2.5 * step:
                    g = math.exp(brentq(h, x0, x1, xtol=1e-15, rtol=ROOT_RTOL, maxiter=500))
                    ea, eb = apparent_viscosity(ra, g), apparent_viscosity(rb, g)
                    worst = max(worst, abs(ea - eb) / ea)
                    crossings.append({"shear_rate_1_s": g, "viscosity_Pa_s": ea, "shear_stress_Pa": ea * g})
            if not crossings:
                hi = "compare_rheology" if live[0][1] > 0 else "rheology"
                warnings.append(f"las curvas de viscosidad no se cruzan entre {CROSSING_RANGE[0]:g} y "
                                f"{CROSSING_RANGE[1]:g} 1/s: {hi} es la más viscosa donde difieren (pueden "
                                "acercarse asintóticamente sin cruzarse, p. ej. Carreau con η∞ = 0 y su rama de ley de "
                                "potencia)")
        checks["crossing_viscosity_mismatch_rel"] = worst
        out.update(crossings=crossings, crossing_search_range_1_s=list(CROSSING_RANGE),
                   log_ratio_compare_to_rheology_at_range_ends=[vals[0], vals[-1]])
    reasons = ["el cruce no iguala las dos viscosidades"] if checks.get("crossing_viscosity_mismatch_rel", 0.0) > 1e-9 else []
    out.update(method="viscosidad aparente η = τ/γ̇ evaluada de la ecuación constitutiva"
                      + ("; cruces por cambio de signo de ln(η_b/η_a) en una malla logarítmica y brentq" if rb else ""),
               warnings=warnings,
               assumptions=["fluido inelástico e independiente del tiempo; cada modelo vale solo en el intervalo de γ̇ "
                            "en que se ajustó"],
               not_included=["viscosidad extensional, efectos de temperatura y presión"],
               evidence={"verified": not reasons, "checks": checks, "tolerance": 1e-9, "reasons": reasons})
    return out


# ------------------------------------------------------------------ dispatcher of the tool
def flow(volumetric_flow_m3_s: float | None = None, diameter_m: float | None = None, length_m: float | None = None,
         density_kg_m3: float | None = None, rheology: dict | None = None, profile_points: int = 21,
         geometry: str = "tube", gap_m: float | None = None, width_m: float | None = None,
         pressure_gradient_Pa_m: float | None = None, positions_m: list | None = None,
         shear_rates_1_s: list | None = None, compare_rheology: dict | None = None,
         free_stream_velocity_m_s: float | None = None) -> dict:
    """Entry point of the tool: tube (classic `pipe_flow` when the flow is given and the model has a closed form), slit,
    inverse problem (pressure gradient -> flow), Carreau-type models, viscosity curves/crossings and, with
    geometry = 'flat_plate', the Blasius boundary layer of a Newtonian fluid."""
    geo = GEOMETRIES.get(str(geometry).strip().lower())
    if geo is None:
        raise ValueError(f"geometry debe ser 'tube', 'slit' (placas paralelas) o 'flat_plate'; se recibió {geometry!r}")
    if geo == "flat_plate":
        rh = parse_rheology(rheology)
        if rh["model"] in GNF_MODELS or rh["tau_y"] > 0.0 or rh["n"] != 1.0:
            raise ValueError("geometry='flat_plate' (capa límite de Blasius) solo admite fluido newtoniano: "
                             "rheology = {'model': 'newtonian', 'viscosity_Pa_s': ...}")
        from packages.transport import boundary_layer as BL
        return BL.flat_plate(free_stream_velocity_m_s, density_kg_m3, rh["K"], length_m, width_m, positions_m)
    want_visc = bool(shear_rates_1_s) or compare_rheology is not None
    has_q, has_g = volumetric_flow_m3_s is not None, pressure_gradient_Pa_m is not None
    if has_q and has_g:
        raise ValueError("da volumetric_flow_m3_s o pressure_gradient_Pa_m, no ambos (uno se calcula del otro)")
    if not has_q and not has_g:
        if not want_visc:
            raise ValueError("falta volumetric_flow_m3_s o pressure_gradient_Pa_m (problema de flujo), o bien "
                             "shear_rates_1_s / compare_rheology (solo curva de viscosidad)")
        out = viscosity_curve(rheology, shear_rates_1_s, compare_rheology)
        return {"mode": "viscosity", "viscosity": {k: out.pop(k) for k in list(out) if k not in (
            "method", "warnings", "assumptions", "not_included", "evidence")}, **out}
    rh = parse_rheology(rheology)
    if geo == "tube":
        if has_q and rh["model"] not in GNF_MODELS:
            out = pipe_flow(volumetric_flow_m3_s, diameter_m, length_m, density_kg_m3, rheology, profile_points)
            if positions_m:
                R = float(diameter_m) / 2.0
                lam_sol = out["laminar_solution"]
                lam_sol["velocity_at_positions"] = _positions(rh, positions_m, R, lam_sol["wall_shear_stress_Pa"], "r_m",
                                                              "distancia al eje del tubo (m)")
                if not out["regime"].startswith("laminar"):
                    out["warnings"].append("velocity_at_positions es del perfil LAMINAR de referencia; el flujo no es "
                                           "laminar")
        else:
            S = _pos("diameter_m", diameter_m, "diámetro interno del tubo") / 2.0
            Q = _pos("volumetric_flow_m3_s", volumetric_flow_m3_s, "caudal volumétrico del fluido") if has_q else None
            out = _laminar_duct("tube", rh, S, Q, _pos("pressure_gradient_Pa_m", pressure_gradient_Pa_m,
                                "gradiente de presión por fricción −dp/dz") if has_g else None,
                                None if length_m is None else _pos("length_m", length_m, "longitud recta del tubo"),
                                None if density_kg_m3 is None else _pos("density_kg_m3", density_kg_m3, "densidad"),
                                profile_points, positions_m)
    else:
        S = _pos("gap_m", gap_m, "separación h entre las placas paralelas (geometry='slit')") / 2.0
        W = None if width_m is None else _pos("width_m", width_m, "ancho de las placas")
        q = None
        if has_q:
            q = _pos("volumetric_flow_m3_s", volumetric_flow_m3_s, "caudal (m³/s con width_m; m²/s por unidad de ancho "
                     "sin width_m)") / (W if W else 1.0)
        out = _laminar_duct("slit", rh, S, q, _pos("pressure_gradient_Pa_m", pressure_gradient_Pa_m,
                            "gradiente de presión por fricción −dp/dz") if has_g else None,
                            None if length_m is None else _pos("length_m", length_m, "longitud de las placas"),
                            None if density_kg_m3 is None else _pos("density_kg_m3", density_kg_m3, "densidad"),
                            profile_points, positions_m, W)
    if want_visc:
        vc = viscosity_curve(rheology, shear_rates_1_s, compare_rheology)
        out["warnings"].extend(vc["warnings"])
        out["evidence"]["checks"].update({f"viscosity_{k}": v for k, v in vc["evidence"]["checks"].items()})
        out["viscosity"] = {k: v for k, v in vc.items() if k not in ("method", "warnings", "assumptions", "not_included",
                                                                     "evidence")}
    return out
