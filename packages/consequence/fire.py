"""Thermal radiation from fires: pool fire (solid flame), jet fire (point source) and BLEVE fireball.

Pool fire, solid-flame model (SFPE Handbook, Beyler «Fire hazard calculations for large open hydrocarbon fires»):
  * burning rate, Burgess et al.: m'' = 1e-3 ΔHc/ΔH*, ΔH* = ΔHv + Cp,l (Tb − Ta) (kg/m² s, ΔH in kJ/kg);
  * flame height, Thomas (1963): H/D = 42 (m''/(ρa sqrt(g D)))^0.61 without wind,
    H/D = 55 (m''/(ρa sqrt(g D)))^0.67 (u*)^−0.21 with wind, u* = u / (g m'' D/ρv)^(1/3);
  * tilt, AGA (1974): cos θ = 1 for u* ≤ 1, 1/sqrt(u*) for u* > 1;
  * surface emissive power: Mudan & Croce (1988) E = 140 e^(−0.12 D) + 20 (1 − e^(−0.12 D)) kW/m², or
    Shokri & Beyler (1989) E = 58·10^(−0.00823 D) kW/m²;
  * view factor of a cylinder to a target element: Mudan (1984) analytic for the vertical cylinder; for any tilt, a
    numerical surface integral (checked against the analytic one in the tests);
  * atmospheric transmissivity: TNO Yellow Book τ = 2.02 (pw·x)^−0.09 (pw water partial pressure in Pa, x in m), only
    when the user gives humidity; otherwise τ = 1 (conservative) and it is said.
Pool fire, TNO Yellow Book (CPR 14E, 2005, §6.5 and worked example §6.6.3): m'' = m''∞(1 − e^(−kβ D)); Thomas wind
length with u_c = (g m'' D/ρ_air)^(1/3); tilt tanΘ/cosΘ = 0.666 Fr^0.333 Re^0.117 (Fr = u²/(gD), Re = u D/ν);
SEPmax = Fs m'' ΔHc/(1 + 4L/D), SEPact = SEPmax (1 − ς) + SEPsoot ς (ς = 0.8, SEPsoot = 20 kW/m², TNO); view factor of
the tilted cylinder (Mudan, YB app. 6.1-7; here the numerical integral, which reproduces the printed values).
Heskestad flame height (SFPE, NUREG-1805): H = 0.235 Q^(2/5) − 1.02 D (Q in kW, H in m).
Fireball (BLEVE), TNO Yellow Book §6.5.6/§6.6.5: r = 3.24 M^0.325, t = 0.852 M^0.26, centre at H = 2r,
Fs = 0.00325 Psv^0.32 (Psv in N/m², Roberts/Hymes), ΔH = ΔHc − ΔHv − Cp ΔT (ΔT = 2000 K − Ta, eq. 6.94),
SEP = ΔH·M·Fs/(4π r² t), view factor (r/X)² for a target facing the centre.
Jet fire: Chamberlain solid flame (YB §6.5.4, `jet_fire_chamberlain`) or point source q = τ·Frad·Qm·ΔHc/(4π r²)
(API 521 / Hajek-Ludwig; same form as NISTIR 6546 for pools), Frad from the user.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from packages.consequence.props import G, water_vapour_pressure


# --------------------------------------------------------------------------- pool fire pieces
def burning_rate_burgess(Hc_J_kg: float, Hvap_J_kg: float, Cp_l_J_kgK: float | None, Tb_K: float, Ta_K: float) -> dict:
    sens = Cp_l_J_kgK * (Tb_K - Ta_K) if (Cp_l_J_kgK and Tb_K > Ta_K) else 0.0
    dh_star = Hvap_J_kg + sens
    return {"m_kg_m2s": 1e-3 * Hc_J_kg / dh_star, "dH_star_J_kg": dh_star,
            "fuente": "Burgess et al., en SFPE Handbook (Beyler) y Crowl & Louvar: m'' = 1e-3·ΔHc/ΔH*",
            "nota": "líquido que hierve bajo la temperatura ambiente: ΔH* = ΔHv" if Tb_K <= Ta_K else ""}


def flame_geometry_thomas(D: float, m: float, rho_air: float, rho_vapour: float, wind_speed: float | None) -> dict:
    mstar = m / (rho_air * math.sqrt(G * D))
    if not wind_speed:
        H = D * 42.0 * mstar ** 0.61
        return {"altura_llama_m": H, "inclinacion_grados": 0.0, "u_estrella": 0.0,
                "fuente": "Thomas (1963), H/D = 42 (m''/(ρa√(gD)))^0.61, sin viento"}
    uc = (G * m * D / rho_vapour) ** (1.0 / 3.0)
    us = wind_speed / uc
    H = D * 55.0 * mstar ** 0.67 * max(us, 1e-9) ** -0.21 if us >= 1.0 else D * 55.0 * mstar ** 0.67
    cos_t = 1.0 if us <= 1.0 else 1.0 / math.sqrt(us)
    return {"altura_llama_m": H, "inclinacion_grados": math.degrees(math.acos(cos_t)), "u_estrella": us,
            "fuente": "Thomas (1963) con viento, H/D = 55 (m'')^0.67 (u*)^−0.21 (u* < 1 → u* = 1); inclinación AGA "
                      "(1974) cos θ = 1/√u*"}


def flame_height_heskestad(D: float, m: float, Hc_J_kg: float) -> dict:
    Q = m * Hc_J_kg / 1000.0 * math.pi * D * D / 4.0
    return {"altura_llama_m": 0.235 * Q ** 0.4 - 1.02 * D, "inclinacion_grados": 0.0, "Q_kW": Q,
            "fuente": "Heskestad, H = 0.235 Q^(2/5) − 1.02 D (SFPE; NUREG-1805 cap. 5), sin viento"}


def burning_rate_infinite(m_inf: float, k_beta: float, D: float) -> dict:
    return {"m_kg_m2s": m_inf * (1.0 - math.exp(-k_beta * D)),
            "fuente": "m'' = m''∞(1 − e^(−kβ D)) (Babrauskas; TNO YB ec. 6.66), m''∞ y kβ del usuario"}


def tno_tilt(D: float, wind_speed: float, nu_air: float) -> dict:
    Fr = wind_speed ** 2 / (G * D)
    Re = wind_speed * D / nu_air
    c = 0.666 * Fr ** 0.333 * Re ** 0.117
    th = math.asin((math.sqrt(4 * c * c + 1) - 1) / (2 * c))
    return {"Fr": Fr, "Re": Re, "tan_cos": c, "inclinacion_grados": math.degrees(th)}


def pool_fire_tno(D: float, m: float, Hc_J_kg: float, Fs: float, rho_air: float, wind_speed: float, nu_air: float,
                  soot_fraction: float = 0.8, sep_soot: float = 20.0) -> dict:
    """TNO Yellow Book pool-fire chain up to the flame (geometry and SEP in kW/m²)."""
    uc = (G * m * D / rho_air) ** (1.0 / 3.0)
    us = wind_speed / uc
    L = D * 55.0 * (m / (rho_air * math.sqrt(G * D))) ** 0.67 * max(us, 1.0) ** -0.21
    tilt = tno_tilt(D, wind_speed, nu_air) if wind_speed > 0 else {"inclinacion_grados": 0.0}
    sep_max = Fs * m * Hc_J_kg / (1.0 + 4.0 * L / D) / 1000.0
    sep = sep_max * (1.0 - soot_fraction) + sep_soot * soot_fraction
    return {"u_c_m_s": uc, "u_estrella": us, "longitud_llama_m": L, "L_D": L / D, **tilt, "SEP_max_kW_m2": sep_max,
            "SEP_kW_m2": sep, "fuente": "TNO Yellow Book CPR 14E (2005) §6.5.5, ec. 6.12-6.20 y 6.71"}


def sep_mudan_croce(D: float) -> float:
    return 140.0 * math.exp(-0.12 * D) + 20.0 * (1.0 - math.exp(-0.12 * D))


def sep_shokri_beyler(D: float) -> float:
    return 58.0 * 10 ** (-0.00823 * D)


def view_factor_vertical_cylinder(D: float, H: float, dist_center: float) -> dict:
    """Mudan (1984) analytic view factors from a vertical cylinder (radius D/2, height H, base at the target's level)
    to a vertical and a horizontal target element at distance `dist_center` from the axis (SFPE Handbook)."""
    S = 2.0 * dist_center / D
    h = 2.0 * H / D
    if S <= 1.0:
        raise ValueError("el blanco está dentro de la llama")
    A = (h * h + S * S + 1.0) / (2.0 * S)
    B = (1.0 + S * S) / (2.0 * S)
    t_a = math.atan(math.sqrt((A + 1.0) * (S - 1.0) / ((A - 1.0) * (S + 1.0))))
    t_b = math.atan(math.sqrt((B + 1.0) * (S - 1.0) / ((B - 1.0) * (S + 1.0))))
    fv = (math.atan(h / math.sqrt(S * S - 1.0)) / (math.pi * S)
          - h / (math.pi * S) * math.atan(math.sqrt((S - 1.0) / (S + 1.0)))
          + A * h / (math.pi * S * math.sqrt(A * A - 1.0)) * t_a)
    fh = (B - 1.0 / S) / (math.pi * math.sqrt(B * B - 1.0)) * t_b - (A - 1.0 / S) / (math.pi * math.sqrt(A * A - 1.0)) * t_a
    return {"F_vertical": fv, "F_horizontal": fh, "F_max": math.hypot(fv, fh)}


def view_factor_cylinder_numeric(D: float, L: float, tilt_deg: float, target_x: float, target_z: float = 0.0,
                                 n_phi: int = 360, n_s: int = 200) -> dict:
    """View factor from a (sheared) tilted cylinder — circular horizontal sections of diameter D whose centres follow
    the axis tilted `tilt_deg` from vertical towards +x, flame length L along the axis — to a target element at
    (target_x, 0, target_z). Convex emitter: every patch whose normal faces the target is seen. Returns F for a
    vertical element facing the fire, a horizontal element facing up, and the maximum (element normal along the
    flux vector)."""
    R = D / 2.0
    th = math.radians(tilt_deg)
    ax = np.array([math.sin(th), 0.0, math.cos(th)])
    phi = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
    s = (np.arange(n_s) + 0.5) / n_s
    PH, SS = np.meshgrid(phi, s, indexing="ij")
    P = (SS[..., None] * L * ax + R * np.stack([np.cos(PH), np.sin(PH), np.zeros_like(PH)], -1))
    nrm = np.stack([np.cos(PH) * math.cos(th), np.sin(PH) * math.cos(th), -np.cos(PH) * math.sin(th)], -1) * R * L
    dA = np.linalg.norm(nrm, axis=-1) * (2 * math.pi / n_phi) * (1.0 / n_s)
    n1 = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    # top cap (horizontal disc), polar grid
    nr = 60
    rr = (np.arange(nr) + 0.5) / nr * R
    RR, PP = np.meshgrid(rr, phi, indexing="ij")
    top_c = L * ax
    Pt = top_c + np.stack([RR * np.cos(PP), RR * np.sin(PP), np.zeros_like(RR)], -1)
    dAt = RR * (R / nr) * (2 * math.pi / n_phi)
    n1t = np.broadcast_to(np.array([0.0, 0.0, 1.0]), Pt.shape)
    Pall = np.concatenate([P.reshape(-1, 3), Pt.reshape(-1, 3)])
    nall = np.concatenate([n1.reshape(-1, 3), n1t.reshape(-1, 3)])
    dAall = np.concatenate([dA.ravel(), dAt.ravel()])
    T = np.array([target_x, 0.0, target_z])
    v = T - Pall
    r = np.linalg.norm(v, axis=1)
    cos1 = np.einsum("ij,ij->i", nall, v) / r
    vis = cos1 > 0
    w = np.where(vis, cos1 * dAall / (math.pi * r * r), 0.0)
    u = -v / r[:, None]                       # unit vectors from target to the patches
    vec = (w[:, None] * u).sum(0)

    def F(n2):
        c2 = u @ n2
        return float((w * np.clip(c2, 0, None)).sum())
    fv = F(np.array([-1.0, 0.0, 0.0]) if target_x > 0 else np.array([1.0, 0.0, 0.0]))
    fh = F(np.array([0.0, 0.0, 1.0]))
    nmax = vec / np.linalg.norm(vec)
    return {"F_vertical": fv, "F_horizontal": fh, "F_max": F(nmax)}


def transmissivity_tno(rh: float | None, T_air_K: float, path_m: float, tau_user: float | None = None) -> tuple[float, str]:
    """Bagster (1989) as given in the TNO Yellow Book (eq. 6.29), stated valid for 1e4 < pw·x < 1e5 N/m."""
    if tau_user is not None:
        return tau_user, "τ del usuario"
    if rh is None:
        return 1.0, "τ = 1 (sin humedad del usuario: conservador)"
    pw = rh * water_vapour_pressure(T_air_K)
    px = pw * max(path_m, 1e-3)
    tau = min(1.0, 2.02 * px ** -0.09)
    flag = "" if 1e4 <= px <= 1e5 else f" (pw·x = {px:.3g} N/m fuera de 1e4–1e5: extrapolado)"
    return tau, f"TNO Yellow Book ec. 6.29 (Bagster 1989) τ = 2.02 (pw·x)^−0.09, pw = {pw:.0f} Pa{flag}"


def pool_fire(D: float, Hc_J_kg: float, Hvap_J_kg: float, Tb_K: float, T_air_K: float, rho_air: float,
              rho_vapour: float, Cp_l_J_kgK: float | None = None, burning_rate: float | None = None,
              wind_speed: float | None = None, rh: float | None = None, sep_method: str = "both",
              target_height: float = 0.0, model: str = "sfpe", burning_rate_inf: float | None = None,
              k_beta: float | None = None, flame_height: str = "thomas", radiant_fraction: float | None = None,
              nu_air: float | None = None, tau_user: float | None = None, soot_fraction: float = 0.8,
              sep_soot: float = 20.0) -> dict:
    """Solid-flame pool fire, model sfpe (Burgess/Thomas/AGA + Mudan-Croce or Shokri-Beyler SEP) or tno (Yellow Book).
    Returns geometry, SEP and `_flux(x)` → (q kW/m², F, τ) for a target downwind at distance x from the pool centre."""
    if burning_rate:
        br = {"m_kg_m2s": burning_rate, "fuente": "dato del usuario"}
    elif burning_rate_inf and k_beta:
        br = burning_rate_infinite(burning_rate_inf, k_beta, D)
    else:
        br = burning_rate_burgess(Hc_J_kg, Hvap_J_kg, Cp_l_J_kgK, Tb_K, T_air_K)
    m = br["m_kg_m2s"]
    if model == "tno":
        if radiant_fraction is None:
            raise ValueError("el modelo TNO pide la fracción radiada Fs (tabla 6.6 del Yellow Book o dato del usuario)")
        geo = pool_fire_tno(D, m, Hc_J_kg, radiant_fraction, rho_air, wind_speed or 0.0, nu_air or 1.5e-5,
                            soot_fraction, sep_soot)
        H, tilt, E = geo["longitud_llama_m"], geo["inclinacion_grados"], geo["SEP_kW_m2"]
        seps, key = {"tno_hollin": E}, "tno_hollin"
    else:
        geo = (flame_height_heskestad(D, m, Hc_J_kg) if flame_height == "heskestad"
               else flame_geometry_thomas(D, m, rho_air, rho_vapour, wind_speed))
        seps = {"mudan_croce": sep_mudan_croce(D), "shokri_beyler": sep_shokri_beyler(D)}
        key = max(seps, key=seps.get) if sep_method == "both" else sep_method
        E = seps[key]
        H, tilt = geo["altura_llama_m"], geo["inclinacion_grados"]
    if H <= 0:
        raise ValueError("altura de llama ≤ 0: fuera del rango de la correlación")

    def flux(x: float) -> tuple[float, float, float]:
        if tilt == 0.0 and target_height == 0.0:
            F = view_factor_vertical_cylinder(D, H, x)["F_max"]
        else:
            F = view_factor_cylinder_numeric(D, H, tilt, x, target_height)["F_max"]
        tau, _ = transmissivity_tno(rh, T_air_K, x - D / 2.0, tau_user)
        return E * F * tau, F, tau

    x_under = D / 2.0 + H * math.sin(math.radians(tilt))
    return {"modelo_charco": model, "tasa_quemado": br, "geometria": geo, "SEP_kW_m2": seps, "SEP_usada": key,
            "E_kW_m2": E, "_flux": flux, "x_min_m": D / 2.0 * 1.05, "x_bajo_llama_m": x_under}


# --------------------------------------------------------------------------- fireball (BLEVE)
def fireball_tno(M_kg: float, Hc_J_kg: float, P_burst_Pa: float | None = None, f_rad: float | None = None,
                 Hvap_J_kg: float | None = None, Cp_l_J_kgK: float | None = None, T_air_K: float | None = None,
                 T_flame_K: float = 2000.0) -> dict:
    r = 3.24 * M_kg ** 0.325
    t = 0.852 * M_kg ** 0.26
    if f_rad is None:
        if P_burst_Pa is None:
            raise ValueError("da la presión de apertura de la PSV/ruptura (Pa) o la fracción radiada")
        f_rad = 0.00325 * P_burst_Pa ** 0.32
        fsrc = "TNO YB ec. 6.21 (Roberts/Hymes): Fs = 0.00325·Psv^0.32, Psv en N/m²"
    else:
        fsrc = "dato del usuario"
    if Hvap_J_kg is not None and Cp_l_J_kgK is not None and T_air_K is not None:
        dH = Hc_J_kg - Hvap_J_kg - Cp_l_J_kgK * (T_flame_K - T_air_K)
        hsrc = f"ΔH = ΔHc − ΔHv − Cp·ΔT con ΔT = {T_flame_K:g} K − Ta (TNO YB ec. 6.94)"
    else:
        dH = Hc_J_kg
        hsrc = "ΔHc sin restar ΔHv ni Cp·ΔT (conservador; da ΔHv y Cp para la corrección TNO)"
    E = dH * M_kg * f_rad / (4.0 * math.pi * r * r * t) / 1000.0
    return {"radio_m": r, "diametro_m": 2 * r, "duracion_s": t, "altura_centro_m": 2 * r, "fraccion_radiada": f_rad,
            "fuente_frad": fsrc, "dH_J_kg": dH, "fuente_dH": hsrc, "E_kW_m2": E,
            "fuente": "TNO Yellow Book CPR 14E (2005) §6.5.6: r = 3.24 M^0.325, t = 0.852 M^0.26, H = 2r, "
                      "SEP = ΔH·M·Fs/(4π r² t)"}


def fireball_flux(fb: dict, ground_distance: float, rh: float | None, T_air_K: float,
                  tau_user: float | None = None) -> tuple[float, float, float]:
    R = fb["radio_m"]
    L = math.hypot(ground_distance, fb["altura_centro_m"])
    F = (R / L) ** 2 if L > R else 1.0
    tau, _ = transmissivity_tno(rh, T_air_K, L - R, tau_user)
    return fb["E_kW_m2"] * F * tau, F, tau


# --------------------------------------------------------------------------- jet fire (point source)
def jet_fire_point_source(Qm_kg_s: float, Hc_J_kg: float, f_rad: float, r: float, tau: float = 1.0) -> float:
    return tau * f_rad * Qm_kg_s * Hc_J_kg / (4.0 * math.pi * r * r) / 1000.0


def distance_to_flux(flux_fn, q_target: float, x_lo: float, x_hi: float = 20_000.0) -> float | None:
    """Farthest distance where q(x) ≥ q_target (last crossing on a log grid, then Brent). None if never reached."""
    f = lambda x: flux_fn(x) - q_target  # noqa: E731
    xs = np.geomspace(x_lo, x_hi, 160)
    vals = np.array([f(x) for x in xs])
    above = np.nonzero(vals >= 0)[0]
    if above.size == 0:
        return None
    k = int(above[-1])
    if k == len(xs) - 1:
        return float("inf")
    return float(brentq(f, xs[k], xs[k + 1], xtol=1e-4))


# --------------------------------------------------------------------------- jet fire (Chamberlain, TNO YB §6.5.4)
def jet_fire_chamberlain(m_kg_s: float, hole_diameter_m: float, P_init_Pa: float, T_s_K: float, W_g: float,
                         Cp_J_kgK: float, Hc_J_kg: float, wind_speed: float, theta_j_deg: float,
                         P_air_Pa: float = 101325.0, W_air: float = 0.02896) -> dict:
    """Chamberlain (1987) solid-flame jet fire, as in the TNO Yellow Book CPR 14E (2005) §6.5.4 and its worked example
    §6.6.2: frustum flame from a choked release of a hydrocarbon gas, flame lifted off by b along the hole axis.
    Eq. 6.54 (W1) is read as Ds(13.5e^(−6Rw)+1.5)(1 − (1 − (ρair/ρj)^½/15)·exp(−70·Ri(Ds)^(C′Rw))): the only form
    that gives the printed 0.71691 m. The flame-tilt branch implemented is the one for Rw ≤ 0.05 (the other is not
    legible in the copy read). Returns intermediates and `_flux(X)` → (q kW/m², F, τ) for a target downwind at
    horizontal distance X from the release point, at the release height."""
    Rc, Gg = 8.31451, G
    W = W_g / (15.816 * W_g + 0.0395)
    gam = 1.0 / (1.0 - Rc / (Cp_J_kgK * W_g))
    Pc = P_init_Pa * (2.0 / (gam + 1.0)) ** (gam / (gam - 1.0))
    if Pc <= P_air_Pa:
        raise ValueError("la fuga no está ahogada: el modelo de Chamberlain del YB es para chorro sónico")
    Tj = T_s_K * (P_air_Pa / P_init_Pa) ** ((gam - 1.0) / gam)
    Mj = math.sqrt(((gam + 1.0) * (Pc / P_air_Pa) ** ((gam - 1.0) / gam) - 2.0) / (gam - 1.0))
    uj = Mj * math.sqrt(gam * Rc * Tj / W_g)
    Rw = wind_speed / uj
    if Rw > 0.05:
        raise ValueError(f"Rw = {Rw:.3f} > 0.05: rama de inclinación del YB no legible en la copia leída (pendiente)")
    rho_air = P_air_Pa * W_air / (Rc * T_s_K)
    rho_j = Pc * W_g / (Rc * Tj)
    Ds = hole_diameter_m * math.sqrt(rho_j / rho_air)
    Ca = 0.024 * (Gg * Ds / uj ** 2) ** (1.0 / 3.0)
    Cb, Cc = 0.2, (2.85 / W) ** (2.0 / 3.0)
    Y = brentq(lambda y: Ca * y ** (5.0 / 3.0) + Cb * y ** (2.0 / 3.0) - Cc, 1e-6, 1e6, xtol=1e-12)
    Lb0 = Y * Ds
    Lb = Lb0 * (0.51 * math.exp(-0.4 * wind_speed) + 0.49) * (1.0 - 6.07e-3 * (theta_j_deg - 90.0))
    Ri_L = (Gg / (Ds ** 2 * uj ** 2)) ** (1.0 / 3.0) * Lb0
    alpha = (theta_j_deg - 90.0) * (1.0 - math.exp(-25.6 * Rw)) + 8000.0 * Rw / Ri_L
    K = 0.185 * math.exp(-20.0 * Rw) + 0.015
    a_r = math.radians(alpha)
    b = Lb * math.sin(math.radians(K * alpha)) / math.sin(a_r)
    Rl = math.sqrt(Lb ** 2 - b ** 2 * math.sin(a_r) ** 2) - b * math.cos(a_r)
    ratio = Tj * W_air / (T_s_K * W_g)                   # ρair/ρj as printed in eq. 6.51
    Ri_D = (Gg / (Ds ** 2 * uj ** 2)) ** (1.0 / 3.0) * Ds
    Cpr = 1000.0 * math.exp(-100.0 * Rw) + 0.8
    W1 = Ds * (13.5 * math.exp(-6.0 * Rw) + 1.5) * (1.0 - (1.0 - math.sqrt(ratio) / 15.0)
                                                     * math.exp(-70.0 * Ri_D ** (Cpr * Rw)))
    W2 = Lb * (0.18 * math.exp(-1.5 * Rw) + 0.31) * (1.0 - 0.47 * math.exp(-25.0 * Rw))
    A_fr = math.pi / 4 * (W1 ** 2 + W2 ** 2) + math.pi / 2 * (W1 + W2) * math.sqrt(Rl ** 2 + ((W2 - W1) / 2) ** 2)
    A_cyl = math.pi / 2 * ((W1 + W2) / 2) ** 2 + math.pi * Rl * (W1 + W2) / 2
    Q = m_kg_s * Hc_J_kg
    Fs = 0.21 * math.exp(-0.00323 * uj) + 0.11
    E = Fs * Q / A_cyl / 1000.0
    R = (W1 + W2) / 4.0
    tj = math.radians(theta_j_deg)

    def geom(X: float) -> tuple[float, float, float]:
        dx = X - b * math.cos(tj)
        Xp = math.hypot(b * math.sin(tj), dx)
        thp = 90.0 - theta_j_deg + alpha - math.degrees(math.atan(b * math.sin(tj) / dx))
        return Xp, thp, Xp - R

    def flux(X: float, rh: float | None = None, T_air: float | None = None, tau_user: float | None = None):
        Xp, thp, x = geom(X)
        if Xp <= R:
            raise ValueError("blanco dentro de la llama")
        F = view_factor_cylinder_numeric(2 * R, Rl, thp, Xp)["F_max"]
        tau, _ = transmissivity_tno(rh, T_air or T_s_K, x, tau_user)
        return E * F * tau, F, tau

    return {"W": W, "gamma": gam, "T_j_K": Tj, "P_c_Pa": Pc, "M_j": Mj, "u_j_m_s": uj, "R_w": Rw,
            "rho_air": rho_air, "rho_j": rho_j, "D_s_m": Ds, "Y": Y, "L_b0_m": Lb0, "L_b_m": Lb, "Ri_Lb0": Ri_L,
            "alpha_grados": alpha, "K": K, "b_m": b, "R_l_m": Rl, "W1_m": W1, "W2_m": W2, "A_tronco_m2": A_fr,
            "A_cilindro_m2": A_cyl, "Q_W": Q, "F_s": Fs, "E_kW_m2": E, "radio_equivalente_m": R,
            "fuente": "Chamberlain (1987) según TNO Yellow Book CPR 14E (2005) §6.5.4, ejemplo §6.6.2; SEP sobre el "
                      "área del cilindro equivalente (como el ejemplo)", "_geom": geom, "_flux": flux}
