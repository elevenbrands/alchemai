"""Catalytic fixed bed with pellet transport, and gas-liquid absorption with reaction (film theory).

What it answers:
  * `packed_bed_reactor`: a 1-D heterogeneous plug-flow fixed bed of porous pellets in which the rate at every axial
    position is η(local)·r, with η from the pellet boundary-value problem of `packages.transport.pellet` solved at the
    local bulk concentration and temperature (internal diffusion, optional external film and intraparticle heat). Gas
    phase, ideal gas, Ergun pressure drop, isothermal / adiabatic / wall-cooled. Profiles, conversion, catalyst
    volume and mass, length for a target conversion, hot spot, and where the bed is diffusion-limited.
  * `gas_liquid_reaction`: absorption of A from a gas into a liquid where it reacts (first order, second order A + νB,
    or instantaneous), by film theory: Hatta number, enhancement factor (numerical film solution, closed forms
    Ha/tanh Ha and E_i = 1 + q, van Krevelen-Hoftijzer approximation), regime, absorption flux with optional gas-film
    resistance; optionally a contactor with well-mixed liquid (bubble column / stirred tank) from the user's kL·a,
    holdup, volume and liquid flow.

Bed model (per tube, z = axial position, A = cross-section, ε = bed voidage, Nt = Nt0 + δ (NA0 − NA)):
    dNA/dz = −(1 − ε) ℜ(C, T) A               ℜ = η·r per pellet volume, C = (NA/Nt) P/(R T)
    Nt cp dT/dz = (−ΔH)(1 − ε) ℜ A + U π D (Tc − T)        (adiabatic: U = 0; isothermal: dT/dz = 0)
    dP/dz = −G/(ρ dp) (1 − ε)/ε³ [150 (1 − ε) μ/dp + 1.75 G]   (Ergun, only when the viscosity is given)
Numerics: scipy `solve_ivp` (RK45) with the pellet BVP warm-started from the previous axial position; first-order
isothermal pellets use the exact η. Evidence: step independence (rtol 1e-6 vs 1e-8), energy closure (adiabatic: the
exact invariant of constant cp; wall-cooled: quadrature of the profile), the warm-started rate against a full pellet
solve at the profile points, every pellet solve verified, and the closed-form conversion when one exists.

Film model (y = x/δ, a = C_A/C_Ai, b = C_B/C_Bb, Ha² = k2 C_Bb D_A/kL², q = D_B C_Bb/(ν D_A C_Ai)):
    a'' = Ha² a b,  b'' = Ha² a b / q;  a(0) = 1, a(1) = C_Ab/C_Ai, b'(0) = 0, b(1) = 1      (first order: b ≡ 1)
solved by `solve_bvp` with continuation in Ha. E = N_A / (kL (C_Ai − C_Ab)).

Sources read 2026-11-14:
  Rawlings & Ekerdt, ch. 7 slides (https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/ch7/slides-masswrxn.pdf):
  fixed-bed design equation and Examples 7.3-7.5.
  P. A. Ramachandran, «Chapter 13 Gas-Liquid Systems», course notes Washington University
  (https://classes.engineering.wustl.edu/eece503/Lecture_Notes/Gas%20Liquid%20Reactions.pdf): film-theory fluxes
  (Eq. A, B), backmixed liquid (Eq. C), E = Ha/tanh Ha, van Krevelen-Hoftijzer with Ha·√(b_i), E_i = 1 + q.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq

from packages.transport import pellet as PL

R_GAS = PL.R_GAS
ODE_RTOL, ODE_RTOL_FINE = 1e-6, 1e-8
STEP_TOL = 1e-4            # relative change of the outlet conversion (and T, P) between the two tolerances
ENERGY_TOL = 1e-3          # energy closure, relative to the heat released
LOCAL_RATE_TOL = 1e-4      # warm-started pellet rate vs a full pellet solve
DIFF_LIMITED_ETA = 0.95    # η_internal below this marks the position as limited by pore diffusion
FILM_TOL, FILM_TOL_FINE = 1e-6, 1e-8
FILM_MESH_TOL = 1e-4
FILM_BALANCE_TOL = 1e-4
FILM_CHECK_TOL = 1e-4      # numerical film vs the closed form (first order)
N_QUAD = 4001
_GL_X, _GL_W = np.polynomial.legendre.leggauss(5)


def _req(name: str, val, what: str, allow_zero: bool = False) -> float:
    if val is None:
        raise ValueError(f"falta {name}: {what}")
    v = float(val)
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} debe ser {'≥' if allow_zero else '>'} 0 ({what}); se recibió {v}")
    return v


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


# ---------------------------------------------------------------------------------------------------------------------
# fixed bed
# ---------------------------------------------------------------------------------------------------------------------
class _LocalRate:
    """η·r at bulk (C, T) for one pellet, warm-started from the previous call (the bed calls it at nearby states)."""

    def __init__(self, p: PL.Pellet, branch: str):
        self.p, self.branch, self.guess = p, branch, None
        k = p.kin
        self.analytic = p.mode == "iso" and k["type"] == "power_law" and k["n"] == 1.0
        self.full_solves = 0

    def __call__(self, C: float, T: float) -> dict:
        p = self.p
        if C <= 0.0:
            return {"rate": 0.0, "eta_overall": None, "eta_internal": None}
        if self.analytic:
            k = float(PL.rate_constant(p.kin, T))
            phi = p.L * math.sqrt(k / p.De)
            eta_i = PL.eta_first_order(p.geometry, phi)
            eta_o = eta_i if p.kc is None else PL.eta_first_order_with_film(p.geometry, phi, p.kc * p.L / p.De)
            return {"rate": eta_o * k * C, "eta_overall": eta_o, "eta_internal": eta_i}
        if p.dead_core:
            sm = PL.solve_dead_core(p, C, T)["selected"]
            return {"rate": sm["observed_rate_mol_m3_s"], "eta_overall": sm["eta_overall"],
                    "eta_internal": sm["eta_internal"]}
        sol = p.solve_from(C, T, *self.guess) if self.guess is not None else None
        if sol is None:
            res = PL.solve_pellet(p, C, T, self.branch)
            self.full_solves += 1
            if not res["converged"]:
                raise RuntimeError("el pellet no convergió en una posición del lecho: "
                                   + "; ".join(res["evidence"]["reasons"]))
            sol = res["solution"]
        self.guess = (sol.x, sol.y)
        sm = p.summary(sol, C, T)
        return {"rate": sm["observed_rate_mol_m3_s"], "eta_overall": sm["eta_overall"], "eta_internal": sm["eta_internal"]}


def _full_pellet(p: PL.Pellet, C: float, T: float, branch: str) -> dict:
    """Independent full pellet solve (continuation + checks) at one axial position."""
    if C <= 0.0:
        return {"rate": 0.0, "eta_overall": None, "eta_internal": None, "verified": True, "n_states": 0}
    res = PL.solve_pellet(p, C, T, branch)
    if not res["converged"]:
        return {"rate": None, "eta_overall": None, "eta_internal": None, "verified": False, "n_states": 0,
                "reasons": res["evidence"]["reasons"]}
    sm = res["selected"]
    return {"rate": sm["observed_rate_mol_m3_s"], "eta_overall": sm["eta_overall"], "eta_internal": sm["eta_internal"],
            "verified": res["evidence"]["verified"], "reasons": res["evidence"]["reasons"],
            "n_states": len(res["steady_states"])}


def packed_bed_reactor(geometry: str, characteristic_length_m: float, kinetics: dict,
                       effective_diffusivity_m2_s: float | None = None, pore: dict | None = None,
                       pellet_density_kg_m3: float | None = None, heat_of_reaction_J_mol: float | None = None,
                       effective_conductivity_W_mK: float | None = None,
                       mass_transfer_coefficient_m_s: float | None = None,
                       heat_transfer_coefficient_W_m2K: float | None = None,
                       bed_voidage: float | None = None, tube_diameter_m: float | None = None, number_of_tubes: int = 1,
                       bed_length_m: float | None = None, target_conversion: float | None = None,
                       feed_molar_flow_A_mol_s: float | None = None, feed_total_molar_flow_mol_s: float | None = None,
                       moles_change_per_mol_A: float | None = None, inlet_temperature_K: float | None = None,
                       inlet_pressure_Pa: float | None = None, energy_mode: str | None = None,
                       gas_heat_capacity_J_molK: float | None = None,
                       wall_heat_transfer_coefficient_W_m2K: float | None = None,
                       coolant_temperature_K: float | None = None, gas_viscosity_Pa_s: float | None = None,
                       feed_molar_mass_kg_mol: float | None = None, ergun_particle_diameter_m: float | None = None,
                       branch: str = "low", n_profile: int = 11) -> dict:
    """1-D heterogeneous fixed bed with local η (see the module docstring)."""
    warnings: list[str] = []
    assumptions = ["flujo pistón de gas ideal, sin dispersión axial ni gradientes radiales en el lecho; una reacción con "
                   "reactivo limitante A",
                   "en cada posición el pellet está en estado estacionario a la concentración y temperatura locales del gas "
                   "(η local del problema de contorno del pellet)"]
    not_included = ["dispersión axial y modelo 2-D (gradiente radial de temperatura en tubos enfriados)",
                    "varias reacciones, reacción reversible, desactivación del catalizador",
                    "lechos de goteo (trickle) y lechos con líquido: solo fase gas",
                    "dependencia de kc y h con la posición (se usan los valores constantes dados)"]
    if energy_mode not in ("isothermal", "adiabatic", "wall_cooled"):
        raise ValueError("falta energy_mode: 'isothermal', 'adiabatic' o 'wall_cooled' (tubo enfriado por la pared)")
    eps = _req("bed_voidage", bed_voidage, "fracción de huecos del lecho, para la fracción de catalizador y Ergun")
    if eps >= 1.0:
        raise ValueError("bed_voidage debe estar entre 0 y 1")
    D = _req("tube_diameter_m", tube_diameter_m, "diámetro interno del tubo (sección de flujo y área de pared)")
    ntub = int(number_of_tubes)
    if ntub < 1:
        raise ValueError("number_of_tubes debe ser ≥ 1")
    NA0 = _req("feed_molar_flow_A_mol_s", feed_molar_flow_A_mol_s, "flujo molar de A a la entrada (todos los tubos)")
    Nt0 = _req("feed_total_molar_flow_mol_s", feed_total_molar_flow_mol_s, "flujo molar total a la entrada")
    if NA0 > Nt0 * (1 + 1e-12):
        raise ValueError("feed_molar_flow_A_mol_s no puede superar feed_total_molar_flow_mol_s")
    if moles_change_per_mol_A is None:
        raise ValueError("falta moles_change_per_mol_A: cambio de moles totales por mol de A convertido (Σν/|ν_A|; 0 para "
                         "A → B), necesario para la concentración del gas")
    delta = float(moles_change_per_mol_A)
    T0 = _req("inlet_temperature_K", inlet_temperature_K, "temperatura del gas a la entrada")
    P0 = _req("inlet_pressure_Pa", inlet_pressure_Pa, "presión a la entrada")
    if (bed_length_m is None) == (target_conversion is None):
        raise ValueError("da bed_length_m (lecho dado: calcula la conversión) o target_conversion (calcula la longitud), "
                         "uno de los dos")
    Xt = None
    if target_conversion is not None:
        Xt = float(target_conversion)
        if not 0.0 < Xt < 1.0:
            raise ValueError("target_conversion debe estar entre 0 y 1")
    Lbed = None if bed_length_m is None else _req("bed_length_m", bed_length_m, "longitud del lecho")
    if branch not in ("low", "high"):
        raise ValueError("branch debe ser 'low' o 'high' (rama del pellet cuando hay varios estados estacionarios)")

    kin = PL.normalize_kinetics(kinetics, pellet_density_kg_m3)
    L = _req("characteristic_length_m", characteristic_length_m, "semiespesor o radio del pellet")
    if effective_diffusivity_m2_s is not None and pore is not None:
        raise ValueError("da effective_diffusivity_m2_s o pore (para estimarla), no ambos")
    if effective_diffusivity_m2_s is not None:
        De = _req("effective_diffusivity_m2_s", effective_diffusivity_m2_s, "difusividad efectiva en el pellet")
        de_info = {"source": "dato del usuario"}
    elif pore is not None:
        est = PL.effective_diffusivity_from_pores(pore, T0)
        De = est["effective_diffusivity_m2_s"]
        de_info = {"source": "ESTIMADA de datos de poro a la temperatura de entrada", **est}
        warnings.append("la difusividad efectiva es una estimación de datos de poro (a la T de entrada), no una medida")
    else:
        raise ValueError("falta effective_diffusivity_m2_s o pore {porosity, tortuosity, mean_pore_radius_m, "
                         "molecular_diffusivity_m2_s, molar_mass_kg_mol}")
    dH = heat_of_reaction_J_mol
    lam, kc, h = effective_conductivity_W_mK, mass_transfer_coefficient_m_s, heat_transfer_coefficient_W_m2K
    if (lam is not None or h is not None) and dH is None:
        raise ValueError("falta heat_of_reaction_J_mol: se dio conductividad del pellet o coeficiente de película térmico")
    cp = U = Tc = None
    if energy_mode != "isothermal":
        if dH is None:
            raise ValueError("falta heat_of_reaction_J_mol (negativo si es exotérmica): balance de energía del lecho")
        cp = _req("gas_heat_capacity_J_molK", gas_heat_capacity_J_molK, "capacidad calorífica molar media del gas")
        if kin["E"] is None:
            warnings.append("sin activation_energy_J_mol la velocidad no depende de T: el perfil térmico no afecta la "
                            "conversión")
    if energy_mode == "wall_cooled":
        U = _req("wall_heat_transfer_coefficient_W_m2K", wall_heat_transfer_coefficient_W_m2K,
                 "coeficiente global lecho-refrigerante referido a la pared interna")
        Tc = _req("coolant_temperature_K", coolant_temperature_K, "temperatura del refrigerante (uniforme)")
    if kin["E"] is not None and kin["E"] > 0 and energy_mode == "isothermal":
        assumptions.append(f"lecho isotérmico a {T0} K por decisión del usuario (energy_mode)")
    p = PL.Pellet(geometry, L, kin, De, dH if (lam is not None or h is not None) else None, lam, kc, h)
    s = p.s
    dp = ergun_particle_diameter_m if ergun_particle_diameter_m is not None else 6.0 * L / (s + 1.0)
    ergun = gas_viscosity_Pa_s is not None or feed_molar_mass_kg_mol is not None
    if ergun:
        mu = _req("gas_viscosity_Pa_s", gas_viscosity_Pa_s, "viscosidad del gas, para Ergun")
        Mf = _req("feed_molar_mass_kg_mol", feed_molar_mass_kg_mol, "masa molar media de la alimentación, para Ergun")
        dp = _req("ergun_particle_diameter_m", dp, "diámetro de partícula de Ergun")
        if ergun_particle_diameter_m is None:
            assumptions.append(f"diámetro de Ergun = 6·Vp/Sp = {dp:.4g} m (esfera: 2R; cilindro largo: 3R, sin tapas)")
    else:
        mu = Mf = None
        warnings.append("sin gas_viscosity_Pa_s ni feed_molar_mass_kg_mol no se calcula la caída de presión (lecho "
                        "isobárico)")
        assumptions.append("lecho isobárico (no se dieron datos para Ergun)")
    A1 = math.pi * D * D / 4.0
    A = A1 * ntub
    perim = math.pi * D * ntub
    mdot = Nt0 * Mf if ergun else None
    G = mdot / A if ergun else None

    def make_rhs(local: _LocalRate):
        def rhs(z, y):
            NA, T, P = y[0], y[1], y[2]
            Nt = Nt0 + delta * (NA0 - NA)
            C = max(NA, 0.0) / Nt * P / (R_GAS * T)
            r = local(C, T)["rate"] if C > 0 else 0.0
            rv = (1.0 - eps) * r
            q_wall = U * perim * (Tc - T) if energy_mode == "wall_cooled" else 0.0
            dT = 0.0 if energy_mode == "isothermal" else ((-dH) * rv * A + q_wall) / (Nt * cp)
            if ergun:
                rho = P * (mdot / Nt) / (R_GAS * T)
                dP = -G / (rho * dp) * (1.0 - eps) / eps ** 3 * (150.0 * (1.0 - eps) * mu / dp + 1.75 * G)
            else:
                dP = 0.0
            return [-rv * A, dT, dP, q_wall]
        return rhs

    def run(rtol: float):
        local = _LocalRate(p, branch)
        rhs = make_rhs(local)
        y0 = [NA0, T0, P0, 0.0]
        atol = [1e-10 * NA0, 1e-9 * T0, 1e-9 * P0, 1e-9 * max(abs(dH or 0.0) * NA0, 1.0)]

        def ev_conv(z, y):
            return y[0] - (1.0 - Xt) * NA0 if Xt is not None else 1.0
        ev_conv.terminal, ev_conv.direction = True, -1.0

        def ev_p(z, y):
            return y[2] - 0.02 * P0
        ev_p.terminal, ev_p.direction = True, -1.0
        if Lbed is not None:
            sol = solve_ivp(rhs, (0.0, Lbed), y0, method="RK45", rtol=rtol, atol=atol, dense_output=True,
                            events=[ev_conv, ev_p])
            pieces, stop = [sol], ("pressure" if sol.status == 1 else "length")
        else:
            r_in = (1.0 - eps) * local(NA0 / Nt0 * P0 / (R_GAS * T0), T0)["rate"]
            z_end = 2.0 * Xt * NA0 / max(r_in * A, 1e-300)
            z0, y, pieces, stop = 0.0, y0, [], None
            for _ in range(40):
                sol = solve_ivp(rhs, (z0, z_end), y, method="RK45", rtol=rtol, atol=atol, dense_output=True,
                                events=[ev_conv, ev_p])
                pieces.append(sol)
                if sol.status != 0:
                    break
                if sol.status == 0 and sol.t[-1] >= z_end:
                    z0, y, z_end = sol.t[-1], sol.y[:, -1], 2.0 * z_end
            if sol.status == 1:
                stop = "target" if len(sol.t_events[0]) else "pressure"
            elif sol.status != 0:
                stop = "failed"
            else:
                stop = "not_reached"
        return pieces, stop, local

    def evaluate(pieces, z):
        for sol in pieces:
            if sol.t[0] - 1e-15 <= z <= sol.t[-1] + 1e-15:
                return sol.sol(z)
        return pieces[-1].sol(pieces[-1].t[-1])

    try:
        coarse, stop_c, _ = run(ODE_RTOL)
        fine, stop_f, local = run(ODE_RTOL_FINE)
    except RuntimeError as exc:
        return _clean({"status": "no_convergence", "evidence": {"verified": False, "reasons": [str(exc)]},
                       "warnings": warnings, "assumptions": assumptions, "not_included": not_included,
                       "method": "lecho 1-D con η local; falló un pellet"})
    method = ("lecho fijo heterogéneo 1-D: balances de moles, energía y Ergun integrados con scipy.integrate.solve_ivp "
              "(RK45); en cada posición, η local del problema de contorno del pellet (packages.transport.pellet: "
              "solve_bvp con arranque desde la posición anterior; primer orden isotérmico: η exacto)")
    if any(pc.status == -1 for pc in coarse + fine):
        return _clean({"status": "no_convergence", "evidence": {"verified": False, "reasons": [
            "la integración axial (solve_ivp) falló"]}, "warnings": warnings, "assumptions": assumptions,
            "not_included": not_included, "method": method})
    reasons: list[str] = []
    z_out = fine[-1].t[-1]
    z_out_c = coarse[-1].t[-1]
    yf, yc = fine[-1].y[:, -1], coarse[-1].y[:, -1]
    X_f, X_c = 1.0 - yf[0] / NA0, 1.0 - yc[0] / NA0
    ev: dict = {"stop_reason": stop_f}
    if Xt is not None:
        ev["length_step_independence_rel"] = abs(z_out - z_out_c) / z_out
        step_rel = ev["length_step_independence_rel"]
    else:
        ev["conversion_step_independence_rel"] = abs(X_f - X_c) / max(X_f, 1e-300)
        step_rel = ev["conversion_step_independence_rel"]
    ev["temperature_step_independence_rel"] = abs(yf[1] - yc[1]) / max(abs(yf[1] - T0), 1e-6 * T0)
    ev["pressure_step_independence_rel"] = abs(yf[2] - yc[2]) / max(abs(P0 - yf[2]), 1e-9 * P0)
    if max(step_rel, ev["temperature_step_independence_rel"] if energy_mode != "isothermal" else 0.0,
           ev["pressure_step_independence_rel"] if ergun else 0.0) > STEP_TOL:
        reasons.append("el resultado cambia más de 1e-4 al apretar la tolerancia de integración axial")
    if stop_f == "not_reached":
        reasons.append("no se alcanzó la conversión objetivo")
    if stop_f == "pressure":
        warnings.append("la presión cae por debajo del 2 % de la de entrada: el lecho no es operable con este caudal")
    if stop_f == "failed":
        reasons.append("la integración axial se detuvo")

    # energy closure
    zq = np.linspace(0.0, z_out, N_QUAD)
    Y = np.array([evaluate(fine, z) for z in zq]).T
    Xr = NA0 - Y[0]
    released = abs(dH or 0.0) * Xr[-1]
    if energy_mode == "adiabatic":
        Ntz = Nt0 + delta * Xr
        exact = T0 + ((-dH) / (cp * delta) * math.log(Ntz[-1] / Nt0) if delta != 0 else (-dH) * Xr[-1] / (cp * Nt0))
        ev["energy_closure_rel"] = abs(Y[1, -1] - exact) / max(abs(exact - T0), 1e-12 * T0)
        ev["energy_closure_method"] = "invariante exacto del balance adiabático con cp constante"
    elif energy_mode == "wall_cooled":
        Ntz = Nt0 + delta * Xr
        Ntm = 0.5 * (Ntz[1:] + Ntz[:-1])
        sens = cp * float(np.sum(Ntm * np.diff(Y[1])))
        ev["energy_closure_rel"] = abs(sens - ((-dH) * Xr[-1] + Y[3, -1])) / max(released, abs(Y[3, -1]), 1e-300)
        ev["energy_closure_method"] = "∫Nt·cp·dT = (−ΔH)·(NA0 − NA) + Q_pared, cuadratura sobre el perfil denso"
    if "energy_closure_rel" in ev and ev["energy_closure_rel"] > ENERGY_TOL:
        reasons.append(f"el balance de energía no cierra ({ev['energy_closure_rel']:.1e})")

    # closed form: isothermal, isobaric, first order, constant moles
    analytic_ok = local.analytic and energy_mode == "isothermal" and not ergun and delta == 0.0
    if analytic_ok:
        r1 = local(1.0, T0)
        kr = r1["rate"]     # η·k at C = 1
        if Xt is not None:
            exact_z = -math.log(1.0 - Xt) * R_GAS * T0 * Nt0 / ((1.0 - eps) * kr * P0 * A)
            ev["analytical_length_m"] = exact_z
            ev["analytical_rel_dev"] = abs(z_out - exact_z) / exact_z
        else:
            exact_X = 1.0 - math.exp(-(1.0 - eps) * kr * P0 * A * z_out / (R_GAS * T0 * Nt0))
            ev["analytical_conversion"] = exact_X
            ev["analytical_rel_dev"] = abs(X_f - exact_X) / exact_X
        if ev["analytical_rel_dev"] > STEP_TOL:
            reasons.append(f"la conversión difiere de la solución cerrada en {ev['analytical_rel_dev']:.1e}")

    # profile with independent pellet solves
    n_profile = max(int(n_profile), 3)
    zp = np.linspace(0.0, z_out, n_profile)
    prof = {k: [] for k in ("z_m", "conversion", "temperature_K", "pressure_Pa", "concentration_A_mol_m3",
                            "eta_overall", "eta_internal", "rate_mol_m3_pellet_s", "diffusion_limited")}
    local_dev, pellet_fail, multi_pts = 0.0, 0, 0
    for z in zp:
        y = evaluate(fine, z)
        Nt = Nt0 + delta * (NA0 - y[0])
        C = max(y[0], 0.0) / Nt * y[2] / (R_GAS * y[1])
        warm = local(C, y[1])
        if local.analytic or p.dead_core:
            full = {**warm, "verified": True, "n_states": 1}
        else:
            full = _full_pellet(p, C, y[1], branch)
        if not full["verified"]:
            pellet_fail += 1
        if full.get("n_states", 1) > 1:
            multi_pts += 1
        if full["rate"] and warm["rate"]:
            local_dev = max(local_dev, abs(warm["rate"] - full["rate"]) / abs(full["rate"]))
        prof["z_m"].append(float(z))
        prof["conversion"].append(float(1.0 - y[0] / NA0))
        prof["temperature_K"].append(float(y[1]))
        prof["pressure_Pa"].append(float(y[2]))
        prof["concentration_A_mol_m3"].append(float(C))
        prof["eta_overall"].append(full["eta_overall"])
        prof["eta_internal"].append(full["eta_internal"])
        prof["rate_mol_m3_pellet_s"].append(full["rate"])
        prof["diffusion_limited"].append(bool(full["eta_internal"] is not None and full["eta_internal"] < DIFF_LIMITED_ETA))
    ev["local_rate_max_rel_dev"] = local_dev
    ev["pellet_solves_not_verified"] = pellet_fail
    if local_dev > LOCAL_RATE_TOL:
        reasons.append(f"la velocidad del pellet con arranque en caliente difiere de la resolución completa en "
                       f"{local_dev:.1e}")
    if pellet_fail:
        reasons.append(f"{pellet_fail} resoluciones del pellet en el perfil no quedaron verificadas")
    if multi_pts:
        warnings.append(f"en {multi_pts} posiciones del perfil el pellet tiene varios estados estacionarios: se siguió la "
                        f"rama '{branch}'; el lecho real depende del arranque")
    ev["verified"] = not reasons
    ev["reasons"] = reasons

    Vbed = A * z_out
    iT = int(np.argmax(Y[1]))
    lim = [z for z, f in zip(prof["z_m"], prof["diffusion_limited"]) if f]
    out = {"status": "ok" if stop_f in ("target", "length") else stop_f,
           "conversion": X_f, "bed_length_m": z_out, "bed_volume_m3": Vbed, "catalyst_volume_m3": (1.0 - eps) * Vbed,
           "outlet_temperature_K": float(yf[1]), "outlet_pressure_Pa": float(yf[2]), "pressure_drop_Pa": P0 - float(yf[2]),
           "outlet_molar_flow_A_mol_s": float(yf[0]), "heat_removed_by_wall_W": -float(yf[3]),
           "hot_spot_temperature_K": float(Y[1, iT]), "hot_spot_position_m": float(zq[iT]),
           "cross_section_m2": A, "ergun_particle_diameter_m": dp if ergun else None,
           "effective_diffusivity": {"value_m2_s": De, **de_info},
           "diffusion_limited_positions_m": lim, "fraction_of_profile_diffusion_limited": len(lim) / n_profile,
           "profile": prof}
    if pellet_density_kg_m3:
        out["catalyst_mass_kg"] = (1.0 - eps) * Vbed * float(pellet_density_kg_m3)
    if lim:
        warnings.append(f"limitación por difusión interna (η interno < {DIFF_LIMITED_ETA}) en {len(lim)} de {n_profile} "
                        "puntos del perfil")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included, method=method, evidence=ev)
    return _clean(out)


# ---------------------------------------------------------------------------------------------------------------------
# gas-liquid film theory
# ---------------------------------------------------------------------------------------------------------------------
def _x_coth(x: float) -> float:
    return 1.0 + x * x / 3.0 if x < 1e-4 else x / math.tanh(x)


def enhancement_first_order(Ha: float) -> float:
    """E = Ha/tanh Ha (film theory, first order, C_Ab = 0)."""
    return _x_coth(Ha)


def enhancement_van_krevelen(Ha: float, Ei: float) -> float:
    """van Krevelen-Hoftijzer: E = Ha√X / tanh(Ha√X), X = (E_i − E)/(E_i − 1); root in (1, E_i)."""
    if Ei <= 1.0 + 1e-12:
        return 1.0
    return brentq(lambda E: E - _x_coth(Ha * math.sqrt(max((Ei - E) / (Ei - 1.0), 0.0))), 1.0, Ei, xtol=1e-14,
                  rtol=1e-13)


def solve_film(Ha: float, q: float | None, ab: float, tol: float = FILM_TOL) -> dict:
    """Numerical film: a'' = Ha² a b, b'' = Ha² a b/q (q None: first order, b ≡ 1). Continuation in Ha² from Ha ≤ 1."""
    second = q is not None

    def fun_for(H2):
        def fun(y, w):
            ab_ = np.maximum(w[0], 0.0) * (np.maximum(w[2], 0.0) if second else 1.0)
            rows = [w[1], H2 * ab_]
            if second:
                rows += [w[3], H2 * ab_ / q]
            return np.vstack(rows)
        return fun

    def bc(wa, wb):
        r = [wa[0] - 1.0, wb[0] - ab]
        if second:
            r += [wa[3], wb[2] - 1.0]
        return np.array(r)

    x = np.unique(np.concatenate([np.linspace(0, 1, 41), np.geomspace(1e-5, 1.0, 60)]))
    H = min(Ha, 1.0)
    w = np.zeros((4 if second else 2, x.size))
    w[0], w[1] = 1.0 - (1.0 - ab) * x, -(1.0 - ab)
    if second:
        w[2] = 1.0
    sol = None
    while True:
        sol = solve_bvp(fun_for(H * H), bc, x, w, tol=tol if H == Ha else 1e-4, max_nodes=100000)
        if sol.status != 0:
            return {"converged": False, "message": sol.message}
        if H == Ha:
            break
        x, w = sol.x, sol.y
        H = min(Ha, 2.0 * H)
    xa, xb = sol.x[:-1], sol.x[1:]
    xg = (0.5 * (xa + xb)[:, None] + 0.5 * (xb - xa)[:, None] * _GL_X[None, :]).ravel()
    wg = (0.5 * (xb - xa)[:, None] * _GL_W[None, :]).ravel()
    yg = sol.sol(xg)
    react = Ha * Ha * float(np.sum(wg * np.maximum(yg[0], 0.0) * (np.maximum(yg[2], 0.0) if second else 1.0)))
    da0, da1 = float(sol.y[1, 0]), float(sol.y[1, -1])
    out = {"converged": True, "flux_interface": -da0, "flux_bulk": -da1, "reaction_in_film": react,
           "film_balance_rel": abs((da1 - da0) - react) / max(-da0, 1e-300), "nodes": int(sol.x.size),
           "b_interface": float(sol.y[2, 0]) if second else 1.0, "sol": sol}
    if second:
        out["stoich_balance_rel"] = abs(q * float(sol.y[3, -1]) - (da1 - da0)) / max(-da0, 1e-300)
    return out


def _film_checked(Ha: float, q: float | None, ab: float) -> tuple[dict, dict]:
    """Film solution at the normal and the fine tolerance, with evidence."""
    f1 = solve_film(Ha, q, ab, FILM_TOL)
    reasons: list[str] = []
    if not f1["converged"]:
        return f1, {"verified": False, "reasons": [f"la solución numérica de la película no convergió ({f1['message']})"]}
    f2 = solve_film(Ha, q, ab, FILM_TOL_FINE)
    ev: dict = {"film_balance_rel": f1["film_balance_rel"], "film_nodes": f1["nodes"]}
    if f2["converged"]:
        ev["mesh_independence_rel"] = abs(f2["flux_interface"] - f1["flux_interface"]) / f2["flux_interface"]
        if ev["mesh_independence_rel"] > FILM_MESH_TOL:
            reasons.append(f"el flujo cambia {ev['mesh_independence_rel']:.1e} al refinar la malla de la película")
        f1 = f2
        ev["film_balance_rel"] = f2["film_balance_rel"]
    else:
        reasons.append("la resolución fina de la película no convergió")
    if ev["film_balance_rel"] > FILM_BALANCE_TOL:
        reasons.append(f"la reacción integrada en la película no iguala la diferencia de flujos "
                       f"({ev['film_balance_rel']:.1e})")
    if "stoich_balance_rel" in f1:
        ev["stoich_balance_rel"] = f1["stoich_balance_rel"]
        if f1["stoich_balance_rel"] > FILM_BALANCE_TOL:
            reasons.append("el flujo de B no cierra con la estequiometría")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    return f1, ev


def gas_liquid_reaction(kinetics_type: str | None = None, liquid_mass_transfer_coefficient_m_s: float | None = None,
                        diffusivity_A_m2_s: float | None = None, interface_concentration_A_mol_m3: float | None = None,
                        gas_partial_pressure_Pa: float | None = None, henry_constant_Pa_m3_mol: float | None = None,
                        gas_mass_transfer_coefficient_mol_m2_s_Pa: float | None = None, rate_constant: float | None = None,
                        bulk_concentration_B_mol_m3: float | None = None, diffusivity_B_m2_s: float | None = None,
                        stoichiometric_ratio_B: float | None = None, bulk_concentration_A_mol_m3: float | None = None,
                        contactor: dict | None = None) -> dict:
    """Absorption with reaction by film theory (see the module docstring)."""
    warnings: list[str] = []
    assumptions = ["teoría de la película (Lewis-Whitman / Hatta): película líquida estancada de espesor δ = D_A/kL, "
                   "estado estacionario; B no volátil; propiedades constantes; isotérmico"]
    not_included = ["teorías de penetración y renovación de superficie (dan E algo distintos; mismas asíntotas)",
                    "reacción reversible, efectos térmicos en la interfase, Marangoni",
                    "columnas de relleno o burbujas con perfiles axiales (solo líquido perfectamente mezclado)",
                    "lechos de goteo (trickle) y suspensiones con catalizador sólido (slurry con transporte al sólido)"]
    if kinetics_type not in ("first_order", "second_order", "instantaneous"):
        raise ValueError("falta kinetics_type: 'first_order' (r = k1·C_A), 'second_order' (A + ν B, r = k2·C_A·C_B) o "
                         "'instantaneous' (A + ν B instantánea)")
    kL = _req("liquid_mass_transfer_coefficient_m_s", liquid_mass_transfer_coefficient_m_s,
              "coeficiente de transferencia del lado líquido sin reacción")
    DA = _req("diffusivity_A_m2_s", diffusivity_A_m2_s, "difusividad de A en el líquido")
    kG = gas_mass_transfer_coefficient_mol_m2_s_Pa
    if interface_concentration_A_mol_m3 is not None:
        if gas_partial_pressure_Pa is not None or kG is not None:
            raise ValueError("da interface_concentration_A_mol_m3 o (gas_partial_pressure_Pa + henry_constant_Pa_m3_mol), "
                             "no ambos")
        Cstar = _req("interface_concentration_A_mol_m3", interface_concentration_A_mol_m3,
                     "concentración de A en el líquido en la interfase")
        p_A = H = None
    else:
        p_A = _req("gas_partial_pressure_Pa", gas_partial_pressure_Pa, "presión parcial de A en el gas (o da "
                   "interface_concentration_A_mol_m3)")
        H = _req("henry_constant_Pa_m3_mol", henry_constant_Pa_m3_mol, "constante de Henry p = H·C de A en el líquido")
        if kG is not None:
            kG = _req("gas_mass_transfer_coefficient_mol_m2_s_Pa", kG, "coeficiente del lado gas")
        else:
            assumptions.append("sin resistencia del lado gas (no se dio kG): C_Ai = p_A/H")
        Cstar = p_A / H
    if kinetics_type != "first_order":
        CB = _req("bulk_concentration_B_mol_m3", bulk_concentration_B_mol_m3, "concentración de B en el seno del líquido")
        DB = _req("diffusivity_B_m2_s", diffusivity_B_m2_s, "difusividad de B en el líquido")
        nu = _req("stoichiometric_ratio_B", stoichiometric_ratio_B, "moles de B por mol de A (ν de A + ν B)")
    else:
        CB = DB = nu = None
    if kinetics_type != "instantaneous":
        k = _req("rate_constant", rate_constant, "k1 [1/s] (primer orden) o k2 [m³/(mol·s)] (segundo orden)")
    else:
        k = None
        if rate_constant is not None:
            raise ValueError("una reacción instantánea no lleva rate_constant")
    if contactor is not None and kinetics_type != "first_order":
        raise ValueError("el contactor con líquido mezclado se resuelve solo para primer orden (o pseudo-primer orden con "
                         "k1 = k2·C_B si B está en exceso)")
    if contactor is None:
        if bulk_concentration_A_mol_m3 is None:
            raise ValueError("falta bulk_concentration_A_mol_m3: concentración de A en el seno del líquido (0 si la "
                             "reacción la consume toda), o contactor para calcularla del balance del líquido")
        Ab = _req("bulk_concentration_A_mol_m3", bulk_concentration_A_mol_m3, "A en el seno del líquido", allow_zero=True)
        if kinetics_type == "instantaneous" and Ab > 0:
            raise ValueError("con reacción instantánea A y B no coexisten en el seno: bulk_concentration_A_mol_m3 "
                             "debe ser 0")
    else:
        Ab = None
    ev: dict = {}
    reasons: list[str] = []
    out: dict = {"status": "ok", "kinetics_type": kinetics_type}
    Ha = math.sqrt(k * DA) / kL if kinetics_type == "first_order" else (
        math.sqrt(k * CB * DA) / kL if kinetics_type == "second_order" else math.inf)

    # ---- first order (with or without contactor) — linear in C_Ai and C_Ab: closed form, checked numerically
    if kinetics_type == "first_order":
        th, ch, sh = (math.tanh(Ha), math.cosh(Ha), math.sinh(Ha)) if Ha < 300 else (1.0, math.inf, math.inf)
        # N_i = kL Ha (Ci − Ab/cosh)/tanh ; N_b = kL Ha (Ci/sinh − Ab/tanh)   (Ramachandran Eq. A, B)
        a11, a12 = kL * Ha / th, -kL * Ha / (th * ch)
        b11, b12 = kL * Ha / sh, -kL * Ha / th
        if contactor is not None:
            c = contactor
            V = _req("reactor_volume_m3", c.get("reactor_volume_m3"), "volumen del contactor (gas + líquido)")
            eL = _req("liquid_holdup", c.get("liquid_holdup"), "fracción de líquido en el contactor")
            QL = _req("liquid_flow_m3_s", c.get("liquid_flow_m3_s"), "caudal de líquido")
            Ain = _req("inlet_concentration_A_mol_m3", c.get("inlet_concentration_A_mol_m3"), "A en el líquido de entrada",
                       allow_zero=True)
            if c.get("kLa_1_s") is not None:
                if c.get("interfacial_area_m2_m3") is not None:
                    raise ValueError("da contactor['kLa_1_s'] o contactor['interfacial_area_m2_m3'], no ambos")
                a_i = _req("kLa_1_s", c["kLa_1_s"], "kL·a volumétrico") / kL
            else:
                a_i = _req("interfacial_area_m2_m3", c.get("interfacial_area_m2_m3"), "área interfacial por volumen de "
                           "contactor (o kLa_1_s)")
            assumptions.append("contactor con líquido perfectamente mezclado y gas de composición uniforme (C_Ai igual "
                               "en todo el contactor); reacción en el seno sobre el volumen de líquido εL·V")
        # unknowns: Ci (if gas film) and Ab (if contactor)
        if contactor is None:
            if kG is None:
                Ci = Cstar
            else:   # kG (p − H Ci) = a11 Ci + a12 Ab
                Ci = (kG * p_A - a12 * Ab) / (kG * H + a11)
        else:
            # bulk: a V (b11 Ci + b12 Ab) = QL (Ab − Ain) + eL V k Ab
            if kG is None:
                Ci = Cstar
                Ab = (a_i * V * b11 * Ci + QL * Ain) / (QL + eL * V * k - a_i * V * b12)
            else:
                M = np.array([[kG * H + a11, a12], [-a_i * V * b11, QL + eL * V * k - a_i * V * b12]])
                Ci, Ab = np.linalg.solve(M, np.array([kG * p_A, QL * Ain]))
                Ci, Ab = float(Ci), float(Ab)
        if Ab > Ci:
            warnings.append("C_A en el seno supera la de la interfase: hay desorción, no absorción")
        Ni = a11 * Ci + a12 * Ab
        Nb = b11 * Ci + b12 * Ab
        film, fev = _film_checked(Ha, None, Ab / Ci)
        ev.update(fev)
        reasons += fev["reasons"]
        if film["converged"]:
            ev["numerical_flux_rel_dev_vs_closed_form"] = abs(film["flux_interface"] * kL * Ci - Ni) / abs(Ni)
            if ev["numerical_flux_rel_dev_vs_closed_form"] > FILM_CHECK_TOL:
                reasons.append("la película numérica difiere de la solución cerrada de primer orden")
        E = Ni / (kL * (Ci - Ab)) if Ci > Ab else None
        out.update(hatta_number=Ha, enhancement_factor=E, enhancement_first_order_closed_form=enhancement_first_order(Ha),
                   interface_concentration_A_mol_m3=Ci, bulk_concentration_A_mol_m3=Ab, absorption_flux_mol_m2_s=Ni,
                   flux_into_bulk_mol_m2_s=Nb, fraction_reacted_in_film=(Ni - Nb) / Ni if Ni else None)
        frac_film = (Ni - Nb) / Ni if Ni else 0.0
        if contactor is not None:
            absorbed = a_i * V * Ni
            bulk_r = eL * V * k * Ab
            leaving = QL * (Ab - Ain)
            film_r = a_i * V * (Ni - Nb)
            ev["contactor_balance_rel"] = abs(absorbed - film_r - bulk_r - leaving) / max(abs(absorbed), 1e-300)
            if ev["contactor_balance_rel"] > 1e-9:
                reasons.append("el balance de A del contactor no cierra")
            out["contactor"] = {"interfacial_area_m2_m3": a_i, "kLa_1_s": kL * a_i,
                                "absorption_rate_mol_m3_s": a_i * Ni, "absorption_rate_mol_s": absorbed,
                                "reacted_in_film_mol_s": film_r, "reacted_in_bulk_mol_s": bulk_r,
                                "leaving_with_liquid_mol_s": leaving,
                                "alpha_gl": kL * a_i * V / QL, "beta_r": eL * V * k / QL,
                                "dimensionless_bulk_concentration": Ab / Ci}
        bi = 1.0
        Ei = None
    # ---- instantaneous: reaction plane, exact in film theory
    elif kinetics_type == "instantaneous":
        Bterm = DB * CB / (nu * DA)        # N = kL (Ci + Bterm)
        if kG is None:
            Ci = Cstar
        else:
            Ci = (kG * p_A - kL * Bterm) / (kG * H + kL)
        if Ci <= 0.0:
            Ci = 0.0
            Ni = kG * p_A
            warnings.append("el plano de reacción llega a la interfase: la absorción la controla el lado gas (C_Ai = 0)")
            E = math.inf
            Ei = math.inf
        else:
            Ni = kL * (Ci + Bterm)
            Ei = 1.0 + Bterm / Ci
            E = Ei
        out.update(hatta_number=Ha, enhancement_factor=E, instantaneous_enhancement_factor=Ei,
                   q_ratio=Bterm / Ci if Ci > 0 else math.inf, interface_concentration_A_mol_m3=Ci,
                   bulk_concentration_A_mol_m3=0.0, absorption_flux_mol_m2_s=Ni, flux_into_bulk_mol_m2_s=0.0,
                   fraction_reacted_in_film=1.0,
                   reaction_plane_position_over_film=(1.0 / Ei if Ci > 0 else 0.0))
        ev.update(verified=True, method="solución exacta del plano de reacción (teoría de película)")
        frac_film, bi = 1.0, 0.0
    # ---- second order: numerical film, vK-H cross-check
    else:
        def flux_at(Ci_):
            q_ = DB * CB / (nu * DA * Ci_)
            f, fe = _film_checked(Ha, q_, Ab / Ci_)
            return f, fe, q_
        if kG is None:
            Ci = Cstar
            film, fev, q = flux_at(Ci)
        else:
            cache = {}

            def mism(Ci_):
                f, fe, q_ = flux_at(Ci_)
                if not f["converged"]:
                    raise RuntimeError("la película numérica no convergió")
                cache["last"] = (f, fe, q_)
                return kG * (p_A - H * Ci_) - kL * Ci_ * f["flux_interface"]
            try:
                Ci = brentq(mism, max(Ab, 1e-12 * Cstar) * (1 + 1e-9), Cstar, xtol=1e-14 * Cstar, rtol=1e-12)
                film, fev, q = flux_at(Ci)
            except (RuntimeError, ValueError) as exc:
                return _clean({"status": "no_convergence", "evidence": {"verified": False, "reasons": [str(exc)]},
                               "warnings": warnings, "assumptions": assumptions, "not_included": not_included,
                               "method": "teoría de película numérica"})
        ev.update(fev)
        reasons += fev["reasons"]
        if not film["converged"]:
            return _clean({"status": "no_convergence", "evidence": ev, "warnings": warnings, "assumptions": assumptions,
                           "not_included": not_included, "method": "teoría de película numérica (solve_bvp)"})
        Ni = kL * Ci * film["flux_interface"]
        Nb = kL * Ci * film["flux_bulk"]
        E = Ni / (kL * (Ci - Ab))
        Ei = 1.0 + q if Ab == 0.0 else None
        bi = film["b_interface"]
        frac_film = (Ni - Nb) / Ni
        out.update(hatta_number=Ha, enhancement_factor=E, q_ratio=q, instantaneous_enhancement_factor=Ei,
                   enhancement_first_order_closed_form=enhancement_first_order(Ha),
                   interface_concentration_A_mol_m3=Ci, bulk_concentration_A_mol_m3=Ab,
                   interface_concentration_B_mol_m3=bi * CB, absorption_flux_mol_m2_s=Ni, flux_into_bulk_mol_m2_s=Nb,
                   fraction_reacted_in_film=frac_film)
        if Ei is not None:
            Evk = enhancement_van_krevelen(Ha, Ei)
            out["enhancement_van_krevelen_hoftijzer"] = Evk
            out["van_krevelen_rel_dev_vs_numerical"] = abs(Evk - E) / E
            assumptions.append("E del resultado = solución numérica de la película (rigurosa en teoría de película); van "
                               "Krevelen-Hoftijzer se da como aproximación de comparación")
        else:
            warnings.append("con A en el seno distinto de 0 no aplican E_i ni van Krevelen-Hoftijzer (definidos con "
                            "C_Ab = 0)")
    # ---- regime (from the solved numbers, not from fixed Ha thresholds)
    Ef1 = enhancement_first_order(Ha) if math.isfinite(Ha) else math.inf
    if kinetics_type == "instantaneous":
        regime = "instantánea: reacción en un plano dentro de la película; E = E_i"
    elif frac_film < 0.1:
        regime = "lenta: la reacción ocurre sobre todo en el seno del líquido (la absorción es casi física)"
    elif Ei is not None and E is not None and E > 0.9 * Ei:
        regime = "casi instantánea: E cerca de E_i, B agotado en la interfase"
    elif bi > 0.9 and E is not None and abs(E - Ef1) / Ef1 < 0.05:
        regime = ("rápida de pseudo-primer orden en la película" if frac_film > 0.9 else
                  "intermedia: reacción en la película y en el seno")
    else:
        regime = "intermedia: reacción en la película con agotamiento parcial de B"
    out["regime"] = regime
    out["regime_indicators"] = {"fraction_reacted_in_film": frac_film, "B_interface_over_bulk": bi,
                                "E_over_E_first_order": (E / Ef1) if (E is not None and math.isfinite(Ef1)
                                                                      and math.isfinite(E)) else None,
                                "E_over_E_instantaneous": (E / Ei) if (E is not None and Ei and math.isfinite(Ei)) else None}
    if kG is not None:
        Ni_ = out["absorption_flux_mol_m2_s"]
        gas_drop = Ni_ / kG
        out["gas_film_fraction_of_driving_force"] = gas_drop / p_A
    if p_A is not None:
        out["gas_partial_pressure_Pa"] = p_A
    ev.setdefault("verified", not reasons)
    if kinetics_type != "instantaneous":
        ev["verified"] = not reasons
        ev["reasons"] = reasons
    else:
        ev.setdefault("reasons", [])
    method = ("teoría de película: solución numérica de los perfiles de A y B (scipy.integrate.solve_bvp con continuación "
              "en Ha) contrastada con las formas cerradas (Ha/tanh Ha; plano de reacción E_i = 1 + q; van "
              "Krevelen-Hoftijzer)")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included, method=method, evidence=ev)
    return _clean(out)


# ---------------------------------------------------------------------------------------------------------------------
# three-phase catalytic reactors: slurry and trickle bed (resistances in series; Fogler 1st ed. §12.1-12.2)
# ---------------------------------------------------------------------------------------------------------------------
THREE_PHASE_STEP_TOL = 1e-4
STEP_NAMES = {"gas_film": "transporte en la película de gas", "gas_liquid": "absorción gas-líquido (lado líquido)",
              "liquid_solid": "transporte líquido-partícula", "pellet": "difusión interna y reacción en el pellet"}


def _three_phase_pellet(geometry, characteristic_length_m, kinetics, effective_diffusivity_m2_s, pore,
                        apparent_rate_constant_m3_kg_s, rho_p: float, T: float, first_order_only: bool,
                        assumptions: list, warnings: list) -> dict:
    """Rate of the limiting reactant per kg of catalyst as a function of its surface concentration and of the other
    reactant's concentration: r'(Cs, C_other). From the user's lumped η·k (first order) or from the pellet engine."""
    if apparent_rate_constant_m3_kg_s is not None:
        if kinetics is not None:
            raise ValueError("da apparent_rate_constant_m3_kg_s (η·k medido, primer orden) o kinetics + datos del pellet, "
                             "no ambos")
        ka = _req("apparent_rate_constant_m3_kg_s", apparent_rate_constant_m3_kg_s, "η·k aparente del pellet")
        assumptions.append("η·k aparente dado por el usuario (primer orden en el reactivo limitante); no se calcula η")
        return {"linear": True, "k_app": lambda Cother: ka, "rate": lambda Cs, Cother: ka * Cs, "pellet": None,
                "other_order": 0.0, "info": {"source": "η·k aparente dado por el usuario"}, "rho_p": rho_p}
    if kinetics is None:
        raise ValueError("falta kinetics (velocidad intrínseca del reactivo limitante, con datos del pellet) o "
                         "apparent_rate_constant_m3_kg_s (η·k aparente medido, primer orden)")
    m_o = kinetics.get("other_reactant_order")
    if m_o not in (0, 1, 0.0, 1.0):
        raise ValueError("falta kinetics['other_reactant_order']: 0 (k ya incluye al otro reactivo, pseudo-orden) o 1 "
                         "(r = k·C_lim^n·C_otro)")
    kin = PL.normalize_kinetics({k: v for k, v in kinetics.items() if k != "other_reactant_order"}, rho_p)
    if first_order_only and not (kin["type"] == "power_law" and kin["n"] == 1.0):
        raise ValueError("el lecho de goteo se resuelve con cinética de primer orden en el reactivo limitante (suma de "
                         "resistencias exacta); para otra cinética use el reactor slurry o una cinética de "
                         "pseudo-primer orden")
    L = _req("characteristic_length_m", characteristic_length_m, "semiespesor o radio del pellet")
    if effective_diffusivity_m2_s is not None and pore is not None:
        raise ValueError("da effective_diffusivity_m2_s o pore, no ambos")
    if effective_diffusivity_m2_s is not None:
        De = _req("effective_diffusivity_m2_s", effective_diffusivity_m2_s, "difusividad efectiva en el pellet lleno de "
                  "líquido")
        info = {"source": "dato del usuario", "effective_diffusivity_m2_s": De}
    elif pore is not None:
        est = PL.effective_diffusivity_from_pores(pore, T)
        De = est["effective_diffusivity_m2_s"]
        info = {"source": "ESTIMADA de datos de poro", **est}
        warnings.append("la difusividad efectiva es una estimación de datos de poro, no una medida")
    else:
        raise ValueError("falta effective_diffusivity_m2_s o pore (para la difusión dentro del pellet)")
    if geometry not in PL.SHAPES:
        raise ValueError("geometry debe ser 'slab', 'cylinder' o 'sphere'")
    linear = kin["type"] == "power_law" and kin["n"] == 1.0
    kv0 = float(PL.rate_constant(kin, T))
    cache: dict = {}

    def pellet_for(Cother):
        key = round(float(Cother), 12) if m_o else 0.0
        if key not in cache:
            k2 = dict(kin, k_ref=kin["k_ref"] * (Cother ** m_o if m_o else 1.0))
            cache[key] = PL.Pellet(geometry, L, k2, De)
        return cache[key]

    def k_app(Cother):          # first order: η·k per kg (m³/(kg s))
        kv = kv0 * (Cother ** m_o if m_o else 1.0)
        if kv <= 0.0:
            return 0.0
        return PL.eta_first_order(geometry, L * math.sqrt(kv / De)) * kv / rho_p

    locals_: dict = {}

    def rate(Cs, Cother):
        if Cs <= 0.0:
            return 0.0
        if linear:
            return k_app(Cother) * Cs
        key = round(float(Cother), 12) if m_o else 0.0
        if key not in locals_:                       # warm-started pellet solver per value of the other reactant
            locals_[key] = _LocalRate(pellet_for(Cother), "low")
        return locals_[key](Cs, T)["rate"] / rho_p
    return {"linear": linear, "k_app": k_app, "rate": rate, "pellet": pellet_for, "other_order": float(m_o),
            "info": info, "rho_p": rho_p}


def _pellet_check(pk: dict, Cs: float, Cother: float, T: float) -> dict:
    """Independent full pellet BVP at one state, compared with the rate used (analytical η for first order)."""
    if pk["pellet"] is None or Cs <= 0:
        return {}
    res = PL.solve_pellet(pk["pellet"](Cother), Cs, T)
    if not res["converged"]:
        return {"pellet_verified": False, "pellet_reasons": res["evidence"]["reasons"]}
    r_bvp = res["selected"]["observed_rate_mol_m3_s"] / pk["rho_p"]
    r_used = pk["rate"](Cs, Cother)
    return {"pellet_verified": bool(res["evidence"]["verified"]), "pellet_eta_internal_bvp": res["selected"]["eta_internal"],
            "pellet_rate_vs_bvp_rel": abs(r_used - r_bvp) / r_bvp}


def three_phase_reactor(reactor_type: str | None = None, temperature_K: float | None = None,
                        henry_constant_Pa_m3_mol: float | None = None, kLa_1_s: float | None = None,
                        kGa_1_s: float | None = None, liquid_solid_mass_transfer_coefficient_m_s: float | None = None,
                        geometry: str | None = None, characteristic_length_m: float | None = None,
                        kinetics: dict | None = None, effective_diffusivity_m2_s: float | None = None,
                        pore: dict | None = None, apparent_rate_constant_m3_kg_s: float | None = None,
                        pellet_density_kg_m3: float | None = None, stoichiometric_ratio_B: float | None = None,
                        liquid_reactant_concentration_mol_m3: float | None = None,
                        gas_partial_pressure_Pa: float | None = None, catalyst_loading_kg_m3: float | None = None,
                        liquid_volume_m3: float | None = None, operation: str | None = None,
                        liquid_flow_m3_s: float | None = None, batch_time_s: float | None = None,
                        target_conversion: float | None = None, limiting_reactant: str | None = None,
                        wetting_efficiency: float | None = None, bed_voidage: float | None = None,
                        column_diameter_m: float | None = None, bed_length_m: float | None = None,
                        feed_molar_flow_A_mol_s: float | None = None, feed_total_gas_molar_flow_mol_s: float | None = None,
                        gas_moles_change_per_mol_A: float | None = None, inlet_pressure_Pa: float | None = None,
                        pressure_gradient_Pa_m: float | None = None, n_profile: int = 11) -> dict:
    """Slurry (batch or continuous liquid) and trickle-bed (gas- or liquid-limited) reactors, isothermal, gas A + ν B(l)."""
    warnings: list[str] = []
    assumptions = ["isotérmico; A del gas + ν B del líquido; resistencias en serie en estado estacionario "
                   "(gas → interfase → "
                   "seno del líquido → superficie del catalizador → interior del pellet), Fogler 1.ª ed. §12.1-12.2",
                   "coeficientes de transferencia y mojado dados por el usuario (no se estiman con correlaciones)"]
    not_included = ["efectos térmicos y evaporación del líquido", "reacción en el seno del líquido sin catalizador",
                    "dispersión axial; distribución del líquido (canalización); regímenes de flujo del lecho de goteo",
                    "acceso del gas por la superficie seca del pellet (la parte no mojada se toma inactiva)"]
    if reactor_type not in ("slurry", "trickle_bed"):
        raise ValueError("falta reactor_type: 'slurry' (catalizador suspendido en el líquido) o 'trickle_bed' (lecho de "
                         "goteo)")
    T = _req("temperature_K", temperature_K, "temperatura del reactor")
    H = _req("henry_constant_Pa_m3_mol", henry_constant_Pa_m3_mol, "constante de Henry p = H·C del gas A en el líquido")
    Hcc = H / (R_GAS * T)                       # C_gas / C_liquid at equilibrium
    if reactor_type == "trickle_bed" and limiting_reactant == "liquid":
        kLa = kLa_1_s
    else:
        kLa = _req("kLa_1_s", kLa_1_s, "coeficiente volumétrico gas-líquido del lado líquido (por m³ de líquido en slurry, "
                   "por m³ de lecho en lecho de goteo)")
    if kGa_1_s is not None:
        kGa = _req("kGa_1_s", kGa_1_s, "coeficiente volumétrico del lado gas (fuerza impulsora en concentración del gas)")
    else:
        kGa = None
        assumptions.append("sin resistencia del lado gas (no se dio kGa_1_s)")
    rho_p = _req("pellet_density_kg_m3", pellet_density_kg_m3, "densidad del pellet (área externa por kg y base másica)")
    kc = liquid_solid_mass_transfer_coefficient_m_s
    if kc is not None:
        kc = _req("liquid_solid_mass_transfer_coefficient_m_s", kc, "coeficiente líquido-partícula")
        if geometry not in PL.SHAPES or characteristic_length_m is None:
            raise ValueError("con liquid_solid_mass_transfer_coefficient_m_s hacen falta geometry y characteristic_length_m "
                             "(área externa por kg = (s+1)/(L·ρp))")
        ac = (PL.SHAPES[geometry] + 1.0) / (float(characteristic_length_m) * rho_p)
        assumptions.append(f"área externa por kg de catalizador a_c = (s+1)/(L·ρp) = {ac:.4g} m²/kg (esfera: 6/(dp·ρp))")
    else:
        ac = None
        assumptions.append("sin resistencia líquido-partícula (no se dio liquid_solid_mass_transfer_coefficient_m_s)")
    nu = _req("stoichiometric_ratio_B", stoichiometric_ratio_B, "moles de B (líquido) por mol de A (gas)")
    CB0 = _req("liquid_reactant_concentration_mol_m3", liquid_reactant_concentration_mol_m3,
               "concentración de B en el líquido alimentado (o inicial en lote)")
    trickle = reactor_type == "trickle_bed"
    pk = _three_phase_pellet(geometry, characteristic_length_m, kinetics, effective_diffusivity_m2_s, pore,
                             apparent_rate_constant_m3_kg_s, rho_p, T, trickle, assumptions, warnings)
    reasons: list[str] = []
    ev: dict = {}
    method = ("resistencias en serie (Fogler 1.ª ed. ec. 12-8/12-30) con η del pellet de packages.transport.pellet "
              "(primer orden: η exacto, contrastado con el problema de contorno)")
    if trickle:
        out = _trickle(pk, T, H, Hcc, kLa, kGa, kc, ac, rho_p, nu, CB0, limiting_reactant, wetting_efficiency, bed_voidage,
                       column_diameter_m, bed_length_m, target_conversion, feed_molar_flow_A_mol_s,
                       feed_total_gas_molar_flow_mol_s, gas_moles_change_per_mol_A, inlet_pressure_Pa,
                       pressure_gradient_Pa_m, liquid_flow_m3_s, n_profile, ev, reasons, warnings, assumptions)
        method += "; flujo pistón de gas y líquido en paralelo, integrado en masa de catalizador con solve_ivp (RK45)"
    else:
        out = _slurry(pk, T, H, Hcc, kLa, kGa, kc, ac, nu, CB0, gas_partial_pressure_Pa, catalyst_loading_kg_m3,
                      liquid_volume_m3, operation, liquid_flow_m3_s, batch_time_s, target_conversion, ev, reasons, warnings,
                      assumptions)
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    out.update(reactor_type=reactor_type, pellet=pk["info"], warnings=warnings, assumptions=assumptions,
               not_included=not_included, method=method, evidence=ev)
    return _clean(out)


def _shares(res: dict) -> dict:
    tot = sum(res.values())
    shares = {k: v / tot for k, v in res.items()}
    ctrl = max(shares, key=shares.get)
    return {"resistances": res, "total_resistance": tot, "shares": shares, "controlling_step": ctrl,
            "controlling_step_label": STEP_NAMES[ctrl]}


def _slurry(pk, T, H, Hcc, kLa, kGa, kc, ac, nu, CB0, p_A, m_given, V, operation, QL, t_batch, Xt, ev, reasons, warnings,
            assumptions) -> dict:
    p_A = _req("gas_partial_pressure_Pa", p_A, "presión parcial de A en el gas (uniforme en el reactor)")
    Cstar = p_A / H
    if operation not in ("batch", "continuous"):
        raise ValueError("falta operation: 'batch' (líquido por lotes, gas continuo) o 'continuous' (líquido continuo "
                         "perfectamente mezclado)")
    V = _req("liquid_volume_m3", V, "volumen de líquido (base de kLa y de la carga de catalizador)")
    assumptions.append("líquido perfectamente mezclado y presión parcial de A uniforme; catalizador uniformemente "
                       "suspendido; kLa y la carga de catalizador por m³ de líquido")

    def resistances(m, CB):
        res = {}
        if kGa is not None:
            res["gas_film"] = 1.0 / (Hcc * kGa)
        res["gas_liquid"] = 1.0 / kLa
        if kc is not None:
            res["liquid_solid"] = 1.0 / (m * kc * ac)
        if pk["linear"]:
            ka = pk["k_app"](CB)
            res["pellet"] = 1.0 / (m * ka) if ka > 0 else math.inf
        return res

    memo: dict = {}

    def solve_point(m, CB):
        """Overall rate R (mol A/(m³ liquid s)) and the concentrations of every step (memoized)."""
        key = (float(m), round(float(CB), 12) if pk["other_order"] else 0.0)
        if key not in memo:
            memo[key] = _solve_point(m, CB)
        return memo[key]

    def _solve_point(m, CB):
        res = resistances(m, CB)
        ext = sum(v for k, v in res.items() if k != "pellet")
        if pk["linear"]:
            R = Cstar / sum(res.values())
            Cs = R * res["pellet"] if math.isfinite(res["pellet"]) else Cstar
        else:
            f = lambda Cs: Cstar - Cs - m * pk["rate"](Cs, CB) * ext          # noqa: E731
            Cs = brentq(f, 0.0, Cstar, xtol=1e-14 * Cstar, rtol=1e-13)
            R = m * pk["rate"](Cs, CB)
            res = dict(res, pellet=Cs / R if R > 0 else math.inf)
        return R, Cs, res

    if (m_given is None) == (Xt is None) and operation == "continuous":
        raise ValueError("en continuo da catalyst_loading_kg_m3 (calcula la conversión de B) o target_conversion (calcula "
                         "la carga de catalizador), uno de los dos")
    if Xt is not None and not 0 < float(Xt) < 1:
        raise ValueError("target_conversion debe estar entre 0 y 1")
    out: dict = {"interface_concentration_A_mol_m3": Cstar}
    if operation == "continuous":
        QL = _req("liquid_flow_m3_s", QL, "caudal de líquido")
        FB0 = QL * CB0

        def conv_of_m(m):           # CSTR: FB0 X = ν R(C_B,out) V
            def g(X):
                return FB0 * X - nu * solve_point(m, CB0 * (1 - X))[0] * V
            return brentq(g, 0.0, 1.0, xtol=1e-14) if g(1.0) > 0 else 1.0
        if m_given is not None:
            m = _req("catalyst_loading_kg_m3", m_given, "kg de catalizador por m³ de líquido")
            X = conv_of_m(m)
        else:
            X = float(Xt)
            hi = 1.0
            while conv_of_m(hi) < X:
                hi *= 4.0
                if hi > 1e9:
                    raise ValueError("ni con 1e9 kg/m³ se alcanza la conversión: la absorción gas-líquido la limita "
                                     f"(máximo {nu * Cstar * kLa * V / FB0:.4g} con resistencia de absorción sola)")
            m = brentq(lambda mm: conv_of_m(mm) - X, 0.0 + 1e-300, hi, xtol=1e-12 * hi, rtol=1e-13)
        R, Cs, res = solve_point(m, CB0 * (1 - X))
        out.update(conversion_B=X, catalyst_loading_kg_m3=m, catalyst_mass_kg=m * V,
                   outlet_concentration_B_mol_m3=CB0 * (1 - X))
        ev["cstr_balance_rel"] = abs(FB0 * X - nu * R * V) / max(FB0 * X, 1e-300)
        if ev["cstr_balance_rel"] > 1e-8:
            reasons.append("el balance del CSTR de líquido no cierra")
        if pk["linear"] and pk["other_order"] == 0.0:
            # Fogler Eq. E12-2.3: V C*/(F_A X) = r_g + r_b + (1/m)(r_c + r_r)  →  closed form for m or X
            r_ext = sum(v for k, v in res.items() if k in ("gas_film", "gas_liquid"))
            r_cat = sum(v for k, v in res.items() if k in ("liquid_solid", "pellet")) * m
            if m_given is None:
                exact = r_cat / (V * Cstar * nu / (FB0 * X) - r_ext)
                ev["analytical_catalyst_loading_kg_m3"] = exact
                ev["analytical_rel_dev"] = abs(m - exact) / exact
            else:
                exact = nu * V * Cstar / (FB0 * (r_ext + r_cat / m))
                ev["analytical_conversion"] = exact
                ev["analytical_rel_dev"] = abs(X - exact) / exact
            if ev["analytical_rel_dev"] > 1e-8:
                reasons.append("difiere de la solución cerrada del CSTR")
    else:
        m = _req("catalyst_loading_kg_m3", m_given, "kg de catalizador por m³ de líquido")
        if (t_batch is None) == (Xt is None):
            raise ValueError("en lote da batch_time_s (calcula la conversión) o target_conversion (calcula el tiempo)")
        NB0 = CB0 * V

        def run(rtol):
            def rhs(t, y):
                return [-nu * solve_point(m, max(y[0], 0.0))[0]]

            def ev_x(t, y):
                return y[0] - CB0 * (1 - Xt) if Xt is not None else 1.0
            ev_x.terminal, ev_x.direction = True, -1.0

            def ev_zero(t, y):
                return y[0] - 1e-9 * CB0
            ev_zero.terminal = True
            R0 = solve_point(m, CB0)[0]
            t_end = t_batch if t_batch is not None else 50.0 * CB0 / (nu * R0)
            return solve_ivp(rhs, (0.0, t_end), [CB0], method="RK45", rtol=rtol, atol=1e-12 * CB0, events=[ev_x, ev_zero])
        s1, s2 = run(1e-7), run(1e-9)
        if Xt is not None and not len(s2.t_events[0]):
            reasons.append("no se alcanzó la conversión objetivo en el horizonte de integración")
        tf, CBf = float(s2.t[-1]), float(s2.y[0, -1])
        X = 1 - CBf / CB0
        ev["step_independence_rel"] = abs(float(s1.y[0, -1]) - CBf) / CB0 if Xt is None else abs(float(s1.t[-1]) - tf) / tf
        if ev["step_independence_rel"] > THREE_PHASE_STEP_TOL:
            reasons.append("el resultado del lote cambia al apretar la tolerancia")
        if pk["other_order"] == 0.0:           # rate independent of B: linear consumption
            R0 = solve_point(m, CB0)[0]
            exact_t = X * CB0 / (nu * R0)
            ev["analytical_time_s"] = exact_t
            ev["analytical_rel_dev"] = abs(tf - exact_t) / exact_t if exact_t > 0 else 0.0
            if ev["analytical_rel_dev"] > 1e-6:
                reasons.append("difiere de la solución cerrada del lote de orden cero en B")
        R, Cs, res = solve_point(m, CBf)
        out.update(conversion_B=X, batch_time_s=tf, catalyst_loading_kg_m3=m, catalyst_mass_kg=m * V,
                   final_concentration_B_mol_m3=CBf, gas_A_consumed_mol=(NB0 - CBf * V) / nu)
        if CBf <= 1e-9 * CB0 * 1.0001:
            warnings.append("B se agotó antes del tiempo dado")
    sh = _shares(res)
    Ci = Cstar - R * res.get("gas_film", 0.0)
    Cb = Ci - R * res["gas_liquid"]
    out.update(overall_rate_mol_m3_s=R, rate_per_kg_catalyst_mol_kg_s=R / m, **sh,
               liquid_interface_concentration_A_mol_m3=Ci, bulk_liquid_concentration_A_mol_m3=Cb,
               surface_concentration_A_mol_m3=Cs)
    ev["series_closure_rel"] = abs(Cstar - Cs - R * sum(v for k, v in res.items() if k != "pellet")) / Cstar
    if ev["series_closure_rel"] > 1e-9 and pk["linear"]:
        reasons.append("las caídas de concentración no suman la fuerza impulsora")
    chk = _pellet_check(pk, Cs, out.get("outlet_concentration_B_mol_m3", out.get("final_concentration_B_mol_m3", CB0)), T)
    if chk:
        _merge_pellet_check(chk, ev, reasons)
    if sh["shares"].get("gas_liquid", 0) > 0.5:
        warnings.append("controla la absorción gas-líquido: más catalizador casi no aumenta la velocidad (mejorar kLa)")
    return out


def _merge_pellet_check(chk: dict, ev: dict, reasons: list) -> None:
    ev.update(chk)
    if not chk.get("pellet_verified"):
        reasons.append("el problema de contorno del pellet no quedó verificado")
    if chk.get("pellet_rate_vs_bvp_rel", 0.0) > 1e-4:
        reasons.append("la velocidad del pellet usada difiere de la resolución completa del problema de contorno")


def _trickle(pk, T, H, Hcc, kLa, kGa, kc, ac, rho_p, nu, CB0, limiting, fw, eps, D, Lbed, Xt, FA0, Ft0, delta, P0, dPdL,
             QL, n_profile, ev, reasons, warnings, assumptions) -> dict:
    if limiting not in ("gas", "liquid"):
        raise ValueError("falta limiting_reactant: 'gas' (limita el reactivo gaseoso A; B en el líquido) o 'liquid' (limita "
                         "el reactivo líquido no volátil B; el líquido está saturado de gas)")
    fw = _req("wetting_efficiency", fw, "fracción de la superficie externa del catalizador mojada por el líquido (0-1]")
    if fw > 1.0:
        raise ValueError("wetting_efficiency debe estar entre 0 y 1")
    eps = _req("bed_voidage", eps, "fracción de huecos del lecho")
    if eps >= 1.0:
        raise ValueError("bed_voidage debe estar entre 0 y 1")
    D = _req("column_diameter_m", D, "diámetro de la columna")
    FA0 = _req("feed_molar_flow_A_mol_s", FA0, "flujo molar de A en el gas de entrada")
    Ft0 = _req("feed_total_gas_molar_flow_mol_s", Ft0, "flujo molar total de gas a la entrada")
    if FA0 > Ft0 * (1 + 1e-12):
        raise ValueError("feed_molar_flow_A_mol_s no puede superar feed_total_gas_molar_flow_mol_s")
    if delta is None:
        raise ValueError("falta gas_moles_change_per_mol_A: cambio de moles de GAS por mol de A consumido (−1 si solo A "
                         "pasa al líquido y el producto queda en el líquido)")
    delta = float(delta)
    P0 = _req("inlet_pressure_Pa", P0, "presión a la entrada")
    QL = _req("liquid_flow_m3_s", QL, "caudal volumétrico de líquido")
    if (Lbed is None) == (Xt is None):
        raise ValueError("da bed_length_m (calcula la conversión) o target_conversion (calcula la masa de catalizador)")
    if Xt is not None and not 0 < float(Xt) < 1:
        raise ValueError("target_conversion debe estar entre 0 y 1")
    if dPdL is None:
        dPdL = 0.0
        warnings.append("sin pressure_gradient_Pa_m el lecho se toma isobárico")
    else:
        dPdL = _req("pressure_gradient_Pa_m", dPdL, "caída de presión por metro de lecho (dato o correlación del usuario)",
                    allow_zero=True)
    if fw < 1.0:
        assumptions.append(f"mojado parcial f = {fw}: solo la fracción mojada transfiere y reacciona (a_c·f y velocidad·f); "
                           "la superficie seca se toma inactiva")
    assumptions.append("flujo pistón de gas y líquido en paralelo (cocorriente), gas ideal; caída de presión lineal dada")
    Ac = math.pi * D * D / 4.0
    solid = Ac * (1.0 - eps) * rho_p                 # kg of catalyst per m of bed
    beta = dPdL / (P0 * solid)                       # P/P0 = 1 − β W
    yA0 = FA0 / Ft0
    epsf = yA0 * delta

    def gas_state(W, XA):
        P = P0 * (1.0 - beta * W)
        yA = FA0 * (1 - XA) / (Ft0 + delta * FA0 * XA)
        return P, yA * P / (R_GAS * T)

    def resist(CB, CAs_for_B=None):
        res = {}
        if limiting == "gas":
            if kGa is not None:
                res["gas_film"] = (1.0 - eps) * rho_p / (Hcc * kGa)
            res["gas_liquid"] = (1.0 - eps) * rho_p / kLa
        if kc is not None:
            res["liquid_solid"] = 1.0 / (fw * kc * ac)
        ka = pk["k_app"](CB if limiting == "gas" else CAs_for_B)
        res["pellet"] = 1.0 / (fw * ka) if ka > 0 else math.inf
        return res

    def local(W, y):
        XA, CB = y[0], max(y[1], 0.0)
        P, CAg = gas_state(W, XA)
        if limiting == "gas":
            res = resist(CB)
            r = (max(CAg, 0.0) / Hcc) / sum(res.values())          # mol A/(kg s)
            return r, r * nu, res, P, CAg
        CAs = max(CAg, 0.0) / Hcc                                    # liquid saturated with the local gas
        res = resist(CB, CAs)
        rB = CB / sum(res.values())
        return rB / nu, rB, res, P, CAg

    def rhs(W, y):
        rA, rB, _, _, _ = local(W, y)
        return [rA / FA0, -rB / QL]

    def run(rtol):
        evs = []
        if Xt is not None:
            if limiting == "gas":
                def e(W, y):
                    return y[0] - Xt
                e.direction = 1.0
            else:
                def e(W, y):
                    return y[1] - CB0 * (1 - Xt)
                e.direction = -1.0
            e.terminal = True
            evs.append(e)

        def pe(W, y):
            return 1.0 - beta * W - 0.02
        pe.terminal = True
        evs.append(pe)
        if Lbed is not None:
            W_end = solid * _req("bed_length_m", Lbed, "altura del lecho")
        else:
            rA0 = local(0.0, [0.0, CB0])[0 if limiting == "gas" else 1]
            W_end = 50.0 * Xt * (FA0 if limiting == "gas" else QL * CB0) / max(rA0, 1e-300)
            if beta > 0:
                W_end = min(W_end, 0.98 / beta)
        return solve_ivp(rhs, (0.0, W_end), [0.0, CB0], method="RK45", rtol=rtol, atol=[1e-12, 1e-12 * CB0],
                         events=evs, dense_output=True)
    s1, s2 = run(1e-7), run(1e-9)
    Wf = float(s2.t[-1])
    XA, CBf = float(s2.y[0, -1]), float(s2.y[1, -1])
    XB = 1 - CBf / CB0
    if Xt is not None:
        if not len(s2.t_events[0]):
            reasons.append("no se alcanzó la conversión objetivo (presión agotada o lecho demasiado largo)")
        ev["step_independence_rel"] = abs(float(s1.t[-1]) - Wf) / Wf
    else:
        ev["step_independence_rel"] = max(abs(float(s1.y[0, -1]) - XA), abs(float(s1.y[1, -1]) - CBf) / CB0)
    if len(s2.t_events[-1]):
        warnings.append("la presión cae al 2 % de la de entrada: el lecho no es operable con ese gradiente")
    if ev["step_independence_rel"] > THREE_PHASE_STEP_TOL:
        reasons.append("el resultado cambia al apretar la tolerancia de integración")
    # closed forms (Fogler 1st ed. Eq. E12-3.7 and Eq. 12-39), only where they are exact
    if limiting == "gas" and pk["other_order"] == 0.0:
        res0 = resist(CB0)
        kvg = 1.0 / (Hcc * sum(res0.values()))
        v0 = Ft0 * R_GAS * T / P0
        lhs = (1 + epsf) * math.log(1.0 / (1.0 - XA)) - epsf * XA
        exact_W = (1 - math.sqrt(max(1 - 2 * beta * lhs * v0 / kvg, 0.0))) / beta if beta > 0 else lhs * v0 / kvg
        ev["analytical_catalyst_mass_kg"] = exact_W
        ev["analytical_rel_dev"] = abs(Wf - exact_W) / exact_W
    elif limiting == "liquid" and beta == 0.0 and FA0 == Ft0:
        kvl = 1.0 / sum(resist(CB0, P0 / H).values())
        exact_W = QL / kvl * math.log(1.0 / (1.0 - XB))
        ev["analytical_catalyst_mass_kg"] = exact_W
        ev["analytical_rel_dev"] = abs(Wf - exact_W) / exact_W
    if ev.get("analytical_rel_dev", 0.0) > 1e-5:
        reasons.append("difiere de la solución cerrada (Fogler ec. E12-3.7 / 12-39)")
    rA_in, rB_in, res_in, _, CAg_in = local(0.0, [0.0, CB0])
    rA_out, _, res_out, P_out, _ = local(Wf, [XA, CBf])
    # pellet BVP at the inlet
    if limiting == "gas":
        Cs = rA_in / (fw * pk["k_app"](CB0))
        chk = _pellet_check(pk, Cs, CB0, T)
    else:
        CAs = CAg_in / Hcc
        Cs = rB_in / (fw * pk["k_app"](CAs))
        chk = _pellet_check(pk, Cs, CAs, T)
    if chk:
        _merge_pellet_check(chk, ev, reasons)
    zs = np.linspace(0.0, Wf, max(int(n_profile), 3))
    prof = {"catalyst_mass_kg": [], "height_m": [], "conversion_A": [], "concentration_B_mol_m3": [], "pressure_Pa": [],
            "rate_A_mol_kg_s": []}
    for W in zs:
        y = s2.sol(W)
        rA, _, _, P, _ = local(W, y)
        for k, v in (("catalyst_mass_kg", W), ("height_m", W / solid), ("conversion_A", y[0]),
                     ("concentration_B_mol_m3", y[1]), ("pressure_Pa", P), ("rate_A_mol_kg_s", rA)):
            prof[k].append(float(v))
    sh_in, sh_out = _shares(res_in), _shares(res_out)
    if limiting == "gas" and XB > 0.1:
        warnings.append(f"B se convierte {XB:.1%}: el agotamiento del reactivo líquido ya cuenta (se integra su balance)")
    if limiting == "liquid":
        assumptions.append("caso limitado por el líquido: el líquido está saturado con el gas local (C_A en la superficie "
                           "= p_A/H); las resistencias de absorción no intervienen")
    return {"limiting_reactant": limiting, "catalyst_mass_kg": Wf, "bed_volume_m3": Wf / ((1 - eps) * rho_p),
            "bed_height_m": Wf / solid, "conversion_A": XA, "conversion_B": XB, "outlet_concentration_B_mol_m3": CBf,
            "outlet_pressure_Pa": float(P_out), "wetting_efficiency": fw,
            "overall_rate_inlet_mol_A_kg_s": rA_in, "overall_rate_outlet_mol_A_kg_s": rA_out,
            "inlet": sh_in, "outlet": sh_out, "controlling_step": sh_in["controlling_step"],
            "controlling_step_label": sh_in["controlling_step_label"], "profile": prof}
