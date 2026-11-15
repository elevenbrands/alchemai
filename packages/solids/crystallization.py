"""Crystallization (bloque solids): yield and heat balance, MSMPR population balance, seeded batch cooling by moments.

What it answers:
  * `crystallizer_yield`: crystals, mother liquor and evaporated solvent of a cooling and/or evaporative crystallizer from the
    user's solubility table (anhydrous solute per kg of solvent), with hydrates, and the heat to remove (sensible heat of the
    feed + heat of crystallization − latent heat of the evaporated solvent); adiabatic (vacuum) case with the evaporation
    solved from the enthalpy balance.
  * `msmpr`: steady mixed-suspension mixed-product-removal crystallizer with size-independent growth and nuclei of zero size:
    n(L) = n0·exp(−L/Gτ), moments, mean, dominant (3Gτ) and mass-median sizes, CV, magma density, production.
  * `msmpr_from_sieve`: growth and nucleation rates from a screen analysis by the ln n vs L regression of the textbooks, plus
    an exact fit of the cut mass fractions (no mid-size approximation) to quantify the bias of wide cuts.
  * `batch_seeded_cooling`: seeded batch cooling crystallization by the method of moments (size-independent growth, power-law
    kinetics of the user, user cooling profile), integrated with SciPy (LSODA).

No solubility, kinetic or thermal datum is built in: the solubility curve, heats, heat capacity, kinetics, densities and shape
factors are the user's. Only the molar mass and latent heat of WATER are read from CoolProp (IAPWS-95) when the solvent is
declared to be water."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import gammainc, gammaincinv

BALANCE_TOL = 1e-9
MOMENT_TOL = 1e-8
STEP_TOL = 1e-5            # batch moments: loose vs tight integration tolerance
R2_WARN = 0.95
MASS_MISFIT_WARN = 0.02    # RMS residual of the cut mass fractions against the best exact MSMPR distribution


def _positive(x, name: str, allow_zero: bool = False) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un número") from None
    if not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        raise ValueError(f"{name} debe ser {'no negativo' if allow_zero else 'positivo'} (recibido {x})")
    return x


def _need(d: dict | None, key: str, what: str, where: str):
    if not isinstance(d, dict) or d.get(key) is None:
        raise ValueError(f"falta {where}['{key}']: {what}")
    return d[key]


def _seal(out: dict, evidence: dict, reasons: list[str]) -> dict:
    evidence["verified"] = not reasons
    evidence["reasons"] = reasons
    out["evidence"] = evidence
    return out


def _water(prop: str, T: float | None = None) -> float:
    from CoolProp.CoolProp import PropsSI
    if prop == "M":
        return float(PropsSI("M", "Water"))
    return float(PropsSI("H", "T", T, "Q", 1, "Water") - PropsSI("H", "T", T, "Q", 0, "Water"))


# --------------------------------------------------------------------------------------------------------- solubility
class _Solubility:
    """The user's solubility table, kg anhydrous solute per kg solvent vs T. Monotone cubic (PCHIP) between ≥ 3 points,
    linear between 2, a single point only at its own temperature. Never extrapolated."""

    def __init__(self, table: dict | None, where: str = "solubility"):
        T = _need(table, "T_K", "temperaturas de la tabla de solubilidad", where)
        s = _need(table, "kg_solute_per_kg_solvent", "solubilidad, kg de soluto ANHIDRO por kg de disolvente", where)
        if len(T) != len(s) or not T:
            raise ValueError(f"{where}: 'T_K' y 'kg_solute_per_kg_solvent' deben tener la misma longitud (≥ 1 punto)")
        pts = sorted((float(a), float(b)) for a, b in zip(T, s))
        if any(b[0] - a[0] <= 0 for a, b in zip(pts, pts[1:])):
            raise ValueError(f"{where}: temperaturas repetidas en la tabla")
        if any(b <= 0 for _, b in pts):
            raise ValueError(f"{where}: la solubilidad debe ser positiva")
        self.T = [p[0] for p in pts]
        self.s = [p[1] for p in pts]
        self.pchip = PchipInterpolator(self.T, self.s) if len(pts) >= 3 else None

    def covers(self, T: float) -> bool:
        return self.T[0] - 1e-9 <= T <= self.T[-1] + 1e-9

    def linear(self, T: float) -> float:
        return float(np.interp(T, self.T, self.s))

    def __call__(self, T: float) -> float:
        if not self.covers(T):
            raise ValueError(f"la tabla de solubilidad cubre {self.T[0]:.2f}-{self.T[-1]:.2f} K y se pide {T:.2f} K: no se "
                             "extrapola; añade puntos a la tabla")
        if self.pchip is None:
            return self.linear(T)
        return float(self.pchip(min(max(T, self.T[0]), self.T[-1])))

    def describe(self, T: float) -> dict:
        exact = any(abs(T - t) < 1e-6 for t in self.T)
        value = self(T)
        return {"T_K": T, "kg_solute_per_kg_solvent": value, "tabulated_point": exact,
                "interpolation": (None if exact else ("PCHIP monótona" if self.pchip is not None else "lineal")),
                "linear_vs_pchip_rel_diff": abs(self.linear(T) - value) / value}


# ----------------------------------------------------------------------------------------------------- yield and heat
def crystallizer_yield(feed_mass_kg: float, final_T_K: float, solubility: dict, evaporation: dict,
                       feed_solute_mass_fraction: float | None = None, feed_solute_per_kg_solvent: float | None = None,
                       crystal_solute_mass_fraction: float | None = None, hydrate: dict | None = None,
                       solvent: str | None = None, energy: dict | None = None) -> dict:
    """Mass (and, with `energy`, heat) balance of a cooling / evaporative / adiabatic crystallizer that leaves a mother
    liquor saturated at `final_T_K`. Basis: `feed_mass_kg` (per batch, or kg/s: then read kg as kg/s and J as W)."""
    warnings: list[str] = []
    assumptions = ["las aguas madres salen saturadas a la temperatura final (equilibrio; sin sobresaturación residual)",
                   "solubilidad en kg de soluto anhidro por kg de disolvente, tomada de la tabla del usuario"]
    not_included = ["cinética y distribución de tamaños (usa crystal_size_distribution)", "impurezas y su efecto en la "
                    "solubilidad", "elevación del punto de ebullición y presión de operación del cristalizador al vacío",
                    "calor de dilución/concentración (cp de la solución constante)", "aguas madres retenidas en la torta"]
    F = _positive(feed_mass_kg, "feed_mass_kg")
    Tf = _positive(final_T_K, "final_T_K")
    if (feed_solute_mass_fraction is None) == (feed_solute_per_kg_solvent is None):
        raise ValueError("da la concentración de la alimentación de una sola forma: feed_solute_mass_fraction (soluto "
                         "anhidro / solución) o feed_solute_per_kg_solvent (kg de soluto anhidro por kg de disolvente)")
    if feed_solute_mass_fraction is not None:
        xF = _positive(feed_solute_mass_fraction, "feed_solute_mass_fraction")
        if xF >= 1:
            raise ValueError("feed_solute_mass_fraction debe ser menor que 1")
    else:
        r = _positive(feed_solute_per_kg_solvent, "feed_solute_per_kg_solvent")
        xF = r / (1.0 + r)
    # crystal composition
    if (crystal_solute_mass_fraction is None) == (hydrate is None):
        raise ValueError("di qué cristaliza: crystal_solute_mass_fraction (1.0 si el cristal es anhidro) o hydrate "
                         "{'anhydrous_molar_mass_kg_mol', 'solvent_moles_per_mole'} si es un hidrato/solvato; uno solo")
    M_crystal = None
    if hydrate is not None:
        Ma = _positive(_need(hydrate, "anhydrous_molar_mass_kg_mol", "masa molar del soluto anhidro", "hydrate"),
                       "anhydrous_molar_mass_kg_mol")
        n = _positive(_need(hydrate, "solvent_moles_per_mole", "moles de disolvente por mol de soluto en el cristal",
                            "hydrate"), "solvent_moles_per_mole", allow_zero=True)
        Ms = hydrate.get("solvent_molar_mass_kg_mol")
        if Ms is None:
            if solvent != "water":
                raise ValueError("falta hydrate['solvent_molar_mass_kg_mol'] (o declara solvent='water' para tomar la masa "
                                 "molar del agua de CoolProp): hace falta para la composición del hidrato")
            Ms = _water("M")
            assumptions.append("masa molar del agua de CoolProp (IAPWS-95)")
        Ms = _positive(Ms, "solvent_molar_mass_kg_mol")
        M_crystal = Ma + n * Ms
        xC = Ma / M_crystal
    else:
        xC = _positive(crystal_solute_mass_fraction, "crystal_solute_mass_fraction")
        if xC > 1:
            raise ValueError("crystal_solute_mass_fraction no puede superar 1")
    sol = _Solubility(solubility)
    s_info = sol.describe(Tf)
    s = s_info["kg_solute_per_kg_solvent"]
    xS = s / (1.0 + s)
    if not s_info["tabulated_point"]:
        assumptions.append(f"solubilidad a la temperatura final INTERPOLADA ({s_info['interpolation']}) entre los puntos de "
                           "la tabla")
        if s_info["linear_vs_pchip_rel_diff"] > 0.02:
            warnings.append("la interpolación lineal y la cúbica de la solubilidad difieren más del 2 %: da más puntos "
                            "cerca de la temperatura final")
    if xC <= xS:
        raise ValueError(f"el cristal ({xC:.4f} de soluto) no es más rico en soluto que las aguas madres saturadas ({xS:.4f}): "
                         "con estos datos no hay cristalización posible (revisa el hidrato o la solubilidad)")
    # evaporation
    mode = _need(evaporation, "mode", "'none', 'mass', 'fraction_of_feed', 'fraction_of_feed_solvent' o 'adiabatic'",
                 "evaporation")
    solvent_in = F * (1.0 - xF)
    if mode == "none":
        W = 0.0
    elif mode == "mass":
        W = _positive(_need(evaporation, "solvent_evaporated_kg", "disolvente evaporado", "evaporation"),
                      "solvent_evaporated_kg", allow_zero=True)
    elif mode == "fraction_of_feed":
        W = F * _positive(_need(evaporation, "fraction", "fracción de la masa de alimentación que se evapora", "evaporation"),
                          "fraction", allow_zero=True)
    elif mode == "fraction_of_feed_solvent":
        W = solvent_in * _positive(_need(evaporation, "fraction", "fracción del disolvente de la alimentación que se evapora",
                                         "evaporation"), "fraction", allow_zero=True)
    elif mode == "adiabatic":
        W = None
    else:
        raise ValueError("evaporation['mode'] debe ser 'none', 'mass', 'fraction_of_feed', 'fraction_of_feed_solvent' o "
                         "'adiabatic'")
    # energy data
    heat = None
    if mode == "adiabatic" and energy is None:
        raise ValueError("falta energy: el cristalizador adiabático (vacío) resuelve la evaporación con el balance de "
                         "entalpía y necesita feed_T_K, solution_cp_J_kgK y el calor de cristalización")
    if energy is not None:
        TF = _positive(_need(energy, "feed_T_K", "temperatura de la alimentación", "energy"), "feed_T_K")
        cp = _positive(_need(energy, "solution_cp_J_kgK", "calor específico medio de la solución de alimentación", "energy"),
                       "solution_cp_J_kgK")
        dHc, dHs = energy.get("heat_of_crystallization_J_kg"), energy.get("heat_of_solution_J_mol")
        if (dHc is None) == (dHs is None):
            raise ValueError("falta energy['heat_of_crystallization_J_kg'] (por kg de cristal tal como cristaliza; positivo "
                             "= calor liberado) o energy['heat_of_solution_J_mol'] (por mol de cristal; positivo = "
                             "disolución endotérmica): uno solo, no hay valor por defecto")
        if dHs is not None:
            Mc = energy.get("crystal_molar_mass_kg_mol") or M_crystal
            if Mc is None:
                raise ValueError("falta energy['crystal_molar_mass_kg_mol'] para pasar el calor de disolución molar a kg de "
                                 "cristal (o da hydrate)")
            dHc = float(dHs) / _positive(Mc, "crystal_molar_mass_kg_mol")
            assumptions.append("calor de cristalización = −calor de disolución (el dato del usuario, a su temperatura de "
                               "referencia, se usa a la temperatura final)")
        dHc = float(dHc)
        lam = energy.get("solvent_latent_heat_J_kg")
        needs_lam = mode == "adiabatic" or (W or 0.0) > 0
        if lam is None and needs_lam:
            if solvent != "water":
                raise ValueError("falta energy['solvent_latent_heat_J_kg'] (o declara solvent='water' para tomar el calor "
                                 "latente del agua de IAPWS a la temperatura final)")
            lam = _water("lambda", Tf)
            assumptions.append("calor latente del agua pura a la temperatura final (IAPWS-95, CoolProp); el vapor sale a "
                               "esa temperatura")
        lam = _positive(lam, "solvent_latent_heat_J_kg") if lam is not None else 0.0
        sensible = F * cp * (TF - Tf)
        if mode == "adiabatic":
            C0 = (F * xF - F * xS) / (xC - xS)               # crystals without evaporation; each kg evaporated adds `slope`
            slope = xS / (xC - xS)
            if lam <= dHc * slope:
                raise ValueError("el calor de cristalización por kg de disolvente evaporado supera al calor latente: el "
                                 "balance adiabático no tiene solución estable con estos datos")
            W = (sensible + C0 * dHc) / (lam - dHc * slope)
            if C0 + W * slope <= 0:                           # no crystals: only flash cooling of the feed
                W = sensible / lam
            if W < 0:
                raise ValueError("el balance adiabático da evaporación negativa: la alimentación entra más fría que la "
                                 "temperatura final; no es un cristalizador adiabático de enfriamiento por evaporación")
        heat = {"feed_T_K": TF, "solution_cp_J_kgK": cp, "heat_of_crystallization_J_kg": dHc,
                "solvent_latent_heat_J_kg": lam, "sensible_heat_of_feed_J": sensible}
        assumptions.append("balance de entalpía con referencia en la temperatura final: calor a retirar = F·cp·(T_alim − "
                           "T_final) + cristales·ΔH_crist − evaporado·λ (cp de la alimentación constante)")
    if W > solvent_in:
        raise ValueError(f"se evapora más disolvente ({W:.4g} kg) del que entra ({solvent_in:.4g} kg)")
    # mass balance: F = S + C + W ; F xF = S xS + C xC
    C = (F * xF - (F - W) * xS) / (xC - xS)
    saturated = True
    if C <= 0:
        C, saturated = 0.0, False
        warnings.append("la alimentación no llega a saturación a la temperatura final: no cristaliza nada (aguas madres "
                        "subsaturadas)")
    S = F - W - C
    if S < -1e-12 * F:
        raise ValueError("el balance pide más soluto y disolvente en los cristales de los que hay: con esa evaporación la "
                         "masa se seca por completo; reduce la evaporación")
    solute_ML = F * xF - C * xC
    solvent_ML = S - solute_ML
    out: dict = {
        "basis": "por la masa de alimentación dada (si es kg/s, lee kg como kg/s y J como W)",
        "feed": {"mass_kg": F, "solute_mass_fraction": xF, "solute_kg": F * xF, "solvent_kg": solvent_in},
        "solubility_at_final_T": s_info, "crystals_kg": C, "mother_liquor_kg": S, "solvent_evaporated_kg": W,
        "crystal": {"solute_mass_fraction": xC, "anhydrous_solute_kg": C * xC, "solvent_of_crystallization_kg": C * (1 - xC),
                    "molar_mass_kg_mol": M_crystal},
        "mother_liquor": {"solute_kg": solute_ML, "solvent_kg": solvent_ML,
                          "solute_mass_fraction": (solute_ML / S if S > 0 else None),
                          "kg_solute_per_kg_solvent": (solute_ML / solvent_ML if solvent_ML > 0 else None),
                          "saturated": saturated},
        "yield": {"solute_recovered_fraction": C * xC / (F * xF), "crystals_per_kg_feed": C / F}}
    evidence = {"total_mass_balance_rel_error": abs(F - S - C - W) / F,
                "solute_balance_rel_error": abs(F * xF - solute_ML - C * xC) / (F * xF),
                "solvent_balance_rel_error": abs(solvent_in - solvent_ML - C * (1 - xC) - W) / solvent_in,
                "mother_liquor_saturation_rel_error": (abs(solute_ML / solvent_ML - s) / s if saturated and solvent_ML > 0
                                                       else 0.0)}
    reasons = []
    if max(evidence.values()) > BALANCE_TOL:
        reasons.append("los balances de masa no cierran")
    if sol.covers(heat["feed_T_K"] if heat else Tf) and heat:
        s_feed = sol(heat["feed_T_K"])
        out["feed"]["saturation_ratio_at_feed_T"] = (xF / (1 - xF)) / s_feed
        if xF / (1 - xF) > s_feed * (1 + 1e-9):
            warnings.append("la alimentación ya está sobresaturada a su temperatura según la tabla: puede traer sólidos")
    if heat is not None:
        q = heat["sensible_heat_of_feed_J"] + C * heat["heat_of_crystallization_J_kg"] - W * heat["solvent_latent_heat_J_kg"]
        heat.update(heat_of_crystallization_released_J=C * heat["heat_of_crystallization_J_kg"],
                    latent_heat_of_evaporation_J=W * heat["solvent_latent_heat_J_kg"], heat_to_remove_J=q)
        out["heat"] = heat
        evidence["enthalpy_balance_rel_error"] = (abs(q) / max(abs(heat["sensible_heat_of_feed_J"]),
                                                               abs(C * heat["heat_of_crystallization_J_kg"]), 1e-30)
                                                  if mode == "adiabatic" else 0.0)
        if evidence["enthalpy_balance_rel_error"] > BALANCE_TOL:
            reasons.append("el balance de entalpía adiabático no cierra")
        if mode == "adiabatic":
            heat["heat_to_remove_J"] = 0.0
        elif q < 0:
            warnings.append("el calor a retirar es negativo: hay que APORTAR calor (la evaporación pide más que lo que dan "
                            "el enfriamiento y la cristalización)")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method="Balances de masa total y de soluto con aguas madres saturadas (solución algebraica exacta); balance "
                      "de entalpía con referencia en la temperatura final; solubilidad interpolada con SciPy PCHIP")
    return _seal(out, evidence, reasons)


# ---------------------------------------------------------------------------------------------------------------- MSMPR
MSMPR_ASSUMPTIONS = ["MSMPR ideal: mezcla perfecta, producto con la misma distribución que el tanque, estado estacionario, "
                     "alimentación sin cristales, sin rotura ni aglomeración",
                     "crecimiento independiente del tamaño (ley de McCabe) y núcleos de tamaño cero",
                     "densidad de población por m³ de suspensión (usa la misma base para la densidad de magma)"]
MSMPR_NOT_INCLUDED = ["retirada de finos, clasificación del producto y lecho fluidizado", "dispersión de la velocidad de "
                      "crecimiento y crecimiento dependiente del tamaño", "aglomeración y rotura",
                      "dinámica y oscilaciones del cristalizador"]


def _msmpr_moment(k: int, n0: float, Gt: float) -> float:
    return n0 * math.factorial(k) * Gt ** (k + 1)


def msmpr(crystal_density_kg_m3: float, volume_shape_factor: float, residence_time_s: float,
          growth_rate_m_s: float | None = None, nucleation_rate_m3_s: float | None = None,
          magma_density_kg_m3: float | None = None, nucleation_law: dict | None = None,
          suspension_volume_m3: float | None = None, sizes_m: list[float] | None = None) -> dict:
    """Steady MSMPR crystallizer from two of (G, B0, magma density), or from the magma density and the nucleation law
    B0 = k·G^i·MT^j. Analytical exponential population density; moments also integrated numerically as evidence."""
    rho = _positive(crystal_density_kg_m3, "crystal_density_kg_m3")
    kv = _positive(volume_shape_factor, "volume_shape_factor")
    tau = _positive(residence_time_s, "residence_time_s")
    warnings: list[str] = []
    assumptions = list(MSMPR_ASSUMPTIONS)
    given = [k for k, v in (("growth_rate_m_s", growth_rate_m_s), ("nucleation_rate_m3_s", nucleation_rate_m3_s),
                            ("magma_density_kg_m3", magma_density_kg_m3), ("nucleation_law", nucleation_law)) if v is not None]
    if sorted(given) == ["growth_rate_m_s", "nucleation_rate_m3_s"]:
        G, B0 = _positive(growth_rate_m_s, "growth_rate_m_s"), _positive(nucleation_rate_m3_s, "nucleation_rate_m3_s")
        n0 = B0 / G
        MT = 6.0 * kv * rho * n0 * (G * tau) ** 4
        route = "G y B0 dados → densidad de magma"
    elif sorted(given) == ["growth_rate_m_s", "magma_density_kg_m3"]:
        G, MT = _positive(growth_rate_m_s, "growth_rate_m_s"), _positive(magma_density_kg_m3, "magma_density_kg_m3")
        n0 = MT / (6.0 * kv * rho * (G * tau) ** 4)
        B0 = n0 * G
        route = "G y densidad de magma dados → B0 que cierra el balance de masa"
    elif sorted(given) == ["magma_density_kg_m3", "nucleation_rate_m3_s"]:
        B0 = _positive(nucleation_rate_m3_s, "nucleation_rate_m3_s")
        MT = _positive(magma_density_kg_m3, "magma_density_kg_m3")
        G = (MT / (6.0 * kv * rho * B0 * tau ** 4)) ** (1.0 / 3.0)
        n0 = B0 / G
        route = "B0 y densidad de magma dados → G que cierra el balance de masa"
    elif sorted(given) == ["magma_density_kg_m3", "nucleation_law"]:
        MT = _positive(magma_density_kg_m3, "magma_density_kg_m3")
        k = _positive(_need(nucleation_law, "k", "constante de B0 = k·G^i·MT^j en SI (G m/s, MT kg/m³, B0 núcleos/m³·s)",
                            "nucleation_law"), "nucleation_law['k']")
        i = float(_need(nucleation_law, "growth_exponent", "exponente i de G", "nucleation_law"))
        j = float(_need(nucleation_law, "magma_exponent", "exponente j de la densidad de magma (0 si no depende)",
                        "nucleation_law"))
        if i + 3.0 <= 0:
            raise ValueError("nucleation_law['growth_exponent'] debe ser mayor que −3 para que exista solución")
        G = (MT ** (1.0 - j) / (6.0 * kv * rho * k * tau ** 4)) ** (1.0 / (i + 3.0))
        B0 = k * G ** i * MT ** j
        n0 = B0 / G
        route = "densidad de magma y ley B0 = k·G^i·MT^j → G y B0 (solución cerrada del balance de masa)"
        assumptions.append("ley de nucleación empírica del usuario B0 = k·G^i·MT^j (correlación de potencias)")
    else:
        raise ValueError("MSMPR: da dos de growth_rate_m_s (G), nucleation_rate_m3_s (B0) y magma_density_kg_m3, o bien "
                         "magma_density_kg_m3 con nucleation_law {'k', 'growth_exponent', 'magma_exponent'}; se recibió: "
                         + (", ".join(given) or "nada"))
    Gt = G * tau
    moments = [_msmpr_moment(k, n0, Gt) for k in range(6)]
    numeric = [n0 * Gt ** (k + 1) * quad(lambda z, k=k: z ** k * math.exp(-z), 0.0, math.inf, epsabs=0.0, epsrel=1e-12)[0]
               for k in range(5)]
    z50 = float(gammaincinv(4.0, 0.5))
    out: dict = {
        "route": route, "growth_rate_m_s": G, "nucleation_rate_m3_s": B0, "nuclei_population_density_m4": n0,
        "residence_time_s": tau, "characteristic_size_G_tau_m": Gt, "magma_density_kg_m3": MT,
        "moments": {f"mu{k}": moments[k] for k in range(5)},
        "moment_units": "mu_k en m^k por m³ de suspensión (mu0 = número, mu1 = longitud, mu2·ka = área, mu3·kv = volumen)",
        "sizes": {"number_mean_m": Gt, "dominant_mass_m": 3.0 * Gt, "mass_median_m": z50 * Gt, "sauter_mean_m": 3.0 * Gt,
                  "mass_mean_m": 4.0 * Gt},
        "coefficient_of_variation": {"mass_weighted": math.sqrt(moments[5] * moments[3] / moments[4] ** 2 - 1.0),
                                     "number_weighted": math.sqrt(moments[2] * moments[0] / moments[1] ** 2 - 1.0)},
        "crystal_number_m3": moments[0], "production_rate_per_volume_kg_m3s": MT / tau}
    if suspension_volume_m3 is not None:
        V = _positive(suspension_volume_m3, "suspension_volume_m3")
        out["suspension_volume_m3"] = V
        out["volumetric_flow_m3_s"] = V / tau
        out["production_rate_kg_s"] = MT * V / tau
    grid = sorted(_positive(x, "sizes_m", allow_zero=True) for x in sizes_m) if sizes_m else [Gt * z for z in
                                                                                             (0.5, 1, 2, 3, z50, 5, 7, 10)]
    out["distribution"] = [{"size_m": L, "population_density_m4": n0 * math.exp(-L / Gt),
                            "cumulative_mass_undersize": float(gammainc(4.0, L / Gt)),
                            "cumulative_number_undersize": 1.0 - math.exp(-L / Gt)} for L in grid]
    evidence = {"moments_numeric_vs_analytic_max_rel_diff": max(abs(a - b) / b for a, b in zip(numeric, moments)),
                "magma_density_from_third_moment_rel_error": abs(rho * kv * numeric[3] - MT) / MT,
                "nucleation_identity_rel_error": abs(n0 * G - B0) / B0,
                "mass_weighted_cv_minus_half": abs(out["coefficient_of_variation"]["mass_weighted"] - 0.5)}
    reasons = []
    if max(evidence.values()) > MOMENT_TOL:
        reasons.append("los momentos numéricos no coinciden con la solución analítica")
    out.update(warnings=warnings, assumptions=assumptions, not_included=list(MSMPR_NOT_INCLUDED),
               method="Balance de población MSMPR: n(L) = n0·exp(−L/Gτ), n0 = B0/G, MT = 6·kv·ρc·n0·(Gτ)^4 "
                      "(Randolph-Larson); momentos comprobados con cuadratura de SciPy (QUADPACK)")
    return _seal(out, evidence, reasons)


def _cut_fractions(Gt: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return gammainc(4.0, upper / Gt) - gammainc(4.0, lower / Gt)


def _ln_n_regression(L: np.ndarray, ln_n: np.ndarray) -> dict:
    m = len(L)
    A = np.vstack([L, np.ones(m)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ln_n, rcond=None)
    resid = ln_n - (slope * L + intercept)
    ss_res, ss_tot = float(resid @ resid), float(((ln_n - ln_n.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    se_slope = math.sqrt(ss_res / (m - 2) / float(((L - L.mean()) ** 2).sum())) if m > 2 else None
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r2, "slope_standard_error": se_slope,
            "residuals": [float(x) for x in resid]}


def msmpr_from_sieve(crystal_density_kg_m3: float, volume_shape_factor: float, residence_time_s: float,
                     magma_density_kg_m3: float, sieve: dict) -> dict:
    """Growth rate G and nucleation rate B0 of an MSMPR crystallizer from a screen analysis of its product."""
    rho = _positive(crystal_density_kg_m3, "crystal_density_kg_m3")
    kv = _positive(volume_shape_factor, "volume_shape_factor")
    tau = _positive(residence_time_s, "residence_time_s")
    MT = _positive(magma_density_kg_m3, "magma_density_kg_m3")
    lo = _need(sieve, "lower_m", "abertura inferior de cada fracción (m); None para el fondo", "sieve")
    up = _need(sieve, "upper_m", "abertura superior de cada fracción (m); None para la fracción más gruesa abierta", "sieve")
    w = _need(sieve, "mass_fraction", "fracción másica retenida en cada corte", "sieve")
    if not len(lo) == len(up) == len(w):
        raise ValueError("sieve['lower_m'], ['upper_m'] y ['mass_fraction'] deben tener la misma longitud")
    total = sum(float(x) for x in w)
    warnings: list[str] = []
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"las fracciones másicas suman {total:.4f}, no 1: da fracciones (0-1), no porcentajes, incluidas "
                         "las fracciones abiertas (fondo y gruesos)")
    if abs(total - 1.0) > 1e-6:
        warnings.append(f"las fracciones másicas suman {total:.4f}; se usan tal cual, sin renormalizar")
    rows, skipped = [], []
    for a, b, x in zip(lo, up, w):
        x = _positive(x, "mass_fraction", allow_zero=True)
        if a is None or b is None or float(a) <= 0 or x <= 0:
            skipped.append({"lower_m": a, "upper_m": b, "mass_fraction": x,
                            "reason": "fracción abierta o vacía: no define ΔL ni tamaño medio"})
            continue
        a, b = float(a), float(b)
        if b <= a:
            raise ValueError("cada fracción necesita upper_m > lower_m")
        rows.append((a, b, x))
    if len(rows) < 3:
        raise ValueError("hacen falta al menos 3 fracciones cerradas (con abertura inferior y superior y masa > 0) para la "
                         "regresión de ln n frente a L")
    rows.sort()
    lower, upper, wf = (np.array([r[i] for r in rows]) for i in range(3))
    L_avg, dL = 0.5 * (lower + upper), upper - lower
    n = MT * wf / (rho * kv * L_avg ** 3 * dL)
    reg = _ln_n_regression(L_avg, np.log(n))
    if reg["slope"] >= 0:
        raise ValueError("ln n no decrece con el tamaño: los datos no corresponden a un MSMPR (pendiente ≥ 0)")
    G = -1.0 / (reg["slope"] * tau)
    n0 = math.exp(reg["intercept"])
    B0 = n0 * G
    Gt = G * tau
    # bias of the mid-size approximation: apply the same regression to exact MSMPR cut fractions of the fitted crystallizer
    MT_fit = 6.0 * kv * rho * n0 * Gt ** 4
    w_model = _cut_fractions(Gt, lower, upper)
    n_model = MT_fit * w_model / (rho * kv * L_avg ** 3 * dL)
    reg_b = _ln_n_regression(L_avg, np.log(n_model))
    G_b, n0_b = -1.0 / (reg_b["slope"] * tau), math.exp(reg_b["intercept"])
    # exact-cut fit: least squares of the measured cut fractions against the MSMPR mass distribution (one parameter, Gτ)
    res = minimize_scalar(lambda u: float(((_cut_fractions(math.exp(u), lower, upper) - wf) ** 2).sum()),
                          bounds=(math.log(Gt) - 3.0, math.log(Gt) + 3.0), method="bounded", options={"xatol": 1e-13})
    Gt_x = math.exp(res.x)
    w_x = _cut_fractions(Gt_x, lower, upper)
    n0_x = MT / (6.0 * kv * rho * Gt_x ** 4)
    assumptions = list(MSMPR_ASSUMPTIONS) + [
        "regresión de los libros: n_i = MT·w_i/(ρc·kv·L̄_i³·ΔL_i) con L̄ la media aritmética de las aberturas; ln n = ln n0 "
        "− L/(Gτ) por mínimos cuadrados ordinarios",
        "tamaño de tamiz tomado como tamaño característico L; el factor de forma kv es el del usuario"]
    ratio = MT_fit / MT
    if reg["r_squared"] < R2_WARN:
        warnings.append(f"R² = {reg['r_squared']:.3f} < {R2_WARN}: la distribución no es exponencial; el cristalizador no "
                        "se comporta como MSMPR (clasificación, finos, crecimiento dependiente del tamaño)")
    if not 0.8 <= ratio <= 1.25:
        warnings.append(f"la densidad de magma que implican n0 y G ajustados es {ratio:.2f} veces la dada: el ajuste no "
                        "cierra el balance de masa (revisa kv, la densidad de magma o las fracciones abiertas)")
    rms = float(math.sqrt(((w_x - wf) ** 2).mean()))
    if rms > MASS_MISFIT_WARN:
        warnings.append("las fracciones másicas se apartan de la distribución MSMPR que mejor las ajusta (residuo RMS "
                        f"{rms:.3f} por corte): el producto no es el de un MSMPR ideal aunque ln n frente a L parezca recto "
                        "(R² es poco sensible porque n lleva L³ en el denominador)")
    if skipped:
        warnings.append(f"{len(skipped)} fracción(es) abierta(s) o vacía(s) quedan fuera de la regresión")
    out: dict = {
        "growth_rate_m_s": G, "nucleation_rate_m3_s": B0, "nuclei_population_density_m4": n0,
        "characteristic_size_G_tau_m": Gt, "dominant_mass_size_m": 3.0 * Gt,
        "mass_median_size_m": float(gammaincinv(4.0, 0.5)) * Gt,
        "regression": {**reg, "points": len(rows),
                       "table": [{"lower_m": float(a), "upper_m": float(b), "mean_size_m": float(l), "mass_fraction": float(x),
                                  "population_density_m4": float(v), "ln_n": float(math.log(v))}
                                 for a, b, l, x, v in zip(lower, upper, L_avg, wf, n)]},
        "magma_density_implied_by_fit_kg_m3": MT_fit, "magma_density_given_kg_m3": MT,
        "magma_density_ratio_fit_to_given": ratio,
        "mid_size_bias": {"growth_rate_rel": G_b / G - 1.0, "nuclei_density_rel": n0_b / n0 - 1.0,
                          "nucleation_rate_rel": (n0_b * G_b) / B0 - 1.0,
                          "meaning": "error relativo que la aproximación de tamaño medio introduce con ESTOS cortes en un "
                                     "MSMPR exacto con los parámetros ajustados"},
        "exact_cut_fit": {"growth_rate_m_s": Gt_x / tau, "nuclei_population_density_m4": n0_x,
                          "nucleation_rate_m3_s": n0_x * Gt_x / tau,
                          "rms_mass_fraction_residual": rms,
                          "meaning": "ajuste por mínimos cuadrados de las fracciones másicas de cada corte a la "
                                     "distribución MSMPR exacta (un parámetro, Gτ); n0 del balance de masa con la "
                                     "densidad de magma dada"},
        "skipped_fractions": skipped}
    evidence = {"regression_points": len(rows), "r_squared": reg["r_squared"],
                "fit_normal_equations_residual": abs(float(np.sum(reg["residuals"]))) / len(rows),
                "exact_cut_fit_converged": bool(res.success),
                "growth_rate_regression_vs_exact_cut_rel_diff": abs(G - Gt_x / tau) / (Gt_x / tau)}
    reasons = []
    if evidence["fit_normal_equations_residual"] > 1e-9:
        reasons.append("la regresión no satisface sus ecuaciones normales")
    if not res.success:
        reasons.append("el ajuste exacto de cortes no convergió")
    out.update(warnings=warnings, assumptions=assumptions, not_included=list(MSMPR_NOT_INCLUDED),
               method="Regresión lineal de ln n frente a L (NumPy lstsq) con n de las fracciones de tamiz; ajuste exacto de "
                      "cortes con la función gamma incompleta (SciPy) y minimize_scalar")
    return _seal(out, evidence, reasons)


# ----------------------------------------------------------------------------------------- seeded batch cooling (moments)
def batch_seeded_cooling(crystal_density_kg_m3: float, volume_shape_factor: float, batch: dict) -> dict:
    """Seeded batch cooling crystallization by the method of moments, per kg of solvent. Anhydrous crystals,
    size-independent growth G = kg·ΔC^g, optional nucleation B = kb·ΔC^b·M^j (nuclei of zero size), ΔC = C − C*(T) in kg/kg."""
    rho = _positive(crystal_density_kg_m3, "crystal_density_kg_m3")
    kv = _positive(volume_shape_factor, "volume_shape_factor")
    where = "batch"
    sol = _Solubility(_need(batch, "solubility", "tabla de solubilidad {'T_K', 'kg_solute_per_kg_solvent'}", where),
                      "batch['solubility']")
    c0 = _positive(_need(batch, "initial_concentration_kg_kg", "concentración inicial, kg de soluto por kg de disolvente",
                         where), "initial_concentration_kg_kg")
    cooling = _need(batch, "cooling", "perfil de enfriamiento {'time_s': [...], 'T_K': [...]}", where)
    tt = [float(x) for x in _need(cooling, "time_s", "tiempos del perfil de enfriamiento", "batch['cooling']")]
    TT = [float(x) for x in _need(cooling, "T_K", "temperaturas del perfil de enfriamiento", "batch['cooling']")]
    if len(tt) != len(TT) or len(tt) < 2 or tt[0] != 0.0 or any(b <= a for a, b in zip(tt, tt[1:])):
        raise ValueError("batch['cooling']: 'time_s' debe empezar en 0, ser creciente y tener la misma longitud que 'T_K' "
                         "(≥ 2 puntos)")
    if not all(sol.covers(T) for T in TT):
        raise ValueError("el perfil de enfriamiento sale del intervalo de la tabla de solubilidad: no se extrapola")
    growth = _need(batch, "growth", "cinética de crecimiento {'k_m_s', 'order'}: G = k·ΔC^order", where)
    kg = _positive(_need(growth, "k_m_s", "constante de crecimiento (m/s por (kg/kg)^order)", "batch['growth']"), "k_m_s")
    g = _positive(_need(growth, "order", "orden de crecimiento en la sobresaturación", "batch['growth']"), "order",
                  allow_zero=True)
    if "nucleation" not in batch:
        raise ValueError("falta batch['nucleation']: da {'k_per_kg_s', 'order', 'magma_exponent'} (B = k·ΔC^order·M^j, "
                         "núcleos por kg de disolvente y s) o None si se desprecia la nucleación; no se supone")
    nuc = batch["nucleation"]
    kb = b = j = 0.0
    if nuc:
        kb = _positive(_need(nuc, "k_per_kg_s", "constante de nucleación", "batch['nucleation']"), "k_per_kg_s",
                       allow_zero=True)
        b = _positive(_need(nuc, "order", "orden de nucleación en la sobresaturación", "batch['nucleation']"), "order",
                      allow_zero=True)
        j = _positive(_need(nuc, "magma_exponent", "exponente de la masa de cristales (0 si no depende)",
                            "batch['nucleation']"), "magma_exponent", allow_zero=True)
    ms = _positive(_need(batch, "seed_mass_kg_per_kg_solvent", "masa de siembra por kg de disolvente (0 sin siembra)", where),
                   "seed_mass_kg_per_kg_solvent", allow_zero=True)
    Ls = (_positive(_need(batch, "seed_size_m", "tamaño de la siembra (monodispersa)", where), "seed_size_m") if ms > 0
          else None)
    if ms == 0 and kb == 0:
        raise ValueError("sin siembra y sin nucleación no cristaliza nada: da seed_mass_kg_per_kg_solvent > 0 o una "
                         "cinética de nucleación")
    Lref = Ls if Ls else 1e-4
    N_seed = ms / (rho * kv * Ls ** 3) if ms > 0 else 0.0
    Nref = N_seed if N_seed > 0 else 1.0 / (rho * kv * Lref ** 3)
    mass_unit = rho * kv * Nref * Lref ** 3                # kg crystal per kg solvent represented by y3 = 1

    def temperature(t: float) -> float:
        return float(np.interp(t, tt, TT))

    def kinetics(c: float, T: float, y3: float) -> tuple[float, float]:
        dc = c - sol(T)
        if dc <= 0:
            return 0.0, 0.0
        G = kg * dc ** g
        B = kb * dc ** b * (mass_unit * max(y3, 0.0)) ** j if kb > 0 else 0.0
        return G, B

    def rhs(t: float, y: np.ndarray) -> list[float]:
        G, B = kinetics(y[6], temperature(t), y[3])
        a = G / Lref
        return [B / Nref, a * y[0], 2 * a * y[1], 3 * a * y[2], 4 * a * y[3], 5 * a * y[4], -3.0 * mass_unit * a * y[2], a]

    seeded = 1.0 if N_seed > 0 else 0.0
    y0 = [seeded] * 6 + [c0, 0.0]                             # y0..y5 scaled moments, y6 = C, y7 = ∫G dt / Lref
    t_end = tt[-1]

    def integrate(rtol: float):
        return solve_ivp(rhs, (0.0, t_end), y0, method="LSODA", rtol=rtol, atol=1e-13, t_eval=np.linspace(0.0, t_end, 41),
                         max_step=min(t_end / 200.0, min(b_ - a_ for a_, b_ in zip(tt, tt[1:]))))

    fine, coarse = integrate(1e-10), integrate(1e-7)
    yf, yc = fine.y[:, -1], coarse.y[:, -1]
    c_end, T_end = float(yf[6]), TT[-1]
    mass_end = mass_unit * float(yf[3])
    seed_size_end = (Ls + Lref * float(yf[7])) if ms > 0 else None
    seed_mass_end = (rho * kv * N_seed * seed_size_end ** 3) if ms > 0 else 0.0
    equilibrium_mass = ms + c0 - sol(T_end)
    mom = [float(yf[k]) * Nref * Lref ** k for k in range(6)]
    profile = []
    for i in range(0, 41, 4):
        t, c = float(fine.t[i]), float(fine.y[6, i])
        T = temperature(t)
        G, B = kinetics(c, T, float(fine.y[3, i]))
        profile.append({"time_s": t, "T_K": T, "concentration_kg_kg": c, "solubility_kg_kg": sol(T),
                        "supersaturation_kg_kg": c - sol(T), "growth_rate_m_s": G, "nucleation_rate_per_kg_s": B,
                        "crystal_mass_kg_per_kg_solvent": mass_unit * float(fine.y[3, i])})
    warnings: list[str] = []
    if c0 < sol(TT[0]):
        warnings.append("la solución inicial está subsaturada: la siembra se disolvería al principio; el modelo no disuelve "
                        "(G = 0 mientras C < C*)")
    max_S = max(p["supersaturation_kg_kg"] for p in profile)
    out: dict = {
        "basis": "por kg de disolvente",
        "final": {"T_K": T_end, "concentration_kg_kg": c_end, "solubility_kg_kg": sol(T_end),
                  "supersaturation_kg_kg": c_end - sol(T_end), "crystal_mass_kg_per_kg_solvent": mass_end,
                  "crystallized_kg_per_kg_solvent": c0 - c_end, "equilibrium_crystal_mass_kg_per_kg_solvent": equilibrium_mass,
                  "approach_to_equilibrium_yield": ((c0 - c_end) / (c0 - sol(T_end)) if c0 > sol(T_end) else None),
                  "crystal_number_per_kg_solvent": mom[0],
                  "number_mean_size_m": (mom[1] / mom[0] if mom[0] > 0 else None),
                  "mass_mean_size_m": (mom[4] / mom[3] if mom[3] > 0 else None),
                  "mass_weighted_cv": (math.sqrt(max(mom[5] * mom[3] / mom[4] ** 2 - 1.0, 0.0)) if mom[4] > 0 else None),
                  "seed_final_size_m": seed_size_end, "seed_grown_mass_kg_per_kg_solvent": seed_mass_end,
                  "nucleated_mass_fraction": ((mass_end - seed_mass_end) / mass_end if mass_end > 0 else None)},
        "moments": {f"mu{k}": mom[k] for k in range(6)},
        "moment_units": "mu_k en m^k por kg de disolvente", "seed_number_per_kg_solvent": N_seed,
        "max_supersaturation_kg_kg": max_S, "profile": profile}
    evidence = {"solver_success": bool(fine.success and coarse.success),
                "solute_balance_rel_error": abs((c0 + ms) - (c_end + mass_end)) / (c0 + ms),
                "step_independence_rel_diff": max(abs(float(yf[k]) - float(yc[k])) / max(abs(float(yf[k])), 1e-300)
                                                  for k in (3, 6)),
                "seed_plus_nuclei_mass_consistent": bool(mass_end - seed_mass_end >= -1e-9 * max(mass_end, 1e-300))}
    reasons = []
    if not evidence["solver_success"]:
        reasons.append("el integrador (LSODA) no terminó: " + str(fine.message))
    if evidence["solute_balance_rel_error"] > 1e-7:
        reasons.append("el balance de soluto no cierra")
    if evidence["step_independence_rel_diff"] > STEP_TOL:
        reasons.append("el resultado cambia con la tolerancia del integrador (no hay independencia del paso)")
    if not evidence["seed_plus_nuclei_mass_consistent"]:
        reasons.append("la masa crecida sobre la siembra supera la masa total de cristales")
    out.update(warnings=warnings,
               assumptions=["cristal anhidro (el disolvente no entra en el cristal); concentraciones por kg de disolvente",
                            "crecimiento independiente del tamaño G = k·ΔC^g con ΔC = C − C*(T) en kg/kg; G = 0 si ΔC ≤ 0 "
                            "(no se modela la disolución)", "núcleos de tamaño cero; siembra monodispersa; sin rotura ni "
                            "aglomeración; mezcla perfecta e isoterma en cada instante (la temperatura sigue el perfil dado)",
                            "perfil de enfriamiento interpolado linealmente entre los puntos del usuario; solubilidad "
                            "interpolada (PCHIP con ≥ 3 puntos)"],
               not_included=["hidratos/solvatos en el modelo por momentos", "balance de energía del lote (el perfil de "
                             "temperatura es un dato)", "distribución de tamaños completa (solo momentos; para la forma "
                             "hace falta resolver el balance de población)", "disolución, maduración de Ostwald, aglomeración"],
               method="Método de los momentos (μ0..μ5) acoplado al balance de soluto, integrado con SciPy solve_ivp (LSODA) "
                      "a dos tolerancias (1e-10 y 1e-7) para comprobar la independencia del paso")
    return _seal(out, evidence, reasons)
