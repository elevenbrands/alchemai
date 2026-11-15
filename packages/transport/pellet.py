"""Diffusion and reaction inside a porous catalyst pellet: effectiveness factor by solving the boundary-value problem.

What it answers: how much of the intrinsic rate a pellet really delivers (internal, external and overall effectiveness
factor), the Thiele modulus and the generalized (Bischoff/Aris/Petersen) modulus, the temperature rise inside the pellet and
across the external film, whether there are several steady states (Weisz-Hicks), and the classical diagnostic criteria
(Weisz-Prater, Mears, Anderson) evaluated from the solved rate.

Model (one reaction, one key reactant A, constant effective properties, 1-D slab / infinite cylinder / sphere, x = r/L):
    (1/x^s) d/dx(x^s dC/dx) = L² r(C, T) / De                 s = 0, 1, 2
    (1/x^s) d/dx(x^s dT/dx) = −L² (−ΔH) r(C, T) / λe          (only with heat of reaction and λe)
    x = 0: symmetry.   x = 1: C = Cb or De/L dC/dx = kc (Cb − C);  T = Tb or λe/L dT/dx = h (Tb − T)
    (heat film without λe: pellet uniform at Ts with h (Ts − Tb) = (−ΔH) · flux of A)
r is per unit PELLET volume: power law k C^n or Langmuir-Hinshelwood k C / (1 + K C)^m, Arrhenius in k (and van 't Hoff in
K when the adsorption enthalpy is given). Every physical datum comes from the caller.

Numerics: `scipy.integrate.solve_bvp` (4th-order collocation, adaptive mesh, singular term handled by scipy) on the coupled
C-T system, reached by continuation in the rate constant from the kinetic side («cold» start) and, when several steady
states are possible, also from the diffusion-limited side («hot» start). Without external films and with order ≥ 1 or
Langmuir-Hinshelwood kinetics, an independent shooting method from the pellet centre (the Weisz-Hicks construction: every
centre concentration gives one Thiele modulus) finds ALL steady states and cross-checks the collocation result.

Evidence → `verified` only when all hold: collocation converged; the fine-tolerance re-solve moves η by < MESH_TOL; the
rate integrated over the pellet equals the flux through the surface within FLUX_TOL; the Prater invariant
T + (−ΔH) De C / λe is constant inside the pellet within PRATER_TOL (coupled case); and, where one exists, the analytical
solution (first order, isothermal) or the shooting solution is matched within CHECK_TOL.

Sources read for the definitions (2026-11-14): Rawlings & Ekerdt, «Chemical Reactor Analysis and Design Fundamentals»,
chapter 7 slides (https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/ch7/slides-masswrxn.pdf): Thiele modulus with
a = Vp/Sp, generalized modulus Eq. 7.42/7.52, external film Eq. 7.53-7.58, Weisz-Hicks Eq. 7.65-7.66. Weisz-Prater
criterion (N_WP = ℜ Rp²/(Cs De) ≤ 0.3): https://en.wikipedia.org/wiki/Weisz–Prater_criterion (Vannice 2005). Mears (1971)
and Anderson (1963) criteria are the forms commonly quoted; they were not read from the original papers.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_bvp, solve_ivp
from scipy.optimize import brentq
from scipy.special import i0e, i1e

R_GAS = 8.314462618
SHAPES = {"slab": 0, "cylinder": 1, "sphere": 2}
KINETIC_TYPES = ("power_law", "langmuir_hinshelwood")
RATE_BASES = ("pellet_volume", "catalyst_mass")
REG_EPS = 1e-8            # smoothing of C^n for n < 1 (dead core): C (C² + (ε Cref)²)^((n−1)/2)
BVP_TOL = 1e-6            # collocation residual tolerance of the reported solution
BVP_TOL_FINE = 1e-8       # tolerance of the mesh-independence re-solve
BVP_MAX_NODES = 50000
MESH_TOL = 1e-4           # relative change of η between the two tolerances
FLUX_TOL = 1e-4           # |∫ r dV − surface flux| / surface flux
PRATER_TOL = 1e-4         # max deviation of the Prater invariant, relative to the temperature scale
CHECK_TOL = 1e-4          # agreement with the analytical solution or with the shooting method
BRANCH_TOL = 1e-3         # two converged solutions are different steady states when η differs by more than this
CONTINUATION_FACTOR = 4.0  # step in the rate constant during continuation (2 in Thiele modulus)
HOT_START_MODULUS = 30.0   # generalized modulus (bulk conditions) at which the hot continuation starts
SHOOT_GRID = 220           # centre values scanned by the shooting method when multiplicity is possible
DEAD_CORE_ORDER = 0.5      # power-law orders below this are solved by shooting from the dead-core front (isothermal only)
IVP_RTOL, IVP_RTOL_FINE = 1e-9, 1e-11
WEISZ_PRATER_LIMIT = 0.3   # N_WP ≤ 0.3: pore diffusion negligible (Vannice 2005, via the source above)
MEARS_LIMIT = 0.15         # Mears (1971): n ℜ R /(kc Cb) < 0.15 and |ΔH| ℜ R E /(h Tb² Rg) < 0.15
ANDERSON_LIMIT = 0.75      # Anderson (1963): |ΔH| ℜ R² E /(λe Ts² Rg) < 0.75
_GL_X, _GL_W = np.polynomial.legendre.leggauss(5)


# ---------------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------------
def _pos(d: dict, key: str, what: str, allow_zero: bool = False) -> float:
    v = d.get(key)
    if v is None:
        raise ValueError(f"falta {key}: {what}")
    v = float(v)
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{key} debe ser {'≥' if allow_zero else '>'} 0 ({what}); se recibió {v}")
    return v


def normalize_kinetics(kinetics: dict, pellet_density_kg_m3: float | None = None) -> dict:
    """Validate the kinetics dict and return it per unit pellet volume: {type, k_ref, T_ref, E, n | (K_ref, m, dH_ads)}.

    `k` is the rate constant at `reference_temperature_K` (needed with `activation_energy_J_mol`), or give
    `pre_exponential` + `activation_energy_J_mol` (k = A exp(−E/RT)). Units of k: (mol/m³)^(1−n) / s on a pellet-volume
    basis, or the same per kg of catalyst (m³ replaced by kg in the rate) with basis 'catalyst_mass'."""
    if not isinstance(kinetics, dict) or kinetics.get("type") not in KINETIC_TYPES:
        raise ValueError("kinetics debe ser un dict con 'type': 'power_law' (r = k·C^n) o 'langmuir_hinshelwood' "
                         "(r = k·C/(1 + K·C)^m); la velocidad intrínseca la da el usuario, no se supone")
    basis = kinetics.get("basis")
    if basis not in RATE_BASES:
        raise ValueError("falta kinetics['basis']: 'pellet_volume' (r en mol por m³ de pellet y segundo) o 'catalyst_mass' "
                         "(mol por kg de catalizador y segundo); un error de base cambia el resultado por la densidad")
    scale = 1.0
    if basis == "catalyst_mass":
        if pellet_density_kg_m3 is None or pellet_density_kg_m3 <= 0:
            raise ValueError("falta pellet_density_kg_m3: se necesita para pasar la velocidad por kg de catalizador a "
                             "velocidad por volumen de pellet")
        scale = float(pellet_density_kg_m3)
    E = kinetics.get("activation_energy_J_mol")
    E = None if E is None else float(E)
    if E is not None and E < 0:
        raise ValueError("activation_energy_J_mol debe ser ≥ 0")
    out = {"type": kinetics["type"], "basis": basis, "E": E}
    if kinetics.get("pre_exponential") is not None:
        if kinetics.get("k") is not None:
            raise ValueError("da kinetics['k'] (a una temperatura de referencia) o kinetics['pre_exponential'] (Arrhenius), "
                             "no ambos")
        if E is None:
            raise ValueError("falta kinetics['activation_energy_J_mol']: con pre_exponential la constante es "
                             "k = A·exp(−E/RT)")
        out.update(k_ref=_pos(kinetics, "pre_exponential", "factor preexponencial de Arrhenius") * scale, T_ref=math.inf)
    else:
        out["k_ref"] = _pos(kinetics, "k", "constante de velocidad intrínseca (sin limitaciones de transporte)") * scale
        T_ref = kinetics.get("reference_temperature_K")
        if E is not None and E > 0 and T_ref is None:
            raise ValueError("falta kinetics['reference_temperature_K']: temperatura a la que vale kinetics['k'], "
                             "necesaria para aplicar la energía de activación")
        out["T_ref"] = float(T_ref) if T_ref is not None else None
        if out["T_ref"] is not None and out["T_ref"] <= 0:
            raise ValueError("reference_temperature_K debe ser > 0")
    if kinetics["type"] == "power_law":
        n = kinetics.get("order")
        if n is None:
            raise ValueError("falta kinetics['order']: orden de reacción n de r = k·C^n")
        n = float(n)
        if n < 0:
            raise ValueError("kinetics['order'] debe ser ≥ 0 (los órdenes negativos se representan con "
                             "'langmuir_hinshelwood')")
        out["n"] = n
    else:
        out["K_ref"] = _pos(kinetics, "K_m3_mol", "constante de adsorción K de r = k·C/(1 + K·C)^m", allow_zero=True)
        m = kinetics.get("inhibition_exponent")
        if m is None:
            raise ValueError("falta kinetics['inhibition_exponent']: exponente m del denominador (1 + K·C)^m")
        out["m"] = float(m)
        if out["m"] < 0:
            raise ValueError("inhibition_exponent debe ser ≥ 0")
        dH = kinetics.get("adsorption_enthalpy_J_mol")
        out["dH_ads"] = None if dH is None else float(dH)
        if out["dH_ads"] is not None and out["T_ref"] in (None, math.inf):
            raise ValueError("adsorption_enthalpy_J_mol necesita kinetics['k'] y K_m3_mol dados a "
                             "reference_temperature_K")
    return out


def rate_constant(kin: dict, T):
    """k(T) per pellet volume. With no activation energy the constant does not depend on T."""
    if kin["E"] is None or kin["E"] == 0:
        return kin["k_ref"] * np.ones_like(np.asarray(T, dtype=float))
    T = np.asarray(T, dtype=float)
    inv_ref = 0.0 if kin["T_ref"] == math.inf else 1.0 / kin["T_ref"]
    return kin["k_ref"] * np.exp(-kin["E"] / R_GAS * (1.0 / T - inv_ref))


def adsorption_constant(kin: dict, T):
    T = np.asarray(T, dtype=float)
    if kin.get("dH_ads") is None:
        return kin["K_ref"] * np.ones_like(T)
    return kin["K_ref"] * np.exp(-kin["dH_ads"] / R_GAS * (1.0 / T - 1.0 / kin["T_ref"]))


def rate(kin: dict, C, T, C_scale: float):
    """Intrinsic rate per pellet volume, mol/(m³ s). Odd in C around 0 so a Newton iterate with C slightly < 0 is pushed
    back; for n < 1 the power is smoothed below REG_EPS · C_scale (dead-core front)."""
    C = np.asarray(C, dtype=float)
    k = rate_constant(kin, T)
    if kin["type"] == "power_law":
        n = kin["n"]
        if n == 1.0:
            return k * C
        if n > 1.0:
            return k * C * np.abs(C) ** (n - 1.0)
        return k * C * (C * C + (REG_EPS * C_scale) ** 2) ** ((n - 1.0) / 2.0)
    K = adsorption_constant(kin, T)
    return k * C / (1.0 + K * np.maximum(C, 0.0)) ** kin["m"]


def apparent_order(kin: dict, C: float, T: float) -> float:
    """d ln r / d ln C at (C, T): n for a power law, 1 − m K C/(1 + K C) for Langmuir-Hinshelwood."""
    if kin["type"] == "power_law":
        return kin["n"]
    KC = float(adsorption_constant(kin, T)) * C
    return 1.0 - kin["m"] * KC / (1.0 + KC)


def knudsen_diffusivity(pore_radius_m: float, T_K: float, molar_mass_kg_mol: float) -> float:
    """D_K = (2/3) r_pore (8 R T / (π M))^(1/2): kinetic theory for a straight cylindrical pore."""
    return 2.0 / 3.0 * pore_radius_m * math.sqrt(8.0 * R_GAS * T_K / (math.pi * molar_mass_kg_mol))


def effective_diffusivity_from_pores(pore: dict, T_K: float) -> dict:
    """De = (ε/τ) / (1/D_molecular + 1/D_Knudsen) (Bosanquet). Every entry of `pore` is the caller's."""
    if not isinstance(pore, dict):
        raise ValueError("pore debe ser un dict con porosity, tortuosity, mean_pore_radius_m, molecular_diffusivity_m2_s "
                         "y molar_mass_kg_mol")
    eps = _pos(pore, "porosity", "porosidad interna del pellet, para estimar la difusividad efectiva")
    tau = _pos(pore, "tortuosity", "tortuosidad del pellet, para estimar la difusividad efectiva (no hay valor típico)")
    rp = _pos(pore, "mean_pore_radius_m", "radio medio de poro, para la difusividad de Knudsen")
    Dm = _pos(pore, "molecular_diffusivity_m2_s", "difusividad molecular del reactivo en el fluido de los poros")
    M = _pos(pore, "molar_mass_kg_mol", "masa molar del reactivo, para la difusividad de Knudsen")
    if eps >= 1:
        raise ValueError("porosity debe estar entre 0 y 1")
    DK = knudsen_diffusivity(rp, T_K, M)
    Dpore = 1.0 / (1.0 / Dm + 1.0 / DK)
    return {"effective_diffusivity_m2_s": eps / tau * Dpore, "knudsen_diffusivity_m2_s": DK,
            "pore_diffusivity_m2_s": Dpore, "knudsen_fraction_of_resistance": (1.0 / DK) / (1.0 / Dm + 1.0 / DK)}


# ---------------------------------------------------------------------------------------------------------------------
# analytical solutions (first order, isothermal)
# ---------------------------------------------------------------------------------------------------------------------
def eta_first_order(shape: str, phi: float) -> float:
    """η of a first-order isothermal reaction, φ = L (k/De)^(1/2) with L the half-thickness or the radius:
    slab tanh φ/φ; cylinder 2 I1(φ)/(φ I0(φ)); sphere 3 (φ coth φ − 1)/φ²."""
    if phi < 1e-2:      # series (the closed forms lose ~1e-8 to cancellation near φ = 1e-4)
        c2, c4 = {"slab": (1.0 / 3.0, 2.0 / 15.0), "cylinder": (1.0 / 8.0, 1.0 / 48.0),
                  "sphere": (1.0 / 15.0, 2.0 / 315.0)}[shape]
        p2 = phi * phi
        return 1.0 - c2 * p2 + c4 * p2 * p2
    if shape == "slab":
        return math.tanh(phi) / phi
    if shape == "cylinder":
        return 2.0 * float(i1e(phi)) / (phi * float(i0e(phi)))
    return 3.0 * (phi / math.tanh(phi) - 1.0) / (phi * phi)


def eta_first_order_with_film(shape: str, phi: float, biot_mass: float) -> float:
    """Overall η (referred to bulk) of a first-order isothermal pellet with an external film: resistances in series,
    1/η_o = 1/η + φ² / ((s + 1) Bi_m), Bi_m = kc L / De."""
    eta = eta_first_order(shape, phi)
    return 1.0 / (1.0 / eta + phi * phi / ((SHAPES[shape] + 1.0) * biot_mass))


def eta_zero_order(shape: str, phi: float) -> float:
    """η of a zero-order isothermal reaction, φ = L (k/(De Cs))^(1/2). 1 until the centre is exhausted, then the
    dead-core solution (slab: closed form; cylinder and sphere: dead-core radius from its transcendental equation)."""
    s = SHAPES[shape]
    if phi * phi <= 2.0 * (s + 1.0):
        return 1.0
    if s == 0:
        return math.sqrt(2.0) / phi
    if s == 1:      # C(1) = 1 with C = C' = 0 at ρ: φ²/4 (1 − ρ² + 2ρ² ln ρ) = 1
        rho = brentq(lambda r: phi * phi / 4.0 * (1.0 - r * r + 2.0 * r * r * math.log(r)) - 1.0, 1e-300, 1.0 - 1e-15)
        return 1.0 - rho * rho
    rho = brentq(lambda r: phi * phi / 6.0 * (1.0 - 3.0 * r * r + 2.0 * r ** 3) - 1.0, 0.0, 1.0)
    return 1.0 - rho ** 3


# ---------------------------------------------------------------------------------------------------------------------
# the pellet model
# ---------------------------------------------------------------------------------------------------------------------
class Pellet:
    """Fixed data of one pellet (geometry, kinetics, transport) and the boundary-value solver for given bulk conditions.

    A class because the fixed-bed model solves the same pellet hundreds of times with a warm start."""

    def __init__(self, geometry: str, size_m: float, kin: dict, De: float, dH: float | None = None,
                 lambda_e: float | None = None, kc: float | None = None, h: float | None = None):
        if geometry not in SHAPES:
            raise ValueError("geometry debe ser 'slab' (placa, tamaño = semiespesor), 'cylinder' (cilindro largo, tamaño = "
                             "radio) o 'sphere' (esfera, tamaño = radio)")
        self.geometry, self.s, self.L, self.kin, self.De = geometry, SHAPES[geometry], float(size_m), kin, float(De)
        self.dH, self.lam, self.kc, self.h = dH, lambda_e, kc, h
        thermal = dH is not None and dH != 0.0 and (lambda_e is not None or h is not None)
        self.mode = "iso" if not thermal else ("coupled" if lambda_e is not None else "iso_film")
        self.dead_core = kin["type"] == "power_law" and kin["n"] < DEAD_CORE_ORDER
        if self.dead_core and thermal:
            raise ValueError(f"orden de reacción < {DEAD_CORE_ORDER} con efectos térmicos no está soportado (frente de núcleo "
                             "muerto no isotérmico); represente la saturación con 'langmuir_hinshelwood' o resuelva el "
                             "pellet isotérmico")
        if thermal and (kin["E"] is None):
            raise ValueError("falta kinetics['activation_energy_J_mol']: sin ella no se puede evaluar el efecto de la "
                             "temperatura del pellet sobre la velocidad")

    # -- problem assembly ---------------------------------------------------------------------------------------------
    def _system(self, Cb: float, Tb: float, lam: float):
        """Dimensionless ODE system in u = C/Cb, θ = T/Tb for the rate multiplied by `lam` (continuation)."""
        s, L, De, kin, mode = self.s, self.L, self.De, self.kin, self.mode
        a2 = lam * L * L / (De * Cb)
        q = -self.dH if self.dH is not None else 0.0
        bT = lam * L * L * q / (self.lam * Tb) if mode == "coupled" else 0.0
        Bim = None if self.kc is None else self.kc * L / De
        Bih = None if (self.h is None or mode != "coupled") else self.h * L / self.lam
        film = q * De * Cb / (self.h * L * Tb) if mode == "iso_film" else 0.0

        def r_of(u, th):
            return rate(kin, Cb * u, Tb * np.clip(th, 0.05, 50.0), Cb)

        if mode == "iso":
            S = np.array([[0.0, 0.0], [0.0, -float(s)]])

            def fun(x, y):
                return np.vstack([y[1], a2 * r_of(y[0], np.ones_like(y[0]))])

            def bc(ya, yb):
                return np.array([ya[1], yb[0] - 1.0 if Bim is None else yb[1] - Bim * (1.0 - yb[0])])
        elif mode == "iso_film":
            S = np.array([[0.0, 0.0, 0.0], [0.0, -float(s), 0.0], [0.0, 0.0, 0.0]])

            def fun(x, y):
                return np.vstack([y[1], a2 * r_of(y[0], y[2]), np.zeros_like(y[0])])

            def bc(ya, yb):
                return np.array([ya[1], yb[0] - 1.0 if Bim is None else yb[1] - Bim * (1.0 - yb[0]),
                                 yb[2] - 1.0 - film * yb[1]])
        else:
            S = np.zeros((4, 4))
            S[1, 1] = S[3, 3] = -float(s)

            def fun(x, y):
                rr = r_of(y[0], y[2])
                return np.vstack([y[1], a2 * rr, y[3], -bT * rr])

            def bc(ya, yb):
                return np.array([ya[1], ya[3], yb[0] - 1.0 if Bim is None else yb[1] - Bim * (1.0 - yb[0]),
                                 yb[2] - 1.0 if Bih is None else yb[3] - Bih * (1.0 - yb[2])])
        return fun, bc, (S if s else None), r_of

    def modulus_bulk(self, Cb: float, Tb: float) -> float:
        """Generalized modulus at bulk conditions, isothermal: (L/(s+1)) r(Cb) / (2 De ∫0^Cb r dC)^(1/2)."""
        return self.generalized_modulus(Cb, Tb, isothermal=True)

    def generalized_modulus(self, Cs: float, Ts: float, isothermal: bool = False) -> float:
        """Bischoff/Aris/Petersen modulus at surface conditions; with heat effects T follows the Prater relation
        T = Ts + (−ΔH) De (Cs − C)/λe inside the integral (Rawlings Eq. 7.66)."""
        B = 0.0 if (isothermal or self.mode != "coupled") else -self.dH * self.De / self.lam

        def f(c):
            return float(rate(self.kin, c, Ts + B * (Cs - c), Cs))

        integral = quad(f, 0.0, Cs, limit=200, epsabs=0.0, epsrel=1e-10)[0]
        return self.L / (self.s + 1.0) * f(Cs) / math.sqrt(2.0 * self.De * integral)

    def _layer(self, Cs: float, Ts: float, lam: float) -> float:
        """Decay rate κ of the diffusion-limited profile u ≈ us exp(−κ(1−x)) for surface conditions (Cs, Ts):
        κ = φs²/((s+1) Φ) with Φ the generalized modulus including the Prater temperature rise (κ = φ for a first-order
        isothermal pellet)."""
        phis2 = self.L ** 2 * float(rate(self.kin, Cs, Ts, Cs)) / (self.De * Cs)
        return math.sqrt(lam) * phis2 / ((self.s + 1.0) * self.generalized_modulus(Cs, Ts))

    def _hot_surface(self, Cb: float, Tb: float, lam: float) -> tuple[float, float, float]:
        """Surface state (us, θs, κ) of the most reactive diffusion-limited state consistent with the film balances,
        from the asymptote η = 1/Φ. Only an initial guess for the hot continuation."""
        q = -self.dH if self.mode != "iso" else 0.0
        Bim = None if self.kc is None else self.kc * self.L / self.De
        if Bim is None and (self.h is None or self.mode == "iso"):
            return 1.0, 1.0, self._layer(Cb, Tb, lam)
        if Bim is None:           # heat film only: θs − 1 = q De Cb κ(θs) /(h L Tb); take the hottest root
            grid = np.geomspace(1e-6, 20.0, 80)
            f = [1.0 + g - 1.0 - q * self.De * Cb * self._layer(Cb, Tb * (1.0 + g), lam) / (self.h * self.L * Tb)
                 for g in grid]
            idx = [i for i in range(len(grid) - 1) if f[i] * f[i + 1] < 0.0]
            ths = 1.0 + (grid[idx[-1] + 1] if idx else grid[0])
            return 1.0, ths, self._layer(Cb, Tb * ths, lam)

        def ths_of(us):
            return 1.0 + q * self.kc * Cb * (1.0 - us) / (self.h * Tb) if (self.h is not None and q) else 1.0

        grid = np.geomspace(1e-9, 1.0, 100)
        f = [Bim * (1.0 - us) - self._layer(Cb * us, Tb * min(ths_of(us), 40.0), lam) * us for us in grid]
        idx = [i for i in range(len(grid) - 1) if f[i] * f[i + 1] < 0.0]
        us = float(grid[idx[0]]) if idx else 1.0                       # smallest root = most ignited state
        ths = min(ths_of(us), 40.0)
        return us, ths, self._layer(Cb * us, Tb * ths, lam)

    def _guess(self, x: np.ndarray, Cb: float, Tb: float, lam: float, hot: bool) -> np.ndarray:
        nvar = {"iso": 2, "iso_film": 3, "coupled": 4}[self.mode]
        y = np.zeros((nvar, x.size))
        y[0] = 1.0
        if nvar > 2:
            y[2] = 1.0
        if not hot:
            return y
        us, ths, kappa = self._hot_surface(Cb, Tb, lam)
        y[0] = us * np.exp(-kappa * (1.0 - x))
        y[1] = kappa * y[0]
        if self.mode == "iso":
            return y
        if self.mode == "coupled":
            beta = -self.dH * self.De * Cb / (self.lam * Tb)
            y[2] = ths + beta * (us - y[0])
            y[3] = -beta * y[1]
        else:
            y[2] = ths
        return y

    @staticmethod
    def _mesh(phiL: float) -> np.ndarray:
        layer = 1.0 - np.geomspace(1e-3 / max(phiL, 1.0), 1.0, 40)
        return np.unique(np.clip(np.concatenate([np.linspace(0.0, 1.0, 41), layer]), 0.0, 1.0))

    def _solve_one(self, Cb, Tb, lam, x, y, tol):
        fun, bc, S, _ = self._system(Cb, Tb, lam)
        return solve_bvp(fun, bc, x, y, S=S, tol=tol, max_nodes=BVP_MAX_NODES, verbose=0)

    def continuation(self, Cb: float, Tb: float, hot: bool = False, tol: float = BVP_TOL):
        """Reach the target problem (lam = 1) from the kinetic side (cold) or from the diffusion-limited side (hot).
        Returns the scipy solution of the last step or None when a step did not converge."""
        Phi0 = self.modulus_bulk(Cb, Tb)
        if hot:
            lam = max(1.0, (HOT_START_MODULUS / Phi0) ** 2)
        else:
            lam = min(1.0, (0.3 / Phi0) ** 2)
        x = self._mesh(self._hot_surface(Cb, Tb, lam)[2] if hot else (self.s + 1.0) * Phi0 * math.sqrt(lam))
        y = self._guess(x, Cb, Tb, lam, hot)
        sol = self._solve_one(Cb, Tb, lam, x, y, tol if lam == 1.0 else 1e-4)
        if sol.status != 0:
            return None
        factor = CONTINUATION_FACTOR
        while lam != 1.0:
            nxt = min(1.0, lam * factor) if not hot else max(1.0, lam / factor)
            trial = self._solve_one(Cb, Tb, nxt, sol.x, sol.y, tol if nxt == 1.0 else 1e-4)
            if trial.status != 0:
                factor = math.sqrt(factor)
                if factor < 1.02:
                    return None
                continue
            sol, lam = trial, nxt
        return sol

    def solve_from(self, Cb: float, Tb: float, x: np.ndarray, y: np.ndarray, tol: float = BVP_TOL):
        """One collocation solve at the target problem from a given guess (warm start of the fixed-bed model)."""
        sol = self._solve_one(Cb, Tb, 1.0, x, y, tol)
        return sol if sol.status == 0 else None

    # -- post-processing ----------------------------------------------------------------------------------------------
    def summary(self, sol, Cb: float, Tb: float) -> dict:
        """Rates, effectiveness factors and surface/centre state of one converged solution."""
        s, L, De = self.s, self.L, self.De
        _, _, _, r_of = self._system(Cb, Tb, 1.0)
        us, dus = float(sol.y[0, -1]), float(sol.y[1, -1])
        if self.mode == "iso":
            ths = thc = 1.0
        elif self.mode == "iso_film":
            ths = thc = float(sol.y[2, -1])
        else:
            ths, thc = float(sol.y[2, -1]), float(sol.y[2, 0])
        rate_flux = (s + 1.0) * De * Cb * dus / (L * L)                      # mol / (m³ pellet · s), from the surface flux
        xa, xb = sol.x[:-1], sol.x[1:]
        xg = (0.5 * (xa + xb)[:, None] + 0.5 * (xb - xa)[:, None] * _GL_X[None, :]).ravel()
        yg = sol.sol(xg)
        thg = yg[2] if self.mode != "iso" else np.ones_like(xg)
        wg = (0.5 * (xb - xa)[:, None] * _GL_W[None, :]).ravel()
        rate_int = (s + 1.0) * float(np.sum(wg * r_of(yg[0], thg) * xg ** s))
        Cs, Ts = us * Cb, ths * Tb
        r_bulk = float(rate(self.kin, Cb, Tb, Cb))
        r_surf = float(rate(self.kin, Cs, Ts, Cb))
        out = {"observed_rate_mol_m3_s": rate_flux, "rate_integrated_mol_m3_s": rate_int,
               "flux_balance_rel": abs(rate_int - rate_flux) / max(abs(rate_flux), 1e-300),
               "eta_overall": rate_flux / r_bulk, "eta_internal": rate_flux / r_surf, "eta_external": r_surf / r_bulk,
               "surface_concentration_mol_m3": Cs, "surface_temperature_K": Ts,
               "center_concentration_mol_m3": float(sol.y[0, 0]) * Cb, "center_temperature_K": thc * Tb,
               "max_temperature_K": float(np.max(sol.y[2])) * Tb if self.mode == "coupled" else Ts,
               "rate_at_bulk_mol_m3_s": r_bulk, "rate_at_surface_mol_m3_s": r_surf}
        if self.mode == "coupled":
            B = -self.dH * De / self.lam
            inv = sol.y[2] * Tb + B * sol.y[0] * Cb
            scale = max(abs(B) * Cb, 1e-9 * Tb)
            out["prater_invariant_rel"] = float(np.max(np.abs(inv - inv[-1]))) / scale
        return out

    # -- independent shooting method (no external film, no dead core) ----------------------------------------------------
    def shooting_applicable(self) -> bool:
        k = self.kin
        return self.kc is None and self.h is None and (k["type"] == "langmuir_hinshelwood" or k["n"] >= 1.0)

    def shooting_states(self, Cs: float, Ts: float, scan: bool) -> dict:
        """All steady states of the pellet with surface conditions fixed (Weisz-Hicks construction).

        y = C/Cs, ξ = x φs with φs² = L² r(Cs,Ts)/(De Cs):  y'' + (s/ξ) y' = f(y), f = r(Cs y, T(y))/r(Cs,Ts), integrated
        in v = ln y from the centre value v0 until y = 1; the distance ξ* is the Thiele modulus that centre value belongs
        to. Roots of ξ*(v0) = φs are the steady states; η_internal = (s+1) y'(ξ*)/ξ*."""
        s = self.s
        B = -self.dH * self.De / self.lam if self.mode == "coupled" else 0.0
        rs = float(rate(self.kin, Cs, Ts, Cs))
        phis = self.L * math.sqrt(rs / (self.De * Cs))

        kin = self.kin
        ks = float(rate_constant(kin, Ts))
        den_s = (1.0 + float(adsorption_constant(kin, Ts)) * Cs) ** kin["m"] if kin["type"] != "power_law" else 1.0

        def g(v):       # f(y)/y written so that it stays finite when y = exp(v) underflows
            y = math.exp(v)
            T = Ts + B * Cs * (1.0 - y)
            kr = float(rate_constant(kin, T)) / ks
            if kin["type"] == "power_law":
                return kr * math.exp((kin["n"] - 1.0) * v)
            return kr * den_s / (1.0 + float(adsorption_constant(kin, T)) * Cs * y) ** kin["m"]

        def shoot(v0, dense=False):
            g0 = g(v0)
            xi0 = min(1e-4, 0.01 * math.sqrt(2.0 * (s + 1.0) * abs(v0))) / math.sqrt(g0)
            start = [v0 + g0 * xi0 * xi0 / (2.0 * (s + 1.0)), g0 * xi0 / (s + 1.0)]

            def ode(xi, w):
                return [w[1], g(min(w[0], 0.0)) - w[1] * w[1] - s * w[1] / xi]

            def hit(xi, w):
                return w[0]
            hit.terminal, hit.direction = True, 1.0
            xi_end = xi0 * 10.0
            while True:
                res = solve_ivp(ode, (xi0, xi_end), start, events=hit, rtol=1e-10, atol=1e-12, method="LSODA",
                                dense_output=dense)
                if res.status == 1:
                    xi = float(res.t_events[0][0])
                    return xi, float(res.y_events[0][0][1]), (res.sol, xi0, g0)
                if res.status != 0 or xi_end > 1e12:
                    raise RuntimeError("shooting did not reach the surface")
                xi_end *= 100.0

        def mismatch(v0):
            return math.log(shoot(v0)[0] / phis)

        lo = -1e-9
        hi = -1.0
        tries = 0
        while mismatch(hi) < 0.0:
            hi *= 4.0
            tries += 1
            if tries > 14:
                raise RuntimeError("shooting bracket not found")
        if scan:
            # the hottest/most reactive core sets how small the centre concentration of an ignited state can be
            g_max = max(g(v) for v in -np.geomspace(1e-6, 700.0, 80))
            deepest = max(-4.0 * hi, 3.0 * phis * math.sqrt(g_max) + 50.0)
            grid = -np.geomspace(-lo, deepest, SHOOT_GRID)
            vals = [mismatch(v) for v in grid]
            brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0.0]
        else:
            brackets = [(lo, hi)]
        states = []
        x = self._mesh(phis * math.sqrt(max(g(-1e-6), g(-50.0), 1.0)))
        beta = B * Cs / Ts
        for a, b in brackets:
            v0 = brentq(mismatch, a, b, xtol=1e-13, rtol=1e-12)
            xi, slope, (dense, xi0, g0) = shoot(v0, dense=True)
            y0 = math.exp(v0)
            xs = np.clip(x * xi, xi0, xi)
            w = dense(xs)
            near = x * xi < xi0                                   # series of the centre
            v = np.where(near, v0 + g0 * (x * xi) ** 2 / (2.0 * (s + 1.0)), w[0])
            dv = np.where(near, g0 * x * xi / (s + 1.0), w[1])
            u = np.exp(np.minimum(v, 0.0))
            guess = np.vstack([u, u * dv * xi] + ([1.0 + beta * (1.0 - u), -beta * u * dv * xi] if B else []))
            if self.mode == "coupled" and not B:
                guess = np.vstack([guess, np.ones_like(u), np.zeros_like(u)])
            states.append({"eta_internal": (s + 1.0) * slope / xi, "center_concentration_mol_m3": y0 * Cs,
                           "center_temperature_K": Ts + B * Cs * (1.0 - y0), "x": x, "y": guess})
        states.sort(key=lambda d: d["eta_internal"])
        return {"thiele_modulus_surface": phis, "states": states, "scanned": bool(scan)}


def solve_dead_core(p: Pellet, Cb: float, Tb: float, n_profile: int = 41) -> dict:
    """Isothermal power law with n < DEAD_CORE_ORDER: the reactant can be exhausted inside the pellet (dead core), where
    collocation fails. One-parameter shooting instead: τ > 0 is the centre value u0 = τ (no dead core) and τ ≤ 0 a dead
    core of radius ρ = −τ, started on the front with the local solution u = A t^p, p = 2/(1−n), A = (φ²/(p(p−1)))^(p/2);
    the surface condition fixes τ (brentq). u = C/Cb, φ² = L² k Cb^(n−1)/De."""
    s, n = p.s, p.kin["n"]
    phi2 = p.L * p.L * float(rate_constant(p.kin, Tb)) * Cb ** (n - 1.0) / p.De
    Bi = None if p.kc is None else p.kc * p.L / p.De
    pw = 2.0 / (1.0 - n)

    def ode(x, w):
        un = max(w[0], 0.0) ** n if n > 0 else 1.0
        return [w[1], phi2 * un - s * w[1] / x, un * x ** s]

    def march(tau, rtol, dense=False):
        if tau > 0.0:
            x0 = 1e-6
            f0 = phi2 * tau ** n
            start = [tau + f0 * x0 * x0 / (2.0 * (s + 1.0)), f0 * x0 / (s + 1.0), tau ** n * x0 ** (s + 1) / (s + 1.0)]
            rho = 0.0
        else:
            rho = -tau
            if rho < 1e-9:          # front at the centre: exact power solution through the origin
                A = (phi2 / (pw * (pw - 1.0 + s))) ** (pw / 2.0)
                x0, t0, rho = 1e-6, 1e-6, 0.0
            else:
                A = (phi2 / (pw * (pw - 1.0))) ** (pw / 2.0)
                t0 = min(1e-6, 1e-3 * rho, 1e-3 * (1.0 - rho))
                x0 = rho + t0
            u = A * t0 ** pw
            start = [u, A * pw * t0 ** (pw - 1.0), u ** n * t0 * x0 ** s / (pw * n + 1.0)]
        res = solve_ivp(ode, (x0, 1.0), start, method="LSODA", rtol=rtol, atol=1e-300, dense_output=dense)
        if res.status != 0:
            raise RuntimeError("dead-core shooting failed")
        return res, rho, x0

    def residual(tau, rtol=IVP_RTOL):
        res, _, _ = march(tau, rtol)
        return res.y[0, -1] - 1.0 if Bi is None else res.y[0, -1] + res.y[1, -1] / Bi - 1.0

    reasons: list[str] = []
    out: dict = {}
    for rtol in (IVP_RTOL, IVP_RTOL_FINE):
        tau = brentq(lambda z: residual(z, rtol), -1.0 + 1e-9, 1.0, xtol=1e-15, rtol=1e-14)
        res, rho, x0 = march(tau, rtol, dense=rtol == IVP_RTOL_FINE)
        us, dus, integral = float(res.y[0, -1]), float(res.y[1, -1]), float(res.y[2, -1])
        out[rtol] = {"eta_overall": (s + 1.0) * dus / phi2, "us": us, "dus": dus, "rho": rho, "tau": tau, "x0": x0,
                     "flux_balance_rel": abs(phi2 * integral - dus) / dus, "res": res}
    c, f = out[IVP_RTOL], out[IVP_RTOL_FINE]
    r_bulk = float(rate(p.kin, Cb, Tb, Cb)) if n > 0 else float(rate_constant(p.kin, Tb))
    ev = {"mesh_independence_rel": abs(c["eta_overall"] - f["eta_overall"]) / f["eta_overall"],
          "flux_balance_rel": f["flux_balance_rel"], "dead_core_radius_over_L": f["rho"],
          "surface_condition_residual": float(abs(residual(f["tau"], IVP_RTOL_FINE)))}
    if ev["mesh_independence_rel"] > MESH_TOL:
        reasons.append(f"η cambia {ev['mesh_independence_rel']:.1e} al apretar la tolerancia de integración")
    if ev["flux_balance_rel"] > FLUX_TOL:
        reasons.append(f"la velocidad integrada difiere del flujo en la superficie en {ev['flux_balance_rel']:.1e}")
    if n == 0.0 and Bi is None:
        phi = math.sqrt(phi2)
        ev["analytical_eta_overall"] = eta_zero_order(p.geometry, phi)
        ev["analytical_rel_dev"] = abs(f["eta_overall"] - ev["analytical_eta_overall"]) / ev["analytical_eta_overall"]
        if ev["analytical_rel_dev"] > CHECK_TOL:
            reasons.append(f"η difiere de la solución analítica de orden cero en {ev['analytical_rel_dev']:.1e}")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    us = f["us"]
    eta_ext = us ** n if n > 0 else 1.0
    sm = {"observed_rate_mol_m3_s": f["eta_overall"] * r_bulk, "eta_overall": f["eta_overall"],
          "eta_internal": f["eta_overall"] / eta_ext, "eta_external": eta_ext, "surface_concentration_mol_m3": us * Cb,
          "surface_temperature_K": Tb, "center_concentration_mol_m3": max(f["tau"], 0.0) * Cb, "center_temperature_K": Tb,
          "max_temperature_K": Tb, "rate_at_bulk_mol_m3_s": r_bulk, "rate_at_surface_mol_m3_s": r_bulk * eta_ext,
          "flux_balance_rel": f["flux_balance_rel"]}
    xs = np.linspace(0.0, 1.0, n_profile)
    inside = xs >= f["x0"]
    u = np.where(inside, 0.0, max(f["tau"], 0.0))
    u[inside] = np.maximum(f["res"].sol(xs[inside])[0], 0.0)
    profile = {"x_over_L": xs.tolist(), "concentration_mol_m3": (u * Cb).tolist(),
               "temperature_K": np.full_like(xs, Tb).tolist()}
    state = {k: sm[k] for k in ("eta_overall", "eta_internal", "observed_rate_mol_m3_s", "surface_concentration_mol_m3",
                                "surface_temperature_K", "center_concentration_mol_m3", "max_temperature_K")}
    return {"converged": True, "selected": sm, "solution": None, "profile": profile, "steady_states": [state],
            "multiplicity_possible": False, "evidence": ev}


def multiplicity_possible(p: Pellet, Cb: float, Tb: float) -> bool:
    """Several steady states need a rate that can rise while the reactant is consumed: exothermic with heat effects, or
    Langmuir-Hinshelwood with negative apparent order at bulk conditions."""
    if p.mode != "iso" and p.dH < 0:
        return True
    return p.kin["type"] == "langmuir_hinshelwood" and apparent_order(p.kin, Cb, Tb) < 0.0


def solve_pellet(p: Pellet, Cb: float, Tb: float, branch: str = "low", n_profile: int = 41) -> dict:
    """Solve the pellet at bulk (Cb, Tb): every steady state found, the selected one in detail, and the evidence."""
    if branch not in ("low", "middle", "high"):
        raise ValueError("branch debe ser 'low' (estado de menor velocidad), 'middle' o 'high' (estado encendido)")
    if p.dead_core:
        try:
            return solve_dead_core(p, Cb, Tb, n_profile)
        except (RuntimeError, ValueError) as exc:
            return {"converged": False, "evidence": {"verified": False, "reasons": [
                f"el disparo desde el frente de núcleo muerto no convergió ({exc})"]}}
    reasons: list[str] = []
    multi = multiplicity_possible(p, Cb, Tb)
    sols = []
    shooting = None
    shooting_error = None
    hot_failed = cold_failed = False
    if p.shooting_applicable():
        # surface conditions are the bulk ones: the shooting method gives every steady state; collocation then
        # converges each of them from the shooting profile (two independent discretisations of the same state)
        try:
            shooting = p.shooting_states(Cb, Tb, scan=multi)
            for st in shooting["states"]:
                sol = p.solve_from(Cb, Tb, st["x"], st["y"])
                if sol is not None:
                    sols.append(("shooting", sol))
        except (RuntimeError, ValueError, OverflowError) as exc:            # reported, never hidden
            shooting_error = str(exc)
    if not sols:
        cold = p.continuation(Cb, Tb, hot=False)
        if cold is None:
            cold_failed = True
        else:
            sols.append(("cold", cold))
        if multi or cold_failed:
            hot = p.continuation(Cb, Tb, hot=True)
            if hot is None:
                hot_failed = True
            else:
                sols.append(("hot", hot))
    if not sols:
        return {"converged": False, "evidence": {"verified": False, "reasons": [
            "la colocación (solve_bvp) no convergió ni desde el régimen cinético ni desde el difusional"]}}
    summaries = [(tag, sol, p.summary(sol, Cb, Tb)) for tag, sol in sols]
    distinct = [summaries[0]]
    for item in summaries[1:]:
        if all(abs(item[2]["eta_overall"] - d[2]["eta_overall"]) > BRANCH_TOL * max(d[2]["eta_overall"], 1e-300)
               for d in distinct):
            distinct.append(item)
    distinct.sort(key=lambda d: d[2]["eta_overall"])
    if branch == "middle" and len(distinct) < 3:
        raise ValueError("branch = 'middle' solo existe cuando se encontraron tres estados estacionarios")
    chosen = distinct[0] if branch == "low" else (distinct[-1] if branch == "high" else distinct[len(distinct) // 2])
    tag, sol, sm = chosen
    fine = None
    for tol_fine in (BVP_TOL_FINE, 10.0 * BVP_TOL_FINE):      # a very steep ignited profile can exhaust the mesh at 1e-8
        fine = p.solve_from(Cb, Tb, sol.x, sol.y, tol=tol_fine)
        if fine is not None:
            break
    ev: dict = {"flux_balance_rel": sm["flux_balance_rel"], "collocation_nodes": int(sol.x.size),
                "collocation_max_residual": float(np.max(sol.rms_residuals))}
    if fine is None:
        reasons.append("la resolución con tolerancia fina (independencia de malla) no convergió")
        ev["mesh_independence_rel"] = None
    else:
        smf = p.summary(fine, Cb, Tb)
        ev["mesh_independence_rel"] = abs(smf["eta_overall"] - sm["eta_overall"]) / smf["eta_overall"]
        ev["collocation_nodes_fine"] = int(fine.x.size)
        ev["fine_tolerance"] = tol_fine
        if ev["mesh_independence_rel"] > MESH_TOL:
            reasons.append(f"η cambia {ev['mesh_independence_rel']:.1e} al refinar la malla (límite {MESH_TOL:.0e})")
        tag, sol, sm = tag, fine, smf
        ev["flux_balance_rel"] = sm["flux_balance_rel"]
    if ev["flux_balance_rel"] > FLUX_TOL:
        reasons.append(f"la velocidad integrada en el pellet difiere del flujo en la superficie en "
                       f"{ev['flux_balance_rel']:.1e} (límite {FLUX_TOL:.0e})")
    if "prater_invariant_rel" in sm:
        ev["prater_invariant_rel"] = sm["prater_invariant_rel"]
        if sm["prater_invariant_rel"] > PRATER_TOL:
            reasons.append(f"el invariante de Prater T + (−ΔH)·De·C/λe no es constante ({sm['prater_invariant_rel']:.1e})")
    first_order = p.kin["type"] == "power_law" and p.kin["n"] == 1.0 and p.mode == "iso"
    if first_order:
        phi = p.L * math.sqrt(float(rate_constant(p.kin, Tb)) / p.De)
        exact = (eta_first_order(p.geometry, phi) if p.kc is None
                 else eta_first_order_with_film(p.geometry, phi, p.kc * p.L / p.De))
        ev["analytical_eta_overall"] = exact
        ev["analytical_rel_dev"] = abs(sm["eta_overall"] - exact) / exact
        if ev["analytical_rel_dev"] > CHECK_TOL:
            reasons.append(f"η difiere de la solución analítica de primer orden en {ev['analytical_rel_dev']:.1e}")
    if shooting is not None:
        best = min(shooting["states"], key=lambda st: abs(st["eta_internal"] - sm["eta_internal"]))
        ev["shooting_eta_internal"] = best["eta_internal"]
        ev["shooting_rel_dev"] = abs(best["eta_internal"] - sm["eta_internal"]) / sm["eta_internal"]
        ev["shooting_states_found"] = len(shooting["states"])
        if ev["shooting_rel_dev"] > CHECK_TOL:
            reasons.append(f"η difiere del método de disparo independiente en {ev['shooting_rel_dev']:.1e}")
        if len(shooting["states"]) != len(distinct):
            reasons.append(f"el método de disparo encuentra {len(shooting['states'])} estados estacionarios y la "
                           f"colocación {len(distinct)}")
    elif shooting_error is not None:
        ev["shooting_error"] = shooting_error
    if multi:
        if shooting is not None and shooting["scanned"]:
            ev["multiplicity_search"] = ("barrido de la concentración central por disparo (encuentra todos los estados con "
                                         "superficie fija) + colocación desde arranque frío y caliente")
        else:
            ev["multiplicity_search"] = ("colocación desde arranque frío (régimen cinético) y caliente (régimen difusional): "
                                         "detecta histéresis, no demuestra que no haya más estados")
            ev["cold_start_converged"], ev["hot_start_converged"] = not cold_failed, not hot_failed
            if hot_failed or cold_failed:
                ev["multiplicity_search"] += ("; uno de los dos arranques no convergió (lo esperable más allá de un punto "
                                              "de ignición o extinción, donde esa rama deja de existir)")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    xs = np.linspace(0.0, 1.0, n_profile)
    ys = sol.sol(xs)
    profile = {"x_over_L": xs.tolist(), "concentration_mol_m3": (ys[0] * Cb).tolist(),
               "temperature_K": ((ys[2] if p.mode == "coupled" else np.full_like(xs, sm["surface_temperature_K"] / Tb))
                                 * Tb).tolist()}
    states = [{"eta_overall": d[2]["eta_overall"], "eta_internal": d[2]["eta_internal"],
               "observed_rate_mol_m3_s": d[2]["observed_rate_mol_m3_s"],
               "surface_concentration_mol_m3": d[2]["surface_concentration_mol_m3"],
               "surface_temperature_K": d[2]["surface_temperature_K"],
               "center_concentration_mol_m3": d[2]["center_concentration_mol_m3"],
               "max_temperature_K": d[2]["max_temperature_K"]} for d in distinct]
    return {"converged": True, "selected": sm, "solution": sol, "profile": profile, "steady_states": states,
            "multiplicity_possible": multi, "evidence": ev}


# ---------------------------------------------------------------------------------------------------------------------
# public engine
# ---------------------------------------------------------------------------------------------------------------------
def catalyst_effectiveness(geometry: str, characteristic_length_m: float, kinetics: dict, concentration_mol_m3: float,
                           temperature_K: float, effective_diffusivity_m2_s: float | None = None, pore: dict | None = None,
                           heat_of_reaction_J_mol: float | None = None,
                           effective_conductivity_W_mK: float | None = None,
                           mass_transfer_coefficient_m_s: float | None = None,
                           heat_transfer_coefficient_W_m2K: float | None = None,
                           pellet_density_kg_m3: float | None = None, branch: str = "low") -> dict:
    """Effectiveness factor of one catalyst pellet at given bulk conditions (see the module docstring for the model)."""
    warnings: list[str] = []
    assumptions = ["una sola reacción y un reactivo limitante; difusión de Fick con difusividad efectiva constante "
                   "(equimolar o diluido), propiedades efectivas uniformes en el pellet",
                   "geometría 1-D: placa (tamaño = semiespesor, o espesor si una cara está sellada), cilindro largo "
                   "(radio, sin efecto de las tapas) o esfera (radio)"]
    for name, val, what in (("characteristic_length_m", characteristic_length_m, "semiespesor o radio del pellet"),
                            ("concentration_mol_m3", concentration_mol_m3, "concentración del reactivo en el fluido"),
                            ("temperature_K", temperature_K, "temperatura del fluido")):
        if val is None or not val > 0:
            raise ValueError(f"falta {name} > 0: {what}")
    L, Cb, Tb = float(characteristic_length_m), float(concentration_mol_m3), float(temperature_K)
    kin = normalize_kinetics(kinetics, pellet_density_kg_m3)
    diffusivity = {"source": "dato del usuario"}
    if effective_diffusivity_m2_s is not None:
        if pore is not None:
            raise ValueError("da effective_diffusivity_m2_s o pore (para estimarla), no ambos")
        De = float(effective_diffusivity_m2_s)
        if not De > 0:
            raise ValueError("effective_diffusivity_m2_s debe ser > 0")
    elif pore is not None:
        est = effective_diffusivity_from_pores(pore, Tb)
        De = est["effective_diffusivity_m2_s"]
        diffusivity = {"source": "ESTIMADA de los datos de poro (Bosanquet: Knudsen + molecular, porosidad/tortuosidad)",
                       **est}
        assumptions.append("difusividad efectiva ESTIMADA: De = (ε/τ)/(1/D_molecular + 1/D_Knudsen), D_Knudsen = (2/3)·r_poro·"
                           "(8RT/πM)^½ evaluada a la temperatura del fluido; poros cilíndricos de un solo tamaño")
        warnings.append("la difusividad efectiva es una estimación de datos de poro, no una medida: la tortuosidad domina "
                        "su incertidumbre")
    else:
        raise ValueError("falta effective_diffusivity_m2_s (difusividad efectiva del reactivo en el pellet) o pore "
                         "{porosity, tortuosity, mean_pore_radius_m, molecular_diffusivity_m2_s, molar_mass_kg_mol} para "
                         "estimarla")
    dH, lam, kc, h = heat_of_reaction_J_mol, effective_conductivity_W_mK, mass_transfer_coefficient_m_s, \
        heat_transfer_coefficient_W_m2K
    for name, val in (("effective_conductivity_W_mK", lam), ("mass_transfer_coefficient_m_s", kc),
                      ("heat_transfer_coefficient_W_m2K", h)):
        if val is not None and not val > 0:
            raise ValueError(f"{name} debe ser > 0")
    if (lam is not None or h is not None) and dH is None:
        raise ValueError("falta heat_of_reaction_J_mol (negativo si es exotérmica): se dio conductividad efectiva o "
                         "coeficiente de película térmico, que solo tienen efecto con calor de reacción")
    if dH is not None and dH != 0.0 and lam is None and h is None:
        raise ValueError("falta effective_conductivity_W_mK (gradiente térmico dentro del pellet) y/o "
                         "heat_transfer_coefficient_W_m2K (película externa): con calor de reacción hace falta al menos uno")
    p = Pellet(geometry, L, kin, De, dH, lam, kc, h)
    res = solve_pellet(p, Cb, Tb, branch)
    method = ("problema de contorno de difusión-reacción (C y T acoplados) por colocación de 4.º orden con malla adaptativa "
              "(scipy.integrate.solve_bvp) y continuación en la constante de velocidad; comprobación independiente por "
              "disparo desde el centro (scipy.integrate.solve_ivp LSODA + brentq) cuando no hay película externa")
    not_included = ["difusión multicomponente (Stefan-Maxwell) y flujo viscoso por cambio de moles dentro del pellet",
                    "varias reacciones o reacción reversible; desactivación y actividad no uniforme (cáscara de huevo)",
                    "pellets finitos o con forma (anillos, trilóbulos): solo placa, cilindro largo y esfera",
                    "estabilidad dinámica de los estados estacionarios (solo se calculan los estados)"]
    if not res["converged"]:
        return {"status": "no_convergence", "evidence": res["evidence"], "warnings": warnings, "assumptions": assumptions,
                "not_included": not_included, "method": method}
    sm, ev = res["selected"], res["evidence"]
    Cs, Ts = sm["surface_concentration_mol_m3"], sm["surface_temperature_K"]
    a = L / (p.s + 1.0)
    r_obs = sm["observed_rate_mol_m3_s"]
    k_s = float(rate_constant(kin, Ts))
    n_app = apparent_order(kin, Cb, Tb)
    if kin["type"] == "power_law":
        thiele = L * math.sqrt(k_s * Cs ** (kin["n"] - 1.0) / De) if (Cs > 0 or kin["n"] >= 1) else math.inf
        thiele_def = "L·(k·Cs^(n−1)/De)^½ con L = semiespesor o radio, a condiciones de superficie"
    else:
        thiele = L * math.sqrt(k_s / De)
        thiele_def = "L·(k/De)^½ con L = semiespesor o radio (k de r = k·C/(1+K·C)^m), a temperatura de superficie"
    Phi = p.generalized_modulus(Cs, Ts)
    out: dict = {
        "status": "ok", "geometry": geometry, "characteristic_length_m": L, "volume_to_surface_m": a,
        "effective_diffusivity_m2_s": De, "effective_diffusivity": diffusivity,
        "eta_internal": sm["eta_internal"], "eta_external": sm["eta_external"], "eta_overall": sm["eta_overall"],
        "definitions": {"eta_internal": "velocidad real / velocidad a condiciones de la superficie del pellet",
                        "eta_external": "velocidad a condiciones de superficie / velocidad a condiciones del fluido",
                        "eta_overall": "velocidad real / velocidad a condiciones del fluido = eta_internal·eta_external",
                        "thiele_modulus": thiele_def,
                        "generalized_modulus": "(V/S)·r(Cs)/(2·De·∫0^Cs r dC)^½ (Bischoff/Aris/Petersen); η → 1/Φ cuando "
                                               "Φ es grande, para cualquier cinética y forma"},
        "thiele_modulus": thiele, "generalized_modulus": Phi,
        "asymptotic_eta_internal": 1.0 / Phi,
        "observed_rate_mol_m3_s": r_obs, "rate_at_bulk_conditions_mol_m3_s": sm["rate_at_bulk_mol_m3_s"],
        "rate_at_surface_conditions_mol_m3_s": sm["rate_at_surface_mol_m3_s"],
        "surface_concentration_mol_m3": Cs, "surface_temperature_K": Ts,
        "center_concentration_mol_m3": sm["center_concentration_mol_m3"],
        "center_temperature_K": sm["center_temperature_K"], "max_temperature_K": sm["max_temperature_K"],
        "film_temperature_rise_K": Ts - Tb, "internal_temperature_rise_K": sm["max_temperature_K"] - Ts,
        "steady_states": res["steady_states"], "multiple_steady_states": len(res["steady_states"]) > 1,
        "selected_branch": branch, "profile": res["profile"]}
    if pellet_density_kg_m3:
        out["observed_rate_mol_kg_s"] = r_obs / float(pellet_density_kg_m3)
    dimless: dict = {}
    if kc is not None:
        dimless["biot_mass"] = kc * L / De
    if dH is not None and dH != 0.0:
        dimless["arrhenius_gamma_surface"] = kin["E"] / (R_GAS * Ts)
        if lam is not None:
            dimless["prater_beta_surface"] = -dH * De * Cs / (lam * Ts)
            dimless["prater_max_internal_rise_K"] = -dH * De * Cs / lam
            if h is not None:
                dimless["biot_heat"] = h * L / lam
    out["dimensionless_groups"] = dimless
    wp = r_obs * L * L / (De * Cs) if Cs > 0 else math.inf
    criteria = {"weisz_prater": {"value": wp, "limit": WEISZ_PRATER_LIMIT, "formula": "ℜ·L²/(De·Cs), ℜ por volumen de pellet",
                                 "internal_diffusion_negligible": bool(wp <= WEISZ_PRATER_LIMIT),
                                 "generalized_value": sm["eta_internal"] * Phi * Phi}}
    if kc is not None:
        v = abs(n_app) * r_obs * L / (kc * Cb)
        criteria["mears_mass"] = {"value": v, "limit": MEARS_LIMIT, "formula": "|n|·ℜ·L/(kc·Cb), n = orden aparente",
                                  "external_mass_transfer_negligible": bool(v < MEARS_LIMIT)}
    if dH is not None and dH != 0.0 and h is not None:
        v = abs(dH) * r_obs * L * kin["E"] / (h * Tb * Tb * R_GAS)
        criteria["mears_heat"] = {"value": v, "limit": MEARS_LIMIT, "formula": "|ΔH|·ℜ·L·E/(h·Tb²·R)",
                                  "film_temperature_rise_negligible": bool(v < MEARS_LIMIT)}
    if dH is not None and dH != 0.0 and lam is not None:
        v = abs(dH) * r_obs * L * L * kin["E"] / (lam * Ts * Ts * R_GAS)
        criteria["anderson"] = {"value": v, "limit": ANDERSON_LIMIT, "formula": "|ΔH|·ℜ·L²·E/(λe·Ts²·R)",
                                "pellet_isothermal": bool(v < ANDERSON_LIMIT)}
    out["criteria"] = criteria
    assumptions.append("los criterios de Weisz-Prater, Mears y Anderson se evalúan con la velocidad YA resuelta y con L = "
                       "semiespesor o radio: son diagnósticos, el resultado riguroso son los factores de efectividad")
    if p.dead_core:
        method = ("orden < 0.5 isotérmico: disparo de un parámetro desde el centro o desde el frente de núcleo muerto "
                  "(scipy.integrate.solve_ivp LSODA + brentq), con la solución local u = A·t^(2/(1−n)) en el frente")
        out["method"] = method
        if ev.get("dead_core_radius_over_L", 0.0) > 0.0:
            warnings.append(f"núcleo muerto: el reactivo se agota en r/L < {ev['dead_core_radius_over_L']:.3f}; ese "
                            "volumen de catalizador no trabaja")
    elif kin["type"] == "power_law" and kin["n"] < 1.0:
        assumptions.append(f"orden n < 1: la potencia C^n se suaviza por debajo de {REG_EPS:.0e}·C del fluido para resolver "
                           "el frente de núcleo muerto (efecto en η menor que la tolerancia de malla)")
    if sm["eta_internal"] < 0.95 and sm["eta_internal"] <= 1.0:
        warnings.append(f"limitación por difusión interna: η interno = {sm['eta_internal']:.3f} (módulo generalizado "
                        f"{Phi:.2f})")
    if sm["eta_external"] < 0.95:
        warnings.append(f"limitación externa: la velocidad a condiciones de superficie es {sm['eta_external']:.3f} de la "
                        "evaluada a condiciones del fluido")
    if sm["eta_overall"] > 1.0 + 1e-9:
        warnings.append(f"η global = {sm['eta_overall']:.3f} > 1: el pellet trabaja más caliente que el fluido "
                        "(reacción exotérmica) o la cinética tiene orden aparente negativo")
    if out["multiple_steady_states"]:
        warnings.append(f"se encontraron {len(res['steady_states'])} estados estacionarios para estas condiciones; se "
                        f"reporta el de la rama '{branch}'. Cuál se alcanza depende de cómo se arranca el reactor")
    elif res["multiplicity_possible"]:
        assumptions.append("se buscaron estados estacionarios múltiples (ver evidence.multiplicity_search) y solo se "
                           "encontró uno")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included, method=out.get("method", method),
               evidence=ev)
    return out
