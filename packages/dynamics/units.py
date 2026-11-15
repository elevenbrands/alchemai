"""First-principles unit models for the dynamic flowsheet (`packages.dynamics.flowsheet`): pure functions, SI units.

Liquid side (incompressible liquid of constant density ρ and heat capacity cp, both the user's):
  * flow laws of a stream — linear resistance q = Δh/R; gravity discharge through an orifice q = Cd·a·√(2gΔh)
    (Torricelli); control valve q[m³/h] = Kv·f(x)·√(ΔP[bar]/SG) (IEC 60534 sizing equation for non-choked liquid,
    Kv = 0.865·Cv); centrifugal pump given by the user's head curve H(q) against a system curve (static head, pressure
    difference, optional valve and piping loss) solved with brentq at every evaluation; a pipe with liquid inertia
    dv/dt = g·Δh/L − f·v|v|/(2D) (Luyben's gravity-flow tank).
  * reaction rate r = k(T)·Πc_i^n_i with k = k0·exp(−Ea/RT) or a constant k (power law, mol/(m³·s)).

Gas side (single component):
  * ideal gas with constant heat-capacity ratio: P = mRT/(MV), u = cv·T, h = cp·T (reference T = 0 K, harmless for
    one component), or a real gas from CoolProp's Helmholtz equation of state with (density, internal energy) as state.
  * orifice / restriction: isentropic nozzle with discharge coefficient, ṁ = Cd·A·√(2ρ0P0·k/(k−1)·(r^(2/k) − r^((k+1)/k)))
    for r = P_down/P_up above the critical ratio (2/(k+1))^(k/(k−1)), and the choked value
    ṁ = Cd·A·√(kρ0P0)·(2/(k+1))^((k+1)/(2(k−1))) below it. With CoolProp the same formulas are evaluated with the real
    upstream density and the real cp/cv (an approximation, declared by the caller in `assumptions`).

Square-root laws are non-Lipschitz at zero driving force; two-sided ones (level difference, pressure ratio near 1) are
made linear inside a tiny band (`REG_HEAD_M`, `REG_PRESSURE_RATIO`), a numerical setting that is reported."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

G = 9.80665                 # standard gravity, m/s²
R_GAS = 8.314462618         # J/(mol K), CODATA 2018
KV_PER_CV = 0.865           # Kv [m³/h, bar] per Cv [US gpm, psi] (IEC 60534-1)
REG_HEAD_M = 1e-9           # |Δh| below which a two-sided √ law is linear (numerical regularisation)
REG_PRESSURE_RATIO = 1e-7   # 1 − P_down/P_up below which the subsonic gas flow is linear (numerical regularisation)

LIQUID_LAWS = ("fixed", "linear_resistance", "gravity_orifice", "valve", "pump", "inertial_pipe", "overflow")
GAS_LAWS = ("fixed", "gas_orifice")
CHARACTERISTICS = ("linear", "equal_percentage", "quick_opening")


def ssqrt(x: float, eps: float) -> float:
    """Signed square root, linear for |x| < eps (continuous; removes the infinite slope at zero)."""
    ax = abs(x)
    if ax >= eps:
        return math.copysign(math.sqrt(ax), x)
    return x / math.sqrt(eps)


def valve_fraction(opening: float, characteristic: str, rangeability: float | None) -> float:
    """Inherent characteristic f(x) = Kv(x)/Kv_max for a stem position x clipped to [0, 1]."""
    x = min(max(opening, 0.0), 1.0)
    if characteristic == "linear":
        return x
    if characteristic == "quick_opening":
        return math.sqrt(x)
    return rangeability ** (x - 1.0)       # equal percentage: f(0) = 1/R, the valve does not shut tight


def valve_flow(kv_m3_h: float, fraction: float, dp_Pa: float, density_kg_m3: float, two_sided: bool) -> float:
    """Liquid flow through a valve, m³/s: q[m³/h] = Kv·f·√(ΔP[bar]/(ρ/1000)); negative ΔP only if two_sided."""
    x = dp_Pa / 1e5 * 1000.0 / density_kg_m3
    if x <= 0.0 and not two_sided:
        return 0.0
    return kv_m3_h * fraction / 3600.0 * ssqrt(x, REG_HEAD_M)


def pump_flow(curve_q: np.ndarray, curve_h: np.ndarray, speed: float, required_head_at_zero_m: float,
              loss_coefficient_s2_m5: float) -> tuple[float, bool]:
    """Operating point of a pump: H_pump(q) = H0 + K·q², q ≥ 0. The curve is the user's table (linear interpolation),
    scaled by the affinity laws for a speed fraction. Returns (q, at_end_of_curve). No flow if the shut-off head does
    not exceed the required head (a discharge check valve is assumed)."""
    if speed <= 0.0:
        return 0.0, False
    qs, hs = curve_q * speed, curve_h * speed * speed

    def f(q: float) -> float:
        return float(np.interp(q, qs, hs)) - required_head_at_zero_m - loss_coefficient_s2_m5 * q * q

    if f(0.0) <= 0.0:
        return 0.0, False
    if f(qs[-1]) >= 0.0:
        return float(qs[-1]), True
    return float(brentq(f, 0.0, qs[-1], xtol=1e-15, rtol=1e-13)), False


def reaction_rate(rx: dict, conc: np.ndarray, temperature_K: float) -> float:
    """Power-law rate, mol/(m³ s). `rx` is a compiled reaction: k0, Ea (0 for a constant k) and (index, order) pairs."""
    k = rx["k0"] * math.exp(-rx["Ea"] / (R_GAS * temperature_K)) if rx["Ea"] != 0.0 else rx["k0"]
    r = k
    for i, n in rx["orders"]:
        c = conc[i]
        if c <= 0.0:
            return 0.0
        r *= c if n == 1.0 else c ** n
    return r


def critical_pressure_ratio(k: float) -> float:
    return (2.0 / (k + 1.0)) ** (k / (k - 1.0))


def orifice_mass_flow(cd_area_m2: float, p_up_Pa: float, rho_up_kg_m3: float, k: float, p_down_Pa: float) -> float:
    """Isentropic nozzle with discharge coefficient, kg/s (≥ 0; the caller orders upstream and downstream)."""
    if p_up_Pa <= 0.0 or p_down_Pa >= p_up_Pa or cd_area_m2 <= 0.0:
        return 0.0
    r = p_down_Pa / p_up_Pa
    if r <= critical_pressure_ratio(k):
        return cd_area_m2 * math.sqrt(k * rho_up_kg_m3 * p_up_Pa) * (2.0 / (k + 1.0)) ** ((k + 1.0) / (2.0 * (k - 1.0)))

    def sub(rr: float) -> float:
        return cd_area_m2 * math.sqrt(2.0 * rho_up_kg_m3 * p_up_Pa * k / (k - 1.0) * (rr ** (2.0 / k) - rr ** ((k + 1.0) / k)))

    if 1.0 - r < REG_PRESSURE_RATIO:
        return sub(1.0 - REG_PRESSURE_RATIO) * (1.0 - r) / REG_PRESSURE_RATIO
    return sub(r)


class IdealGas:
    """Single ideal gas with constant cp/cv. State of a vessel: mass and (unless isothermal) internal energy."""
    model = "ideal"

    def __init__(self, molar_mass_kg_mol: float, heat_capacity_ratio: float | None):
        self.M = molar_mass_kg_mol
        self.k = heat_capacity_ratio
        self.Rs = R_GAS / molar_mass_kg_mol
        self.cv = self.Rs / (heat_capacity_ratio - 1.0) if heat_capacity_ratio else None
        self.cp = self.cv * heat_capacity_ratio if heat_capacity_ratio else None

    def from_TP(self, T: float, P: float) -> tuple[float, float, float]:
        """(density, specific internal energy, specific enthalpy)."""
        return P / (self.Rs * T), (self.cv * T if self.cv else 0.0), (self.cp * T if self.cp else 0.0)

    def from_rho_u(self, rho: float, u: float) -> tuple[float, float, float, float]:
        """(T, P, h, k)."""
        T = u / self.cv
        return T, rho * self.Rs * T, self.cp * T, self.k

    def from_rho_T(self, rho: float, T: float) -> tuple[float, float, float, float]:
        return T, rho * self.Rs * T, (self.cp * T if self.cp else 0.0), (self.k if self.k else 1.0)


class CoolPropGas:
    """Real single-component gas from CoolProp (Helmholtz EOS). Single-phase states only: a two-phase state raises."""
    model = "coolprop"

    def __init__(self, fluid: str):
        import CoolProp
        import CoolProp.CoolProp as CP
        self.CP = CoolProp
        try:
            self.state = CP.AbstractState("HEOS", fluid)
        except Exception as e:
            raise ValueError(f"CoolProp no reconoce el fluido '{fluid}': {str(e)[:120]}") from e
        self.fluid = fluid
        self.M = self.state.molar_mass()
        self.version = CoolProp.__version__

    def _check(self):
        q = self.state.Q()
        if 0.0 <= q <= 1.0 and self.state.phase() == self.CP.iphase_twophase:
            raise ValueError(f"el gas '{self.fluid}' entra en la región de dos fases (condensación): el modelo de recipiente "
                             "de gas solo cubre una fase")

    def from_TP(self, T: float, P: float) -> tuple[float, float, float]:
        s = self.state
        s.update(self.CP.PT_INPUTS, P, T)
        return s.rhomass(), s.umass(), s.hmass()

    def from_rho_u(self, rho: float, u: float) -> tuple[float, float, float, float]:
        s = self.state
        s.update(self.CP.DmassUmass_INPUTS, rho, u)
        self._check()
        return s.T(), s.p(), s.hmass(), s.cpmass() / s.cvmass()

    def from_rho_T(self, rho: float, T: float) -> tuple[float, float, float, float]:
        s = self.state
        s.update(self.CP.DmassT_INPUTS, rho, T)
        self._check()
        return T, s.p(), s.hmass(), s.cpmass() / s.cvmass()
