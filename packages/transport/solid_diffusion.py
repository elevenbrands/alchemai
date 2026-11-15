"""Fickian diffusion WITHOUT reaction in solids, pellets and dense membranes: transient uptake/release and permeation.

What it answers:
  * `transient_diffusion`: unsteady diffusion in a slab (one or two exposed faces), long cylinder, sphere or semi-infinite
    solid, with a fixed surface concentration, a convective film (mass Biot number with a fluid/solid partition coefficient)
    or a well-stirred bath of limited volume (Crank); constant D or D(c) (table, exponential or linear law). Finite cylinder,
    rectangular bar and block by the product of 1-D solutions (constant D, constant surroundings).
  * `membrane_permeation`: solution-diffusion permeation (permeability = D·S) through a film or a laminate of layers in
    series, flat, cylindrical or spherical wall, with optional external films: steady flux, layer resistances, profile, and
    the transient with the time lag (Daynes-Barrer, θ = L²/6D for one flat layer) and the downstream cumulative curve.
  * `fit_diffusivity_from_uptake`: D (and optionally M∞) from the user's uptake-versus-time data, with confidence interval
    and residual diagnostics.

Numerical engine (one for everything): conservative finite volumes on a mesh clustered at the exposed surfaces, in the
variable φ(c) = ∫D dc (Kirchhoff) so that D(c) needs no face averaging, with the exact steady conductance between nodes
(∫dr/r^m), integrated in time by the method of lines with scipy `solve_ivp` (LSODA, banded Jacobian). FEniCSx is not used.

Analytical solutions implemented here only to seal each run (`evidence`): the eigenfunction series of Crank, "The
Mathematics of Diffusion" (2nd ed., ch. 4-6) / Carslaw & Jaeger for constant D — slab β·tanβ = Bi, cylinder β·J1 = Bi·J0,
sphere β·cotβ = 1 − Bi, the limited-volume roots tan q = −αq (slab), αq·J0 + 2J1 = 0 (cylinder), tan q = 3q/(3 + αq²)
(sphere) —, the erfc solution of the semi-infinite solid with a film, and Frisch's integral for the time lag
θ = ∫ G·S·(u_ss − u_0) dV / n_ss (G = fraction of the steady resistance upstream of the point), which reduces to L²/6D.

Nothing physical has a default: geometry, sizes, diffusivity, concentrations, coefficients and partition coefficients are
the caller's. Concentrations are per volume of solid; any consistent unit (mol/m³, kg/m³) works because the problem is
linear in the driving force, the keys say mol/m³."""
from __future__ import annotations

import bisect
import math

import numpy as np
from scipy import integrate, linalg, optimize, special, stats

GEOMETRY_EXPONENT = {"slab": 0, "cylinder": 1, "sphere": 2}
PRODUCT_GEOMETRIES = ("finite_cylinder", "rectangular_bar", "block")
SHAPE_FACTOR = {0: 1.0, 1: 2.0 * math.pi, 2: 4.0 * math.pi}      # flat per m², cylinder per m of length, whole sphere
STP_MOLAR_VOLUME_M3_MOL = 22.414e-3        # ideal gas at 0 °C and 1 atm (the convention of permeability units)
ATM_PA = 101325.0
CMHG_PA = 1333.22368
BARRER_SI = 1e-10 * 1e-6 / STP_MOLAR_VOLUME_M3_MOL * 1e-2 / (1e-4 * CMHG_PA)   # mol·m/(m²·s·Pa) per Barrer

# numerical settings (not physics)
N_CELLS = 200
RTOL = 1e-8
MAX_SERIES_TERMS = 4000
MAX_CELLS = 2000                           # per body (or per layer); the refined check runs on twice as many
MAX_TIMES = 200                            # requested instants per call (semi-infinite: one solve per instant)
MAX_TIMES_SEMI_INFINITE = 30
MAX_RHS_EVALUATIONS = 150_000              # per integration: a stalled LSODA is reported in evidence, never left running
SEMI_INFINITE_DEPTH = 12.0                 # depth of the numerical domain in units of sqrt(D_max·t_end); erfc(6) ~ 2e-17
TOL_MASS_BALANCE = 1e-6
TOL_MESH = 2e-3
TOL_TIME = 1e-5
TOL_ANALYTICAL = 2e-3
SHORT_TIME_CENTER_CHANGE = 1e-3            # centre has changed by less than 0.1 % of the driving force: semi-infinite regime
LONG_TIME_ONE_TERM = 0.01                  # first eigenvalue reproduces the unaccomplished fraction within 1 %


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
def _need(value, name: str, purpose: str) -> float:
    if value is None:
        raise ValueError(f"falta {name}: {purpose}; es un dato del usuario y no se supone")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un número (se recibió {value!r})") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} debe ser un número finito (se recibió {value!r})")
    return v


def _positive(value, name: str, purpose: str) -> float:
    v = _need(value, name, purpose)
    if v <= 0.0:
        raise ValueError(f"{name} debe ser > 0 (se recibió {v:g}): {purpose}")
    return v


def _clean(x):
    """JSON-safe copy: numpy scalars/arrays to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _res(r1, r2, m: int):
    """∫ dr / r^m between two radii: the steady resistance of a shell per unit conductivity and unit shape factor."""
    if m == 0:
        return r2 - r1
    if m == 1:
        return np.log(r2 / r1)
    return 1.0 / r1 - 1.0 / r2


def _vol(r1, r2, m: int):
    """∫ r^m dr: shell volume per unit shape factor."""
    return (r2 ** (m + 1) - r1 ** (m + 1)) / (m + 1)


def _faces(r_in: float, r_out: float, n: int, cluster: str) -> np.ndarray:
    """Cell faces clustered at the outer end, the inner end or both (nested when n is doubled)."""
    xi = np.linspace(0.0, 1.0, n + 1)
    if cluster == "outer":
        s = np.sin(0.5 * np.pi * xi)
    elif cluster == "inner":
        s = 1.0 - np.cos(0.5 * np.pi * xi)
    else:
        s = 0.5 * (1.0 - np.cos(np.pi * xi))
    f = r_in + (r_out - r_in) * s
    f[0], f[-1] = r_in, r_out
    return f


# ---------------------------------------------------------------------------------------------------------------------
# diffusivity D(c) and its Kirchhoff integral φ(c) = ∫ D dc
class _Diffusivity:
    """Constant, exponential D0·exp(β·c), linear D0·(1 + α·c) or piecewise-linear table; `phi` is exact for each."""

    def __init__(self, constant: float | None, law: dict | None):
        if constant is not None and law is not None:
            raise ValueError("dé diffusivity_m2_s (constante) O diffusivity_law (D función de la concentración), no ambos")
        if constant is None and law is None:
            raise ValueError("falta la difusividad en el sólido (diffusivity_m2_s, o diffusivity_law para D(c)): es el dato "
                             "que fija la escala de tiempo y no se estima")
        self.kind = "constant"
        if law is None:
            self.d0 = _positive(constant, "diffusivity_m2_s", "difusividad del soluto en el sólido")
            return
        if not isinstance(law, dict) or "type" not in law:
            raise ValueError("diffusivity_law debe ser un dict con 'type': 'exponential', 'linear' o 'table'")
        self.kind = str(law["type"]).lower()
        if self.kind == "exponential":
            self.d0 = _positive(law.get("D0_m2_s"), "diffusivity_law['D0_m2_s']", "D a concentración cero")
            self.b = _need(law.get("beta_m3_mol"), "diffusivity_law['beta_m3_mol']", "exponente de D = D0·exp(β·c)")
        elif self.kind == "linear":
            self.d0 = _positive(law.get("D0_m2_s"), "diffusivity_law['D0_m2_s']", "D a concentración cero")
            self.b = _need(law.get("alpha_m3_mol"), "diffusivity_law['alpha_m3_mol']", "pendiente de D = D0·(1 + α·c)")
        elif self.kind == "table":
            c = np.asarray(law.get("concentration_mol_m3") or [], dtype=float)
            d = np.asarray(law.get("diffusivity_m2_s") or [], dtype=float)
            if c.size < 2 or c.size != d.size:
                raise ValueError("diffusivity_law de tipo 'table' necesita 'concentration_mol_m3' y 'diffusivity_m2_s' con la "
                                 "misma longitud (≥ 2 puntos)")
            if np.any(np.diff(c) <= 0) or np.any(d <= 0) or not (np.all(np.isfinite(c)) and np.all(np.isfinite(d))):
                raise ValueError("la tabla D(c) debe tener concentraciones estrictamente crecientes y difusividades > 0")
            self.c, self.d = c, d
            self.slope = np.diff(d) / np.diff(c)
            self.phik = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(c))])
            self.c_list, self.d_list = c.tolist(), d.tolist()
            self.slope_list, self.phik_list = self.slope.tolist(), self.phik.tolist()
        else:
            raise ValueError("diffusivity_law['type'] debe ser 'exponential', 'linear' o 'table'")

    @property
    def is_constant(self) -> bool:
        return self.kind == "constant"

    def D(self, c):
        c = np.asarray(c, dtype=float)
        if self.kind == "constant":
            return np.full_like(c, self.d0)
        if self.kind == "exponential":
            return self.d0 * np.exp(self.b * c)
        if self.kind == "linear":
            return self.d0 * (1.0 + self.b * c)
        return np.interp(c, self.c, self.d)

    def phi(self, c):
        c = np.asarray(c, dtype=float)
        if self.kind == "constant":
            return self.d0 * c
        if self.kind == "exponential":
            return self.d0 * c if self.b == 0.0 else self.d0 * np.expm1(self.b * c) / self.b
        if self.kind == "linear":
            return self.d0 * (c + 0.5 * self.b * c * c)
        k = np.clip(np.searchsorted(self.c, c, side="right") - 1, 0, self.c.size - 2)
        dc = c - self.c[k]
        inside = (c >= self.c[0]) & (c <= self.c[-1])
        quad = self.phik[k] + self.d[k] * dc + 0.5 * self.slope[k] * dc * dc
        lo = self.d[0] * (c - self.c[0])
        hi = self.phik[-1] + self.d[-1] * (c - self.c[-1])
        return np.where(inside, quad, np.where(c < self.c[0], lo, hi))

    def D1(self, c: float) -> float:
        """Scalar D (plain floats: the surface Newton iteration calls it thousands of times)."""
        if self.kind == "constant":
            return self.d0
        if self.kind == "exponential":
            return self.d0 * math.exp(self.b * c)
        if self.kind == "linear":
            return self.d0 * (1.0 + self.b * c)
        if c <= self.c_list[0]:
            return self.d_list[0]
        if c >= self.c_list[-1]:
            return self.d_list[-1]
        k = bisect.bisect_right(self.c_list, c) - 1
        return self.d_list[k] + self.slope_list[k] * (c - self.c_list[k])

    def phi1(self, c: float) -> float:
        """Scalar φ."""
        if self.kind == "constant":
            return self.d0 * c
        if self.kind == "exponential":
            return self.d0 * c if self.b == 0.0 else self.d0 * math.expm1(self.b * c) / self.b
        if self.kind == "linear":
            return self.d0 * (c + 0.5 * self.b * c * c)
        if c < self.c_list[0]:
            return self.d_list[0] * (c - self.c_list[0])
        if c > self.c_list[-1]:
            return self.phik_list[-1] + self.d_list[-1] * (c - self.c_list[-1])
        k = min(bisect.bisect_right(self.c_list, c) - 1, len(self.c_list) - 2)
        dc = c - self.c_list[k]
        return self.phik_list[k] + self.d_list[k] * dc + 0.5 * self.slope_list[k] * dc * dc

    def check_range(self, c_lo: float, c_hi: float) -> None:
        if self.kind == "table" and (c_lo < self.c[0] - 1e-12 * abs(self.c[0]) or c_hi > self.c[-1] * (1 + 1e-12)):
            raise ValueError(f"la tabla D(c) cubre {self.c[0]:g}–{self.c[-1]:g} mol/m³ pero la concentración del sólido irá de "
                             f"{c_lo:g} a {c_hi:g} mol/m³: amplíe la tabla, no se extrapola")
        dd = self.D(np.linspace(c_lo, c_hi, 201))
        if np.any(dd <= 0.0) or not np.all(np.isfinite(dd)):
            raise ValueError(f"la ley D(c) da difusividades ≤ 0 o no finitas entre {c_lo:g} y {c_hi:g} mol/m³")

    def bounds(self, c_lo: float, c_hi: float) -> tuple[float, float]:
        dd = self.D(np.linspace(c_lo, c_hi, 201))
        return float(dd.min()), float(dd.max())

    def describe(self) -> dict:
        if self.kind == "constant":
            return {"type": "constant", "diffusivity_m2_s": self.d0}
        if self.kind == "exponential":
            return {"type": "exponential", "D0_m2_s": self.d0, "beta_m3_mol": self.b, "law": "D = D0·exp(β·c)"}
        if self.kind == "linear":
            return {"type": "linear", "D0_m2_s": self.d0, "alpha_m3_mol": self.b, "law": "D = D0·(1 + α·c)"}
        return {"type": "table", "points": int(self.c.size), "law": "interpolación lineal entre los puntos de la tabla"}


# ---------------------------------------------------------------------------------------------------------------------
# analytical series (constant D) — used as evidence of each run
def _bessel_zeros(order: int, n: int) -> np.ndarray:
    """First n positive zeros of J0 or J1: McMahon start + Newton."""
    k = np.arange(1, n + 1, dtype=float)
    if order == 0:
        b = (k - 0.25) * np.pi
        x = b + 1.0 / (8.0 * b)
        for _ in range(8):
            x = x + special.j0(x) / special.j1(x)
    else:
        b = (k + 0.25) * np.pi
        x = b - 3.0 / (8.0 * b)
        for _ in range(8):
            j1 = special.j1(x)
            x = x - j1 / (special.j0(x) - j1 / x)
    return x


def _bisect(f, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lo, hi = lo.astype(float).copy(), hi.astype(float).copy()
    flo = f(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        left = np.sign(fm) == np.sign(flo)
        lo = np.where(left, mid, lo)
        flo = np.where(left, fm, flo)
        hi = np.where(left, hi, mid)
    return 0.5 * (lo + hi)


def eigenvalues(m: int, biot: float, n: int) -> np.ndarray:
    """Roots β_n of slab β·tanβ = Bi, cylinder β·J1(β) = Bi·J0(β), sphere β·cotβ = 1 − Bi (Bi = inf: fixed surface)."""
    k = np.arange(1, n + 1, dtype=float)
    if math.isinf(biot):
        return (k - 0.5) * np.pi if m == 0 else _bessel_zeros(0, n) if m == 1 else k * np.pi
    if m == 0:
        return _bisect(lambda b: b * np.sin(b) - biot * np.cos(b), (k - 1.0) * np.pi, (k - 0.5) * np.pi)
    if m == 1:
        lo = np.concatenate([[0.0], _bessel_zeros(1, n - 1)]) if n > 1 else np.array([0.0])
        return _bisect(lambda b: b * special.j1(b) - biot * special.j0(b), lo, _bessel_zeros(0, n))
    lo = (k - 1.0) * np.pi
    lo[0] = 1e-9
    return _bisect(lambda b: b * np.cos(b) - (1.0 - biot) * np.sin(b), lo, k * np.pi)


def _uptake_weights(m: int, biot: float, beta: np.ndarray) -> np.ndarray:
    """Coefficients w_n of 1 − M_t/M_∞ = Σ w_n·exp(−β_n²·Fo)."""
    base = 2.0 * (m + 1) / beta ** 2
    if math.isinf(biot):
        return base
    s = (1.0, 0.0, -1.0)[m]
    return base / (1.0 + (beta ** 2 + s * biot) / biot ** 2)


def _profile_terms(m: int, beta: np.ndarray, rstar: np.ndarray) -> np.ndarray:
    """C_n·X_n(β_n r*) of the unaccomplished change, shape (positions, terms)."""
    b = beta[None, :]
    x = np.asarray(rstar, dtype=float)[:, None] * b
    if m == 0:
        return 4.0 * np.sin(b) / (2.0 * b + np.sin(2.0 * b)) * np.cos(x)
    if m == 1:
        return 2.0 * special.j1(b) / (b * (special.j0(b) ** 2 + special.j1(b) ** 2)) * special.j0(x)
    return 4.0 * (np.sin(b) - b * np.cos(b)) / (2.0 * b - np.sin(2.0 * b)) * np.sinc(x / np.pi)


def _terms_needed(fo: float) -> int:
    return int(math.ceil(math.sqrt(40.0 / fo) / math.pi)) + 4


def series_fractional_uptake(m: int, biot: float, fo: float, n_terms: int | None = None) -> float:
    """M_t/M_∞ of a slab (m=0, half-thickness), cylinder (1) or sphere (2) at Fourier number fo; Bi = inf for a fixed surface."""
    n = n_terms or min(_terms_needed(fo), MAX_SERIES_TERMS)
    beta = eigenvalues(m, biot, n)
    return float(1.0 - np.sum(_uptake_weights(m, biot, beta) * np.exp(-beta ** 2 * fo)))


def series_unaccomplished_profile(m: int, biot: float, fo: float, rstar) -> np.ndarray:
    """(c − c_eq)/(c0 − c_eq) at r* = r/a (0 = centre or sealed face, 1 = surface)."""
    beta = eigenvalues(m, biot, min(_terms_needed(fo), MAX_SERIES_TERMS))
    return _profile_terms(m, beta, np.atleast_1d(rstar)) @ np.exp(-beta ** 2 * fo)


def bath_roots(m: int, alpha: float, n: int) -> np.ndarray:
    """Non-zero roots q_n for a well-stirred bath of limited volume, α = V_bath/(K·V_solid) (Crank, ch. 4-6)."""
    k = np.arange(1, n + 1, dtype=float)
    if m == 0:
        return _bisect(lambda q: np.sin(q) + alpha * q * np.cos(q), (k - 0.5) * np.pi, k * np.pi)
    if m == 1:
        return _bisect(lambda q: alpha * q * special.j0(q) + 2.0 * special.j1(q), _bessel_zeros(0, n), _bessel_zeros(1, n))
    return _bisect(lambda q: (3.0 + alpha * q * q) * np.sin(q) - 3.0 * q * np.cos(q), k * np.pi, (k + 0.5) * np.pi)


def _bath_weights(m: int, alpha: float, q: np.ndarray) -> np.ndarray:
    nu = m + 1.0
    return 2.0 * nu * alpha * (1.0 + alpha) / (nu * nu * (1.0 + alpha) + alpha * alpha * q * q)


def bath_series_fractional_uptake(m: int, alpha: float, fo: float, n_terms: int | None = None) -> float:
    """M_t/M_∞ from a well-stirred solution of limited volume (no film resistance), Crank ch. 4-6."""
    n = n_terms or min(_terms_needed(fo), MAX_SERIES_TERMS)
    q = bath_roots(m, alpha, n)
    return float(1.0 - np.sum(_bath_weights(m, alpha, q) * np.exp(-q * q * fo)))


def semi_infinite_fraction(x_m, t_s: float, diffusivity: float, h_per_m: float | None):
    """(c − c0)/(c_eq − c0) at depth x of a semi-infinite solid; h = k_c/(K·D) for a film, None for a fixed surface."""
    x = np.asarray(x_m, dtype=float)
    s = math.sqrt(diffusivity * t_s)
    xi = x / (2.0 * s)
    if h_per_m is None:
        return special.erfc(xi)
    return special.erfc(xi) - np.exp(-xi * xi) * special.erfcx(xi + h_per_m * s)


def semi_infinite_uptake_per_area(t_s: float, diffusivity: float, h_per_m: float | None) -> float:
    """∫(c − c0)dx / (c_eq − c0) [m] of a semi-infinite solid."""
    s = math.sqrt(diffusivity * t_s)
    if h_per_m is None:
        return 2.0 * s / math.sqrt(math.pi)
    return 2.0 * s / math.sqrt(math.pi) - (1.0 - float(special.erfcx(h_per_m * s))) / h_per_m


# ---------------------------------------------------------------------------------------------------------------------
# finite-volume / method-of-lines core for one solid body
class _BudgetExceeded(Exception):
    """Raised from a right-hand side when an integration exceeds MAX_RHS_EVALUATIONS (reported, not retried)."""


def _budget_message(exc: _BudgetExceeded) -> str:
    return (f"el integrador LSODA superó {MAX_RHS_EVALUATIONS} evaluaciones sin terminar (detenido en t = {exc.args[0]:.4g} s): "
            "resultado no fiable; revise los datos o reduzca n_cells / la tolerancia")


class _SolidRun:
    """One transient solve: unknowns [c_1..c_n, (C_bath), Q] with Q the cumulative amount through the surface."""

    def __init__(self, m: int, a: float, n: int, diff: _Diffusivity, c0: float, bc: dict, rtol: float, scale: float):
        self.m, self.a, self.n, self.diff, self.c0, self.bc, self.rtol = m, a, n, diff, c0, bc, rtol
        f = _faces(0.0, a, n, "outer")
        self.rc = 0.5 * (f[1:] + f[:-1])
        self.V = _vol(f[:-1], f[1:], m)
        self.g_int = 1.0 / _res(self.rc[:-1], self.rc[1:], m)
        self.g_s = 1.0 / _res(self.rc[-1], a, m)
        self.area = a ** m
        self.vtot = _vol(0.0, a, m)
        self.has_bath = bc["kind"] == "bath"
        self.nq = n + (1 if self.has_bath else 0)
        self.scale = scale
        self.nfev = 0
        self.budget = MAX_RHS_EVALUATIONS
        self.segments: list = []
        self.failed: str | None = None

    # --- boundary -----------------------------------------------------------------------------------------------
    def surface(self, c_last: float, phi_last: float, fluid: float) -> tuple[float, float]:
        """Surface concentration and inward flow (per unit shape factor) given the last cell and the fluid/bath value."""
        bc, diff = self.bc, self.diff
        kc = bc.get("kc")
        if kc is None:
            cs = bc["cs"] if not self.has_bath else bc["K"] * fluid
            return cs, (diff.phi1(cs) - phi_last) * self.g_s
        ka, kp = kc * self.area, bc["K"]
        if diff.is_constant:
            cs = (ka * fluid + diff.d0 * self.g_s * c_last) / (ka / kp + diff.d0 * self.g_s)
            return cs, ka * (fluid - cs / kp)
        lo, hi = sorted((c_last, kp * fluid))
        cs = 0.5 * (lo + hi)
        for _ in range(100):
            fval = (diff.phi1(cs) - phi_last) * self.g_s - ka * (fluid - cs / kp)
            if fval > 0.0:
                hi = cs
            else:
                lo = cs
            new = cs - fval / (diff.D1(cs) * self.g_s + ka / kp)
            if not lo <= new <= hi:
                new = 0.5 * (lo + hi)
            if abs(new - cs) <= 1e-15 * max(abs(new), self.scale):
                cs = new
                break
            cs = new
        return cs, ka * (fluid - cs / kp)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        self.nfev += 1
        if self.nfev > self.budget:
            raise _BudgetExceeded(t)
        n = self.n
        c = y[:n]
        phi = self.diff.phi(c)
        f = (phi[1:] - phi[:-1]) * self.g_int
        fluid = y[n] if self.has_bath else self.bc.get("cf", 0.0)
        _, fs = self.surface(float(c[-1]), float(phi[-1]), float(fluid))
        out = np.empty_like(y)
        dc = np.zeros(n)
        dc[:-1] += f
        dc[1:] -= f
        dc[-1] += fs
        out[:n] = dc / self.V
        if self.has_bath:
            out[n] = -fs / self.bc["vb"]
        out[-1] = fs
        return out

    def jac(self, t: float, y: np.ndarray) -> np.ndarray:
        """Analytic Jacobian in LSODA's packed band format (lband 2, uband 1): packed[1 + i − j, j] = ∂f_i/∂y_j.
        A finite-difference Jacobian is useless here: with cells of 1e-5·a the round-off of the flux differences swamps it."""
        n, ny = self.n, y.size
        c = y[:n]
        dcoef = self.diff.D(c)
        fluid = float(y[n]) if self.has_bath else self.bc.get("cf", 0.0)
        kc = self.bc.get("kc")
        d_last = float(dcoef[-1])
        if kc is None:
            dfs_dc = -d_last * self.g_s
            dfs_db = self.diff.D1(self.bc["K"] * fluid) * self.bc["K"] * self.g_s if self.has_bath else 0.0
        else:
            ka, kp = kc * self.area, self.bc["K"]
            cs, _ = self.surface(float(c[-1]), self.diff.phi1(float(c[-1])), fluid)
            den = self.diff.D1(cs) * self.g_s + ka / kp
            dfs_dc = -(ka / kp) * d_last * self.g_s / den
            dfs_db = ka * (1.0 - (ka / kp) / den)
        pk = np.zeros((4, ny))
        diag = np.zeros(n)
        diag[:-1] -= self.g_int
        diag[1:] -= self.g_int
        pk[1, :n] = diag * dcoef / self.V
        pk[1, n - 1] += dfs_dc / self.V[-1]
        pk[0, 1:n] = self.g_int * dcoef[1:] / self.V[:-1]          # ∂f_i/∂c_{i+1}
        pk[2, :n - 1] = self.g_int * dcoef[:-1] / self.V[1:]       # ∂f_{i+1}/∂c_i
        if self.has_bath:
            vb = self.bc["vb"]
            pk[0, n] = dfs_db / self.V[-1]                         # ∂f_{n−1}/∂C_b
            pk[2, n - 1] = -dfs_dc / vb                            # ∂(dC_b/dt)/∂c_{n−1}
            pk[1, n] = -dfs_db / vb
            pk[3, n - 1] = dfs_dc                                  # ∂(dQ/dt)/∂c_{n−1}
            pk[2, n] = dfs_db                                      # ∂(dQ/dt)/∂C_b
        else:
            pk[2, n - 1] = dfs_dc                                  # ∂(dQ/dt)/∂c_{n−1}
        return pk

    # --- integration -------------------------------------------------------------------------------------------
    def uptake(self, y: np.ndarray) -> float:
        return float(np.dot(self.V, y[:self.n] - self.c0))

    def run(self, times: np.ndarray, target: float | None, m_inf: float | None, t_cap: float | None) -> dict:
        y0 = np.full(self.nq + 1, self.c0, dtype=float)
        if self.has_bath:
            y0[self.n] = self.bc["cb0"]
        y0[-1] = 0.0
        atol = np.full(self.nq + 1, 1e-4 * self.rtol * self.scale)
        atol[-1] = 1e-4 * self.rtol * self.scale * self.vtot
        ev = None
        t_end = float(times[-1]) if times.size else 0.0
        if target is not None:
            def ev(t, y):
                return self.uptake(y) / m_inf - target
            # One pass from t = 0, never a restart: LSODA restarted from a smooth state begins in its non-stiff mode and,
            # at the tight tolerance of the check run, can crawl for minutes before switching to BDF. With requested
            # instants the event is not terminal and the run goes on to the horizon (cheap once the profile is flat).
            ev.terminal = not times.size
            ev.direction = 1.0
            t_end = max(t_end, float(t_cap))
        kw = dict(method="LSODA", rtol=self.rtol, atol=atol, lband=2, uband=1, jac=self.jac, dense_output=True)
        states, t_hit = np.empty((0, self.nq + 1)), None
        try:
            s = integrate.solve_ivp(self.rhs, (0.0, t_end), y0, t_eval=times if times.size else None, events=ev, **kw)
        except _BudgetExceeded as exc:
            self.failed = _budget_message(exc)
            return {"states": states, "t_hit": None}
        if not s.success:
            self.failed = f"el integrador LSODA no terminó: {s.message}"
            return {"states": states, "t_hit": None}
        self.segments.append(s.sol)
        if times.size:
            states = s.y.T
        if ev is not None and s.t_events[0].size:
            t_hit = float(s.t_events[0][0])
        return {"states": states, "t_hit": t_hit}

    def state_at(self, t: float) -> np.ndarray:
        for sol in self.segments:
            if sol.t_min <= t <= sol.t_max:
                return sol(t)
        return self.segments[-1](t)

    # --- post-processing ---------------------------------------------------------------------------------------
    def describe_state(self, y: np.ndarray) -> dict:
        n = self.n
        c = y[:n]
        fluid = y[n] if self.has_bath else self.bc.get("cf", 0.0)
        cs, fs = self.surface(float(c[-1]), self.diff.phi1(float(c[-1])), float(fluid))
        r0, r1 = self.rc[0], self.rc[1]
        center = c[0] - (c[1] - c[0]) * r0 * r0 / (r1 * r1 - r0 * r0)      # zero-gradient parabola through the 2 first nodes
        return {"c": c, "surface": float(cs), "center": float(center), "flow": float(fs), "flux": float(fs) / self.area,
                "uptake": self.uptake(y), "cumulative": float(y[-1]), "bath": float(y[n]) if self.has_bath else None}

    def profile(self, st: dict, positions: np.ndarray) -> np.ndarray:
        r = np.concatenate([[0.0], self.rc, [self.a]])
        c = np.concatenate([[st["center"]], st["c"], [st["surface"]]])
        return np.interp(positions, r, c)


def _solid_boundary(surface_condition, surface_concentration, fluid_concentration, kc, kp, bath_volume, solid_volume, c0,
                    semi_infinite: bool) -> tuple[dict, float, list[str]]:
    """Validated boundary description, equilibrium solid concentration and assumptions."""
    notes: list[str] = []
    if surface_condition is None:
        raise ValueError("falta surface_condition: 'fixed_concentration' (concentración de superficie impuesta), 'convective' "
                         "(película externa con coeficiente de transferencia) o 'limited_bath' (baño agitado de volumen finito)")
    kind = str(surface_condition).lower()
    if kind == "fixed_concentration":
        cs = _need(surface_concentration, "surface_concentration_mol_m3",
                   "concentración en el sólido justo en la superficie expuesta (ya en equilibrio con el fluido)")
        notes.append("Superficie a concentración fija desde t = 0: resistencia externa despreciable (Biot de masa → ∞).")
        return {"kind": "fixed", "cs": cs}, cs, notes
    if kind not in ("convective", "limited_bath"):
        raise ValueError("surface_condition debe ser 'fixed_concentration', 'convective' o 'limited_bath'")
    kp_v = _positive(kp, "partition_coefficient", "reparto de equilibrio K = c_sólido/c_fluido en la superficie (adimensional, "
                     "ambas en mol/m³); ojo: Geankoplis usa el inverso K = c_fluido/c_sólido")
    if kind == "convective":
        cf = _need(fluid_concentration, "fluid_concentration_mol_m3", "concentración del soluto en el seno del fluido")
        k = _positive(kc, "mass_transfer_coefficient_m_s", "coeficiente de película del lado del fluido, k_c")
        notes.append("Película externa: flujo = k_c·(c_fluido − c_superficie/K), fluido de concentración constante.")
        return {"kind": "convective", "kc": k, "K": kp_v, "cf": cf}, kp_v * cf, notes
    if semi_infinite:
        raise ValueError("un baño de volumen limitado no tiene sentido con un sólido semi-infinito: use slab, cylinder o sphere")
    cb0 = _need(fluid_concentration, "fluid_concentration_mol_m3", "concentración INICIAL del baño")
    vb = _positive(bath_volume, "bath_volume_m3", "volumen de líquido del baño (excluido el sólido)")
    vs = _positive(solid_volume, "solid_volume_m3", "volumen total de sólido sumergido en el baño")
    c_inf_fluid = (vb * cb0 + vs * c0) / (vb + vs * kp_v)
    bc = {"kind": "bath", "K": kp_v, "cb0": cb0, "ratio": vb / vs, "alpha": vb / (kp_v * vs)}
    if kc is not None:
        bc["kc"] = _positive(kc, "mass_transfer_coefficient_m_s", "coeficiente de película del lado del fluido, k_c")
        notes.append("Baño perfectamente agitado de volumen finito con película externa k_c.")
    else:
        notes.append("Baño perfectamente agitado de volumen finito SIN resistencia de película (caso de Crank): la superficie "
                     "está en equilibrio con el baño en todo momento; dé mass_transfer_coefficient_m_s si la película cuenta.")
    return bc, kp_v * c_inf_fluid, notes


def _solve_1d(m: int, a: float, diff: _Diffusivity, c0: float, bc: dict, c_eq: float, times: np.ndarray,
              target: float | None, n_cells: int, rtol: float, semi_infinite: bool = False) -> dict:
    """Base run + refined mesh + tighter time tolerance, with mass balance and (constant D) the analytical series."""
    cs_like = [c0, c_eq] + ([bc["K"] * bc["cb0"]] if bc["kind"] == "bath" else [])
    c_lo, c_hi = min(cs_like), max(cs_like)
    diff.check_range(c_lo, c_hi)
    d_min, d_max = diff.bounds(c_lo, c_hi)
    scale = max(abs(c_eq - c0), abs(c_hi - c_lo))
    dc = c_eq - c0
    m_inf = None if semi_infinite else _vol(0.0, a, m) * dc
    kc, kp = bc.get("kc"), bc.get("K")
    biot = math.inf if kc is None else kc * a / (kp * d_max)
    t_cap = None
    if target is not None:
        beta2 = (math.pi ** 2 / 4.0, 5.783, math.pi ** 2)[m]
        tau = a * a / d_min * (1.0 / beta2 + (0.0 if kc is None else kp * d_max / (kc * a) / (m + 1.0)))
        t_cap = max(40.0 * tau * math.log(2.0 / (1.0 - target)), 2.0 * (times[-1] if times.size else 0.0))

    def one(n: int, tol: float) -> tuple[_SolidRun, dict]:
        b = dict(bc)
        if b["kind"] == "bath":
            b["vb"] = _vol(0.0, a, m) * b["ratio"]
        run = _SolidRun(m, a, n, diff, c0, b, tol, scale)
        return run, run.run(times, target, m_inf, t_cap)

    base, out = one(n_cells, rtol)
    fine, out_f = one(2 * n_cells, rtol)
    tight, out_t = one(n_cells, rtol / 100.0)
    reasons: list[str] = [r.failed for r in (base, fine, tight) if r.failed]
    norm = abs(m_inf) if m_inf is not None else None

    states = [base.describe_state(y) for y in out["states"]]
    ev: dict = {"cells": n_cells, "rtol": rtol, "rhs_evaluations": base.nfev}
    if not reasons:
        up = np.array([s["uptake"] for s in states])
        cum = np.array([s["cumulative"] for s in states])
        ref = np.maximum(np.abs(up), 1e-300) if norm is None else norm
        if up.size:
            ev["mass_balance_max_rel_error"] = float(np.max(np.abs(up - cum) / ref))
            if base.has_bath:
                lost = np.array([(bc["cb0"] - s["bath"]) * base.bc["vb"] for s in states])
                ev["bath_balance_max_rel_error"] = float(np.max(np.abs(up - lost) / ref))
            up_f = np.array([fine.uptake(y) for y in out_f["states"]])
            up_t = np.array([tight.uptake(y) for y in out_t["states"]])
            ev["mesh_refinement_max_diff"] = float(np.max(np.abs(up_f - up) / ref))
            ev["time_tolerance_max_diff"] = float(np.max(np.abs(up_t - up) / ref))
            pos = np.linspace(0.0, a, 41)
            prof = np.array([base.profile(s, pos) for s in states])
            prof_f = np.array([fine.profile(fine.describe_state(y), pos) for y in out_f["states"]])
            ev["mesh_refinement_profile_max_diff"] = float(np.max(np.abs(prof_f - prof)) / scale)
        if target is not None:
            hits = [o["t_hit"] for o in (out, out_f, out_t)]
            if any(h is None for h in hits):
                reasons.append("no se alcanzó la fracción objetivo dentro del horizonte de integración")
            else:
                ev["target_time_mesh_rel_diff"] = abs(hits[1] - hits[0]) / hits[0]
                ev["target_time_tolerance_rel_diff"] = abs(hits[2] - hits[0]) / hits[0]
                y_hit = base.state_at(hits[0])
                ev["mass_balance_at_target_rel_error"] = abs(base.uptake(y_hit) - float(y_hit[-1])) / norm
    return {"base": base, "states": states, "t_hit": out["t_hit"], "evidence": ev, "reasons": reasons, "biot": biot,
            "m_inf": m_inf, "d_min": d_min, "d_max": d_max, "dc": dc}


def _analytical_check(m: int, a: float, diff: _Diffusivity, bc: dict, sol: dict, times: np.ndarray, t_hit: float | None,
                      semi_infinite: bool) -> dict:
    """Compare the numerical run with the eigenfunction / erfc solution when D is constant (else say why not)."""
    if not diff.is_constant:
        return {"applicable": False, "why": "D depende de la concentración: no hay serie analítica; la evidencia es balance de "
                                            "masa e independencia de malla y de paso"}
    if bc["kind"] == "bath" and bc.get("kc") is not None:
        return {"applicable": False, "why": "baño finito con película: no se implementó su serie; evidencia por balance y malla"}
    d = diff.d0
    base, states, dc = sol["base"], sol["states"], sol["dc"]
    pos = np.linspace(0.0, a, 41)
    out: dict = {"applicable": True}
    diffs_u, diffs_p, skipped = [], [], 0
    if semi_infinite:
        h = None if bc.get("kc") is None else bc["kc"] / (bc["K"] * d)
        out["solution"] = "sólido semi-infinito: erfc (superficie fija) o erfc − exp·erfc (película), Crank cap. 3"
        for t, s in zip(times, states):
            ref = semi_infinite_uptake_per_area(t, d, h)
            diffs_u.append(abs(s["uptake"] / dc - ref) / ref)
            y = (base.profile(s, pos) - base.c0) / dc
            diffs_p.append(float(np.max(np.abs(y - semi_infinite_fraction(a - pos, t, d, h)))))
    elif bc["kind"] == "bath":
        out["solution"] = "baño agitado de volumen limitado (Crank cap. 4, 5 y 6): tan q = −αq, αq·J0 + 2J1 = 0, tan q = 3q/(3 + αq²)"
        for t, s in zip(times, states):
            fo = d * t / a ** 2
            if _terms_needed(fo) > MAX_SERIES_TERMS:
                skipped += 1
                continue
            diffs_u.append(abs(s["uptake"] / sol["m_inf"] - bath_series_fractional_uptake(m, bc["alpha"], fo)))
    else:
        out["solution"] = ("serie de autofunciones (Crank cap. 4-6): placa β·tanβ = Bi, cilindro β·J1 = Bi·J0, esfera "
                           "β·cotβ = 1 − Bi; Bi → ∞ con superficie fija")
        for t, s in zip(times, states):
            fo = d * t / a ** 2
            if _terms_needed(fo) > MAX_SERIES_TERMS:
                skipped += 1
                continue
            diffs_u.append(abs(s["uptake"] / sol["m_inf"] - series_fractional_uptake(m, sol["biot"], fo)))
            y = (base.profile(s, pos) - sol_ceq(bc, sol)) / (base.c0 - sol_ceq(bc, sol))
            diffs_p.append(float(np.max(np.abs(y - series_unaccomplished_profile(m, sol["biot"], fo, pos / a)))))
        if t_hit is not None:
            fo = d * t_hit / a ** 2
            if _terms_needed(fo) <= MAX_SERIES_TERMS:
                out["target_fractional_uptake_from_series_at_reported_time"] = series_fractional_uptake(m, sol["biot"], fo)
    if bc["kind"] == "bath" and t_hit is not None:
        fo = d * t_hit / a ** 2
        if _terms_needed(fo) <= MAX_SERIES_TERMS:
            out["target_fractional_uptake_from_series_at_reported_time"] = bath_series_fractional_uptake(m, bc["alpha"], fo)
    if diffs_u:
        out["fractional_uptake_max_abs_diff"] = float(max(diffs_u))
    if diffs_p:
        out["profile_max_abs_diff"] = float(max(diffs_p))
    if skipped:
        out["times_skipped_series_too_slow"] = skipped
    return out


def sol_ceq(bc: dict, sol: dict) -> float:
    return sol["base"].c0 + sol["dc"]


def _regime(m: int, biot: float, alpha: float | None, fo: float, center_change: float, series: bool) -> tuple[str, float | None]:
    """Regime label and the one-term (first eigenvalue) fractional uptake when the series applies (constant D)."""
    one_term = None
    long_time = False
    if series and _terms_needed(fo) <= MAX_SERIES_TERMS:
        n = min(_terms_needed(fo), MAX_SERIES_TERMS)
        if alpha is None:
            b = eigenvalues(m, biot, n)
            terms = _uptake_weights(m, biot, b) * np.exp(-b * b * fo)
        else:
            b = bath_roots(m, alpha, n)
            terms = _bath_weights(m, alpha, b) * np.exp(-b * b * fo)
        total = float(np.sum(terms))
        one_term = 1.0 - float(terms[0])
        long_time = total > 0.0 and abs(float(terms[0]) - total) <= LONG_TIME_ONE_TERM * total
    if abs(center_change) < SHORT_TIME_CENTER_CHANGE:
        return "tiempos_cortos_semi_infinito", one_term
    if long_time:
        return "tiempos_largos_primer_autovalor", one_term
    return "intermedio" if series else "intermedio_o_largo_sin_clasificar", one_term


REGIME_CRITERIA = ("tiempos_cortos_semi_infinito: el centro (o la cara sellada) ha cambiado menos del 0.1 % de la fuerza "
                   "impulsora, el sólido se comporta como semi-infinito y la captación crece con √t; "
                   "tiempos_largos_primer_autovalor: el primer término de la serie reproduce la fracción no alcanzada con "
                   "menos del 1 % de error (decaimiento exponencial simple; Fo ≳ 0.2 para superficie fija); intermedio: "
                   "ninguno de los dos. Con D(c) o baño con película no hay serie: solo se distingue el régimen corto.")


def _times(times_s) -> np.ndarray:
    if times_s is None:
        return np.empty(0)
    t = np.unique(np.asarray(list(times_s), dtype=float))
    if t.size and (not np.all(np.isfinite(t)) or t[0] <= 0.0):
        raise ValueError("times_s debe contener tiempos finitos y > 0 (segundos desde que se expone el sólido)")
    return t


def transient_diffusion(geometry: str | None = None, initial_concentration_mol_m3: float | None = None,
                        thickness_m: float | None = None, exposed_faces: int | None = None, radius_m: float | None = None,
                        length_m: float | None = None, width_m: float | None = None, depth_m: float | None = None,
                        diffusivity_m2_s: float | None = None, diffusivity_law: dict | None = None,
                        surface_condition: str | None = None, surface_concentration_mol_m3: float | None = None,
                        fluid_concentration_mol_m3: float | None = None, mass_transfer_coefficient_m_s: float | None = None,
                        partition_coefficient: float | None = None, bath_volume_m3: float | None = None,
                        solid_volume_m3: float | None = None, times_s: list[float] | None = None,
                        target_fractional_uptake: float | None = None, n_cells: int = N_CELLS, profile_points: int = 21,
                        rtol: float = RTOL) -> dict:
    """Unsteady Fickian diffusion (no reaction) in a solid body initially at a uniform concentration.

    geometry: slab (thickness_m + exposed_faces 1|2; the other face sealed), cylinder / sphere (radius_m), semi_infinite,
    or the product geometries finite_cylinder (radius_m, length_m), rectangular_bar (thickness_m, width_m) and block
    (thickness_m, width_m, depth_m) with every face exposed. Returns profiles, average concentration, fractional uptake
    M_t/M_∞, surface flux (positive into the solid), Fourier number, regime and the time to reach a target fraction."""
    if geometry is None:
        raise ValueError("falta geometry: slab, cylinder, sphere, semi_infinite, finite_cylinder, rectangular_bar o block")
    geo = str(geometry).lower()
    c0 = _need(initial_concentration_mol_m3, "initial_concentration_mol_m3", "concentración inicial uniforme del soluto en el sólido")
    times = _times(times_s)
    if target_fractional_uptake is not None and not 0.0 < float(target_fractional_uptake) < 1.0:
        raise ValueError("target_fractional_uptake debe estar entre 0 y 1 (fracción del cambio total hacia el equilibrio)")
    target = None if target_fractional_uptake is None else float(target_fractional_uptake)
    if not times.size and target is None:
        raise ValueError("indique times_s (instantes en los que quiere el perfil) y/o target_fractional_uptake (fracción de "
                         "aproximación al equilibrio cuyo tiempo se busca)")
    if not 20 <= int(n_cells) <= MAX_CELLS:
        raise ValueError(f"n_cells debe estar entre 20 y {MAX_CELLS} (la comprobación de malla usa el doble)")
    if times.size > MAX_TIMES or (str(geometry).lower() == "semi_infinite" and times.size > MAX_TIMES_SEMI_INFINITE):
        raise ValueError(f"demasiados instantes en times_s ({times.size}): máximo {MAX_TIMES} "
                         f"({MAX_TIMES_SEMI_INFINITE} en semi_infinite, que resuelve un dominio por instante)")
    diff = _Diffusivity(diffusivity_m2_s, diffusivity_law)
    if geo in PRODUCT_GEOMETRIES:
        return _product_diffusion(geo, c0, thickness_m, radius_m, length_m, width_m, depth_m, diff, surface_condition,
                                  surface_concentration_mol_m3, fluid_concentration_mol_m3, mass_transfer_coefficient_m_s,
                                  partition_coefficient, solid_volume_m3, times, target, int(n_cells), rtol, exposed_faces)
    semi = geo == "semi_infinite"
    if not semi and geo not in GEOMETRY_EXPONENT:
        raise ValueError(f"geometry '{geometry}' no reconocida: use slab, cylinder, sphere, semi_infinite, finite_cylinder, "
                         "rectangular_bar o block")
    bc, c_eq, notes = _solid_boundary(surface_condition, surface_concentration_mol_m3, fluid_concentration_mol_m3,
                                      mass_transfer_coefficient_m_s, partition_coefficient, bath_volume_m3, solid_volume_m3,
                                      c0, semi)
    if c_eq == c0:
        raise ValueError("el sólido ya está en equilibrio con su entorno (concentración inicial = concentración de equilibrio): "
                         "no hay fuerza impulsora ni transitorio que calcular")
    assumptions = ["Difusión de Fick unidimensional sin reacción, sólido homogéneo e isótropo, sin hinchamiento ni cambio de "
                   "volumen, concentración inicial uniforme.",
                   "K = c_sólido/c_fluido en equilibrio (adimensional); las concentraciones del sólido son por volumen de "
                   "sólido. Cualquier unidad consistente sirve (kg/m³, kg/kg de sólido seco con densidad constante)."] + notes
    warnings: list[str] = []
    if semi:
        if target is not None:
            raise ValueError("un sólido semi-infinito no tiene M_∞: target_fractional_uptake no aplica; pida times_s")
        m, exposed = 0, 1
        _, d_hi = diff.bounds(min(c0, c_eq), max(c0, c_eq))
        diff.check_range(min(c0, c_eq), max(c0, c_eq))
        a = SEMI_INFINITE_DEPTH * math.sqrt(d_hi * float(times[-1]))
        assumptions.append(f"Sólido semi-infinito: cada tiempo se resuelve en un dominio de profundidad {SEMI_INFINITE_DEPTH:g}"
                           "·√(D_max·t) con la cara lejana sellada (el frente no la alcanza).")
    elif geo == "slab":
        m = 0
        thick = _positive(thickness_m, "thickness_m", "espesor total de la placa")
        if exposed_faces not in (1, 2):
            raise ValueError("falta exposed_faces (1 o 2): cuántas caras de la placa están expuestas; con 1 la otra está "
                             "sellada y el camino de difusión es el espesor completo, con 2 es la mitad")
        exposed = int(exposed_faces)
        a = thick / exposed
    else:
        m, exposed = GEOMETRY_EXPONENT[geo], 1
        a = _positive(radius_m, "radius_m", "radio del cilindro largo o de la esfera")
    if semi:        # one solve per requested time, each on a domain scaled with its own penetration depth
        parts = []
        for t in times:
            a_t = SEMI_INFINITE_DEPTH * math.sqrt(d_hi * float(t))
            parts.append((a_t, _solve_1d(0, a_t, diff, c0, bc, c_eq, np.array([float(t)]), None, int(n_cells), float(rtol), True)))
        sol = parts[-1][1]
        per_time = [(float(t), sl["states"][0], sl, a_t) for t, (a_t, sl) in zip(times, parts) if sl["states"]]
    else:
        sol = _solve_1d(m, a, diff, c0, bc, c_eq, times, target, int(n_cells), float(rtol))
        per_time = [(float(t), st, sol, a) for t, st in zip(times, sol["states"])]
    dc, m_inf = sol["dc"], sol["m_inf"]
    reasons = list(sol["reasons"]) if not semi else [r for _, sl in parts for r in sl["reasons"]]
    constant = diff.is_constant
    alpha = bc.get("alpha") if bc["kind"] == "bath" and bc.get("kc") is None else None
    npts = max(int(profile_points), 2)
    pos = np.linspace(0.0, a, npts)
    results = []
    for t, s, sl, a in per_time:
        base = sl["base"]
        row: dict = {"time_s": float(t)}
        if semi:
            pass
        elif constant:
            row["fourier_number"] = diff.d0 * t / a ** 2
        else:
            row["fourier_number_D_min"] = sol["d_min"] * t / a ** 2
            row["fourier_number_D_max"] = sol["d_max"] * t / a ** 2
        cavg_change = s["uptake"] / base.vtot
        if semi:
            row["uptake_mol_m2"] = s["uptake"]
            y = (base.profile(s, np.linspace(0.0, a, 2001)) - c0) / dc
            depth = a - np.linspace(0.0, a, 2001)
            reached = depth[np.abs(y) >= 0.01]
            row["penetration_depth_1pct_m"] = float(reached.max()) if reached.size else 0.0
            row["regime"] = "tiempos_cortos_semi_infinito"
        else:
            frac = s["uptake"] / m_inf
            row.update(fractional_uptake=frac, unaccomplished_fraction=1.0 - frac,
                       average_concentration_mol_m3=c0 + cavg_change, uptake_mol_per_m3_solid=cavg_change)
            if solid_volume_m3 is not None:
                row["uptake_mol"] = cavg_change * float(solid_volume_m3)
            fo_ref = (diff.d0 if constant else sol["d_max"]) * t / a ** 2
            regime, one_term = _regime(m, sol["biot"], alpha, fo_ref, (s["center"] - c0) / dc,
                                       constant and not (bc["kind"] == "bath" and bc.get("kc") is not None))
            row["regime"] = regime
            if one_term is not None:
                row["one_term_fractional_uptake"] = one_term
            row["center_concentration_mol_m3"] = s["center"]
        row["surface_concentration_mol_m3"] = s["surface"]
        row["surface_flux_mol_m2_s"] = s["flux"]
        if s["bath"] is not None:
            row["bath_concentration_mol_m3"] = s["bath"]
        if semi:
            x = np.linspace(0.0, min(a, 4.0 * math.sqrt(sol["d_max"] * t) * 1.5), npts)
            row["profile"] = {"depth_from_surface_m": x.tolist(), "concentration_mol_m3": base.profile(s, a - x).tolist()}
        else:
            row["profile"] = {"position_m": pos.tolist(), "concentration_mol_m3": base.profile(s, pos).tolist()}
        results.append(row)
    out: dict = {"geometry": geo, "diffusion_path_m": None if semi else a, "exposed_faces": exposed if geo == "slab" else None,
                 "diffusivity": diff.describe(), "initial_concentration_mol_m3": c0, "equilibrium_concentration_mol_m3": c_eq,
                 "biot_number_mass": None if math.isinf(sol["biot"]) or semi else sol["biot"],
                 "position_convention": ("profundidad desde la superficie expuesta" if semi else
                                         "position_m = 0 en el centro (o en la cara sellada) y = diffusion_path_m en la "
                                         "superficie expuesta"),
                 "results": results}
    if not constant:
        out["diffusivity"].update(D_min_m2_s=sol["d_min"], D_max_m2_s=sol["d_max"])
        if out["biot_number_mass"] is not None:
            out["biot_number_note"] = "Biot evaluado con D_max del intervalo de concentraciones"
    if bc["kind"] == "bath":
        out["bath"] = {"alpha_bath_capacity_ratio": bc["alpha"], "final_bath_concentration_mol_m3": c_eq / bc["K"],
                       "final_fraction_of_solute_taken_from_bath": 1.0 / (1.0 + bc["alpha"])}
    if not semi:
        out["regime_criteria"] = REGIME_CRITERIA
    if target is not None:
        th = sol["t_hit"]
        out["time_to_target"] = {"target_fractional_uptake": target, "time_s": th}
        if th is not None:
            key = "fourier_number" if constant else "fourier_number_D_max"
            out["time_to_target"][key] = (diff.d0 if constant else sol["d_max"]) * th / a ** 2
        else:
            warnings.append("No se alcanzó la fracción objetivo dentro del horizonte de integración.")
    if semi:
        ev, ana = {}, {}
        for (a_t, sl), t in zip(parts, times):
            one = _analytical_check(0, a_t, diff, bc, sl, np.array([float(t)]), None, True) if not sl["reasons"] else {"applicable": False}
            for src, dst in ((sl["evidence"], ev), (one, ana)):
                for key, val in src.items():        # worst case over the requested times
                    if isinstance(val, float) and isinstance(dst.get(key), float):
                        dst[key] = max(dst[key], val)
                    elif key not in dst:
                        dst[key] = val
    else:
        ev = sol["evidence"]
        ana = _analytical_check(m, a, diff, bc, sol, times, sol["t_hit"], semi) if not reasons else {"applicable": False}
    ev["analytical"] = ana
    checks = [("mass_balance_max_rel_error", TOL_MASS_BALANCE, "el balance de masa (perfil integrado vs flujo de superficie "
               "integrado en el tiempo) no cierra"),
              ("mass_balance_at_target_rel_error", TOL_MASS_BALANCE, "el balance de masa no cierra en el tiempo objetivo"),
              ("bath_balance_max_rel_error", TOL_MASS_BALANCE, "lo que pierde el baño no coincide con lo que gana el sólido"),
              ("mesh_refinement_max_diff", TOL_MESH, "la captación cambia al duplicar la malla (aumente n_cells)"),
              ("mesh_refinement_profile_max_diff", 5 * TOL_MESH, "el perfil cambia al duplicar la malla (aumente n_cells)"),
              ("time_tolerance_max_diff", TOL_TIME, "la captación cambia al apretar 100× la tolerancia temporal"),
              ("target_time_mesh_rel_diff", TOL_MESH, "el tiempo objetivo cambia al duplicar la malla (aumente n_cells)"),
              ("target_time_tolerance_rel_diff", 10 * TOL_TIME, "el tiempo objetivo cambia al apretar la tolerancia temporal")]
    for key, tol, msg in checks:
        if key in ev and not ev[key] <= tol:
            reasons.append(f"{msg}: {ev[key]:.2e} > {tol:.0e}")
    for key in ("fractional_uptake_max_abs_diff", "profile_max_abs_diff"):
        if key in ana and not ana[key] <= TOL_ANALYTICAL:
            reasons.append(f"desviación respecto a la solución analítica ({key}) {ana[key]:.2e} > {TOL_ANALYTICAL:.0e}")
    ev["tolerances"] = {"mass_balance": TOL_MASS_BALANCE, "mesh": TOL_MESH, "time": TOL_TIME, "analytical": TOL_ANALYTICAL}
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    if not constant:
        warnings.append("D variable con la concentración: el resultado depende de la ley D(c) suministrada; no hay solución "
                        "analítica de contraste, la evidencia es numérica (balance, malla, paso).")
    if sol["biot"] < 0.1 and not semi:
        warnings.append(f"Biot de masa = {sol['biot']:.3g} < 0.1: controla la película externa; el perfil interno es casi plano "
                        "y un balance global (capacidad concentrada) daría lo mismo.")
    out.update(warnings=warnings, assumptions=assumptions,
               not_included=["Reacción química o adsorción no lineal dentro del sólido (isoterma no lineal).",
                             "Hinchamiento, encogimiento o frente móvil (secado con contracción, difusión caso II no fickiana).",
                             "Difusividad dependiente del tiempo o de la temperatura durante el transitorio; anisotropía.",
                             "Concentración inicial no uniforme y entorno de concentración variable distinta del baño finito."],
               method=("Volúmenes finitos conservativos en φ(c) = ∫D dc (Kirchhoff) con conductancia estacionaria exacta entre "
                       f"nodos, malla de {int(n_cells)} celdas concentrada en la superficie, método de líneas con scipy "
                       f"solve_ivp LSODA (jacobiano en banda, rtol {rtol:g}); se repite con malla doble y tolerancia 100× más "
                       "estricta, y se contrasta con la serie analítica cuando D es constante."),
               evidence=ev)
    return _clean(out)


def _product_diffusion(geo, c0, thickness_m, radius_m, length_m, width_m, depth_m, diff, surface_condition, cs, cf, kc, kp,
                       solid_volume_m3, times, target, n_cells, rtol, exposed_faces) -> dict:
    """Finite cylinder, rectangular bar and block: product of 1-D unaccomplished fractions (Newman's rule)."""
    if not diff.is_constant:
        raise ValueError("las geometrías finitas por producto de soluciones exigen D constante (la regla del producto no vale "
                         "con D(c)); use slab, cylinder o sphere")
    if exposed_faces not in (None, 2):
        raise ValueError("en finite_cylinder, rectangular_bar y block todas las caras están expuestas (exposed_faces = 2)")
    kind = None if surface_condition is None else str(surface_condition).lower()
    if kind == "limited_bath":
        raise ValueError("la regla del producto exige un entorno de concentración constante: no vale con baño de volumen finito")
    bc, c_eq, notes = _solid_boundary(surface_condition, cs, cf, kc, kp, None, None, c0, False)
    if c_eq == c0:
        raise ValueError("el sólido ya está en equilibrio con su entorno: no hay transitorio que calcular")
    if geo == "finite_cylinder":
        dirs = [("radial", 1, _positive(radius_m, "radius_m", "radio del cilindro finito")),
                ("axial", 0, 0.5 * _positive(length_m, "length_m", "longitud del cilindro finito (ambas tapas expuestas)"))]
    else:
        dirs = [("thickness", 0, 0.5 * _positive(thickness_m, "thickness_m", "primer lado de la sección (expuesto por ambas caras)")),
                ("width", 0, 0.5 * _positive(width_m, "width_m", "segundo lado de la sección (expuesto por ambas caras)"))]
        if geo == "block":
            dirs.append(("depth", 0, 0.5 * _positive(depth_m, "depth_m", "tercer lado del bloque (expuesto por ambas caras)")))
    dc = c_eq - c0
    t_end = float(times[-1]) if times.size else 0.0
    sols, reasons, evid = [], [], {}
    slowest = max(a * a / diff.d0 * (1.0 + (0.0 if bc.get("kc") is None else bc["K"] * diff.d0 / (bc["kc"] * a))) for _, _, a in dirs)
    horizon = max(t_end, 0.0)
    if target is not None:
        horizon = max(horizon, 3.0 * slowest * math.log(2.0 / (1.0 - target)))
    grid = np.unique(np.concatenate([times, [horizon]])) if horizon > 0 else times
    for name, m, a in dirs:
        s = _solve_1d(m, a, diff, c0, bc, c_eq, grid, None, n_cells, rtol)
        s["analytical"] = _analytical_check(m, a, diff, bc, s, grid, None, False) if not s["reasons"] else {"applicable": False}
        sols.append((name, m, a, s))
        reasons += [f"{name}: {r}" for r in s["reasons"]]
        e = dict(s["evidence"])
        e["analytical"] = s["analytical"]
        evid[name] = e
        for key, tol in (("mass_balance_max_rel_error", TOL_MASS_BALANCE), ("mesh_refinement_max_diff", TOL_MESH),
                         ("time_tolerance_max_diff", TOL_TIME)):
            if key in e and not e[key] <= tol:
                reasons.append(f"{name}: {key} = {e[key]:.2e} > {tol:.0e}")
        for key in ("fractional_uptake_max_abs_diff", "profile_max_abs_diff"):
            if key in s["analytical"] and not s["analytical"][key] <= TOL_ANALYTICAL:
                reasons.append(f"{name}: {key} = {s['analytical'][key]:.2e} > {TOL_ANALYTICAL:.0e}")

    def unaccomplished(t: float) -> tuple[float, float, dict]:
        avg, cen, parts = 1.0, 1.0, {}
        for name, m, a, s in sols:
            st = s["base"].describe_state(s["base"].state_at(t))
            ya = 1.0 - st["uptake"] / s["m_inf"]
            yc = 1.0 - (st["center"] - c0) / dc
            parts[name] = {"average_unaccomplished": ya, "center_unaccomplished": yc, "fourier_number": diff.d0 * t / a ** 2,
                           "biot_number_mass": None if math.isinf(s["biot"]) else s["biot"]}
            avg *= ya
            cen *= yc
        return avg, cen, parts

    results = []
    if not reasons:
        for t in times:
            ya, yc, parts = unaccomplished(float(t))
            row = {"time_s": float(t), "fractional_uptake": 1.0 - ya, "unaccomplished_fraction": ya,
                   "average_concentration_mol_m3": c_eq - ya * dc, "center_concentration_mol_m3": c_eq - yc * dc,
                   "center_unaccomplished_fraction": yc, "uptake_mol_per_m3_solid": (1.0 - ya) * dc, "directions": parts}
            if solid_volume_m3 is not None:
                row["uptake_mol"] = (1.0 - ya) * dc * float(solid_volume_m3)
            results.append(row)
    out: dict = {"geometry": geo, "diffusivity": diff.describe(), "initial_concentration_mol_m3": c0,
                 "equilibrium_concentration_mol_m3": c_eq,
                 "diffusion_paths_m": {name: a for name, _, a, _ in sols}, "results": results}
    warnings: list[str] = []
    if target is not None and not reasons:
        f = lambda t: (1.0 - unaccomplished(t)[0]) - target      # noqa: E731
        if f(horizon) < 0.0:
            warnings.append("No se alcanzó la fracción objetivo dentro del horizonte de integración.")
            reasons.append("no se alcanzó la fracción objetivo dentro del horizonte de integración")
            out["time_to_target"] = {"target_fractional_uptake": target, "time_s": None}
        else:
            th = optimize.brentq(f, 1e-12 * horizon, horizon, xtol=1e-12 * horizon, rtol=1e-12)
            out["time_to_target"] = {"target_fractional_uptake": target, "time_s": float(th)}
    evid.update(verified=not reasons, reasons=reasons,
                tolerances={"mass_balance": TOL_MASS_BALANCE, "mesh": TOL_MESH, "time": TOL_TIME, "analytical": TOL_ANALYTICAL})
    out.update(warnings=warnings,
               assumptions=["Regla del producto (Newman): la fracción no alcanzada del cuerpo finito es el producto de las "
                            "fracciones 1-D de cada dirección; es exacta con D constante, concentración inicial uniforme y el "
                            "mismo entorno constante en todas las caras.",
                            "Todas las caras expuestas a la misma condición de superficie; K = c_sólido/c_fluido."] + notes,
               not_included=["Perfil completo 2-D/3-D y flujo local en la superficie (no es uniforme en un cuerpo finito): se dan "
                             "el promedio y el centro.", "D(c), baño de volumen finito y caras selladas en geometrías finitas.",
                             "Reacción, hinchamiento y anisotropía."],
               method=("Producto de soluciones 1-D, cada una resuelta con el mismo motor de volúmenes finitos + LSODA y sellada "
                       "con su serie analítica, balance de masa e independencia de malla y de paso."),
               evidence=evid)
    return _clean(out)


# ---------------------------------------------------------------------------------------------------------------------
# membranes: solution-diffusion through layers in series
def _layer_properties(layer: dict, i: int, mode: str) -> dict:
    if not isinstance(layer, dict):
        raise ValueError(f"layers[{i}] debe ser un dict con el espesor y las propiedades de la capa")
    tag = f"layers[{i}]"
    out = {"name": str(layer.get("name", f"capa {i + 1}")),
           "thickness": _positive(layer.get("thickness_m"), f"{tag}['thickness_m']", "espesor de la capa")}
    d = layer.get("diffusivity_m2_s")
    d = None if d is None else _positive(d, f"{tag}['diffusivity_m2_s']", "difusividad del permeante en la capa")
    if mode == "pressure":
        s = layer.get("solubility_mol_m3_Pa")
        if s is None and layer.get("solubility_m3STP_m3_atm") is not None:
            s = _positive(layer["solubility_m3STP_m3_atm"], f"{tag}['solubility_m3STP_m3_atm']", "solubilidad") \
                / STP_MOLAR_VOLUME_M3_MOL / ATM_PA
        p = layer.get("permeability_mol_m_m2_s_Pa")
        if p is None and layer.get("permeability_barrer") is not None:
            p = _positive(layer["permeability_barrer"], f"{tag}['permeability_barrer']", "permeabilidad") * BARRER_SI
        if p is None and layer.get("permeability_m3STP_m_m2_s_atm") is not None:
            p = _positive(layer["permeability_m3STP_m_m2_s_atm"], f"{tag}['permeability_m3STP_m_m2_s_atm']", "permeabilidad") \
                / STP_MOLAR_VOLUME_M3_MOL / ATM_PA
        names = ("solubility_mol_m3_Pa (o solubility_m3STP_m3_atm)",
                 "permeability_mol_m_m2_s_Pa (o permeability_barrer / permeability_m3STP_m_m2_s_atm)")
    else:
        s = layer.get("partition_coefficient")
        p = layer.get("permeability_m2_s")
        names = ("partition_coefficient (c_capa/c_fluido)", "permeability_m2_s (= D·K)")
    s = None if s is None else _positive(s, f"{tag} solubilidad/reparto", "sorción de equilibrio del permeante en la capa")
    p = None if p is None else _positive(p, f"{tag} permeabilidad", "permeabilidad de la capa")
    given = sum(v is not None for v in (d, s, p))
    if given == 3 and abs(p - d * s) > 1e-6 * p:
        raise ValueError(f"{tag}: la permeabilidad dada ({p:g}) no coincide con D·S = {d * s:g}; dé solo dos de las tres")
    if p is None:
        if d is None or s is None:
            raise ValueError(f"{tag}: faltan propiedades de la capa. Dé diffusivity_m2_s y {names[0]}, o directamente "
                             f"{names[1]} (sirve para el estacionario; el transitorio necesita además D)")
        p = d * s
    elif d is None and s is not None:
        d = p / s
    elif s is None and d is not None:
        s = p / d
    out.update(D=d, S=s, P=p)
    return out


class _Wall:
    """Geometry + layers ordered from the inner radius outward; resistances per unit shape factor."""

    def __init__(self, geometry: str, layers: list[dict], inner_radius: float | None, upstream_inner: bool,
                 film_up: float | None, film_down: float | None):
        self.m = {"flat": 0, "cylindrical": 1, "spherical": 2}[geometry]
        self.up_inner = upstream_inner
        self.layers = layers if upstream_inner else layers[::-1]
        r0 = 0.0 if self.m == 0 else inner_radius
        self.rb = np.concatenate([[r0], r0 + np.cumsum([la["thickness"] for la in self.layers])])
        film_in, film_out = (film_up, film_down) if upstream_inner else (film_down, film_up)
        self.r_film_in = 0.0 if film_in is None else 1.0 / (film_in * self.rb[0] ** self.m)
        self.r_film_out = 0.0 if film_out is None else 1.0 / (film_out * self.rb[-1] ** self.m)
        self.r_layers = np.array([_res(self.rb[i], self.rb[i + 1], self.m) / la["P"] for i, la in enumerate(self.layers)])
        self.r_total = float(self.r_film_in + self.r_layers.sum() + self.r_film_out)

    def r_from_inner(self, r: float, i: int) -> float:
        """Resistance from the inner fluid to radius r inside layer i."""
        return float(self.r_film_in + self.r_layers[:i].sum() + _res(self.rb[i], r, self.m) / self.layers[i]["P"])

    def upstream_fraction(self, r: float, i: int) -> float:
        g = self.r_from_inner(r, i) / self.r_total
        return g if self.up_inner else 1.0 - g

    def frisch_time_lag(self, u_up: float, u_down: float, u_init: float) -> float:
        """θ = ∫ G·S·(u_ss − u_0) dV / n_ss with G the fraction of the steady resistance upstream of the point."""
        n_ss = (u_up - u_down) / self.r_total
        total = 0.0
        for i, la in enumerate(self.layers):
            def f(r, i=i, la=la):
                g = self.upstream_fraction(r, i)
                return g * la["S"] * (u_up - (u_up - u_down) * g - u_init) * r ** self.m
            total += integrate.quad(f, self.rb[i], self.rb[i + 1], epsabs=0.0, epsrel=1e-12)[0]
        return total / n_ss

    def mesh(self, n_per_layer: int) -> dict:
        faces, lay = [self.rb[:1]], []
        for i in range(len(self.layers)):
            f = _faces(self.rb[i], self.rb[i + 1], n_per_layer, "both")
            faces.append(f[1:])
            lay += [i] * n_per_layer
        f = np.concatenate(faces)
        lay = np.array(lay)
        rc = 0.5 * (f[1:] + f[:-1])
        P = np.array([la["P"] for la in self.layers])[lay]
        S = np.array([la["S"] if la["S"] is not None else np.nan for la in self.layers])[lay]
        r_half_lo = _res(f[:-1], rc, self.m) / P
        r_half_hi = _res(rc, f[1:], self.m) / P
        g = 1.0 / (r_half_hi[:-1] + r_half_lo[1:])
        return {"f": f, "rc": rc, "lay": lay, "cap": S * _vol(f[:-1], f[1:], self.m), "g": g,
                "g_in": 1.0 / (self.r_film_in + r_half_lo[0]), "g_out": 1.0 / (r_half_hi[-1] + self.r_film_out)}


def _membrane_transient(wall: _Wall, n_per_layer: int, u_in: float, u_out: float, u_init: float, t_eval: np.ndarray,
                        t_end: float, rtol: float, scale: float) -> dict:
    """Method of lines for the laminate: unknowns [Q_in, u_1..u_N, Q_out] (flows per unit shape factor, inner → outer)."""
    ms = wall.mesh(n_per_layer)
    n, g, cap = ms["rc"].size, ms["g"], ms["cap"]

    nfev = [0]

    def rhs(t, y):
        nfev[0] += 1
        if nfev[0] > MAX_RHS_EVALUATIONS:
            raise _BudgetExceeded(t)
        u = y[1:-1]
        f = (u[1:] - u[:-1]) * g                 # flow from cell i+1 into cell i
        fin = (u_in - u[0]) * ms["g_in"]         # flow entering through the inner face
        fout = (u[-1] - u_out) * ms["g_out"]     # flow leaving through the outer face
        du = np.zeros(n)
        du[:-1] += f
        du[1:] -= f
        du[0] += fin
        du[-1] -= fout
        out = np.empty_like(y)
        out[0], out[1:-1], out[-1] = fin, du / cap, fout
        return out

    pk = np.zeros((3, n + 2))                    # constant Jacobian, packed band format: pk[1 + i − j, j]
    diag = np.zeros(n)
    diag[:-1] -= g
    diag[1:] -= g
    diag[0] -= ms["g_in"]
    diag[-1] -= ms["g_out"]
    pk[1, 1:-1] = diag / cap
    pk[0, 2:-1] = g / cap[:-1]
    pk[2, 1:-2] = g / cap[1:]
    pk[0, 1] = -ms["g_in"]                       # ∂(dQ_in/dt)/∂u_1
    pk[2, n] = ms["g_out"]                       # ∂(dQ_out/dt)/∂u_N
    y0 = np.full(n + 2, u_init, dtype=float)
    y0[0] = y0[-1] = 0.0
    atol = np.full(n + 2, 1e-4 * rtol * scale)
    atol[0] = atol[-1] = 1e-4 * rtol * scale * float(np.sum(cap))
    try:
        s = integrate.solve_ivp(rhs, (0.0, t_end), y0, method="LSODA", rtol=rtol, atol=atol, lband=1, uband=1,
                                jac=lambda t, y: pk, t_eval=np.unique(np.concatenate([t_eval, [t_end]])))
    except _BudgetExceeded as exc:
        return {"failed": _budget_message(exc)}
    if not s.success:
        return {"failed": f"el integrador LSODA no terminó: {s.message}"}
    y = s.y.T
    u = y[:, 1:-1]
    stored = (u - u_init) @ cap
    return {"failed": None, "t": s.t, "q_in": y[:, 0], "q_out": y[:, -1], "stored": stored,
            "f_in": (u_in - u[:, 0]) * ms["g_in"], "f_out": (u[:, -1] - u_out) * ms["g_out"], "mesh": ms}


def _steady_numeric(wall: _Wall, n_per_layer: int, u_in: float, u_out: float) -> float:
    """Steady flow (inner → outer, per unit shape factor) from the finite-volume matrix: independent of the resistance sum."""
    ms = wall.mesh(n_per_layer)
    n, g = ms["rc"].size, ms["g"]
    ab = np.zeros((3, n))
    diag = np.zeros(n)
    diag[:-1] += g
    diag[1:] += g
    diag[0] += ms["g_in"]
    diag[-1] += ms["g_out"]
    ab[0, 1:], ab[1], ab[2, :-1] = -g, diag, -g
    b = np.zeros(n)
    b[0], b[-1] = ms["g_in"] * u_in, ms["g_out"] * u_out
    u = linalg.solve_banded((1, 1), ab, b)
    return float((u_in - u[0]) * ms["g_in"])


def membrane_permeation(layers: list[dict] | None = None, geometry: str | None = None,
                        upstream_partial_pressure_Pa: float | None = None, downstream_partial_pressure_Pa: float | None = None,
                        upstream_concentration_mol_m3: float | None = None, downstream_concentration_mol_m3: float | None = None,
                        area_m2: float | None = None, inner_radius_m: float | None = None, length_m: float | None = None,
                        upstream_side: str | None = None, upstream_film_coefficient: float | None = None,
                        downstream_film_coefficient: float | None = None, transient: bool = False,
                        initial_partial_pressure_Pa: float | None = None, initial_concentration_mol_m3: float | None = None,
                        times_s: list[float] | None = None, cells_per_layer: int = 120, rtol: float = RTOL) -> dict:
    """Steady and transient permeation by solution-diffusion through one layer or a laminate in series.

    Driving force = partial pressures (layers with diffusivity_m2_s + solubility_mol_m3_Pa, or a permeability) or fluid
    concentrations (layers with diffusivity_m2_s + partition_coefficient, or permeability_m2_s). layers are listed from the
    upstream face to the downstream face. transient=True adds the time lag and the downstream cumulative curve."""
    if not layers:
        raise ValueError("falta layers: lista de capas de la cara de alimentación (aguas arriba) a la de permeado, cada una "
                         "con thickness_m y sus propiedades (difusividad y solubilidad, o permeabilidad)")
    if geometry is None or str(geometry).lower() not in ("flat", "cylindrical", "spherical"):
        raise ValueError("falta geometry: 'flat' (película plana), 'cylindrical' (pared de tubo) o 'spherical' (pared de esfera)")
    geo = str(geometry).lower()
    by_p = upstream_partial_pressure_Pa is not None or downstream_partial_pressure_Pa is not None
    by_c = upstream_concentration_mol_m3 is not None or downstream_concentration_mol_m3 is not None
    if by_p == by_c:
        raise ValueError("dé la fuerza impulsora de UNA forma: presiones parciales (upstream_partial_pressure_Pa y "
                         "downstream_partial_pressure_Pa) o concentraciones en los fluidos (upstream_concentration_mol_m3 y "
                         "downstream_concentration_mol_m3)")
    mode = "pressure" if by_p else "concentration"
    unit = "Pa" if by_p else "mol_m3"
    if by_p:
        u_up = _need(upstream_partial_pressure_Pa, "upstream_partial_pressure_Pa", "presión parcial del permeante aguas arriba")
        u_down = _need(downstream_partial_pressure_Pa, "downstream_partial_pressure_Pa",
                       "presión parcial del permeante aguas abajo (0 si se barre o se hace vacío)")
    else:
        u_up = _need(upstream_concentration_mol_m3, "upstream_concentration_mol_m3", "concentración del soluto en el fluido aguas arriba")
        u_down = _need(downstream_concentration_mol_m3, "downstream_concentration_mol_m3",
                       "concentración del soluto en el fluido aguas abajo")
    if u_up < 0 or u_down < 0:
        raise ValueError("las presiones parciales / concentraciones no pueden ser negativas")
    if not 10 <= int(cells_per_layer) <= MAX_CELLS:
        raise ValueError(f"cells_per_layer debe estar entre 10 y {MAX_CELLS}")
    props = [_layer_properties(la, i, mode) for i, la in enumerate(layers)]
    for name, k in (("upstream_film_coefficient", upstream_film_coefficient), ("downstream_film_coefficient", downstream_film_coefficient)):
        if k is not None:
            _positive(k, name, "coeficiente de película externo (mol/(m²·s·Pa) con presiones, m/s con concentraciones)")
    m = {"flat": 0, "cylindrical": 1, "spherical": 2}[geo]
    up_inner = True
    r_in = None
    if m > 0:
        r_in = _positive(inner_radius_m, "inner_radius_m", "radio interior de la pared")
        if upstream_side not in ("inner", "outer"):
            raise ValueError("falta upstream_side ('inner' u 'outer'): de qué lado de la pared curva está la alimentación")
        up_inner = upstream_side == "inner"
    wall = _Wall(geo, props, r_in, up_inner, upstream_film_coefficient, downstream_film_coefficient)
    shape = SHAPE_FACTOR[m]
    du = u_up - u_down
    n_norm = du / wall.r_total                        # per unit shape factor, positive from upstream to downstream
    u_in, u_out = (u_up, u_down) if up_inner else (u_down, u_up)
    out: dict = {"geometry": geo, "driving_force": mode, "driving_force_difference": du}
    r_up, r_down = (wall.rb[0], wall.rb[-1]) if up_inner else (wall.rb[-1], wall.rb[0])
    if m == 0:
        out["flux_mol_m2_s"] = n_norm
        if area_m2 is not None:
            out["rate_mol_s"] = n_norm * _positive(area_m2, "area_m2", "área de membrana")
        size, ref = (float(area_m2) if area_m2 is not None else None), "por m² de membrana"
    elif m == 1:
        out["rate_per_length_mol_s_m"] = shape * n_norm
        out["flux_at_upstream_surface_mol_m2_s"] = n_norm / r_up
        out["flux_at_downstream_surface_mol_m2_s"] = n_norm / r_down
        if length_m is not None:
            out["rate_mol_s"] = shape * n_norm * _positive(length_m, "length_m", "longitud del tubo")
        size, ref = (shape * float(length_m) if length_m is not None else None), "por metro de tubo (÷2π aplicado)"
    else:
        out["rate_mol_s"] = shape * n_norm
        out["flux_at_upstream_surface_mol_m2_s"] = n_norm / r_up ** 2
        out["flux_at_downstream_surface_mol_m2_s"] = n_norm / r_down ** 2
        size, ref = shape, "esfera completa"
    # layer table and interface values, listed upstream → downstream
    order = list(range(len(wall.layers))) if up_inner else list(range(len(wall.layers) - 1, -1, -1))
    rows, u_face = [], u_up - n_norm * (wall.r_film_in if up_inner else wall.r_film_out)
    res_unit = {0: "s·m²·[fuerza impulsora]/mol", 1: "s·m·[fuerza impulsora]/mol", 2: "s·[fuerza impulsora]/mol"}[m]
    for i in order:
        la, r_l = wall.layers[i], float(wall.r_layers[i]) / shape if m else float(wall.r_layers[i])
        u_exit = u_face - n_norm * float(wall.r_layers[i])
        row = {"name": la["name"], "thickness_m": la["thickness"], "resistance": r_l,
               "fraction_of_total_resistance": float(wall.r_layers[i]) / wall.r_total,
               f"upstream_face_{'partial_pressure_Pa' if by_p else 'fluid_equivalent_concentration_mol_m3'}": u_face,
               f"downstream_face_{'partial_pressure_Pa' if by_p else 'fluid_equivalent_concentration_mol_m3'}": u_exit,
               "diffusivity_m2_s": la["D"]}
        if by_p:
            row.update(permeability_mol_m_m2_s_Pa=la["P"], permeability_barrer=la["P"] / BARRER_SI,
                       solubility_mol_m3_Pa=la["S"])
        else:
            row.update(permeability_m2_s=la["P"], partition_coefficient=la["S"])
        if la["S"] is not None:
            row["upstream_face_concentration_in_layer_mol_m3"] = la["S"] * u_face
            row["downstream_face_concentration_in_layer_mol_m3"] = la["S"] * u_exit
            row["profile_shape"] = ("lineal", "logarítmico en r", "proporcional a 1/r")[m]
        if la["D"] is not None:
            row["own_time_lag_L2_over_6D_s"] = la["thickness"] ** 2 / (6.0 * la["D"])
        rows.append(row)
        u_face = u_exit
    out["layers"] = rows
    films = {}
    for label, rf in (("upstream_film", wall.r_film_in if up_inner else wall.r_film_out),
                      ("downstream_film", wall.r_film_out if up_inner else wall.r_film_in)):
        if rf:
            films[label] = {"resistance": rf / shape if m else rf, "fraction_of_total_resistance": rf / wall.r_total}
    if films:
        out["films"] = films
    out["total_resistance"] = wall.r_total / shape if m else wall.r_total
    out["resistance_units"] = res_unit
    out["controlling_resistance"] = max(rows, key=lambda r: r["fraction_of_total_resistance"])["name"]
    assumptions = ["Modelo de solución-difusión: en cada capa c = S·p (o c = K·c_fluido), sorción lineal (Henry) y D constante; "
                   "permeabilidad = D·S; equilibrio en cada interfase y capas en serie perfectamente adheridas.",
                   "Unidades de permeabilidad con presión: mol·m/(m²·s·Pa); 1 Barrer = 3.346e-16 mol·m/(m²·s·Pa) y el m³(STP) es "
                   "a 0 °C y 1 atm (22.414 m³/kmol)."]
    if not films:
        assumptions.append("Sin resistencia de película externa: las caras de la membrana están en equilibrio con los fluidos.")
    warnings: list[str] = []
    reasons: list[str] = []
    n_layers = len(props)
    evidence: dict = {}
    # steady check: finite-volume matrix vs resistance sum (needs no storage)
    num = _steady_numeric(wall, int(cells_per_layer), u_in, u_out)
    num_up = num if up_inner else -num
    evidence["steady_finite_volume_vs_resistance_sum_rel_diff"] = abs(num_up - n_norm) / max(abs(n_norm), 1e-300) if du else abs(num_up)
    if evidence["steady_finite_volume_vs_resistance_sum_rel_diff"] > 1e-8:
        reasons.append("el flujo estacionario del sistema de volúmenes finitos no coincide con la suma de resistencias")
    evidence["interface_closure_rel_error"] = abs(u_face - (wall.r_film_out if up_inner else wall.r_film_in) * n_norm - u_down) \
        / max(abs(u_up), abs(u_down), 1e-300)
    if evidence["interface_closure_rel_error"] > 1e-9:
        reasons.append("el perfil de interfases no cierra en la cara aguas abajo")
    if transient:
        if any(la["D"] is None or la["S"] is None for la in props):
            raise ValueError("el transitorio (time lag) necesita la difusividad de CADA capa además de su permeabilidad o "
                             "solubilidad: la permeabilidad sola solo fija el estacionario")
        init = initial_partial_pressure_Pa if by_p else initial_concentration_mol_m3
        u0 = _need(init, "initial_partial_pressure_Pa" if by_p else "initial_concentration_mol_m3",
                   "estado inicial de la membrana, expresado como la presión parcial/concentración de fluido con la que está en "
                   "equilibrio en t = 0 (0 si está desgasificada, que es el ensayo clásico de time lag)")
        if du == 0.0:
            raise ValueError("no hay diferencia de presión/concentración entre las dos caras: no hay permeación ni time lag")
        theta_f = wall.frisch_time_lag(u_up, u_down, u0)
        slow = sum(la["thickness"] for la in props) ** 2 / min(la["D"] for la in props)
        req = _times(times_s)
        if req.size > MAX_TIMES:
            raise ValueError(f"demasiados instantes en times_s ({req.size}): máximo {MAX_TIMES}")
        t_end = max(15.0 * abs(theta_f), 2.0 * slow / math.pi ** 2 * 6.0, float(req[-1]) if req.size else 0.0)
        if not req.size:
            req = np.linspace(0.0, 6.0 * max(abs(theta_f), slow / 6.0), 31)[1:]
        scale = max(abs(u_up - u0), abs(u_down - u0), abs(du))
        n_ss_inner = (u_in - u_out) / wall.r_total      # inner → outer
        runs, theta, ok = {}, {}, True
        for label, npl, tol in (("base", int(cells_per_layer), rtol), ("fine", 2 * int(cells_per_layer), rtol),
                                ("tight", int(cells_per_layer), rtol / 100.0)):
            for attempt in range(4 if label == "base" else 1):      # the horizon is extended only on the base run
                tr = _membrane_transient(wall, npl, u_in, u_out, u0, req, t_end, tol, scale)
                if tr["failed"]:
                    break
                f_down = tr["f_out"][-1] if up_inner else -tr["f_in"][-1]
                if abs(f_down / n_norm - 1.0) < 1e-7 or attempt == 3 or label != "base":
                    break
                t_end *= 3.0
            if tr["failed"]:
                reasons.append(tr["failed"])
                ok = False
                break
            runs[label] = tr
            q_down = tr["q_out"] if up_inner else -tr["q_in"]
            theta[label] = float(tr["t"][-1] - q_down[-1] / n_norm)
        if ok:
            tr = runs["base"]
            idx = np.searchsorted(tr["t"], req)
            q_down = (tr["q_out"] if up_inner else -tr["q_in"])
            q_up = (tr["q_in"] if up_inner else -tr["q_out"])
            f_down = (tr["f_out"] if up_inner else -tr["f_in"])
            f_up = (tr["f_in"] if up_inner else -tr["f_out"])
            f_end = f_down[-1] / n_norm
            evidence["downstream_flux_over_steady_at_end"] = float(f_end)
            if abs(f_end - 1.0) > 1e-6:
                reasons.append("el flujo aguas abajo no alcanzó el estacionario en el horizonte de integración: el time lag "
                               "numérico no es fiable")
            bal = np.abs((q_up - q_down) - tr["stored"]) / max(float(np.max(np.abs(tr["stored"]))), 1e-300)
            evidence["mass_balance_max_rel_error"] = float(np.max(bal))
            evidence["time_lag_numeric_vs_frisch_integral_rel_diff"] = abs(theta["base"] - theta_f) / abs(theta_f) if theta_f else None
            tscale = max(abs(theta["base"]), 1e-300)
            evidence["time_lag_mesh_refinement_rel_diff"] = abs(theta["fine"] - theta["base"]) / tscale
            evidence["time_lag_time_tolerance_rel_diff"] = abs(theta["tight"] - theta["base"]) / tscale
            qf = (runs["fine"]["q_out"] if up_inner else -runs["fine"]["q_in"])
            qscale = abs(n_norm) * tscale
            evidence["cumulative_curve_mesh_refinement_max_diff"] = float(np.max(np.abs(qf - q_down[: qf.size])) / qscale) \
                if qf.size == q_down.size else None
            for key, tol_k in (("mass_balance_max_rel_error", TOL_MASS_BALANCE),
                               ("time_lag_numeric_vs_frisch_integral_rel_diff", TOL_ANALYTICAL),
                               ("time_lag_mesh_refinement_rel_diff", TOL_MESH), ("time_lag_time_tolerance_rel_diff", 10 * TOL_TIME),
                               ("cumulative_curve_mesh_refinement_max_diff", TOL_MESH)):
                v = evidence.get(key)
                if v is not None and not v <= tol_k:
                    reasons.append(f"{key} = {v:.2e} > {tol_k:.0e}")
            mult = 1.0 if m == 0 else shape
            tl: dict = {"time_lag_s": theta["base"], "time_lag_frisch_integral_s": theta_f,
                        "steady_asymptote": "cantidad acumulada aguas abajo → flujo_estacionario·(t − time_lag)",
                        "initial_equilibrium_value": u0, "amount_reference": ref}
            if n_layers == 1 and m == 0 and not films and u0 == 0.0 and u_down == 0.0:
                tl["time_lag_L2_over_6D_s"] = props[0]["thickness"] ** 2 / (6.0 * props[0]["D"])
                evidence["time_lag_numeric_vs_L2_over_6D_rel_diff"] = abs(theta["base"] - tl["time_lag_L2_over_6D_s"]) \
                    / tl["time_lag_L2_over_6D_s"]
                if evidence["time_lag_numeric_vs_L2_over_6D_rel_diff"] > TOL_ANALYTICAL:
                    reasons.append("el time lag numérico no reproduce L²/6D")
            curve = {"time_s": req.tolist(),
                     "downstream_cumulative": (mult * q_down[idx]).tolist(), "upstream_cumulative": (mult * q_up[idx]).tolist(),
                     "downstream_rate": (mult * f_down[idx]).tolist(), "upstream_rate": (mult * f_up[idx]).tolist(),
                     "steady_asymptote_cumulative": (mult * n_norm * (req - theta["base"])).tolist(),
                     "units": {0: "acumulado mol/m², caudal mol/(m²·s)", 1: "acumulado mol/m de tubo, caudal mol/(s·m)",
                               2: "acumulado mol, caudal mol/s"}[m]}
            if size is not None and m < 2:
                curve["downstream_cumulative_mol"] = (size * q_down[idx]).tolist()
            out["transient"] = {**tl, "curve": curve}
            if u0 != 0.0 or u_down != 0.0:
                warnings.append("Membrana no desgasificada o presión aguas abajo no nula: el time lag ya no es L²/6D; se da el "
                                "numérico y el de la integral de Frisch para ESTAS condiciones.")
            if theta["base"] < 0:
                warnings.append("Time lag negativo: la membrana partía más cargada que el perfil estacionario (se descarga).")
    else:
        if times_s is not None or initial_partial_pressure_Pa is not None or initial_concentration_mol_m3 is not None:
            warnings.append("Se dieron tiempos o estado inicial pero transient = False: solo se calculó el estacionario.")
    if any(r["fraction_of_total_resistance"] > 0.95 for r in rows) and n_layers > 1:
        warnings.append(f"La capa «{out['controlling_resistance']}» concentra más del 95 % de la resistencia: las demás apenas "
                        "cuentan como barrera.")
    evidence.update(verified=not reasons, reasons=reasons)
    out.update(warnings=warnings, assumptions=assumptions,
               not_included=["Permeación de mezclas con acoplamiento, selectividad y balance de módulo (separación con membranas).",
                             "Sorción no lineal (modo dual, Flory-Huggins), plastificación o D dependiente de la concentración.",
                             "Defectos, poros (flujo de Knudsen o viscoso), delaminación y efecto de la temperatura (Arrhenius).",
                             "Acumulación aguas abajo que cambie la presión de permeado (se mantiene constante)."],
               method=("Estacionario: suma de resistencias en serie L/(P·A), ln(r2/r1)/(2π·L·P) o (1/r1 − 1/r2)/(4π·P), "
                       "contrastada con el sistema tridiagonal de volúmenes finitos (scipy solve_banded). Transitorio: volúmenes "
                       f"finitos con {int(cells_per_layer)} celdas por capa concentradas en las interfases, método de líneas con "
                       "scipy solve_ivp LSODA; time lag del asíntota numérica y de la integral de Frisch (scipy quad)."),
               evidence=evidence)
    return _clean(out)


# ---------------------------------------------------------------------------------------------------------------------
# D from uptake data
def _uptake_model(m: int, a: float, d: float, h_over_d: float | None, t: np.ndarray) -> np.ndarray:
    """M_t/M_∞ from the series; h_over_d = k_c/K so that Bi = (k_c/K)·a/D."""
    biot = math.inf if h_over_d is None else h_over_d * a / d
    fo = d * np.asarray(t, dtype=float) / a ** 2
    beta = eigenvalues(m, biot, min(_terms_needed(float(fo.min())), MAX_SERIES_TERMS))
    return 1.0 - np.exp(-np.outer(fo, beta ** 2)) @ _uptake_weights(m, biot, beta)


def fit_diffusivity_from_uptake(geometry: str | None = None, times_s: list[float] | None = None, uptake: list[float] | None = None,
                                uptake_kind: str | None = None, thickness_m: float | None = None,
                                exposed_faces: int | None = None, radius_m: float | None = None,
                                equilibrium_uptake: float | None = None, fit_equilibrium_uptake: bool = False,
                                mass_transfer_coefficient_m_s: float | None = None, partition_coefficient: float | None = None,
                                confidence_level: float = 0.95) -> dict:
    """Least-squares fit of a constant D (and optionally M_∞) to uptake-versus-time data of a slab, cylinder or sphere."""
    if geometry is None or str(geometry).lower() not in GEOMETRY_EXPONENT:
        raise ValueError("falta geometry: 'slab', 'cylinder' o 'sphere' (la forma de la muestra del ensayo de sorción)")
    geo = str(geometry).lower()
    m = GEOMETRY_EXPONENT[geo]
    if times_s is None or uptake is None:
        raise ValueError("faltan los datos del ensayo: times_s (s) y uptake (captación medida en cada tiempo)")
    t = np.asarray(list(times_s), dtype=float)
    y = np.asarray(list(uptake), dtype=float)
    if t.size != y.size:
        raise ValueError("times_s y uptake deben tener la misma longitud")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))) or np.any(t < 0):
        raise ValueError("times_s y uptake deben ser finitos y los tiempos ≥ 0")
    keep = t > 0
    t, y = t[keep], y[keep]
    if uptake_kind not in ("fractional", "amount"):
        raise ValueError("falta uptake_kind: 'fractional' si uptake ya es M_t/M_∞ (0–1) o 'amount' si es la captación en sus "
                         "unidades (g, %, mol), en cuyo caso dé equilibrium_uptake o fit_equilibrium_uptake = true")
    n_par = 2 if fit_equilibrium_uptake else 1
    if t.size < n_par + 2:
        raise ValueError(f"se necesitan al menos {n_par + 2} puntos con t > 0 para ajustar {n_par} parámetro(s) con residuo")
    if geo == "slab":
        thick = _positive(thickness_m, "thickness_m", "espesor total de la película o placa ensayada")
        if exposed_faces not in (1, 2):
            raise ValueError("falta exposed_faces (1 o 2): caras de la muestra expuestas durante el ensayo")
        a = thick / int(exposed_faces)
    else:
        a = _positive(radius_m, "radius_m", "radio de la muestra")
    h_over_d = None
    if mass_transfer_coefficient_m_s is not None or partition_coefficient is not None:
        kc = _positive(mass_transfer_coefficient_m_s, "mass_transfer_coefficient_m_s", "coeficiente de película externo")
        kp = _positive(partition_coefficient, "partition_coefficient", "K = c_sólido/c_fluido")
        h_over_d = kc / kp
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level debe estar entre 0.5 y 1")
    assumptions = ["D constante, sorción fickiana, concentración inicial uniforme y entorno constante durante el ensayo."]
    if uptake_kind == "fractional":
        if fit_equilibrium_uptake or equilibrium_uptake is not None:
            raise ValueError("con uptake_kind = 'fractional' los datos ya están divididos por M_∞: no dé equilibrium_uptake ni "
                             "fit_equilibrium_uptake")
        m_inf0, fit_minf = 1.0, False
    else:
        fit_minf = bool(fit_equilibrium_uptake)
        if fit_minf:
            if equilibrium_uptake is not None:
                raise ValueError("dé equilibrium_uptake O fit_equilibrium_uptake = true, no ambos")
            m_inf0 = float(y[np.argmax(np.abs(y))])
            assumptions.append("M_∞ ajustado junto con D (los datos no lo fijan si no se acercan al equilibrio).")
        else:
            m_inf0 = _need(equilibrium_uptake, "equilibrium_uptake", "captación de equilibrio M_∞ en las mismas unidades que uptake "
                           "(o use fit_equilibrium_uptake = true para ajustarla)")
        if m_inf0 == 0.0:
            raise ValueError("la captación de equilibrio no puede ser 0")
    assumptions.append("Superficie a concentración fija (sin resistencia externa)." if h_over_d is None else
                       "Película externa en serie con k_c y K dados (Bi = k_c·a/(K·D) cambia con D durante el ajuste).")

    def model(p):
        return (p[1] if fit_minf else m_inf0) * _uptake_model(m, a, math.exp(p[0]), h_over_d, t)

    def resid(p):
        return model(p) - y

    # starting value: half-time of the data against the half-time Fourier number of the model (fixed surface)
    fo_half = optimize.brentq(lambda fo: series_fractional_uptake(m, math.inf, fo) - 0.5, 1e-6, 2.0)
    frac = y / m_inf0
    order = np.argsort(t)
    th = np.interp(0.5, frac[order], t[order]) if frac[order][0] < 0.5 < frac[order][-1] and np.all(np.diff(frac[order]) > 0) \
        else float(np.median(t))
    d_start = fo_half * a * a / th
    fits = []
    for d0 in (d_start, d_start * 30.0, d_start / 30.0):
        p0 = [math.log(d0)] + ([m_inf0] if fit_minf else [])
        try:
            fits.append(optimize.least_squares(resid, p0, x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400))
        except Exception as exc:  # noqa: BLE001 — a failed start is reported, not hidden
            fits.append(exc)
    good = [f for f in fits if not isinstance(f, Exception) and f.success]
    reasons: list[str] = []
    if not good:
        raise ValueError("el ajuste por mínimos cuadrados no convergió con ninguno de los tres arranques: revise los datos "
                         "(¿tiempos en segundos?, ¿captación creciente?)")
    best = min(good, key=lambda f: f.cost)
    d_fit = math.exp(best.x[0])
    minf_fit = float(best.x[1]) if fit_minf else m_inf0
    others = [abs(math.exp(f.x[0]) / d_fit - 1.0) for f in good]
    r = best.fun
    dof = t.size - n_par
    s2 = float(r @ r) / dof
    jac = best.jac
    try:
        cov = np.linalg.inv(jac.T @ jac) * s2
    except np.linalg.LinAlgError:
        cov = np.full((n_par, n_par), np.nan)
        reasons.append("matriz de información singular: los datos no determinan los parámetros")
    tq = float(stats.t.ppf(0.5 + confidence_level / 2.0, dof))
    se_ln = math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) and cov[0, 0] >= 0 else math.nan
    ci = [d_fit * math.exp(-tq * se_ln), d_fit * math.exp(tq * se_ln)] if math.isfinite(se_ln) else [None, None]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    fitted = model(best.x)
    rs = r[order]
    dw = float(np.sum(np.diff(rs) ** 2) / np.sum(rs ** 2)) if np.sum(rs ** 2) > 0 else None
    lag1 = float(np.sum(rs[1:] * rs[:-1]) / np.sum(rs ** 2)) if np.sum(rs ** 2) > 0 else None
    signs = np.sign(rs[rs != 0])
    runs = int(1 + np.sum(signs[1:] != signs[:-1])) if signs.size else 0
    n_pos, n_neg = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    runs_expected = 1.0 + 2.0 * n_pos * n_neg / max(n_pos + n_neg, 1)
    rmse = math.sqrt(float(r @ r) / t.size)
    out: dict = {"geometry": geo, "diffusion_path_m": a, "diffusivity_m2_s": d_fit,
                 "diffusivity_confidence_interval_m2_s": ci, "confidence_level": confidence_level,
                 "relative_standard_error_of_D": se_ln if math.isfinite(se_ln) else None,
                 "equilibrium_uptake": minf_fit, "equilibrium_uptake_fitted": fit_minf,
                 "fit": {"points": int(t.size), "parameters": n_par, "rmse": rmse, "rmse_over_equilibrium_uptake": rmse / abs(minf_fit),
                         "r_squared": 1.0 - float(r @ r) / ss_tot if ss_tot > 0 else None,
                         "max_abs_residual": float(np.max(np.abs(r))), "residuals": r.tolist(), "fitted_uptake": fitted.tolist(),
                         "time_s": t.tolist(), "durbin_watson": dw, "lag1_autocorrelation": lag1, "sign_runs": runs,
                         "sign_runs_expected_if_random": runs_expected,
                         "fourier_number_range": [d_fit * float(t.min()) / a ** 2, d_fit * float(t.max()) / a ** 2],
                         "fractional_uptake_range_of_data": [float(np.min(y / minf_fit)), float(np.max(y / minf_fit))]}}
    if fit_minf and np.isfinite(cov[1, 1]):
        se_m = math.sqrt(max(cov[1, 1], 0.0))
        out["equilibrium_uptake_confidence_interval"] = [minf_fit - tq * se_m, minf_fit + tq * se_m]
        out["fit"]["correlation_lnD_Minf"] = float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])) if cov[0, 0] > 0 and cov[1, 1] > 0 else None
    if h_over_d is not None:
        out["biot_number_mass_at_fitted_D"] = h_over_d * a / d_fit
    # cross-checks that use different parts of the data
    cross: dict = {}
    fr = (y / minf_fit)[order]
    ts = t[order]
    if h_over_d is None:
        if fr[0] < 0.5 < fr[-1] and np.all(np.diff(fr) > 0):
            t_half = float(np.interp(0.5, fr, ts))
            cross["half_time_method"] = {"half_time_s": t_half, "fourier_number_at_half_uptake": fo_half,
                                         "diffusivity_m2_s": fo_half * a * a / t_half}
        early = (fr > 0) & (fr <= 0.5)
        if int(np.sum(early)) >= 2:
            slope = float(np.sum(np.sqrt(ts[early]) * fr[early]) / np.sum(ts[early]))      # line through the origin in √t
            # M_t/M_∞ ≈ (m+1)·2·sqrt(Fo/π) at short times (surface/volume = (m+1)/a); curvature neglected
            cross["initial_slope_method"] = {"slope_per_sqrt_s": slope, "points_used": int(np.sum(early)),
                                             "diffusivity_m2_s": math.pi * (slope * a / (2.0 * (m + 1))) ** 2,
                                             "note": "pendiente inicial de M_t/M_∞ frente a √t (exacta en placa; en cilindro y "
                                                     "esfera desprecia la curvatura y subestima D)"}
    out["cross_checks"] = cross
    warnings: list[str] = []
    if out["fit"]["fractional_uptake_range_of_data"][1] < 0.6 and fit_minf:
        warnings.append("Los datos no pasan del 60 % de la captación de equilibrio ajustada: D y M_∞ quedan muy correlacionados; "
                        "mida hasta cerca del equilibrio o fije M_∞.")
    if lag1 is not None and t.size >= 8 and lag1 > 0.5 and rmse > 1e-9 * abs(minf_fit):
        warnings.append("Residuos autocorrelacionados (tramos largos del mismo signo): el modelo fickiano de D constante no "
                        "describe bien la curva; puede haber D(c), resistencia externa, hinchamiento o difusión no fickiana.")
    if math.isfinite(se_ln) and tq * se_ln > 0.5:
        warnings.append("Intervalo de confianza de D muy ancho (más de ±50 %): los datos apenas determinan D.")
    if out["fit"]["fourier_number_range"][0] > 0.5:
        warnings.append("Todos los datos están a Fo > 0.5 (casi en equilibrio): poca información sobre D.")
    for k, v in cross.items():
        if abs(v["diffusivity_m2_s"] / d_fit - 1.0) > 0.25 and (k == "half_time_method" or m == 0):
            warnings.append(f"El método {k} da D = {v['diffusivity_m2_s']:.3g} m²/s, más de un 25 % distinto del ajuste global: "
                            "la forma de la curva no es la fickiana ideal.")
    # evidence: optimizer state, multi-start agreement, and the series model against the numerical engine at the fitted D
    diff = _Diffusivity(d_fit, None)
    bc = {"kind": "fixed", "cs": 1.0} if h_over_d is None else {"kind": "convective", "kc": h_over_d, "K": 1.0, "cf": 1.0}
    tu = np.unique(t)
    sol = _solve_1d(m, a, diff, 0.0, bc, 1.0, tu, None, N_CELLS, RTOL)
    num = np.array([s["uptake"] / sol["m_inf"] for s in sol["states"]]) if not sol["reasons"] else np.full(tu.size, np.nan)
    ser = _uptake_model(m, a, d_fit, h_over_d, tu)
    model_gap = float(np.max(np.abs(num - ser))) if np.all(np.isfinite(num)) else None
    grad = float(np.max(np.abs(best.grad))) if best.grad is not None else None
    evidence = {"optimizer_status": int(best.status), "optimizer_message": str(best.message), "cost_gradient_inf_norm": grad,
                "starts_converged": len(good), "starts_total": len(fits),
                "multi_start_max_rel_diff_in_D": float(max(others)) if others else None,
                "series_model_vs_finite_volume_engine_max_abs_diff": model_gap}
    if len(good) < len(fits):
        reasons.append("no todos los arranques del optimizador convergieron")
    if others and max(others) > 1e-4:
        reasons.append("distintos arranques llevan a distinta D: el mínimo no es único o está mal condicionado")
    if model_gap is None or model_gap > TOL_ANALYTICAL:
        reasons.append("la serie usada como modelo no coincide con el motor numérico a la D ajustada")
    if not math.isfinite(se_ln):
        reasons.append("no se pudo calcular el intervalo de confianza de D")
    evidence.update(verified=not reasons, reasons=reasons,
                    note="verified certifica el ajuste numérico (mínimo único, modelo = motor); la calidad de los datos se "
                         "juzga con el intervalo de confianza y los diagnósticos de residuos")
    out.update(warnings=warnings, assumptions=assumptions,
               not_included=["D dependiente de la concentración (un solo valor medio efectivo en el intervalo del ensayo).",
                             "Sorción en dos etapas, relajación del polímero, hinchamiento y efectos de borde de la muestra.",
                             "Error de medida en el tiempo o en el tamaño de la muestra (D ∝ a²: un 5 % en a es un 10 % en D)."],
               method=("Mínimos cuadrados no lineales (scipy least_squares, tres arranques) sobre ln D con la serie analítica de "
                       "Crank como modelo; intervalo de confianza linealizado con t de Student sobre ln D; contraste con el "
                       "método del tiempo medio y de la pendiente inicial, y del modelo con el motor de volúmenes finitos."),
               evidence=evidence)
    return _clean(out)
