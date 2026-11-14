"""Reference problems with a known answer for the Bayesian tools (bloque M16). None of this is used by the tools:
it is what they are checked against.

Branin and Hartmann-6: Dixon & Szegő, «Towards Global Optimization 2», North-Holland 1978; forms, constants and
optima as tabulated by Surjanovic & Bingham, «Virtual Library of Simulation Experiments» (Simon Fraser University,
https://www.sfu.ca/~ssurjano/optimization.html) and Picheny, Wagner & Ginsbourger, Struct. Multidisc. Optim. 48
(2013) 607-626.
Constrained toy problem: Gramacy, Gray, Le Digabel, Lee, Ranjan, Wells & Wild, «Modeling an augmented Lagrangian for
blackbox constrained optimization», Technometrics 58(1) (2016) 1-11, section 2.
SnAr emulator: the kinetic model of the `SnarBenchmark` of Summit (Felton, Rittig & Lapkin, Chemistry-Methods 1
(2021) 116-122; MIT licence; summit/benchmarks/snar.py at commit 1de682d), which attributes the kinetics to Hone,
Holmes, Akien, Bourne & Muller, React. Chem. Eng. 2 (2017) 103-108, DOI 10.1039/C6RE00109B. It is a published MODEL
of a reaction, not laboratory data.
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------------------------------------------------ Branin
BRANIN_VARS = [{"name": "x1", "param": "", "lower": -5.0, "upper": 10.0}, {"name": "x2", "param": "", "lower": 0.0, "upper": 15.0}]
BRANIN_FMIN = 0.397887
BRANIN_MINIMA = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]


def branin(x) -> float:
    x1, x2 = x
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


# --------------------------------------------------------------------------------------------------------- Hartmann-6
HARTMANN6_VARS = [{"name": f"x{i + 1}", "param": "", "lower": 0.0, "upper": 1.0} for i in range(6)]
HARTMANN6_FMIN = -3.32237
HARTMANN6_XMIN = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])


def hartmann6(x) -> float:
    x = np.asarray(x, float)
    return float(-np.sum(_H_ALPHA * np.exp(-np.sum(_H_A * (x[None, :] - _H_P) ** 2, axis=1))))


# ------------------------------------------------------------------------------------- Gramacy et al. 2016 toy problem
GRAMACY_VARS = [{"name": "x1", "param": "", "lower": 0.0, "upper": 1.0}, {"name": "x2", "param": "", "lower": 0.0, "upper": 1.0}]
GRAMACY_LIMITS = {"c1": {"max": 0.0}, "c2": {"max": 0.0}}
GRAMACY_FMIN = 0.5998          # at x ≈ (0.1954, 0.4044), c1 active (Gramacy et al. 2016)
GRAMACY_XMIN = (0.1954, 0.4044)


def gramacy(x) -> dict:
    """min x1 + x2  s.t.  c1 = 3/2 − x1 − 2·x2 − ½·sin(2π(x1² − 2·x2)) ≤ 0,  c2 = x1² + x2² − 3/2 ≤ 0,  x ∈ [0, 1]²."""
    x1, x2 = x
    return {"solved_ok": True, "objective": x1 + x2,
            "constraints": {"c1": 1.5 - x1 - 2 * x2 - 0.5 * math.sin(2 * math.pi * (x1 ** 2 - 2 * x2)), "c2": x1 ** 2 + x2 ** 2 - 1.5}}


# ------------------------------------------------------------------------------------- SnAr kinetic emulator (Summit)
SNAR_VARS = [{"name": "tau", "lower": 0.5, "upper": 2.0},              # residence time, min
             {"name": "equiv_pldn", "lower": 1.0, "upper": 5.0},       # equivalents of pyrrolidine
             {"name": "conc_dfnb", "lower": 0.1, "upper": 0.5},        # 2,4-difluoronitrobenzene at the inlet, M
             {"name": "temperature", "lower": 30.0, "upper": 120.0}]   # °C
SNAR_M = (159.09, 71.12, 210.21, 210.21, 261.33)                       # g/mol: 1 DFNB, 2 pyrrolidine, 3 ortho, 4 para, 5 bis
_SNAR_K = ((57.9, 33.3), (2.70, 35.3), (0.865, 38.9), (1.63, 44.8))    # (k_ref at 90 °C, 1e-2 M⁻¹ s⁻¹; Ea, kJ/mol)


def snar(tau: float, equiv_pldn: float, conc_dfnb: float, temperature: float, noise_pct: float = 0.0,
         rng: np.random.Generator | None = None) -> dict:
    """Outlet of the plug-flow SnAr reactor of Summit's SnarBenchmark: space-time yield of the ortho product
    (kg m⁻³ h⁻¹), E-factor and yield of the ortho product on DFNB. Same equations and constants as Summit, including
    its 273.71 K offset; integrated here with LSODA at rtol 1e-9 (Summit uses solve_ivp's default RK45, rtol 1e-3, so
    the numbers agree to about three digits). noise_pct = relative normal noise on every outlet concentration, as in
    Summit's `noise_level`."""
    from scipy.integrate import solve_ivp
    R = 8.314 / 1000
    T_ref, T = 90 + 273.71, temperature + 273.71
    ka, kb, kc, kd = (0.6 * k * math.exp(-Ea / R * (1 / T - 1 / T_ref)) for k, Ea in _SNAR_K)   # M⁻¹ min⁻¹

    def rhs(_, C):
        r1, r2, r3, r4 = ka * C[0] * C[1], kb * C[0] * C[1], kc * C[1] * C[2], kd * C[1] * C[3]
        return [-r1 - r2, -r1 - r2 - r3 - r4, r1 - r3, r2 - r4, r3 + r4]

    C0 = [conc_dfnb, equiv_pldn * conc_dfnb, 0.0, 0.0, 0.0]
    C = solve_ivp(rhs, [0, tau], C0, method="LSODA", rtol=1e-9, atol=1e-12).y[:, -1]
    if noise_pct:
        C = np.maximum(C * (1 + (rng or np.random.default_rng()).normal(scale=noise_pct / 100.0, size=5)), 0.0)
    V = 5.0
    q = V / tau
    sty = 6e4 / 1000 * SNAR_M[2] * C[2] * q / V
    waste = 1e-3 * sum(SNAR_M[i] * C[i] * q for i in range(5) if i != 2)
    e_factor = (q * 0.789 + waste) / (1e-3 * SNAR_M[2] * C[2] * q) if C[2] > 1e-12 else 1e3
    return {"sty": float(sty), "e_factor": float(min(e_factor, 1e3)), "yield_ortho": float(C[2] / conc_dfnb)}


# ----------------------------------------------------------------- two-well toy (synthetic, for the restart check)
TWO_WELLS_VARS = [{"name": "x", "param": "1", "lower": 0.0, "upper": 1.0}, {"name": "x", "param": "2", "lower": 0.0, "upper": 1.0}]
TWO_WELLS_FMIN, TWO_WELLS_LOCAL = -1.5, -1.0


def two_wells(x) -> float:
    """Made-up function with two separate wells of width 0.1: a local minimum −1.0 at (0.2, 0.2) and the global one
    −1.5 at (0.8, 0.8) (each well is 1e-16 deep at the other's centre). A single Bayesian run converges to whichever
    it finds first; it exists to test that independent restarts expose the disagreement."""
    a = math.exp(-((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2) / (2 * 0.1 ** 2))
    b = math.exp(-((x[0] - 0.8) ** 2 + (x[1] - 0.8) ** 2) / (2 * 0.1 ** 2))
    return -1.0 * a - 1.5 * b
