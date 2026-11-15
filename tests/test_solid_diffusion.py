"""Validation of the solid-diffusion block: transient diffusion in solids, membrane permeation / time lag, D from uptake.

Three kinds of reference, kept apart on purpose:
  A. ANALYTICAL — the series of Crank, "The Mathematics of Diffusion" (2nd ed., ch. 3-6), written here independently of the
     engine (brentq roots, scipy's own Bessel zeros, Crank's algebraic forms for the profiles, error-function short-time
     forms), the Daynes-Barrer time lag and its laminate / hollow-cylinder / hollow-sphere closed forms.
  B. WORKED EXAMPLES AND TABLES READ ON 2026-11-14 (numbers copied from the page that opened, never from memory):
     * Geankoplis, "Transport Processes and Unit Operations" 3rd ed., Examples 5.3-2, 5.3-3, 5.3-4, 6.5-1, 6.5-2, 7.1-1 and
       7.1-2, as transcribed with data and printed results in the FOSSEE Python Textbook Companion:
       https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter05.ipynb
       https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter06.ipynb
       https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter07.ipynb
       The book reads Y off the Gurney-Lurie / Heisler charts (Figs. 5.3-5, 5.3-6, 5.3-8), so its transient answers carry
       chart-reading error: the bands below are the deviations MEASURED against the book, with the exact series alongside.
     * Incropera, "Fundamentals of Heat and Mass Transfer", Table 5.1 (one-term coefficients ζ1, C1), rows quoted in the
       UC Berkeley ME 109 exam solutions: https://tbp.studentorg.berkeley.edu/exams/3537/download (sphere, Bi = 1 and 2)
       and https://tbp.studentorg.berkeley.edu/exams/337/download (plane wall, Bi → ∞, and θ0*(Fo = 0.78) = 0.1858).
     * Noszticzius et al., arXiv:1304.5163 (https://arxiv.org/pdf/1304.5163), eqs. (1)-(2): gelatin membrane d = 0.5 mm,
       time lag 175 s ↔ D = 2.4e-6 cm²/s, and steady current 30 nmol/s over 28.3 cm² with Δc = 20.1e-3 M ↔ D = 2.6e-6 cm²/s.
  C. INDEPENDENT NUMERICS for what has no closed form: the Boltzmann similarity ODE for D(c) (scipy solve_bvp) and a 2-D
     finite-volume / Crank-Nicolson solve of a finite cylinder for the product rule.
Plus physics identities, refusals on missing inputs and JSON round-trips."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import integrate, optimize, sparse, special
from scipy.sparse.linalg import splu

from packages.agent import wrappers_solid_diffusion as W
from packages.dwsim_client import tools_solid_diffusion as T
from packages.transport import solid_diffusion as SD

GEOS = {"slab": (0, {"thickness_m": 2.0, "exposed_faces": 2}), "cylinder": (1, {"radius_m": 1.0}),
        "sphere": (2, {"radius_m": 1.0})}
FIXED = {"surface_condition": "fixed_concentration", "surface_concentration_mol_m3": 1.0}


def conv(bi: float, cf: float = 1.0, kp: float = 1.0) -> dict:
    """Film with Bi = k_c·a/(K·D) for a = 1, D = 1."""
    return {"surface_condition": "convective", "fluid_concentration_mol_m3": cf, "mass_transfer_coefficient_m_s": bi * kp,
            "partition_coefficient": kp}


def run(geo: str, times, bc: dict, c0: float = 0.0, d: float = 1.0, **kw) -> dict:
    args = dict(GEOS[geo][1]) if geo in GEOS else {}
    args.update(kw)
    r = T.transient_diffusion_solid(geometry=geo, initial_concentration_mol_m3=c0, diffusivity_m2_s=d, times_s=list(times),
                                    **args, **bc)
    json.dumps(r)
    return r


# ---------------------------------------------------------------------------------------------------------------------
# A. independent analytical references (Crank)
def ref_roots(m: int, L: float, n: int) -> np.ndarray:
    """Crank eqs. 4.51 (β tanβ = L), 5.47 (β J1 = L J0), 6.41 (β cotβ + L − 1 = 0), one brentq per root."""
    eps = 1e-10
    if m == 0:
        return np.array([optimize.brentq(lambda b: b * math.tan(b) - L, k * math.pi + eps, (k + 0.5) * math.pi - eps,
                                         xtol=1e-15) for k in range(n)])
    if m == 1:
        j0, j1 = special.jn_zeros(0, n), np.concatenate([[0.0], special.jn_zeros(1, n)])
        return np.array([optimize.brentq(lambda b: b * special.j1(b) - L * special.j0(b), j1[k] + eps, j0[k], xtol=1e-15)
                         for k in range(n)])
    return np.array([optimize.brentq(lambda b: b * math.cos(b) + (L - 1.0) * math.sin(b), k * math.pi + eps,
                                     (k + 1) * math.pi, xtol=1e-15) for k in range(n)])


def ref_uptake_fixed(m: int, fo: float, n: int = 300) -> float:
    """Crank eqs. 4.18, 5.23, 6.20."""
    k = np.arange(n)
    if m == 0:
        return 1.0 - float(np.sum(8.0 / ((2 * k + 1) ** 2 * np.pi ** 2) * np.exp(-(2 * k + 1) ** 2 * np.pi ** 2 * fo / 4.0)))
    if m == 1:
        al = special.jn_zeros(0, n)
        return 1.0 - float(np.sum(4.0 / al ** 2 * np.exp(-al ** 2 * fo)))
    return 1.0 - 6.0 / np.pi ** 2 * float(np.sum(np.exp(-(k + 1) ** 2 * np.pi ** 2 * fo) / (k + 1) ** 2))


def ref_uptake_short(m: int, fo: float) -> float:
    """Short-time (image / error-function) forms, exact up to exp(−1/Fo): slab 2√(Fo/π) (Crank 4.20), sphere 6√(Fo/π) − 3Fo
    (Crank 6.22)."""
    return 2.0 * math.sqrt(fo / math.pi) if m == 0 else 6.0 * math.sqrt(fo / math.pi) - 3.0 * fo


def ref_uptake_conv(m: int, L: float, fo: float, n: int = 120) -> float:
    """Crank eqs. 4.53, 5.49, 6.43 (surface evaporation condition)."""
    b = ref_roots(m, L, n)
    den = (b ** 2 + L ** 2 + L, b ** 2 + L ** 2, b ** 2 + L * (L - 1.0))[m]
    return 1.0 - float(np.sum(2.0 * (m + 1) * L ** 2 * np.exp(-b ** 2 * fo) / (b ** 2 * den)))


def ref_profile_conv(m: int, L: float, fo: float, x: np.ndarray, n: int = 120) -> np.ndarray:
    """(C − C0)/(C∞ − C0) at x = r/a, Crank eqs. 4.50, 5.46, 6.40."""
    b = ref_roots(m, L, n)[None, :]
    x = np.asarray(x, dtype=float)[:, None]
    e = np.exp(-b ** 2 * fo)
    if m == 0:
        return 1.0 - np.sum(2.0 * L * np.cos(b * x) * e / ((b ** 2 + L ** 2 + L) * np.cos(b)), axis=1)
    if m == 1:
        return 1.0 - np.sum(2.0 * L * special.j0(b * x) * e / ((b ** 2 + L ** 2) * special.j0(b)), axis=1)
    return 1.0 - np.sum(2.0 * L * np.sin(b * x) / x * e / ((b ** 2 + L * (L - 1.0)) * np.sin(b)), axis=1)


def ref_bath(m: int, alpha: float, fo: float, n: int = 200) -> float:
    """Well-stirred solution of limited volume, Crank eqs. 4.37-4.38, 5.33-5.34, 6.30-6.31."""
    eps = 1e-10
    if m == 0:
        q = [optimize.brentq(lambda z: math.tan(z) + alpha * z, (k + 0.5) * math.pi + eps, (k + 1) * math.pi, xtol=1e-15)
             for k in range(n)]
    elif m == 1:
        j0, j1 = special.jn_zeros(0, n), special.jn_zeros(1, n)
        q = [optimize.brentq(lambda z: alpha * z * special.j0(z) + 2.0 * special.j1(z), j0[k], j1[k], xtol=1e-15)
             for k in range(n)]
    else:
        q = [optimize.brentq(lambda z: math.tan(z) - 3.0 * z / (3.0 + alpha * z * z), (k + 1) * math.pi,
                             (k + 1.5) * math.pi - eps, xtol=1e-15) for k in range(n)]
    q = np.array(q)
    nu = m + 1.0
    return 1.0 - float(np.sum(2.0 * nu * alpha * (1.0 + alpha) / (nu ** 2 * (1.0 + alpha) + alpha ** 2 * q ** 2)
                              * np.exp(-q ** 2 * fo)))


FO = [0.002, 0.01, 0.05, 0.2, 0.6, 1.0]


@pytest.mark.parametrize("geo", ["slab", "cylinder", "sphere"])
def test_fixed_surface_matches_crank_series(geo):
    m = GEOS[geo][0]
    r = run(geo, FO, FIXED)
    dev = [abs(x["fractional_uptake"] - ref_uptake_fixed(m, x["time_s"])) for x in r["results"]]
    assert max(dev) < 5e-5, dev                      # measured: slab 7e-6, cylinder 1.5e-5, sphere 2.3e-5 (absolute in M_t/M_∞)
    if m != 1:                                       # the two independent closed forms agree with each other and with the engine
        for x in r["results"][:2]:
            assert abs(ref_uptake_fixed(m, x["time_s"]) - ref_uptake_short(m, x["time_s"])) < 1e-12
            assert abs(x["fractional_uptake"] - ref_uptake_short(m, x["time_s"])) < 5e-5
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert r["evidence"]["analytical"]["fractional_uptake_max_abs_diff"] < 5e-5


@pytest.mark.parametrize("geo", ["slab", "cylinder", "sphere"])
@pytest.mark.parametrize("bi", [0.1, 1.0, 10.0])
def test_convective_surface_matches_crank_eigenvalue_series(geo, bi):
    m = GEOS[geo][0]
    fos = [0.05, 0.2, 1.0]
    r = run(geo, fos, conv(bi), profile_points=11)
    assert abs(r["biot_number_mass"] - bi) < 1e-12
    for x in r["results"]:
        fo = x["time_s"]
        assert abs(x["fractional_uptake"] - ref_uptake_conv(m, bi, fo)) < 3e-5            # measured ≤ 9e-6
        pos = np.array(x["profile"]["position_m"])[1:]                                    # skip r = 0 (sin(βr)/r form)
        got = np.array(x["profile"]["concentration_mol_m3"])[1:]
        assert np.max(np.abs(got - ref_profile_conv(m, bi, fo, pos))) < 3e-4, (geo, bi, fo)   # measured ≤ 1.1e-4
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


@pytest.mark.parametrize("geo", ["slab", "cylinder", "sphere"])
@pytest.mark.parametrize("alpha", [0.25, 1.5, 9.0])
def test_limited_volume_bath_matches_crank(geo, alpha):
    m = GEOS[geo][0]
    kp, vs = 2.0, 0.3
    vb = alpha * kp * vs
    fos = [0.01, 0.1, 0.5]
    r = run(geo, fos, {"surface_condition": "limited_bath", "fluid_concentration_mol_m3": 5.0, "partition_coefficient": kp,
                       "bath_volume_m3": vb, "solid_volume_m3": vs}, c0=1.0)
    for x in r["results"]:
        assert abs(x["fractional_uptake"] - ref_bath(m, alpha, x["time_s"])) < 1e-4        # measured ≤ 4e-5
        # solute conservation between bath and solid at every time
        assert abs(vb * (5.0 - x["bath_concentration_mol_m3"]) - vs * x["uptake_mol_per_m3_solid"]) < 1e-9
    c_inf = (vb * 5.0 + vs * 1.0) / (vb + kp * vs)
    assert abs(r["bath"]["final_bath_concentration_mol_m3"] - c_inf) < 1e-12
    assert abs(r["equilibrium_concentration_mol_m3"] - kp * c_inf) < 1e-12
    assert abs(r["bath"]["final_fraction_of_solute_taken_from_bath"] - 1.0 / (1.0 + alpha)) < 1e-12
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


def test_semi_infinite_erfc_fixed_and_film():
    d, t = 4e-9, 3e4
    r = T.transient_diffusion_solid(geometry="semi_infinite", initial_concentration_mol_m3=0.0, diffusivity_m2_s=d,
                                    surface_condition="fixed_concentration", surface_concentration_mol_m3=3.0, times_s=[t, 4 * t],
                                    profile_points=41)
    x = r["results"][0]
    depth, c = np.array(x["profile"]["depth_from_surface_m"]), np.array(x["profile"]["concentration_mol_m3"])
    assert np.max(np.abs(c - 3.0 * special.erfc(depth / (2 * math.sqrt(d * t))))) < 3e-4 * 3.0     # Crank eq. 3.13; measured 4e-5
    assert abs(x["uptake_mol_m2"] / (3.0 * 2 * math.sqrt(d * t / math.pi)) - 1) < 2e-4            # Crank eq. 3.15
    assert abs(x["surface_flux_mol_m2_s"] / (3.0 * math.sqrt(d / (math.pi * t))) - 1) < 2e-3
    assert abs(r["results"][1]["uptake_mol_m2"] / x["uptake_mol_m2"] - 2.0) < 2e-4                 # M ∝ √t
    assert abs(x["penetration_depth_1pct_m"] / (2 * special.erfcinv(0.01) * math.sqrt(d * t)) - 1) < 5e-3
    assert r["evidence"]["verified"]


# ---------------------------------------------------------------------------------------------------------------------
# B. worked examples and tables read on 2026-11-14
def test_geankoplis_7_1_1_urea_in_agar_slab():
    """Ex. 7.1-1 (p. 431): slab 10.16 mm, c0 = 0.1 kmol/m³ urea, surface 0, D = 4.72e-10 m²/s, 10 h. Book (Fig. 5.3-5):
    X = 0.658, Y = 0.275 → 0.0275 kmol/m³ at the centre; Y = 0.172 → 0.0172 at 2.54 mm from the surface; (b) half the
    thickness: Y = 0.0020 → 2.0e-4. Exact series: Y = 0.2508, 0.1773, 0.00192."""
    kw = dict(geometry="slab", exposed_faces=2, initial_concentration_mol_m3=100.0, diffusivity_m2_s=4.72e-10,
              surface_condition="fixed_concentration", surface_concentration_mol_m3=0.0, times_s=[36000.0], profile_points=5)
    r = T.transient_diffusion_solid(thickness_m=10.16e-3, **kw)
    x = r["results"][0]
    assert abs(x["fourier_number"] - 0.658) < 1e-3
    centre, quarter = x["center_concentration_mol_m3"], x["profile"]["concentration_mol_m3"][2]   # 2.54 mm from either face
    assert abs(x["profile"]["position_m"][2] - 2.54e-3) < 1e-12
    exact = 100.0 * (4 / math.pi) * math.exp(-math.pi ** 2 * x["fourier_number"] / 4)             # higher terms < 1e-6
    assert abs(centre - exact) < 2e-3 and abs(quarter - exact * math.cos(math.pi / 4)) < 2e-3
    assert abs(centre / 27.5 - 1) < 0.10          # measured −8.8 % (book chart reading 0.275 vs exact 0.2508)
    assert abs(quarter / 17.2 - 1) < 0.04         # measured +3.1 %
    b = T.transient_diffusion_solid(thickness_m=5.08e-3, **kw)["results"][0]
    assert abs(b["fourier_number"] - 2.632) < 3e-3
    assert abs(b["center_concentration_mol_m3"] / 0.20 - 1) < 0.05      # measured −4.1 % (1.92e-4 vs 2.0e-4 kmol/m³)
    assert r["evidence"]["verified"]


def test_geankoplis_7_1_2_semi_infinite_with_film_and_partition():
    """Ex. 7.1-2 (p. 431): c0 = 0.01, c_fluid = 0.10 kmol/m³, k_c = 2e-7 m/s, K = c_L/c_i = 2.0 (Geankoplis' K is fluid over
    solid → partition_coefficient = 0.5), D = 4e-9 m²/s, t = 3e4 s. Book (Fig. 5.3-3): 1 − Y = 0.62 at the surface →
    0.0348 kmol/m³, 1 − Y = 0.26 at x = 0.01 m → 0.0204 kmol/m³. Exact erfc solution: 0.03389 and 0.02057."""
    r = T.transient_diffusion_solid(geometry="semi_infinite", initial_concentration_mol_m3=10.0, diffusivity_m2_s=4e-9,
                                    surface_condition="convective", fluid_concentration_mol_m3=100.0,
                                    mass_transfer_coefficient_m_s=2e-7, partition_coefficient=0.5, times_s=[3e4], profile_points=201)
    x = r["results"][0]
    assert abs(r["equilibrium_concentration_mol_m3"] - 50.0) < 1e-12
    at_1cm = float(np.interp(0.01, x["profile"]["depth_from_surface_m"], x["profile"]["concentration_mol_m3"]))
    s, h = math.sqrt(4e-9 * 3e4), 2e-7 / (0.5 * 4e-9)
    exact = [10 + 40 * (special.erfc(xx / (2 * s)) - math.exp(h * xx + h * h * s * s) * special.erfc(xx / (2 * s) + h * s))
             for xx in (0.0, 0.01)]                                              # Crank eq. 3.35
    assert abs(x["surface_concentration_mol_m3"] - exact[0]) < 5e-3 and abs(at_1cm - exact[1]) < 5e-3   # measured 5e-4, 2e-3
    assert abs(x["surface_concentration_mol_m3"] / 34.8 - 1) < 0.035     # measured −2.6 % against the chart reading
    assert abs(at_1cm / 20.4 - 1) < 0.015                                 # measured +0.8 %
    assert r["evidence"]["verified"]


def test_geankoplis_heat_analogues_5_3_2_and_5_3_3():
    """Heat conduction = same equation (D ↔ α, k_c/K ↔ h/ρcp, Bi = hL/k). Ex. 5.3-2 (p. 338, butter slab 46.2 mm, one face
    insulated, h = 8.52, k = 0.197, ρ = 998, cp = 2300, 5 h, 277.6 → 297.1 K; chart Y = 0.25 / 0.45 / 0.50 → 292.2 K at the
    surface, 288.3 K 25.4 mm below it, 287.4 K at the insulated face). Ex. 5.3-3 (p. 342, can of pea purée, cylinder
    D = 0.0681 m, α = 2.007e-7, h = 4540, k = 0.830, 0.75 h, 29.4 → 115.6 °C; chart Y = 0.13 → 104.4 °C at the centre)."""
    k, rho, cp, h = 0.197, 998.0, 2300.0, 8.52
    r = T.transient_diffusion_solid(geometry="slab", thickness_m=0.0462, exposed_faces=1, initial_concentration_mol_m3=277.6,
                                    diffusivity_m2_s=k / (rho * cp), surface_condition="convective",
                                    fluid_concentration_mol_m3=297.1, mass_transfer_coefficient_m_s=h / (rho * cp),
                                    partition_coefficient=1.0, times_s=[18000.0], profile_points=463)
    x = r["results"][0]
    assert abs(r["biot_number_mass"] - 2.0) < 0.01 and abs(x["fourier_number"] - 0.724) < 1e-3     # book: m = 0.5, X = 0.724
    below = float(np.interp(0.0462 - 0.0254, x["profile"]["position_m"], x["profile"]["concentration_mol_m3"]))
    got = (x["surface_concentration_mol_m3"], below, x["center_concentration_mol_m3"])
    for g, book in zip(got, (292.2, 288.3, 287.4)):
        assert abs(g - book) < 0.3, got            # measured +0.19, +0.01, −0.23 K (chart reading, 19.5 K driving force)
    ex = 297.1 - 19.5 * ref_profile_conv(0, r["biot_number_mass"], x["fourier_number"], np.array([1.0, 1 - 0.0254 / 0.0462, 0.0]))
    ex = 297.1 - (297.1 - 277.6) * (1.0 - ref_profile_conv(0, r["biot_number_mass"], x["fourier_number"],
                                                           np.array([1.0, 1 - 0.0254 / 0.0462, 0.0])))
    assert np.max(np.abs(np.array(got) - ex)) < 2e-3
    c = T.transient_diffusion_solid(geometry="cylinder", radius_m=0.0681 / 2, initial_concentration_mol_m3=29.4,
                                    diffusivity_m2_s=2.007e-7, surface_condition="convective", fluid_concentration_mol_m3=115.6,
                                    mass_transfer_coefficient_m_s=4540 * 2.007e-7 / 0.830, partition_coefficient=1.0,
                                    times_s=[2700.0])
    centre = c["results"][0]["center_concentration_mol_m3"]
    assert abs(c["results"][0]["fourier_number"] - 0.467) < 1e-3
    y_exact = 1.0 - ref_profile_conv(1, c["biot_number_mass"], c["results"][0]["fourier_number"], np.array([1e-9]))[0]
    assert abs((115.6 - centre) / 86.2 - y_exact) < 1e-4
    assert abs(centre - 104.4) < 2.0               # measured +1.7 °C: exact Y = 0.1105, the book reads 0.13 from Fig. 5.3-8
    assert r["evidence"]["verified"] and c["evidence"]["verified"]


def test_geankoplis_5_3_4_short_cylinder_product_rule():
    """Ex. 5.3-4 (p. 347): the same can, height 0.1016 m, both ends exposed. Book: Yx = 0.13 (radial), Yy = 0.80 (axial,
    Fig. 5.3-6) → 106.6 °C at the centre. Exact factors: 0.1105 and 0.7571 → 108.4 °C."""
    r = T.transient_diffusion_solid(geometry="finite_cylinder", radius_m=0.0681 / 2, length_m=0.1016,
                                    initial_concentration_mol_m3=29.4, diffusivity_m2_s=2.007e-7, surface_condition="convective",
                                    fluid_concentration_mol_m3=115.6, mass_transfer_coefficient_m_s=4540 * 2.007e-7 / 0.830,
                                    partition_coefficient=1.0, times_s=[2700.0])
    json.dumps(r)
    x = r["results"][0]
    assert abs(x["directions"]["axial"]["fourier_number"] - 0.21) < 1e-3
    assert abs(x["directions"]["radial"]["center_unaccomplished"] / 0.13 - 1) < 0.16      # measured −15 % (chart)
    assert abs(x["directions"]["axial"]["center_unaccomplished"] / 0.80 - 1) < 0.06       # measured −5.4 % (chart)
    assert abs(x["center_concentration_mol_m3"] - 106.6) < 2.0                            # measured +1.8 °C
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


def _one_term(geo: str, bi: float | None) -> tuple[float, float]:
    """ζ1 and C1 extracted from the ENGINE's centre value at Fo = 1.0 and 1.5 (higher modes < e^-20)."""
    bc = dict(FIXED, surface_concentration_mol_m3=0.0) if bi is None else conv(bi, cf=0.0)
    th = [x["center_concentration_mol_m3"] for x in run(geo, [1.0, 1.5], bc, c0=1.0)["results"]]
    z2 = math.log(th[0] / th[1]) / 0.5
    return math.sqrt(z2), th[0] * math.exp(z2)


def test_incropera_table_5_1_one_term_coefficients():
    """Incropera Table 5.1 rows read in the Berkeley ME 109 solutions: sphere Bi = 1.0 → ζ1 = 1.5708, C1 = 1.2732;
    sphere Bi = 2.0 → ζ1 = 2.0288, C1 = 1.4793; plane wall Bi → ∞ → ζ1 = 1.5708, C1 = 1.2732 (1.2733 in the text)."""
    for geo, bi, z_tab, c_tab in (("sphere", 1.0, 1.5708, 1.2732), ("sphere", 2.0, 2.0288, 1.4793), ("slab", None, 1.5708, 1.2732)):
        z, c = _one_term(geo, bi)
        assert abs(z - z_tab) < 1.5e-4, (geo, bi, z)       # measured −1.6e-5, −7e-5, −1.6e-5 (table rounded to 4 decimals)
        assert abs(c - c_tab) < 3e-4, (geo, bi, c)         # measured +1.3e-4, +1.9e-4, +4e-5


def test_berkeley_plane_wall_midplane_and_ten_eigenvalues():
    """Same source (exam 337): plane wall with fixed surface temperature, θ0* ≈ 1 at Fo = 0.0078 and
    θ0*(Fo = 0.78) = 1.2733·exp(−(π/2)²·0.78) = 0.1858; table of ζi and Ci, i = 1..10."""
    r = run("slab", [0.0078, 0.78], dict(FIXED, surface_concentration_mol_m3=0.0), c0=1.0)
    th = [x["center_concentration_mol_m3"] for x in r["results"]]
    assert abs(th[0] - 1.0) < 1e-9 and abs(th[1] - 0.1858) < 1e-4       # measured 0.18582
    assert r["results"][0]["regime"] == "tiempos_cortos_semi_infinito"
    assert r["results"][1]["regime"] == "tiempos_largos_primer_autovalor"
    zeta = [1.5708, 4.7124, 7.8540, 10.9956, 14.1372, 17.2788, 20.4204, 23.5619, 26.7035, 29.8451]
    ci = [1.2732, -0.4244, 0.2546, -0.1819, 0.1415, -0.1157, 0.0979, -0.0849, 0.0749, -0.0670]
    beta = SD.eigenvalues(0, math.inf, 10)
    assert np.max(np.abs(beta - zeta)) < 6e-5
    assert np.max(np.abs(SD._profile_terms(0, beta, np.array([0.0]))[0] - ci)) < 6e-5


def test_geankoplis_6_5_membranes():
    """Ex. 6.5-1 (p. 409): H2 through neoprene 0.5 mm, 0.010 atm → 0, S = 0.051 m³(STP)/(m³·atm), D = 1.03e-10 m²/s:
    c_A1 = 2.28e-5 kmol/m³, N_A = 4.69e-12 kmol/(m²·s). Ex. 6.5-2 (p. 411): O2 through polyethylene 0.00015 m,
    0.21 → 0.01 atm, P_M = 4.17e-12 m³(STP)/(s·m²·atm/m): N_A = 2.480e-10 kmol/(m²·s) (2.481e-10 in the notebook)."""
    r = T.membrane_permeation(layers=[{"thickness_m": 0.5e-3, "diffusivity_m2_s": 1.03e-10, "solubility_m3STP_m3_atm": 0.051}],
                              geometry="flat", upstream_partial_pressure_Pa=0.01 * 101325, downstream_partial_pressure_Pa=0.0)
    json.dumps(r)
    assert abs(r["flux_mol_m2_s"] / 4.69e-9 - 1) < 1.5e-3                                    # measured −0.06 % (3 figures)
    assert abs(r["layers"][0]["upstream_face_concentration_in_layer_mol_m3"] / 2.28e-2 - 1) < 3e-3   # measured −0.2 %
    assert r["evidence"]["verified"]
    r = T.membrane_permeation(layers=[{"thickness_m": 0.00015, "permeability_m3STP_m_m2_s_atm": 4.17e-12}], geometry="flat",
                              upstream_partial_pressure_Pa=0.21 * 101325, downstream_partial_pressure_Pa=0.01 * 101325)
    assert abs(r["flux_mol_m2_s"] / 2.481e-7 - 1) < 3e-4                                     # measured −0.02 %
    assert r["layers"][0]["diffusivity_m2_s"] is None and r["evidence"]["verified"]


def test_time_lag_published_example_and_daynes_barrer():
    """arXiv:1304.5163 eqs. (1)-(2): d = 0.5 mm; time lag 175 s ↔ D = 2.4e-6 cm²/s; J = 30 nmol/s, A = 28.3 cm²,
    Δc = 20.1e-3 M ↔ D = 2.6e-6 cm²/s."""
    def cell(d):
        return T.membrane_permeation(layers=[{"thickness_m": 0.5e-3, "diffusivity_m2_s": d, "partition_coefficient": 1.0}],
                                     geometry="flat", upstream_concentration_mol_m3=20.1, downstream_concentration_mol_m3=0.0,
                                     area_m2=28.3e-4, transient=True, initial_concentration_mol_m3=0.0)
    r = cell(2.4e-10)
    tl = r["transient"]
    assert abs(tl["time_lag_s"] / 175.0 - 1) < 0.01                    # measured −0.8 % (the paper rounds D to 2 figures)
    assert abs(tl["time_lag_s"] / (0.5e-3 ** 2 / (6 * 2.4e-10)) - 1) < 2e-4        # θ = L²/6D; measured 5.7e-5
    assert abs(tl["time_lag_L2_over_6D_s"] - 0.5e-3 ** 2 / (6 * 2.4e-10)) < 1e-9
    assert abs(cell(2.6e-10)["rate_mol_s"] / 30e-9 - 1) < 0.02         # measured −1.4 % (D rounded in the paper)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    json.dumps(r)


# ---------------------------------------------------------------------------------------------------------------------
# A (cont.). membranes: analytical transient and closed-form time lags
def test_downstream_cumulative_curve_matches_crank_4_24a():
    L, d, c1 = 2e-4, 3e-11, 40.0
    times = [0.1 * L * L / d * k for k in (0.2, 0.5, 1, 2, 4, 8, 15)]
    r = T.membrane_permeation(layers=[{"thickness_m": L, "diffusivity_m2_s": d, "partition_coefficient": 1.0}], geometry="flat",
                              upstream_concentration_mol_m3=c1, downstream_concentration_mol_m3=0.0, transient=True,
                              initial_concentration_mol_m3=0.0, times_s=times)
    n = np.arange(1, 400)
    ref = np.array([L * c1 * (d * t / L ** 2 - 1 / 6 - 2 / np.pi ** 2 * np.sum((-1.0) ** n / n ** 2
                                                                              * np.exp(-d * n ** 2 * np.pi ** 2 * t / L ** 2)))
                    for t in times])
    got = np.array(r["transient"]["curve"]["downstream_cumulative"])
    assert np.max(np.abs(got - ref)) / (L * c1 / 6) < 2e-4             # measured 6e-5 of the time-lag intercept L·c1/6
    # flux at the downstream face: J/J_ss = 1 + 2 Σ (−1)^n exp(−n²π²Dt/L²)
    jref = np.array([1 + 2 * np.sum((-1.0) ** n * np.exp(-d * n ** 2 * np.pi ** 2 * t / L ** 2)) for t in times])
    assert np.max(np.abs(np.array(r["transient"]["curve"]["downstream_rate"]) / r["flux_mol_m2_s"] - jref)) < 5e-4
    assert r["evidence"]["verified"], r["evidence"]["reasons"]


def test_time_lag_laminate_cylinder_sphere_closed_forms():
    """Closed forms derived from Frisch's integral θ = ∫G·S·u_ss dV / n_ss (degassed wall, zero downstream):
    two-layer laminate θ = [L1²/D1·(R1/6 + R2/2) + L2²/D2·(R1/2 + R2/6)]/(R1 + R2), R = L/P (Barrie et al.), independent of
    the layer order; hollow cylinder θ = [a² − b² + (a² + b²)·ln(b/a)]/(4D·ln(b/a)) (Crank eq. 5.67); hollow sphere (b − a)²/6D."""
    la = [{"thickness_m": 1e-4, "diffusivity_m2_s": 1e-11, "solubility_mol_m3_Pa": 2e-4},
          {"thickness_m": 3e-4, "diffusivity_m2_s": 5e-10, "solubility_mol_m3_Pa": 1e-5}]
    r1, r2 = (x["thickness_m"] / (x["diffusivity_m2_s"] * x["solubility_mol_m3_Pa"]) for x in la)
    th = ((1e-4 ** 2 / 1e-11) * (r1 / 6 + r2 / 2) + (3e-4 ** 2 / 5e-10) * (r1 / 2 + r2 / 6)) / (r1 + r2)
    for layers in (la, la[::-1]):
        r = T.membrane_permeation(layers=layers, geometry="flat", upstream_partial_pressure_Pa=1e5,
                                  downstream_partial_pressure_Pa=0.0, transient=True, initial_partial_pressure_Pa=0.0)
        assert abs(r["flux_mol_m2_s"] / (1e5 / (r1 + r2)) - 1) < 1e-12
        assert abs(r["transient"]["time_lag_s"] / th - 1) < 1e-4            # measured 1.3e-5
        assert abs(r["transient"]["time_lag_frisch_integral_s"] / th - 1) < 1e-10
        assert r["evidence"]["verified"], r["evidence"]["reasons"]
    a, b, d, s = 0.01, 0.015, 1e-10, 1e-5
    for side in ("inner", "outer"):
        for geo in ("cylindrical", "spherical"):
            r = T.membrane_permeation(layers=[{"thickness_m": b - a, "diffusivity_m2_s": d, "solubility_mol_m3_Pa": s}],
                                      geometry=geo, inner_radius_m=a, length_m=2.0, upstream_side=side,
                                      upstream_partial_pressure_Pa=1e5, downstream_partial_pressure_Pa=0.0, transient=True,
                                      initial_partial_pressure_Pa=0.0)
            if geo == "cylindrical":
                th = (a * a - b * b + (a * a + b * b) * math.log(b / a)) / (4 * d * math.log(b / a))
                rate = 2 * math.pi * 2.0 * d * s * 1e5 / math.log(b / a)
            else:
                th, rate = (b - a) ** 2 / (6 * d), 4 * math.pi * d * s * 1e5 / (1 / a - 1 / b)
            assert abs(r["rate_mol_s"] / rate - 1) < 1e-12
            assert abs(r["transient"]["time_lag_s"] / th - 1) < 2e-4, (geo, side)     # measured 6e-5
            assert r["evidence"]["verified"], r["evidence"]["reasons"]
            json.dumps(r)


def test_membrane_steady_identities():
    base = dict(geometry="flat", upstream_partial_pressure_Pa=2e5, downstream_partial_pressure_Pa=5e4)
    one = [{"thickness_m": 1e-4, "permeability_barrer": 10.0}]
    f1 = T.membrane_permeation(layers=one, **base)["flux_mol_m2_s"]
    assert abs(f1 / (10 * SD.BARRER_SI * 1.5e5 / 1e-4) - 1) < 1e-12
    assert abs(SD.BARRER_SI / 3.3464e-16 - 1) < 1e-4                    # 1 Barrer in mol·m/(m²·s·Pa)
    # two equal layers = one of double thickness; splitting does not change the flux
    f2 = T.membrane_permeation(layers=one * 2, **base)["flux_mol_m2_s"]
    assert abs(f2 / (f1 / 2) - 1) < 1e-12
    # a barrier layer dominates and the report says so
    lam = T.membrane_permeation(layers=[{"name": "PE", "thickness_m": 1e-4, "permeability_barrer": 10.0},
                                        {"name": "EVOH", "thickness_m": 1e-5, "permeability_barrer": 1e-3}], **base)
    assert lam["controlling_resistance"] == "EVOH" and lam["layers"][1]["fraction_of_total_resistance"] > 0.99
    assert abs(sum(x["fraction_of_total_resistance"] for x in lam["layers"]) - 1) < 1e-12
    assert abs(lam["layers"][1]["downstream_face_partial_pressure_Pa"] - 5e4) < 1e-6
    # external films add in series; a thin cylindrical wall tends to the flat one
    fl = T.membrane_permeation(layers=one, upstream_film_coefficient=1e-9, downstream_film_coefficient=2e-9, **base)
    rt = 1e-4 / (10 * SD.BARRER_SI) + 1 / 1e-9 + 1 / 2e-9
    assert abs(fl["flux_mol_m2_s"] / (1.5e5 / rt) - 1) < 1e-12 and abs(fl["total_resistance"] / rt - 1) < 1e-12
    cyl = T.membrane_permeation(layers=one, geometry="cylindrical", inner_radius_m=1.0, upstream_side="inner", length_m=1.0,
                                upstream_partial_pressure_Pa=2e5, downstream_partial_pressure_Pa=5e4)
    assert abs(cyl["flux_at_upstream_surface_mol_m2_s"] / f1 - 1) < 1e-4
    assert cyl["flux_at_upstream_surface_mol_m2_s"] > cyl["flux_at_downstream_surface_mol_m2_s"]
    for r in (lam, fl, cyl):
        assert r["evidence"]["verified"]
        json.dumps(r)


def test_membrane_nonzero_initial_and_downstream_conserves_mass():
    la = [{"thickness_m": 1e-4, "diffusivity_m2_s": 1e-11, "solubility_mol_m3_Pa": 2e-4},
          {"thickness_m": 3e-4, "diffusivity_m2_s": 5e-10, "solubility_mol_m3_Pa": 1e-5}]
    r = T.membrane_permeation(layers=la, geometry="flat", upstream_partial_pressure_Pa=1e5, downstream_partial_pressure_Pa=2e4,
                              upstream_film_coefficient=1e-9, downstream_film_coefficient=5e-10, transient=True,
                              initial_partial_pressure_Pa=1e4, times_s=[10, 100, 1000, 1e4])
    ev = r["evidence"]
    assert ev["verified"] and ev["mass_balance_max_rel_error"] < 1e-9
    assert ev["time_lag_numeric_vs_frisch_integral_rel_diff"] < 1e-4          # measured 1e-5
    c = r["transient"]["curve"]
    assert all(u >= d for u, d in zip(c["upstream_cumulative"], c["downstream_cumulative"]))   # the wall is filling
    assert any("desgasificada" in w for w in r["warnings"])


# ---------------------------------------------------------------------------------------------------------------------
# C. independent numerics: D(c) by the Boltzmann similarity ODE, product rule by a 2-D solve
def _boltzmann_uptake(law_d, cs: float) -> tuple[float, object]:
    """Semi-infinite, c(0) = cs, c(∞) = 0, λ = x/√t: (D c')' = −(λ/2) c'. Returns M/√t = −2·D·c'(0) and c(λ)."""
    lam_max = 14.0 * math.sqrt(law_d(cs))

    def ode(lam, y):
        return np.vstack([y[1] / law_d(y[0]), -0.5 * lam * y[1] / law_d(y[0])])

    lam = np.linspace(0.0, lam_max, 400)
    guess = np.vstack([cs * special.erfc(lam / (2 * math.sqrt(law_d(cs)))), -cs * np.exp(-lam ** 2) * 1e-3])
    sol = integrate.solve_bvp(ode, lambda ya, yb: np.array([ya[0] - cs, yb[0]]), lam, guess, tol=1e-9, max_nodes=200000)
    assert sol.success
    return -2.0 * float(sol.sol(0.0)[1]), sol.sol


def test_concentration_dependent_diffusivity_against_boltzmann_ode():
    d0, beta, cs, t = 1e-10, 0.03, 60.0, 5e4             # D rises 6× between the bulk and the surface
    law = {"type": "exponential", "D0_m2_s": d0, "beta_m3_mol": beta}
    r = T.transient_diffusion_solid(geometry="semi_infinite", initial_concentration_mol_m3=0.0, diffusivity_law=law,
                                    surface_condition="fixed_concentration", surface_concentration_mol_m3=cs, times_s=[t, 4 * t],
                                    profile_points=31)
    json.dumps(r)
    m_ref, prof = _boltzmann_uptake(lambda c: d0 * np.exp(beta * np.asarray(c)), cs)
    x = r["results"][0]
    assert abs(x["uptake_mol_m2"] / (m_ref * math.sqrt(t)) - 1) < 5e-4          # measured 6e-5
    depth = np.array(x["profile"]["depth_from_surface_m"])
    assert np.max(np.abs(np.array(x["profile"]["concentration_mol_m3"]) - prof(depth / math.sqrt(t))[0])) < 2e-3 * cs
    assert abs(r["results"][1]["uptake_mol_m2"] / x["uptake_mol_m2"] - 2.0) < 5e-4     # Boltzmann scaling M ∝ √t
    const = 2.0 * cs * math.sqrt(d0 * t / math.pi)
    assert x["uptake_mol_m2"] > 1.5 * const                                     # and far from the constant-D0 answer
    assert r["evidence"]["verified"] and r["evidence"]["analytical"]["applicable"] is False
    assert any("D variable" in w for w in r["warnings"])


def test_diffusivity_laws_are_consistent_with_each_other():
    kw = dict(geometry="sphere", radius_m=1e-3, initial_concentration_mol_m3=0.0, surface_condition="convective",
              fluid_concentration_mol_m3=100.0, mass_transfer_coefficient_m_s=1e-6, partition_coefficient=1.0,
              times_s=[200.0, 1500.0])
    lin = T.transient_diffusion_solid(diffusivity_law={"type": "linear", "D0_m2_s": 1e-10, "alpha_m3_mol": 0.04}, **kw)
    tab = T.transient_diffusion_solid(diffusivity_law={"type": "table", "concentration_mol_m3": [0.0, 100.0],
                                                       "diffusivity_m2_s": [1e-10, 5e-10]}, **kw)      # the same straight line
    flat = T.transient_diffusion_solid(diffusivity_law={"type": "table", "concentration_mol_m3": [0.0, 100.0],
                                                        "diffusivity_m2_s": [2e-10, 2e-10]}, **kw)
    cst = T.transient_diffusion_solid(diffusivity_m2_s=2e-10, **kw)
    for a, b in ((lin, tab), (flat, cst)):
        for xa, xb in zip(a["results"], b["results"]):
            assert abs(xa["fractional_uptake"] - xb["fractional_uptake"]) < 1e-7
    # uptake with D rising from 1e-10 to 5e-10 lies between the two constant-D bounds
    lo = T.transient_diffusion_solid(diffusivity_m2_s=1e-10, **kw)
    hi = T.transient_diffusion_solid(diffusivity_m2_s=5e-10, **kw)
    for x, a, b in zip(lin["results"], lo["results"], hi["results"]):
        assert a["fractional_uptake"] < x["fractional_uptake"] < b["fractional_uptake"]
    assert lin["evidence"]["verified"] and lin["diffusivity"]["D_max_m2_s"] == pytest.approx(5e-10)
    with pytest.raises(ValueError, match="no se extrapola"):
        T.transient_diffusion_solid(diffusivity_law={"type": "table", "concentration_mol_m3": [0.0, 50.0],
                                                     "diffusivity_m2_s": [1e-10, 3e-10]}, **kw)


def _finite_cylinder_2d(radius: float, half_len: float, d: float, kc: float, t_end: float, n: int = 70, steps: int = 400):
    """Unaccomplished fraction (average, centre) of a finite cylinder, film on every face: cell-centred finite volumes in
    (r, z) on a uniform grid, Crank-Nicolson in time. Independent of the engine (2-D, uniform mesh, different integrator)."""
    dr, dz = radius / n, half_len / n
    rf = np.linspace(0.0, radius, n + 1)
    rc = 0.5 * (rf[1:] + rf[:-1])
    vol = (rf[1:] ** 2 - rf[:-1] ** 2) / 2.0 * dz
    idx = lambda i, j: i * n + j                     # noqa: E731  i radial, j axial
    rows, cols, vals = [], [], []
    diag = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            p = idx(i, j)
            if i + 1 < n:
                g = d * rf[i + 1] * dz / dr
                rows += [p, idx(i + 1, j)]
                cols += [idx(i + 1, j), p]
                vals += [g / vol[i], g / vol[i + 1]]
                diag[p] -= g / vol[i]
                diag[idx(i + 1, j)] -= g / vol[i + 1]
            else:
                diag[p] -= radius * dz / (dr / (2 * d) + 1.0 / kc) / vol[i]
            if j + 1 < n:
                g = d * (rf[i + 1] ** 2 - rf[i] ** 2) / 2.0 / dz
                rows += [p, idx(i, j + 1)]
                cols += [idx(i, j + 1), p]
                vals += [g / vol[i], g / vol[i]]
                diag[p] -= g / vol[i]
                diag[idx(i, j + 1)] -= g / vol[i]
            else:
                diag[p] -= (rf[i + 1] ** 2 - rf[i] ** 2) / 2.0 / (dz / (2 * d) + 1.0 / kc) / vol[i]
    a = sparse.csc_matrix((vals, (rows, cols)), shape=(n * n, n * n)) + sparse.diags(diag)
    dt = t_end / steps
    eye = sparse.identity(n * n, format="csc")
    lu = splu((eye - 0.5 * dt * a).tocsc())
    rhs_m = (eye + 0.5 * dt * a).tocsr()
    y = np.ones(n * n)
    for _ in range(4):                               # four half-size implicit-Euler start-up steps damp the CN oscillation
        y = splu((eye - 0.25 * dt * a).tocsc()).solve(y)
    for _ in range(steps - 1):
        y = lu.solve(rhs_m @ y)
    w = np.repeat(vol, n)
    y2 = y.reshape(n, n)
    centre = y2[0, 0] - (y2[1, 0] - y2[0, 0]) / 8.0 - (y2[0, 1] - y2[0, 0]) / 8.0     # even extrapolation to r = 0, z = 0
    return float(w @ y / w.sum()), float(centre)


def test_product_rule_against_independent_2d_solution():
    radius, length, d, kc, t = 0.01, 0.03, 1e-9, 4e-7, 2.0e4          # Bi = 4 radial, 6 axial; Fo = 0.2 radial
    r = T.transient_diffusion_solid(geometry="finite_cylinder", radius_m=radius, length_m=length, initial_concentration_mol_m3=1.0,
                                    diffusivity_m2_s=d, surface_condition="convective", fluid_concentration_mol_m3=0.0,
                                    mass_transfer_coefficient_m_s=kc, partition_coefficient=1.0, times_s=[t],
                                    target_fractional_uptake=0.5)
    x = r["results"][0]
    avg, centre = _finite_cylinder_2d(radius, length / 2, d, kc, t)
    assert abs(x["unaccomplished_fraction"] - avg) < 2e-3            # measured 6e-4 (the 2-D reference is the coarser one)
    assert abs(x["center_unaccomplished_fraction"] - centre) < 2e-3  # measured 4e-4
    # product of Crank's 1-D series, and the reported target time is where the product crosses 0.5
    ya = (1 - ref_uptake_conv(1, kc * radius / d, d * t / radius ** 2)) * (1 - ref_uptake_conv(0, kc * length / 2 / d,
                                                                                               d * t / (length / 2) ** 2))
    assert abs(x["unaccomplished_fraction"] - ya) < 3e-5
    th = r["time_to_target"]["time_s"]
    yh = (1 - ref_uptake_conv(1, kc * radius / d, d * th / radius ** 2)) * (1 - ref_uptake_conv(0, kc * length / 2 / d,
                                                                                                d * th / (length / 2) ** 2))
    assert abs(yh - 0.5) < 1e-4
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    # a block with one very long side tends to the rectangular bar, and a bar with one very long side to the slab
    kw = dict(initial_concentration_mol_m3=1.0, diffusivity_m2_s=d, surface_condition="fixed_concentration",
              surface_concentration_mol_m3=0.0, times_s=[t])
    bar = T.transient_diffusion_solid(geometry="rectangular_bar", thickness_m=0.02, width_m=0.03, **kw)["results"][0]
    blk = T.transient_diffusion_solid(geometry="block", thickness_m=0.02, width_m=0.03, depth_m=50.0, **kw)["results"][0]
    slab = T.transient_diffusion_solid(geometry="slab", thickness_m=0.02, exposed_faces=2, **kw)["results"][0]
    wide = T.transient_diffusion_solid(geometry="rectangular_bar", thickness_m=0.02, width_m=50.0, **kw)["results"][0]
    assert abs(bar["fractional_uptake"] - blk["fractional_uptake"]) < 1e-3
    assert abs(slab["fractional_uptake"] - wide["fractional_uptake"]) < 1e-3
    assert bar["fractional_uptake"] > slab["fractional_uptake"]


# ---------------------------------------------------------------------------------------------------------------------
# numerical evidence: mesh and time-step independence, conservation
def test_second_order_mesh_convergence_and_coarse_mesh_is_flagged():
    errs = []
    for n in (50, 100, 200, 400):
        r = run("sphere", [0.01, 0.1], FIXED, n_cells=n)
        errs.append(max(abs(x["fractional_uptake"] - ref_uptake_fixed(2, x["time_s"])) for x in r["results"]))
        assert r["evidence"]["mesh_refinement_max_diff"] == pytest.approx(0.75 * errs[-1], rel=0.05)   # Richardson: e_h − e_h/2
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    assert all(3.8 < q < 4.2 for q in ratios), (errs, ratios)          # measured 4.00, 4.00, 4.00
    assert errs[2] < 3e-5                                              # default mesh: 2.2e-5
    coarse = run("sphere", [0.001], FIXED, n_cells=20)
    assert coarse["evidence"]["verified"] is False and coarse["evidence"]["reasons"]


def test_evidence_fields_and_time_tolerance_independence():
    r = run("cylinder", [0.02, 0.3], conv(3.0), target_fractional_uptake=0.8)
    ev = r["evidence"]
    assert ev["verified"] and ev["reasons"] == []
    assert ev["mass_balance_max_rel_error"] < 1e-10 and ev["mass_balance_at_target_rel_error"] < 1e-10
    assert ev["time_tolerance_max_diff"] < 1e-6 and ev["mesh_refinement_max_diff"] < 1e-4
    assert ev["target_time_mesh_rel_diff"] < 1e-4 and ev["target_time_tolerance_rel_diff"] < 1e-6
    th = r["time_to_target"]["time_s"]
    assert abs(ref_uptake_conv(1, 3.0, th) - 0.8) < 2e-5
    assert abs(ev["analytical"]["target_fractional_uptake_from_series_at_reported_time"] - 0.8) < 2e-5
    for key in ("warnings", "assumptions", "not_included", "method"):
        assert r[key] is not None and (key == "warnings" or len(r[key]) > 0)
    # the surface flux is the time derivative of the uptake: N·(S/V) = d(c_avg)/dt, here against the series slope
    x = r["results"][1]
    h = 1e-4
    slope = (ref_uptake_conv(1, 3.0, x["time_s"] + h) - ref_uptake_conv(1, 3.0, x["time_s"] - h)) / (2 * h)
    assert abs(x["surface_flux_mol_m2_s"] * 2.0 / slope - 1) < 1e-3      # cylinder S/V = 2/a, a = 1, Δc = 1


def test_target_with_times_runs_in_one_pass_and_runaway_is_reported(monkeypatch):
    """Regression: times_s + target used to restart LSODA after the last instant; at the tight-tolerance check run the
    restarted solver crawled for minutes. Now one pass from t = 0, and any integration over budget is reported."""
    r = run("cylinder", [0.02, 0.3], conv(3.0), target_fractional_uptake=0.8)
    assert r["evidence"]["rhs_evaluations"] < 20000
    alone = T.transient_diffusion_solid(geometry="cylinder", radius_m=1.0, initial_concentration_mol_m3=0.0, diffusivity_m2_s=1.0,
                                        target_fractional_uptake=0.8, **conv(3.0))
    assert alone["time_to_target"]["time_s"] == pytest.approx(r["time_to_target"]["time_s"], rel=1e-6)
    monkeypatch.setattr(SD, "MAX_RHS_EVALUATIONS", 50)
    stalled = run("sphere", [0.1], FIXED)
    json.dumps(stalled)
    assert stalled["evidence"]["verified"] is False and any("evaluaciones" in w for w in stalled["evidence"]["reasons"])
    lam = [{"thickness_m": 1e-4, "diffusivity_m2_s": 1e-10, "solubility_mol_m3_Pa": 1e-5}]
    mem = T.membrane_permeation(layers=lam, geometry="flat", upstream_partial_pressure_Pa=1e5, downstream_partial_pressure_Pa=0.0,
                                transient=True, initial_partial_pressure_Pa=0.0)
    assert mem["evidence"]["verified"] is False and any("evaluaciones" in w for w in mem["evidence"]["reasons"])
    with pytest.raises(ValueError, match="n_cells"):
        run("sphere", [0.1], FIXED, n_cells=5000)
    with pytest.raises(ValueError, match="demasiados instantes"):
        run("sphere", list(np.linspace(0.01, 1.0, 201)), FIXED)


# ---------------------------------------------------------------------------------------------------------------------
# physics identities
def test_symmetry_scaling_and_ordering_identities():
    two = T.transient_diffusion_solid(geometry="slab", thickness_m=4e-3, exposed_faces=2, initial_concentration_mol_m3=5.0,
                                      diffusivity_m2_s=2e-10, times_s=[3000.0], target_fractional_uptake=0.9, **FIXED)
    one = T.transient_diffusion_solid(geometry="slab", thickness_m=2e-3, exposed_faces=1, initial_concentration_mol_m3=5.0,
                                      diffusivity_m2_s=2e-10, times_s=[3000.0], target_fractional_uptake=0.9, **FIXED)
    assert two["results"][0]["fractional_uptake"] == pytest.approx(one["results"][0]["fractional_uptake"], abs=1e-12)
    # desorption mirrors sorption (linear problem): same fraction, opposite flux sign
    des = T.transient_diffusion_solid(geometry="slab", thickness_m=4e-3, exposed_faces=2, initial_concentration_mol_m3=1.0,
                                      diffusivity_m2_s=2e-10, times_s=[3000.0], surface_condition="fixed_concentration",
                                      surface_concentration_mol_m3=5.0)
    assert des["results"][0]["fractional_uptake"] == pytest.approx(two["results"][0]["fractional_uptake"], abs=1e-9)
    assert two["results"][0]["surface_flux_mol_m2_s"] < 0 < des["results"][0]["surface_flux_mol_m2_s"]
    assert two["results"][0]["average_concentration_mol_m3"] < 5.0
    # time scales with a²/D
    big = T.transient_diffusion_solid(geometry="slab", thickness_m=8e-3, exposed_faces=2, initial_concentration_mol_m3=5.0,
                                      diffusivity_m2_s=1e-10, target_fractional_uptake=0.9, **FIXED)
    assert big["time_to_target"]["time_s"] / two["time_to_target"]["time_s"] == pytest.approx(8.0, rel=1e-4)
    assert two["time_to_target"]["fourier_number"] == pytest.approx(0.8481, abs=2e-4)     # Fo for 90 % in a slab (series)
    assert abs(ref_uptake_fixed(0, two["time_to_target"]["fourier_number"]) - 0.9) < 2e-5
    # at equal diffusion path: sphere faster than cylinder faster than slab; uptake monotone in time and within [0, 1]
    fr = {g: [x["fractional_uptake"] for x in run(g, FO, FIXED)["results"]] for g in GEOS}
    for i in range(len(FO)):
        assert fr["slab"][i] < fr["cylinder"][i] < fr["sphere"][i] <= 1.0
    for g in GEOS:
        assert all(b > a > 0 for a, b in zip(fr[g], fr[g][1:]))


def test_biot_limits_lumped_and_fixed_surface():
    # Bi → ∞ recovers the fixed surface; Bi → 0 recovers the lumped balance 1 − exp(−(m+1)·Bi·Fo)
    for geo, (m, _) in GEOS.items():
        huge = run(geo, [0.1], conv(1e7))["results"][0]["fractional_uptake"]
        assert abs(huge - ref_uptake_fixed(m, 0.1)) < 5e-5
        small = run(geo, [20.0], conv(0.01))
        lumped = 1.0 - math.exp(-(m + 1) * 0.01 * 20.0)
        assert abs(small["results"][0]["fractional_uptake"] - lumped) < 2e-3        # O(Bi) correction; measured ≤ 5e-4
        assert any("Biot" in w for w in small["warnings"])
    # the partition coefficient enters as Bi = k_c·a/(K·D) and sets the equilibrium c_eq = K·c_fluid
    r = run("sphere", [0.3], conv(2.0, cf=3.0, kp=4.0))
    assert r["biot_number_mass"] == pytest.approx(2.0) and r["equilibrium_concentration_mol_m3"] == pytest.approx(12.0)
    assert abs(r["results"][0]["fractional_uptake"] - ref_uptake_conv(2, 2.0, 0.3)) < 3e-5
    # a bath much larger than the solid behaves as a fixed surface at K·C_bath
    bath = run("sphere", [0.05], {"surface_condition": "limited_bath", "fluid_concentration_mol_m3": 2.0, "partition_coefficient": 1.5,
                                  "bath_volume_m3": 1e6, "solid_volume_m3": 1.0})
    assert abs(bath["results"][0]["fractional_uptake"] - ref_uptake_fixed(2, 0.05)) < 5e-5
    assert bath["equilibrium_concentration_mol_m3"] == pytest.approx(3.0, rel=1e-5)


def test_bath_with_film_conserves_and_is_slower_than_without():
    kw = {"surface_condition": "limited_bath", "fluid_concentration_mol_m3": 5.0, "partition_coefficient": 2.0,
          "bath_volume_m3": 0.6, "solid_volume_m3": 0.3}
    no_film = run("sphere", [0.02, 0.2], kw)
    film = run("sphere", [0.02, 0.2], dict(kw, mass_transfer_coefficient_m_s=2.0))
    for a, b in zip(film["results"], no_film["results"]):
        assert a["fractional_uptake"] < b["fractional_uptake"]
    assert film["evidence"]["verified"] and film["evidence"]["analytical"]["applicable"] is False
    assert film["evidence"]["bath_balance_max_rel_error"] < 1e-10


# ---------------------------------------------------------------------------------------------------------------------
# D from uptake data
UPTAKE_T = [60, 120, 300, 600, 1200, 2400, 3600, 7200, 14400, 28800, 57600, 115200.0]


def test_fit_recovers_diffusivity_from_series_data_with_noise():
    d, thick = 3e-12, 1e-3
    clean = np.array([ref_uptake_fixed(0, d * t / (thick / 2) ** 2, n=4000) for t in UPTAKE_T])
    r = T.diffusivity_from_uptake(geometry="slab", times_s=UPTAKE_T, uptake=clean.tolist(), uptake_kind="fractional",
                                  thickness_m=thick, exposed_faces=2)
    json.dumps(r)
    assert abs(r["diffusivity_m2_s"] / d - 1) < 1e-6 and r["evidence"]["verified"] and r["warnings"] == []
    assert abs(r["cross_checks"]["initial_slope_method"]["diffusivity_m2_s"] / d - 1) < 0.01
    assert abs(r["cross_checks"]["half_time_method"]["fourier_number_at_half_uptake"] - 0.19674) < 1e-4   # slab: 0.049·L²/t½
    rng = np.random.default_rng(7)
    inside, widths = 0, []
    for sigma in (0.005, 0.02, 0.02, 0.02):
        y = 2.5 * clean + rng.normal(0.0, 2.5 * sigma, clean.size)
        f = T.diffusivity_from_uptake(geometry="slab", times_s=UPTAKE_T, uptake=y.tolist(), uptake_kind="amount",
                                      thickness_m=thick, exposed_faces=2, fit_equilibrium_uptake=True)
        lo, hi = f["diffusivity_confidence_interval_m2_s"]
        inside += lo <= d <= hi
        widths.append(hi / lo)
        assert abs(f["diffusivity_m2_s"] / d - 1) < 0.15 and abs(f["equilibrium_uptake"] / 2.5 - 1) < 0.05
        assert f["evidence"]["verified"] and f["fit"]["r_squared"] > 0.99
        assert f["evidence"]["series_model_vs_finite_volume_engine_max_abs_diff"] < 5e-5
    assert inside >= 3 and widths[0] < min(widths[1:])                 # less noise → narrower interval


def test_fit_with_film_sphere_and_non_fickian_warning():
    d, a, kc, kp = 3e-12, 5e-4, 1e-7, 2.0
    bi = kc * a / (kp * d)
    clean = np.array([ref_uptake_conv(2, bi, d * t / a ** 2, n=400) for t in UPTAKE_T])
    r = T.diffusivity_from_uptake(geometry="sphere", times_s=UPTAKE_T, uptake=clean.tolist(), uptake_kind="fractional",
                                  radius_m=a, mass_transfer_coefficient_m_s=kc, partition_coefficient=kp)
    assert abs(r["diffusivity_m2_s"] / d - 1) < 1e-4 and r["biot_number_mass_at_fitted_D"] == pytest.approx(bi, rel=1e-4)
    # ignoring the film (fitting a fixed-surface model to film-limited data) leaves structured residuals and says so
    t = np.linspace(200, 60000, 30)
    y = np.array([ref_uptake_conv(2, 0.5, d * ti / a ** 2, n=200) for ti in t])
    bad = T.diffusivity_from_uptake(geometry="sphere", times_s=t.tolist(), uptake=y.tolist(), uptake_kind="fractional", radius_m=a)
    assert bad["diffusivity_m2_s"] < 0.5 * d                      # an apparent D far below the true one
    assert bad["fit"]["lag1_autocorrelation"] > 0.5 and any("autocorrelacionados" in w for w in bad["warnings"])


# ---------------------------------------------------------------------------------------------------------------------
# refusals (nothing physical is defaulted) and wrappers
SOLID_OK = dict(geometry="sphere", radius_m=1e-3, initial_concentration_mol_m3=0.0, diffusivity_m2_s=1e-10,
                surface_condition="convective", fluid_concentration_mol_m3=1.0, mass_transfer_coefficient_m_s=1e-6,
                partition_coefficient=1.0, times_s=[10.0])


@pytest.mark.parametrize("drop, word", [("geometry", "geometry"), ("radius_m", "radius_m"),
                                        ("initial_concentration_mol_m3", "initial_concentration_mol_m3"),
                                        ("diffusivity_m2_s", "difusividad"), ("surface_condition", "surface_condition"),
                                        ("fluid_concentration_mol_m3", "fluid_concentration_mol_m3"),
                                        ("mass_transfer_coefficient_m_s", "mass_transfer_coefficient_m_s"),
                                        ("partition_coefficient", "partition_coefficient"), ("times_s", "times_s")])
def test_transient_refuses_missing_inputs(drop, word):
    kw = {k: v for k, v in SOLID_OK.items() if k != drop}
    with pytest.raises(ValueError, match=word):
        T.transient_diffusion_solid(**kw)


def test_other_refusals_are_spanish_and_specific():
    with pytest.raises(ValueError, match="exposed_faces"):
        T.transient_diffusion_solid(**dict(SOLID_OK, geometry="slab", thickness_m=1e-3))
    with pytest.raises(ValueError, match="equilibrio"):
        T.transient_diffusion_solid(**dict(SOLID_OK, initial_concentration_mol_m3=1.0))
    with pytest.raises(ValueError, match="bath_volume_m3"):
        T.transient_diffusion_solid(**dict(SOLID_OK, surface_condition="limited_bath"))
    with pytest.raises(ValueError, match="no ambos"):
        T.transient_diffusion_solid(**dict(SOLID_OK, diffusivity_law={"type": "linear", "D0_m2_s": 1e-10, "alpha_m3_mol": 0.1}))
    with pytest.raises(ValueError, match="D constante"):
        T.transient_diffusion_solid(geometry="block", thickness_m=1, width_m=1, depth_m=1, initial_concentration_mol_m3=0.0,
                                    diffusivity_law={"type": "linear", "D0_m2_s": 1e-10, "alpha_m3_mol": 0.1}, times_s=[1.0], **FIXED)
    with pytest.raises(ValueError, match="semi-infinito"):
        T.transient_diffusion_solid(**dict(SOLID_OK, geometry="semi_infinite", target_fractional_uptake=0.5))
    layer = {"thickness_m": 1e-4, "diffusivity_m2_s": 1e-10, "solubility_mol_m3_Pa": 1e-5}
    ok = dict(layers=[layer], geometry="flat", upstream_partial_pressure_Pa=1e5, downstream_partial_pressure_Pa=0.0)
    for kw, word in ((dict(ok, layers=None), "layers"), (dict(ok, geometry=None), "geometry"),
                     (dict(ok, downstream_partial_pressure_Pa=None), "downstream_partial_pressure_Pa"),
                     (dict(ok, upstream_concentration_mol_m3=1.0), "UNA forma"),
                     (dict(ok, layers=[{"thickness_m": 1e-4, "diffusivity_m2_s": 1e-10}]), "faltan propiedades"),
                     (dict(ok, layers=[{"diffusivity_m2_s": 1e-10, "solubility_mol_m3_Pa": 1e-5}]), "thickness_m"),
                     (dict(ok, transient=True), "initial_partial_pressure_Pa"),
                     (dict(ok, layers=[{"thickness_m": 1e-4, "permeability_barrer": 1.0}], transient=True,
                           initial_partial_pressure_Pa=0.0), "difusividad de CADA capa"),
                     (dict(ok, geometry="cylindrical", inner_radius_m=0.01), "upstream_side"),
                     (dict(ok, geometry="spherical", upstream_side="inner"), "inner_radius_m"),
                     (dict(ok, layers=[dict(layer, permeability_mol_m_m2_s_Pa=1.0)]), "no coincide")):
        with pytest.raises(ValueError, match=word):
            T.membrane_permeation(**kw)
    fit = dict(geometry="slab", times_s=[1, 2, 3, 4, 5.0], uptake=[0.1, 0.14, 0.17, 0.2, 0.22], uptake_kind="fractional",
               thickness_m=1e-3, exposed_faces=2)
    for kw, word in ((dict(fit, uptake_kind=None), "uptake_kind"), (dict(fit, thickness_m=None), "thickness_m"),
                     (dict(fit, exposed_faces=None), "exposed_faces"), (dict(fit, uptake_kind="amount"), "equilibrium_uptake"),
                     (dict(fit, uptake=[0.1, 0.2]), "misma longitud"), (dict(fit, geometry=None), "geometry"),
                     (dict(fit, times_s=[1, 2.0], uptake=[0.1, 0.2]), "al menos"),
                     (dict(fit, mass_transfer_coefficient_m_s=1e-6), "partition_coefficient")):
        with pytest.raises(ValueError, match=word):
            T.diffusivity_from_uptake(**kw)


def test_wrappers_mirror_tools_and_return_json():
    assert [w.name for w in W.AGENT_WRAPPERS] == [f.__name__ for f in T.SOLID_DIFFUSION_TOOLS]
    out = json.loads(W.transient_diffusion_solid.call(dict(SOLID_OK, target_fractional_uptake=0.5)))
    assert out["evidence"]["verified"] is True and out["time_to_target"]["time_s"] > 0
    assert json.loads(W.membrane_permeation.call({"layers": [{"thickness_m": 1e-4, "permeability_barrer": 5.0}], "geometry": "flat",
                                                  "upstream_partial_pressure_Pa": 1e5, "downstream_partial_pressure_Pa": 0.0}))["flux_mol_m2_s"] > 0
    fit = json.loads(W.diffusivity_from_uptake.call({"geometry": "slab", "times_s": UPTAKE_T, "uptake_kind": "fractional",
                                                     "uptake": [ref_uptake_fixed(0, 3e-12 * t / 2.5e-7, n=4000) for t in UPTAKE_T],
                                                     "thickness_m": 1e-3, "exposed_faces": 2}))
    assert fit["diffusivity_m2_s"] == pytest.approx(3e-12, rel=1e-6)
    for w in W.AGENT_WRAPPERS:
        assert len(w.description.strip().splitlines()) <= 12 and w.input_schema["type"] == "object"
