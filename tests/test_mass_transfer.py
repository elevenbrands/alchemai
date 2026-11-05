"""Mass transfer (bloque 22): Kremser/Colburn identities, Onda as published, and Kremser vs DWSIM's rigorous absorber."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.transport.mass_transfer import colburn_NOG, fuller, kremser_fraction_absorbed, kremser_stages, onda, wilke_chang  # noqa: E402


def test_kremser_round_trip_and_limits():
    for A in (0.6, 1.0, 1.5, 3.0):
        f = kremser_fraction_absorbed(A, 7)
        assert kremser_stages(A, 0.02, 0.02 * (1 - f), 1.0) == pytest.approx(7, rel=1e-9)
    assert kremser_fraction_absorbed(0.6, 200) == pytest.approx(0.6, rel=1e-9)        # A < 1 caps recovery at A
    with pytest.raises(ValueError):
        kremser_stages(0.6, 0.02, 0.02 * 0.3, 1.0)                                   # 70 % > 60 % cap


def test_colburn_equals_numeric_integral_and_hetp_identity():
    m, A, yin, yout = 0.8, 1.4, 0.02, 0.002
    L_over_G = A * m
    n = 200000
    ys = [yout + (yin - yout) * i / n for i in range(n + 1)]
    f = [1.0 / (y - m * (y - yout) / L_over_G) for y in ys]
    integral = (yin - yout) / n * (sum(f) - 0.5 * (f[0] + f[-1]))
    assert colburn_NOG(yin, yout, m, A) == pytest.approx(integral, rel=1e-8)
    # straight lines: N_OG / N_stages = ln(A) / (1 − 1/A)
    assert colburn_NOG(yin, yout, m, A) / kremser_stages(A, yin, yout, m) == pytest.approx(math.log(A) / (1 - 1 / A), rel=1e-9)


def test_onda_constants_and_small_packing_switch():
    base = dict(L_kg_m2s=10.0, G_kg_m2s=1.0, rho_L=1000.0, mu_L=1e-3, sigma_L=0.072, D_L=2e-9, rho_G=1.2, mu_G=1.8e-5,
                D_G=1.6e-5, T_K=298.15, a_t=190.0, sigma_c=0.061)
    big, small = onda(Dp=0.025, **base), onda(Dp=0.012, **base)
    assert big["constant_gas"] == 5.23 and small["constant_gas"] == 2.00
    assert 0 < big["wetted_fraction"] < 1
    # kL from Eq. (2) rearranged by hand
    a_w = big["a_w_m2_m3"]
    kL = 0.0051 * (10.0 / (a_w * 1e-3)) ** (2 / 3) * (1e-3 / (1000 * 2e-9)) ** -0.5 * (190 * 0.025) ** 0.4 * (1e-3 * 9.80665 / 1000) ** (1 / 3)
    assert big["kL_m_s"] == pytest.approx(kL, rel=1e-12)


def test_diffusivity_forms():
    assert wilke_chang(298.15, 0.89, 18.015, 2.6, 34.0) == pytest.approx(7.4e-8 * math.sqrt(2.6 * 18.015) * 298.15 / (0.89 * 34.0 ** 0.6) * 1e-4)
    d1, d2 = fuller(300, 101325, 44.1, 16.0, 65.3, 24.4), fuller(300, 2 * 101325, 44.1, 16.0, 65.3, 24.4)
    assert d1 == pytest.approx(2 * d2) and 1e-6 < d1 < 1e-4


def test_kremser_matches_dwsim_absorber():
    spec = importlib.util.spec_from_file_location("nb16", ROOT / "notebooks" / "16_transferencia_de_masa.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    K = nb.k_propane(120.0)
    c = nb.column(120.0, 6)
    A = 0.5 * (120.0 + c["L_bot"]) / (K * 0.5 * (100.0 + c["V_top"]))
    assert c["ok"] and kremser_fraction_absorbed(A, 6) == pytest.approx(c["frac_dwsim"], abs=0.01)
