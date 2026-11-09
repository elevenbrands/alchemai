"""Shell side, boiling and condensation (bloque M4b): pool boiling against Incropera's Example 10.1 (water at 1 atm,
polished copper), Cooper, Kern and Nusselt against their printed equations written here, and the regime limits that
switch the seal off."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_heat as TH  # noqa: E402

G = 9.80665
# saturated water at 1 atm, Incropera table A.6 as used in Example 10.1 (k from Pr = 1.76)
WATER = dict(T_sat_K=373.15, liquid_density_kg_m3=957.9, vapor_density_kg_m3=0.5955, liquid_viscosity_Pa_s=279e-6,
             liquid_cp_J_kgK=4217.0, latent_heat_J_kg=2257e3)
K_L = 4217.0 * 279e-6 / 1.76
BOIL = dict(P_Pa=101325.0, critical_pressure_Pa=22.064e6, MW=18.015, liquid_conductivity_W_mK=K_L,
            surface_tension_N_m=58.9e-3, **WATER)


def test_pool_boiling_reproduces_incropera_example_10_1():
    r = TH.nucleate_boiling(**BOIL, wall_superheat_K=10.0, rohsenow_Csf=0.0128, rohsenow_n=1.0)
    assert r["h_by_method_W_m2K"]["Rohsenow"] * 10.0 == pytest.approx(1.44e5, rel=0.01)     # q''s = 1.44·10⁵ W/m²
    assert r["q_max_W_m2"]["Lienhard_Dhir_K0.149"] == pytest.approx(1.26e6, rel=0.005)      # q''max = 1.26 MW/m²
    assert r["recommended_method"] == "Cooper" and r["evidence"]["verified"]
    assert len(r["h_by_method_W_m2K"]) == 4 and r["method_spread_pct"] > 0


def test_cooper_closed_form_and_film_boiling_flag():
    q = 1.0e5
    r = TH.nucleate_boiling(**BOIL, heat_flux_W_m2=q, surface_roughness_um=1.0)
    pr = 101325.0 / 22.064e6
    h = 55 * pr ** 0.12 * (-math.log10(pr)) ** -0.55 * 18.015 ** -0.5 * q ** 0.67              # Cooper 1984, Rp = 1 μm
    assert r["h_W_m2K"] == pytest.approx(h, rel=1e-12) and r["wall_superheat_K"] == pytest.approx(q / h)
    hot = TH.nucleate_boiling(**BOIL, heat_flux_W_m2=2.0e6)
    assert not hot["evidence"]["verified"] and "película" in hot["evidence"]["reasons"][0]
    with pytest.raises(ValueError, match="exactly one"):
        TH.nucleate_boiling(**BOIL)


def test_kern_against_its_printed_equation():
    m, rho, mu, k, cp, mu_w = 11.0, 995.0, 0.000803, 0.62, 4180.0, 0.000657
    Ds, B, Do, Pt = 0.584, 0.1524, 0.019, 0.0254
    r = TH.shell_side_kern(m, rho, mu, k, cp, Ds, B, 22, Do, Pt, "triangular", mu_w)
    De = 4 * (math.sqrt(3) / 4 * Pt ** 2 - math.pi * Do ** 2 / 8) / (math.pi * Do / 2)
    Re = De * m / (Ds * (Pt - Do) * B / Pt) / mu
    h = 0.36 * k / De * Re ** 0.55 * (cp * mu / k) ** (1 / 3) * (mu / mu_w) ** 0.14
    assert r["h_W_m2K"] == pytest.approx(h, rel=1e-12) and r["evidence"]["verified"]
    assert 0.3 < r["kern_over_ideal_bank"] < 1.0                     # below the no-leakage ideal bank
    sq = TH.shell_side_kern(m, rho, mu, k, cp, Ds, B, 22, Do, Pt, "square", mu_w)
    assert sq["equivalent_diameter_m"] > r["equivalent_diameter_m"]
    tight = TH.shell_side_kern(m, rho, mu, k, cp, Ds, B, 22, Do, 1.2 * Do, "triangular", mu_w)
    assert not tight["evidence"]["verified"] and "1.25" in tight["evidence"]["reasons"][0]


def test_nusselt_vertical_and_horizontal():
    dT, L, D = 6.0, 0.3, 0.019
    kw = dict(liquid_conductivity_W_mK=0.68, wall_T_K=373.15 - dT, **WATER)
    hfg = 2257e3 * (1 + 0.68 * 4217.0 * dT / 2257e3)
    grp = G * 957.9 * (957.9 - 0.5955) * 0.68 ** 3 * hfg / 279e-6
    v = TH.film_condensation("vertical_plate", length_m=L, **kw)
    assert v["h_W_m2K"] == pytest.approx(0.943 * (grp / (dT * L)) ** 0.25, rel=1e-3)
    assert v["evidence"]["formula_match"]
    h1 = TH.film_condensation("horizontal_tube", tube_od_m=D, **kw)
    h10 = TH.film_condensation("horizontal_tube", tube_od_m=D, n_rows=10, **kw)
    assert h1["h_W_m2K"] == pytest.approx(0.729 * (grp / (dT * D)) ** 0.25, rel=1e-12)
    assert h10["h_W_m2K"] == pytest.approx(h1["h_W_m2K"] * 10 ** -0.25, rel=1e-12) and h10["evidence"]["verified"]
    tall = TH.film_condensation("vertical_plate", length_m=1.0, **kw)            # long plate: wavy film
    assert tall["film_regime"] == "laminar ondulado" and not tall["evidence"]["verified"]
    with pytest.raises(ValueError, match="below T_sat"):
        TH.film_condensation("vertical_plate", length_m=L, **{**kw, "wall_T_K": 380.0})


def test_shah_in_tube():
    r = TH.film_condensation("in_tube", mass_flow_kg_s=0.02, quality=0.5, tube_id_m=0.016, P_Pa=101325.0,
                             critical_pressure_Pa=22.064e6, liquid_conductivity_W_mK=0.68, **WATER)
    Re = 0.02 / (math.pi * 0.016 ** 2 / 4) * 0.016 / 279e-6
    h_lo = 0.023 * Re ** 0.8 * (4217.0 * 279e-6 / 0.68) ** 0.4 * 0.68 / 0.016
    pr = 101325.0 / 22.064e6
    assert r["h_W_m2K"] == pytest.approx(h_lo * (0.5 ** 0.8 + 3.8 * 0.5 ** 0.76 * 0.5 ** 0.04 / pr ** 0.38), rel=1e-6)
    assert r["evidence"]["verified"] and set(r["h_by_method_W_m2K"]) == {"Shah", "Boyko_Kruzhilin"}


def test_results_are_plain_json():
    json.dumps(TH.nucleate_boiling(**BOIL, wall_superheat_K=10.0), allow_nan=False)
    json.dumps(TH.shell_side_kern(11.0, 995.0, 0.000803, 0.62, 4180.0, 0.584, 0.1524, 22, 0.019, 0.0254), allow_nan=False)
    json.dumps(TH.film_condensation("horizontal_tube", tube_od_m=0.019, liquid_conductivity_W_mK=0.68,
                                    wall_T_K=367.15, **WATER), allow_nan=False)
