"""Wall effect on a sphere falling along the axis of a tube and Stokes drag split of `particle_settling`.

Published values used (read 2026-11-20):
  * «Studies on falling ball viscometry», arXiv:1202.1400, Table 9: wall factor u_wall/u_unbounded in creeping flow at
    d/D = 0.1 ... 0.6 by the exact theory (0.792, 0.595, 0.422, 0.278, 0.168, 0.090), by the Haberman-Sayre equation
    (0.792, 0.595, 0.422, 0.279, 0.170, 0.094) and by Di Felice at Re -> 0 (0.789, 0.600, 0.435, 0.296, 0.184, 0.101);
    Eqs. 18 and 20-22 give the formulas.
  * Braz. J. Chem. Eng. 16(4), 1999 (doi 10.1590/S0104-66321999000400007), Eqs. 6-7: Munroe f = 1 - b^1.5 (Re > 1000),
    Francis f = ((1 - b)/(1 - 0.475 b))^4 (Re < 0.1, b < 0.8)."""
import json
import math

import pytest

from packages.dwsim_client import tools_particles as TP
from packages.transport import particles as PT

LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
HS_TABLE = (0.792, 0.595, 0.422, 0.279, 0.170, 0.094)
EXACT_TABLE = (0.792, 0.595, 0.422, 0.278, 0.168, 0.090)
DI_FELICE_TABLE = (0.789, 0.600, 0.435, 0.296, 0.184, 0.101)


def test_haberman_sayre_against_published_table():
    for lam, hs, exact in zip(LAMBDAS, HS_TABLE, EXACT_TABLE):
        f = PT.wall_factor_haberman_sayre(lam)
        assert f == pytest.approx(hs, abs=6e-4)                 # table printed to 3 decimals
        assert f == pytest.approx(exact, abs=5e-3)              # and within 0.005 of the exact theory up to 0.6
    # formula written out independently at d/D = 0.3
    lam = 0.3
    K = (1 - 0.75857 * lam ** 5) / (1 - 2.1050 * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5 + 0.72603 * lam ** 6)
    assert PT.wall_factor_haberman_sayre(lam) == pytest.approx(1 / K, rel=1e-14)
    assert PT.wall_factor_haberman_sayre(0.0) == 1.0
    # leading term is Ladenburg/Faxen: 1 - 2.105 d/D
    assert PT.wall_factor_haberman_sayre(1e-4) == pytest.approx(1 - 2.105e-4, rel=1e-7)


def test_di_felice_francis_munroe_formulas():
    for lam, d in zip(LAMBDAS, DI_FELICE_TABLE):
        assert PT.wall_factor_di_felice(lam, 0.0) == pytest.approx(d, abs=6e-4)
    assert PT.wall_factor_di_felice(0.4, 0.0) == pytest.approx((0.6 / (1 - 0.132)) ** 3.3, rel=1e-14)
    assert PT.wall_factor_di_felice(0.4, 1e9) == pytest.approx((0.6 / (1 - 0.132)) ** 0.85, rel=1e-6)
    # (3.3 - alpha)/(alpha - 0.85) = 0.1 Re at Re = 100 -> alpha = 0.80 in the source's words ("0.80 at Re = 100")
    alpha = math.log(PT.wall_factor_di_felice(0.4, 100.0)) / math.log(0.6 / (1 - 0.132))
    assert (3.3 - alpha) / (alpha - 0.85) == pytest.approx(10.0, rel=1e-9)
    assert PT.wall_factor_francis(0.2) == pytest.approx((0.8 / 0.905) ** 4, rel=1e-14)
    assert PT.wall_factor_munroe(0.25) == pytest.approx(0.875, rel=1e-14)


def test_creeping_sphere_in_tube_uses_haberman_sayre():
    """Steel ball 1 mm in glycerol-like liquid, tube 10 mm: Re << 0.1, d/D = 0.1 -> u = 0.792 u_inf."""
    r = TP.particle_settling(1e-3, 7800.0, 1260.0, 1.4, 1.0, column_diameter_m=0.01)
    w = r["wall_effect"]
    assert r["reynolds_terminal"] < 0.1 and w["correlation_used"] == "Haberman_Sayre_1958" and not w["is_estimate"]
    assert w["wall_factor"] == pytest.approx(0.792, abs=6e-4)
    assert r["terminal_velocity_with_wall_m_s"] == pytest.approx(0.792 * r["terminal_velocity_m_s"], rel=1e-3)
    # the unbounded value is untouched: Stokes, g d^2 (rho_p - rho)/(18 mu)
    assert r["terminal_velocity_m_s"] == pytest.approx(9.80665 * 1e-6 * (7800 - 1260) / (18 * 1.4), rel=1e-3)
    assert w["wall_factor_by_correlation"]["Francis_1933"]["applicable"]
    assert not w["wall_factor_by_correlation"]["Munroe_1888"]["applicable"]
    assert r["evidence"]["verified"] and any("1202.1400" in s for s in r["evidence"]["wall_effect_sources"])
    assert r["evidence"]["checks"]["wall_haberman_sayre_vs_exact_theory_table_max_abs"] < 5e-3
    json.dumps(r, ensure_ascii=False)


def test_wall_effect_outside_creeping_flow_is_a_flagged_estimate():
    r = PT.settling(5e-3, 7800.0, 1000.0, 1e-3, 1.0, column_diameter_m=0.02)
    w = r["wall_effect"]
    Re = r["reynolds_terminal"]
    assert Re > 1000 and w["correlation_used"] == "Di_Felice_1996" and w["is_estimate"]
    alpha = (3.3 + 0.085 * Re) / (1 + 0.1 * Re)
    assert w["wall_factor"] == pytest.approx((0.75 / (1 - 0.33 * 0.25)) ** alpha, rel=1e-12)
    assert w["wall_factor_by_correlation"]["Munroe_1888"]["wall_factor"] == pytest.approx(1 - 0.25 ** 1.5)
    assert not w["wall_factor_by_correlation"]["Haberman_Sayre_1958"]["applicable"]
    assert "ESTIMACIÓN" in " ".join(r["warnings"])


def test_wall_effect_limits_and_refusals():
    # d/D >= 0.8 in creeping flow: no verified correlation -> no corrected velocity, and it says so
    r = PT.settling(1e-3, 7800.0, 1260.0, 1.4, 1.0, column_diameter_m=1.2e-3)
    assert r["terminal_velocity_with_wall_m_s"] is None and "0.8" in " ".join(r["warnings"])
    with pytest.raises(ValueError, match="no cabe"):
        PT.settling(1e-3, 7800.0, 1260.0, 1.4, 1.0, column_diameter_m=0.9e-3)
    # non-spherical particle: wall effect not computed, stated
    ns = PT.settling(5e-4, 2650.0, 998.0, 1e-3, 0.7, projected_area_ratio=1.1, column_diameter_m=0.01)
    assert "wall_effect" not in ns and "solo para esferas" in " ".join(ns["warnings"])
    # without the tube diameter nothing changes
    assert "wall_effect" not in PT.settling(1e-3, 7800.0, 1260.0, 1.4, 1.0)


def test_stokes_drag_split():
    """Exact Stokes solution: F = 3 pi mu d U = pi mu d U (form) + 2 pi mu d U (friction); it balances the net weight."""
    d, rho_p, rho, mu = 2.0e-5, 2650.0, 998.0, 1.0e-3
    r = TP.particle_settling(d, rho_p, rho, mu, 1.0)
    s = r["drag_split"]
    U = r["terminal_velocity_m_s"]
    assert r["reynolds_terminal"] < 0.1 and s["available"] and s["reynolds_limit_used"] == 0.1
    assert s["form_drag_N"] == pytest.approx(math.pi * mu * d * U, rel=1e-12)
    assert s["friction_drag_N"] == pytest.approx(2 * math.pi * mu * d * U, rel=1e-12)
    assert s["form_drag_N"] + s["friction_drag_N"] == pytest.approx(3 * math.pi * mu * d * U, rel=1e-12)
    assert s["form_fraction"] == pytest.approx(1 / 3) and s["friction_fraction"] == pytest.approx(2 / 3)
    weight = math.pi / 6 * d ** 3 * (rho_p - rho) * 9.80665
    assert s["stokes_drag_N"] == pytest.approx(weight, rel=2e-2)          # Re < 0.1: Stokes within 2 %
    big = TP.particle_settling(1e-3, rho_p, rho, mu, 1.0)["drag_split"]
    assert not big["available"] and "campo de flujo" in big["reason"]
    ns = TP.particle_settling(2e-5, rho_p, rho, mu, 0.8, projected_area_ratio=1.0)["drag_split"]
    assert not ns["available"]


def test_wrapper_returns_wall_effect():
    from packages.agent import wrappers_particles as W
    out = json.loads(W.AGENT_WRAPPERS[1](particle_diameter_m=1e-3, particle_density_kg_m3=7800.0,
                                         fluid_density_kg_m3=1260.0, fluid_viscosity_Pa_s=1.4, sphericity=1.0,
                                         column_diameter_m=0.005))
    assert out["wall_effect"]["wall_factor"] == pytest.approx(0.595, abs=6e-4) and out["drag_split"]["available"]
