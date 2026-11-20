"""Slit (parallel plates) geometry, Carreau-type rheology, inverse problem and viscosity crossing of
`non_newtonian_pipe_flow`. Every expected value is an exact closed form written out in the test or a number computed
here independently (scipy brentq/quad), never the tool's own output."""
import json
import math

import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from packages.dwsim_client import tools_particles as TP
from packages.transport import non_newtonian as NN

NEWT = {"model": "newtonian", "viscosity_Pa_s": 0.25}


def pl(K, n):
    return {"model": "power_law", "K_Pa_sn": K, "n": n}


def carreau(eta0, lam, n, eta_inf=0.0):
    return {"model": "carreau", "eta0_Pa_s": eta0, "eta_inf_Pa_s": eta_inf, "lambda_s": lam, "n": n}


# ------------------------------------------------------------------ slit, exact solutions
def test_slit_newtonian_exact():
    """Plane Poiseuille flow: u_max = 1.5 u_mean, wall shear rate 6 u_mean/h, -dp/dz = 12 mu u_mean/h^2,
    u(y) = u_max (1 - (2y/h)^2)."""
    h, W, Q, mu, L = 0.004, 0.5, 2.0e-4, 0.25, 3.0
    V = Q / (h * W)
    r = TP.non_newtonian_pipe_flow(Q, None, L, 1100.0, NEWT, geometry="slit", gap_m=h, width_m=W,
                                   positions_m=[0.0, 0.001, 0.002])
    assert r["evidence"]["verified"] and r["regime"] == "laminar"
    assert r["mean_velocity_m_s"] == pytest.approx(V, rel=1e-12)
    assert r["max_to_mean"] == pytest.approx(1.5, rel=1e-10)
    assert r["wall_shear_rate_1_s"] == pytest.approx(6 * V / h, rel=1e-10)
    assert r["wall_shear_stress_Pa"] == pytest.approx(6 * mu * V / h, rel=1e-10)
    assert r["pressure_gradient_Pa_m"] == pytest.approx(12 * mu * V / h ** 2, rel=1e-10)
    assert r["pressure_drop_Pa"] == pytest.approx(12 * mu * V / h ** 2 * L, rel=1e-10)
    for row in r["velocity_at_positions"]:
        assert row["u_m_s"] == pytest.approx(1.5 * V * (1 - (2 * row["y_m"] / h) ** 2), rel=1e-10, abs=1e-15)
    # Reynolds on the hydraulic diameter 2h, f_Fanning = 24/Re
    assert r["generalized_reynolds_slit_hydraulic_diameter"] == pytest.approx(1100.0 * V * 2 * h / mu, rel=1e-10)
    assert r["f_fanning"] == pytest.approx(24 / (1100.0 * V * 2 * h / mu), rel=1e-10)


def test_slit_per_unit_width_and_inverse_roundtrip():
    h = 0.004
    a = TP.non_newtonian_pipe_flow(4.0e-4, rheology=NEWT, geometry="slit", gap_m=h)          # m2/s per unit width
    assert a["flow_per_unit_width_m2_s"] == pytest.approx(4.0e-4) and "volumetric_flow_m3_s" not in a
    assert "sin density_kg_m3" in " ".join(a["warnings"])
    b = TP.non_newtonian_pipe_flow(rheology=NEWT, geometry="slit", gap_m=h,
                                   pressure_gradient_Pa_m=a["pressure_gradient_Pa_m"])
    assert b["flow_per_unit_width_m2_s"] == pytest.approx(4.0e-4, rel=1e-10)
    # exact: q = h^3 G / (12 mu)
    assert b["flow_per_unit_width_m2_s"] == pytest.approx(h ** 3 * a["pressure_gradient_Pa_m"] / (12 * 0.25), rel=1e-10)


@pytest.mark.parametrize("n", [0.35, 0.7, 1.4])
def test_slit_power_law_closed_form(n):
    """Power law between plates: V = n/(2n+1) (tau_w/K)^(1/n) h/2, u_max/V = (2n+1)/(n+1),
    wall shear rate = (2n+1)/n · 2V/h, u(y)/u_max = 1 - (2y/h)^((n+1)/n)."""
    K, h, V = 3.0, 0.006, 0.4
    r = NN.flow(V * h, rheology=pl(K, n), geometry="slit", gap_m=h, density_kg_m3=1000.0, positions_m=[0.0015])
    gw = (2 * n + 1) / n * 2 * V / h
    assert r["evidence"]["verified"]
    assert r["wall_shear_rate_1_s"] == pytest.approx(gw, rel=1e-10)
    assert r["wall_shear_stress_Pa"] == pytest.approx(K * gw ** n, rel=1e-10)
    assert r["pressure_gradient_Pa_m"] == pytest.approx(2 * K * gw ** n / h, rel=1e-10)
    assert r["max_to_mean"] == pytest.approx((2 * n + 1) / (n + 1), rel=1e-10)
    u = r["velocity_at_positions"][0]["u_m_s"]
    assert u == pytest.approx(V * (2 * n + 1) / (n + 1) * (1 - 0.5 ** ((n + 1) / n)), rel=1e-10)


def test_slit_bingham_closed_form_and_no_flow():
    """Bingham between plates: V = h tau_w/(6 mu_p) (1 - 1.5 phi + 0.5 phi^3), phi = tau_y/tau_w; plug half-thickness
    = phi h/2; no flow when tau_w <= tau_y."""
    ty, mup, h, G = 5.0, 0.1, 0.01, 5000.0
    rh = {"model": "bingham", "yield_stress_Pa": ty, "plastic_viscosity_Pa_s": mup}
    r = NN.flow(rheology=rh, geometry="slit", gap_m=h, pressure_gradient_Pa_m=G, density_kg_m3=1000.0)
    tw = G * h / 2
    phi = ty / tw
    assert r["wall_shear_stress_Pa"] == pytest.approx(tw)
    assert r["mean_velocity_m_s"] == pytest.approx(h * tw / (6 * mup) * (1 - 1.5 * phi + 0.5 * phi ** 3), rel=1e-10)
    assert r["plug_half_thickness_m"] == pytest.approx(phi * h / 2)
    assert r["evidence"]["verified"]
    still = NN.flow(rheology=rh, geometry="slit", gap_m=h, pressure_gradient_Pa_m=500.0)
    assert still["regime"] == "sin flujo" and still["mean_velocity_m_s"] == 0.0


def test_tube_inverse_matches_forward_exact_solution():
    """Gradient -> flow in the tube inverts the classic exact solution (Hagen-Poiseuille: Q = pi R^4 G / (8 mu))."""
    D, G = 0.02, 4000.0
    r = NN.flow(None, D, None, 1000.0, NEWT, pressure_gradient_Pa_m=G)
    assert r["volumetric_flow_m3_s"] == pytest.approx(math.pi * (D / 2) ** 4 * G / (8 * 0.25), rel=1e-10)
    assert r["max_to_mean"] == pytest.approx(2.0, rel=1e-10)
    hb = {"model": "herschel_bulkley", "yield_stress_Pa": 4.0, "K_Pa_sn": 0.6, "n": 0.55}
    fwd = NN.flow(3e-4, D, 2.0, 1100.0, hb, positions_m=[0.0, 0.004])
    inv = NN.flow(None, D, 2.0, 1100.0, hb, pressure_gradient_Pa_m=fwd["laminar_solution"]["pressure_drop_Pa"] / 2.0,
                  positions_m=[0.0, 0.004])
    assert inv["volumetric_flow_m3_s"] == pytest.approx(3e-4, rel=1e-9)
    for a, b in zip(fwd["laminar_solution"]["velocity_at_positions"], inv["velocity_at_positions"]):
        assert a["u_m_s"] == pytest.approx(b["u_m_s"], rel=1e-9)


# ------------------------------------------------------------------ Carreau family
@pytest.mark.parametrize("rheo", [carreau(0.25, 3.0, 1.0), carreau(0.25, 0.0, 0.4),
                                  {"model": "carreau_yasuda", "eta0_Pa_s": 0.25, "lambda_s": 0.0, "n": 0.3, "a": 1.3},
                                  {"model": "cross", "eta0_Pa_s": 0.25, "lambda_s": 0.0, "m": 0.8}])
def test_carreau_family_reduces_to_newtonian(rheo):
    """n = 1 or lambda = 0 -> Newtonian of viscosity eta0: Hagen-Poiseuille (tube) and plane Poiseuille (slit)."""
    Q, D, L = 2e-4, 0.02, 2.0
    r = NN.flow(Q, D, L, 1000.0, rheo)
    assert r["evidence"]["verified"] and r["regime"] == "laminar"
    assert r["pressure_drop_Pa"] == pytest.approx(128 * 0.25 * L * Q / (math.pi * D ** 4), rel=1e-9)
    assert r["max_to_mean"] == pytest.approx(2.0, rel=1e-9)
    h, V = 0.004, 0.3
    s = NN.flow(V * h, rheology=rheo, geometry="slit", gap_m=h)
    assert s["pressure_gradient_Pa_m"] == pytest.approx(12 * 0.25 * V / h ** 2, rel=1e-9)
    assert s["max_to_mean"] == pytest.approx(1.5, rel=1e-9)


def test_carreau_high_shear_limit_is_power_law():
    """lambda·gamma >> 1 over nearly all the section (eta_inf = 0): Carreau -> power law with K = eta0 lambda^(n-1).
    Tube: tau_w = K ((3n+1)/(4n) · 8V/D)^n; slit: tau_w = K ((2n+1)/n · 2V/h)^n. The deviation is O((lambda g_w)^-2)."""
    eta0, lam, n = 10.0, 1.0e3, 0.5
    K = eta0 * lam ** (n - 1)
    D, V = 0.02, 1.0
    r = NN.flow(V * math.pi * D * D / 4, D, 1.0, 1000.0, carreau(eta0, lam, n))
    assert r["evidence"]["verified"]
    assert r["wall_shear_stress_Pa"] == pytest.approx(K * ((3 * n + 1) / (4 * n) * 8 * V / D) ** n, rel=1e-5)
    assert r["max_to_mean"] == pytest.approx((3 * n + 1) / (n + 1), rel=1e-4)
    h = 0.004
    s = NN.flow(V * h, rheology=carreau(eta0, lam, n), geometry="slit", gap_m=h)
    assert s["wall_shear_stress_Pa"] == pytest.approx(K * ((2 * n + 1) / n * 2 * V / h) ** n, rel=1e-5)


def test_carreau_tube_against_independent_integration():
    """Intermediate case (plateau and power-law zones both matter): Q recomputed here from the tool's wall stress with
    an independent Rabinowitsch-Mooney integral, the shear rate obtained by brentq inside the integrand."""
    eta0, einf, lam, n, D = 8.0, 0.02, 0.5, 0.45, 0.03
    rh = carreau(eta0, lam, n, einf)
    r = NN.flow(5e-4, D, 4.0, 1050.0, rh, positions_m=[0.0])
    assert r["evidence"]["verified"]
    tw, R = r["wall_shear_stress_Pa"], D / 2

    def eta(g):
        return einf + (eta0 - einf) * (1 + (lam * g) ** 2) ** ((n - 1) / 2)

    def gamma(tau):
        return brentq(lambda g: eta(g) * g - tau, 0.0, 1e9, xtol=1e-14, rtol=1e-14) if tau > 0 else 0.0
    integral, _ = quad(lambda t: t * t * gamma(t), 0.0, tw, epsabs=0, epsrel=1e-11, limit=200)
    assert math.pi * R ** 3 / tw ** 3 * integral == pytest.approx(5e-4, rel=1e-8)
    u0, _ = quad(lambda rr: gamma(tw * rr / R), 0.0, R, epsabs=0, epsrel=1e-11, limit=200)
    assert r["velocity_at_positions"][0]["u_m_s"] == pytest.approx(u0, rel=1e-8)
    # shear thinning: between the Newtonian (2) and plug-like profile, and gradient -> flow returns the same Q
    assert 1.0 < r["max_to_mean"] < 2.0
    inv = NN.flow(None, D, None, 1050.0, rh, pressure_gradient_Pa_m=r["pressure_gradient_Pa_m"])
    assert inv["volumetric_flow_m3_s"] == pytest.approx(5e-4, rel=1e-9)


def test_carreau_turbulent_range_is_flagged_not_extrapolated():
    r = NN.flow(5e-3, 0.02, 1.0, 1000.0, carreau(1e-3, 1e-4, 0.8))
    assert not r["evidence"]["verified"] and "NO aplicable" in r["regime"]


# ------------------------------------------------------------------ apparent viscosity and crossings
def test_viscosity_table_and_crossing_point():
    """Carreau (eta_inf > 0) against the power law of its own intermediate branch: the crossing is found here by
    brentq on the two formulas written out explicitly."""
    eta0, einf, lam, n = 10.0, 0.001, 2.0, 0.4
    K = eta0 * lam ** (n - 1)

    def e_car(g):
        return einf + (eta0 - einf) * (1 + (lam * g) ** 2) ** ((n - 1) / 2)

    def e_pl(g):
        return K * g ** (n - 1)
    g_cross = brentq(lambda g: e_car(g) - e_pl(g), 1.0, 1e3, xtol=1e-14, rtol=1e-14)
    r = TP.non_newtonian_pipe_flow(rheology=carreau(eta0, lam, n, einf), shear_rates_1_s=[0.01, 1.0, 100.0],
                                   compare_rheology=pl(K, n))
    assert r["mode"] == "viscosity" and r["evidence"]["verified"]
    v = r["viscosity"]
    assert len(v["crossings"]) == 1
    assert v["crossings"][0]["shear_rate_1_s"] == pytest.approx(g_cross, rel=1e-9)
    assert v["crossings"][0]["viscosity_Pa_s"] == pytest.approx(e_pl(g_cross), rel=1e-9)
    for row in v["table"]:
        assert row["viscosity_Pa_s"] == pytest.approx(e_car(row["shear_rate_1_s"]), rel=1e-12)
        assert row["viscosity_compare_Pa_s"] == pytest.approx(e_pl(row["shear_rate_1_s"]), rel=1e-12)
    assert v["asymptotes"]["rheology"]["power_law_branch"]["n"] == pytest.approx(n)


def test_viscosity_crossings_exact_cases():
    # Carreau (eta_inf = 0) vs a Newtonian of viscosity mu: [1 + (lam g)^2]^((n-1)/2) = mu/eta0, solved by hand
    eta0, lam, n, mu = 5.0, 1.0, 0.5, 1.0
    r = NN.flow(rheology=carreau(eta0, lam, n), compare_rheology={"model": "newtonian", "viscosity_Pa_s": mu})
    g = math.sqrt((mu / eta0) ** (2 / (n - 1)) - 1) / lam            # sqrt(624) = 24.98
    assert r["viscosity"]["crossings"][0]["shear_rate_1_s"] == pytest.approx(g, rel=1e-9)
    # zero-shear plateau vs power-law branch: gamma = 1/lambda
    a = r["viscosity"]["asymptotes"]["rheology"]
    assert a["plateau_meets_power_law_at_shear_rate_1_s"] == pytest.approx(1 / lam)
    assert a["power_law_branch"]["K_Pa_sn"] == pytest.approx(eta0 * lam ** (n - 1))
    # Carreau with eta_inf = 0 stays below its power-law asymptote at every shear rate: no crossing, and it says so
    none = NN.flow(rheology=carreau(eta0, lam, n), compare_rheology=pl(eta0 * lam ** (n - 1), n))
    assert none["viscosity"]["crossings"] == [] and "no se cruzan" in " ".join(none["warnings"])
    # Bingham vs Newtonian: tau_y/g + mu_p = mu  ->  g = tau_y/(mu - mu_p)
    b = NN.flow(rheology={"model": "bingham", "yield_stress_Pa": 6.0, "plastic_viscosity_Pa_s": 0.2},
                compare_rheology={"model": "newtonian", "viscosity_Pa_s": 0.5}, shear_rates_1_s=[10.0])
    assert b["viscosity"]["crossings"][0]["shear_rate_1_s"] == pytest.approx(6.0 / 0.3, rel=1e-9)
    assert b["viscosity"]["table"][0]["viscosity_Pa_s"] == pytest.approx(0.8)
    # attached to a flow calculation as an extra block
    f = NN.flow(1e-4, 0.02, 1.0, 1000.0, pl(2.0, 0.5), shear_rates_1_s=[4.0])
    assert f["viscosity"]["table"][0]["viscosity_Pa_s"] == pytest.approx(1.0) and "pressure_drop_Pa" in f


# ------------------------------------------------------------------ refusals and plumbing
def test_refusals_in_spanish():
    with pytest.raises(ValueError, match="gap_m"):
        NN.flow(1e-4, rheology=NEWT, geometry="slit")
    with pytest.raises(ValueError, match="geometry"):
        NN.flow(1e-4, 0.02, 1.0, 1000.0, NEWT, geometry="annulus")
    with pytest.raises(ValueError, match="no ambos"):
        NN.flow(1e-4, 0.02, 1.0, 1000.0, NEWT, pressure_gradient_Pa_m=10.0)
    with pytest.raises(ValueError, match="falta volumetric_flow_m3_s o pressure_gradient_Pa_m"):
        NN.flow(None, 0.02, 1.0, 1000.0, NEWT)
    with pytest.raises(ValueError, match="eta0_Pa_s"):
        NN.flow(1e-4, 0.02, 1.0, 1000.0, {"model": "carreau", "lambda_s": 1.0, "n": 0.5})
    with pytest.raises(ValueError, match="rheology.a"):
        NN.flow(1e-4, 0.02, 1.0, 1000.0, {"model": "carreau_yasuda", "eta0_Pa_s": 1.0, "lambda_s": 1.0, "n": 0.5})
    with pytest.raises(ValueError, match="η∞ ≤ η0"):
        NN.flow(1e-4, 0.02, 1.0, 1000.0, carreau(1.0, 1.0, 0.5, eta_inf=2.0))
    with pytest.raises(ValueError, match="no es monótona"):
        NN.flow(1e-4, 0.02, 1.0, 1000.0, {"model": "cross", "eta0_Pa_s": 1.0, "lambda_s": 1.0, "m": 1.5})
    with pytest.raises(ValueError, match="fuera del ducto"):
        NN.flow(1e-4, rheology=NEWT, geometry="slit", gap_m=0.004, positions_m=[0.003])
    with pytest.raises(ValueError, match="shear_rates_1_s"):
        NN.flow(rheology=NEWT, shear_rates_1_s=[-1.0])
    with pytest.raises(ValueError, match="diameter_m"):
        NN.flow(None, None, None, None, NEWT, pressure_gradient_Pa_m=100.0)


def test_results_are_json_and_wrapper_passes_new_arguments():
    from packages.agent import wrappers_particles as W
    w = W.AGENT_WRAPPERS[0]
    out = json.loads(w(volumetric_flow_m3_s=1e-4, rheology=carreau(8.0, 0.5, 0.45), geometry="slit", gap_m=0.004,
                       width_m=0.2, length_m=1.0, density_kg_m3=1000.0, positions_m=[0.0, 0.001],
                       shear_rates_1_s=[1.0]))
    assert out["evidence"]["verified"] and len(out["velocity_at_positions"]) == 2 and "viscosity" in out
    for r in (NN.flow(1e-4, 0.02, 1.0, 1000.0, carreau(8.0, 0.5, 0.45)),
              NN.flow(rheology=NEWT, geometry="slit", gap_m=0.004, pressure_gradient_Pa_m=50.0),
              NN.flow(rheology=NEWT, shear_rates_1_s=[1.0])):
        json.dumps(r, ensure_ascii=False)
        for k in ("warnings", "assumptions", "not_included", "method", "evidence"):
            assert k in r
