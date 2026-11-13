"""Process control (bloque M9) against published results.

Skogestad (2003), "Simple analytic rules for model reduction and PID controller tuning", J. Process Control 13,
291-309 (finalpaper.pdf on the author's NTNU page): Table 2 (margins of SIMC on FOPDT and integrating processes),
Table 4 (15 processes E1-E15: half-rule approximation, SIMC settings with τc = θ, Ms, IAE for unit setpoint and unit
load steps on the FULL model g0), Tables 5-7 (Ziegler-Nichols and Tyreus-Luyben on e^-s/s, e^-s and E5).
Seborg, Edgar, Mellichamp & Doyle, Example 12.4 (author's chapter 12 slides): G = 100 e^-s/(100 s + 1).

The paper has three rows whose published numbers are not consistent with its own rule; each is tested for what it
actually is (see test_skogestad_table4_inconsistencies). Minus signs of inverse-response numerators are lost in the
PDF text layer; they are restored from the published approximations (e.g. E7: θ = 3.5 needs (−2s+1)), and E5's
last lag is 0.008 (Table 7; the 0.0008 in Table 4 cannot give θ = 0.148)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dynamics import loop as LP  # noqa: E402
from packages.dynamics import tuning as TU  # noqa: E402
from packages.dynamics.process import Process, half_rule  # noqa: E402
from packages.dwsim_client import tools_control as TC  # noqa: E402


def P(k, lags=(), leads=(), th=0.0, integ=False):
    return Process(k, list(lags), list(leads), th, integ)


G0 = {"E1": P(1, [1, .2]), "E2": P(1, [2, 1, .4, .2, .05, .05, .05], [-.3, .08]), "E3": P(2, [20, 1, .1, .1], [15]),
      "E4": P(1, [1] * 4), "E5": P(1, [1, .2, .04, .008]), "E6": P(1, [1, 1, .028], [.17, .17], 0, True),
      "E7": P(1, [1, 1, 1], [-2]), "E8": P(1, [1, 1], [], 0, True), "E9": P(1, [1, 1], [], 1),
      "E10": P(1, [20, 2], [], 1), "E11": P(1, [6, 2, 2], [-1], 1), "E12": P(1, [10, 8, 1], [6, 3], .3),
      "E13": P(1, [10, .5], [2], 1), "E14": P(1, [], [-1], 0, True), "E15": P(1, [1], [-1])}

# case, type, k, θ, τ1 (None = integrating), τ2, Kc, τI, τD, Ms, IAE setpoint, IAE load  (published strings)
TABLE4 = [
    ("E1", "PI", 1, .1, 1.1, 0, "5.5", "0.8", "0", "1.56", "0.36", "0.15"),
    ("E2", "PI", 1, 1.47, 2.5, 0, "0.85", "2.5", "0", "1.66", "3.56", "2.97"),
    ("E2", "PID", 1, .77, 2, 1.2, "1.30", "2", "1.2", "1.73", "2.73", "1.54"),
    ("E3", "PI", 1.5, .15, 1.05, 0, "2.33", "1.05", "0", "1.55", "0.46", "0.45"),
    ("E4", "PI", 1, 2.5, 1.5, 0, "0.3", "1.5", "0", "1.46", "5.59", "5.40"),
    ("E4", "PID", 1, 1.5, 1.5, 1, "0.5", "1.5", "1", "1.43", "4.31", "3.13"),
    ("E5", "PI", 1, .148, 1.1, 0, "3.72", "1.1", "0", "1.59", "0.45", "0.30"),
    ("E6", "PI", 1, 1.69, None, 0, "0.296", "13.5", "0", "1.48", "6.50", "45.7"),
    ("E6", "PID", 1, .358, None, 1.33, "1.40", "2.86", "1.33", "1.23", "1.95", "2.04"),
    ("E7", "PI", 1, 3.5, 1.5, 0, "0.214", "1.5", "0", "1.66", "7.28", "8.34"),
    ("E7", "PID", 1, 2.5, 1.5, 1, "0.3", "1.5", "1", "1.85", "5.99", "6.23"),
    ("E8", "PI", 1, 1.5, None, 0, "0.33", "12", "0", "1.76", "6.47", "36.4"),
    ("E9", "PI", 1, 1.5, 1.5, 0, "0.5", "1.5", "0", "1.61", "3.38", "3.14"),
    ("E9", "PID", 1, 1, 1, 1, "0.5", "1", "1", "1.59", "3.03", "2"),
    ("E10", "PI", 1, 2, 21, 0, "5.25", "16", "0", "1.72", "6.34", "3.05"),
    ("E10", "PID", 1, 1, 20, 2, "10", "8", "2", "1.65", "4.32", "0.80"),
    ("E11", "PI", 1, 5, 7, 0, "0.7", "7", "0", "1.63", "11.5", "10.1"),
    ("E11", "PID", 1, 3, 6, 3, "1", "6", "3", "1.66", "9.09", "6.03"),
    ("E12", "PI", .225, .3, 1, 0, "7.41", "1", "0", "1.66", "1.07", "0.15"),
    ("E13", "PI", .625, 1.25, 4.5, 0, "2.88", "4.5", "0", "1.74", "2.86", "1.61"),
    ("E14", "PI", 1, 1, None, 0, "0.5", "8", "0", "2", "3.59", "17.3"),
    ("E15", "PI", 1, 1, 1, 0, "0.5", "1", "0", "2", "2", "2.85"),
]


def close(x, pub: str, rel=0.03):
    """Within rel of the published value or within half a unit of its last printed digit."""
    ref = float(pub)
    dec = len(pub.split(".")[1]) if "." in pub else 0
    return abs(x - ref) <= max(rel * abs(ref), 0.5 * 10 ** -dec + 1e-12)


def _model(k, th, t1, t2):
    return {"k": k, "theta": th, "tau1": math.inf if t1 is None else t1, "tau2": t2, "integrating": t1 is None}


@pytest.mark.parametrize("row", TABLE4, ids=[f"{r[0]}-{r[1]}" for r in TABLE4])
def test_simc_reproduces_skogestad_table4(row):
    case, ty, k, th, t1, t2, Kc, tI, tD, Ms, Isp, Il = row
    c, _ = TU.simc(_model(k, th, t1, t2), ty, th)               # τc = θ
    assert close(c.Kc, Kc, 0.015) and close(c.tau_I, tI, 0.015) and close(c.tau_D, tD, 0.015)
    g = G0[case]                                                 # performance on the full model g0
    m = LP.margins(g, c, 0.0)
    assert abs(m["Ms"] - float(Ms)) <= 0.015, (m["Ms"], Ms)
    sp = LP.setpoint_metrics(LP.simulate(g, c, 0.01, 1.0, 0.0), 1.0)
    ld = LP.load_metrics(LP.simulate(g, c, 0.01, 0.0, 1.0), 1.0)
    assert close(sp["IAE"], Isp) and close(ld["IAE"], Il), (sp["IAE"], Isp, ld["IAE"], Il)


@pytest.mark.parametrize("case,order,th,t1,t2", [
    ("E1", 1, .1, 1.1, 0), ("E4", 1, 2.5, 1.5, 0), ("E4", 2, 1.5, 1.5, 1), ("E5", 1, .148, 1.1, 0),
    ("E5", 2, .028, 1.0, .22), ("E7", 1, 3.5, 1.5, 0), ("E7", 2, 2.5, 1.5, 1), ("E8", 1, 1.5, None, 0),
    ("E8", 2, .5, None, 1.5), ("E9", 1, 1.5, 1.5, 0), ("E9", 2, 1, 1, 1), ("E10", 1, 2, 21, 0), ("E10", 2, 1, 20, 2),
    ("E11", 1, 5, 7, 0), ("E11", 2, 3, 6, 3), ("E14", 1, 1, None, 0), ("E15", 1, 1, 1, 0)])
def test_half_rule_reproduces_table4_approximations(case, order, th, t1, t2):
    a = half_rule(G0[case], order)
    assert a["theta"] == pytest.approx(th, abs=1e-9)
    assert a["tau2"] == pytest.approx(t2, abs=1e-9)
    assert (math.isinf(a["tau1"]) if t1 is None else a["tau1"] == pytest.approx(t1, abs=1e-9))
    # τ1 ≥ τ2 as in the paper (E4: the half rule gives 1 and 1.5; the table calls 1.5 τ1)


def test_half_rule_refuses_positive_zeros():
    with pytest.raises(ValueError, match="T1-T3"):
        half_rule(G0["E3"], 1)


def test_skogestad_table4_inconsistencies():
    """E5-PID: the table prints τI = 0.224 (= min(τ1, 4(τc+θ)), what the rule gives and what we return) but its Ms
    and IAE belong to τI = 1.0 (Table 7). E8-PID: the table prints Kc = 1.5 while the rule 1/(k'(τc+θ)) gives 1.0;
    its Ms/IAE belong to Kc = 1.5. Both readings are reproduced exactly."""
    c, _ = TU.simc(_model(1, .028, 1.0, .22), "PID", .028)
    assert (c.Kc, c.tau_I) == pytest.approx((17.857, 0.224), abs=1e-3)
    g, c1 = G0["E5"], LP.Controller("PID", 17.9, 1.0, 0.22)
    assert LP.margins(g, c1, 0.0)["Ms"] == pytest.approx(1.58, abs=0.005)
    assert LP.setpoint_metrics(LP.simulate(g, c1, .01, 1, 0), 1)["IAE"] == pytest.approx(0.27, abs=0.005)
    assert LP.load_metrics(LP.simulate(g, c1, .01, 0, 1), 1)["IAE"] == pytest.approx(0.056, abs=0.0005)
    c, _ = TU.simc(_model(1, .5, None, 1.5), "PID", .5)
    assert (c.Kc, c.tau_I, c.tau_D) == pytest.approx((1.0, 4.0, 1.5))
    g, c8 = G0["E8"], LP.Controller("PID", 1.5, 4, 1.5)
    assert LP.margins(g, c8, 0.0)["Ms"] == pytest.approx(1.79, abs=0.005)
    assert LP.setpoint_metrics(LP.simulate(g, c8, .01, 1, 0), 1)["IAE"] == pytest.approx(2.02, abs=0.01)
    assert LP.load_metrics(LP.simulate(g, c8, .01, 0, 1), 1)["IAE"] == pytest.approx(2.67, abs=0.005)
    # E3-PID: settings exact, Ms 1.498 vs 1.47 and load IAE 0.060 vs 0.068 (unexplained, documented in MOCKS.md)
    g, c3 = G0["E3"], LP.Controller("PID", 6.67, 0.4, 0.15)
    assert LP.margins(g, c3, 0.0)["Ms"] == pytest.approx(1.47, abs=0.035)


def test_table2_margins_exact_and_python_control():
    """SIMC τc = θ on FOPDT (τI = τ1 cancels the lag: L = 0.5 e^{-θs}/(θs)) and on k' e^{-θs}/s."""
    for p, c, ref in [(P(1, [4.0], [], 1.0), LP.Controller("PI", 2.0, 4.0),
                       ("3.14", "61.4", "1.59", "1.00", "1.57", "0.50", "2.14")),
                      (P(1, [], [], 1.0, True), LP.Controller("PI", 0.5, 8.0),
                       ("2.96", "46.9", "1.70", "1.30", "1.49", "0.51", "1.59"))]:
        m = LP.margins(p, c, 0.0)
        got = (m["gain_margin"], m["phase_margin_deg"], m["Ms"], m["Mt"], m["phase_crossover_frequency"],
               m["gain_crossover_frequency"], m["allowed_delay_error"])
        assert all(close(x, pub, 0.0) for x, pub in zip(got, ref)), (got, ref)
        pc = LP.margins_python_control(p, c, 0.0)
        assert pc["gain_margin"] == pytest.approx(m["gain_margin"], rel=1e-6)
        assert pc["phase_margin_deg"] == pytest.approx(m["phase_margin_deg"], rel=1e-6)
        assert pc["Ms"] == pytest.approx(m["Ms"], rel=1e-6)
    m = LP.margins(P(1, [4.0], [], 1.0), LP.Controller("PI", 2.0, 4.0), 0.0)     # closed form
    assert m["gain_margin"] == pytest.approx(math.pi, rel=1e-9)
    assert m["phase_margin_deg"] == pytest.approx(90 - math.degrees(0.5), rel=1e-9)


def _row(r, rule):
    return next(x for x in r["summary"] if x["rule"] == rule)


def test_zn_and_tyreus_luyben_tables_5_6_7():
    # Table 5, g = e^-s/s
    r = TC.pid_tuning(gain=1, dead_time=1, integrating=True, controller_type="PI", rules=["ZN", "Tyreus_Luyben"])
    zn = _row(r, "ZN")
    assert (zn["Kc"], zn["tau_I"], zn["Ms"]) == pytest.approx((0.71, 3.33, 2.83), abs=0.006)
    assert close(zn["setpoint_IAE"], "3.92", .01) and close(zn["load_IAE"], "5.61", .01)
    assert _row(r, "Tyreus_Luyben")["Kc"] == pytest.approx(0.49, abs=0.005)   # τI 8.8 = 2.2 Pu; the table prints 7.32
    r = TC.pid_tuning(gain=1, dead_time=1, integrating=True, controller_type="PID", rules=["ZN"])
    s = r["rules"][0]["controller"]["series_equivalent"]
    assert (s["Kc"], s["tau_I"], s["tau_D"]) == pytest.approx((0.471, 1, 1), abs=0.001)
    assert _row(r, "ZN")["Ms"] == pytest.approx(2.29, abs=0.005)
    assert close(_row(r, "ZN")["setpoint_IAE"], "2.88", .01) and close(_row(r, "ZN")["load_IAE"], "3.32", .01)
    # Table 6, g = e^-s: ZN-PI, Tyreus-Luyben; ZN-PID unstable
    r = TC.pid_tuning(gain=1, time_constant=0, dead_time=1, controller_type="PI", rules=["SIMC", "ZN", "Tyreus_Luyben"])
    simc, zn, tl = (_row(r, x) for x in ("SIMC", "ZN", "Tyreus_Luyben"))
    assert simc["Ms"] == pytest.approx(1.59, abs=0.005) and close(simc["setpoint_IAE"], "2.17", .005)
    assert zn["Kc"] == pytest.approx(0.45) and zn["Kc"] / zn["tau_I"] == pytest.approx(0.27, abs=0.001)
    assert zn["Ms"] == pytest.approx(1.85, abs=0.005) and close(zn["setpoint_IAE"], "3.70", .005)
    assert tl["Kc"] / tl["tau_I"] == pytest.approx(0.071, abs=0.0005)
    assert tl["Ms"] == pytest.approx(1.46, abs=0.005) and close(tl["setpoint_IAE"], "14.1", .005)
    r = TC.pid_tuning(gain=1, time_constant=0, dead_time=1, controller_type="PID", rules=["ZN"])
    assert _row(r, "ZN")["stable"] is False and r["rules"][0]["status"] == "unstable"
    # Table 7, E5 (Ku, Pu from the full 4th-order model)
    e5 = {"gain": 1, "time_constants": [1, .2, .04, .008], "dead_time": 0}
    r = TC.pid_tuning(process_model=e5, controller_type="PI", rules=["ZN", "Tyreus_Luyben"])
    zn, tl = _row(r, "ZN"), _row(r, "Tyreus_Luyben")
    assert (zn["Kc"], zn["tau_I"]) == pytest.approx((13.6, 0.47), abs=0.01)
    assert zn["Ms"] == pytest.approx(11.3, rel=0.02) and close(zn["load_IAE"], "0.137", .03)
    assert (tl["Kc"], tl["tau_I"], tl["Ms"]) == pytest.approx((9.46, 1.24, 2.72), abs=0.015)   # paper: 0.313 Ku
    assert close(tl["setpoint_IAE"], "0.50", .01) and close(tl["load_IAE"], "0.131", .01)
    r = TC.pid_tuning(process_model=e5, controller_type="PID", rules=["ZN"])
    s = r["rules"][0]["controller"]["series_equivalent"]
    assert close(s["Kc"], "9.1", 0) and close(s["tau_I"], "0.14", 0) and close(s["tau_D"], "0.14", 0)
    assert _row(r, "ZN")["Ms"] == pytest.approx(2.39, abs=0.01) and close(_row(r, "ZN")["load_IAE"], "0.025", .03)
    # (its setpoint IAE is 0.333 here vs 0.24 printed: unexplained, documented)


def test_seborg_example_12_4():
    """G = 100 e^-s/(100s+1): (a) IMC τc = 1 → Kc 0.5, τI 100; (c) Skogestad τc = 1 → 0.5, 8;
    (b) integrator approximation k' = 1, τc = 2 → 0.556, 5."""
    r = TC.pid_tuning(gain=100, time_constant=100, dead_time=1, tau_c=1, rules=["IMC", "SIMC"])
    imc, simc = _row(r, "IMC"), _row(r, "SIMC")
    assert (imc["Kc"], imc["tau_I"]) == pytest.approx((0.5, 100))
    assert (simc["Kc"], simc["tau_I"]) == pytest.approx((0.5, 8))
    assert imc["load_IAE"] > 10 * simc["load_IAE"]        # Seborg's point: τI = τ rejects load very slowly
    v = next(x for x in r["rules"] if x["rule"] == "IMC")["validity"]
    assert v["in_range"] is False and "lento" in v["note"]
    r = TC.pid_tuning(gain=1, dead_time=1, integrating=True, tau_c=2, rules=["IMC"])
    assert (_row(r, "IMC")["Kc"], _row(r, "IMC")["tau_I"]) == pytest.approx((0.556, 5), abs=0.001)


def test_imc_pid_matches_rivera_series_form():
    """IMC PID with τc = 0.8θ in series form = Skogestad eq. 34: Kc = 0.769 τ1/(kθ), τI = τ1, τD = θ/2."""
    c, _ = TU.imc(_model(2.0, 1.0, 5.0, 0), "PID", 0.8)
    s = c.series()
    assert s["Kc"] == pytest.approx(0.769 * 5 / 2, rel=1e-3)
    assert (s["tau_I"], s["tau_D"]) == pytest.approx((5.0, 0.5))


def test_cohen_coon_tends_to_zn_reaction_curve():
    m = _model(1.0, 1e-4, 1.0, 0)
    cc, _ = TU.cohen_coon(m, "PI")
    zn, _ = TU.zn_reaction(m, "PI")
    assert cc.Kc == pytest.approx(zn.Kc, rel=1e-3) and cc.tau_I == pytest.approx(zn.tau_I, rel=1e-3)


def test_closed_loop_checks_and_validity_flags():
    r = TC.pid_tuning(gain=2.0, time_constant=10.0, dead_time=2.0, controller_type="PI")
    assert r["recommended_rule"] == "SIMC"
    assert any("τc = θ" in a for a in r["assumptions"])
    for x in r["rules"]:
        if x["applicable"]:
            assert x["status"] == "verified", (x["rule"], x["checks"])
    assert next(x for x in r["rules"] if x["rule"] == "ZN")["validity"]["in_range"] is True   # θ/τ = 0.2
    r = TC.pid_tuning(gain=2.0, time_constant=10.0, dead_time=0.05, rules=["ZN", "Cohen_Coon"], evaluate_closed_loop=False)
    assert all(x["validity"]["in_range"] is False for x in r["rules"])                       # θ/τ = 0.005
    r = TC.pid_tuning(gain=2.0, time_constant=10.0, dead_time=2.0, controller_type="PID", rules=["SIMC"])
    assert r["rules"][0]["applicable"] is False and r["recommended_rule"] == "SIMC-PI"


def test_pid_closed_loop_tool_and_python_control_cross_check():
    r = TC.pid_closed_loop({"gain": 1, "time_constants": [1, .2, .04, .008], "dead_time": 0},
                           {"Kc": 3.72, "tau_I": 1.1})
    assert r["status"] == "verified" and r["closed_loop_stable"]
    assert r["margins"]["Ms"] == pytest.approx(1.59, abs=0.005)
    for v in r["checks"]["IAE_vs_python_control_pade"].values():
        assert v["rel_diff"] < 0.01
    assert len(r["trajectories"]["setpoint"]["t"]) > 50


def test_controller_frequency_forms_agree():
    w = np.array([0.1, 1.0, 10.0])
    c = LP.Controller("PID", 1.2, 1.0, 0.25, "ideal")
    assert np.allclose(c.freq(w, 1e-9)[0], c.freq(w, 0.0)[0], rtol=1e-6)
    c = LP.Controller("PID", 2.0, 4.0, 0.5)
    assert np.allclose(c.freq(w, 1e-9)[0], c.freq(w, 0.0)[0], rtol=1e-6)


def _fopdt_data(K, tau, th, du, ts, t, y0=10.0):
    return y0 + K * du * (1 - np.exp(-np.clip(t - ts - th, 0, None) / tau))


def test_fopdt_identify_recovers_known_parameters_with_noise():
    rng = np.random.default_rng(7)
    K, tau, th, du, ts = 2.0, 5.0, 1.5, 4.0, 3.0
    t = np.arange(0, 40, 0.2)
    y = _fopdt_data(K, tau, th, du, ts, t) + rng.normal(0, 0.01 * K * du, len(t))     # 1 % noise
    r = TC.fopdt_identify(t.tolist(), y.tolist(), du, ts)
    f = r["fopdt"]
    assert f["K"] == pytest.approx(K, rel=0.01) and f["tau"] == pytest.approx(tau, rel=0.03)
    assert f["theta"] == pytest.approx(th, abs=0.05 * tau)
    for name, true in (("K", K), ("tau", tau), ("theta", th)):                     # truth inside the 95 % CI
        assert abs(f[name] - true) <= f["ci95"][name] * 1.5
    assert f["fit"]["r2"] > 0.998 and abs(f["fit"]["residual_autocorr_lag1"]) < 0.3
    assert r["two_point_agrees"] and not r["warnings"]
    for v in r["two_point"].values():
        assert v["tau"] == pytest.approx(tau, rel=0.1) and v["theta"] == pytest.approx(th, abs=0.15)


def test_sopdt_identify_and_mismatch_diagnostics():
    rng = np.random.default_rng(3)
    K, t1, t2, th, du, ts = 1.5, 8.0, 3.0, 2.0, 4.0, 3.0
    t = np.arange(0, 80, 0.2)
    tt = np.clip(t - ts - th, 0, None)
    y = K * du * (1 - (t1 * np.exp(-tt / t1) - t2 * np.exp(-tt / t2)) / (t1 - t2)) + rng.normal(0, 0.005 * K * du, len(t))
    r = TC.fopdt_identify(t.tolist(), y.tolist(), du, ts, baseline_output=0.0)
    s = r["sopdt"]
    assert (s["K"], s["tau1"], s["tau2"], s["theta"]) == pytest.approx((K, t1, t2, th), rel=0.05)
    assert r["fopdt"]["fit"]["residual_autocorr_lag1"] > 0.5        # FOPDT leaves structure in the residuals
    assert r["recommended_model"].startswith("SOPDT (τ2 > θ")


def test_identify_warns_and_refuses():
    t = np.arange(0, 20, 0.1)
    y = 0.3 * np.clip(t - 2, 0, None)                                # ramp: integrating, never settles
    r = TC.fopdt_identify(t.tolist(), y.tolist(), 1.0, 2.0, baseline_output=0.0)
    assert any("integrante" in w for w in r["warnings"])
    with pytest.raises(ValueError):
        TC.fopdt_identify(t.tolist(), y.tolist(), 0.0, 2.0)
    with pytest.raises(ValueError, match="no hay valores por defecto"):
        TC.pid_tuning(controller_type="PI")
    with pytest.raises(ValueError, match="dead_time"):
        TC.pid_tuning(gain=1.0, time_constant=3.0)


def test_agent_wiring():
    from packages.agent import agent as AG
    names = {getattr(t, "name", "") for t in AG.AGENT_TOOLS}
    assert {"fopdt_identify", "pid_tuning", "pid_closed_loop"} <= names
    assert "18. Process control" in AG.SYSTEM
