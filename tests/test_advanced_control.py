"""Validation of the advanced-control block (advcontrol): linear MPC + PID baseline (part A, repo environment) and the
reinforcement-learning experiment on pc-gym (part B, separate environment; skipped automatically when it is absent).

Sources read on 2026-11-14:
- Rawlings, Mayne & Diehl (2017), «Model Predictive Control: Theory, Computation, and Design», 2nd ed., §1.3.4
  (pp. 21-22), book PDF from https://sites.engineering.ucsb.edu/~jbraw/mpc/ : A = [[4/3, −2/3], [1, 0]], B = [1; 0],
  C = [−2/3, 1], Q = C'C + 0.001 I, R = 0.001, Pf = Q; printed eig(A + B K_N(0)) = {1.307, 0.001} (N = 5),
  {0.989, 0.001} (N = 7), {0.664, 0.001} (N → ∞).
- Wood-Berry column model as printed in Alawad & Alseady (2020), Applied Computer Science 16(2):80-94, eqs. 2 and 5-6,
  doi:10.23743/acs-2020-14, https://ph.pollub.pl/index.php/acs/article/view/3154 (the text extraction of eq. 2 drops
  the «3» of G12's dead time; eq. 6 has e^{−9s} = e^{−(7+3−1)s}, which fixes θ12 = 3).
- Bloor et al. (2024), «PC-Gym: Benchmark Environments for Process Control Problems», arXiv:2410.22093,
  https://arxiv.org/html/2410.22093 : CSTR eqs. 14-15, §4.2 listing (tsim 25, SP 0.85/0.9/0.87, Tc in [295, 302],
  x0 [0.8, 330, 0.8], noise 0.001) and §4.2.1 (oracle horizon 17, Q = I, R = 0). The paper prints no CSTR parameter
  values; they are compared with the APMonitor «Stirred Reactor» page, https://apmonitor.com/pdc/index.php/Main/StirredReactor
  (q = 100, V = 100, ρ = 1000, Cp = 0.239, −ΔH = 5e4, E/R = 8750, k0 = 7.2e10, UA = 5e4, Tf = 350, CAf = 1).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import scipy.optimize as so

from packages.control import mpc
from packages.control import rl_runner as RL
from packages.dwsim_client import tools_advanced_control as TAC

FOPDT = {"gain": 2.0, "time_constants": [10.0], "dead_time": 2.5}
SOPDT_INV = {"gain": -1.5, "time_constants": [5.0, 2.0], "lead_time_constants": [-1.0], "dead_time": 0.7}
WB = [[{"gain": 12.8, "time_constants": [16.7], "dead_time": 1.0}, {"gain": -18.9, "time_constants": [21.0], "dead_time": 3.0}],
      [{"gain": 6.6, "time_constants": [10.9], "dead_time": 7.0}, {"gain": -19.4, "time_constants": [14.4], "dead_time": 3.0}]]
WB_KW = dict(sampling_period=1.0, prediction_horizon=40, control_horizon=8, simulation_time=150, move_weights=[2.0, 2.0],
             process_model=WB, setpoint_changes=[{"time": 2, "output": 0, "value": 1.0}], input_min=[-0.3, -0.3],
             input_max=[0.3, 0.12], input_rate_max=[0.05, 0.05], output_min=[None, -0.2], output_max=[None, 0.2])


def _dump(r: dict) -> None:
    json.dumps(r, allow_nan=False)


@pytest.fixture(scope="module")
def wb_run():
    return TAC.mpc_design_and_simulate(**WB_KW)


# ---------------------------------------------------------------------------------------------------------------------
# Part A — VALIDATED: Rawlings-Mayne-Diehl §1.3.4 (finite-horizon LQ destabilises a stable plant)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("N, book, stable", [(5, 1.307, False), (7, 0.989, True), (200, 0.664, True)])
def test_rmd_134_closed_loop_eigenvalues(N, book, stable):
    Q = [[4 / 9 + 0.001, -2 / 3], [-2 / 3, 1.001]]
    r = TAC.mpc_design_and_simulate(
        sampling_period=1.0, prediction_horizon=N, control_horizon=N, simulation_time=30, move_weights=0.0,
        input_weights=0.001, output_weights=0.0, state_weights=Q,
        state_space={"A": [[4 / 3, -2 / 3], [1, 0]], "B": [[1], [0]], "C": [[-2 / 3, 1]], "dt": 1.0},
        initial_state=[1.0, 1.0], disturbance_model="none", pid_baseline={"skip": True})
    nc = r["controller"]["nominal_closed_loop"]
    eig = sorted(abs(complex(*e)) for e in nc["regulator_eigenvalues"])
    # velocity form adds the u_prev state, whose eigenvalue is 0 when S = 0 (u = −K x exactly)
    assert eig[0] < 1e-9
    assert abs(eig[1] - 0.001) < 5e-4 and abs(eig[2] - book) < 5e-4          # book prints 3 decimals (measured ≤ 4.3e-4)
    assert nc["stable"] is stable
    assert r["evidence"]["verified"]
    if not stable:
        assert any("inestable" in w for w in r["warnings"])
    _dump(r)


# ---------------------------------------------------------------------------------------------------------------------
# Part A — VALIDATED (analytical/independent): long horizon → LQR; QP vs analytic KKT and SLSQP; first QP vs ipopt
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pm, Ts", [(FOPDT, 1.0), (SOPDT_INV, 0.5)])
def test_long_horizon_unconstrained_mpc_equals_dlqr(pm, Ts):
    import control as ct
    parsed = mpc.parse_model(pm, None)
    model = mpc.assemble(parsed["blocks"], 1, 1, Ts)
    inf = np.array([np.inf])
    S, N = 0.3, 120
    c = mpc.design(model, N, N, np.eye(1), S * np.eye(1), np.zeros((1, 1)), None, -inf, inf, inf, -inf, inf, np.ones(1),
                   "output", 1.0)
    # independent velocity-form LQR built here from the model: state (x, u_prev), input Δu, cost (Cx)² + S Δu²
    A, B, C = model["A"], model["B"], model["C"]
    n = A.shape[0]
    Aa = np.block([[A, B], [np.zeros((1, n)), np.eye(1)]])
    Ba = np.vstack([B, np.eye(1)])
    Ca = np.hstack([C, np.zeros((1, 1))])
    K, _, _ = ct.dlqr(Aa, Ba, Ca.T @ Ca, S * np.eye(1))
    assert np.linalg.norm(c["Kreg"] - K) / np.linalg.norm(K) < 1e-9          # measured ≤ 7e-13


def test_qp_solver_analytic_kkt_and_random_vs_slsqp():
    H, g, G, h = np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([-3.0, -1.0]), np.array([[1.0, 1.0]]), np.array([1.0])
    Hi = np.linalg.inv(H)
    lam = -(G @ Hi @ g + h) / (G @ Hi @ G.T)                     # one active constraint, closed form
    xa = -Hi @ (g + G.T @ lam.ravel())
    s = mpc.qp_solve(H, g, G, h)
    assert np.max(np.abs(s["x"] - xa)) < 1e-10 and abs(s["z"][0] - lam.item()) < 1e-10
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        n, m = int(rng.integers(2, 20)), int(rng.integers(1, 40))
        Mx = rng.normal(size=(n, n))
        H = Mx @ Mx.T + 10 ** rng.uniform(-3, 1) * np.eye(n)
        g = rng.normal(size=n) * 10 ** rng.uniform(-1, 1)
        G = rng.normal(size=(m, n))
        x0 = rng.normal(size=n)
        h = G @ x0 + rng.uniform(0.01, 1, size=m)
        s = mpc.qp_solve(H, g, G, h)
        assert s["converged"] and s["kkt"] < 1e-9
        ref = so.minimize(lambda x: 0.5 * x @ H @ x + g @ x, x0, jac=lambda x: H @ x + g, method="SLSQP",
                          constraints=[{"type": "ineq", "fun": lambda x: h - G @ x, "jac": lambda x: -G}],
                          options={"ftol": 1e-14, "maxiter": 2000})
        f = 0.5 * s["x"] @ H @ s["x"] + g @ s["x"]
        worst = max(worst, (f - ref.fun) / (1 + abs(ref.fun)))
    assert worst < 1e-7                                          # never worse than SLSQP beyond its tolerance


def test_wood_berry_first_qp_vs_independent_ipopt(wb_run):
    pyo = pytest.importorskip("pyomo.environ")
    try:
        import idaes  # noqa: F401  (puts ~/.idaes/bin on PATH)
    except ImportError:
        pass
    if not pyo.SolverFactory("ipopt").available(exception_flag=False):
        pytest.skip("ipopt no disponible")
    pl = wb_run["prediction_at_first_event"]
    P, M, k0, S, rho = 40, 8, 2, [2.0, 2.0], 1e4
    G = [[(12.8, 16.7, 1), (-18.9, 21.0, 3)], [(6.6, 10.9, 7), (-19.4, 14.4, 3)]]
    m = pyo.ConcreteModel()
    T = k0 + P
    m.u = pyo.Var(range(T), [0, 1])
    m.y = pyo.Var(range(T + 1), [0, 1], [0, 1])
    m.eps = pyo.Var(bounds=(0, None))
    m.c = pyo.ConstraintList()
    umin, umax, dumax = [-0.3, -0.3], [0.3, 0.12], [0.05, 0.05]
    for i in range(2):                      # each element FOPDT sampled exactly (ZOH, integer dead time)
        for j in range(2):
            K, tau, d = G[i][j]
            a = math.exp(-1.0 / tau)
            m.y[0, i, j].fix(0.0)
            for t in range(T):
                m.c.add(m.y[t + 1, i, j] == a * m.y[t, i, j] + K * (1 - a) * (m.u[t - d, j] if t - d >= 0 else 0.0))
    for t in range(T):
        for j in range(2):
            if t < k0:
                m.u[t, j].fix(0.0)
                continue
            m.u[t, j].setlb(umin[j])
            m.u[t, j].setub(umax[j])
            if t >= k0 + M:
                m.c.add(m.u[t, j] == m.u[t - 1, j])
            else:
                m.c.add(m.u[t, j] - m.u[t - 1, j] <= dumax[j])
                m.c.add(m.u[t - 1, j] - m.u[t, j] <= dumax[j])

    def yo(t, i):
        return m.y[t, i, 0] + m.y[t, i, 1]
    for t in range(k0 + 1, T + 1):
        m.c.add(yo(t, 1) <= 0.2 + m.eps)
        m.c.add(yo(t, 1) >= -0.2 - m.eps)
    m.obj = pyo.Objective(expr=sum((yo(t, i) - [1.0, 0.0][i]) ** 2 for t in range(k0 + 1, T + 1) for i in range(2))
                          + sum(S[j] * (m.u[t, j] - m.u[t - 1, j]) ** 2 for t in range(k0, k0 + M) for j in range(2))
                          + rho * m.eps ** 2)
    res = pyo.SolverFactory("ipopt").solve(m, options={"tol": 1e-12, "bound_relax_factor": 0.0, "constr_viol_tol": 1e-12})
    assert str(res.solver.termination_condition) == "optimal"
    ui = np.array([[pyo.value(m.u[t, j]) for j in range(2)] for t in range(k0, k0 + M)])
    assert pl["time"] == 2.0
    assert np.max(np.abs(ui - np.array(pl["u_planned"]))) < 1e-8          # measured 2.7e-12
    assert np.max(np.abs(np.array(pl["u_planned"])[:, 0] - [0.05, 0.1, 0.15, 0.2, 0.25, 0.2797, 0.2297, 0.1797])) < 1e-4


def test_wood_berry_published_model_and_closed_loop(wb_run):
    r = wb_run
    G0 = np.array(r["model"]["steady_state_gain"])
    assert np.max(np.abs(G0 - [[12.8, -18.9], [6.6, -19.4]])) < 1e-9
    lam = 1 / (1 - (-18.9 * 6.6) / (12.8 * -19.4))                       # analytical RGA from the printed gains
    assert abs(r["model"]["relative_gain_array"][0][0] - lam) < 1e-9 and abs(lam - 2.0094) < 1e-4
    assert abs(6.6 * 18.9 / 12.8 - 9.745) < 1e-3                         # the paper's eq. 6 coefficient, same data
    assert r["model"]["dead_time_states"] == 10                          # input 0: lags up to 7; input 1: up to 3
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert ev["qp"]["max_kkt_residual"] < 1e-9 and ev["hard_constraints_max_rel_violation"] < 1e-8
    u = np.array(r["trajectories"]["mpc_u"])
    du = np.diff(np.hstack([np.zeros((2, 1)), u]), axis=1)
    assert np.all(u[0] <= 0.3 + 1e-9) and np.all(u[1] <= 0.12 + 1e-9) and np.all(u >= -0.3 - 1e-9)
    assert np.max(np.abs(du)) <= 0.05 + 1e-9
    assert r["mpc"]["constraint_activity"]["soft_outputs"][0]["max_violation"] <= 1e-6
    assert max(abs(v) for v in r["mpc"]["indices"]["final_error"]) < 1e-5     # offset-free
    assert r["comparison"]["IAE_mpc_over_pid"][0] < 1 and r["comparison"]["IAE_mpc_over_pid"][1] < 1
    _dump(r)


# ---------------------------------------------------------------------------------------------------------------------
# Part A — PID baseline vs the repo's own closed-loop simulator (consistency)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["setpoint", "load"])
def test_pid_baseline_matches_repo_loop_simulator(kind):
    from packages.dynamics import loop as LP
    from packages.dynamics import tuning as TU
    from packages.dynamics.process import Process, half_rule
    p = Process(2.0, [10.0], [], 2.5)
    hr = half_rule(p, 1)
    c = TU.tune({k: hr[k] for k in ("k", "theta", "integrating", "tau1", "tau2")}, "PI", ["SIMC"], None, TU.ultimate(p))[0]["controller"]
    sp = kind == "setpoint"
    s = LP.simulate(p, c, 0.01, 1.0 if sp else 0.0, 0.0 if sp else 1.0, t_end=300.0)
    ref = LP.setpoint_metrics(s, 1.0) if sp else LP.load_metrics(s, 1.0)
    r = TAC.mpc_design_and_simulate(
        sampling_period=1.0, prediction_horizon=40, control_horizon=10, simulation_time=300, move_weights=0.5,
        process_model=FOPDT, setpoint_changes=[{"time": 0, "value": 1.0}] if sp else None,
        disturbances=None if sp else [{"time": 0, "type": "input", "value": 1.0}])
    pi = r["pid_baseline"]["indices"]
    assert r["pid_baseline"]["loops"][0]["controller"]["Kc"] == pytest.approx(c.Kc)
    assert abs(pi["IAE"][0] / ref["IAE"] - 1) < (0.01 if sp else 1e-6)         # measured 0.87 % / 4e-8
    if sp:   # TV here counts the jump from rest u(0−) = 0 → Kc·r that the repo's TV (diff of u from t = 0) omits
        assert abs((pi["TV_u"][0] - ref["TV_u"]) - c.Kc * 1.0) < 0.02 * c.Kc    # measured 0.77 %
    else:
        assert abs(pi["TV_u"][0] / ref["TV_u"] - 1) < 0.02                     # measured 1.4 %
    assert r["pid_baseline"]["evidence"]["verified"]
    _dump(r)


# ---------------------------------------------------------------------------------------------------------------------
# Part A — CONSISTENCY: linearity, monotonicity, exact dead time, offset-free, integrating plants, mismatch
# ---------------------------------------------------------------------------------------------------------------------
def test_linearity_and_move_weight_monotonicity():
    rows = []
    for S in (0.1, 0.5, 2.0, 8.0):
        res = [TAC.mpc_design_and_simulate(sampling_period=1.0, prediction_horizon=40, control_horizon=10,
                                           simulation_time=150, move_weights=S, process_model=FOPDT,
                                           setpoint_changes=[{"time": 0, "value": a}], pid_baseline={"skip": True})
               for a in (1.0, 2.0)]
        i1, i2 = res[0]["mpc"]["indices"], res[1]["mpc"]["indices"]
        assert i2["IAE"][0] == pytest.approx(2 * i1["IAE"][0], rel=1e-10)          # unconstrained MPC is linear
        assert i2["ISE"][0] == pytest.approx(4 * i1["ISE"][0], rel=1e-10)
        rows.append((i1["IAE"][0], i1["TV_u"][0]))
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(rows, rows[1:]))       # more move weight: slower, smoother


def test_fractional_dead_time_two_grids_and_input_disturbance_model():
    r = TAC.mpc_design_and_simulate(sampling_period=0.5, prediction_horizon=60, control_horizon=15, simulation_time=120,
                                    move_weights=0.2, process_model=SOPDT_INV, setpoint_changes=[{"time": 1, "value": 1.0}],
                                    disturbances=[{"time": 60, "type": "output", "value": 0.3}], disturbance_model="input")
    md = r["evidence"]["model_discretization"]
    assert md["two_grid_step_response_max_rel_diff"] < 1e-10 and r["evidence"]["verified"]
    assert md.get("closed_form_step_response_max_rel_diff", 0.0) < 1e-8
    assert abs(r["mpc"]["indices"]["final_error"][0]) < 1e-6
    assert r["evidence"]["IAE_step_halving_rel_diff"] < 2e-3
    _dump(r)


def test_integrating_process_and_plant_mismatch():
    r = TAC.mpc_design_and_simulate(sampling_period=1.0, prediction_horizon=40, control_horizon=10, simulation_time=200,
                                    move_weights=1.0, process_model={"gain": 0.2, "time_constants": [], "dead_time": 2.0,
                                                                     "integrating": True},
                                    setpoint_changes=[{"time": 0, "value": 1.0}],
                                    disturbances=[{"time": 80, "type": "input", "value": 0.5}])
    assert r["controller"]["disturbance_model"] == "input" and r["evidence"]["verified"]
    assert abs(r["mpc"]["indices"]["final_error"][0]) < 1e-6 and not r["model"]["open_loop_stable"]
    r2 = TAC.mpc_design_and_simulate(sampling_period=0.5, prediction_horizon=60, control_horizon=15, simulation_time=120,
                                     move_weights=0.2, process_model=SOPDT_INV, setpoint_changes=[{"time": 1, "value": 1.0}],
                                     plant_model={"gain": -1.8, "time_constants": [6.0, 2.0], "lead_time_constants": [-1.0],
                                                  "dead_time": 1.0})
    assert r2["evidence"]["verified"] and abs(r2["mpc"]["indices"]["final_error"][0]) < 1e-4   # offset-free despite mismatch
    assert any("plant_model" in a for a in r2["assumptions"])
    _dump(r)
    _dump(r2)


def test_complex_poles_tf_equals_state_space_with_input_delay():
    tf = TAC.mpc_design_and_simulate(sampling_period=0.2, prediction_horizon=50, control_horizon=10, simulation_time=30,
                                     move_weights=0.1, process_model={"num": [1.0], "den": [1.0, 0.6, 1.0], "dead_time": 0.3},
                                     setpoint_changes=[{"time": 0, "value": 1.0}])
    st = TAC.mpc_design_and_simulate(sampling_period=0.2, prediction_horizon=50, control_horizon=10, simulation_time=30,
                                     move_weights=0.1, state_space={"A": [[0, 1], [-1, -0.6]], "B": [[0], [1]],
                                                                    "C": [[1, 0]], "input_delays": [0.3]},
                                     setpoint_changes=[{"time": 0, "value": 1.0}])
    assert tf["mpc"]["indices"]["IAE"][0] == pytest.approx(st["mpc"]["indices"]["IAE"][0], rel=1e-9)
    assert "skipped_reason" in tf["pid_baseline"]                         # complex poles: no FOPDT rule applies
    _dump(tf)
    _dump(st)


# ---------------------------------------------------------------------------------------------------------------------
# refusals (no datum is defaulted)
# ---------------------------------------------------------------------------------------------------------------------
BASE = dict(sampling_period=1.0, prediction_horizon=30, control_horizon=5, simulation_time=60, move_weights=0.5,
            process_model=FOPDT, setpoint_changes=[{"time": 0, "value": 1.0}])


@pytest.mark.parametrize("change, word", [
    ({"sampling_period": None}, "sampling_period"), ({"prediction_horizon": None}, "horizon"),
    ({"simulation_time": None}, "simulation_time"), ({"move_weights": None}, "move_weights"),
    ({"process_model": None}, "modelo"), ({"setpoint_changes": None}, "escenario"),
    ({"input_min": 0.1}, "contener el 0"), ({"control_horizon": 40}, "control_horizon"),
    ({"process_model": {"gain": 2.0, "time_constants": [10.0]}}, "dead_time")])
def test_mpc_refuses_missing_or_invalid_inputs(change, word):
    with pytest.raises(ValueError, match=word):
        TAC.mpc_design_and_simulate(**{**BASE, **change})


@pytest.mark.parametrize("change, word", [
    ({"total_timesteps": None}, "total_timesteps"), ({"seeds": None}, "seeds"),
    ({"scenario": None, "environment": "cstr"}, "env_params"), ({"seeds": [1, 1_000_000]}, "seeds"),
    ({"pid_baseline": [{"state": "Ca", "Kc": -60.0, "tau_I": 1.15}]}, "bias"),
    ({"pid_baseline": [{"state": "Ca", "Kc": -60.0, "bias": 301.4}]}, "tau_I"),
    ({"pid_baseline": [{"state": "Ca", "Kc": -60.0, "tau_I": 1.15, "bias": 310.0}]}, "a_space"),
    ({"algorithm": "DQN"}, "algorithm")])
def test_rl_refuses_missing_inputs_without_running(change, word):
    kw = {**dict(scenario="bloor2024_cstr", algorithm="PPO", total_timesteps=1000, seeds=[1]), **change}
    with pytest.raises(ValueError, match=word):
        TAC.rl_control_experiment(**kw)


def test_rl_absent_environment_message(monkeypatch):
    monkeypatch.setattr(RL, "rl_available", lambda: False)
    r = TAC.rl_control_experiment(scenario="bloor2024_cstr", total_timesteps=1000, seeds=[1, 2, 3])
    assert r["available"] is False and not r["evidence"]["verified"]
    assert any("uv venv --python 3.11" in c for c in r["create_commands"]) and "ALCHEMAI_RL_PYTHON" in r["message"]
    _dump(r)
    with pytest.raises(ValueError, match="total_timesteps"):
        TAC.rl_control_experiment(scenario="bloor2024_cstr", seeds=[1])


# ---------------------------------------------------------------------------------------------------------------------
# Part B — needs ~/tools/alchemai-rl/.venv (skipped otherwise); tiny budget, about 10-15 s
# ---------------------------------------------------------------------------------------------------------------------
needs_rl = pytest.mark.skipif(not RL.rl_available(), reason="entorno de RL aparte ausente (~/tools/alchemai-rl/.venv)")
# SIMC PI for Ca ← Tc, tuned with the repo's own tools: fopdt_identify on a −1 K step of Tc from the steady state
# Ca = 0.85 (Tc = 301.44 K, independent scipy integration of eqs. 14-15) gives K = −0.02022, τ = 1.153, θ = 0.475;
# pid_tuning SIMC (τc = θ) gives Kc = −60.01 K·m³/mol, τI = 1.153 (time unit of the environment).
CSTR_PI = [{"state": "Ca", "input": 0, "Kc": -60.01, "tau_I": 1.1529, "tau_D": 0.0, "bias": 301.44}]


@needs_rl
def test_rl_tiny_experiment_protocol_and_published_model(tmp_path):
    r = TAC.rl_control_experiment(scenario="bloor2024_cstr", algorithm="PPO", total_timesteps=2048, seeds=[1],
                                  eval_episodes=3, hyperparameters={"n_steps": 256, "batch_size": 64},
                                  pid_baseline=CSTR_PI, output_dir=str(tmp_path), timeout_s=300)
    assert r["available"] and r["evidence"]["verified"], r["evidence"]["reasons"]
    rep = r["reproducibility"]
    assert rep["same_policy_parameters"] and rep["eval_return_max_abs_diff"] == 0.0          # bit-for-bit
    msc = r["model_step_check"]
    assert msc["max_rel_diff"] < 1e-4                                   # env step vs eqs. 14-15 (measured 2.6e-6)
    apm = {"q": 100, "V": 100, "rho": 1000, "C": 0.239, "deltaHr": -5e4, "EA_over_R": 8750, "k0": 7.2e10, "UA": 5e4,
           "Ti": 350, "Caf": 1}
    assert all(msc["parameters"][k] == pytest.approx(v, rel=1e-12) for k, v in apm.items())
    assert "error" not in r["oracle"], r["oracle"]
    assert r["evidence"]["oracle_replay_state_max_rel_diff"] < 1e-3
    rnd, orc, pid = r["random_policy"]["return"]["mean"], r["oracle"]["return"], r["pid_baseline"]["return"]["mean"]
    assert orc > pid > rnd                                               # measured −0.0117 > −0.0147 > −0.0458
    assert r["beats_random"]["seeds_better_than_random"] == 1           # measured −0.0265 even at 2048 steps
    assert r["gap_to_oracle"]["absolute"] > 0 and r["vs_pid"]["seeds_better_than_pid"] == 0
    assert r["warnings"][0] == RL.STANDING_WARNING
    assert all(p.endswith(".zip") and (tmp_path / p.split("/")[-1]).exists() for p in r["policy_paths"])
    assert r["versions"]["pcgym"] == "0.1.8"
    _dump(r)


@needs_rl
def test_rl_timeout_kills_process_group(tmp_path):
    job, _ = RL.build_job(environment=None, scenario="bloor2024_cstr", env_params=None, algorithm="PPO",
                          total_timesteps=200_000, seeds=[1], eval_episodes=1, hyperparameters=None, oracle=False,
                          oracle_horizon=None, check_reproducibility=False, output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="no terminó"):
        RL.run_job(job, timeout_s=3)


# =====================================================================================================================
# Nonlinear MPC (packages.control.nmpc via nmpc_design_and_simulate)
# Further sources read on 2026-11-14 in the same Rawlings-Mayne-Diehl PDF: Example 2.5 (pp. 99-101: x+ = x + u,
# ℓ = (x² + u²)/2, Vf = x²/2, |u| ≤ 1, N = 2; printed: u⁰(10) = (−1, −1), x⁰ = 10, 9, 8, V⁰_N(10) = 124, closed-loop law
# κ_N(x) = −sat(3x/5)) and Example 8.1 (pp. 489-490: x+ = x + u − 2u², x0 = 1, N = 1, cost x0² + u0² + 10 x1²,
# reduced objective ψ(u) = 11 + 20u − 29u² − 40u³ + 40u⁴ with two local minima).
# =====================================================================================================================
from packages.control import nmpc as NM  # noqa: E402

CSTR_PARAMS = {"q": 100, "V": 100, "rho": 1000, "C": 0.239, "deltaHr": -5e4, "EA_over_R": 8750, "k0": 7.2e10, "UA": 5e4,
               "Ti": 350, "Caf": 1}
CSTR_MODEL = {"states": ["Ca", "T"], "inputs": ["Tc"], "parameters": CSTR_PARAMS,
              "rhs": {"Ca": "q/V*(Caf - Ca) - k0*exp(-EA_over_R/T)*Ca",
                      "T": "q/V*(Ti - T) - deltaHr/(rho*C)*k0*exp(-EA_over_R/T)*Ca + UA/(rho*C*V)*(Tc - T)"}}
INTEGRATOR = {"time": "discrete", "states": ["x"], "inputs": ["u"], "rhs": {"x": "x + u"}}


def _cstr_f(x, u):          # the same ODEs written directly in Python (independent of the expression parser)
    p = CSTR_PARAMS
    k = p["k0"] * np.exp(-p["EA_over_R"] / x[1])
    return np.array([p["q"] / p["V"] * (p["Caf"] - x[0]) - k * x[0],
                     p["q"] / p["V"] * (p["Ti"] - x[1]) - p["deltaHr"] / (p["rho"] * p["C"]) * k * x[0]
                     + p["UA"] / (p["rho"] * p["C"] * p["V"]) * (u[0] - x[1])])


def _cstr_steady(ca: float) -> tuple[np.ndarray, np.ndarray]:
    p = CSTR_PARAMS
    k = p["q"] / p["V"] * (1 - ca) / ca
    T = p["EA_over_R"] / math.log(p["k0"] / k)
    Tc = T - (p["q"] / p["V"] * (p["Ti"] - T) - p["deltaHr"] / (p["rho"] * p["C"]) * k * ca) / (p["UA"] / (p["rho"] * p["C"] * p["V"]))
    return np.array([ca, T]), np.array([Tc])


@pytest.mark.parametrize("s", [1, 2, 3, 4])
def test_radau_iia_order_conditions(s):
    c, A = NM.radau_iia(s)
    b = A[-1]                                                             # stiffly accurate: b = last row
    assert c[-1] == 1.0
    for m in range(1, 2 * s):                                             # quadrature order 2s − 1: B(2s−1)
        assert abs(b @ c ** (m - 1) - 1 / m) < 1e-13
    for m in range(1, s + 1):                                             # stage order s: C(s)
        assert np.max(np.abs(A @ c ** (m - 1) - c ** m / m)) < 1e-13


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "x.real", "x[0]", "(lambda: 1)()", "x > 1",
                                  "abs(x)", "exp(x, 2)", "y + 1", "'a'", "x if x else u", "True", "exp(x=1)"])
def test_expression_parser_refuses_anything_but_arithmetic(expr):
    with pytest.raises(ValueError):
        NM.parse_model({"states": ["x"], "inputs": ["u"], "rhs": {"x": expr}})


def test_expression_parser_evaluates_like_python():
    pm = NM.parse_model({"states": ["x"], "inputs": ["u"], "parameters": {"a": 2.0},
                         "rhs": {"x": "-a*x^2 + sqrt(u)*exp(-x/a) - log10(u)"}})
    v = NM.evaluate(pm["trees"][0], {"x": 0.7, "u": 3.0, "a": 2.0}, NM.NUMPY_FUNCS)
    assert v == pytest.approx(-2 * 0.7 ** 2 + math.sqrt(3) * math.exp(-0.35) - math.log10(3), rel=1e-15)


def test_rmd_example_2_5_closed_loop_nmpc_and_linear_mpc():
    xs, us = [10.0], []
    for _ in range(14):                                                    # printed law κ_N(x) = −sat(3x/5)
        us.append(-max(-1.0, min(1.0, 3 * xs[-1] / 5)))
        xs.append(xs[-1] + us[-1])
    r = TAC.nmpc_design_and_simulate(model=INTEGRATOR, sampling_period=1.0, prediction_horizon=2, simulation_time=14,
                                     initial_state={"x": 10.0}, setpoints={"x": 0.0}, state_weights={"x": 0.5},
                                     input_weights={"u": 0.5}, input_reference={"u": 0.0}, input_min={"u": -1},
                                     input_max={"u": 1})
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert np.max(np.abs(np.array(r["trajectories"]["states"]["x"]) - xs)) < 1e-8          # measured 6.7e-12
    assert np.max(np.abs(np.array(r["trajectories"]["inputs"]["u"]) - us)) < 1e-8
    assert np.max(np.abs(np.array(r["nmpc"]["first_plan"]["u_plan"]) + 1)) < 1e-8          # u⁰(10) = (−1, −1)
    V = 0.5 * 10.0 ** 2 + r["nmpc"]["first_plan"]["objective"]          # book's V_N includes ℓ(x0) = x0²/2
    assert abs(V - 123.5) < 1e-6 and abs(V - 124) <= 0.5 + 1e-6         # exact 123.5; printed «124» (rounded up)
    lin = TAC.mpc_design_and_simulate(sampling_period=1.0, prediction_horizon=2, control_horizon=2, simulation_time=14,
                                      move_weights=0.0, input_weights=0.5, output_weights=0.0, state_weights=[[0.5]],
                                      state_space={"A": [[1]], "B": [[1]], "C": [[1]], "dt": 1.0}, initial_state=[10.0],
                                      input_min=-1, input_max=1, disturbance_model="none", pid_baseline={"skip": True})
    assert np.max(np.abs(np.array(lin["trajectories"]["mpc_y"][0]) - xs[:14])) < 1e-10     # measured 5.8e-15
    assert np.max(np.abs(np.array(lin["trajectories"]["mpc_u"][0]) - us)) < 1e-10
    _dump(r)


def test_rmd_example_8_1_two_local_minima():
    psi = np.polynomial.Polynomial([11, 20, -29, -40, 40])              # the book's reduced objective
    st = sorted(z.real for z in psi.deriv().roots() if abs(z.imag) < 1e-12 and -1 <= z.real <= 1)
    glob, loc = st[0], st[-1]                                           # −0.4944 (ψ 1.2472) and 0.9888 (ψ 1.9888)
    assert psi(glob) < psi(loc)
    model = {"time": "discrete", "states": ["x"], "inputs": ["u"], "rhs": {"x": "x + u - 2*u**2"}}
    found = {}
    for guess in (-1.0, 1.0):                                           # initial_input = initial guess (no Δu cost)
        r = TAC.nmpc_design_and_simulate(model=model, sampling_period=1.0, prediction_horizon=1, simulation_time=1,
                                         initial_state=[1.0], setpoints={"x": 0.0}, state_weights={"x": 10.0},
                                         input_weights={"u": 1.0}, input_reference={"u": 0.0}, input_min=[-1],
                                         input_max=[1], initial_input=[guess])
        u = r["nmpc"]["first_plan"]["u_plan"][0][0]
        assert abs((1.0 + r["nmpc"]["first_plan"]["objective"]) - psi(u)) < 1e-9
        found[guess] = u
        assert any("LOCALES" in w for w in r["warnings"])
    assert abs(found[-1.0] - glob) < 1e-7 and abs(found[1.0] - loc) < 1e-7   # ipopt is local: the start decides


@needs_rl
def test_nmpc_matches_pcgym_do_mpc_oracle_on_the_cstr(tmp_path):
    o = RL.oracle_trajectory(scenario="bloor2024_cstr", output_dir=str(tmp_path))
    assert o["discretization"] == {"method": "collocation", "collocation_type": "radau", "collocation_deg": 2,
                                   "collocation_ni": 1} and o["horizon"] == 17 and o["R_diag"] == [0.0]
    dt = o["dt"]
    SP = [0.85] * 20 + [0.9] * 20 + [0.87] * 20
    sp, t = [], 0.0
    for _ in range(60):    # pc-gym's p_fun: SP[int(t/dt − 1)] with do-mpc's accumulated float time (lags one step)
        sp.append(SP[min(int(t / dt - 1), 59)])
        t += dt
    r = TAC.nmpc_design_and_simulate(model=CSTR_MODEL, sampling_period=dt, prediction_horizon=17, simulation_time=25,
                                     initial_state=[0.8, 330.0], setpoints={"Ca": sp}, state_weights={"Ca": 1.0},
                                     input_min=[295], input_max=[302], collocation_points=2, elements_per_sample=1)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    xo, uo = np.array(o["x_log"]).T, np.array(o["u_log"][0])
    X = np.array([r["trajectories"]["states"]["Ca"][1:], r["trajectories"]["states"]["T"][1:]]).T
    U = np.array(r["trajectories"]["inputs"]["Tc"])
    assert np.max(np.abs(X[:, 0] - xo[:, 0])) < 1e-4                    # measured 9.5e-6 mol/m³
    assert np.max(np.abs(X[:, 1] - xo[:, 1])) < 0.05                    # measured 7.9e-3 K
    assert np.max(np.abs(U - uo)) < 0.1                                 # measured 0.018 K (range 7 K)
    J_o = float(np.sum((xo[:, 0] - np.array(sp)) ** 2))
    assert abs(r["nmpc"]["indices"]["closed_loop_cost"] / J_o - 1) < 1e-3     # measured −1.8e-4
    _dump(r)


def test_nmpc_tends_to_linear_mpc_for_small_steps():
    xs, us = _cstr_steady(0.85)
    A, B = np.zeros((2, 2)), np.zeros((2, 1))
    for i in range(2):                                                  # central differences of the Python ODEs
        e = np.zeros(2)
        e[i] = 1e-6 * max(1.0, abs(xs[i]))
        A[:, i] = (_cstr_f(xs + e, us) - _cstr_f(xs - e, us)) / (2 * e[i])
    B[:, 0] = (_cstr_f(xs, us + 1e-4) - _cstr_f(xs, us - 1e-4)) / 2e-4
    dt, n, errs = 25 / 60, 30, []
    for d in (1e-3, 1e-2):
        lin = TAC.mpc_design_and_simulate(sampling_period=dt, prediction_horizon=17, control_horizon=17,
                                          simulation_time=n * dt, move_weights=1.0, output_weights=1e4,
                                          state_space={"A": A.tolist(), "B": B.tolist(), "C": [[1, 0]]},
                                          setpoint_changes=[{"time": 0, "value": d}], disturbance_model="none",
                                          pid_baseline={"skip": True})
        nl = TAC.nmpc_design_and_simulate(model=CSTR_MODEL, sampling_period=dt, prediction_horizon=17,
                                          simulation_time=n * dt, initial_state=xs.tolist(), initial_input=us.tolist(),
                                          setpoints={"Ca": xs[0] + d}, state_weights={"Ca": 1e4}, move_weights={"Tc": 1.0})
        assert nl["evidence"]["verified"] and lin["evidence"]["verified"]
        yl, ul = np.array(lin["trajectories"]["mpc_y"][0]), np.array(lin["trajectories"]["mpc_u"][0])
        yn = np.array(nl["trajectories"]["states"]["Ca"][:n]) - xs[0]
        un = np.array(nl["trajectories"]["inputs"]["Tc"]) - us[0]
        errs.append((np.max(np.abs(yl - yn)) / d, np.max(np.abs(ul - un)) / np.max(np.abs(ul))))
    assert errs[0][0] < 3e-3 and errs[0][1] < 2e-2                      # measured 1.6e-3 and 9.1e-3 at δ = 1e-3
    assert 8 < errs[1][0] / errs[0][0] < 12 and 8 < errs[1][1] / errs[0][1] < 12   # deviation ∝ δ (second order)


def test_nmpc_hard_state_limit_and_terminal_region():
    xs, us = _cstr_steady(0.85)
    r = TAC.nmpc_design_and_simulate(model=CSTR_MODEL, sampling_period=25 / 60, prediction_horizon=10, simulation_time=5,
                                     initial_state=xs.tolist(), initial_input=us.tolist(), setpoints={"Ca": 0.80},
                                     state_weights={"Ca": 1e4}, move_weights={"Tc": 1.0}, input_min=[295],
                                     input_max=[310], state_max={"T": 329.0})
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    assert max(r["trajectories"]["states"]["T"]) <= 329.0 * (1 + 1e-6)   # Ca = 0.80 needs T ≈ 331.6 K: unreachable
    assert r["nmpc"]["indices"]["final_error"][0] < -0.01                 # so Ca stays above its setpoint
    ok = TAC.nmpc_design_and_simulate(model=INTEGRATOR, sampling_period=1.0, prediction_horizon=2, simulation_time=3,
                                      initial_state=[1.5], setpoints={"x": 0.0}, state_weights={"x": 0.5},
                                      input_min=[-1], input_max=[1], terminal_tolerance={"x": 0.0})
    assert abs(ok["nmpc"]["first_plan"]["x_pred"][-1][0]) < 1e-8 and ok["evidence"]["verified"]
    bad = TAC.nmpc_design_and_simulate(model=INTEGRATOR, sampling_period=1.0, prediction_horizon=2, simulation_time=3,
                                       initial_state=[10.0], setpoints={"x": 0.0}, state_weights={"x": 0.5},
                                       input_min=[-1], input_max=[1], terminal_tolerance={"x": 0.0})
    assert not bad["evidence"]["verified"] and bad["nmpc"]["stopped"]["step"] == 0     # |u| ≤ 1 cannot reach 0 in 2
    _dump(r)
    _dump(bad)


NM_BASE = dict(model=INTEGRATOR, sampling_period=1.0, prediction_horizon=2, simulation_time=3, initial_state=[1.0],
               setpoints={"x": 0.0}, state_weights={"x": 1.0})


@pytest.mark.parametrize("change, word", [
    ({"model": None}, "model"), ({"sampling_period": None}, "sampling_period"), ({"initial_state": None}, "initial_state"),
    ({"setpoints": {"z": 0.0}}, "desconocidos"), ({"state_weights": None}, "state_weights"),
    ({"move_weights": {"u": 1.0}}, "initial_input"), ({"input_weights": {"u": 1.0}}, "input_reference"),
    ({"model": {"states": ["x"], "inputs": ["u"], "rhs": {"x": "x + u + k"}}}, "k"),
    ({"model": {"states": ["x"], "inputs": ["u"], "rhs": {}}}, "rhs")])
def test_nmpc_refuses_missing_inputs(change, word):
    with pytest.raises(ValueError, match=word):
        TAC.nmpc_design_and_simulate(**{**NM_BASE, **change})
