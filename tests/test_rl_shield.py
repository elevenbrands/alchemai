"""Validation of the RL safety shield and the per-policy verification (bloque M25-L5; engines `packages.control.rl_runner`
and `packages.control.rl_worker`, tool `rl_control_experiment`).

Sources and references:
- Analytical: pc-gym's `first_order_system` is dx/dt = (K·u − x)/τ with K = 1 and τ = 0.5 (dataclass defaults of
  `pcgym/model_classes.py` in the installed pcgym 0.1.8, read 2026-11-15 in ~/tools/alchemai-rl/.venv). With a constant
  input u the exact sampled solution is x_{k+1} = u + (x_k − u)·a, a = exp(−dt/τ). With dt = tsim/N = 0.1, x0 = 0 and the
  constant policy u = 1 against the constraint x ≤ 0.6: x_k = 1 − a^k, so the unshielded episode (k = 1..20) violates in
  exactly 16 steps (k ≥ 5, since a^k < 0.4 ⇔ k > 4.58) with maximum violation 1 − a^20 − 0.6; a one-step shield must
  intervene first at t = 4 (x_4 = 1 − a^4, predicted x_5 = 1 − a^5 > 0.6), and at every step it must intervene iff
  1 + (x_t − 1)·a > 0.6. With a rate limit of 0.3 per step from u = 0 the first applied input is 0.3 and x_1 = 0.3·(1 − a).
- CSTR: the published pc-gym scenario of Bloor et al. (2024), arXiv:2410.22093 (already validated in
  tests/test_advanced_control.py); the shield results on it are MEASURED here (no external answer exists for them).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from packages.control import rl_runner as RL
from packages.dwsim_client import tools_advanced_control as TAC

needs_rl = pytest.mark.skipif(not RL.rl_available(), reason="entorno de RL aparte ausente (~/tools/alchemai-rl/.venv)")
CSTR_PI = [{"state": "Ca", "input": 0, "Kc": -60.01, "tau_I": 1.1529, "tau_D": 0.0, "bias": 301.44}]
CSTR_CONS = {"constraints": {"T": [326.0]}, "cons_type": {"T": [">="]}}
BASE = dict(scenario="bloor2024_cstr", algorithm="PPO", total_timesteps=1000, seeds=[1], env_params=CSTR_CONS)


# ---------------------------------------------------------------------------------------------------------------------
# Offline: refusals, the verdict rule, backwards compatibility
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("change, word", [
    ({"env_params": None, "shield": {"fallback": "nmpc"}}, "restricciones"),
    ({"shield": {}}, "fallback"),
    ({"shield": {"fallback": "lqr"}}, "fallback"),
    ({"shield": {"fallback": "pid"}}, "pid_baseline"),
    ({"shield": {"fallback": "nmpc", "input_rate_max": [1.0]}}, "initial_input"),
    ({"shield": {"fallback": "nmpc", "input_rate_max": [0.0], "initial_input": [300.0]}}, "input_rate_max"),
    ({"shield": {"fallback": "nmpc", "initial_input": [310.0]}}, "a_space"),
    ({"shield": {"fallback": "nmpc", "horizon": 0}}, "horizon"),
    ({"shield": {"fallback": "nmpc", "constraint_margin": {"Ca": 0.01}}}, "constraint_margin"),
    ({"shield": {"fallback": "nmpc", "safety": True}}, "no admitidas"),
    ({"verification": {"episodes_per_case": 2}}, "falta shield"),
    ({"shield": {"fallback": "nmpc"}, "verification": {"disturbances": [{"Ti": 350.0}]}}, "disturbance_bounds"),
    ({"shield": {"fallback": "nmpc"}, "verification": {"disturbances": [{"Ti": [350.0] * 10}],
                                                       "disturbance_bounds": {"Ti": [340, 360]}}}, "N = 60"),
    ({"shield": {"fallback": "nmpc"}, "verification": {"disturbances": [{"Ti": 370.0}],
                                                       "disturbance_bounds": {"Ti": [340, 360]}}}, "sale de"),
    ({"shield": {"fallback": "nmpc"}, "verification": {"initial_states": [[0.8, 330.0]]}}, "initial_states"),
    ({"shield": {"fallback": "nmpc"}, "verification": {"cost_tolerance_pct": 0}}, "cost_tolerance_pct"),
    ({"shield": {"fallback": "nmpc"}, "verification": {"episodes_per_case": 201}}, "episodes_per_case"),
])
def test_shield_and_verification_refuse_missing_or_invalid_inputs(change, word):
    with pytest.raises(ValueError, match=word):
        TAC.rl_control_experiment(**{**BASE, **change})


def test_absent_environment_still_validates_shield(monkeypatch):
    monkeypatch.setattr(RL, "rl_available", lambda: False)
    r = TAC.rl_control_experiment(**BASE, shield={"fallback": "nmpc"}, verification={"cost_tolerance_pct": 20})
    assert r["available"] is False and not r["evidence"]["verified"]
    json.dumps(r)
    with pytest.raises(ValueError, match="fallback"):
        TAC.rl_control_experiment(**BASE, shield={"horizon": 3})


def test_job_without_shield_is_unchanged_and_with_shield_carries_defaults():
    kw = dict(environment=None, scenario="bloor2024_cstr", env_params=CSTR_CONS, algorithm="PPO", total_timesteps=10,
              seeds=[1], eval_episodes=3, hyperparameters=None, oracle=True, oracle_horizon=None,
              check_reproducibility=True, output_dir=None)
    job, _ = RL.build_job(**kw)
    assert "shield" not in job and "verification" not in job          # the experiment's job (and its hash) as before
    job, _ = RL.build_job(**kw, shield={"fallback": "nmpc"})
    assert job["shield"] == {"fallback": "nmpc", "horizon": 1}
    assert job["verification"] == {"episodes_per_case": 3}              # nominal case only, eval_episodes per case
    assert "EMPÍRICA" in RL.STANDING_WARNING and "NO es control verificado" not in RL.STANDING_WARNING


def _policy(viol_steps=0, excess=(10.0, 20.0), override=0.1):
    shd = {"steps_violated_total": viol_steps, "episodes_with_violation": int(viol_steps > 0),
           "max_violation": 0.5 * viol_steps, "cost": {"n": 4}, "override_fraction_of_steps": override}
    return {"policy": "semilla 1", "shielded": shd, "per_case": [{"excess_vs_oracle_pct": e} for e in excess]}


CASES_OK = [{"case": 0, "oracle": {"cost": 1.0}}, {"case": 1, "oracle": {"cost": 2.0}}]


@pytest.mark.parametrize("pol, cases, tol, ok, start", [
    (_policy(), CASES_OK, 25.0, True, RL.VERDICT_PILOT),
    (_policy(excess=(10.0, 25.0)), CASES_OK, 25.0, True, RL.VERDICT_PILOT),          # boundary: ≤ tolerance
    (_policy(excess=(10.0, 25.01)), CASES_OK, 25.0, True, "no apta por coste"),
    (_policy(viol_steps=3), CASES_OK, 25.0, True, "no apta: viola restricciones incluso con escudo"),
    (_policy(), CASES_OK, None, True, "sin dictamen de coste: falta verification.cost_tolerance_pct"),
    (_policy(excess=(10.0, None)), [CASES_OK[0], {"case": 1, "oracle": {"error": "x"}}], 25.0, True,
     "sin dictamen de coste: no hay referencia NMPC"),
    (_policy(), CASES_OK, 25.0, False, "sin dictamen: la comprobación numérica"),
    (_policy(viol_steps=3), CASES_OK, 25.0, False, "sin dictamen: la comprobación numérica"),
])
def test_verdict_rule(pol, cases, tol, ok, start):
    v = RL.verification_verdict(pol, cases, tol, ok)
    assert v["verdict"].startswith(start), v["verdict"]
    assert (v["verdict"] == RL.VERDICT_PILOT) == (ok and pol["shielded"]["steps_violated_total"] == 0
                                                 and v["criteria"]["cost_within_tolerance"] is True)
    json.dumps(v)


# ---------------------------------------------------------------------------------------------------------------------
# RL environment: analytical first-order case with a constant (adversarial) policy, run through the worker itself
# ---------------------------------------------------------------------------------------------------------------------
FOS_ENV = {"N": 21, "tsim": 2.1, "SP": {"x": 0.5}, "x0": [0.0, 0.5], "a_space": {"low": [-1.0], "high": [1.0]},
           "o_space": {"low": [-1.0, 0.0], "high": [1.0, 1.0]}, "constraints": {"x": [0.6]}, "cons_type": {"x": ["<="]}}
FOS_PI = [{"state": "x", "input": 0, "Kc": 2.0, "tau_I": 0.5, "tau_D": 0.0, "bias": 0.0}]
A = math.exp(-0.1 / 0.5)
SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import rl_worker as W
jobs = json.load(open(sys.argv[2]))
out = {k: W.verify(j, [("constante u = 1", lambda o: np.array([1.0]))]) for k, j in jobs.items()}
json.dump(out, open(sys.argv[3], "w"))
"""


def _fos_job(shield: dict) -> dict:
    job, _ = RL.build_job(environment="first_order_system", scenario=None, env_params=FOS_ENV, algorithm="PPO",
                          total_timesteps=1, seeds=[0], eval_episodes=1, hyperparameters=None, oracle=True,
                          oracle_horizon=5, check_reproducibility=False, output_dir=None, pid_baseline=FOS_PI,
                          shield=shield, verification={"cost_tolerance_pct": 50})
    return job


@pytest.fixture(scope="module")
def constant_policy_runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("shield")
    jobs = {"pid": _fos_job({"fallback": "pid"}), "nmpc": _fos_job({"fallback": "nmpc"}),
            "nmpc_rate": _fos_job({"fallback": "nmpc", "input_rate_max": [0.3], "initial_input": [0.0]})}
    (d / "jobs.json").write_text(json.dumps(jobs))
    (d / "verify.py").write_text(SCRIPT)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CONDA", "MAMBA", "PYTHON", "VIRTUAL_ENV", "UV_", "DYLD_", "LD_LIBRARY", "AMPLFUNC"))}
    env["PATH"] = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                                  if p and ".idaes" not in p and not p.endswith(os.path.join(".venv", "bin")))
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", JAX_PLATFORMS="cpu")
    p = subprocess.run([RL.rl_python(), "-I", str(d / "verify.py"), str(Path(RL.WORKER).parent), str(d / "jobs.json"),
                        str(d / "out.json")], env=env, cwd=str(d), capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads((d / "out.json").read_text())


@needs_rl
@pytest.mark.parametrize("key", ["pid", "nmpc"])
def test_constant_policy_unshielded_matches_analytical_solution(constant_policy_runs, key):
    pol = constant_policy_runs[key]["policies"][0]
    raw = pol["unshielded"]
    assert raw["steps_violated_total"] == 16                                     # k = 5..20
    assert raw["max_violation"] == pytest.approx(1 - A ** 20 - 0.6, rel=2e-5)    # measured 1.2e-5 (cvodes vs exact)
    assert raw["cost"]["n"] == 1 and pol["shielded"]["steps"] == 20
    assert constant_policy_runs[key]["model_check_max_rel_diff"] < 1e-12        # predictor = env step (measured ≤ 4.4e-16)


@needs_rl
@pytest.mark.parametrize("key", ["pid", "nmpc"])
def test_shield_decisions_follow_the_analytical_one_step_rule(constant_policy_runs, key):
    tr = constant_policy_runs[key]["policies"][0]["first_episode_trace"]
    first = next(s["t"] for s in tr if s["source"] != "politica")
    assert first == 4                                                            # x_4 = 1 − a^4, x_5 would be 1 − a^5 > 0.6
    checked = 0
    for s in tr:
        pred = 1 + (s["x_measured"][0] - 1) * A                                  # exact next state under u = 1
        if abs(pred - 0.6) < 1e-4:
            continue
        assert (s["source"] != "politica") == (pred > 0.6), s
        checked += 1
    assert checked >= 19
    assert all(s["x_next"][0] <= 0.6 for s in tr)                               # measured: zero violations with the shield


@needs_rl
def test_shield_with_constant_policy_measured_costs_and_verdicts(constant_policy_runs):
    out = {}
    for key in ("pid", "nmpc"):
        res = constant_policy_runs[key]
        pol = res["policies"][0]
        assert pol["shielded"]["steps_violated_total"] == 0 and pol["shielded"]["interventions"]["unresolved"] == 0
        assert pol["unshielded"]["cost"]["mean"] > pol["shielded"]["cost"]["mean"] > res["cases"][0]["oracle"]["cost"]
        out[key] = pol["per_case"][0]["excess_vs_oracle_pct"]
        for tol, word in ((50.0, "no apta por coste"), (90.0, RL.VERDICT_PILOT)):
            assert RL.verification_verdict(pol, res["cases"], tol, True)["verdict"].startswith(word)
    assert out["nmpc"] == pytest.approx(51.36, abs=0.5)                         # measured 51.357 %
    assert out["pid"] == pytest.approx(80.42, abs=0.5)                          # measured 80.423 %
    iv = constant_policy_runs["pid"]["policies"][0]["shielded"]["interventions"]
    assert (iv["fallback_substitutions"], iv["recovery_substitutions"]) == (13, 1)   # measured: one vertex recovery
    assert RL.verification_verdict(constant_policy_runs["pid"]["policies"][0], constant_policy_runs["pid"]["cases"],
                                   60.0, True)["verdict"].startswith("no apta por coste")
    assert RL.verification_verdict(constant_policy_runs["nmpc"]["policies"][0], constant_policy_runs["nmpc"]["cases"],
                                   60.0, True)["verdict"] == RL.VERDICT_PILOT


@needs_rl
def test_rate_limit_projection_is_exact(constant_policy_runs):
    pol = constant_policy_runs["nmpc_rate"]["policies"][0]
    tr = pol["first_episode_trace"]
    assert tr[0]["u_applied"][0] == pytest.approx(0.3, abs=1e-12)               # min(1, 0 + 0.3)
    assert tr[0]["x_next"][0] == pytest.approx(0.3 * (1 - A), rel=1e-5)
    u = [0.0] + [s["u_applied"][0] for s in tr]
    assert max(abs(b - a) for a, b in zip(u, u[1:])) <= 0.3 + 1e-12
    assert pol["shielded"]["interventions"]["rate_projections"] >= 3            # 0 → 0.3 → 0.6 → 0.9 at least
    assert pol["unshielded"]["rate_limit_exceedances"] == 1                     # the raw policy jumps 0 → 1 once
    json.dumps(constant_policy_runs)


# ---------------------------------------------------------------------------------------------------------------------
# RL environment: the whole tool on the published CSTR with a temperature constraint, disturbances and two x0 (~20 s)
# ---------------------------------------------------------------------------------------------------------------------
@needs_rl
def test_tool_cstr_shield_removes_violations_and_reports_the_cost(tmp_path):
    ver = {"initial_states": [[0.8, 330.0, 0.8], [0.85, 327.0, 0.8]], "episodes_per_case": 2, "cost_tolerance_pct": 50,
           "disturbances": [{"Ti": 350.0}, {"Ti": [350.0] * 30 + [353.0] * 30}],
           "disturbance_bounds": {"Ti": [340.0, 360.0]}}
    shield = {"fallback": "nmpc", "input_rate_max": [1.0], "initial_input": [301.44], "horizon": 5}
    r = TAC.rl_control_experiment(scenario="bloor2024_cstr", algorithm="PPO", total_timesteps=2048, seeds=[1],
                                  eval_episodes=2, hyperparameters={"n_steps": 256, "batch_size": 64}, pid_baseline=CSTR_PI,
                                  output_dir=str(tmp_path), timeout_s=600, env_params=CSTR_CONS, shield=shield,
                                  verification=ver)
    assert r["available"] and r["evidence"]["verified"], r["evidence"]["reasons"]
    v = r["verification"]
    assert "EMPÍRICA" in v["scope"] and "NO es una garantía formal" in v["scope"] and v["rule"] == RL.VERDICT_RULE
    assert len(v["cases"]) == 4 and v["policies"][0]["shielded"]["cost"]["n"] == 8
    pol = v["policies"][0]
    assert pol["unshielded"]["episodes_with_violation"] == 8                    # measured 356 steps, max 2.83 K below 326 K
    assert pol["shielded"]["steps_violated_total"] == 0                         # measured
    assert pol["shielded"]["interventions"]["rate_projections"] > 0 and pol["shielded"]["rate_limit_exceedances"] == 0
    assert 0.3 < pol["shielded"]["override_fraction_of_steps"] < 0.8             # measured 0.53
    assert pol["worst_case"]["excess_vs_oracle_pct"] == pytest.approx(101.6, abs=10.0)   # measured 101.6 %
    assert v["verdicts"][0]["verdict"].startswith("no apta por coste")
    assert v["model_check_max_rel_diff"] < 1e-12
    assert any("desfase" in w for w in r["warnings"])                           # oracle replay with a disturbance step
    assert r["warnings"][0] == RL.STANDING_WARNING
    json.dumps(r)
