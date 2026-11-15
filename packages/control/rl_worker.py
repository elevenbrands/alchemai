"""Worker of the reinforcement-learning experiment (bloque advcontrol, part B).

This script is NOT imported by the repo: `packages.control.rl_runner` launches it with the interpreter of a separate
Python 3.11 environment (pc-gym, gymnasium, stable-baselines3, torch, casadi, do-mpc) as `python -I rl_worker.py <job.json>
<result.json>`. It imports nothing from the repo. Everything it prints goes to stderr/stdout noise; the result is the JSON
file named in argv[2].

Protocol of one experiment
  1. Environment: a pc-gym model (Bloor et al. 2024, arXiv:2410.22093) built from the job's env_params, casadi integrator.
  2. One-step model check (cstr only): the state after one env.step against an independent scipy integration of the two
     published ODEs (paper eqs. 14-15) with the parameter values of the installed pc-gym model.
  3. Baselines on the evaluation episodes: uniform random actions, and pc-gym's oracle (NMPC, do-mpc + ipopt, perfect
     model) whose input sequence is REPLAYED in the same environment so that every return is the environment's own, and
     the user's discrete PID loops when given (on the same noisy observation the agent sees).
  4. For every seed: train a stable-baselines3 agent for total_timesteps, save it, evaluate it deterministically on
     eval_episodes episodes whose noise seeds are disjoint from the training seeds.
  5. Reproducibility: the first seed is trained twice; the two runs must give the same policy parameters and returns.
  6. When the job carries a shield (bloque M25-L5): every trained policy is verified over the declared cases (initial
     states × disturbance scenarios, fixed noise seeds) without and with a safety shield — projection to a_space and to
     the rate limits, short-horizon prediction with the model, fallback (the user's PID or pc-gym's NMPC formulation)
     when a declared state constraint would be violated — against the NMPC oracle and the PID on the same episodes.

Returns are sums of the environment reward over one episode (N − 1 steps; pc-gym ends the episode at t = N − 1)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

EVAL_SEED_BASE = 1_000_000          # evaluation noise seeds start here: disjoint from any training seed below it
ALGORITHMS = ("PPO", "SAC", "DDPG", "TD3")


def _env_params(job: dict):
    import numpy as np
    e = job["env"]
    p = {"N": int(e["N"]), "tsim": float(e["tsim"]), "SP": {k: list(map(float, v)) for k, v in e["SP"].items()},
         "o_space": {"low": np.array(e["o_space"]["low"], float), "high": np.array(e["o_space"]["high"], float)},
         "a_space": {"low": np.array(e["a_space"]["low"], float), "high": np.array(e["a_space"]["high"], float)},
         "x0": np.array(e["x0"], float), "model": job["model"], "integration_method": "casadi",
         "normalise_a": True, "normalise_o": True}
    if e.get("r_scale"):
        p["r_scale"] = {k: float(v) for k, v in e["r_scale"].items()}
    if e.get("noise_percentage"):
        p["noise"] = True
        p["noise_percentage"] = float(e["noise_percentage"])
    if e.get("constraints"):
        p["constraints"] = {k: list(map(float, v)) for k, v in e["constraints"].items()}
        p["cons_type"] = {k: list(v) for k, v in e["cons_type"].items()}
        p["r_penalty"] = bool(e.get("r_penalty", False))
        p["done_on_cons_vio"] = False
    return p


def _violations(states: list, names: list, cons: dict | None, ctype: dict | None) -> dict:
    """Constraint violations of one episode from the true (noise-free, unnormalised) states."""
    steps, worst = 0, 0.0
    if cons:
        for x in states:
            bad = False
            for k, vals in cons.items():
                v = x[names.index(k)]
                for c, t in zip(vals, ctype[k]):
                    d = (c - v) if t == ">=" else (v - c)
                    if d > 0:
                        bad, worst = True, max(worst, float(d))
            steps += int(bad)
    return {"steps_violated": steps, "max_violation": worst}


def _rollout(env, act, seed: int, names: list, cons, ctype, keep: bool = False) -> dict:
    import numpy as np
    np.random.seed(seed)                      # pc-gym draws its measurement noise from numpy's global generator
    obs, _ = env.reset()
    ret, states, actions = 0.0, [], []
    n_states = len(names)
    while True:
        a = np.asarray(act(obs), float).reshape(env.action_space.shape)
        obs, r, done, trunc, _ = env.step(a)
        ret += float(np.sum(r))
        states.append([float(v) for v in env.state[:n_states]])
        lo, hi = env.env_params["a_space"]["low"], env.env_params["a_space"]["high"]
        actions.append([float(v) for v in (np.clip(a, -1, 1) + 1) * (hi - lo) / 2 + lo])
        if done or trunc:
            break
    out = {"return": ret, "steps": len(states), **_violations(states, names, cons, ctype)}
    if keep:
        out.update(states=states, actions=actions)
    return out


def _summary(values: list) -> dict:
    import numpy as np
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _pid_policy(loops: list, params: dict, names: list, dt: float):
    """User's discrete PID loops on the (noisy, unnormalised) observation, velocity form with the derivative on the
    measurement: Δu = Kc·[(e_k − e_{k−1}) + dt/τI·e_k − τD/dt·(y_k − 2y_{k−1} + y_{k−2})], u clipped to a_space (the
    velocity form needs no other anti-windup). Bumpless start from the given bias. Returns a fresh-episode factory."""
    import numpy as np
    lo_o, hi_o = np.asarray(params["o_space"]["low"], float), np.asarray(params["o_space"]["high"], float)
    lo_a, hi_a = np.asarray(params["a_space"]["low"], float), np.asarray(params["a_space"]["high"], float)
    sp_names = list(params["SP"])
    nx = len(names)

    def factory():
        u = np.array(lo_a + (hi_a - lo_a) / 2, float)
        for lp in loops:
            u[lp["input"]] = lp["bias"]
        mem = [None] * len(loops)

        def track(u_applied):
            """External reset (bumpless transfer): when another controller moved the input, the velocity form continues
            from the input actually applied (used by the safety shield, where this PID is the fallback)."""
            u[:] = np.asarray(u_applied, float)

        def act(obs):
            o = (np.asarray(obs, float)[:len(lo_o)] + 1) / 2 * (hi_o - lo_o) + lo_o
            for li, lp in enumerate(loops):
                y, r = o[names.index(lp["state"])], o[nx + sp_names.index(lp["state"])]
                e = r - y
                if mem[li] is None:
                    mem[li] = (e, y, y)
                e1, y1, y2 = mem[li]
                du = lp["Kc"] * ((e - e1) + (dt / lp["tau_I"] * e if lp.get("tau_I") else 0.0)
                                 - (lp.get("tau_D") or 0.0) / dt * (y - 2 * y1 + y2))
                j = lp["input"]
                u[j] = min(max(u[j] + du, lo_a[j]), hi_a[j])
                mem[li] = (e, y, y1)
            return 2 * (u - lo_a) / (hi_a - lo_a) - 1
        act.track = track
        return act
    return factory


def _model_step_check(make_env, params: dict) -> dict | None:
    """cstr only: one env.step from x0 against scipy (LSODA, rtol 1e-10) on the published ODEs."""
    import numpy as np
    from scipy.integrate import solve_ivp
    if params["model"] != "cstr":
        return None
    p = dict(params)
    p.pop("noise", None)
    env = make_env(p)
    env.reset()
    m = env.model
    a = np.array([0.3])                                         # some interior action (normalised)
    Tc = float((a[0] + 1) * (p["a_space"]["high"][0] - p["a_space"]["low"][0]) / 2 + p["a_space"]["low"][0])
    x0 = np.array(p["x0"][:2], float)

    def rhs(_, x):                                              # Bloor et al. 2024, eqs. 14-15
        ca, T = x
        k = m.k0 * np.exp(-m.EA_over_R / T)
        return [m.q / m.V * (m.Caf - ca) - k * ca,
                m.q / m.V * (m.Ti - T) - m.deltaHr / (m.rho * m.C) * k * ca + m.UA / (m.rho * m.C * m.V) * (Tc - T)]

    ref = solve_ivp(rhs, (0.0, env.dt), x0, method="LSODA", rtol=1e-10, atol=1e-12).y[:, -1]
    env.step(a)
    got = np.array(env.state[:2], float)
    return {"dt": float(env.dt), "Tc": Tc, "x0": x0.tolist(), "pcgym_state": got.tolist(), "independent_state": ref.tolist(),
            "max_rel_diff": float(np.max(np.abs(got - ref) / np.abs(ref))),
            "parameters": {k: float(getattr(m, k)) for k in ("q", "V", "rho", "C", "deltaHr", "EA_over_R", "k0", "UA", "Ti", "Caf")}}


def _train(job: dict, make_env, params: dict, seed: int, save_to: str | None):
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.monitor import Monitor
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = Monitor(make_env(params))
    cls = getattr(sb3, job["algorithm"])
    model = cls("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **(job.get("hyperparameters") or {}))
    t0 = time.time()
    model.learn(total_timesteps=int(job["total_timesteps"]))
    wall = time.time() - t0
    rewards, lengths = env.get_episode_rewards(), env.get_episode_lengths()
    h = hashlib.sha256()
    for _, v in sorted(model.policy.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    if save_to:
        model.save(save_to)
    return model, {"episode_returns": [float(r) for r in rewards], "episode_lengths": [int(v) for v in lengths],
                   "train_seconds": wall, "policy_sha256": h.hexdigest()}


def _curve(returns: list, lengths: list, bins: int = 20) -> dict:
    import numpy as np
    if not returns:
        return {"episodes": 0, "points": []}
    r, t = np.asarray(returns), np.cumsum(lengths)
    edges = np.unique(np.linspace(0, len(r), min(bins, len(r)) + 1).round().astype(int))
    pts = [{"timesteps": int(t[b - 1]), "episodes": int(b), "mean_return": float(r[a:b].mean())}
           for a, b in zip(edges[:-1], edges[1:]) if b > a]
    k = max(1, len(r) // 10)
    return {"episodes": int(len(r)), "first_10pct_mean_return": float(r[:k].mean()), "last_10pct_mean_return": float(r[-k:].mean()),
            "best_episode_return": float(r.max()), "points": pts}


# ---------------------------------------------------------------------------------------------------------------------
# Safety shield and per-policy verification (bloque M25-L5)
# ---------------------------------------------------------------------------------------------------------------------
VERIFY_SEED_BASE = 2_000_000        # verification noise seeds: disjoint from training (< 1e6) and evaluation (1e6 + i) seeds
INTERVENTION_KEYS = ("bound_projections", "rate_projections", "predicted_violations", "fallback_substitutions",
                     "recovery_substitutions", "unresolved", "fallback_failures", "prediction_failures")
MAX_RECOVERY_INPUTS = 4             # vertex search of the admissible input box only up to 2^4 candidates


def _oracle_replay(make_env, params: dict, horizon: int, names: list, cons, ctype) -> dict:
    """pc-gym's NMPC oracle (do-mpc + ipopt) planned on the noise-free model with the true state; its input sequence is
    replayed in the environment so that its return is the environment's own."""
    import numpy as np
    from pcgym.oracle import oracle as Oracle
    op = {k: v for k, v in params.items() if k not in ("noise", "noise_percentage")}
    x_log, u_log = Oracle(make_env, op, {"N": int(horizon)}).mpc()
    lo, hi = np.asarray(params["a_space"]["low"], float), np.asarray(params["a_space"]["high"], float)
    nu = len(lo)
    it = iter([2 * (u_log[:nu, i] - lo) / (hi - lo) - 1 for i in range(u_log.shape[1])])
    rp = _rollout(make_env(op), lambda o: next(it), 0, names, cons, ctype, keep=True)
    xs = x_log[:, :rp["steps"]]
    rp["replay_vs_oracle_state_max_rel_diff"] = float(np.max(np.abs(np.array(rp["states"]).T - xs)
                                                             / np.maximum(np.abs(xs), 1e-12)))
    return rp


def _case_params(params: dict, case: dict, model_disturbances: list, bounds: dict | None) -> dict:
    """Environment parameters of one verification case: the job's problem with this initial state and disturbances."""
    import numpy as np
    p = dict(params)
    if case.get("x0") is not None:
        p["x0"] = np.array(case["x0"], float)
    d = case.get("disturbances")
    if d:
        unknown = [k for k in d if k not in model_disturbances]
        if unknown:
            raise ValueError(f"verification.disturbances: {unknown} no son perturbaciones de este modelo; admite "
                             f"{model_disturbances}")
        order = [k for k in model_disturbances if k in d]
        p["disturbances"] = {k: [float(x) for x in d[k]] for k in order}
        p["disturbance_bounds"] = {"low": np.array([bounds[k][0] for k in order], float),
                                   "high": np.array([bounds[k][1] for k in order], float)}
    return p


def _shield_context(make_env, cp: dict, names: list, cons: dict, ctype: dict, job: dict) -> dict:
    """Everything the shield needs for one case: the predictor (pc-gym's own casadi/cvodes discretisation of the model,
    built once), the constraint excess with the user's margins, and the NMPC fallback (pc-gym's oracle formulation)."""
    import numpy as np
    sh = job["shield"]
    quiet = {k: v for k, v in cp.items() if k not in ("noise", "noise_percentage")}
    pe = make_env(dict(quiet))
    pe.reset()
    F = pe.int_eng.discretise_model(pe.int_eng.casadi_model_func, pe.dt)
    info = pe.model.info()
    nx, nsp = len(names), len(cp["SP"])
    dist = cp.get("disturbances") or {}
    user_d = [k for k in info["disturbances"] if k in dist]
    margin = {k: float(v) for k, v in (sh.get("constraint_margin") or {}).items()}
    horizon = int(sh.get("horizon") or 1)
    failures = [0]

    def full_input(u, dvals: dict):
        if not dist:
            return np.asarray(u, float)
        return np.concatenate([np.asarray(u, float), [float(dvals[k]) if k in dvals else float(info["parameters"][k])
                                                      for k in info["disturbances"]]])

    def step(x, uk):
        return np.array(F(x0=x, p=uk)["xf"].full(), float).reshape(-1)

    def excess(x) -> float:
        """Largest constraint excess of a state, tightened by the margins (> 0 = violated)."""
        worst = -np.inf
        for k, vals in cons.items():
            v, m = float(x[names.index(k)]), margin.get(k, 0.0)
            for c, t in zip(vals, ctype[k]):
                worst = max(worst, (c + m - v) if t == ">=" else (v - c + m))
        return float(worst)

    def predicted_excess(o, u) -> float:
        """Short-horizon prediction from the MEASURED state, input held, disturbance held at its last measured value.
        A failed integration counts as unsafe."""
        x = np.asarray(o[:nx], float)
        uk = full_input(u, dict(zip(user_d, o[nx + nsp: nx + nsp + len(user_d)])))
        worst = -np.inf
        try:
            for _ in range(horizon):
                x = step(x, uk)
                if not np.all(np.isfinite(x)):
                    raise RuntimeError("non-finite prediction")
                worst = max(worst, excess(x))
        except RuntimeError:
            failures[0] += 1
            return float("inf")
        return worst

    def exact_next(x_true, u, t):
        """One step from the true state with the disturbance the environment applies at this step (model identity)."""
        return step(np.asarray(x_true, float), full_input(u, {k: dist[k][t + 1] for k in user_d}))

    ctx = {"names": names, "n_policy_obs": len(job["env"]["o_space"]["low"]), "fallback": sh["fallback"],
           "rate": None if not sh.get("input_rate_max") else
           np.array([np.inf if r is None else float(r) for r in sh["input_rate_max"]], float),
           "u_init": None if sh.get("initial_input") is None else np.array(sh["initial_input"], float),
           "predicted_excess": predicted_excess, "exact_next": exact_next, "prediction_failures": failures}
    if sh["fallback"] == "nmpc":
        from pcgym.oracle import oracle as Oracle
        mpc, _ = Oracle(make_env, dict(quiet), {"N": int(job["oracle"]["horizon"])}).setup_mpc()
        nu = len(cp["a_space"]["low"])

        def nmpc_start(x0):
            mpc.x0 = np.asarray(x0, float).reshape(-1, 1)
            mpc.set_initial_guess()

        def nmpc_act(t, x_est, u_prev):
            mpc.t0 = float(t * pe.dt)                   # setpoint (and disturbance) schedule at the current step
            if u_prev is not None:
                mpc.u0 = np.asarray(u_prev, float).reshape(-1, 1)
            try:
                u = np.asarray(mpc.make_step(np.asarray(x_est, float).reshape(-1, 1)), float).reshape(-1)[:nu]
            except Exception:  # noqa: BLE001 — a failed fallback is counted, never hidden
                return None
            return u if bool(mpc.solver_stats.get("success", False)) and np.all(np.isfinite(u)) else None
        ctx.update(nmpc_start=nmpc_start, nmpc_act=nmpc_act)
    return ctx


def _verify_rollout(env, policy, seed: int, ctx: dict, cons, ctype, shielded: bool, pid_act=None,
                    keep: bool = False) -> dict:
    """One verification episode of a policy, with or without the shield. Without it the policy's action is applied as
    given (only rate-limit exceedances are counted). With it: project to a_space and to the rate limits, predict the
    short horizon from the measured state, and if a declared constraint would be violated substitute the fallback's
    (projected) action when its predicted violation is smaller; if that one too is predicted to violate, try the vertices
    of the admissible input box (a_space ∩ rate limits) and keep the least violating candidate. Every intervention is
    counted."""
    import itertools
    import numpy as np
    np.random.seed(seed)
    obs, _ = env.reset()
    lo = np.asarray(env.env_params["a_space"]["low"], float)
    hi = np.asarray(env.env_params["a_space"]["high"], float)
    olo, ohi = np.asarray(env.observation_space_base.low, float), np.asarray(env.observation_space_base.high, float)
    names, rate = ctx["names"], ctx["rate"]
    nx, npol = len(names), ctx["n_policy_obs"]
    u_prev = None if ctx["u_init"] is None else ctx["u_init"].copy()
    cnt = {k: 0 for k in INTERVENTION_KEYS}
    f0 = ctx["prediction_failures"][0]
    rate_exc, ret, states, trace, resid = 0, 0.0, [], [], 0.0
    if shielded and ctx["fallback"] == "nmpc":
        ctx["nmpc_start"](np.asarray(env.state[:nx], float))

    def project(u):
        w = np.clip(u, lo, hi)
        if np.any(np.abs(w - u) > 1e-12 * np.maximum(1.0, np.abs(u))):
            cnt["bound_projections"] += 1
        if rate is not None and u_prev is not None:
            w2 = np.clip(w, u_prev - rate, u_prev + rate)
            if np.any(np.abs(w2 - w) > 1e-12 * np.maximum(1.0, np.abs(w))):
                cnt["rate_projections"] += 1
            w = w2
        return w

    while True:
        t = int(env.t)
        o = (np.asarray(obs, float) + 1) / 2 * (ohi - olo) + olo
        u_pol = (np.asarray(policy(np.asarray(obs)[:npol]), float).reshape(lo.shape) + 1) * (hi - lo) / 2 + lo
        u_fb = None if pid_act is None else (np.asarray(pid_act(obs), float) + 1) * (hi - lo) / 2 + lo
        src = "politica"
        if not shielded:
            u = u_pol
            if rate is not None and u_prev is not None and np.any(np.abs(u - u_prev) > rate * (1 + 1e-9) + 1e-12):
                rate_exc += 1
        else:
            u = project(u_pol)
            best = ctx["predicted_excess"](o, u)
            if best > 0:
                cnt["predicted_violations"] += 1
                c2 = None
                if ctx["fallback"] == "pid":
                    c2 = project(u_fb)
                else:
                    c2 = ctx["nmpc_act"](t, o[:nx], u_prev)
                    if c2 is None:
                        cnt["fallback_failures"] += 1
                    else:
                        c2 = project(c2)
                v2 = float("inf") if c2 is None else ctx["predicted_excess"](o, c2)
                if v2 < best:
                    u, src, best = c2, "respaldo", v2
                if best > 0 and len(lo) <= MAX_RECOVERY_INPUTS:
                    blo, bhi = lo.copy(), hi.copy()
                    if rate is not None and u_prev is not None:
                        blo, bhi = np.maximum(lo, u_prev - rate), np.minimum(hi, u_prev + rate)
                    for corner in itertools.product(*zip(blo, bhi)):
                        vc = ctx["predicted_excess"](o, np.array(corner, float))
                        if vc < best:
                            u, src, best = np.array(corner, float), "recuperacion", vc
                if src == "respaldo":
                    cnt["fallback_substitutions"] += 1
                elif src == "recuperacion":
                    cnt["recovery_substitutions"] += 1
                if best > 0:
                    cnt["unresolved"] += 1
        if pid_act is not None:
            pid_act.track(u)
        x_pred = ctx["exact_next"](np.asarray(env.state[:nx], float), u, t)
        obs, r, done, trunc, _ = env.step(2 * (u - lo) / (hi - lo) - 1)
        x_new = np.array(env.state[:nx], float)
        resid = max(resid, float(np.max(np.abs(x_new - x_pred) / np.maximum(np.abs(x_new), 1e-12))))
        ret += float(np.sum(r))
        states.append(x_new.tolist())
        if keep:
            trace.append({"t": t, "u_policy": u_pol.tolist(), "u_applied": u.tolist(), "source": src,
                          "x_measured": o[:nx].tolist(), "x_next": x_new.tolist()})
        u_prev = np.asarray(u, float).copy()
        if done or trunc:
            break
    cnt["prediction_failures"] = ctx["prediction_failures"][0] - f0
    out = {"return": ret, "steps": len(states), **_violations(states, names, cons, ctype), "interventions": cnt,
           "rate_limit_exceedances": rate_exc, "model_check_max_rel_diff": resid}
    if keep:
        out["trace"] = trace
    return out


def _aggregate(eps: list) -> dict:
    return {"cost": _summary([-e["return"] for e in eps]),
            "steps_violated_total": int(sum(e["steps_violated"] for e in eps)),
            "episodes_with_violation": int(sum(e["steps_violated"] > 0 for e in eps)),
            "max_violation": float(max(e["max_violation"] for e in eps)),
            "rate_limit_exceedances": int(sum(e["rate_limit_exceedances"] for e in eps))}


def _excess_pct(cost: float, ref: float | None) -> float | None:
    return None if ref is None or not ref > 0 else 100.0 * (cost - ref) / ref


def verify(job: dict, policies: list) -> dict:
    """Per-policy verification over the declared case set (initial states × disturbance scenarios), with fixed noise
    seeds shared by every controller (paired comparison): each policy without and with the shield, the user's PID alone,
    and pc-gym's NMPC oracle (one noise-free run per case). `policies` = [(label, act(obs) -> normalised action)].
    Cost = −return of the environment (lower is better)."""
    import numpy as np
    from pcgym import make_env
    t0 = time.time()
    params = _env_params(job)
    probe = make_env(params)
    info = probe.model.info()
    names = list(info["states"])
    cons, ctype = params.get("constraints"), params.get("cons_type")
    v, sh = job["verification"], job["shield"]
    make_pid = None
    if job.get("pid_baseline"):
        unknown = [lp["state"] for lp in job["pid_baseline"] if lp["state"] not in names or lp["state"] not in params["SP"]]
        if unknown:
            raise ValueError(f"pid_baseline: los estados {unknown} no son estados con consigna del modelo {names}")
        make_pid = _pid_policy(job["pid_baseline"], params, names, float(probe.dt))
    if sh["fallback"] == "pid" and make_pid is None:
        raise ValueError("shield.fallback = 'pid' necesita pid_baseline")
    cases = [{"x0": x, "disturbances": d} for x in (v.get("initial_states") or [None])
             for d in (v.get("disturbances") or [None])]
    n_ep = int(v["episodes_per_case"])
    case_out, rows, model_res = [], {lab: [] for lab, _ in policies}, 0.0
    for ci, case in enumerate(cases):
        cp = _case_params(params, case, list(info["disturbances"]), v.get("disturbance_bounds"))
        env = make_env(cp)
        ctx = _shield_context(make_env, cp, names, cons, ctype, job)
        seeds = [VERIFY_SEED_BASE + 1000 * ci + e for e in range(n_ep)]
        co = {"case": ci, "x0": [float(x) for x in cp["x0"]],
              "disturbances": sorted(case["disturbances"]) if case["disturbances"] else None,
              "noise_seeds": [seeds[0], seeds[-1]]}
        if job["oracle"]["enabled"]:
            try:
                rp = _oracle_replay(make_env, cp, job["oracle"]["horizon"], names, cons, ctype)
                co["oracle"] = {"cost": -rp["return"], "steps_violated": rp["steps_violated"],
                                "max_violation": rp["max_violation"],
                                "replay_vs_oracle_state_max_rel_diff": rp["replay_vs_oracle_state_max_rel_diff"]}
            except Exception as ex:  # noqa: BLE001 — the oracle is a comparison; its failure is reported
                co["oracle"] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        if make_pid is not None:
            pr = [_rollout(env, make_pid(), s, names, cons, ctype) for s in seeds]
            co["pid"] = {"cost": _summary([-r["return"] for r in pr]),
                         "steps_violated_total": int(sum(r["steps_violated"] for r in pr)),
                         "max_violation": float(max(r["max_violation"] for r in pr))}
        for lab, pol in policies:
            for e, s in enumerate(seeds):
                raw = _verify_rollout(env, pol, s, ctx, cons, ctype, shielded=False)
                shd = _verify_rollout(env, pol, s, ctx, cons, ctype, shielded=True,
                                      pid_act=make_pid() if sh["fallback"] == "pid" else None, keep=(ci == 0 and e == 0))
                rows[lab].append((ci, raw, shd))
                model_res = max(model_res, raw["model_check_max_rel_diff"], shd["model_check_max_rel_diff"])
        case_out.append(co)
    pols = []
    for lab, _ in policies:
        rs = rows[lab]
        raw, shd = _aggregate([r for _, r, _ in rs]), _aggregate([s for _, _, s in rs])
        shd["interventions"] = {k: int(sum(s["interventions"][k] for _, _, s in rs)) for k in INTERVENTION_KEYS}
        shd["steps"] = int(sum(s["steps"] for _, _, s in rs))
        shd["override_fraction_of_steps"] = (shd["interventions"]["fallback_substitutions"]
                                             + shd["interventions"]["recovery_substitutions"]) / max(shd["steps"], 1)
        per_case = []
        for co in case_out:
            cr = [r for c, r, _ in rs if c == co["case"]]
            cs = [s for c, _, s in rs if c == co["case"]]
            mr, ms = float(np.mean([-r["return"] for r in cr])), float(np.mean([-s["return"] for s in cs]))
            oc = co.get("oracle", {}).get("cost")
            pc = co["pid"]["cost"]["mean"] if "pid" in co else None
            per_case.append({"case": co["case"], "cost_unshielded_mean": mr, "cost_shielded_mean": ms,
                             "cost_shielded_max": float(max(-s["return"] for s in cs)), "oracle_cost": oc,
                             "excess_vs_oracle_pct": _excess_pct(ms, oc),
                             "unshielded_excess_vs_oracle_pct": _excess_pct(mr, oc),
                             "pid_cost_mean": pc, "excess_vs_pid_pct": _excess_pct(ms, pc),
                             "steps_violated_unshielded": int(sum(r["steps_violated"] for r in cr)),
                             "steps_violated_shielded": int(sum(s["steps_violated"] for s in cs)),
                             "max_violation_unshielded": float(max(r["max_violation"] for r in cr)),
                             "max_violation_shielded": float(max(s["max_violation"] for s in cs)),
                             "fallback_substitutions": int(sum(s["interventions"]["fallback_substitutions"] for s in cs)),
                             "recovery_substitutions": int(sum(s["interventions"]["recovery_substitutions"] for s in cs))})
        known = [p for p in per_case if p["excess_vs_oracle_pct"] is not None]
        worst = max(known, key=lambda p: p["excess_vs_oracle_pct"]) if known else None
        pols.append({"policy": lab, "unshielded": raw, "shielded": shd, "per_case": per_case,
                     "worst_case": {"case_with_highest_excess_vs_oracle": None if worst is None else worst["case"],
                                    "excess_vs_oracle_pct": None if worst is None else worst["excess_vs_oracle_pct"],
                                    "highest_episode_cost_shielded": shd["cost"]["max"],
                                    "max_violation_shielded": shd["max_violation"],
                                    "max_violation_unshielded": raw["max_violation"]},
                     "first_episode_trace": next(s["trace"] for c, _, s in rs if "trace" in s)})
    return {"cases": case_out, "policies": pols, "episodes_per_case": n_ep, "cases_count": len(cases),
            "episodes_per_policy": len(cases) * n_ep, "noise_seed_base": VERIFY_SEED_BASE,
            "deterministic": not params.get("noise", False), "model_check_max_rel_diff": model_res,
            "seconds": time.time() - t0}


def run(job: dict) -> dict:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import casadi
    import do_mpc
    import gymnasium
    import pcgym  # noqa: F401  (fails early with a clear ImportError if pc-gym is missing)
    import stable_baselines3 as sb3
    from importlib.metadata import version
    from pcgym import make_env
    t_start = time.time()
    params = _env_params(job)
    probe = make_env(params)
    names = list(probe.model.info()["states"])
    cons, ctype = params.get("constraints"), params.get("cons_type")
    n_eval = int(job["eval_episodes"])
    eval_seeds = [EVAL_SEED_BASE + i for i in range(n_eval)]
    out = {"versions": {"python": sys.version.split()[0], "pcgym": version("pcgym"), "gymnasium": gymnasium.__version__,
                        "stable_baselines3": sb3.__version__, "torch": torch.__version__, "casadi": casadi.__version__,
                        "do_mpc": do_mpc.__version__, "numpy": np.__version__},
           "environment": {"model": params["model"], "states": names, "inputs": list(probe.model.info()["inputs"]),
                           "steps_per_episode": int(params["N"]) - 1, "dt": float(probe.dt),
                           "observation_size": int(probe.observation_space.shape[0]), "deterministic": not params.get("noise", False)},
           "eval_seeds": [eval_seeds[0], eval_seeds[-1]]}
    out["model_step_check"] = _model_step_check(make_env, params)
    # baselines -------------------------------------------------------------------------------------------------------
    env = make_env(params)
    rng = np.random.default_rng(int(job.get("random_policy_seed", 0)))
    rnd = [_rollout(env, lambda o: rng.uniform(-1, 1, size=env.action_space.shape), s, names, cons, ctype) for s in eval_seeds]
    out["random_policy"] = {"return": _summary([r["return"] for r in rnd]),
                            "steps_violated": _summary([r["steps_violated"] for r in rnd])}
    pid_ret = None
    if job.get("pid_baseline"):
        unknown = [lp["state"] for lp in job["pid_baseline"] if lp["state"] not in names or lp["state"] not in params["SP"]]
        if unknown:
            raise ValueError(f"pid_baseline: los estados {unknown} no son estados con consigna del modelo {names} / "
                             f"{list(params['SP'])}")
        if any(not 0 <= lp["input"] < env.action_space.shape[0] for lp in job["pid_baseline"]):
            raise ValueError(f"pid_baseline: 'input' fuera de rango (el modelo tiene {env.action_space.shape[0]} acciones)")
        make_pid = _pid_policy(job["pid_baseline"], params, names, float(probe.dt))
        pr = [_rollout(env, make_pid(), s, names, cons, ctype, keep=(s == eval_seeds[0])) for s in eval_seeds]
        pid_ret = float(np.mean([r["return"] for r in pr]))
        out["pid_baseline"] = {"return": _summary([r["return"] for r in pr]),
                               "steps_violated": _summary([r["steps_violated"] for r in pr]),
                               "max_violation": float(max(r["max_violation"] for r in pr)), "loops": job["pid_baseline"],
                               "first_episode_final_states": pr[0]["states"][-1],
                               "first_episode_actions_head": [a[0] for a in pr[0]["actions"][:5]]}
    oracle_ret = None
    if job.get("oracle", {}).get("enabled", True):
        t0 = time.time()
        try:
            horizon = int(job["oracle"].get("horizon", 5))
            rp = _oracle_replay(make_env, params, horizon, names, cons, ctype)
            oracle_ret = rp["return"]
            out["oracle"] = {"return": oracle_ret, "horizon": horizon, "seconds": time.time() - t0,
                             "steps_violated": rp["steps_violated"], "max_violation": rp["max_violation"],
                             "replay_vs_oracle_state_max_rel_diff": rp["replay_vs_oracle_state_max_rel_diff"],
                             "final_states": rp["states"][-1], "first_actions": [a[0] for a in rp["actions"][:5]]}
        except Exception as ex:  # noqa: BLE001 — the oracle is a comparison, its failure must not hide the experiment
            out["oracle"] = {"error": f"{type(ex).__name__}: {str(ex)[:400]}"}
    # agents ----------------------------------------------------------------------------------------------------------
    os.makedirs(job["out_dir"], exist_ok=True)
    seeds = [int(s) for s in job["seeds"]]
    per_seed, means, trained = [], [], []
    for seed in seeds:
        path = os.path.join(job["out_dir"], f"{job['algorithm'].lower()}_{params['model']}_seed{seed}")
        model, info = _train(job, make_env, params, seed, path)
        trained.append((f"semilla {seed}", lambda o, m=model: m.predict(o, deterministic=True)[0]))
        ev = [_rollout(env, lambda o: model.predict(o, deterministic=True)[0], s, names, cons, ctype) for s in eval_seeds]
        rets = [r["return"] for r in ev]
        means.append(float(np.mean(rets)))
        per_seed.append({"seed": seed, "policy_path": path + ".zip", "policy_sha256": info["policy_sha256"],
                         "train_seconds": info["train_seconds"], "learning_curve": _curve(info["episode_returns"], info["episode_lengths"]),
                         "eval_return": _summary(rets), "eval_returns": rets,
                         "eval_steps_violated": _summary([r["steps_violated"] for r in ev]),
                         "eval_max_violation": float(max(r["max_violation"] for r in ev))})
    out["per_seed"] = per_seed
    out["eval_return_across_seeds"] = _summary(means)
    # reproducibility: same seed, same budget → same parameters and same evaluation ------------------------------------
    if job.get("check_reproducibility", True):
        model2, info2 = _train(job, make_env, params, seeds[0], None)
        ev2 = [_rollout(env, lambda o: model2.predict(o, deterministic=True)[0], s, names, cons, ctype)["return"] for s in eval_seeds]
        a = np.asarray(per_seed[0]["eval_returns"])
        out["reproducibility"] = {"seed": seeds[0], "same_policy_parameters": info2["policy_sha256"] == per_seed[0]["policy_sha256"],
                                  "eval_return_max_abs_diff": float(np.max(np.abs(a - np.asarray(ev2)))),
                                  "eval_return_scale": float(np.max(np.abs(a)))}
    rnd_mean = out["random_policy"]["return"]["mean"]
    if oracle_ret is not None:
        gap = oracle_ret - out["eval_return_across_seeds"]["mean"]
        den = oracle_ret - rnd_mean
        out["gap_to_oracle"] = {"absolute": gap, "per_seed": [oracle_ret - m for m in means],
                                "fraction_of_random_to_oracle_gap_remaining": (gap / den) if den > 0 else None}
    out["beats_random"] = {"seeds_better_than_random": int(sum(m > rnd_mean for m in means)), "seeds": len(seeds),
                           "mean_improvement_over_random": out["eval_return_across_seeds"]["mean"] - rnd_mean}
    if pid_ret is not None:
        out["vs_pid"] = {"seeds_better_than_pid": int(sum(m > pid_ret for m in means)), "seeds": len(seeds),
                         "mean_return_minus_pid": out["eval_return_across_seeds"]["mean"] - pid_ret}
    if job.get("shield"):
        out["verification"] = verify(job, trained)
    out["wall_seconds"] = time.time() - t_start
    return out


def oracle_trajectory(job: dict) -> dict:
    """Closed-loop trajectory of pc-gym's own NMPC oracle (do-mpc + ipopt, noise-free, do-mpc's default orthogonal
    collocation) for the job's environment: the independent reference used to validate the repo's NMPC."""
    import numpy as np
    import casadi
    import do_mpc
    from importlib.metadata import version
    from pcgym import make_env
    from pcgym.oracle import oracle as Oracle
    params = _env_params(job)
    params.pop("noise", None)
    params.pop("noise_percentage", None)
    t0 = time.time()
    mp = {"N": int(job["oracle"]["horizon"])}
    orc = Oracle(make_env, params, mp)
    x_log, u_log = orc.mpc()
    mpc_obj, _ = orc.setup_mpc()
    st = mpc_obj.settings
    return {"x_log": np.asarray(x_log).tolist(), "u_log": np.asarray(u_log).tolist(), "dt": float(orc.env.dt),
            "horizon": mp["N"], "Q_diag": np.diag(orc.Q).tolist(), "R_diag": np.diag(np.atleast_2d(orc.R)).tolist(),
            "discretization": {"method": str(st.state_discretization), "collocation_type": str(st.collocation_type),
                               "collocation_deg": int(st.collocation_deg), "collocation_ni": int(st.collocation_ni)},
            "seconds": time.time() - t0,
            "versions": {"pcgym": version("pcgym"), "do_mpc": do_mpc.__version__, "casadi": casadi.__version__}}


def main() -> int:
    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path) as f:
        job = json.load(f)
    try:
        res = {"ok": True, "result": oracle_trajectory(job) if job.get("mode") == "oracle_trajectory" else run(job)}
    except Exception as ex:  # noqa: BLE001 — reported to the caller, which raises it in Spanish
        import traceback
        res = {"ok": False, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-2000:]}
    with open(result_path, "w") as f:
        json.dump(res, f)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
