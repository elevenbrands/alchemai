"""Reinforcement learning for a process-control task on pc-gym: repo side (bloque advcontrol, part B; shield M25-L5).

What it answers: «if I train a standard RL agent on this benchmark control problem with this budget, how good is it
against a random policy and against the NMPC oracle, how much does it vary between seeds, and is the experiment
reproducible?» and, when a safety shield is declared, «with that shield, does each trained policy respect my state
constraints and how far is its cost from the NMPC and the PID over the set of cases I declared?». The second answer is
an EMPIRICAL verification over the tested set (fixed seeds), never a formal guarantee.

The engine lives in another environment (admission rule for engines outside the repo's Python): pc-gym 0.1.8 (MIT; Bloor et
al. 2024, arXiv:2410.22093), gymnasium (MIT), stable-baselines3 (MIT), torch (BSD-style), casadi and do-mpc (LGPL-3, used
unmodified as libraries by pc-gym's oracle) in `~/tools/alchemai-rl/.venv` (Python 3.11). This module only builds the job,
launches `rl_worker.py` with that interpreter in its own process group with a timeout (the group is killed on expiry), and
reads the JSON it writes. Nothing of that stack is imported here.

`evidence.verified` certifies only: (1) same seed and budget → identical policy parameters and identical evaluation returns
(the first seed is trained twice); (2) the evaluation protocol was followed (deterministic policy, evaluation noise seeds
disjoint from the training seeds, complete episodes, every return computed by the environment itself — also the oracle's,
whose inputs are replayed in the environment); (3) for the cstr, one environment step agrees with an independent
integration of the published ODEs; (4) with a shield, the shield's predictor reproduces the environment's step from the
true state and every verification episode is complete. Whether the policy is fit for a pilot test is the separate
`verification.verdicts`, with the explicit rule in `verification.rule`."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

RL_HOME = Path("~/tools/alchemai-rl").expanduser()
WORKER = Path(__file__).with_name("rl_worker.py")
CREATE_COMMANDS = (
    "uv venv --python 3.11 ~/tools/alchemai-rl/.venv",
    'uv pip install --python ~/tools/alchemai-rl/.venv/bin/python "pcgym==0.1.8" "stable-baselines3==2.9.2" "numpy<2.4"',
)
ALGORITHMS = ("PPO", "SAC", "DDPG", "TD3")
# models shipped by pc-gym 0.1.8 (pcgym.make_env.model_mapping)
MODELS = ("cstr", "first_order_system", "nonsmooth_control", "multistage_extraction", "cstr_series_recycle",
          "distillation_column", "multistage_extraction_reactive", "four_tank", "heat_exchanger", "biofilm_reactor",
          "polymerisation_reactor", "crystallization")
EVAL_SEED_BASE = 1_000_000              # must equal rl_worker.EVAL_SEED_BASE
DEFAULT_TIMEOUT_S = 1800
MAX_TIMESTEPS = 2_000_000
MODEL_STEP_TOL = 1e-4                   # pc-gym's casadi integrator vs scipy LSODA at rtol 1e-10 (measured 2.6e-6)
ORACLE_REPLAY_TOL = 1e-3                # oracle's own simulator vs the environment replay (measured 8.6e-6)

# Published benchmark configuration: Bloor et al. (2024), «PC-Gym: Benchmark Environments For Process Control Problems»,
# arXiv:2410.22093v4, Section 4.2 code listing and Section 4.2.1 (oracle horizon 17, Q = I, R = 0); read 2026-11-14 at
# https://arxiv.org/html/2410.22093 . The listing leaves N undefined; N = 60 is the «60 timesteps for 25 minutes» of its
# first line. The paper's custom reward (`sp_track_reward`) is not printed: pc-gym's built-in reward −Σ(x − x*)² is used.
PRESETS = {
    "bloor2024_cstr": {
        "model": "cstr", "oracle_horizon": 17,
        "env": {"N": 60, "tsim": 25.0, "SP": {"Ca": [0.85] * 20 + [0.9] * 20 + [0.87] * 20},
                "o_space": {"low": [0.7, 300.0, 0.8], "high": [1.0, 350.0, 0.9]},
                "a_space": {"low": [295.0], "high": [302.0]}, "x0": [0.8, 330.0, 0.8], "noise_percentage": 0.001},
        "source": "Bloor et al. (2024) arXiv:2410.22093, §4.2 (listado) y §4.2.1; leído el 2026-11-14",
    },
}
STANDING_WARNING = (
    "Una política de RL entrenada no tiene garantía formal de estabilidad ni de respeto de restricciones. Sin escudo "
    "(shield) es un experimento; con escudo, cada acción se proyecta a los límites y al ritmo de cambio y se sustituye por "
    "el controlador de respaldo cuando el modelo predice que violaría una restricción declarada, y la verificación por "
    "política es EMPÍRICA: cubre solo los casos y episodios probados, con el modelo de pc-gym como planta. Antes de usarla "
    "en planta hay que validarla contra el modelo riguroso (con desajuste de modelo) y mantener el control convencional y "
    "las protecciones (SIS) como respaldo.")
SHIELD_FALLBACKS = ("pid", "nmpc")
MAX_SHIELD_HORIZON = 50
MAX_VERIFICATION_EPISODES = 2000        # cases × episodes_per_case
VERDICT_PILOT = "apta para prueba piloto con escudo"
VERDICT_RULE = (
    f"«{VERDICT_PILOT}» solo si se cumplen todas: (1) la comprobación numérica del experimento pasó (evidence.verified); "
    "(2) CERO pasos con violación de las restricciones declaradas (estado verdadero, sin margen) con escudo en todos los "
    "episodios de verificación; (3) hay referencia NMPC en todos los casos; (4) en cada caso, el coste medio con escudo es "
    "≤ (1 + cost_tolerance_pct/100) × coste del NMPC de ese caso. Sin cost_tolerance_pct no se dictamina el coste. Coste = "
    "−retorno del entorno (−Σ r_scale·(x − consigna)²).")


def rl_python() -> str | None:
    """Interpreter of the separate RL environment: $ALCHEMAI_RL_PYTHON or ~/tools/alchemai-rl/.venv/bin/python."""
    for c in (os.environ.get("ALCHEMAI_RL_PYTHON"), str(RL_HOME / ".venv" / "bin" / "python")):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def rl_available() -> bool:
    return rl_python() is not None


def missing_environment_message() -> str:
    return ("el entorno de RL no está instalado (va aparte del repositorio, Python 3.11, para no mezclar torch/jax con sus "
            "dependencias). Créalo con estos dos comandos y vuelve a llamar:\n  " + "\n  ".join(CREATE_COMMANDS)
            + "\n(o apunta la variable ALCHEMAI_RL_PYTHON a un intérprete que tenga pcgym y stable-baselines3)")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def _box(d, name: str) -> dict:
    if not isinstance(d, dict) or "low" not in d or "high" not in d:
        raise ValueError(f"env_params.{name} necesita 'low' y 'high' (listas)")
    lo, hi = [float(v) for v in d["low"]], [float(v) for v in d["high"]]
    if len(lo) != len(hi) or not lo or any(a >= b for a, b in zip(lo, hi)):
        raise ValueError(f"env_params.{name}: 'low' y 'high' deben tener la misma longitud y low < high")
    return {"low": lo, "high": hi}


def build_env(env_params: dict) -> dict:
    """Validate the user's pc-gym problem definition (nothing is defaulted)."""
    known = {"N", "tsim", "SP", "x0", "a_space", "o_space", "r_scale", "noise_percentage", "constraints", "cons_type", "r_penalty"}
    unknown = set(env_params) - known
    if unknown:
        raise ValueError(f"env_params: claves no admitidas {sorted(unknown)}; se aceptan {sorted(known)}")
    need = {"N": "número de pasos del episodio", "tsim": "duración del episodio (unidades de tiempo del modelo)",
            "SP": "consignas {estado: lista de N valores}", "x0": "estado inicial seguido del valor inicial de cada consigna",
            "a_space": "límites de la acción {low, high}", "o_space": "límites de la observación {low, high} (normalización)"}
    missing = [f"{k} ({v})" for k, v in need.items() if env_params.get(k) is None]
    if missing:
        raise ValueError("faltan datos del problema de control en env_params: " + "; ".join(missing)
                         + ". Son del usuario (o usa scenario='bloor2024_cstr', la configuración publicada)")
    N, tsim = int(env_params["N"]), float(env_params["tsim"])
    if N < 3 or not tsim > 0:
        raise ValueError("env_params: N ≥ 3 y tsim > 0")
    sp = {}
    if not isinstance(env_params["SP"], dict) or not env_params["SP"]:
        raise ValueError("env_params.SP debe ser {nombre_del_estado: lista de N consignas}")
    for k, v in env_params["SP"].items():
        v = [float(v)] * N if isinstance(v, (int, float)) else [float(x) for x in v]
        if len(v) != N:
            raise ValueError(f"env_params.SP['{k}'] necesita N = {N} valores (uno por paso) o un escalar")
        sp[str(k)] = v
    env = {"N": N, "tsim": tsim, "SP": sp, "x0": [float(v) for v in env_params["x0"]],
           "a_space": _box(env_params["a_space"], "a_space"), "o_space": _box(env_params["o_space"], "o_space")}
    if len(env["o_space"]["low"]) != len(env["x0"]):
        raise ValueError("env_params: x0 y o_space deben tener la misma longitud (estados del modelo + una entrada por "
                         "consigna, convención de pc-gym)")
    if env_params.get("r_scale"):
        env["r_scale"] = {str(k): float(v) for k, v in env_params["r_scale"].items()}
    if env_params.get("noise_percentage"):
        env["noise_percentage"] = float(env_params["noise_percentage"])
        if not 0 < env["noise_percentage"] < 1:
            raise ValueError("env_params.noise_percentage es una fracción del valor medido, entre 0 y 1")
    if env_params.get("constraints"):
        ct = env_params.get("cons_type")
        if not isinstance(ct, dict) or set(ct) != set(env_params["constraints"]):
            raise ValueError("env_params.constraints necesita cons_type con las mismas claves: {'T': ['<=', '>=']}")
        for k, v in env_params["constraints"].items():
            if len(v) != len(ct[k]) or any(t not in ("<=", ">=") for t in ct[k]):
                raise ValueError(f"env_params.cons_type['{k}']: un '<=' o '>=' por cada valor de constraints['{k}']")
        env["constraints"] = {str(k): [float(x) for x in v] for k, v in env_params["constraints"].items()}
        env["cons_type"] = {str(k): list(v) for k, v in ct.items()}
        env["r_penalty"] = bool(env_params.get("r_penalty", False))
    return env


def _pid_loops(spec, n_actions: int) -> list[dict]:
    """User's PID baseline loops: [{state, input, Kc, tau_I, tau_D, bias}] in the model's own units (nothing defaulted
    except tau_D = 0 and input = 0 for a single-action model)."""
    if not isinstance(spec, (list, tuple)) or not spec:
        raise ValueError("pid_baseline debe ser una lista de lazos {state, input, Kc, tau_I, tau_D, bias}")
    loops = []
    for i, lp in enumerate(spec):
        if not isinstance(lp, dict):
            raise ValueError(f"pid_baseline[{i}] debe ser un dict")
        unknown = set(lp) - {"state", "input", "Kc", "tau_I", "tau_D", "bias"}
        if unknown:
            raise ValueError(f"pid_baseline[{i}]: claves desconocidas {sorted(unknown)}")
        miss = [k for k, why in (("state", "estado controlado, con consigna en SP"),
                                 ("Kc", "ganancia en unidades de acción por unidad del estado; signo = acción del lazo"),
                                 ("bias", "valor inicial de la acción en unidades físicas (salida del PID en t = 0)"))
                if lp.get(k) is None]
        if "tau_I" not in lp:
            miss.append("tau_I")
        if miss:
            raise ValueError(f"pid_baseline[{i}]: faltan {miss} (state = estado controlado con consigna; Kc = ganancia con "
                             "su signo; tau_I = tiempo integral en las unidades de tiempo del modelo, null = sin acción "
                             "integral; bias = acción inicial en unidades físicas). Son ajustes del usuario")
        if lp.get("input") is None and n_actions != 1:
            raise ValueError(f"pid_baseline[{i}]: falta 'input' (índice de la acción que mueve este lazo)")
        tau_i = None if lp["tau_I"] is None else float(lp["tau_I"])
        if tau_i is not None and not tau_i > 0:
            raise ValueError(f"pid_baseline[{i}]: tau_I debe ser > 0 (o null)")
        tau_d = float(lp.get("tau_D") or 0.0)
        if tau_d < 0:
            raise ValueError(f"pid_baseline[{i}]: tau_D debe ser ≥ 0")
        loops.append({"state": str(lp["state"]), "input": int(lp.get("input") or 0), "Kc": float(lp["Kc"]),
                      "tau_I": tau_i, "tau_D": tau_d, "bias": float(lp["bias"])})
    if len({lp["input"] for lp in loops}) != len(loops):
        raise ValueError("pid_baseline: cada acción puede moverla un solo lazo")
    return loops


def _floats(v, n: int, what: str) -> list[float]:
    v = [v] if isinstance(v, (int, float)) and n == 1 else v
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ValueError(f"{what}: se esperan {n} valores (uno por acción)")
    return [float(x) for x in v]


def shield_spec(shield, env: dict, has_pid: bool) -> dict:
    """Validate the user's safety shield. The constraints it protects are env_params.constraints (user data); the
    fallback is the user's PID or pc-gym's NMPC formulation; rate limits need the input applied before the first step."""
    if not isinstance(shield, dict):
        raise ValueError("shield debe ser un dict {fallback, input_rate_max, initial_input, horizon, constraint_margin}")
    unknown = set(shield) - {"fallback", "input_rate_max", "initial_input", "horizon", "constraint_margin"}
    if unknown:
        raise ValueError(f"shield: claves no admitidas {sorted(unknown)}")
    if not env.get("constraints"):
        raise ValueError("el escudo necesita las restricciones de estado que debe proteger: env_params.constraints "
                         "{estado: [valores]} con cons_type {estado: ['<=' o '>=']} (son datos del usuario)")
    fb = shield.get("fallback")
    if fb not in SHIELD_FALLBACKS:
        raise ValueError("falta shield.fallback, el controlador que sustituye a la política cuando el modelo predice una "
                         "violación: 'pid' (tu pid_baseline) o 'nmpc' (NMPC de pc-gym con el modelo y tus restricciones)")
    if fb == "pid" and not has_pid:
        raise ValueError("shield.fallback = 'pid' necesita pid_baseline (tu PID es el controlador de respaldo)")
    lo, hi = env["a_space"]["low"], env["a_space"]["high"]
    nu = len(lo)
    spec: dict = {"fallback": fb, "horizon": 1}
    if shield.get("horizon") is not None:
        spec["horizon"] = int(shield["horizon"])
        if not 1 <= spec["horizon"] <= MAX_SHIELD_HORIZON:
            raise ValueError(f"shield.horizon: pasos de predicción entre 1 y {MAX_SHIELD_HORIZON}")
    if shield.get("initial_input") is not None:
        ii = _floats(shield["initial_input"], nu, "shield.initial_input")
        if any(not a <= u <= b for u, a, b in zip(ii, lo, hi)):
            raise ValueError("shield.initial_input fuera de a_space")
        spec["initial_input"] = ii
    if shield.get("input_rate_max") is not None:
        rate = shield["input_rate_max"]
        rate = [rate] if isinstance(rate, (int, float)) and nu == 1 else rate
        if not isinstance(rate, (list, tuple)) or len(rate) != nu or any(r is not None and not float(r) > 0 for r in rate):
            raise ValueError("shield.input_rate_max: un valor > 0 por acción, en unidades físicas de la acción por paso "
                             "(null = sin límite para esa acción)")
        if "initial_input" not in spec:
            raise ValueError("shield.input_rate_max necesita shield.initial_input (acción aplicada antes del primer paso, "
                             "en "
                             "unidades físicas): sin ella el primer paso no tiene referencia para el límite de cambio")
        spec["input_rate_max"] = [None if r is None else float(r) for r in rate]
    if shield.get("constraint_margin"):
        m = shield["constraint_margin"]
        if not isinstance(m, dict) or set(m) - set(env["constraints"]) or any(float(x) < 0 for x in m.values()):
            raise ValueError("shield.constraint_margin: {estado restringido: margen ≥ 0 en unidades del estado} (el escudo "
                             "endurece la restricción en ese margen al predecir)")
        spec["constraint_margin"] = {str(k): float(x) for k, x in m.items()}
    return spec


def verification_spec(v, env: dict, eval_episodes: int) -> dict:
    """Validate the declared verification set: initial states × disturbance scenarios, episodes per case and the user's
    cost tolerance against the NMPC. Nothing is defaulted except the nominal case and episodes_per_case = eval_episodes."""
    v = {} if v is None else v
    if not isinstance(v, dict):
        raise ValueError("verification debe ser un dict {initial_states, disturbances, disturbance_bounds, "
                         "episodes_per_case, cost_tolerance_pct}")
    unknown = set(v) - {"initial_states", "disturbances", "disturbance_bounds", "episodes_per_case", "cost_tolerance_pct"}
    if unknown:
        raise ValueError(f"verification: claves no admitidas {sorted(unknown)}")
    N, nx0 = env["N"], len(env["x0"])
    spec: dict = {"episodes_per_case": int(v.get("episodes_per_case") or eval_episodes)}
    if not 1 <= spec["episodes_per_case"] <= 200:
        raise ValueError("verification.episodes_per_case debe estar entre 1 y 200")
    n_cases = 1
    if v.get("initial_states") is not None:
        xs = v["initial_states"]
        if not isinstance(xs, (list, tuple)) or not xs or any(not isinstance(x, (list, tuple)) or len(x) != nx0 for x in xs):
            raise ValueError(f"verification.initial_states: lista de estados iniciales, cada uno con {nx0} valores (misma "
                             "convención que x0: estados del modelo y luego una entrada por consigna)")
        spec["initial_states"] = [[float(a) for a in x] for x in xs]
        n_cases *= len(xs)
    if v.get("disturbances") is not None:
        ds = v["disturbances"]
        if not isinstance(ds, (list, tuple)) or not ds or any(not isinstance(d, dict) or not d for d in ds):
            raise ValueError("verification.disturbances: lista de escenarios {perturbación: N valores o un escalar}, p. ej. "
                             "[{'Ti': 350}, {'Ti': [350]*30 + [355]*30}]")
        b = v.get("disturbance_bounds")
        used = sorted({k for d in ds for k in d})
        if not isinstance(b, dict) or set(used) - set(b):
            raise ValueError(f"verification.disturbance_bounds: falta {{perturbación: [mínimo, máximo]}} para {used} "
                             "(pc-gym "
                             "lo usa para normalizar la medición de la perturbación)")
        bounds = {}
        for k in used:
            lo_k, hi_k = (float(x) for x in b[k])
            if not lo_k < hi_k:
                raise ValueError(f"verification.disturbance_bounds['{k}']: mínimo < máximo")
            bounds[k] = [lo_k, hi_k]
        scen = []
        for d in ds:
            sd = {}
            for k, vals in d.items():
                vals = [float(vals)] * N if isinstance(vals, (int, float)) else [float(x) for x in vals]
                if len(vals) != N:
                    raise ValueError(f"verification.disturbances['{k}'] necesita N = {N} valores (uno por paso) o un "
                                     "escalar")
                if any(not bounds[k][0] <= x <= bounds[k][1] for x in vals):
                    raise ValueError(f"verification.disturbances['{k}'] sale de disturbance_bounds {bounds[k]}")
                sd[str(k)] = vals
            scen.append(sd)
        spec["disturbances"], spec["disturbance_bounds"] = scen, bounds
        n_cases *= len(scen)
    if v.get("cost_tolerance_pct") is not None:
        spec["cost_tolerance_pct"] = float(v["cost_tolerance_pct"])
        if not spec["cost_tolerance_pct"] > 0:
            raise ValueError("verification.cost_tolerance_pct debe ser > 0 (cuánto más cara que el NMPC aceptas, en %)")
    if n_cases * spec["episodes_per_case"] > MAX_VERIFICATION_EPISODES:
        raise ValueError(f"verification: {n_cases} casos × {spec['episodes_per_case']} episodios supera el máximo "
                         f"{MAX_VERIFICATION_EPISODES}")
    return spec


def verification_verdict(policy: dict, cases: list, tolerance_pct: float | None, numerics_ok: bool) -> dict:
    """Verdict of one policy from its verification report, with the explicit rule VERDICT_RULE."""
    sh = policy["shielded"]
    excess = [p["excess_vs_oracle_pct"] for p in policy["per_case"]]
    no_ref = [c["case"] for c in cases if "cost" not in (c.get("oracle") or {})]
    have_ref = not no_ref and all(e is not None for e in excess)
    worst = max(excess) if have_ref else None
    within = None if (worst is None or tolerance_pct is None) else worst <= tolerance_pct
    zero = sh["steps_violated_total"] == 0
    if not numerics_ok:
        verdict = "sin dictamen: la comprobación numérica del experimento no pasó (ver evidence.reasons)"
    elif not zero:
        verdict = (f"no apta: viola restricciones incluso con escudo ({sh['steps_violated_total']} pasos en "
                   f"{sh['episodes_with_violation']} de {sh['cost']['n']} episodios; máxima violación "
                   f"{sh['max_violation']:.4g})")
    elif not have_ref:
        verdict = (f"sin dictamen de coste: no hay referencia NMPC en los casos {no_ref or 'con coste nulo'} (oráculo "
                   "desactivado o fallido); cero violaciones con escudo en el conjunto probado")
    elif tolerance_pct is None:
        verdict = ("sin dictamen de coste: falta verification.cost_tolerance_pct (cuánto más cara que el NMPC aceptas, "
                   "en %); "
                   f"cero violaciones con escudo en el conjunto probado y peor caso {worst:.1f} % sobre el NMPC")
    elif not within:
        verdict = (f"no apta por coste: cero violaciones con escudo, pero en el peor caso cuesta {worst:.1f} % más que el "
                   f"NMPC (tolerancia {tolerance_pct:g} %)")
    else:
        verdict = VERDICT_PILOT
    return {"policy": policy["policy"], "verdict": verdict,
            "criteria": {"numerics_verified": bool(numerics_ok), "zero_violations_with_shield": zero,
                         "nmpc_reference_in_every_case": have_ref, "worst_case_excess_vs_oracle_pct": worst,
                         "cost_tolerance_pct": tolerance_pct, "cost_within_tolerance": within,
                         "override_fraction_of_steps": sh["override_fraction_of_steps"]}}


def build_job(*, environment: str | None, scenario: str | None, env_params: dict | None, algorithm: str,
              total_timesteps: int | None, seeds: list[int] | None, eval_episodes: int, hyperparameters: dict | None,
              oracle: bool, oracle_horizon: int | None, check_reproducibility: bool, output_dir: str | None,
              pid_baseline: list[dict] | None = None, shield: dict | None = None,
              verification: dict | None = None) -> tuple[dict, list[str]]:
    assumptions = []
    preset = None
    if scenario is not None:
        if scenario not in PRESETS:
            raise ValueError(f"scenario desconocido: {scenario!r}; disponibles: {sorted(PRESETS)} (o da env_params completo)")
        preset = PRESETS[scenario]
        if environment is not None and environment != preset["model"]:
            raise ValueError(f"scenario '{scenario}' es del entorno '{preset['model']}', no de '{environment}'")
        environment = preset["model"]
    if environment is None:
        raise ValueError(f"falta environment (el modelo de pc-gym): uno de {list(MODELS)}; o scenario='bloor2024_cstr'")
    if environment not in MODELS:
        raise ValueError(f"environment desconocido: {environment!r}; pc-gym 0.1.8 trae {list(MODELS)}")
    if preset is None and env_params is None:
        raise ValueError("falta la definición del problema: env_params (N, tsim, SP, x0, a_space, o_space; datos del "
                         "usuario, pc-gym no trae valores por defecto) o scenario='bloor2024_cstr' (configuración publicada)")
    merged = dict(preset["env"]) if preset else {}
    if env_params:
        merged.update(env_params)
        if preset:
            assumptions.append(f"escenario '{scenario}' con estas claves cambiadas por el usuario: {sorted(env_params)}")
    env = build_env(merged)
    if preset:
        assumptions.append(f"problema de control = escenario publicado '{scenario}' ({preset['source']}); recompensa "
                           "integrada de pc-gym −Σ r_scale·(x − consigna)² (la recompensa propia del artículo no está impresa)")
    algorithm = str(algorithm).upper()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm debe ser uno de {list(ALGORITHMS)} (stable-baselines3)")
    if total_timesteps is None or int(total_timesteps) < 1:
        raise ValueError("falta total_timesteps (pasos de entrenamiento por semilla): es el presupuesto del experimento y lo "
                         "decide el usuario (el artículo de pc-gym usa 50 000)")
    if int(total_timesteps) > MAX_TIMESTEPS:
        raise ValueError(f"total_timesteps demasiado grande (máximo {MAX_TIMESTEPS})")
    if not seeds:
        raise ValueError("faltan seeds (lista de semillas de entrenamiento, p. ej. [1, 2, 3]): la dispersión entre "
                         "semillas es parte del resultado y no se supone")
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= EVAL_SEED_BASE for s in seeds):
        raise ValueError(f"seeds: enteros distintos entre 0 y {EVAL_SEED_BASE - 1} (las semillas ≥ {EVAL_SEED_BASE} se "
                         "reservan para los episodios de evaluación)")
    if int(eval_episodes) < 1 or int(eval_episodes) > 1000:
        raise ValueError("eval_episodes debe estar entre 1 y 1000")
    hp = dict(hyperparameters or {})
    try:
        json.dumps(hp)
    except TypeError:
        raise ValueError("hyperparameters debe ser un dict de valores simples (números, cadenas, listas) para "
                         "stable-baselines3, p. ej. {'learning_rate': 3e-4, 'n_steps': 512}") from None
    if any(k in hp for k in ("policy", "env", "seed", "device", "verbose")):
        raise ValueError("hyperparameters no puede fijar policy, env, seed, device ni verbose (los fija el experimento)")
    if not hp:
        assumptions.append(f"hiperparámetros por defecto de stable-baselines3 para {algorithm} (política MLP 64×64): no se "
                           "sintonizaron; son parte de lo que se evalúa")
    if oracle_horizon is None:
        oracle_horizon = preset["oracle_horizon"] if preset else 5
        if oracle and not preset:
            assumptions.append("horizonte del oráculo NMPC = 5 pasos (valor por defecto de pc-gym): ajuste numérico del "
                               "comparador, no dato del proceso")
    job = {"model": environment, "env": env, "algorithm": algorithm, "total_timesteps": int(total_timesteps), "seeds": seeds,
           "eval_episodes": int(eval_episodes), "hyperparameters": hp,
           "oracle": {"enabled": bool(oracle), "horizon": int(oracle_horizon)}, "check_reproducibility": bool(check_reproducibility)}
    if pid_baseline is not None:
        job["pid_baseline"] = _pid_loops(pid_baseline, len(env["a_space"]["low"]))
        for lp in job["pid_baseline"]:
            lo, hi = env["a_space"]["low"][lp["input"]], env["a_space"]["high"][lp["input"]]
            if not lo <= lp["bias"] <= hi:
                raise ValueError(f"pid_baseline: bias = {lp['bias']:g} fuera de a_space [{lo:g}, {hi:g}] de la acción "
                                 f"{lp['input']}")
    if verification is not None and shield is None:
        raise ValueError("verification compara la política sin y con escudo: falta shield (al menos {'fallback': 'pid' o "
                         "'nmpc'}) y env_params.constraints")
    if shield is not None:
        job["shield"] = shield_spec(shield, env, bool(job.get("pid_baseline")))
        job["verification"] = verification_spec(verification, env, int(eval_episodes))
        if job["shield"]["fallback"] == "nmpc" and not oracle and oracle_horizon is None and not preset:
            assumptions.append("respaldo NMPC con horizonte de 5 pasos (valor por defecto de pc-gym): ajuste numérico")
    tag =hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]
    job["out_dir"] = str(Path(output_dir).expanduser()) if output_dir else str(RL_HOME / "runs" / f"{algorithm.lower()}_{environment}_{tag}")
    return job, assumptions


def run_job(job: dict, timeout_s: int = DEFAULT_TIMEOUT_S, python: str | None = None) -> dict:
    """Run the worker in the RL environment, in its own process group; the group is killed if it exceeds timeout_s."""
    py = python or rl_python()
    if py is None:
        raise RuntimeError(missing_environment_message())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path, result_path = out_dir / "job.json", out_dir / "result.json"
    job_path.write_text(json.dumps(job))
    if result_path.exists():
        result_path.unlink()
    # the child must not inherit the repo's library search paths: importing idaes in this process prepends ~/.idaes/bin
    # to DYLD_LIBRARY_PATH/LD_LIBRARY_PATH/PATH, and casadi in the child then fails to load its own ipopt plugin
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CONDA", "MAMBA", "PYTHON", "VIRTUAL_ENV", "UV_", "DYLD_", "LD_LIBRARY", "AMPLFUNC"))}
    env["PATH"] = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                                  if p and ".idaes" not in p and not p.endswith(os.path.join(".venv", "bin")))
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", JAX_PLATFORMS="cpu")
    proc = subprocess.Popen([py, "-I", str(WORKER), str(job_path), str(result_path)], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True,
                            cwd=str(out_dir))
    try:
        _, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        raise RuntimeError(f"el experimento de RL no terminó en {timeout_s} s y se canceló (grupo de procesos terminado): "
                           "reduce total_timesteps, el número de semillas o eval_episodes, o sube timeout_s") from None
    if not result_path.exists():
        raise RuntimeError(f"el proceso de RL terminó (código {proc.returncode}) sin escribir resultado: {err.strip()[-600:]}")
    res = json.loads(result_path.read_text())
    if not res.get("ok"):
        raise RuntimeError(f"el experimento de RL falló dentro del entorno de pc-gym: {res.get('error')}")
    return res["result"]


def oracle_trajectory(*, scenario: str | None = None, environment: str | None = None, env_params: dict | None = None,
                      oracle_horizon: int | None = None, output_dir: str | None = None, timeout_s: int = 300) -> dict:
    """Noise-free closed-loop trajectory of pc-gym's NMPC oracle (do-mpc), run in the RL environment. Used as the
    independent reference of the repo's NMPC (`packages.control.nmpc`); not an agent tool."""
    job, _ = build_job(environment=environment, scenario=scenario, env_params=env_params, algorithm="PPO",
                       total_timesteps=1, seeds=[0], eval_episodes=1, hyperparameters=None, oracle=True,
                       oracle_horizon=oracle_horizon, check_reproducibility=False, output_dir=output_dir)
    job["mode"] = "oracle_trajectory"
    return run_job(job, timeout_s)


def _verification_section(job: dict, r: dict, reasons: list, warnings: list, assumptions: list) -> dict:
    """Numerical checks, verdicts, warnings and assumptions of the per-policy verification (appends to the lists)."""
    v, sh, vs = r["verification"], job["shield"], job["verification"]
    steps = r["environment"]["steps_per_episode"]
    if v["model_check_max_rel_diff"] > MODEL_STEP_TOL:
        reasons.append("el predictor del escudo no reproduce el paso del entorno desde el estado verdadero (diferencia "
                       f"relativa {v['model_check_max_rel_diff']:.2e})")
    for p in v["policies"]:
        n_pol = v["episodes_per_policy"]
        if p["shielded"]["steps"] != n_pol * steps or p["unshielded"]["cost"]["n"] != n_pol:
            reasons.append(f"verificación de la política {p['policy']}: episodios incompletos")
    for c in v["cases"]:
        o = c.get("oracle") or {}
        if "error" in o:
            warnings.append(f"el oráculo NMPC falló en el caso de verificación {c['case']}: {o['error']}")
        elif o and o["replay_vs_oracle_state_max_rel_diff"] > ORACLE_REPLAY_TOL:
            msg = (f"caso {c['case']}: la trayectoria del oráculo reproducida en el entorno difiere de la de su propio "
                   f"simulador ({o['replay_vs_oracle_state_max_rel_diff']:.2e})")
            if c["disturbances"]:
                warnings.append(msg + "; con perturbación es esperable: el oráculo de pc-gym 0.1.8 indexa la perturbación "
                                "con un paso de desfase respecto del entorno. Su coste se calcula igualmente en el entorno")
            else:
                reasons.append(msg)
    verdicts = [verification_verdict(p, v["cases"], vs.get("cost_tolerance_pct"), not reasons) for p in v["policies"]]
    n_ep = v["episodes_per_policy"]
    for p, vd in zip(v["policies"], verdicts):
        p.update(verdict=vd["verdict"], criteria=vd["criteria"])
        raw, shd, iv = p["unshielded"], p["shielded"], p["shielded"]["interventions"]
        warnings.append(f"política {p['policy']}: sin escudo viola restricciones en {raw['episodes_with_violation']} de "
                        f"{n_ep} "
                        f"episodios (máxima violación {raw['max_violation']:.4g}); con escudo en "
                        f"{shd['episodes_with_violation']} (máxima {shd['max_violation']:.4g}); el escudo sustituyó la "
                        f"acción en {100 * shd['override_fraction_of_steps']:.1f} % de los pasos "
                        f"({iv['fallback_substitutions']} por el respaldo y {iv['recovery_substitutions']} por un vértice "
                        f"de recuperación, de {shd['steps']}). "
                        f"Dictamen: {vd['verdict']}")
        if iv["unresolved"]:
            warnings.append(f"política {p['policy']}: en {iv['unresolved']} pasos ni la política ni el respaldo cumplían la "
                            "predicción; se aplicó la acción de menor violación predicha")
        if iv["fallback_failures"]:
            warnings.append(f"política {p['policy']}: el NMPC de respaldo no convergió en {iv['fallback_failures']} pasos "
                            "(se mantuvo la acción proyectada de la política)")
        if iv["prediction_failures"]:
            warnings.append(f"política {p['policy']}: la predicción del escudo falló en {iv['prediction_failures']} "
                            "evaluaciones (tratadas como inseguras)")
        if raw["rate_limit_exceedances"]:
            warnings.append(f"política {p['policy']}: sin escudo excede input_rate_max en {raw['rate_limit_exceedances']} "
                            "pasos")
    if not vs.get("initial_states") and not vs.get("disturbances"):
        warnings.append("solo se verificó el caso nominal (x0 de env_params, sin perturbación): declara "
                        "verification.initial_states y verification.disturbances para cubrir el conjunto que te importa")
    if v["deterministic"] and vs["episodes_per_case"] > 1:
        warnings.append("entorno sin ruido de medición: los episodios de un mismo caso son idénticos")
    rate = (f" y, con input_rate_max, a ±límite respecto de la acción aplicada en el paso anterior (desde initial_input)"
            if sh.get("input_rate_max") else " (sin límite de ritmo de cambio: no diste input_rate_max)")
    assumptions.append(
        f"escudo: cada acción se recorta a a_space{rate}; luego se predicen {sh['horizon']} paso(s) con el modelo de pc-gym "
        "(integrador cvodes de casadi) desde el estado MEDIDO (con el ruido de env_params), con la acción y la perturbación "
        "mantenidas en su último valor; si un estado predicho viola una restricción declarada"
        + (" (endurecida con constraint_margin)" if sh.get("constraint_margin")
           else " (sin margen: no diste constraint_margin)")
        + ", se calcula la acción del respaldo, se proyecta igual y se aplica si su violación predicha es menor; si aún se "
        "predice violación, se prueban los vértices de la caja de acciones admisibles (a_space ∩ ritmo de cambio, hasta "
        f"{2 ** 4} vértices) y se aplica el candidato de menor violación predicha; si ninguno cumple, el paso se cuenta "
        "como "
        "'unresolved'")
    if sh["fallback"] == "pid":
        assumptions.append("respaldo = tu PID (pid_baseline), que se ejecuta cada paso en paralelo y sigue la acción "
                           "realmente aplicada (transferencia sin salto)")
    else:
        assumptions.append(f"respaldo = NMPC de pc-gym (do-mpc + ipopt, horizonte {job['oracle']['horizon']} pasos, Q = I, "
                           "R = 0, restricciones de estado como cotas duras), resuelto desde el estado medido solo en los "
                           "pasos en que el escudo interviene; con perturbaciones conoce de antemano su trayectoria "
                           "declarada (ventaja optimista)")
    assumptions.append(
        f"verificación: {v['cases_count']} caso(s) (estados iniciales × escenarios de perturbación declarados), "
        f"{vs['episodes_per_case']} episodio(s) por caso con semillas de ruido fijas {v['noise_seed_base']} + 1000·caso + "
        "episodio, las mismas para la política sin escudo, con escudo y el PID (comparación pareada); el NMPC de referencia "
        "es una corrida sin ruido por caso; violaciones sobre el estado verdadero tras cada paso, sin margen; "
        "coste = −retorno")
    if vs.get("disturbances"):
        assumptions.append("la política (entrenada sin perturbación) no ve la medición de la perturbación; el escudo sí")
    return {"scope": (f"verificación EMPÍRICA sobre el conjunto probado: {v['cases_count']} caso(s) × "
                      f"{vs['episodes_per_case']} episodio(s) = {n_ep} episodios por política, semillas fijas; NO es una "
                      "garantía formal de seguridad ni de estabilidad fuera de ese conjunto ni frente a desajuste de "
                      "modelo"),
            "rule": VERDICT_RULE, "shield": sh, "declared_set": vs,
            "verdicts": [{"policy": d["policy"], "verdict": d["verdict"]} for d in verdicts],
            "cases": v["cases"], "policies": v["policies"], "model_check_max_rel_diff": v["model_check_max_rel_diff"],
            "seconds": v["seconds"]}


def rl_experiment(*, environment: str | None = None, scenario: str | None = None, env_params: dict | None = None,
                  algorithm: str = "PPO", total_timesteps: int | None = None, seeds: list[int] | None = None,
                  eval_episodes: int = 10, hyperparameters: dict | None = None, oracle: bool = True,
                  oracle_horizon: int | None = None, check_reproducibility: bool = True, output_dir: str | None = None,
                  timeout_s: int = DEFAULT_TIMEOUT_S, pid_baseline: list[dict] | None = None, shield: dict | None = None,
                  verification: dict | None = None) -> dict:
    """Train, evaluate and compare; with a shield, also verify every trained policy without and with it over the declared
    cases. Raises ValueError (missing/invalid input) or RuntimeError (environment absent, timeout, failure inside pc-gym),
    both in Spanish."""
    job, assumptions = build_job(environment=environment, scenario=scenario, env_params=env_params, algorithm=algorithm,
                                 total_timesteps=total_timesteps, seeds=seeds, eval_episodes=eval_episodes,
                                 hyperparameters=hyperparameters, oracle=oracle, oracle_horizon=oracle_horizon,
                                 check_reproducibility=check_reproducibility, output_dir=output_dir,
                                 pid_baseline=pid_baseline, shield=shield, verification=verification)
    if not rl_available():
        raise RuntimeError(missing_environment_message())
    t0 = time.time()
    r = run_job(job, timeout_s)
    warnings, reasons = [STANDING_WARNING], []
    steps = r["environment"]["steps_per_episode"]
    # --- evidence: reproducibility and protocol -------------------------------------------------------------------------
    rep = r.get("reproducibility")
    if rep is None:
        reasons.append("no se comprobó la reproducibilidad (check_reproducibility=false)")
    elif not rep["same_policy_parameters"] or rep["eval_return_max_abs_diff"] > 1e-12 * max(rep["eval_return_scale"], 1e-300):
        reasons.append(f"la misma semilla no reprodujo el resultado (parámetros iguales: {rep['same_policy_parameters']}; "
                       f"diferencia de retorno {rep['eval_return_max_abs_diff']:.3e})")
    protocol = {"eval_episodes_per_seed": job["eval_episodes"], "deterministic_policy": True,
                "eval_noise_seeds": r["eval_seeds"], "training_seeds": job["seeds"],
                "eval_seeds_disjoint_from_training": all(s < EVAL_SEED_BASE for s in job["seeds"]),
                "complete_episodes": all(p["eval_return"]["n"] == job["eval_episodes"] for p in r["per_seed"]),
                "returns_computed_by_the_environment": True, "steps_per_episode": steps}
    if not (protocol["eval_seeds_disjoint_from_training"] and protocol["complete_episodes"]):
        reasons.append("el protocolo de evaluación no se cumplió (episodios incompletos o semillas solapadas)")
    msc = r.get("model_step_check")
    if msc is not None and msc["max_rel_diff"] > MODEL_STEP_TOL:
        reasons.append(f"un paso del entorno difiere {msc['max_rel_diff']:.2e} de la integración independiente de las EDO")
    orc = r.get("oracle")
    if orc is not None and "error" in orc:
        warnings.append(f"el oráculo NMPC de pc-gym falló y no hay referencia superior: {orc['error']}")
    elif orc is not None and orc["replay_vs_oracle_state_max_rel_diff"] > ORACLE_REPLAY_TOL:
        reasons.append("la trayectoria del oráculo reproducida en el entorno difiere de la de su propio simulador "
                       f"({orc['replay_vs_oracle_state_max_rel_diff']:.2e})")
    # --- honest reading of the result -----------------------------------------------------------------------------------
    br = r["beats_random"]
    if br["seeds_better_than_random"] < br["seeds"]:
        warnings.append(f"solo {br['seeds_better_than_random']} de {br['seeds']} semillas superan a la política aleatoria con "
                        f"este presupuesto ({job['total_timesteps']} pasos): el agente no ha aprendido de forma fiable")
    gap = r.get("gap_to_oracle")
    if gap is not None:
        frac = gap["fraction_of_random_to_oracle_gap_remaining"]
        if gap["absolute"] > 0:
            warnings.append("el agente NO alcanza al oráculo NMPC: retorno medio "
                            f"{r['eval_return_across_seeds']['mean']:.6g} frente a {orc['return']:.6g}"
                            + (f"; queda el {100 * frac:.1f} % de la distancia aleatorio→oráculo" if frac is not None else ""))
        else:
            warnings.append("el agente iguala o supera al oráculo en estos episodios: el oráculo es un NMPC de horizonte "
                            f"finito ({orc['horizon']} pasos) con función objetivo propia, no una cota superior estricta")
    vp = r.get("vs_pid")
    if vp is not None:
        if vp["seeds_better_than_pid"] < vp["seeds"]:
            warnings.append(f"solo {vp['seeds_better_than_pid']} de {vp['seeds']} semillas superan a tu PID en estos episodios "
                            f"(retorno del PID {r['pid_baseline']['return']['mean']:.6g})")
        if r["pid_baseline"]["max_violation"] > 0:
            warnings.append(f"el PID dado viola restricciones en evaluación (máxima violación {r['pid_baseline']['max_violation']:.4g})")
    if r["environment"]["deterministic"]:
        warnings.append("el entorno es determinista (sin ruido de medición): los episodios de evaluación son idénticos y la "
                        "única dispersión es entre semillas de entrenamiento; da noise_percentage para episodios distintos")
    if len(job["seeds"]) < 3:
        warnings.append(f"solo {len(job['seeds'])} semilla(s): la dispersión entre semillas no es estimable con menos de 3")
    viol = max(p["eval_max_violation"] for p in r["per_seed"])
    if viol > 0:
        warnings.append(f"la política viola restricciones en evaluación (máxima violación {viol:.4g}): la penalización en "
                        "la recompensa no las garantiza")
    ver = _verification_section(job, r, reasons, warnings, assumptions) if job.get("shield") else None
    assumptions += [
        "retorno = suma de la recompensa del entorno en un episodio de N − 1 pasos; mayor (menos negativo) es mejor",
        "evaluación con la política determinista (media de la acción), sobre episodios cuyo ruido usa semillas distintas "
        "de las de entrenamiento; mismas consignas y estado inicial que en el entrenamiento salvo que env_params diga otra cosa",
        "oráculo = NMPC de pc-gym (do-mpc + ipopt) con el modelo exacto y el estado verdadero sin ruido; sus entradas se "
        "reproducen en el entorno para que su retorno se calcule igual que el del agente; es UN episodio sin ruido, "
        "mientras que el agente se evalúa en episodios con el ruido de env_params (con ruido pequeño la diferencia es menor "
        "que la dispersión entre semillas, pero no es una comparación episodio a episodio)",
        "entrenamiento en CPU con un hilo y algoritmos deterministas de torch (por eso es reproducible bit a bit)",
    ]
    if job.get("pid_baseline"):
        assumptions.append("PID de la línea base = el del usuario, discreto con periodo dt = tsim/N del entorno, forma de "
                           "velocidad con la derivada sobre la medición, arranque sin salto desde bias y acción recortada a "
                           "a_space; ve la misma observación con ruido que el agente y se evalúa en los mismos episodios")
    not_included = [
        "sintonía de la línea base PID: pc-gym no trae ninguna; solo se evalúa el PID que des en pid_baseline (para "
        "sintonizarlo usa las herramientas de identificación y sintonía PID del repositorio sobre una prueba escalón)",
        "sintonía de hiperparámetros y de la función de recompensa",
        "generalización a otras consignas, estados iniciales, perturbaciones o parámetros: solo se evalúa el escenario dado "
        "(y, con shield, el conjunto declarado en verification)",
        "garantía FORMAL de estabilidad o de seguridad (verificación formal, conjuntos invariantes, RL con restricciones): "
        "el escudo (shield) y la verificación por política son empíricos, sobre los casos y episodios probados",
        "entrenamiento con escudo (el escudo actúa solo al evaluar y verificar la política ya entrenada)",
        "desajuste entre el modelo del escudo y la planta: el escudo predice con el mismo modelo de pc-gym que simula la "
        "planta (solo la medición con ruido lo separa del estado verdadero)",
        "transferencia a la planta real (sim-to-real)",
    ]
    out = {
        "experiment": {"environment": job["model"], "scenario": scenario, "algorithm": job["algorithm"],
                       "total_timesteps_per_seed": job["total_timesteps"], "seeds": job["seeds"],
                       "eval_episodes": job["eval_episodes"], "hyperparameters": job["hyperparameters"],
                       "env_params": job["env"], "output_dir": job["out_dir"]},
        "environment": r["environment"],
        "eval_return": r["eval_return_across_seeds"],
        "eval_return_note": "media y dispersión ENTRE semillas de entrenamiento del retorno medio de evaluación de cada una",
        "random_policy": r["random_policy"], "oracle": orc, "gap_to_oracle": gap, "beats_random": br,
        "pid_baseline": r.get("pid_baseline"), "vs_pid": vp,
        "per_seed": [{k: p[k] for k in ("seed", "eval_return", "eval_steps_violated", "eval_max_violation", "learning_curve",
                                        "train_seconds", "policy_path", "policy_sha256")} for p in r["per_seed"]],
        "policy_paths": [p["policy_path"] for p in r["per_seed"]],
        "reproducibility": rep, "model_step_check": msc, "versions": r["versions"],
        "wall_seconds": time.time() - t0,
        "warnings": warnings, "assumptions": assumptions, "not_included": not_included,
        "method": (f"{job['algorithm']} de stable-baselines3 (política MLP) entrenado en el entorno '{job['model']}' de pc-gym "
                   "(integrador casadi) en un proceso aparte (Python 3.11); evaluación determinista en episodios retenidos; "
                   "comparación con política aleatoria uniforme y con el oráculo NMPC de pc-gym (do-mpc + ipopt)"),
        "evidence": {"verified": not reasons, "reasons": reasons, "reproducibility": rep, "protocol": protocol,
                     "model_step_max_rel_diff": None if msc is None else msc["max_rel_diff"],
                     "oracle_replay_state_max_rel_diff": None if (orc is None or "error" in orc) else orc["replay_vs_oracle_state_max_rel_diff"],
                     "certifies": "SOLO que el experimento es reproducible (misma semilla → mismos parámetros y mismos "
                                  "retornos) y que se siguió el protocolo de evaluación; NO certifica la política como "
                                  "controlador (sin garantía de estabilidad ni de restricciones); con escudo, además, que "
                                  "su predictor reproduce el paso del entorno y que los episodios de verificación están "
                                  "completos. La aptitud de la política es verification.verdicts, empírica"},
    }
    if ver is not None:
        out["verification"] = ver
        out["evidence"]["verification_model_check_max_rel_diff"] = ver["model_check_max_rel_diff"]
    return out
