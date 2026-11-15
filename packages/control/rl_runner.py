"""Reinforcement learning for a process-control task on pc-gym: repo side (bloque advcontrol, part B).

What it answers: «if I train a standard RL agent on this benchmark control problem with this budget, how good is it
against a random policy and against the NMPC oracle, how much does it vary between seeds, and is the experiment
reproducible?» — an EXPERIMENT, never a verified controller.

The engine lives in another environment (admission rule for engines outside the repo's Python): pc-gym 0.1.8 (MIT; Bloor et
al. 2024, arXiv:2410.22093), gymnasium (MIT), stable-baselines3 (MIT), torch (BSD-style), casadi and do-mpc (LGPL-3, used
unmodified as libraries by pc-gym's oracle) in `~/tools/alchemai-rl/.venv` (Python 3.11). This module only builds the job,
launches `rl_worker.py` with that interpreter in its own process group with a timeout (the group is killed on expiry), and
reads the JSON it writes. Nothing of that stack is imported here.

`evidence.verified` certifies only: (1) same seed and budget → identical policy parameters and identical evaluation returns
(the first seed is trained twice); (2) the evaluation protocol was followed (deterministic policy, evaluation noise seeds
disjoint from the training seeds, complete episodes, every return computed by the environment itself — also the oracle's,
whose inputs are replayed in the environment); (3) for the cstr, one environment step agrees with an independent
integration of the published ODEs. It says nothing about stability or constraint satisfaction of the policy."""
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
    "Una política de RL entrenada NO es control verificado: no tiene garantía de estabilidad ni de respeto de restricciones "
    "fuera de los episodios evaluados. Antes de cualquier uso hay que validarla contra el modelo riguroso de la planta (con "
    "perturbaciones, ruido y desajuste), y mantener el control convencional y las protecciones (SIS) como respaldo.")


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


def build_job(*, environment: str | None, scenario: str | None, env_params: dict | None, algorithm: str,
              total_timesteps: int | None, seeds: list[int] | None, eval_episodes: int, hyperparameters: dict | None,
              oracle: bool, oracle_horizon: int | None, check_reproducibility: bool, output_dir: str | None,
              pid_baseline: list[dict] | None = None) -> tuple[dict, list[str]]:
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
    tag = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]
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


def rl_experiment(*, environment: str | None = None, scenario: str | None = None, env_params: dict | None = None,
                  algorithm: str = "PPO", total_timesteps: int | None = None, seeds: list[int] | None = None,
                  eval_episodes: int = 10, hyperparameters: dict | None = None, oracle: bool = True,
                  oracle_horizon: int | None = None, check_reproducibility: bool = True, output_dir: str | None = None,
                  timeout_s: int = DEFAULT_TIMEOUT_S, pid_baseline: list[dict] | None = None) -> dict:
    """Train, evaluate and compare. Raises ValueError (missing/invalid input) or RuntimeError (environment absent, timeout,
    failure inside pc-gym), both in Spanish."""
    job, assumptions = build_job(environment=environment, scenario=scenario, env_params=env_params, algorithm=algorithm,
                                 total_timesteps=total_timesteps, seeds=seeds, eval_episodes=eval_episodes,
                                 hyperparameters=hyperparameters, oracle=oracle, oracle_horizon=oracle_horizon,
                                 check_reproducibility=check_reproducibility, output_dir=output_dir,
                                 pid_baseline=pid_baseline)
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
        "generalización a otras consignas, estados iniciales, perturbaciones o parámetros: solo se evalúa el escenario dado",
        "garantías de estabilidad o de seguridad (capa de seguridad, RL con restricciones, verificación formal)",
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
                                  "controlador (sin garantía de estabilidad ni de restricciones)"},
    }
    return out
