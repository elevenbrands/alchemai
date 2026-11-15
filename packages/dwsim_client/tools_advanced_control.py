"""Agent tools for advanced process control (bloque advcontrol): linear MPC with a PID baseline on the user's model
(`packages.control.mpc`), nonlinear MPC on a user ODE model given as data (`packages.control.nmpc`, pyomo + ipopt), both
in the repo environment, and a reinforcement-learning experiment on pc-gym
(`packages.control.rl_runner`, which runs the RL stack in the separate environment `~/tools/alchemai-rl/.venv`)."""
from __future__ import annotations

import math

import numpy as np

from packages.control import mpc as MPC
from packages.control import nmpc as NMPC
from packages.control import rl_runner as RL


def _clean(x):
    """JSON-safe copy: numpy scalars/arrays to Python, NaN to None and ±inf to the strings 'inf'/'-inf'."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def mpc_design_and_simulate(sampling_period: float | None = None, prediction_horizon: int | None = None,
                            control_horizon: int | None = None, simulation_time: float | None = None,
                            move_weights=None, process_model=None, state_space: dict | None = None,
                            output_weights=None, input_weights=None, state_weights=None, input_min=None,
                            input_max=None, input_rate_max=None, output_min=None, output_max=None,
                            output_soft_weight=None, setpoint_changes: list[dict] | None = None,
                            disturbances: list[dict] | None = None, disturbance_models=None,
                            initial_state: list[float] | None = None, disturbance_model: str = "auto",
                            estimator_q: float = 1.0, plant_model=None, pid_baseline: dict | None = None,
                            simulation_substeps: int | None = None, return_trajectories: bool = True) -> dict:
    """Diseña un MPC lineal (tipo QDMC) sobre TU modelo y lo simula en lazo cerrado en tu escenario, junto con una línea
    base PID sintonizada con las reglas del repositorio (SIMC por defecto) sobre la misma planta y con los mismos límites.

    Todo en variables de DESVIACIÓN respecto al estado estacionario inicial (u = 0, y = 0 en t = 0) y en las unidades de
    tiempo del modelo (no se convierte nada). Nada del proceso ni de la sintonía se supone.

    Modelo (exactamente uno):
    - process_model: función de transferencia. SISO = dict {gain, time_constants, lead_time_constants, dead_time,
      integrating} (como pid_tuning) o {num, den, dead_time} (potencias descendentes de s). MIMO pequeño = matriz
      [fila = salida][columna = entrada] de esos dicts (null = esa entrada no afecta a esa salida). El tiempo muerto es
      exacto (estados de entradas pasadas, sin Padé), también fraccionario respecto al periodo.
    - state_space: {A, B, C, input_delays opcional, dt opcional (modelo discreto)}; D debe ser 0.
    Diseño (decisiones del usuario, obligatorias): sampling_period (> 0), prediction_horizon P y control_horizon M (en
    periodos, 1 ≤ M ≤ P), simulation_time, move_weights (peso de cada Δu; escalar o uno por entrada). Opcionales:
    output_weights (1 por defecto, se avisa), input_weights (penaliza u), state_weights (solo state_space).
    Restricciones (en desviación, deben contener el 0): input_min/input_max y input_rate_max (|Δu| por periodo) duras;
    output_min/output_max blandas con output_soft_weight (por defecto 10⁴ × mayor peso de salida, se avisa). Usa null
    en una lista para «sin límite» en esa variable.
    Escenario: setpoint_changes [{time, output, value}] (valor nuevo de la consigna en desviación), disturbances
    [{time, type 'input'|'output'|'model', index, value}] (escalones no medidos; 'model' usa disturbance_models, una
    matriz salida × perturbación de funciones de transferencia), initial_state (solo state_space).
    disturbance_model: 'auto' (salida = sesgo DMC si el modelo es estable; entrada + filtro de Kalman si es integrante),
    'output', 'input' o 'none' (sin acción integral). estimator_q = varianza perturbación/ruido del filtro de Kalman.
    plant_model: planta distinta del modelo del controlador (prueba de robustez frente a desajuste).
    pid_baseline: {rule 'SIMC'|'IMC'|..., controller_type 'PI'|'PID', tau_c, pairing [[salida, entrada], ...],
    controllers [{Kc, tau_I, tau_D, form}] para usar TUS ajustes, skip true para omitirla}.

    Devuelve: model (estados, ganancia estacionaria, RGA si es cuadrado), controller (horizontes, pesos, ganancia sin
    restricciones, autovalores del lazo nominal y estabilidad, distancia al LQR de horizonte infinito), mpc (índices IAE,
    ISE, ITAE, TV_u, error final por salida; actividad de restricciones: fracción del tiempo en límite de entrada y de
    velocidad, violación de límites blandos; estadística de los QP), pid_baseline (lazos, sintonía, mismos índices),
    comparison (cocientes MPC/PID; < 1 = MPC mejor), trajectories, prediction_at_first_event, warnings, assumptions,
    not_included, method y evidence (verified = QP resueltos a su tolerancia KKT, restricciones duras respetadas,
    discretización del modelo comprobada en dos mallas y en forma cerrada, IAE independiente del paso de integración;
    no certifica que el modelo represente a la planta). La estabilidad nominal sin restricciones se COMPRUEBA, no se
    garantiza (no hay costo terminal): mira controller.nominal_closed_loop.stable y los avisos."""
    return _clean(MPC.mpc_closed_loop(
        sampling_period=sampling_period, prediction_horizon=prediction_horizon, control_horizon=control_horizon,
        simulation_time=simulation_time, move_weights=move_weights, process_model=process_model, state_space=state_space,
        output_weights=output_weights, input_weights=input_weights, state_weights=state_weights, input_min=input_min,
        input_max=input_max, input_rate_max=input_rate_max, output_min=output_min, output_max=output_max,
        output_soft_weight=output_soft_weight, setpoint_changes=setpoint_changes, disturbances=disturbances,
        disturbance_models=disturbance_models, initial_state=initial_state, disturbance_model=disturbance_model,
        estimator_q=estimator_q, plant_model=plant_model, pid_baseline=pid_baseline,
        simulation_substeps=simulation_substeps, return_trajectories=return_trajectories))


def nmpc_design_and_simulate(model: dict | None = None, sampling_period: float | None = None,
                             prediction_horizon: int | None = None, simulation_time: float | None = None,
                             initial_state=None, setpoints: dict | None = None, state_weights=None,
                             control_horizon: int | None = None, move_weights=None, input_weights=None,
                             input_reference=None, initial_input=None, input_min=None, input_max=None,
                             input_rate_max=None, state_min=None, state_max=None, state_soft_weight=None,
                             terminal_weights=None, terminal_tolerance=None, plant_parameters: dict | None = None,
                             collocation_points: int = 3, elements_per_sample: int = 2,
                             return_trajectories: bool = True) -> dict:
    """MPC NO LINEAL sobre TU modelo de EDO (o de ecuaciones en diferencias), resuelto con pyomo + ipopt por colocación
    directa (Radau IIA), simulado en lazo cerrado contra la planta integrada aparte con scipy. Variables en unidades
    FÍSICAS del modelo (no de desviación). Nada del proceso ni de la sintonía se supone.

    - model (datos, nunca código): {states: ['Ca', 'T'], inputs: ['Tc'], parameters: {nombre: número},
      rhs: {estado: 'expresión'}, time: 'continuous' (rhs = dx/dt, por defecto) | 'discrete' (rhs = x[k+1])}. Las
      expresiones solo admiten números, los nombres declarados, + − * / ** (o ^), signo y exp, log, log10, sqrt, sin,
      cos, tan, sinh, cosh, tanh, asin, acos, atan; cualquier otra cosa se rechaza (no se evalúa código).
      Ej.: 'q/V*(Caf - Ca) - k0*exp(-EA_over_R/T)*Ca'.
    - Obligatorios: sampling_period, prediction_horizon (periodos), simulation_time, initial_state ({estado: valor} o
      lista en el orden de states), setpoints ({estado: valor} o {estado: lista de un valor por periodo}; constantes
      en el horizonte, sin anticipación), state_weights ({estado: q > 0} para cada estado con consigna).
    - Opcionales: control_horizon (≤ P; la entrada se mantiene después), move_weights (peso de Δu; requiere
      initial_input), input_weights + input_reference (peso de u − u_ref), input_min/input_max (duros),
      input_rate_max (|Δu| por periodo; requiere initial_input), state_min/state_max (duros en muestreos y nodos de
      colocación; blandos con state_soft_weight), terminal_weights (costo terminal extra) y terminal_tolerance
      (región terminal |x_N − r| ≤ tol), plant_parameters (planta con parámetros distintos: prueba de robustez),
      collocation_points (1-5) y elements_per_sample (malla de colocación).
    Costo: Σ_{j=1..P} q·(x_j − r)² + p·(x_P − r)² + Σ s·Δu² + Σ w·(u − u_ref)² (+ ρ·ε² si blando).

    Devuelve: controller (horizontes, pesos, tamaño del NLP), nmpc (indices: costo en lazo cerrado con el mismo costo
    de etapa, IAE en muestreos, error final, TV_u; constraint_activity; solver con terminaciones de ipopt y tiempos;
    first_plan = plan y predicción del primer problema; objective_per_step), trajectories, warnings (siempre: óptimo
    LOCAL), assumptions, not_included, method y evidence (verified = ipopt óptimo en cada paso, predicción de un paso
    de la colocación igual a la integración de la planta (tolerancia 1e-4 relativa), primer problema independiente
    de la malla (duplicando elementos, 1e-3), límites duros de estado respetados en los muestreos). Sin estimador:
    realimentación del estado medido sin ruido."""
    return _clean(NMPC.nmpc_closed_loop(
        model=model, sampling_period=sampling_period, prediction_horizon=prediction_horizon,
        simulation_time=simulation_time, initial_state=initial_state, setpoints=setpoints, state_weights=state_weights,
        control_horizon=control_horizon, move_weights=move_weights, input_weights=input_weights,
        input_reference=input_reference, initial_input=initial_input, input_min=input_min, input_max=input_max,
        input_rate_max=input_rate_max, state_min=state_min, state_max=state_max, state_soft_weight=state_soft_weight,
        terminal_weights=terminal_weights, terminal_tolerance=terminal_tolerance, plant_parameters=plant_parameters,
        collocation_points=collocation_points, elements_per_sample=elements_per_sample,
        return_trajectories=return_trajectories))


def rl_control_experiment(environment: str | None = None, scenario: str | None = None, env_params: dict | None = None,
                          algorithm: str = "PPO", total_timesteps: int | None = None, seeds: list[int] | None = None,
                          eval_episodes: int = 10, hyperparameters: dict | None = None, oracle: bool = True,
                          oracle_horizon: int | None = None, pid_baseline: list[dict] | None = None,
                          check_reproducibility: bool = True, output_dir: str | None = None,
                          timeout_s: int = RL.DEFAULT_TIMEOUT_S) -> dict:
    """EXPERIMENTO de aprendizaje por refuerzo (no un controlador verificado): entrena un agente de stable-baselines3
    (PPO, SAC, DDPG o TD3) en un entorno de pc-gym (Bloor et al. 2024), lo evalúa con la política determinista en
    episodios retenidos y lo compara con una política aleatoria, con el oráculo NMPC de pc-gym (modelo perfecto) y, si lo
    das, con tu PID. Corre en un entorno de Python aparte (~/tools/alchemai-rl/.venv) con límite de tiempo; si no existe,
    devuelve available = false y los comandos para crearlo.

    - environment: modelo de pc-gym ('cstr', 'first_order_system', 'four_tank', 'multistage_extraction',
      'distillation_column', 'heat_exchanger', 'crystallization', ...), o scenario = 'bloor2024_cstr' (configuración
      publicada: CSTR, 60 pasos en 25 min, consignas de Ca 0.85/0.9/0.87, Tc en [295, 302] K, ruido 0.1 %).
    - env_params (si no hay scenario, todo obligatorio; con scenario, sustituye claves): N (pasos), tsim (duración, en
      las unidades del modelo), SP {estado: N consignas}, x0 (estados y luego una entrada por consigna), a_space y
      o_space {low, high} (límites físicos de acción y observación); opcionales r_scale, noise_percentage (fracción),
      constraints {estado: [valores]} con cons_type {estado: ['<=', '>=']} y r_penalty.
    - total_timesteps (pasos de entrenamiento por semilla) y seeds (lista de semillas, ≥ 3 para estimar la dispersión):
      obligatorios, son el presupuesto del experimento. hyperparameters: kwargs de stable-baselines3 (por defecto los
      de la librería, se avisa). eval_episodes: episodios de evaluación por semilla. oracle_horizon: pasos del NMPC.
    - pid_baseline: [{state, input, Kc, tau_I (null = sin integral), tau_D, bias}] en unidades físicas del modelo
      (Kc en unidades de acción por unidad del estado, con su signo; bias = acción en t = 0); PID discreto con
      dt = tsim/N sobre la misma observación con ruido.
    - timeout_s: el proceso se mata (grupo completo) si lo excede. Medido aquí (CSTR): PPO con 50 000 pasos ~30 s por
      semilla; SAC ~5 s por cada 1000 pasos (unas 8 veces más lento); la comprobación de reproducibilidad entrena la
      primera semilla dos veces.

    Devuelve: eval_return (media y dispersión ENTRE semillas), per_seed (retornos, curva de aprendizaje, violaciones,
    ruta de la política guardada .zip y su hash), random_policy, oracle (retorno, violaciones), gap_to_oracle (absoluto y
    fracción de la distancia aleatorio→oráculo que queda), pid_baseline y vs_pid, beats_random, reproducibility,
    model_step_check (un paso del CSTR frente a una integración independiente de sus EDO publicadas), versions,
    warnings (siempre: sin garantía de estabilidad), assumptions, not_included, method y evidence (verified = SOLO
    reproducible con la misma semilla y protocolo de evaluación seguido). Retorno = suma de la recompensa del entorno
    (−Σ(x − consigna)²); mayor es mejor."""
    kw = dict(environment=environment, scenario=scenario, env_params=env_params, algorithm=algorithm,
              total_timesteps=total_timesteps, seeds=seeds, eval_episodes=eval_episodes, hyperparameters=hyperparameters,
              oracle=oracle, oracle_horizon=oracle_horizon, check_reproducibility=check_reproducibility,
              output_dir=output_dir, pid_baseline=pid_baseline)
    if not RL.rl_available():
        job, _ = RL.build_job(**kw)              # missing user data is still reported first (ValueError)
        msg = RL.missing_environment_message()
        return {"available": False, "message": msg, "create_commands": list(RL.CREATE_COMMANDS),
                "experiment": {"environment": job["model"], "algorithm": job["algorithm"], "seeds": job["seeds"],
                               "total_timesteps_per_seed": job["total_timesteps"]},
                "warnings": [msg, RL.STANDING_WARNING], "assumptions": [], "not_included": ["no se ejecutó nada"],
                "method": "ninguno: el entorno de RL aparte no está instalado",
                "evidence": {"verified": False, "reasons": ["entorno de RL (~/tools/alchemai-rl/.venv) ausente"]}}
    out = RL.rl_experiment(timeout_s=int(timeout_s), **kw)
    out["available"] = True
    return _clean(out)


ADVANCED_CONTROL_TOOLS = [mpc_design_and_simulate, nmpc_design_and_simulate, rl_control_experiment]
