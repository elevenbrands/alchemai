"""Agent-facing wrappers of the advanced-control tools (`packages.dwsim_client.tools_advanced_control`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_advanced_control as TAC


@beta_tool
def mpc_design_and_simulate(sampling_period: float | None = None, prediction_horizon: int | None = None,
                            control_horizon: int | None = None, simulation_time: float | None = None,
                            move_weights: float | list[float] | None = None,
                            process_model: dict[str, object] | list[list[dict[str, object] | None]] | None = None,
                            state_space: dict[str, object] | None = None,
                            output_weights: float | list[float] | None = None,
                            input_weights: float | list[float] | None = None,
                            state_weights: list[float] | list[list[float]] | None = None,
                            input_min: float | list[float | None] | None = None,
                            input_max: float | list[float | None] | None = None,
                            input_rate_max: float | list[float | None] | None = None,
                            output_min: float | list[float | None] | None = None,
                            output_max: float | list[float | None] | None = None,
                            output_soft_weight: float | list[float] | None = None,
                            setpoint_changes: list[dict[str, float]] | None = None,
                            disturbances: list[dict[str, object]] | None = None,
                            disturbance_models: dict[str, object] | list[list[dict[str, object] | None]] | None = None,
                            initial_state: list[float] | None = None, disturbance_model: str = "auto",
                            estimator_q: float = 1.0,
                            plant_model: dict[str, object] | list[list[dict[str, object] | None]] | None = None,
                            pid_baseline: dict[str, object] | None = None, simulation_substeps: int | None = None,
                            return_trajectories: bool = True) -> str:
    """Linear MPC (QDMC-type, exact dead time, offset-free) on the user's FOPDT/SOPDT/TF matrix or state-space model,
    simulated in closed loop on setpoint/disturbance scenarios, plus a PID baseline tuned with the repo's rules (SIMC)
    on the same plant and limits. Deviation variables, model time units. Required design choices: sampling_period,
    prediction_horizon, control_horizon, simulation_time, move_weights. Hard input/rate limits, soft output limits.
    Returns IAE/ISE/TV for MPC and PID, constraint activity, nominal stability check, MPC/PID ratios, trajectories and
    evidence (KKT residuals, hard limits respected, discretisation and step independence)."""
    return json.dumps(TAC.mpc_design_and_simulate(
        sampling_period, prediction_horizon, control_horizon, simulation_time, move_weights, process_model, state_space,
        output_weights, input_weights, state_weights, input_min, input_max, input_rate_max, output_min, output_max,
        output_soft_weight, setpoint_changes, disturbances, disturbance_models, initial_state, disturbance_model,
        estimator_q, plant_model, pid_baseline, simulation_substeps, return_trajectories), default=float, ensure_ascii=False)


@beta_tool
def nmpc_design_and_simulate(model: dict[str, object] | None = None, sampling_period: float | None = None,
                             prediction_horizon: int | None = None, simulation_time: float | None = None,
                             initial_state: dict[str, float] | list[float] | None = None,
                             setpoints: dict[str, float | list[float]] | None = None,
                             state_weights: dict[str, float] | list[float] | None = None,
                             control_horizon: int | None = None,
                             move_weights: dict[str, float] | list[float] | None = None,
                             input_weights: dict[str, float] | list[float] | None = None,
                             input_reference: dict[str, float] | list[float] | None = None,
                             initial_input: dict[str, float] | list[float] | None = None,
                             input_min: dict[str, float] | list[float | None] | None = None,
                             input_max: dict[str, float] | list[float | None] | None = None,
                             input_rate_max: dict[str, float] | list[float | None] | None = None,
                             state_min: dict[str, float] | list[float | None] | None = None,
                             state_max: dict[str, float] | list[float | None] | None = None,
                             state_soft_weight: dict[str, float] | list[float] | None = None,
                             terminal_weights: dict[str, float] | list[float] | None = None,
                             terminal_tolerance: dict[str, float] | list[float | None] | None = None,
                             plant_parameters: dict[str, float] | None = None, collocation_points: int = 3,
                             elements_per_sample: int = 2, return_trajectories: bool = True) -> str:
    """Nonlinear MPC on the user's ODE (or difference-equation) model given as DATA: {states, inputs, parameters,
    rhs: {state: "expression"}} with a restricted, parsed expression language (no code is evaluated). Direct Radau
    collocation with pyomo + ipopt, closed loop against an independent scipy integration of the plant; physical units.
    Required: sampling_period, prediction_horizon, simulation_time, initial_state, setpoints, state_weights. Input,
    rate and state limits, terminal cost/region, plant mismatch optional. Returns trajectories, closed-loop cost, IAE,
    constraint activity and evidence (ipopt optimal each step, one-step prediction vs plant, mesh refinement)."""
    return json.dumps(TAC.nmpc_design_and_simulate(
        model, sampling_period, prediction_horizon, simulation_time, initial_state, setpoints, state_weights,
        control_horizon, move_weights, input_weights, input_reference, initial_input, input_min, input_max,
        input_rate_max, state_min, state_max, state_soft_weight, terminal_weights, terminal_tolerance, plant_parameters,
        collocation_points, elements_per_sample, return_trajectories), default=float, ensure_ascii=False)


@beta_tool
def rl_control_experiment(environment: str | None = None, scenario: str | None = None,
                          env_params: dict[str, object] | None = None, algorithm: str = "PPO",
                          total_timesteps: int | None = None, seeds: list[int] | None = None, eval_episodes: int = 10,
                          hyperparameters: dict[str, object] | None = None, oracle: bool = True,
                          oracle_horizon: int | None = None, pid_baseline: list[dict[str, object]] | None = None,
                          check_reproducibility: bool = True, output_dir: str | None = None,
                          timeout_s: int = 1800) -> str:
    """EXPERIMENT, not a verified controller: train a stable-baselines3 agent (PPO/SAC/DDPG/TD3) on a pc-gym environment
    (or scenario='bloor2024_cstr'), evaluate it on held-out episodes and compare with a random policy, pc-gym's NMPC
    oracle and the user's PID (pid_baseline). total_timesteps and seeds are required. Runs in a separate Python env with
    a timeout; if absent returns available=false and the commands to create it. Returns mean/spread over seeds, gap to
    oracle, violations, learning curves, policy paths; evidence.verified = reproducible and protocol followed only."""
    return json.dumps(TAC.rl_control_experiment(
        environment, scenario, env_params, algorithm, total_timesteps, seeds, eval_episodes, hyperparameters, oracle,
        oracle_horizon, pid_baseline, check_reproducibility, output_dir, timeout_s), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [mpc_design_and_simulate, nmpc_design_and_simulate, rl_control_experiment]
