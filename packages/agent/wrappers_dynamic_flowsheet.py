"""Agent-facing wrappers of the own dynamic flowsheet tools (`packages.dwsim_client.tools_dynamic_flowsheet`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_dynamic_flowsheet as TDF


@beta_tool
def dynamic_flowsheet_run(flowsheet: dict[str, object], t_end_s: float, events: list[dict[str, object]] | None = None,
                          outputs: list[str] | None = None, limits: list[dict[str, object]] | None = None,
                          start_from_steady_state: bool = False, n_points: int = 201, method: str = "Radau",
                          rtol: float = 1e-8) -> str:
    """First-principles dynamic simulation (independent of DWSIM) of a flowsheet given as JSON: tanks (gravity, valve
    Cv/Kv, pump curve, overflow, fill from empty), mixing/heated/jacketed tanks, CSTR with Arrhenius kinetics and energy
    balance, lumped exchangers, gas vessels (pressurization, choked blowdown), isothermal flash drums connected by
    streams, binary distillation columns (Skogestad model; LV/DV level and composition loops), sensor lag/dead time, PID.
    flowsheet = {components, liquid, units, streams, heat_links, blocks, controllers}, SI. events = [{time_s, target
    parameter, value, ramp_s}]; limits = [{variable, max, min}]. No physical input is defaulted. Returns series,
    settling/peaks, incidents, limit violations, final state and evidence (balances, tolerance refinement); report
    evidence.verified and its reasons."""
    return json.dumps(TDF.dynamic_flowsheet_run(flowsheet, t_end_s, events, outputs, limits, start_from_steady_state,
                                                n_points, method, rtol), default=float, ensure_ascii=False)


@beta_tool
def dynamic_flowsheet_steady_state(flowsheet: dict[str, object], outputs: list[str] | None = None) -> str:
    """Steady state of a dynamic flowsheet (same JSON as dynamic_flowsheet_run) found by Newton from the given initial
    state, with local stability: eigenvalues, time constants, oscillation periods. With multiple steady states
    (exothermic CSTR) it returns the one nearest the initial state; repeat from other initial states to map them.
    Report evidence.verified and whether the state is stable."""
    return json.dumps(TDF.dynamic_flowsheet_steady_state(flowsheet, outputs), default=float, ensure_ascii=False)


@beta_tool
def dynamic_step_test(flowsheet: dict[str, object], input: str, step_size: float, output: str, t_end_s: float,
                      step_time_s: float, n_points: int = 200, start_from_steady_state: bool = True) -> str:
    """Open-loop step test on the nonlinear dynamic flowsheet: step parameter `input` by `step_size` at `step_time_s`
    and record variable `output`. A controller that moves `input` goes to manual at its closed-loop steady output.
    Returns the response, apparent gain and fopdt_identify_arguments to pass unchanged to fopdt_identify, then
    pid_tuning. step_size, t_end_s and step_time_s are the user's; none is defaulted."""
    return json.dumps(TDF.dynamic_step_test(flowsheet, input, step_size, output, t_end_s, step_time_s, n_points,
                                            start_from_steady_state), default=float, ensure_ascii=False)


@beta_tool
def dynamic_flash_drum(flash_drum: dict[str, object], t_end_s: float, events: list[dict[str, object]] | None = None,
                       start_from_steady_state: bool = False, n_points: int = 201) -> str:
    """Isothermal dynamic flash drum: liquid and vapour holdup in equilibrium (constant K-values or constant relative
    volatility with Raoult), pressure from a vapour valve, level from a liquid valve, SI. Events step feed flow or
    composition, valve openings or back-pressures. No physical input is defaulted. Returns pressure, level, flows and
    compositions, the Rachford-Rice check of the products against the feed flash, and evidence (component balances,
    tolerance refinement); report evidence.verified and its reasons."""
    return json.dumps(TDF.dynamic_flash_drum(flash_drum, t_end_s, events, start_from_steady_state, n_points),
                      default=float, ensure_ascii=False)


AGENT_WRAPPERS = [dynamic_flowsheet_run, dynamic_flowsheet_steady_state, dynamic_step_test, dynamic_flash_drum]
