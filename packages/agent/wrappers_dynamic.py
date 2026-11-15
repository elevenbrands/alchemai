"""Agent-facing wrapper of the DWSIM dynamic-mode tool (`packages.dwsim_client.tools_dynamic`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_dynamic as TDY


@beta_tool
def dwsim_dynamic_run(compounds: list[str], property_package: str, streams: list[dict[str, object]],
                      units: list[dict[str, object]], step_s: float, duration_s: float, monitors: list[dict[str, str]],
                      events: list[dict[str, object]] | None = None, settle_s: float | None = None,
                      check_step_halving: bool = True, step_tolerance: float = 0.01, max_points: int = 400) -> str:
    """Dynamic simulation with DWSIM's own dynamic mode (startup, shutdown, step responses) of open tanks, liquid Kv
    valves and PID loops, SI units. streams: {name, dynamic_spec "flow"|"pressure", temperature_K, pressure_Pa,
    mass_flow_kg_s, mole_fractions}; tank outlets are "pressure" and go to a valve; valve outlets need pressure_Pa.
    units: tank {name, feed, product, volume_m3, height_m, initial_level_m, initial_content "inlet"|{temperature_K,
    mole_fractions}}, valve {name, feed, product, kv, opening_pct, characteristic}, pid {name, controlled, manipulated,
    setpoint, kp, ki, kd, reverse_acting, output_bias, output_span, output_min, output_max}. monitors/events address
    {object, variable} (level_m, mass_flow_kg_s, temperature_K, mole_fraction:X, opening_pct, setpoint...); events are
    steps {time_s, object, variable, value} or {time_s, object, mole_fractions}. No input is defaulted. Returns time
    series plus evidence (tank mass balance, step-halving); report evidence.verified and its reasons."""
    return json.dumps(TDY.dwsim_dynamic_run(compounds, property_package, streams, units, step_s, duration_s, monitors,
                                            events, settle_s, check_step_halving, step_tolerance, max_points),
                      default=float, ensure_ascii=False)


AGENT_WRAPPERS = [dwsim_dynamic_run]
