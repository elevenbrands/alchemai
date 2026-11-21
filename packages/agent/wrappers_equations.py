"""Agent-facing wrappers of the equation solvers (`packages.dwsim_client.tools_equations`). The docstring the model reads
is the one of the plain function (single source of truth, shared with the MCP server)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_equations as TE


def _doc_of(source):
    def set_doc(func):
        func.__doc__ = source.__doc__
        return func
    return set_doc


@beta_tool
@_doc_of(TE.solve_ode_system)
def solve_ode_system(equations: dict[str, str | float], initial_conditions: dict[str, float], t_end: float,
                     t_start: float = 0.0, parameters: dict[str, float] | None = None,
                     definitions: dict[str, str | float] | None = None, output_times: list[float] | None = None,
                     n_output: int = 11, events: list[dict] | None = None, method: str = "LSODA", rtol: float = 1e-8,
                     atol: float = 1e-10, independent_variable: str = "t", oscillation_window_fraction: float = 0.5,
                     max_step: float | None = None) -> str:
    return json.dumps(TE.solve_ode_system(equations, initial_conditions, t_end, t_start, parameters, definitions,
                                          output_times, n_output, events, method, rtol, atol, independent_variable,
                                          oscillation_window_fraction, max_step), default=float, ensure_ascii=False)


@beta_tool
@_doc_of(TE.solve_bvp_system)
def solve_bvp_system(equations: dict[str, str | float], domain: list[float], boundary_conditions: list[str],
                     parameters: dict[str, float] | None = None, unknown_parameters: dict[str, float] | None = None,
                     initial_guess: dict | None = None, definitions: dict[str, str | float] | None = None,
                     singular_terms: dict[str, str] | None = None, output_points: list[float] | None = None,
                     n_output: int = 11, integrals: dict[str, str] | None = None, integral_weight_power: float = 0.0,
                     continuation: dict | None = None, tol: float = 1e-6, n_mesh: int = 41, max_nodes: int = 20000,
                     independent_variable: str = "x") -> str:
    return json.dumps(TE.solve_bvp_system(equations, domain, boundary_conditions, parameters, unknown_parameters,
                                          initial_guess, definitions, singular_terms, output_points, n_output, integrals,
                                          integral_weight_power, continuation, tol, n_mesh, max_nodes,
                                          independent_variable), default=float, ensure_ascii=False)


@beta_tool
@_doc_of(TE.solve_pde_1d)
def solve_pde_1d(variables: list[str], domain: list[float], t_end: float, initial_conditions: dict[str, str | float],
                 boundary_conditions: dict[str, dict], diffusivity: dict[str, str | float],
                 source: dict[str, str | float] | None = None, velocity: dict[str, str | float] | None = None,
                 time_coefficient: dict[str, str | float] | None = None, geometry: str = "slab",
                 parameters: dict[str, float] | None = None, definitions: dict[str, str | float] | None = None,
                 t_start: float = 0.0, output_times: list[float] | None = None, n_output_times: int = 6,
                 output_points: list[float] | None = None, averages: dict[str, str] | None = None, n_points: int = 101,
                 rtol: float = 1e-6, atol: float = 1e-9, method: str = "BDF", convection_scheme: str = "central",
                 space_variable: str = "x", time_variable: str = "t", oscillation_window_fraction: float = 0.5) -> str:
    return json.dumps(TE.solve_pde_1d(variables, domain, t_end, initial_conditions, boundary_conditions, diffusivity,
                                      source, velocity, time_coefficient, geometry, parameters, definitions, t_start,
                                      output_times, n_output_times, output_points, averages, n_points, rtol, atol, method,
                                      convection_scheme, space_variable, time_variable, oscillation_window_fraction),
                      default=float, ensure_ascii=False)


@beta_tool
@_doc_of(TE.solve_algebraic_system)
def solve_algebraic_system(equations: list[str] | None = None, unknowns: dict | None = None,
                           parameters: dict | None = None, definitions: dict | None = None,
                           outputs: dict | None = None, tolerance: float = 1e-9) -> str:
    return json.dumps(TE.solve_algebraic_system(equations, unknowns, parameters, definitions, outputs, tolerance), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [solve_ode_system, solve_bvp_system, solve_pde_1d, solve_algebraic_system]
