"""Agent-facing wrappers of the momentum-transfer tools (`packages.dwsim_client.tools_momentum`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_momentum as TM


@beta_tool
def adiabatic_gas_pipe(P1_Pa: float, T1_K: float, inner_diameter_m: float, length_m: float, inlet_basis: str,
                       mass_flow_kg_s: float | None = None, P2_Pa: float | None = None, gamma: float | None = None,
                       molar_mass_kg_mol: float | None = None, viscosity_Pa_s: float | None = None,
                       compounds: dict[str, float] | None = None, roughness_m: float = 4.57e-5,
                       darcy_friction_factor: float | None = None) -> str:
    """Adiabatic compressible gas flow with friction in a straight pipe (Fanno, ideal gas, subsonic): outlet P/T/Mach for
    mass_flow_kg_s, or the flow for P2_Pa (give exactly one); choking, critical length L*, max flow, and the isothermal line
    for comparison. inlet_basis 'static' (line P1, T1) or 'stagnation' (reservoir). Gas: gamma + molar_mass_kg_mol, or
    compounds {name: mole fraction} read from the property database (said so). viscosity_Pa_s for Colebrook unless
    darcy_friction_factor (Darcy = 4·Fanning). Returns inlet, outlet, choked, feasible, critical_length_m,
    isothermal_comparison, evidence (ODE of the primitive equations), warnings."""
    return json.dumps(TM.adiabatic_gas_pipe(P1_Pa, T1_K, inner_diameter_m, length_m, inlet_basis, mass_flow_kg_s, P2_Pa,
                                            gamma, molar_mass_kg_mol, viscosity_Pa_s, compounds, roughness_m,
                                            darcy_friction_factor), default=float, ensure_ascii=False)


@beta_tool
def two_phase_pipe_inclined(mass_flow_kg_s: float, quality: float, liquid_density_kg_m3: float,
                            gas_density_kg_m3: float, liquid_viscosity_Pa_s: float, gas_viscosity_Pa_s: float,
                            surface_tension_N_m: float, inner_diameter_m: float, length_m: float, angle_deg: float,
                            inlet_pressure_Pa: float, roughness_m: float = 4.57e-5, n_segments: int = 20) -> str:
    """Gas-liquid pressure drop in an inclined or vertical pipe (angle_deg over horizontal, + uphill): Beggs-Brill with
    inclination (recommended) and a separated model (Woldesemayat-Ghajar void + Whalley-chosen friction) for the spread.
    quality = gas mass fraction (constant, no flashing); properties at the inlet; inlet_pressure_Pa absolute. Returns
    P_out_Pa, dP split into gravity, friction and acceleration, flow pattern and liquid holdup in/out, profile,
    separated_model, method_spread_rel, evidence (check against fluids.Beggs_Brill, segment independence), warnings."""
    return json.dumps(TM.two_phase_pipe_inclined(mass_flow_kg_s, quality, liquid_density_kg_m3, gas_density_kg_m3,
                                                 liquid_viscosity_Pa_s, gas_viscosity_Pa_s, surface_tension_N_m,
                                                 inner_diameter_m, length_m, angle_deg, inlet_pressure_Pa, roughness_m,
                                                 n_segments), default=float, ensure_ascii=False)


@beta_tool
def water_hammer(length_m: float, inner_diameter_m: float, density_kg_m3: float, initial_flow_m3_s: float,
                 upstream_pressure_Pa: float, downstream_pressure_Pa: float, closure_time_s: float | None = None,
                 closure_exponent: float | None = None, closure_curve: dict[str, list[float]] | None = None,
                 wave_speed_m_s: float | None = None, fluid_bulk_modulus_Pa: float | None = None,
                 pipe_elastic_modulus_Pa: float | None = None, wall_thickness_m: float | None = None,
                 restraint_factor: float | None = None, pipe_anchoring: str | None = None,
                 poisson_ratio: float | None = None, darcy_friction_factor: float | None = None,
                 viscosity_Pa_s: float | None = None, roughness_m: float | None = None,
                 vapor_pressure_Pa: float | None = None, elevation_change_m: float = 0.0, n_reaches: int = 40,
                 simulation_time_s: float | None = None) -> str:
    """Water hammer of a valve closure in a reservoir-pipe-valve line: wave speed (Korteweg, or wave_speed_m_s), Joukowsky
    surge, fast/slow closure vs 2L/a, and the method-of-characteristics transient with friction: max/min pressure (where,
    when), valve history, cavitation flag (needs vapor_pressure_Pa). Absolute pressures upstream (reservoir side) and
    downstream (valve discharge). Closure: closure_time_s + closure_exponent (tau = (1 - t/tc)^Em) or closure_curve
    {time_s, tau}. Friction: darcy_friction_factor (0 = none) or viscosity_Pa_s + roughness_m. No column separation."""
    return json.dumps(TM.water_hammer(length_m, inner_diameter_m, density_kg_m3, initial_flow_m3_s, upstream_pressure_Pa,
                                      downstream_pressure_Pa, closure_time_s, closure_exponent, closure_curve,
                                      wave_speed_m_s, fluid_bulk_modulus_Pa, pipe_elastic_modulus_Pa, wall_thickness_m,
                                      restraint_factor, pipe_anchoring, poisson_ratio, darcy_friction_factor,
                                      viscosity_Pa_s, roughness_m, vapor_pressure_Pa, elevation_change_m, n_reaches,
                                      simulation_time_s), default=float, ensure_ascii=False)


@beta_tool
def pump_selection(pump_curve: dict[str, list[float]], speed_rpm: float, density_kg_m3: float,
                   system: dict[str, object], curve_fit: str = "pchip", stages: int = 1, efficiency: float | None = None,
                   motor_efficiency: float | None = None, allow_efficiency_estimate: bool = False,
                   suction: dict[str, float] | None = None, vapor_pressure_Pa: float | None = None,
                   new_speed_rpm: float | None = None, impeller_diameter_m: float | None = None,
                   new_impeller_diameter_m: float | None = None, target_flow_m3_s: float | None = None) -> str:
    """Centrifugal pump on the user's system: operating point, efficiency, shaft/motor power (NEMA size), specific speed at
    the BEP and impeller type, affinity laws (new speed, impeller trim, speed for a target flow), NPSHa vs NPSHr.
    pump_curve {flow_m3_s, head_m, efficiency?, NPSHr_m?} at speed_rpm; system {static_head_m, resistance_s2_m5} or
    {static_head_m, flow_m3_s, head_m}. Efficiency only from the user, or the published Corripio correlation when
    allow_efficiency_estimate=True. suction {vessel_pressure_Pa, liquid_level_above_pump_m, losses_m |
    loss_coefficient_s2_m5} + vapor_pressure_Pa for NPSH."""
    return json.dumps(TM.pump_selection(pump_curve, speed_rpm, density_kg_m3, system, curve_fit, stages, efficiency,
                                        motor_efficiency, allow_efficiency_estimate, suction, vapor_pressure_Pa,
                                        new_speed_rpm, impeller_diameter_m, new_impeller_diameter_m, target_flow_m3_s),
                      default=float, ensure_ascii=False)


@beta_tool
def mechanical_energy_balance(density_kg_m3: float, point_1: dict, point_2: dict, solve_for: str,
                              volumetric_flow_m3_s: float | None = None, mass_flow_kg_s: float | None = None,
                              shaft_power_W: float | None = None, friction_loss_J_kg: float | None = None,
                              efficiency: float | None = None, g_m_s2: float = 9.80665,
                              kinetic_energy_factor: float = 1.0) -> str:
    """Mechanical energy balance (extended Bernoulli) between two points of an incompressible fluid:
    (P2-P1)/rho + (u2^2-u1^2)/2 + g(z2-z1) + F = W_shaft/m. Includes kinetic energy (pipe_network ignores it): siphons,
    tank discharge, venturi, pump/turbine power, friction loss from measured data. point_1/point_2: {P_Pa, z_m, and one of
    velocity_m_s | diameter_m | area_m2 | large_surface: true}. solve_for: the ONE unknown among P1_Pa, P2_Pa, z1_m, z2_m,
    shaft_power_W (to the fluid: pump > 0, turbine < 0), friction_loss_J_kg, volumetric_flow_m3_s. Friction is the user's
    (0 if absent, stated). Returns the value, the five terms in J/kg, velocities, drive power with efficiency, evidence."""
    return json.dumps(TM.mechanical_energy_balance(density_kg_m3, point_1, point_2, solve_for, volumetric_flow_m3_s, mass_flow_kg_s, shaft_power_W, friction_loss_J_kg, efficiency, g_m_s2, kinetic_energy_factor), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [adiabatic_gas_pipe, two_phase_pipe_inclined, water_hammer, pump_selection, mechanical_energy_balance]
