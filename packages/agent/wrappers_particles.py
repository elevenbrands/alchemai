"""Agent-facing wrappers of the non-Newtonian pipe flow and particle tools (`packages.dwsim_client.tools_particles`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_particles as TPA


@beta_tool
def non_newtonian_pipe_flow(volumetric_flow_m3_s: float, diameter_m: float, length_m: float, density_kg_m3: float,
                            rheology: dict[str, object], profile_points: int = 21) -> str:
    """Frictional ΔP of a power-law, Bingham or Herschel-Bulkley liquid in a straight smooth pipe, with regime.
    rheology {model: power_law (K_Pa_sn, n) | bingham (yield_stress_Pa, plastic_viscosity_Pa_s) | herschel_bulkley
    (yield_stress_Pa, K_Pa_sn, n)} is the user's (true K, not K'). Laminar is exact (Rabinowitsch-Mooney, Buckingham-
    Reiner) with velocity profile and plug radius; transition by Ryan-Johnson/Hanks; turbulent Dodge-Metzner (power law),
    Darby (Bingham), Herschel-Bulkley turbulent is a flagged estimate. Returns regime, pressure_drop_Pa,
    wall_shear_stress_Pa,
    f_fanning, generalized_reynolds_metzner_reed, hedstrom_number, transition, laminar_solution, evidence, warnings."""
    return json.dumps(TPA.non_newtonian_pipe_flow(volumetric_flow_m3_s, diameter_m, length_m, density_kg_m3, rheology,
                                                  profile_points), default=float, ensure_ascii=False)


@beta_tool
def particle_settling(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
                      fluid_viscosity_Pa_s: float, sphericity: float, projected_area_ratio: float | None = None,
                      drag_method: str = "Clift", solids_volume_fraction: float | None = None,
                      column_diameter_m: float | None = None, thickener: dict[str, object] | None = None) -> str:
    """Terminal velocity of one particle (drag of the fluids library, default Clift standard curve; spread of 4
    correlations reported), Cd, Re and regime; sphericity < 1 needs projected_area_ratio (Song-Xu). Optional hindered
    settling by Richardson-Zaki (solids_volume_fraction, column_diameter_m for d/D) and thickener area by Coe-Clevenger:
    thickener {feed_flow_m3_s, feed_solids_kg_m3, underflow_solids_kg_m3, batch_tests {concentration_kg_m3,
    settling_velocity_m_s}} (without batch_tests a flagged Richardson-Zaki estimate). All properties are the user's.
    Returns terminal_velocity_m_s, reynolds_terminal, drag_coefficient, regime, hindered, thickener, evidence, warnings."""
    return json.dumps(TPA.particle_settling(particle_diameter_m, particle_density_kg_m3, fluid_density_kg_m3,
                                            fluid_viscosity_Pa_s, sphericity, projected_area_ratio, drag_method,
                                            solids_volume_fraction, column_diameter_m, thickener),
                      default=float, ensure_ascii=False)


@beta_tool
def fluidization(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
                 fluid_viscosity_Pa_s: float, sphericity: float, voidage_mf: float | None = None,
                 bed_mass_kg: float | None = None, column_diameter_m: float | None = None,
                 superficial_velocity_m_s: float | None = None, projected_area_ratio: float | None = None,
                 drag_method: str = "Clift") -> str:
    """Fluidized bed: minimum fluidization by Ergun (needs the user's voidage_mf) and Wen-Yu, terminal velocity, operating
    window u_mf..u_t, bed ΔP = buoyant weight/area (bed_mass_kg + column_diameter_m), height at u_mf, and at
    superficial_velocity_m_s the state (fixed / fluidized / transport), ΔP and, for liquids, Richardson-Zaki expansion.
    Gas beds: no expansion (bubbling). Returns u_mf_ergun_m_s, u_mf_wen_yu_m_s, operating_window_m_s,
    fluidized_bed_pressure_drop_Pa, operating_point, evidence, warnings."""
    return json.dumps(TPA.fluidization(particle_diameter_m, particle_density_kg_m3, fluid_density_kg_m3,
                                       fluid_viscosity_Pa_s, sphericity, voidage_mf, bed_mass_kg, column_diameter_m,
                                       superficial_velocity_m_s, projected_area_ratio, drag_method),
                      default=float, ensure_ascii=False)


@beta_tool
def cyclone_performance(gas_flow_m3_s: float, gas_density_kg_m3: float, gas_viscosity_Pa_s: float,
                        particle_density_kg_m3: float, body_diameter_m: float, geometry: str | dict[str, object] = "lapple",
                        particle_size_distribution: dict[str, list[float]] | None = None,
                        diameters_m: list[float] | None = None, pressure_drop_K: float = 16.0) -> str:
    """Reverse-flow gas cyclone by Lapple: inlet velocity, effective turns, cut size d50, grade efficiency
    1/(1+(d50/d)^2), overall efficiency over the user's PSD {diameter_m, mass_fraction} (no default PSD), and
    Shepherd-Lapple ΔP = K·ab/De²·½ρV_i² (K = 16, tangential inlet). geometry: preset of ratios to D ('lapple',
    'stairmand_he', 'stairmand_ht', 'swift_he', 'swift_gp', 'swift_ht') or a dict of ratios; body_diameter_m is the
    user's. No Barth-Muschelknautz, no solids-loading effect. Returns cut_diameter_m, grade_efficiency, overall,
    pressure_drop_Pa, inlet_velocity_m_s, evidence, warnings."""
    return json.dumps(TPA.cyclone_performance(gas_flow_m3_s, gas_density_kg_m3, gas_viscosity_Pa_s, particle_density_kg_m3,
                                              body_diameter_m, geometry, particle_size_distribution, diameters_m,
                                              pressure_drop_K), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [non_newtonian_pipe_flow, particle_settling, fluidization, cyclone_performance]
