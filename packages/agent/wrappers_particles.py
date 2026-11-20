"""Agent-facing wrappers of the non-Newtonian pipe flow and particle tools (`packages.dwsim_client.tools_particles`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_particles as TPA


@beta_tool
def non_newtonian_pipe_flow(volumetric_flow_m3_s: float | None = None, diameter_m: float | None = None,
                            length_m: float | None = None, density_kg_m3: float | None = None,
                            rheology: dict[str, object] | None = None, profile_points: int = 21, geometry: str = "tube",
                            gap_m: float | None = None, width_m: float | None = None,
                            pressure_gradient_Pa_m: float | None = None, positions_m: list[object] | None = None,
                            shear_rates_1_s: list[float] | None = None,
                            compare_rheology: dict[str, object] | None = None,
                            free_stream_velocity_m_s: float | None = None) -> str:
    """Laminar/turbulent flow of non-Newtonian (or Newtonian) liquids in a tube or slit, viscosity curves, and the
    Blasius flat-plate boundary layer. rheology (user's data) {model: power_law (K_Pa_sn, n; true K, not K') | bingham
    (yield_stress_Pa, plastic_viscosity_Pa_s) | herschel_bulkley (yield_stress_Pa, K_Pa_sn, n) | newtonian (viscosity_Pa_s)
    | carreau (eta0_Pa_s, eta_inf_Pa_s, lambda_s, n) | carreau_yasuda (+ a) | cross (eta0_Pa_s, eta_inf_Pa_s, lambda_s, m)}.
    geometry 'tube' (diameter_m): volumetric_flow_m3_s -> ΔP with regime (needs length_m, density_kg_m3; exact laminar,
    Ryan-Johnson/Hanks transition, Dodge-Metzner / Darby turbulent, Herschel-Bulkley turbulent flagged estimate), or
    pressure_gradient_Pa_m -> flow (laminar only). Carreau/Yasuda/Cross: laminar only, numerical (root finding +
    quadrature). geometry 'slit' (parallel plates, gap_m = h; flow in m³/s with width_m, else m²/s per unit width):
    laminar, any rheology, both directions; returns mean/max velocity, wall shear stress and rate, pressure gradient.
    positions_m: r from the axis / y from the mid-plane where velocity, stress and shear rate are evaluated.
    shear_rates_1_s and/or compare_rheology: apparent viscosity of one or two rheologies and the shear rate(s) where
    they cross, plus Carreau/Cross asymptotes; with no flow/gradient given only that 'viscosity' block is returned.
    geometry 'flat_plate' (Newtonian, laminar, Blasius ODE solved numerically): free_stream_velocity_m_s, density_kg_m3,
    rheology newtonian, length_m = plate length, width_m, positions_m = x values or [x, y] pairs -> delta_99,
    displacement and momentum thickness, local and average skin friction, u and v at (x, y), plate drag (one/both
    sides); warns above Re_x 5e5. Always returns evidence, warnings, assumptions, not_included."""
    return json.dumps(TPA.non_newtonian_pipe_flow(volumetric_flow_m3_s, diameter_m, length_m, density_kg_m3, rheology,
                                                  profile_points, geometry, gap_m, width_m, pressure_gradient_Pa_m,
                                                  positions_m, shear_rates_1_s, compare_rheology,
                                                  free_stream_velocity_m_s), default=float, ensure_ascii=False)


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
    column_diameter_m with a sphere also gives the wall effect on the single sphere falling along the tube axis:
    wall_effect {wall_factor, correlation_used} and terminal_velocity_with_wall_m_s (Haberman-Sayre 1958 for Re < 0.1 and
    d/D < 0.8, Francis 1933 as contrast; Di Felice 1996 flagged estimate at higher Re, Munroe above Re 1000);
    terminal_velocity_m_s stays the unbounded value. drag_split: for a sphere with Re < 0.1, exact Stokes split of
    F = 3πμdU into form (pressure) drag 1/3 and friction (viscous) drag 2/3; otherwise it says a flow solution is needed.
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
