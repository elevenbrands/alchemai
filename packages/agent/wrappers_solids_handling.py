"""Agent-facing wrappers of the solids handling and oil extraction tools (`packages.dwsim_client.tools_solids_handling`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_solids_handling as TSH


@beta_tool
def size_reduction(mass_flow_kg_s: float, law: str, feed_size_m: float | None = None, product_size_m: float | None = None,
                   work_index_kWh_per_ton: float | None = None, ton: str | None = None,
                   kick_constant_J_kg: float | None = None, rittinger_constant_J_m_kg: float | None = None,
                   reference_test: dict[str, float] | None = None, specific_energy_J_kg: float | None = None,
                   equipment: str | None = None, year: int = 2023, cepci: float | None = None) -> str:
    """Crushing/grinding energy and power. law 'bond' (feed/product sizes = 80 % passing; work_index_kWh_per_ton and
    ton 'short'|'metric' are the USER's), 'kick' / 'rittinger' (constant, or reference_test {feed_size_m, product_size_m,
    specific_energy_J_kg} to calibrate it) or 'specific_energy' (measured J/kg; the only route for oilseeds/biomass).
    equipment 'hammer_mill' | 'shredder' | 'cane_crushing_mill' adds BioSTEAM's purchase cost ('crusher': power only).
    Never invent a work index: published averages come back in published_reference and must be passed explicitly and
    flagged. Returns specific_energy_J_kg, power_W (shaft), cost, capital_items (for capital_investment), evidence,
    warnings (correlation range), assumptions, not_included."""
    return json.dumps(TSH.size_reduction(mass_flow_kg_s, law, feed_size_m, product_size_m, work_index_kWh_per_ton, ton,
                                         kick_constant_J_kg, rittinger_constant_J_m_kg, reference_test, specific_energy_J_kg,
                                         equipment, year, cepci), default=float, ensure_ascii=False)


@beta_tool
def solids_conveying(conveyor: str, mass_flow_kg_s: float, length_m: float | None = None, lift_m: float | None = None,
                     bulk_density_kg_m3: float | None = None, friction_factor: float | None = None,
                     tail_pulley_diameter_m: float | None = None, length_equivalency_factor: float | None = None,
                     year: int = 2023, cepci: float | None = None) -> str:
    """Shaft power and purchase cost of a bulk-solids conveyor: conveyor 'belt' | 'screw' (need length_m) |
    'bucket_elevator'. lift_m is mandatory (0 if horizontal). bulk_density_kg_m3 is only needed for the cost (BioSTEAM
    correlations by volumetric flow, independent of length: order of magnitude). Bucket elevator: friction_factor (>= 1)
    or tail_pulley_diameter_m + length_equivalency_factor, equipment data with no default (published ranges are in
    published_reference); it has no open cost correlation. Returns power_W, friction_power_W, lift_power_W, cost,
    capital_items, evidence, warnings, assumptions, not_included."""
    return json.dumps(TSH.solids_conveying(conveyor, mass_flow_kg_s, length_m, lift_m, bulk_density_kg_m3, friction_factor,
                                           tail_pulley_diameter_m, length_equivalency_factor, year, cepci),
                      default=float, ensure_ascii=False)


@beta_tool
def solid_liquid_separation(equipment: str, feed_kg_s: float, feed_solids_mass_fraction: float,
                            solids_recovery_fraction: float, cake_moisture_mass_fraction: float,
                            filtration: dict[str, float] | None = None, sigma: dict[str, object] | None = None,
                            feed_density_kg_m3: float | None = None, year: int = 2023, cepci: float | None = None) -> str:
    """Solid-liquid separation: equipment 'screw_press' | 'decanter_centrifuge' | 'disc_stack_centrifuge' |
    'rotary_vacuum_drum_filter' | 'plate_and_frame_filter'. The split comes from the USER's solids_recovery_fraction and
    cake_moisture_mass_fraction (test or vendor data, never assumed). filtration {specific_cake_resistance_m_kg,
    medium_resistance_1_m, filtrate_viscosity_Pa_s, pressure_drop_Pa, filtrate_density_kg_m3, + cycle_time_s and
    submerged_fraction (drum) or filtration_time_s and downtime_s (press)} gives the filter area; sigma
    {particle_diameter_m, particle_density_kg_m3, liquid_density_kg_m3, liquid_viscosity_Pa_s, bowl {type, rpm, r_inner_m,
    r_outer_m, length_m | n_discs + half_angle_deg}} gives required and machine Sigma. Returns cake, liquid, filtration,
    sigma, cost, capital_items (BioSTEAM correlations with their ranges), evidence, warnings, not_included."""
    return json.dumps(TSH.solid_liquid_separation(equipment, feed_kg_s, feed_solids_mass_fraction, solids_recovery_fraction,
                                                  cake_moisture_mass_fraction, filtration, sigma, feed_density_kg_m3, year,
                                                  cepci), default=float, ensure_ascii=False)


@beta_tool
def oil_extraction(mode: str, seed: dict[str, object] | None = None, expeller: dict[str, object] | None = None,
                   solids: dict[str, float] | None = None, solvent: dict[str, float] | None = None,
                   underflow: dict[str, object] | None = None, target: dict[str, float] | None = None,
                   energy: dict[str, object] | None = None, solvent_loss_kg_per_kg_solids_fed: float | None = None,
                   year: int = 2023, cepci: float | None = None) -> str:
    """Oilseed oil extraction. mode 'expeller': seed {mass_flow_kg_s, oil_mass_fraction, decortication {hull_mass_fraction,
    hull_removal_fraction, oil_mass_fraction_in_hull}} and expeller {oil_recovery_fraction | residual_oil_in_cake_mass_fraction,
    specific_energy_J_kg, estimate_cost} -> crude oil, cake, hull. mode 'solvent' (ideal countercurrent leaching): solids
    {inert_solids_kg_s, oil_kg_s, solvent_kg_s}, solvent {solvent_kg_s, oil_kg_s}, underflow
    {solution_retained_kg_per_kg_inert | curve {oil_mass_fraction_in_solution, kg_inert_per_kg_solution}}, target
    {residual_oil_kg_s | residual_oil_in_meal_mass_fraction | stages}, energy {stream_T_K, stripping_pressure_Pa} ->
    stages, miscella, leached solids, stripping heat. Oil content, recovery, residual oil and retention are USER data;
    published typical values (castor, rapeseed, jatropha, soybean) come back in published_reference to be passed
    explicitly and flagged. No cost for extractors (vendor quote)."""
    return json.dumps(TSH.oil_extraction(mode, seed, expeller, solids, solvent, underflow, target, energy,
                                         solvent_loss_kg_per_kg_solids_fed, year, cepci), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [size_reduction, solids_conveying, solid_liquid_separation, oil_extraction]
