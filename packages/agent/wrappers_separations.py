"""Agent-facing wrappers of the membrane and adsorption tools (`packages.dwsim_client.tools_separations`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_separations as TSE


@beta_tool
def gas_permeation_module(feed_flow_mol_s: float, feed_composition: dict[str, float], feed_pressure_Pa: float,
                          permeate_pressure_Pa: float, flow_pattern: str,
                          permeance_mol_m2_s_Pa: dict[str, float] | None = None,
                          permeability_mol_m_m2_s_Pa: dict[str, float] | None = None, thickness_m: float | None = None,
                          area_m2: float | None = None, stage_cut: float | None = None,
                          target: dict[str, object] | None = None) -> str:
    """One solution-diffusion gas membrane module (multicomponent), flow_pattern complete_mixing | cross_flow | cocurrent |
    countercurrent. User permeances {gas: mol/(m2 s Pa)} or permeabilities + thickness_m (1 GPU = 3.346e-10). Exactly one of
    area_m2 (rating), stage_cut or target {quantity: permeate_mole_fraction | retentate_mole_fraction | recovery_in_permeate
    | recovery_in_retentate, component, value} (design -> area). Returns area, stage cut, permeate/retentate flows and
    compositions, recoveries, pressure-ratio and selectivity limits, profile and evidence; feasible=False if unattainable."""
    return json.dumps(TSE.gas_permeation_module(feed_flow_mol_s, feed_composition, feed_pressure_Pa, permeate_pressure_Pa,
                                                flow_pattern, permeance_mol_m2_s_Pa, permeability_mol_m_m2_s_Pa, thickness_m,
                                                area_m2, stage_cut, target), default=float, ensure_ascii=False)


@beta_tool
def reverse_osmosis(feed_concentration_kg_m3: float, solute_molar_mass_kg_mol: float, ions_per_formula: float, T_K: float,
                    feed_pressure_Pa: float, permeate_pressure_Pa: float, water_permeability_kg_m2_s_Pa: float,
                    solute_permeability_m_s: float, feed_flow_m3_s: float | None = None, area_m2: float | None = None,
                    recovery: float | None = None, osmotic_coefficient: float | None = None,
                    mass_transfer_coefficient_m_s: float | None = None, permeate_density_kg_m3: float | None = None,
                    pump_efficiency: float | None = None, energy_recovery_efficiency: float | None = None) -> str:
    """Reverse osmosis of one solute: N_w = A_w (ΔP − Δπ), N_s = A_s (c_m − c_p) with the user's A_w (kg/(m2 s Pa)) and A_s
    (m/s); osmotic pressure van 't Hoff or with the user's osmotic_coefficient; film-model polarisation if
    mass_transfer_coefficient_m_s. Without feed_flow_m3_s: local fluxes and rejection at the feed. With feed_flow_m3_s and
    area_m2 (rating) or recovery (design): plug-flow module with permeate quality, rejection, area and specific energy
    (pump_efficiency, energy_recovery_efficiency). feasible=False beyond the osmotic limit."""
    return json.dumps(TSE.reverse_osmosis(feed_concentration_kg_m3, solute_molar_mass_kg_mol, ions_per_formula, T_K,
                                          feed_pressure_Pa, permeate_pressure_Pa, water_permeability_kg_m2_s_Pa,
                                          solute_permeability_m_s, feed_flow_m3_s, area_m2, recovery, osmotic_coefficient,
                                          mass_transfer_coefficient_m_s, permeate_density_kg_m3, pump_efficiency,
                                          energy_recovery_efficiency), default=float, ensure_ascii=False)


@beta_tool
def adsorption_fixed_bed(mode: str, bed_length_m: float, breakthrough_fraction: float,
                         superficial_velocity_m_s: float | None = None, bed_void_fraction: float | None = None,
                         bulk_density_kg_m3: float | None = None, components: list[dict[str, object]] | None = None,
                         saturation_fraction: float | None = None, axial_dispersion_m2_s: float | None = None,
                         t_end_s: float | None = None, n_cells: int = 100, lab_curve: dict[str, list[float]] | None = None,
                         new_breakthrough_time_s: float | None = None, new_bed_length_m: float | None = None,
                         solute_feed_rate_kg_s: float | None = None, adsorbent_mass_kg: float | None = None) -> str:
    """Isothermal fixed-bed adsorption. mode 'breakthrough': clean bed, components [{name, feed_concentration_mol_m3,
    k_ldf_1_s, isotherm {model linear K_m3_kg | langmuir q_max_mol_kg b_m3_mol | freundlich K_F n}}] (several Langmuir =
    extended Langmuir), optional axial dispersion -> breakthrough/saturation/stoichiometric times, MTZ length, LUB, bed
    usage, curve. mode 'lub_scaleup': lab_curve {time_s, c_over_c0} of the lab bed (bed_length_m) -> t_b, t_t, t_u, LUB and
    the new bed for new_breakthrough_time_s or new_bed_length_m. Isotherm and k_LDF are the user's. Check evidence.verified."""
    return json.dumps(TSE.adsorption_fixed_bed(mode, bed_length_m, breakthrough_fraction, superficial_velocity_m_s,
                                               bed_void_fraction, bulk_density_kg_m3, components, saturation_fraction,
                                               axial_dispersion_m2_s, t_end_s, n_cells, lab_curve, new_breakthrough_time_s,
                                               new_bed_length_m, solute_feed_rate_kg_s, adsorbent_mass_kg),
                      default=float, ensure_ascii=False)


@beta_tool
def psa_cycle(bed_length_m: float, bed_void_fraction: float, bulk_density_kg_m3: float, T_K: float,
              components: list[dict[str, object]], product_component: str, high_pressure_Pa: float, low_pressure_Pa: float,
              feed_velocity_m_s: float, purge_velocity_m_s: float, pressurization_time_s: float, feed_time_s: float,
              blowdown_time_s: float, purge_time_s: float, pressurization_with: str, bed_diameter_m: float | None = None,
              n_cells: int = 40, max_cycles: int = 200, css_tolerance: float = 1e-4) -> str:
    """Four-step Skarstrom PSA (pressurisation with 'feed' or 'product', feed, countercurrent blowdown, purge with product)
    of a binary gas on one isothermal bed with LDF, run to cyclic steady state. components: two gases {name,
    feed_mole_fraction, k_ldf_1_s, isotherm}; product_component = the weakly adsorbed gas. Returns purity, recovery,
    productivity, per-cycle flows, validation_status, and for linear isotherms the equilibrium-theory comparison. Validated
    only for linear isotherms; trend tool, not industrial design."""
    return json.dumps(TSE.psa_cycle(bed_length_m, bed_void_fraction, bulk_density_kg_m3, T_K, components, product_component,
                                    high_pressure_Pa, low_pressure_Pa, feed_velocity_m_s, purge_velocity_m_s,
                                    pressurization_time_s, feed_time_s, blowdown_time_s, purge_time_s, pressurization_with,
                                    bed_diameter_m, n_cells, max_cycles, css_tolerance), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [gas_permeation_module, reverse_osmosis, adsorption_fixed_bed, psa_cycle]
