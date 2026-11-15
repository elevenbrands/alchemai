"""Agent-facing wrappers of the crystallization and drying tools (`packages.dwsim_client.tools_solids`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_solids as TSO


@beta_tool
def crystallizer_yield(feed_mass_kg: float, final_T_K: float, solubility: dict[str, object], evaporation: dict[str, object],
                       feed_solute_mass_fraction: float | None = None, feed_solute_per_kg_solvent: float | None = None,
                       crystal_solute_mass_fraction: float | None = None, hydrate: dict[str, object] | None = None,
                       solvent: str | None = None, energy: dict[str, object] | None = None) -> str:
    """Yield and heat balance of a cooling / evaporative / adiabatic (vacuum) crystallizer; mother liquor saturated at
    final_T_K. solubility {T_K: [...], kg_solute_per_kg_solvent: [...]} is the user's (anhydrous basis, never
    extrapolated). evaporation {mode: none | mass (+solvent_evaporated_kg) | fraction_of_feed (+fraction) |
    fraction_of_feed_solvent (+fraction) | adiabatic}. Crystal: crystal_solute_mass_fraction (1.0 anhydrous) or hydrate
    {anhydrous_molar_mass_kg_mol, solvent_moles_per_mole}; solvent='water' reads water data from IAPWS. energy {feed_T_K,
    solution_cp_J_kgK, heat_of_crystallization_J_kg or heat_of_solution_J_mol}. Returns crystals_kg, mother_liquor_kg,
    yield, heat.heat_to_remove_J and evidence (mass and enthalpy closure)."""
    return json.dumps(TSO.crystallizer_yield(feed_mass_kg, final_T_K, solubility, evaporation, feed_solute_mass_fraction,
                                             feed_solute_per_kg_solvent, crystal_solute_mass_fraction, hydrate, solvent,
                                             energy), default=float, ensure_ascii=False)


@beta_tool
def crystal_size_distribution(mode: str, crystal_density_kg_m3: float, volume_shape_factor: float,
                              residence_time_s: float | None = None, growth_rate_m_s: float | None = None,
                              nucleation_rate_m3_s: float | None = None, magma_density_kg_m3: float | None = None,
                              nucleation_law: dict[str, float] | None = None, suspension_volume_m3: float | None = None,
                              sizes_m: list[float] | None = None, sieve: dict[str, object] | None = None,
                              batch: dict[str, object] | None = None) -> str:
    """Crystal population balance. mode 'msmpr': residence_time_s plus two of growth_rate_m_s, nucleation_rate_m3_s,
    magma_density_kg_m3 (or magma density + nucleation_law {k, growth_exponent, magma_exponent}) -> n0, moments, dominant
    size 3Gτ, mass median, CV, production. mode 'msmpr_from_sieve': residence_time_s, magma_density_kg_m3 and sieve
    {lower_m, upper_m, mass_fraction} -> G and B0 by ln n vs L regression with R² and the bias of wide cuts. mode
    'batch_seeded': batch {solubility, initial_concentration_kg_kg, cooling {time_s, T_K}, seed_mass_kg_per_kg_solvent,
    seed_size_m, growth {k_m_s, order}, nucleation {k_per_kg_s, order, magma_exponent} | null} by the method of moments.
    Kinetics, crystal density and shape factor are the user's data. Report evidence.verified and warnings."""
    return json.dumps(TSO.crystal_size_distribution(mode, crystal_density_kg_m3, volume_shape_factor, residence_time_s,
                                                    growth_rate_m_s, nucleation_rate_m3_s, magma_density_kg_m3,
                                                    nucleation_law, suspension_volume_m3, sizes_m, sieve, batch),
                      default=float, ensure_ascii=False)


@beta_tool
def dryer_batch(air: dict[str, float], solid: dict[str, float] | None = None,
                constant_rate: dict[str, object] | None = None, falling_rate: dict[str, object] | None = None) -> str:
    """Batch drying time and psychrometrics (CoolProp HumidAirProp). air {T_K, pressure_Pa and one of humidity_ratio_kg_kg,
    relative_humidity, wet_bulb_T_K, dew_point_T_K, water_partial_pressure_Pa}; alone it returns the air state. solid
    {initial_moisture, final_moisture, critical_moisture, equilibrium_moisture (dry basis, total), dry_solid_per_area_kg_m2
    or dry_solid_mass_kg + drying_area_m2, or bed_depth_m + bulk_density_kg_m3}. constant_rate {rate_kg_m2s} measured, or
    {configuration: parallel | perpendicular | through_circulation, velocity_m_s, h_W_m2K, particle}. falling_rate {model:
    linear | diffusion (+diffusivity_m2_s, thickness_m, drying_faces) | curve (+moisture, rate_kg_m2s)}. Returns the
    drying times, the constant rate and evidence; critical and equilibrium moisture are never assumed."""
    return json.dumps(TSO.dryer_batch(air, solid, constant_rate, falling_rate), default=float, ensure_ascii=False)


@beta_tool
def dryer_continuous(air_in: dict[str, float], solid: dict[str, float], flow: str, heat_loss_W: float,
                     air_outlet_T_K: float | None = None, dry_air_flow_kg_s: float | None = None,
                     transfer_units: float | None = None, sizing: dict[str, object] | None = None,
                     air_heater_inlet_T_K: float | None = None) -> str:
    """Steady continuous dryer (rotary, tunnel, belt; not spray or flash): air-solid water and enthalpy balances. air_in
    as in dryer_batch; solid {dry_solid_flow_kg_s, inlet_moisture, outlet_moisture, inlet_T_K, outlet_T_K,
    dry_solid_cp_J_kgK}; flow countercurrent | cocurrent; heat_loss_W (0 if adiabatic). Exactly one of air_outlet_T_K,
    dry_air_flow_kg_s, transfer_units. sizing {diameter_m | cross_section_area_m2 | gas_mass_velocity_kg_m2s, and
    volumetric_heat_transfer_coefficient_W_m3K or correlation 'mccabe_rotary'} gives the length. Returns the air needed,
    outlet air state, heat transferred, transfer units, length and evidence (balance closure, temperature cross)."""
    return json.dumps(TSO.dryer_continuous(air_in, solid, flow, heat_loss_W, air_outlet_T_K, dry_air_flow_kg_s, transfer_units,
                                           sizing, air_heater_inlet_T_K), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [crystallizer_yield, crystal_size_distribution, dryer_batch, dryer_continuous]
