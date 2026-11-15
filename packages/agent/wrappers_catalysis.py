"""Agent-facing wrappers of the catalysis block (same names and order as CATALYSIS_TOOLS)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_catalysis as TCA


@beta_tool
def catalyst_effectiveness(geometry: str | None = None, characteristic_length_m: float | None = None,
                           kinetics: dict[str, object] | None = None, concentration_mol_m3: float | None = None,
                           temperature_K: float | None = None, effective_diffusivity_m2_s: float | None = None,
                           pore: dict[str, object] | None = None, heat_of_reaction_J_mol: float | None = None,
                           effective_conductivity_W_mK: float | None = None,
                           mass_transfer_coefficient_m_s: float | None = None,
                           heat_transfer_coefficient_W_m2K: float | None = None,
                           pellet_density_kg_m3: float | None = None, branch: str = "low") -> str:
    """Effectiveness factor of one porous catalyst pellet (slab half-thickness | long cylinder | sphere radius) at bulk
    C and T: diffusion-reaction BVP, kinetics {type power_law (order) | langmuir_hinshelwood (K_m3_mol,
    inhibition_exponent), basis pellet_volume | catalyst_mass, k + reference_temperature_K or pre_exponential,
    activation_energy_J_mol}. De given or estimated from pore data. Optional external film (kc, h) and heat of reaction
    with λe (Weisz-Hicks, all steady states). Returns internal/external/overall η, Thiele and generalized moduli,
    Weisz-Prater/Mears/Anderson, temperature rises, profile; evidence vs analytical/shooting. Nothing is defaulted."""
    return json.dumps(TCA.catalyst_effectiveness(geometry, characteristic_length_m, kinetics, concentration_mol_m3,
                                                 temperature_K, effective_diffusivity_m2_s, pore, heat_of_reaction_J_mol,
                                                 effective_conductivity_W_mK, mass_transfer_coefficient_m_s,
                                                 heat_transfer_coefficient_W_m2K, pellet_density_kg_m3, branch),
                      default=float, ensure_ascii=False)


@beta_tool
def packed_bed_catalytic_reactor(geometry: str | None = None, characteristic_length_m: float | None = None,
                                 kinetics: dict[str, object] | None = None, effective_diffusivity_m2_s: float | None = None,
                                 pore: dict[str, object] | None = None, pellet_density_kg_m3: float | None = None,
                                 heat_of_reaction_J_mol: float | None = None,
                                 effective_conductivity_W_mK: float | None = None,
                                 mass_transfer_coefficient_m_s: float | None = None,
                                 heat_transfer_coefficient_W_m2K: float | None = None, bed_voidage: float | None = None,
                                 tube_diameter_m: float | None = None, number_of_tubes: int = 1,
                                 bed_length_m: float | None = None, target_conversion: float | None = None,
                                 feed_molar_flow_A_mol_s: float | None = None,
                                 feed_total_molar_flow_mol_s: float | None = None,
                                 moles_change_per_mol_A: float | None = None, inlet_temperature_K: float | None = None,
                                 inlet_pressure_Pa: float | None = None, energy_mode: str | None = None,
                                 gas_heat_capacity_J_molK: float | None = None,
                                 wall_heat_transfer_coefficient_W_m2K: float | None = None,
                                 coolant_temperature_K: float | None = None, gas_viscosity_Pa_s: float | None = None,
                                 feed_molar_mass_kg_mol: float | None = None,
                                 ergun_particle_diameter_m: float | None = None, branch: str = "low",
                                 n_profile: int = 11) -> str:
    """Gas-phase 1-D heterogeneous fixed-bed catalytic reactor with local η·r from the pellet BVP. Pellet args as in
    catalyst_effectiveness; bed_voidage, tube_diameter_m, number_of_tubes; bed_length_m OR target_conversion; feed
    flows (A and total), moles_change_per_mol_A, inlet T and P; energy_mode isothermal | adiabatic (cp, ΔH) |
    wall_cooled (U, coolant T); Ergun when gas_viscosity_Pa_s + feed_molar_mass_kg_mol are given. Returns conversion,
    length, volume, catalyst mass, outlet T/P, hot spot, η profile and diffusion-limited zones; evidence = step
    independence, energy closure, pellet checks, closed form."""
    return json.dumps(TCA.packed_bed_catalytic_reactor(
        geometry, characteristic_length_m, kinetics, effective_diffusivity_m2_s, pore, pellet_density_kg_m3,
        heat_of_reaction_J_mol, effective_conductivity_W_mK, mass_transfer_coefficient_m_s,
        heat_transfer_coefficient_W_m2K, bed_voidage, tube_diameter_m, number_of_tubes, bed_length_m, target_conversion,
        feed_molar_flow_A_mol_s, feed_total_molar_flow_mol_s, moles_change_per_mol_A, inlet_temperature_K,
        inlet_pressure_Pa, energy_mode, gas_heat_capacity_J_molK, wall_heat_transfer_coefficient_W_m2K,
        coolant_temperature_K, gas_viscosity_Pa_s, feed_molar_mass_kg_mol, ergun_particle_diameter_m, branch, n_profile),
        default=float, ensure_ascii=False)


@beta_tool
def gas_liquid_reaction(kinetics_type: str | None = None, liquid_mass_transfer_coefficient_m_s: float | None = None,
                        diffusivity_A_m2_s: float | None = None, interface_concentration_A_mol_m3: float | None = None,
                        gas_partial_pressure_Pa: float | None = None, henry_constant_Pa_m3_mol: float | None = None,
                        gas_mass_transfer_coefficient_mol_m2_s_Pa: float | None = None, rate_constant: float | None = None,
                        bulk_concentration_B_mol_m3: float | None = None, diffusivity_B_m2_s: float | None = None,
                        stoichiometric_ratio_B: float | None = None, bulk_concentration_A_mol_m3: float | None = None,
                        contactor: dict[str, object] | None = None) -> str:
    """Gas absorption with liquid-phase reaction by film theory: kinetics_type first_order (k1) | second_order (A + νB,
    k2, C_B, D_B, ν) | instantaneous. Interface C_Ai, or p_A + Henry H (+ optional kG gas film). Returns Hatta number,
    enhancement factor (numerical film solution) with Ha/tanh Ha, E_i = 1 + q and van Krevelen-Hoftijzer for
    comparison, interfacial B, absorption flux, regime. Optional well-mixed contactor (first order: volume, holdup,
    liquid flow, kLa or area) gives bulk C_A and film/bulk split. Evidence = mesh independence, film balances."""
    return json.dumps(TCA.gas_liquid_reaction(kinetics_type, liquid_mass_transfer_coefficient_m_s, diffusivity_A_m2_s,
                                              interface_concentration_A_mol_m3, gas_partial_pressure_Pa,
                                              henry_constant_Pa_m3_mol, gas_mass_transfer_coefficient_mol_m2_s_Pa,
                                              rate_constant, bulk_concentration_B_mol_m3, diffusivity_B_m2_s,
                                              stoichiometric_ratio_B, bulk_concentration_A_mol_m3, contactor),
                      default=float, ensure_ascii=False)


@beta_tool
def three_phase_catalytic_reactor(reactor_type: str | None = None, temperature_K: float | None = None,
                                  henry_constant_Pa_m3_mol: float | None = None, kLa_1_s: float | None = None,
                                  kGa_1_s: float | None = None,
                                  liquid_solid_mass_transfer_coefficient_m_s: float | None = None,
                                  geometry: str | None = None, characteristic_length_m: float | None = None,
                                  kinetics: dict[str, object] | None = None, effective_diffusivity_m2_s: float | None = None,
                                  pore: dict[str, object] | None = None, apparent_rate_constant_m3_kg_s: float | None = None,
                                  pellet_density_kg_m3: float | None = None, stoichiometric_ratio_B: float | None = None,
                                  liquid_reactant_concentration_mol_m3: float | None = None,
                                  gas_partial_pressure_Pa: float | None = None, catalyst_loading_kg_m3: float | None = None,
                                  liquid_volume_m3: float | None = None, operation: str | None = None,
                                  liquid_flow_m3_s: float | None = None, batch_time_s: float | None = None,
                                  target_conversion: float | None = None, limiting_reactant: str | None = None,
                                  wetting_efficiency: float | None = None, bed_voidage: float | None = None,
                                  column_diameter_m: float | None = None, bed_length_m: float | None = None,
                                  feed_molar_flow_A_mol_s: float | None = None,
                                  feed_total_gas_molar_flow_mol_s: float | None = None,
                                  gas_moles_change_per_mol_A: float | None = None, inlet_pressure_Pa: float | None = None,
                                  pressure_gradient_Pa_m: float | None = None, n_profile: int = 11) -> str:
    """Isothermal three-phase catalytic reactor (gas A + liquid B + solid catalyst) by resistances in series: gas film
    (kGa), gas-liquid (kLa), liquid-solid (kc), pellet η·k (pellet BVP, or user apparent_rate_constant). reactor_type
    slurry (operation batch | continuous; loading → conversion or target → loading/time) or trickle_bed
    (limiting_reactant gas | liquid, wetting_efficiency, plug flow of gas and liquid, pressure gradient). Returns each
    resistance and its share, controlling step, overall rate, conversions, catalyst mass, bed height, profile; evidence
    = series closure, Fogler closed forms, step independence, pellet check. Coefficients are user inputs."""
    return json.dumps(TCA.three_phase_catalytic_reactor(
        reactor_type, temperature_K, henry_constant_Pa_m3_mol, kLa_1_s, kGa_1_s,
        liquid_solid_mass_transfer_coefficient_m_s, geometry, characteristic_length_m, kinetics,
        effective_diffusivity_m2_s, pore, apparent_rate_constant_m3_kg_s, pellet_density_kg_m3,
        stoichiometric_ratio_B, liquid_reactant_concentration_mol_m3, gas_partial_pressure_Pa,
        catalyst_loading_kg_m3, liquid_volume_m3, operation, liquid_flow_m3_s, batch_time_s, target_conversion,
        limiting_reactant, wetting_efficiency, bed_voidage, column_diameter_m, bed_length_m, feed_molar_flow_A_mol_s,
        feed_total_gas_molar_flow_mol_s, gas_moles_change_per_mol_A, inlet_pressure_Pa, pressure_gradient_Pa_m,
        n_profile), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [catalyst_effectiveness, packed_bed_catalytic_reactor, gas_liquid_reaction,
                  three_phase_catalytic_reactor]
