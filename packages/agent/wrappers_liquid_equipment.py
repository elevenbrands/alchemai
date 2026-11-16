"""Agent-facing wrappers of the liquid-side equipment tools (`packages.dwsim_client.tools_liquid_equipment`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_liquid_equipment as TLE


@beta_tool
def liquid_liquid_extraction(mode: str, feed: dict[str, float], flow_basis: str, solvent: dict[str, float] | None = None,
                             T_K: float | None = None, P_Pa: float | None = None, model: str | None = None,
                             activity_parameters: list[dict[str, object]] | None = None,
                             distribution_coefficients: dict[str, float] | None = None, carrier: str | None = None,
                             extraction_solvent: str | None = None, n_stages: int | None = None,
                             target: dict[str, object] | None = None, sizing: dict[str, object] | None = None,
                             costing: dict[str, object] | None = None, year: int = 2023, cepci: float | None = None,
                             max_sweeps: int = 400, tolerance: float = 1e-9) -> str:
    """Liquid-liquid split in a decanter (mode 'decanter') or N ideal countercurrent stages ('countercurrent'). feed /
    solvent {component: flow} in flow_basis 'kg_s' | 'mol_s'. Equilibrium, exactly one: model (NRTL/UNIQUAC with
    activity_parameters, or predictive UNIFAC_LLE/UNIFAC) + T_K + P_Pa, or distribution_coefficients {solute: K} with
    carrier and extraction_solvent (immiscible; target {solute, raffinate_ratio} -> Kremser stages). sizing (all user
    data): {orientation, droplet_diameter_m, residence_time_s, length_to_diameter, fill_fraction,
    interface_height_fraction, continuous_phase light|heavy, continuous_viscosity_Pa_s}. costing {equipment:
    gravity_decanter (+material) | centrifuge}. Returns phases, recoveries, sizing, equipment (for capital_investment)
    and evidence; status != 'ok' means no usable separation. UNIFAC is poor for biodiesel-glycerol-methanol (tool docs)."""
    return json.dumps(TLE.liquid_liquid_extraction(mode, feed, flow_basis, solvent, T_K, P_Pa, model, activity_parameters,
                                                   distribution_coefficients, carrier, extraction_solvent, n_stages, target,
                                                   sizing, costing, year, cepci, max_sweeps, tolerance),
                      default=float, ensure_ascii=False)


@beta_tool
def multi_effect_evaporator(feed_kg_s: float, feed_solids_mass_fraction: float, feed_T_K: float,
                            product_solids_mass_fraction: float, n_effects: int, U_W_m2K: list[float],
                            solution_cp_J_kgK: list[float], boiling_point_rise: dict[str, object],
                            steam_T_K: float | None = None, steam_P_Pa: float | None = None,
                            last_effect_P_Pa: float | None = None, last_effect_saturation_T_K: float | None = None,
                            equal_area: bool = True, effect_P_Pa: list[float] | None = None,
                            costing: dict[str, object] | None = None, year: int = 2023, cepci: float | None = None,
                            tolerance: float = 1e-10) -> str:
    """Forward-feed N-effect evaporator of a non-volatile solute: areas (equal-area design, or effect_P_Pa given), effect
    pressures and temperatures, steam use and steam economy. User data, no defaults: U_W_m2K per effect,
    solution_cp_J_kgK [c0, c1, ...] (cp = c0 + c1 x + ...), boiling_point_rise {polynomial_K: [c1, c2, ...]} |
    {per_effect_K: [...]} | {duhring: {solids_mass_fraction, slope, intercept_K}}; steam_T_K or steam_P_Pa; last_effect_P_Pa
    or last_effect_saturation_T_K. costing {type: vertical_tube | horizontal_tube, material_factor?} prices each body from
    the DOE/NETL 1998 table scaled to year/cepci. Returns effects, steam_kg_s, steam_economy, equipment (for
    capital_investment) and evidence. Condenser and vacuum system are not included."""
    return json.dumps(TLE.multi_effect_evaporator(feed_kg_s, feed_solids_mass_fraction, feed_T_K,
                                                  product_solids_mass_fraction, n_effects, U_W_m2K, solution_cp_J_kgK,
                                                  boiling_point_rise, steam_T_K, steam_P_Pa, last_effect_P_Pa,
                                                  last_effect_saturation_T_K, equal_area, effect_P_Pa, costing, year, cepci,
                                                  tolerance), default=float, ensure_ascii=False)


@beta_tool
def storage_tank(flow_kg_s: float, density_kg_m3: float, fill_fraction: float, holdup_time_s: float | None = None,
                 inventory_days: float | None = None, tank_type: str | None = None, material: str | None = None,
                 aspect_ratio_H_D: float | None = None, number_of_tanks: int | None = None, year: int = 2023,
                 cepci: float | None = None) -> str:
    """Storage / surge tank: volume from flow x (holdup_time_s or inventory_days) / fill_fraction, cylinder from
    aspect_ratio_H_D, purchase cost with BioSTEAM's tank correlations. tank_type cone_roof | floating_roof | field_erected
    | mix_tank and material carbon_steel | stainless_steel | copper | nickel | titanium_clad | titanium go together (omit
    both for sizing only). Above the correlation maximum the volume is split into equal tanks; below the minimum the cost
    is a flagged extrapolation. Returns volumes, geometry_per_tank, cost, equipment (for capital_investment), a
    pressure-vessel cross-check and evidence. Floating-roof costs run 6-40 % below the published reference points."""
    return json.dumps(TLE.storage_tank(flow_kg_s, density_kg_m3, fill_fraction, holdup_time_s, inventory_days, tank_type,
                                       material, aspect_ratio_H_D, number_of_tanks, year, cepci),
                      default=float, ensure_ascii=False)


AGENT_WRAPPERS = [liquid_liquid_extraction, multi_effect_evaporator, storage_tank]
