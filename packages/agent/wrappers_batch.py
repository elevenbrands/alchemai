"""Agent-facing wrappers of the batch-operation tools (`packages.dwsim_client.tools_batch`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_batch as TBA


@beta_tool
def batch_fermentation(kinetics: dict[str, object], initial: dict[str, float], t_end_s: float,
                       feed: dict[str, float] | None = None, target: dict[str, float] | None = None,
                       stoichiometry: dict[str, object] | None = None, heat: dict[str, object] | None = None,
                       agitation_W_m3: float | None = None, schedule: dict[str, object] | None = None, year: int = 2023,
                       cepci: float | None = None, n_points: int = 61) -> str:
    """Batch / fed-batch fermentation (SI; kg/m3 = g/L, rates per second). kinetics {mu_max_1_s, Ks_kg_m3, Yxs_kg_kg,
    optional maintenance_kg_kg_s, kd_1_s, alpha_kg_kg, beta_kg_kg_s, Yps_kg_kg, inhibition [{model: andrews|levenspiel|
    aiba|hyperbolic, ...}]} are the USER's (never assume; published literature values may be proposed with their source).
    initial {volume_m3, biomass_kg_m3, substrate_kg_m3, product_kg_m3}; feed {flow_m3_s, substrate_kg_m3, start_s, stop_s,
    exponential_rate_1_s, max_volume_m3}; target {substrate_conversion | product_kg_m3 | biomass_kg_m3 | substrate_kg_m3};
    stoichiometry {substrate_formula, biomass_formula, product_formula, co2}; heat {basis: substrate|combustion, ...};
    schedule {broth_rate_m3_s | product_rate_kg_s, fill/empty/clean_time_s, working_volume_fraction, n_vessels |
    max_vessel_volume_m3} -> vessels and BioSTEAM/NREL purchase cost (year/cepci). Returns profile, summary, co2, heat,
    cooling_duty, schedule, evidence."""
    return json.dumps(TBA.batch_fermentation(kinetics, initial, t_end_s, feed, target, stoichiometry, heat, agitation_W_m3,
                                             schedule, year, cepci, n_points), default=float, ensure_ascii=False)


@beta_tool
def batch_distillation(components: list[str], charge_mol: float, charge_mole_fractions: list[float],
                       equilibrium: dict[str, object], stop: dict[str, object], mode: str = "rayleigh",
                       n_stages: int | None = None, reflux_ratio: float | None = None,
                       distillate_mole_fraction: float | None = None, boilup_rate_mol_s: float | None = None,
                       heat_of_vaporization_J_mol: float | None = None, n_points: int = 41) -> str:
    """Batch distillation (molar, constant molar overflow, no hold-up). mode 'rayleigh' (simple, any number of
    components), 'constant_reflux' (n_stages theoretical stages INCLUDING the still, reflux_ratio L/D) or
    'constant_distillate' (binary, distillate_mole_fraction of component 1 held by raising reflux). equilibrium
    {type: constant_alpha, alpha} | {type: table, x, y} (binary) | {type: thermo, model, P_Pa} (independent_flash K-values).
    stop: one of {residue_mole_fraction, component}, {distilled_fraction}, {average_distillate_mole_fraction, component},
    {time_s}, {max_reflux_ratio}. boilup_rate_mol_s gives batch time; heat_of_vaporization_J_mol the reboiler heat.
    Returns final still/distillate amounts and compositions, profile, heat and evidence (balance closure, analytical
    Rayleigh / Fenske checks)."""
    return json.dumps(TBA.batch_distillation(components, charge_mol, charge_mole_fractions, equilibrium, stop, mode, n_stages,
                                             reflux_ratio, distillate_mole_fraction, boilup_rate_mol_s,
                                             heat_of_vaporization_J_mol, n_points), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [batch_fermentation, batch_distillation]
