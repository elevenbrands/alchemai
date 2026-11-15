"""Agent-facing wrappers of the solid-diffusion block (same names and order as SOLID_DIFFUSION_TOOLS)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_solid_diffusion as TSD


@beta_tool
def transient_diffusion_solid(geometry: str | None = None, initial_concentration_mol_m3: float | None = None,
                              diffusivity_m2_s: float | None = None, diffusivity_law: dict[str, object] | None = None,
                              surface_condition: str | None = None, surface_concentration_mol_m3: float | None = None,
                              fluid_concentration_mol_m3: float | None = None,
                              mass_transfer_coefficient_m_s: float | None = None, partition_coefficient: float | None = None,
                              thickness_m: float | None = None, exposed_faces: int | None = None, radius_m: float | None = None,
                              length_m: float | None = None, width_m: float | None = None, depth_m: float | None = None,
                              bath_volume_m3: float | None = None, solid_volume_m3: float | None = None,
                              times_s: list[float] | None = None, target_fractional_uptake: float | None = None,
                              n_cells: int = 200, profile_points: int = 21) -> str:
    """Unsteady Fickian diffusion WITHOUT reaction in a solid (uptake, release, drying-type desorption, leaching).
    geometry slab (thickness_m, exposed_faces 1|2) | cylinder | sphere (radius_m) | semi_infinite | finite_cylinder
    (radius_m, length_m) | rectangular_bar | block. D constant (diffusivity_m2_s) or diffusivity_law {type exponential |
    linear | table}. surface_condition fixed_concentration | convective (fluid conc, k_c, partition_coefficient =
    c_solid/c_fluid) | limited_bath (bath_volume_m3, solid_volume_m3). Give times_s and/or target_fractional_uptake.
    Returns M_t/M_inf, profiles, surface flux, Fourier, Biot, regime, time to target; evidence = mass balance, mesh and
    step independence, match with Crank's series. Every physical input is the user's: nothing is defaulted."""
    return json.dumps(TSD.transient_diffusion_solid(geometry, initial_concentration_mol_m3, diffusivity_m2_s, diffusivity_law,
                                                    surface_condition, surface_concentration_mol_m3,
                                                    fluid_concentration_mol_m3, mass_transfer_coefficient_m_s,
                                                    partition_coefficient, thickness_m, exposed_faces, radius_m, length_m,
                                                    width_m, depth_m, bath_volume_m3, solid_volume_m3, times_s,
                                                    target_fractional_uptake, n_cells, profile_points),
                      default=float, ensure_ascii=False)


@beta_tool
def membrane_permeation(layers: list[dict[str, object]] | None = None, geometry: str | None = None,
                        upstream_partial_pressure_Pa: float | None = None, downstream_partial_pressure_Pa: float | None = None,
                        upstream_concentration_mol_m3: float | None = None, downstream_concentration_mol_m3: float | None = None,
                        area_m2: float | None = None, inner_radius_m: float | None = None, length_m: float | None = None,
                        upstream_side: str | None = None, upstream_film_coefficient: float | None = None,
                        downstream_film_coefficient: float | None = None, transient: bool = False,
                        initial_partial_pressure_Pa: float | None = None, initial_concentration_mol_m3: float | None = None,
                        times_s: list[float] | None = None, cells_per_layer: int = 120) -> str:
    """Single-component solution-diffusion permeation through a dense film or a laminate in series (packaging, liners,
    hoses, tube walls); not a membrane separation module. layers upstream -> downstream: {thickness_m + two of
    diffusivity_m2_s, solubility_mol_m3_Pa (or solubility_m3STP_m3_atm / partition_coefficient), permeability
    (permeability_mol_m_m2_s_Pa | permeability_barrer | permeability_m3STP_m_m2_s_atm | permeability_m2_s)}. geometry
    flat | cylindrical | spherical (inner_radius_m, upstream_side inner|outer). Driving force: partial pressures OR fluid
    concentrations. transient=true (needs D per layer and the initial equilibrium value) adds the time lag (L²/6D for
    one flat layer) and the downstream cumulative curve. Returns flux/rate, layer resistances, interface values."""
    return json.dumps(TSD.membrane_permeation(layers, geometry, upstream_partial_pressure_Pa, downstream_partial_pressure_Pa,
                                              upstream_concentration_mol_m3, downstream_concentration_mol_m3, area_m2,
                                              inner_radius_m, length_m, upstream_side, upstream_film_coefficient,
                                              downstream_film_coefficient, transient, initial_partial_pressure_Pa,
                                              initial_concentration_mol_m3, times_s, cells_per_layer),
                      default=float, ensure_ascii=False)


@beta_tool
def diffusivity_from_uptake(geometry: str | None = None, times_s: list[float] | None = None, uptake: list[float] | None = None,
                            uptake_kind: str | None = None, thickness_m: float | None = None, exposed_faces: int | None = None,
                            radius_m: float | None = None, equilibrium_uptake: float | None = None,
                            fit_equilibrium_uptake: bool = False, mass_transfer_coefficient_m_s: float | None = None,
                            partition_coefficient: float | None = None, confidence_level: float = 0.95) -> str:
    """Fit a constant diffusivity to the user's sorption/desorption data (uptake vs time) of a slab (thickness_m,
    exposed_faces), cylinder or sphere (radius_m) by least squares on Crank's solution. uptake_kind 'fractional'
    (M_t/M_inf) or 'amount' (then equilibrium_uptake, or fit_equilibrium_uptake=true). Optional external film
    (mass_transfer_coefficient_m_s, partition_coefficient). Returns D with its confidence interval, RMSE, R², residual
    diagnostics (Durbin-Watson, autocorrelation, sign runs), half-time and initial-slope cross-checks, and warnings for
    non-Fickian residuals or data that do not determine D."""
    return json.dumps(TSD.diffusivity_from_uptake(geometry, times_s, uptake, uptake_kind, thickness_m, exposed_faces, radius_m,
                                                  equilibrium_uptake, fit_equilibrium_uptake, mass_transfer_coefficient_m_s,
                                                  partition_coefficient, confidence_level),
                      default=float, ensure_ascii=False)


AGENT_WRAPPERS = [transient_diffusion_solid, membrane_permeation, diffusivity_from_uptake]
