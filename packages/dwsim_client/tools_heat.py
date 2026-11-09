"""Agent tools for shell-side heat transfer, boiling and condensation (bloque M4b). No flowsheet needed: pass
properties from stream_transport_properties or the user. The engine and its evidence live in
`packages.transport.heat_devices`."""
from __future__ import annotations

from packages.transport import heat_devices as HD


def shell_side_kern(mass_flow_kg_s: float, density_kg_m3: float, viscosity_Pa_s: float, thermal_conductivity_W_mK: float,
                    cp_J_kgK: float, shell_diameter_m: float, baffle_spacing_m: float, n_baffles: int, tube_od_m: float,
                    tube_pitch_m: float, layout: str = "triangular", wall_viscosity_Pa_s: float | None = None) -> dict:
    """Shell-side film coefficient and pressure drop of a shell-and-tube exchanger by Kern's method (1950):
    h = 0.36·(k/De)·Re^0.55·Pr^⅓·(μ/μw)^0.14 with the equivalent diameter of the tube layout ('triangular' or
    'square'), crossflow area Ds·(Pt−Do)·B/Pt, ΔP from Kern's friction chart over n_baffles + 1 crossings. Also
    returns h of an ideal tube bank (Žukauskas) at the same velocity as the no-leakage ceiling. Kern ignores baffle
    leakage and bundle bypass: preliminary sizing only (final rating: shell_side_bell_delaware). evidence.verified = Kern's
    range (2·10³ < Re < 10⁶), TEMA pitch ratio ≥ 1.25 and baffle spacing in [max(Ds/5, 2 in), Ds]. Combine h with
    tube_film_coefficient into U before heat_exchanger_area."""
    return HD.shell_kern(mass_flow_kg_s, density_kg_m3, viscosity_Pa_s, thermal_conductivity_W_mK, cp_J_kgK,
                         shell_diameter_m, baffle_spacing_m, n_baffles, tube_od_m, tube_pitch_m, layout, wall_viscosity_Pa_s)


def shell_side_bell_delaware(mass_flow_kg_s: float, density_kg_m3: float, viscosity_Pa_s: float,
                             thermal_conductivity_W_mK: float, cp_J_kgK: float, shell_diameter_m: float,
                             tube_od_m: float, tube_pitch_m: float, layout_angle: int, baffle_cut_fraction: float,
                             baffle_spacing_m: float, n_baffles: int, outer_tube_limit_diameter_m: float,
                             n_tubes: int | None = None, tube_passes: int = 1,
                             shell_baffle_clearance_m: float | None = None,
                             tube_baffle_clearance_m: float | None = None, sealing_strip_pairs: int = 0,
                             inlet_baffle_spacing_m: float | None = None, outlet_baffle_spacing_m: float | None = None,
                             wall_viscosity_Pa_s: float | None = None) -> dict:
    """Shell-side film coefficient and pressure drop of a segmentally baffled shell-and-tube exchanger by the
    Bell-Delaware method (Bell 1963; Taborek, HEDH): the design-grade method, which corrects an ideal tube bank for
    baffle cut (Jc), shell-baffle and tube-baffle leakage (Jl, Rl), bundle bypass (Jb, Rb), unequal end spacing
    (Js, Rs) and laminar gradient (Jr). Geometry is the user's exchanger: shell ID, tube OD, pitch, layout_angle
    30 (triangular) | 45 (rotated square) | 90 (square), baffle_cut_fraction (cut height / Ds), central baffle
    spacing, n_baffles, outer_tube_limit_diameter_m (Dotl; Ds − Dotl is the bundle-shell gap: ~10-15 mm fixed
    tubesheet, ~90 mm pull-through floating head). Optional: n_tubes (else estimated), diametral clearances
    (else TEMA RCB-4.2/4.3 values, listed in assumptions), sealing_strip_pairs, inlet/outlet spacing. Returns h, ΔP
    split into crossflow/windows/end zones (nozzles excluded), every J and R factor, the geometry areas, and Kern on
    the same shell for comparison. evidence: each J factor read from the digitized Delaware charts must agree with
    the HEDH closed-form fit within 10 %, the ideal bank by Žukauskas and ESDU 73031 within 30 %, and cut, pitch
    ratio, spacing, Re and leakage/bypass ratios inside the charts. Combine h with tube_film_coefficient into U."""
    return HD.shell_bell_delaware(mass_flow_kg_s, density_kg_m3, viscosity_Pa_s, thermal_conductivity_W_mK, cp_J_kgK,
                                  shell_diameter_m, tube_od_m, tube_pitch_m, layout_angle, baffle_cut_fraction,
                                  baffle_spacing_m, n_baffles, outer_tube_limit_diameter_m, n_tubes, tube_passes,
                                  shell_baffle_clearance_m, tube_baffle_clearance_m, sealing_strip_pairs,
                                  inlet_baffle_spacing_m, outlet_baffle_spacing_m, wall_viscosity_Pa_s)


def nucleate_boiling(P_Pa: float, critical_pressure_Pa: float, MW: float, T_sat_K: float, liquid_density_kg_m3: float,
                     vapor_density_kg_m3: float, liquid_viscosity_Pa_s: float, liquid_conductivity_W_mK: float,
                     liquid_cp_J_kgK: float, latent_heat_J_kg: float, surface_tension_N_m: float,
                     heat_flux_W_m2: float | None = None, wall_superheat_K: float | None = None,
                     surface_roughness_um: float = 1.0, rohsenow_Csf: float | None = None,
                     rohsenow_n: float | None = None) -> dict:
    """Nucleate pool boiling (kettle reboiler, vaporizer, boiling side of a jacket): give heat_flux_W_m2 OR
    wall_superheat_K. Returns h by Cooper (recommended: needs only P/Pc, MW and roughness), Mostinski,
    Stephan-Abdelsalam and, if the surface-fluid constants rohsenow_Csf and rohsenow_n are given, Rohsenow; the
    spread between them; the critical heat flux (Zuber K = 0.131 and Lienhard-Dhir K = 0.149) and the fraction of it
    in use. evidence: Cooper (and Rohsenow) recomputed from the printed equations, reduced pressure inside
    0.001-0.9 and q below the critical flux (above it the regime is film boiling and nucleate h does not apply)."""
    return HD.pool_boiling(P_Pa, critical_pressure_Pa, MW, T_sat_K, liquid_density_kg_m3, vapor_density_kg_m3,
                           liquid_viscosity_Pa_s, liquid_conductivity_W_mK, liquid_cp_J_kgK, latent_heat_J_kg,
                           surface_tension_N_m, heat_flux_W_m2, wall_superheat_K, surface_roughness_um, rohsenow_Csf,
                           rohsenow_n)


def film_condensation(geometry: str, T_sat_K: float, liquid_density_kg_m3: float, vapor_density_kg_m3: float,
                      liquid_conductivity_W_mK: float, liquid_viscosity_Pa_s: float, liquid_cp_J_kgK: float,
                      latent_heat_J_kg: float, wall_T_K: float | None = None, length_m: float | None = None,
                      tube_od_m: float | None = None, n_rows: int = 1, mass_flow_kg_s: float | None = None,
                      quality: float | None = None, tube_id_m: float | None = None, P_Pa: float | None = None,
                      critical_pressure_Pa: float | None = None) -> dict:
    """Condensation of a pure saturated vapor. geometry 'vertical_plate' / 'vertical_tube' (needs wall_T_K and
    length_m = height) or 'horizontal_tube' (wall_T_K, tube_od_m, n_rows in a vertical column: h_N = h_1·N^-¼):
    Nusselt laminar film with Rohsenow's modified latent heat h'fg = hfg·(1 + 0.68·Ja); returns h, heat flux, film
    Reynolds and regime. geometry 'in_tube' (mass_flow_kg_s per tube, quality, tube_id_m, P_Pa,
    critical_pressure_Pa): local h by Shah (recommended) and Boyko-Kruzhilin. evidence: the printed Nusselt or Shah
    equation recomputed; unverified if the film is wavy/turbulent (Nusselt underpredicts), Ja > 0.1 or Re_LO < 350.
    Non-condensable gases are not modelled."""
    return HD.condensation(geometry, T_sat_K, liquid_density_kg_m3, vapor_density_kg_m3, liquid_conductivity_W_mK,
                           liquid_viscosity_Pa_s, liquid_cp_J_kgK, latent_heat_J_kg, wall_T_K, length_m, tube_od_m,
                           n_rows, mass_flow_kg_s, quality, tube_id_m, P_Pa, critical_pressure_Pa)


HEAT_TOOLS = [shell_side_kern, shell_side_bell_delaware, nucleate_boiling, film_condensation]
