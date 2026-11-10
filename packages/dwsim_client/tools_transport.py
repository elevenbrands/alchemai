"""Agent tools for transport phenomena on solved streams: pipe hydraulics, pipe sizing, heat-exchanger area,
tube-side film coefficient. Fluid properties come from the DWSIM stream (liquid phase if vapor fraction < 0.5,
else vapor); the correlations live in `packages.transport.hydraulics`."""
from __future__ import annotations

from packages.dwsim_client.tools import _s
from packages.transport import hydraulics as H
from packages.transport import reactor_field as RF
from packages.transport import vessel_cfd as VC
from packages.transport import stirred_tank as ST


def _phase_props(flowsheet_id: str, stream: str) -> dict:
    b = _s(flowsheet_id)["streams"][stream]
    ms = b.Object
    ms = getattr(ms, "__implementation__", ms)
    vf = float(b.VaporFraction or 0.0)
    if 0.02 < vf < 0.98:
        raise ValueError(f"{stream} is two-phase (vapor fraction {vf:.3f}): single-phase correlations do not apply; "
                         "use a pipe unit (add_unit kind='pipe', Beggs-Brill) or split the phases first")
    pr = ms.Phases[2 if vf >= 0.5 else 1].Properties
    get = lambda n: pr.GetType().GetProperty(n).GetValue(pr)
    rho, mu, k, cp = get("density"), get("viscosity"), get("thermalConductivity"), get("heatCapacityCp")
    if rho is None or mu is None:
        raise ValueError(f"{stream}: transport properties not available (solve the flowsheet first)")
    return {"phase": "vapor" if vf >= 0.5 else "liquid", "density_kg_m3": float(rho), "viscosity_Pa_s": float(mu),
            "thermal_conductivity_W_mK": float(k) if k is not None else None,
            "cp_J_kgK": float(cp) * 1000.0 if cp is not None else None,
            "mass_flow_kg_s": float(b.MassFlowKgPerSecond), "P_Pa": float(b.PressurePa), "T_K": float(b.TemperatureK)}


def stream_transport_properties(flowsheet_id: str, stream: str) -> dict:
    """Density, viscosity, thermal conductivity and Cp of a solved single-phase stream (DWSIM property package)."""
    return _phase_props(flowsheet_id, stream)


def pipe_hydraulics(flowsheet_id: str, stream: str, length_m: float, inner_diameter_m: float,
                    roughness_m: float = H.STEEL_ROUGHNESS_M, elevation_change_m: float = 0.0, fittings_K: float = 0.0) -> dict:
    """Pressure drop of a solved single-phase stream in a straight pipe (Darcy-Weisbach, Colebrook/Clamond;
    laminar 64/Re): velocity, Re, friction factor, friction + fittings (ΣK) + static (ρ·g·Δz, positive uphill)
    ΔP and outlet pressure. For a gas, flags when ΔP > 10 % of the inlet P (incompressible assumption breaks)."""
    p = _phase_props(flowsheet_id, stream)
    r = H.pipe_dp(p["density_kg_m3"], p["viscosity_Pa_s"], p["mass_flow_kg_s"], inner_diameter_m, length_m, roughness_m,
                  elevation_change_m, fittings_K)
    r["outlet_P_Pa"] = p["P_Pa"] - r["dP_total_Pa"]
    r["properties"] = p
    if p["phase"] == "vapor" and r["dP_total_Pa"] > 0.1 * p["P_Pa"]:
        r["warning"] = "ΔP > 10 % de P de entrada en gas: el modelo incompresible ya no es válido; dividir la línea"
    return r


def size_pipe(flowsheet_id: str, stream: str, max_velocity_m_s: float, schedule: str = "40") -> dict:
    """Smallest common nominal steel pipe (ASME B36.10M, schedule '40' by default) keeping the velocity of a solved
    single-phase stream at or below max_velocity_m_s; returns NPS, inner diameter, velocity, Re and ΔP per 100 m.
    Typical limits: liquids 1-3 m/s (pump suction ~1), gases 10-30 m/s."""
    p = _phase_props(flowsheet_id, stream)
    r = H.size_pipe(p["density_kg_m3"], p["viscosity_Pa_s"], p["mass_flow_kg_s"], max_velocity_m_s, schedule)
    r["properties"] = p
    return r


def heat_exchanger_area(duty_kW: float, hot_in_K: float, hot_out_K: float, cold_in_K: float, cold_out_K: float,
                        U_W_m2K: float, shell_passes: int = 0) -> dict:
    """Heat-exchanger area A = Q/(U·F·LMTD). shell_passes = 0 → pure counter-current (F = 1); N ≥ 1 → 1-N
    shell-and-tube with 2N tube passes (F by Fakheri). Raises on temperature cross; warns when F < 0.75."""
    return H.hx_area(duty_kW * 1e3, hot_in_K, hot_out_K, cold_in_K, cold_out_K, U_W_m2K, shell_passes)


def tube_film_coefficient(flowsheet_id: str, stream: str, inner_diameter_m: float, velocity_m_s: float) -> dict:
    """Tube-side film heat-transfer coefficient h (W/m²K) of a solved single-phase stream: Gnielinski if turbulent,
    Nu = 3.66 if laminar. Combine inner and outer h into U with fouling before calling heat_exchanger_area."""
    p = _phase_props(flowsheet_id, stream)
    if p["thermal_conductivity_W_mK"] is None or p["cp_J_kgK"] is None:
        raise ValueError(f"{stream}: k or Cp not available")
    r = H.tube_film_coefficient(p["density_kg_m3"], p["viscosity_Pa_s"], p["thermal_conductivity_W_mK"], p["cp_J_kgK"],
                                velocity_m_s, inner_diameter_m)
    r["properties"] = p
    return r


def tubular_reactor_field(volumetric_flow_m3_s: float, tube_diameter_m: float, n_tubes: int, diffusivity_m2_s: float,
                          tube_length_m: float | None = None, volume_m3: float | None = None,
                          k_1_s: float | None = None, A_1_s: float | None = None, E_J_mol: float | None = None, T_K: float | None = None,
                          density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None, reynolds: float | None = None,
                          nz: int = 300, nr: int = 30, figure_path: str | None = None) -> dict:
    """Real tubular reactor vs ideal PFR: first-order isothermal reaction in a bundle of n_tubes laminar tubes, solved as a
    2-D axisymmetric advection-diffusion-reaction field with FEniCSx (separate process, ~5 s). Give the volumetric flow at
    reactor conditions, tube diameter, number of tubes, tube_length_m OR volume_m3, the molecular diffusivity (see
    diffusivity_estimate) and k_1_s or Arrhenius A_1_s + E_J_mol + T_K (E in J/mol). Pass density_kg_m3 and viscosity_Pa_s
    (from stream_transport_properties) so Re confirms the laminar regime; without them the result is not verified.
    Returns X_plug_flow (= ideal PFR, 1 − e^(−kτ)), X_laminar (mixing-cup, with radial diffusion), X_laminar_segregated
    (no radial diffusion), laminar_penalty_points, the outlet radial profile and `evidence` (plug reproduces the analytic
    value, mesh independence, bounds, Re < 2300 → verified). Use it when a card asks how a real/laminar tubular reactor
    differs from the PFR, for radial profiles, or for the effect of tube diameter on conversion."""
    return RF.tubular_reactor_field(volumetric_flow_m3_s, tube_diameter_m, n_tubes, diffusivity_m2_s, tube_length_m, volume_m3,
                                    k_1_s, A_1_s, E_J_mol, T_K, density_kg_m3, viscosity_Pa_s, reynolds, nz, nr, figure_path)


def vessel_residence_time_cfd(length_m: float, width_m: float, height_m: float, volumetric_flow_m3_s: float,
                              kinematic_viscosity_m2_s: float, inlet: dict, outlet: dict,
                              cells_per_max_dimension: int = 30, mesh_study: bool = True,
                              first_order_k_1_s: float | None = None) -> dict:
    """Is this tank really a CSTR? Residence-time behaviour of a flow-through box vessel (no impeller) solved with
    OpenFOAM in a separate process: steady flow (simpleFoam; laminar, or k-ε above inlet Re 2000) and the local mean
    age of the fluid. inlet / outlet = {'face': 'x-'|'x+'|'y-'|'y+'|'z-'|'z+', 'center_m': [c1, c2] on that wall
    (the two other axes in x, y, z order), 'size_m': [s1, s2]} (rectangular openings, at least 2 cells wide).
    Returns τ = V/Q, the volume-mean age over τ (1 = CSTR, 0.5 = plug flow), the RTD variance σ²/τ² (> 1 = dead
    volume or short-circuit), equivalent tanks in series N, an age histogram (stagnant volume = age > 3τ), a
    diagnosis and, with first_order_k_1_s, the conversion of the real RTD next to ideal CSTR and PFR. evidence:
    simpleFoam converged, the flow-weighted outlet age equals τ within 3 % (a theorem), and with mesh_study the coarse
    and 1.5× finer meshes agree (⟨a⟩ 5 %, σ²/τ² 10 % or 0.05). A 30-cell run with mesh study takes about a minute."""
    return VC.vessel_rtd(length_m, width_m, height_m, volumetric_flow_m3_s, kinematic_viscosity_m2_s, inlet, outlet,
                         cells_per_max_dimension, mesh_study, first_order_k_1_s)


def stirred_tank_cfd(tank_diameter_m: float, liquid_height_m: float, impeller_diameter_m: float, clearance_m: float,
                     rotational_speed_rps: float, density_kg_m3: float, viscosity_Pa_s: float, n_baffles: int = 4,
                     baffle_width_m: float | None = None, blade_thickness_m: float | None = None,
                     cells_across_tank: int = 64, mesh_study: bool = True) -> dict:
    """Power draw of a baffled flat-bottom tank stirred by a 6-blade Rushton turbine, solved with OpenFOAM in a
    separate process (steady MRF, k-ε, snappyHexMesh). Geometry is the user's: tank T, liquid height H, impeller D and
    clearance C (bottom to disc); baffle width defaults to T/10 and blade thickness to 0.03 D (stated in assumptions).
    Returns the power number Np from the impeller torque, the power P = Np ρN³D⁵, P/V and W/kg, Re, Fr, tip speed,
    Kolmogorov scale and the mixing time θ95 from Grenville's correlation (labelled: a correlation, not CFD; only
    for H = T and turbulent flow). evidence: torque balance impeller vs walls < 3 % (angular momentum), Np steady
    within 8 % over the last 500 iterations, three meshes (64/49/38 cells across the tank) converging monotonically
    with grid convergence index < 10 % (Celik 2008; Np_extrapolated is the Richardson value), checkMesh, Re > 1e4 and
    Np in the literature band 4-6. With mesh study it takes about 80 minutes on 4 cores; mesh_study=False ~45 min."""
    return ST.stirred_tank_rushton(tank_diameter_m, liquid_height_m, impeller_diameter_m, clearance_m,
                                   rotational_speed_rps, density_kg_m3, viscosity_Pa_s, n_baffles, baffle_width_m,
                                   blade_thickness_m, cells_across_tank, mesh_study)


TRANSPORT_TOOLS = [stream_transport_properties, pipe_hydraulics, size_pipe, heat_exchanger_area, tube_film_coefficient,
                   tubular_reactor_field, vessel_residence_time_cfd, stirred_tank_cfd]
