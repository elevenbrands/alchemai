"""Agent tools for flow devices and non-ideal lines (bloque M4a): relief valves, control valves, fittings,
compressible gas lines and two-phase pressure drop. They need no flowsheet: pass properties from
stream_transport_properties or the user. The engine and its evidence live in `packages.transport.flow_devices`."""
from __future__ import annotations

from packages.transport import flow_devices as FD
from packages.transport.hydraulics import STEEL_ROUGHNESS_M


def relief_valve_sizing(service: str, relieving_rate_kg_s: float, set_pressure_Pa_g: float, overpressure_fraction: float,
                        back_pressure_Pa_g: float, T_K: float | None = None, MW: float | None = None, Z: float | None = None,
                        k: float | None = None, density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None,
                        valve_type: str = "conventional", rupture_disk_upstream: bool = False,
                        Kd: float | None = None) -> dict:
    """Pressure-relief valve sizing by API 520 Part I, orifice letter by API 526 (D…T, several valves if above T).
    service: 'gas' (needs T_K, MW, Z, k = Cp/Cv at relieving conditions; critical or subcritical flow decided by the
    back pressure), 'steam' (needs T_K; Napier with KN and superheat KSH) or 'liquid' (needs density_kg_m3;
    viscosity_Pa_s adds the Kv correction). Pressures are GAUGE: set_pressure_Pa_g and back_pressure_Pa_g (built-up +
    superimposed). overpressure_fraction is the code allowance of the scenario: 0.10 single valve, 0.16 multiple valves,
    0.21 fire (API 521): the user's design basis decides it. valve_type 'conventional' | 'balanced_bellows' (Kb, Kw from
    API 520 figures); rupture_disk_upstream → Kc = 0.9. Kd default is the API preliminary value (0.975 gas/steam, 0.65
    liquid). Returns required_area_mm2, orifice, orifice_area_mm2, valves, flow regime, coefficients, warnings
    (conventional valve with back pressure > 10 % of set) and evidence: the printed API 520 SI equation recomputed
    independently must match. The relieving rate comes from the scenario (fire, blocked outlet, cooling failure): this
    tool never invents it. Use it for the HAZOP 'more pressure' safeguard."""
    return FD.relief_valve(service, relieving_rate_kg_s, set_pressure_Pa_g, overpressure_fraction, back_pressure_Pa_g, T_K,
                           MW, Z, k, density_kg_m3, viscosity_Pa_s, valve_type, rupture_disk_upstream, Kd)


def control_valve_sizing(fluid: str, P1_Pa: float, P2_Pa: float, FL: float, Fd: float, flow_m3_s: float,
                         density_kg_m3: float | None = None, vapor_pressure_Pa: float | None = None,
                         critical_pressure_Pa: float | None = None, viscosity_Pa_s: float | None = None,
                         T_K: float | None = None, MW: float | None = None, gamma: float | None = None,
                         Z: float | None = None, xT: float | None = None, pipe_inlet_diameter_m: float | None = None,
                         pipe_outlet_diameter_m: float | None = None, valve_diameter_m: float | None = None) -> dict:
    """Control-valve flow coefficient by IEC 60534-2-1 / ISA-75.01.01: Kv (m³/h) and Cv (US gpm), choked flow,
    cavitation/flashing (liquid) or critical flow (gas), laminar correction. Absolute pressures P1_Pa > P2_Pa.
    fluid 'liquid': flow_m3_s at inlet conditions + density_kg_m3, vapor_pressure_Pa, critical_pressure_Pa,
    viscosity_Pa_s. fluid 'gas': flow_m3_s at 0 °C and 1 atm (normal m³/s) + T_K, MW, gamma (Cp/Cv), Z,
    viscosity_Pa_s, xT. FL, Fd (and xT) are properties of the valve style from the manufacturer or IEC 60534-2-1
    table 2: ask the user, never assume. With reducers give all three of pipe_inlet_diameter_m,
    pipe_outlet_diameter_m and valve_diameter_m (piping factors FP, FLP, xTP). evidence recomputes the IEC equations
    independently (exact without reducers; within 5 % with reducers, where the standard stops its iteration on the
    conservative side)."""
    return FD.control_valve(fluid, P1_Pa, P2_Pa, FL, Fd, flow_m3_s, density_kg_m3, vapor_pressure_Pa, critical_pressure_Pa,
                            viscosity_Pa_s, T_K, MW, gamma, Z, xT, pipe_inlet_diameter_m, pipe_outlet_diameter_m,
                            valve_diameter_m)


def fittings_resistance(fittings: list[dict], inner_diameter_m: float, Re: float, method: str = "Darby3K") -> dict:
    """Resistance coefficients of pipe fittings and valves, K depending on Re and size: Darby 3-K (default, 34
    fittings) or Hooper 2-K (32). fittings = [{'name': exact catalog name, 'count': n}]; call with fittings=[] to get
    the catalog names. Re and inner_diameter_m from pipe_hydraulics. Returns K per item and K_total, to pass as
    fittings_K to pipe_hydraulics. An unknown name raises with the closest catalog names."""
    return FD.fittings_K(fittings, inner_diameter_m, Re, method)


def compressible_gas_line(P1_Pa: float, T_K: float, MW: float, Z: float, k: float, viscosity_Pa_s: float,
                          inner_diameter_m: float, length_m: float, mass_flow_kg_s: float | None = None,
                          P2_Pa: float | None = None, roughness_m: float = STEEL_ROUGHNESS_M) -> dict:
    """Gas pipeline with large pressure drop (compressible, isothermal ideal gas with constant Z): give mass_flow_kg_s
    to get the outlet pressure, or P2_Pa to get the mass flow. Use it instead of pipe_hydraulics when ΔP > 10 % of P1.
    Returns P2, ΔP, velocities and Mach in/out, the isothermal choke pressure (outlet velocity = √(ZRT/M)), whether
    the line chokes, and the error the incompressible formula would make. A flow above the line's maximum raises;
    a P2 below the choke pressure returns the maximum flow flagged choked. evidence: the momentum ODE integrated
    numerically must reproduce the closed-form P2."""
    return FD.gas_line(P1_Pa, T_K, MW, Z, k, viscosity_Pa_s, inner_diameter_m, length_m, mass_flow_kg_s, P2_Pa, roughness_m)


def two_phase_pressure_drop(mass_flow_kg_s: float, quality: float, liquid_density_kg_m3: float,
                            gas_density_kg_m3: float, liquid_viscosity_Pa_s: float, gas_viscosity_Pa_s: float,
                            surface_tension_N_m: float, inner_diameter_m: float, length_m: float,
                            roughness_m: float = STEEL_ROUGHNESS_M) -> dict:
    """Frictional pressure drop of gas-liquid flow in a horizontal pipe at constant quality (vapor mass fraction,
    0 < x < 1): Friedel, Müller-Steinhagen-Heck, Chisholm and Lockhart-Martinelli side by side, the one recommended by
    Whalley's criterion (viscosity ratio and mass flux), the spread between methods (these correlations scatter
    ±30-50 %: report the range) and the Taitel-Dukler flow regime (stratified, intermittent/slug, annular, bubbly).
    evidence: the recommended method must reduce to single-phase Darcy-Weisbach at x → 0 and x → 1. Elevation and
    acceleration terms are not included (DWSIM pipe unit with Beggs-Brill covers them)."""
    return FD.two_phase_line(mass_flow_kg_s, quality, liquid_density_kg_m3, gas_density_kg_m3, liquid_viscosity_Pa_s,
                             gas_viscosity_Pa_s, surface_tension_N_m, inner_diameter_m, length_m, roughness_m)


FLOW_TOOLS = [relief_valve_sizing, control_valve_sizing, fittings_resistance, compressible_gas_line, two_phase_pressure_drop]
