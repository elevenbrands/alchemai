"""Agent tools for transport phenomena on solved streams: pipe hydraulics, pipe sizing, heat-exchanger area,
tube-side film coefficient. Fluid properties come from the DWSIM stream (liquid phase if vapor fraction < 0.5,
else vapor); the correlations live in `packages.transport.hydraulics`."""
from __future__ import annotations

from packages.dwsim_client.tools import _s
from packages.transport import hydraulics as H


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


TRANSPORT_TOOLS = [stream_transport_properties, pipe_hydraulics, size_pipe, heat_exchanger_area, tube_film_coefficient]
