"""Agent tools for gas absorption (bloque 22): equilibrium stages by Kremser, packed height by Onda + Colburn,
diffusivity estimates. Correlations and their sources: packages/transport/mass_transfer.py."""
from __future__ import annotations

import math

from packages.dwsim_client.tools import _s
from packages.transport import mass_transfer as MT


def absorption_stages(gas_flow_mol_s: float, liquid_flow_mol_s: float, K: float, y_in: float,
                      fraction_absorbed: float | None = None, stages: float | None = None, x_in: float = 0.0) -> dict:
    """Kremser for a dilute solute, straight lines: absorption factor A = L/(K·V). Give fraction_absorbed to get the
    equilibrium stages needed, or stages to get the fraction absorbed. K = y/x of the solute at column conditions (get
    it from a DWSIM flash of gas + solvent). Check against add_unit kind 'absorption_column' (rigorous) when it matters."""
    A = liquid_flow_mol_s / (K * gas_flow_mol_s)
    out = {"absorption_factor_A": A, "max_fraction_absorbed": min(1.0, A) if A < 1 else 1.0}
    if stages is not None:
        out["fraction_absorbed"] = MT.kremser_fraction_absorbed(A, float(stages))
        out["stages"] = stages
    elif fraction_absorbed is not None:
        y_out = y_in * (1.0 - fraction_absorbed)
        out["stages_needed"] = MT.kremser_stages(A, y_in, y_out, K, x_in)
        out["N_OG_colburn"] = MT.colburn_NOG(y_in, y_out, K, A, x_in)
        out["fraction_absorbed"] = fraction_absorbed
    else:
        raise ValueError("give fraction_absorbed or stages")
    if A < 1.2:
        out["note"] = "A < 1.2: recovery limited by solvent rate; consider more solvent (rule of thumb A ≈ 1.2-2)"
    return out


def _props(fid: str, name: str, phase: int) -> dict:
    b = _s(fid)["streams"][name]
    ms = getattr(b.Object, "__implementation__", b.Object)
    pr = ms.Phases[phase].Properties
    get = lambda n: pr.GetType().GetProperty(n).GetValue(pr)
    m, n = float(b.MassFlowKgPerSecond), float(b.MolarFlowMolPerSecond)
    return {"rho": float(get("density")), "mu": float(get("viscosity")), "sigma": get("surfaceTension"),
            "mass_kg_s": m, "molar_mol_s": n, "MW_kg_mol": m / n, "T_K": float(b.TemperatureK)}


def packed_column_height(flowsheet_id: str, gas_stream: str, liquid_stream: str, column_diameter_m: float,
                         packing_specific_area_m2_m3: float, packing_size_m: float, packing_critical_surface_tension_N_m: float,
                         K: float, y_in: float, fraction_absorbed: float, D_liquid_m2_s: float, D_gas_m2_s: float,
                         x_in: float = 0.0) -> dict:
    """Packed absorber height for a dilute solute: Z = H_OG · N_OG. N_OG by Colburn; H_OG = H_G + (K·Gm/Lm)·H_L with
    film coefficients and wetted area from Onda et al. (1968). Properties (ρ, μ, σ, molar mass) come from the solved
    gas inlet and lean liquid streams. Packing data (a_t, nominal size, critical surface tension σc of the material) and
    diffusivities are inputs: take them from a vendor table / handbook or diffusivity_estimate. Does not check flooding:
    keep the gas rate below ~70-80 % of flooding by a separate hydraulic check."""
    g, l = _props(flowsheet_id, gas_stream, 2), _props(flowsheet_id, liquid_stream, 1)
    if l["sigma"] is None:
        raise ValueError(f"{liquid_stream}: no surface tension from DWSIM")
    area = math.pi * column_diameter_m ** 2 / 4.0
    Gm, Lm = g["molar_mol_s"] / area, l["molar_mol_s"] / area
    Gw, Lw = g["mass_kg_s"] / area, l["mass_kg_s"] / area
    A = Lm / (K * Gm)
    y_out = y_in * (1.0 - fraction_absorbed)
    N_OG = MT.colburn_NOG(y_in, y_out, K, A, x_in)
    o = MT.onda(Lw, Gw, l["rho"], l["mu"], float(l["sigma"]), D_liquid_m2_s, g["rho"], g["mu"], D_gas_m2_s, g["T_K"],
                packing_specific_area_m2_m3, packing_size_m, packing_critical_surface_tension_N_m)
    h = MT.packed_height(Gm, Lm, K, o["a_w_m2_m3"], o["kG_m_s"], o["kL_m_s"], g["rho"] / g["MW_kg_mol"], l["rho"] / l["MW_kg_mol"], N_OG)
    return h | {"absorption_factor_A": A, "onda": o, "gas_mass_velocity_kg_m2s": Gw, "liquid_mass_velocity_kg_m2s": Lw,
                "stages_equivalent_kremser": MT.kremser_stages(A, y_in, y_out, K, x_in),
                "HETP_m": h["height_m"] / MT.kremser_stages(A, y_in, y_out, K, x_in),
                "warning": "Sin verificación de inundación ni de mojado mínimo; el artículo de Onda reporta ±20 % para área mojada y kL"}


def diffusivity_estimate(method: str, T_K: float, P_Pa: float | None = None, M_A: float | None = None, M_B: float | None = None,
                         sum_v_A: float | None = None, sum_v_B: float | None = None, mu_solvent_cP: float | None = None,
                         M_solvent: float | None = None, phi_solvent: float | None = None, V_solute_cm3_mol: float | None = None) -> dict:
    """method 'fuller' (gas pair: P_Pa, M_A, M_B and diffusion volumes Σv of A and B from the Fuller-Schettler-Giddings
    table) or 'wilke_chang' (dilute solute in a liquid: solvent viscosity in cP, solvent molar mass, association factor φ,
    solute molar volume at its normal boiling point in cm³/mol). Returns D in m²/s. The volumes and φ are inputs on
    purpose: use the table of the edition you cite."""
    if method == "fuller":
        D = MT.fuller(T_K, P_Pa, M_A, M_B, sum_v_A, sum_v_B)
    elif method == "wilke_chang":
        D = MT.wilke_chang(T_K, mu_solvent_cP, M_solvent, phi_solvent, V_solute_cm3_mol)
    else:
        raise ValueError("method must be fuller or wilke_chang")
    return {"D_m2_s": D, "method": method}


MASS_TRANSFER_TOOLS = [absorption_stages, packed_column_height, diffusivity_estimate]
