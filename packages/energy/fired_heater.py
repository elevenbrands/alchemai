"""Radiant and convection sections of fired heaters and boilers (bloque M18, part B).

Radiant section: one well-stirred gas zone (the Lobo-Evans / Hottel one-gas-zone idea). The flue gas is grey at the gas
temperature, the tubes are a grey sink, the refractory is radiatively adiabatic and the enclosure is «speckled» (every surface
sees sink and refractory in proportion to their areas). The radiation network gives the total exchange area

    1/GS = (1 - e_t)/(e_t A1) + 1 / [ e_g A1 + 1 / ( 1/(e_g A_R) + A_T/(A1 A_R (1 - e_g)) ) ]      A1 = alpha*Acp (+ shield)

and F = GS/A1 is the overall exchange factor: Q_rad = sigma F A1 (Tg^4 - Tw^4). The firebox heat balance fixes Tg, which is
also the bridgewall temperature (complete mixing). Flue-gas enthalpy comes from `packages.energy.combustion` (NASA).

Sourced numbers (nothing typed from memory):
  * Gas emissivity: weighted sum of grey gases of Smith, Shen and Friedman (J. Heat Transfer 104 (1982) 602-608), 3 grey
    gases + clear, coefficients as reproduced in Table 7 of Marzouk and Huckaby, «New Weighted Sum of Gray Gases (WSGG) Models
    for Radiation Calculation in Carbon Capture Simulations», 7th U.S. National Combustion Meeting 2011 (arXiv:2411.18467),
    read 2026-11-14; mixtures by the piecewise-linear rule of its Appendix B. Valid at 1 atm total, 600-2400 K,
    0.001-10 atm m.
  * Mean beam length 3.6 V/A: the same paper («The mean path length, L, was approximated by 3.6 volume/area»).
  * Tube-row effectiveness alpha: exact crossed-strings view factor of a plane to one row of tubes, plus the re-radiation of
    the refractory behind it, alpha = F (2 - F); a test integrates the view factor numerically.
  * Convection across the bank: Zukauskas as implemented in `ht.conv_tube_bank.Nu_Zukauskas_Bejan`; gas viscosity,
    conductivity and cp from Cantera (GRI-Mech 3.0 transport data, mixture averaged).

Every operating input (firing, geometry, tube emissivity, wall temperature or inside coefficient, convective coefficient in
the firebox, setting loss, allowable flux) is the caller's; nothing typical is assumed."""
from __future__ import annotations

import math
from functools import lru_cache

import cantera as ct
from scipy.optimize import brentq

from packages.energy import combustion as CB

SIGMA = ct.stefan_boltzmann            # W/(m2 K4)
ATM = ct.one_atm
WSGG_SOURCE = ("Smith, Shen y Friedman, J. Heat Transfer 104 (1982) 602-608; coeficientes leídos de la Tabla 7 de Marzouk y "
               "Huckaby 2011 (arXiv:2411.18467)")
# node -> (k_i [1/(atm m)], b_ij: a_i = sum_j b_ij T^(j-1))
SMITH_1982 = {
    "CO2_limit": ((0.3966, 15.64, 394.3), ((0.4334e-1, 2.620e-4, -1.560e-7, 2.565e-11),
                                           (-0.4814e-1, 2.822e-4, -1.794e-7, 3.274e-11),
                                           (0.5492e-1, 0.1087e-4, -0.3500e-7, 0.9123e-11))),
    "ratio_1": ((0.4303, 7.055, 178.1), ((5.150e-1, -2.303e-4, 0.9779e-7, -1.494e-11),
                                         (0.7749e-1, 3.399e-4, -2.297e-7, 3.770e-11),
                                         (1.907e-1, -1.824e-4, 0.5608e-7, -0.5122e-11))),
    "ratio_2": ((0.4201, 6.516, 131.9), ((6.508e-1, -5.551e-4, 3.029e-7, -5.353e-11),
                                         (-0.2504e-1, 6.112e-4, -3.882e-7, 6.528e-11),
                                         (2.718e-1, -3.118e-4, 1.221e-7, -1.612e-11))),
    "H2O_limit": ((0.4098, 6.325, 120.5), ((5.977e-1, -5.119e-4, 3.042e-7, -5.564e-11),
                                           (0.5677e-1, 3.333e-4, -1.967e-7, 2.718e-11),
                                           (1.800e-1, -2.334e-4, 1.008e-7, -1.454e-11))),
    "H2O_1atm": ((0.4496, 7.113, 119.7), ((6.324e-1, -8.358e-4, 6.135e-7, -13.03e-11),
                                          (-0.2016e-1, 7.145e-4, -5.212e-7, 9.868e-11),
                                          (3.500e-1, -5.040e-4, 2.425e-7, -3.888e-11))),
}


def _need(d: dict, keys, what: str) -> None:
    missing = [k for k in keys if d.get(k) is None]
    if missing:
        raise ValueError(f"{what}: faltan {missing}; son datos del equipo o de operación y no se suponen")


def gas_emissivity(T_K: float, p_H2O_Pa: float, p_CO2_Pa: float, path_length_m: float, total_pressure_Pa: float = ATM) -> dict:
    """Total emissivity of an H2O-CO2 mixture in a transparent gas (WSGG, Smith et al. 1982)."""
    pw, pc = p_H2O_Pa / ATM, p_CO2_Pa / ATM
    if pw < 0 or pc < 0 or pw + pc <= 0 or path_length_m <= 0 or T_K <= 0:
        raise ValueError("gas_emissivity: da presiones parciales no negativas (alguna positiva), longitud y T positivas")
    rr = pw / (pw + pc)
    top = "H2O_limit" if pw <= 0.5 else "H2O_1atm"
    nodes = ((0.0, "CO2_limit"), (0.5, "ratio_1"), (2.0 / 3.0, "ratio_2"), (1.0, top))
    for (r0, n0), (r1, n1) in zip(nodes, nodes[1:]):
        if rr <= r1 + 1e-12:
            break
    u = (rr - r0) / (r1 - r0)
    (k0, b0), (k1, b1) = SMITH_1982[n0], SMITH_1982[n1]
    pL = (pw + pc) * path_length_m
    eps, weights = 0.0, []
    for i in range(3):
        k = k0[i] + u * (k1[i] - k0[i])
        a = sum((b0[i][j] + u * (b1[i][j] - b0[i][j])) * T_K ** j for j in range(4))
        weights.append(a)
        eps += a * (1.0 - math.exp(-k * pL))
    out_of_range = []
    if not 600.0 <= T_K <= 2400.0:
        out_of_range.append(f"T = {T_K:.0f} K fuera de 600-2400 K")
    if not 0.001 <= pL <= 10.0:
        out_of_range.append(f"pL = {pL:.4g} atm·m fuera de 0.001-10")
    if abs(total_pressure_Pa / ATM - 1.0) > 0.1:
        out_of_range.append(f"presión total {total_pressure_Pa/ATM:.2f} atm; el ajuste es para 1 atm")
    if min(weights) < 0 or not 0 < eps < 1:
        out_of_range.append("pesos o emisividad no físicos")
    return {"emissivity": eps, "pL_atm_m": pL, "H2O_fraction_of_absorbers": rr, "interpolated_between": [n0, n1],
            "in_fitted_range": not out_of_range, "out_of_range": out_of_range, "source": WSGG_SOURCE}


def tube_row_alpha(pitch_m: float, diameter_m: float) -> dict:
    """Fraction of the radiation crossing the cold plane that one row of tubes in front of a refractory wall absorbs."""
    if not 0 < diameter_m <= pitch_m:
        raise ValueError("tube_row_alpha: el paso debe ser mayor o igual que el diámetro exterior")
    x = diameter_m / pitch_m
    direct = 1.0 - math.sqrt(1.0 - x * x) + x * math.atan(math.sqrt(1.0 / (x * x) - 1.0)) if x < 1 else 1.0
    return {"direct_view_factor": direct, "alpha": direct * (2.0 - direct), "pitch_to_diameter": 1.0 / x}


def exchange_factor(gas_emissivity_value: float, sink_area_m2: float, envelope_area_m2: float, tube_emissivity: float) -> float:
    """Overall exchange factor F on the sink area (see the module docstring)."""
    eg, A1, AT, et = gas_emissivity_value, sink_area_m2, envelope_area_m2, tube_emissivity
    if not (0 < eg < 1 and 0 < et <= 1 and 0 < A1 <= AT * (1 + 1e-9)):
        raise ValueError("exchange_factor: emisividades en (0, 1] y área de sumidero no mayor que la envolvente")
    AR = max(AT - A1, 0.0)
    via_refractory = 0.0 if AR <= 0 else 1.0 / (1.0 / (eg * AR) + AT / (A1 * AR * (1.0 - eg)))
    return 1.0 / ((1.0 - et) / (et * A1) + 1.0 / (eg * A1 + via_refractory)) / A1


def _wall_resistance(p: dict, Do: float) -> float:
    """(Tw_outside - T_bulk) per unit outside flux, m2 K/W: inside film + inside fouling + tube wall."""
    Di = p["tube_inside_diameter_m"]
    if not 0 < Di < Do:
        raise ValueError("tube_inside_diameter_m debe ser menor que el diámetro exterior")
    return Do / (Di * p["inside_coefficient_W_m2K"]) + p["inside_fouling_m2K_W"] * Do / Di + \
        Do * math.log(Do / Di) / (2.0 * p["wall_conductivity_W_mK"])


def gas_radiation_coefficient(gas_T_K: float, wall_T_K: float, gas_emissivity_value: float, tube_emissivity: float) -> float:
    """Radiation coefficient of a grey gas to the tube surface that encloses it, W/(m2 K)."""
    Fg = 1.0 / (1.0 / gas_emissivity_value + 1.0 / tube_emissivity - 1.0)
    return SIGMA * Fg * (gas_T_K ** 4 - wall_T_K ** 4) / (gas_T_K - wall_T_K)


def bank_beam_length(transverse_pitch_m: float, longitudinal_pitch_m: float, diameter_m: float) -> float:
    """3.6 V/A of the gas space around one tube of a bank, m."""
    return 3.6 * (transverse_pitch_m * longitudinal_pitch_m - math.pi * diameter_m ** 2 / 4) / (math.pi * diameter_m)


PROCESS_KEYS = ("bulk_T_K", "inside_coefficient_W_m2K", "tube_inside_diameter_m", "wall_conductivity_W_mK",
                "inside_fouling_m2K_W")


def radiant_section(fuel: dict, air: dict, pressure_Pa: float, fuel_flow_kg_s: float, geometry: dict, tube_emissivity: float,
                    convective_coefficient_W_m2K: float, setting_loss_fraction: float, setting_loss_basis: str,
                    excess_air_fraction: float | None = None, flue_O2: dict | None = None,
                    flue_CO_dry_mole_fraction: float | None = None, tube_wall_T_K: float | None = None,
                    process: dict | None = None, gas_emissivity_override: float | None = None,
                    exchange_factor_override: float | None = None) -> dict:
    """Rating of the radiant section: bridgewall temperature, duty absorbed and average flux for a firing rate and a geometry."""
    c = CB.solve_combustion(fuel, air, pressure_Pa, excess_air_fraction, flue_O2, flue_CO_dry_mole_fraction)
    if fuel_flow_kg_s is None or fuel_flow_kg_s <= 0:
        raise ValueError("fuel_flow_kg_s: da el combustible quemado (combustion_efficiency lo calcula para una carga)")
    if setting_loss_fraction is None or setting_loss_basis not in ("HHV", "LHV") or not 0 <= setting_loss_fraction < 0.5:
        raise ValueError("da setting_loss_fraction (pérdida por paredes del hogar, 0 si es nula) y setting_loss_basis")
    if tube_emissivity is None or convective_coefficient_W_m2K is None or convective_coefficient_W_m2K < 0:
        raise ValueError("da tube_emissivity y convective_coefficient_W_m2K (0 si no hay convección en el hogar)")
    g = geometry or {}
    _need(g, ("tube_outside_diameter_m", "tube_pitch_m", "tube_effective_length_m", "n_tubes", "envelope_area_m2"), "geometry")
    if g.get("mean_beam_length_m") is None and g.get("volume_m3") is None and gas_emissivity_override is None \
            and exchange_factor_override is None:
        raise ValueError("geometry: da volume_m3 (la longitud media de haz se toma como 3.6 V/A) o mean_beam_length_m")
    if (tube_wall_T_K is None) == (process is None):
        raise ValueError("da tube_wall_T_K (temperatura media de pared de tubo) o process {bulk_T_K, inside_coefficient_W_m2K, "
                         "tube_inside_diameter_m, wall_conductivity_W_mK, inside_fouling_m2K_W}, uno solo")
    Do, pitch, L, n = (g["tube_outside_diameter_m"], g["tube_pitch_m"], g["tube_effective_length_m"], g["n_tubes"])
    A_tubes, A_cp = n * math.pi * Do * L, n * pitch * L
    row = tube_row_alpha(pitch, Do)
    alpha = g["alpha"] if g.get("alpha") is not None else row["alpha"]
    A_shield = float(g.get("shield_cold_plane_area_m2") or 0.0)
    A1, AT = alpha * A_cp + A_shield, g["envelope_area_m2"]
    if A1 > AT:
        raise ValueError("el plano frío equivalente supera el área de la envolvente: revisa envelope_area_m2 (área interior "
                         "total del hogar, incluida la que cubren los tubos)")
    beam = g.get("mean_beam_length_m") or (3.6 * g["volume_m3"] / AT if g.get("volume_m3") else None)
    wet = c.flue_mol
    p_w, p_c = c.flue.get("H2O", 0.0) / wet * pressure_Pa, c.flue.get("CO2", 0.0) / wet * pressure_Pa
    m = float(fuel_flow_kg_s)
    setting = m * setting_loss_fraction * (c.HHV if setting_loss_basis == "HHV" else c.LHV)
    released = m * (c.LHV + c.credits - c.co_loss)
    R_wall = None
    if process is not None:
        _need(process, PROCESS_KEYS, "process")
        R_wall = _wall_resistance(process, Do)

    def state(Tg: float) -> dict:
        em = {"emissivity": gas_emissivity_override} if gas_emissivity_override is not None else \
            (gas_emissivity(Tg, p_w, p_c, beam, pressure_Pa) if exchange_factor_override is None else {"emissivity": None})
        F = exchange_factor_override if exchange_factor_override is not None else \
            exchange_factor(em["emissivity"], A1, AT, tube_emissivity)

        def duty(Tw):
            q_rad = SIGMA * F * A1 * (Tg ** 4 - Tw ** 4)
            return q_rad, convective_coefficient_W_m2K * A_tubes * (Tg - Tw)
        if R_wall is None:
            Tw = float(tube_wall_T_K)
        else:
            Tb = process["bulk_T_K"]
            Tw = brentq(lambda t: t - Tb - R_wall * (duty(t)[0] * alpha * A_cp / A1 + duty(t)[1]) / A_tubes, Tb, Tg, xtol=1e-8) \
                if Tg > Tb else Tb
        q_rad, q_conv = duty(Tw)
        return {"Tw": Tw, "em": em, "F": F, "q_rad": q_rad, "q_conv": q_conv}

    def residual(Tg):
        s = state(Tg)
        return released - setting - s["q_rad"] - s["q_conv"] - m * c.flue_sensible(Tg)
    T_ad = c.adiabatic_complete_T()
    lo = (tube_wall_T_K if tube_wall_T_K is not None else process["bulk_T_K"]) + 1.0
    if residual(lo) <= 0:
        raise ValueError("con ese fuego el gas no supera la temperatura de pared: no hay solución con transferencia al tubo")
    Tg = brentq(residual, lo, T_ad, xtol=1e-7)
    s = state(Tg)
    q_tubes_rad = s["q_rad"] * alpha * A_cp / A1
    q_shield = s["q_rad"] - q_tubes_rad
    Q_R = q_tubes_rad + s["q_conv"]
    flue_out = m * c.flue_sensible(Tg)
    balance = abs(released - setting - Q_R - q_shield - flue_out) / released
    em = s["em"]
    warnings = list(c.warnings)
    notes = []
    if gas_emissivity_override is None and exchange_factor_override is None and not em["in_fitted_range"]:
        warnings.append("Emisividad del gas fuera del intervalo del ajuste: " + "; ".join(em["out_of_range"]))
    if exchange_factor_override is not None:
        notes.append("Factor de intercambio F dado por el usuario (no calculado con la emisividad del gas).")
    elif gas_emissivity_override is not None:
        notes.append("Emisividad del gas dada por el usuario.")
    out = {
        "bridgewall_T_K": Tg, "effective_gas_T_K": Tg, "adiabatic_complete_combustion_T_K": T_ad,
        "radiant_duty_W": Q_R, "radiant_duty_by_radiation_W": q_tubes_rad, "radiant_duty_by_convection_W": s["q_conv"],
        "shield_radiant_duty_W": q_shield, "flue_gas_sensible_leaving_W": flue_out, "setting_loss_W": setting,
        "heat_release_LHV_W": m * c.LHV, "heat_release_HHV_W": m * c.HHV,
        "absorbed_fraction_of_LHV": (Q_R + q_shield) / (m * c.LHV),
        "average_flux_W_m2_tube_outside": Q_R / A_tubes, "tube_wall_T_K": s["Tw"],
        "tube_wall_from": "dato del usuario" if R_wall is None else
                          "T del fluido + flujo medio × (película interior + ensuciamiento + pared)",
        "radiation": {"gas_emissivity": em["emissivity"], "exchange_factor": s["F"], "alpha": alpha,
                      "alpha_from": "dato del usuario" if g.get("alpha") is not None else
                                    "factor de visión exacto de una fila + rerradiación del refractario",
                      "cold_plane_area_m2": A_cp, "equivalent_cold_plane_area_m2": A1, "envelope_area_m2": AT,
                      "refractory_to_sink_area_ratio": (AT - A1) / A1, "mean_beam_length_m": beam,
                      "pL_atm_m": em.get("pL_atm_m"), "p_H2O_Pa": p_w, "p_CO2_Pa": p_c, "tube_emissivity": tube_emissivity,
                      "gas_emissivity_source": em.get("source")},
        "tube_area_m2": A_tubes, "fuel_kg_s": m, "flue_gas_kg_s": m * c.flue_mass, "excess_air_fraction": c.excess_air,
        "evidence": {"energy_balance_error": balance, "verified_internal": balance < 1e-9,
                     "validated_against_literature": "parcial: ver BENCHMARK.md, filas M18 (balance de una zona y factor de "
                                                     "intercambio reproducidos; la emisividad WSGG difiere de las cartas de "
                                                     "Kern)"},
        "warnings": warnings,
        "assumptions": list(c.assumptions) + notes + [
            "Hogar de una sola zona de gas bien mezclada: la temperatura efectiva del gas es la de salida (pared divisoria). En "
            "hogares altos y estrechos o con quemadores de pared la efectiva puede ser bastante mayor.",
            "Gas gris a la temperatura del gas (misma emisividad para emitir y absorber), tubos grises, refractario "
            "radiativamente adiabático, recinto «moteado».",
            "Longitud media de haz 3.6 V/A salvo que se dé mean_beam_length_m.",
            "Sin llama luminosa ni hollín: solo radiación de CO2 y H2O.",
            "La pérdida por paredes dada se carga entera al hogar."],
        "not_included": ["perfil de flujo de calor a lo largo y alrededor del tubo (flujo pico)", "zonas múltiples",
                         "llama luminosa y partículas", "caída de presión y tiro", "tubos a dos caras o doble fila (da alpha)"],
        "method": {"name": "una zona de gas bien mezclada (Lobo-Evans / Hottel), factor de intercambio de red de radiación",
                   "module": "packages/energy/fired_heater.py", "property_data": CB.SOURCE},
    }
    return out


def radiant_sizing(required_radiant_duty_W: float, allowable_average_flux_W_m2: float, tube_outside_diameter_m: float,
                   tube_effective_length_m: float, tube_pitch_m: float) -> dict:
    """Tube surface for a radiant duty at an allowable average flux given by the user (outside tube area basis)."""
    for v, nme in ((required_radiant_duty_W, "required_radiant_duty_W"), (allowable_average_flux_W_m2,
                   "allowable_average_flux_W_m2"), (tube_outside_diameter_m, "tube_outside_diameter_m"),
                   (tube_effective_length_m, "tube_effective_length_m"), (tube_pitch_m, "tube_pitch_m")):
        if v is None or v <= 0:
            raise ValueError(f"{nme}: dato obligatorio y positivo (el flujo admisible lo fija el usuario, no hay valor típico)")
    area = required_radiant_duty_W / allowable_average_flux_W_m2
    per_tube = math.pi * tube_outside_diameter_m * tube_effective_length_m
    n = math.ceil(area / per_tube - 1e-9)
    row = tube_row_alpha(tube_pitch_m, tube_outside_diameter_m)
    return {"tube_area_required_m2": area, "area_per_tube_m2": per_tube, "n_tubes": n, "tube_area_installed_m2": n * per_tube,
            "average_flux_installed_W_m2": required_radiant_duty_W / (n * per_tube),
            "cold_plane_area_m2": n * tube_pitch_m * tube_effective_length_m, "alpha": row["alpha"],
            "equivalent_cold_plane_area_m2": row["alpha"] * n * tube_pitch_m * tube_effective_length_m,
            "next_step": "Con el número de tubos define el hogar (área de envolvente y volumen) y evalúalo con mode='rating': "
                         "el balance dirá si ese fuego entrega la carga y a qué temperatura de pared divisoria.",
            "assumptions": ["El flujo medio admisible es dato del usuario; el flujo pico (circunferencial y longitudinal) no "
                            "se calcula."]}


@lru_cache(maxsize=1)
def _transport_gas():
    return ct.Solution("gri30.yaml")


def _gas_properties(c: CB.Combustion, T_K: float, P_Pa: float) -> dict:
    gas = _transport_gas()
    x = {"CO2": c.flue.get("CO2", 0.0) + c.flue.get("SO2", 0.0), "H2O": c.flue.get("H2O", 0.0), "N2": c.flue.get("N2", 0.0),
         "O2": c.flue.get("O2", 0.0), "AR": c.flue.get("Ar", 0.0) + c.flue.get("He", 0.0), "CO": c.flue.get("CO", 0.0)}
    gas.TPX = T_K, P_Pa, x
    mu, k, cp = gas.viscosity, gas.thermal_conductivity, gas.cp_mass
    return {"mu": mu, "k": k, "cp": cp, "Pr": cp * mu / k}


BANK_KEYS = ("tube_outside_diameter_m", "tube_inside_diameter_m", "tube_effective_length_m", "tubes_per_row", "n_rows",
             "transverse_pitch_m", "longitudinal_pitch_m", "arrangement", "wall_conductivity_W_mK", "outside_fouling_m2K_W",
             "tube_emissivity")
CONV_PROCESS_KEYS = ("inlet_T_K", "inside_coefficient_W_m2K", "inside_fouling_m2K_W")


def convection_section(fuel: dict, air: dict, pressure_Pa: float, fuel_flow_kg_s: float, gas_inlet_T_K: float, bank: dict,
                       process: dict, excess_air_fraction: float | None = None, flue_O2: dict | None = None,
                       flue_CO_dry_mole_fraction: float | None = None, gas_radiation: bool = True) -> dict:
    """Bare-tube convection bank, row by row, flue gas in cross flow and the process fluid counter-current overall."""
    from ht.conv_tube_bank import Nu_Zukauskas_Bejan
    c = CB.solve_combustion(fuel, air, pressure_Pa, excess_air_fraction, flue_O2, flue_CO_dry_mole_fraction)
    if fuel_flow_kg_s is None or fuel_flow_kg_s <= 0 or gas_inlet_T_K is None:
        raise ValueError("da fuel_flow_kg_s y gas_inlet_T_K (temperatura del gas que entra al banco; la da fired_heater_radiant)")
    b, p = bank or {}, process or {}
    _need(b, BANK_KEYS, "bank")
    _need(p, CONV_PROCESS_KEYS, "process")
    isothermal = bool(p.get("isothermal"))
    if not isothermal and p.get("capacity_rate_W_K") is None:
        raise ValueError("process: da capacity_rate_W_K (flujo × cp del fluido) o isothermal=True (ebullición a T constante)")
    if b["arrangement"] not in ("staggered", "inline"):
        raise ValueError("bank.arrangement: 'staggered' o 'inline'")
    Do, Di, L = b["tube_outside_diameter_m"], b["tube_inside_diameter_m"], b["tube_effective_length_m"]
    n_t, n_r, St, Sl = int(b["tubes_per_row"]), int(b["n_rows"]), b["transverse_pitch_m"], b["longitudinal_pitch_m"]
    if not (0 < Di < Do < St) or Sl <= 0 or n_t < 1 or n_r < 1:
        raise ValueError("bank: se necesita Di < Do < paso transversal, paso longitudinal > 0 y al menos un tubo y una fila")
    width = b.get("duct_width_m") or n_t * St
    gap = width - n_t * Do
    if b["arrangement"] == "staggered":
        diag = math.sqrt(Sl ** 2 + (St / 2) ** 2)
        gap = min(gap, n_t * 2 * (diag - Do))
    if gap <= 0:
        raise ValueError("bank: no queda área libre entre tubos")
    m, m_gas = float(fuel_flow_kg_s), float(fuel_flow_kg_s) * c.flue_mass
    G_max = m_gas / (gap * L)
    A_row = n_t * math.pi * Do * L
    R_in = Do / (Di * p["inside_coefficient_W_m2K"]) + p["inside_fouling_m2K_W"] * Do / Di + \
        Do * math.log(Do / Di) / (2.0 * b["wall_conductivity_W_mK"])
    wet = c.flue_mol
    p_w, p_c = c.flue.get("H2O", 0.0) / wet * pressure_Pa, c.flue.get("CO2", 0.0) / wet * pressure_Pa
    beam = bank_beam_length(St, Sl, Do)
    C_p = None if isothermal else float(p["capacity_rate_W_K"])
    T_in_proc = float(p["inlet_T_K"])
    flags = set()

    def row_solve(Tg_in: float, Tp_hot: float) -> dict:
        def calc(Tg_out):
            Tg = 0.5 * (Tg_in + Tg_out)
            Q = m * (c.flue_sensible(Tg_in) - c.flue_sensible(Tg_out))
            Tp_cold = Tp_hot if isothermal else Tp_hot - Q / C_p
            Tp = 0.5 * (Tp_hot + Tp_cold)
            pr = _gas_properties(c, Tg, pressure_Pa)
            Tw = min(max(Tp + Q / A_row * R_in, Tp), Tg)
            Re = G_max * Do / pr["mu"]
            Nu = Nu_Zukauskas_Bejan(Re=Re, Pr=pr["Pr"], tube_rows=n_r, pitch_parallel=Sl, pitch_normal=St,
                                    Pr_wall=_gas_properties(c, Tw, pressure_Pa)["Pr"])
            h_c = Nu * pr["k"] / Do
            h_r, eg = 0.0, None
            if gas_radiation and Tg - Tw > 1e-6:
                e = gas_emissivity(Tg, p_w, p_c, beam, pressure_Pa)
                eg = e["emissivity"]
                if not e["in_fitted_range"]:
                    flags.add("Emisividad del gas fuera del intervalo del ajuste WSGG (600-2400 K, 0.001-10 atm·m, 1 atm) en "
                              "alguna fila: la parte radiante de esas filas es una extrapolación.")
                h_r = gas_radiation_coefficient(Tg, Tw, eg, b["tube_emissivity"])
            U = 1.0 / (1.0 / (h_c + h_r) + b["outside_fouling_m2K_W"] + R_in)
            return {"Q": Q, "Q_ua": U * A_row * (Tg - Tp), "Tg_out": Tg_out, "Tp_cold": Tp_cold, "Tw": Tw, "Re": Re,
                    "h_c": h_c, "h_r": h_r, "U": U, "eg": eg, "Tg": Tg, "Tp": Tp}
        lo = max(Tp_hot, 300.0)
        f = lambda t: calc(t)["Q"] - calc(t)["Q_ua"]      # noqa: E731
        if Tg_in <= Tp_hot or f(Tg_in - 1e-9) >= 0:
            return calc(Tg_in)
        t_lo = min(lo, Tg_in - 1e-6)
        if f(t_lo) < 0:                                    # the gas would cool below the fluid: pinch in this row
            return calc(t_lo)
        return calc(brentq(f, t_lo, Tg_in, xtol=1e-7))

    def march(Tp_hot_end: float):
        rows, Tg, Tp = [], float(gas_inlet_T_K), Tp_hot_end
        for j in range(n_r):
            r = row_solve(Tg, Tp)
            rows.append(dict(r, row=j + 1, Tg_in=Tg, Tp_hot=Tp))
            Tg, Tp = r["Tg_out"], r["Tp_cold"]
        return rows, Tg, Tp

    if isothermal:
        rows, Tg_out, _ = march(T_in_proc)
    else:
        g = lambda t: march(t)[2] - T_in_proc             # noqa: E731
        hi = float(gas_inlet_T_K)
        if g(T_in_proc) >= 0 or g(hi) <= 0:
            raise ValueError("no hay solución: revisa que el gas entre más caliente que el fluido y la capacidad del fluido")
        rows, Tg_out, _ = march(brentq(g, T_in_proc, hi, xtol=1e-7))
    Q = sum(r["Q"] for r in rows)
    Q_gas = m * (c.flue_sensible(gas_inlet_T_K) - c.flue_sensible(Tg_out))
    rad_share = sum(r["Q"] * r["h_r"] / (r["h_c"] + r["h_r"]) for r in rows) / Q if Q > 0 else 0.0
    Re_all = [r["Re"] for r in rows]
    warnings = list(c.warnings) + sorted(flags)
    if min(Re_all) < 1e3 or max(Re_all) > 2e6:
        warnings.append(f"Reynolds {min(Re_all):.0f}-{max(Re_all):.0f}: fuera del intervalo principal de Zukauskas (1e3-2e6)")
    import CoolProp.CoolProp as CP
    y_w = c.flue.get("H2O", 0.0) / wet
    dew = CP.PropsSI("T", "P", y_w * pressure_Pa, "Q", 0, "Water") if y_w * pressure_Pa > 700 else None
    if dew is not None and min(r["Tw"] for r in rows) < dew:
        warnings.append(f"Alguna pared de tubo queda bajo el punto de rocío del agua ({dew:.1f} K): condensación y corrosión.")
    return {
        "stack_T_K": Tg_out, "gas_inlet_T_K": float(gas_inlet_T_K), "duty_W": Q, "gas_radiation_share_of_duty": rad_share,
        "process_outlet_T_K": rows[0]["Tp_hot"], "process_inlet_T_K": T_in_proc,
        "max_tube_wall_T_K": max(r["Tw"] for r in rows), "outside_area_m2": n_r * A_row, "mass_velocity_kg_m2s": G_max,
        "mean_beam_length_m": beam, "flue_gas_kg_s": m_gas,
        "rows": [{"row": r["row"], "gas_in_K": r["Tg_in"], "gas_out_K": r["Tg_out"], "process_K": r["Tp"], "tube_wall_K": r["Tw"],
                  "duty_W": r["Q"], "flux_W_m2": r["Q"] / A_row, "h_convection_W_m2K": r["h_c"], "h_radiation_W_m2K": r["h_r"],
                  "U_W_m2K": r["U"], "Reynolds": r["Re"], "gas_emissivity": r["eg"]} for r in rows],
        "evidence": {"energy_balance_error": abs(Q - Q_gas) / max(Q_gas, 1e-12), "verified_internal": abs(Q - Q_gas) <= 1e-9 * max(Q_gas, 1.0),
                     "validated_against_literature": False},
        "warnings": warnings,
        "assumptions": list(c.assumptions) + [
            "Tubos lisos. Convección exterior: Zukauskas (ht.conv_tube_bank.Nu_Zukauskas_Bejan) con la velocidad en el área "
            "mínima de paso y la corrección por número de filas del banco entero.",
            "Propiedades de transporte del gas: Cantera, datos de GRI-Mech 3.0, promedio de mezcla (SO2 contado como CO2).",
            "Radiación del gas entre tubos" + (": gas gris WSGG con longitud de haz 3.6 V/A del espacio entre tubos y recinto "
                                               "todo sumidero." if gas_radiation else " NO incluida (gas_radiation=False)."),
            "Fila por fila con temperaturas medias aritméticas; fluido de proceso en contracorriente global"
            + (" (isotérmico)." if isothermal else ", cp constante (capacity_rate_W_K)."),
            "Ancho del ducto = tubos por fila × paso transversal salvo que se dé duct_width_m."],
        "not_included": ["tubos aletados o con pernos", "radiación del hogar que escapa a las primeras filas (va como "
                         "shield_cold_plane_area_m2 en la sección radiante)", "pérdida por paredes del banco",
                         "caída de presión del gas y tiro", "validación contra un ejemplo publicado"],
        "method": {"name": "banco de tubos lisos fila por fila", "module": "packages/energy/fired_heater.py",
                   "property_data": CB.SOURCE},
    }
