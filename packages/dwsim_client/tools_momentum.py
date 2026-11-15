"""Agent tools for momentum transfer in lines (bloque MG): adiabatic compressible gas flow (Fanno), gas-liquid flow in
inclined and vertical pipes, water hammer, and pump selection on a system. No DWSIM flowsheet needed. Engines:
`packages.transport.fanno`, `packages.transport.two_phase_inclined`, `packages.transport.water_hammer` and
`packages.transport.pumps`."""
from __future__ import annotations

from packages.transport import fanno as FA
from packages.transport import pumps as PU
from packages.transport import two_phase_inclined as TP
from packages.transport import water_hammer as WH
from packages.transport.hydraulics import STEEL_ROUGHNESS_M


def adiabatic_gas_pipe(P1_Pa: float, T1_K: float, inner_diameter_m: float, length_m: float, inlet_basis: str,
                       mass_flow_kg_s: float | None = None, P2_Pa: float | None = None, gamma: float | None = None,
                       molar_mass_kg_mol: float | None = None, viscosity_Pa_s: float | None = None,
                       compounds: dict | None = None, roughness_m: float = STEEL_ROUGHNESS_M,
                       darcy_friction_factor: float | None = None) -> dict:
    """Flujo de gas compresible ADIABÁTICO con fricción en un tubo de área constante (flujo de Fanno, gas ideal): estado de
    salida para un flujo dado, o flujo para una presión de salida dada; estrangulamiento (Mach 1 a la salida), longitud
    crítica L* y flujo máximo de la línea; y la misma línea resuelta ISOTÉRMICA (compressible_gas_line) para comparar.
    Usar en tramos cortos y rápidos o aislados (Mach > 0.3); en líneas largas sin aislar el isotérmico es más realista.

    P1_Pa, T1_K: presión absoluta y temperatura de entrada. inlet_basis: 'static' = P1 y T1 estáticas medidas en la línea
      a la entrada; 'stagnation' = condiciones de reposo del recipiente aguas arriba (entrada isentrópica sin pérdidas).
    inner_diameter_m, length_m: geometría del tubo recto (sin accesorios ni cambio de elevación).
    Dar exactamente uno: mass_flow_kg_s (→ P, T, Mach de salida) o P2_Pa (presión absoluta de salida → flujo másico).
    Gas: gamma (Cp/Cv) y molar_mass_kg_mol del usuario, O compounds {nombre: fracción molar} para leer Cp de gas ideal,
      masa molar y viscosidad de la base de propiedades (chemicals/thermo; mezcla por Herning-Zipperer, se declara).
    viscosity_Pa_s: viscosidad del gas a la entrada (para Re y fricción de Colebrook) salvo que se dé darcy_friction_factor.
    roughness_m: rugosidad absoluta (acero comercial por omisión, se declara). darcy_friction_factor: f de Darcy fijo
      (= 4·Fanning) en vez de Colebrook.

    Devuelve mass_flow_kg_s, inlet y outlet {P_Pa, T_K, Mach, velocity_m_s, density_kg_m3, stagnation_P_Pa,
    stagnation_T_K}, dP_Pa, fL_D, fLmax_D_inlet, critical_length_m, choked, feasible (False si el flujo pedido no pasa;
    entonces max_mass_flow_kg_s), friction_factor_darcy, Re, isothermal_comparison, evidence (ODE de las ecuaciones
    primitivas vs forma cerrada, conservación de T0), warnings, assumptions y not_included. Solo rama subsónica.
    Validación: tabla de Fanno γ = 1.4 (Purdue AAE) y ejemplo de Bar-Meir (LibreTexts)."""
    return FA.adiabatic_gas_pipe(P1_Pa, T1_K, inner_diameter_m, length_m, inlet_basis, mass_flow_kg_s, P2_Pa, gamma,
                                 molar_mass_kg_mol, viscosity_Pa_s, compounds, roughness_m, darcy_friction_factor)


def two_phase_pipe_inclined(mass_flow_kg_s: float, quality: float, liquid_density_kg_m3: float,
                            gas_density_kg_m3: float, liquid_viscosity_Pa_s: float, gas_viscosity_Pa_s: float,
                            surface_tension_N_m: float, inner_diameter_m: float, length_m: float, angle_deg: float,
                            inlet_pressure_Pa: float, roughness_m: float = STEEL_ROUGHNESS_M,
                            n_segments: int = 20) -> dict:
    """Flujo gas-líquido en tubo INCLINADO o VERTICAL: caída de presión separada en gravedad (densidad con holdup),
    fricción y aceleración, patrón de flujo y holdup de líquido a lo largo de la línea, por Beggs-Brill con inclinación
    (método recomendado) y por un modelo separado (fracción de vacío de Woldesemayat-Ghajar con inclinación + fricción de
    Friedel/Chisholm/Lockhart-Martinelli según Whalley + aceleración) para dar el intervalo entre métodos.

    mass_flow_kg_s: flujo total; quality: fracción MÁSICA de gas (0 < x < 1), constante (sin flasheo).
    Propiedades a la entrada: liquid_density_kg_m3, gas_density_kg_m3 (la del gas se escala con la presión, gas ideal
      isotérmico), liquid_viscosity_Pa_s, gas_viscosity_Pa_s, surface_tension_N_m.
    inner_diameter_m, length_m (longitud a lo largo del tubo), angle_deg sobre la horizontal (+ ascendente, − descendente,
      ±90 vertical), inlet_pressure_Pa absoluta. roughness_m (acero comercial por omisión). n_segments: tramos de la marcha.

    Devuelve P_out_Pa, dP_total_Pa, dP_gravity_Pa, dP_friction_Pa, dP_acceleration_Pa, flow_pattern_inlet/outlet
    (segregado, transición, intermitente, distribuido), flow_pattern_Taitel_Dukler_inlet (solo |ángulo| ≤ 10°),
    holdup_inlet/outlet, profile, separated_model {...}, method_spread_rel, evidence (gradiente contra la implementación
    independiente fluids.Beggs_Brill e independencia de segmentos), warnings, assumptions y not_included. Para mezclas con
    flasheo a lo largo usar la tubería de DWSIM (add_unit kind='pipe'). Validación: tubería de DWSIM (Beggs-Brill)."""
    return TP.two_phase_pipe_inclined(mass_flow_kg_s, quality, liquid_density_kg_m3, gas_density_kg_m3,
                                      liquid_viscosity_Pa_s, gas_viscosity_Pa_s, surface_tension_N_m, inner_diameter_m,
                                      length_m, angle_deg, inlet_pressure_Pa, roughness_m, n_segments)


def water_hammer(length_m: float, inner_diameter_m: float, density_kg_m3: float, initial_flow_m3_s: float,
                 upstream_pressure_Pa: float, downstream_pressure_Pa: float, closure_time_s: float | None = None,
                 closure_exponent: float | None = None, closure_curve: dict | None = None,
                 wave_speed_m_s: float | None = None, fluid_bulk_modulus_Pa: float | None = None,
                 pipe_elastic_modulus_Pa: float | None = None, wall_thickness_m: float | None = None,
                 restraint_factor: float | None = None, pipe_anchoring: str | None = None,
                 poisson_ratio: float | None = None, darcy_friction_factor: float | None = None,
                 viscosity_Pa_s: float | None = None, roughness_m: float | None = None,
                 vapor_pressure_Pa: float | None = None, elevation_change_m: float = 0.0, n_reaches: int = 40,
                 simulation_time_s: float | None = None) -> dict:
    """Golpe de ariete al cerrar una válvula en un sistema depósito-tubo-válvula: celeridad de la onda con la elasticidad
    del tubo (Korteweg), sobrepresión de Joukowsky ρ·a·V0, cierre rápido o lento frente a 2L/a, y el transitorio completo
    por el método de características con fricción: presión máxima y mínima en toda la línea (cuándo y dónde), historia de
    presión en la válvula y bandera de cavitación.

    length_m, inner_diameter_m, density_kg_m3 del líquido, initial_flow_m3_s (caudal estacionario antes del cierre).
    upstream_pressure_Pa: presión ABSOLUTA en la entrada del tubo desde el depósito (incluye la columna del depósito);
    downstream_pressure_Pa: presión absoluta a la que descarga la válvula. elevation_change_m: cota de la válvula menos la
      de la entrada (0 = horizontal, se declara).
    Cierre: closure_time_s y closure_exponent (apertura relativa τ = (1 − t/tc)^Em; Em = 1 lineal en τ; tc = 0
      instantáneo) O closure_curve {'time_s': [...], 'tau': [...]} con tau(0) = 1.
    Celeridad: wave_speed_m_s directa, O fluid_bulk_modulus_Pa (+ pipe_elastic_modulus_Pa, wall_thickness_m y
      restraint_factor c1 o pipe_anchoring 'free_longitudinal' | 'anchored_both_ends' | 'expansion_joints' con
      poisson_ratio; sin datos del tubo: tubo rígido).
    Fricción: darcy_friction_factor (0 = sin fricción) o viscosity_Pa_s y roughness_m (Colebrook).
    vapor_pressure_Pa: para la bandera de cavitación (sin ella no se evalúa). n_reaches: tramos de la malla MOC;
      simulation_time_s: por omisión tc + 10·(2L/a).

    Devuelve wave_speed_m_s, joukowsky_dP_Pa, two_L_over_a_s, period_4L_over_a_s, closure, P_initial_valve_Pa, P_max_Pa,
    P_min_Pa (con x y t), surge_above_initial_Pa, cavitation, envelope, valve_history, evidence (estado estacionario
    conservado, independencia de malla N vs 2N), warnings, assumptions y not_included (sin separación de columna)."""
    return WH.water_hammer(length_m, inner_diameter_m, density_kg_m3, initial_flow_m3_s, upstream_pressure_Pa,
                           downstream_pressure_Pa, closure_time_s, closure_exponent, closure_curve, wave_speed_m_s,
                           fluid_bulk_modulus_Pa, pipe_elastic_modulus_Pa, wall_thickness_m, restraint_factor,
                           pipe_anchoring, poisson_ratio, darcy_friction_factor, viscosity_Pa_s, roughness_m,
                           vapor_pressure_Pa, elevation_change_m, n_reaches, simulation_time_s)


def pump_selection(pump_curve: dict, speed_rpm: float, density_kg_m3: float, system: dict, curve_fit: str = "pchip",
                   stages: int = 1, efficiency: float | None = None, motor_efficiency: float | None = None,
                   allow_efficiency_estimate: bool = False, suction: dict | None = None,
                   vapor_pressure_Pa: float | None = None, new_speed_rpm: float | None = None,
                   impeller_diameter_m: float | None = None, new_impeller_diameter_m: float | None = None,
                   target_flow_m3_s: float | None = None) -> dict:
    """Bomba centrífuga sobre un sistema: punto de operación (cruce de la curva de la bomba con la del sistema), eficiencia,
    potencia al eje y del motor (tamaño NEMA), velocidad específica en el BEP y tipo de impulsor, leyes de afinidad (cambio
    de velocidad, recorte de impulsor, velocidad para un caudal objetivo) y NPSH disponible contra requerido.

    pump_curve: {'flow_m3_s': [...], 'head_m': [...], opcional 'efficiency': [...] (fracción), opcional 'NPSHr_m': [...]}
      del fabricante, a speed_rpm. density_kg_m3 del líquido.
    system: {'static_head_m', 'resistance_s2_m5'} (H = Hs + k·Q²) o {'static_head_m', 'flow_m3_s': [...], 'head_m': [...]}
      (tabla; por ejemplo de pipe_network). curve_fit: 'pchip' (interpola, no extrapola), 'quadratic' o 'parabola'
      (H = a + c·Q²) por mínimos cuadrados. stages: etapas iguales (n_s por etapa).
    Eficiencia: de la curva, o efficiency del fabricante en el punto; solo si allow_efficiency_estimate=True se usa la
      correlación publicada de Corripio (1982) para bomba y motor (ESTIMACIÓN, se declara). motor_efficiency opcional.
    suction (para NPSH): {'vessel_pressure_Pa' absoluta, 'liquid_level_above_pump_m', 'losses_m' o
      'loss_coefficient_s2_m5'} con vapor_pressure_Pa a la T de bombeo.
    Afinidad: new_speed_rpm; impeller_diameter_m + new_impeller_diameter_m; target_flow_m3_s → velocidad necesaria.

    Devuelve operating_point {flow_m3_s, head_m, efficiency, efficiency_source, hydraulic_power_W, shaft_power_W,
    NPSHr_m}, best_efficiency_point, flow_fraction_of_bep, specific_speed {Ns_US_rpm_gpm_ft, Ns_metric_rpm_m3s_m,
    omega_s_dimensionless, pump_type}, motor, npsh {NPSHa_m, margin_m, ratio}, affinity, evidence, warnings, assumptions y
    not_included. Validación: ejemplo 7.1-7.3 de Maurer, «Hydraulics and Water Resources» (SCU, CC BY-NC)."""
    return PU.pump_selection(pump_curve, speed_rpm, density_kg_m3, system, curve_fit, stages, efficiency,
                             motor_efficiency, allow_efficiency_estimate, suction, vapor_pressure_Pa, new_speed_rpm,
                             impeller_diameter_m, new_impeller_diameter_m, target_flow_m3_s)


MOMENTUM_TOOLS = [adiabatic_gas_pipe, two_phase_pipe_inclined, water_hammer, pump_selection]
