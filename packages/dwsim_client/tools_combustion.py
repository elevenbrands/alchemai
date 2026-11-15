"""Agent tools for boilers and fired heaters (bloque M18). No DWSIM flowsheet needed. Engines: `packages.energy.combustion`
(combustion and heat-loss efficiency) and `packages.energy.fired_heater` (radiant section, convection bank)."""
from __future__ import annotations

from packages.energy import combustion as CB
from packages.energy import fired_heater as FH


def combustion_efficiency(fuel: dict, air: dict, pressure_Pa: float, excess_air_fraction: float | None = None,
                          flue_O2: dict | None = None, flue_CO_dry_mole_fraction: float | None = None,
                          stack_T_K: float | None = None, setting_loss_fraction: float | None = None,
                          setting_loss_basis: str | None = None, absorbed_duty_W: float | None = None,
                          fuel_flow_kg_s: float | None = None, fuel_price_basis: str | None = None) -> dict:
    """Combustión y eficiencia de una caldera u horno por el método de pérdidas: aire estequiométrico y real, gases de
    combustión (base húmeda y seca) por kg de combustible, poder calorífico superior e inferior, punto de rocío del agua,
    temperatura adiabática de llama, eficiencia en base HHV y LHV con el desglose de pérdidas, y combustible y aire para una
    carga. Entalpías de los polinomios NASA (Cantera), no cp constantes. Ningún dato de operación se supone.

    fuel: gas por composición molar {'type': 'gas', 'composition': {'CH4': 0.95, 'C2H6': 0.03, 'N2': 0.02}, 'T_K': 298.15}
      (especies: CH4, C2H6, C3H8, n-C4H10, i-C4H10, n-C5H12, i-C5H12, H2, CO, H2S, C2H4, C3H6, CO2, N2, O2, H2O, Ar, He; su
      HHV y LHV se calculan), o líquido/sólido por análisis elemental {'type': 'ultimate', 'mass_fractions': {'C', 'H', 'O',
      'N', 'S', 'ash', 'moisture'} (como se quema, suman 1), 'HHV_J_kg' o 'LHV_J_kg' (medido, uno solo), opcionales 'T_K' y
      'cp_J_kgK'}.
    air: {'T_K', y 'humidity_ratio_kg_kg' (kg agua/kg aire seco) o 'relative_humidity' (0-1); opcional
      'dry_air_mole_fractions'}. pressure_Pa: presión absoluta del sitio.
    excess_air_fraction (0.15 = 15 %) O flue_O2 {'mole_fraction': 0.03, 'basis': 'dry'|'wet'} medido en chimenea (uno solo; el
      exceso sale del balance exacto). flue_CO_dry_mole_fraction: CO medido en base seca (100 ppm = 1e-4), opcional.
    stack_T_K: temperatura de chimenea; con ella se calcula la eficiencia y entonces setting_loss_fraction (pérdida por
      paredes/radiación de la envolvente, fracción del poder calorífico; 0 si es nula) y setting_loss_basis ('HHV'|'LHV') son
      obligatorios: no hay valor típico.
    absorbed_duty_W O fuel_flow_kg_s: da combustible, aire y gases para esa carga, o la carga para ese combustible.
    fuel_price_basis ('HHV'|'LHV'): base del precio del combustible; devuelve la eficiencia que hay que pasar a utility_costs
      como boiler_efficiency / furnace_efficiency y cuánto se desviaría el costo con la base equivocada.

    Devuelve fuel, air, flue_gas (con water_dew_point_K), adiabatic_flame (equilibrio con disociación por la ruta de Cantera
    cuando el gas está en GRI-Mech 3.0; combustión completa siempre), efficiency {LHV, HHV: efficiency, pérdidas y créditos
    como fracción del poder calorífico, efficiency_on_total_input}, for_utility_costs, firing, evidence (cierre de masa y de
    energía), warnings, assumptions y not_included. Validación: tablas de pérdidas por chimenea del DOE (BENCHMARK.md, M18)."""
    return CB.combustion_efficiency(fuel, air, pressure_Pa, excess_air_fraction, flue_O2, flue_CO_dry_mole_fraction, stack_T_K,
                                    setting_loss_fraction, setting_loss_basis, absorbed_duty_W, fuel_flow_kg_s, fuel_price_basis)


def fired_heater_radiant(mode: str, fuel: dict | None = None, air: dict | None = None, pressure_Pa: float | None = None,
                         fuel_flow_kg_s: float | None = None, geometry: dict | None = None,
                         tube_emissivity: float | None = None, convective_coefficient_W_m2K: float | None = None,
                         setting_loss_fraction: float | None = None, setting_loss_basis: str | None = None,
                         excess_air_fraction: float | None = None, flue_O2: dict | None = None,
                         flue_CO_dry_mole_fraction: float | None = None, tube_wall_T_K: float | None = None,
                         process: dict | None = None, gas_emissivity: float | None = None,
                         exchange_factor: float | None = None, required_radiant_duty_W: float | None = None,
                         allowable_average_flux_W_m2: float | None = None) -> dict:
    """Sección radiante de un horno o caldera con una zona de gas bien mezclada (idea de Lobo-Evans / Hottel).

    mode='rating': con el fuego y la geometría da la temperatura de pared divisoria (bridgewall = temperatura efectiva del
      gas), el calor absorbido por los tubos (radiación + convección), el flujo medio y la fracción absorbida. Necesita fuel,
      air, pressure_Pa y excess_air_fraction o flue_O2 (igual que combustion_efficiency), fuel_flow_kg_s, geometry
      {tube_outside_diameter_m, tube_pitch_m, tube_effective_length_m, n_tubes, envelope_area_m2 (área interior total del
      hogar), volume_m3 (o mean_beam_length_m); opcionales alpha (si no es una fila frente a refractario) y
      shield_cold_plane_area_m2}, tube_emissivity, convective_coefficient_W_m2K (convección en el hogar; 0 si no la
      consideras), setting_loss_fraction + setting_loss_basis, y tube_wall_T_K O process {bulk_T_K, inside_coefficient_W_m2K,
      tube_inside_diameter_m, wall_conductivity_W_mK, inside_fouling_m2K_W} (entonces estima la temperatura de pared).
      gas_emissivity o exchange_factor: solo si quieres imponer los tuyos (de cartas propias).
    mode='sizing': con required_radiant_duty_W y allowable_average_flux_W_m2 (el flujo medio admisible lo fija el usuario; no
      hay valor típico), tube_outside_diameter_m, tube_effective_length_m y tube_pitch_m dentro de geometry, da el área y el
      número de tubos; después se evalúa el hogar con mode='rating'.

    Emisividad del gas: suma ponderada de gases grises de Smith, Shen y Friedman (1982) para CO2 + H2O a 1 atm; longitud de
    haz 3.6 V/A. Validado contra cuatro hogares publicados: ejemplos 19.1 y 19.2 de Kern (−3.2 % y −1.7 % en carga, +33 K y
    +11 K), el hogar de caja del tutorial de Esteem Projects (−1.4 %, +10 a +13 K) y un cilíndrico vertical con hoja de datos
    API 560 de Heat Transfer Consult (pared divisoria −5 K frente a su Lobo-Evans). Banda a esperar: carga radiante ±5 %,
    temperatura de pared divisoria ±35 K, en hornos de una fila de tubos frente a refractario con llama no luminosa; con varios
    diámetros de tubo da alpha y un tubo equivalente (mismas áreas de tubo y de plano frío). La emisividad de Smith 1982
    concuerda con Leckner 1972 (−12 a +18 %); las cartas de Kern son el valor atípico. No hay flujo pico ni zonas. Devuelve
    evidence, warnings, assumptions y not_included."""
    if mode == "sizing":
        g = geometry or {}
        return FH.radiant_sizing(required_radiant_duty_W, allowable_average_flux_W_m2, g.get("tube_outside_diameter_m"),
                                 g.get("tube_effective_length_m"), g.get("tube_pitch_m"))
    if mode != "rating":
        raise ValueError("mode: 'rating' o 'sizing'")
    return FH.radiant_section(fuel, air, pressure_Pa, fuel_flow_kg_s, geometry, tube_emissivity, convective_coefficient_W_m2K,
                              setting_loss_fraction, setting_loss_basis, excess_air_fraction, flue_O2,
                              flue_CO_dry_mole_fraction, tube_wall_T_K, process, gas_emissivity, exchange_factor)


def fired_heater_convection(fuel: dict, air: dict, pressure_Pa: float, fuel_flow_kg_s: float, gas_inlet_T_K: float, bank: dict,
                            process: dict, excess_air_fraction: float | None = None, flue_O2: dict | None = None,
                            flue_CO_dry_mole_fraction: float | None = None, gas_radiation: bool = True) -> dict:
    """Sección de convección de un horno o caldera: gases de combustión cruzando un banco de tubos (lisos, aletados o con
    pernos), fila por fila, hasta la chimenea. Sirve para que la temperatura de chimenea sea un resultado: pásala luego a
    combustion_efficiency.

    fuel, air, pressure_Pa, excess_air_fraction o flue_O2: como en combustion_efficiency. fuel_flow_kg_s: combustible quemado.
    gas_inlet_T_K: temperatura del gas que entra al banco (bridgewall_T_K de fired_heater_radiant).
    bank: {tube_outside_diameter_m, tube_inside_diameter_m, tube_effective_length_m, tubes_per_row, n_rows,
      transverse_pitch_m, longitudinal_pitch_m, arrangement: 'staggered'|'inline', wall_conductivity_W_mK,
      outside_fouling_m2K_W, tube_emissivity; opcional duct_width_m y mean_beam_length_m; tube_type: 'bare' (por defecto),
      'solid_fin' o 'serrated_fin' (aletados: fin_height_m, fin_thickness_m, fins_per_m, fin_conductivity_W_mK; aleta
      segmentada además fin_segment_width_m) o 'studded' (con pernos: stud_length_m, stud_diameter_m, studs_per_plane,
      stud_planes_per_m (planos de pernos por metro de tubo), stud_conductivity_W_mK)}. Bancos con tipos de tubo mezclados
      no se modelan en una sola llamada: encadena una llamada por tramo.
    process: {inlet_T_K (entra por el lado frío del gas, contracorriente), capacity_rate_W_K (flujo × cp) o isothermal: true
      (ebullición), inside_coefficient_W_m2K, inside_fouling_m2K_W}.
    gas_radiation=False quita la radiación del gas entre tubos (cota inferior de la carga; sin efecto con pernos).

    Convección exterior: tubos lisos con Žukauskas 1972 y el arreglo declarado; aletados con la correlación ESCOA revisada
    (Ganapathy, apéndice E) y eficiencia de aleta; con pernos, la correlación del tutorial de Esteem Projects (la radiación del
    gas va dentro de ella); propiedades del gas de Cantera. Validado: tubos lisos, coeficiente ±10 % y carga de la sección
    ±12 % (Incropera 7.7, Çengel 7-7, Nelson ej. 68); aletados en línea, coeficiente y U ±5 % (Ganapathy E.1 y E.10,
    transcritos de una copia sin autorización verificada: pendiente cotejar); aletados en tresbolillo ±10 % (sin ejemplo con
    la versión revisada); con pernos NO validado (sin ejemplo resuelto publicado; en el límite sin pernos su coeficiente queda
    31 % bajo el tubo liso con radiación de este módulo: posible subestimación).
    Devuelve stack_T_K, duty_W, process_outlet_T_K, max_tube_wall_T_K, la tabla por fila, evidence, warnings, assumptions y
    not_included."""
    return FH.convection_section(fuel, air, pressure_Pa, fuel_flow_kg_s, gas_inlet_T_K, bank, process, excess_air_fraction,
                                 flue_O2, flue_CO_dry_mole_fraction, gas_radiation)


COMBUSTION_TOOLS = [combustion_efficiency, fired_heater_radiant, fired_heater_convection]
