"""Agent tools for the CAPITAL of utilities (bloque M26-U): cooling tower, steam boiler with optional turbogenerator and
wastewater treatment. No DWSIM flowsheet needed. Engine: `packages.costing.utilities_capital`. The operating cost of the
same utilities is `utility_costs` (tools_utilities, M17); these tools give the design basis and the equipment items that
`capital_investment` (tools_economics) takes."""
from __future__ import annotations

from packages.costing import utilities_capital as UC


def cooling_tower_design(duty_W: float, supply_T_K: float, return_T_K: float, wet_bulb_T_K: float,
                         cycles_of_concentration: float, dry_bulb_T_K: float | None = None,
                         liquid_to_gas_ratio: float | None = None, air_outlet_T_K: float | None = None,
                         drift_fraction: float | None = None, pressure_Pa: float | None = None,
                         pump_head_m: float | None = None, pump_efficiency: float | None = None,
                         fan_static_pressure_Pa: float | None = None, fan_efficiency: float | None = None,
                         year: int = 2023, cepci: float | None = None) -> dict:
    """Torre de enfriamiento de tiro mecánico: circulación, evaporación, arrastre, purga, repuesto, potencia de ventilador y
    de bombas, y capital comprado e instalado. Balance de agua y energía agua-aire húmedo (IAPWS-95 y HumidAirProp de
    CoolProp) con el aire saliendo saturado.

    Datos tuyos, obligatorios: duty_W (calor a rechazar, W), supply_T_K (agua fría hacia la planta), return_T_K (agua
      caliente de regreso), wet_bulb_T_K (bulbo húmedo de diseño del sitio; el agua fría debe quedar por encima),
      cycles_of_concentration (repuesto ÷ (purga + arrastre), > 1).
    Opcionales del sitio y del diseño: dry_bulb_T_K (si falta, el aire entra saturado y se avisa: con aire seco la
      evaporación sube), pressure_Pa (atmosférica; si falta, nivel del mar), liquid_to_gas_ratio (kg agua/kg aire seco) O
      air_outlet_T_K: con uno de los dos hay flujo de aire de diseño, temperatura de salida y número de Merkel (KaV/L); sin
      ellos la evaporación se evalúa en el límite de aire mínimo y se dice. pump_head_m + pump_efficiency y
      fan_static_pressure_Pa + fan_efficiency calculan las potencias por física.
    Sugeridos con fuente si no los das (vuelven en suggested_inputs y assumptions, puedes sustituirlos): drift_fraction
      (0.005 % de la circulación) y las potencias de ventilador y bomba por t/h de circulación, de Humbird et al. 2011 (NREL).
    year / cepci: año de los costos (tabla CEPCI 2015-2023) o el valor del índice para otro año.

    Devuelve circulation_kg_s y _m3_s, range_K, approach_K, evaporation_kg_s, drift_kg_s, blowdown_kg_s, makeup_kg_s,
    evaporation_bounds_kg_s, air {dry_air_kg_s, liquid_to_gas_ratio, maximum_liquid_to_gas_ratio, inlet, outlet,
    merkel_number}, fan_power_W, pump_power_W (con su base), capital {items listos para capital_investment (torre y bombas,
    con F_BM y base), purchase_cost_usd, installed_cost_usd, selected_tower_estimate, tower_estimates (BioSTEAM = cotización
    de Humbird 2011; lista de equipo de Humbird; Aden 2002; tabla DOE/NETL 2002 de 150-6,000 gpm; Smith 2005) y su
    desviación}, evidence (cierre de los balances), warnings, assumptions, not_included.
    Regla de selección del capital: una tabla publicada que contiene el tamaño gana a una cotización extrapolada.
    Validación: balance de aire y evaporación de la torre publicada por Humbird 2011 (−0.9 a −2.5 %). Capital, contraste
    entre fuentes independientes: torres armadas en campo (cotizaciones NREL de 1998 y 2010) −21 a +16 % en comprado y −7 a
    +3 % en instalado; en 150-6,000 gpm esas cotizaciones extrapoladas se apartan de la tabla DOE/NETL de −76 a +117 % en
    instalado (y 2 a 8 veces en comprado, por alcance), por eso ahí manda la tabla."""
    return UC.cooling_tower_design(duty_W, supply_T_K, return_T_K, wet_bulb_T_K, cycles_of_concentration, dry_bulb_T_K,
                                   liquid_to_gas_ratio, air_outlet_T_K, drift_fraction, pressure_Pa, pump_head_m,
                                   pump_efficiency, fan_static_pressure_Pa, fan_efficiency, year, cepci)


def steam_boiler_design(steam_demand: list[dict], feedwater_T_K: float, fuel: dict, boiler: dict | None = None,
                        combustion: dict | None = None, turbogenerator: dict | None = None, year: int = 2023,
                        cepci: float | None = None) -> dict:
    """Caldera de vapor (paquete, de campo o de biomasa) y turbogenerador opcional a partir de la demanda de vapor por nivel:
    carga térmica, agua de alimentación y purga, combustible, potencia eléctrica y capital. Propiedades IAPWS-95.

    steam_demand: [{name, pressure_Pa (absoluta), flow_kg_s}] que consume la planta. feedwater_T_K: agua de alimentación.
    fuel: {'type': 'natural_gas' | 'fuel_oil' | 'biomass' (bagazo incluido)} más el poder calorífico de TU combustible como
      se quema: 'LHV_J_kg' y/o 'HHV_J_kg'; o para biomasa 'HHV_dry_J_kg' + 'moisture_mass_fraction' (base húmeda) [+
      'hydrogen_dry_mass_fraction' si la eficiencia va en base LHV]. No hace falta si das combustion.
    boiler: {'efficiency' + 'efficiency_basis' ('HHV'|'LHV') tuyas; sin turbina 'superheat_K' (0 = saturado) o 'steam_T_K' y
      opcional 'pressure_Pa' (por defecto el nivel más alto; los menores salen por válvula reductora); 'blowdown_fraction'
      (de la alimentación; si falta se sugiere el 3 % del vapor de Humbird 2011 y se dice)}.
    combustion (en vez de boiler.efficiency): los argumentos de combustion_efficiency {fuel, air, pressure_Pa,
      excess_air_fraction o flue_O2, stack_T_K, setting_loss_fraction, setting_loss_basis}; la eficiencia y el combustible
      salen del método de pérdidas.
    turbogenerator (opcional): {'inlet_pressure_Pa', 'inlet_T_K', 'isentropic_efficiency' (tuya; la EPA publica 52.5 % a
      500 kW, 61.2 % a 3 MW y 78 % a 15 MW en contrapresión), 'mode': 'back_pressure' (todo el vapor sale a los niveles) |
      'condensing' (+ 'condensing_flow_kg_s' y 'condenser_pressure_Pa'), 'generator_efficiency' (si falta se sugiere 0.94,
      EPA 2015)}. La caldera genera entonces a las condiciones de admisión y cada nivel es una extracción.
    year / cepci: año de los costos (tabla CEPCI 2015-2023) o el valor del índice.

    Devuelve steam_generation_kg_s, absorbed_duty_W, feedwater_kg_s, blowdown_kg_s, steam_levels (T, calidad o
    sobrecalentamiento con que llega cada nivel), fuel {fuel_kg_s, heat_release_LHV_W/HHV_W, efficiency_*}, turbogenerator
    {shaft_power_W, electric_power_W, sections, condenser_duty_W, capital}, capital {items para capital_investment,
    purchase_cost_usd, installed_cost_usd, selected_estimate, estimates, purchase_deviation_vs_selected}, evidence, warnings,
    assumptions, not_included.
    Capital: gas o líquido = tabla DOE/NETL 2002 de calderas paquete (10,000-300,000 lb/h, 250 psig) con Smith 2005 de
    contraste; biomasa = BioSTEAM (cotización B&W de Humbird 2011, con limpieza de gases) con la tabla de la EPA 2007 de
    contraste; turbina de contrapresión de 0.5-15 MW = tabla EPA 2015, si no BioSTEAM (Siemens, Humbird 2011).
    Validación: potencia y temperatura de escape de los tres turbogeneradores de la Tabla 4-2 de la EPA (+0.01 a +0.11 %,
    ≤ 0.2 K). El capital no se valida: se contrasta entre fuentes y se informa la dispersión."""
    return UC.steam_boiler_design(steam_demand, feedwater_T_K, fuel, boiler, combustion, turbogenerator, year, cepci)


def wastewater_treatment_cost(flow_m3_s: float, cod_kg_m3: float, configuration: str, cod_removal: dict,
                              include_reverse_osmosis: bool = False, include_brine_evaporator: bool = False,
                              density_kg_m3: float | None = None, electricity_price_usd_kWh: float | None = None,
                              operating_hours_per_year: float | None = None, aeration_kg_O2_per_kWh: float | None = None,
                              biogas_value_usd_GJ: float | None = None, year: int = 2023, cepci: float | None = None) -> dict:
    """Tratamiento convencional de aguas residuales (aerobio, o anaerobio + aerobio): balance de DQO por etapa, metano del
    biogás, demanda de oxígeno, capital instalado y electricidad. ORDEN DE MAGNITUD en capital: se escala la única lista de
    equipo publicada con desglose (NREL, Humbird 2011) y las dos plantas de NREL difieren entre sí por un factor de 2 a 5.

    flow_m3_s y cod_kg_m3 (DQO; 1 g/L = 1 kg/m³): de TU agua. configuration: 'anaerobic_aerobic' | 'aerobic'.
    cod_removal (tuyas, fracciones de la DQO que ENTRA a cada etapa): {'anaerobic', 'anaerobic_to_biomass', 'aerobic',
      'aerobic_to_biomass'} (las anaerobias solo en 'anaerobic_aerobic'). Valores PUBLICADOS que puedes pasar si los aceptas
      (Humbird 2011, agua de 64-87 g/L): 0.91, 0.05, 0.96, 0.22; Aden 2002 (16 g/L): 0.93 anaerobia y 0.90 aerobia.
    include_reverse_osmosis, include_brine_evaporator: añaden el reúso de agua de la planta de referencia (por defecto no).
    density_kg_m3: si falta, la del agua a 25 °C. aeration_kg_O2_per_kWh: eficiencia de TUS aireadores, para la potencia de
    aireación por física; sin ella se escala la potencia de la referencia con la DQO aerobia y se dice.
    electricity_price_usd_kWh + operating_hours_per_year (+ biogas_value_usd_GJ): costo anual de electricidad y crédito del
    biogás; sin ellos solo se informa la potencia. year / cepci: año de los costos o el valor del índice.

    Devuelve COD_load_kg_s, stages {anaerobic, aerobic: DQO que entra, removida, a metano, a lodo, que sale}, biogas
    {methane_kg_s, methane_Nm3_s, energy_LHV_W}, oxygen_demand_kg_s, effluent_COD_kg_m3, overall_COD_removal, capital {items
    para capital_investment (costos ya instalados, F_BM = 1), installed_cost_usd, scope, estimates (este desglose;
    todo por caudal; BioSTEAM WastewaterSystemCost = toda el Área 600 de Humbird; Aden 2002) e
    installed_deviation_vs_selected}, power, operating, reference_plant, evidence (cierre de DQO), warnings, assumptions,
    not_included. Los reactores biológicos se escalan por carga de DQO y el resto por caudal (criterio del módulo).
    Validación: metano estequiométrico 0.3502 Nm³/kg DQO (350 L/kg en Aden 2002); metano y energía del biogás de Humbird
    2011 (+3.9 % y +5 %); remoción total 99.6 %. El capital solo reproduce su fuente. El sistema convencional de BioSTEAM no
    se simula: pide la composición química del agua y su costo es la misma ley de potencia por caudal."""
    return UC.wastewater_treatment_cost(flow_m3_s, cod_kg_m3, configuration, cod_removal, include_reverse_osmosis,
                                        include_brine_evaporator, density_kg_m3, electricity_price_usd_kWh,
                                        operating_hours_per_year, aeration_kg_O2_per_kWh, biogas_value_usd_GJ, year, cepci)


UTILITIES_CAPITAL_TOOLS = [cooling_tower_design, steam_boiler_design, wastewater_treatment_cost]
