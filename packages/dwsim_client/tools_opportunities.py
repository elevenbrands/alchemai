"""Agent tools for the opportunity scanner (bloque M27_O). Engine and evidence live in `packages.opportunities.scan`; the
benchmarks (each with source, URL, period and date read) in `packages/opportunities/benchmarks.json`."""
from __future__ import annotations

from packages.opportunities import scan as SC


def scan_opportunities(process_model: dict, prices: dict | None = None, region: str = "MX", plant_data=None,
                       today: str | None = None) -> dict:
    """Oportunidades de mejora y optimización de una planta, deterministas y explicables, cada una anclada a los ids de
    unidades y corrientes del ProcessModel (para marcarlas en el diagrama) y con su valor en USD/año, su rango, la base del
    cálculo, los supuestos, qué dato la confirmaría y su confianza. NUNCA es un ahorro medido: es una estimación contra una
    referencia publicada. Si falta un dato, la oportunidad sale con value_usd_y = None y la lista `missing`, o va a
    `not_assessed` con el motivo.

    process_model: el JSON común de M27 {name, kind 'refinery'|'plant_model'|'dwsim'|'described', units [{id, label_es, type
      (vocabulario de equipos), section, params, duties_W, metrics}], streams [{id, from, to, kind, flow_bpd, ...}], sections}.
      Claves que leen las reglas en params/metrics/duties_W de cada unidad (SI): throughput_bpd o production_bpd (o se suman
      las corrientes con flow_bpd que entran/salen), production_kg_h, energy_W (energía en sitio) o fuel_W + steam_W +
      electricity_W, absorbed_W (calor absorbido de un horno), efficiency (+ efficiency_basis 'HHV'|'LHV'), T_in_K y T_out_K
      (+ duty_W) de calentadores y enfriadores, hot_/cold_T_in_K/T_out_K de intercambiadores, combustion {fuel, air,
      pressure_Pa, flue_O2 {mole_fraction, basis} o excess_air_fraction, stack_T_K, setting_loss_fraction,
      setting_loss_basis} (como combustion_efficiency), bandwidth_process (alkylation, atmospheric_crude_distillation,
      catalytic_hydrocracking, catalytic_reforming, coking_visbreaking, fluid_catalytic_cracking, hydrotreating,
      isomerization, vacuum_crude_distillation; si no se da, se busca por palabra clave en id/etiqueta y se avisa),
      bandwidth_chemical (ammonia, methanol, ethylene, hydrogen, chlorine, ethanol, benzene, ...), conversion_pct,
      max_conversion_pct y conversion_value_usd_bbl_point (FCC), flow_m3_s, head_m, density_kg_m3, shaft_power_W o
      electric_power_W + motor_efficiency, speed_rpm, stages, pump_class (bombas), flare_kg_h, flare_HV_J_kg,
      flare_carbon_mass_fraction, flare_recoverable_fraction (quemador). process_model.economics: un resultado de
      plant_economics (su sensibilidad se vuelve «confirmar / negociar»). hours_per_year o basis.on_stream_hours.
    prices: {fuel_usd_GJ (base PCS), electricity_usd_kWh, carbon_usd_tCO2, cooling_usd_GJ, usd_per_currency {'MXN': USD por
      MXN}}. Lo que no des se toma del banco de precios publicados (gas natural, electricidad) y se marca como supuesto; el
      carbono y el enfriamiento nunca se suponen.
    region: región del banco de precios ('MX', 'US'...).
    plant_data: datos medidos que mandan sobre el modelo: lista de lecturas (como opportunities_from_plant_data) o dict
      {readings, hours_per_year, heat_integration {streams [como pinch_analysis + unit_id], dT_min_K,
      current_hot_utility_W, current_cold_utility_W, heater_efficiency}, combustion {unit_id: {...}}, steam_system
      {unit_id, inputs: los argumentos de steam_system_diagnostic} o lista de ellos, plant_economics}.
    today: fecha ISO (por defecto hoy).

    Reglas: (a) integración térmica con pinch_analysis (ΔTmin del usuario o el rango típico publicado de NRCan);
    (b) hornos y calderas: combustion_efficiency al O2 / exceso de aire publicado y brecha a la eficiencia de hornos de la
    guía ENERGY STAR; (c) vapor con steam_system_diagnostic; (d) intensidad energética vs estudios Bandwidth del DOE
    (refinación por proceso, químicos por producto); (e) conversión de FCC vs rango publicado; (f) quemador y CO2;
    (g) bombas vs eficiencia mínima DOE; (h) sensibilidad de plant_economics.
    Devuelve opportunities (ordenadas: id, rule, unit_ids, stream_ids, category, title_es, explanation_es, value_usd_y,
    value_low_usd_y, value_high_usd_y, basis_es, assumptions, evidence, confidence, data_to_confirm, missing, rank,
    overlap_group), total_value_usd_y (+ low/high: solo las que no se traslapan; total_method_es explica cómo),
    overlap_groups, naive_sum_usd_y, not_assessed, rule_status, prices_used, hours_per_year, warnings, assumptions,
    not_included, method y evidence. Presenta los valores como estimaciones y los precios de referencia como supuestos."""
    return SC.scan_opportunities(process_model, prices, region, plant_data, today)


def opportunities_from_plant_data(table: list[dict], units: list[dict] | None = None, prices: dict | None = None,
                                  region: str = "MX", today: str | None = None, plant_data: dict | None = None) -> dict:
    """«Meto mis variables y encuentras oportunidades»: recibe las lecturas que el ingeniero pega de su historiador o de
    Excel y dice qué reglas pueden correr con ellas, cuáles no y qué les falta, y el resultado de las que sí.

    table: lista de renglones {unit_id, variable, value, uom, period, basis?}. unit_id 'site'/'planta' para datos de toda la
      planta (horas, precios). Variables (español o inglés): carga/throughput (bpd, m3/d, m3/h), producción (bpd, kg/h,
      t/h), energía, combustible, vapor, electricidad, calor absorbido (W, kW, MW, GJ/h, MMBtu/h, Gcal/h), eficiencia (%,
      base PCS/PCI en uom o basis), O2 en humos ('% dry' o '% base seca'; sin base se rechaza), temperatura de chimenea
      (°C, K, °F), flujo (m3/h, gpm, L/s), carga dinámica (m, ft), densidad, potencia al eje, potencia eléctrica,
      eficiencia del motor, rpm, etapas, flujo a quemador (kg/h), poder calorífico del gas de quemador (MJ/kg), fracción
      de carbono, fracción recuperable, conversión (% peso), conversión máxima, valor por punto de conversión
      (USD/bbl/punto), horas al año, precio del combustible (USD/GJ, USD/MMBtu), de la electricidad (USD/kWh, USD/MWh),
      del carbono (USD/t). Lecturas repetidas se promedian y se avisa si la dispersión pasa de 10 %.
    units: opcional [{id, type, label_es, section, params}] para decir qué equipo es cada unit_id; sin él el tipo se infiere
      de las variables (flujo + carga → bomba; O2 o chimenea → horno; quemador) y se avisa.
    prices, region, today: como en scan_opportunities. plant_data: opcional, las demás llaves de plant_data
      (combustion, steam_system, heat_integration).

    Devuelve readings_accepted, readings_rejected (con el motivo y las unidades válidas), units_inferred, rules_can_run,
    rules_cannot_run (con missing) y todo el resultado de scan_opportunities."""
    return SC.opportunities_from_plant_data(table, units, prices, region, today, plant_data)


OPPORTUNITIES_TOOLS = [scan_opportunities, opportunities_from_plant_data]
