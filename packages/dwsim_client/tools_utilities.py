"""Agent tools for utility costs by region (bloque M17). No DWSIM flowsheet needed. Engine: `packages.costing.utilities`.

`utility_costs` turns the prices a plant pays (fuel, electricity, water) into the cost of each utility level and returns the
`prices` dict that annual_cost / flowsheet_cost / pinch tools consume; `list_utility_price_references` lists the bank of
published, dated reference prices, which is used only when the user names one."""
from __future__ import annotations

from packages.costing import utilities as U


def utility_costs(currency: str, prices: dict | None = None, reference: str | None = None, steam: list[dict] | None = None,
                  fired_heat: dict | None = None, cooling_water: dict | None = None, refrigeration: list[dict] | None = None,
                  electricity: dict | None = None, hours_per_year: float | None = None, exchange_rates: dict | None = None,
                  annual_cost_levels: dict | None = None, ulrich_vasudevan: dict | None = None,
                  today: str | None = None) -> dict:
    """Costo de los servicios de una planta (vapor por nivel de presión, calor a fuego directo, agua de enfriamiento de torre,
    refrigeración por nivel de temperatura, electricidad) a partir de los precios que la planta paga. Balances de energía y de
    agua con IAPWS-95; ninguna eficiencia ni precio se supone.

    currency: moneda de los costos ('USD', 'MXN', 'EUR'...).
    prices (los del usuario, siempre ganan): fuel_per_GJ + fuel_basis ('HHV'|'LHV'), electricity_per_kWh (o
      electricity_time_of_use_per_kWh {periodo: precio}), demand_charge_per_kW_month (opcional), boiler_makeup_water_per_m3,
      cooling_makeup_water_per_m3.
    reference: id de un precio PUBLICADO del banco (list_utility_price_references), por ejemplo 'US-EIA-2026-08'. Solo se usa
      si se nombra; el resultado lo dice primero en warnings («precio de referencia publicado, no tu contrato», con fuente,
      periodo y antigüedad). Sin prices ni reference la herramienta se niega y lista las referencias. El agua nunca viene del banco.
    steam: [{name, pressure_Pa (absoluta), superheat_K (0 = saturado), boiler_efficiency, feedwater_T_K,
      condensate_return_fraction, blowdown_fraction (de la alimentación), makeup_water_T_K, feedwater_heating: 'not_charged'
      (costo de generación tipo DOE) | 'deaerator_steam' (necesita condensate_T_K), auxiliary_power_kWh_per_tonne (opcional)}].
    fired_heat: {furnace_efficiency}.
    cooling_water: {supply_T_K, return_T_K, pump_head_m, pump_efficiency, evaporation_latent_fraction, drift_fraction,
      cycles_of_concentration; opcionales fan_power_kWh_per_m3, treatment_cost_per_m3_makeup}.
    refrigeration: [{name, evaporator_T_K, condenser_T_K, second_law_efficiency (fracción del COP de Carnot); opcional
      heat_rejection: precio por GJ rechazado o 'cooling_water'}].
    electricity: {} para pedir solo el precio; {time_of_use_fractions: {periodo: fracción de la energía}} con precios por
      periodo; {load_factor} + hours_per_year reparte el cargo por demanda.
    exchange_rates: {'MXN': <unidades de currency por 1 MXN>}; nunca hay tipo de cambio por defecto. Con currency ≠ USD, da
      {'USD': <currency por 1 USD>} para obtener `prices` en USD.
    annual_cost_levels: {'heating': <name del nivel de vapor | 'fired_heat'>, 'cooling': <'cooling_water' | name de refrigeración>}
      cuando hay más de un nivel.
    ulrich_vasudevan (segunda ruta, opcional, USD): {cepci, basis: 'process_module'|'grass_roots', boiler_fuel_price_usd_GJ,
      power_fuel_price_usd_GJ, steam_capacity_kg_s, cooling_water_capacity_m3_s, refrigeration_capacity_kW}. Es un precio que
      incluye capital y mano de obra de la planta de servicios (Chem. Eng., abril 2006, Tabla 1), no la misma cantidad.
    today: fecha ISO para calcular la antigüedad de la referencia (por defecto, hoy).

    Devuelve prices {electricity_usd_kWh, heating_usd_GJ, cooling_usd_GJ} listo para annual_cost / flowsheet_cost, el detalle
    por nivel con su desglose (combustible, agua, electricidad), price_sources (qué precio vino de dónde), warnings,
    assumptions, not_included (capital y mantenimiento de los servicios no entran) y los datos de entrada."""
    return U.utility_costs(currency, prices, reference, steam, fired_heat, cooling_water, refrigeration, electricity,
                           hours_per_year, exchange_rates, annual_cost_levels, ulrich_vasudevan, today)


def list_utility_price_references(region: str | None = None) -> dict:
    """Banco de precios industriales PUBLICADOS de gas natural y electricidad por región (EIA de EE. UU., Eurostat, tarifa GDMTH
    de CFE), cada uno con valor, unidad, periodo, fuente, URL, fecha de consulta y advertencias. region filtra por prefijo ('US',
    'US-TX', 'EU27', 'DE', 'MX'). Son referencias, no contratos: utility_costs solo las usa si se nombra una en reference=.
    También dice qué no está en el banco y por qué (por ejemplo el gas natural de México)."""
    return U.list_references(region)


UTILITIES_TOOLS = [utility_costs, list_utility_price_references]
