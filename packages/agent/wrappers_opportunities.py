"""Agent-facing wrappers of the opportunity scanner tools (`packages.dwsim_client.tools_opportunities`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_opportunities as TOP


@beta_tool
def scan_opportunities(process_model: dict, prices: dict | None = None, region: str = "MX",
                       plant_data: dict | None = None, today: str | None = None) -> str:
    """Ranked improvement opportunities of a ProcessModel (units/streams ids, for the diagram), each with value_usd_y and its
    low/high range, basis, assumptions, data_to_confirm, confidence and evidence; value None plus `missing` when an input
    is absent. Rules: pinch heat integration, fired heater/boiler excess air and efficiency, steam system (plant_data.
    steam_system = steam_system_diagnostic args), DOE bandwidth energy intensity, FCC conversion, flaring/CO2, pumps vs DOE
    minimum efficiency, plant_economics sensitivity. prices {fuel_usd_GJ, electricity_usd_kWh, carbon_usd_tCO2,
    cooling_usd_GJ, usd_per_currency}; missing gas/electricity come from the sourced bank, flagged. total_value_usd_y counts
    non-overlapping items only. Estimates, never measured savings."""
    return json.dumps(TOP.scan_opportunities(process_model, prices, region, plant_data, today), default=float,
                      ensure_ascii=False)


@beta_tool
def opportunities_from_plant_data(table: list[dict], units: list[dict] | None = None, prices: dict | None = None,
                                  region: str = "MX", today: str | None = None, plant_data: dict | None = None) -> str:
    """Pasted plant readings [{unit_id, variable, value, uom, period, basis?}] (historian/Excel; Spanish or English variable
    names; unit_id 'site' for hours and prices) -> which opportunity rules can run, which cannot and what each is missing,
    and the result of scan_opportunities on them. units: optional [{id, type, label_es}] (else the type is inferred from
    the variables and flagged). Rejected readings come back with the reason and the valid units of measure."""
    return json.dumps(TOP.opportunities_from_plant_data(table, units, prices, region, today, plant_data), default=float,
                      ensure_ascii=False)


AGENT_WRAPPERS = [scan_opportunities, opportunities_from_plant_data]
