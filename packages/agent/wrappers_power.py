"""Agent-facing wrappers of the power-plant tools (`packages.dwsim_client.tools_power`, bloque M27_N)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_power as TP


@beta_tool
def power_plant_model(plant: str = "laguna_verde", year: int | None = None, unit: str | None = None,
                      overrides: dict[str, object] | None = None, prices: dict[str, float] | None = None,
                      today: str | None = None) -> str:
    """Public-data model of a power station: today 'laguna_verde' (CFE, 2 BWR-5 units, 2317 MWt). BWR steam cycle with
    IAPWS-95 (state points, efficiency, heat rate, condenser and seawater), generation per unit and year vs IAEA PRIS,
    operating economics (CENACE price, NEI/COPAR costs; sourced suggestions), CO2 avoided, process_model for
    process_diagram and opportunities. Turbine efficiency is CALIBRATED to the published gross output. year 1990-2025;
    unit '1'|'2'|None; overrides e.g. {n_feedwater_heaters, condenser_ttd_K, seawater_inlet_K, heaters_out};
    prices {energia_usd_mwh|energia_mxn_mwh, fx_mxn_usd, combustible_usd_mwh, operacion_usd_mwh, capital_usd_mwh}."""
    return json.dumps(TP.power_plant_model(plant, year, unit, overrides, prices, today), default=float, ensure_ascii=False)


@beta_tool
def list_power_plants() -> str:
    """Which power stations have public data loaded for power_plant_model: units, ratings, licences, PRIS years, sources
    and the data that could not be found."""
    return json.dumps(TP.list_power_plants(), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [power_plant_model, list_power_plants]
