"""Agent-facing wrappers of the whole-plant block M26_B (same names and order as PLANTS_TOOLS)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_plants as TPL


@beta_tool
def reference_biorefinery(plant: str, feed_flow_kg_h: float | None = None, operating_hours_y: float | None = None,
                          prices: dict[str, float] | None = None, tea: dict[str, object] | None = None,
                          year: int | None = None, cepci: float | None = None) -> str:
    """Published whole-plant model solved with BioSTEAM: plant = sugarcane_ethanol | corn_ethanol | cornstover_ethanol.
    No other argument = published baseline. Optional overrides: feed_flow_kg_h (wet feed, rescales and re-costs the plant),
    operating_hours_y, prices {stream ID: USD/kg, electricity: USD/kWh, steam, natural_gas}, tea {discount_rate,
    income_tax, plant_life_years, depreciation, lang_factor, construction_schedule, working_capital_over_fci,
    labor_cost_usd_y, ...}, year (2015-2023) or cepci for the equipment cost year. Returns design_basis (origin of every
    value), streams, sections, utilities, equipment (shape of capital_investment), capital (FCI, TCI), operating_cost,
    economics (NPV, IRR, minimum selling price), published_benchmark (deviation from the paper), evidence. Model prices
    are SUGGESTIONS: tell the user which were used. First call takes 20-60 s."""
    return json.dumps(TPL.reference_biorefinery(plant, feed_flow_kg_h, operating_hours_y, prices, tea, year, cepci),
                      default=float, ensure_ascii=False)


@beta_tool
def biodiesel_plant(oil_feed_kg_h: float | None = None, oil: str | None = None, ffa_mass_fraction: float | None = None,
                    water_mass_fraction: float | None = None, operating_hours_y: float | None = None,
                    prices: dict[str, float] | None = None, tea: dict[str, object] | None = None,
                    conversion_per_stage: float | None = None, methanol_excess: float | None = None,
                    catalyst_mass_fraction_of_oil: float | None = None, oil_temperature_K: float = 298.15,
                    year: int | None = None, cepci: float | None = None) -> str:
    """Vegetable oil -> biodiesel + 80 % crude glycerol by alkali transesterification (BioSTEAM's published section).
    REQUIRED user data: oil_feed_kg_h, oil (triolein | tripalmitin | castor = declared pseudo-triglyceride with borrowed
    triolein properties), ffa_mass_fraction (refuses above 0.01: no acid pretreatment model), water_mass_fraction,
    operating_hours_y. Optional: prices {oil, biodiesel, crude_glycerol, methanol, catalyst, electricity, steam, ...}, tea,
    conversion_per_stage, methanol_excess, catalyst_mass_fraction_of_oil, year | cepci. Oil price is never suggested:
    without prices['oil'] only the processing margin is returned. Other prices and TEA parameters are sourced suggestions
    flagged in design_basis. Returns yields, oil_model and the same blocks as reference_biorefinery. For castor oil read
    warnings and not_included (real viscosity and glycerol separation are not modelled)."""
    return json.dumps(TPL.biodiesel_plant(oil_feed_kg_h, oil, ffa_mass_fraction, water_mass_fraction, operating_hours_y,
                                          prices, tea, conversion_per_stage, methanol_excess,
                                          catalyst_mass_fraction_of_oil, oil_temperature_K, year, cepci),
                      default=float, ensure_ascii=False)


@beta_tool
def list_reference_plants() -> str:
    """Catalogue (no simulation) of the published whole-plant models reference_biorefinery can run, with their
    citations, the published values each is compared against, the accepted tea= and prices= keys, and the oils
    biodiesel_plant accepts. Call it first when unsure which plant or key to use."""
    return json.dumps(TPL.list_reference_plants(), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [reference_biorefinery, biodiesel_plant, list_reference_plants]
