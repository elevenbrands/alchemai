"""Agent-facing wrappers of the reference-price and design-basis tools (`packages.dwsim_client.tools_reference_prices`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_reference_prices as TRP


@beta_tool
def reference_prices(items: list[str], region: str = "MX", currency: str = "USD", exchange_rates: dict[str, float] | None = None,
                     today: str | None = None) -> str:
    """PUBLISHED reference prices (never quotes) to propose a first economic picture when the user gave no prices:
    raw materials, products, water, diesel, wages (e.g. 'castor oil', 'methanol', 'ethanol', 'biodiesel', 'sugarcane',
    'corn', 'process water', 'diesel', 'natural gas'). region 'MX-SIN' | 'MX' | 'US' | 'WORLD' with fallback in that
    order (reported). currency 'USD' | 'MXN': MXN↔USD uses the bank's dated Banxico FIX unless exchange_rates
    {'MXN': <currency per 1 MXN>} is given. Returns values {item: {found, value, unit, source, url, period, age_days,
    stale, value_converted, flag, warnings}}, missing (never filled), assumptions to show the user, evidence. Always
    present each value as an assumption the user can override; liquids quoted per gallon stay per m3."""
    return json.dumps(TRP.reference_prices(items, region, currency, exchange_rates, today), default=float, ensure_ascii=False)


@beta_tool
def design_basis_defaults(region: str = "MX", plant_type: str | None = None, today: str | None = None) -> str:
    """PUBLISHED design-basis, fiscal, financial and labour values to start an economic evaluation without asking for
    every number: Mexican ISR (LISR art. 9), PTU, LISR depreciation fractions, Banxico target rate, TIIE 28, CETES,
    FIX MXN/USD, equity risk premium, NREL conventions (on-stream hours, plant life, construction schedule, working
    capital, start-up, financing), zafra length, location factor, wages. plant_type (free text, e.g. 'biodiesel',
    'bioetanol de caña') only PROPOSES the depreciation fraction by keyword. Returns values {id: sourced value},
    by_kind, suggested_depreciation, not_found, assumptions, warnings, evidence. Every value is a flagged assumption."""
    return json.dumps(TRP.design_basis_defaults(region, plant_type, today), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [reference_prices, design_basis_defaults]
