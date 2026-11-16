"""Agent-facing wrappers of the whole-plant economics tools (`packages.dwsim_client.tools_plant_economics`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_plant_economics as TPE


@beta_tool
def plant_economics(equipment: list[dict], raw_materials: list[dict], products: list[dict], utilities: list[dict] | None = None,
                    labour: dict | None = None, capital_factors: dict[str, float] | None = None, basis: dict | None = None,
                    prices: dict | None = None, region: str = "MX", currency: str = "USD", plant_type: str | None = None,
                    main_product: str | None = None, year: int = 2023, cepci: float | None = None,
                    exchange_rates: dict[str, float] | None = None, sensitivity_span: float = 0.2, today: str | None = None) -> str:
    """Whole-plant economics in ONE call from equipment [{name, type|F_BM, purchase_cost_usd, base_purchase_cost_usd, isbl?,
    labour_class?, cost_year?}] and flows [{name, flow, unit ('kg/h','t/h','m3/h','kW','GJ/h','t/y'), price?, price_unit?
    ('USD/t'), reference_item?, density_kg_m3?}] (raw_materials, products, utilities incl. waste). Any ECONOMIC input not
    given (prices, capital_factors, basis: hours, life, tax, discount rate, depreciation, schedule, start-up; labour) is
    PROPOSED from a sourced bank and flagged; unknown prices go to `missing`, never invented. Returns capital levels,
    operating-cost breakdown, cash flow, npv_usd, irr, payback, minimum_selling_price, sensitivity (confirm_first),
    assumed_inputs (source, url, date), user_inputs, missing, inputs_resolved, evidence. Present assumptions as such."""
    return json.dumps(TPE.plant_economics(equipment, raw_materials, products, utilities, labour, capital_factors, basis, prices,
                                          region, currency, plant_type, main_product, year, cepci, exchange_rates, sensitivity_span,
                                          today), default=float, ensure_ascii=False)


@beta_tool
def plant_economics_update(inputs_resolved: dict, overrides: dict) -> str:
    """Re-run plant_economics when the user brings real data. inputs_resolved: that field of the previous result, as is.
    overrides: same keys as plant_economics, e.g. {'prices': {'methanol': {'price': 520, 'price_unit': 'USD/t'}},
    'basis': {'discount_rate': 0.12}, 'capital_factors': {...}, 'labour': {...}, 'equipment': {'R-101':
    {'purchase_cost_usd': 4.1e5}}}; dicts merge key by key, equipment and streams are patched by name. Returns the new
    full result plus changes {rows: input before → after with its effect alone on TCI, operating cost, NPV, IRR and
    minimum selling price; effect_together; n_assumed_before/after; assumed_remaining}."""
    return json.dumps(TPE.plant_economics_update(inputs_resolved, overrides), default=float, ensure_ascii=False)


@beta_tool
def plant_report(result: dict, title: str, path: str | None = None) -> str:
    """Spanish Markdown report of a plant_economics / plant_economics_update result: headline, capital by level and by
    equipment, operating cost by item, cash flow, sensitivity ranking, PTU bounds, the separate table «Supuestos por
    confirmar» (source, URL, period, date read of every assumed value), missing inputs, warnings. path: optional .md
    file to write (its folder must exist). Returns {markdown, path, n_lines, sections}."""
    return json.dumps(TPE.plant_report(result, title, path), default=float, ensure_ascii=False)


@beta_tool
def hazop_from_process_description(units: list[dict], streams: list[dict]) -> str:
    """Rule-based preliminary HAZOP of a plant that is NOT a DWSIM flowsheet. units: [{name, type, inlets, outlets,
    duty_kW?}] with type in the rules' vocabulary (Pump, Compressor, Heater, Cooler, HeatExchanger, Valve, Vessel,
    Separator, Tank, Mixer, Splitter, DistillationColumn, AbsorptionColumn, RCT_CSTR, RCT_PFR, RCT_Batch, RCT_Generic;
    Spanish aliases accepted; other types get generic flow rows only). streams: [{name, T_K, P_Pa, composition {DWSIM
    English name: mole fraction}, vapor_fraction?, molar_flow_mol_s?, mass_flow_kg_s?}]. Returns rows (node, deviation,
    causes, consequences, safeguards, severity 1-5, evidence), markdown, missing, unrecognized_components, evidence.
    Same deterministic rules as hazop_preliminary; not a substitute for a team HAZOP."""
    return json.dumps(TPE.hazop_from_process_description(units, streams), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [plant_economics, plant_economics_update, plant_report, hazop_from_process_description]
