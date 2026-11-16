"""Agent-facing wrappers of the refinery planning tools (`packages.dwsim_client.tools_refinery`, bloque M27_R)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_refinery as TR


@beta_tool
def refinery_model(refinery: str = "tula", crude_slate: dict[str, float] | None = None,
                   utilization: float | dict[str, float] | None = None, prices: dict[str, float] | None = None,
                   unit_overrides: dict[str, object] | None = None, year: int | None = None, period: str | None = None,
                   today: str | None = None, mode: str = "as_operated") -> str:
    """PLANNING model (unit yields + linear balances, not kinetics) of a Pemex refinery from public data: Tula (default),
    cadereyta, madero, minatitlan, salamanca, salina_cruz, dos_bocas. period '2026-01/2026-09' (default), '2026-01/2026-06',
    '2025', '2024'. crude_slate {maya|istmo|olmeca: bpd}; utilization = CDU fraction or {unit: max fraction} (e.g.
    {'coker': 0.25}); prices overrides (USD/bbl, coque/azufre USD/t, gas_seco/energia USD/GJ); unit_overrides
    {capacity_mbd, kerosene_to_jet_fraction, asphalt_bpd}. Returns process_model (for the diagram), product_slate, unit_table,
    balances, energy, emissions, economics (gross/net margin USD/bbl, annual USD, sourced prices), published_comparison.
    mode 'as_operated' (default) solves coker utilization, kerosene-to-jet and butanes-out fractions to reproduce published
    fuel oil, jet and LPG (CALIBRATED, in calibrated_parameters); 'nameplate' runs every unit at published capacity."""
    return json.dumps(TR.refinery_model(refinery, crude_slate, utilization, prices, unit_overrides, year, period, today,
                                        mode),
                      default=float, ensure_ascii=False)


@beta_tool
def list_refineries() -> str:
    """Which Pemex refineries have public data loaded for refinery_model: published capacities per unit (2025), periods,
    latest crude run and production, and the data gaps of each (e.g. Tula coker capacity from a secondary source)."""
    return json.dumps(TR.list_refineries(), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [refinery_model, list_refineries]
