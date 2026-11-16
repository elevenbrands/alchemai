"""Agent-facing wrapper of the one-question dossier (bloque M27)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_dossier as TDS


@beta_tool
def process_dossier(subject: str, refinery: str | None = None, plant: dict[str, object] | None = None, region: str = "MX",
                    prices: dict[str, object] | None = None, plant_data: list[dict[str, object]] | None = None,
                    period: str | None = None, out_dir: str | None = None, theme: str = "light",
                    power_plant: str | None = None, year: int | None = None) -> str:
    """Whole answer to «explain this plant/refinery, its profitability, diagram and opportunities» in ONE call: runs the
    refinery model (refinery='tula'…, as operated), a power station (power_plant='laguna_verde', year) or a plant model (plant={'model': 'biodiesel'|'sugarcane_ethanol'|…,
    …args}), regional economics, the opportunity scanner, and draws the PFD (sections + full) with opportunities numbered
    on their units; writes a Spanish Markdown report and two SVGs and returns their paths, a summary (margins, validation vs
    published data, calibrated parameters, CO2) and the opportunities. Explain from the summary; keep its flags."""
    return json.dumps(TDS.process_dossier(subject, refinery, plant, region, prices, plant_data, period, out_dir, theme,
                                                power_plant, year),
                      default=float, ensure_ascii=False)


AGENT_WRAPPERS = [process_dossier]
