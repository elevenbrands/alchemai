"""Agent-facing wrapper of the design-heuristics review."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_review as TR


@beta_tool
def design_heuristics_review(checks: list[dict] | None = None, flowsheet_id: str | None = None,
                             list_areas: list[str] | None = None) -> str:
    """Design review with published engineering HEURISTICS (engineer's judgement): compares design values with usual
    practice and reports which fall outside, with the source of every rule. checks: [{rule_id, value, label?}] in the
    rule's units; flowsheet_id: a solved DWSIM flowsheet (compressor ratio and discharge T, exchanger approach, cooler and
    heater outlet T, column condenser/reboiler T, R/R_min and N/N_min of shortcut columns are extracted); list_areas: []
    for the catalogue. A warning means "outside usual practice per <source>", not an error: report it with its source."""
    return json.dumps(TR.design_heuristics_review(checks, flowsheet_id, list_areas), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [design_heuristics_review]
