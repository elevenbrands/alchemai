"""Agent-facing wrappers of the process flow diagram block M27_D (same names and order as DIAGRAM_TOOLS)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_diagram as TDG


@beta_tool
def process_diagram(process_model: dict[str, object] | None = None, flowsheet_id: str | None = None,
                    plant_result: dict[str, object] | None = None, description: dict[str, object] | None = None,
                    opportunities: list[dict[str, object]] | None = None, view: str = "sections", theme: str = "light",
                    path: str | None = None, table_streams: list[str] | None = None, today: str | None = None) -> str:
    """Draw a professional process flow diagram (SVG, ISO 10628 / ISA-style symbols, orthogonal numbered streams, stream
    table, legend, title block) from exactly ONE source: process_model (ProcessModel JSON), flowsheet_id (open DWSIM
    flowsheet), plant_result (dict returned by reference_biorefinery / biodiesel_plant) or description ({units [{name, type,
    inlets, outlets}], streams}). opportunities (block O shape) are drawn as numbered badges coloured by category with a
    side list of title and USD/y. view: sections | full | section:<id>; theme: light | dark; path writes the .svg file.
    Returns svg or path, scene (JSON for the app), legend, n_units, n_streams, warnings, evidence (geometry checks)."""
    return json.dumps(TDG.process_diagram(process_model, flowsheet_id, plant_result, description, opportunities, view, theme,
                                          path, table_streams, today), default=float, ensure_ascii=False)


@beta_tool
def diagram_symbols() -> str:
    """Closed vocabulary of equipment types accepted by ProcessModel units[].type (with Spanish names and symbol sizes),
    plus how DWSIM unit kinds and hand-description aliases map onto it. Call before building a ProcessModel by hand."""
    return json.dumps(TDG.diagram_symbols(), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [process_diagram, diagram_symbols]
