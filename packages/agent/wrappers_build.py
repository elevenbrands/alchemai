"""Agent-facing wrappers of the whole-flowsheet tools (`packages.dwsim_client.tools_build`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_build as TB


@beta_tool
def build_flowsheet(name: str, compounds: list[str], property_package: str, streams: list[dict] | None = None,
                    units: list[dict] | None = None, reactions: list[dict] | None = None, solve: bool = True) -> str:
    """Build a WHOLE flowsheet in one call and solve it: use this instead of one call per stream/unit/connection whenever
    the process has more than ~4 units. streams: [{name, temperature_K, pressure_Pa, vapor_fraction, molar_flow_mol_s,
    mass_flow_kg_s, mole_fractions, compound_molar_flows_mol_s}] — list only specified feeds; every other stream named in
    a unit is created empty. units: [{kind, name, params, feeds, products, energy_feeds, energy_products}] with kinds,
    params and port order exactly as in add_unit/connect (reactors: products [vapor, liquid]; columns: [distillate,
    bottoms] + both duties; recycle: feeds [tear], products [recycled]); energy streams are created from their names.
    reactions: [{type: "conversion"|"equilibrium"|"kinetic", name, stoichiometry, base_compound, + the arguments of the
    matching define_*_reaction, reaction_set}]. If an element is rejected nothing is kept and the error names it: fix the
    spec and call again. Returns flowsheet_id, dof {ok, problems} and solve (same report as solve(), with every stream):
    only ok=true is trustworthy. Afterwards refine with set_unit_param/set_stream_spec/sweep/flowsheet_goal_seek."""
    return json.dumps(TB.build_flowsheet(name, compounds, property_package, streams, units, reactions, solve), default=float,
                      ensure_ascii=False)


@beta_tool
def flowsheet_goal_seek(flowsheet_id: str, variables: list[dict], targets: list[dict], tolerance: float = 1e-4,
                        max_solves: int = 60) -> str:
    """Design specification ("adjust") on a built flowsheet, in one call: vary N specs until N readouts hit their targets
    (N = 1 to 3 together), re-solving every trial. variables: [{"kind": "unit"|"stream", "name", "param", "lower",
    "upper"}] (param as in sweep: split_ratio_0, outlet_T_K, molar_flow_mol_s, reflux_ratio, ...). targets: [{"kind",
    "name", "key", "value"}] with key as in sweep ("component_flow.<C>" mol/s, "mole_fractions.<C>", "T_K",
    "molar_flow_mol_s", "energy_streams_kW.<E>"); add "divide_by": {"kind", "name", "key"} for a ratio (e.g. H2/toluene at
    a reactor inlet = 7 by varying the makeup flow; purge fraction for an inert/reactant ratio). tolerance is relative.
    Returns converged, the values found (at_bound flags), readouts, residuals, solves and the final solve report; the
    flowsheet stays solved there. converged=false: the targets are not met — do not report those values."""
    return json.dumps(TB.flowsheet_goal_seek(flowsheet_id, variables, targets, tolerance, max_solves), default=float,
                      ensure_ascii=False)


AGENT_WRAPPERS = [build_flowsheet, flowsheet_goal_seek]
