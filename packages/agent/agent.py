"""AlchemAI agent — Claude + DWSIM tools, speaking the Bench v0 protocol.

    uv run python packages/agent/agent.py < card.json  > answer.json
    (or import `solve_card`)

Rules the agent lives by (system prompt): DWSIM computes, verify() checks, the model only decides what
to build and how to read it. Any number not backed by a solve with ok=true is reported as unverified.
Credentials: ANTHROPIC_API_KEY from the environment, or from the macOS Keychain item
(service ANTHROPIC_API_KEY, account alchemai) — never from files.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anthropic  # noqa: E402
from anthropic import beta_tool  # noqa: E402

from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client import tools_sweep as TS  # noqa: E402
from packages.dwsim_client import tools_transport as TT  # noqa: E402
from packages.dwsim_client import tools_costing as TC  # noqa: E402
from packages.dwsim_client import tools_optimize as TO  # noqa: E402
from packages.dwsim_client import tools_mass_transfer as TM  # noqa: E402
from packages.hazop.report import hazop_preliminary as _hazop  # noqa: E402
from packages.export.flowsheet_export import export_flowsheet as _export  # noqa: E402
from packages.hazop.dexpi_graph import hazop_from_dexpi as _hazop_dexpi  # noqa: E402

MODEL = os.environ.get("ALCHEMAI_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ALCHEMAI_EFFORT", "high")

SYSTEM = """You are AlchemAI, a process-design agent for chemical engineers. You have a deterministic
process simulator (DWSIM 10) behind the tools. Your job on each card: understand the design question,
build the flowsheet with the tools, call solve(), read results, and answer.

Hard rules:
1. The simulator and its verifier are the only source of truth. Never estimate a number by hand when a
   tool can compute it. Hand calculations are allowed only to pick inputs (e.g. a stoichiometric air flow)
   or to search (e.g. try several pressures), never as reported answers.
2. Call solve() before reading anything. If solve() returns ok=false or an error, fix the flowsheet
   (missing products, wrong ports, bad specs) and solve again. Reactors need BOTH a vapor and a liquid
   product stream; columns need distillate, bottoms and both duties; separators need vapor and liquid.
3. Design questions (find a T, P, N, UA, fraction...) are solved by searching with the simulator. Use
   sweep(): it varies one spec over a list of values, re-solves each time and returns the table and the
   interpolated crossing of your target in ONE call. Then sweep a finer range around the crossing (or
   set_unit_param/set_stream_spec + solve) and report the verified value. Never iterate one solve per
   turn when sweep can do it; never stop at an unverified interpolation.
6. If the card asks for hazards, deviations, safeguards or a HAZOP, call hazop_preliminary() after a
   successful solve and summarise its rows; present it as a rule-based preliminary draft.
7. Use exactly the flows, temperatures, pressures and basis stated in the card. If the card omits a
   basis (e.g. a flow rate) and the answer depends on it, state the basis you assumed in notes and list
   the affected quantities in "unverified".
4. Report exactly the quantities the card lists (same names, SI units stated). If you could not
   verify a quantity with a successful solve, still report your best value but list it in "unverified".
5. Efficiencies given in percent in tools are reported as fractions when the card asks for a fraction.
   Duties and powers are reported with the sign the TOOL returns, never flipped by you. DWSIM conventions:
   a Cooler's duty, a column's condenser duty (Qc) and reboiler duty (Qr) are POSITIVE magnitudes even
   though the condenser/cooler remove heat; a Heater used to cool, a valve's or an exothermic reactor's duty
   come out NEGATIVE; compressor/pump power is positive. A card saying "heat removed" or "calor retirado"
   does not ask you to negate the number: report what the tool read. For a polytropic compressor pass
   polytropic_eff_pct (not adiabatic_eff_pct) and read duty_or_power_kW.
   Read each requested quantity from the stream or unit its name points at (T_mixed_K = the mixer outlet,
   not the heater outlet; x_..._dist = distillate, x_..._bot = bottoms) and say in notes which one you read.
8. Some cards are infeasible or inconsistent (temperature cross, pump/compressor whose outlet pressure is
   below the inlet, valve asked to raise pressure, distillate beyond an azeotrope, reflux below R_min, liquid
   demanded above the critical temperature, reactant missing, split ratios or product flows that break the
   mass balance). Never force the simulator to fake a number for these. Check with the tools where you can
   (e.g. solve and see a negative power, a failed column, VF stuck at 1), then set
   "infeasible": {"detected": true, "reason": "<which physical limit, with the numbers>"} and leave the
   answers empty (or flagged unverified). Do NOT flag a card infeasible merely because it is hard, converges
   slowly or the result surprises you: only when a physical law or a limit of the specified system is violated.
9. Specification errors are a second kind of trap. UNDER-specified: the card omits a quantity the answer depends
   on non-linearly (a pressure, a temperature, the composition, a column's reflux ratio or stage count, a
   reaction's conversion). OVER-specified: the card fixes more than the unit can accept (two conflicting specs on
   one unit such as an outlet temperature AND a duty, the outlet of a unit whose outlet is computed, a mixer
   outlet temperature, a split that is also given as flows). In both cases build what you can, call dof_check()
   to confirm, and report "infeasible": {"detected": true, "reason": "under-specified: missing <what>" |
   "over-specified: <which two specs conflict> (and the numbers if they disagree)"}. Never invent the missing
   value and never silently drop one of two conflicting specs. The ONLY omission you may fill yourself is a
   missing flow basis (rule 7), because it just scales extensive results.

Finish with ONE fenced JSON block, nothing after it:
```json
{"answers": {"<quantity>": <number>, ...},
 "evidence": {"converged": true|false, "balances_closed": true|false, "notes": "<1-3 lines: what you built, max residuals>"},
 "unverified": ["<quantity>", ...],
 "infeasible": {"detected": false, "reason": ""}}
```"""


def _keychain_key() -> str | None:
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-a", "alchemai", "-w"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def make_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY") or _keychain_key()
    if not key:
        raise RuntimeError("No Anthropic credentials: set ANTHROPIC_API_KEY or store it in Keychain "
                           "(security add-generic-password -a alchemai -s ANTHROPIC_API_KEY -w)")
    return anthropic.Anthropic(api_key=key)


# Tool wrappers: thin, typed, JSON in/out. The SDK builds the schemas from signatures + docstrings.
@beta_tool
def list_property_packages() -> str:
    """List DWSIM property packages (PengRobinson, SoaveRedlichKwong, NRTL, UNIQUAC, Wilson, SteamTables, CoolProp...)."""
    return json.dumps(T.list_property_packages())


@beta_tool
def new_flowsheet(name: str, compounds: list[str], property_package: str) -> str:
    """Create a headless flowsheet. compounds use DWSIM names: Methane, Ethane, Propane, N-butane, Isobutane,
    N-pentane, N-hexane, N-heptane, Nitrogen, Oxygen, Carbon dioxide, Carbon monoxide, Hydrogen, Water,
    Ethanol, Methanol, Benzene, Toluene. Returns the flowsheet_id to use in every other tool."""
    return json.dumps(T.new_flowsheet(name, compounds, property_package))


@beta_tool
def add_material_stream(flowsheet_id: str, name: str, temperature_K: float | None = None, pressure_Pa: float | None = None,
                        vapor_fraction: float | None = None, molar_flow_mol_s: float | None = None,
                        mass_flow_kg_s: float | None = None, mole_fractions: dict[str, float] | None = None,
                        compound_molar_flows_mol_s: dict[str, float] | None = None) -> str:
    """Add a material stream. A feed needs two of (temperature_K, pressure_Pa, vapor_fraction), one flow and a
    composition (mole_fractions summing to 1, or compound_molar_flows_mol_s). A product stream needs only a name."""
    return json.dumps(T.add_material_stream(flowsheet_id, name, temperature_K, pressure_Pa, vapor_fraction,
                                            molar_flow_mol_s, mass_flow_kg_s, mole_fractions, compound_molar_flows_mol_s))


@beta_tool
def add_energy_stream(flowsheet_id: str, name: str) -> str:
    """Add an energy stream (heat duty or shaft power carrier)."""
    return json.dumps(T.add_energy_stream(flowsheet_id, name))


@beta_tool
def define_conversion_reaction(flowsheet_id: str, name: str, stoichiometry: dict[str, float], base_compound: str,
                               conversion_percent: float, reaction_set: str = "RS1") -> str:
    """Define a conversion reaction. stoichiometry: negative reactants, positive products,
    e.g. {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}. Pass reaction_set to the reactor."""
    return json.dumps(T.define_conversion_reaction(flowsheet_id, name, stoichiometry, base_compound, conversion_percent, reaction_set))


@beta_tool
def define_equilibrium_reaction(flowsheet_id: str, name: str, stoichiometry: dict[str, float], base_compound: str,
                                lnK_expression_of_T: str, reaction_set: str = "RS1") -> str:
    """Define an equilibrium reaction with ln K as an expression in T (kelvin), e.g. "4577.8/T - 4.33"."""
    return json.dumps(T.define_equilibrium_reaction(flowsheet_id, name, stoichiometry, base_compound, lnK_expression_of_T, reaction_set))


@beta_tool
def define_kinetic_reaction(flowsheet_id: str, name: str, stoichiometry: dict[str, float], base_compound: str,
                            direct_orders: dict[str, float], A_forward: float, E_forward_J_mol: float,
                            reverse_orders: dict[str, float] | None = None, A_reverse: float = 0.0, E_reverse_J_mol: float = 0.0,
                            reaction_set: str = "KIN") -> str:
    """Arrhenius kinetic reaction for cstr/pfr units: r = A·exp(−E/RT)·Π C_i^order, concentrations in mol/m³, rate in
    mol/[m³·s] (first order → A in 1/s), E in J/mol. Orders per compound (missing → 0); A_reverse=0 → irreversible."""
    return json.dumps(T.define_kinetic_reaction(flowsheet_id, name, stoichiometry, base_compound, direct_orders, A_forward,
                                                E_forward_J_mol, reverse_orders, A_reverse, E_reverse_J_mol, reaction_set))


@beta_tool
def dof_check(flowsheet_id: str) -> str:
    """Specification / degrees-of-freedom check BEFORE solve(): feed streams need composition, one flow and two of (T, P,
    vapor fraction); produced streams must not be specified; each unit kind needs fixed ports and params, and mutually
    exclusive specs (outlet_T_K + heat_added_kW, adiabatic + polytropic efficiency) are over-specification. Returns ok,
    plain-words problems and per-stream/per-unit status. Bookkeeping, not physics."""
    return json.dumps(T.dof_check(flowsheet_id))


@beta_tool
def add_unit(flowsheet_id: str, kind: str, name: str, params: dict[str, object] | None = None) -> str:
    """Add a unit operation. kind: mixer | splitter | heater | cooler | valve | pump | compressor | expander |
    separator | heat_exchanger | shortcut_column | distillation_column | conversion_reactor |
    equilibrium_reactor | gibbs_reactor | cstr | pfr | recycle | pipe | absorption_column. params by kind: splitter {split_ratios:[...]};
    heater {outlet_T_K | heat_added_kW}; cooler {outlet_T_K | heat_removed_kW}; valve {outlet_P_Pa};
    pump {outlet_P_Pa, efficiency_pct}; compressor/expander {outlet_P_Pa, adiabatic_eff_pct | polytropic_eff_pct};
    heat_exchanger {UA_W_K, hot_dP_Pa, cold_dP_Pa}; shortcut_column {light_key, heavy_key,
    lk_mole_frac_in_bottoms, hk_mole_frac_in_distillate, condenser_P_Pa, reboiler_P_Pa, reflux_ratio};
    distillation_column {stages, feed_stage (1-based, 1 = condenser), reflux_ratio, bottoms_molar_flow_mol_s | distillate_molar_flow_mol_s,
    top_P_Pa, dP_Pa}; *_reactor {reaction_set, mode: "isothermal"|"adiabatic", dP_Pa}; cstr/pfr {volume_m3,
    reaction_set, mode: "isothermal" (= feed T)|"adiabatic", dP_Pa} — products [outlet], energy_feeds [duty] required;
    recycle {}; pipe {length_m, inner_diameter_m, elevation_change_m, increments} (adiabatic, Beggs-Brill, two-phase OK);
    absorption_column {stages (≥ 4), top_P_Pa, dP_Pa}: feeds [lean liquid, gas], products [treated gas, rich liquid]."""
    return json.dumps(T.add_unit(flowsheet_id, kind, name, params))


@beta_tool
def connect(flowsheet_id: str, unit: str, feeds: list[str] | None = None, products: list[str] | None = None,
            energy_feeds: list[str] | None = None, energy_products: list[str] | None = None) -> str:
    """Connect streams to a unit in port order. separator products=[vapor, liquid]; columns
    products=[distillate, bottoms], energy_products=[condenser duty], energy_feeds=[reboiler duty];
    reactors products=[vapor, liquid] (both required), energy_feeds=[heat duty]; heat_exchanger
    feeds=[hot_in, cold_in], products=[hot_out, cold_out]; recycle feeds=[tear], products=[recycled]."""
    return json.dumps(T.connect(flowsheet_id, unit, feeds, products, energy_feeds, energy_products))


@beta_tool
def solve(flowsheet_id: str) -> str:
    """Solve the flowsheet and verify it. Returns ok (all balances closed), solver_errors, per-unit residuals and
    duties, and every stream's T, P, vapor fraction, flows and mole fractions. Call after every change."""
    return json.dumps(T.solve(flowsheet_id), default=float)


@beta_tool
def read_stream(flowsheet_id: str, name: str) -> str:
    """Read one solved stream (T_K, P_Pa, vapor_fraction, flows, mole_fractions)."""
    return json.dumps(T.read_stream(flowsheet_id, name), default=float)


@beta_tool
def read_unit(flowsheet_id: str, name: str) -> str:
    """Read one unit's results: duty/power kW, efficiencies (percent), shortcut N_min/R_min/N/feed stage,
    reactor heat duty, recycle convergence, attached energy streams (kW)."""
    return json.dumps(T.read_unit(flowsheet_id, name), default=float)


@beta_tool
def set_unit_param(flowsheet_id: str, unit: str, param: str, value: float) -> str:
    """Re-specify one numeric parameter of an existing unit (outlet_T_K, outlet_P_Pa, UA_W_K, reflux_ratio,
    adiabatic_eff_pct, polytropic_eff_pct, efficiency_pct, heat_added_kW, heat_removed_kW, split_ratio_0). Then solve()."""
    return json.dumps(TS.set_unit_param(flowsheet_id, unit, param, value))


@beta_tool
def set_stream_spec(flowsheet_id: str, stream: str, temperature_K: float | None = None, pressure_Pa: float | None = None,
                    vapor_fraction: float | None = None, molar_flow_mol_s: float | None = None,
                    mass_flow_kg_s: float | None = None, mole_fractions: dict[str, float] | None = None,
                    compound_molar_flows_mol_s: dict[str, float] | None = None) -> str:
    """Re-specify a feed stream (any subset of T, P, vapor fraction, flow, composition). Then solve()."""
    return json.dumps(TS.set_stream_spec(flowsheet_id, stream, temperature_K, pressure_Pa, vapor_fraction,
                                         molar_flow_mol_s, mass_flow_kg_s, mole_fractions, compound_molar_flows_mol_s))


@beta_tool
def sweep(flowsheet_id: str, vary_kind: str, vary_name: str, vary_param: str, values: list[float],
          read_kind: str, read_name: str, read_key: str, target: float | None = None) -> str:
    """Vary one spec over `values` (unit param as in set_unit_param, or stream spec temperature_K / pressure_Pa /
    vapor_fraction / molar_flow_mol_s / mass_flow_kg_s / "mole_fraction.<Compound>" with the others sharing the rest),
    re-solve each, read one quantity (read_key as in read_unit/read_stream, e.g. "T_K", "vapor_fraction", "power_kW",
    "mole_fractions.Ethanol", "energy_streams_kW.Qr") and return the table, the rows with the minimum/maximum readout
    and the interpolated crossing of `target`. One call = whole search."""
    return json.dumps(TS.sweep(flowsheet_id, vary_kind, vary_name, vary_param, values, read_kind, read_name, read_key, target), default=float)


@beta_tool
def hazop_preliminary(flowsheet_id: str) -> str:
    """Preliminary HAZOP of a SOLVED flowsheet: deterministic deviation rules per unit (guide words × parameters)
    instantiated with the solved T, P, phases, duties and hazard flags. Returns rows (node, parameter, deviation,
    causes, consequences, safeguards, severity 1-5, evidence) and a Markdown table. Use when the card asks for
    hazards, deviations, safeguards or a HAZOP. Report it as a preliminary, rule-based draft."""
    out = _hazop(flowsheet_id)
    out.pop("markdown", None)
    return json.dumps(out, ensure_ascii=False, default=float)


@beta_tool
def export_flowsheet(flowsheet_id: str) -> str:
    """Export a SOLVED flowsheet to SFILES 2.0 (one-line topology) and DEXPI Process files (XML + JSON with the
    solved T, P, mass flow), with round-trip checks. Use only when the card or user asks for SFILES, DEXPI or an
    exportable/shareable flowsheet model."""
    return json.dumps(_export(flowsheet_id), ensure_ascii=False, default=float)


@beta_tool
def hazop_from_dexpi(path: str) -> str:
    """Preliminary HAZOP of a process given as a DEXPI file (.xml DEXPI XML or .json pyDEXPI JSON), with no
    simulation: the same deterministic rules as hazop_preliminary, using the T, P, flows and compositions in the
    file. Returns rows and `missing` (data the file lacked). Use when the user provides a DEXPI file."""
    out = _hazop_dexpi(path)
    out.pop("markdown", None)
    return json.dumps(out, ensure_ascii=False, default=float)


@beta_tool
def stream_transport_properties(flowsheet_id: str, stream: str) -> str:
    """Density, viscosity, thermal conductivity and Cp of a solved single-phase stream (DWSIM property package)."""
    return json.dumps(TT.stream_transport_properties(flowsheet_id, stream), default=float)


@beta_tool
def pipe_hydraulics(flowsheet_id: str, stream: str, length_m: float, inner_diameter_m: float, roughness_m: float = 4.57e-5,
                    elevation_change_m: float = 0.0, fittings_K: float = 0.0) -> str:
    """Pressure drop of a solved single-phase stream in a straight pipe (Darcy-Weisbach, Colebrook; laminar 64/Re):
    velocity, Re, friction factor, friction + fittings (ΣK) + static (ρ·g·Δz, positive uphill) ΔP and outlet P."""
    return json.dumps(TT.pipe_hydraulics(flowsheet_id, stream, length_m, inner_diameter_m, roughness_m, elevation_change_m, fittings_K), default=float)


@beta_tool
def size_pipe(flowsheet_id: str, stream: str, max_velocity_m_s: float, schedule: str = "40") -> str:
    """Smallest common nominal steel pipe (ASME B36.10M) keeping the velocity of a solved single-phase stream at or
    below max_velocity_m_s; returns NPS, inner diameter, velocity, Re and ΔP per 100 m."""
    return json.dumps(TT.size_pipe(flowsheet_id, stream, max_velocity_m_s, schedule), default=float)


@beta_tool
def heat_exchanger_area(duty_kW: float, hot_in_K: float, hot_out_K: float, cold_in_K: float, cold_out_K: float,
                        U_W_m2K: float, shell_passes: int = 0) -> str:
    """Heat-exchanger area A = Q/(U·F·LMTD); shell_passes = 0 → counter-current (F = 1), N ≥ 1 → 1-N shell-and-tube."""
    return json.dumps(TT.heat_exchanger_area(duty_kW, hot_in_K, hot_out_K, cold_in_K, cold_out_K, U_W_m2K, shell_passes), default=float)


@beta_tool
def tube_film_coefficient(flowsheet_id: str, stream: str, inner_diameter_m: float, velocity_m_s: float) -> str:
    """Tube-side film coefficient h (W/m²K) of a solved single-phase stream: Gnielinski if turbulent, Nu = 3.66 laminar."""
    return json.dumps(TT.tube_film_coefficient(flowsheet_id, stream, inner_diameter_m, velocity_m_s), default=float)


@beta_tool
def equipment_cost(equipment: str, size: dict[str, object], options: dict[str, object] | None = None, year: int = 2023,
                   cepci: float | None = None) -> str:
    """Purchase cost (USD f.o.b., not installed) by Seider et al. via IDAES. equipment: heat_exchanger {area_m2,
    shell_pressure_Pa} | pump {power_W, deltaP_Pa, density_kg_m3, flow_m3_s} | compressor {power_W} | vessel
    {diameter_m, length_m, vertical}. options as listed by costing_options. year 2015-2023, else pass cepci."""
    return json.dumps(TC.equipment_cost(equipment, size, options, year, cepci), default=float)


@beta_tool
def costing_options() -> str:
    """Valid option names for equipment_cost (types, materials, motors, tube lengths)."""
    return json.dumps(TC.costing_options())


@beta_tool
def cost_unit(flowsheet_id: str, unit: str, options: dict[str, object] | None = None, year: int = 2023, cepci: float | None = None) -> str:
    """Purchase cost of a solved DWSIM pump or compressor, sized from the simulation."""
    return json.dumps(TC.cost_unit(flowsheet_id, unit, options, year, cepci), default=float)


@beta_tool
def flowsheet_cost(flowsheet_id: str, prices: dict[str, float], hours_per_year: float, interest: float, years: int,
                   exchanger_areas_m2: dict[str, float] | None = None, vessel_sizes: dict[str, object] | None = None,
                   options: dict[str, object] | None = None, lang: str = "lang_1948_fluid", country: str | None = None,
                   year: int = 2023) -> str:
    """Total annual cost of a solved flowsheet: Σ purchase (Seider via IDAES) × Lang × location, annualized by
    CRF(interest, years), + electricity/heating/cooling at the given prices (electricity_usd_kWh, heating_usd_GJ,
    cooling_usd_GJ). Exchangers need exchanger_areas_m2; vessels need vessel_sizes."""
    return json.dumps(TC.flowsheet_cost(flowsheet_id, prices, hours_per_year, interest, years, exchanger_areas_m2, vessel_sizes,
                                        options, lang, country, year), default=float)


@beta_tool
def optimize_design(flowsheet_id: str, vary_kind: str, vary_name: str, vary_param: str, lower: float, upper: float,
                    objective: dict[str, object], constraints: list[dict[str, object]] | None = None, maximize: bool = False,
                    log_scale: bool = False, n_scan: int = 9, refine_iterations: int = 12) -> str:
    """Optimize ONE design variable in one call: scan + golden section, re-solving DWSIM each point. objective
    {"type": "total_annual_cost", <flowsheet_cost args incl. exchanger_specs>} or {"type": "sum", "terms": [{kind, name,
    key, weight}]}; constraints [{kind, name, key, max|min}] are hard limits (e.g. HAZOP discharge T). Returns best,
    curve, active_constraint, flat_range; leaves the flowsheet solved at the optimum."""
    return json.dumps(TO.optimize_design(flowsheet_id, vary_kind, vary_name, vary_param, lower, upper, objective, constraints,
                                         maximize, log_scale, n_scan, refine_iterations), default=float)


@beta_tool
def absorption_stages(gas_flow_mol_s: float, liquid_flow_mol_s: float, K: float, y_in: float, fraction_absorbed: float | None = None,
                      stages: float | None = None, x_in: float = 0.0) -> str:
    """Kremser (dilute solute): A = L/(K V); stages needed for fraction_absorbed, or fraction absorbed for stages."""
    return json.dumps(TM.absorption_stages(gas_flow_mol_s, liquid_flow_mol_s, K, y_in, fraction_absorbed, stages, x_in), default=float)


@beta_tool
def packed_column_height(flowsheet_id: str, gas_stream: str, liquid_stream: str, column_diameter_m: float,
                         packing_specific_area_m2_m3: float, packing_size_m: float, packing_critical_surface_tension_N_m: float,
                         K: float, y_in: float, fraction_absorbed: float, D_liquid_m2_s: float, D_gas_m2_s: float, x_in: float = 0.0) -> str:
    """Packed absorber height Z = H_OG·N_OG (Colburn + Onda 1968 film coefficients), properties from the solved streams."""
    return json.dumps(TM.packed_column_height(flowsheet_id, gas_stream, liquid_stream, column_diameter_m, packing_specific_area_m2_m3,
                                              packing_size_m, packing_critical_surface_tension_N_m, K, y_in, fraction_absorbed,
                                              D_liquid_m2_s, D_gas_m2_s, x_in), default=float)


@beta_tool
def diffusivity_estimate(method: str, T_K: float, P_Pa: float | None = None, M_A: float | None = None, M_B: float | None = None,
                         sum_v_A: float | None = None, sum_v_B: float | None = None, mu_solvent_cP: float | None = None,
                         M_solvent: float | None = None, phi_solvent: float | None = None, V_solute_cm3_mol: float | None = None) -> str:
    """Diffusivity (m²/s): 'fuller' for a gas pair (diffusion volumes as inputs) or 'wilke_chang' for a dilute solute."""
    return json.dumps(TM.diffusivity_estimate(method, T_K, P_Pa, M_A, M_B, sum_v_A, sum_v_B, mu_solvent_cP, M_solvent, phi_solvent,
                                              V_solute_cm3_mol), default=float)


@beta_tool
def packing_catalog() -> str:
    """Stichlmair constants available by name for packed_column_hydraulics, with their source."""
    return json.dumps(TM.packing_catalog())


@beta_tool
def packed_column_hydraulics(flowsheet_id: str, gas_stream: str, liquid_stream: str, packing: str | None = None,
                             packing_constants: dict[str, float] | None = None, flood_fraction: float = 0.7,
                             packed_height_m: float | None = None) -> str:
    """Packed-column diameter at flood_fraction of flooding (Stichlmair 1989), flooding velocity, wet ΔP per m, F-factor."""
    return json.dumps(TM.packed_column_hydraulics(flowsheet_id, gas_stream, liquid_stream, packing, packing_constants, flood_fraction,
                                                  packed_height_m), default=float)


@beta_tool
def optimize_design_multi(flowsheet_id: str, variables: list[dict[str, object]], objective: dict[str, object],
                          constraints: list[dict[str, object]] | None = None, maximize: bool = False, grid_points: int = 5,
                          max_evaluations: int = 150, tolerance: float = 1e-4) -> str:
    """Optimize 2-4 interacting design variables together: coarse grid + Nelder-Mead in normalized coordinates, penalties
    for constraint violations; variables [{kind, name, param, lower, upper, log_scale}], objective/constraints as in
    optimize_design. Returns best values, objective, active constraints, variables at a bound, path."""
    return json.dumps(TO.optimize_design_multi(flowsheet_id, variables, objective, constraints, maximize, grid_points,
                                               max_evaluations, tolerance), default=float)


@beta_tool
def close_flowsheet(flowsheet_id: str) -> str:
    """Discard a flowsheet when you no longer need it (e.g. between search iterations)."""
    return json.dumps(T.close_flowsheet(flowsheet_id))


AGENT_TOOLS = [list_property_packages, new_flowsheet, add_material_stream, add_energy_stream, define_conversion_reaction,
               define_equilibrium_reaction, define_kinetic_reaction, add_unit, connect, dof_check, solve, read_stream, read_unit,
               set_unit_param, set_stream_spec, sweep, stream_transport_properties, pipe_hydraulics, size_pipe,
               heat_exchanger_area, tube_film_coefficient, equipment_cost, costing_options, cost_unit, flowsheet_cost, optimize_design, optimize_design_multi, absorption_stages,
               packed_column_height, diffusivity_estimate, packing_catalog,
               packed_column_hydraulics, hazop_preliminary, export_flowsheet, hazop_from_dexpi, close_flowsheet]

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def _parse_answer(text: str) -> dict:
    m = _JSON_BLOCK.findall(text)
    raw = m[-1] if m else text[text.rfind("{"):]
    try:
        d = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"answers": {}, "evidence": {"notes": "agent did not return valid JSON"}, "unverified": [], "raw": text[-2000:]}
    d.setdefault("answers", {}); d.setdefault("evidence", {}); d.setdefault("unverified", [])
    return d


def solve_card(card: dict, client: anthropic.Anthropic | None = None, model: str = MODEL, effort: str = EFFORT,
               log=None) -> dict:
    """Run the agent on a public bench card (no expected values) and return the protocol answer."""
    client = client or make_client()
    T.SESSIONS.clear()
    user = ("Card:\n```json\n" + json.dumps(card, ensure_ascii=False, indent=1) + "\n```\n"
            "Solve it with the tools and finish with the JSON block.")
    runner = client.beta.messages.tool_runner(
        model=model, max_tokens=16000, system=SYSTEM, tools=AGENT_TOOLS,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": user}],
    )
    final = None
    n_calls = 0
    for message in runner:
        final = message
        for b in message.content:
            if b.type == "tool_use":
                n_calls += 1
                if log:
                    log(f"  tool {b.name}({json.dumps(b.input)[:160]})")
    text = "".join(b.text for b in final.content if b.type == "text") if final else ""
    ans = _parse_answer(text)
    ans.setdefault("evidence", {})["tool_calls"] = n_calls
    ans["evidence"]["model"] = model
    ans["evidence"]["stop_reason"] = getattr(final, "stop_reason", None)
    return ans


if __name__ == "__main__":
    card = json.load(sys.stdin)
    print(json.dumps(solve_card(card, log=lambda s: print(s, file=sys.stderr)), ensure_ascii=False))
