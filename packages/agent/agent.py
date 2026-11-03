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
from packages.hazop.report import hazop_preliminary as _hazop  # noqa: E402

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
   Duties keep the simulator's sign: heat removed from a unit (exothermic reactor, cooler duty read as a
   reactor/heater duty) is NEGATIVE; power consumed by a compressor/pump is positive. For a polytropic
   compressor pass polytropic_eff_pct (not adiabatic_eff_pct) and read duty_or_power_kW.
8. Some cards are infeasible or inconsistent (temperature cross, pump/compressor whose outlet pressure is
   below the inlet, valve asked to raise pressure, distillate beyond an azeotrope, reflux below R_min, liquid
   demanded above the critical temperature, reactant missing, split ratios or product flows that break the
   mass balance). Never force the simulator to fake a number for these. Check with the tools where you can
   (e.g. solve and see a negative power, a failed column, VF stuck at 1), then set
   "infeasible": {"detected": true, "reason": "<which physical limit, with the numbers>"} and leave the
   answers empty (or flagged unverified). Do NOT flag a card infeasible merely because it is hard, converges
   slowly or the result surprises you: only when a physical law or a limit of the specified system is violated.

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
def add_unit(flowsheet_id: str, kind: str, name: str, params: dict[str, object] | None = None) -> str:
    """Add a unit operation. kind: mixer | splitter | heater | cooler | valve | pump | compressor | expander |
    separator | heat_exchanger | shortcut_column | distillation_column | conversion_reactor |
    equilibrium_reactor | gibbs_reactor | cstr | pfr | recycle. params by kind: splitter {split_ratios:[...]};
    heater {outlet_T_K | heat_added_kW}; cooler {outlet_T_K | heat_removed_kW}; valve {outlet_P_Pa};
    pump {outlet_P_Pa, efficiency_pct}; compressor/expander {outlet_P_Pa, adiabatic_eff_pct | polytropic_eff_pct};
    heat_exchanger {UA_W_K, hot_dP_Pa, cold_dP_Pa}; shortcut_column {light_key, heavy_key,
    lk_mole_frac_in_bottoms, hk_mole_frac_in_distillate, condenser_P_Pa, reboiler_P_Pa, reflux_ratio};
    distillation_column {stages, feed_stage, reflux_ratio, bottoms_molar_flow_mol_s | distillate_molar_flow_mol_s,
    top_P_Pa, dP_Pa}; *_reactor {reaction_set, mode: "isothermal"|"adiabatic", dP_Pa}; cstr/pfr {volume_m3,
    reaction_set, mode: "isothermal" (= feed T)|"adiabatic", dP_Pa} — products [outlet], energy_feeds [duty] required;
    recycle {}."""
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
    vapor_fraction / molar_flow_mol_s / mass_flow_kg_s), re-solve each, read one quantity (read_key as in
    read_unit/read_stream, e.g. "T_K", "vapor_fraction", "power_kW", "mole_fractions.Ethanol",
    "energy_streams_kW.Qr") and return the table plus the interpolated crossing of `target`. One call = whole search."""
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
def close_flowsheet(flowsheet_id: str) -> str:
    """Discard a flowsheet when you no longer need it (e.g. between search iterations)."""
    return json.dumps(T.close_flowsheet(flowsheet_id))


AGENT_TOOLS = [list_property_packages, new_flowsheet, add_material_stream, add_energy_stream, define_conversion_reaction,
               define_equilibrium_reaction, define_kinetic_reaction, add_unit, connect, solve, read_stream, read_unit,
               set_unit_param, set_stream_spec, sweep, hazop_preliminary, close_flowsheet]

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
