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
from packages.dwsim_client import tools_kinetics as TK  # noqa: E402
from packages.dwsim_client import tools_economics as TE  # noqa: E402
from packages.dwsim_client import tools_flow as TF  # noqa: E402
from packages.dwsim_client import tools_heat as TH  # noqa: E402
from packages.dwsim_client import tools_energy as TEN  # noqa: E402
from packages.dwsim_client import tools_control as TCO  # noqa: E402
from packages.hazop.report import hazop_preliminary as _hazop  # noqa: E402
from packages.export.flowsheet_export import export_flowsheet as _export  # noqa: E402
from packages.hazop.dexpi_graph import hazop_from_dexpi as _hazop_dexpi  # noqa: E402
from packages.pid.tool import pid_from_hazop as _pid  # noqa: E402
from packages.lopa.tool import lopa_from_hazop as _lopa  # noqa: E402
from packages.consequence import tool as _CQ  # noqa: E402
from packages.network import tool as _NW  # noqa: E402
from packages.vle import tool as _VLE  # noqa: E402

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
10. A kinetic PFR in DWSIM is the IDEAL reactor (plug flow). When the card asks about a REAL tubular reactor, laminar
   flow, a radial conversion profile, the effect of tube diameter / number of tubes, or how much the ideal PFR
   over-predicts, call tubular_reactor_field (FEniCSx field of one tube of the bundle) with k from the kinetics,
   the volumetric flow at reactor conditions, the diffusivity and ρ, μ from stream_transport_properties; report
   X_plug_flow and X_laminar side by side and the seal only if its evidence says verified.
   If the reaction releases or absorbs heat and the card asks about a hot spot, runaway, wall/coolant temperature,
   radial temperature profile, or how far the ideal PFR is from a cooled/adiabatic tube, call
   nonisothermal_tubular_reactor instead: every kinetic and thermal number (A, E, ΔH, C and T of the feed, ρ, cp,
   D and λ, U and coolant T or wall T) must come from the card or a tool; if one is missing report it as
   under-specified, never invent it. Empty tube → velocity_profile 'laminar' with molecular D and λ (pass μ for Re);
   packed bed → 'plug' with the effective D_er and λ_er; gas → fluid_model 'ideal_gas'. Report X, outlet T and hot
   spot of the field next to the PFR's, where they diverge, the assumptions, and the seal only if verified.
   For a tank or basin that DWSIM models as an ideal CSTR, when the card asks about mixing quality, dead volume,
   short-circuiting or where to put the inlet and outlet, call vessel_residence_time_cfd (OpenFOAM) and report
   σ²/τ², N and the conversion of the real RTD next to the ideal CSTR. It has no impeller.
   For an AGITATED tank (impeller power, power number, W/m³, motor size, mixing time) with a Rushton turbine, call
   stirred_tank_cfd with the user's T, H, D, C, speed and ρ, μ (never invent the geometry or the speed); report Np,
   P and P/V with the seal only if verified, and give θ95 as a correlation. It takes ~2 h: say so before calling.
11. DWSIM's conversion reactor at 100 % has NO dissociation: for a flame, burner, furnace or combustion temperature
   call adiabatic_flame (equilibrium with CO, OH, NO; its complete-combustion T is the upper bound) and report both.
   For ignition delay, blow-out / flame stability vs residence time, CO or NO from a combustor, or any detailed
   mechanism, call kinetic_reactor_network (batch, pfr or wsr). Simple power-law kinetics with real thermo stay in
   DWSIM's kinetic CSTR/PFR. Seal only if evidence.verified; outside GRI-Mech's range say so.
12. Purchase cost (equipment_cost) is NOT installed cost. For capital, NPV, IRR, payback or "is it worth it", price each
   unit with equipment_cost, pass purchase_cost_usd, base_purchase_cost_usd and platforms_ladders_purchase_cost_usd to
   capital_investment, then give total_depreciable_capital_usd, working_capital_usd and land_usd to cash_flow_analysis.
   Capital factors, prices, revenue, tax rate, discount rate and depreciation are the user's: if the card does not give
   them, report the gap (rule 9), never assume them.
13. Relief valves, control valves, fittings, gas lines with ΔP > 10 % of P1 and two-phase lines have their own tools
   (relief_valve_sizing, control_valve_sizing, fittings_resistance, compressible_gas_line, two_phase_pressure_drop).
   The relieving rate and overpressure come from the scenario, and FL/Fd/xT from the valve style: they are the user's.
   For two-phase ΔP report the range across methods, not one number.
14. Shell-side h/ΔP (shell_side_kern to estimate, shell_side_bell_delaware to rate a defined exchanger), boiling
   (nucleate_boiling) and condensation (film_condensation) feed U for heat_exchanger_area. Report the spread between
   correlations and the fraction of the critical heat flux in use. Bell-Delaware geometry (Dotl, baffle cut, tube
   count, clearances) is the user's exchanger: if a TEMA default was used, say so from assumptions.
15. Heat integration: pinch_analysis on the stream table (supply/target T and CP or duty of every heater, cooler
   and exchanger side). ΔTmin is a design choice: state it and show a sweep. Report Q_hot_min, Q_cold_min, the
   pinch and, if the current utilities are known, the saving; never claim a network is optimal from targets alone.
   With a built flowsheet use pinch_from_flowsheet (extracts heaters/coolers/exchangers, segments CP on phase change,
   takes heaters/coolers as the current utilities): show the extracted stream table so the user can check it, say
   which units were not extracted (column condensers/reboilers, reactors) and that supply/target T are today's.
16. Minimum energy (or reflux) of a column for given purities/recoveries, or the marginal cost of a spec:
   eo_column_optimize (equation-oriented, IDAES + ipopt). Ask for every feed value, the trays, feed tray, pressure and
   specs (no defaults). Ideal mixtures: activity_model="ideal". Non-ideal (the ideal call refuses with UNIFAC, or water/
   alcohols/ketones/acids): activity_model="NRTL" with binary parameters from ONE labelled source per pair: the user's
   (with citation) > the result of vle_binary_regression on the user's/published VLE data (pass its whole result dict
   as binary_parameters: chain vle_binary_regression → eo_column_optimize) > thermo's ChemSep NRTL databank (automatic
   when neither is given). If the tool says a pair has no parameters, ask for VLE data and run vle_binary_regression;
   never invent parameters. Report RR, boil-up, Q_reboiler and the multipliers as "MW per unit of spec", the property
   sources, the parameter source of every pair, the UNIFAC cross-check, the azeotropes found with those parameters and
   any warnings (extrapolation, Psat of the regression), with the seal only if verified. Excess enthalpy (h^E): leave
   excess_enthalpy unset unless the user asks; the default is CONDITIONAL (the owner's policy): the term is added only
   when every pair's parameters come from a joint VLE + h^E regression that passed its verdict, and is OFF otherwise
   (ChemSep bank, VLE-only regression, user values, a joint fit that failed). ALWAYS tell the user which it was and why
   (properties.excess_enthalpy.decision.reason; when it is off by default the first entry of `warnings` says which
   pairs lack calorimetric support), where h^E comes from (properties.excess_enthalpy.source.text) and how much of
   Q_reb it is or would be (first_order_share_of_reboiler_duty). The way to enable it properly is calorimetric data:
   ask for heat-of-mixing data and run vle_binary_regression with excess_enthalpy_sets. excess_enthalpy=True forces it
   on VLE-only parameters only if the user explicitly asks, and you must say it is not validated by calorimetry and
   can have the wrong sign (ethanol-water: measured h^E negative below ~363 K, VLE-only NRTL positive; Q_reb −5 %).
   Never present a duty without saying whether h^E is in it. If the specs cross an
   azeotrope the tool refuses: explain it and offer entrainer/pressure-swing schemes on the DWSIM column
   (optimize_design_multi).
17. Surrogate optimisation of a built DWSIM flowsheet (slow or many-variable flowsheets, or the user asks for a
   surrogate/neural network/OMLT): optimize_design_surrogate. Ask for the bounds of every variable, the objective and
   every limit (no defaults; prices only from the user), and the accuracy wanted for each constrained readout
   ("tolerance", e.g. 0.002 on a mole fraction). Report the sample count, the held-out error, the surrogate optimum,
   the DWSIM value at that point and the error there; give the DWSIM numbers as the answer. If status is
   "not_verified", say so and why, never quote the surrogate value as the result; offer more samples, narrower bounds
   or optimize_design_multi. Say the surrogate is valid only inside the bounds.
18. Process control (step test, PID tuning, loop performance): fopdt_identify on the user's step-test data
   (time, output, Δu and step time are theirs: ask, never invent), then pid_tuning with K, τ, θ (or a full
   process_model, reduced by the half rule) and pid_closed_loop to judge given settings. Recommend SIMC unless the
   user asks otherwise; show the comparison table (Kc, τI, τD, Ms, GM, PM, IAE setpoint/load) and say which rules are
   out of their validity range. State τc and that it is a tuning choice (default τc = θ). Say a PID needs a
   second-order model (τ2 > θ) to pay off. Quote "verified" only when the tool says so; an unstable loop is a result.
19. P&ID (instrumentation, control loops, relief/check/isolation valves, DEXPI P&ID): pid_from_hazop on the solved
   flowsheet (or dexpi_path for a DEXPI Process file). Its rules and the HAZOP rows decide every element; show the
   loop table and the valve table in Spanish with each tag, its rule/source and the HAZOP rows it answers, plus
   no_instrumentadas and assumptions. Setpoints, alarm limits, PSV set pressures and relief loads are the user's:
   leave them "por definir" (size PSVs with relief_valve_sizing, valves with control_valve_sizing, tune loops with
   pid_tuning). Anything you add beyond the tool's output is a "propuesta" you list apart; never put it in the
   table as if the tool produced it. Quote "verified" only when the tool says so (round trip + official XSD).
20. LOPA and SIL (protection layers, risk reduction, SIL of a trip): lopa_from_hazop on the same flowsheet or DEXPI
   file. The tolerable frequency is the owner's or the company's risk criterion: ASK for it (one value, by HAZOP
   severity, or per scenario); never pick one. Pass example_criterion only if the user asks for an example, and say
   it is an EXAMPLE. Conditional modifiers (ignition, occupancy, fatality), the minutes between alarm and consequence,
   operator opportunities per year and frequencies of causes outside the table are user data: ask, never invent.
   Show the LOPA sheet in Spanish (scenario, causes with frequency and source, credited layers and the ones denied with
   their independence rule, ΠCM, mitigated vs tolerable frequency, state) and the SIF table (required PFD, RRF, SIL,
   demand mode). Say the tables are generic literature values, not site data, and list scenarios left "incompleto".
21. Consequence analysis (what happens if the protection fails: source term, toxic/flammable dispersion, pool fire,
   fireball, jet fire, explosion overpressure): release_source_term → gas_dispersion | fire_radiation |
   explosion_overpressure. Weather (wind, stability or insolation/cloud, temperature, humidity), hole size, pressure,
   inventory, cloud mass, explosion efficiency, multi-energy strength class, soil, jet angle, exposure time and END
   POINTS are user data: ASK, never invent. Offer example_endpoints only if the user asks for an example and say it
   is an EXAMPLE (EPA RMP). If gas_dispersion says the cloud is dense, quote the Britter-McQuaid distances, never
   Gaussian ones; thresholds outside its curves stay "fuera de rango". Baker-Strehlow-Tang is not available. Show in Spanish the tables
   (distance vs concentration / kW/m² / kPa, distances to each end point, receptor probabilities) with each model's
   source and the assumptions. Always say it is preliminary screening, not a substitute for Phast/EFFECTS in a real
   QRA. To feed a LOPA row, pass the tool's para_lopa.overrides_lopa to lopa_from_hazop (severity labelled
   "calculada") only with the user's severity_criterion; a suggested probability modifier needs the owner's OK.

22. Pipe networks and pump systems (loops, several sources/sinks, reservoirs or fixed-pressure boundaries, pumps
   in series/parallel, operating point vs system curve, NPSH, check/PRV/FCV valves): pipe_network. Use
   pipe_hydraulics for ONE pipe of a solved stream; pipe_network for anything with a junction. Pipe lengths,
   diameters (or NPS + schedule), roughness (or a material name from the fluids tables, then say it is a design
   value), node elevations and demands, reservoir/tank levels, pump curves (points or {h0_m, r, n}), efficiencies,
   NPSHr, atmospheric and vapour pressure are user data: ASK, never invent. Show in Spanish the pipe table (flow,
   velocity, Re, f, head loss), the node table (head, pressure, continuity residual), each pump's duty point, head,
   hydraulic and shaft power, NPSHa vs NPSHr, the system curve and the combined curve for parallel pumps, the
   valve states, and the evidence (mass residual vs its tolerance, Hardy Cross agreement). Quote "verified" only
   when the tool says so; list supuestos and avisos (curve extrapolation, transition Re, cavitation).

23. Binary interaction parameters from the user's VLE data (isothermal Pxy, isobaric Txy or P-T-x-y): use
   vle_binary_regression. It fits NRTL, UNIQUAC and Wilson (one or all), with 95 % intervals, correlation, AAD in T or P
   and y, residuals per point, the azeotrope predicted vs the data's, Redlich-Kister/Herington area, Fredenslund point
   and Van Ness tests, and the UNIFAC Dortmund prediction for the same points. The experimental points, their
   uncertainties (sigma), the Antoine constants the user trusts and any reported azeotrope are user data: ASK, never
   invent or "recall" a data set. NRTL alpha, ideal gas, default sigma, r/q from UNIFAC groups and Wilson volumes are
   model choices the tool lists in supuestos: say them. Show in Spanish the parameter table (value ± IC95, units,
   J/mol and cal/mol), the correlation, the statistics, the consistency verdict per set (consistente / inconsistente /
   dudoso, with each test's number and criterion), the azeotrope comparison and the UNIFAC comparison. When the
   parameters are strongly correlated (|ρ| > 0.98) say the individual values are not identifiable with these data.
   When the user ALSO has calorimetric data (heat of mixing / excess enthalpy h^E vs x at one or more temperatures),
   pass them in excess_enthalpy_sets with temperature_dependence="a+b/T": the fit is joint, with separate weights and
   separate AADs. h^E points, their uncertainties and the weights are user data/choices (ASK; defaults are listed in
   supuestos). Report estadistica_hE per temperature, veredicto_conjunto (if representa_VLE_y_hE is false, say plainly
   that one parameter set does NOT represent VLE and h^E, with the failing criteria) and, if asked for the
   trade-off, the he_weight_scan table. Never pick the weight that "looks best" without saying so.
   Never call data "consistent" unless the tool says so, and never feed parameters from an inconsistent set to a
   design without the user's OK. To design a column with the fitted parameters, pass this tool's whole result to
   eo_column_optimize(activity_model="NRTL", binary_parameters=<result>) (NRTL model only; say the Psat source of
   the regression and the column's, Wagner-McGarry, may differ).
24. Choosing the optimizer of a built flowsheet. ONE variable: optimize_design. 2-4 variables on a flowsheet that
   solves in about a second, including integer column stages: optimize_design_multi (direct, most precise at the
   optimum). An algebraic model of the flowsheet, or 1-6 continuous variables with an accuracy to certify:
   optimize_design_surrogate. Each solve takes minutes or fails often, only tens of evaluations are affordable, or
   5-10 variables: optimize_design_bayesian. Ask for the bounds of every variable, the objective, every limit and the
   evaluation budget (no defaults for bounds, prices or limits). Report in Spanish the best point with the DWSIM
   values (engine_at_optimum), how many evaluations were spent and how many failed, which stopping rule fired, the
   model's leave-one-out prediction ± uncertainty at the optimum against DWSIM, and the status. "not_verified" with
   converged_before_budget false means the budget ran out: say there may be a better point and offer more
   evaluations. Never call the result a global optimum: one run converges to the first optimum it finds. When the
   user needs that checked, or the response may have several optima, pass restarts=1 or 2 and report
   restart_agreement (how many independent runs reached the same optimum). Repeat suitability.notes when present
   (they say when a direct optimizer would have been the better choice).
25. Real experiments (laboratory, pilot plant, plant trials: "what should I run next", design of experiments,
   reaction optimisation from data): suggest_experiments. There is NO simulator in this tool: the variables, their
   bounds, every past run with its MEASURED result, the objective and its direction, any limit and the batch size
   are user data. ASK for them; never invent, recall or simulate a measurement, and never fill a missing result.
   Present `proposals` as a table in Spanish headed as PROPUESTAS (experiments to run), with the predicted mean, its
   uncertainty and the 95 % interval of a new measurement, and say plainly that these are model predictions, not
   results. Show current_best (the best MEASURED run), the leave-one-out fit (q2, rmse, coverage) and the fitted
   noise. If mode is "initial_design" say there is no model yet, that the plan only fills the space, and how many
   runs are missing. If weak_fit is true say the proposals are exploration. If a warning suggests "log": true (the
   response spans more than a decade), tell the user and ask before turning it on. One objective only: with two (yield vs
   cost, E-factor) ask the user for weights or run it per objective, and say a Pareto front is not available.
   Categorical variables (solvent, catalyst) are not supported: run it per category. A measured failure is entered
   as "failed": true.

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
    adiabatic_eff_pct, polytropic_eff_pct, efficiency_pct, heat_added_kW, heat_removed_kW, split_ratio_0, bottoms_molar_flow_mol_s (distillation column); columns also
    stages and feed_stage, integers: the column is rebuilt with its connections). Then solve()."""
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
def pid_from_hazop(flowsheet_id: str = "", dexpi_path: str = "") -> str:
    """Minimal P&ID of a SOLVED flowsheet (or of a DEXPI Process file via dexpi_path) as a DEXPI P&ID (XML + JSON):
    deterministic level/pressure/flow/temperature loops, LV column scheme, battery-limit and rotating-equipment
    isolation, and the PSVs, check valves, alarms and trips the HAZOP rows ask for. Each element has an ISA-5.1 tag
    and its trace (rule, source, HAZOP rows); setpoints and limits stay "por definir". Round trip + official DEXPI XSD
    give `verified`. Use when the card or user asks for a P&ID, instrumentation, control loops or DEXPI P&ID."""
    out = _pid(flowsheet_id or None, dexpi_path or None)
    out.pop("markdown", None)
    out.pop("reglas", None)
    return json.dumps(out, ensure_ascii=False, default=float)


@beta_tool
def lopa_from_hazop(flowsheet_id: str = "", dexpi_path: str = "", tolerable_frequency: float | None = None,
                    tolerable_by_severity: dict[str, float] | None = None, example_criterion: str = "",
                    alarm_response_min: float | None = None, conditional_modifiers: list[dict[str, object]] | None = None,
                    overrides: dict[str, dict[str, object]] | None = None, min_severity: int = 3) -> str:
    """LOPA worksheet and required SIL of each trip, from the HAZOP and the minimal P&ID of a SOLVED flowsheet (or a
    DEXPI Process file). Each HAZOP row (severity >= min_severity, or with a PSV/trip) is a scenario; each cause an
    initiating event from a cited generic table (BPCS loop 0.1/yr tied to its real P&ID loop, pump seal, cooling water,
    fire, operator error per opportunity); layers are the PSVs, alarms and check valves traced to the row, with
    deterministic independence rules. tolerable_frequency (1/año) or tolerable_by_severity {"4": 1e-5} is the user's
    risk criterion (never assume it); example_criterion bstg_2007 | kenexis_tmel_1e-4 | kenexis_tmel_1e-5 applies a
    labeled EXAMPLE. alarm_response_min, conditional_modifiers [{nombre, probabilidad, fuente, filas?, nodos?}] and
    overrides {"<fila>": {frecuencia_tolerable, severidad, severidad_fuente, frecuencias {"n.k": f}, oportunidades_por_año {"n.k": n},
    modificadores, capas [{tag, tipo, pfd, componentes, fuente, aplica_a}], pfd_sif {tag: pfd}}} are user data."""
    out = _lopa(flowsheet_id or None, dexpi_path or None, tolerable_frequency, tolerable_by_severity,
                example_criterion or None, alarm_response_min, conditional_modifiers, overrides, min_severity)
    out.pop("markdown", None)
    out.pop("tablas", None)
    return json.dumps(out, ensure_ascii=False, default=float)


def _cq(out: dict) -> str:
    return json.dumps(out, ensure_ascii=False, default=float)


@beta_tool
def release_source_term(kind: str, hole_diameter_m: float | None = None, pressure_Pa: float | None = None,
                        temperature_K: float | None = None, compound_name: str = "", density_kg_m3: float | None = None,
                        molar_mass_kg_mol: float | None = None, gamma: float | None = None, liquid_head_m: float = 0.0,
                        discharge_coefficient: float | None = None, tank_area_m2: float | None = None,
                        ambient_pressure_Pa: float | None = None, coolprop_fluid: str = "",
                        pool_diameter_m: float | None = None, pool_area_m2: float | None = None,
                        wind_speed_m_s: float | None = None, schmidt: float | None = None,
                        ground_temperature_K: float | None = None, soil: str = "", time_s: float | None = None,
                        penetration_factor: float | None = None, boiling_point_K: float | None = None,
                        hvap_J_kg: float | None = None) -> str:
    """Source term of a loss of containment (Crowl & Louvar 4th ed.). kind: liquido (liquid through a hole, Cd 0.61
    unless given; tank_area_m2 + liquid_head_m add the draining time) | gas (choked or subsonic hole flow, Cd 1.0
    unless given, γ from thermo) | flash (flashing fraction; coolprop_fluid for the isenthalpic flash) | charco (pool
    evaporation; Mackay-Matsugu with the user's wind) | charco_ebullicion (cryogenic pool boiling on ground at time_s;
    ground_temperature_K and soil from TNO table 3.5, penetration_factor 8 for dry porous soil). pressure_Pa is
    ABSOLUTE. Hole, pressure, temperature, pool and soil are the user's: never assume them. Screening only."""
    return _cq(_CQ.release_source_term(kind, hole_diameter_m, pressure_Pa, temperature_K, compound_name or None,
                                       density_kg_m3, molar_mass_kg_mol, gamma, liquid_head_m, discharge_coefficient,
                                       tank_area_m2, ambient_pressure_Pa, coolprop_fluid or None,
                                       pool_diameter_m=pool_diameter_m, pool_area_m2=pool_area_m2,
                                       wind_speed_m_s=wind_speed_m_s, schmidt=schmidt,
                                       ground_temperature_K=ground_temperature_K, soil=soil or None, time_s=time_s,
                                       penetration_factor=penetration_factor, boiling_point_K=boiling_point_K,
                                       hvap_J_kg=hvap_J_kg))


@beta_tool
def gas_dispersion(mode: str, wind_speed_m_s: float | None = None, ambient_temperature_K: float | None = None,
                   stability_class: str = "", insolation: str = "", night_cloud: str = "",
                   release_rate_kg_s: float | None = None, release_mass_kg: float | None = None,
                   compound_name: str = "", molar_mass_kg_mol: float | None = None,
                   release_temperature_K: float | None = None, release_height_m: float | None = None,
                   terrain: str = "rural", thresholds: list[dict[str, object]] | None = None,
                   receptor_distance_m: float | None = None, toxic_probit: dict[str, object] | None = None,
                   exposure_time_s: float | None = None, hazop_row: int | None = None,
                   severity_criterion: list[dict[str, object]] | None = None, release_duration_s: float | None = None,
                   initial_concentration: float | None = None, cloud_density_kg_m3: float | None = None) -> str:
    """Gaussian plume (mode continua, release_rate_kg_s) or puff (instantanea, release_mass_kg), Pasquill-Gifford with
    Briggs coefficients. Weather is the user's: wind_speed_m_s, ambient_temperature_K and stability_class A-F (or
    insolation fuerte|moderada|ligera|cubierto by day / night_cloud nublado|despejado). thresholds [{nombre, ppm |
    mg_m3 | fraccion_lfl}] are the user's end points. Britter-McQuaid check: a dense ground-level cloud gets its
    distances from the Britter-McQuaid dense-gas model (release_duration_s for the plume/puff rule), never Gaussian.
    receptor_distance_m + toxic_probit {a, b, n, unidad_c, unidad_t} → probability; hazop_row + severity_criterion
    [{severidad, endpoint, distancia_min_m}] → para_lopa (calculated severity for lopa_from_hazop)."""
    return _cq(_CQ.gas_dispersion(mode, wind_speed_m_s, ambient_temperature_K, stability_class or None,
                                  insolation or None, night_cloud or None, release_rate_kg_s, release_mass_kg,
                                  compound_name or None, molar_mass_kg_mol, release_temperature_K, release_height_m,
                                  terrain, thresholds, receptor_distance_m, toxic_probit, exposure_time_s,
                                  cloud_density_kg_m3=cloud_density_kg_m3, hazop_row=hazop_row,
                                  severity_criterion=severity_criterion, release_duration_s=release_duration_s,
                                  initial_concentration=initial_concentration))


@beta_tool
def fire_radiation(kind: str, ambient_temperature_K: float | None = None, compound_name: str = "",
                   heat_of_combustion_J_kg: float | None = None, pool_diameter_m: float | None = None,
                   burning_rate_kg_m2s: float | None = None, wind_speed_m_s: float | None = None,
                   relative_humidity: float | None = None, sep_method: str = "both", mass_kg: float | None = None,
                   burst_pressure_Pa: float | None = None, pool_model: str = "sfpe",
                   burning_rate_inf_kg_m2s: float | None = None, k_beta_m: float | None = None,
                   flame_height: str = "thomas", transmissivity: float | None = None,
                   radiant_fraction: float | None = None, release_rate_kg_s: float | None = None,
                   distances_m: list[float] | None = None, endpoints_kW_m2: list[dict[str, object]] | None = None,
                   example_endpoints: str = "", exposure_time_s: float | None = None,
                   receptor_distance_m: float | None = None, hazop_row: int | None = None,
                   severity_criterion: list[dict[str, object]] | None = None, jet_model: str = "chamberlain",
                   hole_diameter_m: float | None = None, storage_pressure_Pa: float | None = None,
                   storage_temperature_K: float | None = None, jet_angle_deg: float | None = None) -> str:
    """Thermal radiation vs distance. kind charco (solid-flame pool fire; pool_model sfpe or tno, tno needs
    radiant_fraction), bola_fuego (TNO BLEVE fireball from mass_kg and burst_pressure_Pa) or chorro (jet_model
    chamberlain = TNO solid flame from release_rate_kg_s, hole_diameter_m, storage_pressure_Pa, storage_temperature_K,
    jet_angle_deg from horizontal and wind_speed_m_s; fuente_puntual = the user's radiant_fraction). ambient_temperature_K is the user's; relative_humidity gives
    the transmissivity (else τ = 1). endpoints_kW_m2 [{nombre, kW_m2}] are the user's; example_endpoints "epa_rmp"
    adds a labelled EXAMPLE. receptor_distance_m + exposure_time_s → burn-death probability; hazop_row +
    severity_criterion → para_lopa. Screening only."""
    return _cq(_CQ.fire_radiation(kind, ambient_temperature_K, compound_name or None, heat_of_combustion_J_kg,
                                  pool_diameter_m, burning_rate_kg_m2s, wind_speed_m_s, relative_humidity, sep_method,
                                  mass_kg, burst_pressure_Pa, pool_model, burning_rate_inf_kg_m2s, k_beta_m,
                                  flame_height, transmissivity, None, radiant_fraction, release_rate_kg_s, distances_m,
                                  endpoints_kW_m2, example_endpoints or None, exposure_time_s, receptor_distance_m,
                                  None, hazop_row, severity_criterion, jet_model, hole_diameter_m,
                                  storage_pressure_Pa, storage_temperature_K, jet_angle_deg))


@beta_tool
def explosion_overpressure(method: str = "tnt", flammable_mass_kg: float | None = None,
                           efficiency: float | None = None, compound_name: str = "",
                           heat_of_combustion_J_kg: float | None = None, tnt_mass_kg: float | None = None,
                           ambient_pressure_Pa: float | None = None, distances_m: list[float] | None = None,
                           endpoints_kPa: list[dict[str, object]] | None = None, example_endpoints: str = "",
                           receptor_distance_m: float | None = None, hazop_row: int | None = None,
                           severity_criterion: list[dict[str, object]] | None = None,
                           charge_energy_J: float | None = None, congested_volume_m3: float | None = None,
                           strength_class: int | None = None) -> str:
    """Blast overpressure vs distance by TNT equivalence (Kinney-Graham surface burst). flammable_mass_kg and the
    explosion efficiency are the user's (typically 1-10 %, never assumed); or tnt_mass_kg. endpoints_kPa
    [{nombre, kPa}] are the user's; example_endpoints "epa_rmp" adds 1 psi as a labelled EXAMPLE. receptor_distance_m
    → probits (lung hemorrhage, eardrum, structures, glass); hazop_row + severity_criterion → para_lopa.
    method multi_energia: TNO multi-energy (Díaz Alonso 2006 fits) from charge_energy_J or congested_volume_m3 and the
    user's strength_class 1-10 (never assumed); gives overpressure, duration and impulse. Baker-Strehlow-Tang pending."""
    return _cq(_CQ.explosion_overpressure(method, flammable_mass_kg, efficiency, compound_name or None,
                                          heat_of_combustion_J_kg, tnt_mass_kg, ambient_pressure_Pa, distances_m,
                                          endpoints_kPa, example_endpoints or None, receptor_distance_m, hazop_row,
                                          severity_criterion, charge_energy_J, congested_volume_m3, strength_class))


@beta_tool
def pipe_network(nodes: list[dict[str, object]], links: list[dict[str, object]], fluid: dict[str, object],
                 atmospheric_pressure_Pa: float | None = None, system_curve: bool = True,
                 system_curve_points: int = 11, hardy_cross_check: bool = True) -> str:
    """Steady incompressible pipe network (global gradient of Todini-Pilati, as EPANET; Hardy Cross cross-check).
    nodes: [{id, type: junction (elevation_m, demand_m3_s; negative = injection) | reservoir (head_m) | tank
    (elevation_m, level_m) | pressure (elevation_m, pressure_Pa_g)}]. links: [{id, type, from, to}] with pipe
    (length_m, diameter_m or nps_in+schedule, roughness_m or material, minor_loss_K, fittings [{name, count}] Darby 3-K,
    check_valve, hazen_williams_C, or resistance {K, n} in SI), pump (curve_points [[q_m3_s, h_m], ...] or curve
    {h0_m, r, n}, speed_ratio, efficiency or efficiency_points, npshr_m or npshr_points, centerline_elevation_m),
    prv (diameter_m, setting_pressure_Pa_g, minor_loss_K) or fcv (diameter_m, setting_flow_m3_s). Parallel pumps =
    several pumps between the same two nodes; series = pump, node, pump. fluid: {density_kg_m3, viscosity_Pa_s,
    vapor_pressure_Pa} or {compound, temperature_K} (thermo). Friction = pipe_hydraulics' Colebrook/Clamond. Returns
    flows, velocities, head losses, node heads/pressures, continuity residual per node, pump duty point, power,
    NPSHa vs NPSHr (needs atmospheric_pressure_Pa and vapour pressure), system curve per pump station and evidence.
    Every pipe, curve, elevation and demand is the user's: never assume them."""
    return _cq(_NW.pipe_network(nodes, links, fluid, atmospheric_pressure_Pa, system_curve, system_curve_points,
                                hardy_cross_check))


@beta_tool
def vle_binary_regression(components: list[str], data_sets: list[dict[str, object]], models: list[str] | None = None,
                          nrtl_alpha: float | str | None = None, temperature_dependence: str = "constante",
                          objective: str = "bubble", sigma: dict[str, float] | None = None,
                          psat: dict[str, object] | None = None, vapor_phase: str = "ideal",
                          uniquac_rq: dict[str, list[float]] | None = None,
                          wilson_volumes_m3_mol: dict[str, float] | None = None,
                          reported_azeotropes: dict[str, object] | None = None, compare_unifac: bool = True,
                          published_parameters: dict[str, object] | None = None, legendre_order: int = 4,
                          excess_enthalpy_sets: list[dict[str, object]] | None = None,
                          weights: dict[str, float] | None = None, he_weight_scan: list[float] | None = None,
                          joint_tolerances: dict[str, float] | None = None) -> str:
    """Regress binary activity-model parameters (NRTL, UNIQUAC, Wilson) from the USER's VLE data.
    components [name1, name2]; data_sets [{label, citation, points: [{T_K, P_Pa (or P_kPa), x1, y1}]}], each set
    isothermal, isobaric or P-T-x-y. models subset of NRTL/UNIQUAC/Wilson (default all). nrtl_alpha number or "ajustar"
    (default 0.3, labelled). temperature_dependence "constante" (energy parameters) or "a+b/T". objective "bubble"
    (T or P plus y residuals) or "ml" (maximum likelihood, errors in T, P, x, y). sigma {T_K, P_Pa or P_rel, x1, y1}.
    psat {name: {"antoine": {A, B, C, log, P_unit, T_unit}} or {"antoine_ext": [A1..A7] Aspen PLXANT}}, default
    chemicals. vapor_phase "ideal" or "virial" (Tsonopoulos). uniquac_rq {name: [r, q]} (default UNIFAC groups);
    wilson_volumes_m3_mol {name: V}. reported_azeotropes {label: {x1, T_K or P_Pa}}. published_parameters {model:
    {a12, a21, b12, b21, alpha} or {d12_J_mol, d21_J_mol}} to compare. Returns parameters ± IC95, correlation, AAD,
    residuals, azeotrope, consistency tests (Redlich-Kister/Herington, Fredenslund, Van Ness), UNIFAC Dortmund
    comparison and an independent re-evaluation with thermo. Never invent data points.
    Calorimetry (optional): excess_enthalpy_sets [{label, citation, T_K, points: [{x1, hE_J_mol, optional T_K,
    sigma_hE_J_mol}]}] are fitted TOGETHER with the VLE (h^E = −R·T²·∂(g^E/RT)/∂T of the same parameters; use
    temperature_dependence "a+b/T"). weights {"vle", "hE"} multiply the two blocks of the objective (default 1 and 1);
    sigma also takes hE_J_mol and hE_rel; he_weight_scan [w, ...] repeats the fit with other h^E weights (0 = VLE only)
    and returns the trade-off; joint_tolerances overrides the verdict's limits. Extra output per model: estadistica_hE
    (AAD, bias, wrong-sign points, per set and temperature), residuos_hE, veredicto_conjunto.representa_VLE_y_hE."""
    return _cq(_VLE.vle_binary_regression(components, data_sets, models, nrtl_alpha, temperature_dependence,
                                          objective, sigma, psat, vapor_phase, uniquac_rq, wilson_volumes_m3_mol,
                                          reported_azeotropes, compare_unifac, published_parameters, legendre_order,
                                          excess_enthalpy_sets, weights, he_weight_scan, joint_tolerances))

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
def tubular_reactor_field(volumetric_flow_m3_s: float, tube_diameter_m: float, n_tubes: int, diffusivity_m2_s: float,
                          tube_length_m: float | None = None, volume_m3: float | None = None,
                          k_1_s: float | None = None, A_1_s: float | None = None, E_J_mol: float | None = None, T_K: float | None = None,
                          density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None, reynolds: float | None = None,
                          nz: int = 300, nr: int = 30) -> str:
    """Real tubular reactor vs ideal PFR: first-order isothermal reaction in a bundle of n_tubes laminar tubes solved as a 2-D
    field with FEniCSx (separate process). Needs volumetric flow at reactor conditions, tube diameter, n_tubes, tube_length_m OR
    volume_m3, diffusivity_m2_s (diffusivity_estimate) and k_1_s or A_1_s+E_J_mol+T_K; pass density/viscosity so Re confirms
    laminar flow. Returns X_plug_flow (ideal PFR), X_laminar (with radial diffusion), X_laminar_segregated, the penalty in
    points and evidence (verified only if plug reproduces 1−e^(−kτ), mesh-independent, within bounds and Re < 2300)."""
    return json.dumps(TT.tubular_reactor_field(volumetric_flow_m3_s, tube_diameter_m, n_tubes, diffusivity_m2_s, tube_length_m, volume_m3,
                                               k_1_s, A_1_s, E_J_mol, T_K, density_kg_m3, viscosity_Pa_s, reynolds, nz, nr), default=float)


@beta_tool
def adiabatic_flame(fuel: str, equivalence_ratio: float, T_K: float = 298.15, P_Pa: float = 101325.0,
                    oxidizer: str = "O2:1, N2:3.76", mechanism: str = "gri30.yaml", laminar_flame_speed: bool = False,
                    report_species: list[str] | None = None) -> str:
    """Adiabatic flame of a premixed mixture (Cantera, GRI-Mech 3.0 by default). fuel/oxidizer as compositions in
    mechanism species ('CH4', 'CH4:0.9, C2H6:0.1'; air 'O2:1, N2:3.76'). Returns T_adiabatic_equilibrium_K (with
    dissociation), T_adiabatic_complete_combustion_K (100 % conversion reactor: upper bound), dissociation_penalty_K,
    products and, with laminar_flame_speed=True (~20 s), the laminar flame speed. Check evidence.verified."""
    return json.dumps(TK.adiabatic_flame(fuel, equivalence_ratio, T_K, P_Pa, oxidizer, mechanism, laminar_flame_speed,
                                         report_species), default=float)


@beta_tool
def kinetic_reactor_network(reactor: str, feed: str, T_K: float, P_Pa: float, mechanism: str = "gri30.yaml",
                            energy: str = "adiabatic", end_time_s: float | None = None,
                            residence_times_s: list[float] | None = None, key_species: list[str] | None = None,
                            report_species: list[str] | None = None, initial_state: str = "ignited",
                            custom_species: list[dict[str, object]] | None = None,
                            custom_reactions: list[dict[str, object]] | None = None) -> str:
    """Detailed-kinetics reactor (Cantera). reactor 'batch_constant_volume' | 'batch_constant_pressure' | 'pfr'
    (end_time_s = residence time) | 'wsr' (CSTR at steady state for each residence_times_s, 1-20 values). feed as
    'CH4:1, O2:2, N2:7.52'. energy 'adiabatic' | 'isothermal'. Gives ignition delay (adiabatic batch), blow-out, CO, NO,
    conversions of key_species. Own kinetics (isothermal only): custom_species [{'name','composition':{'C':4,'H':10}}],
    custom_reactions [{'equation':'A => B','A':1e8,'b':0,'Ea_J_mol':1e5,'orders':{'A':1}}], C in mol/m³. Check
    evidence.verified."""
    return json.dumps(TK.kinetic_reactor_network(reactor, feed, T_K, P_Pa, mechanism, energy, end_time_s, residence_times_s,
                                                 key_species, report_species, initial_state, custom_species,
                                                 custom_reactions), default=float)


@beta_tool
def nonisothermal_tubular_reactor(volumetric_flow_m3_s: float, tube_diameter_m: float, n_tubes: int,
                                  inlet_concentration_mol_m3: float, inlet_T_K: float, A_1_s: float, E_J_mol: float,
                                  dH_J_mol: float, density_kg_m3: float, cp_J_kgK: float, diffusivity_m2_s: float,
                                  thermal_conductivity_W_mK: float, wall: dict, velocity_profile: str, fluid_model: str,
                                  tube_length_m: float | None = None, volume_m3: float | None = None,
                                  axial_diffusivity_m2_s: float | None = None,
                                  axial_thermal_conductivity_W_mK: float | None = None,
                                  viscosity_Pa_s: float | None = None, nz: int = 300, nr: int = 30,
                                  mesh_study: bool = True) -> str:
    """Non-isothermal tubular reactor (coupled energy balance) as a 2-D FEniCSx field vs the 1-D non-isothermal PFR.
    First-order Arrhenius A_1_s/E_J_mol, dH_J_mol (< 0 exothermic), feed C and T, ρ and cp, radial diffusivity and
    conductivity (packed bed: effective D_er, λ_er), wall {'mode':'fixed_T','T_wall_K'} | {'mode':'coolant','U_W_m2K',
    'T_coolant_K'} | {'mode':'adiabatic'}, velocity_profile 'laminar'|'plug', fluid_model 'liquid'|'ideal_gas'; all
    from the user, no defaults. Returns X, outlet T, hot spot (T, z, r), radial T profiles, the PFR's X/T/hot spot,
    divergence, the bundle energy balance, assumptions and evidence (energy closure, mesh independence, Re)."""
    return json.dumps(TT.nonisothermal_tubular_reactor(volumetric_flow_m3_s, tube_diameter_m, n_tubes, inlet_concentration_mol_m3,
                                                       inlet_T_K, A_1_s, E_J_mol, dH_J_mol, density_kg_m3, cp_J_kgK,
                                                       diffusivity_m2_s, thermal_conductivity_W_mK, wall, velocity_profile,
                                                       fluid_model, tube_length_m, volume_m3, axial_diffusivity_m2_s,
                                                       axial_thermal_conductivity_W_mK, viscosity_Pa_s, nz, nr, mesh_study),
                      default=float)


@beta_tool
def vessel_residence_time_cfd(length_m: float, width_m: float, height_m: float, volumetric_flow_m3_s: float,
                              kinematic_viscosity_m2_s: float, inlet: dict, outlet: dict,
                              cells_per_max_dimension: int = 30, mesh_study: bool = True,
                              first_order_k_1_s: float | None = None) -> str:
    """Real tank vs ideal CSTR (OpenFOAM, separate process, no impeller): box vessel with one inlet and one outlet
    opening {'face': 'x-'..'z+', 'center_m': [c1, c2], 'size_m': [s1, s2]}. Returns τ, ⟨age⟩/τ, σ²/τ², tanks in
    series N, stagnant volume, diagnosis and first-order conversion vs CSTR/PFR. evidence: converged flow, outlet
    age = τ within 3 % (theorem), mesh study. About a minute per call."""
    return json.dumps(TT.vessel_residence_time_cfd(length_m, width_m, height_m, volumetric_flow_m3_s,
                                                   kinematic_viscosity_m2_s, inlet, outlet, cells_per_max_dimension,
                                                   mesh_study, first_order_k_1_s), default=float)


@beta_tool
def stirred_tank_cfd(tank_diameter_m: float, liquid_height_m: float, impeller_diameter_m: float, clearance_m: float,
                     rotational_speed_rps: float, density_kg_m3: float, viscosity_Pa_s: float, n_baffles: int = 4,
                     baffle_width_m: float | None = None, blade_thickness_m: float | None = None,
                     cells_across_tank: int = 64, mesh_study: bool = True) -> str:
    """Baffled tank with a 6-blade Rushton turbine by MRF CFD (OpenFOAM, separate process). Returns the power number
    Np, power P, P/V, W/kg, Re, Fr, tip speed and θ95 from Grenville's correlation (labelled, H = T only). evidence:
    torque balance impeller vs walls, steady Np, 3 meshes within a 10 % band, checkMesh, Re > 1e4, Np in 4-6.
    About 2 h per call with mesh study."""
    return json.dumps(TT.stirred_tank_cfd(tank_diameter_m, liquid_height_m, impeller_diameter_m, clearance_m,
                                          rotational_speed_rps, density_kg_m3, viscosity_Pa_s, n_baffles, baffle_width_m,
                                          blade_thickness_m, cells_across_tank, mesh_study), default=float)


@beta_tool
def relief_valve_sizing(service: str, relieving_rate_kg_s: float, set_pressure_Pa_g: float, overpressure_fraction: float,
                        back_pressure_Pa_g: float, T_K: float | None = None, MW: float | None = None, Z: float | None = None,
                        k: float | None = None, density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None,
                        valve_type: str = "conventional", rupture_disk_upstream: bool = False, Kd: float | None = None) -> str:
    """Relief valve by API 520 Part I + API 526 orifice letter. service 'gas' (T_K, MW, Z, k), 'steam' (T_K) or
    'liquid' (density_kg_m3, viscosity_Pa_s optional). GAUGE set and back pressures; overpressure_fraction 0.10 single
    valve, 0.16 multiple, 0.21 fire. valve_type 'conventional' | 'balanced_bellows'. Returns required_area_mm2, orifice,
    valves and evidence (printed API equation recomputed)."""
    return json.dumps(TF.relief_valve_sizing(service, relieving_rate_kg_s, set_pressure_Pa_g, overpressure_fraction,
                                             back_pressure_Pa_g, T_K, MW, Z, k, density_kg_m3, viscosity_Pa_s, valve_type,
                                             rupture_disk_upstream, Kd), default=float)


@beta_tool
def control_valve_sizing(fluid: str, P1_Pa: float, P2_Pa: float, FL: float, Fd: float, flow_m3_s: float,
                         density_kg_m3: float | None = None, vapor_pressure_Pa: float | None = None,
                         critical_pressure_Pa: float | None = None, viscosity_Pa_s: float | None = None,
                         T_K: float | None = None, MW: float | None = None, gamma: float | None = None, Z: float | None = None,
                         xT: float | None = None, pipe_inlet_diameter_m: float | None = None,
                         pipe_outlet_diameter_m: float | None = None, valve_diameter_m: float | None = None) -> str:
    """Control valve Kv/Cv by IEC 60534-2-1. 'liquid': flow at inlet conditions + density, vapor and critical pressure,
    viscosity. 'gas': flow in normal m³/s (0 °C, 1 atm) + T_K, MW, gamma, Z, viscosity, xT. FL, Fd, xT are the valve
    style's (user/catalog). Reducers: all three diameters. Returns Kv, Cv, choked, cavitation/flashing, evidence."""
    return json.dumps(TF.control_valve_sizing(fluid, P1_Pa, P2_Pa, FL, Fd, flow_m3_s, density_kg_m3, vapor_pressure_Pa,
                                              critical_pressure_Pa, viscosity_Pa_s, T_K, MW, gamma, Z, xT,
                                              pipe_inlet_diameter_m, pipe_outlet_diameter_m, valve_diameter_m), default=float)


@beta_tool
def fittings_resistance(fittings: list[dict[str, object]], inner_diameter_m: float, Re: float, method: str = "Darby3K") -> str:
    """K of fittings and valves (Darby 3-K or Hooper 2-K, depends on Re and size). fittings = [{'name', 'count'}];
    fittings=[] lists the catalog names. Pass K_total as fittings_K to pipe_hydraulics."""
    return json.dumps(TF.fittings_resistance(fittings, inner_diameter_m, Re, method), default=float)


@beta_tool
def compressible_gas_line(P1_Pa: float, T_K: float, MW: float, Z: float, k: float, viscosity_Pa_s: float,
                          inner_diameter_m: float, length_m: float, mass_flow_kg_s: float | None = None,
                          P2_Pa: float | None = None, roughness_m: float = 4.57e-5) -> str:
    """Isothermal compressible gas line (ΔP > 10 % of P1): mass_flow_kg_s → P2, or P2_Pa → mass flow. Returns Mach,
    choke pressure, choked flag, error of the incompressible formula and evidence (momentum ODE)."""
    return json.dumps(TF.compressible_gas_line(P1_Pa, T_K, MW, Z, k, viscosity_Pa_s, inner_diameter_m, length_m,
                                               mass_flow_kg_s, P2_Pa, roughness_m), default=float)


@beta_tool
def two_phase_pressure_drop(mass_flow_kg_s: float, quality: float, liquid_density_kg_m3: float, gas_density_kg_m3: float,
                            liquid_viscosity_Pa_s: float, gas_viscosity_Pa_s: float, surface_tension_N_m: float,
                            inner_diameter_m: float, length_m: float, roughness_m: float = 4.57e-5) -> str:
    """Two-phase frictional ΔP, horizontal, constant quality: Friedel, Müller-Steinhagen-Heck, Chisholm,
    Lockhart-Martinelli, Whalley's recommendation, spread between methods and Taitel-Dukler regime."""
    return json.dumps(TF.two_phase_pressure_drop(mass_flow_kg_s, quality, liquid_density_kg_m3, gas_density_kg_m3,
                                                 liquid_viscosity_Pa_s, gas_viscosity_Pa_s, surface_tension_N_m,
                                                 inner_diameter_m, length_m, roughness_m), default=float)


@beta_tool
def shell_side_kern(mass_flow_kg_s: float, density_kg_m3: float, viscosity_Pa_s: float, thermal_conductivity_W_mK: float,
                    cp_J_kgK: float, shell_diameter_m: float, baffle_spacing_m: float, n_baffles: int, tube_od_m: float,
                    tube_pitch_m: float, layout: str = "triangular", wall_viscosity_Pa_s: float | None = None) -> str:
    """Shell-side h and ΔP by Kern (preliminary; no leakage/bypass). layout 'triangular' | 'square'. Also gives the
    ideal tube-bank h (Žukauskas) as the ceiling. evidence: Kern's Re range and TEMA pitch/baffle-spacing limits."""
    return json.dumps(TH.shell_side_kern(mass_flow_kg_s, density_kg_m3, viscosity_Pa_s, thermal_conductivity_W_mK,
                                         cp_J_kgK, shell_diameter_m, baffle_spacing_m, n_baffles, tube_od_m, tube_pitch_m,
                                         layout, wall_viscosity_Pa_s), default=float)


@beta_tool
def shell_side_bell_delaware(mass_flow_kg_s: float, density_kg_m3: float, viscosity_Pa_s: float,
                             thermal_conductivity_W_mK: float, cp_J_kgK: float, shell_diameter_m: float,
                             tube_od_m: float, tube_pitch_m: float, layout_angle: int, baffle_cut_fraction: float,
                             baffle_spacing_m: float, n_baffles: int, outer_tube_limit_diameter_m: float,
                             n_tubes: int | None = None, tube_passes: int = 1,
                             shell_baffle_clearance_m: float | None = None,
                             tube_baffle_clearance_m: float | None = None, sealing_strip_pairs: int = 0,
                             inlet_baffle_spacing_m: float | None = None, outlet_baffle_spacing_m: float | None = None,
                             wall_viscosity_Pa_s: float | None = None) -> str:
    """Shell-side h and ΔP by Bell-Delaware (design grade): ideal bank × Jc, Jl, Jb, Js, Jr; ΔP crossflow/windows/
    end zones with Rl, Rb, Rs. layout_angle 30 | 45 | 90; baffle_cut_fraction = cut/Ds; Dotl = outer tube limit.
    Clearances default to TEMA (listed in assumptions). Includes Kern on the same shell. evidence: chart J factors vs
    HEDH fits (10 %), Žukauskas vs ESDU ideal bank (30 %), geometry inside the charts."""
    return json.dumps(TH.shell_side_bell_delaware(mass_flow_kg_s, density_kg_m3, viscosity_Pa_s,
                                                  thermal_conductivity_W_mK, cp_J_kgK, shell_diameter_m, tube_od_m,
                                                  tube_pitch_m, layout_angle, baffle_cut_fraction, baffle_spacing_m,
                                                  n_baffles, outer_tube_limit_diameter_m, n_tubes, tube_passes,
                                                  shell_baffle_clearance_m, tube_baffle_clearance_m,
                                                  sealing_strip_pairs, inlet_baffle_spacing_m,
                                                  outlet_baffle_spacing_m, wall_viscosity_Pa_s), default=float)


@beta_tool
def nucleate_boiling(P_Pa: float, critical_pressure_Pa: float, MW: float, T_sat_K: float, liquid_density_kg_m3: float,
                     vapor_density_kg_m3: float, liquid_viscosity_Pa_s: float, liquid_conductivity_W_mK: float,
                     liquid_cp_J_kgK: float, latent_heat_J_kg: float, surface_tension_N_m: float,
                     heat_flux_W_m2: float | None = None, wall_superheat_K: float | None = None,
                     surface_roughness_um: float = 1.0, rohsenow_Csf: float | None = None,
                     rohsenow_n: float | None = None) -> str:
    """Nucleate pool boiling h (Cooper recommended; Mostinski, Stephan-Abdelsalam, Rohsenow if Csf/n given), spread,
    critical heat flux (Zuber, Lienhard-Dhir) and fraction of it in use. Give heat_flux_W_m2 OR wall_superheat_K."""
    return json.dumps(TH.nucleate_boiling(P_Pa, critical_pressure_Pa, MW, T_sat_K, liquid_density_kg_m3, vapor_density_kg_m3,
                                          liquid_viscosity_Pa_s, liquid_conductivity_W_mK, liquid_cp_J_kgK, latent_heat_J_kg,
                                          surface_tension_N_m, heat_flux_W_m2, wall_superheat_K, surface_roughness_um,
                                          rohsenow_Csf, rohsenow_n), default=float)


@beta_tool
def film_condensation(geometry: str, T_sat_K: float, liquid_density_kg_m3: float, vapor_density_kg_m3: float,
                      liquid_conductivity_W_mK: float, liquid_viscosity_Pa_s: float, liquid_cp_J_kgK: float,
                      latent_heat_J_kg: float, wall_T_K: float | None = None, length_m: float | None = None,
                      tube_od_m: float | None = None, n_rows: int = 1, mass_flow_kg_s: float | None = None,
                      quality: float | None = None, tube_id_m: float | None = None, P_Pa: float | None = None,
                      critical_pressure_Pa: float | None = None) -> str:
    """Pure-vapor condensation h. 'vertical_plate'/'vertical_tube' (wall_T_K, length_m), 'horizontal_tube' (wall_T_K,
    tube_od_m, n_rows): Nusselt with h'fg. 'in_tube' (mass_flow_kg_s per tube, quality, tube_id_m, P_Pa,
    critical_pressure_Pa): Shah and Boyko-Kruzhilin. evidence flags wavy/turbulent films."""
    return json.dumps(TH.film_condensation(geometry, T_sat_K, liquid_density_kg_m3, vapor_density_kg_m3,
                                           liquid_conductivity_W_mK, liquid_viscosity_Pa_s, liquid_cp_J_kgK, latent_heat_J_kg,
                                           wall_T_K, length_m, tube_od_m, n_rows, mass_flow_kg_s, quality, tube_id_m, P_Pa,
                                           critical_pressure_Pa), default=float)


@beta_tool
def pinch_analysis(streams: list[dict], dT_min_K: float, dT_min_sweep_K: list[float] | None = None,
                   current_hot_utility_W: float | None = None, current_cold_utility_W: float | None = None) -> str:
    """Pinch analysis (Problem Table Algorithm): minimum hot/cold utility, pinch, heat recovery, grand composite and
    composite curves, minimum units, streams crossing the pinch, ΔTmin sweep and saving vs current utilities.
    streams = [{'name', 'T_supply_K', 'T_target_K', 'CP_W_K' or 'heat_load_W', optional 'dT_contribution_K'}].
    evidence: independent shifted composite curves touch at the pinch without crossing; energy balance closes."""
    return json.dumps(TEN.pinch_analysis(streams, dT_min_K, dT_min_sweep_K, current_hot_utility_W,
                                        current_cold_utility_W), default=float)


@beta_tool
def pinch_from_flowsheet(flowsheet_id: str, dT_min_K: float, dT_min_sweep_K: list[float] | None = None,
                         linearization_tol_pct: float = 1.0, dT_contribution_K: dict[str, float] | None = None,
                         exclude_units: list[str] | None = None) -> str:
    """Pinch analysis of a built DWSIM flowsheet: re-solves it, turns every heater, cooler and exchanger side into a
    process stream (exchanger → cooler chained as one), segments CP from DWSIM flashes where cp varies or the phase
    changes, counts heaters/coolers as current hot/cold utility and runs pinch_analysis. dT_min_K is the user's
    choice (ask). Returns extracted_streams (show them), unit_balance (ΔH vs DWSIM duty), MER targets, pinch,
    current_vs_target saving, linear_cp_targets, assumptions, warnings (units not extracted) and evidence."""
    return json.dumps(TEN.pinch_from_flowsheet(flowsheet_id, dT_min_K, dT_min_sweep_K, linearization_tol_pct,
                                               dT_contribution_K, exclude_units), default=float)


@beta_tool
def equipment_cost(equipment: str, size: dict[str, object], options: dict[str, object] | None = None, year: int = 2023,
                   cepci: float | None = None) -> str:
    """Purchase cost (USD f.o.b., not installed) by Seider et al. via IDAES. equipment: heat_exchanger {area_m2,
    shell_pressure_Pa} | pump {power_W, deltaP_Pa, density_kg_m3, flow_m3_s} | compressor {power_W} | vessel
    {diameter_m, length_m, vertical, number_of_trays (optional, Seider tray cost; tray_type Sieve|Valve|BubbleCap)} |
    packing {volume_m3, price_usd_per_m3 (your input; no verified public table)}. options as listed by costing_options.
    year 2015-2023, else pass cepci."""
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
def bare_module_factors() -> str:
    """Bare-module factors F_BM by equipment type (Seider 2017 via BioSTEAM): the `type` keys capital_investment accepts."""
    return json.dumps(TE.bare_module_factors(), default=float)


@beta_tool
def capital_investment(equipment: list[dict[str, object]], factors: dict[str, float], location_factor: float = 1.0) -> str:
    """Installed (bare-module) cost and total capital investment. equipment: [{name, type (bare_module_factors key) or
    F_BM, purchase_cost_usd, base_purchase_cost_usd, platforms_ladders_purchase_cost_usd (both from equipment_cost),
    basis}]. factors, all required fractions given by the user (0 if not applicable): site_preparation,
    service_facilities, allocated_costs, contingency_and_fee, land, royalties, startup, working_capital. Check evidence."""
    return json.dumps(TE.capital_investment(equipment, factors, location_factor), default=float)


@beta_tool
def cash_flow_analysis(depreciable_capital_usd: float, revenue_usd_y: float, operating_cost_usd_y: float,
                       plant_life_years: int, income_tax_rate: float, discount_rate: float, depreciation: str,
                       construction_schedule: list[float], working_capital_usd: float = 0.0, land_usd: float = 0.0,
                       startup_years: float = 0.0, startup_revenue_fraction: float = 1.0,
                       startup_operating_cost_fraction: float = 1.0) -> str:
    """Discounted cash flow (real, no loans): NPV, IRR, payback, discounted payback, break-even revenue, yearly table.
    depreciable_capital_usd = C_TDC; operating cost without depreciation; rates as fractions; depreciation 'MACRS7',
    'SL10', 'DDB10', 'SYD10'; construction_schedule fractions summing to 1. Economic inputs are the user's. Check evidence."""
    return json.dumps(TE.cash_flow_analysis(depreciable_capital_usd, revenue_usd_y, operating_cost_usd_y, plant_life_years,
                                            income_tax_rate, discount_rate, depreciation, construction_schedule,
                                            working_capital_usd, land_usd, startup_years, startup_revenue_fraction,
                                            startup_operating_cost_fraction), default=float)


@beta_tool
def flowsheet_cost(flowsheet_id: str, prices: dict[str, float], hours_per_year: float, interest: float, years: int,
                   exchanger_areas_m2: dict[str, float] | None = None, vessel_sizes: dict[str, object] | None = None,
                   options: dict[str, object] | None = None, lang: str = "lang_1948_fluid", country: str | None = None,
                   year: int = 2023) -> str:
    """Total annual cost of a solved flowsheet: Σ purchase (Seider via IDAES) × Lang × location, annualized by
    CRF(interest, years), + electricity/heating/cooling at the given prices (electricity_usd_kWh, heating_usd_GJ,
    cooling_usd_GJ). Exchangers need exchanger_areas_m2; vessels need vessel_sizes; columns take vessel_sizes {unit:
    {diameter_m, tray_spacing_m, extra_height_m}} (shell + stages−2 trays) or {diameter_m, packed_height_m,
    packing_price_usd_per_m3} (shell + packing), and their condenser/reboiler duties count as cooling/heating."""
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
def eo_column_optimize(components: list[str], feed_flow_mol_s: float, feed_T_K: float, feed_P_Pa: float,
                       feed_mole_fractions: dict, n_trays: int, feed_tray: int, pressure_Pa: float, specs: list[dict],
                       reflux_ratio_min: float = 0.1, reflux_ratio_max: float = 20.0, boilup_ratio_min: float = 0.1,
                       boilup_ratio_max: float = 20.0, activity_model: str = "ideal",
                       binary_parameters: list | dict | None = None, excess_enthalpy: bool | None = None) -> str:
    """Equation-oriented (IDAES + ipopt) minimum reboiler duty of a tray column for a mixture of 2+ components
    (English chemicals names), meeting purity/recovery specs: specs = [{"stream": "distillate"|"bottoms", "component",
    "kind": "mole_fraction"|"recovery", "min"}]. Property package generated from chemicals/thermo and checked vs DIPPR.
    activity_model="ideal": Raoult, non-ideal mixtures (UNIFAC azeotrope or |ln γi/γj| > 0.10) refused.
    activity_model="NRTL": NRTL liquid + ideal gas; binary_parameters = the user's pairs [{"components": [i, j],
    "b_ij_K", "b_ji_K" (or dg_*_J_mol / dg_*_cal_mol), optional "a_ij", "a_ji", "alpha", "source"}] and/or the whole
    result of vle_binary_regression; missing pairs come from thermo's ChemSep NRTL bank or are refused (never
    invented). Specs across an azeotrope predicted with those parameters are refused. excess_enthalpy (NRTL's h^E in the
    liquid enthalpy): None (default) = conditional, ON only if every pair comes from a vle_binary_regression fit with
    h^E data that passed its verdict, otherwise OFF with a warning; True/False = the user's explicit choice.
    properties.excess_enthalpy has the decision, where h^E comes from (with citations) and its share of Q_reb. A user
    pair may carry "hE_source" (citation; a label only, it does not turn the term on). Returns optimal reflux/boil-up,
    duties, products, multipliers (MW per unit of spec), parameter source per pair, UNIFAC cross-check, warnings and
    evidence (neighbours, multipliers vs finite differences, NRTL vs thermo)."""
    return json.dumps(TO.eo_column_optimize(components, feed_flow_mol_s, feed_T_K, feed_P_Pa, feed_mole_fractions, n_trays,
                                            feed_tray, pressure_Pa, specs, reflux_ratio_min, reflux_ratio_max,
                                            boilup_ratio_min, boilup_ratio_max, activity_model, binary_parameters,
                                            excess_enthalpy),
                      default=float)


@beta_tool
def optimize_design_multi(flowsheet_id: str, variables: list[dict[str, object]], objective: dict[str, object],
                          constraints: list[dict[str, object]] | None = None, maximize: bool = False, grid_points: int = 5,
                          max_evaluations: int = 150, tolerance: float = 1e-4) -> str:
    """Optimize 1-4 interacting design variables together: coarse grid + Nelder-Mead in normalized coordinates, penalties
    for constraint violations; variables [{kind, name, param, lower, upper, log_scale, integer}] ("integer": true for
    counts such as column stages: each value is enumerated with its own continuous sub-optimization and the result
    carries integer_curve), objective/constraints as in optimize_design. Returns best values, objective, active
    constraints, variables at a bound, path."""
    return json.dumps(TO.optimize_design_multi(flowsheet_id, variables, objective, constraints, maximize, grid_points,
                                               max_evaluations, tolerance), default=float)


@beta_tool
def optimize_design_surrogate(flowsheet_id: str, variables: list[dict[str, object]], objective: dict[str, object],
                              constraints: list[dict[str, object]] | None = None, maximize: bool = False,
                              n_samples: int | None = None, n_holdout: int | None = None, tolerance_pct: float = 0.5,
                              objective_tolerance: float | None = None, seed: int = 0) -> str:
    """Surrogate optimisation of a built DWSIM flowsheet (OMLT + ipopt): samples 1-6 continuous variables
    [{kind, name, param, lower, upper, log_scale}] over the user's bounds (Latin hypercube + corners, n_samples default
    20 + 15·n, independent held-out set), trains one small tanh network per output, embeds them with OMLT and solves
    with ipopt, then re-runs DWSIM at the optimum. objective/constraints as in optimize_design; a constraint may carry
    "tolerance" (absolute accuracy wanted). Returns surrogate_optimum, dwsim_at_optimum, holdout_error,
    error_at_optimum, best_dwsim_sample, evidence and status ("verified" only if DWSIM confirms within tolerance)."""
    return json.dumps(TO.optimize_design_surrogate(flowsheet_id, variables, objective, constraints, maximize, n_samples,
                                                   n_holdout, tolerance_pct, objective_tolerance, seed), default=float)


@beta_tool
def optimize_design_bayesian(flowsheet_id: str, variables: list[dict[str, object]], objective: dict[str, object],
                             constraints: list[dict[str, object]] | None = None, maximize: bool = False,
                             max_evaluations: int | None = None, n_initial: int | None = None, tolerance_pct: float = 0.5,
                             objective_tolerance: float | None = None, stall_iterations: int | None = None, seed: int = 0,
                             restarts: int = 0) -> str:
    """Bayesian optimisation of a built DWSIM flowsheet (Gaussian process + constrained expected improvement) for
    EXPENSIVE or failure-prone solves with a budget of tens of evaluations: 1-10 variables
    [{kind, name, param, lower, upper, log_scale, integer}] (integers native), objective/constraints as in
    optimize_design. Failed solves are infeasible points, never a crash. Stops on max_evaluations (default 20 + 10·n per
    run), expected improvement below 0.1 × tolerance for 5 iterations, or no improvement in stall_iterations.
    restarts = independent extra runs to check for a local optimum (restart_agreement). Returns best re-evaluated in
    DWSIM, gp_at_optimum, gp_diagnostics, history, runs, n_evaluations, stopped_by, evidence, status, suitability. For 1 variable use optimize_design; for 2-4 on a fast flowsheet optimize_design_multi; for an
    algebraic model optimize_design_surrogate."""
    return json.dumps(TO.optimize_design_bayesian(flowsheet_id, variables, objective, constraints, maximize, max_evaluations,
                                                  n_initial, tolerance_pct, objective_tolerance, stall_iterations, seed,
                                                  restarts),
                      default=float)


@beta_tool
def suggest_experiments(variables: list[dict[str, object]], experiments: list[dict[str, object]], objective: dict[str, object],
                        batch_size: int, constraints: list[dict[str, object]] | None = None, seed: int = 0) -> str:
    """Next experiments to run from the USER's own lab/plant data (Bayesian experiment design, no simulator).
    variables [{name, lower, upper, log_scale, integer}]; experiments = past runs, one dict per run with every variable
    and the measured outputs (a run that failed: "failed": true); objective {"name", "maximize"}; constraints
    [{name, max|min}] on other measured outputs ("log": true in objective models log y for responses spanning decades);
    batch_size = how many runs fit in the next round. Returns proposals
    (PROPOSALS, each with predicted mean, uncertainty, 95 % interval, probability of meeting the limits),
    current_best (best MEASURED run), model (noise, leave-one-out fit) and warnings; with fewer than 2·(n + 1) valid
    runs, mode "initial_design" (space-filling plan, no model). One objective, no categorical variables. Never
    invents measurements."""
    return json.dumps(TO.suggest_experiments(variables, experiments, objective, batch_size, constraints, seed), default=float)


@beta_tool
def fopdt_identify(time_s: list[float], output: list[float], input_change: float, step_time_s: float,
                   baseline_output: float | None = None, final_output: float | None = None) -> str:
    """Identify FOPDT (K, τ, θ) and SOPDT models from an open-loop step test by least squares: R², RMSE, residual
    autocorrelation, 95 % CI, plus Smith and Sundaresan-Krishnaswamy two-point estimates as a cross-check, and
    warnings (no steady state, overshoot, inverse response). time_s, output, input_change (Δu) and step_time_s are
    the user's data. Next: pid_tuning with the fitted K, τ, θ."""
    return json.dumps(TCO.fopdt_identify(time_s, output, input_change, step_time_s, baseline_output, final_output),
                      default=float)


@beta_tool
def pid_tuning(gain: float | None = None, time_constant: float | None = None, dead_time: float | None = None,
               second_time_constant: float | None = None, integrating: bool = False,
               process_model: dict[str, object] | None = None, controller_type: str = "PI",
               rules: list[str] | None = None, tau_c: float | None = None, evaluate_closed_loop: bool = True,
               setpoint_step: float = 1.0, load_step: float = 1.0, derivative_filter_alpha: float = 0.01,
               sampling_period: float | None = None) -> str:
    """P/PI/PID settings from a FOPDT/SOPDT (gain, time_constant, dead_time, second_time_constant; integrating=true
    with gain = slope) or a full process_model {gain, time_constants, lead_time_constants, dead_time, integrating}
    reduced by Skogestad's half rule. Rules: SIMC (recommended, Skogestad 2003), IMC (Seborg/Rivera), ZN ultimate
    cycle, ZN reaction curve, Cohen_Coon, Tyreus_Luyben; each with source, validity range and closed-loop Ms, GM,
    PM, setpoint/load IAE, overshoot and settling time. tau_c default θ (in assumptions). summary = comparison table;
    status 'verified' = exact margins match python-control and the simulation's integrated error its closed form."""
    return json.dumps(TCO.pid_tuning(gain, time_constant, dead_time, second_time_constant, integrating, process_model,
                                     controller_type, rules, tau_c, evaluate_closed_loop, setpoint_step, load_step,
                                     derivative_filter_alpha, sampling_period), default=float)


@beta_tool
def pid_closed_loop(process_model: dict[str, object], controller: dict[str, object], setpoint_step: float = 1.0,
                    load_step: float = 1.0, derivative_filter_alpha: float = 0.01) -> str:
    """Closed-loop performance of a given controller {Kc, tau_I, tau_D, form series|ideal} (or {Kc: 0, KI}) on a
    given process_model {gain, time_constants, lead_time_constants, dead_time, integrating}, exact dead time:
    stability, GM, PM, Ms, Mt, setpoint and load-step metrics (overshoot, settling, IAE, peak, input TV),
    trajectories and checks against python-control."""
    return json.dumps(TCO.pid_closed_loop(process_model, controller, setpoint_step, load_step, derivative_filter_alpha),
                      default=float)


@beta_tool
def close_flowsheet(flowsheet_id: str) -> str:
    """Discard a flowsheet when you no longer need it (e.g. between search iterations)."""
    return json.dumps(T.close_flowsheet(flowsheet_id))


AGENT_TOOLS = [list_property_packages, new_flowsheet, add_material_stream, add_energy_stream, define_conversion_reaction,
               define_equilibrium_reaction, define_kinetic_reaction, add_unit, connect, dof_check, solve, read_stream, read_unit,
               set_unit_param, set_stream_spec, sweep, stream_transport_properties, pipe_hydraulics, size_pipe,
               heat_exchanger_area, tube_film_coefficient, tubular_reactor_field, nonisothermal_tubular_reactor, vessel_residence_time_cfd, stirred_tank_cfd, adiabatic_flame, kinetic_reactor_network, relief_valve_sizing,
               control_valve_sizing, fittings_resistance, compressible_gas_line, two_phase_pressure_drop, shell_side_kern,
               shell_side_bell_delaware, nucleate_boiling, film_condensation, pinch_analysis, pinch_from_flowsheet, equipment_cost, costing_options, cost_unit, flowsheet_cost, bare_module_factors, capital_investment, cash_flow_analysis, optimize_design, optimize_design_multi, eo_column_optimize, optimize_design_surrogate, optimize_design_bayesian, suggest_experiments, absorption_stages,
               packed_column_height, diffusivity_estimate, packing_catalog,
               packed_column_hydraulics, fopdt_identify, pid_tuning, pid_closed_loop, hazop_preliminary, export_flowsheet,
               hazop_from_dexpi, pid_from_hazop, lopa_from_hazop, release_source_term, gas_dispersion,
               fire_radiation, explosion_overpressure, pipe_network, vle_binary_regression, close_flowsheet]

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
