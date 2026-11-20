"""Agent-facing wrappers of the `props` block (pure-compound properties and independent flash). Registered by the
integrator; same order and names as `packages.dwsim_client.tools_properties.PROPERTIES_TOOLS`."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_properties as TPR


@beta_tool
def compound_properties(compound: str, T_K: float | None = None, P_Pa: float | None = None,
                        alternatives: bool = False) -> str:
    """Pure-compound lookup without a flowsheet (chemicals/thermo databases): CAS, formula, MW, Tc, Pc, Vc, omega, Tb, Tm,
    triple point, ΔfH°, ΔfG°, S°, heat of fusion and, if T_K is given, Psat, ΔHvap, liquid and ideal-gas Cp, density,
    viscosity, thermal conductivity and surface tension (transport from CoolProp's reference correlation where it matches NIST, else thermo
    with its measured band; with P_Pa a dense gas also gets mu_gas_at_P_Pa_s and k_gas_at_P_W_mK). compound = name
    (English or common Spanish), CAS, formula or SMILES. Every value carries method (data source or estimator), basis
    ('data', 'compilation', 'derived', 'estimate'), the method's T range, in_range/extrapolated and a flag when it is
    NOT measured data: never quote a flagged value as experimental. evidence.verified = the library's data for the
    compound are mutually consistent (Psat(Tb), Psat(Tc), omega definition, Clapeyron). alternatives=True lists the
    other methods valid at T."""
    return json.dumps(TPR.compound_properties(compound, T_K, P_Pa, alternatives), default=float, ensure_ascii=False)


@beta_tool
def independent_flash(components: list[str], mole_fractions: list[float], model: str, T_K: float | None = None,
                      P_Pa: float | None = None, vapor_fraction: float | None = None, H_J_mol: float | None = None,
                      S_J_molK: float | None = None, kij: list[dict[str, object]] | None = None,
                      activity_parameters: list[dict[str, object]] | None = None, use_library_kij: bool = True) -> str:
    """Flash with thermo, independent of DWSIM (second opinion), with Michelsen stability analysis: phases (V, L, L1/L2),
    fractions, compositions, K-values, H, S. model is mandatory: 'PR', 'SRK', 'NRTL'/'UNIQUAC'/'WILSON' (user parameters
    in activity_parameters [{i, j, b_ij_K, b_ji_K, a_ij, a_ji, alpha}], tau_ij = a_ij + b_ij/T), 'UNIFAC',
    'UNIFAC_DORTMUND', 'UNIFAC_LLE' or 'IDEAL'. Give exactly two of T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK (T-P,
    P-H, P-S, P-VF, T-VF). kij = [{i, j, kij}] for PR/SRK; binary_parameters reports each one's source (user, library
    ChemSep PR, zero_by_absence with a warning). evidence: material balance, fugacity equality, Rachford-Rice and
    stability test; verified certifies the numerics for the chosen model, not the model (cubic EOS with kij = 0 is poor
    for water, CO2 and polar mixtures)."""
    return json.dumps(TPR.independent_flash(components, mole_fractions, model, T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK,
                                            kij, activity_parameters, use_library_kij), default=float, ensure_ascii=False)


@beta_tool
def compare_flash(reference: dict[str, object], components: list[str], mole_fractions: list[float], model: str,
                  T_K: float | None = None, P_Pa: float | None = None, vapor_fraction: float | None = None,
                  H_J_mol: float | None = None, S_J_molK: float | None = None, kij: list[dict[str, object]] | None = None,
                  activity_parameters: list[dict[str, object]] | None = None, use_library_kij: bool = True,
                  phase_fraction_tolerance: float = 0.01, mole_fraction_tolerance: float = 0.01,
                  temperature_tolerance_K: float = 0.5, pressure_relative_tolerance: float = 0.01) -> str:
    """Cross-check another engine's flash (e.g. a DWSIM stream) against independent_flash at the same specification.
    reference = {engine, phases: [{type: 'vapor'|'liquid', fraction (molar, of the feed), mole_fractions (list in the
    order of components, or {name: x})}], optional T_K, P_Pa}. Use the same model and binary parameters as the other
    engine. Returns verdict ('agree', 'disagree', 'phase_set_differs', 'independent_not_verified',
    'independent_not_converged'), per-phase differences (independent − reference) and the full independent flash.
    Enthalpies are not compared (different reference states). Agreement certifies the calculation, not the model."""
    return json.dumps(TPR.compare_flash(reference, components, mole_fractions, model, T_K, P_Pa, vapor_fraction, H_J_mol,
                                        S_J_molK, kij, activity_parameters, use_library_kij, phase_fraction_tolerance,
                                        mole_fraction_tolerance, temperature_tolerance_K, pressure_relative_tolerance),
                      default=float, ensure_ascii=False)


@beta_tool
def cubic_eos_volume(components: list[str], model: str, T_K: float, P_Pa: float,
                     mole_fractions: list[float] | None = None, critical_constants: list[dict] | None = None,
                     kij: list[dict] | None = None, root: str = "stable") -> str:
    """Single-phase molar volume and Z from a cubic equation of state at T and P, pure compound or mixture, no flowsheet:
    model = 'RK' (plain Redlich-Kwong 1949), 'SRK' (= 'RKS'), 'PR' or 'VDW' (mandatory, never default it). Use it for
    equation-of-state textbook problems: critical_constants = [{'compound', 'Tc_K', 'Pc_Pa', 'omega'}] imposes the
    statement's constants (anything missing comes from the chemicals database, marked source 'library'). P_Pa is absolute
    (1 atm = 101325 Pa). mole_fractions only for mixtures (van der Waals one-fluid mixing; kij = [{'i','j','kij'}], pairs
    without data stay at 0 and are warned). root = 'stable' (lowest Gibbs, default), 'gas' (largest) or 'liquid'
    (smallest). Returns Z, V_m3_mol, V_L_mol, V_m3_kmol, the ideal-gas volume and the deviation from it, every real
    root, the parameters used and evidence (each root substituted back must reproduce P). It is NOT a flash: for a
    mixture that may condense, also run independent_flash and report both."""
    return json.dumps(TPR.cubic_eos_volume(components, model, T_K, P_Pa, mole_fractions, critical_constants, kij, root),
                      default=float, ensure_ascii=False)


@beta_tool
def reaction_equilibrium(reactions: list[dict[str, object]], T_K: float | list[float], P_Pa: float | None = None,
                         feed_mol: dict[str, float] | None = None, data_overrides: dict[str, dict[str, object]] | None = None,
                         fugacity_coefficients: dict[str, float] | None = None,
                         cantera_mechanism: str | None = "gri30.yaml") -> str:
    """Chemical-reaction equilibrium of an IDEAL-GAS mixture from formation data (no flowsheet, no mechanism): K(T) of
    each reaction and, with a feed, extents, composition and conversions for one or several simultaneous reactions
    restricted to the chosen species/reactions. Standard state: pure ideal gas at T and 1 bar for ALL species.
    reactions = [{'name', 'stoichiometry': {compound: coeff}}] or [{compound: coeff}], negative = reactant; must be
    balanced and independent. T_K number or list. P_Pa absolute, required with feed_mol = {compound: mol}; feed compounds
    absent from every reaction are inerts. Without feed_mol only K(T) is returned. data_overrides = {compound:
    {'Hf_J_mol', 'Gf_J_mol' (ideal gas, 298.15 K, 1 bar), 'Cp_J_molK': number | [c0, c1, ...] J/(mol K) polynomial in T |
    {'A','B','C','D'} of Cp/R = A + B·T + C·T² + D/T²}} for the USER's/statement's data (labelled 'dato del usuario');
    if the databank lacks formation data the tool stops and asks: never invent them. fugacity_coefficients = {compound:
    φ} user constants (default φ = 1). Returns species (ΔfH°, ΔfG°, Cp with their source/method), reactions
    (dH_298_J_mol, dG_298_J_mol, K_298, K_of_T with the rigorous van't Hoff K and a LABELLED constant-ΔH approximation:
    quote K, not the approximation), equilibrium per T (extents_mol, moles_out, mole_fractions, conversion,
    K_consistency, element_balance, gibbs_minimization_check), cantera_comparison (GRI-Mech 3.0 NASA data when all
    species exist there: a data-source comparison), warnings and evidence.verified (analytic vs numerical van't Hoff,
    Π a^ν = K, element balance, independent Lagrange Gibbs minimisation). Not included: EOS fugacity coefficients,
    adiabatic mode, condensed phases (solid carbon, liquid water). Report the data sources with the result."""
    return json.dumps(TPR.reaction_equilibrium(reactions, T_K, P_Pa, feed_mol, data_overrides, fugacity_coefficients,
                                               cantera_mechanism), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [compound_properties, independent_flash, compare_flash, cubic_eos_volume, reaction_equilibrium]
