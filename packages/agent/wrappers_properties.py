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
    viscosity, thermal conductivity and surface tension (P_Pa only sets the phase and the gas density). compound = name
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


AGENT_WRAPPERS = [compound_properties, independent_flash, compare_flash]
