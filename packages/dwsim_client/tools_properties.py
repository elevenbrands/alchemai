"""Agent tools for pure-compound properties and an engine-independent flash (bloque `props`). No DWSIM flowsheet needed.
Engines: `packages.properties.pure` (lookup with provenance on chemicals/thermo) and `packages.properties.flash` (flash on
thermo with an own Michelsen stability test and a fallback multiphase solver)."""
from __future__ import annotations

from packages.properties import flash as FL
from packages.properties import pure as PU


def compound_properties(compound: str, T_K: float | None = None, P_Pa: float | None = None,
                        alternatives: bool = False) -> dict:
    """Propiedades de un compuesto puro con su procedencia, sin construir un flowsheet: identidad (CAS, fórmula, masa
    molar), constantes críticas, factor acéntrico, puntos de ebullición y fusión normales, punto triple, entalpía y energía
    de Gibbs de formación, entropía estándar, calor de fusión y, si se da T_K, presión de vapor, calor de vaporización, Cp
    de líquido y de gas ideal, densidad, viscosidad, conductividad térmica y tensión superficial.

    compound: nombre (inglés o español común), número CAS, fórmula o SMILES. Si no se reconoce, error con nombres parecidos.
    T_K: temperatura para las propiedades dependientes de T (opcional; sin ella solo salen identidad y constantes).
    P_Pa: presión absoluta (opcional, exige T_K): decide la fase estable frente a la presión de vapor y da la densidad del
      gas a T y P y, si el compuesto tiene correlación de referencia de transporte, mu_gas_at_P_Pa_s y k_gas_at_P_W_mK (gas
      denso); las propiedades del líquido son las de saturación o baja presión, y mu_gas_Pa_s / k_gas_W_mK las del gas
      diluido.
    alternatives: True para listar además el valor de cada otro método de la librería válido a esa T.

    Devuelve identity, constants {Tc_K, Pc_Pa, Vc_m3_mol, Zc, acentric_factor, Tb_K, Tm_K, Tt_K, Pt_Pa, Hfus_J_mol,
    Hvap_Tb_J_mol, Hf_gas_J_mol, Hf_liquid_J_mol, Hf_solid_J_mol, S0_gas_J_molK, Gf_gas_J_mol, dipole_moment_debye}, state
    {phase} y properties_at_T {Psat_Pa, Hvap_J_mol, Cp_liquid_J_molK, Cp_ideal_gas_J_molK, V_liquid_m3_mol,
    rho_liquid_kg_m3, rho_gas_kg_m3, mu_liquid_Pa_s, mu_gas_Pa_s, k_liquid_W_mK, k_gas_W_mK, sigma_N_m y, con P_Pa en gas,
    mu_gas_at_P_Pa_s, k_gas_at_P_W_mK}. CADA valor es un
    diccionario con value, method (nombre del método o fuente en chemicals/thermo: HEOS, CRC, DIPPR_PERRY_8E, JOBACK...),
    basis ('data' = medido, compilación evaluada o correlación ajustada a datos; 'compilation' = compilación que no
    distingue medido de estimado; 'derived'; 'estimate' = contribución de grupos o estados correspondientes), estimated,
    T_range_K e in_range (¿T dentro del intervalo del método?), extrapolated y flag cuando el valor NO es un dato medido:
    un valor con flag nunca debe citarse como experimental. data_methods_spread_rel indica cuánto difieren entre sí las
    fuentes basadas en datos. evidence.verified certifica que los datos de la librería para el compuesto son coherentes
    entre sí (Psat en Tb = 1 atm, Psat en Tc = Pc, ω frente a su definición, ΔHvap frente a Clapeyron), no cada valor
    contra una medición. Viscosidad, conductividad y tensión superficial: method COOLPROP_REFERENCE = correlación de
    referencia propia del compuesto (reference_correlation, p. ej. Avgeri-JPCRD-2015-Toluene) que coincidió con NIST
    WebBook dentro del 1 % (N2, O2, Ar, H2, CO2, agua, metanol, benceno, tolueno, n-hexano, R134a; conductividad también
    metano, propano, n-butano, n-heptano; tensión superficial de Mulero en todos salvo el agua, que usa IAPWS); en los
    demás se da el valor de thermo con reference_check (la correlación de CoolProp y su desviación) y measured_band_rel
    (banda medida del método); flag si discrepan más de un 2 %, cerca del punto crítico (T/Tc > 0.95) o si el método tiene
    banda ancha. Validación: NIST Chemistry WebBook e IAPWS-95 (tests/test_properties.py) y NIST WebBook en 19 fluidos
    para transporte (tests/test_transport_properties.py: ≤ 0.6 % con correlación de referencia, ≤ 2.3 % con thermo
    fuera de la zona crítica)."""
    return PU.compound_properties(compound, T_K, P_Pa, alternatives)


def independent_flash(components: list[str], mole_fractions: list[float], model: str | None = None,
                      T_K: float | None = None, P_Pa: float | None = None, vapor_fraction: float | None = None,
                      H_J_mol: float | None = None, S_J_molK: float | None = None, kij: list[dict] | None = None,
                      activity_parameters: list[dict] | None = None, use_library_kij: bool = True) -> dict:
    """Flash de una mezcla con la librería thermo, independiente del motor DWSIM (segunda opinión): fases presentes (V, L,
    o L1 y L2 si hay dos líquidos), fracción de cada fase, composiciones, valores K, entalpía y entropía, con análisis de
    estabilidad de Michelsen que detecta un segundo líquido (ELL y ELLV).

    components, mole_fractions: compuestos (nombre, CAS o fórmula) y composición molar global (debe sumar 1).
    model (OBLIGATORIO, sin valor por defecto): 'PR' o 'SRK' (ecuación cúbica para todas las fases); 'NRTL', 'UNIQUAC' o
      'WILSON' (coeficientes de actividad con TUS parámetros binarios, vapor ideal); 'UNIFAC', 'UNIFAC_DORTMUND' o
      'UNIFAC_LLE' (predictivos por grupos); 'IDEAL' (ley de Raoult).
    Especificación: exactamente dos de T_K, P_Pa, vapor_fraction (0 = burbuja, 1 = rocío), H_J_mol, S_J_molK en las
      combinaciones T-P, P-H, P-S, P-vapor_fraction o T-vapor_fraction. H y S usan la referencia indicada en
      enthalpy_reference (gas ideal a 298.15 K y 101325 Pa): toma H de un flash previo de esta misma herramienta, p. ej.
      para una válvula (mismo H a menor P).
    kij (solo PR/SRK): [{'i': comp, 'j': comp, 'kij': valor}] del usuario. Para los pares sin dato, PR usa la tabla ChemSep
      PR de thermo si use_library_kij es True; lo que falte queda en 0 y se avisa. SRK no tiene tabla: todo par sin kij
      del usuario queda en 0.
    activity_parameters (NRTL/UNIQUAC/WILSON): [{'i', 'j', 'b_ij_K', 'b_ji_K', opcionales 'a_ij', 'a_ji', 'alpha' (NRTL,
      obligatorio)}] con tau_ij = a_ij + b_ij/T (misma convención que la regresión de ELV de este proyecto).

    Devuelve T_K, P_Pa, vapor_fraction, phase_labels, liquid_liquid_split, phases [{label, type, mole_fraction_of_feed,
    mole_fractions, rho_kg_m3, Z, H_J_mol, S_J_molK, ln_fugacity_coefficients}], K_values, H_J_mol, S_J_molK,
    pure_component_data (Tc, Pc, ω y su método), binary_parameters (cada kij o parámetro con source: 'user', 'library' o
    'zero_by_absence'), solver y evidence: cierre del balance de materia, igualdad de fugacidades entre fases, residuo de
    Rachford-Rice, residuo de la especificación y stability {final_state_stable, single_phase_feed, second_liquid_found}
    de una prueba de plano tangente propia. Si thermo devuelve un estado al que le falta una fase, se resuelve de nuevo con
    un solver multifásico propio (se indica en solver); en una mezcla binaria con heteroazeótropo, un flash P-H, P-S o de
    fracción de vapor que cae en la línea trifásica devuelve las tres fases (V, L1, L2) con sus fracciones
    (solver 'own_three_phase_jump_resolution'). verified certifica el cálculo para el modelo elegido, NO que el
    modelo represente la mezcla: con kij = 0 una cúbica falla mucho en mezclas con agua, CO2 o polares (ver warnings).
    Validación: datos publicados de ELV y ELL, CoolProp y soluciones analíticas (tests/test_properties.py)."""
    return FL.independent_flash(components, mole_fractions, model, T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK, kij,
                                activity_parameters, use_library_kij)


def compare_flash(reference: dict, components: list[str], mole_fractions: list[float], model: str | None = None,
                  T_K: float | None = None, P_Pa: float | None = None, vapor_fraction: float | None = None,
                  H_J_mol: float | None = None, S_J_molK: float | None = None, kij: list[dict] | None = None,
                  activity_parameters: list[dict] | None = None, use_library_kij: bool = True,
                  phase_fraction_tolerance: float = 0.01, mole_fraction_tolerance: float = 0.01,
                  temperature_tolerance_K: float = 0.5, pressure_relative_tolerance: float = 0.01) -> dict:
    """Contrasta el resultado de flash de OTRO motor (DWSIM, Aspen, un cálculo propio) con el flash independiente de thermo
    en la misma especificación y da un veredicto.

    reference: {'engine': nombre, 'phases': [{'type': 'vapor'|'liquid', 'fraction': fracción molar de la alimentación,
      'mole_fractions': lista en el orden de components o {nombre: x}}], opcionales 'T_K' y 'P_Pa' del resultado (para
      flashes P-H, P-S o con fracción vaporizada)}.
    components, mole_fractions, model, especificación, kij, activity_parameters: como en independent_flash; usa el MISMO
      modelo y los mismos parámetros binarios que el otro motor o la comparación mide la diferencia de modelos.
    Tolerancias (criterios numéricos de la comparación, modificables): phase_fraction_tolerance y mole_fraction_tolerance
      absolutas, temperature_tolerance_K, pressure_relative_tolerance.

    Devuelve verdict: 'agree', 'disagree' (mismas fases, diferencias mayores que la tolerancia), 'phase_set_differs' (un
    motor ve una fase que el otro no, típico de un segundo líquido sin análisis de estabilidad),
    'independent_not_verified' o 'independent_not_converged'; differences por fase (independiente − referencia),
    state_differences, y el flash independiente completo en independent. No compara entalpías (cada motor tiene su
    referencia). Coincidir certifica el cálculo, no el modelo."""
    return FL.compare_flash(reference, components, mole_fractions, model, T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK, kij,
                            activity_parameters, use_library_kij, phase_fraction_tolerance, mole_fraction_tolerance,
                            temperature_tolerance_K, pressure_relative_tolerance)


PROPERTIES_TOOLS = [compound_properties, independent_flash, compare_flash]
