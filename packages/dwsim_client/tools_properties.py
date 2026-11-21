"""Agent tools for pure-compound properties and an engine-independent flash (bloque `props`). No DWSIM flowsheet needed.
Engines: `packages.properties.pure` (lookup with provenance on chemicals/thermo) and `packages.properties.flash` (flash on
thermo with an own Michelsen stability test and a fallback multiphase solver)."""
from __future__ import annotations

from packages.properties import cubic_eos as CE
from packages.properties import flash as FL
from packages.properties import pure as PU
from packages.properties import reaction_equilibrium as RE


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


def cubic_eos_volume(components: list[str], model: str, T_K: float, P_Pa: float,
                     mole_fractions: list[float] | None = None, critical_constants: list[dict] | None = None,
                     kij: list[dict] | None = None, root: str = "stable") -> dict:
    """Volumen molar y factor de compresibilidad Z de UNA fase con una ecuación de estado cúbica, para un compuesto puro o
    una mezcla a T y P dadas, sin flowsheet: van der Waals, Redlich-Kwong (la original de 1949), Soave-Redlich-Kwong o
    Peng-Robinson. Es la herramienta para los problemas de libro de ecuaciones de estado, porque acepta las constantes
    críticas del enunciado.

    components: compuestos (nombre, CAS o fórmula). mole_fractions: composición molar (suma 1); se omite si es uno solo.
    model (OBLIGATORIO, sin valor por defecto): 'RK', 'SRK' (también 'RKS'), 'PR' o 'VDW'.
    T_K, P_Pa: temperatura y presión ABSOLUTA en Pa (1 atm = 101325 Pa).
    critical_constants: [{'compound', 'Tc_K', 'Pc_Pa', 'omega'}] con las constantes del USUARIO o del enunciado; lo que no
      se dé se toma de la base de datos de chemicals y sale marcado source 'library' con su método. omega solo lo usan
      SRK y PR.
    kij: [{'i', 'j', 'kij'}] del usuario para la regla de mezclado de van der Waals; los pares sin dato quedan en 0 y se
      avisa (es lo que supone un problema de libro que no da kij).
    root: 'stable' (por defecto: la raíz de menor energía de Gibbs a esa composición), 'gas' (la mayor) o 'liquid' (la
      menor). Con una sola raíz real las tres opciones dan lo mismo.

    Devuelve Z, V_m3_mol, V_L_mol, V_m3_kmol de la raíz elegida, root_type, root_is_lowest_gibbs, ideal_gas (volumen de gas
    ideal y desviación relativa de la ecuación frente a él), roots (TODAS las raíces reales con Z, volumen, ln del
    coeficiente de fugacidad y cierre de presión), mixture_parameters (a·alpha, b, A, B), components (Tc, Pc, omega con
    source 'user' o 'library', Tr, Pr, a, alpha, b), binary_parameters, warnings y evidence: cada raíz se sustituye en la
    forma explícita en presión y debe reproducir P (max_pressure_closure_rel). verified certifica el cálculo con la
    ecuación elegida, NO que esa ecuación represente al fluido. NO es un flash: no detecta si la mezcla se separa en dos
    fases; para eso está independent_flash. Validación: clases RK, SRK, PR y VDW de la librería thermo e identidades del
    punto crítico (tests/test_cubic_eos.py)."""
    return CE.cubic_eos_volume(components, model, T_K, P_Pa, mole_fractions, critical_constants, kij, root)


def reaction_equilibrium(reactions: list[dict], T_K: float | list[float], P_Pa: float | None = None,
                         feed_mol: dict | None = None, data_overrides: dict | None = None,
                         fugacity_coefficients: dict | None = None, cantera_mechanism: str | None = "gri30.yaml") -> dict:
    """Equilibrio químico en fase GAS IDEAL a partir de datos termoquímicos de formación, sin flowsheet ni mecanismo:
    constante de equilibrio K(T) de cada reacción y, si se da la alimentación, avances, composición y conversiones de una
    o varias reacciones simultáneas restringidas a las especies y reacciones elegidas. Estado estándar: gas ideal puro a
    T y P° = 1 bar para TODAS las especies.

    reactions: [{'name': opcional, 'stoichiometry': {compuesto: coeficiente}}] o directamente [{compuesto: coeficiente}];
      coeficiente negativo = reactivo, positivo = producto. Compuestos por nombre, CAS o fórmula. Deben estar balanceadas
      y ser linealmente independientes (si no, error que dice qué átomos sobran).
    T_K: temperatura o lista de temperaturas (K). P_Pa: presión ABSOLUTA (Pa); obligatoria con feed_mol.
    feed_mol: {compuesto: moles} de la alimentación; los compuestos que no aparecen en ninguna reacción son INERTES (solo
      diluyen). Sin feed_mol solo se devuelve K(T).
    data_overrides: {compuesto: {'Hf_J_mol', 'Gf_J_mol' (gas ideal, 298.15 K, 1 bar), 'Cp_J_molK'}} con datos del
      USUARIO o del enunciado; Cp_J_molK = número (constante), lista [c0, c1, ...] de Cp = Σc_k·T^k en J/(mol·K), o
      {'A','B','C','D'} de Cp/R = A + B·T + C·T² + D/T² (forma de Smith-Van Ness). Salen marcados 'dato del usuario'.
      Si a un compuesto le faltan datos de formación en la base de datos, la herramienta se detiene y los pide: no inventa.
    fugacity_coefficients: {compuesto: φ} del usuario, constantes (a_i = y_i·φ_i·P/P°); sin ellos φ = 1.
    cantera_mechanism: mecanismo de Cantera para la comparación de datos ('gri30.yaml'; None la omite).

    Devuelve species (CAS, ΔfH° y ΔfG° con su fuente: chemicals.reaction.Hfg método ATCT_G/CRC/..., ΔfG° derivado de S°,
    Cp de thermo.HeatCapacityGas con método e intervalo), reactions [{equation, delta_nu, dH_298_J_mol, dG_298_J_mol,
    dS_298_J_molK, K_298, K_of_T: [{T_K, K, lnK, log10K, dG_T_J_mol, dH_T_J_mol, constant_dH_approximation}]}] donde K es
    la integración rigurosa de van't Hoff con los Cp y constant_dH_approximation es la aproximación ΔH° constante
    (etiquetada, no usarla como resultado), equilibrium por temperatura [{extents_mol, moles_out, mole_fractions,
    partial_pressures_Pa, conversion de cada reactivo alimentado, K_consistency, element_balance,
    gibbs_minimization_check}], cantera_comparison (ln K y composición con los polinomios NASA de GRI-Mech 3.0 cuando
    todas las especies están ahí: compara FUENTES de datos), warnings, assumptions, not_included y evidence.
    evidence.verified exige: ln K(T) analítico = cuadratura numérica de van't Hoff, Π a_i^ν = K en la solución, balance
    de elementos, y la misma composición por minimización directa de Gibbs con multiplicadores de Lagrange; certifica el
    cálculo con esos datos, no que la mezcla sea gas ideal ni que el equilibrio se alcance. NO incluye: φ calculados con
    ecuación de estado, modo adiabático, fases condensadas (carbono sólido, agua líquida). Validación: JANAF/NIST,
    Smith-Van Ness ejemplos 13.4, 13.5 y 13.14 (tests/test_reaction_equilibrium.py)."""
    return RE.reaction_equilibrium(reactions, T_K, P_Pa, feed_mol, data_overrides, fugacity_coefficients,
                                   cantera_mechanism)


def binary_distillation_design(components: list[str], pressure_Pa: float, feed: dict, x_distillate: float,
                            x_bottoms: float | None = None, recovery_in_distillate: float | None = None,
                            model: str = "ideal", parameters: dict | None = None, n_stages: float | None = None,
                            reflux_ratio: float | None = None, psat: dict | None = None, vapor_phase: str = "ideal",
                            wilson_volumes_m3_mol: dict | None = None, uniquac_rq: dict | None = None) -> dict:
    """Diseño de una columna de destilación BINARIA sobre el diagrama de McCabe-Thiele con la curva de equilibrio REAL de
    un modelo de actividad (γ-φ a la presión de la columna), sin DWSIM. Úsala en vez de `shortcut_column` siempre que la
    mezcla no sea ideal (alcohol-agua, cualquier par con azeótropo o con α muy variable): ahí Fenske-Underwood con α
    constante da N_min y R_min equivocados por decenas de por ciento y no ve el azeótropo. También sirve con Raoult.

    components: [ligero, pesado] (el 1.º debe ser el más volátil entre fondo y destilado). pressure_Pa: presión de la columna.
    feed: {z1 (fracción mol del componente 1), molar_flow (opcional, cualquier unidad: los flujos de producto salen en la
      misma), y UNA condición térmica: q, vapor_fraction o T_K (q se estima con Cp y ΔHvap de los puros; sin nada, q = 1)}.
    x_distillate: fracción mol del componente 1 en el destilado. Da además x_bottoms O recovery_in_distillate (fracción del
      componente 1 alimentado que sale por el destilado).
    model: "ideal" (Raoult), "Wilson", "NRTL" o "UNIQUAC". parameters: None = Wilson/NRTL de la base de ChemSep incluida en
      thermo (queda marcado como SUPUESTO de librería); o {b12, b21 [K], a12, a21, alpha (NRTL), V (Wilson), r, q (UNIQUAC)}
      en la convención de vle_binary_regression; o {"energy_J_mol": [d12, d21]}; o {"lambda12", "lambda21"} (Wilson con Λ
      constantes de libro).
    n_stages: etapas de equilibrio disponibles (incluye el rehervidor; condensador total aparte) → reflujo requerido.
    reflux_ratio: reflujo de operación → etapas y plato óptimo de alimentación.
    psat: {nombre: {"antoine": {A, B, C, log, P_unit, T_unit}}} si se quiere otra presión de vapor que la de thermo.
    vapor_phase: "ideal" (por defecto) o "virial".

    Devuelve products (x, D/F, flujos, recuperaciones, T de burbuja), feed {q, q_basis}, azeotropes, relative_volatility
    (cabeza, alimentación, fondo), N_min (reflujo total; exacto para este ELV), R_min con el tipo de pinch (alimentación o
    TANGENTE) y x_pinch, for_n_stages {reflux_ratio_required, optimal_feed_stage} y at_reflux_ratio {stages,
    optimal_feed_stage}, constant_alpha_comparison (Fenske con α media, solo como contraste), assumptions, warnings y
    evidence. Si x_distillate queda más allá del azeótropo o las etapas dadas son menos que N_min, feasible=false con el
    motivo: repórtalo así, no fuerces un número. R_min, etapas a R y R requerido suponen flujo molar constante (sin balance
    de entalpía por etapa): para cargas térmicas y perfiles usa después la columna rigurosa con ese reflujo como arranque.
    Validación: benceno-tolueno contra Fenske/Underwood analíticos con α constante y etanol-agua (azeótropo, pinch
    tangente) en tests/test_mccabe_thiele.py."""
    from packages.vle import mccabe as MC
    from packages.vle.equilibrium import DataError
    from packages.vle.tool import MissingData
    try:
        return MC.design(components, pressure_Pa, feed, x_distillate, x_bottoms, recovery_in_distillate, model, parameters, n_stages, reflux_ratio, psat, vapor_phase, wilson_volumes_m3_mol, uniquac_rq)
    except (DataError, MissingData, KeyError) as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}


def shortcut_distillation_design(components: list[str], feed: dict, pressure_Pa: float, light_key: str, heavy_key: str,
                                 recovery_light_key_in_distillate: float, recovery_heavy_key_in_bottoms: float,
                                 model: str, n_stages: float | None = None, reflux_ratio: float | None = None,
                                 R_over_Rmin: float | None = None, N_over_Nmin: float | None = None,
                                 condenser: str = "total", kij: list[dict] | None = None,
                                 activity_parameters: list[dict] | None = None) -> dict:
    """Diseño por ATAJO de una columna multicomponente (Fenske-Underwood-Gilliland-Kirkbride) sin DWSIM, con K de
    independent_flash. Úsala en vez del `shortcut_column` de DWSIM: aquel evalúa la volatilidad relativa solo en la
    alimentación e infla N_min (~10 % en cortes de puntos de ebullición cercanos) y con él R, N y las cargas; esta usa
    la media geométrica de α en cabeza (rocío del vapor de cabeza) y fondo (burbuja) a la presión de la columna.

    components: nombres. feed: {molar_flows {comp: flujo} o mole_fractions + molar_flow; condición térmica: q, o
      vapor_fraction, o T_K, con P_Pa opcional de la alimentación (si difiere de la columna se hace la expansión
      adiabática); sin nada = líquido saturado a la presión de la columna}. pressure_Pa: presión de la columna.
    light_key / heavy_key y sus recuperaciones (fracción 0.5–1: clave ligera al destilado, clave pesada a fondos).
    model: el de independent_flash ("PR", "SRK", "NRTL"...; kij / activity_parameters opcionales).
    Da UNO de: n_stages, reflux_ratio, R_over_Rmin, N_over_Nmin (ninguno = solo N_min y R_min).
    condenser: "total" o "partial".

    Devuelve feed {q}, products (flujos y fracciones por componente con el reparto de Fenske de los no clave, T de
    cabeza y fondo), relative_volatility_to_heavy_key (cabeza, fondo, alimentación, media usada), N_min, R_min,
    operation {reflux_ratio, stages (incluye el rehervidor), feed_stage_from_top (Kirkbride), condenser_duty,
    reboiler_duty en J/tiempo según las unidades del flujo}, sensitivity (N_min con α solo de la alimentación, como
    contraste), warnings y evidence. Es un atajo: Underwood con una raíz y flujo molar constante; confirma con una
    columna rigurosa. Para binarios no ideales o con azeótropo usa binary_distillation_design."""
    from packages.vle import fug as FUG
    try:
        return FUG.design(components, feed, pressure_Pa, light_key, heavy_key, recovery_light_key_in_distillate, recovery_heavy_key_in_bottoms, model, n_stages, reflux_ratio, R_over_Rmin, N_over_Nmin, condenser, kij, activity_parameters)
    except (ValueError, KeyError) as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}


PROPERTIES_TOOLS = [compound_properties, independent_flash, compare_flash, cubic_eos_volume,
                    binary_distillation_design,
                    shortcut_distillation_design, reaction_equilibrium]
