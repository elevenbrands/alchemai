"""Agent tools for liquid-side equipment (bloque M26_E1): liquid-liquid extraction and decanters, multiple-effect evaporators
and storage tanks, with sizing and purchase cost. No DWSIM flowsheet needed. Engine: `packages.bioprocess.liquid_equipment`
(repo flash engine for the liquid-liquid equilibrium, IAPWS-95 for steam, BioSTEAM / Seider-IDAES / DOE-NETL cost data)."""
from __future__ import annotations

from packages.bioprocess import liquid_equipment as LE


def liquid_liquid_extraction(mode: str, feed: dict, flow_basis: str, solvent: dict | None = None, T_K: float | None = None,
                             P_Pa: float | None = None, model: str | None = None,
                             activity_parameters: list[dict] | None = None, distribution_coefficients: dict | None = None,
                             carrier: str | None = None, extraction_solvent: str | None = None, n_stages: int | None = None,
                             target: dict | None = None, sizing: dict | None = None, costing: dict | None = None,
                             year: int = 2023, cepci: float | None = None, max_sweeps: int = 400,
                             tolerance: float = 1e-9) -> dict:
    """Extracción líquido-líquido: reparto de una mezcla en dos fases líquidas en un decantador o mezclador-sedimentador
    (mode='decanter', una etapa ideal) o en N etapas ideales en contracorriente (mode='countercurrent'), con
    dimensionamiento del decantador por gravedad y costo de compra opcionales.

    feed, solvent: {componente: caudal} en las unidades de flow_basis ('kg_s' o 'mol_s'). solvent es obligatorio en
      contracorriente y opcional en 'decanter' (mezcla que ya trae las dos fases, p. ej. biodiésel + glicerina + metanol).
    El equilibrio se da de UNA de dos formas (no hay valor por defecto):
      a) model + T_K + P_Pa: equilibrio con el motor de flash del repositorio (el mismo de independent_flash). 'NRTL' o
         'UNIQUAC' con TUS activity_parameters [{'i', 'j', 'b_ij_K', 'b_ji_K', opc. 'a_ij', 'a_ji', 'alpha' (NRTL)}], o
         predictivos 'UNIFAC_LLE', 'UNIFAC', 'UNIFAC_DORTMUND'. Los predictivos son una estimación: UNIFAC_LLE no tiene
         grupo para el metanol y, en biodiésel-glicerina-metanol, UNIFAC deja el coeficiente de reparto del metanol entre
         0,24 y 0,55 veces el medido (Lee et al. 2010): para ese sistema usa parámetros ajustados o la opción b.
         extraction_solvent (opcional) dice qué componente identifica a la fase extracto; si no, el mayoritario de solvent.
      b) distribution_coefficients {soluto: K} con carrier (diluyente, se queda en el refinado) y extraction_solvent (se
         queda en el extracto), fases inmiscibles: Y = K·X con X = soluto por unidad de diluyente e Y = soluto por unidad
         de disolvente, en las unidades de los caudales. target {'solute', 'raffinate_ratio'} devuelve las etapas
         necesarias por Kremser (stages_required).
    n_stages: etapas ideales (solo contracorriente).
    sizing (opcional, decantador por gravedad; TODOS son datos tuyos, sin valores por defecto): {'orientation':
      'horizontal'|'vertical', 'droplet_diameter_m', 'residence_time_s', 'length_to_diameter', 'fill_fraction',
      'interface_height_fraction' (altura de la interfase / diámetro, o / altura si es vertical), 'continuous_phase':
      'light'|'heavy', 'continuous_viscosity_Pa_s', opc. 'dispersed_viscosity_Pa_s' (comprobación de arrastre)}. Con model
      las densidades salen de thermo (estimadas; puedes fijarlas con 'light_density_kg_m3', 'heavy_density_kg_m3'); con
      coeficientes constantes hacen falta 'raffinate_density_kg_m3' y 'extract_density_kg_m3' y flow_basis='kg_s'.
      Valores típicos PUBLICADOS que puedes pasar tú si los aceptas (la herramienta no los aplica): gota de 150 µm,
      retención de 2 a 10 min, L/D de 3 a 5 (apuntes de S. D. Al-Bayati, método de Coulson & Richardson vol. 6;
      vuelven en sizing.published_typical_values con su fuente).
    costing (opcional): {'equipment': 'gravity_decanter', 'material': 'CarbonSteel'|'StainlessSteel304'|…} (recipiente de
      Seider vía IDAES, exige sizing) o {'equipment': 'centrifuge'} (correlación de BioSTEAM, hasta 100 m³/h por máquina).
      Una unidad por etapa. year 2015-2023 usa la tabla CEPCI del repositorio; para otro año pasa cepci.

    Devuelve, con model: light_phase y heavy_phase (decanter) o raffinate, extract y stages (contracorriente), cada fase
    con flows_mol_s, flows_kg_s, mole_fractions, mass_fractions, density_kg_m3 y volumetric_flow_m3_s; roles,
    split_to_light_phase, K_light_over_heavy_mole/mass, recovery_to_extract. Con coeficientes: raffinate, extract, solutes
    {K, extraction_factor, fraction_extracted, kremser}, stage_profiles, stages_required. Siempre: sizing {diameter_m,
    length_m, governed_by ('settling'|'residence_time'), stokes_settling_velocity_m_s, interface_area_m2,
    smallest_droplet_separated_m, …}, cost, equipment (lista lista para capital_investment: name, type o F_BM,
    purchase_cost_usd, base_purchase_cost_usd, basis), status, warnings, assumptions, not_included, method y evidence
    (cierre de balances por etapa y global, flash de cada etapa verificado, acuerdo con Kremser, régimen de Stokes). Un
    status distinto de 'ok' (single_liquid, vapor_present, not_verified…) significa que NO hay separación utilizable.
    Validación: Kremser (analítico), McCabe 20.3, Geankoplis 12.5-1 y 12.7-3, decantador de Coulson & Richardson."""
    return LE.liquid_liquid_extraction(mode, feed, flow_basis, solvent, T_K, P_Pa, model, activity_parameters,
                                       distribution_coefficients, carrier, extraction_solvent, n_stages, target, sizing,
                                       costing, year, cepci, max_sweeps, tolerance)


def multi_effect_evaporator(feed_kg_s: float, feed_solids_mass_fraction: float, feed_T_K: float,
                            product_solids_mass_fraction: float, n_effects: int, U_W_m2K: list[float],
                            solution_cp_J_kgK: list[float], boiling_point_rise: dict, steam_T_K: float | None = None,
                            steam_P_Pa: float | None = None, last_effect_P_Pa: float | None = None,
                            last_effect_saturation_T_K: float | None = None, equal_area: bool = True,
                            effect_P_Pa: list[float] | None = None, costing: dict | None = None, year: int = 2023,
                            cepci: float | None = None, tolerance: float = 1e-10) -> dict:
    """Evaporador de múltiple efecto con alimentación en serie (forward feed) para concentrar una solución de soluto no
    volátil (jugo de caña, vinazas, glicerina, salmueras): áreas, presiones y temperaturas de cada efecto, consumo y
    economía de vapor, y costo de compra opcional. Con n_effects = 1 es un evaporador simple.

    feed_kg_s, feed_solids_mass_fraction, feed_T_K: alimentación. product_solids_mass_fraction: concentración final.
    n_effects: número de efectos (decisión de diseño). U_W_m2K: lista con el coeficiente global de cada efecto (tu dato).
    solution_cp_J_kgK: [c0, c1, …] con cp = c0 + c1·x + … J/(kg·K), x = fracción másica de sólidos (tu dato; p. ej. la
      solución de azúcar de Geankoplis es [4190, -2350]).
    boiling_point_rise (elevación del punto de ebullición, tu dato): {'polynomial_K': [c1, c2, …]} (EPE = c1·x + c2·x² + …),
      {'per_effect_K': [...]} o {'duhring': {'solids_mass_fraction': [...], 'slope': [...], 'intercept_K': [...]}}
      (T_ebullición = intercept + slope·T_sat del agua, K; no se extrapola). Sin elevación: {'polynomial_K': [0.0]}.
    Vapor de calefacción saturado: steam_T_K o steam_P_Pa (uno). Último efecto: last_effect_P_Pa o
      last_effect_saturation_T_K (uno; lo fija tu condensador).
    equal_area=True (por defecto): áreas iguales, se resuelven las presiones intermedias. Alternativa: effect_P_Pa con la
      presión de cada efecto, y las áreas resultan distintas.
    costing (opcional): {'type': 'vertical_tube'|'horizontal_tube', opc. 'material_factor' (tuyo; la tabla es acero al
      carbón)}; costo por interpolación en la tabla publicada DOE/NETL-2002/1169 (1998), llevado a year (2015-2023) o cepci.
      La función de costo de evaporadores de BioSTEAM da ~1/15 de esa tabla y solo se informa como segunda opinión.

    Devuelve steam_kg_s, steam_economy (kg evaporados por kg de vapor), total_evaporation_kg_s, product_kg_s, effects
    [{pressure_Pa, vapor_saturation_T_K, boiling_T_K, boiling_point_rise_K, delta_T_K, duty_W, area_m2, liquid_out_kg_s,
    solids_mass_fraction_out, vapor_kg_s}], mean_area_m2, total_area_m2, condenser_duty_W, cost, equipment (lista para
    capital_investment), status, warnings, assumptions, not_included, method y evidence (cierre de materia, de sólidos y
    de entalpía por efecto y global, igualdad de áreas, misma solución desde otro punto de partida). No incluye
    condensador, vacío, precalentadores ni recompresión. Validación: Geankoplis 8.5-1 (áreas, vapor y economía a menos
    del 0,3 % de la solución del libro)."""
    return LE.multi_effect_evaporator(feed_kg_s, feed_solids_mass_fraction, feed_T_K, product_solids_mass_fraction,
                                      n_effects, U_W_m2K, solution_cp_J_kgK, boiling_point_rise, steam_T_K, steam_P_Pa,
                                      last_effect_P_Pa, last_effect_saturation_T_K, equal_area, effect_P_Pa, costing, year,
                                      cepci, tolerance)


def storage_tank(flow_kg_s: float, density_kg_m3: float, fill_fraction: float, holdup_time_s: float | None = None,
                 inventory_days: float | None = None, tank_type: str | None = None, material: str | None = None,
                 aspect_ratio_H_D: float | None = None, number_of_tanks: int | None = None, year: int = 2023,
                 cepci: float | None = None) -> dict:
    """Tanque de almacenamiento o de pulmón: volumen a partir del caudal y del tiempo de retención o de los días de
    inventario, geometría del cilindro y costo de compra.

    flow_kg_s, density_kg_m3: caudal y densidad del líquido. holdup_time_s o inventory_days (exactamente uno).
    fill_fraction: volumen útil / volumen total (0-1]; sin valor por defecto.
    aspect_ratio_H_D (opcional): altura/diámetro; sin él no se dan diámetro ni altura (no hay tabla de tamaños API 650).
    tank_type y material (opcionales, van juntos; sin ellos solo se dimensiona). tank_type: 'cone_roof' (techo cónico,
      37,9-3 785 m³ por tanque), 'floating_roof' (techo flotante, 114-3 785 m³), 'field_erected' (montado en campo, hasta
      50 000 m³) o 'mix_tank' (tanque de proceso con agitación, 0,1-30 m³): correlaciones de BioSTEAM (Seider et al. y
      Apostolakou et al.). material: 'carbon_steel', 'stainless_steel', 'copper', 'nickel', 'titanium_clad', 'titanium'
      (factores de material de BioSTEAM). Por encima del máximo se reparte en tanques iguales; por debajo del mínimo el
      costo es una extrapolación y se avisa. number_of_tanks (opcional) fuerza más tanques.
    year 2015-2023 usa la tabla CEPCI del repositorio; para otro año pasa cepci.

    Devuelve working_volume_m3, total_volume_m3, number_of_tanks, volume_per_tank_m3, geometry_per_tank {diameter_m,
    height_m, liquid_height_m}, cost {purchase_cost_usd, base_purchase_cost_usd, volume_range_m3, F_BM (2,3, de BioSTEAM),
    within_correlation_range}, equipment (lista para capital_investment), cross_check_pressure_vessel (el mismo cilindro
    como recipiente a presión de Seider: cota superior), warnings, assumptions, not_included, method y evidence.
    Validación del costo: tabla DOE/NETL-2002/1169 (techo cónico −18 a +18 %, techo flotante −33 a −6 %) y cotizaciones de
    NREL Humbird 2011 (techo flotante −38 y −40 %): el costo de techo flotante queda BAJO frente a ambas fuentes."""
    return LE.storage_tank(flow_kg_s, density_kg_m3, fill_fraction, holdup_time_s, inventory_days, tank_type, material,
                           aspect_ratio_H_D, number_of_tanks, year, cepci)


LIQUID_EQUIPMENT_TOOLS = [liquid_liquid_extraction, multi_effect_evaporator, storage_tank]
