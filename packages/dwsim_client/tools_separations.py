"""Agent tools for membrane separation and adsorption (bloque membads). No DWSIM flowsheet needed. Engines:
`packages.separations.membrane` (solution-diffusion gas permeation in four flow patterns, reverse osmosis) and
`packages.separations.adsorption` (isothermal fixed-bed breakthrough with LDF, LUB scale-up, Skarstrom PSA cycle)."""
from __future__ import annotations

import math

import numpy as np

from packages.separations import adsorption as AD
from packages.separations import membrane as MB


def _clean(x):
    """JSON-safe copy: numpy scalars to Python, NaN to None, infinities to the strings 'inf' / '-inf'."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def gas_permeation_module(feed_flow_mol_s: float, feed_composition: dict, feed_pressure_Pa: float,
                          permeate_pressure_Pa: float, flow_pattern: str, permeance_mol_m2_s_Pa: dict | None = None,
                          permeability_mol_m_m2_s_Pa: dict | None = None, thickness_m: float | None = None,
                          area_m2: float | None = None, stage_cut: float | None = None, target: dict | None = None) -> dict:
    """Un módulo de membrana para gases por solución-difusión (multicomponente): J_i = Q_i (p_alta x_i − p_baja y_i), presión
    uniforme en cada lado, isotérmico. Las permeancias son datos medidos de tu membrana: no hay base de datos ni valores por
    defecto.

    feed_flow_mol_s: caudal molar de alimentación. feed_composition: {componente: fracción molar} (≥ 2, suman 1).
    feed_pressure_Pa / permeate_pressure_Pa: presiones absolutas (permeado puede ser 0 = vacío ideal).
    flow_pattern: 'complete_mixing' | 'cross_flow' | 'cocurrent' | 'countercurrent' (obligatorio; cambia área y pureza).
    permeance_mol_m2_s_Pa {componente: Q} O permeability_mol_m_m2_s_Pa {componente: P} con thickness_m (Q = P/espesor).
      Conversión útil: 1 GPU = 3.346e-10 mol/(m2 s Pa); 1 barrer = 3.346e-16 mol m/(m2 s Pa).
    Exactamente uno de: area_m2 (evaluar un módulo existente), stage_cut (fracción de la alimentación que permea; diseño) o
      target {'quantity': 'permeate_mole_fraction' | 'retentate_mole_fraction' | 'recovery_in_permeate' |
      'recovery_in_retentate', 'component', 'value'} (diseño: el área mínima que lo cumple).

    Devuelve area_m2, stage_cut, permeate y retentate {flow_mol_s, mole_fractions, component_flows_mol_s, pressure_Pa},
    recoveries por componente, limits (selectividad ideal, permeado a corte 0, techo por relación de presiones y_i ≤ x_i p_h/p_l,
    retenido mínimo de mezcla completa), profile (lados en flujo pistón), evidence (balance por componente con el permeado
    integrado aparte, cambio al relajar la tolerancia, convergencia del disparo en contracorriente, cuadrática cerrada del
    binario en mezcla completa), warnings, assumptions y not_included. Si un objetivo de diseño no se alcanza devuelve
    feasible=False con el intervalo alcanzable. Validación: Geankoplis 13.4-1, 13.4-2, 13.4-3, 13.5-1 y 13.6-1 y Binous (Chem.
    Eng. Educ. 2006) para los cuatro patrones."""
    return _clean(MB.gas_permeation(feed_flow_mol_s, feed_composition, feed_pressure_Pa, permeate_pressure_Pa, flow_pattern,
                                    permeance_mol_m2_s_Pa, permeability_mol_m_m2_s_Pa, thickness_m, area_m2, stage_cut, target))


def reverse_osmosis(feed_concentration_kg_m3: float, solute_molar_mass_kg_mol: float, ions_per_formula: float, T_K: float,
                    feed_pressure_Pa: float, permeate_pressure_Pa: float, water_permeability_kg_m2_s_Pa: float,
                    solute_permeability_m_s: float, feed_flow_m3_s: float | None = None, area_m2: float | None = None,
                    recovery: float | None = None, osmotic_coefficient: float | None = None,
                    mass_transfer_coefficient_m_s: float | None = None, permeate_density_kg_m3: float | None = None,
                    pump_efficiency: float | None = None, energy_recovery_efficiency: float | None = None) -> dict:
    """Ósmosis inversa de un soluto por solución-difusión (Geankoplis cap. 13): N_w = A_w (ΔP − Δπ), N_s = A_s (c_m − c_p).
    A_w y A_s son las constantes medidas de tu membrana (no se estiman).

    feed_concentration_kg_m3: soluto en la alimentación. solute_molar_mass_kg_mol e ions_per_formula (2 para NaCl, 1 si no
      se disocia) para la presión osmótica π = φ ν c R T / M; T_K: temperatura.
    osmotic_coefficient: φ del usuario; si falta se usa van 't Hoff ideal (φ = 1) y se declara en assumptions (sobrestima π de
      sales reales unos puntos porcentuales).
    feed_pressure_Pa / permeate_pressure_Pa: presiones absolutas. water_permeability_kg_m2_s_Pa: A_w (kg de agua/(m2 s Pa);
      desde kg/(m2 s atm) divide entre 101325). solute_permeability_m_s: A_s (0 = rechazo total).
    mass_transfer_coefficient_m_s: k del lado de alimentación para la polarización por modelo de película; sin él no hay
      polarización (optimista). permeate_density_kg_m3: si falta, agua pura a T por IAPWS-95 (CoolProp).
    Sin feed_flow_m3_s: solo flujos locales a la entrada (recuperación → 0). Con feed_flow_m3_s y area_m2 (evaluación) o
      recovery (permeado/alimentación en volumen; diseño → área): canal en flujo pistón integrado en el área.
    pump_efficiency y energy_recovery_efficiency (0-1) dan la energía específica real; sin ellos solo la hidráulica ideal.

    Devuelve local_at_feed (flujos de agua y soluto, concentración de permeado, pared y módulo de polarización, rechazos
    observado e intrínseco, presiones osmóticas), y en módulo: area_m2, recovery, permeate y retentate, rejection_observed,
    flujos medios, local_at_outlet, specific_energy (J/m3 y kWh/m3 de permeado, mínimo reversible) y profile. Si la recuperación
    pedida supera el límite osmótico devuelve feasible=False. evidence: residual de la ecuación de flujo, forma cerrada del
    rechazo sin polarización, balances de agua y soluto, tolerancia, integral cerrada del caso de rechazo total. Validación:
    Geankoplis 13.9-1, 13.9-2 y 13.10-1."""
    return _clean(MB.reverse_osmosis(feed_concentration_kg_m3, solute_molar_mass_kg_mol, ions_per_formula, T_K,
                                     feed_pressure_Pa, permeate_pressure_Pa, water_permeability_kg_m2_s_Pa,
                                     solute_permeability_m_s, feed_flow_m3_s, area_m2, recovery, osmotic_coefficient,
                                     mass_transfer_coefficient_m_s, permeate_density_kg_m3, pump_efficiency,
                                     energy_recovery_efficiency))


def adsorption_fixed_bed(mode: str, bed_length_m: float, breakthrough_fraction: float,
                         superficial_velocity_m_s: float | None = None, bed_void_fraction: float | None = None,
                         bulk_density_kg_m3: float | None = None, components: list | None = None,
                         saturation_fraction: float | None = None, axial_dispersion_m2_s: float | None = None,
                         t_end_s: float | None = None, n_cells: int = 100, lab_curve: dict | None = None,
                         new_breakthrough_time_s: float | None = None, new_bed_length_m: float | None = None,
                         solute_feed_rate_kg_s: float | None = None, adsorbent_mass_kg: float | None = None) -> dict:
    """Adsorción en lecho fijo isotérmico: curva de ruptura simulada o escalado por longitud de lecho no usado (LUB).

    mode='breakthrough': simula un lecho inicialmente limpio. bed_length_m, superficial_velocity_m_s, bed_void_fraction (huecos
      entre partículas), bulk_density_kg_m3 (kg de adsorbente por m3 de lecho), breakthrough_fraction (c/c0 de ruptura, p. ej.
      0.05, criterio tuyo) y components = [{'name', 'feed_concentration_mol_m3', 'k_ldf_1_s' (coeficiente LDF medido o
      ajustado), 'isotherm': {'model': 'linear', 'K_m3_kg'} | {'model': 'langmuir', 'q_max_mol_kg', 'b_m3_mol'} |
      {'model': 'freundlich', 'K_F', 'n'} (q = K_F c^(1/n))}]; varios Langmuir compiten (Langmuir extendido). Opcionales:
      saturation_fraction (por defecto 1 − breakthrough_fraction, declarado), axial_dispersion_m2_s (sin él flujo pistón),
      t_end_s, n_cells (malla; se repite con la mitad para verificar). Devuelve por componente breakthrough_time_s,
      saturation_time_s, stoichiometric_time_s, usable/total_capacity_time_s, bed_usage_at_breakthrough,
      length_of_unused_bed_m y mass_transfer_zone_length_m, la curva c/c0(t) y evidence (balance de materia, integral de
      1 − c/c0 = tiempo estequiométrico, malla, solución exacta de Anzelius/Thomas si la isoterma es lineal sin dispersión,
      ancho de patrón constante si es Langmuir). Supone velocidad constante: adsorbato diluido o líquido.
    mode='lub_scaleup': desde tu curva de laboratorio lab_curve {'time_s': [...], 'c_over_c0': [...]} con bed_length_m (del
      ensayo) y breakthrough_fraction: t_b, t_t = ∫(1 − c/c0)dt, t_u, fracción usada y LUB (Geankoplis 12.3D). Opcional
      new_breakthrough_time_s (→ longitud del lecho nuevo) O new_bed_length_m (→ su tiempo de ruptura), a la misma velocidad
      superficial, partícula y alimentación; solute_feed_rate_kg_s con adsorbent_mass_kg dan la capacidad de saturación.

    Siempre warnings, assumptions, not_included, method y evidence.verified. Validación: Anzelius/Thomas (exacta), patrón
    constante de Langmuir con LDF (forma cerrada) y Geankoplis 12.3-1 (LUB)."""
    if mode == "breakthrough":
        return _clean(AD.fixed_bed_breakthrough(bed_length_m, superficial_velocity_m_s, bed_void_fraction, bulk_density_kg_m3,
                                                components, breakthrough_fraction, saturation_fraction, axial_dispersion_m2_s,
                                                t_end_s, n_cells))
    if mode == "lub_scaleup":
        if lab_curve is None:
            raise ValueError("Falta lab_curve {'time_s': [...], 'c_over_c0': [...]}: la curva de ruptura medida en el laboratorio "
                             "es la base del escalado LUB.")
        return _clean(AD.lub_scaleup(lab_curve.get("time_s"), lab_curve.get("c_over_c0"), bed_length_m, breakthrough_fraction,
                                     new_breakthrough_time_s, new_bed_length_m, solute_feed_rate_kg_s, adsorbent_mass_kg))
    raise ValueError("mode debe ser 'breakthrough' (simular el lecho) o 'lub_scaleup' (escalar desde una curva medida).")


def psa_cycle(bed_length_m: float, bed_void_fraction: float, bulk_density_kg_m3: float, T_K: float, components: list,
              product_component: str, high_pressure_Pa: float, low_pressure_Pa: float, feed_velocity_m_s: float,
              purge_velocity_m_s: float, pressurization_time_s: float, feed_time_s: float, blowdown_time_s: float,
              purge_time_s: float, pressurization_with: str, bed_diameter_m: float | None = None, n_cells: int = 40,
              max_cycles: int = 200, css_tolerance: float = 1e-4) -> dict:
    """Ciclo PSA de Skarstrom de cuatro pasos (presurización, alimentación a presión alta, despresurización a contracorriente,
    purga a contracorriente con producto) de un gas binario en un lecho isotérmico, repetido hasta el estado cíclico
    estacionario. Herramienta de tendencias, no de diseño industrial.

    components: dos gases [{'name', 'feed_mole_fraction', 'k_ldf_1_s', 'isotherm': {'model': 'linear', 'K_m3_kg'} |
      {'model': 'langmuir', 'q_max_mol_kg', 'b_m3_mol'} | {'model': 'freundlich', 'K_F', 'n'}}] con c en mol/m3 de gas.
    product_component: el gas poco adsorbido que sale por el extremo de producto.
    high_pressure_Pa / low_pressure_Pa: presiones absolutas del ciclo. feed_velocity_m_s (a presión alta) y purge_velocity_m_s
      (a presión baja; 0 = sin purga): velocidades superficiales. Duraciones de los cuatro pasos en s.
    pressurization_with: 'feed' o 'product'. bed_diameter_m (opcional) da caudales por lecho.

    Devuelve product_purity_mole_fraction, product_recovery, productivity_mol_kg_s, purge_to_feed ratios, per_cycle_mol_m2
    (alimentación, producto bruto y neto, purga, escapes), perfil del lecho, validation_status y, con isotermas lineales o de
    Langmuir, la teoría de equilibrio (Knaebel-Hill / Shendalman-Mitchell re-derivada; para Langmuir, choque de la
    alimentación y característica más lenta de la purga): tiempo de ruptura, purga de limpieza y la recuperación de este ciclo
    en forma cerrada. evidence: residuo del estado cíclico, cierre de balance del ciclo, malla.
    Validado contra la teoría de equilibrio con isotermas lineales y con Langmuir / Langmuir extendido (presurización con
    producto, alimentación hasta la ruptura o menos, purga por encima de la cota de limpieza; no contra un ciclo publicado).
    Fuera de ese dominio, con presurización con alimentación o con Freundlich queda NO VALIDADO."""
    return _clean(AD.psa_skarstrom(bed_length_m, bed_void_fraction, bulk_density_kg_m3, T_K, components, product_component,
                                   high_pressure_Pa, low_pressure_Pa, feed_velocity_m_s, purge_velocity_m_s,
                                   pressurization_time_s, feed_time_s, blowdown_time_s, purge_time_s, pressurization_with,
                                   bed_diameter_m, n_cells, max_cycles, css_tolerance))


SEPARATIONS_TOOLS = [gas_permeation_module, reverse_osmosis, adsorption_fixed_bed, psa_cycle]
