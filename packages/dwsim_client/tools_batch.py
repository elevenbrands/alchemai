"""Agent tools for batch operations (bloque M26_E2): batch / fed-batch fermentation with vessel scheduling and cost, and
batch distillation. Engines and evidence live in `packages.bioprocess.batch` (scipy integration; BioSTEAM only for the
NREL fermentor cost items, imported lazily and with its global state restored)."""
from __future__ import annotations

from packages.bioprocess import batch as BT


def batch_fermentation(kinetics: dict, initial: dict, t_end_s: float, feed: dict | None = None, target: dict | None = None,
                       stoichiometry: dict | None = None, heat: dict | None = None, agitation_W_m3: float | None = None,
                       schedule: dict | None = None, year: int = 2023, cepci: float | None = None,
                       n_points: int = 61) -> dict:
    """Fermentación por lotes o por lote alimentado (fed-batch): perfil X, S, P, V contra el tiempo, tiempo de lote hasta
    un objetivo, productividad, CO2, calor y pico de enfriamiento; y, si se pide, número y tamaño de tanques para una
    capacidad continua equivalente con su costo de compra. Todo en SI (kg/m³ = g/L; tiempos en s; 1 h⁻¹ = 1/3600 s⁻¹).

    kinetics (TODO del usuario, nada se supone): mu_max_1_s, Ks_kg_m3, Yxs_kg_kg (kg biomasa/kg sustrato); opcionales
      maintenance_kg_kg_s (kg sustrato/kg biomasa/s), kd_1_s (muerte), alpha_kg_kg y beta_kg_kg_s (Luedeking-Piret
      qP = alpha·mu + beta), Yps_kg_kg (sustrato gastado en producto como término aparte; sin él se considera incluido
      en Yxs) e inhibition = lista de {model: 'andrews', Ki_kg_m3} (sustrato, S/(Ks+S+S²/Ki)) | {model: 'levenspiel',
      P_max_kg_m3, n} ((1−P/Pmax)^n) | {model: 'aiba', k_m3_kg} (exp(−k·P)) | {model: 'hyperbolic', Kp_kg_m3} (Kp/(Kp+P)).
      Valores PUBLICADOS que puedes proponer al usuario y pasar explícitamente (revisión de Oliveira et al. 2017,
      IntechOpen, doi:10.5772/65460, leída el 2026-11-16; son de otras cepas y condiciones): levadura floculante en
      continuo (Hojo et al. 1999) mu_max 0.6 h⁻¹, Ks 0.57 g/L, Levenspiel P_max 80 g/L y n 1.8, kd 0.0054 h⁻¹; etanol
      asociado al crecimiento alpha 4.17 g/g; Yxs sobre glucosa 0.11 g/g (levadura) y 0.05 g/g (Zymomonas), máximo
      estequiométrico 0.511 g etanol/g glucosa; intervalos típicos mu_max 0.11-0.56 h⁻¹, Ks 0.07-0.57 g/L.
    initial: {volume_m3, biomass_kg_m3, substrate_kg_m3, product_kg_m3}. t_end_s: tiempo a simular.
    feed (lote alimentado): {flow_m3_s, substrate_kg_m3, start_s, stop_s, exponential_rate_1_s (F = F0·e^(k·(t−start))),
      max_volume_m3}. target (uno): {substrate_conversion} | {product_kg_m3} | {biomass_kg_m3} | {substrate_kg_m3}.
    stoichiometry: {substrate_formula, biomass_formula, product_formula, co2: 'carbon_balance' | {mol_per_mol_product}
      | {mol_per_kg_substrate}} → CO2 y recuperación de carbono. heat: {basis: 'substrate', heat_J_per_kg_substrate} o
      {basis: 'combustion', heats_of_combustion_J_kg: {substrate, biomass, product}} (Hess). agitation_W_m3: P/V del
      usuario (entra en el enfriamiento y en la potencia del agitador).
    schedule: {broth_rate_m3_s | product_rate_kg_s, fill_time_s, empty_time_s, clean_time_s, working_volume_fraction,
      n_vessels | max_vessel_volume_m3, cost (true)}: ciclo = llenado + reacción + vaciado + limpieza; costo con las
      partidas NRELAnaerobicBatchBioreactor de BioSTEAM (NREL 2011: un punto de 3785 m³ por tanque, acero 304; se avisa
      fuera de 378-3785 m³) escalado a year (2015-2023) o cepci; cada equipo sale listo para capital_investment
      ({name, F_BM de BioSTEAM, purchase_cost_usd, base_purchase_cost_usd, basis}).
    Devuelve profile, summary (al objetivo o al final), final, target, co2, heat, cooling_duty, schedule, warnings,
    assumptions, not_included, method y evidence (cierre de balances, segunda integración, solución cerrada de Monod
    cuando aplica; verified certifica los números de esta corrida, no la cinética del usuario)."""
    return BT.batch_fermentation(kinetics, initial, t_end_s, feed, target, stoichiometry, heat, agitation_W_m3, schedule,
                                 year, cepci, n_points)


def batch_distillation(components: list[str], charge_mol: float, charge_mole_fractions: list[float], equilibrium: dict,
                       stop: dict, mode: str = "rayleigh", n_stages: int | None = None, reflux_ratio: float | None = None,
                       distillate_mole_fraction: float | None = None, boilup_rate_mol_s: float | None = None,
                       heat_of_vaporization_J_mol: float | None = None, n_points: int = 41) -> dict:
    """Destilación por lotes de una carga (base molar), con flujo molar constante y retención despreciable en la columna.

    mode: 'rayleigh' (destilación diferencial simple, una etapa, binaria o multicomponente); 'constant_reflux'
      (rectificación con n_stages etapas teóricas CONTANDO el hervidor y reflux_ratio = L/D constante; el destilado
      cambia con el tiempo; multicomponente solo con volatilidades constantes); 'constant_distillate' (binaria: se
      mantiene distillate_mole_fraction del componente 1 subiendo el reflujo, McCabe-Thiele en cada paso).
    components, charge_mol (mol), charge_mole_fractions: la carga del hervidor.
    equilibrium (dato del usuario): {type: 'constant_alpha', alpha: [una por componente]} | {type: 'table', x: [...],
      y: [...]} (binaria, fracción molar del componente 1; no se extrapola) | {type: 'thermo', model: 'PR' | 'SRK' |
      'NRTL' | 'UNIQUAC' | 'WILSON' | 'UNIFAC' | 'UNIFAC_DORTMUND' | 'IDEAL', P_Pa, kij, activity_parameters} (valores K
      de independent_flash en el punto de burbuja).
    stop (uno): {residue_mole_fraction, component} | {distilled_fraction} (destilado/carga) |
      {average_distillate_mole_fraction, component} | {time_s} (necesita boilup_rate_mol_s) | {max_reflux_ratio} (solo
      'constant_distillate').
    boilup_rate_mol_s: vapor generado en el hervidor → tiempo de lote y perfiles contra el tiempo.
    heat_of_vaporization_J_mol: para el calor del hervidor (con 'thermo' se toma del flash si no se da).
    Devuelve final (moles y composiciones del residuo y del destilado acumulado e instantáneo, recuperaciones, tiempo
    de lote, integral de Rayleigh), profile, heat, warnings, assumptions, not_included, method y evidence (cierre del
    balance por componente, independencia de la tolerancia, solución analítica de Rayleigh o límite de Fenske cuando
    aplican). Las etapas son teóricas: la eficiencia de plato y la retención de líquido no se modelan."""
    return BT.batch_distillation(components, charge_mol, charge_mole_fractions, equilibrium, stop, mode, n_stages, reflux_ratio,
                                 distillate_mole_fraction, boilup_rate_mol_s, heat_of_vaporization_J_mol, n_points)


BATCH_TOOLS = [batch_fermentation, batch_distillation]
