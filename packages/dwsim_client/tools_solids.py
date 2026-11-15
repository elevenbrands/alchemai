"""Agent tools for crystallization and drying (bloque solids). No DWSIM flowsheet needed. Engines:
`packages.solids.crystallization` (yield and heat balance, MSMPR population balance, seeded batch cooling by moments) and
`packages.solids.drying` (psychrometrics with CoolProp HumidAirProp, batch and continuous dryers)."""
from __future__ import annotations

from packages.solids import crystallization as CR
from packages.solids import drying as DR


def crystallizer_yield(feed_mass_kg: float, final_T_K: float, solubility: dict, evaporation: dict,
                       feed_solute_mass_fraction: float | None = None, feed_solute_per_kg_solvent: float | None = None,
                       crystal_solute_mass_fraction: float | None = None, hydrate: dict | None = None,
                       solvent: str | None = None, energy: dict | None = None) -> dict:
    """Rendimiento y balance de calor de un cristalizador por enfriamiento, por evaporación o adiabático (vacío): cristales,
    aguas madres (saturadas a la temperatura final), disolvente evaporado y calor a retirar. La curva de solubilidad, el calor
    de cristalización y el cp son del usuario: no hay base de datos interna ni valores por defecto.

    feed_mass_kg: alimentación (por lote; si das kg/s, lee los kg del resultado como kg/s y los J como W).
    final_T_K: temperatura final de las aguas madres.
    solubility: {'T_K': [...], 'kg_solute_per_kg_solvent': [...]} en kg de soluto ANHIDRO por kg de disolvente; un solo punto
      vale si es el de la temperatura final; entre puntos se interpola (PCHIP con ≥ 3) y nunca se extrapola.
    evaporation (obligatorio): {'mode': 'none'} | {'mode': 'mass', 'solvent_evaporated_kg'} | {'mode': 'fraction_of_feed',
      'fraction'} (fracción de la masa total) | {'mode': 'fraction_of_feed_solvent', 'fraction'} | {'mode': 'adiabatic'} (la
      evaporación sale del balance de entalpía con calor retirado = 0; exige energy).
    feed_solute_mass_fraction (soluto anhidro / solución) O feed_solute_per_kg_solvent (kg/kg de disolvente).
    crystal_solute_mass_fraction (1.0 si el cristal es anhidro) O hydrate {'anhydrous_molar_mass_kg_mol',
      'solvent_moles_per_mole', opcional 'solvent_molar_mass_kg_mol'}; con solvent='water' la masa molar y el calor latente
      del agua salen de CoolProp (IAPWS-95).
    energy (opcional): {'feed_T_K', 'solution_cp_J_kgK', y 'heat_of_crystallization_J_kg' (por kg de cristal, positivo =
      liberado) o 'heat_of_solution_J_mol' (por mol de cristal, positivo = disolución endotérmica), opcional
      'solvent_latent_heat_J_kg'}.

    Devuelve crystals_kg, mother_liquor_kg, solvent_evaporated_kg, crystal, mother_liquor, yield
    {solute_recovered_fraction, crystals_per_kg_feed}, heat {heat_to_remove_J y su desglose}, evidence (cierre de masa total,
    soluto y disolvente, saturación de las aguas madres, cierre de entalpía), warnings, assumptions y not_included.
    Validación: Geankoplis 12.11-1 y 12.11-2 y McCabe 27.1."""
    return CR.crystallizer_yield(feed_mass_kg, final_T_K, solubility, evaporation, feed_solute_mass_fraction,
                                 feed_solute_per_kg_solvent, crystal_solute_mass_fraction, hydrate, solvent, energy)


def crystal_size_distribution(mode: str, crystal_density_kg_m3: float, volume_shape_factor: float,
                              residence_time_s: float | None = None, growth_rate_m_s: float | None = None,
                              nucleation_rate_m3_s: float | None = None, magma_density_kg_m3: float | None = None,
                              nucleation_law: dict | None = None, suspension_volume_m3: float | None = None,
                              sizes_m: list | None = None, sieve: dict | None = None, batch: dict | None = None) -> dict:
    """Distribución de tamaños de cristal por balance de población. La cinética (G, B0 o sus leyes), la densidad del cristal
    y el factor de forma de volumen kv (volumen = kv·L³) son datos del usuario.

    mode='msmpr': cristalizador continuo MSMPR en estado estacionario. residence_time_s y dos de growth_rate_m_s (G),
      nucleation_rate_m3_s (B0, núcleos/m³·s) y magma_density_kg_m3 (kg de cristal por m³ de suspensión), o bien
      magma_density_kg_m3 con nucleation_law {'k', 'growth_exponent', 'magma_exponent'} (B0 = k·G^i·MT^j en SI). Opcionales
      suspension_volume_m3 (da caudal y producción) y sizes_m (tamaños donde tabular). Devuelve n0, G, B0, moments, sizes
      {number_mean_m = Gτ, dominant_mass_m = 3Gτ, mass_median_m = 3.67Gτ, mass_mean_m = 4Gτ}, coefficient_of_variation (50 % en
      masa), magma_density_kg_m3, producción y distribution (n(L), masa y número acumulados).
    mode='msmpr_from_sieve': el inverso. residence_time_s, magma_density_kg_m3 y sieve {'lower_m': [...], 'upper_m': [...],
      'mass_fraction': [...]} (fracciones 0-1 retenidas entre dos aberturas; None en la abertura que falta del fondo o de los
      gruesos). Devuelve G y B0 de la regresión de ln n frente a L con R², errores y la tabla; mid_size_bias (sesgo de la
      aproximación de tamaño medio con esos cortes), exact_cut_fit (ajuste exacto de las fracciones) y la densidad de magma
      que implica el ajuste frente a la dada.
    mode='batch_seeded': cristalización por lotes sembrada con enfriamiento, método de los momentos, por kg de disolvente.
      batch = {'solubility': {'T_K', 'kg_solute_per_kg_solvent'}, 'initial_concentration_kg_kg', 'cooling': {'time_s': [0, ...],
      'T_K': [...]}, 'seed_mass_kg_per_kg_solvent', 'seed_size_m', 'growth': {'k_m_s', 'order'} (G = k·ΔC^order, ΔC en kg/kg),
      'nucleation': {'k_per_kg_s', 'order', 'magma_exponent'} o None}. Devuelve final (concentración, sobresaturación, masa
      de cristales, tamaño final de la siembra, fracción nucleada, tamaños medios), moments y profile.

    Siempre: evidence (momentos numéricos frente a analíticos, tercer momento·ρ·kv = densidad de magma, balance de soluto,
    independencia del paso), warnings, assumptions y not_included. Validación: identidades analíticas del MSMPR, McCabe 27.6 y
    límites analíticos del modelo por lotes; la regresión de tamices se valida con fracciones MSMPR exactas."""
    if mode == "msmpr":
        if residence_time_s is None:
            raise ValueError("falta residence_time_s: tiempo de residencia τ = volumen de suspensión / caudal de producto")
        return CR.msmpr(crystal_density_kg_m3, volume_shape_factor, residence_time_s, growth_rate_m_s, nucleation_rate_m3_s,
                        magma_density_kg_m3, nucleation_law, suspension_volume_m3, sizes_m)
    if mode == "msmpr_from_sieve":
        if residence_time_s is None or magma_density_kg_m3 is None or sieve is None:
            raise ValueError("msmpr_from_sieve necesita residence_time_s, magma_density_kg_m3 (kg de cristal por m³ de "
                             "suspensión) y sieve {'lower_m', 'upper_m', 'mass_fraction'}")
        return CR.msmpr_from_sieve(crystal_density_kg_m3, volume_shape_factor, residence_time_s, magma_density_kg_m3, sieve)
    if mode == "batch_seeded":
        if batch is None:
            raise ValueError("batch_seeded necesita batch {'solubility', 'initial_concentration_kg_kg', 'cooling', "
                             "'seed_mass_kg_per_kg_solvent', 'seed_size_m', 'growth', 'nucleation'}")
        return CR.batch_seeded_cooling(crystal_density_kg_m3, volume_shape_factor, batch)
    raise ValueError("mode debe ser 'msmpr', 'msmpr_from_sieve' o 'batch_seeded'")


def dryer_batch(air: dict, solid: dict | None = None, constant_rate: dict | None = None,
                falling_rate: dict | None = None) -> dict:
    """Secado por lotes y psicrometría. Solo con air devuelve el estado psicrométrico; con constant_rate, la velocidad de
    secado constante; con solid, los tiempos del periodo constante y del decreciente. Humedades del sólido en base seca
    (kg agua/kg sólido seco) y TOTALES. Humedad crítica y de equilibrio y la curva de secado son datos del usuario.

    air: {'T_K', 'pressure_Pa' y UNO de 'humidity_ratio_kg_kg' (kg agua/kg aire seco), 'relative_humidity' (0-1),
      'wet_bulb_T_K', 'dew_point_T_K', 'water_partial_pressure_Pa'}.
    solid: {'initial_moisture', 'final_moisture', 'critical_moisture', 'equilibrium_moisture' (0 si ya das humedad libre), y
      la carga: 'dry_solid_per_area_kg_m2' o 'dry_solid_mass_kg' + 'drying_area_m2'; con circulación a través del lecho:
      'bed_depth_m' + 'bulk_density_kg_m3'}.
    constant_rate: {'rate_kg_m2s'} medida, o calculada por transferencia de calor a la superficie a bulbo húmedo:
      {'configuration': 'parallel' | 'perpendicular', 'velocity_m_s' o 'mass_velocity_kg_m2s', opcional 'h_W_m2K'} (sin h se
      estima con Geankoplis 9.6-9 / 9.6-10 y se avisa), o {'configuration': 'through_circulation', 'superficial_velocity_m_s' o
      'dry_air_mass_velocity_kg_m2s', 'particle': {'shape': 'sphere' | 'cylinder', 'diameter_m', 'length_m', 'void_fraction'} o
      {'specific_area_m2_m3', 'equivalent_diameter_m'}, opcional 'h_W_m2K'} (Geankoplis 9.10-13 si Re > 350).
    falling_rate: {'model': 'linear'} (velocidad proporcional a la humedad libre; 'capillary' es la misma ley) |
      {'model': 'diffusion', 'diffusivity_m2_s', 'thickness_m', 'drying_faces': 1 | 2} (placa, serie completa) |
      {'model': 'curve', 'moisture': [...], 'rate_kg_m2s': [...]} (tu curva de velocidad; integra dX/R).

    Devuelve air (humidity_ratio_kg_kg, relative_humidity, wet_bulb_T_K = saturación adiabática, dew_point_K,
    humid_heat_J_kgK, humid_volume_m3_kg, enthalpy_J_kg por kg de aire seco), constant_rate (h, Rc, correlación), drying
    {constant_rate_time_s, falling_rate_time_s, total_time_s, total_time_h}, solid, drying_curve, evidence (balance de
    saturación adiabática, forma cerrada frente a cuadratura numérica, convergencia de la serie), warnings, assumptions y
    not_included (radiación y conducción, encogimiento, spray y flash). Validación: Geankoplis 9.3-1 a 9.3-4, 9.6-2, 9.6-3,
    9.7-1, 9.7-2, 9.9-1, 9.10-1 y McCabe 24.1 y 24.2."""
    return DR.dryer_batch(air, solid, constant_rate, falling_rate)


def dryer_continuous(air_in: dict, solid: dict, flow: str, heat_loss_W: float, air_outlet_T_K: float | None = None,
                     dry_air_flow_kg_s: float | None = None, transfer_units: float | None = None,
                     sizing: dict | None = None, air_heater_inlet_T_K: float | None = None) -> dict:
    """Secador continuo en estado estacionario (rotatorio, de túnel o de banda) como contactor aire-sólido 1-D: balances de
    agua y de entalpía, aire necesario o estado del aire de salida, calor transferido, unidades de transferencia y longitud.
    Los secadores por aspersión (spray) y flash quedan fuera.

    air_in: {'T_K', 'pressure_Pa' y un dato de humedad} como en dryer_batch (aire que entra al secador, ya caliente).
    solid: {'dry_solid_flow_kg_s', 'inlet_moisture', 'outlet_moisture' (base seca), 'inlet_T_K', 'outlet_T_K',
      'dry_solid_cp_J_kgK'}. flow: 'countercurrent' | 'cocurrent'. heat_loss_W: pérdidas al ambiente (0 si adiabático).
    Exactamente uno de: air_outlet_T_K (sale el caudal de aire), dry_air_flow_kg_s (sale la temperatura de salida) o
      transfer_units (Nt de diseño; Badger-Banchero recomiendan 1.5-2: fija la salida con Nt = ln[(T_ent − Tw)/(T_sal − Tw)]).
    sizing (opcional): uno de {'diameter_m', 'cross_section_area_m2', 'gas_mass_velocity_kg_m2s'} más
      'volumetric_heat_transfer_coefficient_W_m3K' (Ua, dato tuyo) o 'correlation': 'mccabe_rotary' (Ua estimado, se avisa).
    air_heater_inlet_T_K (opcional): temperatura del aire antes del calentador; da air_heater_duty_W y thermal_efficiency.

    Devuelve air_in, air_out (estados psicrométricos), dry_air_flow_kg_s, water_evaporated_kg_s, heat_transferred_to_solid_W,
    heat_per_kg_water_evaporated_J_kg, transfer_units, log_mean_temperature_difference_K, sizing {diameter_m, length_m,
    volume_m3, ...}, evidence (cierre de agua y de entalpía, calor frente a vapor IAPWS, cruce de temperaturas), warnings,
    assumptions y not_included (tiempo de residencia, zonas de precalentamiento y humedad ligada). Validación: Geankoplis
    9.10-2 y McCabe 24.4."""
    return DR.dryer_continuous(air_in, solid, flow, heat_loss_W, air_outlet_T_K, dry_air_flow_kg_s, transfer_units, sizing,
                               air_heater_inlet_T_K)


SOLIDS_TOOLS = [crystallizer_yield, crystal_size_distribution, dryer_batch, dryer_continuous]
