"""Agent tools for solids handling and oil extraction (bloque M26_E3). No DWSIM flowsheet needed. Engine:
`packages.bioprocess.solids_handling` (Bond/Kick/Rittinger, conveyor power, constant-pressure filtration, Sigma theory,
expeller balance, countercurrent leaching stages; purchase costs with BioSTEAM's correlations read from the installed
classes, global state of BioSTEAM/thermo left untouched)."""
from __future__ import annotations

from packages.bioprocess import solids_handling as SH


def size_reduction(mass_flow_kg_s: float, law: str, feed_size_m: float | None = None, product_size_m: float | None = None,
                   work_index_kWh_per_ton: float | None = None, ton: str | None = None,
                   kick_constant_J_kg: float | None = None, rittinger_constant_J_m_kg: float | None = None,
                   reference_test: dict | None = None, specific_energy_J_kg: float | None = None,
                   equipment: str | None = None, year: int = 2023, cepci: float | None = None) -> dict:
    """Energía específica y potencia de una reducción de tamaño (trituración/molienda) y, si se pide, costo de compra del
    equipo. El índice de trabajo y las constantes son del material del usuario: no hay valores por defecto.

    mass_flow_kg_s: sólido que se muele.
    law: 'bond' | 'kick' | 'rittinger' | 'specific_energy'.
      'bond': feed_size_m y product_size_m son los tamaños al 80 % pasante (F80, P80); work_index_kWh_per_ton (índice de
        Bond) y ton = 'short' (907.18 kg, la de las tablas de Bond) o 'metric'. e = 10·Wi·(1/√P80 − 1/√F80), µm.
      'kick' (e = K·ln(F/P)) y 'rittinger' (e = K·(1/P − 1/F)): kick_constant_J_kg o rittinger_constant_J_m_kg, O
        reference_test {feed_size_m, product_size_m, specific_energy_J_kg} (un ensayo propio con el que se calibra K).
      'specific_energy': specific_energy_J_kg medido o de proveedor (único camino para semillas y biomasa fibrosa, que no
        tienen índice de Bond publicado); los tamaños son opcionales.
    equipment (opcional, da costo): 'hammer_mill' (2-200 ton cortas/h), 'shredder', 'cane_crushing_mill' (estos dos, de la
      industria de la caña, sin intervalo publicado) o 'crusher' (solo potencia: no hay correlación abierta de costo).
    year / cepci: año del costo (2015-2023 con la tabla CEPCI de IDAES) o el valor del índice para otro año.

    Índices de Bond publicados (kWh/ton corta; 911 Metallurgist, columna «Source 3»; promedio, el material real varía mucho):
    caliza 12.7, carbón 13.0, hematita 12.8, cuarzo 13.6, yeso 6.7, arcilla 6.3, roca fosfórica 9.9, clínker 13.5,
    bauxita 8.8, granito 15.1, dolomita 11.3. Se pasan explícitamente; la herramienta nunca los aplica sola.

    Devuelve specific_energy_J_kg, specific_energy_kWh_per_tonne, power_W (en el eje, sin eficiencia de motor),
    reduction_ratio, cost {purchase_cost_usd, number_of_units, F_BM, cepci}, capital_items (lista lista para
    capital_investment), biosteam_reference_power_W (potencia por defecto de BioSTEAM para ese equipo, solo para comparar),
    published_reference, evidence, warnings (intervalo de la correlación), assumptions y not_included.
    Validación: McCabe ej. 28.1 (169.6 kW) y Geankoplis ej. 14.5-1 (24.1 hp) para Bond. Kick y Rittinger: solo identidades.
    Costos: correlación de BioSTEAM reproducida, sin contraste independiente."""
    return SH.size_reduction(mass_flow_kg_s, law, feed_size_m, product_size_m, work_index_kWh_per_ton, ton,
                             kick_constant_J_kg, rittinger_constant_J_m_kg, reference_test, specific_energy_J_kg, equipment,
                             year, cepci)


def solids_conveying(conveyor: str, mass_flow_kg_s: float, length_m: float | None = None, lift_m: float | None = None,
                     bulk_density_kg_m3: float | None = None, friction_factor: float | None = None,
                     tail_pulley_diameter_m: float | None = None, length_equivalency_factor: float | None = None,
                     year: int = 2023, cepci: float | None = None) -> dict:
    """Potencia en el eje y costo de compra de un transportador mecánico de sólidos a granel.

    conveyor: 'belt' (banda), 'screw' (tornillo) o 'bucket_elevator' (elevador de cangilones).
    mass_flow_kg_s: sólido transportado. lift_m: elevación vertical (obligatoria; 0 si es horizontal).
    length_m: longitud del transportador (banda y tornillo).
    bulk_density_kg_m3: densidad aparente a granel; solo hace falta para el costo (correlación por flujo volumétrico).
    Elevador de cangilones: friction_factor (≥ 1; P = ṁ·g·H·F; KWS publica 1.10-1.30) O tail_pulley_diameter_m con
      length_equivalency_factor (P = ṁ·g·(H + d·Leq); KWS publica Leq 5-15). Son datos del equipo: sin valor por defecto.
    year / cepci: año del costo o valor del índice.

    Banda: P[hp] = 0.00058·ṁ^0.82·L + 0.00182·ṁ·H; tornillo: P[hp] = 0.0146·ṁ^0.85·L (+ ṁ·g·H si sube), con ṁ en lb/s y
    longitudes en ft (correlaciones de Seider tal como están en BioSTEAM ConveyingBelt y ScrewFeeder). Costo: banda hasta
    2000 ft³/h por unidad (mínimo 120), tornillo 400-100 000 ft³/h; NO dependen de la longitud (orden de magnitud). El
    elevador de cangilones no tiene correlación abierta de costo.

    Devuelve power_W, friction_power_W, lift_power_W, lift_power_exact_W (ṁ·g·H), volumetric_flow_m3_s, cost, capital_items,
    evidence, warnings, assumptions, not_included.
    Validación: elevador de cangilones contra el ejemplo resuelto de KWS (6.10 y 6.01 hp). Banda y tornillo: el término de
    elevación es exacto; el de fricción es la correlación transcrita, sin ejemplo independiente. Costos: sin contraste."""
    return SH.solids_conveying(conveyor, mass_flow_kg_s, length_m, lift_m, bulk_density_kg_m3, friction_factor,
                               tail_pulley_diameter_m, length_equivalency_factor, year, cepci)


def solid_liquid_separation(equipment: str, feed_kg_s: float, feed_solids_mass_fraction: float,
                            solids_recovery_fraction: float, cake_moisture_mass_fraction: float,
                            filtration: dict | None = None, sigma: dict | None = None,
                            feed_density_kg_m3: float | None = None, year: int = 2023,
                            cepci: float | None = None) -> dict:
    """Separación sólido-líquido: reparto de corrientes con el desempeño del USUARIO, área de filtro por teoría de filtración
    a presión constante, teoría Sigma para centrífugas y costo de compra.

    equipment: 'screw_press' | 'decanter_centrifuge' | 'disc_stack_centrifuge' | 'rotary_vacuum_drum_filter' |
      'plate_and_frame_filter'.
    feed_kg_s, feed_solids_mass_fraction: suspensión alimentada y su fracción de sólidos insolubles.
    solids_recovery_fraction: fracción de los sólidos que sale en la torta; cake_moisture_mass_fraction: líquido en la torta
      húmeda. Ambos son datos de ensayo o de proveedor: el equipo no predice su propio desempeño.
    filtration (filtros; da el área): {specific_cake_resistance_m_kg (α a la presión de trabajo), medium_resistance_1_m (Rm,
      0 si se desprecia), filtrate_viscosity_Pa_s, pressure_drop_Pa, filtrate_density_kg_m3} y además
      tambor: cycle_time_s (una vuelta), submerged_fraction; filtro prensa: filtration_time_s, downtime_s (por ciclo).
    sigma (centrífugas): {particle_diameter_m (tamaño de corte al 50 %), particle_density_kg_m3, liquid_density_kg_m3,
      liquid_viscosity_Pa_s} → Σ requerida = Q/(2·v_g); opcional bowl {type: 'tubular' | 'solid_bowl' | 'disc_stack', rpm,
      r_inner_m, r_outer_m, y length_m (cilíndrico) o n_discs + half_angle_deg (discos)} → Σ de la máquina, capacidad y
      tamaño de corte al caudal alimentado.
    feed_density_kg_m3: solo para el costo de la centrífuga de discos y la potencia de referencia si no se da sigma.
    year / cepci: año del costo o valor del índice.

    Costos (BioSTEAM): prensa de tornillo 150-12 000 lb/h de alimentación; decantador 2-40 ton cortas/h de sólidos; discos
    hasta 100 m³/h; tambor 10-800 ft² por tambor (necesita filtration); filtro prensa escalado por flujo de torta húmeda
    (filtro de lignina de NREL, sin intervalo).

    Devuelve feed, cake y liquid (kg/s de sólidos y líquido), liquid_recovery_fraction, filtration {area_m2, …}, sigma
    {sigma_required_m2, bowl {sigma_m2, cut_diameter_at_feed_flow_m, …}}, cost, capital_items, biosteam_reference_power_W
    (valor por defecto de BioSTEAM, orientativo), evidence (cierre de balances y de la ecuación), warnings, assumptions y
    not_included.
    Validación: Geankoplis ej. 14.2-2 y 14.2-4 (filtración) y 14.4-2 (centrífuga tubular). Discos: fórmula deducida, solo
    identidades. Costo del tambor: cerca del doble de la tabla DOE/NETL-2002/1169; los demás, sin contraste."""
    return SH.solid_liquid_separation(equipment, feed_kg_s, feed_solids_mass_fraction, solids_recovery_fraction,
                                      cake_moisture_mass_fraction, filtration, sigma, feed_density_kg_m3, year, cepci)


def oil_extraction(mode: str, seed: dict | None = None, expeller: dict | None = None, solids: dict | None = None,
                   solvent: dict | None = None, underflow: dict | None = None, target: dict | None = None,
                   energy: dict | None = None, solvent_loss_kg_per_kg_solids_fed: float | None = None, year: int = 2023,
                   cepci: float | None = None) -> dict:
    """Extracción de aceite de semillas: prensado mecánico (expeller) o extracción con disolvente (hexano) en etapas ideales en
    contracorriente. El contenido de aceite, la recuperación, el aceite residual y la retención son datos del usuario.

    mode='expeller' — balance semilla → aceite crudo + torta (+ cáscara):
      seed {mass_flow_kg_s, oil_mass_fraction (base tal como llega), opcional decortication {hull_mass_fraction,
        hull_removal_fraction, oil_mass_fraction_in_hull}}.
      expeller {oil_recovery_fraction (del aceite que entra a la prensa) O residual_oil_in_cake_mass_fraction; opcionales
        solids_in_crude_oil_mass_fraction (finos), specific_energy_J_kg (por kg alimentado → power_W), estimate_cost (true =
        orden de magnitud con la prensa de tornillo genérica de Seider/BioSTEAM, 150-12 000 lb/h)}.
      Devuelve hull, press_feed, crude_oil, cake {oil_mass_fraction, oil_free_solids_kg_s}, recuperaciones, capacity.
    mode='solvent' — etapas de lixiviación:
      solids {inert_solids_kg_s (sin aceite ni disolvente), oil_kg_s, solvent_kg_s (0 si no se da)}.
      solvent {solvent_kg_s, oil_kg_s (aceite en el disolvente recirculado; 0 si no se da)}.
      underflow {solution_retained_kg_per_kg_inert (retención constante)} O {curve {oil_mass_fraction_in_solution: [...],
        kg_inert_per_kg_solution: [...]}} (retención variable; debe empezar en 0 y no se extrapola).
      target: uno de {residual_oil_kg_s}, {residual_oil_in_meal_mass_fraction} (harina sin disolvente) → número de etapas; o
        {stages} → residual que dan n etapas.
      energy {stream_T_K, stripping_pressure_Pa, opcional solvent_fluid (CoolProp, 'n-Hexane')} → calor bruto para
        evaporar el disolvente de la miscela y de la harina (estimación simple).
      solvent_loss_kg_per_kg_solids_fed → reposición de disolvente.
      Devuelve stages_ideal, stages_ideal_fractional, miscella, leached_solids, oil_recovery_fraction, stage_profile,
      solvent_makeup_kg_s, stripping {total_duty_W, …}, capacity. Sin costo: extractor y desolventizador van por cotización.

    Valores publicados (se pasan explícitamente; vienen también en published_reference con fuente y URL):
      ricino: 30-50 % de aceite; prensado ~45 % de recuperación (hasta 80 % en caliente); torta hasta ~10 % de aceite (Patel
        2016). Colza: 44.7 % de aceite; prensado en frío 66.7-87.3 % (media 77.6 %); 0.08-0.33 GJ/t (media 0.16); preprensado
        50 %; preprensado + hexano 98 % (Widmann 1999). Jatropha: 46.3 % de aceite, 42 % de cáscara; expeller de laboratorio
        30-69 % (Yaduvanshi 2024). Soya con hexano: < 1 % de aceite en la harina; hojuelas con 35-40 % de disolvente; pérdida
        de hexano 2.45e-3 kg/kg de soya (EPA AP-42 9.11.1). Expeller genérico: 5-8 % de aceite residual (Anderson).

    Todos devuelven evidence (cierre de balances total y de aceite, por etapa), warnings, assumptions y not_included.
    Validación: Geankoplis ej. 12.10-1 (4 etapas, retención variable) y solución analítica de retención constante. El
    balance del expeller es una identidad; el calor de agotamiento no está validado."""
    return SH.oil_extraction(mode, seed, expeller, solids, solvent, underflow, target, energy,
                             solvent_loss_kg_per_kg_solids_fed, year, cepci)


SOLIDS_HANDLING_TOOLS = [size_reduction, solids_conveying, solid_liquid_separation, oil_extraction]
