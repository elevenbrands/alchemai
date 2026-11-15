"""Agent tools for catalytic and multiphase reactors (block `catalysis`): effectiveness factor of a porous pellet
(Thiele, Weisz-Hicks), 1-D heterogeneous fixed bed with local η, and gas-liquid absorption with reaction (Hatta,
enhancement factor). Engines and evidence live in `packages.transport.pellet` and
`packages.transport.multiphase_reactor` (scipy solve_bvp / solve_ivp, sealed in every run against analytical limits,
flux balances and step/mesh independence)."""
from __future__ import annotations

from packages.transport import multiphase_reactor as MR
from packages.transport import pellet as PL


def catalyst_effectiveness(geometry: str | None = None, characteristic_length_m: float | None = None,
                           kinetics: dict | None = None, concentration_mol_m3: float | None = None,
                           temperature_K: float | None = None, effective_diffusivity_m2_s: float | None = None,
                           pore: dict | None = None, heat_of_reaction_J_mol: float | None = None,
                           effective_conductivity_W_mK: float | None = None,
                           mass_transfer_coefficient_m_s: float | None = None,
                           heat_transfer_coefficient_W_m2K: float | None = None,
                           pellet_density_kg_m3: float | None = None, branch: str = "low") -> dict:
    """Factor de efectividad de UN pellet de catalizador poroso a condiciones dadas del fluido: «¿cuánto de la velocidad
    intrínseca entrega realmente el pellet?», «¿me limita la difusión en los poros o la película externa?», «¿hay varios
    estados estacionarios?». Resuelve el problema de contorno difusión-reacción (C y T acoplados) por colocación, con
    comprobación por disparo y contra las soluciones analíticas.

    Argumentos (SI; ningún dato físico tiene valor por defecto):
      geometry: 'slab' (placa; tamaño = semiespesor, o espesor si una cara está sellada), 'cylinder' (cilindro largo,
        radio) o 'sphere' (esfera, radio). characteristic_length_m: ese semiespesor o radio [m].
      kinetics: velocidad INTRÍNSECA (sin transporte) del reactivo limitante A:
        {'type': 'power_law', 'order': n, ...} → r = k·C^n (n ≥ 0; n < 0.5 se resuelve con frente de núcleo muerto), o
        {'type': 'langmuir_hinshelwood', 'K_m3_mol', 'inhibition_exponent': m, ...} → r = k·C/(1 + K·C)^m;
        siempre 'basis': 'pellet_volume' (mol/(m³ de pellet·s)) o 'catalyst_mass' (mol/(kg·s), exige
        pellet_density_kg_m3), y 'k' a 'reference_temperature_K' (con 'activation_energy_J_mol') o 'pre_exponential' +
        'activation_energy_J_mol'. Opcional 'adsorption_enthalpy_J_mol' (van 't Hoff en K).
      concentration_mol_m3, temperature_K: concentración de A y temperatura en el SENO del fluido.
      effective_diffusivity_m2_s: De en el pellet; o pore = {porosity, tortuosity, mean_pore_radius_m,
        molecular_diffusivity_m2_s, molar_mass_kg_mol} para ESTIMARLA (Bosanquet Knudsen + molecular; se marca estimada).
      heat_of_reaction_J_mol (negativo = exotérmica) con effective_conductivity_W_mK (gradiente térmico interno) y/o
        heat_transfer_coefficient_W_m2K (película térmica externa): activan el caso no isotérmico (Weisz-Hicks).
      mass_transfer_coefficient_m_s: kc de la película externa (si falta, la superficie está a la concentración del seno).
      branch: 'low' | 'middle' | 'high': estado que se reporta en detalle cuando hay varios.

    Devuelve: eta_internal, eta_external, eta_overall (definiciones en 'definitions'), thiele_modulus,
    generalized_modulus (Bischoff) y su asíntota 1/Φ, observed_rate_mol_m3_s, concentraciones y temperaturas de
    superficie, centro y máxima, saltos térmicos, steady_states (todos los encontrados), dimensionless_groups (Biot,
    β de Prater, γ de Arrhenius), criteria (Weisz-Prater, Mears masa/calor, Anderson evaluados con la velocidad resuelta),
    profile, warnings, assumptions, not_included, method y evidence (verified + independencia de malla, balance
    ∫r dV = flujo en superficie, invariante de Prater, desviación contra la solución analítica o el disparo)."""
    return MR._clean(PL.catalyst_effectiveness(
        geometry=geometry, characteristic_length_m=characteristic_length_m, kinetics=kinetics,
        concentration_mol_m3=concentration_mol_m3, temperature_K=temperature_K,
        effective_diffusivity_m2_s=effective_diffusivity_m2_s, pore=pore, heat_of_reaction_J_mol=heat_of_reaction_J_mol,
        effective_conductivity_W_mK=effective_conductivity_W_mK, mass_transfer_coefficient_m_s=mass_transfer_coefficient_m_s,
        heat_transfer_coefficient_W_m2K=heat_transfer_coefficient_W_m2K, pellet_density_kg_m3=pellet_density_kg_m3,
        branch=branch))


def packed_bed_catalytic_reactor(geometry: str | None = None, characteristic_length_m: float | None = None,
                                 kinetics: dict | None = None, effective_diffusivity_m2_s: float | None = None,
                                 pore: dict | None = None, pellet_density_kg_m3: float | None = None,
                                 heat_of_reaction_J_mol: float | None = None,
                                 effective_conductivity_W_mK: float | None = None,
                                 mass_transfer_coefficient_m_s: float | None = None,
                                 heat_transfer_coefficient_W_m2K: float | None = None, bed_voidage: float | None = None,
                                 tube_diameter_m: float | None = None, number_of_tubes: int = 1,
                                 bed_length_m: float | None = None, target_conversion: float | None = None,
                                 feed_molar_flow_A_mol_s: float | None = None,
                                 feed_total_molar_flow_mol_s: float | None = None,
                                 moles_change_per_mol_A: float | None = None, inlet_temperature_K: float | None = None,
                                 inlet_pressure_Pa: float | None = None, energy_mode: str | None = None,
                                 gas_heat_capacity_J_molK: float | None = None,
                                 wall_heat_transfer_coefficient_W_m2K: float | None = None,
                                 coolant_temperature_K: float | None = None, gas_viscosity_Pa_s: float | None = None,
                                 feed_molar_mass_kg_mol: float | None = None,
                                 ergun_particle_diameter_m: float | None = None, branch: str = "low",
                                 n_profile: int = 11) -> dict:
    """Reactor catalítico de lecho fijo (fase gas) 1-D heterogéneo: en cada posición la velocidad es η(local)·r con η del
    pellet resuelto a la C y T locales. Responde «¿qué conversión da este lecho?», «¿qué longitud / masa de catalizador
    necesito para X?», «¿dónde está el punto caliente?», «¿en qué zona me limita la difusión?».

    Argumentos (SI; ningún dato físico ni de operación tiene valor por defecto):
      Pellet: geometry, characteristic_length_m, kinetics, effective_diffusivity_m2_s o pore, heat_of_reaction_J_mol,
        effective_conductivity_W_mK, mass_transfer_coefficient_m_s, heat_transfer_coefficient_W_m2K, pellet_density_kg_m3
        — mismo significado que en catalyst_effectiveness (kc y h constantes a lo largo del lecho).
      bed_voidage: fracción de huecos ε (la fracción de catalizador es 1 − ε = densidad de lecho / densidad de pellet).
      tube_diameter_m y number_of_tubes (1 si es un solo lecho; el resultado es para todos los tubos).
      bed_length_m (calcula la conversión) o target_conversion (calcula la longitud): uno de los dos.
      feed_molar_flow_A_mol_s, feed_total_molar_flow_mol_s (todos los tubos), moles_change_per_mol_A (Σν/|ν_A|; 0 para
        A → B), inlet_temperature_K, inlet_pressure_Pa.
      energy_mode: 'isothermal' | 'adiabatic' (gas_heat_capacity_J_molK y heat_of_reaction_J_mol) | 'wall_cooled'
        (además wall_heat_transfer_coefficient_W_m2K y coolant_temperature_K).
      gas_viscosity_Pa_s + feed_molar_mass_kg_mol: activan Ergun (si faltan, el lecho es isobárico y se avisa);
        ergun_particle_diameter_m opcional (por defecto 6·Vp/Sp del pellet).
      branch: rama del pellet si hay multiplicidad; n_profile: puntos del perfil con resolución completa del pellet.

    Devuelve: conversion, bed_length_m, bed_volume_m3, catalyst_volume_m3, catalyst_mass_kg (con densidad de pellet),
    outlet_temperature_K, outlet_pressure_Pa, pressure_drop_Pa, hot_spot_temperature_K y posición,
    heat_removed_by_wall_W, profile (z, conversión, T, P, C, η global e interno, zonas limitadas por difusión),
    diffusion_limited_positions_m, warnings, assumptions, not_included, method y evidence (independencia del paso,
    cierre de energía, velocidad local contra resolución completa del pellet, pellets verificados, solución cerrada
    cuando existe)."""
    return MR.packed_bed_reactor(
        geometry=geometry, characteristic_length_m=characteristic_length_m, kinetics=kinetics,
        effective_diffusivity_m2_s=effective_diffusivity_m2_s, pore=pore, pellet_density_kg_m3=pellet_density_kg_m3,
        heat_of_reaction_J_mol=heat_of_reaction_J_mol, effective_conductivity_W_mK=effective_conductivity_W_mK,
        mass_transfer_coefficient_m_s=mass_transfer_coefficient_m_s,
        heat_transfer_coefficient_W_m2K=heat_transfer_coefficient_W_m2K, bed_voidage=bed_voidage,
        tube_diameter_m=tube_diameter_m, number_of_tubes=number_of_tubes, bed_length_m=bed_length_m,
        target_conversion=target_conversion, feed_molar_flow_A_mol_s=feed_molar_flow_A_mol_s,
        feed_total_molar_flow_mol_s=feed_total_molar_flow_mol_s, moles_change_per_mol_A=moles_change_per_mol_A,
        inlet_temperature_K=inlet_temperature_K, inlet_pressure_Pa=inlet_pressure_Pa, energy_mode=energy_mode,
        gas_heat_capacity_J_molK=gas_heat_capacity_J_molK,
        wall_heat_transfer_coefficient_W_m2K=wall_heat_transfer_coefficient_W_m2K,
        coolant_temperature_K=coolant_temperature_K, gas_viscosity_Pa_s=gas_viscosity_Pa_s,
        feed_molar_mass_kg_mol=feed_molar_mass_kg_mol, ergun_particle_diameter_m=ergun_particle_diameter_m,
        branch=branch, n_profile=n_profile)


def gas_liquid_reaction(kinetics_type: str | None = None, liquid_mass_transfer_coefficient_m_s: float | None = None,
                        diffusivity_A_m2_s: float | None = None, interface_concentration_A_mol_m3: float | None = None,
                        gas_partial_pressure_Pa: float | None = None, henry_constant_Pa_m3_mol: float | None = None,
                        gas_mass_transfer_coefficient_mol_m2_s_Pa: float | None = None, rate_constant: float | None = None,
                        bulk_concentration_B_mol_m3: float | None = None, diffusivity_B_m2_s: float | None = None,
                        stoichiometric_ratio_B: float | None = None, bulk_concentration_A_mol_m3: float | None = None,
                        contactor: dict | None = None) -> dict:
    """Absorción de un gas A en un líquido donde reacciona (teoría de la película): «¿cuánto acelera la reacción la
    absorción (factor de mejora E)?», «¿en qué régimen estoy (lenta, rápida, instantánea)?», «¿cuál es el flujo
    absorbido?», y opcionalmente «¿qué absorbe mi columna de burbujas / tanque agitado?».

    Argumentos (SI; ningún dato físico tiene valor por defecto):
      kinetics_type: 'first_order' (r = k1·C_A; rate_constant = k1 [1/s]; también pseudo-primer orden k1 = k2·C_B),
        'second_order' (A + ν B, r = k2·C_A·C_B; rate_constant = k2 [m³/(mol·s)], bulk_concentration_B_mol_m3,
        diffusivity_B_m2_s, stoichiometric_ratio_B = ν) o 'instantaneous' (A + ν B; sin rate_constant).
      liquid_mass_transfer_coefficient_m_s (kL sin reacción), diffusivity_A_m2_s (D_A en el líquido).
      Interfase: interface_concentration_A_mol_m3 (C_Ai) o gas_partial_pressure_Pa + henry_constant_Pa_m3_mol (p = H·C)
        con gas_mass_transfer_coefficient_mol_m2_s_Pa opcional (resistencia del lado gas).
      bulk_concentration_A_mol_m3: A en el seno del líquido (0 si se consume toda), obligatorio salvo con contactor.
      contactor (solo primer orden): {'reactor_volume_m3', 'liquid_holdup', 'liquid_flow_m3_s',
        'inlet_concentration_A_mol_m3', y 'kLa_1_s' o 'interfacial_area_m2_m3'}: líquido perfectamente mezclado y gas
        uniforme; calcula C_A en el seno y lo que reacciona en la película, en el seno y lo que sale con el líquido.

    Devuelve: hatta_number, enhancement_factor (solución numérica de la película; rigurosa en teoría de película),
    enhancement_first_order_closed_form (Ha/tanh Ha), instantaneous_enhancement_factor (E_i = 1 + q),
    enhancement_van_krevelen_hoftijzer y su desviación, q_ratio, concentraciones en la interfase (A y B),
    absorption_flux_mol_m2_s, flux_into_bulk_mol_m2_s, fraction_reacted_in_film, regime con regime_indicators,
    contactor (si se dio), warnings, assumptions, not_included, method y evidence (independencia de malla, reacción
    integrada = diferencia de flujos, cierre estequiométrico de B, contraste con la solución cerrada)."""
    return MR.gas_liquid_reaction(
        kinetics_type=kinetics_type, liquid_mass_transfer_coefficient_m_s=liquid_mass_transfer_coefficient_m_s,
        diffusivity_A_m2_s=diffusivity_A_m2_s, interface_concentration_A_mol_m3=interface_concentration_A_mol_m3,
        gas_partial_pressure_Pa=gas_partial_pressure_Pa, henry_constant_Pa_m3_mol=henry_constant_Pa_m3_mol,
        gas_mass_transfer_coefficient_mol_m2_s_Pa=gas_mass_transfer_coefficient_mol_m2_s_Pa, rate_constant=rate_constant,
        bulk_concentration_B_mol_m3=bulk_concentration_B_mol_m3, diffusivity_B_m2_s=diffusivity_B_m2_s,
        stoichiometric_ratio_B=stoichiometric_ratio_B, bulk_concentration_A_mol_m3=bulk_concentration_A_mol_m3,
        contactor=contactor)


def three_phase_catalytic_reactor(reactor_type: str | None = None, temperature_K: float | None = None,
                                  henry_constant_Pa_m3_mol: float | None = None, kLa_1_s: float | None = None,
                                  kGa_1_s: float | None = None,
                                  liquid_solid_mass_transfer_coefficient_m_s: float | None = None,
                                  geometry: str | None = None, characteristic_length_m: float | None = None,
                                  kinetics: dict | None = None, effective_diffusivity_m2_s: float | None = None,
                                  pore: dict | None = None, apparent_rate_constant_m3_kg_s: float | None = None,
                                  pellet_density_kg_m3: float | None = None, stoichiometric_ratio_B: float | None = None,
                                  liquid_reactant_concentration_mol_m3: float | None = None,
                                  gas_partial_pressure_Pa: float | None = None, catalyst_loading_kg_m3: float | None = None,
                                  liquid_volume_m3: float | None = None, operation: str | None = None,
                                  liquid_flow_m3_s: float | None = None, batch_time_s: float | None = None,
                                  target_conversion: float | None = None, limiting_reactant: str | None = None,
                                  wetting_efficiency: float | None = None, bed_voidage: float | None = None,
                                  column_diameter_m: float | None = None, bed_length_m: float | None = None,
                                  feed_molar_flow_A_mol_s: float | None = None,
                                  feed_total_gas_molar_flow_mol_s: float | None = None,
                                  gas_moles_change_per_mol_A: float | None = None, inlet_pressure_Pa: float | None = None,
                                  pressure_gradient_Pa_m: float | None = None, n_profile: int = 11) -> dict:
    """Reactor catalítico trifásico (gas A + líquido B + catalizador sólido), isotérmico, con resistencias en serie: gas →
    interfase → seno del líquido → superficie del catalizador → interior del pellet (η del motor de pellet). Responde
    «¿qué paso controla y con qué fracción de la resistencia?», «¿qué velocidad global obtengo?», «¿cuánto catalizador
    necesito para X?» y «¿qué conversión da mi lecho de goteo?».

    Argumentos (SI; ningún dato físico ni coeficiente tiene valor por defecto; NO se estiman con correlaciones):
      reactor_type: 'slurry' (catalizador suspendido) o 'trickle_bed' (lecho de goteo).
      temperature_K; henry_constant_Pa_m3_mol (p = H·C del gas A en el líquido).
      kLa_1_s: gas-líquido del lado líquido (por m³ de LÍQUIDO en slurry, por m³ de LECHO en goteo); kGa_1_s opcional
        (lado gas, fuerza impulsora en concentración del gas, misma base); liquid_solid_mass_transfer_coefficient_m_s
        (kc del reactivo limitante; el área por kg sale de la geometría y ρp).
      Pellet: geometry, characteristic_length_m, pellet_density_kg_m3 y kinetics del REACTIVO LIMITANTE por
        catalizador ({'type','basis','k','order',...} como en catalyst_effectiveness, más 'other_reactant_order': 0 si k ya
        incluye al otro reactivo, 1 si r = k·C_lim^n·C_otro) con effective_diffusivity_m2_s o pore; o en su lugar
        apparent_rate_constant_m3_kg_s = η·k medido (primer orden). En lecho de goteo la cinética debe ser de primer
        orden en el limitante.
      stoichiometric_ratio_B (mol de B por mol de A), liquid_reactant_concentration_mol_m3 (B alimentado o inicial).
      Slurry: gas_partial_pressure_Pa, liquid_volume_m3, operation 'batch' (catalyst_loading_kg_m3 y batch_time_s o
        target_conversion) o 'continuous' (liquid_flow_m3_s y catalyst_loading_kg_m3 o target_conversion de B → carga).
      Lecho de goteo: limiting_reactant 'gas' o 'liquid' (B no volátil; líquido saturado de gas), wetting_efficiency
        (fracción mojada, dato del usuario), bed_voidage, column_diameter_m, bed_length_m o target_conversion (de A si
        limita el gas, de B si limita el líquido), feed_molar_flow_A_mol_s, feed_total_gas_molar_flow_mol_s,
        gas_moles_change_per_mol_A (−1 si A pasa al líquido), inlet_pressure_Pa, pressure_gradient_Pa_m (opcional),
        liquid_flow_m3_s. n_profile: puntos del perfil.

    Devuelve: resistances (s en slurry; kg·s/m³ en goteo), shares, controlling_step, overall_rate (por m³ de líquido o
    por kg), conversiones, carga o masa de catalizador, tiempo de lote, altura y volumen del lecho, concentraciones en
    cada paso, profile (goteo), warnings, assumptions, not_included, method y evidence (cierre de la serie, solución
    cerrada de Fogler cuando existe, independencia del paso, problema de contorno del pellet)."""
    return MR.three_phase_reactor(
        reactor_type=reactor_type, temperature_K=temperature_K, henry_constant_Pa_m3_mol=henry_constant_Pa_m3_mol,
        kLa_1_s=kLa_1_s, kGa_1_s=kGa_1_s,
        liquid_solid_mass_transfer_coefficient_m_s=liquid_solid_mass_transfer_coefficient_m_s, geometry=geometry,
        characteristic_length_m=characteristic_length_m, kinetics=kinetics,
        effective_diffusivity_m2_s=effective_diffusivity_m2_s, pore=pore,
        apparent_rate_constant_m3_kg_s=apparent_rate_constant_m3_kg_s, pellet_density_kg_m3=pellet_density_kg_m3,
        stoichiometric_ratio_B=stoichiometric_ratio_B,
        liquid_reactant_concentration_mol_m3=liquid_reactant_concentration_mol_m3,
        gas_partial_pressure_Pa=gas_partial_pressure_Pa, catalyst_loading_kg_m3=catalyst_loading_kg_m3,
        liquid_volume_m3=liquid_volume_m3, operation=operation, liquid_flow_m3_s=liquid_flow_m3_s,
        batch_time_s=batch_time_s, target_conversion=target_conversion, limiting_reactant=limiting_reactant,
        wetting_efficiency=wetting_efficiency, bed_voidage=bed_voidage, column_diameter_m=column_diameter_m,
        bed_length_m=bed_length_m, feed_molar_flow_A_mol_s=feed_molar_flow_A_mol_s,
        feed_total_gas_molar_flow_mol_s=feed_total_gas_molar_flow_mol_s,
        gas_moles_change_per_mol_A=gas_moles_change_per_mol_A, inlet_pressure_Pa=inlet_pressure_Pa,
        pressure_gradient_Pa_m=pressure_gradient_Pa_m, n_profile=n_profile)


CATALYSIS_TOOLS = [catalyst_effectiveness, packed_bed_catalytic_reactor, gas_liquid_reaction,
                   three_phase_catalytic_reactor]
