"""Agent tools for the own first-principles dynamic flowsheet (block `dynflow`): start-up, shutdown, disturbances and
control loops of tanks, CSTRs, lumped exchangers and gas vessels, independent of DWSIM. Engine and evidence live in
`packages.dynamics.flowsheet` (stiff ODEs with scipy solve_ivp/Radau, exact events, balances integrated with the state,
tolerance refinement)."""
from __future__ import annotations

from packages.dynamics import flash_drum as FD
from packages.dynamics import flowsheet as FS

_SCHEMA = """
    flowsheet (dict, todo SI; ningún dato físico tiene valor por defecto):
      components: ["A", "B", …] (opcional; si se da, cada equipo líquido necesita initial.concentration_mol_m3 y cada
        alimentación concentration_mol_m3 de todos, 0 si falta).
      liquid: {density_kg_m3, heat_capacity_J_kgK} comunes (o por equipo); ρ y cp constantes.
      units: lista de equipos con id y type:
        · "tank" / "mixing_tank" / "heated_tank" / "cstr" (mismo modelo, mezcla perfecta): area_m2 + initial.level_m (o
          initial.volume_m3; 0 = arranque desde vacío) y opcional max_level_m (rebose vigilado); o constant_volume: true
          con volume_m3 (necesita una salida 'overflow'). initial.temperature_K activa el balance de energía; sin él,
          temperature_K fija (isotérmico). heat_duty_W (calentador), jacket {UA_W_K, temperature_K}, pressure_Pa (gas
          sobre el líquido, para válvulas/bombas). reactions: [{stoichiometry {A: −1, B: 1}, orders {A: 1},
          pre_exponential (unidades de k) y activation_energy_J_mol, o rate_constant; heat_of_reaction_J_mol (J/mol,
          negativo = exotérmica)}].
        · "heat_exchanger": flow 'countercurrent'|'cocurrent', UA_W_K, cells (≥1), hot/cold {volume_m3, density_kg_m3,
          heat_capacity_J_kgK}, initial {hot: {temperature_K}, cold: {temperature_K}}; puertos '<id>.hot' y '<id>.cold';
          variables '<id>.hot_outlet_temperature_K', '<id>.cold_outlet_temperature_K'.
        · "gas_vessel": volume_m3, gas {model 'ideal', molar_mass_kg_mol, heat_capacity_ratio} o {model 'coolprop',
          fluid}, thermal 'isothermal'|'adiabatic'|'heat_transfer' (UA_W_K, ambient_temperature_K), initial
          {pressure_Pa, temperature_K}.
        · "flash_drum" (necesita components): mismos datos que dynamic_flash_drum salvo feed y liquid_outlet (temperature_K,
          volume_m3, area_m2, liquid_molar_density_mol_m3, liquid_density_kg_m3, equilibrium, vapour_outlet, initial);
          el líquido entra y sale por corrientes (c = x·ρ_molar); su presión la leen las válvulas y bombas conectadas.
          Variables '<id>.pressure_Pa', '.liquid_level_m', '.x.<c>', '.y.<c>', '.vapour_flow_mol_s'; parámetros
          '<id>.vapour.opening', '<id>.vapour.downstream_pressure_Pa'.
        · "distillation_column" binaria (modelo de Skogestad): stages (1 = rehervidor, último = condensador total),
          feed_stage, relative_volatility (o k_values {light, heavy}), hydraulics {tau_L_s, lambda, nominal_holdup_mol,
          nominal_liquid_flow_top_mol_s, nominal_liquid_flow_bottom_mol_s (y nominal_vapour_flow_* si lambda ≠ 0)},
          feed {molar_flow_mol_s, mole_fraction_light, liquid_fraction}, reflux_mol_s, boilup_mol_s, distillate_mol_s,
          bottoms_mol_s, initial {x, holdup_mol} (una cifra o una lista por etapa). Parámetros '<id>.reflux_mol_s',
          '.boilup_mol_s', '.distillate_mol_s', '.bottoms_mol_s', '.feed_mol_s', '.feed_mole_fraction'; variables
          '<id>.distillate_x', '.bottoms_x', '.condenser_holdup_mol', '.reboiler_holdup_mol' y, por etapa, '<id>.x.<i>',
          '<id>.holdup_mol.<i>'. Configuración LV: lazos de nivel con D y B; DV: con L y B; lazos de composición PI
          sobre L y V (controllers). Sin balance de energía ni dinámica de presión.
      streams: [{id, from, to, flow{type,…}}] (sin from = alimentación; sin to = descarga). Leyes de líquido:
        'fixed' {volumetric_flow_m3_s}; 'linear_resistance' {resistance_s_m2} (q = h/R); 'gravity_orifice'
        {discharge_coefficient, area_m2} (Torricelli); 'valve' {Kv_m3_h o Cv_gpm_psi, opening 0–1, characteristic
        'linear'|'equal_percentage' (rangeability)|'quick_opening', upstream_pressure_Pa si es alimentación,
        downstream_pressure_Pa si descarga}; 'pump' {curve_flow_m3_s [desde 0], curve_head_m, static_head_m,
        loss_coefficient_s2_m5, speed_fraction, running, y opcionalmente una válvula en serie con Kv/opening};
        'inertial_pipe' {length_m, diameter_m, darcy_friction_factor, initial_velocity_m_s}; 'overflow' (sale lo que
        entra de un equipo de volumen constante). interacting: true = la descarga depende también del nivel aguas abajo.
        Una alimentación de un equipo con energía lleva temperature_K; con componentes, concentration_mol_m3.
        Gas: 'fixed' {mass_flow_kg_s} o 'gas_orifice' {discharge_coefficient, area_m2, upstream_pressure_Pa y
        upstream_temperature_K si es fuente, downstream_pressure_Pa si descarga} (flujo crítico automático).
      heat_links: [{id, between: [equipo1, equipo2], UA_W_K}] (p. ej. reactor con chaqueta dinámica).
      blocks: [{id, input (variable de estado), time_constant_s y/o dead_time_s}] → '<id>.output' (sensor con atraso y
        tiempo muerto exacto).
      controllers: [{id, measurement, setpoint, manipulated (un parámetro), Kc (signo = acción), tau_I_s, tau_D_s,
        bias, output_min, output_max, tracking_time_s}] PID ideal ISA con anti-windup por retrocálculo.
    Parámetros (destinos de eventos): '<corriente>.volumetric_flow_m3_s', '.opening', '.running' (disparo de bomba = 0),
      '.speed_fraction', '.temperature_K', '.concentration_mol_m3.<c>', '.upstream_pressure_Pa', '.area_m2';
      '<equipo>.heat_duty_W', '.jacket_temperature_K', '.jacket_UA_W_K'; '<controlador>.setpoint'.
    Variables (salidas): '<equipo>.level_m', '.volume_m3', '.temperature_K', '.concentration_mol_m3.<c>',
      '.pressure_Pa', '.mass_kg'; '<corriente>.volumetric_flow_m3_s' o '.mass_flow_kg_s'; '<bloque>.output';
      '<controlador>.output', '.error'. Un error de nombre devuelve la lista válida."""


def dynamic_flowsheet_run(flowsheet: dict, t_end_s: float, events: list[dict] | None = None,
                          outputs: list[str] | None = None, limits: list[dict] | None = None,
                          start_from_steady_state: bool = False, n_points: int = 201, method: str = "Radau",
                          rtol: float = 1e-8) -> dict:
    """Simulación dinámica de un flowsheet propio (independiente de DWSIM) a partir de balances de materia y energía:
    arranque (llenado desde vacío), paro (corte de alimentación, disparo de bomba), transitorios por escalones o rampas
    de caudal, composición o temperatura de alimentación, apertura de válvula, carga de calentador, consigna; lazos PID;
    reactores CSTR exotérmicos; recipientes de gas (presurización y despresurización con flujo crítico).

    Esquema del flowsheet: al final de esta descripción.
    t_end_s: horizonte [s] (obligatorio). events: [{time_s, target (un parámetro), value, ramp_s (0 = escalón)}].
    outputs: variables a reportar (por defecto todas). limits: [{variable, max y/o min}] → limit_violations con primer
    instante (raíz exacta), peor valor y tiempo fuera. start_from_steady_state: busca el estacionario desde el estado
    inicial dado antes de t = 0. n_points: puntos de la serie reducida; method 'Radau'|'BDF'|'LSODA'; rtol.
    Devuelve time_s, series, metrics por salida (inicial, final, mín/máx con su instante, settling_time_s al ±2 % del
    cambio tras el último evento, overshoot_fraction), incidents (rebose, tanque seco), limit_violations, final_state,
    at_steady_state y evidence (verified; balances globales de volumen, componentes y energía durante el transitorio,
    corrida con tolerancia 100 veces menor, eventos acertados exactamente; reasons si no)."""
    return FS.simulate(flowsheet, t_end_s, events=events, outputs=outputs, limits=limits,
                       start_from_steady_state=start_from_steady_state, n_points=n_points, method=method, rtol=rtol)


def dynamic_flowsheet_steady_state(flowsheet: dict, outputs: list[str] | None = None) -> dict:
    """Estado estacionario de un flowsheet dinámico (mismo esquema que dynamic_flowsheet_run) y su estabilidad local en
    lazo abierto o cerrado: valores propios del jacobiano, constantes de tiempo y periodos de oscilación. Newton parte
    del estado inicial dado, así que con estados múltiples (CSTR exotérmico: bajo, intermedio inestable y alto) se
    obtiene el más cercano; repita con otros estados iniciales para mapearlos.

    flowsheet: mismo esquema que dynamic_flowsheet_run.
    Devuelve steady_state (todas las variables), outputs, converged, stable (None si no convergió), marginal (un
    inventario sin nivel propio), eigenvalues_per_s, time_constants_s, oscillation_periods_s y evidence (residuo,
    corrección de Newton, sensibilidad de los valores propios al paso de la diferencia finita)."""
    return FS.steady_state_analysis(flowsheet, outputs=outputs)


def dynamic_step_test(flowsheet: dict, input: str, step_size: float, output: str, t_end_s: float, step_time_s: float,
                      n_points: int = 200, start_from_steady_state: bool = True) -> dict:
    """Prueba de escalón en lazo abierto sobre el modelo no lineal, para identificar un FOPDT y sintonizar un PID.
    input: el parámetro que se mueve (p. ej. 'FJ.volumetric_flow_m3_s', 'V1.opening'); step_size: Δu (obligatorio, con
    signo); output: la variable registrada (p. ej. el '<bloque>.output' de un sensor); t_end_s y step_time_s: duración y
    momento del escalón (obligatorios). Si un controlador mueve 'input', pasa a manual con su salida congelada en el
    valor del estado estacionario en lazo cerrado (o el de t = 0); los demás siguen en automático.

    flowsheet: mismo esquema que dynamic_flowsheet_run.
    Devuelve time_s, output, baseline_output (en el instante del escalón), final_output, apparent_gain (Δy/Δu),
    pre_step_drift, controllers_in_manual, metrics, warnings y fopdt_identify_arguments: pásalos tal cual a
    fopdt_identify (y luego a pid_tuning). evidence como en dynamic_flowsheet_run."""
    return FS.step_test(flowsheet, input, step_size, output, t_end_s, step_time_s, n_points=n_points,
                        start_from_steady_state=start_from_steady_state)


def dynamic_flash_drum(flash_drum: dict, t_end_s: float, events: list[dict] | None = None,
                       start_from_steady_state: bool = False, n_points: int = 201) -> dict:
    """Tambor flash dinámico ISOTÉRMICO (un calentador/enfriador mantiene T): retención de líquido y de vapor en
    equilibrio en cada instante, presión fijada por el inventario de vapor y la salida de vapor, nivel por el líquido y
    la salida de líquido. Responde «¿cómo responden presión, nivel y composiciones a un cambio de alimentación o de
    apertura?» y comprueba que el estado final es el flash de Rachford-Rice de la alimentación.

    flash_drum (SI; nada físico tiene valor por defecto): components (≥ 2), temperature_K, volume_m3 (volumen total),
      area_m2 (sección, nivel = volumen de líquido/área), liquid_molar_density_mol_m3, liquid_density_kg_m3 (si la
      salida de líquido es válvula), equilibrium = {model: 'k_values', K: {c: K}} (K constantes a la T y P de diseño) o
      {model: 'relative_volatility', alpha: {c: α}, reference_vapour_pressure_Pa} (Raoult: K = α·P_ref/P),
      feed = {molar_flow_mol_s, mole_fractions {c: z}}, vapour_outlet = {type 'valve', coefficient_mol_s_Pa05 (V = C·
      f(x)·√(P − P_abajo)), opening 0–1, characteristic, rangeability si es isoporcentual, downstream_pressure_Pa} o
      {type 'fixed', molar_flow_mol_s}; liquid_outlet = {type 'valve', Kv_m3_h, opening, characteristic,
      downstream_pressure_Pa} o {type 'fixed', molar_flow_mol_s}; initial = {liquid_level_m, mole_fractions del
      contenido total (bifásico)}.
    events: [{time_s, target, value}] con target 'feed.molar_flow_mol_s', 'feed.mole_fractions' (value = {c: z}),
      'vapour.opening', 'liquid.opening', 'vapour.downstream_pressure_Pa', 'liquid.downstream_pressure_Pa' o
      '<lado>.molar_flow_mol_s' si es fijo. start_from_steady_state: Newton antes de t = 0.
    Devuelve time_s, series (pressure_Pa, liquid_level_m, holdup_vapour_fraction, vapour/liquid/feed_flow_mol_s, x.c,
    y.c), metrics, final_state, at_steady_state, rachford_rice_check (flash de la alimentación a la presión final frente
    a los productos) y evidence (balances por componente, tolerancia 100 veces menor, eventos exactos; reasons si no).
    Si el tambor se queda sin líquido o se llena, la integración se detiene y se informa."""
    return FD.simulate_flash_drum(flash_drum, t_end_s, events=events, start_from_steady_state=start_from_steady_state,
                                  n_points=n_points)


dynamic_flowsheet_run.__doc__ = dynamic_flowsheet_run.__doc__.rstrip() + "\n" + _SCHEMA

DYNAMIC_FLOWSHEET_TOOLS = [dynamic_flowsheet_run, dynamic_flowsheet_steady_state, dynamic_step_test, dynamic_flash_drum]
