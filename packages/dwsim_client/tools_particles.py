"""Agent tools for momentum transfer of non-Newtonian liquids and particles (bloque MP). No DWSIM flowsheet needed.
Engines: `packages.transport.non_newtonian` (exact laminar Rabinowitsch-Mooney solution, Dodge-Metzner, Hanks, Darby) and
`packages.transport.particles` (drag of the `fluids` library, Richardson-Zaki, Coe-Clevenger, Ergun, Wen-Yu) and
`packages.transport.cyclone` (Lapple cut size and grade efficiency, Shepherd-Lapple pressure drop)."""
from __future__ import annotations

from packages.transport import cyclone as CY
from packages.transport import non_newtonian as NN
from packages.transport import particles as PT


def non_newtonian_pipe_flow(volumetric_flow_m3_s: float | None = None, diameter_m: float | None = None,
                            length_m: float | None = None, density_kg_m3: float | None = None,
                            rheology: dict | None = None, profile_points: int = 21, geometry: str = "tube",
                            gap_m: float | None = None, width_m: float | None = None,
                            pressure_gradient_Pa_m: float | None = None, positions_m: list | None = None,
                            shear_rates_1_s: list | None = None, compare_rheology: dict | None = None,
                            free_stream_velocity_m_s: float | None = None) -> dict:
    """Flujo de un líquido no newtoniano (o newtoniano) en tubo circular o entre placas paralelas, curva de viscosidad
    aparente y capa límite laminar de Blasius sobre placa plana. La reología es del usuario: no hay valores por defecto.

    rheology: {'model': 'power_law', 'K_Pa_sn', 'n'} (τ = K·γ̇^n; K verdadera, no K' = K((3n+1)/4n)^n)
      | {'model': 'bingham', 'yield_stress_Pa', 'plastic_viscosity_Pa_s'} | {'model': 'herschel_bulkley',
      'yield_stress_Pa', 'K_Pa_sn', 'n'} | {'model': 'newtonian', 'viscosity_Pa_s'}
      | {'model': 'carreau', 'eta0_Pa_s', 'eta_inf_Pa_s' (0 si se omite), 'lambda_s', 'n'}:
        η = η∞ + (η0 − η∞)[1 + (λγ̇)²]^((n−1)/2) | {'model': 'carreau_yasuda', ... + 'a'}: [1 + (λγ̇)^a]^((n−1)/a)
      | {'model': 'cross', 'eta0_Pa_s', 'eta_inf_Pa_s', 'lambda_s', 'm'}: η = η∞ + (η0 − η∞)/(1 + (λγ̇)^m).
    geometry:
      'tube' (por defecto): diameter_m y volumetric_flow_m3_s (m³/s) -> ΔP, o pressure_gradient_Pa_m (−dp/dz) -> caudal
        (laminar). Con caudal y ley de potencia/Bingham/Herschel-Bulkley/newtoniano hacen falta length_m y density_kg_m3
        y se da el régimen: laminar EXACTO (Rabinowitsch-Mooney, Buckingham-Reiner), transición de Ryan-Johnson/Hanks,
        turbulento liso de Dodge-Metzner o Darby (Herschel-Bulkley turbulento es ESTIMACIÓN). Carreau/Yasuda/Cross y el
        problema inverso: solo solución laminar (integración numérica), con aviso si el Reynolds supera el crítico.
      'slit': placas paralelas separadas gap_m (h). Caudal en volumetric_flow_m3_s: m³/s si das width_m, o m²/s por
        unidad de ancho si no; o pressure_gradient_Pa_m -> caudal. Solo laminar, cualquier reología. length_m y
        density_kg_m3 opcionales (ΔP; comprobación de Reynolds).
      'flat_plate': capa límite laminar de Blasius (resuelta numéricamente). free_stream_velocity_m_s, density_kg_m3,
        rheology newtoniana, length_m = longitud L de la placa (arrastre), width_m = ancho W, positions_m = lista de x o
        de pares [x, y] (m; x desde el borde de ataque, y desde la pared). Avisa si Re_x > 5e5.
    positions_m (tube/slit): distancias al eje (r) o al plano medio (y) donde evaluar velocidad, τ y γ̇ (perfil laminar).
    profile_points: puntos del perfil tabulado.
    shear_rates_1_s y/o compare_rheology: viscosidad aparente η = τ/γ̇ de rheology (y de compare_rheology) a esas γ̇ y,
      con dos reologías, la(s) γ̇ donde se cruzan (búsqueda de 1e-8 a 1e10 1/s) más las asíntotas de Carreau/Cross
      (rama de ley de potencia y γ̇ donde corta la meseta η0). Sin caudal ni gradiente devuelve solo ese bloque
      ('viscosity'); no hace falta geometría.

    Devuelve (tube clásico) regime, pressure_drop_Pa, pressure_gradient_Pa_m, wall_shear_stress_Pa, f_fanning, f_darcy,
    generalized_reynolds_metzner_reed (bingham_reynolds, hedstrom_number), transition, laminar_solution {perfil u(r),
    wall_shear_rate_1_s, plug_radius_m, centerline_velocity_m_s, velocity_at_positions}, turbulent; (slit, Carreau,
    inverso) mean_velocity_m_s, max_velocity_m_s, max_to_mean, wall_shear_stress_Pa, wall_shear_rate_1_s,
    pressure_gradient_Pa_m, volumetric_flow_m3_s / flow_per_unit_width_m2_s, velocity_profile, velocity_at_positions,
    regime; (flat_plate) blasius_constants, points [{reynolds_x, delta_99_m, displacement_thickness_m,
    momentum_thickness_m, local_skin_friction_coefficient, wall_shear_stress_Pa, u_m_s, v_m_s}], plate {reynolds_L,
    average_skin_friction_coefficient, drag_one_side_N, drag_both_sides_N o por unidad de ancho}; viscosity {table,
    crossings, asymptotes}. Siempre evidence (forma cerrada vs cuadratura; Carreau: cuadratura N vs 2N e integral por
    partes; Blasius: f''(0) = 0.33206 e identidad integral), warnings, assumptions y not_included. Validación:
    soluciones analíticas (tubo y placas, newtoniano, ley de potencia, Bingham), Geankoplis 3.5-1 y 3.5-2, de Nevers
    13.1, 13.3 y 13.5, límites de Carreau, constantes de Blasius."""
    return NN.flow(volumetric_flow_m3_s, diameter_m, length_m, density_kg_m3, rheology, profile_points, geometry, gap_m,
                   width_m, pressure_gradient_Pa_m, positions_m, shear_rates_1_s, compare_rheology,
                   free_stream_velocity_m_s)


def particle_settling(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
                      fluid_viscosity_Pa_s: float, sphericity: float, projected_area_ratio: float | None = None,
                      drag_method: str = "Clift", solids_volume_fraction: float | None = None,
                      column_diameter_m: float | None = None, thickener: dict | None = None) -> dict:
    """Velocidad terminal de una partícula en un fluido newtoniano quieto, sedimentación obstaculizada y área de espesador.

    particle_diameter_m: diámetro (de la esfera de igual volumen si no es esférica). particle_density_kg_m3,
    fluid_density_kg_m3, fluid_viscosity_Pa_s (Pa·s): datos del usuario. sphericity: esfericidad de Wadell (1.0 = esfera).
    projected_area_ratio: OBLIGATORIO si sphericity < 1: área de la esfera de igual volumen / área proyectada en la
      dirección de caída (correlación de Song-Xu, 0.001 < Re < 100).
    drag_method: correlación de esfera de la librería fluids ('Clift' = curva estándar de Clift-Grace-Weber; también
      'Barati', 'Haider_Levenspiel', 'Morrison', ...). Siempre se reporta la dispersión entre cuatro correlaciones.
    solids_volume_fraction: si se da, velocidad obstaculizada por Richardson-Zaki u = u_t(1 − φ)^n (exponente escalonado
      de Richardson-Zaki 1954; column_diameter_m añade el término d/D).
    column_diameter_m: diámetro D del tubo o columna. Además del término d/D, para ESFERA da el efecto de pared sobre la
      partícula sola que cae por el eje: wall_effect {wall_factor = u_pared/u_infinito, correlation_used,
      wall_factor_by_correlation} y terminal_velocity_with_wall_m_s. Flujo reptante (Re < 0.1, d/D < 0.8): Haberman-Sayre
      (1958), con Francis (1933) de contraste; Re mayor: Di Felice (1996), ESTIMACIÓN (y Munroe si Re > 1000).
      terminal_velocity_m_s es siempre la del fluido infinito.
    thickener: {'feed_flow_m3_s', 'feed_solids_kg_m3', 'underflow_solids_kg_m3', opcional 'batch_tests':
      {'concentration_kg_m3': [...], 'settling_velocity_m_s': [...]}} -> área de Coe-Clevenger con las pruebas por lotes
      del usuario; sin batch_tests se ESTIMA con Richardson-Zaki (solo suspensiones no floculadas) y se avisa.

    Devuelve terminal_velocity_m_s, reynolds_terminal, drag_coefficient, regime (Stokes/intermedio/Newton),
    stokes_velocity_m_s, drag_split (solo esfera con Re < 0.1, solución exacta de Stokes F = 3πμdU: form_drag_N = 1/3
    por presión, friction_drag_N = 2/3 por esfuerzo viscoso; fuera de ese límite dice que hace falta el campo de flujo),
    terminal_velocity_by_correlation_m_s, richardson_zaki_exponent, hindered {...}, thickener {area_m2, diameter_m,
    limiting_concentration_kg_m3, solids_flux_kg_m2_s, underflow_m3_s, overflow_m3_s, table}, evidence, warnings, assumptions
    y not_included. Validación: Geankoplis 14.3-1 (Stokes), GPSA ej. 7-1, límite de Stokes, tabla de teoría exacta
    del factor de pared (arXiv:1202.1400, Tabla 9)."""
    return PT.settling(particle_diameter_m, particle_density_kg_m3, fluid_density_kg_m3, fluid_viscosity_Pa_s, sphericity,
                       projected_area_ratio, drag_method, solids_volume_fraction, column_diameter_m, thickener)


def fluidization(particle_diameter_m: float, particle_density_kg_m3: float, fluid_density_kg_m3: float,
                 fluid_viscosity_Pa_s: float, sphericity: float, voidage_mf: float | None = None,
                 bed_mass_kg: float | None = None, column_diameter_m: float | None = None,
                 superficial_velocity_m_s: float | None = None, projected_area_ratio: float | None = None,
                 drag_method: str = "Clift") -> dict:
    """Lecho fluidizado: velocidad mínima de fluidización y terminal, ventana de operación, ΔP y expansión del lecho.

    particle_diameter_m (esfera de igual volumen), particle_density_kg_m3, fluid_density_kg_m3, fluid_viscosity_Pa_s,
    sphericity (1.0 = esfera): datos del usuario. voidage_mf: porosidad en mínima fluidización (medida); con ella se calcula
    u_mf por Ergun (y la ΔP del lecho fijo); sin ella solo Wen-Yu. bed_mass_kg + column_diameter_m: ΔP del lecho fluidizado
    = peso boyante / área y altura en mínima fluidización. superficial_velocity_m_s: punto de operación -> estado (lecho
    fijo / fluidizado / transporte), ΔP y, con líquido, porosidad y altura del lecho expandido por Richardson-Zaki.
    projected_area_ratio: necesario para u_t de partículas no esféricas (Song-Xu). drag_method: correlación de esfera.

    Devuelve u_mf_ergun_m_s, u_mf_wen_yu_m_s, u_mf_used_m_s, archimedes_number, terminal {...},
    operating_window_m_s {u_mf, u_t, ratio_ut_umf}, fluidized_bed_pressure_drop_Pa, bed_height_mf_m, operating_point {...},
    evidence (Ergun en forma cerrada vs raíz de fluids.packed_bed.Ergun; ΔP de Ergun en u_mf = peso del lecho), warnings,
    assumptions y not_included. Con gas no se calcula la expansión (lecho burbujeante). Validación: Geankoplis 3.1-6 y el
    caso de lecho fluidizado de arXiv:2211.01185."""
    return PT.fluidization(particle_diameter_m, particle_density_kg_m3, fluid_density_kg_m3, fluid_viscosity_Pa_s,
                           sphericity, voidage_mf, bed_mass_kg, column_diameter_m, superficial_velocity_m_s,
                           projected_area_ratio, drag_method)


def cyclone_performance(gas_flow_m3_s: float, gas_density_kg_m3: float, gas_viscosity_Pa_s: float,
                        particle_density_kg_m3: float, body_diameter_m: float, geometry: str | dict = "lapple",
                        particle_size_distribution: dict | None = None, diameters_m: list | None = None,
                        pressure_drop_K: float = 16.0) -> dict:
    """Ciclón de flujo inverso por el método de Lapple: velocidad de entrada, vueltas efectivas, diámetro de corte d50,
    curva de eficiencia por tamaño, eficiencia global sobre TU distribución de tamaños y caída de presión.

    gas_flow_m3_s, gas_density_kg_m3, gas_viscosity_Pa_s: gas a condiciones de operación. particle_density_kg_m3: densidad
    real de la partícula. body_diameter_m: diámetro del cuerpo D (dato del usuario).
    geometry: preset de RELACIONES respecto a D: 'lapple', 'stairmand_he', 'stairmand_ht', 'swift_he', 'swift_gp',
      'swift_ht'; o dict {'inlet_height', 'inlet_width', 'outlet_diameter', 'outlet_length', 'body_length', 'total_height',
      'dust_outlet'} (cada uno / D; cono = total_height − body_length; opcional 'inlet': 'tangencial' | 'voluta').
    particle_size_distribution: {'diameter_m': [...], 'mass_fraction': [...]} (diámetro representativo de cada clase,
      fracciones que suman 1); sin ella no hay eficiencia global (no hay distribución por defecto).
    diameters_m: diámetros donde tabular la eficiencia por tamaño. pressure_drop_K: K de Shepherd-Lapple (16 = entrada
      tangencial sin álabes).

    Fórmulas: Ne = (h + (H − h)/2)/a; d50 = √(9μb/(2πNeVi(ρp − ρg))); η = 1/(1 + (d50/d)²); ΔP = K·a·b/De²·½ρgVi².
    Devuelve geometry {dimensions}, inlet_velocity_m_s, effective_turns, cut_diameter_m, grade_efficiency,
    overall {overall_efficiency, table}, pressure_drop_Pa, pressure_drop_velocity_heads, evidence, warnings, assumptions y
    not_included. Sin Barth-Muschelknautz ni efecto de la carga de sólidos. Validación: ejemplo de ciclón Lapple de Penn
    State ME433 (d50 y η a 10 µm) e identidades analíticas."""
    return CY.cyclone_performance(gas_flow_m3_s, gas_density_kg_m3, gas_viscosity_Pa_s, particle_density_kg_m3,
                                  body_diameter_m, geometry, particle_size_distribution, diameters_m, pressure_drop_K)


PARTICLES_TOOLS = [non_newtonian_pipe_flow, particle_settling, fluidization, cyclone_performance]
