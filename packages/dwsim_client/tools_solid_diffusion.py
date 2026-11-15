"""Agent tools for Fickian diffusion without reaction in solids, pellets and dense membranes (block `soliddiff`):
transient uptake/release, permeation with time lag, and D fitted from uptake data. Engine and evidence live in
`packages.transport.solid_diffusion` (finite volumes + scipy LSODA, sealed in every run against Crank's series)."""
from __future__ import annotations

from packages.transport import solid_diffusion as SD


def transient_diffusion_solid(geometry: str | None = None, initial_concentration_mol_m3: float | None = None,
                              diffusivity_m2_s: float | None = None, diffusivity_law: dict | None = None,
                              surface_condition: str | None = None, surface_concentration_mol_m3: float | None = None,
                              fluid_concentration_mol_m3: float | None = None,
                              mass_transfer_coefficient_m_s: float | None = None, partition_coefficient: float | None = None,
                              thickness_m: float | None = None, exposed_faces: int | None = None, radius_m: float | None = None,
                              length_m: float | None = None, width_m: float | None = None, depth_m: float | None = None,
                              bath_volume_m3: float | None = None, solid_volume_m3: float | None = None,
                              times_s: list[float] | None = None, target_fractional_uptake: float | None = None,
                              n_cells: int = 200, profile_points: int = 21) -> dict:
    """Difusión transitoria de Fick SIN reacción en un sólido con concentración inicial uniforme: captación (sorción),
    liberación, desorción tipo secado, lixiviación. Responde «¿cuánto ha entrado/salido a tal tiempo?», «¿cuánto tarda en
    llegar al X % del equilibrio?» y «¿qué perfil hay dentro?».

    Argumentos (SI; ninguno físico tiene valor por defecto):
      geometry: 'slab' (placa: thickness_m y exposed_faces = 1 o 2; con 1 la otra cara está sellada), 'cylinder' (cilindro
        largo, radius_m), 'sphere' (radius_m), 'semi_infinite' (sólido muy grueso, tiempos cortos), y cuerpos finitos por
        producto de soluciones con todas las caras expuestas: 'finite_cylinder' (radius_m, length_m), 'rectangular_bar'
        (thickness_m, width_m) y 'block' (thickness_m, width_m, depth_m).
      initial_concentration_mol_m3: concentración inicial en el sólido (por volumen de sólido; vale cualquier unidad
        consistente, p. ej. kg/m³).
      diffusivity_m2_s: D constante; o diffusivity_law para D(c): {'type': 'exponential', 'D0_m2_s', 'beta_m3_mol'}
        (D = D0·exp(β·c)), {'type': 'linear', 'D0_m2_s', 'alpha_m3_mol'} (D = D0·(1 + α·c)) o {'type': 'table',
        'concentration_mol_m3': [...], 'diffusivity_m2_s': [...]} (no se extrapola fuera de la tabla).
      surface_condition: 'fixed_concentration' (surface_concentration_mol_m3 = concentración EN EL SÓLIDO en la
        superficie), 'convective' (película externa: fluid_concentration_mol_m3, mass_transfer_coefficient_m_s y
        partition_coefficient K = c_sólido/c_fluido en equilibrio; Geankoplis usa el inverso) o 'limited_bath' (baño
        agitado de volumen finito de Crank: fluid_concentration_mol_m3 = concentración INICIAL del baño,
        partition_coefficient, bath_volume_m3, solid_volume_m3; mass_transfer_coefficient_m_s opcional).
      times_s: instantes [s] en que se quiere el estado; target_fractional_uptake: fracción (0–1) de aproximación al
        equilibrio cuyo tiempo se busca. Al menos uno de los dos. solid_volume_m3 (opcional) da la captación total en mol.
      n_cells (20–2000), profile_points: ajustes numéricos. Máximo 200 instantes en times_s (30 en semi_infinite).

    Devuelve por cada tiempo: fractional_uptake (M_t/M_∞), unaccomplished_fraction, average/center/surface
    concentration, surface_flux_mol_m2_s (positivo hacia el sólido), fourier_number, regime (tiempos cortos
    semi-infinito / intermedio / tiempos largos primer autovalor), one_term_fractional_uptake y profile; además
    biot_number_mass, equilibrium_concentration_mol_m3, time_to_target, y en semi_infinite uptake_mol_m2 y
    penetration_depth_1pct_m. evidence.verified exige: balance de masa (perfil integrado = flujo de superficie
    integrado), independencia de malla (×2) y de tolerancia temporal (÷100), y coincidencia con la serie analítica de
    Crank cuando D es constante. No incluye reacción (use la herramienta de pellets catalíticos) ni hinchamiento."""
    return SD.transient_diffusion(geometry=geometry, initial_concentration_mol_m3=initial_concentration_mol_m3,
                                  thickness_m=thickness_m, exposed_faces=exposed_faces, radius_m=radius_m, length_m=length_m,
                                  width_m=width_m, depth_m=depth_m, diffusivity_m2_s=diffusivity_m2_s,
                                  diffusivity_law=diffusivity_law, surface_condition=surface_condition,
                                  surface_concentration_mol_m3=surface_concentration_mol_m3,
                                  fluid_concentration_mol_m3=fluid_concentration_mol_m3,
                                  mass_transfer_coefficient_m_s=mass_transfer_coefficient_m_s,
                                  partition_coefficient=partition_coefficient, bath_volume_m3=bath_volume_m3,
                                  solid_volume_m3=solid_volume_m3, times_s=times_s,
                                  target_fractional_uptake=target_fractional_uptake, n_cells=n_cells,
                                  profile_points=profile_points)


def membrane_permeation(layers: list[dict] | None = None, geometry: str | None = None,
                        upstream_partial_pressure_Pa: float | None = None, downstream_partial_pressure_Pa: float | None = None,
                        upstream_concentration_mol_m3: float | None = None, downstream_concentration_mol_m3: float | None = None,
                        area_m2: float | None = None, inner_radius_m: float | None = None, length_m: float | None = None,
                        upstream_side: str | None = None, upstream_film_coefficient: float | None = None,
                        downstream_film_coefficient: float | None = None, transient: bool = False,
                        initial_partial_pressure_Pa: float | None = None, initial_concentration_mol_m3: float | None = None,
                        times_s: list[float] | None = None, cells_per_layer: int = 120) -> dict:
    """Permeación de UN componente por solución-difusión a través de una película densa o de un laminado de capas en serie
    (envases, recubrimientos, liners de tanque, mangueras, paredes de tubo): flujo estacionario, resistencia de cada capa,
    perfil en las interfases y, con transient = true, el time lag (Daynes-Barrer, θ = L²/6D en una capa plana) y la curva
    acumulada aguas abajo. No es un módulo de separación con membranas (eso es otra herramienta).

    Argumentos (SI; ninguno físico tiene valor por defecto):
      layers: lista de capas DE AGUAS ARRIBA A AGUAS ABAJO, cada una {'thickness_m', y dos de: 'diffusivity_m2_s',
        solubilidad, permeabilidad (= D·S)}; opcional 'name'. Con presiones: 'solubility_mol_m3_Pa' (o
        'solubility_m3STP_m3_atm') y 'permeability_mol_m_m2_s_Pa' (o 'permeability_barrer', o
        'permeability_m3STP_m_m2_s_atm' como en Geankoplis). Con concentraciones: 'partition_coefficient'
        (c_capa/c_fluido) y 'permeability_m2_s'. La permeabilidad sola basta para el estacionario; el transitorio exige D.
      geometry: 'flat' (area_m2 opcional), 'cylindrical' (inner_radius_m, upstream_side 'inner'|'outer', length_m
        opcional) o 'spherical' (inner_radius_m, upstream_side).
      Fuerza impulsora, una de las dos parejas: upstream/downstream_partial_pressure_Pa, o
        upstream/downstream_concentration_mol_m3 (concentraciones en los fluidos).
      upstream_film_coefficient / downstream_film_coefficient: películas externas opcionales, en mol/(m²·s·Pa) con
        presiones o m/s con concentraciones.
      transient: true para el time lag; exige initial_partial_pressure_Pa (o initial_concentration_mol_m3): con qué valor
        de fluido está en equilibrio la membrana en t = 0 (0 = desgasificada). times_s opcional para la curva.

    Devuelve flux_mol_m2_s (plana) o rate_per_length_mol_s_m / rate_mol_s y flujos en ambas superficies (curvas), layers
    (resistencia, fracción de la resistencia total, valores en cada cara, permeabilidad en SI y Barrer, time lag propio),
    films, total_resistance, controlling_resistance y transient {time_lag_s, time_lag_frisch_integral_s,
    time_lag_L2_over_6D_s, curve}. evidence.verified exige: estacionario de volúmenes finitos = suma de resistencias, y en
    el transitorio balance de masa, flujo aguas abajo ya estacionario, time lag numérico = integral de Frisch e
    independencia de malla y de tolerancia."""
    return SD.membrane_permeation(layers=layers, geometry=geometry, upstream_partial_pressure_Pa=upstream_partial_pressure_Pa,
                                  downstream_partial_pressure_Pa=downstream_partial_pressure_Pa,
                                  upstream_concentration_mol_m3=upstream_concentration_mol_m3,
                                  downstream_concentration_mol_m3=downstream_concentration_mol_m3, area_m2=area_m2,
                                  inner_radius_m=inner_radius_m, length_m=length_m, upstream_side=upstream_side,
                                  upstream_film_coefficient=upstream_film_coefficient,
                                  downstream_film_coefficient=downstream_film_coefficient, transient=transient,
                                  initial_partial_pressure_Pa=initial_partial_pressure_Pa,
                                  initial_concentration_mol_m3=initial_concentration_mol_m3, times_s=times_s,
                                  cells_per_layer=cells_per_layer)


def diffusivity_from_uptake(geometry: str | None = None, times_s: list[float] | None = None, uptake: list[float] | None = None,
                            uptake_kind: str | None = None, thickness_m: float | None = None, exposed_faces: int | None = None,
                            radius_m: float | None = None, equilibrium_uptake: float | None = None,
                            fit_equilibrium_uptake: bool = False, mass_transfer_coefficient_m_s: float | None = None,
                            partition_coefficient: float | None = None, confidence_level: float = 0.95) -> dict:
    """Ajusta la difusividad D (constante) de un sólido a los datos de un ensayo de sorción/desorción del usuario
    (captación frente a tiempo) en placa, cilindro o esfera, por mínimos cuadrados sobre la solución de Crank.

    Argumentos: geometry 'slab' (thickness_m, exposed_faces 1|2), 'cylinder' o 'sphere' (radius_m); times_s [s] y uptake
    (misma longitud); uptake_kind = 'fractional' si uptake ya es M_t/M_∞, o 'amount' si está en sus unidades (g, %, mol),
    y entonces equilibrium_uptake (M_∞ medido) o fit_equilibrium_uptake = true para ajustarlo también. Opcional película
    externa: mass_transfer_coefficient_m_s y partition_coefficient (K = c_sólido/c_fluido). Nada se supone: sin tamaño de
    muestra o sin M_∞ se rechaza.

    Devuelve diffusivity_m2_s, su intervalo de confianza (confidence_level, sobre ln D), equilibrium_uptake, fit (RMSE,
    R², residuos, Durbin-Watson, autocorrelación, rachas de signo, intervalo de Fourier cubierto), cross_checks (método
    del tiempo medio y de la pendiente inicial en √t) y avisos cuando los residuos indican comportamiento no fickiano o
    los datos no determinan D. evidence.verified certifica el ajuste (tres arranques dan la misma D y la serie coincide
    con el motor numérico), no la calidad de los datos."""
    return SD.fit_diffusivity_from_uptake(geometry=geometry, times_s=times_s, uptake=uptake, uptake_kind=uptake_kind,
                                          thickness_m=thickness_m, exposed_faces=exposed_faces, radius_m=radius_m,
                                          equilibrium_uptake=equilibrium_uptake, fit_equilibrium_uptake=fit_equilibrium_uptake,
                                          mass_transfer_coefficient_m_s=mass_transfer_coefficient_m_s,
                                          partition_coefficient=partition_coefficient, confidence_level=confidence_level)


SOLID_DIFFUSION_TOOLS = [transient_diffusion_solid, membrane_permeation, diffusivity_from_uptake]
