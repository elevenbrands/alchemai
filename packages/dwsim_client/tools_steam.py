"""Agent tools for the steam diagnostic (bloque M19). No DWSIM flowsheet needed. Engines: `packages.energy.steam_diagnostic`
(orchestrates `combustion_efficiency`, M18, and `utility_costs`, M17, and adds the savings measures) and
`packages.energy.steam_report` (the client-readable report, assembled without any LLM)."""
from __future__ import annotations

from packages.energy import steam_diagnostic as SD
from packages.energy import steam_report as SR


def steam_system_diagnostic(currency: str, fuel: dict | None = None, combustion: dict | None = None,
                            boiler: dict | None = None, water: dict | None = None, fuel_cost: dict | None = None,
                            water_cost: dict | None = None, electricity_cost: dict | None = None,
                            operation: dict | None = None, measures: dict | None = None, reference: str | None = None,
                            exchange_rates: dict | None = None, today: str | None = None) -> dict:
    """Diagnóstico de vapor de una caldera, de punta a punta en una sola llamada: cuánto cuesta hoy el vapor, qué eficiencia
    tiene la caldera y dónde se pierde la energía, cuánto ahorra cada medida y todas juntas, y de dónde sale cada número.
    Encadena combustion_efficiency (M18) y utility_costs (M17) sin repetir su física: la línea base es una corrida de cada
    una; cada medida, la misma corrida con un solo cambio. Ningún dato de la planta, precio, objetivo ni inversión se supone.

    currency: moneda ('MXN', 'USD'...).
    fuel: como en combustion_efficiency (gas por composición molar, o análisis elemental con poder calorífico medido). Si
      es gas natural y la planta declara que NO tiene su análisis: {type: 'natural_gas_unknown', T_K, heating_value?
      {value_J_m3, basis, volume_reference {T_K, P_Pa}}, reference_gas?}. Con poder calorífico (el de heating_value, o
      energy_GJ + volume_m3 en fuel_cost.bill) se ajusta una mezcla metano + etano (o metano + nitrógeno) que lo da
      exactamente; sin él se usa un gas de referencia (por defecto 'GULF-COAST', leído de CoolProp). Siempre marcado como
      SUPUESTO, con su sensibilidad en `fuel_resolution`.
    combustion: {site_pressure_Pa (atmosférica del sitio), air {T_K, relative_humidity o humidity_ratio_kg_kg}, flue_O2
      {mole_fraction, basis 'dry'|'wet'} O excess_air_fraction, stack_T_K, setting_loss_fraction + setting_loss_basis
      ('HHV'|'LHV'; pérdida por paredes) O setting_loss_unknown=true SOLO si la planta declara que no la conoce: entonces
      se usa 1.0 % (PCS) de ORNL/TM-2001/263 sec. 4.2.5, marcado como SUPUESTO en avisos, procedencia y reporte, con su
      sensibilidad en `setting_loss`; sin ninguno de los dos la tool se niega; opcional flue_CO_dry_mole_fraction}.
    boiler: {steam_pressure_Pa (ABSOLUTA), blowdown_fraction (purga, fracción de la alimentación); opcionales superheat_K,
      auxiliary_power_kWh_per_tonne (ventiladores y bombas; sin él la electricidad no se cobra) y rated_steam_capacity_kg_s
      (capacidad de placa: con ella y la producción se conoce la carga; escala la pérdida por paredes publicada)}.
    El resultado trae además: `baseline.emissions` (CO2 directo, del balance de carbono) y el CO2 evitado por medida;
    `max_investment_for_payback` (1, 2 y 3 veces el ahorro anual: tope de inversión, no una cotización); `data_review`
    (datos de la planta que conviene confirmar: señala, no corrige ni rechaza) y `assumed_values` (todo lo supuesto).
    water: {feedwater_heating: 'deaerator_steam' (desaireador o tanque calentado con vapor de la caldera) | 'mixing_only'
      (tanque sin calentador: la temperatura de alimentación se calcula de la mezcla) | 'not_charged' (calentar hasta
      feedwater_T_K no se cobra); feedwater_T_K (no hace falta con 'mixing_only'; si se da, se compara), makeup_water_T_K,
      condensate_return_fraction (del vapor), condensate_T_K}.
    fuel_cost: {price_per_GJ, basis 'HHV'|'LHV'} o {bill: {amount, period, y energy_GJ + basis | volume_m3 +
      volume_reference {T_K, P_Pa} | mass_kg}} (recibos: importe pagado y energía o volumen facturados).
    water_cost: {price_per_m3} o {bill: {amount, volume_m3, period}}. El agua nunca viene del banco.
    electricity_cost (solo si das auxiliary_power_kWh_per_tonne): {price_per_kWh} | {bill: {amount, energy_kWh, period}} |
      {time_of_use_per_kWh {periodo: precio}, time_of_use_fractions}; opcionales demand_charge_per_kW_month y load_factor.
    reference: id de un precio PUBLICADO del banco (list_utility_price_references, p. ej. 'MX-CFE-GDMTH-GOLFO-NORTE-2026-11');
      solo si el usuario lo pide; el resultado y el reporte lo marcan como referencia, no contrato. El gas de México no está.
    operation: {hours_per_year, y steam_generated_kg_s (vapor que sale de la caldera; t/h ÷ 3.6) O fuel_consumption
      {energy_GJ_per_year + basis | volume_m3_per_year + volume_reference | mass_kg_per_year | from_fuel_bill: true}}. Uno da
      el otro por la eficiencia y el resultado dice cuál se derivó; con los dos, da annual_basis ('steam'|'fuel') y se
      comparan. Sin operation solo hay cifras por tonelada.
    measures (cada objetivo es de la planta; sin objetivo la medida sale «no evaluada: falta el dato»):
      reduce_excess_air {target_flue_O2 {mole_fraction, basis} o target_excess_air_fraction; opcional stack_T_K_after},
      reduce_stack_temperature {target_stack_T_K; opcional device 'economizer'|'air_preheater'},
      reduce_blowdown {target_blowdown_fraction},
      recover_blowdown_heat {flash_pressure_Pa y/o drain_T_K},
      increase_condensate_return {target_condensate_return_fraction; opcionales condensate_T_K_after, feedwater_T_K_after}.
      Cada una acepta investment (en currency): solo con ella se calcula el retorno simple; la inversión nunca se estima.
    exchange_rates, today: como en utility_costs.

    Devuelve baseline (boiler: eficiencia y pérdidas; steam: costo por tonelada y por GJ con desglose; annual: combustible,
    agua, electricidad y costo al año), measures (ahorro de energía y dinero, nueva eficiencia y costo, retorno o por qué no),
    combined (todas juntas en una corrida, junto a la suma de las individuales: no son aditivas), provenance (dato de la
    planta / calculado con qué herramienta / referencia publicada con fuente y fecha), missing_data (qué faltó y qué impide),
    evidence (cierres de balance, verified), warnings, assumptions, not_included e inputs. Una medida que baja la chimenea
    al punto de rocío del agua o por debajo no se evalúa; el punto de rocío ácido NO se modela. Validación: ejemplos de los
    Steam Tip Sheets #3, #4, #8, #9 y #10 del DOE y una prueba de caldera de MEASUR (BENCHMARK.md, M19). Para el documento
    que lee el cliente pasa el resultado a steam_diagnostic_report."""
    return SD.steam_system_diagnostic(currency, fuel, combustion, boiler, water, fuel_cost, water_cost, electricity_cost,
                                      operation, measures, reference, exchange_rates, today)


def steam_diagnostic_report(diagnostic: dict | None = None, inputs: dict | None = None, plant_label: str | None = None,
                            notice: str | None = None) -> dict:
    """Reporte del diagnóstico de vapor para el cliente, en español y en Markdown: resumen de un párrafo con los tres
    números que importan, cómo opera hoy la caldera, a dónde se va la energía, medidas ordenadas por ahorro anual, caso
    combinado, lo que no se evaluó y por qué, de dónde sale cada número, supuestos y límites, comprobaciones y glosario.
    El texto se arma de forma determinista con el resultado del motor (sin llamar a un modelo de lenguaje): no puede traer un
    número que el motor no produjo.

    Da inputs (los MISMOS argumentos de steam_system_diagnostic, como dict: es lo más seguro, los números no pasan por el
    agente) o diagnostic (el resultado completo de steam_system_diagnostic, sin modificar). El reporte siempre se reconstruye
    corriendo el motor sobre las entradas; si diagnostic no coincide con esa corrida, la herramienta se niega.
    plant_label: cómo llamar a la planta en el título. notice: aviso que va arriba en negritas (por ejemplo «CASO
    ILUSTRATIVO: no es una planta real»).

    Devuelve markdown (el reporte), headline (los números del resumen), sections, evaluated_measures,
    not_evaluated_measures, uses_published_reference_price, verified, integrity y diagnostic (el resultado del motor usado).
    No muestres al usuario cifras que no estén en markdown o en diagnostic."""
    return SR.steam_diagnostic_report(diagnostic, inputs, plant_label, notice)


STEAM_TOOLS = [steam_system_diagnostic, steam_diagnostic_report]
