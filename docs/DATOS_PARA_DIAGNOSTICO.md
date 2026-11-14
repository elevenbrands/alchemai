# Datos para el diagnóstico de vapor

Lista de lo que hay que pedirle a la planta. Sale de las entradas reales de `steam_system_diagnostic` (bloque M19); la columna
«Parámetro» es el nombre exacto que recibe la herramienta. Ningún dato se supone en silencio: lo que falte se niega o se reporta como faltante.
La única excepción, decidida por el owner el 2026-11-14, es la pérdida por paredes de la caldera: si la planta declara que no la
conoce, se usa un valor publicado y el reporte lo marca como supuesto.
Estado: propuesta sin validar con plantas (0 conversaciones al 2026-11-14).

Quién suele tenerlo: **C** jefe de calderas o de servicios · **A** administración o compras · **P** proveedor (quemador, tratamiento de agua, equipo).

## 1. Indispensables: sin ellos no hay diagnóstico (la herramienta se niega y dice cuáles faltan)

| Dato | Quién | Para qué | Parámetro |
|---|---|---|---|
| Combustible: composición del gas (del proveedor de gas), o análisis elemental y poder calorífico si es líquido o sólido | A / P | Poder calorífico, aire necesario y gases de combustión | `fuel` |
| Lo que se paga por el combustible: recibos (importe y energía, volumen o masa facturados, y el periodo) o precio por GJ. Con recibo en volumen, las condiciones base del medidor; con energía, si es poder calorífico superior o inferior | A | Costo del vapor y valor de cada ahorro | `fuel_cost` (`price_per_GJ` + `basis`, o `bill` con `amount`, `period` y `energy_GJ` + `basis` / `volume_m3` + `volume_reference` / `mass_kg`) |
| Lo que se paga por el agua de repuesto: recibos o precio por m³ (con tratamiento, si se quiere cobrar) | A | Costo del agua en el vapor | `water_cost` (`price_per_m3` o `bill` con `amount`, `volume_m3`, `period`) |
| Presión de operación de la caldera (y la presión atmosférica del sitio, para pasar de manométrica a absoluta) | C | Energía que lleva cada tonelada de vapor; punto de rocío | `boiler.steam_pressure_Pa`, `combustion.site_pressure_Pa` |
| Oxígeno medido en la chimenea, y si el analizador lo da en base seca o húmeda (o el exceso de aire) | C | Cuánto aire sobra: la mayor pérdida controlable | `combustion.flue_O2` o `combustion.excess_air_fraction` |
| Temperatura de los gases en la chimenea, a la misma carga que el O2 | C | Pérdida por chimenea | `combustion.stack_T_K` |
| Temperatura y humedad del aire que entra al quemador | C | Pérdida por chimenea y humedad de los gases | `combustion.air` (`T_K` y `relative_humidity` o `humidity_ratio_kg_kg`) |
| Pérdida por paredes y radiación de la caldera, como porcentaje del poder calorífico (hoja de datos del fabricante o prueba) | P / C | Eficiencia de la caldera. **Si la planta no la conoce**, se declara y se usa 1.0 % (base PCS) de la guía ORNL/TM-2001/263 del DOE (sección 4.2.5: calderas de tubos de agua en buen estado, cerca de plena carga), marcado como supuesto y con su sensibilidad | `combustion.setting_loss_fraction`, `combustion.setting_loss_basis`; o `combustion.setting_loss_unknown` |
| Purga, como porcentaje del agua de alimentación (0 si de verdad no hay) | C / P | Calor y agua que se tiran por la purga | `boiler.blowdown_fraction` |
| Cómo se calienta el agua de alimentación: con vapor de la caldera (desaireador o tanque con serpentín o inyección), sin calentador, o no se quiere cobrar | C | Define cuánto vapor se gasta en la propia casa de calderas | `water.feedwater_heating` (`deaerator_steam`, `mixing_only`, `not_charged`) |
| Temperatura del agua que entra a la caldera (no hace falta si el tanque no tiene calentador; si se da, se compara) | C | Calor que debe poner el combustible | `water.feedwater_T_K` |
| Temperatura del agua de repuesto | C | Balance del tanque de alimentación | `water.makeup_water_T_K` |
| Retorno de condensado, como porcentaje del vapor, y su temperatura | C | Agua de repuesto y calor que se recupera | `water.condensate_return_fraction`, `water.condensate_T_K` |

## 2. Para tener cifras al año (sin ellos solo hay cifras por tonelada de vapor y porcentajes)

| Dato | Quién | Para qué | Si falta | Parámetro |
|---|---|---|---|---|
| Horas de operación al año | C | Pasar de «por tonelada» a «por año» | No hay combustible, agua, costo ni ahorros anuales, ni retorno simple | `operation.hours_per_year` |
| Producción de vapor de la caldera (medidor a la salida), **o** consumo anual de combustible (medidor o recibos de un año) | C / A | Uno da el otro por la eficiencia; el reporte dice cuál se calculó | Igual que arriba | `operation.steam_generated_kg_s` o `operation.fuel_consumption` (`energy_GJ_per_year` + `basis`, `volume_m3_per_year` + `volume_reference`, `mass_kg_per_year` o `from_fuel_bill`) |
| Si se dan los dos: con cuál se calculan las cifras anuales | — | El otro sirve para comparar la eficiencia por entradas y salidas con la del método de pérdidas | La herramienta se niega hasta que se elija | `operation.annual_basis` (`steam` o `fuel`) |

## 3. Opcionales: afinan el resultado

| Dato | Quién | Para qué | Si falta | Parámetro |
|---|---|---|---|---|
| Electricidad de ventiladores y bombas de la caldera, en kWh por tonelada de vapor | C | Cobrar la electricidad del vapor | El costo del vapor no incluye electricidad y el reporte lo dice | `boiler.auxiliary_power_kWh_per_tonne` |
| Lo que se paga por la electricidad: recibos (importe y kWh) o precio por kWh; o tarifa horaria con el reparto de la energía por periodo, cargo por demanda y factor de carga | A | Valorar esa electricidad | Con consumo eléctrico pero sin precio, la herramienta se niega; se puede nombrar una tarifa publicada de CFE (`reference`) y el reporte la marca como referencia, no contrato | `electricity_cost` (`price_per_kWh`, `bill`, `time_of_use_per_kWh`, `time_of_use_fractions`, `demand_charge_per_kW_month`, `load_factor`), `reference` |
| Monóxido de carbono medido en la chimenea | C | Pérdida por combustión incompleta | Se toma combustión completa y se dice | `combustion.flue_CO_dry_mole_fraction` |
| Sobrecalentamiento del vapor, si lo hay | C | Entalpía del vapor | Se toma vapor saturado | `boiler.superheat_K` |
| Tipo de cambio, si un precio está en otra moneda | A | Convertir precios | Nunca se usa un tipo de cambio por defecto | `exchange_rates` |

## 4. Para evaluar cada medida de ahorro (el objetivo siempre lo da la planta o su proveedor)

Sin el objetivo, la medida sale como «no evaluada: falta el dato». Sin la inversión, el ahorro sí se calcula pero el retorno simple no: la inversión nunca se estima.

| Medida | Objetivo que hay que pedir | Quién | Parámetro |
|---|---|---|---|
| Afinar la combustión | O2 (o exceso de aire) que el quemador sostiene sin producir CO; si se sabe, la temperatura de chimenea después de afinar | P / C | `measures.reduce_excess_air` (`target_flue_O2` o `target_excess_air_fraction`, `stack_T_K_after`) |
| Economizador o precalentador de aire | Temperatura de chimenea a la salida del equipo y cuál de los dos es | P | `measures.reduce_stack_temperature` (`target_stack_T_K`, `device`) |
| Bajar la purga | Purga que permite el tratamiento de agua | P | `measures.reduce_blowdown` (`target_blowdown_fraction`) |
| Recuperar el calor de la purga | Presión del tanque de flasheo y/o temperatura de la purga al drenaje | P | `measures.recover_blowdown_heat` (`flash_pressure_Pa`, `drain_T_K`) |
| Regresar más condensado | Porcentaje del vapor que se podría regresar y a qué temperatura; si el calentamiento no se cobra, la nueva temperatura del agua de alimentación | C | `measures.increase_condensate_return` (`target_condensate_return_fraction`, `condensate_T_K_after`, `feedwater_T_K_after`) |
| Cualquiera | Inversión cotizada, en la moneda del diagnóstico | P / A | `investment` dentro de cada medida |

## Qué no se pide porque no se usa todavía

Curva de carga a lo largo del año, purga de fondo, fugas y trampas de vapor, aislamiento, análisis del agua, emisiones. El diagnóstico trabaja con un solo punto de operación promedio y lo declara en sus supuestos.

Unidades: la herramienta recibe SI (K, Pa absolutos, kg/s, fracciones). La planta puede dar °C, kg/cm² manométricos, t/h y porcentajes; la conversión la hace quien captura, y el reporte devuelve °C, bar y kg/cm², t y porcentajes.
