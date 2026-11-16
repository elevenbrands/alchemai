# Refinería Miguel Hidalgo (Tula de Allende, Hidalgo) — proceso, rentabilidad y oportunidades

Periodo 2026-01/2026-09 · modo «as_operated» (calibrado a la producción publicada de Pemex donde Pemex no publica la operación) · generado el 2026-11-16 por AlchemAI. Modelo de PLANEACIÓN por rendimientos, no simulación rigurosa de cada reactor.

Diagramas: [vista por secciones](refineria_de_tula_secciones.svg) · [vista completa](refineria_de_tula_completo.svg)

## Qué hace la refinería

Procesa 221,583 barriles diarios de crudo.

- **Recibo de crudo.** Recibe el crudo (mezcla de Maya, Istmo y Olmeca según lo publicado), lo almacena y lo bombea al tren de proceso. Equipos: Tanques de crudo.
- **Destilación.** Destilación atmosférica y al vacío: separa el crudo por punto de ebullición en gases, nafta, kerosina, gasóleos y residuo. Es la base de todo; las demás unidades transforman estos cortes. Equipos: Calentador de carga atmosférica, Destilación atmosférica, Calentador de vacío, Destilación al vacío.
- **Conversión de residuos y gasóleos.** Conversión: el craqueo catalítico (FCC) rompe gasóleos pesados en gasolina y gas LP; la coquizadora convierte el residuo de vacío en productos ligeros y coque. Aquí se gana la mayor parte del margen. Equipos: Calentador de coquizadora, Coquización retardada (tambores), Reductora de viscosidad, Desintegración catalítica (FCC), Fraccionadora y planta de gas del FCC.
- **Naftas y gasolinas.** Mejoramiento de gasolinas: la reformadora sube el octanaje de la nafta (y produce hidrógeno); la alquilación e isomerización hacen componentes de alto octano para la mezcla. Equipos: Hidrotratadora de naftas, Reformación catalítica, Alquilación, Postratamiento de gasolina catalítica.
- **Hidrodesulfuración de destilados.** Hidrotratamiento (HDS): quita el azufre de nafta, turbosina y diésel con hidrógeno para cumplir la NOM-016-CRE-2016. Equipos: Hidrodesulfuradora de turbosina, Hidrodesulfuradora de diésel.
- **Hidrógeno, gas combustible y azufre.** Servicios: hidrógeno, tratamiento de gases ácidos y recuperación de azufre, vapor, agua y energía. Equipos: Cabezal de hidrógeno, Sistema de gas combustible, Aminas y plantas de azufre (Claus), Chimeneas de calentadores y regenerador.
- **Productos.** Mezclado y almacenamiento de productos terminados para venta. Equipos: Mezcla y tanques de gasolinas, Turbosina, Diésel UBA, Combustóleo, Gas licuado, Asfalto, Manejo de coque.

## Productos (modelo) y validación contra lo publicado

| Producto | Modelo, bbl/d | Modelo, t/h | % vol. del crudo |
|---|---|---|---|
| gasolina | 85,666 | 444.6 | 38.7 |
| turbosina | 20,908 | 112.4 | 9.4 |
| diesel | 59,710 | 339.5 | 26.9 |
| combustoleo | 25,420 | 175.1 | 11.5 |
| asfalto | 2,882 | 19.7 | 1.3 |
| coque | — (se reporta en masa) | 48.5 | 0.0 |
| glp | 11,298 | 40.4 | 5.1 |
| butanos | 5,233 | 20.5 | 2.4 |
| gas_seco | — (se reporta en masa) | 33.2 | 0.0 |

| Producto | Tipo de comparación | Desviación |
|---|---|---|
| gasolina | validación | -4.6 % |
| turbosina | ajustado | 0.0 % |
| diesel | validación | 43.7 % |
| combustoleo | ajustado | 19.3 % |
| glp | ajustado | 0.0 % |
| asfalto | dato de entrada | 0.0 % |
| coque | ajustado | -0.0 % |

**Calidad de la calibración:** Calibración EN EL LÍMITE (el modelo no reproduce el dato publicado): cutter_fraction = 0.00 → combustoleo modelo 25.42 vs publicado 21.31 Mbd. Diagnóstico: el exceso de diésel (+18.2 Mbd) va con un exceso de líquidos totales (+18.1 Mbd): el modelo hace demasiado LÍQUIDO, no sólo lo rutea mal. El autoconsumo de combustibles líquidos no lo explica: aun suponiendo que todo el consumo propio del sector energético (SENER, BNE 2024: 2.9 % de la energía de los líquidos de refinería; diésel 7.0 %) ocurriera en refinerías y Pemex lo excluyera de su elaboración, explicaría a lo más 5.6 Mbd de líquidos (3.1 Mbd de diésel). La causa sigue abierta (cortes de destilación, rendimientos de conversión, mermas no publicadas).

Parámetros CALIBRADOS (no publicados por Pemex; ajustados para reproducir un producto publicado):

- coker_utilization: 0.215 (ajustado a coque)
- cutter_fraction: 0.000 (ajustado a combustoleo)
- kerosene_to_jet_fraction: 0.536 (ajustado a turbosina)
- butanes_to_gas_plants_fraction: 0.782 (ajustado a glp)

## Rentabilidad (margen de operación; la refinería ya existe)

| Indicador | USD/bbl |
|---|---|
| Margen bruto del modelo | 21.92 |
| Margen bruto con la producción PUBLICADA (misma mezcla de crudo y precios) | 9.34 |
| Energía | 2.26 |
| Costo fijo (referencia) | 4.86 |
| Margen neto del modelo | 14.80 |

La cifra más defendible es el margen con la producción publicada: el modelo produce más diésel y líquidos que la refinería real (ver validación), lo que infla su margen. Diagnóstico: mismo crudo y mismos precios, pero con la producción PUBLICADA de los seis productos líquidos en lugar de la modelada (sin gas seco, coque ni azufre). La diferencia con gross_margin_usd_bbl es el efecto de los productos no ajustados (gasolinas, diésel) y del volumen de líquidos que el modelo sobrestima.

Referencia de Pemex: 12.0 USD/bbl (Sistema Nacional de Refinación, promedio anual 2025; Pemex, Comunicado No. 9, 12-feb-2026 (publicado en BMV), https://www.bmv.com.mx/docs-pub/eventemi/eventemi_1533128_1.pdf). No es la misma base: es todo el sistema y otro año.

Precios usados (referencias publicadas, NO cotizaciones):

| Precio | Valor | Fuente |
|---|---|---|
| fx_mxn_usd | 18.55 MXN/USD | Banco de México, indicadores de la página principal (Sistema de Información Económica) |
| istmo | 83.61 USD/bbl | Pemex eBDI, «Precio promedio de exportación de petróleo crudo» (cuadro INDI09), dólares po |
| maya | 78.06 USD/bbl | Pemex eBDI, «Precio promedio de exportación de petróleo crudo» (cuadro INDI09), dólares po |
| olmeca | 92.83 USD/bbl | Pemex eBDI, «Precio promedio de exportación de petróleo crudo» (cuadro INDI09), dólares po |
| gasolina | 113.91 USD/bbl | Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen  |
| turbosina | 141.21 USD/bbl | Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen  |
| diesel | 146.69 USD/bbl | Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen  |
| combustoleo | 54.08 USD/bbl | Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen  |
| asfalto | 81.08 USD/bbl | Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen  |
| glp | 30.58 USD/bbl | U.S. EIA, Mont Belvieu, TX Propane Spot Price FOB, mensual (serie EER_EPLLPA_PF4_Y44MB_DPG |
| gas_seco | 3.97 USD/GJ | U.S. EIA, Natural Gas Industrial Price (Natural Gas Monthly; página con Release Date 10/30 |
| energia | 3.97 USD/GJ | U.S. EIA, Natural Gas Industrial Price (Natural Gas Monthly; página con Release Date 10/30 |
| costo_fijo | 4.86 USD/bbl | Valero Energy Corp., Form 8-K, Exhibit 99.01, resultados del tercer trimestre de 2026, tab |

## Oportunidades de mejora (numeradas en el diagrama)

1. **Completar el arranque de la coquizadora y operarla a su capacidad publicada** — USD 105,518,412/año (rango 52,759,206–105,518,412). Confianza media.
   La coquizadora opera al 22 % (valor CALIBRADO para reproducir el coque publicado). A capacidad, convierte residuo de vacío (que hoy sale como combustóleo, el producto más barato) en gasolina, diésel y gas; el modelo valúa ese cambio con los mismos precios.

No evaluado por falta de datos públicos:

- a_heat_integration: Integración térmica: hace falta al menos una corriente caliente y una fría con T_in_K, T_out_K y carga (o heat_integration.streams).
- b_excess_air: horno_atm: sin datos de combustión (combustible, aire, O2 en humos, temperatura de chimenea, pérdida por paredes) no se puede calcular la eficiencia ni el efecto del exceso de aire.
- b_furnace_efficiency: horno_atm: sin eficiencia (medida, calculada o absorbida/combustible).
- b_excess_air: horno_vacio: sin datos de combustión (combustible, aire, O2 en humos, temperatura de chimenea, pérdida por paredes) no se puede calcular la eficiencia ni el efecto del exceso de aire.
- b_furnace_efficiency: horno_vacio: sin eficiencia (medida, calculada o absorbida/combustible).
- b_excess_air: horno_coquizadora: sin datos de combustión (combustible, aire, O2 en humos, temperatura de chimenea, pérdida por paredes) no se puede calcular la eficiencia ni el efecto del exceso de aire.
- b_furnace_efficiency: horno_coquizadora: sin eficiencia (medida, calculada o absorbida/combustible).
- b_excess_air: reductora_viscosidad: sin datos de combustión (combustible, aire, O2 en humos, temperatura de chimenea, pérdida por paredes) no se puede calcular la eficiencia ni el efecto del exceso de aire.
- b_furnace_efficiency: reductora_viscosidad: sin eficiencia (medida, calculada o absorbida/combustible).
- d_energy_intensity: cdu: la energía del modelo da exactamente la intensidad típica (CT) del estudio: parece salir del mismo benchmark, así que no es una brecha medida. Da la energía medida.
- d_energy_intensity: vdu: la energía del modelo da exactamente la intensidad típica (CT) del estudio: parece salir del mismo benchmark, así que no es una brecha medida. Da la energía medida.
- d_energy_intensity: coquizadora: la energía del modelo da exactamente la intensidad típica (CT) del estudio: parece salir del mismo benchmark, así que no es una brecha medida. Da la energía medida.

## Supuestos, advertencias y lo que no incluye

- Capacidad de coker = 86.0 Mbd tomada de Oil & Gas Journal, R. Brelsford, «Pemex lets contract for Tula refinery», 11-jun-2018 (planta de coquización retardada de 86,000 b/d del proyecto Aprovechamiento de Residuales) [fuente secundaria]: no está en el cuadro de capacidades de la eBDI 2025. Estado: arranque parcial.
- Carga de crudo = proceso PUBLICADO por Pemex para el periodo 2026-01/2026-09 por tipo (maya 49.737 Mbd, istmo 96.677 Mbd, olmeca 75.169 Mbd); Correspondencia nuestra entre los tipos de crudo de la eBDI (Pesado, Ligero, Superligero) y los crudos de exportación Maya, Istmo y Olmeca; las mezclas actuales que recibe cada refinería pueden diferir de los ensayes del año 2000.
- Asfalto = 2882 bbl/d = producción PUBLICADA del periodo 2026-01/2026-09 (dato de entrada, no predicción): se retira del residuo de vacío.
- Conversión del FCC = 66.47 % peso = caso «media» de la tabla publicada (Niccum/KBR 2013): Pemex no publica la conversión de sus FCC; en mode='as_operated' se calibra si hay un producto libre que la identifique.
- La capacidad publicada de «alquilación e isomerización» se asigna completa a alquilación (barriles de alquilado); no hay reparto publicado.
- Hidrotratadoras: toda su carga se hidrotrata a especificación salvo que exista una capacidad PROPIA publicada para esa planta; lo que exceda esa capacidad pasa sin tratar a su producto, marcado fuera de especificación. Nunca se pierde producto. Sin capacidad propia, se compara la carga sumada con la capacidad total publicada (hds_total) y sólo se avisa.
- Precio fx_mxn_usd = 18.55 MXN/USD (FIX del banco de referencia; Banco de México, indicadores de la página principal (Sistema de Información Económica); https://www.banxico.org.mx/canales/singleFix.json; periodo {'start': '2026-11-13', 'end': '2026-11-13'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio istmo = 83.61 USD/bbl (promedio simple de 9 meses del precio de exportación (costo de oportunidad del crudo); Pemex eBDI, «Precio promedio de exportación de petróleo crudo» (cuadro INDI09), dólares por barril; https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=INDI09; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio maya = 78.06 USD/bbl (promedio simple de 9 meses del precio de exportación (costo de oportunidad del crudo); Pemex eBDI, «Precio promedio de exportación de petróleo crudo» (cuadro INDI09), dólares por barril; https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=INDI09; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio olmeca = 92.83 USD/bbl (promedio simple de 8 meses del precio de exportación (costo de oportunidad del crudo); Pemex eBDI, «Precio promedio de exportación de petróleo crudo» (cuadro INDI09), dólares por barril; https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=INDI09; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio gasolina = 113.9 USD/bbl (valor/volumen de ventas internas del SNR (Gasolinas (Pemex Magna + Pemex Premium + gasaviones)) = 13292.81 MXN/m3, a USD con el tipo de cambio de la tabla; Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen de ventas internas» (cuadro RVOLVIN, metros cúbicos); https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=RVALORI; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio turbosina = 141.2 USD/bbl (valor/volumen de ventas internas del SNR (Turbosina) = 16478.45 MXN/m3, a USD con el tipo de cambio de la tabla; Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen de ventas internas» (cuadro RVOLVIN, metros cúbicos); https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=RVALORI; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio diesel = 146.7 USD/bbl (valor/volumen de ventas internas del SNR (Diesel (todas las calidades)) = 17117.30 MXN/m3, a USD con el tipo de cambio de la tabla; Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen de ventas internas» (cuadro RVOLVIN, metros cúbicos); https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=RVALORI; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio combustoleo = 54.08 USD/bbl (valor/volumen de ventas internas del SNR (Combustóleo (pesado + intermedios)) = 6310.44 MXN/m3, a USD con el tipo de cambio de la tabla; Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen de ventas internas» (cuadro RVOLVIN, metros cúbicos); https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=RVALORI; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio asfalto = 81.08 USD/bbl (valor/volumen de ventas internas del SNR (Asfaltos) = 9461.36 MXN/m3, a USD con el tipo de cambio de la tabla; Pemex eBDI, «Valor de las ventas internas» (cuadro RVALORI, millones de pesos) y «Volumen de ventas internas» (cuadro RVOLVIN, metros cúbicos); https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=RVALORI; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio glp = 30.58 USD/bbl (promedio simple ene-sep 2026 del propano Mont Belvieu × 42 gal/bbl; U.S. EIA, Mont Belvieu, TX Propane Spot Price FOB, mensual (serie EER_EPLLPA_PF4_Y44MB_DPG); https://www.eia.gov/dnav/pet/hist_xls/EER_EPLLPA_PF4_Y44MB_DPGm.xls; periodo {'start': '2026-01-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio gas_seco = 3.971 USD/GJ (gas natural del banco de servicios (US): crédito del gas seco de refinería; U.S. EIA, Natural Gas Industrial Price (Natural Gas Monthly; página con Release Date 10/30/2026); https://www.eia.gov/dnav/ng/hist/n3035us3m.htm; periodo {'start': '2026-08-01', 'end': '2026-08-31'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio energia = 3.971 USD/GJ (gas natural del banco de servicios (US): precio de toda la energía en sitio; U.S. EIA, Natural Gas Industrial Price (Natural Gas Monthly; página con Release Date 10/30/2026); https://www.eia.gov/dnav/ng/hist/n3035us3m.htm; periodo {'start': '2026-08-01', 'end': '2026-08-31'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio costo_fijo = 4.86 USD/bbl (gastos de operación de refinación sin depreciación (Valero, región Costa del Golfo de EE. UU.). Referencia de un operador estadounidense grande; la tabla no declara si incluye el gas natural (energía). Pemex no publica su costo operativo por barril por refinería en las fuentes leídas.; Valero Energy Corp., Form 8-K, Exhibit 99.01, resultados del tercer trimestre de 2026, tabla de la región U.S. Gulf Coast (operating expenses excluding depreciation and amortization per barrel of throughput); https://www.sec.gov/Archives/edgar/data/0001035002/000162828026066795/a9302026exh9901earningsrel.htm; periodo {'start': '2026-07-01', 'end': '2026-09-30'}) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- CCR del residuo de vacío de istmo = 22.7 % peso, estimado suponiendo que todo el carbón Conradson del crudo (4.5 %) queda en el residuo de vacío (para Maya ese supuesto da 25.92 vs 25.8 publicado).
- CCR del residuo de vacío de olmeca = 22.4 % peso, estimado suponiendo que todo el carbón Conradson del crudo (2.8 %) queda en el residuo de vacío (para Maya ese supuesto da 25.92 vs 25.8 publicado).
- Coquizadora: coque = (46.88/29.3) × CCR de su carga (relación del ejemplo publicado, forma de Gary-Handwerk); gas, gasolina y gasóleo de coquización en las proporciones del ejemplo; azufre con el reparto publicado del ejemplo.
- FCC: rendimientos (% peso) contra conversión, interpolación lineal entre los tres casos de la Tabla 3 de Niccum (KBR, 2013), planta piloto con carga de 22.5 °API y 0.55 % S; propileno/butilenos por la olefinicidad publicada; iC4/nC4 y reparto de azufre del ejemplo de Jechura; el H2S es un lote adicional y los demás se escalan para cerrar masa. La carga real de Pemex es más pesada y más azufrada.
- LCO: entra a diésel sólo después del destilado primario y dentro de la capacidad de hidrodesulfuración publicada (propia de la HDS de diésel si existe, si no la libre de hds_total); el exceso va a combustóleo (Laredo et al., IMP 2002: el uso de ACL en diésel en México está restringido).
- Reformadora: vector fijo del ejemplo publicado a RON 98 (carga 51.7 °API); el H2S del ejemplo se excluye.
- ⚠ Reductora de viscosidad modelada como paso a combustóleo sin cambio de rendimientos: no se abrió hoy un vector de rendimientos publicado (reduce viscosidad; su pequeña producción de gas y nafta no se ve).
- ⚠ Rendimientos del FCC de una tabla de planta piloto con carga de 22.5 °API y 0.55 % S; la carga modelada tiene 22.8 °API y 2.33 % S (se usan los mismos rendimientos; el azufre con el reparto del ejemplo de Jechura).
- ⚠ Los precios internos de Pemex (MXN, ene-sep 2026) se pasan a USD con el FIX 18.5525 MXN/USD del 2026-11-13 (no es el promedio del periodo): da prices['fx_mxn_usd'] si tienes el promedio.
- ⚠ Gas natural: no hay precio para México en el banco; se usa US (3.971 USD/GJ).
- ⚠ Sin precio publicado para ['coque', 'azufre']: su valor NO se suma a los ingresos (margen conservador). Dalo en prices ({'coque': 'USD/t', 'azufre': 'USD/t'}).
- ⚠ Ensaye maya: la masa sumada de los cortes difiere -0.29 % de la del crudo entero por su °API (inconsistencia de la tabla publicada); el modelo usa la suma de los cortes.
- ⚠ Ensaye istmo: la masa sumada de los cortes difiere -0.73 % de la del crudo entero por su °API (inconsistencia de la tabla publicada); el modelo usa la suma de los cortes.
- ⚠ Ensaye istmo: el residuo 651+ °F publicado (44.3 % vol) no es la suma de gasóleo pesado + residuo de vacío (43.3 % vol); el modelo usa los dos cortes.
- ⚠ Calibración de cutter_fraction: en el límite: el producto publicado no se alcanza dentro de 0-1 (combustoleo: publicado 21.31 Mbd, modelo 25.42 Mbd).
- ⚠ Diagnóstico: el exceso de diésel (+18.2 Mbd) va con un exceso de líquidos totales (+18.1 Mbd): el modelo hace demasiado LÍQUIDO, no sólo lo rutea mal. El autoconsumo de combustibles líquidos no lo explica: aun suponiendo que todo el consumo propio del sector energético (SENER, BNE 2024: 2.9 % de la energía de los líquidos de refinería; diésel 7.0 %) ocurriera en refinerías y Pemex lo excluyera de su elaboración, explicaría a lo más 5.6 Mbd de líquidos (3.1 Mbd de diésel). La causa sigue abierta (cortes de destilación, rendimientos de conversión, mermas no publicadas).
- No incluye: Cinética de reactores, diseño de columnas y efecto de las condiciones de operación: cada unidad usa UN vector de rendimiento publicado a una condición fija (reformadora a RON 98); el FCC usa rendimientos contra conversión de UNA tabla de planta piloto (una carga de 22.5 °API y 0.55 % S, un catalizador), no la carga ni el catalizador de Pemex.
- No incluye: Diluente (cutter) del combustóleo: no se calcula por viscosidad (no hay viscosidad abierta del residuo de vacío ni receta publicada); es una fracción del destilado intermedio (cutter_fraction), calibrada al combustóleo sólo donde éste no identifica ya la coquizadora; donde sí, queda en 0 (no identificable).
- No incluye: Optimización (programación lineal) de la carga, de los cortes y del destino de las corrientes: el ruteo es fijo.
- No incluye: Calidad de mezclas: octano, cetano, presión de vapor, viscosidad del combustóleo y diluentes.
- No incluye: Isomerización (su capacidad está sumada a la de alquilación en la eBDI y no hay reparto publicado), H-Oil, planta de hidrógeno y recuperación real de las plantas de azufre (se reporta el azufre potencial en el gas ácido).
- No incluye: Consumo real de hidrógeno de los hidrotratamientos: se calcula sólo el estequiométrico de la desulfuración (cota inferior).
- No incluye: Mermas, autoconsumos de producto, inventarios, paros y transferencias entre refinerías (no hay dato abierto por refinería; el BNE 2024 da el consumo propio de líquidos de TODO el sector energético, que se usa sólo como cota en published_comparison.liquid_excess_diagnosis).
- No incluye: Costo de construir la refinería (VPN): la refinería existe; la rentabilidad aquí es el margen de operación.
