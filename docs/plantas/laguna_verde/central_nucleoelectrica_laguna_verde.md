# Central Nucleoeléctrica Laguna Verde (CFE), Alto Lucero, Veracruz — ciclo de vapor, unidad 1 — proceso, rentabilidad y oportunidades

Año 2025 · generado el 2026-11-16 por AlchemAI con datos públicos. Ciclo de vapor modelado con IAPWS-95 (CoolProp); el diseño del lado turbina no es público y se calibra (ver abajo).

Diagramas: [vista por secciones](central_nucleoelectrica_laguna_verde_secciones.svg) · [vista completa](central_nucleoelectrica_laguna_verde_completo.svg)

## Qué hace la central

Dos reactores de agua en ebullición (BWR-5) hierven agua dentro de la vasija con el calor de la fisión; el vapor saturado mueve la turbina de alta, pasa por los separadores de humedad y recalentadores (MSR), mueve las turbinas de baja y el generador; el vapor se condensa con agua de mar y el condensado, precalentado con extracciones de la turbina, regresa al reactor. Es un ciclo Rankine directo: el mismo vapor del reactor llega a la turbina.

| Indicador | Valor |
|---|---|
| Potencia térmica por unidad | 2,317 MWt |
| Potencia eléctrica bruta por unidad (calibrada) | 805 MWe |
| Eficiencia bruta | 34.7 % |
| Consumo térmico unitario | 10,362 kJ/kWh |
| Flujo de vapor por unidad | 1,378 kg/s |
| Calor al mar por unidad | 1,522 MW |
| Generación neta 2025 (dos unidades) | 9.83 TWh |

## Rentabilidad (la central ya existe: margen de operación)

| Indicador | Valor |
|---|---|
| Precio de referencia (CENACE) | 52.4 USD/MWh |
| Combustible + operación | 28.04 USD/MWh |
| Margen de operación | 24.4 USD/MWh · 240 MM USD/año |
| Margen después de capital de sostenimiento | 157 MM USD/año |
| CO2 evitado (factor promedio de la red) | 4.37 Mt/año |

Costos de referencia de BWR de EE. UU. (NEI/EUCG), no los de CFE (no publicados).

## Validación contra lo publicado

| Cantidad | Publicado | Modelo | Papel |
|---|---|---|---|
| bruta a 2027 MWt (antes del EPU) | 686.7 MWe | 708.2 | validación |
| bruta de diseño del EPU a 2317 MWt | 817.1 MWe | 805.0 | otra cifra publicada |
| bruta tras el EPU (WNN 2013) | 820.0 MWe | 805.0 | otra cifra publicada |
| calentamiento del agua de mar (descarga − mar), temporada de secas (2027 MWt) | 13.0 K | 10.2 | validación |
| calentamiento del agua de mar (descarga − mar), temporada de lluvias (2027 MWt) | 16.0 K | 10.3 | validación |
| bruta de PRIS, unidad 1 | 805.0 MWe | 805.0 | ajustado |
| bruta de PRIS, unidad 2 | 803.0 MWe | 803.0 | ajustado |

Parámetros CALIBRADOS: {"1": {"turbine_isentropic_efficiency": {"value": 0.795854388305332, "fitted_to": "potencia bruta publicada", "published_gross_W": 805000000.0, "model_gross_W": 805000000.0, "residual_rel": 0.0, "thermal_W": 2317000000.0, "bounds": [0.4, 1.0], "flag": "CALIBRADO a la potencia bruta de PRIS (no es validación); incluye la eficiencia del generador y pérdidas mecánicas"}}, "2": {"turbine_isentropic_efficiency": {"value": 0.793464083761318, "fitted_to": "potencia bruta publicada", "published_gross_W": 803000000.0, "model_gross_W": 802999999.9999932, "residual_rel": -8.461929644326937e-15, "thermal_

## Oportunidades de mejora (numeradas en el diagrama)

1. **Unidad 1: factor de carga 71.4 % en 2025 vs 83 % promedio mundial** — USD 36,987,266/año (rango 12,574,185–36,987,266). Confianza media.
   La unidad 1 entregó 4858 GWh en 2025 (PRIS). Llevarla al promedio mundial de factor de capacidad de 2024 (83 %, WNA) daría 790 GWh más (recargas más cortas, menos salidas forzadas).
2. **Unidad 2: factor de carga 73.3 % en 2025 vs 83 % promedio mundial** — USD 30,849,396/año (rango 10,487,556–30,849,396). Confianza media.
   La unidad 2 entregó 4977 GWh en 2025 (PRIS). Llevarla al promedio mundial de factor de capacidad de 2024 (83 %, WNA) daría 659 GWh más (recargas más cortas, menos salidas forzadas).
3. **Bruta publicada (PRIS) 12–14 MWe por debajo del diseño del EPU** — USD 7,778,808/año (rango 0–7,778,808). Confianza baja.
   El proyecto del EPU (Iberdrola, ENC 2010) declaró 817.1 MWe brutos por unidad; PRIS reporta 805 y 803 MWe. Si la diferencia es real (no sólo una clasificación conservadora), revisar eficiencia de turbinas, vacío del condensador y calentadores.
4. **Recirculación de la pluma térmica a la toma en temporada de lluvias** — USD 1,261,378/año (rango —–—). Confianza baja.
   El estudio publicado midió la toma a 30.0 °C con el mar a 28 °C: la descarga caliente regresa a la toma. Con el ciclo calibrado eso cuesta 6.4 MWe brutos por unidad (condensador de 7.32 a 8.17 kPa). Separar toma y descarga o mejorar la dispersión lo recuperaría.
5. **Calentador 6 fuera de servicio: 1.1 MWe menos** — sin valor anual. Confianza baja.
   Con el calentador superior fuera, el agua entra más fría al reactor; a potencia térmica constante el agua de alimentación baja de 251 a 216 °C y la eficiencia bruta cae. Ilustrativo: la CNLV no publica el número real de calentadores ni la disponibilidad de cada uno.

## Supuestos, advertencias y lo que no incluye

- Precio fx_mxn_usd = 18.55 MXN/USD (FIX del banco de referencia; https://www.banxico.org.mx/canales/singleFix.json) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio energia_mxn_mwh = 972.1 MXN/MWh (promedio simple horario del PND (pz) de la zona VERACRUZ, MDA (mercado del día en adelanto), 8592 h; faltan 168 horas (una semana) en las respuestas del servicio. Precio de la zona de carga Veracruz (no el PML del nodo de la central) y de energía únicamente (sin potencia, CEL ni contratos): CFE no necesariamente vende a este precio.; https://ws01.cenace.gob.mx:8082/SWPEND/SIM/SIN/MDA/VERACRUZ/2025/01/01/2025/01/07/JSON) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio energia_usd_mwh = 52.4 USD/MWh (promedio simple horario del PND (pz) de la zona VERACRUZ, MDA (mercado del día en adelanto), 8592 h; faltan 168 horas (una semana) en las respuestas del servicio. Precio de la zona de carga Veracruz (no el PML del nodo de la central) y de energía únicamente (sin potencia, CEL ni contratos): CFE no necesariamente vende a este precio. Convertido con fx_mxn_usd de esta tabla.; https://ws01.cenace.gob.mx:8082/SWPEND/SIM/SIN/MDA/VERACRUZ/2025/01/01/2025/01/07/JSON) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio combustible_usd_mwh = 5.55 USD/MWh (NEI/EUCG 2025, promedio de 20 centrales BWR de EE. UU. (USD 2025); no es el costo de la CNLV, que CFE no publica; https://www.nei.org/getContentAsset/47fa8caa-9b0d-4029-932c-07f902e82f4f/8d8ff8d6-b2ae-401b-a63c-f6b108e809d2/2024-Costs-in-Context-final.pdf) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio operacion_usd_mwh = 22.49 USD/MWh (NEI/EUCG 2025, promedio de 20 centrales BWR de EE. UU. (USD 2025); no es el costo de la CNLV, que CFE no publica; https://www.nei.org/getContentAsset/47fa8caa-9b0d-4029-932c-07f902e82f4f/8d8ff8d6-b2ae-401b-a63c-f6b108e809d2/2024-Costs-in-Context-final.pdf) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Precio capital_usd_mwh = 8.43 USD/MWh (NEI/EUCG 2025, promedio de 20 centrales BWR de EE. UU. (USD 2025); no es el costo de la CNLV, que CFE no publica; https://www.nei.org/getContentAsset/47fa8caa-9b0d-4029-932c-07f902e82f4f/8d8ff8d6-b2ae-401b-a63c-f6b108e809d2/2024-Costs-in-Context-final.pdf) — VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN.
- Presión de domo 1020 psia = 7.0327 MPa (CNSNS-TR-25), constante en los aumentos de potencia (CPPU); vapor del domo saturado seco (x = 1).
- 6 calentadores cerrados con extracciones por la regla de elevación igual de temperatura entre 41.0 °C (promedio de las descargas publicadas + TTD) y 286.1 °C: SIMPLIFICACIÓN DECLARADA (la CNLV no publica su arreglo); ver cycle.sensitivity_n_heaters.
- Presión de cruce = extracción del calentador N-1; recalentador con vapor vivo hasta la saturación del vapor vivo (TTD 0 K); calentadores y condensador con TTD 0 K; una sola bomba isentrópica condensador → domo; sin pérdidas de presión en líneas: límites termodinámicos declarados, cambiables en overrides.
- Agua de mar a 26.50 °C (promedio de las temporadas publicadas 25 y 28 °C), 31.5 m3/s por unidad (63 m3/s publicados para la central ÷ 2), salinidad 33 g/kg.
- Una sola eficiencia isentrópica para todas las secciones de turbina, CALIBRADA a la bruta de PRIS.
- ⚠ El precio de la energía (MXN) se pasa a USD con el FIX 18.5525 MXN/USD del 2026-11-13, no con el promedio del año 2025: da prices['fx_mxn_usd'] si lo tienes.
- ⚠ Ambas unidades comparten diseño; el diagrama muestra la unidad 1 como representativa.
- No incluye: Neutrónica, termohidráulica del núcleo, recirculación, seguridad nuclear y protección física: fuera de alcance; sólo ingeniería general del ciclo de vapor y economía con datos públicos.
- No incluye: Línea de expansión real de la turbina (eficiencia por etapa, humedad, Baumann), purgas del separador interno y del sistema de limpieza del agua del reactor, sellos y fugas.
- No incluye: Servicios propios detallados: sólo la diferencia bruta − neta publicada por PRIS.
- No incluye: Costo nivelado de la central existente: la inversión original y su financiamiento no son públicos.
- No incluye: Ingresos por potencia, certificados de energías limpias o contratos: sólo energía al PND de la zona Veracruz.
- No encontrado en fuentes abiertas: Presión, temperatura y flujo de vapor principal en la entrada de la turbina después del EPU (sólo la presión de domo 1020 psia es pública).
- No encontrado en fuentes abiertas: Número total de calentadores de agua de alimentación, presiones de extracción y temperatura final del agua de alimentación.
- No encontrado en fuentes abiertas: Presión de cruce HP→LP y temperatura de recalentamiento de los MSR.
- No encontrado en fuentes abiertas: Presión del condensador, diferencia terminal y temperatura de diseño del agua de mar del condensador nuevo (EPU).
- No encontrado en fuentes abiertas: Eficiencias de turbina, generador y bombas; potencia de servicios propios (sólo bruta − neta de PRIS).
- No encontrado en fuentes abiertas: Enriquecimiento, quemado y duración de ciclo publicados por CFE.
- No encontrado en fuentes abiertas: Costos propios de la CNLV (combustible, O&M) en un informe anual abierto de CFE; COPAR vigente (sólo se abrió la edición 2012, para centrales nuevas).
- No encontrado en fuentes abiertas: IEA/NEA «Projected Costs of Generating Electricity 2020»: el comunicado abierto no da cifras de LTO; el informe no se abrió.
- No encontrado en fuentes abiertas: PML del nodo de la central (se usa el PND de la zona Veracruz).
- No encontrado en fuentes abiertas: Mediana del factor de capacidad de la flota BWR (PRIS no legible sin navegador).
