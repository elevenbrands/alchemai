# Diagnóstico de vapor — Fábrica de alimentos, Valle de México (tesina UNAM 2021)

> **CASO REAL PUBLICADO: mediciones de una caldera real tomadas de una tesina de la UNAM (E. L. Martínez Obregón, 2021). No es un cliente. Tres datos no vienen en la tesina y se capturaron para poder calcular: la temperatura del gas (se tomó la del aire), la del agua de repuesto (20 °C) y la purga de 2 %, que la tesina supone. El precio por GJ se leyó como de poder calorífico superior.**

Fecha del cálculo: 2026-11-14. Moneda: MXN. Todas las cifras de este documento salen de dos motores de cálculo comprobados y de los datos que entregó la planta; al final se dice de dónde viene cada una.

## Resumen

Producir vapor le cuesta hoy a la planta **11,187,945 MXN al año** (381.7 MXN por tonelada de vapor entregado), y 100 % de ese costo es combustible. La caldera convierte en vapor **81.6 %** de la energía del combustible que se paga (base PCS); el resto se pierde, sobre todo por la chimenea. Las 2 medidas evaluadas, aplicadas juntas, ahorrarían **501,584 MXN al año** (4.5 % del costo del vapor), que son 4,095 GJ de combustible al año.

> **Aviso: la pérdida por paredes de la caldera es un SUPUESTO tomado de una publicación, no un dato de la planta.** La planta declaró que no la conoce. Se usó 0.9 % de la energía del combustible (base PCS), de Cleaver-Brooks, «Boiler Efficiency Guide», CB-7767, 2010, Tabla 1. Si la pérdida real fuera un punto porcentual mayor, la eficiencia bajaría 1.00 puntos y el vapor costaría 1.24 % más. La eficiencia y el costo de este reporte dejan de ser supuestos cuando la planta entregue ese dato (hoja del fabricante o prueba).

> **Aviso: el costo del agua no está incluido.** La planta no dio lo que paga por el agua de repuesto. El costo del vapor y el ahorro de las medidas que ahorran agua (bajar la purga, regresar condensado) quedan por debajo de lo real.

> **Aviso: la composición del gas es un SUPUESTO, no un análisis de la planta.** La planta declaró que no la tiene. Se usó gas natural de referencia «GULF-COAST» (mezcla predefinida «GulfCoast.mix» de CoolProp 8.0.0 (gases naturales de referencia de NIST; composición leída de la librería, no transcrita)), porque la planta no dio composición ni poder calorífico. Con las otras composiciones probadas la eficiencia cambia como mucho 0.26 puntos. Con el poder calorífico del gas (o la energía y el volumen del recibo) la composición se ajusta a la planta.

## 1. Cómo opera hoy la caldera

| Concepto | Valor | Origen |
|---|---|---|
| Presión de operación | 7.05 bar absolutos (6.4 kg/cm² manométricos); el vapor sale a 165 °C | dato de la planta |
| Agua de alimentación que entra a la caldera | 55 °C | dato de la planta |
| Oxígeno en la chimenea (base seca) y exceso de aire | 2.7 % de O2; 13 % de aire de más | dato de la planta / calculado |
| Temperatura de chimenea | 211 °C | dato de la planta |
| Eficiencia de la caldera (base PCS) | **81.6 %** | calculado |
| Combustible por tonelada de vapor (base PCS) | 3.117 GJ | calculado |
| Agua de repuesto por tonelada de vapor | 1.022 m³ | calculado |
| Costo del vapor | **381.7 MXN por tonelada** (184.9 MXN por GJ de calor entregado) | calculado |
|   de ese costo: combustible / agua / electricidad | 381.7 / 0.0 / no evaluada MXN por tonelada | calculado |
| Horas de operación | 8,760 h al año | dato de la planta |
| Vapor generado por la caldera | 29,309 t al año (3.35 t/h en promedio) | dato de la planta |
| Vapor entregado al proceso | 29,309 t al año | calculado |
| Combustible al año (base PCS) | 91,345 GJ (≈ 2,252,289 m³ normales a 0 °C y 1 atm) | calculado |
| Agua de repuesto al año | 29,961 m³ | calculado |
| Electricidad de la caldera al año | no evaluada: falta el consumo eléctrico por tonelada | calculado |
| **Costo anual del vapor** | **11,187,945 MXN** (combustible 11,187,945; agua 0; electricidad 0) | calculado |
| Emisiones directas de CO2 de la caldera | 4,584 toneladas al año; 156.4 kg por tonelada de vapor | calculado |

Se dieron la producción de vapor y el consumo de combustible: las cifras anuales usan la producción de vapor y el otro dato sirve de comparación (ver «Comprobaciones»).

La caldera trabaja en promedio a 36 % de su capacidad nominal.

La presión atmosférica del sitio (77.1 kPa) se calculó de la altitud (2,250 m); con ella se pasaron a absolutas las presiones de manómetro.

«Por tonelada» siempre es por tonelada de vapor entregado al proceso. La caldera genera algo más, porque una parte del vapor se usa para calentar el agua de alimentación.

Precios usados: combustible 122.48 MXN/GJ (PCS); agua no se dio su precio (el costo del vapor NO incluye el agua).

## 2. A dónde se va la energía del combustible

De cada 100 unidades de energía del combustible (base PCS):

| Destino | Unidades de cada 100 | Qué es |
|---|---|---|
| Se va al vapor que sale de la caldera | 81.3 | es lo que se aprovecha |
| Se va con la purga | 0.3 | agua caliente que se tira para controlar las sales |
| Chimenea: gases secos calientes | 6.2 | el aire que sobra y los gases salen calientes y se llevan calor |
| Chimenea: vapor de agua que se forma al quemar (calor latente) | 9.8 | el hidrógeno del combustible se vuelve vapor y ese vapor no se condensa en la caldera |
| Chimenea: ese mismo vapor de agua, por salir caliente | 1.4 | calor sensible del vapor formado |
| Paredes de la caldera (radiación y convección) | 0.9 | calor que pierde la envolvente al cuarto de calderas. **Valor supuesto de una publicación, no medido en la planta** |
| Calor que ya trae el combustible sobre 25 °C | 0.0 | entra con el aire o el combustible; se resta de las pérdidas |
| Calor que ya trae el aire sobre 25 °C | 0.1 | entra con el aire o el combustible; se resta de las pérdidas |

El vapor y la purga suman la eficiencia de la caldera (81.6 %). El punto de rocío del agua en los gases es 51 °C: la chimenea no debe bajar de ahí.

## 3. Medidas de ahorro

Ordenadas de mayor a menor ahorro. Cada una se calculó repitiendo el estudio completo con un solo cambio: el objetivo que dio la planta. Ahorros al año; la demanda de vapor del proceso es la misma en todos los casos.

| Medida | Qué cambia | Combustible ahorrado (GJ PCS) | Agua ahorrada (m³) | CO2 evitado (t) | Ahorro (MXN) | % del costo del vapor | Inversión | Retorno simple | Inversión máxima para recuperar en 2 años |
|---|---|---|---|---|---|---|---|---|---|
| Bajar la temperatura de chimenea (economizador o precalentador de aire) | temperatura de chimenea: 211 °C → 121 °C | 3,993 | 0 | 200 | **489,114** | 4.4 % | no dada | sin cotización | 978,229 MXN |
| Bajar el exceso de aire (afinar la combustión) | oxígeno en chimenea (base seca): 2.7 % → 2.1 %; exceso de aire: 13.2 % → 10.0 % | 214 | 0 | 11 | **26,178** | 0.2 % | no dada | sin cotización | 52,356 MXN |

La última columna no es una cotización: es dos veces el ahorro anual, es decir, lo más que se puede pagar por la medida para recuperarlo en dos años. Sirve para saber si vale la pena pedir la cotización. El retorno simple solo se calcula con la inversión que cotice el proveedor: nunca se estima.

**Bajar la temperatura de chimenea (economizador o precalentador de aire).** Eficiencia de la caldera 81.6 % → 85.3 %; costo del vapor 381.7 → 365.0 MXN por tonelada.
- La chimenea quedaría 70 °C por encima del punto de rocío del agua (51 °C).
- El agua saldría del economizador a unos 81 °C.
- El objetivo (chimenea a 250 °F (121 °C)) NO lo dio la planta ni un proveedor: es de U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012 (la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar condensación y corrosión); es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el proveedor del equipo dirá a qué temperatura llega de verdad.
- El punto de rocío ÁCIDO no se modela: con azufre en el combustible la corrosión empieza a una temperatura más alta que el punto de rocío del agua. El límite real lo fija el proveedor del equipo.
- La caldera y el equipo de recuperación se tratan como una sola envolvente: el agua de alimentación y el aire se miden a la entrada de esa envolvente y la chimenea a su salida.
- La temperatura del agua a la salida del economizador supone que los gases siguen saliendo de la caldera a la temperatura de chimenea de hoy.

**Bajar el exceso de aire (afinar la combustión).** Eficiencia de la caldera 81.6 % → 81.8 %; costo del vapor 381.7 → 380.8 MXN por tonelada.
- El objetivo (10 % de exceso de aire) NO lo dio la planta ni un proveedor: es de U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012 (en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %); es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra sin producir monóxido de carbono lo decide una prueba de combustión.
- La temperatura de chimenea se dejó igual que hoy. Si al afinar el quemador también baja, el ahorro será mayor: da la temperatura de chimenea después de afinar.

## 4. Todas las medidas juntas

|  | Ahorro en dinero | Ahorro de combustible (PCS) |
|---|---|---|
| Suma de las medidas una por una (no es alcanzable) | 515,293 MXN al año | 4,207 GJ al año |
| **Caso combinado (una sola corrida con todos los cambios)** | **501,584 MXN al año** | 4,095 GJ al año |

El caso combinado da 97 % de la suma. Los ahorros no se suman: cada medida deja menos que ahorrar a las demás (menos gases que enfriar, menos purga de la que recuperar calor, menos combustible sobre el que aplicar la mejora de eficiencia). El caso combinado es una sola corrida con todos los cambios.

Con todas las medidas la eficiencia de la caldera pasaría de 81.6 % a 85.4 % y el vapor costaría 364.6 MXN por tonelada en lugar de 381.7. Retorno simple del conjunto: no calculado: falta la inversión de al menos una medida o las cifras anuales.

Con todas las medidas se dejarían de emitir 206 toneladas de CO2 al año. Para recuperar la inversión en dos años, el conjunto no debería costar más de 1,003,169 MXN (en tres años, 1,504,753 MXN).

## 5. Lo que no se evaluó y por qué

- **Bajar la purga**: no evaluada: falta el dato. Hace falta: la purga que permite el tratamiento de agua, como porcentaje del agua de alimentación. La da el proveedor del tratamiento según los sólidos disueltos.
- **Recuperar el calor de la purga**: no evaluada: falta el dato. Hace falta: la presión del tanque de flasheo y/o la temperatura a la que saldría la purga del intercambiador. Las da el proveedor del equipo.
- **Regresar más condensado**: no evaluada: falta el dato. Hace falta: el porcentaje del vapor que se podría regresar como condensado después de las obras. Sale de un levantamiento de la red de condensado.

Datos que faltaron y qué impide cada uno:

- la pérdida por paredes y radiación de la caldera (hoja de datos del fabricante o prueba): impide dar la eficiencia de la caldera y el costo del vapor como medidos: llevan un valor supuesto de una publicación (0.90 %, base PCS).
- la humedad relativa del aire que entra al quemador: impide contar la humedad del aire: se calculó con aire seco (el efecto máximo se indica en los avisos).
- el análisis de composición del gas (lo da el proveedor de gas o el transportista): impide dar los gases de combustión y el CO2 como medidos: salen de una composición supuesta (un gas de referencia publicado).
- la inversión de «Bajar el exceso de aire (afinar la combustión)» (cotización del proveedor, en MXN): impide el retorno simple de «Bajar el exceso de aire (afinar la combustión)».
- la inversión de «Bajar la temperatura de chimenea (economizador o precalentador de aire)» (cotización del proveedor, en MXN): impide el retorno simple de «Bajar la temperatura de chimenea (economizador o precalentador de aire)».
- el recorrido de la distribución: fugas de vapor, trampas falladas y tramos de tubería sin aislar (cuántos, diámetro del orificio o del tubo, longitud y presión): impide valorar lo que se pierde entre la caldera y el proceso; este estudio cubre solo la casa de calderas.
- lo que se paga por el agua de repuesto (precio por m³, o importe y volumen del recibo; con el tratamiento si se quiere contar): impide contar el agua en el costo del vapor y en el ahorro de las medidas: salen solo con combustible y electricidad.
- el consumo eléctrico de ventiladores y bombas de la caldera (kWh por tonelada de vapor) y el precio de la electricidad: impide cobrar la electricidad de la caldera: el costo del vapor no la incluye.

## 6. De dónde sale cada número

Tres orígenes posibles: **dato de la planta** (lo entregó la planta; no se comprobó en sitio), **calculado** (lo produjo una herramienta a partir de esos datos) y **referencia publicada** (un precio o un valor de una fuente pública con fecha, usado solo porque se pidió o porque la planta declaró que no tiene el dato).

Datos de la planta:

| Dato | Valor |
|---|---|
| Altitud del sitio | 2,250 m sobre el nivel del mar |
| Temperatura del combustible en el quemador | 22.5 °C |
| Temperatura del aire de combustión | 22.5 °C |
| Oxígeno en chimenea (base seca) | 2.70 % |
| Temperatura de chimenea | 211.0 °C |
| Presión de la caldera | 7.050 bar absolutos |
| Sobrecalentamiento | 0 °C de diferencia |
| Purga (del agua de alimentación) | 2.00 % |
| Agua de alimentación | su calentamiento no se cobra al combustible |
| Temperatura del agua de repuesto | 20.0 °C |
| Retorno de condensado (del vapor) | 0.00 % |
| Monóxido de carbono en chimenea (base seca) | 0 ppm |
| Temperatura del agua de alimentación | 55.0 °C |
| Capacidad nominal de la caldera | 9.39 t/h |
| Horas de operación al año | 8,760 h al año |
| Producción de vapor de la caldera | 3.35 t/h |
| Consumo de combustible al año | 92,046 GJ al año (PCS) |

Valores supuestos (no son datos de la planta):

| Dato | Valor | Origen | Detalle |
|---|---|---|---|
| Composición del gas natural | CH4 96.52 %, N2 0.26 %, CO2 0.60 %, C2H6 1.82 %, C3H8 0.46 %, i-C4H10 0.10 %, n-C4H10 0.10 %, i-C5H12 0.05 %, n-C5H12 0.10 % (mol) | referencia publicada | gas natural de referencia «GULF-COAST» (mezcla predefinida «GulfCoast.mix» de CoolProp 8.0.0 (gases naturales de referencia de NIST; composición leída de la librería, no transcrita)), porque la planta no dio composición ni poder calorífico |
| Objetivo de la medida «Bajar el exceso de aire (afinar la combustión)» | 10 % de exceso de aire | referencia publicada | en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %. U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012. Consultada el 2026-11-14. Límite: es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra sin producir monóxido de carbono lo decide una prueba de combustión. https://www.energy.gov/sites/prod/files/2014/05/f16/steam4_boiler_efficiency.pdf |
| Objetivo de la medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)» | chimenea a 250 °F (121 °C) | referencia publicada | la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar condensación y corrosión. U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012. Consultada el 2026-11-14. Límite: es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el proveedor del equipo dirá a qué temperatura llega de verdad. https://www.energy.gov/sites/prod/files/2014/05/f16/steam3_recovery.pdf |
| Pérdida por paredes y radiación de la caldera | 0.90 % del poder calorífico (base PCS) | referencia publicada | pérdidas por radiación y convección esperadas en calderas de tubos de humo de cuatro pasos, por tamaño, presión de operación y porcentaje de carga; se lee la tabla para una caldera de 600 caballos caldera (clase 400-800), 91 psig y 36 % de carga, interpolando entre sus renglones: 0.90 %. Cleaver-Brooks, «Boiler Efficiency Guide», CB-7767, 2010, Tabla 1. Consultada el 2026-11-14. Límites: es la pérdida que el fabricante espera de su modelo de cuatro pasos bien aislado, en aire quieto a 27 °C: una caldera vieja, de otro diseño o con el aislamiento dañado pierde más; la carga usada es el promedio declarado; si la caldera pasa muchas horas a muy baja carga la pérdida real es mayor. |

Precios:

| Precio | Valor | Origen | Detalle |
|---|---|---|---|
| Precio del combustible | 122.4796 MXN/GJ (PCS) | dato de la planta | precio dado por la planta |
| Precio del agua de repuesto | 0.0000 MXN/m³ | no dado | la planta no dio el precio del agua: el costo del vapor y los ahorros NO incluyen el agua |

Cifras calculadas:

| Cifra | Herramienta | A partir de |
|---|---|---|
| Presión atmosférica del sitio (77.1 kPa) | atmósfera estándar de 1976 (librería fluids) | la altitud del sitio |
| Poder calorífico, aire y gases de combustión, punto de rocío del agua | combustion_efficiency (M18, método de pérdidas, entalpías NASA) | combustible, aire, O2 o exceso de aire |
| Eficiencia de la caldera y desglose de pérdidas | combustion_efficiency (M18, método de pérdidas, entalpías NASA) | lo anterior + temperatura de chimenea + pérdida por paredes |
| Combustible, agua de repuesto y costo por tonelada y por GJ de vapor | utility_costs (M17, balance de la isla de vapor con IAPWS-95) | eficiencia (de la herramienta anterior), presión, purga, retorno de condensado, temperaturas y precios |
| Cifras anuales | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | cifras por tonelada × vapor entregado al año. Se dieron la producción de vapor y el consumo de combustible: las cifras anuales usan la producción de vapor y el otro dato sirve de comparación (ver «Comprobaciones»). |
| Ahorro de cada medida y del caso combinado | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | línea base − la misma corrida de los dos motores con el objetivo de la planta |
| Retorno simple | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | inversión (dato de la planta) ÷ ahorro anual; solo si se dio la inversión |

## 7. Supuestos y límites

Esta sección es técnica: está escrita para quien revise el cálculo.

Supuestos del cálculo:

- Precio del combustible en base PCS: las eficiencias de caldera y horno deben estar en esa misma base (con PCI la eficiencia es numéricamente mayor que con PCS).
- Cada nivel se calcula como generado directamente a su presión en una caldera, sin cogeneración ni turbinas de contrapresión (con cogeneración el vapor de baja cuesta menos).
- Calor entregado al proceso = entalpía del vapor − líquido saturado a la misma presión (no se acredita el subenfriamiento del condensado ni el vapor de flasheo).
- Agua y vapor: IAPWS-95 (CoolProp). Las corrientes líquidas se evalúan en la línea de saturación a su temperatura; el trabajo de la bomba de alimentación no se cobra aparte.
- feedwater_heating='not_charged': el combustible solo lleva el agua de feedwater_T_K a vapor (costo de generación tipo DOE Steam Tip Sheet #15). Calentar el agua hasta la temperatura de alimentación no se cobra: es válido si ese calor es recuperado; si no, el costo real es mayor.
- Química supuesta: C → CO2 (o el CO medido), H → H2O, S → SO2, N del combustible → N2 (sin NOx), ceniza inerte, sin carbono sin quemar ni hollín.
- Aire seco: O2 0.2100, N2 0.7800, Ar 0.0100 (composición de air.yaml de Cantera).
- Referencia de entalpía 298.15 K con el agua como vapor; gas ideal. Polinomios NASA Glenn de 9 coeficientes distribuidos con Cantera 3.2.0.
- La purga se descarga como líquido saturado a la presión de la caldera, sin recuperar su calor.
- La humedad del aire de combustión NO es un dato de la planta: se calculó con aire seco.
- La presión atmosférica del sitio (77.1 kPa) no es una lectura de barómetro: se calculó de la altitud (2250 m) con la atmósfera estándar de 1976. Con ella se pasan a absolutas las presiones de manómetro.
- La composición del gas NO es un dato de la planta: es gas natural de referencia «GULF-COAST» (mezcla predefinida «GulfCoast.mix» de CoolProp 8.0.0 (gases naturales de referencia de NIST; composición leída de la librería, no transcrita)), porque la planta no dio composición ni poder calorífico. Nota: n-hexano (0.066 % mol) contado como n-pentano.
- La pérdida por paredes y radiación NO es un dato de la planta: es 0.90 % del poder calorífico (base PCS), tomado de Cleaver-Brooks, «Boiler Efficiency Guide», CB-7767, 2010, Tabla 1 (se lee la tabla para una caldera de 600 caballos caldera (clase 400-800), 91 psig y 36 % de carga, interpolando entre sus renglones: 0.90 %). Límites: es la pérdida que el fabricante espera de su modelo de cuatro pasos bien aislado, en aire quieto a 27 °C: una caldera vieja, de otro diseño o con el aislamiento dañado pierde más; la carga usada es el promedio declarado; si la caldera pasa muchas horas a muy baja carga la pérdida real es mayor.
- La demanda de vapor del proceso (vapor entregado) se mantiene igual en todas las medidas; lo que cambia es el combustible, el agua y la electricidad para producirlo.
- La pérdida por paredes se mantiene como la misma fracción del poder calorífico en todos los casos. En realidad es casi constante en kW, así que al bajar el combustible la fracción sube un poco: el ahorro queda ligeramente sobrestimado.
- Operación en un solo punto promedio: una carga, un O2 y una temperatura de chimenea representan todo el año.
- Los ahorros se valoran con los mismos precios de la línea base.

No está incluido en estas cifras:

- Capital, mano de obra y mantenimiento de la caldera
- Pérdidas de distribución y trampas de vapor
- Tratamiento químico (inclúyelo en el precio del agua de repuesto si lo quieres cobrar)
- Electricidad de ventiladores y bombas de la caldera (da auxiliary_power_kWh_per_tonne para cobrarla)
- combustión: carbono sin quemar en ceniza, hollín y NOx
- combustión: calor sensible de la ceniza
- combustión: vapor de atomización y sopladores de hollín
- combustión: punto de rocío ácido
- combustión: calderas de condensación (chimenea bajo el punto de rocío)
- combustión: purga de caldera (se cuenta en el balance de vapor)
- Inversión de las medidas: nunca se estima; el retorno solo se calcula con la cifra de la planta
- Variación de la carga a lo largo del año, arranques y paros, y eficiencia a carga parcial
- Fugas de vapor, trampas de vapor y aislamiento de tuberías (no se dio el recorrido de la distribución)
- Química del agua: la purga mínima y la calidad del condensado no se comprueban
- Punto de rocío ácido y corrosión de la zona fría
- Emisiones distintas del CO2 directo de la caldera (NOx, metano) y el costo de emitir
- Impuestos, financiamiento y mantenimiento de los equipos nuevos

Avisos del cálculo:

- OBJETIVO PUBLICADO, NO DEL PROVEEDOR: la medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)» se evaluó con chimenea a 250 °F (121 °C), de U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012 (la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar condensación y corrosión); es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el proveedor del equipo dirá a qué temperatura llega de verdad.
- OBJETIVO PUBLICADO, NO DEL PROVEEDOR: la medida «Bajar el exceso de aire (afinar la combustión)» se evaluó con 10 % de exceso de aire, de U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012 (en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %); es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra sin producir monóxido de carbono lo decide una prueba de combustión.
- PÉRDIDA POR PAREDES SUPUESTA, NO MEDIDA: la planta declaró que no conoce la pérdida por paredes y radiación de su caldera. Se usó 0.90 % del poder calorífico (base PCS), de Cleaver-Brooks, «Boiler Efficiency Guide», CB-7767, 2010, Tabla 1: pérdidas por radiación y convección esperadas en calderas de tubos de humo de cuatro pasos, por tamaño, presión de operación y porcentaje de carga; se lee la tabla para una caldera de 600 caballos caldera (clase 400-800), 91 psig y 36 % de carga, interpolando entre sus renglones: 0.90 %. Con un punto porcentual más de pérdida la eficiencia baja 1.00 puntos y el vapor cuesta +1.24 %. Consultada el 2026-11-14; publicación CB-7767 de Cleaver-Brooks (leída en una copia del PDF).
- HUMEDAD DEL AIRE NO DADA: se calculó con aire seco. Con aire saturado a su temperatura la eficiencia cambia -0.29 puntos y el costo del vapor +0.35 %; el valor real queda entre los dos.
- COMPOSICIÓN DEL GAS SUPUESTA, NO ANALIZADA: la planta declaró que no tiene el análisis de su gas. Se usó gas natural de referencia «GULF-COAST» (mezcla predefinida «GulfCoast.mix» de CoolProp 8.0.0 (gases naturales de referencia de NIST; composición leída de la librería, no transcrita)), porque la planta no dio composición ni poder calorífico. Con las composiciones alternativas probadas la eficiencia cambia como mucho 0.26 puntos y el costo del vapor 0.31 %. Pide a tu proveedor el poder calorífico del gas o da energía y volumen del recibo para ajustar la composición.
- COSTO DEL AGUA NO INCLUIDO: la planta no dio lo que paga por el agua de repuesto. El costo del vapor y el ahorro de las medidas que ahorran agua (purga, condensado) quedan por debajo de lo real.
- Se eligió no cobrar el calentamiento del agua: calentar el agua hasta la temperatura de alimentación no se cobra al combustible. Si ese calor sale de vapor de la propia caldera, el costo real del vapor es mayor que el mostrado.

## 8. Comprobaciones

- Balances de masa y energía de los dos motores, en la línea base y en cada medida: cierran (peor residuo 4.8e-11 de la energía del combustible).
- Calor que recibe el vapor (tablas de vapor IAPWS-95) contra eficiencia × combustible: diferencia 0.0e+00.
- Combustible anual por dos rutas (lado del agua y lado de la combustión): diferencia 3.2e-16.
- El caso combinado no ahorra más que la suma de las medidas: se cumple.
- Se dieron producción de vapor y consumo de combustible: la eficiencia por entradas y salidas es 81.0 % y la del método de pérdidas 81.6 % (diferencia -0.6 puntos). Eficiencia por entradas y salidas (calor al vapor medido ÷ combustible medido) contra la del método de pérdidas. Una diferencia grande indica un medidor descalibrado, una purga distinta de la declarada, combustible que se usa en otro equipo o vapor que no se mide.
- «Comprobado» significa que los balances de masa y energía de los dos motores cierran en todos los casos y las rutas independientes coinciden. NO significa que los datos de la planta sean correctos.

## 9. Datos tal como los capturó la planta

Copia de la hoja de datos, en las unidades en que se escribió. El resto del reporte usa unidades métricas.

| Dato | Valor | Unidad |
|---|---|---|
| Moneda de todos los importes | MXN |  |
| Altitud de la planta sobre el nivel del mar | 2250 | m |
| ¿Qué combustible quema la caldera? | Gas natural |  |
| Temperatura del combustible en el quemador | 22.52 | °C |
| Combustible: precio | 122.48 | por GJ |
| Combustible — ¿por energía superior o inferior? | Superior (PCS) |  |
| Presión de operación | 7.05 | bar abs. |
| Sobrecalentamiento (0 si el vapor es saturado) | 0 | °C |
| Purga, del agua de alimentación | 2 | % |
| Capacidad nominal (placa) | 600 | CC (caballos caldera) |
| Tipo de caldera | Tubos de humo (pirotubular) |  |
| Oxígeno en la chimenea | 2.7 | % |
| Oxígeno en la chimenea — ¿base seca o húmeda? | Seca |  |
| Monóxido de carbono en chimenea, base seca (opcional) | 0 | ppm |
| Temperatura de los gases en la chimenea | 211 | °C |
| Temperatura del aire que entra al quemador | 22.52 | °C |
| Pérdida por paredes y radiación de la caldera — No la conozco | Sí |  |
| ¿Cómo se calienta el agua de alimentación? | No cobrar el calentamiento del agua al combustible |  |
| Temperatura del agua que entra a la caldera | 55 | °C |
| Temperatura del agua de repuesto | 20 | °C |
| Condensado que regresa, del vapor entregado | 0 | % |
| Horas de operación al año | 8760 | h al año |
| Producción promedio de vapor de la caldera | 3345.81 | kg/h |
| Consumo anual de combustible, en energía | 92046.3 | GJ |
| Consumo anual de combustible, en energía — ¿superior o inferior? | Superior (PCS) |  |
| Afinar la combustión — No tengo objetivo: usar el publicado (10 % de exceso de aire, solo gas natural) | Sí |  |
| Economizador o precalentador — No tengo objetivo: usar el publicado (121 °C, solo gas natural) | Sí |  |

## Glosario

- **PCS y PCI**: poder calorífico superior e inferior (en inglés HHV y LHV): la energía de un combustible contando, o sin contar, el calor del vapor de agua que se forma al quemarlo. Una eficiencia en base PCS es numéricamente menor que la misma en base PCI; aquí se usa la base en la que se paga el combustible.
- **Exceso de aire**: el aire que entra de más respecto al estrictamente necesario para quemar. Se mide con el oxígeno (O2) que queda en la chimenea. Hace falta algo; lo que sobra se calienta y se tira.
- **Purga**: agua que se saca de la caldera para que no se concentren las sales. Sale a la temperatura de la caldera.
- **Condensado**: el vapor ya usado, convertido en agua caliente y limpia. Regresarlo ahorra agua y combustible.
- **Agua de repuesto**: agua nueva tratada que repone la purga y el condensado que no regresa.
- **Punto de rocío**: temperatura bajo la cual el vapor de agua de los gases se condensa. Con azufre, el ácido condensa a una temperatura más alta (punto de rocío ácido), que este estudio no calcula.
- **Retorno simple**: años en que el ahorro paga la inversión: inversión ÷ ahorro anual, sin intereses ni impuestos.
- **Vapor entregado y vapor generado**: el generado es todo el que sale de la caldera; el entregado es el que llega al proceso después de descontar el que se usa para calentar el agua de alimentación.
