# Diagnóstico de vapor — Planta de ejemplo (datos inventados)

> **CASO ILUSTRATIVO: los datos de esta planta son INVENTADOS para mostrar el formato. No es un cliente ni una validación.**

Fecha del cálculo: 2026-11-14. Moneda: MXN. Todas las cifras de este documento salen de dos motores de cálculo comprobados y de los datos que entregó la planta; al final se dice de dónde viene cada una.

## Resumen

Producir vapor le cuesta hoy a la planta **17,655,704 MXN al año** (551.6 MXN por tonelada de vapor entregado), y 92 % de ese costo es combustible. La caldera convierte en vapor **78.7 %** de la energía del combustible que se paga (base PCS); el resto se pierde, sobre todo por la chimenea. Las 5 medidas evaluadas, aplicadas juntas, ahorrarían **2,458,669 MXN al año** (13.9 % del costo del vapor), que son 13,036 GJ de combustible al año.

> **Aviso: la pérdida por paredes de la caldera es un SUPUESTO tomado de una publicación, no un dato de la planta.** La planta declaró que no la conoce. Se usó 1.6 % de la energía del combustible (base PCS), de G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), mayo 2002, sección 4.2.5 «Shell Loss». Si la pérdida real fuera un punto porcentual mayor, la eficiencia bajaría 1.00 puntos y el vapor costaría 1.18 % más. La eficiencia y el costo de este reporte dejan de ser supuestos cuando la planta entregue ese dato (hoja del fabricante o prueba).

> **Aviso: la composición del gas es un SUPUESTO, no un análisis de la planta.** La planta declaró que no la tiene. Se usó mezcla equivalente de metano + etano ajustada para dar exactamente el poder calorífico medido (952.1 kJ/mol, base PCS; recibo del periodo «octubre 2026»: 9,500.00 GJ ÷ 248,000.0 m³ a 293.15 K y 98066.5 Pa (gas ideal) = 38.306 MJ/m³). Con las otras composiciones probadas la eficiencia cambia como mucho 0.22 puntos. Comprobado contra seis gases naturales de referencia con composición completa: el gas equivalente cambia la eficiencia menos de 0.15 puntos (base PCS) y da el CO2 entre 0.9 y 3.2 % más bajo (9 % más bajo en un gas con 7.6 % de CO2), porque no cuenta el CO2 que el gas ya trae.

## 1. Cómo opera hoy la caldera

| Concepto | Valor | Origen |
|---|---|---|
| Presión de operación | 8.80 bar absolutos (8.0 kg/cm² manométricos); el vapor sale a 174 °C | dato de la planta |
| Agua de alimentación que entra a la caldera | 105 °C | dato de la planta |
| Oxígeno en la chimenea (base seca) y exceso de aire | 7.0 % de O2; 45 % de aire de más | calculado / dato de la planta |
| Temperatura de chimenea | 220 °C | dato de la planta |
| Eficiencia de la caldera (base PCS) | **78.7 %** | calculado |
| Combustible por tonelada de vapor (base PCS) | 3.372 GJ | calculado |
| Agua de repuesto por tonelada de vapor | 0.699 m³ | calculado |
| Costo del vapor | **551.6 MXN por tonelada** (271.2 MXN por GJ de calor entregado) | calculado |
|   de ese costo: combustible / agua / electricidad | 505.7 / 31.5 / 14.4 MXN por tonelada | calculado |
| Horas de operación | 6,000 h al año | dato de la planta |
| Vapor generado por la caldera | 36,000 t al año (6.00 t/h en promedio) | dato de la planta |
| Vapor entregado al proceso | 32,009 t al año | calculado |
| Combustible al año (base PCS) | 107,919 GJ (≈ 2,540,633 m³ normales a 0 °C y 1 atm) | calculado |
| Agua de repuesto al año | 22,376 m³ | calculado |
| Electricidad de la caldera al año | 192,052 kWh | calculado |
| **Costo anual del vapor** | **17,655,704 MXN** (combustible 16,187,871; agua 1,006,908; electricidad 460,925) | calculado |
| Emisiones directas de CO2 de la caldera | 5,446 toneladas al año; 170.2 kg por tonelada de vapor | calculado |

El consumo de combustible se calculó desde la producción de vapor y la eficiencia de la caldera.

Precios usados: combustible 150.00 MXN/GJ (PCS); agua 45.00 MXN/m³; electricidad 2.400 MXN/kWh.

## 2. A dónde se va la energía del combustible

De cada 100 unidades de energía del combustible (base PCS):

| Destino | Unidades de cada 100 | Qué es |
|---|---|---|
| Se va al vapor que sale de la caldera | 77.8 | es lo que se aprovecha |
| Se va con la purga | 0.9 | agua caliente que se tira para controlar las sales |
| Chimenea: gases secos calientes | 8.5 | el aire que sobra y los gases salen calientes y se llevan calor |
| Chimenea: ese mismo vapor de agua, por salir caliente | 1.5 | calor sensible del vapor formado |
| Chimenea: humedad del aire | 0.2 | agua que trae el aire de combustión |
| Paredes de la caldera (radiación y convección) | 1.6 | calor que pierde la envolvente al cuarto de calderas. **Valor supuesto de una publicación, no medido en la planta** |
| Chimenea: vapor de agua que se forma al quemar (calor latente) | 9.7 | el hidrógeno del combustible se vuelve vapor y ese vapor no se condensa en la caldera |

El vapor y la purga suman la eficiencia de la caldera (78.7 %). El punto de rocío del agua en los gases es 52 °C: la chimenea no debe bajar de ahí.

## 3. Medidas de ahorro

Ordenadas de mayor a menor ahorro. Cada una se calculó repitiendo el estudio completo con un solo cambio: el objetivo que dio la planta. Ahorros al año; la demanda de vapor del proceso es la misma en todos los casos.

| Medida | Qué cambia | Combustible ahorrado (GJ PCS) | Agua ahorrada (m³) | CO2 evitado (t) | Ahorro (MXN) | % del costo del vapor | Inversión | Retorno simple | Inversión máxima para recuperar en 2 años |
|---|---|---|---|---|---|---|---|---|---|
| Bajar la temperatura de chimenea (economizador o precalentador de aire) | temperatura de chimenea: 220 °C → 121 °C | 6,665 | 0 | 336 | **999,710** | 5.7 % | no dada | sin cotización | 1,999,420 MXN |
| Regresar más condensado | retorno de condensado, del vapor: 40.0 % → 70.0 % | 3,142 | 9,711 | 159 | **908,227** | 5.1 % | no dada | sin cotización | 1,816,453 MXN |
| Bajar el exceso de aire (afinar la combustión) | oxígeno en chimenea (base seca): 7.0 % → 2.1 %; exceso de aire: 44.9 % → 10.0 % | 2,908 | 0 | 147 | **436,164** | 2.5 % | no dada | sin cotización | 872,328 MXN |
| Recuperar el calor de la purga | calor de la purga: se tira → tanque de flasheo a 1.24 bar absolutos más intercambiador que deja la purga a 40 °C | 2,322 | 471 | 117 | **369,423** | 2.1 % | no dada | sin cotización | 738,846 MXN |
| Bajar la purga | purga, del agua de alimentación: 8.0 % → 5.0 % | 1,036 | 1,248 | 52 | **211,620** | 1.2 % | no dada | sin cotización | 423,240 MXN |

La última columna no es una cotización: es dos veces el ahorro anual, es decir, lo más que se puede pagar por la medida para recuperarlo en dos años. Sirve para saber si vale la pena pedir la cotización. El retorno simple solo se calcula con la inversión que cotice el proveedor: nunca se estima.

**Bajar la temperatura de chimenea (economizador o precalentador de aire).** Eficiencia de la caldera 78.7 % → 83.8 %; costo del vapor 551.6 → 520.4 MXN por tonelada.
- La chimenea quedaría 69 °C por encima del punto de rocío del agua (52 °C).
- El agua saldría del economizador a unos 137 °C.
- El objetivo (chimenea a 250 °F (121 °C)) NO lo dio la planta ni un proveedor: es de U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012 (la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar condensación y corrosión); es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el proveedor del equipo dirá a qué temperatura llega de verdad.
- El punto de rocío ÁCIDO no se modela: con azufre en el combustible la corrosión empieza a una temperatura más alta que el punto de rocío del agua. El límite real lo fija el proveedor del equipo.
- La caldera y el equipo de recuperación se tratan como una sola envolvente: el agua de alimentación y el aire se miden a la entrada de esa envolvente y la chimenea a su salida.
- La temperatura del agua a la salida del economizador supone que los gases siguen saliendo de la caldera a la temperatura de chimenea de hoy.

**Regresar más condensado.** Eficiencia de la caldera 78.7 % → 78.7 %; costo del vapor 551.6 → 523.2 MXN por tonelada.

**Bajar el exceso de aire (afinar la combustión).** Eficiencia de la caldera 78.7 % → 80.8 %; costo del vapor 551.6 → 538.0 MXN por tonelada.
- El objetivo (10 % de exceso de aire) NO lo dio la planta ni un proveedor: es de U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012 (en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %); es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra sin producir monóxido de carbono lo decide una prueba de combustión.
- La temperatura de chimenea se dejó igual que hoy. Si al afinar el quemador también baja, el ahorro será mayor (parámetro stack_T_K_after).

**Recuperar el calor de la purga.** Eficiencia de la caldera 78.7 % → 78.7 %; costo del vapor 551.6 → 540.1 MXN por tonelada.
- 13.1 % de la purga se vuelve vapor de flasheo a 1.24 bar absolutos; el resto calienta el agua de repuesto hasta 28 °C y se va al drenaje a 40 °C.

**Bajar la purga.** Eficiencia de la caldera 78.7 % → 78.7 %; costo del vapor 551.6 → 545.0 MXN por tonelada.
- No se comprueba la química del agua: la purga mínima la fijan los sólidos disueltos que tolera la caldera.

## 4. Todas las medidas juntas

|  | Ahorro en dinero | Ahorro de combustible (PCS) |
|---|---|---|
| Suma de las medidas una por una (no es alcanzable) | 2,925,143 MXN al año | 16,072 GJ al año |
| **Caso combinado (una sola corrida con todos los cambios)** | **2,458,669 MXN al año** | 13,036 GJ al año |

El caso combinado da 84 % de la suma. Los ahorros no se suman: cada medida deja menos que ahorrar a las demás (menos gases que enfriar, menos purga de la que recuperar calor, menos combustible sobre el que aplicar la mejora de eficiencia). El caso combinado es una sola corrida con todos los cambios.

Con todas las medidas la eficiencia de la caldera pasaría de 78.7 % a 84.9 % y el vapor costaría 474.8 MXN por tonelada en lugar de 551.6. Retorno simple del conjunto: no calculado: falta la inversión de al menos una medida o las cifras anuales.

Con todas las medidas se dejarían de emitir 658 toneladas de CO2 al año. Para recuperar la inversión en dos años, el conjunto no debería costar más de 4,917,338 MXN (en tres años, 7,376,007 MXN).

## Pérdidas en la distribución (fugas, trampas y tubería sin aislar)

Salen del recorrido de la planta: los conteos, diámetros y longitudes son datos de la planta. Cada pérdida se valora con el vapor de la línea base (551.6 MXN por tonelada).

| Pérdida | Qué es | Cuánto se pierde | Costo al año (MXN) | CO2 al año (t) |
|---|---|---|---|---|
| Fuga de vapor: cabezal principal | 4 × orificio de 3.0 mm a 8.8 bar abs | 77.8 kg/h de vapor | 257,534 | 79.4 |
| Trampa fallada abierta: línea de proceso 2 | 3 × orificio de 4.0 mm a 8.8 bar abs | 124.5 kg/h de vapor | 412,054 | 127.1 |
| Tubería sin aislar: línea a marmitas | 30 m de 114 mm de diámetro exterior a 174 °C | 24.5 kW de calor (817 W por metro) | 131,617 | 44.3 |
| Tubería sin aislar: bajante a lavado | 12 m de 60 mm de diámetro exterior a 174 °C | 5.4 kW de calor (449 W por metro) | 28,917 | 9.7 |

**En total estas pérdidas cuestan 830,122 MXN al año** y emiten 260.6 toneladas de CO2. Equivalen a 4.8 % del vapor que entrega la caldera. Repararlas todas se paga en dos años si cuesta menos de 1,660,244 MXN.

Con el aislamiento propuesto:

| Tramo | Aislamiento | Pérdida hoy → aislado | Ahorro al año (MXN) |
|---|---|---|---|
| línea a marmitas | 50 mm; la cubierta queda a 38 °C | 24.5 → 1.84 kW (92 % menos) | 121,734 |

Tramos sin aislamiento propuesto (se reporta solo lo que pierden hoy): bajante a lavado.

Valorado con la línea base. Si además se aplican las medidas de la caldera, cada tonelada de vapor cuesta menos y estas pérdidas valen proporcionalmente menos.

Constantes del modelo que no se le piden a la planta:

- Coeficiente de descarga de una fuga (orificio de borde vivo): 0.60. G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), 2002: un orificio de borde vivo deja pasar aproximadamente 60 % del flujo de una tobera bien redondeada. Con 0.60 el motor reproduce su Tabla 6 entre −0.8 % y +0.4 % (la columna de 300 psig, −4 %).
- Coeficiente de descarga de una trampa fallada abierta (orificio reentrante): 0.72. U.S. DOE, Steam Tip Sheet #1, DOE/GO-102012-3401, enero 2012: su tabla usa 0.72. Con 0.72 el motor reproduce esa tabla entre −3.8 % y +2.9 %.
- Emisividad de la superficie del tubo sin aislar: 0.70. elección de este estudio: el U.S. DOE Steam Tip Sheet #2 (enero 2012) no declara la suya; con 0.70 el motor queda entre −5 % y +8 % de su tabla de pérdidas de tubería desnuda (con 0.80, entre +2 % y +16 %).
- Emisividad de la cubierta del aislamiento: 0.90. elección conservadora de este estudio (una cubierta opaca pierde más que una de aluminio brillante, así que el ahorro no queda inflado); cambia poco el resultado porque la pérdida ya aislada es pequeña.

Supuestos de esta sección:

- El vapor de fugas y trampas falladas se valora al costo de una tonelada de vapor de la línea base (combustible, agua y electricidad), como vapor que sale del sistema. Si una trampa descarga a un retorno de condensado cerrado, el agua no se pierde y el costo real es algo menor.
- El calor que pierde la tubería sin aislar se valora solo con el combustible (costo de combustible por GJ de calor entregado), porque el condensado que se forma no sale del sistema.
- Las pérdidas se suponen constantes durante todas las horas de operación declaradas.
- Flujo por orificio con expansión en equilibrio y vapor saturado a la presión dada.

No incluye: fugas en válvulas, bridas y empaques que no se listaron; trampas que fallan cerradas (no pierden vapor; inundan equipos); válvulas, bridas y accesorios sin aislar; tubería de condensado sin aislar; el costo de reparar o aislar: sale de una cotización.

## 5. Lo que no se evaluó y por qué

Datos que faltaron y qué impide cada uno:

- la pérdida por paredes y radiación de la caldera (hoja de datos del fabricante o prueba): impide dar la eficiencia de la caldera y el costo del vapor como medidos: llevan un valor supuesto de una publicación (1.56 %, base PCS).
- el análisis de composición del gas (lo da el proveedor de gas o el transportista): impide dar los gases de combustión y el CO2 como medidos: salen de una composición supuesta (ajustada al poder calorífico del recibo).
- la inversión de «Bajar el exceso de aire (afinar la combustión)» (cotización del proveedor, en MXN): impide el retorno simple de «Bajar el exceso de aire (afinar la combustión)».
- la inversión de «Bajar la temperatura de chimenea (economizador o precalentador de aire)» (cotización del proveedor, en MXN): impide el retorno simple de «Bajar la temperatura de chimenea (economizador o precalentador de aire)».
- la inversión de «Bajar la purga» (cotización del proveedor, en MXN): impide el retorno simple de «Bajar la purga».
- la inversión de «Recuperar el calor de la purga» (cotización del proveedor, en MXN): impide el retorno simple de «Recuperar el calor de la purga».
- la inversión de «Regresar más condensado» (cotización del proveedor, en MXN): impide el retorno simple de «Regresar más condensado».
- el monóxido de carbono (CO) medido en la chimenea: impide contar la pérdida por combustión incompleta: se tomó combustión completa.

## 6. De dónde sale cada número

Tres orígenes posibles: **dato de la planta** (lo entregó la planta; no se comprobó en sitio), **calculado** (lo produjo una herramienta a partir de esos datos) y **referencia publicada** (un precio o un valor de una fuente pública con fecha, usado solo porque se pidió o porque la planta declaró que no tiene el dato).

Datos de la planta:

| Dato | Valor |
|---|---|
| Presión atmosférica del sitio | 0.950 bar absolutos |
| Temperatura del aire de combustión | 25.0 °C |
| Humedad relativa del aire | 50.00 % |
| Oxígeno en chimenea (base seca) | 7.00 % |
| Temperatura de chimenea | 220.0 °C |
| Presión de la caldera | 8.795 bar absolutos |
| Sobrecalentamiento | 0 K de diferencia |
| Purga (del agua de alimentación) | 8.00 % |
| Agua de alimentación | con vapor de la propia caldera (desaireador o tanque calentado) |
| Temperatura del agua de repuesto | 20.0 °C |
| Retorno de condensado (del vapor) | 40.00 % |
| Temperatura del agua de alimentación | 105.0 °C |
| Temperatura del condensado | 80.0 °C |
| Electricidad de ventiladores y bombas | 6 kWh por tonelada de vapor |
| Horas de operación al año | 6,000 h al año |
| Producción de vapor de la caldera | 6.00 t/h |
| Medida «Bajar la purga»: purga objetivo | 5.00 % |
| Medida «Recuperar el calor de la purga»: presión del tanque de flasheo | 1.244 bar absolutos |
| Medida «Recuperar el calor de la purga»: temperatura de la purga al drenaje | 40.0 °C |
| Medida «Regresar más condensado»: retorno de condensado objetivo | 70.00 % |

Valores supuestos (no son datos de la planta):

| Dato | Valor | Origen | Detalle |
|---|---|---|---|
| Composición del gas natural | CH4 90.82 %, C2H6 9.18 % (mol) | calculado | mezcla equivalente de metano + etano ajustada para dar exactamente el poder calorífico medido (952.1 kJ/mol, base PCS; recibo del periodo «octubre 2026»: 9,500.00 GJ ÷ 248,000.0 m³ a 293.15 K y 98066.5 Pa (gas ideal) = 38.306 MJ/m³) |
| Objetivo de la medida «Bajar el exceso de aire (afinar la combustión)» | 10 % de exceso de aire | referencia publicada | en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %. U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012. Consultada el 2026-11-14. Límite: es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra sin producir monóxido de carbono lo decide una prueba de combustión. https://www.energy.gov/sites/prod/files/2014/05/f16/steam4_boiler_efficiency.pdf |
| Objetivo de la medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)» | chimenea a 250 °F (121 °C) | referencia publicada | la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar condensación y corrosión. U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012. Consultada el 2026-11-14. Límite: es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el proveedor del equipo dirá a qué temperatura llega de verdad. https://www.energy.gov/sites/prod/files/2014/05/f16/steam3_recovery.pdf |
| Pérdida por paredes y radiación de la caldera | 1.56 % del poder calorífico (base PCS) | referencia publicada | la mayoría de las calderas de tubos de agua deberían perder menos de 1.0 % de la energía del combustible por la envolvente cerca de plena carga y sin daños en el refractario o el forro (la guía calcula en base PCS y remite a la gráfica de la ABMA); se toma el límite superior de esa afirmación, 1.0 % a plena carga, y se divide entre la carga promedio de la caldera (64 % de su capacidad nominal) porque la misma fuente dice que la pérdida casi no cambia en magnitud con la carga: 1.56 %. G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), mayo 2002, sección 4.2.5 «Shell Loss». Consultada el 2026-11-14. Límites: en calderas de tubos de humo la misma fuente espera una pérdida menor; con refractario o aislamiento dañado la pérdida real es mayor; la carga usada es el promedio declarado; si la caldera pasa muchas horas a muy baja carga la pérdida real es mayor. |

Precios:

| Precio | Valor | Origen | Detalle |
|---|---|---|---|
| Precio del combustible | 150.0000 MXN/GJ (PCS) | calculado | importe pagado 1,425,000.00 MXN ÷ energía 9,500.00 GJ (energía facturada o medida, tal cual); recibos del periodo «octubre 2026» (dato de la planta); precio medio del recibo: incluye todo lo que traiga el importe (transporte, distribución, cargos fijos, impuestos) |
| Precio del agua de repuesto | 45.0000 MXN/m³ | dato de la planta | precio dado por la planta |
| Precio de la electricidad | 2.4000 MXN/kWh | dato de la planta | precio dado por la planta |

Cifras calculadas:

| Cifra | Herramienta | A partir de |
|---|---|---|
| Poder calorífico, aire y gases de combustión, punto de rocío del agua | combustion_efficiency (M18, método de pérdidas, entalpías NASA) | combustible, aire, O2 o exceso de aire |
| Eficiencia de la caldera y desglose de pérdidas | combustion_efficiency (M18, método de pérdidas, entalpías NASA) | lo anterior + temperatura de chimenea + pérdida por paredes |
| Combustible, agua de repuesto y costo por tonelada y por GJ de vapor | utility_costs (M17, balance de la isla de vapor con IAPWS-95) | eficiencia (de la herramienta anterior), presión, purga, retorno de condensado, temperaturas y precios |
| Cifras anuales | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | cifras por tonelada × vapor entregado al año. El consumo de combustible se calculó desde la producción de vapor y la eficiencia de la caldera. |
| Ahorro de cada medida y del caso combinado | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | línea base − la misma corrida de los dos motores con el objetivo de la planta |
| Retorno simple | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | inversión (dato de la planta) ÷ ahorro anual; solo si se dio la inversión |

## 7. Supuestos y límites

Esta sección es técnica: está escrita para quien revise el cálculo y conserva los nombres de los parámetros.

Supuestos del cálculo:

- El combustible se valora al precio medio de los recibos (importe ÷ energía). Si el recibo trae cargos fijos, cada GJ ahorrado vale menos que ese promedio.
- Precio del combustible en base HHV: las eficiencias de caldera y horno deben estar en esa misma base (con LHV la eficiencia es numéricamente mayor que con HHV).
- Cada nivel se calcula como generado directamente a su presión en una caldera, sin cogeneración ni turbinas de contrapresión (con cogeneración el vapor de baja cuesta menos).
- Calor entregado al proceso = entalpía del vapor − líquido saturado a la misma presión (no se acredita el subenfriamiento del condensado ni el vapor de flasheo).
- Agua y vapor: IAPWS-95 (CoolProp). Las corrientes líquidas se evalúan en la línea de saturación a su temperatura; el trabajo de la bomba de alimentación no se cobra aparte.
- Desaireador de contacto directo sin venteo: el agua de alimentación se calienta de la mezcla condensado + repuesto a feedwater_T_K con vapor tomado de la propia caldera.
- Química supuesta: C → CO2 (o el CO medido), H → H2O, S → SO2, N del combustible → N2 (sin NOx), ceniza inerte, sin carbono sin quemar ni hollín.
- Aire seco: O2 0.2100, N2 0.7800, Ar 0.0100 (composición de air.yaml de Cantera; da air.dry_air_mole_fractions para otra).
- Referencia de entalpía 298.15 K con el agua como vapor; gas ideal. Polinomios NASA Glenn de 9 coeficientes distribuidos con Cantera 3.2.0 (nasa_gas.yaml; agua líquida de nasa_condensed.yaml).
- Recuperación de calor de purga: flasheo adiabático a flash_pressure_Pa con el vapor de flasheo enviado íntegro al desaireador; el líquido cede calor al agua de repuesto en un intercambiador sin pérdidas y sale a drain_T_K. Sin pérdidas de calor en el tanque ni en las tuberías.
- En la línea base la purga se descarga como líquido saturado a la presión de la caldera, sin recuperar su calor.
- La composición del gas NO es un dato de la planta: es mezcla equivalente de metano + etano ajustada para dar exactamente el poder calorífico medido (952.1 kJ/mol, base PCS; recibo del periodo «octubre 2026»: 9,500.00 GJ ÷ 248,000.0 m³ a 293.15 K y 98066.5 Pa (gas ideal) = 38.306 MJ/m³). Comprobado contra seis gases naturales de referencia con composición completa: el gas equivalente cambia la eficiencia menos de 0.15 puntos (base PCS) y da el CO2 entre 0.9 y 3.2 % más bajo (9 % más bajo en un gas con 7.6 % de CO2), porque no cuenta el CO2 que el gas ya trae.
- La pérdida por paredes y radiación NO es un dato de la planta: es 1.56 % del poder calorífico (base PCS), tomado de G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), mayo 2002, sección 4.2.5 «Shell Loss» (se toma el límite superior de esa afirmación, 1.0 % a plena carga, y se divide entre la carga promedio de la caldera (64 % de su capacidad nominal) porque la misma fuente dice que la pérdida casi no cambia en magnitud con la carga: 1.56 %). Límites: en calderas de tubos de humo la misma fuente espera una pérdida menor; con refractario o aislamiento dañado la pérdida real es mayor; la carga usada es el promedio declarado; si la caldera pasa muchas horas a muy baja carga la pérdida real es mayor.
- La demanda de vapor del proceso (vapor entregado) se mantiene igual en todas las medidas; lo que cambia es el combustible, el agua y la electricidad para producirlo.
- La pérdida por paredes se mantiene como la misma fracción del poder calorífico en todos los casos. En realidad es casi constante en kW, así que al bajar el combustible la fracción sube un poco: el ahorro queda ligeramente sobrestimado.
- Operación en un solo punto promedio: una carga, un O2 y una temperatura de chimenea representan todo el año.
- Los ahorros se valoran con los mismos precios de la línea base.

No está incluido en estas cifras:

- Capital, mano de obra y mantenimiento de la caldera
- Pérdidas de distribución y trampas de vapor
- Tratamiento químico (inclúyelo en el precio del agua de repuesto si lo quieres cobrar)
- combustión: carbono sin quemar en ceniza, hollín y NOx
- combustión: calor sensible de la ceniza
- combustión: vapor de atomización y sopladores de hollín
- combustión: punto de rocío ácido
- combustión: calderas de condensación (chimenea bajo el punto de rocío)
- combustión: purga de caldera (va en utility_costs como blowdown_fraction)
- Inversión de las medidas: nunca se estima; el retorno solo se calcula con la cifra de la planta
- Variación de la carga a lo largo del año, arranques y paros, y eficiencia a carga parcial
- Fugas de vapor, trampas de vapor y aislamiento de tuberías
- Química del agua: la purga mínima y la calidad del condensado no se comprueban
- Punto de rocío ácido y corrosión de la zona fría
- Emisiones (CO2, NOx) y su costo
- Impuestos, financiamiento y mantenimiento de los equipos nuevos

Avisos del cálculo:

- OBJETIVO PUBLICADO, NO DEL PROVEEDOR: la medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)» se evaluó con chimenea a 250 °F (121 °C), de U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012 (la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar condensación y corrosión); es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el proveedor del equipo dirá a qué temperatura llega de verdad.
- OBJETIVO PUBLICADO, NO DEL PROVEEDOR: la medida «Bajar el exceso de aire (afinar la combustión)» se evaluó con 10 % de exceso de aire, de U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012 (en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %); es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra sin producir monóxido de carbono lo decide una prueba de combustión.
- PÉRDIDA POR PAREDES SUPUESTA, NO MEDIDA: la planta declaró que no conoce la pérdida por paredes y radiación de su caldera. Se usó 1.56 % del poder calorífico (base PCS), de G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), mayo 2002, sección 4.2.5 «Shell Loss»: la mayoría de las calderas de tubos de agua deberían perder menos de 1.0 % de la energía del combustible por la envolvente cerca de plena carga y sin daños en el refractario o el forro (la guía calcula en base PCS y remite a la gráfica de la ABMA); se toma el límite superior de esa afirmación, 1.0 % a plena carga, y se divide entre la carga promedio de la caldera (64 % de su capacidad nominal) porque la misma fuente dice que la pérdida casi no cambia en magnitud con la carga: 1.56 %. Con un punto porcentual más de pérdida la eficiencia baja 1.00 puntos y el vapor cuesta +1.18 %. Consultada el 2026-11-14; https://www.energy.gov/sites/prod/files/2014/04/f15/steam_survey_guide.pdf.
- COMPOSICIÓN DEL GAS SUPUESTA, NO ANALIZADA: la planta declaró que no tiene el análisis de su gas. Se usó mezcla equivalente de metano + etano ajustada para dar exactamente el poder calorífico medido (952.1 kJ/mol, base PCS; recibo del periodo «octubre 2026»: 9,500.00 GJ ÷ 248,000.0 m³ a 293.15 K y 98066.5 Pa (gas ideal) = 38.306 MJ/m³). Con las composiciones alternativas probadas la eficiencia cambia como mucho 0.22 puntos y el costo del vapor 0.26 %. Comprobado contra seis gases naturales de referencia con composición completa: el gas equivalente cambia la eficiencia menos de 0.15 puntos (base PCS) y da el CO2 entre 0.9 y 3.2 % más bajo (9 % más bajo en un gas con 7.6 % de CO2), porque no cuenta el CO2 que el gas ya trae.
- solo algunos tramos traen datos de aislamiento: los totales del caso aislado y del ahorro quedan en None; cada tramo con datos trae su propio ahorro

## 8. Comprobaciones

- Balances de masa y energía de los dos motores, en la línea base y en cada medida: cierran (peor residuo 4.8e-11 de la energía del combustible).
- Calor que recibe el vapor (tablas de vapor IAPWS-95) contra eficiencia × combustible: diferencia 0.0e+00.
- Combustible anual por dos rutas (lado del agua y lado de la combustión): diferencia 1.3e-16.
- El caso combinado no ahorra más que la suma de las medidas: se cumple.
- verified = los balances de masa y energía de los dos motores cierran en todos los casos y las rutas independientes coinciden. NO significa que los datos de la planta sean correctos.

## Glosario

- **PCS y PCI**: poder calorífico superior e inferior (en inglés HHV y LHV): la energía de un combustible contando, o sin contar, el calor del vapor de agua que se forma al quemarlo. Una eficiencia en base PCS es numéricamente menor que la misma en base PCI; aquí se usa la base en la que se paga el combustible.
- **Exceso de aire**: el aire que entra de más respecto al estrictamente necesario para quemar. Se mide con el oxígeno (O2) que queda en la chimenea. Hace falta algo; lo que sobra se calienta y se tira.
- **Purga**: agua que se saca de la caldera para que no se concentren las sales. Sale a la temperatura de la caldera.
- **Condensado**: el vapor ya usado, convertido en agua caliente y limpia. Regresarlo ahorra agua y combustible.
- **Agua de repuesto**: agua nueva tratada que repone la purga y el condensado que no regresa.
- **Punto de rocío**: temperatura bajo la cual el vapor de agua de los gases se condensa. Con azufre, el ácido condensa a una temperatura más alta (punto de rocío ácido), que este estudio no calcula.
- **Retorno simple**: años en que el ahorro paga la inversión: inversión ÷ ahorro anual, sin intereses ni impuestos.
- **Vapor entregado y vapor generado**: el generado es todo el que sale de la caldera; el entregado es el que llega al proceso después de descontar el que se usa para calentar el agua de alimentación.
