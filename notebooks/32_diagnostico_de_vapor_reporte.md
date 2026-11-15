# Diagnóstico de vapor — planta ilustrativa (no es un cliente)

> **CASO ILUSTRATIVO. No es una planta real ni un cliente: todos los datos de operación, precios, objetivos e inversiones de este documento son inventados para mostrar el formato (MOCKS.md, ENTRADAS-ILUSTRATIVAS-M19).**

Fecha del cálculo: 2026-11-14. Moneda: MXN. Todas las cifras de este documento salen de dos motores de cálculo comprobados y de los datos que entregó la planta; al final se dice de dónde viene cada una.

## Resumen

Producir vapor le cuesta hoy a la planta **11,700,777 MXN al año** (354.6 MXN por tonelada de vapor entregado), y 89 % de ese costo es combustible. La caldera convierte en vapor **78.6 %** de la energía del combustible que se paga (base PCS); el resto se pierde, sobre todo por la chimenea. Las 4 medidas evaluadas, aplicadas juntas, ahorrarían **912,471 MXN al año** (7.8 % del costo del vapor), que son 8,996 GJ de combustible al año.

> **Aviso: al menos un precio es una REFERENCIA PUBLICADA, no el contrato de la planta.** Electricidad por periodo horario: base 0.9904, intermedia 1.5333, punta 1.6634 MXN/kWh por periodo horario; CFE, Tarifa GDMTH, cuotas aplicables en noviembre de 2026, división Golfo Norte (Monterrey, Nuevo León); periodo 2026-11; consultada el 2026-11-14; periodo vigente al 2026-11-14; referencia «MX-CFE-GDMTH-GOLFO-NORTE-2026-11»; https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRENegocio/Tarifas/GranDemandaMTH.aspx. Cargo por demanda eléctrica: 444.12 MXN/kW-mes; CFE, Tarifa GDMTH, cuotas aplicables en noviembre de 2026, división Golfo Norte (Monterrey, Nuevo León); periodo 2026-11; consultada el 2026-11-14; periodo vigente al 2026-11-14; referencia «MX-CFE-GDMTH-GOLFO-NORTE-2026-11»; https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRENegocio/Tarifas/GranDemandaMTH.aspx. Las cifras en dinero cambian cuando se sustituya por el precio real.

## 1. Cómo opera hoy la caldera

| Concepto | Valor | Origen |
|---|---|---|
| Presión de operación | 9.00 bar absolutos (8.2 kg/cm² manométricos); el vapor sale a 175 °C | dato de la planta |
| Agua de alimentación que entra a la caldera | 95 °C | dato de la planta |
| Oxígeno en la chimenea (base seca) y exceso de aire | 6.5 % de O2; 40 % de aire de más | dato de la planta / calculado |
| Temperatura de chimenea | 230 °C | dato de la planta |
| Eficiencia de la caldera (base PCS) | **78.6 %** | calculado |
| Combustible por tonelada de vapor (base PCS) | 3.339 GJ | calculado |
| Agua de repuesto por tonelada de vapor | 0.646 m³ | calculado |
| Costo del vapor | **354.6 MXN por tonelada** (174.6 MXN por GJ de calor entregado) | calculado |
|   de ese costo: combustible / agua / electricidad | 317.2 / 24.6 / 12.9 MXN por tonelada | calculado |
| Horas de operación | 6,000 h al año | dato de la planta |
| Vapor generado por la caldera | 36,000 t al año (6.00 t/h en promedio) | dato de la planta |
| Vapor entregado al proceso | 32,997 t al año | calculado |
| Combustible al año (base PCS) | 110,172 GJ (≈ 2,685,445 m³ normales a 0 °C y 1 atm) | calculado |
| Agua de repuesto al año | 21,326 m³ | calculado |
| Electricidad de la caldera al año | 164,985 kWh | calculado |
| **Costo anual del vapor** | **11,700,777 MXN** (combustible 10,466,307; agua 810,398; electricidad 424,071) | calculado |
| Emisiones directas de CO2 de la caldera | 5,563 toneladas al año; 168.6 kg por tonelada de vapor | calculado |

Se dieron la producción de vapor y el consumo de combustible: las cifras anuales usan la producción de vapor y el otro dato sirve de comparación (ver «Comprobaciones»).

«Por tonelada» siempre es por tonelada de vapor entregado al proceso. La caldera genera algo más, porque una parte del vapor se usa para calentar el agua de alimentación.

Precios usados: combustible 95.00 MXN/GJ (PCS); agua 38.00 MXN/m³; electricidad 2.570 MXN/kWh.

## 2. A dónde se va la energía del combustible

De cada 100 unidades de energía del combustible (base PCS):

| Destino | Unidades de cada 100 | Qué es |
|---|---|---|
| Se va al vapor que sale de la caldera | 77.6 | es lo que se aprovecha |
| Se va con la purga | 1.0 | agua caliente que se tira para controlar las sales |
| Chimenea: gases secos calientes | 8.6 | el aire que sobra y los gases salen calientes y se llevan calor |
| Chimenea: vapor de agua que se forma al quemar (calor latente) | 9.7 | el hidrógeno del combustible se vuelve vapor y ese vapor no se condensa en la caldera |
| Chimenea: ese mismo vapor de agua, por salir caliente | 1.6 | calor sensible del vapor formado |
| Chimenea: humedad del aire | 0.2 | agua que trae el aire de combustión |
| Paredes de la caldera (radiación y convección) | 1.5 | calor que pierde la envolvente al cuarto de calderas |
| Calor que ya trae el aire sobre 25 °C | -0.2 | entra con el aire o el combustible; se resta de las pérdidas |

El vapor y la purga suman la eficiencia de la caldera (78.6 %). El punto de rocío del agua en los gases es 54 °C: la chimenea no debe bajar de ahí.

## 3. Medidas de ahorro

Ordenadas de mayor a menor ahorro. Cada una se calculó repitiendo el estudio completo con un solo cambio: el objetivo que dio la planta. Ahorros al año; la demanda de vapor del proceso es la misma en todos los casos.

| Medida | Qué cambia | Combustible ahorrado (GJ PCS) | Agua ahorrada (m³) | CO2 evitado (t) | Ahorro (MXN) | % del costo del vapor | Inversión | Retorno simple | Inversión máxima para recuperar en 2 años |
|---|---|---|---|---|---|---|---|---|---|
| Bajar la temperatura de chimenea (economizador o precalentador de aire) | temperatura de chimenea: 230 °C → 150 °C | 5,487 | 0 | 277 | **521,290** | 4.5 % | 2,800,000 MXN | 5.4 años | 1,042,579 MXN |
| Bajar el exceso de aire (afinar la combustión) | oxígeno en chimenea (base seca): 6.5 % → 3.0 %; exceso de aire: 40.3 % → 15.0 %; temperatura de chimenea: 230 °C → 220 °C | 2,828 | 0 | 143 | **268,694** | 2.3 % | 400,000 MXN | 1.5 años | 537,388 MXN |
| Recuperar el calor de la purga | calor de la purga: se tira → tanque de flasheo a 1.20 bar absolutos más intercambiador que deja la purga a 40 °C | 2,337 | 482 | 118 | **240,284** | 2.1 % | no dada | sin cotización | 480,567 MXN |
| Bajar la purga | purga, del agua de alimentación: 8.0 % → 5.0 % | 1,030 | 1,247 | 52 | **145,187** | 1.2 % | 250,000 MXN | 1.7 años | 290,374 MXN |

La última columna no es una cotización: es dos veces el ahorro anual, es decir, lo más que se puede pagar por la medida para recuperarlo en dos años. Sirve para saber si vale la pena pedir la cotización. El retorno simple solo se calcula con la inversión que cotice el proveedor: nunca se estima.

**Bajar la temperatura de chimenea (economizador o precalentador de aire).** Eficiencia de la caldera 78.6 % → 82.7 %; costo del vapor 354.6 → 338.8 MXN por tonelada.
- La chimenea quedaría 96 °C por encima del punto de rocío del agua (54 °C).
- El agua saldría del economizador a unos 121 °C.
- El punto de rocío ÁCIDO no se modela: con azufre en el combustible la corrosión empieza a una temperatura más alta que el punto de rocío del agua. El límite real lo fija el proveedor del equipo.
- La caldera y el equipo de recuperación se tratan como una sola envolvente: el agua de alimentación y el aire se miden a la entrada de esa envolvente y la chimenea a su salida.
- La temperatura del agua a la salida del economizador supone que los gases siguen saliendo de la caldera a la temperatura de chimenea de hoy.

**Bajar el exceso de aire (afinar la combustión).** Eficiencia de la caldera 78.6 % → 80.7 %; costo del vapor 354.6 → 346.5 MXN por tonelada.
- La chimenea quedaría 163 °C por encima del punto de rocío del agua (57 °C).
- El objetivo de oxígeno lo fija la planta: bajar el aire más de lo que el quemador aguanta produce monóxido de carbono (CO) y combustible sin quemar, que este estudio no predice.

**Recuperar el calor de la purga.** Eficiencia de la caldera 78.6 % → 78.6 %; costo del vapor 354.6 → 347.3 MXN por tonelada.
- 13.5 % de la purga se vuelve vapor de flasheo a 1.20 bar absolutos; el resto calienta el agua de repuesto hasta 30 °C y se va al drenaje a 40 °C.

**Bajar la purga.** Eficiencia de la caldera 78.6 % → 78.6 %; costo del vapor 354.6 → 350.2 MXN por tonelada.
- No se comprueba la química del agua: la purga mínima la fijan los sólidos disueltos que tolera la caldera.

## 4. Todas las medidas juntas

|  | Ahorro en dinero | Ahorro de combustible (PCS) |
|---|---|---|
| Suma de las medidas una por una (no es alcanzable) | 1,175,454 MXN al año | 11,682 GJ al año |
| **Caso combinado (una sola corrida con todos los cambios)** | **912,471 MXN al año** | 8,996 GJ al año |

El caso combinado da 78 % de la suma. Los ahorros no se suman: cada medida deja menos que ahorrar a las demás (menos gases que enfriar, menos purga de la que recuperar calor, menos combustible sobre el que aplicar la mejora de eficiencia). El caso combinado es una sola corrida con todos los cambios.

Con todas las medidas la eficiencia de la caldera pasaría de 78.6 % a 83.7 % y el vapor costaría 326.9 MXN por tonelada en lugar de 354.6. Retorno simple del conjunto: no calculado: falta la inversión de al menos una medida o las cifras anuales.

Con todas las medidas se dejarían de emitir 454 toneladas de CO2 al año. Para recuperar la inversión en dos años, el conjunto no debería costar más de 1,824,942 MXN (en tres años, 2,737,413 MXN).

- La temperatura de chimenea del caso combinado es la menor de los objetivos dados.

## 5. Lo que no se evaluó y por qué

- **Regresar más condensado**: no evaluada: falta el dato. Hace falta: el porcentaje del vapor que se podría regresar como condensado después de las obras. Sale de un levantamiento de la red de condensado.

Datos que faltaron y qué impide cada uno:

- la inversión de «Recuperar el calor de la purga» (cotización del proveedor, en MXN): impide el retorno simple de «Recuperar el calor de la purga».
- el recorrido de la distribución: fugas de vapor, trampas falladas y tramos de tubería sin aislar (cuántos, diámetro del orificio o del tubo, longitud y presión): impide valorar lo que se pierde entre la caldera y el proceso; este estudio cubre solo la casa de calderas.
- el monóxido de carbono (CO) medido en la chimenea: impide contar la pérdida por combustión incompleta: se tomó combustión completa.

## 6. De dónde sale cada número

Tres orígenes posibles: **dato de la planta** (lo entregó la planta; no se comprobó en sitio), **calculado** (lo produjo una herramienta a partir de esos datos) y **referencia publicada** (un precio o un valor de una fuente pública con fecha, usado solo porque se pidió o porque la planta declaró que no tiene el dato).

Datos de la planta:

| Dato | Valor |
|---|---|
| Combustible | CH4 92.00 %, C2H6 5.00 %, C3H8 1.00 %, N2 1.50 %, CO2 0.50 % |
| Presión atmosférica del sitio | 0.950 bar absolutos |
| Temperatura del combustible en el quemador | 25.0 °C |
| Temperatura del aire de combustión | 30.0 °C |
| Humedad relativa del aire | 50.00 % |
| Oxígeno en chimenea (base seca) | 6.50 % |
| Temperatura de chimenea | 230.0 °C |
| Pérdida por paredes y radiación (del poder calorífico, base PCS) | 1.50 % |
| Presión de la caldera | 9.000 bar absolutos |
| Sobrecalentamiento | 0 °C de diferencia |
| Purga (del agua de alimentación) | 8.00 % |
| Agua de alimentación | con vapor de la propia caldera (desaireador o tanque calentado) |
| Temperatura del agua de repuesto | 22.0 °C |
| Retorno de condensado (del vapor) | 45.00 % |
| Temperatura del agua de alimentación | 95.0 °C |
| Temperatura del condensado | 85.0 °C |
| Electricidad de ventiladores y bombas | 5 kWh por tonelada de vapor entregado |
| Horas de operación al año | 6,000 h al año |
| Producción de vapor de la caldera | 6.00 t/h |
| Consumo de combustible al año | 115,000 GJ al año (PCS) |
| Medida «Bajar el exceso de aire (afinar la combustión)»: oxígeno objetivo en chimenea | 3.00 % (base seca) |
| Medida «Bajar el exceso de aire (afinar la combustión)»: temperatura de chimenea después de afinar | 220.0 °C |
| Medida «Bajar el exceso de aire (afinar la combustión)»: inversión | 400,000 MXN |
| Medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)»: temperatura de chimenea objetivo | 150.0 °C |
| Medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)»: equipo | economizador |
| Medida «Bajar la temperatura de chimenea (economizador o precalentador de aire)»: inversión | 2,800,000 MXN |
| Medida «Bajar la purga»: purga objetivo | 5.00 % |
| Medida «Bajar la purga»: inversión | 250,000 MXN |
| Medida «Recuperar el calor de la purga»: presión del tanque de flasheo | 1.200 bar absolutos |
| Medida «Recuperar el calor de la purga»: temperatura de la purga al drenaje | 40.0 °C |

Precios:

| Precio | Valor | Origen | Detalle |
|---|---|---|---|
| Precio del combustible | 95.0000 MXN/GJ (PCS) | calculado | importe pagado 10,925,000.00 MXN ÷ energía 115,000.00 GJ (energía facturada o medida, tal cual); recibos del periodo «12 recibos mensuales (ilustrativos)» (dato de la planta); precio medio del recibo: incluye todo lo que traiga el importe (transporte, distribución, cargos fijos, impuestos) |
| Precio del agua de repuesto | 38.0000 MXN/m³ | dato de la planta | precio dado por la planta |
| Reparto de la energía eléctrica por periodo horario | base 30.00 %, intermedia 58.00 %, punta 12.00 % | dato de la planta | perfil de consumo dado por la planta |
| Factor de carga eléctrico (carga media ÷ demanda facturable) | 75.00 % | dato de la planta | dado por la planta |
| Electricidad por periodo horario | base 0.9904, intermedia 1.5333, punta 1.6634 MXN | referencia publicada | base 0.9904, intermedia 1.5333, punta 1.6634 MXN/kWh por periodo horario; CFE, Tarifa GDMTH, cuotas aplicables en noviembre de 2026, división Golfo Norte (Monterrey, Nuevo León); periodo 2026-11; consultada el 2026-11-14; periodo vigente al 2026-11-14; referencia «MX-CFE-GDMTH-GOLFO-NORTE-2026-11»; https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRENegocio/Tarifas/GranDemandaMTH.aspx |
| Cargo por demanda eléctrica | 444.1200 MXN | referencia publicada | 444.12 MXN/kW-mes; CFE, Tarifa GDMTH, cuotas aplicables en noviembre de 2026, división Golfo Norte (Monterrey, Nuevo León); periodo 2026-11; consultada el 2026-11-14; periodo vigente al 2026-11-14; referencia «MX-CFE-GDMTH-GOLFO-NORTE-2026-11»; https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRENegocio/Tarifas/GranDemandaMTH.aspx |

Cifras calculadas:

| Cifra | Herramienta | A partir de |
|---|---|---|
| Poder calorífico, aire y gases de combustión, punto de rocío del agua | combustion_efficiency (M18, método de pérdidas, entalpías NASA) | combustible, aire, O2 o exceso de aire |
| Eficiencia de la caldera y desglose de pérdidas | combustion_efficiency (M18, método de pérdidas, entalpías NASA) | lo anterior + temperatura de chimenea + pérdida por paredes |
| Combustible, agua de repuesto y costo por tonelada y por GJ de vapor | utility_costs (M17, balance de la isla de vapor con IAPWS-95) | eficiencia (de la herramienta anterior), presión, purga, retorno de condensado, temperaturas y precios |
| Cifras anuales | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | cifras por tonelada × vapor entregado al año. Se dieron la producción de vapor y el consumo de combustible: las cifras anuales usan la producción de vapor y el otro dato sirve de comparación (ver «Comprobaciones»). |
| Ahorro de cada medida y del caso combinado | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | línea base − la misma corrida de los dos motores con el objetivo de la planta |
| Retorno simple | steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores) | inversión (dato de la planta) ÷ ahorro anual; solo si se dio la inversión |

## 7. Supuestos y límites

Esta sección es técnica: está escrita para quien revise el cálculo.

Supuestos del cálculo:

- El combustible se valora al precio medio de los recibos (importe ÷ energía). Si el recibo trae cargos fijos, cada GJ ahorrado vale menos que ese promedio.
- Cargo por demanda repartido en la energía: demanda facturable = carga media en operación ÷ factor de carga, 12 facturas al año, energía = carga media × horas de operación.
- Precio del combustible en base PCS: las eficiencias de caldera y horno deben estar en esa misma base (con PCI la eficiencia es numéricamente mayor que con PCS).
- Cada nivel se calcula como generado directamente a su presión en una caldera, sin cogeneración ni turbinas de contrapresión (con cogeneración el vapor de baja cuesta menos).
- Calor entregado al proceso = entalpía del vapor − líquido saturado a la misma presión (no se acredita el subenfriamiento del condensado ni el vapor de flasheo).
- Agua y vapor: IAPWS-95 (CoolProp). Las corrientes líquidas se evalúan en la línea de saturación a su temperatura; el trabajo de la bomba de alimentación no se cobra aparte.
- Desaireador de contacto directo sin venteo: el agua de alimentación se calienta de la mezcla condensado + repuesto a la temperatura de alimentación con vapor tomado de la propia caldera.
- Química supuesta: C → CO2 (o el CO medido), H → H2O, S → SO2, N del combustible → N2 (sin NOx), ceniza inerte, sin carbono sin quemar ni hollín.
- Aire seco: O2 0.2100, N2 0.7800, Ar 0.0100 (composición de air.yaml de Cantera).
- Referencia de entalpía 298.15 K con el agua como vapor; gas ideal. Polinomios NASA Glenn de 9 coeficientes distribuidos con Cantera 3.2.0.
- Recuperación de calor de purga: flasheo adiabático a la presión del tanque de flasheo con el vapor de flasheo enviado íntegro al desaireador; el líquido cede calor al agua de repuesto en un intercambiador sin pérdidas y sale a la temperatura de drenaje dada. Sin pérdidas de calor en el tanque ni en las tuberías.
- En la línea base la purga se descarga como líquido saturado a la presión de la caldera, sin recuperar su calor.
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
- combustión: purga de caldera (se cuenta en el balance de vapor)
- Inversión de las medidas: nunca se estima; el retorno solo se calcula con la cifra de la planta
- Variación de la carga a lo largo del año, arranques y paros, y eficiencia a carga parcial
- Fugas de vapor, trampas de vapor y aislamiento de tuberías (no se dio el recorrido de la distribución)
- Química del agua: la purga mínima y la calidad del condensado no se comprueban
- Punto de rocío ácido y corrosión de la zona fría
- Emisiones distintas del CO2 directo de la caldera (NOx, metano) y el costo de emitir
- Impuestos, financiamiento y mantenimiento de los equipos nuevos

Avisos del cálculo:

- PRECIO DE REFERENCIA PUBLICADO, NO TU CONTRATO: electricity_time_of_use_per_kWh = {'base': 0.9904, 'intermedia': 1.5333, 'punta': 1.6634} MXN/kWh por periodo horario (CFE, Tarifa GDMTH, cuotas aplicables en noviembre de 2026, división Golfo Norte (Monterrey, Nuevo León); periodo 2026-11; antigüedad el periodo termina en 16 días al 2026-11-14); demand_charge_per_kW_month = 444.12 MXN/kW-mes (CFE, Tarifa GDMTH, cuotas aplicables en noviembre de 2026, división Golfo Norte (Monterrey, Nuevo León); periodo 2026-11; antigüedad el periodo termina en 16 días al 2026-11-14). Referencia «MX-CFE-GDMTH-GOLFO-NORTE-2026-11», consultada el 2026-11-14. Sustitúyelos por los de tu planta en prices= en cuanto los tengas.
- El consumo anual de combustible se tomó de los recibos. Si ese medidor alimenta también otros equipos, la comparación con la producción de vapor (eficiencia por entradas y salidas) sale más baja de lo real.
- Nota de la referencia (electricity_time_of_use_per_kWh): Tarifa de suministro básico para demanda ≥ 100 kW en media tensión. Tres periodos horarios: el precio medio depende de TU perfil de carga (fracción de la energía en base, intermedia y punta), que debes dar. La página consultada no dice si las cuotas incluyen IVA: confírmalo con tu recibo. Cargo fijo de 502.03 MXN/mes no incluido en el precio por kWh.
- Nota de la referencia (demand_charge_per_kW_month): Suma de los cargos de distribución y de capacidad aplicada a UNA sola demanda facturable (supuesto de la herramienta): CFE aplica cada cargo a una demanda distinta (reglas 7.1 y 7.2 de la tarifa: mínimos entre la demanda máxima y la que resulta de la energía y un factor de carga).

## 8. Comprobaciones

- Balances de masa y energía de los dos motores, en la línea base y en cada medida: cierran (peor residuo 4.8e-11 de la energía del combustible).
- Calor que recibe el vapor (tablas de vapor IAPWS-95) contra eficiencia × combustible: diferencia 0.0e+00.
- Combustible anual por dos rutas (lado del agua y lado de la combustión): diferencia 0.0e+00.
- El caso combinado no ahorra más que la suma de las medidas: se cumple.
- Se dieron producción de vapor y consumo de combustible: la eficiencia por entradas y salidas es 75.3 % y la del método de pérdidas 78.6 % (diferencia -3.3 puntos). Eficiencia por entradas y salidas (calor al vapor medido ÷ combustible medido) contra la del método de pérdidas. Una diferencia grande indica un medidor descalibrado, una purga distinta de la declarada, combustible que se usa en otro equipo o vapor que no se mide.
- «Comprobado» significa que los balances de masa y energía de los dos motores cierran en todos los casos y las rutas independientes coinciden. NO significa que los datos de la planta sean correctos.

## Glosario

- **PCS y PCI**: poder calorífico superior e inferior (en inglés HHV y LHV): la energía de un combustible contando, o sin contar, el calor del vapor de agua que se forma al quemarlo. Una eficiencia en base PCS es numéricamente menor que la misma en base PCI; aquí se usa la base en la que se paga el combustible.
- **Exceso de aire**: el aire que entra de más respecto al estrictamente necesario para quemar. Se mide con el oxígeno (O2) que queda en la chimenea. Hace falta algo; lo que sobra se calienta y se tira.
- **Purga**: agua que se saca de la caldera para que no se concentren las sales. Sale a la temperatura de la caldera.
- **Condensado**: el vapor ya usado, convertido en agua caliente y limpia. Regresarlo ahorra agua y combustible.
- **Agua de repuesto**: agua nueva tratada que repone la purga y el condensado que no regresa.
- **Punto de rocío**: temperatura bajo la cual el vapor de agua de los gases se condensa. Con azufre, el ácido condensa a una temperatura más alta (punto de rocío ácido), que este estudio no calcula.
- **Retorno simple**: años en que el ahorro paga la inversión: inversión ÷ ahorro anual, sin intereses ni impuestos.
- **Vapor entregado y vapor generado**: el generado es todo el que sale de la caldera; el entregado es el que llega al proceso después de descontar el que se usa para calentar el agua de alimentación.
