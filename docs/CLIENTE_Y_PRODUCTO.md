# Cliente y producto

> Decisión del owner, 2026-11-14: el primer cliente son las **plantas con vapor**.
> Todo lo demás en este documento es propuesta de trabajo hasta que se valide con plantas reales.
> Estado de la validación: 0 conversaciones con plantas, 0 diagnósticos hechos con datos reales.

## En una frase

AlchemAI es un ingeniero de procesos con IA que entrega el **estudio completo con números comprobados**
(balance, energía, costo con precios reales, riesgo preliminar), en español y sin licencias caras.

El simulador es el motor. Lo que recibe el cliente es el estudio, no el software.

## A quién ayuda

Plantas medianas en México que queman gas para hacer vapor: alimentos, bebidas, químicos, destilerías, papel.

Quien decide: gerente de planta o de mantenimiento/energía. Quien da los datos: el jefe de calderas o de servicios.

## Su problema

- Pagan gas, luz y agua cada mes y no saben cuánto les cuesta realmente una tonelada de vapor.
- No saben cuánto se va por la chimenea, por la purga o por el condensado que no regresa.
- No tienen un ingeniero con simulador; un estudio de consultoría tarda semanas.

## Lo que ofrecemos primero

Un **diagnóstico de vapor**: con los datos que la planta ya tiene, un reporte en días que dice

1. cuánto cuesta hoy su vapor, desglosado en combustible, agua y electricidad;
2. qué eficiencia tiene su caldera y dónde están las pérdidas;
3. qué medidas ahorran, cada una en energía y en pesos al año;
4. qué número viene de qué dato, y qué no se pudo comprobar.

Datos que da la planta (propuesta, por validar): recibos de gas, luz y agua; presión de operación; O₂ y temperatura
de chimenea; temperatura del agua de alimentación; retorno de condensado; purga; horas de operación.

El nombre del diagnóstico y su precio son decisión del owner. No están definidos.

## Con qué se hace (ya construido)

| Pieza | Herramienta | Estado |
|---|---|---|
| Eficiencia de caldera y pérdidas | `combustion_efficiency` (M18) | validada contra tablas del DOE |
| Costo del vapor, agua de torre, electricidad | `utility_costs` (M17) | validada contra hoja del DOE; tarifas CFE en el banco |
| Recuperación de calor entre corrientes | `pinch_analysis`, `pinch_from_flowsheet`, trade-off (M5, bloque 20) | validadas |
| Costo anual e inversión | `annual_cost`, `capital_investment` (M3) | validadas |
| Gas natural de México como referencia | — | falta (IPGN tras CAPTCHA); lo da el cliente |
| Diagnóstico de punta a punta en un solo flujo | — | no existe todavía |

## Qué es nuevo y qué no

No es nuevo:
- Una IA que arma simulaciones: DWSIM trae su propio asistente desde 2026; AVEVA y AspenTech van por ahí.
- Calculadoras de sistemas de vapor: el DOE de EE. UU. publica herramientas gratuitas en inglés (por verificar
  cuáles siguen vigentes y qué cubren).

Lo que no encontramos en otro lado (cuatro búsquedas el 2026-11-14, no un estudio de mercado):
- No da un número sin prueba; si no puede comprobarlo, se niega.
- El estudio entero en un solo lugar, con precios locales fechados (tarifas de CFE).
- En español y con contexto mexicano.
- Revisa a los propios motores: así se encontró el error que DWSIM corrigió (issue #137).

La tecnología sola no protege. Lo que no se copia fácil: un cliente concreto, su idioma, resultados comprobables
y la cara del owner.

## Hipótesis por validar (ninguna comprobada)

1. Las plantas medianas no conocen el costo real de su vapor.
2. Tienen a la mano los datos que pide el diagnóstico.
3. Pagarían por el reporte, y cuánto.
4. El ahorro identificado alcanza para justificar el precio.

Se validan hablando con plantas, no con más código.

## Otros clientes considerados (no elegidos ahora)

- Estudiantes e ingenieros en español: curso y contenido. Audiencia rápida, paga poco.
- Seguridad de procesos (NOM-028-STPS): vale más, exige credenciales o un socio certificado.
