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

Decisión del owner (2026-11-14): el **diagnóstico piloto** para las primeras plantas va **sin costo**, a cambio de los
datos y de poder contar el caso sin el nombre de la empresa. Cuántas plantas entran al piloto no está decidido.

El nombre del diagnóstico y su precio después del piloto son decisión del owner. No están definidos.

## Con qué se hace (ya construido)

| Pieza | Herramienta | Estado |
|---|---|---|
| Eficiencia de caldera y pérdidas | `combustion_efficiency` (M18) | validada contra tablas del DOE |
| Costo del vapor, agua de torre, electricidad | `utility_costs` (M17) | validada contra hoja del DOE; tarifas CFE en el banco |
| Recuperación de calor entre corrientes | `pinch_analysis`, `pinch_from_flowsheet`, trade-off (M5, bloque 20) | validadas |
| Costo anual e inversión | `annual_cost`, `capital_investment` (M3) | validadas |
| Gas natural de México como referencia | — | falta (IPGN tras CAPTCHA); lo da el cliente |
| Diagnóstico de punta a punta en un solo flujo | `steam_system_diagnostic` (M19) | hecho: línea base, 5 medidas, caso combinado, procedencia; reproduce los ejemplos de 5 hojas del DOE (3 a ≤ 0.04 %, 2 con diferencia explicada) |
| Reporte para el cliente | `steam_diagnostic_report` (M19) | hecho: Markdown en español armado sin modelo de lenguaje; ejemplo ilustrativo en `notebooks/32_diagnostico_de_vapor_reporte.md` |
| Lista de datos que se piden a la planta | `docs/DATOS_PARA_DIAGNOSTICO.md` | hecha desde las entradas reales del motor; sin validar con plantas |
| Planta que no conoce la pérdida por paredes de su caldera | `combustion.setting_loss_unknown` (M19b) | hecho: usa 1.0 % (PCS) de la guía ORNL/TM-2001/263 del DOE, marcado como supuesto y con su sensibilidad; sin validar con calderas reales |
| Diagnóstico con datos de una planta real | — | no existe: 0 plantas |

## Qué es nuevo y qué no

No es nuevo:
- Una IA que arma simulaciones: DWSIM trae su propio asistente desde 2026; AVEVA y AspenTech van por ahí.
- Calculadoras de sistemas de vapor: el DOE de EE. UU. publica MEASUR, gratuita y de código abierto (verificado el
  2026-11-14 en su repositorio `ORNL-AMO/AMO-Tools-Desktop` y en energy.gov: versión 1.10.0 del 2026-10-21, web en
  measur.ornl.gov y de escritorio, licencia permisiva propia de UT-Battelle). Su módulo de vapor es el sucesor de SSAT y
  SSMT (la página antigua del Steam System Modeler ya responde 404): modela caldera, turbinas, tanques de flasheo,
  desaireador y hasta tres cabezales, compara escenarios y trae calculadoras de pérdida por chimenea, purga y economizador.
  Es decir, la física de este diagnóstico ya existe gratis, y con más alcance en cabezales y turbinas. Lo que el
  diagnóstico hace y no vimos en MEASUR (leído de su documentación y código, no usado): calcular la eficiencia desde la
  composición del gas y el O₂ medido en vez de pedirla, partir de recibos y tarifas de CFE en pesos, negarse o declarar qué
  faltó, decir de dónde sale cada número y entregar el reporte en español. No encontramos archivos de traducción en su
  repositorio. Una prueba de caldera de MEASUR se reproduce a +0.003 %.

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
