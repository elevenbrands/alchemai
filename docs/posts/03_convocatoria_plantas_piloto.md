# Post 3 — Convocatoria: plantas piloto para el diagnóstico de vapor (borrador para LinkedIn, español)

> Estado: LISTO PARA PUBLICAR (2026-11-14). No publicado: lo publica el owner.
> Decidido por el owner: el piloto va **sin costo**, a cambio de los datos y de poder contar el caso sin el nombre de
> la empresa. Decidido por delegación del owner («toma las decisiones»): 3 plantas y contacto por mensaje directo.
> Cualquiera de las dos cosas se cambia en una línea antes de publicar. El nombre del diagnóstico sigue siendo
> «diagnóstico de vapor» y el precio después del piloto no está definido.
> Las cifras de ejemplo son de una planta INVENTADA (`notebooks/32_diagnostico_de_vapor_reporte.md`); el texto lo dice.

---

¿Sabes cuánto te cuesta una tonelada de vapor en tu planta?

No el recibo del gas: el vapor. Con el combustible, el agua, la purga, el condensado que no regresa y lo que se va por la chimenea.

Si la respuesta es «no exactamente», este mensaje es para ti. Sin ese número no se puede saber qué mejora vale la pena.

Estoy buscando **3 plantas en México con caldera de gas** para hacerles un diagnóstico de vapor piloto **sin costo**. A cambio pido dos cosas: los datos de la lista de abajo y permiso para contar el caso sin el nombre de la empresa.

**Qué necesito de la planta** (datos que normalmente ya existen; se llenan en una hoja de Excel, en sus unidades, y lo que no se sepa se deja vacío):
- Recibos de gas, agua y luz de los últimos meses.
- Presión de operación de la caldera.
- Oxígeno y temperatura en la chimenea (de la última prueba de combustión).
- Temperatura del agua de alimentación, porcentaje de purga y cuánto condensado regresa.
- Horas de operación al año.

**Qué recibe la planta**, en días y no en semanas:
- Cuánto cuesta hoy su vapor, desglosado en combustible, agua y electricidad.
- Qué eficiencia tiene su caldera y a dónde se va cada parte de la energía.
- Cuánto ahorraría cada medida (afinar la combustión, economizador, bajar la purga, recuperar su calor, regresar más condensado), en energía y en pesos al año.
- Si hacen un recorrido: cuánto cuestan sus fugas de vapor, trampas falladas y tubería sin aislar.
- Las emisiones de CO2 de la caldera y cuánto baja cada medida.
- De dónde sale cada número, y qué no se pudo calcular por falta de datos.

Para que se vea el formato: en una planta de ejemplo **inventada** (6 t/h de vapor, 6 000 horas al año), el reporte dice que el vapor cuesta 355 pesos por tonelada, que la caldera aprovecha 78.6 % del combustible, y que cuatro medidas juntas ahorrarían 7.8 % del costo anual. Son números de muestra, no de un cliente.

Lo que **no** es: no es una auditoría en sitio, no mide nada (usa los datos que la planta entrega), no revisa trampas ni fugas, y no sustituye al proveedor de la caldera. Los datos de la planta no se publican.

Los cálculos los hace AlchemAI, un proyecto de código abierto. Sus cálculos están comprobados contra tablas y ejemplos publicados del Departamento de Energía de Estados Unidos y contra las mediciones publicadas de una caldera real en México, donde la eficiencia calculada quedó a menos de un punto de la medida.

Si te interesa o conoces a alguien a cargo de calderas o energía en una planta, escríbeme por mensaje directo aquí en LinkedIn.

#Vapor #EficienciaEnergética #Calderas #IngenieríaQuímica #Manufactura #México

---

Notas para el owner:
- Este post pide algo; conviene publicarlo después del post 2 (el del error en DWSIM), que es el que da credibilidad.
- La pérdida por paredes de la caldera, que una planta casi nunca tiene, ya no bloquea (decisión del owner,
  2026-11-14): si la planta dice que no la conoce se usa 1.0 % de una guía del DOE y el reporte lo marca como supuesto.
- Si alguna planta candidata es de tu empleador o de un cliente suyo, revisa antes su política de conflicto de interés.
- Rutas para llegar a la primera planta además del post: compañeros de generación que trabajen en plantas;
  proveedores de calderas y de tratamiento de agua (visitan plantas y les sirve llevar un diagnóstico); cámaras y
  clústeres industriales de tu ciudad. Ninguna se ha intentado.
- Qué se le pide exactamente a la planta: `docs/DATOS_PARA_DIAGNOSTICO.md`.
- A la planta que conteste se le manda la hoja `ejemplos/plantilla_diagnostico_vapor.xlsx`; con la hoja llena, el reporte
  sale con `./vapor diagnostico` (guía en `docs/GUIA_DIAGNOSTICO_VAPOR.md`). Un ejemplo del reporte, con planta inventada,
  está en `ejemplos/diagnostico_vapor/reporte.pdf`: sirve como muestra para enseñar qué recibe la planta.
- El diagnóstico ya cubre también fugas, trampas falladas, tubería sin aislar y CO2. El texto del post no lo dice; si se
  quiere mencionar, agregar una línea en «Qué recibe la planta».
