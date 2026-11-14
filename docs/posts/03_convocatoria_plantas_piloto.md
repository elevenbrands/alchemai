# Post 3 — Convocatoria: plantas piloto para el diagnóstico de vapor (borrador para LinkedIn, español)

> Estado: BORRADOR 2026-11-14. No publicado. Lo publica el owner si lo aprueba.
> Decisiones del owner que este texto necesita antes de publicarse (marcadas con ⟦ ⟧): si el piloto es gratuito o
> con costo, cuántas plantas, cómo quiere que lo contacten y el nombre del diagnóstico.
> Las cifras de ejemplo son de una planta INVENTADA (`notebooks/32_diagnostico_de_vapor_reporte.md`); el texto lo dice.

---

¿Sabes cuánto te cuesta una tonelada de vapor en tu planta?

No el recibo del gas: el vapor. Con el combustible, el agua, la purga, el condensado que no regresa y lo que se va por la chimenea.

Si la respuesta es «no exactamente», este mensaje es para ti. Sin ese número no se puede saber qué mejora vale la pena.

Estoy buscando **⟦3⟧ plantas en México con caldera de gas** para hacerles un diagnóstico de vapor piloto ⟦sin costo, a cambio de poder contar el caso sin nombre de la empresa⟧.

**Qué necesito de la planta** (datos que normalmente ya existen):
- Recibos de gas, agua y luz de los últimos meses.
- Presión de operación de la caldera.
- Oxígeno y temperatura en la chimenea (de la última prueba de combustión).
- Temperatura del agua de alimentación, porcentaje de purga y cuánto condensado regresa.
- Horas de operación al año.

**Qué recibe la planta**, en días y no en semanas:
- Cuánto cuesta hoy su vapor, desglosado en combustible, agua y electricidad.
- Qué eficiencia tiene su caldera y a dónde se va cada parte de la energía.
- Cuánto ahorraría cada medida (afinar la combustión, economizador, bajar la purga, recuperar su calor, regresar más condensado), en energía y en pesos al año.
- De dónde sale cada número, y qué no se pudo calcular por falta de datos.

Para que se vea el formato: en una planta de ejemplo **inventada** (6 t/h de vapor, 6 000 horas al año), el reporte dice que el vapor cuesta 355 pesos por tonelada, que la caldera aprovecha 78.6 % del combustible, y que cuatro medidas juntas ahorrarían 7.8 % del costo anual. Son números de muestra, no de un cliente.

Lo que **no** es: no es una auditoría en sitio, no mide nada (usa los datos que la planta entrega), no revisa trampas ni fugas, y no sustituye al proveedor de la caldera. Los datos de la planta no se publican.

Los cálculos los hace AlchemAI, un proyecto de código abierto. Sus cálculos de caldera y de vapor están comprobados contra tablas y ejemplos publicados del Departamento de Energía de Estados Unidos.

Si te interesa o conoces a alguien a cargo de calderas o energía en una planta, ⟦escríbeme por mensaje directo⟧.

#Vapor #EficienciaEnergética #Calderas #IngenieríaQuímica #Manufactura #México

---

Notas para el owner:
- Este post pide algo; conviene publicarlo después del post 2 (el del error en DWSIM), que es el que da credibilidad.
- Lo que el diagnóstico exige hoy y una planta casi nunca tiene: la pérdida por paredes de la caldera. Decidir qué
  hacer con eso (ver `NEXT_STEP.md`, decisiones de M19) antes de recibir datos reales.
- Si alguna planta candidata es de tu empleador o de un cliente suyo, revisa antes su política de conflicto de interés.
- Rutas para llegar a la primera planta además del post: compañeros de generación que trabajen en plantas;
  proveedores de calderas y de tratamiento de agua (visitan plantas y les sirve llevar un diagnóstico); cámaras y
  clústeres industriales de tu ciudad. Ninguna se ha intentado.
- Qué se le pide exactamente a la planta: `docs/DATOS_PARA_DIAGNOSTICO.md`.
