# Paquete para una planta piloto

Todo lo que se le manda a una planta que acepte el diagnóstico de vapor piloto, y lo que se hace cuando regresa la hoja.
Nada de esta carpeta se ha enviado a nadie: lo envía el owner.

## Qué se le manda a la planta (3 archivos)

1. `plantilla_diagnostico_vapor.xlsx`: la hoja que llenan, en sus unidades. Se puede imprimir (5 páginas) y llenar a mano.
2. `ejemplo_de_reporte.pdf`: lo que van a recibir. Es el caso de una caldera real publicada (tesina de la UNAM, 2021), no
   un cliente.
3. El mensaje de `mensaje_para_la_planta.md`, pegado en el correo o en el mensaje.

## Qué se hace cuando regresa la hoja

```bash
./vapor diagnostico hoja_de_la_planta.xlsx --salida reporte_planta
```

- Si la terminal lista errores de captura o datos indispensables que faltan, se le piden a la planta tal cual salen
  (cada uno trae su celda y su nombre).
- Si el reporte sale, leer primero «Valores supuestos» y «Datos por confirmar» de la terminal: es lo que hay que
  comentarle a la planta antes de mandar el reporte.
- Se manda `reporte_planta/reporte.pdf`.

## Qué se le pide a cambio (decisión del owner, 2026-11-14)

El piloto va sin costo. A cambio: los datos, y permiso para contar el caso sin el nombre de la empresa.

## Lo que hay que decirle con honestidad

- El diagnóstico usa los datos que entrega la planta; no mide nada en sitio.
- Está comprobado contra ejemplos del Departamento de Energía de EE. UU. y contra las mediciones publicadas de una
  caldera real en México (eficiencia calculada a menos de un punto de la medida). Su planta sería de las primeras en
  usarlo directamente.
- No estima cuánto cuestan las mejoras: da el ahorro y el tope de inversión para recuperarla en dos años.
