Recall (alcance datos): 30/30 = 100% · todos: 34/39 = 87%

### Estabilizadora de crudo amargo con rehervidor de vapor (nodo 5) — 8/9 (48 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| M1 | datos | Más nivel en el fondo de la columna | V-102 · Flow · No / less flow (heating medium) |
| M2 | datos | Menos nivel en el fondo | V-102 · Level · Less level (bottoms) |
| M3 | datos | Más flujo de vapor al rehervidor | V-102 · Pressure · More pressure |
| M4 | datos | Más presión en la línea de gas de domo | V-102 · Pressure · More pressure |
| M5 | datos | Menos flujo de crudo estabilizado a la bomba | P-101 · Pressure · Less pressure (suction) |
| M6 | datos | No hay flujo de vapor al rehervidor | V-102 · Flow · No / less flow (heating medium) |
| M7 | datos | Más presión en el rehervidor por fuga de tubos | E-101 · Composition · Other (tube leak) |
| M8 | fuera | Más flujo de gas al lavador por crudo con más H2S | **no encontrada** |
| M9 | fuera | Pérdida de contención por recipiente roto o válvula abierta a la atmósfera | P-101 · Flow · Less flow (leak / loss of containment) |

### Tanque de almacenamiento de hexano T-301 (ejemplo CCPS reanalizado) — 8/8 (25 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| H1 | datos | Más nivel | T-301 · Level · More level |
| H2 | datos | Más presión | T-301 · Pressure · More pressure |
| H3 | datos | Menos presión (vacío) | T-301 · Pressure · Less pressure (vacuum) |
| H4 | datos | Menos nivel / sin flujo de entrada | J-OUT · Flow · No / less flow |
| H5 | datos | Menos temperatura | T-301 · Temperature · Less temperature (winter) |
| H6 | datos | Más temperatura | T-301 · Temperature · More temperature |
| H7 | datos | Fuga (pérdida de contención) | J-UNLOAD · Flow · Less flow (leak / loss of containment) |
| H8 | fuera | Menos presión por bloqueo tras vaporizado | T-301 · Pressure · Less pressure (vacuum) |

### Tanque de nitrato de amonio líquido caliente al 83 % con serpentín de vapor — 7/9 (35 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| N1 | datos | No hay flujo en la recirculación / descarga | T-AN · Flow · No / less flow |
| N2 | datos | Más flujo de vapor al serpentín o rotura del serpentín | T-AN · Temperature · More temperature (decomposition) |
| N3 | datos | Más presión en el serpentín | PCV-STM · Pressure · More pressure (downstream) |
| N4 | datos | Menos presión en el tanque (vacío) | T-AN · Pressure · Less pressure (vacuum) |
| N5 | datos | Menos temperatura | T-AN · Temperature · Less temperature (solidification) |
| N6 | datos | Menos nivel | P-AN · Pressure · Less pressure (suction) |
| N7 | datos | Otra composición: entra condensado o amoniaco de más | T-AN · Composition · Other (toxic vapors from atmospheric tank) |
| N8 | fuera | Quemaduras al tomar muestra | **no encontrada** |
| N9 | fuera | Evento externo | **no encontrada** |

### Tanque de hidrógeno líquido con escudo enfriado por vapor, calentador y regulador a celda de combustible — 6/8 (18 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| W1 | datos | Más flujo del regulador / por el disco de ruptura | V-H2 · Pressure · More pressure |
| W2 | datos | Menos o nada de flujo del regulador | V-H2 · Flow · No / less flow |
| W3 | datos | Más nivel | V-H2 · Level · More level |
| W4 | datos | Más presión por encima del MAWP | V-H2 · Pressure · More pressure |
| W5 | datos | Más temperatura | V-H2 · Temperature · More temperature |
| W6 | datos | Menos temperatura | PCV-FC · Temperature · Less temperature (cold front downstream) |
| W7 | fuera | Menos presión por venteo abierto en almacenamiento | **no encontrada** |
| W8 | fuera | Movimiento involuntario (tornillos, ajuste del regulador) | **no encontrada** |

### Evaporador de amoniaco en ciclo de deshielo (incidente Millard Refrigerated Services, 2010) — 5/5 (24 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| A1 | datos | Más flujo de amoniaco líquido al evaporador durante el deshielo | E-EVAP · Pressure · More pressure (cold side blocked in) |
| A2 | datos | Más flujo de gas caliente a la succión de baja temperatura | XV-HG · Pressure · More pressure (downstream) |
| A3 | datos | Menos presión en los serpentines del evaporador | E-EVAP · Pressure · Less pressure (condensation-induced hammer) |
| A4 | fuera | Líquido residual en el ciclo de deshielo | E-EVAP · Pressure · Less pressure (condensation-induced hammer) |
| A5 | fuera | Arranque tras una interrupción | E-EVAP · Pressure · Less pressure (condensation-induced hammer) |
