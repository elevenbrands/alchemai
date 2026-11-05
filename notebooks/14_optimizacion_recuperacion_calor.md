# 14 · Costo anual del flowsheet y primera optimización (bloque 20, 2026-11-05)

**Qué se agregó.**
- `packages/costing/annual.py` calcula el costo total anual: compra × factor de Lang × factor de ubicación, anualizado con el factor de recuperación de capital, más la operación (electricidad, vapor y agua de enfriamiento).
- Solo uso factores de Lang verificados en fuentes:
  - Lang original: 4.74 para fluidos y 3.10 para sólidos;
  - revisión de Peters, Timmerhaus y West (2003) para fluidos: 5.0 para capital fijo y 6.0 para capital total;
  - o un número que dé el usuario.
- Factor de ubicación: la tabla de IDAES, que para México da de 0.84 a 0.92 y cita a Seider.
- Precios y horas son datos de entrada obligatorios; no hay valores por defecto.
- Tool `flowsheet_cost` (29 tools). Cotiza bombas y compresores desde la simulación y los intercambiadores con el área que se le da. Lista lo que no pudo costear en `not_costed`.
- **Corrección del costeo:** la correlación de Seider es cuadrática en ln(área) y, bajo su mínimo, sube otra vez. Con ella, un intercambiador de 0.1 m² costaba 133 500 USD, más que uno de 40 m². Bajo el mínimo, cerca de 9 m² en acero al carbono, ahora se cotiza al mínimo y se avisa.

## Problema: área óptima de un intercambiador alimentación-producto

`packages/costing/heat_recovery.py`:
- **Corrientes.** Alimentación de agua: 10 kg/s de 300 a 395 K. Producto: 10 kg/s de 400 a 305 K.
- **Equipos.** E-1 recupera calor a contracorriente con U = 850. El calentador H-1 usa vapor a 450 K con U = 1100. El enfriador C-1 usa agua de enfriamiento de 290 a 300 K con U = 850.
- **Costeo.** Los tres son de coraza y tubos en U, en acero al carbono. Capital: Lang 4.74, 10 % a 10 años.
- **Precios ilustrativos.** Vapor 8 USD/GJ, agua de enfriamiento 0.5 USD/GJ, 8000 h/año. Están registrados en MOCKS.md.

En cada punto, DWSIM resuelve el flowsheet con E-1 en modo CalcBothTemp_UA. `heat_exchanger_area` dimensiona H-1 y C-1, y `flowsheet_cost` suma todo.

| Área E-1 (m²) | Aproximación (K) | Vapor (kW) | Agua de enfr. (kW) | Capital anualizado (USD/año) | Operación (USD/año) | Costo total anual (USD/año) |
|---|---|---|---|---|---|---|
| 10 | 83.3 | 3281 | 3285 | 53,941 | 803,325 | 857,266 |
| 14 | 78.0 | 3054 | 3058 | 53,670 | 747,653 | 801,323 |
| 20 | 71.4 | 2779 | 2782 | 53,596 | 680,281 | 733,877 |
| 28 | 63.9 | 2460 | 2464 | 53,745 | 602,360 | 656,105 |
| 40 | 55.6 | 2111 | 2115 | 54,182 | 516,820 | 571,002 |
| 56 | 46.9 | 1749 | 1752 | 55,033 | 428,126 | 483,160 |
| 80 | 38.5 | 1395 | 1398 | 56,500 | 341,438 | 397,938 |
| 113 | 30.6 | 1068 | 1072 | 58,889 | 261,461 | 320,350 |
| 159 | 23.8 | 782 | 786 | 62,648 | 191,511 | 254,158 |
| 225 | 18.1 | 544 | 547 | 68,415 | 133,128 | 201,543 |
| 319 | 13.5 | 352 | 356 | 76,891 | 86,277 | 163,168 |
| 450 | 10.0 | 203 | 207 | 89,305 | 49,854 | 139,159 |
| 637 | 7.3 | 91 | 94 | 107,795 | 22,231 | 130,026 |
| 900 | 5.2 | 7 | 10 | 135,679 | 1,679 | 137,358 |

**Óptimo (sección dorada):** E-1 = 650 m², aproximación 7.12 K, recupera 3904 kW, vapor 85 kW, costo total anual 129,992 USD/año (capital 109,204 + operación 20,787).

La curva es plana entre 450 y 900 m²: de 637 a 900 m² el costo cambia solo 6 %. La aproximación óptima de unos 7 K
para agua-agua es coherente con la regla práctica de 10 K, que sale de este mismo compromiso. Con vapor más caro el
óptimo se mueve hacia más área. La herramienta permite repetir el estudio con precios reales.

## Límites
- Los precios y las U son ilustrativos. El resultado vale como método, no como número de diseño.
- El capital usa el factor de Lang global, no factores de módulo desnudo por equipo.
- Sin costo de bombeo por la caída de presión del intercambiador, sin ensuciamiento y sin impuestos ni depreciación.

## El agente resuelve la optimización (Opus, `claude -p`, 29 tools)

Tarjeta `optimo_recuperacion_calor` del bench v3: **1.00 en 188 s**. El agente armó el flowsheet en DWSIM y barrió el UA
de E-1 entre 42 500 y 850 000 W/K. Costeó cada punto con las herramientas y refinó entre 600 y 700 m². Respondió:

| Cantidad | Agente | Oráculo |
|---|---|---|
| Área de E-1 | 650 m² | 650.3 m² |
| Aproximación | 7.0 K | 7.12 K |
| Costo total anual | 130 595 USD/año | 129 992 USD/año (+0.46 %) |

Hizo 49 llamadas, registradas en la traza MCP `notebooks/bench_runs/v3_optimo_cc_opus_run1.mcp_trace.jsonl`. La tool
genérica de optimización del bloque 21 debe bajar ese número.
