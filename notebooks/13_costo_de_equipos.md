# 13 · Costo de compra de equipos (bloque 19, 2026-11-05)

**Por qué.** El bloque 18 dio tamaños: área, diámetro y potencia. Sin costo no se puede comparar diseños ni optimizar,
que es la fase 3 del plan. Este bloque convierte un tamaño en dólares.

**Cómo.** No transcribí coeficientes de memoria. Uso la implementación de IDAES de las correlaciones de Seider, Seader,
Lewin y Widagdo, «Product and Process Design Principles», 3.ª ed. Es código abierto con licencia BSD y la uso tal cual:
1. Un bloque sustituto de equipo lleva el tamaño, igual que en los tests de IDAES.
2. IDAES arma sus propias restricciones de costo.
3. Pyomo las resuelve por componentes fuertemente conexos. Cada ecuación es explícita en una variable, así que no hace falta ipopt.

Archivos: `packages/costing/sslw.py` y 3 herramientas en `packages/dwsim_client/tools_costing.py` (28 en total):
- `equipment_cost` cotiza intercambiador, bomba, compresor o recipiente;
- `costing_options` lista los tipos y materiales válidos;
- `cost_unit` toma potencia, ΔP, densidad y caudal de una bomba o compresor ya resuelto en DWSIM.

**Moneda.** IDAES trae el CEPCI de 1990 a 2023; el de 2023 es 797.9. Para otro año la herramienta exige el índice
(`cepci`) en vez de inventarlo.

## Verificación: mismos valores que los tests de IDAES

| Equipo (entradas del test de IDAES) | IDAES test (USD 2018) | AlchemAI (USD 2018) | Diferencia |
|---|---|---|---|
| Intercambiador tubo en U, inox/inox, 12 ft, 1000 m², 2 atm | 476,063.0 | 476,063.2 | +0.0001 % |
| Bomba centrífuga de níquel, factor 1.4, motor cerrado, 10.1 kW, 0.5 bar | 62,868.3 | 62,868.3 | -0.0001 % |
| Compresor centrífugo inox, motor eléctrico, 101.4 kW | 300,695.0 | 300,695.0 | +0.0000 % |
| Recipiente vertical de acero al carbono, D 2 m, con plataformas | 35,958.0 | 35,958.0 | -0.0001 % |

Esto prueba que se usa IDAES correctamente, no que Seider sea exacto. La validación independiente contra un ejemplo
resuelto del libro sigue pendiente: no encontré uno en fuente abierta.

## Economía de escala del intercambiador (CEPCI 2023, 3 bar)

| Área (m²) | Costo de compra USD 2023 (tubo en U, inox/inox) |
|---|---|
| 20 | 71,585 |
| 58.15 | 96,166 |
| 100 | 121,740 |
| 300 | 234,429 |
| 1000 | 631,526 |

Entre 100 y 1000 m² el exponente queda dentro de la regla de los seis décimos (test).

## Límites
- Es **costo de compra** f.o.b., no instalado. Falta multiplicar por un factor de módulo desnudo o de Lang, y esos factores no están implementados.
- Cubre intercambiador, bomba, compresor y recipiente. IDAES también trae columnas con platos, ventiladores, sopladores, turbinas y hornos, pero no están expuestos aún.
- No hay costo de operación: utilidades ni electricidad.

## Bench v3: el agente con dimensionamiento + costo (Opus, `claude -p`, 28 tools)

| Tarjeta | Resultado | Cómo lo resolvió |
|---|---|---|
| Área y costo de un enfriador 1-2 | **1.00** en 28 s (58.15 m², 96 165 USD) | Resolvió el enfriador en DWSIM, sacó el área con `heat_exchanger_area` y cotizó con `equipment_cost` |
| Potencia y costo de una bomba | **1.00** en 26 s (33.43 kW, 27 976 USD) | Resolvió la bomba en DWSIM y la cotizó con `cost_unit` |

Archivos: `notebooks/bench_runs/v3_costo_cc_opus_run1.*`, incluida la traza MCP de cada llamada.
