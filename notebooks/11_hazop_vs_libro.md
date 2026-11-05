# 11 · HAZOP preliminar contra HAZOP publicados (bloque 17, 2026-11-05)

**Pregunta.** ¿El HAZOP por reglas de AlchemAI encuentra los escenarios que encontraron HAZOP publicados y las
investigaciones de accidentes?

**Método.**
1. Cinco procesos, cada uno escrito a mano como archivo DEXPI Process (°C, bar, kg/h) en `packages/hazop/book/cases.py`.
   Entran por `hazop_from_dexpi`, igual que un archivo de otra herramienta. Los números de operación son ilustrativos:
   las fuentes dan el proceso y los peligros, casi nunca el punto de operación.
2. 26 escenarios de referencia parafraseados de las fuentes en `packages/hazop/book/reference.yaml`, escritos y
   **commiteados antes de correr AlchemAI** (commit cd0d922). 22 se pueden deducir de los datos del proceso; 4 piden
   información que un DEXPI de proceso no trae.
3. Un escenario cuenta como encontrado si una fila tiene el nodo correcto, la desviación correcta y el concepto clave
   en causas, consecuencias o salvaguardas (`packages/hazop/book/score.py`).

**Fuentes.**
- Lawley (1974) / Kletz, «Hazop and Hazan» cap. 2: alimentación de una planta de dimerización. Proceso verificado en el ICI Safety Newsletter 68; tabla reconstruida de copias de cursos.
- CCPS, «Guidelines for Hazard Evaluation Procedures»: sección de reacción de DAP. Verificado en el paquete público de la Universidad Tecnológica de Toyohashi.
- Crowl y Louvar, «Chemical Process Safety»: reactor exotérmico con serpentín y agitador. Reconstruido de copias de clase.
- CSB 2005-04-I-TX: torre de refinado y tambor de purga de BP Texas City.
- CSB 2013-03-I-LA: rehervidor aislado de su PSV en Williams Olefins.

Turton y Sinnott/Towler no se usaron: no encontré su ejemplo de HAZOP en una fuente abierta para verificarlo.

## Resultados

| Corrida | Escenarios de datos | Todos |
|---|---|---|
| 1.ª a ciegas (reglas del bloque 16) | 10/22 | 10/26 |
| Misma corrida con el calificador corregido (2 coincidencias falsas: L8, D4) | **9/22 = 41 %** | 9/26 |
| Con 8 grupos de reglas nuevas (este bloque) | **22/22** | 22/26 |

Los 4 que faltan son de alcance «fuera»: corrosión por ácidos orgánicos (L9), drenaje y purga para mantenimiento (L10),
agua de enfriamiento contaminada (C5) y alivio a la atmósfera en vez de a quemador (B4). Ninguno se deduce de un
DEXPI de proceso; el último sí sería visible en un P&ID.

**Reglas nuevas** (genéricas, valen para cualquier flowsheet):
- líquido atrapado que se calienta: alivio térmico en líneas, y lado frío aislado de un intercambiador con BLEVE si es gas licuado inflamable (Williams);
- fuga y pérdida de contención en bombas, válvulas y compresores con material inflamable o tóxico;
- tanques y recipientes con líquido: más temperatura y más presión de vapor; congelamiento si hay agua;
- separador con agua y orgánico: arrastre de agua aguas abajo;
- reactor: reactivo tóxico sin reaccionar que pasa aguas abajo, con el nombre del compuesto y del equipo;
- tanque atmosférico con tóxico: vapores al área de trabajo (DAP);
- reactor exotérmico: enfriamiento de más con acumulación, y flujo inverso del refrigerante; tanque agitado: falta de agitación (un DEXPI con `ReactionProcessType.TANK` se lee como CSTR);
- columna: más nivel en fondos, con líquido al domo y al alivio (Texas City).

## Qué NO prueba esto
- **El 22/22 no es independiente.** Las reglas se escribieron conociendo las referencias. La medición honesta es la línea base de 41 %. Para medir de nuevo hace falta un conjunto apartado que las reglas no hayan visto.
- **No mide precisión.** Las filas crecieron: dimerización de 17 a 26 y DAP de 24 a 36. No se calificó cuántas sobran.
- **La equivalencia se mantiene.** Los 13 flowsheets siguen dando el mismo HAZOP desde DWSIM y desde DEXPI. Las reglas nuevas cambian las filas de ambos caminos por igual.
- Es un HAZOP preliminar por reglas. No sustituye la sesión multidisciplinaria.

## Detalle (corrida final)

Recall (alcance datos): 22/22 = 100% · todos: 22/26 = 85%

### Sección de alimentación de una planta de dimerización de olefinas — 8/10 (26 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| L1 | datos | No hay flujo | J-1 · Flow · No / less flow |
| L2 | datos | Más flujo | T-2 · Level · More level |
| L3 | datos | Más presión | J-1 · Pressure · More pressure (deadhead) |
| L4 | datos | Más presión | J-1 · Pressure · More pressure (blocked-in liquid, thermal expansion) |
| L5 | datos | Menos flujo | J-1 · Flow · Less flow (leak / loss of containment) |
| L6 | datos | Más temperatura | T-1 · Temperature · More temperature |
| L7 | datos | Menos temperatura | T-1 · Temperature · Less temperature (winter) |
| L8 | datos | Parte de / más agua | T-2 · Composition · Part of (water carryover) |
| L9 | fuera | Además de | **no encontrada** |
| L10 | fuera | Otro (mantenimiento) | **no encontrada** |

### Sección de reacción de fosfato diamónico (DAP) — 5/5 (36 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| D1 | datos | Menos o nada de ácido fosfórico | R-1 · Composition · Wrong ratio / composition |
| D2 | datos | Más amoniaco | R-1 · Composition · Wrong ratio / composition |
| D3 | datos | Más flujo de ambos reactivos | R-1 · Temperature · More temperature (runaway) |
| D4 | datos | Más ácido fosfórico | R-1 · Composition · Wrong ratio / composition |
| D5 | datos | Amoniaco en el tanque abierto de DAP | T-3 · Composition · Other (toxic vapors from atmospheric tank) |

### Reactor exotérmico con serpentín de enfriamiento y agitador — 4/5 (10 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| C1 | datos | No hay enfriamiento | R-1 · Temperature · More temperature (runaway) |
| C2 | datos | Flujo inverso del agua de enfriamiento | R-1 · Flow · Reverse flow (cooling medium) |
| C3 | datos | Más enfriamiento | R-1 · Temperature · Less temperature (more cooling) |
| C4 | datos | No hay agitación | R-1 · Agitation · No agitation |
| C5 | fuera | Además de (agua contaminada) | **no encontrada** |

### Torre separadora de refinado y tambor de purga (BP Texas City, 2005) — 4/5 (17 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| B1 | datos | Más nivel en la torre | T-1 · Level · More level (bottoms) |
| B2 | datos | Más presión en la torre | T-1 · Pressure · More pressure |
| B3 | datos | Más nivel en el tambor de purga | F-20 · Level · More level |
| B4 | fuera | Alivio a la atmósfera | **no encontrada** |
| B5 | datos | Más temperatura | T-1 · Temperature · More temperature (reboiler) |

### Rehervidor de la fraccionadora de propileno (Williams Olefins, Geismar, 2013) — 1/1 (11 filas)

| Ref | Alcance | Desviación publicada | Fila de AlchemAI |
|---|---|---|---|
| W1 | datos | Más presión en el lado de proceso aislado | E-B · Pressure · More pressure (cold side blocked in) |

Archivos: `notebooks/hazop_libro/*.xml` (los 5 DEXPI), `notebooks/hazop_libro/ccps_dap_hazop.xlsx` (HAZOP del DAP).
