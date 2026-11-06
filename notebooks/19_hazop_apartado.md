# 19 · HAZOP preliminar sobre un conjunto apartado: recall sin sesgo y precisión (bloque 25, 2026-11-05)

**Pregunta.** El bloque 17 llegó a 22/22 escribiendo reglas con las referencias a la vista. ¿Cuánto encuentran esas
reglas en HAZOP publicados que **no** habían visto, y cuántas de las filas que generan sobran o están mal?

**Método.**
1. Cinco HAZOP publicados nuevos, leídos completos en la fuente (ninguno se usó al escribir las reglas):
   estabilizadora de crudo amargo con rehervidor de vapor (Marhavilas et al. 2020, *Sustainability*, CC BY, nodo 5);
   tanque de hexano T-301 del CCPS reanalizado (Single, Schmidt y Denecke 2020, *J. Loss Prev.*, acceso abierto);
   tanque de nitrato de amonio al 83 % a 130-150 °C (Benedetti-Márquez et al. 2018, *Chem. Eng. Trans.*);
   tanque de hidrógeno líquido con calentador y regulador (WSU HYPER Lab 2020); evaporador de amoniaco en deshielo
   (SafeChE Univ. de Michigan 2021, hoja didáctica: el caso más débil).
2. 39 escenarios parafraseados en `packages/hazop/book/heldout.yaml` (30 deducibles de datos, 9 «fuera») y los 5 procesos
   como DEXPI a mano en `heldout_cases.py`, **commiteados antes de correr** (2edb0fe). Mismo calificador del bloque 17.
3. Rúbrica de precisión escrita antes de correr (`PRECISION_RUBRIC.md`): cada fila es *coincide* (la empareja el
   calificador), *plausible*, *ruido* o *espuria*. Las etiqueta un revisor independiente (Claude Sonnet) que ve solo la
   descripción del proceso y las filas, nunca las reglas ni la referencia.

## Resultados

| Corrida | Recall datos | Recall todos | Filas | Precisión útil | Ruido | Espurias |
|---|---|---|---|---|---|---|
| **A ciegas**, reglas del bloque 17, criterios originales | **25/30 = 83 %** | 27/39 | 154 | **106/154 = 69 %** | 44 (29 %) | 4 (2.6 %) |
| Misma corrida, calificador endurecido (6 coincidencias flojas: M2, M3, M6, N5, A1, A5) | 23/30 = 77 % | 24/39 | 154 | — | — | — |
| Con 7 reglas nuevas + 5 correcciones (este bloque) | **30/30** | 34/39 | 150 | **114/150 = 76 %** | 36 (24 %) | **0** |

La cifra oficial de generalización es la primera fila: **83 % de recall a ciegas** (41 % en el bloque 17 antes de
escribir las reglas genéricas). Los 5 escenarios de datos que faltaban eran de sustancia o de régimen térmico, no de
tipo de equipo: vacío en tanques atmosféricos (H3, N4), descomposición térmica del nitrato (N2), frente frío a sellos
no criogénicos (W6) y golpe hidráulico por condensación (A3). Los 5 «fuera» sin encontrar son muestreo, choque de
vehículo, vibración/tornillos, cambio de calidad del crudo y arranque tras corte: ninguno sale de un DEXPI de proceso.

**Precisión.** El revisor aceptaría 7 de cada 10 filas a ciegas; una de cada 4 es ruido (filas genéricas de
flujo repetidas en válvulas y recipientes) y 5 estaban mal. Las 5 espurias eran bugs reales y se corrigieron:
- «Menos reflujo» en una estabilizadora sin reflujo → solo si hay producto líquido de domo.
- Congelamiento de agua en un tanque a 140 °C → la regla de invierno mira la T de la corriente que lleva agua, no la del amoniaco de dosificación.
- «Lado frío bloqueado» en un serpentín dentro de un tanque ventilado → no aplica a un lado atmosférico.
- Fuga «inflamable» de una solución acuosa de nitrato con 0.1 % de amoniaco → la marca inflamable exige > 1 % molar (la tóxica sigue a 100 ppm).
- Vacío en el tanque criogénico de hidrógeno (rompedor de vacío metería aire a −253 °C) → la regla de vacío no aplica a criogénicos.

Para el ruido: las válvulas entre dos equipos ya no repiten «sin flujo» y «flujo inverso» de sus vecinos, y
«más presión aguas abajo» solo sale si la válvula de verdad separa dos niveles de presión. Ruido 29 % → 24 %.

**Reglas nuevas** (sustancia y régimen térmico, válidas para cualquier flowsheet):
- punto de fusión desde `chemicals` (CRC primero; OPEN_NTBKM da 286 K para azufre, real 388 K): si un componente mayoritario funde arriba de 20 °C, tanques/bombas/tuberías ganan «solidificación» al perder calentamiento;
- `UNSTABLE` (nitrato de amonio, peróxido de hidrógeno, óxido de etileno, acetileno, nitrometano, nitroglicerina; CCPS/NFPA 400): «descomposición» con SIS de alta T, medio atemperado, inundación;
- tanque/recipiente de líquido a < 1.5 bar: «vacío» (vaciado, venteo tapado, enfriamiento) con PVRV;
- unidad con entrada < −40 °C y salida más caliente: «frente frío aguas abajo» a sellos y válvulas no criogénicos;
- vapor caliente + líquido frío del mismo componente en un equipo (ΔT > 50 K): «golpe hidráulico por condensación» con interlock de válvulas;
- columnas: «sin medio de calentamiento» (sin despojamiento, tóxicos a fondos) y «menos nivel de fondos» (blowby a la bomba).

**Verificación.** `tests/test_hazop_book.py`: bloque 17 sigue en 22/22; apartado 30/30; banderas de sustancia (CRC vs
OPEN_NTBKM, 0.1 % de amoniaco no es inflamable). Las dos rutas del grafo (DWSIM y DEXPI) siguen dando el mismo HAZOP en
los 13 flowsheets de referencia. Suite completa verde.

**Límites.** El 30/30 vuelve a no ser independiente. La precisión «útil» la juzga un modelo, no un ingeniero de
planta; el 19 % estricto es cota inferior porque los HAZOP publicados no son exhaustivos. Los números de operación
de los 5 DEXPI son ilustrativos (las fuentes casi nunca los dan). El siguiente conjunto apartado debe correrse igual:
referencia commiteada, reglas congeladas, etiquetado sin ver reglas.

Archivos: `notebooks/hazop_apartado/` (recall_blind.md, rows_blind.json, labels.yaml, labels_new.yaml, precision_blind.md,
precision.md, recall.md, rows.json, dexpi/). Corredor: `notebooks/19_hazop_apartado.py`.
