# 10 · HAZOP preliminar sobre DEXPI (bloque 16)

Pregunta: ¿puede AlchemAI hacer el HAZOP preliminar de un proceso que llega como archivo DEXPI, sin haberlo
simulado? Es el hueco sin competidor del plan (Fase 2): no hay HAZOP asistido abierto sobre un formato estándar.

## Qué se construyó

- `packages/hazop/dexpi_graph.py`: lee un modelo DEXPI Process (XML de DEXPI o JSON de pyDEXPI) y arma el mismo
  grafo de proceso que el lector de DWSIM, así que las reglas del HAZOP corren sin cambios. Tipo de equipo desde la
  clase DEXPI (Compressing → compresor, Distilling → columna, ReactingChemicals → reactor…); T, P y flujo másico
  desde la corriente, con conversión de °C, °F, bar, kPa, MPa, kg/h y kg/min; composición desde el MaterialState.
  Lo que el archivo no trae se reporta en `missing` y nunca se inventa.
- El exportador DEXPI del bloque 15 ahora escribe la composición en la forma estándar de DEXPI: una plantilla de
  componentes (MaterialTemplate) y, por corriente, un MaterialState con fracciones y flujos molares por componente.
  La fracción vapor va en la descripción del estado.
- Tool nueva `hazop_from_dexpi(path)` en MCP y agente (20 tools).
- `packages/hazop/dexpi_examples.py`: un tren de compresión de gas húmedo armado a mano en DEXPI, con tags de
  ingeniería, °C, bar y kg/h, como lo escribiría otra herramienta.

## Resultados

| Prueba | Resultado |
|---|---|
| 12 flowsheets de referencia + gas húmedo: HAZOP desde DWSIM vs desde su DEXPI (XML y JSON) | **13/13 idénticos fila por fila**, sin datos faltantes |
| DEXPI hecho a mano, sin simulador (compresor K-100, enfriador E-100, separador V-100) | 13 filas; unidades convertidas bien (30 bar, 105 °C = 378.15 K, 18 000 kg/h = 5 kg/s); marca de inflamable e hidratos encendidas por la composición |
| Mismo archivo sin composición | 9 filas; `missing` lista las 5 corrientes; nada se marca inflamable ni con hidratos |

El HAZOP de DWSIM excluye en la comparación las filas del bloque Recycle: es un bloque lógico del simulador, no
un equipo, y no existe en DEXPI. Es una mejora, no una pérdida.

## Hueco de regla encontrado y cerrado

El ejemplo a mano saca gas húmedo del enfriador a 15 °C y 64 bar, condición clásica de hidratos, y el HAZOP no tenía
la desviación «menos temperatura» en enfriadores. Se agregó **«Less temperature (overcooling)»** para todo enfriador:
causas (utilidad fría a máxima capacidad, bajo flujo, baja T ambiente), consecuencias (hidratos con la T y P del
caso cuando hay agua y formadores, condensación de pesados, congelamiento bajo 0 °C) y salvaguardas (inhibidor
MEG/metanol o deshidratación, alarma TAL). Aplica igual al HAZOP desde DWSIM: el gas húmedo del test ahora también
la tiene en E-1.

## Límites

- Probado con archivos que escribe AlchemAI y con uno hecho a mano con pyDEXPI. **No se ha probado con un DEXPI de
  otra herramienta** (por ejemplo, un P&ID exportado de un CAD). Esos suelen ser DEXPI de planta (equipos y tuberías,
  nivel P&ID), no DEXPI Process; leer el nivel P&ID es otro trabajo.
- Sin carga térmica en el archivo, la exotermicidad de un reactor no se evalúa y se avisa en `missing`.
- Sigue sin validarse el XML contra el XSD oficial de DEXPI.
- Es un HAZOP **preliminar** por reglas: no sustituye la sesión con el equipo multidisciplinario.

Ejemplo completo en `notebooks/exports_ejemplo/compresion_gas_hecho_a_mano.xml` con su HAZOP en `.md` y `.xlsx`.
Reproducir: `uv run pytest -q tests/test_hazop_dexpi.py` (16 pruebas).
