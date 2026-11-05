# 09 · Exportar el flowsheet a SFILES 2.0 y DEXPI (bloque 15)

Pregunta: ¿puede AlchemAI entregar el flowsheet que resolvió en formatos abiertos que otras herramientas lean,
sin perder lo que DWSIM calculó? Es la semana 8 del plan adelantada (flowsheets como texto y grafo).

- **SFILES 2.0** (Vogel et al. 2023): el flowsheet en una línea de texto. Es el formato con el que se entrenan los
  modelos de autocompletado de flowsheets (TU Delft) y sirve para comparar topologías.
- **DEXPI Process** (modelo de proceso de DEXPI 2.0, nivel PFD): pasos de proceso con puertos y las corrientes entre
  ellos, con T, P y flujo másico resueltos. Es el puente hacia P&ID y hacia otras herramientas de ingeniería.

Implementación: `packages/export/flowsheet_export.py`, con `SFILES2` 1.2.0 y `pyDEXPI` 2.1.0 (ambas MIT). La fuente
es el mismo grafo que lee el HAZOP (`hazop.graph.build_graph`), así que la exportación describe exactamente lo que
DWSIM resolvió. Tool nueva `export_flowsheet` en el servidor MCP y en el agente (19 tools).

Mapeo de equipos DWSIM → SFILES → DEXPI:

| DWSIM | SFILES | DEXPI Process |
|---|---|---|
| Mixer (NodeIn) | mix | Mixing |
| Splitter (NodeOut) | splt | SplittingMaterial |
| Valve | v | RegulatingFlow |
| Vessel / Separator | flash | SeparatingByFlash |
| Heater / Cooler | hex | SupplyingThermalEnergy / RemovingThermalEnergy |
| HeatExchanger (2 lados) | hex con etiquetas 1_in/1_out, 2_in/2_out | ExchangingThermalEnergy |
| Pump / Compressor / Expander | pp / comp / turb | Pumping / Compressing / ProcessStep |
| DistillationColumn / ShortcutColumn | dist con tout/bout | Distilling |
| Reactores (conversión, equilibrio, Gibbs, CSTR, PFR) | r | ReactingChemicals (TANK / TUBULAR / sin especificar) |
| Recycle de DWSIM (bloque lógico) | se elimina y las dos corrientes se unen | — |
| Corriente sin equipo arriba / abajo | raw-k / prod-k | Source / Sink |

## Resultado sobre los 12 flowsheets de referencia

Todos se exportan y se vuelven a leer igual: SFILES re-parseado con el parser de SFILES2 es isomorfo al grafo
original (nodos comparados por tipo) y el DEXPI XML recargado con pyDEXPI da los mismos pasos, conexiones y
temperaturas.

| Flowsheet | SFILES 2.0 |
|---|---|
| mezclador | `(raw)(mix)<&\|(raw)&\|(prod)` |
| valvula_jt | `(raw)(v)(prod)` |
| separador | `(raw)(flash)[(prod)](prod)` |
| intercambiador | `(raw){1_in}(hex){1}{1_out}(prod)n\|(raw){2_in}(hex){1}{2_out}(prod)` |
| bomba / compresor_politropico | `(raw)(pp)(prod)` / `(raw)(comp)(prod)` |
| reactor_conversion / reactor_equilibrio | `(raw)(r)[(prod)](prod)` |
| columna_rigurosa / depropanizadora_fug | `(raw)(dist)[{bout}(prod)]{tout}(prod)` |
| reciclo | `(raw)(mix)<1(hex)(splt)1(prod)` (el `<1 … 1` es el lazo de reciclo) |
| compresor_enfriador | `(raw)(comp)(hex)(prod)` |

Ejemplos de archivos en `notebooks/exports_ejemplo/` (reciclo, columna etanol-agua, intercambiador): `.sfiles`,
`.xml` (DEXPI), `.json` (pyDEXPI) y `.graph.json` (resumen legible con T, P y flujo por corriente).

## Límites anotados

- **El XML no se validó contra el esquema XSD oficial de DEXPI**: pyDEXPI no lo incluye. La verificación es ida y
  vuelta con el cargador de pyDEXPI. Validar contra el XSD queda pendiente.
- DEXPI lleva T, P y flujo másico por corriente y la carga térmica del equipo en su descripción; **la composición por
  corriente no se exporta todavía** (va en el `.graph.json`), y las corrientes de energía de DWSIM no tienen equipo
  de origen, así que no se modelan como `EnergyFlow`.
- SFILES describe solo topología: no lleva condiciones de operación por diseño del formato.
- Probado con los equipos de los 12 flowsheets de referencia; absorción y expansor están mapeados pero sin caso propio.

Reproducir: `uv run pytest -q tests/test_export.py` (14 pruebas). Desde el agente o MCP:
`export_flowsheet(flowsheet_id)` después de `solve()`; escribe en `exports/<flowsheet_id>/` (ignorado por git).
