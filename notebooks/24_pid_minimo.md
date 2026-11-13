# 24 · P&ID mínimo desde el flowsheet y el HAZOP (DEXPI P&ID), bloque M10

Tool `pid_from_hazop` (`packages/pid/`). Toma un flowsheet resuelto de DWSIM (o un DEXPI Process) y su HAZOP
preliminar determinista. Escribe un P&ID mínimo como **DEXPI 2.0.1 P&ID** (XML + JSON) con pyDEXPI 2.1.0.

## Reglas (deterministas; cada elemento lleva la suya)

| Id | Regla | Fuente |
|---|---|---|
| R-L1 | Lazo de nivel en todo recipiente con inventario de líquido. Manipula la salida de líquido. Una por fase líquida; la acuosa es la interfase. | Luyben, Tyréus y Luyben (1998), paso 6 |
| R-P1 | Control de presión en todo recipiente con espacio de gas, sobre su salida de vapor. No aplica al tanque atmosférico. | Luyben et al. (1998), paso 6 |
| R-C1 | Columna: VT → presión, D → nivel del acumulador (MD), B → nivel de fondos (MB) | Skogestad (1997), Fig. 1 |
| R-C2 | Columna: lazo de temperatura de plato con el rehervidor (V) | Skogestad (2007) |
| R-F1 / R-F2 | Lazo de flujo en la descarga de bombas y compresores. En el compresor, FV es un marcador del elemento de capacidad. | Towler y Sinnott (2013), cap. 5 |
| R-T1 / R-T2 | Lazo de temperatura en calentadores, enfriadores y reactores con carga, actuando sobre la utilidad | Luyben et al. (1998), paso 8 |
| R-K1 | Si la salida de un recipiente alimenta una bomba o compresor con lazo de flujo, el inventario va en cascada a ese FIC | Seborg et al., control en cascada |
| R-V1 | Una válvula del flowsheet pasa a ser el elemento final del lazo que la necesita. Si ninguno la reclama, queda como HV. | ISA-5.1 |
| R-I1 | Aislamiento en los límites de batería y en succión y descarga de equipo rotatorio | Towler y Sinnott (2013), cap. 5 |
| H-PSV | Alivio donde una fila del HAZOP pide PSV, alivio térmico o disco de ruptura. Un dispositivo por punto protegido, con todos sus escenarios. | API 521 / API 520 / ASME VIII-1 |
| H-CHK | Válvula check donde una fila del HAZOP la pide | Fila del HAZOP |
| H-ALM | Alarma alta o baja para las desviaciones de L, P, T o F con severidad ≥ 3 | ISA-18.2 |
| H-TRIP | Disparo con sensor independiente para las desviaciones con severidad ≥ 4. El SIL sale de la LOPA, que está pendiente. | IEC 61511 |

Tags ISA-5.1 (LIC-101, LT-101, LV-101, LAH-101, LSHH-501, PSV-301). Las válvulas de bloqueo y las check no tienen
letra ISA: van como BV-nn y CHK-nn. Consignas, límites, puntos de disparo y presiones de ajuste quedan
«por definir (dato del usuario)». Las filas del HAZOP que piden algo que esta tool no dibuja (analizadores,
detectores de gas, procedimientos) salen en `no_instrumentadas`, cada una con su razón.

## Evidencia (`tests/test_pid.py`, 31 tests)

1. **Cobertura de reglas** en los 22 casos de referencia del HAZOP: 12 flowsheets de DWSIM y 10 HAZOP publicados
   escritos a mano como DEXPI.
   - Cada recipiente con líquido tiene un lazo de nivel por fase líquida, y cada columna tiene dos.
   - Cada fila «More pressure» tiene su PSV y cada «Reverse flow» su check.
   - Una fila tiene alarma o disparo **si y solo si** lo pide el umbral. Lo mismo para PSV y check: existen si y
     solo si la salvaguarda los pide.
   - No sobra nada: no hay lazos de nivel, flujo, presión o temperatura fuera de sus reglas, y las BV son exactamente
     las de límite de batería más las de equipo rotatorio.
   - Ninguna consigna está inventada.
   - Prueba de mutación: un PSV espurio y un lazo de nivel faltante hacen fallar la prueba.
2. **Ida y vuelta.** El XML y el JSON se recargan en pyDEXPI con el mismo resumen: equipos y boquillas, líneas,
   segmentos y tubos, ítems, funciones de instrumento con transmisor y actuador, lazos, sistemas de actuación,
   señales y notas. Además coinciden con la especificación: los mismos tags, lazos, ítems y equipos.
3. **Validación.**
   - **XSD oficial DEXPI 2.0.1** (`gitlab.com/dexpi/Specification`, commit `eef0f0c6`): los 22 P&ID son válidos.
     También lo son los DEXPI Process de `export_flowsheet`, que era un pendiente del bloque 15.
   - **Reglas DEXPI de pyDEXPI** (atributos obligatorios, conectividad y orden de segmentos, un instrumento por
     lazo, ...): 0 avisos.
   - Alcance del XSD: es estructural (Model/Object/Components/Data). La conformidad de clases y atributos la dan las
     clases de pyDEXPI generadas de DEXPI 2.0.1 y su validador. El chequeo contra `Plant.xml` oficial queda
     pendiente porque `data.dexpi.org` no respondió.
4. **Estructura publicada: Skogestad (1997), Fig. 1, configuración LV**, con el lazo de temperatura de Skogestad
   (2007). La figura del PDF del autor es legible como texto: «VT PC», «MD LC → D», «MB LC → B». En las 5 columnas
   (2 de DWSIM, 3 de HAZOP publicados) las reglas dan exactamente cuatro lazos: PIC con el condensador, LIC del
   acumulador con D, LIC de fondos con B (o en cascada al FIC de la bomba de fondos) y TIC con el rehervidor.
   Coincide con la figura.

## Ejemplo: estabilizadora de condensado (Marhavilas, HAZOP publicado)

`exports_ejemplo/marhavilas_estabilizadora_pid.xml`: 8 equipos, 20 líneas, 35 lazos (59 funciones) y 47
elementos de tubería; `verified = true`. Extracto:

| Lazo | Variable | Equipo | Funciones | Elemento final | Descripción |
|---|---|---|---|---|---|
| 103 | nivel | V-101 | LIC-103, LAH-103, LAL-103 | LV-103 | nivel de V-101; salida de líquido 2 |
| 104 | nivel | V-101 | LIC-104 | LV-104 | nivel de interfase (fase acuosa 2w); responde a la fila «water carryover» |
| 107 | nivel | V-102 | LIC-107, LAL-107, LAH-107 | cascada → FIC-101 | fondos de la columna con B vía la bomba P-101 |
| 111 | presión | V-103 | PIC-111, PAH-111 | cascada → FIC-102 | presión del separador con el compresor C-101 |
| 501 | nivel | V-101 | LSHH-501 | SIS por definir | disparo con sensor independiente |

## Pendiente

- Dibujo: pyDEXPI solo dibuja un `Diagram` con geometría. Generar el acomodo es el hueco de «layout y ruteo» (fase 3).
- Validar tipos y propiedades contra los modelos oficiales `Core.xml` y `Plant.xml` cuando `data.dexpi.org` responda.
- Bombas de desplazamiento positivo: la regla las trata como centrífugas. Columnas con L/D > 6: el nivel del
  acumulador debería ir con L. En ambos casos falta el dato en el grafo.
- El SIL de los disparos sale de la LOPA, que todavía no existe.
