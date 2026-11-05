# 12 · Fenómenos de transporte: hidráulica y área de intercambiadores (bloque 18, 2026-11-05)

**Pregunta del owner.** ¿Cubrimos fenómenos de transporte? Hasta el bloque 17, no. Los equipos de DWSIM que usaba el
agente resuelven balances de materia, energía y equilibrio, pero ninguna herramienta calculaba caída de presión en
tubería, diámetros, coeficientes de película o áreas. Sin eso no hay dimensionamiento ni costo (fase 3 del plan).

**Qué se agregó.**
- `packages/transport/hydraulics.py`, funciones puras sobre `fluids` y `ht` (MIT, Caleb Bell):
  - Darcy-Weisbach con Colebrook resuelto exacto (Clamond), y 64/Re en laminar;
  - accesorios ΣK y carga estática;
  - diámetro nominal cédula 40 por velocidad (ASME B36.10M, sin 1¼, 2½, 3½ ni 5 in);
  - LMTD con factor F de Fakheri para 1-N corazas;
  - Gnielinski para el coeficiente del lado tubos, y U con ensuciamiento.
- 5 herramientas del agente en `packages/dwsim_client/tools_transport.py`. Toman densidad, viscosidad, k y Cp de la corriente resuelta en DWSIM, en la fase correcta, y rechazan corrientes en dos fases.
- Nuevo equipo `pipe` en `add_unit`: el Pipe Segment de DWSIM, con Beggs-Brill, adiabático y en acero comercial. Este sí maneja dos fases.

## A · Caída de presión: herramienta propia vs tubería de DWSIM (misma corriente)

| Caso | Re | Régimen | v (m/s) | ΔP fricción propio (Pa) | ΔP fricción DWSIM (Pa) | Error | ΔP estático propio / DWSIM (Pa) |
|---|---|---|---|---|---|---|---|
| P1 agua 4 in | 124,332 | turbulento | 1.220 | 14,190 | 14,261 | -0.49 % | 0 / 0 |
| P2 agua 60 °C 2 in, sube 10 m | 155,958 | turbulento | 1.409 | 77,675 | 77,941 | -0.34 % | 96,441 / 96,372 |
| P3 agua 6 in 500 m | 330,201 | turbulento | 2.149 | 125,746 | 126,241 | -0.39 % | 0 / 0 |
| P4 n-hexano líquido 3 in | 826,849 | turbulento | 4.796 | 259,192 | 259,585 | -0.15 % | 0 / 0 |
| P5 metano gas 20 bar 6 in | 3,556,078 | turbulento | 20.049 | 52,796 | 53,509 | -1.33 % | 0 / 0 |
| P6 glicerol laminar 2 in | 12 | laminar | 0.185 | 110,251 | 110,179 | +0.07 % | 0 / 0 |

Las seis quedan dentro de 1.4 %. El gas es el peor caso, porque DWSIM integra el cambio de densidad por incrementos y la
herramienta propia la toma constante; por eso avisa cuando ΔP pasa del 10 % de la presión de entrada. El caso laminar
coincide además con Hagen-Poiseuille exacto (test).

## B · Área: DWSIM resuelve con U y A dados; la herramienta debe recuperar A

| Caso | Q DWSIM (kW) | T salida caliente / fría (K) | LMTD (K) | Área dada a DWSIM (m²) | Área recuperada (m²) | Error |
|---|---|---|---|---|---|---|
| H1 agua/agua | 1171.5 | 325.04 / 314.02 | 34.45 | 40.0 | 40.004 | +0.01 % |
| H2 n-hexano/agua | 1218.9 | 331.17 / 320.54 | 45.04 | 60.0 | 60.131 | +0.22 % |

## Lo que no se validó
- **Coeficiente de película contra DWSIM.** Se intentó con la tubería de DWSIM en modo «estimar U». Reportó h interno = 0 y enfrió el agua a temperatura ambiente en 50 m, así que no sirve de referencia. Gnielinski queda validado solo por la biblioteca y por rango (test).
- **Lado coraza.** Kern y Bell-Delaware no están. U sigue siendo un dato o sale de `overall_U` con h supuestos.
- **Transferencia de masa con coeficientes.** Columnas de relleno, HTU/NTU y kLa no están.
- **Caída de presión en dos fases propia.** Solo existe la de DWSIM, con Beggs-Brill.

## Trampas de la Fluent API encontradas
- `WithGlobalUA(x)` es **U·A en W/K**; DWSIM calcula U = UA/área. Pasar 850 con 40 m² da U = 21 W/m²K.
- `PipeSection(índice, "Tubulaosimples", 1, incrementos, "Aço Comum", L_m, Δz_m, DE_in, DI_in)` lleva los diámetros en **pulgadas**. Además hay que poner `Profile.Status = OK` o DWSIM responde «perfil hidráulico no definido».
- Las propiedades de transporte viven en `Phases[1]` (líquido) y `Phases[2]` (vapor).

Reproducir: `PYTHONPATH=. uv run python notebooks/12_fenomenos_transporte.py A` (o `B`); salidas en
`notebooks/12_tuberias_vs_dwsim.jsonl` y `notebooks/12_intercambiadores_vs_dwsim.jsonl`.

## C · Bench v3: el agente con las herramientas nuevas (Opus, `claude -p`, 25 tools)

| Tarjeta | Corrida 1 | Repetición |
|---|---|---|
| Área de un enfriador 1-2 (Q, LMTD, F, área) | **1.00** en 24 s | — |
| Diámetro de línea de n-hexano por velocidad | **1.00** en 33 s (8 in, 1.417 m/s) | — |
| Caída de presión en una línea que sube 15 m | **0.00**: agotó los 600 s sin respuesta | **1.00** en 33 s |

En la corrida 1 la tarjeta de tubería se colgó. No hubo traza y el cuelgue no se reprodujo. El arnés mató el grupo de
procesos y no quedaron servidores huérfanos. Para la próxima vez, el servidor MCP registra cada llamada con su duración
si se define `ALCHEMAI_MCP_TRACE=<archivo>`. En la repetición, la traza muestra cómo resolvió el agente: armó la
tubería en DWSIM con el equipo `pipe` y la verificó con `pipe_hydraulics`; las dos coinciden.

Como en el bloque 14, las 3 tarjetas quedan al 100 % entre corridas, no en una sola pasada.
Archivos: `notebooks/bench_runs/v3_transporte_cc_opus_run1.*`, `..._run2_dp.log`, `..._run2_dp_mcp_trace.jsonl`.
