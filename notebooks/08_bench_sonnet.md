# 08 · Bench completo con Sonnet (bloque 14)

Pregunta: ¿un modelo más chico que Opus resuelve las 39 tarjetas con las mismas 18 tools y el mismo
prompt? Hasta aquí todo el bench se había medido solo con Opus (100 % en v0, v1, v1b y v2).

Comando: `ALCHEMAI_CC_MODEL=sonnet uv run python packages/bench/runner.py --agent claude-code --version all --out <jsonl>`.

## Corrida 1 (prompt de bloque 13, sin cambios) — `bench_runs/all_cc_sonnet_run1.jsonl`

| Versión | Sonnet | Opus (corridas previas) | Tiempo Sonnet |
|---|---|---|---|
| v0 (20 factibles) | 92.2 % (16/20 perfectas) | 100 % | 892 s |
| v1 (10 trampas físicas) | 100 % (10/10) | 100 % | 451 s |
| v1b (5 trampas de especificación) | 100 % (5/5) | 100 % | 157 s |
| v2 (4 polares, elección de paquete) | 90.6 % (3/4 perfectas) | 100 % | 120 s |
| **Total** | **95.1 % sobre 39** | 100 % | 1 599 s (41 s/tarjeta vs 29 s Opus) |

Las 15 trampas se detectaron todas y en las 4 polares Sonnet eligió NRTL solo y lo escribió en las
notas. **No hace falta la «regla 10» de selección de paquete.**

Fallos de la corrida 1 (5 tarjetas):

| Tarjeta | Cantidad | Sonnet | Esperado | Causa |
|---|---|---|---|---|
| compresor | Q_cooler_kW | −79.34 | 79.34 | signo |
| fug | Qc_kW | −1917.69 | 1917.66 | signo |
| columna_rigurosa | Qc_kW | −2955.89 | 2955.89 | signo |
| polar_columna_metanol_agua | Qc_kW | −3548.24 | 3548.24 | signo |
| reciclo | T_mixed_K | 400.0 | 373.12 | leyó la salida del calentador, no la del mezclador |

Los cuatro fallos de signo tienen la misma raíz: la regla 5 del prompt decía «el calor retirado es
NEGATIVO» y Sonnet la aplicó al pie de la letra sobre condensadores y enfriadores, aunque DWSIM los
reporta como magnitudes positivas (y las tarjetas esperan lo que reporta DWSIM). Opus nunca negó el
número; Sonnet sí. Los valores absolutos eran correctos en los cuatro casos. En reciclo el enunciado
no decía qué temperatura se pedía (solo «flujos en estado estacionario») y el nombre `T_mixed_K` no le
bastó.

## Cambios tras la corrida 1

1. **Regla 5 reescrita**: «reporta el signo que devuelve la TOOL, nunca lo inviertas»; se listan las
   convenciones de DWSIM (Cooler, Qc y Qr positivos; Heater usado para enfriar, válvula y reactor
   exotérmico negativos; potencia positiva) y se aclara que «calor retirado» en el enunciado no pide
   negar. Además: «lee cada cantidad de la corriente o unidad a la que apunta su nombre (T_mixed_K =
   salida del mezclador) y di en notas cuál leíste».
2. **Enunciado de `reciclo`** ahora pide explícitamente «flujos de reciclo y purga en estado estacionario
   y temperatura de la corriente mezclada (salida del mezclador, antes del calentador)». Es la única
   tarjeta cuyo yaml cambió; las otras 38 se regeneraron byte a byte iguales.

## Corrida 2 (regla 5 nueva) — `bench_runs/v0_cc_sonnet_run2.jsonl` · `bench_runs/v2_polares_cc_sonnet_run2.jsonl`

| Versión | Corrida 2 | Tiempo |
|---|---|---|
| v0 | 92.5 % (17/20 perfectas) | 1 352 s |
| v2 | **100 % (4/4)** | 118 s |

Los cinco fallos de la corrida 1 quedaron al 100 % (compresor, fug, columna_rigurosa, polar_columna
con Qc positivo; reciclo con T_mixed_K = 373.12 K). Aparecieron dos fallos nuevos, ninguno del modelo:

| Tarjeta | Qué pasó | Causa | Arreglo |
|---|---|---|---|
| diseno_columna_etapas | 0 % tras agotar los 600 s | DWSIM se colgó en una columna (no determinista: la misma tarjeta salió en 44 s en la corrida 1). El runner mató a `claude -p` pero no al servidor MCP, que quedó huérfano al 98 % de CPU. Se encontraron otros dos huérfanos iguales del 2 de noviembre (54 h de CPU cada uno). | `run_claude_p` ahora lanza `claude -p` en su propio grupo de procesos y al vencer el tiempo mata al grupo entero (`_kill_group`, SIGTERM → SIGKILL). Verificado: grupo de 3 procesos muerto en 0.17 s. |
| reactor_conversion | y_CH4 = 1.5e-8 vs 0.0 esperado | Tolerancia relativa 1 % sobre un valor esperado 0 = banda de ancho cero. DWSIM deja ~1e-8 del reactivo con 100 % de conversión; Opus y la corrida 1 reportaron 0.0 redondeado. | `build_cards`: toda cantidad esperada exactamente 0 lleva banda absoluta 1e-4 (`ZERO_ABS_TOL`). Solo cambió `reactor_conversion.yaml`. |

## Corrida 3 — las dos tarjetas repetidas (`bench_runs/v0_cc_sonnet_run3_refallos.jsonl`)

`diseno_columna_etapas` 100 % en 53 s (R_min 1.2807, N_min 9.161, N 16.91, etapa 8.59, D 50.0) y
`reactor_conversion` 100 % en 34 s (y_CH4 = 4e-17 dentro de la banda). Sin servidores huérfanos al terminar.

## Conclusión

- **Sonnet resuelve el bench**: 15/15 trampas, 4/4 polares eligiendo NRTL solo, y las 20 factibles al 100 %
  entre las corridas 2 y 3 tras dos arreglos del prompt y dos del bench/arnés. No hizo falta la regla 10 de
  selección de paquete.
- Cada fallo tuvo causa identificable y ninguna fue «el modelo no sabe ingeniería»: una regla ambigua de
  signo, un enunciado que no pedía la temperatura, una banda de tolerancia cero y un cuelgue de DWSIM.
- Caveat honesto: las 39 no se han vuelto a correr en una sola pasada con el prompt nuevo (v1/v1b solo tienen
  la corrida 1, que ya era 15/15 y la regla 5 no las toca). Sonnet tarda ~40 % más que Opus por tarjeta.
- Límite abierto: un `solve()` de DWSIM puede no regresar nunca; el arnés ya no deja huérfanos pero el
  servidor MCP no tiene timeout propio por llamada (sería un hilo .NET sin forma limpia de interrumpir).

Reproducir: `ALCHEMAI_CC_MODEL=sonnet uv run python packages/bench/runner.py --agent claude-code --version all --out <jsonl>`.
