# Post 1 — Lanzamiento de AlchemAI (borrador para LinkedIn, español)

> Estado: BORRADOR 2026-11-05. No publicar sin el OK del owner (también decide si el repo se abre primero).

---

Llevo un mes construyendo **AlchemAI**: un agente de IA para diseño de procesos químicos que **no adivina**.

El problema con «la IA en ingeniería» es que un modelo de lenguaje responde con seguridad aunque esté mal.
En ingeniería de procesos eso no sirve: un balance que no cierra no es una opinión, es un error.

Así que le puse reglas:

1. **El simulador es la verdad.** El agente arma el flowsheet y lo corre en DWSIM (motor abierto, el mismo tipo de
   cálculo que Aspen o HYSYS). Solo reporta lo que converge y cierra balances de masa y energía.
2. **Referencias antes de correr.** Escribí las respuestas de 50 casos de diseño y las guardé en git *antes* de que
   el agente los viera. Lo mismo con 10 HAZOP publicados (Lawley, CCPS, Crowl, CSB Texas City, Williams y cinco más
   que las reglas nunca vieron).
3. **Nada de datos inventados.** Constantes con fuente o entrada del usuario. Punto.

Resultados, con evidencia reproducible en el repo:

- 50 casos de diseño (flash, columnas, reactores cinéticos, reciclos, absorbedores, tuberías, costos, optimización):
  **100 %** con Claude, incluyendo 15 «trampas» con especificaciones imposibles que el agente **detecta** en vez de
  resolver, con 0 falsas alarmas.
- HAZOP preliminar automático sobre el grafo del proceso: en 5 HAZOP publicados que las reglas no habían visto,
  encontró el **83 %** de los escenarios a ciegas; un revisor independiente aceptó 3 de cada 4 filas y no halló
  ninguna errónea tras la corrección.
- Optimización con límites del HAZOP como restricciones duras: compresión en 3 etapas igual a la teoría ±1 %;
  columna etanol-agua con el reflujo óptimo en 1.12 × R_min (justo donde dicen los libros: 1.1-1.5).
- Transporte, transferencia de masa y costos verificados contra DWSIM, contra los artículos originales (Onda 1968,
  Stichlmair 1989) y contra IDAES (Seider).

Lo que **no** hace: no reemplaza a un equipo de HAZOP, no tiene UI todavía (es un servidor MCP: se usa desde
Claude Desktop, Claude Code o Cursor) y los precios de los ejemplos son ilustrativos.

Es código abierto (MIT; DWSIM corre aparte bajo su propia licencia): [enlace al repo]

Si eres ingeniero de procesos o das clase de diseño, me interesa que lo rompas. ¿Qué caso le pondrías?

#IngenieríaQuímica #ProcessEngineering #IA #DWSIM #HAZOP #ChemE

---

Notas para el owner:
- Sustituir «[enlace al repo]» por https://github.com/elevenbrands/alchemai al abrir el repo.
- Cifras tomadas de `BENCHMARK.md` al 2026-11-05 (bloques 13-26). Si se publica después, actualizar.
- Posible imagen: la tabla de `notebooks/20_variables_enteras.md` (costo vs etapas) o la de recall/precisión del
  notebook 19.
