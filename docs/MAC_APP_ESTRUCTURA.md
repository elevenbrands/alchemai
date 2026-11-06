# AlchemAI para Mac — estructura (gate 1, antes de Figma)

Decisión del owner, 2026-11-06: la pantalla de AlchemAI es una **app de Mac en SwiftUI**, no web.
Este documento fija la ruta técnica y el alcance del MVP. El diseño en Figma empieza cuando el owner elige la plantilla.

## 1. Qué es la app

Es la mesa de trabajo del ingeniero de procesos.
1. Escribe el problema en español: «separar etanol de agua hasta 80 % en el destilado».
2. Ve al agente armar el diagrama de flujo y correr cada herramienta en DWSIM.
3. Revisa corrientes, equipos y balances, cada número con su sello de evidencia.
4. Saca el HAZOP preliminar, el costo y la optimización, y exporta.

Regla del producto, igual que en el bench: **el motor es la verdad y la IA solo propone.** Todo número de la app dice de dónde salió (el simulador, el reporte del agente o una suposición) y si quedó verificado.

## 2. Arquitectura

```
┌───────────── AlchemAI.app (SwiftUI, Swift 6, macOS 26+) ─────────────┐
│ Vistas: Proyectos · Conversación · Diagrama · Corrientes · HAZOP ·  │
│         Costo · Ajustes                                              │
│ Cliente MCP (modelcontextprotocol/swift-sdk, Apache-2.0/MIT)         │
│ FluxoEngine (Swift, propio): previsualizaciones instantáneas         │
└──────────────┬───────────────────────────────────────────────────────┘
               │ stdio, JSON-RPC (MCP) + flujo de eventos del agente
┌──────────────▼──────────── proceso «alchemai serve» (Python, MIT) ───┐
│ 36 tools de packages/dwsim_client · agente packages/agent ·          │
│ HAZOP · costo · optimización · export SFILES/DEXPI                   │
└──────────────┬───────────────────────────────────────────────────────┘
               │ pythonnet (mismo proceso Python, DWSIM fuera de la app)
       DWSIM 10 (GPL-3), instalado aparte por el usuario
```

Decisiones:
- **El agente sigue en Python.** La app no reimplementa el bucle del agente. Así lo que mide el bench es exactamente lo que corre en la app. Swift solo pinta y controla.
- **Un proceso hijo, `alchemai serve`.** Expone las mismas tools por MCP para que la app edite a mano (cambiar una especificación y volver a resolver sin LLM). Agrega un método de sesión del agente que emite eventos: paso, llamada a herramienta, resultado, respuesta final con evidencia. Es una capa delgada sobre `packages/dwsim_mcp/server.py` y `packages/agent/`.
- **DWSIM nunca va dentro de la app.** Es GPL-3. La app detecta la instalación del usuario (hoy `~/Applications/DWSIM10`) o lo guía para bajarlo de la release oficial. La frontera de licencia es la misma de siempre: proceso separado.
- **Python embebido.** La app lleva un intérprete propio con `uv` (python-build-standalone) y las dependencias fijadas por `uv.lock`. El usuario no instala Python.
- **Modelo del agente.** Para el owner sirve la suscripción vía `claude -p`, como en el bench. Para otros usuarios, su llave de Anthropic en el Llavero de macOS, nunca en archivos. Opus por defecto, según el bloque 29.
- **FluxoEngine** se reutiliza como paquete Swift para cálculos instantáneos sin arrancar DWSIM: flash, propiedades y vista previa de una columna corta. El diagrama de Fluxo sirve de punto de partida para el lienzo.

## 3. Distribución

**Recomendación: fuera de la Mac App Store, con Developer ID y notarización (DMG).** Hay tres razones.
1. La app arranca un intérprete de Python y el runtime de .NET de DWSIM fuera de su paquete. El sandbox obligatorio de la Mac App Store lo bloquea o lo vuelve frágil (permisos sobre carpetas del usuario, JIT de .NET).
2. DWSIM es GPL-3, y el GPL choca con los términos de la App Store. Aunque DWSIM no va en el paquete, la app no sirve sin él.
3. El cobro sería por licencia o suscripción propia (Stripe o Paddle), no por compra dentro de la app. **El precio y la pasarela son decisión del owner.**

La cuenta de Apple Developer ya existe. Developer ID y notarización no cuestan extra.

## 4. Repositorio

- App: repo **privado** nuevo `apps/alchemai-mac` (1 app = 1 repo, regla de la fábrica). Es el producto que se cobra.
- Motor: sigue en `alchemai`, **público** y MIT. La app lo fija por versión (tag) y lo empaqueta.
- La fábrica no tiene plantilla para macOS: `new_app_factory.sh` solo acepta iOS (swiftui o flutter). Se adapta a mano el kit de builds (Makefile, `build.sh`, xcodegen) con destino macOS.

## 5. Inventario de pantallas (entrada para Figma)

| # | Pantalla | Contenido | Estados |
|---|---|---|---|
| 1 | Bienvenida y motor | Detecta DWSIM y Python embebido; guía de instalación de DWSIM | encontrado · falta DWSIM · versión vieja |
| 2 | Proyectos | Lista de casos en la barra lateral; nuevo caso desde texto o desde ejemplo del bench | vacío · con casos |
| 3 | Conversación de diseño | Enunciado, pasos del agente en vivo, cada herramienta con entrada y salida plegables, respuesta con sello de evidencia | pensando · llamando herramienta · terminado · infactible detectado |
| 4 | Diagrama de flujo | Lienzo con equipos y corrientes, generado del grafo SFILES; clic abre el inspector | resuelto · sin resolver · error de convergencia |
| 5 | Inspector | Equipo o corriente: T, P, flujo, composición y especificaciones editables; botón «Resolver» sin LLM | verificado · editado sin resolver |
| 6 | Corrientes y balances | Tabla de corrientes; cierre de masa, componentes y energía por equipo | cerrado · abierto con residuo |
| 7 | HAZOP preliminar | Tabla nodo · palabra guía · causa · consecuencia · salvaguarda; filas de regla vs propuesta del LLM; exportar a Excel | — |
| 8 | Costo y optimización | Costo por equipo, costo total anual, curva del óptimo (también la entera de etapas) | — |
| 9 | Exportar | SFILES 2.0, DEXPI, Excel, PDF del reporte | — |
| 10 | Ajustes | Modelo y esfuerzo, llave en el Llavero, ruta de DWSIM, unidades | — |
| 11 | Licencia o suscripción | Solo si el owner define cobro | — |

## 6. Alcance del MVP

Entra:
- pantallas 1 a 7 y 10;
- un caso de extremo a extremo: enunciado → agente → diagrama → balances → HAZOP;
- abrir los 50 casos del bench como ejemplos.

Queda para después:
- costo y optimización en pantalla (las tools ya existen);
- exportar a PDF;
- cobro;
- edición gráfica del diagrama arrastrando equipos.

## 7. Riesgos

- **`solve()` de DWSIM a veces no regresa**: columna con PR en mezcla polar. La app necesita un botón «Detener» que mate el proceso, igual que `_kill_group` del arnés.
- **Tamaño del paquete**: Python más las dependencias científicas pesan cientos de MB, sin contar DWSIM. Se mide en el primer build.
- **Primer arranque de DWSIM** (pythonnet + .NET): hay que medir el tiempo y mostrar progreso.

## 8. Lo que decide el owner

1. La plantilla de Figma Community, entre las 5 propuestas.
2. Distribución fuera de la App Store, que es la recomendación.
3. Cobro y precio (puede esperar al final del MVP).

## 9. Diseño en Figma (2026-11-06, pendiente de aprobación del owner)

Archivo: https://www.figma.com/design/Cb75xN1JHDXuPpUOzz37VQ (registro en `.figma/project.json`, capturas en `docs/design/mac/`).
Plantilla elegida por el owner: kit oficial **macOS 27** como librería enlazada, más los patrones del **AI Agent UI Kit** para el panel del agente (solo referencia visual, sin activos copiados). Moodboard en la página 01.

Pantallas: Bienvenida y motor, Conversación (claro y oscuro), Diagrama de flujo con inspector, Corrientes y balances, HAZOP preliminar y Ajustes del agente.
Componentes propios: insignia de evidencia (verificado, sin verificar, propuesta de la IA, suposición, infactible), fila de llamada a herramienta, botón de barra, estado del motor y barra lateral.
Todos los números salen de una corrida real de DWSIM: columna etanol-agua NRTL, 32 etapas, alimentación en la 22, R 1.2665 (x_D 0.8015, Qc 2 233.6 kW, Qr 2 766.6 kW, 7 filas de HAZOP por regla).

Notas para SwiftUI:
- Los íconos de la barra lateral son glifos del kit usados como marcadores; en la app van SF Symbols.
- La fuente monoespaciada en Figma es JetBrains Mono porque SF Mono no está disponible; en la app es SF Mono.
- En modo oscuro, el kit pierde los textos de la barra lateral al cambiar de modo; la captura oscura usa una barra desacoplada. En SwiftUI la barra es nativa y no tiene este problema.
- Las desviaciones del HAZOP salen del motor en inglés; la app las traduce al mostrarlas.
- El ícono es un marcador: la marca pasa por el Premium Brand Gate después de aprobar este diseño.
