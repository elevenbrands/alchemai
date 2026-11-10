# Mapa de motores — qué resuelve cada área de la ingeniería química y con qué

> Objetivo del owner (2026-11-08): **una sola aplicación** donde el ingeniero le pregunta a su LLM y el agente
> resuelve con la herramienta correcta: calor, masa, momento, balances, reacción, separación, costos, seguridad y
> diseño de planta. Este mapa dice, por área, qué motor responde, si ya está dentro de AlchemAI, cómo entra y qué
> falta. Es el plan de cobertura; `APP.md` sigue siendo la verdad de lo construido.

**Principio que no cambia:** el agente nunca calcula; elige una tool determinista, el motor da el número y el
sello (`Verificado` / `Sin verificar` / `Propuesta IA` / `Infactible`) sale de la evidencia del motor, no del texto.
Cada motor nuevo entra con el mismo patrón que DWSIM: (1) caso de referencia con respuesta conocida, (2) tool con
entradas SI y salida JSON, (3) la tool reproduce la referencia en un test, (4) si la licencia es GPL, proceso aparte.

## Leyenda

| Marca | Significado |
|---|---|
| ✅ | Tool en AlchemAI con test que reproduce una referencia (número de tools entre paréntesis) |
| 🧪 | Motor instalado y probado en este Mac con un caso verificado, **sin tool** todavía |
| 📦 | Instalado en el entorno de AlchemAI (`uv sync`), sin usar aún |
| ⬜ | Motor elegido, sin instalar |
| 🚫 | Sin opción abierta razonable o decisión que es del owner |

Estado verificado el 2026-11-09: **53 tools**, DWSIM 10.2.12 (`~/Applications/DWSIM10`), thermo 0.6.1, chemicals 1.5.2,
fluids 1.3.1, ht 1.2.0, cantera 3.2.0, idaes 2.13.0 + pyomo 6.10.1 (**sin ipopt**), omlt 1.2.2, biosteam 2.54.0,
SFILES2 1.2.0, pydexpi 2.1.0; fuera del entorno: OpenFOAM v2606 (`openfoam`), FEniCSx dolfinx 0.11.0
(`~/mamba/envs/fenicsx`). Repo hermano `transporte-lab` guarda los casos de CFD/FEM.

## 1 · Termodinámica y propiedades

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Equilibrio de fases (PR, SRK, NRTL, UNIQUAC…), flash PT/PH/PS, puntos de burbuja y rocío, azeótropos | ✅ (`list_property_packages`, `read_stream`, `sweep`) | DWSIM | GPL-3 · proceso aparte (pythonnet) | VLLE en polares no lo hace DWSIM-NRTL ni Fluxo (anotado bloque 12) |
| Propiedades de compuestos puros (Psat DIPPR, Tfus, ΔHf, ω…) | ✅ parcial (dentro de HAZOP y de Fluxo) | `chemicals`, `thermo` | MIT · en proceso | Tool pública de propiedades (`compound_properties`) para que el agente consulte sin armar flowsheet |
| Flash independiente de DWSIM (segunda opinión, VLLE con Michelsen) | 📦 | `thermo.Flash`, `FlashVLN` | MIT · en proceso | Segunda opinión termodinámica automática cuando DWSIM y `thermo` difieren |
| Regresión de parámetros binarios con datos experimentales | ⬜ | DWSIM (regresión) o `thermo` + scipy | — | No explorado; depende de datos del usuario |

## 2 · Balances de materia y energía (simulación estacionaria)

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Flowsheet completo: corrientes, mezcladores, divisores, válvulas, bombas, compresores, intercambiadores, flash, columnas, reactores, reciclo | ✅ (12 tools base + `set_*`, `dof_check`, `sweep`) | DWSIM Fluent API | GPL-3 · proceso aparte | Un `solve()` puede no regresar (columna PR polar); falta timeout por llamada en el MCP |
| Verificación de balances y convergencia (el sello) | ✅ (`solve` → `verify.py`) | propio | MIT | — |
| Grados de libertad antes de resolver | ✅ (`dof_check`) | propio | MIT | Es contabilidad de la especificación, no física |
| Modelado orientado a ecuaciones (optimización simultánea, flowsheets de IDAES) | 📦 | IDAES/Pyomo | BSD · en proceso | **Falta ipopt** (conda-forge o binario `idaes get-extensions`); sin él solo sirven los bloques explícitos (costeo) |
| Dinámica (arranque, paro, transitorios) | ⬜ | DWSIM modo dinámico (`DWSIM.DynamicsManager.dll` presente) | GPL-3 · proceso aparte | Nunca probado desde la Fluent API |

## 3 · Reacción química

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Reactores de conversión, equilibrio, CSTR y PFR cinéticos dentro del flowsheet | ✅ (`define_*_reaction`, `add_unit cstr/pfr`) | DWSIM | GPL-3 · proceso aparte | `WithIsothermalTemperature` roto en 10.2.12 (usar `Isothermal()`) |
| Cinética detallada: mecanismos (GRI-Mech), llama adiabática, retardo de ignición, WSR, redes de reactores | ✅ (`adiabatic_flame`, `kinetic_reactor_network`, M2 2026-11-08: llama 2226 K, PFR/CSTR = analítico, ignición 4.53 ms, apagado 0.2-0.5 ms, S_L 0.378 m/s) | Cantera 3.2 (`IdealGasReactor`, `ReactorNet`, `FreeFlame`) | BSD · en proceso | Cinética propia solo isotérmica (sin termo real de las especies del usuario); catálisis de superficie y llamas de difusión sin tool |
| **Reactor real con geometría**: perfil radial, laminar vs pistón, difusión radial | ✅ (`tubular_reactor_field`, M1 2026-11-08: pistón 0.7940 = analítico; laminar 0.7819, malla independiente, Re < 2300) | FEniCSx | LGPL-3 · **proceso aparte** en el env micromamba (`ALCHEMAI_FENICSX_PYTHON`) | No isotérmico (ahí el PFR ideal se separa más del campo); puntos calientes; la app de Mac empaquetada no lleva FEniCSx |
| Reactores multifásicos, catalíticos con transporte en el pellet (Thiele) | ⬜ | FEniCSx o modelo 1D propio (scipy) | — | Después del tubular no isotérmico |

## 4 · Separaciones

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Columnas shortcut (FUG) y rigurosas, absorbedores, flash | ✅ (`add_unit`, etapas enteras en `optimize_design_multi`) | DWSIM | GPL-3 · proceso aparte | Extractores líquido-líquido y columnas con tres fases no probados |
| Etapas y altura de absorción: Kremser, Colburn, Onda, difusividades | ✅ (3 tools, bloque 22) | propio sobre literatura + `chemicals` | MIT | Datos de relleno del ejemplo son ilustrativos (MOCKS RELLENO-ILUSTRATIVO) |
| Hidráulica de columnas empacadas: inundación, ΔP, diámetro | ✅ (2 tools, bloque 23) | `fluids` Stichlmair | MIT | Platos (Fair, arrastre, lloriqueo) no cubiertos; catálogo de rellenos es fuente secundaria |
| Membranas, adsorción (PSA), cristalización, secado | ⬜ | modelos 1D propios (scipy) o BioSTEAM (unidades de bioprocesos) | — | Sin motor abierto maduro para PSA; se decide por demanda |

## 5 · Momento (flujo de fluidos)

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| ΔP en tubería monofásica, diámetro nominal (B36.10M), velocidad, régimen | ✅ (3 tools, bloque 18; ±1.4 % vs DWSIM) | `fluids` Colebrook/Clamond | MIT | Gas con ΔP > 10 % de P entrada (usar `fluids.compressible`) |
| Tubería en el flowsheet (dos fases, Beggs-Brill) | ✅ (kind `pipe`) | DWSIM Pipe Segment | GPL-3 | Solo lo que DWSIM da; sin ΔP bifásico propio |
| Accesorios, válvulas de control, flujo compresible, dos fases propio | ✅ (4 tools, bloque M4a; ejemplos 1-3 de IEC 60534-2-1, EDO de momento, límites de una fase) | `fluids` (IEC 60534-2-1, Darby 3-K, Friedel/MSH/Chisholm/L-M) + forma cerrada isotérmica propia | MIT · en proceso | Bombas (eficiencia) sin tool; gas adiabático (Fanno) no cubierto; dos fases solo fricción horizontal |
| **Campo de flujo con geometría**: tanque agitado, mezclado, tiempo de homogeneización, zonas muertas, turbulencia | 🟡 flujo continuo ✅ (`vessel_residence_time_cfd`, M6a: tanque prismático, simpleFoam laminar/k-ε + edad media local → τ, σ²/τ², N, volumen estancado; teorema edad de salida = V/Q y estudio de malla); impulsor 🧪 (Rushton MRF en `transporte-lab` caso 3: Np 4.67 vs ≈ 5, balance de torque 0.26 %, 19 min en 4 núcleos; sin tool) | OpenFOAM v2606 | **GPL-3 · proceso aparte** (`openfoam -c`, caso en $TMPDIR sin espacios) | M6b: impulsor con MRF (número de potencia vs literatura, tiempo de mezclado); tanque cilíndrico con mamparas |
| Redes de tuberías (mallas, bombas en paralelo) | ⬜ | modelo propio (Hardy-Cross con `fluids`) | — | Baja prioridad hasta que un caso lo pida |

## 6 · Calor

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Área de intercambiador: LMTD·F (Fakheri), Gnielinski lado tubo, U | ✅ (2 tools; área de DWSIM a 0.2 %) | `ht` | MIT | — |
| Lado coraza: Kern, Bell-Delaware, arreglo de tubos, claros | ✅ Kern (`shell_side_kern`, M4b) y Bell-Delaware (`shell_side_bell_delaware`, M4c: Jc/Jl/Jb/Js/Jr y Rl/Rb/Rs, claros TEMA por defecto declarados, arreglos 30/45/90) | `ht` | MIT · en proceso | Banco ideal con la correlación j/f de Taborek (hoy Žukauskas + ESDU); boquillas; validar contra un ejemplo publicado con datos legibles |
| Ebullición, condensación, radiación, aislamiento | ✅ ebullición nucleada + flujo crítico y condensación (2 tools, bloque M4b; Incropera ej. 10.1 reproducido) · radiación y aislamiento 📦 | `ht` | MIT · en proceso | Ebullición en flujo (Chen) y película; radiación y aislamiento sin tool |
| Integración energética (pinch, redes de intercambiadores) | ✅ targets (`pinch_analysis`, M5: tabla del problema, MER, curvas compuestas y gran compuesta, unidades mínimas, barrido de ΔTmin, ahorro vs servicios actuales) + trade-off recuperación-costo (bloque 20) | propio + DWSIM | MIT | Diseño de la red (HEN) por el método del pinch, objetivo de área (Bath) y de costo (supertargeting), servicios múltiples, extracción automática de corrientes desde el flowsheet |
| **Campo de temperatura con geometría**: conducción en sólidos, puntos calientes, pared de reactor | 🧪 (`conduccion_2d.py`, orden 2 verificado contra analítico) | FEniCSx | LGPL-3 · proceso aparte | Entra junto con el reactor no isotérmico |
| Hornos y calderas (radiación de llama) | ⬜ | Cantera (llama) + `ht.radiation` | BSD/MIT | Sin caso de referencia elegido |

## 7 · Masa (transferencia)

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Coeficientes de película (Onda), HTU/NTU, difusividades Wilke-Chang y Fuller | ✅ (bloque 22) | propio + `chemicals` | MIT | Correlaciones para estructurado (Rocha-Bravo-Fair) |
| **Campo de concentración con geometría**: advección-difusión-reacción axisimétrica, SUPG | ✅ (`tubular_reactor_field`) | FEniCSx | LGPL-3 · proceso aparte | Órdenes ≠ 1 y varias especies |
| Difusión en sólidos/pellets, membranas | ⬜ | FEniCSx / scipy | — | Por demanda |

## 8 · Costos y economía

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Costo de compra de equipos (HX, bomba, compresor, recipiente, platos, relleno), CEPCI 1990-2023 | ✅ (3 tools, bloque 19; exacto vs tests de IDAES) | IDAES costeo SSLW (Seider) | BSD · en proceso | Costo de compra, **no instalado**; CEPCI > 2023 lo da el usuario |
| Capital fijo (Lang), factor de ubicación (México 0.84-0.92), anualización (CRF), costo de operación → costo total anual | ✅ (`flowsheet_cost`, bloque 20) | propio | MIT | Precios siempre del usuario (MOCKS PRECIOS-ILUSTRATIVOS) |
| Factores de módulo desnudo por equipo, inversión total (sitio, servicios, contingencia, terreno, arranque, capital de trabajo) | ✅ (`bare_module_factors`, `capital_investment`, M3 2026-11-08) | propio; F_BM de Seider 2017 leídos de BioSTEAM (no Turton: mismo libro que el costo de compra de IDAES) | MIT/NCSA · en proceso | Porcentajes de inversión siempre del usuario; relleno y equipos sin F_BM en BioSTEAM los da el usuario |
| Flujo de caja, VPN, TIR, periodo de recuperación, depreciación | ✅ (`cash_flow_analysis`, M3: = fórmulas cerradas y = funciones de BioSTEAM) | propio, VPN recalculado con las funciones de `biosteam._tea` | MIT | Sin préstamos ni inflación; MACRS es de EE. UU. (México: SL con tasa de la LISR) |
| Costo de servicios (vapor, agua, electricidad) por región | ⬜ | tablas del usuario | — | Nunca defaults: la regla de la fábrica |

## 9 · Optimización y modelos sustitutos

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Una variable (barrido + sección dorada), restricciones, infactibilidad | ✅ (`optimize_design`, bloque 21) | scipy + DWSIM | BSD/GPL | — |
| 2-4 variables continuas + enteras (etapas), Nelder-Mead | ✅ (`optimize_design_multi`, bloques 24/26) | scipy + DWSIM | BSD/GPL | Sin gradientes: > 4 variables es lento |
| NLP/MINLP riguroso (ecuaciones simultáneas) | 📦 | Pyomo + IDAES | BSD | **ipopt** (y para MINLP: Bonmin/Couenne o SCIP) |
| Modelos sustitutos (red neuronal sobre datos de DWSIM) embebidos en Pyomo | 📦 | OMLT | BSD | Bloque 32 del plan (tema 7 de estudio) |
| Optimización bayesiana, experimentos | ⬜ | Summit | — | Fase 3 del plan |

## 10 · Seguridad de proceso

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| HAZOP preliminar por reglas sobre el grafo (desde DWSIM o DEXPI) + propuestas LLM marcadas | ✅ (`hazop_preliminary`, `hazop_from_dexpi`; recall 30/30 apartado, 0 espurias) | propio + Claude | MIT | Precisión útil 76 %: sigue habiendo ruido |
| Dimensionamiento de válvulas de alivio (API 520/521), ruido | ✅ (`relief_valve_sizing`, bloque M4a; ejemplos 1, 2, 4 y 5 de API 520 Parte I) · ruido 📦 | `fluids.safety_valve` + API 526 | MIT · en proceso | Conectar a la fila HAZOP «más presión»; ruido API 521 sin tool; la tasa de alivio sale del escenario (no la decide la tool) |
| LOPA, SIL, árbol de fallas | ⬜ | propio (tablas CCPS) | — | Fase 2 del plan |
| Dispersión, incendio, explosión (consecuencias) | ⬜ | modelos propios (Gaussiano, TNT-equivalente) | — | Sin motor abierto maduro; por demanda |

## 11 · Diseño de planta y documentación

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Exportar el flowsheet a estándares abiertos (SFILES 2.0, DEXPI Process) | ✅ (`export_flowsheet`; 12/12 ida y vuelta) | SFILES2, pyDEXPI | MIT | Validar contra el XSD oficial de DEXPI |
| Diagrama PFD con etiquetas y resultados | ✅ (app Mac: `LienzoDiagrama`, PDF) | propio SwiftUI | — | — |
| Reporte PDF del caso (conversación, corrientes, equipos, balances, HAZOP) | ✅ (B8 de la app) | propio | — | — |
| P&ID (instrumentación, lazos, válvulas) | ⬜ | DEXPI P&ID vía pyDEXPI | MIT | Generar instrumentación mínima desde reglas + HAZOP; nadie lo tiene abierto |
| Layout y ruteo de tubería | 🚫 hoy | ninguno abierto maduro | — | Hueco de investigación (fase 3): es diferenciador, no un motor que se instala |
| 3D / CAD de planta | 🚫 | Plant 3D (Windows, COM) | decisión del owner (Parallels) | No construir sobre Aspen/E3D (sin API abierta) |

## 12 · Control y dinámica

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Funciones de transferencia, sintonía PID, respuesta a escalón | ⬜ | `python-control` o scipy.signal | BSD | Bloque de estudio (Seborg) + tool `pid_tuning`; sin dependencias pesadas |
| Simulación dinámica del flowsheet | ⬜ | DWSIM dinámico | GPL-3 | Ver §2 |
| RL / control avanzado | ⬜ | pc-gym (Python 3.11, venv aparte) | — | Fase 3 |

## Orden propuesto de incorporación

Criterio: primero lo que el owner pidió (transporte con geometría), luego lo que ya está instalado y validado
(barato), luego lo de puro Python con alto valor económico, y al final lo que exige instalar o investigar.

| # | Bloque | Área | Por qué ahora | Dependencias nuevas |
|---|---|---|---|---|
| 1 ✅ | `tubular_reactor_field` (FEniCSx, proceso aparte) — hecho 2026-11-08 | 3, 6, 7 | Patrón «motor externo en otro entorno» fijado en `packages/transport/reactor_field.py` (`run_solver`): OpenFOAM lo reutiliza | ninguna |
| 2 ✅ | `kinetic_reactor_network` + `adiabatic_flame` (Cantera) — hecho 2026-11-08 | 3 | Validado en el notebook 03; en proceso; cubre combustión y cinética detallada | ninguna |
| 3 ✅ | Economía completa: módulo desnudo (Seider vía BioSTEAM), flujo de caja, VPN/TIR — hecho 2026-11-08 | 8 | Puro Python; es lo que convierte un costo en una decisión | ninguna (tablas con fuente) |
| 4 | ✅ 2026-11-09 (M4a + M4b + M4c, 51 tools): alivio API 520, válvulas IEC, accesorios, gas compresible, dos fases, Kern, Bell-Delaware, ebullición, condensación | 5, 6, 10 | Librerías ya instaladas; cada tool = referencia + test | ninguna |
| 5 | ✅ 2026-11-09 (M5, 52 tools): pinch formal (curvas compuestas, MER, ΔTmin) | 6 | Sin dependencias; conecta con `flowsheet_cost` | ninguna |
| 6 | 🟡 M6a ✅ 2026-11-09 (53 tools): tanque de flujo continuo OpenFOAM vs CSTR (`vessel_residence_time_cfd`). M6b: Rushton MRF probado (caso 3 de `transporte-lab`, Np 4.67 vs ≈ 5); falta la tool `stirred_tank_cfd` | 5 | Segundo motor externo; mezclado es pregunta frecuente | ninguna (OpenFOAM instalado) |
| 7 | Reactor tubular no isotérmico (FEniCSx, energía acoplada) | 3, 6 | Donde el PFR ideal y el campo se separan de verdad | ninguna |
| 8 | ipopt → IDAES/Pyomo riguroso; OMLT (bloque 32) | 2, 9 | Abre optimización simultánea y sustitutos | **ipopt** (conda-forge; sin costo) |
| 9 | Control: `pid_tuning`, respuesta dinámica | 12 | Tema de estudio del plan | `python-control` (BSD) |
| 10 | P&ID mínimo desde HAZOP (DEXPI P&ID) | 11 | Diferenciador; nadie lo tiene abierto | ninguna |
| — | Layout/ruteo, 3D, LOPA, dispersión | 10, 11 | Investigación o decisión del owner | — |

## Reglas de admisión de un motor nuevo

1. Licencia compatible con MIT en proceso (MIT/BSD/Apache/LGPL con enlace dinámico); GPL solo como proceso aparte.
2. Un caso de referencia con respuesta conocida **antes** de la tool; la tool lo reproduce en `tests/`.
3. Entradas en SI con nombres explícitos; nada de defaults para precios, propiedades o datos del usuario.
4. Salida con `evidence` propia del motor (convergencia, independencia de malla, error vs analítico) que decide el sello.
5. Si corre en otro entorno (micromamba, binario), la tool lanza el proceso con timeout y mata el grupo al vencer,
   igual que `run_claude_p`.
6. Lo que es hipótesis (difusividad estimada, propiedad de orden de magnitud) va en la respuesta como tal y en `MOCKS.md`.
