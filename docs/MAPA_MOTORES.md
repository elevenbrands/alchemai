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

Estado verificado el 2026-11-14: **71 tools** (M16: `optimize_design_bayesian` y `suggest_experiments`, scikit-learn 1.9.1), DWSIM 10.2.12 (`~/Applications/DWSIM10`), thermo 0.6.1, chemicals 1.5.2,
fluids 1.3.1, ht 1.2.0, cantera 3.2.0, idaes 2.13.0 + pyomo 6.10.1 + ipopt 3.13.2 (idaes-ext 3.4.2, `~/.idaes/bin`), omlt 1.2.2, scikit-learn 1.9.1, biosteam 2.54.0,
SFILES2 1.2.0, pydexpi 2.1.0, CoolProp 8.0.0; fuera del entorno: OpenFOAM v2606 (`openfoam`), FEniCSx dolfinx 0.11.0
(`~/mamba/envs/fenicsx`). Repo hermano `transporte-lab` guarda los casos de CFD/FEM.

## 1 · Termodinámica y propiedades

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Equilibrio de fases (PR, SRK, NRTL, UNIQUAC…), flash PT/PH/PS, puntos de burbuja y rocío, azeótropos | ✅ (`list_property_packages`, `read_stream`, `sweep`) | DWSIM | GPL-3 · proceso aparte (pythonnet) | VLLE en polares no lo hace DWSIM-NRTL ni Fluxo (anotado bloque 12) |
| Propiedades de compuestos puros (Psat DIPPR, Tfus, ΔHf, ω…) | ✅ parcial (dentro de HAZOP y de Fluxo) | `chemicals`, `thermo` | MIT · en proceso | Tool pública de propiedades (`compound_properties`) para que el agente consulte sin armar flowsheet |
| Flash independiente de DWSIM (segunda opinión, VLLE con Michelsen) | 📦 | `thermo.Flash`, `FlashVLN` | MIT · en proceso | Segunda opinión termodinámica automática cuando DWSIM y `thermo` difieren |
| Regresión de parámetros binarios con datos experimentales | ✅ (`vle_binary_regression`, M14; M15c 2026-11-13: ajuste conjunto VLE + h^E calorimétrico con pesos y AAD separados y veredicto «un juego representa ambos»; sintético recuperado 12/12, etanol-agua real: NO los representa y la tool lo dice; desde M15c-2 ese veredicto decide si la columna EO enciende h^E por defecto) | `thermo` (modelos y verificación) + scipy (mínimos cuadrados, máxima verosimilitud) | MIT/BSD · en proceso | NRTL/UNIQUAC/Wilson desde Pxy, Txy o PTxy con IC 95 %, Redlich-Kister/Herington, Fredenslund, Van Ness y UNIFAC Do. Los parámetros NRTL ya entran a `eo_column_optimize` (M15: el resultado se pasa tal cual en `binary_parameters`). Con `excess_enthalpy_sets` ajusta también h^E (Gibbs-Helmholtz de los mismos parámetros). Falta: dependencia en T más rica (e·ln T, f·T) o una correlación propia de h^E, LLE, datos de varias fuentes con pesos de calidad (TDE), un ejemplo DECHEMA legible y llevar UNIQUAC/Wilson a la columna EO |

## 2 · Balances de materia y energía (simulación estacionaria)

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Flowsheet completo: corrientes, mezcladores, divisores, válvulas, bombas, compresores, intercambiadores, flash, columnas, reactores, reciclo | ✅ (12 tools base + `set_*`, `dof_check`, `sweep`) | DWSIM Fluent API | GPL-3 · proceso aparte | Un `solve()` puede no regresar (columna PR polar); falta timeout por llamada en el MCP |
| Verificación de balances y convergencia (el sello) | ✅ (`solve` → `verify.py`) | propio | MIT | — |
| Grados de libertad antes de resolver | ✅ (`dof_check`) | propio | MIT | Es contabilidad de la especificación, no física |
| Modelado orientado a ecuaciones (optimización simultánea, flowsheets de IDAES) | ✅ columna (`eo_column_optimize`: M8a 2026-11-13 caso de IDAES reproducido, DWSIM a 0.5 %; M8b 2026-11-13 cualquier mezcla ideal con paquete generado de `chemicals`, hexano-heptano vs DWSIM a 0.55 %; M15 2026-11-13 no ideales con NRTL: columna metanol-agua publicada (Simone 2015, Aspen/gPROMS) reproducida, DWSIM con los mismos parámetros ≤ 0.62 %; M15b 2026-11-13 etanol-agua cerrado: Q_reb vs DWSIM igual con igual +0.20 %, arranque robusto con homotopía y continuación de specs, rechazos por N_min/azeótropo; M15c 2026-11-13 h^E contrastado con calorimetría: el de parámetros solo-VLE tiene el signo contrario al medido por debajo de ~363 K y deja Q_reb a −5.0 % del valor con h^E medido (apagado: −1.2 %); la salida ahora dice de dónde viene h^E y cuánto pesa; M15c-2 2026-11-13, decisión del owner: `excess_enthalpy` por defecto condicional, encendido solo con parámetros de un ajuste conjunto VLE + h^E que pase el veredicto; referencias con parámetros solo-VLE recalculadas: metanol-agua Q_reb 1497.6 → 1507.8 kW, etanol-agua 1030.0 → 1070.8 kW, DWSIM «ExpData» ≤ 0.72 %) | IDAES/Pyomo + ipopt (idaes-ext) | BSD · en proceso; ipopt EPL ejecutable aparte | NRTL con parámetros del usuario, de `vle_binary_regression` o del banco ChemSep (nunca inventados); se niega si las specs cruzan un azeótropo o el vapor no es ideal. Falta: UNIQUAC/Wilson, vapor no ideal (asociación, presión), fronteras de destilación con 3+ compuestos y azeótropo, arranque robusto en columnas acuosas, Q_reb +6.5 % vs DWSIM en etanol-agua sin explicar; sustitutos = M8c |
| Dinámica (arranque, paro, transitorios) | ⬜ | DWSIM modo dinámico (`DWSIM.DynamicsManager.dll` presente) | GPL-3 · proceso aparte | Nunca probado desde la Fluent API |

## 3 · Reacción química

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Reactores de conversión, equilibrio, CSTR y PFR cinéticos dentro del flowsheet | ✅ (`define_*_reaction`, `add_unit cstr/pfr`) | DWSIM | GPL-3 · proceso aparte | `WithIsothermalTemperature` roto en 10.2.12 (usar `Isothermal()`) |
| Cinética detallada: mecanismos (GRI-Mech), llama adiabática, retardo de ignición, WSR, redes de reactores | ✅ (`adiabatic_flame`, `kinetic_reactor_network`, M2 2026-11-08: llama 2226 K, PFR/CSTR = analítico, ignición 4.53 ms, apagado 0.2-0.5 ms, S_L 0.378 m/s) | Cantera 3.2 (`IdealGasReactor`, `ReactorNet`, `FreeFlame`) | BSD · en proceso | Cinética propia solo isotérmica (sin termo real de las especies del usuario); catálisis de superficie y llamas de difusión sin tool |
| **Reactor real con geometría**: perfil radial, laminar vs pistón, difusión radial | ✅ (`tubular_reactor_field`, M1 2026-11-08: pistón 0.7940 = analítico; laminar 0.7819, malla independiente, Re < 2300) | FEniCSx | LGPL-3 · **proceso aparte** en el env micromamba (`ALCHEMAI_FENICSX_PYTHON`) | No isotérmico ✅ M7 2026-11-13 (`nonisothermal_tubular_reactor`); pendiente: cinética no de primer orden, varias reacciones, caso 2-D publicado; la app de Mac empaquetada no lleva FEniCSx |
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
| **Campo de flujo con geometría**: tanque agitado, mezclado, tiempo de homogeneización, zonas muertas, turbulencia | ✅ flujo continuo (`vessel_residence_time_cfd`, M6a: tanque prismático, simpleFoam laminar/k-ε + edad media local → τ, σ²/τ², N, volumen estancado; teorema edad de salida = V/Q y estudio de malla); impulsor ✅ (`stirred_tank_cfd`, M6b: Rushton con mamparas, MRF k-ε, snappyHexMesh; Np, P, P/V, θ95 por correlación de Grenville; balance de torque, estabilidad, checkMesh e independencia de malla por banda entre 3 mallas (< 10 %; el GCI no aplica porque el aspa delgada salta con la malla) ✅; Np 4.85 ± 3.5 % vs ≈ 5; ~95 min por llamada) | OpenFOAM v2606 | **GPL-3 · proceso aparte** (`openfoam -c`, caso en $TMPDIR sin espacios) | Tiempo de mezclado por CFD (trazador transitorio); otros impulsores (pitched blade, hélice) |
| Redes de tuberías (mallas, bombas en paralelo) | ✅ (`pipe_network`, M13 2026-11-13: gradiente global de Todini-Pilati (el de EPANET) con contraste Hardy Cross; mallas, varias fuentes y sumideros, reservorios, tanques y fronteras de presión fija; bombas en serie y en paralelo con la curva del usuario, curva de sistema y curva combinada, NPSHa vs NPSHr, potencia hidráulica y al eje; check, PRV y FCV; residuo de masa por nodo ≤ 1e-9 m³/s; EPANET 2.2 Net1 y 5 variantes (H-W idéntico; D-W ≤ 0.1 %, explicado por Swamee-Jain y g = 32.2 ft/s²), Jeppson 1974 (redes de 4 y 5 lazos, 3 bombas desde 3 reservorios, 3 reservorios con H-W), Larock Ex. 2.7, Evett & Liu 12.11 y Cameron NPSH ej. 1-4 reproducidos; límite de un tubo = `pipe_hydraulics` exacto) | propio (numpy) con `fluids` Colebrook/Clamond; EPANET 2.2 vía WNTR solo para la referencia | MIT · en proceso (EPANET dominio público/MIT y WNTR BSD, en entorno desechable aparte) | Sin transitorios (golpe de ariete) ni simulación en el tiempo (tanques a nivel fijo); un solo fluido isotérmico; sin ejemplo publicado legible de bombas en paralelo con su curva |

## 6 · Calor

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Área de intercambiador: LMTD·F (Fakheri), Gnielinski lado tubo, U | ✅ (2 tools; área de DWSIM a 0.2 %) | `ht` | MIT | — |
| Lado coraza: Kern, Bell-Delaware, arreglo de tubos, claros | ✅ Kern (`shell_side_kern`, M4b) y Bell-Delaware (`shell_side_bell_delaware`, M4c: Jc/Jl/Jb/Js/Jr y Rl/Rb/Rs, claros TEMA por defecto declarados, arreglos 30/45/90) | `ht` | MIT · en proceso | Banco ideal con la correlación j/f de Taborek (hoy Žukauskas + ESDU); boquillas; validar contra un ejemplo publicado con datos legibles |
| Ebullición, condensación, radiación, aislamiento | ✅ ebullición nucleada + flujo crítico y condensación (2 tools, bloque M4b; Incropera ej. 10.1 reproducido) · radiación y aislamiento 📦 | `ht` | MIT · en proceso | Ebullición en flujo (Chen) y película; radiación y aislamiento sin tool |
| Integración energética (pinch, redes de intercambiadores) | ✅ targets (`pinch_analysis`, M5: tabla del problema, MER, curvas compuestas y gran compuesta, unidades mínimas, barrido de ΔTmin, ahorro vs servicios actuales) + extracción automática de corrientes desde el flowsheet (`pinch_from_flowsheet`, M5b: calentadores, enfriadores e intercambiadores, CP segmentado en cambio de fase, Kemp reproducido en DWSIM) + trade-off recuperación-costo (bloque 20) | propio + DWSIM | MIT | Diseño de la red (HEN) por el método del pinch, objetivo de área (Bath) y de costo (supertargeting), servicios múltiples, condensadores/rehervidores de columnas y reactores en la extracción |
| **Campo de temperatura con geometría**: conducción en sólidos, puntos calientes, pared de reactor | ✅ en el reactor (`nonisothermal_tubular_reactor`, M7 2026-11-13: T(z, r), punto caliente, pared fija/refrigerante/adiabática; cierre de energía ≤ 1e-5, malla 2×, límites ΔH = 0 y adiabático, modelo de Rawlings & Ekerdt reproducido); conducción en sólidos sigue 🧪 (`conduccion_2d.py`) | FEniCSx | LGPL-3 · proceso aparte | Conducción en sólidos/pared con espesor como tool |
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
| NLP/MINLP riguroso (ecuaciones simultáneas) | ✅ NLP (M8a); MINLP 📦 (bonmin/couenne ya instalados con idaes-ext, sin tool) | Pyomo + IDAES | BSD | Tool MINLP (etapas/alimentación enteras) pendiente |
| Modelos sustitutos (red neuronal sobre datos de DWSIM) embebidos en Pyomo | ✅ (`optimize_design_surrogate`, M8c 2026-11-13: compresor 3 etapas y columna hexano-heptano = óptimo directo de DWSIM a ≤ 0.17 %, re-verificado en DWSIM) | OMLT 1.2.2 + ipopt (red tanh en numpy) | BSD · en proceso; ipopt ejecutable aparte | Solo continuas (1-6); sin muestreo adaptativo; ReLU/MILP sin usar |
| Optimización bayesiana (flowsheet caro o que falla) y diseño bayesiano de experimentos (datos del usuario) | ✅ (`optimize_design_bayesian`, `suggest_experiments`, M16 2026-11-14: Branin 20/20 y problema con restricciones de Gramacy 20/20; compresor de 3 etapas y columna hexano-heptano = óptimo directo de DWSIM a ≤ 0.012 % con 33-34 evaluaciones (17 sin el reinicio de comprobación) contra 80-145 y 67; etapas enteras en 50-66 solves (20-36 sin reinicio) contra 442; SnAr de Summit: STY a ≤ 1 % en 19/20 campañas de 40 experimentos, azar 0/20) | scikit-learn 1.9.1 (`GaussianProcessRegressor`/`Classifier`, Matérn 5/2) + adquisición propia (mejora esperada con restricciones, en logaritmos) | BSD-3 · en proceso | **No garantiza el óptimo global**: con el reinicio por defecto (decisión del owner, 2026-11-14) Hartmann-6 acierta 13/20 y los óptimos equivocados que salen «verified» bajan de 12 a 4 de 20; no los elimina. Factor E de SnAr: 1/20 (10/20 con log). Sin multiobjetivo (frente de Pareto) ni variables categóricas. Summit (MIT) no instala (pandas < 2); BoTorch (MIT) evaluado y descartado por peso (torch, 759 MB) sin mejora en Hartmann-6 |

## 10 · Seguridad de proceso

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| HAZOP preliminar por reglas sobre el grafo (desde DWSIM o DEXPI) + propuestas LLM marcadas | ✅ (`hazop_preliminary`, `hazop_from_dexpi`; recall 30/30 apartado, 0 espurias) | propio + Claude | MIT | Precisión útil 76 %: sigue habiendo ruido |
| Dimensionamiento de válvulas de alivio (API 520/521), ruido | ✅ (`relief_valve_sizing`, bloque M4a; ejemplos 1, 2, 4 y 5 de API 520 Parte I) · ruido 📦 | `fluids.safety_valve` + API 526 | MIT · en proceso | Conectar a la fila HAZOP «más presión»; ruido API 521 sin tool; la tasa de alivio sale del escenario (no la decide la tool) |
| LOPA y SIL (capas de protección, reducción de riesgo, SIL de cada disparo) | ✅ (`lopa_from_hazop`, bloque M11 2026-11-13: HAZOP → P&ID M10 → hoja LOPA; ejemplo HSE/BSTG 2007 App. 1 y su variante SIL 2, dos ejemplos Kenexis y Casey 2019 reproducidos; bandas IEC 61511 en sus bordes; reglas de independencia I-1…I-6 sobre los tags del P&ID) · árbol de fallas ⬜ | propio (tablas genéricas con fuente) | MIT · en proceso | Tablas genéricas, no del sitio; CCPS 2001/2015 sin copia legible (valores vía fuentes secundarias, sus ejemplos no se reproducen); la frecuencia tolerable es del owner; causas fuera de la tabla (taponamiento, runaway…) quedan por definir; sin árbol de fallas ni verificación de PFDavg de la SIF (arquitectura, intervalo de prueba) |
| Dispersión, incendio, explosión (consecuencias) | ✅ (`release_source_term`, `gas_dispersion`, `fire_radiation`, `explosion_overpressure`, bloque M12 2026-11-13: descarga, flash y evaporación; pluma/puff de Pasquill-Gifford con criterio de gas denso de Britter-McQuaid; charco SFPE y TNO, bola de fuego TNO, chorro como fuente puntual; TNT; probits; ejemplos de Crowl & Louvar 4.ª ed., TNO Yellow Book 2005, NUREG-1805, NISTIR 6546 y PGS 1 reproducidos; severidad calculada hacia `lopa_from_hazop`; M12b: multi-energía TNO (Díaz Alonso 2006), chorro Chamberlain (YB §6.6.2), gas denso Britter-McQuaid (LNG 367 m) y charco en ebullición) · BST ⬜ | propio (modelos publicados con fuente) + CoolProp | MIT · en proceso | Cribado, no QRA (Phast/EFFECTS siguen siendo la herramienta validada); BST sin curvas legibles; Chamberlain solo Rw ≤ 0.05; Cm* de Britter-McQuaid de fuentes secundarias; HyRAM+ (GPL-3, solo H₂/CH₄/propano) como posible contraste en proceso aparte |

## 11 · Diseño de planta y documentación

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Exportar el flowsheet a estándares abiertos (SFILES 2.0, DEXPI Process) | ✅ (`export_flowsheet`; 12/12 ida y vuelta; XSD oficial DEXPI 2.0.1 válido, 2026-11-13) | SFILES2, pyDEXPI | MIT | Tipos contra `Plant.xml`/`Process.xml` oficiales (data.dexpi.org no respondió) |
| Diagrama PFD con etiquetas y resultados | ✅ (app Mac: `LienzoDiagrama`, PDF) | propio SwiftUI | — | — |
| Reporte PDF del caso (conversación, corrientes, equipos, balances, HAZOP) | ✅ (B8 de la app) | propio | — | — |
| P&ID (instrumentación, lazos, válvulas) | ✅ (`pid_from_hazop`, bloque M10 2026-11-13: reglas con fuente + PSV/check/alarmas/disparos del HAZOP, tags ISA-5.1 trazados; 22 casos con ida y vuelta y XSD oficial; Skogestad 1997 Fig. 1 reproducida; SIL de sus disparos por `lopa_from_hazop`, M11) | DEXPI P&ID vía pyDEXPI 2.1.0 | MIT · en proceso | Sin dibujo (pyDEXPI pide geometría: hueco de layout); bombas de desplazamiento positivo; el SIL no se escribe aún en el DEXPI |
| Layout y ruteo de tubería | 🚫 hoy | ninguno abierto maduro | — | Hueco de investigación (fase 3): es diferenciador, no un motor que se instala |
| 3D / CAD de planta | 🚫 | Plant 3D (Windows, COM) | decisión del owner (Parallels) | No construir sobre Aspen/E3D (sin API abierta) |

## 12 · Control y dinámica

| Necesidad | Hoy | Motor | Licencia · cómo entra | Hueco |
|---|---|---|---|---|
| Funciones de transferencia, sintonía PID, respuesta a escalón | ✅ (3 tools, bloque M9 2026-11-13: `fopdt_identify`, `pid_tuning`, `pid_closed_loop`; Skogestad 2003 Tablas 2, 4 (22 de 25 filas) y 5-7, Seborg Ej. 12.4) | `packages/dynamics` propio (retardo exacto) + `python-control` 0.10.2 como contraste | BSD · en proceso | Reglas T1-T3 de ceros positivos sin automatizar; Cohen-Coon sin ejemplo publicado; ITAE/Åström-Hägglund (AMIGO) no incluidos; sin control multivariable ni anti-windup |
| Simulación dinámica del flowsheet | ⬜ | DWSIM dinámico | GPL-3 | Ver §2; la Fluent API que usamos no expone el modo dinámico (el MCP oficial lo tiene en fuente, sin compilar): la prueba escalón para M9 viene del usuario |
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
| 5 | ✅ 2026-11-09 (M5, 52 tools): pinch formal (curvas compuestas, MER, ΔTmin). M5b ✅ 2026-11-13 (55 tools): corrientes extraídas del flowsheet (`pinch_from_flowsheet`) | 6 | Sin dependencias; conecta con `flowsheet_cost` | ninguna |
| 6 | ✅ M6a ✅ 2026-11-09 (53 tools): tanque de flujo continuo OpenFOAM vs CSTR (`vessel_residence_time_cfd`). M6b ✅ 2026-11-10 (54 tools): `stirred_tank_cfd` Rushton MRF, Np 4.85 ± 3.5 % vs ≈ 5, verificado con banda entre 3 mallas | 5 | Segundo motor externo; mezclado es pregunta frecuente | ninguna (OpenFOAM instalado) |
| 7 | ✅ 2026-11-13 (M7, 56 tools): reactor tubular no isotérmico (FEniCSx, energía acoplada, P2 + SUPG) vs PFR 1-D (`nonisothermal_tubular_reactor`) | 3, 6 | Donde el PFR ideal y el campo se separan de verdad | ninguna |
| 8 | M8a ✅ 2026-11-13 (57 tools): ipopt + `eo_column_optimize` (IDAES); M8b ✅ 2026-11-13: cualquier mezcla ideal (paquete IDAES generado de `chemicals`, cribado UNIFAC); M8c ✅ 2026-11-13 (58 tools): `optimize_design_surrogate` (OMLT, red sobre muestras de DWSIM, re-verificada en DWSIM). **Bloque 8 cerrado** | 2, 9 | Abre optimización simultánea y sustitutos | **ipopt** (conda-forge; sin costo) |
| 9 ✅ | Control — hecho 2026-11-13 (M9, 61 tools): `fopdt_identify` (FOPDT/SOPDT por mínimos cuadrados + Smith y Sundaresan-Krishnaswamy), `pid_tuning` (SIMC por defecto, IMC, ZN último ciclo y curva de reacción, Cohen-Coon, Tyreus-Luyben; regla de la mitad), `pid_closed_loop` (márgenes exactos, Ms, IAE, sobrepaso) | 12 | Tema de estudio del plan | `python-control` 0.10.2 (BSD) |
| 10 ✅ | P&ID mínimo desde HAZOP (DEXPI P&ID) — hecho 2026-11-13 (M10, 62 tools): `pid_from_hazop`, DEXPI 2.0.1 con ida y vuelta, XSD oficial y Skogestad 1997 Fig. 1 | 11 | Diferenciador; nadie lo tiene abierto | ninguna |
| 11 ✅ | LOPA y SIL desde el HAZOP y el P&ID — hecho 2026-11-13 (M11, 63 tools): `lopa_from_hazop`, independencia por tags del P&ID, SIL por IEC 61511, ejemplos HSE/BSTG 2007, Kenexis y Casey 2019 reproducidos | 10 | Cierra la cadena HAZOP → P&ID → nivel de protección | ninguna |
| 12 ✅ | Análisis de consecuencias — hecho 2026-11-13 (M12 + M12b, 67 tools): términos fuente y charco en ebullición, dispersión gaussiana y gas denso Britter-McQuaid, charco/bola de fuego/chorro Chamberlain, TNT y multi-energía, probits; enlace de severidad calculada a la LOPA. BST pendiente | 10 | Completa la seguridad: del HAZOP a qué pasa si la protección falla | ninguna (CoolProp ya en el entorno) |
| 13 ✅ | Redes de tuberías y sistemas de bombeo — hecho 2026-11-13 (M13, 68 tools): `pipe_network` (gradiente global + Hardy Cross, bombas serie/paralelo, curva de sistema, NPSH, check/PRV/FCV); EPANET Net1, Jeppson 1974, Larock y Cameron reproducidos | 5 | Lo eligió el owner tras M12: la hidráulica pasa de un tubo a la red de planta | ninguna (WNTR + EPANET solo para regenerar la referencia, en entorno desechable) |
| 14 ✅ | Regresión de parámetros binarios desde datos VLE — hecho 2026-11-13 (M14, 69 tools): `vle_binary_regression` (NRTL con α fijo o ajustado, UNIQUAC con r/q de grupos UNIFAC, Wilson con volúmenes de thermo; punto de burbuja o máxima verosimilitud Britt-Luecke; IC 95 % y correlación; Redlich-Kister/Herington, Fredenslund, Van Ness; azeótropo; UNIFAC Dortmund) | 1 | Lo eligió el owner tras M13: los casos reales con mezclas no ideales necesitan parámetros propios | ninguna |
| 15 ✅ | Columna EO para no ideales — hecho 2026-11-13 (M15, 69 tools, misma tool): `eo_column_optimize(activity_model="NRTL")` con parámetros del usuario, de `vle_binary_regression` o del banco ChemSep; caso metanol-agua de Simone 2015 (TU Wien) reproducido; DWSIM con los mismos parámetros: metanol-agua ≤ 0.62 %; etanol-agua (cadena con la regresión) composiciones ≤ 0.86 % pero Q_reb +6.5 %. **M15b (2026-11-13)**: el +6.5 % era el término de exceso de DWSIM (∂γ/∂T en vez de ∂ln γ/∂T), cerrado término a término (0.03 kW sin explicar de 54.6); igual con igual Q_reb +0.20 %; etanol-agua con 15 platos y x_D 0.75/0.80/0.82 verificados; McCabe-Thiele propio rechaza lo inalcanzable y lo que queda a < 0.03 K del azeótropo **M15c (2026-11-13)**: h^E de etanol-agua contra calorimetría publicada (ThermoML: Nagamachi 2006, Brandt 2014, 83 puntos): el h^E de parámetros solo-VLE es positivo donde el medido es negativo; Q_reb −5.0 % con el término, −1.2 % sin él, frente al h^E medido. `vle_binary_regression` ajusta VLE + h^E a la vez; un juego NRTL a + b/T no representa ambos en etanol-agua (dicho por la tool). La columna etiqueta la procedencia de h^E. **M15c-2 (2026-11-13)**: el owner eligió el valor por defecto condicional (opción C): `excess_enthalpy=None` enciende h^E solo si todos los pares vienen de un ajuste conjunto VLE + h^E que pasó el veredicto; si no, apagado con aviso. Referencias recalculadas (historia conservada en `BENCHMARK.md`) | 2, 4 | Cierra el hueco de M8b (solo ideales) y conecta M14 con la columna | ninguna |
| 16 ✅ | Optimización bayesiana y diseño de experimentos — hecho 2026-11-14 (M16, 71 tools): `optimize_design_bayesian` (GP + mejora esperada con restricciones, fallos del motor como puntos infactibles con imputación pesimista, enteros nativos, escala log, semillas, tres reglas de paro, un reinicio independiente por defecto con estado `runs_disagree` si las corridas difieren (M16b); mejor punto reevaluado en DWSIM) y `suggest_experiments` (datos del usuario, sin simulador: propuestas etiquetadas, lote «Kriging believer», diseño inicial si faltan datos, ruido ajustado, `log` opcional). Evidencia en `notebooks/29_optimizacion_bayesiana.json` | 9 | Lo eligió el owner tras M15c: primera capacidad nueva después de endurecer la columna no ideal | scikit-learn (BSD-3) |
| — | Layout/ruteo, 3D, árbol de fallas, BST | 10, 11 | Investigación o decisión del owner | — |

## Reglas de admisión de un motor nuevo

1. Licencia compatible con MIT en proceso (MIT/BSD/Apache/LGPL con enlace dinámico); GPL solo como proceso aparte.
2. Un caso de referencia con respuesta conocida **antes** de la tool; la tool lo reproduce en `tests/`.
3. Entradas en SI con nombres explícitos; nada de defaults para precios, propiedades o datos del usuario.
4. Salida con `evidence` propia del motor (convergencia, independencia de malla, error vs analítico) que decide el sello.
5. Si corre en otro entorno (micromamba, binario), la tool lanza el proceso con timeout y mata el grupo al vencer,
   igual que `run_claude_p`.
6. Lo que es hipótesis (difusividad estimada, propiedad de orden de magnitud) va en la respuesta como tal y en `MOCKS.md`.
