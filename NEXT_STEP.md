# NEXT_STEP — cursor

Actualizado: 2026-11-13 (M1-M15b cerrados: etanol-agua en la columna EO con NRTL)

## Dónde estamos
**Bloque M15b (2026-11-13): etanol-agua cerrado en `eo_column_optimize` con NRTL, 69 tools (misma tool), ✅.**
- El Q_reb +6.5 % vs DWSIM no era nuestro: DWSIM 10.2.12 suma −R·T²·Σx·∂γ/∂T como «exceso» (debería ser ∂ln γ/∂T). Cierre término a término: 54.55 de 54.58 kW.
- Igual con igual (`excess_enthalpy=False` vs DWSIM «ExpData»): etanol-agua Q_reb +0.20 %, Q_cond +0.09 %; metanol-agua −0.49 %, −0.72 %.
- Arranque: continuación de las dos specs sobre el sistema cuadrado + homotopía de no idealidad + ε hasta que selle. Verificados 15 platos x_D 0.70 y 20 platos x_D 0.75/0.80/0.82 (34-87 s).
- McCabe-Thiele propio (`packages/eo/shortcut.py`) rechaza antes del NLP: sin etapas ni a reflujo total, reflujo máximo insuficiente, o producto a < 0.03 K del azeótropo.
- Metanol-agua de M15 sin cambios (RR 4.4911, Q_reb 1497.6 kW).

Pendiente (decisiones del owner marcadas):
1. **Owner**: h^E de etanol-agua pesa −3.8 % en Q_reb y viene de parámetros de VLE. ¿Se deja por defecto (consistente con Gibbs-Helmholtz) o se busca h^E calorimétrico publicado y parámetros con dependencia en T?
2. **Owner**: ¿reportar a DWSIM la fórmula del término de exceso?
3. x_D +0.9 a +1.1 % vs DWSIM en etanol-agua: atribuido a la Psat de DWSIM (T_D +0.30 K), sin auditar.
4. `simulate_column_nrtl` (RR y D fijos) sigue frágil en etanol-agua; con 1 o 3+ specs y en multicomponentes la ruta es el NLP de M15.
5. Destilado más allá de x 0.82 en etanol-agua: hace falta otra formulación de equilibrio (complementariedad en vez de SmoothVLE).
6. Medir el efecto de ε en la ruta ideal (M8b) y decidir si se le aplica la continuación (**owner**; no se midió en M15b).
7. Una columna no ideal publicada que imprima sus parámetros NRTL; UNIQUAC/Wilson; vapor no ideal; fronteras con 3+ compuestos.

Siguiente: **decisión del owner** (pendientes 1-7 o pasar a otro tema: layout, 3D, árbol de fallas, BST).

**Bloque M15 (2026-11-13): `eo_column_optimize` con NRTL, 69 tools (misma tool), ✅ con dos desviaciones declaradas.**
- `activity_model="NRTL"` + `binary_parameters` (usuario > resultado de `vle_binary_regression` > banco ChemSep de thermo; nunca por defecto). Rechaza pares sin parámetros, specs del otro lado de un azeótropo, 3+ compuestos con azeótropo y vapor con |B·P/RT| > 0.10.
- Caso publicado Simone 2015 (metanol-agua, Aspen/gPROMS): T a ≤ 0.4 K, Q_cond +0.2 %, metanol arriba −0.5 %, Q_reb −1.4 % (Aspen y gPROMS difieren 1.3 % entre sí).
- DWSIM con los mismos parámetros en el óptimo metanol-agua: todo ≤ 0.62 % (Q_reb +0.29 %, Q_cond −0.62 %).
- Hallazgo: el ε de SmoothVLE (0.01 K) mete vapor ficticio en líquidos saturados; en la ruta NRTL se continúa a 1e-4 K y el sello lo comprueba.
- Cadena regresión → columna verificada en etanol-agua (Kamihama 2012, 20 platos, x_D 0.70).

Pendiente al cerrar M15 (1 y 2 cerrados en M15b):
1. Etanol-agua de la cadena vs DWSIM: Q_reb +6.5 % (composiciones ≤ 0.86 %, Q_cond +0.88 %); causa no identificada (reflujo bajo: Q_reb lo domina el calor sensible/de mezcla de la alimentación subenfriada). Comparar h de la alimentación y de los fondos entre los dos motores.
2. Robustez del arranque en columnas acuosas: etanol-agua 15 platos con x_D 0.70-0.75 no converge; junto al azeótropo (x_D 0.80) el pellizco deja vapor ficticio y no hay sello. Probar inicialización por perfil (T y x lineales) o otra formulación de equilibrio.
3. Medir el mismo efecto de ε en la ruta ideal (M8b) y decidir si se le aplica la continuación (cambia sus números de referencia; decisión del owner).
4. Una columna no ideal publicada que imprima sus parámetros NRTL (la de Simone 2015 no los trae).
5. UNIQUAC/Wilson en la columna EO; vapor no ideal (Hayden-O'Connell, asociación); fronteras de destilación con 3+ compuestos.

**Bloque M14 (2026-11-13): regresión de parámetros binarios desde datos VLE, 69 tools, ✅ en su núcleo.** Paquete `packages/vle/` y tool `vle_binary_regression`:
- NRTL (α fijo o ajustado), UNIQUAC (r, q de grupos UNIFAC) y Wilson (V de thermo) desde Pxy, Txy o PTxy, uno o varios conjuntos.
- Dos funciones objetivo: punto de burbuja (T o P, más y) y máxima verosimilitud de Britt-Luecke. Salen IC 95 %, correlación, AAD y residuos por punto.
- Consistencia por conjunto: Redlich-Kister/Herington, Fredenslund, Van Ness directa y desviación ΔP/Δy. También el azeótropo de los datos contra el de cada modelo y UNIFAC Dortmund.
- Cada ajuste se re-evalúa con las clases de `thermo`.
- Evidencia (16 tests):
  - Hartanto 2025: n-hexano + etanol, parámetros publicados dentro del IC 95 % con su misma función objetivo.
  - Etanol + agua de Kamihama 2012 y Voutsas 2011: D = 3.85 contra 3.8 publicado, y los dos consistentes.
  - Recuperación de parámetros sintéticos (≥ 24/30 dentro del IC) y rechazo de un conjunto que viola Gibbs-Duhem.

Pendiente:
1. **Página DECHEMA legible** (Gmehling y Onken, por biblioteca), para reproducir un ajuste publicado sin la correlación a-b de una sola isobara.
2. **Conjunto inconsistente publicado con datos legibles.** Hoy el rechazo se prueba con un conjunto sintético. Bredig 1927 y el SI de ACS Omega 2025 no se pudieron abrir.
3. Discrepancias de Hartanto 2025: la RMSD(T) de sus parámetros sale 0.79 K, no 0.38, y su Van Ness (ΔP 0.4 %) sale 1.9 %. Se podría cerrar con la isobara de 61.3 kPa de Ortega y Espiau.
4. LLE; varias fuentes con pesos de calidad (TDE); asociación en el virial (Hayden-O'Connell).
5. ~~Usar los parámetros regresados en `eo_column_optimize`~~ → hecho en M15 (NRTL).

Siguiente: **decisión del owner**: llevar los parámetros a la columna (M8b no ideal o DWSIM) o pasar a otro tema (layout, 3D, árbol de fallas, BST).

**Bloque M13 (2026-11-13): redes de tuberías y sistemas de bombeo, 68 tools, ✅.** Paquete `packages/network/` y tool `pipe_network`:
- Gradiente global de Todini-Pilati (el de EPANET) con contraste Hardy Cross; misma fricción que `pipe_hydraulics`.
- Mallas, varias fuentes y sumideros, reservorios, tanques y fronteras de presión fija; bombas en serie y en paralelo con la curva del usuario; curva de sistema por estación y curva combinada; NPSHa vs NPSHr; potencia; check, PRV y FCV.
- Residuo de masa por nodo ≤ 1e-9 m³/s (tolerancia en el código).
- Evidencia (35 tests): EPANET 2.2 Net1 y 5 variantes (corrido aparte vía WNTR); Jeppson 1974 (4 redes); Larock Ex. 2.7; Evett y Liu 12.11; Cameron NPSH 1-4; límite de un tubo exacto; paralelo y serie en forma cerrada.

Pendiente:
1. Un ejemplo publicado de **bombas en paralelo con su curva legible** (Larock Ex. 2.5, Karassik o Crane por biblioteca).
2. El Hardy Cross clásico de Jeppson pp. 74-75 viendo la figura: la solución impresa no cierra (¿demanda o tubo mal transcrito?).
3. Simulación en el tiempo (tanques que se llenan y vacían, controles) y transitorios (golpe de ariete): fuera de alcance.
4. Leer la red desde un `.inp` de EPANET o desde el flowsheet de DWSIM.

Siguiente: **decisión del owner** — pulir M13 (`.inp`, simulación en el tiempo) o pasar a otro tema (layout, 3D, árbol de fallas, BST).

**Sub-bloque M12b (2026-11-13): consecuencias ampliadas, 67 tools, ✅** (las cuatro tools de M12 crecen, sin tools nuevas):
- `explosion_overpressure(method="multi_energia")`: ajustes de Díaz Alonso 2006 (tesis abierta). La clase 3 lleva una errata corregida y documentada. Reproduce YB §5.6.3, C&L 6-12 y el anexo II.2 de la tesis.
- `fire_radiation(kind="chorro")` con Chamberlain del TNO: el ejemplo YB §6.6.2 se reproduce paso a paso (0.417 kW/m²).
- `gas_dispersion`: la nube densa a nivel del suelo recibe distancias de **Britter-McQuaid** (LNG 367 m) en lugar de quedarse sin resultado.
- `release_source_term(kind="charco_ebullicion")`: YB §3.6.4.1 (2.71 kg/s, 108 kg).

Pendiente:
1. **Baker-Strehlow-Tang**: hacen falta curvas legibles (Tang y Baker 1999 o Díaz Alonso 2008 por biblioteca, o el PDF de h2tools abierto a mano en el navegador). Ya está la tabla de Mf.
2. Chamberlain con Rw > 0.05.
3. Cm* de Britter-McQuaid contra una copia impresa (C&L 3.ª ed. o CCPS).
4. HyRAM+ en proceso aparte.
5. CCPS CPQRA.

Siguiente: **decisión del owner** — conseguir BST (acceso a biblioteca) o pasar a otro tema (layout, 3D, árbol de fallas).

**Bloque M12 (2026-11-13): análisis de consecuencias, 67 tools, ✅ en su núcleo.** Paquete `packages/consequence/` y cuatro tools:
- `release_source_term`: líquido y gas por orificio (ahogado/subsónico), vaciado de tanque, fracción de flash (CoolProp o Cp), evaporación de charco.
- `gas_dispersion`: pluma y puff de Pasquill-Gifford (Briggs, rural/urbano), distancias a los umbrales del usuario y probit tóxico. Con el criterio de gas denso de Britter-McQuaid, una nube densa a nivel del suelo **no se calcula**.
- `fire_radiation`: charco (SFPE o TNO, factor de vista numérico), bola de fuego TNO, chorro como fuente puntual.
- `explosion_overpressure`: equivalencia TNT con Kinney-Graham y probits de sobrepresión.
- Cada tool devuelve `para_lopa`: con el criterio de severidad del usuario, `lopa_from_hazop` recibe la severidad etiquetada «calculada» (override nuevo `severidad`).
- Paso 0: sin librería abierta general (HyRAM+ es GPL-3 y solo H₂/CH₄/propano; Phast y ALOHA cerrados) → modelos publicados en Python puro.
- Clima, orificio, inventario, η, exposición y puntos finales nunca por defecto. Único EJEMPLO: EPA RMP 40 CFR 68.22(a)(2). Toda salida dice «cribado, no sustituye a Phast/EFFECTS en un QRA real».

Evidencia (`tests/test_consequence.py`, 79 tests):
- Crowl & Louvar 4.ª ed., ejemplos 4-1, 4-2, 4-4, 4-6, 3-9, 5-1, 5-2, 6-10, 6-11, 6-13 y 2-2. **Errata hallada**: el Cmax de 5-1(b) está impreso ×10.
- TNO Yellow Book 2005: charco confinado (4.581e3 W/m²) y no confinado, y la BLEVE (20.26e3; el 20.361e3 impreso es errata).
- NUREG-1805 ej. 5.11-1 (3.05 kW/m²), NISTIR 6546 problema 3 y probits de PGS 1.
- Límites: flujo ahogado en su frontera, gas denso en su umbral y bordes de las clases de Pasquill.

Pendiente (siguiente sub-bloque, en orden):
1. **Multi-energía TNO y Baker-Strehlow-Tang.** Hace falta un ajuste de curvas con fuente legible: Díaz Alonso 2006/2008, o el informe MSB 2017 de Johansson, que no se encontró. Tests listos para cuando lo haya: C&L ej. 6-12 (R̄ = 2.4 → 13.2 kPa) y YB §5.6.3 (tablas 5.6/5.7).
2. **Chorro de fuego Chamberlain**, con YB §6.6.2 como test: 30 pasos, 0.417 kW/m², 10 erratas anotadas en la nota de la sesión.
3. **Modelo de gas denso** (correlaciones de Britter-McQuaid), para que la nube densa tenga distancias.
4. Charco en ebullición (criogénico, C&L ec. 4-107/4-108).
5. HyRAM+ como proceso aparte para contrastar H₂/CH₄.
6. CCPS CPQRA cuando haya copia legible.

Siguiente: **decisión del owner** sobre el sub-bloque M12b (multi-energía/BST, Chamberlain, gas denso), o volver a los temas de investigación (layout, 3D, árbol de fallas).

**Bloque M11 (2026-11-13): LOPA y SIL desde el HAZOP y el P&ID, 63 tools, ✅.** Paquete `packages/lopa/` y tool `lopa_from_hazop`.
- Una fila del HAZOP = un escenario; cada causa = un evento iniciador de una tabla genérica con fuente, ligado al lazo real del P&ID cuando es una falla del BPCS.
- Capas acreditables: las PSV, alarmas y checks del P&ID trazadas a esa fila, más las que declare el usuario. Seis reglas de independencia deterministas (I-1…I-6) sobre los tags.
- Cada disparo del P&ID recibe PFD requerida, RRF y SIL (IEC 61511-1, demanda baja o alta).
- La frecuencia tolerable nunca va por defecto: sin ella, «sin criterio» y criterios de EJEMPLO etiquetados.

Evidencia (`tests/test_lopa.py`, 49 tests):
- Reproducidos: HSE/BSTG 2007 App. 1 (5.1e-7, con el «-» de IPL1 en IE3 saliendo de la regla I-1) y su variante SIL 2 (2.3e-7; errata del documento en la columna IE1), dos ejemplos de Kenexis (RRR 10 y 100) y Casey 2019 (7.125e-5).
- **CCPS 2001 no se reprodujo: no hubo copia legible.** Sus valores entran vía fuentes secundarias (SAFEChE/Crowl-Louvar, Casey).
- Bandas SIL en sus bordes; independencia probada en los 10 HAZOP publicados; extremo a extremo DEXPI → P&ID (XSD oficial) → LOPA con tags consistentes, más 2 flowsheets de DWSIM.

Pendiente:
- árbol de fallas;
- verificar la PFDavg de la SIF (arquitectura, intervalo de prueba);
- escribir el SIL en el DEXPI;
- tablas CCPS 2015;
- causas fuera de la tabla (taponamiento, runaway…), que hoy quedan por definir.

Siguiente: **decisión del owner** — criterio de riesgo propio para usar la LOPA en un caso real, y qué tema de investigación sigue (layout, 3D, dispersión, árbol de fallas).

**Bloque M10 (2026-11-13): P&ID mínimo desde el HAZOP en DEXPI P&ID, 62 tools, ✅.** Paquete `packages/pid/` y tool `pid_from_hazop`.
- Entrada: un flowsheet resuelto o un DEXPI Process.
- Reglas deterministas, cada una con su fuente:
  - nivel en todo inventario de líquido;
  - presión en espacios de gas;
  - flujo en la descarga de bombas y compresores;
  - temperatura en calentadores, enfriadores y reactores;
  - columna LV de Skogestad;
  - cascadas;
  - aislamiento en límites de batería y rotatorios.
- Elementos que salen del HAZOP: PSV, check, alarmas (severidad ≥ 3) y disparos (≥ 4), solo donde una fila los pide.
- Cada elemento lleva tag ISA-5.1 y su trazabilidad. Las consignas quedan «por definir».
- Salida: XML y JSON DEXPI 2.0.1, más tablas en español.

Evidencia (`tests/test_pid.py`, 31 tests):
- Cobertura exacta en 22 casos: 12 de DWSIM y 10 HAZOP publicados. Sin espurios, y la prueba de mutación los detecta.
- Ida y vuelta en pyDEXPI idéntica (XML y JSON), con 0 avisos del validador.
- **XSD oficial DEXPI 2.0.1 válido** en los 22 casos y también en los DEXPI Process de `export_flowsheet`.
- Skogestad 1997 Fig. 1 (LV) más el lazo de temperatura de 2007, reproducidos en 5 columnas.

Pendiente:
- dibujar el P&ID (hace falta acomodo; es el hueco de layout);
- tipos contra `Plant.xml` oficial (`data.dexpi.org` no respondió);
- bombas de desplazamiento positivo;
- L/D > 6;
- SIL por LOPA.

Siguiente según el mapa: solo quedan temas de investigación (layout y ruteo, 3D, LOPA, dispersión). **Decisión del owner.**

**Bloque M9 (2026-11-13): control de procesos, 61 tools, ✅.** Paquete `packages/dynamics/` y tres tools:
- `fopdt_identify`: ajuste FOPDT/SOPDT a la prueba escalón del usuario (mínimos cuadrados, IC 95 %, autocorrelación de residuos), contrastado con Smith y Sundaresan-Krishnaswamy;
- `pid_tuning`: SIMC (recomendada, τc = θ por defecto y declarado), IMC, ZN (último ciclo y curva de reacción), Cohen-Coon y Tyreus-Luyben, cada una con fuente y rango de validez, más la regla de la mitad para modelos de orden alto;
- `pid_closed_loop`: GM, PM, Ms, Mt, sobrepaso, asentamiento e IAE de consigna y de carga, con el retardo exacto.

Evidencia (`tests/test_control.py`, 53 tests):
- Skogestad 2003 Tabla 2 exacta; **Tabla 4: 22 de 25 filas** (ajustes, Ms e IAE sobre el modelo completo); Tablas 5-7 (ZN, Tyreus-Luyben; ZN-PID en retardo puro inestable).
- Seborg Ej. 12.4: IMC 0.5/100, Skogestad 0.5/8, integrador 0.556/5.
- FOPDT sintético con 1 % de ruido recuperado (τ 4.965 vs 5, verdad dentro del IC 95 %).
- Márgenes = python-control (Padé 8) a ≤ 1e-6; error integrado = forma cerrada.

Tres filas del artículo no cuadran con su propia regla (E5-PID, E8-PID, E3-PID; detalle en `APP.md`). La prueba escalón desde DWSIM dinámico queda pendiente: la Fluent API no expone la dinámica. 289 tests en verde (5 omitidos). Siguiente según el mapa: **bloque 10, P&ID mínimo desde HAZOP (DEXPI)**, en otra sesión.

**Bloque M8c (2026-11-13): `optimize_design_surrogate`, 58 tools, ✅. Bloque 8 cerrado.** La tool hace cuatro cosas:
- muestrea un flowsheet de DWSIM (hipercubo latino más las esquinas; conjunto apartado independiente);
- entrena una red tanh por salida (numpy y scipy; sin dependencias nuevas);
- la embebe con OMLT (`ReducedSpaceSmoothNNFormulation`) y resuelve con ipopt desde 6 arranques;
- vuelve a correr DWSIM en el óptimo y sella solo si DWSIM lo confirma.

Evidencia:
- **Compresor de 3 etapas a 27 bar:** error apartado 0.037 %. Óptimo 3.026/9.076 bar contra 3.021/9.087 directo (malla 21×21: 3.025/9.100). DWSIM 891.838 kW contra 891.837.
- **Con descarga ≤ 430 K:** 2.6997/8.579 contra 2.699/8.591; 892.674 kW contra 892.676.
- **Columna hexano-heptano** (RR × fondos, purezas ≥ 0.95): error apartado en purezas 0.0009 (tolerancia 0.002). RR 2.2314 contra 2.2322 directo; DWSIM 4106.9 kW con purezas 0.95000. `eo_column_optimize` queda a 2.6 % en RR por la base de propiedades.
- Una discrepancia en la verificación da `not_verified`.

Con DWSIM rápido (0.05 s por corrida), el directo es más rápido; el sustituto paga con flowsheets lentos o con más variables. 236 tests en verde. Siguiente: **M9 control** (`pid_tuning`, python-control), en otra sesión.

**Bloque M8b (2026-11-13): `eo_column_optimize` para cualquier mezcla ideal, 57 tools, ✅.** La misma tool ahora recibe `components` (2 o más), fracciones de alimentación y `specs` de pureza o recuperación. El paquete de propiedades de IDAES se genera de `chemicals`/`thermo` (Psat Wagner-McGarry, cp gas Poling, densidad DIPPR 105, cp líquido ajustado a thermo, ΔHvap = DIPPR 106 en Tsat), con las fuentes citadas y comprobado en cada llamada. UNIFAC Dortmund rechaza las mezclas no ideales (azeótropo o |ln γi/γj| > 0.10). Evidencia: misma fuente a ≤ 2e-11 (sin errores de unidades); BT reproduce M8a (Q_reb −1.6 %, RR 0.5 %); **hexano-heptano vs DWSIM en el óptimo: Q_reb 0.55 %, Q_cond 0.39 %, T 0.03 K**; BTX con recuperaciones, alimentación subenfriada: DWSIM Q_reb 0.87 %; multiplicadores = diferencias finitas ≤ 0.16 %. Siguiente: **M8c OMLT** (sustitutos), en otra sesión.

**Bloque M8a (2026-11-13): ipopt + `eo_column_optimize`, 57 tools, ✅.** ipopt de idaes-ext 3.4.2 (`idaes get-extensions`). Columna benceno-tolueno de IDAES optimizada como un NLP: carga mínima del rehervidor con dos purezas, multiplicadores = MW por unidad de pureza. Evidencia: caso de prueba de IDAES reproducido; vecinos ± 1 % y multiplicadores = diferencias finitas (0.16 %); DWSIM en el óptimo a 0.4-0.5 % en cargas. Hallazgo: `BT_ideal` de IDAES tiene el cp líquido 1000× menor (unidades J/kmol); corregido en copia propia (ΔHvap ≤ 2.3 % de DIPPR). 216 tests en verde.

**Bloque M7 (2026-11-13): `nonisothermal_tubular_reactor`, 56 tools, ✅.** Campo 2-D acoplado c–T en FEniCSx (proceso aparte, P2 + SUPG, Newton con continuación en ΔH y pseudo-transitorio) contra el PFR 1-D no isotérmico (scipy): X, T de salida, punto caliente (T, z, r), perfiles radiales y dónde divergen. Evidencia: cierre de energía ≤ 1e-5; malla 2×; ΔH = 0 = M1 (0.78189); adiabático = PFR a 3e-6; modelo publicado de Rawlings & Ekerdt (fig. 6.35, o-xileno) reproducido por el PFR y por el campo con mezcla radial fuerte (T_max 719.365 vs 719.35 K). Sin caso 2-D publicado legible; con λ_er finito ese tubo se desboca y el solver lo reporta como no convergido. 210 tests en verde.

**Bloque M5b (2026-11-13): `pinch_from_flowsheet`, 55 tools, ✅.** Toma calentadores, enfriadores e intercambiadores de un flowsheet resuelto, encadena tramos de una misma corriente, segmenta CP con flashes de DWSIM (cambio de fase localizado a 0.1 K), servicio actual = calentadores/enfriadores y corre `pinch_analysis`. Evidencia: ΔH de corrientes = duty de DWSIM (tol. 1e-3, medido ≤ 1e-7); Kemp armado en DWSIM con intercambiador: CP lineal 20.0/60.0 kW, pinch 90/80 °C; cp real del agua 19.1/59.1 kW; latente de vapor a 1 atm 225.7 kW vs IAPWS 225.6 kW. 195 tests en verde. No extrae condensadores/rehervidores ni reactores (los lista).

**Bloque M6b (2026-11-10): `stirred_tank_cfd`, 54 tools, ✅ (M6b-2 cerró el sello).** Rushton con mamparas por MRF k-ε: Np, P, P/V, θ95 (Grenville, correlación). Aspa a nivel 3 de refinamiento y `maxBoundarySkewness 4` (todas las mallas pasan checkMesh). El GCI no aplica (el aspa de 3 mm salta con la malla de fondo: 49 → 5.13 y 50 → 4.81 en el mismo caso), así que el sello es una **banda entre 3 mallas**: cada malla gruesa a < 10 % de la fina, reportada como incertidumbre. Corrida de punta a punta verificada: Np 4.85 ± 3.5 % (64/49/38 → 4.85/5.02/4.70), torque 0.7-2.4 %, 285 W/m³, 93 min.

**Bloque M6a (2026-11-09): `vessel_residence_time_cfd`, 53 tools.** OpenFOAM simpleFoam + edad media local en un tanque prismático; τ, σ²/τ², N, volumen estancado y conversión real vs CSTR; teorema edad de salida = V/Q y estudio de malla. Sin impulsor (M6b).

**Bloque M5 (2026-11-09): `pinch_analysis`, 52 tools.** Tabla del problema, MER, curvas compuestas y gran compuesta, unidades mínimas, barrido de ΔTmin y ahorro vs servicios actuales; Kemp (20/60 kW) y Smith (7.5/10 MW) reproducidos; evidencia = curvas desplazadas que se tocan sin cruzarse.

**Bloque M4c (2026-11-09): `shell_side_bell_delaware`, 51 tools.** Jc/Jl/Jb/Js/Jr y Rl/Rb/Rs con banco ideal Žukauskas (ESDU de contraste), claros TEMA por defecto declarados en `assumptions`, Kern de comparación. Evidencia = gráficas vs ajustes HEDH y dos bancos ideales; sin ejemplo publicado legible todavía.

**Bloque M4 (2026-11-09): fluids/ht como tools, 50 tools.** M4a: `relief_valve_sizing` (API 520 + API 526), `control_valve_sizing` (IEC 60534-2-1),
`fittings_resistance`, `compressible_gas_line` (isotérmico propio + EDO), `two_phase_pressure_drop`; M4b: `shell_side_kern`, `nucleate_boiling`, `film_condensation`.
Ejemplos de API 520 (1, 2, 4, 5), IEC (1-3) e Incropera 10.1 reproducidos; 16 tests; reglas 13-14 del agente; 163 tests en verde.
**Bloque M3 (2026-11-08): economía completa, 42 tools** (`bare_module_factors`, `capital_investment`, `cash_flow_analysis`): F_BM de Seider leídos
de BioSTEAM, inversión total con porcentajes del usuario, flujo de caja = fórmulas cerradas y = funciones de BioSTEAM; 15 tests; regla 12 del agente.
**Bloque M2 (2026-11-08): Cantera como tools, 39 tools** (`adiabatic_flame`, `kinetic_reactor_network`): notebook 03 reproducido por la tool y por MCP
(llama 2226 K vs completa 2327 K, PFR/CSTR = analítico, ignición 4.53 ms, apagado 0.2-0.5 ms, S_L 0.378 m/s); 7 tests; regla 11 del agente.
**Bloque M1 (2026-11-08): `tubular_reactor_field`, primer motor externo (FEniCSx como proceso aparte, 37 tools)**: Caso 1 reproducido por la
tool y por MCP (pistón 0.7940 = analítico, laminar 0.7819, penalización 1.2 puntos, Re 2000, sello verificado con 4 comprobaciones); 6 tests.
Mapa de 12 áreas en `docs/MAPA_MOTORES.md`.
**Bloque 29 (2026-11-06): bench completo con tres modelos** (`notebooks/22_bench_tres_modelos.md`): una pasada de las 50 tarjetas
por modelo: Opus 97.0 % (99.0 % con `reciclo` corregida), Sonnet 94.0 %, Haiku 92.4 %; trampas 15/15 · 14/15 · 15/15; la brecha está en v3
(95/82/77 %). Opus detectó que `reciclo` no daba T ni P del agua fresca: corregido. Datos en `notebooks/bench_runs/m2/`.
**Corrección 2026-11-06: deriva de la alimentación al reconstruir columnas** (óptimo del bloque 26 no reproducible; corregido a 29 etapas, R 1.394, 885 080 USD/año; tarjeta y calificaciones actualizadas: Opus 100 % en una pasada).
**Bloque 30 (2026-11-06): constantes de Fluxo** (`notebooks/21_fluxo_constantes.md`): 166/167 dentro de tolerancia; ω H₂S 0.090 → 0.100.
**Bloque 26 (2026-11-05): variables enteras + costo de platos/relleno** (`notebooks/20_variables_enteras.md`): etapas enteras × reflujo en la
columna etanol-agua → 32 etapas, R 1.267, R/R_min riguroso 1.12; platos de Seider exactos; tarjeta `optimo_etapas_reflujo` (50 tarjetas).
**Bloque 25 (2026-11-05): conjunto apartado de HAZOP** (`notebooks/19_hazop_apartado.md`): 5 HAZOP publicados nuevos; recall a ciegas
25/30 = 83 % → 30/30 con reglas de sustancia/régimen térmico; precisión útil 69 % → 76 %, espurias 2.6 % → 0 (revisor independiente,
rúbrica previa). El owner pidió seguir en orden: 26 variables enteras + costo de relleno, 27 Fluxo 1.1 UI Figma, 28 repo público + post.
**Bloque 24 (2026-11-05): optimización multivariable** (`notebooks/18_optimizacion_multivariable.md`): `optimize_design_multi`
(36 tools), malla + Nelder-Mead; compresión 3 etapas 3.021/9.087 bar vs teoría 3/9; con límite del HAZOP 2.699/8.591 bar;
absorbedor aceite + enfriamiento con óptimo de esquina sensible al precio. Bench v3 = 10 tarjetas (49 en total).
Bloques 18-23 del mismo día: transporte, costo, costo anual, optimizador 1D, masa, hidráulica.
Fluxo: rama `feature/reactor-cinetico`, sin tocar desde el bloque 12.

## Siguiente acción exacta
Orden fijado por el owner (2026-11-05, «sigue con esos 4 en orden»):
1. ✅ Conjunto apartado de HAZOP (bloque 25).
2. ✅ Bloque 26: variables enteras + costo de platos y relleno.
3. ✅ **Bloque 27** (Figma aprobado por el owner e implementado en SwiftUI en `apps/fluxo` rama `feature/reactor-cinetico`, commit del 2026-11-05; falta TestFlight/QA físico). Antes:: pantallas Figma `iPhone/11 Nuevo reactor cinético` (23147:861) e `iPhone/12 Base · NRTL` (23147:979) en ZRNmiWDmd6NjWIU0bfW6Fi; falta revisión del owner + Premium Brand Gate. Apple rechazó Fluxo 1.0.0 (4.3.0 Spam, 2026-10-20); el owner respondió 2026-10-27; sin veredicto.
4. ✅ **Bloque 28**: repo PÚBLICO desde 2026-11-05 (OK del owner «Aprobado y sigue»); post listo en `docs/posts/01_lanzamiento_alchemai.md` con el enlace, lo publica el owner en LinkedIn.

**Decisión del owner 2026-11-06: se siguen AlchemAI (producto principal, sistema con IA en la Mac) y Fluxo (app iOS, canal móvil del motor) en paralelo.**

Siguiente (plan mes 2):
1. ✅ Bloque 29: tabla de tres modelos.
2. ✅ Bloque 30: constantes de Fluxo.
**Visión del owner 2026-11-08: una sola aplicación que cubra todo lo que hacen los software de ingeniería química** (calor, masa, momento,
balances, reacción, separación, costos, seguridad, diseño de planta); el agente elige el motor. Mapa por área en `docs/MAPA_MOTORES.md`
(12 áreas, estado ✅/🧪/📦/⬜ verificado, orden propuesto de 10 bloques). Primer caso de dos motores: `transporte-lab/reactor/` (DWSIM PFR
0.7945 vs FEniCSx laminar 0.7818). M1 `tubular_reactor_field` ✅, M2 Cantera ✅ y M3 economía ✅ 2026-11-08. M4 fluids/ht ✅ 2026-11-09 y M4c Bell-Delaware ✅ (51 tools; falta validarlo contra un ejemplo publicado con datos legibles). M5 pinch ✅ (52 tools). M6a tanque de flujo continuo CFD ✅ (53 tools). M6b viabilidad ✅ 2026-11-10 (caso 3 de `transporte-lab`: Rushton MRF, Np 4.67 vs ≈ 5, balance de torque 0.26 %, 19 min con `mpirun -np 4` desde `$TMPDIR`). M6b-2 ✅ 2026-11-10: sello por banda entre mallas, Np 4.85 ± 3.5 % verificado. M5b ✅ 2026-11-13 (55 tools): `pinch_from_flowsheet`. M7 ✅ 2026-11-13 (56 tools): `nonisothermal_tubular_reactor`. M8a ✅ 2026-11-13 (57 tools): `eo_column_optimize`. **Siguiente: M8b (cualquier mezcla ideal en IDAES con parámetros verificados de `chemicals`) y M8c OMLT (sustituto de red neuronal sobre datos de DWSIM embebido en Pyomo).**
3. **Bloque 31**: borrador del whitepaper ES/EN «LLM agents for process design: a verified benchmark» desde BENCHMARK.md (publicar = OK del owner).
4. ✅ Bloque 32: modelos sustitutos con OMLT = M8c (`optimize_design_surrogate`, 2026-11-13).
5. Fluxo: exportar casos a JSON/SFILES.
**Decisión del owner 2026-11-06: la UI es una app de Mac en SwiftUI** («App Mac Swift»). Estructura en `docs/MAC_APP_ESTRUCTURA.md`
(agente en Python como proceso hijo, DWSIM aparte por GPL, Developer ID fuera de la App Store, repo privado `apps/alchemai-mac`).
Plantilla elegida por el owner 2026-11-06: macOS 27 + patrones del AI Agent UI Kit. **Diseño Figma listo** (Cb75xN1JHDXuPpUOzz37VQ, 7 pantallas, capturas en `docs/design/mac/`): **APROBADO por el owner 2026-11-06**. `alchemai serve` (`packages/serve/cli.py`) ✅ y app `apps/alchemai-mac` B1 ✅ 2026-11-06 (E2E real en la app). Siguiente en la app: B2 diagrama; luego Premium Brand Gate (ícono).

## Bloqueante actual
Ninguno. Limitaciones anotadas: PR no condensa 1 % de agua a 280 K / 60 bar; `CSTRBuilder.WithIsothermalTemperature` roto
en DWSIM 10.2.12 (usar `Isothermal()`); un `solve()` de DWSIM puede no regresar (columna PR polar; también una vez la columna shortcut con Sonnet): el arnés ya mata el grupo al vencer el timeout, pero el servidor MCP no tiene timeout por llamada; en Fluxo usar NRTL para polares (solo motor; la UI aún no lo expone).

## Decisiones pendientes del owner
- Publicar el post de lanzamiento en LinkedIn (lo hace el owner; repo ya público).
- App de Mac: plantilla de Figma (bloqueante), distribución fuera de la App Store (recomendada), cobro y precio.
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
