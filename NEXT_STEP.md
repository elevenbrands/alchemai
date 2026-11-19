# NEXT_STEP — cursor

Actualizado: 2026-11-19 (prueba con libro, 146 tools; antes M26: plantas completas con precios sugeridos con fuente, 135 tools; antes M25, momento completo)

## Dónde estamos
**Prueba con libro (2026-11-19, 146 tools).** Finlayson, *Introduction to Chemical Engineering Computing*: capítulo 2 hecho (8/20 → 20/20 tras `cubic_eos_volume`). Faltan los capítulos 3 a 11 (~72 problemas). Material de la prueba (texto, enunciados parafraseados, corridas) en `../alchemai-libros/finlayson/`, fuera del repo por derechos de autor.

**M27 (2026-11-16): una pregunta → planta completa, 143 tools.** `process_dossier` (refinería de Tula u otras 6 de Pemex; plantas modelo) da explicación, rentabilidad, PFD con oportunidades y reporte (`docs/plantas/tula/`). Siguiente: (1) aprobación del owner de la pantalla Figma «08 Planta completa (dossier)» (nodo 12:832) y programarla en la app Mac (dibujar la escena JSON de `process_diagram` de forma nativa); (2) caso Laguna Verde (pendiente del owner); (3) bajar la desviación de diésel de Tula (+41.5 %) con rendimientos de FCC por conversión.

**Hecho 2026-11-16:** Laguna Verde (`docs/plantas/laguna_verde/`) y Dos Bocas (`docs/plantas/dos_bocas/`, modelo débil: diésel y GLP en el límite de la calibración). FCC por conversión y diluyente del combustóleo hechos (Salina Cruz y Salamanca validan diésel); Tula +44.3 % y Dos Bocas siguen sin explicar con fuentes abiertas.

**M26 (2026-11-16): plantas completas, 135 tools.** «Desarrolla una planta de X» ya da proceso, balances, equipos, inversión, costo de operación, flujo de caja, precio mínimo, sensibilidad, reporte y HAZOP preliminar, sugiriendo los datos económicos desde el banco con fuente (regla 31). Casos: `notebooks/26_planta_biodiesel_higuerilla.py` y `26_planta_etanol_cana.py`. Siguiente propuesto al owner: M27 «escáner de oportunidades desde datos de planta» (variables crudas del historiador o Excel → lista de oportunidades con su valor). App Mac: certificado Developer ID creado e importado 2026-11-16; falta reempaquetar el motor con M26, firmar, notarizar y probar el DMG.

**Decisión del owner (2026-11-14): el primer cliente son las plantas con vapor.** Definición del cliente, el problema, la oferta (diagnóstico de vapor) y las hipótesis sin validar en `docs/CLIENTE_Y_PRODUCTO.md`. Motivo: DWSIM ya trae su propio asistente de IA, así que «IA que arma simulaciones» no es lo nuevo; lo que se vende es el estudio comprobado. 0 conversaciones con plantas hasta hoy.

**M25 (2026-11-15): límites de M24 y transferencia de momento, 114 tools.** Pedido del owner: «arranca y termina límites y después momento». Hecho y verificado (detalle en APP.md y en las filas del mapa):
- PSA con Langmuir validado contra teoría de equilibrio (sin ciclo publicado: artículos de pago).
- Balance de energía en el flash y la columna dinámicos; quedan fuera la columna multicomponente, la energía del flash dentro del flowsheet y el modo dinámico de DWSIM.
- Hornos contra cuatro hogares; tubos con espárragos sin validar; Ganapathy sigue sin fuente legítima.
- Transporte de propiedades contra NIST (19 fluidos); líquido cerca del punto crítico hasta 50 % (marcado).
- RL con escudo y veredicto empírico por política; no es garantía formal.
- Momento: 8 tools nuevas (no newtonianos, sedimentación, fluidización, ciclones, Fanno, dos fases inclinado, golpe de ariete, bombas). Bingham turbulento (Darby) −24 % contra un punto medido y ΔP del ciclón en conflicto: declarados.
- Errores de DWSIM en Beggs-Brill inclinado: S ya corregido en su `master`; N_LV reportado el 2026-11-15 en https://github.com/DanWBR/dwsim10/issues/140. Cuando DWSIM lo corrija y salga en una versión, actualizar DWSIM y volver a correr `tests/test_momentum.py::test_against_dwsim_pipe_segment` (sus tolerancias de subida y bajada asumen los dos errores).
- Suite completa (2026-11-15): 1230 passed, 22 skipped. Se corrigió una dependencia de orden: thermosteam (BioSTEAM) deja `RAISE_PROPERTY_CALCULATION_ERROR = True` global en thermo; `compound_properties` lo fija mientras calcula y lo restaura.

**M24 (2026-11-14): las 53 capacidades del mapa en ✅, 106 tools.** Pedido del owner: «termina las 53 hoy». Se hicieron en paralelo con agentes (primero con Fable, que se quedó sin cupo a medias; se terminaron con Opus desde lo que ya estaba en disco) y cada bloque lo verifiqué: corrí sus pruebas y una comprobación propia a mano (η de una esfera a φ = 3, serie de Crank, MSMPR, permeado en mezcla completa, IAPWS, Fenske). Detalle por fila en `docs/MAPA_MOTORES.md` y validaciones en BENCHMARK (filas M24).
- Límites que siguen y que cada herramienta declara: PSA con isoterma no lineal sin validar; hornos con un solo horno radiante de referencia y sin tubos con espárragos; flash y columna dinámicos sin balance de energía; transporte de propiedades de puros (salvo agua) sin validar; MPC no lineal solo con óptimo local; una política de RL es un experimento.
- Fuentes: los ejemplos de Luyben y de Ganapathy se transcribieron de copias de autorización no verificada; se quitaron los enlaces y hay que cotejarlos con un ejemplar legítimo (MOCKS `FUENTES-COPIA-NO-VERIFICADA-M24`).
- Lo que sigue sin depender de código: ninguna planta piloto propia.

**M20 y M21 (2026-11-14): el diagnóstico de vapor quedó todo en uno, 80 tools.** El owner pidió terminar lo necesario para que el proyecto sirva a las empresas sin encadenar aplicaciones, y delegó las decisiones («toma las decisiones en pro de que sea lo más preciso y práctico»). Qué se decidió y se hizo:
- **Gas natural sin análisis (M20a).** `fuel = {type: 'natural_gas_unknown'}`: la composición se ajusta al poder calorífico del recibo (energía ÷ volumen) o al del proveedor: metano + etano, o metano + nitrógeno si es más pobre. Sin poder calorífico, gas de referencia «GULF-COAST» leído de CoolProp. Medido contra seis gases de referencia: eficiencia a menos de 0.15 puntos; CO2 0.9-3.2 % bajo. Módulo `packages/energy/fuel_reference.py`.
- **CO2, tope de inversión, revisión de datos, carga (M20a).** `baseline.emissions` por balance de carbono y CO2 evitado por medida; `max_investment_for_payback` (1, 2 y 3 veces el ahorro: no es una cotización); `data_review` señala datos sospechosos sin corregirlos; la pérdida por paredes publicada se divide entre la carga si se da la capacidad de placa.
- **Objetivos publicados (M20c).** `use_published_target` en exceso de aire (10 %, DOE Tip Sheet #4) y chimenea (250 °F, DOE Tip Sheet #3), solo gas natural y solo por nombre.
- **Pérdidas de distribución (M20b).** `distribution` = fugas, trampas falladas abiertas y tubería sin aislar. Motor `steam_distribution.py` (hecho por un subagente, 80 pruebas; revisé su reporte y comprobé un valor por mi cuenta) y valoración `steam_audit.py` con el costo y el CO2 del vapor de la línea base. Fugas y trampas reproducen las tablas del DOE entre −4 y +3 %. La tubería desnuda NO está validada contra el DOE (−5 a +8 % con la emisividad elegida).
- **Todo en uno (M21).** `./vapor plantilla archivo.xlsx` escribe la hoja que llena la planta en sus unidades; `./vapor diagnostico archivo.xlsx` la convierte en `reporte.html` (autocontenido, con indicadores y tres gráficas), `reporte.pdf` (con Chrome local), `reporte.md` y `resultado.json`. Tools `steam_workbook_template` y `steam_diagnostic_from_workbook`. Guía en `docs/GUIA_DIAGNOSTICO_VAPOR.md`; ejemplo con planta inventada en `ejemplos/diagnostico_vapor/`.
- Supuestos y elecciones mías, todos en MOCKS: `GAS-SIN-COMPOSICION-M20`, `OBJETIVOS-PUBLICADOS-M20`, `REVISION-DATOS-M20`, `CONSTANTES-DISTRIBUCION-M20`, `CAPTURA-EXCEL-M21`. Validaciones en `BENCHMARK.md`, filas M20 y M21.
- Pruebas: `tests/test_steam_diagnostic.py` (35), `tests/test_steam_distribution.py` (80), `tests/test_steam_intake.py` (10). Suite completa después de M23c (commit 706d65a, árbol limpio): 757 passed, 22 skipped.
- **Revisión independiente (2026-11-14):** un agente que no escribió el código llenó la hoja con otra planta inventada, en °F, psig y lb/h, y recalculó 23 cifras a mano: todas coinciden (0.0 a 1.2 %). Encontró errores de etiquetado y de redacción, no de cálculo; se corrigieron: la presión atmosférica calculada de la altitud ya no sale como dato de la planta (`combustion.site_altitude_m`), los avisos de captura aparecen igual en Markdown, HTML e indicadores, el aviso de fugas enormes entra a la revisión de datos, el reporte ya no muestra nombres internos fuera de la sección 6, incluye la copia de los datos como se capturaron (sección 9) y la electricidad de la caldera se pide por tonelada generada (`auxiliary_power_kWh_per_tonne_generated`). No se revisó el comportamiento en Excel real (las listas desplegables) ni las páginas 4 en adelante del PDF.

- **M22 (2026-11-14): datos medidos de calderas reales publicadas, y cierre de pendientes.** (1) Caldera de tubos de humo de una fábrica de alimentos en México (tesina de la UNAM, 2021): con su O2, chimenea y aire medidos, la eficiencia de combustión del motor es 82.30 % contra 82.26 % del analizador; exceso de aire 13.23 contra 13.18 %; CO2 10.36 contra 10.28 %. El diagnóstico completo (gas sin análisis, paredes supuestas) da 79.50 % por pérdidas contra 80.95 % por entradas y salidas. (2) Caldera acuotubular en Tucumán (2008) con análisis del gas: CO2 y lambda coinciden; sus pérdidas y su eficiencia directa difieren 14 puntos y el diagnóstico lo señala. (3) Ejemplo de combustóleo de la BEE, pérdida por pérdida: eficiencia a −0.10 puntos. Cuaderno 33 y `tests/test_real_boilers.py` (6 pruebas). (4) Los datos reales destaparon un fallo: gas a menos de 25 °C con pentanos detenía el motor de combustión; corregido. (5) Tubería desnuda validada: −0.3 % contra 3E Plus v4.1 y −2.3 a +0.5 % contra la tabla de ASHRAE; la emisividad pasó de 0.70 a 0.80, que es la que usa 3E Plus. (6) PDF revisado completo (13 páginas). (7) La hoja pasa por LibreOffice sin perder listas ni cambiar el resultado; se leen bien «8%» y los miles con coma, y la coma decimal se rechaza en vez de adivinarse.

- **M23 (2026-11-14): caso real de punta a punta y salidas para lo que la planta no tiene.** La caldera real de la tesina de la UNAM se capturó en la hoja de Excel y se corrió con `./vapor caso-real` (`ejemplos/caso_real_unam_2021/`): 81.57 % por pérdidas contra 80.95 % por entradas y salidas con los medidores de la propia planta (+0.62 puntos). Nuevo: pérdida por paredes de calderas de tubos de humo desde la tabla del fabricante (Cleaver-Brooks CB-7767), diagnóstico sin precio del agua y sin humedad del aire (avisado), y lectura de unidades y opciones escritas a mano, para que la hoja no dependa de las listas desplegables. Todo en MOCKS `SALIDAS-SIN-DATO-M23`.

- **M23c (2026-11-14): impresión y paquete piloto.** Al imprimir, la hoja se partía en 17 páginas con las columnas separadas; ahora sale en 5, horizontal, con todas las columnas juntas y el encabezado repetido. Carpeta `piloto/` con lo que se manda a una planta (plantilla, ejemplo de reporte de la caldera real, mensaje para copiar y pegar) y qué hacer cuando regresa la hoja. Nada se envió.

No cubierto (sigue abierto):
1. **Ninguna planta piloto propia.** Hay dos calderas reales medidas por terceros; ninguna llenó la plantilla ni recibió el reporte. La plantilla, el reporte y todos los supuestos son hipótesis hasta el primer piloto. Es lo único que falta para saber si esto sirve.
2. Una caldera por hoja y un punto de operación promedio: sin curva de carga ni varias calderas.
3. Válvulas y bridas sin aislar no se calculan. La tabla de tubería del DOE Tip Sheet #2 corresponde al método anterior a 2002 (con la ecuación de Heilman sale dentro de 5 %); el motor usa el método vigente y se valida contra 3E Plus y ASHRAE.
9. Excel: la hoja SÍ abre en Excel para la web de Microsoft (visor sin sesión, 2026-11-14): sin aviso de reparación, cuatro pestañas, formato y valores correctos. Falta solo desplegar una lista en Excel con edición (el visor es de solo lectura y el Excel de escritorio de este Mac está corrupto). La hoja acepta unidades escritas a mano si una lista no apareciera. Sí pasó el validador del Open XML SDK de Microsoft con 0 errores (Office 2010, 2019 y 365), se abre y se vuelve a guardar en LibreOffice sin cambios, y acepta unidades escritas a mano.
10. Pérdida por paredes de tubos de humo: tabla de un fabricante, respaldada en orden de magnitud por la guía de afinación publicada por la EPA (menos de 0.5 % a carga moderada). Las dos describen calderas en buen estado.
4. Combustibles líquidos y gas LP: solo con análisis o composición de la planta; los objetivos publicados no aplican.
5. Inversiones y costos de reparación: nunca se estiman.
6. Punto de rocío ácido, química del agua, emisiones distintas del CO2 directo.
7. Sin formulario web ni versión en la app de Mac: la captura es la hoja de Excel y el comando.
8. Filas del mapa que no sirven al diagnóstico (dinámica del flowsheet, membranas y secado, control avanzado, reactores multifásicos): se dejaron fuera a propósito.

**M19b (2026-11-14): pérdida por paredes desconocida → valor publicado, marcado como supuesto.** Decisión del owner («Usar valor publicado»). Antes el diagnóstico se negaba sin `combustion.setting_loss_fraction`, que una planta casi nunca tiene. Ahora, si la planta declara `combustion.setting_loss_unknown=true`, el motor usa 1.0 % del poder calorífico superior, de G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263, sección 4.2.5 (PDF de energy.gov leído hoy: «menos de 1.0 %» para calderas de tubos de agua cerca de plena carga y en buen estado; tomé el tope). Nunca en silencio: sin el dato y sin la declaración se sigue negando y el mensaje dice cómo declararlo; dar las dos cosas se rechaza. Marcado en el primer aviso (`PÉRDIDA POR PAREDES SUPUESTA, NO MEDIDA`), en `missing_data`, en `assumptions`, en la procedencia como referencia publicada y en el reporte (aviso bajo el resumen, tabla de pérdidas, sección 6). `setting_loss.sensitivity` repite la línea base con un punto porcentual más: en el caso de prueba la eficiencia baja 1.00 punto y el vapor cuesta 1.17 % más. Cambios: `packages/energy/steam_diagnostic.py`, `steam_report.py`, docstring de la tool, regla 28 del agente (solo pasa la declaración si la planta lo dice), `docs/DATOS_PARA_DIAGNOSTICO.md`, MOCKS `PERDIDA-PAREDES-SUPUESTA-M19B`, 3 pruebas nuevas (28 en `tests/test_steam_diagnostic.py`). Suite completa 647 passed, 22 skipped (antes 644 y 22). Siguen en 78 tools. No cubierto: la gráfica de la ABMA (Fig. 4 de la guía) no se leyó, así que el valor no depende del tamaño ni del tipo de caldera ni de la carga; no validado con ninguna caldera real; `combustion_efficiency` (M18) sola sigue exigiendo el dato.

**Bloque M19 (2026-11-14): diagnóstico de vapor, 78 tools, ✅ con límites declarados.** Es el primer producto para el cliente elegido. Antes un agente tenía que encadenar cinco tools a mano y nada calculaba medidas de ahorro; ahora llama a una y obtiene el estudio, y una segunda lo vuelve el reporte que lee el cliente.
- `steam_system_diagnostic`: de lo que una planta puede entregar (combustible y recibos o precios, agua, electricidad, presión, O2 o exceso de aire, chimenea, aire, pérdida por paredes, agua de alimentación, retorno de condensado, purga, horas y producción de vapor o consumo de combustible) da: (1) línea base: eficiencia con desglose de pérdidas, costo del vapor por tonelada y por GJ con desglose, combustible, agua, electricidad y costo al año; (2) cinco medidas, cada una repitiendo los mismos motores con un solo cambio: bajar el exceso de aire, bajar la chimenea (economizador o precalentador), bajar la purga, recuperar el calor de la purga (flasheo y/o intercambiador) y regresar más condensado; (3) el caso combinado en una sola corrida, junto a la suma de las individuales; (4) retorno simple solo si la planta da la inversión; (5) procedencia de cada cifra (dato de la planta, calculado con qué herramienta, referencia publicada con fuente y fecha) y la lista de lo que faltó y qué impide; (6) `warnings`, `assumptions`, `not_included` y `evidence` con `verified`.
- No repite física: orquesta `combustion_efficiency` (M18) y `utility_costs` (M17). Lo único nuevo en un motor es la clave opcional `blowdown_heat_recovery` del nivel de vapor de `utility_costs` (flasheo al desaireador e intercambiador al repuesto, con el balance de la isla cerrado).
- Qué pasa si falta algo: un dato esencial (precio del combustible o del agua, O2, chimenea, pérdida por paredes, purga...) → se niega una vez con la lista completa. Un objetivo → «no evaluada: falta el dato» con lo que hace falta. La inversión → ahorro sí, retorno no. Horas o producción → solo cifras por tonelada. Chimenea objetivo en o bajo el punto de rocío del agua → no se evalúa; el rocío ácido no se modela y se dice.
- `steam_diagnostic_report`: Markdown en español para un gerente de planta (resumen con tres números, línea base, a dónde se va la energía, medidas ordenadas por ahorro, caso combinado, lo no evaluado, de dónde sale cada número, supuestos, comprobaciones, glosario). Se arma sin modelo de lenguaje; siempre vuelve a correr el motor y rechaza un resultado con un número cambiado.
- `docs/DATOS_PARA_DIAGNOSTICO.md`: lista de una página de qué pedir a la planta, para qué y qué pasa si falta; una prueba la ata a las entradas reales del motor.
- Validación con los ejemplos de cinco Steam Tip Sheets del DOE (enero 2012, PDF de energy.gov leídos hoy; se pudieron abrir los cinco), corridos por la misma tool: #9 purga 72 604 contra 72 631 USD (−0.04 %); #8 condensado 121 740 contra 121 760 USD (−0.02 %); #10 calor de purga 8 940 contra 8 500 MMBtu (+5.2 %; −0.03 % contra su propia fórmula sin el redondeo de su tabla); #4 combustión 28 383 contra 29 482 MMBtu (−3.7 %, por el desfase de 0.4-0.6 puntos de eficiencia ya medido en M18); #3 economizador 286 224 contra 386 400 USD (−25.9 %: la hoja no es consistente con la tabla de combustión del propio DOE, que da 296 887 USD; el motor queda a −3.6 % de esa). Nada se ajustó para coincidir.
- Herramientas del DOE (verificado hoy en su repositorio y en energy.gov): MEASUR 1.10.0 (2026-10-21) está vigente, es gratuita y de código abierto con licencia permisiva de UT-Battelle; su módulo de vapor sucede a SSAT/SSMT (la página antigua del SSMT da 404) y modela caldera, turbinas, flasheo, desaireador y hasta tres cabezales. La física de este diagnóstico ya existe ahí. Una prueba de caldera de MEASUR se reproduce a +0.003 %. MEASUR no se instaló.
- Caso ILUSTRATIVO (inventado, no es un cliente; MOCKS `ENTRADAS-ILUSTRATIVAS-M19`), con la tarifa GDMTH de CFE nombrada como referencia y marcada en el reporte: vapor 11 700 777 MXN/año (354.6 MXN/t), eficiencia 78.6 % (PCS); cuatro medidas evaluadas, juntas 912 471 MXN/año (7.8 %) contra 1 175 454 de la suma; la quinta sale «no evaluada: falta el dato».
- Evidencia: `notebooks/32_diagnostico_de_vapor.py`, `.json` y `_reporte.md`, `tests/test_steam_diagnostic.py` (25 pruebas, sin red), filas M19 de `BENCHMARK.md`, MOCKS `ENTRADAS-ILUSTRATIVAS-M19`, `SUPUESTOS-DIAGNOSTICO-M19`, `DIAGNOSTICO-PENDIENTE-M19`. Suite completa 644 passed, 22 skipped (antes 619 y 22).

No cubierto en M19:
1. Ningún diagnóstico con datos de una planta real. La lista de datos y el reporte son hipótesis hasta que una planta los use.
2. Un solo punto de operación promedio: sin curva de carga, arranques ni eficiencia a carga parcial.
3. Sin rocío ácido, calderas de condensación, fugas, trampas de vapor, aislamiento, química del agua (la purga mínima no se comprueba), emisiones ni su costo.
4. La inversión de las medidas nunca se estima; sin cotización no hay retorno.
5. La pérdida por paredes: resuelto el 2026-11-14 (M19b, abajo en «Dónde estamos»). Si la planta declara que no la conoce se usa 1.0 % (PCS) de ORNL/TM-2001/263, marcado como supuesto; sin dato ni declaración el diagnóstico se sigue negando.
6. Gas natural de México sin referencia en el banco: el precio lo da la planta (recibos).
7. La hoja #3 del DOE no se reproduce y no hay tercera fuente para decidir; MEASUR no se usó, solo se leyó.
8. El reporte sale en Markdown; no hay PDF ni formato con marca.

**Bloque M18 (2026-11-14): hornos y calderas, 76 tools, ✅ parcial.** Fila «Hornos y calderas» del mapa: ⬜ → ✅ parcial.
- `combustion_efficiency` ✅: de un combustible (gas por composición molar, o líquido/sólido por análisis elemental con poder calorífico medido), el aire (temperatura, humedad) y el exceso de aire o el O2 medido (base seca o húmeda; CO medido opcional) da aire estequiométrico y real, gases húmedos y secos, HHV y LHV, punto de rocío del agua, llama adiabática (ruta de Cantera existente) y, con la temperatura de chimenea y la pérdida por paredes, la eficiencia en base HHV y LHV con el desglose de pérdidas, más combustible y aire para una carga. Entalpías de los polinomios NASA de Cantera. Nada de operación se supone.
- `fired_heater_radiant` (parcial): hogar de una zona bien mezclada. `rating` da pared divisoria, carga radiante, flujo medio y temperatura de pared de tubo; `sizing` da área y número de tubos con el flujo admisible del usuario.
- `fired_heater_convection` (sin validar contra literatura): banco de tubos lisos fila por fila hasta la chimenea, para que la temperatura de chimenea sea un resultado.
- Validación de la combustión: DOE Steam Tip Sheet #4 (25 valores) +0.30 a +0.64 puntos de eficiencia; ORNL/TM-2001/263 Tabla B.1 (180 valores) +0.28 a +0.58 puntos de pérdida; el resultado queda entre las dos tablas. Pares exceso ↔ O2 de ambas a ≤ 0.5 puntos. Horno de Al-Haj Ibrahim 2008: PCS +0.02 %, aire +0.09 %, gases +0.08 %.
- Sección radiante contra las cartas de Kern (vía FOSSEE): alpha 0.935 contra 0.937; factor de intercambio 0.626 y 0.525 contra 0.635 y 0.56; emisividad del gas +7 % a trayectoria corta y −20 % a trayectoria larga (≈ 2 % en la carga del ejemplo).
- Enlace con M17: la eficiencia se pasa a `utility_costs` como dato del usuario, en la misma base que el precio del combustible; la tool avisa cuánto se desvía el costo con la base equivocada (−9.8 % en el ejemplo). `utilities.py` no cambió.
- Evidencia: `notebooks/31_hornos_y_calderas.py` y `.json`, `tests/test_fired_heaters.py` (19 pruebas, sin red), filas M18 de `BENCHMARK.md`, MOCKS `EMISIVIDAD-GAS-M18`, `SUPUESTOS-HORNOS-M18`, `ENTRADAS-ILUSTRATIVAS-M18`, `HORNOS-PENDIENTES-M18`. Suite completa 619 passed, 22 skipped (antes 600 y 22).

No cubierto en M18:
1. Ningún ejemplo publicado completo de hogar reproducido. Kern, Lobo y Evans 1939, Wimpress, Couper y Serth no se pudieron abrir; CORE respondió con un desafío anti-bots (no se evadió). El ejemplo de Al-Haj Ibrahim 2010 no cierra consigo mismo (dice 1278 K; su ecuación da 966 K).
2. La eficiencia de Al-Haj Ibrahim 2008 (75.78 %) no se reproduce (73.2 %): su poder calorífico neto no es consistente con su gas.
3. La emisividad del gas (WSGG de Smith 1982, leída de una reproducción) difiere 20 % de las cartas de Kern a trayectorias largas; sin tercera fuente para decidir.
4. Convección: sin ejemplo publicado; tubos lisos solamente.
5. Sin flujo pico, zonas múltiples, llama luminosa, tubos aletados, rocío ácido, calderas de condensación, inquemados de sólidos, tiro ni caída de presión. La llama de equilibrio no sale con aire bajo 300 K ni con especies fuera de GRI-Mech 3.0 (se da la de combustión completa).

**Bloque M17 (2026-11-14): costo de servicios por región, 73 tools, ✅ con límites declarados.** Fila «Costo de servicios (vapor, agua, electricidad) por región» del mapa: ⬜ → ✅.
- `utility_costs`: de los precios que la planta paga (combustible por GJ, electricidad, agua) al costo de cada nivel: vapor por tonelada y por GJ entregado, calor a fuego directo, agua de torre por m³ y por GJ, refrigeración por GJ a cada temperatura, electricidad efectiva con tarifa horaria y cargo por demanda. Devuelve el dict `prices` (USD) de `annual_cost`/`flowsheet_cost`, el desglose (combustible, agua, electricidad), de dónde vino cada precio, supuestos y lo no incluido. Eficiencias, retorno de condensado, purga, ciclos y perfil de carga son obligatorios: no hay valores típicos.
- `list_utility_price_references`: 10 referencias publicadas leídas el 2026-11-14 (EIA: EE. UU., Texas, Luisiana, agosto 2026; Eurostat: UE-27, DE, ES, FR, NL, 2026-S1; CFE GDMTH: Oriente y Golfo Norte, noviembre 2026). Solo se usan si se nombran; el resultado lo dice primero con fuente, periodo y antigüedad. Sin precios ni referencia la tool se niega.
- Validación: Tabla 1 del DOE Steam Tip Sheet #15 (15 valores) a ≤ 0.15 % y su ejemplo 9.398 contra 9.39 USD/1 000 lb (+0.08 %). Ulrich y Vasudevan 2006: 0.1061 / 0.04758 / 0.018586 / 4.022e-6 contra 0.106 / 0.048 / 0.019 / 4.0e-6 (redondeo). Balance de la isla de vapor con residuo 0; vapor ≥ combustible ÷ eficiencia en 8 de 8; refrigeración creciente al enfriar.
- Cadena en el caso del bloque 20 (recuperación de calor, DWSIM): con precios calculados el óptimo pasa de 650 m² (ilustrativos fijos) a 647 (usuario), 590 (EIA EE. UU.) y 504 m² (Texas).
- Evidencia: `notebooks/30_costo_de_servicios.py` y `.json`, `tests/test_utilities.py` (27 pruebas, sin red), filas M17 de `BENCHMARK.md`, MOCKS `PRECIOS-REFERENCIA-M17`, `SUPUESTOS-SERVICIOS-M17`, `ENTRADAS-ILUSTRATIVAS-M17`, `SERVICIOS-PENDIENTES-M17`. Suite completa 600 passed, 22 skipped (antes 573 y 22).

No cubierto en M17:
1. Gas natural de México: el IPGN de la CNE pide un reCAPTCHA; no se evadió. Con referencia mexicana solo hay electricidad; el combustible lo da el usuario.
2. Agua de enfriamiento y refrigeración de primeros principios sin ejemplo de libro reproducido (Turton cap. 8 y Seider cap. 17 no accesibles): tienen identidades de balance y el contraste de la ruta 2.
3. Capital y mantenimiento de los servicios, cogeneración y turbinas de contrapresión, agua de un solo paso, agua helada o salmuera en lazo, aceite térmico, aire comprimido, nitrógeno, tratamiento de aguas, costo de emisiones.
4. Tarifa de CFE: el cargo fijo no entra y distribución + capacidad se aplican a una sola demanda facturable; las fracciones por periodo horario las da el usuario (no se calculan desde el calendario de la tarifa).
5. El banco caduca: nadie lo refresca solo. Hoy avisa cuando una referencia pasa de un año.

**Bloque M16b (2026-11-14): `restarts = 1` por defecto en `optimize_design_bayesian`, decisión del owner, 71 tools (mismas tools), ✅.**
- Una corrida independiente comprueba el óptimo. Si coinciden: `verified` (baja el riesgo de un óptimo local, no demuestra el global). Si difieren más que la tolerancia: `runs_disagree`, con el mejor punto y la razón. Si no alcanzó el presupuesto: `not_verified`. `restarts=0` explícito queda como elección del usuario.
- Medido (20 semillas): en Hartmann-6 los óptimos equivocados que salen `verified` bajan de 12 a 4 de 20; aciertos 8 → 13 de 20. Branin y Gramacy 20/20. DWSIM: mismo óptimo con 33-34 evaluaciones (antes 17) contra 80-145 y 67; etapas enteras 50-66 solves contra 442.
- `suggest_experiments` no cambia (no tiene lazo de optimización). Suite completa 573 passed, 22 skipped (antes 569 y 22); las 3 lentas pasan con `ALCHEMAI_SLOW_BAYES=1`.

Abierto tras M16b:
1. El reinicio no elimina los verificados falsos: 4 de 20 en Hartmann-6 y 4 de 20 en la función de dos pozos, cuando las dos corridas caen en el mismo mínimo local. Opciones sin implementar: `restarts=2` por defecto (triple de evaluaciones) o un diseño inicial de la segunda corrida alejado de la primera.
2. 8 de los 13 aciertos de Hartmann-6 salen `runs_disagree`: correctos, pero no verificados.

**Bloque M16 (2026-11-14): optimización bayesiana y diseño bayesiano de experimentos, 71 tools, ✅ con límites declarados.** Fila «Optimización bayesiana, experimentos» del mapa: ⬜ → ✅.
- `optimize_design_bayesian` (flowsheet de DWSIM caro o que falla, 1-10 variables, enteros nativos) y `suggest_experiments` (datos de laboratorio o planta del usuario, sin simulador). Paquete `packages/bo/`; motor scikit-learn 1.9.1 (BSD-3) con adquisición propia. Summit no instala (pandas < 2); BoTorch se evaluó y se descartó (759 MB, sin mejora).
- Números: Branin 20/20, problema con restricciones 20/20, Hartmann-6 8/20 (13/20 con 2 reinicios). DWSIM: mismo óptimo que el directo con 17 evaluaciones contra 80-145 (Nelder-Mead) y 67 (sustituto); etapas enteras en 20-36 solves contra 442. SnAr: STY 19/20 contra 0/20 al azar; factor E 1/20 (10/20 con log).
- Evidencia: `notebooks/29_optimizacion_bayesiana.py` y `.json`, `tests/test_bayesian.py` (21 rápidas, 3 lentas con `ALCHEMAI_SLOW_BAYES=1`), filas M16 de `BENCHMARK.md`, MOCKS `SUPUESTOS-BAYESIANA`, `SUPUESTOS-EXPERIMENTOS`, `EMULADOR-SNAR`, `RESPUESTA-INVENTADA-PRUEBAS`. Suite completa 569 passed, 22 skipped (antes 548 y 19).

Pendiente (decisiones del owner marcadas):
1. ~~Owner: `restarts` por defecto.~~ **Decidido por el owner el 2026-11-14: 1 («Sí, un reinicio»); hecho en M16b.**
2. **Owner: umbrales de paro y tolerancia** (0.1 × tolerancia en 5 iteraciones, mínimo max(10, 5·n) guiadas, sin mejora en max(20, 10·n), tolerancia 0.5 %). Los puse yo y los ajusté con las tres funciones de prueba.
3. **Owner: `"log": true` automático o no.** Hoy lo decide el usuario y la tool solo avisa (factor E de SnAr: 1/20 sin log, 10/20 con log).
4. Multiobjetivo (frente de Pareto: rendimiento contra costo o factor E). Rutas: BoTorch qNEHVI en un entorno aparte (MIT, 759 MB) o ParEGO sobre el motor actual. No implementado.
5. Variables categóricas (disolvente, catalizador) en `suggest_experiments`. No implementado; la tool las rechaza.
6. Cotejar las constantes del emulador SnAr con el artículo de Hone 2017 (pubs.rsc.org devolvió 403); hoy vienen del código de Summit.
7. El factor E no llega al 1 % en 40 experimentos en la mitad de las campañas ni con log; con ruido baja a 3-4 de 20.
8. Sin caso real con evaluaciones de minutos: los flowsheets probados resuelven en 0.05-1 s, donde la bayesiana ahorra evaluaciones pero no tiempo.

Siguiente: **decisión del owner** sobre los pendientes 2-3, o el siguiente tema (multiobjetivo, layout, 3D, árbol de fallas, BST).

**Bloque M15c-2 (2026-11-13): valor por defecto condicional de `excess_enthalpy`, 69 tools (mismas tools), ✅.** Decisión del owner del 2026-11-13, opción C, transmitida por la sesión principal.
- `eo_column_optimize` y `simulate_column_nrtl`: `excess_enthalpy=None` (nuevo valor por defecto) enciende h^E solo si los parámetros de todos los pares vienen de un ajuste conjunto VLE + h^E de `vle_binary_regression` que pasó su veredicto. En cualquier otro caso queda apagado y `warnings[0]` dice qué pares no tienen respaldo calorimétrico y cómo encenderlo.
- `True` o `False` explícitos siempre mandan y salen etiquetados como elección del usuario (`properties.excess_enthalpy.decision`). Con `True` sobre parámetros solo-VLE se conserva la etiqueta «sin validación calorimétrica».
- La ruta ideal (M8b) no tiene término de exceso y no cambió.
- Referencias con parámetros solo-VLE recalculadas con re-solves (historia conservada en `BENCHMARK.md` y en los tests con `True` explícito):
  - Metanol-agua publicada (Simone 2015): Q_reb 1481.7 → 1491.6 kW (vs Aspen −1.41 % → −0.76 %), Q_cond 820.6 → 822.2 kW (+0.19 % → +0.39 %).
  - Metanol-agua óptimo: RR 4.4911 → 4.4938, Q_reb 1497.6 → 1507.8 kW; vs DWSIM «ExpData» Q_reb −0.49 %, Q_cond −0.72 %.
  - Etanol-agua cadena: RR 0.7008 → 0.7030, Q_reb 1030.0 → 1070.8 kW; vs DWSIM «ExpData» +0.20 %, +0.09 %.
  - Arranque M15b (x_D 0.70 con 15 platos, 0.75, 0.80, 0.82): Q_reb +3.0 a +4.2 %; vs DWSIM «ExpData» −0.51 a −0.70 %.
- Con los datos reales de hoy el modo automático nunca enciende el término: los cuatro ajustes conjuntos de etanol-agua fallan el veredicto (el de 363 K, por 0.04 J/mol). La rama «encendido» está probada con el ajuste sintético.
- De paso: un resultado de regresión sin cita rompía `resolve_parameters` (fallo de M15); corregido.
- Tests: 4 rápidos nuevos; los lentos llevan cada caso con `True` (historia M15/M15b) y con el valor por defecto (M15c-2): 51 de 51 con `ALCHEMAI_SLOW_EO_NRTL=1`. Suite completa 548 passed, 19 skipped.

Pendiente (decisiones del owner marcadas):
1. **Owner: tolerancias del veredicto** (`JOINT_TOLERANCES`). Ahora deciden el valor por defecto de la columna; siguen como las puse, sin decidir.
2. **Owner (propuesta, no implementada): ¿un par del usuario puede contar como validado?** Hoy `hE_source` (cita) es solo una etiqueta y no enciende el término. Opción: aceptar del usuario la evidencia mínima (cita + AAD de h^E + intervalo de T) para encenderlo por defecto.
3. Opción D de M15c (h^E medido como correlación propia en la entalpía líquida): no forma parte de este bloque.

**Bloque M15c (2026-11-13): h^E de etanol-agua contra calorimetría y ajuste conjunto VLE + h^E, 69 tools (mismas tools), ✅ con un resultado negativo declarado.**
- Datos reales: 83 puntos de h^E en 5 isotermas (298.15-383.15 K), de los archivos ThermoML del NIST (Nagamachi y Francesconi 2006; Brandt et al. 2014). Guardados con cita en `tests/data/he_ethanol_water_thermoml.json`.
- La hipótesis se confirma: el h^E medido es negativo por debajo de ~363 K y el de parámetros solo-VLE es positivo siempre. AAD global 443 J/mol con el término (regresión de Kamihama), 463 (ChemSep), 232 sin término.
- Columna de referencia de M15b: Q_reb 1030.0 kW con el término, 1070.8 sin él, ≈ 1084 con el h^E medido (estimación por balance de entalpía). Encendido −5.0 %, apagado −1.2 %.
- `vle_binary_regression` ajusta VLE + h^E a la vez (pesos y AAD separados, IC 95 %, veredicto). Sintético recuperado 12/12. Etanol-agua real: un juego NRTL (a + b/T) no representa ambos, y la tool lo dice.
- Los juegos conjuntos corren en `eo_column_optimize` de punta a punta (6 re-solves con sello; DWSIM «ExpData» con los mismos a + b/T: cargas ≤ 0.22 %). No mejoran la carga: el que conserva el VLE (solo 363 K) queda igual que apagado (−1.2 %).
- La salida de la columna dice de dónde viene h^E y cuánto pesa en Q_reb. En M15c el valor por defecto de `excess_enthalpy` no se tocó (cambió en M15c-2).
- Tests: 16 rápidos + 1 lento nuevos (`tests/test_excess_enthalpy.py`); suite completa 544 passed, 13 skipped.

Pendiente (decisiones del owner marcadas):
1. ~~**Owner: valor por defecto de `excess_enthalpy`.**~~ → **decidido por el owner el 2026-11-13: opción C (condicional)**, implementada en M15c-2 (arriba). Evidencia: un solo sistema (etanol-agua), que es un caso difícil conocido.
   - A. Dejarlo encendido (hoy). Es consistente con Gibbs-Helmholtz y es lo que hace Aspen con sus parámetros; con parámetros solo-VLE puede dar el signo contrario (etanol-agua: Q_reb −5.0 %). Ya sale etiquetado.
   - B. Apagarlo por defecto. Etanol-agua queda en −1.2 %. Cambia los números de referencia de M15/M15b (metanol-agua 1497.6 → 1507.8 kW).
   - C. Condicional: encendido solo si todos los pares vienen de un ajuste con h^E que pase el veredicto; si no, apagado con aviso. En etanol-agua equivale a B. También cambia las referencias con parámetros solo-VLE.
   - D. (Capacidad nueva, otro bloque) llevar el h^E medido a la entalpía líquida como correlación propia de los datos (Redlich-Kister en x, polinomio en T), con γ de los parámetros de VLE. Es la única vía que mete el h^E real en la columna; rompe la consistencia formal entre γ y h^E.
   - Recomendación: C como política y D para etanol-agua. D sigue sin decidir.
2. **Owner: tolerancias del veredicto** (`JOINT_TOLERANCES`: AAD_T 0.5 K, AAD_y 0.01, AAD_hE 10 % del máximo) y σ de h^E por defecto (5 J/mol + 1 %). Son elecciones mías, en `MOCKS.md`.
3. Las cargas «con h^E medido» son una estimación (flujos y reflujo fijos): reproduce el par resuelto con/sin a 1.2 kW en la referencia y a 3-6 kW en los juegos conjuntos. Un re-solve exacto pide la opción D.
4. No hay isoterma medida cerca de 350 K: la alimentación se interpola entre 333.15 y 363.15 K (±9 J/mol según el orden de la interpolación). Larkin 1975 (298-383 K), Ott 1986-87, Costigan 1980, Lama y Lu 1965 y Boyne y Williamson 1967 no se buscaron uno por uno: ThermoML bastó.
5. Dependencia en T más rica (e·ln T, f·T) en la tool y en la columna: el ensayo fuera de la tool baja el AAD de h^E de 107 a 49 J/mol y tampoco cumple las tolerancias (`notebooks/27b`).
6. UNIQUAC y Wilson también ajustan h^E en la tool, pero solo NRTL está probado con datos reales y solo NRTL entra a la columna.
7. Siguen abiertos de M15b: re-medir el término de exceso cuando salga DWSIM 10.2.17 (issue #137 corregido en su `master` el 2026-11-14, commit `901b173`); x_D +1.1 % vs DWSIM (Psat); `simulate_column_nrtl` frágil en etanol-agua; destilado más allá de x 0.82; ε en la ruta ideal (**owner**).

Siguiente: **decisión del owner** sobre el pendiente 1 (y, si elige D, ese bloque), o pasar a otro tema (layout, 3D, árbol de fallas, BST).

**Bloque M15b (2026-11-13): etanol-agua cerrado en `eo_column_optimize` con NRTL, 69 tools (misma tool), ✅.**
- El Q_reb +6.5 % vs DWSIM no era nuestro: DWSIM 10.2.12 suma −R·T²·Σx·∂γ/∂T como «exceso» (debería ser ∂ln γ/∂T). Cierre término a término: 54.55 de 54.58 kW.
- Igual con igual (`excess_enthalpy=False` vs DWSIM «ExpData»): etanol-agua Q_reb +0.20 %, Q_cond +0.09 %; metanol-agua −0.49 %, −0.72 %.
- Arranque: continuación de las dos specs sobre el sistema cuadrado + homotopía de no idealidad + ε hasta que selle. Verificados 15 platos x_D 0.70 y 20 platos x_D 0.75/0.80/0.82 (34-87 s).
- McCabe-Thiele propio (`packages/eo/shortcut.py`) rechaza antes del NLP: sin etapas ni a reflujo total, reflujo máximo insuficiente, o producto a < 0.03 K del azeótropo.
- Metanol-agua de M15 sin cambios (RR 4.4911, Q_reb 1497.6 kW).

Pendiente (decisiones del owner marcadas):
1. ~~**Owner**: h^E de etanol-agua pesa −3.8 % en Q_reb y viene de parámetros de VLE. ¿Se deja por defecto o se busca h^E calorimétrico publicado?~~ → contrastado en M15c; queda la decisión del valor por defecto (pendiente 1 de M15c).
2. ~~¿Reportar a DWSIM la fórmula del término de exceso?~~ → reportado el 2026-11-13 con OK del owner: https://github.com/DanWBR/dwsim10/issues/137 (fórmula confirmada en su fuente, `master` 9ab62b6; medido solo NRTL en 10.2.12). **Corregido por DWSIM el 2026-11-14:** el mantenedor confirmó el error y cerró el issue como completado; commit `901b173` de `DanWBR/dwsim10` (`master`) cambia los seis modelos (NRTL, UNIQUAC, UNIFAC, MODFAC, NIST-MFAC, Wilson) a d ln γ/dT, leído en su fuente (`NRTL.vb` usa `Math.Log(gamma2) - Math.Log(gamma1)`). Su respuesta dice que DWSIM da ahora 362.7 J/mol a 298.15 K y x1 = 0.10. Aún sin liberar (última versión publicada: 10.2.16; saldrá en 10.2.17). Falta: cuando salga 10.2.17, re-medir con `hex_repro` y volver a comparar la columna con h^E contra «LiqCp_Excess» (no medido por nosotros todavía).
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
**Tras M19 (2026-11-14):** el siguiente paso del producto no es código: es probar el diagnóstico de vapor con los datos de una planta real (a quién y cuándo lo decide el owner; nadie ha sido contactado). Lo que se le pide está en `docs/DATOS_PARA_DIAGNOSTICO.md`; lo que se le entrega es el Markdown de `steam_diagnostic_report` (ejemplo ilustrativo en `notebooks/32_diagnostico_de_vapor_reporte.md`). Sin planta, lo que sí puede avanzar son las decisiones 3 a 8 de M19 (abajo).

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
- **Delegadas el 2026-11-14** («sé autónomo y toma las decisiones»): composición del gas desconocida, objetivos publicados, umbrales de revisión de datos, constantes de fugas y aislamiento, formato de captura (Excel) y de entrega (HTML y PDF). Las tomé y están en MOCKS con su fuente o marcadas como elección mía; cualquiera se puede revertir. Siguen siendo del owner: nombre y precio de la oferta, cuántas plantas piloto y cómo lo contactan, publicar, y todo lo que cueste dinero.
- **M19, diagnóstico de vapor** (2026-11-14):
  1. Nombre del diagnóstico y precio después del piloto. No están definidos y no se propusieron. **Decidido (owner, 2026-11-14): el piloto va sin costo**, a cambio de los datos y de poder contar el caso sin el nombre de la empresa.
  2. Con qué planta se prueba primero y quién la contacta. Nadie fue contactado. El owner no tiene planta a la mano (2026-11-14); borrador de convocatoria en `docs/posts/03_convocatoria_plantas_piloto.md` (no publicado), al que le faltan dos decisiones: cuántas plantas y cómo lo contactan. Rutas sin intentar: compañeros de generación en plantas, proveedores de calderas y de tratamiento de agua, cámaras y clústeres industriales.
  3. Pérdida por paredes: **decidido (owner, 2026-11-14): «Usar valor publicado»**. Hecho en M19b. Queda abierto: el valor es el tope de la fuente (1.0 %), elección mía; ¿debe depender del tipo de caldera (tubos de humo pierden menos), del tamaño (curva ABMA, no leída) y de la carga (a media carga la fuente implica hasta 2 %)? La decisión cubre solo este dato del diagnóstico de vapor; los demás valores típicos (M18 punto 1, M17 punto 1) siguen sin decidir.
  4. Criterio de aceptación del diagnóstico: ¿bastan los ejemplos del DOE (tres a ≤ 0.04 %, #4 a −3.7 %, #3 no reproducible) o hace falta un caso de planta con medición antes y después?
  5. Pérdida por paredes entre casos: se mantiene como fracción del poder calorífico (hoy) o constante en kW.
  6. Cuando se dan producción de vapor y consumo de combustible: hoy el usuario elige la base anual (`annual_basis`). ¿Fijar una regla, y a partir de qué diferencia entre las dos eficiencias se avisa o se niega? Hoy solo se reporta la diferencia.
  7. Margen sobre el punto de rocío del agua: hoy solo se rechaza en o bajo el punto de rocío y se informa el margen. ¿Fijar un margen mínimo, o modelar el rocío ácido?
  8. Precio de los ahorros con recibos: hoy precio medio (importe ÷ energía). ¿Pedir el cargo variable por separado?
  9. Alcance siguiente: carga variable (varios puntos de operación), trampas y fugas, aislamiento, emisiones de CO2, varias calderas, precalentador de aire con su propia temperatura de aire.
  10. Formato de entrega del reporte: Markdown (hoy), PDF, documento con marca. ¿Captura de datos en una hoja o formulario para la planta?
  11. ¿Refrescar MEASUR como contraste permanente (instalarlo y comparar caso por caso)?
  12. Qué hacer con la hoja #3 del DOE: se deja documentada como no reproducible o se busca una tercera fuente (ASME PTC 4, un manual de fabricante).
- **M18, hornos y calderas** (2026-11-14):
  1. ¿Ofrecer valores típicos etiquetados como supuesto (pérdida por paredes, emisividad de tubo, flujo medio admisible por servicio, convección en el hogar, exceso de aire por tipo de quemador)? Hoy se exigen todos.
  2. Emisividad del gas: se queda la WSGG de Smith 1982 o se busca otra (Leckner, Bordbar 2014, Alberti 2018, cartas de Hottel de una fuente legible). Hoy difiere 20 % de las cartas de Kern a trayectorias largas.
  3. Criterio de aceptación de la eficiencia: el resultado queda entre dos tablas del DOE que difieren ≈ 0.8 puntos entre sí. ¿Basta ±0.6 puntos o hace falta un caso con composición declarada (ASME PTC 4, API 560 anexo G)?
  4. Aire seco por defecto: `air.yaml` de Cantera (21/78/1) o una composición estándar citada (20.95 % O2).
  5. Temperatura de referencia: 25 °C (datos NASA) fija, o también 15 °C / 60 °F como en API 560.
  6. ¿Conectar la eficiencia calculada a `utility_costs` de forma automática? Hoy es un dato que el usuario pasa.
  7. Convección: ¿buscar un ejemplo publicado y añadir tubos aletados (ESCOA, `ht.air_cooler`)?
  8. Alcance siguiente: flujo pico y temperatura máxima de metal (API 530), modelo de zonas, rocío ácido, calderas de condensación, sólidos con inquemados, tiro.
  9. ¿Ampliar `adiabatic_flame` (aire bajo 300 K, especies fuera de GRI-Mech 3.0)? Hoy esos casos dan solo la temperatura de combustión completa.
- **M17, costo de servicios** (2026-11-14):
  1. ¿Ofrecer valores típicos etiquetados como supuesto (eficiencia de caldera, purga, ciclos de concentración, η de segunda ley) cuando el usuario no los da? Hoy se exigen.
  2. Refresco del banco: cada cuánto, quién, y si pasado cierto tiempo la tool debe negarse en vez de avisar (hoy avisa a partir de 365 días).
  3. Gas natural de México: descargar el IPGN a mano (CAPTCHA) o autorizar otra fuente.
  4. Cobertura del banco: bandas de Eurostat (hoy ID y I4), otras tarifas de CFE (DIST, DIT) y otras divisiones, más estados de EE. UU.
  5. Cargo por demanda de CFE: ¿implementar las reglas 7.1 y 7.2 exactas (demandas distintas para capacidad y distribución) y el cargo fijo?
  6. ¿Conectar `utility_costs` a `flowsheet_cost`, pinch y los optimizadores (hoy solo entrega el dict `prices`), y cobrar niveles distintos por equipo?
  7. ¿Cogeneración y costo del vapor por nivel con turbinas? ¿Capital de los servicios?
  8. La ruta de Ulrich y Vasudevan (precio con capital, CEPCI del usuario): ¿se queda como contraste?
- Publicar el post de lanzamiento en LinkedIn (lo hace el owner; repo ya público).
- App de Mac: plantilla de Figma (bloqueante), distribución fuera de la App Store (recomendada), cobro y precio.
- Fluxo 1.1/1.2: cuándo tocarlas (1.0.0 en App Review desde 2026-10-19).
