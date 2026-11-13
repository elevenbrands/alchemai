"""Generic LOPA data: initiating-event frequencies, IPL PFDs, SIL bands and EXAMPLE risk criteria, each with its source.

These are GENERIC literature values (order of magnitude), not site data. A site LOPA must use the site's or the
company's own data (and its own risk criterion). Every value here was read on 2026-11-13 from a legible source;
where the primary source (CCPS 2001, CCPS 2015) was not available, the value comes from a secondary source that
reproduces it and the source string says so. Nothing here was filled in from memory.
"""
from __future__ import annotations

# ---------------------------------------------------------------- sources
SAFECHE = ("SAFEChE (U. de Michigan), tutorial «Layers of Protection (LOPA)», Apéndice A (FOIE) y Apéndice B (PFD), "
           "que cita a Crowl y Louvar, Chemical Process Safety (Pearson, 2019); valores de origen CCPS 2001. "
           "safeche.engin.umich.edu/tutorials/lopa-tutorial (leído 2026-11-13)")
KENEXIS = ("Kenexis, «Layer of Protection Analysis (LOPA) — Participant Training», presentación P2SAC (Purdue), "
           "engineering.purdue.edu/P2SAC/presentations/documents/12.23.lopa: tablas «Initiating Events – Typical» y "
           "«Credit for Layers of Protection», reglas «Typical IPL Usage Rules»")
CASEY = ("Casey (2019), «Limitations and misuse of LOPA», Loss Prevention Bulletin 265, IChemE, pp. 13-15: Tabla 1 "
         "(frecuencias del libro AIChE/CCPS 2001) y ejemplos de modificadores condicionales mal usados")
BSTG = ("HSE / Buncefield Standards Task Group (2007), «Safety and environmental standards for fuel storage sites», "
        "Final report, Apéndice 1 «Example LOPA assessment for an overfill scenario», pp. 76-92")
BSTG_HUMAN = BSTG + ", ¶54: cita a CCPS 2001 («0.1 con 10 minutos de respuesta, 0.01 con 40 minutos»)"
IEC61511 = ("IEC 61511-1:2016 (= ISA 84.00.01-2004), Tabla 4 (modo de demanda baja: PFDavg) y Tabla 5 (demanda alta o "
            "continua: frecuencia de falla peligrosa por hora); demanda baja = no más de una demanda por año")
CCPS2001 = ("CCPS (2001), Layer of Protection Analysis: Simplified Process Risk Assessment, AIChE: fuente primaria de las "
            "tablas; sin copia legible en esta sesión, sus valores se toman de las fuentes secundarias citadas")
CCPS2015 = ("CCPS (2015), Guidelines for Initiating Events and Independent Protection Layers in LOPA, AIChE: sin copia "
            "legible en esta sesión; ningún valor de esta tabla sale de ahí")

PER_YEAR, PER_OPPORTUNITY = "por año", "por oportunidad"

# ---------------------------------------------------------------- initiating events
# key -> (description, value, unit, source)
INITIATING_EVENTS: dict[str, tuple[str, float, str, str]] = {
    "bpcs_loop": ("Falla de un lazo de instrumentos del BPCS (sensor, controlador o elemento final)", 0.1, PER_YEAR,
                  SAFECHE + "; " + KENEXIS + " (Basic Process Control Loop Failure 1/10 años)"),
    "pump_seal": ("Falla del sello de una bomba", 0.1, PER_YEAR, SAFECHE),
    "pump_failure": ("Falla o paro de una bomba", 0.1, PER_YEAR, KENEXIS + " (Pump Failure 1/10 años)"),
    "compressor_failure": ("Falla o paro de un compresor", 0.1, PER_YEAR, KENEXIS + " (Compressor Failure 1/10 años)"),
    "cooling_water": ("Falla del agua de enfriamiento", 0.1, PER_YEAR, SAFECHE + "; " + CASEY + " (Cooling failure 1e-1/año)"),
    "external_fire": ("Fuego externo", 0.1, PER_YEAR,
                      SAFECHE + " (External fire 1e-1/año; Casey 2019 da 1e-2/año para fuego externo GRANDE: se usa el "
                      "valor conservador)"),
    "operator_error": ("Error del operador al ejecutar una tarea rutinaria", 0.01, PER_OPPORTUNITY,
                       SAFECHE + " (Operator failure 1e-2/oportunidad; FOIE = oportunidades/año × HEP)"),
    "loto": ("Falla del procedimiento de bloqueo y etiquetado (LOTO)", 1e-3, PER_OPPORTUNITY, CASEY),
    "safety_valve_spurious": ("Apertura inesperada de una válvula de seguridad", 0.01, PER_YEAR, SAFECHE),
    "third_party": ("Intervención de terceros (impacto de vehículo, etc.)", 0.01, PER_YEAR, SAFECHE + "; " + CASEY),
    "atmospheric_tank": ("Falla de un tanque atmosférico", 1e-3, PER_YEAR, SAFECHE),
    "pressure_vessel": ("Falla residual de un recipiente a presión", 1e-6, PER_YEAR, SAFECHE),
    "piping_leak_100m": ("Fuga de tubería (10 % de la sección) por cada 100 m", 1e-3, PER_YEAR, SAFECHE),
    "piping_rupture_100m": ("Ruptura total de tubería por cada 100 m", 1e-5, PER_YEAR, CASEY),
}

# ---------------------------------------------------------------- independent protection layers
# key -> (description, PFD or None = no default credit, condition, source)
IPLS: dict[str, tuple[str, float | None, str, str]] = {
    "psv": ("Válvula de alivio de presión (resorte, servicio limpio)", 0.01,
            "dimensionada para ESTE escenario (relief_valve_sizing) y con programa de prueba e inspección",
            KENEXIS + "; " + SAFECHE),
    "rupture_disc": ("Disco de ruptura (servicio limpio)", 0.01, "dimensionado para este escenario", KENEXIS + "; " + SAFECHE),
    "bpcs_loop": ("Lazo del BPCS independiente del evento iniciador", 0.1,
                  "no comparte sensor, controlador ni elemento final con el evento iniciador; RRF ≤ 10",
                  KENEXIS + "; " + SAFECHE),
    "alarm_operator": ("Alarma + respuesta del operador", 0.1,
                       "alarma anunciada, puesto atendido, procedimiento escrito y al menos 10 min entre la alarma y la "
                       "consecuencia", BSTG_HUMAN + "; " + KENEXIS + " (Operator Response to Alarm 0.1)"),
    "check_valve_dual": ("Dos válvulas check en serie (servicio limpio)", 0.1, "servicio limpio, probadas",
                         KENEXIS + " (Check Valves (dual, clean service) 0.1); SAFEChE da 1e-2: se usa el conservador"),
    "check_valve_single": ("Una válvula check", None,
                           "sin crédito por defecto: las fuentes legibles discrepan (SAFEChE/Crowl-Louvar 1e-2 para una "
                           "sola; Kenexis 0.1 solo para dos en serie). Acreditarla exige la PFD y la justificación del "
                           "usuario", SAFECHE + "; " + KENEXIS),
    "sif_sil1": ("Función instrumentada de seguridad SIL 1", 0.1, "verificada (PFDavg ≤ 0.1)", KENEXIS + "; " + IEC61511),
    "sif_sil2": ("Función instrumentada de seguridad SIL 2", 0.01, "verificada (PFDavg ≤ 0.01)", KENEXIS + "; " + IEC61511),
    "sif_sil3": ("Función instrumentada de seguridad SIL 3", 0.001, "verificada (PFDavg ≤ 0.001)", KENEXIS + "; " + IEC61511),
    "dike": ("Dique o muro de contención", 0.01, "para el escenario de derrame que contiene", SAFECHE),
    "flame_arrestor": ("Arrestador de flama o detonación", 0.01, "diseñado, instalado y mantenido", SAFECHE),
}

ALARM_MIN_RESPONSE_MIN = 10.0     # minutes between alarm and consequence for the 0.1 credit (CCPS 2001 via BSTG ¶54)

# ---------------------------------------------------------------- SIL bands (IEC 61511-1 Tables 4 and 5)
# (SIL, lower bound inclusive, upper bound exclusive)
SIL_PFD_BANDS = [(4, 1e-5, 1e-4), (3, 1e-4, 1e-3), (2, 1e-3, 1e-2), (1, 1e-2, 1e-1)]
SIL_PFH_BANDS = [(4, 1e-9, 1e-8), (3, 1e-8, 1e-7), (2, 1e-7, 1e-6), (1, 1e-6, 1e-5)]
LOW_DEMAND_MAX_PER_YEAR = 1.0
HOURS_PER_YEAR = 8760.0

# ---------------------------------------------------------------- EXAMPLE risk criteria (never applied silently)
EXAMPLE_CRITERIA: dict[str, dict] = {
    "bstg_2007": {"frecuencia_tolerable_por_año": 1e-6,
                  "descripción": "EJEMPLO, no criterio del owner: 1e-6/año para 2-10 fatalidades, frontera de la región "
                                 "«ampliamente aceptable» del ejemplo HSE/BSTG (HSE R2P2). El propio documento advierte "
                                 "que no hay criterios reconocidos por escenario único.",
                  "fuente": BSTG + ", ¶36 y ¶43-45"},
    "kenexis_tmel_1e-4": {"frecuencia_tolerable_por_año": 1e-4,
                          "descripción": "EJEMPLO, no criterio del owner: TMEL 1e-4/año del ejemplo explícito de Kenexis",
                          "fuente": KENEXIS},
    "kenexis_tmel_1e-5": {"frecuencia_tolerable_por_año": 1e-5,
                          "descripción": "EJEMPLO, no criterio del owner: TMEL 1e-5/año del ejemplo de la columna de "
                                         "destilación de Kenexis",
                          "fuente": KENEXIS},
}
CCPS_CRITERIA_NOTE = ("Los criterios de ejemplo de CCPS 2001 (cap. 7) no se ofrecen: no hubo copia legible. Los de esta "
                      "tabla son ejemplos publicados por HSE/BSTG y Kenexis, marcados como EJEMPLO.")

INDEPENDENCE_RULES: dict[str, tuple[str, str]] = {
    "I-1": ("Una capa que comparte un componente (sensor, lazo, controlador o elemento final) con la causa no recibe "
            "crédito", CCPS2001 + "; " + KENEXIS + " («If a BPCS control loop failure was the initiating event, don't "
                                                   "use equipment from a failed BPCS loop to justify IPL credit»)"),
    "I-2": ("Si la causa es una falla del BPCS, se acredita a lo más una capa que dependa del BPCS (lazo o alarma), y "
            "solo si no comparte componentes con el lazo que falló; el controlador del BPCS se supone común",
            KENEXIS + "; regla conservadora alineada con IEC 61511-1 (límites del BPCS: RRF ≤ 10 por capa)"),
    "I-3": ("Al operador se le acredita una sola vez por evento iniciador", KENEXIS + " («Don't take credit for the "
                                                                              "operator more than once»)"),
    "I-4": ("Una misma SIF no cuenta como dos capas", KENEXIS + " («Don't identify the SIS for more than one IPL»)"),
    "I-5": (f"Alarma + operador solo con al menos {ALARM_MIN_RESPONSE_MIN:.0f} min entre la alarma y la consecuencia "
            "(dato del usuario; sin él no hay crédito)", BSTG_HUMAN),
    "I-6": ("Especificidad: la capa debe estar trazada a la fila del HAZOP del escenario (P&ID M10) o declarada por el "
            "usuario para ese escenario", CCPS2001 + "; " + KENEXIS + " (Specificity)"),
}
