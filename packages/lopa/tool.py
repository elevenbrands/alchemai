"""Agent / MCP tool `lopa_from_hazop` (bloque M11): LOPA worksheet and SIL of each trip, from the HAZOP and the P&ID."""
from __future__ import annotations

from pathlib import Path

from packages.lopa import data as D
from packages.lopa.engine import evaluate
from packages.lopa.from_pid import DEFAULT_MIN_SEVERITY, build_scenarios, tags_consistent
from packages.pid.rules import build_pid

ASSUMPTIONS = [
    "Frecuencias de eventos iniciadores y PFD de capas: valores GENÉRICOS de literatura (orden de magnitud), no datos "
    "del sitio; cada uno con su fuente en `tablas`. CCPS 2001 y 2015 no se consultaron directamente (sin copia "
    "legible): sus valores vienen de fuentes secundarias que los reproducen.",
    "La frecuencia tolerable es el criterio de riesgo del owner o de la empresa: sin ella no hay brecha ni SIL. Los "
    "criterios de `criterios_de_ejemplo` son EJEMPLOS publicados y solo se aplican si se piden por nombre.",
    "Modificadores condicionales (ignición, ocupación, fatalidad, detección) solo del usuario; sin ellos = 1 "
    "(conservador). Los de «tiempo en riesgo» no valen para fallas latentes (Casey 2019).",
    "Un escenario = una fila del HAZOP; cada causa es un evento iniciador. Dos causas que son la misma falla del mismo "
    "lazo se fusionan (no se cuenta dos veces).",
    "Causas fuera de la tabla genérica (taponamiento, runaway, contaminación, blowby…) quedan con frecuencia «por "
    "definir»: el escenario se marca incompleto y su brecha es una cota inferior.",
    "Error del operador: 1e-2 por oportunidad; las oportunidades por año son dato del usuario (sin ellas no hay "
    "frecuencia).",
    "Capas acreditables: solo los elementos del P&ID (M10) trazados a la fila (PSV 0.01, alarma + operador 0.1 con ≥ 10 "
    "min de respuesta) y las que declare el usuario. Una check sola no recibe crédito por defecto.",
    "Las alarmas del P&ID usan el transmisor del lazo del BPCS: no reciben crédito si la causa es la falla de ese lazo "
    "(I-1). El controlador del BPCS se supone común a todos los lazos (I-2).",
    "Los disparos del P&ID son las SIF por dimensionar: se supone sensor (5xx) y elemento final propios; si el elemento "
    "final es la válvula del BPCS, declárala en `componentes` y la independencia se pierde.",
    "Bandas SIL de IEC 61511-1 (Tablas 4 y 5) citadas sin copia del estándar a la mano; los límites del BPCS (RRF ≤ 10) "
    "se aplican como regla conservadora.",
    "La PSV se acredita con 0.01 solo si está dimensionada para el escenario (relief_valve_sizing); no se verifica aquí.",
]


def _ie_row(e) -> dict:
    t = D.INITIATING_EVENTS.get(e.kind)
    return {"causa": e.cause, "tipo": t[0] if t else ("dato del usuario" if e.kind == "usuario" else "fuera de la tabla"),
            "frecuencia_por_año": e.frequency, "fuente": e.source, "tags": e.tags, "nota": e.note}


def worksheet(scenarios, allow_multiple_operator: bool = False) -> tuple[list[dict], list[dict]]:
    rows, sifs = [], {}
    for sc in scenarios:
        ev = evaluate(sc, allow_multiple_operator)
        m = ev["mitigada"]
        capas = []
        for l in sc.layers:
            per_ie = [next(c for c in x["capas"] if c["tag"] == l.tag) for x in m["eventos"]] if m["eventos"] else []
            capas.append({"tag": l.tag, "tipo": l.description, "pfd": l.pfd, "fuente": l.source, "nota": l.note,
                          "crédito_por_causa": [{"causa": x["causa"], "acreditada": c["acreditada"], "razón": c["razón"]}
                                                for x, c in zip(m["eventos"], per_ie)]})
        row = {"escenario": sc.ref, "nodo": sc.node, "desviación": sc.deviation, "consecuencia": sc.consequence,
               "severidad_hazop": sc.severity, "eventos_iniciadores": [_ie_row(e) for e in sc.events],
               "frecuencias_intermedias": [{"causa": x["causa"], "f_IE": x["frecuencia_por_año"], "pfd_capas": x["pfd_capas"],
                                            "f_intermedia": x["frecuencia_intermedia"]} for x in m["eventos"]],
               "capas": capas,
               "modificadores": [{"nombre": x.name, "probabilidad": x.probability, "fuente": x.source} for x in sc.modifiers],
               "producto_modificadores": m["producto_modificadores"], "frecuencia_mitigada_por_año": m["frecuencia_mitigada"],
               "frecuencia_tolerable_por_año": sc.tolerable, "origen_tolerable": sc.tolerable_source or "por definir",
               "brecha_mitigada_sobre_tolerable": ev["brecha"], "estado": ev["estado"], "sif": ev["sif"],
               "salvaguardas_no_acreditadas": sc.not_credited, "avisos": ev["avisos"]}
        rows.append(row)
        for s in ev["sif"]:
            if "pfd_requerida" not in s:
                sifs.setdefault(s["sif"], {"sif": s["sif"], "escenarios": [], "pfd_requerida": None, "sil": None,
                                           "estado": s["estado"]})["escenarios"].append(sc.ref)
                continue
            cur = sifs.setdefault(s["sif"], {"sif": s["sif"], "escenarios": [], "pfd_requerida": None, "sil": None})
            cur["escenarios"].append(sc.ref)
            if cur["pfd_requerida"] is None or s["pfd_requerida"] < cur["pfd_requerida"]:
                cur.update(pfd_requerida=s["pfd_requerida"], rrf_requerido=s["rrf_requerido"], sil=s["sil"],
                           sil_texto=s["sil_texto"], modo=s["modo"], estado=s["estado"], escenario_que_manda=sc.ref,
                           alcanzable=s["alcanzable"])
    return rows, list(sifs.values())


def markdown(rows: list[dict], sifs: list[dict]) -> str:
    f = lambda x: "—" if x is None else f"{x:.2e}"
    out = ["| Escenario | Causas (f/año) | Capas acreditadas | ΠCM | f mitigada | f tolerable | Estado |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        causes = "; ".join(f"{e['causa'][:40]} ({f(e['frecuencia_por_año'])})" for e in r["eventos_iniciadores"])
        cred = ", ".join(sorted({c["tag"] for c in r["capas"] if any(x["acreditada"] for x in c["crédito_por_causa"])})) or "—"
        out.append(f"| {r['escenario']} | {causes} | {cred} | {r['producto_modificadores']:g} | "
                   f"{f(r['frecuencia_mitigada_por_año'])} | {f(r['frecuencia_tolerable_por_año'])} | {r['estado']} |")
    out += ["", "| SIF | PFD requerida | RRF | SIL | Modo | Escenario que manda |", "|---|---|---|---|---|---|"]
    for s in sifs:
        rrf = s.get("rrf_requerido")
        out.append(f"| {s['sif']} | {f(s.get('pfd_requerida'))} | {'—' if rrf is None else f'{rrf:.1f}'} | "
                   f"{s.get('sil_texto', s.get('estado'))} | {s.get('modo', '—')} | {s.get('escenario_que_manda', '—')} |")
    return "\n".join(out)


def lopa_from_graph(g, tolerable_frequency: float | None = None, tolerable_by_severity: dict | None = None,
                    example_criterion: str | None = None, alarm_response_min: float | None = None,
                    conditional_modifiers: list[dict] | None = None, overrides: dict | None = None,
                    min_severity: int = DEFAULT_MIN_SEVERITY, allow_multiple_operator: bool = False) -> dict:
    spec = build_pid(g)
    tsrc = "usuario" if tolerable_frequency is not None else ""
    if tolerable_frequency is None and example_criterion:
        if example_criterion not in D.EXAMPLE_CRITERIA:
            raise ValueError(f"criterio de ejemplo desconocido: {example_criterion}; opciones {list(D.EXAMPLE_CRITERIA)}")
        ex = D.EXAMPLE_CRITERIA[example_criterion]
        tolerable_frequency, tsrc = ex["frecuencia_tolerable_por_año"], f"EJEMPLO «{example_criterion}»: {ex['fuente']}"
    scs = build_scenarios(spec, min_severity, tolerable_frequency, tsrc, tolerable_by_severity, alarm_response_min,
                          conditional_modifiers, overrides)
    rows, sifs = worksheet(scs, allow_multiple_operator)
    has_criterion = tolerable_frequency is not None or bool(tolerable_by_severity) or any(
        "frecuencia_tolerable" in v for v in (overrides or {}).values())
    out = {"name": spec.name, "hoja_lopa": rows, "sif": sifs, "trazabilidad_tags": tags_consistent(scs, spec),
           "conteo": {"filas_hazop": len(spec.hazop_rows), "escenarios": len(rows),
                      "por_estado": {k: sum(r["estado"] == k for r in rows) for k in sorted({r["estado"] for r in rows})},
                      "causas_sin_frecuencia": sum(e["frecuencia_por_año"] is None for r in rows for e in r["eventos_iniciadores"])},
           "assumptions": ASSUMPTIONS,
           "tablas": {"eventos_iniciadores": {k: {"descripción": v[0], "valor": v[1], "unidad": v[2], "fuente": v[3]}
                                              for k, v in D.INITIATING_EVENTS.items()},
                      "capas": {k: {"descripción": v[0], "pfd": v[1], "condición": v[2], "fuente": v[3]} for k, v in D.IPLS.items()},
                      "reglas_de_independencia": {k: {"regla": v[0], "fuente": v[1]} for k, v in D.INDEPENDENCE_RULES.items()},
                      "bandas_sil_pfd": [{"sil": s, "pfd_desde": lo, "pfd_hasta_excl": hi} for s, lo, hi in D.SIL_PFD_BANDS],
                      "bandas_sil_pfh": [{"sil": s, "pfh_desde": lo, "pfh_hasta_excl": hi} for s, lo, hi in D.SIL_PFH_BANDS],
                      "fuente_sil": D.IEC61511},
           "markdown": markdown(rows, sifs)}
    if not has_criterion:
        out["falta_criterio"] = ("Sin frecuencia tolerable: la brecha y el SIL quedan por definir. Es el criterio de riesgo "
                                 "del owner o de la empresa (por severidad o por escenario). Ejemplos publicados, solo como "
                                 "referencia: ver criterios_de_ejemplo.")
        out["criterios_de_ejemplo"] = {**D.EXAMPLE_CRITERIA, "nota": D.CCPS_CRITERIA_NOTE}
    return out


def lopa_from_hazop(flowsheet_id: str | None = None, dexpi_path: str | None = None, tolerable_frequency: float | None = None,
                    tolerable_by_severity: dict | None = None, example_criterion: str | None = None,
                    alarm_response_min: float | None = None, conditional_modifiers: list[dict] | None = None,
                    overrides: dict | None = None, min_severity: int = DEFAULT_MIN_SEVERITY) -> dict:
    """LOPA worksheet (one row per HAZOP scenario) and required SIL of each trip, for a SOLVED flowsheet
    (flowsheet_id) or a DEXPI Process file (dexpi_path). Builds the HAZOP and the minimal P&ID (pid_from_hazop rules),
    turns each HAZOP row with severity >= min_severity (or answered by a PSV or a trip) into a scenario: each cause is an
    initiating event from a cited generic table, tied to the real P&ID loop when it is a BPCS failure; credited layers
    are the PSVs, alarms and check valves traced to that row, with deterministic independence rules (no credit for a
    layer that shares a sensor, loop or final element with the cause). Mitigated frequency = ΠCM × Σ f_IE × ΠPFD.
    The tolerable frequency is the owner's risk criterion: give tolerable_frequency (1/año), tolerable_by_severity
    {"4": 1e-5, ...} or per-scenario overrides; example_criterion ("bstg_2007", "kenexis_tmel_1e-4",
    "kenexis_tmel_1e-5") applies a published EXAMPLE, labeled as such. Without any, gap and SIL stay «por definir».
    Each trip gets required PFD, RRF and SIL band (IEC 61511-1 Table 4; Table 5 if demand > 1/año).
    conditional_modifiers [{nombre, probabilidad, fuente, filas?, nodos?}] come only from the user.
    overrides {"<fila>": {frecuencia_tolerable, frecuencias {"n.k": f}, oportunidades_por_año {"n.k": n},
    modificadores [...], capas [{tag, tipo, pfd, componentes, fuente, aplica_a}], pfd_sif {tag: pfd},
    tiempo_respuesta_min}}. alarm_response_min = minutes between alarm and consequence (user data; without it alarms
    get no credit)."""
    if bool(flowsheet_id) == bool(dexpi_path):
        raise ValueError("da flowsheet_id (flowsheet resuelto) o dexpi_path (DEXPI Process), uno de los dos")
    if flowsheet_id:
        from packages.dwsim_client import tools as T
        from packages.hazop.graph import build_graph
        g, missing = build_graph(T._s(flowsheet_id)["fs"], flowsheet_id), []
    else:
        from packages.hazop.dexpi_graph import graph_from_dexpi, load_dexpi
        p = Path(dexpi_path)
        g, missing = graph_from_dexpi(load_dexpi(p), p.stem)
    out = lopa_from_graph(g, tolerable_frequency, tolerable_by_severity, example_criterion, alarm_response_min,
                          conditional_modifiers, overrides, min_severity)
    out["missing"] = missing
    return out
