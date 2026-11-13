"""HAZOP rows + minimal P&ID (bloque M10) → LOPA scenarios, deterministically.

* Scenario: one HAZOP row (deviation + consequence) with severity ≥ `min_severity`, or any row the P&ID answered with
  a PSV or a trip.
* Initiating events: one per cause of the row, classified by fixed rules into the generic table of `data` (BPCS loop
  failure, pump seal, pump or compressor stop, cooling water, external fire, operator error per opportunity). A BPCS
  cause is tied to the real loop of the P&ID (its controller, transmitter and final element tags), so independence is
  checked on components, not on names. A cause outside the table keeps frequency «por definir» (dato del usuario).
* Layers: only the P&ID elements traced to that HAZOP row (PSV, check valve, alarm, trip) plus what the user
  declares. Trips are the SIFs whose SIL the LOPA determines. HAZOP safeguards without a P&ID element (procedures,
  inspection, detectors, design) are listed as not credited, with the reason.
"""
from __future__ import annotations

import re

from packages.lopa.data import INITIATING_EVENTS, IPLS, PER_OPPORTUNITY
from packages.lopa.engine import InitiatingEvent, Layer, Modifier, Scenario
from packages.pid.rules import VAR, PidSpec, row_ref

DEFAULT_MIN_SEVERITY = 3
NON_INITIATOR = ("falla de válvula check",)       # failure of the safeguard itself: it is the IPL's PFD, not a cause
OPERATOR_WORDS = ("por error", "válvula de bloqueo cerrada", "válvulas del lado frío cerradas", "no vaciado",
                  "aislada entre válvulas", "orden de arranque")
BLOCKED_WORDS = ("salida bloqueada", "salida de gas bloqueada", "salida de vapor bloqueada", "bloqueo aguas abajo",
                 "válvula de descarga cerrada", "salida de fondos cerrada")
BPCS_WORDS = ("falla del control", "abierta en falla", "falla cerrada", "que falla", "abierta de más", "sin control",
              "transmisor", "control antisurge", "válvula de líquido cerrada")
COOLING_WORDS = ("agua de enfriamiento", "utilidad fría", "pérdida de enfriamiento", "falla del condensador")


def _var_of_cause(cause: str, parameter: str) -> str | None:
    c = cause.lower()
    if "nivel" in c:
        return "L"
    if "presión" in c or "antisurge" in c:
        return "P"
    if any(k in c for k in ("temperatura", "control de t", "medio de calentamiento", "vapor al rehervidor", "refrigerante",
                            "utilidad fría")):
        return "T"
    return VAR.get(parameter)


class _Pid:
    """Lookups on the P&ID spec."""

    def __init__(self, spec: PidSpec):
        self.spec = spec
        self.loops = [i for i in spec.instruments if i.category == "lazo"]
        self.by_tag = {i.tag: i for i in spec.instruments}
        self.items = {it.tag: it for it in spec.items}

    def all_tags(self) -> set[str]:
        return set(self.by_tag) | {i.sensor for i in self.spec.instruments} | set(self.items)

    def loop_components(self, ins) -> tuple[set[str], list[str]]:
        comps = {ins.tag, ins.sensor, f"lazo:{ins.loop}"}
        tags = [ins.tag, ins.sensor]
        fe = ins.final_element
        if fe in self.items:
            comps.add(fe)
            tags.append(fe)
        if ins.cascade_to and ins.cascade_to in self.by_tag:        # a cascade fails with either controller
            slave = self.by_tag[ins.cascade_to]
            comps |= {slave.tag, slave.sensor, f"lazo:{slave.loop}"}
            tags += [slave.tag, slave.sensor]
            if slave.final_element in self.items:
                comps.add(slave.final_element)
                tags.append(slave.final_element)
        return comps, tags

    def loop_for(self, node: str, var: str | None, cause: str):
        c = cause.lower()
        if var is None:                     # no variable to read from the cause: do not guess a loop
            return None
        cand = [x for x in self.loops if x.unit == node and x.variable == var]
        if len(cand) > 1:
            pick = [x for x in cand if ("fondos" in c) == ("fondos" in x.detail)]
            cand = pick or cand
        if cand:
            return cand[0]
        # a flowsheet valve node: the loop whose final element is that valve
        for it in self.spec.items:
            if it.dwsim == node and it.category == "valvula_control":
                for x in self.loops:
                    if x.final_element == it.tag:
                        return x
        return None

    def outlet_loop(self, node: str, cause: str):
        """A BPCS loop whose control valve sits on the outlet line of `node` that the cause names (gas/vapor outlet →
        vapor lines, bottoms/liquid → liquid lines, else any outlet): a blocked outlet may be that valve failing closed."""
        c = cause.lower()
        outs = [l for l in self.spec.lines.values() if l.kind == "proceso" and l.src == node]
        vf = lambda l: l.vapor_fraction if l.vapor_fraction is not None and l.vapor_fraction == l.vapor_fraction else None
        if "gas" in c or "vapor" in c:
            outs = [l for l in outs if vf(l) is not None and vf(l) >= 0.5]
        elif "fondos" in c or "líquido" in c:
            outs = [l for l in outs if vf(l) is not None and vf(l) < 0.5]
        names = {l.name for l in outs}
        hits = []
        for x in self.loops:
            it = self.items.get(x.final_element)
            if it is not None and it.line in names:
                hits.append(x)
            elif x.cascade_to and x.unit == node and x.sensing_line in names | {None}:
                slave = self.by_tag.get(x.cascade_to)
                sl = self.items.get(slave.final_element) if slave else None
                if sl is not None and sl.line in names or slave is not None and slave.sensing_line in names:
                    hits.append(x)
        return hits[0] if len(hits) == 1 else None    # ambiguous (several outlets with valves): do not guess


def classify_cause(cause: str, row: dict, pid: _Pid) -> InitiatingEvent:
    c = cause.lower()
    node, kind = row["node"], pid.spec.units.get(row["node"], row["kind"])

    def table(key: str, comps=None, tags=None, bpcs=False, operator=False, note="") -> InitiatingEvent:
        desc, val, unit, src = INITIATING_EVENTS[key]
        freq = val if unit != PER_OPPORTUNITY else None
        if unit == PER_OPPORTUNITY:
            note = (note + "; " if note else "") + (f"{desc}: {val:g} {unit}; la frecuencia = oportunidades por año × "
                                                    f"{val:g}; las oportunidades por año son dato del usuario")
        return InitiatingEvent(cause, key, freq, src, set(comps or ()), list(tags or ()), bpcs, operator, note)

    def bpcs(loop, why: str) -> InitiatingEvent:
        if loop is None:
            return table("bpcs_loop", {f"bpcs:{node}"}, [], bpcs=True,
                         note=f"{why}; el P&ID no tiene un lazo que trazar en {node}")
        comps, tags = pid.loop_components(loop)
        return table("bpcs_loop", comps, tags, bpcs=True, note=f"{why}: lazo {loop.tag} ({', '.join(tags)})")

    if any(k in c for k in NON_INITIATOR):
        return InitiatingEvent(cause, "no_iniciador", None, "regla de esta tool",
                               note="es la falla de la salvaguarda (la check), no un evento iniciador: su PFD va en la capa")
    if any(k in c for k in OPERATOR_WORDS):
        return table("operator_error", {"operador"}, operator=True)
    if any(k in c for k in BLOCKED_WORDS):
        loop = pid.outlet_loop(node, cause)
        if loop is not None:
            return bpcs(loop, "salida bloqueada por la válvula de control que falla cerrada")
        return table("operator_error", {"operador"}, operator=True, note="salida cerrada por una válvula manual")
    if any(k in c for k in BPCS_WORDS):
        return bpcs(pid.loop_for(node, _var_of_cause(cause, row["parameter"]), cause), "falla del BPCS")
    if "fuga en sello" in c and kind == "Pump":
        return table("pump_seal", {node}, [node])
    if "paro del equipo" in c or "falla de la bomba" in c:
        if kind == "Pump" or "bomba" in c:
            return table("pump_failure", {node}, [node] if kind == "Pump" else [])
        if kind == "Compressor":
            return table("compressor_failure", {node}, [node])
    if "fuego externo" in c:
        return table("external_fire", {"fuego"})
    if any(k in c for k in COOLING_WORDS):
        return table("cooling_water", {f"utilidad-fría:{node}"})
    return InitiatingEvent(cause, "sin_dato", None, "—",
                           note="causa fuera de la tabla genérica: frecuencia por definir (dato del usuario)")


def _layers_for(ref: str, pid: _Pid, alarm_response_min: float | None) -> list[Layer]:
    out = []
    for it in pid.spec.items:
        if ref not in it.hazop:
            continue
        if it.isa == "PSV":
            desc, pfd, cond, src = IPLS["psv"]
            out.append(Layer(it.tag, "psv", desc, pfd, src, {it.tag}, note=f"condición: {cond}"))
        elif it.isa == "CHK":
            desc, pfd, cond, src = IPLS["check_valve_single"]
            out.append(Layer(it.tag, "check_valve_single", desc, pfd, src, {it.tag}, note=cond))
    for ins in pid.spec.instruments:
        if ref not in ins.hazop:
            continue
        if ins.category == "alarma":
            desc, pfd, cond, src = IPLS["alarm_operator"]
            out.append(Layer(ins.tag, "alarm_operator", desc, pfd, src, {ins.tag, ins.sensor, f"lazo:{ins.loop}", "BPCS"},
                             bpcs=True, operator=True, response_time_min=alarm_response_min,
                             note=f"condición: {cond}; usa el transmisor {ins.sensor}"))
        elif ins.category == "disparo":
            out.append(Layer(ins.tag, "sif", f"Función instrumentada de seguridad {ins.tag} ({ins.detail})", None,
                             "IEC 61511-1 (SIL por LOPA)", {ins.tag, ins.sensor, f"lazo:{ins.loop}", f"elemento-final-SIS-{ins.loop}"},
                             sif=True, note="sensor independiente del BPCS (P&ID M10); se supone elemento final propio"))
        elif ins.category == "lazo":            # a BPCS loop the HAZOP row asked for (e.g. interface level)
            desc, pfd, cond, src = IPLS["bpcs_loop"]
            comps, _ = pid.loop_components(ins)
            out.append(Layer(ins.tag, "bpcs_loop", desc, pfd, src, comps | {"BPCS"}, bpcs=True, note=f"condición: {cond}"))
    return out


def _user_layer(d: dict, n_events: int) -> Layer:
    kind = d.get("tipo", "usuario")
    base = IPLS.get(kind)
    pfd = d.get("pfd", base[1] if base else None)
    comps = set(d.get("componentes", [])) | {d["tag"]}
    applies = d.get("aplica_a")
    return Layer(d["tag"], kind, d.get("descripción", base[0] if base else kind), pfd,
                 d.get("fuente", base[3] if base else "usuario"), comps,
                 bpcs=bool(d.get("bpcs", kind in ("bpcs_loop", "alarm_operator"))),
                 operator=bool(d.get("operador", kind == "alarm_operator")), sif=kind.startswith("sif"),
                 response_time_min=d.get("tiempo_respuesta_min"),
                 applies_to=None if applies is None else {int(a) - 1 for a in applies if 0 < int(a) <= n_events},
                 note="declarada por el usuario")


def _merge_events(events: list[InitiatingEvent]) -> list[InitiatingEvent]:
    """Two causes that are the same failure (same table entry on the same components, e.g. «falla del control de T» and
    «válvula del refrigerante abierta de más» on TIC-104) are ONE initiating event: counting both double-counts it."""
    out: list[InitiatingEvent] = []
    for e in events:
        twin = next((o for o in out if e.kind not in ("sin_dato", "usuario") and o.kind == e.kind
                     and o.components == e.components and o.frequency == e.frequency and e.components), None)
        if twin is None:
            out.append(e)
        else:
            twin.cause += f" / {e.cause}"
            twin.note += "; causas fusionadas: son la misma falla (no se cuenta dos veces)"
    return out


def _event_key(n: int, k: int) -> str:
    return f"{n}.{k}"


def build_scenarios(spec: PidSpec, min_severity: int = DEFAULT_MIN_SEVERITY, tolerable: float | None = None,
                    tolerable_source: str = "", tolerable_by_severity: dict | None = None,
                    alarm_response_min: float | None = None, modifiers: list[dict] | None = None,
                    overrides: dict | None = None) -> list[Scenario]:
    """One LOPA scenario per selected HAZOP row of the P&ID spec. `overrides` is keyed by the HAZOP row number (1-based,
    as in «HAZOP #n»): {frecuencia_tolerable, modificadores, frecuencias {"n.k" or cause: 1/año},
    oportunidades_por_año {"n.k" or cause: n}, capas [...], pfd_sif {tag: PFDavg verificada},
    tiempo_respuesta_min}."""
    pid = _Pid(spec)
    overrides = {str(k): v for k, v in (overrides or {}).items()}
    tolerable_by_severity = {str(k): v for k, v in (tolerable_by_severity or {}).items()}
    traced_refs = {h for e in spec.items + spec.instruments if e.rule in ("H-PSV", "H-TRIP") for h in e.hazop}
    out = []
    for n, row in enumerate(spec.hazop_rows, start=1):
        ref = row_ref(n - 1, _RowView(row))
        ov = overrides.get(str(n), {})
        sev = int(ov["severidad"]) if "severidad" in ov else row["severity"]
        sev_src = ov.get("severidad_fuente", "usuario") if "severidad" in ov else "HAZOP"
        if sev < min_severity and ref not in traced_refs and not ov:
            continue
        events = [classify_cause(c, row, pid) for c in row["causes"]]
        freqs = ov.get("frecuencias", {})
        opps = ov.get("oportunidades_por_año", {})
        for k, e in enumerate(events, start=1):
            key = _event_key(n, k)
            if key in freqs or e.cause in freqs:
                e.frequency = float(freqs.get(key, freqs.get(e.cause)))
                e.kind, e.source = "usuario", "dato del usuario"
            elif e.kind in INITIATING_EVENTS and INITIATING_EVENTS[e.kind][2] == PER_OPPORTUNITY and (key in opps or e.cause in opps):
                n_opp = float(opps.get(key, opps.get(e.cause)))
                e.frequency = n_opp * INITIATING_EVENTS[e.kind][1]
                e.note += f"; {n_opp:g} oportunidades/año (usuario)"
        events = _merge_events([e for e in events if e.kind != "no_iniciador"])
        layers = _layers_for(ref, pid, ov.get("tiempo_respuesta_min", alarm_response_min))
        for tag, p in ov.get("pfd_sif", {}).items():
            for l in layers:
                if l.sif and l.tag == tag:
                    l.pfd, l.source, l.note = float(p), "PFDavg verificada por el usuario", l.note + "; PFD del usuario"
        layers += [_user_layer(d, len(events)) for d in ov.get("capas", [])]
        src_mods = ov["modificadores"] if "modificadores" in ov else [
            m for m in (modifiers or []) if _in_scope(m, n, row["node"])]
        mods = [Modifier(m["nombre"], float(m["probabilidad"]), m.get("fuente", "usuario")) for m in src_mods]
        if "frecuencia_tolerable" in ov:
            tol, tsrc = float(ov["frecuencia_tolerable"]), "usuario (este escenario)"
        elif str(sev) in tolerable_by_severity:
            tol, tsrc = float(tolerable_by_severity[str(sev)]), f"usuario (severidad {sev}, {sev_src})"
        else:
            tol, tsrc = tolerable, tolerable_source
        not_credited = [{"salvaguarda": s, "razón": "no es una capa verificable en el P&ID (procedimiento, inspección, "
                                                    "detección, diseño o control normal); declárala en capas si lo es"}
                        for s in row["safeguards"] if not _drawn(s, layers)]
        out.append(Scenario(ref, row["node"], row["deviation"], "; ".join(row["consequences"]), row["severity"], events,
                            layers, mods, tol, tsrc, not_credited, sev, sev_src))
    return out


ALARM_TOKEN = re.compile(r"\b[LPTF]A[HL]{1,2}\b")
TRIP_TOKEN = re.compile(r"\b[LPTF]A?S?(HH|LL)\b")


def _in_scope(m: dict, n: int, node: str) -> bool:
    """A global conditional modifier applies to the rows (`filas`) or nodes (`nodos`) it names; to all if it names none."""
    filas, nodos = m.get("filas"), m.get("nodos")
    if filas is None and nodos is None:
        return True
    return (filas is not None and n in {int(f) for f in filas}) or (nodos is not None and node in nodos)


def _drawn(safeguard: str, layers: list[Layer]) -> bool:
    """Is this HAZOP safeguard represented by one of the scenario's layers?"""
    s = safeguard.lower()
    kinds = {l.kind for l in layers}
    alarm = "alarma" in s or bool(ALARM_TOKEN.search(safeguard))
    trip = any(k in s for k in ("paro", "corte", "cierre", "interlock", "sis ")) or bool(TRIP_TOKEN.search(safeguard))
    return (any(k in s for k in ("psv", "alivio", "disco de ruptura")) and "psv" in kinds) \
        or ("check" in s and "check_valve_single" in kinds) or (alarm and "alarm_operator" in kinds) \
        or (trip and "sif" in kinds)


class _RowView:
    """Adapter so `row_ref` (which reads attributes) works on a row dict."""

    def __init__(self, d: dict):
        self.__dict__.update(d)


def tags_consistent(scenarios: list[Scenario], spec: PidSpec) -> dict:
    """Every tag the LOPA points to (failed loop of a cause, credited layer, SIF) exists in the P&ID."""
    known = _Pid(spec).all_tags()
    bad = []
    for sc in scenarios:
        for e in sc.events:
            bad += [(sc.ref, t) for t in e.tags if t not in known and t not in spec.units]
        for l in sc.layers:
            if l.note != "declarada por el usuario" and l.tag not in known:
                bad.append((sc.ref, l.tag))
    return {"ok": not bad, "faltantes": bad}
