"""Deterministic LOPA engine: initiating events × credited IPLs × conditional modifiers → mitigated event frequency,
gap against the tolerable frequency, and the required PFD / RRF / SIL of each SIF to be sized.

One scenario = one consequence (a HAZOP row). It may have several initiating events (causes); each IPL is credited
per initiating event, as in the HSE/BSTG (2007) worked example, where the tank gauge alarm gets no credit for the
initiating event «tank gauge fails»:

    f_mitigated = Π CM × Σ_i [ f_IE,i × Π_{j credited for i} PFD_j ]

Independence is enforced by rules I-1 … I-6 (`data.INDEPENDENCE_RULES`); a layer that fails a rule gets PFD = 1 and
the reason is reported. Nothing here picks a risk criterion: without a tolerable frequency the gap and the SIL stay
"por definir".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from packages.lopa.data import (ALARM_MIN_RESPONSE_MIN, HOURS_PER_YEAR, IEC61511, LOW_DEMAND_MAX_PER_YEAR, SIL_PFD_BANDS,
                                SIL_PFH_BANDS)

# layers that act before an automatic trip (they lower the demand on it) and after it (they mitigate its failure)
BEFORE_SIF = {"bpcs_loop", "alarm_operator"}
AFTER_SIF = {"psv", "rupture_disc", "dike", "flame_arrestor", "check_valve_dual", "check_valve_single"}
TIME_AT_RISK_WORDS = ("tiempo", "time at risk", "fracción del tiempo", "horas", "exposición temporal", "time-at-risk")


@dataclass
class InitiatingEvent:
    cause: str
    kind: str                         # key of data.INITIATING_EVENTS, "usuario" or "sin_dato"
    frequency: float | None           # per year; None = not known (user data needed)
    source: str
    components: set[str] = field(default_factory=set)
    tags: list[str] = field(default_factory=list)      # P&ID tags of the failed equipment (traceability)
    bpcs: bool = False
    operator: bool = False
    note: str = ""


@dataclass
class Layer:
    tag: str
    kind: str                         # key of data.IPLS, "usuario"
    description: str
    pfd: float | None                 # None = no default credit (or a SIF still to be sized)
    source: str
    components: set[str] = field(default_factory=set)
    bpcs: bool = False
    operator: bool = False
    sif: bool = False
    traced: bool = True               # I-6: traced to this scenario (P&ID) or declared by the user for it
    response_time_min: float | None = None
    applies_to: set[int] | None = None  # indices of the initiating events it can act on; None = all
    note: str = ""


@dataclass
class Modifier:
    name: str
    probability: float
    source: str                       # "usuario" or a labeled example

    @property
    def time_at_risk(self) -> bool:
        return any(w in self.name.lower() for w in TIME_AT_RISK_WORDS)


@dataclass
class Scenario:
    ref: str
    node: str
    deviation: str
    consequence: str
    severity: int | None
    events: list[InitiatingEvent]
    layers: list[Layer]
    modifiers: list[Modifier] = field(default_factory=list)
    tolerable: float | None = None
    tolerable_source: str = ""
    not_credited: list[dict] = field(default_factory=list)   # HAZOP safeguards that are not IPLs here, and why


def credit(ev: InitiatingEvent, i: int, layer: Layer, credited: list[Layer], allow_multiple_operator: bool = False,
           ignore_pfd: bool = False) -> tuple[bool, str]:
    """Can `layer` be credited for initiating event `ev` (index i), given the layers already credited for it?"""
    if layer.applies_to is not None and i not in layer.applies_to:
        return False, "no actúa sobre este evento iniciador (declarado por el usuario)"
    if not layer.traced:
        return False, "I-6: no está trazada a esta fila del HAZOP ni declarada para el escenario"
    shared = sorted(layer.components & ev.components)
    if shared:
        return False, f"I-1: comparte {', '.join(shared)} con la causa"
    if layer.kind == "alarm_operator":
        if layer.response_time_min is None:
            return False, "I-5: tiempo entre la alarma y la consecuencia por definir (dato del usuario)"
        if layer.response_time_min < ALARM_MIN_RESPONSE_MIN:
            return False, f"I-5: {layer.response_time_min:g} min < {ALARM_MIN_RESPONSE_MIN:g} min para responder"
    if ev.bpcs and layer.bpcs and any(c.bpcs for c in credited):
        return False, "I-2: la causa es una falla del BPCS y ya se acreditó otra capa del BPCS"
    if layer.operator and not allow_multiple_operator and any(c.operator for c in credited):
        return False, "I-3: el operador ya se acreditó en otra capa"
    if layer.sif and any(c.sif and c.tag == layer.tag for c in credited):
        return False, "I-4: la misma SIF no cuenta dos veces"
    if layer.pfd is None and not ignore_pfd:
        return False, ("SIF por dimensionar: su PFD requerida se calcula abajo" if layer.sif
                       else "sin PFD acreditable por defecto (dato del usuario)")
    return True, "acreditada"


def _credited_for(sc: Scenario, i: int, allow_multiple_operator: bool, skip: Layer | None = None) -> tuple[list[Layer], list[dict]]:
    ev = sc.events[i]
    credited: list[Layer] = []
    log = []
    # strongest layers first, so the conservative "one per kind" rules keep the best one
    for layer in sorted(sc.layers, key=lambda l: (l.pfd is None, l.pfd if l.pfd is not None else 1.0)):
        if layer is skip:
            continue
        ok, why = credit(ev, i, layer, credited, allow_multiple_operator)
        if ok:
            credited.append(layer)
        log.append({"tag": layer.tag, "acreditada": ok, "pfd": layer.pfd if ok else 1.0, "razón": why})
    return credited, log


def cm_product(sc: Scenario) -> float:
    return math.prod(m.probability for m in sc.modifiers) if sc.modifiers else 1.0


def mitigated(sc: Scenario, allow_multiple_operator: bool = False) -> dict:
    """Mitigated event frequency with the layers credited (SIFs still to be sized count as PFD = 1)."""
    rows, total, missing = [], 0.0, []
    for i, ev in enumerate(sc.events):
        credited, log = _credited_for(sc, i, allow_multiple_operator)
        pfd = math.prod(l.pfd for l in credited) if credited else 1.0
        f = None if ev.frequency is None else ev.frequency * pfd
        if f is None:
            missing.append(ev.cause)
        else:
            total += f
        rows.append({"causa": ev.cause, "frecuencia_por_año": ev.frequency, "pfd_capas": pfd, "frecuencia_intermedia": f,
                     "capas": log})
    cm = cm_product(sc)
    return {"eventos": rows, "producto_modificadores": cm, "frecuencia_mitigada": total * cm if sc.events else None,
            "completa": not missing, "causas_sin_frecuencia": missing}


def _clean(x: float) -> float:
    """Round to 12 significant digits so that products like 1e-5 / (0.1 × 0.01) land on the band edge they mean."""
    return float(f"{x:.12g}")


def sil_for_pfd(pfd_required: float) -> dict:
    """SIL band (IEC 61511-1 Table 4, low demand) that a SIF needs to reach PFDavg ≤ pfd_required.
    Bands are [low, high): a required PFD of exactly 0.01 is SIL 1 (RRF 100), just below 0.01 is SIL 2."""
    pfd_required = _clean(pfd_required)
    if pfd_required >= 1.0:
        return {"sil": None, "rrf": 1.0 / pfd_required if pfd_required > 0 else math.inf,
                "texto": "no requiere reducción adicional: las otras capas ya cumplen", "alcanzable": True}
    rrf = 1.0 / pfd_required
    if pfd_required >= 0.1:
        return {"sil": 0, "rrf": rrf, "texto": "RRF ≤ 10: no exige SIL; basta una capa independiente con PFD ≤ la "
                                                "requerida (o una SIF sin clasificación SIL)", "alcanzable": True}
    for sil, lo, hi in SIL_PFD_BANDS:
        if lo <= pfd_required < hi:
            txt = f"SIL {sil}"
            if sil == 4:
                txt += " (evitar en la industria de proceso: conviene rediseñar o agregar capas)"
            return {"sil": sil, "rrf": rrf, "texto": txt, "alcanzable": True}
    return {"sil": None, "rrf": rrf, "texto": "más allá de SIL 4 (PFD < 1e-5): una SIF no basta; rediseñar o agregar "
                                              "capas independientes", "alcanzable": False}


def sil_for_pfh(pfh_required: float) -> dict:
    """SIL band (IEC 61511-1 Table 5, high or continuous demand) for a dangerous failure rate ≤ pfh_required (1/h)."""
    pfh_required = _clean(pfh_required)
    if pfh_required >= 1e-5:
        return {"sil": 0, "texto": "PFH ≥ 1e-5/h: no exige SIL", "alcanzable": True}
    for sil, lo, hi in SIL_PFH_BANDS:
        if lo <= pfh_required < hi:
            return {"sil": sil, "texto": f"SIL {sil} (demanda alta)", "alcanzable": True}
    return {"sil": None, "texto": "más allá de SIL 4 en demanda alta: rediseñar", "alcanzable": False}


def size_sif(sc: Scenario, sif: Layer, allow_multiple_operator: bool = False) -> dict:
    """Required PFD of an (unsized) SIF so that the scenario meets its tolerable frequency.

    Initiating events split in two sets: A, those the SIF can act on (independent of it), and B, the rest.
    PFD_req = (F_tol − F_B) / F_A, with F_A and F_B the mitigated frequencies without the SIF."""
    if sc.tolerable is None:
        return {"sif": sif.tag, "estado": "sin criterio",
                "texto": "falta la frecuencia tolerable (criterio del owner o de la empresa): SIL por definir"}
    cm = cm_product(sc)
    fa = fb = demand = 0.0
    after_max = 0.0   # least mitigation after the SIF among the events it protects (conservative)
    missing = []
    for i, ev in enumerate(sc.events):
        credited, _ = _credited_for(sc, i, allow_multiple_operator, skip=sif)
        if ev.frequency is None:
            missing.append(ev.cause)
            continue
        f = ev.frequency * (math.prod(l.pfd for l in credited) if credited else 1.0)
        ok, _ = credit(ev, i, sif, credited, allow_multiple_operator, ignore_pfd=True)
        if ok:
            fa += f * cm
            demand += ev.frequency * math.prod(l.pfd for l in credited if l.kind in BEFORE_SIF)
            after_max = max(after_max, math.prod(l.pfd for l in credited if l.kind in AFTER_SIF))
        else:
            fb += f * cm
    out = {"sif": sif.tag, "f_con_sif_posible": fa, "f_sin_proteccion_de_la_sif": fb, "demanda_por_año": demand,
           "causas_sin_frecuencia": missing}
    if fa == 0.0:
        out.update(estado="sin demanda", texto="ningún evento iniciador con frecuencia conocida llega a esta SIF")
        return out
    if fb >= sc.tolerable:
        out.update(estado="no alcanzable", texto="las causas que la SIF no protege ya superan la frecuencia tolerable")
        return out
    pfd_req = (sc.tolerable - fb) / fa
    band = sil_for_pfd(pfd_req)
    out.update(estado="calculado" if not missing else "cota (faltan frecuencias)", pfd_requerida=pfd_req,
               rrf_requerido=band["rrf"], sil=band["sil"], sil_texto=band["texto"], alcanzable=band["alcanzable"],
               modo="demanda baja" if demand <= LOW_DEMAND_MAX_PER_YEAR else "demanda alta", fuente_sil=IEC61511,
               nota="la SIF se verifica a PFDavg ≤ la requerida, no solo a la banda del SIL")
    if demand > LOW_DEMAND_MAX_PER_YEAR:
        pfh = (sc.tolerable - fb) / (cm * after_max) / HOURS_PER_YEAR
        hb = sil_for_pfh(pfh)
        out.update(pfh_requerida_por_hora=pfh, sil=hb["sil"], sil_texto=hb["texto"], alcanzable=hb["alcanzable"],
                   nota="más de una demanda por año: se usa la tasa de falla peligrosa por hora (IEC 61511-1 Tabla 5)")
    if missing:
        out["nota"] += "; faltan frecuencias de algunas causas: la PFD requerida real es menor (más exigente)"
    return out


def evaluate(sc: Scenario, allow_multiple_operator: bool = False) -> dict:
    m = mitigated(sc, allow_multiple_operator)
    out = {"ref": sc.ref, "mitigada": m}
    f = m["frecuencia_mitigada"]
    if sc.tolerable is None:
        out.update(estado="sin criterio", brecha=None)
    elif f is None:
        out.update(estado="sin causas", brecha=None)
    elif all(e.frequency is None for e in sc.events):
        out.update(estado="incompleto: ninguna causa tiene frecuencia (dato del usuario)", brecha=None)
    else:
        ratio = f / sc.tolerable
        out["brecha"] = ratio
        if ratio <= 1.0:
            out["estado"] = "cumple" if m["completa"] else "incompleto: cumple solo con las causas que tienen frecuencia"
        else:
            out["estado"] = "no cumple" if m["completa"] else "no cumple (y faltan frecuencias: la brecha real es mayor)"
    sifs = [l for l in sc.layers if l.sif and l.pfd is None and l.traced]
    out["sif"] = [size_sif(sc, s, allow_multiple_operator) for s in sifs]
    out["avisos"] = [f"modificador «{x.name}» es de tiempo en riesgo: no aplica a fallas latentes (Casey 2019)"
                     for x in sc.modifiers if x.time_at_risk]
    if allow_multiple_operator and sum(l.operator for l in sc.layers) > 1:
        out["avisos"].append("se acreditó al operador en más de una capa (sin diversidad de respuesta)")
    return out


def alarm_with_operator_pfd(pfd_alarm: float, pfd_operator: float) -> float:
    """PFD of «alarm + operator response» as in HSE/BSTG (2007) ¶33: PFD = (1 − PFD_alarm) × PFD_operator + PFD_alarm."""
    return (1.0 - pfd_alarm) * pfd_operator + pfd_alarm
