"""Design-heuristics review: the "engineer's judgement" layer. It does not compute a design; it criticises one.

Rules come from `heuristics_catalog.json`: each has a range (low/high) or a typical value, the source consulted (open
university notes or public documents, with URL and location) and the source those notes cite (Walas, Seider, Towler,
Turton). Rules were transcribed by a research subagent from the consulted documents, not from memory; several sources
disagree and are kept as separate rules. A finding means "outside usual practice according to <source>", never "wrong":
heuristics are starting points for preliminary design.

Two inputs, combinable:
  * `checks`: [{rule_id, value, label?}] — quantities the caller already has (R/R_min, N/N_min, NPSHA/NPSHR, τ...).
  * `flowsheet_id`: a solved DWSIM flowsheet; quantities are extracted automatically from compressors (stage pressure
    ratio, discharge temperature), heat exchangers (minimum approach), coolers and heaters (outlet temperature against
    cooling-water and steam limits), and column products (condenser and reboiler temperatures).
"""
from __future__ import annotations

import json
import os
from typing import Any

_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "heuristics_catalog.json")
TYPICAL_BAND = 0.25          # a value with only a "typical" is reported (info) when it differs more than this
EPS = 1e-6                   # relative slack so that a value set exactly at a limit is not reported
# How a rule's numbers are read when the catalogue's low/high/typical alone would mislead (decided here, from the rule
# text; the catalogue keeps the numbers as transcribed):
#   "min": the typical value is a MINIMUM (ΔT_min): below it is outside usual practice, above it is fine.
#   "low_only": only the lower bound matters (lowest temperature cooling water reaches; condensing with water).
#   "high_only": only the upper bound matters.
#   "info_above": crossing the low value is a design consequence to report, not a deviation (above 400 °C: fired heater).
CHECK = {"hx.min_approach.walas": "min", "hx.min_approach.seider.above_ambient": "min",
         "hx.min_approach.seider.high_temp": "min", "hx.approach.csjmu.recovery": "min",
         "util.cw.min_process_temp.towler": "low_only", "dist.condenser_cw_temp": "low_only",
         "dist.reboiler_max_temp": "high_only", "hx.furnace_threshold": "info_above"}


def catalog() -> list[dict]:
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _by_id() -> dict[str, dict]:
    return {r["id"]: r for r in catalog()}


def _source(r: dict) -> dict:
    s = r.get("source_consulted") or {}
    return {"consulted": s.get("title"), "url": s.get("url"), "location": s.get("location"),
            "cited_by_it": r.get("source_cited"), "confidence": r.get("confidence")}


def evaluate(rule: dict, value: float, label: str | None = None) -> dict:
    lo, hi, typ = rule.get("low"), rule.get("high"), rule.get("typical")
    v = float(value)
    f: dict[str, Any] = {"rule_id": rule["id"], "area": rule["area"], "rule": rule["rule_es"], "quantity": rule["quantity"],
                         "value": v, "units": rule["units"], "range": {"low": lo, "high": hi, "typical": typ},
                         "applies_when": rule.get("applies_when"), "source": _source(rule)}
    if label:
        f["item"] = label
    mode = CHECK.get(rule["id"])
    if mode == "min" and typ is not None:
        lo, typ = typ, None
    elif mode == "low_only":
        hi = None
    elif mode == "high_only":
        lo = None
    elif mode == "info_above":
        crossed = lo is not None and v > lo
        f.update(status="consequence" if crossed else "within_usual", severity="info" if crossed else "ok",
                 message=(f"{v:g} {rule['units']} supera {lo:g}: {rule['rule_es']}" if crossed else "no aplica a este valor"))
        f["source"] = _source(rule)
        return f
    f["check"] = mode or "range/typical"
    if lo is not None and v < lo * (1 - EPS) - EPS:
        f.update(status="below_usual", severity="warning",
                 message=f"{v:g} {rule['units']} está por DEBAJO de la práctica usual (≥ {lo:g}) según {f['source']['consulted']}")
    elif hi is not None and v > hi * (1 + EPS) + EPS:
        f.update(status="above_usual", severity="warning",
                 message=f"{v:g} {rule['units']} está por ENCIMA de la práctica usual (≤ {hi:g}) según {f['source']['consulted']}")
    elif lo is not None or hi is not None:
        f.update(status="within_usual", severity="ok", message="dentro de la práctica usual")
    elif typ is not None:
        dev = (v - typ) / typ if typ else None
        f["deviation_from_typical"] = dev
        if dev is not None and abs(dev) > TYPICAL_BAND:
            f.update(status="differs_from_typical", severity="info",
                     message=f"{v:g} difiere {100 * dev:+.0f} % del valor típico {typ:g} {rule['units']}: no es un límite, "
                             "revisa si la diferencia está justificada")
        else:
            f.update(status="near_typical", severity="ok", message="cerca del valor típico")
    else:
        f.update(status="not_evaluable", severity="info", message="la regla no tiene intervalo ni valor típico numérico")
    if rule.get("notes"):
        f["rule_notes"] = rule["notes"]
    return f


# ------------------------------------------------------------------------------------------- flowsheet extraction
def _C(T_K: float) -> float:
    return T_K - 273.15


def _stream(fid, name):
    from packages.dwsim_client import tools as T
    return T.read_stream(fid, name)


def from_flowsheet(flowsheet_id: str) -> tuple[list[tuple[str, float, str]], list[str]]:
    """[(rule_id, value, label)] measured on the solved flowsheet, and notes on what could not be extracted."""
    from packages.dwsim_client import tools as T
    s = T.SESSIONS[flowsheet_id]
    items: list[tuple[str, float, str]] = []
    notes: list[str] = []
    for unit, kind in s["kinds"].items():
        conn = s.get("conn", {}).get(unit, {})
        feeds, prods = conn.get("feeds") or [], conn.get("products") or []
        try:
            fin = [_stream(flowsheet_id, n) for n in feeds]
            fout = [_stream(flowsheet_id, n) for n in prods]
        except Exception as ex:  # noqa: BLE001
            notes.append(f"{unit}: no se pudieron leer sus corrientes ({type(ex).__name__})")
            continue
        if kind == "compressor" and fin and fout:
            r = fout[0]["P_Pa"] / fin[0]["P_Pa"]
            items.append(("comp.ratio_per_stage", r, f"{unit}: relación de compresión"))
            items.append(("comp.max_discharge_temp", _C(fout[0]["T_K"]), f"{unit}: T de descarga"))
        elif kind == "heat_exchanger" and len(fin) == 2 and len(fout) == 2:
            pairs = list(zip(fin, fout))
            hot, cold = sorted(pairs, key=lambda p: -p[0]["T_K"])
            d1 = hot[0]["T_K"] - cold[1]["T_K"]
            d2 = hot[1]["T_K"] - cold[0]["T_K"]
            dmin = min(d1, d2)
            level = max(hot[0]["T_K"], cold[1]["T_K"])
            if min(cold[0]["T_K"], hot[1]["T_K"]) < 298.15:
                rid = "hx.min_approach.seider.below_ambient"
            elif _C(level) > 150:
                rid = "hx.min_approach.seider.high_temp"
            else:
                rid = "hx.min_approach.seider.above_ambient"
            items.append((rid, dmin, f"{unit}: acercamiento mínimo (contracorriente)"))
            items.append(("hx.min_approach.walas", dmin, f"{unit}: acercamiento mínimo (contracorriente)"))
            if dmin <= 0:
                notes.append(f"{unit}: acercamiento ≤ 0 K: cruce de temperaturas, imposible en contracorriente pura")
        elif kind == "cooler" and fout:
            Tc = _C(fout[0]["T_K"])
            items.append(("util.cw.min_process_temp.towler", Tc, f"{unit}: T de salida del enfriador"))
        elif kind == "heater" and fout:
            Th = _C(fout[0]["T_K"])
            if Th > 150:
                items.append(("dist.reboiler_max_temp", Th, f"{unit}: T de salida (límite del vapor de 150 psig)"))
            items.append(("hx.furnace_threshold", Th, f"{unit}: T de salida (¿hace falta horno?)"))
        elif kind in ("distillation_column", "shortcut_column") and len(fout) >= 2:
            items.append(("dist.condenser_cw_temp", _C(fout[0]["T_K"]), f"{unit}: T del destilado (condensador)"))
            items.append(("dist.reboiler_max_temp", _C(fout[1]["T_K"]), f"{unit}: T de fondos (rehervidor)"))
            if kind == "shortcut_column":
                try:
                    u = T.read_unit(flowsheet_id, unit)
                    R = float(s["params"][unit].get("reflux_ratio")) if s["params"][unit].get("reflux_ratio") else None
                    if R and u.get("R_min"):
                        items.append(("dist.reflux_ratio.range", R / float(u["R_min"]), f"{unit}: R/R_min"))
                    if u.get("N_theoretical") and u.get("N_min"):
                        items.append(("dist.stages_vs_min", float(u["N_theoretical"]) / float(u["N_min"]), f"{unit}: N/N_min"))
                except Exception as ex:  # noqa: BLE001
                    notes.append(f"{unit}: no se leyó R_min/N_min ({type(ex).__name__})")
            else:
                notes.append(f"{unit}: columna rigurosa sin R_min: da R/R_min en checks (p. ej. de shortcut_distillation_design)")
    return items, notes


# ------------------------------------------------------------------------------------------- tool body
def review(checks: list[dict] | None = None, flowsheet_id: str | None = None, list_areas: list[str] | None = None) -> dict:
    rules = _by_id()
    if list_areas is not None:
        sel = [r for r in rules.values() if not list_areas or r["area"] in list_areas]
        return {"ok": True, "areas": sorted({r["area"] for r in rules.values()}),
                "rules": [{"rule_id": r["id"], "area": r["area"], "rule": r["rule_es"], "quantity": r["quantity"],
                           "low": r.get("low"), "high": r.get("high"), "typical": r.get("typical"), "units": r["units"],
                           "applies_when": r.get("applies_when"), "confidence": r.get("confidence")} for r in sel]}
    if not checks and not flowsheet_id:
        raise ValueError("da checks [{rule_id, value}], flowsheet_id, o list_areas para ver el catálogo")
    findings, notes = [], []
    for i, c in enumerate(checks or []):
        rid = c.get("rule_id")
        if rid not in rules:
            near = [k for k in rules if rid and rid.split(".")[0] == k.split(".")[0]][:8]
            raise ValueError(f"checks[{i}]: rule_id desconocido {rid!r}; del área: {near}. Usa list_areas para el catálogo")
        if c.get("value") is None:
            raise ValueError(f"checks[{i}]: falta value (en {rules[rid]['units']})")
        findings.append(evaluate(rules[rid], c["value"], c.get("label")))
    if flowsheet_id:
        items, n2 = from_flowsheet(flowsheet_id)
        notes += n2
        for rid, v, lab in items:
            if rid in rules:
                findings.append(evaluate(rules[rid], v, lab))
            else:
                notes.append(f"regla {rid} no está en el catálogo")
    order = {"warning": 0, "info": 1, "ok": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 3))
    warn = [f for f in findings if f["severity"] == "warning"]
    return {"ok": True, "n_checked": len(findings), "n_outside_usual_practice": len(warn),
            "findings": findings, "notes": notes,
            "how_to_read": ("Cada hallazgo dice si el valor queda fuera de la práctica usual SEGÚN la fuente citada. Las "
                            "heurísticas son puntos de partida de diseño preliminar, no límites físicos: un valor fuera "
                            "del intervalo pide una justificación, no es un error. Varias fuentes discrepan (acercamiento "
                            "mínimo, agua de enfriamiento, vapor de media): se muestran por separado."),
            "catalog": {"rules": len(rules), "file": "packages/review/heuristics_catalog.json",
                        "provenance": "reglas transcritas de documentos abiertos (apuntes universitarios que reproducen a "
                                      "Walas, Seider y Towler; DOE); ver source.url de cada hallazgo"}}
