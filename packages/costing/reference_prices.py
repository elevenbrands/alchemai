"""Reference prices and design basis (bloque M26_P): a bank of PUBLISHED values, each opened on the web on its `date_read`,
that lets the agent propose a first economic picture of a plant in Mexico (U.S. and world as fallback regions).

Nothing here is physics: it is a dated lookup. Every value comes back flagged «VALOR DE REFERENCIA PUBLICADO, NO TU
COTIZACIÓN» with its source, URL, period, age and a `stale` mark, so the caller can show it as an assumption the user
overrides. An item that is not in the bank is never filled: `found` is False, with close matches and, when it was searched
and not found, the reason.

Data: data/reference_prices.json (entries + the list of what was searched and not found). Natural gas and electricity are
NOT duplicated: they are read from the utility bank of bloque M17 (packages/costing/utilities.py).
Units: a price published per gallon stays per volume (USD/m3): no density is assumed. Unit conversions use exact factors and
are stored next to the published number. Currency conversion MXN↔USD uses the dated Banxico FIX of the bank, or the
caller's `exchange_rates` ({currency of the value: units of the requested currency per 1 unit}), and says which.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

BANK_PATH = Path(__file__).parent / "data" / "reference_prices.json"
FLAG = "VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN"
REGION_ORDER = ["MX-SIN", "MX", "US", "WORLD"]            # fallback order; a lookup starts at the requested region
KINDS = ("raw_material", "product", "utility", "labour", "fiscal", "financial", "design_basis")
REQUIRED_FIELDS = ("id", "item", "aliases", "kind", "region", "value", "unit", "currency", "basis", "period", "source", "url",
                   "date_read", "notes", "source_tier")
# Days after the END of the period at which a value is marked stale. None = statutory or convention: it does not age by date.
STALE_AFTER_DAYS = {"raw_material": 365, "product": 365, "utility": 365, "labour": 365, "financial": 90, "fiscal": None,
                    "design_basis": None}
DESIGN_BASIS_KINDS = ("design_basis", "fiscal", "financial", "labour")
METHOD = ("Consulta de un banco JSON de valores publicados y fechados (no hay cálculo físico). Coincidencia exacta del nombre "
          "normalizado (sin acentos ni mayúsculas) contra id, nombre y alias; conversión de unidades con factores exactos "
          "guardados junto al valor publicado; conversión MXN↔USD con el FIX fechado del banco o con el tipo de cambio del "
          "usuario. Biblioteca estándar de Python (json, datetime, difflib, unicodedata).")
NOT_INCLUDED = [
    "Cotizaciones reales, contratos, descuentos por volumen, fletes hasta tu planta, aranceles y costos de importación.",
    "IVA: ningún valor del banco se corrige por IVA; los costos del proyecto se manejan sin IVA.",
    "Datos físicos de tu material (contenido de aceite, KARBE, rendimientos, humedad): nunca salen de este banco.",
    "Proyección de precios: cada valor es del periodo que indica, no un pronóstico.",
]
_DESIGN_BASIS_FIELDS = ("id", "item", "kind", "region", "value", "unit", "currency", "basis", "low", "high", "table", "period",
                        "source", "url", "date_read", "source_tier", "notes", "age_days", "stale", "fallback_region_used", "flag")
_UTILITY_KEYS = {
    "fuel_per_GJ": ("gas natural", "natural gas", "gas", "ng", "combustible", "fuel"),
    "electricity_per_kWh": ("electricidad", "electricity", "electric power", "energia electrica", "power", "luz"),
}
_DEPRECIATION_BY_PLANT = [          # (keywords in plant_type, entry id); the first hit wins. The classification is ours.
    (("biodiesel", "quimic", "chem", "petroquim", "farma", "pharma"), "depreciation_chemical_mx"),
    (("azucar", "sugar", "ingenio", "etanol", "ethanol", "aceite", "oil", "molienda", "milling", "grano", "grain"),
     "depreciation_sugar_oils_milling_mx"),
    (("aliment", "food", "bebida", "beverage"), "depreciation_food_beverages_mx"),
]


def _norm(text: str) -> str:
    """Lower case, no accents, alphanumerics separated by single spaces."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


@lru_cache
def load_bank() -> dict:
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))


def _names(entry: dict) -> set[str]:
    return {_norm(entry["id"]), _norm(entry["id"].replace("_", " ")), _norm(entry["item"]), *(_norm(a) for a in entry["aliases"])}


def region_chain(region: str) -> list[str]:
    """Regions tried in order: the requested one, then the rest of MX-SIN → MX → US → WORLD that lie after it."""
    r = str(region).strip().upper()
    if r in REGION_ORDER:
        return REGION_ORDER[REGION_ORDER.index(r):]
    if r.startswith("MX-"):
        return [r, "MX", "US", "WORLD"]
    if r.startswith("US-"):
        return [r, "US", "WORLD"]
    return [r, "WORLD"]


@lru_cache
def bank_integrity() -> dict:
    """Checks of the bank itself in this run: every entry sourced and dated, conversions reproduce the stored value."""
    bank = load_bank()
    reasons: list[str] = []
    worst = 0.0
    ids = [e.get("id") for e in bank["entries"]]
    if len(set(ids)) != len(ids):
        reasons.append("hay ids repetidos en el banco")
    for e in bank["entries"]:
        missing = [k for k in REQUIRED_FIELDS if k not in e or (e[k] in ("", None) and k != "currency")]
        if missing:
            reasons.append(f"{e.get('id')}: faltan campos {missing}")
            continue
        if not str(e["url"]).startswith("https://"):
            reasons.append(f"{e['id']}: la URL no es https")
        try:
            date.fromisoformat(e["date_read"])
            if date.fromisoformat(e["period"]["start"]) > date.fromisoformat(e["period"]["end"]):
                reasons.append(f"{e['id']}: el periodo empieza después de terminar")
        except (ValueError, KeyError, TypeError):
            reasons.append(f"{e['id']}: fecha de lectura o periodo ilegibles")
        if "conversion" in e:
            res = abs(e["published_value"] * e["conversion"]["factor"] - e["value"]) / abs(e["value"])
            worst = max(worst, res)
            if res > 1e-4:
                reasons.append(f"{e['id']}: valor publicado × factor no reproduce el valor guardado (residuo {res:.2e})")
        if not (e.get("low", e["value"]) <= e["value"] <= e.get("high", e["value"])):
            reasons.append(f"{e['id']}: el valor está fuera de su rango low-high")
    return {"verified": not reasons, "bank_version": bank["version"], "n_entries": len(bank["entries"]),
            "n_searched_not_found": len(bank["not_found"]), "max_unit_conversion_residual": worst, "reasons": reasons,
            "scope": "certifica la integridad del banco y de las conversiones de esta consulta, no que el valor publicado sea el "
                     "precio que pagarás"}


def _age(entry: dict, today: date) -> tuple[int, bool, int | None, str]:
    days = (today - date.fromisoformat(entry["period"]["end"])).days
    limit = entry["stale_after_days"] if "stale_after_days" in entry else STALE_AFTER_DAYS[entry["kind"]]
    stale = limit is not None and days > limit
    text = f"{days} días (≈ {days / 30.44:.0f} meses)" if days >= 0 else f"el periodo termina en {-days} días"
    return days, stale, limit, text


def _fx(bank: dict, src: str, dst: str, exchange_rates: dict | None) -> tuple[float | None, str]:
    """Units of `dst` per 1 unit of `src`, and where the rate comes from."""
    if exchange_rates and exchange_rates.get(src) is not None:
        rate = float(exchange_rates[src])
        if not rate > 0:
            raise ValueError(f"exchange_rates['{src}'] = {exchange_rates[src]}: debe ser positivo ({dst} por 1 {src}).")
        return rate, f"tipo de cambio del usuario: {rate} {dst} por 1 {src}"
    if {src, dst} == {"MXN", "USD"}:
        fix = next(e for e in bank["entries"] if e["id"] == bank["fx_entry_id"])
        rate = fix["value"] if src == "USD" else 1.0 / fix["value"]
        return rate, (f"FIX de Banco de México del {fix['period']['end']}: {fix['value']} MXN por USD ({fix['url']}, leído el "
                      f"{fix['date_read']})")
    return None, (f"no hay tipo de cambio {src}→{dst} en el banco (solo MXN↔USD): da exchange_rates={{'{src}': <{dst} por 1 "
                  f"{src}>}}")


def _present(entry: dict, bank: dict, region: str, currency: str | None, exchange_rates: dict | None, today: date) -> dict:
    out = {"found": True, **entry, "flag": FLAG, "region_requested": region, "region_used": entry["region"],
           "fallback_region_used": entry["region"] != region}
    days, stale, limit, age_text = _age(entry, today)
    out.update(age_days=days, stale=stale, stale_after_days=limit, age_text=age_text, today=today.isoformat())
    warnings = [f"{FLAG}: {entry['item']} = {entry['value']} {entry['unit']} ({entry['basis']}; {entry['source']}; periodo "
                f"{entry['period']['start']} a {entry['period']['end']}; antigüedad {age_text} al {today.isoformat()}; leído el "
                f"{entry['date_read']}). Es una suposición: sustitúyela por tu dato en cuanto lo tengas."]
    if out["fallback_region_used"]:
        out["fallback_note"] = (f"No hay valor para la región {region}: se usa el de {entry['region']} (orden de respaldo "
                                f"{' → '.join(region_chain(region))}).")
        warnings.append(out["fallback_note"])
    if stale:
        warnings.append(f"ATENCIÓN, VALOR VIEJO: el periodo terminó hace {days} días (límite {limit} para «{entry['kind']}»). "
                        f"Busca el dato vigente.")
    if entry["source_tier"] != "primaria":
        warnings.append("Fuente secundaria (consultora de precios, prensa o recopilación): no es estadística oficial ni precio "
                        "publicado por el productor.")
    cur = entry["currency"]
    target = (currency or cur or "").upper() or None
    out.update(value_converted=entry["value"], unit_converted=entry["unit"], currency_converted=cur, exchange_rate_applied=None,
               exchange_rate_source=None)
    if cur is None:
        if currency:
            out["exchange_rate_source"] = "no aplica: el valor no es monetario"
    elif target != cur:
        rate, src = _fx(bank, cur, target, exchange_rates)
        out["exchange_rate_source"] = src
        if rate is None:
            out.update(value_converted=None, unit_converted=None, currency_converted=None)
            warnings.append(f"Sin conversión a {target}: {src}.")
        else:
            out.update(value_converted=entry["value"] * rate, unit_converted=entry["unit"].replace(cur, target, 1),
                       currency_converted=target, exchange_rate_applied=rate)
            for k in ("low", "high"):
                if k in entry:
                    out[f"{k}_converted"] = entry[k] * rate
            warnings.append(f"Convertido de {cur} a {target} con {src}.")
    out["warnings"] = warnings
    return out


def _lookup_utility(key: str, item: str, region: str, currency: str | None, exchange_rates: dict | None, today: date,
                    bank: dict) -> dict:
    """Natural gas and electricity live in the utility bank of bloque M17; a scalar price is returned, a tariff is pointed to."""
    from packages.costing import utilities as U
    ub = U.load_bank()
    base = {"item": item, "kind": "utility", "region_requested": region, "delegated_to": "packages.costing.utilities (banco "
            "data/utility_price_references.json); para costear servicios usa utility_costs(reference=<id>)", "flag": FLAG}
    for r in region_chain(region):
        refs = {rid: ref for rid, ref in sorted(ub["references"].items()) if ref["region"].upper() == r}
        tariffs = []
        for rid, ref in refs.items():
            for e in ref["entries"]:
                if e["key"] == key:
                    val, how = U._entry_per_unit(e)
                    unit = f"{e['currency']}/GJ" if key == "fuel_per_GJ" else f"{e['currency']}/kWh"
                    entry = {"id": rid, "item": e["commodity"], "aliases": [item], "kind": "utility", "region": r, "value": val,
                             "unit": unit, "currency": e["currency"], "basis": e.get("fuel_basis", "precio medio publicado"),
                             "period": {"start": e["period_end"][:8] + "01", "end": e["period_end"]}, "source": e["source"],
                             "url": e["url"], "date_read": ref["retrieved"], "notes": e.get("caveat", ""),
                             "source_tier": "primaria", "published_value": e["value"], "published_unit": e["unit"],
                             "conversion": {"text": how}}
                    out = _present(entry, bank, region, currency, exchange_rates, today)
                    out.update(delegated_to=base["delegated_to"], utility_reference_id=rid)
                    return out
                if key == "electricity_per_kWh" and e["key"] == "electricity_time_of_use_per_kWh":
                    tariffs.append({"utility_reference_id": rid, "label": ref["label"], "time_of_use": e["value"],
                                    "unit": e["unit"], "period": e["period"], "source": e["source"], "url": e["url"]})
        if tariffs:
            return {**base, "found": False, "region_used": r, "fallback_region_used": r != region, "time_of_use_tariffs": tariffs,
                    "reason": f"Para {r} el banco de servicios tiene tarifas por periodo horario (no un precio único): pásale a "
                              f"utility_costs el id en reference= y tu reparto horario; aquí no se promedia.",
                    "close_matches": [t["utility_reference_id"] for t in tariffs]}
    not_inc = [n for n in ub.get("not_included", []) if n["region"].upper() in region_chain(region)]
    return {**base, "found": False, "fallback_region_used": False, "close_matches": sorted(ub["references"]),
            "reason": f"El banco de servicios no tiene «{item}» para {' → '.join(region_chain(region))}."
                      + "".join(f" {n['what']}: {n['why']}" for n in not_inc)}


def lookup(item: str, region: str = "MX", currency: str | None = None, exchange_rates: dict | None = None,
           today: date | None = None) -> dict:
    """One published reference value. Never raises for an unknown item: returns found False with close matches."""
    bank = load_bank()
    today = today or date.today()
    region = str(region).strip().upper()
    q = _norm(item)
    if not q:
        return {"found": False, "item": item, "region_requested": region, "fallback_region_used": False, "flag": FLAG,
                "reason": "item vacío: da el nombre de lo que buscas.", "close_matches": []}
    for key, names in _UTILITY_KEYS.items():
        if q in names:
            return _lookup_utility(key, item, region, currency, exchange_rates, today, bank)
    chain = region_chain(region)
    for r in chain:
        hits = [e for e in bank["entries"] if e["region"] == r and q in _names(e)]
        if len(hits) == 1:
            return _present(hits[0], bank, region, currency, exchange_rates, today)
        if len(hits) > 1:
            return {"found": False, "item": item, "region_requested": region, "fallback_region_used": False, "flag": FLAG,
                    "reason": f"«{item}» es ambiguo en {r}: pide uno de los ids de close_matches.",
                    "close_matches": [h["id"] for h in hits]}
    everywhere = [e for e in bank["entries"] if q in _names(e)]
    searched = [n for n in bank["not_found"] if q == _norm(n["item"]) or q in {_norm(a) for a in n["aliases"]}]
    vocabulary = {}
    for e in bank["entries"]:
        for n in _names(e):
            vocabulary.setdefault(n, e["id"])
    close = [vocabulary[m] for m in difflib.get_close_matches(q, list(vocabulary), n=6, cutoff=0.6)]
    close += [e["id"] for e in bank["entries"] if q and any(q in n.split() or q in n for n in _names(e))]
    close = list(dict.fromkeys([e["id"] for e in everywhere] + close))[:8]
    if everywhere:
        reason = (f"«{item}» existe en el banco solo para {sorted({e['region'] for e in everywhere})}, fuera del orden de respaldo "
                  f"de {region} ({' → '.join(chain)}).")
    elif searched:
        reason = (f"«{item}» se buscó el {searched[0]['searched']} y NO se encontró en una fuente abierta y fechada: "
                  f"{searched[0]['why']} No se rellena: pide el dato al usuario.")
    else:
        reason = f"«{item}» no está en el banco de valores de referencia. No se rellena: pide el dato al usuario."
    return {"found": False, "item": item, "region_requested": region, "fallback_region_used": False, "flag": FLAG,
            "reason": reason, "close_matches": close, "searched_not_found": bool(searched)}


def _envelope(extra_warnings: list[str]) -> dict:
    ev = bank_integrity()
    return {"warnings": extra_warnings, "flag": FLAG, "method": METHOD, "not_included": list(NOT_INCLUDED), "evidence": dict(ev)}


def suggest(items: list[str], region: str = "MX", currency: str = "USD", exchange_rates: dict | None = None,
            today: date | None = None) -> dict:
    """Several reference values at once: `values` (one lookup per item), `found`, `missing` and the assumptions to show."""
    if isinstance(items, str) or not items:
        raise ValueError("items: da una lista con al menos un nombre (por ejemplo ['methanol', 'aceite de soya', 'diesel']).")
    today = today or date.today()
    values, found, missing, assumptions, warnings = {}, [], [], [], []
    for it in items:
        r = lookup(it, region, currency, exchange_rates, today)
        values[it] = r
        if not r["found"]:
            missing.append({"item": it, "reason": r["reason"], "close_matches": r.get("close_matches", [])})
            continue
        found.append(it)
        shown = (f"{r['value_converted']:.6g} {r['unit_converted']}" if r["value_converted"] is not None and
                 r["currency_converted"] != r["currency"] else f"{r['value']} {r['unit']}")
        assumptions.append(f"{it}: {shown} [{r['region_used']}, {r['basis']}, periodo {r['period']['start']} a "
                           f"{r['period']['end']}, fuente: {r['source']}, {r['url']}] — {FLAG}.")
        warnings.extend(f"{it}: {w}" for w in r["warnings"][1:])
    if missing:
        warnings.insert(0, f"Sin valor de referencia para {[m['item'] for m in missing]}: no se suponen; hay que pedirlos.")
    out = {"region": str(region).upper(), "currency": currency, "today": today.isoformat(), "values": values, "found": found,
           "missing": missing, "assumptions": assumptions, **_envelope(warnings)}
    return out


def design_basis(region: str = "MX", plant_type: str | None = None, today: date | None = None) -> dict:
    """Every design-basis, fiscal, financial and labour value of the bank for a region (with fallback), each flagged."""
    bank = load_bank()
    today = today or date.today()
    region = str(region).strip().upper()
    chain = region_chain(region)
    values, assumptions, warnings = {}, [], []
    for e in bank["entries"]:
        if e["kind"] in DESIGN_BASIS_KINDS and e["region"] in chain:
            r = _present(e, bank, region, None, None, today)
            values[e["id"]] = {k: r[k] for k in _DESIGN_BASIS_FIELDS if k in r}
            assumptions.append(f"{e['item']}: {e['value']} {e['unit']} ({e['basis']}) — suposición, ver values['{e['id']}'].")
            if r["stale"]:
                warnings.append(f"{e['id']}: valor viejo ({r['age_text']}).")
    by_kind = {k: [i for i, v in values.items() if v["kind"] == k] for k in DESIGN_BASIS_KINDS}
    not_found = [n for n in bank["not_found"] if n["kind"] in DESIGN_BASIS_KINDS]
    if "MX" not in chain:
        not_found.append({"item": f"parámetros fiscales y laborales de {region}", "kind": "fiscal", "searched": bank["version"],
                          "why": "el banco solo tiene los de México; para otra región los da el usuario", "aliases": []})
    out = {"region": region, "region_chain": chain, "plant_type": plant_type, "today": today.isoformat(), "values": values,
           "by_kind": by_kind, "not_found": not_found, "assumptions": assumptions}
    if "MX" in chain:
        dep_ids = [i for i in values if i.startswith("depreciation_")]
        sug = {"options": dep_ids, "suggested_id": None, "suggested_value": None,
               "note": "Da plant_type para proponer una fracción; la clasificación fiscal definitiva es de un fiscalista."}
        if plant_type:
            p = _norm(plant_type)
            sid = next((i for kws, i in _DEPRECIATION_BY_PLANT if any(k in p for k in kws)), "depreciation_other_mx")
            sug.update(suggested_id=sid, suggested_value=values[sid]["value"],
                       note=f"Para «{plant_type}» se propone {values[sid]['item']} ({values[sid]['value']:.0%} anual, "
                            f"{values[sid]['basis']}). La asignación de tu planta a esa fracción es una PROPUESTA de esta "
                            f"herramienta por palabras clave, no un criterio fiscal: confírmala.")
            assumptions.append("Depreciación fiscal propuesta: " + sug["note"])
        out["suggested_depreciation"] = sug
    warnings.insert(0, "Todos son valores publicados o convenciones de un estudio de referencia, devueltos como SUPOSICIONES que "
                       "el usuario puede cambiar. Las convenciones de NREL (horas, vida, construcción, financiamiento) son de un "
                       "estudio de EE. UU. de 2011.")
    if not_found:
        warnings.append(f"No encontrados (no se rellenan): {[n['item'] for n in not_found]}.")
    out.update(_envelope(warnings))
    return out
