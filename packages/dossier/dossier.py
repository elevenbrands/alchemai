"""One question → the whole plant (bloque M27): explanation, profitability, PFD with opportunities marked, report.

`process_dossier` chains engines that already exist and are validated on their own — it adds NO new physics:
  * refinery  → `packages.refinery.model.run_model` (as operated, calibrated where Pemex does not publish)
  * biodiesel / reference biorefinery → `packages.plants` + `packages.costing.plant_bridge.plant_economics_from_model`
  * opportunities → `packages.opportunities.scan.scan_opportunities` (+ for a refinery, the value of the idle capacity of
    each conversion unit whose utilization was CALIBRATED, computed by re-running the same model with that unit at its
    published capacity: same prices, same yields; restart CAPEX not included)
  * diagram → `packages.diagram` (sections view + full view, opportunities numbered on their units)
It writes `<out_dir>/<slug>.md`, `<slug>_secciones.svg`, `<slug>_completo.svg` and returns the paths plus a compact summary
the agent turns into prose. Every number keeps the evidence and the flags of the engine that produced it.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

SECTION_TEXT = {
    "crudo": "Recibe el crudo (mezcla de Maya, Istmo y Olmeca según lo publicado), lo almacena y lo bombea al tren de proceso.",
    "destilacion": "Destilación atmosférica y al vacío: separa el crudo por punto de ebullición en gases, nafta, kerosina, "
                   "gasóleos y residuo. Es la base de todo; las demás unidades transforman estos cortes.",
    "conversion": "Conversión: el craqueo catalítico (FCC) rompe gasóleos pesados en gasolina y gas LP; la coquizadora "
                  "convierte el residuo de vacío en productos ligeros y coque. Aquí se gana la mayor parte del margen.",
    "gasolinas": "Mejoramiento de gasolinas: la reformadora sube el octanaje de la nafta (y produce hidrógeno); la "
                 "alquilación e isomerización hacen componentes de alto octano para la mezcla.",
    "hidrotratamiento": "Hidrotratamiento (HDS): quita el azufre de nafta, turbosina y diésel con hidrógeno para cumplir la "
                        "NOM-016-CRE-2016.",
    "servicios": "Servicios: hidrógeno, tratamiento de gases ácidos y recuperación de azufre, vapor, agua y energía.",
    "productos": "Mezclado y almacenamiento de productos terminados para venta.",
}


def _slug(text: str) -> str:
    s = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "proceso"


def _fmt(x, nd=0) -> str:
    if x is None:
        return "—"
    return f"{x:,.{nd}f}"


def _refinery_capacity_opportunities(name: str, base: dict, period: str | None, prices: dict | None) -> list[dict]:
    """Value of running each CALIBRATED (under-used) conversion unit at its published capacity, same model and prices."""
    from packages.refinery import model as RM
    out = []
    cal = base.get("calibrated_parameters") or {}
    base_net = (base.get("economics") or {}).get("annual_usd")
    for key, unit_id in (("coker_utilization", "coker"),):
        c = cal.get(key)
        if not c or base_net is None or c.get("value") is None or c["value"] >= 0.95:
            continue
        # what-if: only the coker changes; every other calibrated parameter is held at its base value (otherwise the
        # cutter / splits would be re-fitted to the published slate and absorb the change)
        held = {k: v["value"] for k, v in cal.items() if k != key and k in RM.OVERRIDE_KEYS}
        alt = RM.run_model(refinery=name, period=period, utilization={unit_id: 1.0}, prices=prices, unit_overrides=held)
        alt_net = (alt.get("economics") or {}).get("annual_usd")
        if alt_net is None:
            continue
        delta = alt_net - base_net
        unit = next((u for u in base["process_model"]["units"] if u["id"].startswith("coqu") or unit_id in u["id"]), None)
        out.append({
            "id": f"capacidad_{unit_id}", "unit_ids": [unit["id"]] if unit else [], "stream_ids": [],
            "category": "yield",
            "title_es": "Completar el arranque de la coquizadora y operarla a su capacidad publicada",
            "explanation_es": (f"La coquizadora opera al {c['value'] * 100:.0f} % (valor CALIBRADO para reproducir el "
                               f"{c.get('fitted_to', 'combustóleo')} publicado). A capacidad, convierte residuo de vacío (que hoy sale como combustóleo, el producto "
                               "más barato) en gasolina, diésel y gas; el modelo valúa ese cambio con los mismos precios."),
            "value_usd_y": delta, "value_low_usd_y": 0.5 * delta, "value_high_usd_y": delta,
            "basis_es": "diferencia del margen neto anual del mismo modelo con la coquizadora al 100 % vs como opera",
            "assumptions": ["rendimientos fijos publicados de coquización (curso CBEN 409, Colorado School of Mines)",
                            "mismos precios de referencia; no incluye la inversión ni el tiempo para terminar el arranque",
                            "el valor bajo es la mitad: el modelo sobrestima diésel y líquidos (ver validación)",
                            "el coque no tiene precio publicado: se valúa en cero (eso baja el valor)"],
            "evidence": {"base_annual_usd": base_net, "alternative_annual_usd": alt_net}, "confidence": "media",
            "rule": "capacidad_ociosa", "include_in_total": True})
    return out


def _explanation_refinery(r: dict) -> list[str]:
    lines = []
    secs = {s["id"]: s["label_es"] for s in r["process_model"]["sections"]}
    by_sec: dict[str, list[str]] = {}
    for u in r["process_model"]["units"]:
        by_sec.setdefault(u["section"], []).append(u["label_es"])
    for sid, label in secs.items():
        text = SECTION_TEXT.get(sid, "")
        lines.append(f"- **{label}.** {text} Equipos: {', '.join(by_sec.get(sid, []))}.")
    return lines


def _liquid_excess(r: dict) -> dict | None:
    """Short volume diagnosis for the app panel: model vs published liquids as % of the crude run."""
    d = (r.get("published_comparison") or {}).get("liquid_excess_diagnosis") or {}
    crude = r.get("crude_bpd")
    if not d.get("applicable") or not crude or d.get("liquids_excess_mbd") is None:
        return None
    pub = sum(v for k, v in (r.get("published_products_bpd") or {}).items() if v) or None
    excess = 1000.0 * d["liquids_excess_mbd"]
    out = {"liquids_excess_bpd": excess, "diesel_excess_bpd": 1000.0 * (d.get("diesel_excess_mbd") or 0.0),
           "own_use_upper_bound_bpd": 1000.0 * (d.get("own_use_upper_bound_liquids_mbd") or 0.0),
           "excess_pct_of_crude": 100.0 * excess / crude, "source": d.get("source"), "published_bpd": pub}
    out["text_es"] = (f"El modelo hace {excess / 1000:.1f} mil bbl/d de líquidos más que lo publicado "
                      f"({out['excess_pct_of_crude']:.1f} % del crudo), casi igual al exceso de diésel "
                      f"({out['diesel_excess_bpd'] / 1000:.1f} mil). El autoconsumo explica a lo más "
                      f"{out['own_use_upper_bound_bpd'] / 1000:.1f} mil; el resto no tiene dato público (mermas, "
                      "transferencias, cortes de destilación).")
    return out


def _refinery_dossier(name: str, region: str, prices: dict | None, plant_data, period: str | None) -> tuple[dict, dict, list]:
    from packages.opportunities import scan as SC
    from packages.refinery import model as RM
    r = RM.run_model(refinery=name, period=period, prices=prices)
    scan = SC.scan_opportunities(r["process_model"], prices=None, region=region, plant_data=plant_data)
    opps = _refinery_capacity_opportunities(name, r, period, prices) + list(scan.get("opportunities") or [])
    for i, o in enumerate(sorted(opps, key=lambda o: -(o.get("value_usd_y") or 0)), 1):
        o["rank"] = i
    e = r["economics"]
    summary = {
        "kind": "refinery", "name": r["name"], "period": r["period"], "crude_bpd": r["crude_bpd"], "mode": r["mode"],
        "gross_margin_model_usd_bbl": e.get("gross_margin_usd_bbl"),
        "gross_margin_at_published_slate_usd_bbl": e.get("gross_margin_at_published_slate_usd_bbl"),
        "net_margin_model_usd_bbl": e.get("net_margin_usd_bbl"), "annual_net_model_usd": e.get("annual_usd"),
        "pemex_reference": e.get("pemex_variable_margin_2025"),
        "calibrated_parameters": {k: v.get("value") for k, v in (r.get("calibrated_parameters") or {}).items()},
        "validation": [{"product": v.get("product"), "deviation_pct": v.get("deviation_pct")}
                       for v in (r.get("published_comparison") or {}).get("validation_rows", [])],
        "energy_GJ_bbl": (r.get("energy") or {}).get("GJ_per_bbl"), "co2": (r.get("emissions") or {}),
        "scan_not_assessed": scan.get("not_assessed"), "evidence_verified": r["evidence"].get("verified"),
        "calibration_summary": r.get("calibration_summary"),
        "liquid_excess": _liquid_excess(r),
    }
    return r, summary, opps


def _md_refinery(r: dict, s: dict, opps: list, svg_sec: str, svg_full: str) -> str:
    e = r["economics"]
    L = [f"# {r['name']} — proceso, rentabilidad y oportunidades", "",
         f"Periodo {r['period']} · modo «{r['mode']}» (calibrado a la producción publicada de Pemex donde Pemex no publica "
         f"la operación) · generado el {date.today().isoformat()} por AlchemAI. Modelo de PLANEACIÓN por rendimientos, no "
         "simulación rigurosa de cada reactor.", "",
         f"Diagramas: [vista por secciones]({Path(svg_sec).name}) · [vista completa]({Path(svg_full).name})", "",
         "## Qué hace la refinería", "", f"Procesa {_fmt(r['crude_bpd'])} barriles diarios de crudo.", ""]
    L += _explanation_refinery(r)
    L += ["", "## Productos (modelo) y validación contra lo publicado", "",
          "| Producto | Modelo, bbl/d | Modelo, t/h | % vol. del crudo |", "|---|---|---|---|"]
    for p, v in r["product_slate"].items():
        if isinstance(v, dict) and (v.get("bpd") or v.get("kg_h")):
            bpd = _fmt(v["bpd"]) if v.get("bpd") else "— (se reporta en masa)"
            L.append(f"| {p} | {bpd} | {_fmt((v.get('kg_h') or 0) / 1000, 1)} | {_fmt(v.get('vol_pct_of_crude'), 1)} |")
    L += ["", "| Producto | Tipo de comparación | Desviación |", "|---|---|---|"]
    for row in (r.get("published_comparison") or {}).get("rows", []):
        L.append(f"| {row.get('product')} | {row.get('kind') or row.get('role') or ''} | "
                 f"{_fmt(row.get('deviation_pct'), 1)} % |")
    cs = r.get("calibration_summary") or {}
    if cs.get("text_es"):
        L += ["", f"**Calidad de la calibración:** {cs['text_es']}"]
    L += ["", "Parámetros CALIBRADOS (no publicados por Pemex; ajustados para reproducir un producto publicado):", ""]
    for k, v in (r.get("calibrated_parameters") or {}).items():
        L.append(f"- {k}: {v.get('value'):.3f} (ajustado a {v.get('fitted_to')})")
    L += ["", "## Rentabilidad (margen de operación; la refinería ya existe)", "",
          "| Indicador | USD/bbl |", "|---|---|",
          f"| Margen bruto del modelo | {_fmt(e.get('gross_margin_usd_bbl'), 2)} |",
          f"| Margen bruto con la producción PUBLICADA (misma mezcla de crudo y precios) | "
          f"{_fmt(e.get('gross_margin_at_published_slate_usd_bbl'), 2)} |",
          f"| Energía | {_fmt(e.get('energy_cost_usd_bbl'), 2)} |", f"| Costo fijo (referencia) | {_fmt(e.get('fixed_cost_usd_bbl'), 2)} |",
          f"| Margen neto del modelo | {_fmt(e.get('net_margin_usd_bbl'), 2)} |", "",
          "La cifra más defendible es el margen con la producción publicada: el modelo produce más diésel y líquidos que la "
          "refinería real (ver validación), lo que infla su margen. " + str(e.get("gross_margin_at_published_slate_note", "")), ""]
    if e.get("pemex_variable_margin_2025"):
        ref = e["pemex_variable_margin_2025"]
        L += [f"Referencia de Pemex: {ref.get('value')} {ref.get('unit')} ({ref.get('scope')}; {ref.get('source')}, "
              f"{ref.get('url')}). No es la misma base: es todo el sistema y otro año.", ""]
    L += ["Precios usados (referencias publicadas, NO cotizaciones):", "", "| Precio | Valor | Fuente |", "|---|---|---|"]
    for k, v in (e.get("prices") or {}).items():
        if isinstance(v, dict):
            L.append(f"| {k} | {_fmt(v.get('value'), 2)} {v.get('unit', '')} | {str(v.get('source', ''))[:90]} |")
    L += ["", "## Oportunidades de mejora (numeradas en el diagrama)", ""]
    if not opps:
        L.append("Ninguna valuada con los datos públicos. Con lecturas de planta (opportunities_from_plant_data) se evalúan "
                 "integración de calor, hornos, vapor y energía por unidad.")
    for o in sorted(opps, key=lambda o: o.get("rank", 99)):
        val = (f"USD {_fmt(o.get('value_usd_y'))}/año (rango {_fmt(o.get('value_low_usd_y'))}–{_fmt(o.get('value_high_usd_y'))})"
               if o.get("value_usd_y") is not None else "sin valor: faltan datos")
        L += [f"{o.get('rank')}. **{o['title_es']}** — {val}. Confianza {o.get('confidence')}.", f"   {o.get('explanation_es', '')}"]
    na = s.get("scan_not_assessed") or []
    if na:
        L += ["", "No evaluado por falta de datos públicos:", ""]
        L += [f"- {x.get('rule', '')}: {x.get('reason_es') or x.get('reason') or ''}" if isinstance(x, dict) else f"- {x}"
              for x in na[:12]]
    L += ["", "## Supuestos, advertencias y lo que no incluye", ""]
    L += [f"- {a}" for a in (r.get("assumptions") or [])[:25]]
    L += [f"- ⚠ {w}" for w in (r.get("warnings") or [])[:15]]
    L += [f"- No incluye: {n}" for n in (r.get("not_included") or [])[:15]]
    return "\n".join(L) + "\n"


def _power_dossier(plant: str, year, prices) -> tuple[dict, dict, list]:
    from packages.dwsim_client import tools_power as TP
    r = TP.power_plant_model(plant, year, None, None, prices)
    opps = list(r.get("opportunities") or [])
    for i, o in enumerate(sorted(opps, key=lambda o: -(o.get("value_usd_y") or 0)), 1):
        o["rank"] = i
    e, c = r["economics"], r["cycle"]
    summary = {"kind": "power", "name": r["process_model"]["name"], "year": r.get("year"),
               "net_generation_MWh": e.get("net_generation_MWh"), "price_usd_mwh": e.get("price_usd_mwh"),
               "operating_margin_usd_mwh": e.get("operating_margin_usd_mwh"),
               "operating_margin_usd_y": e.get("operating_margin_usd_y"),
               "margin_after_capital_usd_y": e.get("margin_after_capital_usd_y"),
               "gross_efficiency": c.get("gross_efficiency"), "heat_rate_kJ_kWh": c.get("heat_rate_kJ_kWh"),
               "co2_avoided_t_y": (r.get("emissions_avoided") or {}).get("co2_avoided_t_y"),
               "calibrated_parameters": r.get("calibrated_parameters"),
               "validation": [x for x in (r.get("published_comparison") or {}).get("rows", [])],
               "evidence_verified": r["evidence"].get("verified")}
    return r, summary, opps


def _md_power(r: dict, s: dict, opps: list, svg_sec: str, svg_full: str) -> str:
    e, c = r["economics"], r["cycle"]
    L = [f"# {r['process_model']['name']} — proceso, rentabilidad y oportunidades", "",
         f"Año {r.get('year')} · generado el {date.today().isoformat()} por AlchemAI con datos públicos. Ciclo de vapor "
         "modelado con IAPWS-95 (CoolProp); el diseño del lado turbina no es público y se calibra (ver abajo).", "",
         f"Diagramas: [vista por secciones]({Path(svg_sec).name}) · [vista completa]({Path(svg_full).name})", "",
         "## Qué hace la central", "",
         "Dos reactores de agua en ebullición (BWR-5) hierven agua dentro de la vasija con el calor de la fisión; el vapor "
         "saturado mueve la turbina de alta, pasa por los separadores de humedad y recalentadores (MSR), mueve las turbinas "
         "de baja y el generador; el vapor se condensa con agua de mar y el condensado, precalentado con extracciones de "
         "la turbina, regresa al reactor. Es un ciclo Rankine directo: el mismo vapor del reactor llega a la turbina.", "",
         "| Indicador | Valor |", "|---|---|",
         f"| Potencia térmica por unidad | {_fmt(c.get('thermal_W', 0) / 1e6)} MWt |",
         f"| Potencia eléctrica bruta por unidad (calibrada) | {_fmt(c.get('gross_W', 0) / 1e6)} MWe |",
         f"| Eficiencia bruta | {_fmt(100 * (c.get('gross_efficiency') or 0), 1)} % |",
         f"| Consumo térmico unitario | {_fmt(c.get('heat_rate_kJ_kWh'))} kJ/kWh |",
         f"| Flujo de vapor por unidad | {_fmt(c.get('steam_flow_kg_s'))} kg/s |",
         f"| Calor al mar por unidad | {_fmt((c.get('condenser_duty_W') or 0) / 1e6)} MW |",
         f"| Generación neta {r.get('year')} (dos unidades) | {_fmt((e.get('net_generation_MWh') or 0) / 1e6, 2)} TWh |", "",
         "## Rentabilidad (la central ya existe: margen de operación)", "", "| Indicador | Valor |", "|---|---|",
         f"| Precio de referencia (CENACE) | {_fmt(e.get('price_usd_mwh'), 1)} USD/MWh |",
         f"| Combustible + operación | {_fmt((e.get('fuel_usd_mwh') or 0) + (e.get('operating_usd_mwh') or 0), 2)} USD/MWh |",
         f"| Margen de operación | {_fmt(e.get('operating_margin_usd_mwh'), 1)} USD/MWh · {_fmt((e.get('operating_margin_usd_y') or 0) / 1e6)} MM USD/año |",
         f"| Margen después de capital de sostenimiento | {_fmt((e.get('margin_after_capital_usd_y') or 0) / 1e6)} MM USD/año |",
         f"| CO2 evitado (factor promedio de la red) | {_fmt((s.get('co2_avoided_t_y') or 0) / 1e6, 2)} Mt/año |", "",
         "Costos de referencia de BWR de EE. UU. (NEI/EUCG), no los de CFE (no publicados).", "",
         "## Validación contra lo publicado", "", "| Cantidad | Publicado | Modelo | Papel |", "|---|---|---|---|"]
    for row in (r.get("published_comparison") or {}).get("rows", []):
        L.append(f"| {row.get('quantity')} | {row.get('published')} {row.get('unit', '')} | {_fmt(row.get('model'), 1)} | "
                 f"{row.get('role')} |")
    L += ["", "Parámetros CALIBRADOS: " + json.dumps(r.get("calibrated_parameters"), ensure_ascii=False, default=float)[:600],
          "", "## Oportunidades de mejora (numeradas en el diagrama)", ""]
    for o in sorted(opps, key=lambda o: o.get("rank", 99)):
        val = (f"USD {_fmt(o.get('value_usd_y'))}/año (rango {_fmt(o.get('value_low_usd_y'))}–{_fmt(o.get('value_high_usd_y'))})"
               if o.get("value_usd_y") is not None else "sin valor anual")
        L += [f"{o.get('rank')}. **{o['title_es']}** — {val}. Confianza {o.get('confidence')}.", f"   {o.get('explanation_es', '')}"]
    L += ["", "## Supuestos, advertencias y lo que no incluye", ""]
    L += [f"- {a}" for a in (r.get("assumptions") or [])[:25]]
    L += [f"- ⚠ {w}" for w in (r.get("warnings") or [])[:15]]
    L += [f"- No incluye: {n}" for n in (r.get("not_included") or [])[:15]]
    L += [f"- No encontrado en fuentes abiertas: {n}" for n in (r.get("not_found") or [])[:15]]
    return "\n".join(L) + "\n"


def _plant_dossier(plant: dict, region: str, prices: dict | None, plant_data) -> tuple[dict, dict, list, dict]:
    from packages.costing import plant_bridge as PB
    from packages.diagram import adapters as AD
    from packages.opportunities import scan as SC
    from packages.plants import biodiesel as BD
    from packages.plants import reference as RF
    kind = plant.get("model", "biodiesel")
    args = {k: v for k, v in plant.items() if k != "model"}
    m = BD.biodiesel_plant(**args) if kind == "biodiesel" else RF.reference_biorefinery(kind, **args)
    eco = PB.plant_economics_from_model(m, region=region, prices=prices)
    pm = AD.from_plant_result(m)
    pm["economics"] = eco
    scan = SC.scan_opportunities(pm, region=region, plant_data=plant_data)
    opps = list(scan.get("opportunities") or [])
    summary = {"kind": kind, "npv_usd": eco.get("npv_usd"), "irr": eco.get("irr"),
               "minimum_selling_price": eco.get("minimum_selling_price"),
               "TCI_usd": (eco.get("capital_summary") or {}).get("total_capital_investment_usd"),
               "confirm_first": (eco.get("sensitivity") or {}).get("confirm_first"), "n_assumed": eco.get("n_assumed"),
               "missing": eco.get("missing"), "published_benchmark": m.get("published_benchmark")}
    return m, summary, opps, {"eco": eco, "pm": pm, "scan": scan}


def process_dossier(subject: str, refinery: str | None = None, plant: dict | None = None, region: str = "MX",
                    prices: dict | None = None, plant_data=None, period: str | None = None, out_dir: str | None = None,
                    theme: str = "light", power_plant: str | None = None, year: int | None = None) -> dict:
    """See module docstring. Exactly one of `refinery` (e.g. "tula") or `plant` ({model: 'biodiesel'|'sugarcane_ethanol'|…,
    + that model's arguments}). Returns {summary, opportunities, report_path, diagram_sections_path, diagram_full_path,
    markdown, warnings, evidence}."""
    from packages.dwsim_client import tools_diagram as TD
    if sum(bool(x) for x in (refinery, plant, power_plant)) != 1:
        raise ValueError("process_dossier: da exactamente uno: refinery='tula' (u otra de list_refineries), power_plant="
                         "'laguna_verde' (list_power_plants) o plant={'model': 'biodiesel'|'sugarcane_ethanol'|'corn_ethanol'|"
                         "'cornstover_ethanol', …argumentos del modelo}.")
    base = Path(out_dir) if out_dir else Path.home() / "Documents" / "AlchemAI" / "dossiers"
    base.mkdir(parents=True, exist_ok=True)
    slug = _slug(subject)
    sec_path, full_path, md_path = (str(base / f"{slug}_secciones.svg"), str(base / f"{slug}_completo.svg"),
                                    str(base / f"{slug}.md"))
    if refinery:
        r, summary, opps = _refinery_dossier(refinery, region, prices, plant_data, period)
        pm = r["process_model"]
    elif power_plant:
        r, summary, opps = _power_dossier(power_plant, year, prices)
        pm = r["process_model"]
    else:
        r, summary, opps, extra = _plant_dossier(plant, region, prices, plant_data)
        pm = extra["pm"]
    d1 = TD.process_diagram(process_model=pm, opportunities=opps, view="sections", theme=theme, path=sec_path)
    d2 = TD.process_diagram(process_model=pm, opportunities=opps, view="full", theme=theme, path=full_path)
    if refinery:
        md = _md_refinery(r, summary, opps, sec_path, full_path)
    elif power_plant:
        md = _md_power(r, summary, opps, sec_path, full_path)
    else:
        from packages.costing import plant as PL
        md = PL.plant_report(extra["eco"], subject)["markdown"]
        md = (f"Diagramas: [vista por secciones]({Path(sec_path).name}) · [vista completa]({Path(full_path).name})\n\n" + md
              + "\n## Oportunidades (numeradas en el diagrama)\n\n"
              + "\n".join(f"{i}. **{o['title_es']}** — " + (f"USD {_fmt(o.get('value_usd_y'))}/año" if o.get("value_usd_y")
                                                             is not None else "sin valor: faltan datos")
                          for i, o in enumerate(opps, 1)) + "\n")
    Path(md_path).write_text(md, encoding="utf-8")
    ok = bool(d1["evidence"]["verified"] and d2["evidence"]["verified"] and summary.get("evidence_verified", True) is not False)
    return {"summary": summary, "opportunities": opps, "report_path": md_path, "diagram_sections_path": sec_path,
            "diagram_full_path": full_path, "markdown_preview": md[:4000],
            "warnings": (d1.get("warnings") or []) + (d2.get("warnings") or []),
            "method": "Encadena motores validados por separado (refinería/planta, economía, escáner, diagrama); no agrega física.",
            "evidence": {"verified": ok, "diagram_sections": d1["evidence"], "diagram_full": d2["evidence"]}}
