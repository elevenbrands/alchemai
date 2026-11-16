"""Agent tools for whole-plant economics (bloque M26_W). No DWSIM flowsheet needed.
Engine: `packages.costing.plant` (orchestrates capital_investment, cash_flow_analysis and the reference-price bank).

`plant_economics` turns an equipment list plus material/utility flows into capital, operating cost, cash flow, NPV, IRR and
minimum selling price, proposing every missing ECONOMIC input from a sourced reference (flagged); `plant_economics_update`
replaces assumptions with the user's real data; `plant_report` writes the Spanish Markdown report; and
`hazop_from_process_description` runs the repo's rule-based preliminary HAZOP on a plant described by hand."""
from __future__ import annotations

import math

from packages.costing import plant as PL

# unit `type` → kind understood by packages.hazop.rules (DWSIM ObjectType vocabulary); anything else gets generic flow rows only
HAZOP_KINDS = ("Compressor", "Pump", "Cooler", "Heater", "Vessel", "Separator", "Tank", "Valve", "HeatExchanger", "RCT_Conversion",
               "RCT_Equilibrium", "RCT_Gibbs", "RCT_CSTR", "RCT_PFR", "RCT_Batch", "RCT_Generic", "ShortcutColumn",
               "DistillationColumn", "AbsorptionColumn", "Mixer", "Splitter", "Pipe")
HAZOP_ALIASES = {
    "bomba": "Pump", "compresor": "Compressor", "enfriador": "Cooler", "calentador": "Heater", "recipiente": "Vessel",
    "separador": "Separator", "decantador": "Separator", "flash": "Separator", "tanque": "Tank", "valvula": "Valve",
    "intercambiador": "HeatExchanger", "reactor": "RCT_Generic", "cstr": "RCT_CSTR", "pfr": "RCT_PFR", "reactor batch": "RCT_Batch",
    "fermentador": "RCT_Batch", "columna": "DistillationColumn", "destilacion": "DistillationColumn",
    "absorbedor": "AbsorptionColumn", "mezclador": "Mixer", "divisor": "Splitter", "tuberia": "Pipe"}


def plant_economics(equipment: list[dict], raw_materials: list[dict], products: list[dict], utilities: list[dict] | None = None,
                    labour: dict | None = None, capital_factors: dict | None = None, basis: dict | None = None,
                    prices: dict | None = None, region: str = "MX", currency: str = "USD", plant_type: str | None = None,
                    main_product: str | None = None, year: int = 2023, cepci: float | None = None,
                    exchange_rates: dict | None = None, sensitivity_span: float = 0.2, today: str | None = None) -> dict:
    """Primer cuadro económico COMPLETO de una planta a partir de su lista de equipos y sus flujos: inversión, costo de operación,
    flujo de caja, VPN, TIR, recuperación y precio mínimo de venta, en UNA llamada. Cada dato ECONÓMICO que no des se propone
    de una referencia publicada y fechada y vuelve marcado «VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN»; lo que no está en
    el banco NO se inventa (va a `missing`). Los datos físicos (flujos, densidades) siempre son tuyos. Todo se calcula en USD.

    equipment: [{name, type (clave de bare_module_factors) o F_BM, purchase_cost_usd, base_purchase_cost_usd, basis}] tal como lo
      devuelven equipment_cost y las herramientas de dimensionamiento. Opcionales por equipo: cost_year o cepci (se reescala a
      year/cepci), isbl (False en servicios, almacenamiento, caldera, tratamiento de aguas; por omisión True), labour_class
      ('fluid' | 'solids' | 'none') para contar operadores.
    raw_materials / products / utilities: [{name, flow, unit, price?, price_unit?, reference_item?, density_kg_m3?}] o
      [{name, annual_cost, currency?}]. unit: kg/h, t/h, lb/h, m3/h, L/h, gal/h, kWh/h o kW, GJ/h, MMBtu/h, o por año (t/y, gal/y);
      'ton' se rechaza por ambiguo ('t' métrica o 'short_ton'). price_unit: 'USD/t', 'MXN/kWh', 'USD/m3', 'USD/gal'… Los residuos
      con costo de disposición van en utilities. Orden del precio: price del elemento → prices[name] → banco (reference_item si el
      nombre difiere: 'castor oil', 'methanol', 'biodiesel', 'glycerol', 'natural gas', 'process water'…) → faltante. Si flujo y
      precio están uno en masa y otro en volumen hace falta density_kg_m3 (tuya; si no, error).
    labour: {positions: [{name, count, annual_salary}], currency} o {total_salaries_per_year, currency} o {operators,
      annual_salary_per_operator} o {fluid_steps, solids_steps}; burden_fraction opcional. Sin labour: operadores por la
      correlación de Alkhayat-Gerrard sobre los equipos y salario promedio del IMSS (solo México), ambos marcados.
    capital_factors: fracciones de capital_investment (site_preparation, service_facilities, allocated_costs, contingency_and_fee,
      land, royalties, startup, working_capital). Las que no des salen del conjunto de Humbird et al. 2011 (NREL): 9 % + 4.5 % +
      4 % del ISBL, 60 % de indirectos sobre el costo directo, capital de trabajo 5 % del capital fijo.
    basis: on_stream_hours, plant_life_years, income_tax_rate, discount_rate, depreciation ('MACRS7', 'SL10'…),
      construction_schedule, startup_years, startup_revenue_fraction, startup_variable_cost_fraction,
      startup_fixed_cost_fraction, location_factor, land_usd, maintenance_fraction_of_isbl, insurance_tax_fraction_of_fci. Lo que
      falte sale del banco (convenciones de NREL; ISR de México 30 % SIN PTU; depreciación de la LISR según plant_type).
    prices: {name: {price, price_unit}}. region: 'MX-SIN' | 'MX' | 'US' | 'WORLD'. currency: moneda del resumen adicional
      (headline_in_currency). exchange_rates: {'MXN': <USD por 1 MXN>} (si no, FIX de Banxico del banco, marcado).
    year / cepci: base de costos a la que se llevan los equipos que traen cost_year o cepci. sensitivity_span: ± para la
      sensibilidad (0.2 = ±20 %; se usa el rango low-high del banco cuando existe). today: fecha ISO para la antigüedad.

    Devuelve capital (todos los niveles de capital_investment) y capital_summary; operating_costs {breakdown por rubro en USD/año y
    por unidad de producto principal, items, variable/fixed/total, complete}; revenue; cash_flow (tabla anual); npv_usd, irr,
    payback_years; minimum_selling_price {value, unit, npv_usd_at_minimum_price, is_upper_bound}; ptu (cotas, México);
    sensitivity {rows ordenadas por impacto, confirm_first}; assumed_inputs [{input, label, value, unit, source, url, period,
    date_read, age_days, stale, flag, note}]; user_inputs; missing [{input, reason, blocks}]; inputs_resolved {call,
    frozen_call}; warnings, assumptions, not_included, method y evidence {verified, complete, residuos}. Si falta el precio de un
    producto se devuelven costos y precio mínimo (VPN y TIR None); si falta el de una materia prima o servicio, la inversión y
    los costos parciales (VPN, TIR y precio mínimo None). Presenta SIEMPRE los supuestos como supuestos."""
    return PL.plant_economics(equipment, raw_materials, products, utilities, labour, capital_factors, basis, prices, region, currency,
                              plant_type, main_product, year, cepci, exchange_rates, sensitivity_span, today)


def plant_economics_update(inputs_resolved: dict, overrides: dict) -> dict:
    """Actualiza un plant_economics cuando el usuario trae datos reales. inputs_resolved: el campo inputs_resolved del resultado
    anterior, tal cual. overrides: los datos nuevos con las mismas claves de plant_economics, por ejemplo
    {'prices': {'methanol': {'price': 520, 'price_unit': 'USD/t'}}, 'basis': {'discount_rate': 0.12},
     'capital_factors': {'contingency_and_fee': 0.35}, 'labour': {'operators': 16, 'annual_salary_per_operator': 15000},
     'equipment': {'R-101': {'purchase_cost_usd': 410000}}, 'raw_materials': {'aceite': {'flow': 1300}}}.
    Los dict se mezclan clave por clave; equipos y corrientes se corrigen por nombre. Un dato dado aquí deja de ser supuesto
    (también sirve para CONFIRMAR un supuesto: pásalo con su mismo valor).

    Devuelve exactamente plant_economics sobre la llamada mezclada, más changes {rows: [{input, before, before_origin
    ('supuesto' | 'usuario' | 'faltante'), after, effect_alone {total_capital_investment_usd, operating_cost_usd_y, npv_usd, irr,
    minimum_selling_price}}], before, after, effect_together, n_assumed_before, n_assumed_after, assumed_remaining}."""
    return PL.plant_economics_update(inputs_resolved, overrides)


def plant_report(result: dict, title: str, path: str | None = None) -> dict:
    """Reporte en Markdown (español) de un resultado de plant_economics o plant_economics_update (o de otro resultado con las
    mismas claves; las secciones ausentes se omiten): resultado, inversión por nivel y por equipo, costo de operación por rubro
    y por corriente, flujo de caja, sensibilidad («qué confirmar primero»), PTU, la tabla separada «Supuestos por confirmar»
    con fuente, URL, periodo y fecha de lectura de cada valor supuesto, las entradas faltantes, advertencias y lo no incluido.
    title: título del reporte. path: si se da, escribe ahí el archivo .md (la carpeta debe existir). Devuelve {markdown, path,
    n_lines, sections}."""
    return PL.plant_report(result, title, path)


def _hazop_kind(kind: str) -> tuple[str, bool]:
    k = str(kind).strip()
    for known in HAZOP_KINDS:
        if k.lower() == known.lower():
            return known, True
    alias = HAZOP_ALIASES.get(PL._norm(k))
    return (alias, True) if alias else (k, False)


def hazop_from_process_description(units: list[dict], streams: list[dict]) -> dict:
    """HAZOP preliminar por reglas de una planta que NO es un flowsheet de DWSIM, descrita a mano. Arma el mismo grafo de proceso
    que usa hazop_preliminary y le aplica las mismas reglas deterministas (palabras guía × parámetro por tipo de equipo, con los
    números del caso). No sustituye un HAZOP con equipo multidisciplinario.

    units: [{name, type, inlets: [corrientes], outlets: [corrientes], duty_kW? (negativo = calor retirado; en reactores decide si
      es exotérmico)}]. type en el vocabulario de las reglas: Pump, Compressor, Heater, Cooler, HeatExchanger, Valve, Vessel,
      Separator, Tank, Mixer, Splitter, Pipe, DistillationColumn, AbsorptionColumn, ShortcutColumn, RCT_CSTR, RCT_PFR,
      RCT_Batch, RCT_Conversion, RCT_Equilibrium, RCT_Gibbs, RCT_Generic (se aceptan alias en español: bomba, reactor, tanque,
      columna, intercambiador, fermentador…). Un tipo sin reglas propias (centrífuga, filtro, prensa, molino) recibe solo las
      filas genéricas de flujo y se informa en `missing`.
    streams: [{name, T_K, P_Pa, composition {compuesto: fracción molar}, vapor_fraction?, molar_flow_mol_s?, mass_flow_kg_s?}].
      T_K, P_Pa y composition son obligatorios (son tu proceso). Los peligros salen de la composición con nombres en inglés
      como en DWSIM ('Methanol', 'Ethanol', 'N-hexane', 'Hydrogen', 'Ammonia', 'Water'…): un compuesto con otro nombre no
      activa banderas de inflamable o tóxico y se informa.

    Devuelve n_rows, rows [{node, kind, parameter, deviation, causes, consequences, safeguards, severity 1-5, evidence, source}],
    markdown, graph_units, missing (datos que faltaron y sus consecuencias), unrecognized_components, warnings, assumptions,
    not_included, method y evidence."""
    from packages.hazop.graph import FLAMMABLE, HYDRATE_FORMERS, TOXIC, UNSTABLE, ProcessGraph, UnitNode, melting_point_K, stream_node
    from packages.hazop.report import to_markdown
    from packages.hazop.rules import hazop_rows
    if not units or not streams:
        raise ValueError("Da units [{name, type, inlets, outlets}] y streams [{name, T_K, P_Pa, composition}]: sin ellos no hay nodos "
                         "que analizar.")
    missing: list[str] = []
    nodes, unknown = {}, set()
    for i, s in enumerate(streams):
        name = str(s.get("name") or "")
        lack = [k for k in ("T_K", "P_Pa", "composition") if s.get(k) in (None, {}, "")]
        if not name or lack:
            raise ValueError(f"streams[{i}] «{name}»: faltan {['name'] * (not name) + lack}. La temperatura, la presión y la composición "
                             "de cada corriente son de TU proceso y deciden las desviaciones y la severidad; no se suponen.")
        if name in nodes:
            raise ValueError(f"streams: el nombre «{name}» está repetido.")
        comp = {str(c): float(x) for c, x in s["composition"].items()}
        total = sum(comp.values())
        if total <= 0 or min(comp.values()) < 0 or float(s["T_K"]) <= 0 or float(s["P_Pa"]) <= 0:
            raise ValueError(f"stream «{name}»: T_K y P_Pa deben ser positivos y la composición no negativa con suma > 0.")
        if abs(total - 1.0) > 1e-6:
            missing.append(f"{name}: la composición suma {total:.4f}; se normalizó a 1")
            comp = {c: x / total for c, x in comp.items()}
        vf = s.get("vapor_fraction")
        if vf is None:
            missing.append(f"{name}: sin vapor_fraction (las reglas que dependen de que sea líquido no se activan)")
        for key in ("molar_flow_mol_s", "mass_flow_kg_s"):
            if s.get(key) is None:
                missing.append(f"{name}: sin {key} (solo afecta el texto de evidencia)")
        unknown |= {c for c in comp if c not in FLAMMABLE | TOXIC | HYDRATE_FORMERS | UNSTABLE | {"Water"} and melting_point_K(c) is None}
        nodes[name] = stream_node(name, float(s["T_K"]), float(s["P_Pa"]), float("nan") if vf is None else float(vf),
                                  float(s.get("molar_flow_mol_s") or 0.0), float(s.get("mass_flow_kg_s") or 0.0), comp)
    graph_units, edges = {}, []
    for i, u in enumerate(units):
        name = str(u.get("name") or "")
        if not name or not u.get("type"):
            raise ValueError(f"units[{i}] «{name}»: faltan name o type.")
        if name in graph_units or name in nodes:
            raise ValueError(f"units: el nombre «{name}» está repetido (o es el de una corriente).")
        inl, outl = [str(x) for x in u.get("inlets") or []], [str(x) for x in u.get("outlets") or []]
        absent = [x for x in inl + outl if x not in nodes]
        if absent:
            raise ValueError(f"unit «{name}»: las corrientes {absent} no están en streams.")
        if not inl and not outl:
            raise ValueError(f"unit «{name}»: da inlets y/o outlets; sin corrientes no hay estado con que evaluar el nodo.")
        kind, known = _hazop_kind(u["type"])
        if not known:
            missing.append(f"{name}: el tipo «{u['type']}» no tiene reglas propias (solo filas genéricas de flujo)")
        duty = u.get("duty_kW")
        if duty is None and kind.startswith("RCT"):
            missing.append(f"{name}: sin duty_kW (no se evalúa si la reacción es exotérmica)")
        tin, tout = [nodes[x].T_K for x in inl], [nodes[x].T_K for x in outl]
        pin, pout = [nodes[x].P_Pa for x in inl], [nodes[x].P_Pa for x in outl]
        graph_units[name] = UnitNode(name=name, kind=kind, inlets=inl, outlets=outl, energy_in=[], energy_out=[],
                                     duty_kW=None if duty is None else float(duty), T_in_K=max(tin) if tin else None,
                                     T_out_K=max(tout) if tout else None, P_in_Pa=max(pin) if pin else None,
                                     P_out_Pa=max(pout) if pout else None,
                                     exothermic=bool(kind.startswith("RCT") and duty is not None and float(duty) < 0))
        edges += [(x, name) for x in inl] + [(name, x) for x in outl]
    graph = ProcessGraph(name="descripción", streams=nodes, units=graph_units, edges=edges)
    rows = hazop_rows(graph)
    used = {x for u in graph_units.values() for x in u.inlets + u.outlets}
    orphan = sorted(set(nodes) - used)
    warnings = [f"Corrientes que no entran ni salen de ninguna unidad: {orphan}."] if orphan else []
    if unknown:
        warnings.append(f"Compuestos no reconocidos (sin banderas de inflamable/tóxico ni punto de fusión): {sorted(unknown)}. Usa "
                        "los nombres en inglés de DWSIM o revisa a mano sus peligros.")
    by_node = {n: sum(r.node == n for r in rows) for n in graph_units}
    reasons = [f"{n}: ninguna regla produjo filas" for n, k in by_node.items() if k == 0]
    severities_ok = all(1 <= r.severity <= 5 for r in rows)
    if not severities_ok:
        reasons.append("hay severidades fuera de 1-5")
    clean = lambda v: None if isinstance(v, float) and not math.isfinite(v) else v
    return {
        "n_rows": len(rows), "rows": [r.to_dict() for r in rows], "markdown": to_markdown(rows, "HAZOP preliminar (descripción)"),
        "graph_units": {k: {"kind": v.kind, "inlets": v.inlets, "outlets": v.outlets, "duty_kW": v.duty_kW, "exothermic": v.exothermic,
                            "n_rows": by_node[k]} for k, v in graph_units.items()},
        "stream_flags": {k: {"flammable": v.flammable, "toxic": v.toxic, "has_water": v.has_water, "hydrate_risk": v.hydrate_risk,
                             "solidifies": v.solidifies, "unstable": v.unstable, "vapor_fraction": clean(v.vapor_fraction)}
                         for k, v in nodes.items()},
        "missing": missing, "unrecognized_components": sorted(unknown), "warnings": warnings,
        "assumptions": ["T y P de entrada y salida de cada unidad = máximo de sus corrientes (misma convención que el lector de "
                        "DWSIM).", "Un reactor es exotérmico solo si duty_kW < 0 (calor retirado).",
                        "Las banderas de peligro salen de listas cortas de compuestos (inflamables, tóxicos, formadores de hidratos, "
                        "inestables) por nombre; no hay cálculo de inflamabilidad ni de toxicidad."],
        "not_included": ["Instrumentación y lazos de control (no hay P&ID): las salvaguardas son las típicas de cada regla.",
                         "Análisis de riesgo (frecuencia, LOPA, SIL): la severidad 1-5 es un orden preliminar.",
                         "Sólidos, polvos combustibles, electricidad estática y peligros mecánicos de equipo sin reglas.",
                         "Revisión por un equipo multidisciplinario, que este listado no sustituye."],
        "method": "Grafo de proceso armado a mano (packages.hazop.graph: stream_node, UnitNode, ProcessGraph) y las mismas reglas "
                  "deterministas de hazop_preliminary (packages.hazop.rules.hazop_rows), sin DWSIM ni LLM.",
        "evidence": {"verified": not reasons, "reasons": reasons, "every_unit_has_rows": all(k > 0 for k in by_node.values()),
                     "severities_in_range": severities_ok, "n_units": len(graph_units), "n_streams": len(nodes),
                     "all_stream_references_resolved": True,
                     "scope": "certifica que el grafo se armó completo y que las reglas corrieron sobre cada unidad; no certifica "
                              "que el listado de peligros esté completo"},
    }


PLANT_ECONOMICS_TOOLS = [plant_economics, plant_economics_update, plant_report, hazop_from_process_description]
