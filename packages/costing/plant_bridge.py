"""Bridge from a whole-plant MODEL (`biodiesel_plant`, `reference_biorefinery`) to the region's economics (`plant_economics`).

Why: the plant models carry the economics of the paper they reproduce (U.S. tax, 2013 prices, the paper's discount rate),
while the owner asks for the plant in Mexico. Both routes answering the same question with different numbers is a defect,
so this module is the single economic route for a modelled plant: the model gives the PHYSICS (equipment with its
purchase cost, feeds, products, utility demands, on-stream hours) and `plant_economics` gives the ECONOMICS with the
region's sourced bank, flagged assumptions, the ranked sensitivity and the report keys.

Price order for every flow: the caller's `prices[name]` → the reference-price bank (region fallback stated) → the model's
own published price, as a SOURCED assumption (citation, URL and price year of the paper) → missing. Nothing is invented.
"""
from __future__ import annotations

from packages.costing import plant as PL
from packages.costing import reference_prices as RP

READ = "2026-11-16"
HUANG = {"source": "Huang, Long y Singh 2016, Biofuels Bioprod. Bioref. 10:299-315, Tabla 4 (precios de 2013; el precio del "
                   "modelo publicado biorefineries/BioSTEAM)", "url": "https://www.osti.gov/servlets/purl/1437267",
         "period": {"start": "2013-01-01", "end": "2013-12-31"}, "date_read": READ}
SEIDER_POWER = {"source": "BioSTEAM 2.54 PowerUtility.price, que cita Seider et al. 2017 (precio de electricidad del modelo)",
                "url": "https://github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/biosteam/_power_utility.py",
                "period": {"start": "2017-01-01", "end": "2017-12-31"}, "date_read": READ}
HUMBIRD = {"source": "Humbird et al. 2011, NREL/TP-5100-47764 (precios del modelo publicado, dólares de 2007)",
           "url": "https://docs.nlr.gov/docs/fy11osti/47764.pdf", "period": {"start": "2007-01-01", "end": "2007-12-31"},
           "date_read": READ}
KWIATKOWSKI = {"source": "Kwiatkowski et al. 2006, Ind. Crops Prod. 23:288-296 (precios del modelo publicado; solo el resumen "
                         "es abierto)", "url": "https://doi.org/10.1016/j.indcrop.2005.08.004",
               "period": {"start": "2005-01-01", "end": "2005-12-31"}, "date_read": READ}
MODEL_PRICE_SOURCE = {"biodiesel": HUANG, "sugarcane_ethanol": HUANG, "corn_ethanol": KWIATKOWSKI,
                      "cornstover_ethanol": HUMBIRD}

# stream ID of the models → item of the reference-price bank (None: the bank has no such item; the model price is used)
BANK_ITEM = {"methanol": "methanol", "biodiesel": "biodiesel", "crude_glycerol": "glycerol", "NaOH": "caustic soda",
             "ethanol": "ethanol", "HCl": None, "catalyst": None, "biodiesel_wash_water": None}
OIL_BANK_ITEM = {"castor": "castor oil", "tripalmitin": "palm oil", "triolein": None}
NAMES_ES = {"vegetable_oil": "aceite", "methanol": "metanol", "catalyst": "catalizador (metóxido de sodio en metanol)",
            "HCl": "ácido clorhídrico", "NaOH": "sosa cáustica", "biodiesel_wash_water": "agua de lavado",
            "biodiesel": "biodiésel", "crude_glycerol": "glicerina cruda", "ethanol": "etanol", "sugarcane": "caña de azúcar",
            "corn": "maíz", "cornstover": "rastrojo de maíz"}
# densities (kg/m3) needed only when the bank prices a product per volume; MEASURED values with their source
DENSITY = {"ethanol": (789.0, "etanol anhidro a 20 °C, CRC Handbook (valor de tabla; dato físico, no económico)")}
BIODIESEL_DENSITY = {"castor": (937.6, "biodiésel de ricino medido a 15 °C, ACS Omega 2026, "
                                       "https://pmc.ncbi.nlm.nih.gov/articles/PMC13347325/ (citado por biodiesel_plant)")}
FEED_ROLE = "alimentación"
MAIN_ROLE = "producto principal"


def _model_key(model: dict) -> str:
    plant = model.get("plant")
    if isinstance(plant, str) and plant in MODEL_PRICE_SOURCE:
        return plant
    if "oil_model" in model or "yields" in model:
        return "biodiesel"
    raise ValueError("model_result: pasa el resultado COMPLETO de biodiesel_plant o de reference_biorefinery (no lo reconozco).")


def _hours(model: dict) -> float:
    for d in model.get("design_basis") or []:
        p = str(d.get("parameter", ""))
        if "horas de operación" in p or "operating hours" in p:
            return float(d["value"])
    raise ValueError("model_result: no trae las horas de operación en design_basis.")


def _oil_kind(model: dict) -> str | None:
    om = model.get("oil_model") or {}
    for k in ("oil", "key", "id"):
        if isinstance(om.get(k), str):
            return om[k]
    text = str(om).lower()
    return next((k for k in OIL_BANK_ITEM if k in text), None)


def _item(stream: dict, model_key: str, oil: str | None, region: str, prices: dict, warnings: list) -> dict:
    sid = stream["id"]
    name = NAMES_ES.get(sid, sid)
    it = {"name": name, "flow": float(stream["flow_kg_h"]), "unit": "kg/h"}
    if PL._norm(name) in {PL._norm(k) for k in prices}:
        return it                                     # the caller's price wins (plant_economics reads prices[name])
    bank = OIL_BANK_ITEM.get(oil) if sid == "vegetable_oil" else BANK_ITEM.get(sid, sid)
    if bank:
        r = RP.lookup(bank, region)
        dens = BIODIESEL_DENSITY.get(oil) if sid == "biodiesel" else DENSITY.get(sid)
        per_volume = r.get("found") and any(u in str(r.get("unit", "")) for u in ("m3", "/L", "gal"))
        if r.get("found") and (not per_volume or dens):
            it["reference_item"] = bank
            if per_volume:
                it["density_kg_m3"] = dens[0]
                warnings.append(f"Densidad de {name} {dens[0]} kg/m³ ({dens[1]}) para convertir el precio del banco, que está "
                                "por volumen.")
            return it
        if per_volume:
            warnings.append(f"El banco tiene {name} por volumen y no hay una densidad medida con fuente para esta corriente: se "
                            "usa el precio por kg del modelo publicado.")
    p = float(stream.get("price_usd_kg") or 0.0)
    if p > 0.0:
        it.update(price=p, price_unit="USD/kg", price_source={**MODEL_PRICE_SOURCE[model_key], "note": (
            "precio del modelo publicado (no está en el banco de precios de la región): año del artículo, no de hoy")})
        warnings.append(f"Precio de {name}: sin referencia en el banco para {region}; se usa el del modelo publicado "
                        f"({p:.4g} USD/kg, {MODEL_PRICE_SOURCE[model_key]['period']['start'][:4]}), marcado como supuesto.")
        return it
    if sid == "vegetable_oil" and oil == "castor":
        return it                                     # bank lookup by name below will report it missing with the reason
    it["reference_item"] = bank or sid                # plant_economics will report it missing (nothing invented)
    return it


def plant_economics_from_model(model_result: dict, region: str = "MX", currency: str = "USD", prices: dict | None = None,
                               basis: dict | None = None, capital_factors: dict | None = None, labour: dict | None = None,
                               include_electricity_export: bool = False, year: int = 2023, cepci: float | None = None,
                               sensitivity_span: float = 0.2, today: str | None = None) -> dict:
    """Economics of a modelled plant in `region`: the model's physics through `plant_economics` (see module docstring).

    Returns the full `plant_economics` result plus `model_comparison` (the model's own paper-based economics next to the
    regional ones, so a difference is explained, not hidden) and the model's `published_benchmark`."""
    if not isinstance(model_result, dict) or "equipment" not in model_result or "streams" not in model_result:
        raise ValueError("model_result: pasa el resultado COMPLETO de biodiesel_plant o de reference_biorefinery.")
    key = _model_key(model_result)
    prices = dict(prices or {})
    warnings: list[str] = []
    oil = _oil_kind(model_result) if key == "biodiesel" else None
    streams = model_result["streams"]
    feeds = [s for s in streams if s.get("role") == FEED_ROLE and float(s.get("flow_kg_h") or 0) > 0
             and not (float(s.get("price_usd_kg") or 0) == 0 and s["id"] not in ("vegetable_oil",) and
                      s["id"] not in BANK_ITEM and s["id"] != model_result.get("products", {}).get("feed"))]
    prod_ids = {model_result["products"]["main_product"]} | {c["stream"] for c in model_result["products"].get("coproducts", [])}
    prods = [s for s in streams if s["id"] in prod_ids and float(s.get("flow_kg_h") or 0) > 0]
    if not prods:
        raise ValueError("model_result: no encuentro el producto principal entre sus corrientes.")
    raw = [_item(s, key, oil, region, prices, warnings) for s in feeds]
    products = [_item(s, key, oil, region, prices, warnings) for s in prods]
    main_name = NAMES_ES.get(model_result["products"]["main_product"], model_result["products"]["main_product"])

    util = model_result.get("utilities") or {}
    utilities = []
    net_w = float(util.get("net_electricity_export_W") or 0.0)
    elec = {**SEIDER_POWER, "note": "precio de electricidad del modelo; para México usa tu tarifa (CFE GDMTH por periodo)"}
    if net_w < 0:
        utilities.append({"name": "electricidad comprada", "flow": -net_w / 1000.0, "unit": "kW",
                          **({} if "electricidad comprada" in prices else
                             {"price": 0.0782, "price_unit": "USD/kWh", "price_source": elec})})
    elif net_w > 0 and include_electricity_export:
        products.append({"name": "electricidad exportada", "flow": net_w / 1000.0, "unit": "kW",
                         **({} if "electricidad exportada" in prices else
                            {"price": 0.0782, "price_unit": "USD/kWh", "price_source": elec})})
        warnings.append("La exportación eléctrica del modelo publicado sale 2-3 veces la del artículo (published_benchmark): el "
                        "ingreso por electricidad está sobrestimado.")
    elif net_w > 0:
        warnings.append(f"El modelo exporta {net_w / 1e6:.1f} MW netos de electricidad; NO se cuentan como ingreso porque el "
                        "modelo sobrestima la exportación 2-3 veces contra el artículo (include_electricity_export=True para "
                        "contarla).")
    for h in util.get("heat_utilities") or []:
        flow = float(h.get("demand_flow_kg_h") or 0.0)
        cost = float(h.get("cost_usd_y") or 0.0)
        if flow > 0 and cost > 0:
            p = cost / (flow * _hours(model_result))
            nm = f"servicio {h['agent']}"
            utilities.append({"name": nm, "flow": flow, "unit": "kg/h", **({} if nm in prices else {
                "price": p, "price_unit": "USD/kg", "price_source": {**MODEL_PRICE_SOURCE[key], "note":
                                                                     "precio del agente de servicio en el modelo publicado"}})})
        elif flow > 0:
            warnings.append(f"Servicio «{h['agent']}»: el modelo publicado no le pone precio (0 USD); queda sin costo de operación "
                            "(su capital va en la torre/caldera si el modelo la incluye).")

    b = {"on_stream_hours": _hours(model_result), **(basis or {})}
    plant_type = "biodiesel" if key == "biodiesel" else "ethanol sugar" if key == "sugarcane_ethanol" else "ethanol"
    res = PL.plant_economics(model_result["equipment"], raw, products, utilities or None, labour, capital_factors, b, prices,
                             region, currency, plant_type, main_name, year, cepci, None, sensitivity_span, today)
    res["warnings"] = warnings + res.get("warnings", [])
    me = model_result.get("economics") or {}
    res["model_comparison"] = {
        "note": ("Economía del modelo publicado (precios, impuestos y tasa del artículo, en EE. UU.) frente a la de esta corrida "
                 f"({region}, banco de referencias): la diferencia viene de los supuestos económicos, no del proceso."),
        "model": {"npv_usd": me.get("npv_usd"), "irr": me.get("irr"), "discount_rate": me.get("discount_rate"),
                  "income_tax": me.get("income_tax"), "minimum_selling_price_usd_kg": me.get("minimum_selling_price_usd_kg"),
                  "TCI_usd": (model_result.get("capital") or {}).get("TCI_usd")},
        "regional": {"npv_usd": res.get("npv_usd"), "irr": res.get("irr"),
                     "minimum_selling_price": res.get("minimum_selling_price"),
                     "TCI_usd": (res.get("capital_summary") or {}).get("total_capital_investment_usd")}}
    if model_result.get("published_benchmark"):
        res["published_benchmark"] = model_result["published_benchmark"]
    res["model_assumptions"] = model_result.get("assumptions", [])
    res["model_not_included"] = model_result.get("not_included", [])
    res["method"] = (f"Física del modelo publicado ({key}) + economía regional con plant_economics. " + str(res.get("method", "")))
    return res
