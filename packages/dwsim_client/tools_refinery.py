"""Agent tools for the refinery planning model (bloque M27_R). No DWSIM flowsheet needed.
Engine: `packages.refinery.model` (public Pemex eBDI data, OGJ 2000 crude assays, published worked-example yield vectors,
DOE 2015 energy intensities). `refinery_model` returns the ProcessModel JSON that the diagram (M27_D) draws and the
opportunity scanner (M27_O) annotates."""
from __future__ import annotations

from datetime import date

from packages.refinery import model as RM


def _today(today: str | None) -> date | None:
    if today is None:
        return None
    try:
        return date.fromisoformat(today)
    except ValueError as exc:
        raise ValueError(f"today = {today!r}: usa formato ISO AAAA-MM-DD.") from exc


def refinery_model(refinery: str = "tula", crude_slate: dict | None = None, utilization=None, prices: dict | None = None,
                   unit_overrides: dict | None = None, year: int | None = None, period: str | None = None,
                   today: str | None = None, mode: str = "as_operated") -> dict:
    """Modelo de PLANEACIÓN (rendimientos por unidad + balances lineales; NO cinética ni simulación rigurosa) de una refinería
    de Pemex con datos PÚBLICOS: proceso completo, diagrama de proceso (process_model), productos, energía, CO2 y
    rentabilidad (margen de operación; la refinería existe, no se calcula VPN de construirla). Empieza con Tula (Miguel
    Hidalgo); también 'cadereyta', 'madero', 'minatitlan', 'salamanca', 'salina_cruz', 'dos_bocas' (mismos datos de la eBDI).

    refinery: clave de list_refineries ('tula' por defecto; acepta 'miguel_hidalgo', 'olmeca').
    period: periodo publicado de carga y producción: '2026-01/2026-09' (por defecto, acumulado más reciente),
      '2026-01/2026-06', '2025' o '2024'. year=2024|2025|2026 elige el periodo de ese año si no das period.
    crude_slate: carga propia en barriles por día {'maya'|'pesado': b/d, 'istmo'|'ligero': b/d, 'olmeca'|'superligero': b/d};
      por defecto la publicada del periodo (crudos sin ensaye abierto, p. ej. «reconstituido», se excluyen y se avisa).
    utilization: número = fracción de la capacidad atmosférica (reparte con la mezcla publicada) o dict por unidad
      {'cdu','vdu','coker','visbreaker','fcc','reformer','alkylation_isomerization','hds_total': fracción máxima de su
      capacidad}. Por defecto cada unidad toma toda su carga disponible hasta su capacidad publicada. Ej.: la coquizadora de
      Tula opera parcialmente: {'coker': 0.25}.
    prices: sustituye precios sugeridos: crudos 'maya','istmo','olmeca' y productos 'gasolina','turbosina','diesel',
      'combustoleo','asfalto','glp' en USD/bbl; 'coque' y 'azufre' en USD/t; 'gas_seco' y 'energia' en USD/GJ; 'costo_fijo'
      en USD/bbl; 'fx_mxn_usd' en MXN/USD.
    unit_overrides: {'capacity_mbd': {unidad: Mbd}, 'kerosene_to_jet_fraction': 0-1 (resto a diésel),
      'butanes_to_gas_plants_fraction': 0-1, 'asphalt_bpd': b/d (por defecto la producción publicada del periodo),
      'fcc_conversion_wt_pct': 58.98-76.33 % peso (conversión del FCC = 100 − LCO − aceite decantado; rendimientos del FCC
      contra conversión de la tabla de planta piloto de KBR, Niccum 2013; sin extrapolar)}. El LCO del FCC va a diésel
      sólo dentro de la capacidad de hidrodesulfuración publicada; el exceso va a combustóleo. 'cutter_fraction': 0-1,
      fracción del destilado intermedio (diésel primario + querosina no turbosina + LCO) enviada sin tratar a combustóleo
      como diluente (Pemex documenta diluyentes en su combustóleo pero no publica cuánto).
    today: fecha ISO para la antigüedad de los precios.
    mode: 'as_operated' (por defecto: la refinería COMO OPERA en el periodo) o 'nameplate' (cada unidad a su capacidad
      publicada). En 'as_operated' lo que Pemex no publica pero se ve en su producción se RESUELVE para reproducirla, un
      parámetro por producto: utilización de la coquizadora → combustóleo, fracción de querosina a turbosina → turbosina,
      fracción de destilado a combustóleo como diluente → combustóleo SÓLO donde no lo usa ya la coquizadora (ahí el diésel
      queda como validación), fracción de butanos enviada a los complejos procesadores de gas → gas licuado (Pemex excluye la mezcla de butanos de
      su gas licuado de refinería) y, si el diésel queda libre, conversión del FCC → diésel (sólo si se alcanza dentro de la
      tabla publicada; si no, queda en el caso central 66.47 % peso y el diésel sigue como validación; ver
      fcc_conversion_identification). Esos valores salen en calibrated_parameters marcados CALIBRADOS (no validación); la
      validación son los productos no ajustados (published_comparison.validation_rows: gasolinas, diésel) y el total de
      líquidos. Un valor que des tú (utilization['coker'] o unit_overrides) no se calibra.

    Devuelve process_model (ProcessModel: units con type del vocabulario de diagram, params con capacity_bpd, feed_bpd,
    utilization_of_capacity, bandwidth_process; duties_W.energy_W; metrics.throughput_bpd; streams con flow_bpd, flow_kg_h,
    sulfur_wt_pct, cut y price; sections), product_slate {producto: bpd, kg_h, sulfur_mg_kg, vol_pct_of_crude}, unit_table,
    balances (masa global y por unidad, azufre, hidrógeno), energy (W por unidad y total, GJ/bbl), emissions (t CO2/d y
    t/mil bbl contra la meta Pemex 2026), economics {gross_margin_usd_bbl, energy_cost_usd_bbl, fixed_cost_usd_bbl,
    net_margin_usd_bbl, annual_usd, prices con fuente/URL/periodo, missing_prices}, mode, calibrated_parameters,
    published_comparison (modelo vs
    producción publicada por producto, desviación %), assumptions, warnings, not_included, method y evidence. Todo precio o
    dato sugerido es suposición marcada que el usuario puede cambiar."""
    return RM.run_model(refinery=refinery, period=period, year=year, crude_slate=crude_slate, utilization=utilization,
                        prices=prices, unit_overrides=unit_overrides, today=_today(today), mode=mode)


def list_refineries() -> dict:
    """Qué refinerías de Pemex tienen datos públicos cargados para refinery_model y qué falta. Devuelve refineries
    {clave: {name, capacity_mbd_2025 por unidad (destilación atmosférica, vacío, FCC, reductora, reformación,
    hidrodesulfuración, alquilación e isomerización, coquización), periods disponibles, crude_mbd y products_mbd del periodo
    más reciente, model_available, gaps}}, sources (cuadros de la eBDI con URL), warnings, not_included, method y evidence."""
    data = RM._load("pemex_refineries.json")
    out, warnings = {}, []
    for key, ref in sorted(data["refineries"].items()):
        extra = RM._load("tula.json") if key == "tula" else {}
        caps = dict(ref["capacity_mbd_2025"])
        gaps = [f"sin {k} en el cuadro de capacidades 2025" for k, v in caps.items() if v in (None, 0.0)]
        for k, o in extra.get("capacity_overrides_mbd", {}).items():
            caps[k] = o["value"]
            gaps = [g for g in gaps if k not in g]
            gaps.append(f"capacidad de {k} ({o['value']} Mbd) de fuente secundaria: {o['source']}; estado: {o['status']}")
        latest = ref["periods"][RM.DEFAULT_PERIOD]
        unmodeled = {t: v for t, v in latest["crude_mbd"].items() if t not in ("total", "pesado", "ligero", "superligero") and v}
        if unmodeled:
            gaps.append(f"crudos sin ensaye abierto que el modelo excluye: {unmodeled} Mbd")
        gaps += extra.get("not_in_public_capacity_table", [])
        out[key] = {"name": ref["name"], "capacity_mbd_2025": caps, "periods": sorted(ref["periods"]),
                    "latest_period": RM.DEFAULT_PERIOD, "crude_mbd": latest["crude_mbd"], "products_mbd": latest["products_mbd"],
                    "model_available": True, "gaps": gaps, "project_status": extra.get("project_status", [])}
    return {"refineries": out, "sources": data["sources"], "date_read": data["date_read"], "warnings": warnings,
            "not_included": ["Refinería Deer Park (EE. UU., filial de Pemex): no está en los cuadros de la eBDI usados.",
                             "Capacidades de plantas de hidrógeno y azufre: no publicadas en la eBDI."],
            "method": "Lectura de los cuadros publicados de la eBDI de Pemex guardados en packages/refinery/data (sin cálculo).",
            "evidence": {"verified": True, "n_refineries": len(out),
                         "reasons": [], "note": "verified = los datos cargados existen y son legibles; no certifica la fuente."}}


REFINERY_TOOLS = [refinery_model, list_refineries]
