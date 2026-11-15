"""Client-readable report of the steam diagnostic (bloque M19), in Spanish Markdown.

The text is assembled deterministically from the result of `steam_system_diagnostic`: no LLM call, no number that the
engine did not produce. Only unit conversions for display happen here (K → °C, Pa → bar and kg/cm², fraction → %, kg/s →
t/h). The report is always built from a fresh run of the engine on the inputs; a diagnostic dict passed in is only accepted
if it matches that run, so a result edited on its way here cannot become a report.
"""
from __future__ import annotations

from packages.energy import steam_diagnostic as SD

BASIS_ES = {"HHV": "PCS", "LHV": "PCI"}
LOSS_ES = {
    "dry_gas": ("Chimenea: gases secos calientes", "el aire que sobra y los gases salen calientes y se llevan calor"),
    "water_latent": ("Chimenea: vapor de agua que se forma al quemar (calor latente)",
                     "el hidrógeno del combustible se vuelve vapor y ese vapor no se condensa en la caldera"),
    "water_from_hydrogen_sensible": ("Chimenea: ese mismo vapor de agua, por salir caliente", "calor sensible del vapor formado"),
    "fuel_moisture_sensible": ("Chimenea: humedad del combustible", "agua que ya traía el combustible"),
    "air_moisture_sensible": ("Chimenea: humedad del aire", "agua que trae el aire de combustión"),
    "unburned_CO": ("Combustión incompleta (CO)", "combustible que sale sin terminar de quemarse"),
    "setting": ("Paredes de la caldera (radiación y convección)", "calor que pierde la envolvente al cuarto de calderas"),
}
CREDIT_ES = {"fuel_sensible": "Calor que ya trae el combustible sobre 25 °C", "air_sensible": "Calor que ya trae el aire sobre 25 °C"}
GLOSSARY = [
    ("PCS y PCI", "poder calorífico superior e inferior (en inglés HHV y LHV): la energía de un combustible contando, o sin "
                  "contar, el calor del vapor de agua que se forma al quemarlo. Una eficiencia en base PCS es numéricamente "
                  "menor que la misma en base PCI; aquí se usa la base en la que se paga el combustible."),
    ("Exceso de aire", "el aire que entra de más respecto al estrictamente necesario para quemar. Se mide con el oxígeno (O2) "
                       "que queda en la chimenea. Hace falta algo; lo que sobra se calienta y se tira."),
    ("Purga", "agua que se saca de la caldera para que no se concentren las sales. Sale a la temperatura de la caldera."),
    ("Condensado", "el vapor ya usado, convertido en agua caliente y limpia. Regresarlo ahorra agua y combustible."),
    ("Agua de repuesto", "agua nueva tratada que repone la purga y el condensado que no regresa."),
    ("Punto de rocío", "temperatura bajo la cual el vapor de agua de los gases se condensa. Con azufre, el ácido condensa a "
                       "una temperatura más alta (punto de rocío ácido), que este estudio no calcula."),
    ("Retorno simple", "años en que el ahorro paga la inversión: inversión ÷ ahorro anual, sin intereses ni impuestos."),
    ("Vapor entregado y vapor generado", "el generado es todo el que sale de la caldera; el entregado es el que llega al "
                                         "proceso después de descontar el que se usa para calentar el agua de alimentación."),
]


def _n(x, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def _pct(x, d: int = 1) -> str:
    return f"{100.0 * x:.{d}f} %"


def _C(T_K) -> str:
    return f"{float(T_K) - 273.15:.0f} °C"


def _cell(x) -> str:
    return str(x).replace("|", "/").replace("\n", " ")


def _table(head: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out + [""]


def _plant_value(p: dict) -> str:
    v, unit = p["value"], p["unit"]
    if isinstance(v, dict):
        return ", ".join(f"{k} {100 * float(x):.2f} %" if isinstance(x, (int, float)) else f"{k} = {x}" for k, x in v.items())
    if isinstance(v, str) or v is None:
        return str(v)
    v = float(v)
    if unit == "K":
        return f"{v - 273.15:.1f} °C"
    if unit == "Pa":
        return f"{v / 1e5:.3f} bar absolutos"
    if unit.startswith("fracción"):
        return f"{100 * v:.2f} %{unit.removeprefix('fracción')}"
    if unit == "kg/s":
        return f"{v * 3.6:.2f} t/h"
    return f"{v:,.0f} {unit}" if abs(v) >= 1000 else f"{v:g} {unit}"


# internal names that reach engine texts -> what a plant manager reads
PLAIN_NAMES = [
    ("a feedwater_T_K", "a la temperatura de alimentación"), ("a flash_pressure_Pa", "a la presión del tanque de flasheo"),
    ("sale a drain_T_K", "sale a la temperatura de drenaje dada"),
    (" (va en utility_costs como blowdown_fraction)", " (se cuenta en el balance de vapor)"),
    ("; da air.dry_air_mole_fractions para otra", ""), (" (nasa_gas.yaml; agua líquida de nasa_condensed.yaml)", ""),
    ("water.feedwater_heating='not_charged': calentar", "Se eligió no cobrar el calentamiento del agua: calentar"),
    ("hasta feedwater_T_K", "hasta la temperatura de alimentación"),
    ("feedwater_heating='mixing_only': la", "Tanque sin calentador: la"),
    ("el balance del desaireador de utility_costs", "el balance del desaireador"),
]


def _plain(text: str) -> str:
    """Engine sentence -> client sentence: heating-value acronyms in Spanish and no internal parameter names."""
    for old, new in PLAIN_NAMES:
        text = text.replace(old, new)
    return _es(text)


def _es(text: str) -> str:
    """HHV/LHV → the Spanish abbreviations the glossary explains."""
    return text.replace("HHV", "PCS").replace("LHV", "PCI").replace("K de diferencia", "°C de diferencia")


def _change(c: dict) -> str:
    def one(v):
        if isinstance(v, str) or v is None:
            return str(v)
        if c.get("unit") == "K":
            return f"{float(v) - 273.15:.0f} °C"
        if c.get("unit") == "fracción":
            return f"{100 * float(v):.1f} %"
        return f"{float(v):.4g}"
    return f"{c['what']}: {one(c['from'])} → {one(c['to'])}"


def _headline(d: dict) -> dict:
    """The comparable numbers of a diagnostic, used to check that a result passed in is the engine's own."""
    b = d["baseline"]
    h = {"baseline.efficiency": b["boiler"]["efficiency"], "baseline.cost_per_tonne": b["steam"]["cost_per_tonne"]}
    if b.get("annual"):
        h["baseline.annual_cost"] = b["annual"]["cost"]["total"]
    for m in d["measures"]:
        if m["status"] == "evaluated":
            h[f"{m['id']}.cost_saving_per_tonne"] = m["savings"]["per_tonne"]["cost"]
            if m["savings"]["annual"]:
                h[f"{m['id']}.annual_saving"] = m["savings"]["annual"]["cost"]["total"]
        else:
            h[f"{m['id']}.status"] = 0.0
    c = d.get("combined")
    if c and c.get("savings"):
        h["combined.cost_saving_per_tonne"] = c["savings"]["per_tonne"]["cost"]
    return h


def steam_diagnostic_report(diagnostic: dict | None = None, inputs: dict | None = None, plant_label: str | None = None,
                            notice: str | None = None, capture_flags: list[str] | None = None,
                            captured_data: list[list] | None = None) -> dict:
    """Markdown report (Spanish) of a steam diagnostic; see the module docstring and tools_steam.steam_diagnostic_report."""
    if (diagnostic is None) == (inputs is None):
        raise ValueError("Da inputs (los mismos argumentos de steam_system_diagnostic) o diagnostic (su resultado completo); "
                         "uno de los dos.")
    if diagnostic is not None:
        if not isinstance(diagnostic, dict) or "inputs" not in diagnostic or "baseline" not in diagnostic:
            raise ValueError("diagnostic: pasa el resultado completo de steam_system_diagnostic (trae 'inputs' y 'baseline').")
        inputs = diagnostic["inputs"]
    d = SD.steam_system_diagnostic(**inputs)
    integrity = {"rebuilt_from_inputs": True, "checked_against_given_result": diagnostic is not None}
    if diagnostic is not None:
        mine, theirs = _headline(d), _headline(diagnostic)
        bad = sorted(k for k in set(mine) | set(theirs)
                     if k not in mine or k not in theirs or abs(mine[k] - theirs[k]) > 1e-9 * max(1.0, abs(mine[k])))
        if bad:
            raise ValueError("El diagnóstico recibido no coincide con el que producen sus propias entradas (difieren: "
                             f"{bad}). No se escribe un reporte con números que el motor no produjo: vuelve a correr "
                             "steam_system_diagnostic y pasa su resultado sin modificar, o pasa inputs.")

    cur, basis = d["currency"], d["fuel_basis"]
    bes = BASIS_ES[basis]
    base, boiler, steam, annual = d["baseline"], d["baseline"]["boiler"], d["baseline"]["steam"], d["baseline"]["annual"]
    evaluated = [m for m in d["measures"] if m["status"] == "evaluated"]
    skipped = [m for m in d["measures"] if m["status"] != "evaluated"]
    comb = d.get("combined")
    comb_ok = bool(comb and comb.get("savings"))
    refs = [r for r in d["provenance"]["published_references"] if r.get("kind") != "setting_loss"]
    loss = d.get("setting_loss") or {}
    loss_assumed = bool(loss.get("assumed"))
    gas = d.get("fuel_resolution") or {}
    gas_assumed = bool(gas.get("assumed"))
    flags = list((d.get("data_review") or {}).get("flags", []))
    flags += [{"id": "capture", "message": x.removeprefix("REVISAR DATO: ")} for x in (capture_flags or [])]
    em = base["emissions"]
    L: list[str] = []
    numbers: dict = {}

    L.append(f"# Diagnóstico de vapor{' — ' + plant_label if plant_label else ''}")
    L.append("")
    if notice:
        L += [f"> **{notice}**", ""]
    L += [f"Fecha del cálculo: {d['inputs']['today']}. Moneda: {cur}. Todas las cifras de este documento salen de dos motores de "
          f"cálculo comprobados y de los datos que entregó la planta; al final se dice de dónde viene cada una.", ""]

    # ------------------------------------------------------------------------------------------------- summary
    L += ["## Resumen", ""]
    part = []
    if annual:
        share = annual["cost"]["fuel"] / annual["cost"]["total"]
        part.append(f"Producir vapor le cuesta hoy a la planta **{_n(annual['cost']['total'])} {cur} al año** "
                    f"({_n(steam['cost_per_tonne'], 1)} {cur} por tonelada de vapor entregado), y {_pct(share, 0)} de ese costo es "
                    f"combustible.")
        numbers["annual_cost"] = annual["cost"]["total"]
    else:
        share = steam["breakdown_per_tonne"]["fuel"] / steam["cost_per_tonne"]
        part.append(f"Producir vapor le cuesta hoy a la planta **{_n(steam['cost_per_tonne'], 1)} {cur} por tonelada de vapor "
                    f"entregado**, y {_pct(share, 0)} de ese costo es combustible. No se dieron las horas ni la producción, así "
                    f"que no hay cifras anuales.")
    numbers["cost_per_tonne"] = steam["cost_per_tonne"]
    numbers["efficiency"] = boiler["efficiency"]
    part.append(f"La caldera convierte en vapor **{_pct(boiler['efficiency'])}** de la energía del combustible que se paga "
                f"(base {bes}); el resto se pierde, sobre todo por la chimenea.")
    if comb_ok:
        s = comb["savings"]
        if s["annual"]:
            part.append(f"Las {len(evaluated)} medidas evaluadas, aplicadas juntas, ahorrarían **{_n(s['annual']['cost']['total'])} "
                        f"{cur} al año** ({_pct(s['cost_fraction'])} del costo del vapor), que son "
                        f"{_n(s['annual']['fuel_GJ'])} GJ de combustible al año.")
            numbers["combined_annual_saving"] = s["annual"]["cost"]["total"]
        else:
            part.append(f"Las {len(evaluated)} medidas evaluadas, aplicadas juntas, bajarían el costo del vapor en "
                        f"**{_n(s['per_tonne']['cost'], 1)} {cur} por tonelada** ({_pct(s['cost_fraction'])}).")
        numbers["combined_saving_per_tonne"] = s["per_tonne"]["cost"]
    elif len(evaluated) == 1:
        s = evaluated[0]["savings"]
        if s["annual"]:
            part.append(f"La única medida evaluada ({evaluated[0]['label'].lower()}) ahorraría **{_n(s['annual']['cost']['total'])} "
                        f"{cur} al año** ({_pct(s['cost_fraction'])} del costo del vapor).")
            numbers["single_measure_annual_saving"] = s["annual"]["cost"]["total"]
        else:
            part.append(f"La única medida evaluada ({evaluated[0]['label'].lower()}) bajaría el costo en "
                        f"**{_n(s['per_tonne']['cost'], 1)} {cur} por tonelada** ({_pct(s['cost_fraction'])}).")
    else:
        part.append("**No se evaluó ninguna medida de ahorro**: faltan los objetivos de la planta (ver «Lo que no se evaluó»).")
    L += [" ".join(part), ""]
    if refs:
        L += ["> **Aviso: al menos un precio es una REFERENCIA PUBLICADA, no el contrato de la planta.** "
              + " ".join(f"{r['item']}: {r['detail']}." for r in refs)
              + " Las cifras en dinero cambian cuando se sustituya por el precio real.", ""]
    if loss_assumed:
        sv = loss["sensitivity"]
        L += ["> **Aviso: la pérdida por paredes de la caldera es un SUPUESTO tomado de una publicación, no un dato de la "
              f"planta.** La planta declaró que no la conoce. Se usó {_pct(loss['value'])} de la energía del combustible "
              f"(base {'PCS' if loss['basis'] == 'HHV' else 'PCI'}), de {loss['source']}. Si la pérdida real fuera un punto "
              f"porcentual mayor, la eficiencia bajaría {_n(-sv['efficiency_change_points'], 2)} puntos y el vapor costaría "
              f"{_pct(sv['cost_change_fraction'], 2)} más. La eficiencia y el costo de este reporte dejan de ser supuestos "
              "cuando la planta entregue ese dato (hoja del fabricante o prueba).", ""]
    if not d["prices"].get("water_price_given", True):
        L += ["> **Aviso: el costo del agua no está incluido.** La planta no dio lo que paga por el agua de repuesto. El costo "
              "del vapor y el ahorro de las medidas que ahorran agua (bajar la purga, regresar condensado) quedan por debajo "
              "de lo real.", ""]
    if gas_assumed:
        worst = max(abs(x["efficiency_change_points"]) for x in gas["sensitivity"])
        L += ["> **Aviso: la composición del gas es un SUPUESTO, no un análisis de la planta.** La planta declaró que no la "
              f"tiene. Se usó {_es(gas['description'])}. Con las otras composiciones probadas la eficiencia cambia como mucho "
              f"{_n(worst, 2)} puntos. " + (_es(gas["surrogate_validation"]) if gas.get("surrogate_validation") else
                                            "Con el poder calorífico del gas (o la energía y el volumen del recibo) la "
                                            "composición se ajusta a la planta."), ""]
    if flags:
        L += ["**Datos que conviene confirmar antes de decidir** (la revisión no corrige ni descarta ningún dato):", ""]
        L += [f"- {x['message']}" for x in flags] + [""]
    if not d["evidence"]["verified"]:
        L += [f"> **ATENCIÓN: las comprobaciones internas NO cerraron** ({'; '.join(d['evidence']['reasons'])}). No uses estas "
              f"cifras sin revisar.", ""]

    # ------------------------------------------------------------------------------------------------ baseline
    L += ["## 1. Cómo opera hoy la caldera", ""]
    rows = [["Presión de operación", f"{steam['pressure_Pa'] / 1e5:.2f} bar absolutos "
                                     f"({steam['pressure_gauge_Pa'] / SD.KGF_CM2:.1f} kg/cm² manométricos); el vapor sale a "
                                     f"{_C(steam['steam_T_K'])}", SD.PLANT],
            ["Agua de alimentación que entra a la caldera", _C(steam["feedwater_T_K"]),
             SD.CALC if d["inputs"]["water"]["feedwater_heating"] == "mixing_only" else SD.PLANT],
            ["Oxígeno en la chimenea (base seca) y exceso de aire",
             f"{_pct(boiler['flue_O2_dry_mole_fraction'])} de O2; {_pct(boiler['excess_air_fraction'], 0)} de aire de más",
             (SD.PLANT + " / " + SD.CALC) if (d["inputs"].get("combustion") or {}).get("flue_O2") else (SD.CALC + " / " + SD.PLANT)],
            ["Temperatura de chimenea", _C(boiler["stack_T_K"]), SD.PLANT],
            [f"Eficiencia de la caldera (base {bes})", f"**{_pct(boiler['efficiency'])}**", SD.CALC],
            [f"Combustible por tonelada de vapor (base {bes})", f"{_n(steam['fuel_GJ_per_tonne'], 3)} GJ", SD.CALC],
            ["Agua de repuesto por tonelada de vapor", f"{_n(steam['makeup_m3_per_tonne'], 3)} m³", SD.CALC],
            ["Costo del vapor", f"**{_n(steam['cost_per_tonne'], 1)} {cur} por tonelada** "
                                f"({_n(steam['cost_per_GJ_delivered'], 1)} {cur} por GJ de calor entregado)", SD.CALC]]
    br = steam["breakdown_per_tonne"]
    rows.append(["  de ese costo: combustible / agua / electricidad",
                 f"{_n(br['fuel'], 1)} / {_n(br['water'], 1)} / "
                 + (f"{_n(br['power'], 1)}" if steam["power_kWh_per_tonne"] is not None else "no evaluada") + f" {cur} por tonelada",
                 SD.CALC])
    if annual:
        h = d["inputs"]["operation"]["hours_per_year"]
        rows += [["Horas de operación", f"{_n(h)} h al año", SD.PLANT],
                 ["Vapor generado por la caldera", f"{_n(annual['steam_generated_tonne'])} t al año "
                                                   f"({_n(annual['steam_generated_tonne'] / h, 2)} t/h en promedio)",
                  SD.PLANT if base["annual_basis"] == "steam" else SD.CALC],
                 ["Vapor entregado al proceso", f"{_n(annual['steam_delivered_tonne'])} t al año", SD.CALC],
                 [f"Combustible al año (base {bes})", f"{_n(annual['fuel_GJ'])} GJ"
                  + (f" (≈ {_n(annual['fuel_Nm3'])} m³ normales a 0 °C y 1 atm)" if annual["fuel_Nm3"] else ""),
                  SD.PLANT if base["annual_basis"] == "fuel" else SD.CALC],
                 ["Agua de repuesto al año", f"{_n(annual['makeup_water_m3'])} m³", SD.CALC],
                 ["Electricidad de la caldera al año", f"{_n(annual['electricity_kWh'])} kWh" if annual["electricity_kWh"] is not None
                  else "no evaluada: falta el consumo eléctrico por tonelada", SD.CALC],
                 ["**Costo anual del vapor**", f"**{_n(annual['cost']['total'])} {cur}** (combustible {_n(annual['cost']['fuel'])}; "
                                                f"agua {_n(annual['cost']['water'])}; electricidad {_n(annual['cost']['power'])})", SD.CALC]]
    rows.append(["Emisiones directas de CO2 de la caldera",
                 (f"{_n(em['co2_tonne_per_year'])} toneladas al año; " if em["co2_tonne_per_year"] is not None else "")
                 + f"{_n(em['co2_kg_per_tonne_steam'], 1)} kg por tonelada de vapor", SD.CALC])
    L += _table(["Concepto", "Valor", "Origen"], rows)
    if base.get("derived_note"):
        L += [base["derived_note"], ""]
    if base.get("load_fraction") is not None:
        L += [f"La caldera trabaja en promedio a {_pct(base['load_fraction'], 0)} de su capacidad nominal.", ""]
    if (base.get("site_pressure") or {}).get("from_altitude_m") is not None:
        sp = base["site_pressure"]
        L += [f"La presión atmosférica del sitio ({_n(sp['Pa'] / 1e3, 1)} kPa) se calculó de la altitud "
              f"({_n(sp['from_altitude_m'])} m); con ella se pasaron a absolutas las presiones de manómetro.", ""]
    L += ["«Por tonelada» siempre es por tonelada de vapor entregado al proceso. La caldera genera algo más, porque una parte "
          "del vapor se usa para calentar el agua de alimentación.", ""]
    L += [f"Precios usados: combustible {_n(d['prices']['fuel_per_GJ'], 2)} {cur}/GJ ({bes}); agua "
          + (f"{_n(d['prices']['boiler_makeup_water_per_m3'], 2)} {cur}/m³" if d["prices"].get("water_price_given", True)
             else "no se dio su precio (el costo del vapor NO incluye el agua)")
          + (f"; electricidad {_n(d['prices']['electricity_per_kWh'], 3)} {cur}/kWh" if d["prices"]["electricity_per_kWh"] is not None else "")
          + ".", ""]

    # ---------------------------------------------------------------------------------------------- energy split
    L += ["## 2. A dónde se va la energía del combustible", "",
          f"De cada 100 unidades de energía del combustible (base {bes}):", ""]
    rows = [["Se va al vapor que sale de la caldera", _n(100 * boiler["useful_steam_heat_fraction_of_fuel"], 1), "es lo que se aprovecha"],
            ["Se va con la purga", _n(100 * boiler["blowdown_heat_fraction_of_fuel"], 1), "agua caliente que se tira para controlar las sales"]]
    losses = boiler["losses_fraction_of_heating_value"]
    for k, v in [(k, losses[k]) for k in LOSS_ES if k in losses]:
        if abs(v) > 5e-5:
            note = " **Valor supuesto de una publicación, no medido en la planta**" if k == "setting" and loss_assumed else ""
            rows.append([LOSS_ES[k][0], _n(100 * v, 1), LOSS_ES[k][1] + ("." + note if note else "")])
    for k, v in boiler["credits_fraction_of_heating_value"].items():
        if abs(v) > 5e-5:
            rows.append([CREDIT_ES[k], _n(-100 * v, 1), "entra con el aire o el combustible; se resta de las pérdidas"])
    L += _table(["Destino", "Unidades de cada 100", "Qué es"], rows)
    L += [f"El vapor y la purga suman la eficiencia de la caldera ({_pct(boiler['efficiency'])}). "
          + (f"El punto de rocío del agua en los gases es {_C(boiler['water_dew_point_K'])}: la chimenea no debe bajar de ahí."
             if boiler["water_dew_point_K"] else ""), ""]

    # ------------------------------------------------------------------------------------------------- measures
    L += ["## 3. Medidas de ahorro", ""]
    if evaluated:
        has_annual = evaluated[0]["savings"]["annual"] is not None
        order = sorted(evaluated, key=lambda m: -(m["savings"]["annual"]["cost"]["total"] if has_annual else m["savings"]["per_tonne"]["cost"]))
        rows = []
        for m in order:
            s = m["savings"]
            change = "; ".join(_change(c) for c in m["changed"])
            pay = (f"{m['simple_payback_years']:.1f} años" if m["simple_payback_years"] is not None
                   else "sin cotización" if m["investment"] is None else (m["payback_note"] or "no calculado"))
            inv = f"{_n(m['investment'])} {cur}" if m["investment"] is not None else "no dada"
            if has_annual:
                cap = s["annual"].get("max_investment_for_payback")
                rows.append([m["label"], change, _n(s["annual"]["fuel_GJ"]), _n(s["annual"]["makeup_water_m3"]),
                             _n(s["annual"]["co2_tonne"]), f"**{_n(s['annual']['cost']['total'])}**", _pct(s["cost_fraction"]),
                             inv, pay, f"{_n(cap['2'])} {cur}" if cap else "no ahorra"])
            else:
                rows.append([m["label"], change, _n(s["per_tonne"]["fuel_GJ"], 4), _n(s["per_tonne"]["makeup_m3"], 4),
                             _n(s["per_tonne"]["co2_kg"], 2), f"**{_n(s['per_tonne']['cost'], 2)}**", _pct(s["cost_fraction"]),
                             inv, pay, "sin cifras anuales"])
        unit = "al año" if has_annual else "por tonelada de vapor"
        L += [f"Ordenadas de mayor a menor ahorro. Cada una se calculó repitiendo el estudio completo con un solo cambio: el "
              f"objetivo que dio la planta. Ahorros {unit}; la demanda de vapor del proceso es la misma en todos los casos.", ""]
        L += _table(["Medida", "Qué cambia", f"Combustible ahorrado (GJ {bes})", "Agua ahorrada (m³)",
                     "CO2 evitado (t)" if has_annual else "CO2 evitado (kg)", f"Ahorro ({cur})",
                     "% del costo del vapor", "Inversión", "Retorno simple", "Inversión máxima para recuperar en 2 años"], rows)
        L += ["La última columna no es una cotización: es dos veces el ahorro anual, es decir, lo más que se puede pagar por la "
              "medida para recuperarlo en dos años. Sirve para saber si vale la pena pedir la cotización. El retorno simple "
              "solo se calcula con la inversión que cotice el proveedor: nunca se estima.", ""]
        for m in order:
            r = m["result"]
            L.append(f"**{m['label']}.** Eficiencia de la caldera {_pct(base['boiler']['efficiency'])} → {_pct(r['boiler']['efficiency'])}; "
                     f"costo del vapor {_n(steam['cost_per_tonne'], 1)} → {_n(r['steam']['cost_per_tonne'], 1)} {cur} por tonelada.")
            ck = m.get("checks") or {}
            if ck.get("margin_over_water_dew_point_K") is not None:
                L.append(f"- La chimenea quedaría {ck['margin_over_water_dew_point_K']:.0f} °C por encima del punto de rocío del "
                         f"agua ({_C(ck['water_dew_point_K'])}).")
            if ck.get("feedwater_T_after_economizer_K") is not None:
                L.append(f"- El agua saldría del economizador a unos {_C(ck['feedwater_T_after_economizer_K'])}.")
            rec = r["steam"].get("blowdown_heat_recovery")
            if rec:
                bits = []
                if rec["flash_pressure_Pa"] is not None:
                    bits.append(f"{_pct(rec['flash_steam_fraction_of_blowdown'])} de la purga se vuelve vapor de flasheo a "
                                f"{rec['flash_pressure_Pa'] / 1e5:.2f} bar absolutos")
                if rec["drain_T_K"] is not None:
                    bits.append(f"el resto calienta el agua de repuesto hasta {_C(rec['makeup_T_after_exchanger_K'])} y se va al "
                                f"drenaje a {_C(rec['drain_T_K'])}")
                L.append("- " + "; ".join(bits) + ".")
            L += [f"- {_plain(n)}" for n in m["notes"]]
            L.append("")
    else:
        L += ["No se evaluó ninguna medida: ver la sección siguiente.", ""]

    # ------------------------------------------------------------------------------------------------- combined
    L += ["## 4. Todas las medidas juntas", ""]
    if comb_ok:
        s, si = comb["savings"], comb["sum_of_individual"]
        if s["annual"]:
            rows = [["Suma de las medidas una por una (no es alcanzable)", f"{_n(si['annual']['cost_total'])} {cur} al año",
                     f"{_n(si['annual']['fuel_GJ'])} GJ al año"],
                    ["**Caso combinado (una sola corrida con todos los cambios)**", f"**{_n(s['annual']['cost']['total'])} {cur} al año**",
                     f"{_n(s['annual']['fuel_GJ'])} GJ al año"]]
        else:
            rows = [["Suma de las medidas una por una (no es alcanzable)", f"{_n(si['per_tonne']['cost'], 2)} {cur} por tonelada",
                     f"{_n(si['per_tonne']['fuel_GJ'], 4)} GJ por tonelada"],
                    ["**Caso combinado (una sola corrida con todos los cambios)**", f"**{_n(s['per_tonne']['cost'], 2)} {cur} por tonelada**",
                     f"{_n(s['per_tonne']['fuel_GJ'], 4)} GJ por tonelada"]]
        L += _table(["", "Ahorro en dinero", f"Ahorro de combustible ({bes})"], rows)
        r = comb["result"]
        L += [f"El caso combinado da {_pct(comb['combined_over_sum'], 0)} de la suma. {comb['explanation']}", "",
              f"Con todas las medidas la eficiencia de la caldera pasaría de {_pct(boiler['efficiency'])} a "
              f"{_pct(r['boiler']['efficiency'])} y el vapor costaría {_n(r['steam']['cost_per_tonne'], 1)} {cur} por tonelada "
              f"en lugar de {_n(steam['cost_per_tonne'], 1)}."
              + (f" Inversión total declarada {_n(comb['investment'])} {cur}: retorno simple {comb['simple_payback_years']:.1f} años."
                 if comb.get("simple_payback_years") is not None else " Retorno simple del conjunto: " + (comb.get("payback_note") or "no calculado") ), ""]
        if s["annual"]:
            cap = s["annual"].get("max_investment_for_payback")
            L += [f"Con todas las medidas se dejarían de emitir {_n(s['annual']['co2_tonne'])} toneladas de CO2 al año."
                  + (f" Para recuperar la inversión en dos años, el conjunto no debería costar más de {_n(cap['2'])} {cur} "
                     f"(en tres años, {_n(cap['3'])} {cur})." if cap else ""), ""]
        L += [f"- {n}" for n in comb.get("notes", [])] + ([""] if comb.get("notes") else [])
    elif comb and comb.get("reason"):
        L += [comb["reason"].capitalize() + ".", ""]
    else:
        L += ["No aplica: no se evaluó ninguna medida.", ""]

    # ---------------------------------------------------------------------------------------------- not evaluated
    dist = d.get("distribution")
    if dist:
        L += ["## 4b. Pérdidas en la distribución (fugas, trampas y tubería sin aislar)", "",
              "Salen del recorrido de la planta: los conteos, diámetros y longitudes son datos de la planta. Cada pérdida se "
              f"valora con el vapor de la línea base ({_n(dist['valuation']['cost_per_tonne_steam'], 1)} {cur} por tonelada).", ""]
        yearly = dist["hours_per_year"] is not None
        rows = []
        for key, label in (("leaks", "Fuga de vapor"), ("failed_traps", "Trampa fallada abierta")):
            for it in (dist[key] or {}).get("items", []):
                rows.append([f"{label}: {it['description']}", f"{it['count']:.0f} × orificio de {it['orifice_diameter_m'] * 1e3:.1f} mm a "
                             f"{it['steam_pressure_Pa'] / 1e5:.1f} bar abs", f"{_n(it['steam_kg_h'], 1)} kg/h de vapor",
                             _n(it["cost_per_year"]) if yearly else "sin horas", _n(it["co2_tonne_per_year"], 1) if yearly else "—"])
        for it in (dist["bare_pipes"] or {}).get("items", []):
            rows.append([f"Tubería sin aislar: {it['description']}", f"{it['length_m']:.0f} m de {it['outer_diameter_m'] * 1e3:.0f} mm "
                         f"de diámetro exterior a {_C(it['surface_T_K'])}", f"{_n(it['heat_loss_W'] / 1e3, 1)} kW de calor "
                         f"({_n(it['heat_loss_W_per_m'])} W por metro)", _n(it["cost_per_year"]) if yearly else "sin horas",
                         _n(it["co2_tonne_per_year"], 1) if yearly else "—"])
        L += _table(["Pérdida", "Qué es", "Cuánto se pierde", f"Costo al año ({cur})", "CO2 al año (t)"], rows)
        tot = dist["total"]
        if yearly:
            cap = tot["max_investment_for_payback"]
            L += [f"**En total estas pérdidas cuestan {_n(tot['cost_per_year'])} {cur} al año** y emiten "
                  f"{_n(tot['co2_tonne_per_year'], 1)} toneladas de CO2."
                  + (f" Equivalen a {_pct(tot['share_of_delivered_steam'])} del vapor que entrega la caldera."
                     if tot["share_of_delivered_steam"] is not None else "")
                  + (f" Repararlas todas se paga en dos años si cuesta menos de {_n(cap['2'])} {cur}." if cap else ""), ""]
        ins = [it for it in (dist["bare_pipes"] or {}).get("items", []) if it.get("insulation")]
        if ins:
            L += ["Con el aislamiento propuesto:", ""]
            L += _table(["Tramo", "Aislamiento", "Pérdida hoy → aislado", f"Ahorro al año ({cur})"],
                        [[it["description"], f"{it['insulation']['thickness_m'] * 1e3:.0f} mm; la cubierta queda a "
                          f"{_C(it['insulation']['jacket_T_K'])}", f"{_n(it['heat_loss_W'] / 1e3, 1)} → "
                          f"{_n(it['insulation']['heat_loss_W'] / 1e3, 2)} kW ({_pct(it['insulation']['saving_fraction'], 0)} menos)",
                          _n(it["insulation"]["saving_cost_per_year"]) if yearly else "sin horas"] for it in ins])
        pending = [it for it in (dist["bare_pipes"] or {}).get("items", []) if it.get("insulation_note")]
        if pending:
            L += ["Tramos sin aislamiento propuesto (se reporta solo lo que pierden hoy): "
                  + ", ".join(it["description"] for it in pending) + ".", ""]
        L += [dist["valuation"]["note"], "", "Constantes del modelo que no se le piden a la planta:", ""]
        L += [f"- {c['what'].capitalize()}: {c['value']:.2f}" + (f". Fuente: {c['source']}." if c["source"] else " (dado por el usuario).")
              for c in dist["model_constants"].values()] + [""]
        L += ["Supuestos de esta sección:", ""] + [f"- {a}" for a in dist["assumptions"][len(dist["model_constants"]):]] + [""]
        L += ["No incluye: " + "; ".join(dist["not_included"]) + ".", ""]

    L += ["## 5. Lo que no se evaluó y por qué", ""]
    if skipped:
        for m in skipped:
            L.append(f"- **{m['label']}**: {_plain(m['reason'])}." + (f" Hace falta: {m['needs']}." if m["needs"] else ""))
        L.append("")
    other = [x for x in d["missing_data"] if not x["prevents"].startswith("evaluar la medida")]
    if other:
        L += ["Datos que faltaron y qué impide cada uno:", ""]
        L += [f"- {x['item']}: impide {x['prevents']}." for x in other] + [""]
    if not skipped and not other:
        L += ["Nada: se dieron todos los datos que el diagnóstico sabe usar.", ""]

    # ------------------------------------------------------------------------------------------------ provenance
    L += ["## 6. De dónde sale cada número", "",
          "Tres orígenes posibles: **dato de la planta** (lo entregó la planta; no se comprobó en sitio), **calculado** (lo "
          "produjo una herramienta a partir de esos datos) y **referencia publicada** (un precio o un valor de una fuente "
          "pública con fecha, usado solo porque se pidió o porque la planta declaró que no tiene el dato).", "",
          "Datos de la planta:", ""]
    L += _table(["Dato", "Valor"], [[_es(p["item"]), _es(_plant_value(p))] for p in d["provenance"]["plant_data"]])
    if d.get("assumed_values"):
        L += ["Valores supuestos (no son datos de la planta):", ""]
        L += _table(["Dato", "Valor", "Origen", "Detalle"],
                    [[v["item"], _es(v["value"]), v["origin"], _es(v["detail"])] for v in d["assumed_values"]])
    L += ["Precios:", ""]
    L += _table(["Precio", "Valor", "Origen", "Detalle"],
                [[p["item"], _plant_value(p) if isinstance(p["value"], dict) or p["unit"].startswith("fracción")
                  else _es(f"{_n(p['value'], 4)} {p['unit']}"), p["origin"], p["detail"]] for p in d["provenance"]["prices"]]
                + [[p["item"], ", ".join(f"{k} {_n(x, 4)}" for k, x in p["value"].items()) + f" {p['unit']}"
                    if isinstance(p["value"], dict) else f"{_n(p['value'], 4)} {p['unit']}", p["origin"], p["detail"]] for p in refs])
    L += ["Cifras calculadas:", ""]
    L += _table(["Cifra", "Herramienta", "A partir de"], [[c["figure"], c["tool"], c["from"]] for c in d["provenance"]["calculated"]])

    # ------------------------------------------------------------------------------------------ assumptions, limits
    L += ["## 7. Supuestos y límites", "",
          "Esta sección es técnica: está escrita para quien revise el cálculo.", "",
          "Supuestos del cálculo:", ""]
    L += [f"- {_plain(a)}" for a in d["assumptions"]] + ["", "No está incluido en estas cifras:", ""]
    L += [f"- {_plain(a)}" for a in d["not_included"]] + [""]
    if d["warnings"]:
        L += ["Avisos del cálculo:", ""] + [f"- {_plain(w)}" for w in d["warnings"]] + [""]

    # -------------------------------------------------------------------------------------------------- evidence
    ev = d["evidence"]
    L += ["## 8. Comprobaciones", "",
          f"- Balances de masa y energía de los dos motores, en la línea base y en cada medida: "
          f"{'cierran' if ev['verified'] else 'NO cierran'} (peor residuo {ev['worst_residual_fraction']:.1e} de la energía del combustible).",
          f"- Calor que recibe el vapor (tablas de vapor IAPWS-95) contra eficiencia × combustible: diferencia "
          f"{ev['steam_side_duty_vs_efficiency_times_fuel']:.1e}."]
    if "annual_fuel_two_routes_difference_fraction" in ev:
        L.append(f"- Combustible anual por dos rutas (lado del agua y lado de la combustión): diferencia "
                 f"{ev['annual_fuel_two_routes_difference_fraction']:.1e}.")
    if "combined_not_above_sum" in ev:
        L.append("- El caso combinado no ahorra más que la suma de las medidas: "
                 + ("se cumple." if ev["combined_not_above_sum"]["holds"] else "NO se cumple; revisar."))
    if "input_output_cross_check" in ev:
        x = ev["input_output_cross_check"]
        L.append(f"- Se dieron producción de vapor y consumo de combustible: la eficiencia por entradas y salidas es "
                 f"{_pct(x['input_output_efficiency'])} y la del método de pérdidas {_pct(x['heat_loss_efficiency'])} "
                 f"(diferencia {x['difference_points']:+.1f} puntos). {x['meaning']}")
    if "feed_tank_mixing_check" in ev:
        x = ev["feed_tank_mixing_check"]
        L.append(f"- Temperatura del agua de alimentación calculada por mezcla: {_C(x['feedwater_T_K_calculated'])}"
                 + (f"; medida: {_C(x['feedwater_T_K_measured'])} (diferencia {x['measured_minus_calculated_K']:+.1f} °C). {x['meaning']}"
                    if x["feedwater_T_K_measured"] is not None else "."))
    L += ["- " + ev["meaning"].replace("verified = ", "«Comprobado» significa que "), ""]
    if captured_data:
        L += ["## 9. Datos tal como los capturó la planta", "",
              "Copia de la hoja de datos, en las unidades en que se escribió. El resto del reporte usa unidades métricas.", ""]
        L += _table(["Dato", "Valor", "Unidad"], [[str(a), str(b), str(c)] for a, b, c in captured_data])

    L += ["## Glosario", ""] + [f"- **{k}**: {v}" for k, v in GLOSSARY] + [""]
    md = "\n".join(L)
    return {"markdown": md, "headline": numbers, "currency": cur, "fuel_basis": basis,
            "sections": [x[3:] for x in L if x.startswith("## ")], "integrity": integrity,
            "evaluated_measures": [m["id"] for m in evaluated], "not_evaluated_measures": [m["id"] for m in skipped],
            "uses_published_reference_price": bool(refs), "uses_assumed_setting_loss": loss_assumed, "uses_assumed_fuel_composition": gas_assumed,
            "data_flags": len(flags), "capture_flags": list(capture_flags or []), "verified": ev["verified"], "diagnostic": d}

