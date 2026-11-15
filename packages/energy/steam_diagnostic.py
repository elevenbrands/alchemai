"""Steam system diagnostic (bloque M19): one deterministic flow from what a plant can hand over to the study it is promised.

It orchestrates two engines and adds nothing to their physics:
  * `packages.energy.combustion.combustion_efficiency` (M18): boiler efficiency by the heat-loss method, flue gas, dew point.
  * `packages.costing.utilities.utility_costs` (M17): steam island balance with IAPWS-95 and the cost of steam.

Baseline = one call of each. Every savings measure = the same two calls with one thing changed (the target the plant gives).
The combined case = one call of each with every evaluated change applied at once. The process steam demand (steam delivered)
is held constant across cases. Nothing about the plant is assumed: a missing essential datum is refused with the list of
everything missing; a missing target, investment or operating figure degrades the result and is listed in `missing_data`.

Published checks (read on 2026-11-14, see tests/test_steam_diagnostic.py and notebooks/32_diagnostico_de_vapor.py): the worked
examples of U.S. DOE Steam Tip Sheets #3, #4, #8, #9 and #10 (January 2012) and a boiler unit test of ORNL's MEASUR suite.
"""
from __future__ import annotations

import copy
from datetime import date

from packages.costing import utilities as U
from packages.energy import combustion as CB
from packages.energy import fuel_reference as FR
from packages.energy import steam_audit as SA

TOOL_COMB = "combustion_efficiency (M18, método de pérdidas, entalpías NASA)"
TOOL_COST = "utility_costs (M17, balance de la isla de vapor con IAPWS-95)"
TOOL_DIAG = "steam_system_diagnostic (M19: resta entre dos corridas de los mismos motores)"
PLANT, CALC, REF = "dato de la planta", "calculado", "referencia publicada"
KGF_CM2 = 98066.5
HEATING_MODES = ("deaerator_steam", "not_charged", "mixing_only")

MEASURE_ORDER = ("reduce_excess_air", "reduce_stack_temperature", "reduce_blowdown", "recover_blowdown_heat",
                 "increase_condensate_return")
MEASURE_LABEL = {
    "reduce_excess_air": "Bajar el exceso de aire (afinar la combustión)",
    "reduce_stack_temperature": "Bajar la temperatura de chimenea (economizador o precalentador de aire)",
    "reduce_blowdown": "Bajar la purga",
    "recover_blowdown_heat": "Recuperar el calor de la purga",
    "increase_condensate_return": "Regresar más condensado",
}
MEASURE_KEYS = {
    "reduce_excess_air": ["target_flue_O2", "target_excess_air_fraction", "stack_T_K_after", "investment",
                          "use_published_target"],
    "reduce_stack_temperature": ["target_stack_T_K", "device", "investment", "use_published_target"],
    "reduce_blowdown": ["target_blowdown_fraction", "investment"],
    "recover_blowdown_heat": ["flash_pressure_Pa", "drain_T_K", "investment"],
    "increase_condensate_return": ["target_condensate_return_fraction", "condensate_T_K_after", "feedwater_T_K_after",
                                   "investment"],
}
MEASURE_NEEDS = {      # plain: what the client reads; parameter: what the agent passes
    "reduce_excess_air": {
        "plain": "el oxígeno (O2) o el exceso de aire que el quemador puede sostener sin producir monóxido de carbono. Lo da el "
                 "fabricante del quemador o una prueba de afinación",
        "parameter": "measures.reduce_excess_air.target_flue_O2 {'mole_fraction', 'basis'} o target_excess_air_fraction"},
    "reduce_stack_temperature": {
        "plain": "la temperatura de chimenea a la que se llegaría con un economizador o un precalentador de aire. La da el "
                 "proveedor del equipo",
        "parameter": "measures.reduce_stack_temperature.target_stack_T_K"},
    "reduce_blowdown": {
        "plain": "la purga que permite el tratamiento de agua, como porcentaje del agua de alimentación. La da el proveedor "
                 "del tratamiento según los sólidos disueltos",
        "parameter": "measures.reduce_blowdown.target_blowdown_fraction"},
    "recover_blowdown_heat": {
        "plain": "la presión del tanque de flasheo y/o la temperatura a la que saldría la purga del intercambiador. Las da el "
                 "proveedor del equipo",
        "parameter": "measures.recover_blowdown_heat.flash_pressure_Pa y/o drain_T_K"},
    "increase_condensate_return": {
        "plain": "el porcentaje del vapor que se podría regresar como condensado después de las obras. Sale de un "
                 "levantamiento de la red de condensado",
        "parameter": "measures.increase_condensate_return.target_condensate_return_fraction"},
}
HEATING_ES = {"deaerator_steam": "con vapor de la propia caldera (desaireador o tanque calentado)",
              "mixing_only": "sin calentador: mezcla de condensado y agua de repuesto",
              "not_charged": "su calentamiento no se cobra al combustible"}
COMBUSTION_KEYS = ["site_pressure_Pa", "air", "flue_O2", "excess_air_fraction", "flue_CO_dry_mole_fraction", "stack_T_K",
                   "setting_loss_fraction", "setting_loss_basis", "setting_loss_unknown"]
# Owner decision (2026-11-14): when the plant declares it does not know its setting loss, a published value is used and
# labelled as an assumption everywhere. It is never applied silently: the caller must say `setting_loss_unknown: true`.
# The value is the upper limit of the statement in the source (read 2026-11-14 from the PDF on energy.gov).
SETTING_LOSS_FLAG = "PÉRDIDA POR PAREDES SUPUESTA, NO MEDIDA"
# Targets a plant without a supplier's figure may ask for by name (`use_published_target: true`). Natural gas only: the
# sources state them for natural gas. Both PDFs read on 2026-11-14 from energy.gov.
TARGET_FLAG = "OBJETIVO PUBLICADO, NO DEL PROVEEDOR"
NATURAL_GAS_MIN_METHANE = 0.70
_DOE = "https://www.energy.gov/sites/prod/files/2014/05/f16/"
PUBLISHED_TARGETS = {
    "reduce_excess_air": {
        "id": "DOE-STEAM-TIP-4-EXCESS-AIR", "set": {"target_excess_air_fraction": 0.10}, "text": "10 % de exceso de aire",
        "statement": "en sistemas de gas natural bien diseñados es alcanzable un exceso de aire de 10 %",
        "source": "U.S. DOE, Steam Tip Sheet #4, DOE/GO-102012-3405, enero 2012", "url": _DOE + "steam4_boiler_efficiency.pdf",
        "retrieved": "2026-11-14",
        "caveat": "es lo que la publicación llama alcanzable en un sistema bien diseñado; si el quemador de la planta lo logra "
                  "sin producir monóxido de carbono lo decide una prueba de combustión"},
    "reduce_stack_temperature": {
        "id": "DOE-STEAM-TIP-3-STACK", "set": {"target_stack_T_K": (250.0 - 32.0) / 1.8 + 273.15},
        "text": "chimenea a 250 °F (121 °C)",
        "statement": "la temperatura más baja a la que se pueden enfriar los gases es 250 °F con gas natural, para evitar "
                     "condensación y corrosión",
        "source": "U.S. DOE, Steam Tip Sheet #3, DOE/GO-102012-3412, enero 2012", "url": _DOE + "steam3_recovery.pdf",
        "retrieved": "2026-11-14",
        "caveat": "es el límite inferior que da la publicación, así que el ahorro es el máximo posible con un economizador; el "
                  "proveedor del equipo dirá a qué temperatura llega de verdad"},
}
# measured by tests/test_steam_diagnostic.py::test_fitted_gas_against_the_reference_gases; change both together
FUEL_SURROGATE_NOTE = ("Comprobado contra seis gases naturales de referencia con composición completa: el gas equivalente cambia "
                       "la eficiencia menos de 0.15 puntos (base PCS) y da el CO2 entre 0.9 y 3.2 % más bajo (9 % más bajo "
                       "en un gas con 7.6 % de CO2), porque no cuenta el CO2 que el gas ya trae.")
SETTING_LOSS_REFERENCE = {
    "id": "ORNL-TM-2001-263-SHELL-LOSS", "value": 0.010, "basis": "HHV",
    "source": "G. Harrell, Steam System Survey Guide, ORNL/TM-2001/263 (U.S. DOE), mayo 2002, sección 4.2.5 «Shell Loss»",
    "statement": "la mayoría de las calderas de tubos de agua deberían perder menos de 1.0 % de la energía del combustible "
                 "por la envolvente cerca de plena carga y sin daños en el refractario o el forro (la guía calcula en base "
                 "PCS y remite a la gráfica de la ABMA)",
    "value_choice": "se toma el límite superior de esa afirmación, 1.0 %",
    "limits": ["a carga parcial la fracción real es mayor: la pérdida es casi constante en kW, así que a media carga la misma "
               "fuente implica hasta 2 %",
               "en calderas de tubos de humo la misma fuente espera una pérdida menor",
               "con refractario o aislamiento dañado la pérdida real es mayor"],
    "sensitivity_step": 0.010,
    "url": "https://www.energy.gov/sites/prod/files/2014/04/f15/steam_survey_guide.pdf", "retrieved": "2026-11-14",
}
BOILER_KEYS = ["steam_pressure_Pa", "superheat_K", "blowdown_fraction", "auxiliary_power_kWh_per_tonne",
               "rated_steam_capacity_kg_s"]
# plausibility limits of the data review (bloque M20). They flag, they never refuse; see MOCKS REVISION-DATOS-M20
CHECK_EXCESS_AIR_HIGH, CHECK_EXCESS_AIR_LOW, CHECK_BLOWDOWN_HIGH = 1.00, 0.05, 0.10
CHECK_EFFICIENCY_GAP_POINTS, CHECK_HEATING_VALUE_GAP, CHECK_LOAD_LOW = 5.0, 0.02, 0.25
PAYBACK_YEARS = (1, 2, 3)
WATER_KEYS = ["feedwater_heating", "feedwater_T_K", "makeup_water_T_K", "condensate_return_fraction", "condensate_T_K"]
OPERATION_KEYS = ["hours_per_year", "steam_generated_kg_s", "fuel_consumption", "annual_basis"]
FUEL_COST_KEYS = ["price_per_GJ", "basis", "bill"]
WATER_COST_KEYS = ["price_per_m3", "bill"]
ELECTRICITY_COST_KEYS = ["price_per_kWh", "bill", "time_of_use_per_kWh", "time_of_use_fractions", "demand_charge_per_kW_month",
                         "load_factor"]


def _known(d: dict, allowed: list[str], where: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: se esperaba un dict con claves {allowed}.")
    extra = sorted(k for k in d if k not in allowed)
    if extra:
        raise ValueError(f"{where}: claves desconocidas {extra}. Válidas: {allowed}.")


def _positive(v, name: str) -> float:
    v = float(v)
    if not v > 0:
        raise ValueError(f"{name} = {v}: debe ser positivo.")
    return v


# --------------------------------------------------------------------------------------------------- fuel quantities
def _fuel_energy_GJ(q: dict, fuel_out: dict, where: str, suffix: str = "") -> tuple[float, str, str]:
    """Energy of a fuel quantity given as energy (+ basis), gas volume (+ reference conditions) or mass → (GJ, basis, how)."""
    e, v, m = q.get("energy_GJ" + suffix), q.get("volume_m3" + suffix), q.get("mass_kg" + suffix)
    if (e is None and v is None and m is None) or (m is not None and (e is not None or v is not None)):
        raise ValueError(f"{where}: da energy_GJ{suffix} (con basis 'HHV'|'LHV'), volume_m3{suffix} (con "
                         f"volume_reference {{'T_K', 'P_Pa'}}) o mass_kg{suffix}. Energía y volumen pueden ir juntos: manda la "
                         f"energía y el volumen sirve para comprobar (o ajustar) el poder calorífico del gas.")
    if e is not None:
        if q.get("basis") not in ("HHV", "LHV"):
            raise ValueError(f"{where}: con energy_GJ{suffix} di basis 'HHV' (poder calorífico superior) o 'LHV' (inferior): es "
                             f"la base en la que tu proveedor factura la energía. No se supone.")
        return _positive(e, f"{where}.energy_GJ{suffix}"), q["basis"], "energía facturada o medida, tal cual"
    if v is not None:
        if fuel_out.get("HHV_J_mol") is None:
            raise ValueError(f"{where}: volume_m3{suffix} solo aplica a un gas dado por composición molar.")
        ref = q.get("volume_reference") or {}
        if ref.get("T_K") is None or ref.get("P_Pa") is None:
            raise ValueError(f"{where}: con volume_m3{suffix} da volume_reference {{'T_K', 'P_Pa'}}: las condiciones base a las que "
                             f"tu medidor o tu recibo expresa el volumen (vienen en el recibo o en el contrato). No se suponen.")
        n = float(ref["P_Pa"]) * _positive(v, f"{where}.volume_m3{suffix}") / (CB.R_GAS * float(ref["T_K"]))
        return (n * fuel_out["HHV_J_mol"] / 1e9, "HHV",
                f"volumen × P/(R·T) a {ref['T_K']} K y {ref['P_Pa']} Pa (gas ideal) × poder calorífico superior de la "
                f"composición dada ({fuel_out['HHV_J_mol'] / 1e3:.1f} kJ/mol, {TOOL_COMB})")
    return (_positive(m, f"{where}.mass_kg{suffix}") * fuel_out["HHV_J_kg"] / 1e9, "HHV",
            f"masa × poder calorífico superior ({fuel_out['HHV_J_kg'] / 1e6:.3f} MJ/kg, {TOOL_COMB})")


# --------------------------------------------------------------------------------------------------------- prices
def _prices(fuel_cost, water_cost, electricity_cost, fuel_out: dict, reference: str | None, currency: str):
    """Translate what the plant pays (a price or a bill) into the `prices` dict of utility_costs, with provenance."""
    prices: dict = {}
    prov: list[dict] = []
    assumptions: list[str] = []
    elec_spec: dict | None = None
    bill_note = ("precio medio del recibo: incluye todo lo que traiga el importe (transporte, distribución, cargos fijos, "
                 "impuestos)")
    if fuel_cost is not None:
        _known(fuel_cost, FUEL_COST_KEYS, "fuel_cost")
        if (fuel_cost.get("price_per_GJ") is None) == (fuel_cost.get("bill") is None):
            raise ValueError("fuel_cost: da price_per_GJ (con basis 'HHV'|'LHV') o bill {'amount', y energy_GJ + basis, "
                             "volume_m3 + volume_reference, o mass_kg; 'period'}; uno de los dos.")
        if fuel_cost.get("price_per_GJ") is not None:
            if fuel_cost.get("basis") not in ("HHV", "LHV"):
                raise ValueError("fuel_cost.basis: di si tu precio es por GJ de poder calorífico superior ('HHV') o inferior "
                                 "('LHV'). No se supone.")
            prices["fuel_per_GJ"], prices["fuel_basis"] = float(fuel_cost["price_per_GJ"]), fuel_cost["basis"]
            prov.append({"item": "Precio del combustible", "value": prices["fuel_per_GJ"],
                         "unit": f"{currency}/GJ ({prices['fuel_basis']})", "origin": PLANT, "detail": "precio dado por la planta",
                         "parameter": "fuel_cost.price_per_GJ"})
        else:
            b = fuel_cost["bill"]
            _known(b, ["amount", "energy_GJ", "basis", "volume_m3", "volume_reference", "mass_kg", "period"], "fuel_cost.bill")
            if b.get("amount") is None or not b.get("period"):
                raise ValueError("fuel_cost.bill: da amount (importe pagado) y period (texto: qué periodo cubren los recibos).")
            gj, basis, how = _fuel_energy_GJ(b, fuel_out, "fuel_cost.bill")
            prices["fuel_per_GJ"], prices["fuel_basis"] = _positive(b["amount"], "fuel_cost.bill.amount") / gj, basis
            prov.append({"item": "Precio del combustible", "value": prices["fuel_per_GJ"], "unit": f"{currency}/GJ ({basis})",
                         "origin": CALC, "detail": f"importe pagado {b['amount']:,.2f} {currency} ÷ energía {gj:,.2f} GJ "
                                                   f"({how}); recibos del periodo «{b['period']}» ({PLANT}); {bill_note}"})
            assumptions.append("El combustible se valora al precio medio de los recibos (importe ÷ energía). Si el recibo trae "
                               "cargos fijos, cada GJ ahorrado vale menos que ese promedio.")
    if water_cost is not None:
        _known(water_cost, WATER_COST_KEYS, "water_cost")
        if (water_cost.get("price_per_m3") is None) == (water_cost.get("bill") is None):
            raise ValueError("water_cost: da price_per_m3 o bill {'amount', 'volume_m3', 'period'}; uno de los dos.")
        if water_cost.get("price_per_m3") is not None:
            prices["boiler_makeup_water_per_m3"] = float(water_cost["price_per_m3"])
            prov.append({"item": "Precio del agua de repuesto", "value": prices["boiler_makeup_water_per_m3"],
                         "unit": f"{currency}/m³", "origin": PLANT, "detail": "precio dado por la planta",
                         "parameter": "water_cost.price_per_m3"})
        else:
            b = water_cost["bill"]
            _known(b, ["amount", "volume_m3", "period"], "water_cost.bill")
            if b.get("amount") is None or b.get("volume_m3") is None or not b.get("period"):
                raise ValueError("water_cost.bill: da amount, volume_m3 y period.")
            prices["boiler_makeup_water_per_m3"] = _positive(b["amount"], "water_cost.bill.amount") / _positive(
                b["volume_m3"], "water_cost.bill.volume_m3")
            prov.append({"item": "Precio del agua de repuesto", "value": prices["boiler_makeup_water_per_m3"],
                         "unit": f"{currency}/m³", "origin": CALC,
                         "detail": f"importe pagado {b['amount']:,.2f} {currency} ÷ {b['volume_m3']:,.1f} m³; recibos del "
                                   f"periodo «{b['period']}» ({PLANT}); {bill_note}"})
    if electricity_cost is not None:
        _known(electricity_cost, ELECTRICITY_COST_KEYS, "electricity_cost")
        given = [k for k in ("price_per_kWh", "bill", "time_of_use_per_kWh") if electricity_cost.get(k) is not None]
        if len(given) > 1:
            raise ValueError("electricity_cost: da solo uno de price_per_kWh, bill o time_of_use_per_kWh.")
        if not given and reference is None:
            raise ValueError("electricity_cost: da price_per_kWh, bill {'amount', 'energy_kWh', 'period'} o "
                             "time_of_use_per_kWh, o nombra una referencia publicada en reference=.")
        if given == ["price_per_kWh"]:
            prices["electricity_per_kWh"] = float(electricity_cost["price_per_kWh"])
            prov.append({"item": "Precio de la electricidad", "value": prices["electricity_per_kWh"],
                         "unit": f"{currency}/kWh", "origin": PLANT, "detail": "precio dado por la planta",
                         "parameter": "electricity_cost.price_per_kWh"})
        elif given == ["bill"]:
            b = electricity_cost["bill"]
            _known(b, ["amount", "energy_kWh", "period"], "electricity_cost.bill")
            if b.get("amount") is None or b.get("energy_kWh") is None or not b.get("period"):
                raise ValueError("electricity_cost.bill: da amount, energy_kWh y period.")
            prices["electricity_per_kWh"] = _positive(b["amount"], "electricity_cost.bill.amount") / _positive(
                b["energy_kWh"], "electricity_cost.bill.energy_kWh")
            prov.append({"item": "Precio de la electricidad", "value": prices["electricity_per_kWh"],
                         "unit": f"{currency}/kWh", "origin": CALC,
                         "detail": f"importe pagado {b['amount']:,.2f} {currency} ÷ {b['energy_kWh']:,.0f} kWh; recibos del "
                                   f"periodo «{b['period']}» ({PLANT}); precio medio: incluye demanda, cargos fijos e impuestos "
                                   f"si vienen en el importe"})
        elif given == ["time_of_use_per_kWh"]:
            prices["electricity_time_of_use_per_kWh"] = dict(electricity_cost["time_of_use_per_kWh"])
            prov.append({"item": "Precio de la electricidad por periodo horario",
                         "value": prices["electricity_time_of_use_per_kWh"], "unit": f"{currency}/kWh", "origin": PLANT,
                         "detail": "precios dados por la planta", "parameter": "electricity_cost.time_of_use_per_kWh"})
        if electricity_cost.get("demand_charge_per_kW_month") is not None:
            prices["demand_charge_per_kW_month"] = float(electricity_cost["demand_charge_per_kW_month"])
            prov.append({"item": "Cargo por demanda", "value": prices["demand_charge_per_kW_month"],
                         "unit": f"{currency}/kW-mes", "origin": PLANT, "detail": "cargo dado por la planta",
                         "parameter": "electricity_cost.demand_charge_per_kW_month"})
        elec_spec = {k: electricity_cost[k] for k in ("time_of_use_fractions", "load_factor")
                     if electricity_cost.get(k) is not None}
        for k, label, what in (("time_of_use_fractions", "Reparto de la energía eléctrica por periodo horario",
                                "perfil de consumo dado por la planta"),
                               ("load_factor", "Factor de carga eléctrico (carga media ÷ demanda facturable)",
                                "dado por la planta")):
            if k in elec_spec:
                prov.append({"item": label, "value": elec_spec[k], "unit": "fracción", "origin": PLANT, "detail": what,
                             "parameter": f"electricity_cost.{k}"})
    return prices, elec_spec, prov, assumptions


# ---------------------------------------------------------------------------------------------------------- cases
def _level(spec: dict, eta: float, feedwater_T_K: float, heating: str) -> dict:
    lv = {"name": "vapor", "pressure_Pa": spec["steam_pressure_Pa"], "superheat_K": spec["superheat_K"],
          "boiler_efficiency": eta, "feedwater_T_K": feedwater_T_K,
          "condensate_return_fraction": spec["condensate_return_fraction"], "blowdown_fraction": spec["blowdown_fraction"],
          "makeup_water_T_K": spec["makeup_water_T_K"], "feedwater_heating": heating}
    for k in ("condensate_T_K", "auxiliary_power_kWh_per_tonne", "blowdown_heat_recovery"):
        if spec.get(k) is not None:
            lv[k] = spec[k]
    return lv


def _mixing_feedwater_T(spec: dict) -> float:
    """Feed-water temperature of a feed tank with no heater: the temperature at which the deaerator balance of M17 needs no
    heating steam. Closed form of that balance; the recovered blow-down heat per kg is read from M17 itself (probe call)."""
    CP = U._cp()
    r, beta = float(spec["condensate_return_fraction"]), float(spec["blowdown_fraction"])
    h_mu = U._h_liq(float(spec["makeup_water_T_K"]))
    if r > 0 and spec.get("condensate_T_K") is None:
        raise ValueError("water.condensate_T_K: con feedwater_heating='mixing_only' y retorno de condensado hace falta la "
                         "temperatura del condensado que regresa.")
    hc = U._h_liq(float(spec["condensate_T_K"])) if r > 0 else h_mu
    R = 0.0
    rec = spec.get("blowdown_heat_recovery")
    if rec:
        P_fl = rec.get("flash_pressure_Pa")
        T_probe = (CP.PropsSI("T", "P", float(P_fl), "Q", 0, "Water") if P_fl is not None
                   else CP.PropsSI("T", "P", float(spec["steam_pressure_Pa"]), "Q", 0, "Water")) - 0.01
        lv = U.steam_level(_level(spec, 1.0, T_probe, "deaerator_steam"), 0.0, 0.0, 0.0)
        got = lv["blowdown_heat_recovery"]
        R = (got["flash_steam_heat_GJ_per_tonne"] + got["heat_to_makeup_GJ_per_tonne"]) * 1e6 / lv["mass_per_kg_delivered"]["blowdown"]
    return U._T_liq(h_mu + (1.0 - beta) * r * (hc - h_mu) + beta * R)


def _run_case(spec: dict, money: dict) -> dict:
    """One call of each engine for one operating state. Returns their raw outputs plus the few numbers the diagnostic adds."""
    comb = CB.combustion_efficiency(
        spec["fuel"], spec["air"], spec["site_pressure_Pa"], excess_air_fraction=spec.get("excess_air_fraction"),
        flue_O2=spec.get("flue_O2"), flue_CO_dry_mole_fraction=spec.get("flue_CO_dry_mole_fraction"),
        stack_T_K=spec["stack_T_K"], setting_loss_fraction=spec["setting_loss_fraction"],
        setting_loss_basis=spec["setting_loss_basis"], fuel_price_basis=money["basis"], adiabatic_flame=False)
    eta = comb["for_utility_costs"]["efficiency_to_pass"]
    heating = spec["feedwater_heating"]
    if heating == "mixing_only":
        T_fw, m17_mode = _mixing_feedwater_T(spec), "deaerator_steam"
    else:
        T_fw, m17_mode = float(spec["feedwater_T_K"]), heating
    level = _level(spec, eta, T_fw, m17_mode)
    cost = U.utility_costs(money["currency"], money["prices"], money["reference"], steam=[level],
                           electricity=money["elec_spec"] if spec.get("auxiliary_power_kWh_per_tonne") is not None else None,
                           hours_per_year=money["hours"], exchange_rates=money["exchange_rates"], today=money["today"])
    return {"combustion": comb, "costs": cost, "steam": cost["steam"][0], "level": level, "efficiency": eta,
            "feedwater_T_K": T_fw}


def _summary(case: dict, basis: str, delivered_t_y: float | None, hours: float | None) -> dict:
    """The figures of one case, in one flat place. Annual figures only when the steam demand is known."""
    c, s = case["combustion"], case["steam"]
    eff = c["efficiency"]
    m, h = s["mass_per_kg_delivered"], s["enthalpy_J_kg"]
    fuel_J = s["fuel_GJ_per_tonne"] * 1e6                                         # J of fuel (price basis) per kg delivered
    duty = m["generated"] * (h["steam"] - h["feedwater"]) + m["blowdown"] * (h["saturated_liquid"] - h["feedwater"])
    aux = case["level"].get("auxiliary_power_kWh_per_tonne")
    out = {
        "boiler": {
            "efficiency": case["efficiency"], "efficiency_basis": basis,
            "efficiency_HHV": eff["HHV"]["efficiency"], "efficiency_LHV": eff["LHV"]["efficiency"],
            "losses_fraction_of_heating_value": dict(eff[basis]["losses_fraction_of_heating_value"]),
            "credits_fraction_of_heating_value": dict(eff[basis]["credits_fraction_of_heating_value"]),
            "excess_air_fraction": c["air"]["excess_air_fraction"],
            "flue_O2_dry_mole_fraction": c["flue_gas"]["dry_mole_fractions"].get("O2", 0.0),
            "stack_T_K": eff["stack_T_K"], "water_dew_point_K": c["flue_gas"]["water_dew_point_K"],
            "blowdown_heat_fraction_of_fuel": m["blowdown"] * (h["saturated_liquid"] - h["feedwater"]) / fuel_J,
            "useful_steam_heat_fraction_of_fuel": m["generated"] * (h["steam"] - h["feedwater"]) / fuel_J,
        },
        "steam": {
            "pressure_Pa": s["pressure_Pa"], "saturation_T_K": s["saturation_T_K"], "steam_T_K": s["steam_T_K"],
            "feedwater_T_K": case["feedwater_T_K"],
            "cost_per_tonne": s["cost_per_tonne"], "cost_per_GJ_delivered": s["cost_per_GJ_delivered"],
            "breakdown_per_tonne": dict(s["breakdown_per_tonne"]),
            "breakdown_per_GJ_delivered": dict(s["breakdown_per_GJ_delivered"]),
            "heat_delivered_GJ_per_tonne": s["heat_delivered_GJ_per_tonne"],
            "fuel_GJ_per_tonne": s["fuel_GJ_per_tonne"], "makeup_m3_per_tonne": s["makeup_m3_per_tonne"],
            "power_kWh_per_tonne": float(aux) if aux is not None else None,
            "mass_per_kg_delivered": dict(m),
            "boiler_duty_GJ_per_tonne": duty / 1e6,
            **({"blowdown_heat_recovery": s["blowdown_heat_recovery"]} if "blowdown_heat_recovery" in s else {}),
        },
        "annual": None,
    }
    # direct CO2 of combustion, from the carbon the engine put in the flue gas (complete combustion unless CO was measured)
    fg = c["flue_gas"]
    co2_per_kg_fuel = fg["mol_per_kg_fuel"] * fg["wet_mole_fractions"].get("CO2", 0.0) * CB._mw("CO2")
    carbon_out = fg["mol_per_kg_fuel"] * (fg["wet_mole_fractions"].get("CO2", 0.0) + fg["wet_mole_fractions"].get("CO", 0.0))
    fuel_kg_per_tonne = s["fuel_GJ_per_tonne"] * 1e9 / c["fuel"][f"{basis}_J_kg"]
    out["emissions"] = {
        "co2_kg_per_kg_fuel": co2_per_kg_fuel, "co2_kg_per_GJ_fuel_HHV": co2_per_kg_fuel / (c["fuel"]["HHV_J_kg"] / 1e9),
        "co2_kg_per_tonne_steam": fuel_kg_per_tonne * co2_per_kg_fuel, "co2_tonne_per_year": None,
        "carbon_balance_error": abs(carbon_out / c["fuel"]["elements_mol_per_kg"]["C"] - 1.0),
        "scope": "CO2 directo de la combustión en la caldera. No incluye metano ni N2O, la electricidad comprada ni la "
                 "cadena de suministro del combustible"}
    if delivered_t_y is not None:
        out["emissions"]["co2_tonne_per_year"] = delivered_t_y * out["emissions"]["co2_kg_per_tonne_steam"] / 1000.0
        f = c["fuel"]
        fuel_GJ = delivered_t_y * s["fuel_GJ_per_tonne"]
        hv = f["HHV_J_kg"] if basis == "HHV" else f["LHV_J_kg"]
        fuel_kg = fuel_GJ * 1e9 / hv
        br = s["breakdown_per_tonne"]
        out["annual"] = {
            "steam_delivered_tonne": delivered_t_y, "steam_generated_tonne": delivered_t_y * m["generated"],
            "fuel_GJ": fuel_GJ, "fuel_basis": basis, "fuel_GJ_HHV": fuel_kg * f["HHV_J_kg"] / 1e9,
            "fuel_GJ_LHV": fuel_kg * f["LHV_J_kg"] / 1e9, "fuel_kg": fuel_kg,
            "fuel_Nm3": fuel_kg * f["HHV_J_kg"] / f["HHV_J_Nm3"] if f.get("HHV_J_Nm3") else None,
            "makeup_water_m3": delivered_t_y * s["makeup_m3_per_tonne"],
            "electricity_kWh": delivered_t_y * float(aux) if aux is not None else None,
            "cost": {"fuel": delivered_t_y * br["fuel"], "water": delivered_t_y * br["water"],
                     "power": delivered_t_y * br["power"], "total": delivered_t_y * s["cost_per_tonne"]},
            "heat_delivered_GJ": delivered_t_y * s["heat_delivered_GJ_per_tonne"],
            "average_firing_kW": fuel_GJ * 1e6 / (hours * 3600.0) if hours else None,
        }
    return out


def _savings(base: dict, new: dict) -> dict:
    """Baseline minus case. Positive = the measure saves."""
    b, n = base["steam"], new["steam"]
    out = {
        "efficiency_gain_points": 100.0 * (new["boiler"]["efficiency"] - base["boiler"]["efficiency"]),
        "per_tonne": {"cost": b["cost_per_tonne"] - n["cost_per_tonne"],
                      "fuel_GJ": b["fuel_GJ_per_tonne"] - n["fuel_GJ_per_tonne"],
                      "makeup_m3": b["makeup_m3_per_tonne"] - n["makeup_m3_per_tonne"],
                      "co2_kg": base["emissions"]["co2_kg_per_tonne_steam"] - new["emissions"]["co2_kg_per_tonne_steam"]},
        "fuel_fraction": 1.0 - n["fuel_GJ_per_tonne"] / b["fuel_GJ_per_tonne"],
        "cost_fraction": 1.0 - n["cost_per_tonne"] / b["cost_per_tonne"],
        "annual": None,
    }
    if base["annual"] is not None:
        ba, na = base["annual"], new["annual"]
        out["annual"] = {
            "fuel_GJ": ba["fuel_GJ"] - na["fuel_GJ"], "fuel_basis": ba["fuel_basis"],
            "makeup_water_m3": ba["makeup_water_m3"] - na["makeup_water_m3"],
            "electricity_kWh": (ba["electricity_kWh"] - na["electricity_kWh"]) if ba["electricity_kWh"] is not None else None,
            "cost": {k: ba["cost"][k] - na["cost"][k] for k in ("fuel", "water", "power", "total")},
            "co2_tonne": base["emissions"]["co2_tonne_per_year"] - new["emissions"]["co2_tonne_per_year"],
        }
        total = out["annual"]["cost"]["total"]
        # the most that can be invested for the measure to pay back in N years: arithmetic on the saving, not an estimate
        out["annual"]["max_investment_for_payback"] = ({str(n): n * total for n in PAYBACK_YEARS} if total > 0 else None)
    return out


def _residuals(case: dict) -> dict:
    s, c = case["steam"], case["combustion"]
    br = s["breakdown_per_tonne"]
    out = {k: abs(v) for k, v in s["energy_balance"].items() if k.endswith("residual_fraction")}
    out["combustion_mass_balance_error"] = c["evidence"]["mass_balance_error"]
    if "energy_balance_error" in c["evidence"]:
        out["combustion_energy_balance_error"] = c["evidence"]["energy_balance_error"]
    out["cost_breakdown_residual_fraction"] = abs(br["fuel"] + br["water"] + br["power"] - s["cost_per_tonne"]) / s["cost_per_tonne"]
    return out


# ------------------------------------------------------------------------------------------------------- measures
def _published_target(name: str, m: dict, spec0: dict, base: dict) -> tuple[dict | None, str | None]:
    """The published target of a measure, or why it does not apply. Never used unless the measure asks for it by name."""
    if name not in PUBLISHED_TARGETS:
        raise ValueError(f"measures.{name}.use_published_target: no hay objetivo publicado para esta medida; da el objetivo de "
                         "la planta.")
    T = PUBLISHED_TARGETS[name]
    own = [k for k in ("target_flue_O2", "target_excess_air_fraction", "target_stack_T_K") if m.get(k) is not None]
    if own:
        raise ValueError(f"measures.{name}: diste use_published_target y también {own}. Usa el objetivo de la planta o el "
                         "publicado, no los dos.")
    fuel = spec0["fuel"]
    if fuel.get("type") != "gas" or (fuel.get("composition") or {}).get("CH4", 0.0) < NATURAL_GAS_MIN_METHANE:
        return None, ("el objetivo publicado es para gas natural y este combustible no lo es; hace falta el objetivo del "
                      "proveedor")
    if name == "reduce_excess_air" and base["boiler"]["excess_air_fraction"] <= T["set"]["target_excess_air_fraction"]:
        return None, (f"la caldera ya opera con {100 * base['boiler']['excess_air_fraction']:.1f} % de exceso de aire, igual o "
                      f"mejor que el objetivo publicado ({T['text']})")
    if name == "reduce_stack_temperature" and spec0["stack_T_K"] <= T["set"]["target_stack_T_K"]:
        return None, (f"la chimenea ya está a {spec0['stack_T_K'] - 273.15:.0f} °C, igual o más fría que el objetivo publicado "
                      f"({T['text']})")
    return T, None


def _measure_changes(name: str, m: dict, spec0: dict, base_case: dict) -> tuple[dict | None, str | None, list[str], list[dict]]:
    """→ (changes to the baseline spec, reason if not evaluated, notes, what changed). No target → not evaluated."""
    notes: list[str] = []
    heating = spec0["feedwater_heating"]
    if name == "reduce_excess_air":
        o2, ea = m.get("target_flue_O2"), m.get("target_excess_air_fraction")
        if (o2 is None) == (ea is None):
            return None, "falta el dato" if o2 is None else "da target_flue_O2 o target_excess_air_fraction, no los dos", notes, []
        ch = {"flue_O2": o2, "excess_air_fraction": ea}
        changed = []                                                              # filled from the two combustion results
        if m.get("stack_T_K_after") is not None:
            ch["stack_T_K"] = float(m["stack_T_K_after"])
        else:
            notes.append("La temperatura de chimenea se dejó igual que hoy. Si al afinar el quemador también baja, el ahorro "
                         "será mayor (parámetro stack_T_K_after).")
        notes.append("El objetivo de oxígeno lo fija la planta: bajar el aire más de lo que el quemador aguanta produce "
                     "monóxido de carbono (CO) y combustible sin quemar, que este estudio no predice.")
        return ch, None, notes, changed
    if name == "reduce_stack_temperature":
        if m.get("target_stack_T_K") is None:
            return None, "falta el dato", notes, []
        if m.get("device") not in (None, "economizer", "air_preheater"):
            raise ValueError("measures.reduce_stack_temperature.device: 'economizer' o 'air_preheater'.")
        T = float(m["target_stack_T_K"])
        notes.append("El punto de rocío ÁCIDO no se modela: con azufre en el combustible la corrosión empieza a una temperatura "
                     "más alta que el punto de rocío del agua. El límite real lo fija el proveedor del equipo.")
        notes.append("La caldera y el equipo de recuperación se tratan como una sola envolvente: el agua de alimentación y el "
                     "aire se miden a la entrada de esa envolvente y la chimenea a su salida.")
        return {"stack_T_K": T}, None, notes, [{"what": "temperatura de chimenea", "from": spec0["stack_T_K"], "to": T,
                                                 "unit": "K"}]
    if name == "reduce_blowdown":
        if m.get("target_blowdown_fraction") is None:
            return None, "falta el dato", notes, []
        b = float(m["target_blowdown_fraction"])
        notes.append("No se comprueba la química del agua: la purga mínima la fijan los sólidos disueltos que tolera la caldera.")
        return {"blowdown_fraction": b}, None, notes, [{"what": "purga, del agua de alimentación",
                                                         "from": spec0["blowdown_fraction"], "to": b, "unit": "fracción"}]
    if name == "recover_blowdown_heat":
        rec = {k: float(m[k]) for k in ("flash_pressure_Pa", "drain_T_K") if m.get(k) is not None}
        if not rec:
            return None, "falta el dato", notes, []
        if heating == "not_charged":
            return None, ("el calentamiento del agua de alimentación se declaró como no cobrado al combustible "
                          "(feedwater_heating='not_charged'), así que recuperar el calor de la purga no cambia el resultado. "
                          "Hay que describir el sistema como calentado con vapor ('deaerator_steam') o como tanque sin "
                          "calentador ('mixing_only')"), notes, []
        to = " más ".join(([f"tanque de flasheo a {rec['flash_pressure_Pa'] / 1e5:.2f} bar absolutos"] if "flash_pressure_Pa" in rec else [])
                        + ([f"intercambiador que deja la purga a {rec['drain_T_K'] - 273.15:.0f} °C"] if "drain_T_K" in rec else []))
        return {"blowdown_heat_recovery": rec}, None, notes, [{"what": "calor de la purga", "from": "se tira", "to": to,
                                                                "unit": "texto"}]
    if name == "increase_condensate_return":
        if m.get("target_condensate_return_fraction") is None:
            return None, "falta el dato", notes, []
        r = float(m["target_condensate_return_fraction"])
        ch: dict = {"condensate_return_fraction": r}
        changed = [{"what": "retorno de condensado, del vapor", "from": spec0["condensate_return_fraction"], "to": r,
                    "unit": "fracción"}]
        T_after = m.get("condensate_T_K_after")
        if T_after is not None:
            ch["condensate_T_K"] = float(T_after)
            if spec0.get("condensate_T_K") is not None:
                changed.append({"what": "temperatura del condensado", "from": float(spec0["condensate_T_K"]),
                                "to": float(T_after), "unit": "K"})
        elif spec0.get("condensate_T_K") is None and heating != "not_charged":
            return None, ("falta la temperatura a la que regresaría el condensado (condensate_T_K_after): hoy no hay retorno, "
                          "así que no hay una medida"), notes, []
        if heating == "not_charged":
            if m.get("feedwater_T_K_after") is not None:
                ch["feedwater_T_K"] = float(m["feedwater_T_K_after"])
                changed.append({"what": "temperatura del agua de alimentación", "from": float(spec0["feedwater_T_K"]),
                                "to": ch["feedwater_T_K"], "unit": "K"})
            else:
                notes.append("Aquí solo se cuenta el ahorro de AGUA: el de combustible no aparece porque la temperatura del "
                             "agua de alimentación se dejó fija y su calentamiento no se cobra (feedwater_heating="
                             "'not_charged'). Para verlo hay que dar la nueva temperatura (feedwater_T_K_after) o describir "
                             "el sistema como 'deaerator_steam' o 'mixing_only'.")
        elif m.get("feedwater_T_K_after") is not None:
            raise ValueError("measures.increase_condensate_return.feedwater_T_K_after solo aplica con "
                             "water.feedwater_heating='not_charged'.")
        return ch, None, notes, changed
    raise ValueError(f"medida desconocida: {name}")


def _apply(spec0: dict, changes: dict) -> dict:
    spec = copy.deepcopy(spec0)
    for k, v in changes.items():
        spec[k] = v
    if "flue_O2" in changes or "excess_air_fraction" in changes:                  # exactly one of the two drives M18
        spec["flue_O2"], spec["excess_air_fraction"] = changes.get("flue_O2"), changes.get("excess_air_fraction")
    return spec


def _stack_checks(name: str, m: dict, spec: dict, case_before: dict, money: dict) -> tuple[str | None, list[str], dict]:
    """Dew-point and economizer feasibility checks for a case whose stack temperature was lowered."""
    notes: list[str] = []
    extra: dict = {}
    probe = _run_case({**spec, "stack_T_K": case_before["combustion"]["efficiency"]["stack_T_K"]}, money)   # same air, old stack
    dew = probe["combustion"]["flue_gas"]["water_dew_point_K"]
    T = float(spec["stack_T_K"])
    extra["water_dew_point_K"] = dew
    if dew is not None:
        extra["margin_over_water_dew_point_K"] = T - dew
        if T <= dew:
            return (f"el objetivo ({T:.1f} K) está en o bajo el punto de rocío del agua de los gases ({dew:.1f} K, {TOOL_COMB}): "
                    f"habría condensación y el balance con el agua como vapor ya no vale (no se modelan calderas de "
                    f"condensación)"), notes, extra
    fuel = spec["fuel"]
    sulphur = (fuel.get("composition", {}).get("H2S", 0) or 0) > 0 or (fuel.get("mass_fractions", {}).get("S", 0) or 0) > 0
    if sulphur:
        notes.append("ATENCIÓN: el combustible trae azufre. El punto de rocío ácido (no modelado) queda por encima del punto "
                     "de rocío del agua; no bajes la chimenea a este objetivo sin el dato del proveedor.")
    if m.get("device") == "economizer":
        T_fw = probe["feedwater_T_K"]
        if T <= T_fw:
            return (f"un economizador no puede enfriar los gases ({T:.1f} K) por debajo del agua de alimentación que entra "
                    f"({T_fw:.1f} K)"), notes, extra
        new = _run_case(spec, money)
        s = new["steam"]
        fuel_kg = s["fuel_GJ_per_tonne"] * 1e6 / (new["combustion"]["fuel"][f"{money['basis']}_J_kg"])    # kg fuel per kg delivered
        q = fuel_kg * (new["combustion"]["efficiency"]["absorbed_J_per_kg_fuel"]
                       - probe["combustion"]["efficiency"]["absorbed_J_per_kg_fuel"]) / s["mass_per_kg_delivered"]["feedwater"]
        h_out = s["enthalpy_J_kg"]["feedwater"] + q
        extra["economizer_heat_J_per_kg_feedwater"] = q
        if h_out >= s["enthalpy_J_kg"]["saturated_liquid"]:
            extra["feedwater_T_after_economizer_K"] = s["saturation_T_K"]
            notes.append("ATENCIÓN: el calor recuperado llevaría el agua de alimentación hasta la saturación (economizador con "
                         "ebullición). Revisa el objetivo con el proveedor.")
        else:
            extra["feedwater_T_after_economizer_K"] = U._T_liq(h_out)
        notes.append("La temperatura del agua a la salida del economizador supone que los gases siguen saliendo de la caldera "
                     "a la temperatura de chimenea de hoy.")
    return None, notes, extra


# ------------------------------------------------------------------------------------------------------ main entry
def steam_system_diagnostic(currency: str, fuel: dict | None = None, combustion: dict | None = None,
                            boiler: dict | None = None, water: dict | None = None, fuel_cost: dict | None = None,
                            water_cost: dict | None = None, electricity_cost: dict | None = None,
                            operation: dict | None = None, measures: dict | None = None, reference: str | None = None,
                            exchange_rates: dict | None = None, today: str | None = None,
                            distribution: dict | None = None) -> dict:
    """Steam diagnostic of one boiler; see the module docstring and tools_steam.steam_system_diagnostic."""
    if not currency or not isinstance(currency, str):
        raise ValueError("currency: di en qué moneda quieres el diagnóstico (por ejemplo 'MXN' o 'USD').")
    currency = currency.upper()
    day = (date.fromisoformat(today) if today else date.today()).isoformat()
    inputs = copy.deepcopy({"currency": currency, "fuel": fuel, "combustion": combustion, "boiler": boiler, "water": water,
                            "fuel_cost": fuel_cost, "water_cost": water_cost, "electricity_cost": electricity_cost,
                            "operation": operation, "measures": measures, "reference": reference,
                            "exchange_rates": exchange_rates, "today": day, "distribution": distribution})
    combustion, boiler, water = dict(combustion or {}), dict(boiler or {}), dict(water or {})
    operation, measures = dict(operation or {}), dict(measures or {})
    _known(combustion, COMBUSTION_KEYS, "combustion")
    _known(boiler, BOILER_KEYS, "boiler")
    _known(water, WATER_KEYS, "water")
    _known(operation, OPERATION_KEYS, "operation")
    _known(measures, list(MEASURE_ORDER), "measures")
    for k, mv in measures.items():
        _known(mv or {}, MEASURE_KEYS[k], f"measures.{k}")
    heating = water.get("feedwater_heating")
    fuel_info = None
    if isinstance(fuel, dict) and fuel.get("type") == FR.UNKNOWN_TYPE:            # declared: the plant has no gas analysis
        fuel, fuel_info = FR.resolve_natural_gas(fuel, fuel_cost)

    # ---- essentials: refuse once, naming everything that is missing and what it is for
    miss: list[str] = []
    if not fuel:
        miss.append("fuel: el combustible (gas por composición molar, o análisis elemental con poder calorífico medido), para "
                    "el poder calorífico y los gases de combustión. Si es gas natural y la planta no tiene su análisis, "
                    "decláralo con fuel = {'type': 'natural_gas_unknown', 'T_K': ...}: la composición se ajusta al poder "
                    "calorífico del recibo (energía y volumen facturados) y se marca como supuesta")
    if combustion.get("site_pressure_Pa") is None:
        miss.append("combustion.site_pressure_Pa: presión atmosférica del sitio, para el punto de rocío y la humedad del aire")
    if not combustion.get("air"):
        miss.append("combustion.air {'T_K', y 'relative_humidity' o 'humidity_ratio_kg_kg'}: temperatura y humedad del aire "
                    "de combustión, para la pérdida por chimenea")
    if (combustion.get("flue_O2") is None) == (combustion.get("excess_air_fraction") is None):
        miss.append("combustion.flue_O2 {'mole_fraction', 'basis': 'dry'|'wet'} o combustion.excess_air_fraction (uno solo): el "
                    "O2 medido en chimenea, para saber cuánto aire sobra")
    if combustion.get("stack_T_K") is None:
        miss.append("combustion.stack_T_K: temperatura de los gases en la chimenea, para la pérdida por chimenea")
    loss_unknown = combustion.get("setting_loss_unknown")
    if loss_unknown not in (None, True, False):
        raise ValueError("combustion.setting_loss_unknown: solo acepta true o false.")
    if loss_unknown and combustion.get("setting_loss_fraction") is not None:
        raise ValueError("combustion: diste setting_loss_fraction y también setting_loss_unknown=true. Si la planta conoce su "
                         "pérdida por paredes, quita setting_loss_unknown; si no la conoce, quita setting_loss_fraction.")
    if loss_unknown and combustion.get("setting_loss_basis") is not None:
        raise ValueError("combustion.setting_loss_basis no va con setting_loss_unknown=true: la base la fija la publicación "
                         f"de la que sale el valor supuesto ({SETTING_LOSS_REFERENCE['basis']}).")
    if not loss_unknown and (combustion.get("setting_loss_fraction") is None
                             or combustion.get("setting_loss_basis") not in ("HHV", "LHV")):
        miss.append("combustion.setting_loss_fraction y setting_loss_basis ('HHV'|'LHV'): pérdida por paredes y radiación de "
                    "la envolvente como fracción del poder calorífico (hoja de datos del fabricante o prueba), para la "
                    "eficiencia de la caldera. Si la planta no la conoce, decláralo con combustion.setting_loss_unknown=true: "
                    f"se usará {100.0 * SETTING_LOSS_REFERENCE['value']:.1f} % ({SETTING_LOSS_REFERENCE['basis']}) de una fuente "
                    "publicada, marcado como supuesto y no como dato de la planta")
    if boiler.get("steam_pressure_Pa") is None:
        miss.append("boiler.steam_pressure_Pa: presión ABSOLUTA de operación de la caldera, para la entalpía del vapor")
    if boiler.get("blowdown_fraction") is None:
        miss.append("boiler.blowdown_fraction: purga como fracción del agua de alimentación (0 si de verdad no hay), para el "
                    "calor y el agua que se van por la purga")
    if heating not in HEATING_MODES:
        miss.append("water.feedwater_heating: 'deaerator_steam' (desaireador o tanque calentado con vapor de la caldera), "
                    "'mixing_only' (tanque sin calentador: la temperatura sale de la mezcla de condensado y repuesto) o "
                    "'not_charged' (el calentamiento hasta feedwater_T_K no se cobra al combustible)")
    elif heating != "mixing_only" and water.get("feedwater_T_K") is None:
        miss.append("water.feedwater_T_K: temperatura del agua que entra a la caldera, para el calor que el combustible debe dar")
    if water.get("makeup_water_T_K") is None:
        miss.append("water.makeup_water_T_K: temperatura del agua de repuesto, para el balance del tanque de alimentación")
    if water.get("condensate_return_fraction") is None:
        miss.append("water.condensate_return_fraction: fracción del vapor que regresa como condensado (0 si no regresa nada), "
                    "para el agua de repuesto y el calor que se recupera")
    elif float(water["condensate_return_fraction"]) > 0 and heating in ("deaerator_steam", "mixing_only") \
            and water.get("condensate_T_K") is None:
        miss.append("water.condensate_T_K: temperatura del condensado que regresa, para el balance del tanque de alimentación")
    if fuel_cost is None and reference is None:
        miss.append("fuel_cost: lo que pagas por el combustible (price_per_GJ + basis, o bill con importe y energía o volumen "
                    "facturados), para el costo del vapor. No hay precio por defecto")
    if water_cost is None:
        miss.append("water_cost: lo que pagas por el agua de repuesto (price_per_m3, o bill con importe y volumen; incluye el "
                    "tratamiento si quieres cobrarlo), para el costo del vapor. El agua nunca viene del banco de referencias")
    if miss:
        raise ValueError("Faltan datos para el diagnóstico de vapor; no se suponen. " + " | ".join(miss) + ".")

    # average load of the boiler, only when the plant gives its rated capacity and its production
    load_fraction = None
    if boiler.get("rated_steam_capacity_kg_s") is not None and operation.get("steam_generated_kg_s") is not None:
        rated = _positive(boiler["rated_steam_capacity_kg_s"], "boiler.rated_steam_capacity_kg_s")
        load_fraction = _positive(operation["steam_generated_kg_s"], "operation.steam_generated_kg_s") / rated
        if load_fraction > 1.10:
            raise ValueError(f"operation.steam_generated_kg_s es {100 * load_fraction:.0f} % de boiler.rated_steam_capacity_kg_s: "
                             "la producción promedio no puede superar la capacidad nominal. Revisa los dos datos y sus unidades.")
    spec0 = {
        "fuel": fuel, "air": combustion["air"], "site_pressure_Pa": float(combustion["site_pressure_Pa"]),
        "flue_O2": combustion.get("flue_O2"), "excess_air_fraction": combustion.get("excess_air_fraction"),
        "flue_CO_dry_mole_fraction": combustion.get("flue_CO_dry_mole_fraction"), "stack_T_K": float(combustion["stack_T_K"]),
        "setting_loss_fraction": (SETTING_LOSS_REFERENCE["value"] / load_fraction if loss_unknown and load_fraction
                                  else SETTING_LOSS_REFERENCE["value"] if loss_unknown
                                  else float(combustion["setting_loss_fraction"])),
        "setting_loss_basis": SETTING_LOSS_REFERENCE["basis"] if loss_unknown else combustion["setting_loss_basis"],
        "steam_pressure_Pa": float(boiler["steam_pressure_Pa"]), "superheat_K": float(boiler.get("superheat_K") or 0.0),
        "blowdown_fraction": float(boiler["blowdown_fraction"]),
        "auxiliary_power_kWh_per_tonne": boiler.get("auxiliary_power_kWh_per_tonne"),
        "feedwater_heating": heating, "feedwater_T_K": water.get("feedwater_T_K"),
        "makeup_water_T_K": float(water["makeup_water_T_K"]),
        "condensate_return_fraction": float(water["condensate_return_fraction"]), "condensate_T_K": water.get("condensate_T_K"),
        "blowdown_heat_recovery": None,
    }

    # ---- prices (a first combustion call gives the heating values a volume or mass bill needs)
    fuel_out = CB.combustion_efficiency(fuel, spec0["air"], spec0["site_pressure_Pa"],
                                        excess_air_fraction=spec0["excess_air_fraction"], flue_O2=spec0["flue_O2"],
                                        flue_CO_dry_mole_fraction=spec0["flue_CO_dry_mole_fraction"],
                                        adiabatic_flame=False)["fuel"]
    prices, elec_spec, price_prov, assumptions = _prices(fuel_cost, water_cost, electricity_cost, fuel_out, reference, currency)
    resolved, _ = U.resolve_prices(currency, prices, reference, exchange_rates, date.fromisoformat(day))
    U._require(resolved, [("fuel_per_GJ", "el vapor")], reference)
    basis = resolved["fuel_per_GJ"]["fuel_basis"]
    hours = operation.get("hours_per_year")
    if hours is not None:
        hours = _positive(hours, "operation.hours_per_year")
        if hours > 8784:
            raise ValueError(f"operation.hours_per_year = {hours}: un año tiene como mucho 8 784 horas.")
    money = {"currency": currency, "prices": prices, "reference": reference, "elec_spec": elec_spec, "hours": hours,
             "exchange_rates": exchange_rates, "today": day, "basis": basis}

    warnings: list[str] = []
    missing: list[dict] = []
    not_included: list[str] = []

    # ---- baseline
    base_case = _run_case(spec0, money)
    s0, m0 = base_case["steam"], base_case["steam"]["mass_per_kg_delivered"]
    f0 = base_case["combustion"]["fuel"]

    # ---- annual basis: steam production or fuel consumption (one gives the other through the efficiency)
    delivered_t_y, derived, derived_note, annual_basis, cross = None, None, None, None, None
    steam_kg_s, fc = operation.get("steam_generated_kg_s"), operation.get("fuel_consumption")
    if (steam_kg_s is not None or fc is not None) and hours is None:
        raise ValueError("operation.hours_per_year: sin las horas de operación al año no se pueden dar cifras anuales.")
    from_steam = from_fuel = None
    bill_as_consumption = False
    if steam_kg_s is not None:
        from_steam = _positive(steam_kg_s, "operation.steam_generated_kg_s") * 3.6 * hours / m0["generated"]
    fuel_how = None
    if fc is not None:
        _known(fc, ["energy_GJ_per_year", "basis", "volume_m3_per_year", "volume_reference", "mass_kg_per_year",
                    "from_fuel_bill"], "operation.fuel_consumption")
        if fc.get("from_fuel_bill"):
            bill = (fuel_cost or {}).get("bill")
            if not bill:
                raise ValueError("operation.fuel_consumption.from_fuel_bill: necesita fuel_cost.bill.")
            gj, b_fc, fuel_how = _fuel_energy_GJ(bill, f0, "fuel_cost.bill")
            fuel_how = (f"energía de los recibos del periodo «{bill['period']}», que declaras que cubren exactamente un año de "
                        f"operación y solo esta caldera ({fuel_how})")
            bill_as_consumption = True
        else:
            gj, b_fc, fuel_how = _fuel_energy_GJ(fc, f0, "operation.fuel_consumption", "_per_year")
        gj_basis = gj * f0[f"{basis}_J_kg"] / f0[f"{b_fc}_J_kg"]
        from_fuel = gj_basis / s0["fuel_GJ_per_tonne"]
    if from_steam is not None and from_fuel is not None:
        ab = operation.get("annual_basis")
        if ab not in ("steam", "fuel"):
            raise ValueError("Diste producción de vapor Y consumo de combustible: di con cuál se calculan las cifras anuales en "
                             "operation.annual_basis ('steam' o 'fuel'). El otro se usa para comparar.")
        delivered_t_y, annual_basis = (from_steam if ab == "steam" else from_fuel), ab
        derived_note = ("Se dieron la producción de vapor y el consumo de combustible: las cifras anuales usan "
                        + ("la producción de vapor" if ab == "steam" else "el consumo de combustible")
                        + " y el otro dato sirve de comparación (ver «Comprobaciones»).")
        duty_GJ = from_steam * (m0["generated"] * (s0["enthalpy_J_kg"]["steam"] - s0["enthalpy_J_kg"]["feedwater"])
                                + m0["blowdown"] * (s0["enthalpy_J_kg"]["saturated_liquid"] - s0["enthalpy_J_kg"]["feedwater"])) / 1e6
        eta_io = duty_GJ / (from_fuel * s0["fuel_GJ_per_tonne"])
        cross = {"input_output_efficiency": eta_io, "heat_loss_efficiency": base_case["efficiency"], "basis": basis,
                 "difference_points": 100.0 * (eta_io - base_case["efficiency"]),
                 "steam_from_fuel_over_steam_measured": from_fuel / from_steam,
                 "meaning": "Eficiencia por entradas y salidas (calor al vapor medido ÷ combustible medido) contra la del método "
                            "de pérdidas. Una diferencia grande indica un medidor descalibrado, una purga distinta de la "
                            "declarada, combustible que se usa en otro equipo o vapor que no se mide."}
    elif from_steam is not None:
        delivered_t_y, annual_basis, derived = from_steam, "steam", "consumo de combustible"
        derived_note = "El consumo de combustible se calculó desde la producción de vapor y la eficiencia de la caldera."
    elif from_fuel is not None:
        delivered_t_y, annual_basis, derived = from_fuel, "fuel", "producción de vapor"
        derived_note = "La producción de vapor se calculó desde el consumo de combustible y la eficiencia de la caldera."
    else:
        missing.append({"item": "las horas de operación al año y la producción de vapor (o el consumo de combustible)",
                        "parameter": "operation.hours_per_year y operation.steam_generated_kg_s (u operation.fuel_consumption)",
                        "prevents": "las cifras anuales: combustible, agua, costo y ahorros al año, y el retorno simple. Se dan "
                                    "solo las cifras por tonelada de vapor y en porcentaje"})

    if bill_as_consumption:
        warnings.append("El consumo anual de combustible se tomó de los recibos. Si ese medidor alimenta también otros equipos, "
                        + ("la producción de vapor y los ahorros en dinero quedan sobrestimados." if annual_basis == "fuel" else
                           "la comparación con la producción de vapor (eficiencia por entradas y salidas) sale más baja de lo real."))
    base = _summary(base_case, basis, delivered_t_y, hours)
    if delivered_t_y is not None:
        firing = CB.combustion_efficiency(
            fuel, spec0["air"], spec0["site_pressure_Pa"], excess_air_fraction=spec0["excess_air_fraction"],
            flue_O2=spec0["flue_O2"], flue_CO_dry_mole_fraction=spec0["flue_CO_dry_mole_fraction"],
            stack_T_K=spec0["stack_T_K"], setting_loss_fraction=spec0["setting_loss_fraction"],
            setting_loss_basis=spec0["setting_loss_basis"],
            absorbed_duty_W=delivered_t_y * 1000.0 / (hours * 3600.0) * base["steam"]["boiler_duty_GJ_per_tonne"] * 1e6,
            adiabatic_flame=False)["firing"]
        base["firing"] = firing
    base["derived_quantity"], base["derived_note"], base["annual_basis"] = derived, derived_note, annual_basis

    # ---- setting loss: plant datum, or the published value when the plant declares it does not know it
    R = SETTING_LOSS_REFERENCE
    assert R["basis"] == "HHV"                                                    # the texts below say «base PCS»
    if loss_unknown:
        step, used = R["sensitivity_step"], spec0["setting_loss_fraction"]
        hi = _summary(_run_case(_apply(spec0, {"setting_loss_fraction": used + step}), money), basis, None, None)
        d_cost = hi["steam"]["cost_per_tonne"] - base["steam"]["cost_per_tonne"]
        if load_fraction:
            choice = (f"{R['value_choice']} a plena carga, y se divide entre la carga promedio de la caldera "
                      f"({100.0 * load_fraction:.0f} % de su capacidad nominal) porque la misma fuente dice que la pérdida casi "
                      f"no cambia en magnitud con la carga: {100.0 * used:.2f} %")
            limits = [x for x in R["limits"] if not x.startswith("a carga parcial")] + [
                "la carga usada es el promedio declarado; si la caldera pasa muchas horas a muy baja carga la pérdida real es "
                "mayor"]
        else:
            choice, limits = R["value_choice"], list(R["limits"])
        setting_loss = {
            "origin": REF, "assumed": True, "value": used, "full_load_value": R["value"], "load_fraction": load_fraction,
            "basis": R["basis"], "reference": R["id"],
            "source": R["source"], "statement": R["statement"], "value_choice": choice, "limits": limits,
            "url": R["url"], "retrieved": R["retrieved"],
            "sensitivity": {
                "setting_loss_fraction": used + step, "step_points": 100.0 * step,
                "efficiency": hi["boiler"]["efficiency"],
                "efficiency_change_points": 100.0 * (hi["boiler"]["efficiency"] - base["boiler"]["efficiency"]),
                "cost_per_tonne": hi["steam"]["cost_per_tonne"], "cost_per_tonne_change": d_cost,
                "cost_change_fraction": d_cost / base["steam"]["cost_per_tonne"],
                "meaning": "La misma línea base con un punto porcentual más de pérdida por paredes. Sirve para ver cuánto "
                           "pesa el supuesto; no es un intervalo de confianza."}}
        warnings.append(
            f"{SETTING_LOSS_FLAG}: la planta declaró que no conoce la pérdida por paredes y radiación de su caldera. Se usó "
            f"{100.0 * used:.2f} % del poder calorífico (base PCS), de {R['source']}: {R['statement']}; "
            f"{choice}. Con un punto porcentual más de pérdida la eficiencia baja "
            f"{-setting_loss['sensitivity']['efficiency_change_points']:.2f} puntos y el vapor cuesta "
            f"{100.0 * setting_loss['sensitivity']['cost_change_fraction']:+.2f} %. Consultada el {R['retrieved']}; {R['url']}.")
        missing.append({"item": "la pérdida por paredes y radiación de la caldera (hoja de datos del fabricante o prueba)",
                        "parameter": "combustion.setting_loss_fraction y setting_loss_basis",
                        "prevents": "dar la eficiencia de la caldera y el costo del vapor como medidos: llevan un valor "
                                    f"supuesto de una publicación ({100.0 * used:.2f} %, base PCS)"})
    else:
        setting_loss = {"origin": PLANT, "assumed": False, "value": spec0["setting_loss_fraction"],
                        "basis": spec0["setting_loss_basis"]}

    # ---- fuel: plant analysis, or the composition resolved for a plant that declared it has none
    fuel_resolution = {"origin": PLANT, "assumed": False}
    if fuel_info:
        alternatives = ([("metano puro", {"CH4": 1.0})] if fuel_info["method"] == "fit" else
                        [(f"gas de referencia «{g}»", FR.reference_gas(g)["composition"]) for g in FR.REFERENCE_GASES
                         if g != fuel_info["reference_gas"]])
        sens = []
        for label, comp in alternatives:
            alt = _summary(_run_case(_apply(spec0, {"fuel": {"type": "gas", "composition": comp, "T_K": fuel["T_K"]}}), money),
                           basis, None, None)
            sens.append({"gas": label,
                         "efficiency_change_points": 100.0 * (alt["boiler"]["efficiency"] - base["boiler"]["efficiency"]),
                         "cost_per_tonne_change_fraction": alt["steam"]["cost_per_tonne"] / base["steam"]["cost_per_tonne"] - 1.0,
                         "co2_per_tonne_change_fraction": alt["emissions"]["co2_kg_per_tonne_steam"]
                         / base["emissions"]["co2_kg_per_tonne_steam"] - 1.0})
        worst_eff = max(abs(x["efficiency_change_points"]) for x in sens)
        worst_cost = max(abs(x["cost_per_tonne_change_fraction"]) for x in sens)
        fuel_resolution = {
            "origin": CALC if fuel_info["method"] == "fit" else REF, "assumed": True, "method": fuel_info["method"],
            "composition": fuel_info["composition"], "description": fuel_info["description"],
            "measured_heating_value": fuel_info.get("measured"), "heating_value_check": fuel_info.get("heating_value_check"),
            "reference_gas": fuel_info.get("reference_gas"), "source": fuel_info.get("source"),
            "sensitivity": sens, "sensitivity_meaning":
                "La misma línea base con otra composición de gas y los mismos precios por GJ. Sirve para ver cuánto pesa el "
                "supuesto; no es un intervalo de confianza.",
            "surrogate_validation": FUEL_SURROGATE_NOTE if fuel_info["method"] == "fit" else None}
        tail = (FUEL_SURROGATE_NOTE if fuel_info["method"] == "fit" else
                "Pide a tu proveedor el poder calorífico del gas o da energía y volumen del recibo para ajustar la composición.")
        warnings.append(
            f"{FR.FUEL_UNKNOWN_FLAG}: la planta declaró que no tiene el análisis de su gas. Se usó {fuel_info['description']}. "
            f"Con las composiciones alternativas probadas la eficiencia cambia como mucho {worst_eff:.2f} puntos y el costo del "
            f"vapor {100.0 * worst_cost:.2f} %. {tail}")
        warnings.extend(fuel_info["warnings"])
        missing.append({"item": "el análisis de composición del gas (lo da el proveedor de gas o el transportista)",
                        "parameter": "fuel.composition",
                        "prevents": "dar los gases de combustión y el CO2 como medidos: salen de una composición supuesta ("
                                    + ("ajustada al poder calorífico del recibo" if fuel_info["method"] == "fit"
                                       else "un gas de referencia publicado") + ")"})
    hv_check = None
    bill0 = (fuel_cost or {}).get("bill") or {}
    if not fuel_info and f0.get("HHV_J_mol") and bill0.get("energy_GJ") is not None and bill0.get("volume_m3") is not None:
        ref = bill0.get("volume_reference") or {}
        if ref.get("T_K") is None or ref.get("P_Pa") is None:
            raise ValueError("fuel_cost.bill: con energy_GJ y volume_m3 juntos da volume_reference {'T_K', 'P_Pa'} (condiciones "
                             "base del volumen) para comparar el poder calorífico facturado con el de la composición.")
        billed = float(bill0["energy_GJ"]) * 1e9 / (float(ref["P_Pa"]) * float(bill0["volume_m3"]) / (CB.R_GAS * float(ref["T_K"])))
        own = f0[f"{bill0['basis']}_J_mol"]
        hv_check = {"billed_J_mol": billed, "composition_J_mol": own, "basis": bill0["basis"],
                    "difference_fraction": billed / own - 1.0,
                    "meaning": "Poder calorífico que implica el recibo (energía ÷ volumen) contra el de la composición dada."}

    # ---- measures: the same engines with one thing changed
    results: list[dict] = []
    case_objs: dict = {}
    combined_changes: dict = {}
    combined_parts: list[str] = []
    published_targets: list[tuple[str, dict]] = []
    for name in MEASURE_ORDER:
        m = dict(measures.get(name) or {})
        entry: dict = {"id": name, "label": MEASURE_LABEL[name], "status": "not_evaluated", "reason": None, "needs": None,
                       "changed": [], "notes": [], "result": None, "savings": None, "investment": m.get("investment"),
                       "simple_payback_years": None, "payback_note": None}
        target = None
        if m.pop("use_published_target", None):
            target, why = _published_target(name, m, spec0, base)
            if target is None:
                entry["reason"] = f"no evaluada: {why}"
                results.append(entry)
                continue
            m.update(target["set"])
            if name == "reduce_stack_temperature" and m.get("device") is None:
                m["device"] = "economizer"
        changes, reason, notes, changed = _measure_changes(name, m, spec0, base_case)
        if target is not None:
            entry["published_target"] = {k: target[k] for k in ("id", "text", "statement", "source", "url", "retrieved", "caveat")}
            notes = [f"El objetivo ({target['text']}) NO lo dio la planta ni un proveedor: es de {target['source']} "
                     f"({target['statement']}); {target['caveat']}."] + [n for n in notes if not n.startswith("El objetivo de oxígeno")]
            published_targets.append((name, target))
        entry["notes"], entry["changed"] = notes, changed
        if changes is None:
            entry["reason"] = f"no evaluada: {reason}"
            if reason == "falta el dato":
                entry["needs"] = MEASURE_NEEDS[name]["plain"]
                entry["needs_parameter"] = MEASURE_NEEDS[name]["parameter"]
                missing.append({"item": MEASURE_NEEDS[name]["plain"], "parameter": MEASURE_NEEDS[name]["parameter"],
                                "prevents": f"evaluar la medida «{MEASURE_LABEL[name]}»"})
            results.append(entry)
            continue
        spec = _apply(spec0, changes)
        extra: dict = {}
        if "stack_T_K" in changes and float(changes["stack_T_K"]) < spec0["stack_T_K"]:
            why, more, extra = _stack_checks(name, m, spec, base_case, money)
            entry["notes"] += more
            if why:
                entry["reason"] = f"no evaluada: {why}"
                entry["checks"] = extra
                warnings.append(f"Medida «{MEASURE_LABEL[name]}» {entry['reason']}.")
                results.append(entry)
                continue
        try:
            case = _run_case(spec, money)
        except ValueError as e:
            entry["reason"] = f"no evaluada: los motores rechazaron el objetivo ({e})"
            warnings.append(f"Medida «{MEASURE_LABEL[name]}» {entry['reason']}")
            results.append(entry)
            continue
        entry["status"] = "evaluated"
        case_objs[name] = case
        entry["result"] = _summary(case, basis, delivered_t_y, hours)
        if name == "reduce_excess_air":
            b_, n_ = base["boiler"], entry["result"]["boiler"]
            entry["changed"] = [
                {"what": "oxígeno en chimenea (base seca)", "from": b_["flue_O2_dry_mole_fraction"],
                 "to": n_["flue_O2_dry_mole_fraction"], "unit": "fracción"},
                {"what": "exceso de aire", "from": b_["excess_air_fraction"], "to": n_["excess_air_fraction"], "unit": "fracción"},
            ] + ([{"what": "temperatura de chimenea", "from": spec0["stack_T_K"], "to": spec["stack_T_K"], "unit": "K"}]
                 if spec["stack_T_K"] != spec0["stack_T_K"] else [])
            if n_["excess_air_fraction"] >= b_["excess_air_fraction"]:
                entry["notes"].append("ATENCIÓN: el objetivo no reduce el exceso de aire respecto a la operación de hoy.")
        entry["savings"] = _savings(base, entry["result"])
        entry["residuals"] = _residuals(case)
        if extra:
            entry["checks"] = extra
        if entry["savings"]["per_tonne"]["cost"] < 0:
            entry["notes"].append("ATENCIÓN: con este objetivo el vapor sale MÁS caro que hoy; hay que revisar el objetivo.")
            warnings.append(f"La medida «{MEASURE_LABEL[name]}» no ahorra con el objetivo dado (costo por tonelada "
                            f"{-entry['savings']['per_tonne']['cost']:+.4g} {currency}).")
        for k, v in changes.items():
            if k == "stack_T_K" and "stack_T_K" in combined_changes:
                v = min(float(v), float(combined_changes["stack_T_K"]))
                combined_parts.append("La temperatura de chimenea del caso combinado es la menor de los objetivos dados.")
            combined_changes[k] = v
        results.append(entry)

    # ---- payback (only with the plant's own investment figure and annual savings)
    for e in results:
        if e["status"] != "evaluated":
            continue
        inv = e["investment"]
        if inv is None:
            e["payback_note"] = "no calculado: falta la inversión (cotización del proveedor). No se estima."
            missing.append({"item": f"la inversión de «{e['label']}» (cotización del proveedor, en {currency})",
                            "parameter": f"measures.{e['id']}.investment",
                            "prevents": f"el retorno simple de «{e['label']}»"})
        elif e["savings"]["annual"] is None:
            e["payback_note"] = "no calculado: faltan las cifras anuales (horas y producción o consumo)."
        elif e["savings"]["annual"]["cost"]["total"] <= 0:
            e["payback_note"] = "no calculado: la medida no ahorra con el objetivo dado."
        else:
            e["simple_payback_years"] = _positive(inv, f"measures.{e['id']}.investment") / e["savings"]["annual"]["cost"]["total"]
            e["payback_note"] = "inversión (dato de la planta) ÷ ahorro anual (calculado). Sin impuestos, financiamiento ni mantenimiento."

    # ---- combined case: every evaluated change at once
    evaluated = [e for e in results if e["status"] == "evaluated"]
    combined = None
    if len(evaluated) >= 2:
        combined = {"measures": [e["id"] for e in evaluated], "status": "evaluated", "reason": None, "notes": sorted(set(combined_parts))}
        spec_c = _apply(spec0, combined_changes)
        why = None
        if "stack_T_K" in combined_changes and float(combined_changes["stack_T_K"]) < spec0["stack_T_K"]:
            why, more, extra = _stack_checks("combined", measures.get("reduce_stack_temperature") or {}, spec_c, base_case, money)
            combined["checks"] = extra
            combined["notes"] += [n for n in more if n.startswith("ATENCIÓN")]
        try:
            case_c = None if why else _run_case(spec_c, money)
        except ValueError as e:
            why = f"los motores rechazaron la combinación ({e})"
        if why:
            combined.update({"status": "not_evaluated", "reason": f"no evaluado: {why}"})
            warnings.append(f"Caso combinado {combined['reason']}.")
        else:
            case_objs["combined"] = case_c
            combined["result"] = _summary(case_c, basis, delivered_t_y, hours)
            combined["savings"] = _savings(base, combined["result"])
            combined["residuals"] = _residuals(case_c)
            ind_t = sum(e["savings"]["per_tonne"]["cost"] for e in evaluated)
            ind_f = sum(e["savings"]["per_tonne"]["fuel_GJ"] for e in evaluated)
            combined["sum_of_individual"] = {
                "per_tonne": {"cost": ind_t, "fuel_GJ": ind_f},
                "annual": None if delivered_t_y is None else {
                    "cost_total": sum(e["savings"]["annual"]["cost"]["total"] for e in evaluated),
                    "fuel_GJ": sum(e["savings"]["annual"]["fuel_GJ"] for e in evaluated)}}
            combined["combined_over_sum"] = combined["savings"]["per_tonne"]["cost"] / ind_t if ind_t else None
            combined["explanation"] = ("Los ahorros no se suman: cada medida deja menos que ahorrar a las demás (menos gases que "
                                       "enfriar, menos purga de la que recuperar calor, menos combustible sobre el que aplicar "
                                       "la mejora de eficiencia). El caso combinado es una sola corrida con todos los cambios.")
            invs = [e["investment"] for e in evaluated]
            if all(i is not None for i in invs) and delivered_t_y is not None and combined["savings"]["annual"]["cost"]["total"] > 0:
                combined["investment"] = float(sum(invs))
                combined["simple_payback_years"] = combined["investment"] / combined["savings"]["annual"]["cost"]["total"]
            else:
                combined["investment"], combined["simple_payback_years"] = None, None
                combined["payback_note"] = "no calculado: falta la inversión de al menos una medida o las cifras anuales."
    elif len(evaluated) == 1:
        combined = {"measures": [evaluated[0]["id"]], "status": "not_applicable",
                    "reason": "solo se evaluó una medida: el caso combinado es esa misma medida"}

    # ---- distribution losses (leaks, failed traps, bare pipe), valued with the baseline steam
    dist = None
    if distribution:
        after = combined["result"]["steam"]["cost_per_tonne"] if combined and combined.get("result") else None
        dist = SA.distribution_losses(
            distribution, spec0["site_pressure_Pa"], currency, hours, base["steam"]["cost_per_tonne"],
            base["steam"]["breakdown_per_GJ_delivered"]["fuel"], base["emissions"]["co2_kg_per_tonne_steam"],
            base["steam"]["heat_delivered_GJ_per_tonne"], delivered_t_y, after)
        warnings.extend(w for w in dist["warnings"] if w not in warnings)
    else:
        missing.append({"item": "el recorrido de la distribución: fugas de vapor, trampas falladas y tramos de tubería sin "
                                "aislar (cuántos, diámetro del orificio o del tubo, longitud y presión)",
                        "parameter": "distribution",
                        "prevents": "valorar lo que se pierde entre la caldera y el proceso; este estudio cubre solo la casa de "
                                    "calderas"})

    # ---- other missing or optional data
    if boiler.get("auxiliary_power_kWh_per_tonne") is None:
        missing.append({"item": "el consumo eléctrico de ventiladores y bombas de la caldera (kWh por tonelada de vapor) y el "
                                "precio de la electricidad",
                        "parameter": "boiler.auxiliary_power_kWh_per_tonne y electricity_cost",
                        "prevents": "cobrar la electricidad de la caldera: el costo del vapor no la incluye"})
        if electricity_cost is not None:
            warnings.append("Diste el precio de la electricidad pero no boiler.auxiliary_power_kWh_per_tonne: la electricidad "
                            "de la caldera no entró al costo.")
    if combustion.get("flue_CO_dry_mole_fraction") is None:
        missing.append({"item": "el monóxido de carbono (CO) medido en la chimenea",
                        "parameter": "combustion.flue_CO_dry_mole_fraction",
                        "prevents": "contar la pérdida por combustión incompleta: se tomó combustión completa"})
    if heating == "not_charged":
        warnings.append("water.feedwater_heating='not_charged': calentar el agua hasta feedwater_T_K no se cobra al combustible. "
                        "Si ese calor sale de vapor de la propia caldera, el costo real del vapor es mayor que el mostrado.")
    measured_fw = water.get("feedwater_T_K")
    mixing_check = None
    if heating == "mixing_only":
        mixing_check = {"feedwater_T_K_calculated": base_case["feedwater_T_K"], "feedwater_T_K_measured": measured_fw,
                        "measured_minus_calculated_K": (float(measured_fw) - base_case["feedwater_T_K"]) if measured_fw is not None else None,
                        "meaning": "Si la medida es menor que la calculada, el tanque pierde calor o regresa menos condensado "
                                   "(o más frío) del declarado; si es mayor, hay una fuente de calor no declarada."}
        assumptions.append("feedwater_heating='mixing_only': la temperatura del agua de alimentación se calcula como la mezcla "
                           "sin pérdidas de condensado y agua de repuesto (el balance del desaireador de utility_costs con "
                           "vapor de calentamiento igual a cero).")

    # ---- data review: plausibility of what the plant gave. It flags, it never refuses and it never corrects a datum
    flags: list[dict] = []

    def flag(ident: str, text: str) -> None:
        flags.append({"id": ident, "message": text})

    T_sat, T_fw, T_stack = base["steam"]["saturation_T_K"], base["steam"]["feedwater_T_K"], spec0["stack_T_K"]
    ea0 = base["boiler"]["excess_air_fraction"]
    if T_stack <= T_fw:
        flag("stack_below_feedwater", f"La chimenea ({T_stack - 273.15:.0f} °C) está igual o más fría que el agua de alimentación "
             f"({T_fw - 273.15:.0f} °C). Los gases no pueden salir más fríos que el agua que los enfría: revisa el punto de "
             "medición o las unidades.")
    elif T_stack < T_sat:
        flag("stack_below_saturation", f"La chimenea ({T_stack - 273.15:.0f} °C) está más fría que el vapor "
             f"({T_sat - 273.15:.0f} °C). Eso solo ocurre si la caldera ya tiene economizador o precalentador de aire: confírmalo.")
    if ea0 > CHECK_EXCESS_AIR_HIGH:
        flag("excess_air_high", f"El exceso de aire sale {100 * ea0:.0f} %, muy alto. Suele indicar entrada de aire falso, toma "
             "de muestra después de una dilución o un analizador sin calibrar: confirma la medición antes de invertir.")
    elif ea0 < CHECK_EXCESS_AIR_LOW:
        flag("excess_air_low", f"El exceso de aire sale {100 * ea0:.1f} %, muy bajo. Con tan poco aire suele aparecer monóxido de "
             "carbono: mide el CO en chimenea y dalo, porque hoy se tomó combustión completa." if
             combustion.get("flue_CO_dry_mole_fraction") is None else
             f"El exceso de aire sale {100 * ea0:.1f} %, muy bajo: confirma la medición de O2.")
    if spec0["blowdown_fraction"] > CHECK_BLOWDOWN_HIGH:
        flag("blowdown_high", f"La purga es {100 * spec0['blowdown_fraction']:.1f} % del agua de alimentación, alta. Confirma "
             "que es fracción del agua de alimentación y no del vapor, y cómo se midió (conductividad o aforo).")
    if cross and abs(cross["difference_points"]) > CHECK_EFFICIENCY_GAP_POINTS:
        flag("efficiency_gap", f"La eficiencia por entradas y salidas ({100 * cross['input_output_efficiency']:.1f} %) y la del "
             f"método de pérdidas ({100 * cross['heat_loss_efficiency']:.1f} %) difieren {cross['difference_points']:+.1f} "
             "puntos. Alguno de los medidores (vapor, combustible) o de los datos de chimenea no cuadra: revisa antes de usar "
             "las cifras anuales.")
    if hv_check and abs(hv_check["difference_fraction"]) > CHECK_HEATING_VALUE_GAP:
        flag("billed_heating_value", f"El poder calorífico que implica el recibo difiere {100 * hv_check['difference_fraction']:+.1f} "
             "% del de la composición dada. Revisa las condiciones base del volumen, la base (superior o inferior) o la "
             "composición.")
    if load_fraction is not None and load_fraction < CHECK_LOAD_LOW:
        flag("low_load", f"La caldera trabaja en promedio a {100 * load_fraction:.0f} % de su capacidad nominal. A carga tan baja "
             "las pérdidas fijas pesan mucho y un solo punto de operación representa mal el año.")
    data_review = {
        "checks_run": ["stack_below_feedwater", "stack_below_saturation", "excess_air_high", "excess_air_low", "blowdown_high"]
        + (["efficiency_gap"] if cross else []) + (["billed_heating_value"] if hv_check else [])
        + (["low_load"] if load_fraction is not None else []),
        "flags": flags,
        "limits": {"excess_air_high": CHECK_EXCESS_AIR_HIGH, "excess_air_low": CHECK_EXCESS_AIR_LOW,
                   "blowdown_high": CHECK_BLOWDOWN_HIGH, "efficiency_gap_points": CHECK_EFFICIENCY_GAP_POINTS,
                   "heating_value_gap": CHECK_HEATING_VALUE_GAP, "low_load": CHECK_LOAD_LOW},
        "meaning": "Revisión de plausibilidad de los datos de la planta. Señala lo que conviene confirmar; no corrige ni "
                   "descarta ningún dato."}
    warnings.extend(f"REVISAR DATO: {x['message']}" for x in flags)

    # ---- collect warnings, assumptions and limits of the engines (once each)
    def add(dst: list[str], items) -> None:
        for x in items:
            if x not in dst:
                dst.append(x)

    for label, cs in [("línea base", base_case)] + list(case_objs.items()):
        tag = "" if label == "línea base" else f"[{MEASURE_LABEL.get(label, 'caso combinado')}] "
        add(warnings, [w if label == "línea base" or w in warnings else tag + w
                       for w in cs["costs"]["warnings"] + cs["combustion"]["warnings"]])
        add(assumptions, [a for a in cs["costs"]["assumptions"]
                          if not (heating == "mixing_only" and a.startswith("Desaireador de contacto directo"))
                          and not a.startswith("La purga se descarga")])
        add(assumptions, cs["combustion"]["assumptions"])
    if not any(e["id"] == "recover_blowdown_heat" for e in evaluated):
        add(assumptions, ["La purga se descarga como líquido saturado a la presión de la caldera, sin recuperar su calor."])
    else:
        add(assumptions, ["En la línea base la purga se descarga como líquido saturado a la presión de la caldera, sin "
                          "recuperar su calor."])
    if fuel_info:
        add(assumptions, [f"La composición del gas NO es un dato de la planta: es {fuel_info['description']}."
                          + (" " + FUEL_SURROGATE_NOTE if fuel_info["method"] == "fit" else "")
                          + "".join(f" Nota: {n}." for n in fuel_info.get("notes", []))])
    if loss_unknown:
        add(assumptions, [f"La pérdida por paredes y radiación NO es un dato de la planta: es "
                          f"{100.0 * setting_loss['value']:.2f} % del poder calorífico (base PCS), tomado de {R['source']} "
                          f"({setting_loss['value_choice']}). Límites: " + "; ".join(setting_loss["limits"]) + "."])
    add(assumptions, [
        "La demanda de vapor del proceso (vapor entregado) se mantiene igual en todas las medidas; lo que cambia es el "
        "combustible, el agua y la electricidad para producirlo.",
        "La pérdida por paredes se mantiene como la misma fracción del poder calorífico en todos los casos. En realidad es casi "
        "constante en kW, así que al bajar el combustible la fracción sube un poco: el ahorro queda ligeramente sobrestimado.",
        "Operación en un solo punto promedio: una carga, un O2 y una temperatura de chimenea representan todo el año.",
        "Los ahorros se valoran con los mismos precios de la línea base.",
    ])
    add(not_included, [x.removeprefix("steam: ") for x in base_case["costs"]["not_included"]])
    add(not_included, [f"combustión: {x}" for x in base_case["combustion"]["not_included"]])
    add(not_included, [
        "Inversión de las medidas: nunca se estima; el retorno solo se calcula con la cifra de la planta",
        "Variación de la carga a lo largo del año, arranques y paros, y eficiencia a carga parcial",
        "Fugas de vapor, trampas de vapor y aislamiento de tuberías",
        "Química del agua: la purga mínima y la calidad del condensado no se comprueban",
        "Punto de rocío ácido y corrosión de la zona fría",
        "Emisiones (CO2, NOx) y su costo",
        "Impuestos, financiamiento y mantenimiento de los equipos nuevos",
    ])

    # ---- provenance
    def leaf(label: str, value, unit: str, path: str) -> dict:
        return {"item": label, "value": value, "unit": unit, "origin": PLANT, "detail": path}

    P_site = spec0["site_pressure_Pa"]
    air = combustion["air"]
    plant = [
        *([] if fuel_info else [
            leaf("Combustible", fuel.get("composition") or fuel.get("mass_fractions"),
                 "fracción molar" if fuel.get("type") == "gas" else "fracción másica", "fuel")]),
        leaf("Presión atmosférica del sitio", P_site, "Pa", "combustion.site_pressure_Pa"),
        leaf("Temperatura del aire de combustión", float(air["T_K"]), "K", "combustion.air.T_K"),
        (leaf("Humedad relativa del aire", float(air["relative_humidity"]), "fracción", "combustion.air.relative_humidity")
         if air.get("relative_humidity") is not None else
         leaf("Humedad del aire", float(air.get("humidity_ratio_kg_kg", 0.0)), "kg de agua por kg de aire seco",
              "combustion.air.humidity_ratio_kg_kg")),
        (leaf(f"Oxígeno en chimenea (base {'seca' if combustion['flue_O2'].get('basis') == 'dry' else 'húmeda'})",
              float(combustion["flue_O2"]["mole_fraction"]), "fracción", "combustion.flue_O2")
         if combustion.get("flue_O2") is not None else
         leaf("Exceso de aire", float(combustion["excess_air_fraction"]), "fracción", "combustion.excess_air_fraction")),
        leaf("Temperatura de chimenea", spec0["stack_T_K"], "K", "combustion.stack_T_K"),
        *([] if loss_unknown else [
            leaf(f"Pérdida por paredes y radiación (del poder calorífico, base {spec0['setting_loss_basis']})",
                 spec0["setting_loss_fraction"], "fracción", "combustion.setting_loss_fraction")]),
        leaf("Presión de la caldera", spec0["steam_pressure_Pa"], "Pa", "boiler.steam_pressure_Pa"),
        leaf("Sobrecalentamiento", spec0["superheat_K"], "K de diferencia", "boiler.superheat_K"),
        leaf("Purga (del agua de alimentación)", spec0["blowdown_fraction"], "fracción", "boiler.blowdown_fraction"),
        leaf("Agua de alimentación", HEATING_ES[heating], "texto", "water.feedwater_heating"),
        leaf("Temperatura del agua de repuesto", spec0["makeup_water_T_K"], "K", "water.makeup_water_T_K"),
        leaf("Retorno de condensado (del vapor)", spec0["condensate_return_fraction"], "fracción", "water.condensate_return_fraction"),
    ]
    if combustion.get("flue_CO_dry_mole_fraction") is not None:
        plant.append(leaf("Monóxido de carbono en chimenea (base seca)", float(combustion["flue_CO_dry_mole_fraction"]) * 1e6,
                          "ppm", "combustion.flue_CO_dry_mole_fraction"))
    if measured_fw is not None:
        plant.append(leaf("Temperatura del agua de alimentación", float(measured_fw), "K", "water.feedwater_T_K"))
    if spec0.get("condensate_T_K") is not None:
        plant.append(leaf("Temperatura del condensado", float(spec0["condensate_T_K"]), "K", "water.condensate_T_K"))
    if boiler.get("auxiliary_power_kWh_per_tonne") is not None:
        plant.append(leaf("Electricidad de ventiladores y bombas", float(boiler["auxiliary_power_kWh_per_tonne"]),
                          "kWh por tonelada de vapor", "boiler.auxiliary_power_kWh_per_tonne"))
    if hours is not None:
        plant.append(leaf("Horas de operación al año", hours, "h al año", "operation.hours_per_year"))
    if steam_kg_s is not None:
        plant.append(leaf("Producción de vapor de la caldera", float(steam_kg_s), "kg/s", "operation.steam_generated_kg_s"))
    if fc is not None:
        plant.append({"item": "Consumo de combustible al año", "value": gj, "unit": f"GJ al año ({b_fc})",
                      "origin": PLANT if fuel_how and fuel_how.startswith("energía facturada") else CALC,
                      "detail": f"operation.fuel_consumption: {fuel_how}"})
    target_label = {
        "target_flue_O2": "oxígeno objetivo en chimenea", "target_excess_air_fraction": "exceso de aire objetivo",
        "stack_T_K_after": "temperatura de chimenea después de afinar", "target_stack_T_K": "temperatura de chimenea objetivo",
        "device": "equipo", "target_blowdown_fraction": "purga objetivo", "flash_pressure_Pa": "presión del tanque de flasheo",
        "drain_T_K": "temperatura de la purga al drenaje", "target_condensate_return_fraction": "retorno de condensado objetivo",
        "condensate_T_K_after": "temperatura del condensado después", "feedwater_T_K_after": "temperatura del agua de alimentación después",
        "investment": "inversión"}
    for name in MEASURE_ORDER:
        for k, v in (measures.get(name) or {}).items():
            if v is None or k == "use_published_target":
                continue
            if k == "target_flue_O2":
                v, unit = float(v["mole_fraction"]), f"fracción (base {'seca' if v.get('basis') == 'dry' else 'húmeda'})"
            elif k == "device":
                v, unit = {"economizer": "economizador", "air_preheater": "precalentador de aire"}[v], "texto"
            else:
                unit = (currency if k == "investment" else "K" if "T_K" in k else "Pa" if k.endswith("_Pa") else "fracción")
            plant.append(leaf(f"Medida «{MEASURE_LABEL[name]}»: {target_label[k]}", v, unit, f"measures.{name}.{k}"))
    published = []
    if loss_unknown:
        published.append({"kind": "setting_loss", "item": "Pérdida por paredes y radiación de la caldera",
                          "value": setting_loss["value"], "unit": f"fracción del poder calorífico (base PCS)", "origin": REF,
                          "reference": R["id"], "source": R["source"], "retrieved": R["retrieved"], "url": R["url"],
                          "detail": f"{R['statement']}; {setting_loss['value_choice']}; {R['source']}; consultada el {R['retrieved']}; "
                                    f"referencia «{R['id']}»; {R['url']}"})
    sources = base_case["costs"]["price_sources"]
    price_es = {"fuel_per_GJ": "Combustible", "electricity_per_kWh": "Electricidad",
                "electricity_time_of_use_per_kWh": "Electricidad por periodo horario",
                "demand_charge_per_kW_month": "Cargo por demanda eléctrica"}
    for k, src in sources.items():
        if src["origin"].startswith("referencia"):
            pv = src["published_value"]
            pub = ", ".join(f"{a} {b}" for a, b in pv.items()) if isinstance(pv, dict) else f"{pv}"
            age = (f"antigüedad {src['age_days']} días al {day}" if src["age_days"] >= 0
                   else f"periodo vigente al {day}")
            published.append({"item": price_es.get(k, k), "value": src["value"], "unit": currency, "origin": REF,
                              "reference": reference, "source": src["source"], "period": src["period"],
                              "retrieved": src["retrieved"], "age_days": src["age_days"], "url": src["url"],
                              "detail": f"{pub} {src['published_unit']}; {src['source']}; periodo {src['period']}; consultada "
                                        f"el {src['retrieved']}; {age}; referencia «{reference}»; {src['url']}"})
    calculated = [
        {"figure": "Poder calorífico, aire y gases de combustión, punto de rocío del agua", "origin": CALC, "tool": TOOL_COMB,
         "from": "combustible, aire, O2 o exceso de aire"},
        {"figure": "Eficiencia de la caldera y desglose de pérdidas", "origin": CALC, "tool": TOOL_COMB,
         "from": "lo anterior + temperatura de chimenea + pérdida por paredes"},
        {"figure": "Combustible, agua de repuesto y costo por tonelada y por GJ de vapor", "origin": CALC, "tool": TOOL_COST,
         "from": "eficiencia (de la herramienta anterior), presión, purga, retorno de condensado, temperaturas y precios"},
        {"figure": "Cifras anuales", "origin": CALC, "tool": TOOL_DIAG,
         "from": "cifras por tonelada × vapor entregado al año. " + (derived_note or "Sin datos de operación: no calculadas.")},
        {"figure": "Ahorro de cada medida y del caso combinado", "origin": CALC, "tool": TOOL_DIAG,
         "from": "línea base − la misma corrida de los dos motores con el objetivo de la planta"},
        {"figure": "Retorno simple", "origin": CALC, "tool": TOOL_DIAG,
         "from": "inversión (dato de la planta) ÷ ahorro anual; solo si se dio la inversión"},
    ]
    if heating == "mixing_only":
        calculated.insert(2, {"figure": "Temperatura del agua de alimentación", "origin": CALC, "tool": TOOL_COST,
                              "from": "mezcla de condensado y agua de repuesto (y calor de purga recuperado, si aplica)"})

    # ---- evidence
    res0 = _residuals(base_case)
    worst = max(res0.values())
    reasons: list[str] = []
    if not base_case["combustion"]["evidence"]["verified"]:
        reasons += base_case["combustion"]["evidence"]["reasons"]
    for label, rr in [("línea base", res0)] + [(e["label"], e["residuals"]) for e in evaluated] + (
            [("caso combinado", combined["residuals"])] if combined and combined.get("residuals") else []):
        bad = {k: v for k, v in rr.items() if v > 1e-8}
        worst = max(worst, max(rr.values()))
        if bad:
            reasons.append(f"{label}: no cierra {bad}")
    evidence: dict = {"baseline_residuals": res0, "worst_residual_fraction": worst}
    duty_route = abs(base["steam"]["boiler_duty_GJ_per_tonne"] - base_case["efficiency"] * s0["fuel_GJ_per_tonne"]) / base["steam"]["boiler_duty_GJ_per_tonne"]
    evidence["steam_side_duty_vs_efficiency_times_fuel"] = duty_route
    if duty_route > 1e-8:
        reasons.append("el calor al vapor (IAPWS) no coincide con eficiencia × combustible")
    if delivered_t_y is not None:
        hv_key = f"heat_release_{basis}_W"
        route = abs(base["firing"][hv_key] * hours * 3600.0 / 1e9 - base["annual"]["fuel_GJ"]) / base["annual"]["fuel_GJ"]
        evidence["annual_fuel_two_routes_difference_fraction"] = route
        if route > 1e-8:
            reasons.append("el combustible anual por la ruta de combustión no coincide con el de la isla de vapor")
    evidence["measures_direction"] = {e["id"]: {"cost_per_tonne_saving": e["savings"]["per_tonne"]["cost"],
                                                "saves": e["savings"]["per_tonne"]["cost"] >= 0} for e in evaluated}
    if combined and combined.get("savings"):
        c_t, s_t = combined["savings"]["per_tonne"]["cost"], combined["sum_of_individual"]["per_tonne"]["cost"]
        all_save = all(e["savings"]["per_tonne"]["cost"] >= 0 for e in evaluated)
        ok = c_t <= s_t * (1 + 1e-9) + 1e-12 or not all_save
        evidence["combined_not_above_sum"] = {"combined_per_tonne": c_t, "sum_per_tonne": s_t, "holds": bool(c_t <= s_t * (1 + 1e-9) + 1e-12)}
        if not ok:
            reasons.append("el caso combinado ahorra más que la suma de las medidas individuales y no hay explicación")
    if cross:
        evidence["input_output_cross_check"] = cross
    if hv_check:
        evidence["billed_heating_value_check"] = hv_check
    evidence["carbon_balance_error"] = base["emissions"]["carbon_balance_error"]
    if evidence["carbon_balance_error"] > 1e-9:
        reasons.append("el carbono del combustible no coincide con el CO2 y CO de los gases")
    if fuel_info and fuel_info["method"] == "fit":
        chk = fuel_info["heating_value_check"]
        evidence["fitted_heating_value_error"] = abs(chk["fitted_J_mol"] / chk["measured_J_mol"] - 1.0)
        if evidence["fitted_heating_value_error"] > 1e-9:
            reasons.append("la composición ajustada no reproduce el poder calorífico medido")
    if mixing_check:
        evidence["feed_tank_mixing_check"] = mixing_check
    evidence["verified"] = not reasons
    evidence["reasons"] = reasons
    evidence["meaning"] = ("verified = los balances de masa y energía de los dos motores cierran en todos los casos y las rutas "
                           "independientes coinciden. NO significa que los datos de la planta sean correctos.")

    P = spec0["steam_pressure_Pa"]
    base["steam"]["pressure_gauge_Pa"] = P - P_site
    assumed_values = []
    if fuel_info:
        assumed_values.append({
            "id": "fuel_composition", "item": "Composición del gas natural",
            "value": ", ".join(f"{k} {100 * v:.2f} %" for k, v in fuel_info["composition"].items()) + " (mol)",
            "origin": fuel_resolution["origin"], "detail": fuel_info["description"]})
    for name, target in published_targets:
        assumed_values.append({
            "id": f"target_{name}", "item": f"Objetivo de la medida «{MEASURE_LABEL[name]}»", "value": target["text"],
            "origin": REF, "detail": f"{target['statement']}. {target['source']}. Consultada el {target['retrieved']}. "
                                     f"Límite: {target['caveat']}. {target['url']}"})
        warnings.insert(0, f"{TARGET_FLAG}: la medida «{MEASURE_LABEL[name]}» se evaluó con {target['text']}, de "
                           f"{target['source']} ({target['statement']}); {target['caveat']}.")
    if loss_unknown:
        assumed_values.append({
            "id": "setting_loss", "item": "Pérdida por paredes y radiación de la caldera",
            "value": f"{100.0 * setting_loss['value']:.2f} % del poder calorífico (base PCS)", "origin": REF,
            "detail": f"{R['statement']}; {setting_loss['value_choice']}. {R['source']}. Consultada el {R['retrieved']}. "
                      "Límites: " + "; ".join(setting_loss["limits"]) + "."})
    first = (U.REFERENCE_FLAG, SETTING_LOSS_FLAG, FR.FUEL_UNKNOWN_FLAG, TARGET_FLAG)
    warnings = [w for w in warnings if w.startswith(first)] + [w for w in warnings if not w.startswith(first)]
    return {
        "currency": currency, "fuel_basis": basis, "fuel": {"HHV_J_kg": f0["HHV_J_kg"], "LHV_J_kg": f0["LHV_J_kg"],
                                                           "HHV_J_Nm3": f0.get("HHV_J_Nm3"), "type": f0["type"]},
        "prices": {"fuel_per_GJ": resolved["fuel_per_GJ"]["value"], "fuel_basis": basis,
                   "boiler_makeup_water_per_m3": sources["boiler_makeup_water_per_m3"]["value"],
                   "electricity_per_kWh": base_case["costs"]["electricity"]["per_kWh"] if "electricity" in base_case["costs"] else None},
        "baseline": base, "setting_loss": setting_loss, "fuel_resolution": fuel_resolution, "assumed_values": assumed_values,
        "data_review": data_review, "measures": results, "combined": combined, "distribution": dist,
        "provenance": {"plant_data": plant, "prices": price_prov, "published_references": published, "calculated": calculated},
        "missing_data": missing, "evidence": evidence, "warnings": warnings, "assumptions": assumptions,
        "not_included": not_included, "inputs": inputs,
        "method": {"name": "diagnóstico de vapor: línea base y medidas por repetición de los mismos motores con un cambio",
                   "module": "packages/energy/steam_diagnostic.py", "engines": [TOOL_COMB, TOOL_COST],
                   "validated_against": "ejemplos de los U.S. DOE Steam Tip Sheets #3, #4, #8, #9 y #10 (enero 2012) y una "
                                        "prueba unitaria de caldera de MEASUR-Tools-Suite (ORNL); ver BENCHMARK.md, M19"},
    }
