"""Utility costs by region (bloque M17): from the primary prices a plant pays (fuel, electricity, water) to the cost of
each utility level: steam at a pressure, fired heat, cooling-tower water, refrigeration at a temperature, electricity.

Route 1 (physics, the default): energy and water balances with water/steam properties from CoolProp (IAPWS-95, MIT).
Every efficiency, return fraction, cycles of concentration, etc. is an input of the user: nothing engineering is defaulted.
Route 2 (optional, independent): Ulrich & Vasudevan, "How to Estimate Utility Costs", Chemical Engineering, April 2006,
pp. 66-69, Table 1: C_S,u = a·(CE PCI) + b·C_S,f. It is a PRICE that includes the capital and labour of the utility plant,
so it is not the same quantity as route 1 (operating cost of fuel + water + power only).

Primary prices come from the user. A bank of published, dated reference prices (data/utility_price_references.json) is
used ONLY when the user names a reference; a result built on it says so first in its warnings. User prices always win.
Currency is explicit; conversion only with an exchange rate given by the user.

Validation (tests/test_utilities.py, notebooks/30_costo_de_servicios.py): U.S. DOE Steam Tip Sheet #15 (Table 1 and its
example) for the steam route; the two worked examples of Ulrich & Vasudevan 2006 for route 2.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

G_STD = 9.80665                 # m/s²
BTU_J = 1055.05585262           # J per Btu (IT)
KWH_PER_GJ = 1e9 / 3.6e6        # 277.78 kWh per GJ
P_ATM = 101325.0                # Pa
BANK_PATH = Path(__file__).parent / "data" / "utility_price_references.json"

PRICE_KEYS = {
    "fuel_per_GJ": "combustible por GJ (di la base en fuel_basis: HHV o LHV)",
    "electricity_per_kWh": "electricidad por kWh (precio de energía)",
    "electricity_time_of_use_per_kWh": "electricidad por periodo horario, dict {periodo: precio por kWh}",
    "demand_charge_per_kW_month": "cargo por demanda por kW-mes (opcional)",
    "boiler_makeup_water_per_m3": "agua de repuesto de caldera (tratada) por m³",
    "cooling_makeup_water_per_m3": "agua de repuesto de la torre de enfriamiento por m³",
}
NUMERIC_PRICE_KEYS = [k for k in PRICE_KEYS if k != "electricity_time_of_use_per_kWh"]
REFERENCE_FLAG = "PRECIO DE REFERENCIA PUBLICADO, NO TU CONTRATO"


def _cp():
    import CoolProp.CoolProp as CP
    return CP


def _h_liq(T: float) -> float:
    """Enthalpy of saturated liquid water at T (J/kg). Liquid streams (feed water, condensate, make-up) are evaluated on
    the saturation line: the pressure effect (≈ v·ΔP) is the feed-pump work, not charged here."""
    return _cp().PropsSI("H", "T", T, "Q", 0, "Water")


def _need(d: dict, keys: list[str], where: str) -> None:
    miss = [k for k in keys if d.get(k) is None]
    if miss:
        raise ValueError(f"{where}: faltan {miss}. Son datos de ingeniería de TU planta; no se suponen valores típicos.")


def _known(d: dict, allowed: list[str], where: str) -> None:
    extra = sorted(k for k in d if k not in allowed)
    if extra:
        raise ValueError(f"{where}: claves desconocidas {extra}. Válidas: {sorted(allowed)}.")


def _frac(v: float, name: str, where: str, lo_open: bool = False, hi_open: bool = False) -> float:
    v = float(v)
    if not (0.0 <= v <= 1.0) or (lo_open and v == 0.0) or (hi_open and v == 1.0):
        raise ValueError(f"{where}: {name} = {v} fuera de rango (fracción entre 0 y 1).")
    return v


# ------------------------------------------------------------------------------------------------------------ bank
@lru_cache(maxsize=1)
def load_bank() -> dict:
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))


def _entry_per_unit(e: dict) -> tuple[float | dict, str]:
    """Value of a bank entry in the canonical unit of its key, and how it was derived from the published number."""
    v, unit = e["value"], e["unit"]
    if e["key"] == "fuel_per_GJ" and unit.startswith("USD/Mcf"):
        gj_per_mcf = e["conversion"]["heat_content_btu_per_cf"] * 1000.0 * BTU_J / 1e9
        return v / gj_per_mcf, (f"{v} USD/Mcf ÷ {gj_per_mcf:.5f} GJ/Mcf (contenido de calor "
                                f"{e['conversion']['heat_content_btu_per_cf']} Btu/pie³, {e['conversion']['heat_content_period']})")
    if e["key"] == "electricity_per_kWh" and unit.startswith("centavos"):
        return v / 100.0, f"{v} centavos/kWh ÷ 100"
    return v, "valor publicado, sin conversión"


def list_references(region: str | None = None) -> dict:
    """References in the bank, optionally filtered by region prefix (US, US-TX, EU27, DE, MX...)."""
    bank = load_bank()
    out = []
    for rid, r in bank["references"].items():
        if region and not (r["region"].upper().startswith(region.upper()) or rid.upper().startswith(region.upper())):
            continue
        prices = []
        for e in r["entries"]:
            val, how = _entry_per_unit(e)
            prices.append({"key": e["key"], "value": val, "currency": e["currency"], "commodity": e["commodity"],
                           "published_value": e["value"], "published_unit": e["unit"], "derivation": how,
                           "period": e["period"], "source": e["source"], "url": e["url"], "caveat": e.get("caveat", ""),
                           **({"fuel_basis": e["fuel_basis"]} if "fuel_basis" in e else {})})
        out.append({"id": rid, "region": r["region"], "label": r["label"], "currency": r["currency"],
                    "retrieved": r["retrieved"], "prices": prices})
    return {"bank_version": bank["version"], "references": out, "not_included": bank["not_included"],
            "rule": "Son precios publicados, no tu contrato. utility_costs solo los usa si nombras uno en reference=; "
                    "tus precios (prices=) siempre ganan. El agua nunca viene del banco.",
            "available_ids": [o["id"] for o in out]}


def _age(period_end: str, today: date) -> tuple[int, str]:
    days = (today - date.fromisoformat(period_end)).days
    return days, (f"{days} días (≈ {days / 30.44:.0f} meses)" if days >= 0 else f"el periodo termina en {-days} días")


def resolve_prices(currency: str, prices: dict | None, reference: str | None, exchange_rates: dict | None,
                   today: date) -> tuple[dict, list[str]]:
    """Primary prices in `currency`, each with its origin. User prices win over the reference key by key."""
    prices = dict(prices or {})
    unknown = [k for k in prices if k not in PRICE_KEYS and k != "fuel_basis"]
    if unknown:
        raise ValueError(f"prices: claves desconocidas {unknown}. Válidas: {sorted(PRICE_KEYS)} y fuel_basis.")
    bank = load_bank()
    ids = sorted(bank["references"])
    resolved: dict = {}
    warnings: list[str] = []
    if reference is not None:
        if reference not in bank["references"]:
            raise ValueError(f"reference {reference!r} no está en el banco. Referencias disponibles: {ids}")
        ref = bank["references"][reference]
        used = []
        for e in ref["entries"]:
            if prices.get(e["key"]) is not None:
                continue                                                         # the user's price wins
            val, how = _entry_per_unit(e)
            cur = e["currency"]
            fx = None
            if cur != currency:
                fx = (exchange_rates or {}).get(cur)
                if fx is None:
                    raise ValueError(
                        f"La referencia {reference} está en {cur} y pediste {currency}: da exchange_rates={{'{cur}': <{currency} "
                        f"por 1 {cur}>}}. Nunca se usa un tipo de cambio por defecto.")
                val = {k: v * fx for k, v in val.items()} if isinstance(val, dict) else val * fx
            days, age = _age(e["period_end"], today)
            resolved[e["key"]] = {"value": val, "currency": currency, "origin": f"referencia {reference}",
                                  "published_value": e["value"], "published_unit": e["unit"], "derivation": how,
                                  "exchange_rate_applied": fx, "period": e["period"], "age_days": days, "source": e["source"],
                                  "url": e["url"], "retrieved": ref["retrieved"], "caveat": e.get("caveat", ""),
                                  **({"fuel_basis": e["fuel_basis"]} if "fuel_basis" in e else {}),
                                  **({"components": e["components"]} if "components" in e else {})}
            used.append((e, age, days))
        if used:
            parts = [f"{e['key']} = {e['value']} {e['unit']} ({e['source']}; periodo {e['period']}; antigüedad {age} al "
                     f"{today.isoformat()})" for e, age, _ in used]
            w = (f"{REFERENCE_FLAG}: {'; '.join(parts)}. Referencia «{reference}», consultada el {ref['retrieved']}. "
                 f"Sustitúyelos por los de tu planta en prices= en cuanto los tengas.")
            if max(d for _, _, d in used) > 365:
                w += " ATENCIÓN: al menos uno tiene más de un año; busca el dato vigente."
            warnings.append(w)
            warnings.extend(f"Nota de la referencia ({e['key']}): {e['caveat']}" for e, _, _ in used if e.get("caveat"))
        else:
            warnings.append(f"Nombraste la referencia «{reference}» pero diste tus propios precios para todo lo que ofrece: "
                            f"no se usó ningún precio de referencia.")
    for k in PRICE_KEYS:
        if prices.get(k) is None:
            continue
        v = prices[k]
        if k == "electricity_time_of_use_per_kWh":
            if not isinstance(v, dict) or not v or any(float(x) < 0 for x in v.values()):
                raise ValueError("prices.electricity_time_of_use_per_kWh: dict {periodo: precio por kWh} con precios ≥ 0.")
            v = {p: float(x) for p, x in v.items()}
        else:
            v = float(v)
            if v < 0:
                raise ValueError(f"prices.{k} = {v}: un precio no puede ser negativo.")
        resolved[k] = {"value": v, "currency": currency, "origin": "usuario"}
    if "fuel_per_GJ" in resolved and resolved["fuel_per_GJ"]["origin"] == "usuario":
        basis = prices.get("fuel_basis")
        if basis not in ("HHV", "LHV"):
            raise ValueError("prices.fuel_basis: di si tu precio de combustible es por GJ de poder calorífico superior ('HHV') o "
                             "inferior ('LHV'). La eficiencia de la caldera o del horno debe estar en la misma base.")
        resolved["fuel_per_GJ"]["fuel_basis"] = basis
    return resolved, warnings


def _require(resolved: dict, needs: list[tuple[str, str]], reference: str | None) -> None:
    """Refuse once, naming every missing primary price, what it is for and where it can come from."""
    bank = load_bank()
    ids = sorted(bank["references"])
    in_bank = {e["key"] for r in bank["references"].values() for e in r["entries"]}
    miss = [(k, why) for k, why in needs if k not in resolved]
    if not miss:
        return
    lines = []
    for k, why in miss:
        if k in in_bank:
            src = (f"la referencia «{reference}» no lo trae; " if reference else "") + "dalo en prices= o nombra una referencia en reference="
        else:
            src = "no existe en el banco de referencias: siempre es el de tu planta, dalo en prices="
        lines.append(f"«{k}» ({PRICE_KEYS[k]}), para {why}: {src}")
    msg = "Faltan precios; no se suponen. " + " | ".join(lines) + "."
    if any(k in in_bank for k, _ in miss):
        msg += f" Referencias publicadas disponibles: {ids} (list_utility_price_references da su fuente y periodo)."
    raise ValueError(msg)


# ------------------------------------------------------------------------------------------------------ electricity
def electricity_cost(resolved: dict, spec: dict | None, hours_per_year: float | None, reference: str | None,
                     needed_for: str = "la electricidad") -> dict:
    """Effective price per kWh: energy price (flat, or time-of-use prices weighted by the user's energy fractions) plus,
    if the user gives a load factor, the demand charge spread over the operating hours."""
    spec = dict(spec or {})
    _known(spec, ["time_of_use_fractions", "load_factor"], "electricity")
    assumptions, warnings, not_included = [], [], []
    tou_frac = spec.get("time_of_use_fractions")
    flat, tou = resolved.get("electricity_per_kWh"), resolved.get("electricity_time_of_use_per_kWh")
    if flat is None and tou is None:
        _require(resolved, [("electricity_per_kWh", needed_for)], reference)
    if flat is not None and tou is not None and flat["origin"] != tou["origin"]:
        use_tou = tou["origin"] == "usuario"                                     # the user's price wins over the reference
    else:
        use_tou = tou is not None and (flat is None or tou_frac is not None)
    if not use_tou:
        energy, origin, detail = flat["value"], flat["origin"], {"kind": "flat"}
        if tou_frac is not None:
            warnings.append("electricity.time_of_use_fractions ignorado: se usó tu precio único por kWh (prices.electricity_per_kWh).")
    else:
        p, origin = tou["value"], tou["origin"]
        if tou_frac is None:
            raise ValueError(f"El precio de electricidad disponible es por periodo horario {sorted(p)} ({origin}): da "
                             f"electricity={{'time_of_use_fractions': {{periodo: fracción de TU energía}}}} (suman 1). El "
                             f"perfil de carga es de tu planta; no se supone.")
        if set(tou_frac) != set(p) or abs(sum(float(x) for x in tou_frac.values()) - 1.0) > 1e-6 or any(float(x) < 0 for x in tou_frac.values()):
            raise ValueError(f"electricity.time_of_use_fractions debe tener exactamente los periodos {sorted(p)} con fracciones ≥ 0 "
                             f"que sumen 1.")
        energy = sum(float(tou_frac[k]) * p[k] for k in p)
        detail = {"kind": "time_of_use", "prices_per_kWh": p, "energy_fractions": {k: float(tou_frac[k]) for k in p}}
    demand_adder = 0.0
    dc = resolved.get("demand_charge_per_kW_month")
    lf = spec.get("load_factor")
    if dc is not None and lf is not None:
        lf = _frac(lf, "load_factor", "electricity", lo_open=True)
        if not hours_per_year or hours_per_year <= 0:
            raise ValueError("electricity: para repartir el cargo por demanda da hours_per_year (horas de operación al año).")
        demand_adder = 12.0 * dc["value"] / (lf * hours_per_year)
        detail["demand"] = {"charge_per_kW_month": dc["value"], "origin": dc["origin"], "load_factor": lf,
                            "hours_per_year": hours_per_year, "adder_per_kWh": demand_adder,
                            "formula": "12 · cargo / (factor de carga · horas al año)"}
        assumptions.append("Cargo por demanda repartido en la energía: demanda facturable = carga media en operación ÷ factor "
                           "de carga, 12 facturas al año, energía = carga media × horas de operación.")
        if dc["origin"] != origin:
            warnings.append(f"El precio de energía viene de «{origin}» y el cargo por demanda de «{dc['origin']}»: verifica que tu "
                            f"precio por kWh no incluya ya la demanda.")
    elif dc is not None:
        not_included.append(f"Cargo por demanda de {dc['value']:.6g} {dc['currency']}/kW-mes ({dc['origin']}): no entra en el "
                            f"precio por kWh porque no diste electricity.load_factor (carga media ÷ demanda facturable).")
    elif lf is not None:
        warnings.append("electricity.load_factor dado pero no hay cargo por demanda (prices.demand_charge_per_kW_month): no se aplicó.")
    return {"per_kWh": energy + demand_adder, "energy_per_kWh": energy, "demand_adder_per_kWh": demand_adder,
            "per_GJ": (energy + demand_adder) * KWH_PER_GJ, "origin": origin, "detail": detail,
            "assumptions": assumptions, "warnings": warnings, "not_included": not_included}


# ------------------------------------------------------------------------------------------------------------ steam
STEAM_REQUIRED = ["name", "pressure_Pa", "boiler_efficiency", "feedwater_T_K", "condensate_return_fraction",
                  "blowdown_fraction", "makeup_water_T_K", "feedwater_heating"]
STEAM_OPTIONAL = ["superheat_K", "condensate_T_K", "auxiliary_power_kWh_per_tonne"]


def steam_level(level: dict, fuel_per_GJ: float, water_per_m3: float, electricity_per_kWh: float | None = None) -> dict:
    """Cost of steam generated at one pressure, per tonne delivered to the process and per GJ delivered.

    level: name, pressure_Pa (absolute), superheat_K (0 = saturated), boiler_efficiency (fuel-to-steam, same heating-value
    basis as the fuel price), feedwater_T_K (boiler inlet), condensate_return_fraction (of the steam delivered),
    blowdown_fraction (of the feed water), makeup_water_T_K, feedwater_heating: 'not_charged' (the fuel only takes the feed
    water from feedwater_T_K to steam, as the DOE benchmark does) or 'deaerator_steam' (the feed water is heated from the
    condensate + make-up mix to feedwater_T_K with steam taken from the boiler: needs condensate_T_K if there is return).
    Optional auxiliary_power_kWh_per_tonne (fans, feed pumps), charged at the electricity price.
    """
    CP = _cp()
    where = f"steam[{level.get('name', '?')}]"
    _known(level, STEAM_REQUIRED + STEAM_OPTIONAL, where)
    _need(level, STEAM_REQUIRED, where)
    P = float(level["pressure_Pa"])
    sh = float(level.get("superheat_K") or 0.0)
    eta = _frac(level["boiler_efficiency"], "boiler_efficiency", where, lo_open=True)
    r = _frac(level["condensate_return_fraction"], "condensate_return_fraction", where)
    beta = _frac(level["blowdown_fraction"], "blowdown_fraction", where, hi_open=True)
    T_fw, T_mu = float(level["feedwater_T_K"]), float(level["makeup_water_T_K"])
    mode = level["feedwater_heating"]
    if mode not in ("not_charged", "deaerator_steam"):
        raise ValueError(f"{where}: feedwater_heating debe ser 'not_charged' o 'deaerator_steam'.")
    if not (CP.PropsSI("ptriple", "Water") < P < CP.PropsSI("pcrit", "Water")):
        raise ValueError(f"{where}: pressure_Pa = {P} fuera del intervalo de saturación del agua.")
    if sh < 0:
        raise ValueError(f"{where}: superheat_K no puede ser negativo.")
    T_sat = CP.PropsSI("T", "P", P, "Q", 1, "Water")
    if not (273.16 < T_mu < 373.0):
        raise ValueError(f"{where}: makeup_water_T_K = {T_mu} fuera de 273.16-373 K.")
    if not (273.16 < T_fw < T_sat):
        raise ValueError(f"{where}: feedwater_T_K = {T_fw} debe estar entre 273.16 K y la saturación a esa presión ({T_sat:.2f} K).")
    h_s = CP.PropsSI("H", "P", P, "Q", 1, "Water") if sh == 0 else CP.PropsSI("H", "T", T_sat + sh, "P", P, "Water")
    h_f = CP.PropsSI("H", "P", P, "Q", 0, "Water")
    h_fw, h_mu = _h_liq(T_fw), _h_liq(T_mu)
    rho_mu = CP.PropsSI("D", "T", T_mu, "P", P_ATM, "Water")
    assumptions = [
        "Cada nivel se calcula como generado directamente a su presión en una caldera, sin cogeneración ni turbinas de "
        "contrapresión (con cogeneración el vapor de baja cuesta menos).",
        "Calor entregado al proceso = entalpía del vapor − líquido saturado a la misma presión (no se acredita el subenfriamiento "
        "del condensado ni el vapor de flasheo).",
        "Agua y vapor: IAPWS-95 (CoolProp). Las corrientes líquidas se evalúan en la línea de saturación a su temperatura; el "
        "trabajo de la bomba de alimentación no se cobra aparte.",
        "La purga se descarga como líquido saturado a la presión de la caldera, sin recuperar su calor.",
    ]
    not_included = ["Capital, mano de obra y mantenimiento de la caldera", "Pérdidas de distribución y trampas de vapor",
                    "Tratamiento químico (inclúyelo en el precio del agua de repuesto si lo quieres cobrar)"]
    T_c = level.get("condensate_T_K")
    h_c = None
    if T_c is not None:
        T_c = float(T_c)
        if not (273.16 < T_c <= T_sat):
            raise ValueError(f"{where}: condensate_T_K = {T_c} debe estar entre 273.16 K y la saturación ({T_sat:.2f} K).")
        h_c = _h_liq(T_c)
    if mode == "deaerator_steam":
        if r > 0 and h_c is None:
            raise ValueError(f"{where}: con feedwater_heating='deaerator_steam' y retorno de condensado da condensate_T_K.")
        hc = h_c if r > 0 else h_mu
        den = (h_s - h_mu) - (h_fw - h_mu) / (1.0 - beta)
        gross = ((h_s - h_mu) - r * (hc - h_mu)) / den                           # kg generated per kg delivered
        if gross < 1.0 - 1e-9:
            raise ValueError(f"{where}: la mezcla de condensado y repuesto ya está más caliente que feedwater_T_K = {T_fw} K; "
                             f"da la temperatura real de entrada a la caldera o usa feedwater_heating='not_charged'.")
        gross = max(gross, 1.0)
        assumptions.append("Desaireador de contacto directo sin venteo: el agua de alimentación se calienta de la mezcla "
                           "condensado + repuesto a feedwater_T_K con vapor tomado de la propia caldera.")
    else:
        gross = 1.0
        assumptions.append("feedwater_heating='not_charged': el combustible solo lleva el agua de feedwater_T_K a vapor (costo de "
                           "generación tipo DOE Steam Tip Sheet #15). Calentar el agua hasta feedwater_T_K no se cobra: es válido "
                           "si ese calor es recuperado; si no, el costo real es mayor.")
    feed = gross / (1.0 - beta)
    blowdown = beta * feed
    deaerator = gross - 1.0
    makeup = feed - r - deaerator                                                 # kg per kg delivered
    if makeup < -1e-9:
        raise ValueError(f"{where}: el balance de agua da repuesto negativo ({makeup:.4f} kg/kg); revisa las fracciones.")
    makeup = max(makeup, 0.0)
    absorbed = gross * (h_s - h_fw) + blowdown * (h_f - h_fw)                     # J per kg delivered, taken up in the boiler
    fuel = absorbed / eta
    q_del = h_s - h_f
    c_fuel = fuel_per_GJ * fuel / 1e6                                             # per tonne (J/kg ≡ kJ/t ·1e-3 → GJ/t = J/kg / 1e6)
    c_water = makeup * 1000.0 / rho_mu * water_per_m3
    aux = level.get("auxiliary_power_kWh_per_tonne")
    c_power = 0.0
    if aux is not None:
        if electricity_per_kWh is None:
            raise ValueError(f"{where}: auxiliary_power_kWh_per_tonne dado pero no hay precio de electricidad.")
        c_power = float(aux) * electricity_per_kWh
    else:
        not_included.append("Electricidad de ventiladores y bombas de la caldera (da auxiliary_power_kWh_per_tonne para cobrarla)")
    total = c_fuel + c_water + c_power
    gj_t = q_del / 1e6
    # Independent closure of the balance: boiler control volume, and (deaerator mode) the whole island.
    boiler_res = (feed * h_fw + eta * fuel) - (gross * h_s + blowdown * h_f)
    balance = {"boiler_residual_fraction": boiler_res / (eta * fuel)}
    if mode == "deaerator_steam":
        island_in = r * (h_c if r > 0 else 0.0) + makeup * h_mu + eta * fuel
        island_out = 1.0 * h_s + blowdown * h_f
        balance["island_residual_fraction"] = (island_in - island_out) / (eta * fuel)
        deaer_res = (r * (h_c if r > 0 else 0.0) + makeup * h_mu + deaerator * h_s) - feed * h_fw
        balance["deaerator_residual_fraction"] = deaer_res / (feed * h_fw)
    elif h_c is not None or r == 0:
        free = feed * h_fw - (r * (h_c or 0.0) + makeup * h_mu)
        balance["feedwater_preheat_not_charged_GJ_per_tonne"] = free / 1e6
    return {
        "name": level["name"], "pressure_Pa": P, "saturation_T_K": T_sat, "steam_T_K": T_sat + sh,
        "cost_per_tonne": total, "cost_per_GJ_delivered": total / gj_t, "heat_delivered_GJ_per_tonne": gj_t,
        "breakdown_per_tonne": {"fuel": c_fuel, "water": c_water, "power": c_power},
        "breakdown_per_GJ_delivered": {"fuel": c_fuel / gj_t, "water": c_water / gj_t, "power": c_power / gj_t},
        "fuel_GJ_per_tonne": fuel / 1e6, "fuel_GJ_per_GJ_delivered": fuel / q_del,
        "floor_fuel_per_GJ_over_efficiency": fuel_per_GJ / eta,
        "mass_per_kg_delivered": {"generated": gross, "feedwater": feed, "blowdown": blowdown, "deaerator_steam": deaerator,
                                  "makeup": makeup, "condensate_return": r},
        "makeup_m3_per_tonne": makeup * 1000.0 / rho_mu,
        "enthalpy_J_kg": {"steam": h_s, "saturated_liquid": h_f, "feedwater": h_fw, "makeup": h_mu,
                          **({"condensate": h_c} if h_c is not None else {})},
        "energy_balance": balance, "assumptions": assumptions, "not_included": not_included,
    }


def fired_heat(spec: dict, fuel_per_GJ: float) -> dict:
    """Direct-fired furnace: cost per GJ absorbed by the process = fuel price / furnace efficiency."""
    _known(spec, ["furnace_efficiency"], "fired_heat")
    _need(spec, ["furnace_efficiency"], "fired_heat")
    eta = _frac(spec["furnace_efficiency"], "furnace_efficiency", "fired_heat", lo_open=True)
    return {"cost_per_GJ_absorbed": fuel_per_GJ / eta, "fuel_GJ_per_GJ_absorbed": 1.0 / eta,
            "breakdown_per_GJ": {"fuel": fuel_per_GJ / eta},
            "assumptions": ["Eficiencia del horno en la misma base de poder calorífico que el precio del combustible."],
            "not_included": ["Capital y mantenimiento del horno", "Electricidad de ventiladores"]}


# ---------------------------------------------------------------------------------------------------- cooling water
CW_REQUIRED = ["supply_T_K", "return_T_K", "pump_head_m", "pump_efficiency", "evaporation_latent_fraction", "drift_fraction",
               "cycles_of_concentration"]


def cooling_water(spec: dict, electricity_per_kWh: float, water_per_m3: float) -> dict:
    """Recirculating cooling-tower water, per m³ circulated and per GJ removed from the process.

    spec: supply_T_K, return_T_K, pump_head_m (total head of the circulation pumps), pump_efficiency (overall, wire to water),
    evaporation_latent_fraction (share of the tower duty rejected by evaporation; 1.0 is the upper bound), drift_fraction (of
    the circulation), cycles_of_concentration (> 1). Optional fan_power_kWh_per_m3, treatment_cost_per_m3_makeup.
    """
    CP = _cp()
    _known(spec, CW_REQUIRED + ["fan_power_kWh_per_m3", "treatment_cost_per_m3_makeup"], "cooling_water")
    _need(spec, CW_REQUIRED, "cooling_water")
    Ts, Tr = float(spec["supply_T_K"]), float(spec["return_T_K"])
    if not (273.16 < Ts < Tr < 373.0):
        raise ValueError("cooling_water: se requiere 273.16 K < supply_T_K < return_T_K < 373 K.")
    eta = _frac(spec["pump_efficiency"], "pump_efficiency", "cooling_water", lo_open=True)
    f_lat = _frac(spec["evaporation_latent_fraction"], "evaporation_latent_fraction", "cooling_water")
    drift_f = _frac(spec["drift_fraction"], "drift_fraction", "cooling_water")
    N = float(spec["cycles_of_concentration"])
    head = float(spec["pump_head_m"])
    if N <= 1.0:
        raise ValueError("cooling_water: cycles_of_concentration debe ser > 1 (repuesto ÷ (purga + arrastre)).")
    if head < 0:
        raise ValueError("cooling_water: pump_head_m no puede ser negativo.")
    rho = CP.PropsSI("D", "T", Ts, "P", P_ATM, "Water")
    q_m3 = rho * (_h_liq(Tr) - _h_liq(Ts))                                        # J removed per m³ circulated
    T_mean = 0.5 * (Ts + Tr)
    h_fg = CP.PropsSI("H", "T", T_mean, "Q", 1, "Water") - CP.PropsSI("H", "T", T_mean, "Q", 0, "Water")
    evap = f_lat * q_m3 / h_fg                                                    # kg per m³ circulated
    drift = drift_f * rho
    warnings = []
    blowdown = evap / (N - 1.0) - drift
    effective_N = N
    if blowdown < 0:
        blowdown = 0.0
        effective_N = (evap + drift) / drift
        warnings.append(f"El arrastre solo ya saca más agua que la purga necesaria para {N:g} ciclos: purga = 0 y los ciclos "
                        f"reales son {effective_N:.2f}.")
    makeup = evap + drift + blowdown
    pump_kwh = rho * G_STD * head / eta / 3.6e6
    fan = spec.get("fan_power_kWh_per_m3")
    treat = spec.get("treatment_cost_per_m3_makeup")
    not_included = ["Capital y mantenimiento de la torre y las bombas"]
    if fan is None:
        not_included.append("Electricidad de los ventiladores de la torre (da fan_power_kWh_per_m3 para cobrarla)")
    if treat is None:
        not_included.append("Tratamiento químico del agua (da treatment_cost_per_m3_makeup para cobrarlo)")
    c_pump = pump_kwh * electricity_per_kWh
    c_fan = float(fan or 0.0) * electricity_per_kWh
    makeup_m3 = makeup / rho
    c_water = makeup_m3 * water_per_m3
    c_treat = makeup_m3 * float(treat or 0.0)
    total = c_pump + c_fan + c_water + c_treat
    gj = q_m3 / 1e9
    return {
        "cost_per_m3": total, "cost_per_1000_m3": 1000.0 * total, "cost_per_GJ_removed": total / gj,
        "heat_removed_GJ_per_m3": gj, "m3_per_GJ_removed": 1.0 / gj,
        "breakdown_per_m3": {"pump_power": c_pump, "fan_power": c_fan, "makeup_water": c_water, "treatment": c_treat},
        "breakdown_per_GJ_removed": {"pump_power": c_pump / gj, "fan_power": c_fan / gj, "makeup_water": c_water / gj,
                                     "treatment": c_treat / gj},
        "power_kWh_per_m3": {"pump": pump_kwh, "fan": float(fan or 0.0)},
        "water_kg_per_m3_circulated": {"evaporation": evap, "drift": drift, "blowdown": blowdown, "makeup": makeup},
        "makeup_fraction_of_circulation": makeup / rho, "evaporation_fraction_of_circulation": evap / rho,
        "cycles_of_concentration_effective": effective_N, "latent_heat_J_kg": h_fg, "density_kg_m3": rho,
        "warnings": warnings,
        "assumptions": [
            "Torre de enfriamiento con recirculación (no agua de un solo paso).",
            "Evaporación = fracción latente × carga térmica ÷ calor latente del agua a la temperatura media del circuito.",
            "Ciclos de concentración = repuesto ÷ (purga + arrastre); el m³ circulado y el repuesto se miden a la temperatura de "
            "suministro.",
            "pump_efficiency es la eficiencia global (bomba × motor) y pump_head_m la carga total del circuito.",
        ],
        "not_included": not_included,
    }


# ---------------------------------------------------------------------------------------------------- refrigeration
REFRIG_REQUIRED = ["name", "evaporator_T_K", "condenser_T_K", "second_law_efficiency"]


def refrigeration_level(level: dict, electricity_per_kWh: float, cooling_water_per_GJ: float | None = None) -> dict:
    """Mechanical refrigeration at one temperature level, per GJ removed: COP = second_law_efficiency × Carnot COP between
    the refrigerant's evaporating and condensing temperatures. Optional heat_rejection: a price per GJ rejected, or
    'cooling_water' to charge the condenser duty at the cooling-water cost computed in the same call."""
    where = f"refrigeration[{level.get('name', '?')}]"
    _known(level, REFRIG_REQUIRED + ["heat_rejection"], where)
    _need(level, REFRIG_REQUIRED, where)
    Te, Tc = float(level["evaporator_T_K"]), float(level["condenser_T_K"])
    eta = _frac(level["second_law_efficiency"], "second_law_efficiency", where, lo_open=True)
    if not (0.0 < Te < Tc):
        raise ValueError(f"{where}: se requiere 0 < evaporator_T_K < condenser_T_K (kelvin).")
    cop_carnot = Te / (Tc - Te)
    cop = eta * cop_carnot
    kwh_gj = KWH_PER_GJ / cop
    c_power = kwh_gj * electricity_per_kWh
    rej = level.get("heat_rejection")
    rej_gj = 1.0 + 1.0 / cop
    c_rej = 0.0
    not_included = ["Capital y mantenimiento del sistema de refrigeración"]
    if rej is None:
        not_included.append(f"Rechazo de calor del condensador ({rej_gj:.3f} GJ por GJ removido): da heat_rejection (precio por GJ "
                            f"o 'cooling_water') para cobrarlo")
    elif rej == "cooling_water":
        if cooling_water_per_GJ is None:
            raise ValueError(f"{where}: heat_rejection='cooling_water' requiere el bloque cooling_water en la misma llamada.")
        c_rej = rej_gj * cooling_water_per_GJ
    else:
        c_rej = rej_gj * float(rej)
    return {"name": level["name"], "evaporator_T_K": Te, "condenser_T_K": Tc, "cop_carnot": cop_carnot, "cop": cop,
            "power_kWh_per_GJ_removed": kwh_gj, "heat_rejected_GJ_per_GJ_removed": rej_gj,
            "cost_per_GJ_removed": c_power + c_rej, "breakdown_per_GJ_removed": {"power": c_power, "heat_rejection": c_rej},
            "assumptions": ["COP = eficiencia de segunda ley × COP de Carnot entre las temperaturas de evaporación y condensación "
                            "del refrigerante (da las del refrigerante, no las del proceso: incluye tú los acercamientos).",
                            "Compresión con motor eléctrico; la eficiencia de segunda ley incluye compresor, motor y pérdidas del "
                            "ciclo."],
            "not_included": not_included}


# ------------------------------------------------------------------------------------- route 2: Ulrich & Vasudevan
UV_SOURCE = ("Ulrich, G. D. y Vasudevan, P. T., «How to Estimate Utility Costs», Chemical Engineering, abril de 2006, pp. 66-69, "
             "Tabla 1: C_S,u = a·(CE PCI) + b·C_S,f")
UV_BASIS = ("process_module", "grass_roots")


def ulrich_vasudevan(utility: str, cepci: float, fuel_price_usd_GJ: float, basis: str = "process_module", **kw) -> dict:
    """Utility PRICE in USD by Ulrich & Vasudevan (2006), Table 1. utility: electricity (purchased), steam (pressure_barg,
    capacity_kg_s), cooling_water (capacity_m3_s), refrigeration (T_K, capacity_kW). basis: process_module | grass_roots.
    The fuel price is the boiler fuel for steam and the power-station fuel for the others (their note a), on HHV."""
    if basis not in UV_BASIS:
        raise ValueError(f"ulrich_vasudevan: basis debe ser uno de {UV_BASIS}.")
    pm = basis == "process_module"

    def rng(name: str, v: float, lo: float, hi: float) -> float:
        v = float(v)
        if not (lo <= v <= hi):
            raise ValueError(f"ulrich_vasudevan[{utility}]: {name} = {v} fuera del intervalo de la correlación ({lo} a {hi}).")
        return v

    if utility == "electricity":
        a, b, unit = 1.3e-4, 0.010, "USD/kWh"
        echo = {"case": "comprada a un tercero"}
    elif utility == "steam":
        p = rng("pressure_barg", kw["pressure_barg"], 1.0, 46.0)
        ms = rng("capacity_kg_s", kw["capacity_kg_s"], 0.06, 40.0)
        a, b, unit = (2.7e-5 if pm else 2.3e-5) * ms ** -0.9, 0.0034 * p ** 0.05, "USD/kg"
        echo = {"pressure_barg": p, "capacity_kg_s": ms}
    elif utility == "cooling_water":
        q = rng("capacity_m3_s", kw["capacity_m3_s"], 0.01, 10.0)
        a, b, unit = (0.0001 + 3.0e-5 / q) if pm else (0.00007 + 2.5e-5 / q), 0.003, "USD/m3"
        echo = {"capacity_m3_s": q}
    elif utility == "refrigeration":
        T = rng("T_K", kw["T_K"], 1e-9, 300.0)
        qc = rng("capacity_kW", kw["capacity_kW"], 1.0, 1000.0)
        a, b, unit = (0.6 if pm else 0.5) * qc ** -0.9 * T ** -3, 1.1e6 * T ** -5, "USD/kJ"
        echo = {"T_K": T, "capacity_kW": qc}
    else:
        raise ValueError("ulrich_vasudevan: utility debe ser electricity, steam, cooling_water o refrigeration.")
    price = a * float(cepci) + b * float(fuel_price_usd_GJ)
    return {"utility": utility, "price": price, "unit": unit, "a": a, "b": b, "cepci": float(cepci),
            "fuel_price_usd_GJ": float(fuel_price_usd_GJ), "basis": basis, "inflation_term": a * float(cepci),
            "fuel_term": b * float(fuel_price_usd_GJ), "source": UV_SOURCE, **echo}


def _second_route(spec: dict, steam_out: list[dict], cw_out: dict | None, refrig_out: list[dict], has_electricity: bool) -> dict:
    _known(spec, ["cepci", "basis", "boiler_fuel_price_usd_GJ", "power_fuel_price_usd_GJ", "steam_capacity_kg_s",
                  "cooling_water_capacity_m3_s", "refrigeration_capacity_kW"], "ulrich_vasudevan")
    _need(spec, ["cepci", "basis"], "ulrich_vasudevan")
    cepci, basis = spec["cepci"], spec["basis"]
    out: dict = {"source": UV_SOURCE, "currency": "USD", "basis": basis, "cepci": float(cepci),
                 "note": "Ruta independiente y empírica: es un PRECIO que incluye capital, mano de obra y mantenimiento de la planta "
                         "de servicios; la ruta 1 es solo costo de operación (combustible + agua + electricidad). No son la misma "
                         "cantidad: sirve de contraste de orden de magnitud, no de validación.",
                 "skipped": []}

    def power_fuel() -> float:
        _need(spec, ["power_fuel_price_usd_GJ"], "ulrich_vasudevan (electricidad, agua de enfriamiento y refrigeración: precio del "
                                                 "combustible de la central eléctrica)")
        return spec["power_fuel_price_usd_GJ"]

    def attempt(label: str, fn):
        try:
            return fn()
        except ValueError as e:
            out["skipped"].append(f"{label}: {e}")
            return None

    if has_electricity and spec.get("power_fuel_price_usd_GJ") is not None:
        out["electricity"] = ulrich_vasudevan("electricity", cepci, power_fuel(), basis)
    if steam_out:
        _need(spec, ["boiler_fuel_price_usd_GJ", "steam_capacity_kg_s"], "ulrich_vasudevan (vapor)")
        out["steam"] = []
        for s in steam_out:
            r = attempt(f"steam[{s['name']}]", lambda s=s: ulrich_vasudevan(
                "steam", cepci, spec["boiler_fuel_price_usd_GJ"], basis, pressure_barg=(s["pressure_Pa"] - P_ATM) / 1e5,
                capacity_kg_s=spec["steam_capacity_kg_s"]))
            if r:
                r.update(name=s["name"], price_per_tonne=1000.0 * r["price"],
                         price_per_GJ_delivered=1000.0 * r["price"] / s["heat_delivered_GJ_per_tonne"])
                out["steam"].append(r)
    if cw_out:
        _need(spec, ["cooling_water_capacity_m3_s"], "ulrich_vasudevan (agua de enfriamiento)")
        r = attempt("cooling_water", lambda: ulrich_vasudevan("cooling_water", cepci, power_fuel(), basis,
                                                              capacity_m3_s=spec["cooling_water_capacity_m3_s"]))
        if r:
            r["price_per_GJ_removed"] = r["price"] / cw_out["heat_removed_GJ_per_m3"]
            out["cooling_water"] = r
    if refrig_out:
        _need(spec, ["refrigeration_capacity_kW"], "ulrich_vasudevan (refrigeración)")
        out["refrigeration"] = []
        for lv in refrig_out:
            r = attempt(f"refrigeration[{lv['name']}]", lambda lv=lv: ulrich_vasudevan(
                "refrigeration", cepci, power_fuel(), basis, T_K=lv["evaporator_T_K"],
                capacity_kW=spec["refrigeration_capacity_kW"]))
            if r:
                r.update(name=lv["name"], price_per_GJ_removed=1e6 * r["price"])
                out["refrigeration"].append(r)
    return out


# ------------------------------------------------------------------------------------------------------- top level
def utility_costs(currency: str, prices: dict | None = None, reference: str | None = None, steam: list[dict] | None = None,
                  fired_heat_spec: dict | None = None, cooling_water_spec: dict | None = None,
                  refrigeration: list[dict] | None = None, electricity: dict | None = None,
                  hours_per_year: float | None = None, exchange_rates: dict | None = None,
                  annual_cost_levels: dict | None = None, ulrich_vasudevan_spec: dict | None = None,
                  today: str | None = None) -> dict:
    """Cost of every requested utility level from primary prices; see the module docstring and tools_utilities.utility_costs."""
    if not currency or not isinstance(currency, str):
        raise ValueError("currency: di en qué moneda quieres los costos (por ejemplo 'USD' o 'MXN').")
    currency = currency.upper()
    day = date.fromisoformat(today) if today else date.today()
    steam, refrigeration = list(steam or []), list(refrigeration or [])
    wants_electricity = electricity is not None
    if not (steam or fired_heat_spec or cooling_water_spec or refrigeration or wants_electricity):
        raise ValueError("Nada que calcular: da al menos uno de steam, fired_heat, cooling_water, refrigeration o electricity.")
    for group, label in ((steam, "steam"), (refrigeration, "refrigeration")):
        names = [lv.get("name") for lv in group]
        if len(set(names)) != len(names):
            raise ValueError(f"{label}: cada nivel necesita un name distinto.")
    resolved, warnings = resolve_prices(currency, prices, reference, exchange_rates, day)
    assumptions: list[str] = []
    not_included: list[str] = []
    out: dict = {"currency": currency}

    need_power = bool(cooling_water_spec or refrigeration or wants_electricity
                      or any(lv.get("auxiliary_power_kWh_per_tonne") is not None for lv in steam))
    needs: list[tuple[str, str]] = []
    if steam or fired_heat_spec:
        needs.append(("fuel_per_GJ", "el vapor y el calor a fuego directo"))
    if steam:
        needs.append(("boiler_makeup_water_per_m3", "el agua de repuesto de la caldera"))
    if cooling_water_spec:
        needs.append(("cooling_makeup_water_per_m3", "el agua de repuesto de la torre"))
    if need_power and "electricity_time_of_use_per_kWh" not in resolved:
        needs.append(("electricity_per_kWh", "la electricidad, el agua de enfriamiento y la refrigeración"))
    _require(resolved, needs, reference)

    elec = None
    if need_power:
        elec = electricity_cost(resolved, electricity, hours_per_year, reference)
        warnings += elec.pop("warnings")
        assumptions += elec.pop("assumptions")
        not_included += [f"electricidad: {x}" for x in elec.pop("not_included")]
        out["electricity"] = elec
    p_el = elec["per_kWh"] if elec else None

    if steam or fired_heat_spec:
        p_fuel = resolved["fuel_per_GJ"]["value"]
        basis = resolved["fuel_per_GJ"].get("fuel_basis")
        assumptions.append(f"Precio del combustible en base {basis}: las eficiencias de caldera y horno deben estar en esa misma "
                           f"base (con LHV la eficiencia es numéricamente mayor que con HHV).")
    if steam:
        out["steam"] = [steam_level(lv, p_fuel, resolved["boiler_makeup_water_per_m3"]["value"], p_el) for lv in steam]
    if fired_heat_spec:
        out["fired_heat"] = fired_heat(fired_heat_spec, p_fuel)
    if cooling_water_spec:
        out["cooling_water"] = cooling_water(cooling_water_spec, p_el, resolved["cooling_makeup_water_per_m3"]["value"])
        warnings += out["cooling_water"].pop("warnings")
    if refrigeration:
        cw_gj = out["cooling_water"]["cost_per_GJ_removed"] if "cooling_water" in out else None
        out["refrigeration"] = [refrigeration_level(lv, p_el, cw_gj) for lv in refrigeration]

    # one flat list of levels the user can point annual_cost at
    heating = {s["name"]: s["cost_per_GJ_delivered"] for s in out.get("steam", [])}
    if "fired_heat" in out:
        heating["fired_heat"] = out["fired_heat"]["cost_per_GJ_absorbed"]
    cooling = {r["name"]: r["cost_per_GJ_removed"] for r in out.get("refrigeration", [])}
    if "cooling_water" in out:
        cooling["cooling_water"] = out["cooling_water"]["cost_per_GJ_removed"]
    out["levels_per_GJ"] = {"heating": heating, "cooling": cooling}

    sel = dict(annual_cost_levels or {})
    chosen: dict = {}
    for kind, table in (("heating", heating), ("cooling", cooling)):
        if kind in sel:
            if sel[kind] not in table:
                raise ValueError(f"annual_cost_levels.{kind} = {sel[kind]!r} no es un nivel calculado; opciones: {sorted(table)}.")
            chosen[kind] = sel[kind]
        elif len(table) == 1:
            chosen[kind] = next(iter(table))
        elif len(table) > 1:
            warnings.append(f"Hay {len(table)} niveles de {'calentamiento' if kind == 'heating' else 'enfriamiento'} "
                            f"({sorted(table)}): elige cuál va a annual_cost con annual_cost_levels={{'{kind}': ...}}; el dict prices "
                            f"no trae esa clave.")
    to_usd = 1.0 if currency == "USD" else None
    if currency != "USD":
        fx = (exchange_rates or {}).get("USD")
        if fx:
            to_usd = 1.0 / float(fx)
    if to_usd is None:
        out["prices"] = None
        out["prices_note"] = (f"annual_cost y flowsheet_cost trabajan en USD (los costos de equipo están en USD). Los costos están en "
                              f"{currency}: da exchange_rates={{'USD': <{currency} por 1 USD>}} para obtener el dict prices en USD. "
                              f"Nunca se usa un tipo de cambio por defecto.")
    else:
        p: dict = {}
        if elec:
            p["electricity_usd_kWh"] = elec["per_kWh"] * to_usd
        if "heating" in chosen:
            p["heating_usd_GJ"] = heating[chosen["heating"]] * to_usd
        if "cooling" in chosen:
            p["cooling_usd_GJ"] = cooling[chosen["cooling"]] * to_usd
        out["prices"] = p
        out["prices_note"] = ("Listo para annual_cost / flowsheet_cost (USD). " +
                              (f"Convertido de {currency} con tu tipo de cambio ({exchange_rates['USD']} {currency} por USD). "
                               if currency != "USD" else "") +
                              f"Niveles usados: {chosen or 'ninguno'}.")
    out["annual_cost_levels"] = chosen

    for sec in ("steam", "refrigeration"):
        for lv in out.get(sec, []):
            assumptions += [a for a in lv.pop("assumptions") if a not in assumptions]
            not_included += [f"{sec}: {x}" for x in lv.pop("not_included") if f"{sec}: {x}" not in not_included]
    for sec in ("fired_heat", "cooling_water"):
        if sec in out:
            assumptions += out[sec].pop("assumptions")
            not_included += [f"{sec}: {x}" for x in out[sec].pop("not_included")]
    if ulrich_vasudevan_spec:
        out["second_route"] = _second_route(ulrich_vasudevan_spec, out.get("steam", []), out.get("cooling_water"),
                                            out.get("refrigeration", []), elec is not None)
    out["price_sources"] = resolved
    out["warnings"] = warnings
    out["assumptions"] = assumptions
    out["not_included"] = not_included
    out["inputs"] = {"currency": currency, "prices": prices, "reference": reference, "steam": steam, "fired_heat": fired_heat_spec,
                     "cooling_water": cooling_water_spec, "refrigeration": refrigeration, "electricity": electricity,
                     "hours_per_year": hours_per_year, "exchange_rates": exchange_rates, "annual_cost_levels": annual_cost_levels,
                     "ulrich_vasudevan": ulrich_vasudevan_spec, "today": day.isoformat()}
    out["method"] = {"route": "balances de energía y de agua con IAPWS-95 (CoolProp)", "module": "packages/costing/utilities.py",
                     "validated_against": "U.S. DOE Steam Tip Sheet #15 (DOE/GO-102012-3391, enero 2012), Tabla 1 y ejemplo; "
                                          "Ulrich y Vasudevan 2006 (ruta 2)"}
    return out
