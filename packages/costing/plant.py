"""Whole-plant economics orchestrator (bloque M26_W): from "equipment list + material/utility flows" to capital, operating
cost, cash flow, NPV, IRR and minimum selling price in ONE call, proposing every missing ECONOMIC input from a sourced bank.

Nothing new is solved here: the capital levels come from `packages.costing.economics.capital_investment`, the yearly cash
flow, NPV, IRR and break-even revenue from `cash_flow_analysis` (same module, cross-checked there against BioSTEAM), and the
proposed values from `packages.costing.reference_prices` (bloque M26_P). This module (1) reconciles units and currencies,
(2) resolves each input in the order caller → bank → NOT FOUND (never filled), (3) records where every number came from,
and (4) ranks the assumed inputs by their one-at-a-time effect on the NPV. All money is computed in USD (the engines' unit);
`currency` only adds a converted headline.

Published factor set used when the caller gives none (ONE set, read on 2026-11-16 in Humbird et al. 2011, NREL/TP-5100-47764,
https://docs.nlr.gov/docs/fy11osti/47764.pdf, Tables 25, 27, 28, 31 and 33), mapped onto `capital_investment`:
  site_preparation   = 9 % of ISBL installed cost («site development»)                       exact name
  service_facilities = 4.5 % of ISBL («additional piping» to storage and utilities)           approximate name
  allocated_costs    = 4 % of ISBL («warehouse»)                                              approximate name
  contingency_and_fee = 60 % of total direct cost (prorateable 10 + field expenses 10 + home office and construction 20 +
                        project contingency 10 + other costs 10)                              five items lumped into one
  land = 0 (Humbird gives a fixed amount, not a factor: pass basis['land_usd']), royalties = 0 (none in Humbird),
  startup = 0 (inside «other costs»), working capital = 5 % of the fixed capital investment (FCI = C_TDC).
The ISBL percentages are multiplied by the ISBL share of the installed cost (equipment items with `isbl: False` are
outside battery limits) to become fractions of C_TBM. Fixed operating costs of the same report: labour burden 90 % of
salaries (it covers plant overhead), maintenance 3 % of ISBL installed cost, property insurance and tax 0.7 % of FCI.
Humbird's percentages were published for an installed cost built with ITS installation factors (Table 25, 1.0-3.1);
applying them on Guthrie/Seider bare-module costs (F_BM 3-4) is a mix of sets and is reported as such, never silently.

Operator head-count when the caller gives none: Alkhayat and Gerrard's correlation N_OL = (6.29 + 31.7·P² + 0.23·N_np)^0.5
operators per shift and 1095/245 ≈ 4.5 operators per position, both rounded up, read on 2026-11-16 in Verret, Qiao and
Barghout, «Foundations of Chemical and Biological Engineering I», section 6.4 (LibreTexts, CC BY-NC-SA 4.0).

Mexico: the tax rate proposed is the ISR alone (LISR art. 9, 30 %). The PTU (10 %) is NOT folded into the rate: its base is
the «renta gravable» without loss carry-forward (LISR art. 9, LFT arts. 120 and 128) and since 2021 it is capped at three
months of each worker's salary or the three-year average received (LFT art. 127-VIII), so for a capital-intensive plant it
behaves like a payroll cost, not like a 10 % tax. Both bounds are reported in `ptu` (laws read on 2026-11-16 at
https://www.diputados.gob.mx/LeyesBiblio/pdf/LISR.pdf and .../LFT.pdf).
"""
from __future__ import annotations

import copy
import math
import re
import unicodedata
from contextlib import contextmanager
from datetime import date

from packages.costing import economics as EC
from packages.costing import reference_prices as RP
from packages.costing.sslw import CEPCI_IDAES

FLAG = RP.FLAG
DATE_READ = "2026-11-16"
HUMBIRD = {"source": "Humbird et al. (2011), Process Design and Economics for Biochemical Conversion of Lignocellulosic Biomass to "
                     "Ethanol, NREL/TP-5100-47764", "url": "https://docs.nlr.gov/docs/fy11osti/47764.pdf",
           "period": {"start": "2011-05-01", "end": "2011-05-31"}, "date_read": DATE_READ}
# Table 27 (additional direct costs, fractions of the ISBL installed cost; indirect costs, fractions of the total direct cost)
HUMBIRD_DIRECT_OF_ISBL = {"site_preparation": (0.09, "«site development» 9 % del costo instalado ISBL"),
                          "service_facilities": (0.045, "«additional piping» 4.5 % del ISBL (nombre aproximado: tubería hacia "
                                                        "almacenamiento y servicios)"),
                          "allocated_costs": (0.04, "«warehouse» 4 % del ISBL (nombre aproximado: almacén)")}
HUMBIRD_INDIRECT_OF_TDC = {"prorateable_expenses": 0.10, "field_expenses": 0.10, "home_office_and_construction_fee": 0.20,
                           "project_contingency": 0.10, "other_costs_startup_permits": 0.10}
HUMBIRD_WORKING_CAPITAL_OF_FCI = 0.05
HUMBIRD_FIXED = {"labour_burden_fraction": (0.90, "fracción de los salarios", "Tabla 31: «labor burden (90 %)», cubre gastos "
                                                  "generales de planta, prestaciones, seguridad, servicios"),
                 "maintenance_fraction_of_isbl": (0.03, "fracción del costo instalado ISBL por año", "Tabla 31: «maintenance 3.0 % of "
                                                                                                     "ISBL»"),
                 "insurance_tax_fraction_of_fci": (0.007, "fracción del capital fijo (FCI) por año",
                                                   "Tabla 31: «property insurance 0.7 % of FCI» (seguro e impuesto predial)")}
# Table 25: installed cost = purchased cost × multiplier. Offered to the caller (pass it as F_BM); never applied silently.
HUMBIRD_INSTALLATION_FACTORS = {
    "agitators_carbon_steel": 1.6, "agitators_stainless_steel": 1.5, "boiler": 1.8, "compressors_motor_driven": 1.6,
    "cooling_tower": 1.5, "distillation_columns_stainless_steel": 2.4, "heat_exchangers_shell_tube_stainless_steel": 2.2,
    "heat_exchangers_plate_frame_stainless_steel": 1.8, "heat_exchangers_air_cooled": 2.8, "inline_mixers": 1.0,
    "skidded_equipment": 1.8, "solids_handling_equipment_incl_filters": 1.7, "pressure_vessels_carbon_steel": 3.1,
    "pressure_vessels_stainless_steel": 2.0, "pretreatment_reactor_system": 1.5, "pumps_stainless_steel": 2.3,
    "pumps_carbon_steel": 3.1, "tanks_field_erected_carbon_steel": 1.7, "tanks_field_erected_stainless_steel": 1.5,
    "tanks_storage_plastic": 3.0, "tanks_storage_carbon_steel": 2.6, "tanks_storage_stainless_steel": 1.8, "turbogenerator": 1.8}
LABOUR_CORRELATION = {
    "a": 6.29, "b": 31.7, "c": 0.23, "shifts_per_year": 1095, "shifts_per_operator": 245,
    "source": "Verret, Qiao y Barghout, Foundations of Chemical and Biological Engineering I, sección 6.4 «Cost of Operating "
              "Labour» (LibreTexts, CC BY-NC-SA 4.0): correlación de Alkhayat y Gerrard (AACE Transactions, 1984)",
    "url": "https://eng.libretexts.org/Bookshelves/Chemical_Engineering/Foundations_of_Chemical_and_Biological_Engineering_I_"
           "(Verret_Qiao_Barghout)/06%3A_Process_Economics/6.04%3A_Cost_of_Operating_Labour",
    "period": {"start": "2024-05-22", "end": "2024-05-22"}, "date_read": DATE_READ}
MX_LAW = {"lisr": "https://www.diputados.gob.mx/LeyesBiblio/pdf/LISR.pdf", "lft": "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFT.pdf"}
NO_LABOUR_STEP = ("pump", "vessel_vertical", "vessel_horizontal", "storage_tank", "platforms_ladders")   # pumps and vessels

_MASS = {"kg": 1.0, "g": 1e-3, "t": 1000.0, "tonne": 1000.0, "tonelada": 1000.0, "lb": 0.45359237, "short_ton": 907.18474,
         "ton_us": 907.18474, "st": 907.18474}                                                       # kg per unit (exact)
_VOLUME = {"m3": 1.0, "l": 1e-3, "gal": 3.785411784e-3, "bbl": 0.158987294928}                       # m3 per unit (US gallon)
_ENERGY = {"gj": 1.0, "mj": 1e-3, "kj": 1e-6, "kwh": 3.6e-3, "mwh": 3.6, "mmbtu": 1.05505585262}     # GJ per unit
_TIME_H = {"s": 1.0 / 3600.0, "min": 1.0 / 60.0, "h": 1.0, "hr": 1.0, "d": 24.0, "day": 24.0}        # hours per unit
_YEAR = ("y", "yr", "year", "a", "ano", "año")
_POWER = {"kw": "kwh", "mw": "mwh"}
_BASE_UNIT = {"mass": "kg", "volume": "m3", "energy": "GJ"}
_DEFAULT_PRICE_UNIT = {"mass": "t", "volume": "m3", "energy": "GJ"}
ROLES = ("raw_materials", "products", "utilities")
ROLE_ES = {"raw_materials": "materia prima", "products": "producto", "utilities": "servicio"}
BLOCKED = ["npv", "irr", "payback", "minimum_selling_price"]
REL_TOL = 1e-9

METHOD = ("Orquestación sin física nueva: capital con packages.costing.economics.capital_investment (estructura de Seider, F_BM "
          "por equipo); flujo de caja real y sin préstamos, VPN, TIR e ingreso de equilibrio con cash_flow_analysis (brentq de "
          "SciPy, contrastado ahí con las funciones de BioSTEAM); precio mínimo de venta = (ingreso de equilibrio − ingreso de "
          "los demás productos) / cantidad anual del producto principal, verificado volviendo a correr el flujo a ese precio; "
          "valores propuestos del banco fechado de packages.costing.reference_prices y de un único conjunto publicado de factores "
          "(Humbird et al. 2011); sensibilidad de un factor a la vez sobre cada valor supuesto.")
NOT_INCLUDED = [
    "Financiamiento: el flujo de caja es 100 % capital propio, sin préstamos ni intereses (la convención de NREL de 40 % capital "
    "propio y préstamo a 10 años al 8 % no se modela; en el caso de Humbird 2011 eso mueve el precio mínimo ≈ 3 %).",
    "Inflación y escalamiento de precios: términos reales y precios constantes durante toda la vida de la planta.",
    "IVA, aranceles, fletes hasta tu planta, impuestos estatales sobre nómina y permisos.",
    "Varias clases de activos con distinta depreciación (se usa UNA sola tabla para todo el capital depreciable).",
    "Valor de rescate, desmantelamiento y costos periódicos (cambio de catalizador o de mangas cada n años), salvo que los "
    "des anualizados con annual_cost.",
    "Límite de diez ejercicios para amortizar pérdidas fiscales (LISR art. 57): las pérdidas se trasladan sin caducar.",
    "Ingeniería de detalle del capital: es una estimación por factores (clase 4-5), no un presupuesto.",
]


# --------------------------------------------------------------------------- small helpers

@contextmanager
def _process_state_kept():
    """The engines chained here import BioSTEAM (F_BM table, cash-flow cross-check); thermosteam flips thermo's global
    RAISE_PROPERTY_CALCULATION_ERROR flags on import. Leave them as they were found."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    saved = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    try:
        yield
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = saved


def _clean(x):
    """JSON-safe deep copy (numpy scalars → Python, non-finite floats → None, tuples → lists)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if hasattr(x, "item") and not isinstance(x, (int, float)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (int, float)):
        return x
    return str(x)


def _num(value, what: str, positive: bool = False, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} = {value!r}: debe ser un número.") from None
    if not math.isfinite(v) or (positive and v <= 0) or (minimum is not None and v < minimum) or (maximum is not None and v > maximum):
        lim = "> 0" if positive else f"entre {minimum} y {maximum}" if maximum is not None else f"≥ {minimum}"
        raise ValueError(f"{what} = {value!r}: fuera de rango (debe ser {lim}).")
    return v


def _norm(text: str) -> str:
    """Lower case, no accents, alphanumerics separated by single spaces (to match names of prices and items)."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _unit_key(text: str) -> str:
    return str(text).strip().replace("³", "3").replace("^", "").replace(" ", "").lower()


def _quantity_unit(q: str, where: str) -> tuple[str, float]:
    """(dimension, base units per 1 unit) of a quantity unit: mass in kg, volume in m3, energy in GJ."""
    k = _unit_key(q)
    if k in ("ton", "tons", "tn", "ton."):
        raise ValueError(f"{where}: «{q}» es ambiguo. Usa 't' (tonelada métrica, 1000 kg) o 'short_ton' (tonelada corta de EE. UU., "
                         "907.18474 kg).")
    for dim, table in (("mass", _MASS), ("volume", _VOLUME), ("energy", _ENERGY)):
        if k in table:
            return dim, table[k]
    raise ValueError(f"{where}: unidad «{q}» no reconocida. Masa: {sorted(_MASS)}; volumen: {sorted(_VOLUME)}; energía: "
                     f"{sorted(_ENERGY)}.")


def _flow_unit(unit: str, where: str) -> tuple[str, float, bool]:
    """(dimension, base units per hour — or per year when annual — for a flow of 1, annual?)."""
    k = _unit_key(unit)
    if k in _POWER:
        dim, per = _quantity_unit(_POWER[k], where)
        return dim, per, False
    if "/" not in k:
        raise ValueError(f"{where}: la unidad de flujo «{unit}» debe ser cantidad/tiempo (kg/h, t/h, m3/h, kWh/h, GJ/h, t/y…) o "
                         "una potencia (kW, MW).")
    num, den = k.rsplit("/", 1)
    dim, per = _quantity_unit(num, where)
    if den in _YEAR:
        return dim, per, True
    if den not in _TIME_H:
        raise ValueError(f"{where}: tiempo «{den}» no reconocido en «{unit}» (usa s, min, h, d o y).")
    return dim, per / _TIME_H[den], False


def _price_unit(unit: str, where: str) -> tuple[str, str, float, str]:
    """(currency, dimension, base units per priced unit, quantity unit as written) of 'USD/t', 'MXN/kWh', 'USD/gal'…"""
    text = str(unit).strip().replace(" ", "")
    if "/" not in text:
        raise ValueError(f"{where}: la unidad de precio «{unit}» debe ser moneda/cantidad (USD/t, MXN/kWh, USD/m3, USD/gal…).")
    cur, q = text.split("/", 1)
    dim, per = _quantity_unit(q, where)
    return cur.upper(), dim, per, q


def _age_days(period_end: str, today: date) -> int:
    return (today - date.fromisoformat(period_end)).days


def _record(key: str, label: str, value, unit: str, src: dict, today: date, note: str = "", **extra) -> dict:
    """One assumed input: where the number came from, in the shape shown to the user."""
    return {"input": key, "label": label, "value": value, "unit": unit, "source": src["source"], "url": src["url"],
            "period": f"{src['period']['start']} a {src['period']['end']}", "date_read": src["date_read"],
            "age_days": src.get("age_days", _age_days(src["period"]["end"], today)), "stale": bool(src.get("stale", False)),
            "low": extra.pop("low", None), "high": extra.pop("high", None), "flag": FLAG, "note": note, **extra}


def _bank(v: dict, key: str, label: str, value=None, unit: str | None = None, note: str | None = None, **extra) -> dict:
    """Assumed-input record from a value of bloque P's bank (already aged and flagged by it)."""
    src = {"source": v["source"], "url": v["url"], "period": v["period"], "date_read": v["date_read"],
           "age_days": v.get("age_days"), "stale": v.get("stale", False)}
    rec = _record(key, label, v["value"] if value is None else value, unit or v["unit"], src, date.today(),
                  note if note is not None else f"{v.get('basis', '')}. {v.get('notes', '')}".strip(),
                  bank_id=v.get("id"), region_used=v.get("region", v.get("region_used")), **extra)
    return rec


# --------------------------------------------------------------------------- resolution of the inputs

class _Ctx:
    """What the resolution collects: assumed and user inputs, what is missing, messages, and the FX rates used."""

    def __init__(self, call: dict):
        self.call = call
        self.today = date.fromisoformat(call["today"]) if call.get("today") else date.today()
        self.region = str(call["region"]).strip().upper()
        self.chain = RP.region_chain(self.region)
        self.assumed: list[dict] = []
        self.user: list[dict] = []
        self.missing: list[dict] = []
        self.warnings: list[str] = []
        self.assumptions: list[str] = []
        self.not_included: list[str] = list(NOT_INCLUDED)
        self.scalars: dict[str, float] = {}
        self.fx_seen: set[str] = set()
        self._basis = None

    def design_basis(self) -> dict:
        if self._basis is None:
            self._basis = RP.design_basis(self.region, self.call.get("plant_type"), self.today)
        return self._basis

    def add_user(self, key: str, value, unit: str = "") -> None:
        self.user.append({"input": key, "value": value, "unit": unit})

    def add_missing(self, key: str, reason: str, blocks: list[str], **extra) -> None:
        self.missing.append({"input": key, "reason": reason, "blocks": blocks, **extra})

    def fx(self, cur: str) -> float | None:
        """USD per 1 unit of `cur`; registers the scalar 'fx.<cur>' (assumed from the bank's dated FIX, or the caller's)."""
        cur = cur.upper()
        if cur == "USD":
            return 1.0
        key = f"fx.{cur}"
        if key in self.scalars:
            return self.scalars[key]
        if cur in self.fx_seen:
            return None
        self.fx_seen.add(cur)
        rates = self.call.get("exchange_rates") or {}
        if rates.get(cur) is not None:
            self.scalars[key] = _num(rates[cur], f"exchange_rates['{cur}'] (USD por 1 {cur})", positive=True)
            self.add_user(f"exchange_rates.{cur}", self.scalars[key], f"USD por 1 {cur}")
            return self.scalars[key]
        if cur == "MXN":
            fix = RP.lookup("fx_fix_mxn_usd", "MX", None, None, self.today)
            if fix.get("found"):
                self.scalars[key] = 1.0 / fix["value"]
                self.assumed.append(_bank(fix, f"exchange_rates.{cur}", "tipo de cambio (USD por 1 MXN)", 1.0 / fix["value"],
                                          "USD por 1 MXN", f"FIX de Banco de México: {fix['value']} MXN por USD.", scalar=key))
                return self.scalars[key]
        self.add_missing(f"exchange_rates.{cur}", f"No hay tipo de cambio {cur}→USD: da exchange_rates={{'{cur}': <USD por 1 {cur}>}}.",
                         BLOCKED)
        return None


def _scale_equipment(call: dict, ctx: _Ctx) -> tuple[list[dict], dict]:
    """Equipment at one cost basis: items carrying `cost_year` or `cepci` are rescaled to the target index."""
    equipment = call.get("equipment")
    if not isinstance(equipment, list) or not equipment:
        raise ValueError("equipment: da la lista de equipos [{name, type o F_BM, purchase_cost_usd, base_purchase_cost_usd}] que "
                         "devuelven equipment_cost y las herramientas de dimensionamiento; sin ella no hay inversión que calcular.")
    year, cepci = call.get("year"), call.get("cepci")
    target = _num(cepci, "cepci", positive=True) if cepci is not None else CEPCI_IDAES.get(year)
    label = (f"CEPCI {target} (dado por el usuario)" if cepci is not None else f"CEPCI {target} ({year}, tabla de IDAES)"
             if target else f"año {year} sin índice")
    out, untouched, names = [], [], set()
    for i, e in enumerate(equipment):
        if not isinstance(e, dict) or "purchase_cost_usd" not in e:
            raise ValueError(f"equipment[{i}]: falta purchase_cost_usd (costo de compra en USD) del equipo; es la base del capital.")
        e = dict(e)
        e["name"] = str(e.get("name") or f"equipo {i + 1}")
        if e["name"] in names:
            raise ValueError(f"equipment: el nombre «{e['name']}» está repetido; cada equipo necesita un nombre único.")
        names.add(e["name"])
        if e.get("type") is None and e.get("F_BM") is None:
            raise ValueError(f"equipment «{e['name']}»: da type (clave de bare_module_factors) o F_BM (factor de instalación con su "
                             "fuente); sin él no hay costo instalado.")
        own = e.get("cepci") if e.get("cepci") is not None else CEPCI_IDAES.get(e.get("cost_year")) if e.get("cost_year") else None
        if e.get("cost_year") is not None and own is None:
            raise ValueError(f"equipment «{e['name']}»: no hay CEPCI de {e['cost_year']} en la tabla (2015-2023); da `cepci` del equipo.")
        if own is not None:
            if target is None:
                raise ValueError(f"No hay CEPCI de {year} en la tabla de IDAES (2015-2023): da cepci=<índice> para llevar «{e['name']}» "
                                 "a ese año.")
            ratio = target / _num(own, f"cepci de «{e['name']}»", positive=True)
            for k in ("purchase_cost_usd", "base_purchase_cost_usd", "platforms_ladders_purchase_cost_usd"):
                if e.get(k) is not None:
                    e[k] = float(e[k]) * ratio
            e["basis"] = f"{label}; reescalado ×{ratio:.4f} desde CEPCI {own}"
            e["cepci"], e["cost_year"] = target, year if cepci is None else e.get("cost_year")
        else:
            untouched.append(e["name"])
        e["isbl"] = bool(e.get("isbl", True))
        out.append(e)
    if untouched and len(untouched) < len(out):
        ctx.warnings.append(f"Equipos sin cost_year ni cepci, NO reescalados (se suponen ya en {label}): {untouched}.")
    elif untouched:
        ctx.assumptions.append(f"Los costos de compra se toman tal cual (ningún equipo trae cost_year ni cepci): se suponen todos "
                               f"en la misma base, {label}.")
    return out, {"year": year, "cepci": target, "label": label}


def _resolve_capital(call: dict, ctx: _Ctx, equipment: list[dict]) -> dict:
    """The eight factors of capital_investment plus location factor and land: caller first, else Humbird's set."""
    given = dict(call.get("capital_factors") or {})
    unknown = [k for k in given if k not in EC.CAPITAL_FACTORS]
    if unknown:
        raise ValueError(f"capital_factors: claves desconocidas {unknown}. Las válidas son {list(EC.CAPITAL_FACTORS)} (fracciones).")
    basis = call.get("basis") or {}
    zero = {k: 0.0 for k in EC.CAPITAL_FACTORS}
    try:
        probe = EC.capital_investment([{k: v for k, v in e.items() if k != "isbl"} for e in equipment], zero, 1.0)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"equipment: capital_investment no acepta la lista ({exc}). Cada equipo necesita purchase_cost_usd ≥ 0, "
                         "type (clave de bare_module_factors) o F_BM, y base_purchase_cost_usd ≤ purchase_cost_usd si lo das.") from None
    cbm = {r["name"]: r["bare_module_cost_usd"] for r in probe["equipment"]}
    total = sum(cbm.values())
    isbl = sum(cbm[e["name"]] for e in equipment if e["isbl"])
    share = isbl / total if total > 0 else 1.0
    if all(e["isbl"] for e in equipment):
        ctx.assumptions.append("Ningún equipo está marcado isbl=False: todo el costo instalado cuenta como ISBL (dentro de límites "
                               "de batería) para sitio, almacén, tubería adicional y mantenimiento. Marca isbl=False en servicios, "
                               "almacenamiento, caldera y tratamiento de aguas.")
    for k in EC.CAPITAL_FACTORS:
        if k in given:
            ctx.scalars[f"capital_factors.{k}"] = _num(given[k], f"capital_factors['{k}']", minimum=0.0, maximum=2.0)
            ctx.add_user(f"capital_factors.{k}", ctx.scalars[f"capital_factors.{k}"], "fracción")
    table = "Tablas 27 y 28"
    for k, (frac, text) in HUMBIRD_DIRECT_OF_ISBL.items():
        if k not in given:
            ctx.scalars[f"capital_factors.{k}"] = frac * share
            ctx.assumed.append(_record(f"capital_factors.{k}", k.replace("_", " "), frac * share, "fracción de C_TBM",
                                       HUMBIRD, ctx.today, f"{table}: {text} × fracción ISBL del costo instalado ({share:.3f}).",
                                       scalar=f"capital_factors.{k}"))
    if "contingency_and_fee" not in given:
        tot = sum(HUMBIRD_INDIRECT_OF_TDC.values())
        ctx.scalars["capital_factors.contingency_and_fee"] = tot
        ctx.assumed.append(_record("capital_factors.contingency_and_fee", "indirectos, contingencia y honorarios", tot,
                                   "fracción de C_DPI (costo directo total)", HUMBIRD, ctx.today,
                                   f"{table}: suma de los cinco indirectos de Humbird (prorrateables 10 %, gastos de campo 10 %, "
                                   "oficina central y construcción 20 %, contingencia 10 %, otros —arranque, permisos— 10 %), "
                                   "agrupados en el único factor que tiene capital_investment.", scalar="capital_factors.contingency_and_fee"))
    for k, why in (("land", "Humbird da el terreno como monto fijo (1.8 MM USD de 2007 para su planta), no como factor: da "
                            "basis['land_usd'] o capital_factors['land']"),
                   ("royalties", "Humbird no tiene regalías"), ("startup", "el arranque ya está en «otros costos» (10 % del costo "
                                                                           "directo), dentro de contingency_and_fee")):
        if k not in given:
            ctx.scalars[f"capital_factors.{k}"] = 0.0
            ctx.assumptions.append(f"capital_factors['{k}'] = 0: {why}.")
    wc_basis = "C_TPI"
    if "working_capital" not in given:
        wc_basis = "FCI"
        v = ctx.design_basis()["values"].get("working_capital_fraction")
        if v is None:
            ctx.scalars["capital_factors.working_capital"] = 0.0
            ctx.add_missing("capital_factors.working_capital", "El banco no tiene fracción de capital de trabajo: dala.", [])
        else:
            ctx.scalars["capital_factors.working_capital"] = v["value"]
            ctx.assumed.append(_bank(v, "capital_factors.working_capital", "capital de trabajo", unit="fracción del capital fijo "
                                     "(FCI = C_TDC)", scalar="capital_factors.working_capital"))
    if given and len(given) < len(EC.CAPITAL_FACTORS):
        ctx.warnings.append(f"MEZCLA DE CONJUNTOS: {sorted(given)} son tuyos y el resto de capital_factors viene de Humbird et al. "
                            "(2011). Revisa que no dupliquen ni omitan conceptos.")
    if "location_factor" in basis:
        ctx.scalars["basis.location_factor"] = _num(basis["location_factor"], "basis['location_factor']", positive=True)
        ctx.add_user("basis.location_factor", ctx.scalars["basis.location_factor"], "fracción")
    else:
        v = ctx.design_basis()["values"].get("location_factor_mx") if "MX" in ctx.chain else None
        if v is None:
            ctx.scalars["basis.location_factor"] = 1.0
            ctx.assumptions.append(f"Sin factor de localización para {ctx.region} en el banco: se usa 1.0 (costos de EE. UU. sin "
                                   "ajustar). Da basis['location_factor'] si tienes uno.")
        else:
            ctx.scalars["basis.location_factor"] = v["value"]
            ctx.assumed.append(_bank(v, "basis.location_factor", "factor de localización", unit="fracción sobre C_TBM",
                                     low=v.get("low"), high=v.get("high"), scalar="basis.location_factor"))
    ctx.scalars["basis.land_usd"] = _num(basis.get("land_usd", 0.0), "basis['land_usd']", minimum=0.0)
    if "land_usd" in basis:
        ctx.add_user("basis.land_usd", ctx.scalars["basis.land_usd"], "USD")
    seider = sorted({e["name"] for e in equipment if e.get("F_BM") is None})
    humbird_caps = [k for k in ("site_preparation", "service_facilities", "allocated_costs", "contingency_and_fee") if k not in given]
    if seider and humbird_caps:
        ctx.warnings.append(
            f"MEZCLA DE CONJUNTOS EN EL CAPITAL: {len(seider)} equipos usan F_BM de módulo desnudo de Seider/Guthrie (vía `type`, "
            "valores 3-4) y los factores de sitio e indirectos son de Humbird et al. (2011), publicados para aplicarse sobre un "
            "costo instalado con SUS factores de instalación (Tabla 25: 1.0-3.1; promedio de su planta 1.50). La combinación no "
            "es un conjunto publicado consistente y tiende a sobrestimar la inversión. Para un conjunto consistente da F_BM por "
            "equipo con los factores de instalación de Humbird (published_factor_set.installation_factors) o tus propios "
            "capital_factors.")
    return {"working_capital_basis": wc_basis, "isbl_share": share}


_BASIS_FROM_BANK = [      # (key, bank id, Spanish label, minimum, maximum)
    ("on_stream_hours", "on_stream_hours", "horas de operación por año", 1.0, 8784.0),
    ("plant_life_years", "plant_life", "vida de la planta", 1.0, 100.0),
    ("discount_rate", "discount_rate_nrel", "tasa de descuento", -0.49, 0.99),
    ("startup_years", "startup_time", "duración del arranque", 0.0, 50.0),
]
_STARTUP_FRACTIONS = [("startup_revenue_fraction", "ingresos durante el arranque"), ("startup_variable_cost_fraction", "costos variables"),
                      ("startup_fixed_cost_fraction", "costos fijos")]
_BASIS_KEYS = ("on_stream_hours", "plant_life_years", "income_tax_rate", "discount_rate", "depreciation", "construction_schedule",
               "startup_years", "startup_revenue_fraction", "startup_variable_cost_fraction", "startup_fixed_cost_fraction",
               "location_factor", "land_usd", "maintenance_fraction_of_isbl", "insurance_tax_fraction_of_fci")


def _resolve_basis(call: dict, ctx: _Ctx) -> dict:
    basis = dict(call.get("basis") or {})
    unknown = [k for k in basis if k not in _BASIS_KEYS]
    if unknown:
        raise ValueError(f"basis: claves desconocidas {unknown}. Las válidas son {list(_BASIS_KEYS)}.")
    out: dict = {}
    for key, bank_id, label, lo, hi in _BASIS_FROM_BANK:
        if key in basis:
            ctx.scalars[f"basis.{key}"] = _num(basis[key], f"basis['{key}']", minimum=lo, maximum=hi)
            ctx.add_user(f"basis.{key}", ctx.scalars[f"basis.{key}"])
            continue
        v = ctx.design_basis()["values"].get(bank_id)
        if v is None:
            ctx.add_missing(f"basis.{key}", f"El banco no tiene «{label}» para {ctx.region}: dalo en basis['{key}'].", BLOCKED)
        else:
            ctx.scalars[f"basis.{key}"] = float(v["value"])
            ctx.assumed.append(_bank(v, f"basis.{key}", label, scalar=f"basis.{key}"))
    startup = ctx.design_basis()["values"].get("startup_time") or {}
    table = {r.get("concepto"): r.get("fraccion") for r in startup.get("table", [])}
    for key, concept in _STARTUP_FRACTIONS:
        if key in basis:
            ctx.scalars[f"basis.{key}"] = _num(basis[key], f"basis['{key}']", minimum=0.0, maximum=2.0)
            ctx.add_user(f"basis.{key}", ctx.scalars[f"basis.{key}"], "fracción")
        elif ctx.scalars.get("basis.startup_years") == 0.0:
            ctx.scalars[f"basis.{key}"] = 1.0          # no start-up period: the fraction has no effect, so it is not an assumption
        elif table.get(concept) is not None:
            ctx.scalars[f"basis.{key}"] = float(table[concept])
            ctx.assumed.append(_bank(startup, f"basis.{key}", f"arranque: {concept}", float(table[concept]), "fracción del valor a "
                                     "régimen", scalar=f"basis.{key}"))
        else:
            ctx.scalars[f"basis.{key}"] = 1.0
            ctx.assumptions.append(f"basis['{key}'] = 1.0 (sin convención en el banco: arranque sin penalización en ese rubro).")
    # income tax: ISR only (see module docstring for why the PTU is not folded in)
    out["tax_from_bank"] = False
    if "income_tax_rate" in basis:
        ctx.scalars["basis.income_tax_rate"] = _num(basis["income_tax_rate"], "basis['income_tax_rate']", minimum=0.0, maximum=0.99)
        ctx.add_user("basis.income_tax_rate", ctx.scalars["basis.income_tax_rate"], "fracción")
    else:
        v = ctx.design_basis()["values"].get("isr_corporate_mx") if "MX" in ctx.chain else None
        if v is None:
            ctx.add_missing("basis.income_tax_rate", f"El banco solo tiene el impuesto sobre la renta de México; para {ctx.region} "
                            "da basis['income_tax_rate'] (fracción).", BLOCKED)
        else:
            out["tax_from_bank"] = True
            ctx.scalars["basis.income_tax_rate"] = float(v["value"])
            ctx.assumed.append(_bank(v, "basis.income_tax_rate", "impuesto sobre la renta (ISR, sin PTU)", unit="fracción del "
                                     "resultado fiscal", scalar="basis.income_tax_rate"))
    # depreciation: one schedule for all the depreciable capital
    if "depreciation" in basis:
        out["depreciation"] = str(basis["depreciation"])
        try:
            EC.depreciation_schedule(out["depreciation"])
        except ValueError:
            raise ValueError(f"basis['depreciation'] = {out['depreciation']!r}: usa 'MACRS3|5|7|10|15|20', 'SL<n>', 'DDB<n>' o "
                             "'SYD<n>' (n = años).") from None
        ctx.add_user("basis.depreciation", out["depreciation"])
    else:
        sug = ctx.design_basis().get("suggested_depreciation") if "MX" in ctx.chain else None
        sid = (sug or {}).get("suggested_id") or ("depreciation_other_mx" if sug else None)
        v = ctx.design_basis()["values"].get(sid) if sid else None
        if v is None:
            out["depreciation"] = None
            ctx.add_missing("basis.depreciation", f"El banco solo tiene las tasas de depreciación de la LISR de México; para "
                            f"{ctx.region} da basis['depreciation'] ('MACRS7', 'SL10'…).", BLOCKED)
        else:
            years = math.ceil(1.0 / v["value"] - 1e-9)
            out["depreciation"] = f"SL{years}"
            why = ("propuesta por palabras clave de plant_type" if sug.get("suggested_id") else "sin plant_type: fracción «otras "
                   "actividades»")
            ctx.assumed.append(_bank(v, "basis.depreciation", "depreciación fiscal", out["depreciation"], "tabla (línea recta)",
                                     f"{v['basis']}: {v['value']:.0%} anual máximo → línea recta en {years} años ({1 / years:.2%} "
                                     f"anual, sin rebasar el máximo); {why}. La clasificación fiscal la decide un fiscalista."))
    # construction schedule
    if "construction_schedule" in basis:
        out["construction_schedule"] = [float(x) for x in basis["construction_schedule"]]
        ctx.add_user("basis.construction_schedule", out["construction_schedule"], "fracciones por año")
    else:
        v = ctx.design_basis()["values"].get("construction_schedule")
        if v is None or not v.get("table"):
            out["construction_schedule"] = None
            ctx.add_missing("basis.construction_schedule", "Da basis['construction_schedule'] (fracciones del capital por año de "
                            "construcción que sumen 1).", BLOCKED)
        else:
            out["construction_schedule"] = [float(r["fraccion_del_capital"]) for r in v["table"]]
            ctx.assumed.append(_bank(v, "basis.construction_schedule", "calendario de construcción", out["construction_schedule"],
                                     "fracciones del capital por año"))
    sched = out["construction_schedule"]
    if sched is not None and (not sched or any(x < 0 for x in sched) or abs(sum(sched) - 1.0) > 1e-6):
        raise ValueError(f"basis['construction_schedule'] = {sched}: deben ser fracciones ≥ 0 por año de construcción que sumen 1 "
                         "(por ejemplo [0.08, 0.60, 0.32]).")
    life = ctx.scalars.get("basis.plant_life_years")
    if life is not None:
        if abs(life - round(life)) > 1e-9:
            raise ValueError(f"basis['plant_life_years'] = {life}: debe ser un número entero de años.")
        if out["depreciation"] and len(EC.depreciation_schedule(out["depreciation"])) > life:
            raise ValueError(f"La depreciación {out['depreciation']} dura más que la vida de la planta ({life:g} años): alarga "
                             "basis['plant_life_years'] o acorta basis['depreciation'].")
        if ctx.scalars.get("basis.startup_years", 0.0) >= life:
            raise ValueError("basis['startup_years'] debe ser menor que basis['plant_life_years'].")
    # fixed operating-cost factors (Humbird, Table 31)
    for key in ("maintenance_fraction_of_isbl", "insurance_tax_fraction_of_fci"):
        frac, unit, text = HUMBIRD_FIXED[key]
        if key in basis:
            ctx.scalars[f"basis.{key}"] = _num(basis[key], f"basis['{key}']", minimum=0.0, maximum=1.0)
            ctx.add_user(f"basis.{key}", ctx.scalars[f"basis.{key}"], unit)
        else:
            ctx.scalars[f"basis.{key}"] = frac
            ctx.assumed.append(_record(f"basis.{key}", key.split("_fraction")[0].replace("_", " "), frac, unit, HUMBIRD, ctx.today,
                                       text, scalar=f"basis.{key}"))
    return out


def _count_steps(equipment: list[dict]) -> tuple[int, int, list[str]]:
    """Fluid (non-particulate) and solids steps of the correlation, from `labour_class` or the F_BM `type` of each item."""
    fluid = solids = 0
    guessed = []
    for e in equipment:
        cls = e.get("labour_class")
        if cls is None:
            if e.get("type") in NO_LABOUR_STEP:
                cls = "none"
            else:
                cls = "fluid"
                if e.get("type") is None:
                    guessed.append(e["name"])
        if cls not in ("fluid", "solids", "none"):
            raise ValueError(f"equipment «{e['name']}»: labour_class debe ser 'fluid', 'solids' o 'none' (bombas y recipientes).")
        fluid += cls == "fluid"
        solids += cls == "solids"
    return fluid, solids, guessed


def operators_from_steps(fluid_steps: int, solids_steps: int) -> dict:
    """Alkhayat-Gerrard operators per shift and total operators (both rounded up, as in the source read)."""
    c = LABOUR_CORRELATION
    raw = math.sqrt(c["a"] + c["b"] * solids_steps ** 2 + c["c"] * fluid_steps)
    per_shift = math.ceil(raw - 1e-12)
    per_position = c["shifts_per_year"] / c["shifts_per_operator"]
    return {"operators_per_shift_raw": raw, "operators_per_shift": per_shift, "operators_per_position": per_position,
            "operators": math.ceil(per_shift * round(per_position, 1) - 1e-12)}


def _resolve_labour(call: dict, ctx: _Ctx, equipment: list[dict]) -> dict:
    lab = dict(call.get("labour") or {})
    known = ("positions", "total_salaries_per_year", "currency", "operators", "operators_per_shift", "fluid_steps", "solids_steps",
             "annual_salary_per_operator", "burden_fraction")
    unknown = [k for k in lab if k not in known]
    if unknown:
        raise ValueError(f"labour: claves desconocidas {unknown}. Las válidas son {list(known)}.")
    cur = str(lab.get("currency", "USD")).upper()
    out = {"mode": None, "currency": cur, "detail": {}}
    if "burden_fraction" in lab:
        ctx.scalars["labour.burden_fraction"] = _num(lab["burden_fraction"], "labour['burden_fraction']", minimum=0.0, maximum=5.0)
        ctx.add_user("labour.burden_fraction", ctx.scalars["labour.burden_fraction"], "fracción de los salarios")
    else:
        frac, unit, text = HUMBIRD_FIXED["labour_burden_fraction"]
        ctx.scalars["labour.burden_fraction"] = frac
        ctx.assumed.append(_record("labour.burden_fraction", "carga laboral y gastos generales de planta", frac, unit, HUMBIRD,
                                   ctx.today, text, scalar="labour.burden_fraction"))
    if lab.get("positions") or lab.get("total_salaries_per_year") is not None:
        total = 0.0
        if lab.get("total_salaries_per_year") is not None:
            total = _num(lab["total_salaries_per_year"], "labour['total_salaries_per_year']", minimum=0.0)
        for i, pos in enumerate(lab.get("positions") or []):
            if "count" not in pos or "annual_salary" not in pos:
                raise ValueError(f"labour['positions'][{i}]: da count (personas) y annual_salary (salario anual por persona, en "
                                 f"{cur}); son la nómina de TU planta.")
            total += _num(pos["count"], "count", minimum=0.0) * _num(pos["annual_salary"], "annual_salary", minimum=0.0)
        out["mode"] = "total"
        ctx.scalars["labour.total_salaries"] = total
        ctx.add_user("labour.total_salaries_per_year", total, f"{cur}/y")
        if ctx.fx(cur) is None:
            out["mode"] = None
        return out
    # head-count: caller's, else the published correlation on the step counts
    c = LABOUR_CORRELATION
    if lab.get("operators") is not None:
        ctx.scalars["labour.operators"] = _num(lab["operators"], "labour['operators']", minimum=0.0)
        ctx.add_user("labour.operators", ctx.scalars["labour.operators"], "operadores (plantilla total)")
    else:
        if lab.get("operators_per_shift") is not None:
            per_shift = _num(lab["operators_per_shift"], "labour['operators_per_shift']", minimum=0.0)
            ops = math.ceil(per_shift * 4.5 - 1e-12)
            note = f"{per_shift:g} operadores por turno (tuyos) × 1095/245 ≈ 4.5 operadores por puesto, redondeado hacia arriba."
            out["detail"] = {"operators_per_shift": per_shift, "operators": ops}
        else:
            fluid, solids, guessed = _count_steps(equipment)
            origin = "contados de la lista de equipos (bombas, recipientes y tanques no cuentan)"
            if lab.get("fluid_steps") is not None or lab.get("solids_steps") is not None:
                fluid = int(_num(lab.get("fluid_steps", 0), "labour['fluid_steps']", minimum=0.0))
                solids = int(_num(lab.get("solids_steps", 0), "labour['solids_steps']", minimum=0.0))
                origin, guessed = "dados por el usuario", []
            n = operators_from_steps(fluid, solids)
            ops = n["operators"]
            note = (f"N_OL = (6.29 + 31.7·P² + 0.23·N_np)^0.5 con P = {solids} pasos con sólidos y N_np = {fluid} pasos de fluidos "
                    f"({origin}) = {n['operators_per_shift_raw']:.2f} → {n['operators_per_shift']} por turno × 4.5 → {ops} operadores.")
            if guessed:
                note += (f" Equipos sin `type` ni `labour_class`, contados como paso de fluidos: {guessed}; si manejan sólidos da "
                         "labour_class='solids' (cambia mucho el resultado).")
                ctx.warnings.append(f"Plantilla de operadores: {len(guessed)} equipos sin clasificar se contaron como pasos de "
                                    f"fluidos ({guessed}). Marca labour_class='solids' en los que manejan sólidos o da labour.")
            out["detail"] = {"fluid_steps": fluid, "solids_steps": solids, **n}
        ctx.scalars["labour.operators"] = float(ops)
        ctx.assumed.append(_record("labour.operators", "plantilla de operadores de turno", ops, "operadores", c, ctx.today, note,
                                   scalar="labour.operators"))
        ctx.not_included.append("Personal que no es operador de turno (supervisión, laboratorio, mantenimiento, ingeniería, "
                                "administración): la correlación de plantilla solo da operadores; agrégalo en labour['positions'].")
    if lab.get("annual_salary_per_operator") is not None:
        ctx.scalars["labour.annual_salary"] = _num(lab["annual_salary_per_operator"], "labour['annual_salary_per_operator']", minimum=0.0)
        ctx.add_user("labour.annual_salary_per_operator", ctx.scalars["labour.annual_salary"], f"{cur}/y")
    else:
        v = ctx.design_basis()["values"].get("imss_average_wage_mx") if "MX" in ctx.chain else None
        if v is None or v.get("currency") is None or not str(v["unit"]).lower().endswith("/day"):
            ctx.add_missing("labour.annual_salary_per_operator", f"El banco solo tiene salarios de México; para {ctx.region} da "
                            "labour['annual_salary_per_operator'] (o labour['positions']).", BLOCKED)
            return out
        cur = out["currency"] = v["currency"]
        ctx.scalars["labour.annual_salary"] = v["value"] * 365.0
        ctx.assumed.append(_bank(v, "labour.annual_salary_per_operator", "salario anual por operador", v["value"] * 365.0, f"{cur}/y",
                                 f"{v['value']} {v['unit']} × 365 días pagados (LFT art. 69: descanso semanal con goce de salario). "
                                 f"{v.get('basis', '')}: promedio de TODOS los sectores, no el de un operador de planta química; antes "
                                 "de prestaciones (van en burden_fraction).", scalar="labour.annual_salary"))
    out["mode"] = "operators" if ctx.fx(cur) is not None else None
    return out


def _resolve_flows(call: dict, ctx: _Ctx) -> list[dict]:
    """Every raw material, product and utility with its flow and price in consistent units; prices caller → bank → missing."""
    prices = call.get("prices") or {}
    price_by_name = {_norm(k): (k, v) for k, v in prices.items()}
    used_prices = set()
    flows: list[dict] = []
    for role in ROLES:
        items = call.get(role) or []
        if role != "utilities" and not items:
            raise ValueError(f"{role}: da al menos un elemento [{{name, flow, unit}}]; sin {ROLE_ES[role]}s no hay balance económico.")
        for i, it in enumerate(items):
            where = f"{role}[{i}]"
            if not isinstance(it, dict) or not it.get("name"):
                raise ValueError(f"{where}: falta name.")
            name = str(it["name"])
            where = f"{ROLE_ES[role]} «{name}»"
            skey = f"price.{role}.{name}"
            f = {"name": name, "role": role, "scalar": skey, "annual": False, "price_origin": None, "density_kg_m3": None}
            if it.get("annual_cost") is not None:
                cur = str(it.get("currency", "USD")).upper()
                f.update(mode="annual_cost", currency=cur, price_origin="user")
                ctx.scalars[skey] = _num(it["annual_cost"], f"{where}: annual_cost", minimum=0.0)
                ctx.add_user(f"{role}.{name}.annual_cost", ctx.scalars[skey], f"{cur}/y")
                if ctx.fx(cur) is None:
                    f["price_origin"] = None
                flows.append(f)
                continue
            if it.get("flow") is None or not (it.get("unit") or it.get("flow_unit")):
                raise ValueError(f"{where}: faltan flow y unit (por ejemplo flow=1250, unit='kg/h'); es el balance de TU planta y "
                                 "fija el costo o el ingreso anual.")
            unit = it.get("unit") or it.get("flow_unit")
            dim, per_time, annual = _flow_unit(unit, where)
            f.update(mode="flow", flow=_num(it["flow"], f"{where}: flow", minimum=0.0), flow_unit=str(unit), flow_dim=dim,
                     flow_base=per_time, annual=annual)
            if it.get("density_kg_m3") is not None:
                f["density_kg_m3"] = _num(it["density_kg_m3"], f"{where}: density_kg_m3", positive=True)
            # price: the item's own → prices[name] → bank
            price = unit_p = None
            if it.get("price") is not None:
                price, unit_p = it["price"], it.get("price_unit")
            elif _norm(name) in price_by_name:
                k, pv = price_by_name[_norm(name)]
                used_prices.add(k)
                if isinstance(pv, dict):
                    price, unit_p = pv.get("price"), pv.get("price_unit")
                elif isinstance(pv, (list, tuple)) and len(pv) == 2:
                    price, unit_p = pv
                else:
                    raise ValueError(f"prices['{k}']: da {{'price': valor, 'price_unit': 'USD/t'}} (un número sin unidad no se acepta).")
            if price is not None:
                if not unit_p:
                    raise ValueError(f"{where}: el precio {price!r} necesita price_unit (USD/t, MXN/kWh, USD/m3, USD/gal…).")
                cur, pdim, pper, q = _price_unit(unit_p, where)
                ctx.scalars[skey] = _num(price, f"{where}: price", minimum=0.0)
                src = it.get("price_source")
                if isinstance(src, dict) and src.get("source") and src.get("url") and src.get("period"):
                    # a SOURCED suggestion carried by the item (e.g. the published price of a plant model): an assumption,
                    # never a user input
                    f.update(currency=cur, price_dim=pdim, price_base=pper, price_unit=f"{cur}/{q}", price_origin="assumed")
                    ctx.assumed.append(_record(f"prices.{name}", f"precio de {where}", ctx.scalars[skey], f["price_unit"],
                                               src, ctx.today, str(src.get("note", "")), scalar=skey, role=role,
                                               source_tier=src.get("source_tier", "modelo publicado")))
                else:
                    f.update(currency=cur, price_dim=pdim, price_base=pper, price_unit=f"{cur}/{q}", price_origin="user")
                    ctx.add_user(f"prices.{name}", ctx.scalars[skey], f["price_unit"])
            else:
                ref = it.get("reference_item") or name
                r = RP.lookup(ref, ctx.region, None, None, ctx.today)
                blocks = BLOCKED[:3] if role == "products" else BLOCKED
                if not r.get("found"):
                    ctx.add_missing(f"prices.{name}", f"Sin precio de {where}: {r.get('reason', 'no está en el banco')}", blocks,
                                    close_matches=r.get("close_matches", []), role=role)
                else:
                    try:
                        cur, pdim, pper, q = _price_unit(r["unit"], where)
                    except ValueError:
                        ctx.add_missing(f"prices.{name}", f"El banco tiene «{r['item']}» en {r['unit']}, que no es un precio por "
                                        f"cantidad utilizable para {where}: da el precio.", blocks, role=role)
                    else:
                        f.update(currency=cur, price_dim=pdim, price_base=pper, price_unit=f"{cur}/{q}", price_origin="assumed")
                        ctx.scalars[skey] = float(r["value"])
                        note = f"{r.get('basis', '')}. {r.get('notes', '')}".strip()
                        if r.get("fallback_region_used"):
                            note = f"{r['fallback_note']} {note}"
                        if _norm(ref) != _norm(name):
                            note = f"Tomado del banco como «{ref}». {note}"
                        ctx.assumed.append(_bank(r, f"prices.{name}", f"precio de {where}", unit=f["price_unit"], note=note,
                                                 low=r.get("low"), high=r.get("high"), scalar=skey, role=role,
                                                 source_tier=r.get("source_tier")))
                        if r.get("stale"):
                            ctx.warnings.append(f"Precio de {where}: VALOR VIEJO ({r['age_text']}); busca el dato vigente.")
            if f["price_origin"]:
                if f["flow_dim"] != f["price_dim"]:
                    if {f["flow_dim"], f["price_dim"]} != {"mass", "volume"}:
                        raise ValueError(f"{where}: el flujo está en {f['flow_unit']} y el precio en {f['price_unit']}; entre energía "
                                         "y masa o volumen no hay conversión sin un poder calorífico: da el flujo en la unidad del "
                                         "precio.")
                    if f["density_kg_m3"] is None:
                        raise ValueError(f"{where}: el flujo está en {f['flow_unit']} y el precio en {f['price_unit']}"
                                         f"{' (valor del banco)' if f['price_origin'] == 'assumed' else ''}: falta density_kg_m3 "
                                         "(densidad de TU corriente, kg/m3) para convertir entre masa y volumen. No se supone.")
                if ctx.fx(f["currency"]) is None:
                    f["price_origin"] = None
            elif it.get("price_unit"):
                cur, pdim, pper, q = _price_unit(it["price_unit"], where)
                f.update(wanted_unit=(cur, pdim, pper, q))
            flows.append(f)
    extra = [k for k in prices if k not in used_prices]
    if extra:
        ctx.warnings.append(f"prices: {extra} no coincide con ningún material, producto o servicio (o ese elemento ya traía price).")
    return flows


def _resolve(call: dict) -> tuple[dict, _Ctx]:
    span = _num(call.get("sensitivity_span", 0.2), "sensitivity_span", minimum=0.0, maximum=0.95)
    ctx = _Ctx(call)
    equipment, cost_basis = _scale_equipment(call, ctx)
    cap = _resolve_capital(call, ctx, equipment)
    basis = _resolve_basis(call, ctx)
    labour = _resolve_labour(call, ctx, equipment)
    flows = _resolve_flows(call, ctx)
    products = [f for f in flows if f["role"] == "products"]
    main = call.get("main_product") or products[0]["name"]
    if main not in [f["name"] for f in products]:
        raise ValueError(f"main_product = «{main}» no está en products ({[f['name'] for f in products]}).")
    if not call.get("main_product") and len(products) > 1:
        ctx.assumptions.append(f"main_product no dado: se toma el primer producto de la lista, «{main}».")
    p = {"equipment": equipment, "cost_basis": cost_basis, "flows": flows, "labour": labour, "main_product": main,
         "depreciation": basis["depreciation"], "construction_schedule": basis["construction_schedule"],
         "working_capital_basis": cap["working_capital_basis"], "isbl_share": cap["isbl_share"], "scalars": ctx.scalars,
         "tax_from_bank": basis["tax_from_bank"], "span": span}
    return p, ctx


# --------------------------------------------------------------------------- evaluation of one resolved set

def _flow_money(f: dict, s: dict, hours: float | None) -> dict:
    """Annual quantity (in the priced unit) and annual USD of one flow, by two unit paths (base SI units; priced unit per hour)."""
    row = {"name": f["name"], "role": f["role"], "price_origin": f["price_origin"], "usd_per_year": None, "quantity_per_year": None,
           "quantity_unit": None, "price": None, "price_unit": None, "price_usd": None, "usd_per_year_check": None}
    if f["mode"] == "annual_cost":
        if f["price_origin"]:
            row["usd_per_year"] = row["usd_per_year_check"] = s[f["scalar"]] * s.get(f"fx.{f['currency']}", 1.0)
            row.update(price=s[f["scalar"]], price_unit=f"{f['currency']}/y")
        return row
    time = 1.0 if f["annual"] else hours
    if time is None:
        return row
    base_year = f["flow"] * f["flow_base"] * time                  # kg, m3 or GJ per year (flow dimension)
    if not f["price_origin"]:
        if f.get("wanted_unit"):
            _, pdim, pper, q = f["wanted_unit"]
            if pdim == f["flow_dim"]:
                row.update(quantity_per_year=base_year / pper, quantity_unit=q)
                return row
            if f["density_kg_m3"] and {pdim, f["flow_dim"]} == {"mass", "volume"}:
                conv = base_year / f["density_kg_m3"] if pdim == "volume" else base_year * f["density_kg_m3"]
                row.update(quantity_per_year=conv / pper, quantity_unit=q)
                return row
        q = _DEFAULT_PRICE_UNIT[f["flow_dim"]]
        row.update(quantity_per_year=base_year / _quantity_unit(q, "")[1], quantity_unit=q)
        return row
    if f["price_dim"] != f["flow_dim"]:
        base_year = base_year / f["density_kg_m3"] if f["price_dim"] == "volume" else base_year * f["density_kg_m3"]
    fx = s.get(f"fx.{f['currency']}", 1.0)
    price_usd = s[f["scalar"]] * fx
    qty = base_year / f["price_base"]
    per_time = f["flow"] * f["flow_base"]
    if f["price_dim"] != f["flow_dim"]:
        per_time = per_time / f["density_kg_m3"] if f["price_dim"] == "volume" else per_time * f["density_kg_m3"]
    row.update(quantity_per_year=qty, quantity_unit=f["price_unit"].split("/", 1)[1], price=s[f["scalar"]], price_unit=f["price_unit"],
               price_usd=price_usd, usd_per_year=base_year * (price_usd / f["price_base"]),
               usd_per_year_check=(per_time / f["price_base"] * price_usd) * time)
    return row


def _capital(p: dict, s: dict) -> dict:
    """capital_investment with the eight factors; land may come as an amount and working capital as a fraction of FCI."""
    eq = [{k: v for k, v in e.items() if k not in ("isbl", "labour_class", "cost_year", "cepci")} for e in p["equipment"]]
    f = {k: s[f"capital_factors.{k}"] for k in EC.CAPITAL_FACTORS}
    lf = s["basis.location_factor"]
    if s["basis.land_usd"] > 0 or p["working_capital_basis"] == "FCI":
        first = EC.capital_investment(eq, {**f, "land": 0.0, "working_capital": 0.0}, lf)
        f["land"] = f["land"] + s["basis.land_usd"] / first["total_depreciable_capital_usd"]
        if p["working_capital_basis"] == "FCI":
            f["working_capital"] = f["working_capital"] / (1.0 + f["land"] + f["royalties"] + f["startup"])
    cap = EC.capital_investment(eq, f, lf)
    isbl_names = {e["name"] for e in p["equipment"] if e["isbl"]}
    cap["isbl_installed_usd"] = sum(r["bare_module_cost_usd"] for r in cap["equipment"] if r["name"] in isbl_names) * lf
    cap["factors_equivalent"] = f
    return cap


def _evaluate(p: dict, s: dict, full: bool = True) -> dict:
    """Capital, operating cost, revenue, cash flow and minimum selling price of one set of scalars `s`."""
    cap = _capital(p, s)
    hours = s.get("basis.on_stream_hours")
    rows = [_flow_money(f, s, hours) for f in p["flows"]]
    by_role = {r: [x for x in rows if x["role"] == r] for r in ROLES}
    tot = {r: sum(x["usd_per_year"] or 0.0 for x in by_role[r]) for r in ROLES}
    complete = {r: all(x["usd_per_year"] is not None for x in by_role[r]) for r in ROLES}
    lab = p["labour"]
    fx_l = s.get(f"fx.{lab['currency']}", 1.0)
    salaries = (s["labour.total_salaries"] * fx_l if lab["mode"] == "total" else
                s["labour.operators"] * s["labour.annual_salary"] * fx_l if lab["mode"] == "operators" else None)
    burden = salaries * s["labour.burden_fraction"] if salaries is not None else None
    fci = cap["total_depreciable_capital_usd"]
    maintenance = s["basis.maintenance_fraction_of_isbl"] * cap["isbl_installed_usd"]
    insurance = s["basis.insurance_tax_fraction_of_fci"] * fci
    variable = tot["raw_materials"] + tot["utilities"]
    fixed = (salaries or 0.0) + (burden or 0.0) + maintenance + insurance
    costs_complete = complete["raw_materials"] and complete["utilities"] and salaries is not None
    main = next(x for x in by_role["products"] if x["name"] == p["main_product"])
    others = [x for x in by_role["products"] if x["name"] != p["main_product"]]
    other_rev = sum(x["usd_per_year"] or 0.0 for x in others)
    others_complete = all(x["usd_per_year"] is not None for x in others)
    out = {"capital": cap, "rows": rows, "raw_materials_usd_y": tot["raw_materials"], "utilities_usd_y": tot["utilities"],
           "labour_usd_y": salaries, "overhead_usd_y": burden, "maintenance_usd_y": maintenance, "insurance_tax_usd_y": insurance,
           "variable_usd_y": variable, "fixed_usd_y": fixed, "operating_cost_usd_y": variable + fixed, "costs_complete": costs_complete,
           "revenue_usd_y": tot["products"], "revenue_complete": complete["products"], "main_row": main,
           "by_products_usd_y": other_rev, "by_products_complete": others_complete, "cash_flow": None, "npv_usd": None, "irr": None,
           "payback_years": None, "discounted_payback_years": None, "msp": None, "msp_verification": None,
           "msp_is_upper_bound": False, "depreciable_usd": fci + cap["royalties_usd"] + cap["startup_usd"]}
    need = ("basis.plant_life_years", "basis.income_tax_rate", "basis.discount_rate", "basis.startup_years", "basis.on_stream_hours")
    if not costs_complete or any(k not in s for k in need) or not p["depreciation"] or not p["construction_schedule"]:
        return out
    opex = variable + fixed
    blend = ((s["basis.startup_variable_cost_fraction"] * variable + s["basis.startup_fixed_cost_fraction"] * fixed) / opex
             if opex > 0 else 1.0)
    args = dict(depreciable_capital_usd=out["depreciable_usd"], operating_cost_usd_y=opex,
                plant_life_years=int(round(s["basis.plant_life_years"])), income_tax_rate=s["basis.income_tax_rate"],
                discount_rate=s["basis.discount_rate"], depreciation=p["depreciation"],
                construction_schedule=p["construction_schedule"], working_capital_usd=cap["working_capital_usd"],
                land_usd=cap["land_usd"], startup_years=s["basis.startup_years"],
                startup_revenue_fraction=s["basis.startup_revenue_fraction"], startup_operating_cost_fraction=blend)
    out["cash_flow_args"] = args
    all_revenue = out["revenue_complete"]
    cf = EC.cash_flow_analysis(revenue_usd_y=tot["products"] if all_revenue else other_rev, **args)
    if all_revenue:
        out.update(cash_flow=cf, npv_usd=cf["npv_usd"], irr=cf["irr"], payback_years=cf["payback_years"],
                   discounted_payback_years=cf["discounted_payback_years"])
    else:
        out["cash_flow_break_even_only"] = cf
    qty = main["quantity_per_year"]
    if cf["break_even_revenue_usd_y"] is not None and qty:
        out["msp"] = (cf["break_even_revenue_usd_y"] - other_rev) / qty
        out["msp_is_upper_bound"] = not others_complete
        if full:
            again = EC.cash_flow_analysis(revenue_usd_y=max(other_rev + out["msp"] * qty, 0.0), **args)
            out["msp_verification"] = {"npv_usd_at_minimum_price": again["npv_usd"], "cash_flow_evidence": again["evidence"],
                                       "scale_usd": cf["total_capital_spent_usd"]}
    return out


# --------------------------------------------------------------------------- sensitivity

def _perturbed(p: dict, rec: dict, base: dict, span: float) -> list[tuple[str, dict, dict, object]]:
    """(label, p, scalars, value) for the low and the high case of one assumed input; [] when it cannot be varied."""
    s0 = p["scalars"]
    key = rec.get("scalar")
    cases = []
    if key is not None:
        v0 = s0[key]
        lo, hi = v0 * (1.0 - span), v0 * (1.0 + span)
        origin = f"±{span:.0%}"
        if rec.get("low") is not None and rec.get("high") is not None and rec["low"] <= rec["value"] <= rec["high"] and rec["low"] < rec["high"]:
            lo, hi, origin = float(rec["low"]), float(rec["high"]), "rango low-high del banco"
        if key == "basis.plant_life_years":
            dep = len(EC.depreciation_schedule(p["depreciation"])) if p["depreciation"] else 1
            lo, hi = max(round(lo), dep, math.floor(s0.get("basis.startup_years", 0)) + 1), max(round(hi), dep)
        if key == "basis.on_stream_hours":
            hi = min(hi, 8760.0)
        if key == "basis.income_tax_rate":
            hi = min(hi, 0.95)
        for tag, val in (("low", lo), ("high", hi)):
            cases.append((tag, p, {**s0, key: val}, val, origin))
    elif rec["input"] == "basis.depreciation" and str(p["depreciation"]).upper().startswith("SL"):
        n = int(str(p["depreciation"])[2:])
        life = int(round(s0.get("basis.plant_life_years", n)))
        for tag, m in (("low", max(1, round(n * (1 - span)))), ("high", min(life, max(n, round(n * (1 + span)))))):
            cases.append((tag, {**p, "depreciation": f"SL{m}"}, s0, f"SL{m}", f"±{span:.0%} en años"))
    return cases


def _sensitivity(p: dict, ctx: _Ctx, base: dict) -> dict:
    span = p["span"]
    rows, skipped = [], []
    if span <= 0:
        return {"span": span, "rows": [], "ranked_by": None, "skipped": [], "note": "sensitivity_span = 0: no se calculó."}
    metric = "npv_usd" if base["npv_usd"] is not None else "msp" if base["msp"] is not None else "tci_plus_opex"
    for rec in ctx.assumed:
        cases = _perturbed(p, rec, base, span)
        if not cases:
            skipped.append({"input": rec["input"], "why": "no es un escalar (tabla o calendario): cámbialo con plant_economics_update"})
            continue
        row = {"input": rec["input"], "label": rec["label"], "base_value": rec["value"], "unit": rec["unit"], "range": cases[0][4]}
        ok = True
        for tag, pp, ss, val, _ in cases:
            try:
                r = _evaluate(pp, ss, full=False)
            except ValueError as exc:
                skipped.append({"input": rec["input"], "why": f"el caso {tag} no es evaluable: {exc}"})
                ok = False
                break
            row[f"{tag}_value"] = val
            row[f"npv_usd_{tag}"] = r["npv_usd"]
            row[f"minimum_selling_price_{tag}"] = r["msp"]
            row[f"total_capital_investment_usd_{tag}"] = r["capital"]["total_capital_investment_usd"]
            row[f"operating_cost_usd_y_{tag}"] = r["operating_cost_usd_y"]
        if not ok:
            continue
        d = lambda k: (abs(row[f"{k}_high"] - row[f"{k}_low"]) if row[f"{k}_high"] is not None and row[f"{k}_low"] is not None else None)
        row["npv_swing_usd"] = d("npv_usd")
        row["minimum_selling_price_swing"] = d("minimum_selling_price")
        row["impact"] = (row["npv_swing_usd"] if metric == "npv_usd" else row["minimum_selling_price_swing"] if metric == "msp" else
                         (d("total_capital_investment_usd") or 0.0) + (d("operating_cost_usd_y") or 0.0))
        rows.append(row)
    rows.sort(key=lambda r: -(r["impact"] or 0.0))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    names = {"npv_usd": "cambio del VPN (USD) entre el caso bajo y el alto", "msp": "cambio del precio mínimo de venta",
             "tci_plus_opex": "cambio de la inversión total más el del costo de operación anual (no hay VPN ni precio mínimo)"}
    return {"span": span, "ranked_by": names[metric], "rows": rows, "skipped": skipped,
            "confirm_first": [r["input"] for r in rows[:5]],
            "note": "Un factor a la vez sobre cada valor SUPUESTO, dejando lo demás en su valor base. Es la lista de qué confirmar "
                    "primero, no un análisis de riesgo: no hay correlaciones ni probabilidades."}


# --------------------------------------------------------------------------- assembly

def _ptu(p: dict, ctx: _Ctx, base: dict) -> dict | None:
    """Bounds of Mexico's profit sharing, which is NOT in the tax rate: capped (payroll-like cost) and uncapped (10 % tax)."""
    if not p["tax_from_bank"]:
        return None
    v = ctx.design_basis()["values"].get("ptu_mx")
    if v is None:
        return None
    isr, rate = p["scalars"]["basis.income_tax_rate"], float(v["value"])
    out = {"included_in_tax_rate": False, "rate": rate, "source": v["source"], "url": v["url"], "law_urls": MX_LAW,
           "base": "renta gravable de la LISR (LFT art. 120), sin restar la PTU pagada ni las pérdidas de ejercicios anteriores "
                   "(LISR art. 9); no se compensan años de pérdida con años de ganancia (LFT art. 128)",
           "cap": "por trabajador, tres meses de su salario o el promedio de la PTU recibida en los últimos tres años, lo que más "
                  "le favorezca (LFT art. 127-VIII, DOF 23-04-2021 y 24-12-2024); empresas nuevas exentas el primer año (art. 126)",
           "why_not_in_rate": "con el tope por trabajador, en una planta intensiva en capital la PTU se comporta como un costo de "
                              "nómina (≈ 3 meses de salarios), no como un impuesto del 10 %; por eso la tasa usada es solo el ISR.",
           "uncapped_combined_rate": isr + rate * (1.0 - isr), "cap_estimate_usd_y": None,
           "npv_usd_with_cap_as_yearly_cost": None, "npv_usd_at_uncapped_combined_rate": None}
    args = base.get("cash_flow_args")
    if base["labour_usd_y"] is not None:
        out["cap_estimate_usd_y"] = 0.25 * base["labour_usd_y"]
    if args and base["npv_usd"] is not None:
        rev = base["revenue_usd_y"]
        hi = EC.cash_flow_analysis(revenue_usd_y=rev, **{**args, "income_tax_rate": out["uncapped_combined_rate"]})
        out["npv_usd_at_uncapped_combined_rate"] = hi["npv_usd"]
        if out["cap_estimate_usd_y"] is not None:
            lo = EC.cash_flow_analysis(revenue_usd_y=rev, **{**args, "operating_cost_usd_y": args["operating_cost_usd_y"]
                                                              + out["cap_estimate_usd_y"]})
            out["npv_usd_with_cap_as_yearly_cost"] = lo["npv_usd"]
    out["note"] = ("Cotas, no un cálculo fiscal: «con tope» suma 3 meses de los salarios como costo anual deducible; «sin tope» usa "
                   "ISR + PTU·(1 − ISR) como tasa única (PTU deducible del ISR, LISR art. 9-I). El VPN real queda entre el del "
                   "resultado (sin PTU) y esas cotas.")
    return out


def _evidence(p: dict, base: dict, ctx: _Ctx) -> dict:
    cap, s = base["capital"], p["scalars"]
    f = cap["factors_equivalent"]
    reasons: list[str] = []
    c_tbm = sum(r["bare_module_cost_usd"] for r in cap["equipment"]) * s["basis.location_factor"]
    by_hand = (c_tbm * (1 + f["site_preparation"] + f["service_facilities"] + f["allocated_costs"]) * (1 + f["contingency_and_fee"])
               * (1 + f["land"] + f["royalties"] + f["startup"]) * (1 + f["working_capital"]))
    tci = cap["total_capital_investment_usd"]
    ev = {"capital_by_hand_residual": abs(by_hand - tci) / max(tci, 1.0),
          "capital_components_sum_to_total": bool(cap["evidence"]["components_sum_to_total"]),
          "capital_investment_verified": bool(cap["evidence"]["verified"])}
    if ev["capital_by_hand_residual"] > REL_TOL or not ev["capital_components_sum_to_total"]:
        reasons.append("la inversión recalculada a mano no coincide con capital_investment")
    if p["working_capital_basis"] == "FCI":
        ev["working_capital_of_fci_residual"] = abs(cap["working_capital_usd"] - s["capital_factors.working_capital"]
                                                    * cap["total_depreciable_capital_usd"]) / max(tci, 1.0)
        if ev["working_capital_of_fci_residual"] > REL_TOL:
            reasons.append("el capital de trabajo no es la fracción pedida del capital fijo")
    pairs = [(r["usd_per_year"], r["usd_per_year_check"]) for r in base["rows"] if r["usd_per_year"] is not None]
    ev["flows_recomputed_residual"] = max([abs(a - b) / max(abs(a), 1.0) for a, b in pairs] + [0.0])
    parts = (base["raw_materials_usd_y"] + base["utilities_usd_y"] + (base["labour_usd_y"] or 0.0) + (base["overhead_usd_y"] or 0.0)
             + base["maintenance_usd_y"] + base["insurance_tax_usd_y"])
    ev["operating_cost_sum_residual"] = abs(parts - base["operating_cost_usd_y"]) / max(base["operating_cost_usd_y"], 1.0)
    if ev["flows_recomputed_residual"] > REL_TOL or ev["operating_cost_sum_residual"] > REL_TOL:
        reasons.append("costos o ingresos recalculados de flujo × precio × horas no coinciden")
    cf = base["cash_flow"]
    if cf is not None:
        r = s["basis.discount_rate"]
        scale = cf["total_capital_spent_usd"]
        npv_table = sum(row["cash_flow"] / (1.0 + r) ** row["t"] for row in cf["table"])
        ev["npv_from_table_residual"] = abs(npv_table - cf["npv_usd"]) / scale
        ev["cash_flow_capital_equals_tci_residual"] = abs(scale - tci) / max(tci, 1.0)
        ev["cash_flow_engine_verified"] = bool(cf["evidence"]["verified"])
        if cf["irr"] is not None:
            ev["npv_at_irr_residual"] = abs(sum(row["cash_flow"] / (1.0 + cf["irr"]) ** row["t"] for row in cf["table"])) / scale
            if ev["npv_at_irr_residual"] > 1e-6:
                reasons.append("el VPN a la TIR no es cero")
        if ev["npv_from_table_residual"] > 1e-9 or ev["cash_flow_capital_equals_tci_residual"] > REL_TOL:
            reasons.append("la suma de flujos descontados no reproduce el VPN o el capital del flujo no es la inversión total")
        if not ev["cash_flow_engine_verified"]:
            reasons += [f"flujo de caja: {x}" for x in cf["evidence"]["reasons"]]
    ver = base["msp_verification"]
    if ver is not None:
        ev["npv_at_minimum_price_residual"] = abs(ver["npv_usd_at_minimum_price"]) / ver["scale_usd"]
        if ev["npv_at_minimum_price_residual"] > 1e-6:
            reasons.append("el VPN al precio mínimo de venta no es cero")
    ev["bank_verified"] = bool(RP.bank_integrity()["verified"])
    if not ev["bank_verified"]:
        reasons.append("el banco de valores de referencia no pasó su verificación de integridad")
    ev["complete"] = not ctx.missing
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    ev["scope"] = ("certifica la aritmética de esta corrida (identidades de capital, de flujo × precio × horas y del flujo de caja); "
                   "NO certifica los valores supuestos ni tus datos. `complete` False = hubo entradas sin resolver.")
    return ev


def _frozen_call(call: dict, p: dict, base: dict) -> dict:
    """The same call with every resolved value written explicitly (all become the caller's): reproduces the numbers."""
    s, cap = p["scalars"], base["capital"]
    fz = copy.deepcopy(call)
    fz["equipment"] = [{k: v for k, v in e.items() if k not in ("cost_year", "cepci")} for e in p["equipment"]]
    fz["capital_factors"] = {k: s[f"capital_factors.{k}"] for k in EC.CAPITAL_FACTORS}
    if p["working_capital_basis"] == "FCI":
        fz["capital_factors"]["working_capital"] = cap["factors_equivalent"]["working_capital"]
    b = {k.split(".", 1)[1]: v for k, v in s.items() if k.startswith("basis.")}
    if p["depreciation"]:
        b["depreciation"] = p["depreciation"]
    if p["construction_schedule"]:
        b["construction_schedule"] = list(p["construction_schedule"])
    if "plant_life_years" in b:
        b["plant_life_years"] = int(round(b["plant_life_years"]))
    fz["basis"] = b
    lab = {"burden_fraction": s["labour.burden_fraction"], "currency": p["labour"]["currency"]}
    if p["labour"]["mode"] == "total":
        lab["total_salaries_per_year"] = s["labour.total_salaries"]
    elif "labour.operators" in s:
        lab["operators"] = s["labour.operators"]
        if "labour.annual_salary" in s:
            lab["annual_salary_per_operator"] = s["labour.annual_salary"]
    fz["labour"] = lab
    fz["prices"] = None
    for role in ROLES:
        items = []
        for it, f in zip(call.get(role) or [], [f for f in p["flows"] if f["role"] == role]):
            it = dict(it)
            if f["mode"] == "flow" and f["price_origin"]:
                it["price"], it["price_unit"] = s[f["scalar"]], f["price_unit"]
            items.append(it)
        fz[role] = items
    fx = {k.split(".", 1)[1]: v for k, v in s.items() if k.startswith("fx.")}
    if fx:
        fz["exchange_rates"] = {**(call.get("exchange_rates") or {}), **fx}
    fz["main_product"] = p["main_product"]
    return fz


def _cost_table(base: dict, qty) -> dict:
    per = lambda x: (x / qty if (x is not None and qty) else None)
    items = [("raw_materials", "materias primas", base["raw_materials_usd_y"]), ("utilities", "servicios y residuos", base["utilities_usd_y"]),
             ("labour", "mano de obra (salarios)", base["labour_usd_y"]),
             ("overhead", "carga laboral y gastos generales de planta", base["overhead_usd_y"]),
             ("maintenance", "mantenimiento", base["maintenance_usd_y"]),
             ("insurance_tax", "seguro e impuesto predial", base["insurance_tax_usd_y"])]
    main = base["main_row"]
    return {"breakdown": [{"category": k, "label": lab, "usd_per_year": v, "per_unit_of_main_product": per(v)} for k, lab, v in items],
            "items": [{**{k: r[k] for k in ("name", "role", "quantity_per_year", "quantity_unit", "price", "price_unit", "price_usd",
                                             "price_origin", "usd_per_year")}, "per_unit_of_main_product": per(r["usd_per_year"])}
                      for r in base["rows"] if r["role"] != "products"],
            "variable_usd_y": base["variable_usd_y"], "fixed_usd_y": base["fixed_usd_y"], "total_usd_y": base["operating_cost_usd_y"],
            "total_per_unit_of_main_product": per(base["operating_cost_usd_y"]), "main_product_unit": main["quantity_unit"],
            "complete": base["costs_complete"], "excludes": "depreciación (va en el flujo de caja) e impuestos sobre la utilidad"}


def _run(call: dict, sensitivity: bool = True) -> dict:
    p, ctx = _resolve(call)
    base = _evaluate(p, p["scalars"], full=True)
    s, cap, main = p["scalars"], base["capital"], base["main_row"]
    qty = main["quantity_per_year"]
    cap_reasons = cap["evidence"]["reasons"]
    ctx.warnings += [f"Capital: {x}" for x in cap_reasons]
    if not base["costs_complete"]:
        ctx.warnings.insert(0, "COSTOS INCOMPLETOS: falta el precio de una materia prima o servicio, o la mano de obra (ver missing). "
                               "Se devuelven la inversión y los costos parciales; VPN, TIR y precio mínimo quedan en None.")
    elif base["npv_usd"] is None and base["msp"] is not None:
        ctx.warnings.insert(0, "INGRESOS INCOMPLETOS: falta el precio de un producto (ver missing). VPN y TIR quedan en None; el "
                               "precio mínimo de venta del producto principal sí se calcula.")
    msp = None
    if base["msp"] is not None:
        unit = main["quantity_unit"]
        msp = {"product": p["main_product"], "value": base["msp"], "unit": f"USD/{unit}", "annual_quantity": qty,
               "annual_quantity_unit": f"{unit}/y", "discount_rate": s["basis.discount_rate"],
               "npv_usd_at_minimum_price": base["msp_verification"]["npv_usd_at_minimum_price"],
               "current_price": main["price_usd"], "current_price_origin": main["price_origin"],
               "margin_over_minimum": (main["price_usd"] / base["msp"] - 1.0) if main["price_usd"] and base["msp"] > 0 else None,
               "is_upper_bound": base["msp_is_upper_bound"],
               "note": ("Precio del producto principal que hace VPN = 0 a la tasa de descuento, con los demás productos a su precio."
                        + (" COTA SUPERIOR: algún subproducto no tiene precio y no acredita ingreso." if base["msp_is_upper_bound"] else "")
                        + (" Valor ≤ 0: los subproductos pagan todo el costo." if base["msp"] <= 0 else ""))}
    for rec in ctx.assumed:
        txt = f"{rec['label']} = {rec['value'] if not isinstance(rec['value'], float) else format(rec['value'], '.6g')} {rec['unit']}"
        ctx.assumptions.append(f"{txt} [{rec['source']}; periodo {rec['period']}; leído el {rec['date_read']}] — {FLAG}.")
        if rec["stale"]:
            ctx.warnings.append(f"{rec['input']}: valor viejo ({rec['age_days']} días desde el fin de su periodo).")
    ctx.assumptions.append("Tasa de impuesto = solo ISR; la PTU no se suma a la tasa (ver `ptu` y not_included)."
                           if p["tax_from_bank"] else "La tasa de impuesto es la que diste; no se agrega PTU ni otro gravamen.")
    ctx.assumptions.append("El capital que se gasta y deprecia en el flujo de caja es C_TDC + regalías + arranque; el terreno y el "
                           "capital de trabajo se recuperan al final. La suma es la inversión total (TCI).")
    if any(r["price_origin"] == "assumed" for r in base["rows"]):
        ctx.assumptions.append(f"Los precios supuestos son de su periodo publicado y la inversión está en {p['cost_basis']['label']}: "
                               "términos reales de años distintos mezclados; actualiza el CEPCI o los precios si la diferencia importa.")
    if p["tax_from_bank"]:
        ctx.not_included.append("PTU (10 % de la renta gravable, con tope de tres meses de salario por trabajador): no está en la "
                                "tasa de impuesto; sus cotas están en `ptu`.")
    out = {
        "region": ctx.region, "currency": call.get("currency", "USD"), "calculation_currency": "USD", "plant_type": call.get("plant_type"),
        "main_product": p["main_product"], "cost_basis": p["cost_basis"], "today": ctx.today.isoformat(),
        "capital": cap,
        "capital_summary": {"total_purchase_cost_usd": cap["total_purchase_cost_usd"],
                            "total_bare_module_cost_usd": cap["total_bare_module_cost_usd"],
                            "isbl_installed_usd": cap["isbl_installed_usd"],
                            "direct_permanent_investment_usd": cap["direct_permanent_investment_usd"],
                            "fixed_capital_investment_usd": cap["total_depreciable_capital_usd"],
                            "land_usd": cap["land_usd"], "working_capital_usd": cap["working_capital_usd"],
                            "total_capital_investment_usd": cap["total_capital_investment_usd"],
                            "depreciable_in_cash_flow_usd": base["depreciable_usd"],
                            "factors_passed_to_capital_investment": cap["factors_equivalent"],
                            "location_factor": s["basis.location_factor"]},
        "operating_costs": _cost_table(base, qty),
        "revenue": {"items": [{k: r[k] for k in ("name", "quantity_per_year", "quantity_unit", "price", "price_unit", "price_usd",
                                                  "price_origin", "usd_per_year")} for r in base["rows"] if r["role"] == "products"],
                    "total_usd_y": base["revenue_usd_y"] if base["revenue_complete"] else None,
                    "by_products_usd_y": base["by_products_usd_y"], "complete": base["revenue_complete"]},
        "labour": {"mode": p["labour"]["mode"], "currency": p["labour"]["currency"], **p["labour"]["detail"],
                   "salaries_usd_y": base["labour_usd_y"], "burden_usd_y": base["overhead_usd_y"]},
        "cash_flow": base["cash_flow"], "npv_usd": base["npv_usd"], "irr": base["irr"], "payback_years": base["payback_years"],
        "discounted_payback_years": base["discounted_payback_years"], "minimum_selling_price": msp,
        "ptu": _ptu(p, ctx, base),
        "sensitivity": _sensitivity(p, ctx, base) if sensitivity else None,
        "assumed_inputs": [{k: v for k, v in r.items() if k != "scalar"} for r in ctx.assumed], "user_inputs": ctx.user,
        "missing": ctx.missing, "n_assumed": len(ctx.assumed), "n_user": len(ctx.user), "n_missing": len(ctx.missing),
        "published_factor_set": {"source": HUMBIRD["source"], "url": HUMBIRD["url"], "date_read": DATE_READ, "cost_year": 2007,
                                 "direct_of_isbl": {k: v[0] for k, v in HUMBIRD_DIRECT_OF_ISBL.items()},
                                 "indirect_of_total_direct_cost": HUMBIRD_INDIRECT_OF_TDC,
                                 "working_capital_of_fci": HUMBIRD_WORKING_CAPITAL_OF_FCI,
                                 "fixed_operating": {k: v[0] for k, v in HUMBIRD_FIXED.items()},
                                 "installation_factors": HUMBIRD_INSTALLATION_FACTORS,
                                 "installation_factors_note": "Tabla 25: costo instalado = compra × multiplicador. Pásalos como F_BM "
                                                              "del equipo (con base_purchase_cost_usd = purchase_cost_usd) para "
                                                              "un conjunto consistente con los factores de Humbird."},
        "inputs_resolved": {"call": call, "frozen_call": _frozen_call(call, p, base),
                            "scalars": dict(s), "depreciation": p["depreciation"], "construction_schedule": p["construction_schedule"],
                            "note": "call = lo que diste (plant_economics(**call) repite esta corrida; es lo que usa "
                                    "plant_economics_update); frozen_call = lo mismo con TODO valor resuelto escrito como si fuera "
                                    "tuyo (plant_economics(**frozen_call) da los mismos números con cero supuestos)."},
        "warnings": ctx.warnings, "assumptions": ctx.assumptions, "not_included": ctx.not_included, "method": METHOD,
        "flag": FLAG, "evidence": _evidence(p, base, ctx),
    }
    cur = str(call.get("currency", "USD")).upper()
    if cur != "USD":
        rate = ctx.fx(cur)
        if rate:
            conv = lambda x: (x / rate if x is not None else None)
            out["headline_in_currency"] = {"currency": cur, "usd_per_unit_of_currency": rate,
                                           "total_capital_investment": conv(cap["total_capital_investment_usd"]),
                                           "operating_cost_per_year": conv(base["operating_cost_usd_y"]),
                                           "revenue_per_year": conv(out["revenue"]["total_usd_y"]), "npv": conv(base["npv_usd"]),
                                           "minimum_selling_price": conv(base["msp"]),
                                           "note": "Conversión de los totales en USD con el mismo tipo de cambio de la corrida."}
    return _clean(out)


CALL_KEYS = ("equipment", "raw_materials", "products", "utilities", "labour", "capital_factors", "basis", "prices", "region",
             "currency", "plant_type", "main_product", "year", "cepci", "exchange_rates", "sensitivity_span", "today")


def plant_economics(equipment: list[dict], raw_materials: list[dict], products: list[dict], utilities: list[dict] | None = None,
                    labour: dict | None = None, capital_factors: dict | None = None, basis: dict | None = None,
                    prices: dict | None = None, region: str = "MX", currency: str = "USD", plant_type: str | None = None,
                    main_product: str | None = None, year: int = 2023, cepci: float | None = None,
                    exchange_rates: dict | None = None, sensitivity_span: float = 0.2, today: str | None = None) -> dict:
    """First complete economic picture of a plant from its equipment list and its flows; see the module docstring.
    Every economic input the caller does not give is proposed from a sourced, dated reference and flagged; a price that is
    not in the bank is never filled (it goes to `missing`). Physical data (flows, densities) are always the caller's."""
    call = _clean(dict(equipment=equipment, raw_materials=raw_materials, products=products, utilities=utilities, labour=labour,
                       capital_factors=capital_factors, basis=basis, prices=prices, region=region, currency=currency,
                       plant_type=plant_type, main_product=main_product, year=year, cepci=cepci, exchange_rates=exchange_rates,
                       sensitivity_span=sensitivity_span, today=today))
    with _process_state_kept():
        return _run(call)


# --------------------------------------------------------------------------- update with the user's real data

_MERGE_DICTS = ("basis", "capital_factors", "labour", "prices", "exchange_rates")
_HEADLINE = (("total_capital_investment_usd", lambda r: r["capital_summary"]["total_capital_investment_usd"]),
             ("operating_cost_usd_y", lambda r: r["operating_costs"]["total_usd_y"]),
             ("npv_usd", lambda r: r["npv_usd"]), ("irr", lambda r: r["irr"]),
             ("minimum_selling_price", lambda r: (r["minimum_selling_price"] or {}).get("value")))


def merge_overrides(call: dict, overrides: dict) -> dict:
    """The original call with the user's real data merged in (dicts merge key by key; lists patch their items by name)."""
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("overrides: da un dict con al menos un dato real, con las mismas claves de plant_economics (prices, basis, "
                         "capital_factors, labour, equipment, raw_materials, products, utilities, exchange_rates…).")
    unknown = [k for k in overrides if k not in CALL_KEYS]
    if unknown:
        raise ValueError(f"overrides: claves desconocidas {unknown}. Las válidas son {list(CALL_KEYS)}.")
    new = copy.deepcopy(call)
    for k, v in overrides.items():
        if k in _MERGE_DICTS:
            if not isinstance(v, dict):
                raise ValueError(f"overrides['{k}'] debe ser un dict.")
            new[k] = {**(new.get(k) or {}), **copy.deepcopy(v)}
            if k == "prices":                      # a price given here wins over an older price written inside the item
                names = {_norm(n) for n in v}
                for role in ROLES:
                    for it in new.get(role) or []:
                        if _norm(it.get("name", "")) in names:
                            it.pop("price", None)
                            it.pop("price_source", None)
        elif k in ("equipment",) + ROLES:
            patch = v if isinstance(v, dict) else {str(x.get("name")): {kk: vv for kk, vv in x.items() if kk != "name"} for x in v}
            items = new.get(k) or []
            index = {str(it.get("name")): it for it in items}
            for name, fields in patch.items():
                if name in index:
                    index[name].update(copy.deepcopy(fields))
                elif k == "equipment" and "purchase_cost_usd" in fields or k != "equipment" and "flow" in fields:
                    items.append({"name": name, **copy.deepcopy(fields)})
                else:
                    raise ValueError(f"overrides['{k}']: «{name}» no existe ({sorted(index)}); para agregarlo da el elemento completo.")
            new[k] = items
        else:
            new[k] = copy.deepcopy(v)
    return _clean(new)


def _leaves(overrides: dict) -> list[tuple[str, dict, object]]:
    """(input key, single-leaf override, new value) per changed value, to show its effect alone."""
    out = []
    for k, v in overrides.items():
        if k in _MERGE_DICTS:
            out += [(f"{k}.{kk}", {k: {kk: vv}}, vv) for kk, vv in v.items()]
        elif k in ("equipment",) + ROLES:
            patch = v if isinstance(v, dict) else {str(x.get("name")): {kk: vv for kk, vv in x.items() if kk != "name"} for x in v}
            out += [(f"{k}.{name}", {k: {name: fields}}, fields) for name, fields in patch.items()]
        else:
            out.append((k, {k: v}, v))
    return out


_LABOUR_INPUT = {"labour.positions": "labour.total_salaries_per_year"}


def _value_used(result: dict, call: dict, key: str):
    """(value, origin) an input had in a result: supuesto, usuario, faltante or the caller's own item."""
    key = _LABOUR_INPUT.get(key, key)
    for origin, rows in (("supuesto", result["assumed_inputs"]), ("usuario", result["user_inputs"])):
        for r in rows:
            if r["input"] == key:
                return r["value"], origin
    if any(m["input"] == key for m in result["missing"]):
        return None, "faltante"
    head, _, name = key.partition(".")
    if head in ("equipment",) + ROLES:
        item = next((it for it in call.get(head) or [] if str(it.get("name")) == name), None)
        return (item, "usuario") if item is not None else (None, "no estaba")
    return call.get(key) if key in call else None, "usuario" if call.get(key) is not None else "no estaba"


def plant_economics_update(inputs_resolved: dict, overrides: dict) -> dict:
    """Replace assumptions with the user's real data and show what moved. The new result is exactly `plant_economics` on the
    merged call; `changes` gives, per override, the value before and after and its effect alone and all together."""
    call = inputs_resolved.get("call") if isinstance(inputs_resolved, dict) else None
    if not isinstance(call, dict) or "equipment" not in call:
        raise ValueError("inputs_resolved: pasa tal cual el campo inputs_resolved que devolvió plant_economics (necesita su «call»).")
    merged = merge_overrides(call, overrides)
    with _process_state_kept():
        before = _run(_clean(call), sensitivity=False)
        after = _run(merged)
        head = lambda r: {k: f(r) for k, f in _HEADLINE}
        h0, h1 = head(before), head(after)
        delta = lambda a, b: {k: (b[k] - a[k] if a[k] is not None and b[k] is not None else None) for k in a}
        rows = []
        leaves = _leaves(overrides)
        for key, single, new_value in leaves:
            alone = h1 if len(leaves) == 1 else head(_run(merge_overrides(call, single), sensitivity=False))
            v0, origin = _value_used(before, call, key)
            rows.append({"input": key, "before": v0, "before_origin": origin, "after": new_value, "effect_alone": delta(h0, alone)})
    out = dict(after)
    out["changes"] = {"rows": rows, "before": h0, "after": h1, "effect_together": delta(h0, h1),
                      "n_assumed_before": before["n_assumed"], "n_assumed_after": after["n_assumed"],
                      "n_missing_before": before["n_missing"], "n_missing_after": after["n_missing"],
                      "assumed_remaining": [r["input"] for r in after["assumed_inputs"]],
                      "note": "effect_alone = cambio de cada dato por separado sobre la corrida original; effect_together = todos a la "
                              "vez (no es la suma si los efectos interactúan). «before» se recalcula hoy con la misma llamada original."}
    return _clean(out)


# --------------------------------------------------------------------------- report

def _fmt(x, digits: int = 0, pct: bool = False) -> str:
    if x is None:
        return "—"
    if isinstance(x, (list, dict)):
        return str(x)
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "sí" if x else "no"
    if pct:
        return f"{100 * x:.{digits}f} %"
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.2f} MM"
    if digits != 4:
        return f"{x:,.{digits}f}"
    return f"{x:,.0f}" if abs(x) >= 1000 and float(x).is_integer() else f"{x:,.2f}" if abs(x) >= 100 else f"{x:.4g}"


def plant_report(result: dict, title: str, path: str | None = None) -> dict:
    """Spanish Markdown report of a plant_economics result (or any result with the same keys; absent sections are skipped)."""
    if not isinstance(result, dict) or not ({"capital_summary", "operating_costs", "assumed_inputs"} & set(result)):
        raise ValueError("result: pasa el resultado de plant_economics (o uno con sus claves: capital_summary, operating_costs, "
                         "cash_flow, sensitivity, assumed_inputs, missing).")
    g = result.get
    L = [f"# {title}", "", f"Estimación económica preliminar por factores. Región {g('region', '—')}; cálculo en USD; base de costos de "
         f"equipo: {(g('cost_basis') or {}).get('label', '—')}; fecha {g('today', '—')}.", "",
         f"**{g('n_assumed', len(g('assumed_inputs') or []))} valores son SUPUESTOS de referencias publicadas** (tabla al final) y "
         f"{g('n_missing', len(g('missing') or []))} entradas no se pudieron resolver. Ningún supuesto es una cotización.", ""]
    msp = g("minimum_selling_price")
    L += ["## Resultado", "", "| Indicador | Valor |", "|---|---|",
          f"| Inversión total (TCI) | {_fmt((g('capital_summary') or {}).get('total_capital_investment_usd'))} USD |",
          f"| Costo de operación anual (sin depreciación) | {_fmt((g('operating_costs') or {}).get('total_usd_y'))} USD/año |",
          f"| Ingresos anuales | {_fmt((g('revenue') or {}).get('total_usd_y'))} USD/año |",
          f"| VPN | {_fmt(g('npv_usd'))} USD |", f"| TIR | {_fmt(g('irr'), 2, pct=True)} |",
          f"| Recuperación (desde el arranque) | {_fmt(g('payback_years'), 1)} años |"]
    if msp:
        L.append(f"| Precio mínimo de venta de «{msp['product']}» (VPN = 0) | {_fmt(msp['value'], 4)} {msp['unit']}"
                 f"{' (cota superior)' if msp.get('is_upper_bound') else ''} |")
        if msp.get("current_price") is not None:
            L.append(f"| Precio usado de «{msp['product']}» ({'supuesto' if msp.get('current_price_origin') == 'assumed' else 'tuyo'}) | "
                     f"{_fmt(msp['current_price'], 4)} {msp['unit']} |")
    L.append("")
    cs = g("capital_summary")
    if cs:
        L += ["## Inversión", "", "| Nivel | USD |", "|---|---|"]
        for lab, k in (("Compra de equipo", "total_purchase_cost_usd"), ("Costo instalado (C_TBM, con factor de localización)",
                       "total_bare_module_cost_usd"), ("— del cual ISBL", "isbl_installed_usd"),
                       ("Inversión directa (C_DPI)", "direct_permanent_investment_usd"),
                       ("Capital fijo depreciable (FCI = C_TDC)", "fixed_capital_investment_usd"), ("Terreno", "land_usd"),
                       ("Capital de trabajo", "working_capital_usd"), ("**Inversión total (TCI)**", "total_capital_investment_usd")):
            L.append(f"| {lab} | {_fmt(cs.get(k))} |")
        L.append("")
        eq = (g("capital") or {}).get("equipment") or []
        if eq:
            L += ["| Equipo | Compra, USD | F_BM | Instalado, USD | Fuente del F_BM |", "|---|---|---|---|---|"]
            L += [f"| {e['name']} | {_fmt(e['purchase_cost_usd'])} | {_fmt(e['F_BM'], 2)} | {_fmt(e['bare_module_cost_usd'])} | "
                  f"{e.get('F_BM_source', '')} |" for e in eq]
            L.append("")
    oc = g("operating_costs")
    if oc:
        unit = oc.get("main_product_unit") or "unidad"
        L += ["## Costo de operación", "", f"| Rubro | USD/año | USD por {unit} de producto principal |", "|---|---|---|"]
        L += [f"| {b['label']} | {_fmt(b['usd_per_year'])} | {_fmt(b['per_unit_of_main_product'], 4)} |" for b in oc.get("breakdown", [])]
        L += [f"| **Total{'' if oc.get('complete', True) else ' (INCOMPLETO)'}** | {_fmt(oc.get('total_usd_y'))} | "
              f"{_fmt(oc.get('total_per_unit_of_main_product'), 4)} |", ""]
        items = oc.get("items") or []
        if items:
            L += ["| Corriente | Cantidad anual | Precio | Origen del precio | USD/año |", "|---|---|---|---|---|"]
            org = {"assumed": "SUPUESTO", "user": "tuyo", None: "FALTA"}
            L += [f"| {i['name']} | {_fmt(i['quantity_per_year'])} {i.get('quantity_unit') or ''} | {_fmt(i['price'], 4)} "
                  f"{i.get('price_unit') or ''} | {org.get(i.get('price_origin'), i.get('price_origin'))} | {_fmt(i['usd_per_year'])} |"
                  for i in items]
            L.append("")
    cf = g("cash_flow")
    if cf:
        L += ["## Flujo de caja", "", f"{cf.get('conventions', '')}", "", "| Año | Capital | Ingresos | Costos | Depreciación | Impuesto | "
              "Flujo | Flujo descontado acumulado |", "|---|---|---|---|---|---|---|---|"]
        tab = cf.get("table") or []
        show = tab if len(tab) <= 14 else tab[:8] + tab[-3:]
        for i, r in enumerate(show):
            if len(tab) > 14 and i == 8:
                L.append("| … | | | | | | | |")
            L.append(f"| {r['t']} | {_fmt(r['capital'] + r['working_capital'] + r['land'])} | {_fmt(r['revenue'])} | "
                     f"{_fmt(r['operating_cost'])} | {_fmt(r['depreciation'])} | {_fmt(r['tax'])} | {_fmt(r['cash_flow'])} | "
                     f"{_fmt(r['cumulative_discounted'])} |")
        L.append("")
    sens = g("sensitivity")
    if sens and sens.get("rows"):
        L += ["## Sensibilidad: qué confirmar primero", "", f"Un factor a la vez; ordenado por {sens.get('ranked_by')}.", "",
              "| # | Entrada supuesta | Base | Bajo → alto | VPN bajo → alto, USD | Precio mínimo bajo → alto |", "|---|---|---|---|---|---|"]
        L += [f"| {r['rank']} | {r['label']} (`{r['input']}`) | {_fmt(r['base_value'], 4)} {r['unit']} | {_fmt(r.get('low_value'), 4)} → "
              f"{_fmt(r.get('high_value'), 4)} ({r['range']}) | {_fmt(r.get('npv_usd_low'))} → {_fmt(r.get('npv_usd_high'))} | "
              f"{_fmt(r.get('minimum_selling_price_low'), 4)} → {_fmt(r.get('minimum_selling_price_high'), 4)} |" for r in sens["rows"]]
        L.append("")
    ptu = g("ptu")
    if ptu:
        L += ["## PTU (no incluida en la tasa de impuesto)", "", ptu["why_not_in_rate"], "",
              f"- Tope estimado (3 meses de salarios): {_fmt(ptu.get('cap_estimate_usd_y'))} USD/año → VPN "
              f"{_fmt(ptu.get('npv_usd_with_cap_as_yearly_cost'))} USD.",
              f"- Sin tope (tasa combinada {_fmt(ptu['uncapped_combined_rate'], 1, pct=True)}): VPN "
              f"{_fmt(ptu.get('npv_usd_at_uncapped_combined_rate'))} USD.", ""]
    L += ["## Supuestos por confirmar", "", f"Cada valor de esta tabla es un **{FLAG}**: sustitúyelo con plant_economics_update.", "",
          "| Entrada | Valor | Unidad | Fuente | URL | Periodo | Leído | Nota |", "|---|---|---|---|---|---|---|---|"]
    for a in g("assumed_inputs") or []:
        L.append(f"| {a.get('label', a['input'])} (`{a['input']}`) | {_fmt(a['value'], 4)} | {a.get('unit', '')} | {a.get('source', '')} | "
                 f"{a.get('url', '')} | {a.get('period', '')}{' — VIEJO' if a.get('stale') else ''} | {a.get('date_read', '')} | "
                 f"{str(a.get('note', '')).replace('|', '/')} |")
    L.append("")
    if g("missing"):
        L += ["## Entradas faltantes (no se rellenaron)", "", "| Entrada | Por qué | Qué bloquea |", "|---|---|---|"]
        L += [f"| `{m['input']}` | {str(m['reason']).replace('|', '/')} | {', '.join(m.get('blocks') or []) or 'nada'} |" for m in g("missing")]
        L.append("")
    for head, key in (("Advertencias", "warnings"), ("No incluido", "not_included")):
        if g(key):
            L += [f"## {head}", ""] + [f"- {w}" for w in g(key)] + [""]
    ev = g("evidence") or {}
    L += ["## Evidencia de la corrida", "", f"verified = {ev.get('verified')}; complete = {ev.get('complete')}. {ev.get('scope', '')}",
          ""] + [f"- {r}" for r in ev.get("reasons", [])]
    markdown = "\n".join(L).rstrip() + "\n"
    written = None
    if path:
        from pathlib import Path
        target = Path(path).expanduser()
        if not target.parent.is_dir():
            raise ValueError(f"path: la carpeta {target.parent} no existe.")
        target.write_text(markdown, encoding="utf-8")
        written = str(target)
    return {"markdown": markdown, "path": written, "n_lines": markdown.count("\n"), "title": title,
            "sections": [x[3:] for x in L if x.startswith("## ")]}
