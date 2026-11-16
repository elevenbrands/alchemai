"""Published whole-plant biorefinery models run through BioSTEAM's `biorefineries` package (bloque M26_B).

What it answers: «give me the published sugarcane / corn dry-grind / corn-stover ethanol plant, at its baseline or at my
feed rate, operating hours, prices and TEA parameters»: stream table, product rates, utilities, equipment list with purchase
and installed cost, capital roll-up, operating cost, minimum selling price, IRR and NPV.

Nothing is typed here except (a) the catalogue of models with the numbers PUBLISHED for them (read in the cited sources
on 2026-11-16, used only to report the deviation of the installed package, never to tune it) and (b) numerical tolerances.
Every design and economic value comes from the installed `biorefineries` model (its own defaults, each attributed) or
from the caller.

BioSTEAM keeps global state (thermo package, CEPCI, utility agents and prices, main flowsheet, thermo's
RAISE_PROPERTY_CALCULATION_ERROR flag). `biosteam_session` saves all of it, resets BioSTEAM to its own defaults so a run
never depends on what ran before, and restores the caller's state on exit. A model is rebuilt on every call (0.3-3 s once
BioSTEAM is imported): results cannot depend on call order.
"""
from __future__ import annotations

import math
import re
import sys
import warnings
from contextlib import contextmanager

MASS_CLOSURE_TOL = 5e-3        # |in - out| / in of the whole plant (BioSTEAM closes facilities by specification, not exactly)
NPV_ZERO_TOL = 1e-6            # |NPV at the solved minimum price| / TCI
INTERNAL_STREAM_FRACTION = 0.01   # inter-section streams listed when above this fraction of the main feed
MAX_INTERNAL_STREAMS = 30
MAX_WARNINGS = 25
BENCHMARK_WARN_PCT = 20.0      # baseline deviation from the published number above which every run carries a warning
BIOSTEAM_DEFAULT_CE = 567.5    # biosteam.process_tools.default_CEPCI
ETHANOL_KG_PER_L = 0.789       # biorefineries.cornstover._process_settings.ethanol_density_kgL
L_PER_GAL = 3.78541            # idem (liter_per_gallon)
KG_PER_US_TON = 907.18474
READ = "2026-11-16"

HUMBIRD = ("Humbird et al. 2011, NREL/TP-5100-47764 (Figura ES-1, Tablas 28, 33 y 34; USD de 2007), "
           "https://www.osti.gov/servlets/purl/1013269")
HUANG = ("Huang, Long y Singh 2016, Biofuels Bioprod. Bioref. 10:299-315, caña con 0 % de lípidos (Tablas 4, 5, 6 y 8, Fig. 3; "
         "precios de 2013), https://www.osti.gov/servlets/purl/1437267")
KWIATKOWSKI = ("Kwiatkowski, McAloon, Taylor y Johnston 2006, Ind. Crops Prod. 23:288-296 (solo el RESUMEN, vía Europe PMC "
               "AGR IND43906338; el artículo completo es de pago), "
               "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:IND43906338&resultType=core&format=json")

# What each published model is, where the installed package says it comes from, and the numbers published for it.
PLANTS = {
    "sugarcane_ethanol": {
        "description": "Etanol de caña de azúcar con cogeneración de bagazo (molienda, clarificación, fermentación, destilación, "
                       "deshidratación con tamices, caldera y turbogenerador)",
        "module": "biorefineries.sugarcane", "feed": "sugarcane", "product": "ethanol",
        "model_citation": "biorefineries.sugarcane 2.35.1 (Cortés-Peña et al. 2020, ACS Sustainable Chem. Eng. 8:3302-3310, "
                          "doi 10.1021/acssuschemeng.9b07040; diseño de Huang et al. 2016)",
        "benchmark_source": HUANG,
        "published": {"ethanol_ML_y": 141.0, "ethanol_L_per_t_feed": 88.4, "purchase_cost_usd": 50.3e6, "FCI_usd": 151.0e6,
                      "TCI_usd": 158.5e6, "surplus_electricity_kWh_per_t_feed": 61.0, "unit_production_cost_usd_per_L": 0.48,
                      "feed_t_y": 1.6e6},
    },
    "corn_ethanol": {
        "description": "Etanol de maíz por molienda seca (licuefacción, sacarificación y fermentación simultáneas, destilación, "
                       "DDGS y aceite crudo); vapor, gas natural y electricidad comprados",
        "module": "biorefineries.corn", "feed": "corn", "product": "ethanol",
        "model_citation": "biorefineries.corn 2.35.1 (modelo de Kwiatkowski et al. 2006 según su README)",
        "benchmark_source": KWIATKOWSKI,
        "published": {"ethanol_ML_y": 151.0, "ethanol_kg_y": 119e6},
    },
    "cornstover_ethanol": {
        "description": "Etanol celulósico de rastrojo de maíz (pretratamiento con ácido diluido, hidrólisis enzimática y "
                       "cofermentación, destilación, tratamiento de aguas, caldera y turbogenerador)",
        "module": "biorefineries.cornstover", "feed": "cornstover", "product": "ethanol",
        "model_citation": "biorefineries.cornstover 2.35.1 (Cortés-Peña et al. 2020, doi 10.1021/acssuschemeng.9b07040; diseño "
                          "de Humbird et al. 2011)",
        "benchmark_source": HUMBIRD,
        "published": {"minimum_selling_price_usd_per_gal": 2.15, "ethanol_MMgal_y": 61.0, "ethanol_gal_per_dry_ton": 79.0,
                      "installed_equipment_cost_usd": 232.0e6, "TDC_usd": 250.4e6, "FCI_usd": 400.6e6, "TCI_usd": 422.5e6,
                      "surplus_electricity_kWh_per_gal": 1.8},
    },
}

# caller's TEA key -> (attribute of the BioSTEAM TEA object, unit, meaning)
TEA_KEYS = {
    "discount_rate": ("IRR", "fracción", "tasa de descuento del VPN y del precio mínimo"),
    "income_tax": ("income_tax", "fracción", "impuesto sobre la renta"),
    "plant_life_years": ("duration", "años", "vida de la planta"),
    "depreciation": ("depreciation", "", "esquema de depreciación (MACRS3|5|7|10|15|20, SL<n>, DDB<n>, SYD<n>)"),
    "lang_factor": ("lang_factor", "", "factor de Lang (FCI = factor × costo de compra)"),
    "construction_schedule": ("construction_schedule", "fracciones por año", "calendario de construcción"),
    "working_capital_over_fci": ("WC_over_FCI", "fracción del FCI", "capital de trabajo"),
    "startup_months": ("startup_months", "meses", "arranque"),
    "finance_fraction": ("finance_fraction", "fracción", "parte del capital financiada con deuda"),
    "finance_interest": ("finance_interest", "fracción", "interés del préstamo"),
    "finance_years": ("finance_years", "años", "plazo del préstamo"),
    "labor_cost_usd_y": ("labor_cost", "USD/año", "mano de obra"),
    "maintenance": ("maintenance", "fracción del FCI por año", "mantenimiento"),
    "property_tax": ("property_tax", "fracción del FCI por año", "impuesto predial"),
    "property_insurance": ("property_insurance", "fracción del FCI por año", "seguro"),
    "administration": ("administration", "fracción del FCI por año", "administración"),
}
_MISSING = object()
_pristine: dict = {}
_baseline_benchmark: dict = {}     # plant -> deviation table of the published baseline (pure data)


# ------------------------------------------------------------------------------------------------ BioSTEAM global state

@contextmanager
def biosteam_session():
    """Run BioSTEAM code without leaking global state. Yields (bst, tmo, caught_warnings).

    Saved and restored: thermo's RAISE_PROPERTY_CALCULATION_ERROR flags (read BEFORE thermosteam is imported, because the
    import itself flips one), thermosteam's default thermo, bst.CE, electricity price, heat-utility agents, stream-utility
    prices, flowsheet registry and current flowsheet, ID ticket counters, thermosteam's chemical cache. Inside, BioSTEAM
    starts from its own defaults and with empty chemical caches."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    flags = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import biosteam as bst
        import thermosteam as tmo
        F = bst.main_flowsheet
        registry = F.flowsheet.__dict__
        if not _pristine:
            _pristine.update(stream_utility_prices=dict(bst.stream_utility_prices),
                             raise_flag=TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR)
        saved = {"thermo": getattr(tmo.settings, "_thermo", _MISSING), "CE": bst.CE, "power": bst.PowerUtility.price,
                 "heating": bst.HeatUtility.heating_agents, "cooling": bst.HeatUtility.cooling_agents,
                 "stream_utility_prices": dict(bst.stream_utility_prices), "registry": dict(registry),
                 "chemical_cache": dict(tmo.Chemical.chemical_cache),
                 "flowsheet": registry.get(F.ID), "tickets": [dict(c.ticket_numbers) for c in (bst.Stream, bst.Unit, bst.System)]}
        try:
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = _pristine["raise_flag"]   # what thermosteam expects
            bst.process_tools.default_utilities()      # new agent objects: the caller's are not mutated
            bst.CE = BIOSTEAM_DEFAULT_CE
            bst.stream_utility_prices.clear()
            bst.stream_utility_prices.update(_pristine["stream_utility_prices"])
            for c in (bst.Stream, bst.Unit, bst.System):
                c.ticket_numbers.clear()
            _clear_chemical_caches(tmo)
            yield bst, tmo, caught
        finally:
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flags
            if saved["thermo"] is _MISSING:
                if hasattr(tmo.settings, "_thermo"):
                    del tmo.settings._thermo
            else:
                tmo.settings._thermo = saved["thermo"]
            _clear_chemical_caches(tmo)
            tmo.Chemical.chemical_cache.update(saved["chemical_cache"])
            bst.CE = saved["CE"]
            bst.PowerUtility.price = saved["power"]
            bst.HeatUtility.heating_agents, bst.HeatUtility.cooling_agents = saved["heating"], saved["cooling"]
            bst.stream_utility_prices.clear()
            bst.stream_utility_prices.update(saved["stream_utility_prices"])
            registry.clear()
            registry.update(saved["registry"])
            if saved["flowsheet"] is not None:
                F.set_flowsheet(saved["flowsheet"])
            for c, t in zip((bst.Stream, bst.Unit, bst.System), saved["tickets"]):
                c.ticket_numbers.clear()
                c.ticket_numbers.update(t)


def _clear_chemical_caches(tmo) -> None:
    """thermosteam memoises Chemical objects by ID while a `biorefineries` chemical-set factory runs, and each factory
    memoises its result: a model loaded after another one would silently reuse the other's chemicals (e.g. a Glucose
    defined as liquid by one and as solid by the next), so results would depend on call order. Both caches are emptied."""
    tmo.Chemical.chemical_cache.clear()
    for name, module in list(sys.modules.items()):
        if name.startswith("biorefineries"):
            for obj in list(vars(module).values()):
                for cell in getattr(obj, "__closure__", None) or ():
                    try:
                        inner = cell.cell_contents
                    except ValueError:
                        continue
                    if hasattr(inner, "cache_clear"):
                        inner.cache_clear()


def translate_warnings(caught) -> list[str]:
    """BioSTEAM / thermosteam cost, design and numerical warnings raised during the run, in Spanish, deduplicated."""
    out: list[str] = []
    for w in caught:
        kind = w.category.__name__
        text = re.sub(r"\x1b\[[0-9;]*m", "", re.sub(r"\s+", " ", str(w.message))).strip()
        if kind not in ("CostWarning", "DesignWarning", "RuntimeWarning", "UserWarning", "ConvergenceWarning") or any(
                k in text for k in ("cached chemical returned", "has been replaced in registry")):
            continue
        m = re.search(r"<(\w+): ([^>]*)> (.*?) \(([^()]*)\) is out of bounds \(([^()]*)\)", text)
        if m:
            msg = (f"{m.group(2)} ({m.group(1)}): {m.group(3)} = {m.group(4)} queda fuera del intervalo de validez de la "
                   f"correlación de costo de BioSTEAM ({m.group(5)}); el costo de ese equipo es una extrapolación")
        elif "Dortmund groups" in text:
            msg = f"BioSTEAM: {text.split(' has no')[0]} no tiene grupos UNIFAC-Dortmund; se ignoran sus interacciones en el equilibrio"
        elif "vacuum pressure vessel" in text:
            msg = ("BioSTEAM (aviso de diseño): no implementa el código ASME para recipientes a vacío; el espesor de pared de "
                   "esos recipientes puede ser inexacto y pueden requerir anillos de refuerzo")
        else:
            label = {"CostWarning": "aviso de costo", "DesignWarning": "aviso de diseño",
                     "RuntimeWarning": "aviso numérico", "UserWarning": "aviso"}.get(kind, kind)
            msg = f"BioSTEAM ({label}): «{text}»"
        if msg not in out:
            out.append(msg)
    if len(out) > MAX_WARNINGS:
        out = out[:MAX_WARNINGS] + [f"… y {len(out) - MAX_WARNINGS} avisos más de BioSTEAM"]
    return out


def resolve_cepci(year: int | None, cepci: float | None, model_ce: float, model_ce_label: str,
                  model_year: int | None = None) -> tuple[float, str, int | None]:
    """Cost index of the run: the caller's `cepci`, else IDAES's table for `year` (repo convention), else the model's own."""
    if cepci is not None:
        if not (isinstance(cepci, (int, float)) and cepci > 0):
            raise ValueError("cepci debe ser un índice CEPCI positivo")
        return float(cepci), f"CEPCI {float(cepci):g} (dado por el usuario)", year
    if year is not None:
        from packages.costing.sslw import CEPCI_IDAES
        if year not in CEPCI_IDAES:
            raise ValueError(f"la tabla CEPCI del repositorio (IDAES) cubre {min(CEPCI_IDAES)}-{max(CEPCI_IDAES)}; para {year} dé "
                             "cepci=<valor del índice>")
        return CEPCI_IDAES[year], f"CEPCI {CEPCI_IDAES[year]} ({year}, tabla de IDAES)", year
    return float(model_ce), f"CEPCI {model_ce:g} ({model_ce_label})", model_year


# ------------------------------------------------------------------------------------------------------- model loaders

def _load_sugarcane(bst):
    import inspect

    from biorefineries import sugarcane as sc
    from biorefineries.tea.conventional_ethanol_tea import create_conventional_ethanol_tea
    sc.load()
    tea = sc.sugarcane_tea
    tea.IRR = inspect.signature(create_conventional_ethanol_tea).parameters["IRR"].default   # load() leaves the solved IRR
    return sc.sugarcane_sys, tea, "CEPCI de 2019 según biorefineries.cane.process_settings", 2019, []


def _load_corn(bst):
    from biorefineries import corn as cn
    cls = cn.units.SSF
    had = "_ins_size_is_fixed" in cls.__dict__
    old = cls.__dict__.get("_ins_size_is_fixed")
    cls._ins_size_is_fixed = False       # compatibility shim, see the assumption below
    try:
        br = cn.Biorefinery(cache={})
    finally:
        if had:
            cls._ins_size_is_fixed = old
        else:
            del cls._ins_size_is_fixed
    note = ("biorefineries.corn 2.35.1 no carga tal cual con BioSTEAM 2.54 (su fermentador SSF recibe 2 corrientes y la clase "
            "base BatchBioreactor ahora fija 1 entrada): se levanta ese límite de número de entradas durante la construcción "
            "(atributo _ins_size_is_fixed), sin tocar ningún dato del modelo")
    tea = br.TEA
    tea.IRR = cn.default_tea_parameters["IRR"]        # Biorefinery() leaves the solved IRR
    return br.system, tea, "CEPCI 567 «de 2013» según biorefineries.corn.process_settings", 2013, [note]


def _load_cornstover(bst):
    from biorefineries import cellulosic
    br = cellulosic.Biorefinery("corn stover ethanol", load=False, save=False)
    return br.sys, br.tea, "CEPCI de 2007 según biorefineries.cellulosic.process_settings", 2007, []


_LOADERS = {"sugarcane_ethanol": _load_sugarcane, "corn_ethanol": _load_corn, "cornstover_ethanol": _load_cornstover}


# ------------------------------------------------------------------------------------------------- overrides (caller)

def _positive(x, name: str, what: str) -> float:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} debe ser un número positivo ({what})")
    return float(x)


def price_keys(system, bst) -> dict:
    """Price keys a caller may override: every feed/product stream by ID, plus electricity, steam and natural gas."""
    keys = {s.ID: s for s in (*system.feeds, *system.products) if s.ID and not re.fullmatch(r"s\d+", s.ID)}
    return keys


def apply_prices(prices: dict | None, system, bst, basis: list) -> None:
    if not prices:
        return
    if not isinstance(prices, dict):
        raise ValueError("prices debe ser un diccionario {clave: precio}")
    keys = price_keys(system, bst)
    for k, v in prices.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"prices[{k!r}] debe ser un número (USD/kg; USD/kWh para electricity)")
        if k == "electricity":
            bst.PowerUtility.price = float(v)
            basis.append({"parameter": "precio de la electricidad", "value": float(v), "unit": "USD/kWh", "origin": "usuario"})
        elif k == "steam":
            for a in bst.HeatUtility.heating_agents:
                if "steam" in a.ID:
                    a.regeneration_price = float(v) * 18.01528
            basis.append({"parameter": "precio del vapor comprado (todas las presiones)", "value": float(v), "unit": "USD/kg",
                          "origin": "usuario"})
        elif k == "natural_gas":
            bst.stream_utility_prices["Natural gas"] = float(v)
            basis.append({"parameter": "precio del gas natural", "value": float(v), "unit": "USD/kg", "origin": "usuario"})
        elif k in keys:
            keys[k].price = float(v)
            basis.append({"parameter": f"precio de {k}", "value": float(v), "unit": "USD/kg", "origin": "usuario"})
        else:
            raise ValueError(f"prices[{k!r}] no es una corriente con precio de este modelo; claves válidas: "
                             f"{sorted(keys) + ['electricity', 'natural_gas', 'steam']}")


def apply_tea(tea_overrides: dict | None, tea, basis: list) -> None:
    if not tea_overrides:
        return
    if not isinstance(tea_overrides, dict):
        raise ValueError("tea debe ser un diccionario {parámetro: valor}")
    for k, v in tea_overrides.items():
        if k not in TEA_KEYS:
            raise ValueError(f"tea[{k!r}] no es un parámetro reconocido; válidos: {sorted(TEA_KEYS)}")
        attr, unit, meaning = TEA_KEYS[k]
        if not hasattr(tea, attr) or (attr == "lang_factor" and tea.lang_factor is None):
            raise ValueError(f"tea[{k!r}] ({meaning}) no existe en el análisis económico de este modelo "
                             f"({type(tea).__name__}); p. ej. el modelo de rastrojo usa factores de instalación por equipo, "
                             "no un factor de Lang")
        if k == "plant_life_years":
            n = int(_positive(v, "tea['plant_life_years']", "años de operación"))
            tea.duration = (tea.duration[0], tea.duration[0] + n)
        elif k == "depreciation":
            if not isinstance(v, str):
                raise ValueError("tea['depreciation'] debe ser texto, p. ej. 'MACRS7' o 'SL10'")
            try:
                tea.depreciation = v
            except Exception as e:  # BioSTEAM's own validation message
                raise ValueError(f"tea['depreciation']={v!r} no es un esquema que BioSTEAM acepte ({e})") from None
        elif k == "construction_schedule":
            sched = [float(x) for x in v]
            if not sched or abs(sum(sched) - 1.0) > 1e-9 or min(sched) < 0:
                raise ValueError("tea['construction_schedule'] debe ser una lista de fracciones no negativas que sumen 1")
            tea.construction_schedule = sched
        else:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f"tea[{k!r}] debe ser un número no negativo ({meaning}, {unit})")
            setattr(tea, attr, float(v))
        basis.append({"parameter": meaning, "value": v, "unit": unit, "origin": "usuario"})


# ---------------------------------------------------------------------------------------------------------- extraction

def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _section(unit, bst) -> str:
    m = re.search(r"(\d)\d\d$", unit.ID) or re.search(r"(\d)\d\d", unit.ID)
    if m:
        return f"{m.group(1)}00"
    return "servicios" if isinstance(unit, bst.Facility) else "otros"


def _stream_row(s, role: str, hours: float, section_of: dict) -> dict:
    total = s.F_mass
    comp = {}
    if total > 0:
        for c in s.available_chemicals:
            w = s.imass[c.ID] / total
            if w >= 1e-3:
                comp[c.ID] = round(float(w), 5)
    comp = dict(sorted(comp.items(), key=lambda kv: -kv[1])[:8])
    src, snk = getattr(s, "source", None), getattr(s, "sink", None)
    return {"id": s.ID, "role": role, "flow_kg_h": float(total), "T_K": float(s.T), "P_Pa": float(s.P), "phase": s.phase,
            "mass_fractions": comp, "from_unit": src.ID if src else None, "to_unit": snk.ID if snk else None,
            "from_section": section_of.get(src.ID) if src else None, "to_section": section_of.get(snk.ID) if snk else None,
            "price_usd_kg": float(s.price or 0.0), "value_usd_y": float((s.price or 0.0) * total * hours)}


def _hhv_W(stream) -> float | None:
    """Higher heating value flow. HHV, not LHV: thermosteam gives liquid water a negative LHV (its latent heat), which would
    poison a sum over process-water feeds."""
    try:
        return _f(stream.HHV / 3.6)          # kJ/h -> W
    except Exception:
        return None


def extract_results(bst, system, tea, feed, product, cost_basis: str, cepci: float, cost_year: int | None,
                    fbm_note: str) -> dict:
    """Plain-dict picture of a simulated BioSTEAM system and its TEA (shared by the reference plants and the biodiesel plant)."""
    hours = float(tea.operating_days * 24.0)
    units = list(system.units)
    section_of = {u.ID: _section(u, bst) for u in units}

    feeds = [s for s in system.feeds if s.F_mass > 0]
    prods = [s for s in system.products if s.F_mass > 0]
    streams = [_stream_row(s, "alimentación", hours, section_of) for s in sorted(feeds, key=lambda s: -s.F_mass)]
    streams += [_stream_row(s, "producto principal" if s is product else "salida", hours, section_of)
                for s in sorted(prods, key=lambda s: -s.F_mass)]
    internal = [s for s in system.streams if s.source and s.sink and s.F_mass >= INTERNAL_STREAM_FRACTION * feed.F_mass
                and section_of.get(s.source.ID) != section_of.get(s.sink.ID)]
    internal.sort(key=lambda s: -s.F_mass)
    streams += [_stream_row(s, "entre secciones", hours, section_of) for s in internal[:MAX_INTERNAL_STREAMS]]

    # utilities: consumers have positive agent flow; on-site producers (boiler, cooling tower) negative
    heat: dict = {}
    power_use = power_made = 0.0
    sections: dict = {}
    equipment = []
    for u in units:
        sec = sections.setdefault(section_of[u.ID], {"section": section_of[u.ID], "unit_ids": [], "unit_types": [],
                                                     "purchase_cost_usd": 0.0, "installed_cost_usd": 0.0, "power_W": 0.0,
                                                     "heating_duty_W": 0.0, "cooling_duty_W": 0.0})
        sec["unit_ids"].append(u.ID)
        if u.line not in sec["unit_types"]:
            sec["unit_types"].append(u.line)
        pu = u.power_utility
        power_use += pu.consumption * 1e3
        power_made += pu.production * 1e3
        sec["power_W"] += pu.consumption * 1e3
        for hu in u.heat_utilities:
            if hu.agent is None or hu.flow == 0:
                continue
            row = heat.setdefault(hu.agent.ID, {"agent": hu.agent.ID, "demand_duty_W": 0.0, "demand_flow_kg_h": 0.0,
                                                "onsite_production_duty_W": 0.0, "cost_usd_y": 0.0,
                                                "kind": "calentamiento" if hu.agent in bst.HeatUtility.heating_agents
                                                else "enfriamiento"})
            duty_W = abs(hu.duty) / 3.6
            if hu.flow > 0:
                row["demand_duty_W"] += duty_W
                row["demand_flow_kg_h"] += hu.flow * hu.agent.MW
                sec["heating_duty_W" if row["kind"] == "calentamiento" else "cooling_duty_W"] += duty_W
            else:
                row["onsite_production_duty_W"] += duty_W
            row["cost_usd_y"] += hu.cost * hours
        C_p, C_bm, C_base = float(u.purchase_cost or 0.0), float(u.installed_cost or 0.0), float(u.baseline_purchase_cost or 0.0)
        if C_p > 0:
            sec["purchase_cost_usd"] += C_p
            sec["installed_cost_usd"] += C_bm
            if not 0.0 < C_base <= C_p:      # capital_investment needs 0 < base <= purchase (BioSTEAM factors can be < 1)
                C_base = C_p
            f_bm = 1.0 + (C_bm - C_p) / C_base     # so that C_P + C_base·(F_BM − 1) is BioSTEAM's installed cost exactly
            equipment.append({"name": u.ID, "description": u.line, "section": section_of[u.ID], "F_BM": f_bm,
                              "purchase_cost_usd": C_p, "base_purchase_cost_usd": C_base,
                              "installed_cost_usd": C_bm, "cepci": cepci, "cost_year": cost_year,
                              "basis": f"{cost_basis}; correlación de costo de BioSTEAM {bst.__version__}; F_BM efectivo de "
                                       f"BioSTEAM ({fbm_note})"})
    section_rows = sorted(sections.values(), key=lambda r: r["section"])

    # capital
    purchase = float(tea.purchase_cost)
    installed_bm = float(sum(e["installed_cost_usd"] for e in equipment))
    lang = getattr(tea, "lang_factor", None)
    capital = {"purchase_cost_usd": purchase, "installed_equipment_cost_bm_usd": installed_bm,
               "DPI_usd": float(tea.DPI), "TDC_usd": float(tea.TDC), "FCI_usd": float(tea.FCI),
               "working_capital_usd": float(tea.TCI - tea.FCI), "TCI_usd": float(tea.TCI), "cepci": cepci,
               "cost_year": cost_year, "basis": cost_basis,
               "method": (f"FCI = factor de Lang {lang:g} × costo de compra (análisis {type(tea).__name__}); el costo instalado "
                          "por equipo con F_BM se da como referencia y NO es el que usa este FCI") if lang else
               (f"costo instalado = Σ compra × F_BM de cada equipo; TDC, FCI y TCI con los factores de {type(tea).__name__} "
                "(almacén, sitio, tubería, indirectos, contingencia)")}
    if lang is None:
        for k in ("warehouse", "site_development", "additional_piping", "proratable_costs", "field_expenses", "construction",
                  "contingency", "other_indirect_costs"):
            if hasattr(tea, k):
                capital.setdefault("factors", {})[k] = float(getattr(tea, k))

    # operating cost and economics
    material = [{"stream": s.ID, "flow_kg_h": float(s.F_mass), "price_usd_kg": float(s.price), "cost_usd_y": float(s.cost * hours)}
                for s in feeds if s.price]
    sales = [{"stream": s.ID, "flow_kg_h": float(s.F_mass), "price_usd_kg": float(s.price), "sales_usd_y": float(s.cost * hours),
              "main_product": s is product} for s in prods if s.price]
    coproduct_credit = sum(r["sales_usd_y"] for r in sales if not r["main_product"])
    life = int(tea.duration[1] - tea.duration[0])
    product_kg_y = float(product.F_mass * hours)
    depreciation_sl = float(tea.FCI) / life
    net_cost = float(tea.AOC) + depreciation_sl - coproduct_credit
    operating = {"material_cost_usd_y": float(tea.material_cost), "utility_cost_usd_y": float(tea.utility_cost),
                 "variable_operating_cost_usd_y": float(tea.VOC), "fixed_operating_cost_usd_y": float(tea.FOC),
                 "annual_operating_cost_usd_y": float(tea.AOC), "materials": material, "sales": sales,
                 "total_sales_usd_y": float(tea.sales), "coproduct_credit_usd_y": float(coproduct_credit),
                 "electricity_price_usd_kWh": float(bst.PowerUtility.price),
                 "note": "utility_cost es neto: la electricidad vendida aparece como costo negativo"}
    for k in ("labor_cost", "fringe_benefits", "labor_burden", "supplies", "maintenance", "property_tax", "property_insurance",
              "administration"):
        if hasattr(tea, k):
            operating.setdefault("fixed_cost_parameters", {})[k] = _f(getattr(tea, k))
    production_cost = {"straight_line_depreciation_usd_y": depreciation_sl, "net_cost_usd_y": net_cost,
                       "unit_production_cost_usd_kg": net_cost / product_kg_y if product_kg_y > 0 else None,
                       "definition": "(costo anual de operación + depreciación lineal FCI/vida − ventas de coproductos) / "
                                     "producto principal; convención de Huang et al. 2016 (Tabla 8)"}

    discount = float(tea.IRR)
    npv = float(tea.NPV)
    try:
        irr = _f(tea.solve_IRR())
    except Exception:
        irr = None
    msp = npv_at_msp = None
    try:
        msp = float(tea.solve_price(product))
        old = product.price
        product.price = msp
        npv_at_msp = float(tea.NPV)
        product.price = old
    except Exception:
        pass
    economics = {"discount_rate": discount, "npv_usd": npv, "irr": irr, "minimum_selling_price_usd_kg": msp,
                 "product_price_usd_kg": float(product.price or 0.0), "plant_life_years": life,
                 "income_tax": float(tea.income_tax), "depreciation": str(tea.depreciation),
                 "operating_hours_y": hours,
                 "note": "npv_usd e irr son a los precios indicados; el precio mínimo de venta es el que anula el VPN a la "
                         "tasa de descuento"}

    # evidence
    m_in, m_out = sum(s.F_mass for s in system.feeds), sum(s.F_mass for s in system.products)
    mass_res = abs(m_in - m_out) / m_in if m_in > 0 else math.inf
    net_export = power_made - power_use
    lhv_in = [_hhv_W(s) for s in feeds]
    lhv_main = _hhv_W(product)
    purchased_heat = sum(r["demand_duty_W"] - r["onsite_production_duty_W"] for r in heat.values() if r["kind"] == "calentamiento")
    purchased_heat = max(purchased_heat, 0.0)
    first_law = None
    if lhv_main is not None and all(v is not None for v in lhv_in) and sum(lhv_in) > 0:
        e_in = sum(lhv_in) + max(-net_export, 0.0) + purchased_heat
        first_law = (lhv_main + max(net_export, 0.0)) / e_in
    reasons = []
    if not mass_res <= MASS_CLOSURE_TOL:
        reasons.append(f"el balance de masa global no cierra: residuo relativo {mass_res:.2e} > {MASS_CLOSURE_TOL:g}")
    if first_law is not None and not 0.0 < first_law < 1.0:
        reasons.append(f"la energía útil (PCS del producto + electricidad exportada) es {first_law:.3f} veces la energía "
                       "que entra: viola la primera ley")
    npv_res = abs(npv_at_msp) / float(tea.TCI) if npv_at_msp is not None and tea.TCI > 0 else None
    if npv_res is None:
        reasons.append("BioSTEAM no pudo resolver el precio mínimo de venta")
    elif npv_res > NPV_ZERO_TOL:
        reasons.append(f"el VPN al precio mínimo no es cero: |VPN|/TCI = {npv_res:.2e}")
    evidence = {"verified": not reasons, "reasons": reasons, "mass_in_kg_h": float(m_in), "mass_out_kg_h": float(m_out),
                "mass_balance_relative_residual": float(mass_res), "mass_balance_tolerance": MASS_CLOSURE_TOL,
                "first_law_useful_energy_over_energy_in": first_law,
                "first_law_definition": "(PCS del producto principal + electricidad neta exportada) / (PCS de todas las "
                                        "alimentaciones + electricidad y calor comprados); debe quedar entre 0 y 1",
                "npv_at_minimum_price_over_tci": npv_res,
                "installed_cost_sum_vs_tea_relative_difference": abs(installed_bm - float(tea.installed_equipment_cost))
                / float(tea.installed_equipment_cost) if lang is None and tea.installed_equipment_cost else None}

    from packages.diagram.adapters import biosteam_streams     # unit-to-unit topology, so the full PFD view is connected
    internal = biosteam_streams(system)
    return {"streams": streams, "internal_streams": internal, "sections": section_rows,
            "products": {"main_product": product.ID, "main_product_kg_h": float(product.F_mass),
                         "main_product_kg_y": product_kg_y, "feed": feed.ID, "feed_kg_h": float(feed.F_mass),
                         "product_per_feed_kg_kg": float(product.F_mass / feed.F_mass) if feed.F_mass > 0 else None,
                         "coproducts": [{"stream": r["stream"], "flow_kg_h": r["flow_kg_h"]} for r in sales
                                        if not r["main_product"]]},
            "utilities": {"power_consumption_W": power_use, "power_production_W": power_made,
                          "net_electricity_export_W": net_export, "heat_utilities": sorted(heat.values(), key=lambda r: r["agent"]),
                          "purchased_heating_duty_W": purchased_heat},
            "equipment": equipment, "capital": capital, "operating_cost": operating, "production_cost": production_cost,
            "economics": economics, "evidence": evidence}


def _benchmark(plant: str, r: dict, product, feed, hours: float) -> list[dict]:
    """Model value next to each number published for the plant (only meaningful at the published baseline)."""
    pub = PLANTS[plant]["published"]
    ethanol_kg_h = float(product.imass["Ethanol"])
    ethanol_L_y = ethanol_kg_h * hours / ETHANOL_KG_PER_L
    feed_t_h = float(feed.F_mass) / 1e3
    dry_ton_h = float(feed.F_mass - feed.imass["Water"]) / KG_PER_US_TON
    export_kW = r["utilities"]["net_electricity_export_W"] / 1e3
    msp = r["economics"]["minimum_selling_price_usd_kg"]
    upc = r["production_cost"]["unit_production_cost_usd_kg"]
    model = {
        "ethanol_ML_y": ethanol_L_y / 1e6, "ethanol_kg_y": ethanol_kg_h * hours,
        "ethanol_MMgal_y": ethanol_L_y / L_PER_GAL / 1e6, "ethanol_L_per_t_feed": ethanol_kg_h / ETHANOL_KG_PER_L / feed_t_h,
        "ethanol_gal_per_dry_ton": ethanol_kg_h / ETHANOL_KG_PER_L / L_PER_GAL / dry_ton_h,
        "purchase_cost_usd": r["capital"]["purchase_cost_usd"], "FCI_usd": r["capital"]["FCI_usd"],
        "TCI_usd": r["capital"]["TCI_usd"], "TDC_usd": r["capital"]["TDC_usd"],
        "installed_equipment_cost_usd": r["capital"]["installed_equipment_cost_bm_usd"],
        "surplus_electricity_kWh_per_t_feed": export_kW / feed_t_h,
        "surplus_electricity_kWh_per_gal": export_kW / (ethanol_kg_h / ETHANOL_KG_PER_L / L_PER_GAL),
        "minimum_selling_price_usd_per_gal": msp * ETHANOL_KG_PER_L * L_PER_GAL if msp is not None else None,
        "unit_production_cost_usd_per_L": upc * ETHANOL_KG_PER_L if upc is not None else None,
        "feed_t_y": feed_t_h * hours,
    }
    rows = []
    for k, v in pub.items():
        mv = model.get(k)
        rows.append({"quantity": k, "published": v, "model": mv,
                     "deviation_pct": 100.0 * (mv - v) / v if mv is not None else None})
    return rows


# ---------------------------------------------------------------------------------------------------------- public API

def list_reference_plants() -> dict:
    """Catalogue of the published plant models, their citations and the numbers published for them (no simulation)."""
    plants = [{"plant": k, "description": v["description"], "feed_stream": v["feed"], "product_stream": v["product"],
               "model": v["model_citation"], "published_benchmark_source": v["benchmark_source"],
               "published_values": v["published"]} for k, v in PLANTS.items()]
    return {"plants": plants, "tea_parameters": {k: f"{m} [{u}]" if u else m for k, (_, u, m) in TEA_KEYS.items()},
            "price_keys": "ID de cualquier corriente de alimentación o producto del modelo (aparecen en streams) más "
                          "'electricity' (USD/kWh), 'steam' y 'natural_gas' (USD/kg)",
            "warnings": [], "assumptions": [f"valores publicados leídos en las fuentes citadas el {READ}"],
            "not_included": ["otros modelos de `biorefineries` (oilcane, lipidcane, lactic, LAOs…): no se validaron en este "
                             "bloque"],
            "method": "catálogo estático; los modelos se ejecutan con reference_biorefinery",
            "evidence": {"verified": True, "reasons": []}}


def reference_biorefinery(plant: str, feed_flow_kg_h: float | None = None, operating_hours_y: float | None = None,
                          prices: dict | None = None, tea: dict | None = None, year: int | None = None,
                          cepci: float | None = None) -> dict:
    """Run a published whole-plant model at its baseline or rescaled to the caller's feed, hours, prices and TEA parameters.

    With `year` and `cepci` both None the capital is in the model's own cost year (the published baseline); otherwise every
    purchase cost is computed at the caller's CEPCI while prices not overridden stay in the model's price basis (flagged)."""
    if plant not in PLANTS:
        raise ValueError(f"plant={plant!r} no está disponible; opciones: {sorted(PLANTS)}")
    meta = PLANTS[plant]
    baseline = feed_flow_kg_h is None and operating_hours_y is None and not prices and not tea and year is None and cepci is None
    if feed_flow_kg_h is not None:
        _positive(feed_flow_kg_h, "feed_flow_kg_h", "kg/h de la alimentación principal, tal como entra (húmeda)")
    if operating_hours_y is not None:
        if _positive(operating_hours_y, "operating_hours_y", "horas de operación al año") > 8784:
            raise ValueError("operating_hours_y no puede exceder 8784 h (un año)")
    with biosteam_session() as (bst, tmo, caught):
        system, tea_obj, ce_label, model_year, notes = _LOADERS[plant](bst)
        fs = system.flowsheet.stream
        feed, product = fs[meta["feed"]], fs[meta["product"]]
        model_ce = float(bst.CE)
        basis = [
            {"parameter": f"alimentación ({feed.ID})", "value": float(feed.F_mass), "unit": "kg/h", "origin": "modelo publicado"},
            {"parameter": "horas de operación", "value": float(tea_obj.operating_days * 24), "unit": "h/año",
             "origin": "modelo publicado"},
            {"parameter": "tasa de descuento", "value": float(tea_obj.IRR), "unit": "fracción", "origin": "modelo publicado"},
            {"parameter": "impuesto sobre la renta", "value": float(tea_obj.income_tax), "unit": "fracción",
             "origin": "modelo publicado"},
            {"parameter": "vida de la planta", "value": int(tea_obj.duration[1] - tea_obj.duration[0]), "unit": "años",
             "origin": "modelo publicado"},
            {"parameter": "depreciación", "value": str(tea_obj.depreciation), "unit": "", "origin": "modelo publicado"},
            {"parameter": "precio de la electricidad", "value": float(bst.PowerUtility.price), "unit": "USD/kWh",
             "origin": "modelo publicado"},
        ]
        if getattr(tea_obj, "lang_factor", None):
            basis.append({"parameter": "factor de Lang", "value": float(tea_obj.lang_factor), "unit": "",
                          "origin": "modelo publicado"})
        for s in (*system.feeds, *system.products):
            if s.price and s.F_mass > 0:
                basis.append({"parameter": f"precio de {s.ID}", "value": float(s.price), "unit": "USD/kg",
                              "origin": "modelo publicado"})
        for b in basis:
            b["source"] = meta["model_citation"]
        n_default = len(basis)
        target_ce, cost_basis, cost_year = resolve_cepci(year, cepci, model_ce, ce_label, model_year)
        if feed_flow_kg_h is not None:
            feed.F_mass = float(feed_flow_kg_h)
            basis.append({"parameter": f"alimentación ({feed.ID})", "value": float(feed_flow_kg_h), "unit": "kg/h",
                          "origin": "usuario"})
        if operating_hours_y is not None:
            tea_obj.operating_days = float(operating_hours_y) / 24.0
            basis.append({"parameter": "horas de operación", "value": float(operating_hours_y), "unit": "h/año",
                          "origin": "usuario"})
        apply_prices(prices, system, bst, basis)
        apply_tea(tea, tea_obj, basis)
        overridden = {b["parameter"] for b in basis[n_default:]}
        for b in basis[:n_default]:
            if b["parameter"] in overridden:
                b["origin"] = "modelo publicado (sustituido por el usuario)"
        sim_error = None
        if not baseline:
            bst.CE = target_ce
            try:
                system.simulate()
            except Exception as e:          # reported, never hidden
                sim_error = f"{type(e).__name__}: {e}"
        hours = float(tea_obj.operating_days * 24.0)
        fbm = ("factores de instalación de Humbird et al. 2011 que trae cada unidad" if plant == "cornstover_ethanol"
               else "factores de módulo desnudo por tipo de equipo que trae cada unidad")
        out = extract_results(bst, system, tea_obj, feed, product, cost_basis, target_ce, cost_year, fbm)
        bench = _benchmark(plant, out, product, feed, hours)
        ethanol_kg_h = float(product.imass["Ethanol"])
        bst_warnings = translate_warnings(caught)
        version = f"BioSTEAM {bst.__version__}, biorefineries 2.35.1"
    msp = out["economics"]["minimum_selling_price_usd_kg"]
    out["economics"].update(
        minimum_selling_price_usd_per_L=msp * ETHANOL_KG_PER_L if msp is not None else None,
        minimum_selling_price_usd_per_gal=msp * ETHANOL_KG_PER_L * L_PER_GAL if msp is not None else None,
        volume_basis=f"{ETHANOL_KG_PER_L} kg/L y {L_PER_GAL} L/gal (constantes de biorefineries); el precio es por kg de la "
                     "corriente de producto tal como sale (etanol desnaturalizado)")
    out["products"]["ethanol_pure_kg_h"] = ethanol_kg_h
    out["products"]["ethanol_pure_L_y"] = ethanol_kg_h * hours / ETHANOL_KG_PER_L
    if sim_error:
        out["evidence"]["verified"] = False
        out["evidence"]["reasons"].insert(0, f"la simulación de BioSTEAM no convergió o falló ({sim_error}); los números "
                                             "siguientes NO son un balance resuelto")
    if baseline:
        _baseline_benchmark[plant] = bench
    elif plant not in _baseline_benchmark:
        reference_biorefinery(plant)                 # pure data, cached per process: cannot change later results
    bench = _baseline_benchmark[plant]
    warns = list(bst_warnings)
    for row in bench:
        if row["deviation_pct"] is not None and abs(row["deviation_pct"]) > BENCHMARK_WARN_PCT:
            warns.append(f"el modelo instalado, en su línea base, se aparta {row['deviation_pct']:+.0f} % del valor publicado de "
                         f"{row['quantity']} ({row['model']:.4g} frente a {row['published']:.4g}; {meta['benchmark_source']})")
    if not baseline and (year is not None or cepci is not None) and abs(target_ce - model_ce) > 1e-9:
        warns.append(f"el capital está a {cost_basis} pero los precios que usted no dio siguen en la base del modelo "
                     f"publicado ({ce_label}): mezcla de años; dé también los precios de ese año")
    if feed_flow_kg_h is not None:
        warns.append("planta reescalada: BioSTEAM redimensiona y recostea cada equipo con sus correlaciones; la mano de obra y "
                     "los demás costos fijos del modelo NO se reescalan solos (dé tea['labor_cost_usd_y'])")
    out = {"plant": plant, "description": meta["description"], "baseline": baseline, "design_basis": basis, **out,
           "published_benchmark": {"source": meta["benchmark_source"], "date_read": READ,
                                   "note": "valores del modelo en su LÍNEA BASE (sin argumentos opcionales) frente a los "
                                           "publicados; no cambian con sus argumentos",
                                   "values": bench},
           "warnings": warns,
           "assumptions": [f"todo valor de diseño y económico no dado por usted es el del modelo publicado: {meta['model_citation']}",
                           "precios y parámetros económicos del modelo: SUGERIDOS, sustituibles con prices= y tea=",
                           f"costos de equipo en {cost_basis}", *notes],
           "not_included": ["terreno, permisos e ingeniería específica del sitio más allá de los factores del modelo",
                            "factor de localización (los modelos son de EE. UU.)",
                            "análisis de incertidumbre (Monte Carlo) del modelo original",
                            "cualquier cambio de composición de la materia prima: se usa la del modelo publicado"],
           "method": f"modelo de planta completa de {meta['module']} resuelto con {version} (balances de materia y energía "
                     "secuenciales modulares, dimensionamiento y costo por unidad, flujo de caja descontado de BioSTEAM); "
                     "reconstruido en cada llamada dentro de una sesión que restaura el estado global de BioSTEAM"}
    return out
