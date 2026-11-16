"""Solids handling and oil extraction (bloque M26_E3): size reduction, mechanical conveying, solid-liquid separation and
oil extraction by expeller and by countercurrent solvent leaching.

What it answers
  * `size_reduction`: specific energy and power to grind/crush by Bond (work index of the USER), Kick or Rittinger (constant
    of the user or calibrated on the user's own test), and the purchase cost of a hammer mill / shredder / cane crushing
    mill with BioSTEAM's correlations.
  * `solids_conveying`: shaft power of a belt conveyor, a screw conveyor and a bucket elevator, and purchase cost of the
    belt and the screw (BioSTEAM `ConveyingBelt`, `ScrewFeeder`).
  * `solid_liquid_separation`: outlet split from the USER's solids recovery and cake moisture; filter area by
    constant-pressure filtration theory (rotary vacuum drum, plate-and-frame); Sigma theory for sedimenting centrifuges
    (tubular / solid bowl and disc stack); purchase cost with BioSTEAM's `solids_separation` correlations.
  * `oil_extraction`: (a) expeller mass balance seed -> crude oil + cake (+ hull); (b) hexane extraction as ideal
    countercurrent leaching stages with constant or variable underflow, miscella, solvent make-up and stripping heat.

Equations
  Bond            e = 10 Wi (1/sqrt(P80) - 1/sqrt(F80))      kWh/ton, sizes in micrometres (Wi in kWh per SHORT ton in Bond's
                                                             tables; the caller states which ton)
  Kick            e = K_K ln(F/P)                             Rittinger   e = K_R (1/P - 1/F)
  Filtration      t = mu alpha c_s v^2 / (2 dP) + mu R_m v / dP,  v = filtrate volume per unit area formed in the time t
                  (Geankoplis section 14.2, examples 14.2-1 to 14.2-4; continuous drum: t = f t_cycle)
  Sigma theory    Q = 2 v_g Sigma (50 % cut size), v_g = d^2 (rho_p - rho) g / (18 mu) (Stokes)
                  cylindrical bowl   Sigma = omega^2 V / (2 g ln[2 r2/(r1 + r2)])        (Geankoplis example 14.4-2)
                  disc stack         Sigma = 2 pi n omega^2 (r2^3 - r1^3) / (3 g tan(theta))   (Ambler; derived for plug flow
                                     between discs, theta = half angle of the cone measured from the axis)
  Leaching        ideal stages (solution retained by the solids has the composition of the overflow), total and solute
                  balances stage by stage from the feed end (Geankoplis example 12.10-1)

Costs are PURCHASE costs with BioSTEAM's own correlations (most of them Seider et al. Table 16.32 scaled by BioSTEAM to
CE 567): the coefficients, bounds and F_BM are read from the installed BioSTEAM classes at run time (lazy import; no unit is
simulated, so no flowsheet, thermo package or price of BioSTEAM is touched) and scaled with CEPCI to the caller's year. The
two formulas BioSTEAM hard-codes inside methods (solid-bowl centrifuge, rotary vacuum filter) and its two conveyor power
formulas are transcribed here and checked against the installed source text on every run (`evidence`).

Nothing of the user's material is assumed: work index, cake resistance, cake moisture, recovery, oil content, residual oil
and underflow retention have no defaults. `published_values()` returns small tables of PUBLISHED typical values (source,
URL, date read) that a caller may pass explicitly; the engine never applies them."""
from __future__ import annotations

import math
import warnings as _pywarnings
from contextlib import contextmanager
from functools import lru_cache

G = 9.80665
LB = 0.45359237                 # kg
FT = 0.3048                     # m
FT2 = FT * FT
FT3 = FT ** 3
SHORT_TON = 2000.0 * LB         # kg
HP = 745.7                      # W, the value BioSTEAM uses (0.7457 kW/hp)
KWH = 3.6e6                     # J
DATE_READ = "2026-11-16"
MAX_STAGES = 200                # numerical guard of the stage-to-stage march
BALANCE_TOL = 1e-9              # relative closure demanded of every balance

TON_KG = {"short": SHORT_TON, "metric": 1000.0}

SIZE_REDUCTION_EQUIPMENT = {"hammer_mill": "molino de martillos", "shredder": "desfibradora (shredder)",
                            "cane_crushing_mill": "tándem de molinos de caña (crushing mill)"}
SEPARATORS = ("screw_press", "decanter_centrifuge", "disc_stack_centrifuge", "rotary_vacuum_drum_filter",
              "plate_and_frame_filter")

# --------------------------------------------------------------------------------------------------------------------------
# Published typical values (never applied by the engine; the caller passes them explicitly). All read on DATE_READ.
# --------------------------------------------------------------------------------------------------------------------------
_URL_911 = "https://www.911metallurgist.com/blog/table-of-bond-work-index-by-minerals"
_URL_CASTOR = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5015816/"
_URL_TFZ = ("https://www.tfz.bayern.de/mam/cms08/en/dateien/"
            "vegetableoilproductionindecentralisedplants_agenergyathen1999_wi.pdf")
_URL_CIGR = "https://cigrjournal.org/index.php/Ejounral/article/download/8503/4329/43641"
_URL_AP42 = "https://www.epa.gov/sites/default/files/2020-10/documents/c9s11-1.pdf"
_URL_ANDERSON = "https://www.andersonintl.com/wp-content/uploads/2019/04/Oilseed-Processing-Basics.pdf"
_URL_KWS = "https://www.kwsmfg.com/engineering-guides/bucket-elevator/horsepower-calculation/"

WORK_INDEX_PUBLISHED = {
    "unit": "kWh por tonelada CORTA (907.18 kg); pasar ton='short'",
    "source": "911 Metallurgist, «Table of Bond Work Index by Minerals», columna «Source 3, kWh/sh.t»",
    "url": _URL_911, "date_read": DATE_READ,
    "values": {"limestone": 12.7, "coal": 13.0, "hematite": 12.8, "quartz": 13.6, "gypsum_rock": 6.7, "clay": 6.3,
               "phosphate_rock": 9.9, "cement_clinker": 13.5, "bauxite": 8.8, "granite": 15.1, "dolomite": 11.3},
    "flag": ("promedios de minerales frágiles; la misma tabla da para caliza un intervalo de 4 a 36 en 177 ensayos: el "
             "índice del material real debe medirse. No hay índice de Bond publicado para semillas oleaginosas ni biomasa "
             "fibrosa: para ellas use law='specific_energy' con un ensayo o dato de proveedor"),
}
BUCKET_ELEVATOR_PUBLISHED = {
    "source": "KWS Manufacturing, «Bucket Elevator Horsepower Calculation» (guía de ingeniería)", "url": _URL_KWS,
    "date_read": DATE_READ,
    "friction_factor_range": [1.10, 1.30],
    "length_equivalency_factor_range": [5.0, 15.0],
    "flag": "intervalos de un fabricante; el factor depende del elevador concreto",
}
OILSEED_PUBLISHED = {
    "date_read": DATE_READ,
    "castor": {
        "seed_oil_mass_fraction_range": [0.30, 0.50],
        "pressing_oil_recovery_fraction": {"low_temperature_screw_press": 0.45, "hot_hydraulic_pressing_up_to": 0.80},
        "residual_oil_in_cake_mass_fraction_up_to": 0.10,
        "source": "Patel et al. (2016), «Castor Oil: Properties, Uses, and Optimization of Processing Parameters in "
                  "Commercial Production», Lipid Insights 9:1-12", "url": _URL_CASTOR},
    "rapeseed_canola": {
        "seed_oil_mass_fraction": 0.447,
        "pressing_oil_recovery_fraction_range": [0.667, 0.873], "pressing_oil_recovery_fraction_mean": 0.776,
        "pressing_specific_energy_J_kg_range": [0.08e6, 0.33e6], "pressing_specific_energy_J_kg_mean": 0.16e6,
        "prepress_oil_recovery_fraction": 0.50, "prepress_plus_solvent_oil_recovery_fraction": 0.98,
        "source": "Widmann (TFZ Bayern, 1999), «Vegetable oil production in decentralised plants…»: 20 plantas de "
                  "prensado en frío de colza medidas en operación (energía sin filtrado del aceite)", "url": _URL_TFZ},
    "jatropha": {
        "seed_oil_mass_fraction": 0.4627, "hull_mass_fraction": 0.42,
        "pressing_oil_recovery_fraction_range": [0.30, 0.687],
        "source": "Yaduvanshi et al. (2024), CIGR Journal 26(4):151: expeller de laboratorio (8-15 kg/h), mezclas "
                  "almendra-cáscara con 10-40 % de cáscara y 10-20 % de humedad. El resumen no aclara si el 46.27 % de "
                  "aceite es de la semilla entera o de la almendra", "url": _URL_CIGR},
    "soybean": {
        "solvent_extraction_residual_oil_in_meal_mass_fraction_below": 0.01,
        "solvent_in_extracted_flakes_mass_fraction_range": [0.35, 0.40],
        "hexane_loss_kg_per_kg_seed_mean": 2.45e-3,
        "source": "EPA AP-42 sección 9.11.1 «Vegetable Oil Processing» (11/95): hojuelas con 35-40 % de disolvente al "
                  "salir del extractor; pérdida media total de hexano 3.3 L/Mg = 4.9 lb/ton de soya (factor de calidad "
                  "D). Aceite residual < 1 %: Anderson International, «Oilseed Processing Basics» (fabricante)",
        "url": _URL_AP42, "url_residual_oil": _URL_ANDERSON},
    "generic_expeller": {
        "residual_oil_in_cake_mass_fraction_range": [0.05, 0.08],
        "source": "Anderson International, «Oilseed Processing Basics» (dato de fabricante)", "url": _URL_ANDERSON},
    "flag": ("valores típicos publicados; el contenido de aceite, la recuperación y el aceite residual del material y de la "
             "prensa reales son datos del usuario"),
}


def published_values() -> dict:
    """Published typical values with their sources (never applied silently): Bond work indices, bucket-elevator friction
    factors, oilseed pressing/extraction data, and the reference duties BioSTEAM carries in its unit classes."""
    out = {"bond_work_index": WORK_INDEX_PUBLISHED, "bucket_elevator": BUCKET_ELEVATOR_PUBLISHED,
           "oilseeds": OILSEED_PUBLISHED}
    b = _biosteam_data()
    out["biosteam_reference_duties"] = {
        "source": f"clases de unidad de BioSTEAM {b['version']} (leídas de la instalación)", "date_read": DATE_READ,
        "hammer_mill_specific_energy_J_kg": _item_energy(b["items"]["hammer_mill"]),
        "shredder_specific_energy_J_kg": _item_energy(b["items"]["shredder"]),
        "cane_crushing_mill_specific_energy_J_kg": _item_energy(b["items"]["cane_crushing_mill"]),
        "screw_press_specific_energy_J_kg": b["screw_press_kWh_per_t"] * KWH / 1000.0,
        "solid_bowl_centrifuge_energy_J_m3_feed": b["solids_centrifuge"]["kWh_per_m3"] * KWH,
        "disc_stack_centrifuge_energy_J_m3_feed": b["items"]["disc_stack_centrifuge"]["kW"] * KWH,
        "flag": ("valores por defecto de BioSTEAM para un servicio concreto (molino de martillos: molienda seca de maíz; "
                 "shredder y molinos: caña de azúcar; prensa de tornillo: Perry 18-126 según BioSTEAM): orientativos"),
    }
    return out


# --------------------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------------------
def _req(name: str, value, purpose: str) -> float:
    if value is None:
        raise ValueError(f"falta {name}: {purpose}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un número ({purpose})") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} debe ser un número finito ({purpose})")
    return v


def _pos(name: str, value, purpose: str) -> float:
    v = _req(name, value, purpose)
    if v <= 0.0:
        raise ValueError(f"{name} debe ser > 0 ({purpose})")
    return v


def _nonneg(name: str, value, purpose: str) -> float:
    v = _req(name, value, purpose)
    if v < 0.0:
        raise ValueError(f"{name} debe ser ≥ 0 ({purpose})")
    return v


def _frac(name: str, value, purpose: str, open_low: bool = False, open_high: bool = False) -> float:
    v = _req(name, value, purpose)
    lo_ok = v > 0.0 if open_low else v >= 0.0
    hi_ok = v < 1.0 if open_high else v <= 1.0
    if not (lo_ok and hi_ok):
        raise ValueError(f"{name} debe ser una fracción entre 0 y 1 ({purpose})")
    return v


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _evidence(checks: dict, reasons: list[str]) -> dict:
    return {"verified": not reasons, **checks, "reasons": reasons}


# --------------------------------------------------------------------------------------------------------------------------
# BioSTEAM: correlations read from the installed classes, global state left as found
# --------------------------------------------------------------------------------------------------------------------------
def _snapshot(bst) -> dict:
    snap = {"CE": bst.CE, "power_price": bst.PowerUtility.price, "thermo": getattr(bst.settings, "_thermo", None)}
    try:
        snap["flowsheet"] = bst.main_flowsheet.get_flowsheet()
    except Exception:  # noqa: BLE001 - no flowsheet registered yet
        snap["flowsheet"] = None
    agents = list(bst.HeatUtility.heating_agents) + list(bst.HeatUtility.cooling_agents)
    snap["agents"] = [(a, a.regeneration_price, a.heat_transfer_price) for a in agents]
    return snap


def _restore(bst, snap: dict) -> None:
    if bst.CE != snap["CE"]:
        bst.CE = snap["CE"]
    if bst.PowerUtility.price != snap["power_price"]:
        bst.PowerUtility.price = snap["power_price"]
    if getattr(bst.settings, "_thermo", None) is not snap["thermo"]:
        bst.settings._thermo = snap["thermo"]
    try:
        current = bst.main_flowsheet.get_flowsheet()
    except Exception:  # noqa: BLE001
        current = None
    if snap["flowsheet"] is not None and current is not snap["flowsheet"]:
        bst.main_flowsheet.set_flowsheet(snap["flowsheet"])
    for a, reg, htp in snap["agents"]:
        if a.regeneration_price != reg:
            a.regeneration_price = reg
        if a.heat_transfer_price != htp:
            a.heat_transfer_price = htp


@contextmanager
def _biosteam_guard():
    """thermosteam flips thermo's global RAISE_PROPERTY_CALCULATION_ERROR flags when it is imported or builds a chemical,
    which breaks other engines of this repo. Everything BioSTEAM-related runs inside this guard: the two flags, bst.CE,
    the thermo package, the main flowsheet and the utility prices are restored on exit, and Python warnings are captured."""
    import sys

    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    flags = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    bst = sys.modules.get("biosteam")
    snap = _snapshot(bst) if bst is not None and hasattr(bst, "main_flowsheet") else None
    try:
        with _pywarnings.catch_warnings(record=True) as caught:
            _pywarnings.simplefilter("always")
            yield caught
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flags
        bst = sys.modules.get("biosteam")
        if bst is not None and snap is not None:
            _restore(bst, snap)


def _squash(text: str) -> str:
    return "".join(text.split())


@lru_cache(maxsize=1)
def _biosteam_data() -> dict:
    """Coefficients, bounds, base index and F_BM of every correlation used here, read once from BioSTEAM's classes."""
    with _biosteam_guard():
        import inspect

        import biosteam as bst
        U = bst.units

        def item(cls, key: str) -> dict:
            x = cls.cost_items[key]
            return {"class": f"biosteam.units.{cls.__name__}", "key": key, "basis": x._basis, "units": x._units,
                    "S": float(x.S), "lb": x.lb, "ub": x.ub, "CE": float(x.CE), "cost": x.cost, "n": x.n,
                    "kW": float(x.kW or 0.0), "f": x.f, "F_BM": float(cls._F_BM_default[key])}

        def has(method, *snippets: str) -> bool:
            src = _squash(inspect.getsource(method))
            return all(_squash(s) in src for s in snippets)

        data = {
            "version": bst.__version__, "CE_default": float(bst.CE),
            "items": {
                "hammer_mill": item(U.HammerMill, "Hammer mill"),
                "shredder": item(U.Shredder, "Shredder"),
                "cane_crushing_mill": item(U.CrushingMill, "Crushing mill"),
                "belt": item(U.ConveyingBelt, "Conveying belt"),
                "screw": item(U.ScrewFeeder, "Screw feeder"),
                "screw_press": item(U.ScrewPress, "Screw press"),
                "plate_and_frame_filter": item(U.PressureFilter, "Pressure filter"),
                "disc_stack_centrifuge": item(U.LiquidsCentrifuge, "Liquids centrifuge"),
            },
            "screw_press_kWh_per_t": float(U.ScrewPress.kWh_per_bmt),
            "solids_centrifuge": {
                "range_short_ton_h": tuple(float(v) for v in U.SolidsCentrifuge.solids_loading_range["scroll_solid_bowl"]),
                "kWh_per_m3": float(U.SolidsCentrifuge.kWhr_per_m3), "coefficient": 68040.0, "exponent": 0.5, "CE": 567.0,
                "F_BM": 2.03,
                "source_ok": has(U.SolidsCentrifuge._design, "68040*(ts**0.5)", "cost *= bst.CE / 567",
                                 "self.F_BM['Centrifuges'] = 2.03", "ts *= 0.0011023")},
            "rotary_vacuum_filter": {
                "range_ft2": tuple(float(v) for v in U.RotaryVacuumFilter._bounds["Individual area"]),
                "coefficients": (11.796, -0.1905, 0.0554), "CE": 567.0,
                "F_BM": float(U.RotaryVacuumFilter._F_BM_default["Vessels"]),
                "source_ok": has(U.RotaryVacuumFilter._cost, "np.exp(11.796 - 0.1905 * logArea + 0.0554 * logArea**2)",
                                 "Cost * bst.CE/567")},
            "belt": {"minimum_flow_ft3_h": float(U.ConveyingBelt._minimum_flow),
                     "default_length_ft": float(U.ConveyingBelt.length), "default_height_ft": float(U.ConveyingBelt.height),
                     "source_ok": has(U.ConveyingBelt._design,
                                      "0.7457 * (0.00058 * F_mass**0.82 * self.length + self.height*0.00182*F_mass)",
                                      "feed.F_mass * 0.0006124")},
            "screw": {"default_length_ft": float(U.ScrewFeeder.length),
                      "source_ok": has(U.ScrewFeeder._design, "0.0146 * F_mass**0.85 * self.length * 0.7457",
                                       "0.0006124 * feed.F_mass")},
        }
    return data


def _item_energy(it: dict) -> float:
    """Reference duty of a BioSTEAM cost item as J per kg of its basis flow (kW per unit of basis flow)."""
    per_kg_h = {"ton/hr": SHORT_TON, "kg/hr": 1.0}[it["units"]]
    return it["kW"] * 1000.0 * 3600.0 / (it["S"] * per_kg_h)


def _cepci(year: int, cepci: float | None) -> tuple[float, str]:
    if cepci is not None:
        c = _pos("cepci", cepci, "índice CEPCI del año de la estimación")
        return c, f"CEPCI {c:g} (dado por el usuario)"
    from packages.costing.sslw import CEPCI_IDAES
    if year not in CEPCI_IDAES:
        raise ValueError(f"la tabla CEPCI de IDAES cubre 2015-2023; para {year} pase cepci=<valor del índice>")
    return CEPCI_IDAES[year], f"CEPCI {CEPCI_IDAES[year]} ({year}, tabla de IDAES)"


def _decorated_cost(it: dict, S: float) -> dict:
    """BioSTEAM's `@cost` algorithm (biosteam/units/decorators/_cost.py): below `lb` the cost is evaluated AT `lb`; above
    `ub` the flow is split in N = ceil(S/ub) equal units."""
    n_units, clamped, S_eval = 1, False, S
    if it["lb"] is not None and S < it["lb"]:
        S_eval, clamped = it["lb"], True
        F = S_eval / it["S"]
    elif it["ub"] is not None:
        n_units = max(1, math.ceil(S / it["ub"]))
        F = S / it["S"] / n_units
    else:
        F = S / it["S"]
    per_unit = it["f"](F) if it["f"] else it["cost"] * F ** it["n"]
    return {"per_unit_at_base": float(per_unit), "n_units": n_units, "clamped_to_lower_bound": clamped,
            "size_per_unit": F * it["S"], "reference_power_W": it["kW"] * 1000.0 * (S_eval / it["S"])}


def _capital_item(name: str, per_unit_at_base: float, n_units: int, base_CE: float, F_BM: float, F_BM_source: str,
                  year: int, cepci: float | None, correlation: str, size_text: str) -> dict:
    index, index_basis = _cepci(year, cepci)
    per_unit = per_unit_at_base * index / base_CE
    total = per_unit * n_units
    return {"name": name, "F_BM": F_BM, "F_BM_source": F_BM_source, "purchase_cost_usd": total,
            "base_purchase_cost_usd": total, "purchase_cost_per_unit_usd": per_unit, "number_of_units": n_units,
            "cost_year": year if cepci is None else None, "cepci": index, "correlation_cepci": base_CE,
            "basis": f"{correlation}; {size_text}; costo de compra (no instalado), {index_basis}"}


def _range_warnings(it: dict, S: float, dc: dict, what: str) -> list[str]:
    w = []
    if dc["clamped_to_lower_bound"]:
        w.append(f"{what}: {S:.4g} {it['units']} está por debajo del mínimo de la correlación ({it['lb']:g} {it['units']}); "
                 "BioSTEAM evalúa el costo EN el mínimo (se hizo igual): el costo real de un equipo tan pequeño es incierto")
    if dc["n_units"] > 1:
        w.append(f"{what}: {S:.4g} {it['units']} supera el máximo de la correlación ({it['ub']:g} {it['units']}); se "
                 f"costean {dc['n_units']} unidades iguales en paralelo, como hace BioSTEAM")
    if it["lb"] is None and it["ub"] is None:
        ratio = S / it["S"]
        if not 0.1 <= ratio <= 10.0:
            w.append(f"{what}: BioSTEAM no publica intervalo de validez para esta correlación (un solo punto base de "
                     f"{it['S']:g} {it['units']} con exponente {it['n']:g}); el tamaño pedido es {ratio:.3g} veces la base: "
                     "extrapolación fuerte")
    return w


def _source_reason(ok: bool, what: str) -> list[str]:
    return [] if ok else [f"el código fuente de BioSTEAM instalado ya no contiene la fórmula transcrita de {what}: revisar "
                          "la transcripción antes de usar este resultado"]


# --------------------------------------------------------------------------------------------------------------------------
# 1. Size reduction
# --------------------------------------------------------------------------------------------------------------------------
def size_reduction(mass_flow_kg_s: float, law: str, feed_size_m: float | None = None, product_size_m: float | None = None,
                   work_index_kWh_per_ton: float | None = None, ton: str | None = None,
                   kick_constant_J_kg: float | None = None, rittinger_constant_J_m_kg: float | None = None,
                   reference_test: dict | None = None, specific_energy_J_kg: float | None = None,
                   equipment: str | None = None, year: int = 2023, cepci: float | None = None) -> dict:
    """Specific energy and power of a size reduction (Bond, Kick, Rittinger or the user's own specific energy) and, if
    `equipment` is given, its purchase cost with BioSTEAM's correlation (module docstring)."""
    m = _pos("mass_flow_kg_s", mass_flow_kg_s, "flujo másico de sólido que se muele")
    if law not in ("bond", "kick", "rittinger", "specific_energy"):
        raise ValueError("law debe ser 'bond', 'kick', 'rittinger' o 'specific_energy'")
    warnings: list[str] = []
    assumptions: list[str] = []
    reasons: list[str] = []
    checks: dict = {}
    F = P = None
    if law != "specific_energy" or feed_size_m is not None or product_size_m is not None:
        F = _pos("feed_size_m", feed_size_m, "tamaño de la alimentación (para Bond: malla por la que pasa el 80 %)")
        P = _pos("product_size_m", product_size_m, "tamaño del producto (para Bond: malla por la que pasa el 80 %)")
        if P >= F:
            raise ValueError("product_size_m debe ser menor que feed_size_m: una reducción de tamaño hace más fino el sólido")
    constant = None
    if law == "bond":
        Wi = _pos("work_index_kWh_per_ton", work_index_kWh_per_ton,
                  "índice de trabajo de Bond del material (dato del usuario; published_values() lista valores publicados)")
        if ton not in TON_KG:
            raise ValueError("falta ton ('short' o 'metric'): tonelada a la que se refiere el índice de trabajo (las tablas "
                             "de Bond están en kWh por tonelada corta de 907.18 kg)")
        e_kWh_ton = 10.0 * Wi * (1.0 / math.sqrt(P * 1e6) - 1.0 / math.sqrt(F * 1e6))
        e = e_kWh_ton * KWH / TON_KG[ton]
        # same law as written in McCabe's example 28.1 (sizes in mm, kW per ton/h): an independent arithmetic route
        e_mccabe = 0.3162 * Wi * (1.0 / math.sqrt(P * 1e3) - 1.0 / math.sqrt(F * 1e3)) * KWH / TON_KG[ton]
        checks["bond_two_forms_rel_diff"] = _rel(e, e_mccabe)
        if checks["bond_two_forms_rel_diff"] > 2e-4:
            reasons.append("las dos formas de la ley de Bond no coinciden")
        assumptions += [
            "ley de Bond: e = 10·Wi·(1/√P80 − 1/√F80), tamaños en µm al 80 % pasante; Wi es la energía para llevar el material "
            "desde tamaño infinito hasta 80 % pasante 100 µm",
            "potencia bruta de molienda en húmedo en circuito cerrado (definición del índice); no se aplican los factores de "
            "corrección de Bond (molienda en seco, circuito abierto, sobretamaño de alimentación, finura)"]
        if P < 50e-6:
            warnings.append("producto más fino que ~50 µm: la ley de Bond subestima la energía de molienda ultrafina")
    elif law in ("kick", "rittinger"):
        name = "kick_constant_J_kg" if law == "kick" else "rittinger_constant_J_m_kg"
        given = kick_constant_J_kg if law == "kick" else rittinger_constant_J_m_kg
        shape = (lambda f, p: math.log(f / p)) if law == "kick" else (lambda f, p: 1.0 / p - 1.0 / f)
        if (given is None) == (reference_test is None):
            raise ValueError(f"para law='{law}' dé {name} (constante del material) O reference_test {{feed_size_m, "
                             "product_size_m, specific_energy_J_kg}} (un ensayo propio con el que se calibra), uno solo")
        if given is not None:
            constant = _pos(name, given, "constante de la ley para este material y este equipo")
        else:
            f0 = _pos("reference_test.feed_size_m", reference_test.get("feed_size_m"), "tamaño de alimentación del ensayo")
            p0 = _pos("reference_test.product_size_m", reference_test.get("product_size_m"), "tamaño de producto del ensayo")
            e0 = _pos("reference_test.specific_energy_J_kg", reference_test.get("specific_energy_J_kg"),
                      "energía específica medida en el ensayo")
            if p0 >= f0:
                raise ValueError("reference_test: product_size_m debe ser menor que feed_size_m")
            constant = e0 / shape(f0, p0)
            checks["reference_test_reproduced_rel_diff"] = _rel(constant * shape(f0, p0), e0)
            assumptions.append("constante calibrada con el ensayo del usuario; vale para el mismo material y tipo de equipo")
        e = constant * shape(F, P)
        assumptions.append("ley de Kick: e = K·ln(F/P) (energía igual para igual razón de reducción; trituración gruesa)"
                           if law == "kick" else
                           "ley de Rittinger: e = K·(1/P − 1/F) (energía proporcional a la superficie nueva; molienda fina)")
    else:
        e = _pos("specific_energy_J_kg", specific_energy_J_kg, "energía específica de molienda medida o de proveedor")
        assumptions.append("energía específica dada por el usuario (ensayo o proveedor); no se aplica ninguna ley de molienda")
    power = e * m
    if not (math.isfinite(power) and power > 0.0):
        reasons.append("la potencia calculada no es finita y positiva")
    out = {"law": law, "specific_energy_J_kg": e, "specific_energy_kWh_per_tonne": e / KWH * 1000.0, "power_W": power,
           "mass_flow_kg_s": m, "feed_size_m": F, "product_size_m": P,
           "reduction_ratio": (F / P) if F is not None else None, "law_constant": constant,
           "capital_items": [], "cost": None}
    not_included = ["eficiencia del motor y de la transmisión (la potencia es la que consume la molienda en el eje)",
                    "distribución de tamaños del producto, clasificación y carga circulante",
                    "desgaste de martillos/forros y consumo de medios de molienda"]
    if equipment is not None:
        if equipment == "crusher":
            not_included.append("costo de trituradoras de quijadas/giratorias/de cono: BioSTEAM no trae correlación y la "
                                "tabla 16.32 de Seider no pudo leerse en fuente abierta; pedir cotización")
        elif equipment not in SIZE_REDUCTION_EQUIPMENT:
            raise ValueError(f"equipment debe ser uno de {sorted(SIZE_REDUCTION_EQUIPMENT)} o 'crusher' (sin costo)")
        else:
            b = _biosteam_data()
            it = b["items"][equipment]
            label = SIZE_REDUCTION_EQUIPMENT[equipment]
            S = m * 3600.0 / (SHORT_TON if it["units"] == "ton/hr" else 1.0)
            dc = _decorated_cost(it, S)
            warnings += _range_warnings(it, S, dc, label)
            rng = (f"válida de {it['lb']:g} a {it['ub']:g} {it['units']} (ton = tonelada corta)" if it["lb"] is not None
                   else f"un solo punto base de {it['S']:g} {it['units']}, sin intervalo publicado")
            ci = _capital_item(label, dc["per_unit_at_base"], dc["n_units"], it["CE"], it["F_BM"],
                               f"F_BM de BioSTEAM ({it['class']})", year, cepci,
                               f"correlación de {it['class']} (BioSTEAM {b['version']}), {rng}",
                               f"{dc['size_per_unit']:.4g} {it['units']} por unidad")
            out["capital_items"] = [ci]
            out["cost"] = {"purchase_cost_usd": ci["purchase_cost_usd"], "number_of_units": ci["number_of_units"],
                           "F_BM": ci["F_BM"], "cepci": ci["cepci"], "cost_year": ci["cost_year"],
                           "correlation_cepci": it["CE"], "biosteam_default_cepci": b["CE_default"]}
            out["biosteam_reference_power_W"] = dc["reference_power_W"]
            assumptions.append(f"biosteam_reference_power_W es la potencia que BioSTEAM asigna a este equipo para su "
                               f"servicio de referencia ({it['kW']:g} kW por cada {it['S']:g} {it['units']}); se da solo para "
                               "comparar con power_W, que es el resultado de la ley pedida")
            if equipment != "hammer_mill":
                warnings.append(f"{label}: la correlación de BioSTEAM es de la industria de la caña de azúcar; para otro "
                                "material es solo un orden de magnitud")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method="leyes de molienda de Bond/Kick/Rittinger en forma cerrada; costo con la correlación de BioSTEAM leída "
                      "de la clase instalada y escalada con CEPCI",
               evidence=_evidence(checks, reasons), published_reference=WORK_INDEX_PUBLISHED)
    return out


# --------------------------------------------------------------------------------------------------------------------------
# 2. Mechanical conveying
# --------------------------------------------------------------------------------------------------------------------------
def solids_conveying(conveyor: str, mass_flow_kg_s: float, length_m: float | None = None, lift_m: float | None = None,
                     bulk_density_kg_m3: float | None = None, friction_factor: float | None = None,
                     tail_pulley_diameter_m: float | None = None, length_equivalency_factor: float | None = None,
                     year: int = 2023, cepci: float | None = None) -> dict:
    """Shaft power of a belt conveyor, screw conveyor or bucket elevator and purchase cost of belt and screw (BioSTEAM)."""
    if conveyor not in ("belt", "screw", "bucket_elevator"):
        raise ValueError("conveyor debe ser 'belt', 'screw' o 'bucket_elevator'")
    m = _pos("mass_flow_kg_s", mass_flow_kg_s, "flujo másico de sólido transportado")
    H = _nonneg("lift_m", lift_m, "elevación vertical entre carga y descarga (0 si es horizontal)")
    m_lb_s = m / LB
    lift_exact = m * G * H
    warnings: list[str] = []
    reasons: list[str] = []
    checks: dict = {}
    assumptions = ["potencia en el eje del transportador; no incluye la eficiencia del motor ni del reductor"]
    not_included = ["arranque con carga, sobrecargas y factor de servicio del accionamiento",
                    "estructura, tolvas, chutes, cubiertas y captación de polvo"]
    out: dict = {"conveyor": conveyor, "mass_flow_kg_s": m, "lift_m": H, "capital_items": [], "cost": None}
    b = None
    if conveyor in ("belt", "screw"):
        L = _pos("length_m", length_m, "longitud del transportador")
        if H > L:
            raise ValueError("lift_m no puede ser mayor que length_m (longitud del transportador)")
        b = _biosteam_data()
        L_ft, H_ft = L / FT, H / FT
        if conveyor == "belt":
            # biosteam.units.ConveyingBelt._design: 0.7457*(0.00058*F_mass**0.82*length + height*0.00182*F_mass) kW
            friction = HP * 0.00058 * m_lb_s ** 0.82 * L_ft
            lift = HP * 0.00182 * m_lb_s * H_ft
            reasons += _source_reason(b["belt"]["source_ok"], "ConveyingBelt._design")
            assumptions.append("potencia de la correlación de Seider tal como la implementa BioSTEAM (ConveyingBelt): "
                               "P[hp] = 0.00058·ṁ^0.82·L + 0.00182·ṁ·H con ṁ en lb/s y L, H en ft")
        else:
            # biosteam.units.ScrewFeeder._design: 0.0146*F_mass**0.85*length*0.7457 kW (no lift term)
            friction = HP * 0.0146 * m_lb_s ** 0.85 * L_ft
            lift = lift_exact
            reasons += _source_reason(b["screw"]["source_ok"], "ScrewFeeder._design")
            assumptions.append("potencia de fricción de la correlación de Seider tal como la implementa BioSTEAM "
                               "(ScrewFeeder): P[hp] = 0.0146·ṁ^0.85·L con ṁ en lb/s y L en ft")
            if H > 0.0:
                assumptions.append("BioSTEAM no trae término de elevación para el tornillo: se suma la potencia exacta de "
                                   "elevación ṁ·g·H (añadido de este módulo)")
                warnings.append("tornillo inclinado: la capacidad cae con la inclinación y la potencia real sube más que "
                                "ṁ·g·H; verificar con el fabricante")
        out["length_m"] = L
        checks["lift_term_vs_m_g_H_rel_diff"] = _rel(lift, lift_exact) if H > 0.0 else 0.0
        if checks["lift_term_vs_m_g_H_rel_diff"] > 5e-3:
            reasons.append("el término de elevación no coincide con ṁ·g·H")
    else:
        if H <= 0.0:
            raise ValueError("lift_m debe ser > 0 para un elevador de cangilones (altura de descarga)")
        if friction_factor is not None and (tail_pulley_diameter_m is not None or length_equivalency_factor is not None):
            raise ValueError("dé friction_factor O (tail_pulley_diameter_m y length_equivalency_factor), no ambos")
        if friction_factor is not None:
            Ff = _req("friction_factor", friction_factor, "factor multiplicador de fricción del elevador (≥ 1)")
            if Ff < 1.0:
                raise ValueError("friction_factor debe ser ≥ 1 (1.0 = solo elevar el producto)")
            lift, friction = lift_exact, lift_exact * (Ff - 1.0)
            assumptions.append("método del factor de fricción (KWS, ec. 6): P = ṁ·g·H·F")
        elif tail_pulley_diameter_m is not None or length_equivalency_factor is not None:
            d = _pos("tail_pulley_diameter_m", tail_pulley_diameter_m, "diámetro de la polea de cola (método de longitud "
                     "equivalente)")
            Leq = _pos("length_equivalency_factor", length_equivalency_factor, "factor de longitud equivalente Leq")
            lift, friction = lift_exact, m * G * d * Leq
            assumptions.append("método de la longitud equivalente (KWS, ec. 5): P = ṁ·g·(H + d·Leq)")
        else:
            raise ValueError("falta friction_factor (o tail_pulley_diameter_m con length_equivalency_factor): la fricción "
                             "del elevador (excavado de los cangilones, deslizamiento, rodamientos) es un dato del equipo; "
                             "KWS publica 1.10-1.30 para el factor y 5-15 para Leq (published_values())")
        not_included.append("costo de compra del elevador de cangilones: BioSTEAM no trae correlación y la tabla 16.32 de "
                            "Seider no pudo leerse en fuente abierta; pedir cotización")
        out["published_reference"] = BUCKET_ELEVATOR_PUBLISHED
    power = friction + lift
    out.update(power_W=power, friction_power_W=friction, lift_power_W=lift, lift_power_exact_W=lift_exact)
    if not (math.isfinite(power) and power > 0.0):
        reasons.append("la potencia calculada no es finita y positiva")
    if conveyor in ("belt", "screw"):
        if bulk_density_kg_m3 is None:
            not_included.append("costo de compra: falta bulk_density_kg_m3 (densidad aparente del sólido a granel), porque la "
                                "correlación de BioSTEAM se basa en el flujo volumétrico")
        else:
            rho_b = _pos("bulk_density_kg_m3", bulk_density_kg_m3, "densidad aparente del sólido a granel")
            q = m / rho_b
            S = q * 3600.0 / FT3
            it = b["items"][conveyor]
            dc = _decorated_cost(it, S)
            label = "transportador de banda" if conveyor == "belt" else "transportador/alimentador de tornillo"
            warnings += _range_warnings(it, S, dc, label)
            if conveyor == "belt":
                if S < b["belt"]["minimum_flow_ft3_h"]:
                    warnings.append(f"{label}: {S:.4g} ft³/h está por debajo del mínimo de diseño de BioSTEAM "
                                    f"({b['belt']['minimum_flow_ft3_h']:g} ft³/h)")
                rng = f"flujo volumétrico hasta {it['ub']:g} ft³/h por banda (mínimo {b['belt']['minimum_flow_ft3_h']:g})"
                ref_L = b["belt"]["default_length_ft"] * FT
                fbm_src = f"F_BM de BioSTEAM ({it['class']})"
            else:
                rng = f"válida de {it['lb']:g} a {it['ub']:g} ft³/h"
                ref_L = b["screw"]["default_length_ft"] * FT
                fbm_src = "BioSTEAM no define F_BM para ScrewFeeder (usa 1.0 por omisión): dé su propio factor de instalación"
                warnings.append("tornillo: BioSTEAM no trae factor de módulo desnudo (F_BM = 1.0): el costo instalado queda "
                                "subestimado si no se da un F_BM propio")
            warnings.append(f"{label}: la correlación de costo de BioSTEAM depende solo del flujo volumétrico, NO de la "
                            f"longitud (BioSTEAM supone {ref_L:.1f} m por omisión; aquí {L:.1f} m): orden de magnitud")
            ci = _capital_item(label, dc["per_unit_at_base"], dc["n_units"], it["CE"], it["F_BM"], fbm_src, year, cepci,
                               f"correlación de {it['class']} (BioSTEAM {b['version']}), {rng}",
                               f"{dc['size_per_unit']:.4g} ft³/h por unidad")
            out["capital_items"] = [ci]
            out["volumetric_flow_m3_s"] = q
            out["cost"] = {"purchase_cost_usd": ci["purchase_cost_usd"], "number_of_units": ci["number_of_units"],
                           "F_BM": ci["F_BM"], "cepci": ci["cepci"], "cost_year": ci["cost_year"],
                           "correlation_cepci": it["CE"], "biosteam_default_cepci": b["CE_default"]}
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method="correlaciones de potencia de Seider implementadas en BioSTEAM (banda, tornillo) y fórmula de potencia "
                      "de elevador de cangilones de KWS; costo con la correlación de BioSTEAM escalada con CEPCI",
               evidence=_evidence(checks, reasons))
    return out


# --------------------------------------------------------------------------------------------------------------------------
# 3. Solid-liquid separation
# --------------------------------------------------------------------------------------------------------------------------
def filtration_area(filtrate_m3_s: float, dry_cake_per_filtrate_kg_m3: float, specific_cake_resistance_m_kg: float,
                    medium_resistance_1_m: float, filtrate_viscosity_Pa_s: float, pressure_drop_Pa: float,
                    filtration_time_s: float, cycle_time_s: float) -> dict:
    """Area of a filter working at constant pressure: in every cycle of `cycle_time_s` the cloth filters during
    `filtration_time_s` (drum: submerged fraction x cycle) and must pass the filtrate of the whole cycle."""
    a = filtrate_viscosity_Pa_s * specific_cake_resistance_m_kg * dry_cake_per_filtrate_kg_m3 / (2.0 * pressure_drop_Pa)
    bb = filtrate_viscosity_Pa_s * medium_resistance_1_m / pressure_drop_Pa
    disc = bb * bb + 4.0 * a * filtration_time_s
    v = 2.0 * filtration_time_s / (bb + math.sqrt(disc))          # stable root of a v² + b v − t = 0
    area = filtrate_m3_s * cycle_time_s / v
    return {"area_m2": area, "filtrate_per_area_per_cycle_m3_m2": v,
            "cake_dry_mass_per_area_per_cycle_kg_m2": v * dry_cake_per_filtrate_kg_m3,
            "mean_filtrate_flux_m3_m2_s": v / cycle_time_s,
            "medium_share_of_resistance_at_end": bb / (bb + 2.0 * a * v),
            "equation_residual_rel": abs(a * v * v + bb * v - filtration_time_s) / filtration_time_s}


def filtration_time(filtrate_volume_m3: float, area_m2: float, dry_cake_per_filtrate_kg_m3: float,
                    specific_cake_resistance_m_kg: float, medium_resistance_1_m: float, filtrate_viscosity_Pa_s: float,
                    pressure_drop_Pa: float) -> float:
    """Time to collect a filtrate volume through a given area at constant pressure (t = Kp V²/2 + B V)."""
    Kp = (filtrate_viscosity_Pa_s * specific_cake_resistance_m_kg * dry_cake_per_filtrate_kg_m3
          / (area_m2 ** 2 * pressure_drop_Pa))
    B = filtrate_viscosity_Pa_s * medium_resistance_1_m / (area_m2 * pressure_drop_Pa)
    return Kp * filtrate_volume_m3 ** 2 / 2.0 + B * filtrate_volume_m3


def sigma_bowl(bowl: dict) -> dict:
    """Sigma (equivalent settling area, m²) of a cylindrical bowl (tubular / solid bowl) or of a disc stack."""
    kind = bowl.get("type")
    if kind not in ("tubular", "solid_bowl", "disc_stack"):
        raise ValueError("sigma.bowl.type debe ser 'tubular', 'solid_bowl' o 'disc_stack'")
    rpm = _pos("sigma.bowl.rpm", bowl.get("rpm"), "velocidad de giro del tazón")
    r1 = _pos("sigma.bowl.r_inner_m", bowl.get("r_inner_m"), "radio interior (superficie del líquido o borde interior del "
              "disco)")
    r2 = _pos("sigma.bowl.r_outer_m", bowl.get("r_outer_m"), "radio exterior (pared del tazón o borde exterior del disco)")
    if r1 >= r2:
        raise ValueError("sigma.bowl: r_inner_m debe ser menor que r_outer_m")
    w = 2.0 * math.pi * rpm / 60.0
    if kind == "disc_stack":
        n = _pos("sigma.bowl.n_discs", bowl.get("n_discs"), "número de espacios entre discos")
        th = _pos("sigma.bowl.half_angle_deg", bowl.get("half_angle_deg"), "semiángulo del cono de los discos medido desde "
                  "el eje de giro")
        if th >= 90.0:
            raise ValueError("sigma.bowl.half_angle_deg debe ser menor que 90°")
        sigma = 2.0 * math.pi * n * w * w * (r2 ** 3 - r1 ** 3) / (3.0 * G * math.tan(math.radians(th)))
        return {"type": kind, "sigma_m2": sigma, "omega_rad_s": w, "g_force_at_outer_radius": w * w * r2 / G,
                "formula": "Σ = 2π·n·ω²·(r2³ − r1³)/(3·g·tan θ)"}
    length = _pos("sigma.bowl.length_m", bowl.get("length_m"), "longitud de la zona cilíndrica de clarificación")
    V = math.pi * length * (r2 * r2 - r1 * r1)
    sigma = w * w * V / (2.0 * G * math.log(2.0 * r2 / (r1 + r2)))
    return {"type": kind, "sigma_m2": sigma, "omega_rad_s": w, "g_force_at_outer_radius": w * w * r2 / G,
            "liquid_volume_m3": V, "formula": "Σ = ω²·V/(2·g·ln[2·r2/(r1 + r2)])"}


def solid_liquid_separation(equipment: str, feed_kg_s: float, feed_solids_mass_fraction: float,
                            solids_recovery_fraction: float, cake_moisture_mass_fraction: float,
                            filtration: dict | None = None, sigma: dict | None = None,
                            feed_density_kg_m3: float | None = None, year: int = 2023,
                            cepci: float | None = None) -> dict:
    """Outlet split of a screw press, decanter, disc-stack clarifier, rotary vacuum drum filter or plate-and-frame filter from
    the user's recovery and cake moisture; filter area or Sigma; purchase cost with BioSTEAM's correlation."""
    if equipment not in SEPARATORS:
        raise ValueError(f"equipment debe ser uno de {list(SEPARATORS)}")
    F = _pos("feed_kg_s", feed_kg_s, "flujo másico de la suspensión alimentada")
    x = _frac("feed_solids_mass_fraction", feed_solids_mass_fraction, "fracción másica de sólidos insolubles en la "
              "alimentación", open_low=True, open_high=True)
    rec = _frac("solids_recovery_fraction", solids_recovery_fraction, "fracción de los sólidos alimentados que sale en la "
                "torta (dato de ensayo o de proveedor)", open_low=True)
    mc = _frac("cake_moisture_mass_fraction", cake_moisture_mass_fraction, "fracción másica de líquido en la torta húmeda "
               "(dato de ensayo o de proveedor)", open_high=True)
    is_filter = equipment in ("rotary_vacuum_drum_filter", "plate_and_frame_filter")
    is_centrifuge = equipment in ("decanter_centrifuge", "disc_stack_centrifuge")
    if filtration is not None and not is_filter:
        raise ValueError("filtration solo aplica a rotary_vacuum_drum_filter y plate_and_frame_filter")
    if sigma is not None and not is_centrifuge:
        raise ValueError("sigma solo aplica a decanter_centrifuge y disc_stack_centrifuge")
    warnings: list[str] = []
    reasons: list[str] = []
    checks: dict = {}
    assumptions = ["el reparto sale de la recuperación de sólidos y de la humedad de torta del USUARIO; el equipo no predice "
                   "su propio desempeño",
                   "el líquido de la torta tiene la misma composición que el líquido clarificado (sin lavado)"]
    not_included = ["lavado y soplado de la torta, ayudas de filtración y floculantes",
                    "compresibilidad de la torta (dé la resistencia específica a la presión de trabajo)"]
    solids_in, liquid_in = F * x, F * (1.0 - x)
    cake_solids = rec * solids_in
    cake_liquid = cake_solids * mc / (1.0 - mc)
    if cake_liquid > liquid_in * (1.0 + 1e-12):
        raise ValueError(f"la torta pedida retendría {cake_liquid:.4g} kg/s de líquido y la alimentación solo trae "
                         f"{liquid_in:.4g} kg/s: cake_moisture_mass_fraction es mayor que la humedad de la propia alimentación")
    cake = cake_solids + cake_liquid
    liq_solids = solids_in - cake_solids
    liq_liquid = liquid_in - cake_liquid
    liq = liq_solids + liq_liquid
    checks["mass_balance_rel"] = abs(F - cake - liq) / F
    checks["solids_balance_rel"] = abs(solids_in - cake_solids - liq_solids) / solids_in
    if max(checks["mass_balance_rel"], checks["solids_balance_rel"]) > BALANCE_TOL:
        reasons.append("el balance de masa no cierra")
    out: dict = {
        "equipment": equipment,
        "feed": {"total_kg_s": F, "solids_kg_s": solids_in, "liquid_kg_s": liquid_in},
        "cake": {"total_kg_s": cake, "solids_kg_s": cake_solids, "liquid_kg_s": cake_liquid,
                 "moisture_mass_fraction": mc, "wet_to_dry_mass_ratio": 1.0 / (1.0 - mc)},
        "liquid": {"total_kg_s": liq, "liquid_kg_s": liq_liquid, "solids_kg_s": liq_solids,
                   "solids_mass_fraction": liq_solids / liq if liq > 0.0 else None},
        "liquid_recovery_fraction": liq_liquid / liquid_in,
        "capital_items": [], "cost": None, "filtration": None, "sigma": None,
    }
    b = _biosteam_data()
    area_ft2 = None
    q_feed = None
    if feed_density_kg_m3 is not None:
        q_feed = F / _pos("feed_density_kg_m3", feed_density_kg_m3, "densidad de la suspensión alimentada")

    if is_filter and filtration is not None:
        fl = filtration
        alpha = _pos("filtration.specific_cake_resistance_m_kg", fl.get("specific_cake_resistance_m_kg"),
                     "resistencia específica de la torta α a la presión de trabajo (ensayo de filtración)")
        Rm = _nonneg("filtration.medium_resistance_1_m", fl.get("medium_resistance_1_m"),
                     "resistencia del medio filtrante Rm (0 si se desprecia, dicho explícitamente)")
        mu = _pos("filtration.filtrate_viscosity_Pa_s", fl.get("filtrate_viscosity_Pa_s"), "viscosidad del filtrado")
        dP = _pos("filtration.pressure_drop_Pa", fl.get("pressure_drop_Pa"), "caída de presión a través de torta y medio")
        rho = _pos("filtration.filtrate_density_kg_m3", fl.get("filtrate_density_kg_m3"), "densidad del filtrado")
        if liq_liquid <= 0.0:
            raise ValueError("no queda filtrado: toda el agua de la alimentación sale con la torta")
        q_filtrate = liq_liquid / rho
        cs = cake_solids / q_filtrate
        if equipment == "rotary_vacuum_drum_filter":
            tc = _pos("filtration.cycle_time_s", fl.get("cycle_time_s"), "tiempo de una vuelta del tambor")
            f_sub = _frac("filtration.submerged_fraction", fl.get("submerged_fraction"), "fracción del tambor sumergida en "
                          "la suspensión", open_low=True, open_high=True)
            t_f = f_sub * tc
            res = filtration_area(q_filtrate, cs, alpha, Rm, mu, dP, t_f, tc)
            res.update(cycle_time_s=tc, filtration_time_s=t_f, submerged_fraction=f_sub)
            assumptions.append("filtro continuo de tambor: cada punto de la tela filtra a presión constante durante f·t_ciclo "
                               "(Geankoplis sección 14.2, con resistencia del medio)")
        else:
            t_f = _pos("filtration.filtration_time_s", fl.get("filtration_time_s"), "tiempo de filtración de cada ciclo")
            down = _nonneg("filtration.downtime_s", fl.get("downtime_s"), "tiempo muerto por ciclo (lavado, descarga, "
                           "armado; 0 si no hay, dicho explícitamente)")
            tc = t_f + down
            res = filtration_area(q_filtrate, cs, alpha, Rm, mu, dP, t_f, tc)
            V_cycle = q_filtrate * tc
            t_back = filtration_time(V_cycle, res["area_m2"], cs, alpha, Rm, mu, dP)
            checks["filtration_time_recomputed_rel_diff"] = _rel(t_back, t_f)
            if checks["filtration_time_recomputed_rel_diff"] > 1e-8:
                reasons.append("el tiempo de filtración recalculado con el área no reproduce el pedido")
            res.update(cycle_time_s=tc, filtration_time_s=t_f, downtime_s=down, filtrate_per_cycle_m3=V_cycle,
                       cake_dry_mass_per_cycle_kg=cake_solids * tc)
            assumptions.append("filtro prensa por lotes a presión constante: t = Kp·V²/2 + B·V (Geankoplis ej. 14.2-2); el "
                               "área es la que filtra en filtration_time_s todo el filtrado de un ciclo")
        res.update(filtrate_m3_s=q_filtrate, dry_cake_per_filtrate_kg_m3=cs)
        checks["filtration_equation_residual_rel"] = res.pop("equation_residual_rel")
        if checks["filtration_equation_residual_rel"] > 1e-9:
            reasons.append("la ecuación de filtración no quedó satisfecha")
        out["filtration"] = res
        area_ft2 = res["area_m2"] / FT2
    elif is_filter:
        not_included.append("área de filtración: falta filtration {specific_cake_resistance_m_kg, medium_resistance_1_m, "
                            "filtrate_viscosity_Pa_s, pressure_drop_Pa, filtrate_density_kg_m3, y cycle_time_s + "
                            "submerged_fraction (tambor) o filtration_time_s + downtime_s (filtro prensa)}")

    if is_centrifuge and sigma is not None:
        sg = sigma
        d = _pos("sigma.particle_diameter_m", sg.get("particle_diameter_m"), "diámetro de la partícula que debe separarse "
                 "(tamaño de corte al 50 %)")
        rho_p = _pos("sigma.particle_density_kg_m3", sg.get("particle_density_kg_m3"), "densidad de la partícula")
        rho_l = _pos("sigma.liquid_density_kg_m3", sg.get("liquid_density_kg_m3"), "densidad del líquido")
        mu = _pos("sigma.liquid_viscosity_Pa_s", sg.get("liquid_viscosity_Pa_s"), "viscosidad del líquido")
        if rho_p <= rho_l:
            raise ValueError("sigma: la partícula debe ser más densa que el líquido para sedimentar hacia la pared")
        q = solids_in / rho_p + liquid_in / rho_l
        if q_feed is not None and _rel(q, q_feed) > 0.05:
            warnings.append("feed_density_kg_m3 difiere más de 5 % del volumen ideal sólido + líquido calculado con las "
                            "densidades de sigma; el caudal de Sigma usa el volumen ideal")
        q_feed = q
        vg = d * d * (rho_p - rho_l) * G / (18.0 * mu)
        res = {"feed_volumetric_flow_m3_s": q, "stokes_settling_velocity_1g_m_s": vg, "sigma_required_m2": q / (2.0 * vg),
               "particle_reynolds_1g": d * vg * rho_l / mu}
        assumptions += ["teoría Sigma: Q = 2·v_g·Σ con v_g de Stokes a 1 g; el tamaño dado es el de corte al 50 %",
                        "caudal de alimentación como volumen ideal de sólido + líquido; partículas esféricas sin "
                        "sedimentación impedida"]
        bowl = sg.get("bowl")
        if bowl is not None:
            sb = sigma_bowl(bowl)
            wanted = "disc_stack" if equipment == "disc_stack_centrifuge" else None
            if (wanted == "disc_stack") != (sb["type"] == "disc_stack"):
                warnings.append(f"sigma.bowl.type = {sb['type']} no corresponde a equipment = {equipment}")
            w, r2 = sb["omega_rad_s"], _pos("sigma.bowl.r_outer_m", bowl.get("r_outer_m"), "radio exterior")
            d_cut = math.sqrt(18.0 * mu * q / (2.0 * sb["sigma_m2"] * (rho_p - rho_l) * G))
            v_wall = w * w * r2 * d_cut ** 2 * (rho_p - rho_l) / (18.0 * mu)
            re_wall = d_cut * v_wall * rho_l / mu
            sb.update(capacity_for_particle_m3_s=2.0 * vg * sb["sigma_m2"], cut_diameter_at_feed_flow_m=d_cut,
                      particle_reynolds_at_outer_radius=re_wall,
                      sigma_machine_over_required=sb["sigma_m2"] / res["sigma_required_m2"])
            checks["sigma_cut_size_roundtrip_rel"] = _rel(
                2.0 * (d_cut ** 2 * (rho_p - rho_l) * G / (18.0 * mu)) * sb["sigma_m2"], q)
            if checks["sigma_cut_size_roundtrip_rel"] > 1e-9:
                reasons.append("el tamaño de corte no reproduce el caudal")
            if re_wall > 1.0:
                warnings.append(f"Reynolds de partícula {re_wall:.2g} en la pared del tazón: fuera del régimen de Stokes, la "
                                "teoría Sigma sobrestima la capacidad")
            if sb["sigma_machine_over_required"] < 1.0:
                warnings.append("la Σ de la máquina es menor que la requerida: no separa ese tamaño a ese caudal")
            if sb["type"] == "solid_bowl":
                assumptions.append("decantador tratado como tazón cilíndrico (sin la playa cónica ni el tornillo): la Σ real "
                                   "es menor; los fabricantes aplican una eficiencia a la Σ teórica")
            res["bowl"] = sb
        not_included.append("eficiencia de la Σ real frente a la teórica (dato del fabricante)")
        out["sigma"] = res
    elif is_centrifuge:
        not_included.append("teoría Sigma: falta sigma {particle_diameter_m, particle_density_kg_m3, liquid_density_kg_m3, "
                            "liquid_viscosity_Pa_s, y opcionalmente bowl}")

    # ---- purchase cost (BioSTEAM) and BioSTEAM's reference duty
    ref_power = None
    ci = None
    ver = b["version"]
    if equipment == "screw_press":
        it = b["items"]["screw_press"]
        S = F * 3600.0 / LB
        dc = _decorated_cost(it, S)
        warnings += _range_warnings(it, S, dc, "prensa de tornillo")
        ci = _capital_item("prensa de tornillo", dc["per_unit_at_base"], dc["n_units"], it["CE"], it["F_BM"],
                           f"F_BM de BioSTEAM ({it['class']})", year, cepci,
                           f"correlación de {it['class']} (BioSTEAM {ver}; Seider, «expression - screw press»), válida de "
                           f"{it['lb']:g} a {it['ub']:g} lb/h de alimentación húmeda",
                           f"{dc['size_per_unit']:.4g} lb/h por unidad")
        ref_power = (b["screw_press_kWh_per_t"] * KWH / 1000.0 * F,
                     f"{b['screw_press_kWh_per_t']:g} kWh por tonelada alimentada (ScrewPress.kWh_per_bmt de BioSTEAM, que "
                     "cita Perry 18-126)")
    elif equipment == "decanter_centrifuge":
        sc = b["solids_centrifuge"]
        ts = solids_in * 3600.0 / SHORT_TON
        lo, hi = sc["range_short_ton_h"]
        n_units = max(1, math.ceil(ts / hi))
        if ts < lo:
            warnings.append(f"decantador: {ts:.4g} ton cortas/h de sólidos está por debajo del mínimo de la correlación "
                            f"({lo:g}); BioSTEAM la evalúa igual (extrapolación hacia abajo)")
        if ts > hi:
            warnings.append(f"decantador: {ts:.4g} ton cortas/h de sólidos supera el máximo de la correlación ({hi:g}); "
                            f"BioSTEAM pide {n_units} centrífugas pero costea con la carga TOTAL en una sola fórmula "
                            "(se hizo igual): extrapolación")
        reasons += _source_reason(sc["source_ok"], "SolidsCentrifuge._design")
        ci = _capital_item("centrífuga decantadora (tazón sólido con tornillo)", sc["coefficient"] * ts ** sc["exponent"], 1,
                           sc["CE"], sc["F_BM"], "F_BM de BioSTEAM (biosteam.units.SolidsCentrifuge)", year, cepci,
                           f"correlación de biosteam.units.SolidsCentrifuge (BioSTEAM {ver}; Seider, «scroll solid bowl»): "
                           f"68040·S^0.5 a CE 567, S = ton cortas/h de sólidos, válida de {lo:g} a {hi:g}",
                           f"{ts:.4g} ton cortas/h de sólidos")
        ci["number_of_units"] = n_units
        ci["purchase_cost_per_unit_usd"] = ci["purchase_cost_usd"] / n_units
        if q_feed is not None:
            ref_power = (sc["kWh_per_m3"] * KWH * q_feed, f"{sc['kWh_per_m3']:g} kWh por m³ alimentado "
                         "(SolidsCentrifuge.kWhr_per_m3 de BioSTEAM)")
    elif equipment == "disc_stack_centrifuge":
        it = b["items"]["disc_stack_centrifuge"]
        if q_feed is None:
            not_included.append("costo de la centrífuga de discos: falta feed_density_kg_m3 (o sigma con densidades), porque "
                                "la correlación de BioSTEAM se basa en el caudal volumétrico alimentado")
        else:
            S = q_feed * 3600.0
            dc = _decorated_cost(it, S)
            warnings += _range_warnings(it, S, dc, "centrífuga de discos")
            ci = _capital_item("centrífuga de discos", dc["per_unit_at_base"], dc["n_units"], it["CE"], it["F_BM"],
                               f"F_BM de BioSTEAM ({it['class']})", year, cepci,
                               f"correlación de {it['class']} (BioSTEAM {ver}; modelo de biodiésel del USDA), hasta "
                               f"{it['ub']:g} m³/h por máquina", f"{dc['size_per_unit']:.4g} m³/h por unidad")
            ref_power = (dc["reference_power_W"], f"{it['kW']:g} kW por m³/h alimentado (BioSTEAM LiquidsCentrifuge)")
            warnings.append("centrífuga de discos: la correlación de BioSTEAM es de una centrífuga de discos líquido-líquido "
                            "(biodiésel); aplicada a un clarificador sólido-líquido es un orden de magnitud")
    elif equipment == "rotary_vacuum_drum_filter":
        rv = b["rotary_vacuum_filter"]
        if area_ft2 is None:
            not_included.append("costo del filtro de tambor: necesita el área (dé filtration)")
        else:
            lo, hi = rv["range_ft2"]
            n_units = max(1, math.ceil(area_ft2 / hi))
            ia = area_ft2 / n_units
            if ia < lo:
                warnings.append(f"filtro de tambor: {ia:.4g} ft² está por debajo del mínimo de la correlación ({lo:g} ft²)")
            if n_units > 1:
                warnings.append(f"filtro de tambor: {area_ft2:.4g} ft² supera el máximo por tambor ({hi:g} ft²); se costean "
                                f"{n_units} tambores iguales, como hace BioSTEAM")
            c0, c1, c2 = rv["coefficients"]
            ln = math.log(ia)
            reasons += _source_reason(rv["source_ok"], "RotaryVacuumFilter._cost")
            ci = _capital_item("filtro rotatorio de tambor al vacío", math.exp(c0 + c1 * ln + c2 * ln * ln), n_units,
                               rv["CE"], rv["F_BM"], "F_BM de BioSTEAM (biosteam.units.RotaryVacuumFilter['Vessels'])", year,
                               cepci,
                               f"correlación de biosteam.units.RotaryVacuumFilter (BioSTEAM {ver}; Seider): "
                               f"exp(11.796 − 0.1905·ln A + 0.0554·ln² A) a CE 567, A en ft² por tambor, válida de {lo:g} a "
                               f"{hi:g} ft²", f"{ia:.4g} ft² por tambor")
            not_included.append("sistema de vacío del filtro de tambor (BioSTEAM lo costea aparte con VacuumSystem)")
            warnings.append("filtro de tambor: esta correlación (Seider vía BioSTEAM) da cerca del doble que la tabla de "
                            "costos de DOE/NETL-2002/1169 (ICARUS, 1998) para la misma área: incertidumbre de factor ~2")
    else:
        it = b["items"]["plate_and_frame_filter"]
        S = cake * 3600.0
        dc = _decorated_cost(it, S)
        warnings += _range_warnings(it, S, dc, "filtro prensa")
        ci = _capital_item("filtro prensa (pressure filter)", dc["per_unit_at_base"], dc["n_units"], it["CE"], it["F_BM"],
                           f"F_BM de BioSTEAM ({it['class']}['Pressure filter'])", year, cepci,
                           f"correlación de {it['class']} (BioSTEAM {ver}; filtro prensa de lignina del diseño NREL 2011), "
                           f"base {it['S']:g} kg/h de torta húmeda, exponente {it['n']:g}, sin intervalo publicado",
                           f"{dc['size_per_unit']:.4g} kg/h de torta húmeda")
        warnings.append("filtro prensa: BioSTEAM lo costea por flujo de torta húmeda (escalado del filtro de lignina de "
                        "NREL), no por área; no usa el área calculada con la teoría de filtración")
        not_included.append("auxiliares del filtro prensa que BioSTEAM costea aparte (compresores y filtros de aire, "
                            "tanques, bombas de alimentación y de lavado de telas, transportadores de torta)")
    if ci is not None:
        out["capital_items"] = [ci]
        out["cost"] = {"purchase_cost_usd": ci["purchase_cost_usd"], "number_of_units": ci["number_of_units"],
                       "F_BM": ci["F_BM"], "cepci": ci["cepci"], "cost_year": ci["cost_year"],
                       "correlation_cepci": ci["correlation_cepci"], "biosteam_default_cepci": b["CE_default"]}
    if ref_power is not None:
        out["biosteam_reference_power_W"] = ref_power[0]
        assumptions.append(f"biosteam_reference_power_W: {ref_power[1]}; valor por omisión de BioSTEAM, orientativo, no un "
                           "dato de este equipo")
    else:
        not_included.append("consumo eléctrico del equipo (dato del fabricante)")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method="balance de masa con el desempeño del usuario; teoría de filtración a presión constante (raíz cerrada "
                      "de la cuadrática) y teoría Sigma con Stokes; costo con la correlación de BioSTEAM escalada con CEPCI",
               evidence=_evidence(checks, reasons))
    return out


# --------------------------------------------------------------------------------------------------------------------------
# 4. Oil extraction
# --------------------------------------------------------------------------------------------------------------------------
def expeller_press(seed: dict, expeller: dict, year: int = 2023, cepci: float | None = None) -> dict:
    """Mass balance of mechanical oil extraction: seed -> (hull) + crude oil + press cake. Recovery or residual oil is the
    user's datum."""
    if not isinstance(seed, dict) or not isinstance(expeller, dict):
        raise ValueError("mode='expeller' necesita seed {mass_flow_kg_s, oil_mass_fraction, …} y expeller "
                         "{oil_recovery_fraction o residual_oil_in_cake_mass_fraction, …}")
    Fs = _pos("seed.mass_flow_kg_s", seed.get("mass_flow_kg_s"), "flujo de semilla tal como llega")
    w_oil = _frac("seed.oil_mass_fraction", seed.get("oil_mass_fraction"), "fracción másica de aceite de la semilla tal como "
                  "llega (base húmeda; análisis del lote)", open_low=True, open_high=True)
    warnings: list[str] = []
    reasons: list[str] = []
    assumptions = ["la recuperación de aceite (o el aceite residual de la torta) es un dato del usuario: depende de la "
                   "semilla, del acondicionamiento (humedad, cocción) y de la prensa"]
    not_included = ["acondicionamiento de la semilla (limpieza, quebrado, cocción, secado) y su consumo de vapor",
                    "agua evaporada en la prensa y enfriamiento de la torta",
                    "clarificación del aceite crudo (tamizado, decantación, filtrado) salvo los finos declarados"]
    oil_in = Fs * w_oil
    hull = hull_oil = 0.0
    dec = seed.get("decortication")
    if dec is not None:
        w_hull = _frac("seed.decortication.hull_mass_fraction", dec.get("hull_mass_fraction"),
                       "fracción másica de cáscara en la semilla", open_low=True, open_high=True)
        removal = _frac("seed.decortication.hull_removal_fraction", dec.get("hull_removal_fraction"),
                        "fracción de la cáscara que se retira antes de prensar")
        w_oil_hull = _frac("seed.decortication.oil_mass_fraction_in_hull", dec.get("oil_mass_fraction_in_hull"),
                           "fracción másica de aceite en la corriente de cáscara retirada (incluye almendra arrastrada; 0 si "
                           "se desprecia, dicho explícitamente)", open_high=True)
        hull = Fs * w_hull * removal
        hull_oil = hull * w_oil_hull
        if hull_oil >= oil_in:
            raise ValueError("la cáscara retirada se llevaría todo el aceite de la semilla: revisar oil_mass_fraction_in_hull")
    press_feed = Fs - hull
    press_oil = oil_in - hull_oil
    non_oil = press_feed - press_oil
    rec_in = expeller.get("oil_recovery_fraction")
    res_in = expeller.get("residual_oil_in_cake_mass_fraction")
    if (rec_in is None) == (res_in is None):
        raise ValueError("falta el desempeño de la prensa: dé expeller.oil_recovery_fraction (fracción del aceite que entra a "
                         "la prensa y sale como aceite) O expeller.residual_oil_in_cake_mass_fraction (aceite en la torta), "
                         "uno solo; published_values()['oilseeds'] lista valores publicados para ricino, colza, jatropha y soya")
    if rec_in is not None:
        rec = _frac("expeller.oil_recovery_fraction", rec_in, "fracción del aceite recuperada", open_low=True)
        oil_out = rec * press_oil
        cake = press_feed - oil_out
        cake_oil = press_oil - oil_out
    else:
        w_c = _frac("expeller.residual_oil_in_cake_mass_fraction", res_in, "fracción másica de aceite en la torta",
                    open_high=True)
        cake = non_oil / (1.0 - w_c)
        cake_oil = cake * w_c
        oil_out = press_oil - cake_oil
        if oil_out <= 0.0:
            raise ValueError(f"con {w_c:.3g} de aceite residual la torta retiene todo el aceite que entra a la prensa "
                             f"(la alimentación a la prensa tiene {press_oil / press_feed:.3g}): no sale aceite")
    fines_frac = expeller.get("solids_in_crude_oil_mass_fraction")
    fines = 0.0
    if fines_frac is not None:
        ff = _frac("expeller.solids_in_crude_oil_mass_fraction", fines_frac, "fracción másica de finos (sólidos) en el aceite "
                   "crudo que sale de la prensa", open_high=True)
        fines = oil_out * ff / (1.0 - ff)
        if fines >= cake - cake_oil:
            raise ValueError("los finos declarados en el aceite superan los sólidos disponibles en la torta")
        cake -= fines
        assumptions.append("los finos del aceite crudo son sólidos sin aceite que se restan de la torta")
    else:
        assumptions.append("aceite crudo sin finos (separación ideal): dé solids_in_crude_oil_mass_fraction si los conoce")
    crude = oil_out + fines
    checks = {"mass_balance_rel": abs(Fs - hull - crude - cake) / Fs,
              "oil_balance_rel": abs(oil_in - hull_oil - oil_out - cake_oil) / oil_in}
    if max(checks.values()) > BALANCE_TOL:
        reasons.append("el balance de masa no cierra")
    out: dict = {
        "mode": "expeller",
        "seed": {"total_kg_s": Fs, "oil_kg_s": oil_in, "oil_mass_fraction": w_oil},
        "hull": {"total_kg_s": hull, "oil_kg_s": hull_oil} if dec is not None else None,
        "press_feed": {"total_kg_s": press_feed, "oil_kg_s": press_oil, "oil_mass_fraction": press_oil / press_feed},
        "crude_oil": {"total_kg_s": crude, "oil_kg_s": oil_out, "solids_kg_s": fines},
        "cake": {"total_kg_s": cake, "oil_kg_s": cake_oil, "oil_mass_fraction": cake_oil / cake,
                 "oil_free_solids_kg_s": cake - cake_oil},
        "oil_recovery_fraction_of_press_feed": oil_out / press_oil,
        "oil_recovery_fraction_of_seed": oil_out / oil_in,
        "oil_yield_kg_per_kg_seed": oil_out / Fs,
        "capacity": {"press_feed_kg_s": press_feed, "press_feed_tonne_per_day": press_feed * 86.4},
        "capital_items": [], "cost": None, "power_W": None,
    }
    e_spec = expeller.get("specific_energy_J_kg")
    if e_spec is not None:
        out["power_W"] = _pos("expeller.specific_energy_J_kg", e_spec, "energía eléctrica por kg alimentado a la prensa") \
            * press_feed
        assumptions.append("potencia = energía específica del usuario × alimentación a la prensa")
    else:
        not_included.append("potencia de la prensa: falta expeller.specific_energy_J_kg (J por kg alimentado); publicados: "
                            "0.08-0.33 GJ/t de colza en 20 plantas de prensado en frío (media 0.16; Widmann 1999) y 37.2 "
                            "kWh/t en la prensa de tornillo de BioSTEAM (Perry 18-126)")
    if expeller.get("estimate_cost"):
        b = _biosteam_data()
        it = b["items"]["screw_press"]
        S = press_feed * 3600.0 / LB
        dc = _decorated_cost(it, S)
        warnings += _range_warnings(it, S, dc, "prensa de tornillo")
        ci = _capital_item("prensa de tornillo (expeller)", dc["per_unit_at_base"], dc["n_units"], it["CE"], it["F_BM"],
                           f"F_BM de BioSTEAM ({it['class']})", year, cepci,
                           f"correlación de {it['class']} (BioSTEAM {b['version']}; Seider, «expression - screw press»), "
                           f"válida de {it['lb']:g} a {it['ub']:g} lb/h de alimentación",
                           f"{dc['size_per_unit']:.4g} lb/h por unidad")
        out["capital_items"] = [ci]
        out["cost"] = {"purchase_cost_usd": ci["purchase_cost_usd"], "number_of_units": ci["number_of_units"],
                       "F_BM": ci["F_BM"], "cepci": ci["cepci"], "cost_year": ci["cost_year"],
                       "correlation_cepci": it["CE"], "biosteam_default_cepci": b["CE_default"]}
        warnings.append("costo: correlación genérica de prensa de tornillo para expresión de líquidos (Seider vía BioSTEAM), "
                        "no un precio de expeller de oleaginosas con cocedor y acondicionador: orden de magnitud; confirmar "
                        "con cotización")
    else:
        not_included.append("costo de compra del expeller: no existe correlación abierta específica de expellers de "
                            "oleaginosas; pedir cotización (expeller.estimate_cost=true da el orden de magnitud con la "
                            "prensa de tornillo genérica de Seider/BioSTEAM)")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method="balance de masa total y de aceite en forma cerrada", evidence=_evidence(checks, reasons),
               published_reference=OILSEED_PUBLISHED)
    return out


def _retention(underflow: dict):
    """Return (R(y) = kg of solution retained per kg of inert solid, y_max of the data, description)."""
    if not isinstance(underflow, dict):
        raise ValueError("falta underflow: retención de solución por los sólidos (dato de ensayo)")
    const = underflow.get("solution_retained_kg_per_kg_inert")
    curve = underflow.get("curve")
    if (const is None) == (curve is None):
        raise ValueError("underflow: dé solution_retained_kg_per_kg_inert (retención constante, kg de solución por kg de "
                         "sólido inerte) O curve {oil_mass_fraction_in_solution: [...], kg_inert_per_kg_solution: [...]} "
                         "(retención variable, la N de Geankoplis), uno solo")
    if const is not None:
        R0 = _pos("underflow.solution_retained_kg_per_kg_inert", const, "kg de solución retenida por kg de sólido inerte")
        return (lambda y: R0), 1.0, "retención constante (underflow constante)"
    ys = curve.get("oil_mass_fraction_in_solution")
    Ns = curve.get("kg_inert_per_kg_solution")
    if not ys or not Ns or len(ys) != len(Ns) or len(ys) < 2:
        raise ValueError("underflow.curve necesita oil_mass_fraction_in_solution y kg_inert_per_kg_solution con el mismo "
                         "número de puntos (≥ 2)")
    ys = [float(v) for v in ys]
    Ns = [float(v) for v in Ns]
    if any(b <= a for a, b in zip(ys, ys[1:])) or ys[0] < 0.0 or ys[-1] > 1.0 or min(Ns) <= 0.0:
        raise ValueError("underflow.curve: las fracciones de aceite deben ser crecientes y estar en [0, 1], y N > 0")
    if ys[0] > 0.0:
        raise ValueError("underflow.curve debe empezar en fracción de aceite 0 (retención con disolvente puro): no se "
                         "extrapola")
    if len(ys) >= 3:
        from scipy.interpolate import PchipInterpolator
        fN = PchipInterpolator(ys, Ns)
        kind = "PCHIP"
    else:
        fN = lambda y: Ns[0] + (Ns[1] - Ns[0]) * (y - ys[0]) / (ys[1] - ys[0])  # noqa: E731
        kind = "lineal"

    def R(y: float) -> float:
        if y < -1e-12 or y > ys[-1] * (1.0 + 1e-12):
            raise ArithmeticError(f"la fracción de aceite {y:.4g} sale del intervalo de la curva de retención "
                                  f"[0, {ys[-1]:g}]")
        return 1.0 / float(fN(min(max(y, 0.0), ys[-1])))
    return R, ys[-1], f"retención variable interpolada ({kind}) en la curva del usuario, sin extrapolar"


def _leach_march(B: float, L0: float, y0: float, V1: float, x1: float, R, y_target: float, n_max: int):
    """Stage-to-stage balances from the feed end. Returns (stages, status): 'reached' when the underflow got down to
    y_target, 'max_stages', 'pinch' (the concentration stopped falling), 'negative' (the balances ask for a negative
    concentration: the cascade is longer than needed) or 'out_of_curve'."""
    stages: list[dict] = []
    L_prev, y_prev, V_n, x_n = L0, y0, V1, x1
    net_total = V1 - L0                    # V_{n+1} − L_n, constant along the cascade
    net_oil = V1 * x1 - L0 * y0            # V_{n+1} x_{n+1} − L_n y_n, constant along the cascade
    for n in range(1, n_max + 1):
        y_n = x_n                                              # ideal stage
        if y_n < 0.0:
            return stages, "negative"
        try:
            L_n = B * R(y_n)
        except ArithmeticError:
            return stages, "out_of_curve"
        V_next = L_n + net_total
        if V_next <= 0.0:
            return stages, "pinch"
        x_next = (L_n * y_n + net_oil) / V_next
        stages.append({"stage": n, "underflow_solution_kg_s": L_n, "underflow_oil_mass_fraction": y_n,
                       "overflow_kg_s": V_n, "overflow_oil_mass_fraction": x_n,
                       "overflow_in_kg_s": V_next, "overflow_in_oil_mass_fraction": x_next,
                       "total_balance_residual_kg_s": L_prev + V_next - L_n - V_n,
                       "oil_balance_residual_kg_s": L_prev * y_prev + V_next * x_next - L_n * y_n - V_n * x_n})
        if y_n <= y_target * (1.0 + 1e-10):
            return stages, "reached"
        if x_next >= x_n * (1.0 - 1e-12):
            return stages, "pinch"
        L_prev, y_prev, V_n, x_n = L_n, y_n, V_next, x_next
    return stages, "max_stages"


def solvent_leaching(solids: dict, solvent: dict, underflow: dict, target: dict, energy: dict | None = None,
                     solvent_loss_kg_per_kg_solids_fed: float | None = None, max_stages: int = MAX_STAGES) -> dict:
    """Ideal countercurrent leaching stages (oil = solute, solvent, inert solids) with constant or variable underflow."""
    for name, d in (("solids", solids), ("solvent", solvent), ("target", target)):
        if not isinstance(d, dict):
            raise ValueError(f"mode='solvent' necesita {name} (diccionario); ver la descripción de la herramienta")
    B = _pos("solids.inert_solids_kg_s", solids.get("inert_solids_kg_s"), "sólido inerte (sin aceite ni disolvente) que "
             "entra al extractor")
    oil0 = _pos("solids.oil_kg_s", solids.get("oil_kg_s"), "aceite que entra con los sólidos")
    solv0 = _nonneg("solids.solvent_kg_s", solids.get("solvent_kg_s", 0.0), "disolvente que ya trae el sólido")
    Vs = _pos("solvent.solvent_kg_s", solvent.get("solvent_kg_s"), "disolvente alimentado al extractor (sin contar su "
              "aceite)")
    Vo = _nonneg("solvent.oil_kg_s", solvent.get("oil_kg_s", 0.0), "aceite que trae el disolvente recirculado")
    R, y_max, underflow_text = _retention(underflow)
    warnings: list[str] = []
    reasons: list[str] = []
    assumptions = ["etapas ideales: la solución retenida por el sólido tiene la composición del extracto de la etapa",
                   "el extracto (miscela) no arrastra sólidos; el sólido inerte no se disuelve",
                   underflow_text,
                   "solids.solvent_kg_s y solvent.oil_kg_s valen 0 si no se dan (corrientes sin ese componente)"]
    not_included = ["cinética de extracción y difusión en la hojuela (eficiencia de etapa real)",
                    "humedad del sólido y agua en la miscela", "arrastre de finos en la miscela",
                    "costo de compra del extractor y del desolventizador-tostador: no hay correlación abierta; pedir "
                    "cotización"]
    L0 = oil0 + solv0
    y0 = oil0 / L0
    VN1 = Vs + Vo
    xN1 = Vo / VN1
    M = L0 + VN1
    oil_total = oil0 + Vo

    def end_state(yN: float):
        LN = B * R(yN)
        V1 = M - LN
        if V1 <= 0.0:
            raise ArithmeticError("los sólidos retienen todo el líquido: no sale miscela")
        x1 = (oil_total - LN * yN) / V1
        return LN, V1, x1

    given = [k for k in ("residual_oil_kg_s", "residual_oil_in_meal_mass_fraction", "stages") if target.get(k) is not None]
    if len(given) != 1:
        raise ValueError("target: dé exactamente uno de residual_oil_kg_s (aceite que sale con el sólido agotado), "
                         "residual_oil_in_meal_mass_fraction (aceite / (inerte + aceite), harina sin disolvente) o stages "
                         "(número de etapas ideales, para calcular el residual)")
    from scipy.optimize import brentq
    y_lo = xN1 * (1.0 + 1e-9) + 1e-15         # the last underflow cannot be leaner than the entering solvent
    y_hi = min(y_max, 1.0) * (1.0 - 1e-12)
    mode_target = given[0]
    n_spec = None
    if mode_target == "stages":
        n_spec = int(_pos("target.stages", target.get("stages"), "número de etapas ideales"))
        if n_spec != target.get("stages") or n_spec > max_stages:
            raise ValueError(f"target.stages debe ser un entero entre 1 y {max_stages}")

        def miss(yN: float) -> float:
            """y after exactly n_spec stages minus yN: > 0 if yN is too ambitious for n_spec stages, < 0 if n_spec stages
            do better than yN."""
            try:
                _, V1_, x1_ = end_state(yN)
            except ArithmeticError:
                return 1.0
            st, status = _leach_march(B, L0, y0, V1_, x1_, R, -1.0, n_spec)
            if len(st) == n_spec:
                return st[-1]["underflow_oil_mass_fraction"] - yN
            return -1.0 if status == "negative" else 1.0

        grid = [y_lo + (y_hi - y_lo) * (i / 600.0) ** 3 for i in range(601)]
        lo = hi = None
        g_prev = miss(grid[0])
        for a_, c_ in zip(grid, grid[1:]):
            g_next = miss(c_)
            if g_prev > 0.0 >= g_next:
                lo, hi = a_, c_
                break
            g_prev = g_next
        if lo is None:
            raise ValueError("no se encontró un residual compatible con ese número de etapas y esa relación de disolvente "
                             "(revise la curva de retención y los flujos)")
        yN = brentq(miss, lo, hi, xtol=1e-16, rtol=1e-14)
        oil_res = B * R(yN) * yN
    else:
        if mode_target == "residual_oil_kg_s":
            oil_res = _pos("target.residual_oil_kg_s", target.get("residual_oil_kg_s"), "aceite que sale con el sólido "
                           "agotado")
        else:
            wm = _frac("target.residual_oil_in_meal_mass_fraction", target.get("residual_oil_in_meal_mass_fraction"),
                       "aceite en la harina sin disolvente", open_low=True, open_high=True)
            oil_res = wm * B / (1.0 - wm)
        if oil_res >= oil_total:
            raise ValueError("el aceite residual pedido no es menor que el aceite que entra: no hay extracción")
        f = lambda y: B * R(y) * y - oil_res  # noqa: E731
        if f(y_lo) > 0.0:
            raise ValueError("el residual pedido es menor que el aceite que retiene el sólido mojado con el disolvente que "
                             "entra: inalcanzable aunque haya infinitas etapas")
        if f(y_hi) < 0.0:
            raise ValueError("el residual pedido queda fuera de la curva de retención dada (solución más rica que su "
                             "último punto): no se extrapola")
        yN = brentq(f, y_lo, y_hi, xtol=1e-16, rtol=1e-14)
    try:
        LN, V1, x1 = end_state(yN)
        if not 0.0 < x1 < 1.0:
            raise ArithmeticError("la miscela resultante no tiene una composición física")
        if x1 > y_max:
            raise ArithmeticError(f"la miscela ({x1:.3g} de aceite) sale del intervalo de la curva de retención")
    except ArithmeticError as exc:
        raise ValueError(f"especificación inalcanzable con este disolvente y esta retención: {exc}") from exc
    if mode_target == "stages":
        stages, status = _leach_march(B, L0, y0, V1, x1, R, -1.0, n_spec)
        if len(stages) != n_spec:
            raise ValueError("no se pudo cerrar la cascada con ese número de etapas")
    else:
        stages, status = _leach_march(B, L0, y0, V1, x1, R, yN, max_stages)
        if status == "pinch" or not stages:
            raise ValueError("especificación inalcanzable: con este disolvente la concentración deja de bajar antes de "
                             f"llegar al residual pedido (pinzamiento en la etapa {len(stages)}); aumente el disolvente")
        if status != "reached":
            reasons.append(f"no se alcanzó el residual pedido en {max_stages} etapas ({status})")
    n = len(stages)
    y_last = stages[-1]["underflow_oil_mass_fraction"]
    y_before = stages[-2]["underflow_oil_mass_fraction"] if n >= 2 else y0
    frac = (n - 1) + ((y_before - yN) / (y_before - y_last) if y_before > y_last else 1.0)
    if mode_target == "stages":
        frac = float(n)
    scale = M
    checks = {
        "overall_total_balance_rel": abs(L0 + VN1 - LN - V1) / scale,
        "overall_oil_balance_rel": abs(oil_total - LN * yN - V1 * x1) / oil_total,
        "max_stage_total_balance_rel": max(abs(s["total_balance_residual_kg_s"]) for s in stages) / scale,
        "max_stage_oil_balance_rel": max(abs(s["oil_balance_residual_kg_s"]) for s in stages) / oil_total,
        "last_stage_solvent_composition_rel_diff": (_rel(stages[-1]["overflow_in_oil_mass_fraction"] + 1.0, xN1 + 1.0)
                                                    if mode_target == "stages" else None),
        "monotonic_profile": all(a["underflow_oil_mass_fraction"] > c["underflow_oil_mass_fraction"]
                                 for a, c in zip(stages, stages[1:])),
    }
    if max(checks["overall_total_balance_rel"], checks["overall_oil_balance_rel"], checks["max_stage_total_balance_rel"],
           checks["max_stage_oil_balance_rel"]) > 1e-8:
        reasons.append("los balances de etapa no cierran")
    if not checks["monotonic_profile"]:
        reasons.append("el perfil de concentraciones no es monótono")
    if mode_target == "stages" and checks["last_stage_solvent_composition_rel_diff"] > 1e-7:
        reasons.append("la última etapa no cierra con la composición del disolvente alimentado")
    solvent_in_meal = LN * (1.0 - yN)
    solvent_in_miscella = V1 * (1.0 - x1)
    out: dict = {
        "mode": "solvent",
        "stages_ideal_fractional": frac, "stages_ideal": n,
        "residual_oil_kg_s": LN * yN, "residual_oil_in_meal_mass_fraction": LN * yN / (B + LN * yN),
        "oil_recovery_fraction": (oil0 - LN * yN) / oil0,
        "miscella": {"total_kg_s": V1, "oil_mass_fraction": x1, "oil_kg_s": V1 * x1, "solvent_kg_s": solvent_in_miscella},
        "leached_solids": {"inert_solids_kg_s": B, "solution_kg_s": LN, "oil_mass_fraction_in_solution": yN,
                           "oil_kg_s": LN * yN, "solvent_kg_s": solvent_in_meal,
                           "solvent_mass_fraction_of_wet_solids": solvent_in_meal / (B + LN),
                           "kg_inert_per_kg_solution": B / LN},
        "feed": {"inert_solids_kg_s": B, "solution_kg_s": L0, "oil_mass_fraction_in_solution": y0},
        "solvent_feed": {"total_kg_s": VN1, "oil_mass_fraction": xN1,
                         "solvent_to_solids_kg_per_kg": Vs / (B + oil0 + solv0)},
        "stage_profile": [{k: v for k, v in s.items() if not k.endswith("residual_kg_s")} for s in stages],
        "solvent_makeup_kg_s": None, "stripping": None, "capital_items": [], "cost": None,
        "capacity": {"solids_feed_kg_s": B + oil0 + solv0, "solids_feed_tonne_per_day": (B + oil0 + solv0) * 86.4},
    }
    if mode_target != "stages" and n >= 1:
        out["note_on_integer_stages"] = ("con el número entero de etapas el residual real es algo menor que el pedido; "
                                         "llame con target {'stages': n} para obtenerlo")
    if solvent_loss_kg_per_kg_solids_fed is not None:
        loss = _nonneg("solvent_loss_kg_per_kg_solids_fed", solvent_loss_kg_per_kg_solids_fed, "pérdida total de disolvente "
                       "por kg de sólido alimentado")
        out["solvent_makeup_kg_s"] = loss * (B + oil0 + solv0)
        out["solvent_recycled_kg_s"] = Vs + solv0 - out["solvent_makeup_kg_s"]
        assumptions.append("reposición de disolvente = pérdidas del usuario × sólido alimentado (estado estacionario)")
    else:
        not_included.append("reposición de disolvente: falta solvent_loss_kg_per_kg_solids_fed (publicado para soya: media "
                            "de 4.9 lb/ton = 2.45e-3 kg/kg, EPA AP-42 9.11.1, calidad D)")
    if energy is not None:
        out["stripping"], w_e, a_e, r_e = _stripping_heat(energy, solvent_in_miscella, solvent_in_meal)
        warnings += w_e
        assumptions += a_e
        reasons += r_e
        not_included.append("calor sensible del aceite y del sólido, vapor directo de agotamiento, tostado de la harina y "
                            "recuperación de calor (economizadores) en la destilación de la miscela")
    else:
        not_included.append("calor para evaporar el disolvente: falta energy {stream_T_K, stripping_pressure_Pa}")
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method="balances total y de aceite etapa a etapa desde el extremo de alimentación (etapas ideales); raíces "
                      "con scipy.optimize.brentq; interpolación PCHIP de la curva de retención",
               evidence=_evidence(checks, reasons), published_reference=OILSEED_PUBLISHED)
    return out


def _stripping_heat(energy: dict, solvent_in_miscella: float, solvent_in_meal: float):
    T_in = _pos("energy.stream_T_K", energy.get("stream_T_K"), "temperatura a la que salen la miscela y el sólido del "
                "extractor")
    P = _pos("energy.stripping_pressure_Pa", energy.get("stripping_pressure_Pa"), "presión a la que se evapora el "
             "disolvente")
    fluid = energy.get("solvent_fluid", "n-Hexane")
    warnings, assumptions, reasons = [], [], []
    from CoolProp.CoolProp import PropsSI
    try:
        T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
        h_vap = PropsSI("H", "P", P, "Q", 1, fluid)
        h_liq_sat = PropsSI("H", "P", P, "Q", 0, fluid)
        if T_in < T_sat:
            h_in = PropsSI("H", "T", T_in, "P", P, fluid)
        else:
            h_in = h_liq_sat
            warnings.append(f"stream_T_K ≥ temperatura de saturación del disolvente a esa presión ({T_sat:.2f} K): se toma "
                            "líquido saturado como estado inicial")
    except ValueError as exc:
        raise ValueError(f"CoolProp no pudo evaluar el disolvente {fluid!r} a esa presión: {exc}") from exc
    dh = h_vap - h_in
    latent = h_vap - h_liq_sat
    res = {"solvent_fluid": fluid, "saturation_T_K": T_sat, "latent_heat_J_kg": latent, "enthalpy_rise_J_kg": dh,
           "miscella_solvent_kg_s": solvent_in_miscella, "meal_solvent_kg_s": solvent_in_meal,
           "miscella_duty_W": dh * solvent_in_miscella, "meal_duty_W": dh * solvent_in_meal,
           "total_duty_W": dh * (solvent_in_miscella + solvent_in_meal)}
    assumptions += [f"calor de agotamiento = disolvente × (entalpía de vapor saturado − entalpía del líquido a stream_T_K) "
                    f"de {fluid} puro (CoolProp): estimación simple; el hexano comercial es una mezcla de isómeros",
                    "no se resta el calor recuperado al condensar el disolvente; es el calor bruto de evaporación"]
    if not (dh > 0.0 and math.isfinite(dh)):
        reasons.append("la entalpía de evaporación del disolvente no es positiva")
    return res, warnings, assumptions, reasons


def oil_extraction(mode: str, seed: dict | None = None, expeller: dict | None = None, solids: dict | None = None,
                   solvent: dict | None = None, underflow: dict | None = None, target: dict | None = None,
                   energy: dict | None = None, solvent_loss_kg_per_kg_solids_fed: float | None = None, year: int = 2023,
                   cepci: float | None = None) -> dict:
    """Oil extraction: mode 'expeller' (mechanical press balance) or 'solvent' (countercurrent leaching stages)."""
    if mode == "expeller":
        return expeller_press(seed, expeller, year, cepci)
    if mode == "solvent":
        return solvent_leaching(solids, solvent, underflow, target, energy, solvent_loss_kg_per_kg_solids_fed)
    raise ValueError("mode debe ser 'expeller' (prensado mecánico) o 'solvent' (extracción con disolvente en contracorriente)")
