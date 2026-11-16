"""Utilities capital (bloque M26-U): design basis and CAPITAL of a cooling tower, a steam boiler with an optional
turbogenerator, and a conventional wastewater treatment plant. Operating cost of steam, cooling water and refrigeration
lives in `packages/costing/utilities.py` (M17) and is not repeated here.

Physics (verified on every run by closure):
  * Cooling tower: water and energy balance of water against humid air. Liquid water from IAPWS-95 and humid air from
    CoolProp's HumidAirProp (ASHRAE RP-1485 real-gas model); both share the IAPWS reference state of water. Air leaves
    saturated (Merkel's assumption). Merkel number by quadrature when the air flow is known.
  * Boiler: steam-side duty with IAPWS-95; fuel from the user's efficiency and heating value or from the repo's heat-loss
    engine (`packages.energy.combustion.combustion_efficiency`). Turbogenerator: section-by-section expansion with the
    user's isentropic efficiency.
  * Wastewater: COD balance by stage; methane from the stoichiometry CH4 + 2 O2 -> CO2 + 2 H2O (1 kg COD = 1 kg O2).

Capital: every number is a published point or table, read on READ_DATE in the open documents of SOURCES, scaled with the
exponent of its own source and moved in time with the CEPCI. BioSTEAM's facility correlations (cooling tower, boiler,
turbogenerator, wastewater system) are read from the @cost decorators in the source of the installed BioSTEAM, without
importing it (nothing global is touched); they ARE the vendor quotes of Humbird et al. (2011), so an agreement between
BioSTEAM and Humbird is not a validation. The independent points are Aden et al.
(2002), the DOE/NETL tables of Loh et al. (2002), Smith (2005, Table 2.1) and the two EPA CHP catalogs.

Selection rule of this module (stated in every result): a published TABLE whose range contains the size wins over a
single quote extrapolated with a power law; all the other estimates are returned next to it with their deviation.

Suggested design-basis inputs (owner's rule of 2026-11-16): when an optional design datum is not given, a value READ in
one of SOURCES is used and returned in `suggested_inputs` and `assumptions` with its source; the caller can override it.
Nothing physical of the user's streams (temperatures, duties, flows, heating values, COD, removals) is ever suggested.
"""
from __future__ import annotations

import ast
import math
import warnings as _warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

G_STD = 9.80665                      # m/s²
P_ATM = 101325.0                     # Pa
US_GAL_M3 = 3.785411784e-3           # exact
LB_KG = 0.45359237                   # exact
HP_W = 745.6998715822702             # mechanical horsepower
KCAL_J = 4184.0                      # thermochemical kcal, the one of the NREL reports' "Gcal"
M_H2O = 18.01528                     # g/mol
M_CH4 = 16.0425                      # g/mol (IUPAC atomic weights)
M_O2 = 31.9988                       # g/mol
M_H2 = 2.01588                       # g/mol
NORMAL_M3_PER_MOL = 8.314462618 * 273.15 / 101325.0   # ideal gas at 0 °C and 1 atm
READ_DATE = "2026-11-16"

SOURCES = {
    "humbird2011": {
        "citation": "Humbird, Davis, Tao, Kinchin, Hsu, Aden, Schoen, Lukas, Olthof, Worley, Sexton, Dudgeon (2011). Process "
                    "Design and Economics for Biochemical Conversion of Lignocellulosic Biomass to Ethanol. NREL/TP-5100-47764",
        "url": "https://docs.nlr.gov/docs/fy11osti/47764.pdf", "read": READ_DATE},
    "aden2002": {
        "citation": "Aden, Ruth, Ibsen, Jechura, Neeves, Sheehan, Wallace, Montague, Slayton, Lukas (2002). Lignocellulosic "
                    "Biomass to Ethanol Process Design and Economics. NREL/TP-510-32438",
        "url": "https://docs.nlr.gov/docs/fy02osti/32438.pdf", "read": READ_DATE},
    "netl2002": {
        "citation": "Loh, Lyons, White (2002). Process Equipment Cost Estimation, Final Report. DOE/NETL-2002/1169 "
                    "(Apéndice B: costos comprado/instalado generados con ICARUS, dólares del 1er trimestre de 1998)",
        "url": "https://www.osti.gov/servlets/purl/797810", "read": READ_DATE},
    "smith2005": {
        "citation": "Smith, R. (2005). Chemical Process Design and Integration, Wiley. Tabla 2.1 (costos entregados, enero de "
                    "2000, índice CE compuesto 391.1)",
        "url": "https://fenix.ciencias.ulisboa.pt/downloadFile/1970462275930926/Process%20Economics%20in%20R%20Smith_%20"
               "Chemical%20Process%20%20Design%20and%20Integration_%20John%20Wiley%20and%20Sons%202005.pdf",
        "read": READ_DATE},
    "epa_chp2015": {
        "citation": "U.S. EPA Combined Heat and Power Partnership (marzo de 2015). Catalog of CHP Technologies, Section 4: "
                    "Steam Turbines, Tabla 4-2 (sistemas típicos disponibles en 2014, datos de Elliott Group)",
        "url": "https://www.epa.gov/sites/default/files/2015-07/documents/"
               "catalog_of_chp_technologies_section_4._technology_characterization_-_steam_turbines.pdf", "read": READ_DATE},
    "epa_biomass2007": {
        "citation": "U.S. EPA Combined Heat and Power Partnership (septiembre de 2007). Biomass CHP Catalog of Technologies, "
                    "cap. 5, Tablas 5-4, 5-5 y 5-7 (datos de Antares Group 2003 y proveedores; el capítulo no declara el año "
                    "de los dólares)",
        "url": "https://www.epa.gov/sites/default/files/2015-07/documents/"
               "biomass_combined_heat_and_power_catalog_of_technologies_5._biomass_conversion_technologies.pdf",
        "read": READ_DATE},
    "cepci": {"citation": "Chemical Engineering Plant Cost Index, promedios anuales: tabla que distribuye IDAES "
                          "(idaes/core/base/costing_base.py); 1998 y 2000 coinciden con la Tabla 11 de Loh et al. (2002)",
              "url": "https://github.com/IDAES/idaes-pse/blob/main/idaes/core/base/costing_base.py", "read": READ_DATE},
}
# Annual CEPCI of the years of the published points (IDAES table; 1998 and 2000 also read in Loh et al. 2002, Table 11).
CEPCI_REF = {1998: 389.5, 2000: 394.1, 2003: 402.0, 2007: 525.4, 2010: 550.8, 2014: 576.1}
CEPCI_SMITH_JAN_2000 = 391.1

# ---- Humbird et al. 2011 (2010 quotes; equipment list of Appendix A and the text of sections 3.6-3.9) -------------------
HUMBIRD_TOWER = {"cost_usd": 1_375_000.0, "year": 2010, "flow_kg_h": 10_037_820.0, "exponent": 0.60, "installation": 1.5,
                 "description": "M-902, Marley SPX, fibra de vidrio, 44,200 gpm, 750 hp"}
HUMBIRD_CW = {"circulation_kg_h": 11_923_904.0, "fan_kW": 1021.0, "pump_kW": 1598.0, "duty_Gcal_h": 97.401,
              "supply_C": 28.0, "return_C": 37.0, "evaporation_kg_h": 137_363.0, "blowdown_kg_h": 17_678.0,
              "makeup_kg_h": 155_041.0, "air_in_kg_h": 6_549_403.0, "air_in_water_kg_h": 128_274.0, "air_in_C": 25.0,
              "air_out_kg_h": 6_687_697.0, "air_out_water_kg_h": 266_568.0, "air_out_C": 37.0,
              "drift_fraction_of_circulation": 0.00005, "pump_head_ft": 100.0}
HUMBIRD_BLOWDOWN_OF_STEAM = 0.03
# Area 600 (wastewater): installed costs of 2010 (installation factor 1.0 in the source), all scaled in the source on the
# hydraulic flow strm.a600.601 = 393,100 kg/h with exponent 0.6. group: which part of the plant the item belongs to.
HUMBIRD_WWT_FLOW_KG_H = 393_100.0
HUMBIRD_WWT_EXPONENT = 0.60
HUMBIRD_WWT_DESIGN = {"hydraulic_MMgal_d": 2.15, "total_COD_g_L": 87.0, "anaerobic_removal": 0.91,
                      "anaerobic_to_biomass": 0.05, "aerobic_removal": 0.96, "aerobic_to_biomass": 0.22,
                      "COD_to_digester_kg_h": 26_500.0, "methane_g_per_kg_COD_removed": 228.0,
                      "biogas_MMkcal_h": 65.0, "total_removal": 0.996, "power_share_of_41_MW": 0.18}
HUMBIRD_WWT_ITEMS = [
    ("T-606", "Cuenca anaerobia con cubierta (concreto)", 27_000_000.0, "anaerobic_reactor"),
    ("M-606", "Quemador de emergencia de biogás", 32_955.0, "anaerobic"),
    ("P-607", "Bomba de lodo anaerobio de desecho", 93_300.0, "anaerobic"),
    ("H-602", "Enfriador de alimentación al digestor anaerobio", 83_863.0, "anaerobic"),
    ("T-608", "Digestor aerobio (lagunas de aireación, concreto)", 2_700_000.0, "aerobic_reactor"),
    ("B-608", "Sopladores del digestor aerobio", 1_933_750.0, "aerobic_reactor"),
    ("M-632", "Sistema de alimentación de sosa", 22_800.0, "aerobic"),
    ("P-608", "Bomba de alimentación a la cuenca de aireación", 84_000.0, "aerobic"),
    ("P-609", "Bomba de retorno de lodo activado", 177_300.0, "aerobic"),
    ("R-609", "Biorreactor de membrana (clarificación)", 5_248_750.0, "aerobic"),
    ("P-602", "Bomba de alimentación", 231_488.0, "common"),
    ("S-611", "Centrífuga de lodos", 6_493_500.0, "sludge"),
    ("P-611", "Bomba de alimentación a centrífuga", 61_200.0, "sludge"),
    ("P-612", "Bomba de centrado", 70_800.0, "sludge"),
    ("C-614", "Tornillo de lodo aerobio", 25_000.0, "sludge"),
    ("M-630", "Sistema de adición de polímero", 9_300.0, "sludge"),
    ("S-610", "Ósmosis inversa", 2_210_979.0, "reverse_osmosis"),
    ("M-640", "Evaporador de salmuera (recompresión mecánica)", 3_801_095.0, "brine_evaporator"),
]
# ---- Aden et al. 2002 (1998 quotes) -------------------------------------------------------------------------------------
ADEN_TOWER = {"cost_usd": 1_659_000.0, "year": 1998, "duty_kcal_h": 147.8e6, "flow_gpm": 71_000.0, "exponent": 0.78,
              "installation": 1.2, "description": "M-902, tiro inducido, fibra de vidrio, 5 celdas, 71,000 gpm"}
ADEN_WWT = {"installed_usd": 3_252_573.0, "year": 1998, "COD_kg_h": 1600.0, "COD_g_L": 16.0,
            "methane_L_per_kg_COD": 350.0, "digester_g_COD_L_day": 12.0,
            "description": "A600: criba, digestor anaerobio, laguna aerobia, clarificador y filtro banda (sin membranas, "
                           "ósmosis ni evaporador)"}
# ---- Loh et al. 2002 (DOE/NETL), Appendix B, first quarter of 1998 ------------------------------------------------------
NETL_TOWER = {"gpm": [150, 300, 600, 1000, 2000, 3000, 4000, 5000, 6000],
              "purchased": [4000, 6500, 11400, 18000, 34400, 50900, 67100, 83200, 99200],
              "installed": [60200, 65000, 70500, 81700, 106100, 134200, 158800, 180400, 211100], "year": 1998,
              "description": "torre ensamblada en fábrica con ventiladores, accionamientos y bacín; base: rango 15 °F, "
                             "aproximación 10 °F, bulbo húmedo 75 °F"}
NETL_BOILER = {"lb_h": [10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000],
               "purchased": [91_700, 148_100, 212_700, 305_700, 439_400, 568_400, 694_000, 816_900],
               "installed": [283_100, 368_900, 468_900, 607_300, 783_600, 920_600, 1_109_100, 1_238_600], "year": 1998,
               "pressure_psig": 250.0,
               "description": "caldera paquete armada en taller, 250 psig y 100 °F de sobrecalentamiento, con ventiladores de "
                              "tiro forzado, instrumentos, quemadores, sopladores de hollín, desaireador, inyección de "
                              "químicos, domos y chimenea"}
# ---- Smith 2005, Table 2.1 (delivered cost, January 2000): C = CB (Q/QB)^M -----------------------------------------------
SMITH = {"package_boiler": {"QB": 50_000.0, "CB": 4.64e5, "range": (50_000.0, 350_000.0), "M": 0.96, "unit": "kg/h de vapor",
                            "description": "caldera paquete (tubos de humo), acero al carbono"},
         "field_erected_boiler": {"QB": 20_000.0, "CB": 3.28e5, "range": (10_000.0, 800_000.0), "M": 0.81,
                                  "unit": "kg/h de vapor", "description": "caldera armada en campo (tubos de agua), acero al "
                                                                          "carbono"},
         "cooling_tower": {"QB": 10.0, "CB": 4.43e3, "range": (10.0, 40.0), "M": 0.63, "unit": "m³/h de agua",
                           "description": "torre de enfriamiento de tiro forzado"}}
# ---- EPA CHP catalog 2015, Table 4-2: back-pressure steam turbine generator sets (2014) ---------------------------------
EPA_TURBINE = {"kW": [500.0, 3000.0, 15000.0], "equipment_usd_kW": [668.0, 401.0, 392.0],
               "installed_usd_kW": [1136.0, 682.0, 666.0], "generator_gearbox_efficiency": [0.94, 0.94, 0.96], "year": 2014,
               "description": "turbina de contrapresión con caja de engranes, generador, control, acoplamientos, sistema de "
                              "aceite y empaque; no incluye caldera ni sistema de vapor"}
# ---- EPA Biomass CHP catalog 2007, Table 5-4: biomass boilers (nominal dollars, year not stated) -------------------------
EPA_BIOMASS = {"stoker": {"lb_h": [20_000, 165_000, 250_000], "equipment": [1_195_000, 7_980_000, 10_790_000],
                          "installed_boiler_system": [1_990_000, 18_000_000, 23_250_000]},
               "fluidized_bed": {"lb_h": [20_000, 175_000, 260_000], "equipment": [6_175_000, 14_490_000, 19_790_000],
                                 "installed_boiler_system": [6_970_000, 24_510_000, 32_250_000]},
               "years": (2003, 2007), "pressure_psig": [275, 750, 750], "band": 0.35,
               "pressure_cost_factor": {"150-250 psig": "1.0", "600-750 psig": "1.15-1.25", "1,250-1,500 psig": "1.5-2.0"}}

FUEL_TYPES = {"natural_gas": "gas natural", "fuel_oil": "combustóleo o diésel", "biomass": "biomasa o bagazo"}
EXTRAPOLATION_FACTOR = 10.0     # a single quote scaled further than this factor from its size gets a warning (module's choice)
CLOSURE_TOL = 1e-8
MERKEL_TOL = 1e-4


# ------------------------------------------------------------------ helpers -----------------------------------------------
def _cp():
    import CoolProp.CoolProp as CP
    return CP


def _ha(out: str, **kw) -> float:
    from CoolProp.HumidAirProp import HAPropsSI
    (k1, v1), (k2, v2), (k3, v3) = kw.items()
    return float(HAPropsSI(out, k1, v1, k2, v2, k3, v3))


def _h_liq(T: float) -> float:
    """Enthalpy of saturated liquid water at T (J/kg), the convention of packages/costing/utilities.py."""
    return float(_cp().PropsSI("H", "T", T, "Q", 0, "Water"))


def _need(value, name: str, what: str):
    if value is None:
        raise ValueError(f"Falta {name}: {what}. Es un dato tuyo; esta herramienta no lo supone.")
    return value


def _pos(value, name: str, what: str) -> float:
    v = float(_need(value, name, what))
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} debe ser un número positivo ({what}); llegó {value!r}.")
    return v


def _frac(value, name: str, what: str, hi_open: bool = True, lo_open: bool = False) -> float:
    v = float(_need(value, name, what))
    bad = (v < 0 or v > 1) or (hi_open and v >= 1) or (lo_open and v <= 0)
    if not math.isfinite(v) or bad:
        raise ValueError(f"{name} es una fracción entre 0 y 1 ({what}); llegó {value!r}.")
    return v


def _index(year: int, cepci: float | None) -> tuple[float, str]:
    """Target CEPCI with the repo convention of packages/costing/sslw.py (table 2015-2023; later years need cepci=)."""
    if cepci is not None:
        if not cepci > 0:
            raise ValueError("cepci debe ser positivo (valor del índice CEPCI del año al que quieres los costos).")
        return float(cepci), f"CEPCI {cepci} (dado por el usuario)"
    from packages.costing.sslw import CEPCI_IDAES
    if year not in CEPCI_IDAES:
        raise ValueError(f"No hay CEPCI para {year} en la tabla del repositorio ({min(CEPCI_IDAES)}-{max(CEPCI_IDAES)}): pasa "
                         "cepci=<valor del índice> para ese año.")
    return float(CEPCI_IDAES[year]), f"CEPCI {CEPCI_IDAES[year]} ({year}, tabla de IDAES)"


def _loglog(x: float, xs: list, ys: list) -> tuple[float, bool]:
    """Log-log interpolation in a published table; outside it, the slope of the end segment. Returns (y, in_range)."""
    inside = xs[0] * (1 - 1e-9) <= x <= xs[-1] * (1 + 1e-9)        # tolerant to unit-conversion round-off at the ends
    i = 0 if x <= xs[0] else len(xs) - 2 if x >= xs[-1] else max(j for j in range(len(xs) - 1) if xs[j] <= x)
    slope = math.log(ys[i + 1] / ys[i]) / math.log(xs[i + 1] / xs[i])
    return ys[i] * (x / xs[i]) ** slope, inside


def _src(key: str) -> str:
    s = SOURCES[key]
    return f"{s['citation']}. {s['url']} (leído {s['read']})"


@contextmanager
def _biosteam_guard():
    """Only for biosteam_cost_items(imported=True). Import BioSTEAM lazily and leave the process as it was: thermo's
    RAISE_PROPERTY_CALCULATION_ERROR flags (thermosteam flips them at import), bst.CE, the electricity price, the main
    flowsheet and thermosteam's thermo setting."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    flags = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    missing = object()
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            import biosteam as bst
            import thermosteam as tmo
        saved = (bst.CE, bst.PowerUtility.price, bst.main_flowsheet.get_flowsheet(),
                 getattr(tmo.settings, "_thermo", missing))
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                yield bst
        finally:
            bst.CE, bst.PowerUtility.price = saved[0], saved[1]
            if bst.main_flowsheet.get_flowsheet() is not saved[2]:
                bst.main_flowsheet.set_flowsheet(saved[2])
            if saved[3] is missing:
                if hasattr(tmo.settings, "_thermo"):
                    del tmo.settings._thermo
            elif getattr(tmo.settings, "_thermo", missing) is not saved[3]:
                tmo.settings._thermo = saved[3]
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flags


BIOSTEAM_COST_CLASSES = {"cooling_tower": ("facilities/_cooling_tower.py", "CoolingTower"),
                         "boiler_turbogenerator": ("facilities/_boiler_turbogenerator.py", "BoilerTurbogenerator"),
                         "wastewater": ("wastewater/conventional.py", "WastewaterSystemCost")}


def _literal_number(node):
    """Value of a numeric literal expression of a decorator argument (7139/1.05, 466183. / 4363., -1, 1375e3)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_number(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        x, y = _literal_number(node.left), _literal_number(node.right)
        return {ast.Add: x + y, ast.Sub: x - y, ast.Mult: x * y, ast.Div: x / y}[type(node.op)]
    raise ValueError("argumento no literal en un decorador @cost de BioSTEAM; esta versión de BioSTEAM no es la esperada")


def _cost_decorators(path: Path, class_name: str) -> dict:
    """The @cost(basis, ID, CE=, cost=, n=, S=, kW=, BM=, units=) decorators of one BioSTEAM class, read from its source."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            units = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(getattr(t, "id", None) == "_units" for t in stmt.targets):
                    units = ast.literal_eval(stmt.value)
            out = {}
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "cost"):
                    continue
                kw = {k.arg: _literal_number(k.value) for k in dec.keywords}
                basis, name = _literal_number(dec.args[0]), (_literal_number(dec.args[1]) if len(dec.args) > 1 else kw.get("ID"))
                out[name] = {"S": float(kw.get("S", 1.0)), "units": kw.get("units") or units.get(basis), "basis": basis,
                             "cost": float(kw["cost"]), "CE": float(kw["CE"]), "n": float(kw["n"]),
                             "kW": float(kw.get("kW") or 0.0), "BM": float(kw.get("BM", 1.0))}
            return out
    raise ValueError(f"BioSTEAM no trae la clase {class_name} en {path.name}; esta versión de BioSTEAM no es la esperada")


@lru_cache(maxsize=2)
def biosteam_cost_items(imported: bool = False) -> dict:
    """The cost correlations of BioSTEAM's facilities: {key: {S, units, basis, cost, CE, n, kW, BM}}, with
    cost = cost·(size/S)^n at index CE and electricity = kW·size/S (BioSTEAM's `cost` decorator).

    Default: read from the SOURCE of the installed BioSTEAM (the literal @cost decorators) without importing it, so no
    global state of BioSTEAM, thermosteam or thermo is touched and the call costs milliseconds. imported=True reads the
    same numbers from the imported classes inside `_biosteam_guard` (slow import; used to check the source reading)."""
    from importlib.metadata import version
    out = {"biosteam_version": version("biosteam")}
    if imported:
        with _biosteam_guard() as bst:
            from biosteam.wastewater import conventional as ww
            classes = {"cooling_tower": bst.facilities.CoolingTower, "boiler_turbogenerator": bst.facilities.BoilerTurbogenerator,
                       "wastewater": ww.WastewaterSystemCost}
            for key, cls in classes.items():
                for name, item in cls.cost_items.items():
                    out[f"{key}/{name}"] = {"S": float(item.S), "units": cls._units.get(item.basis, item.units),
                                            "basis": item.basis, "cost": float(item.cost), "CE": float(item.CE),
                                            "n": float(item.n), "kW": float(item.kW or 0.0),
                                            "BM": float(cls._F_BM_default.get(name, 1.0))}
        return out
    from importlib.util import find_spec
    root = Path(find_spec("biosteam").submodule_search_locations[0])     # locating a top-level package does not import it
    for key, (rel, class_name) in BIOSTEAM_COST_CLASSES.items():
        for name, item in _cost_decorators(root / rel, class_name).items():
            out[f"{key}/{name}"] = item
    return out


def _bst_cost(key: str, size: float, index: float) -> dict:
    """One BioSTEAM cost item evaluated at `size` (in the item's own units) and moved to `index`."""
    it = biosteam_cost_items()[key]
    ratio = size / it["S"]
    return {"purchase_cost_usd": it["cost"] * ratio ** it["n"] * index / it["CE"], "F_BM": it["BM"], "size_ratio": ratio,
            "exponent": it["n"], "reference_cost_usd": it["cost"], "reference_size": it["S"], "reference_units": it["units"],
            "reference_cepci": it["CE"], "electricity_W": 1000.0 * it["kW"] * ratio}


def _item(name: str, purchase: float, f_bm: float, basis: str) -> dict:
    """Shape that capital_investment accepts (tools_economics): C_BM = C_P + C_P,base·(F_BM − 1)."""
    return {"name": name, "F_BM": f_bm, "purchase_cost_usd": purchase, "base_purchase_cost_usd": purchase, "basis": basis,
            "installed_cost_usd": purchase * f_bm}


def _extrapolation(ratio: float, what: str, warnings: list) -> None:
    if ratio > EXTRAPOLATION_FACTOR or ratio < 1 / EXTRAPOLATION_FACTOR:
        warnings.append(f"{what}: el tamaño es {ratio:.3g} veces el de la cotización de referencia; una ley de potencia "
                        f"extrapolada más de {EXTRAPOLATION_FACTOR:g}× es solo orden de magnitud.")


def _deviation(a: float, b: float) -> float:
    return a / b - 1.0


# ------------------------------------------------------------------ cooling tower ------------------------------------------
def _air_state(T: float, P: float, W: float | None = None) -> dict:
    """Humid air per kg of dry air: saturated at T when W is None."""
    w = _ha("W", T=T, P=P, R=1.0) if W is None else W
    return {"T_K": T, "humidity_ratio_kg_kg": w, "enthalpy_J_kg_dry_air": _ha("H", T=T, P=P, W=w),
            "volume_m3_kg_dry_air": _ha("V", T=T, P=P, W=w)}


def tower_air_balance(duty_W: float, cold_T_K: float, hot_T_K: float, air_in: dict, pressure_Pa: float,
                      dry_air_kg_s: float | None = None, air_outlet_T_K: float | None = None) -> dict:
    """Water and energy balance of the tower with saturated outlet air (per unit time):
        water : E = G (W_out − W_in)
        energy: Q + E h_f(T_cold) = G (h_out − h_in)        (Q = heat the circulating water picked up in the plant)
    Give the dry-air flow (solves the outlet temperature) or the outlet temperature (solves the air flow)."""
    from scipy.optimize import brentq
    hf = _h_liq(cold_T_K)
    h_in, w_in = air_in["enthalpy_J_kg_dry_air"], air_in["humidity_ratio_kg_kg"]

    def net(T):        # heat taken by 1 kg of dry air leaving saturated at T, net of the enthalpy of the evaporated water
        s = _air_state(T, pressure_Pa)
        return s["enthalpy_J_kg_dry_air"] - h_in - (s["humidity_ratio_kg_kg"] - w_in) * hf, s

    if air_outlet_T_K is not None:
        q, out = net(air_outlet_T_K)
        if q <= 0:
            raise ValueError("air_outlet_T_K es tan baja que el aire no se lleva calor: debe superar la temperatura de "
                             "saturación adiabática del aire de entrada.")
        G = duty_W / q
    else:
        G = dry_air_kg_s
        lo, hi = 273.16, hot_T_K
        f = lambda T: G * net(T)[0] - duty_W   # noqa: E731
        if f(hi) < 0:
            q_max = net(hot_T_K)[0]
            raise ValueError("El aire no alcanza: saldría más caliente que el agua que entra a la torre. El flujo mínimo de "
                             f"aire seco para esta carga es {duty_W / q_max:.4g} kg/s (aire saliendo saturado a la "
                             "temperatura del agua caliente); baja liquid_to_gas_ratio.")
        if f(lo) > 0:
            raise ValueError("El balance de aire no tiene solución sobre 0 °C: revisa la carga y el flujo de aire.")
        T = brentq(f, lo, hi, xtol=1e-9, rtol=1e-12)
        q, out = net(T)
    E = G * (out["humidity_ratio_kg_kg"] - w_in)
    closure = abs(duty_W + E * hf - G * (out["enthalpy_J_kg_dry_air"] - h_in)) / duty_W
    return {"dry_air_kg_s": G, "evaporation_kg_s": E, "outlet": out, "energy_closure": closure}


def merkel_number(cold_T_K: float, hot_T_K: float, air_in_enthalpy_J_kg: float, liquid_to_gas_ratio: float,
                  pressure_Pa: float) -> dict:
    """Merkel's tower characteristic KaV/L = ∫ dh_f / (h_s(T_w) − h_a) for counterflow, with the air operating line
    h_a = h_in + (L/G)(h_f(T_w) − h_f(T_cold)). Adaptive quadrature checked against composite Simpson (200 intervals)."""
    from scipy.integrate import quad
    hf_c = _h_liq(cold_T_K)
    cp = _cp()

    def driving(T):
        return _ha("H", T=T, P=pressure_Pa, R=1.0) - (air_in_enthalpy_J_kg + liquid_to_gas_ratio * (_h_liq(T) - hf_c))

    def integrand(T):
        return float(cp.PropsSI("C", "T", T, "Q", 0, "Water")) / driving(T)

    n = 200
    Ts = [cold_T_K + (hot_T_K - cold_T_K) * i / n for i in range(n + 1)]
    d = [driving(T) for T in Ts]
    if min(d) <= 0:
        return {"feasible": False, "minimum_driving_force_J_kg": min(d)}
    val, _ = quad(integrand, cold_T_K, hot_T_K, epsrel=1e-10, limit=200)
    y = [float(cp.PropsSI("C", "T", T, "Q", 0, "Water")) / di for T, di in zip(Ts, d)]
    simpson = (hot_T_K - cold_T_K) / n / 3 * (y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]))
    return {"feasible": True, "merkel_number": val, "simpson_rel_diff": abs(val - simpson) / val,
            "minimum_driving_force_J_kg": min(d)}


def _tower_capital(circulation_kg_s: float, circulation_m3_s: float, duty_W: float, index: float, index_basis: str,
                   warnings: list) -> dict:
    gpm = circulation_m3_s / US_GAL_M3 * 60.0
    m3_h = circulation_m3_s * 3600.0
    kg_h = circulation_kg_s * 3600.0
    est = {}
    b = _bst_cost("cooling_tower/Cooling tower", kg_h / M_H2O, index)
    est["biosteam"] = {"purchase_cost_usd": b["purchase_cost_usd"], "installed_cost_usd": b["purchase_cost_usd"] * b["F_BM"],
                       "F_BM": b["F_BM"], "in_range": 1 / EXTRAPOLATION_FACTOR <= b["size_ratio"] <= EXTRAPOLATION_FACTOR,
                       "size_ratio_to_reference": b["size_ratio"],
                       "basis": f"BioSTEAM {biosteam_cost_items()['biosteam_version']} CoolingTower: {b['reference_cost_usd']:,.0f}"
                                f" USD (CE {b['reference_cepci']:g}) a {b['reference_size']:,.0f} kmol/h de agua, exponente "
                                f"{b['exponent']}, F_BM {b['F_BM']} (de BioSTEAM). Es la cotización Marley SPX de Humbird 2011."}
    ratio_h = kg_h / HUMBIRD_TOWER["flow_kg_h"]
    c = HUMBIRD_TOWER["cost_usd"] * ratio_h ** HUMBIRD_TOWER["exponent"] * index / CEPCI_REF[2010]
    est["humbird2011_table"] = {"purchase_cost_usd": c, "installed_cost_usd": c * HUMBIRD_TOWER["installation"],
                                "F_BM": HUMBIRD_TOWER["installation"],
                                "in_range": 1 / EXTRAPOLATION_FACTOR <= ratio_h <= EXTRAPOLATION_FACTOR,
                                "size_ratio_to_reference": ratio_h,
                                "basis": f"{HUMBIRD_TOWER['description']}: 1,375,000 USD de 2010 a 10,037,820 kg/h, exponente "
                                         "0.60 e instalación 1.5 como los imprime la lista de equipo del informe (BioSTEAM usa "
                                         f"0.7). {_src('humbird2011')}"}
    ratio_a = duty_W / (ADEN_TOWER["duty_kcal_h"] * KCAL_J / 3600.0)
    c = ADEN_TOWER["cost_usd"] * ratio_a ** ADEN_TOWER["exponent"] * index / CEPCI_REF[1998]
    est["aden2002"] = {"purchase_cost_usd": c, "installed_cost_usd": c * ADEN_TOWER["installation"],
                       "F_BM": ADEN_TOWER["installation"],
                       "in_range": 1 / EXTRAPOLATION_FACTOR <= ratio_a <= EXTRAPOLATION_FACTOR, "size_ratio_to_reference": ratio_a,
                       "basis": f"{ADEN_TOWER['description']}: 1,659,000 USD de 1998 a 147.8 Gcal/h, escalado por carga térmica "
                                f"con exponente 0.78, instalación 1.2. {_src('aden2002')}"}
    p, inside = _loglog(gpm, NETL_TOWER["gpm"], NETL_TOWER["purchased"])
    inst, _ = _loglog(gpm, NETL_TOWER["gpm"], NETL_TOWER["installed"])
    f = index / CEPCI_REF[1998]
    est["netl2002"] = {"purchase_cost_usd": p * f, "installed_cost_usd": inst * f, "F_BM": inst / p, "in_range": inside,
                       "basis": f"tabla de {NETL_TOWER['description']}; 150-6,000 gpm, interpolación log-log; el instalado es el "
                                f"de la tabla (F_BM = instalado/comprado de la fuente). {_src('netl2002')}"}
    s = SMITH["cooling_tower"]
    c = s["CB"] * (m3_h / s["QB"]) ** s["M"] * index / CEPCI_SMITH_JAN_2000
    est["smith2005"] = {"purchase_cost_usd": c, "installed_cost_usd": None, "F_BM": None,
                        "in_range": s["range"][0] <= m3_h <= s["range"][1],
                        "basis": f"{s['description']}: 4.43e3 USD (10 m³/h)·(Q/10)^0.63, válido 10-40 m³/h. {_src('smith2005')}"}
    if est["netl2002"]["in_range"]:
        chosen, why = "netl2002", "la circulación cae dentro de la tabla publicada de 150-6,000 gpm (torre de fábrica)"
    elif gpm > NETL_TOWER["gpm"][-1]:
        chosen, why = "biosteam", ("la circulación supera la tabla de torres de fábrica (6,000 gpm): torre armada en campo, "
                                   "cotización de Humbird 2011 escalada por BioSTEAM")
        _extrapolation(est["biosteam"]["size_ratio_to_reference"], "Torre de enfriamiento (BioSTEAM/Humbird)", warnings)
    elif est["smith2005"]["in_range"]:
        chosen, why = "smith2005", "la circulación cae en el intervalo de la correlación de Smith (10-40 m³/h)"
    else:
        chosen, why = "netl2002", "ninguna fuente cubre este tamaño; se extrapola la tabla de torres de fábrica"
        warnings.append(f"Torre de enfriamiento: {gpm:.3g} gpm queda por debajo de toda fuente publicada leída (150 gpm o "
                        "10 m³/h); el costo es una extrapolación.")
    sel = est[chosen]
    f_bm = sel["F_BM"] if sel["F_BM"] is not None else est["biosteam"]["F_BM"]
    f_bm_note = "" if sel["F_BM"] is not None else " F_BM 1.5 de BioSTEAM (la fuente del costo no da instalado)."
    items = [_item("Torre de enfriamiento", sel["purchase_cost_usd"], f_bm, sel["basis"] + f_bm_note + f" Índice: {index_basis}.")]
    bp = _bst_cost("cooling_tower/Cooling water pump", kg_h / M_H2O, index)
    items.append(_item("Bombas de agua de enfriamiento", bp["purchase_cost_usd"], bp["F_BM"],
                       f"BioSTEAM CoolingTower: {bp['reference_cost_usd']:,.0f} USD (CE {bp['reference_cepci']:g}) a "
                       f"{bp['reference_size']:,.0f} kmol/h, exponente {bp['exponent']}, F_BM {bp['F_BM']} (de BioSTEAM); son "
                       f"las tres bombas P-902 de Humbird 2011 (16,120 gpm y 100 ft cada una). Índice: {index_basis}."))
    _extrapolation(bp["size_ratio"], "Bombas de agua de enfriamiento (BioSTEAM/Humbird)", warnings)
    dev = {k: _deviation(v["purchase_cost_usd"], sel["purchase_cost_usd"]) for k, v in est.items() if k != chosen}
    dev_inst = {k: _deviation(v["installed_cost_usd"], items[0]["installed_cost_usd"]) for k, v in est.items()
                if k != chosen and v["installed_cost_usd"]}
    return {"items": items, "purchase_cost_usd": sum(i["purchase_cost_usd"] for i in items),
            "installed_cost_usd": sum(i["installed_cost_usd"] for i in items), "selected_tower_estimate": chosen,
            "selection_reason": why, "tower_estimates": est, "tower_purchase_deviation_vs_selected": dev,
            "tower_installed_deviation_vs_selected": dev_inst, "circulation_gpm": gpm, "cepci": index, "cepci_basis": index_basis}


def cooling_tower_design(duty_W: float, supply_T_K: float, return_T_K: float, wet_bulb_T_K: float,
                         cycles_of_concentration: float, dry_bulb_T_K: float | None = None,
                         liquid_to_gas_ratio: float | None = None, air_outlet_T_K: float | None = None,
                         drift_fraction: float | None = None, pressure_Pa: float | None = None,
                         pump_head_m: float | None = None, pump_efficiency: float | None = None,
                         fan_static_pressure_Pa: float | None = None, fan_efficiency: float | None = None,
                         year: int = 2023, cepci: float | None = None) -> dict:
    """Mechanical-draft cooling tower from the duty and the water and wet-bulb temperatures; see the module docstring and
    tools_utilities_capital.cooling_tower_design."""
    Q = _pos(duty_W, "duty_W", "calor que la torre debe rechazar")
    Tc = _pos(supply_T_K, "supply_T_K", "temperatura del agua fría que sale de la torre hacia la planta")
    Th = _pos(return_T_K, "return_T_K", "temperatura del agua caliente que regresa a la torre")
    Twb = _pos(wet_bulb_T_K, "wet_bulb_T_K", "temperatura de bulbo húmedo de diseño del sitio")
    C = _pos(cycles_of_concentration, "cycles_of_concentration", "ciclos de concentración = repuesto ÷ (purga + arrastre)")
    if not 273.16 < Tc < Th < 373.0:
        raise ValueError("Se requiere 273.16 K < supply_T_K < return_T_K < 373 K.")
    if Twb >= Tc:
        raise ValueError(f"El agua no puede enfriarse hasta o por debajo del bulbo húmedo: supply_T_K ({Tc} K) debe ser mayor "
                         f"que wet_bulb_T_K ({Twb} K).")
    if C <= 1:
        raise ValueError("cycles_of_concentration debe ser > 1.")
    if liquid_to_gas_ratio is not None and air_outlet_T_K is not None:
        raise ValueError("Da liquid_to_gas_ratio o air_outlet_T_K, no los dos.")
    if (pump_head_m is None) != (pump_efficiency is None):
        raise ValueError("Para la potencia de bombeo por física da pump_head_m y pump_efficiency juntos.")
    if (fan_static_pressure_Pa is None) != (fan_efficiency is None):
        raise ValueError("Para la potencia del ventilador por física da fan_static_pressure_Pa y fan_efficiency juntos.")
    warnings, assumptions, suggested = [], [], []
    index, index_basis = _index(year, cepci)
    if pressure_Pa is None:
        P = P_ATM
        assumptions.append("Presión atmosférica no dada: 101 325 Pa (nivel del mar). En sitios altos pasa pressure_Pa: cambia "
                           "la humedad de saturación y la evaporación.")
    else:
        P = _pos(pressure_Pa, "pressure_Pa", "presión atmosférica del sitio")
    if dry_bulb_T_K is None:
        air_in = _air_state(Twb, P)
        assumptions.append("Bulbo seco no dado: el aire entra saturado a la temperatura de bulbo húmedo. Con aire más seco la "
                           "evaporación real es mayor (el aire además se enfría); pasa dry_bulb_T_K para contarlo.")
    else:
        if dry_bulb_T_K < Twb:
            raise ValueError("dry_bulb_T_K no puede ser menor que wet_bulb_T_K.")
        air_in = _air_state(float(dry_bulb_T_K), P, _ha("W", T=float(dry_bulb_T_K), P=P, B=Twb))
    cp = _cp()
    rho = float(cp.PropsSI("D", "T", Tc, "Q", 0, "Water"))
    m_w = Q / (_h_liq(Th) - _h_liq(Tc))
    h_fg = float(cp.PropsSI("H", "T", Tc, "Q", 1, "Water")) - _h_liq(Tc)
    all_latent = Q / h_fg
    limit = tower_air_balance(Q, Tc, Th, air_in, P, air_outlet_T_K=Th)          # minimum air: outlet saturated at T_hot
    lg_max = m_w / limit["dry_air_kg_s"]
    merkel = None
    if liquid_to_gas_ratio is not None:
        lg = _pos(liquid_to_gas_ratio, "liquid_to_gas_ratio", "kg de agua circulante por kg de aire seco")
        bal = tower_air_balance(Q, Tc, Th, air_in, P, dry_air_kg_s=m_w / lg)
        basis = "balance psicrométrico con liquid_to_gas_ratio del usuario"
    elif air_outlet_T_K is not None:
        if not Twb < air_outlet_T_K <= Th:
            raise ValueError("air_outlet_T_K debe quedar entre wet_bulb_T_K y return_T_K (contracorriente).")
        bal = tower_air_balance(Q, Tc, Th, air_in, P, air_outlet_T_K=float(air_outlet_T_K))
        basis = "balance psicrométrico con air_outlet_T_K del usuario"
    else:
        bal = limit
        basis = "balance psicrométrico en el límite de aire mínimo (aire saliendo saturado a la temperatura del agua caliente)"
        assumptions.append("No se dio liquid_to_gas_ratio ni air_outlet_T_K: la evaporación se evalúa en el límite de aire "
                           "mínimo (torre ideal de área infinita). El flujo de aire que se informa es el mínimo "
                           "termodinámico, no el de diseño.")
    air_known = liquid_to_gas_ratio is not None or air_outlet_T_K is not None
    G, E = bal["dry_air_kg_s"], bal["evaporation_kg_s"]
    lg = m_w / G
    if air_known:
        merkel = merkel_number(Tc, Th, air_in["enthalpy_J_kg_dry_air"], lg, P)
        if not merkel["feasible"]:
            warnings.append("La línea de operación del aire cruza la curva de saturación (fuerza impulsora ≤ 0 en algún punto): "
                            "con este L/G la torre es imposible en contracorriente; aumenta el aire.")
    if drift_fraction is None:
        drift_f = HUMBIRD_CW["drift_fraction_of_circulation"]
        suggested.append({"input": "drift_fraction", "value": drift_f, "unit": "fracción de la circulación",
                          "source": _src("humbird2011"), "note": "«windage would be 0.005% of the total flow to the tower»"})
    else:
        drift_f = _frac(drift_fraction, "drift_fraction", "arrastre de gotas como fracción de la circulación")
    drift = drift_f * m_w
    blowdown = E / (C - 1.0) - drift
    achieved = C
    if blowdown < 0:
        blowdown = 0.0
        achieved = (E + drift) / drift
        warnings.append(f"El arrastre solo ya purga más de lo que piden {C:g} ciclos: purga = 0 y los ciclos reales son "
                        f"{achieved:.3g}.")
    makeup = E + drift + blowdown
    # --- power
    if pump_head_m is not None:
        H = _pos(pump_head_m, "pump_head_m", "carga total de las bombas de circulación")
        eta_p = _frac(pump_efficiency, "pump_efficiency", "eficiencia bomba + motor", hi_open=False, lo_open=True)
        pump_W, pump_basis = m_w * G_STD * H / eta_p, "física: ṁ·g·H/η con pump_head_m y pump_efficiency del usuario"
    else:
        pump_W = 1000.0 * HUMBIRD_CW["pump_kW"] * m_w * 3600.0 / HUMBIRD_CW["circulation_kg_h"]
        pump_basis = "referencia publicada escalada linealmente con la circulación"
        suggested.append({"input": "potencia de bombeo", "value": HUMBIRD_CW["pump_kW"] / HUMBIRD_CW["circulation_kg_h"] * 1000,
                          "unit": "kW por t/h de circulación", "source": _src("humbird2011"),
                          "note": "corriente de trabajo WP902 = 1598 kW para 11,923,904 kg/h (bombas de 100 ft de carga); da "
                                  "pump_head_m y pump_efficiency para calcularla"})
    if fan_static_pressure_Pa is not None and air_known:
        dp = _pos(fan_static_pressure_Pa, "fan_static_pressure_Pa", "presión estática del ventilador")
        eta_f = _frac(fan_efficiency, "fan_efficiency", "eficiencia ventilador + motor", hi_open=False, lo_open=True)
        fan_W = G * bal["outlet"]["volume_m3_kg_dry_air"] * dp / eta_f
        fan_basis = "física: caudal de aire de salida × presión estática ÷ eficiencia (tiro inducido)"
    else:
        if fan_static_pressure_Pa is not None:
            warnings.append("fan_static_pressure_Pa se ignoró: sin liquid_to_gas_ratio o air_outlet_T_K no hay flujo de aire de "
                            "diseño; la potencia del ventilador es la de referencia.")
        fan_W = 1000.0 * HUMBIRD_CW["fan_kW"] * m_w * 3600.0 / HUMBIRD_CW["circulation_kg_h"]
        fan_basis = "referencia publicada escalada linealmente con la circulación"
        suggested.append({"input": "potencia del ventilador", "value": HUMBIRD_CW["fan_kW"] / HUMBIRD_CW["circulation_kg_h"] * 1000,
                          "unit": "kW por t/h de circulación", "source": _src("humbird2011"),
                          "note": "corriente de trabajo WM902 = 1021 kW para 11,923,904 kg/h; da liquid_to_gas_ratio, "
                                  "fan_static_pressure_Pa y fan_efficiency para calcularla"})
    capital = _tower_capital(m_w, m_w / rho, Q, index, index_basis, warnings)
    bst_power = (_bst_cost("cooling_tower/Cooling tower", m_w * 3600 / M_H2O, index)["electricity_W"]
                 + _bst_cost("cooling_tower/Cooling water pump", m_w * 3600 / M_H2O, index)["electricity_W"])
    for s in suggested:
        assumptions.append(f"SUGERIDO (puedes sustituirlo): {s['input']} = {s['value']:.4g} {s['unit']}. Fuente: {s['source']}.")
    assumptions += ["El aire sale saturado (hipótesis de Merkel).",
                    "El repuesto entra a la temperatura del agua fría y el arrastre no entra al balance de energía.",
                    "La carga es el calor que el agua circulante recoge en la planta; el calor de las bombas no se suma."]
    reasons = []
    if bal["energy_closure"] > CLOSURE_TOL:
        reasons.append("el balance de energía agua-aire no cierra")
    water_closure = abs(makeup - E - drift - blowdown) / makeup
    if merkel is not None and (not merkel["feasible"] or merkel["simpson_rel_diff"] > MERKEL_TOL):
        reasons.append("la torre no es factible con ese aire o la integral de Merkel no converge")
    if E > 1.5 * all_latent:
        warnings.append("La evaporación supera 1.5 veces la carga ÷ calor latente: el aire de entrada es muy seco y la torre "
                        "evapora sobre todo para enfriar el aire; revisa el flujo de aire.")
    return {
        "circulation_kg_s": m_w, "circulation_m3_s": m_w / rho, "range_K": Th - Tc, "approach_K": Tc - Twb,
        "evaporation_kg_s": E, "drift_kg_s": drift, "blowdown_kg_s": blowdown, "makeup_kg_s": makeup,
        "cycles_of_concentration": achieved, "evaporation_fraction_of_circulation": E / m_w,
        "evaporation_latent_fraction_of_duty": E * h_fg / Q, "evaporation_basis": basis,
        "evaporation_bounds_kg_s": {"minimum_air_limit": limit["evaporation_kg_s"], "all_duty_as_latent_heat": all_latent},
        "air": {"design_flow_known": air_known, "dry_air_kg_s": G, "liquid_to_gas_ratio": lg,
                "maximum_liquid_to_gas_ratio": lg_max, "inlet": air_in, "outlet": bal["outlet"], "pressure_Pa": P,
                "merkel_number": merkel["merkel_number"] if merkel and merkel["feasible"] else None},
        "fan_power_W": fan_W, "fan_power_basis": fan_basis, "pump_power_W": pump_W, "pump_power_basis": pump_basis,
        "power_cross_check": {"biosteam_fan_plus_pump_W": bst_power, "this_result_W": fan_W + pump_W,
                              "note": "BioSTEAM asigna 1598 kW a la torre y 1021 kW a la bomba (al revés que las corrientes "
                                      "WM902 y WP902 de Humbird 2011) y los escala con los flujos de la cotización, no con "
                                      "los 11,923,904 kg/h del balance: da ~15 % más que el informe."},
        "capital": capital, "cost_year": year, "cepci": index, "suggested_inputs": suggested,
        "warnings": warnings, "assumptions": assumptions,
        "not_included": ["dimensionamiento del relleno, número de celdas y huella (se entrega el número de Merkel requerido)",
                         "tratamiento químico del agua y su costo", "bacín de concreto, tubería de distribución y cimentación "
                         "fuera de lo que ya trae el costo instalado de la fuente", "operación en invierno, penacho y ruido",
                         "costo de operación (está en utility_costs, bloque cooling_water)"],
        "method": {"name": "balance de agua y energía agua-aire húmedo con aire de salida saturado; número de Merkel por "
                           "cuadratura; capital por tablas y cotizaciones publicadas escaladas",
                   "libraries": "CoolProp (IAPWS-95 y HumidAirProp), SciPy (brentq, quad), BioSTEAM (solo sus correlaciones)",
                   "module": "packages/costing/utilities_capital.py", "sources": [_src(k) for k in
                                                                                  ("humbird2011", "aden2002", "netl2002",
                                                                                   "smith2005")]},
        "evidence": {"verified": not reasons, "reasons": reasons, "energy_balance_closure": bal["energy_closure"],
                     "water_balance_closure": water_closure,
                     "merkel_quadrature_vs_simpson": merkel["simpson_rel_diff"] if merkel and merkel["feasible"] else None,
                     "evaporation_below_all_latent_bound": E <= all_latent,
                     "note": "verified certifica los balances de esta corrida; el capital es un estimado de estudio y su "
                             "dispersión entre fuentes está en capital.tower_purchase_deviation_vs_selected"},
    }


# ------------------------------------------------------------------ boiler and turbogenerator ------------------------------
def _steam(P: float, T: float | None = None, h: float | None = None, s: float | None = None) -> dict:
    """State of water/steam at P and one of T, h, s (IAPWS-95). T equal to saturation means dry saturated vapour."""
    cp = _cp()
    if not 611.7 < P < 22.0e6:
        raise ValueError(f"Presión de vapor fuera del intervalo subcrítico que cubre esta herramienta (612 Pa a 22 MPa): {P!r}.")
    Tsat = float(cp.PropsSI("T", "P", P, "Q", 1, "Water"))
    if T is not None:
        if T < Tsat - 1e-6:
            raise ValueError(f"A {P:.6g} Pa el vapor satura a {Tsat:.2f} K; {T:.2f} K sería líquido.")
        if T - Tsat < 1e-6:
            h = float(cp.PropsSI("H", "P", P, "Q", 1, "Water"))
        else:
            h = float(cp.PropsSI("H", "P", P, "T", T, "Water"))
    elif s is not None:
        h = float(cp.PropsSI("H", "P", P, "S", s, "Water"))
    hl = float(cp.PropsSI("H", "P", P, "Q", 0, "Water"))
    hv = float(cp.PropsSI("H", "P", P, "Q", 1, "Water"))
    x = (h - hl) / (hv - hl)
    return {"pressure_Pa": P, "T_K": float(cp.PropsSI("T", "P", P, "H", h, "Water")), "enthalpy_J_kg": h,
            "entropy_J_kgK": float(cp.PropsSI("S", "P", P, "H", h, "Water")), "saturation_T_K": Tsat,
            "quality": min(max(x, 0.0), 1.0) if x <= 1.0 else None,
            "superheat_K": max(float(cp.PropsSI("T", "P", P, "H", h, "Water")) - Tsat, 0.0) if x > 1.0 else 0.0}


def turbine_expansion(inlet_pressure_Pa: float, inlet_T_K: float, isentropic_efficiency: float,
                      sections: list[dict]) -> dict:
    """Steam turbine as consecutive sections; each expands from the actual outlet of the previous one to its
    outlet_pressure_Pa with the same isentropic efficiency, and carries flow_kg_s. Returns states and shaft power."""
    eta = _frac(isentropic_efficiency, "isentropic_efficiency", "eficiencia isentrópica de la turbina", hi_open=False,
                lo_open=True)
    state = _steam(inlet_pressure_Pa, T=inlet_T_K)
    inlet, out, shaft, ideal = state, [], 0.0, 0.0
    for sec in sections:
        P2 = sec["outlet_pressure_Pa"]
        if P2 >= state["pressure_Pa"]:
            raise ValueError("Cada nivel de vapor que sale de la turbina debe estar a menor presión que el anterior y que la "
                             "entrada.")
        h_s = _steam(P2, s=state["entropy_J_kgK"])["enthalpy_J_kg"]
        h2 = state["enthalpy_J_kg"] - eta * (state["enthalpy_J_kg"] - h_s)
        new = _steam(P2, h=h2)
        w = sec["flow_kg_s"] * (state["enthalpy_J_kg"] - h2)
        shaft += w
        ideal += sec["flow_kg_s"] * (state["enthalpy_J_kg"] - h_s)
        out.append({"outlet": new, "flow_kg_s": sec["flow_kg_s"], "isentropic_outlet_enthalpy_J_kg": h_s, "shaft_power_W": w,
                    "entropy_generated_J_kgK": new["entropy_J_kgK"] - state["entropy_J_kgK"]})
        state = new
    return {"inlet": inlet, "sections": out, "shaft_power_W": shaft, "isentropic_power_W": ideal}


def biomass_heating_values(HHV_dry_J_kg: float, moisture_mass_fraction: float,
                           hydrogen_dry_mass_fraction: float | None = None) -> dict:
    """As-fired heating values of a wet solid fuel from its dry-basis HHV: HHV_af = HHV_dry (1 − w);
    LHV_af = HHV_af − h_fg(25 °C) [w + (M_H2O/M_H2) H_dry (1 − w)] (needs the hydrogen content)."""
    w = _frac(moisture_mass_fraction, "moisture_mass_fraction", "humedad del combustible como se quema, base húmeda")
    hhv = _pos(HHV_dry_J_kg, "HHV_dry_J_kg", "poder calorífico superior en base seca") * (1 - w)
    out = {"HHV_J_kg": hhv, "moisture_mass_fraction": w}
    if hydrogen_dry_mass_fraction is not None:
        H = _frac(hydrogen_dry_mass_fraction, "hydrogen_dry_mass_fraction", "hidrógeno elemental en base seca")
        cp = _cp()
        hfg = float(cp.PropsSI("H", "T", 298.15, "Q", 1, "Water") - cp.PropsSI("H", "T", 298.15, "Q", 0, "Water"))
        out.update({"LHV_J_kg": hhv - hfg * (w + M_H2O / M_H2 * H * (1 - w)), "latent_heat_298K_J_kg": hfg})
    return out


def _fuel_heating_values(fuel: dict) -> dict:
    kind = fuel.get("type")
    if kind not in FUEL_TYPES:
        raise ValueError(f"Falta fuel['type']: {sorted(FUEL_TYPES)} ({'; '.join(f'{k} = {v}' for k, v in FUEL_TYPES.items())}). "
                         "Decide además qué correlación de capital aplica.")
    if "HHV_dry_J_kg" in fuel:
        if "moisture_mass_fraction" not in fuel:
            raise ValueError("Con fuel['HHV_dry_J_kg'] falta fuel['moisture_mass_fraction'] (humedad como se quema, base "
                             "húmeda): es un dato de TU combustible.")
        hv = biomass_heating_values(fuel["HHV_dry_J_kg"], fuel["moisture_mass_fraction"], fuel.get("hydrogen_dry_mass_fraction"))
        hv["basis"] = "calculado de HHV base seca y humedad del usuario (como se quema)"
        return hv
    hv = {k: float(fuel[k]) for k in ("HHV_J_kg", "LHV_J_kg") if fuel.get(k) is not None}
    if not hv:
        raise ValueError("Falta el poder calorífico de TU combustible como se quema: fuel['LHV_J_kg'] o fuel['HHV_J_kg'] (o, "
                         "para biomasa, 'HHV_dry_J_kg' + 'moisture_mass_fraction' [+ 'hydrogen_dry_mass_fraction' para el "
                         "LHV]). Alternativa: el bloque combustion, que lo calcula de la composición.")
    if any(v <= 0 for v in hv.values()):
        raise ValueError("El poder calorífico debe ser positivo.")
    hv["basis"] = "dato del usuario (como se quema)"
    return hv


def _boiler_capital(fuel_type: str, steam_kg_s: float, boiler_pressure_Pa: float, index: float, index_basis: str,
                    warnings: list) -> dict:
    kg_h = steam_kg_s * 3600.0
    lb_h = kg_h / LB_KG
    psig = (boiler_pressure_Pa - P_ATM) / 6894.757293168
    est = {}
    parts = {k: _bst_cost(f"boiler_turbogenerator/{k}", kg_h, index)
             for k in ("Boiler", "Deaerator", "Hot process water softener system", "Amine addition pkg")}
    b = parts["Boiler"]
    est["biosteam"] = {"purchase_cost_usd": sum(p["purchase_cost_usd"] for p in parts.values()),
                       "installed_cost_usd": sum(p["purchase_cost_usd"] * p["F_BM"] for p in parts.values()),
                       "in_range": 1 / EXTRAPOLATION_FACTOR <= b["size_ratio"] <= EXTRAPOLATION_FACTOR,
                       "size_ratio_to_reference": b["size_ratio"],
                       "basis": f"BioSTEAM {biosteam_cost_items()['biosteam_version']} BoilerTurbogenerator (caldera, desaireador, "
                                "suavizador y paquete de amina): caldera 28,550,000 USD (CE 551) a 238,686 kg/h, exponente "
                                "0.6, F_BM 1.8 (de BioSTEAM). Es la cotización Babcock & Wilcox de Humbird 2011 para una "
                                "caldera de BIOMASA de 900 psig y 454 °C con precalentador de aire, secador por aspersión para "
                                "desulfurar y casa de bolsas."}
    p, inside = _loglog(lb_h, NETL_BOILER["lb_h"], NETL_BOILER["purchased"])
    inst, _ = _loglog(lb_h, NETL_BOILER["lb_h"], NETL_BOILER["installed"])
    f = index / CEPCI_REF[1998]
    est["netl2002_package"] = {"purchase_cost_usd": p * f, "installed_cost_usd": inst * f, "F_BM": inst / p, "in_range": inside,
                               "basis": f"tabla de {NETL_BOILER['description']}; 10,000-300,000 lb/h, interpolación log-log; el "
                                        f"instalado es el de la tabla. {_src('netl2002')}"}
    for key in ("package_boiler", "field_erected_boiler"):
        s = SMITH[key]
        est[f"smith2005_{key}"] = {"purchase_cost_usd": s["CB"] * (kg_h / s["QB"]) ** s["M"] * index / CEPCI_SMITH_JAN_2000,
                                   "installed_cost_usd": None, "in_range": s["range"][0] <= kg_h <= s["range"][1],
                                   "basis": f"{s['description']}: {s['CB']:.3g} USD ({s['QB']:,.0f} kg/h)·(Q/QB)^{s['M']}, válido "
                                            f"{s['range'][0]:,.0f}-{s['range'][1]:,.0f} kg/h, sin corrección por presión. "
                                            f"{_src('smith2005')}"}
    for key in ("stoker", "fluidized_bed"):
        t = EPA_BIOMASS[key]
        eq, inside = _loglog(lb_h, t["lb_h"], t["equipment"])
        inst, _ = _loglog(lb_h, t["lb_h"], t["installed_boiler_system"])
        lo, hi = (index / CEPCI_REF[y] for y in (EPA_BIOMASS["years"][1], EPA_BIOMASS["years"][0]))
        est[f"epa2007_biomass_{key}"] = {
            "purchase_cost_usd": eq * 0.5 * (lo + hi), "installed_cost_usd": inst * 0.5 * (lo + hi), "in_range": inside,
            "purchase_cost_range_usd": [eq * lo, eq * hi], "installed_cost_range_usd": [inst * lo, inst * hi],
            "basis": f"caldera de biomasa {'de parrilla (stoker)' if key == 'stoker' else 'de lecho fluidizado'}: equipo y "
                     "«sistema de caldera instalado» (sin patio de combustible), 275 psig saturado en el tamaño chico y 750 "
                     "psig en los grandes; la fuente no declara el año de los dólares: el intervalo supone 2003 o 2007 y el "
                     f"valor central es su media; incertidumbre declarada ±35 %. {_src('epa_biomass2007')}"}
    if fuel_type == "biomass":
        chosen, why = "biosteam", ("combustible sólido: caldera de biomasa con limpieza de gases (cotización de Humbird 2011 "
                                   "escalada por BioSTEAM); contraste independiente: tabla de la EPA")
        items = [_item({"Boiler": "Caldera de biomasa (con precalentador de aire, desulfuración y casa de bolsas)",
                        "Deaerator": "Desaireador", "Hot process water softener system": "Suavizador de agua de proceso",
                        "Amine addition pkg": "Paquete de adición de amina"}[k], p["purchase_cost_usd"], p["F_BM"],
                       f"BioSTEAM BoilerTurbogenerator «{k}»: {p['reference_cost_usd']:,.0f} USD (CE {p['reference_cepci']:g}) a "
                       f"{p['reference_size']:,.0f} kg/h de vapor, exponente {p['exponent']}, F_BM {p['F_BM']} (de BioSTEAM; "
                       f"Humbird 2011). Índice: {index_basis}.") for k, p in parts.items()]
        _extrapolation(b["size_ratio"], "Caldera de biomasa (BioSTEAM/Humbird)", warnings)
        compare = ("epa2007_biomass_stoker", "epa2007_biomass_fluidized_bed")
    else:
        if est["netl2002_package"]["in_range"] or not est["smith2005_field_erected_boiler"]["in_range"]:
            chosen, why = "netl2002_package", "caldera paquete de gas o líquido: tabla publicada de 10,000-300,000 lb/h"
            if not est["netl2002_package"]["in_range"]:
                warnings.append(f"Caldera: {lb_h:,.0f} lb/h queda fuera de la tabla publicada (10,000-300,000 lb/h); el costo es "
                                "una extrapolación.")
            sel = est[chosen]
            items = [_item("Caldera paquete (con desaireador, ventiladores y chimenea)", sel["purchase_cost_usd"], sel["F_BM"],
                           sel["basis"] + f" F_BM = instalado/comprado de la fuente. Índice: {index_basis}.")]
        else:
            chosen, why = "smith2005_field_erected_boiler", ("supera la tabla de calderas paquete (300,000 lb/h): caldera de "
                                                             "tubos de agua armada en campo")
            sel = est[chosen]
            items = [_item("Caldera armada en campo (tubos de agua)", sel["purchase_cost_usd"], parts["Boiler"]["F_BM"],
                           sel["basis"] + " F_BM 1.8 de BioSTEAM para caldera (la fuente del costo no da instalado). Índice: "
                           f"{index_basis}.")]
        if psig > NETL_BOILER["pressure_psig"] * 1.05:
            warnings.append(f"La caldera opera a {psig:,.0f} psig y el costo publicado es para 250 psig. Factores de costo por "
                            f"presión publicados (EPA 2007, Tabla 5-5): {EPA_BIOMASS['pressure_cost_factor']}. No se aplicó "
                            "ninguno: multiplícalo tú si procede.")
        compare = ("smith2005_package_boiler", "smith2005_field_erected_boiler")
    purchase = sum(i["purchase_cost_usd"] for i in items)
    installed = sum(i["installed_cost_usd"] for i in items)
    dev = {k: _deviation(est[k]["purchase_cost_usd"], purchase) for k in compare}
    return {"items": items, "purchase_cost_usd": purchase, "installed_cost_usd": installed, "selected_estimate": chosen,
            "selection_reason": why, "estimates": est, "comparable_estimates": list(compare),
            "purchase_deviation_vs_selected": dev, "steam_lb_h": lb_h, "cepci": index, "cepci_basis": index_basis,
            "note": "Las estimaciones no comparables (por ejemplo la caldera de biomasa de BioSTEAM frente a una caldera "
                    "paquete de gas) se listan para que se vea la diferencia de alcance, no como contraste."}


def _turbine_capital(electric_W: float, mode: str, index: float, index_basis: str, warnings: list) -> dict:
    kW = electric_W / 1000.0
    b = _bst_cost("boiler_turbogenerator/Turbogenerator", kW, index)
    est = {"biosteam": {"purchase_cost_usd": b["purchase_cost_usd"], "installed_cost_usd": b["purchase_cost_usd"] * b["F_BM"],
                        "F_BM": b["F_BM"], "in_range": 1 / EXTRAPOLATION_FACTOR <= b["size_ratio"] <= EXTRAPOLATION_FACTOR,
                        "size_ratio_to_reference": b["size_ratio"],
                        "basis": "BioSTEAM BoilerTurbogenerator «Turbogenerator»: 9,500,000 USD (CE 551) a 42,200 kW, exponente "
                                 "0.6, F_BM 1.8 (de BioSTEAM). Es la cotización Siemens de Humbird 2011: turbina multietapa de "
                                 "extracción y condensación con generador."}}
    eq_kw, inside = _loglog(kW, EPA_TURBINE["kW"], EPA_TURBINE["equipment_usd_kW"])
    in_kw, _ = _loglog(kW, EPA_TURBINE["kW"], EPA_TURBINE["installed_usd_kW"])
    f = index / CEPCI_REF[2014]
    est["epa_chp2015"] = {"purchase_cost_usd": eq_kw * kW * f, "installed_cost_usd": in_kw * kW * f, "F_BM": in_kw / eq_kw,
                          "in_range": inside,
                          "basis": f"tabla de {EPA_TURBINE['description']}; 500-15,000 kW, USD/kW de 2014 con interpolación "
                                   f"log-log; instalado = el de la tabla (50-70 % sobre el equipo). {_src('epa_chp2015')}"}
    if mode == "back_pressure" and inside:
        chosen, why = "epa_chp2015", "turbina de contrapresión dentro de la tabla publicada de 500-15,000 kW"
    else:
        chosen = "biosteam"
        why = ("turbina de condensación: cotización de Humbird 2011 escalada por BioSTEAM" if mode == "condensing" else
               "fuera de la tabla de turbinas de contrapresión (500-15,000 kW): cotización de Humbird 2011 escalada por BioSTEAM")
        _extrapolation(b["size_ratio"], "Turbogenerador (BioSTEAM/Humbird)", warnings)
    sel = est[chosen]
    item = _item("Turbogenerador de vapor", sel["purchase_cost_usd"], sel["F_BM"], sel["basis"] + f" Índice: {index_basis}.")
    other = "biosteam" if chosen == "epa_chp2015" else "epa_chp2015"
    return {"items": [item], "purchase_cost_usd": item["purchase_cost_usd"], "installed_cost_usd": item["installed_cost_usd"],
            "selected_estimate": chosen, "selection_reason": why, "estimates": est,
            "purchase_deviation_vs_selected": {other: _deviation(est[other]["purchase_cost_usd"], sel["purchase_cost_usd"])}}


def steam_boiler_design(steam_demand: list[dict], feedwater_T_K: float, fuel: dict, boiler: dict | None = None,
                        combustion: dict | None = None, turbogenerator: dict | None = None, year: int = 2023,
                        cepci: float | None = None) -> dict:
    """Steam boiler (and optional turbogenerator) from the steam demand by pressure level; see the module docstring and
    tools_utilities_capital.steam_boiler_design."""
    if not steam_demand:
        raise ValueError("Falta steam_demand: lista de niveles [{name, pressure_Pa (absoluta), flow_kg_s}] que la planta "
                         "consume.")
    levels = []
    for i, lv in enumerate(steam_demand):
        levels.append({"name": str(lv.get("name", f"nivel {i + 1}")),
                       "pressure_Pa": _pos(lv.get("pressure_Pa"), f"steam_demand[{i}].pressure_Pa", "presión absoluta del nivel"),
                       "flow_kg_s": _pos(lv.get("flow_kg_s"), f"steam_demand[{i}].flow_kg_s", "vapor que consume ese nivel")})
    levels.sort(key=lambda d: -d["pressure_Pa"])
    Tfw = _pos(feedwater_T_K, "feedwater_T_K", "temperatura del agua de alimentación a la caldera")
    fuel = dict(_need(fuel, "fuel", "combustible: {'type': natural_gas | fuel_oil | biomass, ...}"))
    boiler = dict(boiler or {})
    warnings, assumptions, suggested, reasons = [], [], [], []
    index, index_basis = _index(year, cepci)
    fuel_type = fuel.get("type")
    if fuel_type not in FUEL_TYPES:
        raise ValueError(f"Falta fuel['type']: uno de {sorted(FUEL_TYPES)}. Decide la correlación de capital de la caldera.")
    m_demand = sum(lv["flow_kg_s"] for lv in levels)
    turbine = None
    m_cond = 0.0
    if turbogenerator is not None:
        tg = dict(turbogenerator)
        P_in = _pos(tg.get("inlet_pressure_Pa"), "turbogenerator.inlet_pressure_Pa", "presión absoluta de admisión")
        T_in = _pos(tg.get("inlet_T_K"), "turbogenerator.inlet_T_K", "temperatura de admisión")
        eta_is = _need(tg.get("isentropic_efficiency"), "turbogenerator.isentropic_efficiency",
                       "eficiencia isentrópica de TU turbina (del fabricante); la Tabla 4-2 de la EPA (2015) publica 52.5 % a "
                       "500 kW, 61.2 % a 3 MW y 78 % a 15 MW para turbinas de contrapresión")
        mode = tg.get("mode")
        if mode not in ("back_pressure", "condensing"):
            raise ValueError("turbogenerator.mode: 'back_pressure' (todo el vapor sale a los niveles de proceso) o 'condensing' "
                             "(además condensa turbogenerator.condensing_flow_kg_s a condenser_pressure_Pa).")
        sections, through = [], m_demand
        if mode == "condensing":
            m_cond = _pos(tg.get("condensing_flow_kg_s"), "turbogenerator.condensing_flow_kg_s", "vapor que llega al condensador")
            P_cond = _pos(tg.get("condenser_pressure_Pa"), "turbogenerator.condenser_pressure_Pa", "presión absoluta del "
                          "condensador")
            if P_cond >= levels[-1]["pressure_Pa"]:
                raise ValueError("condenser_pressure_Pa debe ser menor que el nivel de proceso de menor presión.")
            through += m_cond
        for lv in levels:
            sections.append({"outlet_pressure_Pa": lv["pressure_Pa"], "flow_kg_s": through})
            through -= lv["flow_kg_s"]
        if mode == "condensing":
            sections.append({"outlet_pressure_Pa": P_cond, "flow_kg_s": m_cond})
        exp = turbine_expansion(P_in, T_in, eta_is, sections)
        gen = tg.get("generator_efficiency")
        if gen is None:
            gen = EPA_TURBINE["generator_gearbox_efficiency"][0]
            suggested.append({"input": "turbogenerator.generator_efficiency", "value": gen, "unit": "fracción (generador + "
                              "caja de engranes)", "source": _src("epa_chp2015"),
                              "note": "94 % a 500 kW y 3 MW, 96 % a 15 MW; se usa el valor bajo"})
        gen = _frac(gen, "turbogenerator.generator_efficiency", "eficiencia generador + caja", hi_open=False, lo_open=True)
        steam_state = exp["inlet"]
        m_steam = m_demand + m_cond
        delivered = []
        for lv, sec in zip(levels, exp["sections"]):
            delivered.append({**lv, **{k: sec["outlet"][k] for k in ("T_K", "enthalpy_J_kg", "quality", "superheat_K")}})
            if sec["outlet"]["quality"] is not None and sec["outlet"]["quality"] < 1:
                warnings.append(f"El vapor del nivel «{lv['name']}» sale húmedo de la turbina (calidad "
                                f"{sec['outlet']['quality']:.3f}).")
        exhaust = exp["sections"][-1]["outlet"]
        if mode == "condensing" and exhaust["quality"] is not None:
            warnings.append(f"Escape al condensador con calidad {exhaust['quality']:.3f}: revisa con el fabricante la humedad "
                            "admisible en las últimas etapas.")
        out_enthalpy = sum(lv["flow_kg_s"] * d["enthalpy_J_kg"] for lv, d in zip(levels, delivered))
        if mode == "condensing":
            out_enthalpy += m_cond * exhaust["enthalpy_J_kg"]
        closure = abs(m_steam * steam_state["enthalpy_J_kg"] - out_enthalpy - exp["shaft_power_W"]) / (
            m_steam * steam_state["enthalpy_J_kg"])
        if closure > CLOSURE_TOL:
            reasons.append("el balance de energía de la turbina no cierra")
        if any(s["entropy_generated_J_kgK"] < -1e-6 for s in exp["sections"]):
            reasons.append("una sección de la turbina reduce la entropía")
        electric = exp["shaft_power_W"] * gen
        turbine = {"mode": mode, "inlet": steam_state, "isentropic_efficiency": float(eta_is), "generator_efficiency": gen,
                   "sections": exp["sections"], "shaft_power_W": exp["shaft_power_W"], "electric_power_W": electric,
                   "isentropic_power_W": exp["isentropic_power_W"], "energy_closure": closure,
                   "condensing_flow_kg_s": m_cond, "condenser_duty_W": (m_cond * (exhaust["enthalpy_J_kg"] - float(
                       _cp().PropsSI("H", "P", exhaust["pressure_Pa"], "Q", 0, "Water"))) if mode == "condensing" else 0.0)}
        assumptions.append("Todas las secciones de la turbina usan la misma eficiencia isentrópica; sin fugas por sellos ni "
                           "pérdidas mecánicas aparte de generator_efficiency.")
        P_boiler = P_in
    else:
        P_boiler = float(boiler.get("pressure_Pa") or levels[0]["pressure_Pa"])
        if P_boiler < levels[0]["pressure_Pa"]:
            raise ValueError("boiler['pressure_Pa'] no puede ser menor que el nivel de vapor de mayor presión.")
        if "steam_T_K" in boiler and boiler["steam_T_K"] is not None:
            steam_state = _steam(P_boiler, T=float(boiler["steam_T_K"]))
        else:
            sh = _need(boiler.get("superheat_K"), "boiler['superheat_K'] (o boiler['steam_T_K'])",
                       "sobrecalentamiento del vapor que genera la caldera (0 = saturado)")
            if sh < 0:
                raise ValueError("boiler['superheat_K'] no puede ser negativo.")
            steam_state = _steam(P_boiler, T=float(_cp().PropsSI("T", "P", P_boiler, "Q", 1, "Water")) + float(sh))
        m_steam = m_demand
        delivered = []
        for lv in levels:
            st = _steam(lv["pressure_Pa"], h=steam_state["enthalpy_J_kg"]) if lv["pressure_Pa"] < P_boiler else steam_state
            delivered.append({**lv, **{k: st[k] for k in ("T_K", "enthalpy_J_kg", "quality", "superheat_K")}})
        if len(levels) > 1 or levels[0]["pressure_Pa"] < P_boiler:
            assumptions.append("Sin turbina: la caldera genera a una sola presión y los niveles menores salen por válvula "
                               "reductora (isoentálpica), sin atemperación; el flujo generado es la suma de las demandas.")
    bd = boiler.get("blowdown_fraction")
    if bd is None:
        bd = HUMBIRD_BLOWDOWN_OF_STEAM / (1 + HUMBIRD_BLOWDOWN_OF_STEAM)
        suggested.append({"input": "boiler.blowdown_fraction", "value": bd, "unit": "fracción del agua de alimentación",
                          "source": _src("humbird2011"), "note": "«Boiler blowdown is 3% of steam production» = 0.03/1.03 de la "
                                                                 "alimentación"})
    bd = _frac(bd, "boiler['blowdown_fraction']", "purga como fracción del agua de alimentación")
    cp = _cp()
    Tsat_b = float(cp.PropsSI("T", "P", P_boiler, "Q", 0, "Water"))
    if Tfw >= Tsat_b:
        raise ValueError(f"feedwater_T_K ({Tfw} K) debe ser menor que la saturación de la caldera ({Tsat_b:.2f} K).")
    h_fw = _h_liq(Tfw)
    m_fw = m_steam / (1 - bd)
    m_bd = m_fw - m_steam
    h_drum = float(cp.PropsSI("H", "P", P_boiler, "Q", 0, "Water"))
    Q_abs = m_steam * (steam_state["enthalpy_J_kg"] - h_fw) + m_bd * (h_drum - h_fw)
    # --- fuel
    eff_given = boiler.get("efficiency")
    if combustion is not None and eff_given is not None:
        raise ValueError("Da boiler['efficiency'] o el bloque combustion (que la calcula), no los dos.")
    fuel_out = {"type": fuel_type}
    if combustion is not None:
        from packages.energy import combustion as CB
        try:
            comb = CB.combustion_efficiency(**combustion, absorbed_duty_W=Q_abs, adiabatic_flame=False)
        except TypeError as e:
            raise ValueError(f"combustion: argumentos de combustion_efficiency no reconocidos ({e}).") from None
        if "firing" not in comb:
            raise ValueError("combustion: falta stack_T_K (y setting_loss_fraction + setting_loss_basis) para que el motor de "
                             "combustión calcule la eficiencia.")
        m_fuel = comb["firing"]["fuel_kg_s"]
        fuel_out.update({"fuel_kg_s": m_fuel, "LHV_J_kg": comb["fuel"]["LHV_J_kg"], "HHV_J_kg": comb["fuel"]["HHV_J_kg"],
                         "efficiency_LHV": comb["efficiency"]["LHV"]["efficiency"],
                         "efficiency_HHV": comb["efficiency"]["HHV"]["efficiency"],
                         "efficiency_basis": "calculada con combustion_efficiency (método de pérdidas)",
                         "air_kg_s": comb["firing"]["moist_air_kg_s"], "flue_gas_kg_s": comb["firing"]["flue_gas_kg_s"],
                         "heating_value_basis": comb["fuel"]["heating_value_source"]})
        warnings += comb["warnings"]
        assumptions += comb["assumptions"]
        if not comb["evidence"]["verified"]:
            reasons += [f"combustión: {r}" for r in comb["evidence"]["reasons"]]
        fuel_out["combustion_evidence"] = comb["evidence"]
    else:
        eta = _need(eff_given, "boiler['efficiency']", "eficiencia de TU caldera (calor al vapor ÷ poder calorífico quemado), o "
                    "pasa el bloque combustion para calcularla con combustion_efficiency")
        eta = _frac(eta, "boiler['efficiency']", "eficiencia de la caldera", hi_open=True, lo_open=True)
        basis = boiler.get("efficiency_basis")
        if basis not in ("HHV", "LHV"):
            raise ValueError("Falta boiler['efficiency_basis']: 'HHV' o 'LHV', la base del poder calorífico con que se definió "
                             "la eficiencia.")
        hv = _fuel_heating_values(fuel)
        key = f"{basis}_J_kg"
        if key not in hv:
            raise ValueError(f"La eficiencia está en base {basis} pero el combustible no trae {key}: da ese poder calorífico "
                             "(para biomasa en base LHV hace falta hydrogen_dry_mass_fraction) o cambia la base.")
        m_fuel = Q_abs / (eta * hv[key])
        fuel_out.update({"fuel_kg_s": m_fuel, "LHV_J_kg": hv.get("LHV_J_kg"), "HHV_J_kg": hv.get("HHV_J_kg"),
                         f"efficiency_{basis}": eta, "efficiency_basis": f"dato del usuario, base {basis}",
                         "heating_value_basis": hv["basis"]})
        if "LHV_J_kg" in hv and "HHV_J_kg" in hv:
            other = "LHV" if basis == "HHV" else "HHV"
            fuel_out[f"efficiency_{other}"] = eta * hv[key] / hv[f"{other}_J_kg"]
    for b_ in ("LHV", "HHV"):
        if fuel_out.get(f"{b_}_J_kg"):
            fuel_out[f"heat_release_{b_}_W"] = m_fuel * fuel_out[f"{b_}_J_kg"]
    capital = _boiler_capital(fuel_type, m_steam, P_boiler, index, index_basis, warnings)
    aux = _bst_cost("boiler_turbogenerator/Boiler", m_steam * 3600.0, index)["electricity_W"]
    if turbine is not None:
        tcap = _turbine_capital(turbine["electric_power_W"], turbine["mode"], index, index_basis, warnings)
        turbine["capital"] = tcap
        capital["items"] = capital["items"] + tcap["items"]
        capital["boiler_purchase_cost_usd"], capital["boiler_installed_cost_usd"] = (capital["purchase_cost_usd"],
                                                                                     capital["installed_cost_usd"])
        capital["purchase_cost_usd"] += tcap["purchase_cost_usd"]
        capital["installed_cost_usd"] += tcap["installed_cost_usd"]
        if turbine["mode"] == "condensing":
            warnings.append("El condensador de superficie y su agua de enfriamiento no están en el capital: dimensiona la "
                            "torre con cooling_tower_design usando condenser_duty_W.")
    for s in suggested:
        assumptions.append(f"SUGERIDO (puedes sustituirlo): {s['input']} = {s['value']:.4g} {s['unit']}. Fuente: {s['source']}.")
    assumptions.append("Entalpía del agua de alimentación y de la purga sobre la línea de saturación; la purga sale como "
                       "líquido saturado a la presión de la caldera y no se recupera su calor.")
    steam_closure = abs(Q_abs - (m_steam * steam_state["enthalpy_J_kg"] + m_bd * h_drum - m_fw * h_fw)) / Q_abs
    if steam_closure > CLOSURE_TOL:
        reasons.append("el balance de energía del lado vapor no cierra")
    out = {
        "steam_generation_kg_s": m_steam, "boiler_pressure_Pa": P_boiler, "boiler_steam": steam_state,
        "feedwater_kg_s": m_fw, "feedwater_T_K": Tfw, "blowdown_kg_s": m_bd, "blowdown_fraction": bd,
        "absorbed_duty_W": Q_abs, "specific_duty_J_per_kg_steam": Q_abs / m_steam, "steam_levels": delivered,
        "fuel": fuel_out, "turbogenerator": turbine, "capital": capital,
        "reference_auxiliary_power_W": aux,
        "reference_auxiliary_power_basis": "BioSTEAM: 1371 kW por 238,686 kg/h de vapor (caldera de biomasa de Humbird 2011), "
                                           "lineal con el vapor; solo orientativo para calderas de gas",
        "cost_year": year, "cepci": index, "suggested_inputs": suggested, "warnings": warnings, "assumptions": assumptions,
        "not_included": ["bombas de agua de alimentación y su potencia", "tratamiento de agua de repuesto y retorno de "
                         "condensado (van en utility_costs, bloque steam)", "patio y manejo de combustible sólido",
                         "condensador de la turbina, torre de enfriamiento asociada y subestación eléctrica",
                         "atemperadores y válvulas reductoras", "corrección del costo por presión y sobrecalentamiento",
                         "costo de operación (está en utility_costs)"],
        "method": {"name": "balance de energía del lado vapor (IAPWS-95); combustible por eficiencia o por el método de "
                           "pérdidas; turbina por secciones con eficiencia isentrópica; capital por tablas y cotizaciones "
                           "publicadas escaladas",
                   "libraries": "CoolProp (IAPWS-95), Cantera vía packages.energy.combustion, BioSTEAM (solo sus correlaciones)",
                   "module": "packages/costing/utilities_capital.py",
                   "sources": [_src(k) for k in ("humbird2011", "netl2002", "smith2005", "epa_chp2015", "epa_biomass2007")]},
        "evidence": {"verified": not reasons, "reasons": reasons, "steam_side_energy_closure": steam_closure,
                     "turbine_energy_closure": turbine["energy_closure"] if turbine else None,
                     "note": "verified certifica los balances de esta corrida; el capital es un estimado de estudio y su "
                             "dispersión entre fuentes está en capital.purchase_deviation_vs_selected"},
    }
    return out


# ------------------------------------------------------------------ wastewater ---------------------------------------------
def methane_stoichiometry() -> dict:
    """CH4 + 2 O2 -> CO2 + 2 H2O: the COD of methane, and the methane made per kg of COD converted."""
    kg_cod_per_kg = 2 * M_O2 / M_CH4
    return {"kg_COD_per_kg_CH4": kg_cod_per_kg, "kg_CH4_per_kg_COD": 1 / kg_cod_per_kg,
            "Nm3_CH4_per_kg_COD": 1000.0 / (2 * M_O2) * NORMAL_M3_PER_MOL,
            "normal_m3_definition": "0 °C y 101 325 Pa, gas ideal"}


def _methane_lhv() -> float:
    """LHV of methane at 298.15 K from the NASA polynomials of the repo's combustion engine (J/kg)."""
    from packages.energy import combustion as CB
    h = lambda k: CB.h_molar(k, 298.15)   # noqa: E731
    return -(h("CO2") + 2 * h("H2O") - h("CH4") - 2 * h("O2")) / (M_CH4 / 1000.0)


def wastewater_treatment_cost(flow_m3_s: float, cod_kg_m3: float, configuration: str, cod_removal: dict,
                              include_reverse_osmosis: bool = False, include_brine_evaporator: bool = False,
                              density_kg_m3: float | None = None, electricity_price_usd_kWh: float | None = None,
                              operating_hours_per_year: float | None = None, aeration_kg_O2_per_kWh: float | None = None,
                              biogas_value_usd_GJ: float | None = None, year: int = 2023, cepci: float | None = None) -> dict:
    """Conventional wastewater treatment (aerobic or anaerobic + aerobic): COD balance, methane, capital by scaling the
    published itemised cost and electricity; see the module docstring and tools_utilities_capital.wastewater_treatment_cost."""
    V = _pos(flow_m3_s, "flow_m3_s", "caudal de agua residual")
    cod = _pos(cod_kg_m3, "cod_kg_m3", "demanda química de oxígeno del agua residual (1 g/L = 1 kg/m³)")
    if configuration not in ("anaerobic_aerobic", "aerobic"):
        raise ValueError("configuration: 'anaerobic_aerobic' (digestor anaerobio con biogás + pulido aerobio) o 'aerobic' "
                         "(lodos activados).")
    rem = dict(_need(cod_removal, "cod_removal", "fracciones de remoción de DQO de TU agua por etapa"))
    hint = ("Valores publicados (Humbird 2011, sec. 3.6.2, agua de 64-87 g/L de DQO): anaerobic 0.91 con anaerobic_to_biomass "
            "0.05; aerobic 0.96 con aerobic_to_biomass 0.22. Pásalos explícitamente si los aceptas.")
    need = (["anaerobic", "anaerobic_to_biomass"] if configuration == "anaerobic_aerobic" else []) + ["aerobic",
                                                                                                    "aerobic_to_biomass"]
    for k in need:
        if rem.get(k) is None:
            raise ValueError(f"Falta cod_removal['{k}']: fracción de la DQO que ENTRA a la etapa que se "
                             f"{'convierte en lodo (biomasa)' if k.endswith('biomass') else 'remueve'}. {hint}")
        _frac(rem[k], f"cod_removal['{k}']", "fracción de la DQO que entra a la etapa", hi_open=False)
    for st in ("anaerobic", "aerobic"):
        if st in need and rem[f"{st}_to_biomass"] > rem[st]:
            raise ValueError(f"cod_removal['{st}_to_biomass'] no puede superar cod_removal['{st}'] (es parte de lo removido).")
    warnings, assumptions, suggested, reasons = [], [], [], []
    index, index_basis = _index(year, cepci)
    if density_kg_m3 is None:
        rho = float(_cp().PropsSI("D", "T", 298.15, "P", P_ATM, "Water"))
        assumptions.append(f"Densidad del agua residual no dada: la del agua pura a 25 °C ({rho:.1f} kg/m³); solo afecta al "
                           "escalamiento por caudal másico.")
    else:
        rho = _pos(density_kg_m3, "density_kg_m3", "densidad del agua residual")
    load = V * cod
    st = methane_stoichiometry()
    stages = {}
    cod_in_aer = load
    methane_kg_s = 0.0
    if configuration == "anaerobic_aerobic":
        removed = rem["anaerobic"] * load
        to_ch4 = (rem["anaerobic"] - rem["anaerobic_to_biomass"]) * load
        methane_kg_s = to_ch4 * st["kg_CH4_per_kg_COD"]
        cod_in_aer = load - removed
        stages["anaerobic"] = {"COD_in_kg_s": load, "COD_removed_kg_s": removed, "COD_to_methane_kg_s": to_ch4,
                               "COD_to_biomass_kg_s": rem["anaerobic_to_biomass"] * load, "COD_out_kg_s": cod_in_aer,
                               "methane_kg_s": methane_kg_s, "methane_Nm3_s": to_ch4 * st["Nm3_CH4_per_kg_COD"],
                               "methane_kg_per_kg_COD_removed": methane_kg_s / removed if removed else 0.0}
    removed_a = rem["aerobic"] * cod_in_aer
    oxidized = (rem["aerobic"] - rem["aerobic_to_biomass"]) * cod_in_aer
    effluent = cod_in_aer - removed_a
    stages["aerobic"] = {"COD_in_kg_s": cod_in_aer, "COD_removed_kg_s": removed_a, "oxygen_demand_kg_s": oxidized,
                         "COD_to_biomass_kg_s": rem["aerobic_to_biomass"] * cod_in_aer, "COD_out_kg_s": effluent}
    sinks = (effluent + oxidized + stages["aerobic"]["COD_to_biomass_kg_s"]
             + (stages["anaerobic"]["COD_to_methane_kg_s"] + stages["anaerobic"]["COD_to_biomass_kg_s"]
                if "anaerobic" in stages else 0.0))
    cod_closure = abs(load - sinks) / load
    if cod_closure > CLOSURE_TOL:
        reasons.append("el balance de DQO no cierra")
    lhv = _methane_lhv()
    biogas = {"methane_kg_s": methane_kg_s, "methane_Nm3_s": methane_kg_s / (M_CH4 / 1000.0) * NORMAL_M3_PER_MOL,
              "methane_LHV_J_kg": lhv, "energy_LHV_W": methane_kg_s * lhv, "stoichiometry": st,
              "note": "solo el metano; el CO2 y el H2S del biogás no se calculan (dependen de la composición del agua)"}
    # --- capital: itemised Humbird 2011, biological reactors on COD load, the rest on hydraulic flow
    kg_h = V * rho * 3600.0
    r_flow = kg_h / HUMBIRD_WWT_FLOW_KG_H
    ref_cod_kg_h = HUMBIRD_WWT_DESIGN["hydraulic_MMgal_d"] * 1e6 * US_GAL_M3 / 24.0 * HUMBIRD_WWT_DESIGN["total_COD_g_L"]
    ref_aer_kg_h = ref_cod_kg_h * (1 - HUMBIRD_WWT_DESIGN["anaerobic_removal"])
    r_ana = load * 3600.0 / ref_cod_kg_h
    r_aer = cod_in_aer * 3600.0 / ref_aer_kg_h
    f = index / CEPCI_REF[2010]
    n = HUMBIRD_WWT_EXPONENT
    groups = {"common", "aerobic", "aerobic_reactor", "sludge"}
    if configuration == "anaerobic_aerobic":
        groups |= {"anaerobic", "anaerobic_reactor"}
    if include_reverse_osmosis:
        groups.add("reverse_osmosis")
    if include_brine_evaporator:
        groups.add("brine_evaporator")
    items, hydraulic_same_scope = [], 0.0
    for tag, name, cost, group in HUMBIRD_WWT_ITEMS:
        if group not in groups:
            continue
        ratio, on = (r_ana, "carga de DQO al digestor") if group == "anaerobic_reactor" else (
            (r_aer, "carga de DQO a la etapa aerobia") if group == "aerobic_reactor" else (r_flow, "caudal"))
        installed = cost * ratio ** n * f
        hydraulic_same_scope += cost * r_flow ** n * f
        items.append({"name": f"{tag} {name}", "F_BM": 1.0, "purchase_cost_usd": installed, "base_purchase_cost_usd": installed,
                      "installed_cost_usd": installed, "scaled_on": on, "size_ratio_to_reference": ratio,
                      "basis": f"Humbird 2011, lista de equipo del Área 600: {cost:,.0f} USD INSTALADO de 2010 (factor de "
                               f"instalación 1.0 en la fuente, por eso F_BM = 1), exponente {n}, escalado por {on}. Índice: "
                               f"{index_basis}."})
    total = sum(i["installed_cost_usd"] for i in items)
    bst_lump = _bst_cost("wastewater/Wastewater system", kg_h, index)
    full_scope = sum(c for _, _, c, _ in HUMBIRD_WWT_ITEMS)
    aden_ratio = load * 3600.0 / ADEN_WWT["COD_kg_h"]
    for ratio, what in ((r_flow, "caudal"), (r_ana if configuration == "anaerobic_aerobic" else r_aer, "carga de DQO")):
        _extrapolation(ratio, f"Tratamiento de aguas (referencia Humbird 2011, {what})", warnings)
    conc_ratio = cod / (ref_cod_kg_h / (HUMBIRD_WWT_FLOW_KG_H / rho))
    if not 0.5 <= r_ana / r_flow <= 2.0:
        warnings.append(f"Tu agua tiene {cod:.3g} kg/m³ de DQO y la planta de referencia ~{ref_cod_kg_h / HUMBIRD_WWT_FLOW_KG_H * rho:.0f}"
                        " kg/m³: los reactores biológicos se escalaron por carga de DQO y el resto por caudal; el escalamiento "
                        f"solo por caudal de BioSTEAM daría {_deviation(hydraulic_same_scope, total):+.0%} para el mismo alcance.")
    if configuration == "aerobic":
        warnings.append("Configuración solo aerobia: la laguna y los sopladores de la referencia tratan el 9 % de la DQO (lo que "
                        "deja el digestor anaerobio); aquí se escalaron a toda tu carga de DQO, fuera del diseño original.")
    capital = {
        "items": items, "installed_cost_usd": total, "purchase_cost_usd": total,
        "scope": sorted(groups), "selected_estimate": "humbird2011_itemized_cod_and_flow",
        "selection_reason": "única fuente con costo desglosado; los reactores biológicos se dimensionan por carga orgánica "
                            "(Aden 2002 dimensiona el digestor a 12 g DQO/L·día), así que se escalan por DQO y no por caudal",
        "estimates": {
            "humbird2011_itemized_cod_and_flow": {"installed_cost_usd": total, "basis": "este resultado"},
            "humbird2011_itemized_flow_only": {"installed_cost_usd": hydraulic_same_scope,
                                               "basis": "mismo alcance, todo escalado por caudal como hace la fuente"},
            "biosteam": {"installed_cost_usd": bst_lump["purchase_cost_usd"], "size_ratio_to_reference": bst_lump["size_ratio"],
                         "basis": f"BioSTEAM {biosteam_cost_items()['biosteam_version']} WastewaterSystemCost: 50,280,080 USD "
                                  "(CE 551) a 393,100 kg/h, exponente 0.6, F_BM 1; es la suma de TODA el Área 600 de Humbird "
                                  f"2011 ({full_scope:,.0f} USD, con membranas, ósmosis inversa y evaporador), por caudal."},
            "aden2002": {"installed_cost_usd": ADEN_WWT["installed_usd"] * aden_ratio ** n * index / CEPCI_REF[1998],
                         "size_ratio_to_reference": aden_ratio,
                         "basis": f"{ADEN_WWT['description']}: 3,252,573 USD instalado (1998) para 1,600 kg/h de DQO (16 g/L), "
                                  f"escalado aquí por carga de DQO con exponente {n} (el exponente es de este módulo: la fuente "
                                  f"escala cada equipo por separado). {_src('aden2002')}"}},
        "cepci": index, "cepci_basis": index_basis}
    capital["installed_deviation_vs_selected"] = {k: _deviation(v["installed_cost_usd"], total)
                                                  for k, v in capital["estimates"].items() if k != capital["selected_estimate"]}
    warnings.append("El capital de tratamiento de aguas depende sobre todo del alcance: las dos plantas publicadas por NREL "
                    "(2002 y 2011) difieren en un factor de 2 a 5 una vez escaladas (ver capital.installed_deviation_vs_selected)."
                    " Úsalo como orden de magnitud.")
    # --- electricity and operating cost
    ref_power = bst_lump["electricity_W"]
    power = {"reference_power_W": ref_power,
             "reference_basis": "BioSTEAM: 6,799 kW (7,139/1.05) por 393,100 kg/h, lineal con el caudal; Humbird 2011 atribuye al "
                                "Área 600 el 18 % de 41 MW. Dominan los sopladores de aireación, que en realidad siguen a la "
                                "DQO aerobia y no al caudal."}
    if aeration_kg_O2_per_kWh is not None:
        ae = _pos(aeration_kg_O2_per_kWh, "aeration_kg_O2_per_kWh", "eficiencia de aireación en campo de TUS aireadores")
        power["aeration_power_W"] = oxidized / ae * 3.6e6
        power["aeration_basis"] = "física: demanda de oxígeno (DQO oxidada en la etapa aerobia) ÷ eficiencia de aireación"
        power["used_W"] = power["aeration_power_W"]
        assumptions.append("La potencia usada es solo la de aireación calculada; bombeo, centrífugas y membranas no se suman.")
    else:
        power["used_W"] = ref_power * (r_aer / r_flow if r_flow else 1.0)
        power["used_basis"] = ("potencia de referencia escalada con la carga de DQO a la etapa aerobia en vez del caudal "
                               "(criterio de este módulo); da aeration_kg_O2_per_kWh para calcular la aireación")
        suggested.append({"input": "potencia eléctrica del tratamiento", "value": ref_power / 1000.0 / (kg_h / 1000.0),
                          "unit": "kW por t/h de agua (antes de corregir por DQO)", "source": _src("humbird2011"),
                          "note": "vía BioSTEAM WastewaterSystemCost"})
    operating = {"electricity_W": power["used_W"]}
    if electricity_price_usd_kWh is not None or operating_hours_per_year is not None or biogas_value_usd_GJ is not None:
        hours = _pos(operating_hours_per_year, "operating_hours_per_year", "horas de operación al año de TU planta")
        operating["operating_hours_per_year"] = hours
        if electricity_price_usd_kWh is not None:
            price = _pos(electricity_price_usd_kWh, "electricity_price_usd_kWh", "precio de la electricidad")
            operating["electricity_cost_usd_y"] = power["used_W"] / 1000.0 * hours * price
        if biogas_value_usd_GJ is not None:
            operating["biogas_credit_usd_y"] = biogas["energy_LHV_W"] * hours * 3600.0 / 1e9 * float(biogas_value_usd_GJ)
        operating["net_cost_usd_y"] = operating.get("electricity_cost_usd_y", 0.0) - operating.get("biogas_credit_usd_y", 0.0)
    else:
        operating["note"] = ("sin electricity_price_usd_kWh y operating_hours_per_year solo se informa la potencia; el precio "
                             "es tuyo (o de list_utility_price_references)")
    for s in suggested:
        assumptions.append(f"SUGERIDO (puedes sustituirlo): {s['input']} = {s['value']:.4g} {s['unit']}. Fuente: {s['source']}.")
    assumptions += ["Las fracciones de remoción y de lodo son del usuario y se aplican a la DQO que entra a cada etapa.",
                    "Metano por estequiometría: 1 kg de DQO convertida = 0.25 kg de CH4 (0.35 Nm³); la DQO que va a lodo no da "
                    "metano."]
    return {
        "flow_m3_s": V, "flow_kg_s": V * rho, "COD_kg_m3": cod, "COD_load_kg_s": load, "configuration": configuration,
        "stages": stages, "effluent_COD_kg_s": effluent, "effluent_COD_kg_m3": effluent / V,
        "overall_COD_removal": 1 - effluent / load, "biogas": biogas, "oxygen_demand_kg_s": oxidized,
        "sludge_COD_kg_s": sum(s["COD_to_biomass_kg_s"] for s in stages.values()),
        "capital": capital, "power": power, "operating": operating, "cost_year": year, "cepci": index,
        "reference_plant": {"COD_load_kg_h": ref_cod_kg_h, "flow_kg_h": HUMBIRD_WWT_FLOW_KG_H,
                            "COD_load_ratio": r_ana, "aerobic_COD_load_ratio": r_aer, "flow_ratio": r_flow,
                            "COD_concentration_ratio": conc_ratio,
                            "basis": "base de diseño de Brown and Caldwell en Humbird 2011, Tabla 22: 2.15 MMgal/d y 87 g/L de "
                                     "DQO total; remoción anaerobia 91 %"},
        "suggested_inputs": suggested, "warnings": warnings, "assumptions": assumptions,
        "not_included": ["nitrificación y su demanda de oxígeno y de sosa", "nutrientes, polímero y sosa (costo de químicos)",
                         "masa de lodo y su disposición (se informa como DQO; falta la relación DQO/SSV de tu lodo)",
                         "CO2 y H2S del biogás, limpieza y compresión del biogás", "pretratamiento (cribas, igualación, "
                         "neutralización) y descarga", "mano de obra y mantenimiento", "costos indirectos del proyecto"],
        "method": {"name": "balance de DQO por etapa con las remociones del usuario; metano estequiométrico; capital por la "
                           "lista de equipo instalada de Humbird 2011 escalada (reactores por DQO, resto por caudal)",
                   "libraries": "CoolProp (densidad del agua), polinomios NASA vía packages.energy.combustion (LHV del metano), "
                                "BioSTEAM (solo su correlación WastewaterSystemCost; su sistema convencional no se simula: "
                                "necesita la composición química del agua y su costo es esta misma ley de potencia)",
                   "module": "packages/costing/utilities_capital.py", "sources": [_src("humbird2011"), _src("aden2002")]},
        "evidence": {"verified": not reasons, "reasons": reasons, "COD_balance_closure": cod_closure,
                     "methane_Nm3_per_kg_COD_converted": st["Nm3_CH4_per_kg_COD"],
                     "note": "verified certifica el balance de DQO de esta corrida; el capital NO está validado más allá de "
                             "reproducir la fuente (ver capital.installed_deviation_vs_selected)"},
    }
