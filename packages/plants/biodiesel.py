"""Vegetable oil -> biodiesel (FAME) + crude glycerol by alkali-catalysed transesterification with methanol (bloque M26_B).

The flowsheet is the published BioSTEAM section `biorefineries.biodiesel.create_transesterification_and_biodiesel_
separation_system` (the biodiesel area of the lipid-cane / oilcane biorefineries: Cortés-Peña et al. 2020, design of Huang et
al. 2016, with reagent ratios of the USDA model): oil tank, two stirred reactors at 60 °C with a glycerol centrifuge
between them, water wash, vacuum dryer, methanol recovery column and glycerol concentration to 80 % w/w. This module only
(a) builds the chemical set for the oil the caller chose, (b) feeds the section, (c) attaches BioSTEAM's conventional TEA
and (d) reads the results. Balances, sizing and costs are BioSTEAM's.

Which oils the chemical database can represent (checked in thermosteam 0.54 / chemicals, 2026-11-16):
  * tripalmitin (CAS 555-44-2): vapour pressure (Landolt-Börnstein), liquid heat capacity (Zábranský) and viscosity
    (Viswanath-Natarajan) are data correlations, UNIFAC groups present -> used as it is. Methyl palmitate: reference EOS fits.
  * triolein (CAS 122-32-7): present, but every property method is a corresponding-states / group-contribution estimate and
    some are wrong (heat of formation +2.49 MJ/mol, liquid viscosity 0.012 mPa·s, density 1000 kg/m³ at 60 °C) -> NOT used.
    BioSTEAM's own published pseudo-component «TriOlein» is used instead (tripalmitin + 3 CH=CH groups, 900 kg/m³,
    Hf -1776 kJ/mol), exactly as in the validated oilcane/lipid-cane models. Methyl oleate: reference EOS fits.
  * trilinolein (CAS 537-40-6): estimates only, no UNIFAC groups -> refused.
  * triricinolein (CAS 2540-54-7): absent. Ricinoleic acid and methyl ricinoleate exist with estimate-only methods (the
    estimated density of methyl ricinoleate at 60 °C is 836 kg/m³; measured castor biodiesel is 937.6 kg/m³ at 15.5 °C)
    -> castor oil is a DECLARED pseudo-triglyceride: formula and molar mass of triricinolein, every other property copied
    from BioSTEAM's TriOlein on a mass basis; its ester has the formula and molar mass of methyl ricinoleate and the
    properties of methyl oleate on a mass basis. The stoichiometry (yields per kg) is therefore castor's; the physical
    behaviour (viscosity, methanol and glycerol solubility, phase split) is that of an oleic oil and is flagged.

Prices and TEA parameters not given by the caller are SUGGESTED from Huang et al. 2016 (Tables 4, 5 and 6, read
2026-11-16) and returned flagged; the oil price has no default. The free-fatty-acid limit of the alkali route is the one
stated in Van Gerpen et al. 2004 (NREL/SR-510-36244).
"""
from __future__ import annotations

import math

from packages.plants import reference as R

HUANG_URL = "https://www.osti.gov/servlets/purl/1437267"
HUANG_T4 = f"Huang et al. 2016, Tabla 4 (precios de 2013), {HUANG_URL}"
HUANG_T5 = f"Huang et al. 2016, Tabla 5, {HUANG_URL}"
VANGERPEN = ("Van Gerpen, Shanks, Pruszko, Clements y Knothe 2004, «Biodiesel Production Technology», NREL/SR-510-36244, "
             "https://www.osti.gov/servlets/purl/15008801")
HUMBIRD_URL = "https://www.osti.gov/servlets/purl/1013269"
CASTOR_PROPS = ("Rodríguez-Ramos et al. 2026, ACS Omega, doi 10.1021/acsomega.5c10172 (PMC13347325), "
                "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13347325/fullTextXML")
CASTOR_SEP = ("Heliyon 2023, «Experimental and theoretical assessment of phenomena linked with separation and purification of "
              "biodiesel from Ricinus communis seed oil» (PMC10220419), "
              "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10220419/fullTextXML")

FFA_ALKALI_LIMIT = 0.01      # Van Gerpen 2004: «free fatty acid contents greater than about 1% must be pretreated»
FFA_NEGLIGIBLE = 0.005       # idem: «certainly if it is less than 0.5%, the FFAs can be ignored»
STEAM_USD_KG = 0.017         # Huang 2016 Table 4: steam 17 USD/t
ELECTRICITY_USD_KWH = 0.065  # Huang 2016 Table 4
REACTOR_T_K = 333.15         # Huang 2016: 60 °C, 1 h per reactor (values wired in the BioSTEAM section)
HUANG_CAPACITY_L_Y = 96e6    # plant for which Huang's labour (50 employees) was stated
CATALYST_BASE = 0.0025       # Van Gerpen 2004: «Sodium methoxide: 0.25% of triglycerides weight» ...
CATALYST_PER_FFA = 0.190     # ... and with FFA «[%FFA](0.190) + 0.25 %»
SECTION_X_CATALYST = 1.05e-5  # mole fraction of NaOCH3 in the methanol feed wired in the BioSTEAM section («USDA model»)

OILS = {
    "triolein": {"label": "trioleína (aceite oleico genérico; pseudo-componente TriOlein de BioSTEAM)", "ester": "oleato de metilo"},
    "tripalmitin": {"label": "tripalmitina (grasa saturada C16; datos de la base de compuestos)", "ester": "palmitato de metilo"},
    "castor": {"label": "aceite de ricino / higuerilla (pseudo-triglicérido declarado: trirricinoleína)",
               "ester": "ricinoleato de metilo (pseudo-componente)"},
}
OIL_ALIASES = {"ricino": "castor", "higuerilla": "castor", "castor_oil": "castor", "triricinolein": "castor",
               "trioleina": "triolein", "trioleína": "triolein", "tripalmitina": "tripalmitin"}
PRICE_ALIASES = {"oil": "vegetable_oil", "wash_water": "biodiesel_wash_water"}
# default prices of the BioSTEAM section (stream -> USD/kg) and where each comes from
DEFAULT_PRICE_SOURCE = {
    "methanol": HUANG_T4, "catalyst": HUANG_T4 + "; BioSTEAM aplica los 2.93 USD/kg a la solución al 25 % de NaOCH3 en metanol",
    "NaOH": HUANG_T4, "HCl": HUANG_T4 + " (0.205 USD/kg; BioSTEAM lo multiplica por 0.35 para el ácido diluido)",
    "biodiesel_wash_water": HUANG_T4 + " (0.353 USD/t)",
    "biodiesel": HUANG_T5 + " (1.22 USD/L, promedio de 2013; 1.38 USD/kg según biorefineries.biodiesel.streams)",
    "crude_glycerol": HUANG_T4 + " (glicerol crudo al 80 %)",
}
CRUDE_GLYCEROL_USD_KG = 0.21   # Huang 2016 Table 4
# TEA parameters suggested when the caller gives none (all Huang 2016 except the discount rate)
DEFAULT_TEA = {
    "discount_rate": (0.10, f"Humbird et al. 2011, Tabla 33 (tasa de descuento de una planta «n-ésima» de biocombustible), {HUMBIRD_URL}"),
    "income_tax": (0.35, HUANG_T5), "plant_life_years": (20, HUANG_T5), "depreciation": ("MACRS7", HUANG_T5),
    "lang_factor": (3.0, f"Huang et al. 2016, Tabla 6 (capital fijo = 3 × costo total de equipo), {HUANG_URL}"),
    "construction_schedule": ([0.4, 0.6], HUANG_T5), "working_capital_over_fci": (0.05, HUANG_T5),
    "labor_cost_usd_y": (2.5e6, HUANG_T4 + " (50 empleados a 50 000 USD/año, planta de 96 millones de L/año)"),
    "maintenance": (0.01, HUANG_T4), "property_tax": (0.001, HUANG_T4), "property_insurance": (0.005, HUANG_T4),
    "administration": (0.005, HUANG_T4),
}
FRINGE_BENEFITS = 0.40         # Huang 2016 Table 4: 40 % of labour
SUPPLIES = 0.20                # idem: operating supplies 20 % of operating labour


def list_oils() -> dict:
    """Oils `biodiesel_plant` accepts and how each is represented (no simulation)."""
    return {k: v["label"] for k, v in OILS.items()}


def _fraction(x, name: str, what: str) -> float:
    if x is None:
        raise ValueError(f"falta {name}: {what}")
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or not 0.0 <= x < 1.0:
        raise ValueError(f"{name} debe ser una fracción másica entre 0 y 1 ({what})")
    return float(x)


def _chemicals(tmo, oil: str):
    """Chemical set of the section for one oil. IDs the BioSTEAM section expects: TAG, DAG, MAG, FFA, Biodiesel, group Lipid."""
    from biorefineries.cane._chemicals import create_acyl_olein
    from thermosteam import functional as fn
    chems = tmo.Chemicals([tmo.Chemical("Water", cache=False), tmo.Chemical("Methanol", cache=False),
                           tmo.Chemical("Glycerol", cache=False)])
    for ID in ("HCl", "NaOH", "NaOCH3"):                     # as in biorefineries.cane: solutes that occupy no volume
        chems.append(tmo.Chemical.blank(ID, phase="s", formula=ID))
    for ID in ("HCl", "NaOH"):
        c = getattr(chems, ID)
        c.V.add_model(fn.rho_to_V(rho=1e5, MW=c.MW), top_priority=True)
    liquid_methanol = chems.Methanol.at_state(phase="l", copy=True)
    chems.NaOCH3.copy_models_from(liquid_methanol, ["V", "sigma", "kappa", "Cn"])
    triolein = create_acyl_olein(3)
    methyl_oleate = tmo.Chemical("Methyl oleate", cache=False)
    if oil == "triolein":
        tag = triolein
        ester = tmo.Chemical("Biodiesel", cache=False, search_ID="Methyl oleate")
    elif oil == "tripalmitin":
        tag = tmo.Chemical("TriPalmitin", cache=False, search_ID="Tripalmitin", phase="l")
        ester = tmo.Chemical("Biodiesel", cache=False, search_ID="Methyl palmitate")
    else:  # castor: declared pseudo-components, see the module docstring
        tag = tmo.Chemical("TriRicinolein", cache=False, search_db=False, phase="l", phase_ref="l", formula="C57H104O9",
                           Hf=triolein.Hf)
        tag._Dortmund = triolein.Dortmund.copy()
        tag.V.add_model(fn.rho_to_V(rho=900.0, MW=tag.MW))                     # BioSTEAM's TriOlein: 900 kg/m3
        tag.Cn.add_model(triolein.Cn(298.15) / triolein.MW * tag.MW)           # same J/(kg K)
        tag.copy_models_from(triolein, ["mu", "sigma", "kappa"])
        liquid_oleate = methyl_oleate.at_state("l", copy=True)
        ester = tmo.Chemical("Biodiesel", cache=False, search_db=False, phase="l", phase_ref="l", formula="C19H36O3",
                             Hf=methyl_oleate.Hf)
        ratio = ester.MW / methyl_oleate.MW
        ester._Dortmund = methyl_oleate.Dortmund.copy()
        ester.V.add_model(lambda T, P=None: liquid_oleate.V(T, 101325.0) * ratio, 250.0, 600.0)   # same kg/m3
        ester.Cn.add_model(lambda T: liquid_oleate.Cn(T) * ratio, 250.0, 600.0)                 # same J/(kg K)
        ester.copy_models_from(liquid_oleate, ["mu", "sigma", "kappa", "Psat", "Hvap"])
    chems.extend([ester, create_acyl_olein(0), create_acyl_olein(1), create_acyl_olein(2), tag])
    for c in chems:
        c.default()
    chems.compile()
    chems.set_synonym("OleicAcid", "FFA")
    chems.set_synonym("MonoOlein", "MAG")
    chems.set_synonym("DiOlein", "DAG")
    chems.set_synonym(tag.ID, "TAG")
    chems.define_group("Lipid", ("FFA", "MAG", "DAG", "TAG"))
    chems.set_synonym("Water", "H2O")
    return chems, tag, ester


def _restore_makeup_balance(unit) -> None:
    """BioSTEAM 2.54 decorates MassBalance with @static, which overwrites its `_run` with a pass-through: the unit that sets
    the fresh methanol and catalyst of this section does nothing and the plant «creates» methanol (mass residual 10 %). The
    documented calculation of that same unit (thermosteam.separations.material_balance with its own arguments) is put back on
    this one instance."""
    from thermosteam.separations import material_balance

    def run_balance():
        material_balance(chemical_IDs=unit.chemical_IDs, variable_inlets=unit.variable_inlets,
                         constant_outlets=unit.constant_outlets, constant_inlets=unit.constant_inlets,
                         is_exact=unit.is_exact, balance=unit.balance)
    unit._run = run_balance


def biodiesel_plant(oil_feed_kg_h: float | None = None, oil: str | None = None, ffa_mass_fraction: float | None = None,
                    water_mass_fraction: float | None = None, operating_hours_y: float | None = None,
                    prices: dict | None = None, tea: dict | None = None, conversion_per_stage: float | None = None,
                    methanol_excess: float | None = None, catalyst_mass_fraction_of_oil: float | None = None,
                    oil_temperature_K: float = 298.15, year: int | None = None, cepci: float | None = None) -> dict:
    """Biodiesel plant for `oil_feed_kg_h` of oil (as received: triglyceride + FFA + water). Same result shape as
    `reference_biorefinery`, plus `yields`, `oil_model` and the suggested prices and TEA parameters with their sources."""
    if oil_feed_kg_h is None:
        raise ValueError("falta oil_feed_kg_h: flujo de aceite tal como llega (kg/h), para dimensionar la planta")
    R._positive(oil_feed_kg_h, "oil_feed_kg_h", "kg/h de aceite tal como llega")
    if oil is None:
        raise ValueError(f"falta oil: qué aceite se procesa; opciones {sorted(OILS)} (list_oils las describe)")
    key = OIL_ALIASES.get(str(oil).strip().lower(), str(oil).strip().lower())
    if key in ("trilinolein", "trilinoleina", "trilinoleína"):
        raise ValueError("oil='trilinolein': la base de compuestos solo tiene estimaciones para la trilinoleína (sin grupos UNIFAC ni "
                         "correlaciones de datos); no se modela con propiedades inventadas. Use 'triolein' y declárelo")
    if key not in OILS:
        raise ValueError(f"oil={oil!r} no está disponible; opciones {sorted(OILS)}: solo triglicéridos que la base de compuestos "
                         "representa sin inventar propiedades (más el ricino como pseudo-componente declarado)")
    ffa = _fraction(ffa_mass_fraction, "ffa_mass_fraction", "ácidos grasos libres del aceite (fracción másica), decide si la ruta "
                    "alcalina es viable y cuánto catalizador pide")
    water = _fraction(water_mass_fraction, "water_mass_fraction", "humedad del aceite (fracción másica)")
    if ffa > FFA_ALKALI_LIMIT:
        raise ValueError(f"ffa_mass_fraction={ffa:g} excede el límite de la ruta alcalina (~{FFA_ALKALI_LIMIT:g}: «free fatty acid "
                         f"contents greater than about 1% must be pretreated», {VANGERPEN}). Este modelo NO incluye el "
                         "pretratamiento (esterificación ácida o glicerólisis de los AGL, su catalizador, el secado posterior ni la "
                         "pérdida por jabones); refine o pretrate el aceite hasta ≤ 1 % de AGL y vuelva a llamar")
    if ffa + water >= 1.0:
        raise ValueError("ffa_mass_fraction + water_mass_fraction debe ser menor que 1")
    if operating_hours_y is None:
        raise ValueError("falta operating_hours_y: horas de operación al año (convierte kg/h en producción y costos anuales)")
    if R._positive(operating_hours_y, "operating_hours_y", "horas de operación al año") > 8784:
        raise ValueError("operating_hours_y no puede exceder 8784 h (un año)")
    for name, v in (("conversion_per_stage", conversion_per_stage),):
        if v is not None and not (isinstance(v, (int, float)) and 0.0 < v <= 1.0):
            raise ValueError(f"{name} debe estar entre 0 y 1 (conversión del triglicérido en cada uno de los dos reactores)")
    if methanol_excess is not None and not (isinstance(methanol_excess, (int, float)) and methanol_excess >= 0.0):
        raise ValueError("methanol_excess debe ser ≥ 0 (exceso sobre el estequiométrico; 1.0 = 100 % = relación molar 6:1)")
    if catalyst_mass_fraction_of_oil is not None and not (isinstance(catalyst_mass_fraction_of_oil, (int, float))
                                                          and 0.0 < catalyst_mass_fraction_of_oil < 0.05):
        raise ValueError("catalyst_mass_fraction_of_oil debe estar entre 0 y 0.05 (kg de metóxido de sodio puro por kg de "
                         "triglicérido)")
    R._positive(oil_temperature_K, "oil_temperature_K", "temperatura del aceite que llega, K")
    cat_target = (CATALYST_BASE + CATALYST_PER_FFA * ffa if catalyst_mass_fraction_of_oil is None
                  else float(catalyst_mass_fraction_of_oil))
    prices = dict(prices or {})
    for k in list(prices):
        if k in PRICE_ALIASES:
            prices[PRICE_ALIASES[k]] = prices.pop(k)
    tea = dict(tea or {})
    X = 0.90 if conversion_per_stage is None else float(conversion_per_stage)
    excess = 1.0 if methanol_excess is None else float(methanol_excess)

    with R.biosteam_session() as (bst, tmo, caught):
        from biorefineries.biodiesel import create_transesterification_and_biodiesel_separation_system
        from biorefineries.tea import ConventionalEthanolTEA
        chems, tag, ester = _chemicals(tmo, key)
        bst.settings.set_thermo(chems)
        target_ce, cost_basis, cost_year = R.resolve_cepci(year, cepci, R.BIOSTEAM_DEFAULT_CE,
                                                           "valor por omisión de BioSTEAM, año 2017 en la tabla de IDAES", 2017)
        bst.CE = target_ce
        bst.PowerUtility.price = ELECTRICITY_USD_KWH
        for a in bst.HeatUtility.heating_agents:
            if "steam" in a.ID:
                a.regeneration_price = STEAM_USD_KG * 18.01528
        for a in bst.HeatUtility.cooling_agents:           # no sourced price: cooling is reported as duty, at zero cost
            a.regeneration_price = 0.0
            a.heat_transfer_price = 0.0
        bst.main_flowsheet.set_flowsheet(bst.Flowsheet("biodiesel_plant"))
        F = float(oil_feed_kg_h)
        feed = bst.Stream("vegetable_oil", TAG=F * (1.0 - ffa - water), FFA=F * ffa, Water=F * water, units="kg/hr",
                          T=float(oil_temperature_K))
        system = create_transesterification_and_biodiesel_separation_system(ins=feed)
        u, s = system.flowsheet.unit, system.flowsheet.stream
        _restore_makeup_balance(u.B401)
        tag_kg_h = F * (1.0 - ffa - water)
        # NaOCH3 fed is exactly proportional to x_catalyst; first guess from the first reactor, corrected after one solve
        x_cat = cat_target * tag_kg_h / chems.NaOCH3.MW / (3.0 * (1.0 + excess) * tag_kg_h / tag.MW)
        for reactor in (u.R401, u.R402):
            reactor.efficiency = X
            reactor.excess_methanol = excess
            reactor.x_catalyst = x_cat
        product, glycerol = s.biodiesel, s.crude_glycerol
        glycerol.price = CRUDE_GLYCEROL_USD_KG
        basis = [{"parameter": "aceite alimentado", "value": F, "unit": "kg/h", "origin": "usuario"},
                 {"parameter": "ácidos grasos libres", "value": ffa, "unit": "fracción másica", "origin": "usuario"},
                 {"parameter": "humedad del aceite", "value": water, "unit": "fracción másica", "origin": "usuario"},
                 {"parameter": "horas de operación", "value": float(operating_hours_y), "unit": "h/año", "origin": "usuario"},
                 {"parameter": "conversión por reactor (dos en serie)", "value": X, "unit": "fracción",
                  "origin": "usuario" if conversion_per_stage is not None else "sugerido",
                  "source": None if conversion_per_stage is not None else
                  f"Huang et al. 2016: 90 % en cada reactor, 99 % global (aceite de soya / caña), {HUANG_URL}"},
                 {"parameter": "exceso de metanol sobre el estequiométrico", "value": excess, "unit": "fracción",
                  "origin": "usuario" if methanol_excess is not None else "sugerido",
                  "source": None if methanol_excess is not None else
                  f"100 % de exceso (6:1 molar): sección de BioSTEAM (modelo USDA) y Fig. 3 de {VANGERPEN}"},
                 {"parameter": "metóxido de sodio (puro) por kg de triglicérido", "value": cat_target, "unit": "kg/kg",
                  "origin": "usuario" if catalyst_mass_fraction_of_oil is not None else "sugerido",
                  "source": None if catalyst_mass_fraction_of_oil is not None else
                  f"«Sodium methoxide: 0.25% of triglycerides weight» y «[%FFA](0.190) + 0.25 %», {VANGERPEN}. La sección de "
                  f"BioSTEAM trae x = {SECTION_X_CATALYST:g} molar en el metanol (unas 650 veces menos), que no se usa"},
                 {"parameter": "temperatura y tiempo de reacción", "value": REACTOR_T_K, "unit": "K (1 h por reactor)",
                  "origin": "modelo publicado", "source": f"Huang et al. 2016 (60 °C, 1 h), {HUANG_URL}"},
                 {"parameter": "temperatura del aceite que llega", "value": float(oil_temperature_K), "unit": "K",
                  "origin": "usuario" if oil_temperature_K != 298.15 else "supuesto (25 °C)"},
                 {"parameter": "precio de la electricidad", "value": ELECTRICITY_USD_KWH, "unit": "USD/kWh", "origin": "sugerido",
                  "source": HUANG_T4},
                 {"parameter": "precio del vapor comprado (todas las presiones)", "value": STEAM_USD_KG, "unit": "USD/kg",
                  "origin": "sugerido", "source": HUANG_T4 + " (17 USD/t)"}]
        for ID, src in DEFAULT_PRICE_SOURCE.items():
            basis.append({"parameter": f"precio de {ID}", "value": float(s[ID].price), "unit": "USD/kg", "origin": "sugerido",
                          "source": src})
        n_default = len(basis)
        tea_values = {k: v for k, (v, _) in DEFAULT_TEA.items()}
        life = tea_values["plant_life_years"]
        start = year if year is not None else 2023
        tea_obj = ConventionalEthanolTEA(
            system, IRR=tea_values["discount_rate"], duration=(start, start + life), depreciation=tea_values["depreciation"],
            income_tax=tea_values["income_tax"], operating_days=float(operating_hours_y) / 24.0,
            lang_factor=tea_values["lang_factor"], construction_schedule=tuple(tea_values["construction_schedule"]),
            WC_over_FCI=tea_values["working_capital_over_fci"], labor_cost=tea_values["labor_cost_usd_y"],
            fringe_benefits=FRINGE_BENEFITS, property_tax=tea_values["property_tax"],
            property_insurance=tea_values["property_insurance"], supplies=SUPPLIES, maintenance=tea_values["maintenance"],
            administration=tea_values["administration"])
        for k, (v, src) in DEFAULT_TEA.items():
            attr, unit, meaning = R.TEA_KEYS[k]
            basis.append({"parameter": meaning, "value": v, "unit": unit, "origin": "sugerido", "source": src})
        n_default = len(basis)
        R.apply_prices(prices, system, bst, basis)
        R.apply_tea(tea, tea_obj, basis)
        overridden = {b["parameter"] for b in basis[n_default:]}
        for b in basis[:n_default]:
            if b["parameter"] in overridden:
                b["origin"] += " (sustituido por el usuario)"
        oil_price_given = bool(feed.price)
        sim_error = None
        try:
            system.simulate()
            fed = float(s.catalyst.imass["NaOCH3"]) / tag_kg_h
            if fed > 0 and abs(fed / cat_target - 1.0) > 1e-9:
                for reactor in (u.R401, u.R402):
                    reactor.x_catalyst = x_cat * cat_target / fed
                system.simulate()
        except Exception as e:              # reported, never hidden
            sim_error = f"{type(e).__name__}: {e}"
        out = R.extract_results(bst, system, tea_obj, feed, product, cost_basis, target_ce, cost_year,
                                "factores de módulo desnudo por tipo de equipo que trae cada unidad")
        hours = float(operating_hours_y)
        tag_in = float(feed.imass["TAG"])
        fresh_methanol = float(s.methanol.imass["Methanol"] + s.catalyst.imass["Methanol"])
        methanol_to_reactors = float(u.R401.ins[1].imass["Methanol"] + u.R402.ins[1].imass["Methanol"])
        makeup_residual = abs(fresh_methanol + float(u.D401.outs[0].imass["Methanol"]) - methanol_to_reactors) / methanol_to_reactors
        wastewater = s.wastewater
        cold = product.copy()
        cold.T = 288.15
        rho15 = float(cold.rho)
        yields = {
            "biodiesel_stream_kg_per_kg_oil": float(product.F_mass / F),
            "ester_kg_per_kg_oil": float(product.imass["Biodiesel"] / F),
            "ester_kg_per_kg_triglyceride": float(product.imass["Biodiesel"] / tag_in),
            "ester_stoichiometric_kg_per_kg_triglyceride": float(3.0 * ester.MW / tag.MW),
            "overall_triglyceride_conversion": float(1.0 - sum(st.imass["TAG"] for st in system.products) / tag_in),
            "crude_glycerol_kg_per_kg_oil": float(glycerol.F_mass / F),
            "glycerol_pure_kg_per_kg_oil": float(glycerol.imass["Glycerol"] / F),
            "glycerol_stoichiometric_kg_per_kg_triglyceride": float(chems.Glycerol.MW / tag.MW),
            "crude_glycerol_mass_fraction_glycerol": float(glycerol.imass["Glycerol"] / glycerol.F_mass),
            "methanol_fresh_kg_per_kg_oil": fresh_methanol / F,
            "methanol_stoichiometric_kg_per_kg_triglyceride": float(3.0 * chems.Methanol.MW / tag.MW),
            "sodium_methoxide_kg_per_kg_oil": float(s.catalyst.imass["NaOCH3"] / F),
            "sodium_methoxide_kg_per_kg_triglyceride": float(s.catalyst.imass["NaOCH3"] / tag_in),
            "hydrochloric_acid_solution_kg_per_kg_oil": float(s.HCl.F_mass / F),
            "sodium_hydroxide_kg_per_kg_oil": float(s.NaOH.F_mass / F),
            "wash_water_kg_per_kg_oil": float(s.biodiesel_wash_water.F_mass / F),
            "wastewater_kg_per_kg_oil": float(wastewater.F_mass / F),
            "biodiesel_mass_fractions": {c.ID: float(product.imass[c.ID] / product.F_mass) for c in product.available_chemicals
                                         if product.imass[c.ID] / product.F_mass > 1e-6},
            "ffa_fate_fraction": ({st.ID: float(st.imass["FFA"] / (F * ffa)) for st in system.products
                                   if st.imass["FFA"] > 1e-9 * F} if ffa > 0 else {}),
            "biodiesel_density_15C_kg_m3": rho15,
            "biodiesel_L_y": float(product.F_mass * hours / rho15 * 1e3),
        }
        oil_model = {"oil": key, "description": OILS[key]["label"], "triglyceride": tag.ID, "triglyceride_formula": tag.formula,
                     "triglyceride_MW_g_mol": float(tag.MW), "ester": OILS[key]["ester"], "ester_formula": ester.formula,
                     "ester_MW_g_mol": float(ester.MW), "pseudo_component": key != "tripalmitin",
                     "triglyceride_Tm_K": R._f(getattr(tag, "Tm", None))}
        bst_warnings = R.translate_warnings(caught)
        version = f"BioSTEAM {bst.__version__}, biorefineries 2.35.1"

    econ, pc = out["economics"], out["production_cost"]
    msp = econ["minimum_selling_price_usd_kg"]
    econ["minimum_selling_price_usd_per_L"] = msp * rho15 / 1e3 if msp is not None else None
    pc["unit_production_cost_usd_per_L"] = (pc["unit_production_cost_usd_kg"] * rho15 / 1e3
                                            if pc["unit_production_cost_usd_kg"] is not None else None)
    oil_cost = sum(r["cost_usd_y"] for r in out["operating_cost"]["materials"] if r["stream"] == "vegetable_oil")
    gross = out["operating_cost"]["annual_operating_cost_usd_y"] + pc["straight_line_depreciation_usd_y"]
    pc["oil_share_of_gross_production_cost"] = oil_cost / gross if oil_price_given and gross > 0 else None
    pc["glycerol_credit_share_of_gross_production_cost"] = (out["operating_cost"]["coproduct_credit_usd_y"] / gross
                                                            if gross > 0 else None)
    warns = list(bst_warnings)
    if not oil_price_given:
        econ["minimum_selling_price_excluding_oil_usd_kg"] = msp
        econ["minimum_selling_price_excluding_oil_usd_per_L"] = econ["minimum_selling_price_usd_per_L"]
        pc["unit_production_cost_excluding_oil_usd_kg"] = pc["unit_production_cost_usd_kg"]
        for d, k in ((econ, "minimum_selling_price_usd_kg"), (econ, "minimum_selling_price_usd_per_L"), (econ, "npv_usd"),
                     (econ, "irr"), (pc, "unit_production_cost_usd_kg"), (pc, "unit_production_cost_usd_per_L"), (pc, "net_cost_usd_y")):
            d[k] = None
        warns.append("no se dio el precio del aceite (prices={'oil': USD/kg}) y no se sugiere ninguno: se entregan los costos y "
                     "el precio mínimo SIN aceite (margen de transformación); VPN, TIR y costo de producción completos quedan "
                     "en None")
    if sim_error:
        out["evidence"]["verified"] = False
        out["evidence"]["reasons"].insert(0, f"la simulación de BioSTEAM no convergió o falló ({sim_error}); los números "
                                             "siguientes NO son un balance resuelto")
    ev = out["evidence"]
    ev["methanol_makeup_balance_relative_residual"] = makeup_residual
    if makeup_residual > R.MASS_CLOSURE_TOL:
        ev["verified"] = False
        ev["reasons"].append(f"el metanol fresco más el reciclado no iguala el alimentado a los reactores (residuo "
                             f"{makeup_residual:.2e})")
    ev["ester_yield_over_stoichiometric"] = yields["ester_kg_per_kg_triglyceride"] / yields["ester_stoichiometric_kg_per_kg_triglyceride"]
    if ev["ester_yield_over_stoichiometric"] > 1.0 + 1e-6:
        ev["verified"] = False
        ev["reasons"].append("el rendimiento de éster excede el estequiométrico")
    if FFA_NEGLIGIBLE < ffa <= FFA_ALKALI_LIMIT:
        warns.append(f"AGL entre 0.5 % y 1 %: la ruta alcalina funciona pero con más jabón y catalizador ({VANGERPEN})")
    if ffa > 0:
        fate = ", ".join(f"{100 * v:.0f} % en {k}" for k, v in yields["ffa_fate_fraction"].items())
        warns.append("los ácidos grasos libres no reaccionan en este modelo (no se forma jabón ni agua): se tratan como ácido "
                     f"oleico inerte que sale {fate}; el catalizador adicional que piden sí se suma (0.190 × %AGL, "
                     "NREL/SR-510-36244)")
    capacity_L_y = yields["biodiesel_L_y"]
    if "labor_cost_usd_y" not in tea and not 0.5 <= capacity_L_y / HUANG_CAPACITY_L_Y <= 2.0:
        warns.append(f"la mano de obra sugerida (2.5 MM USD/año, 50 empleados) es de una planta de 96 millones de L/año; esta "
                     f"produce {capacity_L_y / 1e6:.1f} millones de L/año: dé tea['labor_cost_usd_y']")
    if year is not None or cepci is not None:
        warns.append(f"el capital está a {cost_basis}; los precios sugeridos son de 2013 (Huang et al. 2016): dé los precios del "
                     "año de su estudio")
    assumptions = [
        "diagrama: sección publicada de BioSTEAM (dos reactores agitados en serie con centrífuga intermedia, lavado con agua, "
        "secado al vacío, columna de recuperación de metanol, concentración del glicerol al 80 %); catalizador metóxido de "
        "sodio al 25 % en metanol; relaciones de ácido y sosa del modelo USDA que trae la sección",
        "las centrífugas reparten por factores fijos de la sección publicada (derivados para aceite de soya/caña), no por "
        "equilibrio líquido-líquido",
        "precios, tasa de descuento, impuestos, mano de obra y factor de Lang SUGERIDOS con fuente (ver design_basis); "
        "sustitúyalos con prices= y tea=",
        "BioSTEAM 2.54 anula el cálculo de la unidad de balance B401 (reposición de metanol y catalizador); se restituye en "
        "esta instancia con la misma función y argumentos que documenta la unidad (thermosteam.separations.material_balance)",
        "agua de enfriamiento y agua helada a costo cero (no hay precio con fuente); sus cargas térmicas sí se reportan",
        f"costos de equipo en {cost_basis}; FCI = factor de Lang × costo de compra",
    ]
    not_included = [
        "almacenamiento de 25 días de materia prima y producto, recepción, edificios y servicios auxiliares (caldera, torre de "
        "enfriamiento, tratamiento de efluentes): solo están los tanques de proceso de la sección",
        "pretratamiento del aceite (desgomado, neutralización, esterificación ácida de AGL) y extracción del aceite de la semilla",
        "formación de jabones, pérdidas por emulsión y purificación del glicerol más allá del 80 %",
        "cinética de la transesterificación: la conversión por reactor es un dato, no un resultado",
        "cumplimiento de norma del biodiésel (ASTM D6751 / EN 14214): no se evalúa",
    ]
    if key == "triolein":
        assumptions.append("aceite = pseudo-componente «TriOlein» de BioSTEAM (tripalmitina + 3 grupos CH=CH, 900 kg/m³ constantes, cp "
                           "y viscosidad de la tripalmitina, Hf −1776 kJ/mol); la trioleína de la base de compuestos solo tiene "
                           "estimaciones y no se usa. Éster = oleato de metilo (ecuación de estado de referencia). Un aceite real "
                           "(soya, canola, girasol) es una mezcla: aquí es trioleína pura")
    elif key == "tripalmitin":
        assumptions.append("aceite = tripalmitina de la base de compuestos (presión de vapor Landolt-Börnstein, cp Zábranský, "
                           "viscosidad Viswanath-Natarajan); éster = palmitato de metilo")
        warns.append(f"la tripalmitina funde a {oil_model['triglyceride_Tm_K']:.1f} K, por encima de los {REACTOR_T_K:.2f} K del "
                     "reactor: una grasa totalmente saturada no es líquida en estas condiciones; úsela solo como caso límite")
    else:
        assumptions.append(
            "ACEITE DE RICINO = PSEUDO-TRIGLICÉRIDO DECLARADO: fórmula y masa molar de la trirricinoleína (C57H104O9, "
            f"{oil_model['triglyceride_MW_g_mol']:.2f} g/mol; CAS 2540-54-7 no existe en la base de compuestos). TODAS las demás "
            "propiedades son PRESTADAS del «TriOlein» de BioSTEAM en base másica: densidad (900 kg/m³), calor específico, "
            "viscosidad, tensión superficial, conductividad térmica, grupos UNIFAC (sin el OH) y entalpía de formación")
        assumptions.append(
            f"ÉSTER DE RICINO = pseudo-componente con fórmula y masa molar del ricinoleato de metilo (C19H36O3, "
            f"{oil_model['ester_MW_g_mol']:.2f} g/mol) y propiedades PRESTADAS del oleato de metilo en base másica (densidad, cp, "
            "viscosidad, presión de vapor, calor de vaporización, grupos UNIFAC, entalpía de formación); por eso el calor de "
            "reacción es el de la trioleína. Los rendimientos por kg sí son los de la estequiometría del ricino")
        warns.append(f"el biodiésel de ricino real es mucho más viscoso y denso que el modelado: 15.65 mm²/s a 40 °C (límites 6 "
                     f"ASTM D6751 y 5 EN 14214) y 0.9376 g/cm³ a 15.5 °C (límite 0.860-0.900); el aceite tiene ~90 % de ácido "
                     f"ricinoleico hidroxilado. Puro no cumple norma: se usa en mezcla ({CASTOR_PROPS})")
        warns.append("el grupo OH del ricinoleato forma puentes de hidrógeno con metanol y glicerol y aumenta su solubilidad mutua; "
                     "con mucho metanol la diferencia de densidad entre fases tiende a cero y la separación del glicerol se "
                     f"alarga (equilibrio observado a las 24 h): las centrífugas de este modelo son optimistas ({CASTOR_SEP})")
        warns.append("la conversión por reactor sugerida (90 %) es de aceite de soya/caña; para ricino dé la suya "
                     "(conversion_per_stage)")
        not_included += ["equilibrio líquido-líquido éster de ricino-glicerol-metanol y la evaporación de metanol previa a la "
                         "separación que suele requerir",
                         "efecto de la alta viscosidad del aceite y del éster de ricino en bombas, intercambiadores y agitación",
                         "propiedades reales del ricinoleato de metilo (la base de compuestos solo tiene estimaciones)"]
    return {"plant": "biodiesel_plant", "description": f"Biodiésel por transesterificación alcalina de {OILS[key]['label']}",
            "oil_model": oil_model, "design_basis": basis, "yields": yields, **out,
            "warnings": warns, "assumptions": assumptions, "not_included": not_included,
            "method": f"sección biorefineries.biodiesel (create_transesterification_and_biodiesel_separation_system) resuelta con "
                      f"{version}: balances secuenciales modulares, dimensionamiento y costo por unidad, flujo de caja descontado "
                      "(ConventionalEthanolTEA de BioSTEAM); estado global de BioSTEAM restaurado al salir"}
