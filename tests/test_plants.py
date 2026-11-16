"""Block M26_B: published reference biorefineries (`reference_biorefinery`) and vegetable-oil biodiesel (`biodiesel_plant`).

Reference numbers. Every one was READ on 2026-11-16 in the source cited; the bands asserted are the deviations MEASURED
that day with BioSTEAM 2.54.0 + biorefineries 2.35.1 (nothing was tuned; the bad ones are asserted as bad on purpose, so a
future package version that changes them is noticed).

  [HUM] Humbird et al. 2011, NREL/TP-5100-47764, «Process Design and Economics for Biochemical Conversion of
        Lignocellulosic Biomass to Ethanol», Figure ES-1, Tables 28 and 34 (2007 USD).
        https://www.osti.gov/servlets/purl/1013269
  [HUA] Huang, Long & Singh 2016, Biofuels Bioprod. Bioref. 10:299-315, Tables 4, 5, 6, 8 and Fig. 3 text (sugarcane with 0 %
        lipids; soybean and lipid-cane biodiesel). https://www.osti.gov/servlets/purl/1437267
  [KWI] Kwiatkowski, McAloon, Taylor & Johnston 2006, Ind. Crops Prod. 23:288-296 — ABSTRACT only (full text is paywalled).
        https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:IND43906338&resultType=core&format=json
  [HAA] Haas, McAloon, Yee & Foglia 2006, Bioresource Technology 97:671-678 — ABSTRACT only (full text is paywalled; no
        open USDA copy was found). https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:15935657&resultType=core&format=json
  [VG]  Van Gerpen, Shanks, Pruszko, Clements & Knothe 2004, «Biodiesel Production Technology», NREL/SR-510-36244:
        Fig. 3 mass balance, FFA limit of the alkali route, catalyst amounts. https://www.osti.gov/servlets/purl/15008801
  [ACS] Rodríguez-Ramos et al. 2026, ACS Omega, doi 10.1021/acsomega.5c10172 (PMC13347325): castor biodiesel properties.
Not opened: Cortés-Peña et al. 2020 (ACS returns 403 to automated clients); its two benchmarks [HUM] and [HUA] were read instead.
"""
from __future__ import annotations

import json
import math

import pytest

from packages.dwsim_client import tools_plants as TP
from packages.plants import biodiesel as BD
from packages.plants import reference as RF

# ------------------------------------------------------------------ published values (source, as printed) and measured bands
HUMBIRD = {  # [HUM] Fig. ES-1 / Table 28 / Table 34          value        band of the measured deviation, %
    "minimum_selling_price_usd_per_gal": (2.15, (-12.0, -5.0)),      # measured -8.4 %
    "ethanol_MMgal_y": (61.0, (-3.5, 0.0)),                          # -1.6 %
    "ethanol_gal_per_dry_ton": (79.0, (-3.5, 0.0)),                  # -1.5 %
    "installed_equipment_cost_usd": (232.0e6, (-22.0, -12.0)),       # -17 %
    "TDC_usd": (250.4e6, (15.0, 30.0)),                              # +23 % (BioSTEAM's TDC includes more than NREL's)
    "FCI_usd": (400.6e6, (-24.0, -12.0)),                            # -18 %
    "TCI_usd": (422.5e6, (-24.0, -12.0)),                            # -19 %
    "surplus_electricity_kWh_per_gal": (1.8, (80.0, 140.0)),         # +110 %: NOT reproduced
}
HUANG_SUGARCANE = {  # [HUA] Tables 6 and 8, Fig. 3 text, sugarcane 0 % lipids
    "ethanol_ML_y": (141.0, (-4.5, 0.0)),                            # -2.4 %
    "ethanol_L_per_t_feed": (88.4, (-5.0, 0.0)),                     # -2.7 %
    "purchase_cost_usd": (50.3e6, (25.0, 40.0)),                     # +32 %
    "FCI_usd": (151.0e6, (25.0, 40.0)),                              # +32 %
    "TCI_usd": (158.5e6, (25.0, 40.0)),                              # +32 %
    "surplus_electricity_kWh_per_t_feed": (61.0, (150.0, 260.0)),    # +204 %: NOT reproduced
    "unit_production_cost_usd_per_L": (0.48, (-23.0, -8.0)),         # -15 % (consequence of the electricity credit)
    "feed_t_y": (1.6e6, (-0.1, 0.1)),
}
KWIATKOWSKI = {  # [KWI] abstract
    "ethanol_ML_y": (151.0, (-2.0, 2.0)),                            # -0.3 %
    "ethanol_kg_y": (119e6, (-2.0, 2.0)),                            # -0.2 %
}
KWI_COST = {0.071: 0.235, 0.125: 0.365}        # [KWI] abstract: corn price USD/kg -> ethanol production cost USD/L
# [HAA] abstract
HAAS_CAPACITY_L_Y = 37_854_118.0
HAAS_CAPITAL_USD = 11.3e6
HAAS_OIL_USD_KG = 0.52
HAAS_COST_USD_L = 0.53
HAAS_OIL_SHARE = 0.88
HAAS_SLOPE_OIL = (0.075, 0.01)                 # USD/gal of biodiesel per USD/lb of oil  -> 7.5 lb oil per gal
HAAS_SLOPE_GLYCEROL = (0.0085, 0.01)           # USD/gal per USD/lb of 80 % glycerol     -> 0.85 lb per gal
LB, GAL = 0.45359237, 3.785411784
# [VG] Fig. 3: 100 lb oil + 21.71 lb methanol -> 100.45 lb biodiesel + 10.40 lb glycerol + 10.86 lb excess methanol
VG_ESTER, VG_GLYCEROL, VG_METHANOL = 1.0045, 0.1040, 0.2171 - 0.1086
# [HUA] Table 6, «Biodiesel production» equipment cost, lipid-cane 20 % lipids, 96 million L/yr in 200 days
HUANG_BIODIESEL_SECTION_USD = 5.3e6
OIL_HAAS = dict(ffa_mass_fraction=0.0, water_mass_fraction=0.0005, operating_hours_y=8000.0)


def _conversion(X: float) -> float:
    """Analytical: stage 1 converts X; centrifuge C401 sends 1 % of the lipid to the glycerol phase (split 0.99 of the
    published section); stage 2 converts X of the rest."""
    return 1.0 - (1.0 - X) * (0.01 + 0.99 * (1.0 - X))


def _dev(rows: list[dict]) -> dict:
    return {r["quantity"]: r for r in rows}


def _check_bands(result: dict, table: dict) -> None:
    rows = _dev(result["published_benchmark"]["values"])
    for quantity, (published, (lo, hi)) in table.items():
        row = rows[quantity]
        assert row["published"] == published
        assert lo <= row["deviation_pct"] <= hi, (quantity, row)


def _common(r: dict) -> None:
    json.dumps(r)
    for k in ("warnings", "assumptions", "not_included"):
        assert isinstance(r[k], list)
    assert isinstance(r["method"], str) and "BioSTEAM" in r["method"]
    ev = r["evidence"]
    assert ev["verified"] is True, ev["reasons"]
    assert ev["mass_balance_relative_residual"] <= RF.MASS_CLOSURE_TOL
    assert 0.0 < ev["first_law_useful_energy_over_energy_in"] < 1.0
    assert ev["npv_at_minimum_price_over_tci"] <= RF.NPV_ZERO_TOL
    for e in r["equipment"]:
        assert 0.0 < e["base_purchase_cost_usd"] <= e["purchase_cost_usd"] and e["F_BM"] >= 1.0 and e["cepci"] > 0


@pytest.fixture(scope="module")
def sugarcane():
    return RF.reference_biorefinery("sugarcane_ethanol")


@pytest.fixture(scope="module")
def cornstover():
    return RF.reference_biorefinery("cornstover_ethanol")


@pytest.fixture(scope="module")
def corn():
    return RF.reference_biorefinery("corn_ethanol")


@pytest.fixture(scope="module")
def haas():
    """Triolein plant at the capacity of [HAA] (37.85 million L/yr) with their soybean-oil price."""
    probe = BD.biodiesel_plant(oil_feed_kg_h=4000.0, oil="triolein", **OIL_HAAS)
    feed = 4000.0 * HAAS_CAPACITY_L_Y / probe["yields"]["biodiesel_L_y"]
    return BD.biodiesel_plant(oil_feed_kg_h=feed, oil="triolein", prices={"oil": HAAS_OIL_USD_KG}, **OIL_HAAS)


# ------------------------------------------------------------------------------------------------ reference biorefineries

def test_catalogue():
    c = TP.list_reference_plants()
    json.dumps(c)
    assert {p["plant"] for p in c["plants"]} == {"sugarcane_ethanol", "corn_ethanol", "cornstover_ethanol"}
    assert set(c["oils"]) == {"triolein", "tripalmitin", "castor"}
    assert c["evidence"]["verified"] is True


def test_cornstover_against_humbird_2011(cornstover):
    _common(cornstover)
    _check_bands(cornstover, HUMBIRD)
    assert cornstover["economics"]["discount_rate"] == pytest.approx(0.10)       # [HUM] Table 33
    assert cornstover["economics"]["plant_life_years"] == 30                      # [HUM] Table 33
    assert cornstover["economics"]["operating_hours_y"] == pytest.approx(8410.0, rel=1e-3)   # [HUM] Table 34
    assert cornstover["products"]["feed_kg_h"] == pytest.approx(104_167.0, rel=1e-3)         # [HUM] wet feed
    assert cornstover["capital"]["cepci"] == pytest.approx(525.4) and cornstover["capital"]["cost_year"] == 2007
    assert any("surplus_electricity" in w for w in cornstover["warnings"])        # the bad deviation travels with every run


def test_sugarcane_against_huang_2016(sugarcane):
    _common(sugarcane)
    _check_bands(sugarcane, HUANG_SUGARCANE)
    basis = {b["parameter"]: b for b in sugarcane["design_basis"]}
    assert basis["factor de Lang"]["value"] == 3.0                                # [HUA] Table 6
    assert basis["precio de la electricidad"]["value"] == pytest.approx(0.065)    # [HUA] Table 4
    assert sugarcane["economics"]["plant_life_years"] == 20 and sugarcane["economics"]["income_tax"] == 0.35   # Table 5
    assert sugarcane["economics"]["operating_hours_y"] == pytest.approx(4800.0)   # 200 days
    assert sugarcane["capital"]["FCI_usd"] == pytest.approx(3.0 * sugarcane["capital"]["purchase_cost_usd"])
    assert sugarcane["utilities"]["net_electricity_export_W"] > 0


def test_corn_against_kwiatkowski_2006_abstract(corn):
    _common(corn)
    _check_bands(corn, KWIATKOWSKI)
    assert any("_ins_size_is_fixed" in a for a in corn["assumptions"])            # the compatibility shim is declared
    assert corn["utilities"]["net_electricity_export_W"] < 0                      # buys power
    measured = {}
    for corn_price, published in KWI_COST.items():
        r = RF.reference_biorefinery("corn_ethanol", prices={"corn": corn_price})
        cost_L = r["production_cost"]["unit_production_cost_usd_kg"] * RF.ETHANOL_KG_PER_L
        measured[corn_price] = cost_L
        assert -12.0 <= 100.0 * (cost_L / published - 1.0) <= 0.0, (corn_price, cost_L)   # measured -6.3 % and -5.4 %
    slope = (measured[0.125] - measured[0.071]) / (0.125 - 0.071)                 # kg corn per L ethanol
    published_slope = (0.365 - 0.235) / (0.125 - 0.071)
    assert abs(slope / published_slope - 1.0) < 0.08                              # measured -3.7 %


def test_equipment_feeds_capital_investment(cornstover, sugarcane):
    from packages.dwsim_client import tools_economics as TE
    zero = dict(site_preparation=0, service_facilities=0, allocated_costs=0, contingency_and_fee=0, land=0, royalties=0,
                startup=0, working_capital=0)
    for r in (cornstover, sugarcane):
        ci = TE.capital_investment(r["equipment"], zero)
        total = ci["total_bare_module_cost_usd"]
        assert ci["total_purchase_cost_usd"] == pytest.approx(r["capital"]["purchase_cost_usd"], rel=1e-9)
        assert total == pytest.approx(r["capital"]["installed_equipment_cost_bm_usd"], rel=1e-9)
        assert sum(e["purchase_cost_usd"] for e in r["equipment"]) == pytest.approx(r["capital"]["purchase_cost_usd"], rel=1e-9)
        assert sum(s["purchase_cost_usd"] for s in r["sections"]) == pytest.approx(r["capital"]["purchase_cost_usd"], rel=1e-9)
    assert cornstover["evidence"]["installed_cost_sum_vs_tea_relative_difference"] < 1e-9


def test_results_do_not_depend_on_call_order(sugarcane, cornstover, corn):
    """A, B, A (with other plants and a biodiesel run in between): the second A is the first one."""
    BD.biodiesel_plant(oil_feed_kg_h=1500.0, oil="castor", ffa_mass_fraction=0.002, water_mass_fraction=0.0,
                       operating_hours_y=7000.0)
    for first in (sugarcane, cornstover, corn):
        again = RF.reference_biorefinery(first["plant"])
        for block, key in (("capital", "purchase_cost_usd"), ("capital", "TCI_usd"), ("products", "main_product_kg_h"),
                           ("economics", "minimum_selling_price_usd_kg"), ("economics", "irr"),
                           ("utilities", "net_electricity_export_W"), ("production_cost", "unit_production_cost_usd_kg")):
            assert again[block][key] == pytest.approx(first[block][key], rel=1e-9), (first["plant"], key)
        for x, y in zip(again["published_benchmark"]["values"], first["published_benchmark"]["values"]):
            assert x["quantity"] == y["quantity"] and x["model"] == pytest.approx(y["model"], rel=1e-9)


def test_rescaling_and_overrides(sugarcane):
    f0 = sugarcane["products"]["feed_kg_h"]
    same = RF.reference_biorefinery("sugarcane_ethanol", feed_flow_kg_h=f0)
    half = RF.reference_biorefinery("sugarcane_ethanol", feed_flow_kg_h=f0 / 2)
    double = RF.reference_biorefinery("sugarcane_ethanol", feed_flow_kg_h=2 * f0)
    for r in (same, half, double):
        _common(r)
    assert same["capital"]["purchase_cost_usd"] == pytest.approx(sugarcane["capital"]["purchase_cost_usd"], rel=2e-3)
    p = [r["products"]["main_product_kg_h"] for r in (half, same, double)]
    assert p[0] == pytest.approx(p[1] / 2, rel=5e-3) and p[2] == pytest.approx(2 * p[1], rel=5e-3)   # product ∝ feed
    c = [r["capital"]["purchase_cost_usd"] for r in (half, same, double)]
    assert c[0] < c[1] < c[2] and c[2] / c[1] < 2.0 and c[1] / c[0] < 2.0                              # economy of scale
    m = [r["economics"]["minimum_selling_price_usd_kg"] for r in (half, same, double)]
    assert m[0] > m[1] > m[2]
    assert any("reescalada" in w for w in double["warnings"])
    # prices: a higher product price raises IRR and NPV and leaves the minimum selling price alone
    rich = RF.reference_biorefinery("sugarcane_ethanol", prices={"ethanol": 1.0})
    assert rich["economics"]["irr"] > same["economics"]["irr"] and rich["economics"]["npv_usd"] > same["economics"]["npv_usd"]
    assert rich["economics"]["minimum_selling_price_usd_kg"] == pytest.approx(same["economics"]["minimum_selling_price_usd_kg"])
    # a lower discount rate lowers the minimum selling price; IRR does not move
    cheap = RF.reference_biorefinery("sugarcane_ethanol", tea={"discount_rate": 0.10})
    assert cheap["economics"]["minimum_selling_price_usd_kg"] < same["economics"]["minimum_selling_price_usd_kg"]
    assert cheap["economics"]["irr"] == pytest.approx(same["economics"]["irr"], rel=1e-9)
    # IRR identity: discounting at the IRR gives NPV = 0
    at_irr = RF.reference_biorefinery("sugarcane_ethanol", tea={"discount_rate": same["economics"]["irr"]})
    assert abs(at_irr["economics"]["npv_usd"]) < 1e-6 * at_irr["capital"]["TCI_usd"]
    # more hours on the same hardware: same capital, more product, lower minimum price
    longer = RF.reference_biorefinery("sugarcane_ethanol", operating_hours_y=6000.0)
    assert longer["capital"]["purchase_cost_usd"] == pytest.approx(same["capital"]["purchase_cost_usd"], rel=1e-9)
    assert longer["economics"]["minimum_selling_price_usd_kg"] < same["economics"]["minimum_selling_price_usd_kg"]
    # cost index: purchase cost is proportional to CEPCI
    doubled = RF.reference_biorefinery("sugarcane_ethanol", cepci=2 * 607.5)
    assert doubled["capital"]["purchase_cost_usd"] == pytest.approx(2 * same["capital"]["purchase_cost_usd"], rel=1e-9)
    assert any("mezcla de años" in w for w in doubled["warnings"])
    y2023 = RF.reference_biorefinery("sugarcane_ethanol", year=2023)
    assert y2023["capital"]["purchase_cost_usd"] == pytest.approx(same["capital"]["purchase_cost_usd"] * 797.9 / 607.5, rel=1e-9)
    origins = {b["parameter"]: b["origin"] for b in cheap["design_basis"] if b["origin"] == "usuario"}
    assert "tasa de descuento del VPN y del precio mínimo" in origins


def test_reference_refusals():
    for kwargs, fragment in (
            (dict(plant="palm_oil"), "no está disponible"),
            (dict(plant="sugarcane_ethanol", feed_flow_kg_h=-5.0), "feed_flow_kg_h"),
            (dict(plant="sugarcane_ethanol", operating_hours_y=9000.0), "8784"),
            (dict(plant="sugarcane_ethanol", prices={"etanol": 1.0}), "claves válidas"),
            (dict(plant="sugarcane_ethanol", tea={"tasa": 0.1}), "no es un parámetro reconocido"),
            (dict(plant="cornstover_ethanol", tea={"lang_factor": 3.0}), "Lang"),
            (dict(plant="sugarcane_ethanol", year=2030), "cepci")):
        with pytest.raises(ValueError, match=fragment):
            RF.reference_biorefinery(**kwargs)


def test_global_state_is_restored(sugarcane, haas):
    """BioSTEAM's globals and thermo's error flag are as they were; another engine of the repo still works afterwards."""
    import biosteam as bst
    import thermosteam as tmo
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    bst.CE, bst.PowerUtility.price = 612.3, 0.111
    agents = (bst.HeatUtility.heating_agents, bst.HeatUtility.cooling_agents)
    steam_price = bst.HeatUtility.get_agent("low_pressure_steam").regeneration_price
    sup = dict(bst.stream_utility_prices)
    mine = bst.Flowsheet("mine_M26B")
    bst.main_flowsheet.set_flowsheet(mine)
    tmo.settings.set_thermo(["Water", "Ethanol"], cache=True)
    thermo_before = tmo.settings.get_thermo()
    # set AFTER set_thermo: thermosteam flips this flag to True whenever it builds chemicals, not only on import
    TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = False        # thermo's own default, what the repo expects
    try:
        RF.reference_biorefinery("cornstover_ethanol", prices={"electricity": 0.2, "steam": 0.03}, cepci=700.0)
        BD.biodiesel_plant(oil_feed_kg_h=900.0, oil="triolein", ffa_mass_fraction=0.0, water_mass_fraction=0.0,
                           operating_hours_y=8000.0, year=2022)
        with pytest.raises(ValueError):
            RF.reference_biorefinery("sugarcane_ethanol", prices={"nada": 1.0})      # also restored after an error
        assert TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR is False
        assert MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR is False
        assert bst.CE == 612.3 and bst.PowerUtility.price == 0.111
        assert bst.HeatUtility.heating_agents is agents[0] and bst.HeatUtility.cooling_agents is agents[1]
        assert bst.HeatUtility.get_agent("low_pressure_steam").regeneration_price == steam_price
        assert dict(bst.stream_utility_prices) == sup
        assert bst.main_flowsheet.ID == "mine_M26B" and tmo.settings.get_thermo() is thermo_before
        assert "biodiesel_plant" not in bst.main_flowsheet.flowsheet.__dict__
        from packages.dwsim_client.tools_properties import compound_properties
        r = compound_properties("R134a", 250.0)
        assert r["identity"] and r["properties_at_T"]
        json.dumps(r, default=float)
    finally:
        bst.process_tools.default_utilities()
        bst.CE = 567.5
        bst.main_flowsheet.set_flowsheet("default")


# ------------------------------------------------------------------------------------------------------------- biodiesel

def test_biodiesel_stoichiometry_against_nrel_mass_balance(haas):
    _common(haas)
    y = haas["yields"]
    assert y["ester_stoichiometric_kg_per_kg_triglyceride"] == pytest.approx(VG_ESTER, rel=2e-4)
    assert y["glycerol_stoichiometric_kg_per_kg_triglyceride"] == pytest.approx(VG_GLYCEROL, rel=2e-3)
    assert y["methanol_stoichiometric_kg_per_kg_triglyceride"] == pytest.approx(VG_METHANOL, rel=2e-3)
    assert y["overall_triglyceride_conversion"] == pytest.approx(_conversion(0.90), rel=1e-9)   # two 90 % stages, [HUA]: «99 %»
    assert y["ester_kg_per_kg_triglyceride"] <= y["ester_stoichiometric_kg_per_kg_triglyceride"] * y["overall_triglyceride_conversion"] * (1 + 1e-9)
    assert 0.97 < haas["evidence"]["ester_yield_over_stoichiometric"] < 0.995
    assert y["methanol_fresh_kg_per_kg_oil"] == pytest.approx(VG_METHANOL * 0.989, rel=0.03)       # consumed ≈ reacted
    assert y["glycerol_pure_kg_per_kg_oil"] <= VG_GLYCEROL
    assert y["crude_glycerol_mass_fraction_glycerol"] == pytest.approx(0.80, abs=5e-3)   # [HAA] 80 % w/w; measured 0.797 (salts)
    assert y["sodium_methoxide_kg_per_kg_triglyceride"] == pytest.approx(0.0025, rel=1e-6)         # [VG] 0.25 % of triglyceride
    assert haas["evidence"]["methanol_makeup_balance_relative_residual"] < 1e-9


def test_biodiesel_against_haas_2006_abstract(haas):
    y, pc, cap = haas["yields"], haas["production_cost"], haas["capital"]
    assert y["biodiesel_L_y"] == pytest.approx(HAAS_CAPACITY_L_Y, rel=1e-4)
    # oil per litre of biodiesel, from the published cost sensitivity (7.5 lb per gal): measured -1.4 %
    oil_kg_per_L = haas["products"]["feed_kg_h"] * 8000.0 / y["biodiesel_L_y"]
    published = HAAS_SLOPE_OIL[0] / HAAS_SLOPE_OIL[1] * LB / GAL
    assert -3.0 < 100 * (oil_kg_per_L / published - 1) < 0.0
    # the same number recovered as a slope of the tool's own production cost
    f = haas["products"]["feed_kg_h"]
    dearer = BD.biodiesel_plant(oil_feed_kg_h=f, oil="triolein", prices={"oil": HAAS_OIL_USD_KG + 0.022}, **OIL_HAAS)
    slope = (dearer["production_cost"]["unit_production_cost_usd_per_L"] - pc["unit_production_cost_usd_per_L"]) / 0.022
    assert slope == pytest.approx(oil_kg_per_L, rel=1e-6)
    assert -4.5 < 100 * (slope / (0.020 / 0.022) - 1) < 0.0          # against the rounded SI figures of the abstract: -2.5 %
    # 80 % glycerol per litre (0.85 lb per gal): measured +12 % (this section recovers ~99 % of the glycerol)
    glycerol_kg_per_L = y["crude_glycerol_kg_per_kg_oil"] * oil_kg_per_L
    assert 5.0 < 100 * (glycerol_kg_per_L / (HAAS_SLOPE_GLYCEROL[0] / HAAS_SLOPE_GLYCEROL[1] * LB / GAL) - 1) < 20.0
    # capital: 11.3 MM USD published; the section (no 25-day storage, no buildings/utilities) gives about half: -47 %
    assert -60.0 < 100 * (cap["FCI_usd"] / HAAS_CAPITAL_USD - 1) < -35.0
    # unit production cost 0.53 USD/L and 88 % of it oil: measured +19 % and 70 % (2013 reagent prices and Huang's labour)
    assert 10.0 < 100 * (pc["unit_production_cost_usd_per_L"] / HAAS_COST_USD_L - 1) < 30.0
    assert 0.62 < pc["oil_share_of_gross_production_cost"] < HAAS_OIL_SHARE
    assert 0.02 < pc["glycerol_credit_share_of_gross_production_cost"] < 0.06       # published: «approximately 6 %»


def test_biodiesel_section_capital_against_huang_2016(haas):
    """[HUA] Table 6: 5.3 MM USD of equipment for the biodiesel area of the 20 %-lipid cane plant (96 million L/yr, 200 days)."""
    feed = haas["products"]["feed_kg_h"] * (96e6 / 4800.0) / (HAAS_CAPACITY_L_Y / 8000.0)
    r = BD.biodiesel_plant(oil_feed_kg_h=feed, oil="triolein", ffa_mass_fraction=0.0, water_mass_fraction=0.0005,
                           operating_hours_y=4800.0, cepci=567.0)
    assert r["yields"]["biodiesel_L_y"] == pytest.approx(96e6, rel=1e-3)
    assert -32.0 < 100 * (r["capital"]["purchase_cost_usd"] / HUANG_BIODIESEL_SECTION_USD - 1) < -12.0   # measured -22 %
    assert r["economics"]["irr"] is None and r["economics"]["minimum_selling_price_usd_kg"] is None      # no oil price given
    assert r["economics"]["minimum_selling_price_excluding_oil_usd_kg"] > 0
    assert any("precio del aceite" in w for w in r["warnings"])


def test_biodiesel_scaling_and_monotonicity(haas):
    f = haas["products"]["feed_kg_h"]
    double = BD.biodiesel_plant(oil_feed_kg_h=2 * f, oil="triolein", prices={"oil": HAAS_OIL_USD_KG}, **OIL_HAAS)
    _common(double)
    for k in ("ester_kg_per_kg_oil", "crude_glycerol_kg_per_kg_oil", "methanol_fresh_kg_per_kg_oil"):
        assert double["yields"][k] == pytest.approx(haas["yields"][k], rel=1e-6)                 # intensive
    ratio = double["capital"]["purchase_cost_usd"] / haas["capital"]["purchase_cost_usd"]
    assert 1.0 < ratio < 2.0
    assert double["production_cost"]["unit_production_cost_usd_per_L"] < haas["production_cost"]["unit_production_cost_usd_per_L"]
    low = BD.biodiesel_plant(oil_feed_kg_h=f, oil="triolein", conversion_per_stage=0.80, prices={"oil": HAAS_OIL_USD_KG},
                             **OIL_HAAS)
    assert low["yields"]["overall_triglyceride_conversion"] == pytest.approx(_conversion(0.80), rel=1e-9)
    assert low["yields"]["ester_kg_per_kg_oil"] < haas["yields"]["ester_kg_per_kg_oil"]
    less = BD.biodiesel_plant(oil_feed_kg_h=f, oil="triolein", methanol_excess=0.5, prices={"oil": HAAS_OIL_USD_KG}, **OIL_HAAS)
    assert less["yields"]["methanol_fresh_kg_per_kg_oil"] == pytest.approx(haas["yields"]["methanol_fresh_kg_per_kg_oil"], rel=0.02)
    assert less["utilities"]["purchased_heating_duty_W"] < haas["utilities"]["purchased_heating_duty_W"]   # less to recover
    ffa = BD.biodiesel_plant(oil_feed_kg_h=f, oil="triolein", ffa_mass_fraction=0.008, water_mass_fraction=0.0005,
                             operating_hours_y=8000.0, prices={"oil": HAAS_OIL_USD_KG})
    assert ffa["yields"]["sodium_methoxide_kg_per_kg_triglyceride"] == pytest.approx(0.0025 + 0.190 * 0.008, rel=1e-6)  # [VG]
    assert ffa["yields"]["ester_kg_per_kg_oil"] < haas["yields"]["ester_kg_per_kg_oil"]
    assert any("AGL entre 0.5 % y 1 %" in w for w in ffa["warnings"])


def test_castor_oil_is_a_declared_pseudo_component(haas):
    f = haas["products"]["feed_kg_h"]
    r = BD.biodiesel_plant(oil_feed_kg_h=f, oil="higuerilla", prices={"oil": HAAS_OIL_USD_KG}, **OIL_HAAS)
    _common(r)
    om, y = r["oil_model"], r["yields"]
    assert om["oil"] == "castor" and om["pseudo_component"] is True
    assert om["triglyceride_formula"] == "C57H104O9" and om["triglyceride_MW_g_mol"] == pytest.approx(933.43, abs=0.02)
    assert om["ester_formula"] == "C19H36O3" and om["ester_MW_g_mol"] == pytest.approx(312.49, abs=0.01)
    # atoms close: C57H104O9 + 3 CH4O = 3 C19H36O3 + C3H8O3
    assert om["triglyceride_MW_g_mol"] + 3 * 32.042 == pytest.approx(3 * om["ester_MW_g_mol"] + 92.094, abs=0.02)
    # stoichiometry of castor, not of triolein: less glycerol and methanol per kg, by the molar-mass ratio
    ratio = 885.43 / 933.43
    assert y["glycerol_stoichiometric_kg_per_kg_triglyceride"] == pytest.approx(haas["yields"]["glycerol_stoichiometric_kg_per_kg_triglyceride"] * ratio, rel=1e-4)
    assert y["methanol_fresh_kg_per_kg_oil"] == pytest.approx(haas["yields"]["methanol_fresh_kg_per_kg_oil"] * ratio, rel=5e-3)
    assert y["ester_kg_per_kg_oil"] < haas["yields"]["ester_kg_per_kg_oil"]
    text = " ".join(r["assumptions"])
    assert "PSEUDO-TRIGLICÉRIDO DECLARADO" in text and "PRESTADAS" in text and "TriOlein" in text
    assert any("15.65" in w and "0.9376" in w for w in r["warnings"])            # [ACS] measured castor biodiesel
    assert any("puentes de hidrógeno" in w for w in r["warnings"])
    assert any("equilibrio líquido-líquido" in n for n in r["not_included"])
    # the borrowed density is the oleate's (mass basis): the model does NOT know castor biodiesel is denser
    assert y["biodiesel_density_15C_kg_m3"] == pytest.approx(haas["yields"]["biodiesel_density_15C_kg_m3"], rel=1e-3)
    assert y["biodiesel_density_15C_kg_m3"] < 937.6


def test_tripalmitin_runs_on_database_properties():
    r = BD.biodiesel_plant(oil_feed_kg_h=2000.0, oil="tripalmitin", prices={"oil": 0.5}, **OIL_HAAS)
    _common(r)
    assert r["oil_model"]["pseudo_component"] is False and r["oil_model"]["triglyceride_MW_g_mol"] == pytest.approx(807.32, abs=0.02)
    assert any("funde" in w for w in r["warnings"])


def test_biodiesel_refusals():
    ok = dict(oil_feed_kg_h=1000.0, oil="triolein", ffa_mass_fraction=0.0, water_mass_fraction=0.0, operating_hours_y=8000.0)
    for change, fragment in (
            (dict(oil_feed_kg_h=None), "falta oil_feed_kg_h"), (dict(oil=None), "falta oil"),
            (dict(ffa_mass_fraction=None), "falta ffa_mass_fraction"), (dict(water_mass_fraction=None), "falta water_mass_fraction"),
            (dict(operating_hours_y=None), "falta operating_hours_y"),
            (dict(ffa_mass_fraction=0.03), "límite de la ruta alcalina"),            # [VG]: above ~1 % FFA pretreat first
            (dict(oil="trilinolein"), "solo tiene estimaciones"), (dict(oil="soya"), "no está disponible"),
            (dict(conversion_per_stage=1.2), "conversion_per_stage"), (dict(methanol_excess=-0.1), "methanol_excess"),
            (dict(catalyst_mass_fraction_of_oil=0.2), "catalyst_mass_fraction_of_oil"),
            (dict(prices={"aceite": 1.0}), "claves válidas"), (dict(tea={"vida": 10}), "no es un parámetro reconocido"),
            (dict(year=2031), "cepci")):
        with pytest.raises(ValueError, match=fragment):
            BD.biodiesel_plant(**{**ok, **change})
    assert math.isclose(BD.FFA_ALKALI_LIMIT, 0.01)


def test_tools_and_wrappers_are_aligned():
    from packages.agent import wrappers_plants as WP
    assert [f.__name__ for f in TP.PLANTS_TOOLS] == [getattr(w, "name", getattr(w, "__name__", None)) for w in WP.AGENT_WRAPPERS]
    for f in TP.PLANTS_TOOLS:
        assert f.__doc__ and len(f.__doc__) > 200
