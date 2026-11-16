"""Reference prices and design basis (bloque M26_P). Offline: the bank is a JSON file; nothing is fetched here.

The numbers asserted in PUBLISHED below were read on 2026-11-16 in the URL of each line (they are the admission case of the
lookup: bank value == number read in the source). Everything else is logic: schema, staleness, fallback, conversions.
"""
import json
from datetime import date

import pytest

from packages.agent import wrappers_reference_prices as W
from packages.costing import reference_prices as RP
from packages.costing import utilities as U
from packages.dwsim_client import tools_reference_prices as T

T0 = date(2026, 11, 16)
BANK = RP.load_bank()
ENTRIES = {e["id"]: e for e in BANK["entries"]}

# (entry id, published value in the published unit) — source, URL, date read 2026-11-16
PUBLISHED = [
    ("fx_fix_mxn_usd", 18.5525),            # Banxico FIX 13-nov-2026, https://www.banxico.org.mx/canales/singleFix.json
    ("banxico_target_rate", 6.25),          # Banxico tasa objetivo 16-nov-2026, https://www.banxico.org.mx/canales/singleTasaObj.json
    ("tiie_28", 6.5147),                    # Banxico TIIE 28 13-nov-2026, https://www.banxico.org.mx/canales/singleTIIE28.json
    ("cetes_28", 6.20),                     # Banxico CETES 28 10-nov-2026, https://www.banxico.org.mx/canales/singleCetes28.json
    ("minimum_wage_mx", 315.04),            # CONASAMI tabla 2026, https://www.gob.mx/cms/uploads/attachment/file/1041076/Tabla_de_Salarios_M_nimos_2026.pdf
    ("minimum_wage_mx_border", 440.87),     # same table (Zona Libre de la Frontera Norte)
    ("imss_average_wage_mx", 664.7),        # IMSS boletín 583/2026, https://mexico.quadratin.com.mx/www/wp-content/uploads/2026/11/Empleo-Octubre-2026.pdf
    ("isr_corporate_mx", 0.30),             # LISR art. 9, https://www.diputados.gob.mx/LeyesBiblio/pdf/LISR.pdf
    ("depreciation_chemical_mx", 0.09),     # LISR art. 35-V, same PDF
    ("depreciation_sugar_oils_milling_mx", 0.05),   # LISR art. 35-I
    ("depreciation_food_beverages_mx", 0.08),       # LISR art. 35-IV
    ("depreciation_other_mx", 0.10),        # LISR art. 35-XIV
    ("depreciation_buildings_mx", 0.05),    # LISR art. 34-I-b
    ("ptu_mx", 0.10),                       # DOF 18-09-2020, https://sidof.segob.gob.mx/notas/docFuente/5600763
    ("sugar_reference_price_mx", 14704.53),  # DOF 30-10-2026, https://sidof.segob.gob.mx/notas/docFuente/5802605
    ("water_federal_rights_mx", 31.8622),   # LFD art. 223-A zona 1 subterránea, https://www.diputados.gob.mx/LeyesBiblio/pdf/LFD.pdf
    ("wastewater_discharge_mx", 51.34),     # LFD art. 277-B-II tipo C, same PDF
    ("water_industrial_culiacan", 40.43),   # JAPAC tarifa industrial 2026, https://japac.gob.mx/servicios/tarifas/
    ("diesel_mx", 26.99),                   # Profeco vía La Crónica 12-oct-2026 (MXN/L), URL in the bank
    ("diesel_us", 6.127),                   # EIA weekly retail 9-nov-2026 (USD/gal), https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm
    ("methanol_us", 1450.0),                # Methanex NA nov-2026 (USD/MT), https://www.methanex.com/about-methanol/pricing/
    ("methanol_world", 700.0),              # Methanex Asia Pacific nov-2026
    ("soybean_oil_world", 1546.0),          # World Bank Pink Sheet 3-nov-2026 (Oct 2026), URL in the bank
    ("palm_oil_world", 1120.0),             # same
    ("corn_yellow_world", 226.5),           # same
    ("raw_sugar_world", 0.43),              # same (USD/kg)
    ("raw_sugar_us", 0.92),                 # same (USD/kg)
    ("ethanol_fuel_us", 1.8525),            # USDA AMS 3616, Iowa FOB-T average (USD/gal), https://www.ams.usda.gov/mnreports/ams_3616.pdf
    ("corn_yellow_us", 4.25),               # USDA AMS 3618 (USD/bu), https://www.ams.usda.gov/mnreports/ams_3618.pdf
    ("ddgs_us", 165.19),                    # USDA AMS 3618 Iowa (USD/short ton)
    ("distillers_corn_oil_us", 65.42),      # USDA AMS 3618 Iowa (cents/lb)
    ("sorghum_us", 3.8175),                 # USDA AMS 2886 Kansas SW (USD/bu), https://www.ams.usda.gov/mnreports/ams_2886.pdf
    ("used_cooking_oil_us", 46.25),         # USDA AMS 3510 yellow grease Minnesota (cents/lb), https://www.ams.usda.gov/mnreports/ams_3510.pdf
    ("biodiesel_b100_us", 5.42),            # AFDC July 2026 (USD/gal), https://afdc.energy.gov/fuels/prices.html
    ("castor_oil_world", 1741.41),          # Procurement Resource Oct 2026 NL CIF (secondary), URL in the bank
    ("castor_oil_us", 1708.05),             # same page, USA FOB
    ("glycerol_us", 1552.38), ("sulfuric_acid_us", 436.35), ("hexane_us", 1571.0), ("caustic_soda_us", 399.13),   # same site
    ("on_stream_hours", 8410.0),            # Humbird 2011 Table 34, https://docs.nlr.gov/docs/fy11osti/47764.pdf
    ("plant_life", 30.0), ("construction_schedule", 3.0), ("working_capital_fraction", 0.05), ("equity_fraction", 0.40),
    ("discount_rate_nrel", 0.10),           # Humbird 2011 Table 33
    ("startup_time", 3),                    # months, same table
    ("zafra_duration_mx", 262.0),           # Cuenta Pública 2024, URL in the bank
    ("location_factor_mx", 0.88),           # IDAES location_factors.json (GitHub raw), URL in the bank
    ("equity_risk_premium_mx", 6.69),       # Damodaran Jan-2026 (%), https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
    ("cepci_2024_06", 798.8),               # toweringskills.com cost indices (secondary)
    ("inflation_mx", 3.75),                 # Banxico, https://www.banxico.org.mx/canales/singleInflacion.json
]


# ---------------------------------------------------------------------------------------------- schema of the bank
@pytest.mark.parametrize("e", BANK["entries"], ids=lambda e: e["id"])
def test_entry_schema(e):
    for k in RP.REQUIRED_FIELDS:
        assert k in e, f"{e['id']}: falta {k}"
    assert e["kind"] in RP.KINDS and e["region"] in RP.REGION_ORDER
    assert isinstance(e["aliases"], list) and e["aliases"] and all(isinstance(a, str) and a for a in e["aliases"])
    assert isinstance(e["value"], (int, float)) and e["value"] > 0
    assert e["url"].startswith("https://")
    assert len(e["source"]) > 20 and e["basis"] and e["unit"] and e["notes"]
    start, end = date.fromisoformat(e["period"]["start"]), date.fromisoformat(e["period"]["end"])
    assert start <= end
    assert e["source_tier"] in ("primaria", "secundaria")
    assert e["currency"] is None or e["unit"].startswith(e["currency"] + "/")
    if "low" in e or "high" in e:
        assert e["low"] <= e["value"] <= e["high"]
    if e["source_tier"] == "secundaria":
        assert "SECUNDARIA" in e["notes"]


def test_every_entry_has_date_read_and_source():
    for e in BANK["entries"]:
        assert e.get("date_read"), f"{e['id']} sin date_read"
        assert date.fromisoformat(e["date_read"]) <= T0
        assert e.get("source") and e.get("url"), f"{e['id']} sin fuente"
    assert len({e["id"] for e in BANK["entries"]}) == len(BANK["entries"])
    assert all(n["why"] and n["searched"] for n in BANK["not_found"])


def test_bank_matches_the_numbers_read_in_the_sources():
    assert {i for i, _ in PUBLISHED} == set(ENTRIES) - {"sugarcane_mx", "soybean_oil_us", "depreciation_renewable_cogeneration_mx"}
    for eid, published in PUBLISHED:
        e = ENTRIES[eid]
        assert e.get("published_value", e["value"]) == pytest.approx(published, rel=1e-12), eid
    assert ENTRIES["soybean_oil_us"]["low"] == pytest.approx(62.17 * 22.046226218, abs=0.006)    # USDA range 62.17-69.42 ¢/lb
    assert ENTRIES["soybean_oil_us"]["high"] == pytest.approx(69.42 * 22.046226218, abs=0.006)
    assert ENTRIES["depreciation_renewable_cogeneration_mx"]["value"] == 1.0                    # LISR art. 34-XIII
    assert [r["fraccion_del_capital"] for r in ENTRIES["construction_schedule"]["table"]] == [0.08, 0.60, 0.32]
    assert sum(r["fraccion_del_capital"] for r in ENTRIES["construction_schedule"]["table"]) == pytest.approx(1.0)
    assert ENTRIES["on_stream_hours"]["value"] / 8760.0 == pytest.approx(0.96, abs=0.001)       # "96% uptime" in the same report
    zones = {r["zona"]: r for r in ENTRIES["water_federal_rights_mx"]["table"]}
    assert zones[4]["superficial_MXN_m3"] == 2.7295 and zones[2]["subterranea_MXN_m3"] == 12.3333
    assert all(zones[z]["superficial_MXN_m3"] > zones[z + 1]["superficial_MXN_m3"] for z in (1, 2, 3))   # scarcer = dearer


# ---------------------------------------------------------------------------------------------- unit conversions
def test_unit_conversion_identities():
    lb, gal, bu, short = 0.45359237, 3.785411784e-3, 56 * 0.45359237 / 1000, 0.90718474
    for e in BANK["entries"]:
        if "conversion" in e:
            assert e["published_value"] * e["conversion"]["factor"] == pytest.approx(e["value"], rel=1e-4), e["id"]
    # stored values are rounded to 0.01 of their unit
    assert ENTRIES["distillers_corn_oil_us"]["value"] == pytest.approx(65.42 / 100 / lb * 1000, abs=0.006)
    assert ENTRIES["ethanol_fuel_us"]["value"] == pytest.approx(1.8525 / gal, abs=0.006)
    assert ENTRIES["diesel_us"]["value"] == pytest.approx(6.127 / gal, abs=0.006)
    assert ENTRIES["corn_yellow_us"]["value"] == pytest.approx(4.25 / bu, abs=0.006)
    assert ENTRIES["ddgs_us"]["value"] == pytest.approx(165.19 / short, abs=0.006)
    assert ENTRIES["diesel_mx"]["value"] == pytest.approx(26.99 * 1000)
    assert ENTRIES["sugarcane_mx"]["value"] == pytest.approx(0.57 * 14704.53, abs=0.01)         # LDSCA art. 58
    assert ENTRIES["methanol_us"]["value"] == pytest.approx(4.36 * 332.6, rel=1e-3)             # Methanex: 4.36 USD/gal, 332.6 gal/MT
    assert RP.bank_integrity()["verified"] is True and RP.bank_integrity()["reasons"] == []


def test_currency_conversion_with_bank_fix_and_with_user_rate():
    fix = ENTRIES["fx_fix_mxn_usd"]["value"]
    r = RP.lookup("diesel", "MX", "USD", None, T0)
    assert r["found"] and r["currency"] == "MXN" and r["unit_converted"] == "USD/m3"
    assert r["value_converted"] == pytest.approx(26990.0 / fix) and "FIX de Banco de México" in r["exchange_rate_source"]
    back = RP.lookup("methanol", "US", "MXN", None, T0)
    assert back["value_converted"] == pytest.approx(1450.0 * fix) and back["low_converted"] if "low" in back else True
    assert (r["value_converted"] * fix) == pytest.approx(r["value"])                           # round trip MXN→USD→MXN
    u = RP.lookup("diesel", "MX", "USD", {"MXN": 0.05}, T0)
    assert u["value_converted"] == pytest.approx(26990.0 * 0.05) and "usuario" in u["exchange_rate_source"]
    same = RP.lookup("diesel", "MX", "MXN", None, T0)
    assert same["value_converted"] == same["value"] and same["exchange_rate_applied"] is None
    native = RP.lookup("diesel", "MX", None, None, T0)
    assert native["currency_converted"] == "MXN" and native["value_converted"] == native["value"]
    eur = RP.lookup("diesel", "MX", "EUR", None, T0)
    assert eur["found"] and eur["value_converted"] is None and any("Sin conversión" in w for w in eur["warnings"])
    frac = RP.lookup("isr", "MX", "USD", None, T0)
    assert frac["value_converted"] == 0.30 and frac["unit_converted"] == "fraction"
    with pytest.raises(ValueError, match="positivo"):
        RP.lookup("diesel", "MX", "USD", {"MXN": -1.0}, T0)


# ---------------------------------------------------------------------------------------------- staleness
def test_staleness_with_a_fixed_today():
    r = RP.lookup("methanol", "US", None, None, T0)                    # period ends 2026-11-30
    assert r["age_days"] == -14 and r["stale"] is False and r["stale_after_days"] == 365
    assert RP.lookup("methanol", "US", None, None, date(2027, 11, 30))["stale"] is False        # exactly 365 days
    late = RP.lookup("methanol", "US", None, None, date(2027, 12, 1))
    assert late["age_days"] == 366 and late["stale"] is True and any("VALOR VIEJO" in w for w in late["warnings"])
    assert RP.lookup("tiie 28", "MX", None, None, T0)["stale"] is False                          # financial: 90 days
    assert RP.lookup("tiie 28", "MX", None, None, date(2027, 2, 12))["stale"] is True            # 91 days after 2026-11-13
    assert RP.lookup("isr", "MX", None, None, date(2040, 1, 1))["stale"] is False                # statutory: does not age
    assert RP.lookup("plant life", "MX", None, None, T0)["stale"] is False                       # convention
    z = RP.lookup("zafra", "MX", None, None, T0)                                                 # zafra 2023/24: 842 days old
    assert z["age_days"] == (T0 - date(2024, 7, 27)).days == 842 and z["stale"] is True
    assert RP.lookup("location factor", "MX", None, None, T0)["stale"] is True                   # 2017 data
    ages = [RP.lookup("corn", "US", None, None, date(2026, 11, d))["age_days"] for d in (14, 20, 30)]
    assert ages == sorted(ages) and ages[1] - ages[0] == 6                                       # age grows with today


# ---------------------------------------------------------------------------------------------- region fallback
def test_region_chain_and_fallback_order():
    assert RP.region_chain("MX-SIN") == ["MX-SIN", "MX", "US", "WORLD"]
    assert RP.region_chain("mx") == ["MX", "US", "WORLD"] and RP.region_chain("US") == ["US", "WORLD"]
    assert RP.region_chain("WORLD") == ["WORLD"] and RP.region_chain("MX-JAL") == ["MX-JAL", "MX", "US", "WORLD"]
    w = RP.lookup("process water", "MX-SIN", None, None, T0)
    assert w["id"] == "water_industrial_culiacan" and w["fallback_region_used"] is False
    w2 = RP.lookup("process water", "MX", None, None, T0)
    assert w2["id"] == "water_federal_rights_mx" and w2["fallback_region_used"] is False
    m = RP.lookup("Methanol", "MX-SIN", None, None, T0)                 # no MX price: US comes before WORLD
    assert m["id"] == "methanol_us" and m["fallback_region_used"] is True and m["region_used"] == "US"
    assert "MX-SIN → MX → US → WORLD" in m["fallback_note"] and m["fallback_note"] in m["warnings"]
    assert RP.lookup("methanol", "WORLD", None, None, T0)["id"] == "methanol_world"
    p = RP.lookup("aceite de palma", "MX", None, None, T0)
    assert p["region_used"] == "WORLD" and p["fallback_region_used"] is True
    s = RP.lookup("ISR", "MX-SIN", None, None, T0)
    assert s["region_used"] == "MX" and s["fallback_region_used"] is True
    no = RP.lookup("minimum wage", "US", None, None, T0)                # a Mexican value is never used for the U.S.
    assert no["found"] is False and "minimum_wage_mx" in no["close_matches"]
    assert RP.lookup("ACEITE DE HIGUERILLA", "MX", None, None, T0)["id"] == RP.lookup("castor oil", "MX", None, None, T0)["id"]


def test_unknown_and_searched_not_found_items_never_raise_and_are_never_filled():
    r = RP.lookup("unobtainium", "MX", "USD", None, T0)
    assert r["found"] is False and "value" not in r and isinstance(r["close_matches"], list) and r["reason"]
    c = RP.lookup("castor seed", "MX", None, None, T0)
    assert c["found"] is False and c["searched_not_found"] is True and "castor_oil_us" in c["close_matches"]
    g = RP.lookup("crude glycerol", "MX", None, None, T0)
    assert g["found"] is False and "glycerol_us" in g["close_matches"]          # the unspecified grade is not passed off as crude
    d = RP.lookup("depreciation", "MX", None, None, T0)
    assert d["found"] is False and len([i for i in d["close_matches"] if i.startswith("depreciation_")]) >= 5
    assert RP.lookup("", "MX", None, None, T0)["found"] is False
    assert RP.lookup("molasses", "MX", None, None, T0)["found"] is False and RP.lookup("co2", "MX", None, None, T0)["found"] is False
    assert RP.lookup("wacc mexico", "MX", None, None, T0)["searched_not_found"] is True


def test_natural_gas_and_electricity_are_delegated_to_the_utility_bank():
    ub = U.load_bank()
    ref = ub["references"]["US-EIA-2026-08"]
    gas = next(e for e in ref["entries"] if e["key"] == "fuel_per_GJ")
    expected, _ = U._entry_per_unit(gas)
    r = RP.lookup("natural gas", "US", None, None, T0)
    assert r["found"] and r["utility_reference_id"] == "US-EIA-2026-08" and r["unit"] == "USD/GJ"
    assert r["value"] == pytest.approx(expected) and r["url"] == gas["url"] and "utilities" in r["delegated_to"]
    assert r["flag"] == RP.FLAG and r["age_days"] == (T0 - date.fromisoformat(gas["period_end"])).days
    mx = RP.lookup("gas natural", "MX", None, None, T0)                  # no Mexican gas price in that bank: U.S. fallback, said
    assert mx["found"] and mx["fallback_region_used"] is True and mx["region_used"] == "US"
    el = RP.lookup("electricidad", "MX", "USD", None, T0)                # Mexico: time-of-use tariff, not averaged here
    assert el["found"] is False and el["time_of_use_tariffs"] and all(i.startswith("MX-CFE") for i in el["close_matches"])
    us = RP.lookup("electricity", "US", "MXN", None, T0)
    assert us["found"] and us["unit_converted"] == "MXN/kWh"
    assert us["value_converted"] == pytest.approx(us["value"] * ENTRIES["fx_fix_mxn_usd"]["value"])
    assert not any(_n in ("natural gas", "electricity") for e in BANK["entries"] for _n in RP._names(e))    # not duplicated


# ---------------------------------------------------------------------------------------------- suggest / design_basis / tools
def test_suggest_many_items_and_missing():
    items = ["castor oil", "methanol", "sodium methoxide", "process water", "diesel", "unobtainium"]
    s = RP.suggest(items, "MX-SIN", "USD", None, T0)
    json.dumps(s)
    assert s["found"] == ["castor oil", "methanol", "process water", "diesel"]
    assert [m["item"] for m in s["missing"]] == ["sodium methoxide", "unobtainium"] and set(s["values"]) == set(items)
    assert len(s["assumptions"]) == 4 and all(RP.FLAG in a for a in s["assumptions"])
    assert s["evidence"]["verified"] is True and s["method"] and s["not_included"] and s["flag"] == RP.FLAG
    assert "no se suponen" in s["warnings"][0]
    assert s["values"]["process water"]["value_converted"] == pytest.approx(40.43 / 18.5525)
    for it in s["found"]:
        v = s["values"][it]
        assert v["flag"] == RP.FLAG and v["source"] and v["url"].startswith("https://") and v["warnings"][0].startswith(RP.FLAG)
    with pytest.raises(ValueError, match="lista"):
        RP.suggest([], "MX")
    with pytest.raises(ValueError, match="lista"):
        RP.suggest("methanol", "MX")


def test_design_basis_values_suggestion_and_not_found():
    d = RP.design_basis("MX", "biodiésel de higuerilla", T0)
    json.dumps(d)
    need = {"isr_corporate_mx", "ptu_mx", "depreciation_chemical_mx", "banxico_target_rate", "tiie_28", "fx_fix_mxn_usd",
            "on_stream_hours", "plant_life", "construction_schedule", "working_capital_fraction", "zafra_duration_mx",
            "location_factor_mx", "minimum_wage_mx", "imss_average_wage_mx", "equity_risk_premium_mx", "discount_rate_nrel"}
    assert need <= set(d["values"])
    assert all(v["kind"] in RP.DESIGN_BASIS_KINDS and v["flag"] == RP.FLAG and v["source"] and v["url"] for v in d["values"].values())
    assert sorted(i for ids in d["by_kind"].values() for i in ids) == sorted(d["values"])
    assert d["suggested_depreciation"]["suggested_id"] == "depreciation_chemical_mx"
    assert d["suggested_depreciation"]["suggested_value"] == 0.09 and "PROPUESTA" in d["suggested_depreciation"]["note"]
    assert any("WACC" in n["item"] for n in d["not_found"]) and any("IVA" in n["item"] for n in d["not_found"])
    assert d["values"]["on_stream_hours"]["fallback_region_used"] is True and d["evidence"]["verified"] is True
    assert RP.design_basis("MX", "ingenio azucarero", T0)["suggested_depreciation"]["suggested_value"] == 0.05
    assert RP.design_basis("MX", "planta de alimentos", T0)["suggested_depreciation"]["suggested_value"] == 0.08
    assert RP.design_basis("MX", "fábrica de tornillos", T0)["suggested_depreciation"]["suggested_id"] == "depreciation_other_mx"
    assert RP.design_basis("MX", None, T0)["suggested_depreciation"]["suggested_id"] is None
    us = RP.design_basis("US", None, T0)
    assert "isr_corporate_mx" not in us["values"] and "on_stream_hours" in us["values"] and "suggested_depreciation" not in us
    assert any("US" in n["item"] for n in us["not_found"])
    sin = RP.design_basis("MX-SIN", None, T0)
    assert set(sin["values"]) == set(RP.design_basis("MX", None, T0)["values"])


def test_tools_and_agent_wrappers():
    assert [f.__name__ for f in T.REFERENCE_PRICES_TOOLS] == ["reference_prices", "design_basis_defaults"]
    assert [getattr(w, "name", None) or w.__name__ for w in W.AGENT_WRAPPERS] == [f.__name__ for f in T.REFERENCE_PRICES_TOOLS]
    a = T.reference_prices(["ethanol", "sugarcane", "corn"], "MX", "USD", None, "2026-11-16")
    assert a == RP.suggest(["ethanol", "sugarcane", "corn"], "MX", "USD", None, T0) and len(json.dumps(a)) < 20000
    assert a["values"]["sugarcane"]["unit"].startswith("MXN/t_azucar_recuperable") and "KARBE" in a["values"]["sugarcane"]["notes"]
    b = T.design_basis_defaults("MX", "bioetanol de caña", "2026-11-16")
    assert b["suggested_depreciation"]["suggested_value"] == 0.05 and len(json.dumps(b, ensure_ascii=False)) < 40000
    with pytest.raises(ValueError, match="ISO"):
        T.reference_prices(["ethanol"], today="16/11/2026")
    out = json.loads(W.AGENT_WRAPPERS[0](items=["methanol"], region="MX", currency="MXN", today="2026-11-16"))
    assert out["values"]["methanol"]["value_converted"] == pytest.approx(1450.0 * 18.5525)
    out2 = json.loads(W.AGENT_WRAPPERS[1](region="MX", plant_type="biodiesel", today="2026-11-16"))
    assert out2["values"]["isr_corporate_mx"]["value"] == 0.30
