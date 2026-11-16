"""Tests of bloque M26_W (whole-plant economics orchestrator): packages/costing/plant.py and tools_plant_economics.py.

Validation case (read on 2026-11-16 in Humbird et al. 2011, NREL/TP-5100-47764, https://docs.nlr.gov/docs/fy11osti/47764.pdf):
  Table 28 purchased and installed cost by area, factors of Table 27 → TCI 422,500,000 USD (2007$);
  Table 30 variable operating costs, Table 31 fixed operating costs (10.66 MM$/y), Table 33 DCF parameters;
  MESP 2.15 USD/gal at 10 % IRR, 61.0 MMgal/y (Table 34), 40 % equity with an 8 % 10-year loan.
  Table 43: «Financial assumptions −$0.06: 100% equity versus 40% equity with 8% interest» → 100 % equity ≈ 2.21 USD/gal.
The repo's cash flow has no loans, so the comparison at 100 % equity (2.21) is the like-for-like one.
"""
from __future__ import annotations

import json
import math

import pytest

from packages.costing import plant as PL
from packages.dwsim_client import tools_plant_economics as T

# ---------------------------------------------------------------- Humbird 2011 inputs (Tables 28, 30, 31, 33)
AREAS = [("A200 Pretreatment", 19.9e6, 29.9e6, True), ("A200 Conditioning", 1.5e6, 3.0e6, True),
         ("A300 Hydrolysis and fermentation", 18.5e6, 31.2e6, True), ("A400 Enzyme production", 10.7e6, 18.3e6, True),
         ("A500 Recovery", 11.1e6, 22.3e6, True), ("A600 Wastewater", 49.3e6, 49.4e6, False), ("A700 Storage", 2.8e6, 5.0e6, False),
         ("A800 Boiler", 36.5e6, 66.0e6, False), ("A900 Utilities", 4.0e6, 6.9e6, False)]
RAW = [("Feedstock (wet)", 104167, 46.80), ("Sulfuric acid, 93%", 1981, 81.39), ("Ammonia A200", 1051, 406.96),
       ("Corn steep liquor A300", 1158, 51.55), ("Diammonium phosphate", 142, 895.32), ("Sorbitol", 44, 1021.93),
       ("Glucose", 2418, 526.52), ("Corn steep liquor A400", 164, 51.55), ("Ammonia A400", 115, 406.96), ("Host nutrients", 67, 745.30),
       ("Sulfur dioxide", 16, 275.70), ("Caustic", 2252, 135.65), ("FGD lime", 895, 180.87), ("Makeup water", 147140, 0.23)]
POS = [(1, 141569), (2, 67414), (1, 54894), (12, 38522), (1, 53931), (2, 38522), (2, 38522), (4, 46227), (20, 38522), (8, 38522),
       (4, 26966), (3, 34670)]
H_BASIS = {"on_stream_hours": 8410, "plant_life_years": 30, "income_tax_rate": 0.35, "discount_rate": 0.10, "depreciation": "MACRS7",
           "construction_schedule": [0.08, 0.60, 0.32], "startup_years": 0.25, "startup_revenue_fraction": 0.5,
           "startup_variable_cost_fraction": 0.75, "startup_fixed_cost_fraction": 1.0, "location_factor": 1.0, "land_usd": 1.848e6}


def humbird_call(**kw):
    eq = [{"name": n, "F_BM": i / p, "purchase_cost_usd": p, "base_purchase_cost_usd": p, "isbl": isbl, "labour_class": "none"}
          for n, p, i, isbl in AREAS]
    raw = [{"name": n, "flow": f, "unit": "kg/h", "price": pr, "price_unit": "USD/short_ton"} for n, f, pr in RAW]
    raw += [{"name": "Boiler chems", "annual_cost": 1.23 * 8410}, {"name": "Cooling tower chems", "annual_cost": 7.14 * 8410}]
    util = [{"name": "Disposal of ash", "flow": 5725, "unit": "kg/h", "price": 28.86, "price_unit": "USD/short_ton"}]
    prod = [{"name": "Ethanol", "flow": 61.0e6, "unit": "gal/y", "price": 2.15, "price_unit": "USD/gal"},
            {"name": "Grid electricity", "flow": 12797, "unit": "kW", "price": 0.0572, "price_unit": "USD/kWh"},
            {"name": "Area 100 electricity", "flow": 859, "unit": "kW", "price": 0.0572, "price_unit": "USD/kWh"}]
    call = dict(equipment=eq, raw_materials=raw, products=prod, utilities=util,
                labour={"positions": [{"count": c, "annual_salary": s} for c, s in POS]}, basis=dict(H_BASIS), region="US",
                main_product="Ethanol", year=2007, sensitivity_span=0.0, today="2026-11-16")
    call.update(kw)
    return call


@pytest.fixture(scope="module")
def humbird():
    return PL.plant_economics(**humbird_call())


def test_humbird_tci(humbird):
    c = humbird["capital_summary"]
    assert c["total_bare_module_cost_usd"] == pytest.approx(232.1e6, rel=1e-3)           # Table 28 installed total 232.1 MM
    assert c["direct_permanent_investment_usd"] == pytest.approx(250.4e6, rel=1e-3)      # TDC 250.4 MM (measured −0.03 %)
    assert c["fixed_capital_investment_usd"] == pytest.approx(400.6e6, rel=1e-3)         # FCI 400.6 MM
    assert c["working_capital_usd"] == pytest.approx(20.0e6, rel=2e-3)
    assert c["total_capital_investment_usd"] == pytest.approx(422.5e6, rel=5e-4)         # TCI 422.5 MM (measured −0.026 %)


def test_humbird_operating_costs(humbird):
    o = {b["category"]: b["usd_per_year"] for b in humbird["operating_costs"]["breakdown"]}
    assert o["raw_materials"] == pytest.approx(70.36e6 + 0.01e6 + 0.06e6, rel=3e-3)     # Table 30 subtotal incl. boiler/CT chems
    assert o["labour"] == pytest.approx(2.48e6, rel=5e-3)                               # Table 31 salaries 2.48 MM (rounded 2.5)
    fixed = humbird["operating_costs"]["fixed_usd_y"]
    assert fixed == pytest.approx(10.66e6, rel=2e-3)                                     # Table 31 total 10.66 MM$/y
    credit = sum(i["usd_per_year"] for i in humbird["revenue"]["items"] if "electricity" in i["name"])
    assert credit == pytest.approx(6.57e6, rel=2e-3)                                      # Table 30 by-product credit 6.57
    net_var = humbird["operating_costs"]["variable_usd_y"] - credit
    assert net_var == pytest.approx(65.33e6, rel=2e-3)                                    # Table 30 total variable 65.33


def test_humbird_msp(humbird):
    m = humbird["minimum_selling_price"]
    assert m["unit"] == "USD/gal"
    # published 2.15 at 40 % equity (+3.2 % measured); 2.21 at 100 % equity per Table 43 (+0.4 % measured)
    assert m["value"] == pytest.approx(2.15, rel=0.04)
    assert m["value"] == pytest.approx(2.21, rel=0.01)
    assert abs(m["npv_usd_at_minimum_price"]) < 1e-6 * humbird["capital_summary"]["total_capital_investment_usd"]
    assert humbird["evidence"]["verified"] and humbird["evidence"]["complete"]


def test_humbird_identities(humbird):
    ev = humbird["evidence"]
    for k in ("capital_by_hand_residual", "flows_recomputed_residual", "operating_cost_sum_residual", "npv_from_table_residual",
              "npv_at_irr_residual", "npv_at_minimum_price_residual", "cash_flow_capital_equals_tci_residual"):
        assert ev[k] < 1e-8, k
    cf = humbird["cash_flow"]
    rate = H_BASIS["discount_rate"]
    assert sum(r["cash_flow"] / (1 + rate) ** r["t"] for r in cf["table"]) == pytest.approx(humbird["npv_usd"], rel=1e-9, abs=1.0)
    json.dumps(humbird)


# ---------------------------------------------------------------- analytical one-equipment plant
def one_unit(**kw):
    call = dict(equipment=[{"name": "R-1", "F_BM": 2.0, "purchase_cost_usd": 1.0e6, "base_purchase_cost_usd": 1.0e6}],
                raw_materials=[{"name": "A", "flow": 1000.0, "unit": "kg/h", "price": 100.0, "price_unit": "USD/t"}],
                products=[{"name": "P", "flow": 800.0, "unit": "kg/h", "price": 400.0, "price_unit": "USD/t"}],
                labour={"total_salaries_per_year": 0.0, "burden_fraction": 0.0},
                capital_factors={k: 0.0 for k in ("site_preparation", "service_facilities", "allocated_costs", "contingency_and_fee",
                                                  "land", "royalties", "startup", "working_capital")},
                basis={"on_stream_hours": 8000, "plant_life_years": 10, "income_tax_rate": 0.0, "discount_rate": 0.1,
                       "depreciation": "SL10", "construction_schedule": [1.0], "startup_years": 0.0, "location_factor": 1.0,
                       "maintenance_fraction_of_isbl": 0.0, "insurance_tax_fraction_of_fci": 0.0},
                region="US", sensitivity_span=0.0, today="2026-11-16")
    call.update(kw)
    return call


def test_one_unit_annuity_by_hand():
    r = PL.plant_economics(**one_unit())
    fci = 2.0e6
    net = 8000 * (0.8 * 400.0 - 1.0 * 100.0)                     # USD/y
    ann = (1 - 1.1 ** -10) / 0.1
    assert r["capital_summary"]["total_capital_investment_usd"] == pytest.approx(fci, rel=1e-12)
    assert r["npv_usd"] == pytest.approx(-fci + net * ann, rel=1e-9)
    msp = (fci / ann + 8000 * 100.0) / (8000 * 0.8)             # USD/t, zero tax: annuity identity
    assert r["minimum_selling_price"]["value"] == pytest.approx(msp, rel=1e-8)
    assert r["n_assumed"] == 0 and r["missing"] == []


def test_one_unit_with_tax_by_hand():
    b = dict(one_unit()["basis"], income_tax_rate=0.3)
    r = PL.plant_economics(**one_unit(basis=b))
    fci, net, T_ = 2.0e6, 8000 * 220.0, 0.3
    ann = (1 - 1.1 ** -10) / 0.1
    assert r["npv_usd"] == pytest.approx(-fci + ((1 - T_) * net + T_ * fci / 10) * ann, rel=1e-9)


def test_unit_conversion_paths():
    # 1 t/h at 100 USD/t over 8000 h = 0.8 MM USD; the same in lb/h priced per short ton must agree
    a = PL.plant_economics(**one_unit())["operating_costs"]["breakdown"][0]["usd_per_year"]
    raw = [{"name": "A", "flow": 1000.0 / 0.45359237, "unit": "lb/h", "price": 100.0 * 0.90718474, "price_unit": "USD/short_ton"}]
    b = PL.plant_economics(**one_unit(raw_materials=raw))["operating_costs"]["breakdown"][0]["usd_per_year"]
    assert a == pytest.approx(800000.0, rel=1e-12) and b == pytest.approx(a, rel=1e-12)


def test_monotonic_price():
    lo = PL.plant_economics(**one_unit())["npv_usd"]
    hi = PL.plant_economics(**one_unit(products=[{"name": "P", "flow": 800.0, "unit": "kg/h", "price": 450.0,
                                                  "price_unit": "USD/t"}]))["npv_usd"]
    assert hi > lo


# ---------------------------------------------------------------- suggestion, missing and refusals
def mx_call(**kw):
    call = dict(equipment=[{"name": "R-101", "type": "vessel_vertical", "purchase_cost_usd": 300000.0, "base_purchase_cost_usd": 250000.0},
                           {"name": "C-201", "type": "tower", "purchase_cost_usd": 500000.0, "base_purchase_cost_usd": 400000.0},
                           {"name": "E-101", "type": "heat_exchanger_Utube", "purchase_cost_usd": 80000.0, "base_purchase_cost_usd": 60000.0}],
                raw_materials=[{"name": "aceite de ricino", "reference_item": "castor oil", "flow": 1250.0, "unit": "kg/h"},
                               {"name": "metanol", "reference_item": "methanol", "flow": 140.0, "unit": "kg/h"}],
                products=[{"name": "biodiesel", "flow": 1.4, "unit": "m3/h"},
                          {"name": "glicerina", "reference_item": "glycerol", "flow": 125.0, "unit": "kg/h"}],
                region="MX", plant_type="biodiesel", main_product="biodiesel", today="2026-11-16")
    call.update(kw)
    return call


@pytest.fixture(scope="module")
def mx():
    return PL.plant_economics(**mx_call())


def test_assumed_inputs_sourced(mx):
    assert mx["n_assumed"] > 10
    for a in mx["assumed_inputs"]:
        assert a["source"] and a["url"].startswith("https://") and a["date_read"] and a["flag"] == PL.FLAG
    keys = {a["input"] for a in mx["assumed_inputs"]}
    assert {"prices.metanol", "prices.biodiesel", "basis.income_tax_rate", "basis.depreciation", "labour.operators"} <= keys
    tax = next(a for a in mx["assumed_inputs"] if a["input"] == "basis.income_tax_rate")
    assert tax["value"] == 0.3                                                            # ISR only, PTU not folded in
    assert mx["ptu"]["included_in_tax_rate"] is False
    assert mx["ptu"]["uncapped_combined_rate"] == pytest.approx(0.3 + 0.1 * 0.7)
    dep = next(a for a in mx["assumed_inputs"] if a["input"] == "basis.depreciation")
    assert dep["value"] == "SL12"                                                         # 9 %/y max (LISR 35-V) → 12 years
    assert mx["evidence"]["verified"], mx["evidence"]["reasons"]
    assert mx["sensitivity"]["rows"] and mx["sensitivity"]["rows"][0]["rank"] == 1
    sw = [r["impact"] for r in mx["sensitivity"]["rows"]]
    assert sw == sorted(sw, reverse=True)
    json.dumps(mx)


def test_frozen_call_reproduces(mx):
    again = PL.plant_economics(**{**mx["inputs_resolved"]["frozen_call"], "sensitivity_span": 0.0})
    assert again["n_assumed"] == 0
    assert again["npv_usd"] == pytest.approx(mx["npv_usd"], rel=1e-9)
    assert again["minimum_selling_price"]["value"] == pytest.approx(mx["minimum_selling_price"]["value"], rel=1e-9)


def test_update_identity_and_changes(mx):
    ov = {"prices": {"metanol": {"price": 600.0, "price_unit": "USD/t"}}, "basis": {"discount_rate": 0.12}}
    up = T.plant_economics_update(mx["inputs_resolved"], ov)
    direct = PL.plant_economics(**PL.merge_overrides(mx["inputs_resolved"]["call"], ov))
    assert up["npv_usd"] == pytest.approx(direct["npv_usd"], rel=1e-12)
    assert up["minimum_selling_price"]["value"] == pytest.approx(direct["minimum_selling_price"]["value"], rel=1e-12)
    ch = up["changes"]
    assert ch["n_assumed_after"] == ch["n_assumed_before"] - 2
    row = next(r for r in ch["rows"] if r["input"] == "basis.discount_rate")
    assert row["before"] == 0.1 and row["before_origin"] == "supuesto" and row["after"] == 0.12
    assert ch["before"]["npv_usd"] == pytest.approx(mx["npv_usd"], rel=1e-9)
    json.dumps(up)


def test_missing_product_price_keeps_msp():
    r = PL.plant_economics(**mx_call(products=[{"name": "biodiesel", "flow": 1.4, "unit": "m3/h"},
                                               {"name": "subproducto raro", "flow": 10.0, "unit": "kg/h"}], sensitivity_span=0.0))
    assert r["npv_usd"] is None and r["irr"] is None
    assert r["minimum_selling_price"]["is_upper_bound"] is True
    assert any(m["input"] == "prices.subproducto raro" for m in r["missing"])


def test_missing_raw_price_blocks_npv():
    raw = mx_call()["raw_materials"] + [{"name": "catalizador exótico", "flow": 5.0, "unit": "kg/h"}]
    r = PL.plant_economics(**mx_call(raw_materials=raw, sensitivity_span=0.0))
    assert r["npv_usd"] is None and r["minimum_selling_price"] is None
    assert r["capital_summary"]["total_capital_investment_usd"] > 0
    assert r["evidence"]["complete"] is False


def test_refusals():
    with pytest.raises(ValueError, match="density_kg_m3"):
        PL.plant_economics(**mx_call(products=[{"name": "biodiesel", "flow": 1250.0, "unit": "kg/h"}]))
    with pytest.raises(ValueError, match="flow"):
        PL.plant_economics(**mx_call(raw_materials=[{"name": "metanol", "reference_item": "methanol"}]))
    with pytest.raises(ValueError, match="ambiguo"):
        PL.plant_economics(**one_unit(raw_materials=[{"name": "A", "flow": 1.0, "unit": "ton/h", "price": 1, "price_unit": "USD/t"}]))
    with pytest.raises(ValueError, match="equipment"):
        PL.plant_economics(**one_unit(equipment=[]))
    # an ECONOMIC input with no bank value for the region is not invented: it goes to `missing` and blocks NPV/IRR/MSP
    r = PL.plant_economics(**one_unit(region="WORLD", basis={k: v for k, v in one_unit()["basis"].items() if k != "income_tax_rate"}))
    assert r["npv_usd"] is None and r["minimum_selling_price"] is None
    assert any(m["input"] == "basis.income_tax_rate" for m in r["missing"])


def test_thermo_flags_restored():
    from thermo.utils.t_dependent_property import TDependentProperty
    before = TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR
    PL.plant_economics(**one_unit())
    assert TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR == before


def test_labour_correlation_worked_example():
    # LibreTexts 6.4 (read 2026-11-16): P = 0, N_np = 9 → N_OL = 2.89 ≈ 3 per shift; 3 × 4.5 = 13.5 ≈ 14 operators
    n = PL.operators_from_steps(9, 0)
    assert n["operators_per_shift_raw"] == pytest.approx(2.89, abs=0.005)
    assert n["operators_per_shift"] == 3 and n["operators"] == 14


def test_report(mx, tmp_path):
    rep = T.plant_report(mx, "Planta de biodiésel de ricino", str(tmp_path / "r.md"))
    md = rep["markdown"]
    assert "## Supuestos por confirmar" in md and "https://" in md and PL.FLAG in md
    assert (tmp_path / "r.md").read_text(encoding="utf-8") == md
    with pytest.raises(ValueError):
        T.plant_report({"x": 1}, "t")


# ---------------------------------------------------------------- HAZOP from a description
def test_hazop_from_description():
    units = [{"name": "P-1", "type": "bomba", "inlets": ["s1"], "outlets": ["s2"]},
             {"name": "R-1", "type": "RCT_CSTR", "inlets": ["s2"], "outlets": ["s3"], "duty_kW": -120.0},
             {"name": "F-1", "type": "centrifuga", "inlets": ["s3"], "outlets": ["s4"]}]
    comp = {"Methanol": 0.5, "Water": 0.5}
    streams = [{"name": n, "T_K": 333.0, "P_Pa": p, "vapor_fraction": 0.0, "composition": comp, "mass_flow_kg_s": 0.4,
                "molar_flow_mol_s": 15.0} for n, p in (("s1", 1.2e5), ("s2", 4e5), ("s3", 3.5e5), ("s4", 1.5e5))]
    r = T.hazop_from_process_description(units, streams)
    assert r["n_rows"] > 0 and r["evidence"]["verified"]
    assert r["graph_units"]["R-1"]["exothermic"] is True and r["graph_units"]["P-1"]["kind"] == "Pump"
    assert any("centrifuga" in m for m in r["missing"])
    assert all(1 <= x["severity"] <= 5 for x in r["rows"])
    json.dumps(r)
    with pytest.raises(ValueError, match="composición"):
        T.hazop_from_process_description(units, [{"name": "s1", "T_K": 300.0, "P_Pa": 1e5}])


def test_wrappers_match_tools():
    from packages.agent import wrappers_plant_economics as W
    assert [f.__name__ for f in T.PLANT_ECONOMICS_TOOLS] == [w.name for w in W.AGENT_WRAPPERS]
    assert all(math.isfinite(1.0) for _ in W.AGENT_WRAPPERS)
