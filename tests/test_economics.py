"""Plant economics (bloque M3): cash flow against closed-form engineering-economy results and against BioSTEAM's own
cash-flow functions; bare-module capital with Seider's factors read from BioSTEAM."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools as T  # noqa: E402,F401  (resolves the tools ↔ tools_costing import order)
from packages.dwsim_client import tools_costing as TC  # noqa: E402
from packages.dwsim_client import tools_economics as TE  # noqa: E402

FACTORS = dict(site_preparation=0.05, service_facilities=0.10, allocated_costs=0.0, contingency_and_fee=0.18, land=0.02,
               royalties=0.0, startup=0.10, working_capital=0.15)   # test inputs, not recommendations


def pa(i, n):
    """Present worth of a uniform series (P/A, i, n)."""
    return (1 - (1 + i) ** -n) / i


def test_npv_and_irr_of_a_uniform_series():
    C, n, i = 1.0e6, 10, 0.10
    R = C / pa(0.12, n)                                                   # sized so that the IRR is exactly 12 %
    r = TE.cash_flow_analysis(C, R, 0.0, n, 0.0, i, "SL10", [1.0])
    assert r["npv_usd"] == pytest.approx(-C + R * pa(i, n), rel=1e-10)
    assert r["irr"] == pytest.approx(0.12, abs=1e-9)
    assert r["evidence"]["verified"] and r["evidence"]["crosscheck_biosteam"]


def test_after_tax_cash_flow_with_straight_line_depreciation():
    C, R, O, T_, n, i = 2.0e6, 9.0e5, 3.0e5, 0.30, 10, 0.08
    r = TE.cash_flow_analysis(C, R, O, n, T_, i, "SL10", [1.0])
    yearly = (R - O) * (1 - T_) + T_ * C / n                             # after-tax cash + depreciation tax shield
    assert r["npv_usd"] == pytest.approx(-C + yearly * pa(i, n), rel=1e-10)
    assert r["table"][1]["tax"] == pytest.approx(T_ * (R - O - C / n))


def test_working_capital_and_land_come_back_at_the_end():
    C, WC, L, n, i = 1.0e6, 2.0e5, 5.0e4, 12, 0.10
    base = TE.cash_flow_analysis(C, 3.0e5, 0.0, n, 0.0, i, "SL12", [1.0])
    r = TE.cash_flow_analysis(C, 3.0e5, 0.0, n, 0.0, i, "SL12", [1.0], working_capital_usd=WC, land_usd=L)
    assert r["npv_usd"] - base["npv_usd"] == pytest.approx(-(WC + L) + (WC + L) / (1 + i) ** n, rel=1e-10)


@pytest.mark.parametrize("dep", ["MACRS3", "MACRS7", "MACRS15", "DDB10", "SYD10", "SL5"])
def test_matches_biosteam_with_construction_startup_and_carried_losses(dep):
    # two construction years, half-year startup at half revenue, MACRS front-loading → early losses carried forward
    r = TE.cash_flow_analysis(5.0e6, 1.6e6, 1.1e6, 20, 0.30, 0.10, dep, [0.4, 0.6], working_capital_usd=6.0e5,
                              land_usd=1.0e5, startup_years=0.5, startup_revenue_fraction=0.5,
                              startup_operating_cost_fraction=0.75)
    ev = r["evidence"]
    assert ev["crosscheck_biosteam"], (r["npv_usd"], ev["npv_biosteam_usd"])
    assert ev["depreciation_complete"] and ev["verified"]
    assert [row["t"] for row in r["table"][:3]] == [-1, 0, 1]


def test_macrs7_is_the_irs_table_and_life_must_cover_it():
    r = TE.cash_flow_analysis(1.0e6, 4.0e5, 1.0e5, 10, 0.3, 0.1, "MACRS7", [1.0])
    assert r["depreciation_schedule"] == pytest.approx([.1429, .2449, .1749, .1249, .0893, .0892, .0893, .0446])
    ddb = TE.cash_flow_analysis(1.0e6, 4.0e5, 1.0e5, 10, 0.3, 0.1, "DDB10", [1.0])["depreciation_schedule"]
    assert ddb[0] == pytest.approx(0.2) and sum(ddb) == pytest.approx(1.0)         # switches to SL, fully depreciates
    with pytest.raises(ValueError, match="longer than the plant life"):
        TE.cash_flow_analysis(1.0e6, 4.0e5, 1.0e5, 5, 0.3, 0.1, "MACRS7", [1.0])
    with pytest.raises(ValueError, match="summing to 1"):
        TE.cash_flow_analysis(1.0e6, 4.0e5, 1.0e5, 10, 0.3, 0.1, "SL10", [0.5, 0.4])


def test_break_even_revenue_gives_zero_npv():
    r = TE.cash_flow_analysis(3.0e6, 1.0e6, 4.0e5, 15, 0.30, 0.12, "SL10", [0.5, 0.5], working_capital_usd=3e5)
    be = TE.cash_flow_analysis(3.0e6, r["break_even_revenue_usd_y"], 4.0e5, 15, 0.30, 0.12, "SL10", [0.5, 0.5],
                               working_capital_usd=3e5)
    assert abs(be["npv_usd"]) < 1.0


def test_bare_module_factors_are_seiders():
    f = TE.bare_module_factors()["factors"]
    assert (f["heat_exchanger_Utube"], f["pump"], f["compressor"]) == (3.17, 3.3, 2.15)
    assert (f["vessel_vertical"], f["vessel_horizontal"], f["tower"], f["platforms_ladders"]) == (4.16, 3.05, 4.3, 1.0)


def test_capital_investment_from_equipment_cost():
    pu = TC.equipment_cost("pump", {"power_W": 2e4, "deltaP_Pa": 5e5, "density_kg_m3": 1000, "flow_m3_s": 0.03},
                           {"material": "StainlessSteel"})
    ve = TC.equipment_cost("vessel", {"diameter_m": 2.0, "length_m": 20.0, "vertical": True, "number_of_trays": 30})
    assert pu["base_purchase_cost_usd"] < pu["purchase_cost_usd"]               # stainless and F_T 2.0 vs cast iron, F_T 1
    assert ve["base_purchase_cost_usd"] == pytest.approx(ve["purchase_cost_usd"])  # carbon steel is the base
    eq = [{"name": "P-101", "type": "pump", **{k: pu[k] for k in ("purchase_cost_usd", "base_purchase_cost_usd", "basis")}},
          {"name": "T-101", "type": "tower", "platforms_ladders_purchase_cost_usd": ve["platforms_ladders_purchase_cost_usd"],
           **{k: ve[k] for k in ("purchase_cost_usd", "base_purchase_cost_usd", "basis")}}]
    r = TE.capital_investment(eq, FACTORS)
    p, t = r["equipment"]
    assert p["bare_module_cost_usd"] == pytest.approx(pu["purchase_cost_usd"] + pu["base_purchase_cost_usd"] * 2.3)
    pl = ve["platforms_ladders_purchase_cost_usd"]
    assert t["bare_module_cost_usd"] == pytest.approx((ve["purchase_cost_usd"] - pl) * 4.3 + pl)
    tbm = p["bare_module_cost_usd"] + t["bare_module_cost_usd"]
    assert r["total_bare_module_cost_usd"] == pytest.approx(tbm)
    tdc = tbm * 1.15 * 1.18
    assert r["total_depreciable_capital_usd"] == pytest.approx(tdc)
    assert r["total_capital_investment_usd"] == pytest.approx(tdc * 1.12 * 1.15)
    assert r["evidence"]["verified"]


def test_capital_investment_refuses_missing_factors_and_flags_gaps():
    eq = [{"name": "X", "type": "pump", "purchase_cost_usd": 1e5}]
    with pytest.raises(ValueError, match="factors missing"):
        TE.capital_investment(eq, {"contingency_and_fee": 0.18})
    r = TE.capital_investment(eq, FACTORS)                                     # no base cost → F_BM·C_P, not verified
    assert r["equipment"][0]["bare_module_cost_usd"] == pytest.approx(3.3e5) and not r["evidence"]["verified"]
    mixed = [{"name": "A", "type": "pump", "purchase_cost_usd": 1e5, "base_purchase_cost_usd": 1e5, "basis": "CEPCI 797.9"},
             {"name": "B", "F_BM": 2.0, "purchase_cost_usd": 1e5, "base_purchase_cost_usd": 1e5, "basis": "CEPCI 600"}]
    r = TE.capital_investment(mixed, FACTORS)
    assert not r["evidence"]["same_cost_basis"] and not r["evidence"]["verified"]


def test_results_are_plain_json():
    import json
    r = TE.cash_flow_analysis(5.0e6, 1.6e6, 1.1e6, 20, 0.30, 0.10, "MACRS7", [0.4, 0.6], working_capital_usd=6e5)
    json.dumps(r, allow_nan=False)                                             # MCP serializes without a default hook
    json.dumps(TE.capital_investment([{"name": "X", "F_BM": 2.0, "purchase_cost_usd": 1e5, "base_purchase_cost_usd": 1e5}],
                                     FACTORS), allow_nan=False)
