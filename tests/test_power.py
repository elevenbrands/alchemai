"""Tests of the BWR cycle and the Laguna Verde model (bloque M27_N). Offline; every reference number was READ on 2026-11-16.

Sources (date read 2026-11-16):
  [RK]   nuclear-power.com «Example of Rankine cycle – problem with solution» (6 MPa sat. steam, 0.008 MPa)
         https://www.nuclear-power.com/nuclear-engineering/thermodynamics/thermodynamic-cycles/rankine-cycle-steam-turbine-cycle/example-of-rankine-cycle-problem-with-solution
  [PRIS] IAEA PRIS via WNA https://world-nuclear.org/nuclear-reactor-database/details/Laguna-Verde-1 (and -2)
  [IBE]  Merino Teillet et al., ENC 2010 (OSTI 21449482): gross 686.7 MWe at 2027 MWt https://www.osti.gov/etdeweb/biblio/21449482
  [RL]   Ramírez-León et al. 2013: 63 m3/s, sea 25/28 °C, discharge 38/44 °C https://www.scipedia.com/public/Ramirez-Leon_et_al_2013a
"""
from __future__ import annotations

import json

import pytest

from packages.diagram.symbols import EQUIPMENT_TYPES, LABELS_ES, NOMINAL_SIZE, SYMBOLS, symbol_svg
from packages.dwsim_client import tools_power as TP
from packages.power import nuclear as N


@pytest.fixture(scope="module")
def lv():
    return TP.power_plant_model(today="2026-11-16")


# ------------------------------------------------------------------------------------------- VALIDATED (sources read)
def test_rankine_textbook_example():
    """[RK] ideal cycle: x4 0.694, w_T 945, w_P 5.7, q_in 2605.3 kJ/kg, eta 36.1 % (5 values)."""
    r = N.rankine_example()
    p, m = r["published"], r["model"]
    assert m["x4"] == pytest.approx(p["x4"], rel=0.002)                       # measured -0.02 %
    assert m["w_turbine_kJ_kg"] == pytest.approx(p["w_turbine_kJ_kg"], rel=0.003)   # measured -0.13 %
    assert m["q_in_kJ_kg"] == pytest.approx(p["q_in_kJ_kg"], rel=0.001)     # measured -0.02 %
    assert m["efficiency"] == pytest.approx(p["efficiency"], rel=0.005)     # measured -0.27 % (published rounded)
    # pump work: published 5.7 is a table interpolation; v*dP = 1.008e-3*5992 = 6.04 kJ/kg -> measured +5.9 %
    assert m["w_pump_kJ_kg"] == pytest.approx(p["w_pump_kJ_kg"], rel=0.07)
    assert abs(r["energy_residual_rel"]) < 1e-9


def test_pre_uprate_gross_output(lv):
    """[IBE] gross 686.7 MWe at 2027 MWt, NOT used in the fit (fit = PRIS 805 MWe at 2317 MWt): measured +3.1 %."""
    row = next(r for r in lv["published_comparison"]["rows"] if r["quantity"].startswith("bruta a 2027"))
    assert row["role"] == "validación"
    assert 0.0 < row["deviation_rel"] < 0.05


def test_seawater_rise_vs_published(lv):
    """[RL] seawater rise 13 K (dry) / 16 K (rainy) at 2027 MWt with 63 m3/s for the plant: model ~10.3 K (-21 % / -36 %).
    The model is LOW: either the 63 m3/s is not all for the condensers at once or the published discharge is local."""
    rows = [r for r in lv["published_comparison"]["rows"] if r["quantity"].startswith("calentamiento")]
    assert len(rows) == 2
    for r in rows:
        assert -0.40 < r["deviation_rel"] < -0.15


# ------------------------------------------------------------------------------------------------- CONSISTENCY
def test_calibration_and_balances(lv):
    for u, c in lv["calibrated_parameters"].items():
        e = c["turbine_isentropic_efficiency"]
        assert abs(e["residual_rel"]) < 1e-9 and 0.6 < e["value"] < 0.95
    ev = lv["evidence"]
    assert ev["verified"]
    assert ev["energy_balance_residual_rel"] < 1e-6 and ev["plant_balance_residual_rel"] < 1e-6
    assert ev["mass_balance_residual"] < 1e-9
    cy = lv["cycle"]
    q = cy["duties_W"]
    assert q["reactor"] + q["feed_pump"] == pytest.approx(q["turbine_shaft"] + q["condenser"], rel=1e-9)
    assert cy["heat_rate_kJ_kWh"] == pytest.approx(3600 / cy["gross_efficiency"], rel=1e-12)


def test_generation_identity_pris(lv):
    """[PRIS] reference power x hours x load factor reproduces published GWh to rounding (definition identity)."""
    for g in lv["generation"].values():
        assert abs(g["deviation_rel"]) < 0.002


def test_monotonicity():
    d = N._load()
    des = N._design(d, {})
    kw = {**des["kw"], "heaters_out": []}
    hot = N.cycle_with_sea(2317e6, eta_turbine=0.8, **{**kw, "sea_in_K": 303.15})
    cold = N.cycle_with_sea(2317e6, eta_turbine=0.8, **{**kw, "sea_in_K": 298.15})
    assert cold["gross_W"] > hot["gross_W"] and cold["p_cond_Pa"] < hot["p_cond_Pa"]
    lo = N.cycle_with_sea(2317e6, eta_turbine=0.75, **kw)
    assert lo["gross_W"] < cold["gross_W"] or lo["gross_W"] < hot["gross_W"]
    out = N.cycle_with_sea(2317e6, eta_turbine=0.8, **{**kw, "heaters_out": [6]})
    assert out["feedwater_T_K"] < hot["feedwater_T_K"]
    assert abs(out["plant_balance_residual_rel"]) < 1e-9
    no_regen = N.bwr_cycle(des["p_dome_Pa"], 7000.0, [], 1.0, msr=False)
    regen = N.bwr_cycle(des["p_dome_Pa"], 7000.0, des["kw"]["extraction_pressures_Pa"], 1.0, msr=False)
    assert regen["efficiency_cycle"] > no_regen["efficiency_cycle"]


def test_carnot_limit():
    d = N._load()
    des = N._design(d, {})
    c = N.bwr_cycle(des["p_dome_Pa"], 7000.0, des["kw"]["extraction_pressures_Pa"], 1.0)
    assert c["efficiency_cycle"] < 1 - N.tsat(7000.0) / N.tsat(des["p_dome_Pa"])


def test_economics_emissions_opportunities(lv):
    ec = lv["economics"]
    assert ec["revenue_usd_y"] == pytest.approx(ec["price_usd_mwh"] * ec["net_generation_MWh"])
    assert ec["prices"]["energia_mxn_mwh"]["url"].startswith("https://ws01.cenace.gob.mx")
    assert lv["emissions_avoided"]["co2_avoided_t_y"] == pytest.approx(ec["net_generation_MWh"] * 0.444)
    ids = {u["id"] for u in lv["process_model"]["units"]}
    for o in lv["opportunities"]:
        assert set(o["unit_ids"]) <= ids and o["category"] in ("energy", "yield", "utilities", "reliability", "emissions",
                                                               "economics")
    assert [o["rank"] for o in lv["opportunities"]] == list(range(1, len(lv["opportunities"]) + 1))
    for u in lv["process_model"]["units"]:
        assert u["type"] in EQUIPMENT_TYPES
    json.dumps(lv)


def test_no_price_year_and_user_price():
    r = TP.power_plant_model(year=2019, today="2026-11-16")
    assert r["economics"]["price_usd_mwh"] is None and any("No hay precio" in w for w in r["warnings"])
    r2 = TP.power_plant_model(year=2019, unit="2", prices={"energia_usd_mwh": 50.0}, today="2026-11-16")
    assert r2["economics"]["price_usd_mwh"] == 50.0
    json.dumps(r2)


def test_refusals():
    with pytest.raises(ValueError):
        TP.power_plant_model(plant="atucha")
    with pytest.raises(ValueError):
        TP.power_plant_model(unit="3")
    with pytest.raises(ValueError):
        TP.power_plant_model(year=1980)
    with pytest.raises(ValueError):
        TP.power_plant_model(overrides={"turbine_eff": 0.9})
    with pytest.raises(ValueError):
        TP.power_plant_model(prices={"uranio": 1.0})
    json.dumps(TP.list_power_plants())


# --------------------------------------------------------------------------------------------- new diagram symbols
def test_power_symbols():
    for t in ("nuclear_reactor", "steam_turbine", "generator", "condenser", "moisture_separator_reheater"):
        assert t in EQUIPMENT_TYPES and t in SYMBOLS and t in LABELS_ES and t in NOMINAL_SIZE
        svg = symbol_svg(t, 0, 0, 80, 80)
        assert svg.startswith("<g") and f'data-type="{t}"' in svg


def test_diagram_draws(lv, tmp_path):
    from packages.dwsim_client.tools_diagram import process_diagram
    out = process_diagram(process_model=lv["process_model"], opportunities=lv["opportunities"], view="full")
    assert out["evidence"]["verified"], out["evidence"]
