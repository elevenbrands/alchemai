"""Tests of the refinery planning model (bloque M27_R). Offline; every reference number below was READ on 2026-11-16.

Sources (date read 2026-11-16):
  [FCC]  J. Jechura, CBEN 409 «Catalytic Cracking», Colorado School of Mines, slide 16 «FCCU Yield Example»
         https://people.mines.edu/jjechura/wp-content/uploads/sites/120/2019/02/CBEN409_07_Catalytic_Cracking.pdf
  [COK]  J. Jechura, CBEN 409 «Coking», slide 14 «Coker Calculations Using Gary et al. Correlations»
         https://people.mines.edu/jjechura/wp-content/uploads/sites/120/2019/02/CBEN409_06_Coking.pdf
  [ALK]  J. Jechura, CBEN 409 «Gasoline Upgrading», slide 52 «Phillips Alkylation Process Mass Balance»
         https://people.mines.edu/jjechura/wp-content/uploads/sites/120/2019/02/CBEN409_09_Gasoline_Upgrading.pdf
  [OGJ]  «Four Mexican crude assays updated», Oil & Gas Journal, 15-may-2000 https://www.ogj.com/home/article/17215393/guide-to-world-crudes
  [UNAM] O. Sánchez Alonso, tesis UNAM 2012 (ligeros % vol) https://ru.dgb.unam.mx/bitstream/20.500.14330/TES01000679501/3/0679501_A1.pdf
  [BDI]  Pemex eBDI RPROREF / REPETRE https://ebdi.pemex.com/bdi/bdiController.do?action=cuadro&cvecua=RPROREF
"""
from __future__ import annotations

import calendar
import json

import pytest

from packages.dwsim_client import tools_refinery as TR
from packages.refinery import model as M


@pytest.fixture(scope="module")
def tula():
    return TR.refinery_model("tula", today="2026-11-16", mode="nameplate")


@pytest.fixture(scope="module")
def tula_op():
    return TR.refinery_model("tula", today="2026-11-16")


# ------------------------------------------------------------------------------------------- VALIDATED (sources read)
def test_fcc_example_volume_yields_from_mass_yields():
    """[FCC] wt% -> vol% with the published standard densities reproduces the published vol% (8 values)."""
    sg = 0.9042
    kpb = M.kg_per_bbl(sg)
    feed = M._flow(25000.0, 25000.0 * kpb, 25000.0 * kpb * 0.005)
    out = M.fcc_unit(feed)
    published_vol = {"propane": 2.56, "propylene": 5.80, "isobutane": 5.59, "n_butane": 1.96, "butylenes": 7.61,
                     "gasoline": 57.05, "lco": 21.20, "hco": 6.80}
    dev = {k: abs(100 * out[k]["bpd"] / 25000.0 - v) for k, v in published_vol.items()}
    assert max(dev.values()) < 0.03, dev            # measured max 0.02 vol% points (rounding of the published table)
    # sulfur: published light-gas recovery 38.4 % of 39,575 lb/d -> 15,193 lb/d (our H2S-S); measured within 0.1 %
    assert out["acid_gas"]["s_kg_d"] / feed["s_kg_d"] == pytest.approx(38.4 / 100.1, rel=1e-12)


def test_coker_example_volumes_and_h2s():
    """[COK] 40,000 bpd of SG 1.0679, CCR 29.3, S 6.47 %: coke 46.88 wt%, gasoline 30.04 vol%, gas oil 23.19 vol%,
    H2S 308,586 lb/d (from 30 % of the sulfur)."""
    kpb = M.kg_per_bbl(1.0679)
    m = 40000.0 * kpb
    feed = M._flow(40000.0, m, m * 0.0647, m * 0.293)
    out = M.coker_unit(feed)
    assert out["_wt_pct"]["coke"] == pytest.approx(46.88, abs=1e-9)          # consistency: the ratio is taken from it
    assert 100 * out["coker_gasoline"]["bpd"] / 40000 == pytest.approx(30.04, abs=0.01)   # measured 30.04
    assert 100 * out["coker_gas_oil"]["bpd"] / 40000 == pytest.approx(23.19, abs=0.01)    # measured 23.19
    h2s_lb_d_per_lb_feed = out["acid_gas"]["kg_d"] / m
    assert h2s_lb_d_per_lb_feed * 14957881 == pytest.approx(308586, rel=2e-4)            # measured 1e-5


def test_alkylation_closure_vs_published_alkylate():
    """[ALK] alkylate per olefin closed by mass balance vs the published 390.17/190.14: deviation 0.27 %."""
    out = M.alkylation_unit(100.0, 1e9, 1e9)
    r = out["ratios"]
    assert r["alkylate_per_olefin"] == pytest.approx(r["alkylate_per_olefin_published"], rel=0.004)
    assert out["binding"] == "olefinas"


def test_assay_light_ends_and_maya_vr_ccr():
    """[OGJ]+[UNAM]: light ends by difference = 0.3/1.3/1.8 vol%; Maya VR CCR estimated from the whole-crude CCR (all
    in the vacuum residue) 25.92 vs 25.8 published (0.47 %)."""
    _, chk = M.crude_unit({"maya": 1000.0, "istmo": 1000.0, "olmeca": 1000.0})
    for c in chk:
        assert chk[c]["light_ends_by_difference_vol_pct"] == pytest.approx(chk[c]["light_ends_published_vol_pct"], abs=1e-9)
    assert chk["maya"]["vr_ccr_wt_estimated_all_ccr_in_vr"] == pytest.approx(25.8, rel=0.006)
    # published-table inconsistencies measured (reported as warnings, not fixed)
    assert chk["istmo"]["mass_from_cuts_vs_whole_crude_api_rel"] == pytest.approx(-0.0073, abs=0.0005)
    assert chk["maya"]["mass_from_cuts_vs_whole_crude_api_rel"] == pytest.approx(-0.0029, abs=0.0005)


def test_day_weighted_average_reproduces_pemex_ytd():
    """[BDI] Tula crude, monthly Jan-Sep 2026 (Mbd) -> official Jan-Sep 2026 cumulative 221.583 Mbd."""
    monthly = [221.737, 251.549, 250.231, 262.807, 256.946, 243.088, 203.768, 182.424, 123.456]
    days = [calendar.monthrange(2026, m)[1] for m in range(1, 10)]
    assert sum(v * d for v, d in zip(monthly, days)) / sum(days) == pytest.approx(221.583, abs=5e-4)


def test_published_production_tula_jan_sep_2026(tula):
    """[BDI] NAMEPLATE model at the published crude run vs published production (NOT calibrated). Measured deviations:
    gasoline +9.8 %, jet +86.5 % (all kerosene to jet), diesel +0.3 %, fuel oil -78.4 % (coker at full feed vs partial
    start-up), LPG +66.4 %, total liquids +9.5 %."""
    rows = {r["product"]: r["deviation_pct"] for r in tula["published_comparison"]["rows"]}
    expected = {"gasolina": 9.8, "turbosina": 86.5, "diesel": 0.3, "combustoleo": -78.4, "glp": 66.4, "asfalto": 0.0}
    for k, v in expected.items():
        assert rows[k] == pytest.approx(v, abs=0.5), (k, rows[k])
    assert tula["published_comparison"]["liquids_deviation_pct"] == pytest.approx(9.5, abs=0.5)


def test_as_operated_calibration_and_validation(tula_op):
    """AS-OPERATED: three CALIBRATED parameters (one per product, within 0..1) reproduce fuel oil, jet and LPG; the
    non-fitted products are the validation [BDI Jan-Sep 2026]: gasoline +0.8 %, diesel +41.5 %, total liquids +9.6 %
    (measured, not tuned)."""
    cal = tula_op["calibrated_parameters"]
    assert set(cal) == {"coker_utilization", "kerosene_to_jet_fraction", "butanes_to_gas_plants_fraction"}
    for name, c in cal.items():
        assert 0.0 <= c["value"] <= 1.0 and c["status"].startswith("resuelto"), name
        assert abs(c["residual_rel"]) < 1e-6, name
        assert "CALIBRADO" in c["flag"]
    assert cal["coker_utilization"]["value"] == pytest.approx(0.259, abs=0.002)
    assert cal["kerosene_to_jet_fraction"]["value"] == pytest.approx(0.536, abs=0.002)
    assert cal["butanes_to_gas_plants_fraction"]["value"] == pytest.approx(0.811, abs=0.002)
    comp = tula_op["published_comparison"]
    roles = {r["product"]: r["role"] for r in comp["rows"]}
    assert roles == {"gasolina": "validación", "diesel": "validación", "turbosina": "ajustado", "combustoleo": "ajustado",
                     "glp": "ajustado", "asfalto": "dato de entrada"}
    val = {r["product"]: r["deviation_pct"] for r in comp["validation_rows"]}
    assert val["gasolina"] == pytest.approx(0.8, abs=0.3)
    assert val["diesel"] == pytest.approx(41.5, abs=0.5)
    assert comp["liquids_deviation_pct"] == pytest.approx(9.6, abs=0.3)
    assert len(cal) <= len([r for r in comp["rows"] if r["role"] != "dato de entrada"])
    ec = tula_op["economics"]
    assert ec["gross_margin_usd_bbl"] == pytest.approx(23.03, abs=0.05)
    assert ec["net_margin_usd_bbl"] == pytest.approx(15.87, abs=0.05)
    assert ec["gross_margin_at_published_slate_usd_bbl"] == pytest.approx(9.34, abs=0.05)
    assert tula_op["evidence"]["verified"] and tula_op["mode"] == "as_operated"
    json.dumps(tula_op)


def test_as_operated_respects_user_values_and_zero_targets():
    r = TR.refinery_model("tula", utilization={"coker": 0.5})
    assert "coker_utilization" not in r["calibrated_parameters"]
    cad = TR.refinery_model("cadereyta")               # Cadereyta publishes 0 jet fuel -> fraction solved at the bound 0
    k = cad["calibrated_parameters"]["kerosene_to_jet_fraction"]
    assert k["value"] == 0.0 and k["status"] == "resuelto (en el límite)"
    own = TR.refinery_model("tula", crude_slate={"maya": 100000})
    assert own["mode"] == "nameplate" and own["calibrated_parameters"] == {}


# --------------------------------------------------------------------------------------- CONSISTENCY (identities)
def test_balances_close(tula):
    b = tula["balances"]
    assert b["units_max_mass_closure_rel"] < 1e-12
    assert b["overall_mass_closure_rel"] < 1e-12
    assert b["sulfur_closure_rel"] < 1e-12
    assert sum(b["sulfur_distribution_pct"].values()) == pytest.approx(100.0, abs=1e-9)
    assert tula["evidence"]["verified"] is True
    assert tula["crude_bpd"] == pytest.approx(221583.0)


def test_hydrotreater_identity_and_spec():
    kpb = M.kg_per_bbl(0.85)
    feed = M._flow(1000.0, 1000.0 * kpb, 1000.0 * kpb * 0.01)
    out = M.hydrotreater(feed, 15.0)
    p = out["product"]
    assert 1e6 * p["s_kg_d"] / p["kg_d"] == pytest.approx(15.0, rel=1e-9)
    assert feed["kg_d"] + out["h2_kg_d"] == pytest.approx(p["kg_d"] + out["acid_gas"]["kg_d"], rel=1e-14)
    assert out["h2_kg_d"] / out["s_removed_kg_d"] == pytest.approx(M.M_H2 / M.M_S)
    clean = M.hydrotreater(M._flow(1.0, 100.0, 0.0), 15.0)
    assert clean["s_removed_kg_d"] == 0.0


def test_product_specs_met(tula):
    ps = tula["product_slate"]
    assert ps["diesel"]["sulfur_mg_kg"] == pytest.approx(15.0, rel=1e-6)
    assert ps["turbosina"]["sulfur_mg_kg"] <= 3000.0 + 1e-6
    assert ps["gasolina"]["sulfur_mg_kg"] < 31.0


def test_monotonic_coker_and_limits():
    lo = TR.refinery_model("tula", utilization={"coker": 0.0}, mode="nameplate")
    mid = TR.refinery_model("tula", utilization={"coker": 0.25}, mode="nameplate")
    hi = TR.refinery_model("tula", mode="nameplate")
    fo = [r["product_slate"]["combustoleo"]["bpd"] for r in (lo, mid, hi)]
    assert fo[0] > fo[1] > fo[2]
    assert lo["product_slate"]["coque"]["kg_h"] == 0.0
    assert mid["economics"]["gross_margin_usd_bbl"] < hi["economics"]["gross_margin_usd_bbl"]


def test_kerosene_split_conserves_mass():
    a = TR.refinery_model("tula", unit_overrides={"kerosene_to_jet_fraction": 0.5}, mode="nameplate")
    b = TR.refinery_model("tula", mode="nameplate")
    ja, da = a["product_slate"]["turbosina"]["bpd"], a["product_slate"]["diesel"]["bpd"]
    jb, db = b["product_slate"]["turbosina"]["bpd"], b["product_slate"]["diesel"]["bpd"]
    assert ja + da == pytest.approx(jb + db, rel=1e-12)
    assert ja == pytest.approx(jb / 2, rel=1e-12)


def test_energy_and_co2_identities(tula):
    e, em = tula["energy"], tula["emissions"]
    assert e["refinery_total_estimated_W"] == pytest.approx(e["units_studied_W"] / 0.68)
    cdu_W = 109100 * tula["crude_bpd"] * M.BTU_J / 86400
    assert e["duties_W"]["cdu"] == pytest.approx(cdu_W)
    fc = M.fuel_constants()
    assert fc["co2_kg_GJ_ch4"] == pytest.approx(44.0095 / 16.043 / (fc["lhv_ch4_J_kg"] / 1e9), rel=2e-3)
    assert em["co2_t_per_kbbl"] == pytest.approx(39.5, abs=0.3)          # measured; Pemex 2026 target 38.7


def test_economics_shape(tula):
    ec = tula["economics"]
    assert ec["net_margin_usd_bbl"] == pytest.approx(ec["gross_margin_usd_bbl"] - ec["energy_cost_usd_bbl"]
                                                      - ec["fixed_cost_usd_bbl"])
    assert ec["annual_usd"] == pytest.approx(ec["net_margin_usd_bbl"] * tula["crude_bpd"] * 365)
    assert set(ec["missing_prices"]) == {"coque", "azufre"}
    for k, p in ec["prices"].items():
        assert p["source"] and p["unit"], k
    over = TR.refinery_model("tula", prices={"coque": 50.0, "azufre": 100.0}, mode="nameplate")
    assert over["economics"]["complete"] is True
    assert over["economics"]["gross_margin_usd_bbl"] > ec["gross_margin_usd_bbl"]


def test_process_model_contract(tula):
    pm = tula["process_model"]
    try:
        from packages.diagram.symbols import EQUIPMENT_TYPES
    except ImportError:
        EQUIPMENT_TYPES = list(M._types())
    ids = {u["id"] for u in pm["units"]}
    secs = {s["id"] for s in pm["sections"]}
    assert pm["kind"] == "refinery"
    for u in pm["units"]:
        assert u["type"] in EQUIPMENT_TYPES and u["section"] in secs
        if u["duties_W"]:
            assert u["params"]["bandwidth_process"] and ("throughput_bpd" in u["metrics"] or "production_bpd" in u["metrics"])
    for s in pm["streams"]:
        assert s["from"] in ids or s["from"] is None
        assert s["to"] in ids or s["to"] is None
    assert len({s["id"] for s in pm["streams"]}) == len(pm["streams"])


def test_all_refineries_and_json():
    lst = TR.list_refineries()
    json.dumps(lst)
    assert set(lst["refineries"]) == {"cadereyta", "madero", "minatitlan", "salamanca", "salina_cruz", "tula", "dos_bocas"}
    assert lst["refineries"]["tula"]["capacity_mbd_2025"]["cdu"] == 315.0
    for k in ("salina_cruz", "dos_bocas"):
        r = TR.refinery_model(k)
        json.dumps(r)
        assert r["evidence"]["verified"]
    sc = TR.refinery_model("salina_cruz")
    assert any("reconstituido" in w for w in sc["warnings"])


def test_json_and_periods(tula):
    json.dumps(tula)
    r = TR.refinery_model("tula", year=2025)
    assert r["period"] == "2025" and r["crude_bpd"] == pytest.approx(212244.0, abs=1.5)   # sum of types 212.243 vs published total 212.244 (Pemex rounding)
    json.dumps(TR.refinery_model("tula", crude_slate={"maya": 100000, "istmo": 100000}))


# ---------------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("kwargs", [
    {"refinery": "deer_park"}, {"period": "2019"}, {"year": 2019}, {"crude_slate": {"brent": 1000}},
    {"crude_slate": {"maya": -5}}, {"crude_slate": {"maya": 0}}, {"utilization": {"fcc": 2.0}}, {"utilization": {"x": 0.5}},
    {"utilization": "alta"}, {"prices": {"oro": 1}}, {"prices": {"maya": -1}}, {"unit_overrides": {"foo": 1}},
    {"unit_overrides": {"capacity_mbd": {"fcc": -1}}}, {"unit_overrides": {"kerosene_to_jet_fraction": 1.5}},
    {"today": "16/11/2026"}, {"mode": "real"}, {"unit_overrides": {"butanes_to_gas_plants_fraction": 2}},
])
def test_refusals(kwargs):
    with pytest.raises(ValueError):
        TR.refinery_model(**kwargs)
