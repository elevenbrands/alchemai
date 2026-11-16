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
  [KBR]  P. K. Niccum (KBR), «Diesel Creation in the FCC Centered Refinery», FCCU seminar Galveston 2013, Table 3
         (pilot plant, one feed 22.5 °API / 0.55 % S, three severities)
         https://refiningcommunity.com/wp-content/uploads/2017/07/Diesel-Creation-in-the-FCC-Centered-Refinery-Niccum-KBR-FCCU-Galveston-2013.pdf
  [IMP]  G. C. Laredo et al., Rev. Soc. Quím. Méx. 46(2) 115-119 (2002), LCO of six Pemex refineries
         https://www.scielo.org.mx/pdf/rsqm/v46n2/v46n2a7.pdf
  [GLO]  Pemex eBDI «Términos usados en la Industria Petrolera» (2006), entry «Combustóleo»: vacuum residue blended with
         other residues, «se utilizan diluyentes para ajustar las especificaciones», 475-550 SSF at 50 °C
         https://ebdi.pemex.com/bdi/docs/Diccio_marzo2006-4_en.pdf
  [IIR]  Industrial Info Resources, Dos Bocas, 5-feb-2026 (alkylation / butane isomerization still in start-up)
         https://www.industrialinfo.com/news/article/pemexs-dos-bocas-refinery-moves-toward-full-operations--352954
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
    out = M.fcc_example_unit(feed)
    published_vol = {"propane": 2.56, "propylene": 5.80, "isobutane": 5.59, "n_butane": 1.96, "butylenes": 7.61,
                     "gasoline": 57.05, "lco": 21.20, "hco": 6.80}
    dev = {k: abs(100 * out[k]["bpd"] / 25000.0 - v) for k, v in published_vol.items()}
    assert max(dev.values()) < 0.03, dev            # measured max 0.02 vol% points (rounding of the published table)
    # sulfur: published light-gas recovery 38.4 % of 39,575 lb/d -> 15,193 lb/d (our H2S-S); measured within 0.1 %
    assert out["acid_gas"]["s_kg_d"] / feed["s_kg_d"] == pytest.approx(38.4 / 100.1, rel=1e-12)


KBR_CONV = [58.98, 66.47, 76.33]
KBR_WT = {"dry_gas": [1.23, 2.08, 3.50], "c3_lpg": [2.97, 4.26, 7.27], "c4_lpg": [5.98, 7.88, 11.57],
          "gasoline": [43.21, 46.98, 46.00], "lco": [27.42, 24.47, 16.01], "slurry": [13.60, 9.06, 7.66],
          "coke": [5.59, 5.27, 7.99]}


def test_fcc_yields_reproduce_kbr_table():
    """[KBR] Table 3: the three published severities are reproduced exactly (21 yields, 6 olefinicities, 9 API gravities;
    published to 0.01 wt% / 0.1 °API; measured deviation 0)."""
    for i, x in enumerate(KBR_CONV):
        r = M.fcc_yields_wt(x)
        for k, v in KBR_WT.items():
            assert r["wt_pct"][k] == pytest.approx(v[i], abs=1e-9), (x, k)
        assert 100 * r["c3_olefinicity"] == pytest.approx([83.8, 83.8, 85.7][i], abs=1e-9)
        assert 100 * r["c4_olefinicity"] == pytest.approx([66.7, 68.5, 67.0][i], abs=1e-9)
        assert r["api"]["lco"] == pytest.approx([22.2, 17.0, 11.3][i], abs=1e-9)
        assert r["api"]["slurry"] == pytest.approx([6.0, -0.8, -7.4][i], abs=1e-9)
        assert r["api"]["gasoline"] == pytest.approx([56.6, 57.2, 55.9][i], abs=1e-9)


def test_fcc_closure_conversion_identity_and_monotonicity():
    """Consistency: yields sum to 100 wt% and conversion = 100 - LCO - slurry at every conversion of the table range; LCO,
    slurry fall and C3, C4, dry gas rise with conversion; gasoline rises then falls (max at the 66.47 case), as published."""
    xs = sorted([58.98 + i * (76.33 - 58.98) / 40 for i in range(41)] + [66.47])
    ys = [M.fcc_yields_wt(x)["wt_pct"] for x in xs]
    for x, y in zip(xs, ys):
        assert sum(y.values()) == pytest.approx(100.0, abs=1e-9)
        assert 100.0 - y["lco"] - y["slurry"] == pytest.approx(x, abs=1e-9)
    for k, sign in (("lco", -1), ("slurry", -1), ("c3_lpg", 1), ("c4_lpg", 1), ("dry_gas", 1)):
        assert all(sign * (b[k] - a[k]) > 0 for a, b in zip(ys, ys[1:])), k
    g = [y["gasoline"] for y in ys]
    assert max(g) == pytest.approx(46.98, abs=1e-9) and g[0] < max(g) and g[-1] < max(g)
    # unit level: a sour Maya-type feed (3 % S) closes in mass and sulfur; H2S is an extra lump, the rest scaled
    kpb = M.kg_per_bbl(0.93)
    feed = M._flow(50000.0, 50000.0 * kpb, 50000.0 * kpb * 0.03)
    for x in (58.98, 70.0, 76.33):
        out = M.fcc_unit(feed, x)
        prods = {k: v for k, v in out.items() if not k.startswith("_")}
        assert sum(f["kg_d"] for f in prods.values()) == pytest.approx(feed["kg_d"], rel=1e-12)
        assert sum(f["s_kg_d"] for f in prods.values()) == pytest.approx(feed["s_kg_d"], rel=1e-12)
        assert out["dry_gas"]["kg_d"] > 0
    with pytest.raises(ValueError):
        M.fcc_yields_wt(80.0)
    with pytest.raises(ValueError):
        TR.refinery_model("tula", unit_overrides={"fcc_conversion_wt_pct": 50.0}, mode="nameplate")


def test_fcc_kbr_table_vs_jechura_example_cross_check():
    """Independent cross-check of two open sources ([KBR] pilot plant 22.5 °API vs [FCC] Jechura/Gary correlations 25 °API):
    at Jechura's conversion (100 - 20.60 - 7.84 = 71.56 wt%) the KBR table gives LCO 20.10 vs 20.60 wt% (-2.4 %), gasoline
    46.47 vs 47.15, C4 9.79 vs 9.81, slurry/HCO 8.34 vs 7.84, C3 5.81 vs 4.77, coke 6.67 vs 4.90 (measured)."""
    ex = M._load("unit_yields.json")["fcc"]["products"]
    x = 100.0 - ex["lco"]["wt_pct"] - ex["hco"]["wt_pct"]
    y = M.fcc_yields_wt(x)["wt_pct"]
    assert y["lco"] == pytest.approx(ex["lco"]["wt_pct"], rel=0.03)
    assert y["gasoline"] == pytest.approx(ex["gasoline"]["wt_pct"], rel=0.02)
    assert y["c4_lpg"] == pytest.approx(sum(ex[k]["wt_pct"] for k in ("isobutane", "n_butane", "butylenes")), rel=0.01)
    assert y["slurry"] == pytest.approx(ex["hco"]["wt_pct"], rel=0.07)
    assert y["coke"] == pytest.approx(6.67, abs=0.01)          # 36 % above Jechura's 4.90: the sources differ on coke


def test_lco_limited_by_hds_capacity_excess_to_fuel_oil():
    """[IMP] LCO reaches diesel only within hydrodesulfurization capacity: with hds_total cut to 150 Mbd at Tula the LCO that
    does not fit goes to fuel oil; LCO is conserved and the diesel pool is still on spec."""
    base = TR.refinery_model("tula", today="2026-11-16", mode="nameplate")
    low = TR.refinery_model("tula", today="2026-11-16", mode="nameplate", unit_overrides={"capacity_mbd": {"hds_total": 150}})
    u0 = {u["unit"]: u for u in base["unit_table"]}
    u1 = {u["unit"]: u for u in low["unit_table"]}
    assert u0["hds_diesel"]["lco_to_fuel_oil_bpd"] == 0.0
    lco = u1["fcc"]["products_bpd"]["lco"]
    assert u1["hds_diesel"]["lco_to_diesel_bpd"] + u1["hds_diesel"]["lco_to_fuel_oil_bpd"] == pytest.approx(lco, rel=1e-12)
    assert u1["hds_diesel"]["lco_to_fuel_oil_bpd"] > 0
    d_fo = low["product_slate"]["combustoleo"]["bpd"] - base["product_slate"]["combustoleo"]["bpd"]
    assert d_fo == pytest.approx(u1["hds_diesel"]["lco_to_fuel_oil_bpd"], rel=1e-9)
    assert low["product_slate"]["diesel"]["sulfur_mg_kg"] == pytest.approx(15.0, rel=1e-6)
    assert low["evidence"]["verified"]
    assert any(s["id"] == "s_lco_combustoleo" for s in low["process_model"]["streams"])


def test_fcc_conversion_calibration_flags_and_bounds():
    """Madero: conversion is identified by diesel inside the table (69.60 wt%, CALIBRATED, diesel no longer validation);
    Tula: not identifiable (at the 76.33 edge the diesel gap remains) -> central case 66.47, warned. Never more fitted
    parameters than published (non-input) products."""
    m = TR.refinery_model("madero", today="2026-11-16")
    c = m["calibrated_parameters"]["fcc_conversion_wt_pct"]
    assert c["value"] == pytest.approx(69.60, abs=0.02) and c["status"] == "resuelto" and c["bounds"] == [58.98, 76.33]
    assert abs(c["residual_rel"]) < 1e-6 and "CALIBRADO" in c["flag"]
    roles = {r["product"]: r["role"] for r in m["published_comparison"]["rows"]}
    assert roles["diesel"] == "ajustado" and roles["gasolina"] == "validación"
    for r in (m, TR.refinery_model("tula", today="2026-11-16")):
        n_pub = len([x for x in r["published_comparison"]["rows"] if x["role"] not in ("dato de entrada",
                                                                                       "sin dato publicado (N/D)")])
        assert len(r["calibrated_parameters"]) <= n_pub
    t = TR.refinery_model("tula", today="2026-11-16")
    assert any("La conversión del FCC NO se calibra" in w for w in t["warnings"])
    u = TR.refinery_model("tula", today="2026-11-16", unit_overrides={"fcc_conversion_wt_pct": 70.0})
    assert "fcc_conversion_wt_pct" not in u["calibrated_parameters"]
    assert {x["unit"]: x for x in u["unit_table"]}["fcc"]["conversion_wt_pct"] == 70.0


def test_cutter_fraction_conserves_and_moves_distillate_to_fuel_oil():
    """[GLO] Cutter stock: a share of the middle-distillate pool goes untreated to fuel oil; diesel + fuel oil volume is
    conserved (the diesel hydrotreater keeps volume), sulfur closes, and the cutter shows as streams on the diagram."""
    a = TR.refinery_model("salina_cruz", today="2026-11-16", mode="nameplate")
    b = TR.refinery_model("salina_cruz", today="2026-11-16", mode="nameplate", unit_overrides={"cutter_fraction": 0.3})
    pa, pb = a["product_slate"], b["product_slate"]
    hd = {u["unit"]: u for u in b["unit_table"]}["hds_diesel"]
    assert pb["combustoleo"]["bpd"] - pa["combustoleo"]["bpd"] == pytest.approx(hd["cutter_to_fuel_oil_bpd"], rel=1e-9)
    assert pa["diesel"]["bpd"] - pb["diesel"]["bpd"] == pytest.approx(hd["cutter_to_fuel_oil_bpd"], rel=1e-9)
    assert b["evidence"]["verified"] and pb["diesel"]["sulfur_mg_kg"] == pytest.approx(15.0, rel=1e-6)
    ids = {x["id"] for x in b["process_model"]["streams"]}
    assert {"s_diluente_primario", "s_lco_combustoleo"} <= ids
    with pytest.raises(ValueError):
        TR.refinery_model("tula", unit_overrides={"cutter_fraction": 1.2}, mode="nameplate")


def test_cutter_calibrated_only_where_fuel_oil_is_free_diesel_is_validation():
    """[BDI][GLO] Salina Cruz and Salamanca (no coker): cutter_fraction CALIBRATED to fuel oil (0.509, 0.248) and diesel is
    the independent validation: Salina Cruz diesel +92.5 % -> -5.5 %, Salamanca +44.4 % -> +8.5 % (measured). Where the
    coker already uses fuel oil (Tula) the cutter is not identifiable and stays 0, warned."""
    for ref, x, d in (("salina_cruz", 0.509, -5.51), ("salamanca", 0.2481, 8.54)):
        r = TR.refinery_model(ref, today="2026-11-16")
        c = r["calibrated_parameters"]["cutter_fraction"]
        assert c["value"] == pytest.approx(x, abs=0.002) and c["status"] == "resuelto" and abs(c["residual_rel"]) < 1e-6
        roles = {q["product"]: q["role"] for q in r["published_comparison"]["rows"]}
        assert roles["combustoleo"] == "ajustado" and roles["diesel"] == "validación"
        dev = {q["product"]: q["deviation_pct"] for q in r["published_comparison"]["rows"]}
        assert dev["diesel"] == pytest.approx(d, abs=0.3)
        assert "fcc_conversion_wt_pct" not in r["calibrated_parameters"]
        assert r["fcc_conversion_identification"]["status"].startswith("no calibrada: el diésel se reserva")
    t = TR.refinery_model("tula", today="2026-11-16")
    assert "cutter_fraction" not in t["calibrated_parameters"]
    assert any(w.startswith("cutter_fraction NO se calibra") for w in t["warnings"])


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
    gasoline +5.9 %, jet +86.5 % (all kerosene to jet), diesel +3.1 %, fuel oil -76.2 % (coker at full feed vs partial
    start-up), LPG +72.4 %, total liquids +8.8 % (FCC at the KBR central case 66.47 wt%; before M27 R2 with the fixed
    Jechura vector: +9.8 / +86.5 / +0.3 / -78.4 / +66.4 / +9.5)."""
    rows = {r["product"]: r["deviation_pct"] for r in tula["published_comparison"]["rows"]}
    expected = {"gasolina": 5.85, "turbosina": 86.47, "diesel": 3.14, "combustoleo": -76.22, "glp": 72.39, "asfalto": 0.0}
    for k, v in expected.items():
        assert rows[k] == pytest.approx(v, abs=0.3), (k, rows[k])
    assert tula["published_comparison"]["liquids_deviation_pct"] == pytest.approx(8.83, abs=0.3)


def test_as_operated_calibration_and_validation(tula_op):
    """AS-OPERATED: three CALIBRATED parameters (one per product, within 0..1) reproduce fuel oil, jet and LPG; the FCC
    conversion is NOT identifiable by diesel (even at the table edge 76.33 wt% the model makes 54.6 vs 41.6 Mbd), so it
    stays at the KBR central case and diesel stays validation [BDI Jan-Sep 2026]: gasoline -2.6 %, diesel +44.3 %, total
    liquids +8.6 % (measured, not tuned; with the fixed Jechura vector it was +0.8 / +41.5 / +9.6: diesel got WORSE by
    2.8 points because the KBR central case makes more LCO, 24.5 vs 20.6 wt%)."""
    cal = tula_op["calibrated_parameters"]
    assert set(cal) == {"coker_utilization", "kerosene_to_jet_fraction", "butanes_to_gas_plants_fraction"}
    for name, c in cal.items():
        assert 0.0 <= c["value"] <= 1.0 and c["status"].startswith("resuelto"), name
        assert abs(c["residual_rel"]) < 1e-6, name
        assert "CALIBRADO" in c["flag"]
    assert cal["coker_utilization"]["value"] == pytest.approx(0.264, abs=0.002)
    assert cal["kerosene_to_jet_fraction"]["value"] == pytest.approx(0.536, abs=0.002)
    assert cal["butanes_to_gas_plants_fraction"]["value"] == pytest.approx(0.842, abs=0.002)
    conv = tula_op["fcc_conversion_identification"]
    assert conv["value_at_bound"] == 76.33 and conv["used_value"] == 66.47
    assert conv["diesel_model_bpd_at_bound"] == pytest.approx(54554, rel=2e-3)
    assert "fcc_conversion_wt_pct" not in cal
    comp = tula_op["published_comparison"]
    roles = {r["product"]: r["role"] for r in comp["rows"]}
    assert roles == {"gasolina": "validación", "diesel": "validación", "turbosina": "ajustado", "combustoleo": "ajustado",
                     "glp": "ajustado", "asfalto": "dato de entrada"}
    val = {r["product"]: r["deviation_pct"] for r in comp["validation_rows"]}
    assert val["gasolina"] == pytest.approx(-2.61, abs=0.3)
    assert val["diesel"] == pytest.approx(44.30, abs=0.5)
    assert comp["liquids_deviation_pct"] == pytest.approx(8.56, abs=0.3)
    assert len(cal) <= len([r for r in comp["rows"] if r["role"] != "dato de entrada"])
    ec = tula_op["economics"]
    assert ec["gross_margin_usd_bbl"] == pytest.approx(22.17, abs=0.05)
    assert ec["net_margin_usd_bbl"] == pytest.approx(15.02, abs=0.05)
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


def test_dos_bocas_unpublished_products_and_bounds():
    """[BDI] Dos Bocas Jan-Sep 2026 publishes jet, fuel oil and asphalt as N/D: they are neither compared nor calibrated;
    the kerosene split is fitted to diesel instead and hits its bound (diesel 59.0 vs 78.2 Mbd, -24.5 %; was -25.8 %),
    so the FCC conversion has no free product and is not calibrated; [IIR] alkylation off in 2026; the butanes split hits
    its bound (LPG 11.56 vs 1.61 Mbd, +619 %; was +535 % with alkylation on). Validation: gasoline -0.1 % (was +11.4 %),
    published-liquids basis -5.9 % (was -1.8 %)."""
    r = TR.refinery_model("dos_bocas")
    comp, cal, summ = r["published_comparison"], r["calibrated_parameters"], r["calibration_summary"]
    roles = {x["product"]: x["role"] for x in comp["rows"]}
    assert roles["turbosina"] == roles["combustoleo"] == roles["asfalto"] == "sin dato publicado (N/D)"
    assert all(x["deviation_pct"] is None for x in comp["rows"] if x["published_mbd"] is None)
    assert cal["kerosene_to_jet_fraction"]["fitted_to"] == "diesel" and cal["kerosene_to_jet_fraction"]["value"] == 0.0
    assert set(summ["at_bound"]) == {"kerosene_to_jet_fraction", "butanes_to_gas_plants_fraction"}
    assert not summ["all_resolved"] and "EN EL LÍMITE" in summ["text_es"]
    dev = {x["product"]: x["deviation_pct"] for x in comp["rows"]}
    assert dev["gasolina"] == pytest.approx(-0.06, abs=0.3)
    assert dev["diesel"] == pytest.approx(-24.51, abs=0.3)
    assert dev["glp"] == pytest.approx(618.6, abs=2.0)
    assert comp["liquids_deviation_pct"] == pytest.approx(-5.92, abs=0.3)
    assert r["fcc_conversion_identification"]["status"].startswith("no calibrada: el diésel ya lo ajusta")
    assert not any(u["unit"] == "alkylation" for u in r["unit_table"])
    assert any("alkylation_isomerization: utilización = 0.0" in a for a in r["assumptions"])
    assert r["product_slate"]["butanos"]["bpd"] > 0
    hds = {u["id"]: u["params"] for u in r["process_model"]["units"] if u["id"].startswith("hds_")}
    assert hds["hds_diesel"]["capacity_bpd"] == 68000.0 and hds["hds_naftas"]["capacity_bpd"] == 55000.0
    assert all(p["utilization_of_capacity"] is not None for p in hds.values())
    assert r["product_slate"]["diesel"]["sulfur_mg_kg"] == pytest.approx(15.0, rel=1e-6)
    json.dumps(r)


def test_hydrotreater_over_capacity_bypasses_untreated_not_lost():
    """A hydrotreater with its own published capacity never loses product: the excess goes untreated (off-spec flag)."""
    a = TR.refinery_model("dos_bocas", utilization={"hds_total": 0.5}, mode="nameplate")
    b = TR.refinery_model("dos_bocas", mode="nameplate")
    u = {x["unit"]: x for x in a["unit_table"]}["hds_naftas"]
    assert u["bypass_untreated_bpd"] > 0
    assert u["products_bpd"]["product"] == pytest.approx(u["feed_bpd"], rel=1e-12)
    assert a["product_slate"]["diesel"]["bpd"] == pytest.approx(b["product_slate"]["diesel"]["bpd"], rel=1e-9)
    assert any("FUERA DE ESPECIFICACIÓN" in w for w in a["warnings"])
    assert a["balances"]["overall_mass_closure_rel"] < 1e-12 and a["balances"]["sulfur_closure_rel"] < 1e-12


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
    # pool average slightly above 30 mg/kg: the reformer keeps the hydrotreated naphtha's sulfur in a smaller C5+ mass
    # (model property, measured 31.02 with the KBR FCC table; 30 is the NOM average spec)
    assert ps["gasolina"]["sulfur_mg_kg"] < 31.1


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
