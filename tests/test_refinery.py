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
  [ASF]  ASF Cuenta Pública 2016, auditoría 0474 «Contratos para la comercialización de coque», real coke production 2016
         (t) https://www.asf.gob.mx/Trans/Informes/IR2016ii/Documentos/Auditorias/2016_0474_a.pdf ; eBDI REPETRE annual 2016
  [BNE]  SENER, Balance Nacional de Energía 2024, Table 6.3 (own use of the energy sector and refinery output, PJ)
         https://odondebuenr.com.mx/wp-content/uploads/2026/04/Balance-Nacional-de-Energia-2024.pdf
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
    """Madero with the cutter fixed by the user (so diesel is free): conversion is identified by diesel inside the table
    (68.82 wt%, CALIBRATED, diesel no longer validation); Tula with the cutter fixed: not identifiable (at the 76.33 edge
    the diesel gap remains) -> central case 66.47, warned. Never more fitted parameters than published products."""
    m = TR.refinery_model("madero", today="2026-11-16", unit_overrides={"cutter_fraction": 0.0})
    c = m["calibrated_parameters"]["fcc_conversion_wt_pct"]
    assert c["value"] == pytest.approx(68.82, abs=0.02) and c["status"] == "resuelto" and c["bounds"] == [58.98, 76.33]
    assert abs(c["residual_rel"]) < 1e-6 and "CALIBRADO" in c["flag"]
    roles = {r["product"]: r["role"] for r in m["published_comparison"]["rows"]}
    assert roles["diesel"] == "ajustado" and roles["gasolina"] == "validación"
    t = TR.refinery_model("tula", today="2026-11-16", unit_overrides={"cutter_fraction": 0.0})
    for r in (m, t, TR.refinery_model("tula", today="2026-11-16")):
        n_pub = len([x for x in r["published_comparison"]["rows"] if x["role"] not in ("dato de entrada",
                                                                                       "sin dato publicado (N/D)")])
        assert len(r["calibrated_parameters"]) <= n_pub
    assert any("La conversión del FCC NO se calibra" in w for w in t["warnings"])
    assert t["fcc_conversion_identification"]["used_value"] == 66.47
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
    """[BDI][GLO][ASF] cutter_fraction CALIBRATED to fuel oil wherever fuel oil is free: no coker (Salina Cruz 0.509,
    Salamanca 0.248) or coker identified by published coke (Cadereyta 0.243). Diesel is the independent validation:
    Salina Cruz +92.5 % -> -5.5 %, Salamanca +44.4 % -> +8.5 %, Cadereyta +15.9 % -> -11.0 % (measured). At Tula the
    model already over-makes fuel oil with no cutter: cutter at its bound 0, flagged."""
    for ref, x, d in (("salina_cruz", 0.509, -5.51), ("salamanca", 0.2481, 8.54), ("cadereyta", 0.2432, -11.01)):
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
    c = t["calibrated_parameters"]["cutter_fraction"]
    assert c["value"] == 0.0 and c["status"].startswith("en el límite") and "cutter_fraction" in t["calibration_summary"]["at_bound"]


def test_coke_factor_derivation_and_coker_identified_by_coke():
    """[ASF][BDI] t/bbl of Pemex coke DERIVED from 2016 real production (ASF: Cadereyta 560,179.0 t, Madero 601,400.7 t,
    Minatitlán 770,747.5 t) over eBDI 2016 coke (12.556, 12.198, 16.148 Mbd), 366 days: 0.1219 / 0.1347 / 0.1304, pooled
    0.12908 t/bbl. The coker is fitted to published coke where coke is published and a coker exists; at Salamanca and
    Salina Cruz (coke published, no coker in the eBDI) coke is reported as not comparable."""
    y = M._load("unit_yields.json")["pemex_coke_t_per_bbl"]
    d = y["derivation"]
    per = {k: d["tonnes_2016"][k] / d["days_2016"] / (1000 * d["mbd_2016"][k]) for k in d["tonnes_2016"]}
    for k, v in per.items():
        assert v == pytest.approx(d["t_per_bbl"][k], abs=5e-5), k
    pooled = sum(d["tonnes_2016"].values()) / d["days_2016"] / (1000 * sum(d["mbd_2016"].values()))
    assert pooled == pytest.approx(y["value"], abs=5e-5) and min(per.values()) >= y["range"][0] - 1e-5
    for ref, util in (("tula", 0.2152), ("cadereyta", 0.409), ("madero", 0.445), ("minatitlan", 0.385), ("dos_bocas", 0.4254)):
        r = TR.refinery_model(ref, today="2026-11-16")
        c = r["calibrated_parameters"]["coker_utilization"]
        assert c["fitted_to"] == "coque" and c["value"] == pytest.approx(util, abs=0.003) and c["unit"].startswith("t/d"), ref
        row = {q["product"]: q for q in r["published_comparison"]["rows"]}["coque"]
        assert row["model_t_d"] == pytest.approx(row["published_t_d_derived"], rel=1e-6)
    s = TR.refinery_model("salamanca", today="2026-11-16")
    row = {q["product"]: q for q in s["published_comparison"]["rows"]}["coque"]
    assert row["role"].startswith("no comparable")


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
    """AS-OPERATED Tula: the coker is now identified by published COKE (9.025 Mbd x 0.12908 t/bbl, factor derived from ASF
    2016 tonnes / eBDI 2016 barrels) -> 0.215 (was 0.264 from fuel oil); fuel oil is then free for the cutter, but even
    with NO cutter the model makes 19.3 % more fuel oil than published, so cutter_fraction sits at its bound 0 (flagged):
    the cutter does not explain Tula's diesel. Validation [BDI Jan-Sep 2026]: gasoline -4.6 %, diesel +43.7 %, liquids
    +9.6 % (measured; previous rounds: diesel +41.5 fixed Jechura FCC, +44.3 KBR FCC)."""
    cal = tula_op["calibrated_parameters"]
    assert set(cal) == {"coker_utilization", "cutter_fraction", "kerosene_to_jet_fraction", "butanes_to_gas_plants_fraction"}
    for name, c in cal.items():
        assert 0.0 <= c["value"] <= 1.0 and "CALIBRADO" in c["flag"], name
    for name in ("coker_utilization", "kerosene_to_jet_fraction", "butanes_to_gas_plants_fraction"):
        assert cal[name]["status"] == "resuelto" and abs(cal[name]["residual_rel"]) < 1e-6, name
    assert cal["coker_utilization"]["fitted_to"] == "coque"
    assert cal["coker_utilization"]["value"] == pytest.approx(0.2152, abs=0.002)
    assert cal["cutter_fraction"]["value"] == 0.0 and cal["cutter_fraction"]["status"].startswith("en el límite")
    assert cal["cutter_fraction"]["residual_rel"] == pytest.approx(0.193, abs=0.005)
    assert "cutter_fraction" in tula_op["calibration_summary"]["at_bound"]
    assert cal["kerosene_to_jet_fraction"]["value"] == pytest.approx(0.536, abs=0.002)
    assert cal["butanes_to_gas_plants_fraction"]["value"] == pytest.approx(0.782, abs=0.002)
    assert tula_op["fcc_conversion_identification"]["status"].startswith("no calibrada: el diésel se reserva")
    assert "fcc_conversion_wt_pct" not in cal
    comp = tula_op["published_comparison"]
    roles = {r["product"]: r["role"] for r in comp["rows"]}
    assert roles == {"gasolina": "validación", "diesel": "validación", "turbosina": "ajustado", "combustoleo": "ajustado",
                     "glp": "ajustado", "asfalto": "dato de entrada", "coque": "ajustado"}
    val = {r["product"]: r["deviation_pct"] for r in comp["validation_rows"]}
    assert val["gasolina"] == pytest.approx(-4.64, abs=0.3)
    assert val["diesel"] == pytest.approx(43.70, abs=0.5)
    assert comp["liquids_deviation_pct"] == pytest.approx(9.64, abs=0.3)
    assert len(cal) <= len([r for r in comp["rows"] if r["role"] != "dato de entrada"])
    ec = tula_op["economics"]
    assert ec["gross_margin_usd_bbl"] == pytest.approx(21.92, abs=0.05)
    assert ec["net_margin_usd_bbl"] == pytest.approx(14.80, abs=0.05)
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
    # coker identified by published coke 29.336 Mbd -> 0.425 (before: all vacuum residue to the coker)
    assert cal["coker_utilization"]["fitted_to"] == "coque" and cal["coker_utilization"]["value"] == pytest.approx(0.4254, abs=0.002)
    assert dev["coque"] == pytest.approx(0.0, abs=1e-4)
    assert dev["gasolina"] == pytest.approx(-7.71, abs=0.3)
    assert dev["diesel"] == pytest.approx(-25.59, abs=0.3)
    assert dev["glp"] == pytest.approx(537.8, abs=2.0)
    assert comp["liquids_deviation_pct"] == pytest.approx(-11.04, abs=0.3)
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


def test_dossier_coker_what_if_holds_other_calibrations():
    """The dossier's «coker at 100 %» what-if changes only the coker: the other calibrated parameters are held (otherwise
    the cutter would be re-fitted to published fuel oil and absorb the change). Tula: positive, measured +105.5 M USD/y."""
    from packages.dossier import dossier as D
    base = M.run_model("tula", today=__import__("datetime").date(2026, 11, 16))
    ops = D._refinery_capacity_opportunities("tula", base, None, None)
    assert len(ops) == 1 and ops[0]["value_usd_y"] == pytest.approx(105.5e6, rel=0.02)
    assert "coque publicado" in ops[0]["explanation_es"]


def test_tula_liquid_excess_not_explained_by_own_use(tula_op):
    """[BNE] Tula: diesel excess +18.2 Mbd ~ total-liquid excess +18.1 Mbd (too much liquid, not mis-routing). Own use of
    liquid fuels of the WHOLE energy sector = 42.00 PJ over 1,445.32 PJ of refinery liquids (2.91 %; diesel 23.94/344.46 =
    6.95 %): even all of it inside refineries and excluded from Pemex's «elaboración» explains <= 5.6 Mbd (diesel 3.1).
    Not modelled as a parameter (no per-refinery figure); stated in warnings and calibration_summary."""
    d = tula_op["published_comparison"]["liquid_excess_diagnosis"]
    assert d["own_use_share_liquids_energy"] == pytest.approx(42.00 / 1445.32, rel=1e-9)
    assert d["own_use_share_diesel_energy"] == pytest.approx(23.94 / 344.46, rel=1e-9)
    assert d["diesel_excess_mbd"] == pytest.approx(18.16, abs=0.2)
    assert d["liquids_excess_mbd"] == pytest.approx(18.11, abs=0.2)
    assert d["own_use_upper_bound_liquids_mbd"] == pytest.approx(5.62, abs=0.05)
    assert d["own_use_upper_bound_diesel_mbd"] == pytest.approx(3.10, abs=0.05)
    assert d["own_use_upper_bound_liquids_mbd"] < 0.5 * d["liquids_excess_mbd"]
    assert "demasiado LÍQUIDO" in tula_op["calibration_summary"]["text_es"]
    assert "demasiado LÍQUIDO" in d["text_es"]
