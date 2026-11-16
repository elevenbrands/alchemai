"""Liquid-side equipment (bloque M26_E1): textbook worked examples, published tables, analytical limits, conservation, refusals.

Sources, all read on 2026-11-16:
  [GK8]  Geankoplis example 8.5-1 (triple-effect evaporator, sugar solution). Data: FOSSEE Python Textbook Companion, ch. 8,
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter08.ipynb
         Solution of the BOOK (second trial: L, V, S, q, A, T, steam economy) as reproduced slide by slide in
         https://www.slideshare.net/AKhapre/evaporationppt-254161500 (slides 35-50). The companion's own printed areas
         (111.4 / 113.7 / 111.6 m2, economy 1.999) are NOT the reference: its third-effect balance uses T2 where the book
         uses T3 (`b2 = L3*Cp3*T2 - L3*H3`), which is why it differs from the book's 105.0 m2.
  [GK12] same companion, ch. 12: examples 12.5-1 / 12.5-2 (one equilibrium stage, acetic acid-water-isopropyl ether, phase
         compositions read from the book's figure) and 12.7-3 (nicotine-water-kerosene, 5 stages by stepping):
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter12.ipynb
  [MC]   FOSSEE Scilab Textbook Companion, McCabe-Smith-Harriott (book 839): example 20.3 (acetone-water-MIK countercurrent,
         99 % recovery, 3.4 ideal stages) and example 2.2 (decanter hold-up volume):
         https://raw.githubusercontent.com/FOSSEE/Scilab-TBC-Uploads/master/839/CH20/EX20.3/Example_20_3.sce
         https://raw.githubusercontent.com/FOSSEE/Scilab-TBC-Uploads/master/839/CH2/EX2.2/Example_2_2.sce
  [LEE]  Lee, Lo & Lin, Fluid Phase Equilib. 299 (2010) 180-190: tie-lines glycerol + methanol + methyl oleate at 298.2,
         308.2 and 318.2 K, 101 kPa, from the NIST ThermoML archive (data set 5 of the file):
         https://trc.nist.gov/ThermoML/10.1016/j.fluid.2010.10.010.xml
  [DEC]  S. D. Al-Bayati, «Liquid-liquid separators» (University of Baghdad lecture, Coulson & Richardson vol. 6 method),
         worked examples 1 (vertical) and 2 (horizontal): https://uomus.edu.iq/img/lectures21/MUCLecture_2024_4232530.pdf
  [NETL] DOE/NETL-2002/1169, Loh, Lyons & White, «Process Equipment Cost Estimation» (1st quarter 1998 USD): tables
         «Storage Tanks» and «Evaporator», and Table 11 (CEPCI 1998 = 389.5): https://www.osti.gov/servlets/purl/797810
  [HUM]  NREL/TP-5100-47764 (Humbird et al. 2011), Appendix A equipment list and Table 23 (T-701, T-710, T-703):
         https://www.osti.gov/servlets/purl/1013269
BioSTEAM reproducing BioSTEAM is not used as validation anywhere; BioSTEAM's numbers appear only as correlations under
test against [NETL] / [HUM] or as a second opinion."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.bioprocess import liquid_equipment as LE  # noqa: E402
from packages.dwsim_client import tools_liquid_equipment as T  # noqa: E402

P = 101325.0
GAL = 0.003785411784
FT = 0.3048

# ---------------------------------------------------------------------------------------------------------------------
# multiple-effect evaporator
# ---------------------------------------------------------------------------------------------------------------------
GK851 = dict(feed_kg_s=22680.0 / 3600.0, feed_solids_mass_fraction=0.10, feed_T_K=26.7 + 273.15,
             product_solids_mass_fraction=0.50, n_effects=3, U_W_m2K=[3123.0, 1987.0, 1136.0],
             solution_cp_J_kgK=[4190.0, -2350.0], boiling_point_rise={"polynomial_K": [1.78, 6.22]},
             steam_T_K=121.1 + 273.15, last_effect_saturation_T_K=51.67 + 273.15)        # [GK8]
# book, second trial [GK8 slides 45-50]
GK851_BOOK = {"L_kg_h": [17005.0, 10952.0, 4536.0], "V_kg_h": [5675.0, 6053.0, 6416.0], "S_kg_h": 8960.0,
              "q_W": [5.476e6, 3.539e6, 3.855e6], "A_m2": [104.6, 105.6, 104.9], "A_mean_m2": 105.0,
              "T_C": [104.33, 87.11, 54.12], "economy": 2.025}


@pytest.fixture(scope="module")
def gk851():
    return T.multi_effect_evaporator(**GK851)


def test_evaporator_geankoplis_8_5_1(gk851):
    r = gk851
    assert r["status"] == "ok" and r["evidence"]["verified"]
    # measured: mean area -0.04 %, steam -0.04 %, economy +0.04 %, L and V within 0.14 %, duties within 0.10 %, T within 0.08 K
    assert r["mean_area_m2"] == pytest.approx(GK851_BOOK["A_mean_m2"], rel=2e-3)
    assert r["steam_kg_s"] * 3600 == pytest.approx(GK851_BOOK["S_kg_h"], rel=2e-3)
    assert r["steam_economy"] == pytest.approx(GK851_BOOK["economy"], rel=2e-3)
    for e, lb, vb, qb, ab, tb in zip(r["effects"], GK851_BOOK["L_kg_h"], GK851_BOOK["V_kg_h"], GK851_BOOK["q_W"],
                                     GK851_BOOK["A_m2"], GK851_BOOK["T_C"]):
        assert e["liquid_out_kg_s"] * 3600 == pytest.approx(lb, rel=2.5e-3)
        assert e["vapor_kg_s"] * 3600 == pytest.approx(vb, rel=2.5e-3)
        assert e["duty_W"] == pytest.approx(qb, rel=2e-3)
        assert e["area_m2"] == pytest.approx(ab, rel=7e-3)       # the book stops after two trials: its areas still differ by 1 %
        assert e["boiling_T_K"] - 273.15 == pytest.approx(tb, abs=0.12)
    ev = r["evidence"]
    assert ev["overall_enthalpy_balance_rel"] < 1e-10 and ev["equal_area_spread_rel"] < 1e-9
    assert ev["initial_guess_independence_rel"] < 1e-8
    json.dumps(r)


def test_evaporator_single_effect_analytical():
    """One effect, no boiling-point rise, feed at its boiling point, cp independent of x: the enthalpy balance reduces to
    S·λs = V·(Hg(T1) − cp·(T1 − 273.15)) and A = S·λs/(U·(Ts − T1)) (derived here, steam tables from CoolProp)."""
    from CoolProp.CoolProp import PropsSI
    ts, t1, cp, u, f = 393.15, 353.15, 4000.0, 2000.0, 2.0
    r = T.multi_effect_evaporator(feed_kg_s=f, feed_solids_mass_fraction=0.1, feed_T_K=t1, product_solids_mass_fraction=0.4,
                                  n_effects=1, U_W_m2K=[u], solution_cp_J_kgK=[cp], boiling_point_rise={"polynomial_K": [0.0]},
                                  steam_T_K=ts, last_effect_saturation_T_K=t1)
    lam = PropsSI("H", "T", ts, "Q", 1, "Water") - PropsSI("H", "T", ts, "Q", 0, "Water")
    v = f * (1 - 0.1 / 0.4)
    s = v * (PropsSI("H", "T", t1, "Q", 1, "Water") - cp * (t1 - 273.15)) / lam
    assert r["evidence"]["verified"]
    assert r["steam_kg_s"] == pytest.approx(s, rel=1e-9)
    assert r["effects"][0]["area_m2"] == pytest.approx(s * lam / (u * (ts - t1)), rel=1e-9)
    assert r["total_evaporation_kg_s"] == pytest.approx(v, rel=1e-12)


def test_evaporator_identities(gk851):
    base = {k: v for k, v in GK851.items() if k not in ("n_effects", "U_W_m2K")}
    economy = []
    for n in (1, 2, 3, 4):
        r = T.multi_effect_evaporator(n_effects=n, U_W_m2K=[2000.0] * n, **base)
        assert r["evidence"]["verified"]
        economy.append(r["steam_economy"])
        temps = [e["boiling_T_K"] for e in r["effects"]]
        assert temps == sorted(temps, reverse=True)
    assert economy == sorted(economy) and all(e < n for e, n in zip(economy, (1, 2, 3, 4)))
    # the pressures of the equal-area solution, given back as a specification, return the same areas and steam
    p = [e["pressure_Pa"] for e in gk851["effects"]]
    args = {k: v for k, v in GK851.items() if k != "last_effect_saturation_T_K"}
    r = T.multi_effect_evaporator(effect_P_Pa=p, **args)
    assert r["evidence"]["verified"] and not r["equal_area"]
    assert r["steam_kg_s"] == pytest.approx(gk851["steam_kg_s"], rel=1e-8)
    assert [e["area_m2"] for e in r["effects"]] == pytest.approx([e["area_m2"] for e in gk851["effects"]], rel=1e-7)
    # a Dühring line of slope 1 and constant intercept is a constant boiling-point rise
    a = T.multi_effect_evaporator(**{**GK851, "boiling_point_rise": {"per_effect_K": [2.0, 2.0, 2.0]}})
    b = T.multi_effect_evaporator(**{**GK851, "boiling_point_rise": {
        "duhring": {"solids_mass_fraction": [0.0, 0.6], "slope": [1.0, 1.0], "intercept_K": [2.0, 2.0]}}})
    assert a["steam_kg_s"] == pytest.approx(b["steam_kg_s"], rel=1e-9)
    assert a["mean_area_m2"] > gk851["mean_area_m2"]          # 6 K of total rise instead of 3.4 K: less driving force
    # Dühring data that do not reach the product concentration: reported, never extrapolated
    c = T.multi_effect_evaporator(**{**GK851, "boiling_point_rise": {
        "duhring": {"solids_mass_fraction": [0.0, 0.3], "slope": [1.0, 1.0], "intercept_K": [2.0, 2.0]}}})
    assert not c["evidence"]["verified"] and c["status"] == "not_verified"
    assert any("Dühring" in x for x in c["evidence"]["reasons"])


def test_evaporator_refusals():
    for missing in ("feed_kg_s", "feed_T_K", "U_W_m2K", "solution_cp_J_kgK", "boiling_point_rise", "n_effects"):
        args = dict(GK851)
        args[missing] = None
        with pytest.raises(ValueError, match="Falta|debe"):
            LE.multi_effect_evaporator(**args)
    with pytest.raises(ValueError, match="steam_T_K"):
        LE.multi_effect_evaporator(**{**GK851, "steam_T_K": None})
    with pytest.raises(ValueError, match="último"):
        LE.multi_effect_evaporator(**{**GK851, "last_effect_saturation_T_K": None})
    with pytest.raises(ValueError, match="fuerza impulsora"):
        LE.multi_effect_evaporator(**{**GK851, "steam_T_K": 326.0})
    with pytest.raises(ValueError, match="effect_P_Pa"):
        LE.multi_effect_evaporator(**{**GK851, "equal_area": False})
    with pytest.raises(ValueError, match="type"):
        LE.multi_effect_evaporator(**GK851, costing={})


# [NETL] table «Evaporator», purchased equipment cost, 1st quarter 1998 USD: area ft2 -> (vertical tube, horizontal tube)
NETL_EVAPORATOR = {100: (62600, 34500), 500: (151600, 81100), 1000: (221900, 117100), 2000: (324700, 169000),
                   3000: (405700, 209500), 4000: (475200, 244100), 5000: (537100, 274400), 6000: (593700, 302600)}


def test_evaporator_cost_table_and_biosteam_second_opinion(gk851):
    """The cost IS the published table (so its points are reproduced by construction); what is measured here is how far
    BioSTEAM 2.54's own evaporator cost functions are from that table: 0.06-0.12 of it, hence second opinion only."""
    warnings: list[str] = []
    for kind, col in (("vertical_tube", 0), ("horizontal_tube", 1)):
        areas = [a * LE.FT2_M2 for a in NETL_EVAPORATOR]
        c = LE._evaporator_cost(areas, {"type": kind}, 2023, LE.NETL_CEPCI_1998, warnings)
        for row, ref in zip(c["effects"], NETL_EVAPORATOR.values()):
            assert row["purchase_cost_usd"] == pytest.approx(ref[col], rel=1e-9) and row["within_table_range"]
            assert 0.05 < row["biosteam_to_table_ratio"] < 0.13
        assert c["F_BM"] == 2.45
    assert any("BioSTEAM" in w and "NO se usa" in w for w in warnings)
    r = T.multi_effect_evaporator(**GK851, costing={"type": "vertical_tube", "material_factor": 2.0}, year=2023)
    assert r["cost"]["purchase_cost_usd"] == pytest.approx(2.0 * r["cost"]["base_purchase_cost_usd"])
    assert len(r["equipment"]) == 3 and all(e["type"] == "evaporator" for e in r["equipment"])
    # geometric interpolation between the 1,000 and 2,000 ft2 rows of the table, scaled 1998 -> 2023
    a = gk851["effects"][0]["area_m2"] / LE.FT2_M2
    ref = 221900 * (324700 / 221900) ** (math.log(a / 1000) / math.log(2)) * 797.9 / 389.5
    assert r["cost"]["effects"][0]["base_purchase_cost_usd"] == pytest.approx(ref, rel=1e-9)
    out = LE._evaporator_cost([5.0], {"type": "vertical_tube"}, 2023, None, warnings)
    assert not out["effects"][0]["within_table_range"] and any("EXTRAPOLADO" in w for w in warnings)
    json.dumps(r)


# ---------------------------------------------------------------------------------------------------------------------
# liquid-liquid extraction, constant distribution coefficients
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k, s_over_f, n, solvent_solute", [(2.0, 1.5, 4, 0.0), (0.5, 2.0, 3, 0.0), (0.4, 1.0, 6, 0.0),
                                                            (0.9, 2.0, 5, 0.002), (3.0, 0.2, 2, 0.0), (1.2, 1.0, 1, 0.0),
                                                            (1.0, 1.0, 4, 0.0)])
def test_extraction_kremser_analytical(k, s_over_f, n, solvent_solute):
    """Kremser, written out independently: (X_N − X*)/(X_0 − X*) = (E − 1)/(E^(N+1) − 1), 1/(N + 1) if E = 1."""
    fc, x0 = 10.0, 0.05
    s = s_over_f * fc
    r = T.liquid_liquid_extraction("countercurrent", {"water": fc, "A": fc * x0}, "kg_s", {"S": s, "A": solvent_solute},
                                   distribution_coefficients={"A": k}, carrier="water", extraction_solvent="S", n_stages=n)
    e = k * s / fc
    x_star = solvent_solute / (s * k)
    frac = 1.0 / (n + 1) if abs(e - 1) < 1e-12 else (e - 1.0) / (e ** (n + 1) - 1.0)
    assert r["solutes"]["A"]["raffinate_ratio"] == pytest.approx(x_star + (x0 - x_star) * frac, rel=1e-10)
    assert r["evidence"]["verified"] and r["evidence"]["kremser_rel_diff"] < 1e-10
    total = fc * x0 + solvent_solute
    assert r["raffinate"]["flows"]["A"] + r["extract"]["flows"]["A"] == pytest.approx(total, rel=1e-12)
    assert r["raffinate"]["flows"]["water"] == fc and r["extract"]["flows"]["S"] == s
    json.dumps(r)


def test_extraction_constant_k_limits_and_stage_count():
    feed, solv = {"water": 1.0, "A": 0.02}, {"S": 2.0}
    kw = dict(distribution_coefficients={"A": 1.5}, carrier="water", extraction_solvent="S")
    one = T.liquid_liquid_extraction("decanter", feed, "kg_s", solv, **kw)
    assert one["solutes"]["A"]["raffinate_ratio"] == pytest.approx(0.02 / (1 + 1.5 * 2.0), rel=1e-12)     # X1 = X0/(1 + E)
    same = T.liquid_liquid_extraction("countercurrent", feed, "kg_s", solv, n_stages=1, **kw)
    assert same["solutes"]["A"]["raffinate_ratio"] == pytest.approx(one["solutes"]["A"]["raffinate_ratio"], rel=1e-12)
    rec = [T.liquid_liquid_extraction("countercurrent", feed, "kg_s", solv, n_stages=n, **kw)["solutes"]["A"]
           ["fraction_extracted"] for n in (1, 2, 4, 8)]
    assert rec == sorted(rec) and rec[-1] == pytest.approx(1 - 2 / (3 ** 9 - 1), rel=1e-12)         # E = 3, N = 8
    # E < 1: the unextracted fraction cannot go below 1 − E however many stages
    low = T.liquid_liquid_extraction("countercurrent", feed, "kg_s", {"S": 0.4}, n_stages=60, **kw)
    assert low["solutes"]["A"]["fraction_extracted"] == pytest.approx(0.6, rel=1e-6)
    t = T.liquid_liquid_extraction("countercurrent", feed, "kg_s", {"S": 0.4}, n_stages=3,
                                   target={"solute": "A", "raffinate_ratio": 0.02 * 0.3}, **kw)
    assert t["stages_required"]["reachable"] is False
    # the stage count asked back from the tool's own N-stage answer is N
    five = T.liquid_liquid_extraction("countercurrent", feed, "kg_s", solv, n_stages=5, **kw)
    back = T.liquid_liquid_extraction("countercurrent", feed, "kg_s", solv, n_stages=5,
                                      target={"solute": "A", "raffinate_ratio": five["solutes"]["A"]["raffinate_ratio"]}, **kw)
    assert back["stages_required"]["stages"] == pytest.approx(5.0, rel=1e-9)
    # a non-distributing solute (K = 0) stays in the raffinate
    z = T.liquid_liquid_extraction("decanter", feed, "kg_s", solv, distribution_coefficients={"A": 0.0}, carrier="water",
                                   extraction_solvent="S")
    assert z["extract"]["flows"]["A"] == 0.0 and z["raffinate"]["flows"]["A"] == pytest.approx(0.02)


def test_extraction_geankoplis_12_7_3_nicotine():
    """[GK12] 100 kg/h of water with 1 % nicotine, 200 kg/h of kerosene with 0.05 %, raffinate down to 0.1 %: the companion
    steps 5 theoretical stages on the tabulated equilibrium. The table's slope y/x is 0.798 at its dilute end (x = 0.00101)
    and 0.916 at the top (x = 0.0202); with a constant K the tool brackets the stepped count: 4.87 and 3.77 stages."""
    x, y = [0.001010, 0.00246, 0.005, 0.00746, 0.00988, 0.0202], [0.000806, 0.001959, 0.00454, 0.00682, 0.00904, 0.0185]
    k_low, k_high = y[0] / x[0], y[-1] / x[-1]
    out = {}
    for k in (k_low, k_high):
        r = T.liquid_liquid_extraction("countercurrent", {"water": 99.0, "nicotine": 1.0}, "kg_s",
                                       {"kerosene": 199.9, "nicotine": 0.1}, distribution_coefficients={"nicotine": k},
                                       carrier="water", extraction_solvent="kerosene", n_stages=5,
                                       target={"solute": "nicotine", "raffinate_ratio": 0.001 / 0.999})
        assert r["evidence"]["verified"]
        out[k] = r
    assert out[k_low]["stages_required"]["stages"] == pytest.approx(4.87, abs=0.02)
    assert out[k_low]["stages_required"]["ideal_stages_integer"] == 5                     # the companion's stepped count
    assert out[k_high]["stages_required"]["stages"] == pytest.approx(3.77, abs=0.02)
    assert out[k_low]["solutes"]["nicotine"]["raffinate_ratio"] < 0.001 / 0.999            # 5 stages meet the specification
    assert out[k_low]["raffinate"]["flows"]["water"] == 99.0 and out[k_low]["extract"]["flows"]["kerosene"] == 199.9


# ---------------------------------------------------------------------------------------------------------------------
# liquid-liquid extraction with the repo flash engine
# ---------------------------------------------------------------------------------------------------------------------
def test_decanter_geankoplis_12_5_1_unifac_lle():
    """[GK12] 30 kg isopropyl ether + 10 kg acetic acid + 60 kg water; tie-line READ FROM THE BOOK'S FIGURE: extract
    (0.04 acid, 0.02 water, 0.94 ether), raffinate (0.12, 0.86, 0.02). The companion does not print the temperature: 293.15 K
    is assumed here. Measured with UNIFAC_LLE: extract (0.036, 0.005, 0.959), raffinate (0.129, 0.867, 0.004) -> at most
    0.019 in mass fraction. Extract mass 31.0 kg; the three component balances of the figure readings give 25.0 (acid, the
    one the book uses, ill-conditioned), 30.4 (ether) and 31.0 kg (water); the book's graphical lever rule gives 27.6 kg."""
    r = T.liquid_liquid_extraction("decanter", {"water": 60.0, "acetic acid": 10.0, "diisopropyl ether": 30.0}, "kg_s",
                                   T_K=293.15, P_Pa=P, model="UNIFAC_LLE")
    assert r["status"] == "ok" and r["evidence"]["verified"]
    ext, raf = r["light_phase"], r["heavy_phase"]
    book_ext = {"acetic acid": 0.04, "water": 0.02, "diisopropyl ether": 0.94}
    book_raf = {"acetic acid": 0.12, "water": 0.86, "diisopropyl ether": 0.02}
    assert max(abs(ext["mass_fractions"][c] - v) for c, v in book_ext.items()) < 0.02
    assert max(abs(raf["mass_fractions"][c] - v) for c, v in book_raf.items()) < 0.02
    assert 25.0 <= ext["total_kg_s"] <= 31.5
    assert ext["total_kg_s"] + raf["total_kg_s"] == pytest.approx(100.0, rel=1e-9)
    assert ext["density_kg_m3"] < raf["density_kg_m3"]
    json.dumps(r)


def test_countercurrent_mccabe_20_3_and_identities():
    """[MC] 100 of feed (40 % acetone in water) against 100 of MIK at 25 °C: 99 % recovery of acetone needs 3.4 ideal stages
    (graphical), extract with about 0.272 acetone. Measured with UNIFAC_LLE: 96.9 % with 2 stages and 99.35 % with 3, so the
    model needs 2.7 stages where the book reads 3.4 (it predicts a somewhat more favourable distribution)."""
    feed, solv = {"acetone": 40.0, "water": 60.0}, {"methyl isobutyl ketone": 100.0}
    res = {n: T.liquid_liquid_extraction("countercurrent", feed, "kg_s", solv, T_K=298.15, P_Pa=P, model="UNIFAC_LLE",
                                         n_stages=n, tolerance=1e-8) for n in (1, 2, 3)}
    rec = [res[n]["recovery_to_extract"]["acetone"] for n in (1, 2, 3)]
    assert all(res[n]["evidence"]["verified"] for n in res)
    assert rec == sorted(rec)
    assert rec[1] == pytest.approx(0.9686, abs=0.004) and rec[2] == pytest.approx(0.9935, abs=0.003)
    assert rec[1] < 0.99 < rec[2]
    assert res[3]["extract"]["mass_fractions"]["acetone"] == pytest.approx(0.272, abs=0.01)
    for n, r in res.items():
        ev = r["evidence"]
        assert ev["overall_balance_closure_rel"] < 1e-6 and ev["stage_balance_residual_rel"] < 1e-6
        assert len(r["stages"]) == n and ev["all_stage_flashes_verified"]
        assert r["raffinate"]["total_kg_s"] + r["extract"]["total_kg_s"] == pytest.approx(200.0, rel=1e-6)
    # one countercurrent stage is the decanter of the mixed streams
    dec = T.liquid_liquid_extraction("decanter", feed, "kg_s", solv, T_K=298.15, P_Pa=P, model="UNIFAC_LLE")
    assert dec["roles"] == {"extract": "light", "raffinate": "heavy", "solvent_key": "methyl isobutyl ketone"}
    assert dec["light_phase"]["flows_kg_s"]["acetone"] == pytest.approx(res[1]["extract"]["flows_kg_s"]["acetone"], rel=1e-6)
    json.dumps(res[3])


# [LEE] ester-rich phase (methanol, glycerol) and glycerol-rich phase (methanol, glycerol), mole fractions; rest methyl oleate
LEE_TIE_LINES = [
    (298.2, 0.2154, 0.0, 0.2513, 0.7487), (298.2, 0.3185, 0.0, 0.4369, 0.5631), (298.2, 0.4548, 0.0017, 0.6339, 0.3661),
    (298.2, 0.5122, 0.0034, 0.7248, 0.2743), (298.2, 0.5626, 0.0065, 0.7927, 0.2063), (298.2, 0.5818, 0.0114, 0.8243, 0.1738),
    (308.2, 0.2438, 0.0, 0.2616, 0.7384), (308.2, 0.3163, 0.0042, 0.445, 0.555), (308.2, 0.4546, 0.004, 0.6265, 0.3735),
    (308.2, 0.5377, 0.0058, 0.7235, 0.276), (308.2, 0.6029, 0.0085, 0.7954, 0.2033), (308.2, 0.6693, 0.0096, 0.8265, 0.1717),
    (318.2, 0.2305, 0.0018, 0.2481, 0.7519), (318.2, 0.3082, 0.0058, 0.4101, 0.5899), (318.2, 0.4648, 0.0074, 0.6, 0.4),
    (318.2, 0.5423, 0.0092, 0.716, 0.2836), (318.2, 0.6299, 0.0084, 0.7815, 0.2172), (318.2, 0.6787, 0.0076, 0.8042, 0.1939)]


def test_biodiesel_glycerol_methanol_tie_lines_unifac_is_poor():
    """Honest record of a BAD agreement. For every published tie-line the feed is the midpoint of the two measured phases.
    UNIFAC (the only predictive model of the engine that covers the three compounds) keeps the two phases but puts far too
    little methanol in the ester phase: K_methanol(ester/glycerol, mole fractions) is 0.24-0.55 of the measured one (mean
    0.43), the ester-phase methanol mole fraction is off by 0.22 on average (max 0.30) and the glycerol-phase one by 0.08.
    UNIFAC_LLE cannot be used (no methanol group in its table) and UNIFAC_DORTMUND has no groups for glycerol."""
    names = ("methyl oleate", "glycerol", "methanol")
    ratios, d_ester, d_gly = [], [], []
    for t, x1m, x1g, x2m, x2g in LEE_TIE_LINES:
        ester = {"methanol": x1m, "glycerol": x1g, "methyl oleate": 1 - x1m - x1g}
        gly = {"methanol": x2m, "glycerol": x2g, "methyl oleate": max(1 - x2m - x2g, 0.0)}
        r = T.liquid_liquid_extraction("decanter", {c: 0.5 * (ester[c] + gly[c]) for c in names}, "mol_s", T_K=t, P_Pa=101e3,
                                       model="UNIFAC")
        assert r["status"] == "ok" and r["evidence"]["verified"]
        light, heavy = r["light_phase"]["mole_fractions"], r["heavy_phase"]["mole_fractions"]
        assert light["methyl oleate"] > 0.5 and heavy["methyl oleate"] < 0.01          # the phases are the right ones
        ratios.append((light["methanol"] / heavy["methanol"]) / (x1m / x2m))
        d_ester.append(abs(light["methanol"] - x1m))
        d_gly.append(abs(heavy["methanol"] - x2m))
    assert 0.22 < min(ratios) and max(ratios) < 0.57 and sum(ratios) / len(ratios) == pytest.approx(0.43, abs=0.02)
    assert sum(d_ester) / len(d_ester) == pytest.approx(0.223, abs=0.01) and max(d_ester) < 0.31
    assert sum(d_gly) / len(d_gly) == pytest.approx(0.076, abs=0.01)
    feed = {"methyl oleate": 0.3, "glycerol": 0.3, "methanol": 0.4}
    with pytest.raises(ValueError, match="CH3OH"):
        T.liquid_liquid_extraction("decanter", feed, "mol_s", T_K=298.2, P_Pa=101e3, model="UNIFAC_LLE")
    with pytest.raises(ValueError, match="glycerol"):
        T.liquid_liquid_extraction("decanter", feed, "mol_s", T_K=298.2, P_Pa=101e3, model="UNIFAC_DORTMUND")


def test_extraction_no_split_is_reported_and_refusals():
    r = T.liquid_liquid_extraction("decanter", {"water": 1.0, "ethanol": 1.0}, "mol_s", T_K=298.15, P_Pa=P, model="UNIFAC")
    assert r["status"] == "single_liquid" and not r["evidence"]["verified"] and r["light_phase"] is None
    assert any("NO se separa" in w for w in r["warnings"])
    json.dumps(r)
    feed = {"water": 1.0, "A": 0.1}
    with pytest.raises(ValueError, match="mode"):
        LE.liquid_liquid_extraction(None, feed, "kg_s")
    with pytest.raises(ValueError, match="flow_basis"):
        LE.liquid_liquid_extraction("decanter", feed, None)
    with pytest.raises(ValueError, match="exactamente uno"):
        LE.liquid_liquid_extraction("decanter", feed, "kg_s")
    with pytest.raises(ValueError, match="T_K"):
        LE.liquid_liquid_extraction("decanter", feed, "kg_s", model="UNIFAC", P_Pa=P)
    with pytest.raises(ValueError, match="solvent"):
        LE.liquid_liquid_extraction("countercurrent", feed, "kg_s", model="UNIFAC", T_K=300.0, P_Pa=P, n_stages=2)
    with pytest.raises(ValueError, match="n_stages"):
        LE.liquid_liquid_extraction("countercurrent", feed, "kg_s", {"S": 1.0}, distribution_coefficients={"A": 1.0},
                                    carrier="water", extraction_solvent="S")
    with pytest.raises(ValueError, match="carrier"):
        LE.liquid_liquid_extraction("decanter", feed, "kg_s", {"S": 1.0}, distribution_coefficients={"A": 1.0})
    with pytest.raises(ValueError, match="coeficiente de distribución"):
        LE.liquid_liquid_extraction("decanter", feed, "kg_s", {"S": 1.0}, distribution_coefficients={}, carrier="water",
                                    extraction_solvent="S")
    with pytest.raises(ValueError, match="parámetros binarios"):
        LE.liquid_liquid_extraction("decanter", {"water": 1.0, "acetic acid": 0.1, "diisopropyl ether": 1.0}, "mol_s",
                                    model="NRTL", T_K=300.0, P_Pa=P)


# ---------------------------------------------------------------------------------------------------------------------
# gravity settler
# ---------------------------------------------------------------------------------------------------------------------
def _sizing(**over):
    base = {"orientation": "horizontal", "droplet_diameter_m": 150e-6, "residence_time_s": 1.0, "length_to_diameter": 4.0,
            "fill_fraction": 1.0, "interface_height_fraction": 0.5, "continuous_phase": "heavy",
            "continuous_viscosity_Pa_s": 1e-3, "dispersed_viscosity_Pa_s": 3e-3}
    return base | over


def test_decanter_coulson_richardson_examples():
    """[DEC] oil (900 kg/m3, 3 mPa·s) dispersed in water (1000 kg/m3, 1 mPa·s), 150 µm drops. The lecture rounds the Stokes
    velocity to 1.2e-3 m/s (exact 1.2258e-3), which is the whole deviation: velocity +2.2 %, areas -2.1 / -2.3 %, diameters
    -1.1 / +0.1 %, entrained drop +1.2 %."""
    h = LE.size_gravity_settler(5000 / 900 / 3600, 25000 / 1000 / 3600, 900.0, 1000.0, _sizing())          # example 2
    assert h["governed_by"] == "settling" and h["stokes_regime"]
    assert h["stokes_settling_velocity_m_s"] == pytest.approx(1.2e-3, rel=0.025)
    assert h["interface_area_required_m2"] == pytest.approx(5.787, rel=0.025)
    assert h["diameter_m"] == pytest.approx(1.2028, rel=0.015) and h["length_m"] == pytest.approx(4.8112, rel=0.015)
    assert h["entrainment_check"]["dispersed_source_phase_velocity_m_s"] == pytest.approx(2.6667e-4, rel=0.025)
    assert h["entrainment_check"]["largest_continuous_droplet_carried_m"] == pytest.approx(1.21e-4, rel=0.02)
    assert h["smallest_droplet_separated_m"] == pytest.approx(150e-6, rel=1e-9) and h["droplets_settle"]
    v = LE.size_gravity_settler(1000 / 900 / 3600, 5000 / 1000 / 3600, 900.0, 1000.0,
                                _sizing(orientation="vertical", length_to_diameter=2.0))                   # example 1
    assert v["interface_area_required_m2"] == pytest.approx(1.16, rel=0.03)
    assert v["diameter_m"] == pytest.approx(1.2, rel=0.02) and v["length_m"] == pytest.approx(2.4, rel=0.02)
    assert v["entrainment_check"]["largest_continuous_droplet_carried_m"] == pytest.approx(1.2e-4, rel=0.03)


def test_decanter_identities_and_mccabe_holdup():
    q_l, q_h = 2e-3, 3e-3
    a = LE.size_gravity_settler(q_l, q_h, 850.0, 1000.0, _sizing(residence_time_s=600.0, droplet_diameter_m=1e-3,
                                                                 continuous_viscosity_Pa_s=5e-3))
    assert a["governed_by"] == "residence_time" and a["liquid_residence_time_s"] == pytest.approx(600.0, rel=1e-12)
    assert a["vessel_volume_m3"] == pytest.approx((q_l + q_h) * 600.0, rel=1e-12)
    assert a["interface_area_m2"] == pytest.approx(a["diameter_m"] * a["length_m"], rel=1e-12)       # interface on the axis
    assert a["heavy_layer"]["volume_m3"] + a["light_layer"]["volume_m3"] == pytest.approx(a["vessel_volume_m3"], rel=1e-12)
    assert a["stokes_settling_velocity_m_s"] == pytest.approx(9.80665 * 1e-6 * 150.0 / (18 * 5e-3), rel=1e-12)
    b = LE.size_gravity_settler(q_l, q_h, 850.0, 1000.0, _sizing(residence_time_s=600.0, droplet_diameter_m=2e-3,
                                                                 continuous_viscosity_Pa_s=5e-3))
    assert b["stokes_settling_velocity_m_s"] == pytest.approx(4 * a["stokes_settling_velocity_m_s"], rel=1e-12)   # ~ d²
    part = LE.size_gravity_settler(q_l, q_h, 850.0, 1000.0, _sizing(residence_time_s=600.0, fill_fraction=0.8,
                                                                    interface_height_fraction=0.3))
    assert part["heavy_layer"]["volume_m3"] + part["light_layer"]["volume_m3"] == pytest.approx(
        0.8 * part["vessel_volume_m3"], rel=1e-9)
    # [MC] example 2.2: 2 x 43.8 gal/min held 23 min in a vessel 95 % full -> 2,014.8 / 0.95 = 2,120.8 gal
    q = 43.8 * GAL / 60.0
    m = LE.size_gravity_settler(q, q, 865.0, 1153.0, _sizing(residence_time_s=23 * 60.0, fill_fraction=0.95,
                                                             length_to_diameter=5.0, droplet_diameter_m=1e-3,
                                                             continuous_phase="light", continuous_viscosity_Pa_s=1.1e-3))
    assert m["governed_by"] == "residence_time"
    assert m["vessel_volume_m3"] / GAL == pytest.approx(2 * 43.8 * 23 / 0.95, rel=1e-9)
    big = LE.size_gravity_settler(q_l, q_h, 850.0, 1000.0, _sizing(droplet_diameter_m=5e-3))
    assert not big["stokes_regime"] and any("Stokes" in w for w in big["warnings"])
    for key in ("orientation", "droplet_diameter_m", "residence_time_s", "length_to_diameter", "fill_fraction",
                "interface_height_fraction", "continuous_phase", "continuous_viscosity_Pa_s"):
        bad = _sizing()
        del bad[key]
        with pytest.raises(ValueError, match="Falta"):
            LE.size_gravity_settler(q_l, q_h, 850.0, 1000.0, bad)
    with pytest.raises(ValueError, match="densidad"):
        LE.size_gravity_settler(q_l, q_h, 1000.0, 1000.0, _sizing())


# ---------------------------------------------------------------------------------------------------------------------
# storage tank
# ---------------------------------------------------------------------------------------------------------------------
# [NETL] table «Storage Tanks»: capacity gal -> purchased cost, 1st quarter 1998 USD, carbon steel (A515)
NETL_TANKS = {"floating_roof": {50000: 118000, 75000: 128200, 100000: 143200, 250000: 197700, 500000: 267800, 750000: 335700,
                                1000000: 396600},
              "cone_roof": {50000: 42400, 75000: 48900, 100000: 59200, 250000: 98600, 500000: 157800, 750000: 214800,
                            1000000: 266100}}


def _tank(gallons, **kw):
    return T.storage_tank(flow_kg_s=1000.0 * gallons * GAL / 86400.0, density_kg_m3=1000.0, fill_fraction=1.0,
                          inventory_days=1.0, **kw)


def test_tank_cost_against_netl_table():
    """BioSTEAM's Seider correlations at the table's CEPCI (389.5). Measured: cone roof +18.0 % (75,000 gal) to -18.1 %
    (1,000,000 gal); floating roof -33.3 % (50,000 gal) to -5.6 % (500,000 gal), i.e. always below the table."""
    dev = {}
    for kind, table in NETL_TANKS.items():
        for gallons, ref in table.items():
            r = _tank(gallons, tank_type=kind, material="carbon_steel", cepci=389.5)
            assert r["evidence"]["verified"] and r["number_of_tanks"] == 1 and r["cost"]["within_correlation_range"]
            dev[kind, gallons] = r["cost"]["purchase_cost_usd"] / ref - 1.0
    cone = [v for (k, _), v in dev.items() if k == "cone_roof"]
    floating = [v for (k, _), v in dev.items() if k == "floating_roof"]
    assert max(cone) == pytest.approx(0.180, abs=0.005) and min(cone) == pytest.approx(-0.181, abs=0.005)
    assert max(floating) == pytest.approx(-0.056, abs=0.005) and min(floating) == pytest.approx(-0.333, abs=0.005)


def test_tank_cost_and_geometry_against_humbird_2011():
    """[HUM] T-710: 65,000 gal floating roof, carbon steel, $200,000 (2009); T-701: 2 x 750,000 gal floating roof, A285C,
    $1,340,000 (2009). The report scales 2009 -> 2007 by 198,927/197,602 (its own columns) and BioSTEAM carries 525.4 as the
    2007 index. Measured: -39.8 % and -37.9 % (the correlation is well below these vendor quotes; the slope between the
    two quotes, 0.49, agrees with the correlation's 0.507). T-703: 12,600 gal as 12 ft dia x 15 ft H."""
    to_2007 = 198927.0 / 197602.0
    one = _tank(65000, tank_type="floating_roof", material="carbon_steel", cepci=525.4)
    assert one["cost"]["purchase_cost_usd"] / (200000 * to_2007) - 1 == pytest.approx(-0.398, abs=0.005)
    two = _tank(1500000, tank_type="floating_roof", material="carbon_steel", cepci=525.4)
    assert two["number_of_tanks"] == 2 and two["volume_per_tank_gal"] == pytest.approx(750000, rel=1e-9)
    assert two["cost"]["purchase_cost_usd"] / (1340000 * to_2007) - 1 == pytest.approx(-0.379, abs=0.005)
    assert any("2 tanques" in w for w in two["warnings"])
    assert math.log(1340000 / 2 / 200000) / math.log(750000 / 65000) == pytest.approx(0.494, abs=0.002)
    g = _tank(12600, aspect_ratio_H_D=15.0 / 12.0)
    assert g["cost"] is None and g["equipment"] == []
    assert g["geometry_per_tank"]["diameter_m"] / FT == pytest.approx(12.0, rel=0.005)       # measured -0.24 %
    assert g["geometry_per_tank"]["height_m"] / FT == pytest.approx(15.0, rel=0.005)


def test_tank_identities_ranges_and_refusals():
    a = T.storage_tank(flow_kg_s=2.0, density_kg_m3=800.0, fill_fraction=0.8, holdup_time_s=7200.0)
    assert a["working_volume_m3"] == pytest.approx(2.0 / 800.0 * 7200.0) and a["total_volume_m3"] == pytest.approx(22.5)
    b = T.storage_tank(flow_kg_s=2.0, density_kg_m3=800.0, fill_fraction=0.8, inventory_days=7200.0 / 86400.0,
                       aspect_ratio_H_D=2.0)
    assert b["total_volume_m3"] == pytest.approx(a["total_volume_m3"], rel=1e-12)
    geo = b["geometry_per_tank"]
    assert math.pi * geo["diameter_m"] ** 2 * geo["height_m"] / 4 == pytest.approx(22.5, rel=1e-12)
    assert geo["liquid_height_m"] == pytest.approx(0.8 * geo["height_m"])
    cs = _tank(100000, tank_type="cone_roof", material="carbon_steel")
    ss = _tank(100000, tank_type="cone_roof", material="stainless_steel", aspect_ratio_H_D=1.25)
    assert ss["cost"]["purchase_cost_usd"] == pytest.approx(2.0 * cs["cost"]["purchase_cost_usd"])     # BioSTEAM's F_M
    assert ss["cost"]["base_purchase_cost_usd"] == pytest.approx(cs["cost"]["purchase_cost_usd"])
    assert cs["cost"]["basis"] == "CEPCI 797.9 (2023, tabla de IDAES)" and cs["cost"]["F_BM"] == 2.3
    assert cs["equipment"][0]["type"] == "storage_tank"
    cross = ss["cross_check_pressure_vessel"]
    assert cross["available"] and cross["ratio_to_tank_correlation"] > 1.0           # a pressure vessel is the upper bound
    # 265·V^0.513 with V in gallons at CEPCI 567 is what BioSTEAM states for the cone roof; scaled to 2023
    assert cs["cost"]["purchase_cost_usd"] == pytest.approx(265 * 100000 ** 0.513 * 797.9 / 567, rel=1e-9)
    small = T.storage_tank(flow_kg_s=1.0, density_kg_m3=1000.0, fill_fraction=0.9, holdup_time_s=3600.0, tank_type="cone_roof",
                           material="carbon_steel")
    assert not small["cost"]["within_correlation_range"]
    assert any("EXTRAPOLACIÓN" in w for w in small["warnings"]) and any("BioSTEAM" in w for w in small["warnings"])
    mix = T.storage_tank(flow_kg_s=1.0, density_kg_m3=1000.0, fill_fraction=0.9, holdup_time_s=3600.0, tank_type="mix_tank",
                         material="stainless_steel")
    assert mix["cost"]["within_correlation_range"] and mix["warnings"] == []
    assert mix["cost"]["purchase_cost_usd"] == pytest.approx(12080 * 4.0 ** 0.525 * 797.9 / 525.4, rel=1e-9)
    three = _tank(1500000, tank_type="cone_roof", material="carbon_steel", number_of_tanks=3)
    assert three["number_of_tanks"] == 3 and three["volume_per_tank_gal"] == pytest.approx(500000, rel=1e-9)
    with pytest.raises(ValueError, match="al menos 2"):
        _tank(1500000, tank_type="cone_roof", material="carbon_steel", number_of_tanks=1)
    with pytest.raises(ValueError, match="cepci"):
        _tank(100000, tank_type="cone_roof", material="carbon_steel", year=2026)
    for missing in ("flow_kg_s", "density_kg_m3", "fill_fraction"):
        args = dict(flow_kg_s=1.0, density_kg_m3=1000.0, fill_fraction=0.9, holdup_time_s=10.0)
        args[missing] = None
        with pytest.raises(ValueError, match="Falta"):
            LE.storage_tank(**args)
    with pytest.raises(ValueError, match="exactamente uno"):
        LE.storage_tank(flow_kg_s=1.0, density_kg_m3=1000.0, fill_fraction=0.9)
    with pytest.raises(ValueError, match="a la vez"):
        LE.storage_tank(flow_kg_s=1.0, density_kg_m3=1000.0, fill_fraction=0.9, holdup_time_s=10.0, tank_type="cone_roof")
    for r in (a, b, cs, ss, small, mix, three):
        json.dumps(r)


# ---------------------------------------------------------------------------------------------------------------------
# sizing + cost through the tool, capital_investment shape, BioSTEAM hygiene, wrappers
# ---------------------------------------------------------------------------------------------------------------------
def test_extraction_sizing_cost_and_capital_investment_shape():
    from packages.costing import economics as EC
    from packages.costing import sslw
    sizing = {"orientation": "horizontal", "droplet_diameter_m": 150e-6, "residence_time_s": 1800.0, "length_to_diameter": 4.0,
              "fill_fraction": 0.95, "interface_height_fraction": 0.3, "continuous_phase": "light",
              "continuous_viscosity_Pa_s": 3e-3}
    feed = {"methyl oleate": 0.9, "glycerol": 0.1, "methanol": 0.1}
    r = T.liquid_liquid_extraction("decanter", feed, "kg_s", T_K=333.15, P_Pa=P, model="UNIFAC", sizing=sizing,
                                   costing={"equipment": "gravity_decanter", "material": "StainlessSteel304"})
    assert r["status"] == "ok" and r["evidence"]["verified"] and r["sizing"]["density_source"].startswith("estimadas")
    s = r["sizing"]
    q = r["light_phase"]["volumetric_flow_m3_s"] + r["heavy_phase"]["volumetric_flow_m3_s"]
    assert s["governed_by"] == "residence_time" and s["vessel_volume_m3"] == pytest.approx(q * 1800.0 / 0.95, rel=1e-9)
    ref = sslw.cost_vessel(s["diameter_m"], s["length_m"], vertical=False, material="StainlessSteel304", year=2023)
    assert r["cost"]["purchase_cost_usd"] == pytest.approx(ref["purchase_cost_usd"], rel=1e-12)
    assert r["equipment"][0]["type"] == "vessel_horizontal"
    assert r["equipment"][0]["base_purchase_cost_usd"] < r["equipment"][0]["purchase_cost_usd"]
    with pytest.raises(ValueError, match="sizing"):
        T.liquid_liquid_extraction("decanter", feed, "kg_s", T_K=333.15, P_Pa=P, model="UNIFAC",
                                   costing={"equipment": "gravity_decanter", "material": "CarbonSteel"})
    c = T.liquid_liquid_extraction(
        "countercurrent", {"water": 99.0 / 3600, "nicotine": 1.0 / 3600}, "kg_s", {"kerosene": 199.9 / 3600},
        distribution_coefficients={"nicotine": 0.9}, carrier="water", extraction_solvent="kerosene", n_stages=3,
        sizing=_sizing(residence_time_s=600.0, raffinate_density_kg_m3=1000.0, extract_density_kg_m3=800.0),
        costing={"equipment": "centrifuge"})
    q_h = 100.0 / 1000.0 + c["extract"]["total"] * 3600 / 800.0   # m3/h: the larger of inlet and outlet of each phase
    assert c["cost"]["flow_m3_h"] == pytest.approx(q_h, rel=1e-9) and c["cost"]["number_of_units"] == 3
    # consistency only: the correlation BioSTEAM states (28,100·Q^0.574 at CEPCI 525.4), three units, scaled to 2023
    assert c["cost"]["purchase_cost_usd"] == pytest.approx(3 * 28100 * c["cost"]["flow_m3_h"] ** 0.574 * 797.9 / 525.4, rel=1e-9)
    assert c["equipment"][0]["F_BM"] == 2.03 and c["sizing"]["number_of_units"] == 3
    with pytest.raises(ValueError, match="raffinate_density_kg_m3"):
        T.liquid_liquid_extraction("decanter", {"water": 1.0, "A": 0.1}, "kg_s", {"S": 1.0},
                                   distribution_coefficients={"A": 1.0}, carrier="water", extraction_solvent="S",
                                   sizing=_sizing())
    tank = _tank(100000, tank_type="cone_roof", material="stainless_steel")
    evap = T.multi_effect_evaporator(**GK851, costing={"type": "horizontal_tube"})
    equipment = r["equipment"] + c["equipment"] + tank["equipment"] + evap["equipment"]
    factors = {k: 0.0 for k in EC.CAPITAL_FACTORS}
    cap = EC.capital_investment(equipment, factors)
    assert "bases distintas" not in json.dumps(cap, default=float, ensure_ascii=False)       # one single CEPCI basis
    assert [row["F_BM"] for row in cap["equipment"]] == [3.05, 2.03, 2.3, 2.45, 2.45, 2.45]
    for x in (r, c, tank, evap):
        json.dumps(x)


def test_biosteam_and_thermo_globals_are_left_as_found():
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    with LE._biosteam_session() as (bst, _, _):
        pass
    import thermosteam as tmo
    agent = bst.HeatUtility.heating_agents[0]
    original = (bst.CE, bst.PowerUtility.price, agent.regeneration_price, TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR,
                MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    try:
        for flag in (False, True):
            bst.CE, bst.PowerUtility.price, agent.regeneration_price = 612.3, 0.123, 0.4567
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flag
            flowsheet, thermo = bst.main_flowsheet.get_flowsheet(), getattr(tmo.settings, "_thermo", None)
            a = _tank(100000, tank_type="cone_roof", material="carbon_steel")
            T.multi_effect_evaporator(**GK851, costing={"type": "vertical_tube"})
            T.liquid_liquid_extraction("decanter", {"water": 1.0, "A": 0.1}, "kg_s", {"S": 1.0},
                                       distribution_coefficients={"A": 1.0}, carrier="water", extraction_solvent="S",
                                       sizing=_sizing(raffinate_density_kg_m3=1000.0, extract_density_kg_m3=800.0),
                                       costing={"equipment": "centrifuge"})
            assert (bst.CE, bst.PowerUtility.price, agent.regeneration_price) == (612.3, 0.123, 0.4567)
            assert TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR is flag
            assert MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR is flag
            assert bst.main_flowsheet.get_flowsheet() is flowsheet and getattr(tmo.settings, "_thermo", None) is thermo
            # the cost does not depend on the library's CE of the moment: it is scaled out
            assert a["cost"]["purchase_cost_usd"] == pytest.approx(265 * 100000 ** 0.513 * 797.9 / 567, rel=1e-9)
        with LE._biosteam_session() as (b2, _, _):                 # whatever is changed inside a session is undone
            b2.CE, b2.PowerUtility.price = 1.0, 9.9
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = False
        assert (bst.CE, bst.PowerUtility.price) == (612.3, 0.123)
        assert TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR is True
    finally:
        (bst.CE, bst.PowerUtility.price, agent.regeneration_price, TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR,
         MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR) = original


def test_tools_and_agent_wrappers_are_aligned():
    from packages.agent import wrappers_liquid_equipment as W
    assert [f.__name__ for f in T.LIQUID_EQUIPMENT_TOOLS] == [w.name for w in W.AGENT_WRAPPERS]
    assert len(T.LIQUID_EQUIPMENT_TOOLS) == 3 and all(f.__doc__ and len(f.__doc__) > 400 for f in T.LIQUID_EQUIPMENT_TOOLS)
    out = json.loads(W.storage_tank.call({"flow_kg_s": 1.0, "density_kg_m3": 900.0, "fill_fraction": 0.9,
                                          "inventory_days": 3.0, "aspect_ratio_H_D": 1.2}))
    assert out["evidence"]["verified"] and out["total_volume_m3"] == pytest.approx(3 * 86400 / 900.0 / 0.9)
    out = json.loads(W.liquid_liquid_extraction.call({
        "mode": "countercurrent", "feed": {"water": 1.0, "A": 0.01}, "flow_basis": "kg_s", "solvent": {"S": 1.0},
        "distribution_coefficients": {"A": 2.0}, "carrier": "water", "extraction_solvent": "S", "n_stages": 3}))
    assert out["evidence"]["verified"] and out["solutes"]["A"]["fraction_extracted"] == pytest.approx(1 - 1 / 15, rel=1e-9)
    out = json.loads(W.multi_effect_evaporator.call(dict(GK851)))
    assert out["evidence"]["verified"] and out["steam_economy"] == pytest.approx(2.026, abs=0.002)
