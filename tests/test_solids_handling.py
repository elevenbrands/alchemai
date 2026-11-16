"""Block M26_E3: size reduction, solids conveying, solid-liquid separation and oil extraction.

Reference cases (all read on 2026-11-16 in open sources; deviations are the ones MEASURED with this code, never tuned):
  [GK14] Geankoplis, Transport Processes and Unit Operations (3rd ed.), examples 14.2-1 to 14.2-4 (constant-pressure
         filtration, rotary drum), 14.4-2 (tubular centrifuge) and 14.5-1 (Bond), inputs and computed outputs as reproduced
         by the FOSSEE Python Textbook Companion (IIT Bombay):
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter14.ipynb
  [GK12] Same companion, chapter 12, example 12.10-1 (countercurrent leaching of oil from meal, variable underflow):
         https://tbc-python.fossee.in/convert-notebook/TRANSPORT_PROCESSES_AND_UNIT_OPERATIONS/GeankoplisChapter12.ipynb
  [MC28] McCabe, Smith & Harriott example 28.1 (Bond, limestone, Wi = 12.74, 100 ton/h, 2 in -> 1/8 in: 169.6 kW), as worked
         in ChemEnggCalc «Bond's crushing law and work index»: https://chemenggcalc.com/?p=366
  [KWS]  KWS Manufacturing, «Bucket Elevator Horsepower Calculation», worked example 100,000 lb/h of sand lifted 105 ft:
         6.10 HP (friction factor 1.15) and 6.01 HP (tail pulley 2 ft, Leq = 7):
         https://www.kwsmfg.com/engineering-guides/bucket-elevator/horsepower-calculation/
  [NETL] Loh, Lyons & White, «Process Equipment Cost Estimation», DOE/NETL-2002/1169, table «Filter - Drum Filter»
         (purchased cost, 1st quarter 1998 dollars) and Table 11 (CEPCI 1998 = 389.5): https://www.osti.gov/servlets/purl/797810
  Bond's law form and the short-ton basis of the work index: Holdich, Fundamentals of Particle Technology, eq. 11.5,
         https://www.particles.org.uk/particle_technology_book/chapter_11.pdf
Not found in an open source: Seider's Table 16.32 (cost of conveyors, crushers, bucket elevators) and a worked example of
belt/screw conveyor power with Seider's correlation; those parts are consistency-only (see each test).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from packages.bioprocess import solids_handling as SH
from packages.dwsim_client import tools_solids_handling as T

FT, LB, HP = 0.3048, 0.45359237, 745.7
SHORT_TON_H = 2000.0 * LB / 3600.0          # kg/s in one short ton per hour
H = 3600.0


# ------------------------------------------------------------------------------------------------------------- size reduction
def test_bond_mccabe_28_1():
    """[MC28] 100 ton/h limestone, 80 % passing 2 in -> 1/8 in, Wi = 12.74 kWh/short ton: 169.6 kW. Measured -0.02 %."""
    r = T.size_reduction(100 * SHORT_TON_H, "bond", 2 * 0.0254, 0.0254 / 8, work_index_kWh_per_ton=12.74, ton="short")
    assert r["power_W"] / 1e3 == pytest.approx(169.6, rel=5e-4)
    assert r["evidence"]["verified"]


def test_bond_geankoplis_14_5_1():
    """[GK14] 10 ton/h iron ore, 3 in -> 1/8 in, Ei = 12.68: 24.1 hp. Measured -0.34 % (the book's constant 1.46 is rounded)."""
    r = T.size_reduction(10 * SHORT_TON_H, "bond", 3 * 0.0254, 0.0254 / 8, work_index_kWh_per_ton=12.68, ton="short")
    assert r["power_W"] / HP == pytest.approx(24.1, rel=5e-3)


def test_size_reduction_identities():
    bond = lambda p, **k: T.size_reduction(2.0, "bond", 0.05, p, work_index_kWh_per_ton=13.0,  # noqa: E731
                                           ton=k.get("ton", "short"))["power_W"]
    assert bond(1e-3) > bond(5e-3) > bond(2e-2)                              # finer product costs more
    assert bond(1e-3, ton="short") / bond(1e-3, ton="metric") == pytest.approx(1000.0 / (2000.0 * LB))
    # Bond is additive along a path: 50 mm -> 5 mm -> 0.5 mm equals 50 mm -> 0.5 mm
    def e(f, p):
        return T.size_reduction(1.0, "bond", f, p, work_index_kWh_per_ton=13.0, ton="short")["specific_energy_J_kg"]
    assert e(0.05, 5e-3) + e(5e-3, 5e-4) == pytest.approx(e(0.05, 5e-4), rel=1e-12)
    # Kick: same energy for the same reduction ratio; Rittinger: proportional to new surface (1/P - 1/F)
    k = lambda f, p: T.size_reduction(1.0, "kick", f, p, kick_constant_J_kg=5e3)["specific_energy_J_kg"]  # noqa: E731
    assert k(0.02, 0.01) == pytest.approx(k(0.01, 0.005), rel=1e-12) == pytest.approx(5e3 * math.log(2.0))
    def rt(f, p):
        return T.size_reduction(1.0, "rittinger", f, p, rittinger_constant_J_m_kg=2.0)["specific_energy_J_kg"]
    assert rt(1e-3, 2.5e-4) == pytest.approx(3.0 * rt(1e-3, 5e-4), rel=1e-12)
    # calibration on the user's test reproduces the test and scales by the law
    test = {"feed_size_m": 0.02, "product_size_m": 0.005, "specific_energy_J_kg": 9e3}
    assert T.size_reduction(1.0, "kick", 0.02, 0.005, reference_test=test)["specific_energy_J_kg"] == pytest.approx(9e3)
    assert T.size_reduction(1.0, "rittinger", 0.02, 0.0025, reference_test=test)["specific_energy_J_kg"] == pytest.approx(
        9e3 * (1 / 0.0025 - 1 / 0.02) / (1 / 0.005 - 1 / 0.02))
    assert T.size_reduction(3.0, "specific_energy", specific_energy_J_kg=2e4)["power_W"] == pytest.approx(6e4)


def test_size_reduction_cost_is_biosteam_correlation_with_its_range():
    """CONSISTENCY ONLY: the cost is BioSTEAM's HammerMill correlation (4310 S^0.78 at CE 567, 2-200 short ton/h, F_BM 2.3),
    read from the installed class; no independent published cost point was found."""
    r = T.size_reduction(10 * SHORT_TON_H, "specific_energy", specific_energy_J_kg=2e4, equipment="hammer_mill", cepci=567.0)
    item = r["capital_items"][0]
    assert item["purchase_cost_usd"] == pytest.approx(4310.0 * 10 ** 0.78, rel=1e-9)
    assert item["F_BM"] == 2.3 and item["base_purchase_cost_usd"] == item["purchase_cost_usd"] and not r["warnings"]
    assert r["biosteam_reference_power_W"] == pytest.approx(61.7e3)
    # CEPCI scaling with the repo convention (IDAES table) and with the caller's index
    r23 = T.size_reduction(10 * SHORT_TON_H, "specific_energy", specific_energy_J_kg=2e4, equipment="hammer_mill", year=2023)
    assert r23["cost"]["purchase_cost_usd"] == pytest.approx(item["purchase_cost_usd"] * 797.9 / 567.0)
    assert r23["cost"]["cost_year"] == 2023 and r23["cost"]["correlation_cepci"] == 567.0
    with pytest.raises(ValueError, match="cepci"):
        T.size_reduction(1.0, "specific_energy", specific_energy_J_kg=2e4, equipment="hammer_mill", year=2026)
    # below the range: evaluated at the lower bound, with a warning; above: parallel units, with a warning
    low = T.size_reduction(0.5 * SHORT_TON_H, "specific_energy", specific_energy_J_kg=2e4, equipment="hammer_mill", cepci=567.0)
    assert low["cost"]["purchase_cost_usd"] == pytest.approx(4310.0 * 2 ** 0.78) and "mínimo" in low["warnings"][0]
    high = T.size_reduction(500 * SHORT_TON_H, "specific_energy", specific_energy_J_kg=2e4, equipment="hammer_mill",
                            cepci=567.0)
    assert high["cost"]["number_of_units"] == 3
    assert high["cost"]["purchase_cost_usd"] == pytest.approx(3 * 4310.0 * (500 / 3) ** 0.78)
    crusher = T.size_reduction(1.0, "specific_energy", specific_energy_J_kg=2e4, equipment="crusher")
    assert crusher["cost"] is None and any("cotización" in s for s in crusher["not_included"])
    shred = T.size_reduction(500e3 / H, "specific_energy", specific_energy_J_kg=2e4, equipment="shredder", cepci=567.3)
    assert shred["cost"]["purchase_cost_usd"] == pytest.approx(2.5e6)
    assert shred["biosteam_reference_power_W"] == pytest.approx(3e6)


# ------------------------------------------------------------------------------------------------------------------ conveying
def test_bucket_elevator_kws_worked_example():
    """[KWS] 100,000 lb/h lifted 105 ft: 6.10 HP with F = 1.15 and 6.01 HP with d = 2 ft, Leq = 7. Measured -0.03 % / 0.00 %
    (HP = 745.7 W)."""
    m = 100000 * LB / H
    r = T.solids_conveying("bucket_elevator", m, lift_m=105 * FT, friction_factor=1.15)
    assert r["power_W"] / HP == pytest.approx(6.10, rel=1e-3)
    assert r["lift_power_W"] == pytest.approx(m * 9.80665 * 105 * FT) and r["cost"] is None
    r = T.solids_conveying("bucket_elevator", m, lift_m=105 * FT, tail_pulley_diameter_m=2 * FT, length_equivalency_factor=7)
    assert r["power_W"] / HP == pytest.approx(6.01, rel=1e-3)
    assert any("cangilones" in s and "cotización" in s for s in r["not_included"])


def test_belt_and_screw_power_are_the_biosteam_seider_correlations():
    """CONSISTENCY ONLY for the friction term (BioSTEAM's transcription of Seider's correlations, checked against the installed
    source text); the lift term is analytical (m g H, the correlation's 0.00182 hp per lb/s ft is 1/550 rounded: +0.10 %)."""
    m, L, Hh = 10.0, 30.0, 5.0
    belt = T.solids_conveying("belt", m, length_m=L, lift_m=Hh)
    ml = m / LB
    assert belt["friction_power_W"] == pytest.approx(HP * 0.00058 * ml ** 0.82 * L / FT, rel=1e-12)
    assert belt["lift_power_W"] == pytest.approx(m * 9.80665 * Hh, rel=1.2e-3)
    assert belt["evidence"]["verified"] and belt["evidence"]["lift_term_vs_m_g_H_rel_diff"] < 1.2e-3
    assert belt["cost"] is None and any("bulk_density_kg_m3" in s for s in belt["not_included"])
    screw = T.solids_conveying("screw", m, length_m=9.0, lift_m=0.0)
    assert screw["power_W"] == pytest.approx(HP * 0.0146 * ml ** 0.85 * 9.0 / FT, rel=1e-12)
    # monotonic in flow, length and lift; a horizontal belt has no lift power
    assert T.solids_conveying("belt", m, length_m=L, lift_m=0.0)["lift_power_W"] == 0.0
    assert T.solids_conveying("belt", 2 * m, length_m=L, lift_m=Hh)["power_W"] > belt["power_W"]
    assert T.solids_conveying("belt", m, length_m=2 * L, lift_m=Hh)["power_W"] > belt["power_W"]
    assert T.solids_conveying("screw", m, length_m=9.0, lift_m=3.0)["power_W"] == pytest.approx(
        screw["power_W"] + m * 9.80665 * 3.0)
    b = SH._biosteam_data()
    assert b["belt"]["source_ok"] and b["screw"]["source_ok"]


def test_conveyor_cost_is_biosteam_correlation_with_its_range():
    """CONSISTENCY ONLY: ConveyingBelt 813 Q^0.38 (Q ft3/h, up to 2000 per belt, F_BM 1.61) and ScrewFeeder 1096 Q^0.22
    (400-1e5 ft3/h) at CE 567. Live BioSTEAM units gave the same numbers within 4e-6 (scratch check, not run here)."""
    rho = 600.0
    q_ft3_h = 5.0 / rho * H / FT ** 3
    belt = T.solids_conveying("belt", 5.0, length_m=12.0, lift_m=3.0, bulk_density_kg_m3=rho, cepci=567.0)
    assert belt["cost"]["purchase_cost_usd"] == pytest.approx(813.0 * q_ft3_h ** 0.38, rel=1e-9) and belt["cost"]["F_BM"] == 1.61
    assert any("NO de la longitud" in w for w in belt["warnings"])
    two = T.solids_conveying("belt", 15.0, length_m=12.0, lift_m=3.0, bulk_density_kg_m3=rho, cepci=567.0)
    assert two["cost"]["number_of_units"] == 2
    assert two["cost"]["purchase_cost_usd"] == pytest.approx(2 * 813.0 * (3 * q_ft3_h / 2) ** 0.38, rel=1e-9)
    screw = T.solids_conveying("screw", 5.0, length_m=9.0, lift_m=0.0, bulk_density_kg_m3=rho, cepci=567.0)
    assert screw["cost"]["purchase_cost_usd"] == pytest.approx(1096.0 * q_ft3_h ** 0.22, rel=1e-9)
    assert screw["cost"]["F_BM"] == 1.0 and any("F_BM" in w for w in screw["warnings"])
    small = T.solids_conveying("screw", 0.2, length_m=9.0, lift_m=0.0, bulk_density_kg_m3=rho, cepci=567.0)
    assert small["cost"]["purchase_cost_usd"] == pytest.approx(1096.0 * 400.0 ** 0.22) and "mínimo" in small["warnings"][0]


# ---------------------------------------------------------------------------------------------------- solid-liquid separation
GK_T = [4.4, 9.5, 16.3, 24.6, 34.7, 46.1, 59.0, 73.6, 89.4, 107.3]                                  # s      [GK14] 14.2-1
GK_V = [0.498e-3, 1.0e-3, 1.501e-3, 2.0e-3, 2.498e-3, 3.002e-3, 3.506e-3, 4.004e-3, 4.502e-3, 5.009e-3]   # m3
GK_MU, GK_CS, GK_DP, GK_A_LAB, GK_A_PRESS = 8.937e-4, 23.47, 338e3, 0.0439, 0.873 * 20


def test_filtration_geankoplis_14_2_1_and_14_2_2():
    """[GK14] 14.2-1: regression of t/V vs V gives alpha = 1.79e11 m/kg and Rm = 1.126e11 1/m (FOSSEE's least squares; the
    companion warns that the book's graphical slope and intercept differ: the constants it quotes in 14.2-3, Kp = 37.93 and
    B = 16.10, correspond to alpha = 1.863e11 and Rm = 1.063e11, see the next test). 14.2-2: with the regression values,
    3.37 m3 through 20 frames of 0.873 m2 takes 264.61 s. Measured: alpha +0.11 %, Rm +0.03 % (rounding of the printed
    values), time 0.000 %."""
    slope, intercept = np.polyfit(GK_V, np.divide(GK_T, GK_V), 1)
    alpha = 2.0 * slope * GK_A_LAB ** 2 * GK_DP / (GK_MU * GK_CS)
    Rm = intercept * GK_A_LAB * GK_DP / GK_MU
    assert alpha == pytest.approx(1.79e11, rel=3e-3) and Rm == pytest.approx(1.126e11, rel=1e-3)
    assert SH.filtration_time(3.37, GK_A_PRESS, GK_CS, alpha, Rm, GK_MU, GK_DP) == pytest.approx(264.61, rel=1e-4)
    # the model reproduces the laboratory run it was fitted to within 0.9 s
    back = [SH.filtration_time(v, GK_A_LAB, GK_CS, alpha, Rm, GK_MU, GK_DP) for v in GK_V]
    assert max(abs(a - b) for a, b in zip(back, GK_T)) < 0.95


def test_plate_and_frame_area_geankoplis_14_2_2_book_constants():
    """[GK14] 14.2-3 quotes the book's constants for the 17.46 m2 press: Kp = 37.93 s/m6, B = 16.10 s/m3 and 269.7 s for
    3.37 m3. Asked the other way round (area that filters 3.37 m3 in 269.7 s) the tool returns 17.458 m2: -0.01 %."""
    alpha = 37.93 * GK_A_PRESS ** 2 * GK_DP / (GK_MU * GK_CS)
    Rm = 16.10 * GK_A_PRESS * GK_DP / GK_MU
    assert SH.filtration_time(3.37, GK_A_PRESS, GK_CS, alpha, Rm, GK_MU, GK_DP) == pytest.approx(269.7, rel=3e-4)
    assert alpha == pytest.approx(1.863e11, rel=1e-3) and Rm == pytest.approx(1.063e11, rel=1e-3)
    q, rho = 3.37 / 269.7, 997.0
    cake_solids = GK_CS * q
    feed = cake_solids + q * rho + cake_solids                     # cake at 50 % moisture, all solids recovered
    r = T.solid_liquid_separation("plate_and_frame_filter", feed, cake_solids / feed, 1.0, 0.5, filtration={
        "specific_cake_resistance_m_kg": alpha, "medium_resistance_1_m": Rm, "filtrate_viscosity_Pa_s": GK_MU,
        "pressure_drop_Pa": GK_DP, "filtrate_density_kg_m3": rho, "filtration_time_s": 269.7, "downtime_s": 0.0})
    f = r["filtration"]
    assert f["dry_cake_per_filtrate_kg_m3"] == pytest.approx(GK_CS) and f["area_m2"] == pytest.approx(17.46, rel=3e-4)
    assert f["filtrate_per_cycle_m3"] == pytest.approx(3.37) and r["evidence"]["verified"]
    # downtime leaves the filtering time but more filtrate per cycle: larger area
    r2 = T.solid_liquid_separation("plate_and_frame_filter", feed, cake_solids / feed, 1.0, 0.5, filtration={
        "specific_cake_resistance_m_kg": alpha, "medium_resistance_1_m": Rm, "filtrate_viscosity_Pa_s": GK_MU,
        "pressure_drop_Pa": GK_DP, "filtrate_density_kg_m3": rho, "filtration_time_s": 269.7, "downtime_s": 1200.0})
    assert r2["filtration"]["area_m2"] > 2.0 * f["area_m2"]


def _drum(feed=0.778, alpha=4.37e9 * 67e3 ** 0.3, Rm=0.0, cepci=None, **kw):
    fl = {"specific_cake_resistance_m_kg": alpha, "medium_resistance_1_m": Rm, "filtrate_viscosity_Pa_s": 0.8937e-3,
          "pressure_drop_Pa": 67e3, "filtrate_density_kg_m3": 996.9, "cycle_time_s": 250.0, "submerged_fraction": 0.33}
    fl.update(kw)
    return T.solid_liquid_separation("rotary_vacuum_drum_filter", feed, 0.191, 1.0, 0.5, filtration=fl, cepci=cepci)


def test_rotary_drum_geankoplis_14_2_4():
    """[GK14] 14.2-4: 0.778 kg slurry/s, cx = 0.191, wet/dry cake m = 2.0 (50 % moisture), dP = 67 kPa, cycle 250 s, 33 %
    submerged, alpha = 4.37e9 dP^0.3, Rm neglected: A = 6.66 m2 (FOSSEE's output). Measured +0.02 %."""
    r = _drum()
    f = r["filtration"]
    assert f["dry_cake_per_filtrate_kg_m3"] == pytest.approx(996.9 * 0.191 / (1 - 2.0 * 0.191), rel=1e-12)   # cs of the example
    assert f["area_m2"] == pytest.approx(6.66, rel=1e-3)
    assert r["cake"]["wet_to_dry_mass_ratio"] == pytest.approx(2.0) and r["evidence"]["verified"]
    # identities: with Rm = 0 the area goes with sqrt(alpha) and 1/sqrt(submergence); a medium resistance adds area
    assert _drum(alpha=4 * 4.37e9 * 67e3 ** 0.3)["filtration"]["area_m2"] == pytest.approx(2 * f["area_m2"], rel=1e-12)
    assert _drum(submerged_fraction=0.33 / 4)["filtration"]["area_m2"] == pytest.approx(2 * f["area_m2"], rel=1e-12)
    assert _drum(feed=2 * 0.778)["filtration"]["area_m2"] == pytest.approx(2 * f["area_m2"], rel=1e-12)
    with_medium = _drum(Rm=1e11)
    assert with_medium["filtration"]["area_m2"] > f["area_m2"]
    assert 0.0 < with_medium["filtration"]["medium_share_of_resistance_at_end"] < 1.0


NETL_DRUM_FILTER_1998 = {100: 63400.0, 250: 87700.0, 500: 120200.0, 750: 145000.0}      # ft2 -> purchased USD   [NETL]


def test_rotary_drum_cost_against_netl_table():
    """[NETL] independent published table (ICARUS, 1Q-1998 USD) vs the BioSTEAM/Seider correlation scaled to CEPCI 389.5:
    the correlation gives 1.94-2.02 times the table over 100-750 ft2. Reported as is (not tuned): the purchase cost of a
    drum filter carries a factor-of-two uncertainty between sources; the tool says so in its warnings."""
    a1 = _drum()["filtration"]["area_m2"]
    ratios = []
    for area_ft2, usd in NETL_DRUM_FILTER_1998.items():
        r = _drum(feed=0.778 * area_ft2 * FT ** 2 / a1, cepci=389.5)
        assert r["filtration"]["area_m2"] / FT ** 2 == pytest.approx(area_ft2)
        ln = math.log(area_ft2)
        assert r["cost"]["purchase_cost_usd"] == pytest.approx(
            math.exp(11.796 - 0.1905 * ln + 0.0554 * ln * ln) * 389.5 / 567.0, rel=1e-9)
        assert r["cost"]["F_BM"] == 2.32 and r["evidence"]["verified"]
        ratios.append(r["cost"]["purchase_cost_usd"] / usd)
    assert 1.90 < min(ratios) and max(ratios) < 2.05
    assert any("NETL" in w for w in r["warnings"])
    big = _drum(feed=0.778 * 2000 * FT ** 2 / a1, cepci=567.0)
    assert big["cost"]["number_of_units"] == 3 and any("tambores" in w for w in big["warnings"])


def test_centrifuge_geankoplis_14_4_2():
    """[GK14] 14.4-2: tubular bowl r1 = 0.00716, r2 = 0.02225, b = 0.1970 m, 23 000 rpm, particles 1461 kg/m3 in a liquid of
    801 kg/m3 and 0.1 Pa s, q = 0.002832 m3/h: critical diameter 0.7468 um. Measured -0.005 %."""
    q, x = 0.002832 / H, 0.01
    feed = q / (x / 1461.0 + (1 - x) / 801.0)
    r = T.solid_liquid_separation("decanter_centrifuge", feed, x, 0.9, 0.6, sigma={
        "particle_diameter_m": 1e-6, "particle_density_kg_m3": 1461.0, "liquid_density_kg_m3": 801.0,
        "liquid_viscosity_Pa_s": 0.1,
        "bowl": {"type": "tubular", "rpm": 23000, "r_inner_m": 0.00716, "r_outer_m": 0.02225, "length_m": 0.1970}})
    s = r["sigma"]
    assert s["feed_volumetric_flow_m3_s"] == pytest.approx(q)
    assert s["bowl"]["cut_diameter_at_feed_flow_m"] * 1e6 == pytest.approx(0.7468, rel=2e-4)
    assert r["evidence"]["verified"]
    # Sigma theory identities: Q = 2 vg Sigma, required Sigma goes with 1/d^2
    vg = 1e-12 * (1461.0 - 801.0) * 9.80665 / (18 * 0.1)
    assert s["stokes_settling_velocity_1g_m_s"] == pytest.approx(vg) and s["sigma_required_m2"] == pytest.approx(q / (2 * vg))
    assert s["bowl"]["capacity_for_particle_m3_s"] == pytest.approx(2 * vg * s["bowl"]["sigma_m2"])


def test_sigma_limits_and_scaling():
    """CONSISTENCY ONLY (analytical limits): a thin liquid layer has Sigma = wall area x g-number; the disc-stack formula
    (derived, not read in a worked example) scales with n, rpm^2 and r2^3 - r1^3, and follows from the plug-flow integral."""
    rpm, r2, L = 3000.0, 0.25, 0.8
    thin = SH.sigma_bowl({"type": "solid_bowl", "rpm": rpm, "r_inner_m": r2 * (1 - 1e-6), "r_outer_m": r2, "length_m": L})
    w = 2 * math.pi * rpm / 60
    assert thin["sigma_m2"] == pytest.approx(2 * math.pi * r2 * L * w * w * r2 / 9.80665, rel=1e-5)
    d = {"type": "disc_stack", "rpm": 6000.0, "r_inner_m": 0.05, "r_outer_m": 0.15, "n_discs": 80, "half_angle_deg": 40.0}
    s0 = SH.sigma_bowl(d)["sigma_m2"]
    assert SH.sigma_bowl(dict(d, n_discs=160))["sigma_m2"] == pytest.approx(2 * s0)
    assert SH.sigma_bowl(dict(d, rpm=12000.0))["sigma_m2"] == pytest.approx(4 * s0)
    # independent quadrature of the settling path between two discs: the particle that crosses half the gap while the
    # liquid flows from r2 to r1 (flow Q/n per channel) defines Q = 2 vg Sigma
    from scipy.integrate import quad
    w, th, n = 2 * math.pi * 6000.0 / 60, math.radians(40.0), 80
    integral, _ = quad(lambda r: w * w * r / 9.80665 * math.cos(th) * 2 * math.pi * r / math.sin(th), 0.05, 0.15)
    assert s0 == pytest.approx(n * integral, rel=1e-10)        # vg·n·∫… = Q/2  ->  Sigma = n·∫…


def test_separator_split_and_costs_follow_the_users_performance():
    r = T.solid_liquid_separation("screw_press", 3000 / H, 0.30, 0.95, 0.5, cepci=567.0)
    assert r["cake"]["solids_kg_s"] * H == pytest.approx(855.0) and r["cake"]["total_kg_s"] * H == pytest.approx(1710.0)
    assert r["liquid"]["total_kg_s"] * H == pytest.approx(1290.0) and r["liquid"]["solids_kg_s"] * H == pytest.approx(45.0)
    S = 3000 / LB
    assert r["cost"]["purchase_cost_usd"] == pytest.approx(
        math.exp(11.0991 - 0.3580 * math.log(S) + 0.05853 * math.log(S) ** 2), rel=1e-9)          # BioSTEAM ScrewPress
    assert r["cost"]["F_BM"] == 1.39 and r["biosteam_reference_power_W"] == pytest.approx(37.2 * 3.0 * 1e3)
    # decanter: 5 short ton/h of solids -> 68040 S^0.5 at CE 567, F_BM 2.03 (BioSTEAM SolidsCentrifuge)
    feed = 5 * SHORT_TON_H / 0.13
    dec = T.solid_liquid_separation("decanter_centrifuge", feed, 0.13, 0.98, 0.6, feed_density_kg_m3=1040.0, cepci=567.0)
    assert dec["cost"]["purchase_cost_usd"] == pytest.approx(68040.0 * 5 ** 0.5, rel=1e-9) and dec["cost"]["F_BM"] == 2.03
    assert dec["biosteam_reference_power_W"] == pytest.approx(1.40 * 3.6e6 * feed / 1040.0)
    assert any("sigma" in s for s in dec["not_included"]) and not dec["warnings"]
    disc = T.solid_liquid_separation("disc_stack_centrifuge", 30000 / H, 0.01, 0.99, 0.7, feed_density_kg_m3=1000.0,
                                     cepci=525.4)
    assert disc["cost"]["purchase_cost_usd"] == pytest.approx(28100.0 * 30 ** 0.574, rel=1e-9)
    assert T.solid_liquid_separation("disc_stack_centrifuge", 30000 / H, 0.01, 0.99, 0.7)["cost"] is None
    pf = T.solid_liquid_separation("plate_and_frame_filter", 10000 / H, 0.2, 0.99, 0.5, cepci=551.0)
    assert pf["cake"]["total_kg_s"] * H == pytest.approx(3960.0)
    assert pf["cost"]["purchase_cost_usd"] == pytest.approx(3294700.0 * (3960.0 / 31815.0) ** 0.8, rel=1e-9)
    b = SH._biosteam_data()
    assert b["solids_centrifuge"]["source_ok"] and b["rotary_vacuum_filter"]["source_ok"]
    for res in (r, dec, disc, pf):
        assert res["evidence"]["verified"] and res["evidence"]["mass_balance_rel"] < 1e-12
        item = res["capital_items"][0]
        assert {"name", "F_BM", "purchase_cost_usd", "base_purchase_cost_usd", "basis"} <= set(item)
    with pytest.raises(ValueError, match="humedad"):
        T.solid_liquid_separation("screw_press", 1.0, 0.30, 1.0, 0.8)       # cake wetter than the feed


def test_capital_items_feed_capital_investment():
    from packages.dwsim_client import tools_economics as TE
    items = (T.size_reduction(10 * SHORT_TON_H, "specific_energy", specific_energy_J_kg=2e4,
                              equipment="hammer_mill")["capital_items"]
             + T.solid_liquid_separation("screw_press", 3000 / H, 0.30, 0.95, 0.5)["capital_items"])
    factors = dict(site_preparation=0.0, service_facilities=0.0, allocated_costs=0.0, contingency_and_fee=0.0, land=0.0,
                   royalties=0.0, startup=0.0, working_capital=0.0)
    ci = TE.capital_investment(items, factors)
    expected = sum(i["purchase_cost_usd"] * i["F_BM"] for i in items)
    assert ci["total_bare_module_cost_usd"] == pytest.approx(expected, rel=1e-9)
    assert ci["total_purchase_cost_usd"] == pytest.approx(sum(i["purchase_cost_usd"] for i in items), rel=1e-9)


# ------------------------------------------------------------------------------------------------------------- oil extraction
GK_N = [2.00, 1.98, 1.94, 1.89, 1.82, 1.75, 1.68, 1.61]          # kg inert solid / kg solution retained      [GK12] 12.10-1
GK_Y = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]                  # kg oil / kg solution


def _gk_leach(target):
    return T.oil_extraction("solvent", solids={"inert_solids_kg_s": 2000 / H, "oil_kg_s": 800 / H, "solvent_kg_s": 50 / H},
                            solvent={"solvent_kg_s": 1310 / H, "oil_kg_s": 20 / H},
                            underflow={"curve": {"oil_mass_fraction_in_solution": GK_Y, "kg_inert_per_kg_solution": GK_N}},
                            target=target)


def test_leaching_geankoplis_12_10_1():
    """[GK12] 12.10-1: 2000 kg/h inert meal with 800 kg oil and 50 kg benzene, solvent 1310 kg benzene + 20 kg oil, leached
    solids to carry 120 kg oil, underflow N(y) tabulated. FOSSEE: L_N = 1014.0 kg/h, y_N = 0.1183, V_1 = 1166.0 kg/h,
    x_1 = 0.600, 4 stages (the book solves it graphically; its own printed numbers were not read).
    Measured vs FOSSEE: L_N -0.10 %, y_N +0.14 %, V_1 +0.09 %, x_1 -0.03 %, stages 4 (3.97 fractional). The differences
    come from the interpolation of N(y): PCHIP here, quadratic in FOSSEE."""
    r = _gk_leach({"residual_oil_kg_s": 120 / H})
    assert r["leached_solids"]["solution_kg_s"] * H == pytest.approx(1014.0, rel=2e-3)
    assert r["leached_solids"]["oil_mass_fraction_in_solution"] == pytest.approx(0.1183, rel=2e-3)
    assert r["miscella"]["total_kg_s"] * H == pytest.approx(1166.0, rel=2e-3)
    assert r["miscella"]["oil_mass_fraction"] == pytest.approx(0.600, rel=1e-3)
    assert r["stages_ideal"] == 4 and 3.85 < r["stages_ideal_fractional"] < 4.0
    ev = r["evidence"]
    assert ev["verified"] and ev["max_stage_oil_balance_rel"] < 1e-12 and ev["monotonic_profile"]
    assert r["residual_oil_kg_s"] * H == pytest.approx(120.0) and r["oil_recovery_fraction"] == pytest.approx(680 / 800)
    # rating: exactly 4 ideal stages leave slightly less oil than the 120 kg/h asked, and 3 stages leave more
    r4, r3 = _gk_leach({"stages": 4}), _gk_leach({"stages": 3})
    assert r4["evidence"]["verified"] and r4["evidence"]["last_stage_solvent_composition_rel_diff"] < 1e-9
    assert r4["residual_oil_kg_s"] * H < 120.0 < r3["residual_oil_kg_s"] * H
    assert r4["stage_profile"][-1]["overflow_in_oil_mass_fraction"] == pytest.approx(20 / 1330, rel=1e-9)
    # the meal-basis target is the same specification written another way
    r_meal = _gk_leach({"residual_oil_in_meal_mass_fraction": 120.0 / 2120.0})
    assert r_meal["leached_solids"]["solution_kg_s"] == pytest.approx(r["leached_solids"]["solution_kg_s"], rel=1e-10)


def _const_leach(target, V=1.2, R=0.5, B=1.0, oil0=0.25, solv0=0.05, energy=None, loss=None):
    return T.oil_extraction("solvent", solids={"inert_solids_kg_s": B, "oil_kg_s": oil0, "solvent_kg_s": solv0},
                            solvent={"solvent_kg_s": V}, underflow={"solution_retained_kg_per_kg_inert": R}, target=target,
                            energy=energy, solvent_loss_kg_per_kg_solids_fed=loss)


def test_leaching_constant_underflow_analytical():
    """ANALYTICAL (derived for this test): constant retention L = B·R and pure solvent V give, with a = V/L,
    y_N = oil0 / [L + V1 (a^N - 1)/(a - 1)], V1 = L0 + V - L. The stage march reproduces it to 1e-10 for N = 1..7."""
    V, R, B, oil0, solv0 = 1.2, 0.5, 1.0, 0.25, 0.05
    L, L0 = B * R, oil0 + solv0
    a, V1 = V / L, L0 + V - L
    closed = lambda n: L * oil0 / (L + V1 * (a ** n - 1) / (a - 1))  # noqa: E731
    for n in range(1, 8):
        r = _const_leach({"stages": n})
        assert r["residual_oil_kg_s"] == pytest.approx(closed(n), rel=1e-10)
        assert r["evidence"]["verified"] and r["stages_ideal"] == n
    # design: asking for (slightly more than) the 3-stage residual needs 3 stages; slightly less needs 4
    assert _const_leach({"residual_oil_kg_s": closed(3) * 1.0001})["stages_ideal"] == 3
    r = _const_leach({"residual_oil_kg_s": closed(3) * 0.9999})
    assert r["stages_ideal"] == 4 and r["stages_ideal_fractional"] == pytest.approx(3.0, abs=2e-3)
    # more solvent -> fewer stages for the same residual; a stricter residual -> more stages
    spec = {"residual_oil_kg_s": closed(5)}
    assert _const_leach(spec, V=2.4)["stages_ideal"] < _const_leach(spec)["stages_ideal"] == 5
    assert _const_leach({"residual_oil_kg_s": closed(5) / 10})["stages_ideal"] > 5
    # constant-underflow output: the wet solids carry R kg of solution per kg of inert
    assert r["leached_solids"]["solution_kg_s"] == pytest.approx(L)
    assert r["leached_solids"]["kg_inert_per_kg_solution"] == pytest.approx(1 / R)


def test_leaching_infeasible_and_solvent_accounting():
    with pytest.raises(ValueError, match="inalcanzable"):
        _const_leach({"residual_oil_kg_s": 1e-3}, V=0.3)          # V/L < 1: the cascade pinches before the target
    with pytest.raises(ValueError, match="no hay extracción"):
        _const_leach({"residual_oil_kg_s": 0.3})
    r = _const_leach({"residual_oil_kg_s": 0.005}, energy={"stream_T_K": 328.15, "stripping_pressure_Pa": 101325.0},
                     loss=2.45e-3)
    solvent_out = r["miscella"]["solvent_kg_s"] + r["leached_solids"]["solvent_kg_s"]
    assert solvent_out == pytest.approx(1.2 + 0.05, rel=1e-12)                       # solvent is conserved
    assert r["solvent_makeup_kg_s"] == pytest.approx(2.45e-3 * 1.30)
    s = r["stripping"]
    # NOT VALIDATED against a plant figure: internal consistency with CoolProp's n-hexane only
    from CoolProp.CoolProp import PropsSI
    assert s["saturation_T_K"] == pytest.approx(PropsSI("T", "P", 101325.0, "Q", 0, "n-Hexane"))
    assert s["enthalpy_rise_J_kg"] > s["latent_heat_J_kg"] > 3.0e5
    assert s["total_duty_W"] == pytest.approx(s["enthalpy_rise_J_kg"] * solvent_out)
    assert r["evidence"]["verified"]


def test_expeller_mass_balance_identities():
    """CONSISTENCY ONLY: the expeller is a mass balance on the user's recovery / residual oil (no predictive model)."""
    seed = {"mass_flow_kg_s": 1.0, "oil_mass_fraction": 0.45}
    a = T.oil_extraction("expeller", seed=seed, expeller={"residual_oil_in_cake_mass_fraction": 0.08})
    assert a["cake"]["total_kg_s"] == pytest.approx(0.55 / 0.92) and a["crude_oil"]["oil_kg_s"] == pytest.approx(1 - 0.55 / 0.92)
    b = T.oil_extraction("expeller", seed=seed, expeller={"oil_recovery_fraction": a["oil_recovery_fraction_of_press_feed"]})
    assert b["cake"]["oil_mass_fraction"] == pytest.approx(0.08, rel=1e-12)          # the two specifications are equivalent
    assert a["evidence"]["verified"] and a["power_W"] is None and a["cost"] is None
    assert any("cotización" in s for s in a["not_included"])
    # decortication: hull leaves first with its own oil; the balance still closes
    dec = {"hull_mass_fraction": 0.25, "hull_removal_fraction": 0.8, "oil_mass_fraction_in_hull": 0.02}
    c = T.oil_extraction("expeller", seed=dict(seed, decortication=dec),
                         expeller={"oil_recovery_fraction": 0.776, "solids_in_crude_oil_mass_fraction": 0.1,
                                   "specific_energy_J_kg": 0.16e6, "estimate_cost": True}, cepci=567.0)
    assert c["hull"]["total_kg_s"] == pytest.approx(0.2) and c["hull"]["oil_kg_s"] == pytest.approx(0.004)
    assert c["press_feed"]["total_kg_s"] == pytest.approx(0.8)
    total = c["hull"]["total_kg_s"] + c["crude_oil"]["total_kg_s"] + c["cake"]["total_kg_s"]
    oil = c["hull"]["oil_kg_s"] + c["crude_oil"]["oil_kg_s"] + c["cake"]["oil_kg_s"]
    assert total == pytest.approx(1.0, rel=1e-12) and oil == pytest.approx(0.45, rel=1e-12)
    assert c["crude_oil"]["solids_kg_s"] / c["crude_oil"]["total_kg_s"] == pytest.approx(0.1)
    assert c["power_W"] == pytest.approx(0.16e6 * 0.8)
    S = 0.8 * H / LB
    assert c["cost"]["purchase_cost_usd"] == pytest.approx(
        math.exp(11.0991 - 0.3580 * math.log(S) + 0.05853 * math.log(S) ** 2), rel=1e-9)
    assert any("orden de magnitud" in w for w in c["warnings"])
    # the cake of the press is the feed of the solvent extractor
    cake = a["cake"]
    lx = T.oil_extraction("solvent", solids={"inert_solids_kg_s": cake["oil_free_solids_kg_s"], "oil_kg_s": cake["oil_kg_s"]},
                          solvent={"solvent_kg_s": 0.6}, underflow={"solution_retained_kg_per_kg_inert": 0.6},
                          target={"residual_oil_in_meal_mass_fraction": 0.01})
    assert lx["evidence"]["verified"] and lx["residual_oil_in_meal_mass_fraction"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------------------------- refusals, JSON, hygiene, wiring
def test_missing_user_data_is_refused_in_spanish():
    cases = [
        (lambda: T.size_reduction(1.0, "bond", 0.05, 0.005), "work_index_kWh_per_ton"),
        (lambda: T.size_reduction(1.0, "bond", 0.05, 0.005, work_index_kWh_per_ton=12.7), "ton"),
        (lambda: T.size_reduction(1.0, "bond", None, 0.005, work_index_kWh_per_ton=12.7, ton="short"), "feed_size_m"),
        (lambda: T.size_reduction(1.0, "kick", 0.05, 0.005), "kick_constant_J_kg"),
        (lambda: T.size_reduction(1.0, "specific_energy"), "specific_energy_J_kg"),
        (lambda: T.size_reduction(1.0, "bond", 0.005, 0.05, work_index_kWh_per_ton=12.7, ton="short"), "menor"),
        (lambda: T.solids_conveying("belt", 1.0, length_m=10.0), "lift_m"),
        (lambda: T.solids_conveying("belt", 1.0, lift_m=1.0), "length_m"),
        (lambda: T.solids_conveying("bucket_elevator", 1.0, lift_m=10.0), "friction_factor"),
        (lambda: T.solids_conveying("pneumatic", 1.0, lift_m=10.0), "conveyor"),
        (lambda: T.solid_liquid_separation("screw_press", 1.0, 0.3, None, 0.5), "solids_recovery_fraction"),
        (lambda: T.solid_liquid_separation("screw_press", 1.0, 0.3, 0.9, None), "cake_moisture_mass_fraction"),
        (lambda: T.solid_liquid_separation("rotary_vacuum_drum_filter", 1.0, 0.3, 0.9, 0.5,
                                           filtration={"medium_resistance_1_m": 0.0}), "specific_cake_resistance_m_kg"),
        (lambda: T.solid_liquid_separation("decanter_centrifuge", 1.0, 0.3, 0.9, 0.5, sigma={"particle_diameter_m": 1e-6}),
         "particle_density_kg_m3"),
        (lambda: T.solid_liquid_separation("screw_press", 1.0, 0.3, 0.9, 0.5, sigma={"particle_diameter_m": 1e-6}), "sigma"),
        (lambda: T.oil_extraction("expeller", seed={"mass_flow_kg_s": 1.0}, expeller={}), "oil_mass_fraction"),
        (lambda: T.oil_extraction("expeller", seed={"mass_flow_kg_s": 1.0, "oil_mass_fraction": 0.4}, expeller={}),
         "oil_recovery_fraction"),
        (lambda: T.oil_extraction("solvent", solids={"inert_solids_kg_s": 1.0, "oil_kg_s": 0.2},
                                  solvent={"solvent_kg_s": 1.0}, target={"stages": 3}), "underflow"),
        (lambda: T.oil_extraction("solvent", solids={"inert_solids_kg_s": 1.0, "oil_kg_s": 0.2},
                                  solvent={"solvent_kg_s": 1.0}, underflow={"solution_retained_kg_per_kg_inert": 0.5},
                                  target={}), "target"),
        (lambda: T.oil_extraction("soxhlet"), "mode"),
    ]
    for call, word in cases:
        with pytest.raises(ValueError, match=word):
            call()


def test_every_result_is_json_and_carries_the_house_keys():
    results = [
        T.size_reduction(2.0, "bond", 0.05, 0.002, work_index_kWh_per_ton=12.7, ton="short", equipment="hammer_mill"),
        T.size_reduction(2.0, "kick", 0.05, 0.02, kick_constant_J_kg=4e3, equipment="cane_crushing_mill"),
        T.solids_conveying("belt", 5.0, length_m=20.0, lift_m=4.0, bulk_density_kg_m3=650.0),
        T.solids_conveying("screw", 5.0, length_m=8.0, lift_m=1.0, bulk_density_kg_m3=650.0),
        T.solids_conveying("bucket_elevator", 5.0, lift_m=15.0, friction_factor=1.2),
        _drum(Rm=5e10),
        T.solid_liquid_separation("disc_stack_centrifuge", 5.0, 0.02, 0.97, 0.7, sigma={
            "particle_diameter_m": 2e-6, "particle_density_kg_m3": 1100.0, "liquid_density_kg_m3": 920.0,
            "liquid_viscosity_Pa_s": 0.03, "bowl": {"type": "disc_stack", "rpm": 6000.0, "r_inner_m": 0.05,
                                                    "r_outer_m": 0.15, "n_discs": 100, "half_angle_deg": 40.0}}),
        _gk_leach({"residual_oil_kg_s": 120 / H}),
        T.oil_extraction("expeller", seed={"mass_flow_kg_s": 1.0, "oil_mass_fraction": 0.45},
                         expeller={"oil_recovery_fraction": 0.45}),
        SH.published_values(),
    ]
    for r in results[:-1]:
        s = json.dumps(r, ensure_ascii=False)
        assert "NaN" not in s and "Infinity" not in s
        for key in ("warnings", "assumptions", "not_included", "method", "evidence"):
            assert key in r, key
        assert isinstance(r["evidence"]["verified"], bool) and r["evidence"]["verified"], r["evidence"]
    json.dumps(results[-1], ensure_ascii=False)
    pub = results[-1]
    assert pub["bond_work_index"]["values"]["limestone"] == 12.7 and pub["bond_work_index"]["url"].startswith("https://")
    assert pub["oilseeds"]["castor"]["url"].startswith("https://") and pub["oilseeds"]["date_read"] == "2026-11-16"
    assert pub["biosteam_reference_duties"]["screw_press_specific_energy_J_kg"] == pytest.approx(37.2 * 3.6e3)


def test_biosteam_global_state_is_left_as_found():
    import biosteam as bst
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty

    def state():
        return (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR, bst.CE,
                bst.PowerUtility.price, getattr(bst.settings, "_thermo", None),
                tuple((a.regeneration_price, a.heat_transfer_price)
                      for a in list(bst.HeatUtility.heating_agents) + list(bst.HeatUtility.cooling_agents)))

    saved = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    try:
        for flags in ((False, False), (True, True)):
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flags
            before = state()
            SH._biosteam_data.cache_clear()                       # force the BioSTEAM read again
            T.size_reduction(10 * SHORT_TON_H, "specific_energy", specific_energy_J_kg=2e4, equipment="hammer_mill")
            T.solid_liquid_separation("screw_press", 3000 / H, 0.30, 0.95, 0.5)
            T.solids_conveying("belt", 5.0, length_m=20.0, lift_m=4.0, bulk_density_kg_m3=650.0)
            assert state() == before
        # the guard itself restores whatever is changed inside it
        before = state()
        with SH._biosteam_guard():
            TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = not before[0]
            bst.CE = 1.0
            bst.PowerUtility.price = 9.0
        assert state() == before
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = saved


def test_no_biosteam_import_at_module_import_time():
    import subprocess
    import sys
    code = ("import sys; import packages.dwsim_client.tools_solids_handling as T; "
            "r = T.size_reduction(1.0, 'bond', 0.05, 0.005, work_index_kWh_per_ton=12.7, ton='short'); "
            "assert r['power_W'] > 0; "
            "assert 'biosteam' not in sys.modules and 'thermosteam' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tools_and_agent_wrappers_are_aligned():
    from packages.agent import wrappers_solids_handling as W
    assert [f.__name__ for f in T.SOLIDS_HANDLING_TOOLS] == [w.name for w in W.AGENT_WRAPPERS]
    out = json.loads(W.AGENT_WRAPPERS[0](mass_flow_kg_s=100 * SHORT_TON_H, law="bond", feed_size_m=0.0508,
                                         product_size_m=0.003175, work_index_kWh_per_ton=12.74, ton="short"))
    assert out["power_W"] / 1e3 == pytest.approx(169.6, rel=5e-4)
    out = json.loads(W.AGENT_WRAPPERS[3](mode="expeller", seed={"mass_flow_kg_s": 1.0, "oil_mass_fraction": 0.45},
                                         expeller={"residual_oil_in_cake_mass_fraction": 0.08}))
    assert out["evidence"]["verified"]
