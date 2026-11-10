"""Pinch analysis (bloque M5) against two textbook problems and the properties every target must have.

Kemp, *Pinch Analysis and Process Integration* (2007), four-stream example, ΔTmin = 10 K: Q_H,min = 20 kW,
Q_C,min = 60 kW, pinch at 90 °C (hot) / 80 °C (cold). Smith, *Chemical Process Design and Integration* (2016),
four-stream example, ΔTmin = 10 K: Q_H,min = 7.5 MW, Q_C,min = 10 MW, shifted pinch 145 °C."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools_energy as TE  # noqa: E402

K = 273.15
KEMP = [dict(name="1", T_supply_K=20 + K, T_target_K=135 + K, CP_W_K=2e3),
        dict(name="2", T_supply_K=170 + K, T_target_K=60 + K, CP_W_K=3e3),
        dict(name="3", T_supply_K=80 + K, T_target_K=140 + K, CP_W_K=4e3),
        dict(name="4", T_supply_K=150 + K, T_target_K=30 + K, CP_W_K=1.5e3)]
SMITH = [dict(name="R1 feed", T_supply_K=20 + K, T_target_K=180 + K, CP_W_K=0.2e6),
         dict(name="R1 product", T_supply_K=250 + K, T_target_K=40 + K, CP_W_K=0.15e6),
         dict(name="R2 feed", T_supply_K=140 + K, T_target_K=230 + K, CP_W_K=0.3e6),
         dict(name="R2 product", T_supply_K=200 + K, T_target_K=80 + K, CP_W_K=0.25e6)]


def test_kemp_four_stream_targets():
    r = TE.pinch_analysis(KEMP, 10.0)
    assert r["Q_hot_min_W"] == pytest.approx(20e3) and r["Q_cold_min_W"] == pytest.approx(60e3)
    (p,) = r["pinch"]
    assert (p["T_hot_side_K"] - K, p["T_cold_side_K"] - K) == pytest.approx((90.0, 80.0))
    assert r["heat_recovery_W"] == pytest.approx(450e3) and not r["threshold_problem"]
    ev = r["evidence"]
    assert ev["verified"] and ev["curves_touch_without_crossing"] and ev["target_is_minimum"]
    assert r["minimum_units"]["total_MER"] == 7 and r["minimum_units"]["without_pinch_split"] == 5
    gcc = {round(g["T_shifted_K"] - K, 6): g["heat_flow_W"] for g in r["grand_composite_curve"]}
    assert gcc[85.0] == pytest.approx(0.0, abs=1e-6) and gcc[165.0] == pytest.approx(20e3)


def test_smith_four_stream_targets():
    r = TE.pinch_analysis(SMITH, 10.0)
    assert r["Q_hot_min_W"] == pytest.approx(7.5e6) and r["Q_cold_min_W"] == pytest.approx(10e6)
    assert r["pinch"][0]["T_shifted_K"] - K == pytest.approx(145.0) and r["evidence"]["verified"]
    crossing = {s["name"] for s in r["streams"] if s["crosses_pinch"]}
    assert crossing == {"R1 feed", "R1 product", "R2 product"}          # R2 feed starts at the pinch


def test_problem_table_closes_and_composites_are_consistent():
    r = TE.pinch_analysis(KEMP, 10.0)
    assert sum(row["surplus_W"] for row in r["problem_table"]) == pytest.approx(r["Q_cold_min_W"] - r["Q_hot_min_W"])
    hot, cold = r["hot_composite_curve"], r["cold_composite_curve"]
    assert hot[-1]["H_W"] == pytest.approx(510e3) and cold[0]["H_W"] == pytest.approx(60e3)       # cold starts at Q_C
    assert cold[-1]["H_W"] - cold[0]["H_W"] == pytest.approx(470e3)


def test_dT_min_sweep_threshold_and_current_utilities():
    r = TE.pinch_analysis(KEMP, 10.0, dT_min_sweep_K=[5, 10, 20, 30], current_hot_utility_W=100e3,
                          current_cold_utility_W=140e3)
    QH = [s["Q_hot_min_W"] for s in r["dT_min_sweep"]]
    assert QH == sorted(QH) and QH[1] == pytest.approx(20e3)
    assert r["dT_min_sweep"][0]["pinch_shifted_K"] is None                # ΔTmin 5 K: below the threshold
    c = r["current_vs_target"]
    assert c["hot_excess_W"] == pytest.approx(80e3) and c["cold_excess_W"] == pytest.approx(80e3) and c["balance_consistent"]
    th = TE.pinch_analysis([dict(T_supply_K=400, T_target_K=300, CP_W_K=10), dict(T_supply_K=300, T_target_K=350, CP_W_K=10)], 10)
    assert th["threshold_problem"] and th["Q_hot_min_W"] == 0 and th["Q_cold_min_W"] == pytest.approx(500)
    assert th["evidence"]["verified"] and th["pinch"] == [] and th["warnings"]
    low = TE.pinch_analysis(KEMP, 10.0, current_hot_utility_W=10e3)
    assert any("MENOR" in w for w in low["warnings"])


def test_heat_load_input_and_individual_contributions():
    by_load = [dict(s, CP_W_K=None, heat_load_W=s["CP_W_K"] * abs(s["T_supply_K"] - s["T_target_K"])) for s in KEMP]
    assert TE.pinch_analysis(by_load, 10.0)["Q_hot_min_W"] == pytest.approx(20e3)
    # every stream with ΔTc = 5 K is the same as ΔTmin = 10 K
    own = [dict(s, dT_contribution_K=5.0) for s in KEMP]
    assert TE.pinch_analysis(own, 0.0)["Q_hot_min_W"] == pytest.approx(20e3)
    with pytest.raises(ValueError, match="hot stream"):
        TE.pinch_analysis([KEMP[0], KEMP[2]], 10.0)
    with pytest.raises(ValueError, match="CP_W_K"):
        TE.pinch_analysis([dict(T_supply_K=400, T_target_K=300), KEMP[0]], 10.0)


def test_plain_json():
    json.dumps(TE.pinch_analysis(SMITH, 10.0, dT_min_sweep_K=[5, 20], current_hot_utility_W=9e6), allow_nan=False)
