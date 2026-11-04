"""Bench v2: 4 polar-mixture cards whose statements never name a property package. The oracle scores 100 %,
the null agent 0 %, and the expected values are DWSIM-NRTL (Peng-Robinson is off by 6.6 K on the bubble point,
0.03 in VF, and does not converge the methanol-water column; see polar_cases.main)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.bench.runner import load_cards, null_agent, oracle_agent, run_bench  # noqa: E402


def test_four_polar_cards_without_package_in_statement():
    cards = load_cards(version="v2")
    assert len(cards) == 4 and all("polares" in c["tags"] for c in cards)
    for c in cards:
        for lang in ("es", "en"):
            txt = c["statement"][lang].lower()
            assert "nrtl" not in txt and "peng" not in txt and "robinson" not in txt, c["id"]
            assert "paquete" in txt or "package" in txt
        assert c["requires"] == ["converged", "balances_closed"]


def test_expected_values_are_nrtl_physics():
    e = {c["id"]: {k: v["value"] for k, v in c["expected"].items()} for c in load_cards(version="v2")}
    b = e["polar_burbuja_rocio_etanol_agua"]
    assert 352.5 < b["T_bubble_K"] < 353.5 and 357.0 < b["T_dew_K"] < 358.0        # PR gives 346.4 / 356.5
    f = e["polar_flash_metanol_agua"]
    assert 0.52 < f["VF"] < 0.55 and f["y_MeOH"] > 0.4 > f["x_MeOH"]
    col = e["polar_columna_metanol_agua"]
    assert col["x_MeOH_dist"] > 0.98 and col["x_MeOH_bot"] < 0.01 and col["Qc_kW"] > 0 and col["Qr_kW"] > 0
    az = e["polar_azeotropo_etanol_agua"]
    assert 0.85 < az["x_az_EtOH"] < 0.90 and 351.0 < az["T_az_K"] < 351.8


def test_oracle_full_marks_null_zero():
    cards = load_cards(version="v2")
    assert run_bench(oracle_agent, cards, verbose=False)[0] == 1.0
    assert run_bench(null_agent, cards, verbose=False)[0] == 0.0
