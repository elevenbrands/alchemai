"""Packed-column hydraulics (bloque 23): Stichlmair 1989 worked example and the diameter search."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.transport.column_hydraulics import PACKINGS, flooding_velocity, size_column, wet_dp_per_m  # noqa: E402

PAPER = dict(voidage=0.68, specific_area=260.0, C1=32.0, C2=7.0, C3=1.0)


def test_stichlmair_paper_example():
    assert flooding_velocity(5e-3, 5.0, 1200.0, 5e-5, PAPER) == pytest.approx(0.6394323542746928, rel=1e-9)
    assert wet_dp_per_m(0.4, 5e-3, 5.0, 1200.0, 5e-5, PAPER) == pytest.approx(539.876823725352, rel=1e-9)


def test_diameter_hits_the_flood_fraction():
    for f in (0.5, 0.7, 0.85):
        r = size_column(1.0, 0.005, 5.0, 1200.0, 5e-5, PAPER, f)
        assert r["flood_fraction"] == pytest.approx(f, rel=1e-4)
        assert r["gas_velocity_m_s"] == pytest.approx(1.0 / (math.pi * r["diameter_m"] ** 2 / 4), rel=1e-9)
    d50, d85 = size_column(1.0, 0.005, 5.0, 1200.0, 5e-5, PAPER, 0.5), size_column(1.0, 0.005, 5.0, 1200.0, 5e-5, PAPER, 0.85)
    assert d50["diameter_m"] > d85["diameter_m"] and d50["dP_wet_Pa_per_m"] < d85["dP_wet_Pa_per_m"]


def test_more_liquid_floods_earlier_and_catalog_order():
    assert flooding_velocity(2e-3, 5.0, 1200.0, 5e-5, PAPER) > flooding_velocity(1e-2, 5.0, 1200.0, 5e-5, PAPER)
    dia = {k: size_column(0.5, 0.004, 8.0, 700.0, 1.1e-5, k, 0.7)["diameter_m"] for k in ("Mellapak 250Y", "Pall rings 25 mm", "Pall rings 16 mm")}
    assert dia["Mellapak 250Y"] < dia["Pall rings 25 mm"] < dia["Pall rings 16 mm"]      # structured > larger random > smaller random capacity
    with pytest.raises(ValueError):
        size_column(0.5, 0.004, 8.0, 700.0, 1.1e-5, "Unobtanium", 0.7)
    assert set(PACKINGS) >= {"Mellapak 250Y", "Pall rings 25 mm"}
