"""Equipment purchase cost (bloque 19): IDAES's SSLW costing used verbatim, no external solver."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.costing.sslw import CEPCI_IDAES, cost_compressor, cost_heat_exchanger, cost_pump, cost_vessel  # noqa: E402


def test_reproduces_idaes_reference_values():
    # Same inputs and expected values as idaes/models/costing/tests/test_SSLW.py (USD 2018)
    assert cost_heat_exchanger(1000, 2 * 101325, year=2018)["purchase_cost_usd"] == pytest.approx(476063, rel=1e-5)
    assert cost_pump(10143.79, 50000, 986.64, 0.182592, material="Nickel", pump_type_factor=1.4, motor_type="Enclosed",
                     year=2018)["purchase_cost_usd"] == pytest.approx(62868.3, rel=1e-5)
    assert cost_compressor(101410.4, material="StainlessSteel", year=2018)["purchase_cost_usd"] == pytest.approx(300695, rel=1e-5)
    assert cost_vessel(2, 0.00075, year=2018)["purchase_cost_usd"] == pytest.approx(35958, rel=1e-5)


def test_cepci_conversion_and_user_index():
    a, b = cost_heat_exchanger(200, year=2018), cost_heat_exchanger(200, year=2023)
    assert b["purchase_cost_usd"] / a["purchase_cost_usd"] == pytest.approx(CEPCI_IDAES[2023] / CEPCI_IDAES[2018], rel=1e-9)
    c = cost_heat_exchanger(200, cepci=800.0, year=2025)
    assert c["purchase_cost_usd"] == pytest.approx(a["usd_ce500"] * 800.0 / 500.0, rel=1e-9) and "usuario" in c["basis"]
    with pytest.raises(ValueError):
        cost_heat_exchanger(200, year=2025)


def test_economy_of_scale():
    lo, hi = cost_heat_exchanger(100)["purchase_cost_usd"], cost_heat_exchanger(1000)["purchase_cost_usd"]
    n = math.log(hi / lo) / math.log(10)
    assert 0.4 < n < 0.8          # six-tenths-rule territory


def test_bad_option_lists_valid_names():
    with pytest.raises(ValueError, match="CarbonSteel"):
        cost_vessel(2, 5, material="Unobtanium")
