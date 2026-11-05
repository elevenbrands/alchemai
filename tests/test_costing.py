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


def test_small_exchanger_priced_at_correlation_minimum():
    from packages.costing.sslw import cost_heat_exchanger_floored
    kw = {"shell_pressure_Pa": 4e5, "material": "CarbonSteelCarbonSteel"}
    tiny, at_min, big = (cost_heat_exchanger_floored(a, **kw) for a in (0.1, 9.0, 40.0))
    assert cost_heat_exchanger(0.1, **kw)["purchase_cost_usd"] > big["purchase_cost_usd"]     # the raw correlation misbehaves
    assert tiny["purchase_cost_usd"] <= at_min["purchase_cost_usd"] + 1e-6 and "warning" in tiny
    assert 5.0 < tiny["priced_at_area_m2"] < 15.0 and "warning" not in big


def test_annual_cost_crf_and_lang():
    from packages.costing.annual import LANG, annual_cost, crf, location_factor
    assert crf(0.10, 10) == pytest.approx(0.162745, rel=1e-5) and crf(0.0, 5) == 0.2
    r = annual_cost({"E-1": 100_000.0}, {"H-1": ("heating", 1000.0), "P-1": ("electricity", 10.0)},
                    {"heating_usd_GJ": 8.0, "electricity_usd_kWh": 0.1}, 8000, 0.10, 10, "lang_1948_fluid", 1.0)
    assert r["fixed_capital_usd"] == pytest.approx(474_000.0)
    assert r["operating_by_unit_usd_y"]["H-1"] == pytest.approx(1000 * 8000 * 3600 / 1e6 * 8.0)
    assert r["total_annual_cost_usd_y"] == pytest.approx(474_000 * crf(0.1, 10) + 230_400 + 8_000)
    assert LANG["peters_2003_fluid_tci"] == 6.0 and 0.84 <= location_factor("Mexico") <= 0.92
    with pytest.raises(KeyError):
        annual_cost({}, {"C-1": ("cooling", 5.0)}, {}, 8000, 0.1, 10)
