"""The three baseline flowsheets must solve in DWSIM 10 and pass the deterministic verifier.

Reference numbers are DWSIM 10.2.12 (PR) from notebooks/baseline_fluxo_vs_dwsim.md; the
tolerances are regression bands, not physics claims.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.bench.cases_baseline import ALL, compresor_enfriador, depropanizadora_fug, flash_gas_natural  # noqa: E402
from packages.dwsim_client.verify import verify  # noqa: E402


@pytest.mark.parametrize("name", list(ALL))
def test_baseline_balances_close(name):
    fs, _ = ALL[name]()
    rep = verify(fs)
    assert rep.solved, rep.solver_errors
    assert rep.ok, rep.markdown()


def test_flash_regression():
    fs, h = flash_gas_natural()
    verify(fs)
    assert h["feed"].VaporFraction == pytest.approx(0.8762, abs=2e-3)
    assert h["liq"].OverallMoleFraction("Carbon dioxide") == pytest.approx(0.0264, abs=5e-4)


def test_compressor_regression():
    fs, h = compresor_enfriador()
    rep = verify(fs)
    k = next(u for u in rep.units if u.name == "K-1")
    assert h["hp"].TemperatureK == pytest.approx(315.8, abs=0.2)
    assert k.duty_kw == pytest.approx(52.82, abs=0.1)
    assert h["e"].HeatRemovedKW == pytest.approx(79.34, abs=0.1)


def test_fug_regression():
    fs, h = depropanizadora_fug()
    rep = verify(fs)
    t = next(u for u in rep.units if u.name == "T-1")
    assert t.energy_in_kw == pytest.approx(2123.0, rel=2e-3)   # reboiler
    assert t.energy_out_kw == pytest.approx(1917.7, rel=2e-3)  # condenser
    assert h["dest"].MolarFlowMolPerSecond == pytest.approx(50.0, abs=0.05)
    assert h["sc"].MinimumStages == pytest.approx(9.16, abs=0.05)
    assert h["sc"].MinimumRefluxRatio == pytest.approx(1.281, abs=0.01)
