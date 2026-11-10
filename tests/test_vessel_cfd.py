"""Flow-through vessel RTD by OpenFOAM (bloque M6a). The age field is checked against what must hold whatever the
solver: the flow-weighted outlet age equals τ = V/Q (closed-vessel theorem), plug flow gives ⟨a⟩ = τ/2 and
σ² ≈ 0, and a real tank answer does not move with the mesh. Skipped where OpenFOAM is not installed."""
import json
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.transport import vessel_cfd as VC  # noqa: E402

needs_foam = pytest.mark.skipif(not VC.openfoam_available(), reason="OpenFOAM not installed")
FULL_X = ({"face": "x-", "center_m": [0.1, 0.1], "size_m": [0.2, 0.2]},
          {"face": "x+", "center_m": [0.1, 0.1], "size_m": [0.2, 0.2]})


@needs_foam
def test_plug_flow_limit():
    r = VC.vessel_rtd(1.0, 0.2, 0.2, 0.002, 1e-6, *FULL_X, cells_per_max_dimension=20, mesh_study=False, walls="slip")
    assert r["space_time_tau_s"] == pytest.approx(20.0)
    assert r["outlet_mean_age_s"] == pytest.approx(20.0, rel=0.005)            # theorem: outlet age = V/Q
    assert r["mean_age_over_tau"] == pytest.approx(0.5, rel=0.06)              # plug: age = x/U, mean τ/2
    assert r["normalized_variance"] < 0.1 and r["tanks_in_series_N"] > 10
    assert not r["evidence"]["verified"] and "malla" in " ".join(r["evidence"]["reasons"])   # no mesh study, no seal


@needs_foam
def test_real_tank_short_circuit_is_worse_than_cstr():
    r = VC.vessel_rtd(2.0, 1.0, 1.0, 0.002, 1e-6, {"face": "x-", "center_m": [0.5, 0.85], "size_m": [0.15, 0.15]},
                      {"face": "x+", "center_m": [0.5, 0.15], "size_m": [0.15, 0.15]}, cells_per_max_dimension=30,
                      first_order_k_1_s=0.002)
    ev = r["evidence"]
    assert ev["verified"], ev["reasons"]
    assert ev["outlet_age_vs_tau_rel"] < 0.01 and ev["mesh"]["mean_age_change"] < 0.02
    assert r["turbulence"] == "k-epsilon" and r["normalized_variance"] > 1.05 and "CSTR" in r["diagnosis"]
    x = r["first_order_conversion"]
    assert x["gamma_RTD_N"] < x["CSTR_ideal"] < x["PFR_ideal"]                  # bypass costs conversion
    N, Da = r["tanks_in_series_N"], 0.002 * 1000.0
    assert x["gamma_RTD_N"] == pytest.approx(1 - (1 + Da / N) ** (-N))
    assert sum(h["volume_fraction"] for h in r["age_histogram"]) == pytest.approx(1.0)
    json.dumps(r, allow_nan=False)


@needs_foam
def test_bad_openings_are_rejected():
    with pytest.raises(ValueError, match="2 cells"):
        VC.vessel_rtd(2.0, 1.0, 1.0, 0.002, 1e-6, {"face": "x-", "center_m": [0.5, 0.5], "size_m": [0.05, 0.05]},
                      FULL_X[1], cells_per_max_dimension=30)
    with pytest.raises(ValueError, match="outside"):
        VC.vessel_rtd(1.0, 0.2, 0.2, 0.002, 1e-6, {"face": "x-", "center_m": [0.15, 0.1], "size_m": [0.2, 0.2]},
                      FULL_X[1], cells_per_max_dimension=20)
    with pytest.raises(ValueError, match="face"):
        VC.vessel_rtd(1.0, 0.2, 0.2, 0.002, 1e-6, {"face": "top", "center_m": [0.1, 0.1], "size_m": [0.2, 0.2]},
                      FULL_X[1], cells_per_max_dimension=20)


def test_agent_module_aliases_are_unique():
    src = (Path(__file__).resolve().parents[1] / "packages" / "agent" / "agent.py").read_text()
    aliases = re.findall(r"^from packages\.\S+ import \S+ as (\w+)", src, flags=re.M)
    assert len(aliases) == len(set(aliases)), aliases                          # two modules under one alias = bug
    assert math.isfinite(VC.THEOREM_TOL)
