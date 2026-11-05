"""HAZOP from DEXPI (bloque 16): same rows as from DWSIM for exported flowsheets; works on hand-built DEXPI."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import packages.bench.cases_baseline as cb  # noqa: E402
from packages.bench.cases_ref import CASES  # noqa: E402
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.verify import verify  # noqa: E402
from packages.export.flowsheet_export import export_all  # noqa: E402
from packages.hazop.dexpi_examples import gas_compression, write  # noqa: E402
from packages.hazop.dexpi_graph import graph_from_dexpi, hazop_from_dexpi, load_dexpi  # noqa: E402
from packages.hazop.graph import build_graph  # noqa: E402
from packages.hazop.rules import hazop_rows  # noqa: E402
from test_hazop import _wet_gas  # noqa: E402

BUILDERS = [(c.id, c.build) for c in CASES] + [("compresor_enfriador", cb.compresor_enfriador),
                                                ("depropanizadora_fug", cb.depropanizadora_fug)]


def _same_as_dwsim(g, tmp_path):
    ref = [r.to_dict() for r in hazop_rows(g) if r.kind != "OT_Recycle"]   # the recycle block is logical, not equipment
    files = export_all(g, tmp_path)["files"]
    for fmt in ("dexpi_xml", "dexpi_json"):
        h = hazop_from_dexpi(files[fmt])
        assert h["missing"] == [] and h["rows"] == ref, fmt
    return ref


@pytest.mark.parametrize("cid,build", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_reference_flowsheets_same_hazop(cid, build, tmp_path):
    out = build()
    fs = out[0] if isinstance(out, tuple) else out
    verify(fs)
    assert _same_as_dwsim(build_graph(fs, cid), tmp_path)


def test_wet_gas_same_hazop_including_hydrates(tmp_path):
    fid = _wet_gas()
    ref = _same_as_dwsim(build_graph(T._s(fid)["fs"], fid), tmp_path)
    jt = [r for r in ref if r["node"] == "VLV-1" and "Joule-Thomson" in r["deviation"]]
    assert jt and any("hidratos" in c for c in jt[0]["consequences"])
    T.close_flowsheet(fid)


def test_hand_built_dexpi_without_simulator(tmp_path):
    h = hazop_from_dexpi(write(gas_compression(), tmp_path, "gas"))
    assert h["missing"] == [] and h["units"] == {"K-100": "Compressor", "E-100": "Cooler", "V-100": "Separator"}
    g, _ = graph_from_dexpi(load_dexpi(tmp_path / "gas.xml"))
    assert g.units["K-100"].P_in_Pa == pytest.approx(30e5) and g.units["K-100"].T_out_K == pytest.approx(378.15)
    assert g.streams["100"].mass_flow_kg_s == pytest.approx(5.0) and g.streams["102"].hydrate_risk
    over = [r for r in h["rows"] if r["node"] == "E-100" and "overcooling" in r["deviation"]]
    assert over and "hidratos" in over[0]["consequences"][0] and "inhibidor" in over[0]["safeguards"][0]
    assert all("no inflamable" not in r["evidence"] for r in h["rows"])


def test_missing_composition_is_reported_not_invented(tmp_path):
    h = hazop_from_dexpi(write(gas_compression(with_composition=False), tmp_path, "gas_nocomp"))
    assert len(h["missing"]) == 5 and all("composition" in m for m in h["missing"])
    assert all("no inflamable" in r["evidence"] for r in h["rows"])
    assert not any("hidratos" in " ".join(r["consequences"]) for r in h["rows"])


def test_tool_registered():
    assert T.hazop_from_dexpi in T.TOOLS
