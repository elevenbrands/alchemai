"""SFILES 2.0 and DEXPI Process exports of solved flowsheets (bloque 15)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import packages.bench.cases_baseline as cb
from packages.bench.cases_ref import CASES  # noqa: E402
from packages.dwsim_client import tools as T
from packages.dwsim_client.verify import verify
from packages.export.flowsheet_export import export_all, export_graph, to_sfiles
from packages.hazop.graph import build_graph

BUILDERS = [(c.id, c.build) for c in CASES] + [("compresor_enfriador", cb.compresor_enfriador),
                                                ("depropanizadora_fug", cb.depropanizadora_fug)]

EXPECTED_SFILES = {
    "valvula_jt": "(raw)(v)(prod)",
    "separador": "(raw)(flash)[(prod)](prod)",
    "intercambiador": "(raw){1_in}(hex){1}{1_out}(prod)n|(raw){2_in}(hex){1}{2_out}(prod)",
    "columna_rigurosa": "(raw)(dist)[{bout}(prod)]{tout}(prod)",
    "reciclo": "(raw)(mix)<1(hex)(splt)1(prod)",
    "compresor_enfriador": "(raw)(comp)(hex)(prod)",
}


def _graph(cid, build):
    out = build()
    fs = out[0] if isinstance(out, tuple) else out
    verify(fs)
    return build_graph(fs, cid)


@pytest.mark.parametrize("cid,build", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_reference_flowsheets_roundtrip(cid, build, tmp_path):
    r = export_all(_graph(cid, build), tmp_path)
    assert r["sfiles_roundtrip_ok"], r["sfiles"]
    assert r["dexpi_roundtrip_ok"]
    if cid in EXPECTED_SFILES:
        assert r["sfiles"] == EXPECTED_SFILES[cid]
    for k in ("dexpi_xml", "dexpi_json", "sfiles", "graph_json"):
        assert (tmp_path / r["files"][k].split("/")[-1]).stat().st_size > 0


def test_recycle_block_is_dropped_and_streams_joined():
    eg = export_graph(_graph("reciclo", dict(BUILDERS)["reciclo"]))
    assert eg.dropped == ["REC-1"]
    loop = [e for e in eg.edges if e.streams == ["recycle", "recycle-in"]]
    assert len(loop) == 1 and (loop[0].src, loop[0].dst) == ("splt-1", "mix-1")
    assert loop[0].data.mass_flow_kg_s == pytest.approx(40.0, abs=1e-2)
    assert "OT_Recycle" not in to_sfiles(eg)


def test_export_tool_on_session(tmp_path):
    fid = T.new_flowsheet("pump", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "water", temperature_K=300, pressure_Pa=101325, mass_flow_kg_s=10)
    T.add_material_stream(fid, "hp")
    T.add_energy_stream(fid, "W")
    T.add_unit(fid, "pump", "P-1", {"outlet_P_Pa": 10e5, "efficiency_pct": 75})
    T.connect(fid, "P-1", feeds=["water"], products=["hp"], energy_feeds=["W"])
    assert T.solve(fid)["ok"]
    r = T.export_flowsheet(fid, str(tmp_path))
    assert r["sfiles"] == "(raw)(pp)(prod)" and r["sfiles_roundtrip_ok"] and r["dexpi_roundtrip_ok"]
    g = json.loads((tmp_path / f"{fid}.graph.json").read_text())
    hp = [e for e in g["edges"] if e["streams"] == ["hp"]][0]
    assert hp["P_Pa"] == pytest.approx(10e5, rel=1e-6)
    assert "Pumping" in (tmp_path / f"{fid}.xml").read_text()
    T.close_flowsheet(fid)
