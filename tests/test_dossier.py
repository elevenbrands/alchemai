"""One-question dossier (bloque M27): chains validated engines; here only the chaining is checked (consistency)."""
import json
import xml.etree.ElementTree as ET

import pytest

from packages.dossier import dossier as D


def test_tula_dossier_end_to_end(tmp_path):
    r = D.process_dossier("Refinería de Tula", refinery="tula", out_dir=str(tmp_path))
    s = r["summary"]
    assert r["evidence"]["verified"] and s["mode"] == "as_operated"
    assert s["crude_bpd"] == pytest.approx(221583.0)
    # the honest headline exists and differs from the model slate margin (the model overpredicts diesel)
    assert s["gross_margin_at_published_slate_usd_bbl"] < s["gross_margin_model_usd_bbl"]
    # the idle-coker opportunity: on the coker unit, positive, low <= value <= high, and drawn on the diagram
    cap = [o for o in r["opportunities"] if o.get("rule") == "capacidad_ociosa"]
    assert len(cap) == 1 and cap[0]["unit_ids"] and cap[0]["value_usd_y"] > 0
    assert cap[0]["value_low_usd_y"] <= cap[0]["value_usd_y"] <= cap[0]["value_high_usd_y"]
    for p in (r["diagram_sections_path"], r["diagram_full_path"]):
        ET.parse(p)                                        # valid SVG
    md = open(r["report_path"], encoding="utf-8").read()
    assert "CALIBRADOS" in md and "Oportunidades" in md and "producción PUBLICADA" in md
    json.dumps(r, default=float)


def test_dossier_refuses_ambiguous_subject(tmp_path):
    with pytest.raises(ValueError):
        D.process_dossier("x", out_dir=str(tmp_path))
    with pytest.raises(ValueError):
        D.process_dossier("x", refinery="tula", plant={"model": "biodiesel"}, out_dir=str(tmp_path))


def test_slug_handles_accents():
    assert D._slug("Refinería de Tula") == "refineria_de_tula"


def test_laguna_verde_dossier(tmp_path):
    r = D.process_dossier("Central Laguna Verde", power_plant="laguna_verde", out_dir=str(tmp_path))
    s = r["summary"]
    assert r["evidence"]["verified"] and s["kind"] == "power"
    assert 0.30 < s["gross_efficiency"] < 0.40                    # BWR range (Carnot-bounded, checked in test_power)
    assert s["operating_margin_usd_y"] == pytest.approx(s["net_generation_MWh"] * s["operating_margin_usd_mwh"], rel=1e-9)
    assert r["opportunities"]
    md = open(r["report_path"], encoding="utf-8").read()
    assert "CALIBRADOS" in md and "Validación" in md
    ET.parse(r["diagram_full_path"])


def test_tula_summary_carries_the_liquid_volume_diagnosis(tmp_path):
    s = D.process_dossier("Refinería de Tula", refinery="tula", out_dir=str(tmp_path))["summary"]
    le = s["liquid_excess"]
    assert le and le["liquids_excess_bpd"] == pytest.approx(18106, rel=0.02)
    assert le["own_use_upper_bound_bpd"] < le["liquids_excess_bpd"] and "no tiene dato público" in le["text_es"]
