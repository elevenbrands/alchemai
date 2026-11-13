"""Minimal P&ID from flowsheet + HAZOP as DEXPI P&ID (bloque M10).

Evidence:
* rule coverage on the HAZOP reference cases (12 DWSIM flowsheets + 10 published HAZOPs as hand-built DEXPI):
  every liquid inventory has a level loop, every overpressure row a PSV, every reverse-flow row a check valve,
  every alarm/trip-grade deviation an alarm/trip, and nothing that no rule or HAZOP row asked for;
* the DEXPI P&ID loads back in pyDEXPI (XML and JSON) with the same equipment, items, instruments, loops,
  actuators and signals, passes pyDEXPI's DEXPI rules, and validates against the official DEXPI 2.0.1 XSD;
* published control structure: Skogestad (1997) Fig. 1, LV configuration, plus the temperature loop of
  Skogestad (2007), reproduced on every distillation column.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import packages.bench.cases_baseline as cb  # noqa: E402
from packages.bench.cases_ref import CASES  # noqa: E402
from packages.dwsim_client.verify import verify  # noqa: E402
from packages.hazop.book.cases import CASES as BOOK  # noqa: E402
from packages.hazop.book.heldout_cases import HELDOUT  # noqa: E402
from packages.hazop.dexpi_graph import graph_from_dexpi  # noqa: E402
from packages.hazop.graph import build_graph  # noqa: E402
from packages.hazop.rules import HazopRow, hazop_rows  # noqa: E402
from packages.pid import dexpi_pid as DP  # noqa: E402
from packages.pid.rules import (ABSORBERS, COLUMNS, INLINE, NOT_ALARMED, REACTORS_WITH_INVENTORY, ROTATING, RULES,  # noqa: E402
                                VAR, VESSELS, build_pid, row_ref)
from packages.pid.tool import pid_from_graph, pid_from_hazop  # noqa: E402

BUILDERS = [(c.id, c.build) for c in CASES] + [("compresor_enfriador", cb.compresor_enfriador),
                                                ("depropanizadora_fug", cb.depropanizadora_fug)]
DEXPI_CASES = {**BOOK, **HELDOUT}


def _dwsim_graph(cid, build):
    out = build()
    fs = out[0] if isinstance(out, tuple) else out
    verify(fs)
    return build_graph(fs, cid)


def _liquid(l):
    return l.vapor_fraction is not None and l.vapor_fraction == l.vapor_fraction and l.vapor_fraction < 0.5


def check_coverage(g):
    spec = build_pid(g)
    rows = [r for r in hazop_rows(g) if r.kind != "OT_Recycle"]
    refs = {row_ref(i, r): r for i, r in enumerate(rows)}
    lines = spec.lines
    loops = [i for i in spec.instruments if i.category == "lazo"]
    outs = lambda u: [l for l in lines.values() if l.kind == "proceso" and l.src == u]
    ins = lambda u: [l for l in lines.values() if l.kind == "proceso" and l.dst == u]

    # every element is traced to a known rule; HAZOP elements to at least one real HAZOP row
    for e in spec.items + spec.instruments:
        assert e.rule in RULES, e
        assert all(h in refs for h in e.hazop), e
        if e.rule.startswith("H-"):
            assert e.hazop, f"{e.tag}: elemento HAZOP sin fila"

    # level loops: one per liquid outlet on every vessel / reactor with liquid inventory, two per column (MD, MB)
    for u in g.units.values():
        L = [x for x in loops if x.unit == u.name and x.variable == "L"]
        n_liq = sum(_liquid(l) for l in outs(u.name))
        if u.kind in VESSELS or u.kind in REACTORS_WITH_INVENTORY:
            if n_liq:
                assert len(L) == n_liq, (u.name, L)
            elif u.kind in VESSELS and any(_liquid(l) for l in ins(u.name)):
                assert len(L) == 1, (u.name, L)
            else:
                assert not L
        elif u.kind in COLUMNS:
            assert len(L) == 2, (u.name, L)
        elif u.kind in ABSORBERS:
            assert len(L) == 1
        else:
            assert not L, f"lazo de nivel espurio en {u.name} ({u.kind})"
        # flow loops only (and always) on rotating equipment with a discharge
        F = [x for x in loops if x.unit == u.name and x.variable == "F"]
        assert len(F) == (1 if u.kind in ROTATING and outs(u.name) else 0), (u.name, F)
        # temperature loops: heaters, coolers, reactors with duty, distillation columns
        T = [x for x in loops if x.unit == u.name and x.variable == "T"]
        want_T = (u.kind in ("Heater", "Cooler") or u.kind in COLUMNS
                  or (u.kind.startswith("RCT") and (u.exothermic or abs(u.duty_kW or 0.0) > 1e-6)))
        assert len(T) == int(want_T), (u.name, T)
        # pressure loops: one per vapor outlet of a vessel (not an atmospheric tank), one per column
        P = [x for x in loops if x.unit == u.name and x.variable == "P"]
        if u.kind in VESSELS or (u.kind in REACTORS_WITH_INVENTORY and n_liq):
            n_vap = sum(1 for l in outs(u.name) if l.vapor_fraction is not None and l.vapor_fraction == l.vapor_fraction
                        and l.vapor_fraction >= 0.5 and not (u.kind == "Tank" and (l.P_Pa or 0) < 1.3e5))
            assert len(P) == n_vap, (u.name, P)
        elif u.kind in COLUMNS or u.kind in ABSORBERS:
            assert len(P) == 1
        else:
            assert not P, f"lazo de presión espurio en {u.name}"

    # HAZOP-driven devices: complete and nothing extra
    psv = [i for i in spec.items if i.isa == "PSV"]
    chk = [i for i in spec.items if i.isa == "CHK"]
    alarms = [i for i in spec.instruments if i.category == "alarma"]
    trips = [i for i in spec.instruments if i.category == "disparo"]
    for ref, r in refs.items():
        sg = " | ".join(r.safeguards).lower()
        if r.parameter == "Pressure" and r.deviation.startswith("More pressure"):
            assert any(ref in p.hazop for p in psv), f"sobrepresión sin PSV: {ref}"
        if "reverse flow" in r.deviation.lower():
            assert any(ref in c.hazop for c in chk), f"flujo inverso sin check: {ref}"
        alarmable = (r.parameter in VAR and not any(k in r.deviation.lower() for k in NOT_ALARMED)
                     and r.deviation.lower().startswith(("more", "less", "no", "high")))
        assert any(ref in a.hazop for a in alarms) == (alarmable and r.severity >= 3), ref
        assert any(ref in t.hazop for t in trips) == (alarmable and r.severity >= 4), ref
        assert any(ref in p.hazop for p in psv) == any(k in sg for k in ("psv", "alivio", "disco de ruptura")), ref
        assert any(ref in c.hazop for c in chk) == ("check" in sg), ref

    # isolation: battery limits + suction/discharge of rotating equipment, nothing else
    bv = [i for i in spec.items if i.isa == "BV"]
    want_bv = sum((l.src is None) + (l.dst is None) + (spec.units.get(l.src or "") in ROTATING)
                  + (spec.units.get(l.dst or "") in ROTATING) for l in lines.values() if l.kind == "proceso")
    assert len(bv) == want_bv

    # no design value is invented
    for i in spec.instruments:
        assert "por definir" in i.setpoint or i.setpoint == "—", i
    for p in psv:
        assert "por definir" in p.setpoint
    return spec


def check_dexpi(g, tmp_path, stem):
    out = pid_from_graph(g, tmp_path, stem)
    ev = out["evidence"]
    assert ev["roundtrip_xml"] and ev["roundtrip_json"], stem
    assert all(v for k, v in ev["matches_spec"].items() if k != "diff"), ev["matches_spec"]["diff"]
    assert ev["semantic_issues"] == [], ev["semantic_issues"][:3]
    if ev["xsd"]["status"] == "pendiente":
        pytest.skip("XSD oficial no disponible (sin red)")
    assert ev["xsd"]["status"] == "valido", ev["xsd"]["errors"]
    assert out["verified"]
    return out


@pytest.mark.parametrize("cid,build", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_reference_flowsheets(cid, build, tmp_path):
    g = _dwsim_graph(cid, build)
    check_coverage(g)
    check_dexpi(g, tmp_path, cid)


@pytest.mark.parametrize("cid", list(DEXPI_CASES))
def test_published_hazop_cases_as_dexpi(cid, tmp_path):
    g, _ = graph_from_dexpi(DEXPI_CASES[cid](), cid)
    spec = check_coverage(g)
    assert not spec.undetermined
    check_dexpi(g, tmp_path, cid)


def _column_loops(spec, col):
    return {x.variable + ":" + x.final_element: x for x in spec.instruments if x.unit == col and x.category == "lazo"}


@pytest.mark.parametrize("src", ["columna_rigurosa", "depropanizadora_fug", "csb_texas_city", "csb_williams",
                                 "marhavilas_estabilizadora"])
def test_skogestad_lv_configuration(src):
    """Skogestad (1997) Fig. 1: VT → p (PC), D → MD (LC), B → MB (LC), L and V left for composition; Skogestad (2007):
    close a fast temperature loop (here with V, the reboiler). The rules give exactly these four loops on the column."""
    if src in DEXPI_CASES:
        g, _ = graph_from_dexpi(DEXPI_CASES[src](), src)
    else:
        g = _dwsim_graph(src, dict(BUILDERS)[src])
    spec = build_pid(g)
    for col in [n for n, k in spec.units.items() if k in COLUMNS]:
        outs = sorted([l for l in spec.lines.values() if l.src == col and l.kind == "proceso"], key=lambda l: l.T_K)
        D, B = outs[0].name, outs[-1].name
        loops = [x for x in spec.instruments if x.unit == col and x.category == "lazo"]
        assert len(loops) == 4, [x.tag for x in loops]
        by = {(x.variable, x.rule): x for x in loops}
        p = by[("P", "R-C1")]
        assert spec.lines[next(i.line for i in spec.items if i.tag == p.final_element)].name == f"util-{col}-condensador"
        lv = [x for x in loops if x.variable == "L"]

        def acts_on(x):
            if x.final_element.startswith("cascada"):
                slave = next(i for i in spec.instruments if i.tag == x.cascade_to)
                return next(l.name for l in spec.lines.values() if l.dst == slave.unit and l.kind == "proceso")
            it = next(i for i in spec.items if i.tag == x.final_element)
            return it.line
        assert sorted(acts_on(x) for x in lv) == sorted([D, B]), "niveles con D y B (configuración LV)"
        assert "acumulador" in next(x for x in lv if acts_on(x) == D).detail
        assert "fondos" in next(x for x in lv if acts_on(x) == B).detail
        t = by[("T", "R-C2")]
        assert next(i.line for i in spec.items if i.tag == t.final_element) == f"util-{col}-rehervidor"


def test_tool_on_session_and_on_dexpi_file(tmp_path):
    from packages.dwsim_client import tools as T
    from packages.hazop.dexpi_examples import gas_compression, write
    from test_hazop import _wet_gas
    fid = _wet_gas()
    out = pid_from_hazop(flowsheet_id=fid, out_dir=str(tmp_path / "s"))
    T.close_flowsheet(fid)
    assert out["verified"], out["evidence"]
    assert {l["variable"] for l in out["lazos"]} >= {"nivel", "presión"}
    assert all(e["regla"] in RULES and e["fuente"] for e in out["instrumentos"] + out["elementos_de_tuberia"])
    path = write(gas_compression(), tmp_path, "gas")
    d = pid_from_hazop(dexpi_path=str(path), out_dir=str(tmp_path / "d"))
    assert d["verified"] and d["missing"] == []
    tags = {i["tag"] for i in d["instrumentos"]}
    assert any(t.startswith("FIC-") for t in tags) and any(t.startswith("TIC-") for t in tags)    # K-100, E-100
    with pytest.raises(ValueError):
        pid_from_hazop()


def test_every_dexpi_element_carries_its_trace(tmp_path):
    g = _dwsim_graph("separador", dict(BUILDERS)["separador"])
    spec = build_pid(g)
    files = DP.save(DP.to_dexpi_pid(spec), tmp_path, "sep")
    pm = DP.load(files["dexpi_pid_xml"]).conceptualModel
    text = lambda o: " ".join(s.value for n in o.referencedNotes for s in n.noteText.singleLanguageStrings)
    for f in pm.processInstrumentationFunctions:
        assert any(r in text(f) for r in RULES) and "Fuente:" in text(f)
    for s in pm.pipingNetworkSystems:
        for seg in s.segments:
            for it in seg.items:
                if type(it).__name__ in ("GateValve", "CheckValve", "GlobeValve", "SpringLoadedGlobeSafetyValve"):
                    assert any(r in text(it) for r in RULES), type(it).__name__
    psv = [i for i in spec.items if i.isa == "PSV"]
    assert psv and all("HAZOP #" in t for p in psv for t in [" ".join(p.hazop)])


def test_llm_rows_become_proposals():
    g = _dwsim_graph("bomba", dict(BUILDERS)["bomba"])
    extra = HazopRow("P-1", "Pump", "Flow", "Reverse flow", ["x"], ["y"], ["Válvula check en descarga"], 2, "", source="proposal")
    spec = build_pid(g, hazop_rows(g) + [extra])
    chk = [i for i in spec.items if i.isa == "CHK"]
    assert len(chk) == 1 and chk[0].origin == "propuesta"
    assert all(i.origin != "propuesta" for i in spec.items if i.isa != "CHK")


def test_dexpi_process_exports_pass_official_xsd(tmp_path):
    """Pending since bloque 15: the DEXPI Process files of export_flowsheet against the official DEXPI 2.0.1 XSD."""
    from packages.export.flowsheet_export import export_all
    from packages.hazop.dexpi_examples import gas_compression, write
    files = export_all(_dwsim_graph("columna_rigurosa", dict(BUILDERS)["columna_rigurosa"]), tmp_path)["files"]
    for xml in (files["dexpi_xml"], write(gas_compression(), tmp_path, "gas")):
        r = DP.validate_xsd(xml)
        if r["status"] == "pendiente":
            pytest.skip("XSD oficial no disponible (sin red)")
        assert r["status"] == "valido", r["errors"]
