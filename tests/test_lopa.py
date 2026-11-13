"""LOPA and SIL determination from the HAZOP and the minimal P&ID (bloque M11).

Evidence:
* published worked examples reproduced number by number:
  - HSE/BSTG (2007), Safety and environmental standards for fuel storage sites, Appendix 1 (tank overfill, three
    initiating events, two IPLs, four conditional modifiers) and its SIL 2 variant;
  - Kenexis LOPA training (P2SAC, Purdue): the «explicit» example and the distillation-column example;
  - Casey (2019), Loss Prevention Bulletin 265, Example 2 (agitator failure with time-at-risk modifiers);
  CCPS (2001) itself was not available legibly: its worked examples are NOT claimed here;
* SIL bands of IEC 61511-1 Tables 4 and 5 tested at their edges;
* independence rules I-1 … I-6: no credit for a layer that shares a component with the cause (P&ID tags);
* end to end on the 10 published HAZOP cases (DEXPI) and on DWSIM flowsheets: HAZOP → P&ID (M10) → LOPA, every tag
  of the LOPA exists in the P&ID written by pid_from_hazop.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import packages.bench.cases_baseline as cb  # noqa: E402
from packages.dwsim_client.verify import verify  # noqa: E402
from packages.hazop.book.cases import CASES as BOOK  # noqa: E402
from packages.hazop.book.cases import write_all  # noqa: E402
from packages.hazop.book.heldout_cases import HELDOUT  # noqa: E402
from packages.hazop.dexpi_graph import graph_from_dexpi  # noqa: E402
from packages.hazop.graph import build_graph  # noqa: E402
from packages.lopa import data as D  # noqa: E402
from packages.lopa.engine import (InitiatingEvent, Layer, Modifier, Scenario, alarm_with_operator_pfd, credit,  # noqa: E402
                                  evaluate, mitigated, sil_for_pfd, sil_for_pfh, size_sif)
from packages.lopa.from_pid import build_scenarios, tags_consistent  # noqa: E402
from packages.lopa.tool import lopa_from_graph, lopa_from_hazop  # noqa: E402
from packages.pid.rules import build_pid  # noqa: E402
from packages.pid.tool import pid_from_hazop  # noqa: E402

DEXPI_CASES = {**BOOK, **HELDOUT}


def ie(cause, f, comps=(), **kw):
    return InitiatingEvent(cause, "usuario", f, "ejemplo publicado", set(comps), **kw)


def layer(tag, pfd, comps=(), kind="usuario", **kw):
    return Layer(tag, kind, tag, pfd, "ejemplo publicado", set(comps) | {tag}, **kw)


# ------------------------------------------------------------------ published worked examples
def _bstg(ipl2_by_event, cm1=0.4):
    """HSE/BSTG (2007) Appendix 1: gasoline tank A overfill, 2-10 fatalities."""
    ie1 = 52 * 0.01 * 0.1                 # ¶18-20: 52 fillings/yr × 1/100 ullage error × 0.1 missed by cross-checks
    ie2 = (52 * 2) * 0.001 * 0.5 * 0.1    # ¶21-24: 104 transfers to B/C × 1/1000 line-up × 0.5 tank high × 0.1
    events = [ie("IE1 ullage incorrecto", ie1, operator=True), ie("IE2 alineación incorrecta", ie2, operator=True),
              ie("IE3 falla del ATG", 0.0006, {"ATG"})]                  # ¶26: FMEA, 0.0006/yr
    # IPL1: ATG + its alarms + operator (¶29, PFD 0.1); IPL2: independent high-level alarm + operator (¶30-33)
    layers = [Layer("IPL1", "alarm_operator", "ATG + alarmas + operador", 0.1, "BSTG ¶29", {"IPL1", "ATG"}, bpcs=True,
                    operator=True, response_time_min=40)]
    layers += [Layer(f"IPL2-{i}", "usuario", "IHLA + operador", p, "BSTG ¶30-33", {"IHLA"}, operator=True,
                     applies_to={i}) for i, p in enumerate(ipl2_by_event)]
    mods = [Modifier("CM1 detección", cm1, "BSTG ¶34"), Modifier("CM2 ignición", 0.3, "BSTG ¶34"),
            Modifier("CM3 ocupación", 0.05, "BSTG ¶34"), Modifier("CM4 fatalidad", 0.8, "BSTG ¶34")]
    return Scenario("BSTG App. 1", "Tank A", "sobrellenado", "explosión, 2-10 fatalidades", None, events, layers, mods,
                    1e-6, "BSTG ¶36 (ejemplo)")


def test_bstg_2007_overfill_example():
    assert 52 * 0.01 * 0.1 == pytest.approx(0.052) and 104 * 0.001 * 0.5 * 0.1 == pytest.approx(0.0052)
    ipl2 = alarm_with_operator_pfd(0.007, 0.01)                  # ¶33: ((1-0.007)*0.01)+0.007
    assert ipl2 == pytest.approx(0.01693) and round(ipl2, 3) == 0.017
    sc = _bstg([0.017] * 3)
    r = evaluate(sc, allow_multiple_operator=True)               # the example credits the operator in both IPLs
    m = r["mitigada"]
    # ¶35: 0.4x0.3x0.05x0.8x{(0.052x0.1x0.017)+(0.0052x0.1x0.017)+(0.0006x0.017)}
    assert m["producto_modificadores"] == pytest.approx(0.0048)
    inter = [x["frecuencia_intermedia"] for x in m["eventos"]]
    assert inter == pytest.approx([0.052 * 0.1 * 0.017, 0.0052 * 0.1 * 0.017, 0.0006 * 0.017])
    assert m["frecuencia_mitigada"] == pytest.approx(5.1568e-7, rel=1e-4)
    # printed «5.1 E-07»: the equation as printed gives 5.157e-7; the unrounded IPL2 (0.01693) gives 5.14e-7 → 5.1
    assert float(f"{evaluate(_bstg([ipl2] * 3), True)['mitigada']['frecuencia_mitigada']:.2g}") == 5.1e-7
    # the table's «-»: no credit for the ATG alarm when the ATG failure is the initiating event (I-1)
    ipl1_ie3 = next(c for c in m["eventos"][2]["capas"] if c["tag"] == "IPL1")
    assert not ipl1_ie3["acreditada"] and "I-1" in ipl1_ie3["razón"] and "ATG" in ipl1_ie3["razón"]
    # ¶36: below 1e-6/yr → broadly acceptable in the example
    assert r["estado"] == "cumple" and r["brecha"] == pytest.approx(0.5157, rel=1e-3)
    assert any("más de una capa" in a for a in r["avisos"])
    # with the default rule (operator credited once, I-3) the same data give a higher, more conservative frequency
    strict = evaluate(sc)
    assert strict["mitigada"]["frecuencia_mitigada"] > m["frecuencia_mitigada"]


def test_bstg_2007_sil2_variant_and_its_table_typo():
    """SIL 2 variant (p. 92): IHLA with ESD, CM1 0.6, printed likelihood «2.3 E-07». It is reproduced with IPL2 = 0.005
    in the three columns; the table prints 0.017 in the IE1 column, which would give 6.8e-7 (not what it prints)."""
    f = evaluate(_bstg([0.005] * 3, cm1=0.6), True)["mitigada"]["frecuencia_mitigada"]
    assert f == pytest.approx(2.2752e-7, rel=1e-9) and float(f"{f:.2g}") == 2.3e-7
    typo = evaluate(_bstg([0.017, 0.005, 0.005], cm1=0.6), True)["mitigada"]["frecuencia_mitigada"]
    assert typo == pytest.approx(6.77e-7, rel=1e-3) and float(f"{typo:.2g}") != 2.3e-7


def test_kenexis_explicit_example():
    """Init. evt. BPCS fails 0.1/yr; IPLs operator 0.1 and (dual) check valve 0.1 → 1.0E-3; TMEL 1.0E-4 → RRR 10."""
    sc = Scenario("Kenexis", "—", "—", "—", None, [ie("BPCS falla", 0.1, {"LIC-1", "LT-1", "LV-1"}, bpcs=True)],
                  [layer("operador", 0.1, kind="alarm_operator", operator=True, response_time_min=20),
                   layer("CHK", 0.1, kind="check_valve_dual")], tolerable=1e-4)
    r = evaluate(sc)
    assert r["mitigada"]["frecuencia_mitigada"] == pytest.approx(1e-3) and r["brecha"] == pytest.approx(10.0)
    assert D.IPLS["check_valve_dual"][1] == 0.1 and D.IPLS["alarm_operator"][1] == 0.1
    s = sil_for_pfd(1 / r["brecha"])
    assert s["sil"] == 0 and s["rrf"] == pytest.approx(10.0)          # RRF 10 does not need a SIL-rated SIF


def test_kenexis_distillation_column_example():
    """Column CL-101 overpressure: PC-05 malfunction drives PV-04 closed (0.1/yr); BPCS interlock 0 IPL, operator
    response 0 IPL, pressure relief 2 IPL (RRF 100) → 1/1,000; TMEL 1.0E-5 → required risk reduction 100. The slide's
    reasons for the zero credits are not legible: here the interlock shares the BPCS controller with PC-05 and the
    operator has no response time given. Its «Req. SIL» cell is not legible either; the IEC band of RRF 100 is SIL 1."""
    cause = ie("PC-05 falla y cierra PV-04", 0.1, {"PC-05", "PT-03", "PV-04", "controlador-BPCS"}, bpcs=True)
    sif = Layer("SIF-01", "sif", "PAHH 2oo3 cierra XV-01/XV-02", None, "slide", {"SIF-01", "PT-01", "PT-02", "XV-01"}, sif=True)
    sc = Scenario("Kenexis CL-101", "CL-101", "más presión", "ruptura", None, [cause],
                  [layer("interlock BPCS", 0.1, {"controlador-BPCS"}, kind="bpcs_loop", bpcs=True),
                   layer("operador", 0.1, kind="alarm_operator", operator=True),
                   layer("PSV", 0.01, kind="psv"), sif], tolerable=1e-5)
    r = evaluate(sc)
    caps = {c["tag"]: c for c in r["mitigada"]["eventos"][0]["capas"]}
    assert not caps["interlock BPCS"]["acreditada"] and "I-1" in caps["interlock BPCS"]["razón"]
    assert not caps["operador"]["acreditada"] and "I-5" in caps["operador"]["razón"]
    assert r["mitigada"]["frecuencia_mitigada"] == pytest.approx(1e-3)
    s = r["sif"][0]
    assert s["rrf_requerido"] == pytest.approx(100.0) and s["pfd_requerida"] == pytest.approx(0.01)
    assert s["sil"] == 1 and s["modo"] == "demanda baja"


def test_casey_2019_agitator_example():
    """LPB 265 Example 2: 0.1/yr (agitator failure, a BPCS failure) × 0.25 × 0.285 × 0.01 (disk) = 7.125e-5/yr; without the
    two time-at-risk modifiers, 1e-3/yr. The engine flags them, as Casey argues, because the failure is latent."""
    mods = [Modifier("fracción del año con el producto", 0.25, "Casey 2019"),
            Modifier("tiempo de reacción / tiempo de lote", 0.285, "Casey 2019")]
    sc = Scenario("Casey ej. 2", "reactor", "sin agitación", "runaway", None,
                  [ie("falla del agitador", 0.1, {"agitador"}, bpcs=True)], [layer("disco", 0.01, kind="rupture_disc")], mods)
    r = evaluate(sc)
    assert r["mitigada"]["frecuencia_mitigada"] == pytest.approx(7.125e-5)
    assert len(r["avisos"]) == 1 and "tiempo en riesgo" in r["avisos"][0]
    sc.modifiers = []
    assert mitigated(sc)["frecuencia_mitigada"] == pytest.approx(1e-3)


# ------------------------------------------------------------------ SIL bands at their edges
@pytest.mark.parametrize("pfd,sil", [
    (1.0, None), (0.5, 0), (0.1, 0), (math.nextafter(0.1, 0), 1), (0.05, 1), (0.01, 1), (math.nextafter(0.01, 0), 2),
    (1e-3, 2), (math.nextafter(1e-3, 0), 3), (1e-4, 3), (math.nextafter(1e-4, 0), 4), (1e-5, 4)])
def test_sil_bands_low_demand(pfd, sil):
    if pfd in (math.nextafter(0.1, 0), math.nextafter(0.01, 0), math.nextafter(1e-3, 0), math.nextafter(1e-4, 0)):
        pfd = pfd * (1 - 1e-9)                   # beyond the 12-digit cleaning: really below the edge
    assert sil_for_pfd(pfd)["sil"] == sil


def test_sil_beyond_4_and_rrf():
    s = sil_for_pfd(1e-5 * (1 - 1e-9))
    assert s["sil"] is None and not s["alcanzable"]
    assert sil_for_pfd(0.01)["rrf"] == pytest.approx(100.0) and sil_for_pfd(1e-4)["sil"] == 3
    assert "evitar" in sil_for_pfd(5e-5)["texto"]
    # float noise does not move a band: 1e-4 / (0.1 × 0.1) is RRF 100, SIL 1, not SIL 2
    assert 1e-4 / (0.1 * 0.1) < 0.01 and sil_for_pfd(1e-4 / (0.1 * 0.1))["sil"] == 1


@pytest.mark.parametrize("pfh,sil", [(1e-5, 0), (1e-5 * (1 - 1e-9), 1), (1e-6, 1), (1e-6 * (1 - 1e-9), 2), (1e-7, 2),
                                     (1e-8, 3), (1e-8 * (1 - 1e-9), 4), (1e-9, 4), (1e-9 * (1 - 1e-9), None)])
def test_sil_bands_high_demand(pfh, sil):
    assert sil_for_pfh(pfh)["sil"] == sil


def test_high_demand_mode_uses_pfh():
    sif = Layer("SIF", "sif", "trip", None, "—", {"SIF"}, sif=True)
    sc = Scenario("hd", "n", "d", "c", None, [ie("causa frecuente", 2.0, {"x"})], [sif], tolerable=1e-4)
    s = size_sif(sc, sif)
    assert s["modo"] == "demanda alta" and s["pfh_requerida_por_hora"] == pytest.approx(1e-4 / 8760)
    assert s["sil"] == 3                         # 1e-4/8760 = 1.14e-8 per hour: SIL 3 band [1e-8, 1e-7)


# ------------------------------------------------------------------ independence rules
def test_i1_shared_component_gets_no_credit():
    cause = ie("falla del lazo", 0.1, {"LIC-101", "LT-101", "LV-101", "lazo:101"}, bpcs=True)
    for shared in ("LT-101", "LV-101", "lazo:101"):
        ok, why = credit(cause, 0, layer("X", 0.1, {shared}), [])
        assert not ok and shared in why and why.startswith("I-1")
    assert credit(cause, 0, layer("X", 0.1, {"LT-901"}), [])[0]


def test_i2_one_bpcs_layer_when_cause_is_bpcs():
    cause = ie("falla del lazo", 0.1, {"LIC-101"}, bpcs=True)
    a, b = layer("LAH-102", 0.1, bpcs=True), layer("TIC-103", 0.1, bpcs=True)
    assert credit(cause, 0, a, [])[0]
    ok, why = credit(cause, 0, b, [a])
    assert not ok and why.startswith("I-2")
    assert credit(ie("fuego", 0.1, {"fuego"}), 0, b, [a])[0]          # not a BPCS cause: both may count


def test_i3_operator_once_and_i4_sif_once_and_i5_time_and_i6_trace():
    cause = ie("x", 0.1, {"a"})
    op1 = layer("op1", 0.1, operator=True)
    assert not credit(cause, 0, layer("op2", 0.1, operator=True), [op1])[0]
    assert credit(cause, 0, layer("op2", 0.1, operator=True), [op1], allow_multiple_operator=True)[0]
    s = Layer("SIF-1", "sif", "s", 0.01, "—", {"SIF-1"}, sif=True)
    assert not credit(cause, 0, s, [s])[0]
    for t, ok in ((None, False), (9.9, False), (10.0, True), (40.0, True)):
        assert credit(cause, 0, layer("LAH", 0.1, kind="alarm_operator", response_time_min=t), [])[0] is ok
    assert not credit(cause, 0, layer("PSV", 0.01, traced=False), [])[0]
    assert not credit(cause, 0, layer("CHK", None, kind="check_valve_single"), [])[0]


@pytest.mark.parametrize("cid", list(DEXPI_CASES))
def test_independence_on_the_pid_of_published_hazops(cid):
    """On every published HAZOP: no credited layer shares a component with its cause; an alarm on the transmitter of
    the failed BPCS loop is never credited; a trip (independent sensor) is never blocked by I-1."""
    g, _ = graph_from_dexpi(DEXPI_CASES[cid](), cid)
    spec = build_pid(g)
    scs = build_scenarios(spec, tolerable=1e-5, alarm_response_min=15)
    seen_block = 0
    for sc in scs:
        m = mitigated(sc)
        for e, row in zip(sc.events, m["eventos"]):
            for c in row["capas"]:
                l = next(x for x in sc.layers if x.tag == c["tag"])
                if c["acreditada"]:
                    assert not (l.components & e.components), (sc.ref, e.cause, l.tag)
                if l.kind == "alarm_operator" and e.kind == "bpcs_loop" and e.tags and \
                        next(i for i in spec.instruments if i.tag == l.tag).sensor in e.tags:
                    assert not c["acreditada"] and c["razón"].startswith("I-1")
                    seen_block += 1
                if l.sif:
                    assert not c["razón"].startswith("I-1"), (sc.ref, l.tag)
    assert all(e.kind != "bpcs_loop" or e.tags or "no tiene un lazo" in e.note for sc in scs for e in sc.events)
    if cid in ("ccps_dap", "marhavilas_estabilizadora"):
        assert seen_block > 0


def test_same_loop_twice_is_one_initiating_event():
    g, _ = graph_from_dexpi(BOOK["ccps_dap"](), "ccps_dap")
    sc = next(s for s in build_scenarios(build_pid(g)) if "more cooling" in s.deviation)
    bpcs = [e for e in sc.events if e.kind == "bpcs_loop"]
    assert len(bpcs) == 1 and " / " in bpcs[0].cause and bpcs[0].tags[0] == "TIC-104"


def test_operator_error_needs_opportunities_from_the_user():
    g, _ = graph_from_dexpi(BOOK["ccps_dap"](), "ccps_dap")
    spec = build_pid(g)
    scs = build_scenarios(spec)
    sc = next(s for s in scs if any(e.kind == "operator_error" for e in s.events))
    n = int(sc.ref.split("#")[1].split(":")[0])
    k = next(i for i, e in enumerate(sc.events, 1) if e.kind == "operator_error")
    assert sc.events[k - 1].frequency is None
    sc2 = next(s for s in build_scenarios(spec, overrides={n: {"oportunidades_por_año": {f"{n}.{k}": 12}}}) if s.ref == sc.ref)
    assert sc2.events[k - 1].frequency == pytest.approx(0.12)


# ------------------------------------------------------------------ tolerable frequency is never silent
def test_no_criterion_no_gap_no_sil_and_examples_are_labeled():
    g, _ = graph_from_dexpi(BOOK["ccps_dap"](), "ccps_dap")
    o = lopa_from_graph(g)
    assert o["falta_criterio"] and all(r["estado"] == "sin criterio" for r in o["hoja_lopa"])
    assert all(s.get("sil") is None for s in o["sif"]) and "bstg_2007" in o["criterios_de_ejemplo"]
    e = lopa_from_graph(g, example_criterion="bstg_2007", alarm_response_min=15)
    assert "falta_criterio" not in e and all("EJEMPLO" in r["origen_tolerable"] for r in e["hoja_lopa"])
    assert all(v["descripción"].startswith("EJEMPLO") for v in D.EXAMPLE_CRITERIA.values())
    with pytest.raises(ValueError):
        lopa_from_graph(g, example_criterion="ccps")
    by_sev = lopa_from_graph(g, tolerable_by_severity={4: 1e-5}, alarm_response_min=15)
    assert {r["frecuencia_tolerable_por_año"] for r in by_sev["hoja_lopa"] if r["severidad_hazop"] == 4} == {1e-5}
    assert all(r["frecuencia_tolerable_por_año"] is None for r in by_sev["hoja_lopa"] if r["severidad_hazop"] != 4)


def test_modifiers_only_from_user_and_scoped():
    g, _ = graph_from_dexpi(BOOK["ccps_dap"](), "ccps_dap")
    spec = build_pid(g)
    assert all(not s.modifiers for s in build_scenarios(spec))
    scs = build_scenarios(spec, modifiers=[{"nombre": "ignición", "probabilidad": 0.3, "fuente": "usuario", "nodos": ["R-1"]}])
    assert all((s.node == "R-1") == bool(s.modifiers) for s in scs)


def test_tables_cite_a_source_and_are_generic():
    for k, (desc, val, unit, src) in D.INITIATING_EVENTS.items():
        assert desc and val > 0 and unit in (D.PER_YEAR, D.PER_OPPORTUNITY) and len(src) > 20, k
    for k, (desc, pfd, cond, src) in D.IPLS.items():
        assert desc and cond and len(src) > 20 and (pfd is None or 0 < pfd < 1), k
    assert D.INITIATING_EVENTS["bpcs_loop"][1] == 0.1 and D.IPLS["psv"][1] == 0.01 and D.IPLS["bpcs_loop"][1] == 0.1


# ------------------------------------------------------------------ end to end: HAZOP → P&ID → LOPA
def _pid_tags(pid_out: dict) -> set[str]:
    return ({i["tag"] for i in pid_out["instrumentos"]} | {i["transmisor"] for i in pid_out["instrumentos"]}
            | {e["tag"] for e in pid_out["elementos_de_tuberia"]})


def _lopa_tags(o: dict) -> set[str]:
    t = {s["sif"] for s in o["sif"]}
    for r in o["hoja_lopa"]:
        t |= {c["tag"] for c in r["capas"]}
        t |= {x for e in r["eventos_iniciadores"] for x in e["tags"]}
    return t


def test_end_to_end_published_hazops_through_dexpi_pid(tmp_path):
    files = write_all(tmp_path / "dexpi", DEXPI_CASES)
    for cid, path in files.items():
        pid = pid_from_hazop(dexpi_path=str(path), out_dir=str(tmp_path / cid))
        assert pid["verified"], (cid, pid["evidence"])
        o = lopa_from_hazop(dexpi_path=str(path), example_criterion="kenexis_tmel_1e-5", alarm_response_min=15)
        assert o["trazabilidad_tags"]["ok"], (cid, o["trazabilidad_tags"])
        units = {i["equipo"] for i in pid["instrumentos"]}
        lt = _lopa_tags(o)
        assert lt - units <= _pid_tags(pid), (cid, sorted(lt - units - _pid_tags(pid)))
        trips = {i["tag"] for i in pid["instrumentos"] if i["tipo"] == "disparo (SIS)"}
        assert {s["sif"] for s in o["sif"]} == trips, cid             # every trip of the P&ID gets its SIL here
        assert all(s.get("pfd_requerida") is None or 0 < s["pfd_requerida"] for s in o["sif"])
        assert o["conteo"]["escenarios"] == len(o["hoja_lopa"]) > 0


@pytest.mark.parametrize("cid,build", [("compresor_enfriador", cb.compresor_enfriador),
                                       ("depropanizadora_fug", cb.depropanizadora_fug)])
def test_end_to_end_on_dwsim_flowsheets(cid, build):
    out = build()
    fs = out[0] if isinstance(out, tuple) else out
    verify(fs)
    g = build_graph(fs, cid)
    spec = build_pid(g)
    o = lopa_from_graph(g, tolerable_frequency=1e-5, alarm_response_min=15)
    scs = build_scenarios(spec, tolerable=1e-5, alarm_response_min=15)
    assert tags_consistent(scs, spec)["ok"] and o["trazabilidad_tags"]["ok"]
    assert o["hoja_lopa"] and all(r["origen_tolerable"] == "usuario" for r in o["hoja_lopa"])
    assert any(e["tipo"] != "fuera de la tabla" for r in o["hoja_lopa"] for e in r["eventos_iniciadores"])
