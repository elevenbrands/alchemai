"""design_heuristics_review with seeded design errors: it must flag them and leave sound designs alone.

The catalogue (packages/review/heuristics_catalog.json) is checked for structure and provenance: every rule has a
source URL and either a range or a typical value. Seeded flowsheets: a single compressor with ratio 10 (usual ≤ 4 per
stage, discharge far above 204 °C), a counter-current exchanger with a 2 K approach, a cooler to 25 °C (below what
cooling water reaches), against a two-stage intercooled train and a 20 K approach that must pass.
"""
import pytest

from packages.dwsim_client import tools as T
from packages.dwsim_client import tools_review as TR
from packages.review import heuristics as H


def test_catalogue_structure_and_provenance():
    rules = H.catalog()
    assert len(rules) >= 100 and len({r["id"] for r in rules}) == len(rules)
    for r in rules:
        assert r["source_consulted"]["url"].startswith("http"), r["id"]
        assert any(r.get(k) is not None for k in ("low", "high", "typical")), r["id"]
        if r.get("low") is not None and r.get("high") is not None:
            assert r["low"] <= r["high"], r["id"]
    listed = TR.design_heuristics_review(list_areas=["distillation"])
    assert listed["ok"] and listed["rules"] and all(r["area"] == "distillation" for r in listed["rules"])


def test_explicit_checks_flag_and_pass():
    out = TR.design_heuristics_review(checks=[
        {"rule_id": "dist.reflux_ratio.range", "value": 1.05, "label": "columna casi en reflujo mínimo"},
        {"rule_id": "dist.reflux_ratio.range", "value": 1.3},
        {"rule_id": "pump.npsh_margin.doe", "value": 1.05},
        {"rule_id": "vessel.drum_LD", "value": 3.0},
        {"rule_id": "dist.stages_vs_min", "value": 3.5}])
    st = {(f["rule_id"], f["value"]): f for f in out["findings"]}
    assert st[("dist.reflux_ratio.range", 1.05)]["status"] == "below_usual"
    assert st[("dist.reflux_ratio.range", 1.3)]["status"] == "within_usual"
    assert st[("pump.npsh_margin.doe", 1.05)]["severity"] == "warning"
    assert st[("vessel.drum_LD", 3.0)]["severity"] == "ok"
    assert st[("dist.stages_vs_min", 3.5)]["status"] == "differs_from_typical"     # typical only: info, not warning
    assert out["n_outside_usual_practice"] == 2
    assert out["findings"][0]["severity"] == "warning" and out["findings"][0]["source"]["url"]
    assert "error" in TR.design_heuristics_review(checks=[{"rule_id": "dist.nonsense", "value": 1}])
    assert "error" in TR.design_heuristics_review()


def _compressor_train(ratios):
    fid = T.new_flowsheet("comp", ["Nitrogen"], "PengRobinson1978")["flowsheet_id"]
    T.add_material_stream(fid, "s0", temperature_K=300.0, pressure_Pa=1e5, molar_flow_mol_s=10.0, mole_fractions={"Nitrogen": 1.0})
    P, inlet = 1e5, "s0"
    for i, r in enumerate(ratios):
        P *= r
        out = f"c{i}"
        T.add_material_stream(fid, out)
        T.add_energy_stream(fid, f"W{i}")
        T.add_unit(fid, "compressor", f"K{i}", {"outlet_P_Pa": P, "adiabatic_eff_pct": 75.0})
        T.connect(fid, f"K{i}", [inlet], [out], [f"W{i}"], None)
        inlet = out
        if i < len(ratios) - 1:
            cooled = f"ic{i}"
            T.add_material_stream(fid, cooled)
            T.add_energy_stream(fid, f"Q{i}")
            T.add_unit(fid, "cooler", f"IC{i}", {"outlet_T_K": 313.15})
            T.connect(fid, f"IC{i}", [inlet], [cooled], None, [f"Q{i}"])
            inlet = cooled
    assert T.solve(fid)["solved"]
    return fid


def test_flowsheet_single_stage_ratio_10_is_flagged_two_stage_is_not():
    bad = TR.design_heuristics_review(flowsheet_id=_compressor_train([10.0]))
    f = {x["rule_id"]: x for x in bad["findings"] if x.get("item", "").startswith("K0")}
    assert f["comp.ratio_per_stage"]["status"] == "above_usual" and f["comp.ratio_per_stage"]["value"] == pytest.approx(10.0)
    assert f["comp.max_discharge_temp"]["status"] == "above_usual"
    good = TR.design_heuristics_review(flowsheet_id=_compressor_train([3.2, 3.125]))
    comp = [x for x in good["findings"] if x["rule_id"].startswith("comp.")]
    assert comp and all(x["severity"] == "ok" for x in comp), comp
    coolers = [x for x in good["findings"] if x["rule_id"] == "util.cw.min_process_temp.towler"]
    assert coolers and all(x["severity"] == "ok" for x in coolers)      # 40 °C after intercooling: reachable with water


def _exchanger(UA_W_K, cooler_out_K=None):
    """Equal water flows (C ≈ 4.2 kW/K) in counter-current: ε = NTU/(1 + NTU), approach = 60 K·(1 − ε)."""
    fid = T.new_flowsheet("hx", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "h_in", temperature_K=363.15, pressure_Pa=3e5, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "c_in", temperature_K=303.15, pressure_Pa=3e5, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    for n in ("h_out", "c_out"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "heat_exchanger", "E", {"UA_W_K": UA_W_K})
    T.connect(fid, "E", ["h_in", "c_in"], ["h_out", "c_out"], None, None)
    if cooler_out_K:
        T.add_material_stream(fid, "h_cold")
        T.add_energy_stream(fid, "Qc")
        T.add_unit(fid, "cooler", "C", {"outlet_T_K": cooler_out_K})
        T.connect(fid, "C", ["h_out"], ["h_cold"], None, ["Qc"])
    sol = T.solve(fid)
    return fid, sol


def test_exchanger_approach_and_cooler_below_cooling_water():
    fid, sol = _exchanger(29 * 4190.0, cooler_out_K=298.15)    # NTU 29: ~2 K approach
    if not sol.get("solved"):
        pytest.skip(f"DWSIM no resolvió el intercambiador con esta especificación: {sol.get('solver_errors')}")
    out = TR.design_heuristics_review(flowsheet_id=fid)
    hx = [x for x in out["findings"] if x["rule_id"].startswith("hx.min_approach")]
    assert hx and all(x["severity"] == "warning" for x in hx) and min(x["value"] for x in hx) == pytest.approx(2.0, abs=0.5)
    cw = [x for x in out["findings"] if x["rule_id"] == "util.cw.min_process_temp.towler"]
    assert cw and cw[0]["status"] == "below_usual"
    fid2, sol2 = _exchanger(2 * 4190.0)                         # NTU 2: ~20 K approach
    out2 = TR.design_heuristics_review(flowsheet_id=fid2)
    hx2 = [x for x in out2["findings"] if x["rule_id"].startswith("hx.min_approach")]
    assert hx2 and all(x["severity"] == "ok" for x in hx2), hx2
