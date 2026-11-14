"""M15c: excess enthalpy (h^E) of ethanol-water against calorimetry, the joint VLE + h^E regression and the h^E
provenance label of the NRTL equation-oriented column.

Measured h^E: tests/data/he_ethanol_water_thermoml.json (Nagamachi y Francesconi, J. Chem. Thermodyn. 38 (2006) 461;
Brandt, Horstmann, Steinigeweg y Gmehling, Fluid Phase Equilib. 376 (2014) 48), numbers read by script from the NIST
ThermoML files. VLE: Kamihama 2012 (tests/data/vle_ethanol_water_vlizard.json).

Fast: h^E of the three models vs thermo, the tool unchanged without h^E sets, synthetic recovery of a and b from noisy
VLE + h^E, weights and refusals, the measured-vs-model facts, the honest verdict of the joint fit on real data, the
provenance labels. Slow (ALCHEMAI_SLOW_EO_NRTL=1): a joint a + b/T set through eo_column_optimize and DWSIM «ExpData».
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from packages.eo import column_optimize as C
from packages.eo import nrtl_properties as NP
from packages.vle import models as M
from packages.vle.equilibrium import Psat, System, VaporPhase
from packages.vle.tool import JOINT_TOLERANCES, vle_binary_regression

DATA = Path(__file__).parent / "data"
EW = ["ethanol", "water"]
slow = pytest.mark.skipif(os.environ.get("ALCHEMAI_SLOW_EO_NRTL") != "1" or not C.ipopt_available(),
                          reason="slow M15 reference/DWSIM tests: ALCHEMAI_SLOW_EO_NRTL=1")
_CACHE: dict = {}


def _vle():
    d = json.loads((DATA / "vle_ethanol_water_vlizard.json").read_text())
    k = next(k for k in d["sets"] if "101" in k)
    return d["components"], [{"label": k, "points": d["sets"][k], "citation": d["citation"]}]


def _he(temps=None):
    d = json.loads((DATA / "he_ethanol_water_thermoml.json").read_text())
    return [{"label": lab, "T_K": s["T_K"], "citation": d["sources"][s["source"]]["citation"],
             "points": [{"x1": q["x1"], "hE_J_mol": q["hE_J_mol"]} for q in s["points"]]}
            for lab, s in d["sets"].items() if temps is None or s["T_K"] in temps]


def _fit(key, **kw):
    if key not in _CACHE:
        comps, vle = _vle()
        _CACHE[key] = vle_binary_regression(comps, vle, models=["NRTL"], compare_unifac=False, **kw)
        assert _CACHE[key]["ok"], _CACHE[key]
    return _CACHE[key]


def _vle_only():
    return _fit("vle_only")


def _joint_all():
    return _fit("joint_all", temperature_dependence="a+b/T", excess_enthalpy_sets=_he(), he_weight_scan=[0, 0.01, 1, 100])


def _joint_363():
    return _fit("joint_363", temperature_dependence="a+b/T", excess_enthalpy_sets=_he([363.15]))


# ------------------------------------------------------------------------------------------------ models
@pytest.mark.parametrize("model,extra", [("NRTL", {"alpha": 0.37}), ("UNIQUAC", {"r": (2.1055, 0.92), "q": (1.972, 1.4)}),
                                         ("Wilson", {"V": (58.6e-6, 18.07e-6)})])
def test_excess_enthalpy_matches_thermo(model, extra):
    """Analytic h^E = −R·T²·∂(g^E/RT)/∂T vs thermo's GibbsExcess.HE() (independent code), with a_ij ≠ 0."""
    rng = np.random.default_rng(1)
    for _ in range(25):
        p = {"a12": rng.uniform(-2, 2), "a21": rng.uniform(-2, 2), "b12": rng.uniform(-800, 800),
             "b21": rng.uniform(-800, 800), **extra}
        T, x = rng.uniform(290, 390), rng.uniform(0.02, 0.98)
        ref = M.thermo_excess_enthalpy(model, T, x, p)
        assert float(M.excess_enthalpy(model, T, x, p)) == pytest.approx(ref, rel=1e-8, abs=1e-6)
    assert float(M.excess_enthalpy(model, 350.0, 0.3, {"b12": 0.0, "b21": 0.0, **extra})) == 0.0


# ------------------------------------------------------------------------------------------------ backward compatibility
def test_without_hE_sets_the_tool_is_unchanged():
    """No h^E keys in the output, the committed M15 parameters to the last digit (notebooks/25_eo_nrtl.json), and the
    h^E-only arguments are refused when there are no h^E sets."""
    r = _vle_only()
    m = r["modelos"]["NRTL"]
    assert not any("hE" in k for k in (*r, *m, *r["objetivo"]))
    vals = {q["nombre"]: q["valor"] for q in m["parametros"]}
    assert vals["b12"] == pytest.approx(-51.26244374049498, rel=1e-12)
    assert vals["b21"] == pytest.approx(687.4526578258977, rel=1e-12)
    comps, vle = _vle()
    for kw in ({"weights": {"hE": 2.0}}, {"he_weight_scan": [1.0]}, {"joint_tolerances": {"AAD_y": 0.02}}):
        bad = vle_binary_regression(comps, vle, models=["NRTL"], **kw)
        assert not bad["ok"] and "excess_enthalpy_sets" in bad["error"]


# ------------------------------------------------------------------------------------------------ synthetic recovery
SYN_TRUE = {"a12": 0.8, "a21": -1.2, "b12": -350.0, "b21": 1100.0, "alpha": 0.3}


def _synthetic(seed):
    """Isobaric T-x-y at 101.325 kPa and h^E at three temperatures from known a + b/T NRTL parameters, plus noise
    (σ_T 0.1 K, σ_y 0.005, σ_hE 5 J/mol + 1 %: the tool's labelled defaults)."""
    S = System((Psat("ethanol", "64-17-5", None), Psat("water", "7732-18-5", None)), VaporPhase("ideal", ("", "")))
    rng = np.random.default_rng(seed)
    x = np.linspace(0.03, 0.97, 24)
    T, y = S.bubble_T(np.full_like(x, 101325.0), x, lambda T_, x_: M.ln_gamma("NRTL", T_, x_, SYN_TRUE), T0=np.full_like(x, 355.0))
    vle = [{"T_K": float(t + rng.normal(0, 0.1)), "P_Pa": 101325.0, "x1": float(a),
            "y1": float(np.clip(b + rng.normal(0, 0.005), 1e-4, 1 - 1e-4))} for t, a, b in zip(T, x, y)]
    he = []
    for Th in (313.15, 343.15, 373.15):
        xs = np.linspace(0.05, 0.95, 12)
        h = M.excess_enthalpy("NRTL", np.full_like(xs, Th), xs, SYN_TRUE)
        he.append({"label": f"sint_{Th}", "T_K": Th,
                   "points": [{"x1": float(a), "hE_J_mol": float(b + rng.normal(0, 5 + 0.01 * abs(b)))} for a, b in zip(xs, h)]})
    return vle, he


def test_synthetic_joint_fit_recovers_a_and_b():
    """12 noisy replicates: every true a and b inside its 95 % interval in ≥ 9 of them (expected 95 %; binomial 12
    trials gives P(≤ 8) ≈ 0.02 %), the joint verdict is «representa», and all starts reach the same optimum."""
    hits = dict.fromkeys(("a12", "a21", "b12", "b21"), 0)
    N = 12
    for seed in range(N):
        vle, he = _synthetic(seed)
        r = vle_binary_regression(EW, [{"label": "sint", "points": vle}], models=["NRTL"], temperature_dependence="a+b/T",
                                  nrtl_alpha=0.3, compare_unifac=False, excess_enthalpy_sets=he)
        assert r["ok"], r
        m = r["modelos"]["NRTL"]
        for q in m["parametros"]:
            if q["nombre"] in hits:
                hits[q["nombre"]] += abs(q["valor"] - SYN_TRUE[q["nombre"]]) <= q["ic95"]
        assert m["veredicto_conjunto"]["representa_VLE_y_hE"], m["veredicto_conjunto"]
        costs = m["arranques_conjuntos"]["costs"]
        assert max(costs) - min(costs) < 1e-6 * max(costs)
        assert m["verificacion_thermo"]["max_dif_hE_J_mol"] < 1e-5
    assert all(h >= 9 for h in hits.values()), hits


def test_hE_data_is_what_identifies_b():
    """Same synthetic set: with VLE alone on one isobar the interval of b is ≥ 20 times wider than with h^E."""
    vle, he = _synthetic(0)
    kw = dict(models=["NRTL"], temperature_dependence="a+b/T", nrtl_alpha=0.3, compare_unifac=False)
    alone = vle_binary_regression(EW, [{"label": "sint", "points": vle}], **kw)
    joint = vle_binary_regression(EW, [{"label": "sint", "points": vle}], excess_enthalpy_sets=he, **kw)
    ci = lambda r: {q["nombre"]: q["ic95"] for q in r["modelos"]["NRTL"]["parametros"]}  # noqa: E731
    for b in ("b12", "b21"):
        assert ci(alone)[b] > 20 * ci(joint)[b]
        assert ci(joint)[b] < 20.0


def test_maximum_likelihood_objective_takes_the_hE_block():
    """objective="ml" (errors in T, x, P, y of the VLE points) with the calorimetric block: converges, recovers the
    parameters and agrees with the bubble-point objective within the intervals."""
    vle, he = _synthetic(1)
    kw = dict(models=["NRTL"], temperature_dependence="a+b/T", nrtl_alpha=0.3, compare_unifac=False, excess_enthalpy_sets=he)
    ml = vle_binary_regression(EW, [{"label": "sint", "points": vle}], objective="ml", **kw)
    bub = vle_binary_regression(EW, [{"label": "sint", "points": vle}], **kw)
    m, b = ml["modelos"]["NRTL"], bub["modelos"]["NRTL"]
    assert m["convergencia"]["exito"], m["convergencia"]
    assert "bloque calorimétrico" in ml["objetivo"]["descripcion"]
    for q, qb in zip(m["parametros"], b["parametros"]):
        assert abs(q["valor"] - SYN_TRUE[q["nombre"]]) <= q["ic95"]
        assert abs(q["valor"] - qb["valor"]) <= q["ic95"]


def test_weights_sigmas_and_refusals():
    vle, he = _synthetic(3)
    sets = [{"label": "sint", "points": vle}]
    kw = dict(models=["NRTL"], temperature_dependence="a+b/T", nrtl_alpha=0.3, compare_unifac=False)
    r = vle_binary_regression(EW, sets, excess_enthalpy_sets=he, **kw)
    assert r["objetivo"]["pesos"] == {"vle": 1.0, "hE": 1.0} and "S = w_vle·S_vle + w_hE·S_hE" in r["objetivo"]["descripcion"]
    assert r["objetivo"]["tolerancias_veredicto"] == JOINT_TOLERANCES
    sup = " ".join(r["supuestos"])
    assert "σ de h^E por defecto" in sup and "pesos de los bloques VLE y h^E = 1 y 1" in sup and "tolerancias del veredicto" in sup
    assert [c["label"] for c in r["conjuntos_hE"]] == [s["label"] for s in he]
    m = r["modelos"]["NRTL"]
    assert len(m["residuos_hE"]) == 36 and len(m["estadistica_hE"]["por_conjunto_y_T"]) == 3
    # weight 0 on h^E: the VLE-only parameters, and h^E still compared (predicted, not fitted)
    w0 = vle_binary_regression(EW, sets, excess_enthalpy_sets=he, weights={"vle": 1.0, "hE": 0.0}, **kw)
    alone = vle_binary_regression(EW, sets, **kw)
    v0 = {q["nombre"]: q["valor"] for q in w0["modelos"]["NRTL"]["parametros"]}
    va = {q["nombre"]: q["valor"] for q in alone["modelos"]["NRTL"]["parametros"]}
    assert v0 == pytest.approx(va, rel=1e-9) and "estadistica_hE" in w0["modelos"]["NRTL"]
    # the user's σ (point > set > global) silence the default note; the user's weights and tolerances silence theirs
    he_u = [{**s, "sigma_hE_J_mol": 8.0} for s in he]
    u = vle_binary_regression(EW, sets, excess_enthalpy_sets=he_u, weights={"vle": 1.0, "hE": 0.5},
                              joint_tolerances={"AAD_T_K": 0.3}, **kw)
    su = " ".join(u["supuestos"])
    assert "σ de h^E por defecto" not in su and "pesos de los bloques" not in su and "tolerancias del veredicto" not in su
    assert all(q["sigma_J_mol"] == 8.0 for q in u["modelos"]["NRTL"]["residuos_hE"])
    assert u["objetivo"]["tolerancias_veredicto"]["AAD_T_K"] == 0.3
    # refusals: missing h^E value or temperature, unknown weight key, negative weight
    no_h = [{"label": "x", "T_K": 320.0, "points": [{"x1": 0.3}]}]
    no_T = [{"label": "x", "points": [{"x1": 0.3, "hE_J_mol": -100.0}]}]
    for bad_sets, word in ((no_h, "hE_J_mol"), (no_T, "T_K")):
        bad = vle_binary_regression(EW, sets, excess_enthalpy_sets=bad_sets, **kw)
        assert not bad["ok"] and word in bad["error"] and "no se suponen" in bad["error"]
    assert not vle_binary_regression(EW, sets, excess_enthalpy_sets=he, weights={"he": 1.0}, **kw)["ok"]
    assert not vle_binary_regression(EW, sets, excess_enthalpy_sets=he, weights={"hE": -1.0}, **kw)["ok"]
    # tau = b/T with h^E is allowed, with a warning
    c = vle_binary_regression(EW, sets, excess_enthalpy_sets=he, models=["NRTL"], nrtl_alpha=0.3, compare_unifac=False)
    assert c["ok"] and any("constante" in a for a in c["avisos"])


# ------------------------------------------------------------------------------------------------ real data
def test_measured_hE_file_is_cited_and_says_what_calorimetry_says():
    d = json.loads((DATA / "he_ethanol_water_thermoml.json").read_text())
    assert "10.1016/j.jct.2005.06.018" in d["sources"]["nagamachi_2006"]["citation"]
    assert "10.1016/j.fluid.2014.05.031" in d["sources"]["brandt_2014"]["citation"]
    assert "sin teclear ni leer de gráficas" in d["how_obtained"]
    n = {s["T_K"]: len(s["points"]) for s in d["sets"].values()}
    assert n == {298.15: 12, 323.15: 15, 333.15: 22, 363.15: 21, 383.15: 13}
    h = {s["T_K"]: np.array([q["hE_J_mol"] for q in s["points"]]) for s in d["sets"].values()}
    x = {s["T_K"]: np.array([q["x1"] for q in s["points"]]) for s in d["sets"].values()}
    # exothermic over the whole range at 298.15 and 323.15 K (minimum −770 J/mol at x = 0.15), endothermic at 383.15 K,
    # and S-shaped in between: the sign changes with temperature and with composition
    assert h[298.15].max() < 0 and h[323.15].max() < 0 and h[383.15].min() > 0
    assert h[298.15].min() == -770.0 and x[298.15][h[298.15].argmin()] == 0.15
    assert h[363.15][x[363.15] < 0.12].max() < 0 < h[363.15][x[363.15] > 0.15].min()


def test_vle_only_hE_has_the_wrong_sign_and_is_worse_than_no_term():
    """h^E of NRTL parameters fitted to VLE only (the M15 chain and the ChemSep bank) vs calorimetry: positive
    everywhere while the measured one is negative below ~363 K; its AAD is larger than the AAD of «no excess term» at
    every isotherm up to 363.15 K, and smaller only at 383.15 K."""
    wrong = {298.15: 12, 323.15: 15, 333.15: 18, 363.15: 7, 383.15: 0}
    for prm in (NP.resolve_parameters(EW, _vle_only()), NP.resolve_parameters(EW, None)):
        assert prm["pairs"][0]["hE_basis"]["kind"] == "solo_VLE"
        for s in _he():
            T = s["T_K"]
            xs = np.array([q["x1"] for q in s["points"]])
            h = np.array([q["hE_J_mol"] for q in s["points"]])
            m = np.array([NP.excess_enthalpy_numeric(prm, T, [x, 1 - x]) for x in xs])
            assert m.min() > 0
            assert int(np.sum(np.sign(m) != np.sign(h))) == wrong[T]
            with_term, without = np.mean(np.abs(m - h)), np.mean(np.abs(h))
            assert (with_term > without) == (T < 380.0), (T, with_term, without)
    # the M15b feed (10 mol % ethanol, 350 K): +388 J/mol from the model, between two measured negatives
    feed = NP.excess_enthalpy_numeric(NP.resolve_parameters(EW, _vle_only()), 350.0, [0.1, 0.9])
    assert feed == pytest.approx(387.9, abs=0.5)


def test_joint_fit_on_real_data_states_that_one_nrtl_set_does_not_represent_both():
    """All five isotherms + Kamihama, α = 0.3, σ-weights 1 and 1 (nothing tuned): the tool says NO, with numbers. The
    weight scan shows the trade-off in both directions."""
    r = _joint_all()
    m = r["modelos"]["NRTL"]
    v = m["veredicto_conjunto"]
    assert v["representa_VLE_y_hE"] is False and any("NO representa VLE y h^E" in a for a in r["avisos"])
    assert m["estadistica"]["AAD_T_K"] == pytest.approx(1.348, abs=0.02) and m["estadistica"]["AAD_y"] == pytest.approx(0.0276, abs=0.001)
    assert m["estadistica_hE"]["AAD_hE_J_mol"] == pytest.approx(106.6, abs=1.0)
    assert {"AAD_T_K", "AAD_y"} <= set(v["no_cumple"]) and sum("AAD_hE" in c for c in v["no_cumple"]) == 5
    scan = m["compromiso_VLE_hE"]["barrido"]
    assert [s["peso_hE"] for s in scan] == [0, 0.01, 1, 100]
    aad_T, aad_h = [s["AAD_T_K"] for s in scan], [s["AAD_hE_J_mol"] for s in scan]
    assert aad_T == sorted(aad_T) and aad_h == sorted(aad_h, reverse=True)      # more h^E weight: better h^E, worse VLE
    assert aad_T[0] < 0.1 and aad_h[0] > 400 and aad_T[-1] > 5 and aad_h[-1] > 75
    assert not any(s["representa_VLE_y_hE"] for s in scan)
    # still far better in h^E than the VLE-only parameters (442.6 J/mol) or no term at all (231.7 J/mol)
    assert m["estadistica_hE"]["AAD_hE_J_mol"] < 0.5 * 231.7


def test_joint_fit_local_to_the_column_keeps_vle_and_gets_the_sign_right():
    """Kamihama + the 363.15 K isotherm only (the closest to an atmospheric column): VLE inside its tolerances, h^E
    AAD 37 J/mol with the measured sign at the M15b feed; outside 363 K it does not hold (said by the tool's warning
    and by the AAD at the other isotherms)."""
    r = _joint_363()
    m = r["modelos"]["NRTL"]
    assert m["estadistica"]["AAD_T_K"] < 0.2 and m["estadistica"]["AAD_y"] < 0.005
    assert m["estadistica_hE"]["AAD_hE_J_mol"] == pytest.approx(36.6, abs=1.0)
    ok = {c["criterio"]: c["cumple"] for c in m["veredicto_conjunto"]["criterios"]}
    assert ok["AAD_T_K"] and ok["AAD_y"]
    prm = NP.resolve_parameters(EW, r)
    assert NP.excess_enthalpy_numeric(prm, 350.0, [0.1, 0.9]) < 0
    s298 = _he([298.15])[0]
    far = np.mean([abs(NP.excess_enthalpy_numeric(prm, 298.15, [q["x1"], 1 - q["x1"]]) - q["hE_J_mol"]) for q in s298["points"]])
    assert far > 300


# ------------------------------------------------------------------------------------------------ provenance labels
def test_hE_provenance_of_every_parameter_source():
    bank = NP.resolve_parameters(EW, None)
    assert bank["pairs"][0]["hE_basis"] == {"kind": "solo_VLE"}
    reg = NP.resolve_parameters(EW, _vle_only())
    assert reg["pairs"][0]["hE_basis"] == {"kind": "solo_VLE"}
    joint = NP.resolve_parameters(EW, _joint_all())
    hb = joint["pairs"][0]["hE_basis"]
    assert hb["kind"] == "ajustado_a_hE" and hb["represents_VLE_and_hE"] is False and hb["weights"] == {"vle": 1.0, "hE": 1.0}
    assert any("Brandt" in c for c in hb["data"]) and any("Nagamachi" in c for c in hb["data"])
    assert hb["T_K"] == [298.15, 323.15, 333.15, 363.15, 383.15] and len(hb["AAD_hE_by_set_and_T"]) == 5
    assert joint["pairs"][0]["a_ij"] != 0.0 and joint["pairs"][0]["a_ji"] != 0.0      # T-dependent parameters accepted
    comps, vle = _vle()
    w0 = vle_binary_regression(comps, vle, models=["NRTL"], compare_unifac=False, temperature_dependence="a+b/T",
                               excess_enthalpy_sets=_he([363.15]), weights={"hE": 0.0})
    hb0 = NP.resolve_parameters(EW, w0)["pairs"][0]["hE_basis"]
    assert hb0["kind"] == "solo_VLE" and "solo comparado" in hb0["note"] and hb0["AAD_hE_J_mol"] > 300
    user = {"components": EW, "b_ij_K": -29.2, "b_ji_K": 624.9, "alpha": 0.2937, "source": "cita del usuario"}
    assert NP.resolve_parameters(EW, [user])["pairs"][0]["hE_basis"] == {"kind": "no_declarado"}
    cited = NP.resolve_parameters(EW, [{**user, "hE_source": "Brandt et al. 2014"}])["pairs"][0]["hE_basis"]
    assert cited == {"kind": "declarado_por_el_usuario", "data": ["Brandt et al. 2014"]}
    with pytest.raises(ValueError, match="hE_source"):
        NP.resolve_parameters(EW, [{**user, "hE_source": " "}])


def test_column_report_says_where_hE_comes_from_and_its_share_of_the_duty():
    """excess_enthalpy_report on the M15b optimum (numbers of notebooks/26, no solve here): class of the h^E source,
    Spanish text, and the first-order share of Q_reb."""
    feed = {"z": {"ethanol": 0.1, "water": 0.9}, "T_K": 350.0, "flow_mol_s": 100.0}
    opt = {"distillate": {"mole_fractions": {"ethanol": 0.70, "water": 0.30}, "T_K": 351.689, "flow_mol_s": 13.66865},
           "bottoms": {"mole_fractions": {"ethanol": 0.005, "water": 0.995}, "T_K": 371.6786, "flow_mol_s": 86.33135},
           "reboiler_duty_W": 1030.023e3}
    vle_only = NP.resolve_parameters(EW, _vle_only())
    on = NP.excess_enthalpy_report(vle_only, 101325.0, feed, opt, True)
    assert on["source"]["class"] == "solo_VLE" and "SIN validación calorimétrica" in on["source"]["text"]
    assert "signo contrario" in on["source"]["text"]
    assert on["first_order_share_of_reboiler_duty"] == pytest.approx(-0.0313, abs=5e-4)
    assert "ya incluido" in on["share_meaning"]
    off = NP.excess_enthalpy_report(vle_only, 101325.0, feed, opt, False)
    assert off["source"]["class"] == "termino_de_exceso_apagado" and off["source"]["of_parameters"] == "solo_VLE"
    assert "apagado" in off["share_meaning"] and not off["included_in_liquid_enthalpy"]
    joint = NP.excess_enthalpy_report(NP.resolve_parameters(EW, _joint_all()), 101325.0, feed, opt, True)
    assert joint["source"]["class"] == "ajustado_a_hE" and "no representa VLE y h^E" in joint["source"]["text"]
    assert any("Brandt" in c for c in joint["source"]["pairs"][0]["data"])
    assert joint["streams"]["feed"]["hE_J_mol"] < 0 < on["streams"]["feed"]["hE_J_mol"]
    local = NP.excess_enthalpy_source(NP.resolve_parameters(EW, _joint_363()), True)
    assert local["class"] == "ajustado_a_hE" and local["pairs"][0]["T_K"] == [363.15]
    user = NP.resolve_parameters(EW, [{"components": EW, "b_ij_K": -29.2, "b_ji_K": 624.9, "alpha": 0.2937, "source": "x"}])
    assert NP.excess_enthalpy_source(user, True)["class"] == "no_declarado"


def test_idaes_package_takes_the_temperature_dependent_set():
    """The joint a + b/T parameters inside the IDAES property package: ln γ and h^E equal to thermo.nrtl.NRTL and to a
    finite difference of g^E/RT (no solve)."""
    prm = NP.resolve_parameters(EW, _joint_363())
    cfg, prov = NP.build_configuration(EW, 101325.0, prm, extra_T=[350.0])
    chk = NP.nrtl_check(cfg, prm, [333.15, 350.0, 363.15, 372.0])
    assert chk["verified"], (chk["max_abs_ln_gamma_diff"], chk["max_rel_hE_diff"])
    assert prov["liquid_model"]["pairs"][0]["hE_basis"]["kind"] == "ajustado_a_hE"


def test_tool_signatures_expose_the_new_arguments():
    import inspect
    from packages.dwsim_client import tools as T  # noqa: F401  (import order, see M8c handover)
    from packages.dwsim_client import tools_optimize as TO
    from packages.agent import agent as A
    assert inspect.signature(TO.eo_column_optimize).parameters["excess_enthalpy"].default is True      # default unchanged
    assert inspect.signature(C.eo_column_optimize).parameters["excess_enthalpy"].default is True
    reg = inspect.signature(vle_binary_regression).parameters
    assert reg["excess_enthalpy_sets"].default is None and reg["weights"].default is None
    for name in ("vle_binary_regression", "eo_column_optimize"):
        fn = getattr(A, name)
        doc = (getattr(fn, "__doc__", None) or getattr(getattr(fn, "func", None), "__doc__", "") or "")
        assert "excess_enthalpy" in doc, name


# ------------------------------------------------------------------------------------------------ slow: column + DWSIM
EW_SPECS = [{"stream": "distillate", "component": "ethanol", "kind": "mole_fraction", "min": 0.70},
            {"stream": "bottoms", "component": "water", "kind": "mole_fraction", "min": 0.995}]
# notebooks/27_eo_nrtl_excess_enthalpy.json, case J_363 (EO vs DWSIM «ExpData» there: Q_reb +0.22 %, Q_cond +0.09 %)
J363_REF = {"Q_reb_on_kW": 1066.1717, "Q_reb_off_kW": 1066.2512, "dwsim_tol": 0.005}


@slow
def test_joint_set_runs_through_the_column_and_matches_dwsim_like_for_like():
    """Kamihama + 363.15 K h^E (a + b/T) → eo_column_optimize on the M15b reference column, h^E on and off; the
    numbers of notebooks/27; DWSIM «ExpData» with the SAME a + b/T parameters vs the run without h^E."""
    from packages.eo import dwsim_nrtl as DN
    reg = _joint_363()
    out = {}
    for hE in (True, False):
        r = C.eo_column_optimize(EW, 100.0, 350.0, 101325.0, {"ethanol": 0.1, "water": 0.9}, 20, 14, 101325.0, EW_SPECS,
                                 activity_model="NRTL", binary_parameters=reg, excess_enthalpy=hE)
        assert r["evidence"]["verified"], r["evidence"]["reasons"]
        src = r["properties"]["excess_enthalpy"]["source"]
        assert src["class"] == ("ajustado_a_hE" if hE else "termino_de_exceso_apagado")
        assert ("Origen de ese h^E" in r["assumptions"][1]) == hE
        out[hE] = r
    on, off = out[True], out[False]
    assert on["reboiler_duty_MW"] * 1e3 == pytest.approx(J363_REF["Q_reb_on_kW"], rel=2e-4)
    assert off["reboiler_duty_MW"] * 1e3 == pytest.approx(J363_REF["Q_reb_off_kW"], rel=2e-4)
    # fitted at 363.15 K, the model gives −3 J/mol at the 350 K feed where calorimetry says about −154: on ≈ off
    assert -10 < on["properties"]["excess_enthalpy"]["streams"]["feed"]["hE_J_mol"] < 0
    o = off["optimum"]
    dw = DN.column(["Ethanol", "Water"], 350.0, 101325.0, {"Ethanol": 10.0, "Water": 90.0}, 22, 15, o["reflux_ratio"],
                   101325.0, bottoms_mol_s=o["bottoms"]["flow_mol_s"], params=NP.resolve_parameters(EW, reg),
                   dwsim_names={"ethanol": "Ethanol", "water": "Water"}, poynting=False, enthalpy_mode="ExpData")
    assert dw["ok"], dw.get("solver_errors")
    assert off["reboiler_duty_MW"] * 1e3 == pytest.approx(dw["reboiler_duty_kW"], rel=J363_REF["dwsim_tol"])
    assert -off["condenser_duty_MW"] * 1e3 == pytest.approx(dw["condenser_duty_kW"], rel=J363_REF["dwsim_tol"])


