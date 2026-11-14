"""Bayesian optimisation and experiment design (bloque M16): `optimize_design_bayesian` and `suggest_experiments`.

Fast: the core against functions with a known optimum (Branin, the constrained problem of Gramacy et al. 2016), failed
evaluations, integers, log scale, seeds, stopping rules and refusals; the closed-form leave-one-out and the log-EI
against independent calculations; `suggest_experiments` modes, labels and refusals; one real DWSIM run (the
constrained three-stage compressor of notebook 23, direct optimum 2.699/8.591 bar, 892.68 kW).
Slow (ALCHEMAI_SLOW_BAYES=1): success rates over seeds (Hartmann-6 included), the three DWSIM cases against
optimize_design_multi, a DWSIM column whose solve fails in part of the range, integer stages, and the SnAr campaign
against random sampling. Their numbers are recorded in notebooks/29_optimizacion_bayesiana.json and BENCHMARK.md.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.bo import benchmarks as B  # noqa: E402
from packages.bo import core as C  # noqa: E402
from packages.bo.experiments import MIN_POINTS, PROPOSAL, suggest_experiments  # noqa: E402

slow = pytest.mark.skipif(os.environ.get("ALCHEMAI_SLOW_BAYES") != "1", reason="slow M16 runs (≈ 15 min): ALCHEMAI_SLOW_BAYES=1")


def _f(fun):
    return lambda x: {"solved_ok": True, "objective": fun(x), "constraints": {}}


def _nb(name):
    if name == "nb29":       # imported by name so its functions can be sent to worker processes
        import tempfile
        os.environ.setdefault("ALCHEMAI_NB29_OUT", str(Path(tempfile.gettempdir()) / "alchemai_nb29_test.json"))   # keep the recorded evidence
        sys.path.insert(0, str(ROOT / "notebooks"))
        return importlib.import_module("29_optimizacion_bayesiana")
    spec = importlib.util.spec_from_file_location(name, ROOT / "notebooks" / "23_sustituto_omlt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------------------------- reference problems
def test_reference_functions_have_their_published_optima():
    for m in B.BRANIN_MINIMA:
        assert B.branin(m) == pytest.approx(B.BRANIN_FMIN, abs=2e-6)
    assert B.hartmann6(B.HARTMANN6_XMIN) == pytest.approx(B.HARTMANN6_FMIN, abs=2e-5)
    g = B.gramacy(B.GRAMACY_XMIN)
    assert g["objective"] == pytest.approx(B.GRAMACY_FMIN, abs=1e-4) and abs(g["constraints"]["c1"]) < 1e-4 and g["constraints"]["c2"] < 0


def test_gramacy_optimum_by_independent_search():
    """The published constrained optimum 0.5998 (Gramacy et al. 2016), recomputed with SLSQP from a dense grid of starts."""
    from scipy.optimize import minimize
    best = math.inf
    for a in np.linspace(0.05, 0.95, 7):
        for b in np.linspace(0.05, 0.95, 7):
            r = minimize(lambda x: x[0] + x[1], [a, b], method="SLSQP", bounds=[(0, 1)] * 2,
                         constraints=[{"type": "ineq", "fun": lambda x, k=k: -B.gramacy(x)["constraints"][k]} for k in ("c1", "c2")])
            if r.success and all(v <= 1e-7 for v in B.gramacy(r.x)["constraints"].values()):
                best = min(best, r.fun)
    assert best == pytest.approx(B.GRAMACY_FMIN, abs=2e-4)


def test_snar_emulator_balances_and_optimum():
    r = B.snar(1.0, 2.0, 0.3, 80.0)
    assert 0 < r["yield_ortho"] < 1 and r["sty"] == pytest.approx(60 * 210.21 * r["yield_ortho"] * 0.3 / 1.0, rel=1e-9)
    nb = _nb("nb29")
    for key, spec in nb.SNAR.items():          # the optimum found by differential evolution (notebook 29 `truth`)
        assert B.snar(*spec["x"])[key] == pytest.approx(spec["optimum"], rel=2e-4)
        rng = np.random.default_rng(0)
        lo = np.array([v["lower"] for v in B.SNAR_VARS]); hi = np.array([v["upper"] for v in B.SNAR_VARS])
        ys = [B.snar(*(lo + rng.random(4) * (hi - lo)))[key] for _ in range(300)]
        assert (max(ys) <= spec["optimum"] * 1.0001) if spec["maximize"] else (min(ys) >= spec["optimum"] * 0.9999)


# --------------------------------------------------------------------------------------------------- model pieces
def test_log_ei_matches_direct_and_high_precision():
    import mpmath as mp
    u = np.array([-40.0, -12.0, -6.5, -6.0001, -5.9999, -3.0, 0.0, 2.5])
    got = C.log_h(u)
    for ui, gi in zip(u, got):
        ref = mp.log(mp.npdf(ui) + ui * mp.ncdf(ui))
        assert gi == pytest.approx(float(ref), abs=2e-3 if ui < -6 else 1e-6)   # series truncated at 1/u⁶ below −6


def test_leave_one_out_closed_form_equals_refitting():
    rng = np.random.default_rng(3)
    Z = rng.random((14, 2))
    y = np.array([B.branin([-5 + 15 * a, 15 * b]) for a, b in Z])
    gp = C.fit_gp(Z, y, noisy=True, seed=0)
    lo = C.loo(gp)
    from sklearn.gaussian_process import GaussianProcessRegressor
    for i in (0, 5, 13):
        keep = [j for j in range(14) if j != i]
        g = GaussianProcessRegressor(gp.kernel_, alpha=1e-10, optimizer=None, normalize_y=False)
        g.fit(Z[keep], (y[keep] - gp._y_train_mean) / gp._y_train_std)        # same standardisation, same hyper-parameters
        m, s = g.predict(Z[[i]], return_std=True)
        assert lo["mean"][i] == pytest.approx(m[0] * gp._y_train_std + gp._y_train_mean, rel=1e-6, abs=1e-6)
        assert lo["sd"][i] == pytest.approx(s[0] * gp._y_train_std, rel=1e-5)
    s = C.loo_summary(gp)
    assert s["n"] == 14 and 0 <= s["fraction_within_2_sd"] <= 1 and s["q2"] <= 1


# ------------------------------------------------------------------------------------------------------- the loop
@pytest.mark.parametrize("seed", [0, 1])
def test_branin_within_tolerance_and_budget(seed):
    r = C.bayes_optimize(_f(B.branin), B.BRANIN_VARS, max_evaluations=40, objective_tolerance=0.01, seed=seed)
    assert r["best"]["objective"] - B.BRANIN_FMIN <= 0.01 and r["n_evaluations"] <= 40
    x = (r["best"]["values"]["x1."], r["best"]["values"]["x2."])
    assert min(math.dist(x, m) for m in B.BRANIN_MINIMA) < 0.6
    assert r["engine_at_optimum"]["objective"] == r["best"]["objective"] and r["evidence"]["reevaluation_reproduces"]
    assert len(r["history"]) == r["n_evaluations"] and r["history"][-1]["predicted_sd"] > 0
    g = r["gp_at_optimum"]
    assert g["engine_value"] == r["best"]["objective"] and g["leave_one_out_sd"] > 0 and g["abs_error"] >= 0
    assert r["gp_diagnostics"]["objective"]["leave_one_out"]["q2"] > 0.9 and r["gp_diagnostics"]["one_step_ahead"]["n"] > 0
    assert r["method"]["engine"].startswith("scikit-learn") and r["method"]["license"] == "BSD-3-Clause"
    assert r["suitability"]["notes"]                     # a cheap 2-variable problem: the tool says a direct optimiser fits better


def test_constrained_problem_reaches_the_known_optimum():
    r = C.bayes_optimize(B.gramacy, B.GRAMACY_VARS, B.GRAMACY_LIMITS, max_evaluations=60, objective_tolerance=0.006, seed=0)
    assert r["status"] == "verified", r["reasons"]
    assert r["best"]["objective"] == pytest.approx(B.GRAMACY_FMIN, abs=0.006)
    assert r["best"]["constraints"]["c1"] <= 0 and r["best"]["constraints"]["c2"] <= 0     # hard constraints: a feasible evaluated point
    assert r["active_constraints"] == ["c1 ≤ 0.0"] and r["engine_constraint_violation"] == {"c1": 0.0, "c2": 0.0}
    assert any(not h["feasible"] for h in r["history"]) and set(r["gp_diagnostics"]) >= {"objective", "c1", "c2"}


def test_failed_evaluations_are_infeasible_points_not_crashes():
    calls = {"n": 0}

    def evaluate(x):
        calls["n"] += 1
        if x[0] > 6.0:
            raise RuntimeError("solver diverged")            # an exception in the engine
        if x[1] > 12.0:
            return {"solved_ok": False, "objective": None, "constraints": {}}     # a solve that did not converge
        if x[0] < -4.5:
            return {"solved_ok": True, "objective": float("nan"), "constraints": {}}
        return {"solved_ok": True, "objective": B.branin(x), "constraints": {}}

    r = C.bayes_optimize(evaluate, B.BRANIN_VARS, max_evaluations=45, objective_tolerance=0.01, seed=0)
    assert r["n_failed_evaluations"] >= 3 and r["n_evaluations"] == len(r["history"]) and calls["n"] == r["n_evaluations"] + 1
    failed = [h for h in r["history"] if not h["solved_ok"]]
    assert all(h["objective"] is None and not h["feasible"] for h in failed) and any("solver diverged" in h.get("error", "") for h in failed)
    # the only minimum left in the region that solves is (π, 2.275)
    assert r["best"]["objective"] - B.BRANIN_FMIN <= 0.01
    assert math.dist((r["best"]["values"]["x1."], r["best"]["values"]["x2."]), B.BRANIN_MINIMA[1]) < 0.6
    late = [h for h in r["history"] if h["phase"] == "bayesiana"][10:]
    assert sum(not h["solved_ok"] for h in late) <= len(late) // 2          # the model learned where the engine fails
    assert r["n_failed_evaluations"] <= 15            # without the pessimistic imputation: 48 failures in 60 evaluations


def test_engine_that_never_solves_ends_cleanly():
    def evaluate(x):
        raise ValueError("no property package")

    r = C.bayes_optimize(evaluate, B.BRANIN_VARS, max_evaluations=9, seed=0)
    assert r["status"] == "not_verified" and r["feasible"] is False and r["n_failed_evaluations"] == 9
    assert "no property package" in r["history"][0]["error"] and "ninguna evaluación resolvió" in r["note"]


def test_same_seed_same_run_and_budget_stop_is_not_verified():
    a = C.bayes_optimize(_f(B.branin), B.BRANIN_VARS, max_evaluations=12, objective_tolerance=0.01, seed=5)
    b = C.bayes_optimize(_f(B.branin), B.BRANIN_VARS, max_evaluations=12, objective_tolerance=0.01, seed=5)
    c = C.bayes_optimize(_f(B.branin), B.BRANIN_VARS, max_evaluations=12, objective_tolerance=0.01, seed=6)
    assert [h["values"] for h in a["history"]] == [h["values"] for h in b["history"]]
    assert [h["values"] for h in a["history"]] != [h["values"] for h in c["history"]]
    assert a["stopped_by"]["reason"] == "budget" and a["n_evaluations"] == 12
    assert a["status"] == "not_verified" and a["evidence"]["converged_before_budget"] is False and "presupuesto" in a["reasons"][0]


def test_integer_and_log_scale_variables():
    """f = (n − 7)² + (log10 x − 2)², n integer 1..12, x in 1..1e4 on a log scale: optimum n = 7, x = 100."""
    V = [{"name": "T-1", "param": "stages", "lower": 1, "upper": 12, "integer": True},
         {"name": "feed", "param": "flow", "lower": 1.0, "upper": 1e4, "log_scale": True}]
    r = C.bayes_optimize(_f(lambda x: (x[0] - 7) ** 2 + (math.log10(x[1]) - 2) ** 2), V, max_evaluations=40, objective_tolerance=0.01, seed=0)
    ns = [h["values"]["T-1.stages"] for h in r["history"]]
    assert all(isinstance(n, int) and 1 <= n <= 12 for n in ns) and r["method"]["integer_variables"] == ["T-1.stages"]
    assert r["best"]["values"]["T-1.stages"] == 7 and r["best"]["values"]["feed.flow"] == pytest.approx(100, rel=0.3)
    assert r["best"]["objective"] <= 0.01
    pts = [tuple(h["values"].values()) for h in r["history"]]
    assert len(set(pts)) == len(pts)                                   # no point evaluated twice
    xs = np.array([h["values"]["feed.flow"] for h in r["history"] if h["phase"] == "inicial"])
    assert np.sum(xs < 100) >= 1 and np.sum(xs > 100) >= 1             # the log scale spreads the design over the decades


def test_all_integer_space_is_enumerated_at_most_once():
    V = [{"name": "T-1", "param": "stages", "lower": 3, "upper": 9, "integer": True}]
    r = C.bayes_optimize(_f(lambda x: (x[0] - 8) ** 2 + 1.0), V, max_evaluations=30, objective_tolerance=1e-6, seed=0)
    assert r["best"]["values"] == {"T-1.stages": 8} and r["n_evaluations"] <= 7
    assert r["stopped_by"]["reason"] in ("space_exhausted", "budget", "expected_improvement", "no_improvement")


def test_refusals():
    ok = _f(B.branin)
    with pytest.raises(ValueError, match="1 a 10 variables"):
        C.bayes_optimize(ok, [{"name": f"x{i}", "param": "", "lower": 0, "upper": 1} for i in range(11)])
    with pytest.raises(ValueError, match="cotas"):
        C.bayes_optimize(ok, [{"name": "x", "param": "p", "lower": 0}])
    with pytest.raises(ValueError, match="escala log"):
        C.bayes_optimize(ok, [{"name": "x", "param": "p", "lower": 0.0, "upper": 1.0, "log_scale": True}])
    with pytest.raises(ValueError, match="entera y en escala log"):
        C.bayes_optimize(ok, [{"name": "x", "param": "p", "lower": 1, "upper": 9, "log_scale": True, "integer": True}])
    with pytest.raises(ValueError, match="max o min"):
        C.bayes_optimize(ok, B.BRANIN_VARS, {"c": {}})


# --------------------------------------------------------------------------------------------- suggest_experiments
VARS_E = [{"name": "T_C", "lower": 40.0, "upper": 120.0}, {"name": "tau_min", "lower": 0.5, "upper": 5.0, "log_scale": True}]


def _lab(T, tau, rng=None):
    """Made-up smooth response for the unit tests only (not a chemistry model): maximum 0.9 at 90 °C, 2 min."""
    y = 0.9 * math.exp(-((T - 90) / 25) ** 2 - (math.log(tau / 2.0) / 0.8) ** 2)
    return y + (rng.normal(scale=0.01) if rng is not None else 0.0)


def test_suggest_without_data_returns_a_labelled_initial_design():
    r = suggest_experiments(VARS_E, [], {"name": "yield", "maximize": True}, batch_size=6, seed=1)
    assert r["mode"] == "initial_design" and r["model"] is None and len(r["proposals"]) == 6 and r["current_best"] is None
    assert "DISEÑO INICIAL" in r["note"] and "hipercubo latino" in " ".join(r["assumptions"])
    for p in r["proposals"]:
        assert p["label"] == PROPOSAL and p["predicted_mean"] is None and "yield" not in p and "yield" not in p["inputs"]
        assert 40 <= p["inputs"]["T_C"] <= 120 and 0.5 <= p["inputs"]["tau_min"] <= 5
    T = sorted(p["inputs"]["T_C"] for p in r["proposals"])          # Latin hypercube: one point per stratum
    assert all(40 + 80 * i / 6 <= t <= 40 + 80 * (i + 1) / 6 for i, t in enumerate(T))


def test_suggest_with_too_few_data_fills_the_space_away_from_what_exists():
    data = [{"T_C": 60.0, "tau_min": 1.0, "yield": 0.21}, {"T_C": 100.0, "tau_min": 3.0, "yield": 0.55}]
    r = suggest_experiments(VARS_E, data, {"name": "yield", "maximize": True}, batch_size=3)
    assert r["mode"] == "initial_design" and r["n_valid"] == 2 and MIN_POINTS(2) == 6 and "Faltan 4" in r["note"]
    assert r["current_best"]["experiment"] == 2 and r["current_best"]["yield"] == 0.55          # a measured row, untouched
    for p in r["proposals"]:
        assert all(abs(p["inputs"]["T_C"] - d["T_C"]) > 1 or abs(p["inputs"]["tau_min"] - d["tau_min"]) > 0.05 for d in data)


def test_suggest_with_a_model_proposes_towards_the_optimum_and_never_invents_data():
    rng = np.random.default_rng(0)
    pts = [(50, 0.7), (50, 3.5), (110, 0.7), (110, 3.5), (80, 1.5), (65, 2.5), (100, 1.0), (75, 4.5), (115, 2.0), (45, 1.2)]
    data = [{"T_C": float(T), "tau_min": tau, "yield": _lab(T, tau, rng)} for T, tau in pts]
    snapshot = [dict(d) for d in data]
    r = suggest_experiments(VARS_E, data, {"name": "yield", "maximize": True}, batch_size=4, seed=0)
    assert data == snapshot and r["n_experiments"] == 10 and r["mode"] == "model" and len(r["proposals"]) == 4
    best = max(snapshot, key=lambda d: d["yield"])
    assert r["current_best"]["yield"] == best["yield"] and r["current_best"]["inputs"]["T_C"] == best["T_C"]
    for p in r["proposals"]:
        assert p["label"] == PROPOSAL and p["predicted_sd"] > 0 and "yield" not in p
        lo, hi = p["interval95_new_measurement"]
        assert lo < p["predicted_mean"] < hi and 40 <= p["inputs"]["T_C"] <= 120 and 0.5 <= p["inputs"]["tau_min"] <= 5
    pts4 = [(p["inputs"]["T_C"], p["inputs"]["tau_min"]) for p in r["proposals"]]
    assert len(set(pts4)) == 4                                           # a batch, not the same point four times
    first = r["proposals"][0]["inputs"]
    assert abs(first["T_C"] - 90) < 20 and 1.0 < first["tau_min"] < 4.0  # the first pick goes to the promising region
    m = r["model"]
    assert m["leave_one_out"]["yield"]["n"] == 10 and m["noise_sd_fitted"]["yield"] >= 0 and "PROPUESTAS" in r["note"]
    again = suggest_experiments(VARS_E, data, {"name": "yield", "maximize": True}, batch_size=4, seed=0)
    assert [p["inputs"] for p in again["proposals"]] == [p["inputs"] for p in r["proposals"]]


def test_suggest_handles_failed_runs_and_measured_limits():
    rng = np.random.default_rng(1)
    data = []
    for T, tau in [(50, 0.7), (50, 3.5), (110, 0.7), (80, 1.5), (65, 2.5), (100, 1.0), (75, 4.5), (45, 1.2), (90, 2.0), (95, 3.0)]:
        data.append({"T_C": float(T), "tau_min": tau, "yield": _lab(T, tau, rng), "impurity": 0.002 * (T - 40) * tau})
    data += [{"T_C": 118.0, "tau_min": 4.8, "failed": True}, {"T_C": 112.0, "tau_min": 4.0, "failed": True}]      # reactor fouled
    r = suggest_experiments(VARS_E, data, {"name": "yield", "maximize": True}, batch_size=3,
                            constraints=[{"name": "impurity", "max": 0.2}], seed=0)
    assert r["mode"] == "model" and r["n_failed"] == 2 and r["n_valid"] == 10 and r["model"]["failure_model"]
    assert r["current_best"]["constraints"]["impurity"] <= 0.2
    for p in r["proposals"]:
        assert 0 <= p["probability_within_limits"] <= 1 and "impurity" in p["constraints"]
        assert not (p["inputs"]["T_C"] > 110 and p["inputs"]["tau_min"] > 3.9)      # stays away from the failed corner


def test_suggest_refusals():
    obj = {"name": "yield", "maximize": True}
    d = [{"T_C": 60.0, "tau_min": 1.0, "yield": 0.2}]
    with pytest.raises(ValueError, match="Pareto"):
        suggest_experiments(VARS_E, d, [obj, {"name": "cost", "maximize": False}], 2)
    with pytest.raises(ValueError, match="dirección"):
        suggest_experiments(VARS_E, d, {"name": "yield"}, 2)
    with pytest.raises(ValueError, match="categóricas"):
        suggest_experiments(VARS_E + [{"name": "solvent", "categories": ["EtOH", "MeCN"]}], d, obj, 2)
    with pytest.raises(ValueError, match="no se inventan mediciones"):
        suggest_experiments(VARS_E, [{"T_C": 60.0, "tau_min": 1.0}], obj, 2)
    with pytest.raises(ValueError, match="no se rellenan"):
        suggest_experiments(VARS_E, [{"T_C": 60.0, "yield": 0.3}], obj, 2)
    with pytest.raises(ValueError, match="batch_size"):
        suggest_experiments(VARS_E, d, obj, 0)
    with pytest.raises(ValueError, match="cotas"):
        suggest_experiments([{"name": "T_C", "lower": 40.0}], d, obj, 2)
    with pytest.raises(ValueError, match="falta la medición de impurity"):
        suggest_experiments(VARS_E, d, obj, 2, constraints=[{"name": "impurity", "max": 1.0}])


# ---------------------------------------------------------------------------------------------- wiring and DWSIM
def test_tools_are_wired():
    from packages.agent import agent as AG
    from packages.dwsim_client import tools as T
    from packages.dwsim_client import tools_optimize as TO
    for name in ("optimize_design_bayesian", "suggest_experiments"):
        assert getattr(TO, name) in TO.OPTIMIZE_TOOLS and getattr(TO, name) in T.TOOLS
        assert any(getattr(t, "name", "") == name for t in AG.AGENT_TOOLS)
    assert "PROPOSAL" in TO.suggest_experiments.__doc__ and "optimize_design_multi" in TO.optimize_design_bayesian.__doc__
    with pytest.raises(ValueError, match="objective.type"):
        TO.optimize_design_bayesian("none", [{"kind": "unit", "name": "K-1", "param": "outlet_P_Pa", "lower": 1e5, "upper": 2e5}], {"type": "x"})


def test_dwsim_constrained_compressor_matches_the_direct_optimum():
    """Real DWSIM, stage-1 discharge ≤ 430 K: optimize_design_multi gives 2.699/8.591 bar, 892.68 kW in 80 solves."""
    nb = _nb("nb23")
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_bayesian
    fid = nb.three_stage()
    r = optimize_design_bayesian(fid, nb.VARS_K, nb.OBJ_K, nb.LIMIT_B, seed=0)
    assert r["feasible"] and r["best"]["objective"] == pytest.approx(892.68, rel=0.001)
    assert r["best"]["constraints"]["s1.T_K"] <= 430.0 and r["engine_at_optimum"]["objective"] == r["best"]["objective"]
    assert r["n_evaluations"] <= 40 and r["n_failed_evaluations"] == 0
    assert T.read_stream(fid, "s1")["T_K"] == pytest.approx(r["best"]["constraints"]["s1.T_K"])       # left solved at the optimum
    T.close_flowsheet(fid)


@slow
def test_success_rates_on_test_functions():
    """Recorded (20 seeds, notebook 29): Branin 20/20, Gramacy 20/20, Hartmann-6 8/20 in one run and 13/20 with two
    restarts in the same 120 evaluations (the others stop in a local minimum, mostly −3.2032)."""
    nb = _nb("nb29")
    out = nb.functions(20)
    assert out["branin"]["n_success"] >= 19 and out["gramacy"]["n_success"] >= 19
    assert out["hartmann6"]["n_success"] >= 6 and out["hartmann6_restarts2"]["n_success"] >= 11
    assert out["hartmann6_restarts2"]["n_success"] > out["hartmann6"]["n_success"]


@slow
def test_dwsim_cases_against_direct_optimizer_and_failing_column():
    nb = _nb("nb29")
    out = nb.dwsim(seeds=(0, 1, 2))
    for case in ("A", "B", "C"):
        assert out[case]["bayesian_rel_gap_pct_max"] <= 0.1, case
        assert out[case]["bayesian_evaluations_median"] < out[case]["direct_nelder_mead"]["n_evaluations"] / 2
        assert all(r["status"] == "verified" for r in out[case]["bayesian"])
    assert out["B"]["bayesian"][0]["active_constraints"] == ["s1.T_K ≤ 430.0"]
    # a column whose bottoms spec exceeds the feed in part of the range: DWSIM fails there, the run goes on
    nb23 = _nb("nb23")
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_bayesian
    fid = nb23.hexane_heptane_column()
    var = [nb23.VARS_C[0], nb23.VARS_C[1] | {"upper": 120.0}]
    r = optimize_design_bayesian(fid, var, nb23.OBJ_C, nb23.CONS_C, seed=0)
    T.close_flowsheet(fid)
    assert r["n_failed_evaluations"] >= 1 and r["feasible"] and r["best"]["constraints"]["dist.mole_fractions.N-hexane"] >= 0.95
    assert r["best"]["objective"] == pytest.approx(out["C"]["direct_nelder_mead"]["objective"], rel=0.01)


@slow
def test_integer_stages_and_snar_campaign():
    """Recorded: 31 stages, R 1.358, 884 913-884 921 USD/year in 20-36 solves (enumeration: 29, 1.394, 885 080 in 442)."""
    nb = _nb("nb29")
    it = nb.integers(seeds=(0,))
    assert it["bayesian"][0]["all_integer"] and abs(it["bayesian"][0]["rel_gap_pct"]) <= 0.5 and it["bayesian"][0]["n_evaluations"] <= 41
    s = nb.snar(10, noise_levels=(0.0,))
    assert s["sty_noise0"]["bayes_success_by_round"][-1] >= 8 and s["sty_noise0"]["random_success_by_round"][-1] <= 1
    for key in ("e_factor_noise0", "e_factor_log_noise0"):       # converges towards the optimum, beats random; 1 % only in part
        assert s[key]["bayes_median_gap_pct_by_round"][-1] < 0.5 * s[key]["random_median_gap_pct_by_round"][-1]
    assert s["e_factor_log_noise0"]["bayes_success_by_round"][-1] >= s["e_factor_noise0"]["bayes_success_by_round"][-1]
