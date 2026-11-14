"""Notebook 29 — Bayesian optimisation (bloque M16): `optimize_design_bayesian` and `suggest_experiments`.

functions  Branin (f* = 0.397887), Hartmann-6 (f* = −3.32237) and the constrained toy problem of Gramacy et al. 2016
           (f* = 0.5998), N seeds each: success rate at a tolerance and a budget stated BEFORE running (CRITERIA).
dwsim      The repo's own known optima (notebook 23): A three-stage compressor, B the same with the HAZOP limit,
           C the hexane-heptane column. Bayesian vs optimize_design_multi (grid + Nelder-Mead) vs
           optimize_design_surrogate (OMLT + ipopt) on the same flowsheet: optimum and DWSIM evaluations spent.
integers   Ethanol-water column of notebook 20, integer stages (8-60) × reflux, annual cost: Bayesian vs the recorded
           enumeration of optimize_design_multi (29 stages, R 1.394, 885 080 USD/year, 442 solves).
snar       `suggest_experiments` against the SnAr kinetic emulator of Summit (Hone et al. 2017 kinetics), which the
           tool never sees: rounds of 4 experiments from zero data, vs uniform random sampling at the same budget.
truth      Global optimum of the SnAr emulator by differential evolution (independent of the tools).
Usage: PYTHONPATH=. uv run python notebooks/29_optimizacion_bayesiana.py functions|dwsim|integers|snar|truth [n_seeds]
Results are merged into notebooks/29_optimizacion_bayesiana.json.
"""
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from packages.bo import benchmarks as B  # noqa: E402
from packages.bo import core as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402
OUT = Path(os.environ.get("ALCHEMAI_NB29_OUT") or ROOT / "notebooks" / "29_optimizacion_bayesiana.json")

# Success criteria, fixed before the runs: reach f* + tolerance within the budget, with the tool's own stopping rules on.
CRITERIA = {"branin": {"tolerance": 0.01, "budget": 40, "fmin": B.BRANIN_FMIN},
            "hartmann6": {"tolerance": 0.033, "budget": 120, "fmin": B.HARTMANN6_FMIN},        # 1 % of |f*|; local minimum −3.2032 fails it
            "gramacy": {"tolerance": 0.006, "budget": 60, "fmin": B.GRAMACY_FMIN},             # 1 % of f*; local optimum 0.75 fails it
            "hartmann6_restarts2": {"tolerance": 0.033, "budget": 120, "fmin": B.HARTMANN6_FMIN, "restarts": 2}}   # same budget, 2 restarts
SNAR = {"sty": {"maximize": True, "optimum": 11562.26, "x": [0.5, 5.0, 0.5, 30.899]},          # `truth` below
        "e_factor": {"maximize": False, "optimum": 8.4516, "x": [2.0, 1.1839, 0.5, 62.078]}}
SNAR_TOL_PCT, SNAR_BATCH, SNAR_ROUNDS = 1.0, 4, 10


def _save(key, value):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=1, default=float))


# ------------------------------------------------------------------------------------------------- test functions
def _run_function(a):
    name, seed = a
    c = CRITERIA[name]
    if name == "branin":
        r = C.bayes_optimize(lambda x: {"solved_ok": True, "objective": B.branin(x), "constraints": {}}, B.BRANIN_VARS,
                             max_evaluations=c["budget"], objective_tolerance=c["tolerance"], seed=seed)
    elif name.startswith("hartmann6"):
        r = C.bayes_optimize(lambda x: {"solved_ok": True, "objective": B.hartmann6(x), "constraints": {}}, B.HARTMANN6_VARS,
                             max_evaluations=c["budget"], objective_tolerance=c["tolerance"], seed=seed, restarts=c.get("restarts", 0))
    else:
        r = C.bayes_optimize(B.gramacy, B.GRAMACY_VARS, B.GRAMACY_LIMITS, max_evaluations=c["budget"],
                             objective_tolerance=c["tolerance"], seed=seed)
    gap = r["best"]["objective"] - c["fmin"]
    return {"seed": seed, "gap": gap, "success": bool(gap <= c["tolerance"]), "n_evaluations": r["n_evaluations"],
            "stopped_by": r["stopped_by"]["reason"], "status": r["status"], "model_seconds": r["suitability"]["model_seconds"],
            "runs_best": [x["best_objective"] for x in r["runs"]],
            "restarts_disagree": bool((r.get("restart_agreement") or {}).get("runs_at_a_worse_optimum"))}


def functions(n_seeds=20, names=("branin", "gramacy", "hartmann6", "hartmann6_restarts2")):
    out = {}
    for name in names:
        with Pool(7) as p:
            rows = p.map(_run_function, [(name, s) for s in range(n_seeds)])
        ev = [r["n_evaluations"] for r in rows]
        out[name] = {"criteria": CRITERIA[name], "n_seeds": n_seeds, "n_success": sum(r["success"] for r in rows),
                     "evaluations_median": float(np.median(ev)), "evaluations_min_max": [min(ev), max(ev)],
                     "gap_median": float(np.median([r["gap"] for r in rows])), "gap_max": max(r["gap"] for r in rows),
                     "stopped_by": {k: sum(r["stopped_by"] == k for r in rows) for k in {r["stopped_by"] for r in rows}},
                     "verified_but_failed": sum(r["status"] == "verified" and not r["success"] for r in rows),
                     "failed_and_flagged_by_restart_disagreement": sum(r["restarts_disagree"] and not r["success"] for r in rows), "runs": rows}
        print(name, {k: v for k, v in out[name].items() if k != "runs"}, flush=True)
        _save("functions_" + name, out[name])
    return out


# ---------------------------------------------------------------------------------------------------------- DWSIM
def _nb23():
    spec = importlib.util.spec_from_file_location("nb23", ROOT / "notebooks" / "23_sustituto_omlt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dwsim(cases=("A", "B", "C"), seeds=(0, 1, 2, 3, 4)):
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_bayesian, optimize_design_multi, optimize_design_surrogate
    nb = _nb23()
    out = {}
    for name in cases:
        build, var, obj, cons = {"A": (nb.three_stage, nb.VARS_K, nb.OBJ_K, None), "B": (nb.three_stage, nb.VARS_K, nb.OBJ_K, nb.LIMIT_B),
                                 "C": (nb.hexane_heptane_column, nb.VARS_C, nb.OBJ_C, nb.CONS_C)}[name]
        fid = build(); t = time.time()
        d = optimize_design_multi(fid, var, obj, constraints=cons, max_evaluations=150)
        direct = {"values": d["best"]["values"], "objective": d["best"]["objective"], "constraints": d["best"].get("constraints"),
                  "n_evaluations": d["n_evaluations"], "seconds": round(time.time() - t, 1)}
        T.close_flowsheet(fid)
        fid = build(); t = time.time()
        s = optimize_design_surrogate(fid, var, obj, cons)
        sur = {"status": s["status"], "values": s.get("surrogate_optimum", {}).get("values"),
               "objective": (s.get("dwsim_at_optimum") or {}).get("objective"), "n_evaluations": s["n_dwsim_solves"],
               "seconds": round(time.time() - t, 1)}
        T.close_flowsheet(fid)
        runs = []
        for seed in seeds:
            fid = build(); t = time.time()
            r = optimize_design_bayesian(fid, var, obj, cons, seed=seed)
            T.close_flowsheet(fid)
            runs.append({"seed": seed, "status": r["status"], "reasons": r.get("reasons"), "stopped_by": r["stopped_by"]["reason"],
                         "values": r["best"]["values"], "objective": r["best"]["objective"], "constraints": r["best"]["constraints"],
                         "n_evaluations": r["n_evaluations"] + 1, "n_failed": r["n_failed_evaluations"],
                         "rel_gap_pct": 100 * (r["best"]["objective"] - direct["objective"]) / abs(direct["objective"]),
                         "gp_at_optimum": r["gp_at_optimum"], "loo": r["gp_diagnostics"]["objective"]["leave_one_out"],
                         "one_step_ahead": r["gp_diagnostics"].get("one_step_ahead"), "active_constraints": r["active_constraints"],
                         "seconds": round(time.time() - t, 1), "model_seconds": r["suitability"]["model_seconds"],
                         "suitability_notes": r["suitability"]["notes"]})
            print(name, "seed", seed, runs[-1]["status"], runs[-1]["stopped_by"], runs[-1]["n_evaluations"], runs[-1]["values"],
                  round(runs[-1]["objective"], 3), "gap %", round(runs[-1]["rel_gap_pct"], 4), flush=True)
        out[name] = {"direct_nelder_mead": direct, "surrogate_omlt": sur, "bayesian": runs,
                     "bayesian_evaluations_median": float(np.median([r["n_evaluations"] for r in runs])),
                     "bayesian_rel_gap_pct_max": max(r["rel_gap_pct"] for r in runs)}
        print(name, "direct", direct, "\n  surrogate", sur, flush=True)
        _save("dwsim_" + name, out[name])
    return out


def integers(seeds=(0, 1, 2)):
    """Ethanol-water column of notebook 20 (bloque 26): stages 8-60 (INTEGER) × reflux 1-5, total annual cost with
    illustrative prices (MOCKS.md, PRECIOS-ILUSTRATIVOS), x_D ≥ 0.80. Known optimum, recorded in
    notebooks/20_variables_enteras.jsonl by optimize_design_multi (enumeration of the stages, 442 DWSIM solves):
    29 stages, R = 1.394, 885 080 USD/year. The direct run is NOT repeated here (≈ 10 min); its record is read."""
    from packages.dwsim_client import tools as T
    from packages.dwsim_client.tools_optimize import optimize_design_bayesian
    spec = importlib.util.spec_from_file_location("nb20", ROOT / "notebooks" / "20_variables_enteras.py")
    nb20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb20)
    rec = [json.loads(l) for l in (ROOT / "notebooks" / "20_variables_enteras.jsonl").read_text().splitlines()]
    d = next(r for r in rec if r["section"] == "optimum")
    direct = {"values": d["best"]["values"], "objective": d["best"]["objective"], "n_evaluations": d["n_evaluations"],
              "source": "notebooks/20_variables_enteras.jsonl (recorded, not re-run)"}
    var = [{"kind": "unit", "name": "T-101", "param": "stages", "lower": 8, "upper": 60, "integer": True},
           {"kind": "unit", "name": "T-101", "param": "reflux_ratio", "lower": 1.0, "upper": 5.0}]
    cons = [{"kind": "stream", "name": "dist", "key": "mole_fractions.Ethanol", "min": 0.80}]
    print("direct (recorded)", direct, flush=True)
    runs = []
    for seed in seeds:
        fid = nb20.column(); t = time.time()
        r = optimize_design_bayesian(fid, var, nb20.COST, cons, seed=seed)
        T.close_flowsheet(fid)
        runs.append({"seed": seed, "status": r["status"], "reasons": r.get("reasons"), "stopped_by": r["stopped_by"]["reason"],
                     "values": r["best"]["values"], "objective": r["best"]["objective"], "constraints": r["best"]["constraints"],
                     "n_evaluations": r["n_evaluations"] + 1, "n_failed": r["n_failed_evaluations"],
                     "n_infeasible": sum(1 for h in r["history"] if h["solved_ok"] and not h["feasible"]),
                     "rel_gap_pct": 100 * (r["best"]["objective"] - direct["objective"]) / abs(direct["objective"]),
                     "all_integer": all(isinstance(h["values"]["T-101.stages"], int) for h in r["history"]),
                     "stages_tried": sorted({h["values"]["T-101.stages"] for h in r["history"]}), "seconds": round(time.time() - t, 1)})
        print("bayes", runs[-1], flush=True)
    out = {"direct_enumeration": direct, "bayesian": runs}
    _save("integers", out)
    return out


# ----------------------------------------------------------------------------------------------- suggest_experiments
def _snar_campaign(a):
    key, seed, noise_pct, use_log = a
    from packages.bo.experiments import suggest_experiments
    spec = SNAR[key]
    rng = np.random.default_rng(10_000 + seed)
    data, trace = [], []
    names = [v["name"] for v in B.SNAR_VARS]
    best = None
    for rnd in range(SNAR_ROUNDS):
        r = suggest_experiments(B.SNAR_VARS, data, {"name": key, "maximize": spec["maximize"], "log": use_log}, SNAR_BATCH, seed=seed)
        for p in r["proposals"]:
            y = B.snar(*[p["inputs"][n] for n in names], noise_pct=noise_pct, rng=rng)[key]      # the «laboratory»
            data.append(p["inputs"] | {key: y})
            true = B.snar(*[p["inputs"][n] for n in names])[key]                                  # noise-free value, for scoring only
            best = true if best is None else (max(best, true) if spec["maximize"] else min(best, true))
        trace.append(best)
    lo = np.array([v["lower"] for v in B.SNAR_VARS]); hi = np.array([v["upper"] for v in B.SNAR_VARS])
    rr = np.random.default_rng(20_000 + seed)
    rand = [B.snar(*(lo + rr.random(4) * (hi - lo)))[key] for _ in range(SNAR_BATCH * SNAR_ROUNDS)]
    acc = np.maximum.accumulate(rand) if spec["maximize"] else np.minimum.accumulate(rand)
    return {"seed": seed, "bayes_best_by_round": trace, "random_best_by_round": [float(acc[(i + 1) * SNAR_BATCH - 1]) for i in range(SNAR_ROUNDS)],
            "last_mode": r["mode"], "last_q2": (r.get("model") or {}).get("leave_one_out", {}).get(key, {}).get("q2")}


def snar(n_seeds=20, noise_levels=(0.0, 2.0), cases=(("sty", False), ("e_factor", False), ("e_factor", True))):
    """cases = (objective, model log y?). The E-factor spans two decades over the box (8.5 to the cap of 1000)."""
    out = {}
    for key, use_log in cases:
        spec = SNAR[key]
        for noise in noise_levels:
            with Pool(7) as p:
                rows = p.map(_snar_campaign, [(key, s, noise, use_log) for s in range(n_seeds)])
            opt = spec["optimum"]
            gap = lambda v: 100 * abs(v - opt) / abs(opt)
            ok = lambda v: gap(v) <= SNAR_TOL_PCT
            res = {"objective": key, "maximize": spec["maximize"], "log": use_log, "optimum": opt, "noise_pct": noise, "n_seeds": n_seeds,
                   "experiments_per_round": SNAR_BATCH, "rounds": SNAR_ROUNDS, "tolerance_pct": SNAR_TOL_PCT,
                   "bayes_success_by_round": [sum(ok(r["bayes_best_by_round"][i]) for r in rows) for i in range(SNAR_ROUNDS)],
                   "random_success_by_round": [sum(ok(r["random_best_by_round"][i]) for r in rows) for i in range(SNAR_ROUNDS)],
                   "bayes_median_gap_pct_by_round": [float(np.median([gap(r["bayes_best_by_round"][i]) for r in rows])) for i in range(SNAR_ROUNDS)],
                   "random_median_gap_pct_by_round": [float(np.median([gap(r["random_best_by_round"][i]) for r in rows])) for i in range(SNAR_ROUNDS)],
                   "bayes_worst_gap_pct_final": max(gap(r["bayes_best_by_round"][-1]) for r in rows),
                   "random_best_gap_pct_final": min(gap(r["random_best_by_round"][-1]) for r in rows), "runs": rows}
            tag = f"{key}{'_log' if use_log else ''}_noise{noise:g}"
            out[tag] = res
            print(tag, {k: v for k, v in res.items() if k != "runs"}, flush=True)
            _save("snar_" + tag, res)
    return out


def truth():
    from scipy.optimize import differential_evolution
    bounds = [(v["lower"], v["upper"]) for v in B.SNAR_VARS]
    out = {}
    for key, spec in SNAR.items():
        sgn = -1.0 if spec["maximize"] else 1.0
        runs = [differential_evolution(lambda x: sgn * B.snar(*x)[key], bounds, seed=s, tol=1e-10, popsize=30, maxiter=300) for s in range(3)]
        best = min(runs, key=lambda r: r.fun)
        out[key] = {"x": [float(v) for v in best.x], "value": float(sgn * best.fun), "all_three_starts": [float(sgn * r.fun) for r in runs]}
        print(key, out[key], flush=True)
    _save("snar_truth_differential_evolution", out)
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "functions"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    {"functions": lambda: functions(n), "dwsim": dwsim, "integers": integers, "snar": lambda: snar(n), "truth": truth}[arg]()
