"""Neural-network surrogate of a DWSIM flowsheet, embedded in Pyomo with OMLT and optimised with ipopt (bloque M8c).

Pipeline (one call):
  1. Design of experiments over the user's bounds: Latin hypercube (scipy.stats.qmc, fixed seed) plus the 2^n corners
     of the box (n ≤ 4), in normalised coordinates z ∈ [0, 1]^n (log scale when the variable asks for it). An
     INDEPENDENT Latin hypercube of n_holdout points is the held-out set: it is never used to train or to choose the
     network, only to measure the error.
  2. Every point is solved in DWSIM (the evaluator); points DWSIM cannot solve are dropped and counted.
  3. One small network per output (objective and each constrained readout): tanh hidden layers, linear output, inputs
     2z − 1, standardised outputs folded back into the last layer, L2 penalty. Architecture and penalty are chosen by
     5-fold cross-validation on the training set only; weights are fitted by L-BFGS (scipy) with analytic gradients.
     Plain numpy: no torch/keras/sklearn (not in the env, and not needed for networks this small).
  4. Each network becomes an OMLT `NetworkDefinition` (DenseLayer, tanh) with
     `ReducedSpaceSmoothNNFormulation`: the outputs are smooth closed-form expressions of the n decision variables, so
     ipopt gets exact first and second derivatives and needs no initial values for hidden neurons. With a few dozen
     neurons a full-space formulation only adds variables and equalities. A ReLU network as a MILP (cbc from
     idaes-ext is free) would give the global optimum of a piecewise-linear surrogate, but ReLU needs many more DWSIM
     samples for the same smooth accuracy; the global search is covered instead by multistart from the best sampled
     points.
  5. ipopt solves min/max objective s.t. constraint outputs within the user's limits and z ∈ [0, 1]^n.
  6. DWSIM runs at the proposed optimum. The surrogate is never trusted: `verified` needs the held-out error and the
     error at the optimum within tolerance, the DWSIM constraints met within tolerance, and no DWSIM sample that is
     feasible and better than the DWSIM value at the optimum by more than the tolerance.

The surrogate is valid only inside the bounds and only where DWSIM solves; this is stated in `assumptions`.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Any, Callable

import numpy as np

ARCHITECTURES = ((6,), (10,), (8, 8))
PENALTIES = (1e-6, 1e-4)
CV_FOLDS = 5
MAX_STARTS = 6
FORMULATION_WHY = ("OMLT ReducedSpaceSmoothNNFormulation (tanh): las salidas son expresiones suaves de las n variables "
                   "de decisión, ipopt recibe derivadas exactas y no necesita valores iniciales de neuronas ocultas; "
                   "con redes de decenas de neuronas el espacio completo solo agrega variables. ReLU + MILP (cbc es "
                   "gratis) daría el óptimo global de un sustituto lineal a trozos, pero ReLU pide muchas más corridas "
                   "de DWSIM para la misma precisión; la búsqueda global se cubre con arranques múltiples.")


# ----------------------------------------------------------------------------------------------- design of experiments
def design(n_vars: int, n_train: int, n_holdout: int, seed: int = 0, corners: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Training points (Latin hypercube + box corners) and an independent held-out Latin hypercube, both in [0, 1]^n."""
    from scipy.stats import qmc
    train = qmc.LatinHypercube(d=n_vars, seed=seed).random(n_train)
    if corners and n_vars <= 4:
        train = np.vstack([train, np.array(list(itertools.product((0.0, 1.0), repeat=n_vars)))])
    hold = qmc.LatinHypercube(d=n_vars, seed=seed + 1000).random(n_holdout)
    return train, hold


def to_physical(z: np.ndarray, variables: list[dict]) -> np.ndarray:
    z = np.atleast_2d(z)
    out = np.empty_like(z, dtype=float)
    for j, v in enumerate(variables):
        lo, hi = float(v["lower"]), float(v["upper"])
        out[:, j] = (np.exp(math.log(lo) + z[:, j] * (math.log(hi) - math.log(lo))) if v.get("log_scale")
                     else lo + z[:, j] * (hi - lo))
    return out


# ----------------------------------------------------------------------------------------------------- the network
class TanhNet:
    """Dense network u = 2z − 1 → tanh hidden layers → linear output, output de-standardised (y = μ + σ·out)."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray], mu: float, sd: float):
        self.weights, self.biases, self.mu, self.sd = weights, biases, float(mu), float(sd)

    def predict(self, z: np.ndarray) -> np.ndarray:
        a = 2.0 * np.atleast_2d(z) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.tanh(a @ W + b)
        return self.mu + self.sd * (a @ self.weights[-1] + self.biases[-1])[:, 0]

    def folded_layers(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Weights acting on z directly (input map 2z − 1 folded into layer 1, output scaling into the last layer)."""
        Ws = [w.copy() for w in self.weights]
        bs = [b.copy() for b in self.biases]
        bs[0] = bs[0] - Ws[0].sum(axis=0)
        Ws[0] = 2.0 * Ws[0]
        Ws[-1] = self.sd * Ws[-1]
        bs[-1] = self.mu + self.sd * bs[-1]
        return list(zip(Ws, bs))

    @property
    def size(self) -> list[int]:
        return [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]


def _shapes(n_in: int, hidden: tuple[int, ...]) -> list[tuple[int, int]]:
    sizes = [n_in, *hidden, 1]
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def _unpack(p: np.ndarray, shapes):
    Ws, bs, k = [], [], 0
    for (a, b) in shapes:
        Ws.append(p[k:k + a * b].reshape(a, b)); k += a * b
        bs.append(p[k:k + b]); k += b
    return Ws, bs


def _loss_grad(p, shapes, U, t, alpha):
    Ws, bs = _unpack(p, shapes)
    acts = [U]
    for W, b in zip(Ws[:-1], bs[:-1]):
        acts.append(np.tanh(acts[-1] @ W + b))
    out = (acts[-1] @ Ws[-1] + bs[-1])[:, 0]
    r = out - t
    n = len(t)
    loss = float(r @ r) / n + alpha * sum(float((W * W).sum()) for W in Ws)
    g = (2.0 / n) * r[:, None]
    gW, gb = [None] * len(Ws), [None] * len(Ws)
    for i in range(len(Ws) - 1, -1, -1):
        gW[i] = acts[i].T @ g + 2.0 * alpha * Ws[i]
        gb[i] = g.sum(axis=0)
        if i:
            g = (g @ Ws[i].T) * (1.0 - acts[i] ** 2)
    return loss, np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(gW, gb)])


def _fit_once(Z, y, hidden, alpha, seed) -> TanhNet:
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
    t = (y - mu) / sd
    U = 2.0 * Z - 1.0
    shapes = _shapes(Z.shape[1], hidden)
    p0 = np.concatenate([np.concatenate([rng.normal(0.0, 1.0 / math.sqrt(a), a * b), np.zeros(b)]) for a, b in shapes])
    res = minimize(_loss_grad, p0, args=(shapes, U, t, alpha), jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "gtol": 1e-10})
    Ws, bs = _unpack(res.x, shapes)
    return TanhNet(Ws, bs, mu, sd)


def fit_surrogate(Z: np.ndarray, y: np.ndarray, seed: int = 0, restarts: int = 3) -> tuple[TanhNet, dict]:
    """Choose architecture × L2 penalty by 5-fold CV on (Z, y), then refit on all of it (best of `restarts` inits)."""
    n = len(y)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), min(CV_FOLDS, n))
    scores = {}
    for hidden, alpha in itertools.product(ARCHITECTURES, PENALTIES):
        if sum(a * b + b for a, b in _shapes(Z.shape[1], hidden)) > 4 * n:      # far more weights than data: skip
            continue
        err = []
        for k, f in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), f)
            net = _fit_once(Z[tr], y[tr], hidden, alpha, seed + k)
            err.append(net.predict(Z[f]) - y[f])
        scores[(hidden, alpha)] = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
    (hidden, alpha), cv = min(scores.items(), key=lambda kv: kv[1])
    best, best_rmse = None, math.inf
    for r in range(restarts):
        net = _fit_once(Z, y, hidden, alpha, seed + 100 + r)
        rmse = float(np.sqrt(np.mean((net.predict(Z) - y) ** 2)))
        if rmse < best_rmse:
            best, best_rmse = net, rmse
    return best, {"hidden_layers": list(hidden), "l2_penalty": alpha, "cv_rmse": cv, "train_rmse": best_rmse,
                  "n_weights": sum(a * b + b for a, b in _shapes(Z.shape[1], hidden))}


# ---------------------------------------------------------------------------------------------------- OMLT + ipopt
def network_definition(net: TanhNet):
    from omlt.neuralnet import NetworkDefinition
    from omlt.neuralnet.layer import DenseLayer, InputLayer
    n = net.size[0]
    nd = NetworkDefinition(scaled_input_bounds={i: (0.0, 1.0) for i in range(n)})
    prev = InputLayer([n])
    nd.add_layer(prev)
    layers = net.folded_layers()
    for i, (W, b) in enumerate(layers):
        act = "linear" if i == len(layers) - 1 else "tanh"
        d = DenseLayer(prev.output_size, [W.shape[1]], weights=W, biases=b, activation=act)
        nd.add_layer(d)
        nd.add_edge(prev, d)
        prev = d
    return nd


def build_model(nets: dict[str, TanhNet], objective_key: str, limits: dict[str, dict], maximize: bool):
    """Pyomo model: z ∈ [0,1]^n, one OMLT block per network sharing z, objective and constraint limits."""
    import pyomo.environ as pyo
    from omlt import OmltBlock
    from omlt.neuralnet import ReducedSpaceSmoothNNFormulation
    n = next(iter(nets.values())).size[0]
    m = pyo.ConcreteModel()
    m.z = pyo.Var(range(n), bounds=(0.0, 1.0), initialize=0.5)
    m.nn = pyo.Block(list(nets))
    m.link = pyo.ConstraintList()
    for key, net in nets.items():
        b = m.nn[key]
        b.omlt = OmltBlock()
        b.omlt.build_formulation(ReducedSpaceSmoothNNFormulation(network_definition(net)))
        for i in range(n):
            m.link.add(b.omlt.inputs[i] == m.z[i])
    out = lambda key: m.nn[key].omlt.outputs[0]
    m.obj = pyo.Objective(expr=out(objective_key), sense=pyo.maximize if maximize else pyo.minimize)
    m.limits = pyo.ConstraintList()
    for key, lim in limits.items():
        if lim.get("max") is not None:
            m.limits.add(out(key) <= float(lim["max"]))
        if lim.get("min") is not None:
            m.limits.add(out(key) >= float(lim["min"]))
    return m


def _set_start(m, nets, z0):
    for i, v in enumerate(z0):
        m.z[i].set_value(float(v))
    for key, net in nets.items():
        b = m.nn[key].omlt
        for i, v in enumerate(z0):
            b.inputs[i].set_value(float(v))
        b.outputs[0].set_value(float(net.predict(np.array([z0]))[0]))


def solve_surrogate(nets, objective_key, limits, maximize, starts: list[np.ndarray]) -> dict:
    """ipopt from each start; returns the best locally optimal point and how many starts reached it. Embedding check:
    at each solution the OMLT output variables (constrained to the network expression) must equal the numpy forward
    pass of the same network."""
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition as TC
    from packages.eo.column_optimize import _solver
    m = build_model(nets, objective_key, limits, maximize)
    solver = _solver()
    found = []
    for z0 in starts:
        _set_start(m, nets, z0)
        res = solver.solve(m, tee=False, load_solutions=False)
        if res.solver.termination_condition in (TC.optimal, TC.locallyOptimal):
            m.solutions.load_from(res)
            z = np.clip(np.array([pyo.value(m.z[i]) for i in m.z]), 0.0, 1.0)
            emb = max(abs(pyo.value(m.nn[k].omlt.outputs[0]) - float(net.predict(np.array([z]))[0]))
                      / max(abs(float(net.predict(np.array([z]))[0])), 1e-12) for k, net in nets.items())
            found.append({"z": z, "objective": float(pyo.value(m.obj)), "embedding": emb})
    if not found:
        return {"ok": False, "n_starts": len(starts), "n_converged": 0}
    sgn = -1.0 if maximize else 1.0
    best = min(found, key=lambda f: sgn * f["objective"])
    same = [f for f in found if np.max(np.abs(f["z"] - best["z"])) <= 1e-3]
    return {"ok": True, "z": best["z"], "objective": best["objective"], "n_starts": len(starts), "n_converged": len(found),
            "n_converged_to_best": len(same), "embedding_max_rel_diff": best["embedding"]}


# ------------------------------------------------------------------------------------------------------------ driver
Evaluator = Callable[[list[float]], dict]   # x (physical) → {"solved_ok", "objective", "constraints": {key: value}}


def _tolerances(objective_tolerance, tolerance_pct, limits, y_obj):
    tol_obj = (float(objective_tolerance) if objective_tolerance is not None
               else tolerance_pct / 100.0 * float(np.median(np.abs(y_obj))))
    tol_c = {k: float(lim["tolerance"]) if lim.get("tolerance") is not None else
             tolerance_pct / 100.0 * abs(float(lim["max"] if lim.get("max") is not None else lim["min"]))
             for k, lim in limits.items()}
    return tol_obj, tol_c


def surrogate_optimize(evaluate: Evaluator, variables: list[dict], limits: dict[str, dict], maximize: bool = False,
                       n_samples: int | None = None, n_holdout: int | None = None, tolerance_pct: float = 0.5,
                       objective_tolerance: float | None = None, seed: int = 0, recorded: dict | None = None,
                       return_samples: bool = False) -> dict:
    """Core of optimize_design_surrogate with an injected evaluator (DWSIM in the tool; recorded data or an analytic
    function in the tests). limits = {constraint_key: {"max"|"min": value, "tolerance": optional absolute}}.
    `recorded` = {"train": [[x..., ], ...], "train_rows": [...], ...} replays stored samples instead of evaluating them."""
    t0 = time.time()
    n = len(variables)
    if not 1 <= n <= 6:
        raise ValueError("1 a 6 variables de decisión")
    for v in variables:
        if v.get("integer"):
            raise ValueError("variables enteras: usa optimize_design_multi (el sustituto es para variables continuas)")
        if not float(v["lower"]) < float(v["upper"]):
            raise ValueError(f"cota inferior < superior en {v['name']}.{v['param']}")
        if v.get("log_scale") and float(v["lower"]) <= 0:
            raise ValueError(f"escala log necesita cotas > 0 en {v['name']}.{v['param']}")
    n_train = int(n_samples) if n_samples else 20 + 15 * n
    n_hold = int(n_holdout) if n_holdout else max(10, n_train // 4)
    sample_count_assumed = not n_samples
    Zt, Zh = design(n, n_train, n_hold, seed)
    keys = list(limits)

    def run(Z, rows=None):
        rows = rows if rows is not None else [evaluate(list(x)) for x in to_physical(Z, variables)]
        ok = [i for i, r in enumerate(rows) if r.get("solved_ok") and r.get("objective") is not None
              and all((r.get("constraints") or {}).get(k) is not None for k in keys)]
        Y = {"objective": np.array([rows[i]["objective"] for i in ok], float)}
        for k in keys:
            Y[k] = np.array([rows[i]["constraints"][k] for i in ok], float)
        return Z[ok], Y, rows, len(rows) - len(ok)

    if recorded:
        Zt, Zh = np.array(recorded["train_z"]), np.array(recorded["holdout_z"])
    Zt_all, Zh_all = Zt.copy(), Zh.copy()
    if recorded:
        Zt, Yt, rows_t, fail_t = run(Zt, recorded["train_rows"])
        Zh, Yh, rows_h, fail_h = run(Zh, recorded["holdout_rows"])
    else:
        Zt, Yt, rows_t, fail_t = run(Zt)
        Zh, Yh, rows_h, fail_h = run(Zh)
    if len(Zt) < 5 * n or len(Zh) < 3:
        return {"status": "not_verified", "verified": False, "n_dwsim_solves": len(rows_t) + len(rows_h),
                "note": f"DWSIM resolvió muy pocos puntos ({len(Zt)} de entrenamiento, {len(Zh)} apartados): estrecha las cotas"}

    nets, info = {}, {}
    for k, y in Yt.items():
        nets[k], info[k] = fit_surrogate(Zt, y, seed)

    tol_obj, tol_c = _tolerances(objective_tolerance, tolerance_pct, limits, Yt["objective"])
    tol = {"objective": tol_obj} | tol_c
    hold = {}
    for k, net in nets.items():
        e = net.predict(Zh) - Yh[k]
        hold[k] = {"max_abs_error": float(np.max(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                   "max_rel_error_pct": float(100 * np.max(np.abs(e) / np.maximum(np.abs(Yh[k]), 1e-12))),
                   "max_error_pct_of_sampled_range": float(100 * np.max(np.abs(e)) / max(float(np.ptp(Yt[k])), 1e-12)),
                   "tolerance_abs": tol[k], "within": bool(np.max(np.abs(e)) <= tol[k])}

    sgn = -1.0 if maximize else 1.0
    feas = lambda Y, i: all((lim.get("max") is None or Y[k][i] <= lim["max"]) and (lim.get("min") is None or Y[k][i] >= lim["min"])
                            for k, lim in limits.items())
    Zall = np.vstack([Zt, Zh])
    Yall = {k: np.concatenate([Yt[k], Yh[k]]) for k in Yt}
    idx_feas = [i for i in range(len(Zall)) if feas(Yall, i)]
    ranked = sorted(idx_feas, key=lambda i: sgn * Yall["objective"][i])
    starts = [Zall[i] for i in ranked[:MAX_STARTS - 1]] + [np.full(n, 0.5)]
    sol = solve_surrogate(nets, "objective", limits, maximize, starts)
    best_sample = None
    if ranked:
        i = ranked[0]
        best_sample = {"values": _named(variables, to_physical(Zall[i], variables)[0]), "objective": float(Yall["objective"][i]),
                       "constraints": {k: float(Yall[k][i]) for k in keys}}
    base = {"method": {
        "design": f"hipercubo latino (scipy.stats.qmc, semilla {seed}) + {2 ** n if n <= 4 else 0} esquinas; apartado = otro hipercubo latino independiente",
        "n_train_requested": n_train + (2 ** n if n <= 4 else 0), "n_holdout_requested": n_hold,
        "n_train_used": int(len(Zt)), "n_holdout_used": int(len(Zh)), "n_failed_solves": int(fail_t + fail_h),
        "networks": info, "formulation": FORMULATION_WHY, "solver": "ipopt (idaes-ext), arranques múltiples desde las mejores muestras factibles + centro"},
        "holdout_error": hold, "best_dwsim_sample": best_sample, "n_dwsim_solves": len(rows_t) + len(rows_h)}
    assumptions = [
        f"Cotas de las variables = las del usuario; el sustituto solo es válido dentro de ellas y donde DWSIM resuelve.",
        f"Muestras: {n_train} de hipercubo latino{' (número por defecto 20 + 15·n, supuesto del método)' if sample_count_assumed else ''}"
        f" + esquinas; {n_hold} apartadas{' (por defecto max(10, n/4))' if not n_holdout else ''}.",
        f"Tolerancia: {'objetivo ±' + format(tol_obj, '.4g') + ' (dada por el usuario)' if objective_tolerance is not None else f'{tolerance_pct} % de la mediana del objetivo (±{tol_obj:.4g})'}; "
        + "; ".join(f"{k} ±{v:.4g}" + (" (dada)" if limits[k].get("tolerance") is not None else f" ({tolerance_pct} % del límite)") for k, v in tol_c.items()),
        "Óptimo local de ipopt sobre el sustituto; varios arranques reducen pero no eliminan el riesgo de un óptimo local.",
    ]
    if not sol["ok"]:
        return base | {"status": "not_verified", "verified": False, "assumptions": assumptions, "elapsed_s": round(time.time() - t0, 1),
                       "note": "ipopt no encontró un óptimo del sustituto (límites inalcanzables dentro de las cotas)"}

    x_opt = to_physical(sol["z"], variables)[0]
    sur = {"objective": sol["objective"]} | {k: float(nets[k].predict(np.array([sol["z"]]))[0]) for k in keys}
    dw = evaluate(list(x_opt))
    at_bound = [f"{v['name']}.{v['param']}" for v, z in zip(variables, sol["z"]) if min(z, 1 - z) <= 1e-3]
    out = base | {"surrogate_optimum": {"values": _named(variables, x_opt), "objective": sur["objective"],
                                        "constraints": {k: sur[k] for k in keys}},
                  "ipopt": {k: sol[k] for k in ("n_starts", "n_converged", "n_converged_to_best")},
                  "variables_at_bound": at_bound or None, "assumptions": assumptions}
    checks = {"holdout_within_tolerance": all(h["within"] for h in hold.values()),
              "embedding_matches_network": bool(sol["embedding_max_rel_diff"] <= 1e-6),
              "dwsim_solved_at_optimum": bool(dw.get("solved_ok")) and dw.get("objective") is not None}
    if checks["dwsim_solved_at_optimum"]:
        dwc = dw.get("constraints") or {}
        out["dwsim_at_optimum"] = {"objective": float(dw["objective"]), "constraints": {k: float(dwc[k]) for k in keys}}
        err = {"objective": {"abs_error": abs(dw["objective"] - sur["objective"]),
                             "rel_error_pct": 100 * abs(dw["objective"] - sur["objective"]) / max(abs(dw["objective"]), 1e-12),
                             "tolerance_abs": tol_obj}}
        for k in keys:
            err[k] = {"abs_error": abs(dwc[k] - sur[k]), "rel_error_pct": 100 * abs(dwc[k] - sur[k]) / max(abs(dwc[k]), 1e-12),
                      "tolerance_abs": tol_c[k]}
        for e in err.values():
            e["within"] = bool(e["abs_error"] <= e["tolerance_abs"])
        out["error_at_optimum"] = err
        viol = {k: float(max(0.0, (dwc[k] - lim["max"]) if lim.get("max") is not None else 0.0,
                       (lim["min"] - dwc[k]) if lim.get("min") is not None else 0.0)) for k, lim in limits.items()}
        out["dwsim_constraint_violation"] = viol
        checks["error_at_optimum_within_tolerance"] = all(e["within"] for e in err.values())
        checks["dwsim_constraints_met_within_tolerance"] = all(viol[k] <= tol_c[k] for k in keys)
        checks["no_sample_better_than_optimum"] = bool(best_sample is None or
                                                       sgn * (best_sample["objective"] - dw["objective"]) >= -tol_obj)
    out["embedding_max_rel_diff"] = float(sol["embedding_max_rel_diff"])
    if return_samples:
        out["samples"] = {"train_z": Zt_all.tolist(), "train_rows": rows_t, "holdout_z": Zh_all.tolist(), "holdout_rows": rows_h}
    verified = all(checks.values()) and len(checks) == 6
    out["n_dwsim_solves"] = base["n_dwsim_solves"] + 1
    out["evidence"] = checks | {"verified": verified}
    out["verified"] = verified
    out["status"] = "verified" if verified else "not_verified"
    failed = [k for k, v in checks.items() if not v]
    out["note"] = ("óptimo del sustituto confirmado en DWSIM dentro de la tolerancia" + ("; variable en la cota: amplía el rango" if at_bound else "")
                   if verified else "NO verificado: " + ", ".join(failed) + ". No confíes en el sustituto: más muestras, cotas más "
                   "estrechas alrededor del óptimo, o optimize_design_multi directo sobre DWSIM.")
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _named(variables, x):
    return {f"{v['name']}.{v['param']}": float(xi) for v, xi in zip(variables, x)}
