"""Bayesian optimisation with a Gaussian-process surrogate (bloque M16).

Method, one iteration:
  1. Variables are mapped to z ∈ [0, 1]^n (log scale when the variable asks for it). Integer variables are snapped to
     their integer levels BEFORE the model sees them, in the data and in every candidate (the transformation of
     Garrido-Merchán & Hernández-Lobato, Neurocomputing 380 (2020) 20-35: the acquisition is evaluated only at valid
     integer points, so the optimiser never proposes a non-integer value or the same integer twice by rounding).
  2. One Gaussian process per output (objective and every constrained readout): scikit-learn
     `GaussianProcessRegressor`, kernel constant × Matérn 5/2 with one length scale per variable (ARD) + white noise,
     outputs standardised, hyper-parameters by maximum marginal likelihood with restarts (Rasmussen & Williams,
     «Gaussian Processes for Machine Learning», MIT Press 2006, ch. 2, 4, 5).
  3. Points where the engine FAILED (no convergence, exception) carry no outputs. They are kept, in two ways: a
     `GaussianProcessClassifier` (Laplace approximation, R&W ch. 3) learns the probability that the engine solves and
     the acquisition is multiplied by it; and in the objective model each failed point is imputed with a pessimistic
     prediction (mean + 1 sd on the bad side, after Forrester, Sóbester & Keane, Proc. R. Soc. A 462 (2006) 935-945),
     so the uncertainty, and with it the expected improvement, collapses where the engine fails. Without the second
     the search keeps returning to the failing region (measured: 43-52 failed evaluations of 60 on a Branin with three
     failing zones). A failed evaluation never aborts the run.
  4. Acquisition = expected improvement × Π probability of feasibility × probability that the engine solves
     (EI: Jones, Schonlau & Welch, J. Global Optim. 13 (1998) 455-492; constrained EI: Schonlau, Welch & Jones 1998;
     Gardner et al., ICML 2014, PMLR 32(2):937-945; Gelbart, Snoek & Adams, UAI 2014). Until a feasible point exists
     the acquisition is the probability of feasibility alone. It is computed in logarithms with the asymptotic
     expansion of EI far below the incumbent (the numerical issue of Ament et al., NeurIPS 2023, «Unexpected
     improvements to expected improvement»), so it stays informative where plain EI underflows to zero.
  5. The acquisition is maximised over a scrambled Sobol set plus Gaussian perturbations of the best points, and the
     best candidates are polished with L-BFGS-B on the continuous variables.
  6. Stopping is explicit and reported: evaluation budget, max expected improvement below the objective tolerance for
     `ei_patience` consecutive iterations, or no improvement larger than the tolerance in `stall_iterations`.
  7. The best feasible point is RE-EVALUATED in the engine. The Gaussian process is never the answer.

Diagnostics of the model: closed-form leave-one-out prediction (R&W eq. 5.10-5.12, hyper-parameters held fixed) and
the one-step-ahead record (what the model predicted at each proposed point before the engine evaluated it).

The same model and acquisition serve `suggest_experiments` (packages/bo/experiments.py), with fitted noise and a
batch built by the Kriging-believer heuristic (Ginsbourger, Le Riche & Carraro 2010).
"""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable

import numpy as np
from scipy.special import ndtr, log_ndtr

Evaluator = Callable[[list[float]], dict]

# Algorithmic defaults (every one is listed in MOCKS.md, sección M16)
MAX_VARIABLES = 10                    # above this a GP with one length scale per variable has too little data
MATERN_NU = 2.5
LENGTH_SCALE_BOUNDS = (0.05, 20.0)    # in normalised [0, 1] coordinates
SIGNAL_VARIANCE_BOUNDS = (1e-2, 1e3)  # of the standardised output
NOISE_BOUNDS_DETERMINISTIC = (1e-10, 1e-5)   # variance of the standardised output: a simulator repeats itself
NOISE_BOUNDS_NOISY = (1e-6, 1.0)             # measurements: the noise is fitted
GP_RESTARTS = 2
N_SOBOL = 2048
N_LOCAL = 256
LOCAL_SCALES = (0.1, 0.03, 0.01, 0.003)
N_POLISH = 5
DUPLICATE_DISTANCE = 1e-6
EI_PATIENCE = 5
EI_STOP_FRACTION = 0.1                # stop when max EI < this fraction of the objective tolerance ...
MIN_MODEL_ITERATIONS = lambda n: max(10, 5 * n)   # ... but never before this many model-guided evaluations
ENGINE = "scikit-learn GaussianProcessRegressor"


def engine_info() -> dict:
    import sklearn
    return {"engine": ENGINE, "version": sklearn.__version__, "license": "BSD-3-Clause",
            "kernel": f"constante × Matérn ν = {MATERN_NU} (una escala de longitud por variable) + ruido blanco",
            "acquisition": "mejora esperada × probabilidad de factibilidad × probabilidad de que el motor resuelva (en logaritmos)"}


# ------------------------------------------------------------------------------------------------------- the space
class Space:
    """Bounds, log scale and integer levels of the decision variables; z ∈ [0, 1]^n ↔ physical x."""

    def __init__(self, variables: list[dict], label: Callable[[dict], str]):
        if not 1 <= len(variables) <= MAX_VARIABLES:
            raise ValueError(f"1 a {MAX_VARIABLES} variables de decisión (con más, un proceso gaussiano no tiene datos "
                             "suficientes por dimensión: reduce variables con un análisis de sensibilidad)")
        self.n = len(variables)
        self.names = [label(v) for v in variables]
        if len(set(self.names)) != self.n:
            raise ValueError("variables repetidas")
        self.lo, self.hi, self.log, self.int = [], [], [], []
        for v, name in zip(variables, self.names):
            if v.get("lower") is None or v.get("upper") is None:
                raise ValueError(f"faltan las cotas de {name}: las cotas son del usuario, no hay valores por defecto")
            lo, hi = float(v["lower"]), float(v["upper"])
            if not lo < hi:
                raise ValueError(f"cota inferior < superior en {name}")
            lg, it = bool(v.get("log_scale")), bool(v.get("integer"))
            if lg and lo <= 0:
                raise ValueError(f"escala log necesita cotas > 0 en {name}")
            if it and lg:
                raise ValueError(f"{name}: entera y en escala log a la vez no está soportado")
            if it:
                lo, hi = float(math.ceil(lo - 1e-9)), float(math.floor(hi + 1e-9))
                if not lo < hi:
                    raise ValueError(f"{name}: el rango entero necesita al menos dos valores")
            self.lo.append(lo); self.hi.append(hi); self.log.append(lg); self.int.append(it)
        self.cont = [j for j in range(self.n) if not self.int[j]]
        self.n_integer_points = (int(np.prod([self.hi[j] - self.lo[j] + 1 for j in range(self.n)]))
                                 if not self.cont else None)

    def to_x(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, float))
        x = np.empty_like(z)
        for j in range(self.n):
            lo, hi = self.lo[j], self.hi[j]
            if self.log[j]:
                x[:, j] = np.exp(math.log(lo) + z[:, j] * (math.log(hi) - math.log(lo)))
            else:
                x[:, j] = lo + z[:, j] * (hi - lo)
            if self.int[j]:
                x[:, j] = np.clip(np.round(x[:, j]), lo, hi)
        return x

    def to_z(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        z = np.empty_like(x)
        for j in range(self.n):
            lo, hi = self.lo[j], self.hi[j]
            z[:, j] = ((np.log(x[:, j]) - math.log(lo)) / (math.log(hi) - math.log(lo)) if self.log[j]
                       else (x[:, j] - lo) / (hi - lo))
        return z

    def snap(self, z: np.ndarray) -> np.ndarray:
        """Clip to the box and move integer coordinates onto their levels."""
        z = np.clip(np.atleast_2d(np.asarray(z, float)), 0.0, 1.0)
        if len(self.cont) < self.n:
            z = self.to_z(self.to_x(z))
        return z

    def named(self, x) -> dict:
        return {k: (int(round(float(v))) if it else float(v)) for k, v, it in zip(self.names, np.ravel(x), self.int)}


def initial_design(space: Space, n_points: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points (Owen scrambling, scipy.stats.qmc), snapped to the integer levels, without repeats."""
    from scipy.stats import qmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # Sobol balance warning when n is not a power of two
        z = space.snap(qmc.Sobol(d=space.n, scramble=True, seed=seed).random(max(n_points * 4, 8)))
    return _dedupe(z, np.empty((0, space.n)))[:n_points]


def _dedupe(z: np.ndarray, existing: np.ndarray) -> np.ndarray:
    keep: list[np.ndarray] = []
    ref = [e for e in existing]
    for row in z:
        if all(np.max(np.abs(row - r)) > DUPLICATE_DISTANCE for r in ref):
            keep.append(row); ref.append(row)
    return np.array(keep).reshape(-1, z.shape[1])


# ------------------------------------------------------------------------------------------------------- the model
def fit_gp(Z: np.ndarray, y: np.ndarray, noisy: bool, seed: int, theta0: np.ndarray | None = None,
           restarts: int = GP_RESTARTS, optimise: bool = True):
    """Gaussian process on standardised y. `theta0` warm-starts the hyper-parameters (log scale, sklearn's theta);
    with optimise=False the hyper-parameters stay at theta0 (used by the Kriging-believer batch)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    n = Z.shape[1]
    nb = NOISE_BOUNDS_NOISY if noisy else NOISE_BOUNDS_DETERMINISTIC
    kernel = (ConstantKernel(1.0, SIGNAL_VARIANCE_BOUNDS)
              * Matern(length_scale=np.full(n, 0.5), length_scale_bounds=LENGTH_SCALE_BOUNDS, nu=MATERN_NU)
              + WhiteKernel(1e-2 if noisy else 1e-8, nb))
    if theta0 is not None:
        kernel = kernel.clone_with_theta(np.clip(theta0, kernel.bounds[:, 0], kernel.bounds[:, 1]))
    gp = GaussianProcessRegressor(kernel, alpha=1e-10, normalize_y=True, random_state=seed,
                                  n_restarts_optimizer=restarts if optimise else 0,
                                  optimizer="fmin_l_bfgs_b" if optimise else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # hyper-parameter at a bound: reported in describe_gp instead
        gp.fit(Z, np.asarray(y, float))
    return gp


def predict(gp, Z: np.ndarray, latent: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation in the units of y. latent=True removes the fitted noise variance (the
    uncertainty of the underlying function, which is what the acquisition needs)."""
    mu, sd = gp.predict(np.atleast_2d(Z), return_std=True)
    if latent:
        noise = float(gp.kernel_.k2.noise_level) * float(gp._y_train_std) ** 2
        sd = np.sqrt(np.maximum(sd ** 2 - noise, 1e-300))
    return mu, np.maximum(sd, 1e-150)


def describe_gp(gp, names: list[str]) -> dict:
    k = gp.kernel_
    ls = np.atleast_1d(k.k1.k2.length_scale)
    at = [n for n, l in zip(names, ls) if l <= LENGTH_SCALE_BOUNDS[0] * 1.01 or l >= LENGTH_SCALE_BOUNDS[1] * 0.99]
    return {"length_scales_normalised": {n: float(l) for n, l in zip(names, ls)},
            "signal_variance_standardised": float(k.k1.k1.constant_value),
            "noise_sd": float(math.sqrt(k.k2.noise_level) * gp._y_train_std),
            "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
            "length_scales_at_bound": at or None}


def loo(gp) -> dict:
    """Closed-form leave-one-out predictions with the hyper-parameters held at their fitted values (Rasmussen &
    Williams 2006, eq. 5.10-5.12): μ₋ᵢ = yᵢ − [K⁻¹y]ᵢ / [K⁻¹]ᵢᵢ, σ²₋ᵢ = 1 / [K⁻¹]ᵢᵢ (σ includes the fitted noise)."""
    from scipy.linalg import solve_triangular
    L = gp.L_
    Linv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    d = np.sum(Linv ** 2, axis=0)                 # diag(K⁻¹)
    a = np.ravel(gp.alpha_)
    s, m = float(gp._y_train_std), float(gp._y_train_mean)
    y = np.ravel(gp.y_train_) * s + m
    mu = y - (a / d) * s
    sd = np.sqrt(1.0 / d) * s
    return {"y": y, "mean": mu, "sd": sd}


def loo_summary(gp) -> dict:
    r = loo(gp)
    e = r["mean"] - r["y"]
    ss = float(np.sum((r["y"] - r["y"].mean()) ** 2))
    zs = e / r["sd"]
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs_error": float(np.max(np.abs(e))),
            "q2": float(1.0 - np.sum(e ** 2) / ss) if ss > 0 else None,
            "rmse_pct_of_range": float(100 * np.sqrt(np.mean(e ** 2)) / np.ptp(r["y"])) if np.ptp(r["y"]) > 0 else None,
            "fraction_within_2_sd": float(np.mean(np.abs(zs) <= 2.0)),
            "note": "dejar-uno-fuera en forma cerrada con los hiperparámetros fijos (Rasmussen y Williams 2006, ec. 5.10-5.12)"}


def fit_solve_classifier(Z: np.ndarray, ok: np.ndarray, seed: int):
    """P(engine solves | z); None while every point solved (or every point failed)."""
    ok = np.asarray(ok, bool)
    if ok.all() or not ok.any():
        return None
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=0.3, length_scale_bounds=LENGTH_SCALE_BOUNDS, nu=MATERN_NU)
    clf = GaussianProcessClassifier(kernel, random_state=seed, n_restarts_optimizer=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Z, ok.astype(int))
    return clf


# -------------------------------------------------------------------------------------------------- the acquisition
def log_h(u: np.ndarray) -> np.ndarray:
    """log(φ(u) + u·Φ(u)): EI = σ·h(u). Direct for u > −6, asymptotic series φ(u)/u²·(1 − 3/u² + 15/u⁴ − 105/u⁶) below."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    hi = u > -6.0
    uh = u[hi]
    out[hi] = np.log(np.maximum(np.exp(-0.5 * uh * uh) / math.sqrt(2 * math.pi) + uh * ndtr(uh), 1e-300))
    v = u[~hi]
    out[~hi] = (-0.5 * v * v - 0.5 * math.log(2 * math.pi) - 2.0 * np.log(-v)
                + np.log1p(-3.0 / v ** 2 + 15.0 / v ** 4 - 105.0 / v ** 6))
    return out


class Models:
    """The fitted objective GP, one GP per constrained output and the solve classifier, with the acquisition."""

    def __init__(self, obj, cons: dict, clf, limits: dict[str, dict], sgn: float, incumbent: float | None):
        self.obj, self.cons, self.clf, self.limits, self.sgn, self.incumbent = obj, cons, clf, limits, sgn, incumbent

    def log_feasibility(self, Z: np.ndarray) -> np.ndarray:
        lp = np.zeros(len(Z))
        for k, gp in self.cons.items():
            mu, sd = predict(gp, Z)
            lim = self.limits[k]
            if lim.get("max") is not None:
                lp += log_ndtr((lim["max"] - mu) / sd)
            if lim.get("min") is not None:
                lp += log_ndtr((mu - lim["min"]) / sd)
        if self.clf is not None:
            lp += np.log(np.clip(self.clf.predict_proba(Z)[:, 1], 1e-9, 1.0))
        return lp

    def log_ei(self, Z: np.ndarray) -> np.ndarray:
        mu, sd = predict(self.obj, Z)
        return log_h((self.incumbent - self.sgn * mu) / sd) + np.log(sd)

    def log_acquisition(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        lp = self.log_feasibility(Z)
        if self.incumbent is None or self.obj is None:
            return lp
        return self.log_ei(Z) + lp


def maximise_acquisition(models: Models, space: Space, Z_seen: np.ndarray, anchors: np.ndarray, rng: np.random.Generator,
                         allow_repeat: bool = False) -> tuple[np.ndarray | None, float]:
    """Best new point of the acquisition: Sobol set + perturbations of the anchors, then L-BFGS-B on the continuous
    coordinates from the best candidates. Returns (z, log acquisition); z is None when no new point exists (an
    all-integer space that has been exhausted)."""
    from scipy.optimize import minimize
    from scipy.stats import qmc
    n = space.n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand = [qmc.Sobol(d=n, scramble=True, seed=int(rng.integers(0, 2 ** 31 - 1))).random(N_SOBOL)]
    for a in anchors:
        for s in LOCAL_SCALES:
            cand.append(a + rng.normal(scale=s, size=(max(4, N_LOCAL // (len(LOCAL_SCALES) * max(1, len(anchors)))), n)))
    Z = space.snap(np.vstack(cand))
    is_global = np.arange(len(Z)) < N_SOBOL
    if not allow_repeat and len(Z_seen):
        d = np.max(np.abs(Z[:, None, :] - Z_seen[None, :, :]), axis=2).min(axis=1)
        Z, is_global = Z[d > DUPLICATE_DISTANCE], is_global[d > DUPLICATE_DISTANCE]
    if space.n_integer_points is not None:
        Z, first = np.unique(Z, axis=0, return_index=True)
        is_global = is_global[first]
    if not len(Z):
        return None, -math.inf
    a = models.log_acquisition(Z)
    order = np.argsort(-a)
    best_z, best_a = Z[order[0]].copy(), float(a[order[0]])
    if space.cont:
        cont = space.cont
        # polish the best candidates overall AND the best of the space-filling set: the first group sits next to the
        # incumbent, the second keeps the search global (a far-away peak of the acquisition is not missed)
        starts = list(order[:N_POLISH]) + [i for i in order if is_global[i]][:N_POLISH]
        for i in dict.fromkeys(starts):
            z0 = Z[i].copy()

            def neg(c, z0=z0):
                z = z0.copy(); z[cont] = c
                return -float(models.log_acquisition(z[None, :])[0])

            res = minimize(neg, z0[cont], method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(cont), options={"maxiter": 40})
            z = z0.copy(); z[cont] = np.clip(res.x, 0.0, 1.0)
            if -res.fun > best_a and (allow_repeat or not len(Z_seen)
                                      or np.max(np.abs(z[None, :] - Z_seen), axis=1).min() > DUPLICATE_DISTANCE):
                best_z, best_a = z, float(-res.fun)
    return best_z, best_a


def _is_feasible(row: dict, limits: dict[str, dict]) -> tuple[bool, list[str]]:
    if not row.get("solved_ok") or row.get("objective") is None:
        return False, ["el motor no resolvió"]
    viol = []
    c = row.get("constraints") or {}
    for k, lim in limits.items():
        v = c.get(k)
        if v is None:
            return False, [f"{k} sin valor"]
        if lim.get("max") is not None and v > lim["max"]:
            viol.append(f"{k} = {v:.6g} > {lim['max']}")
        if lim.get("min") is not None and v < lim["min"]:
            viol.append(f"{k} = {v:.6g} < {lim['min']}")
    return not viol, viol


def build_models(Z: np.ndarray, rows: list[dict], limits: dict[str, dict], sgn: float, noisy: bool, seed: int,
                 theta: dict | None = None, optimise: bool = True) -> tuple[Models, dict]:
    """Fit every model to the evaluated points. `theta` = hyper-parameters of the previous iteration (warm start)."""
    theta = theta or {}
    ok = np.array([bool(r.get("solved_ok")) and r.get("objective") is not None
                   and all((r.get("constraints") or {}).get(k) is not None for k in limits) for r in rows])
    Zok = Z[ok]
    rok = [r for r, o in zip(rows, ok) if o]
    obj = None
    cons: dict = {}
    new_theta: dict = {}
    real = None
    if len(rok) >= 2:
        y = np.array([r["objective"] for r in rok], float)
        obj = real = fit_gp(Zok, y, noisy, seed, theta.get("objective"), optimise=optimise)
        new_theta["objective"] = obj.kernel_.theta
        if not ok.all():
            # Failed points have no objective. Leaving them out keeps the model's uncertainty (and so the expected
            # improvement) high exactly where the engine fails. They are imputed with a pessimistic prediction of the
            # model fitted to the real points, mean + 1 sd on the bad side, with the hyper-parameters unchanged
            # (after Forrester, Sóbester & Keane, Proc. R. Soc. A 462 (2006) 935-945, who impute ŷ + ŝ²; ŷ + ŝ is used
            # here because it has the units of y). Imputed values never enter the incumbent, the diagnostics or the output.
            mu, sd = predict(real, Z[~ok])
            obj = fit_gp(np.vstack([Zok, Z[~ok]]), np.concatenate([y, mu + sgn * sd]), noisy, seed, real.kernel_.theta, optimise=False)
        for k in limits:
            c = np.array([r["constraints"][k] for r in rok], float)
            cons[k] = fit_gp(Zok, c, noisy, seed, theta.get(k), optimise=optimise)
            new_theta[k] = cons[k].kernel_.theta
    clf = fit_solve_classifier(Z, ok, seed)
    feas = [i for i, r in enumerate(rok) if _is_feasible(r, limits)[0]]
    incumbent = None
    if obj is not None and feas:
        if noisy:      # with noise the incumbent is the best posterior mean among the feasible measured points
            incumbent = float(np.min(sgn * predict(real, Zok[feas])[0]))
        else:
            incumbent = float(min(sgn * rok[i]["objective"] for i in feas))
    m = Models(obj, cons, clf, limits, sgn, incumbent)
    m.real = real                # objective GP on the real points only (diagnostics, reported predictions)
    return m, new_theta


# ------------------------------------------------------------------------------------------------------- the loop
def bayes_optimize(evaluate: Evaluator, variables: list[dict], limits: dict[str, dict] | None = None, maximize: bool = False,
                   max_evaluations: int | None = None, n_initial: int | None = None, tolerance_pct: float = 0.5,
                   objective_tolerance: float | None = None, stall_iterations: int | None = None, seed: int = 0,
                   restarts: int = 0, label: Callable[[dict], str] | None = None, engine_name: str = "DWSIM") -> dict:
    """Core of optimize_design_bayesian with an injected evaluator (DWSIM in the tool; an analytic function in the
    tests). evaluate(x) → {"solved_ok": bool, "objective": float, "constraints": {key: value}}; an exception or
    solved_ok false is a failed point (kept as infeasible, never fatal). limits = {key: {"max"|"min": value,
    "tolerance": optional absolute}}. restarts = independent extra runs (new initial design, a model that sees only its
    own points) started after a run converges, while the budget lasts: the cheap check against a local optimum."""
    t0 = time.time()
    limits = limits or {}
    space = Space(variables, label or (lambda v: f"{v['name']}.{v['param']}"))
    n = space.n
    for k, lim in limits.items():
        if lim.get("max") is None and lim.get("min") is None:
            raise ValueError(f"la restricción {k} necesita max o min")
    if not 0 <= int(restarts) <= 5:
        raise ValueError("restarts entre 0 y 5")
    sgn = -1.0 if maximize else 1.0
    budget_assumed, init_assumed, stall_assumed = max_evaluations is None, n_initial is None, stall_iterations is None
    budget = int(max_evaluations) if max_evaluations is not None else (20 + 10 * n) * (1 + int(restarts))
    n_init = int(n_initial) if n_initial is not None else 2 * (n + 1)
    stall = int(stall_iterations) if stall_iterations is not None else max(20, 10 * n)
    if space.n_integer_points is not None:
        budget, restarts = min(budget, space.n_integer_points), 0
    n_init = max(2, min(n_init, budget))
    rng = np.random.default_rng(seed)

    Z = np.empty((0, n))
    rows: list[dict] = []
    engine_s: list[float] = []

    def run(z: np.ndarray, phase: str, i_run: int, extra: dict | None = None) -> dict:
        nonlocal Z
        x = space.to_x(z)[0]
        t = time.time()
        try:
            r = dict(evaluate([float(v) for v in x]))
        except Exception as ex:                      # a crash of the engine is a failed point, not a failed run
            r = {"solved_ok": False, "objective": None, "constraints": {}, "error": f"{type(ex).__name__}: {ex}"[:200]}
        engine_s.append(time.time() - t)
        if r.get("objective") is not None and not math.isfinite(float(r["objective"])):
            r["solved_ok"], r["objective"] = False, None
        feas, viol = _is_feasible(r, limits)
        row = {"i": len(rows) + 1, "run": i_run, "phase": phase, "values": space.named(x), "solved_ok": bool(r.get("solved_ok")),
               "objective": None if r.get("objective") is None else float(r["objective"]),
               "constraints": {k: (None if v is None else float(v)) for k, v in (r.get("constraints") or {}).items()},
               "feasible": feas, "violations": viol or None}
        if r.get("error"):
            row["error"] = r["error"]
        row.update(extra or {})
        rows.append(row)
        Z = np.vstack([Z, space.snap(z)])
        return row

    def best_index(start: int = 0) -> int | None:
        feas = [i for i in range(start, len(rows)) if rows[i]["feasible"]]
        return min(feas, key=lambda i: sgn * rows[i]["objective"]) if feas else None

    def tol_obj() -> float:
        if objective_tolerance is not None:
            return float(objective_tolerance)
        b = best_index()
        return tolerance_pct / 100.0 * max(abs(rows[b]["objective"]) if b is not None else 0.0, 1e-12)

    fit_s = 0.0
    runs: list[dict] = []
    for i_run in range(int(restarts) + 1):
        left = budget - len(rows)
        if i_run and left < n_init + EI_PATIENCE:        # not enough budget left for another meaningful run
            break
        start = len(rows)
        for z in _dedupe(initial_design(space, n_init, seed + 7919 * i_run), Z)[:left]:
            run(z, "inicial", i_run)
        theta: dict = {}
        stop, ei_low, since_improvement = None, 0, 0
        b0 = best_index(start)
        best_seen = sgn * rows[b0]["objective"] if b0 is not None else math.inf
        while stop is None:
            if len(rows) >= budget:
                stop = {"reason": "budget", "text": f"se agotó el presupuesto de {budget} evaluaciones"}
                break
            t = time.time()
            models, theta = build_models(Z[start:], rows[start:], limits, sgn, False, seed, theta)
            if models.obj is None:                   # fewer than two solved points: keep filling the space
                extra_z = _dedupe(initial_design(space, len(rows) - start + 8, seed + 7919 * i_run + 1 + len(rows)), Z)
                z, pred = (extra_z[0] if len(extra_z) else None), None
                if z is None:
                    stop = {"reason": "space_exhausted", "text": "todos los puntos enteros del espacio ya fueron evaluados"}
                    break
            else:
                cur = range(start, len(rows))
                feas_sorted = sorted((i for i in cur if rows[i]["feasible"]), key=lambda i: sgn * rows[i]["objective"])
                anchors = Z[feas_sorted[:3]] if feas_sorted else Z[[i for i in cur if rows[i]["solved_ok"]][:3]]
                z, la = maximise_acquisition(models, space, Z, anchors, rng)
                if z is None:
                    stop = {"reason": "space_exhausted", "text": "todos los puntos enteros del espacio ya fueron evaluados"}
                    break
                mu, sd = predict(models.obj, z[None, :])
                pred = {"predicted_mean": float(mu[0]), "predicted_sd": float(sd[0]),
                        "probability_feasible": float(math.exp(models.log_feasibility(z[None, :])[0]))}
                if models.incumbent is not None:
                    ei = float(math.exp(min(la, 700.0)))  # constrained expected improvement, objective units
                    pred["expected_improvement"] = ei
                    ei_low = ei_low + 1 if ei < EI_STOP_FRACTION * tol_obj() else 0
            fit_s += time.time() - t
            if (models.obj is not None and models.incumbent is not None and ei_low >= EI_PATIENCE
                    and len(rows) - start - n_init >= MIN_MODEL_ITERATIONS(n)):
                stop = {"reason": "expected_improvement", "text": f"la mejora esperada máxima quedó por debajo de {EI_STOP_FRACTION:g} × la "
                        f"tolerancia ({EI_STOP_FRACTION * tol_obj():.4g}) en {EI_PATIENCE} iteraciones seguidas"}
                break
            row = run(z, "bayesiana", i_run, pred)
            if row["feasible"] and sgn * row["objective"] < best_seen - tol_obj():
                since_improvement, ei_low = 0, 0                 # a real improvement contradicts a negligible EI
            elif row["solved_ok"]:
                since_improvement += 1                           # a failed solve says nothing about the objective: not counted
            if row["feasible"]:
                best_seen = min(best_seen, sgn * row["objective"])
            if since_improvement >= stall and best_index(start) is not None:
                stop = {"reason": "no_improvement", "text": f"sin mejora mayor que la tolerancia ({tol_obj():.4g}) en {stall} "
                        "evaluaciones resueltas seguidas"}
        b = best_index(start)
        runs.append({"run": i_run, "first_evaluation": start + 1, "n_evaluations": len(rows) - start, "stopped_by": stop,
                     "best_objective": None if b is None else rows[b]["objective"], "best_values": None if b is None else rows[b]["values"],
                     "best_index": b, "theta": theta})
        if stop["reason"] in ("budget", "space_exhausted"):
            break

    return _report(space, rows, Z, limits, sgn, runs, evaluate, engine_name, seed, t0, fit_s, engine_s,
                   {"budget": budget, "budget_assumed": budget_assumed, "n_initial": n_init, "init_assumed": init_assumed,
                    "stall": stall, "stall_assumed": stall_assumed, "tolerance_pct": tolerance_pct,
                    "objective_tolerance": objective_tolerance, "restarts": int(restarts)})


def _report(space, rows, Z, limits, sgn, runs, evaluate, engine_name, seed, t0, fit_s, engine_s, cfg) -> dict:
    n = space.n
    n_failed = sum(1 for r in rows if not r["solved_ok"])
    feas = [i for i, r in enumerate(rows) if r["feasible"]]
    tol_pct, obj_tol = cfg["tolerance_pct"], cfg["objective_tolerance"]
    method = engine_info() | {
        "initial_design": f"Sobol con mezcla de Owen (scipy.stats.qmc, semilla {seed}), {cfg['n_initial']} puntos por corrida",
        "stopping_rules": {"max_evaluations": cfg["budget"], "expected_improvement_below": f"{EI_STOP_FRACTION:g} × tolerancia, "
                           f"{EI_PATIENCE} iteraciones seguidas, tras al menos {MIN_MODEL_ITERATIONS(n)} evaluaciones guiadas por el modelo",
                           "no_improvement_in": cfg["stall"]},
        "restarts_requested": cfg["restarts"],
        "integer_variables": [k for k, it in zip(space.names, space.int) if it] or None,
        "seed": seed}
    ib = min(feas, key=lambda i: sgn * rows[i]["objective"]) if feas else None
    run_best = next((r for r in runs if ib is not None and r["first_evaluation"] - 1 <= ib < r["first_evaluation"] - 1 + r["n_evaluations"]), runs[-1])
    stop = run_best["stopped_by"]
    base: dict[str, Any] = {"method": method, "stopped_by": stop, "n_evaluations": len(rows), "n_failed_evaluations": n_failed,
                            "runs": [{k: v for k, v in r.items() if k not in ("theta", "best_index")} for r in runs],
                            "history": rows}
    first_error = next((r["error"] for r in rows if r.get("error")), None)
    med = float(np.median(engine_s)) if engine_s else 0.0
    suit = []
    if n >= 8:
        suit.append(f"{n} variables: la optimización bayesiana pierde eficiencia al crecer la dimensión; con más de "
                    f"{MAX_VARIABLES} la herramienta se niega.")
    if med < 1.0 and n <= 4:
        suit.append(f"Cada evaluación de {engine_name} tardó {med:.2f} s (mediana): el motor gastó {float(np.sum(engine_s)):.1f} s en total "
                    f"y el modelo bayesiano {fit_s:.1f} s. En un problema así de rápido la bayesiana ahorra evaluaciones pero no "
                    "tiempo: optimize_design_multi (directo) es más simple y no depende de un modelo. La bayesiana conviene cuando "
                    "cada evaluación cuesta minutos, falla a menudo o el presupuesto es de decenas de evaluaciones.")
    base["suitability"] = {"n_variables": n, "median_engine_seconds_per_evaluation": round(med, 3),
                           "engine_seconds": round(float(np.sum(engine_s)), 2), "model_seconds": round(fit_s, 2),
                           "notes": suit or None}
    base["assumptions"] = [
        "Cotas de las variables = las del usuario; no se evalúa ni se propone nada fuera de ellas.",
        f"Presupuesto: {cfg['budget']} evaluaciones{' (por defecto (20 + 10·n) por corrida, supuesto del método)' if cfg['budget_assumed'] else ''}; "
        f"diseño inicial de {cfg['n_initial']} puntos{' (por defecto 2·(n + 1))' if cfg['init_assumed'] else ''}; paro sin mejora en "
        f"{cfg['stall']} evaluaciones{' (por defecto max(20, 10·n))' if cfg['stall_assumed'] else ''}.",
        "Tolerancia del objetivo: " + (f"±{obj_tol:.4g} (dada por el usuario)" if obj_tol is not None
                                       else f"{tol_pct} % del mejor valor factible"),
        "El proceso gaussiano supone una respuesta suave (Matérn 5/2) y determinista. NO garantiza el óptimo global: en "
        "funciones con varios mínimos una corrida converge al que encuentra primero (Hartmann-6: ver BENCHMARK.md); "
        "`restarts` lanza corridas independientes para comprobarlo.",
        "Las restricciones son duras: el óptimo reportado es siempre un punto evaluado que las cumple.",
    ]
    if ib is None:
        return base | {"status": "not_verified", "verified": False, "feasible": False, "elapsed_s": round(time.time() - t0, 1),
                       "first_error": first_error,
                       "note": ("ninguna evaluación resolvió" + (f" (primer error: {first_error})" if first_error else "")
                                if n_failed == len(rows) else "ningún punto factible") + ": amplía las cotas o relaja un límite"}
    best = rows[ib]
    tol = float(obj_tol) if obj_tol is not None else tol_pct / 100.0 * max(abs(best["objective"]), 1e-12)
    tol_c = {}
    for k, lim in limits.items():        # given tolerance, else tolerance_pct of the limit (of the observed range when the limit is 0)
        ref = abs(float(lim["max"] if lim.get("max") is not None else lim["min"]))
        if ref == 0.0:
            seen = [r["constraints"][k] for r in rows if (r.get("constraints") or {}).get(k) is not None]
            ref = float(np.ptp(seen)) if seen else 0.0
        tol_c[k] = float(lim["tolerance"]) if lim.get("tolerance") is not None else tol_pct / 100.0 * ref

    # the model of the run that found the best point (the diagnostics below describe exactly this model)
    a, b = run_best["first_evaluation"] - 1, run_best["first_evaluation"] - 1 + run_best["n_evaluations"]
    models, _ = build_models(Z[a:b], rows[a:b], limits, sgn, False, seed, run_best["theta"])
    ok_idx = [i for i in range(a, b) if rows[i]["solved_ok"] and rows[i]["objective"] is not None
              and all((rows[i].get("constraints") or {}).get(k) is not None for k in limits)]
    pos = ok_idx.index(ib)
    gp_diag: dict[str, Any] = {}
    at_opt: dict[str, Any] = {}
    if models.obj is not None:
        for key, gp in [("objective", models.real)] + list(models.cons.items()):
            gp_diag[key] = {"hyperparameters": describe_gp(gp, space.names), "leave_one_out": loo_summary(gp)}
        lo_ = loo(models.real)
        at_opt = {"leave_one_out_mean": float(lo_["mean"][pos]), "leave_one_out_sd": float(lo_["sd"][pos]),
                  "explanation": "predicción del proceso gaussiano en el óptimo SIN usar ese punto (dejar-uno-fuera, "
                                 "hiperparámetros fijos), contra el valor real del motor"}
        if best.get("predicted_mean") is not None:
            at_opt["predicted_before_evaluation"] = {"mean": best["predicted_mean"], "sd": best["predicted_sd"]}
    one_step = [(r["predicted_mean"], r["predicted_sd"], r["objective"]) for r in rows[a:b]
                if r.get("predicted_mean") is not None and r["objective"] is not None]
    if one_step:
        e = np.array([m - y for m, _, y in one_step])
        zs = e / np.maximum(np.array([s for _, s, _ in one_step]), 1e-300)
        gp_diag["one_step_ahead"] = {"n": len(one_step), "rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs_error": float(np.max(np.abs(e))),
                                     "fraction_within_2_sd": float(np.mean(np.abs(zs) <= 2.0)),
                                     "note": "error de la predicción hecha ANTES de evaluar cada punto propuesto (fuera de muestra)"}

    # the engine is the truth: evaluate the best point again
    x_best = [best["values"][k] for k in space.names]
    try:
        re = dict(evaluate([float(v) for v in x_best]))
    except Exception as ex:
        re = {"solved_ok": False, "objective": None, "constraints": {}, "error": f"{type(ex).__name__}: {ex}"[:200]}
    checks = {"engine_solved_at_optimum": bool(re.get("solved_ok")) and re.get("objective") is not None}
    out = base | {"feasible": True, "tolerance": {"objective_abs": tol, "constraints_abs": tol_c or None},
                  "best": {"values": best["values"], "objective": best["objective"], "constraints": best["constraints"] or None,
                           "found_at_evaluation": best["i"], "run": best["run"]}}
    out["n_engine_evaluations_total"] = len(rows) + 1
    if checks["engine_solved_at_optimum"]:
        rc = {k: float(v) for k, v in (re.get("constraints") or {}).items() if v is not None}
        out["engine_at_optimum"] = {"objective": float(re["objective"]), "constraints": rc or None}
        viol = {k: float(max(0.0, (rc[k] - lim["max"]) if lim.get("max") is not None else 0.0,
                             (lim["min"] - rc[k]) if lim.get("min") is not None else 0.0)) for k, lim in limits.items() if k in rc}
        out["engine_constraint_violation"] = viol or None
        checks["engine_constraints_met"] = len(viol) == len(limits) and all(v <= 0.0 for v in viol.values())
        checks["reevaluation_reproduces"] = abs(float(re["objective"]) - best["objective"]) <= tol
        out["best"]["objective"] = float(re["objective"])
        out["best"]["constraints"] = rc or None
        if at_opt:
            err = abs(at_opt["leave_one_out_mean"] - float(re["objective"]))
            at_opt["engine_value"] = float(re["objective"])
            at_opt["abs_error"] = err
            at_opt["error_in_sd"] = err / max(at_opt["leave_one_out_sd"], 1e-300)
            checks["gp_consistent_at_optimum"] = bool(err <= max(tol, 3.0 * at_opt["leave_one_out_sd"]))
    out["gp_at_optimum"] = at_opt or None
    out["gp_diagnostics"] = gp_diag or None
    checks["converged_before_budget"] = bool(stop["reason"] in ("expected_improvement", "no_improvement", "space_exhausted"))
    zb = space.to_z([x_best])[0]
    # a limit is active when the engine value at the optimum sits within the constraint tolerance of it
    act = []
    rc_now = (out.get("engine_at_optimum") or {}).get("constraints") or best["constraints"] or {}
    for k, lim in limits.items():
        v = rc_now.get(k)
        if v is None:
            continue
        if lim.get("max") is not None and lim["max"] - v <= tol_c[k]:
            act.append(f"{k} ≤ {lim['max']}")
        if lim.get("min") is not None and v - lim["min"] <= tol_c[k]:
            act.append(f"{k} ≥ {lim['min']}")
    out["active_constraints"] = act or None
    at_bound = [k for k, z in zip(space.names, zb) if min(z, 1 - z) <= 1e-3]
    out["variables_at_bound"] = at_bound or None
    # independent restarts: do the converged runs agree?
    done = [r for r in runs if r["best_objective"] is not None and r["stopped_by"]["reason"] != "budget"]
    multi = None
    if len(done) >= 2:
        worse = [r for r in done if sgn * (r["best_objective"] - best["objective"]) > tol]
        multi = {"converged_runs": len(done), "runs_at_the_best_optimum": len(done) - len(worse),
                 "runs_at_a_worse_optimum": [{"run": r["run"], "objective": r["best_objective"], "values": r["best_values"]} for r in worse],
                 "text": ("las corridas independientes coinciden dentro de la tolerancia" if not worse else
                          f"{len(worse)} de {len(done)} corridas convergieron a un óptimo peor: el problema tiene varios óptimos "
                          "locales y no hay garantía de que el mejor encontrado sea el global")}
    out["restart_agreement"] = multi
    verified = all(checks.values()) and len(checks) == 5
    out["evidence"] = checks | {"verified": verified}
    out["verified"] = verified
    out["status"] = "verified" if verified else "not_verified"
    failed = [k for k, v in checks.items() if not v]
    reasons = {"engine_solved_at_optimum": "el motor no resolvió al reevaluar el óptimo",
               "engine_constraints_met": "el motor no cumple las restricciones en el óptimo reevaluado",
               "reevaluation_reproduces": "la reevaluación no reproduce el valor registrado (motor ruidoso)",
               "gp_consistent_at_optimum": "el proceso gaussiano no predice el óptimo dentro de 3 σ ni de la tolerancia",
               "converged_before_budget": "se agotó el presupuesto antes de que se cumpliera un criterio de convergencia: "
                                          "puede haber un punto mejor; da más evaluaciones"}
    out["reasons"] = [reasons[k] for k in failed] or None
    out["note"] = (("mejor punto confirmado en " + engine_name + " (" + stop["text"] + "); «verified» significa que el motor "
                    "confirma el punto y que la búsqueda convergió, no que el óptimo sea global"
                    + ("; " + multi["text"] if multi else "")
                    + ("; variable en la cota: amplía el rango" if at_bound else ""))
                   if verified else "NO verificado: " + "; ".join(reasons[k] for k in failed) + ". El mejor punto evaluado sí "
                   "es un resultado real del motor, pero no hay evidencia de que sea el óptimo.")
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out
