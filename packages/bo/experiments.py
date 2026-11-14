"""Bayesian design of the next experiments from the USER's own laboratory or plant data (bloque M16).

No simulator: the only data are the measurements the user gives. The tool fits the Gaussian process of
packages/bo/core.py with FITTED noise (measurements do not repeat exactly) and proposes the next batch with
constrained expected improvement. The batch is built with the Kriging-believer heuristic (Ginsbourger, Le Riche &
Carraro, «Kriging is well-suited to parallelize optimization», in Computational Intelligence in Expensive
Optimization Problems, Springer 2010, pp. 131-162): after each choice the model is updated as if the measurement had
come out at its predicted mean, which pushes the next choice elsewhere. Those «believed» values are internal: the
predicted mean and uncertainty reported for every proposal come from the model fitted to the real measurements only.

Nothing here is a measurement. Every row of `proposals` is a PROPOSAL to run. With fewer than 2·(n + 1) valid
measurements no model is fitted and the tool returns a space-filling design instead (Latin hypercube when there is
no data, greedy maximin over a scrambled Sobol set when some data exist), and says so.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from packages.bo import core as C

MIN_POINTS = lambda n: 2 * (n + 1)      # valid measurements needed before a model is fitted (MOCKS.md, M16)
MAX_BATCH = 24
WEAK_Q2 = 0.5                           # below this leave-one-out Q² the proposals are flagged as exploration
LOG_HINT_RATIO = 20.0                   # max/min of a positive objective above which the tool suggests "log": true
PROPOSAL = "PROPUESTA: experimento por hacer, no es una medición"


def _space_filling(space: C.Space, Z_seen: np.ndarray, n_points: int, seed: int) -> tuple[np.ndarray, str]:
    from scipy.stats import qmc
    if not len(Z_seen):
        z = space.snap(qmc.LatinHypercube(d=space.n, seed=seed).random(n_points))
        z = C._dedupe(z, Z_seen)
        if len(z) == n_points:
            return z, f"hipercubo latino (scipy.stats.qmc, semilla {seed})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand = C._dedupe(space.snap(qmc.Sobol(d=space.n, scramble=True, seed=seed).random(1024)), Z_seen)
    chosen: list[np.ndarray] = []
    ref = [z for z in np.clip(Z_seen, 0.0, 1.0)]
    for _ in range(min(n_points, len(cand))):
        if ref:
            d = np.min(np.linalg.norm(cand[:, None, :] - np.array(ref)[None, :, :], axis=2), axis=1)
            i = int(np.argmax(d))
        else:
            i = int(np.argmin(np.linalg.norm(cand - 0.5, axis=1)))
        chosen.append(cand[i]); ref.append(cand[i])
        cand = np.delete(cand, i, axis=0)
    return np.array(chosen).reshape(-1, space.n), ("máxima distancia mínima a los experimentos ya hechos, sobre 1024 puntos "
                                                   f"de Sobol (scipy.stats.qmc, semilla {seed})")


def suggest_experiments(variables: list[dict], experiments: list[dict], objective: dict, batch_size: int,
                        constraints: list[dict] | None = None, seed: int = 0) -> dict:
    """Bayesian design of the NEXT experiments from the user's own lab or plant data (no simulator, bloque M16).
    variables = [{"name", "lower", "upper", "log_scale": bool, "integer": bool}] (1-10; the bounds are the user's;
    categorical variables are not supported). experiments = the user's past runs, one dict per run with every variable
    name and the measured outputs, e.g. {"T_C": 80, "tau_min": 1.5, "yield": 0.62}; a run that could not be measured
    is {"T_C": ..., "tau_min": ..., "failed": true} and teaches the model where the process does not work.
    objective = {"name": "yield", "maximize": true} (the direction is mandatory; add "log": true to model log(y) when a
    positive response spans more than a decade, e.g. E-factor, impurity, cost: the tool warns when it looks so). constraints = [{"name": "impurity",
    "max"|"min": value}] on other measured outputs. batch_size = how many experiments the user can run next (theirs
    to choose). Returns `proposals` (each labelled as a PROPOSAL, with the predicted mean, the uncertainty of the
    mean, the 95 % interval of a new measurement and the probability of meeting the limits), `current_best` (the best
    MEASURED run), `model` (hyper-parameters, fitted noise and leave-one-out fit: q2, rmse, coverage) and `warnings`.
    With fewer than 2·(n + 1) valid runs it returns mode "initial_design": a space-filling plan, no model, and says
    how many runs are missing. Never invents measurements. ONE objective only: for several (yield vs cost, E-factor)
    the user must give the weights of a single combined objective; a Pareto front is not implemented."""
    if isinstance(objective, (list, tuple)) or "objectives" in (objective or {}):
        raise ValueError("un solo objetivo: el frente de Pareto (multiobjetivo) no está implementado; combina los objetivos "
                         "en uno con pesos que dé el usuario, o corre la herramienta una vez por objetivo")
    if not objective or not objective.get("name") or objective.get("maximize") is None:
        raise ValueError('objective = {"name": <columna medida>, "maximize": true|false}: la dirección la da el usuario')
    for v in variables:
        if v.get("categories") or v.get("values") or v.get("type") in ("categorical", "categorica", "categórica"):
            raise ValueError(f"{v.get('name')}: las variables categóricas (disolvente, catalizador) no están soportadas; "
                             "corre la herramienta por separado para cada categoría")
    batch = int(batch_size)
    if not 1 <= batch <= MAX_BATCH:
        raise ValueError(f"batch_size entre 1 y {MAX_BATCH}")
    space = C.Space(variables, lambda v: str(v["name"]))
    cons = constraints or []
    limits = {str(c["name"]): {"max": c.get("max"), "min": c.get("min")} for c in cons}
    for k, lim in limits.items():
        if lim["max"] is None and lim["min"] is None:
            raise ValueError(f"la restricción {k} necesita max o min")
    yname, maximize, use_log = str(objective["name"]), bool(objective["maximize"]), bool(objective.get("log"))
    if yname in space.names or any(k in space.names or k == yname for k in limits):
        raise ValueError("el objetivo y las restricciones deben ser columnas distintas de las variables")
    sgn = -1.0 if maximize else 1.0

    rows, X, warn = [], [], []
    for i, e in enumerate(experiments or []):
        miss = [k for k in space.names if e.get(k) is None]
        if miss:
            raise ValueError(f"experimento {i + 1}: falta {', '.join(miss)} (no se rellenan datos faltantes)")
        x = [float(e[k]) for k in space.names]
        if any(lg and xv <= 0 for lg, xv in zip(space.log, x)):
            raise ValueError(f"experimento {i + 1}: valor ≤ 0 en una variable en escala log")
        if e.get("failed"):
            rows.append({"solved_ok": False, "objective": None, "constraints": {}})
        else:
            if e.get(yname) is None:
                raise ValueError(f"experimento {i + 1}: falta la medición de {yname}; si el experimento falló márcalo "
                                 '"failed": true (no se inventan mediciones)')
            miss_c = [k for k in limits if e.get(k) is None]
            if miss_c:
                raise ValueError(f"experimento {i + 1}: falta la medición de {', '.join(miss_c)}")
            y = float(e[yname])
            if not math.isfinite(y):
                raise ValueError(f"experimento {i + 1}: {yname} no es un número finito")
            rows.append({"solved_ok": True, "objective": y, "constraints": {k: float(e[k]) for k in limits}})
        X.append(x)
    Z = space.to_z(np.array(X)) if X else np.empty((0, space.n))
    outside = [i + 1 for i, z in enumerate(Z) if np.any(z < -1e-9) or np.any(z > 1 + 1e-9)]
    if outside:
        warn.append(f"experimentos fuera de las cotas dadas (se usan en el modelo, pero solo se propone dentro de las cotas): {outside}")
    valid = [i for i, r in enumerate(rows) if r["solved_ok"]]
    n_failed = len(rows) - len(valid)
    feas = [i for i in valid if C._is_feasible(rows[i], limits)[0]]
    best = None
    if feas:
        ib = min(feas, key=lambda i: sgn * rows[i]["objective"])
        best = {"experiment": ib + 1, "inputs": space.named(X[ib]), yname: rows[ib]["objective"],
                "constraints": rows[ib]["constraints"] or None, "what": "mejor experimento MEDIDO que cumple los límites"}
    base = {"objective": {"name": yname, "maximize": maximize, "log": use_log}, "n_experiments": len(rows), "n_valid": len(valid),
            "n_failed": n_failed, "n_feasible": len(feas), "current_best": best, "method": C.engine_info() | {"seed": seed}}
    ys = np.array([rows[i]["objective"] for i in valid], float)
    if use_log and len(ys) and float(ys.min()) <= 0:
        raise ValueError(f'objective "log": true necesita {yname} > 0 en todas las mediciones')
    if not use_log and len(ys) >= 2 and float(ys.min()) > 0 and float(ys.max() / ys.min()) > LOG_HINT_RATIO:
        warn.append(f"{yname} abarca un factor de {ys.max() / ys.min():.0f} entre la menor y la mayor medición: un proceso gaussiano "
                    'ajusta mal respuestas así; considera objective "log": true (modela log y), es decisión del usuario')
    need = MIN_POINTS(space.n)
    constant = len(valid) >= 2 and float(np.ptp(ys)) == 0.0
    if len(valid) < need or constant:
        Zn, how = _space_filling(space, Z, batch, seed)
        why = (f"todas las mediciones de {yname} son iguales: no hay nada que modelar" if constant else
               f"{len(valid)} mediciones válidas; hacen falta {need} (2·(n + 1), n = {space.n}) para ajustar un modelo")
        return base | {"mode": "initial_design", "model": None,
                       "proposals": [{"label": PROPOSAL, "inputs": space.named(x), "predicted_mean": None, "predicted_sd": None,
                                      "why": "diseño que llena el espacio (sin modelo)"} for x in space.to_x(Zn)],
                       "warnings": warn or None,
                       "assumptions": [f"Sin modelo: {why}.", f"Diseño: {how}.",
                                       "Cotas = las del usuario; ningún valor de la respuesta es estimado o inventado."],
                       "note": f"DISEÑO INICIAL, no optimización: {why}. Faltan {max(0, need - len(valid))} experimentos válidos "
                               "para que la herramienta proponga con un modelo."}

    rng = np.random.default_rng(seed)
    # with "log": true the model works on log(y); everything reported is transformed back to the units of y
    rows_m = [r | {"objective": math.log(r["objective"])} if (use_log and r["solved_ok"]) else r for r in rows]
    back = (lambda m: math.exp(m)) if use_log else (lambda m: m)
    models, theta = C.build_models(Z, rows_m, limits, sgn, True, seed)
    real = models
    Zb, rows_b = Z.copy(), list(rows_m)
    props = []
    noise_sd = {k: float(math.sqrt(gp.kernel_.k2.noise_level) * gp._y_train_std) for k, gp in [(yname, real.real)] + list(real.cons.items())}
    for q in range(batch):
        if q:
            models, _ = C.build_models(Zb, rows_b, limits, sgn, True, seed, theta, optimise=False)
        order = sorted((i for i, r in enumerate(rows_b) if C._is_feasible(r, limits)[0]), key=lambda i: sgn * rows_b[i]["objective"])
        anchors = np.clip(Zb[order[:3]] if order else Zb[valid[:3]], 0.0, 1.0)
        z, la = C.maximise_acquisition(models, space, Zb, anchors, rng)
        if z is None:
            warn.append("no quedan puntos nuevos en el espacio entero dado")
            break
        mu, sd = C.predict(real.real, z[None, :])
        sd_obs = math.sqrt(float(sd[0]) ** 2 + noise_sd[yname] ** 2)
        p = {"label": PROPOSAL, "inputs": space.named(space.to_x(z)[0]), "predicted_mean": back(float(mu[0])),
             "predicted_sd": float(sd[0]) * (back(float(mu[0])) if use_log else 1.0),
             "interval95_new_measurement": [back(float(mu[0] - 1.96 * sd_obs)), back(float(mu[0] + 1.96 * sd_obs))],
             "probability_within_limits": float(math.exp(real.log_feasibility(z[None, :])[0])) if (limits or real.clf is not None) else None,
             "order_in_batch": q + 1}
        if real.incumbent is not None:
            p["expected_improvement" + ("_in_log_y" if use_log else "")] = float(math.exp(min(float(real.log_acquisition(z[None, :])[0]), 700.0)))
        cpred = {}
        for k, gp in real.cons.items():
            m, s = C.predict(gp, z[None, :])
            cpred[k] = {"predicted_mean": float(m[0]), "predicted_sd": float(s[0])}
        p["constraints"] = cpred or None
        props.append(p)
        Zb = np.vstack([Zb, z])
        rows_b.append({"solved_ok": True, "objective": float(mu[0]), "constraints": {k: v["predicted_mean"] for k, v in cpred.items()}})

    lo = {k: C.loo_summary(gp) for k, gp in [(yname, real.real)] + list(real.cons.items())}
    q2 = lo[yname]["q2"]
    weak = q2 is None or q2 < WEAK_Q2
    if weak:
        warn.append(f"ajuste débil (Q² dejar-uno-fuera = {q2 if q2 is None else round(q2, 2)} < {WEAK_Q2}): el modelo aún no predice "
                    "bien; toma las propuestas como exploración y no cites las medias predichas como resultados")
    if best is not None:
        m, s = C.predict(real.real, Z[[best["experiment"] - 1]])
        best["model_mean_here"], best["model_sd_here"] = back(float(m[0])), float(s[0]) * (back(float(m[0])) if use_log else 1.0)
    else:
        warn.append("ningún experimento medido cumple los límites: las propuestas buscan primero la región factible")
    return base | {"mode": "model", "proposals": props,
                   "model": {"hyperparameters": {k: C.describe_gp(gp, space.names) for k, gp in [(yname, real.real)] + list(real.cons.items())},
                             "noise_sd_fitted": noise_sd, "leave_one_out": lo, "weak_fit": weak,
                             "failure_model": ("clasificador de proceso gaussiano sobre los experimentos fallidos + imputación pesimista "
                                               "interna (media + 1 σ del lado malo) en el modelo del objetivo") if real.clf is not None else None},
                   "warnings": warn or None,
                   "assumptions": [
                       "Cotas = las del usuario; solo se propone dentro de ellas.",
                       "Respuesta suave (Matérn 5/2) con ruido de medición constante en todo el espacio, ajustado a los datos.",
                       f"Lote de {batch} por la heurística «Kriging believer»: las propuestas 2..n suponen que las anteriores salen en "
                       "su media predicha; no es el lote óptimo conjunto.",
                       f"Modelo desde {need} mediciones válidas (2·(n + 1)); aviso de ajuste débil si Q² < {WEAK_Q2} (supuestos del método).",
                       "Un solo objetivo; sin variables categóricas."]
                   + (["Objetivo modelado en log(y) a petición del usuario: `predicted_mean` es la MEDIANA predicha (exp de la media "
                       "en log), `predicted_sd` ≈ mediana × σ(log y), el ruido ajustado y el dejar-uno-fuera están en log(y)."] if use_log else []),
                   "note": "Las filas de `proposals` son PROPUESTAS de experimentos: sus medias e incertidumbres son predicciones "
                           "del modelo, no mediciones. Mide, agrega los resultados a `experiments` y vuelve a llamar."}
