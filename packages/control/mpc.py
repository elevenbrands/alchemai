"""Linear model predictive control with an exact dead time, offset-free tracking and a PID baseline (bloque advcontrol).

What it answers: «with this model, these horizons, weights and limits, how does an MPC behave on these setpoint and
disturbance scenarios, which constraints become active, and what does it buy over the PID I would tune with the usual rules?»

Model (deviation variables around the initial steady state, any consistent time unit). Every transfer-function element
k·Π(T s+1)/(s^n·Π(τ s+1))·e^{-θ s} (or num/den + dead time, or a state-space model with input delays) is sampled with a
zero-order hold. The dead time is exact: θ = d·h + f with 0 ≤ f < h gives (Åström & Wittenmark, Computer-Controlled
Systems, §2.3)

    x[k+1] = e^{A h} x[k] + Γ0 u[k−d] + Γ1 u[k−d−1],   Γ0 = ∫_0^{h−f} e^{A s} ds·B,   Γ1 = e^{A(h−f)} ∫_0^f e^{A s} ds·B

and the past inputs are states of the model (no Padé anywhere). The plant is integrated with the same formula on a grid
h = Ts/n finer than the controller period, so the trajectories between samples are exact and the two discretisations (Ts
and Ts/n, different d and f) check each other.

Controller (QDMC family: Garcia & Morshedi 1986; Rawlings, Mayne & Diehl 2017, ch. 1-2). With z = [x; d; u_prev] and
v = Δu the stage costs are

    J = Σ_{j=1..P} (y_j − r)' Qy (y_j − r) + x_j' Qx x_j + u_{j−1}' R u_{j−1}  +  Σ_{j=0..M−1} Δu_j' S Δu_j  +  Σ ρ_i ε_i²

subject to u_min ≤ u ≤ u_max, |Δu| ≤ Δu_max (hard) and y_min − ε ≤ y ≤ y_max + ε, ε ≥ 0 (soft, one slack per constrained
output). No terminal constraint or terminal cost: nominal stability is not guaranteed by construction (Rawlings §1.3.4
shows a horizon that destabilises a stable plant) and is therefore CHECKED from the eigenvalues of the unconstrained
loop. Offset-free tracking comes from an integrating disturbance d estimated from the measurement: «output» (d added to
y, deadbeat gain = the DMC bias, stable models only) or «input» (d added to u, steady-state Kalman filter, needed for
integrating processes); with R = 0 and Qx = 0 the only steady state with zero gradient has y = r.

The quadratic programme is condensed (decision = the M moves and the slacks) and solved here: Cholesky when no
constraint is active, otherwise a Mehrotra predictor-corrector interior-point method (Nocedal & Wright, Numerical
Optimization, alg. 16.4) polished by an exact solve on the identified active set. Every solve reports its KKT residuals.

PID baseline: the repo's own rules (`packages.dynamics.tuning`, SIMC by default) on the same plant, same scenario and
same limits, run on the fine grid. Multiloop PI for MIMO, each loop tuned on its own element without detuning."""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import scipy.signal as ss

from packages.dynamics import loop as LP
from packages.dynamics import tuning as TU
from packages.dynamics.process import Process, from_dict, half_rule

KKT_TOL = 1e-7              # relative KKT residual accepted for a QP solve
HARD_TOL = 1e-8             # relative violation accepted for a hard constraint in closed loop
MODEL_TOL = 1e-8            # two discretisations of the same model (Ts and Ts/n) must agree to this
HALVING_TOL = 2e-3          # indices on the grid h and h/2 must agree to this
MAX_SUBSTEPS = 100
MAX_PREDICTION_ROWS = 40000
SERIES_POINTS = 400


# ---------------------------------------------------------------------------------------------------------------------
# quadratic programming
# ---------------------------------------------------------------------------------------------------------------------
def qp_solve(H: np.ndarray, g: np.ndarray, G: np.ndarray | None = None, h: np.ndarray | None = None, tol: float = 1e-10,
             max_iter: int = 80, accept: float = 1e-8) -> dict:
    """min ½ x'Hx + g'x  s.t.  G x ≤ h, with H symmetric positive definite. Returns x, the multipliers z ≥ 0, the
    iterations, the KKT residuals (stationarity, primal feasibility, complementarity; relative) and `converged` = the
    largest of them is below `accept` (judged on the returned point, not on how the iteration ended)."""
    H = np.asarray(H, float)
    g = np.asarray(g, float)
    try:
        cf = sla.cho_factor(H)
    except sla.LinAlgError:
        raise ValueError("el problema cuadrático no es estrictamente convexo (H no es definida positiva): da pesos de "
                         "movimiento (move_weights) o de entrada (input_weights) > 0 en todas las entradas") from None
    x = -sla.cho_solve(cf, g)
    m = 0 if G is None else int(G.shape[0])
    if m == 0:
        return _qp_result(H, g, np.zeros((0, g.size)), np.zeros(0), x, np.zeros(0), 0, True, "sin restricciones")
    G = np.asarray(G, float)
    h = np.asarray(h, float)
    if np.all(G @ x <= h):
        return _qp_result(H, g, G, h, x, np.zeros(m), 0, True, "ninguna restricción activa")
    # rows scaled to unit norm so that the slacks are comparable (multipliers are scaled back at the end)
    rn = np.linalg.norm(G, axis=1)
    rn[rn == 0] = 1.0
    Gs, hs = G / rn[:, None], h / rn
    s, z = np.ones(m), np.ones(m)
    it = 0

    def step(rd, rp, rc, lu):
        dx = sla.cho_solve(lu, -rd - Gs.T @ ((z * rp - rc) / s))
        ds = -rp - Gs @ dx
        dz = (-rc - z * ds) / s
        return dx, ds, dz

    def alpha_max(v, dv):
        neg = dv < 0
        return min(1.0, float(np.min(-v[neg] / dv[neg]))) if np.any(neg) else 1.0

    # starting point (Nocedal & Wright §16.6): one affine step from s = z = 1, then keep both ≥ 1
    rd, rp = H @ x + g + Gs.T @ z, Gs @ x + s - hs
    lu = sla.cho_factor(H + Gs.T @ ((z / s)[:, None] * Gs))
    _, ds, dz = step(rd, rp, s * z, lu)
    s, z = np.maximum(1.0, np.abs(s + ds)), np.maximum(1.0, np.abs(z + dz))
    sg, sh = 1.0 + float(np.max(np.abs(g))), 1.0 + float(np.max(np.abs(hs)))
    for it in range(1, max_iter + 1):
        rd, rp = H @ x + g + Gs.T @ z, Gs @ x + s - hs
        mu = float(s @ z) / m
        if np.max(np.abs(rd)) <= tol * sg and np.max(np.abs(rp)) <= tol * sh and mu <= 1e-3 * tol * sg:
            break
        if float(np.min(s)) < 1e-200 or float(np.min(z)) < 1e-200:
            break
        try:
            lu = sla.cho_factor(H + Gs.T @ ((z / s)[:, None] * Gs))
        except (sla.LinAlgError, ValueError):   # z/s too ill-conditioned to factor: the active-set pass decides
            break
        dxa, dsa, dza = step(rd, rp, s * z, lu)
        a = min(alpha_max(s, dsa), alpha_max(z, dza))
        sigma = (float((s + a * dsa) @ (z + a * dza)) / m / mu) ** 3
        dx, ds, dz = step(rd, rp, s * z + dsa * dza - sigma * mu, lu)
        if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dz))):
            break
        eta = max(0.995, 1.0 - mu)
        a = min(1.0, eta * alpha_max(s, ds), eta * alpha_max(z, dz))
        x, s, z = x + a * dx, s + a * ds, z + a * dz
        if a < 1e-12:
            break
    z_true = z / rn
    # active-set pass from the set the interior point identified: exact solve on the working set, drop the most negative
    # multiplier or add the most violated row, until it is a KKT point (accepted only then; otherwise the IPM point stays)
    work = list(np.nonzero(z > s)[0])
    n = g.size
    sh_true = 1.0 + float(np.max(np.abs(h)))
    for _ in range(40):
        if work:
            Ga = G[work]
            K = np.block([[H, Ga.T], [Ga, np.zeros((len(work), len(work)))]])
            sol = np.linalg.lstsq(K, np.concatenate([-g, h[work]]), rcond=None)[0]
            xp, lam = sol[:n], sol[n:]
        else:
            xp, lam = -sla.cho_solve(cf, g), np.zeros(0)
        viol = G @ xp - h
        if lam.size and float(np.min(lam)) < -1e-10 * (1.0 + float(np.max(np.abs(lam)))):
            work.pop(int(np.argmin(lam)))
            continue
        worst = int(np.argmax(viol))
        if viol[worst] > 1e-12 * sh_true:
            if worst in work:
                break
            work.append(worst)
            continue
        zp = np.zeros(m)
        zp[work] = np.maximum(lam, 0.0)
        if np.max(np.abs(H @ xp + g + G.T @ zp)) <= 1e-9 * sg:
            x, z_true = xp, zp
        break
    return _qp_result(H, g, G, h, x, z_true, it, None, "punto interior (Mehrotra) + pasada de conjunto activo", accept)


def _qp_result(H, g, G, h, x, z, it, converged, how, accept: float = 1e-8) -> dict:
    sg = 1.0 + max(float(np.max(np.abs(g))), float(np.max(np.abs(H @ x)))) if g.size else 1.0
    stat = float(np.max(np.abs(H @ x + g + (G.T @ z if z.size else 0.0)))) / sg
    if z.size:
        slack = h - G @ x
        sh = 1.0 + float(np.max(np.abs(h)))
        prim = max(0.0, float(np.max(-slack))) / sh
        comp = float(np.max(np.abs(z * slack))) / (sg * sh)
        active = int(np.sum(slack <= 1e-9 * sh))
    else:
        prim = comp = 0.0
        active = 0
    kkt = max(stat, prim, comp, max(0.0, float(-np.min(z))) / sg if z.size else 0.0)
    return {"x": x, "z": z, "iterations": it, "converged": bool(kkt <= accept) if converged is None else bool(converged),
            "how": how, "active": active, "stationarity": stat, "primal_infeasibility": prim, "complementarity": comp, "kkt": kkt,
            "objective": float(0.5 * x @ H @ x + g @ x)}


# ---------------------------------------------------------------------------------------------------------------------
# model: parsing and exact discretisation with dead time
# ---------------------------------------------------------------------------------------------------------------------
def _process_from_polys(num: np.ndarray, den: np.ndarray, delay: float) -> Process | None:
    """Factored form of num/den when every pole and zero is real (needed only to tune the PID baseline)."""
    try:
        zr = np.roots(num) if num.size > 1 else np.zeros(0)
        pr = np.roots(den)
        if np.any(np.abs(zr.imag) > 1e-6 * (1 + np.abs(zr))) or np.any(np.abs(pr.imag) > 1e-6 * (1 + np.abs(pr))):
            return None
        zr, pr = zr.real, pr.real
        n_int = int(np.sum(np.abs(pr) < 1e-12))
        if n_int > 1 or np.any(np.abs(zr) < 1e-12):
            return None
        lags = [-1.0 / p for p in pr if abs(p) >= 1e-12]
        return Process(float(num[-1] / den[-1 - n_int]), lags, [-1.0 / q for q in zr], delay, bool(n_int))
    except (ValueError, ZeroDivisionError, IndexError):
        return None


def _element(e, where: str) -> dict | None:
    """One transfer-function element: factored ({gain, time_constants, lead_time_constants, dead_time, integrating}) or
    polynomial ({num, den, dead_time}, descending powers of s)."""
    if e is None:
        return None
    if not isinstance(e, dict):
        raise ValueError(f"{where}: cada elemento del modelo es un dict (o null si esa entrada no afecta a esa salida)")
    if "num" in e or "den" in e:
        unknown = set(e) - {"num", "den", "dead_time"}
        if unknown:
            raise ValueError(f"{where}: claves desconocidas {sorted(unknown)}; forma polinómica = num, den, dead_time")
        if "num" not in e or "den" not in e or "dead_time" not in e:
            raise ValueError(f"{where}: la forma polinómica necesita 'num', 'den' (potencias descendentes de s) y "
                             "'dead_time' (0 si no hay): son datos del usuario")
        num = np.trim_zeros(np.atleast_1d(np.asarray(e["num"], float)), "f")
        den = np.trim_zeros(np.atleast_1d(np.asarray(e["den"], float)), "f")
        delay = float(e["dead_time"])
        if den.size == 0 or not np.all(np.isfinite(den)) or not np.all(np.isfinite(num)):
            raise ValueError(f"{where}: num/den deben ser finitos y den ≠ 0")
        if not math.isfinite(delay) or delay < 0:
            raise ValueError(f"{where}: el tiempo muerto debe ser ≥ 0")
        if num.size == 0:
            return None
        proc = _process_from_polys(num, den, delay)
    else:
        proc = from_dict(e)
        num, den = proc.polys()
        delay = proc.delay
    if num.size >= den.size:
        raise ValueError(f"{where}: el elemento tiene paso directo (tantos ceros como polos); el MPC necesita un modelo "
                         "estrictamente propio: añade la dinámica rápida que falta (actuador, sensor)")
    return {"num": np.asarray(num, float), "den": np.asarray(den, float), "delay": delay, "process": proc}


def _matrix(spec, what: str) -> list[list[dict | None]]:
    if isinstance(spec, dict):
        rows = [[spec]]
    elif isinstance(spec, (list, tuple)) and spec and all(isinstance(r, (list, tuple)) for r in spec):
        rows = [list(r) for r in spec]
    else:
        raise ValueError(f"{what}: da un dict (SISO) o una matriz [fila = salida][columna = entrada] de dicts "
                         "(null = esa entrada no afecta a esa salida)")
    if len({len(r) for r in rows}) != 1 or not rows[0]:
        raise ValueError(f"{what}: todas las filas deben tener el mismo número de columnas")
    out = [[_element(e, f"{what}[{i}][{j}]") for j, e in enumerate(r)] for i, r in enumerate(rows)]
    return out


def _tf_blocks(mat: list[list[dict | None]], n_out: int, col_offset: int = 0) -> list[dict]:
    blocks = []
    for i, row in enumerate(mat):
        for j, e in enumerate(row):
            if e is None:
                continue
            A, B, C, _ = ss.tf2ss(e["num"], e["den"])
            Cf = np.zeros((n_out, A.shape[0]))
            Cf[i] = C[0]
            blocks.append({"A": A, "B": B, "C": Cf, "inputs": [j + col_offset], "delays": [e["delay"]], "dt": None})
    return blocks


def _ss_block(s: dict, what: str) -> dict:
    unknown = set(s) - {"A", "B", "C", "D", "dt", "input_delays"}
    if unknown:
        raise ValueError(f"{what}: claves desconocidas {sorted(unknown)}; se aceptan A, B, C, D, dt, input_delays")
    for k in ("A", "B", "C"):
        if k not in s:
            raise ValueError(f"{what}: falta la matriz '{k}' del modelo en espacio de estados (dato del usuario)")
    A = np.atleast_2d(np.asarray(s["A"], float))
    B = np.atleast_2d(np.asarray(s["B"], float))
    C = np.atleast_2d(np.asarray(s["C"], float))
    n = A.shape[0]
    if A.shape != (n, n) or B.shape[0] != n or C.shape[1] != n:
        raise ValueError(f"{what}: dimensiones incompatibles (A n×n, B n×nu, C ny×n)")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(C))):
        raise ValueError(f"{what}: A, B y C deben ser finitas")
    if s.get("D") is not None and np.any(np.asarray(s["D"], float) != 0):
        raise ValueError(f"{what}: D ≠ 0 (paso directo) no está cubierto; el MPC necesita un modelo estrictamente propio")
    nu = B.shape[1]
    delays = [float(v) for v in (s.get("input_delays") or [0.0] * nu)]
    if len(delays) != nu or any((not math.isfinite(v)) or v < 0 for v in delays):
        raise ValueError(f"{what}: input_delays necesita un retardo ≥ 0 por entrada")
    dt = s.get("dt")
    return {"A": A, "B": B, "C": C, "inputs": list(range(nu)), "delays": delays, "dt": float(dt) if dt else None}


def _int_expm(A: np.ndarray, B: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """e^{A t} and ∫_0^t e^{A s} ds·B from one matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = sla.expm(M * t)
    return E[:n, :n], E[:n, n:]


def assemble(blocks: list[dict], n_in: int, n_out: int, h: float) -> dict:
    """Discrete model x+ = A x + B u, y = C x at step h: block states first, then one chain of past values per delayed
    input (chain[l] = u[k−l]). Exact for inputs held over h, whatever the dead times."""
    parts, need = [], [0] * n_in
    for b in blocks:
        n = b["A"].shape[0]
        cols = []
        if b["dt"] is None:
            Phi = sla.expm(b["A"] * h)
            for c, (j, th) in enumerate(zip(b["inputs"], b["delays"])):
                d = int(math.floor(th / h + 1e-9))
                f = th - d * h
                f = 0.0 if f < 1e-9 * h else f
                Pa, G0 = _int_expm(b["A"], b["B"][:, [c]], h - f)
                G1 = Pa @ _int_expm(b["A"], b["B"][:, [c]], f)[1] if f > 0 else np.zeros((n, 1))
                cols.append((j, d, G0[:, 0], G1[:, 0]))
        else:
            if abs(b["dt"] - h) > 1e-9 * h:
                raise ValueError(f"el modelo discreto tiene dt = {b['dt']:g} y el periodo pedido es {h:g}: deben coincidir "
                                 "(un modelo discreto no se puede remuestrear aquí)")
            Phi = b["A"]
            for c, (j, th) in enumerate(zip(b["inputs"], b["delays"])):
                d = int(round(th / h))
                if abs(d * h - th) > 1e-9 * h:
                    raise ValueError("en un modelo discreto los retardos de entrada deben ser múltiplos enteros de dt")
                cols.append((j, d, b["B"][:, c], np.zeros(n)))
        for j, d, G0, G1 in cols:
            if np.any(G0):
                need[j] = max(need[j], d)
            if np.any(G1):
                need[j] = max(need[j], d + 1)
        parts.append((Phi, cols))
    nb = sum(b["A"].shape[0] for b in blocks)
    off = [nb + sum(need[:j]) for j in range(n_in)]
    N = nb + sum(need)
    A, B, C = np.zeros((N, N)), np.zeros((N, n_in)), np.zeros((n_out, N))
    r = 0
    for b, (Phi, cols) in zip(blocks, parts):
        n = Phi.shape[0]
        A[r:r + n, r:r + n] = Phi
        C[:, r:r + n] = b["C"]
        for j, d, G0, G1 in cols:
            for lag, Gm in ((d, G0), (d + 1, G1)):
                if not np.any(Gm):
                    continue
                if lag == 0:
                    B[r:r + n, j] += Gm
                else:
                    A[r:r + n, off[j] + lag - 1] += Gm
        r += n
    for j in range(n_in):
        for lag in range(1, need[j] + 1):
            if lag == 1:
                B[off[j], j] = 1.0
            else:
                A[off[j] + lag - 1, off[j] + lag - 2] = 1.0
    return {"A": A, "B": B, "C": C, "n_block": nb, "lags": need, "h": h}


def _time_scales(blocks: list[dict]) -> list[float]:
    sc = []
    for b in blocks:
        if b["dt"] is None:
            sc += [1.0 / abs(v) for v in np.linalg.eigvals(b["A"]) if abs(v) > 1e-12]
        sc += [d for d in b["delays"] if d > 0]
    return sc


def parse_model(process_model, state_space, what: str = "process_model") -> dict:
    """Blocks, sizes and (for transfer-function models) the parsed element matrix."""
    if (process_model is None) == (state_space is None):
        raise ValueError("da el modelo del proceso: process_model (función de transferencia, SISO o matriz) O state_space "
                         "(A, B, C); exactamente uno, no hay modelo por defecto")
    if process_model is not None:
        mat = _matrix(process_model, what)
        ny, nu = len(mat), len(mat[0])
        blocks = _tf_blocks(mat, ny)
        for j in range(nu):
            if all(mat[i][j] is None for i in range(ny)):
                raise ValueError(f"{what}: la entrada {j} no afecta a ninguna salida")
        return {"blocks": blocks, "nu": nu, "ny": ny, "elements": mat, "kind": "transfer_function"}
    b = _ss_block(state_space, "state_space")
    return {"blocks": [b], "nu": b["B"].shape[1], "ny": b["C"].shape[0], "elements": None, "kind": "state_space"}


# ---------------------------------------------------------------------------------------------------------------------
# controller design
# ---------------------------------------------------------------------------------------------------------------------
def _vec(v, n: int, name: str, default: float | None = None) -> np.ndarray:
    """Scalar or list of n (None entries = the default); None with no default is an error."""
    if v is None:
        if default is None:
            raise ValueError(f"falta {name}")
        return np.full(n, float(default))
    if isinstance(v, (int, float)):
        return np.full(n, float(v))
    if len(v) != n:
        raise ValueError(f"{name} necesita {n} valor(es) (uno por variable) o un escalar; llegaron {len(v)}")
    if default is None and any(x is None for x in v):
        raise ValueError(f"{name}: faltan valores (null)")
    return np.array([float(default) if x is None else float(x) for x in v])


def design(model: dict, P: int, M: int, Qy: np.ndarray, S: np.ndarray, R: np.ndarray, Qx: np.ndarray | None,
           u_min: np.ndarray, u_max: np.ndarray, du_max: np.ndarray, y_min: np.ndarray, y_max: np.ndarray,
           rho: np.ndarray, disturbance_model: str, estimator_q: float) -> dict:
    """Prediction matrices, constant QP data, unconstrained gain, estimator gain and nominal eigenvalues."""
    A, B, C = model["A"], model["B"], model["C"]
    n, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    rho_ol = float(np.max(np.abs(np.linalg.eigvals(A)))) if n else 0.0
    stable = rho_ol < 1 - 1e-8
    dm = disturbance_model
    if dm == "auto":
        dm = "output" if stable else "input"
    if dm == "output":
        if not stable:
            raise ValueError("el modelo de perturbación 'output' (sesgo DMC) solo vale para modelos estables en lazo "
                             "abierto; este tiene un integrador o un polo inestable: usa disturbance_model='input'")
        Bd, Cd = np.zeros((n, ny)), np.eye(ny)
    elif dm == "input":
        if nu != ny:
            raise ValueError("disturbance_model='input' necesita tantas entradas como salidas (una perturbación "
                             "integrante por salida); con un sistema no cuadrado usa 'output'")
        Bd, Cd = B.copy(), np.zeros((ny, nu))
    elif dm == "none":
        Bd, Cd = np.zeros((n, 0)), np.zeros((ny, 0))
    else:
        raise ValueError("disturbance_model debe ser 'auto', 'output', 'input' o 'none'")
    nd = Bd.shape[1]
    nz = n + nd + nu
    if P * nz > MAX_PREDICTION_ROWS or M * nu > 600:
        raise ValueError(f"problema demasiado grande (P·estados = {P * nz}, M·entradas = {M * nu}): reduce los horizontes "
                         "o aumenta el periodo de muestreo (los tiempos muertos se guardan como estados)")
    Az = np.zeros((nz, nz))
    Az[:n, :n], Az[:n, n:n + nd], Az[:n, n + nd:] = A, Bd, B
    Az[n:, n:] = np.eye(nd + nu)
    Bz = np.vstack([B, np.zeros((nd, nu)), np.eye(nu)])
    Cz = np.hstack([C, Cd, np.zeros((ny, nu))])
    W = Cz.T @ Qy @ Cz
    if Qx is not None:
        W[:Qx.shape[0], :Qx.shape[0]] += Qx
    W[n + nd:, n + nd:] += R
    Sz, Sv = np.zeros((P * nz, nz)), np.zeros((P * nz, M * nu))
    pw = [np.eye(nz)]
    for _ in range(P):
        pw.append(Az @ pw[-1])
    AB = [p @ Bz for p in pw[:P]]
    H = np.kron(np.eye(M), S)
    Fz, Fr = np.zeros((M * nu, nz)), np.zeros((M * nu, P * ny))
    CQ = Cz.T @ Qy
    for j in range(1, P + 1):
        rows = slice((j - 1) * nz, j * nz)
        Sz[rows] = pw[j]
        for i in range(min(j, M)):
            Sv[rows, i * nu:(i + 1) * nu] = AB[j - 1 - i]
        Svj = Sv[rows]                  # moves beyond M are zero: in velocity form u simply stays at its last value
        H += Svj.T @ W @ Svj
        Fz += Svj.T @ W @ pw[j]
        Fr[:, (j - 1) * ny:j * ny] = Svj.T @ CQ
    H = 0.5 * (H + H.T)
    # constraints  G w ≤ hc − Ez z0,  w = [V; ε]
    soft = [i for i in range(ny) if math.isfinite(y_min[i]) or math.isfinite(y_max[i])]
    ns = len(soft)
    nw = M * nu + ns
    rows_G, rows_h, rows_E, tags = [], [], [], []

    def add(gv, hc, ez, tag):
        rows_G.append(gv); rows_h.append(hc); rows_E.append(ez); tags.append(tag)

    for k in range(M):
        for j in range(nu):
            pick = np.zeros(nz); pick[n + nd + j] = 1.0
            cum = np.zeros(nw)
            cum[[i * nu + j for i in range(k + 1)]] = 1.0
            if math.isfinite(u_max[j]):
                add(cum, u_max[j], pick, ("u_max", j, k))
            if math.isfinite(u_min[j]):
                add(-cum, -u_min[j], -pick, ("u_min", j, k))
            if math.isfinite(du_max[j]):
                e = np.zeros(nw); e[k * nu + j] = 1.0
                add(e, du_max[j], np.zeros(nz), ("du_max", j, k))
                add(-e, du_max[j], np.zeros(nz), ("du_min", j, k))
    for si, i in enumerate(soft):
        for j in range(1, P + 1):
            rows = slice((j - 1) * nz, j * nz)
            gy = np.zeros(nw); gy[:M * nu] = Cz[i] @ Sv[rows]
            ez = Cz[i] @ Sz[rows]
            if math.isfinite(y_max[i]):
                gg = gy.copy(); gg[M * nu + si] = -1.0
                add(gg, y_max[i], ez, ("y_max", i, j))
            if math.isfinite(y_min[i]):
                gg = -gy; gg[M * nu + si] = -1.0
                add(gg, -y_min[i], -ez, ("y_min", i, j))
        e = np.zeros(nw); e[M * nu + si] = -1.0
        add(e, 0.0, np.zeros(nz), ("slack", i, 0))
    Hw = np.zeros((nw, nw))
    Hw[:M * nu, :M * nu] = H
    for si, i in enumerate(soft):
        Hw[M * nu + si, M * nu + si] = rho[i]
    try:
        cf = sla.cho_factor(H)
    except sla.LinAlgError:
        raise ValueError("el problema cuadrático no es estrictamente convexo: da move_weights > 0 (o input_weights > 0) "
                         "en todas las entradas") from None
    Kz = sla.cho_solve(cf, Fz)[:nu]
    Kr = sla.cho_solve(cf, Fr)[:nu]
    Kr_const = sum(Kr[:, j * ny:(j + 1) * ny] for j in range(P))
    # estimator on xe = [x; d]
    ne = n + nd
    Ae, Be, Ce = Az[:ne, :ne], np.vstack([B, np.zeros((nd, nu))]), Cz[:, :ne]
    if dm == "output":
        L = np.vstack([np.zeros((n, ny)), np.eye(ny)])
    elif dm == "input":
        Qw = sla.block_diag(1e-10 * estimator_q * np.eye(n), estimator_q * np.eye(nd))
        try:
            Pk = sla.solve_discrete_are(Ae.T, Ce.T, Qw, np.eye(ny))
        except (sla.LinAlgError, ValueError):
            raise ValueError("no se pudo calcular el estimador (filtro de Kalman estacionario): el modelo con "
                             "perturbación a la entrada no es detectable; prueba disturbance_model='output'") from None
        L = Pk @ Ce.T @ np.linalg.inv(Ce @ Pk @ Ce.T + np.eye(ny))
    else:
        L = np.zeros((ne, ny))
    rho_est = float(np.max(np.abs(np.linalg.eigvals((np.eye(ne) - L @ Ce) @ Ae)))) if (ne and dm != "none") else 0.0
    if dm == "input" and rho_est >= 1 - 1e-9:
        raise ValueError("el estimador con perturbación a la entrada no converge (modelo no detectable); prueba "
                         "disturbance_model='output' o revisa el modelo")
    # nominal unconstrained regulator on (x, u_prev)
    idx = list(range(n)) + list(range(n + nd, nz))
    Ar, Br, Kreg = Az[np.ix_(idx, idx)], Bz[idx], Kz[:, idx]
    eig = np.linalg.eigvals(Ar - Br @ Kreg)
    out = {"n": n, "nu": nu, "ny": ny, "nd": nd, "nz": nz, "P": P, "M": M, "Az": Az, "Bz": Bz, "Cz": Cz, "Sz": Sz, "Sv": Sv,
           "H": H, "Hw": Hw, "Fz": Fz, "Fr": Fr, "G": np.array(rows_G).reshape(len(rows_G), nw),
           "hc": np.array(rows_h, float), "Ez": np.array(rows_E).reshape(len(rows_E), nz), "tags": tags, "soft": soft,
           "Kz": Kz, "Kr": Kr, "Kr_const": Kr_const, "Ae": Ae, "Be": Be, "Ce": Ce, "L": L, "disturbance_model": dm,
           "open_loop_spectral_radius": rho_ol, "estimator_spectral_radius": rho_est, "regulator_eigenvalues": eig,
           "regulator_spectral_radius": float(np.max(np.abs(eig))) if eig.size else 0.0, "Ar": Ar, "Br": Br, "Kreg": Kreg,
           "Wr": W[np.ix_(idx, idx)], "S": S}
    return out


def plan(ctrl: dict, z0: np.ndarray, r_stack: np.ndarray) -> dict:
    """Solve the QP at one sampling instant. Returns the qp_solve dict plus the predicted moves and outputs."""
    nv = ctrl["M"] * ctrl["nu"]
    g = np.concatenate([ctrl["Fz"] @ z0 - ctrl["Fr"] @ r_stack, np.zeros(len(ctrl["soft"]))])
    G = ctrl["G"] if ctrl["G"].shape[0] else None
    sol = qp_solve(ctrl["Hw"], g, G, (ctrl["hc"] - ctrl["Ez"] @ z0) if G is not None else None)
    if G is not None:                     # ε ≥ 0 is always tight when no output limit is touched: not «activity»
        slack = (ctrl["hc"] - ctrl["Ez"] @ z0) - G @ sol["x"]
        tight = slack <= 1e-9 * (1.0 + float(np.max(np.abs(ctrl["hc"]))))
        sol["active"] = int(sum(1 for t, tag in zip(tight, ctrl["tags"]) if t and tag[0] != "slack"))
    sol["moves"] = sol["x"][:nv].reshape(ctrl["M"], ctrl["nu"])
    sol["slack"] = sol["x"][nv:]
    return sol


def predicted_outputs(ctrl: dict, z0: np.ndarray, moves: np.ndarray) -> np.ndarray:
    Z = ctrl["Sz"] @ z0 + ctrl["Sv"] @ moves.ravel()
    return (Z.reshape(ctrl["P"], ctrl["nz"]) @ ctrl["Cz"].T)


def infinite_horizon_gain(ctrl: dict) -> np.ndarray | None:
    """LQR gain of the same stage cost on (x, u_prev) with input Δu (needs S > 0); None when the Riccati equation has
    no stabilising solution (e.g. more inputs than outputs with R = 0)."""
    S = ctrl["S"]
    if np.min(np.linalg.eigvalsh(S)) <= 0:
        return None
    try:
        X = sla.solve_discrete_are(ctrl["Ar"], ctrl["Br"], ctrl["Wr"], S)
    except (sla.LinAlgError, ValueError):
        return None
    return np.linalg.solve(S + ctrl["Br"].T @ X @ ctrl["Br"], ctrl["Br"].T @ X @ ctrl["Ar"])


# ---------------------------------------------------------------------------------------------------------------------
# scenario and closed-loop simulation
# ---------------------------------------------------------------------------------------------------------------------
def _scenario(setpoint_changes, disturbances, ny: int, nu: int, ndm: int, Ts: float, n_steps: int, n_sub: int) -> dict:
    Nf = n_steps * n_sub
    h = Ts / n_sub
    r = np.zeros((Nf + 1, ny)); din = np.zeros((Nf + 1, nu)); dout = np.zeros((Nf + 1, ny)); dmod = np.zeros((Nf + 1, ndm))
    notes, first = [], None
    for ev in sorted(setpoint_changes or [], key=lambda e: e.get("time", 0.0)):
        if "time" not in ev or "value" not in ev:
            raise ValueError("cada cambio de consigna necesita 'time' y 'value' (y 'output' si hay varias salidas)")
        i = int(ev.get("output", 0 if ny == 1 else -1))
        if not 0 <= i < ny:
            raise ValueError(f"cambio de consigna: 'output' debe ser un índice entre 0 y {ny - 1}")
        k = int(math.ceil(float(ev["time"]) / Ts - 1e-9))
        if abs(k * Ts - float(ev["time"])) > 1e-9 * Ts:
            notes.append(f"el cambio de consigna en t = {ev['time']:g} se aplica en el siguiente instante de muestreo "
                         f"(t = {k * Ts:g})")
        if k <= n_steps:
            r[k * n_sub:, i] = float(ev["value"])
            first = k if first is None else min(first, k)
    for ev in sorted(disturbances or [], key=lambda e: e.get("time", 0.0)):
        if "time" not in ev or "value" not in ev or "type" not in ev:
            raise ValueError("cada perturbación necesita 'time', 'value' y 'type' ('input', 'output' o 'model')")
        kind = ev["type"]
        arr, nn = {"input": (din, nu), "output": (dout, ny), "model": (dmod, ndm)}.get(kind, (None, 0))
        if arr is None:
            raise ValueError("tipo de perturbación desconocido: usa 'input' (suma a la entrada del proceso), 'output' "
                             "(suma a la salida) o 'model' (canal de disturbance_models)")
        if kind == "model" and ndm == 0:
            raise ValueError("perturbación de tipo 'model' sin disturbance_models: da el modelo de la perturbación")
        j = int(ev.get("index", 0 if nn == 1 else -1))
        if not 0 <= j < nn:
            raise ValueError(f"perturbación '{kind}': 'index' debe estar entre 0 y {nn - 1}")
        f = int(math.ceil(float(ev["time"]) / h - 1e-9))
        if f <= Nf:
            arr[f:, j] = float(ev["value"])
            first = f // n_sub if first is None else min(first, f // n_sub)
    return {"r": r, "din": din, "dout": dout, "dmod": dmod, "Nf": Nf, "h": h, "notes": notes, "first_event_step": first or 0}


def _indices(yp: np.ndarray, sc: dict) -> dict:
    """IAE, ISE and ITAE per output, integrating each interval with the setpoint and output bias in force during it."""
    h, Nf = sc["h"], sc["Nf"]
    el = sc["r"][:Nf] - yp[:Nf] - sc["dout"][:Nf]
    er = sc["r"][:Nf] - yp[1:] - sc["dout"][:Nf]
    t = (np.arange(Nf) + 0.5) * h
    iae = h * np.sum(np.abs(el) + np.abs(er), axis=0) / 2
    ise = h * np.sum(el ** 2 + el * er + er ** 2, axis=0) / 3
    itae = h * np.sum(t[:, None] * (np.abs(el) + np.abs(er)), axis=0) / 2
    return {"IAE": iae.tolist(), "ISE": ise.tolist(), "ITAE": itae.tolist(), "final_error": er[-1].tolist()}


def simulate_mpc(ctrl: dict, plant: dict, sc: dict, n_steps: int, n_sub: int, x0: np.ndarray | None,
                 state_feedback: bool, model: dict, plan_step: int | None = None) -> dict:
    Ap, Bp, Cp = plant["A"], plant["B"], plant["C"]
    nu, ny, n, nd = ctrl["nu"], ctrl["ny"], ctrl["n"], ctrl["nd"]
    nb = model["n_block"]
    xp = np.zeros(Ap.shape[0])
    if x0 is not None:
        xp[:len(x0)] = x0
    xe = np.zeros(n + nd)
    xm = np.zeros(n)                       # controller copy of the model state (state feedback: its own input history)
    up = np.zeros(nu)
    Nf = sc["Nf"]
    yp = np.empty((Nf + 1, ny)); Y = np.empty((n_steps, ny)); U = np.empty((n_steps, nu))
    stats = {"solves": 0, "unconstrained": 0, "max_iterations": 0, "max_kkt": 0.0, "not_converged": 0,
             "steps_with_active_constraints": 0, "max_slack": 0.0}
    plan_out = None
    diverged = False
    for k in range(n_steps):
        f = k * n_sub
        y = Cp @ xp + sc["dout"][f]
        if state_feedback:
            xm[:nb] = xp[:nb]
            z0 = np.concatenate([xm, up])
        else:
            if k:
                xe = ctrl["Ae"] @ xe + ctrl["Be"] @ up
            xe = xe + ctrl["L"] @ (y - ctrl["Ce"] @ xe)
            z0 = np.concatenate([xe, up])
        sol = plan(ctrl, z0, np.tile(sc["r"][f], ctrl["P"]))
        stats["solves"] += 1
        stats["unconstrained"] += int(sol["iterations"] == 0)
        stats["max_iterations"] = max(stats["max_iterations"], sol["iterations"])
        stats["max_kkt"] = max(stats["max_kkt"], sol["kkt"])
        stats["not_converged"] += int(not sol["converged"])
        stats["steps_with_active_constraints"] += int(sol["active"] > 0)
        if sol["slack"].size:
            stats["max_slack"] = max(stats["max_slack"], float(np.max(sol["slack"])))
        if plan_step is not None and k == plan_step:
            plan_out = {"z0": z0, "moves": sol["moves"].copy(), "y_pred": predicted_outputs(ctrl, z0, sol["moves"]),
                        "u_prev": up.copy(), "slack": sol["slack"].copy(), "objective": sol["objective"]}
        u = up + sol["moves"][0]
        Y[k], U[k] = y, u
        if state_feedback:
            xm = model["A"] @ xm + model["B"] @ u
        for s in range(n_sub):
            ff = f + s
            yp[ff] = Cp @ xp
            xp = Ap @ xp + Bp @ np.concatenate([u + sc["din"][ff], sc["dmod"][ff]])
        up = u
        if not np.all(np.isfinite(xp)) or np.max(np.abs(xp)) > 1e150:
            diverged = True
            yp[f + n_sub:] = np.nan
            Y[k + 1:], U[k + 1:] = np.nan, np.nan
            break
    if not diverged:
        yp[Nf] = Cp @ xp
    return {"yp": yp, "Y": Y, "U": U, "stats": stats, "plan": plan_out, "diverged": diverged}


def _pid_matrices(c: LP.Controller, alpha: float):
    """State-space matrices of the repo's controller. `Controller.ss` keeps the integral state only when KI > 0, so a
    reverse-acting controller (negative-gain process, Kc < 0) is realised as the mirror controller with the sign of its
    output flipped — the same linear law."""
    sign = 1.0
    if c.Kc < 0 or (c.Kc == 0 and (c.KI or 0.0) < 0):
        c, sign = LP.Controller(c.kind, -c.Kc, c.tau_I, c.tau_D, c.form, -(c.KI or 0.0)), -1.0
    Ac, by, br, cx, dy, dr = c.ss(alpha)
    return Ac, by, br, sign * np.asarray(cx, float), sign * dy, sign * dr


def simulate_pid(loops: list[dict], plant: dict, sc: dict, n_steps: int, n_sub: int, nu: int, ny: int, u_min: np.ndarray,
                 u_max: np.ndarray, du_max: np.ndarray, x0: np.ndarray | None) -> dict:
    """Multiloop PID on the fine grid (h = Ts/n_sub): controller states advanced exactly with the measurement held over
    h, clamping to the input limits, slew limit Δu_max per Ts, conditional-integration anti-windup."""
    Ap, Bp, Cp = plant["A"], plant["B"], plant["C"]
    h, Nf = sc["h"], sc["Nf"]
    mats = []
    for lp in loops:
        Ac, by, br, cx, dy, dr = _pid_matrices(lp["controller"], lp["alpha"])
        nc = Ac.shape[0]
        if nc:
            Mx = np.zeros((nc + 2, nc + 2))
            Mx[:nc, :nc], Mx[:nc, nc], Mx[:nc, nc + 1] = Ac, by, br
            E = sla.expm(Mx * h)
            mats.append((E[:nc, :nc], E[:nc, nc], E[:nc, nc + 1], np.asarray(cx, float), float(dy), float(dr), np.zeros(nc)))
        else:
            mats.append((np.zeros((0, 0)), np.zeros(0), np.zeros(0), np.zeros(0), float(dy), float(dr), np.zeros(0)))
    states = [m[6] for m in mats]
    xp = np.zeros(Ap.shape[0])
    if x0 is not None:
        xp[:len(x0)] = x0
    slew = du_max / n_sub
    u = np.zeros(nu)
    yp = np.empty((Nf + 1, ny)); Uf = np.empty((Nf, nu))
    diverged = False
    for f in range(Nf):
        yplant = Cp @ xp
        y = yplant + sc["dout"][f]
        r = sc["r"][f]
        for li, lp in enumerate(loops):
            i, j = lp["output"], lp["input"]
            Phi, Gy, Gr, cx, dy, dr, _ = mats[li]
            xc = states[li]
            raw = float(cx @ xc) + dy * y[i] + dr * r[i]
            lim = min(max(raw, u_min[j], u[j] - slew[j]), u_max[j], u[j] + slew[j])
            new = Phi @ xc + Gy * y[i] + Gr * r[i] if xc.size else xc
            if xc.size and lim != raw and (raw - lim) * cx[0] * (new[0] - xc[0]) > 0:
                new[0] = xc[0]
            states[li] = new
            u[j] = lim
        yp[f], Uf[f] = yplant, u
        xp = Ap @ xp + Bp @ np.concatenate([u + sc["din"][f], sc["dmod"][f]])
        if not np.all(np.isfinite(xp)) or np.max(np.abs(xp)) > 1e150:
            diverged = True
            yp[f + 1:], Uf[f + 1:] = np.nan, np.nan
            break
    if not diverged:
        yp[Nf] = Cp @ xp
    return {"yp": yp, "Uf": Uf, "diverged": diverged}


def pid_loops(elements, spec: dict | None, nu: int, ny: int) -> tuple[list[dict] | None, dict]:
    """Loops of the PID baseline: given by the user or tuned with the repo's rules on each paired element."""
    spec = dict(spec or {})
    unknown = set(spec) - {"skip", "rule", "controller_type", "tau_c", "pairing", "controllers", "derivative_filter_alpha"}
    if unknown:
        raise ValueError(f"pid_baseline: claves desconocidas {sorted(unknown)}")
    if spec.get("skip"):
        return None, {"skipped_reason": "omitido a petición (pid_baseline.skip)"}
    pairing = spec.get("pairing") or [[i, i] for i in range(min(nu, ny))]
    rule, ctype = spec.get("rule", "SIMC"), str(spec.get("controller_type", "PI")).upper()
    alpha = float(spec.get("derivative_filter_alpha", 0.01))
    given = spec.get("controllers")
    if given is not None and len(given) != len(pairing):
        raise ValueError("pid_baseline.controllers necesita un controlador por lazo del emparejamiento")
    tau_c = spec.get("tau_c")
    loops, rows = [], []
    for li, (i, j) in enumerate(pairing):
        if not (0 <= i < ny and 0 <= j < nu):
            raise ValueError("pid_baseline.pairing: pares [salida, entrada] fuera de rango")
        if given is not None:
            c = LP.controller_from_dict(given[li])
            rows.append({"output": i, "input": j, "source": "dado por el usuario", "controller": c.as_dict()})
        else:
            e = elements[i][j] if elements is not None else None
            if e is None or e["process"] is None:
                return None, {"skipped_reason": (
                    f"el elemento salida {i} / entrada {j} no es una función de transferencia con polos y ceros reales "
                    "(o el modelo es de espacio de estados): las reglas de sintonía del repositorio necesitan un "
                    "FOPDT/SOPDT; da pid_baseline.controllers con los ajustes de planta si quieres la comparación")}
            p = e["process"]
            try:
                hr = half_rule(p, 2 if ctype == "PID" else 1)
            except ValueError as ex:
                return None, {"skipped_reason": str(ex)}
            m = {k: hr[k] for k in ("k", "theta", "integrating", "tau1", "tau2")}
            tc = tau_c[li] if isinstance(tau_c, (list, tuple)) else tau_c
            row = TU.tune(m, ctype, [rule], None if tc is None else float(tc), TU.ultimate(p))[0]
            if not row["applicable"]:
                return None, {"skipped_reason": f"lazo salida {i} / entrada {j}: {row['reason']}"}
            c = row["controller"]
            rows.append({"output": i, "input": j, "source": row["source"], "rule": rule, "tau_c": row["tau_c"],
                         "model_used_for_tuning": m, "controller": c.as_dict()})
        loops.append({"output": i, "input": j, "controller": c, "alpha": alpha})
    return loops, {"loops": rows, "controller_type": ctype if given is None else "dado",
                   "derivative_filter_alpha": alpha}


# ---------------------------------------------------------------------------------------------------------------------
# evidence helpers
# ---------------------------------------------------------------------------------------------------------------------
def _step_response(m: dict, j: int, n: int, every: int = 1) -> np.ndarray:
    """Outputs at steps 0, every, 2·every, … of the discrete model m for a unit step in input j."""
    x = np.zeros(m["A"].shape[0])
    out = np.empty((n + 1, m["C"].shape[0]))
    b = m["B"][:, j]
    for k in range(n * every + 1):
        if k % every == 0:
            out[k // every] = m["C"] @ x
        x = m["A"] @ x + b
    return out


def _analytic_step(p: Process, t: np.ndarray) -> np.ndarray | None:
    """Closed-form unit-step response of k·e^{-θs}/Π(τ_i s+1) with distinct lags (partial fractions); None otherwise."""
    if p.integrating or p.leads or not p.lags:
        return None
    lags = np.array(p.lags)
    if lags.size > 1 and np.min(np.abs(np.diff(np.sort(lags)))) < 1e-6 * np.max(lags):
        return None
    tt = np.maximum(t - p.delay, 0.0)
    y = np.ones_like(tt)
    nl = lags.size
    for i, ti in enumerate(lags):
        coef = ti ** (nl - 1) / np.prod([ti - tj for k, tj in enumerate(lags) if k != i]) if nl > 1 else 1.0
        y -= coef * np.exp(-tt / ti)
    return p.gain * np.where(t > p.delay, y, 0.0)


def model_evidence(parsed: dict, model: dict, plant: dict, Ts: float, n_sub: int, n: int) -> dict:
    """Agreement of the two discretisations (Ts and Ts/n_sub) and of both with the closed-form step response."""
    nu, ny = parsed["nu"], parsed["ny"]
    worst, scale, ana_worst, ana_n = 0.0, 0.0, 0.0, 0
    t = np.arange(n + 1) * Ts
    for j in range(nu):
        a = _step_response(model, j, n)
        b = _step_response(plant, j, n, n_sub) if n_sub > 1 else a
        sc = max(float(np.max(np.abs(a))), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b))) / sc)
        scale = max(scale, sc)
        if parsed["elements"] is not None:
            for i in range(ny):
                e = parsed["elements"][i][j]
                ya = _analytic_step(e["process"], t) if e is not None and e["process"] is not None else None
                if ya is not None:
                    ana_n += 1
                    ana_worst = max(ana_worst, float(np.max(np.abs(a[:, i] - ya))) / max(float(np.max(np.abs(ya))), 1e-300))
    out = {"two_grid_step_response_max_rel_diff": worst, "substeps": n_sub}
    if ana_n:
        out.update(closed_form_step_response_max_rel_diff=ana_worst, closed_form_elements_checked=ana_n)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the engine entry point
# ---------------------------------------------------------------------------------------------------------------------
def _activity(U: np.ndarray, u_min, u_max, du_max, yp: np.ndarray, sc: dict, y_min, y_max, every: int) -> dict:
    """Constraint activity of a closed-loop run. U = inputs at each of its own steps; `every` fine steps per U row."""
    nu, ny = U.shape[1], yp.shape[1]
    dU = np.diff(np.vstack([np.zeros((1, nu)), U]), axis=0)
    n = U.shape[0]
    ins = []
    worst = 0.0
    for j in range(nu):
        tol = 1e-7 * max(1.0, abs(u_max[j]) if math.isfinite(u_max[j]) else 1.0, abs(u_min[j]) if math.isfinite(u_min[j]) else 1.0)
        row = {"input": j, "fraction_at_max": float(np.mean(U[:, j] >= u_max[j] - tol)) if math.isfinite(u_max[j]) else 0.0,
               "fraction_at_min": float(np.mean(U[:, j] <= u_min[j] + tol)) if math.isfinite(u_min[j]) else 0.0}
        lim = du_max[j]
        if math.isfinite(lim):
            row["fraction_at_rate_limit"] = float(np.mean(np.abs(dU[:, j]) >= lim / sc["rate_div"] * (1 - 1e-7)))
            worst = max(worst, float(np.max(np.abs(dU[:, j]) - lim / sc["rate_div"])) / max(1.0, lim))
        else:
            row["fraction_at_rate_limit"] = 0.0
        if math.isfinite(u_max[j]):
            worst = max(worst, float(np.max(U[:, j] - u_max[j])) / max(1.0, abs(u_max[j])))
        if math.isfinite(u_min[j]):
            worst = max(worst, float(np.max(u_min[j] - U[:, j])) / max(1.0, abs(u_min[j])))
        ins.append(row)
    outs = []
    y = yp[:sc["Nf"]] + sc["dout"][:sc["Nf"]]
    for i in range(ny):
        if not (math.isfinite(y_min[i]) or math.isfinite(y_max[i])):
            continue
        over = np.maximum(y[:, i] - y_max[i], 0.0) if math.isfinite(y_max[i]) else np.zeros(len(y))
        under = np.maximum(y_min[i] - y[:, i], 0.0) if math.isfinite(y_min[i]) else np.zeros(len(y))
        v = np.maximum(over, under)
        outs.append({"output": i, "max_violation": float(np.max(v)), "fraction_of_time_violated": float(np.mean(v > 1e-9)),
                     "integrated_violation": float(np.sum(v) * sc["h"])})
    return {"inputs": ins, "soft_outputs": outs, "hard_max_rel_violation": max(worst, 0.0), "steps": n}


def _settled(yp: np.ndarray, U: np.ndarray, sc: dict) -> bool:
    Nf = sc["Nf"]
    k0 = int(0.9 * Nf)
    if Nf - k0 < 2:
        return False
    for arr in (sc["r"], sc["din"], sc["dout"], sc["dmod"]):
        if arr.shape[1] and np.any(np.ptp(arr[k0:], axis=0) > 0):
            return False
    y = yp[k0:]
    span = np.maximum(np.ptp(yp, axis=0), 1e-12)
    uspan = np.maximum(np.ptp(U, axis=0), 1e-12)
    return bool(np.all(np.ptp(y, axis=0) <= 2e-3 * span) and np.all(np.ptp(U[int(0.9 * len(U)):], axis=0) <= 2e-3 * uspan))


def _series(t: np.ndarray, arrs: dict[str, np.ndarray]) -> dict:
    idx = np.unique(np.linspace(0, len(t) - 1, min(len(t), SERIES_POINTS)).round().astype(int))
    out = {"time": t[idx].tolist()}
    for k, a in arrs.items():
        out[k] = [a[idx, c].tolist() for c in range(a.shape[1])]
    return out


def mpc_closed_loop(*, sampling_period: float | None, prediction_horizon: int | None, control_horizon: int | None,
                    simulation_time: float | None, move_weights=None, process_model=None, state_space=None,
                    output_weights=None, input_weights=None, state_weights=None, input_min=None, input_max=None,
                    input_rate_max=None, output_min=None, output_max=None, output_soft_weight=None,
                    setpoint_changes: list[dict] | None = None, disturbances: list[dict] | None = None,
                    disturbance_models=None, initial_state: list[float] | None = None, disturbance_model: str = "auto",
                    estimator_q: float = 1.0, plant_model=None, pid_baseline: dict | None = None,
                    simulation_substeps: int | None = None, return_trajectories: bool = True) -> dict:
    """Design a linear MPC on the given model and simulate it in closed loop on the given scenario, with a PID baseline
    on the same plant. See the module docstring for the formulation. Raises ValueError (Spanish) on missing data."""
    if sampling_period is None or not float(sampling_period) > 0:
        raise ValueError("falta sampling_period (> 0): el periodo de ejecución del MPC, en las unidades de tiempo del "
                         "modelo; es una decisión de diseño del usuario (típico: 1/10 a 1/5 de la constante de tiempo "
                         "dominante), no se supone")
    if prediction_horizon is None or control_horizon is None:
        raise ValueError("faltan prediction_horizon y control_horizon (en periodos de muestreo): son decisiones de "
                         "diseño del usuario; el de predicción debe cubrir el tiempo de asentamiento en lazo abierto")
    if simulation_time is None or not float(simulation_time) > 0:
        raise ValueError("falta simulation_time (> 0): la duración del escenario a simular")
    Ts, P, M = float(sampling_period), int(prediction_horizon), int(control_horizon)
    if P < 1 or M < 1 or M > P:
        raise ValueError("los horizontes deben cumplir 1 ≤ control_horizon ≤ prediction_horizon")
    parsed = parse_model(process_model, state_space)
    nu, ny, blocks = parsed["nu"], parsed["ny"], parsed["blocks"]
    discrete = any(b["dt"] is not None for b in blocks)
    if move_weights is None and input_weights is None:
        raise ValueError("faltan move_weights (peso de cada movimiento Δu; decisión de sintonía del usuario, depende de "
                         "las unidades): sin él —o sin input_weights— el problema no tiene solución única")
    S = np.diag(_vec(move_weights, nu, "move_weights", 0.0))
    R = np.diag(_vec(input_weights, nu, "input_weights", 0.0))
    Qy = np.diag(_vec(output_weights, ny, "output_weights", 1.0))
    if np.any(np.diag(S) < 0) or np.any(np.diag(R) < 0) or np.any(np.diag(Qy) < 0):
        raise ValueError("los pesos (output_weights, move_weights, input_weights) deben ser ≥ 0")
    assumptions, warnings, not_included = [], [], []
    if output_weights is None:
        assumptions.append("output_weights = 1 en todas las salidas (no se dieron): el peso relativo salida/movimiento "
                           "es la sintonía del MPC, revísalo con las unidades de tus variables")
    Qx = None
    nb_model = sum(b["A"].shape[0] for b in blocks)
    if state_weights is not None:
        if parsed["kind"] != "state_space":
            raise ValueError("state_weights solo tiene sentido con un modelo state_space (pondera sus estados)")
        Qx = np.asarray(state_weights, float)
        Qx = np.diag(Qx) if Qx.ndim == 1 else Qx
        if Qx.shape != (nb_model, nb_model) or np.max(np.abs(Qx - Qx.T)) > 1e-12 * (1 + np.max(np.abs(Qx))) \
                or np.min(np.linalg.eigvalsh(0.5 * (Qx + Qx.T))) < -1e-12 * (1 + np.max(np.abs(Qx))):
            raise ValueError(f"state_weights debe ser una matriz {nb_model}×{nb_model} simétrica semidefinida positiva "
                             "(o su diagonal)")
    u_min, u_max = _vec(input_min, nu, "input_min", -math.inf), _vec(input_max, nu, "input_max", math.inf)
    du_max = _vec(input_rate_max, nu, "input_rate_max", math.inf)
    y_min, y_max = _vec(output_min, ny, "output_min", -math.inf), _vec(output_max, ny, "output_max", math.inf)
    if np.any(u_min > 0) or np.any(u_max < 0):
        raise ValueError("los límites de entrada deben contener el 0: todo está en variables de desviación respecto al "
                         "estado estacionario inicial (input_min ≤ 0 ≤ input_max); resta el valor inicial de la entrada")
    if np.any(du_max <= 0) or np.any(y_min > y_max):
        raise ValueError("input_rate_max debe ser > 0 y output_min ≤ output_max")
    soft_used = bool(np.any(np.isfinite(y_min)) or np.any(np.isfinite(y_max)))
    rho_default = 1e4 * max(1.0, float(np.max(np.diag(Qy))))
    rho = _vec(output_soft_weight, ny, "output_soft_weight", rho_default)
    if np.any(rho <= 0):
        raise ValueError("output_soft_weight debe ser > 0")
    if soft_used and output_soft_weight is None:
        assumptions.append(f"peso de la holgura de las restricciones de salida ρ = {rho_default:g} (10⁴ × el mayor peso de "
                           "salida): ajuste numérico; más alto = menos violación y movimientos más bruscos")
    if not math.isfinite(float(estimator_q)) or float(estimator_q) <= 0:
        raise ValueError("estimator_q debe ser > 0")
    # plant (may differ from the controller model) and disturbance channels
    if plant_model is not None:
        is_ss = isinstance(plant_model, dict) and "A" in plant_model
        pparsed = parse_model(None if is_ss else plant_model, plant_model if is_ss else None, "plant_model")
        if (pparsed["nu"], pparsed["ny"]) != (nu, ny):
            raise ValueError("plant_model debe tener las mismas entradas y salidas que el modelo del controlador")
    else:
        pparsed = parsed
    dm_blocks, ndm = [], 0
    if disturbance_models is not None:
        dmat = _matrix(disturbance_models, "disturbance_models")
        if len(dmat) != ny:
            raise ValueError(f"disturbance_models necesita {ny} fila(s) (una por salida) y una columna por perturbación")
        ndm = len(dmat[0])
        dm_blocks = _tf_blocks(dmat, ny, col_offset=nu)
    if discrete and (plant_model is not None and not any(b["dt"] is not None for b in pparsed["blocks"]) or dm_blocks):
        raise ValueError("con un modelo discreto la planta y las perturbaciones deben ser el mismo modelo discreto "
                         "(no se mezclan tiempos continuo y discreto)")
    plant_blocks = pparsed["blocks"] + dm_blocks
    n_steps = int(round(float(simulation_time) / Ts))
    if n_steps < 1:
        raise ValueError("simulation_time debe cubrir al menos un periodo de muestreo")
    if discrete or any(b["dt"] is not None for b in plant_blocks):
        n_sub = 1
    elif simulation_substeps is not None:
        n_sub = int(simulation_substeps)
        if n_sub < 1:
            raise ValueError("simulation_substeps debe ser ≥ 1")
    else:
        sc_t = _time_scales(plant_blocks)
        n_sub = min(MAX_SUBSTEPS, max(1, int(math.ceil(Ts / (min(sc_t) / 20.0))))) if sc_t else 1
    if n_steps > 50000 or n_steps * n_sub > 1_000_000:
        raise ValueError(f"simulación demasiado larga ({n_steps} periodos × {n_sub} subpasos): reduce simulation_time o "
                         "aumenta sampling_period")
    x0 = None
    if initial_state is not None:
        if parsed["kind"] != "state_space" or plant_model is not None:
            raise ValueError("initial_state solo vale con un modelo state_space sin plant_model (es el estado inicial de "
                             "ese modelo, en desviación)")
        x0 = np.asarray(initial_state, float)
        if x0.shape != (nb_model,):
            raise ValueError(f"initial_state necesita {nb_model} valores (uno por estado del modelo)")
    if not (setpoint_changes or disturbances or (x0 is not None and np.any(x0 != 0))):
        raise ValueError("no hay escenario que simular: da setpoint_changes, disturbances o un initial_state ≠ 0 (el "
                         "sistema parte del estado estacionario en variables de desviación)")
    model = assemble(blocks, nu, ny, Ts)
    ctrl = design(model, P, M, Qy, S, R, Qx, u_min, u_max, du_max, y_min, y_max, rho, disturbance_model, float(estimator_q))
    dm = ctrl["disturbance_model"]
    state_feedback = dm == "none"
    if state_feedback and plant_model is not None:
        raise ValueError("disturbance_model='none' (realimentación del estado del modelo, sin estimador) no admite "
                         "plant_model distinto: sin modelo de perturbación no hay corrección del desajuste")
    loops, pid_info = pid_loops(parsed["elements"], pid_baseline, nu, ny)

    def run(ns: int) -> dict:
        plant = assemble(plant_blocks, nu + ndm, ny, Ts / ns)
        sc = _scenario(setpoint_changes, disturbances, ny, nu, ndm, Ts, n_steps, ns)
        res = simulate_mpc(ctrl, plant, sc, n_steps, ns, x0, state_feedback, model, sc["first_event_step"])
        res.update(plant=plant, sc=sc, indices=_indices(res["yp"], sc) if not res["diverged"] else None)
        if loops is not None:
            pr = simulate_pid(loops, plant, sc, n_steps, ns, nu, ny, u_min, u_max, du_max, x0)
            pr["indices"] = _indices(pr["yp"], sc) if not pr["diverged"] else None
            res["pid"] = pr
        return res

    main = run(n_sub)
    sc = main["sc"]
    half = run(2 * n_sub) if n_sub * 2 * n_steps <= 2_000_000 and not discrete else None
    reasons = []
    # --- model ------------------------------------------------------------------------------------------------------
    stable_ol = ctrl["open_loop_spectral_radius"] < 1 - 1e-8
    minfo = {"kind": parsed["kind"], "inputs": nu, "outputs": ny, "states": int(model["A"].shape[0]),
             "dead_time_states": int(sum(model["lags"])), "open_loop_spectral_radius": ctrl["open_loop_spectral_radius"],
             "open_loop_stable": stable_ol}
    if stable_ol:
        G0 = model["C"] @ np.linalg.solve(np.eye(model["A"].shape[0]) - model["A"], model["B"])
        minfo["steady_state_gain"] = G0.tolist()
        if nu == ny and nu > 1 and abs(np.linalg.det(G0)) > 1e-12 * np.max(np.abs(G0)) ** nu:
            minfo["relative_gain_array"] = (G0 * np.linalg.inv(G0).T).tolist()
    mev = {}
    if plant_model is None:
        mev = model_evidence(parsed, model, assemble(blocks, nu, ny, Ts / n_sub), Ts, n_sub, min(P + sum(model["lags"]) + 5, 600))
        if mev["two_grid_step_response_max_rel_diff"] > MODEL_TOL:
            reasons.append("las discretizaciones del modelo en Ts y Ts/n no coinciden "
                           f"({mev['two_grid_step_response_max_rel_diff']:.2e})")
        if mev.get("closed_form_step_response_max_rel_diff", 0.0) > MODEL_TOL:
            reasons.append("el modelo discreto no reproduce la respuesta al escalón en forma cerrada "
                           f"({mev['closed_form_step_response_max_rel_diff']:.2e})")
    # --- controller -------------------------------------------------------------------------------------------------
    eig = ctrl["regulator_eigenvalues"]
    eig = eig[np.argsort(-np.abs(eig))]
    nominal_stable = ctrl["regulator_spectral_radius"] < 1 - 1e-9 and (state_feedback or ctrl["estimator_spectral_radius"] < 1 - 1e-9)
    Kinf = infinite_horizon_gain(ctrl)
    cinfo = {"sampling_period": Ts, "prediction_horizon": P, "control_horizon": M, "output_weights": np.diag(Qy).tolist(),
             "move_weights": np.diag(S).tolist(), "input_weights": np.diag(R).tolist(), "disturbance_model": dm,
             "decision_variables": int(ctrl["Hw"].shape[0]), "inequality_constraints": int(ctrl["G"].shape[0]),
             "nominal_closed_loop": {
                 "regulator_spectral_radius": ctrl["regulator_spectral_radius"],
                 "regulator_eigenvalues": [[float(v.real), float(v.imag)] for v in eig[:12]],
                 "estimator_spectral_radius": ctrl["estimator_spectral_radius"], "stable": bool(nominal_stable)},
             "unconstrained_gain": {"state": ctrl["Kreg"][:, :ctrl["n"]].tolist(),
                                    "previous_input": ctrl["Kreg"][:, ctrl["n"]:].tolist(),
                                    "setpoint": ctrl["Kr_const"].tolist(),
                                    "law": "Δu = −K_state·x̂ − K_previous_input·u_prev + K_setpoint·(r − d̂_salida)"}}
    if Kinf is not None:
        cinfo["infinite_horizon_lqr"] = {
            "gain_rel_diff": float(np.linalg.norm(ctrl["Kreg"] - Kinf) / max(np.linalg.norm(Kinf), 1e-300)),
            "note": "distancia de la ganancia sin restricciones a la del LQR de horizonte infinito con el mismo costo; "
                    "tiende a 0 al alargar los horizontes (con M = P). Informativo, no es criterio de verificación"}
    if not nominal_stable:
        warnings.append("el lazo nominal SIN restricciones es inestable con estos horizontes y pesos (radio espectral "
                        f"{max(ctrl['regulator_spectral_radius'], ctrl['estimator_spectral_radius']):.4g} ≥ 1): un MPC sin "
                        "costo terminal no garantiza estabilidad (Rawlings, Mayne y Diehl §1.3.4); alarga "
                        "prediction_horizon o sube move_weights")
    # --- closed loop ------------------------------------------------------------------------------------------------
    sc["rate_div"] = 1.0
    st = main["stats"]
    mpc_out = {"qp": {**{k: st[k] for k in ("solves", "unconstrained", "max_iterations", "not_converged",
                                             "steps_with_active_constraints")},
                      "max_kkt_residual": st["max_kkt"], "max_predicted_slack": st["max_slack"]},
               "diverged": main["diverged"]}
    if main["diverged"]:
        warnings.append("la simulación del MPC diverge (lazo inestable con estos ajustes y este escenario)")
        reasons.append("la simulación en lazo cerrado diverge")
    else:
        U = main["U"]
        act = _activity(U, u_min, u_max, du_max, main["yp"], sc, y_min, y_max, n_sub)
        dU = np.diff(np.vstack([np.zeros((1, nu)), U]), axis=0)
        mpc_out["indices"] = {**main["indices"], "TV_u": np.sum(np.abs(dU), axis=0).tolist(),
                              "u_max_abs": np.max(np.abs(U), axis=0).tolist(),
                              "du_max_abs": np.max(np.abs(dU), axis=0).tolist()}
        mpc_out["constraint_activity"] = act
        mpc_out["settled"] = _settled(main["yp"], U, sc)
        if act["hard_max_rel_violation"] > HARD_TOL:
            reasons.append(f"una restricción dura de entrada se viola en {act['hard_max_rel_violation']:.2e}")
        if not mpc_out["settled"]:
            warnings.append("el lazo con MPC no llegó a régimen al final de la simulación: los índices integrales "
                            "dependen de simulation_time; alárgalo")
        elif np.max(np.abs(main["indices"]["final_error"])) > 1e-2 * max(float(np.max(np.ptp(main["yp"], axis=0))),
                                                                          float(np.max(np.abs(sc["r"]))), 1e-12):
            why = []
            if np.any(np.diag(R) > 0):
                why.append("input_weights > 0 penaliza alejarse de la entrada inicial")
            if nu < ny:
                why.append("hay menos entradas que salidas")
            if dm == "none":
                why.append("no hay modelo de perturbación (sin acción integral)")
            if any(a["fraction_at_max"] + a["fraction_at_min"] > 0 for a in act["inputs"]):
                why.append("una entrada termina saturada")
            warnings.append("queda error en régimen (offset " + ", ".join(f"{v:.4g}" for v in main["indices"]["final_error"])
                            + ")" + (": " + "; ".join(why) if why else ""))
        for o in act["soft_outputs"]:
            if o["max_violation"] > 0:
                warnings.append(f"la salida {o['output']} viola su límite blando hasta en {o['max_violation']:.4g} durante "
                                f"el {100 * o['fraction_of_time_violated']:.1f} % del tiempo (restricción blanda: se "
                                "penaliza, no se impide)")
    if st["not_converged"] or st["max_kkt"] > KKT_TOL:
        reasons.append(f"{st['not_converged']} QP sin converger; residuo KKT máximo {st['max_kkt']:.2e}")
    halving = None
    if half is not None and not main["diverged"] and not half["diverged"]:
        a, b = np.array(main["indices"]["IAE"]), np.array(half["indices"]["IAE"])
        halving = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12 * (1 + np.max(np.abs(b))))))
        if halving > HALVING_TOL:
            reasons.append(f"el IAE cambia {halving:.2e} al partir a la mitad el paso de integración")
    # --- PID baseline -----------------------------------------------------------------------------------------------
    pid_out = dict(pid_info)
    comparison = None
    if loops is not None:
        pr = main["pid"]
        if nu > 1 or ny > 1:
            warnings.append("la línea base PID es multilazo: cada lazo se sintoniza sobre su propio elemento, sin "
                            "desintonizar por interacción (no se aplica BLT ni desacopladores)")
        if len(loops) < ny:
            warnings.append("la línea base PID controla menos salidas que el MPC (no hay entrada para todas)")
        pid_out["diverged"] = pr["diverged"]
        if pr["diverged"]:
            warnings.append("la línea base PID es inestable en este escenario (diverge): no hay índices que comparar")
        else:
            Uf = pr["Uf"]
            sc_pid = dict(sc, rate_div=float(n_sub))
            pact = _activity(Uf, u_min, u_max, du_max, pr["yp"], sc_pid, y_min, y_max, 1)
            dUf = np.diff(np.vstack([np.zeros((1, nu)), Uf]), axis=0)
            pid_out["indices"] = {**pr["indices"], "TV_u": np.sum(np.abs(dUf), axis=0).tolist(),
                                  "u_max_abs": np.max(np.abs(Uf), axis=0).tolist()}
            pid_out["constraint_activity"] = {k: pact[k] for k in ("inputs", "soft_outputs")}
            pid_out["settled"] = _settled(pr["yp"], Uf, sc)
            pev = {"verified": True, "reasons": []}
            if half is not None and not half["pid"]["diverged"]:
                a, b = np.array(pr["indices"]["IAE"]), np.array(half["pid"]["indices"]["IAE"])
                pev["IAE_step_halving_rel_diff"] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12 * (1 + np.max(np.abs(b))))))
                if pev["IAE_step_halving_rel_diff"] > 2e-2:
                    pev["verified"] = False
                    pev["reasons"].append("el IAE del PID cambia más de 2 % al partir el paso a la mitad: da "
                                          "simulation_substeps mayor")
            pid_out["evidence"] = pev
            if not pid_out["settled"]:
                warnings.append("el lazo con PID no llegó a régimen al final de la simulación")
            if not main["diverged"]:
                mi, pi = mpc_out["indices"], pid_out["indices"]

                def ratio(a, b):
                    return [float(x / y) if y > 0 else None for x, y in zip(a, b)]
                comparison = {"IAE_mpc_over_pid": ratio(mi["IAE"], pi["IAE"]), "ISE_mpc_over_pid": ratio(mi["ISE"], pi["ISE"]),
                              "TV_u_mpc_over_pid": ratio(mi["TV_u"], pi["TV_u"]),
                              "note": "cociente < 1 = el MPC es mejor en ese índice; mismo escenario, misma planta y mismos "
                                      "límites de entrada; el PID se ejecuta en continuo (paso fino) y el MPC cada Ts"}
    # --- output -----------------------------------------------------------------------------------------------------
    out = {"model": minfo, "controller": cinfo, "mpc": mpc_out, "pid_baseline": pid_out, "comparison": comparison}
    if return_trajectories and not main["diverged"]:
        t = np.arange(n_steps) * Ts
        fi = np.arange(n_steps) * n_sub
        tr = _series(t, {"setpoint": sc["r"][fi], "mpc_y": main["Y"], "mpc_u": main["U"]})
        if loops is not None and not main["pid"]["diverged"]:
            tr.update({k: v for k, v in _series(t, {"pid_y": main["pid"]["yp"][fi] + sc["dout"][fi],
                                                     "pid_u": main["pid"]["Uf"][fi]}).items() if k != "time"})
        tr["note"] = ("valores en los instantes de muestreo del MPC (salida medida, entrada aplicada durante el periodo "
                      "siguiente); cada serie es una lista por variable")
        out["trajectories"] = tr
        pl = main["plan"]
        if pl is not None:
            out["prediction_at_first_event"] = {
                "time": float(sc["first_event_step"] * Ts), "u_planned": (pl["u_prev"] + np.cumsum(pl["moves"], axis=0)).tolist(),
                "y_predicted": pl["y_pred"].tolist(), "slack": pl["slack"].tolist(),
                "note": "plan en lazo abierto del primer QP tras el primer evento: M movimientos y P salidas predichas"}
    assumptions += [
        "todo en variables de desviación respecto al estado estacionario inicial (u = 0, y = 0 en t = 0); los límites y "
        "consignas se dan en esas mismas variables",
        "unidades de tiempo = las del modelo (constantes de tiempo, tiempo muerto, sampling_period, simulation_time)",
        "retenedor de orden cero: la entrada del MPC es constante durante cada periodo; la consigna futura se supone "
        "constante en el horizonte (sin anticipación)",
        {"output": "perturbación integrante a la SALIDA con ganancia deadbeat (el sesgo de DMC): rechaza despacio las "
                   "perturbaciones que entran antes de una dinámica lenta",
         "input": f"perturbación integrante a la ENTRADA estimada con un filtro de Kalman estacionario (estimator_q = "
                  f"{float(estimator_q):g} = varianza de la perturbación / varianza del ruido de medición; ajuste de sintonía)",
         "none": "sin modelo de perturbación: realimentación del estado completo del modelo, sin acción integral (solo "
                 "válido como regulador nominal)"}[dm],
        "las perturbaciones no se miden (sin acción anticipativa): el MPC solo las ve por su efecto en la salida",
    ]
    if plant_model is not None:
        assumptions.append("la planta simulada es plant_model, distinta del modelo del controlador (prueba de robustez)")
    if loops is not None:
        assumptions.append("PID de la línea base: forma serie del repositorio (derivada sobre la medición filtrada), "
                           "saturación en los mismos límites de entrada, límite de velocidad Δu_max por periodo Ts y "
                           "anti-windup por integración condicional; sin restricciones de salida")
    assumptions += sc["notes"]
    not_included += [
        "MPC no lineal: está en otra herramienta (nmpc_design_and_simulate, modelo EDO del usuario con pyomo+ipopt)",
        "restricción o costo terminal: la estabilidad nominal se comprueba a posteriori (autovalores), no se garantiza",
        "ruido de medición, perturbaciones medidas (acción anticipativa) y anticipación de consigna",
        "estabilidad robusta ante desajuste del modelo: solo se puede explorar caso a caso con plant_model",
        "capa de optimización económica / cálculo de objetivos en estado estacionario (RTO)",
    ]
    ev = {"verified": not reasons, "reasons": reasons,
          "qp": {"solves": st["solves"], "solved_without_active_constraints": st["unconstrained"],
                 "max_kkt_residual": st["max_kkt"], "tolerance": KKT_TOL, "not_converged": st["not_converged"]},
          "hard_constraints_max_rel_violation": mpc_out.get("constraint_activity", {}).get("hard_max_rel_violation"),
          "model_discretization": mev or {"note": "planta distinta del modelo: no se comparan"},
          "IAE_step_halving_rel_diff": halving, "simulation_substeps": n_sub,
          "nominal_unconstrained_loop_stable": bool(nominal_stable),
          "certifies": "la numérica de esta corrida (QP resueltos a su tolerancia KKT, restricciones duras respetadas, "
                       "discretización exacta del tiempo muerto, índices independientes del paso); NO certifica que el "
                       "modelo represente a la planta ni la estabilidad con desajuste"}
    out.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
               method=("MPC lineal tipo QDMC (costo cuadrático en error de salida y movimientos, horizonte de predicción P y "
                       "de control M, restricciones duras de entrada y blandas de salida), modelo discretizado con retenedor "
                       "de orden cero y tiempo muerto exacto (estados de entradas pasadas), QP condensado resuelto con "
                       "Cholesky o punto interior de Mehrotra propio + pulido en el conjunto activo (numpy/scipy), "
                       "estimador de perturbación integrante para seguimiento sin offset; línea base PID con las reglas "
                       "de packages.dynamics.tuning"),
               evidence=ev)
    return out
