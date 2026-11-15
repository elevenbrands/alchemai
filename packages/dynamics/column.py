"""Dynamic binary distillation column as a unit of the flowsheet engine (`packages.dynamics.flowsheet`).

Model (Skogestad's "column A" model, https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html): NT stages counted from
the bottom (1 = reboiler, NT = total condenser), feed at stage NF mixed into the liquid, constant relative volatility,
constant molar flows, equilibrium on every stage, no vapour holdup, linearized liquid (tray) hydraulics:
    y_i = α x_i / (1 + (α − 1) x_i)
    V_i = V_B (i < NF),  V_i = V_B + (1 − q_F) F (NF ≤ i < NT)
    L_i = L0_i + (M_i − M0_i)/τ_L + λ (V_{i−1} − V0_{i−1})        trays 2 … NT−1 (L0 below / above the feed)
    L_NT = L_T (reflux)
    dM_i/dt = L_{i+1} − L_i + V_{i−1} − V_i (+ F at NF);   d(M_i x_i)/dt likewise with x and y (+ F z_F at NF)
    reboiler:  dM_1/dt = L_2 − V_B − B,          condenser: dM_NT/dt = V_{NT−1} − L_T − D
States are conservative (M_i and M_i x_i), so the light-component and total balances close over a transient. Reflux,
boilup, distillate and bottoms are parameters: the engine's PID controllers manipulate them (LV, DV, … configurations).
Not modelled: energy balance, pressure dynamics, vapour holdup, tray efficiency, Francis-weir hydraulics."""
from __future__ import annotations

import numpy as np

from packages.dynamics.flowsheet import _num, _req      # flowsheet imports this module lazily


class _Col:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _vector(v, n: int, where: str, nonneg: bool = False) -> np.ndarray:
    if isinstance(v, (list, tuple)):
        if len(v) != n:
            raise ValueError(f"{where} debe tener {n} valores (uno por etapa, de abajo arriba)")
        return np.array([_num(a, where, nonneg=nonneg) for a in v])
    return np.full(n, _num(v, where, nonneg=nonneg))


def compile_column(m, uid: str, u: dict) -> _Col:
    """Read the column, create its states and parameters in the model `m` (flowsheet.Model)."""
    w = f"la columna '{uid}'"
    NT = int(_num(_req(u, "stages", w, "el número de etapas incluidos rehervidor (1) y condensador total (NT)"),
                  f"'stages' de '{uid}'", positive=True))
    NF = int(_num(_req(u, "feed_stage", w, "la etapa de alimentación contada desde abajo (rehervidor = 1)"),
                  f"'feed_stage' de '{uid}'", positive=True))
    if NT < 3 or not 2 <= NF <= NT - 1:
        raise ValueError(f"'{uid}': se necesitan ≥ 3 etapas y 2 ≤ feed_stage ≤ stages − 1")
    if u.get("relative_volatility") is not None:
        alpha = _num(u["relative_volatility"], f"'relative_volatility' de '{uid}'", positive=True)
    else:
        kv = _req(u, "k_values", w, "el equilibrio: 'relative_volatility' α o 'k_values' {light, heavy} (α = K_l/K_p)")
        kl = _num(_req(kv, "light", f"'k_values' de '{uid}'", "K del ligero"), "K del ligero", positive=True)
        kh = _num(_req(kv, "heavy", f"'k_values' de '{uid}'", "K del pesado"), "K del pesado", positive=True)
        alpha = kl / kh
        m.assumptions.append(f"'{uid}': K constantes del usuario convertidos en volatilidad relativa constante α = "
                             f"{alpha:.6g} (para un binario, y = αx/(1 + (α − 1)x) normaliza y)")
    if alpha <= 1.0:
        raise ValueError(f"'{uid}': la volatilidad relativa del ligero debe ser > 1")
    hy = _req(u, "hydraulics", w, "la hidráulica lineal de platos {tau_L_s, lambda, nominal_holdup_mol, "
                                  "nominal_liquid_flow_top_mol_s, nominal_liquid_flow_bottom_mol_s}")
    hw = f"'hydraulics' de '{uid}'"
    tau = _num(_req(hy, "tau_L_s", hw, "la constante de tiempo hidráulica de cada plato"), f"tau_L_s de '{uid}'",
               positive=True)
    lam = _num(_req(hy, "lambda", hw, "el efecto del vapor sobre el líquido (efecto K2; 0 si no se conoce)"),
               f"lambda de '{uid}'")
    M0 = _vector(_req(hy, "nominal_holdup_mol", hw, "la retención nominal de los platos"), NT,
                 f"nominal_holdup_mol de '{uid}'", nonneg=True)
    L0 = _num(_req(hy, "nominal_liquid_flow_top_mol_s", hw, "el caudal de líquido nominal sobre la alimentación"),
              f"nominal_liquid_flow_top_mol_s de '{uid}'", nonneg=True)
    L0b = _num(_req(hy, "nominal_liquid_flow_bottom_mol_s", hw, "el caudal de líquido nominal bajo la alimentación"),
               f"nominal_liquid_flow_bottom_mol_s de '{uid}'", nonneg=True)
    V0 = V0t = 0.0
    if lam != 0.0:
        V0 = _num(_req(hy, "nominal_vapour_flow_bottom_mol_s", hw, "el efecto K2 (λ ≠ 0)"), "V0 bajo la alimentación",
                  nonneg=True)
        V0t = _num(_req(hy, "nominal_vapour_flow_top_mol_s", hw, "el efecto K2 (λ ≠ 0)"), "V0 sobre la alimentación",
                   nonneg=True)
    fd = _req(u, "feed", w, "la alimentación {molar_flow_mol_s, mole_fraction_light, liquid_fraction}")
    fw = f"'feed' de '{uid}'"
    P = m.P
    keys = {"F": f"{uid}.feed_mol_s", "zF": f"{uid}.feed_mole_fraction", "qF": f"{uid}.feed_liquid_fraction",
            "LT": f"{uid}.reflux_mol_s", "VB": f"{uid}.boilup_mol_s", "D": f"{uid}.distillate_mol_s",
            "B": f"{uid}.bottoms_mol_s"}
    P[keys["F"]] = _num(_req(fd, "molar_flow_mol_s", fw, "el caudal de alimentación"), keys["F"], nonneg=True)
    P[keys["zF"]] = _num(_req(fd, "mole_fraction_light", fw, "la composición de la alimentación"), keys["zF"], nonneg=True)
    P[keys["qF"]] = _num(_req(fd, "liquid_fraction", fw, "la fracción líquida de la alimentación (1 = líquido saturado)"),
                         keys["qF"], nonneg=True)
    for k, what in (("LT", "reflux_mol_s"), ("VB", "boilup_mol_s"), ("D", "distillate_mol_s"), ("B", "bottoms_mol_s")):
        P[keys[k]] = _num(_req(u, what, w, f"el caudal '{what}' (variable manipulada; no se supone)"), keys[k],
                          nonneg=True)
    ini = _req(u, "initial", w, "el estado inicial {x (fracción del ligero por etapa o una sola), holdup_mol}")
    x0 = _vector(_req(ini, "x", f"'initial' de '{uid}'", "las composiciones iniciales"), NT, f"initial.x de '{uid}'",
                 nonneg=True)
    h0 = _vector(_req(ini, "holdup_mol", f"'initial' de '{uid}'", "las retenciones iniciales"), NT,
                 f"initial.holdup_mol de '{uid}'", nonneg=True)
    if np.any(x0 > 1.0) or np.any(h0 <= 0.0):
        raise ValueError(f"'{uid}': initial.x debe estar en [0, 1] y initial.holdup_mol ser > 0")
    sc = float(max(np.max(h0), np.max(M0), 1e-30))
    iM = len(m._y0)
    for k in range(NT):
        m._state(f"{uid}.holdup_mol.{k + 1}", h0[k], sc)
    iN = len(m._y0)
    for k in range(NT):
        m._state(f"{uid}.light_mol.{k + 1}", h0[k] * x0[k], sc)
    col = _Col(id=uid, kind="column", NT=NT, NF=NF, alpha=alpha, tau=tau, lam=lam, M0=M0, L0=L0, L0b=L0b, V0=V0, V0t=V0t,
               keys=keys, iM=iM, iN=iN, x=x0.copy(), M=h0.copy(), L=np.zeros(NT), scale=sc)
    m.assumptions.append(f"'{uid}': columna binaria de {NT} etapas de equilibrio (rehervidor = 1, condensador total = "
                         f"{NT}), volatilidad relativa constante, flujos molares constantes, sin retención de vapor, "
                         "hidráulica lineal de platos L = L0 + (M − M0)/τ_L + λ·ΔV (modelo de Skogestad); sin balance de "
                         "energía ni dinámica de presión")
    return col


def state_views(col: _Col, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = y[col.iM:col.iM + col.NT]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = y[col.iN:col.iN + col.NT] / M
    return M, x


def column_rhs(col: _Col, y: np.ndarray, dy: np.ndarray, P: dict) -> tuple[float, float, float, float]:
    """Writes dM/dt and d(Mx)/dt; returns (feed, light fed, products, light in products) per second."""
    NT, NF, a = col.NT, col.NF, col.alpha
    M, x = state_views(col, y)
    col.M, col.x = M, x
    F, zF, qF = P[col.keys["F"]], P[col.keys["zF"]], P[col.keys["qF"]]
    LT, VB, D, B = P[col.keys["LT"]], P[col.keys["VB"]], P[col.keys["D"]], P[col.keys["B"]]
    if np.any(M <= 0.0):                                  # a stage ran dry: no composition; the integrator rejects it
        dy[col.iM:col.iM + NT] = np.nan
        dy[col.iN:col.iN + NT] = np.nan
        return 0.0, 0.0, 0.0, 0.0
    yv = a * x[:NT - 1] / (1.0 + (a - 1.0) * x[:NT - 1])
    V = np.full(NT - 1, VB)
    V[NF - 1:] += (1.0 - qF) * F
    L = np.zeros(NT)
    i = np.arange(1, NF)                                  # stages 2 … NF (0-based 1 … NF−1)
    L[i] = col.L0b + (M[i] - col.M0[i]) / col.tau + col.lam * (V[i - 1] - col.V0)
    i = np.arange(NF, NT - 1)                             # stages NF+1 … NT−1
    L[i] = col.L0 + (M[i] - col.M0[i]) / col.tau + col.lam * (V[i - 1] - col.V0t)
    L[NT - 1] = LT
    col.L = L
    dM = np.zeros(NT)
    dn = np.zeros(NT)
    j = np.arange(1, NT - 1)
    dM[j] = L[j + 1] - L[j] + V[j - 1] - V[j]
    dn[j] = L[j + 1] * x[j + 1] - L[j] * x[j] + V[j - 1] * yv[j - 1] - V[j] * yv[j]
    dM[NF - 1] += F
    dn[NF - 1] += F * zF
    dM[0] = L[1] - V[0] - B
    dn[0] = L[1] * x[1] - V[0] * yv[0] - B * x[0]
    dM[NT - 1] = V[NT - 2] - LT - D
    dn[NT - 1] = V[NT - 2] * yv[NT - 2] - LT * x[NT - 1] - D * x[NT - 1]
    dy[col.iM:col.iM + NT] = dM
    dy[col.iN:col.iN + NT] = dn
    return F, F * zF, D + B, D * x[NT - 1] + B * x[0]


def register_variables(m, col: _Col):
    """Getters (current values) and pure functions of the state (for controllers and dead time)."""
    g, p, uid, NT = m.getters, m.pure, col.id, col.NT
    for k in range(NT):
        g[f"{uid}.x.{k + 1}"] = lambda c=col, k=k: float(c.x[k])
        p[f"{uid}.x.{k + 1}"] = lambda y, c=col, k=k: float(y[c.iN + k] / y[c.iM + k])
        g[f"{uid}.holdup_mol.{k + 1}"] = lambda c=col, k=k: float(c.M[k])
        p[f"{uid}.holdup_mol.{k + 1}"] = lambda y, c=col, k=k: float(y[c.iM + k])
        m.verbose.add(f"{uid}.x.{k + 1}")
        m.verbose.add(f"{uid}.holdup_mol.{k + 1}")
    for name, k in (("distillate_x", NT - 1), ("bottoms_x", 0)):
        g[f"{uid}.{name}"] = g[f"{uid}.x.{k + 1}"]
        p[f"{uid}.{name}"] = p[f"{uid}.x.{k + 1}"]
    for name, k in (("condenser_holdup_mol", NT - 1), ("reboiler_holdup_mol", 0)):
        g[f"{uid}.{name}"] = g[f"{uid}.holdup_mol.{k + 1}"]
        p[f"{uid}.{name}"] = p[f"{uid}.holdup_mol.{k + 1}"]
