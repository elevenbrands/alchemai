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

Optional energy mode (`energy_balance` given; constant molar overflow is then NOT assumed): Raoult's law with the user's
Antoine vapour pressures at the user's stage pressures (constant in time), saturated liquid on every stage
(T_i = bubble point of x_i), ideal-solution enthalpies with constant heat capacities and the heat of vaporisation at
T_ref (Kirchhoff in T, no heat of mixing). With no vapour holdup the liquid enthalpy h_i(x_i) is a function of the
states, so each stage energy balance d(M_i h_i)/dt = Σ in − Σ out (+ F h_F, + Q_R) becomes, by the chain rule
d(M h)/dt = h dM/dt + h'(x) M dx/dt (h' = dh/dx along the bubble curve), a linear equation for the vapour leaving the
stage (index reduction done analytically):
    V_i [H_i − h_i − h'_i (y_i − x_i)] = L_{i+1} [h_{i+1} − h_i − h'_i (x_{i+1} − x_i)]
                                         + V_{i−1} [H_{i−1} − h_i − h'_i (y_{i−1} − x_i)]
                                         (+ F [h_F − h_i − h'_i (z_F − x_i)] on the feed stage)
solved from the reboiler (Q_R, the manipulated variable instead of the boilup) upwards (the λ term of L_{i+1} is
moved to the left). The total condenser returns saturated liquid; its duty results: Q_C = V_{NT−1}[H − h − h'(y − x)].
The liquid holdup enthalpy Σ M_i h_i is not a state: the closure ∫(F h_F + Q_R − Q_C − D h_D − B h_B) dt = ΔΣ M h is a
genuine check of the analytical index reduction and of the integration.
Not modelled: pressure dynamics, vapour holdup, tray efficiency, Francis-weir hydraulics, multicomponent mixtures."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from packages.dynamics.flowsheet import _num, _req      # flowsheet imports this module lazily

LN10 = math.log(10.0)
BUBBLE_TOL = 1e-14          # relative Newton step on the bubble temperature at which it has converged


class _Col:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _vector(v, n: int, where: str, nonneg: bool = False) -> np.ndarray:
    if isinstance(v, (list, tuple)):
        if len(v) != n:
            raise ValueError(f"{where} debe tener {n} valores (uno por etapa, de abajo arriba)")
        return np.array([_num(a, where, nonneg=nonneg) for a in v])
    return np.full(n, _num(v, where, nonneg=nonneg))


def _read_energy(uid: str, eb: dict, NT: int) -> dict:
    w = f"'energy_balance' de '{uid}'"
    if not isinstance(eb, dict):
        raise ValueError(f"{w} debe ser un diccionario {{pressure_Pa, reference_temperature_K, light, heavy}}")
    Pst = _vector(_req(eb, "pressure_Pa", w, "la presión de las etapas (una cifra o una por etapa, de abajo arriba; "
                                            "constante en el tiempo)"), NT, f"pressure_Pa de '{uid}'", nonneg=True)
    if np.any(Pst <= 0.0):
        raise ValueError(f"'{uid}': pressure_Pa debe ser > 0")
    Tref = _num(_req(eb, "reference_temperature_K", w, "la temperatura a la que se dan los calores de vaporización "
                     "(referencia de entalpías)"), f"reference_temperature_K de '{uid}'", positive=True)
    ant, cpl, cpv, dh = [], [], [], []
    for comp in ("light", "heavy"):
        cw = f"{w}.{comp}"
        cd = _req(eb, comp, w, f"las propiedades del {'ligero' if comp == 'light' else 'pesado'} {{antoine {{A, B, C}}, "
                                "cp_liquid_J_molK, cp_vapour_J_molK, heat_of_vaporization_J_mol}")
        ad = _req(cd, "antoine", cw, "la presión de vapor log10(P_sat/Pa) = A − B/(T/K + C)")
        ant.append([_num(_req(ad, k, f"{cw}.antoine", f"la constante {k} de Antoine"), f"Antoine {k} ({comp})")
                    for k in ("A", "B", "C")])
        cpl.append(_num(_req(cd, "cp_liquid_J_molK", cw, "el cp molar del líquido (constante)"), f"cp_liquid ({comp})",
                        nonneg=True))
        cpv.append(_num(_req(cd, "cp_vapour_J_molK", cw, "el cp molar del gas ideal (constante)"), f"cp_vapour ({comp})",
                        nonneg=True))
        dh.append(_num(_req(cd, "heat_of_vaporization_J_mol", cw, "el calor de vaporización a la temperatura de "
                            "referencia"), f"heat_of_vaporization ({comp})", positive=True))
    ant = np.array(ant)
    if np.any(ant[:, 1] <= 0.0):
        raise ValueError(f"'{uid}': la constante B de Antoine debe ser > 0")
    return {"Pst": Pst, "Tref": Tref, "ant": ant, "cpL": np.array(cpl), "cpV": np.array(cpv), "dH": np.array(dh)}


def compile_column(m, uid: str, u: dict) -> _Col:
    """Read the column, create its states and parameters in the model `m` (flowsheet.Model)."""
    w = f"la columna '{uid}'"
    NT = int(_num(_req(u, "stages", w, "el número de etapas incluidos rehervidor (1) y condensador total (NT)"),
                  f"'stages' de '{uid}'", positive=True))
    NF = int(_num(_req(u, "feed_stage", w, "la etapa de alimentación contada desde abajo (rehervidor = 1)"),
                  f"'feed_stage' de '{uid}'", positive=True))
    if NT < 3 or not 2 <= NF <= NT - 1:
        raise ValueError(f"'{uid}': se necesitan ≥ 3 etapas y 2 ≤ feed_stage ≤ stages − 1")
    eb = u.get("energy_balance")
    energy = eb is not None and eb is not False
    alpha = None
    if energy:
        if u.get("relative_volatility") is not None or u.get("k_values") is not None:
            raise ValueError(f"'{uid}': con 'energy_balance' el equilibrio sale de Raoult con las constantes de Antoine de "
                             "energy_balance.light/heavy; quite 'relative_volatility' / 'k_values'")
        if u.get("boilup_mol_s") is not None:
            raise ValueError(f"'{uid}': con 'energy_balance' el vapor resulta de los balances de energía: dé "
                             "'reboiler_duty_W' y quite 'boilup_mol_s'")
        ed = _read_energy(uid, eb, NT)
    elif u.get("relative_volatility") is not None:
        alpha = _num(u["relative_volatility"], f"'relative_volatility' de '{uid}'", positive=True)
    else:
        kv = _req(u, "k_values", w, "el equilibrio: 'relative_volatility' α o 'k_values' {light, heavy} (α = K_l/K_p)")
        kl = _num(_req(kv, "light", f"'k_values' de '{uid}'", "K del ligero"), "K del ligero", positive=True)
        kh = _num(_req(kv, "heavy", f"'k_values' de '{uid}'", "K del pesado"), "K del pesado", positive=True)
        alpha = kl / kh
        m.assumptions.append(f"'{uid}': K constantes del usuario convertidos en volatilidad relativa constante α = "
                             f"{alpha:.6g} (para un binario, y = αx/(1 + (α − 1)x) normaliza y)")
    if alpha is not None and alpha <= 1.0:
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
    manipulated = (("LT", "reflux_mol_s"), ("VB", "boilup_mol_s"), ("D", "distillate_mol_s"), ("B", "bottoms_mol_s"))
    if energy:
        has_q, has_t = fd.get("liquid_fraction") is not None, fd.get("temperature_K") is not None
        if has_q == has_t:
            raise ValueError(f"'{uid}': con 'energy_balance' la alimentación lleva exactamente uno de 'liquid_fraction' "
                             "(saturada, 0 = vapor saturado, 1 = líquido saturado, a la presión de su etapa) o "
                             "'temperature_K' (líquido a esa temperatura, p. ej. subenfriado)")
        if has_q:
            P[keys["qF"]] = _num(fd["liquid_fraction"], keys["qF"], nonneg=True)
            if P[keys["qF"]] > 1.0:
                raise ValueError(f"'{uid}': liquid_fraction debe estar entre 0 y 1 (para líquido subenfriado dé "
                                 "temperature_K)")
        else:
            del keys["qF"]
            keys["TF"] = f"{uid}.feed_temperature_K"
            P[keys["TF"]] = _num(fd["temperature_K"], keys["TF"], positive=True)
        del keys["VB"]
        keys["QR"] = f"{uid}.reboiler_duty_W"
        P[keys["QR"]] = _num(_req(u, "reboiler_duty_W", w, "el calor del rehervidor (variable manipulada; no se supone)"),
                             keys["QR"], nonneg=True)
        manipulated = tuple(mv for mv in manipulated if mv[0] != "VB")
    else:
        P[keys["qF"]] = _num(_req(fd, "liquid_fraction", fw, "la fracción líquida de la alimentación (1 = líquido "
                                  "saturado)"), keys["qF"], nonneg=True)
    for k, what in manipulated:
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
               keys=keys, iM=iM, iN=iN, x=x0.copy(), M=h0.copy(), L=np.zeros(NT), scale=sc, energy=energy)
    if not energy:
        m.assumptions.append(f"'{uid}': columna binaria de {NT} etapas de equilibrio (rehervidor = 1, condensador total = "
                             f"{NT}), volatilidad relativa constante, flujos molares constantes, sin retención de vapor, "
                             "hidráulica lineal de platos L = L0 + (M − M0)/τ_L + λ·ΔV (modelo de Skogestad); sin "
                             "balance de "
                             "energía (modo de flujos molares constantes elegido; active energy_balance para los balances "
                             "de energía por plato) ni dinámica de presión")
        return col
    col.__dict__.update(ed)
    col.antl = [tuple(float(v) for v in row) for row in col.ant]
    col.cpL, col.cpV, col.dH = (tuple(float(v) for v in a) for a in (col.cpL, col.cpV, col.dH))
    col.T = bubble_temperature(col, x0, col.Pst)
    if not np.all(np.isfinite(col.T)):
        raise ValueError(f"'{uid}': no se encontró la temperatura de burbuja de la composición inicial con Antoine")
    col.xT = x0.copy()
    pl, ph, _, _ = _psat(col, col.T)
    if np.any(pl <= ph):
        raise ValueError(f"'{uid}': con las constantes de Antoine dadas el 'light' no es el más volátil en todas las "
                         "etapas (P_sat,ligero ≤ P_sat,pesado)")
    col.V = np.zeros(NT - 1)
    col.QC = 0.0
    col.e_in = col.e_out = 0.0
    col.hF_cache = {}
    col.escale = float(max(np.sum(h0), np.sum(M0)) * max(col.dH))
    m.assumptions.append(f"'{uid}': columna binaria de {NT} etapas de equilibrio (rehervidor = 1, condensador total = {NT}) "
                         "con balance de energía en cada etapa (sin flujos molares constantes): ley de Raoult con presiones "
                         "de vapor de Antoine del usuario a la presión de cada etapa (constante; sin dinámica de presión), "
                         "líquido saturado en cada etapa (T = punto de burbuja), entalpías de solución ideal con cp de "
                         "líquido y de gas ideal constantes y calor de vaporización a "
                         f"{col.Tref:g} K corregido por Kirchhoff, sin calor de mezcla ni pérdidas; sin retención de "
                         "vapor; el calor del rehervidor es la variable manipulada (en lugar del vapor) y el condensador "
                         "total devuelve líquido saturado (su calor resulta); hidráulica lineal de platos L = L0 + "
                         "(M − M0)/τ_L + λ·ΔV")
    return col


# ------------------------------------------------------------------------------------------------ energy mode
def _psat(col: _Col, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Antoine P_sat of light and heavy and their T-derivatives at the temperatures T."""
    (al, bl, cl), (ah, bh, ch) = col.antl
    tl, th = T + cl, T + ch
    pl, ph = 10.0 ** (al - bl / tl), 10.0 ** (ah - bh / th)
    return pl, ph, pl * (LN10 * bl) / (tl * tl), ph * (LN10 * bh) / (th * th)


def bubble_temperature(col: _Col, x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Bubble temperature of the binary liquids x at the pressures P (Raoult): Newton on ln(x P_l + (1 − x) P_h) = ln P,
    vectorised over stages, always from the composition-weighted pure saturation temperatures (no warm start: the
    result is a deterministic function of x, which a stiff integrator at tight tolerance needs). NaN where it does not
    converge."""
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    (al, bl, cl), (ah, bh, ch) = col.antl
    tmin = max(-cl, -ch) + 1.0
    lp = np.log10(P)
    T = x * (bl / (al - lp) - cl) + (1.0 - x) * (bh / (ah - lp) - ch)
    done = False
    for _ in range(60):
        if not np.all(T > tmin):
            T = np.where(np.isfinite(T), np.maximum(T, tmin + 1.0), tmin + 50.0)
        pl, ph, dpl, dph = _psat(col, T)
        f = x * pl + (1.0 - x) * ph
        dT = -np.log(f / P) * f / (x * dpl + (1.0 - x) * dph)
        worst = float(np.max(np.abs(dT) / T))
        if not math.isfinite(worst):
            return np.full(x.shape, np.nan)
        if worst > 0.2:
            dT = np.clip(dT, -0.2 * T, 0.2 * T)
        T = T + dT
        if done:
            return T
        if worst < BUBBLE_TOL:
            done = True                                           # one more step for the last digits
    return np.full(x.shape, np.nan)


def _stage_properties(col: _Col, x: np.ndarray, T: np.ndarray) -> tuple:
    """Liquid enthalpy h, its derivative along the bubble curve h' = dh/dx, vapour composition y and vapour enthalpy H
    (y and H of the vapour in equilibrium with each stage's liquid), J/mol."""
    pl, ph, dpl, dph = _psat(col, T)
    (cl, ch), (vl, vh), (dl, dh) = col.cpL, col.cpV, col.dH
    dT = T - col.Tref
    hl, hh = cl * dT, ch * dT
    h = x * hl + (1.0 - x) * hh
    hp = (hl - hh) - (x * cl + (1.0 - x) * ch) * (pl - ph) / (x * dpl + (1.0 - x) * dph)
    yv = x * pl / (x * pl + (1.0 - x) * ph)              # Raoult, normalised (insensitive to round-off in T)
    H = yv * (dl + vl * dT) + (1.0 - yv) * (dh + vh * dT)
    return h, hp, yv, H


def feed_enthalpy(col: _Col, P: dict) -> float:
    """Molar enthalpy of the feed: liquid at feed_temperature_K, or saturated at the feed-stage pressure with liquid
    fraction q (flash at fixed vapour fraction 1 − q, brentq in T between bubble and dew point)."""
    zF = P[col.keys["zF"]]
    if "TF" in col.keys:
        TF = P[col.keys["TF"]]
        return float((zF * col.cpL[0] + (1.0 - zF) * col.cpL[1]) * (TF - col.Tref))
    q = P[col.keys["qF"]]
    key = (zF, q)
    if key in col.hF_cache:
        return col.hF_cache[key]
    Pf = float(col.Pst[col.NF - 1])
    b = 1.0 - q
    z = np.array([zF, 1.0 - zF])
    A, B, C = col.ant[:, 0], col.ant[:, 1], col.ant[:, 2]

    def ps(T):
        return 10.0 ** (A - B / (T + C))

    def g(T):                                                   # Rachford-Rice at fixed β, increasing in T
        K = ps(T) / Pf
        return float(np.sum(z * (K - 1.0) / (1.0 + b * (K - 1.0))))

    tsat = B / (A - math.log10(Pf)) - C
    lo, hi = float(np.min(tsat)) - 1.0, float(np.max(tsat)) + 1.0
    T = float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-15, maxiter=500))
    K = ps(T) / Pf
    xl = z / (1.0 + b * (K - 1.0))
    yl = K * xl
    dT = T - col.Tref
    cpl, cpv, dh = np.array(col.cpL), np.array(col.cpV), np.array(col.dH)
    hF = float((1.0 - b) * np.sum(xl * cpl * dT) + b * np.sum(yl * (dh + cpv * dT)))
    if len(col.hF_cache) > 64:
        col.hF_cache.clear()
    col.hF_cache[key] = hF
    return hF


def refresh_temperatures(col: _Col):
    """Bubble temperatures of the current liquids (called before the controllers read the column)."""
    _temperatures(col, col.x)


def _temperatures(col: _Col, x: np.ndarray) -> np.ndarray:
    """Bubble temperatures of x, reused when x has not changed since the last call (controllers, then balances)."""
    if np.array_equal(x, col.xT):
        return col.T
    T = bubble_temperature(col, x, col.Pst)
    if np.all(np.isfinite(T)):
        col.T, col.xT = T, x.copy()
    return T


def holdup_enthalpy(col: _Col, y: np.ndarray) -> float:
    """Σ M_i h_i of the liquid holdups, J (reference: liquid at T_ref)."""
    M, x = state_views(col, y)
    T = bubble_temperature(col, x, col.Pst)
    h = _stage_properties(col, x, T)[0]
    return float(np.sum(M * h))


def _column_rhs_energy(col: _Col, y: np.ndarray, dy: np.ndarray, P: dict) -> tuple[float, float, float, float]:
    NT, NF, lam = col.NT, col.NF, col.lam
    M, x = state_views(col, y)
    col.M, col.x = M, x
    F, zF = P[col.keys["F"]], P[col.keys["zF"]]
    LT, D, B, QR = P[col.keys["LT"]], P[col.keys["D"]], P[col.keys["B"]], P[col.keys["QR"]]

    def fail():
        dy[col.iM:col.iM + NT] = np.nan
        dy[col.iN:col.iN + NT] = np.nan
        col.e_in = col.e_out = 0.0
        return 0.0, 0.0, 0.0, 0.0

    if np.any(M <= 0.0) or not np.all(np.isfinite(x)):
        return fail()
    T = _temperatures(col, x)
    if not np.all(np.isfinite(T)):
        return fail()
    h, hp, yv, H = _stage_properties(col, x, T)
    hF = feed_enthalpy(col, P)
    Lb = np.zeros(NT)                                            # tray liquid flow without its λ·V_{i−1} term
    i = np.arange(1, NF)
    Lb[i] = col.L0b + (M[i] - col.M0[i]) / col.tau - lam * col.V0
    i = np.arange(NF, NT - 1)
    Lb[i] = col.L0 + (M[i] - col.M0[i]) / col.tau - lam * col.V0t
    # coefficients of V_j D_j = L_{j+1} A_j + V_{j−1} E_j (+ F G_j), vectorised; the recursion itself runs on floats
    A = h[1:] - h[:-1] - hp[:-1] * (x[1:] - x[:-1])             # j = 0 … NT−2
    Dd = H - h - hp * (yv - x)                                  # j = 0 … NT−1 (last unused)
    E = np.zeros(NT)
    E[1:] = H[:-1] - h[1:] - hp[1:] * (yv[:-1] - x[1:])
    A, Dd, E, Lbl = A.tolist(), Dd.tolist(), E.tolist(), Lb.tolist()
    g = F * (hF - h[NF - 1] - hp[NF - 1] * (zF - x[NF - 1]))
    V = [0.0] * (NT - 1)
    Ll = [0.0] * NT
    Ll[NT - 1] = LT
    v = 0.0
    for j in range(NT - 1):                                     # reboiler (j = 0, Q_R) and trays upwards
        r = QR if j == 0 else v * E[j]
        if j == NF - 1:
            r += g
        if j + 1 <= NT - 2:
            dj = Dd[j] - lam * A[j]
            if not dj > 0.0:
                return fail()
            v = (Lbl[j + 1] * A[j] + r) / dj
            Ll[j + 1] = Lbl[j + 1] + lam * v
        else:
            if not Dd[j] > 0.0:
                return fail()
            v = (LT * A[j] + r) / Dd[j]
        if v < 0.0:                                             # a stage with no vapour: outside the model
            return fail()
        V[j] = v
    V, L = np.array(V), np.array(Ll)
    k = NT - 1
    col.QC = float(V[k - 1] * (H[k - 1] - h[k] - hp[k] * (yv[k - 1] - x[k])))
    col.V, col.L = V, L
    dM = np.zeros(NT)
    dn = np.zeros(NT)
    j = np.arange(1, NT - 1)
    dM[j] = L[j + 1] - L[j] + V[j - 1] - V[j]
    dn[j] = L[j + 1] * x[j + 1] - L[j] * x[j] + V[j - 1] * yv[j - 1] - V[j] * yv[j]
    dM[NF - 1] += F
    dn[NF - 1] += F * zF
    dM[0] = L[1] - V[0] - B
    dn[0] = L[1] * x[1] - V[0] * yv[0] - B * x[0]
    dM[k] = V[k - 1] - LT - D
    dn[k] = V[k - 1] * yv[k - 1] - LT * x[k] - D * x[k]
    dy[col.iM:col.iM + NT] = dM
    dy[col.iN:col.iN + NT] = dn
    col.e_in = F * hF + QR
    col.e_out = D * h[k] + B * h[0] + col.QC
    return F, F * zF, D + B, D * x[k] + B * x[0]


# ------------------------------------------------------------------------------------------------ both modes
def state_views(col: _Col, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = y[col.iM:col.iM + col.NT]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = y[col.iN:col.iN + col.NT] / M
    return M, x


def column_rhs(col: _Col, y: np.ndarray, dy: np.ndarray, P: dict) -> tuple[float, float, float, float]:
    """Writes dM/dt and d(Mx)/dt; returns (feed, light fed, products, light in products) per second."""
    if col.energy:
        return _column_rhs_energy(col, y, dy, P)
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
    if not col.energy:
        return

    def t_pure(y, c, k):
        x = np.array([y[c.iN + k] / y[c.iM + k]])
        return float(bubble_temperature(c, x, c.Pst[k:k + 1])[0])

    for k in range(NT):
        g[f"{uid}.temperature_K.{k + 1}"] = lambda c=col, k=k: float(c.T[k])
        p[f"{uid}.temperature_K.{k + 1}"] = lambda y, c=col, k=k: t_pure(y, c, k)
        m.verbose.add(f"{uid}.temperature_K.{k + 1}")
    for k in range(NT - 1):
        g[f"{uid}.vapour_flow_mol_s.{k + 1}"] = lambda c=col, k=k: float(c.V[k])
        m.verbose.add(f"{uid}.vapour_flow_mol_s.{k + 1}")
    for name, k in (("top_temperature_K", NT - 1), ("bottom_temperature_K", 0)):
        g[f"{uid}.{name}"] = g[f"{uid}.temperature_K.{k + 1}"]
        p[f"{uid}.{name}"] = p[f"{uid}.temperature_K.{k + 1}"]
    g[f"{uid}.condenser_duty_W"] = lambda c=col: float(c.QC)
    g[f"{uid}.reboiler_vapour_mol_s"] = lambda c=col: float(c.V[0])
    g[f"{uid}.top_vapour_mol_s"] = lambda c=col: float(c.V[-1])
