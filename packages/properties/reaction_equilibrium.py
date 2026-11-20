"""Chemical-reaction equilibrium of an ideal-gas mixture from thermochemical (formation) data: `reaction_equilibrium`.

Standard state: pure ideal gas at T and P° = 1 bar, for EVERY species (gas-phase ΔfH°, ΔfG° at 298.15 K).

  ΔH°rxn(298), ΔG°rxn(298) = Σ ν_i·ΔfH°_i, Σ ν_i·ΔfG°_i                         K(298) = exp(−ΔG°/R·T0)
  g_i(T) = ΔfH°_i + ∫Cp_i dT − T·[(ΔfH°_i − ΔfG°_i)/T0 + ∫(Cp_i/T) dT]        (ideal-gas Cp, limits T0 → T)
  ln K(T) = −Σ ν_i·g_i(T)/(R·T)      — the rigorous van't Hoff integration (Smith, Van Ness & Abbott, eqs. 13.11-13.18);
            the reference-element terms left out of g_i cancel in any element-balanced reaction.
  ln K(T) ≈ ln K(298) − ΔH°(298)/R·(1/T − 1/T0)                               — constant-ΔH° approximation (labelled)

Equilibrium at T, P and feed n0: n = n0 + N·ξ (N = stoichiometric matrix, ξ = extents); the extents minimise the convex
G(ξ)/RT = Σ_j ξ_j·ΔG°_j/RT + Σ_i n_i·ln(y_i·φ_i·P/P°), i.e. Π (y_i·φ_i·P/P°)^ν_ij = K_j for every reaction (damped Newton
with the analytic Hessian, started from the strictly feasible point of a linear programme). φ_i = 1 (ideal gas) unless
the user gives fugacity coefficients, which are then held constant.

Evidence, all computed here: (1) K consistency, Π a_i^ν at the solution against K; (2) element balance; (3) the same
equilibrium by a DIFFERENT formulation — Lagrange-multiplier Gibbs minimisation over the species (n_i = n_tot·exp(−g_i/RT
− ln(φ_i·P/P°) + Σ_k λ_k·B_ki), with B the element matrix when the given reactions span every independent reaction
among the species, or the invariants of the given reactions when they are a chosen subset); (4) ln K(T) by numerical
quadrature of d ln K/dT = ΔH°(T)/RT² on point values of Cp, against the analytic integrals; (5) Cantera (NASA
polynomials of a shipped mechanism, GRI-Mech 3.0 by default) for ln K(T) and, when the reactions span the full space,
for the composition of a phase restricted to the same species. (5) compares DATA SOURCES, not arithmetic.

Data: ΔfH°gas from `chemicals.reaction.Hfg`, S°gas from `chemicals.reaction.S0g`, ΔfG°gas derived by
`thermo.ChemicalConstantsPackage` (ΔfH° − T0·[S° − ΣS° of the elements]), Cp from `thermo.HeatCapacityGas`. A compound
without formation data stops the tool; the user's ΔfH°, ΔfG°, Cp are accepted and labelled "dato del usuario".
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog, root

from packages.properties import pure as PU

R_GAS = PU.R_GAS
T0 = PU.T_STD
P_STD = 1.0e5                 # Pa, standard-state pressure of the formation data (1 bar)
STOICH_TOL = 1e-8             # element imbalance tolerated in a reaction (atoms per unit of the largest coefficient)
LNK_QUAD_TOL = 1e-5           # |ln K analytic − ln K by numerical quadrature|
LNK_CONSISTENCY_TOL = 1e-8    # |Σ ν ln a − ln K| at the solution
ELEMENT_TOL = 1e-10           # relative element-balance error
GIBBS_Y_TOL = 1e-7            # |Δy| between the extent solution and the Lagrange-multiplier Gibbs minimisation
# Cantera vs this databank: different compilations (ATcT/CRC here, NASA/Burcat fits there). 0.5 kJ/mol per mole of
# species in the reaction is the usual spread of ΔfH° between evaluated compilations (e.g. NH3 −45.56 ATcT vs −45.90 CRC).
CANTERA_DG_TOL_J_PER_NU = 500.0
CANTERA_Y_TOL = 2e-2          # a ΔG° difference inside the tolerance above moves K by tens of percent, hence y by ~0.01
CP_PREFERENCE = ("TRCIG", "WEBBOOK_SHOMATE", "JANAF", "HEOS_FIT", "COOLPROP", "POLING_POLY", "JOBACK")
CP_ESTIMATES = {"JOBACK", "LASTOVKA_SHAW"}
USER = "dato del usuario"
# stable molecules of GRI-Mech 3.0 identified by CAS (a formula match alone could pick the wrong isomer)
GRI_BY_CAS = {"1333-74-0": "H2", "7782-44-7": "O2", "7732-18-5": "H2O", "7722-84-1": "H2O2", "630-08-0": "CO",
              "124-38-9": "CO2", "74-82-8": "CH4", "67-56-1": "CH3OH", "50-00-0": "CH2O", "74-86-2": "C2H2",
              "74-85-1": "C2H4", "74-84-0": "C2H6", "74-98-6": "C3H8", "7727-37-9": "N2", "7664-41-7": "NH3",
              "10102-43-9": "NO", "10102-44-0": "NO2", "10024-97-2": "N2O", "74-90-8": "HCN", "7440-37-1": "AR"}


class _UserCp:
    """Cp of the user: a number (constant, J/(mol K)); a list [c0, c1, ...] of Cp = Σ c_k·T^k in J/(mol K); or a dict
    {"A", "B", "C", "D"} of the textbook form Cp/R = A + B·T + C·T² + D·T⁻² (Smith-Van Ness table C.1), T in K."""
    method = USER
    MSG = ("Cp_J_molK del usuario: un número (Cp constante, J/(mol·K)), la lista [c0, c1, ...] de Cp = c0 + c1·T + c2·T² "
           "+ ... en J/(mol·K), o {'A', 'B', 'C', 'D'} de Cp/R = A + B·T + C·T² + D/T² (T en K).")

    def __init__(self, spec):
        try:
            if isinstance(spec, dict):
                if not spec or any(k not in ("A", "B", "C", "D") for k in spec):
                    raise ValueError
                self.given = {k: float(v) for k, v in spec.items()}
                self.c = [R_GAS * self.given.get(k, 0.0) for k in ("A", "B", "C")]
                self.d = R_GAS * self.given.get("D", 0.0)
            else:
                self.given = [float(v) for v in (spec if isinstance(spec, (list, tuple)) else [spec])]
                self.c, self.d = list(self.given), 0.0
            if not self.c or not all(math.isfinite(v) for v in [*self.c, self.d]):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(self.MSG) from None

    def cp(self, T):
        return sum(c * T ** k for k, c in enumerate(self.c)) + self.d / T ** 2

    def dH(self, T):
        return sum(c * (T ** (k + 1) - T0 ** (k + 1)) / (k + 1) for k, c in enumerate(self.c)) - self.d * (1 / T - 1 / T0)

    def dS(self, T):
        return self.c[0] * math.log(T / T0) + sum(c * (T ** k - T0 ** k) / k for k, c in enumerate(self.c) if k > 0) \
            - self.d / 2 * (T ** -2 - T0 ** -2)

    def describe(self, Ts):
        return {"method": USER, "source": USER, "given": self.given, "in_range": True, "estimated": False,
                "note": "intervalo de validez no declarado: lo responde el usuario"}


class _LibCp:
    """Ideal-gas Cp of thermo.HeatCapacityGas with ONE method fixed for every temperature of the call."""

    def __init__(self, cas: str, name: str, Ts):
        from thermo import HeatCapacityGas
        self.obj = HeatCapacityGas(CASRN=cas)
        lim = dict(self.obj.T_limits)
        lo, hi = min([T0, *Ts]), max([T0, *Ts])
        avail = [m for m in CP_PREFERENCE if m in lim] + [m for m in lim if m not in CP_PREFERENCE and "_SHOMATE_" not in m
                                                         and m not in ("POLING_CONST", "CRCSTD", "VDI_TABULAR")]
        if not avail:
            raise ValueError(f"Falta el Cp de gas ideal de «{name}» en la base de datos de thermo. Pide al usuario su Cp y "
                             f"pásalo en data_overrides={{'{name}': {{'Cp_J_molK': número o [c0, c1, ...]}}}}; no se inventa.")
        cover = [m for m in avail if lim[m][0] <= lo + 1e-6 and lim[m][1] >= hi - 1e-6]
        data_cover = [m for m in cover if m not in CP_ESTIMATES]
        self.method = (data_cover or cover or avail)[0]
        self.limits = tuple(float(v) for v in lim[self.method])
        self.in_range = bool(cover)

    def cp(self, T):
        return float(self.obj.calculate(T, self.method))

    def dH(self, T):
        return float(self.obj.calculate_integral(T0, T, self.method))

    def dS(self, T):
        return float(self.obj.calculate_integral_over_T(T0, T, self.method))

    def describe(self, Ts):
        return {"method": self.method, "source": f"thermo.HeatCapacityGas, método {self.method}",
                "T_range_K": list(self.limits), "in_range": self.in_range, "estimated": self.method in CP_ESTIMATES}


def _species(name: str, override: dict | None, Ts) -> dict:
    from chemicals import reaction
    meta, _ = PU.identify(name)
    cas = meta.CASs
    with PU._thermo_quiet():
        pkg = PU._package(cas)[0]
    atoms = {k: float(v) for k, v in (pkg.atomss[0] or {}).items()}
    ov = dict(override or {})
    unknown = [k for k in ov if k not in ("Hf_J_mol", "Gf_J_mol", "Cp_J_molK")]
    if unknown:
        raise ValueError(f"data_overrides de «{name}»: claves desconocidas {unknown}; se aceptan Hf_J_mol, Gf_J_mol (gas "
                         "ideal, 298.15 K, 1 bar) y Cp_J_molK.")
    hf_lib = reaction.Hfg(cas)
    s0_lib = reaction.S0g(cas)
    sf_lib = pkg.Sfgs[0] if s0_lib is not None else None
    lib_ok = hf_lib is not None and pkg.Hfgs[0] is not None and abs(pkg.Hfgs[0] - hf_lib) < 1e-6 * max(1.0, abs(hf_lib))
    if ov.get("Hf_J_mol") is not None:
        hf, hf_src = float(ov["Hf_J_mol"]), USER
    elif hf_lib is not None:
        hf, hf_src = float(hf_lib), f"chemicals.reaction.Hfg, método {reaction.Hfg_methods(cas)[0]}"
    else:
        hf = hf_src = None
    if ov.get("Gf_J_mol") is not None:
        gf, gf_src = float(ov["Gf_J_mol"]), USER
    elif hf is not None and sf_lib is not None and (lib_ok or hf_src == USER):
        gf = hf - T0 * float(sf_lib)
        s_src = f"S°gas de chemicals.reaction.S0g, método {reaction.S0g_methods(cas)[0]}"
        gf_src = (f"derivado: ΔfH° − T0·(S°gas − ΣS° de los elementos); {s_src}; entropías de los elementos de "
                  "chemicals.periodic_table (thermo.ChemicalConstantsPackage)")
        if hf_src == USER:
            gf_src += f"; ΔfH° = {USER}"
    else:
        gf = gf_src = None
    missing = [lab for lab, v in (("ΔfH°gas (Hf_J_mol)", hf), ("ΔfG°gas (Gf_J_mol)", gf)) if v is None]
    if missing:
        raise ValueError(f"Faltan datos de formación de «{name}» (CAS {cas}) como gas ideal a 298.15 K y 1 bar: "
                         f"{', '.join(missing)}. La base de datos de chemicals no los tiene y no se inventan: pide al "
                         f"usuario sus valores y pásalos en data_overrides={{'{name}': {{'Hf_J_mol': ..., 'Gf_J_mol': ...}}}}.")
    cp = _UserCp(ov["Cp_J_molK"]) if ov.get("Cp_J_molK") is not None else _LibCp(cas, name, Ts)
    return {"name": name, "cas": cas, "formula": meta.formula, "atoms": atoms, "Hf": hf, "Gf": gf, "Sf": (hf - gf) / T0,
            "cp": cp, "Hf_source": hf_src, "Gf_source": gf_src}


def _g(sp: dict, T: float) -> float:
    """Gibbs energy of the species at T on the formation basis (element reference terms omitted: they cancel)."""
    if abs(T - T0) < 1e-12:
        return sp["Gf"]
    return sp["Hf"] + sp["cp"].dH(T) - T * (sp["Sf"] + sp["cp"].dS(T))


def _lnk_quadrature(sps, nu, lnK0: float, dH0: float, T: float, n: int = 400) -> float:
    """ln K(T) = ln K(T0) + ∫ ΔH°(T')/(R·T'²) dT', with ΔH°(T') = ΔH°(T0) + ∫ΔCp dT'', both by Simpson's rule on point
    values of Cp (no analytic integral involved)."""
    if abs(T - T0) < 1e-9:
        return lnK0
    t = np.linspace(T0, T, 2 * n + 1)
    dcp = np.array([sum(v * sp["cp"].cp(float(tt)) for v, sp in zip(nu, sps)) for tt in t])
    h = (T - T0) / (2 * n)
    # cumulative ΔH at the even nodes by Simpson over each pair of intervals
    dH_even = dH0 + np.concatenate([[0.0], np.cumsum(h / 3 * (dcp[0:-2:2] + 4 * dcp[1:-1:2] + dcp[2::2]))])
    # midpoints (odd nodes) by the quadratic through the three nodes of the pair
    dH_odd = dH_even[:-1] + h / 12 * (5 * dcp[0:-2:2] + 8 * dcp[1:-1:2] - dcp[2::2])
    dH = np.empty_like(t)
    dH[0::2], dH[1::2] = dH_even, dH_odd
    f = dH / (R_GAS * t ** 2)
    return lnK0 + float(h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()))


def _solve_extents(N, n0, lnK, c, n_inert: float):
    """Extents ξ with Σ_i ν_ij·(ln y_i + c_i) = ln K_j, n = n0 + N·ξ > 0 (c_i = ln(φ_i·P/P°)). Returns ξ, info."""
    S, Rn = N.shape
    scale = max(float(n0.sum() + n_inert), 1e-300)
    # strictly feasible start: maximise t with n0 + N·ξ ≥ t
    lp = linprog(np.r_[np.zeros(Rn), -1.0], A_ub=np.c_[-N, np.ones(S)], b_ub=n0,
                 bounds=[(None, None)] * Rn + [(None, scale)], method="highs")
    if not lp.success or lp.x[-1] <= 1e-13 * scale:
        raise ValueError("Con esa alimentación alguna especie de las reacciones no puede tener moles positivos (falta un "
                         "reactivo o un producto que no se puede formar): el equilibrio queda en el borde y esta "
                         "herramienta no lo resuelve. Revisa feed_mol y las reacciones.")
    xi = lp.x[:Rn].copy()
    s = N.sum(axis=0)

    def state(x):
        n = n0 + N @ x
        if np.any(n <= 0):
            return None
        nt = n.sum() + n_inert
        G = float(-(x @ lnK) + np.sum(n * (np.log(n / nt) + c)) + (n_inert * math.log(n_inert / nt) if n_inert > 0 else 0.0))
        return n, nt, G

    n, nt, G = state(xi)
    it, f = 0, None
    for it in range(1, 400):
        f = N.T @ (np.log(n / nt) + c) - lnK
        if np.max(np.abs(f)) < 1e-13:
            break
        H = N.T @ (N / n[:, None]) - np.outer(s, s) / nt
        try:
            dx = np.linalg.solve(H + 1e-14 * np.trace(H) / Rn * np.eye(Rn), -f)
        except np.linalg.LinAlgError:
            dx = -np.linalg.lstsq(H, f, rcond=None)[0]
        dn = N @ dx
        neg = dn < 0
        a = min(1.0, 0.98 * float(np.min(-n[neg] / dn[neg]))) if neg.any() else 1.0
        ok = False
        for _ in range(60):
            st = state(xi + a * dx)
            if st is not None and st[2] <= G + 1e-4 * a * float(f @ dx) + 1e-15 * abs(G):
                ok = True
                break
            a *= 0.5
        if not ok:
            break
        xi = xi + a * dx
        n, nt, G = st
    f = N.T @ (np.log(n / nt) + c) - lnK
    return xi, {"iterations": it, "max_residual_lnK": float(np.max(np.abs(f))), "residuals": f, "start_min_mol": float(lp.x[-1])}


def _gibbs_lagrange(g_RT, c, B, b, n_inert: float, n_start):
    """Gibbs minimisation by Lagrange multipliers (Smith, Van Ness & Abbott §13.9): n_i = n_tot·exp(−g_i/RT − c_i +
    Σ_k λ_k·B_ki); unknowns λ and ln n_tot; equations B·n = b and Σn + n_inert = n_tot."""
    C = B.shape[0]
    q = -(g_RT + c)
    nt0 = n_start.sum() + n_inert
    lam0 = np.linalg.lstsq(B.T, np.log(n_start / nt0) - q, rcond=None)[0]
    bs = np.maximum(np.abs(B) @ n_start, 1e-300)

    def res(z):
        lam, lnt = z[:C], z[C]
        with np.errstate(over="ignore"):
            n = np.exp(lnt + q + B.T @ lam)
        return np.r_[(B @ n - b) / bs, (n.sum() + n_inert) / math.exp(lnt) - 1.0]

    best = None
    for method in ("hybr", "lm"):
        try:
            sol = root(res, np.r_[lam0, math.log(nt0)], method=method, tol=1e-14)
        except Exception:  # noqa: BLE001 - try the other solver
            continue
        r = float(np.max(np.abs(res(sol.x))))
        if best is None or r < best[1]:
            best = (sol.x, r)
        if r < 1e-10:
            break
    if best is None or not np.isfinite(best[1]):
        return None, None
    z = best[0]
    return np.exp(z[C] + q + B.T @ z[:C]), best[1]


def _cantera(sps, N, reactive, T, P, n0_all, lnK, mechanism: str, full_space: bool, has_phi: bool, y_ours) -> dict:
    try:
        import cantera as ct
    except Exception:  # noqa: BLE001
        return {"compared": False, "reason": "cantera no está instalado"}
    if mechanism not in ("gri30.yaml", "gri30_highT.yaml"):
        return {"compared": False, "reason": f"la correspondencia de especies por CAS solo está definida para GRI-Mech 3.0 "
                                             f"(gri30.yaml), no para {mechanism}"}
    names = [GRI_BY_CAS.get(sp["cas"]) for sp in sps]
    miss = [sp["name"] for sp, nm in zip(sps, names) if nm is None]
    if miss:
        return {"compared": False, "reason": f"especies sin equivalente en {mechanism}: {miss}"}
    gas = ct.Solution(mechanism)
    lo = max(gas.species(nm).thermo.min_temp for nm in names)
    hi = min(gas.species(nm).thermo.max_temp for nm in names)
    if not lo <= T <= hi:
        return {"compared": False, "reason": f"T = {T} K fuera del intervalo de los polinomios NASA de {mechanism} ({lo}-{hi} K)"}
    gas.TP = T, P_STD
    g_RT = gas.standard_gibbs_RT
    idx = [gas.species_index(nm) for nm in names]
    out = {"compared": True, "mechanism": mechanism, "species": dict(zip([sp["name"] for sp in sps], names)),
           "data": "polinomios NASA del mecanismo (fuente de datos distinta de chemicals/thermo)", "reactions": []}
    ok = True
    ir = [i for i, r_ in enumerate(reactive) if r_]
    for j in range(N.shape[1]):
        lnk_ct = -float(sum(N[k, j] * g_RT[idx[i]] for k, i in enumerate(ir)))
        ddg = (lnk_ct - lnK[j]) * R_GAS * T          # = ΔG°(here) − ΔG°(Cantera)
        tol = CANTERA_DG_TOL_J_PER_NU * float(np.abs(N[:, j]).sum())
        agree = abs(ddg) <= tol
        ok &= agree
        out["reactions"].append({"index": j + 1, "lnK_cantera": lnk_ct, "lnK_here": float(lnK[j]),
                                 "K_cantera": math.exp(lnk_ct) if lnk_ct < 700 else None,
                                 "dG_difference_J_mol": float(ddg), "tolerance_J_mol": tol, "agree": bool(agree)})
    if y_ours is not None:
        if not full_space:
            out["composition"] = {"compared": False, "reason": "las reacciones dadas son un subconjunto de las posibles "
                                  "entre estas especies (inertes incluidos); Cantera equilibra todas"}
        elif has_phi:
            out["composition"] = {"compared": False, "reason": "con coeficientes de fugacidad del usuario la fase de "
                                  "Cantera (gas ideal) no es comparable"}
        else:
            sub = ct.Solution(thermo="ideal-gas", species=[gas.species(nm) for nm in names])
            sub.TPX = T, P, {nm: float(v) for nm, v in zip(names, n0_all) if v > 0}
            try:
                sub.equilibrate("TP")
                y_ct = np.array([sub.X[sub.species_index(nm)] for nm in names])
                dy = float(np.max(np.abs(y_ct - y_ours)))
                out["composition"] = {"compared": True, "mole_fractions_cantera": dict(zip([sp["name"] for sp in sps],
                                                                                           map(float, y_ct))),
                                      "max_abs_dy": dy, "tolerance": CANTERA_Y_TOL, "agree": bool(dy <= CANTERA_Y_TOL),
                                      "note": "fase de gas ideal de Cantera restringida a las mismas especies, equilibrada "
                                              "a T y P (minimización de Gibbs con balance de elementos)"}
                ok &= dy <= CANTERA_Y_TOL
            except Exception as e:  # noqa: BLE001
                out["composition"] = {"compared": False, "reason": f"Cantera no convergió: {e}"}
    out["agree"] = bool(ok)
    return out


def _parse_reactions(reactions) -> list[dict]:
    if isinstance(reactions, dict):
        reactions = [reactions]
    if not isinstance(reactions, (list, tuple)) or not reactions:
        raise ValueError("Falta 'reactions': lista de reacciones, cada una {'stoichiometry': {compuesto: coeficiente}} con "
                         "coeficientes negativos para reactivos y positivos para productos.")
    out = []
    for k, r in enumerate(reactions):
        if not isinstance(r, dict) or not r:
            raise ValueError(f"Reacción {k + 1}: se espera un diccionario {{compuesto: coeficiente}}.")
        st = r.get("stoichiometry", r if "stoichiometry" not in r and "name" not in r else None)
        if not isinstance(st, dict) or not st:
            raise ValueError(f"Reacción {k + 1}: falta 'stoichiometry' = {{compuesto: coeficiente}}.")
        try:
            st = {str(a): float(v) for a, v in st.items() if float(v) != 0.0}
        except (TypeError, ValueError):
            raise ValueError(f"Reacción {k + 1}: los coeficientes estequiométricos deben ser números.") from None
        if not any(v < 0 for v in st.values()) or not any(v > 0 for v in st.values()):
            raise ValueError(f"Reacción {k + 1}: necesita al menos un reactivo (coeficiente negativo) y un producto "
                             "(coeficiente positivo).")
        out.append({"name": str(r.get("name") or f"R{k + 1}") if "stoichiometry" in r else f"R{k + 1}", "st": st})
    return out


def _equation(st: dict) -> str:
    def side(items):
        return " + ".join((f"{abs(v):g} " if abs(v) != 1 else "") + a for a, v in items)
    return side([(a, v) for a, v in st.items() if v < 0]) + " = " + side([(a, v) for a, v in st.items() if v > 0])


def reaction_equilibrium(reactions: list[dict], T_K: float | list[float], P_Pa: float | None = None,
                         feed_mol: dict | None = None, data_overrides: dict | None = None,
                         fugacity_coefficients: dict | None = None, cantera_mechanism: str | None = "gri30.yaml") -> dict:
    rx = _parse_reactions(reactions)
    Ts = [T_K] if isinstance(T_K, (int, float)) else list(T_K or [])
    if not Ts or any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in Ts):
        raise ValueError("T_K debe ser una temperatura absoluta en K (> 0) o una lista de ellas.")
    Ts = [float(t) for t in Ts]
    if feed_mol is not None and (not isinstance(feed_mol, dict) or not feed_mol):
        raise ValueError("feed_mol debe ser {compuesto: moles} de la alimentación (los que no reaccionan son inertes).")
    if feed_mol and (P_Pa is None or not isinstance(P_Pa, (int, float)) or not math.isfinite(P_Pa) or P_Pa <= 0):
        raise ValueError("Con feed_mol hace falta P_Pa: presión ABSOLUTA en Pa (> 0) del equilibrio.")
    overrides = dict(data_overrides or {})
    warnings: list[str] = []

    # ---- species: every name resolved to a CAS; the same compound under two names is one species
    order, by_cas, alias = [], {}, {}

    def reg(name: str) -> str:
        if name in alias:
            return alias[name]
        cas = PU.identify(name)[0].CASs
        if cas not in by_cas:
            by_cas[cas] = name
            order.append(name)
        alias[name] = by_cas[cas]
        return alias[name]

    for r in rx:
        st = {}
        for a, v in r["st"].items():
            st[reg(a)] = st.get(reg(a), 0.0) + v
        r["st"] = {a: v for a, v in st.items() if v != 0.0}
    feed = {}
    for a, v in (feed_mol or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"feed_mol['{a}'] debe ser un número de moles.") from None
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"feed_mol['{a}'] debe ser ≥ 0.")
        feed[reg(a)] = feed.get(reg(a), 0.0) + v
    ov_by = {}
    for a, v in overrides.items():
        if not isinstance(v, dict):
            raise ValueError(f"data_overrides['{a}'] debe ser un diccionario con Hf_J_mol, Gf_J_mol y/o Cp_J_molK.")
        cas = PU.identify(a)[0].CASs
        if cas not in by_cas:
            raise ValueError(f"data_overrides: «{a}» no participa en las reacciones ni en la alimentación.")
        ov_by[by_cas[cas]] = v
    phi_by = {}
    for a, v in (fugacity_coefficients or {}).items():
        cas = PU.identify(a)[0].CASs
        if cas not in by_cas or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"fugacity_coefficients['{a}']: compuesto ajeno al problema o coeficiente no positivo.")
        phi_by[by_cas[cas]] = float(v)

    reactive_names = [a for a in order if any(a in r["st"] for r in rx)]
    inert_names = [a for a in order if a not in reactive_names]
    with PU._thermo_quiet():
        sps = [_species(a, ov_by.get(a), Ts) for a in reactive_names]
        inert_meta = [PU.identify(a)[0] for a in inert_names]
        inert_atoms = [{k: float(v) for k, v in (PU._package(m.CASs)[0].atomss[0] or {}).items()} for m in inert_meta]
    S, Rn = len(sps), len(rx)
    N = np.array([[r["st"].get(a, 0.0) for r in rx] for a in reactive_names])
    elements = sorted({e for sp in sps for e in sp["atoms"]})
    A = np.array([[sp["atoms"].get(e, 0.0) for sp in sps] for e in elements])
    imb = A @ N
    for j, r in enumerate(rx):
        if np.max(np.abs(imb[:, j])) > STOICH_TOL * np.max(np.abs(N[:, j])):
            bad = {e: float(v) for e, v in zip(elements, imb[:, j]) if abs(v) > STOICH_TOL}
            raise ValueError(f"La reacción {r['name']} ({_equation(r['st'])}) no está balanceada: sobran/faltan átomos {bad} "
                             "(productos − reactivos). Corrige los coeficientes.")
    if np.linalg.matrix_rank(N, tol=1e-9) < Rn:
        raise ValueError("Las reacciones no son linealmente independientes (una es combinación de las otras): quita la "
                         "redundante; el equilibrio solo necesita un conjunto independiente.")
    rank_A = int(np.linalg.matrix_rank(A, tol=1e-9))
    full_space = Rn == S - rank_A
    # with the inerts counted as possible reactants too (what an element-constrained equilibrium of the whole phase does)
    el_all = sorted(set(elements) | {e for at in inert_atoms for e in at})
    A_all = np.array([[at.get(e, 0.0) for at in [sp["atoms"] for sp in sps] + inert_atoms] for e in el_all])
    full_space_all = Rn == A_all.shape[1] - int(np.linalg.matrix_rank(A_all, tol=1e-9))

    for sp in sps:
        d = sp["cp"].describe(Ts)
        if not d["in_range"]:
            warnings.append(f"Cp de gas ideal de {sp['name']} ({d['method']}) EXTRAPOLADO: su intervalo {d.get('T_range_K')} K "
                            f"no cubre de 298.15 K a las temperaturas pedidas; K(T) es menos fiable.")
        if d["estimated"]:
            warnings.append(f"Cp de gas ideal de {sp['name']} ESTIMADO por contribución de grupos ({d['method']}), no medido.")
        if USER in (sp["Hf_source"], sp["Gf_source"]) and sp["Hf_source"] != sp["Gf_source"]:
            warnings.append(f"{sp['name']}: ΔfH° y ΔfG° vienen de fuentes distintas (usuario y base de datos); comprueba "
                            "que sean coherentes entre sí.")
    cp_ok = all(sp["cp"].describe(Ts)["in_range"] for sp in sps)

    species_out = [{"compound": sp["name"], "CAS": sp["cas"], "formula": sp["formula"], "role": "reactive",
                    "Hf_gas_298_J_mol": {"value": sp["Hf"], "source": sp["Hf_source"]},
                    "Gf_gas_298_J_mol": {"value": sp["Gf"], "source": sp["Gf_source"]},
                    "Cp_ideal_gas": {**sp["cp"].describe(Ts), "value_298_J_molK": sp["cp"].cp(T0)}} for sp in sps] + \
                  [{"compound": a, "CAS": m.CASs, "formula": m.formula, "role": "inert",
                    "note": "inerte: solo diluye (no necesita datos termoquímicos)"} for a, m in zip(inert_names, inert_meta)]

    # ---- reactions at 298.15 K
    dH0 = N.T @ np.array([sp["Hf"] for sp in sps])
    dG0 = N.T @ np.array([sp["Gf"] for sp in sps])
    lnK0 = -dG0 / (R_GAS * T0)
    rx_out = []
    for j, r in enumerate(rx):
        rx_out.append({"name": r["name"], "equation": _equation(r["st"]), "stoichiometry": r["st"],
                       "delta_nu": float(N[:, j].sum()), "dH_298_J_mol": float(dH0[j]), "dG_298_J_mol": float(dG0[j]),
                       "dS_298_J_molK": float((dH0[j] - dG0[j]) / T0), "lnK_298": float(lnK0[j]),
                       "K_298": math.exp(lnK0[j]) if abs(lnK0[j]) < 700 else None,
                       "character": "exotérmica" if dH0[j] < 0 else "endotérmica", "K_of_T": []})

    # ---- per temperature: K(T), equilibrium, evidence
    n0 = np.array([feed.get(a, 0.0) for a in reactive_names])
    n_inert_each = np.array([feed.get(a, 0.0) for a in inert_names])
    n_inert = float(n_inert_each.sum())
    all_names = reactive_names + inert_names
    P = float(P_Pa) if P_Pa is not None else None
    equilibria, reasons = [], []
    worst_quad = 0.0
    cantera_out = []
    for T in Ts:
        g = np.array([_g(sp, T) for sp in sps])
        dG = N.T @ g
        lnK = -dG / (R_GAS * T)
        dH = dH0 + N.T @ np.array([sp["cp"].dH(T) for sp in sps])
        for j in range(Rn):
            lnk_q = _lnk_quadrature(sps, N[:, j], float(lnK0[j]), float(dH0[j]), T)
            lnk_c = float(lnK0[j] - dH0[j] / R_GAS * (1.0 / T - 1.0 / T0))
            worst_quad = max(worst_quad, abs(lnk_q - lnK[j]))
            rx_out[j]["K_of_T"].append({
                "T_K": T, "lnK": float(lnK[j]), "K": math.exp(lnK[j]) if abs(lnK[j]) < 700 else None,
                "log10K": float(lnK[j] / math.log(10.0)), "dG_T_J_mol": float(dG[j]), "dH_T_J_mol": float(dH[j]),
                "lnK_numerical_quadrature": lnk_q,
                "constant_dH_approximation": {"lnK": lnk_c, "K": math.exp(lnk_c) if abs(lnk_c) < 700 else None,
                                              "K_ratio_to_rigorous": math.exp(lnk_c - lnK[j]) if abs(lnk_c - lnK[j]) < 700 else None,
                                              "label": "APROXIMACIÓN ΔH° constante = ΔH°(298.15 K); no es el valor riguroso"}})
        y_all = None
        if feed:
            c = np.array([math.log(phi_by.get(a, 1.0) * P / P_STD) for a in reactive_names])
            xi, info = _solve_extents(N, n0, lnK, c, n_inert)
            n = n0 + N @ xi
            nt = float(n.sum() + n_inert)
            y = n / nt
            n_all = np.r_[n, n_inert_each]
            y_all = n_all / nt
            lnQ = N.T @ (np.log(y) + c)
            k_err = float(np.max(np.abs(lnQ - lnK)))
            el_in, el_out = A @ n0, A @ n
            el_err = float(np.max(np.abs(el_out - el_in) / np.maximum(np.abs(el_in), 1e-300)))
            # independent formulation: Lagrange-multiplier Gibbs minimisation over the species
            if full_space:
                u, sv, vt = np.linalg.svd(A, full_matrices=False)
                B = vt[:rank_A] * 1.0
                basis = "balance de elementos (las reacciones dadas abarcan todas las independientes entre estas especies)"
            else:
                u, sv, vt = np.linalg.svd(N.T, full_matrices=True)
                B = vt[Rn:]
                basis = ("invariantes de las reacciones dadas (subconjunto elegido: las demás reacciones posibles entre "
                         "estas especies se consideran congeladas)")
            lp = linprog(np.r_[np.zeros(Rn), -1.0], A_ub=np.c_[-N, np.ones(S)], b_ub=n0,
                         bounds=[(None, None)] * Rn + [(None, max(nt, 1.0))], method="highs")
            n_g, res_g = _gibbs_lagrange(g / (R_GAS * T), c, B, B @ n0, n_inert, n0 + N @ lp.x[:Rn])
            if n_g is None:
                gibbs = {"converged": False, "constraints": basis}
                reasons.append(f"T = {T} K: la minimización de Gibbs independiente no convergió")
            else:
                y_g = n_g / (n_g.sum() + n_inert)
                dy = float(np.max(np.abs(y_g - y)))
                gibbs = {"converged": bool(res_g < 1e-8), "constraints": basis, "max_constraint_residual_rel": res_g,
                         "mole_fractions": dict(zip(reactive_names, map(float, y_g))), "max_abs_dy_vs_extents": dy,
                         "tolerance": GIBBS_Y_TOL, "agree": bool(dy <= GIBBS_Y_TOL and res_g < 1e-8),
                         "method": "multiplicadores de Lagrange: n_i = n_tot·exp(−g_i/RT − ln(φ_i·P/P°) + Σλ_k·B_ki), "
                                   "scipy.optimize.root; usa g_i(T) de cada especie, no las K"}
                if not gibbs["agree"]:
                    reasons.append(f"T = {T} K: la minimización de Gibbs independiente difiere de la solución por "
                                   f"avances (max |Δy| = {dy:.2e})")
            if k_err > LNK_CONSISTENCY_TOL:
                reasons.append(f"T = {T} K: Π a_i^ν no reproduce K (max |Δ ln K| = {k_err:.2e}); equilibrio prácticamente "
                               "completo o mal condicionado")
            if el_err > ELEMENT_TOL:
                reasons.append(f"T = {T} K: balance de elementos con error relativo {el_err:.2e}")
            conv = {}
            for a, nin, nout in zip(reactive_names, n0, n):
                if nin > 0 and any(r["st"].get(a, 0.0) < 0 for r in rx):
                    conv[a] = float((nin - nout) / nin)
            equilibria.append({
                "T_K": T, "P_Pa": P, "extents_mol": {r["name"]: float(v) for r, v in zip(rx, xi)},
                "moles_out": dict(zip(all_names, map(float, n_all))), "total_moles_out": nt,
                "total_moles_in": float(n0.sum() + n_inert),
                "mole_fractions": dict(zip(all_names, map(float, y_all))),
                "partial_pressures_Pa": dict(zip(all_names, map(float, y_all * P))),
                "conversion": conv,
                "K_consistency": [{"reaction": r["name"], "ln_activity_product": float(q), "lnK": float(k),
                                   "difference": float(q - k)} for r, q, k in zip(rx, lnQ, lnK)],
                "element_balance": {"elements": elements, "in_mol": list(map(float, el_in)), "out_mol": list(map(float, el_out)),
                                    "max_rel_error": el_err},
                "gibbs_minimization_check": gibbs,
                "solver": {"iterations": info["iterations"], "max_residual_lnK": info["max_residual_lnK"]}})
        if cantera_mechanism:
            inert_sps = [{"name": a, "cas": m.CASs} for a, m in zip(inert_names, inert_meta)]
            cmp_ = _cantera(sps + inert_sps, N, [True] * S + [False] * len(inert_sps), T, P or P_STD,
                            np.r_[n0, n_inert_each], lnK, cantera_mechanism, full_space_all, bool(phi_by), y_all) \
                if (not inert_sps or feed) else None
            if cmp_ is not None:
                cantera_out.append({"T_K": T, **cmp_})
                if cmp_.get("compared") and not cmp_["agree"]:
                    warnings.append(f"T = {T} K: Cantera ({cantera_mechanism}) y la base de datos chemicals/thermo difieren "
                                    "más que la tolerancia entre compilaciones (ver cantera_comparison): la discrepancia es "
                                    "de DATOS termoquímicos, decide con el usuario qué fuente manda.")
    if worst_quad > LNK_QUAD_TOL:
        reasons.append(f"ln K(T) analítico y por cuadratura numérica difieren en {worst_quad:.2e}")
    if not cp_ok:
        reasons.append("algún Cp de gas ideal se usa fuera de su intervalo de validez (ver warnings)")
    if phi_by:
        warnings.append("Coeficientes de fugacidad del usuario tomados como CONSTANTES (no se recalculan con la "
                        "composición de equilibrio); las especies sin dato usan φ = 1.")
    if P is not None and P > 1.0e6 and not phi_by:
        warnings.append(f"P = {P / 1e5:.1f} bar con gas ideal (φ = 1): a presión alta la no idealidad cambia la conversión; "
                        "da fugacity_coefficients o trata el resultado como aproximación.")
    if feed and not full_space_all:
        n_all_rx = A_all.shape[1] - int(np.linalg.matrix_rank(A_all, tol=1e-9))
        warnings.append(f"Las {Rn} reacciones dadas no agotan las {n_all_rx} reacciones independientes posibles entre "
                        "las especies presentes (inertes incluidos): el resultado es el equilibrio RESTRINGIDO a las "
                        "reacciones elegidas; un reactor de Gibbs con balance de elementos daría otra composición.")
    return {
        "status": "ok", "phase": "gas ideal" + (" con coeficientes de fugacidad del usuario" if phi_by else ""),
        "standard_state": "gas ideal puro a T y P° = 1 bar (100000 Pa) para TODAS las especies; datos de formación a 298.15 K",
        "species": species_out, "reactions": rx_out,
        "equilibrium": equilibria or None,
        "fugacity_coefficients": {a: {"value": v, "source": USER} for a, v in phi_by.items()} or None,
        "cantera_comparison": cantera_out or None,
        "R_J_molK": R_GAS,
        "method": "K(T) por integración rigurosa de van't Hoff con los Cp de gas ideal (g_i(T) de cada especie); equilibrio "
                  "por avances de reacción (Newton amortiguado sobre G(ξ), convexa) desde un punto interior factible.",
        "warnings": warnings,
        "assumptions": ["Mezcla de gases ideales (a_i = y_i·P/P°)" + ("; φ_i del usuario constantes" if phi_by else ""),
                        "Todas las especies en fase gas: no se comprueba condensación ni se admiten sólidos o líquidos",
                        "Solo ocurren las reacciones dadas; las especies de feed_mol que no aparecen en ellas son inertes"],
        "not_included": ["Coeficientes de fugacidad calculados con una ecuación de estado (se aceptan los del usuario)",
                         "Modo adiabático (temperatura de equilibrio por balance de energía)",
                         "Fases condensadas (carbono sólido, agua líquida) y equilibrio de fases simultáneo",
                         "Cinética: el equilibrio es el límite, no dice si se alcanza"],
        "evidence": {"verified": not reasons, "reasons": reasons,
                     "max_lnK_difference_analytic_vs_quadrature": float(worst_quad),
                     "reactions_span_all_independent": bool(full_space),
                     "reactions_span_all_independent_with_inerts": bool(full_space_all),
                     "checks": ["ln K(T) analítico frente a cuadratura numérica de van't Hoff sobre valores puntuales de Cp"]
                               + (["Π a_i^ν frente a K en la solución", "balance de elementos",
                                   "minimización de Gibbs por multiplicadores de Lagrange (otra formulación) frente a los "
                                   "avances"] if feed else []),
                     "note": "verified certifica el CÁLCULO con los datos indicados en species (y que los Cp se usan dentro "
                             "de su intervalo); la comparación con Cantera mide la diferencia entre FUENTES de datos y se "
                             "informa aparte en cantera_comparison. No certifica que la mezcla real sea gas ideal ni que "
                             "el equilibrio se alcance."},
    }
