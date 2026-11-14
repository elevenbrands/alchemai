"""NRTL liquid + ideal-gas vapour for the equation-oriented column (bloque M15 of docs/MAPA_MOTORES.md).

M8b only solved ideal VLE (Raoult). Here the IDAES generic package of `ideal_properties.build_configuration` keeps
every pure-component property (Wagner-McGarry Psat, Poling cp, DIPPR 105/106, fitted liquid cp; sources cited there)
and only the liquid phase changes:

  f_i^L = x_i · γ_i(T, x) · Psat_i(T)          modified Raoult (γ-φ with φ = 1, no Poynting factor)
  h^L   = Σ x_i h_i^L(T) + h^E(T, x)          h^E = −R T² ∂(g^E/RT)/∂T at constant x (analytic, same parameters)

  NRTL (Renon & Prausnitz, AIChE J. 14 (1968) 135), matrix convention of `thermo.nrtl.NRTL` (tau[i][j], 0-based):
    tau_ij = a_ij + b_ij/T,  G_ij = exp(−alpha_ij tau_ij),  tau_ii = 0
    ln γ_i = Σ_j x_j tau_ji G_ji / Σ_k x_k G_ki
             + Σ_j [x_j G_ij / Σ_k x_k G_kj] (tau_ij − Σ_m x_m tau_mj G_mj / Σ_k x_k G_kj)

Bubble and dew points use IDAES's general LogBubbleDew (not the Raoult-only IdealBubbleDew).

Binary parameters: exactly ONE labelled source per pair, in this order (never a default value):
  (a) the user's own values (with their citation),
  (b) the result dict of `vle_binary_regression` (NRTL model), passed as is,
  (c) the ChemSep NRTL databank shipped offline by `thermo` (thermo.interaction_parameters.IPDB "ChemSep NRTL";
      the same table DWSIM loads for its NRTL package).
A pair with none of the three is refused ("corre vle_binary_regression con datos del sistema").
"""
from __future__ import annotations

import itertools
import math

import numpy as np

R = 8.314462618
CAL = 4.184
CHEMSEP_TABLE = "ChemSep NRTL"
SOURCE_KINDS = ("usuario", "vle_binary_regression", "banco ChemSep NRTL (thermo)")
GAMMA_CHECK_TOL = 1e-8            # Pyomo ln γ vs thermo.nrtl.NRTL (same parameters → must agree to round-off)
HE_CHECK_TOL = 1e-4               # analytic h^E vs a central finite difference of g^E/RT (relative)
UNIFAC_LN_GINF_WARN = 0.5         # |ln γ∞(NRTL) − ln γ∞(UNIFAC)| above which the cross-check warns
SOURCES = {
    "model": "NRTL, Renon y Prausnitz, AIChE J. 14 (1968) 135; misma convención de matrices que thermo.nrtl.NRTL",
    "excess_enthalpy": "h^E = −R·T²·∂(g^E/RT)/∂T analítica con los mismos parámetros (Gibbs-Helmholtz)",
    "chemsep": "ChemSep NRTL (Kooijman y Taylor, banco de parámetros binarios de ChemSep), incluido sin conexión en "
               "thermo.interaction_parameters.IPDB['ChemSep NRTL']; tau_ij = b_ij/T (b en K, convertido de cal/mol)",
    "bubble_dew": "IDAES LogBubbleDew (forma general con fugacidades), no IdealBubbleDew; equilibrio en forma "
                  "log_fugacity",
    "unifac": "UNIFAC modificado Dortmund (thermo.unifac, DDBST) solo como comparación cruzada de γ∞ y azeótropos",
}


class MissingParameters(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ parameter sources
def _cas(name: str) -> str:
    from chemicals import CAS_from_any
    return CAS_from_any(name)


def _num(d: dict, key: str, where: str) -> float:
    v = d.get(key)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}: falta {key} numérico")
    return float(v)


def _user_pair(item: dict, names: list[str], cas: dict) -> tuple[tuple[str, str], dict]:
    comps = item.get("components")
    if not isinstance(comps, (list, tuple)) or len(comps) != 2:
        raise ValueError(f"binary_parameters: cada par del usuario lleva components=[i, j] ({item})")
    by_cas = {v: k for k, v in cas.items()}
    try:
        i, j = (by_cas[_cas(c)] for c in comps)
    except Exception as ex:  # noqa: BLE001
        raise ValueError(f"binary_parameters: {comps} no son compuestos de la columna {names}") from ex
    where = f"par {i}/{j} del usuario"
    src = item.get("source")
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"{where}: falta source (la cita de donde salen los parámetros)")
    if "b_ij_K" in item or "b_ji_K" in item:
        bij, bji, unit = _num(item, "b_ij_K", where), _num(item, "b_ji_K", where), "b en K"
    elif "dg_ij_J_mol" in item or "dg_ji_J_mol" in item:
        bij, bji = _num(item, "dg_ij_J_mol", where) / R, _num(item, "dg_ji_J_mol", where) / R
        unit = "Δg en J/mol (b = Δg/R)"
    elif "dg_ij_cal_mol" in item or "dg_ji_cal_mol" in item:
        bij, bji = _num(item, "dg_ij_cal_mol", where) * CAL / R, _num(item, "dg_ji_cal_mol", where) * CAL / R
        unit = "Δg en cal/mol (b = Δg·4.184/R)"
    else:
        raise ValueError(f"{where}: da b_ij_K y b_ji_K (tau = a + b/T) o dg_ij_J_mol/dg_ji_J_mol o dg_ij_cal_mol/dg_ji_cal_mol")
    aij = _num(item, "a_ij", where) if "a_ij" in item else None
    aji = _num(item, "a_ji", where) if "a_ji" in item else None
    if (aij is None) != (aji is None):
        raise ValueError(f"{where}: a_ij y a_ji van juntos (o ninguno: forma tau = b/T)")
    alpha = _num(item, "alpha", where)
    if not 0 < alpha < 1:
        raise ValueError(f"{where}: alpha fuera de (0, 1)")
    p = {"a_ij": aij or 0.0, "a_ji": aji or 0.0, "b_ij_K": bij, "b_ji_K": bji, "alpha": alpha,
         "source_kind": SOURCE_KINDS[0], "source": src.strip(),
         "form": unit + ("; tau = a + b/T" if aij is not None else "; forma tau = b/T (sin término a, tal como se dio)")}
    if item.get("T_range_K"):
        p["T_range_K"] = [float(t) for t in item["T_range_K"]]
    return (i, j), p


def _regression_pair(res: dict, names: list[str], cas: dict) -> tuple[tuple[str, str], dict]:
    if res.get("ok") is not True:
        raise ValueError(f"el resultado de vle_binary_regression no es válido: {res.get('error', 'ok != true')}")
    nrtl = (res.get("modelos") or {}).get("NRTL")
    if not nrtl:
        raise ValueError("el resultado de vle_binary_regression no trae el modelo NRTL (corre la regresión con 'NRTL')")
    rc = res["componentes"]["CAS"]
    by_cas = {v: k for k, v in cas.items()}
    if rc[0] not in by_cas or rc[1] not in by_cas:
        raise ValueError(f"la regresión es de {res['componentes']['1']}/{res['componentes']['2']}, no de un par de {names}")
    i, j = by_cas[rc[0]], by_cas[rc[1]]
    vals = dict(nrtl.get("fijos") or {})
    for prm in nrtl["parametros"]:
        vals[prm["nombre"]] = prm["valor"]
    for k in ("b12", "b21", "alpha"):
        if k not in vals:
            raise ValueError(f"la regresión NRTL no trae {k}")
    ci = {prm["nombre"]: prm.get("ic95") for prm in nrtl["parametros"]}
    Ts = [r["T_K"] for r in nrtl.get("residuos", []) if "T_K" in r]
    p = {"a_ij": float(vals.get("a12", 0.0)), "a_ji": float(vals.get("a21", 0.0)), "b_ij_K": float(vals["b12"]),
         "b_ji_K": float(vals["b21"]), "alpha": float(vals["alpha"]), "source_kind": SOURCE_KINDS[1],
         "source": "vle_binary_regression con los datos: " + "; ".join(c.get("cita", c.get("label", "")) for c in res.get("conjuntos", [])),
         "form": "tau = a + b/T (convención de packages/vle/models.py = thermo)", "ci95": ci,
         "regression_psat": res.get("psat"), "regression_vapor": res.get("fase_vapor"),
         "regression_statistics": nrtl.get("estadistica")}
    if Ts:
        p["T_range_K"] = [float(min(Ts)), float(max(Ts))]
    return (i, j), p


def _chemsep_pair(i: str, j: str, cas: dict) -> dict | None:
    from thermo.interaction_parameters import IPDB
    pair = [cas[i], cas[j]]
    if not IPDB.has_ip_specific(CHEMSEP_TABLE, pair, "bij"):
        return None
    b = IPDB.get_ip_asymmetric_matrix(CHEMSEP_TABLE, pair, "bij")
    al = IPDB.get_ip_asymmetric_matrix(CHEMSEP_TABLE, pair, "alphaij")
    return {"a_ij": 0.0, "a_ji": 0.0, "b_ij_K": float(b[0][1]), "b_ji_K": float(b[1][0]), "alpha": float(al[0][1]),
            "source_kind": SOURCE_KINDS[2], "source": SOURCES["chemsep"],
            "form": "tau = b/T (forma del banco ChemSep)",
            "dg_cal_mol_as_in_chemsep": [float(b[0][1]) * R / CAL, float(b[1][0]) * R / CAL]}


def resolve_parameters(names: list[str], binary_parameters=None) -> dict:
    """Matrices tau_a, tau_b (K) and alpha in the order of `names`, plus one labelled record per pair."""
    names = list(names)
    cas = {n: _cas(n) for n in names}
    given: dict[frozenset, tuple[tuple[str, str], dict]] = {}
    items = binary_parameters if isinstance(binary_parameters, list) else ([binary_parameters] if binary_parameters else [])
    for kind in (0, 1):          # (a) user before (b) regression, whatever the list order
        for it in items:
            if not isinstance(it, dict):
                raise ValueError("binary_parameters: lista de pares del usuario y/o resultados de vle_binary_regression")
            is_reg = it.get("herramienta") == "vle_binary_regression"
            if is_reg != bool(kind):
                continue
            (i, j), p = _regression_pair(it, names, cas) if is_reg else _user_pair(it, names, cas)
            key = frozenset((i, j))
            if key in given:
                if given[key][1]["source_kind"] == p["source_kind"]:
                    raise ValueError(f"binary_parameters: el par {i}/{j} viene dos veces de la misma fuente")
                continue        # a higher-preference source already gave this pair
            given[key] = ((i, j), p)
    n = len(names)
    ta, tb, al = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    pairs, missing = [], []
    for a, b in itertools.combinations(names, 2):
        key = frozenset((a, b))
        if key in given:
            (i, j), p = given[key]
        else:
            i, j = a, b
            p = _chemsep_pair(a, b, cas)
            if p is None:
                missing.append(f"{a}/{b}")
                continue
        ii, jj = names.index(i), names.index(j)
        ta[ii, jj], ta[jj, ii] = p["a_ij"], p["a_ji"]
        tb[ii, jj], tb[jj, ii] = p["b_ij_K"], p["b_ji_K"]
        al[ii, jj] = al[jj, ii] = p["alpha"]
        pairs.append({"i": i, "j": j, **p})
    if missing:
        raise MissingParameters(
            "faltan parámetros NRTL para " + ", ".join(missing) + ": no están en binary_parameters ni en el banco ChemSep "
            "NRTL de thermo, y no se inventan. Corre vle_binary_regression con datos VLE publicados del par y pasa su "
            "resultado en binary_parameters (o da tus propios parámetros con su cita).")
    return {"names": names, "CAS": cas, "tau_a": ta.tolist(), "tau_b_K": tb.tolist(), "alpha": al.tolist(), "pairs": pairs}


# ------------------------------------------------------------------------------------------------ numeric NRTL (thermo)
def thermo_nrtl(params: dict, T: float, xs) -> "object":
    from thermo.nrtl import NRTL
    return NRTL(T=float(T), xs=[float(x) for x in xs], tau_as=params["tau_a"], tau_bs=params["tau_b_K"],
                alpha_cs=params["alpha"])


def gammas(params: dict, T: float, xs) -> list[float]:
    return [float(g) for g in thermo_nrtl(params, T, xs).gammas()]


def _ge_RT(params: dict, T: float, xs) -> float:
    ta, tb, al = (np.asarray(params[k]) for k in ("tau_a", "tau_b_K", "alpha"))
    x = np.asarray(xs, float)
    tau = ta + tb / T
    G = np.exp(-al * tau)
    return float(sum(x[i] * (x @ (tau[:, i] * G[:, i])) / (x @ G[:, i]) for i in range(len(x))))


def excess_enthalpy_numeric(params: dict, T: float, xs) -> float:
    h = 1e-3 * T
    return -R * T ** 2 * (_ge_RT(params, T + h, xs) - _ge_RT(params, T - h, xs)) / (2 * h)


# ------------------------------------------------------------------------------------------------ Pyomo / IDAES
def _ln_gamma_expr(P: dict, names, x: dict, T, i: str):
    from pyomo.environ import exp, units as pyunits
    idx = {n: k for k, n in enumerate(names)}
    ta, tb, al = P["tau_a"], P["tau_b_K"], P["alpha"]

    def tau(k, l):
        return 0.0 if k == l else ta[idx[k]][idx[l]] + tb[idx[k]][idx[l]] * pyunits.K / T

    def G(k, l):
        return 1.0 if k == l else exp(-al[idx[k]][idx[l]] * tau(k, l))

    D = {l: sum(x[k] * G(k, l) for k in names) for l in names}
    t1 = sum(x[j] * tau(j, i) * G(j, i) for j in names) / D[i]
    t2 = sum(x[j] * G(i, j) / D[j] * (tau(i, j) - sum(x[m] * tau(m, j) * G(m, j) for m in names) / D[j]) for j in names)
    return t1 + t2


def _excess_enthalpy_expr(P: dict, names, x: dict, T, Rgas):
    from pyomo.environ import exp, units as pyunits
    idx = {n: k for k, n in enumerate(names)}
    ta, tb, al = P["tau_a"], P["tau_b_K"], P["alpha"]
    total = 0
    for i in names:
        S, Sd, D, Dd = 0, 0, 0, 0
        for j in names:
            if j == i:
                D += x[j]
                continue
            a, b, c = ta[idx[j]][idx[i]], tb[idx[j]][idx[i]], al[idx[j]][idx[i]]
            t = a + b * pyunits.K / T
            dt = -b * pyunits.K / T ** 2
            g = exp(-c * t)
            S += x[j] * t * g
            Sd += x[j] * dt * g * (1 - c * t)
            D += x[j] * g
            Dd += x[j] * g * (-c * dt)
        total += x[i] * (Sd * D - S * Dd) / D ** 2
    return -Rgas * T ** 2 * total


def make_liquid_eos(params: dict):
    """IDAES EoS class (subclass of Ideal): NRTL γ in the liquid fugacity and h^E in the liquid enthalpy; the vapour
    phase keeps the plain Ideal class."""
    from pyomo.environ import exp
    from idaes.models.properties.modular_properties.eos.ideal import Ideal
    from idaes.models.properties.modular_properties.base.utility import get_method, get_component_object as cobj
    names = list(params["names"])

    def _liq(b, p):
        return b.params.get_phase(p).is_liquid_phase()

    def lng(b, x, T, j):
        return _ln_gamma_expr(params, names, x, T, j)

    def fug(b, p, j, x, T):
        return x[j] * exp(lng(b, x, T, j)) * get_method(b, "pressure_sat_comp", j)(b, cobj(b, j), T)

    def logfug(b, p, j, x, log_xj, T):
        return log_xj + lng(b, x, T, j) + get_method(b, "pressure_sat_comp", comp=j, log_expression=True)(b, cobj(b, j), T)

    class NRTLLiquid(Ideal):
        """NRTL liquid (modified Raoult) — parameters baked in from resolve_parameters()."""
        nrtl_parameters = params
        electrolyte_support = False

        @staticmethod
        def act_coeff_phase_comp(b, p, j):
            if not _liq(b, p):
                return 1
            x = {k: b.mole_frac_phase_comp[p, k] for k in names}
            return exp(lng(b, x, b.temperature, j))

        @staticmethod
        def act_phase_comp(b, p, j):
            return b.mole_frac_phase_comp[p, j] * NRTLLiquid.act_coeff_phase_comp(b, p, j)

        @staticmethod
        def enth_mol_phase(b, p):
            base = Ideal.enth_mol_phase(b, p)
            if not _liq(b, p):
                return base
            x = {k: b.mole_frac_phase_comp[p, k] for k in names}
            return base + _excess_enthalpy_expr(params, names, x, b.temperature, Ideal.gas_constant(b))

        @staticmethod
        def fug_phase_comp(b, p, j):
            if not _liq(b, p):
                return Ideal.fug_phase_comp(b, p, j)
            return fug(b, p, j, {k: b.mole_frac_phase_comp[p, k] for k in names}, b.temperature)

        @staticmethod
        def fug_phase_comp_eq(b, p, j, pp):
            if not _liq(b, p):
                return Ideal.fug_phase_comp_eq(b, p, j, pp)
            return fug(b, p, j, {k: b.mole_frac_phase_comp[p, k] for k in names}, b._teq[pp])

        @staticmethod
        def log_fug_phase_comp(b, p, j):
            if not _liq(b, p):
                return Ideal.log_fug_phase_comp(b, p, j)
            x = {k: b.mole_frac_phase_comp[p, k] for k in names}
            return logfug(b, p, j, x, b.log_mole_frac_phase_comp[p, j], b.temperature)

        @staticmethod
        def log_fug_phase_comp_eq(b, p, j, pp):
            if not _liq(b, p):
                return Ideal.log_fug_phase_comp_eq(b, p, j, pp)
            x = {k: b.mole_frac_phase_comp[p, k] for k in names}
            return logfug(b, p, j, x, b.log_mole_frac_phase_comp[p, j], b._teq[pp])

        @staticmethod
        def log_fug_phase_comp_Tbub(b, p, j, pp):
            if not _liq(b, p):
                return Ideal.log_fug_phase_comp_Tbub(b, p, j, pp)
            x = {k: b.mole_frac_comp[k] for k in names}
            return logfug(b, p, j, x, b.log_mole_frac_comp[j], b.temperature_bubble[pp])

        @staticmethod
        def log_fug_phase_comp_Tdew(b, p, j, pp):
            if not _liq(b, p):
                return Ideal.log_fug_phase_comp_Tdew(b, p, j, pp)
            x = {k: b._mole_frac_tdew[pp[0], pp[1], k] for k in names}
            return logfug(b, p, j, x, b.log_mole_frac_tdew[pp[0], pp[1], j], b.temperature_dew[pp])

        @staticmethod
        def log_fug_phase_comp_Pbub(b, p, j, pp):
            if not _liq(b, p):
                return Ideal.log_fug_phase_comp_Pbub(b, p, j, pp)
            x = {k: b.mole_frac_comp[k] for k in names}
            return logfug(b, p, j, x, b.log_mole_frac_comp[j], b.temperature)

        @staticmethod
        def log_fug_phase_comp_Pdew(b, p, j, pp):
            if not _liq(b, p):
                return Ideal.log_fug_phase_comp_Pdew(b, p, j, pp)
            x = {k: b._mole_frac_pdew[pp[0], pp[1], k] for k in names}
            return logfug(b, p, j, x, b.log_mole_frac_pdew[pp[0], pp[1], j], b.temperature)

        @staticmethod
        def entr_mol_phase(b, p):
            raise NotImplementedError("entropía no implementada en la fase NRTL (la columna no la usa)")

        @staticmethod
        def gibbs_mol_phase(b, p):
            raise NotImplementedError("energía de Gibbs no implementada en la fase NRTL (la columna no la usa)")

    return NRTLLiquid


def build_configuration(names: list[str], P_Pa: float, params: dict, extra_T: list[float] = ()) -> tuple[dict, dict]:
    """M8b configuration with the liquid phase replaced by NRTL and LogBubbleDew."""
    from packages.eo import ideal_properties as IP
    from idaes.models.properties.modular_properties.phase_equil.bubble_dew import LogBubbleDew
    from idaes.models.properties.modular_properties.phase_equil.forms import log_fugacity
    cfg, prov = IP.build_configuration(names, P_Pa, extra_T)
    cfg["phases"]["Liq"]["equation_of_state"] = make_liquid_eos(params)
    cfg["bubble_dew_method"] = LogBubbleDew
    # ln x + ln γ + ln Psat = ln y + ln P: same equations as the fugacity form, better conditioned for dilute and
    # aqueous columns (ethanol-water 10 % feed: 3/18 starts converge vs 0/18 with x·γ·Psat = y·P, M15)
    for c in cfg["components"].values():
        c["phase_equilibrium_form"] = {("Vap", "Liq"): log_fugacity}
    prov["liquid_model"] = {"model": SOURCES["model"], "excess_enthalpy": SOURCES["excess_enthalpy"],
                            "bubble_dew": SOURCES["bubble_dew"], "pairs": params["pairs"]}
    return cfg, prov


def nrtl_check(cfg: dict, params: dict, temperatures: list[float]) -> dict:
    """γ and h^E computed BY THE IDAES PACKAGE vs thermo.nrtl.NRTL (independent code, same parameters) and vs a finite
    difference of g^E/RT, on a grid of compositions × temperatures."""
    from pyomo.environ import ConcreteModel, value, units as pyunits
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    names = params["names"]
    eos = cfg["phases"]["Liq"]["equation_of_state"]
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.p = GenericParameterBlock(**cfg)
    m.fs.sb = m.fs.p.build_state_block([0], defined_state=True)
    s = m.fs.sb[0]
    s.flow_mol.fix(1)
    s.pressure.fix(101325)
    n = len(names)
    grid = [np.eye(n)[k] * 0.98 + 0.02 / n for k in range(n)] + [np.full(n, 1.0 / n)]
    worst_g, worst_h, rows = 0.0, 0.0, []
    for T in temperatures:
        s.temperature.fix(T)
        for xs in grid:
            xd = {c: float(v) for c, v in zip(names, xs)}
            lg = [value(_ln_gamma_expr(params, names, xd, s.temperature, c)) for c in names]
            ref = [math.log(g) for g in gammas(params, T, xs)]
            he = value(pyunits.convert(_excess_enthalpy_expr(params, names, xd, s.temperature, eos.gas_constant(s)),
                                       pyunits.J / pyunits.mol))
            he_fd = excess_enthalpy_numeric(params, T, xs)
            worst_g = max(worst_g, max(abs(a - b) for a, b in zip(lg, ref)))
            worst_h = max(worst_h, abs(he - he_fd) / max(abs(he_fd), 1.0))
            rows.append({"T_K": T, "x": xd, "ln_gamma_idaes": lg, "ln_gamma_thermo": ref, "hE_J_mol": he,
                         "hE_finite_difference_J_mol": he_fd})
    return {"rows": rows, "max_abs_ln_gamma_diff": worst_g, "max_rel_hE_diff": worst_h,
            "tolerances": {"ln_gamma": GAMMA_CHECK_TOL, "hE": HE_CHECK_TOL},
            "verified": worst_g <= GAMMA_CHECK_TOL and worst_h <= HE_CHECK_TOL}


# ------------------------------------------------------------------------------------------------ screening
def _psat(names):
    from thermo import VaporPressure
    out = {}
    for c in names:
        vp = VaporPressure(CASRN=_cas(c))
        vp.method = "WAGNER_MCGARRY"          # the same correlation the IDAES package uses (RPP4 ← Wagner-McGarry)
        out[c] = vp
    return out


def _bubble_binary(params, names, ps, P, x1):
    from scipy.optimize import brentq
    a, b = names

    def f(T):
        g = gammas(params, T, [x1, 1 - x1])
        return x1 * g[0] * ps[a](T) + (1 - x1) * g[1] * ps[b](T) - P
    Ta, Tb = ps[a].solve_property(P), ps[b].solve_property(P)
    T = brentq(f, min(Ta, Tb) - 60, max(Ta, Tb) + 60, xtol=1e-10)
    g = gammas(params, T, [x1, 1 - x1])
    return T, g[0] * ps[a](T) / (g[1] * ps[b](T)), g


def binary_azeotropes(params: dict, P_Pa: float, n_grid: int = 201) -> list[dict]:
    """Azeotropes of every binary pair at P with the chosen NRTL parameters (α_ij − 1 changes sign along x_i)."""
    from scipy.optimize import brentq
    names = params["names"]
    ps = _psat(names)
    out = []
    for a, b in itertools.combinations(names, 2):
        sub = sub_parameters(params, [a, b])
        xs = np.linspace(1e-4, 1 - 1e-4, n_grid)
        al = np.array([_bubble_binary(sub, [a, b], ps, P_Pa, x)[1] for x in xs])
        s = np.sign(al - 1.0)
        for k in np.where(s[:-1] * s[1:] < 0)[0]:
            xa = brentq(lambda x: _bubble_binary(sub, [a, b], ps, P_Pa, x)[1] - 1.0, xs[k], xs[k + 1], xtol=1e-10)
            T, _, g = _bubble_binary(sub, [a, b], ps, P_Pa, xa)
            Ta, Tb = ps[a].solve_property(P_Pa), ps[b].solve_property(P_Pa)
            kind = "mínimo punto de ebullición" if T < min(Ta, Tb) else ("máximo punto de ebullición" if T > max(Ta, Tb)
                                                                           else "intermedio")
            out.append({"pair": [a, b], "x_az": {a: float(xa), b: float(1 - xa)}, "T_az_K": float(T), "kind": kind,
                        "gammas": g})
    return out


def sub_parameters(params: dict, sub: list[str]) -> dict:
    idx = [params["names"].index(c) for c in sub]
    pick = lambda M: [[M[i][j] for j in idx] for i in idx]  # noqa: E731
    return {"names": list(sub), "tau_a": pick(params["tau_a"]), "tau_b_K": pick(params["tau_b_K"]),
            "alpha": pick(params["alpha"]), "pairs": [p for p in params["pairs"] if {p["i"], p["j"]} <= set(sub)]}


def unifac_cross_check(params: dict, P_Pa: float) -> dict:
    """γ∞ with NRTL (chosen parameters) vs Dortmund UNIFAC at the boiling point of the solvent; azeotrope presence
    with each model. Reported, never used to refuse."""
    from thermo.unifac import UNIFAC, UNIFAC_group_assignment_DDBST, DOUFSG, DOUFIP2016
    from packages.eo.ideal_properties import ideality_check
    names = params["names"]
    groups = {c: UNIFAC_group_assignment_DDBST(_cas(c), "MODIFIED_UNIFAC") for c in names}
    ps = _psat(names)
    rows, warn = [], []
    unif = ideality_check(names, P_Pa) if all(groups.values()) else None
    nrtl_az = {frozenset(a["pair"]) for a in binary_azeotropes(params, P_Pa, n_grid=81)}
    for a, b in itertools.combinations(names, 2):
        sub = sub_parameters(params, [a, b])
        r = {"pair": [a, b]}
        for solute, solvent, xs in ((a, b, [0.0, 1.0]), (b, a, [1.0, 0.0])):
            T = ps[solvent].solve_property(P_Pa)
            g_n = gammas(sub, T, xs)[0 if solute == a else 1]
            g_u = None
            if groups[a] and groups[b]:
                g_u = UNIFAC.from_subgroups(T, xs, [groups[a], groups[b]], version=1, interaction_data=DOUFIP2016,
                                            subgroups=DOUFSG).gammas()[0 if solute == a else 1]
            r[f"gamma_inf_{solute}_en_{solvent}"] = {"T_K": T, "NRTL": g_n, "UNIFAC_Do": g_u}
            if g_u is not None and abs(math.log(g_n) - math.log(g_u)) > UNIFAC_LN_GINF_WARN:
                warn.append(f"{solute} en {solvent}: γ∞ NRTL {g_n:.2f} vs UNIFAC {g_u:.2f} (difieren más de "
                            f"e^{UNIFAC_LN_GINF_WARN}); revisa la fuente de los parámetros")
        u_az = None
        if unif is not None:
            u_az = next((pp["azeotrope_predicted"] for pp in unif["pairs"] if set(pp["pair"]) == {a, b}), None)
        r["azeotrope"] = {"NRTL": frozenset((a, b)) in nrtl_az, "UNIFAC_Do": u_az}
        if u_az is not None and u_az != r["azeotrope"]["NRTL"]:
            warn.append(f"{a}/{b}: UNIFAC {'sí' if u_az else 'no'} predice azeótropo y NRTL {'sí' if r['azeotrope']['NRTL'] else 'no'}")
        rows.append(r)
    return {"pairs": rows, "warnings": warn, "method": SOURCES["unifac"]}


def azeotrope_refusals(params: dict, P_Pa: float, feed_z: dict, specs: list[dict]) -> tuple[list[dict], list[str]]:
    """Spanish reasons to refuse specs that a column cannot reach because they lie across a predicted azeotrope.
    Binary: the products stay in the feed's region [0, x_az] or [x_az, 1]. ≥3 components with any binary azeotrope:
    refused (distillation boundaries are not evaluated)."""
    az = binary_azeotropes(params, P_Pa)
    names = params["names"]
    reasons = []
    if az and len(names) > 2:
        reasons.append("hay azeótropos binarios predichos (" + "; ".join(
            f"{a['pair'][0]}/{a['pair'][1]} a x={a['x_az'][a['pair'][0]]:.3f}" for a in az) +
            ") en una mezcla de 3 o más compuestos: esta tool no evalúa fronteras de destilación multicomponente")
        return az, reasons
    if not az:
        return az, reasons
    c1 = names[0]
    cuts = sorted(a["x_az"][c1] for a in az)
    z1 = feed_z[c1]
    edges = [0.0] + cuts + [1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo < z1 < hi:
            break
    else:
        reasons.append(f"la alimentación ({c1} = {z1:.4f}) está en el azeótropo predicho; no se puede separar por destilación simple")
        return az, reasons
    for sp in specs:
        if sp["kind"] != "mole_fraction":
            continue
        need_lo = sp["min"] if sp["component"] == c1 else None
        need_hi = 1 - sp["min"] if sp["component"] != c1 else None
        if need_lo is not None and need_lo > hi - 1e-9:
            reasons.append(f"spec {sp['component']} ≥ {sp['min']} en {sp['stream']}: cruza el azeótropo predicho "
                           f"({c1} = {hi:.4f} a {P_Pa:.0f} Pa con estos parámetros NRTL); la alimentación ({c1} = {z1:.4f}) "
                           f"queda del otro lado")
        if need_hi is not None and need_hi < lo + 1e-9:
            reasons.append(f"spec {sp['component']} ≥ {sp['min']} en {sp['stream']}: cruza el azeótropo predicho "
                           f"({c1} = {lo:.4f} a {P_Pa:.0f} Pa con estos parámetros NRTL); la alimentación ({c1} = {z1:.4f}) "
                           f"queda del otro lado")
    return az, reasons


def parameter_range_warnings(params: dict, T_lo: float, T_hi: float) -> list[str]:
    out = []
    for p in params["pairs"]:
        rng = p.get("T_range_K")
        if rng and (T_lo < rng[0] - 1 or T_hi > rng[1] + 1):
            out.append(f"{p['i']}/{p['j']}: los parámetros ({p['source_kind']}) vienen de datos en {rng[0]:.1f}-{rng[1]:.1f} K "
                       f"y la columna opera en {T_lo:.1f}-{T_hi:.1f} K: extrapolación")
        if p["source_kind"] == SOURCE_KINDS[1] and p.get("regression_psat"):
            out.append(f"{p['i']}/{p['j']}: la regresión usó Psat {p['regression_psat']}; la columna usa Wagner-McGarry. "
                       "Si difieren, el equilibrio de la columna no reproduce exactamente el ajuste")
    return out


def thermo_bubble_points(params: dict, P_Pa: float, compositions: dict) -> dict:
    """Independent bubble temperature (thermo FlashVL: GibbsExcessLiquid with thermo.nrtl.NRTL + IdealGas, Wagner-McGarry
    Psat, no Poynting, no φsat) of each named liquid composition at P."""
    from thermo import ChemicalConstantsPackage, PropertyCorrelationsPackage, FlashVL, GibbsExcessLiquid, IdealGas
    from thermo import HeatCapacityGas
    names = params["names"]
    cas = [_cas(c) for c in names]
    consts = ChemicalConstantsPackage.constants_from_IDs(cas)
    ps = _psat(names)
    HG = [HeatCapacityGas(CASRN=c) for c in cas]
    corr = PropertyCorrelationsPackage(consts, VaporPressures=[ps[c] for c in names], HeatCapacityGases=HG,
                                       skip_missing=True)
    T0, x0 = 350.0, [1.0 / len(names)] * len(names)
    ge = thermo_nrtl(params, T0, x0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        liq = GibbsExcessLiquid(VaporPressures=[ps[c] for c in names], HeatCapacityGases=HG, GibbsExcessModel=ge,
                                use_Poynting=False, use_phis_sat=False, T=T0, P=P_Pa, zs=x0)
    gas = IdealGas(HeatCapacityGases=HG, T=T0, P=P_Pa, zs=x0)
    flasher = FlashVL(consts, corr, liquid=liq, gas=gas)
    out = {}
    for k, z in compositions.items():
        zs = [max(float(z[c]), 1e-12) for c in names]
        s = sum(zs)
        zs = [v / s for v in zs]
        try:
            res = flasher.flash(P=P_Pa, VF=0.0, zs=zs)
            out[k] = {"T_bubble_K": float(res.T), "y_bubble": dict(zip(names, map(float, res.gas.zs))),
                      "method": "FlashVL PVF (VF = 0)"}
        except Exception:  # noqa: BLE001  (thermo's PVF flash fails near a pure component)
            from scipy.optimize import brentq

            def resid(T):
                ph = liq.to(T=T, P=P_Pa, zs=zs)
                return sum(z * g * ps[c](T) for z, g, c in zip(zs, ph.gammas(), names)) - P_Pa
            ts = [ps[c].solve_property(P_Pa) for c in names]
            T = brentq(resid, min(ts) - 60, max(ts) + 60, xtol=1e-10)
            g = liq.to(T=T, P=P_Pa, zs=zs).gammas()
            y = [z * gi * ps[c](T) / P_Pa for z, gi, c in zip(zs, g, names)]
            out[k] = {"T_bubble_K": float(T), "y_bubble": dict(zip(names, map(float, y))),
                      "method": "Σ x·γ·Psat = P (brentq) con γ de GibbsExcessLiquid/thermo.nrtl; el flash PVF de thermo no converge aquí"}
    return {"points": out, "engine": "thermo FlashVL (GibbsExcessLiquid + thermo.nrtl.NRTL, IdealGas; Psat Wagner-McGarry)"}
