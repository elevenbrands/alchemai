"""Agent / MCP tool of bloque M14: `vle_binary_regression`, binary interaction parameters from the user's VLE data.

Data (T, P, x1, y1), their uncertainties and any reported azeotrope are the USER's: missing ones raise, never default.
Model choices that are not data (NRTL α = 0.3 when not given, ideal gas, default σ, r/q from UNIFAC groups, Wilson
volumes at 298.15 K, Psat from chemicals) are listed in `supuestos` with their source.

M15c: optional calorimetric sets (`excess_enthalpy_sets`: T, x1, h^E) are fitted together with the VLE sets, with
separate weights (`weights`), separate statistics (`estadistica_hE`, per set and per temperature) and a stated verdict
on whether ONE parameter set represents both (`representa_VLE_y_hE`, tolerances JOINT_TOLERANCES or the user's).
Without `excess_enthalpy_sets` the tool returns exactly what it returned before.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from packages.vle import consistency as C
from packages.vle import models as M
from packages.vle import regression as RG
from packages.vle.equilibrium import DataError, Psat, System, VaporPhase

DEFAULT_SIGMA = {"T_K": 0.1, "P_rel": 0.001, "x1": 0.001, "y1": 0.005}
SIGMA_NOTE = ("σ por defecto (supuesto, no del usuario): T 0.1 K, P 0.1 %, x 0.001, y 0.005, valores típicos de un "
              "ebullómetro de recirculación; con las incertidumbres reales de la medición cambian los pesos y los IC")
DEFAULT_SIGMA_HE = {"hE_J_mol": 5.0, "hE_rel": 0.01}
SIGMA_HE_NOTE = ("σ de h^E por defecto (supuesto, no del usuario): 5 J/mol + 1 % de |h^E|, orden típico de un calorímetro "
                 "de flujo isotérmico; da sigma['hE_J_mol'] / sigma['hE_rel'] o sigma_hE_J_mol por punto o por conjunto")
# Stated tolerances of the verdict «one parameter set represents VLE and h^E» (tool choices, MOCKS.md). Δy ≤ 0.01 is
# the criterion of Fredenslund's point test (Fredenslund, Gmehling y Rasmussen 1977); the others are engineering choices.
JOINT_TOLERANCES = {"AAD_T_K": 0.5, "AAD_P_pct": 1.0, "AAD_y": 0.01, "AAD_hE_rel_of_max": 0.10}
JOINT_TOL_NOTE = ("tolerancias del veredicto (elección de la tool, no datos): AAD_T ≤ 0.5 K, AAD_P ≤ 1 %, AAD_y ≤ 0.01 "
                  "(criterio de la prueba de punto de Fredenslund) y, por cada conjunto de h^E, AAD ≤ 10 % del máximo "
                  "|h^E| medido en ese conjunto; da joint_tolerances para usar las tuyas")
HE_OBJECTIVE = ("más el bloque calorimétrico Σ[(h^E_calc − h^E)/σ_hE]² con h^E_calc = −R·T²·∂(g^E/RT)/∂T de los mismos "
                "parámetros (Gibbs-Helmholtz): S = w_vle·S_vle + w_hE·S_hE")
OBJECTIVES = {
    "bubble": ("Mínimos cuadrados de punto de burbuja: isobáricos Σ[(T_calc−T)/σT]² + Σ[(y_calc−y)/σy]² a (P, x) "
               "medidos; isotérmicos o P-T-x-y Σ[(P_calc−P)/(σP·P)]² + Σ[(y_calc−y)/σy]² a (T, x) medidos"),
    "ml": ("Máxima verosimilitud con error en todas las variables (Britt y Luecke 1973): T̂ y x̂ verdaderos por punto, "
           "Σ[(T̂−T)/σT]² + [(x̂−x)/σx]² + [(P̂−P)/σP]² + [(ŷ−y)/σy]²; misma forma que «Maximum-likelihood» de Aspen Plus"),
}
SOURCES = {
    "NRTL": "Renon y Prausnitz, AIChE J. 14 (1968) 135",
    "UNIQUAC": "Abrams y Prausnitz, AIChE J. 21 (1975) 116 (z = 10)",
    "Wilson": "Wilson, J. Am. Chem. Soc. 86 (1964) 127",
    "UNIFAC_Do": "UNIFAC modificado (Dortmund), parámetros DOUFIP2016 de thermo; grupos DDBST",
    "rq": "r y q como suma de Rk y Qk de los subgrupos UNIFAC originales (thermo.unifac.UFSG, volúmenes y áreas de Bondi)",
    "V": "volumen molar líquido de thermo.VolumeLiquid a 298.15 K (método por defecto, nombrado)",
}


class MissingData(ValueError):
    pass


def _points(s: dict, label: str) -> list[dict]:
    pts = []
    for k, q in enumerate(s.get("points") or []):
        P = q.get("P_Pa", None if q.get("P_kPa") is None else 1e3 * float(q["P_kPa"]))
        miss = [n for n, v in (("T_K", q.get("T_K")), ("P_Pa", P), ("x1", q.get("x1"))) if v is None]
        if miss:
            raise MissingData(f"conjunto {label}, punto {k + 1}: faltan datos del usuario (no se suponen): {miss}")
        x1 = float(q["x1"])
        y1 = None if q.get("y1") is None else float(q["y1"])
        if not 0 <= x1 <= 1 or (y1 is not None and not 0 <= y1 <= 1):
            raise MissingData(f"conjunto {label}, punto {k + 1}: fracciones fuera de [0, 1]")
        pts.append({"T_K": float(q["T_K"]), "P_Pa": float(P), "x1": x1, "y1": y1})
    if not pts:
        raise MissingData(f"conjunto {label}: sin puntos")
    return pts


def _kind(pts) -> str:
    T = np.array([q["T_K"] for q in pts])
    P = np.array([q["P_Pa"] for q in pts])
    if (P.max() - P.min()) / P.mean() <= 0.005 and T.max() - T.min() > 0.2:
        return "isobaric"
    if T.max() - T.min() <= 0.2:
        return "isothermal"
    return "PTxy"


def _cas(name: str) -> str:
    from chemicals import CAS_from_any
    try:
        return CAS_from_any(name)
    except Exception as e:  # noqa: BLE001
        raise MissingData(f"compuesto no reconocido por chemicals: {name} ({e})") from e


def _rq(name, cas, user):
    if user and name in user:
        r, q = user[name]
        return float(r), float(q), "del usuario"
    from thermo.unifac import UNIFAC_RQ, UNIFAC_group_assignment_DDBST
    g = UNIFAC_group_assignment_DDBST(cas, "UNIFAC")
    if not g:
        raise MissingData(f"{name}: sin grupos UNIFAC para r y q; da uniquac_rq {{'{name}': [r, q]}}")
    r, q = UNIFAC_RQ(g)
    return float(r), float(q), f"UNIFAC (subgrupos {dict(g)})"


def _vol(name, cas, user):
    if user and name in user:
        return float(user[name]), "del usuario"
    from thermo import VolumeLiquid
    vl = VolumeLiquid(CASRN=cas)
    v = vl.T_dependent_property(298.15)
    if v is None:
        raise MissingData(f"{name}: thermo no da volumen líquido; da wilson_volumes_m3_mol")
    return float(v), f"thermo.VolumeLiquid {vl.method} a 298.15 K"


def _thermo_lng(model, p):
    def f(T, x):
        T, x = np.broadcast_arrays(np.asarray(T, float), np.asarray(x, float))
        out = [M.thermo_ln_gamma(model, t, z, p) for t, z in zip(T.ravel(), x.ravel())]
        a = np.array(out).reshape(T.shape + (2,))
        return a[..., 0], a[..., 1]
    return f


def _unifac_lng(cas):
    from thermo.unifac import DOUFIP2016, DOUFSG, UNIFAC, UNIFAC_group_assignment_DDBST
    groups = [UNIFAC_group_assignment_DDBST(c, "MODIFIED_UNIFAC") for c in cas]
    if not all(groups):
        return None

    def f(T, x):
        T, x = np.broadcast_arrays(np.asarray(T, float), np.asarray(x, float))
        res = []
        for t, z in zip(T.ravel(), x.ravel()):
            g = UNIFAC.from_subgroups(T=float(t), xs=[float(z), 1 - float(z)], chemgroups=groups, version=1,
                                      interaction_data=DOUFIP2016, subgroups=DOUFSG).gammas()
            res.append(np.log(g))
        a = np.array(res).reshape(T.shape + (2,))
        return a[..., 0], a[..., 1]
    return f, groups


def _stats(d: RG.Data, Tc, Pc, yc) -> dict:
    m = d.inner
    ib, it = m & (d.kind == "isobaric"), m & (d.kind != "isobaric")
    hy = m & d.has_y
    out = {"n_puntos": int(m.sum())}
    if ib.any():
        e = Tc[ib] - d.T[ib]
        out.update(AAD_T_K=float(np.mean(np.abs(e))), RMSD_T_K=float(np.sqrt(np.mean(e ** 2))),
                   max_T_K=float(np.max(np.abs(e))), sesgo_T_K=float(np.mean(e)))
    if it.any():
        e = (Pc[it] / d.P[it] - 1) * 100
        out.update(AAD_P_pct=float(np.mean(np.abs(e))), RMSD_P_pct=float(np.sqrt(np.mean(e ** 2))),
                   max_P_pct=float(np.max(np.abs(e))), sesgo_P_pct=float(np.mean(e)))
    if hy.any():
        e = yc[hy] - d.y[hy]
        out.update(AAD_y=float(np.mean(np.abs(e))), RMSD_y=float(np.sqrt(np.mean(e ** 2))),
                   max_y=float(np.max(np.abs(e))), sesgo_y=float(np.mean(e)))
    return out


def _residuals(d, Tc, Pc, yc, labels):
    rows = []
    for k in np.where(d.inner)[0]:
        r = {"conjunto": labels[d.set[k]], "x1": float(d.x[k]), "T_K": float(d.T[k]), "P_Pa": float(d.P[k])}
        if d.kind[k] == "isobaric":
            r.update(T_calc_K=float(Tc[k]), dT_K=float(Tc[k] - d.T[k]))
        else:
            r.update(P_calc_Pa=float(Pc[k]), dP_pct=float((Pc[k] / d.P[k] - 1) * 100))
        if d.has_y[k]:
            r.update(y1=float(d.y[k]), y1_calc=float(yc[k]), dy=float(yc[k] - d.y[k]))
        rows.append(r)
    return rows


def _data_azeotrope(pts, kind):
    q = sorted([p for p in pts if p["y1"] is not None and 0 < p["x1"] < 1], key=lambda p: p["x1"])
    for a, b in zip(q, q[1:]):
        da, db = a["y1"] - a["x1"], b["y1"] - b["x1"]
        if da == 0:
            return {"x1": a["x1"], "T_K": a["T_K"], "P_Pa": a["P_Pa"], "metodo": "punto medido con y = x"}
        if da * db < 0:
            w = da / (da - db)
            return {"x1": a["x1"] + w * (b["x1"] - a["x1"]), "T_K": a["T_K"] + w * (b["T_K"] - a["T_K"]),
                    "P_Pa": a["P_Pa"] + w * (b["P_Pa"] - a["P_Pa"]),
                    "metodo": "interpolación lineal de (y − x) entre los dos puntos medidos que cambian de signo"}
    return None


def _set_azeotropes(system, lng, s):
    T = np.array([q["T_K"] for q in s["points"]])
    P = np.array([q["P_Pa"] for q in s["points"]])
    if s["kind"] == "isobaric":
        return system.azeotropes(lng, P=float(P.mean()))
    if s["kind"] == "isothermal":
        return system.azeotropes(lng, T=float(T.mean()))
    return []


def _refine_az(system, lng, s, x0):
    """Azeotrope with another (slower) γ function, bracketed around x0."""
    T = np.mean([q["T_K"] for q in s["points"]])
    P = np.mean([q["P_Pa"] for q in s["points"]])

    def d(x):
        xa = np.array([x])
        if s["kind"] == "isobaric":
            return float(system.bubble_T(np.array([P]), xa, lng)[1][0] - x)
        return float(system.bubble_P(np.array([T]), xa, lng)[1][0] - x)
    for w in (0.01, 0.03, 0.1):
        lo, hi = max(1e-4, x0 - w), min(1 - 1e-4, x0 + w)
        if d(lo) * d(hi) < 0:
            return brentq(d, lo, hi, xtol=1e-12)
    return None


def _he_sets(excess_enthalpy_sets, sig, supuestos, note_default: bool) -> list[dict]:
    """Calorimetric sets of the user: every point needs T (its own or the set's), x1 and h^E; σ per point from the
    point, the set, or the labelled default. Points with x1 = 0 or 1 (h^E = 0 by definition) are left out."""
    sets, defaulted = [], False
    for i, s in enumerate(excess_enthalpy_sets):
        label = s.get("label") or f"hE_{i + 1}"
        pts = []
        for k, q in enumerate(s.get("points") or []):
            T = q.get("T_K", s.get("T_K"))
            h = q.get("hE_J_mol")
            miss = [n for n, v in (("T_K", T), ("x1", q.get("x1")), ("hE_J_mol", h)) if v is None]
            if miss:
                raise MissingData(f"conjunto de h^E {label}, punto {k + 1}: faltan datos del usuario (no se suponen): {miss}")
            x1 = float(q["x1"])
            if not 0 <= x1 <= 1:
                raise MissingData(f"conjunto de h^E {label}, punto {k + 1}: x1 fuera de [0, 1]")
            if x1 in (0.0, 1.0):
                continue
            sg = q.get("sigma_hE_J_mol", s.get("sigma_hE_J_mol"))
            if sg is None:
                sg = sig["hE_J_mol"] + sig["hE_rel"] * abs(float(h))
                defaulted = True
            if float(sg) <= 0:
                raise MissingData(f"conjunto de h^E {label}, punto {k + 1}: σ de h^E debe ser > 0")
            pts.append({"T_K": float(T), "x1": x1, "hE_J_mol": float(h), "sigma_J_mol": float(sg)})
        if not pts:
            raise MissingData(f"conjunto de h^E {label}: sin puntos con 0 < x1 < 1")
        sets.append({"label": label, "points": pts, "citation": s.get("citation")})
    if defaulted and note_default:
        supuestos.append(SIGMA_HE_NOTE)
    return sets


def _he_stats(model, p, he: RG.HEData, labels) -> tuple[dict, list[dict]]:
    """AAD, RMSD, bias and sign mismatches of h^E, overall and per set × temperature; residual rows per point."""
    calc = M.excess_enthalpy(model, he.T, he.x, p)
    e = calc - he.h

    def block(m):
        return {"n": int(m.sum()), "AAD_hE_J_mol": float(np.mean(np.abs(e[m]))), "RMSD_hE_J_mol": float(np.sqrt(np.mean(e[m] ** 2))),
                "max_hE_J_mol": float(np.max(np.abs(e[m]))), "sesgo_hE_J_mol": float(np.mean(e[m])),
                "max_abs_hE_medido_J_mol": float(np.max(np.abs(he.h[m]))),
                "signo_contrario": int(np.sum(np.sign(calc[m]) != np.sign(he.h[m]))),
                "hE_medido_min_max_J_mol": [float(he.h[m].min()), float(he.h[m].max())],
                "hE_calc_min_max_J_mol": [float(calc[m].min()), float(calc[m].max())]}
    out = block(np.ones(he.n, bool))
    per = []
    for i, lab in enumerate(labels):
        for T in sorted(set(np.round(he.T[he.set == i], 3))):
            m = (he.set == i) & (np.abs(he.T - T) < 2e-3)
            per.append({"conjunto": lab, "T_K": float(T), **block(m)})
    out["por_conjunto_y_T"] = per
    rows = [{"conjunto": labels[he.set[k]], "T_K": float(he.T[k]), "x1": float(he.x[k]), "hE_J_mol": float(he.h[k]),
             "hE_calc_J_mol": float(calc[k]), "dhE_J_mol": float(e[k]), "sigma_J_mol": float(he.sig[k])} for k in range(he.n)]
    return out, rows


def _joint_verdict(st_vle: dict, st_he: dict, tol: dict) -> dict:
    checks = []
    for key in ("AAD_T_K", "AAD_P_pct", "AAD_y"):
        if key in st_vle:
            checks.append({"criterio": key, "valor": st_vle[key], "tolerancia": tol[key], "cumple": bool(st_vle[key] <= tol[key])})
    for b in st_he["por_conjunto_y_T"]:
        lim = tol["AAD_hE_rel_of_max"] * b["max_abs_hE_medido_J_mol"]
        checks.append({"criterio": f"AAD_hE {b['conjunto']} {b['T_K']:.2f} K", "valor": b["AAD_hE_J_mol"], "tolerancia": lim,
                       "cumple": bool(b["AAD_hE_J_mol"] <= lim)})
    return {"representa_VLE_y_hE": all(c["cumple"] for c in checks), "criterios": checks,
            "no_cumple": [c["criterio"] for c in checks if not c["cumple"]]}


def _published(model, spec, fixed):
    p = dict(fixed)
    if "d12_J_mol" in spec:
        p["b12"], p["b21"] = M.b_from_energy(model, float(spec["d12_J_mol"]), float(spec["d21_J_mol"]))
        p["a12"] = p["a21"] = 0.0
    else:
        for k in ("a12", "a21", "b12", "b21"):
            p[k] = float(spec.get(k, 0.0))
    if model == "NRTL" and spec.get("alpha") is not None:
        p["alpha"] = float(spec["alpha"])
    return p


def vle_binary_regression(components: list[str], data_sets: list[dict], models: list[str] | None = None,
                          nrtl_alpha: float | str | None = None, temperature_dependence: str = "constante",
                          objective: str = "bubble", sigma: dict | None = None, psat: dict | None = None,
                          vapor_phase: str = "ideal", uniquac_rq: dict | None = None,
                          wilson_volumes_m3_mol: dict | None = None, reported_azeotropes: dict | None = None,
                          compare_unifac: bool = True, published_parameters: dict | None = None,
                          legendre_order: int = 4, excess_enthalpy_sets: list[dict] | None = None,
                          weights: dict | None = None, he_weight_scan: list[float] | None = None,
                          joint_tolerances: dict | None = None) -> dict:
    try:
        return _run(components, data_sets, models, nrtl_alpha, temperature_dependence, objective, sigma, psat,
                    vapor_phase, uniquac_rq, wilson_volumes_m3_mol, reported_azeotropes, compare_unifac,
                    published_parameters, legendre_order, excess_enthalpy_sets, weights, he_weight_scan,
                    joint_tolerances)
    except (MissingData, DataError, ValueError) as e:
        return {"ok": False, "error": str(e)}


def _run(components, data_sets, models, nrtl_alpha, temperature_dependence, objective, sigma, psat, vapor_phase,
         uniquac_rq, wilson_volumes, reported_az, compare_unifac, published, legendre_order,
         excess_enthalpy_sets=None, weights=None, he_weight_scan=None, joint_tolerances=None) -> dict:
    if not components or len(components) != 2:
        raise MissingData("components: se esperan exactamente dos compuestos [1, 2]")
    if not data_sets:
        raise MissingData("data_sets: faltan los datos experimentales del usuario (no se inventan)")
    models = list(models or M.MODELS)
    bad = [m for m in models if m not in M.MODELS]
    if bad:
        raise MissingData(f"modelos desconocidos {bad}; disponibles {list(M.MODELS)}")
    if objective not in OBJECTIVES:
        raise MissingData("objective: 'bubble' o 'ml'")
    if temperature_dependence not in ("constante", "a+b/T"):
        raise MissingData("temperature_dependence: 'constante' o 'a+b/T'")
    supuestos, avisos = [], []
    n1, n2 = components
    cas = (_cas(n1), _cas(n2))
    psat = psat or {}
    ps = (Psat(n1, cas[0], psat.get(n1)), Psat(n2, cas[1], psat.get(n2)))
    for p in ps:
        if p.kind == "thermo":
            supuestos.append(f"Psat de {p.name}: {p.source} (no la del usuario; da 'psat' para usar la tuya)")
    vap = VaporPhase("ideal" if vapor_phase == "ideal" else vapor_phase, cas)
    if vapor_phase == "ideal":
        supuestos.append("fase vapor ideal (Raoult modificado, Φ = 1): válido a presiones bajas a moderadas")
    system = System(ps, vap)

    sets, labels = [], []
    for i, s in enumerate(data_sets):
        label = s.get("label") or f"conjunto_{i + 1}"
        pts = _points(s, label)
        sets.append({"label": label, "points": pts, "kind": _kind(pts), "citation": s.get("citation")})
        labels.append(label)
        for p_ in ps:
            if p_.T_range:
                Ts = [q["T_K"] for q in pts]
                if min(Ts) < p_.T_range[0] or max(Ts) > p_.T_range[1]:
                    avisos.append(f"{label}: T fuera del rango de la Psat de {p_.name} ({p_.T_range})")
    d = RG.Data(sets)
    sig = dict(DEFAULT_SIGMA)
    if sigma:
        for k in ("T_K", "x1", "y1", "P_rel"):
            if sigma.get(k) is not None:
                sig[k] = float(sigma[k])
        if sigma.get("P_Pa") is not None:
            sig["P_rel"] = float(sigma["P_Pa"]) / float(np.mean(d.P))
            supuestos.append(f"σP absoluto {sigma['P_Pa']} Pa convertido a relativo con la P media ({sig['P_rel']:.2e})")
    if not sigma or any(sigma.get(k) is None for k in ("T_K", "y1")):
        supuestos.append(SIGMA_NOTE)

    he = he_labels = he_sets = tol = None
    w_vle = w_he = 1.0
    if not excess_enthalpy_sets and (weights or he_weight_scan or joint_tolerances):
        raise MissingData("weights, he_weight_scan y joint_tolerances solo aplican con excess_enthalpy_sets")
    if excess_enthalpy_sets:
        sig_he = dict(DEFAULT_SIGMA_HE)
        user_keys = [k for k in sig_he if sigma and sigma.get(k) is not None]
        for k in user_keys:
            sig_he[k] = float(sigma[k])
        he_sets = _he_sets(excess_enthalpy_sets, sig_he, supuestos, note_default=len(user_keys) < len(sig_he))
        he_labels = [s_["label"] for s_ in he_sets]
        he = RG.HEData(he_sets)
        sig = {**sig, **sig_he}
        if weights:
            unknown = [k for k in weights if k not in ("vle", "hE")]
            if unknown:
                raise MissingData(f"weights: claves desconocidas {unknown}; se esperan 'vle' y 'hE'")
            w_vle, w_he = float(weights.get("vle", 1.0)), float(weights.get("hE", 1.0))
            if w_vle <= 0 or w_he < 0:
                raise MissingData("weights: vle > 0 y hE ≥ 0 (hE = 0 deja h^E fuera del ajuste y solo lo predice)")
        else:
            supuestos.append("pesos de los bloques VLE y h^E = 1 y 1 (mínimos cuadrados ponderados por σ, sin peso extra); "
                             "da weights={'vle': .., 'hE': ..} para cambiarlos")
        tol = dict(JOINT_TOLERANCES)
        if joint_tolerances:
            unknown = [k for k in joint_tolerances if k not in tol]
            if unknown:
                raise MissingData(f"joint_tolerances: claves desconocidas {unknown}; disponibles {list(tol)}")
            tol.update({k: float(v) for k, v in joint_tolerances.items()})
        else:
            supuestos.append(JOINT_TOL_NOTE)
        for s_ in he_sets:
            for p_ in ps:
                Ts = [q["T_K"] for q in s_["points"]]
                if p_.T_range and (min(Ts) < p_.T_range[0] or max(Ts) > p_.T_range[1]):
                    avisos.append(f"{s_['label']}: T fuera del rango de la Psat de {p_.name} ({p_.T_range})")
        if temperature_dependence == "constante":
            avisos.append("h^E con dependencia 'constante' (τ = b/T): los mismos dos parámetros deben dar g^E y su "
                          "derivada en T; usa temperature_dependence='a+b/T' para separar nivel (a) y pendiente (b)")

    fixed = {}
    fit_alpha = isinstance(nrtl_alpha, str) and nrtl_alpha.lower().startswith("ajust")
    if "NRTL" in models and not fit_alpha:
        if nrtl_alpha is None:
            fixed_alpha = 0.3
            supuestos.append("NRTL α = 0.3 (supuesto: valor recomendado por Renon y Prausnitz 1968 para muchas "
                             "mezclas; da nrtl_alpha o 'ajustar')")
        else:
            fixed_alpha = float(nrtl_alpha)
    rq, vols = {}, {}
    if "UNIQUAC" in models:
        for n, c in zip(components, cas):
            r, q, src = _rq(n, c, uniquac_rq)
            rq[n] = (r, q)
            if src != "del usuario":
                supuestos.append(f"UNIQUAC r, q de {n} = {r:.4f}, {q:.4f}: {src}; {SOURCES['rq']}")
    if "Wilson" in models:
        for n, c in zip(components, cas):
            v, src = _vol(n, c, wilson_volumes)
            vols[n] = v
            if src != "del usuario":
                supuestos.append(f"Wilson V de {n} = {v * 1e6:.2f} cm³/mol: {src}")

    n_inner = int(d.inner.sum())
    out_models = {}
    for model in models:
        fx = {}
        names = ["b12", "b21"]
        if temperature_dependence == "a+b/T":
            names = ["a12", "a21", "b12", "b21"]
        if model == "NRTL":
            if fit_alpha:
                names.append("alpha")
            else:
                fx["alpha"] = fixed_alpha
        if model == "UNIQUAC":
            fx["r"] = (rq[n1][0], rq[n2][0])
            fx["q"] = (rq[n1][1], rq[n2][1])
        if model == "Wilson":
            fx["V"] = (vols[n1], vols[n2])
        if n_inner < len(names) + 1:
            raise MissingData(f"{model}: {n_inner} puntos con 0 < x1 < 1 no bastan para {len(names)} parámetros")
        r = RG.fit(system, model, d, fx, names, objective, sig, he, w_vle, w_he)
        p = r["p"]
        Tc, Pc, yc = RG.predict(system, model, p, d)
        lng = lambda T, x, p=p, model=model: M.ln_gamma(model, T, x, p)
        units = {"a12": "-", "a21": "-", "b12": "K", "b21": "K", "alpha": "-"}
        params = [{"nombre": n, "valor": float(v), "ic95": float(c), "error_estandar": float(e), "unidad": units[n]}
                  for n, v, c, e in zip(names, r["theta"], r["ci95"], r["se"])]
        energ = M.energy_parameters(model, p)
        if temperature_dependence == "a+b/T":
            energ["nota"] = "Δ de energía = ±R·b (la parte a_ij es adimensional y aparte)"
        else:
            for k, nm in ((0, "12"), (1, "21")):
                key = [kk for kk in energ if kk.endswith(f"{nm}_J_mol")][0]
                params.append({"nombre": key.replace("_J_mol", ""), "valor": energ[key],
                               "ic95": float(r["ci95"][k] * M.R), "unidad": "J/mol"})
        # independent re-evaluation with thermo's own GibbsExcess classes
        tl = _thermo_lng(model, p)
        m = d.inner
        a, b = lng(d.T[m], d.x[m])
        ta, tb = tl(d.T[m], d.x[m])
        Tt, Pt, yt = d.T.copy(), d.P.copy(), np.full_like(d.T, np.nan)
        ib, it = m & (d.kind == "isobaric"), m & (d.kind != "isobaric")
        if ib.any():
            Tt[ib], yt[ib] = system.bubble_T(d.P[ib], d.x[ib], tl, T0=d.T[ib])
        if it.any():
            Pt[it], yt[it] = system.bubble_P(d.T[it], d.x[it], tl)
        az = {}
        for s in sets:
            pred = _set_azeotropes(system, lng, s)
            for a_ in pred:
                xt = _refine_az(system, tl, s, a_["x1"])
                a_["x1_con_thermo"] = xt
            az[s["label"]] = pred
        corr = [[float(v) for v in row] for row in r["corr"]]
        if any(abs(corr[i][j]) > 0.98 for i in range(len(names)) for j in range(len(names)) if i != j):
            avisos.append(f"{model}: parámetros muy correlacionados (|ρ| > 0.98); los IC son amplios y los valores "
                          "individuales poco identificables con estos datos")
        out_models[model] = {
            "fuente": SOURCES[model], "parametros": params, "energias": energ,
            "fijos": {k: (list(v) if isinstance(v, tuple) else v) for k, v in fx.items()},
            "correlacion": {"nombres": names, "matriz": corr}, "grados_libertad": r["dof"],
            "varianza_residual_s2": r["s2"], "funcion_objetivo": r["cost"], "numero_condicion_correlacion": r["cond_corr"],
            "convergencia": {"exito": r["success"], "mensaje": r["message"], "evaluaciones": r["nfev"]},
            "estadistica": _stats(d, Tc, Pc, yc), "residuos": _residuals(d, Tc, Pc, yc, labels),
            "azeotropo_predicho": az,
            "verificacion_thermo": {
                "max_dif_ln_gamma": float(max(np.max(np.abs(a - ta)), np.max(np.abs(b - tb)))),
                "estadistica_con_thermo": _stats(d, Tt, Pt, yt),
                "clase": {"NRTL": "thermo.nrtl.NRTL", "UNIQUAC": "thermo.uniquac.UNIQUAC",
                          "Wilson": "thermo.wilson.Wilson"}[model]},
        }
        if he is not None:
            st_he, rows_he = _he_stats(model, p, he, he_labels)
            om = out_models[model]
            om["estadistica_hE"] = st_he
            om["residuos_hE"] = rows_he
            om["arranques_conjuntos"] = r.get("starts")
            om["verificacion_thermo"]["max_dif_hE_J_mol"] = float(max(
                abs(M.thermo_excess_enthalpy(model, T_, x_, p) - c_) for T_, x_, c_ in
                zip(he.T, he.x, M.excess_enthalpy(model, he.T, he.x, p))))
            om["veredicto_conjunto"] = _joint_verdict(om["estadistica"], st_he, tol)
            if not om["veredicto_conjunto"]["representa_VLE_y_hE"]:
                avisos.append(f"{model}: un solo juego de parámetros NO representa VLE y h^E dentro de las tolerancias "
                              f"(falla: {', '.join(om['veredicto_conjunto']['no_cumple'])})")
            if he_weight_scan:
                scan = []
                for w in he_weight_scan:
                    w = float(w)
                    if w < 0:
                        raise MissingData("he_weight_scan: pesos ≥ 0")
                    rs = RG.fit(system, model, d, fx, names, objective, sig, he, w_vle, w)
                    Ts_, Ps_, ys_ = RG.predict(system, model, rs["p"], d)
                    sv, sh = _stats(d, Ts_, Ps_, ys_), _he_stats(model, rs["p"], he, he_labels)[0]
                    scan.append({"peso_hE": w, "peso_vle": w_vle,
                                 "parametros": {n: float(v) for n, v in zip(names, rs["theta"])},
                                 **{k: v for k, v in sv.items() if k.startswith("AAD")},
                                 "AAD_hE_J_mol": sh["AAD_hE_J_mol"],
                                 "AAD_hE_por_conjunto_y_T": [{"conjunto": b["conjunto"], "T_K": b["T_K"],
                                                              "AAD_hE_J_mol": b["AAD_hE_J_mol"]} for b in sh["por_conjunto_y_T"]],
                                 "representa_VLE_y_hE": _joint_verdict(sv, sh, tol)["representa_VLE_y_hE"],
                                 "convergencia": rs["success"]})
                om["compromiso_VLE_hE"] = {"nota": "mismo ajuste repetido con otros pesos del bloque h^E (peso 0 = solo "
                                                   "VLE, h^E predicho); los parámetros reportados arriba son los del "
                                                   "peso pedido en weights", "barrido": scan}
        if r["latent"] is not None:
            out_models[model]["valores_verdaderos_estimados"] = [
                {"x1_medido": float(x0), "x1_estimado": float(x1), "T_medida_K": float(t0), "T_estimada_K": float(t1)}
                for x0, x1, t0, t1 in zip(d.x[m], r["latent"]["x1"], d.T[m], r["latent"]["T_K"])]
        if published and model in published:
            pp = _published(model, published[model], fx)
            Tq, Pq, yq = RG.predict(system, model, pp, d)
            inside = {}
            for n, v, c in zip(names, r["theta"], r["ci95"]):
                if n in pp:
                    inside[n] = {"publicado": pp[n], "ajustado": float(v), "ic95": float(c),
                                 "dentro_ic95": bool(abs(pp[n] - v) <= c)}
            out_models[model]["parametros_publicados"] = {"fuente": published[model].get("source"),
                                                          "comparacion": inside,
                                                          "estadistica_con_publicados": _stats(d, Tq, Pq, yq)}

    key = "AAD_T_K" if (d.kind[d.inner] == "isobaric").any() else "AAD_P_pct"
    ranked = sorted(out_models, key=lambda k: (out_models[k]["estadistica"].get(key, 0),
                                               out_models[k]["estadistica"].get("AAD_y", 0)))
    best = ranked[0]
    pbest = dict(out_models[best]["fijos"])
    for prm in out_models[best]["parametros"]:
        if prm["nombre"] in ("a12", "a21", "b12", "b21", "alpha"):
            pbest[prm["nombre"]] = prm["valor"]
    for k in ("r", "q", "V"):
        if k in pbest:
            pbest[k] = tuple(pbest[k])
    lbest = lambda T, x: M.ln_gamma(best, T, x, pbest)

    unifac = None
    if compare_unifac:
        u = _unifac_lng(cas)
        if u is None:
            avisos.append("UNIFAC Dortmund: sin asignación de grupos DDBST para algún compuesto; sin comparación")
        else:
            ul, groups = u
            m = d.inner
            Tu, Pu, yu = d.T.copy(), d.P.copy(), np.full_like(d.T, np.nan)
            ib, it = m & (d.kind == "isobaric"), m & (d.kind != "isobaric")
            if ib.any():
                Tu[ib], yu[ib] = system.bubble_T(d.P[ib], d.x[ib], ul, T0=d.T[ib])
            if it.any():
                Pu[it], yu[it] = system.bubble_P(d.T[it], d.x[it], ul)
            uaz = {}
            for s in sets:
                pred = []
                for a_ in _set_azeotropes(system, lbest, s) or [{"x1": 0.5}]:
                    xa = _refine_az(system, ul, s, a_["x1"])
                    if xa is not None:
                        pred.append(xa)
                uaz[s["label"]] = pred
            unifac = {"fuente": SOURCES["UNIFAC_Do"], "grupos": [dict(g) for g in groups],
                      "estadistica": _stats(d, Tu, Pu, yu), "azeotropo_x1_cerca_del_ajustado": uaz,
                      "nota": "predicción sin ajuste; búsqueda del azeótropo UNIFAC solo cerca del del modelo ajustado"}

    cons, azr = [], []
    for s in sets:
        pts = [q for q in s["points"] if 0 < q["x1"] < 1 and q["y1"] is not None]
        T, P, x, y = (np.array([q[c] for q in pts]) for c in ("T_K", "P_Pa", "x1", "y1"))
        entry = {"conjunto": s["label"], "tipo": s["kind"], "n": len(pts)}
        if s["kind"] == "PTxy":
            entry["aviso"] = "conjunto ni isotérmico ni isobárico: las pruebas de área y de punto no aplican"
        elif len(pts) >= 3:
            iso = s["kind"] == "isobaric"
            entry["area_redlich_kister"] = C.area_test(system, T, P, x, y, iso)
            entry["punto_fredenslund"] = C.point_test(system, T, P, x, y, iso, legendre_order)
            entry["van_ness_directa"] = C.van_ness_direct(system, T, P, x, y, legendre_order)
            entry["desviacion_modelo"] = {"modelo": best, **C.model_deviation(system, lbest, T, P, x, y)}
            verdicts = [entry["area_redlich_kister"].get("consistente"), entry["punto_fredenslund"].get("consistente")]
            entry["veredicto"] = ("consistente" if all(v is True for v in verdicts) else
                                  "inconsistente" if all(v is False for v in verdicts) else "dudoso (las pruebas discrepan)")
        pure = [q for q in s["points"] if q["x1"] in (0.0, 1.0)]
        if pure:
            entry["componente_puro"] = C.pure_component_check(system, pure)
        cons.append(entry)
        rep = (reported_az or {}).get(s["label"])
        azr.append({"conjunto": s["label"], "de_los_datos": _data_azeotrope(s["points"], s["kind"]),
                    "reportado_por_el_usuario": rep,
                    "predicho_por_modelo": {mm: out_models[mm]["azeotropo_predicho"][s["label"]] for mm in out_models}})

    return {"ok": True, "herramienta": "vle_binary_regression", "componentes": {"1": n1, "2": n2, "CAS": list(cas)},
            "conjuntos": [{"label": s["label"], "tipo": s["kind"], "n": len(s["points"]), "cita": s["citation"]}
                          for s in sets],
            "psat": [p.source for p in ps], "fase_vapor": vap.source,
            **({} if he is None else {"conjuntos_hE": [
                {"label": s_["label"], "n": len(s_["points"]), "cita": s_["citation"],
                 "T_K": sorted({q["T_K"] for q in s_["points"]})} for s_ in he_sets]}),
            "objetivo": {"tipo": objective, "descripcion": OBJECTIVES[objective] + ("" if he is None else "; " + HE_OBJECTIVE),
                         "sigmas": sig, "dependencia_T": temperature_dependence,
                         **({} if he is None else {"pesos": {"vle": w_vle, "hE": w_he}, "tolerancias_veredicto": tol})},
            "modelos": out_models, "mejor_modelo": {"modelo": best, "criterio": f"menor {key}, luego AAD_y"},
            "consistencia": cons, "azeotropo": azr, "unifac_dortmund": unifac,
            "supuestos": supuestos, "avisos": avisos,
            "convenciones": M.__doc__.strip().splitlines()[3:8]}
