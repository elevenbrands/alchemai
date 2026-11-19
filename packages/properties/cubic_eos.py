"""Single-phase molar volume from a two-parameter cubic equation of state (van der Waals, Redlich-Kwong, Soave-Redlich-Kwong,
Peng-Robinson) for a pure compound or a mixture at given T and P, with the critical constants of the user's statement when
they are given (textbook problems fix Tc and Pc) and the `chemicals` database otherwise.

It is NOT a flash: it solves the cubic at the overall composition and reports every real root. Whether a mixture splits
into two phases is a question for `packages.properties.flash`.

All four equations are written in one form, with (u, w) the constants of the attractive term:

    P = R T / (v - b) - a alpha(T) / (v^2 + u b v + w b^2)

    model   u   w    Omega_a                  Omega_b                  alpha(Tr)
    VDW     0   0    27/64                    1/8                      1
    RK      1   0    1/(9 (2^(1/3) - 1))      (2^(1/3) - 1)/3          Tr^-0.5
    SRK     1   0    same as RK               same as RK               [1 + m (1 - Tr^0.5)]^2, m = 0.480 + 1.574 w - 0.176 w^2
    PR      2  -1    0.457235529              0.0777960739             [1 + m (1 - Tr^0.5)]^2, m = 0.37464 + 1.54226 w - 0.26992 w^2

with a = Omega_a R^2 Tc^2 / Pc and b = Omega_b R Tc / Pc. Mixtures use the van der Waals one-fluid rules,
a_mix = sum_i sum_j z_i z_j sqrt(a_i alpha_i a_j alpha_j) (1 - kij), b_mix = sum_i z_i b_i.

In Z = P v / (R T), with A = a alpha P / (R T)^2 and B = b P / (R T):

    Z^3 - (1 + B - u B) Z^2 + (A + w B^2 - u B - u B^2) Z - (A B + w B^2 + w B^3) = 0

References: Redlich & Kwong, Chem. Rev. 44 (1949) 233; Soave, Chem. Eng. Sci. 27 (1972) 1197; Peng & Robinson, Ind. Eng.
Chem. Fundam. 15 (1976) 59; the general (u, w) form is that of Reid, Prausnitz & Poling, *The Properties of Gases and
Liquids*, 4th ed., table 3-5.
"""
from __future__ import annotations

import math

import numpy as np

from packages.properties import pure as PU

R_GAS = PU.R_GAS
_CBRT2M1 = 2.0 ** (1.0 / 3.0) - 1.0
MODELS = {
    "VDW": {"u": 0.0, "w": 0.0, "oa": 27.0 / 64.0, "ob": 1.0 / 8.0, "name": "van der Waals (1873)"},
    "RK": {"u": 1.0, "w": 0.0, "oa": 1.0 / (9.0 * _CBRT2M1), "ob": _CBRT2M1 / 3.0, "name": "Redlich-Kwong (1949)"},
    "SRK": {"u": 1.0, "w": 0.0, "oa": 1.0 / (9.0 * _CBRT2M1), "ob": _CBRT2M1 / 3.0,
            "name": "Soave-Redlich-Kwong (1972)"},
    "PR": {"u": 2.0, "w": -1.0, "oa": 0.45723552892138218, "ob": 0.077796073903888455, "name": "Peng-Robinson (1976)"},
}
_ALIASES = {"RKS": "SRK", "SOAVE": "SRK", "REDLICH-KWONG": "RK", "REDLICH_KWONG": "RK", "REDLICH KWONG": "RK",
            "REDLICH-KWONG-SOAVE": "SRK", "SOAVE-REDLICH-KWONG": "SRK", "PENG-ROBINSON": "PR", "PENG ROBINSON": "PR",
            "VAN DER WAALS": "VDW", "VANDERWAALS": "VDW"}
NEEDS_OMEGA = {"SRK", "PR"}
P_CLOSURE_TOL = 1e-9          # |P(v from the root)/P - 1| accepted for a root of the cubic
SELECT = ("stable", "gas", "liquid")


def _alpha(model: str, Tr: float, omega: float | None) -> float:
    if model == "VDW":
        return 1.0
    if model == "RK":
        return Tr ** -0.5
    if model == "SRK":
        m = 0.480 + 1.574 * omega - 0.176 * omega ** 2
    else:
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    return (1.0 + m * (1.0 - math.sqrt(Tr))) ** 2


def _ln_phi(Z: float, A: float, B: float, u: float, w: float) -> float:
    """ln of the fugacity coefficient of the fluid as a whole (residual Gibbs energy / RT) at a root of the cubic."""
    d = math.sqrt(u * u - 4.0 * w)
    if d == 0.0:
        return Z - 1.0 - math.log(Z - B) - A / Z
    return Z - 1.0 - math.log(Z - B) - A / (B * d) * math.log((2.0 * Z + B * (u + d)) / (2.0 * Z + B * (u - d)))


def _pressure(v: float, T: float, a_alpha: float, b: float, u: float, w: float) -> float:
    return R_GAS * T / (v - b) - a_alpha / (v * v + u * b * v + w * b * b)


def _roots(A: float, B: float, u: float, w: float) -> list[float]:
    c = [1.0, -(1.0 + B - u * B), A + w * B * B - u * B - u * B * B, -(A * B + w * B * B + w * B ** 3)]
    out = []
    for r in np.roots(c):
        if abs(r.imag) > 1e-7 * max(1.0, abs(r.real)):
            continue
        z = float(r.real)
        for _ in range(50):                                  # Newton polish on the cubic
            f = ((c[0] * z + c[1]) * z + c[2]) * z + c[3]
            df = (3.0 * c[0] * z + 2.0 * c[1]) * z + c[2]
            if df == 0.0:
                break
            step = f / df
            z -= step
            if abs(step) < 1e-15 * max(1.0, abs(z)):
                break
        if z > B and not any(abs(z - o) < 1e-10 * max(1.0, abs(o)) for o in out):
            out.append(z)
    return sorted(out)


def _user_constants(critical_constants, names, cas) -> dict[int, dict]:
    got: dict[int, dict] = {}
    for e in critical_constants or []:
        if not isinstance(e, dict) or "compound" not in e:
            raise ValueError("Cada elemento de critical_constants es {'compound': nombre, 'Tc_K': ..., 'Pc_Pa': ..., "
                             "'omega': ...}; falta 'compound'.")
        meta, _ = PU.identify(e["compound"])
        if meta.CASs not in cas:
            raise ValueError(f"critical_constants: «{e['compound']}» no está en components ({', '.join(names)}).")
        i = cas.index(meta.CASs)
        if i in got:
            raise ValueError(f"critical_constants: {names[i]} está repetido.")
        d = {}
        for k in ("Tc_K", "Pc_Pa", "omega"):
            if e.get(k) is None:
                continue
            v = e[k]
            if not isinstance(v, (int, float)) or not math.isfinite(v) or (k != "omega" and v <= 0):
                raise ValueError(f"critical_constants de {names[i]}: {k} debe ser un número finito"
                                 + (" y positivo." if k != "omega" else "."))
            d[k] = float(v)
        unknown = set(e) - {"compound", "Tc_K", "Pc_Pa", "omega"}
        if unknown:
            raise ValueError(f"critical_constants de {names[i]}: claves no reconocidas {sorted(unknown)}; se aceptan Tc_K, "
                             "Pc_Pa (Pa absolutos) y omega.")
        got[i] = d
    return got


def cubic_eos_volume(components: list[str], model: str, T_K: float, P_Pa: float, mole_fractions: list[float] | None = None,
                     critical_constants: list[dict] | None = None, kij: list[dict] | None = None,
                     root: str = "stable") -> dict:
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("Falta 'components': lista con al menos un compuesto (nombre, CAS o fórmula).")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Falta 'model': 'RK' (Redlich-Kwong), 'SRK' (Soave), 'PR' (Peng-Robinson) o 'VDW' (van der Waals). "
                         "No hay valor por defecto: la ecuación de estado la elige el usuario.")
    key = model.strip().upper()
    key = _ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Modelo «{model}» no disponible. Opciones: 'RK', 'SRK', 'PR', 'VDW'.")
    for nm, v in (("T_K", T_K), ("P_Pa", P_Pa)):
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{nm} debe ser un número positivo (T en K, P absoluta en Pa).")
    if root not in SELECT:
        raise ValueError("root debe ser 'stable' (la raíz de menor energía de Gibbs), 'gas' (la mayor) o 'liquid' (la menor).")
    T, P = float(T_K), float(P_Pa)
    n = len(components)
    metas = [PU.identify(c)[0] for c in components]
    cas = [m.CASs for m in metas]
    names = [m.common_name for m in metas]
    if len(set(cas)) != n:
        raise ValueError("components tiene un compuesto repetido.")
    if mole_fractions is None:
        if n > 1:
            raise ValueError("Con más de un compuesto hace falta mole_fractions (composición molar, suma 1).")
        zs = [1.0]
    else:
        if len(mole_fractions) != n or any(not isinstance(z, (int, float)) or not math.isfinite(z) or z < 0
                                           for z in mole_fractions):
            raise ValueError("mole_fractions debe tener un número no negativo por compuesto.")
        s = float(sum(mole_fractions))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"mole_fractions suma {s:.6g}, no 1. Normaliza la composición (la herramienta no lo hace "
                             "por ti para no esconder un error de captura).")
        zs = [float(z) / s for z in mole_fractions]

    from chemicals import acentric, critical
    user = _user_constants(critical_constants, names, cas)
    m = MODELS[key]
    warnings: list[str] = []
    comp_out, a_al, bs = [], [], []
    for i in range(n):
        row = {"compound": names[i], "CAS": cas[i], "mole_fraction": zs[i]}
        vals = {}
        for k, func, mfunc, needed in (("Tc_K", critical.Tc, critical.Tc_methods, True),
                                       ("Pc_Pa", critical.Pc, critical.Pc_methods, True),
                                       ("omega", acentric.omega, acentric.omega_methods, key in NEEDS_OMEGA)):
            if not needed:
                if k in user.get(i, {}):
                    warnings.append(f"{names[i]}: el factor acéntrico dado no se usa; {key} no lo lleva.")
                continue
            if k in user.get(i, {}):
                vals[k] = user[i][k]
                row[k] = {"value": vals[k], "source": "user"}
            else:
                c = PU._constant(cas[i], func, mfunc)
                if c["value"] is None:
                    raise ValueError(f"La base de datos no tiene {k} de {names[i]}; dalo en critical_constants.")
                vals[k] = c["value"]
                row[k] = {"value": c["value"], "source": "library", "method": c["method"], "basis": c["basis"]}
                if c.get("flag"):
                    row[k]["flag"] = c["flag"]
                    warnings.append(f"{names[i]}: {k} de la base de datos no es un dato medido ({c['method']}).")
        Tr = T / vals["Tc_K"]
        al = _alpha(key, Tr, vals.get("omega"))
        a = m["oa"] * R_GAS ** 2 * vals["Tc_K"] ** 2 / vals["Pc_Pa"]
        b = m["ob"] * R_GAS * vals["Tc_K"] / vals["Pc_Pa"]
        row.update(Tr=Tr, Pr=P / vals["Pc_Pa"], a_Pa_m6_mol2=a, alpha=al, b_m3_mol=b)
        comp_out.append(row)
        a_al.append(a * al)
        bs.append(b)

    kmat = [[0.0] * n for _ in range(n)]
    binary = []
    seen = set()
    for e in kij or []:
        if not isinstance(e, dict) or not {"i", "j", "kij"} <= set(e):
            raise ValueError("Cada elemento de kij es {'i': comp, 'j': comp, 'kij': valor}.")
        ia = cas.index(PU.identify(e["i"])[0].CASs) if PU.identify(e["i"])[0].CASs in cas else -1
        ib = cas.index(PU.identify(e["j"])[0].CASs) if PU.identify(e["j"])[0].CASs in cas else -1
        if ia < 0 or ib < 0 or ia == ib:
            raise ValueError(f"kij: el par {e['i']}-{e['j']} no corresponde a dos compuestos distintos de components.")
        if not isinstance(e["kij"], (int, float)) or not math.isfinite(e["kij"]):
            raise ValueError("kij: el valor de cada par debe ser numérico.")
        pair = (min(ia, ib), max(ia, ib))
        if pair in seen:
            raise ValueError(f"kij: el par {names[pair[0]]}-{names[pair[1]]} está repetido.")
        seen.add(pair)
        kmat[ia][ib] = kmat[ib][ia] = float(e["kij"])
    for i in range(n):
        for j in range(i + 1, n):
            src = "user" if (i, j) in seen else "zero_by_absence"
            binary.append({"i": names[i], "j": names[j], "kij": kmat[i][j], "source": src})
    if n > 1 and any(b["source"] == "zero_by_absence" for b in binary):
        warnings.append("Hay pares con kij = 0 por ausencia de dato (regla de mezclado clásica sin corrección). Es lo que "
                        "piden los problemas de libro; en mezclas con agua, CO2 o polares el volumen real puede diferir.")

    a_mix = sum(zs[i] * zs[j] * math.sqrt(a_al[i] * a_al[j]) * (1.0 - kmat[i][j]) for i in range(n) for j in range(n))
    b_mix = sum(zs[i] * bs[i] for i in range(n))
    A = a_mix * P / (R_GAS * T) ** 2
    B = b_mix * P / (R_GAS * T)
    u, w = m["u"], m["w"]
    zroots = _roots(A, B, u, w)
    if not zroots:
        raise ValueError("La ecuación cúbica no tiene raíz física (Z > B) en estas condiciones; revisa T, P y las "
                         "constantes críticas.")
    roots = []
    worst = 0.0
    for z in zroots:
        v = z * R_GAS * T / P
        res = abs(_pressure(v, T, a_mix, b_mix, u, w) / P - 1.0)
        worst = max(worst, res)
        roots.append({"Z": z, "V_m3_mol": v, "V_L_mol": v * 1e3, "V_m3_kmol": v * 1e3,
                      "ln_fugacity_coefficient": _ln_phi(z, A, B, u, w), "pressure_closure_rel": res})
    three = len(roots) == 3
    if three:
        roots[0]["type"], roots[1]["type"], roots[2]["type"] = "liquid-like", "unstable (no physical meaning)", "gas-like"
        cand = [roots[0], roots[2]]
    else:
        for r in roots:
            r["type"] = "single real root"
        cand = roots
    stable = min(cand, key=lambda r: r["ln_fugacity_coefficient"])
    stable["lowest_gibbs"] = True
    chosen = {"stable": stable, "gas": cand[-1], "liquid": cand[0]}[root]
    if three:
        other = cand[0] if stable is cand[-1] else cand[-1]
        warnings.append(
            f"Tres raíces reales: la de menor energía de Gibbs a esta composición es la {stable['type']} "
            f"(Z = {stable['Z']:.5g}); la {other['type']} (Z = {other['Z']:.5g}) es metaestable. "
            + ("Se devuelve la raíz pedida con root, que NO es la estable. " if chosen is not stable else ""))
    if n > 1:
        warnings.append("Volumen de UNA fase a la composición global: esta herramienta no hace equilibrio de fases. Si la "
                        "mezcla puede condensar (p. ej. agua a alta presión), comprueba con independent_flash.")
    else:
        Tc0 = comp_out[0]["Tc_K"]["value"]
        if T < Tc0 and three:
            warnings.append("Compuesto puro por debajo de Tc con tres raíces: la fase estable depende de P frente a la "
                            "presión de saturación de la ecuación; aquí se decide por la energía de Gibbs de cada raíz.")
    v_id = R_GAS * T / P
    reasons = []
    if worst > P_CLOSURE_TOL:
        reasons.append(f"alguna raíz no reproduce P al sustituirla en la ecuación (error relativo {worst:.2e})")
    return {
        "status": "ok", "model": key, "model_name": m["name"], "T_K": T, "P_Pa": P,
        "Z": chosen["Z"], "V_m3_mol": chosen["V_m3_mol"], "V_L_mol": chosen["V_L_mol"], "V_m3_kmol": chosen["V_m3_kmol"],
        "root_returned": root, "root_type": chosen["type"], "root_is_lowest_gibbs": chosen is stable,
        "ideal_gas": {"Z": 1.0, "V_m3_mol": v_id, "V_L_mol": v_id * 1e3, "V_m3_kmol": v_id * 1e3,
                      "deviation_of_eos_rel": chosen["V_m3_mol"] / v_id - 1.0},
        "roots": roots,
        "mixture_parameters": {"a_alpha_Pa_m6_mol2": a_mix, "b_m3_mol": b_mix, "A": A, "B": B,
                               "mixing_rule": "van der Waals one-fluid" if n > 1 else "pure compound"},
        "components": comp_out, "binary_parameters": binary,
        "R_J_molK": R_GAS,
        "method": f"{m['name']}: cúbica en Z resuelta analíticamente por autovalores y pulida con Newton; constantes "
                  "Omega_a, Omega_b y alpha(T) de la publicación original (ver packages/properties/cubic_eos.py).",
        "warnings": warnings,
        "assumptions": [f"El fluido obedece la ecuación {m['name']} con las constantes críticas indicadas en components."]
                       + (["Regla de mezclado de van der Waals de un fluido; kij = 0 en los pares sin dato."] if n > 1 else []),
        "not_included": ["Equilibrio de fases (usa independent_flash)", "Traslación de volumen (Péneloux)",
                         "Constantes críticas efectivas de gases cuánticos (H2, He)"],
        "evidence": {"verified": not reasons, "reasons": reasons, "max_pressure_closure_rel": worst,
                     "n_real_roots": len(roots),
                     "note": "verified certifica que las raíces satisfacen la ecuación elegida con las constantes "
                             "indicadas, NO que esa ecuación represente al fluido."},
    }
