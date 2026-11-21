"""Multicomponent shortcut distillation (Fenske-Underwood-Gilliland-Kirkbride) with K-values from an own flash.

Why an own shortcut: DWSIM's ShortcutColumn evaluates the relative volatilities at ONE condition (the feed), which on
close-boiling splits inflates N_min by ~10 % and with it R, N and the duties (Finlayson 6.1/6.3/6.4). Here the
volatilities are evaluated where Fenske needs them — at the top (dew point of the overhead vapour) and at the bottom
(bubble point of the bottoms) at column pressure — with `packages.properties.flash` (thermo), and their geometric mean
is used. The non-key split comes from the Fenske equation, iterated with the end temperatures until the product
compositions stop changing.

  Fenske      N_min = ln[(d_LK/b_LK)(b_HK/d_HK)] / ln alpha_LK,HK ;  d_i/b_i = (d_HK/b_HK) alpha_i^N_min
  Underwood   sum alpha_i z_i/(alpha_i − theta) = 1 − q, alpha_HK < theta < alpha_LK ;
              R_min = sum alpha_i x_D,i/(alpha_i − theta) − 1          (class-1 form: one theta between the keys)
  Gilliland   Molokanov et al. (1972): Y = 1 − exp[(1 + 54.4X)/(11 + 117.2X) · (X − 1)/sqrt(X)],
              X = (R − R_min)/(R + 1), Y = (N − N_min)/(N + 1)
  Kirkbride   N_R/N_S = [(z_HK/z_LK)(x_B,LK/x_D,HK)^2 (B/D)]^0.206

N counts equilibrium stages including the partial reboiler (and the partial condenser when there is one).
Duties: Q_C = D(R + 1)(H_V,top − H_D) and Q_R from the overall enthalpy balance, with H from the same flash model.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

from packages.properties import flash as FL


def _flash(comps, zs, model, P, kij, act, **spec):
    zs = [max(float(v), 1e-12) for v in zs]             # a zero fraction would drop the component from the K table
    tot = sum(zs)
    return FL.independent_flash(comps, [v / tot for v in zs], model=model, P_Pa=P, kij=kij, activity_parameters=act, **spec)


def _K(r, comps):
    k = r["K_values"]
    k = k.get("V/L", k) if isinstance(k, dict) else k
    return [float(k[c]) for c in comps]


def gilliland_N(R: float, Rmin: float, Nmin: float) -> float:
    X = (R - Rmin) / (R + 1.0)
    Y = 1.0 - math.exp((1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / math.sqrt(X))
    return (Nmin + Y) / (1.0 - Y) if Y < 1.0 - 1e-12 else 1e12


def design(components: list[str], feed: dict, pressure_Pa: float, light_key: str, heavy_key: str,
           recovery_light_key_in_distillate: float, recovery_heavy_key_in_bottoms: float, model: str,
           n_stages: float | None = None, reflux_ratio: float | None = None, R_over_Rmin: float | None = None,
           N_over_Nmin: float | None = None, condenser: str = "total", kij: list[dict] | None = None,
           activity_parameters: list[dict] | None = None) -> dict:
    comps = list(components)
    if light_key not in comps or heavy_key not in comps:
        raise ValueError("light_key y heavy_key deben estar en components (mismos nombres)")
    if sum(v is not None for v in (n_stages, reflux_ratio, R_over_Rmin, N_over_Nmin)) > 1:
        raise ValueError("da solo uno de n_stages, reflux_ratio, R_over_Rmin, N_over_Nmin")
    if condenser not in ("total", "partial"):
        raise ValueError("condenser debe ser 'total' o 'partial'")
    P = float(pressure_Pa)
    if feed.get("molar_flows") is not None:
        f = [float(feed["molar_flows"].get(c, 0.0)) for c in comps]
    else:
        zs0 = feed["mole_fractions"]
        zs0 = [float(zs0.get(c, 0.0)) for c in comps] if isinstance(zs0, dict) else [float(v) for v in zs0]
        F0 = float(feed.get("molar_flow", 1.0))
        f = [F0 * v / sum(zs0) for v in zs0]
    F = sum(f)
    z = [v / F for v in f]
    iL, iH = comps.index(light_key), comps.index(heavy_key)
    rL, rH = float(recovery_light_key_in_distillate), float(recovery_heavy_key_in_bottoms)
    if not (0.5 < rL < 1 and 0.5 < rH < 1):
        raise ValueError("las recuperaciones de las claves deben estar entre 0.5 y 1 (fracción, no %)")
    warnings: list[str] = []
    # feed condition at column pressure
    bub = _flash(comps, z, model, P, kij, activity_parameters, vapor_fraction=0.0)
    dew = _flash(comps, z, model, P, kij, activity_parameters, vapor_fraction=1.0)
    Kf = _K(bub, comps)
    if Kf[iL] <= Kf[iH]:
        raise ValueError(f"{light_key} no es más volátil que {heavy_key} a la presión de la columna con este modelo")
    if feed.get("q") is not None:
        q, hF, q_basis = float(feed["q"]), None, "del usuario"
    else:
        Pf = float(feed.get("P_Pa", P))
        if feed.get("vapor_fraction") is not None or feed.get("T_K") is not None:
            spec = ({"vapor_fraction": float(feed["vapor_fraction"])} if feed.get("vapor_fraction") is not None
                    else {"T_K": float(feed["T_K"])})
            ff = _flash(comps, z, model, Pf, kij, activity_parameters, **spec)
            if abs(Pf - P) > 1.0:                                   # adiabatic let-down to column pressure
                ff = _flash(comps, z, model, P, kij, activity_parameters, H_J_mol=float(ff["H_J_mol"]))
        else:
            ff = bub
            warnings.append("alimentación tomada como líquido saturado a la presión de la columna (no se dio q, T_K ni vapor_fraction)")
        hF = float(ff["H_J_mol"])
        q = (float(dew["H_J_mol"]) - hF) / (float(dew["H_J_mol"]) - float(bub["H_J_mol"]))
        q_basis = "q = (H_rocío − H_alim)/(H_rocío − H_burbuja) a la presión de la columna, con el mismo modelo"
    if hF is None:
        hF = float(dew["H_J_mol"]) - q * (float(dew["H_J_mol"]) - float(bub["H_J_mol"]))
    # Fenske with end-point volatilities, iterated on the non-key split
    d = [0.0] * len(comps)
    for i in range(len(comps)):
        d[i] = f[i] * (rL if Kf[i] >= Kf[iL] else (1 - rH) if Kf[i] <= Kf[iH] else 0.5)
    d[iL], d[iH] = rL * f[iL], (1 - rH) * f[iH]
    alpha_f = [k / Kf[iH] for k in Kf]
    top = bot = None
    for it in range(60):
        b = [fi - di for fi, di in zip(f, d)]
        D, B = sum(d), sum(b)
        xD, xB = [v / D for v in d], [max(v, 0.0) / B for v in b]
        top = _flash(comps, xD, model, P, kij, activity_parameters, vapor_fraction=1.0)   # overhead vapour: dew point
        bot = _flash(comps, xB, model, P, kij, activity_parameters, vapor_fraction=0.0)
        Kt, Kb = _K(top, comps), _K(bot, comps)
        at, ab = [k / Kt[iH] for k in Kt], [k / Kb[iH] for k in Kb]
        alpha = [math.sqrt(x * y) for x, y in zip(at, ab)]
        Nmin = math.log((d[iL] / b[iL]) * (b[iH] / d[iH])) / math.log(alpha[iL])
        new = []
        for i in range(len(comps)):
            ratio = (d[iH] / b[iH]) * alpha[i] ** Nmin            # d_i / b_i
            new.append(f[i] * ratio / (1.0 + ratio))
        new[iL], new[iH] = rL * f[iL], (1 - rH) * f[iH]
        change = max(abs(n_ - o) for n_, o in zip(new, d)) / F
        d = new
        if change < 1e-10:
            break
    else:
        warnings.append("la distribución de los no clave no convergió en 60 iteraciones")
    b = [fi - di for fi, di in zip(f, d)]
    D, B = sum(d), sum(b)
    xD, xB = [v / D for v in d], [v / B for v in b]
    # Underwood (alpha = geometric mean of top and bottom, as in Fenske)
    und = lambda th: sum(a * zi / (a - th) for a, zi in zip(alpha, z)) - (1.0 - q)
    eps = 1e-9 * (alpha[iL] - 1.0)
    theta = brentq(und, 1.0 + eps, alpha[iL] - eps)
    Rmin = sum(a * x / (a - theta) for a, x in zip(alpha, xD)) - 1.0
    between = [c for c, a in zip(comps, alpha) if 1.0 < a < alpha[iL] and c not in (light_key, heavy_key)]
    if between:
        warnings.append(f"hay componentes entre las claves ({', '.join(between)}): Underwood con una sola raíz es aproximado")
    out_op: dict = {}
    R = N = None
    if Rmin <= 0:
        warnings.append("R_min ≤ 0 con Underwood: la separación no necesita reflujo según el atajo; revisa las claves")
    else:
        if reflux_ratio is not None:
            R = float(reflux_ratio)
        elif R_over_Rmin is not None:
            R = float(R_over_Rmin) * Rmin
        elif n_stages is not None or N_over_Nmin is not None:
            N = float(n_stages) if n_stages is not None else float(N_over_Nmin) * Nmin
            if N <= Nmin:
                out_op = {"feasible": False, "reason": f"{N:.2f} etapas no superan N_min = {Nmin:.2f}"}
            else:
                R = brentq(lambda r: gilliland_N(r, Rmin, Nmin) - N, Rmin * 1.0005, Rmin * 1e4 + 1.0)
        if R is not None:
            if R <= Rmin:
                out_op = {"feasible": False, "reason": f"R = {R:.4g} no supera R_min = {Rmin:.4g}"}
            else:
                N = gilliland_N(R, Rmin, Nmin)
                ratio = ((z[iH] / z[iL]) * (xB[iL] / xD[iH]) ** 2 * (B / D)) ** 0.206
                NR = N * ratio / (1.0 + ratio)
                out_op = {"feasible": True, "reflux_ratio": R, "R_over_R_min": R / Rmin, "stages": N,
                          "N_over_N_min": N / Nmin, "stages_above_feed_kirkbride": NR, "stages_below_feed_kirkbride": N - NR,
                          "feed_stage_from_top": NR + 1.0}
                hV = float(top["H_J_mol"])
                dist = top if condenser == "partial" else _flash(comps, xD, model, P, kij, activity_parameters, vapor_fraction=0.0)
                hD, hB = float(dist["H_J_mol"]), float(bot["H_J_mol"])
                if condenser == "total":
                    Qc = D * (R + 1.0) * (hV - hD)
                else:   # vapour distillate: only the reflux is condensed (latent heat of the distillate mixture)
                    hL = float(_flash(comps, xD, model, P, kij, activity_parameters, vapor_fraction=0.0)["H_J_mol"])
                    Qc = D * R * (hV - hL)
                    warnings.append("condensador parcial: carga estimada con el calor de condensación de una mezcla de la "
                                    "composición del destilado (el reflujo real es algo más pesado)")
                Qr = D * hD + B * hB + Qc - F * hF
                out_op.update(condenser_duty=Qc, reboiler_duty=Qr, duty_units="J por unidad de tiempo si los flujos están "
                              "en mol por unidad de tiempo (J/mol × flujo molar dado)",
                              T_distillate_K=float(dist["T_K"]))
    res = {
        "ok": True, "method": "Fenske-Underwood-Gilliland(Molokanov)-Kirkbride con α = media geométrica de cabeza "
              "(punto de rocío del vapor de cabeza) y fondo (punto de burbuja) a la presión de la columna; K de "
              f"independent_flash ({bub.get('model')})",
        "components": comps, "pressure_Pa": P, "condenser": condenser,
        "feed": {"molar_flow": F, "mole_fractions": dict(zip(comps, z)), "q": q, "q_basis": q_basis,
                 "T_bubble_K": float(bub["T_K"]), "T_dew_K": float(dew["T_K"])},
        "products": {"distillate_flow": D, "bottoms_flow": B, "distillate_flows": dict(zip(comps, d)),
                     "bottoms_flows": dict(zip(comps, b)), "x_distillate": dict(zip(comps, xD)),
                     "x_bottoms": dict(zip(comps, xB)), "T_top_dew_K": float(top["T_K"]), "T_bottoms_bubble_K": float(bot["T_K"])},
        "relative_volatility_to_heavy_key": {"top": dict(zip(comps, at)), "bottom": dict(zip(comps, ab)),
                                             "feed_bubble": dict(zip(comps, alpha_f)), "used_geometric_mean": dict(zip(comps, alpha))},
        "N_min": Nmin, "R_min": Rmin, "underwood_theta": theta, "operation": out_op, "warnings": warnings,
        "assumptions": list(bub.get("assumptions", [])),
        "not_included": ["balance etapa a etapa (es un atajo: confirma con una columna rigurosa)", "eficiencias de plato",
                         "caída de presión en la columna", "azeótropos y mezclas fuertemente no ideales (para binarios usa "
                         "binary_distillation_design)"],
    }
    alpha_feed_N = math.log((d[iL] / b[iL]) * (b[iH] / d[iH])) / math.log(alpha_f[iL])
    res["sensitivity"] = {"N_min_if_alpha_at_feed_only": alpha_feed_N,
                          "note": "N_min que saldría con α evaluada solo en la alimentación (lo que hace el ShortcutColumn de "
                                  "DWSIM): solo como contraste"}
    bal = max(abs(di + bi - fi) for di, bi, fi in zip(d, b, f)) / F
    res["evidence"] = {"checks": {"component_balance_rel": bal, "fenske_iterations": it + 1,
                                  "underwood_residual": abs(und(theta)),
                                  "alpha_LK_top_over_bottom": at[iL] / ab[iL]},
                       "verified": bool(bal < 1e-9 and abs(und(theta)) < 1e-8 and Rmin > 0),
                       "meaning": "balance por componente y raíz de Underwood cerrados; certifica la aritmética del atajo "
                                  "con ESTE modelo, no que el atajo represente la columna real"}
    if not 0.8 < at[iL] / ab[iL] < 1.25:
        warnings.append(f"α de la clave ligera cambia {at[iL]:.3f} (cabeza) → {ab[iL]:.3f} (fondo): la media geométrica es "
                        "una aproximación gruesa; confirma con una columna rigurosa")
    res["ok"] = res["evidence"]["verified"]
    return res
