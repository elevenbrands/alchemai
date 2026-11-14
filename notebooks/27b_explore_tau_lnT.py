"""M15c, exploratory (NOT in the tool): would tau = a + b/T + e·ln T (+ f·T) let ONE NRTL set represent the VLE and
the h^E of ethanol-water? Same data, sigmas (T 0.1 K, y 0.005, h^E 5 J/mol + 1 %) and weights 1 and 1 as the joint fit
of vle_binary_regression; α = 0.3; h^E by central difference of g^E/RT (independent of packages/vle/models.py: the
a + b/T row must reproduce the tool's J_all_T fit of notebooks/27).
Run: PYTHONPATH=. uv run python notebooks/27b_explore_tau_lnT.py  (seconds) → notebooks/27b_explore_tau_lnT.json"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R = 8.314462618
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
vle = json.loads((root / "tests/data/vle_ethanol_water_vlizard.json").read_text())
k = next(k for k in vle["sets"] if "101" in k)
pts = [q for q in vle["sets"][k] if 0 < q["x1"] < 1]
Tv, Pv, xv, yv = (np.array([q[c] for q in pts]) for c in ("T_K", "P_Pa", "x1", "y1"))
from packages.vle.equilibrium import Psat, System, VaporPhase  # noqa: E402
from packages.vle.tool import _cas  # noqa: E402
he = json.loads((root / "tests/data/he_ethanol_water_thermoml.json").read_text())
Th = np.concatenate([[s["T_K"]] * len(s["points"]) for s in he["sets"].values()])
xh = np.array([q["x1"] for s in he["sets"].values() for q in s["points"]])
hh = np.array([q["hE_J_mol"] for s in he["sets"].values() for q in s["points"]])
sg = 5 + 0.01 * np.abs(hh)
cas = (_cas("ethanol"), _cas("water"))
S = System((Psat("ethanol", cas[0], None), Psat("water", cas[1], None)), VaporPhase("ideal", cas))
T0 = 355.0
def taus(th, T, form):
    # th: c12, c21 (tau at T0), b12, b21, [e12, e21], [f12, f21]
    c12, c21, b12, b21 = th[:4]
    t12 = c12 + b12 * (1 / T - 1 / T0); t21 = c21 + b21 * (1 / T - 1 / T0)
    if form >= 1:
        t12 = t12 + th[4] * np.log(T / T0); t21 = t21 + th[5] * np.log(T / T0)
    if form >= 2:
        t12 = t12 + th[6] * (T - T0) / 100; t21 = t21 + th[7] * (T - T0) / 100
    return t12, t21
def lng(th, T, x, form, al=0.3):
    x1, x2 = x, 1 - x
    t12, t21 = taus(th, T, form)
    G12, G21 = np.exp(-al * t12), np.exp(-al * t21)
    d1, d2 = x1 + x2 * G21, x2 + x1 * G12
    return x2 ** 2 * (t21 * (G21 / d1) ** 2 + t12 * G12 / d2 ** 2), x1 ** 2 * (t12 * (G12 / d2) ** 2 + t21 * G21 / d1 ** 2)
def ge(th, T, x, form):
    a, b = lng(th, T, x, form)
    return x * a + (1 - x) * b
def hE(th, T, x, form):
    h = 1e-3 * T
    return -R * T ** 2 * (ge(th, T + h, x, form) - ge(th, T - h, x, form)) / (2 * h)
def res(th, form, w):
    Tc, yc = S.bubble_T(Pv, xv, lambda T, x: lng(th, T, x, form), T0=Tv)
    r = np.concatenate([(Tc - Tv) / 0.1, (yc - yv) / 0.005, np.sqrt(w) * (hE(th, Th, xh, form) - hh) / sg])
    return np.where(np.isfinite(r), r, 1e3)
out = {"note": "exploratorio: estas formas de tau NO están en la tool ni en la columna", "rows": []}
for form, n in ((0, 4), (1, 6), (2, 8)):
    best = None
    for b0 in ((0, 0), (300, -400), (900, -1100), (-50, 690)):
        th0 = np.zeros(n); th0[:2] = (-0.15, 1.9); th0[2:4] = b0
        try:
            s = least_squares(res, th0, args=(form, 1.0), x_scale="jac", max_nfev=600)
        except Exception as ex:
            continue
        if best is None or s.cost < best.cost: best = s
    th = best.x
    Tc, yc = S.bubble_T(Pv, xv, lambda T, x: lng(th, T, x, form), T0=Tv)
    e = hE(th, Th, xh, form) - hh
    per = [round(float(np.mean(np.abs(e[Th == s_["T_K"]]))), 1) for s_ in he["sets"].values()]
    lim = [round(0.1 * float(np.max(np.abs(hh[Th == s_["T_K"]]))), 1) for s_ in he["sets"].values()]
    out["rows"].append({"form": ["a+b/T", "a+b/T+e·lnT", "a+b/T+e·lnT+f·T"][form], "n_parameters": n,
                        "AAD_T_K": float(np.mean(np.abs(Tc - Tv))), "AAD_y": float(np.mean(np.abs(yc - yv))),
                        "AAD_hE_J_mol": float(np.mean(np.abs(e))), "AAD_hE_by_isotherm_J_mol": dict(zip(he["sets"], per)),
                        "tolerance_10pct_of_max_J_mol": dict(zip(he["sets"], lim)),
                        "hE_feed_10pct_350K_J_mol": float(hE(th, 350.0, 0.1, form)),
                        "theta_internal_c12_c21_b12_b21_e12_e21_f12_f21": th.tolist()})
    print(["a+b/T", "a+b/T+e·lnT", "a+b/T+e·lnT+f·T"][form], "npar", n, "AAD_T", round(float(np.mean(np.abs(Tc - Tv))), 3), "AAD_y", round(float(np.mean(np.abs(yc - yv))), 4),
          "AAD_hE", round(float(np.mean(np.abs(e))), 1), "per T", per, "tol(10% max)", lim, "feed350", round(float(hE(th, 350.0, 0.1, form)), 1), "theta", np.round(th, 3).tolist())
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
