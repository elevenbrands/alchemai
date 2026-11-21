"""Generalized compressibility (Lee-Kesler 1975) — the corresponding-states chart of the textbooks, computed.

Z(Tr, Pr, ω) = Z0 + (ω/ω_r)(Z_r − Z0), with Z0 the simple fluid and Z_r the reference fluid (n-octane, ω_r = 0.3978),
each from the same modified BWR equation in the ideal reduced volume Vr = Pc·V/(R·Tc):

  Z = 1 + B/Vr + C/Vr² + D/Vr⁵ + c4/(Tr³Vr²)·(β + γ/Vr²)·exp(−γ/Vr²)
  B = b1 − b2/Tr − b3/Tr² − b4/Tr³,  C = c1 − c2/Tr + c3/Tr³,  D = d1 + d2/Tr

Constants as published by Lee & Kesler (AIChE J. 21, 510, 1975), WRITTEN FROM MEMORY in this module; they are checked
against CoolProp's reference equations of state in tests/test_lee_kesler.py (argon, methane, nitrogen, CO2, propane,
n-octane), which would not agree within a few percent if a constant were wrong. Mixtures: Kay's rule (molar averages of
Tc, Pc, ω). Hydrogen and helium: Newton's corrections Tc + 8 K and Pc + 8 atm (textbook rule, e.g. Felder & Rousseau;
also written from memory). Measured against CoolProp for H2 and He (50–300 K, 2–20 MPa): with the two-parameter chart
(Z0 only, as in the textbooks) the correction brings Z within 0.2–4 % (without it 3–9 % high); with the three-parameter
form and the library's negative ω it makes Z WORSE (3–7 % low; without it 0.1–1.2 %). Hence the default: corrected only
for chart='two_parameter'.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

R = 8.314462618
ATM = 101325.0
OMEGA_R = 0.3978
SIMPLE = dict(b1=0.1181193, b2=0.265728, b3=0.154790, b4=0.030323, c1=0.0236744, c2=0.0186984, c3=0.0, c4=0.042724,
              d1=0.155488e-4, d2=0.623689e-4, beta=0.65392, gamma=0.060167)
REFERENCE = dict(b1=0.2026579, b2=0.331511, b3=0.027655, b4=0.203488, c1=0.0313385, c2=0.0503618, c3=0.016901,
                 c4=0.041577, d1=0.48736e-4, d2=0.0740336e-4, beta=1.226, gamma=0.03754)
QUANTUM = {"1333-74-0": "hidrógeno", "7440-59-7": "helio"}


def _z_of(c: dict, Tr: float, Vr: float) -> float:
    B = c["b1"] - c["b2"] / Tr - c["b3"] / Tr ** 2 - c["b4"] / Tr ** 3
    C = c["c1"] - c["c2"] / Tr + c["c3"] / Tr ** 3
    D = c["d1"] + c["d2"] / Tr
    e = c["gamma"] / Vr ** 2
    return 1 + B / Vr + C / Vr ** 2 + D / Vr ** 5 + c["c4"] / (Tr ** 3 * Vr ** 2) * (c["beta"] + e) * math.exp(-e)


def _roots_Vr(c: dict, Tr: float, Pr: float) -> list[float]:
    f = lambda v: _z_of(c, Tr, v) - Pr * v / Tr
    grid = [10 ** (-1.6 + i * 0.004) for i in range(1901)]        # Vr from 0.025 to ~1e6
    vals = [f(v) for v in grid]
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-14) for i in range(1900) if vals[i] * vals[i + 1] < 0]


def z_fluid(c: dict, Tr: float, Pr: float, phase: str) -> tuple[float, int]:
    roots = _roots_Vr(c, Tr, Pr)
    if not roots:
        raise ValueError(f"sin raíz de Lee-Kesler en Tr = {Tr:.3f}, Pr = {Pr:.3f}")
    v = max(roots) if phase == "gas" else min(roots)
    return Pr * v / Tr, len(roots)


def z_lk(Tr: float, Pr: float, omega: float, phase: str = "gas") -> dict:
    z0, n0 = z_fluid(SIMPLE, Tr, Pr, phase)
    zr, nr = z_fluid(REFERENCE, Tr, Pr, phase)
    z1 = (zr - z0) / OMEGA_R
    return {"Z": z0 + omega * z1, "Z0": z0, "Z1": z1, "several_roots": n0 > 1 or nr > 1}


def pseudo_critical(components: list[str], mole_fractions: list[float] | None, critical: dict | None,
                    quantum_gas_correction: bool):
    from chemicals import CAS_from_any, Pc, Tc, omega
    critical = critical or {}
    zs = list(mole_fractions) if mole_fractions else [1.0]
    if len(zs) != len(components):
        raise ValueError("mole_fractions debe tener un valor por componente")
    tot = sum(zs)
    zs = [z / tot for z in zs]
    rows, notes = [], []
    for name, z in zip(components, zs):
        user = critical.get(name, {})
        try:
            cas = CAS_from_any(name)
        except Exception:  # noqa: BLE001
            cas = None
            if not {"Tc_K", "Pc_Pa", "omega"} <= set(user):
                raise ValueError(f"compuesto no reconocido: {name}; da critical {{'{name}': {{Tc_K, Pc_Pa, omega}}}}") from None
        tc = float(user.get("Tc_K", Tc(cas) if cas else math.nan))
        pc = float(user.get("Pc_Pa", Pc(cas) if cas else math.nan))
        w = float(user.get("omega", omega(cas) if cas else math.nan))
        src = "del usuario" if user else "chemicals (Tc, Pc, ω)"
        if quantum_gas_correction and cas in QUANTUM:
            tc, pc = tc + 8.0, pc + 8.0 * ATM
            notes.append(f"{name}: corrección de Newton para {QUANTUM[cas]} (Tc + 8 K, Pc + 8 atm)")
        if any(math.isnan(v) for v in (tc, pc, w)):
            raise ValueError(f"faltan constantes críticas de {name}; dalas en critical")
        rows.append({"component": name, "y": z, "Tc_K": tc, "Pc_Pa": pc, "omega": w, "source": src})
    tc_m = sum(r["y"] * r["Tc_K"] for r in rows)
    pc_m = sum(r["y"] * r["Pc_Pa"] for r in rows)
    w_m = sum(r["y"] * r["omega"] for r in rows)
    if quantum_gas_correction is False and any(QUANTUM.get(_safe_cas(r["component"])) for r in rows):
        notes.append("hay hidrógeno o helio sin corrección de Newton: con quantum_gas_correction=True se aplica Tc + 8 K, Pc + 8 atm")
    return rows, tc_m, pc_m, w_m, notes


def _safe_cas(name):
    from chemicals import CAS_from_any
    try:
        return CAS_from_any(name)
    except Exception:  # noqa: BLE001
        return None


def generalized_compressibility(components: list[str], T_K: float | None = None, P_Pa: float | None = None,
                                molar_volume_m3_mol: float | None = None, mole_fractions: list[float] | None = None,
                                moles: float | None = None, volume_m3: float | None = None, phase: str = "gas",
                                critical: dict | None = None, quantum_gas_correction: bool | None = None,
                                chart: str = "lee_kesler") -> dict:
    if chart not in ("lee_kesler", "two_parameter"):
        raise ValueError("chart debe ser 'lee_kesler' (tres parámetros, con ω) o 'two_parameter' (z(Tr, Pr) sin ω, la gráfica de los libros)")
    if quantum_gas_correction is None:
        quantum_gas_correction = chart == "two_parameter"
    if phase not in ("gas", "liquid"):
        raise ValueError("phase debe ser 'gas' o 'liquid'")
    if volume_m3 is not None:
        if moles is None:
            raise ValueError("con volume_m3 da también moles")
        if molar_volume_m3_mol is not None:
            raise ValueError("da molar_volume_m3_mol o volume_m3 + moles, no ambos")
        molar_volume_m3_mol = float(volume_m3) / float(moles)
    given = {k: v for k, v in (("T_K", T_K), ("P_Pa", P_Pa), ("molar_volume_m3_mol", molar_volume_m3_mol)) if v is not None}
    if len(given) != 2:
        raise ValueError("da exactamente dos de T_K, P_Pa y el volumen (molar_volume_m3_mol o volume_m3 + moles)")
    rows, Tc_, Pc_, w, notes = pseudo_critical(components, mole_fractions, critical, quantum_gas_correction)
    if chart == "two_parameter":
        w = 0.0
        notes.append("gráfica de dos parámetros: z(Tr, Pr) del fluido simple de Lee-Kesler, sin factor acéntrico")
    if T_K is not None and P_Pa is not None:
        Tr, Pr = float(T_K) / Tc_, float(P_Pa) / Pc_
        zz = z_lk(Tr, Pr, w, phase)
        T, P = float(T_K), float(P_Pa)
        V = zz["Z"] * R * T / P
    elif T_K is not None:                     # T, V given: find P
        T, V = float(T_K), float(molar_volume_m3_mol)
        Tr = T / Tc_
        g = lambda lp: z_lk(Tr, math.exp(lp) / Pc_, w, phase)["Z"] * R * T / math.exp(lp) - V
        P = math.exp(_bracket_root(g, math.log(1e2), math.log(Pc_ * 30)))
        Pr = P / Pc_
        zz = z_lk(Tr, Pr, w, phase)
    else:                                      # P, V given: find T
        P, V = float(P_Pa), float(molar_volume_m3_mol)
        Pr = P / Pc_
        g = lambda t: z_lk(t / Tc_, Pr, w, phase)["Z"] * R * t - P * V
        T = _bracket_root(g, max(0.3 * Tc_, 1.0), 15 * Tc_)
        Tr = T / Tc_
        zz = z_lk(Tr, Pr, w, phase)
    out = {"ok": True, "Z": zz["Z"], "Z0_simple_fluid": zz["Z0"], "Z1_correction": zz["Z1"], "T_K": T, "P_Pa": P,
           "molar_volume_m3_mol": V, "reduced": {"Tr": Tr, "Pr": Pr, "Vr_ideal": Pc_ * V / (R * Tc_)},
           "pseudo_critical": {"Tc_K": Tc_, "Pc_Pa": Pc_, "omega": w, "rule": "regla de Kay (promedios molares)" if len(rows) > 1 else "componente puro",
                               "components": rows},
           "Z_ideal_gas": 1.0, "phase": phase, "assumptions": notes, "warnings": [],
           "chart": chart,
           "method": ("Lee-Kesler (1975), Z = Z0 + (ω/0.3978)(Z_ref − Z0)" if chart == "lee_kesler" else
                      "fluido simple de Lee-Kesler, z(Tr, Pr) sin ω: la gráfica generalizada de dos parámetros de los libros"),
           "not_included": ["entalpía y entropía residuales", "reglas de mezcla de Lee-Kesler-Plöcker (se usa Kay)"]}
    if moles is not None:
        out["moles"] = float(moles)
        out["volume_m3"] = float(moles) * V
    if zz["several_roots"]:
        out["warnings"].append(f"la ecuación tiene más de una raíz en Tr = {Tr:.3f}, Pr = {Pr:.3f} (zona de dos fases o "
                               f"cerca de ella): se tomó la de {'vapor' if phase == 'gas' else 'líquido'}")
    if Tr < 0.3 or Tr > 4.0 or Pr > 10.0:
        out["warnings"].append(f"Tr = {Tr:.2f}, Pr = {Pr:.2f} fuera del intervalo de ajuste de Lee-Kesler (0.3 ≤ Tr ≤ 4, Pr ≤ 10)")
    check = abs(P * V / (R * T) - out["Z"]) / out["Z"]
    out["evidence"] = {"checks": {"PV_over_RT_minus_Z_rel": check}, "verified": check < 1e-8,
                       "meaning": "Z, T, P y V son coherentes entre sí; la exactitud de Lee-Kesler es la de una gráfica "
                                  "generalizada (típico 1–3 % para gases no polares, peor para polares y cerca del punto crítico)"}
    if any(r["source"] != "del usuario" for r in rows):
        out["assumptions"].append("Tc, Pc y ω de chemicals donde el usuario no los dio")
    out["ok"] = out["evidence"]["verified"]
    return out


def _bracket_root(g, lo: float, hi: float) -> float:
    n = 200
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    vs = []
    for x in xs:
        try:
            vs.append(g(x))
        except ValueError:
            vs.append(math.nan)
    for i in range(n):
        if vs[i] == vs[i] and vs[i + 1] == vs[i + 1] and vs[i] * vs[i + 1] < 0:
            return brentq(g, xs[i], xs[i + 1], xtol=1e-12 * max(abs(xs[i]), 1.0))
    raise ValueError("no se encontró solución en el intervalo de Lee-Kesler para esos datos")
