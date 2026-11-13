"""Vapour pressures, gamma-phi bubble points and azeotropes for a binary.

Modified Raoult (ideal gas): y_i P = x_i γ_i Psat_i. Optional gas-phase correction (gamma/phi formulation, Smith,
Van Ness & Abbott, ch. 14): y_i Φ_i P = x_i γ_i Psat_i with Φ_i = exp{[(B_ii - V_i^L)(P - Psat_i) + P y_j² δ_12]/RT},
δ_12 = 2B_12 - B_11 - B_22; B from Tsonopoulos (chemicals.virial, non-polar form, no association terms) and the
Prausnitz combining rules for B_12 (Tc12 = √(Tc1Tc2), Vc12 = ((Vc1^⅓+Vc2^⅓)/2)³, Zc12 and ω12 arithmetic means). V_i^L from thermo.VolumeLiquid.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from packages.vle.models import R, ln_gamma


class DataError(ValueError):
    pass


# ---------------------------------------------------------------- vapour pressure
_P_UNIT = {"Pa": 1.0, "kPa": 1e3, "bar": 1e5, "mmHg": 133.322387415, "atm": 101325.0, "MPa": 1e6}


class Psat:
    """Psat(T) in Pa from chemicals/thermo (default method, named) or from a user equation:
      {"antoine": {A, B, C, "log": "log10"|"ln", "P_unit": "mmHg"|"kPa"|"Pa"|"bar", "T_unit": "C"|"K"}}  log P = A - B/(T + C)
      {"antoine_ext": [A1..A7] (Aspen PLXANT: ln P[kPa] = A1 + A2/(T+A3) + A4 T + A5 ln T + A6 T^A7, T in K),
       optional "T_range_K": [lo, hi]}"""

    def __init__(self, name: str, cas: str, spec: dict | None):
        self.name, self.cas, self.spec = name, cas, spec or {}
        if "antoine" in self.spec:
            a = self.spec["antoine"]
            miss = [k for k in ("A", "B", "C") if a.get(k) is None]
            if miss:
                raise DataError(f"Antoine de {name}: faltan {miss}")
            self.kind = "antoine"
            self.source = (f"Antoine del usuario: {a.get('log', 'log10')} P[{a.get('P_unit', 'mmHg')}] = A - B/(T[{a.get('T_unit', 'C')}] + C)"
                           f", A={a['A']}, B={a['B']}, C={a['C']}")
            self.T_range = self.spec.get("T_range_K")
        elif "antoine_ext" in self.spec:
            c = list(self.spec["antoine_ext"])
            if len(c) != 7:
                raise DataError(f"antoine_ext de {name}: se esperan 7 constantes A1..A7 (forma PLXANT de Aspen)")
            self.kind = "antoine_ext"
            self.source = ("Antoine extendida del usuario (forma PLXANT de Aspen Plus): ln P[kPa] = A1 + A2/(T+A3) + A4·T + "
                           f"A5·ln T + A6·T^A7, T en K; constantes {c}")
            self.T_range = self.spec.get("T_range_K")
        else:
            from thermo import VaporPressure
            self.kind = "thermo"
            self._vp = VaporPressure(CASRN=cas)
            if self._vp.method is None:
                raise DataError(f"thermo no tiene presión de vapor para {name} ({cas}); da una ecuación de Antoine")
            self.T_range = [self._vp.T_limits[self._vp.method][0], self._vp.T_limits[self._vp.method][1]]
            self.source = f"thermo.VaporPressure, método {self._vp.method} (chemicals; rango {self.T_range[0]:.1f}-{self.T_range[1]:.1f} K)"

    def __call__(self, T) -> np.ndarray:
        T = np.asarray(T, float)
        if self.kind == "antoine":
            a = self.spec["antoine"]
            t = T - 273.15 if a.get("T_unit", "C") == "C" else T
            v = a["A"] - a["B"] / (t + a["C"])
            P = 10.0 ** v if a.get("log", "log10") == "log10" else np.exp(v)
            return P * _P_UNIT[a.get("P_unit", "mmHg")]
        if self.kind == "antoine_ext":
            A1, A2, A3, A4, A5, A6, A7 = self.spec["antoine_ext"]
            return 1e3 * np.exp(A1 + A2 / (T + A3) + A4 * T + A5 * np.log(T) + A6 * T ** A7)
        return np.vectorize(self._vp.T_dependent_property, otypes=[float])(T)

    def Tsat(self, P: float) -> float:
        f = lambda T: math.log(float(self(T))) - math.log(P)
        lo, hi = 150.0, 700.0
        if self.T_range:
            lo, hi = max(lo, self.T_range[0]), min(hi, self.T_range[1] - 1e-6)
        return brentq(f, lo, hi)


# ---------------------------------------------------------------- vapour phase
class VaporPhase:
    def __init__(self, kind: str, cas: tuple[str, str]):
        self.kind = kind
        if kind == "ideal":
            self.source = "gas ideal (Φ = 1)"
            return
        if kind != "virial":
            raise DataError("vapor_phase debe ser 'ideal' o 'virial'")
        from chemicals import Tc, Pc, Vc, omega
        from thermo import VolumeLiquid
        c = [dict(Tc=Tc(s), Pc=Pc(s), Vc=Vc(s), w=omega(s)) for s in cas]
        for s, d in zip(cas, c):
            if any(v is None for v in d.values()):
                raise DataError(f"chemicals no tiene Tc/Pc/Vc/ω de {s}; usa vapor_phase='ideal'")
        self.c = c
        Tc12 = math.sqrt(c[0]["Tc"] * c[1]["Tc"])
        Vc12 = ((c[0]["Vc"] ** (1 / 3) + c[1]["Vc"] ** (1 / 3)) / 2) ** 3
        Zc12 = (c[0]["Pc"] * c[0]["Vc"] / (R * c[0]["Tc"]) + c[1]["Pc"] * c[1]["Vc"] / (R * c[1]["Tc"])) / 2
        self.c12 = dict(Tc=Tc12, Pc=Zc12 * R * Tc12 / Vc12, w=(c[0]["w"] + c[1]["w"]) / 2)
        self._vl = [VolumeLiquid(CASRN=s) for s in cas]
        self.source = ("virial truncado en B (Tsonopoulos no polar de chemicals.virial; B12 por las reglas de "
                       "Prausnitz: Tc12 geométrica, Vc12 cúbica, Zc12 y ω12 medias) con Poynting (V líquido de thermo.VolumeLiquid). Sin términos de "
                       "asociación: aproximado para alcoholes, agua y ácidos")

    def _B(self, T, d):
        from chemicals.virial import BVirial_Tsonopoulos
        return np.vectorize(lambda t: BVirial_Tsonopoulos(t, d["Tc"], d["Pc"], d["w"]), otypes=[float])(T)

    def phis(self, T, P, y1, ps1, ps2):
        if self.kind == "ideal":
            one = np.ones_like(np.asarray(T, float))
            return one, one
        B11, B22, B12 = self._B(T, self.c[0]), self._B(T, self.c[1]), self._B(T, self.c12)
        V1 = np.vectorize(self._vl[0].T_dependent_property, otypes=[float])(T)
        V2 = np.vectorize(self._vl[1].T_dependent_property, otypes=[float])(T)
        d12 = 2 * B12 - B11 - B22
        y2 = 1 - y1
        RT = R * np.asarray(T, float)
        F1 = np.exp(((B11 - V1) * (P - ps1) + P * y2 ** 2 * d12) / RT)
        F2 = np.exp(((B22 - V2) * (P - ps2) + P * y1 ** 2 * d12) / RT)
        return F1, F2


# ---------------------------------------------------------------- bubble points
class System:
    def __init__(self, psat: tuple[Psat, Psat], vapor: VaporPhase):
        self.psat, self.vapor = psat, vapor

    def gammas_fn(self, model, p):
        return lambda T, x1: ln_gamma(model, T, x1, p)

    def bubble_P(self, T, x1, lng):
        """lng(T, x1) -> (ln γ1, ln γ2). Returns P [Pa], y1."""
        T, x1 = np.asarray(T, float), np.asarray(x1, float)
        l1, l2 = lng(T, x1)
        ps1, ps2 = self.psat[0](T), self.psat[1](T)
        a1, a2 = x1 * np.exp(l1) * ps1, (1 - x1) * np.exp(l2) * ps2
        P = a1 + a2
        y1 = a1 / P
        if self.vapor.kind == "ideal":
            return P, y1
        for _ in range(60):
            F1, F2 = self.vapor.phis(T, P, y1, ps1, ps2)
            Pn = a1 / F1 + a2 / F2
            y1n = a1 / F1 / Pn
            if np.max(np.abs(Pn / P - 1)) < 1e-11 and np.max(np.abs(y1n - y1)) < 1e-11:
                P, y1 = Pn, y1n
                break
            P, y1 = Pn, y1n
        return P, y1

    def bubble_T(self, P, x1, lng, T0=None):
        """Bubble temperature at pressure P [Pa] (array) by vectorised Newton on ln P(T); brentq fallback per point."""
        P, x1 = np.asarray(P, float), np.asarray(x1, float)
        if T0 is None:
            T0 = np.array([x * self.psat[0].Tsat(p) + (1 - x) * self.psat[1].Tsat(p) for x, p in zip(x1, P)])
        T = np.array(T0, float)
        ok = False
        for _ in range(40):
            f = np.log(self.bubble_P(T, x1, lng)[0]) - np.log(P)
            h = 1e-3
            df = (np.log(self.bubble_P(T + h, x1, lng)[0]) - np.log(P) - f) / h
            step = np.clip(f / df, -20, 20)
            if not np.all(np.isfinite(step)):
                break
            T = T - step
            if np.max(np.abs(step)) < 1e-9:
                ok = True
                break
        if not ok:
            for i in range(len(T)):
                g = lambda t: float(np.log(self.bubble_P(np.array([t]), x1[i:i + 1], lng)[0][0]) - np.log(P[i]))
                T[i] = brentq(g, 150.0, 700.0)
        return T, self.bubble_P(T, x1, lng)[1]

    def azeotropes(self, lng, T=None, P=None, n=400):
        """Interior x1 where y1 = x1 (isothermal at T or isobaric at P)."""
        xs = np.linspace(1e-4, 1 - 1e-4, n)

        def d(x):
            x = np.atleast_1d(np.asarray(x, float))
            if T is not None:
                return self.bubble_P(np.full_like(x, T), x, lng)[1] - x
            return self.bubble_T(np.full_like(x, P), x, lng)[1] - x
        v = d(xs)
        out = []
        for i in range(n - 1):
            if np.sign(v[i]) != np.sign(v[i + 1]):
                xa = brentq(lambda z: float(d(z)[0]), xs[i], xs[i + 1], xtol=1e-12)
                if T is not None:
                    Pa = float(self.bubble_P(np.array([T]), np.array([xa]), lng)[0][0])
                    out.append({"x1": xa, "T_K": float(T), "P_Pa": Pa})
                else:
                    Ta = float(self.bubble_T(np.array([P]), np.array([xa]), lng)[0][0])
                    out.append({"x1": xa, "T_K": Ta, "P_Pa": float(P)})
        return out
