"""Pure-component properties for consequence models, from `thermo.Chemical` (MIT, ChEDL) and CoolProp (MIT).

Every value returned carries its origin so the tool can list it in `assumptions`. Weather, inventory and release data
never come from here: those are the user's.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

R = 8.314462618          # J/(mol K)
G = 9.80665              # m/s2
P_ATM = 101325.0         # Pa


@dataclass
class Compound:
    name: str
    cas: str
    M: float                 # kg/mol
    Tb: float                # K, normal boiling point
    Hvap_Tb: float           # J/kg at Tb
    Hc_lower: float          # J/kg, net (lower) heat of combustion, > 0 if combustible, 0 otherwise
    LFL: float | None        # volume fraction
    UFL: float | None
    source: str = "thermo.Chemical 0.6.1 (ChEDL)"

    def Cp_liquid(self, T: float) -> float | None:
        """Liquid Cp (J/kg K) at T, or None."""
        from thermo import Chemical
        try:
            c = Chemical(self.cas, T=T, P=max(P_ATM, _psat_safe(self.cas, T) * 1.01))
            return float(c.Cpl) if c.Cpl else None
        except Exception:  # noqa: BLE001
            return None

    def psat(self, T: float) -> float:
        return _psat_safe(self.cas, T)

    def cp_ideal_gas_molar(self, T: float) -> float:
        from thermo import Chemical
        return float(Chemical(self.cas, T=T).Cpgm)


def _psat_safe(cas: str, T: float) -> float:
    from thermo import Chemical
    return float(Chemical(cas, T=T).Psat)


@lru_cache(maxsize=64)
def compound(name: str) -> Compound:
    from thermo import Chemical
    c = Chemical(name)
    hc = -float(c.Hc_lower) if c.Hc_lower is not None and c.Hc_lower < 0 else 0.0
    lfl = float(c.LFL) if c.LFL is not None and c.LFL > 0 and hc > 0 else None
    ufl = float(c.UFL) if c.UFL is not None and lfl is not None and c.UFL > lfl else None
    hv = float(Chemical(name, T=float(c.Tb)).Hvap)
    return Compound(name, c.CAS, float(c.MW) / 1000.0, float(c.Tb), hv, hc, lfl, ufl)


def water_vapour_pressure(T: float) -> float:
    """Saturation pressure of water (Pa), CoolProp (IAPWS-95)."""
    import CoolProp.CoolProp as CP
    return float(CP.PropsSI("P", "T", T, "Q", 0, "Water"))


def air_density(T: float, P: float = P_ATM) -> float:
    return P * 0.028964 / (R * T)
