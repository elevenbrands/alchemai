"""Packed-column hydraulics (bloque 23): flooding gas velocity, irrigated pressure drop and the column diameter for a
chosen fraction of flooding, with the Stichlmair, Bravo & Fair (1989) model as implemented in `fluids` (MIT); the
`fluids` docstrings reproduce the paper's worked example (wet ΔP 539.9 Pa/m, flooding 0.639 m/s).

PACKINGS: Stichlmair constants as tabulated by Arachchige & Melaaen (2012), reproduced in Chem. Eng. Trans. 70 (2018)
1891-1896 (AIDIC, Table 2), citing Stichlmair et al. (1989). Secondary source: confirm with the vendor or the 1989 paper
before a final design; any packing can be passed as a dict instead.
"""
from __future__ import annotations

import math

PACKINGS = {
    "BX": {"voidage": 0.86, "specific_area": 450.0, "C1": 15.0, "C2": 2.0, "C3": 0.35, "kind": "structured"},
    "Flexipac 250Y": {"voidage": 0.99, "specific_area": 250.0, "C1": 0.866, "C2": -0.088, "C3": 0.698, "kind": "structured"},
    "Mellapak 250Y": {"voidage": 0.98, "specific_area": 250.0, "C1": 1.0, "C2": 1.0, "C3": 0.32, "kind": "structured"},
    "Pall rings 16 mm": {"voidage": 0.93, "specific_area": 341.0, "C1": 0.05, "C2": 1.0, "C3": 3.0, "kind": "random", "size_m": 0.016},
    "Pall rings 25 mm": {"voidage": 0.94, "specific_area": 205.0, "C1": 0.05, "C2": 1.0, "C3": 3.0, "kind": "random", "size_m": 0.025},
    "IMTP 25": {"voidage": 0.97, "specific_area": 207.0, "C1": 0.815, "C2": -0.106, "C3": 1.499, "kind": "random", "size_m": 0.025},
}
SOURCE = "Stichlmair et al. 1989 via Arachchige & Melaaen 2012 (CET 70, 2018, Tabla 2): fuente secundaria"


def _pk(packing) -> dict:
    if isinstance(packing, str):
        if packing not in PACKINGS:
            raise ValueError(f"unknown packing {packing!r}: {sorted(PACKINGS)} or pass a dict voidage/specific_area/C1/C2/C3")
        return PACKINGS[packing]
    return packing


def flooding_velocity(Vl: float, rhog: float, rhol: float, mug: float, packing) -> float:
    from fluids.packed_tower import Stichlmair_flood
    p = _pk(packing)
    return float(Stichlmair_flood(Vl=Vl, rhog=rhog, rhol=rhol, mug=mug, voidage=p["voidage"], specific_area=p["specific_area"],
                                  C1=p["C1"], C2=p["C2"], C3=p["C3"]))


def wet_dp_per_m(Vg: float, Vl: float, rhog: float, rhol: float, mug: float, packing) -> float:
    from fluids.packed_tower import Stichlmair_wet
    p = _pk(packing)
    return float(Stichlmair_wet(Vg=Vg, Vl=Vl, rhog=rhog, rhol=rhol, mug=mug, voidage=p["voidage"], specific_area=p["specific_area"],
                                C1=p["C1"], C2=p["C2"], C3=p["C3"], H=1.0))


def size_column(gas_m3_s: float, liquid_m3_s: float, rhog: float, rhol: float, mug: float, packing, flood_fraction: float = 0.7) -> dict:
    """Diameter where the gas velocity is `flood_fraction` of the flooding velocity at that same diameter's liquid load
    (bisection on D: flooding depends on the liquid superficial velocity, which depends on D)."""
    if not 0.2 <= flood_fraction <= 0.9:
        raise ValueError("flood_fraction should be 0.2-0.9 (design practice ~0.6-0.8)")

    def ratio(D):
        A = math.pi * D * D / 4.0
        try:
            vf = flooding_velocity(liquid_m3_s / A, rhog, rhol, mug, packing)
        except Exception:                         # no flooding solution: the liquid load alone floods this diameter
            return math.inf
        if not math.isfinite(vf) or vf <= 0:
            return math.inf
        return (gas_m3_s / A) / vf - flood_fraction

    # start from a 3 m/s gas velocity guess and bracket
    hi = max(0.05, math.sqrt(4.0 * gas_m3_s / (math.pi * 3.0)))
    lo = hi
    while ratio(lo) <= 0 and lo > 0.01:
        lo /= 2.0
    while ratio(hi) > 0:
        lo, hi = hi, hi * 1.5
        if hi > 40:
            raise ValueError("diameter above 40 m: split the gas into parallel columns")
    while ratio(hi) > 0:
        lo, hi = hi, hi * 2.0
        if hi > 40:
            raise ValueError("diameter above 40 m: split the gas into parallel columns")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if hi - lo < 1e-6:
            break
        if ratio(mid) > 0:
            lo = mid
        else:
            hi = mid
    D = hi
    A = math.pi * D * D / 4.0
    Vg, Vl = gas_m3_s / A, liquid_m3_s / A
    vf = flooding_velocity(Vl, rhog, rhol, mug, packing)
    return {"diameter_m": D, "area_m2": A, "gas_velocity_m_s": Vg, "liquid_velocity_m_s": Vl, "flooding_velocity_m_s": vf,
            "flood_fraction": Vg / vf, "F_factor_Pa05": Vg * math.sqrt(rhog), "dP_wet_Pa_per_m": wet_dp_per_m(Vg, Vl, rhog, rhol, mug, packing),
            "packing": packing if isinstance(packing, str) else "custom", "constants_source": SOURCE if isinstance(packing, str) else "user"}
