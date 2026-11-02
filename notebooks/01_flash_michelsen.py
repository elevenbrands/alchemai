"""Week 1 study notebook — PT flash, Michelsen stability and the k_ij question.

Question raised by the baseline (notebooks/baseline_fluxo_vs_dwsim.md): Fluxo and DWSIM agree
on VF to 0.13 % but disagree 26 % on x_CO2. Both use Peng-Robinson. Is it the k_ij set?

Here the same flash is solved with `thermo` three times: k_ij = 0, Fluxo's Knapp subset, and a
"CO2-only" set, and Michelsen's tangent-plane stability test is run on the feed.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from thermo import PRMIX, CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL  # noqa: E402

IDS = ["methane", "ethane", "propane", "n-butane", "n-pentane", "nitrogen", "carbon dioxide"]
Z = [0.80, 0.08, 0.04, 0.02, 0.02, 0.02, 0.02]
T, P = 230.0, 30e5

# Fluxo's k_ij subset (apps/fluxo/.../BaseDeComponentes.swift, Knapp et al. 1982)
FLUXO_KIJ = {
    ("nitrogen", "methane"): 0.031, ("nitrogen", "ethane"): 0.052, ("nitrogen", "propane"): 0.085,
    ("nitrogen", "n-butane"): 0.080, ("nitrogen", "carbon dioxide"): -0.017,
    ("carbon dioxide", "methane"): 0.092, ("carbon dioxide", "ethane"): 0.132,
    ("carbon dioxide", "propane"): 0.124, ("carbon dioxide", "n-butane"): 0.133,
    ("carbon dioxide", "n-pentane"): 0.122, ("methane", "n-pentane"): 0.019,
}


def kij_matrix(pairs):
    n = len(IDS)
    k = np.zeros((n, n))
    for (a, b), v in pairs.items():
        i, j = IDS.index(a), IDS.index(b)
        k[i, j] = k[j, i] = v
    return k.tolist()


constants, props = ChemicalConstantsPackage.from_IDs(IDS)


def flash_with(kijs):
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
    fl = FlashVL(constants, props, liquid=liq, gas=gas)
    return fl, fl.flash(T=T, P=P, zs=Z)


rows = []
for label, pairs in (("k_ij = 0", {}), ("Fluxo (Knapp subset)", FLUXO_KIJ),
                     ("solo CO2", {k: v for k, v in FLUXO_KIJ.items() if "carbon dioxide" in k})):
    fl, res = flash_with(kij_matrix(pairs))
    x = res.liquid0.zs
    rows.append((label, res.VF, x[IDS.index("carbon dioxide")], x[0], x[2]))

print(f"{'caso':24s} {'VF':>8s} {'x_CO2':>8s} {'x_CH4':>8s} {'x_C3':>8s}")
for label, vf, xco2, xch4, xc3 in rows:
    print(f"{label:24s} {vf:8.4f} {xco2:8.4f} {xch4:8.4f} {xc3:8.4f}")
print(f"{'Fluxo (motor Swift)':24s} {0.8751:8.4f} {0.0209:8.4f} {0.2780:8.4f} {0.2125:8.4f}")
print(f"{'DWSIM 10.2.12':24s} {0.8762:8.4f} {0.0264:8.4f} {0.2747:8.4f} {0.2092:8.4f}")

# Michelsen stability on the feed (tangent plane distance): unstable => a second phase exists.
fl, res = flash_with(kij_matrix(FLUXO_KIJ))
print("\nMichelsen stability test (feed, Fluxo k_ij):")
gas_feed = fl.gas.to(T=T, P=P, zs=Z)
liq_feed = fl.liquids[0].to(T=T, P=P, zs=Z)
for label, min_phase, other in (("feed as vapor", gas_feed, liq_feed), ("feed as liquid", liq_feed, gas_feed)):
    stable, info = fl.stability_test_Michelsen(T, P, Z, min_phase, other)
    appearing = next((v for v in info if isinstance(v, (list, tuple)) and len(v) == len(Z)), None)
    print(f"  {label:15s}: stable? {bool(stable)!s:5s}  appearing phase zs="
          f"{np.round(appearing, 4).tolist() if appearing is not None else info}")

# Bubble / dew temperatures at 30 bar for context
try:
    tb = fl.flash(P=P, VF=0.0, zs=Z).T
    td = fl.flash(P=P, VF=1.0, zs=Z).T
    print(f"\nT burbuja @30 bar = {tb:.2f} K, T rocío = {td:.2f} K  (flash a {T} K está dentro)")
except Exception as ex:  # noqa: BLE001
    print("bubble/dew failed:", ex)
