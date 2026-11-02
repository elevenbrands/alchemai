"""Week 2 study notebook — NRTL for ethanol-water and the azeotropic ceiling of the bench column.

The reference column (bench card `columna_rigurosa`, DWSIM NRTL, 20 stages, R = 2) gives a distillate of
x_EtOH = 0.8076. Why can't it go to 0.99? Because ethanol-water has a minimum-boiling azeotrope at
x ≈ 0.894 (351.3 K at 1 atm). Here the Txy diagram is reproduced with `thermo` + NRTL (ChemSep
parameters) and the azeotrope is located, so the ceiling is a number we computed, not a fact we recalled.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from thermo import (CEOSGas, ChemicalConstantsPackage, FlashVL, GibbsExcessLiquid, IGMIX, NRTL)  # noqa: E402
from thermo.interaction_parameters import IPDB  # noqa: E402

IDS = ["ethanol", "water"]
constants, props = ChemicalConstantsPackage.from_IDs(IDS)
P = 101325.0

alpha = IPDB.get_ip_asymmetric_matrix("ChemSep NRTL", constants.CASs, "alphaij")
bij = IPDB.get_ip_asymmetric_matrix("ChemSep NRTL", constants.CASs, "bij")     # tau_ij = b_ij / T  (K)
GE = NRTL(T=350.0, xs=[0.5, 0.5], tau_bs=bij, alpha_cs=alpha)
liquid = GibbsExcessLiquid(VaporPressures=props.VaporPressures, HeatCapacityGases=props.HeatCapacityGases,
                           VolumeLiquids=props.VolumeLiquids, GibbsExcessModel=GE,
                           equilibrium_basis="Psat", caloric_basis="Psat", T=350.0, P=P, zs=[0.5, 0.5])
gas = CEOSGas(IGMIX, eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
              HeatCapacityGases=props.HeatCapacityGases, T=350.0, P=P, zs=[0.5, 0.5])
flasher = FlashVL(constants, props, liquid=liquid, gas=gas)

print(f"{'x_EtOH':>7s} {'T_bub K':>9s} {'y_EtOH':>8s} {'y-x':>8s}")
rows = []
for x in np.linspace(0.02, 0.98, 25):
    res = flasher.flash(P=P, VF=0.0, zs=[x, 1 - x])
    y = res.gas.zs[0]
    rows.append((x, res.T, y))
    print(f"{x:7.3f} {res.T:9.2f} {y:8.4f} {y - x:+8.4f}")

# Azeotrope: y − x changes sign. Fine scan + linear interpolation (thermo's PVF flash can fail
# right at the azeotrope, so we never evaluate exactly there).
fine = []
for x in np.linspace(0.80, 0.96, 81):
    try:
        res = flasher.flash(P=P, VF=0.0, zs=[x, 1 - x])
        fine.append((x, res.T, res.gas.zs[0] - x))
    except Exception:  # noqa: BLE001
        continue
k = next(i for i in range(len(fine) - 1) if fine[i][2] > 0 and fine[i + 1][2] <= 0)
(x0, T0, d0), (x1, T1, d1) = fine[k], fine[k + 1]
x_az = x0 + (x1 - x0) * d0 / (d0 - d1)
T_az = T0 + (T1 - T0) * (x_az - x0) / (x1 - x0)
print(f"\nAzeótropo NRTL (ChemSep): x_EtOH = {x_az:.4f}, T = {T_az:.2f} K   (literatura: 0.894, 351.3 K)")
print("Destilado del bench (DWSIM NRTL, 20 etapas, R = 2): x_EtOH = 0.8076 → por debajo del azeótropo, como debe ser.")
print("Conclusión: con destilación ordinaria a 1 atm no se pasa de ~0.894; para etanol anhidro hace falta")
print("destilación extractiva, a vacío/presión (el azeótropo se mueve) o tamices moleculares.")
