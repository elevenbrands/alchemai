"""Smoke test of the chemical-engineering stack. Reproduces the two rows in BENCHMARK.md."""
import warnings
warnings.filterwarnings("ignore")

from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashVL

constants, props = ChemicalConstantsPackage.from_IDs(["methane", "ethane", "propane"])
eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
flasher = FlashVL(constants, props, liquid=liq, gas=gas)
res = flasher.flash(T=230.0, P=30e5, zs=[0.7, 0.2, 0.1])
print(f"thermo PR flash C1/C2/C3 230 K 30 bar: VF={res.VF:.4f} phase={res.phase}")

import cantera as ct

g = ct.Solution("gri30.yaml")
g.TPX = 1000, ct.one_atm, "CH4:1, O2:2, N2:7.52"
r = ct.IdealGasConstPressureReactor(g)
net = ct.ReactorNet([r])
net.advance(1.0)
print(f"cantera GRI30 batch 1 s: T={r.T:.1f} K")

import idaes, pyomo.environ  # noqa: F401
from Flowsheet_Class.nx_to_sfiles import nx_to_SFILES  # noqa: F401
import pydexpi  # noqa: F401

print(f"idaes {idaes.__version__}, SFILES2 and pydexpi import OK")
